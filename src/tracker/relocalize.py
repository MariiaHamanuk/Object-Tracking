"""Training-free V-JEPA re-localization for OSTrack (first step of fusion_v2, docs/vjepa_integration.md).

OSTrack only searches a window of ~2 box sizes around its last box; after a long occlusion the target
often comes back outside it (EDA: median 2.5 box sizes). Here a memory of the target's V-JEPA
features is matched against the whole frame and, when OSTrack looks lost, its search window is moved
to the best match. Nothing is trained.

At every frame that has a cached V-JEPA map (every 8 frames):
  memory   m = normalize(m_first) + normalize(m_recent), where m_first is pooled in the GT box of
             frame 0 and m_recent is an EMA over frames where OSTrack is confident
  sim      = cosine(m, every token of the map)
  move to the best token if both
    - OSTrack doubts:   its score < score_ratio x running mean of its confident scores
    - V-JEPA disagrees: best similarity - similarity at OSTrack's box > margin, and the best token
                        lies outside OSTrack's current search window
The new search centre keeps the size of the last confident box.
With `candidates` > 1 the `candidates` most similar places outside the search window (at least
`min_sep` tokens apart, i.e. different objects) are returned, for OSTrack to verify on the same frame
(OSTrackRunner.track(verify=True)): the search region only moves if OSTrack itself is more confident
there, which rejects look-alike distractors and moves made while OSTrack is still on the target.
With `surprise_gate` (needs a cache from cache_vjepa.py --surprise) "doubt" also needs the V-JEPA
predictor to have been surprised recently at the last confident box: surprise there (1 - cos of the
predicted vs the real "now", src/vjepa/predictor.py) above `surprise_ratio` x its running mean on
confident frames, within the last `surprise_window` frames. A drop of OSTrack's score from blur or a
scale change does not surprise the world model; a target vanishing behind something does.
Thresholds are fixed a priori, not tuned on the evaluation videos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.tracker.fusion import VJEPACache, search_crop
from src.vjepa.encoder import pool_box


@dataclass
class RelocParams:
    confident_score: float = 0.6   # frames above this update the memory and the score baseline
    ema: float = 0.1               # memory update rate
    score_ratio: float = 0.75      # "doubt" = score below this fraction of the running confident mean
    margin: float = 0.10           # cosine gap between best location and OSTrack's box
    search_factor: float = 4.0     # OSTrack's search window (to skip moves it can already see)
    candidates: int = 1            # >1: propose several places for OSTrack to verify
    min_sep: int = 2               # tokens between candidates (Chebyshev distance)
    surprise_gate: bool = False    # doubt also needs a recent surprise of the V-JEPA predictor
    surprise_ratio: float = 1.5    # "surprised" = box surprise above this x running mean on confident frames
    surprise_window: int = 64      # frames a surprise keeps the gate open


class VJEPARelocalizer:
    def __init__(self, cache: VJEPACache, frame_w: int, frame_h: int, init_box, params: RelocParams = RelocParams()):
        self.cache, self.w, self.h, self.p = cache, frame_w, frame_h, params
        self.cached = set(int(t) for t in cache.frames)
        self.m_first = F.normalize(self._pool(0, init_box), dim=0)
        self.m_recent = self.m_first.clone()
        self.last_box = np.asarray(init_box, dtype=float)
        self.score_mean = None
        if params.surprise_gate and cache.surprise is None:
            raise ValueError("surprise_gate needs a V-JEPA cache made with cache_vjepa.py --surprise")
        self.surprise_mean, self.last_surprise = None, -10**9
        self.log = []  # one row per checked frame, for analysis

    def _map(self, t: int) -> torch.Tensor:
        return torch.from_numpy(self.cache.at(t).astype(np.float32))

    def _pool(self, t, box) -> torch.Tensor:
        return pool_box(self._map(t), box, self.w, self.h)

    def __call__(self, i: int, box, score) -> np.ndarray | None:
        confident = i == 0 or (np.isfinite(score) and score >= self.p.confident_score)
        if confident:
            self.last_box = np.asarray(box, dtype=float)
            if i > 0:
                self.score_mean = score if self.score_mean is None else 0.9 * self.score_mean + 0.1 * score
        if i == 0 or i not in self.cached:
            return None

        fmap = self._map(i)
        if confident:  # refresh the recent memory while OSTrack is on target
            cur = F.normalize(pool_box(fmap, box, self.w, self.h), dim=0)
            self.m_recent = F.normalize((1 - self.p.ema) * self.m_recent + self.p.ema * cur, dim=0)
        m = F.normalize(self.m_first + self.m_recent, dim=0)
        sim = F.cosine_similarity(fmap, m[None, None], dim=-1)                      # (G, G)
        s_box = float(F.cosine_similarity(pool_box(fmap, box, self.w, self.h), m, dim=0))
        g = sim.shape[0]
        x1, y1, side = search_crop(box, self.p.search_factor)

        def centre(k):
            r, c = divmod(int(k), g)
            return (c + 0.5) * self.w / g, (r + 0.5) * self.h / g

        def is_outside(cx, cy):
            return not (x1 <= cx <= x1 + side and y1 <= cy <= y1 + side)

        order = torch.argsort(sim.flatten(), descending=True)
        s_best = float(sim.flatten()[order[0]])
        outside = is_outside(*centre(order[0]))
        doubt = self.score_mean is not None and score < self.p.score_ratio * self.score_mean
        extra = {}
        if self.cache.surprise is not None:
            surp = torch.from_numpy(self.cache.surprise_at(i).astype(np.float32))[..., None]
            s_surp = float(pool_box(surp, self.last_box, self.w, self.h))
            surprised = self.surprise_mean is not None and s_surp > self.p.surprise_ratio * self.surprise_mean
            if surprised:
                self.last_surprise = i
            if confident and not surprised:
                self.surprise_mean = s_surp if self.surprise_mean is None else 0.9 * self.surprise_mean + 0.1 * s_surp
            extra = dict(surprise=s_surp, surprised=bool(surprised))
            if self.p.surprise_gate:
                doubt = doubt and i - self.last_surprise <= self.p.surprise_window
        move = doubt and outside and s_best - s_box > self.p.margin
        self.log.append(dict(frame=i, score=float(score), s_box=s_box, s_best=s_best, doubt=bool(doubt),
                             outside=outside, moved=bool(move), **extra))
        if not move:
            return None

        picked = []  # most similar places outside the window, min_sep tokens apart (different objects)
        for k in order.tolist():
            r, c = divmod(k, g)
            if not is_outside(*centre(k)) or any(max(abs(r - pr), abs(c - pc)) < self.p.min_sep for pr, pc in picked):
                continue
            picked.append((r, c))
            if len(picked) == self.p.candidates:
                break
        w, h = self.last_box[2:]
        boxes = [np.array([cx - w / 2, cy - h / 2, w, h]) for cx, cy in (centre(r * g + c) for r, c in picked)]
        return boxes[0] if self.p.candidates == 1 else np.stack(boxes)
