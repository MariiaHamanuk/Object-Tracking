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
Occlusion warning (`warning`, needs a cache from cache_vjepa.py --surprise for "both"): the predictor's
surprise at the last confident box is high when the target starts to go behind something (dev, frames
just before a full occlusion vs calm frames: AUROC 0.68, and nearly independent of OSTrack's score; both
together 0.72), but not during a full occlusion (0.56) - so it is used as an early warning, not as a gate
for moves. Signal = (z of the surprise + z of the score drop) / sqrt(2), z against running statistics of
confident frames (causal). When it exceeds `warn_z` the tracker is "warned" until OSTrack is confident
again (at least `warn_min` frames): the target memory, last box and score statistics are frozen (no
occluder / distractor gets into the memory) and moves are easier (`warn_score_ratio`, `warn_margin`;
OSTrack still verifies them). warning="score" is the same without the predictor (control).

With a `detector` (src/tracker/detector.YOLODetector) the trigger is the same, but the candidates
also include up to `yolo_k` detections outside the search window, ranked by V-JEPA similarity of
their box to the memory: whole objects with real boxes. `log` records how many came from each source.
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
    propose: bool = True           # False: only keep the target memory up to date (for the prior head)
    yolo_k: int = 3                # with a detector: detections added to the candidates
    warning: str = "off"           # occlusion warning: "both" = predictor surprise + score drop, "score" = score only
    warn_z: float = 2.0            # warning when the (combined) z-score exceeds this
    warn_min: int = 16             # frames the warning lasts at least
    warn_score_ratio: float = 0.9  # while warned: easier "doubt"
    warn_margin: float = 0.05      # while warned: smaller V-JEPA margin


class VJEPARelocalizer:
    def __init__(self, cache: VJEPACache, frame_w: int, frame_h: int, init_box, params: RelocParams = RelocParams(),
                 detector=None):
        self.cache, self.w, self.h, self.p = cache, frame_w, frame_h, params
        self.detector = detector
        self.needs_image = detector is not None  # OSTrackRunner.track then passes the frame
        self.cached = set(int(t) for t in cache.frames)
        self.m_first = F.normalize(self._pool(0, init_box), dim=0)
        self.m_recent = self.m_first.clone()
        self.last_box = np.asarray(init_box, dtype=float)
        self.score_mean = None
        if (params.surprise_gate or params.warning == "both") and cache.surprise is None:
            raise ValueError("surprise_gate / warning='both' need a V-JEPA cache made with cache_vjepa.py --surprise")
        self.surprise_mean, self.last_surprise = None, -10**9
        self.score_var, self.surprise_var, self.n_stats = 0.0, 0.0, 0
        self.warned, self.warned_since = False, -10**9
        self.log = []  # one row per checked frame, for analysis

    def _map(self, t: int) -> torch.Tensor:
        return torch.from_numpy(self.cache.at(t).astype(np.float32))

    def _pool(self, t, box) -> torch.Tensor:
        return pool_box(self._map(t), box, self.w, self.h)

    def __call__(self, i: int, box, score, image=None) -> np.ndarray | None:
        confident = i == 0 or (np.isfinite(score) and score >= self.p.confident_score)
        learn = confident and not self.warned  # while warned nothing is learned (memory, box, statistics)
        if learn:
            self.last_box = np.asarray(box, dtype=float)
            if i > 0:
                if self.score_mean is None:
                    self.score_mean = score
                else:
                    self.score_var = 0.9 * self.score_var + 0.1 * (score - self.score_mean) ** 2
                    self.score_mean = 0.9 * self.score_mean + 0.1 * score
        if i == 0 or i not in self.cached:
            return None

        fmap = self._map(i)
        if learn:  # refresh the recent memory while OSTrack is on target
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
        extra, z_surp = {}, 0.0
        if self.cache.surprise is not None:
            surp = torch.from_numpy(self.cache.surprise_at(i).astype(np.float32))[..., None]
            s_surp = float(pool_box(surp, self.last_box, self.w, self.h))
            surprised = self.surprise_mean is not None and s_surp > self.p.surprise_ratio * self.surprise_mean
            if surprised:
                self.last_surprise = i
            if self.surprise_mean is not None:
                z_surp = (s_surp - self.surprise_mean) / np.sqrt(self.surprise_var + 1e-6)
            if learn and not surprised:
                if self.surprise_mean is None:
                    self.surprise_mean = s_surp
                else:
                    self.surprise_var = 0.9 * self.surprise_var + 0.1 * (s_surp - self.surprise_mean) ** 2
                    self.surprise_mean = 0.9 * self.surprise_mean + 0.1 * s_surp
            extra = dict(surprise=s_surp, surprised=bool(surprised), z_surprise=z_surp)
        if learn:
            self.n_stats += 1

        if self.p.warning != "off" and self.score_mean is not None and self.n_stats >= 10:  # statistics warmed up
            z_score = (self.score_mean - score) / np.sqrt(self.score_var + 1e-6)
            signal = z_score if self.p.warning == "score" else (z_surp + z_score) / np.sqrt(2)
            if signal > self.p.warn_z:
                self.warned, self.warned_since = True, i
            elif (self.warned and i - self.warned_since >= self.p.warn_min
                  and confident and score >= self.p.warn_score_ratio * self.score_mean):
                self.warned = False
            extra.update(z_score=z_score, warn_signal=signal, warned=self.warned)

        ratio, margin = ((self.p.warn_score_ratio, self.p.warn_margin) if self.warned
                         else (self.p.score_ratio, self.p.margin))
        doubt = self.score_mean is not None and score < ratio * self.score_mean
        if self.p.surprise_gate:
            doubt = doubt and i - self.last_surprise <= self.p.surprise_window
        move = self.p.propose and doubt and outside and s_best - s_box > margin
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
        n_yolo = 0
        if self.detector is not None and image is not None:
            dets = [d for d in self.detector(image) if d[2] > 1 and d[3] > 1 and is_outside(d[0] + d[2] / 2, d[1] + d[3] / 2)]
            sims = [float(F.cosine_similarity(pool_box(fmap, d, self.w, self.h), m, dim=0)) for d in dets]
            for k in np.argsort(sims)[::-1][: self.p.yolo_k]:
                boxes.append(np.asarray(dets[k], dtype=float))
                n_yolo += 1
        self.log[-1].update(n_vjepa=len(picked), n_yolo=n_yolo)
        return boxes[0] if self.p.candidates == 1 and n_yolo == 0 else np.stack(boxes)
