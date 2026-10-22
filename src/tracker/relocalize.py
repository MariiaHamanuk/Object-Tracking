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


class VJEPARelocalizer:
    def __init__(self, cache: VJEPACache, frame_w: int, frame_h: int, init_box, params: RelocParams = RelocParams()):
        self.cache, self.w, self.h, self.p = cache, frame_w, frame_h, params
        self.cached = set(int(t) for t in cache.frames)
        self.m_first = F.normalize(self._pool(0, init_box), dim=0)
        self.m_recent = self.m_first.clone()
        self.last_box = np.asarray(init_box, dtype=float)
        self.score_mean = None
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
        r, c = divmod(int(sim.argmax()), g)
        s_best = float(sim[r, c])
        cx, cy = (c + 0.5) * self.w / g, (r + 0.5) * self.h / g

        x1, y1, side = search_crop(box, self.p.search_factor)
        outside = not (x1 <= cx <= x1 + side and y1 <= cy <= y1 + side)
        doubt = self.score_mean is not None and score < self.p.score_ratio * self.score_mean
        move = doubt and outside and s_best - s_box > self.p.margin
        self.log.append(dict(frame=i, score=float(score), s_box=s_box, s_best=s_best, doubt=bool(doubt),
                             outside=outside, moved=bool(move)))
        if not move:
            return None
        w, h = self.last_box[2:]
        return np.array([cx - w / 2, cy - h / 2, w, h])
