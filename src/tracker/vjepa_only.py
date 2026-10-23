"""Tracking with V-JEPA alone (no OSTrack, nothing trained) - a sanity check of how much the frozen
features know about *where* the target is, not a competitive tracker.

Same target memory as the re-localizer (src/tracker/relocalize.py):
  m = normalize(m_first) + normalize(m_recent), m_recent = EMA over confident frames.
At every cached frame (every 8th):
  sim    = cosine(m, every token), upsampled 16x16 -> 64x64 for sub-token positions
  prior  = Gaussian around the last centre (sigma = `motion` box sizes), so a far look-alike only wins
           if it is clearly more similar; sim * prior picks the peak
  box    = bounding box of the connected region around the peak where sim >= peak - half of
           (peak - median); its size is smoothed with the last size (EMA `size_ema`)
  confident = peak >= `confident_ratio` x running mean of peaks -> update m_recent and the centre prior
Between cached frames the last box is held (causal: the next map is not known yet).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.tracker.fusion import VJEPACache
from src.vjepa.encoder import pool_box


@dataclass
class VJEPAOnlyParams:
    up: int = 64                 # upsampled similarity grid
    motion: float = 1.5          # sigma of the centre prior, in box sizes (relative to the frame grid)
    size_ema: float = 0.3        # weight of the new size estimate
    ema: float = 0.1             # memory update rate
    confident_ratio: float = 0.9


class VJEPAOnlyTracker:
    def __init__(self, cache: VJEPACache, frame_w: int, frame_h: int, init_box,
                 params: VJEPAOnlyParams = VJEPAOnlyParams()):
        self.cache, self.w, self.h, self.p = cache, frame_w, frame_h, params
        self.box = np.asarray(init_box, dtype=float)
        self.m_first = F.normalize(pool_box(self._map(0), self.box, frame_w, frame_h), dim=0)
        self.m_recent = self.m_first.clone()
        self.peak_mean = None

    def _map(self, t: int) -> torch.Tensor:
        return torch.from_numpy(self.cache.at(t).astype(np.float32))

    def step(self, t: int) -> np.ndarray:
        """(box, peak similarity) for the cached frame t."""
        p, u = self.p, self.p.up
        fmap = self._map(t)
        m = F.normalize(self.m_first + self.m_recent, dim=0)
        sim = F.cosine_similarity(fmap, m[None, None], dim=-1)
        sim = F.interpolate(sim[None, None], size=(u, u), mode="bilinear", align_corners=False)[0, 0].numpy()

        # centre prior in upsampled-grid units (the frame is squashed to a square grid)
        x, y, bw, bh = self.box
        cx, cy = (x + bw / 2) / self.w * u, (y + bh / 2) / self.h * u
        size = np.sqrt((bw / self.w) * (bh / self.h)) * u
        yy, xx = np.mgrid[:u, :u] + 0.5
        prior = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (p.motion * max(size, 1.0)) ** 2))
        r, c = np.unravel_index(np.argmax((sim - sim.min()) * prior), sim.shape)
        peak = float(sim[r, c])

        # extent: connected region around the peak above half-way between median and peak
        labels, _ = ndimage.label(sim >= peak - 0.5 * (peak - np.median(sim)))
        rows, cols = np.nonzero(labels == labels[r, c])
        nw = (cols.max() - cols.min() + 1) / u * self.w
        nh = (rows.max() - rows.min() + 1) / u * self.h
        bw, bh = (1 - p.size_ema) * bw + p.size_ema * nw, (1 - p.size_ema) * bh + p.size_ema * nh
        ncx, ncy = (c + 0.5) / u * self.w, (r + 0.5) / u * self.h
        self.box = np.array([ncx - bw / 2, ncy - bh / 2, bw, bh])

        if self.peak_mean is None or peak >= p.confident_ratio * self.peak_mean:
            cur = F.normalize(pool_box(fmap, self.box, self.w, self.h), dim=0)
            self.m_recent = F.normalize((1 - p.ema) * self.m_recent + p.ema * cur, dim=0)
            self.peak_mean = peak if self.peak_mean is None else 0.9 * self.peak_mean + 0.1 * peak
        return self.box, peak

    def track(self, n_frames: int):
        """Boxes (n, 4) and scores (n,) for all frames; uncached frames hold the last box."""
        boxes, scores = np.zeros((n_frames, 4)), np.full(n_frames, np.nan)
        boxes[0] = self.box
        cached = set(int(t) for t in self.cache.frames)
        score = np.nan
        for t in range(1, n_frames):
            if t in cached:
                _, score = self.step(t)
            boxes[t], scores[t] = self.box, score
        return boxes, scores
