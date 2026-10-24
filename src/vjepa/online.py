"""V-JEPA 2 predictor run during tracking (needs the tracker's box, so it cannot be cached).

Keeps the frames the causal clips need (even frames of the last 31, already resized to 256 x 256) and,
at a check frame t, builds the same clip as scripts/cache_vjepa.py (t-30, t-28, ..., t).
"""
from __future__ import annotations

import numpy as np
import torch

from src.vjepa.encoder import clip_indices
from src.vjepa.predictor import VJEPAPredictor


class OnlinePredictor:
    def __init__(self, model: str | None = None, clip_len: int = 16, stride: int = 2):
        self.pred = VJEPAPredictor(model) if model else VJEPAPredictor()
        self.clip_len, self.stride = clip_len, stride
        self.frames: dict[int, torch.Tensor] = {}

    def reset(self):
        self.frames = {}

    def push(self, i: int, image_rgb: np.ndarray):
        """Call for every frame (only every `stride`-th is kept)."""
        if i % self.stride:
            return
        self.frames[i] = self.pred.preprocess([image_rgb])[0, 0].half()     # (3, 256, 256)
        for k in [k for k in self.frames if k < i - self.stride * self.clip_len]:
            del self.frames[k]

    def clip(self, t: int) -> torch.Tensor:
        idx = clip_indices(t, self.clip_len, self.stride)
        return torch.stack([self.frames[int(k)] for k in idx])[None].float()  # (1, T, 3, 256, 256)

    def target_surprise(self, t: int, box, frame_w: int, frame_h: int, real_last) -> float:
        return self.pred.target_surprise(self.clip(t), box, frame_w, frame_h, torch.as_tensor(real_last))

    def future_map(self, t: int, ahead: int = 2) -> torch.Tensor:
        return self.pred.future_map(self.clip(t), ahead)
