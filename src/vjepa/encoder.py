"""Frozen V-JEPA 2 video encoder that turns a clip of full frames into a spatial feature map.

The whole frame is squashed to 256x256 (no centre crop, unlike the default HF processor) so that
token (i, j) always covers the same region of the original frame and the map can be cached
independently of the tracker. With tubelet 2 x 16 x 16 a clip of T frames gives T/2 x 16 x 16 tokens;
`encode` keeps the last temporal slice = the scene "now", with the motion of the clip folded in.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VJEPAEncoder:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, dtype=None):
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        cfg = self.model.config
        self.size, self.patch, self.tubelet = cfg.crop_size, cfg.patch_size, cfg.tubelet_size
        self.grid = self.size // self.patch          # 16
        self.dim = cfg.hidden_size                   # 1024 for ViT-L

    def preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        """RGB uint8 frames (H, W, 3) -> (1, T, 3, 256, 256) normalised."""
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        return ((x - MEAN) / STD).unsqueeze(0)

    @torch.no_grad()
    def encode(self, frames: list[np.ndarray], last_only: bool = True) -> torch.Tensor:
        """frames: T RGB frames (T even). Returns (16, 16, C) for the last time step,
        or (T/2, 16, 16, C) with last_only=False. Float32 on CPU."""
        if len(frames) % self.tubelet:
            raise ValueError(f"need a multiple of {self.tubelet} frames, got {len(frames)}")
        x = self.preprocess(frames).to(self.device, self.dtype)
        tokens = self.model.get_vision_features(pixel_values_videos=x)  # (1, T/2 * 16 * 16, C)
        maps = tokens.float().view(len(frames) // self.tubelet, self.grid, self.grid, -1).cpu()
        return maps[-1] if last_only else maps

    @torch.no_grad()
    def encode_batch(self, clips: list[list[np.ndarray]]) -> torch.Tensor:
        """Several equally long clips at once -> (B, 16, 16, C) maps of their last time step (fp16, CPU)."""
        x = torch.cat([self.preprocess(c) for c in clips]).to(self.device, self.dtype)
        tokens = self.model.get_vision_features(pixel_values_videos=x)
        t = len(clips[0]) // self.tubelet
        return tokens.view(len(clips), t, self.grid, self.grid, -1)[:, -1].half().cpu()


def clip_indices(t: int, clip_len: int = 16, stride: int = 2) -> np.ndarray:
    """Frame indices of the causal clip ending at frame t (only past frames; padded at the start)."""
    return np.clip(t - stride * np.arange(clip_len)[::-1], 0, None)


def box_to_grid(box, frame_w: int, frame_h: int, grid: int = 16):
    """(x, y, w, h) in pixels -> (x0, y0, x1, y1) in token units of the squashed full-frame grid."""
    x, y, w, h = box
    sx, sy = grid / frame_w, grid / frame_h
    return x * sx, y * sy, (x + w) * sx, (y + h) * sy


def pool_box(fmap: torch.Tensor, box, frame_w: int, frame_h: int) -> torch.Tensor:
    """Area-weighted mean of the (G, G, C) map inside a pixel box -> (C,). Works for sub-token boxes."""
    g = fmap.shape[0]
    x0, y0, x1, y1 = box_to_grid(box, frame_w, frame_h, g)
    edges = torch.arange(g + 1, dtype=torch.float32)
    wx = (torch.minimum(edges[1:], torch.tensor(x1)) - torch.maximum(edges[:-1], torch.tensor(x0))).clamp(min=0)
    wy = (torch.minimum(edges[1:], torch.tensor(y1)) - torch.maximum(edges[:-1], torch.tensor(y0))).clamp(min=0)
    w = wy[:, None] * wx[None, :]
    if w.sum() == 0:  # box outside the frame: nearest token
        cx, cy = min(max((x0 + x1) / 2, 0), g - 1e-3), min(max((y0 + y1) / 2, 0), g - 1e-3)
        return fmap[int(cy), int(cx)]
    return (fmap * w[..., None]).sum((0, 1)) / w.sum()
