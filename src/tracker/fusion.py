"""Feature-level fusion of V-JEPA 2 maps into OSTrack (fusion_v1, see docs/vjepa_integration.md).

OSTrack's backbone and box head stay frozen. Between them, every one of the 16 x 16 search tokens gets
the V-JEPA token from the same place of the frame added through a small residual MLP:

    x' = x + MLP([x ; P(v)])        (last MLP layer zero-initialised -> x' = x before training)

V-JEPA maps cover the whole (squashed) frame, so the part under OSTrack's search crop is resampled
to the 16 x 16 search grid with `align_to_crop`. Note: OSTrack's candidate elimination zeroes ~66% of
the search tokens it considers background; the fusion adds V-JEPA features to those as well.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def search_crop(box, factor: float = 4.0) -> tuple[float, float, float]:
    """OSTrack's square crop around a box (lib/train/data/processing_utils.sample_target): (x1, y1, side)."""
    x, y, w, h = box
    side = math.ceil(math.sqrt(w * h) * factor)
    return round(x + 0.5 * w - side * 0.5), round(y + 0.5 * h - side * 0.5), side


def align_to_crop(fmap: torch.Tensor, crops: torch.Tensor, frame_wh: torch.Tensor, out: int = 16) -> torch.Tensor:
    """Resample full-frame maps to the search crops.

    fmap: (B, G, G, C) V-JEPA maps of the whole frame; crops: (B, 3) = x1, y1, side in pixels;
    frame_wh: (B, 2). Returns (B, out*out, C) in OSTrack's token order (row-major). Parts of the crop
    outside the frame get zeros, like OSTrack's zero-padded crop."""
    b = fmap.shape[0]
    cells = (torch.arange(out, device=fmap.device, dtype=torch.float32) + 0.5) / out          # cell centres, 0..1
    px = crops[:, 0:1] + cells[None] * crops[:, 2:3]                                          # (B, out) pixels
    py = crops[:, 1:2] + cells[None] * crops[:, 2:3]
    gx = 2 * px / frame_wh[:, 0:1] - 1                                                        # -> [-1, 1]
    gy = 2 * py / frame_wh[:, 1:2] - 1
    grid = torch.stack(torch.broadcast_tensors(gx[:, None, :], gy[:, :, None]), dim=-1)        # (B, out, out, 2)
    sampled = F.grid_sample(fmap.permute(0, 3, 1, 2).float(), grid.float(), mode="bilinear",
                            padding_mode="zeros", align_corners=False)                        # (B, C, out, out)
    return sampled.flatten(2).transpose(1, 2).reshape(b, out * out, -1)


class FusionV1(nn.Module):
    def __init__(self, dim: int = 768, vjepa_dim: int = 1024, hidden: int = 768):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(vjepa_dim), nn.Linear(vjepa_dim, dim))
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """x: (B, N, dim) OSTrack search tokens; v: (B, N, vjepa_dim) aligned V-JEPA tokens."""
        return x + self.mlp(torch.cat([x, self.proj(v)], dim=-1))


class FusedOSTrack(nn.Module):
    """Frozen OSTrack network + trainable fusion between backbone and box head."""

    def __init__(self, ostrack_net: nn.Module, fusion: nn.Module):
        super().__init__()
        self.net, self.fusion = ostrack_net, fusion
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.n_search = self.net.feat_len_s  # 256

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()  # frozen part always in eval mode (no dropout / drop-path)
        return self

    def forward(self, template, search, vjepa_tokens, ce_template_mask=None, ce_keep_rate=None):
        with torch.no_grad():
            feat, aux = self.net.backbone(z=template, x=search, ce_template_mask=ce_template_mask,
                                          ce_keep_rate=ce_keep_rate)
        feat = feat[-1] if isinstance(feat, list) else feat
        s = self.fusion(feat[:, -self.n_search:], vjepa_tokens.to(feat.dtype))
        out = self.net.forward_head(torch.cat([feat[:, :-self.n_search], s], dim=1), None)
        out.update(aux)
        return out


class VJEPACache:
    """Cached maps of one video (scripts/cache_vjepa.py). `at(t)` returns the latest map computed
    from frames <= t, so tracking stays causal."""

    def __init__(self, path):
        d = np.load(path)
        self.frames, self.maps = d["frames"], d["maps"]

    def at(self, t: int) -> np.ndarray:
        i = max(int(np.searchsorted(self.frames, t, side="right")) - 1, 0)
        return self.maps[i]
