"""Feature-level fusion of V-JEPA 2 maps into OSTrack (fusion_v1, see docs/vjepa_integration.md).

OSTrack's backbone and box head stay frozen. Between them, every one of the 16 x 16 search tokens gets
the V-JEPA token from the same place of the frame added through a small residual MLP:

    x' = x + MLP([x ; P(v)])        (last MLP layer zero-initialised -> x' = x before training)

V-JEPA maps cover the whole (squashed) frame, so the part under OSTrack's search crop is resampled
to the 16 x 16 search grid with `align_to_crop`. Note: OSTrack's candidate elimination zeroes ~66% of
the search tokens it considers background; the fusion adds V-JEPA features to those as well.
"""
from __future__ import annotations

import copy
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


HEAD_PARTS = {"none": (), "score": ("conv1_ctr", "conv2_ctr", "conv3_ctr", "conv4_ctr", "conv5_ctr"),
              "all": ("",)}  # which layers of OSTrack's CenterPredictor may learn


class FusedOSTrack(nn.Module):
    """Frozen OSTrack network + trainable fusion between backbone and box head.

    train_head="score" / "all" also unfreezes the score branch / the whole box head (fine-tuned from
    OSTrack's weights; BatchNorm statistics stay frozen). A frozen copy of the original head is kept as
    `teacher`, and in training mode out["teacher_score_map"] is the original OSTrack's score map on
    the unfused features (for a distillation loss that keeps the head from forgetting OSTrack)."""

    def __init__(self, ostrack_net: nn.Module, fusion: nn.Module, train_head: str = "none"):
        super().__init__()
        self.net, self.fusion = ostrack_net, fusion
        self.teacher = copy.deepcopy(self.net.box_head) if train_head != "none" else None
        for p in self.net.parameters():
            p.requires_grad_(False)
        for name, p in self.net.box_head.named_parameters():
            if any(name.startswith(part) for part in HEAD_PARTS[train_head]):
                p.requires_grad_(True)
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.n_search = self.net.feat_len_s  # 256

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()  # frozen part always in eval mode (no dropout / drop-path, frozen BatchNorm)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, template, search, vjepa_tokens, ce_template_mask=None, ce_keep_rate=None):
        with torch.no_grad():
            feat, aux = self.net.backbone(z=template, x=search, ce_template_mask=ce_template_mask,
                                          ce_keep_rate=ce_keep_rate)
        feat = feat[-1] if isinstance(feat, list) else feat
        s = self.fusion(feat[:, -self.n_search:], vjepa_tokens.to(feat.dtype))
        out = self.net.forward_head(torch.cat([feat[:, :-self.n_search], s], dim=1), None)
        out.update(aux)
        if self.training and self.teacher is not None:
            with torch.no_grad():
                x = feat[:, -self.n_search:]
                g = int(math.isqrt(self.n_search))
                out["teacher_score_map"] = self.teacher(x.transpose(1, 2).reshape(len(x), -1, g, g), None)[0]
        return out


N_PRIOR = 4  # channels of prior_tokens


def prior_tokens(fmap: torch.Tensor, m_first: torch.Tensor, m_recent: torch.Tensor, crops: torch.Tensor,
                 frame_wh: torch.Tensor) -> torch.Tensor:
    """V-JEPA target-memory prior for OSTrack's head (fusion_v2a), resampled to the search grid.

    fmap: (B, G, G, C) full-frame V-JEPA maps; m_first / m_recent: (B, C) target memory (pooled in the
    first box / EMA over confident frames, as in src/tracker/relocalize.py); crops, frame_wh: as in
    align_to_crop. Channels per search token:
      0  cos(token, m_first)
      1  cos(token, m_recent)
      2  cos(token, m) - max over the whole frame of cos(token, m), m = both memories: 0 = the place
         that looks most like the target anywhere in the frame, < 0 = something looks more like it
      3  1 inside the frame, 0 in the zero-padded part of the crop
    Returns (B, 256, 4). Only 4 numbers per token instead of 1024 features: little to overfit."""
    f = F.normalize(fmap.float(), dim=-1)
    mf, mr = F.normalize(m_first.float(), dim=-1), F.normalize(m_recent.float(), dim=-1)
    m = F.normalize(mf + mr, dim=-1)
    sim_f = torch.einsum("bijc,bc->bij", f, mf)
    sim_r = torch.einsum("bijc,bc->bij", f, mr)
    sim = torch.einsum("bijc,bc->bij", f, m)
    rel = sim - sim.flatten(1).amax(1)[:, None, None]
    maps = torch.stack([sim_f, sim_r, rel, torch.ones_like(sim)], dim=-1)
    return align_to_crop(maps, crops, frame_wh)


class PriorHead(nn.Module):
    """OSTrack's CenterPredictor (frozen) + the V-JEPA prior in its score branch.

    mode "A":  score logits += conv3x3(prior)                       (37 parameters)
    mode "AB": also conv1_ctr(x) += conv3x3(prior) before its BatchNorm, so the score branch can
               combine the prior with what OSTrack sees               (+9.5k parameters)
    All new layers start at zero, i.e. exactly OSTrack. Box size / offset branches are untouched."""

    def __init__(self, head: nn.Module, mode: str = "AB", k: int = N_PRIOR):
        super().__init__()
        self.head, self.mode = head, mode
        self.to_logit = nn.Conv2d(k, 1, 3, padding=1)
        self.to_ctr1 = nn.Conv2d(k, head.conv1_ctr[0].out_channels, 3, padding=1) if mode == "AB" else None
        for layer in (self.to_logit, self.to_ctr1):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def new_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("head.")]

    def forward(self, x, prior, gt_score_map=None):
        """x: (B, 768, 16, 16) search features; prior: (B, k, 16, 16)."""
        h = self.head
        c1 = h.conv1_ctr[0](x)
        if self.to_ctr1 is not None:
            c1 = c1 + self.to_ctr1(prior)
        c = h.conv5_ctr(h.conv4_ctr(h.conv3_ctr(h.conv2_ctr(h.conv1_ctr[2](h.conv1_ctr[1](c1))))))
        score = torch.clamp((c + self.to_logit(prior)).sigmoid(), min=1e-4, max=1 - 1e-4)
        size = torch.clamp(h.conv5_size(h.conv4_size(h.conv3_size(h.conv2_size(h.conv1_size(x))))).sigmoid(),
                           min=1e-4, max=1 - 1e-4)
        offset = h.conv5_offset(h.conv4_offset(h.conv3_offset(h.conv2_offset(h.conv1_offset(x)))))
        bbox = h.cal_bbox(score if gt_score_map is None else gt_score_map.unsqueeze(1), size, offset)
        return score, bbox, size, offset


class PriorOSTrack(nn.Module):
    """Frozen OSTrack whose head also reads the V-JEPA memory prior (fusion_v2a). Only the new layers of
    PriorHead learn."""

    def __init__(self, ostrack_net: nn.Module, mode: str = "AB"):
        super().__init__()
        self.net = ostrack_net
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.prior_head = PriorHead(self.net.box_head, mode)
        self.n_search = self.net.feat_len_s
        self.box_head = self.net.box_head  # what OSTrack's tracker reads (feat_sz, cal_bbox)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, template, search, prior, ce_template_mask=None, ce_keep_rate=None):
        """prior: (B, 256, k) from prior_tokens."""
        with torch.no_grad():
            feat, aux = self.net.backbone(z=template, x=search, ce_template_mask=ce_template_mask,
                                          ce_keep_rate=ce_keep_rate)
        feat = feat[-1] if isinstance(feat, list) else feat
        x = feat[:, -self.n_search:]
        g = int(math.isqrt(self.n_search))
        x = x.transpose(1, 2).reshape(len(x), -1, g, g)
        p = prior.to(x.dtype).transpose(1, 2).reshape(len(x), -1, g, g)
        score, bbox, size, offset = self.prior_head(x, p)
        out = {"pred_boxes": bbox.view(len(x), 1, 4), "score_map": score, "size_map": size, "offset_map": offset}
        out.update(aux)
        return out


class VJEPACache:
    """Cached maps of one video (scripts/cache_vjepa.py). `at(t)` returns the latest map computed
    from frames <= t, so tracking stays causal."""

    def __init__(self, path):
        d = np.load(path)
        self.frames, self.maps = d["frames"], d["maps"]
        self.surprise = d["surprise"] if "surprise" in d.files else None  # (N, 16, 16), cache_vjepa.py --surprise

    def _index(self, t: int) -> int:
        return max(int(np.searchsorted(self.frames, t, side="right")) - 1, 0)

    def at(self, t: int) -> np.ndarray:
        return self.maps[self._index(t)]

    def surprise_at(self, t: int) -> np.ndarray:
        return self.surprise[self._index(t)]
