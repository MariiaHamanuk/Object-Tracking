"""V-JEPA 2 predictor: predict hidden tokens of a clip from the visible ones.

Token index = slot * 256 + row * 16 + col; a slot is 2 frames, a 16-frame clip has 8 slots.
"""
from __future__ import annotations

import numpy as np
import torch

from src.vjepa.encoder import VJEPAEncoder, box_to_grid


class VJEPAPredictor(VJEPAEncoder):
    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_vision_features(pixel_values_videos=x.to(self.device, self.dtype))[0].float().cpu()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, ctx: np.ndarray, tgt: np.ndarray) -> torch.Tensor:
        """Predict `tgt` tokens of clip(s) x from the `ctx` tokens -> (len(tgt), C) or (B, len(tgt), C)."""
        # HF's forward encodes the whole clip and drops targets afterwards, which leaks them through
        # attention; as in V-JEPA training only the context tokens go through the encoder here
        enc, dev = self.model.encoder, self.device
        h = enc.embeddings(x.to(dev, self.dtype))
        b = h.shape[0]
        ci = torch.as_tensor(np.sort(ctx), device=dev)[None].expand(b, -1)
        ti = torch.as_tensor(np.sort(tgt), device=dev)[None].expand(b, -1)
        hc = torch.gather(h, 1, ci[..., None].expand(-1, -1, h.shape[-1]))
        for layer in enc.layer:
            hc = layer(hc, ci)[0]
        hc = enc.layernorm(hc)
        full = h.new_zeros(h.shape).scatter(1, ci[..., None].expand_as(hc), hc)  # predictor gathers ctx itself
        out = self.model.predictor(encoder_hidden_states=full, context_mask=[ci], target_mask=[ti])
        out = out.last_hidden_state.float().cpu()
        return out[0] if b == 1 else out

    @torch.no_grad()
    def encode_with_surprise(self, clips: list[list[np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Map of the last slot and surprise = 1 - cos(predicted, real) when it is predicted from the earlier slots."""
        x = torch.cat([self.preprocess(c) for c in clips])
        g, slots = self.grid, len(clips[0]) // self.tubelet
        real = self.model.get_vision_features(pixel_values_videos=x.to(self.device, self.dtype))
        real = real.float().view(len(clips), slots, g * g, -1)[:, -1].cpu()
        last = self.slot_tokens(slots - 1)
        pred = self.predict(x, np.arange(last[0]), last)
        pred = pred if pred.dim() == 3 else pred[None]
        surprise = 1 - torch.nn.functional.cosine_similarity(pred, real, dim=-1)
        return real.view(len(clips), g, g, -1).half(), surprise.view(len(clips), g, g).half()

    @torch.no_grad()
    def target_surprise(self, x: torch.Tensor, box, frame_w: int, frame_h: int, real_last: torch.Tensor) -> float:
        """Surprise when only the box in the last slot is hidden and predicted from everything else."""
        g, slots = self.grid, x.shape[1] // self.tubelet
        hide = self.box_tokens(box, frame_w, frame_h, slots - 1)
        pred = self.predict(x, np.setdiff1d(np.arange(slots * g * g), hide), hide).mean(0)
        real = real_last.reshape(g * g, -1)[hide - (slots - 1) * g * g].float().mean(0)
        return float(1 - torch.nn.functional.cosine_similarity(pred, real, dim=0))

    @torch.no_grad()
    def future_map(self, x: torch.Tensor, ahead: int = 2) -> torch.Tensor:
        """Predicted (G, G, C) map `ahead` slots after the end of the clip."""
        g, slots = self.grid, x.shape[1] // self.tubelet
        future = self.slot_tokens(slots - 1 + ahead)
        return self.predict(x, np.arange(slots * g * g), future).view(g, g, -1)

    def slot_tokens(self, slot: int) -> np.ndarray:
        g = self.grid
        return slot * g * g + np.arange(g * g)

    def box_tokens(self, box, frame_w: int, frame_h: int, slot: int) -> np.ndarray:
        g = self.grid
        x0, y0, x1, y1 = box_to_grid(box, frame_w, frame_h, g)
        c0, r0 = int(np.clip(np.floor(x0), 0, g - 1)), int(np.clip(np.floor(y0), 0, g - 1))
        c1, r1 = int(np.clip(np.ceil(x1), c0 + 1, g)), int(np.clip(np.ceil(y1), r0 + 1, g))
        rr, cc = np.mgrid[r0:r1, c0:c1]
        return (slot * g * g + rr * g + cc).ravel()
