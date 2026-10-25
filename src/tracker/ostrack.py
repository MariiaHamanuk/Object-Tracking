"""Wrapper over the vendored OSTrack that tracks one video and also returns per-frame confidence."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OSTRACK_DIR = REPO_ROOT / "baselines" / "OSTrack"
DEFAULT_CONFIG = "vitb_256_mae_ce_32x4_ep300"


def setup_ostrack(save_dir: Path):
    """Put OSTrack on sys.path and write the local.py it requires."""
    if str(OSTRACK_DIR) not in sys.path:
        sys.path.insert(0, str(OSTRACK_DIR))
    local = OSTRACK_DIR / "lib" / "test" / "evaluation" / "local.py"
    if not local.exists():
        local.write_text(
            "from lib.test.evaluation.environment import EnvSettings\n\n"
            "def local_env_settings():\n"
            "    settings = EnvSettings()\n"
            f"    settings.prj_dir = r'{OSTRACK_DIR}'\n"
            f"    settings.save_dir = r'{Path(save_dir).resolve()}'\n"
            "    return settings\n")


class OSTrackRunner:
    def __init__(self, checkpoint: Path, config: str = DEFAULT_CONFIG, save_dir: Path = REPO_ROOT / "output"):
        setup_ostrack(save_dir)
        from lib.test.parameter.ostrack import parameters
        from lib.test.tracker.ostrack import OSTrack

        params = parameters(config)
        params.checkpoint = str(checkpoint)
        params.debug = 0
        self.tracker = OSTrack(params, "lasot")
        self.vjepa = None
        self._record_scores()

    def _record_scores(self):
        """Record the peak of the score map of every forward pass."""
        net, forward = self.tracker.network, self.tracker.network.forward
        self.last_score = np.nan

        def forward_with_score(*args, **kwargs):
            out = forward(*args, **kwargs)
            self.last_score = float(out["score_map"].max())
            return out

        net.forward = forward_with_score

    def enable_fusion(self, fusion_weights: Path, head_weights: Path | None = None):
        """Use a trained FusionV1 (and optionally a fine-tuned head); call set_vjepa for every video."""
        import torch
        from src.tracker.fusion import FusedOSTrack, FusionV1

        fusion = FusionV1()
        fusion.load_state_dict(torch.load(fusion_weights, map_location="cpu"))
        if head_weights is not None:
            self.tracker.network.box_head.load_state_dict(torch.load(head_weights, map_location="cpu"))
        fused = FusedOSTrack(self.tracker.network, fusion).to(next(self.tracker.network.parameters()).device).eval()
        runner = self

        class Network(torch.nn.Module):  # OSTrack's tracker only uses forward(...) and .box_head
            def __init__(self):
                super().__init__()
                self.fused, self.box_head = fused, fused.net.box_head

            def forward(self, template, search, ce_template_mask=None, **kwargs):
                return self.fused(template, search, runner._vjepa_tokens(search.device), ce_template_mask)

        self.tracker.network = Network()
        self._record_scores()

    def enable_prior(self, prior_weights: Path):
        """Use the prior head; call set_vjepa(cache, w, h, memory) with the relocalizer as memory."""
        import torch
        from src.tracker.fusion import PriorOSTrack

        ckpt = torch.load(prior_weights, map_location="cpu")
        model = PriorOSTrack(self.tracker.network, ckpt["mode"])
        model.prior_head.load_state_dict(ckpt["prior_head"], strict=False)
        device = next(self.tracker.network.parameters()).device
        model = model.to(device).eval()
        runner = self

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model, self.box_head = model, model.box_head

            def forward(self, template, search, ce_template_mask=None, **kwargs):
                return self.model(template, search, runner._prior(search.device), ce_template_mask)

        self.tracker.network = Network()
        self._record_scores()

    def set_vjepa(self, cache, frame_w: int, frame_h: int, memory=None):
        self.vjepa = (cache, frame_w, frame_h)
        self.memory = memory

    def _prior(self, device):
        import torch
        from src.tracker.fusion import prior_tokens, search_crop

        cache, w, h = self.vjepa
        fmap = torch.from_numpy(cache.at(self._frame).astype(np.float32))[None].to(device)
        crop = torch.tensor([search_crop(self.tracker.state, self.tracker.params.search_factor)],
                            dtype=torch.float32, device=device)
        return prior_tokens(fmap, self.memory.m_first[None].to(device), self.memory.m_recent[None].to(device),
                            crop, torch.tensor([[w, h]], dtype=torch.float32, device=device))

    def _vjepa_tokens(self, device):
        import torch
        from src.tracker.fusion import align_to_crop, search_crop

        cache, w, h = self.vjepa
        fmap = torch.from_numpy(cache.at(self._frame).astype(np.float32))[None].to(device)
        crop = torch.tensor([search_crop(self.tracker.state, self.tracker.params.search_factor)],
                            dtype=torch.float32, device=device)
        return align_to_crop(fmap, crop, torch.tensor([[w, h]], dtype=torch.float32, device=device))

    def track(self, frames: list[Path], init_box, hook=None, verify: bool = False, verify_min: float = 0.0) -> dict:
        """Track `frames` from `init_box`.

        hook(i, box, score[, image]) may return candidate boxes. Without `verify` the first one becomes
        the new state; with `verify` OSTrack re-runs around each candidate on the same frame and keeps
        the best only if its score beats the current one and reaches `verify_min`."""
        n = len(frames)
        boxes, scores, times = np.zeros((n, 4)), np.full(n, np.nan), np.zeros(n)
        moved, checks = [], []
        for i, f in enumerate(frames):
            image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            self._frame = i
            t = time.perf_counter()
            if i == 0:
                self.tracker.initialize(image, {"init_bbox": list(map(float, init_box))})
                boxes[0] = init_box
            else:
                boxes[i] = self.tracker.track(image)["target_bbox"]
                scores[i] = self.last_score
            if hook is None:
                cands = None
            elif getattr(hook, "needs_image", False):
                cands = hook(i, boxes[i], scores[i], image)
            else:
                cands = hook(i, boxes[i], scores[i])
            if cands is not None:
                cands = [cands] if np.ndim(cands) == 1 else list(cands)
                if not verify:
                    self.tracker.state = [float(v) for v in cands[0]]
                    moved.append(i)
                else:
                    best = self._verify(image, cands, boxes[i], scores[i], verify_min)
                    checks.append(dict(frame=i, score=float(scores[i]), n_candidates=len(cands),
                                       best_score=best[1], accepted=best[0] is not None, best_index=best[2]))
                    if best[0] is not None:
                        boxes[i], scores[i] = best[0], best[1]
                        moved.append(i)
            times[i] = time.perf_counter() - t
        return dict(boxes=boxes, scores=scores, times=times, moved=np.array(moved, dtype=int), checks=checks)

    def _verify(self, image, candidates, box, score, min_score: float = 0.0):
        """Returns (best box or None, its score, its index or -1); leaves the tracker state on the winner."""
        best_box, best_score, best_idx = None, float(score), -1
        for k, c in enumerate(candidates):
            self.tracker.state = [float(v) for v in c]
            out = self.tracker.track(image)["target_bbox"]
            if self.last_score > max(best_score, min_score):
                best_box, best_score, best_idx = np.asarray(out, dtype=float), float(self.last_score), k
        self.tracker.state = [float(v) for v in (best_box if best_box is not None else box)]
        return best_box, best_score, best_idx
