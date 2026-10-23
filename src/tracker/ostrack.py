"""Thin wrapper over the vendored OSTrack (baselines/OSTrack) that tracks one video at a time.

Unlike OSTrack's own tracking/test.py it also returns a per-frame confidence (max of the raw score
map), which the occlusion analysis and the V-JEPA fusion need.
"""
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
    """Put OSTrack on sys.path and write the local.py it insists on (paths are unused here)."""
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
        self.vjepa = None  # (cache, frame_w, frame_h) while a fused video is tracked
        self._record_scores()

    def _record_scores(self):
        """Record the confidence of every forward pass without touching OSTrack's code."""
        net, forward = self.tracker.network, self.tracker.network.forward
        self.last_score = np.nan

        def forward_with_score(*args, **kwargs):
            out = forward(*args, **kwargs)
            self.last_score = float(out["score_map"].max())
            return out

        net.forward = forward_with_score

    def enable_fusion(self, fusion_weights: Path, head_weights: Path | None = None):
        """Insert a trained FusionV1 between OSTrack's backbone and box head (src/tracker/fusion.py),
        optionally with the fine-tuned box head (head.pth from train_fusion.py --train-head).
        Call set_vjepa(cache, w, h) for every video before track()."""
        import torch
        from src.tracker.fusion import FusedOSTrack, FusionV1

        fusion = FusionV1()
        fusion.load_state_dict(torch.load(fusion_weights, map_location="cpu"))
        if head_weights is not None:
            self.tracker.network.box_head.load_state_dict(torch.load(head_weights, map_location="cpu"))
        fused = FusedOSTrack(self.tracker.network, fusion).to(next(self.tracker.network.parameters()).device).eval()
        runner = self

        class Network(torch.nn.Module):  # what OSTrack's tracker calls: forward(...) and .box_head
            def __init__(self):
                super().__init__()
                self.fused, self.box_head = fused, fused.net.box_head

            def forward(self, template, search, ce_template_mask=None, **kwargs):
                return self.fused(template, search, runner._vjepa_tokens(search.device), ce_template_mask)

        self.tracker.network = Network()
        self._record_scores()

    def set_vjepa(self, cache, frame_w: int, frame_h: int):
        self.vjepa = (cache, frame_w, frame_h)

    def _vjepa_tokens(self, device):
        """V-JEPA map of the current frame resampled to the search region OSTrack is about to cut."""
        import torch
        from src.tracker.fusion import align_to_crop, search_crop

        cache, w, h = self.vjepa
        fmap = torch.from_numpy(cache.at(self._frame).astype(np.float32))[None].to(device)
        crop = torch.tensor([search_crop(self.tracker.state, self.tracker.params.search_factor)],
                            dtype=torch.float32, device=device)
        return align_to_crop(fmap, crop, torch.tensor([[w, h]], dtype=torch.float32, device=device))

    def track(self, frames: list[Path], init_box, hook=None, verify: bool = False, verify_min: float = 0.0) -> dict:
        """frames: image paths in order; init_box: (x, y, w, h) on frames[0].
        hook(i, box, score) -> None, one box or a list of candidate boxes, called after every frame.
          verify=False: the (first) returned box replaces the tracker state, so the search region of
                        frame i+1 is cut around it.
          verify=True:  OSTrack re-runs on the SAME frame with its search region around every
                        candidate; the best candidate is kept only if its score beats the current one,
                        and then also replaces this frame's output box. With verify_min > 0 the
                        winner's score must also reach verify_min (a confident detection, not just
                        a less bad one than the lost box).
        Returns boxes (N, 4), scores (N,) with NaN for the init frame, per-frame seconds, the frames
        where the search region moved, and `checks` (one row per verified candidate set)."""
        n = len(frames)
        boxes, scores, times = np.zeros((n, 4)), np.full(n, np.nan), np.zeros(n)
        moved, checks = [], []
        for i, f in enumerate(frames):
            image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            self._frame = i  # V-JEPA fusion reads the map of the current frame (causal cache lookup)
            t = time.perf_counter()
            if i == 0:
                self.tracker.initialize(image, {"init_bbox": list(map(float, init_box))})
                boxes[0] = init_box
            else:
                boxes[i] = self.tracker.track(image)["target_bbox"]
                scores[i] = self.last_score
            cands = hook(i, boxes[i], scores[i]) if hook is not None else None
            if cands is not None:
                cands = [cands] if np.ndim(cands) == 1 else list(cands)
                if not verify:
                    self.tracker.state = [float(v) for v in cands[0]]
                    moved.append(i)
                else:
                    best = self._verify(image, cands, boxes[i], scores[i], verify_min)
                    checks.append(dict(frame=i, score=float(scores[i]), n_candidates=len(cands),
                                       best_score=best[1], accepted=best[0] is not None))
                    if best[0] is not None:
                        boxes[i], scores[i] = best[0], best[1]
                        moved.append(i)
            times[i] = time.perf_counter() - t
        return dict(boxes=boxes, scores=scores, times=times, moved=np.array(moved, dtype=int), checks=checks)

    def _verify(self, image, candidates, box, score, min_score: float = 0.0):
        """Run OSTrack around every candidate on the same frame; keep the best one only if OSTrack is
        more confident there than at its current box (and at least min_score). Leaves the tracker
        state on the winner."""
        best_box, best_score = None, float(score)
        for c in candidates:
            self.tracker.state = [float(v) for v in c]
            out = self.tracker.track(image)["target_bbox"]
            if self.last_score > max(best_score, min_score):
                best_box, best_score = np.asarray(out, dtype=float), float(self.last_score)
        self.tracker.state = [float(v) for v in (best_box if best_box is not None else box)]
        return best_box, best_score
