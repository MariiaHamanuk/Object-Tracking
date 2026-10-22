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

        # record the confidence of every forward pass without touching OSTrack's code
        net, forward = self.tracker.network, self.tracker.network.forward
        self.last_score = np.nan

        def forward_with_score(*args, **kwargs):
            out = forward(*args, **kwargs)
            self.last_score = float(out["score_map"].max())
            return out

        net.forward = forward_with_score

    def track(self, frames: list[Path], init_box) -> dict:
        """frames: image paths in order; init_box: (x, y, w, h) on frames[0].
        Returns boxes (N, 4), scores (N,) with NaN for the init frame, and per-frame seconds."""
        n = len(frames)
        boxes, scores, times = np.zeros((n, 4)), np.full(n, np.nan), np.zeros(n)
        for i, f in enumerate(frames):
            image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            t = time.perf_counter()
            if i == 0:
                self.tracker.initialize(image, {"init_bbox": list(map(float, init_box))})
                boxes[0] = init_box
            else:
                boxes[i] = self.tracker.track(image)["target_bbox"]
                scores[i] = self.last_score
            times[i] = time.perf_counter() - t
        return dict(boxes=boxes, scores=scores, times=times)
