"""Track with V-JEPA alone from the cache.

    python scripts/track_vjepa_only.py --sequences configs/splits/dev_occlusion.txt \\
        --vjepa-cache cache/vjepa --out results_raw/vjepa_only/lasot
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.lasot import read_split  # noqa: E402
from src.data.remote_zip import extract_frames, open_class_zip  # noqa: E402
from src.tracker.fusion import VJEPACache  # noqa: E402
from src.tracker.vjepa_only import VJEPAOnlyTracker  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", required=True, help="split file or comma-separated names")
    p.add_argument("--vjepa-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, default=REPO_ROOT / "tmp_frames")
    p.add_argument("--max-frames", type=int, default=None, help="only for quick tests")
    a = p.parse_args()

    seqs = sorted(read_split(a.sequences)) if Path(a.sequences).is_file() else a.sequences.split(",")
    a.out.mkdir(parents=True, exist_ok=True)
    for k, seq in enumerate(seqs, 1):
        if (a.out / f"{seq}.txt").exists():
            continue
        seq_dir = extract_frames(open_class_zip(seq.rsplit("-", 1)[0]), seq, [0], a.work / "vjepa_only")
        gt = np.loadtxt(seq_dir / "groundtruth.txt", delimiter=",", ndmin=2)
        h, w = cv2.imread(str(next((seq_dir / "img").glob("*.jpg")))).shape[:2]
        n = len(gt) if a.max_frames is None else min(len(gt), a.max_frames)
        boxes, scores = VJEPAOnlyTracker(VJEPACache(a.vjepa_cache / f"{seq}.npz"), w, h, gt[0]).track(n)
        np.savetxt(a.out / f"{seq}_score.txt", scores, fmt="%.4f")
        np.savetxt(a.out / f"{seq}.txt", boxes, fmt="%.2f", delimiter="\t")
        shutil.rmtree(seq_dir, ignore_errors=True)
        print(f"[{k}/{len(seqs)}] {seq}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
