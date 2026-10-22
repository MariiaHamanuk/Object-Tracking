"""Pick the development subset: occlusion-dense test videos, balanced over classes.

    python scripts/make_dev_subset.py            # -> configs/splits/dev_occlusion.txt (40 videos)
    python scripts/make_dev_subset.py --n 30 --max-per-class 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import config as C  # noqa: E402
from src.data.lasot import load_sequence, read_split  # noqa: E402
from src.data.subsets import dev_occlusion_subset, occlusion_profile  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--max-per-class", type=int, default=1)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "configs" / "splits" / "dev_occlusion.txt")
    a = p.parse_args()

    test = sorted(read_split(C.SPLIT_FILE))
    seqs = [load_sequence(C.DATA_ROOT / s.rsplit("-", 1)[0] / s, read_image_size=False) for s in test]
    prof = occlusion_profile(seqs)
    dev = dev_occlusion_subset(prof, a.n, a.max_per_class)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(dev.seq) + "\n")
    print(f"{len(dev)} videos -> {a.out}")
    print(f"frames: {dev.n_frames.sum()} ({dev.n_frames.sum() / prof.n_frames.sum():.0%} of test)")
    for col in ("n_episodes", "n_long", "n_very_long"):
        print(f"{col}: {dev[col].sum()} of {prof[col].sum()} in test ({dev[col].sum() / prof[col].sum():.0%})")


if __name__ == "__main__":
    main()
