"""Write the dev (occlusion-dense test videos) and train subsets.

    python scripts/make_dev_subset.py
    python scripts/make_dev_subset.py --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import config as C  # noqa: E402
from src.data.lasot import load_sequence, read_split  # noqa: E402
from src.data.subsets import dev_occlusion_subset, occlusion_profile, train_subset  # noqa: E402

SPLITS = REPO_ROOT / "configs" / "splits"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--n", type=int, default=40, help="test: number of videos")
    p.add_argument("--max-per-class", type=int, default=1, help="test: videos per class")
    p.add_argument("--per-class", type=int, default=4, help="train: videos per class")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    split_file = C.SPLIT_FILE if a.split == "test" else C.DATA_ROOT / "training_set.txt"
    names = sorted(read_split(split_file))
    seqs = [load_sequence(C.DATA_ROOT / s.rsplit("-", 1)[0] / s, read_image_size=False) for s in names]
    prof = occlusion_profile(seqs)
    if a.split == "test":
        sub = dev_occlusion_subset(prof, a.n, a.max_per_class)
        out = a.out or SPLITS / "dev_occlusion.txt"
    else:
        sub = train_subset(prof, a.per_class)
        out = a.out or SPLITS / "train_fusion.txt"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sub.seq) + "\n")
    print(f"{len(sub)} videos ({sub.cls.nunique()} classes) -> {out}")
    print(f"frames: {sub.n_frames.sum()} ({sub.n_frames.sum() / prof.n_frames.sum():.0%} of {a.split})")
    for col in ("n_episodes", "n_long", "n_very_long"):
        print(f"{col}: {sub[col].sum()} of {prof[col].sum()} in {a.split} ({sub[col].sum() / prof[col].sum():.0%})")


if __name__ == "__main__":
    main()
