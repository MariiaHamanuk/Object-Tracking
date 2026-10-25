"""Fetch LaSOT annotations, a few frames per video, split files and attributes (no full frames).

    python scripts/fetch_lasot_annotations.py
    python scripts/fetch_lasot_annotations.py --classes person,cat --samples 4
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data.remote_zip import ANNOT_FILES, HF_FILE, list_classes, open_class_zip, retry  # noqa: E402

ATT_URL = "https://raw.githubusercontent.com/HengLan/LaSOT_Evaluation_Toolkit/master/annos/att/{}.txt"
SPLIT_FILES = ("testing_set.txt", "training_set.txt")
DONE_MARK = ".fetched"


def sample_indices(n_frames: int, n: int) -> np.ndarray:
    # same spacing as plots.show_samples, so the sanity-check frames are on disk
    return np.unique(np.linspace(0, n_frames - 1, n).astype(int)) if n > 0 else np.array([0])


def fetch_class(cls: str, root: Path, n_samples: int) -> str:
    cls_dir = root / cls
    if (cls_dir / DONE_MARK).exists():
        return f"{cls}: already fetched"
    z = open_class_zip(cls)
    names = z.namelist()
    frames: dict[str, list[str]] = {}
    for n in names:
        parts = n.split("/")
        if len(parts) == 3 and parts[1] == "img" and parts[2].lower().endswith(".jpg"):
            frames.setdefault(parts[0], []).append(n)

    seq_names = sorted({n.split("/")[0] for n in names if n.split("/")[0].startswith(f"{cls}-")})
    for seq in seq_names:
        seq_dir = cls_dir / seq
        (seq_dir / "img").mkdir(parents=True, exist_ok=True)
        for f in ANNOT_FILES:
            if f"{seq}/{f}" in names:
                (seq_dir / f).write_bytes(z.read(f"{seq}/{f}"))
        imgs = sorted(frames.get(seq, []))
        (seq_dir / "n_images.txt").write_text(str(len(imgs)))
        for i in sample_indices(len(imgs), n_samples) if imgs else []:
            (seq_dir / imgs[i].split("/", 1)[1]).write_bytes(z.read(imgs[i]))
    (cls_dir / DONE_MARK).write_text("ok")
    return f"{cls}: {len(seq_names)} sequences"


def fetch_attributes(seq_names: list[str], att_dir: Path, workers: int):
    att_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seq_names if not (att_dir / f"{s}.txt").exists()]
    session = requests.Session()

    def get(seq):
        try:
            r = retry(lambda: session.get(ATT_URL.format(seq), timeout=60), tries=4)
        except requests.RequestException:
            return False
        (att_dir / f"{seq}.txt").write_text(r.text)
        return True

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(get, todo))
    print(f"[attributes] downloaded {sum(ok)}/{len(todo)} (missing: {len(todo) - sum(ok)})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=C.DATA_ROOT, help="output dir (default: $LASOT_ROOT or ./data)")
    p.add_argument("--classes", default="all", help="comma-separated, or 'all'")
    p.add_argument("--samples", type=int, default=4, help="evenly spaced frames to keep per video (incl. first)")
    p.add_argument("--workers", type=int, default=4, help="classes fetched in parallel")
    a = p.parse_args()

    root = a.root
    root.mkdir(parents=True, exist_ok=True)
    classes = list_classes() if a.classes == "all" else a.classes.split(",")
    print(f"{len(classes)} classes -> {root}")

    for f in SPLIT_FILES:
        if not (root / f).exists():
            (root / f).write_bytes(retry(lambda: requests.get(HF_FILE.format(f), timeout=60)).content)

    with ThreadPoolExecutor(a.workers) as ex:
        futures = {c: ex.submit(fetch_class, c, root, a.samples) for c in classes}
        failed = []
        for c, fut in futures.items():
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                failed.append(c)
                print(f"{c}: FAILED ({e})", flush=True)

    seqs = sorted(p.name for c in classes for p in (root / c).glob(f"{c}-*") if p.is_dir())
    fetch_attributes(seqs, C.ATTRIBUTE_DIR if a.root == C.DATA_ROOT else root / "att", a.workers * 4)
    if failed:
        sys.exit(f"failed classes: {','.join(failed)} - run again to retry")


if __name__ == "__main__":
    main()
