"""Parsing of LaSOT sequences (annotation files + first-frame size)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GT_NAMES = ("groundtruth.txt", "ground_truth.txt")  # official name vs. the one in your copy
IMG_EXT = (".jpg", ".jpeg", ".png")


@dataclass
class Sequence:
    cls: str
    name: str
    path: Path
    boxes: np.ndarray            # (N, 4) x, y, w, h  (absent frames are usually 0,0,0,0)
    full_occ: np.ndarray         # (N,) uint8
    out_view: np.ndarray         # (N,) uint8
    nlp: str
    n_images: int
    frame_size: tuple = (np.nan, np.nan)   # (W, H)
    issues: list = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.boxes)

    @property
    def absent(self) -> np.ndarray:
        return (self.full_occ | self.out_view).astype(bool)

    @property
    def present(self) -> np.ndarray:
        return ~self.absent

    @property
    def valid_box(self) -> np.ndarray:
        """Target present AND box has positive size."""
        return self.present & (self.boxes[:, 2] > 0) & (self.boxes[:, 3] > 0)


def _read_numbers(path: Path) -> list[list[float]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(v) for v in re.split(r"[,\s]+", line) if v])
    return rows


def _read_flags(path: Path) -> np.ndarray:
    # LaSOT stores flags as one comma-separated line; also tolerate one-per-line.
    vals = [v for row in _read_numbers(path) for v in row]
    return np.array(vals, dtype=np.uint8)


def _find(seq_dir: Path, names) -> Path | None:
    for n in names:
        if (seq_dir / n).exists():
            return seq_dir / n
    return None


def seq_number(path: Path) -> int:
    m = re.search(r"-(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def list_sequences(root: Path, cls: str) -> list[Path]:
    cls_dir = Path(root) / cls
    if not cls_dir.is_dir():
        raise FileNotFoundError(f"{cls_dir} not found")
    return sorted((p for p in cls_dir.glob(f"{cls}-*") if p.is_dir()), key=seq_number)


def load_sequence(seq_dir: Path, read_image_size: bool = True) -> Sequence:
    from PIL import Image

    seq_dir = Path(seq_dir)
    cls = seq_dir.name.rsplit("-", 1)[0]
    gt = _find(seq_dir, GT_NAMES)
    if gt is None:
        raise FileNotFoundError(f"no groundtruth file in {seq_dir}")
    boxes = np.array(_read_numbers(gt), dtype=float)
    foc = _read_flags(seq_dir / "full_occlusion.txt")
    oov = _read_flags(seq_dir / "out_of_view.txt")
    nlp_path = seq_dir / "nlp.txt"
    nlp = nlp_path.read_text().strip() if nlp_path.exists() else ""

    imgs = sorted(p for p in (seq_dir / "img").glob("*") if p.suffix.lower() in IMG_EXT)
    size = (np.nan, np.nan)
    if read_image_size and imgs:
        with Image.open(imgs[0]) as im:
            size = im.size  # (W, H)
    # scripts/fetch_lasot_annotations.py keeps only a few frames and stores the real count here
    count_file = seq_dir / "n_images.txt"
    n_images = int(count_file.read_text()) if count_file.exists() else len(imgs)

    issues = []
    n = len(boxes)
    if len(foc) != n or len(oov) != n:
        issues.append(f"length mismatch: boxes={n} foc={len(foc)} oov={len(oov)}")
    if n_images and n_images != n:
        issues.append(f"images={n_images} != boxes={n}")
    if boxes.shape[1] != 4:
        issues.append(f"boxes have {boxes.shape[1]} columns")
    # align flag lengths so downstream code does not crash; the issue stays recorded
    foc = np.resize(foc, n) if len(foc) != n else foc
    oov = np.resize(oov, n) if len(oov) != n else oov

    s = Sequence(cls, seq_dir.name, seq_dir, boxes, foc, oov, nlp, n_images, size, issues)
    if (s.present & ~s.valid_box).any():
        s.issues.append(f"{int((s.present & ~s.valid_box).sum())} 'present' frames with empty box")
    if (s.absent & (boxes[:, 2] * boxes[:, 3] > 0)).any():
        s.issues.append(f"{int((s.absent & (boxes[:, 2] * boxes[:, 3] > 0)).sum())} absent frames with non-empty box")
    return s


def load_classes(root: Path, classes: list[str], read_image_size: bool = True) -> list[Sequence]:
    out = []
    for c in classes:
        out += [load_sequence(p, read_image_size) for p in list_sequences(root, c)]
    return out


def read_split(path) -> set[str] | None:
    """One sequence name per line. Returns None if the file is missing."""
    if path is None or not Path(path).exists():
        return None
    return {l.strip() for l in Path(path).read_text().splitlines() if l.strip()}


def load_attribute_matrix(att_dir, seq_names) -> pd.DataFrame:
    """Raw official attributes: index = sequence, columns 0..13 (file order), bool values."""
    rows, missing = {}, []
    for n in seq_names:
        p = Path(att_dir) / f"{n}.txt"
        if not p.exists():
            missing.append(n)
            continue
        vals = [int(float(v)) for row in _read_numbers(p) for v in row]
        if len(vals) != 14:
            raise ValueError(f"{p} has {len(vals)} values, expected 14")
        rows[n] = vals
    if missing:
        print(f"[attributes] {len(missing)} sequences without attribute file, e.g. {missing[:3]}")
    return pd.DataFrame.from_dict(rows, orient="index", columns=range(14)).astype(bool)
