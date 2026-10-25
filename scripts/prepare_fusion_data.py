"""Training samples for the fusion: occlusion, absent and normal crops with V-JEPA maps and memory.

    python scripts/prepare_fusion_data.py --max-videos 2 --per-video 4
    python scripts/prepare_fusion_data.py --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.lasot import load_sequence, read_split  # noqa: E402
from src.data.remote_zip import extract_frames, open_class_zip  # noqa: E402
from src.eda.stats import runs  # noqa: E402
from src.tracker.fusion import search_crop  # noqa: E402
from src.tracker.ostrack import setup_ostrack  # noqa: E402
from src.vjepa.encoder import DEFAULT_MODEL, VJEPAEncoder, clip_indices, pool_box  # noqa: E402

TEMPLATE_FACTOR, TEMPLATE_SIZE = 2.0, 128
SEARCH_FACTOR, SEARCH_SIZE = 4.0, 256
CENTER_JITTER, SCALE_JITTER = 3.0, 0.25   # OSTrack's search jitter
MAX_TEMPLATE_GAP = 100
OCC_OFFSET = 10


def plan_samples(s, n_per_kind: int, rng: np.random.Generator) -> list[dict]:
    """Choose (template frame, search frame, crop-centre box) triples for one video."""
    ok = s.valid_box
    plans = []

    def template_for(t):
        cand = np.flatnonzero(ok[max(0, t - MAX_TEMPLATE_GAP):t]) + max(0, t - MAX_TEMPLATE_GAP)
        return int(rng.choice(cand)) if len(cand) else None

    # occlusion: search crop around the box where the target vanished, frames after it returns
    starts, lengths = runs(s.absent)
    eps = [(a, l) for a, l in zip(starts, lengths) if a > 0 and a + l < s.n_frames and ok[a - 1]]
    for a, l in sorted(eps, key=lambda e: -e[1])[:n_per_kind]:
        back = a + l
        cand = [t for t in range(back, min(back + OCC_OFFSET, s.n_frames)) if ok[t]]
        tmpl = template_for(a)
        if cand and tmpl is not None:
            plans.append(dict(kind=1, t=int(rng.choice(cand)), template_t=tmpl, centre=s.boxes[a - 1].copy()))
            plans.append(dict(kind=2, t=int(rng.integers(a, back)), template_t=tmpl, centre=s.boxes[a - 1].copy()))

    n_norm = max(sum(p["kind"] == 1 for p in plans), min(n_per_kind, 4))
    cand = np.flatnonzero(ok[1:]) + 1
    for t in rng.choice(cand, size=min(n_norm, len(cand)), replace=False) if len(cand) else []:
        tmpl = template_for(int(t))
        if tmpl is None:
            continue
        x, y, w, h = s.boxes[t]
        size = np.array([w, h]) * np.exp(rng.standard_normal(2) * SCALE_JITTER)
        centre = np.array([x + w / 2, y + h / 2]) + np.sqrt(size.prod()) * CENTER_JITTER * (rng.random(2) - 0.5)
        plans.append(dict(kind=0, t=int(t), template_t=tmpl, centre=np.r_[centre - size / 2, size]))
    return plans


def crop_with_box(image, centre_box, gt_box, factor, size):
    """OSTrack's crop + the GT box in normalised crop coordinates (x, y, w, h in 0..1)."""
    from lib.train.data.processing_utils import sample_target, transform_image_to_crop
    crop, rf, _ = sample_target(image, list(map(float, centre_box)), factor, output_sz=size)
    box = transform_image_to_crop(torch.tensor(gt_box, dtype=torch.float32),
                                  torch.tensor(centre_box, dtype=torch.float32), rf,
                                  torch.tensor([size, size], dtype=torch.float32), normalize=True)
    return crop, box.numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", type=Path, default=REPO_ROOT / "configs" / "splits" / "train_fusion.txt")
    p.add_argument("--per-video", type=int, default=15, help="occlusion samples per video (normal: as many)")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--static", action="store_true", help="control: V-JEPA sees the search frame repeated")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "cache" / "fusion_data")
    p.add_argument("--work", type=Path, default=REPO_ROOT / "tmp_frames")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()

    setup_ostrack(a.work / "ostrack_output")
    out = a.out / ("static" if a.static else "motion")
    out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(read_split(a.sequences))[: a.max_videos][a.shard::a.num_shards]
    todo = [s for s in seqs if not (out / f"{s}.npz").exists()]
    print(f"shard {a.shard}/{a.num_shards}: {len(seqs)} videos, {len(todo)} to prepare -> {out}", flush=True)
    if not todo:
        return
    enc = VJEPAEncoder(a.model)
    work = a.work / f"fusion_shard{a.shard}"
    t0, n_total, zips = time.time(), 0, {}

    for k, seq in enumerate(todo, 1):
        cls = seq.rsplit("-", 1)[0]
        try:
            if cls not in zips:
                zips = {cls: open_class_zip(cls)}
            seq_dir = extract_frames(zips[cls], seq, [], work)
            s = load_sequence(seq_dir, read_image_size=False)
            rng = np.random.default_rng(zlib.crc32(seq.encode()))
            plans = plan_samples(s, a.per_video, rng)
            if not plans:
                print(f"[{k}/{len(todo)}] {seq}: no samples", flush=True)
                continue
            clips = [[pl["t"]] * a.clip_len if a.static else clip_indices(pl["t"], a.clip_len, a.stride)
                     for pl in plans]
            mem_clips = [clip_indices(0, a.clip_len, a.stride)] + [clip_indices(pl["template_t"], a.clip_len, a.stride)
                                                                    for pl in plans]
            need = np.unique(np.r_[np.concatenate(clips + mem_clips), [pl["template_t"] for pl in plans]])
            extract_frames(zips[cls], seq, need, work)
        except Exception as e:
            print(f"[{k}/{len(todo)}] {seq}: FAILED ({e})", flush=True)
            continue

        frames: dict[int, np.ndarray] = {}

        def img(i):
            if i not in frames:
                frames[i] = cv2.cvtColor(cv2.imread(str(seq_dir / "img" / f"{i + 1:08d}.jpg")), cv2.COLOR_BGR2RGB)
            return frames[i]

        h, w = img(plans[0]["t"]).shape[:2]
        rec = {k_: [] for k_ in ("template", "template_box", "search", "search_box", "crop", "kind", "t", "in_crop")}
        for pl in plans:
            tc, tb = crop_with_box(img(pl["template_t"]), s.boxes[pl["template_t"]], s.boxes[pl["template_t"]],
                                   TEMPLATE_FACTOR, TEMPLATE_SIZE)
            sc, sb = crop_with_box(img(pl["t"]), pl["centre"], s.boxes[pl["t"]], SEARCH_FACTOR, SEARCH_SIZE)
            if pl["kind"] == 2:  # LaSOT often keeps a stale box on absent frames
                sb = np.zeros(4, dtype=np.float32)
            cx, cy = (sb[0] + sb[2] / 2, sb[1] + sb[3] / 2) if pl["kind"] != 2 else (-1.0, -1.0)
            rec["template"].append(tc)
            rec["template_box"].append(tb)
            rec["search"].append(sc)
            rec["search_box"].append(sb)
            rec["crop"].append(search_crop(pl["centre"], SEARCH_FACTOR))
            rec["kind"].append(pl["kind"])
            rec["t"].append(pl["t"])
            rec["in_crop"].append(0 <= cx <= 1 and 0 <= cy <= 1)
        def encode(cl):
            return np.concatenate([enc.encode_batch([[img(int(i)) for i in c] for c in cl[b:b + a.batch]]).numpy()
                                   for b in range(0, len(cl), a.batch)])

        mem_boxes = [s.boxes[0]] + [s.boxes[pl["template_t"]] for pl in plans]
        mem = np.stack([pool_box(torch.from_numpy(m.astype(np.float32)), bx, w, h).numpy()
                        for m, bx in zip(encode(mem_clips), mem_boxes)]).astype(np.float16)
        arrays = {k_: np.asarray(v) for k_, v in rec.items()}
        np.savez(out / f"{seq}.npz", vmap=encode(clips), frame_wh=np.array([w, h]),
                 mem_first=np.repeat(mem[:1], len(plans), 0), mem_recent=mem[1:], **arrays)
        shutil.rmtree(seq_dir, ignore_errors=True)

        n_total += len(plans)
        el = time.time() - t0
        print(f"[{k}/{len(todo)}] {seq}: {len(plans)} samples ({int((arrays['kind'] == 1).sum())} occlusion, "
              f"{int((arrays['kind'] == 2).sum())} absent, "
              f"{int(arrays['in_crop'].sum())} with target in crop) | {n_total / el:.2f} samples/s | "
              f"eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
