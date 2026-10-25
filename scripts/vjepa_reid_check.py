"""Can V-JEPA features re-find the target after an occlusion? Memory from before the episode is
matched against the map of the first frame after it, with a real clip and with a repeated frame.

    python scripts/vjepa_reid_check.py --max-episodes 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.lasot import load_sequence  # noqa: E402
from src.data.remote_zip import extract_frames, open_class_zip  # noqa: E402
from src.eval.metrics import iou  # noqa: E402
from src.vjepa.encoder import VJEPAEncoder, box_to_grid, clip_indices, pool_box  # noqa: E402


def load_rgb(seq_dir: Path, i: int) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(seq_dir / "img" / f"{i + 1:08d}.jpg")), cv2.COLOR_BGR2RGB)


def overlap_mask(box, w, h, g=16) -> np.ndarray:
    x0, y0, x1, y1 = box_to_grid(box, w, h, g)
    e = np.arange(g + 1)
    ox = np.minimum(e[1:], x1) - np.maximum(e[:-1], x0) > 0
    oy = np.minimum(e[1:], y1) - np.maximum(e[:-1], y0) > 0
    return oy[:, None] & ox[None, :]


def score(m: torch.Tensor, fmap: torch.Tensor, gt_box, trk_box, w, h) -> dict:
    sim = torch.nn.functional.cosine_similarity(fmap, m[None, None], dim=-1).numpy()
    gt_mask = overlap_mask(gt_box, w, h, sim.shape[0])
    s_gt = float(torch.nn.functional.cosine_similarity(pool_box(fmap, gt_box, w, h), m, dim=0))
    s_trk = float(torch.nn.functional.cosine_similarity(pool_box(fmap, trk_box, w, h), m, dim=0))
    top = np.unravel_index(sim.argmax(), sim.shape)
    return dict(top1_hit=bool(gt_mask[top]), chance=float(gt_mask.mean()),
                gt_percentile=float((sim < s_gt).mean() * 100), sim_gt=s_gt, sim_ostrack=s_trk)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, default=REPO_ROOT / "results" / "ostrack_official_dev" / "episodes.csv")
    p.add_argument("--tracker", type=Path, default=REPO_ROOT / "data" / "raw_results" / "ostrack_official" / "lasot")
    p.add_argument("--min-length", type=int, default=6)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--work", type=Path, default=REPO_ROOT / "tmp_frames" / "reid_check")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "vjepa_reid_check")
    a = p.parse_args()

    ep = pd.read_csv(a.episodes)
    ep = ep[ep.locked_before & (ep.length >= a.min_length)].reset_index(drop=True)
    if a.max_episodes:
        ep = ep.sample(min(a.max_episodes, len(ep)), random_state=0).reset_index(drop=True)
    print(f"{len(ep)} episodes from {ep.seq.nunique()} videos", flush=True)

    enc = VJEPAEncoder()
    a.out.mkdir(parents=True, exist_ok=True)
    done_file = a.out / "episodes.csv"
    rows = pd.read_csv(done_file).to_dict("records") if done_file.exists() else []
    done = {(r["seq"], r["start"]) for r in rows}
    t0, zips = time.time(), {}

    for k, e in ep.iterrows():
        if (e.seq, e.start) in done:
            continue
        cls = e.seq.rsplit("-", 1)[0]
        before = int(e.start) - 1
        after = int(e.start + e.length)
        need = np.r_[clip_indices(before, a.clip_len, a.stride), clip_indices(after, a.clip_len, a.stride)]
        if cls not in zips:
            zips = {cls: open_class_zip(cls)}
        seq_dir = extract_frames(zips[cls], e.seq, need, a.work)
        s = load_sequence(seq_dir, read_image_size=False)
        h, w = load_rgb(seq_dir, before).shape[:2]
        trk = np.loadtxt(a.tracker / f"{e.seq}.txt", delimiter="\t", ndmin=2)
        trk_after, gt_after = trk[after], s.boxes[after]

        row = dict(seq=e.seq, start=int(e.start), length=int(e.length), kind=e.kind,
                   ostrack_iou_after=float(iou(trk_after[None], gt_after[None])[0]))
        for mode in ("motion", "static"):
            def clip(t):
                idx = clip_indices(t, a.clip_len, a.stride) if mode == "motion" else [t] * a.clip_len
                return [load_rgb(seq_dir, i) for i in idx]
            m = pool_box(enc.encode(clip(before)), s.boxes[before], w, h)
            for key, v in score(m, enc.encode(clip(after)), gt_after, trk_after, w, h).items():
                row[f"{mode}_{key}"] = v
        rows.append(row)
        pd.DataFrame(rows).to_csv(done_file, index=False)
        el = time.time() - t0
        print(f"[{k + 1}/{len(ep)}] {e.seq} len={e.length}: motion top1={row['motion_top1_hit']} "
              f"pct={row['motion_gt_percentile']:.0f} | static top1={row['static_top1_hit']} "
              f"pct={row['static_gt_percentile']:.0f} | {el / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    lost = df[df.ostrack_iou_after < 0.5]
    summary = {"n_episodes": len(df), "n_ostrack_lost_at_reappearance": len(lost)}
    for mode in ("motion", "static"):
        summary[mode] = dict(
            top1_hit=float(df[f"{mode}_top1_hit"].mean() * 100),
            top1_chance=float(df[f"{mode}_chance"].mean() * 100),
            gt_percentile=float(df[f"{mode}_gt_percentile"].mean()),
            beats_ostrack_when_lost=float((lost[f"{mode}_sim_gt"] > lost[f"{mode}_sim_ostrack"]).mean() * 100)
            if len(lost) else None)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
