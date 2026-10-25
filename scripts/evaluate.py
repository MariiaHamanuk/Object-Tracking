"""AUC / Precision / Normalized Precision and occlusion recovery of tracker outputs on LaSOT.

    python scripts/evaluate.py --results data/raw_results/ostrack_ours/lasot --name ostrack_ours
    python scripts/evaluate.py --results ... --name ... --sequences configs/splits/dev_occlusion.txt

Writes results/<name>/{metrics.json, per_sequence.csv, episodes.csv}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import config as C  # noqa: E402
from src.data.lasot import load_sequence, read_split  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.occlusion import episodes, summarize_episodes  # noqa: E402


def load_boxes(path: Path) -> np.ndarray:
    for delim in ("\t", ","):
        try:
            return np.loadtxt(path, delimiter=delim, ndmin=2)
        except ValueError:
            pass
    raise ValueError(f"cannot parse {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True, help="folder with <seq>.txt files")
    p.add_argument("--name", required=True, help="experiment name -> results/<name>/")
    p.add_argument("--sequences", type=Path, default=None, help="split file; default: official test set")
    p.add_argument("--window", type=int, default=30, help="recovery window after reappearance (frames)")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--lock-window", type=int, default=10, help="frames before an episode checked for 'on target'")
    a = p.parse_args()

    seqs = sorted(read_split(a.sequences or C.SPLIT_FILE))
    missing = [s for s in seqs if not (a.results / f"{s}.txt").exists()]
    if missing:
        print(f"WARNING: {len(missing)} sequences without results are skipped, e.g. {missing[:3]}")
    seqs = [s for s in seqs if s not in missing]

    curves, rows, eps = [], [], []
    for name in seqs:
        s = load_sequence(C.DATA_ROOT / name.rsplit("-", 1)[0] / name, read_image_size=False)
        err = M.sequence_errors(load_boxes(a.results / f"{name}.txt"), s.boxes, s.present)
        c = M.sequence_curves(err)
        curves.append(c)
        rows.append(dict(seq=name, cls=s.cls, n_frames=s.n_frames, AUC=c["success"].mean() * 100,
                         Precision=c["precision"][20] * 100, NormPrecision=c["norm_precision"][20] * 100,
                         frac_absent=s.absent.mean()))
        eps += episodes(name, err["iou"], s.full_occ.astype(bool), s.out_view.astype(bool), err["valid"],
                        window=a.window, iou_thr=a.iou_thr, lock_window=a.lock_window)

    per_seq, ep = pd.DataFrame(rows), pd.DataFrame(eps)
    with_occ = per_seq.frac_absent > 0
    res = dict(
        results=str(a.results), split=str(a.sequences or C.SPLIT_FILE), window=a.window, iou_thr=a.iou_thr,
        lock_window=a.lock_window,
        overall=M.summarize(curves),
        videos_with_absence=M.summarize([c for c, w in zip(curves, with_occ) if w]),
        videos_without_absence=M.summarize([c for c, w in zip(curves, with_occ) if not w]),
        occlusion=summarize_episodes(ep),
    )
    out = REPO_ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    per_seq.to_csv(out / "per_sequence.csv", index=False)
    ep.to_csv(out / "episodes.csv", index=False)

    o, oc = res["overall"], res["occlusion"]
    print(f"{a.name}: {o['n_sequences']} videos | AUC {o['AUC']:.2f} | P {o['Precision']:.2f} | "
          f"NP {o['NormPrecision']:.2f}")
    for k in ("videos_with_absence", "videos_without_absence"):
        if res[k]["n_sequences"]:
            print(f"  {k}: {res[k]['n_sequences']} videos, AUC {res[k]['AUC']:.2f}")
    if oc:
        print(f"  ORR {oc['ORR']:.1f}% on {oc['n_locked_before']} episodes (tracker on target before) | "
              f"median frames to recover {oc['median_frames_to_recover']:.0f}")
        for k, v in oc["ORR_by_length"].items():
            print(f"    length {k:>7}: n={v['n']:4d}  ORR={v['ORR'] if v['ORR'] is None else round(v['ORR'], 1)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
