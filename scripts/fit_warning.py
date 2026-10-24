"""Fit the "learned" occlusion warning (src/tracker/relocalize.py, warning="learned") on TRAIN videos.

Input: the <seq>_reloc.csv logs of a run on configs/splits/warning_train.txt (every check's causal
features: ranks of the score drop / predictor surprise / ..., V-JEPA similarities) and the LaSOT
annotations. Label of a check at frame t: the target is hidden at some frame in [t, t + horizon].
A logistic regression is fitted per feature set and scored by leave-one-video-out AUROC, so the
predictor's contribution is measured on train, not on the dev videos we report.

    python scripts/fit_warning.py --logs results_raw/warning_train/lasot --out configs/warning_weights.json

Writes the weights of the best feature set that uses the predictor, and a table of all sets.
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
import src.config as C  # noqa: E402
from src.data.lasot import load_sequence  # noqa: E402

SETS = {
    "score only (no predictor)": ["r_score"],
    "score + V-JEPA similarity": ["r_score", "s_box", "gap"],
    "score + surprise": ["r_score", "r_surp"],
    "score + local surprise": ["r_score", "r_local"],
    "score + local + rise": ["r_score", "r_local", "r_rise"],
    "score + online surprise": ["r_score", "r_online"],
    "all": ["r_score", "r_surp", "r_local", "r_rise", "r_online", "s_box", "gap"],
}
PREDICTOR = {"r_surp", "r_local", "r_rise", "r_online"}


def auroc(s: np.ndarray, label: np.ndarray) -> float:
    pos, neg = s[label], s[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()[: len(pos)]
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    from sklearn.linear_model import LogisticRegression

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "configs" / "warning_weights.json")
    p.add_argument("--horizon", type=int, default=16)
    a = p.parse_args()

    rows = []
    for f in sorted(a.logs.glob("*_reloc.csv")):
        seq = f.name[: -len("_reloc.csv")]
        s = load_sequence(C.DATA_ROOT / seq.rsplit("-", 1)[0] / seq, read_image_size=False)
        d = pd.read_csv(f)
        d["seq"] = seq
        d["y"] = [bool(s.absent[t:t + a.horizon + 1].any()) for t in d.frame]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    print(f"{len(d)} checks from {d.seq.nunique()} videos, {d.y.mean():.1%} positive")

    table, fits = [], {}
    for name, feats in SETS.items():
        if any(f not in d for f in feats):
            continue
        X, y = d[feats].to_numpy(float), d.y.to_numpy(bool)
        out = np.zeros(len(d))
        for seq in d.seq.unique():  # leave one video out
            m = (d.seq == seq).to_numpy()
            out[m] = LogisticRegression(max_iter=1000).fit(X[~m], y[~m]).decision_function(X[m])
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        fits[name] = dict(bias=float(clf.intercept_[0]), coef=dict(zip(feats, map(float, clf.coef_[0]))))
        table.append(dict(features=name, auroc_leave_video_out=auroc(out, y)))
    table = pd.DataFrame(table)
    print(table.round(3).to_string(index=False))
    with_pred = table[[bool(set(SETS[n]) & PREDICTOR) for n in table.features]]
    best = with_pred.sort_values("auroc_leave_video_out").iloc[-1].features
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(features=best, **fits[best], table=table.to_dict("records")), indent=2))
    print(f"-> {a.out} ({best})")


if __name__ == "__main__":
    main()
