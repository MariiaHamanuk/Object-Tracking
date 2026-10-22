"""Compare two EDA runs (CSV folders written by src/eda/eda_lasot.ipynb), e.g. person-only vs full LaSOT.

    python scripts/compare_eda.py                                  # results/eda_person vs results/eda
    python scripts/compare_eda.py --old results/eda_person --new results/eda --out results/eda/comparison.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQ_COLS = ["n_frames", "duration_s", "frac_foc", "frac_oov", "frac_absent",
            "n_foc_episodes", "n_oov_episodes", "max_foc_len", "max_oov_len"]


def load(d: Path) -> dict[str, pd.DataFrame]:
    return dict(seqs=pd.read_csv(d / "sequences.csv"),
                episodes=pd.read_csv(d / "episodes.csv"),
                attr=pd.read_csv(d / "attributes_per_sequence.csv", index_col=0))


def overview(r: dict) -> pd.Series:
    s = r["seqs"]
    return pd.Series({
        "classes": s.cls.nunique(), "sequences": len(s), "frames": int(s.n_frames.sum()),
        "hours": round(s.duration_s.sum() / 3600, 2),
        "% videos with FOC": 100 * (s.n_foc > 0).mean(), "% videos with OOV": 100 * (s.n_oov > 0).mean(),
        "% frames absent": 100 * s.n_foc.add(s.n_oov).sub(s.n_both).sum() / s.n_frames.sum(),
        "absence episodes": len(r["episodes"]), "resolutions": s.resolution.nunique(),
        "sequences with issues": int(s.has_issues.sum()),
    })


def episode_summary(r: dict) -> pd.DataFrame:
    e = r["episodes"]
    return e.groupby("kind").length.describe(percentiles=[.5, .9]).rename(columns={"50%": "median", "90%": "p90"})


def attr_prevalence(r: dict) -> pd.Series:
    a = r["attr"].drop(columns="cls")
    return 100 * a.astype(bool).mean()


def consistency(old: dict, new: dict) -> pd.DataFrame:
    """Sequences present in both runs must give the same numbers - otherwise parsing changed."""
    o, n = old["seqs"].set_index("seq"), new["seqs"].set_index("seq")
    common = o.index.intersection(n.index)
    cols = [c for c in SEQ_COLS + ["width", "height", "n_foc", "n_oov"] if c in o and c in n]
    diff = (o.loc[common, cols] - n.loc[common, cols]).abs() > 1e-9
    bad = diff[diff.any(axis=1)]
    return pd.DataFrame({"differs": [", ".join(r.index[r]) for _, r in bad.iterrows()]}, index=bad.index)


def class_ranking(new: dict, classes_old: set[str]) -> pd.DataFrame:
    s = new["seqs"]
    g = s.groupby("cls").agg(videos=("seq", "size"), mean_frames=("n_frames", "mean"),
                             frac_absent=("frac_absent", "mean"),
                             pct_videos_foc=("n_foc", lambda x: 100 * (x > 0).mean()),
                             pct_videos_oov=("n_oov", lambda x: 100 * (x > 0).mean()))
    g["rank_absent"] = g.frac_absent.rank(ascending=False).astype(int)
    g["in_old_run"] = g.index.isin(classes_old)
    return g.sort_values("frac_absent", ascending=False)


def md(df) -> str:
    return df.to_markdown(floatfmt=".3g")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, default=REPO_ROOT / "results" / "eda_person")
    p.add_argument("--new", type=Path, default=REPO_ROOT / "results" / "eda")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "eda" / "comparison.md")
    a = p.parse_args()
    old, new = load(a.old), load(a.new)
    old_cls = set(old["seqs"].cls)
    new_same = {k: (v[v.cls.isin(old_cls)] if "cls" in v else v) for k, v in new.items()}

    parts = [f"# EDA comparison: `{a.old.name}` vs `{a.new.name}`\n"]

    parts += ["## Overview\n", md(pd.DataFrame({"old": overview(old), "new": overview(new)})), ""]

    parts += ["## Per-sequence statistics (mean)\n",
              md(pd.DataFrame({"old": old["seqs"][SEQ_COLS].mean(), "new": new["seqs"][SEQ_COLS].mean(),
                               "new / old": new["seqs"][SEQ_COLS].mean() / old["seqs"][SEQ_COLS].mean()})), ""]

    parts += ["## Absence-episode length (frames)\n",
              md(pd.concat({"old": episode_summary(old), "new": episode_summary(new)})), ""]

    po, pn = attr_prevalence(old), attr_prevalence(new)
    prev = pd.DataFrame({"old %": po, "new %": pn}).reindex(pn.index.union(po.index, sort=False))
    prev["new % (old classes)"] = attr_prevalence(new_same)
    parts += ["## Attribute prevalence (% of videos)\n",
              "Old run had derived proxies only; new run uses official labels where available.\n",
              md(prev), ""]

    bad = consistency(old, new)
    n_common = len(set(old["seqs"].seq) & set(new["seqs"].seq))
    verdict = ("No overlapping sequences." if n_common == 0 else
               f"All {n_common} overlapping sequences match." if bad.empty else md(bad))
    parts += ["## Consistency on overlapping sequences\n", verdict, ""]

    parts += ["## Classes ranked by mean fraction of absent frames\n",
              md(class_ranking(new, old_cls)), ""]

    text = "\n".join(parts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
