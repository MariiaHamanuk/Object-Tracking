"""LaSOT statistics as DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.lasot import Sequence

ATTRIBUTES = ["CM", "VC", "ROT", "SV", "DEF", "BC", "POC", "FOC", "MB", "IV", "ARC", "OV", "LR", "FM"]
DERIVED_ATTRS = ["SV", "ARC", "LR", "FM", "FOC", "OV"]  # computable from the annotations


def runs(mask: np.ndarray):
    """Start indices and lengths of consecutive True runs."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def sequence_table(seqs: list[Sequence], fps: int = C.FPS) -> pd.DataFrame:
    rows = []
    for s in seqs:
        n = s.n_frames
        foc, oov = s.full_occ.astype(bool), s.out_view.astype(bool)
        both = foc & oov
        fs, fl = runs(foc)
        os_, ol = runs(oov)
        abs_starts, _ = runs(s.absent)
        rows.append(dict(
            cls=s.cls, seq=s.name, n_frames=n, duration_s=n / fps,
            width=s.frame_size[0], height=s.frame_size[1],
            n_foc=int(foc.sum()), n_oov=int(oov.sum()), n_both=int(both.sum()),
            frac_foc=foc.mean(), frac_oov=oov.mean(), frac_absent=s.absent.mean(),
            n_foc_episodes=len(fs), n_oov_episodes=len(os_),
            max_foc_len=int(fl.max()) if len(fl) else 0,
            max_oov_len=int(ol.max()) if len(ol) else 0,
            first_absence_pos=(abs_starts[0] / n) if len(abs_starts) else np.nan,
            has_issues=bool(s.issues), nlp=s.nlp,
        ))
    df = pd.DataFrame(rows)
    df["resolution"] = df.width.astype("Int64").astype(str) + "x" + df.height.astype("Int64").astype(str)
    return df


def frame_table(seqs: list[Sequence]) -> pd.DataFrame:
    parts = []
    for s in seqs:
        x, y, w, h = s.boxes.T
        W, H = s.frame_size
        valid = s.valid_box
        with np.errstate(divide="ignore", invalid="ignore"):
            area = np.where(valid, w * h, np.nan)
            aspect = np.where(valid, w / np.where(h > 0, h, np.nan), np.nan)
        idx = np.flatnonzero(valid)
        scale_ratio = np.full(s.n_frames, np.nan)
        ar_ratio = np.full(s.n_frames, np.nan)
        if len(idx):
            scale_ratio = area / area[idx[0]]
            ar_ratio = aspect / aspect[idx[0]]
        disp = np.full(s.n_frames, np.nan)
        both = valid[1:] & valid[:-1]
        cx, cy = x + w / 2, y + h / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.hypot(np.diff(cx), np.diff(cy)) / np.sqrt(area[:-1])
        disp[1:] = np.where(both, d, np.nan)
        state = np.select([s.full_occ.astype(bool) & s.out_view.astype(bool), s.full_occ.astype(bool), s.out_view.astype(bool)],
                          ["both", "full_occlusion", "out_of_view"], default="present")
        parts.append(pd.DataFrame(dict(
            cls=s.cls, seq=s.name, frame=np.arange(s.n_frames), state=state,
            x=x, y=y, w=w, h=h, area_px=area, rel_area=area / (W * H), aspect=aspect,
            scale_ratio=scale_ratio, ar_ratio=ar_ratio, disp_norm=disp,
        )))
    return pd.concat(parts, ignore_index=True)


def derived_attributes(frames: pd.DataFrame, seqs_df: pd.DataFrame) -> pd.DataFrame:
    """SV, ARC, LR, FM, FOC, OV from the annotations (LaSOT paper, Table 2)."""
    g = frames.groupby("seq")
    lo, hi = C.SV_RANGE
    alo, ahi = C.ARC_RANGE
    out = pd.DataFrame(index=seqs_df.seq)
    out["SV"] = g.scale_ratio.apply(lambda r: bool(((r < lo) | (r > hi)).any()))
    out["ARC"] = g.ar_ratio.apply(lambda r: bool(((r < alo) | (r > ahi)).any()))
    out["LR"] = g.area_px.apply(lambda a: bool((a < C.LR_AREA_PX).any()))
    out["FM"] = g.disp_norm.apply(lambda d: bool((d > C.FM_DISP_RATIO).any()))
    out["FOC"] = (seqs_df.set_index("seq").n_foc > 0)
    out["OV"] = (seqs_df.set_index("seq").n_oov > 0)
    out.insert(0, "cls", seqs_df.set_index("seq").cls)
    return out


def check_attribute_columns(raw: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """Agreement between each raw attribute column and each derived attribute (checks the column order)."""
    common = raw.index.intersection(derived.index)
    out = pd.DataFrame({a: [(raw.loc[common, c] == derived.loc[common, a]).mean() for c in raw.columns]
                        for a in DERIVED_ATTRS}, index=raw.columns)
    out.insert(0, "prevalence", raw.loc[common].mean())
    return out


def name_attribute_columns(raw: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    assert len(order) == 14 and set(order) == set(ATTRIBUTES), f"order must be a permutation of {ATTRIBUTES}"
    out = raw.copy()
    out.columns = order
    return out


def merge_official_attributes(derived: pd.DataFrame, official: pd.DataFrame | None) -> pd.DataFrame:
    """Official labels override the derived ones."""
    if official is None:
        return derived
    out = derived.copy()
    for a in official.columns:
        out[a] = official[a].reindex(out.index).fillna(False).astype(bool)
    return out


def attribute_counts(attr: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in attr.columns if c != "cls"]
    return attr.groupby("cls")[cols].sum().astype(int)


def coverage_table(attr: pd.DataFrame, test_names: set[str]) -> pd.DataFrame:
    """Class x attribute: number of test videos with the attribute."""
    t = attr[attr.index.isin(test_names)]
    return attribute_counts(t)


def temporal_density(seqs: list[Sequence], kind: str = "full_occ", bins: int = 50) -> pd.DataFrame:
    """Fraction of absent frames per video and normalised-time bin."""
    rows, names = [], []
    for s in seqs:
        mask = getattr(s, kind).astype(float)
        idx = np.arange(s.n_frames) * bins // s.n_frames
        rows.append(np.bincount(idx, weights=mask, minlength=bins) / np.bincount(idx, minlength=bins))
        names.append(s.name)
    return pd.DataFrame(rows, index=names, columns=np.linspace(0, 1, bins, endpoint=False))


REAPPEAR_BINS = [0, 5, 30, 150, np.inf]
REAPPEAR_LABELS = ["1-5", "6-30", "31-150", ">150"]


def reappearance_table(seqs: list[Sequence]) -> pd.DataFrame:
    """Per absence episode: displacement (in box sizes) and scale change between vanishing and reappearing."""
    rows = []
    for s in seqs:
        starts, lengths = runs(s.absent)
        for a, l in zip(starts, lengths):
            b, e = a - 1, a + l
            if b < 0 or e >= s.n_frames or not (s.valid_box[b] and s.valid_box[e]):
                continue
            x0, y0, w0, h0 = s.boxes[b]
            x1, y1, w1, h1 = s.boxes[e]
            rows.append(dict(cls=s.cls, seq=s.name, start=int(a), length=int(l),
                             disp=np.hypot(x1 + w1 / 2 - x0 - w0 / 2, y1 + h1 / 2 - y0 - h0 / 2) / np.sqrt(w0 * h0),
                             scale=(w1 * h1) / (w0 * h0)))
    df = pd.DataFrame(rows, columns=["cls", "seq", "start", "length", "disp", "scale"])
    df["length_bin"] = pd.cut(df.length, REAPPEAR_BINS, labels=REAPPEAR_LABELS)
    return df


def split_attribute_prevalence(attr: pd.DataFrame, test_names: set[str]) -> pd.DataFrame:
    """% of videos with each attribute in train vs test."""
    a = attr.drop(columns="cls").astype(bool)
    is_test = a.index.isin(test_names)
    out = pd.DataFrame({"train": a[~is_test].mean() * 100, "test": a[is_test].mean() * 100})
    out["diff"] = out.test - out.train
    return out.round(1)


def split_summary(seqs_df: pd.DataFrame, test_names: set[str]) -> pd.DataFrame:
    d = seqs_df.assign(split=np.where(seqs_df.seq.isin(test_names), "test", "train"))
    return d.groupby("split").agg(videos=("seq", "size"), mean_frames=("n_frames", "mean"),
                                  frac_absent=("frac_absent", "mean"),
                                  pct_with_absence=("frac_absent", lambda x: 100 * (x > 0).mean()),
                                  foc_episodes=("n_foc_episodes", "mean"), oov_episodes=("n_oov_episodes", "mean"))


def episodes_table(seqs: list[Sequence]) -> pd.DataFrame:
    rows = []
    for s in seqs:
        for kind in ("full_occ", "out_view"):
            st, ln = runs(getattr(s, kind).astype(bool))
            for a, b in zip(st, ln):
                rows.append(dict(cls=s.cls, seq=s.name, kind=kind, start_frame=int(a),
                                 start_norm=a / s.n_frames, length=int(b)))
    return pd.DataFrame(rows, columns=["cls", "seq", "kind", "start_frame", "start_norm", "length"])
