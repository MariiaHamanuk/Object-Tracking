"""Occlusion recovery around absence episodes.

recovered: IoU > iou_thr in the `window` frames after the target returns.
locked_before: IoU > iou_thr in one of the last `lock_window` visible frames before the episode
(a window, because the GT box shrinks as the target disappears). ORR uses these episodes, ORR_all all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda.stats import runs

LENGTH_BINS = [0, 5, 30, 150, np.inf]
LENGTH_LABELS = ["1-5", "6-30", "31-150", ">150"]


def episodes(seq_name: str, iou: np.ndarray, full_occ: np.ndarray, out_view: np.ndarray, gt_valid: np.ndarray,
             window: int = 30, iou_thr: float = 0.5, lock_window: int = 10) -> list[dict]:
    absent = (full_occ | out_view).astype(bool)
    starts, lengths = runs(absent)
    n = len(absent)
    rows = []
    for k, (a, l) in enumerate(zip(starts, lengths)):
        before, back = a - 1, a + l
        if before < 0 or back >= n:
            continue  # nothing to recover from
        stop = starts[k + 1] if k + 1 < len(starts) else n
        win = np.arange(back, min(back + window, stop))
        win = win[gt_valid[win]]
        if len(win) == 0:
            continue
        hit = np.flatnonzero(iou[win] > iou_thr)
        pre = np.arange(max(0, a - lock_window), a)
        pre = pre[gt_valid[pre]]
        foc, oov = full_occ[a:back].any(), out_view[a:back].any()
        rows.append(dict(
            seq=seq_name, start=int(a), length=int(l),
            kind="both" if foc and oov else "full_occlusion" if foc else "out_of_view",
            locked_before=bool(len(pre) and (iou[pre] > iou_thr).any()),
            recovered=bool(len(hit)),
            frames_to_recover=int(win[hit[0]] - back) if len(hit) else np.nan,
            iou_after=float(np.clip(iou[win], 0, None).mean()),
        ))
    return rows


def summarize_episodes(ep: pd.DataFrame) -> dict:
    """Headline numbers; ORR is on episodes where the tracker was on target right before."""
    if ep.empty:
        return {}
    locked = ep[ep.locked_before]
    out = dict(
        n_episodes=len(ep), n_locked_before=len(locked),
        ORR=float(locked.recovered.mean() * 100) if len(locked) else np.nan,
        ORR_all=float(ep.recovered.mean() * 100),
        median_frames_to_recover=float(locked.frames_to_recover.median()) if len(locked) else np.nan,
        mean_iou_after=float(locked.iou_after.mean() * 100) if len(locked) else np.nan,
    )
    bins = pd.cut(locked.length, LENGTH_BINS, labels=LENGTH_LABELS)
    by_len = locked.groupby(bins, observed=False).recovered.agg(["size", "mean"])
    out["ORR_by_length"] = {k: dict(n=int(r["size"]), ORR=float(r["mean"] * 100) if r["size"] else None)
                            for k, r in by_len.iterrows()}
    by_kind = locked.groupby("kind").recovered.agg(["size", "mean"])
    out["ORR_by_kind"] = {k: dict(n=int(r["size"]), ORR=float(r["mean"] * 100)) for k, r in by_kind.iterrows()}
    return out
