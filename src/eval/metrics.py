"""LaSOT metrics with the pytracking protocol (absent frames count as failures)."""
from __future__ import annotations

import numpy as np

OVERLAP_THR = np.arange(0.0, 1.05, 0.05)
CENTER_THR = np.arange(0, 51, dtype=float)      # Precision is reported at 20 px
CENTER_NORM_THR = np.arange(0, 51) / 100.0      # Normalized Precision at 0.20


def iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    tl = np.maximum(pred[:, :2], gt[:, :2])
    br = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0, gt[:, :2] + gt[:, 2:] - 1.0)
    inter = np.clip(br - tl + 1.0, 0, None).prod(1)
    union = pred[:, 2:].prod(1) + gt[:, 2:].prod(1) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def center_error(pred: np.ndarray, gt: np.ndarray, normalized: bool = False) -> np.ndarray:
    pc = pred[:, :2] + 0.5 * (pred[:, 2:] - 1.0)
    gc = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    if normalized:
        with np.errstate(divide="ignore", invalid="ignore"):
            pc, gc = pc / gt[:, 2:], gc / gt[:, 2:]
    return np.sqrt(((pc - gc) ** 2).sum(1))


def clean_prediction(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Same fixes as pytracking: trim extra frames, hold the last box on empty predictions, frame 0 = GT."""
    pred = np.array(pred[: len(gt)], dtype=float)
    if len(pred) < len(gt):
        raise ValueError(f"prediction has {len(pred)} frames, ground truth {len(gt)}")
    for i in range(1, len(pred)):
        if (pred[i, 2:] == 0).any():
            pred[i] = pred[i - 1]
    pred[0] = gt[0]
    return pred


def sequence_errors(pred: np.ndarray, gt: np.ndarray, visible: np.ndarray) -> dict:
    """Per-frame IoU / centre errors with LaSOT's handling of absent and unannotated frames."""
    pred = clean_prediction(pred, gt)
    valid = (gt[:, 2:] > 0).all(1) & visible
    with np.errstate(invalid="ignore"):
        ov = np.where(valid, iou(pred, gt), -1.0)
        ce = np.where(valid, center_error(pred, gt), np.inf)
        cn = np.where(valid, center_error(pred, gt, normalized=True), -1.0)
    cn[~visible] = np.inf
    return dict(iou=ov, center=ce, center_norm=cn, valid=valid, pred=pred)


def sequence_curves(err: dict) -> dict:
    n = len(err["iou"])
    return dict(
        success=(err["iou"][:, None] > OVERLAP_THR[None]).sum(0) / n,
        precision=(err["center"][:, None] <= CENTER_THR[None]).sum(0) / n,
        norm_precision=(err["center_norm"][:, None] <= CENTER_NORM_THR[None]).sum(0) / n,
    )


def summarize(curves: list[dict]) -> dict:
    """Average per-sequence curves -> the three headline numbers (in %)."""
    if not curves:
        return dict(n_sequences=0)
    s = np.mean([c["success"] for c in curves], 0) * 100
    p = np.mean([c["precision"] for c in curves], 0) * 100
    np_ = np.mean([c["norm_precision"] for c in curves], 0) * 100
    return dict(AUC=float(s.mean()), OP50=float(s[10]), OP75=float(s[15]),
                Precision=float(p[20]), NormPrecision=float(np_[20]), n_sequences=len(curves))
