"""Plot helpers (matplotlib only). Every function returns the Figure; pass `save=` to also write a PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C


def _finish(fig, save):
    fig.tight_layout()
    if save:
        p = Path(save)
        if not p.is_absolute():
            p = C.FIG_DIR / p
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
    return fig


def _classes(df):
    return list(dict.fromkeys(df.cls))


# --------------------------------------------------------------------------- size / aspect
def plot_size_aspect(frames: pd.DataFrame, save=None):
    f = frames.dropna(subset=["rel_area"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c in _classes(f):
        d = f[f.cls == c]
        ax[0].hist(np.log10(d.rel_area), bins=60, alpha=0.5, density=True, label=c)
        ax[1].hist(np.log2(d.aspect), bins=60, alpha=0.5, density=True, label=c)
    ax[0].set(xlabel="log10(box area / frame area)", ylabel="density", title="Relative target size")
    ax[1].set(xlabel="log2(w / h)   (0 = square)", title="Aspect ratio")
    ax[1].axvline(0, color="k", lw=0.5)
    per_seq = f.groupby(["cls", "seq"]).rel_area.median().reset_index()
    ax[2].boxplot([np.log10(per_seq[per_seq.cls == c].rel_area) for c in _classes(f)], tick_labels=_classes(f))
    ax[2].set(ylabel="log10(median rel. area)", title="Per-sequence median size")
    ax[0].legend()
    return _finish(fig, save)


# --------------------------------------------------------------------------- absence
def plot_absence(seqs_df: pd.DataFrame, save=None):
    d = seqs_df.sort_values(["cls", "frac_absent"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw=dict(width_ratios=[3, 1]))
    xs = np.arange(len(d))
    ax[0].bar(xs, d.frac_foc, label="full occlusion", color="tab:red")
    ax[0].bar(xs, d.frac_oov, bottom=d.frac_foc, label="out of view", color="tab:blue")
    ax[0].set(xticks=xs, xticklabels=d.seq, ylabel="fraction of frames", title="Absent frames per sequence")
    ax[0].tick_params(axis="x", rotation=90, labelsize=7)
    ax[0].legend()
    cls = _classes(d)
    w = 0.38
    for i, (col, lab, colr) in enumerate([("frac_foc", "full occlusion", "tab:red"), ("frac_oov", "out of view", "tab:blue")]):
        ax[1].bar(np.arange(len(cls)) + (i - .5) * w, [d[d.cls == c][col].mean() for c in cls], w, label=lab, color=colr)
    ax[1].set(xticks=range(len(cls)), xticklabels=cls, title="Mean per class")
    return _finish(fig, save)


# --------------------------------------------------------------------------- length / resolution
def plot_length_resolution(seqs_df: pd.DataFrame, save=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for c in _classes(seqs_df):
        ax[0].hist(seqs_df[seqs_df.cls == c].n_frames, bins=20, alpha=0.5, label=c)
    ax[0].set(xlabel="frames per video", ylabel="# videos", title="Video length")
    ax[0].legend()
    vc = seqs_df.groupby(["cls", "resolution"]).size().unstack(0).fillna(0).astype(int)
    vc.plot.barh(ax=ax[1], stacked=True)
    ax[1].set(xlabel="# videos", title="Resolution")
    return _finish(fig, save)


# --------------------------------------------------------------------------- attributes
def plot_attribute_counts(attr_counts: pd.DataFrame, title="Videos per attribute", save=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    attr_counts.T.plot.bar(ax=ax)
    ax.set(ylabel="# videos", title=title)
    return _finish(fig, save)


def plot_coverage(cov: pd.DataFrame, save=None):
    """Heatmap of test videos per class x attribute; zeros are outlined (blind spots)."""
    fig, ax = plt.subplots(figsize=(1 + 0.6 * cov.shape[1], 1 + 0.5 * cov.shape[0]))
    im = ax.imshow(cov.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set(xticks=range(cov.shape[1]), xticklabels=cov.columns, yticks=range(cov.shape[0]), yticklabels=cov.index,
           title="Test videos per class x attribute (red = none)")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.values[i, j]
            ax.text(j, i, int(v), ha="center", va="center", color="red" if v == 0 else "black")
    fig.colorbar(im, ax=ax)
    return _finish(fig, save)


# --------------------------------------------------------------------------- temporal density
def plot_temporal_density(dens_foc: pd.DataFrame, dens_oov: pd.DataFrame, save=None):
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw=dict(height_ratios=[3, 1]))
    for j, (name, d, cmap) in enumerate([("Full occlusion", dens_foc, "Reds"), ("Out of view", dens_oov, "Blues")]):
        order = d.sum(axis=1).sort_values(ascending=False).index
        ax[0, j].imshow(d.loc[order].values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                        extent=[0, 1, len(d), 0])
        ax[0, j].set(title=f"{name}: per-video timeline", ylabel="video (sorted)", xlabel="normalised time")
        ax[1, j].plot(d.columns, d.mean(axis=0))
        ax[1, j].fill_between(d.columns, d.mean(axis=0), alpha=0.3)
        ax[1, j].set(title=f"{name}: mean density across videos", xlabel="normalised time", ylabel="fraction absent")
    return _finish(fig, save)


def plot_episodes(episodes: pd.DataFrame, save=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for kind, col in [("full_occ", "tab:red"), ("out_view", "tab:blue")]:
        e = episodes[episodes.kind == kind]
        if len(e):
            ax[0].hist(np.log10(e.length), bins=30, alpha=0.5, color=col, label=f"{kind} (n={len(e)})")
            ax[1].hist(e.start_norm, bins=20, alpha=0.5, color=col, label=kind)
    ax[0].set(xlabel="log10(episode length, frames)", title="Absence-episode length")
    ax[1].set(xlabel="normalised start time", title="When do absences start?")
    ax[0].legend()
    return _finish(fig, save)


# --------------------------------------------------------------------------- sanity check
def show_samples(seq, n=4, save=None):
    """Draw GT boxes on n evenly spaced frames (absent frames get a red border)."""
    from PIL import Image
    imgs = sorted((seq.path / "img").glob("*"))
    idx = np.linspace(0, seq.n_frames - 1, n).astype(int)
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 3.5))
    for a, i in zip(np.atleast_1d(ax), idx):
        # LaSOT frames are 00000001.jpg, ...; with only sampled frames on disk, pick by name
        img = seq.path / "img" / f"{i + 1:08d}.jpg"
        a.imshow(Image.open(img if img.exists() else imgs[min(i, len(imgs) - 1)]))
        x, y, w, h = seq.boxes[i]
        if w > 0 and h > 0:
            a.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="lime", lw=2))
        state = "FOC" if seq.full_occ[i] else "OOV" if seq.out_view[i] else ""
        a.set_title(f"{seq.name} #{i} {state}", fontsize=9, color="red" if state else "black")
        a.axis("off")
    return _finish(fig, save)
