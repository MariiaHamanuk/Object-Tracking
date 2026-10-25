"""EDA plots; every function returns the figure and saves it with `save=`."""
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


MAX_OVERLAID = 8  # more classes than this are pooled instead of overlaid


def _classes(df):
    return list(dict.fromkeys(df.cls))


def _class_ticks(ax, labels, axis="x"):
    getattr(ax, f"set_{axis}ticks")(range(len(labels)))
    getattr(ax, f"set_{axis}ticklabels")(labels, fontsize=7 if len(labels) > 20 else 9,
                                         rotation=90 if axis == "x" and len(labels) > 8 else 0)


def plot_size_aspect(frames: pd.DataFrame, save=None):
    f = frames.dropna(subset=["rel_area"])
    cls = _classes(f)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw=dict(width_ratios=[1, 1, 2]))
    groups = [(c, f[f.cls == c]) for c in cls] if len(cls) <= MAX_OVERLAID else [("all classes", f)]
    for c, d in groups:
        ax[0].hist(np.log10(d.rel_area), bins=60, alpha=0.5, density=True, label=c)
        ax[1].hist(np.log2(d.aspect), bins=60, alpha=0.5, density=True, label=c)
    ax[0].set(xlabel="log10(box area / frame area)", ylabel="density", title="Relative target size")
    ax[1].set(xlabel="log2(w / h)   (0 = square)", title="Aspect ratio")
    ax[1].axvline(0, color="k", lw=0.5)
    per_seq = f.groupby(["cls", "seq"]).rel_area.median().reset_index()
    order = per_seq.groupby("cls").rel_area.median().sort_values().index.tolist()
    ax[2].boxplot([np.log10(per_seq[per_seq.cls == c].rel_area) for c in order])
    _class_ticks(ax[2], order)
    ax[2].set(ylabel="log10(median rel. area)", title="Per-video median size, classes sorted")
    ax[0].legend(fontsize=8)
    return _finish(fig, save)


def plot_absence(seqs_df: pd.DataFrame, save=None):
    many = len(seqs_df) > 60
    d = seqs_df.sort_values("frac_absent" if many else ["cls", "frac_absent"], ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw=dict(width_ratios=[1, 2]))
    xs = np.arange(len(d))
    ax[0].bar(xs, d.frac_foc, width=1.0, label="full occlusion", color="tab:red")
    ax[0].bar(xs, d.frac_oov, width=1.0, bottom=d.frac_foc, label="out of view", color="tab:blue")
    if many:
        ax[0].set(xlabel=f"video, sorted ({(d.frac_absent > 0).mean():.0%} have absent frames)")
    else:
        ax[0].set(xticks=xs, xticklabels=d.seq)
        ax[0].tick_params(axis="x", rotation=90, labelsize=7)
    ax[0].set(ylabel="fraction of frames", title="Absent frames per video")
    ax[0].legend()
    per_cls = d.groupby("cls")[["frac_foc", "frac_oov"]].mean()
    per_cls = per_cls.loc[per_cls.sum(1).sort_values(ascending=False).index]
    ax[1].bar(range(len(per_cls)), per_cls.frac_foc, color="tab:red")
    ax[1].bar(range(len(per_cls)), per_cls.frac_oov, bottom=per_cls.frac_foc, color="tab:blue")
    _class_ticks(ax[1], per_cls.index.tolist())
    ax[1].set(title="Mean per class, sorted")
    return _finish(fig, save)


def plot_length_resolution(seqs_df: pd.DataFrame, save=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cls = _classes(seqs_df)
    groups = [(c, seqs_df[seqs_df.cls == c]) for c in cls] if len(cls) <= MAX_OVERLAID else [("all", seqs_df)]
    for c, d in groups:
        ax[0].hist(d.n_frames, bins=40, alpha=0.5, label=c)
    ax[0].set(xlabel="frames per video", ylabel="# videos", title="Video length")
    ax[0].legend(fontsize=8)
    seqs_df.resolution.value_counts().head(12).sort_values().plot.barh(ax=ax[1])
    ax[1].set(xlabel="# videos", title=f"Resolution (top 12 of {seqs_df.resolution.nunique()})")
    return _finish(fig, save)


def plot_attribute_counts(attr_counts: pd.DataFrame, title="Videos per attribute", save=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(attr_counts) <= MAX_OVERLAID:
        attr_counts.T.plot.bar(ax=ax)
    else:
        attr_counts.sum().sort_values(ascending=False).plot.bar(ax=ax)
        title += f" (all {len(attr_counts)} classes)"
    ax.set(ylabel="# videos", title=title)
    return _finish(fig, save)


def plot_split_attributes(prev: pd.DataFrame, save=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    prev[["train", "test"]].plot.bar(ax=ax)
    ax.set(ylabel="% of videos", title="Attribute prevalence: train vs test")
    return _finish(fig, save)


def plot_reappearance(re: pd.DataFrame, save=None):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(np.log10(re.length), bins=40, color="gray")
    ax[0].set(xlabel="log10(episode length, frames)", ylabel="# episodes", title="Absence length")
    b = re.groupby("length_bin", observed=True).disp.median()
    ax[1].bar(b.index.astype(str), b.values)
    ax[1].set(xlabel="episode length (frames)", ylabel="median displacement (box sizes)",
              title="How far from where it vanished")
    ax[2].hist(np.log2(re.scale.clip(1 / 16, 16)), bins=40, color="tab:purple")
    ax[2].axvline(-1, color="k", ls="--", lw=0.8)
    ax[2].axvline(1, color="k", ls="--", lw=0.8)
    ax[2].set(xlabel="log2(area after / area before)", title="Size change across the episode")
    return _finish(fig, save)


def plot_coverage(cov: pd.DataFrame, save=None):
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


def show_samples(seq, n=4, save=None):
    from PIL import Image
    imgs = sorted((seq.path / "img").glob("*"))
    idx = np.linspace(0, seq.n_frames - 1, n).astype(int)
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 3.5))
    for a, i in zip(np.atleast_1d(ax), idx):
        img = seq.path / "img" / f"{i + 1:08d}.jpg"
        a.imshow(Image.open(img if img.exists() else imgs[min(i, len(imgs) - 1)]))
        x, y, w, h = seq.boxes[i]
        if w > 0 and h > 0:
            a.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="lime", lw=2))
        state = "FOC" if seq.full_occ[i] else "OOV" if seq.out_view[i] else ""
        a.set_title(f"{seq.name} #{i} {state}", fontsize=9, color="red" if state else "black")
        a.axis("off")
    return _finish(fig, save)
