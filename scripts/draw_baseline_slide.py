"""Baseline slide charts, one PDF (+ PNG preview) each, in the red slide palette.

    python scripts/draw_baseline_slide.py --out docs/figures/baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
BINS = ["1-5", "6-30", "31-150", ">150"]
RED = ["#7d1616", "#a52c2c", "#c87777", "#e5c4c4", "#f4efef"]  # dark -> light
EDGE, TEXT_DARK, GRID = "#7a6a6a", "#2b2b2b", "#e9e1e1"
FONT = next((f for f in ("Fira Sans", "Segoe UI", "Arial") if any(f == x.name for x in font_manager.fontManager.ttflist)),
            "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 13, "axes.edgecolor": EDGE, "axes.labelcolor": TEXT_DARK,
                     "xtick.color": TEXT_DARK, "ytick.color": TEXT_DARK, "pdf.fonttype": 42})


def ink(color: str) -> str:
    return "white" if color in RED[:3] else TEXT_DARK


def metrics(name):
    return json.loads((ROOT / "results" / name / "metrics.json").read_text())


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out / f"{name}.pdf")


def reproduction(out: Path):
    off, ours = metrics("ostrack_official"), metrics("ostrack_ours")
    keys = ("AUC", "Precision", "NormPrecision")
    rows = [["", "AUC", "P", "NP"],
            ["Official", *(f"{off['overall'][k]:.2f}" for k in keys)],
            ["Ours", *(f"{ours['overall'][k]:.2f}" for k in keys)]]
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.axis("off")
    t = ax.table(cellText=rows, cellLoc="center", loc="center", bbox=[0, 0, 1, 1], colWidths=[0.31, 0.23, 0.23, 0.23])
    t.auto_set_font_size(False)
    t.set_fontsize(16)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.8)
        color = RED[0] if r == 0 else RED[3] if r == 2 else RED[4]
        if r > 0 and c == 0:
            color = RED[2]
        cell.set_facecolor(color)
        cell.get_text().set_color(ink(color))
        if r == 0 or c == 0:
            cell.get_text().set_weight("bold")
    save(fig, out, "1_reproduction")


def reappearance(out: Path):
    reap = pd.read_csv(ROOT / "results" / "eda" / "reappearance.csv")
    g = reap.groupby("length_bin", observed=True).disp
    med = g.median().reindex(BINS)
    outside = g.apply(lambda d: (d > 2).mean() * 100).reindex(BINS)
    colors = [RED[3], RED[2], RED[1], RED[0]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(BINS, med, color=colors, edgecolor=EDGE, lw=0.8, width=0.7)
    ax.axhline(2, color=TEXT_DARK, ls="--", lw=1.8)
    ax.text(-0.4, 2.05, "search window reaches ~2 box sizes", ha="left", va="bottom", fontsize=12, color=TEXT_DARK)
    for bar, o, c in zip(bars, outside, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{o:.0f}%\noutside", ha="center",
                va="center", fontsize=13, color=ink(c), weight="bold")
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("occlusion length (frames)")
    ax.set_ylabel("median distance from where it vanished\n(box sizes)")
    style(ax)
    save(fig, out, "2_reappearance")


def recovery(out: Path):
    by = metrics("ostrack_ours")["occlusion"]["ORR_by_length"]
    colors = [RED[0], RED[1], RED[2], RED[3]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(BINS, [by[b]["ORR"] for b in BINS], color=colors, edgecolor=EDGE, lw=0.8, width=0.7)
    for bar, b, c in zip(bars, BINS, colors):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, bar.get_height() + 1.5, f"{by[b]['ORR']:.0f}%", ha="center", fontsize=15, weight="bold",
                color=TEXT_DARK)
        ax.text(x, 4, f"n = {by[b]['n']}", ha="center", fontsize=11, color=ink(c))
    ax.set_ylim(0, 100)
    ax.set_xlabel("occlusion length (frames)")
    ax.set_ylabel("recovered within 30 frames (%)")
    style(ax)
    save(fig, out, "3_recovery_by_length")


def outcomes(out: Path):
    eps = pd.read_csv(ROOT / "results" / "ostrack_ours_dev" / "episodes.csv")
    n = len(eps)
    parts = [("on target before\nrecovered", (eps.locked_before & eps.recovered).sum(), RED[4]),
             ("on target before\nnot recovered", (eps.locked_before & ~eps.recovered).sum(), RED[3]),
             ("lost before\nrecovered anyway", (~eps.locked_before & eps.recovered).sum(), RED[2]),
             ("lost before\nnot recovered", (~eps.locked_before & ~eps.recovered).sum(), RED[0])]
    fig, ax = plt.subplots(figsize=(14, 2.6))
    left = 0
    for label, k, c in parts:
        w = 100 * k / n
        ax.barh(0, w, left=left, color=c, height=0.6, edgecolor=EDGE, lw=0.8)
        ax.text(left + w / 2, 0, f"{w:.0f}%\n{label}", ha="center", va="center", fontsize=13 if w > 15 else 11,
                color=ink(c))
        left += w
    lost = 100 * (~eps.locked_before).sum() / n
    ax.annotate("", xy=(100, 0.42), xytext=(100 - lost, 0.42), arrowprops=dict(arrowstyle="<->", lw=1.5,
                                                                                  color=TEXT_DARK))
    ax.text(100 - lost / 2, 0.5, f"{lost:.0f}% of occlusions are lost before they even start", ha="center",
            va="bottom", fontsize=15, weight="bold", color=RED[0])
    ax.text(0, 0.5, f"{n} occlusions, OSTrack on dev", ha="left", va="bottom", fontsize=12, color=TEXT_DARK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.35, 0.85)
    ax.axis("off")
    save(fig, out, "4_occlusion_outcomes")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "baseline")
    a = p.parse_args()
    print("font:", FONT)
    reproduction(a.out)
    reappearance(a.out)
    recovery(a.out)
    outcomes(a.out)


if __name__ == "__main__":
    main()
