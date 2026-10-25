"""Architecture figures of the best variants, in the style of the JEPA diagrams.

    python scripts/draw_architectures.py --out docs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ENC, PRED, COST, ARROW, OST, LAYER = "#9b4f96", "#22a33a", "#e8323c", "#3567b0", "#d99a1e", "#3567b0"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "cm"})
COLS = {"before": 4.0, "during": 9.15, "after": 14.3}


class Fig:
    def __init__(self, title: str, subtitle: str):
        self.fig, self.ax = plt.subplots(figsize=(16.5, 9.4))
        ax = self.ax
        ax.set_xlim(0, 17.2)
        ax.set_ylim(-0.45, 9.6)
        ax.axis("off")
        ax.text(0.2, 9.5, title, fontsize=22, weight="bold", va="top")
        ax.text(0.2, 8.95, subtitle, fontsize=13, va="top", color="#444")
        for x in (6.57, 11.73):
            ax.plot([x, x], [0.1, 8.1], ls="--", color="#f39c32", lw=1.6)
        ax.plot([0.15, 17.0], [5.6, 5.6], ls="--", color="#f39c32", lw=1.6)
        for name, x in COLS.items():
            ax.text(x, 8.05, f"{name.upper()} OCCLUSION", ha="center", va="bottom", fontsize=14, weight="bold")
        ax.text(0.2, 6.85, "OSTrack", color=OST, fontsize=15, weight="bold", va="center")
        ax.text(0.2, 3.2, "V-JEPA 2\n(frozen)", color=LAYER, fontsize=15, va="center")

    def enc(self, x, y, label="Enc", w=1.5, h=0.85):
        hs = h * 0.38
        t = np.linspace(0, np.pi, 40)
        top = np.c_[x + w / 2 * np.cos(t), y - h / 2 + hs + (h - hs) * np.sin(t)]
        pts = np.r_[[[x - w / 2, y - h / 2], [x + w / 2, y - h / 2]], top]
        self.ax.add_patch(Polygon(pts, closed=True, fc="white", ec=ENC, lw=2.2, zorder=3))
        self.ax.text(x, y - 0.05, label, ha="center", va="center", fontsize=16, zorder=4)
        return (x, y - h / 2), (x, y + h / 2)

    def pred(self, x, y, label="Pred", w=1.6, h=0.75):
        rx = h / 2
        t = np.linspace(-np.pi / 2, np.pi / 2, 40)
        right = np.c_[x + w / 2 - rx + rx * np.cos(t), y + (h / 2) * np.sin(t)]
        pts = np.r_[[[x - w / 2, y + h / 2], [x - w / 2, y - h / 2]], right]
        self.ax.add_patch(Polygon(pts, closed=True, fc="white", ec=PRED, lw=2.2, zorder=3))
        self.ax.text(x - 0.1, y, label, ha="center", va="center", fontsize=16, zorder=4)
        return (x - w / 2, y), (x + w / 2, y)

    def box(self, x, y, text, w=2.3, h=0.7, fs=12.5, bold=False):
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0", fc="#fde4e4",
                                         ec=COST, lw=2.2, zorder=3))
        self.ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4, weight="bold" if bold else None)

    def ost(self, x, y, text, w=3.9, h=0.95):
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                         fc="#fff3d6", ec=OST, lw=2.2, zorder=3))
        self.ax.text(x, y, text, ha="center", va="center", fontsize=12.5, zorder=4)

    def circ(self, x, y, label, latent=False, r=0.3, dashed=False):
        self.ax.add_patch(Circle((x, y), r, fc="white" if latent else "#cfcfcf", ec=COST if latent else "black",
                                 lw=2.2, ls="--" if dashed else "-", zorder=3))
        self.ax.text(x, y, label, ha="center", va="center", fontsize=15, zorder=4)

    def arrow(self, a, b, rad=0.0, color=ARROW, ls="-", lw=1.9, label=None, at=None, ha="left"):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color, ls=ls,
                                          connectionstyle=f"arc3,rad={rad}", zorder=2, shrinkA=2, shrinkB=2))
        if label:
            self.ax.text(*at, label, fontsize=13, color=color, va="center", ha=ha)

    def note(self, text):
        self.ax.text(8.6, -0.4, text, ha="center", va="bottom", fontsize=11.5, color="#555", style="italic")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
        self.fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(self.fig)


def encode_column(f: Fig, col: str, frame: str, s_label: str, label_x: float = 0.12):
    x = COLS[col]
    f.circ(x, 0.5, frame)
    bot, top = f.enc(x, 1.75)
    f.arrow((x, 0.8), bot, color=ENC, lw=3)
    f.ax.text(x + label_x, top[1] + 0.28, s_label, fontsize=15, va="center")
    return top


def past_to_pred(f: Fig, x: float, p_in):
    """The predictor reads the earlier time slots of the same clip, encoded by Enc."""
    px, py = p_in[0] + 0.7, p_in[1] - 0.38
    f.arrow((x - 0.6, 2.1), (px, py), rad=-0.2, label="past slots", at=(x - 1.45, 2.2), ha="right")


def no_predictor(out: Path):
    f = Fig("Best without the predictor",
            "Relocation verified by OSTrack + memory guard + drift check + motion prior  "
            "(AUC 57.47, ORR_all 45.58 on dev)")
    b, d, a = COLS["before"], COLS["during"], COLS["after"]
    sb = encode_column(f, "before", "$x_{t_0}$", "$s[t_0]$")
    sd = encode_column(f, "during", "$x_{t_1}$", "$s[t_1]$")
    sa = encode_column(f, "after", "$x_{t_2}$", "$s[t_2]$")

    f.ost(b, 6.85, "tracks the target\nscore high")
    f.box(b, 3.35, "Guard: box still like\nthe first appearance", w=2.6, h=0.85)
    f.arrow(sb, (b, 2.92))
    f.arrow((b, 6.37), (b, 3.78), ls="--", label="box", at=(b + 0.1, 5.1))
    f.circ(b - 1.65, 4.7, "$m$", latent=True)
    f.arrow((b - 0.75, 3.78), (b - 1.45, 4.45), label="update", at=(b - 1.25, 4.0), ha="right")
    f.circ(b + 1.65, 4.7, "$v$", latent=True)
    f.arrow((b + 0.75, 3.78), (b + 1.45, 4.45), label="track", at=(b + 1.3, 4.0), ha="left")

    f.ost(d, 6.85, "doubts (score drops) or\nconfidently on a distractor")
    f.circ(d - 1.75, 4.7, "$m$", latent=True, dashed=True)
    f.box(d, 3.35, "Drift check: target clearly\nelsewhere, gap > 0.15", w=2.9, h=0.85)
    f.arrow(sd, (d, 2.92))
    f.arrow((d - 1.55, 4.45), (d - 0.9, 3.78), color=COST)
    f.arrow((d, 6.37), (d, 3.78), ls="--", label="score", at=(d + 0.1, 5.1))

    f.ost(a, 6.85, "verifies each candidate\nscore $\\geq$ 0.6 $\\rightarrow$ relocate")
    f.circ(a - 1.75, 4.7, "$m$", latent=True, dashed=True)
    f.circ(a + 1.75, 4.7, "$v$", latent=True, dashed=True)
    f.box(a, 3.35, "$\\cos(m, s)$ over the frame\n$-\\lambda\\cdot$dist to the path", w=2.9, h=0.85)
    f.arrow(sa, (a, 2.92))
    f.arrow((a - 1.55, 4.45), (a - 0.9, 3.78), color=COST)
    f.arrow((a + 1.55, 4.45), (a + 0.9, 3.78), color=COST)
    f.box(a, 4.7, "3 candidates", w=1.9, h=0.55)
    f.arrow((a, 3.78), (a, 4.42))
    f.arrow((a, 4.98), (a, 6.37), label="verify", at=(a + 0.1, 5.85))
    f.note("Dashed circles: state carried from earlier frames ($m$ target memory, $v$ target track). "
           "Encoder only, nothing trained.")
    f.save(out / "arch_best_no_predictor")


def warning_v1(out: Path):
    f = Fig("Best ORR_all, with the predictor: occlusion warning v1",
            "Predictor surprise + score drop freeze the memory and relax relocation  (AUC 57.00, ORR_all 45.82 on dev)")
    b, d, a = COLS["before"], COLS["during"], COLS["after"]
    sb = encode_column(f, "before", "$x_{t_0}$", "$s[t_0]$", label_x=-0.95)
    sd = encode_column(f, "during", "$x_{t_1}$", "$s[t_1]$", label_x=-0.15)
    sa = encode_column(f, "after", "$x_{t_2}$", "$s[t_2]$")

    f.ost(b, 6.85, "tracks the target\nscore high")
    f.circ(b - 1.5, 4.7, "$m$", latent=True)
    f.arrow((b - 0.2, 2.2), (b - 1.35, 4.42), rad=0.15, label="update", at=(b - 1.05, 3.45), ha="right")
    p_in, p_out = f.pred(b + 1.0, 3.35)
    f.arrow((b + 0.3, 2.2), p_in, rad=-0.35)
    f.box(b + 1.0, 4.7, "surprise low", w=1.8, h=0.55)
    f.arrow((b + 1.0, 3.73), (b + 1.0, 4.42))

    f.ost(d, 6.85, "loses the target\nscore $\\downarrow$")
    p_in, p_out = f.pred(d - 1.25, 3.35)
    past_to_pred(f, d, p_in)
    f.box(d + 1.1, 3.35, "$D(\\hat{s}, s)$", w=1.25, h=0.6)
    f.arrow(p_out, (d + 0.47, 3.35))
    f.arrow(sd, (d + 1.1, 3.05), rad=0.25)
    f.box(d, 4.7, "WARNING", w=1.8, h=0.55, bold=True)
    f.arrow((d + 1.1, 3.65), (d + 0.5, 4.42), label="surprise $\\uparrow$", at=(d + 1.0, 4.1))
    f.arrow((d, 6.37), (d, 4.98), label="score $\\downarrow$", at=(d + 0.1, 5.85))
    f.box(d - 1.55, 5.25, "freeze $m$", w=1.4, h=0.45, fs=12)
    f.box(d + 1.5, 5.25, "relax thresholds", w=1.75, h=0.45, fs=12)
    f.arrow((d - 0.9, 4.75), (d - 1.3, 5.02), color=COST)
    f.arrow((d + 0.9, 4.75), (d + 1.3, 5.02), color=COST)

    f.ost(a, 6.85, "verifies each candidate\nscore $\\geq$ 0.6 $\\rightarrow$ relocate")
    f.circ(a - 1.75, 4.7, "$m$", latent=True, dashed=True)
    f.box(a, 3.35, "$\\cos(m, s)$ over the frame", w=2.7, h=0.6)
    f.arrow(sa, (a, 3.05))
    f.arrow((a - 1.55, 4.45), (a - 0.8, 3.66), color=COST)
    f.box(a, 4.7, "3 candidates", w=1.9, h=0.55)
    f.arrow((a, 3.66), (a, 4.42))
    f.arrow((a, 4.98), (a, 6.37), label="verify", at=(a + 0.1, 5.85))
    f.note("Pred predicts the current moment from the past frames; $D$ = 1 - cos(predicted, real) at the target. "
           "Dashed circle: frozen memory.")
    f.save(out / "arch_warning_v1")


def final_predictor(out: Path):
    f = Fig("Best with the predictor: final system without the memory guard",
            "Predictor warning + drift check + motion prior + future candidates  (AUC 57.71, ORR_all 45.35 on dev)")
    b, d, a = COLS["before"], COLS["during"], COLS["after"]
    sb = encode_column(f, "before", "$x_{t_0}$", "$s[t_0]$", label_x=-1.05)
    sd = encode_column(f, "during", "$x_{t_1}$", "$s[t_1]$", label_x=-0.15)
    sa = encode_column(f, "after", "$x_{t_2}$", "$s[t_2]$", label_x=-0.95)

    f.ost(b, 6.85, "tracks the target\nscore high")
    f.circ(b - 1.6, 4.7, "$m$", latent=True)
    f.arrow((b - 0.25, 2.2), (b - 1.42, 4.42), rad=0.15, label="update", at=(b - 1.15, 3.45), ha="right")
    f.circ(b + 1.6, 4.7, "$v$", latent=True)
    f.arrow((b + 0.25, 2.2), (b + 1.42, 4.42), rad=-0.15, label="track", at=(b + 1.15, 3.45), ha="left")

    f.ost(d, 6.85, "doubts, or confidently\non a distractor (drift check)")
    p_in, p_out = f.pred(d - 1.25, 3.0)
    past_to_pred(f, d, p_in)
    f.box(d + 1.1, 3.0, "$D(\\hat{s}, s)$", w=1.25, h=0.6)
    f.arrow(p_out, (d + 0.47, 3.0))
    f.arrow(sd, (d + 1.1, 2.7), rad=0.25)
    f.box(d, 4.35, "WARNING: top 5% of\nsurprise + score drop", w=2.7, h=0.75, fs=12)
    f.arrow((d + 1.1, 3.3), (d + 0.7, 3.97))
    f.arrow((d, 6.37), (d, 4.73), label="score", at=(d + 0.1, 5.85))
    f.box(d - 1.75, 5.2, "freeze $m$", w=1.4, h=0.45, fs=12)
    f.arrow((d - 1.2, 4.73), (d - 1.5, 4.97), color=COST)

    f.ost(a, 6.85, "verifies each candidate\nscore $\\geq$ 0.6 $\\rightarrow$ relocate")
    p_in, p_out = f.pred(a + 1.95, 1.75)
    f.arrow((a + 0.75, 1.75), p_in)
    f.ax.text(a + 1.95, 2.3, "$\\hat{s}[t_2{+}4]$", fontsize=14, ha="center")
    f.circ(a - 2.0, 4.7, "$m$", latent=True, dashed=True)
    f.circ(a + 2.0, 4.7, "$v$", latent=True, dashed=True)
    f.box(a - 0.1, 3.45, "$\\cos(m, s)$, $\\cos(m, \\hat{s})$\n$-\\lambda\\cdot$dist to the path", w=3.0, h=0.85,
          fs=12)
    f.arrow(sa, (a - 0.1, 3.02))
    f.arrow((a + 1.6, 2.55), (a + 0.95, 3.02), label="future", at=(a + 1.8, 2.75), ha="left")
    f.arrow((a - 1.8, 4.45), (a - 1.1, 3.88), color=COST)
    f.arrow((a + 1.8, 4.45), (a + 1.0, 3.88), color=COST)
    f.box(a - 0.1, 4.7, "3 + 2 candidates", w=2.1, h=0.55)
    f.arrow((a - 0.1, 3.88), (a - 0.1, 4.42))
    f.arrow((a - 0.1, 4.98), (a - 0.1, 6.37), label="verify", at=(a, 5.85))
    f.note("Encoder: target memory $m$ and full-frame search. Predictor: occlusion warning (during) "
           "and the predicted future map $\\hat{s}$ (after).")
    f.save(out / "arch_final_predictor")


def grid_icon(ax, x, y, size=0.7, n=6):
    """A small n x n grid standing for the 16x16 feature map."""
    for k in range(n + 1):
        t = k * size / n
        ax.plot([x, x + size], [y + t, y + t], color=ENC, lw=0.9, zorder=3)
        ax.plot([x + t, x + t], [y, y + size], color=ENC, lw=0.9, zorder=3)


def vjepa_overview(out: Path):
    f = Fig.__new__(Fig)
    f.fig, f.ax = plt.subplots(figsize=(15, 6.2))
    ax = f.ax
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.1, 6.0)
    ax.axis("off")

    # context: past frames -> Enc -> Pred
    f.circ(1.6, 0.55, "$x_{<t}$", r=0.42)
    ax.text(1.6, -0.05, "past frames", ha="center", va="top", fontsize=15)
    bot, top = f.enc(1.6, 2.0, w=1.7, h=0.95)
    f.arrow((1.6, 0.97), bot, color=ENC, lw=3)
    p_in, p_out = f.pred(4.9, 4.4, w=1.9, h=0.9)
    f.arrow(top, p_in, rad=-0.3, label="$s[<t]$", at=(1.75, 3.75), ha="left")
    ax.text(7.0, 4.25, "$\\hat{s}[t]$: predicted now", fontsize=15, ha="center", va="top")

    # now: 16 frames -> Enc -> s[t]
    f.circ(9.0, 0.55, "$x_t$", r=0.42)
    ax.text(9.0, -0.05, "16 frames (~1 s)", ha="center", va="top", fontsize=15)
    bot, top = f.enc(9.0, 2.0, w=1.7, h=0.95)
    f.arrow((9.0, 0.97), bot, color=ENC, lw=3)
    grid_icon(ax, 9.6, 2.2)
    ax.text(10.45, 2.55, "$s[t]$: 16$\\times$16 map\nof the whole frame", fontsize=14, va="center")

    # predicted vs real -> surprise
    f.box(9.0, 4.4, "$D(\\hat{s}, s)$", w=1.6, h=0.75, fs=16)
    f.arrow(p_out, (8.2, 4.4))
    f.arrow(top, (9.0, 4.02), label="real now", at=(8.1, 3.3), ha="right")
    f.box(9.0, 5.55, "surprise = 1 $-$ cos", w=2.6, h=0.6, fs=14)
    f.arrow((9.0, 4.78), (9.0, 5.25))

    # s[t] vs target memory -> where the target is
    f.circ(13.9, 2.0, "$m$", latent=True, r=0.38)
    ax.text(13.9, 1.35, "target\nmemory", ha="center", va="top", fontsize=13)
    f.box(12.0, 4.4, "cos$(m, s)$\nover the frame", w=2.3, h=0.9, fs=14)
    f.arrow((9.03, 3.35), (10.83, 4.25), rad=-0.15)
    f.arrow((13.75, 2.38), (12.6, 3.95), color=COST)
    f.box(12.0, 5.55, "target found anywhere", w=2.9, h=0.6, fs=14)
    f.arrow((12.0, 4.85), (12.0, 5.25))

    ax.text(0.15, 5.85, "V-JEPA 2 (frozen)", color=LAYER, fontsize=17, va="top")
    f.save(out / "vjepa_overview")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    a = p.parse_args()
    no_predictor(a.out)
    warning_v1(a.out)
    final_predictor(a.out)
    vjepa_overview(a.out)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
