from pathlib import Path
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("cscloud2026_latex/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "ink": "#232323",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "blue": "#2B6CB0",
    "blue_soft": "#D8E7F7",
    "teal": "#2C7A7B",
    "teal_soft": "#D7F0EE",
    "violet": "#6B5CA5",
    "violet_soft": "#E5E1F5",
    "peach": "#C56A3A",
    "peach_soft": "#F5E0D2",
    "green": "#2F855A",
    "green_soft": "#DDEFE5",
    "neutral": "#8A8F98",
    "neutral_soft": "#F3F4F6",
}


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_all(fig, stem):
    base = OUT_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=450, bbox_inches="tight")


def add_box(ax, xy, wh, text, fc, ec, fontsize=7, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=PALETTE["ink"],
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.18,
    )
    return patch


def add_arrow(ax, start, end, color=None, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        color=color or PALETTE["muted"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def make_architecture():
    fig, ax = plt.subplots(figsize=(7.05, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.012, 0.972, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.055,
        0.965,
        "Tree-guided confidence-preserved correction",
        fontsize=8.5,
        fontweight="bold",
        va="top",
        color=PALETTE["ink"],
    )

    add_box(
        ax,
        (0.055, 0.71),
        (0.17, 0.13),
        "npm package\narchive",
        PALETTE["neutral_soft"],
        PALETTE["neutral"],
        weight="bold",
    )
    add_box(
        ax,
        (0.285, 0.69),
        (0.20, 0.17),
        "Static risk\nfeature extraction\nmanifest | scripts\nsource layout",
        PALETTE["blue_soft"],
        PALETTE["blue"],
        fontsize=6.7,
        weight="bold",
    )
    add_arrow(ax, (0.225, 0.775), (0.285, 0.775))

    add_box(
        ax,
        (0.565, 0.77),
        (0.17, 0.13),
        "Tree teacher\nRF / HGB",
        PALETTE["violet_soft"],
        PALETTE["violet"],
        weight="bold",
    )
    add_box(
        ax,
        (0.785, 0.77),
        (0.16, 0.13),
        "Teacher score\nconfidence\nRF-HGB gap",
        PALETTE["violet_soft"],
        PALETTE["violet"],
        fontsize=6.5,
    )
    add_arrow(ax, (0.485, 0.785), (0.565, 0.835))
    add_arrow(ax, (0.735, 0.835), (0.785, 0.835))

    add_box(
        ax,
        (0.555, 0.47),
        (0.18, 0.14),
        "Leaf-path\nembedding\ninteraction regions",
        PALETTE["teal_soft"],
        PALETTE["teal"],
        fontsize=6.7,
        weight="bold",
    )
    add_box(
        ax,
        (0.285, 0.45),
        (0.20, 0.18),
        "Dense feature\nencoder\nstandardized\npackage vector",
        PALETTE["blue_soft"],
        PALETTE["blue"],
        fontsize=6.7,
        weight="bold",
    )
    add_arrow(ax, (0.65, 0.77), (0.65, 0.61))
    add_arrow(ax, (0.385, 0.69), (0.385, 0.63))

    add_box(
        ax,
        (0.36, 0.20),
        (0.31, 0.16),
        "TreeCross correction branch\nfuses dense features,\nteacher priors, and leaf embeddings",
        PALETTE["peach_soft"],
        PALETTE["peach"],
        fontsize=6.8,
        weight="bold",
    )
    add_arrow(ax, (0.385, 0.45), (0.46, 0.36))
    add_arrow(ax, (0.645, 0.47), (0.575, 0.36))
    add_arrow(ax, (0.825, 0.77), (0.64, 0.36), rad=-0.12)

    add_box(
        ax,
        (0.765, 0.19),
        (0.18, 0.17),
        "Confidence gate\npreserve confident\nteacher decisions",
        PALETTE["green_soft"],
        PALETTE["green"],
        fontsize=6.7,
        weight="bold",
    )
    add_arrow(ax, (0.67, 0.28), (0.765, 0.28))
    add_arrow(ax, (0.865, 0.77), (0.865, 0.36))

    add_box(
        ax,
        (0.765, 0.035),
        (0.18, 0.095),
        "Final malicious\npackage score",
        "#FFFFFF",
        PALETTE["green"],
        fontsize=6.8,
        weight="bold",
    )
    add_arrow(ax, (0.855, 0.19), (0.855, 0.13), color=PALETTE["green"])

    save_all(fig, "fig1_tgc_mpd_architecture")
    plt.close(fig)


def make_f1_comparison():
    methods = ["LR", "RF", "HGB", "MLP", "GuardDog", "SAP-XGB", "TGC-MPD"]
    scoped = np.array([0.9113, 0.9707, 0.9724, 0.9444, 0.9291, 0.8799, 0.9668])
    scoped_std = np.array([0.0088, 0.0069, 0.0113, 0.0077, np.nan, np.nan, 0.0106])
    hard = np.array([0.8787, 0.9388, 0.9458, 0.9139, 0.8990, 0.8735, 0.9535])
    hard_std = np.array([0.0202, 0.0141, 0.0203, 0.0229, np.nan, np.nan, 0.0118])

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"width_ratios": [1.65, 1.0]})

    # Panel a: hard-case ranking, which is the main robustness claim.
    order = np.argsort(hard)
    ordered_methods = [methods[i] + (" (ref.)" if methods[i] in {"GuardDog", "SAP-XGB"} else "") for i in order]
    ordered_hard = hard[order]
    ordered_std = hard_std[order]
    y = np.arange(len(ordered_methods))
    colors = [PALETTE["neutral"] for _ in ordered_methods]
    colors[ordered_methods.index("TGC-MPD")] = PALETTE["blue"]
    axes[0].barh(y, ordered_hard, color=colors, edgecolor="white", height=0.58)
    for i, (v, s, m) in enumerate(zip(ordered_hard, ordered_std, ordered_methods)):
        if np.isfinite(s):
            axes[0].errorbar(v, i, xerr=s, fmt="none", ecolor=PALETTE["ink"], elinewidth=0.8, capsize=2.2)
        label_color = PALETTE["blue"] if m == "TGC-MPD" else PALETTE["ink"]
        axes[0].text(v + 0.0025, i + 0.28, f"{v:.3f}", va="bottom", ha="left", fontsize=6.5, color=label_color)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ordered_methods, fontsize=6.8)
    axes[0].set_xlim(0.84, 0.970)
    axes[0].set_xlabel("F1 score", fontsize=7)
    axes[0].set_title("Hard-case detection", fontsize=8, fontweight="bold", pad=6)
    axes[0].grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="both", labelsize=6.5, length=0)
    axes[0].spines["left"].set_visible(False)
    axes[0].spines["bottom"].set_color("#AAB0BA")
    axes[0].text(-0.16, 1.05, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold")

    # Panel b: transparent trade-off relative to the strongest HGB baseline.
    settings = ["scoped\nbalanced", "hard\ncase"]
    deltas = np.array([scoped[-1] - scoped[2], hard[-1] - hard[2]])
    delta_colors = [PALETTE["peach"], PALETTE["green"]]
    x = np.arange(len(settings))
    axes[1].axhline(0, color=PALETTE["ink"], linewidth=0.8)
    axes[1].bar(x, deltas, width=0.58, color=delta_colors, edgecolor="white")
    for i, d in enumerate(deltas):
        va = "bottom" if d >= 0 else "top"
        offset = 0.0007 if d >= 0 else -0.0007
        sign = "+" if d >= 0 else ""
        axes[1].text(i, d + offset, f"{sign}{d:.4f}", ha="center", va=va, fontsize=6.8, color=PALETTE["ink"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settings, fontsize=6.8)
    axes[1].set_ylabel("F1 difference vs HGB", fontsize=7)
    axes[1].set_title("F1 difference from HGB", fontsize=8, fontweight="bold", pad=6)
    axes[1].set_ylim(-0.008, 0.010)
    axes[1].grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].tick_params(axis="both", labelsize=6.5, length=0)
    axes[1].spines["left"].set_color("#AAB0BA")
    axes[1].spines["bottom"].set_color("#AAB0BA")
    axes[1].text(-0.18, 1.05, "b", transform=axes[1].transAxes, fontsize=9, fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save_all(fig, "fig2_main_f1_comparison")
    plt.close(fig)


def main():
    apply_style()
    make_architecture()
    make_f1_comparison()
    print("Generated:")
    for name in [
        "fig1_tgc_mpd_architecture",
        "fig2_main_f1_comparison",
    ]:
        print(f"  {OUT_DIR / (name + '.pdf')}")
        print(f"  {OUT_DIR / (name + '.svg')}")
        print(f"  {OUT_DIR / (name + '.png')}")


if __name__ == "__main__":
    main()

