"""
Visualisation functions for the cross-modal XAI pipeline.

Figures generated:
  1. Qualitative Attribution Comparison (4-panel)
  2. Consistency Heatmap (structures x methods, AOS)
  3. Per-Structure Consistency Boxplot (SSIM)
  4. Attribution Centroid Displacement Map
  5. Failure Case Analysis (correlation scatter)
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_MAP, NUM_CLASSES


# ---------------------------------------------------------------------------
# Figure 1: Qualitative Attribution Comparison
# ---------------------------------------------------------------------------

def plot_attribution_comparison(
    mri_image: np.ndarray,
    ct_image: np.ndarray,
    mri_attr: np.ndarray,
    ct_attr: np.ndarray,
    structure_name: str,
    method: str,
    save_path: Optional[Path] = None,
    alpha: float = 0.5,
):
    """
    4-panel figure: MRI | MRI+attr | CT | CT+attr

    Args:
        mri_image: (H, W) greyscale float [0,1]
        ct_image:  (H, W) greyscale float [0,1]
        mri_attr:  (H, W) attribution map [0,1]
        ct_attr:   (H, W) attribution map [0,1]
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Attribution Comparison — {structure_name} ({method})", fontsize=13)

    cmap_attr = "hot"

    axes[0].imshow(mri_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("MRI")

    axes[1].imshow(mri_image, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(mri_attr, cmap=cmap_attr, alpha=alpha, vmin=0, vmax=1)
    axes[1].set_title("MRI + Attribution")

    axes[2].imshow(ct_image, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("CT")

    axes[3].imshow(ct_image, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(ct_attr, cmap=cmap_attr, alpha=alpha, vmin=0, vmax=1)
    axes[3].set_title("CT + Attribution")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Consistency Heatmap
# ---------------------------------------------------------------------------

def plot_consistency_heatmap(
    scores: dict,   # {method: {structure_name: aos_score}}
    metric: str = "AOS",
    save_path: Optional[Path] = None,
):
    """
    Heatmap: rows = structures, columns = methods, values = metric score.
    Green = consistent, Red = inconsistent.
    """
    methods = list(scores.keys())
    structures = [LABEL_MAP[i] for i in range(1, NUM_CLASSES)]

    matrix = np.zeros((len(structures), len(methods)))
    for j, method in enumerate(methods):
        for i, struct in enumerate(structures):
            matrix[i, j] = scores[method].get(struct, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 2), max(5, len(structures))))
    sns.heatmap(
        matrix,
        xticklabels=methods,
        yticklabels=structures,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Cross-Modal Explanation Consistency ({metric})")
    ax.set_xlabel("Attribution Method")
    ax.set_ylabel("Cardiac Structure")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Per-Structure SSIM Boxplot
# ---------------------------------------------------------------------------

def plot_ssim_boxplot(
    ssim_data: dict,   # {structure_name: {method: [ssim_values]}}
    save_path: Optional[Path] = None,
):
    """
    Grouped boxplot: one group per structure, one box per method.
    """
    structures = list(ssim_data.keys())
    methods = list(next(iter(ssim_data.values())).keys())
    palette = sns.color_palette("Set2", n_colors=len(methods))

    all_rows = []
    for struct in structures:
        for method in methods:
            for v in ssim_data[struct][method]:
                all_rows.append({"Structure": struct, "Method": method, "SSIM": v})

    import pandas as pd
    df = pd.DataFrame(all_rows)

    fig, ax = plt.subplots(figsize=(max(8, len(structures) * 1.5), 5))
    sns.boxplot(
        data=df, x="Structure", y="SSIM", hue="Method",
        palette=palette, ax=ax,
    )
    ax.set_title("Per-Structure SSIM Consistency Distribution")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="threshold=0.5")
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Attribution Centroid Displacement Map
# ---------------------------------------------------------------------------

def plot_centroid_map(
    mean_image: np.ndarray,
    centroids: dict,   # {structure_name: {"mri": (cy, cx), "ct": (cy, cx)}}
    save_path: Optional[Path] = None,
):
    """
    Plot MRI centroids (circles) and CT centroids (crosses) on mean anatomy.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(mean_image, cmap="gray")

    colors = plt.cm.tab10(np.linspace(0, 1, len(centroids)))
    patches = []
    for (name, cent), color in zip(centroids.items(), colors):
        cy_mri, cx_mri = cent["mri"]
        cy_ct,  cx_ct  = cent["ct"]
        ax.plot(cx_mri, cy_mri, "o", color=color, markersize=10, markeredgecolor="white")
        ax.plot(cx_ct,  cy_ct,  "x", color=color, markersize=10, markeredgewidth=2)
        ax.plot([cx_mri, cx_ct], [cy_mri, cy_ct], "-", color=color, linewidth=1.2, alpha=0.6)
        patches.append(mpatches.Patch(color=color, label=name))

    mri_marker = plt.Line2D([], [], marker="o", color="w", markerfacecolor="grey",
                            label="MRI centroid", markersize=8)
    ct_marker  = plt.Line2D([], [], marker="x", color="grey", label="CT centroid",
                            markersize=8, markeredgewidth=2)
    ax.legend(handles=patches + [mri_marker, ct_marker],
              loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_title("Attribution Centroid Displacement Map")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Failure Case Analysis (Consistency vs Dice scatter)
# ---------------------------------------------------------------------------

def plot_consistency_dice_scatter(
    consistency_scores: list[float],
    dice_scores: list[float],
    structure_name: str = "all",
    save_path: Optional[Path] = None,
):
    """
    Scatter plot: x = AOS (consistency), y = Dice (segmentation quality).
    Hypothesis: high consistency -> high Dice.
    """
    from scipy.stats import spearmanr

    x = np.array(consistency_scores)
    y = np.array(dice_scores)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.6, edgecolors="k", linewidth=0.4)

    # Regression line
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, m * xfit + b, "r--", linewidth=1.5)
        r, p = spearmanr(x, y)
        ax.set_title(
            f"Consistency vs Dice — {structure_name}\n"
            f"Spearman r={r:.3f}, p={p:.3f}"
        )
    else:
        ax.set_title(f"Consistency vs Dice — {structure_name}")

    ax.set_xlabel("Attribution Overlap Score (AOS)")
    ax.set_ylabel("Dice Score")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
