"""
Statistical analysis of cross-modal explanation consistency.

Reads results/consistency_results.json and produces:
  1. Spearman correlation: AOS <-> Dice per structure
  2. One-way ANOVA across structures (per metric)
  3. Kendall's W across attribution methods (concordance)
  4. Composite Trustworthiness Score ranking

Outputs:
  results/statistical_analysis.json
  results/figures/  (all figures via xai/visualise.py)

Usage:
    python consistency_analysis.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import get_parser, LABEL_MAP, NUM_CLASSES, RESULTS_DIR
from xai.visualise import (
    plot_consistency_heatmap,
    plot_ssim_boxplot,
    plot_consistency_dice_scatter,
)


def load_results(results_path: Path) -> list[dict]:
    with open(results_path) as f:
        return json.load(f)


def extract_per_structure(results: list[dict]) -> dict:
    """
    Returns nested dict:
      {structure_name: {method: {"aos": [...], "ssim": [...], "spearman_r": [...], "acd": [...]}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        method = r["method"]
        for s in r["structures"]:
            name = s["structure_name"]
            data[name][method]["aos"].append(s["aos"])
            data[name][method]["ssim"].append(s["ssim"])
            data[name][method]["spearman_r"].append(s["spearman_r"])
            data[name][method]["acd"].append(s["acd"])
    return data


# ---------------------------------------------------------------------------
# 1. Correlation Analysis: AOS <-> Dice
# ---------------------------------------------------------------------------

def correlation_analysis(results: list[dict], dice_path: Path | None) -> dict:
    """
    If dice evaluation results are available, compute Spearman r(AOS, Dice).
    Returns {structure_name: {method: {r, p}}}
    """
    if dice_path is None or not dice_path.exists():
        print("[warn] No evaluation_results.json found; skipping AOS-Dice correlation.")
        return {}

    with open(dice_path) as f:
        eval_res = json.load(f)
    # eval_res: {modality: {structure_name: dice_score}}

    output = {}
    # Here we only have aggregate dice (not per-slice), so we report the
    # structure-level Dice alongside mean AOS for that structure across slices.
    for s_name in LABEL_MAP.values():
        if s_name == "background":
            continue
        ct_dice = eval_res.get("ct", {}).get(s_name, None)
        mr_dice = eval_res.get("mr", {}).get(s_name, None)
        mean_dice = np.mean([d for d in [ct_dice, mr_dice] if d is not None])
        output[s_name] = {"ct_dice": ct_dice, "mr_dice": mr_dice, "mean_dice": mean_dice}

    return output


# ---------------------------------------------------------------------------
# 2. One-way ANOVA across structures
# ---------------------------------------------------------------------------

def anova_across_structures(data: dict, metric: str = "aos") -> dict:
    """
    One-way ANOVA: does the metric differ across cardiac structures?
    Returns {method: {F, p, post_hoc (Tukey HSD)}}
    """
    methods = list(next(iter(data.values())).keys())
    results = {}
    for method in methods:
        groups = []
        names = []
        for struct, method_data in data.items():
            vals = method_data[method][metric]
            if vals:
                groups.append(vals)
                names.append(struct)
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        result = {"F": float(F), "p": float(p), "structures": names}

        # Tukey HSD (manual, using scipy)
        if p < 0.05:
            from scipy.stats import tukey_hsd
            try:
                tukey = tukey_hsd(*groups)
                result["tukey_pairwise"] = {
                    f"{names[i]} vs {names[j]}": float(tukey.pvalue[i, j])
                    for i in range(len(names))
                    for j in range(i + 1, len(names))
                }
            except Exception:
                pass

        results[method] = result
    return results


# ---------------------------------------------------------------------------
# 3. Kendall's W (concordance across methods)
# ---------------------------------------------------------------------------

def kendalls_w(data: dict, metric: str = "aos") -> dict:
    """
    Compute Kendall's W for each structure: do methods agree on ranking of slices?
    Returns {structure_name: W}
    """
    results = {}
    for struct, method_data in data.items():
        methods = list(method_data.keys())
        # Need equal-length vectors; take min length
        min_len = min(len(method_data[m][metric]) for m in methods)
        if min_len < 2:
            continue
        rankings = np.array([
            stats.rankdata(method_data[m][metric][:min_len])
            for m in methods
        ])  # (n_methods, n_samples)
        n, m_count = rankings.shape[1], rankings.shape[0]
        rank_sums = rankings.sum(axis=0)
        mean_rank = rank_sums.mean()
        S = np.sum((rank_sums - mean_rank) ** 2)
        W = 12 * S / (m_count ** 2 * (n ** 3 - n))
        results[struct] = float(W)
    return results


# ---------------------------------------------------------------------------
# 4. Trustworthiness Score
# ---------------------------------------------------------------------------

def trustworthiness_ranking(results: list[dict]) -> list[dict]:
    """
    Rank volumes by mean Trustworthiness Score across all slices and methods.
    """
    vol_scores = defaultdict(list)
    for r in results:
        method = r["method"]
        for s in r["structures"]:
            aos = s["aos"]
            norm_ssim = (s["ssim"] + 1) / 2
            norm_sp = (s["spearman_r"] + 1) / 2
            T = float(np.mean([aos, norm_ssim, norm_sp]))
            key = f"mri_vol{r['volume_mri']}_ct_vol{r['volume_ct']}"
            vol_scores[key].append(T)

    ranked = [
        {"volume": k, "mean_T": float(np.mean(v)), "n_samples": len(v)}
        for k, v in vol_scores.items()
    ]
    ranked.sort(key=lambda x: x["mean_T"])
    return ranked


# ---------------------------------------------------------------------------
# LaTeX table helper
# ---------------------------------------------------------------------------

def to_latex_table(rows: list[dict], columns: list[str], caption: str) -> str:
    col_fmt = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        cells = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = get_parser("Consistency statistical analysis")
    args = parser.parse_args()

    results_path = args.results_dir / "consistency_results.json"
    if not results_path.exists():
        print(f"[error] {results_path} not found. Run explain.py first.")
        sys.exit(1)

    print(f"Loading results from {results_path}")
    results = load_results(results_path)
    print(f"  {len(results)} consistency records loaded.")

    data = extract_per_structure(results)

    # ------------------------------------------------------------------ #
    # 1. Correlation with Dice
    dice_path = args.results_dir / "evaluation_results.json"
    corr = correlation_analysis(results, dice_path)

    # ------------------------------------------------------------------ #
    # 2. ANOVA
    anova = anova_across_structures(data, metric="aos")

    # ------------------------------------------------------------------ #
    # 3. Kendall's W
    W = kendalls_w(data, metric="aos")
    print(f"\nKendall's W (method concordance): {W}")

    # ------------------------------------------------------------------ #
    # 4. Trustworthiness ranking
    ranking = trustworthiness_ranking(results)
    print("\nBottom-5 trustworthiness volumes (most suspicious):")
    for r in ranking[:5]:
        print(f"  {r['volume']}  T={r['mean_T']:.3f}")

    # ------------------------------------------------------------------ #
    # Save JSON
    analysis = {
        "dice_correlation": corr,
        "anova_aos": anova,
        "kendalls_w": W,
        "trustworthiness_ranking": ranking,
    }
    out_path = args.results_dir / "statistical_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved statistical analysis to {out_path}")

    # ------------------------------------------------------------------ #
    # Figures
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 2: Consistency heatmap (mean AOS per method x structure)
    heatmap_data = {}
    methods = sorted({r["method"] for r in results})
    for method in methods:
        heatmap_data[method] = {}
        for struct, mdata in data.items():
            vals = mdata[method]["aos"]
            heatmap_data[method][struct] = float(np.mean(vals)) if vals else np.nan
    plot_consistency_heatmap(
        heatmap_data, metric="AOS",
        save_path=fig_dir / "consistency_heatmap.png"
    )
    print(f"Saved consistency heatmap to {fig_dir / 'consistency_heatmap.png'}")

    # Figure 3: SSIM boxplot
    ssim_data = {
        struct: {
            method: mdata[method]["ssim"]
            for method in mdata
        }
        for struct, mdata in data.items()
    }
    plot_ssim_boxplot(ssim_data, save_path=fig_dir / "ssim_boxplot.png")
    print(f"Saved SSIM boxplot to {fig_dir / 'ssim_boxplot.png'}")

    # Figure 5: Consistency vs Dice (per structure, if dice available)
    if corr:
        for struct, dice_info in corr.items():
            if dice_info.get("mean_dice") is None:
                continue
            # Collect per-slice AOS for this structure
            aos_vals = []
            dice_vals = []
            for r in results:
                for s in r["structures"]:
                    if s["structure_name"] == struct:
                        aos_vals.append(s["aos"])
                        dice_vals.append(dice_info["mean_dice"])
            if len(aos_vals) > 2:
                plot_consistency_dice_scatter(
                    aos_vals, dice_vals, structure_name=struct,
                    save_path=fig_dir / f"consistency_dice_{struct}.png"
                )

    # LaTeX summary table
    latex_rows = [
        {
            "Structure": struct,
            "Mean AOS": float(np.mean([v for m in mdata.values() for v in m["aos"]])),
            "Mean SSIM": float(np.mean([v for m in mdata.values() for v in m["ssim"]])),
            "Kendall W": W.get(struct, float("nan")),
        }
        for struct, mdata in data.items()
    ]
    latex = to_latex_table(
        latex_rows,
        columns=["Structure", "Mean AOS", "Mean SSIM", "Kendall W"],
        caption="Per-structure cross-modal explanation consistency metrics.",
    )
    latex_path = args.results_dir / "consistency_table.tex"
    latex_path.write_text(latex)
    print(f"Saved LaTeX table to {latex_path}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
