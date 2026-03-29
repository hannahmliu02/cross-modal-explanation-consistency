# Cross-Modal Explanation Consistency for Cardiac Segmentation

Evaluates whether a cardiac segmentation model's saliency maps are consistent
across MRI and CT — using explanation consistency as a trustworthiness metric.

If a model truly learns anatomy (not modality artefacts), its attribution maps
for the same cardiac structure should look similar in both MRI and CT.

**Dataset**: MM-WHS (Multi-Modality Whole Heart Segmentation), pre-processed 256×256 npz slices
**Model**: Shared-encoder U-Net (MRI + CT trained jointly)
**XAI methods**: GradCAM, Integrated Gradients, SmoothGrad
**Novel metrics**: Attribution Overlap Score (AOS), Attribution Centroid Displacement (ACD)

---

## Setup

```bash
pip install -r requirements.txt
```

Data lives at `data/pack/processed_data/` (extracted from `pack.zip`). No preprocessing needed.

---

## Pipeline

Run steps in order:

### 1. Train
```bash
python train.py
```
Trains the shared-encoder U-Net on mixed MRI+CT batches for 100 epochs.
Saves the best checkpoint (by val Dice) to `models/checkpoints/best_model.pth`.
Pass `--no-wandb` to disable Weights & Biases logging.

### 2. Evaluate
```bash
python evaluate.py
```
Reports per-structure Dice for CT and MRI separately.
Target: mean Dice > 0.75 on both modalities.

### 3. Explain
```bash
python explain.py
```
Runs GradCAM, Integrated Gradients, and SmoothGrad on all validation slices.
Saves attribution maps (`.npy`) and comparison figures (`.png`) to `results/attributions/`.
Outputs `results/consistency_results.json`.

```bash
# Run only one method for a quick check:
python explain.py --methods gradcam

# Limit to first 20 slices for debugging:
python explain.py --max-slices 20
```

### 4. Analyse
```bash
python consistency_analysis.py
```
Runs ANOVA, Kendall's W, and Trustworthiness Score ranking.
Outputs `results/statistical_analysis.json`, all figures to `results/figures/`,
and `results/consistency_table.tex`.

### 5. Baseline comparison (optional)
```bash
python baseline_comparison.py
```
Trains separate MRI and CT models and computes the same consistency metrics.
Expected result: lower consistency than the shared-encoder model.

---

## Output Structure

```
results/
├── consistency_results.json       # per-slice consistency metrics
├── statistical_analysis.json      # ANOVA, Kendall's W, T-score ranking
├── evaluation_results.json        # per-structure Dice
├── consistency_table.tex          # LaTeX table for report
├── baseline_consistency_results.json
└── figures/
    ├── consistency_heatmap.png    # structures × methods AOS heatmap
    ├── ssim_boxplot.png           # per-structure SSIM distributions
    └── consistency_dice_*.png     # AOS vs Dice scatter plots
```

Attribution maps and comparison figures saved under `results/attributions/`.

---

## Cardiac Structures

| Label | Structure |
|-------|-----------|
| 1 | Left ventricle (LV) |
| 2 | Right ventricle (RV) |
| 3 | Left atrium (LA) |
| 4 | Right atrium (RA) |
| 5 | Myocardium (Myo) |
| 6 | Aorta (Ao) |
| 7 | Pulmonary artery (PA) |

---

## Key CLI Options

All scripts accept:
- `--device cuda` / `--device cpu`
- `--data-pack` (override data directory, default: `data/pack/processed_data`)
- `--checkpoints-dir`, `--results-dir` (override output paths)
- `--epochs`, `--batch-size`, `--lr` (train/baseline only)
- `--no-wandb` (train only)
