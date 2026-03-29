"""
Evaluate the trained model on the validation set.

Reports per-structure Dice for MRI and CT separately.

Usage:
    python evaluate.py [--checkpoints-dir models/checkpoints] [--device cuda]
"""

import json
import sys
from pathlib import Path

import torch
from monai.metrics import DiceMetric
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import get_parser, NUM_CLASSES, LABEL_MAP
from data.dataset import CardiacSliceDataset
from models.unet import load_checkpoint
from torch.utils.data import DataLoader


def one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    b, h, w = label.shape
    oh = torch.zeros(b, num_classes, h, w, device=label.device)
    oh.scatter_(1, label.unsqueeze(1), 1)
    return oh


def evaluate_modality(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  eval", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            preds_oh = one_hot(preds, NUM_CLASSES)
            label_oh = one_hot(labels, NUM_CLASSES)
            dice_metric(y_pred=preds_oh, y=label_oh)

    per_class = dice_metric.aggregate().tolist()   # (NUM_CLASSES-1,)
    dice_metric.reset()
    return per_class


def main():
    parser = get_parser("Evaluate trained model")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    ckpt_path = args.checkpoints_dir / "best_model.pth"
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = load_checkpoint(ckpt_path, device=str(device))
    print(f"Loaded checkpoint from {ckpt_path}")

    results = {}
    for modality in ["ct", "mr"]:
        ds = CardiacSliceDataset(
            args.data_pack, modality=modality, split="val"
        )
        if len(ds) == 0:
            print(f"  [warn] No {modality} validation samples found, skipping.")
            continue
        per_class = evaluate_modality(model, ds, device)
        results[modality] = {
            LABEL_MAP[i + 1]: per_class[i]
            for i in range(len(per_class))
        }
        mean_dice = sum(per_class) / len(per_class)
        print(f"\n{modality.upper()} val Dice (mean={mean_dice:.4f}):")
        for name, d in results[modality].items():
            print(f"  {name:20s}: {d:.4f}")

    out_path = args.results_dir / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
