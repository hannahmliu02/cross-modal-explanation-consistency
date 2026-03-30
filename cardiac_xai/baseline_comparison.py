"""
Baseline comparison: separate MRI and CT models (no shared encoder).

Trains two independent U-Nets (one per modality) and evaluates their
cross-modal explanation consistency. Expected result: lower consistency
than the shared-encoder model, showing architectural choice matters.

Usage:
    python baseline_comparison.py [--epochs 100] [--device cuda]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.utils import set_determinism
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import get_parser, NUM_CLASSES, LABEL_MAP, RESULTS_DIR
from data.dataset import CardiacSliceDataset
from xai.attribution import Attributor
from xai.consistency import compute_consistency
from torch.utils.data import DataLoader


def build_modality_unet():
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    )


def attach_gradcam_layer(net):
    """Point GradCAM at the last encoder block, matching SharedEncoderUNet."""
    encoder_layers = net.model[0]
    # Use object.__setattr__ to avoid registering as a submodule
    object.__setattr__(net, "gradcam_target_layer", list(encoder_layers.children())[-1])
    return net


def one_hot(label, num_classes):
    b, h, w = label.shape
    oh = torch.zeros(b, num_classes, h, w, device=label.device)
    oh.scatter_(1, label.unsqueeze(1), 1)
    return oh


def train_single_modality(modality, data_pack, epochs, lr, batch_size, device, ckpt_dir, patience=25):
    ds_train = CardiacSliceDataset(data_pack, modality=modality, split="train")
    ds_val   = CardiacSliceDataset(data_pack, modality=modality, split="val")
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = build_modality_unet().to(device)
    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = 0.0
    patience_counter = 0
    ckpt_path = ckpt_dir / f"baseline_{modality}_best.pth"
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
    structure_names = list(LABEL_MAP.values())[1:]
    training_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(dl_train, desc=f"[{modality}] epoch {epoch}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = loss_fn(logits, one_hot(labels, NUM_CLASSES))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in dl_val:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                preds  = torch.argmax(logits, 1)
                preds_oh = one_hot(preds, NUM_CLASSES)
                label_oh = one_hot(labels, NUM_CLASSES)
                dice_metric(preds_oh, label_oh)
                dice_metric_batch(preds_oh, label_oh)

        val_dice = dice_metric.aggregate().mean().item()
        per_class = dice_metric_batch.aggregate().tolist()
        dice_metric.reset()
        dice_metric_batch.reset()

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        else:
            patience_counter += 1

        log_entry = {"epoch": epoch, "val_dice": val_dice, "patience": patience_counter}
        for name, d in zip(structure_names, per_class):
            log_entry[name] = d
        training_log.append(log_entry)

        print(f"  [{modality}] epoch {epoch:03d}  val dice {val_dice:.4f}  patience {patience_counter}/{patience}")
        col_w = 14
        print(f"    {'structure':<20}  {'val_dice':>{col_w}}")
        for name, d in zip(structure_names, per_class):
            print(f"    {name:<20}  {d:>{col_w}.4f}")

        if patience_counter >= patience:
            print(f"  [{modality}] Early stopping at epoch {epoch}")
            break

    # Save training log
    import csv
    log_path = ckpt_dir.parent / "results" / f"baseline_{modality}_training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if training_log:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
            writer.writeheader()
            writer.writerows(training_log)
        print(f"  [{modality}] Training log saved to {log_path}")

    # Reload best
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    attach_gradcam_layer(model)
    print(f"  [{modality}] Best val dice: {best_dice:.4f}")
    return model


def run_baseline_consistency(ct_model, mr_model, ct_val, mr_val, device, methods):
    """Compute consistency metrics using separately trained models."""
    all_results = []

    # Pair by volume+slice so we compare the same anatomical location
    ct_index = {(s["volume"], s["slice"]): i for i, s in enumerate(ct_val.samples)}
    mr_index = {(s["volume"], s["slice"]): i for i, s in enumerate(mr_val.samples)}
    common_keys = sorted(set(ct_index) & set(mr_index))

    for key in tqdm(common_keys, desc="baseline consistency"):
        ct_sample = ct_val[ct_index[key]]
        mr_sample = mr_val[mr_index[key]]
        ct_image = ct_sample["image"].unsqueeze(0).to(device)
        mr_image = mr_sample["image"].unsqueeze(0).to(device)
        ct_label = ct_sample["label"].numpy()
        mr_label = mr_sample["label"].numpy()

        for method in methods:
            ct_attributor = Attributor(ct_model, method=method)
            mr_attributor = Attributor(mr_model, method=method)

            ct_structs = set(ct_label.ravel()) - {0}
            mr_structs = set(mr_label.ravel()) - {0}
            common = sorted(ct_structs & mr_structs)

            for sid in common:
                ct_attr = ct_attributor.explain(ct_image, target_class=int(sid))
                mr_attr = mr_attributor.explain(mr_image, target_class=int(sid))
                consistency = compute_consistency(
                    attr_mri=mr_attr, attr_ct=ct_attr,
                    label_mri=mr_label, label_ct=ct_label,
                    method=f"baseline_{method}",
                    volume_mri=mr_sample["volume"],
                    volume_ct=ct_sample["volume"],
                    slice_mri=mr_sample["slice"],
                    slice_ct=ct_sample["slice"],
                )
                all_results.append(consistency.to_dict())

            ct_attributor.cleanup()
            mr_attributor.cleanup()

    return all_results


def main():
    parser = get_parser("Baseline comparison: separate MRI + CT models")
    parser.add_argument("--methods", nargs="+", default=["gradcam"],
                        help="Attribution methods for baseline comparison")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load existing baseline checkpoints instead of retraining")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    set_determinism(seed=42)

    if args.skip_training:
        ct_model = build_modality_unet().to(device)
        mr_model = build_modality_unet().to(device)
        for modality, model in [("ct", ct_model), ("mr", mr_model)]:
            ckpt_path = args.checkpoints_dir / f"baseline_{modality}_best.pth"
            if not ckpt_path.exists():
                print(f"[error] Checkpoint not found: {ckpt_path}. Run without --skip-training first.")
                sys.exit(1)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            attach_gradcam_layer(model)
            print(f"Loaded {modality} model from {ckpt_path}")
    else:
        print("Training separate CT model...")
        ct_model = train_single_modality(
            "ct", args.data_pack, args.epochs, args.lr,
            args.batch_size, device, args.checkpoints_dir, patience=args.patience
        )

        print("Training separate MRI model...")
        mr_model = train_single_modality(
            "mr", args.data_pack, args.epochs, args.lr,
            args.batch_size, device, args.checkpoints_dir, patience=args.patience
        )

    ct_val = CardiacSliceDataset(args.data_pack, modality="ct", split="val")
    mr_val = CardiacSliceDataset(args.data_pack, modality="mr", split="val")

    print("Running baseline consistency analysis...")
    results = run_baseline_consistency(
        ct_model, mr_model, ct_val, mr_val, device, args.methods
    )

    out_path = args.results_dir / "baseline_consistency_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))

    # Summary
    aos_vals = [
        s["aos"]
        for r in results
        for s in r["structures"]
    ]
    print(f"\nBaseline mean AOS: {np.mean(aos_vals):.4f} ± {np.std(aos_vals):.4f}")
    print(f"Saved baseline results to {out_path}")
    print("Compare with shared-encoder results in results/consistency_results.json")


if __name__ == "__main__":
    main()
