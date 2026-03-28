"""
Training script for the shared-encoder U-Net on mixed MRI + CT slices.

Usage:
    python train.py [--epochs 100] [--batch-size 16] [--lr 1e-4]
                    [--device cuda] [--no-wandb]
"""

import sys
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import get_parser, NUM_CLASSES, LABEL_MAP, CHECKPOINTS_DIR
from data.dataset import build_dataloaders
from models.unet import SharedEncoderUNet



def one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert (B, H, W) int label to (B, C, H, W) one-hot float."""
    b, h, w = label.shape
    oh = torch.zeros(b, num_classes, h, w, device=label.device)
    oh.scatter_(1, label.unsqueeze(1), 1)
    return oh


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            images = batch["image"].to(device)       # (B, 1, H, W)
            labels = batch["label"].to(device)       # (B, H, W)

            logits = model(images)                    # (B, C, H, W)
            labels_oh = one_hot(labels, NUM_CLASSES)  # (B, C, H, W)

            loss = loss_fn(logits, labels_oh)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1, keepdim=True)  # (B, 1, H, W)
            preds_oh = one_hot(preds.squeeze(1), NUM_CLASSES)
            dice_metric(y_pred=preds_oh, y=labels_oh)

    mean_loss = total_loss / len(loader)
    per_class_dice = dice_metric.aggregate()           # (C-1,) excl. background
    mean_dice = per_class_dice.mean().item()
    dice_metric.reset()
    return mean_loss, mean_dice, per_class_dice.tolist()


def main():
    parser = get_parser("Train shared-encoder U-Net")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    set_determinism(seed=42)

    print(f"Device: {device}")
    train_dl, val_dl = build_dataloaders(
        data_dir=args.data_pack,
        batch_size=args.batch_size,
    )
    print(f"Train batches: {len(train_dl)}  Val batches: {len(val_dl)}")

    model = SharedEncoderUNet(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    patience_counter = 0

    structure_names = list(LABEL_MAP.values())[1:]  # excl. background

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_per_class = run_epoch(
            model, train_dl, loss_fn, optimizer, device, train=True
        )
        val_loss, val_dice, val_per_class = run_epoch(
            model, val_dl, loss_fn, optimizer, device, train=False
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f}"
        )
        col_w = 14
        print(f"  {'structure':<20}  {'val_dice':>{col_w}}")
        for name, d in zip(structure_names, val_per_class):
            print(f"  {name:<20}  {d:>{col_w}.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            ckpt_path = args.checkpoints_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_per_class_dice": val_per_class,
                },
                ckpt_path,
            )
            print(f"  -> Saved best model (val dice {best_dice:.4f}) to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  [patience {patience_counter}/{args.patience}]")
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
