"""
Main XAI pipeline: compute cross-modal attribution maps and consistency metrics.

For each paired MRI/CT slice in the validation set:
  1. Run inference (MRI and CT)
  2. For each attribution method: compute attribution maps for both modalities
  3. Compute consistency metrics
  4. Save maps (.npy) and visualisations (.png)

Results saved to: results/consistency_results.json

Usage:
    python explain.py [--device cuda] [--methods gradcam ig smoothgrad]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    get_parser, NUM_CLASSES, LABEL_MAP, MIN_LABEL_AREA, RESULTS_DIR
)
from data.dataset import CardiacSliceDataset
from models.unet import load_checkpoint
from xai.attribution import Attributor
from xai.consistency import compute_consistency
from xai.visualise import plot_attribution_comparison


METHODS = ["gradcam", "ig", "smoothgrad"]
STRUCTURES_TO_VISUALISE = {1, 5, 6}   # LV, Myocardium, Aorta


def get_args():
    parser = get_parser("XAI explanation pipeline")
    parser.add_argument(
        "--methods", nargs="+", default=METHODS,
        choices=METHODS, help="Attribution methods to run"
    )
    parser.add_argument(
        "--max-slices", type=int, default=None,
        help="Max slices per modality to process (for debugging)"
    )
    return parser.parse_args()


def filter_slices(dataset: CardiacSliceDataset, min_area: int = MIN_LABEL_AREA):
    """Keep only slices where at least one foreground structure has >= min_area pixels."""
    kept = []
    for i, sample in enumerate(dataset.samples):
        if sample["structures_present"]:
            kept.append(i)
    return kept


def paired_slices(ct_dataset, mr_dataset):
    """
    Pair CT and MRI validation slices by volume index and slice index.
    Returns list of (ct_idx, mr_idx) tuples.
    """
    ct_index = {
        (s["volume"], s["slice"]): i
        for i, s in enumerate(ct_dataset.samples)
    }
    mr_index = {
        (s["volume"], s["slice"]): i
        for i, s in enumerate(mr_dataset.samples)
    }
    # Match by slice position; volumes are renumbered per modality so we pair
    # by position within the validation set (vol 0 CT <-> vol 0 MRI, etc.)
    pairs = []
    for (vol, sl), ct_i in ct_index.items():
        if (vol, sl) in mr_index:
            pairs.append((ct_i, mr_index[(vol, sl)]))
    return pairs


def explain_slice(
    model,
    ct_sample: dict,
    mr_sample: dict,
    methods: list[str],
    device: torch.device,
    save_dir: Path,
    pair_key: str,
) -> list[dict]:
    """
    Run all attribution methods for one CT/MRI slice pair.
    Returns list of consistency result dicts.
    """
    ct_image = ct_sample["image"].unsqueeze(0).to(device)   # (1,1,H,W)
    mr_image = mr_sample["image"].unsqueeze(0).to(device)   # (1,1,H,W)
    ct_label = ct_sample["label"].numpy()                    # (H,W)
    mr_label = mr_sample["label"].numpy()                    # (H,W)

    results = []

    for method in methods:
        attr_save_dir = save_dir / method / pair_key
        attr_save_dir.mkdir(parents=True, exist_ok=True)

        attributor = Attributor(model, method=method)
        try:
            # Get present structures in both
            ct_structs = set(np.unique(ct_label)) - {0}
            mr_structs = set(np.unique(mr_label)) - {0}
            common = sorted(ct_structs & mr_structs)

            ct_attr_all = {}
            mr_attr_all = {}

            for sid in common:
                ct_attr = attributor.explain(ct_image, target_class=int(sid))
                mr_attr = attributor.explain(mr_image, target_class=int(sid))
                ct_attr_all[sid] = ct_attr
                mr_attr_all[sid] = mr_attr

                # Save attribution maps
                np.save(attr_save_dir / f"ct_attr_class{sid}.npy", ct_attr)
                np.save(attr_save_dir / f"mr_attr_class{sid}.npy", mr_attr)

                # Visualise selected structures
                if sid in STRUCTURES_TO_VISUALISE:
                    fig = plot_attribution_comparison(
                        mri_image=mr_sample["image"][0].numpy(),
                        ct_image=ct_sample["image"][0].numpy(),
                        mri_attr=mr_attr,
                        ct_attr=ct_attr,
                        structure_name=LABEL_MAP[sid],
                        method=method,
                    )
                    fig.savefig(
                        attr_save_dir / f"comparison_class{sid}.png",
                        dpi=100, bbox_inches="tight"
                    )
                    import matplotlib.pyplot as plt
                    plt.close(fig)

            # Consistency metrics using FULL attribution maps (summed across structs)
            # Use per-class maps for a richer comparison
            for sid in common:
                consistency = compute_consistency(
                    attr_mri=mr_attr_all[sid],
                    attr_ct=ct_attr_all[sid],
                    label_mri=mr_label,
                    label_ct=ct_label,
                    method=method,
                    volume_mri=mr_sample["volume"],
                    volume_ct=ct_sample["volume"],
                    slice_mri=mr_sample["slice"],
                    slice_ct=ct_sample["slice"],
                )
                results.append(consistency.to_dict())

        finally:
            attributor.cleanup()

    return results


def main():
    args = get_args()
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
    print(f"Loaded model from {ckpt_path}")

    ct_val = CardiacSliceDataset(args.data_pack, modality="ct", split="val")
    mr_val = CardiacSliceDataset(args.data_pack, modality="mr", split="val")

    if len(ct_val) == 0 or len(mr_val) == 0:
        print("[error] No validation samples found. Run preprocessing first.")
        sys.exit(1)

    pairs = paired_slices(ct_val, mr_val)
    if args.max_slices:
        pairs = pairs[:args.max_slices]

    print(f"Processing {len(pairs)} CT/MRI slice pairs with methods: {args.methods}")

    save_dir = args.results_dir / "attributions"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ct_idx, mr_idx in tqdm(pairs, desc="Explaining slices"):
        ct_sample = ct_val[ct_idx]
        mr_sample = mr_val[mr_idx]
        pair_key = f"vol{ct_sample['volume']:03d}_sl{ct_sample['slice']:03d}"

        res = explain_slice(
            model, ct_sample, mr_sample,
            methods=args.methods,
            device=device,
            save_dir=save_dir,
            pair_key=pair_key,
        )
        all_results.extend(res)

        # Clear MPS cache to prevent memory accumulation slowing things down
        if device.type == "mps":
            torch.mps.empty_cache()

    out_path = args.results_dir / "consistency_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
    print(f"\nSaved {len(all_results)} consistency results to {out_path}")


if __name__ == "__main__":
    main()
