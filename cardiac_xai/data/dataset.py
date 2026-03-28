"""
PyTorch datasets for pre-packaged MM-WHS npz slices.

Data layout (from pack.zip):
  data/pack/processed_data/
    ct_256/{train,val,test}/npz/ct_XXXX_slice_YYY.npz
    mr_256/{train,val,test}/npz/mr_XXXX_slice_YYY.npz

Each .npz contains:
  image : (256, 256) float64  -- normalised intensity
  label : (256, 256) uint8    -- integer class 0-7
"""

import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PACK


_FNAME_RE = re.compile(r"(ct|mr)_(\d+)_slice_(\d+)\.npz")


def _parse_filename(path: Path):
    m = _FNAME_RE.match(path.name)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


class CardiacSliceDataset(Dataset):
    """
    Dataset of 2D axial slices from pre-packaged MM-WHS npz files.

    Returns dicts with keys:
      image    : FloatTensor (1, H, W)  -- single-channel, normalised [~0,1]
      label    : LongTensor  (H, W)     -- integer class 0-7
      modality : str   "ct" or "mr"
      volume   : int   e.g. 1001
      slice    : int   e.g. 150
    """

    def __init__(
        self,
        data_dir: Path = DATA_PACK,
        modality: Literal["ct", "mr", "both"] = "both",
        split: Literal["train", "val", "test"] = "train",
    ):
        self.samples = []
        data_dir = Path(data_dir)
        modalities = ["ct", "mr"] if modality == "both" else [modality]

        for mod in modalities:
            npz_dir = data_dir / f"{mod}_256" / split / "npz"
            if not npz_dir.exists():
                print(f"[warn] {npz_dir} not found, skipping")
                continue
            for p in sorted(npz_dir.glob("*.npz")):
                _, vol, sl = _parse_filename(p)
                if vol is not None:
                    self.samples.append({
                        "path": p,
                        "modality": mod,
                        "volume": vol,
                        "slice": sl,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        d = np.load(s["path"])
        image = d["image"].astype(np.float32)   # (H, W)
        label = d["label"].astype(np.int64)     # (H, W)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),  # (1, H, W)
            "label": torch.from_numpy(label),               # (H, W)
            "modality": s["modality"],
            "volume": s["volume"],
            "slice": s["slice"],
        }


def build_dataloaders(
    data_dir: Path = DATA_PACK,
    batch_size: int = 16,
    num_workers: int = 4,
):
    train_ds = CardiacSliceDataset(data_dir, modality="both", split="train")
    val_ds   = CardiacSliceDataset(data_dir, modality="both", split="val")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl


if __name__ == "__main__":
    ds = CardiacSliceDataset(split="train")
    print(f"Train: {len(ds)} samples")
    item = ds[0]
    print(f"  image: {item['image'].shape} {item['image'].dtype}")
    print(f"  label: {item['label'].shape}  unique={item['label'].unique().tolist()}")
    print(f"  modality={item['modality']}  vol={item['volume']}  slice={item['slice']}")
