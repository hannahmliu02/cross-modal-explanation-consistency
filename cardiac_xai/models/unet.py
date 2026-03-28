"""
Shared-encoder U-Net for cross-modal cardiac segmentation.

The same encoder processes both MRI and CT inputs, enabling direct
comparison of attribution maps across modalities in the same feature space.

Input:  (B, 1, H, W)  -- single-channel (greyscale MRI or CT)
Output: (B, 8, H, W)  -- logits for 8 classes (background + 7 structures)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import UNet

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_CLASSES


class SharedEncoderUNet(nn.Module):
    """
    Thin wrapper around MONAI's UNet.

    Using MONAI's implementation directly ensures we get battle-tested
    skip connections and residual units. The 'shared encoder' property
    is inherent: a single model processes both modalities during training,
    so there is no separate encoder per modality.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        # Expose the last encoder layer for Grad-CAM hooking.
        # In MONAI's UNet the encoder is self.model.model[0] (a sequential
        # of DownBlock modules). The last downblock before the bottleneck is
        # at index -2 (index -1 is the bottleneck / upsampling path start).
        # We expose a convenient handle that attribution.py can hook onto.
        self.encoder_layers = self.model.model[0]  # SequentialDown

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Last convolutional block before the bottleneck."""
        # SequentialDown is a nn.Sequential; last child is the deepest encoder block.
        children = list(self.encoder_layers.children())
        return children[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_checkpoint(path: str | Path, device: str = "cpu") -> SharedEncoderUNet:
    model = SharedEncoderUNet()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = SharedEncoderUNet()
    x = torch.randn(2, 1, 192, 192)
    y = model(x)
    assert y.shape == (2, 8, 192, 192), f"Unexpected output shape: {y.shape}"
    print(f"Output shape: {y.shape}  [OK]")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
