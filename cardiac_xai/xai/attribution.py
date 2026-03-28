"""
Attribution method wrappers using Captum.

Supports three methods:
  - gradcam    : Grad-CAM on the last encoder layer
  - ig         : Integrated Gradients (baseline = zero image, steps=50)
  - smoothgrad : SmoothGrad (Gaussian noise, N=50 samples)

Usage:
    attributor = Attributor(model, method="gradcam")
    attr_map = attributor.explain(image_tensor, target_class=1)  # (H, W)
    all_maps = attributor.explain_all_classes(image_tensor)      # (C, H, W)
"""

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_CLASSES, IG_STEPS, SMOOTHGRAD_SAMPLES, SMOOTHGRAD_STD


class GradCAMAttributor:
    """
    Grad-CAM hooked onto a target layer.

    Activations and gradients are captured via forward/backward hooks.
    The attribution map is upsampled to input resolution.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def explain(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Args:
            image: (1, 1, H, W) float tensor on model's device
            target_class: int in [0, NUM_CLASSES)
        Returns:
            attr_map: (H, W) numpy array, normalised [0, 1]
        """
        self.model.eval()
        image = image.requires_grad_(True)

        logits = self.model(image)                      # (1, C, H, W)
        score = logits[0, target_class].sum()
        self.model.zero_grad()
        score.backward(retain_graph=False)

        acts = self._activations   # (1, C_feat, H', W')
        grads = self._gradients    # (1, C_feat, H', W')

        # Global average pool the gradients (weights)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C_feat, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input resolution
        H, W = image.shape[-2:]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = _normalise(cam)
        return cam


class IGAttributor:
    """Integrated Gradients (Captum-based)."""

    def __init__(self, model: nn.Module, steps: int = IG_STEPS):
        self.model = model
        self.steps = steps

    def explain(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Args:
            image: (1, 1, H, W) float tensor
        Returns:
            attr_map: (H, W) numpy array, normalised [0, 1]
        """
        from captum.attr import IntegratedGradients

        self.model.eval()

        def forward_fn(x):
            return self.model(x)[:, target_class].sum(dim=(1, 2))

        ig = IntegratedGradients(forward_fn)
        baseline = torch.zeros_like(image)
        attr = ig.attribute(image, baselines=baseline, n_steps=self.steps)
        attr = attr.detach().squeeze().cpu().numpy()  # (H, W) or (1, H, W)
        if attr.ndim == 3:
            attr = attr[0]
        attr = np.abs(attr)
        attr = _normalise(attr)
        return attr


class SmoothGradAttributor:
    """SmoothGrad: average gradient magnitude over noise-perturbed inputs."""

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = SMOOTHGRAD_SAMPLES,
        noise_std: float = SMOOTHGRAD_STD,
    ):
        self.model = model
        self.n_samples = n_samples
        self.noise_std = noise_std

    def explain(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Args:
            image: (1, 1, H, W) float tensor
        Returns:
            attr_map: (H, W) numpy array, normalised [0, 1]
        """
        self.model.eval()
        total_grad = torch.zeros_like(image)

        for _ in range(self.n_samples):
            noise = torch.randn_like(image) * self.noise_std
            inp = (image.detach() + noise).requires_grad_(True)
            logits = self.model(inp)
            score = logits[0, target_class].sum()
            self.model.zero_grad()
            score.backward()
            if inp.grad is not None:
                total_grad += inp.grad.detach().abs()

        avg_grad = (total_grad / self.n_samples).squeeze().cpu().numpy()
        if avg_grad.ndim == 3:
            avg_grad = avg_grad[0]
        avg_grad = _normalise(avg_grad)
        return avg_grad


class Attributor:
    """
    Unified interface for all attribution methods.

    Args:
        model: SharedEncoderUNet (or any nn.Module)
        method: one of "gradcam", "ig", "smoothgrad"
    """

    def __init__(
        self,
        model: nn.Module,
        method: Literal["gradcam", "ig", "smoothgrad"] = "gradcam",
    ):
        self.model = model
        self.method = method
        self._impl = self._build(method)

    def _build(self, method):
        if method == "gradcam":
            # Try to get the gradcam_target_layer from model
            target_layer = getattr(self.model, "gradcam_target_layer", None)
            if target_layer is None:
                # Fallback: last named module with weight
                named = list(self.model.named_modules())
                for name, mod in reversed(named):
                    if isinstance(mod, nn.Conv2d):
                        target_layer = mod
                        break
            return GradCAMAttributor(self.model, target_layer)
        elif method == "ig":
            return IGAttributor(self.model)
        elif method == "smoothgrad":
            return SmoothGradAttributor(self.model)
        else:
            raise ValueError(f"Unknown method: {method}")

    def explain(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Compute attribution map for a single class.

        Args:
            image: (1, 1, H, W) float tensor (batch size 1)
            target_class: int
        Returns:
            (H, W) numpy array normalised [0, 1]
        """
        return self._impl.explain(image, target_class)

    def explain_all_classes(
        self, image: torch.Tensor, classes: list[int] | None = None
    ) -> np.ndarray:
        """
        Compute attribution maps for all (or specified) classes.

        Returns:
            (C, H, W) numpy array, one map per class
        """
        if classes is None:
            classes = list(range(NUM_CLASSES))
        maps = []
        for c in classes:
            maps.append(self.explain(image, c))
        return np.stack(maps)   # (C, H, W)

    def cleanup(self):
        """Remove hooks (GradCAM only)."""
        if isinstance(self._impl, GradCAMAttributor):
            self._impl.remove_hooks()


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if flat."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
