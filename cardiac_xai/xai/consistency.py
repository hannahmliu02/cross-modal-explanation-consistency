"""
Cross-modal explanation consistency metrics.

Core novel contribution: quantify how consistently a model explains
the same cardiac structure when presented with MRI vs CT images.

All metrics take a pair of attribution maps (one MRI, one CT) and
a ground-truth label mask, returning per-structure scores.

Metrics
-------
1. SSIM  -- Structural Similarity Index
2. AOS   -- Attribution Overlap Score (novel): IoU of binarised maps
3. Spearman -- Rank correlation within the structure mask
4. ACD   -- Attribution Centroid Displacement (novel): L2 dist of mass centroids
"""

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_MAP, NUM_CLASSES, ATTRIBUTION_THRESHOLD


@dataclass
class StructureConsistency:
    structure_id: int
    structure_name: str
    ssim: float
    aos: float          # Attribution Overlap Score
    spearman_r: float
    spearman_p: float
    acd: float          # Attribution Centroid Displacement (pixels, normalised)
    acd_raw: float      # ACD in pixels (unnormalised)
    n_pixels_mri: int   # pixels in structure mask used
    n_pixels_ct: int


@dataclass
class ConsistencyResult:
    method: str
    volume_mri: int
    volume_ct: int
    slice_mri: int
    slice_ct: int
    structures: list[StructureConsistency] = field(default_factory=list)

    def mean_aos(self) -> float:
        scores = [s.aos for s in self.structures]
        return float(np.mean(scores)) if scores else 0.0

    def mean_ssim(self) -> float:
        scores = [s.ssim for s in self.structures]
        return float(np.mean(scores)) if scores else 0.0

    def trustworthiness_score(self) -> float:
        """
        Composite Trustworthiness Score T (novel):
        T = mean(AOS, normalised_SSIM, normalised_Spearman)
        SSIM and Spearman are already in [-1, 1] / [0, 1], AOS in [0, 1].
        We map Spearman r from [-1,1] -> [0,1] via (r+1)/2.
        """
        if not self.structures:
            return 0.0
        scores = []
        for s in self.structures:
            aos = s.aos
            norm_ssim = (s.ssim + 1) / 2   # SSIM in [-1,1], map to [0,1]
            norm_sp = (s.spearman_r + 1) / 2
            scores.append(np.mean([aos, norm_ssim, norm_sp]))
        return float(np.mean(scores))

    def to_dict(self):
        d = asdict(self)
        d["mean_aos"] = self.mean_aos()
        d["mean_ssim"] = self.mean_ssim()
        d["trustworthiness_score"] = self.trustworthiness_score()
        return d


def compute_consistency(
    attr_mri: np.ndarray,
    attr_ct: np.ndarray,
    label_mri: np.ndarray,
    label_ct: np.ndarray,
    method: str,
    volume_mri: int = 0,
    volume_ct: int = 0,
    slice_mri: int = 0,
    slice_ct: int = 0,
    threshold: float = ATTRIBUTION_THRESHOLD,
) -> ConsistencyResult:
    """
    Compute all consistency metrics between an MRI and CT attribution map pair.

    Args:
        attr_mri: (H, W) float in [0, 1]
        attr_ct:  (H, W) float in [0, 1]
        label_mri: (H, W) int  -- GT segmentation for the MRI slice
        label_ct:  (H, W) int  -- GT segmentation for the CT slice
        method: attribution method name (for bookkeeping)
        threshold: binarisation threshold for AOS (default 0.5)

    Returns:
        ConsistencyResult with one StructureConsistency per present structure
    """
    result = ConsistencyResult(
        method=method,
        volume_mri=volume_mri,
        volume_ct=volume_ct,
        slice_mri=slice_mri,
        slice_ct=slice_ct,
    )

    H, W = attr_mri.shape
    # Structures present in BOTH slices
    ids_mri = set(np.unique(label_mri)) - {0}
    ids_ct  = set(np.unique(label_ct))  - {0}
    common_ids = sorted(ids_mri & ids_ct)

    for sid in common_ids:
        mask_mri = (label_mri == sid)
        mask_ct  = (label_ct  == sid)

        sc = _structure_consistency(
            attr_mri, attr_ct, mask_mri, mask_ct, sid, threshold, H, W
        )
        result.structures.append(sc)

    return result


def _structure_consistency(
    attr_mri: np.ndarray,
    attr_ct: np.ndarray,
    mask_mri: np.ndarray,
    mask_ct: np.ndarray,
    sid: int,
    threshold: float,
    H: int,
    W: int,
) -> StructureConsistency:
    name = LABEL_MAP.get(sid, f"class_{sid}")

    # --- 1. SSIM over full image ---
    ssim_score = ssim(attr_mri, attr_ct, data_range=1.0)

    # --- 2. AOS: IoU of binarised maps ---
    # threshold is a percentile (0.5 = top 50% of attribution mass)
    # If either map is flat (all zeros), AOS is undefined — return 0.
    if attr_mri.max() < 1e-8 or attr_ct.max() < 1e-8:
        aos = 0.0
    else:
        t_mri = np.percentile(attr_mri, threshold * 100)
        t_ct  = np.percentile(attr_ct,  threshold * 100)
        bin_mri = (attr_mri >= t_mri).astype(bool)
        bin_ct  = (attr_ct  >= t_ct).astype(bool)
        intersection = (bin_mri & bin_ct).sum()
        union = (bin_mri | bin_ct).sum()
        aos = float(intersection / union) if union > 0 else 0.0

    # --- 3. Spearman within structure mask (union of both masks) ---
    union_mask = mask_mri | mask_ct
    px_mri = attr_mri[union_mask].ravel()
    px_ct  = attr_ct[union_mask].ravel()
    if len(px_mri) < 2:
        sp_r, sp_p = 0.0, 1.0
    else:
        result = spearmanr(px_mri, px_ct)
        sp_r = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
        sp_p = float(result.pvalue)    if hasattr(result, "pvalue")    else float(result[1])
        sp_r = sp_r if np.isfinite(sp_r) else 0.0

    # --- 4. ACD: centroid displacement ---
    c_mri = _centroid(attr_mri)
    c_ct  = _centroid(attr_ct)
    acd_raw = float(np.linalg.norm(np.array(c_mri) - np.array(c_ct)))
    # Normalise by structure diameter (sqrt of mask area)
    diam = float(np.sqrt(max(mask_mri.sum(), mask_ct.sum(), 1)))
    acd_norm = acd_raw / diam

    return StructureConsistency(
        structure_id=sid,
        structure_name=name,
        ssim=float(ssim_score),
        aos=aos,
        spearman_r=sp_r,
        spearman_p=sp_p,
        acd=acd_norm,
        acd_raw=acd_raw,
        n_pixels_mri=int(mask_mri.sum()),
        n_pixels_ct=int(mask_ct.sum()),
    )


def _centroid(attr: np.ndarray):
    """Compute the mass centroid of an attribution map."""
    total = attr.sum()
    if total < 1e-8:
        return (attr.shape[0] / 2, attr.shape[1] / 2)
    ys, xs = np.meshgrid(
        np.arange(attr.shape[0]), np.arange(attr.shape[1]), indexing="ij"
    )
    cy = float((ys * attr).sum() / total)
    cx = float((xs * attr).sum() / total)
    return cy, cx


# ---------------------------------------------------------------------------
# Unit tests (run with: python -m pytest xai/consistency.py or python xai/consistency.py)
# ---------------------------------------------------------------------------

def _test_perfect_consistency():
    """Identical maps should give SSIM=1, AOS=1, Spearman=1, ACD=0."""
    rng = np.random.default_rng(0)
    attr = rng.random((64, 64)).astype(np.float32)
    label = np.zeros((64, 64), dtype=np.int64)
    label[16:48, 16:48] = 1  # LV

    r = compute_consistency(attr, attr, label, label, method="test")
    assert len(r.structures) == 1
    sc = r.structures[0]
    assert abs(sc.ssim - 1.0) < 1e-5, f"SSIM={sc.ssim}"
    assert abs(sc.aos  - 1.0) < 1e-5, f"AOS={sc.aos}"
    assert abs(sc.spearman_r - 1.0) < 1e-5, f"Spearman={sc.spearman_r}"
    assert abs(sc.acd_raw) < 1e-5, f"ACD={sc.acd_raw}"
    print("test_perfect_consistency PASSED")


def _test_zero_maps():
    """Zero maps should be handled gracefully (no NaN/inf)."""
    attr = np.zeros((64, 64), dtype=np.float32)
    label = np.ones((64, 64), dtype=np.int64)
    r = compute_consistency(attr, attr, label, label, method="test")
    for sc in r.structures:
        assert np.isfinite(sc.ssim)
        assert np.isfinite(sc.aos)
        assert np.isfinite(sc.spearman_r)
        assert np.isfinite(sc.acd)
    print("test_zero_maps PASSED")


def _test_orthogonal_maps():
    """Non-overlapping binarised maps should give AOS close to 0."""
    attr_a = np.zeros((64, 64), dtype=np.float32)
    attr_b = np.zeros((64, 64), dtype=np.float32)
    attr_a[:32, :] = 1.0   # top half hot
    attr_b[32:, :] = 1.0   # bottom half hot
    label = np.ones((64, 64), dtype=np.int64)
    r = compute_consistency(attr_a, attr_b, label, label, method="test")
    sc = r.structures[0]
    assert sc.aos < 0.05, f"AOS={sc.aos} (expected ~0)"
    print("test_orthogonal_maps PASSED")


if __name__ == "__main__":
    _test_perfect_consistency()
    _test_zero_maps()
    _test_orthogonal_maps()
    print("\nAll unit tests passed.")
