"""Small reductions shared by the PyTorch and fused loss implementations."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

WeightType = Literal["square", "simple", "uniform"]
EmptyTarget = Literal["zero", "uniform"]


def masked_overlap_terms(
    numerator: Tensor, denominator: Tensor, valid_counts: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Safe ratios and averaging scales, excluding entirely unannotated cases."""
    active = (valid_counts > 0).unsqueeze(-1)
    numerator = torch.where(active, numerator, 1)
    denominator = torch.where(active, denominator, 1)
    scale = active.to(numerator.dtype) / (active.sum() * numerator.shape[-1]).clamp_min(1)
    return numerator, denominator, scale


def generalized_dice_terms(
    intersection: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    weight_type: WeightType,
    empty_target: EmptyTarget,
    smooth: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return numerator, denominator and class weights; classes are the last axis.

    Work in float64 only on the small per-class statistics, preserving inverse-volume
    weights for both very large masks and very small soft targets. Avoid taking reciprocals
    of zero even in inactive branches, so gradients into soft targets remain well-defined.
    """
    intersection, prediction, target = intersection.double(), prediction.double(), target.double()
    if weight_type == "uniform":
        weights = torch.ones_like(target)
    else:
        present = target != 0
        safe_volume = torch.where(present, target, 1)
        weights = safe_volume.reciprocal()
        if weight_type == "square":
            weights = weights.square()
        weights = torch.where(present, weights, 0)
        largest = weights.amax(-1, keepdim=True)
        if empty_target == "uniform":
            largest = torch.where(present.any(-1, keepdim=True), largest, 1)
        weights = torch.where(present, weights, largest)
    numerator = 2 * (weights * intersection).sum(-1, keepdim=True) + smooth
    denominator = (weights * (prediction + target)).sum(-1, keepdim=True) + smooth
    # Zero-weight cases contribute zero loss and zero gradient, also with smooth=0.
    unused = (weights == 0).all(-1, keepdim=True)
    numerator = torch.where(unused, 1, numerator)
    denominator = torch.where(unused, 1, denominator)
    return numerator, denominator, weights
