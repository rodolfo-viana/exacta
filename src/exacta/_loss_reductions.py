"""Small reductions shared by the PyTorch and fused loss implementations."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

WeightType = Literal["square", "simple", "uniform"]
EmptyTarget = Literal["zero", "uniform"]


class _ErrorRatio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, error: Tensor, numerator: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(error, numerator)
        return error / (numerator + error)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        error, numerator = ctx.saved_tensors
        denominator = numerator + error
        scaled = grad / denominator
        # d/dE [E / (N + E)] = N / D^2 and d/dN = -E / D^2: one product each. Differentiating the quotient
        # and the denominator separately would give 1/D - E/D^2 for the first, which cancels where E/D -> 1.
        return scaled * (numerator / denominator), -scaled * (error / denominator)


def error_ratio(error: Tensor, numerator: Tensor) -> Tensor:
    """``error / (numerator + error)``, the loss ``1 - numerator / denominator`` of an overlap ratio.

    ``denominator = numerator + error`` in every overlap loss, with ``error`` and ``numerator`` sums of
    non-negative terms, so that the quotient is the loss itself, accurate relatively (``1 - ratio``
    loses every digit below the roundoff), and its derivatives ``numerator / denominator**2`` and
    ``-error / denominator**2`` have no cancellation either.
    """
    return _ErrorRatio.apply(error, numerator)


def overlap_power(error: Tensor, power: float) -> Tensor:
    """Power with a zero derivative at the exact-zero endpoint."""
    if power == 1:
        return error
    zero = error == 0
    return torch.where(zero, 0, torch.where(zero, 1, error).pow(power))


def masked_overlap_terms(
    error: Tensor, numerator: Tensor, valid_counts: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Safe error and numerator (loss zero) and averaging scales, excluding entirely unannotated cases."""
    active = (valid_counts > 0).unsqueeze(-1)
    error = torch.where(active, error, 0)
    numerator = torch.where(active, numerator, 1)
    scale = active.to(numerator.dtype) / (active.sum() * numerator.shape[-1]).clamp_min(1)
    return error, numerator, scale


def generalized_dice_terms(
    intersection: Tensor,
    mismatch: Tensor,
    target: Tensor,
    *,
    weight_type: WeightType,
    empty_target: EmptyTarget,
    smooth: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return error, numerator and class weights; classes are the last axis.

    ``mismatch`` is ``Σ p·(1-g) + Σ (1-p)·g`` per class (``Σ p + Σ g - 2 Σ p·g``), so that the denominator
    ``Σ_c w_c (Σ p + Σ g) + smooth`` of the generalized Dice ratio is ``numerator + error`` with
    ``numerator = 2 Σ_c w_c Σ p·g + smooth`` and ``error = Σ_c w_c mismatch_c``, both sums of non-negative
    terms, and the loss is ``error / (numerator + error)``.

    Work in float64 only on the small per-class statistics, preserving inverse-volume
    weights for both very large masks and very small soft targets. Avoid taking reciprocals
    of zero even in inactive branches, so gradients into soft targets remain well-defined.
    """
    intersection, mismatch, target = intersection.double(), mismatch.double(), target.double()
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
    error = (weights * mismatch).sum(-1, keepdim=True)
    # Zero-weight cases contribute zero loss and zero gradient, also with smooth=0.
    unused = (weights == 0).all(-1, keepdim=True)
    numerator = torch.where(unused, 1, numerator)
    error = torch.where(unused, 0, error)
    return error, numerator, weights
