"""Distance-weighted segmentation objectives; geometry is detached from autograd."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exacta import _native, _triton_edt
from exacta._inputs import as_spacing
from exacta.losses.segmentation import (
    Activation, TargetFormat, _check_mask, _check_target_format, _one_hot, _upcast, _voxel_mean,
)


def _geometry_input(input: Tensor, target: Tensor, activation: Activation,
                    target_format: TargetFormat, include_background: bool,
                    mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
    input = _upcast(input)
    if input.ndim not in (4, 5) or any(n == 0 for n in input.shape):
        raise ValueError("geometry losses require nonempty (B, C, H, W) or (B, C, D, H, W)")
    valid = _check_mask(mask, input)
    # A mask excludes loss terms, but must not manufacture boundaries in target geometry.
    truth = _one_hot(target, input, target_format).to(input.dtype)
    if target_format == "one-hot" and not bool(((truth == 0) | (truth == 1)).all()):
        raise ValueError("geometry losses require complete binary targets; ignored labels are not geometry")
    if valid is not None:
        input = torch.where(valid[:, None], input, 0)
    probs = input.softmax(1) if activation == "softmax" else input.sigmoid() if activation == "sigmoid" else input
    if not include_background:
        if input.shape[1] < 2:
            raise ValueError("include_background=False needs at least 2 channels")
        probs, truth = probs[:, 1:], truth[:, 1:]
    return probs, truth, valid


def _options(activation: Activation, target_format: TargetFormat) -> None:
    if activation not in ("softmax", "sigmoid", "none"):
        raise ValueError("activation must be 'softmax', 'sigmoid' or 'none'")
    _check_target_format(target_format)


@torch.no_grad()
def distance_maps(target: Tensor, *, spacing: Any, signed: bool = False) -> Tensor:
    """Distance maps for binary ``(B, C, H, W)`` or ``(B, C, D, H, W)`` masks.

    Returns float64 on the input device, using C++ on CPU and Exacta's exact Triton
    EDT on CUDA. ``signed=False`` gives distance to the opposite label's voxel centres
    (inside plus outside EDT). Uniform masks have a zero map, matching MONAI's
    HausdorffDT convention. ``signed=True`` gives distance-to-foreground minus
    distance-to-background: negative inside. Signed maps treat the outside of the
    image as background (one-voxel padding); an empty mask has constant positive
    physical image diagonal ``sqrt(sum((size_i * spacing_i)**2))``.

    Spacing is required, in spatial axis order; a scalar, shared vector, or (B, ndim)
    array. Maps may be precomputed/reused only while target geometry and spacing stay
    unchanged. These are voxel-centre distances, without a half-voxel correction.
    """
    if target.ndim not in (4, 5) or any(n == 0 for n in target.shape):
        raise ValueError("target must be nonempty (B, C, H, W) or (B, C, D, H, W)")
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("target must contain only 0 and 1")
    if target.device.type == "cpu":
        edt = _native.edt_sq
    elif target.device.type == "cuda" and _triton_edt.available():
        edt = _triton_edt.edt_sq
    else:
        raise NotImplementedError("distance_maps supports CPU and CUDA with Triton")
    spacings = as_spacing(spacing, batch=target.shape[0], ndim=target.ndim - 2)
    masks = target.bool()
    out = torch.empty_like(target, dtype=torch.float64)
    # Shared spacing permits a single batched transform, avoiding per-class launches.
    groups = [(slice(None), spacings[0])] if len(set(spacings)) == 1 else [(slice(b, b+1), h) for b, h in enumerate(spacings)]
    for selection, h in groups:
        mask = masks[selection]
        dims = tuple(range(2, mask.ndim))
        present = mask.any(dims, keepdim=True)
        outside = edt(mask, h, batch_ndim=2).sqrt_()
        if signed:
            padded = F.pad(mask, (1, 1) * len(h), value=False)
            crop = (slice(None), slice(None), *(slice(1, -1) for _ in h))
            inside = edt(~padded, h, batch_ndim=2)[crop].sqrt()
            diagonal = math.sqrt(sum((n * s)**2 for n, s in zip(mask.shape[2:], h)))
            out[selection] = torch.where(present, outside - inside, diagonal)
        else:
            mixed = present & (~mask).any(dims, keepdim=True)
            inside = edt(~mask, h, batch_ndim=2).sqrt_()
            out[selection] = torch.where(mixed, inside + outside, 0)
    return out


def _cached_map(value: Tensor, reference: Tensor) -> Tensor:
    if value.shape != reference.shape or value.device != reference.device:
        raise ValueError("distance map must match selected channels, shape and device")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("distance map must contain finite floating-point values")
    return value.detach().to(reference.dtype)


class BoundaryLoss(nn.Module):
    """Mean ``probability * signed_target_distance`` (Kervadec et al., 2019).

    The loss can be negative and is normally added to a regional loss such as Dice.
    Uses the signed voxel-centre convention in :func:`distance_maps`, including its
    explicit empty/full-mask policies. Forward accepts a precomputed ``distance_map``
    for the selected channels; otherwise ``spacing`` is required. A mask affects the
    reduction only; target labels must remain complete throughout the image.
    """

    def __init__(self, *, activation: Activation = "softmax", target_format: TargetFormat = "index",
                 include_background: bool = False) -> None:
        super().__init__()
        _options(activation, target_format)
        self.activation, self.target_format = activation, target_format
        self.include_background = include_background

    def forward(self, input: Tensor, target: Tensor, *, spacing: Any = None,
                distance_map: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        probs, truth, valid = _geometry_input(input, target, self.activation, self.target_format,
                                               self.include_background, mask)
        distances = distance_maps(truth, spacing=spacing, signed=True) if distance_map is None else distance_map
        distances = _cached_map(distances, probs)
        return _voxel_mean(probs * distances, valid, channels=probs.shape[1])


class HausdorffDTLoss(BoundaryLoss):
    """Distance-transform surrogate ``mean((p-y)^2 * (d_p^alpha + d_y^alpha))``.

    This is a training surrogate, not Hausdorff distance or HD95. Distances use hard
    masks at ``p > threshold`` and are detached; only the squared error is
    differentiated. Uniform hard masks have distance zero (MONAI convention).
    Consequently two opposite uniform hard masks give zero: combine with Dice/CE.
    Per-channel thresholding is used also for softmax outputs, not argmax.
    ``distance_map`` caches the unpowered target map from ``distance_maps``.
    Masking excludes reduction terms but prediction geometry uses the whole image;
    all input values must be finite, including outside the reduction mask.
    """

    def __init__(self, *, alpha: float = 2.0, threshold: float = 0.5,
                 activation: Activation = "softmax", target_format: TargetFormat = "index",
                 include_background: bool = False) -> None:
        super().__init__(activation=activation, target_format=target_format, include_background=include_background)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must lie in (0, 1)")
        self.alpha, self.threshold = alpha, threshold

    def forward(self, input: Tensor, target: Tensor, *, spacing: Any = None,
                distance_map: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        valid = _check_mask(mask, input)
        probs, truth, _ = _geometry_input(input, target, self.activation, self.target_format,
                                           self.include_background)
        if not bool(torch.isfinite(probs).all()):
            raise ValueError("prediction geometry requires finite probabilities everywhere")
        target_dt = distance_maps(truth, spacing=spacing) if distance_map is None else distance_map
        target_dt = _cached_map(target_dt, probs)
        if bool((target_dt < 0).any()):
            raise ValueError("Hausdorff target distance map must be nonnegative")
        pred_dt = distance_maps(probs.detach() > self.threshold, spacing=spacing).to(probs.dtype)
        weighted = (probs - truth).square() * (pred_dt.pow(self.alpha) + target_dt.pow(self.alpha))
        return _voxel_mean(weighted, valid, channels=probs.shape[1])
