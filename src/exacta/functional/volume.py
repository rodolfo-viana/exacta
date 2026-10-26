"""Physical 3D volumes from the shared hard-segmentation counts."""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor

from exacta._inputs import InputFormat, Segmentations, as_spacing, like_input
from exacta.functional.overlap import _confusion_counts_and_seg

VolumeUnit = Literal["mm3", "ml"]


def predicted_volume(
    pred: Any, target: Any, *, spacing: Any, unit: VolumeUnit = "mm3",
    input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
) -> Any:
    """Predicted physical volume for each case/class, as a float64 (B, K) array.

    Hard 3D inputs follow :func:`dice`. Spacing is in millimetres: a scalar, three
    axis spacings, or (B, 3). ``unit='ml'`` divides cubic millimetres by 1000.
    The target is required to apply the same ignored-label/observation domain as
    the other metrics. Only included voxels contribute. An absent class has zero
    volume; a case with no observed voxels has NaN. 2D inputs are rejected.
    """
    counts, seg = _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_volume("predicted_volume", counts, _voxel_volume(seg, spacing, unit)), pred)


def target_volume(
    pred: Any, target: Any, *, spacing: Any, unit: VolumeUnit = "mm3",
    input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
) -> Any:
    """Target physical volume; arguments and observation policy as in :func:`predicted_volume`."""
    counts, seg = _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_volume("target_volume", counts, _voxel_volume(seg, spacing, unit)), pred)


def volume_error(
    pred: Any, target: Any, *, spacing: Any, unit: VolumeUnit = "mm3", absolute: bool = False,
    input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
) -> Any:
    """Predicted minus target physical volume, optionally absolute (not relative).

    Positive signed error means overestimation. Arguments and observation policy
    as in :func:`predicted_volume`. Error is zero for two observed empty masks.
    """
    counts, seg = _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_volume("volume_error", counts, _voxel_volume(seg, spacing, unit), absolute), pred)


def _voxel_volume(seg: Segmentations, spacing: Any, unit: str) -> Tensor:
    if seg.ndim != 3:
        raise ValueError("physical volume requires 3D inputs; 2D pixels measure area")
    if unit not in ("mm3", "ml"):
        raise ValueError("unit must be 'mm3' or 'ml'; spacing is in millimetres")
    if spacing is None:
        raise ValueError("spacing in millimetres is required for physical volumes")
    volumes = [math.prod(s) / (1000 if unit == "ml" else 1)
               for s in as_spacing(spacing, batch=seg.batch, ndim=3)]
    if any(not math.isfinite(v) or v <= 0 for v in volumes):
        raise ValueError("spacing product must be finite and positive in the requested volume unit")
    return torch.tensor(volumes, device=seg.pred.device, dtype=torch.float64).reshape(-1, 1)


def _score_volume(name: str, counts: tuple[Tensor, Tensor, Tensor, Tensor],
                  scale: Tensor, absolute: bool = False) -> Tensor:
    tp, fp, fn, tn = counts
    value = tp + fp if name == "predicted_volume" else tp + fn if name == "target_volume" else fp - fn
    if absolute:
        value = value.abs()
    return torch.where(tp + fp + fn + tn > 0, value.double() * scale, torch.nan)
