"""Physical 3D volumes from the shared hard-segmentation counts."""
from __future__ import annotations

from typing import Any, Literal

from exacta._inputs import InputFormat, like_input
from exacta.functional._physical import _physical_scale, _score_size
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
    return like_input(_score_size("predicted_volume", counts, _physical_scale(seg, spacing, unit, 3)), pred)


def target_volume(
    pred: Any, target: Any, *, spacing: Any, unit: VolumeUnit = "mm3",
    input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
) -> Any:
    """Target physical volume; arguments and observation policy as in :func:`predicted_volume`."""
    counts, seg = _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_size("target_volume", counts, _physical_scale(seg, spacing, unit, 3)), pred)


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
    return like_input(_score_size("volume_error", counts, _physical_scale(seg, spacing, unit, 3), absolute), pred)
