"""Physical 2D areas from the shared hard-segmentation counts."""
from __future__ import annotations

from typing import Any, Literal

from exacta._inputs import InputFormat, like_input
from exacta.functional._physical import _physical_scale, _score_size
from exacta.functional.overlap import _confusion_counts_and_seg

AreaUnit = Literal["mm2", "cm2"]


def predicted_area(
    pred: Any, target: Any, *, spacing: Any, unit: AreaUnit = "mm2",
    input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
) -> Any:
    """Predicted physical area for each case/class, as a float64 (B, K) array.

    Hard 2D inputs follow :func:`dice`. Spacing is in millimetres: a scalar, two
    axis spacings in image-axis order, or (B, 2). ``unit='cm2'`` divides square
    millimetres by 100. Pixels represent rectangular cells, not polygon contours.
    The target supplies the ignored-label/observation domain. Only included pixels
    contribute. An absent class has zero area; a case with no observed pixels has
    NaN. 3D inputs are rejected. Overlapping channels are measured independently.
    """
    counts, seg = _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_size("predicted_area", counts, _physical_scale(seg, spacing, unit, 2)), pred)


def target_area(
    pred: Any, target: Any, *, spacing: Any, unit: AreaUnit = "mm2",
    input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
) -> Any:
    """Target physical area; arguments and observation policy as in :func:`predicted_area`."""
    counts, seg = _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_size("target_area", counts, _physical_scale(seg, spacing, unit, 2)), pred)


def area_error(
    pred: Any, target: Any, *, spacing: Any, unit: AreaUnit = "mm2", absolute: bool = False,
    input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
) -> Any:
    """Predicted minus target physical area, optionally absolute (not relative).

    Positive signed error means overestimation. Arguments and observation policy
    as in :func:`predicted_area`. Error is zero for two observed empty masks.
    """
    counts, seg = _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_size("area_error", counts, _physical_scale(seg, spacing, unit, 2), absolute), pred)
