"""Stateful segmentation metrics. Arguments match the functions in :mod:`exacta.functional`."""

from __future__ import annotations

import math
from typing import Any, Literal

from torch import Tensor

from exacta import functional
from exacta._inputs import InputFormat, to_torch
from exacta.functional.surface import Backend, Combine, Surface
from exacta.metrics.base import PerCaseMetric


class _OverlapMetric(PerCaseMetric):
    _fn: Any

    def __init__(
        self,
        *,
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        both_empty: float = math.nan,
    ) -> None:
        super().__init__()
        self._kwargs: dict[str, Any] = dict(
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            both_empty=both_empty,
        )

    def _batch(self, pred: Tensor, target: Tensor) -> Tensor:
        return self._fn(pred, target, **self._kwargs)


class Dice(_OverlapMetric):
    """Dice coefficient per case and class; see :func:`exacta.functional.dice`."""

    _fn = staticmethod(functional.dice)


class IoU(_OverlapMetric):
    """Intersection over union per case and class; see :func:`exacta.functional.iou`."""

    _fn = staticmethod(functional.iou)


class ClDice(_OverlapMetric):
    """Hard centreline Dice per case/class, using C++ Lee skeletonization."""

    _fn = staticmethod(functional.cldice)

    def update(self, pred: Any, target: Any, *, pred_skeleton: Any = None,
               target_skeleton: Any = None) -> Tensor:
        return self._record(self._fn(to_torch(pred), to_torch(target), pred_skeleton=pred_skeleton,
                                     target_skeleton=target_skeleton, **self._kwargs))


class _RatioMetric(PerCaseMetric):
    _fn: Any

    def __init__(
        self,
        *,
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        zero_division: float = math.nan,
        **definition: Any,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            zero_division=zero_division,
            **definition,
        )

    def _batch(self, pred: Tensor, target: Tensor) -> Tensor:
        return self._fn(pred, target, **self._kwargs)


class Sensitivity(_RatioMetric):
    """Sensitivity (recall) per case and class; see :func:`exacta.functional.sensitivity`."""

    _fn = staticmethod(functional.sensitivity)


class Precision(_RatioMetric):
    """Precision per case and class; see :func:`exacta.functional.precision`."""

    _fn = staticmethod(functional.precision)


class Specificity(_RatioMetric):
    """Specificity per case and class; see :func:`exacta.functional.specificity`."""

    _fn = staticmethod(functional.specificity)


class VolumeSimilarity(_RatioMetric):
    """Volume similarity per case and class; see :func:`exacta.functional.volume_similarity`."""

    _fn = staticmethod(functional.volume_similarity)

    def __init__(
        self,
        *,
        convention: Literal["taha", "itk"] = "taha",
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        zero_division: float = math.nan,
    ) -> None:
        super().__init__(
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            zero_division=zero_division,
            convention=convention,
        )


class RelativeVolumeDifference(_RatioMetric):
    """Relative volume difference per case and class; see
    :func:`exacta.functional.relative_volume_difference`."""

    _fn = staticmethod(functional.relative_volume_difference)

    def __init__(
        self,
        *,
        absolute: bool = False,
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        zero_division: float = math.nan,
    ) -> None:
        super().__init__(
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            zero_division=zero_division,
            absolute=absolute,
        )


class _SurfaceMetric(PerCaseMetric):
    """Base for distance metrics.

    ``spacing`` can be set once in the constructor or passed to every ``update`` call, which is
    how to use per-case spacing read from image headers.
    """

    _fn: Any

    def __init__(self, spacing: Any, **kwargs: Any) -> None:
        super().__init__()
        self._spacing = spacing
        self._kwargs = kwargs

    def update(self, pred: Any, target: Any, spacing: Any = None) -> Tensor:
        """Add a batch; ``spacing`` overrides the constructor's spacing for this batch."""
        spacing = self._spacing if spacing is None else spacing
        return self._record(self._fn(to_torch(pred), to_torch(target), spacing=spacing, **self._kwargs))


class HausdorffDistance(_SurfaceMetric):
    """Percentile Hausdorff distance; see :func:`exacta.functional.hausdorff_distance`."""

    _fn = staticmethod(functional.hausdorff_distance)

    def __init__(
        self,
        *,
        spacing: Any = None,
        percentile: float = 100.0,
        combine: Combine = "max",
        surface: Surface = "voxels",
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        both_empty: float = math.nan,
        one_empty: float = math.inf,
        backend: Backend = "auto",
    ) -> None:
        super().__init__(
            spacing,
            percentile=percentile,
            combine=combine,
            surface=surface,
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            both_empty=both_empty,
            one_empty=one_empty,
            backend=backend,
        )


class HD95(HausdorffDistance):
    """95th-percentile Hausdorff distance; arguments as in :class:`HausdorffDistance`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(percentile=95.0, **kwargs)


class ASSD(_SurfaceMetric):
    """Average symmetric surface distance; see :func:`exacta.functional.assd`."""

    _fn = staticmethod(functional.assd)

    def __init__(
        self,
        *,
        spacing: Any = None,
        combine: Combine = "pooled",
        surface: Surface = "voxels",
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        both_empty: float = math.nan,
        one_empty: float = math.inf,
        backend: Backend = "auto",
    ) -> None:
        super().__init__(
            spacing,
            combine=combine,
            surface=surface,
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            both_empty=both_empty,
            one_empty=one_empty,
            backend=backend,
        )


class NSD(_SurfaceMetric):
    """Normalized surface Dice; see :func:`exacta.functional.nsd`."""

    _fn = staticmethod(functional.nsd)

    def __init__(
        self,
        *,
        tolerance: Any,
        spacing: Any = None,
        surface: Surface = "voxels",
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        both_empty: float = math.nan,
        one_empty: float = 0.0,
        backend: Backend = "auto",
    ) -> None:
        super().__init__(
            spacing,
            tolerance=tolerance,
            surface=surface,
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            both_empty=both_empty,
            one_empty=one_empty,
            backend=backend,
        )
