"""Stateful segmentation metrics. Arguments match the functions in :mod:`exacta.functional`."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor

from exacta import functional
from exacta._inputs import InputFormat, to_torch
from exacta._tabular import gathered
from exacta.functional import overlap
from exacta.functional.surface import DEFAULT_TIE_RTOL, Backend, Combine, Surface
from exacta.functional.overlap import _check_beta, _check_ignore_index
from exacta.functional.area import AreaUnit
from exacta.functional.topology import SkeletonMethod
from exacta.functional.volume import VolumeUnit
from exacta.metrics.base import MetricResult, PerCaseMetric


Aggregate = Literal["cases", "pooled"]


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


class _CountMetric(PerCaseMetric):
    """A ratio of confusion counts, averaged over cases or computed from counts pooled over cases.

    ``aggregate="cases"`` (default) records the (B, C) score of every case; ``compute`` averages
    each class over its cases, skipping NaN, then averages the classes. ``aggregate="pooled"``
    keeps only TP, FP, FN and TN summed over every case (and process), (C, 4) values, and
    ``compute`` scores those totals: one value per class for the whole dataset, as
    mmsegmentation's ``IoUMetric`` (mIoU) or a "global Dice". Pooled scores weight cases by
    their voxels, so large structures dominate; they are a different estimand from case means.
    A class undefined over the whole dataset (e.g. absent from every prediction and target for
    Dice) is NaN and left out of ``mean``; a class only ever predicted scores 0 and counts.
    """

    _name: str

    def __init__(self, *, input_format: InputFormat, num_classes: int | None, include_background: bool,
                 ignore_index: int | None, aggregate: Aggregate, **options: Any) -> None:
        super().__init__()
        if aggregate not in ("cases", "pooled"):
            raise ValueError(f"aggregate must be 'cases' or 'pooled', got {aggregate!r}")
        self.aggregate = aggregate
        self._layout: dict[str, Any] = dict(
            input_format=input_format, num_classes=num_classes, include_background=include_background,
            ignore_index=_check_ignore_index(ignore_index, input_format),
        )
        self._options = options
        self._counts: Tensor | None = None

    def update(self, pred: Any, target: Any, *, mask: Any = None) -> Tensor:
        """Add a batch and return its (B, C) per-case scores; the mask is not retained."""
        lay = self._layout
        counts = overlap._confusion_counts(
            to_torch(pred), to_torch(target), lay["input_format"], lay["num_classes"],
            lay["include_background"], ignore_index=lay["ignore_index"], mask=mask,
        )
        values = overlap._score_counts(self._name, counts, **self._options)
        if self.aggregate == "cases":
            return self._record(values)
        self._counts = _add_counts(self._counts, counts)
        return values

    def compute(self) -> MetricResult:
        if self.aggregate == "cases":
            return super().compute()
        totals = _pooled_counts(self._counts, (type(self).__name__, self._layout, self._options))
        per_class = overlap._score_counts(self._name, totals, **self._options)[0]
        return MetricResult(None, per_class, per_class.nanmean())

    def reset(self) -> None:
        super().reset()
        self._counts = None


def _add_counts(state: Tensor | None, counts: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    """Running (C, 4) TP, FP, FN, TN totals."""
    batch = torch.stack(counts, -1).sum(0)
    if state is None:
        return batch
    if state.shape != batch.shape or state.device != batch.device:
        raise ValueError("class count and device must remain constant across updates")
    return state + batch


def _pooled_counts(state: Tensor | None, config: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sum (C, 4) totals over distributed ranks, including ranks without updates; (1, C) each."""
    # str() makes NaN options compare equal across processes.
    rows = gathered(state, ("pooled_counts", str(config)))
    # Ranks exchange float64, which holds every count exactly up to 2**53 voxels.
    total = torch.stack([row.double() for row in rows]).sum(0).round().long().reshape(-1, 4)
    return tuple(total[None, :, i] for i in range(4))  # type: ignore[return-value]


class _MaskedOverlapMetric(_CountMetric):
    def __init__(self, *, input_format: InputFormat = "index", num_classes: int | None = None,
                 include_background: bool = True, both_empty: float = math.nan,
                 ignore_index: int | None = None, aggregate: Aggregate = "cases") -> None:
        super().__init__(input_format=input_format, num_classes=num_classes,
                         include_background=include_background, ignore_index=ignore_index,
                         aggregate=aggregate, both_empty=both_empty)


class Dice(_MaskedOverlapMetric):
    """Dice coefficient per case and class (see :func:`exacta.functional.dice`), averaged over
    cases or, with ``aggregate="pooled"``, computed from dataset-pooled counts."""

    _name = "dice"


class IoU(_MaskedOverlapMetric):
    """Intersection over union per case and class (see :func:`exacta.functional.iou`); with
    ``aggregate="pooled"``, the dataset IoU of each class and their mean (mIoU)."""

    _name = "iou"


class FBeta(_MaskedOverlapMetric):
    """F-beta score per case and class; see :func:`exacta.functional.fbeta`."""

    _name = "fbeta"

    def __init__(self, *, beta: float = 1.0, input_format: InputFormat = "index",
                 num_classes: int | None = None, include_background: bool = True,
                 both_empty: float = math.nan, ignore_index: int | None = None,
                 aggregate: Aggregate = "cases") -> None:
        _check_beta(beta)
        super().__init__(input_format=input_format, num_classes=num_classes,
                         include_background=include_background, both_empty=both_empty,
                         ignore_index=ignore_index, aggregate=aggregate)
        self._options["beta"] = beta


class ClDice(_OverlapMetric):
    """Hard centreline Dice per case/class; see :func:`exacta.functional.cldice`."""

    _fn = staticmethod(functional.cldice)

    def __init__(self, *, input_format: InputFormat = "index", num_classes: int | None = None,
                 include_background: bool = True, both_empty: float = math.nan,
                 method: SkeletonMethod | None = None) -> None:
        super().__init__(input_format=input_format, num_classes=num_classes,
                         include_background=include_background, both_empty=both_empty)
        if method not in (None, "zhang", "lee"):
            raise ValueError(f"method must be 'zhang', 'lee' or None, got {method!r}")
        self._kwargs["method"] = method

    def update(self, pred: Any, target: Any, *, pred_skeleton: Any = None,
               target_skeleton: Any = None) -> Tensor:
        return self._record(self._fn(to_torch(pred), to_torch(target), pred_skeleton=pred_skeleton,
                                     target_skeleton=target_skeleton, **self._kwargs))


class BettiError(PerCaseMetric):
    """Absolute Betti-number error per case and class; see :func:`exacta.functional.betti_error`."""

    def __init__(self, *, connectivity: int, dimension: int = 0, input_format: InputFormat = "index",
                 num_classes: int | None = None, include_background: bool = True) -> None:
        super().__init__()
        self._kwargs: dict[str, Any] = dict(connectivity=connectivity, dimension=dimension, input_format=input_format,
                                            num_classes=num_classes, include_background=include_background)

    def _batch(self, pred: Tensor, target: Tensor) -> Tensor:
        return functional.betti_error(pred, target, **self._kwargs)


class _RatioMetric(_CountMetric):
    def __init__(
        self,
        *,
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        zero_division: float = math.nan,
        ignore_index: int | None = None,
        aggregate: Aggregate = "cases",
        **definition: Any,
    ) -> None:
        super().__init__(input_format=input_format, num_classes=num_classes,
                         include_background=include_background, ignore_index=ignore_index,
                         aggregate=aggregate, zero_division=zero_division, **definition)


class Sensitivity(_RatioMetric):
    """Sensitivity (recall) per case and class; see :func:`exacta.functional.sensitivity`.
    The class mean of pooled sensitivity is mmsegmentation's mean accuracy (mAcc)."""

    _name = "sensitivity"


class Precision(_RatioMetric):
    """Precision per case and class; see :func:`exacta.functional.precision`."""

    _name = "precision"


class Specificity(_RatioMetric):
    """Specificity per case and class; see :func:`exacta.functional.specificity`."""

    _name = "specificity"


class VolumeSimilarity(_RatioMetric):
    """Volume similarity per case and class; see :func:`exacta.functional.volume_similarity`."""

    _name = "volume_similarity"

    def __init__(
        self,
        *,
        convention: Literal["taha", "itk"] = "taha",
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        zero_division: float = math.nan,
        ignore_index: int | None = None,
        aggregate: Aggregate = "cases",
    ) -> None:
        if convention not in ("taha", "itk"):
            raise ValueError(f"convention must be 'taha' or 'itk', got {convention!r}")
        super().__init__(
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            zero_division=zero_division,
            ignore_index=ignore_index,
            aggregate=aggregate,
            convention=convention,
        )


class RelativeVolumeDifference(_RatioMetric):
    """Relative volume difference per case and class; see
    :func:`exacta.functional.relative_volume_difference`."""

    _name = "relative_volume_difference"

    def __init__(
        self,
        *,
        absolute: bool = False,
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        zero_division: float = math.nan,
        ignore_index: int | None = None,
        aggregate: Aggregate = "cases",
    ) -> None:
        super().__init__(
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            zero_division=zero_division,
            ignore_index=ignore_index,
            aggregate=aggregate,
            absolute=absolute,
        )


class _PhysicalMetric(PerCaseMetric):
    _fn: Any

    def _configure(self, spacing: Any, unit: str, input_format: InputFormat,
                   num_classes: int | None, include_background: bool, ignore_index: int | None) -> None:
        self._spacing = spacing
        self._kwargs = dict(unit=unit, input_format=input_format, num_classes=num_classes,
                            include_background=include_background,
                            ignore_index=_check_ignore_index(ignore_index, input_format))

    def update(self, pred: Any, target: Any, *, spacing: Any = None, mask: Any = None) -> Tensor:
        """Record physical measures; spacing overrides the constructor for this batch."""
        return self._record(self._fn(to_torch(pred), to_torch(target), mask=mask,
                                    spacing=self._spacing if spacing is None else spacing, **self._kwargs))


class _VolumeMetric(_PhysicalMetric):
    def __init__(self, *, spacing: Any = None, unit: VolumeUnit = "mm3",
                 input_format: InputFormat = "index", num_classes: int | None = None,
                 include_background: bool = True, ignore_index: int | None = None) -> None:
        super().__init__()
        if unit not in ("mm3", "ml"):
            raise ValueError("unit must be 'mm3' or 'ml'")
        self._configure(spacing, unit, input_format, num_classes, include_background, ignore_index)


class _AreaMetric(_PhysicalMetric):
    def __init__(self, *, spacing: Any = None, unit: AreaUnit = "mm2",
                 input_format: InputFormat = "index", num_classes: int | None = None,
                 include_background: bool = True, ignore_index: int | None = None) -> None:
        super().__init__()
        if unit not in ("mm2", "cm2"):
            raise ValueError("unit must be 'mm2' or 'cm2'")
        self._configure(spacing, unit, input_format, num_classes, include_background, ignore_index)


class PredictedArea(_AreaMetric):
    """Predicted 2D area in mm² or cm², averaged equally across observed cases."""

    _fn = staticmethod(functional.predicted_area)


class TargetArea(_AreaMetric):
    """Target 2D area; see :func:`exacta.functional.target_area`."""

    _fn = staticmethod(functional.target_area)


class AreaError(_AreaMetric):
    """Signed or absolute physical area error; see :func:`exacta.functional.area_error`."""

    _fn = staticmethod(functional.area_error)

    def __init__(self, *, absolute: bool = False, spacing: Any = None, unit: AreaUnit = "mm2",
                 input_format: InputFormat = "index", num_classes: int | None = None,
                 include_background: bool = True, ignore_index: int | None = None) -> None:
        super().__init__(spacing=spacing, unit=unit, input_format=input_format, num_classes=num_classes,
                         include_background=include_background, ignore_index=ignore_index)
        self._kwargs["absolute"] = absolute


class PredictedVolume(_VolumeMetric):
    """Predicted 3D volume in mm³ or mL, averaged equally across observed cases."""

    _fn = staticmethod(functional.predicted_volume)


class TargetVolume(_VolumeMetric):
    """Target 3D volume; see :func:`exacta.functional.target_volume`."""

    _fn = staticmethod(functional.target_volume)


class VolumeError(_VolumeMetric):
    """Signed or absolute physical volume error; see :func:`exacta.functional.volume_error`."""

    _fn = staticmethod(functional.volume_error)

    def __init__(self, *, absolute: bool = False, spacing: Any = None, unit: VolumeUnit = "mm3",
                 input_format: InputFormat = "index", num_classes: int | None = None,
                 include_background: bool = True, ignore_index: int | None = None) -> None:
        super().__init__(spacing=spacing, unit=unit, input_format=input_format, num_classes=num_classes,
                         include_background=include_background, ignore_index=ignore_index)
        self._kwargs["absolute"] = absolute


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
        tie_rtol: float = DEFAULT_TIE_RTOL,
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
            tie_rtol=tie_rtol,
            surface=surface,
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            both_empty=both_empty,
            one_empty=one_empty,
            backend=backend,
        )
