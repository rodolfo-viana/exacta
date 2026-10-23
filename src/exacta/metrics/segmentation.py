"""Stateful segmentation metrics. Arguments match the functions in :mod:`exacta.functional`."""

from __future__ import annotations

import math
from typing import Any

from torch import Tensor

from exacta import functional
from exacta._inputs import InputFormat
from exacta.functional.surface import Backend
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
        self._kwargs = dict(
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


class HausdorffDistance(PerCaseMetric):
    """Percentile Hausdorff distance; see :func:`exacta.functional.hausdorff_distance`.

    ``spacing`` can be set once here or passed to every ``update`` call, which is how to use
    per-case spacing read from image headers.
    """

    def __init__(
        self,
        *,
        spacing: Any = None,
        percentile: float = 100.0,
        input_format: InputFormat = "index",
        num_classes: int | None = None,
        include_background: bool = True,
        both_empty: float = math.nan,
        one_empty: float = math.inf,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        self._spacing = spacing
        self._kwargs = dict(
            percentile=percentile,
            input_format=input_format,
            num_classes=num_classes,
            include_background=include_background,
            both_empty=both_empty,
            one_empty=one_empty,
            backend=backend,
        )

    def update(self, pred: Any, target: Any, spacing: Any = None) -> Tensor:
        """Add a batch; ``spacing`` overrides the constructor's spacing for this batch."""
        return super().update(pred, target, spacing=self._spacing if spacing is None else spacing)

    def _batch(self, pred: Tensor, target: Tensor, *, spacing: Any) -> Tensor:
        return functional.hausdorff_distance(pred, target, spacing=spacing, **self._kwargs)


class HD95(HausdorffDistance):
    """95th-percentile Hausdorff distance; arguments as in :class:`HausdorffDistance`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(percentile=95.0, **kwargs)
