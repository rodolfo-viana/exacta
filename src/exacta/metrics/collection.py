"""Stateful shared metric evaluation, with one distributed payload for all requested scores."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from exacta._inputs import InputFormat, to_torch
from exacta.functional.collection import MetricSelection, _selection, evaluate
from exacta.functional.surface import Backend
from exacta.functional import overlap
from exacta.functional.overlap import _check_ignore_index
from exacta.metrics.base import MetricResult, _all_gather_cat
from exacta.metrics.segmentation import Aggregate, _add_counts, _pooled_counts

_POOLED = ("dice", "iou", "fbeta", "sensitivity", "precision", "specificity",
           "volume_similarity", "relative_volume_difference")


class MetricCollection:
    """Accumulate shared evaluations; return a mapping of names to ``MetricResult``.

    Arguments follow :func:`exacta.functional.evaluate`. Spacing in ``update`` overrides
    the constructor. ``aggregate="pooled"`` keeps only dataset-pooled confusion counts and
    scores them in ``compute``, as the overlap metrics' ``aggregate`` option does; it accepts
    only ratios of counts (Dice, IoU, F-beta, sensitivity, precision, specificity and the
    relative volume measures), not distances, clDice or physical sizes. Every distributed rank must call ``compute`` with identical metric
    order and configuration, including ranks without local updates. Stores only detached
    scores, never segmentations or surface distances.
    """

    def __init__(
        self, *, metrics: MetricSelection, spacing: Any = None,
        input_format: InputFormat = "index", num_classes: int | None = None,
        include_background: bool = True, backend: Backend = "auto",
        ignore_index: int | None = None, aggregate: Aggregate = "cases",
    ) -> None:
        self._metrics = _selection(metrics)
        if aggregate not in ("cases", "pooled"):
            raise ValueError(f"aggregate must be 'cases' or 'pooled', got {aggregate!r}")
        if aggregate == "pooled":
            unsupported = [name for name in self._metrics if name not in _POOLED]
            if unsupported:
                raise ValueError(f"aggregate='pooled' supports only ratios of confusion counts "
                                 f"{_POOLED}; got {unsupported}")
        self.aggregate = aggregate
        self._spacing = spacing
        self._kwargs = dict(input_format=input_format, num_classes=num_classes,
                            include_background=include_background, backend=backend,
                            ignore_index=_check_ignore_index(ignore_index, input_format))
        self._values: list[Tensor] = []
        self._counts: Tensor | None = None

    def update(self, pred: Any, target: Any, *, spacing: Any = None, mask: Any = None) -> dict[str, Tensor]:
        if self.aggregate == "pooled":
            kw = self._kwargs
            counts = overlap._confusion_counts(
                to_torch(pred), to_torch(target), kw["input_format"], kw["num_classes"],  # type: ignore[arg-type]
                kw["include_background"], ignore_index=kw["ignore_index"], mask=mask,  # type: ignore[arg-type]
            )
            self._counts = _add_counts(self._counts, counts)
            return {name: overlap._score_counts(name, counts, **options) for name, options in self._metrics.items()}
        values = evaluate(to_torch(pred), to_torch(target), metrics=self._metrics, mask=mask,
                          spacing=self._spacing if spacing is None else spacing, **self._kwargs)  # type: ignore[arg-type]
        self._values.append(torch.stack(list(values.values()), -1).flatten(1).detach())
        return values

    def compute(self) -> dict[str, MetricResult]:
        if self.aggregate == "pooled":
            totals = _pooled_counts(self._counts, ("MetricCollection", self._kwargs, self._metrics))
            results = {}
            for name, options in self._metrics.items():
                per_class = overlap._score_counts(name, totals, **options)[0]
                results[name] = MetricResult(None, per_class, per_class.nanmean())
            return results
        packed = _all_gather_cat(torch.cat(self._values) if self._values else None)
        if packed is None:
            raise RuntimeError("MetricCollection.compute() called before update()")
        unpacked = packed.reshape(packed.shape[0], packed.shape[1] // len(self._metrics), len(self._metrics))
        results = {}
        for i, name in enumerate(self._metrics):
            per_case = unpacked[..., i]
            per_class = per_case.nanmean(0)
            results[name] = MetricResult(per_case, per_class, per_class.nanmean())
        return results

    def signatures(self) -> dict[str, str]:
        """Reporting signature of every metric, with ``aggregate``; see :mod:`exacta.conventions`."""
        from exacta.conventions import signatures

        return signatures(self._metrics, aggregate=self.aggregate)

    def reset(self) -> None:
        self._values.clear()
        self._counts = None
