"""Stateful shared metric evaluation, with one distributed payload for all requested scores."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from exacta._inputs import InputFormat, to_torch
from exacta.functional.collection import MetricSelection, _selection, evaluate
from exacta.functional.surface import Backend
from exacta.metrics.base import MetricResult, _all_gather_cat


class MetricCollection:
    """Accumulate shared evaluations; return a mapping of names to ``MetricResult``.

    Arguments follow :func:`exacta.functional.evaluate`. Spacing in ``update`` overrides
    the constructor. Every distributed rank must call ``compute`` with identical metric
    order and configuration, including ranks without local updates. Stores only detached
    scores, never segmentations or surface distances.
    """

    def __init__(
        self, *, metrics: MetricSelection, spacing: Any = None,
        input_format: InputFormat = "index", num_classes: int | None = None,
        include_background: bool = True, backend: Backend = "auto",
    ) -> None:
        self._metrics = _selection(metrics)
        self._spacing = spacing
        self._kwargs = dict(input_format=input_format, num_classes=num_classes,
                            include_background=include_background, backend=backend)
        self._values: list[Tensor] = []

    def update(self, pred: Any, target: Any, *, spacing: Any = None) -> dict[str, Tensor]:
        values = evaluate(to_torch(pred), to_torch(target), metrics=self._metrics,
                          spacing=self._spacing if spacing is None else spacing, **self._kwargs)  # type: ignore[arg-type]
        self._values.append(torch.stack(list(values.values()), -1).flatten(1).detach())
        return values

    def compute(self) -> dict[str, MetricResult]:
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

    def reset(self) -> None:
        self._values.clear()
