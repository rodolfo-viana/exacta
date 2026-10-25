"""Case-averaged probability scores and pooled voxel calibration."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from exacta._inputs import to_torch
from exacta.functional.calibration import (
    CalibrationMode, ReliabilityDiagram, _bin_sums, _diagram, brier_score,
    expected_calibration_error, negative_log_likelihood,
)
from exacta.metrics.base import PerCaseMetric, _all_gather_cat


class _ProbabilityMetric(PerCaseMetric):
    _fn: Any

    def __init__(self, *, from_logits: bool = False, ignore_index: int | None = None) -> None:
        super().__init__()
        self._kwargs: dict[str, Any] = dict(from_logits=from_logits, ignore_index=ignore_index)

    def update(self, pred: Any, target: Any, *, mask: Any = None) -> Tensor:
        return self._record(self._fn(to_torch(pred), to_torch(target), mask=mask, **self._kwargs))


class BrierScore(_ProbabilityMetric):
    """Accumulate per-case Brier scores; compute averages cases with equal weight."""
    _fn = staticmethod(brier_score)


class NegativeLogLikelihood(_ProbabilityMetric):
    """Accumulate per-case NLL; compute averages cases with equal weight."""
    _fn = staticmethod(negative_log_likelihood)


class ExpectedCalibrationError(_ProbabilityMetric):
    """Mean of per-case ECE values; use Calibration for pooled voxel bins."""
    _fn = staticmethod(expected_calibration_error)

    def __init__(self, *, bins: int = 15, mode: CalibrationMode = "top-label",
                 from_logits: bool = False, ignore_index: int | None = None) -> None:
        super().__init__(from_logits=from_logits, ignore_index=ignore_index)
        self._kwargs.update(bins=bins, mode=mode)


class Calibration:
    """Stream pooled reliability bins across cases/batches, storing only O(classes*bins).

    ``compute`` sums bin statistics across distributed ranks (all must call with
    identical options, including empty ranks). Result shapes are (1,K,bins), with
    (1,K) ECE. Pooling weights voxels equally, unlike ExpectedCalibrationError's
    case average. Background can dominate top-label calibration: inspect classwise
    diagrams as well. Repeated/padded distributed samples count more than once.
    """

    def __init__(self, *, bins: int = 15, mode: CalibrationMode = "top-label",
                 from_logits: bool = False, ignore_index: int | None = None) -> None:
        self.bins = bins
        self._kwargs = dict(bins=bins, mode=mode, from_logits=from_logits, ignore_index=ignore_index)
        self._sums: Tensor | None = None

    @torch.no_grad()
    def update(self, pred: Any, target: Any, *, mask: Any = None) -> None:
        values = _bin_sums(pred, target, mask=mask, **self._kwargs).sum(0)  # type: ignore[arg-type]
        if self._sums is None:
            self._sums = values
        else:
            if values.shape != self._sums.shape:
                raise ValueError("calibration channel counts must agree across updates")
            self._sums += values.to(self._sums.device)

    def compute(self) -> ReliabilityDiagram:
        packed = _all_gather_cat(None if self._sums is None else self._sums.reshape(1, -1))
        if packed is None:
            raise RuntimeError("Calibration.compute() called before update()")
        return _diagram(packed.sum(0).reshape(1, -1, 3, self.bins))

    def reset(self) -> None:
        self._sums = None
