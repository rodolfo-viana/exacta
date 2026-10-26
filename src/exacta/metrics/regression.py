"""Streaming regression without storing predictions."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from exacta._tabular import gathered
from exacta.functional import regression as r


class RegressionMetrics:
    """Pool weighted errors and centered moments in O(D) memory.

    ``update`` accepts the functional regression layouts, masks and sample weights.
    ``compute`` returns a dictionary of float64 tensors, aggregating all updates
    and distributed ranks before calculating scores. All ranks must call compute
    with identical settings, including ranks without data. Set the local CUDA
    device before NCCL collectives. Avoid padded samplers that duplicate samples.
    Repeated computes are idempotent and do not modify local state.
    """

    def __init__(self, *, metrics: tuple[str, ...] = r._NAMES,
                 multioutput: r.Multioutput = "uniform_average", force_finite: bool = False) -> None:
        r._options(multioutput, force_finite)
        if not metrics or any(name not in r._NAMES for name in metrics):
            raise ValueError(f"metrics must be a nonempty selection from {r._NAMES}")
        if multioutput == "variance_weighted" and any(n not in ("r2", "explained_variance") for n in metrics):
            raise ValueError("variance_weighted is only defined for r2 and explained_variance")
        self.metrics, self.multioutput, self.force_finite = tuple(metrics), multioutput, force_finite
        self._state: Tensor | None = None

    @torch.no_grad()
    def update(self, pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None) -> None:
        state = r._statistics(pred, target, mask, sample_weight)
        self._state = state if self._state is None else r._merge(self._state, state)

    @torch.no_grad()
    def compute(self) -> dict[str, Tensor]:
        rows = gathered(self._state, ("regression", self.metrics, self.multioutput, self.force_finite))
        state = rows[0].reshape(r._ROWS, -1)
        for row in rows[1:]:
            state = r._merge(state, row.reshape(r._ROWS, -1))
        return r._scores(state, self.metrics, self.multioutput, self.force_finite)

    def reset(self) -> None:
        self._state = None
