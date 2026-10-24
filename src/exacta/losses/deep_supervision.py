"""Loss composition over explicitly paired resolution levels."""

from __future__ import annotations

import math
from collections.abc import Sequence

from torch import Tensor, nn


class DeepSupervisionLoss(nn.Module):
    """Weighted sum of one loss over several resolutions, without interpolation.

    ``weights`` must be finite, nonnegative, and contain a positive entry. They are
    used exactly as supplied, without normalization. Pass equal-length sequences of
    predictions and targets to forward, and optionally per-resolution voxel masks.
    A zero-weight level is skipped entirely and receives no gradient. Resolution
    order is chosen by the caller and must match the weights. The wrapped loss must
    return a scalar and accept ``mask`` when masks are supplied.

    Under distributed training, every rank must use identical weights and call the
    same levels in the same order; the wrapped loss owns its reduction semantics.
    """

    def __init__(self, loss: nn.Module, weights: Sequence[float]) -> None:
        super().__init__()
        self.loss = loss
        self.weights = tuple(float(w) for w in weights)
        if (not self.weights or any(not math.isfinite(w) or w < 0 for w in self.weights)
                or not any(w > 0 for w in self.weights)):
            raise ValueError("weights must be finite, nonnegative and contain a positive entry")

    def forward(
        self, inputs: Sequence[Tensor], targets: Sequence[Tensor], *,
        masks: Sequence[Tensor | None] | None = None,
    ) -> Tensor:
        if isinstance(inputs, Tensor) or isinstance(targets, Tensor):
            raise TypeError("inputs and targets must be sequences of tensors, one per resolution")
        if len(inputs) != len(self.weights) or len(targets) != len(self.weights):
            raise ValueError("inputs, targets and weights must have the same length")
        if masks is not None and (isinstance(masks, Tensor) or len(masks) != len(self.weights)):
            raise ValueError("masks must be a sequence with one entry per resolution")
        total: Tensor | None = None
        for i, weight in enumerate(self.weights):
            if weight == 0:
                continue
            value = (self.loss(inputs[i], targets[i]) if masks is None else
                     self.loss(inputs[i], targets[i], mask=masks[i]))
            if value.ndim != 0:
                raise ValueError("the wrapped loss must return a scalar")
            term = weight * value
            total = term if total is None else total + term
        assert total is not None
        return total
