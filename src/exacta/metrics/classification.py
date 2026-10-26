"""Dataset-pooled classification metrics."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from exacta._tabular import gathered
from exacta.functional import classification as c


class ClassificationMetrics:
    """Stream O(C) confusion counts, without averaging batch-level scores.

    Constructor options match functional.classification.evaluate. update accepts
    pred, target, mask and sample_weight. compute returns a dictionary of float64
    tensors and sums across distributed ranks, including ranks without updates.
    All ranks must call with identical settings; set the current device for NCCL.
    Evaluation samplers must avoid padding/duplicating samples. Input mutation and
    repeated computes do not change stored state. Empty observed domains follow
    zero_division, but compute before any update on all ranks raises RuntimeError.
    """

    def __init__(self, *, task: c.Task, num_classes: int | None = None,
                 metrics: tuple[str, ...] = c._NAMES, average: c.Average = "macro",
                 threshold: float = .5, from_logits: bool = False,
                 ignore_index: int | None = None, zero_division: float = float("nan"), beta: float = 1.) -> None:
        c._options(task, num_classes, threshold, from_logits, average, zero_division, beta, ignore_index)
        if not metrics or any(n not in (*c._NAMES, "fbeta", "cohen_kappa") for n in metrics):
            raise ValueError("unknown or empty classification metrics selection")
        if task == "multilabel" and "cohen_kappa" in metrics:
            raise ValueError("cohen_kappa requires binary or multiclass labels")
        self._kwargs = dict(task=task, num_classes=num_classes, threshold=threshold,
                            from_logits=from_logits, ignore_index=ignore_index)
        self.task, self.metrics, self.average = task, tuple(metrics), average
        self.zero_division, self.beta = zero_division, beta
        self._state: Tensor | None = None

    @torch.no_grad()
    def update(self, pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None) -> None:
        state = c._counts(pred, target, mask=mask, sample_weight=sample_weight, **self._kwargs)  # type: ignore[arg-type]
        if self._state is None:
            self._state = state
        else:
            if self._state.shape != state.shape or self._state.device != state.device:
                raise ValueError("class count and device must remain constant across updates")
            self._state += state

    @torch.no_grad()
    def compute(self) -> dict[str, Tensor]:
        # str(NaN) compares equal across processes, unlike NaN itself.
        config = ("classification", tuple(self._kwargs.items()), self.metrics, self.average, str(self.zero_division), self.beta)
        states = gathered(self._state, config)
        counts = torch.stack(states).sum(0).reshape(4, -1)
        return c._scores(counts, self.metrics, self.task, self.average, self.zero_division, self.beta)

    def reset(self) -> None:
        self._state = None


class ClassificationRanking:
    """Pool AP/AUROC and PR/ROC curves across classification samples and ranks.

    Uses the existing ranking engine: exact mode retains observed scores; binned
    mode stores O(C*bins) counts. compute returns a RankingResult with scores
    (1,C) and curves[0][class]; the first axis is the complete pooled dataset.
    Every rank must call compute with the same configuration, including empty
    ranks. Use an unpadded sampler. Supports masks/ignore_index, not weights.
    """

    def __init__(self, *, task: c.Task, num_classes: int | None = None,
                 from_logits: bool = False, ignore_index: int | None = None, bins: int | None = None) -> None:
        from exacta.metrics.ranking import VoxelRanking

        c._options(task, num_classes, .5, from_logits, ignore_index=ignore_index)
        self.task, self.num_classes = task, num_classes
        self.from_logits, self.ignore_index = from_logits, ignore_index
        self._ranking = VoxelRanking(from_logits=from_logits, multilabel=task == "multilabel", bins=bins)

    def update(self, pred: Any, target: Any, *, mask: Any = None) -> None:
        p, t, valid = c._probability_inputs(pred, target, self.task, self.num_classes,
                                           self.from_logits, mask, self.ignore_index)
        self._ranking.update(p, t, mask=valid)

    def compute(self) -> Any:
        from exacta._tabular import check_config

        check_config(("classification_ranking", self.task, self.num_classes, self.from_logits, self.ignore_index))
        return self._ranking.compute()

    def reset(self) -> None:
        self._ranking.reset()
