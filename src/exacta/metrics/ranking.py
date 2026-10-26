"""Case-averaged and dataset-pooled voxel ranking accumulators."""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from exacta._inputs import to_torch
from exacta.functional import ranking as r
from exacta.losses.segmentation import _check_ignore_index
from exacta.metrics.base import PerCaseMetric, _all_gather_cat


class _CaseRanking(PerCaseMetric):
    _field: str

    def __init__(self, *, from_logits: bool = False, multilabel: bool = False,
                 include_background: bool = True, ignore_index: int | None = None,
                 bins: int | None = None) -> None:
        super().__init__()
        if multilabel and ignore_index is not None:
            raise ValueError("multilabel targets use mask instead of ignore_index")
        self._kwargs: dict[str, Any] = dict(from_logits=from_logits, multilabel=multilabel,
                            include_background=include_background,
                            ignore_index=_check_ignore_index(ignore_index, "index"), bins=r._check_bins(bins))

    def update(self, pred: Any, target: Any, *, mask: Any = None) -> Tensor:
        function = r.average_precision if self._field == "average_precision" else r.auroc
        return self._record(function(to_torch(pred), to_torch(target), mask=mask, **self._kwargs))


class AveragePrecision(_CaseRanking):
    """Voxel AP averaged equally over cases (NaNs skipped); see functional.voxel_ranking."""
    _field = "average_precision"


class AUROC(_CaseRanking):
    """Voxel AUROC averaged equally over cases; this is not pooled-dataset AUROC."""
    _field = "auroc"


class VoxelRanking:
    """Pool voxel predictions across updates/ranks before computing PR/ROC, AP and AUROC.

    ``compute`` returns a RankingResult with score shape (1,K), representing the
    pooled dataset. Exact mode retains detached selected scores and binary labels,
    O(total observed voxels × classes); distributed compute gathers each class on
    every rank. Binned mode retains only 2*K*bins int64 counts and all-reduces them,
    with temporary update buffers proportional to the current batch. Both modes
    include empty ranks. All ranks must call compute with identical configuration;
    with NCCL, set each rank's current CUDA device first. Avoid padded evaluation
    samplers that duplicate cases. Update inputs may be mutated afterward.
    """

    _classes: int | None
    _device: torch.device | None
    _hist: Tensor | None
    _chunks: list[list[tuple[Tensor, Tensor]]]

    def __init__(self, *, from_logits: bool = False, multilabel: bool = False,
                 include_background: bool = True, ignore_index: int | None = None,
                 bins: int | None = None) -> None:
        if not isinstance(multilabel, bool):
            raise TypeError("multilabel must be a bool")
        if multilabel and ignore_index is not None:
            raise ValueError("multilabel targets use mask instead of ignore_index")
        self.bins = r._check_bins(bins)
        self._kwargs: dict[str, Any] = dict(from_logits=from_logits, multilabel=multilabel,
                            include_background=include_background,
                            ignore_index=_check_ignore_index(ignore_index, "index"))
        self.reset()

    @torch.no_grad()
    def update(self, pred: Any, target: Any, *, mask: Any = None) -> None:
        data = r._inputs(pred, target, mask=mask, **self._kwargs)
        if self._classes is not None and data.classes != self._classes:
            raise ValueError("class count must remain constant across updates")
        if self._device is not None and data.pred.device != self._device:
            raise ValueError("device must remain constant across updates")
        if self._classes is None:
            self._classes, self._device = data.classes, data.pred.device
            self._chunks = [[] for _ in range(data.classes)]
            if self.bins is not None:
                self._hist = torch.zeros((data.classes, 2, self.bins), dtype=torch.int64, device=data.pred.device)
        for c, (scores, positives) in enumerate(data.rows(None, self.bins)):
            if self.bins is None:
                self._chunks[c].append((scores, positives))
            else:
                assert self._hist is not None
                self._hist[c] += r._histogram(scores, positives, self.bins)

    @torch.no_grad()
    def compute(self) -> r.RankingResult:
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        device = self._device or torch.device("cpu")
        gather_device = device
        classes = self._classes
        if distributed:
            gather_device = (torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl"
                             else torch.device("cpu"))
            if self._device is None:
                device = gather_device
            config = torch.tensor([classes if classes is not None else -1, self.bins or 0,
                                   int(self._kwargs["from_logits"]), int(self._kwargs["multilabel"]),
                                   int(self._kwargs["include_background"])], device=gather_device)
            gathered = [torch.empty_like(config) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, config)
            configs = [x.tolist() for x in gathered]
            counts = {x[0] for x in configs if x[0] >= 0}
            if len(counts) > 1 or any(x[1:] != configs[0][1:] for x in configs):
                raise ValueError("ranking configuration/class counts must agree across ranks")
            classes = counts.pop() if counts else None
        if classes is None:
            raise RuntimeError("VoxelRanking.compute() called before update()")
        ap = torch.empty((1, classes), device=device, dtype=torch.float64)
        auc = torch.empty_like(ap)
        curves = []
        if self.bins is not None:
            hist = (self._hist.to(gather_device).clone() if self._hist is not None else
                    torch.zeros((classes, 2, self.bins), device=gather_device, dtype=torch.int64))
            if distributed:
                dist.all_reduce(hist)
            hist = hist.to(device)
            for c in range(classes):
                curve, ap[0, c], auc[0, c] = r._histogram_curve(hist[c])
                curves.append(curve)
        else:
            for c in range(classes):
                chunks = self._chunks[c] if self._classes is not None else []
                packed = None
                if chunks:
                    scores, positives = (torch.cat([x[j] for x in chunks]) for j in (0, 1))
                    packed = torch.stack((scores.double(), positives.double()), 1)
                packed = _all_gather_cat(packed)
                if packed is None:
                    packed = torch.empty((0, 2), dtype=torch.float64, device=device)
                packed = packed.to(device)
                curve, ap[0, c], auc[0, c] = r._curve(packed[:, 0], packed[:, 1].bool(), None)
                curves.append(curve)
        return r.RankingResult(ap, auc, (tuple(curves),), self.bins)

    def reset(self) -> None:
        """Discard all retained scores or counts."""
        self._classes = None
        self._device = None
        self._hist = None
        self._chunks = []
