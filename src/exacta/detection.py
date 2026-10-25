"""Confidence-ranked lesion detection: FROC and non-interpolated average precision."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from exacta import components
from exacta._inputs import InputFormat, to_torch, validate
from exacta.lesions import _overlap
from exacta.metrics.base import _all_gather_cat


@dataclass(frozen=True)
class DetectionCurve:
    """One class's dataset-level detection curve (CPU tensors, descending thresholds).

    The first threshold is +inf (no predictions). Equal confidences form one step.
    ``average_precision`` is sum(delta_recall * precision), without an interpolated
    precision envelope. Undetected target lesions remain in the recall denominator.
    No targets: recall/AP are NaN. No predictions with targets: AP=0.
    """
    thresholds: Tensor
    true_positives: Tensor
    false_positives: Tensor
    precision: Tensor
    recall: Tensor
    false_positives_per_case: Tensor
    average_precision: Tensor
    num_targets: int
    num_cases: int

    def sensitivity_at(self, fp_per_case: float | Sequence[float] | Tensor) -> Tensor:
        """Best achieved recall at each FP/case budget; stepwise, no interpolation."""
        budgets = torch.as_tensor(fp_per_case, dtype=torch.float64, device="cpu")
        if not bool(torch.isfinite(budgets).all() & (budgets >= 0).all()):
            raise ValueError("FP/case budgets must be finite and nonnegative")
        if not self.num_targets:
            return torch.full_like(budgets, torch.nan)
        allowed = self.false_positives_per_case <= budgets[..., None]
        return torch.where(allowed, self.recall, 0).amax(-1)


@dataclass(frozen=True)
class DetectionResult:
    """One curve per selected class; class IDs correspond to segmentation channels."""
    class_ids: tuple[int, ...]
    curves: tuple[DetectionCurve, ...]

    @property
    def average_precision(self) -> Tensor:
        return torch.stack([curve.average_precision for curve in self.curves])


def _curve(scores: Tensor, hits: Tensor, targets: int, cases: int) -> DetectionCurve:
    order = scores.argsort(descending=True, stable=True)
    scores, hits = scores[order], hits[order].long()
    end = torch.ones(scores.shape, dtype=torch.bool)
    end[:-1] = scores[:-1] != scores[1:]
    tp = torch.cat((torch.zeros(1, dtype=torch.int64), hits.cumsum(0)[end]))
    fp = torch.cat((torch.zeros(1, dtype=torch.int64), (1-hits).cumsum(0)[end]))
    thresholds = torch.cat((torch.tensor([torch.inf], dtype=torch.float64), scores[end]))
    precision = torch.where(tp+fp > 0, tp.double() / (tp+fp).clamp_min(1), 1)
    recall = tp.double() / targets if targets else torch.full_like(precision, torch.nan)
    ap = ((recall[1:]-recall[:-1]) * precision[1:]).sum() if targets else torch.tensor(torch.nan, dtype=torch.float64)
    return DetectionCurve(thresholds, tp, fp, precision, recall, fp.double()/cases, ap, targets, cases)


class LesionDetection:
    """Stream lesion confidences/matches and compute dataset AP/FROC.

    ``update(pred, target, confidence)`` takes hard segmentations and a floating
    confidence map with the prediction's shape, in [0,1]. Connected components are
    candidates; their confidence is max (default) or mean over component voxels.
    Candidate extraction is fixed before sweeping confidence; components are not
    re-extracted at each operating threshold.

    Per case/class, match predictions in decreasing confidence to the highest-IoU
    still-unmatched target if IoU >= ``iou_threshold``. Each target/prediction is used
    once. Confidence ties use prediction component ID; IoU ties use target component
    ID (both raster order). Curves group confidence ties and expose no within-tie
    operating points. This is an explicit greedy protocol, not a challenge-specific
    PI-CAI or BraTS scoring implementation.

    Volumes use existing C++/Triton component/overlap kernels. Matching and sorting
    work on small CPU lesion tables. Only scores, hit flags and counts are retained.
    ``compute`` gathers these small tables across distributed ranks, including ranks
    without updates; all ranks must call with identical settings. Include every case,
    including lesion-free scans, and avoid duplicate distributed-sampler padding.
    """

    _targets: Tensor | None
    _cases: int
    _events: list[Tensor]

    def __init__(self, *, connectivity: int, iou_threshold: float = 0.1,
                 confidence_reduction: Literal["max", "mean"] = "max",
                 input_format: InputFormat = "index", num_classes: int | None = None,
                 include_background: bool = False, backend: components.Backend = "auto") -> None:
        if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must lie in (0, 1]")
        if confidence_reduction not in ("max", "mean"):
            raise ValueError("confidence_reduction must be 'max' or 'mean'")
        self.connectivity, self.iou_threshold = connectivity, iou_threshold
        self.confidence_reduction, self.backend = confidence_reduction, backend
        self._kwargs = dict(input_format=input_format, num_classes=num_classes, include_background=include_background)
        self._first = 0 if include_background else 1
        self.reset()

    @torch.no_grad()
    def update(self, pred: Any, target: Any, confidence: Any) -> None:
        seg = validate(to_torch(pred), to_torch(target), **self._kwargs)  # type: ignore[arg-type]
        conf = to_torch(confidence)
        if conf.shape != seg.pred.shape or conf.device != seg.pred.device:
            raise ValueError("confidence must have the prediction's shape and device")
        if not conf.is_floating_point() or not bool(torch.isfinite(conf).all() & (conf >= 0).all() & (conf <= 1).all()):
            raise ValueError("confidence must be finite floating-point values in [0,1]")
        classes = len(seg.classes)
        if self._targets is not None and len(self._targets) != classes:
            raise ValueError("selected class counts must agree across updates")
        added_targets = torch.zeros(classes, dtype=torch.float64)
        events = []
        for b in range(seg.batch):
            for col, cls in enumerate(seg.classes):
                p, t = seg.masks(b, cls)
                lp, npred = components._label(p, self.connectivity, self.backend)
                lt, ntrue = components._label(t, self.connectivity, self.backend)
                added_targets[col] += ntrue
                overlaps, sp, st = _overlap(lp, lt, npred, ntrue)
                c = conf[b, cls] if seg.one_hot else conf[b]
                values = torch.zeros(npred + 1, dtype=torch.float64, device=c.device)
                if self.confidence_reduction == "max":
                    values.scatter_reduce_(0, lp.flatten().long(), c.flatten().double(), reduce="amax", include_self=True)
                else:
                    values.scatter_add_(0, lp.flatten().long(), c.flatten().double())
                    values[1:] /= torch.tensor(sp, dtype=torch.float64, device=c.device).clamp_min(1)
                scores = values[1:].cpu().tolist()
                candidates: dict[int, list[tuple[float, int]]] = {}
                for (i, j), intersection in overlaps.items():
                    iou = intersection / (sp[i] + st[j] - intersection)
                    if iou >= self.iou_threshold:
                        candidates.setdefault(i, []).append((iou, j))
                used = set()
                for i in sorted(range(npred), key=lambda i: (-scores[i], i)):
                    options = [(iou, j) for iou, j in candidates.get(i, []) if j not in used]
                    if options:
                        _, matched = min(options, key=lambda item: (-item[0], item[1]))
                        used.add(matched)
                    events.append((col, scores[i], int(bool(options))))
        # Update atomically only after all input/case validation and matching succeeds.
        self._cases += seg.batch
        self._targets = added_targets if self._targets is None else self._targets + added_targets
        if events:
            self._events.append(torch.tensor(events, dtype=torch.float64))

    def compute(self) -> DetectionResult:
        stats = None if self._targets is None else torch.cat((torch.tensor([self._cases], dtype=torch.float64), self._targets))[None]
        totals = _all_gather_cat(stats)
        if totals is None or not bool(totals[:, 0].sum()):
            raise RuntimeError("LesionDetection.compute() needs at least one case")
        totals = totals.cpu().sum(0)
        gathered = _all_gather_cat(torch.cat(self._events) if self._events else None)
        events = torch.empty((0, 3), dtype=torch.float64) if gathered is None else gathered.cpu()
        curves = []
        for col, targets in enumerate(totals[1:].tolist()):
            selected = events[events[:, 0] == col]
            curves.append(_curve(selected[:, 1], selected[:, 2], int(targets), int(totals[0])))
        return DetectionResult(tuple(range(self._first, self._first+len(curves))), tuple(curves))

    def reset(self) -> None:
        self._targets = None
        self._cases = 0
        self._events = []


def lesion_detection(pred: Any, target: Any, confidence: Any, *, connectivity: int,
                     iou_threshold: float = 0.1, confidence_reduction: Literal["max", "mean"] = "max",
                     input_format: InputFormat = "index", num_classes: int | None = None,
                     include_background: bool = False, backend: components.Backend = "auto") -> DetectionResult:
    """Compute AP/FROC for one batch treated as a dataset; see LesionDetection."""
    metric = LesionDetection(connectivity=connectivity, iou_threshold=iou_threshold,
                             confidence_reduction=confidence_reduction, input_format=input_format,
                             num_classes=num_classes, include_background=include_background, backend=backend)
    metric.update(pred, target, confidence)
    return metric.compute()
