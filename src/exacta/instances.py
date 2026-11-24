"""Instance segmentation evaluation from instance label maps.

``exacta.lesions`` derives instances as connected components of semantic masks, which merges
instances that touch (nuclei, cells, glands). Here every case is an instance label map: 0 is
background and every other value one instance, so touching instances stay apart and an
instance need not be connected. Values need not be contiguous; they are renumbered in
increasing order, which also fixes the order that breaks ties below.

With ``P`` the predicted and ``G`` the target instances, ``I(p, g)`` their shared voxels and
``IoU(p, g) = I / |p ∪ g|``:

* Matching (``tp``, ``fp``, ``fn``): pairs whose IoU exceeds ``threshold`` (reaches it with
  ``strict=False``) are matched one-to-one. Above 0.5 with ``strict=True`` (the default, as in
  HoVer-Net's ``get_fast_pq``, COCO panopticapi and the 2018 Data Science Bowl) every instance has
  at most one candidate, so the matching is unique; at exactly 0.5, ``strict=False`` can give a
  prediction two candidates. Below that, ``matching="greedy"`` accepts pairs by decreasing IoU
  (Panoptica's naive matching) and ``"hungarian"`` maximises the summed IoU of the accepted
  pairs (Panoptica's bipartite matching), solved separately on every group of mutually
  overlapping instances (SciPy). HoVer-Net assigns first and thresholds afterwards, and StarDist
  first maximises the number of matches, so below 0.5 their counts can differ.
* ``precision = tp / |P|``, ``recall = tp / |G|``, ``f1 = tp / (tp + (fp + fn) / 2)`` (Kirillov
  et al.'s detection/recognition quality), ``accuracy = tp / (tp + fp + fn)`` (the per-threshold
  score of the 2018 Data Science Bowl, StarDist's ``accuracy``), ``sq`` the mean IoU of matched
  pairs and ``pq = sq · f1`` (Kirillov et al., CVPR 2019).
* ``aji``: Aggregated Jaccard Index (Kumar et al., IEEE TMI 2017), as HoVer-Net's
  ``get_fast_aji``: every target instance takes the prediction of highest IoU with it (largest
  intersection, then lowest value, on ties: HoVer-Net's ``argmax`` over an IoU with a 1e-6 term
  in the denominator); ``Σ I / Σ |p ∪ g|`` over those pairs, with the whole area of unpaired
  target instances, and of predictions no target took, added to the union. A prediction taken
  by several targets counts in each union, the over-penalisation HoVer-Net's AJI+ removes.
* ``aji_plus``: AJI+ (Graham et al., Medical Image Analysis 2019), the same sums over the
  one-to-one pairing of positive IoU that maximises the summed IoU (``get_fast_aji_plus``;
  SciPy). Equal-sum pairings are resolved per overlapping group, so tied optima may differ
  from a solve of the whole image.
* ``seg``: the Cell Tracking Challenge SEG measure (Maška et al., Bioinformatics 2014): each
  target instance ``g`` scores ``IoU(p, g)`` with the prediction covering more than half of it,
  ``I(p, g) > |g| / 2`` (at most one), or 0; ``seg`` is their mean. False positives do not count.

Undefined scores are NaN: everything when a case has no instances at all (``both_empty``),
precision without predictions, recall and ``seg`` without targets. When there are instances
but no match, ``f1``, ``accuracy``, ``sq`` and ``pq`` are 0 (Panoptica, HoVer-Net, StarDist).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import torch
from torch import Tensor

from exacta._inputs import like_input, to_torch
from exacta.metrics.base import _all_gather_cat

Matching = Literal["greedy", "hungarian"]
Aggregate = Literal["cases", "pooled"]

_MATCHED = ("tp", "fp", "fn", "precision", "recall", "f1", "accuracy", "sq", "pq")


@dataclass(frozen=True)
class InstanceScores:
    """Instance scores (see the module docstring). From :func:`evaluate`, one value per case:
    shape (B,), or (B, T) for the matching scores when ``threshold`` is a sequence of T values
    (``aji``, ``aji_plus`` and ``seg`` do not depend on it). From
    :meth:`InstanceMetrics.compute`, the dataset value: shape () or (T,)."""

    tp: Any
    fp: Any
    fn: Any
    precision: Any
    recall: Any
    f1: Any
    accuracy: Any
    sq: Any
    pq: Any
    aji: Any
    aji_plus: Any
    seg: Any


def evaluate(
    pred: Any,
    target: Any,
    *,
    threshold: float | Sequence[float] = 0.5,
    strict: bool = True,
    matching: Matching = "greedy",
    both_empty: float = math.nan,
) -> InstanceScores:
    """Match and score the instances of every case (see the module docstring).

    Args:
        pred, target: Instance label maps of shape (B, H, W) or (B, D, H, W), nonnegative
            integers, 0 for background. Any array library with DLPack (PyTorch, NumPy, ...).
        threshold: IoU threshold of a match, or a sequence of thresholds evaluated together.
        strict: Match when IoU > threshold (default) instead of >=.
        matching: ``"greedy"`` or ``"hungarian"``; they agree when the threshold is at least
            0.5 with ``strict=True``.
        both_empty: Every score of a case without any predicted or target instance.

    Returns:
        :class:`InstanceScores` of float64 arrays in the array library of ``pred``.
    """
    p, t = _check_maps(pred, target)
    thresholds = _check_options(threshold, strict, matching)
    rows = [_case_statistics(p[b], t[b], thresholds, strict, matching) for b in range(p.shape[0])]
    stats = torch.stack(rows) if rows else torch.empty((0, _width(len(thresholds))), dtype=torch.float64)
    scores = _scores(stats, len(thresholds), both_empty)
    single = not isinstance(threshold, Sequence)
    out = {}
    for name, value in scores.items():
        if single and name in _MATCHED:
            value = value[..., 0]
        out[name] = like_input(value.to(p.device), pred)
    return InstanceScores(**out)


def dsb2018(pred: Any, target: Any) -> Any:
    """The 2018 Data Science Bowl score of every case: ``accuracy`` averaged over the IoU
    thresholds 0.50, 0.55, ..., 0.95 (each ``k / 20``), with strict matching (IoU above the
    threshold), as the competition's evaluation page describes it (Caicedo et al., Nature
    Methods 2019). StarDist's ``matching_dataset`` uses IoU >= threshold instead. Cases without
    instances are NaN. Returns a (B,) float64 array."""
    scores = evaluate(pred, target, threshold=[k / 20 for k in range(10, 20)], strict=True)
    return like_input(to_torch(scores.accuracy).mean(-1), pred)


class InstanceMetrics:
    """Accumulate instance statistics over batches (and processes) and score the dataset.

    ``aggregate="cases"`` (default) averages each case's score, skipping NaN, as HoVer-Net and
    PanNuke report image means. ``aggregate="pooled"`` sums the statistics over all cases first:
    PQ is ``Σ IoU / (Σ tp + (Σ fp + Σ fn) / 2)``, as COCO panopticapi and the CoNIC challenge
    pool per class, AJI is ``Σ I / Σ |p ∪ g|`` and SEG the mean over every target instance, the
    Cell Tracking Challenge's dataset SEG. ``tp``, ``fp`` and ``fn`` are dataset totals in both
    modes. For per-class (panoptic) scores, give each class its own accumulator with the
    instances of that class only. CoNIC's mPQ+ then averages the pooled class PQ values, with
    its 1e-6 terms making classes without instances 0 rather than NaN.

    Under ``torch.distributed`` every rank must call ``compute`` with identical settings,
    including ranks without updates; use a sampler that does not repeat cases.
    """

    def __init__(self, *, threshold: float | Sequence[float] = 0.5, strict: bool = True,
                 matching: Matching = "greedy", aggregate: Aggregate = "cases",
                 both_empty: float = math.nan) -> None:
        self._thresholds = _check_options(threshold, strict, matching)
        if aggregate not in ("cases", "pooled"):
            raise ValueError(f"aggregate must be 'cases' or 'pooled', got {aggregate!r}")
        self._single = not isinstance(threshold, Sequence)
        self.strict, self.matching, self.aggregate, self.both_empty = strict, matching, aggregate, both_empty
        self._rows: list[Tensor] = []

    def update(self, pred: Any, target: Any) -> None:
        p, t = _check_maps(pred, target)
        for b in range(p.shape[0]):
            self._rows.append(_case_statistics(p[b], t[b], self._thresholds, self.strict, self.matching))

    def compute(self) -> InstanceScores:
        from exacta._tabular import check_config

        check_config(("instances", self._thresholds, self.strict, self.matching, self.aggregate, str(self.both_empty)))
        stats = _all_gather_cat(torch.stack(self._rows) if self._rows else None)
        if stats is None:
            raise RuntimeError("InstanceMetrics.compute() called before update()")
        count = len(self._thresholds)
        if self.aggregate == "pooled":
            scores = {k: v[0] for k, v in _scores(stats.sum(0, keepdim=True), count, self.both_empty).items()}
        else:
            per_case = _scores(stats, count, self.both_empty)
            scores = {k: v.nanmean(0) for k, v in per_case.items()}
            totals = _scores(stats.sum(0, keepdim=True), count, self.both_empty)
            for name in ("tp", "fp", "fn"):
                scores[name] = totals[name][0]
        if self._single:
            scores = {k: v[0] if k in _MATCHED else v for k, v in scores.items()}
        return InstanceScores(**scores)

    def reset(self) -> None:
        self._rows.clear()


# Per case, one float64 row: for each threshold (tp, fp, fn, Σ IoU of matches), then the number
# of predicted and target instances, AJI and AJI+ intersections and unions, and the SEG sum.
def _width(thresholds: int) -> int:
    return 4 * thresholds + 7


def _scores(stats: Tensor, thresholds: int, both_empty: float) -> dict[str, Tensor]:
    matched = stats[:, : 4 * thresholds].reshape(-1, thresholds, 4)
    tp, fp, fn, iou = matched.unbind(-1)
    n_pred, n_target, aji_i, aji_u, plus_i, plus_u, seg = stats[:, 4 * thresholds:].unbind(-1)
    nan = torch.tensor(math.nan, dtype=torch.float64, device=stats.device)

    def ratio(num: Tensor, den: Tensor, empty: Tensor | float) -> Tensor:
        return torch.where(den > 0, num / torch.where(den > 0, den, 1), empty)

    detections = tp + (fp + fn) / 2
    out = dict(
        tp=tp, fp=fp, fn=fn,
        precision=ratio(tp, tp + fp, nan), recall=ratio(tp, tp + fn, nan),
        f1=ratio(tp, detections, 0.0), accuracy=ratio(tp, tp + fp + fn, 0.0),
        sq=ratio(iou, tp, 0.0), pq=ratio(iou, detections, 0.0),
        aji=ratio(aji_i, aji_u, 0.0), aji_plus=ratio(plus_i, plus_u, 0.0),
        seg=ratio(seg, n_target, nan),
    )
    empty = (n_pred + n_target) == 0
    for name, value in out.items():
        if name not in ("tp", "fp", "fn"):
            mask = empty[:, None] if value.ndim == 2 else empty
            out[name] = torch.where(mask, both_empty, value)
    return out


def _check_maps(pred: Any, target: Any) -> tuple[Tensor, Tensor]:
    p, t = to_torch(pred), to_torch(target)
    if p.shape != t.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(p.shape)} and {tuple(t.shape)}")
    if p.device != t.device:
        raise ValueError("pred and target must be on the same device")
    if p.ndim not in (3, 4):
        raise ValueError(f"instance maps must have shape (B, H, W) or (B, D, H, W), got {tuple(p.shape)}")
    checked = []
    for name, x in (("pred", p), ("target", t)):
        if x.dtype.is_floating_point or x.dtype.is_complex or x.dtype == torch.bool:
            raise TypeError(f"{name} must hold integer instance labels, got {x.dtype}")
        # Unsigned maps (uint16 TIFF masks, StarDist/Cellpose output) are widened to int64:
        # PyTorch lacks min/max for most unsigned types. uint64 is reinterpreted, so values from
        # 2**63 become negative and are rejected below rather than wrapping silently.
        if x.dtype == torch.uint64:
            x = x.view(torch.int64)
        elif x.dtype in (torch.uint16, torch.uint32):
            x = x.to(torch.int64)
        if x.numel() and int(x.min()) < 0:
            raise ValueError(f"{name} instance labels must be nonnegative integers below 2**63 (0 is background)")
        checked.append(x)
    return checked[0], checked[1]


def _check_options(threshold: float | Sequence[float], strict: bool, matching: str) -> tuple[float, ...]:
    values = tuple(threshold) if isinstance(threshold, Sequence) else (threshold,)
    if not values:
        raise ValueError("threshold must not be an empty sequence")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
            raise ValueError(f"thresholds must be numbers in [0, 1], got {value!r}")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a bool")
    if matching not in ("greedy", "hungarian"):
        raise ValueError(f"matching must be 'greedy' or 'hungarian', got {matching!r}")
    return tuple(float(v) for v in values)


def _relabel(labels: Tensor) -> tuple[Tensor, int]:
    """Instance values renumbered 1..n in increasing order (int32), and n."""
    flat = labels.reshape(-1).long()
    if not flat.numel():
        return torch.zeros(labels.shape, dtype=torch.int32, device=labels.device), 0
    top = int(flat.max())
    if top <= 4 * flat.numel() + 1024:  # a lookup table is cheaper than sorting
        present = torch.bincount(flat, minlength=top + 1) > 0
        present[0] = False
        lut = torch.cumsum(present, 0) * present
        count = int(present.sum())
        return lut[flat].to(torch.int32).reshape(labels.shape), count
    values, inverse = torch.unique(flat, return_inverse=True)
    shift = 1 if int(values[0]) != 0 else 0
    return (inverse + shift).to(torch.int32).reshape(labels.shape), values.numel() - (1 - shift)


def _case_statistics(pred: Tensor, target: Tensor, thresholds: tuple[float, ...], strict: bool,
                     matching: str) -> Tensor:
    from exacta.lesions import _overlap

    p, n_pred = _relabel(pred)
    t, n_target = _relabel(target)
    if n_pred > 2**31 - 2 or n_target > 2**31 - 2:
        raise ValueError("too many instances in one case")
    inter, size_pred, size_target = _overlap(p, t, n_pred, n_target)
    # Pairs (prediction j, target i) that meet, with their intersection and IoU, in (j, i) order.
    pairs = [(j, i, count, count / (size_pred[j] + size_target[i] - count)) for (j, i), count in inter.items()]

    row: list[float] = []
    for value in thresholds:
        accepted = [(j, i, iou) for j, i, _, iou in pairs if (iou > value if strict else iou >= value)]
        found = _hungarian(accepted) if matching == "hungarian" else _greedy(accepted)
        tp = len(found)
        row += [tp, n_pred - tp, n_target - tp, sum(iou for _, _, iou in found)]

    # AJI: every target takes its best prediction (IoU, then intersection, then lowest value).
    best: dict[int, tuple[float, int, int]] = {}
    for j, i, count, iou in pairs:
        key = (iou, count, -j)
        if i not in best or key > best[i]:
            best[i] = key
    aji_i = aji_u = 0.0
    taken = set()
    for i, (_, count, minus_j) in best.items():
        j = -minus_j
        taken.add(j)
        aji_i += count
        aji_u += size_pred[j] + size_target[i] - count
    aji_u += sum(size_target[i] for i in range(n_target) if i not in best)
    aji_u += sum(size_pred[j] for j in range(n_pred) if j not in taken)

    # AJI+: one-to-one pairing of positive IoU with the largest summed IoU.
    plus = _hungarian([(j, i, iou) for j, i, _, iou in pairs])
    counts = {(j, i): c for j, i, c, _ in pairs}
    plus_i = float(sum(counts[j, i] for j, i, _ in plus))
    plus_u = sum(size_pred[j] + size_target[i] - counts[j, i] for j, i, _ in plus)
    paired_pred, paired_target = {j for j, _, _ in plus}, {i for _, i, _ in plus}
    plus_u += sum(size_pred[j] for j in range(n_pred) if j not in paired_pred)
    plus_u += sum(size_target[i] for i in range(n_target) if i not in paired_target)

    # SEG: the prediction covering more than half of a target instance (2 I > |g|: exact).
    seg = sum(iou for _, i, count, iou in pairs if 2 * count > size_target[i])
    row += [n_pred, n_target, aji_i, aji_u, plus_i, plus_u, seg]
    return torch.tensor(row, dtype=torch.float64)


def _greedy(candidates: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Accept pairs by decreasing IoU, one-to-one; the stable sort keeps (j, i) order on ties."""
    used_pred: set[int] = set()
    used_target: set[int] = set()
    found = []
    for j, i, iou in sorted(candidates, key=lambda c: c[2], reverse=True):
        if j not in used_pred and i not in used_target:
            used_pred.add(j)
            used_target.add(i)
            found.append((j, i, iou))
    return found


def _hungarian(candidates: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """One-to-one pairs with the largest summed IoU, solved per connected group of candidates."""
    if not candidates:
        return []
    # A candidate graph with at most one pair per prediction and per target is already optimal.
    preds = [j for j, _, _ in candidates]
    targets = [i for _, i, _ in candidates]
    if len(set(preds)) == len(preds) and len(set(targets)) == len(targets):
        return list(candidates)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise ImportError("matching='hungarian' and AJI+ need SciPy") from error
    import numpy as np

    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for j, i, _ in candidates:
        a, b = find(("p", j)), find(("t", i))
        if a != b:
            parent[a] = b
    groups: dict[tuple[str, int], list[tuple[int, int, float]]] = {}
    for c in candidates:
        groups.setdefault(find(("p", c[0])), []).append(c)
    found = []
    for group in groups.values():
        if len(group) == 1:
            found += group
            continue
        rows = sorted({i for _, i, _ in group})
        cols = sorted({j for j, _, _ in group})
        row_of = {v: k for k, v in enumerate(rows)}
        col_of = {v: k for k, v in enumerate(cols)}
        score = np.zeros((len(rows), len(cols)))
        for j, i, iou in group:
            score[row_of[i], col_of[j]] = iou
        for a, b in zip(*linear_sum_assignment(score, maximize=True)):
            if score[a, b] > 0:
                found.append((cols[b], rows[a], float(score[a, b])))
    return found


