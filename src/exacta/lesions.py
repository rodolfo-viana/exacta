"""Lesion-wise (instance) evaluation.

The connected components of prediction and target are the lesions. They are matched to each other,
and the matching gives detection counts and per-lesion scores. How lesions are matched is where
tools differ most, so ``matching`` picks the strategy and each one reproduces a reference:

* ``"greedy"``: overlapping (prediction, target) pairs are sorted by ``score`` (IoU or Dice), best
  first, and accepted one-to-one while the score reaches ``threshold`` (exceeds it with
  ``strict=True``). Panoptica's ``NaiveThresholdMatching``. With IoU above 0.5 no instance can
  have two candidates, so the order does not matter there.
* ``"hungarian"``: the one-to-one assignment with the largest total score among pairs that reach
  the threshold. Panoptica's ``MaxBipartiteMatching``; needs SciPy.
* ``"merge"``: as ``"greedy"``, but a matched target also takes further predictions whenever their
  union scores higher, so a lesion predicted in pieces is merged. Panoptica's
  ``MaximizeMergeMatching``.
* ``"overlap"``: every predicted component that touches a target lesion grown by ``dilation``
  steps matches it, many-to-many; target lesions whose grown regions meet are first merged into
  one. The BraTS 2023 lesion-wise evaluation (see :func:`brats2023`).

Ties between equal scores are broken by component number, prediction first (Panoptica's order);
components are numbered as ``scipy.ndimage.label`` numbers them (:func:`exacta.components.label`).

Scores, each a (B, C) array:

* ``tp``, ``fp``, ``fn``: matched target lesions, unmatched predicted components, unmatched
  target lesions. ``precision``, ``recall``, and ``f1 = tp / (tp + (fp + fn) / 2)``, which is
  Panoptica's recognition quality (RQ).
* ``sq_iou``, ``sq_dice``: mean IoU or Dice of each matched target lesion with its prediction(s)
  (Panoptica's segmentation quality); ``pq_iou``, ``pq_dice`` = SQ x F1 (panoptic quality). As in
  Panoptica, F1, SQ and PQ are 0 when nothing is matched and NaN when there are no lesions at all.
* ``lesion_dice``: the sum of the Dice of every target lesion with its prediction(s) (0 when
  missed), divided by the number of target lesions plus false positives (BraTS 2023).
* ``lesion_hd95`` (when ``spacing`` is given): the same average of each target lesion's HD95,
  with ``missed_distance`` for every missed lesion and false positive (BraTS uses 374).

Target lesions of volume ``min_target_volume`` or less are left out of every score (BraTS drops
small lesions); predictions that touch them are not false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from exacta import components
from exacta._inputs import InputFormat, as_spacing, like_input, to_torch, validate
from exacta.functional.surface import Backend, Surface, hausdorff_distance

Matching = Literal["greedy", "hungarian", "merge", "overlap"]
Score = Literal["iou", "dice"]


@dataclass(frozen=True)
class LesionScores:
    """Lesion-wise scores, each a (B, C) float64 array (see the module docstring)."""

    tp: Any
    fp: Any
    fn: Any
    precision: Any
    recall: Any
    f1: Any
    sq_iou: Any
    sq_dice: Any
    pq_iou: Any
    pq_dice: Any
    lesion_dice: Any
    lesion_hd95: Any  # NaN throughout when no spacing is given


def evaluate(
    pred: Any,
    target: Any,
    *,
    matching: Matching,
    connectivity: int,
    score: Score = "iou",
    threshold: float = 0.5,
    strict: bool = False,
    dilation: int = 0,
    dilation_connectivity: int | None = None,
    min_target_volume: float = 0.0,
    spacing: Any = None,
    surface: Surface = "voxels",
    missed_distance: float = math.inf,
    both_empty: float = math.nan,
    both_empty_distance: float = math.nan,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    backend: Backend = "auto",
) -> LesionScores:
    """Match the lesions of every case and class and score them (see the module docstring).

    Args:
        pred, target: Hard segmentations, as in :func:`exacta.functional.dice`.
        matching: ``"greedy"``, ``"hungarian"``, ``"merge"`` or ``"overlap"``.
        connectivity: Neighbourhood of the connected components, as in
            :func:`exacta.components.label` (BraTS: 3, i.e. 26 neighbours in 3-D).
        score: Pair score used for matching (not ``"overlap"``): ``"iou"`` or ``"dice"``.
        threshold, strict: A pair can match when its score is >= ``threshold`` (> if ``strict``).
        dilation: ``"overlap"`` only: steps of binary dilation of the target lesions before
            matching, with the ``dilation_connectivity`` neighbourhood (BraTS: 3 or 1 steps,
            connectivity 2).
        min_target_volume: Target lesions of at most this volume are ignored, in units of
            ``spacing`` cubed (or voxels without spacing).
        spacing: Voxel size, as in :func:`exacta.functional.hd95`; needed for ``lesion_hd95``.
        surface: Surface representation of ``lesion_hd95``, as in :func:`exacta.functional.hd95`
            (BraTS uses ``"surfels"``).
        missed_distance: HD95 counted for a missed lesion or a false positive (BraTS: 374).
        both_empty, both_empty_distance: ``lesion_dice`` and ``lesion_hd95`` when there are no
            lesions at all (BraTS: 1 and 0).
        input_format, num_classes, include_background: As in :func:`exacta.functional.dice`.
        backend: Kernels for components and distances, as in :func:`exacta.functional.hd95`.
    """
    if matching not in ("greedy", "hungarian", "merge", "overlap"):
        raise ValueError(f"matching must be 'greedy', 'hungarian', 'merge' or 'overlap', got {matching!r}")
    if score not in ("iou", "dice"):
        raise ValueError(f"score must be 'iou' or 'dice', got {score!r}")
    if dilation < 0 or (dilation and matching != "overlap"):
        raise ValueError("dilation is a non-negative number of steps, used only with matching='overlap'")
    if dilation and dilation_connectivity is None:
        raise ValueError("dilation needs dilation_connectivity, the neighbourhood of each step")
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )
    spacings = as_spacing(spacing, batch=seg.batch, ndim=seg.ndim) if spacing is not None else None
    shape = (seg.batch, len(seg.classes))
    out = {f.name: torch.full(shape, math.nan, dtype=torch.float64) for f in fields(LesionScores)}
    for b in range(seg.batch):
        present, lo, hi = (x.tolist() for x in seg.extents(b))
        spatial = seg.pred.shape[seg.pred.ndim - seg.ndim :]
        for k, cls in enumerate(seg.classes):
            # Work inside the box of both masks, widened by the dilation steps so that dilation is
            # unaffected (outside the image, scipy.ndimage.binary_dilation sees background too).
            if present[0][cls] or present[1][cls]:
                ends = [
                    (min(l for l, h in pairs if h >= 0), max(h for l, h in pairs if h >= 0))
                    for pairs in zip(*[zip(lo[r][cls], hi[r][cls]) for r in (0, 1) if present[r][cls]])
                ]
                box = tuple(
                    slice(max(first - dilation, 0), min(last + dilation + 1, n))
                    for (first, last), n in zip(ends, spatial)
                )
            else:
                box = tuple(slice(0, 1) for _ in spatial)  # no lesions: any small region will do
            p, t = seg.masks(b, cls, box)
            values = _evaluate_pair(
                p,
                t,
                matching=matching,
                connectivity=connectivity,
                score=score,
                threshold=threshold,
                strict=strict,
                dilation=dilation,
                dilation_connectivity=dilation_connectivity or 1,
                min_target_volume=min_target_volume,
                spacing=spacings[b] if spacings is not None else None,
                surface=surface,
                missed_distance=missed_distance,
                both_empty=both_empty,
                both_empty_distance=both_empty_distance,
                backend=backend,
            )
            for name, value in values.items():
                out[name][b, k] = value
    return LesionScores(**{name: like_input(value.to(seg.pred.device), pred) for name, value in out.items()})


def brats2023(
    pred: Any,
    target: Any,
    *,
    challenge: Literal["GLI", "SSA", "MEN", "PED", "MET"],
    spacing: Any,
    input_format: InputFormat = "one-hot",
    num_classes: int | None = None,
    include_background: bool = True,
    backend: Backend = "auto",
) -> LesionScores:
    """The BraTS 2023 lesion-wise evaluation of every case and class (or region).

    Matches as BraTS-2023-Metrics (``get_LesionWiseScores``) does: 26-connected components,
    target lesions dilated with the 18-neighbourhood by 3 steps (1 for MEN and MET) and merged
    where their dilations meet, every predicted component touching a dilated lesion matched to
    it, lesions of 50 mm^3 or less (2 for MET) ignored, HD95 on DeepMind's surface elements with
    374 for misses and false positives, and Dice 1 / HD95 0 when there are no lesions. Its
    ``LesionWise_Score_Dice`` is ``lesion_dice``, ``LesionWise_Score_HD95`` is ``lesion_hd95``
    and ``Num_TP/FP/FN`` are ``tp/fp/fn``.

    BraTS scores the regions WT, TC and ET, not the labels; :func:`brats_regions` builds them as
    one-hot channels, which is why ``input_format`` defaults to ``"one-hot"`` here.
    """
    if challenge not in ("GLI", "SSA", "MEN", "PED", "MET"):
        raise ValueError(f"challenge must be 'GLI', 'SSA', 'MEN', 'PED' or 'MET', got {challenge!r}")
    return evaluate(
        pred,
        target,
        matching="overlap",
        connectivity=3,
        dilation=1 if challenge in ("MEN", "MET") else 3,
        dilation_connectivity=2,
        min_target_volume=2.0 if challenge == "MET" else 50.0,
        spacing=spacing,
        surface="surfels",
        missed_distance=374.0,
        both_empty=1.0,
        both_empty_distance=0.0,
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
        backend=backend,
    )


def brats_regions(labels: Any) -> Any:
    """BraTS regions as one-hot channels (B, 3, *spatial): whole tumour (labels 1, 2, 3), tumour
    core (1, 3) and enhancing tumour (3), from (B, *spatial) label maps."""
    x = to_torch(labels)
    regions = torch.stack([(x == 1) | (x == 2) | (x == 3), (x == 1) | (x == 3), x == 3], dim=1)
    return like_input(regions, labels)


def _evaluate_pair(
    p: Tensor,
    t: Tensor,
    *,
    matching: Matching,
    connectivity: int,
    score: Score,
    threshold: float,
    strict: bool,
    dilation: int,
    dilation_connectivity: int,
    min_target_volume: float,
    spacing: tuple[float, ...] | None,
    surface: Surface,
    missed_distance: float,
    both_empty: float,
    both_empty_distance: float,
    backend: Backend,
) -> dict[str, float]:
    pred_labels, n_pred = components._label(p, connectivity, "auto")
    if matching == "overlap":
        # Growing a target lesion and taking its connected component of the grown target is the
        # same thing (the grown target is the union of the grown lesions), so one dilation and one
        # labelling give every lesion's search region, and lesions whose regions meet are merged.
        grown = _dilate(t, dilation, dilation_connectivity) if dilation else t
        regions, n_target = components._label(grown, connectivity, "auto")
        target_labels = torch.where(t, regions, 0)
    else:
        target_labels, n_target = components._label(t, connectivity, "auto")
        regions = target_labels

    # Contingency tables: voxels of each predicted component in each target lesion (and region).
    inter = _pairs(pred_labels, target_labels, n_pred, n_target)
    touch = inter if matching != "overlap" else _pairs(pred_labels, regions, n_pred, n_target)
    size_pred = torch.bincount(pred_labels.reshape(-1).long(), minlength=n_pred + 1)[1:].tolist()
    size_target = torch.bincount(target_labels.reshape(-1).long(), minlength=n_target + 1)[1:].tolist()
    inter_list, touch_list = inter.tolist(), touch.tolist()

    if matching == "overlap":
        matches = [[j for j in range(n_pred) if touch_list[j][i] > 0] for i in range(n_target)]
    else:
        matches = _match(matching, inter_list, size_pred, size_target, score, threshold, strict)

    voxel = math.prod(spacing) if spacing is not None else 1.0
    kept = [i for i in range(n_target) if size_target[i] * voxel > min_target_volume]
    matched_preds = {j for found in matches for j in found}
    fp = n_pred - len(matched_preds)
    tp = sum(1 for i in kept if matches[i])
    fn = len(kept) - tp

    dice, iou = {}, {}
    for i in kept:
        overlap = sum(inter_list[j][i] for j in matches[i])
        predicted = sum(size_pred[j] for j in matches[i])
        dice[i] = 2.0 * overlap / (size_target[i] + predicted)
        iou[i] = overlap / (size_target[i] + predicted - overlap)

    instances = n_pred + len(kept) > 0
    f1 = (tp / (tp + 0.5 * fp + 0.5 * fn) if tp else 0.0) if instances else math.nan
    sq_iou = (sum(iou[i] for i in kept if matches[i]) / tp if tp else 0.0) if instances else math.nan
    sq_dice = (sum(dice[i] for i in kept if matches[i]) / tp if tp else 0.0) if instances else math.nan
    lesions = len(kept) + fp
    values = dict(
        tp=float(tp),
        fp=float(fp),
        fn=float(fn),
        precision=tp / (tp + fp) if tp + fp else math.nan,
        recall=tp / (tp + fn) if tp + fn else math.nan,
        f1=f1,
        sq_iou=sq_iou,
        sq_dice=sq_dice,
        pq_iou=sq_iou * f1,
        pq_dice=sq_dice * f1,
        lesion_dice=sum(dice.values()) / lesions if lesions else both_empty,
        lesion_hd95=math.nan,
    )
    if spacing is not None:
        total = fp * missed_distance
        for i in kept:
            if not matches[i]:
                total += missed_distance
                continue
            lesion = target_labels == i + 1
            found = torch.isin(pred_labels, torch.tensor([j + 1 for j in matches[i]], device=p.device))
            hd = hausdorff_distance(
                found[None, None], lesion[None, None], spacing=spacing, percentile=95.0, surface=surface,
                input_format="one-hot", backend=backend,
            )[0, 0].item()  # fmt: skip
            total += missed_distance if math.isinf(hd) else hd
        values["lesion_hd95"] = total / lesions if lesions else both_empty_distance
    return values


def _pairs(a: Tensor, b: Tensor, na: int, nb: int) -> Tensor:
    """(na, nb) voxel counts of each label of ``a`` (1..na) within each label of ``b`` (1..nb)."""
    keys = a.reshape(-1).long() * (nb + 1) + b.reshape(-1).long()
    return torch.bincount(keys, minlength=(na + 1) * (nb + 1)).view(na + 1, nb + 1)[1:, 1:]


def _pair_score(overlap: int, size_target: int, size_pred: int, score: Score) -> float:
    """IoU or Dice of a pair, with Panoptica's operation order (``a + b`` first)."""
    if score == "iou":
        return float(overlap) / (float(size_target) + float(size_pred) - float(overlap))
    return 2.0 * float(overlap) / (float(size_target) + float(size_pred))


def _beats(value: float, threshold: float, strict: bool) -> bool:
    return value > threshold if strict else value >= threshold


def _match(
    matching: Matching,
    inter: list[list[int]],
    size_pred: list[int],
    size_target: list[int],
    score: Score,
    threshold: float,
    strict: bool,
) -> list[list[int]]:
    """Predicted components matched to each target lesion (0-based), one-to-one or merged."""
    n_target = len(size_target)
    # Overlapping pairs in (prediction, target) order, then by score, best first; the sort is
    # stable, so ties keep the component order.
    candidates = [
        (_pair_score(inter[j][i], size_target[i], size_pred[j], score), j, i)
        for j in range(len(size_pred))
        for i in range(n_target)
        if inter[j][i] > 0
    ]
    candidates.sort(key=lambda c: c[0], reverse=True)
    matches: list[list[int]] = [[] for _ in range(n_target)]

    if matching == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:
            raise ImportError("matching='hungarian' needs SciPy") from error
        import numpy as np

        if not candidates:
            return matches
        cost = np.full((n_target, len(size_pred)), 1.0 + 1e-6)  # Panoptica's cost for no match
        for value, j, i in candidates:
            if _beats(value, threshold, strict):
                cost[i, j] = 1.0 - value
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1.0:
                matches[int(i)].append(int(j))
        return matches

    used: set[int] = set()
    best: dict[int, float] = {}
    for value, j, i in candidates:
        if j in used:
            continue
        if matches[i]:
            if matching != "merge":
                continue
            members = matches[i] + [j]
            overlap = sum(inter[m][i] for m in members)
            merged = _pair_score(overlap, size_target[i], sum(size_pred[m] for m in members), score)
            if merged > best[i]:
                matches[i].append(j)
                used.add(j)
                best[i] = merged
        elif _beats(value, threshold, strict):
            matches[i].append(j)
            used.add(j)
            best[i] = value
    return matches


def _dilate(mask: Tensor, steps: int, connectivity: int) -> Tensor:
    """Binary dilation by ``steps`` steps of the ``connectivity`` neighbourhood, with nothing
    beyond the image (``scipy.ndimage.binary_dilation`` with ``iterations=steps``)."""
    ndim = mask.ndim
    offsets = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * ndim, indexing="ij"))
    structure = (offsets.abs().sum(0) <= connectivity).to(torch.float32)
    conv = F.conv3d if ndim == 3 else F.conv2d if ndim == 2 else F.conv1d
    kernel = structure[None, None].to(mask.device)
    x = mask.to(torch.float32)[None, None]
    for _ in range(steps):
        x = (conv(x, kernel, padding=1) > 0).to(torch.float32)
    return x[0, 0] > 0
