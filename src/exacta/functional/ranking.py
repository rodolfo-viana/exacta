"""Voxel ranking with grouped ties and explicitly approximate fixed-width bins."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterator

import torch
from torch import Tensor

from exacta import _multilabel
from exacta._inputs import like_input, to_torch
from exacta.losses.segmentation import _check_ignore_index, _check_index_target, _check_mask, _prepare_masked_input


@dataclass(frozen=True)
class RankingCurve:
    """One case/class curve, including an initial 'predict none' point at +inf.

    Thresholds decrease; prediction is score >= threshold. Precision at the initial
    point is 1. Recall/TPR are NaN without positives, FPR is NaN without negatives.
    Exact thresholds are probabilities, raw binary/multilabel logits, or multiclass
    log-softmax values, according to the input. Binned thresholds are probabilities.
    Arrays remain in the prediction's array library/device.
    """
    thresholds: Any
    precision: Any
    recall: Any
    fpr: Any
    tpr: Any


@dataclass(frozen=True)
class RankingResult:
    """AP and AUROC (B,K), plus curves indexed as ``curves[case][class]``.

    AP is the sum of recall increments times precision (not trapezoidal PR area).
    No positives gives NaN AP; all-positive observed targets give AP=1. AUROC is
    NaN unless both outcomes occur. Empty inputs give NaN scores. ``bins=None``
    is exact; otherwise scores are quantized to equal-width probability bins,
    with half credit for positive/negative pairs in the same bin in AUROC.
    """
    average_precision: Any
    auroc: Any
    curves: tuple[tuple[RankingCurve, ...], ...]
    bins: int | None


def _check_bins(bins: int | None) -> int | None:
    if bins is not None and (isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 2):
        raise ValueError("bins must be None (exact) or an integer >= 2")
    return None if bins is None else int(bins)


@dataclass(frozen=True)
class _RankingInput:
    pred: Tensor
    target: Tensor
    valid: Tensor
    multilabel: bool
    first: int
    from_logits: bool

    @property
    def classes(self) -> int:
        return self.pred.shape[1] - self.first

    def rows(self, case: int | None, bins: int | None) -> Iterator[tuple[Tensor, Tensor]]:
        p, t, v = self.pred, self.target, self.valid
        if case is not None:
            p, t, v = p[case:case+1], t[case:case+1], v[case:case+1]
        categorical = not self.multilabel and p.shape[1] > 1
        # Log-softmax preserves ranks even when tiny probabilities would underflow.
        logp = p.double().log_softmax(1) if categorical and self.from_logits else None
        for c in range(self.first, p.shape[1]):
            keep = v[:, c if v.shape[1] > 1 else 0] if self.multilabel else v
            score = p[:, c]
            if self.from_logits:
                if categorical:
                    assert logp is not None
                    score = logp[:, c]
                    if bins is not None:
                        score = score.exp()
                elif bins is not None:
                    score = score.double().sigmoid()
            positive = t[:, c] == 1 if self.multilabel else t == (c if categorical else 1)
            yield score[keep], positive[keep]


def _inputs(pred: Any, target: Any, *, from_logits: bool, multilabel: bool,
            include_background: bool, ignore_index: int | None, mask: Any) -> _RankingInput:
    if not isinstance(multilabel, bool):
        raise TypeError("multilabel must be a bool")
    if multilabel:
        _multilabel.check_options(True, "none", ignore_index)
        p, t, m = _multilabel.inputs(pred, target, mask)
        v = torch.ones_like(t, dtype=torch.bool) if m is None else m
        bad_target = (((t != 0) & (t != 1)) & v).any()
        observed = v
    else:
        p, t = to_torch(pred).detach(), to_torch(target).detach()
        m = _check_mask(None if mask is None else to_torch(mask), p)
        p, t, valid = _prepare_masked_input(p, t, "index", _check_ignore_index(ignore_index, "index"), m)
        _check_index_target(t, p)
        if p.ndim not in (4, 5):
            raise ValueError("pred must have shape (B,C,H,W) or (B,C,D,H,W)")
        v = torch.ones_like(t, dtype=torch.bool) if valid is None else valid
        bad_target = (((t < 0) | (t >= max(2, p.shape[1]))) & v).any()
        observed = v[:, None]
    bad_finite = ((~torch.isfinite(p)) & observed).any()
    bad_range = (((p < 0) | (p > 1)) & observed).any() if not from_logits else bad_finite.new_zeros(())
    bad_sum = bad_finite.new_zeros(())
    if not from_logits and not multilabel and p.shape[1] > 1:
        bad_sum = (((p.double().sum(1) - 1).abs() > 1e-3) & v).any()
    errors = torch.stack((bad_target, bad_finite, bad_range, bad_sum)).tolist()
    if errors[0]:
        raise ValueError("included targets must be binary multilabel values or valid class indices")
    if errors[1]:
        raise ValueError("included predictions must be finite")
    if errors[2]:
        raise ValueError("probabilities must lie in [0, 1]")
    if errors[3]:
        raise ValueError("multiclass probabilities must sum to one")
    first = 0 if include_background else 1
    if first and p.shape[1] < 2:
        raise ValueError("include_background=False needs at least two channels; C=1 already denotes foreground")
    return _RankingInput(p, t, v, multilabel, first, from_logits)


def _histogram(scores: Tensor, positives: Tensor, bins: int) -> Tensor:
    index = (scores.double() * bins).long().clamp_(0, bins-1)
    index = index + positives.long() * bins
    return torch.bincount(index, minlength=2*bins).reshape(2, bins)


def _curve(scores: Tensor, positives: Tensor, bins: int | None) -> tuple[RankingCurve, Tensor, Tensor]:
    if bins is not None:
        return _histogram_curve(_histogram(scores, positives, bins))
    ranked, order = scores.sort(descending=True, stable=True)
    fg = positives[order]
    tp = fg.cumsum(0)
    fp = (~fg).cumsum(0)
    end = torch.cat((ranked[1:] != ranked[:-1], torch.ones_like(ranked[:1], dtype=torch.bool)))
    return _finish_curve(ranked[end].double(), tp[end], fp[end])


def _histogram_curve(hist: Tensor) -> tuple[RankingCurve, Tensor, Tensor]:
    bins = hist.shape[-1]
    fp, tp = hist.flip(-1).cumsum(-1).unbind(0)
    thresholds = torch.arange(bins-1, -1, -1, device=hist.device, dtype=torch.float64) / bins
    return _finish_curve(thresholds, tp, fp)


def _finish_curve(thresholds: Tensor, tp: Tensor, fp: Tensor) -> tuple[RankingCurve, Tensor, Tensor]:
    zero = thresholds.new_zeros(1)
    tp, fp = torch.cat((zero, tp.double())), torch.cat((zero, fp.double()))
    thresholds = torch.cat((zero + torch.inf, thresholds))
    npos, nneg = tp[-1], fp[-1]
    precision = torch.where(tp + fp > 0, tp / (tp + fp).clamp_min(1), 1)
    recall = torch.where(npos > 0, tp / npos.clamp_min(1), torch.nan)
    fpr = torch.where(nneg > 0, fp / nneg.clamp_min(1), torch.nan)
    ap = torch.where(npos > 0, ((recall[1:] - recall[:-1]) * precision[1:]).sum(), torch.nan)
    auc = torch.where((npos > 0) & (nneg > 0),
                      ((fpr[1:] - fpr[:-1]) * (recall[1:] + recall[:-1]) * .5).sum(), torch.nan)
    return RankingCurve(thresholds, precision, recall, fpr, recall), ap, auc


def _convert_curve(curve: RankingCurve, pred: Any) -> RankingCurve:
    return RankingCurve(*(like_input(x, pred) for x in (
        curve.thresholds, curve.precision, curve.recall, curve.fpr, curve.tpr)))


@torch.no_grad()
def voxel_ranking(pred: Any, target: Any, *, from_logits: bool = False, multilabel: bool = False,
                  include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
                  bins: int | None = None) -> RankingResult:
    """Per-case voxel PR/ROC curves, AP and AUROC on 2D/3D segmentation predictions.

    C=1 means foreground binary scores with index target; C>=2 means categorical
    one-vs-rest evaluation. ``multilabel=True`` uses independent binary channels
    with target shape (B,C,*spatial). Shared boolean masks and index ignore_index
    follow overlap metrics; multilabel also accepts per-label masks and rejects
    ignore_index. Channel 0 may be dropped for C>=2.

    Exact mode uses native CPU/CUDA sorting, groups score ties, and needs O(voxels)
    memory per class. ``bins>=2`` quantizes probabilities into equal-width bins
    (last bin includes 1), approximating ranking; increase bins to reduce ties.
    These are voxel metrics, distinct from lesion detection AP. For dataset-pooled
    accumulation, use :class:`exacta.metrics.VoxelRanking`.
    """
    bins = _check_bins(bins)
    data = _inputs(pred, target, from_logits=from_logits, multilabel=multilabel,
                   include_background=include_background, ignore_index=ignore_index, mask=mask)
    return _evaluate(data, pred, bins, keep_curves=True)


def _evaluate(data: _RankingInput, pred: Any, bins: int | None, *, keep_curves: bool) -> RankingResult:
    ap = torch.empty((data.pred.shape[0], data.classes), dtype=torch.float64, device=data.pred.device)
    auc = torch.empty_like(ap)
    curves = []
    for b in range(data.pred.shape[0]):
        case = []
        for c, (scores, positives) in enumerate(data.rows(b, bins)):
            curve, ap[b, c], auc[b, c] = _curve(scores, positives, bins)
            if keep_curves:
                case.append(_convert_curve(curve, pred))
        if keep_curves:
            curves.append(tuple(case))
    return RankingResult(like_input(ap, pred), like_input(auc, pred), tuple(curves), bins)


@torch.no_grad()
def average_precision(pred: Any, target: Any, *, from_logits: bool = False, multilabel: bool = False,
                      include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
                      bins: int | None = None) -> Any:
    """Per-case voxel AP (B,K); arguments and conventions as in :func:`voxel_ranking`."""
    bins = _check_bins(bins)
    data = _inputs(pred, target, from_logits=from_logits, multilabel=multilabel,
                   include_background=include_background, ignore_index=ignore_index, mask=mask)
    return _evaluate(data, pred, bins, keep_curves=False).average_precision


@torch.no_grad()
def auroc(pred: Any, target: Any, *, from_logits: bool = False, multilabel: bool = False,
          include_background: bool = True, ignore_index: int | None = None, mask: Any = None,
          bins: int | None = None) -> Any:
    """Per-case voxel AUROC (B,K); arguments and conventions as in :func:`voxel_ranking`."""
    bins = _check_bins(bins)
    data = _inputs(pred, target, from_logits=from_logits, multilabel=multilabel,
                   include_background=include_background, ignore_index=ignore_index, mask=mask)
    return _evaluate(data, pred, bins, keep_curves=False).auroc
