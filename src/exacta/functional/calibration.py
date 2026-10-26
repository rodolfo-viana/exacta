"""Probability quality and reliability summaries without volume-sized one-hot targets."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from exacta._inputs import like_input, to_torch
from exacta import _multilabel
from exacta._multilabel import Average
from exacta.losses.segmentation import _check_ignore_index, _check_index_target, _check_mask, _prepare_masked_input

CalibrationMode = Literal["top-label", "classwise"]


@dataclass(frozen=True)
class ReliabilityDiagram:
    """Equal-width reliability bins, shaped (B, K, bins).

    K=1 for top-label calibration; K=C for classwise calibration. Empty-bin mean
    confidence/accuracy are NaN. ``ece`` is (B,K), NaN for no valid voxels.
    ``counts`` are int64; other arrays are float64. ``edges`` has bins+1 entries.
    For classwise bins, accuracy is the fraction with that class as the target.
    """
    counts: Any
    confidence: Any
    accuracy: Any
    edges: Any
    ece: Any


def _probabilities(pred: Any, target: Any, from_logits: bool, ignore_index: int | None,
                   mask: Any, *, need_nll: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p, t = to_torch(pred), to_torch(target)
    m = None if mask is None else to_torch(mask)
    m = _check_mask(m, p)
    ignore_index = _check_ignore_index(ignore_index, "index")
    p, t, valid = _prepare_masked_input(p, t, "index", ignore_index, m)
    _check_index_target(t, p)
    if p.ndim not in (4, 5):
        raise ValueError("pred must have shape (B,C,H,W) or (B,C,D,H,W)")
    classes = p.shape[1]
    t = t.long()
    if not bool(((t >= 0) & (t < max(2, classes))).all()):
        raise ValueError("target contains an out-of-range class index")
    if valid is None:
        valid = torch.ones_like(t, dtype=torch.bool)
    if not bool(torch.isfinite(p).all()):
        raise ValueError("valid predictions must be finite")
    nll = p[:, 0]  # Unused when need_nll=False; avoid an extra log/CE volume pass.
    if from_logits:
        if classes == 1:
            if need_nll:
                nll = F.binary_cross_entropy_with_logits(p[:, 0], t.to(p.dtype), reduction="none")
            p = p.sigmoid()
        elif need_nll:
            logp = p.log_softmax(1)
            nll = -logp.gather(1, t[:, None]).squeeze(1)
            p = logp.exp()
        else:
            p = p.softmax(1)
    else:
        if not bool(((p >= 0) & (p <= 1)).all()):
            raise ValueError("probabilities must lie in [0, 1]")
        if classes > 1:
            total = p.sum(1)
            if not bool((~valid | ((total - 1).abs() <= 1e-3)).all()):
                raise ValueError("multiclass probabilities must sum to one")
            if need_nll:
                nll = -p.gather(1, t[:, None]).squeeze(1).log()
        elif need_nll:
            nll = -torch.where(t.bool(), p[:, 0], 1 - p[:, 0]).log()
    return p, t, valid, nll


def _case_mean(terms: Tensor, valid: Tensor) -> Tensor:
    count = valid.flatten(1).sum(1)
    total = torch.where(valid, terms, 0).flatten(1).sum(1, dtype=torch.float64)
    return torch.where(count > 0, total / count.clamp_min(1), torch.nan)[:, None]


@torch.no_grad()
def brier_score(pred: Any, target: Any, *, from_logits: bool = False,
                ignore_index: int | None = None, mask: Any = None,
                multilabel: bool = False, average: Average = "none") -> Any:
    """Per-case (B,1) Brier score; lower is better.

    C>=2: mean over valid voxels of sum_c (p_c - one_hot(y)_c)^2 (range [0,2]).
    C=1: binary mean squared error (range [0,1]), targets 0/1. These conventions
    differ by a factor of two for equivalent one-/two-channel binary models.
    Empty cases return NaN. ``from_logits=True`` uses softmax, or sigmoid for C=1.
    With ``multilabel=True``, target is binary (B,C,*spatial), logits use independent
    sigmoids, and scores are mean squared errors per label in [0,1]. ``average="none"``
    returns (B,C); ``"macro"`` averages observed label means and ``"micro"`` weights
    observed label-voxel pairs equally, both (B,1). Masks may also be (B,C,*spatial),
    allowing different observations per label; ignore_index is unsupported. Empty labels
    return NaN and are skipped by macro averaging. Float64 arithmetic/reductions are used.
    This is a proper probability score, not a pure measure of calibration.
    """
    _multilabel.check_options(multilabel, average, ignore_index)
    if multilabel:
        sums = _multilabel.statistics(pred, target, mask, from_logits, "brier")
        return like_input(_multilabel.score(sums, average), pred)
    p, t, valid, _ = _probabilities(pred, target, from_logits, ignore_index, mask)
    if p.shape[1] == 1:
        terms = (p[:, 0] - t).square()
    else:
        # Subtract at the true class before squaring: avoids cancellation near perfection.
        error = p.scatter_add(1, t[:, None], -torch.ones_like(t[:, None], dtype=p.dtype))
        terms = error.square().sum(1)
    return like_input(_case_mean(terms, valid), pred)


@torch.no_grad()
def negative_log_likelihood(pred: Any, target: Any, *, from_logits: bool = False,
                            ignore_index: int | None = None, mask: Any = None,
                            multilabel: bool = False, average: Average = "none") -> Any:
    """Per-case (B,1) mean NLL, in nats. C=1 means binary; C>=2 means multiclass.

    Logits use stable log-softmax/BCE. With probabilities, zero probability assigned
    to the true class correctly yields +inf; probabilities are not silently clipped.
    With ``multilabel=True``, each channel is Bernoulli: both positive and negative outcomes
    contribute. The same layouts, masks and averaging as :func:`brier_score` apply. This is
    mean per-label NLL, not a sum over labels of a joint event's NLL.
    Empty cases give NaN. A proper probability score, not a pure calibration measure.
    """
    _multilabel.check_options(multilabel, average, ignore_index)
    if multilabel:
        sums = _multilabel.statistics(pred, target, mask, from_logits, "nll")
        return like_input(_multilabel.score(sums, average), pred)
    _, _, valid, nll = _probabilities(pred, target, from_logits, ignore_index, mask, need_nll=True)
    return like_input(_case_mean(nll, valid), pred)


def _calibration_mode(mode: CalibrationMode | None, multilabel: bool) -> CalibrationMode:
    if mode is None:
        return "classwise" if multilabel else "top-label"
    if mode not in ("top-label", "classwise"):
        raise ValueError("mode must be 'top-label' or 'classwise'")
    if multilabel and mode == "top-label":
        raise ValueError("multilabel calibration requires mode='classwise'; top-label is undefined")
    return mode


def _bin_sums(pred: Any, target: Any, from_logits: bool, ignore_index: int | None,
              mask: Any, bins: int, mode: CalibrationMode | None, *, multilabel: bool = False) -> Tensor:
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be a positive integer")
    _multilabel.check_options(multilabel, "none", ignore_index)
    mode = _calibration_mode(mode, multilabel)
    if multilabel:
        return _multilabel.statistics(pred, target, mask, from_logits, "bins", int(bins))
    if mode not in ("top-label", "classwise"):
        raise ValueError("mode must be 'top-label' or 'classwise'")
    p, t, valid, _ = _probabilities(pred, target, from_logits, ignore_index, mask)
    if mode == "top-label":
        if p.shape[1] == 1:
            positive = p[:, 0] > 0.5  # ties choose class 0, as multiclass argmax does
            confidence = torch.where(positive, p[:, 0], 1-p[:, 0])
            correct = positive == t
        else:
            confidence, label = p.max(1)
            correct = label == t
        pairs = [(confidence, correct)]
    else:
        pairs = [(p[:, c], t == (1 if p.shape[1] == 1 else c)) for c in range(p.shape[1])]
    packed = []
    valid = valid.flatten(1)
    for confidence, correct in pairs:
        confidence, correct = confidence.flatten(1), correct.flatten(1)
        indices = (confidence.double() * bins).long().clamp_max(bins - 1)
        count = torch.zeros((p.shape[0], bins), dtype=torch.int64, device=p.device)
        count.scatter_add_(1, indices, valid.long())
        confidence_sum = torch.zeros_like(count, dtype=torch.float64)
        accuracy_sum = torch.zeros_like(confidence_sum)
        confidence_sum.scatter_add_(1, indices, torch.where(valid, confidence, 0).double())
        accuracy_sum.scatter_add_(1, indices, (correct & valid).double())
        packed.append(torch.stack((count.double(), confidence_sum, accuracy_sum), dim=-2))
    return torch.stack(packed, dim=1)  # B,K,3,bins; count, confidence sum, correct sum


def _diagram(sums: Tensor) -> ReliabilityDiagram:
    count, confidence, correct = sums.unbind(-2)
    total = count.sum(-1)
    ece = (correct - confidence).abs().sum(-1) / total.clamp_min(1)
    return ReliabilityDiagram(
        count.long(), torch.where(count > 0, confidence / count.clamp_min(1), torch.nan),
        torch.where(count > 0, correct / count.clamp_min(1), torch.nan),
        torch.linspace(0, 1, count.shape[-1] + 1, device=count.device, dtype=torch.float64),
        torch.where(total > 0, ece, torch.nan),
    )


@torch.no_grad()
def reliability_diagram(pred: Any, target: Any, *, bins: int = 15, mode: CalibrationMode | None = None,
                        from_logits: bool = False, ignore_index: int | None = None,
                        mask: Any = None, multilabel: bool = False) -> ReliabilityDiagram:
    """Per-case equal-width bins: [k/bins,(k+1)/bins), last bin includes 1.

    Top-label calibration bins maximum confidence against prediction correctness;
    classwise bins each class probability against its binary target membership.
    ECE is sum_bin count/N * |accuracy-confidence| and depends on the bin choice.
    The same mask/ignore conventions as Brier/NLL apply. This is a numeric summary
    suitable for plotting, without a plotting dependency.
    With ``multilabel=True``, target is binary (B,C,*spatial); default mode is classwise
    and top-label mode is rejected. Each label's positive probability is binned against
    its observed positive frequency. A mask may select voxels separately per channel.
    Without multilabel, the default remains top-label. Diagrams always retain label bins;
    use :func:`expected_calibration_error` for macro/micro reductions.
    """
    result = _diagram(_bin_sums(pred, target, from_logits, ignore_index, mask, bins, mode, multilabel=multilabel))
    return ReliabilityDiagram(*(like_input(x, pred) for x in
                                (result.counts, result.confidence, result.accuracy, result.edges, result.ece)))


@torch.no_grad()
def expected_calibration_error(pred: Any, target: Any, *, bins: int = 15, mode: CalibrationMode | None = None,
                               from_logits: bool = False, ignore_index: int | None = None,
                               mask: Any = None, multilabel: bool = False,
                               average: Average = "none") -> Any:
    """(B,K) ECE from :func:`reliability_diagram`; K=1 or C for classwise mode.

    For multilabel predictions, ``average="none"`` retains (B,C), ``"macro"`` averages
    nonempty labels' ECEs, and ``"micro"`` pools label-voxel pairs into common bins before
    taking absolute calibration gaps. Macro/micro return (B,1). Micro errors can cancel
    across labels; it is not a weighted mean of per-label ECEs.
    """
    _multilabel.check_options(multilabel, average, ignore_index)
    sums = _bin_sums(pred, target, from_logits, ignore_index, mask, bins, mode, multilabel=multilabel)
    return like_input(_multilabel.ece(sums, average), pred)
