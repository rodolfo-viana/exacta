"""Dataset-level classification, with explicit tasks and averaging conventions.

Binary inputs have shape (N,), multiclass predictions (N,) labels or (N,C)
probabilities/logits with targets (N,), and multilabel inputs (N,C). Class labels
are zero-based. Float predictions use probabilities unless from_logits=True.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Literal

import torch
from torch import Tensor

from exacta import _native, _triton_counts
from exacta._inputs import like_input
from exacta._tabular import divide, observations, real_tensor

Task = Literal["binary", "multiclass", "multilabel"]
Average = Literal["none", "micro", "macro", "weighted"]
_NAMES: tuple[str, ...] = ("accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc")


def _probability_inputs(pred: Any, target: Any, task: str, num_classes: int | None,
                        from_logits: bool, mask: Any, ignore_index: int | None) -> tuple[Tensor, Tensor, Tensor]:
    """View samples as one pooled spatial domain, reusing validated probability kernels."""
    _options(task, num_classes, .5, from_logits, ignore_index=ignore_index)
    p = real_tensor(pred, "pred")
    t = real_tensor(target, "target", p.device)
    if not p.is_floating_point():
        raise ValueError("probability metrics require floating predictions")
    if task == "binary":
        if p.ndim != 1 or p.shape != t.shape:
            raise ValueError("binary probability inputs must have shape (N,)")
    elif task == "multiclass":
        if t.ndim != 1 or p.shape != (t.shape[0], num_classes):
            raise ValueError("multiclass probabilities require (N,C) pred and (N,) target")
    elif p.ndim != 2 or p.shape != t.shape or p.shape[1] == 0 or (num_classes is not None and p.shape[1] != num_classes):
        raise ValueError("multilabel probability inputs must have shape (N,C) matching num_classes")
    valid, _ = observations(t, mask, None)
    if ignore_index is not None:
        valid &= t != ignore_index
    if task == "binary" and t.is_floating_point():
        if bool((((t != 0) & (t != 1)) & valid).any()):
            raise ValueError("observed binary targets must be 0 or 1")
        t = torch.where(valid, t, 0).long()
    scores = p.reshape(1, 1, 1, -1) if task == "binary" else p.T[None, :, None, :]
    labels = t.T[None, :, None, :] if task == "multilabel" else t.reshape(1, 1, -1)
    observed = valid.T[None, :, None, :] if task == "multilabel" else valid.reshape(1, 1, -1)
    return scores, labels, observed


def _options(task: str, num_classes: int | None, threshold: float, from_logits: bool,
             average: str = "macro", zero_division: float = float("nan"), beta: float = 1.,
             ignore_index: int | None = None) -> None:
    if task not in ("binary", "multiclass", "multilabel"):
        raise ValueError("task must be binary, multiclass or multilabel")
    if num_classes is not None and (isinstance(num_classes, bool) or not isinstance(num_classes, Integral)
                                    or num_classes < (2 if task != "multilabel" else 1)):
        raise ValueError("num_classes must be an integer >= 2 (>= 1 for multilabel)")
    if task == "binary" and num_classes not in (None, 2):
        raise ValueError("binary classification has two classes")
    if task == "multiclass" and num_classes is None:
        raise ValueError("multiclass requires an explicit num_classes")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if not isinstance(from_logits, bool):
        raise TypeError("from_logits must be a bool")
    if average not in ("none", "micro", "macro", "weighted"):
        raise ValueError("average must be none, micro, macro or weighted")
    if zero_division not in (0, 1) and not math.isnan(zero_division):
        raise ValueError("zero_division must be 0, 1 or NaN")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral)):
        raise TypeError("ignore_index must be an integer")


def _prepare(pred: Any, target: Any, task: str, num_classes: int | None, threshold: float,
             from_logits: bool, mask: Any, sample_weight: Any, ignore_index: int | None
             ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, int]:
    p = real_tensor(pred, "pred")
    t = real_tensor(target, "target", p.device)
    if task == "multilabel":
        if p.ndim != 2 or p.shape != t.shape or p.shape[1] == 0:
            raise ValueError("multilabel inputs must have identical shape (N,C), C > 0")
        classes = p.shape[1]
        if num_classes is not None and classes != num_classes:
            raise ValueError("num_classes must match the multilabel channel count")
    else:
        if t.ndim != 1 or (p.shape != t.shape and not (task == "multiclass" and p.shape == (t.shape[0], num_classes))):
            raise ValueError("binary uses (N,); multiclass uses (N,) labels or (N,C) scores with (N,) targets")
        classes = int(num_classes) if task == "multiclass" else 1  # type: ignore[arg-type]
        if task == "multiclass" and (t.is_floating_point() or (p.ndim == 1 and p.is_floating_point())):
            raise ValueError("multiclass labels must have an integer dtype")
    valid, weight = observations(t, mask, sample_weight)
    if ignore_index is not None:
        valid &= t != ignore_index
    if p.is_floating_point():
        observed = valid[:, None] if task == "multiclass" else valid
        errors = [((~torch.isfinite(p)) & observed).any()]
        if not from_logits:
            errors.append((((p < 0) | (p > 1)) & observed).any())
            if task == "multiclass":
                errors.append((((p.double().sum(1) - 1).abs() > 1e-3) & valid).any())
        if bool(torch.stack(errors).any()):
            raise ValueError("observed scores must be finite; probabilities must be in [0,1] and multiclass rows sum to one")
        if task == "multiclass":
            p = p.argmax(1)
        else:
            cutoff = threshold
            if from_logits:
                cutoff = -math.inf if threshold == 0 else math.inf if threshold == 1 else math.log(threshold / (1 - threshold))
            p = p >= cutoff
    elif from_logits or (task == "multiclass" and p.ndim == 2):
        raise ValueError("logits and score matrices require a floating dtype")
    return p, t, valid, weight, classes


def _matrix(p: Tensor, t: Tensor, valid: Tensor, weight: Tensor | None, classes: int) -> Tensor:
    if weight is None and not (p.is_floating_point() or t.is_floating_point()):
        if p.device.type == "cpu" and _native.available():
            found = _native.confusion(p, t, classes, mask=valid)
            if found is None:
                raise ValueError("observed labels must be in range")
            return found.T  # Existing segmentation kernels use predicted rows.
        if p.is_cuda and _triton_counts.available() and _triton_counts.fits(classes):
            found = _triton_counts.confusion(p[None], t[None], classes, mask=valid[None])
            if found is None:
                raise ValueError("observed labels must be in range")
            return found[0].T
    bad = ((p < 0) | (p >= classes) | (t < 0) | (t >= classes) | (p != p.long()) | (t != t.long())) & valid
    if bool(bad.any()):
        raise ValueError("observed labels must be integer values in range")
    index = t[valid].long() * classes + p[valid].long()
    weights = None if weight is None else weight[valid]
    # bincount's empty weighted result is int64 on some Torch versions; keep the contract.
    result = torch.bincount(index, weights=weights, minlength=classes * classes).reshape(classes, classes)
    return result.double() if weight is not None else result


def _counts(pred: Any, target: Any, task: str, num_classes: int | None, threshold: float,
            from_logits: bool, mask: Any, sample_weight: Any, ignore_index: int | None) -> Tensor:
    p, t, valid, weight, classes = _prepare(pred, target, task, num_classes, threshold,
                                           from_logits, mask, sample_weight, ignore_index)
    if task == "multiclass":
        if classes <= 32:
            matrix = _matrix(p, t, valid, weight, classes).double()
            return torch.stack((matrix.diag(), matrix.sum(0), matrix.sum(1), matrix.sum().expand(classes)))
        if bool((((p < 0) | (p >= classes) | (t < 0) | (t >= classes)) & valid).any()):
            raise ValueError("observed labels must be in range")
        # Large class spaces retain O(C) counts, without a C-by-C confusion matrix.
        p, t = p[valid].long(), t[valid].long()
        w = torch.ones_like(p, dtype=torch.float64) if weight is None else weight[valid]
        pc = torch.zeros(classes, dtype=torch.float64, device=p.device).scatter_add_(0, p, w)
        tc = torch.zeros_like(pc).scatter_add_(0, t, w)
        tp = torch.zeros_like(pc).scatter_add_(0, t, w * (p == t))
        return torch.stack((tp, pc, tc, w.sum().expand(classes)))
    if p.ndim == 1:
        p, t, valid = (x[:, None] for x in (p, t, valid))
        if weight is not None:
            weight = weight[:, None]
    if weight is None:
        args = (p.T.contiguous(), t.T.contiguous(), valid.T.contiguous(), 1)
        found = None
        if p.device.type == "cpu" and _native.available():
            found = _native.masked_channel_counts(*args)
        elif p.is_cuda and _triton_counts.available():
            found = _triton_counts.masked_channel_counts(*args)
        if found is not None:
            return torch.stack(found).double()
    if bool(((((p != 0) & (p != 1)) | ((t != 0) & (t != 1))) & valid).any()):
        raise ValueError("observed binary and multilabel labels must be 0 or 1")
    p, t = (p == 1) & valid, (t == 1) & valid
    w = valid.double() if weight is None else torch.where(valid, weight, 0)
    return torch.stack(((w * (p & t)).sum(0), (w * p).sum(0), (w * t).sum(0), w.sum(0)))


def _average(value: Tensor, support: Tensor, average: str) -> Tensor:
    if average == "none":
        return value
    if average == "weighted":
        return divide(torch.where(support > 0, value * support, 0).sum(), support.sum())
    return value.mean()


def _scores(counts: Tensor, names: tuple[str, ...], task: str, average: str,
            zero_division: float, beta: float) -> dict[str, Tensor]:
    tp, pc, tc, n = counts
    tn = n - pc - tc + tp
    out = {}
    for name in names:
        if name == "accuracy":
            out[name] = divide(tp.sum(), n[0], zero_division) if task == "multiclass" else divide((tp + tn).sum(), n.sum(), zero_division)
            continue
        if name == "cohen_kappa":
            if task == "multilabel":
                raise ValueError("cohen_kappa requires binary or multiclass labels")
            pp, tt = (pc, tc) if task == "multiclass" else (torch.cat((n - pc, pc)), torch.cat((n - tc, tc)))
            total = n[0]
            correct = tp.sum() if task == "multiclass" else (tp + tn).sum()
            expected = (pp * tt).sum()
            out[name] = divide(correct * total - expected, total.square() - expected, zero_division)
            continue
        if name == "mcc" and task == "multiclass":
            denominator = (pc * (n - pc).clamp_min(0)).sum().sqrt() * (tc * (n - tc).clamp_min(0)).sum().sqrt()
            # Different weighted reduction orders can leave n-p_c a few ulps above
            # zero for a constant prediction. Its MCC is still undefined, exactly.
            variable = ((pc > 0).sum() > 1) & ((tc > 0).sum() > 1)
            denominator = torch.where(variable, denominator, 0)
            out[name] = divide(tp.sum() * n[0] - (pc * tc).sum(), denominator, zero_division).clamp(-1, 1)
            continue
        if name == "balanced_accuracy" and task == "multiclass":
            # Fixed label universe: absent classes follow zero_division, just like macro recall.
            out[name] = divide(tp, tc, zero_division).mean()
            continue
        a, b, c, d = (x.sum().reshape(1) for x in (tp, pc, tc, n)) if average == "micro" else (tp, pc, tc, n)
        negative = d - b - c + a
        if name == "precision":
            value = divide(a, b, zero_division)
        elif name == "recall":
            value = divide(a, c, zero_division)
        elif name == "specificity":
            value = divide(negative, d - c, zero_division)
        elif name in ("f1", "fbeta"):
            # Scale by beta² for large beta instead of overflowing it.
            weight = beta if name == "fbeta" else 1.
            ratio = (1 / weight) ** 2 if weight > 1 else weight ** 2
            numerator = (1 + ratio) * a
            denominator = c + ratio * b if weight > 1 else ratio * c + b
            value = divide(numerator, denominator, zero_division)
        elif name == "mcc":
            value = divide(a * negative - (b - a) * (c - a), b.sqrt() * c.sqrt() * (d - b).sqrt() * (d - c).sqrt(), zero_division)
        else:
            value = (divide(a, c, zero_division) + divide(negative, d - c, zero_division)) / 2
        out[name] = _average(value, c, "none" if average == "none" else average)
    return out


@torch.no_grad()
def evaluate(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
             metrics: tuple[str, ...] = _NAMES, average: Average = "macro", threshold: float = .5,
             from_logits: bool = False, mask: Any = None, sample_weight: Any = None,
             ignore_index: int | None = None, zero_division: float = float("nan"), beta: float = 1.) -> dict[str, Any]:
    """Compute shared counts once. See the module docstring for input shapes.

    Binary rates describe class 1. Multilabel accuracy is decision accuracy, not
    exact-match accuracy. Multiclass accuracy, MCC, kappa and balanced accuracy are
    scalars regardless of average. Other rates support per-class (none), pooled
    (micro), equal-class (macro) or target-support-weighted means. Macro includes
    NaNs. Absent classes remain in the declared label universe. Excluded values
    (mask, ignore_index or zero sample weight) are not validated or counted.
    Undefined denominators give zero_division, NaN by default.
    """
    _options(task, num_classes, threshold, from_logits, average, zero_division, beta, ignore_index)
    if not metrics or any(n not in (*_NAMES, "fbeta", "cohen_kappa") for n in metrics):
        raise ValueError("unknown or empty classification metrics selection")
    if task == "multilabel" and "cohen_kappa" in metrics:
        raise ValueError("cohen_kappa requires binary or multiclass labels")
    counts = _counts(pred, target, task, num_classes, threshold, from_logits, mask, sample_weight, ignore_index)
    return {name: like_input(value, pred) for name, value in _scores(counts, tuple(metrics), task, average, zero_division, beta).items()}


@torch.no_grad()
def confusion_matrix(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
                     threshold: float = .5, from_logits: bool = False, mask: Any = None,
                     sample_weight: Any = None, ignore_index: int | None = None) -> Any:
    """Target rows, predicted columns; (2,2), (C,C), or (C,2,2) for multilabel.

    Counts are int64 without sample weights, float64 with weights. No normalization.
    """
    _options(task, num_classes, threshold, from_logits, ignore_index=ignore_index)
    if task == "multilabel":
        tp, pc, tc, n = _counts(pred, target, task, num_classes, threshold, from_logits, mask, sample_weight, ignore_index)
        matrix = torch.stack((n - pc - tc + tp, pc - tp, tc - tp, tp), 1).reshape(-1, 2, 2)
        if sample_weight is None:
            matrix = matrix.long()
    else:
        p, t, valid, weight, classes = _prepare(pred, target, task, num_classes, threshold, from_logits, mask, sample_weight, ignore_index)
        matrix = _matrix(p, t, valid, weight, max(2, classes))
    return like_input(matrix, pred)


@torch.no_grad()
def top_k_accuracy(pred: Any, target: Any, *, k: int = 5, mask: Any = None,
                   sample_weight: Any = None, ignore_index: int | None = None) -> Any:
    """Multiclass (N,C) finite scores, with equal scores favoring the lower class index."""
    p, t = real_tensor(pred, "pred"), real_tensor(target, "target")
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("pred must have shape (N,C), C >= 2")
    _options("multiclass", p.shape[1], .5, True, ignore_index=ignore_index)
    if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= p.shape[1]:
        raise ValueError("k must be an integer between 1 and C")
    _, t, valid, weight, classes = _prepare(p, t, "multiclass", p.shape[1], .5, True, mask, sample_weight, ignore_index)
    if bool((((t < 0) | (t >= classes)) & valid).any()):
        raise ValueError("observed targets must be in range")
    labels = torch.where(valid, t, 0).long()
    score = p.gather(1, labels[:, None])
    better = (p > score) | ((p == score) & (torch.arange(classes, device=p.device) < labels[:, None]))
    hit = better.sum(1) < k
    w = valid.double() if weight is None else torch.where(valid, weight, 0)
    return like_input(divide((hit * w).sum(), w.sum()), pred)


def accuracy(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """Fraction of correct decisions (not subset accuracy for multilabel). See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("accuracy",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["accuracy"]


def precision(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """Positive predictive value. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("precision",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["precision"]


def recall(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """Sensitivity / true-positive rate. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("recall",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["recall"]


def specificity(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """True-negative rate. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("specificity",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["specificity"]


def f1(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """F1 score, computed directly from counts. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("f1",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["f1"]


def fbeta(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """F-beta score; beta > 1 gives recall more weight. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("fbeta",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["fbeta"]


def balanced_accuracy(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """Mean recall over declared multiclass labels; mean sensitivity/specificity for binary labels. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("balanced_accuracy",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["balanced_accuracy"]


def mcc(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """Matthews correlation coefficient. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("mcc",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["mcc"]


def cohen_kappa(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
         average: Average = "macro", threshold: float = .5, from_logits: bool = False,
         mask: Any = None, sample_weight: Any = None, ignore_index: int | None = None,
         zero_division: float = float("nan"), beta: float = 1.) -> Any:
    """Unweighted Cohen kappa for binary or multiclass labels. See :func:`evaluate` for options."""
    return evaluate(pred, target, task=task, num_classes=num_classes, metrics=("cohen_kappa",),
                    average=average, threshold=threshold, from_logits=from_logits, mask=mask,
                    sample_weight=sample_weight, ignore_index=ignore_index,
                    zero_division=zero_division, beta=beta)["cohen_kappa"]


@torch.no_grad()
def brier_score(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
                   from_logits: bool = False, mask: Any = None, ignore_index: int | None = None) -> Any:
    """Dataset-pooled brier score; see the classification Wiki for score shapes.

    Binary/multiclass Brier and NLL and top-label ECE return scalars. Multilabel
    probability scores and all AP/AUROC calls return (C,) scores (binary C=1).
    AP/AUROC evaluate each multiclass label one-versus-rest. Undefined scores are
    NaN. Ranking bins=None is exact; integer bins quantizes probabilities. These
    probability APIs accept masks/ignore_index, but not sample weights.
    """
    from exacta.functional import calibration as backend

    p, t, valid = _probability_inputs(pred, target, task, num_classes, from_logits, mask, ignore_index)
    result = backend.brier_score(p, t, from_logits=from_logits, multilabel=task == "multilabel", mask=valid)[0]
    if task != "multilabel":
        result = result[0]
    return like_input(result, pred)


@torch.no_grad()
def negative_log_likelihood(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
                   from_logits: bool = False, mask: Any = None, ignore_index: int | None = None) -> Any:
    """Dataset-pooled negative log likelihood; see the classification Wiki for score shapes.

    Binary/multiclass Brier and NLL and top-label ECE return scalars. Multilabel
    probability scores and all AP/AUROC calls return (C,) scores (binary C=1).
    AP/AUROC evaluate each multiclass label one-versus-rest. Undefined scores are
    NaN. Ranking bins=None is exact; integer bins quantizes probabilities. These
    probability APIs accept masks/ignore_index, but not sample weights.
    """
    from exacta.functional import calibration as backend

    p, t, valid = _probability_inputs(pred, target, task, num_classes, from_logits, mask, ignore_index)
    result = backend.negative_log_likelihood(p, t, from_logits=from_logits, multilabel=task == "multilabel", mask=valid)[0]
    if task != "multilabel":
        result = result[0]
    return like_input(result, pred)


@torch.no_grad()
def expected_calibration_error(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
                   from_logits: bool = False, mask: Any = None, ignore_index: int | None = None, bins: int = 15) -> Any:
    """Dataset-pooled expected calibration error; see the classification Wiki for score shapes.

    Binary/multiclass Brier and NLL and top-label ECE return scalars. Multilabel
    probability scores and all AP/AUROC calls return (C,) scores (binary C=1).
    AP/AUROC evaluate each multiclass label one-versus-rest. Undefined scores are
    NaN. Ranking bins=None is exact; integer bins quantizes probabilities. These
    probability APIs accept masks/ignore_index, but not sample weights.
    """
    from exacta.functional import calibration as backend

    p, t, valid = _probability_inputs(pred, target, task, num_classes, from_logits, mask, ignore_index)
    result = backend.expected_calibration_error(p, t, from_logits=from_logits, multilabel=task == "multilabel", mask=valid, bins=bins)[0]
    if task != "multilabel":
        result = result[0]
    return like_input(result, pred)


@torch.no_grad()
def average_precision(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
                   from_logits: bool = False, mask: Any = None, ignore_index: int | None = None, bins: int | None = None) -> Any:
    """Dataset-pooled average precision; see the classification Wiki for score shapes.

    Binary/multiclass Brier and NLL and top-label ECE return scalars. Multilabel
    probability scores and all AP/AUROC calls return (C,) scores (binary C=1).
    AP/AUROC evaluate each multiclass label one-versus-rest. Undefined scores are
    NaN. Ranking bins=None is exact; integer bins quantizes probabilities. These
    probability APIs accept masks/ignore_index, but not sample weights.
    """
    from exacta.functional import ranking as backend

    p, t, valid = _probability_inputs(pred, target, task, num_classes, from_logits, mask, ignore_index)
    result = backend.average_precision(p, t, from_logits=from_logits, multilabel=task == "multilabel", mask=valid, bins=bins)[0]
    return like_input(result, pred)


@torch.no_grad()
def auroc(pred: Any, target: Any, *, task: Task, num_classes: int | None = None,
                   from_logits: bool = False, mask: Any = None, ignore_index: int | None = None, bins: int | None = None) -> Any:
    """Dataset-pooled auroc; see the classification Wiki for score shapes.

    Binary/multiclass Brier and NLL and top-label ECE return scalars. Multilabel
    probability scores and all AP/AUROC calls return (C,) scores (binary C=1).
    AP/AUROC evaluate each multiclass label one-versus-rest. Undefined scores are
    NaN. Ranking bins=None is exact; integer bins quantizes probabilities. These
    probability APIs accept masks/ignore_index, but not sample weights.
    """
    from exacta.functional import ranking as backend

    p, t, valid = _probability_inputs(pred, target, task, num_classes, from_logits, mask, ignore_index)
    result = backend.auroc(p, t, from_logits=from_logits, multilabel=task == "multilabel", mask=valid, bins=bins)[0]
    return like_input(result, pred)
