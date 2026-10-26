"""Dataset-level regression metrics for (N,) or (N,D) real arrays.

Predictions come first. Masks exclude individual targets or complete samples;
sample weights have shape (N,). Zero-weight entries are unobserved. Functional
results retain the prediction's array library. All statistics use float64.
"""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from exacta import _native
from exacta._inputs import like_input
from exacta._tabular import divide, observations, real_tensor

Multioutput = Literal["raw_values", "uniform_average", "variance_weighted"]
_NAMES: tuple[str, ...] = ("mae", "mse", "rmse", "r2", "explained_variance", "pearson", "concordance_correlation")
_ROWS = 11


def _options(multioutput: str, force_finite: bool) -> None:
    if multioutput not in ("raw_values", "uniform_average", "variance_weighted"):
        raise ValueError("multioutput must be raw_values, uniform_average or variance_weighted")
    if not isinstance(force_finite, bool):
        raise TypeError("force_finite must be a bool")


def _inputs(pred: Any, target: Any, mask: Any, sample_weight: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p = real_tensor(pred, "pred")
    t = real_tensor(target, "target", p.device)
    if p.shape != t.shape or p.ndim not in (1, 2) or (p.ndim == 2 and p.shape[1] == 0):
        raise ValueError("pred and target must have the same shape (N,) or (N,D), D > 0")
    valid, weight = observations(t, mask, sample_weight)
    if not bool(((torch.isfinite(p) & torch.isfinite(t)) | ~valid).all()):
        raise ValueError("observed predictions and targets must be finite")
    p, t = torch.where(valid, p.double(), 0), torch.where(valid, t.double(), 0)
    w = valid.double() if weight is None else torch.where(valid, weight, 0)
    if p.ndim == 1:
        p, t, w, valid = (x[:, None] for x in (p, t, w, valid))
    return p, t, w, valid


def _native_statistics(pred: Any, target: Any, mask: Any, sample_weight: Any, *, moments: bool) -> Tensor | None:
    p = real_tensor(pred, "pred")
    if p.device.type != "cpu" or not _native.available():
        return None
    t = real_tensor(target, "target", p.device)
    if p.shape != t.shape or p.ndim not in (1, 2) or (p.ndim == 2 and p.shape[1] == 0):
        raise ValueError("pred and target must have the same shape (N,) or (N,D), D > 0")
    m = None if mask is None else real_tensor(mask, "mask", p.device)
    if m is not None:
        shapes = (t.shape, t.shape[:1], (t.shape[0], 1)) if t.ndim == 2 else (t.shape,)
        if m.dtype != torch.bool or m.shape not in shapes:
            raise ValueError("mask must be boolean with target shape or shape (N,)")
        if m.ndim == 1:
            m = m[:, None]
    w = None if sample_weight is None else real_tensor(sample_weight, "sample_weight", p.device)
    if w is not None and w.shape != t.shape[:1]:
        raise ValueError("sample_weight must have shape (N,)")
    if p.ndim == 1:
        p, t = p[:, None], t[:, None]
    return _native.regression_stats(p, t, m, w, moments=moments)


def _moments(x: Tensor, weight: Tensor, total: Tensor) -> tuple[Tensor, Tensor]:
    if x.shape[0] == 0:
        return total.clone(), x
    # Anchor at an observed value before summing: stable even for a large DC offset.
    first = (weight > 0).long().argmax(0, keepdim=True)
    anchor = x.gather(0, first).squeeze(0)
    centered = torch.where(weight > 0, x - anchor, 0)
    shift = (centered * weight).sum(0) / torch.where(total > 0, total, 1)
    return anchor + shift, torch.where(weight > 0, centered - shift, 0)


def _statistics(pred: Any, target: Any, mask: Any, sample_weight: Any) -> Tensor:
    native = _native_statistics(pred, target, mask, sample_weight, moments=True)
    if native is not None:
        return native
    p, t, w, valid = _inputs(pred, target, mask, sample_weight)
    total, count = w.sum(0), valid.sum(0)
    mp, cp = _moments(p, w, total)
    mt, ct = _moments(t, w, total)
    error = p - t
    me, ce = _moments(error, w, total)
    return torch.stack((total, count, mp, mt, (cp.square() * w).sum(0),
                        (ct.square() * w).sum(0), (cp * ct * w).sum(0),
                        me, (ce.square() * w).sum(0), (error.abs() * w).sum(0),
                        (error.square() * w).sum(0)))


def _merge(a: Tensor, b: Tensor) -> Tensor:
    """Weighted parallel centered moments (Chan); never average batch scores."""
    if a.shape != b.shape or a.device != b.device:
        raise ValueError("output count and device must remain constant across updates")
    result = a + b
    total = result[0]
    ratio = divide(b[0], total, 0)
    cross = a[0] * ratio
    for mean, moment in ((2, 4), (3, 5), (7, 8)):
        delta = b[mean] - a[mean]
        result[mean] = a[mean] + delta * ratio
        result[moment] += delta.square() * cross
    result[6] += (b[2] - a[2]) * (b[3] - a[3]) * cross
    return result


def _reduce(values: Tensor, multioutput: str, variance: Tensor | None = None) -> Tensor:
    if multioutput == "raw_values":
        return values
    if multioutput == "variance_weighted":
        if variance is None:
            raise ValueError("variance_weighted is only defined for r2 and explained_variance")
        # Constant outputs carry no weight; when all are constant, use the uniform mean.
        total = variance.sum()
        weighted = torch.where(variance > 0, values * variance, 0).sum() / total
        return torch.where(total > 0, weighted, values.mean())
    return values.mean()


def _scores(state: Tensor, names: tuple[str, ...], multioutput: str, force_finite: bool) -> dict[str, Tensor]:
    w, count, mp, mt, vp, vt, cov, _, ve, ae, se = state
    result = {}
    for name in names:
        variance = None
        if name == "mae":
            value = divide(ae, w)
        elif name in ("mse", "rmse"):
            value = divide(se, w)
            if name == "rmse":
                value = value.sqrt()
        elif name in ("r2", "explained_variance"):
            numerator = se if name == "r2" else ve
            value = 1 - numerator / vt
            if force_finite:
                value = torch.where(vt == 0, (numerator == 0).double(), value)
            value = torch.where((count >= (2 if name == "r2" else 1)) & (w > 0), value, float("nan"))
            variance = vt
        elif name == "pearson":
            value = (cov / vp.sqrt() / vt.sqrt()).clamp(-1, 1)
        else:
            value = 2 * cov / (vp + vt + w * (mp - mt).square())
        result[name] = _reduce(value, multioutput, variance)
    return result


@torch.no_grad()
def evaluate(pred: Any, target: Any, *, metrics: tuple[str, ...] = _NAMES,
             mask: Any = None, sample_weight: Any = None,
             multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> dict[str, Any]:
    """Share reductions across metrics; return a dictionary of scalars or (D,) arrays.

    ``raw_values`` returns one score per output. Uniform averages include NaNs.
    Variance weighting is available only for R² and explained variance. Constant
    targets give NaN/-inf for these scores unless ``force_finite=True`` replaces
    perfect/imperfect constant predictions with 1/0. R² needs two observations.
    Pearson is NaN for constant inputs; concordance is NaN for identical constants.
    Empty outputs are NaN. Metrics never build an autograd graph.
    """
    _options(multioutput, force_finite)
    if not metrics or any(name not in _NAMES for name in metrics):
        raise ValueError(f"metrics must be a nonempty selection from {_NAMES}")
    if multioutput == "variance_weighted" and any(n not in ("r2", "explained_variance") for n in metrics):
        raise ValueError("variance_weighted is only defined for r2 and explained_variance")
    names = tuple(metrics)
    if all(name in ("mae", "mse", "rmse") for name in names):
        # Simple error metrics do not pay for covariance/centered-moment passes.
        native = _native_statistics(pred, target, mask, sample_weight, moments=False)
        if native is not None:
            results = _scores(native, names, multioutput, force_finite)
        else:
            p, t, w, _ = _inputs(pred, target, mask, sample_weight)
            error, total = p - t, w.sum(0)
            results = {}
            mse = divide((error.square() * w).sum(0), total) if any(n != "mae" for n in names) else total
            for name in names:
                value = divide((error.abs() * w).sum(0), total) if name == "mae" else mse
                results[name] = _reduce(value.sqrt() if name == "rmse" else value, multioutput)
    else:
        results = _scores(_statistics(pred, target, mask, sample_weight), names, multioutput, force_finite)
    return {name: like_input(value, pred) for name, value in results.items()}


def mae(pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None,
         multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> Any:
    """Mean absolute error. See :func:`evaluate` for the common contract."""
    return evaluate(pred, target, metrics=("mae",), mask=mask, sample_weight=sample_weight,
                    multioutput=multioutput, force_finite=force_finite)["mae"]


def mse(pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None,
         multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> Any:
    """Mean squared error. See :func:`evaluate` for the common contract."""
    return evaluate(pred, target, metrics=("mse",), mask=mask, sample_weight=sample_weight,
                    multioutput=multioutput, force_finite=force_finite)["mse"]


def rmse(pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None,
         multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> Any:
    """Root mean squared error, taking the square root per output before averaging. See :func:`evaluate` for the common contract."""
    return evaluate(pred, target, metrics=("rmse",), mask=mask, sample_weight=sample_weight,
                    multioutput=multioutput, force_finite=force_finite)["rmse"]


def r2(pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None,
         multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> Any:
    """Coefficient of determination; see evaluate for constant-target semantics. See :func:`evaluate` for the common contract."""
    return evaluate(pred, target, metrics=("r2",), mask=mask, sample_weight=sample_weight,
                    multioutput=multioutput, force_finite=force_finite)["r2"]


def explained_variance(pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None,
         multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> Any:
    """Explained variance; insensitive to a constant prediction bias. See :func:`evaluate` for the common contract."""
    return evaluate(pred, target, metrics=("explained_variance",), mask=mask, sample_weight=sample_weight,
                    multioutput=multioutput, force_finite=force_finite)["explained_variance"]


def pearson(pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None,
         multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> Any:
    """Weighted Pearson product-moment correlation. See :func:`evaluate` for the common contract."""
    return evaluate(pred, target, metrics=("pearson",), mask=mask, sample_weight=sample_weight,
                    multioutput=multioutput, force_finite=force_finite)["pearson"]


def concordance_correlation(pred: Any, target: Any, *, mask: Any = None, sample_weight: Any = None,
         multioutput: Multioutput = "uniform_average", force_finite: bool = False) -> Any:
    """Lin concordance correlation, including location and scale disagreement. See :func:`evaluate` for the common contract."""
    return evaluate(pred, target, metrics=("concordance_correlation",), mask=mask, sample_weight=sample_weight,
                    multioutput=multioutput, force_finite=force_finite)["concordance_correlation"]
