"""Independent NumPy/SciPy implementations used as ground truth in tests and benchmarks.

These follow the definitions in exacta's docstrings but share no code with exacta.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(p: np.ndarray, t: np.ndarray) -> float:
    den = p.sum() + t.sum()
    return np.nan if den == 0 else 2 * (p & t).sum() / den


def iou(p: np.ndarray, t: np.ndarray) -> float:
    union = (p | t).sum()
    return np.nan if union == 0 else (p & t).sum() / union


def fbeta(p: np.ndarray, t: np.ndarray, beta: float) -> float:
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    if tp + fp + fn == 0:
        return np.nan
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def _divide(num: float, den: float, zero_division: float) -> float:
    return zero_division if den == 0 else num / den


def sensitivity(p: np.ndarray, t: np.ndarray, zero_division: float = np.nan) -> float:
    return _divide((p & t).sum(), t.sum(), zero_division)


def precision(p: np.ndarray, t: np.ndarray, zero_division: float = np.nan) -> float:
    return _divide((p & t).sum(), p.sum(), zero_division)


def specificity(p: np.ndarray, t: np.ndarray, zero_division: float = np.nan) -> float:
    return _divide((~p & ~t).sum(), (~t).sum(), zero_division)


def volume_similarity(
    p: np.ndarray, t: np.ndarray, convention: str = "taha", zero_division: float = np.nan
) -> float:
    vp, vt = int(p.sum()), int(t.sum())
    if vp + vt == 0:
        return zero_division
    return 1 - abs(vp - vt) / (vp + vt) if convention == "taha" else 2 * (vp - vt) / (vp + vt)


def relative_volume_difference(
    p: np.ndarray, t: np.ndarray, absolute: bool = False, zero_division: float = np.nan
) -> float:
    vp, vt = int(p.sum()), int(t.sum())
    difference = abs(vp - vt) if absolute else vp - vt
    return _divide(difference, vt, zero_division)


def boundary(m: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(m.ndim, 1)
    return m & ~ndimage.binary_erosion(m, structure=structure, border_value=0)


def surface_distances(p: np.ndarray, t: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Distances from each boundary voxel of p to the boundary of t, and from t to p."""
    bp, bt = boundary(p), boundary(t)
    d_pt = ndimage.distance_transform_edt(~bt, sampling=spacing)[bp]
    d_tp = ndimage.distance_transform_edt(~bp, sampling=spacing)[bt]
    return d_pt, d_tp


def _symmetric(stat, d_pt: np.ndarray, d_tp: np.ndarray, combine: str) -> float:
    if combine == "pooled":
        return stat(np.concatenate([d_pt, d_tp]))
    a, b = stat(d_pt), stat(d_tp)
    return {"max": max(a, b), "mean": (a + b) / 2}[combine]


def hausdorff(
    p: np.ndarray, t: np.ndarray, spacing, percentile: float = 100.0, combine: str = "max"
) -> float:
    if not p.any() and not t.any():
        return np.nan
    if not p.any() or not t.any():
        return np.inf
    return _symmetric(lambda d: np.percentile(d, percentile), *surface_distances(p, t, spacing), combine)


def assd(p: np.ndarray, t: np.ndarray, spacing, combine: str = "pooled") -> float:
    if not p.any() and not t.any():
        return np.nan
    if not p.any() or not t.any():
        return np.inf
    return _symmetric(np.mean, *surface_distances(p, t, spacing), combine)


def nsd(p: np.ndarray, t: np.ndarray, spacing, tolerance: float) -> float:
    if not p.any() and not t.any():
        return np.nan
    if not p.any() or not t.any():
        return 0.0
    d_pt, d_tp = surface_distances(p, t, spacing)
    return ((d_pt <= tolerance).sum() + (d_tp <= tolerance).sum()) / (d_pt.size + d_tp.size)


def per_case(fn, pred: np.ndarray, target: np.ndarray, num_classes: int, spacings=None, **kw) -> np.ndarray:
    """Apply a binary reference metric to every (case, class) of index label maps."""
    out = np.empty((pred.shape[0], num_classes))
    for b in range(pred.shape[0]):
        for c in range(num_classes):
            args = (pred[b] == c, target[b] == c)
            if spacings is not None:
                args += (spacings[b],)
            out[b, c] = fn(*args, **kw)
    return out


def _instance_tables(true: np.ndarray, pred: np.ndarray):
    """Dense intersection/union tables between instance values (rows: target, columns: pred)."""
    t_ids = [v for v in np.unique(true) if v != 0]
    p_ids = [v for v in np.unique(pred) if v != 0]
    inter = np.zeros((len(t_ids), len(p_ids)))
    union = np.zeros((len(t_ids), len(p_ids)))
    t_area = np.array([(true == v).sum() for v in t_ids], dtype=float)
    p_area = np.array([(pred == v).sum() for v in p_ids], dtype=float)
    for a, tv in enumerate(t_ids):
        for b, pv in enumerate(p_ids):
            inter[a, b] = ((true == tv) & (pred == pv)).sum()
            union[a, b] = t_area[a] + p_area[b] - inter[a, b]
    return inter, union, t_area, p_area


def aji(true: np.ndarray, pred: np.ndarray) -> float:
    """AJI as the algorithm of HoVer-Net's get_fast_aji: per target, argmax of a dense IoU table
    with 1e-6 added to the union; unpaired target/prediction areas join the union."""
    inter, union, t_area, p_area = _instance_tables(true, pred)
    if not len(t_area) and not len(p_area):
        return np.nan
    if not len(t_area):
        return 0.0
    if not len(p_area):
        return 0.0
    iou = inter / (union + 1e-6)
    best = iou.argmax(1)
    paired = np.nonzero(iou.max(1) > 0)[0]
    total_i = inter[paired, best[paired]].sum()
    total_u = union[paired, best[paired]].sum()
    total_u += t_area[np.setdiff1d(np.arange(len(t_area)), paired)].sum()
    total_u += p_area[np.setdiff1d(np.arange(len(p_area)), best[paired])].sum()
    return total_i / total_u


def aji_plus(true: np.ndarray, pred: np.ndarray) -> float:
    """AJI+ as the algorithm of HoVer-Net's get_fast_aji_plus: one assignment on the dense IoU
    table, pairs of positive IoU kept."""
    from scipy.optimize import linear_sum_assignment

    inter, union, t_area, p_area = _instance_tables(true, pred)
    if not len(t_area) and not len(p_area):
        return np.nan
    if not len(t_area) or not len(p_area):
        return 0.0
    iou = inter / (union + 1e-6)
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] > 0
    rows, cols = rows[keep], cols[keep]
    total_u = union[rows, cols].sum() + t_area[np.setdiff1d(np.arange(len(t_area)), rows)].sum()
    total_u += p_area[np.setdiff1d(np.arange(len(p_area)), cols)].sum()
    return inter[rows, cols].sum() / total_u


def panoptic(true: np.ndarray, pred: np.ndarray, threshold: float = 0.5):
    """(tp, fp, fn, Σ IoU) of strict IoU > threshold matching for threshold >= 0.5 (unique)."""
    inter, union, t_area, p_area = _instance_tables(true, pred)
    iou = inter / np.where(union > 0, union, 1)
    rows, cols = np.nonzero(iou > threshold)
    tp = len(rows)
    return tp, len(p_area) - tp, len(t_area) - tp, iou[rows, cols].sum()


def ctc_seg(true: np.ndarray, pred: np.ndarray) -> float:
    """SEG: mean over target instances of the IoU with the prediction covering > half of it."""
    inter, union, t_area, _ = _instance_tables(true, pred)
    if not len(t_area):
        return np.nan
    scores = []
    for a in range(len(t_area)):
        hit = np.nonzero(2 * inter[a] > t_area[a])[0]
        scores.append(inter[a, hit[0]] / union[a, hit[0]] if len(hit) else 0.0)
    return float(np.mean(scores))
