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
