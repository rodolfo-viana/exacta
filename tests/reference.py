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


def hausdorff(p: np.ndarray, t: np.ndarray, spacing, percentile: float = 100.0) -> float:
    if not p.any() and not t.any():
        return np.nan
    if not p.any() or not t.any():
        return np.inf
    bp, bt = boundary(p), boundary(t)
    d_pt = ndimage.distance_transform_edt(~bt, sampling=spacing)[bp]
    d_tp = ndimage.distance_transform_edt(~bp, sampling=spacing)[bt]
    return max(np.percentile(d_pt, percentile), np.percentile(d_tp, percentile))


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
