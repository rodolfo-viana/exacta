"""Thin Python layer over the compiled kernels in ``exacta._C``."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

try:
    from exacta import _C
except ImportError:  # running from a source tree that was never built
    _C = None


def available() -> bool:
    return _C is not None


def edt_sq(features: Tensor, spacing: Sequence[float]) -> Tensor:
    """Squared Euclidean distance from every voxel to the nearest nonzero voxel of ``features``.

    Each value is the exact floating-point minimum, over nonzero voxels p, of
    ``sum_i ((p_i - q_i) * h_i)^2`` rounded and summed in axis order, which is the expression
    SciPy's ``distance_transform_edt`` evaluates. CPU only. Returns float64; every value is
    ``inf`` when ``features`` is all zero.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    features = features.to(torch.uint8).contiguous()
    out = torch.empty(features.shape, dtype=torch.float64)
    _C.edt_sq(features, out, [float(s) for s in spacing])
    return out
