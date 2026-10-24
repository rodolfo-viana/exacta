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


def label(mask: Tensor, connectivity: int) -> tuple[Tensor, int]:
    """Connected-component labels (int32) of a CPU mask and the number of components.

    Components are numbered 1, 2, ... in C order of their first voxel, as scipy.ndimage.label
    numbers them; see ``exacta_label_u8`` in ``kernels.h``.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    # A bool tensor already holds one byte of 0 or 1 per element; viewing it avoids a copy.
    mask = mask.contiguous().view(torch.uint8) if mask.dtype == torch.bool else (mask != 0).to(torch.uint8)
    out = torch.empty(mask.shape, dtype=torch.int32)
    count = _C.label(mask, out, int(connectivity))
    return out, count
