"""Connected components of binary masks.

``connectivity`` is a required argument because tools default differently: ``scipy.ndimage.label``
uses face neighbours (``connectivity=1``), while MONAI's ``KeepLargestConnectedComponent`` uses
full connectivity (``connectivity=ndim``, 26 neighbours in 3-D). The same mask can have a
different number of components under each.

Every function takes a single image of 1 to 3 dimensions, with no batch or channel axis, as any
DLPack array (PyTorch, NumPy, CuPy, JAX); nonzero elements are foreground. Results come back in
the array library and on the device of the input. ``backend="auto"`` labels CPU inputs with the
native kernel and CUDA inputs with a Triton kernel; both give the same labels.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from exacta import _native, _triton_label
from exacta._inputs import like_input, to_torch

Backend = Literal["auto", "native", "triton"]


def label(mask: Any, *, connectivity: int, backend: Backend = "auto") -> tuple[Any, int]:
    """Label the connected components of a binary mask.

    Args:
        mask: A 1-D to 3-D array; nonzero elements are foreground.
        connectivity: Elements are neighbours when their indices differ by at most 1 along each
            axis and along at most ``connectivity`` axes. In 3-D, 1 gives 6 neighbours (shared
            faces), 2 gives 18 (and edges), 3 gives 26 (and corners).
        backend: ``"native"`` (CPU kernel), ``"triton"`` (GPU kernel), or ``"auto"`` for the one
            that matches the input's device. CUDA inputs given ``"native"`` are copied to the CPU.

    Returns:
        ``(labels, count)``: int32 labels with the mask's shape, 0 for the background and
        1..count for the components, numbered in C order of their first element exactly as
        ``scipy.ndimage.label`` numbers them.
    """
    labels, count = _label(to_torch(mask), connectivity, backend)
    return like_input(labels, mask), count


def largest(mask: Any, *, connectivity: int, backend: Backend = "auto") -> Any:
    """The largest connected component of a binary mask, as a boolean array.

    Ties go to the component whose first element comes first in C order. An empty mask gives an
    all-false array. ``connectivity`` and ``backend`` are as in :func:`label`.
    """
    labels, count = _label(to_torch(mask), connectivity, backend)
    if count == 0:
        return like_input(torch.zeros(labels.shape, dtype=torch.bool, device=labels.device), mask)
    biggest = int(_sizes(labels, count).argmax()) + 1  # argmax returns the first of tied maxima
    return like_input(labels == biggest, mask)


def remove_small(mask: Any, min_size: int, *, connectivity: int, backend: Backend = "auto") -> Any:
    """Remove the connected components with fewer than ``min_size`` elements.

    Returns a boolean array that keeps every component of ``min_size`` elements or more.
    ``connectivity`` and ``backend`` are as in :func:`label`.
    """
    if min_size < 0:
        raise ValueError(f"min_size must be non-negative, got {min_size}")
    labels, count = _label(to_torch(mask), connectivity, backend)
    background = torch.zeros(1, dtype=torch.bool, device=labels.device)
    keep = torch.cat([background, _sizes(labels, count) >= min_size])
    if labels.device.type == "cpu" and _native.available():
        kept = _native.select_labels(labels, keep)
        if kept is None:
            raise RuntimeError("labels out of range")  # unreachable: labelling numbers 0..count
        return like_input(kept, mask)
    return like_input(keep[labels], mask)


def _sizes(labels: Tensor, count: int) -> Tensor:
    """Number of elements (int64, (count,)) of each component 1..count of a label map."""
    if labels.device.type == "cpu" and _native.available():
        sizes = _native.histogram(labels, count + 1)
        if sizes is None:
            raise RuntimeError("labels out of range")  # unreachable: labelling numbers 0..count
        return sizes[1:]
    # Only the foreground: atomic additions to the background's count would take turns.
    return torch.bincount(labels[labels > 0], minlength=count + 1)[1:]


def _label(mask: Tensor, connectivity: int, backend: Backend) -> tuple[Tensor, int]:
    if not 1 <= mask.ndim <= 3:
        raise ValueError(f"mask must have 1 to 3 dimensions, got shape {tuple(mask.shape)}")
    if not (isinstance(connectivity, int) and 1 <= connectivity <= mask.ndim):
        raise ValueError(f"connectivity must be an integer in [1, {mask.ndim}], got {connectivity!r}")
    on_gpu = mask.device.type == "cuda"
    if backend == "auto":
        backend = "triton" if on_gpu and _triton_label.available() else "native"
    if backend == "triton":
        if not on_gpu:
            raise NotImplementedError("the Triton kernel runs on GPU; use backend='native' or 'auto' on CPU")
        if not _triton_label.available():
            raise RuntimeError("backend='triton' needs Triton, which this PyTorch build does not include")
        return _triton_label.label(mask, connectivity)
    if backend != "native":
        raise ValueError(f"backend must be 'auto', 'native' or 'triton', got {backend!r}")
    if not _native.available():
        raise RuntimeError("exacta was installed without its compiled extension")
    labels, count = _native.label(mask.cpu(), connectivity)
    return labels.to(mask.device), count
