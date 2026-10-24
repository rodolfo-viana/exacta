"""Connected components of binary masks.

``connectivity`` is a required argument because tools default differently: ``scipy.ndimage.label``
uses face neighbours (``connectivity=1``), while MONAI's ``KeepLargestConnectedComponent`` uses
full connectivity (``connectivity=ndim``, 26 neighbours in 3-D). The same mask can have a
different number of components under each.

Every function takes a single image of 1 to 3 dimensions, with no batch or channel axis, as any
DLPack array (PyTorch, NumPy, CuPy, JAX); nonzero elements are foreground. Results come back in
the array library and on the device of the input. Labelling runs on the CPU, so CUDA inputs are
copied there and back.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from exacta import _native
from exacta._inputs import like_input, to_torch


def label(mask: Any, *, connectivity: int) -> tuple[Any, int]:
    """Label the connected components of a binary mask.

    Args:
        mask: A 1-D to 3-D array; nonzero elements are foreground.
        connectivity: Elements are neighbours when their indices differ by at most 1 along each
            axis and along at most ``connectivity`` axes. In 3-D, 1 gives 6 neighbours (shared
            faces), 2 gives 18 (and edges), 3 gives 26 (and corners).

    Returns:
        ``(labels, count)``: int32 labels with the mask's shape, 0 for the background and
        1..count for the components, numbered in C order of their first element exactly as
        ``scipy.ndimage.label`` numbers them.
    """
    labels, count = _label(to_torch(mask), connectivity)
    return like_input(labels, mask), count


def largest(mask: Any, *, connectivity: int) -> Any:
    """The largest connected component of a binary mask, as a boolean array.

    Ties go to the component whose first element comes first in C order. An empty mask gives an
    all-false array. ``connectivity`` is as in :func:`label`.
    """
    labels, count = _label(to_torch(mask), connectivity)
    if count == 0:
        return like_input(torch.zeros(labels.shape, dtype=torch.bool, device=labels.device), mask)
    sizes = torch.bincount(labels.flatten(), minlength=count + 1)
    biggest = int(sizes[1:].argmax()) + 1  # argmax returns the first of tied maxima
    return like_input(labels == biggest, mask)


def remove_small(mask: Any, min_size: int, *, connectivity: int) -> Any:
    """Remove the connected components with fewer than ``min_size`` elements.

    Returns a boolean array that keeps every component of ``min_size`` elements or more.
    ``connectivity`` is as in :func:`label`.
    """
    if min_size < 0:
        raise ValueError(f"min_size must be non-negative, got {min_size}")
    labels, count = _label(to_torch(mask), connectivity)
    keep = torch.bincount(labels.flatten(), minlength=count + 1) >= min_size
    keep[0] = False
    return like_input(keep[labels.long()], mask)


def _label(mask: Tensor, connectivity: int) -> tuple[Tensor, int]:
    if not 1 <= mask.ndim <= 3:
        raise ValueError(f"mask must have 1 to 3 dimensions, got shape {tuple(mask.shape)}")
    if not (isinstance(connectivity, int) and 1 <= connectivity <= mask.ndim):
        raise ValueError(f"connectivity must be an integer in [1, {mask.ndim}], got {connectivity!r}")
    if not _native.available():
        raise RuntimeError("exacta was installed without its compiled extension")
    labels, count = _native.label(mask.cpu(), connectivity)
    return labels.to(mask.device), count
