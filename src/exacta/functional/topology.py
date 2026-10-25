"""Hard centreline Dice using Lee skeletons, independent of the soft training surrogate."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from exacta import _native
from exacta._inputs import InputFormat, like_input, to_torch, validate


@torch.no_grad()
def skeletonize(mask: Any) -> Any:
    """Lee thinning of one binary 2-D/3-D mask, returning a boolean array.

    Runs the compiled C++ CPU kernel (same deletion order as scikit-image's
    ``skeletonize(method='lee')``). CUDA inputs are copied to CPU and back: Lee's
    sequential rechecking is not GPU parallel. For repeated GPU metric evaluation,
    precompute skeletons explicitly. SoftClDiceLoss stays on the GPU throughout.
    The image exterior is background. No spatial resampling is performed.
    """
    x = to_torch(mask)
    if x.ndim not in (2, 3):
        raise ValueError("skeletonize expects one 2-D or 3-D mask")
    if not bool(((x == 0) | (x == 1)).all()):
        raise ValueError("mask must contain only 0 and 1")
    if not _native.available():
        raise RuntimeError("skeletonize requires Exacta's compiled extension")
    source = x.to(device="cpu", dtype=torch.uint8).contiguous()
    result = torch.empty_like(source)
    _native._C.skeleton(source, result)
    return like_input(result.bool().to(x.device), mask)


@torch.no_grad()
def cldice(pred: Any, target: Any, *, input_format: InputFormat = "index",
           num_classes: int | None = None, include_background: bool = True,
           both_empty: float = math.nan, pred_skeleton: Any = None,
           target_skeleton: Any = None) -> Any:
    """Hard clDice: harmonic mean of skeleton precision and skeleton sensitivity.

    Returns (B, C) float64 in the prediction's array library. One-empty masks give 0;
    both-empty use ``both_empty``. A nonempty mask with an empty Lee skeleton gets
    score 0 (Lee thinning can erase even-width objects). Precomputed skeletons must
    be binary subsets of their masks, with shape (B, selected_C, *spatial).
    Skeleton voxels are counted uniformly; anisotropic voxels are not resampled.
    """
    seg = validate(to_torch(pred), to_torch(target), input_format=input_format,
                   num_classes=num_classes, include_background=include_background)
    shape = (seg.batch, len(seg.classes), *seg.pred.shape[-seg.ndim:])
    cached: list[Tensor | None] = []
    for value in (pred_skeleton, target_skeleton):
        if value is None:
            cached.append(None)
            continue
        x = to_torch(value)
        if x.shape != shape or x.device != seg.pred.device:
            raise ValueError(f"skeleton must have shape {shape} and the segmentation's device")
        if not bool(((x == 0) | (x == 1)).all()):
            raise ValueError("skeleton must be binary")
        cached.append(x.bool())
    out = torch.empty(shape[:2], dtype=torch.float64, device=seg.pred.device)
    for b in range(seg.batch):
        for col, cls in enumerate(seg.classes):
            p, t = seg.masks(b, cls)
            sp = skeletonize(p) if cached[0] is None else cached[0][b, col]
            st = skeletonize(t) if cached[1] is None else cached[1][b, col]
            if bool((sp & ~p).any() | (st & ~t).any()):
                raise ValueError("skeleton must be a subset of its mask")
            np, nt = sp.sum().double(), st.sum().double()
            a, b_hits = (sp & t).sum().double(), (st & p).sum().double()
            denominator = a * nt + b_hits * np
            value = 2 * a * b_hits / denominator.clamp_min(1)
            out[b, col] = torch.where(~p.any() & ~t.any(), both_empty, value)
    return like_input(out, pred)
