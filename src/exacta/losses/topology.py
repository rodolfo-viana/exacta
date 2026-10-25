"""Differentiable clDice using fused C++ CPU morphology and PyTorch CUDA pooling."""
from __future__ import annotations

import math
from numbers import Integral
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exacta.losses.geometry import _geometry_input, _options
from exacta.losses.segmentation import Activation, TargetFormat
from exacta import _native, _soft_pool

SkeletonBackend = Literal["auto", "torch", "native"]


def soft_skeletonize(probs: Tensor, *, iterations: int = 10, backend: SkeletonBackend = "auto") -> Tensor:
    """Iterative soft skeleton (Shit et al., 2021) for (B,C,H,W) or (B,C,D,H,W).

    Cross-shaped min-pooling erosion, full 3x3[x3] max-pooling dilation and ReLU.
    Border pooling excludes values outside the image, matching the authors' code.
    ``iterations`` counts erosions after the initial opening. Thick structures can
    require more iterations; this finite-iteration surrogate is not a Lee skeleton.
    ``backend='native'`` uses fused C++ CPU morphology with PyTorch autograd;
    ``'torch'`` uses pooling on CPU/CUDA. ``'auto'`` uses C++ for eager CPU and
    PyTorch elsewhere (including torch.compile). Native ties choose the first
    raster neighbour, a valid subgradient that can differ from PyTorch at ties.
    Both paths support higher-order differentiation away from nondifferentiable ties.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if probs.ndim not in (4, 5) or not probs.is_floating_point() or any(n == 0 for n in probs.shape):
        raise ValueError("probs must be nonempty floating-point (B,C,H,W) or (B,C,D,H,W)")
    if probs.dtype in (torch.float16, torch.bfloat16):
        probs = probs.float()
    if backend not in ("auto", "torch", "native"):
        raise ValueError("backend must be 'auto', 'torch' or 'native'")
    native = backend == "native" or (backend == "auto" and probs.device.type == "cpu"
                                      and not torch.compiler.is_compiling() and _native.available())
    if native and (probs.device.type != "cpu" or not _native.available()):
        raise NotImplementedError("native soft skeletons require CPU and the compiled extension")
    ndim = probs.ndim - 2
    pool = F.max_pool2d if ndim == 2 else F.max_pool3d

    def erode(x: Tensor) -> Tensor:
        if native:
            return _soft_pool.pool(x, True)
        erosions = []
        for axis in range(ndim):
            kernel = tuple(3 if i == axis else 1 for i in range(ndim))
            padding = tuple(1 if i == axis else 0 for i in range(ndim))
            erosions.append(-pool(-x, kernel, stride=1, padding=padding))
        value = erosions[0]
        for e in erosions[1:]:
            value = torch.minimum(value, e)
        return value

    def opening(x: Tensor) -> Tensor:
        eroded = erode(x)
        return _soft_pool.pool(eroded, False) if native else pool(eroded, 3, stride=1, padding=1)

    skeleton = F.relu(probs - opening(probs))
    for _ in range(iterations):
        probs = erode(probs)
        delta = F.relu(probs - opening(probs))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class SoftClDiceLoss(nn.Module):
    """``1 - 2*tprec*tsens/(tprec+tsens)``, averaged over cases and classes.

    tprec=(sum(skel(p)*y)+smooth)/(sum(skel(p))+smooth), and tsens exchanges
    p/y. Uses the finite-iteration soft skeleton above; it does not equal hard
    clDice. ``smooth`` must be positive so empty skeletons are defined. Target
    labels must be complete and binary. No ignore/mask support: removing voxels
    changes connectivity. Combine with a regional objective for thick/full masks.
    ``target_skeleton`` can cache soft target skeletons for selected channels.
    """

    iterations: int
    smooth: float
    activation: Activation
    target_format: TargetFormat
    include_background: bool
    backend: SkeletonBackend

    def __init__(self, *, iterations: int = 10, smooth: float = 1.0,
                 activation: Activation = "softmax", target_format: TargetFormat = "index",
                 include_background: bool = False, backend: SkeletonBackend = "auto") -> None:
        super().__init__()
        _options(activation, target_format)
        if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be finite and positive")
        self.iterations, self.smooth = iterations, smooth
        self.activation, self.target_format = activation, target_format
        self.include_background = include_background
        if backend not in ("auto", "torch", "native"):
            raise ValueError("backend must be 'auto', 'torch' or 'native'")
        self.backend = backend

    def forward(self, input: Tensor, target: Tensor, *, target_skeleton: Tensor | None = None) -> Tensor:
        probs, truth, _ = _geometry_input(input, target, self.activation, self.target_format, self.include_background)
        sp = soft_skeletonize(probs, iterations=self.iterations, backend=self.backend)
        if target_skeleton is None:
            with torch.no_grad():
                st = soft_skeletonize(truth, iterations=self.iterations, backend=self.backend)
        else:
            if target_skeleton.shape != truth.shape or target_skeleton.device != truth.device:
                raise ValueError("target_skeleton must match selected target channels, shape and device")
            st = target_skeleton.detach().to(probs.dtype)
            if not bool(torch.isfinite(st).all() & (st >= 0).all() & (st <= truth).all()):
                raise ValueError("target_skeleton must be finite and lie between zero and target")
        dims = tuple(range(2, probs.ndim))
        tp = ((sp * truth).sum(dims) + self.smooth) / (sp.sum(dims) + self.smooth)
        ts = ((st * probs).sum(dims) + self.smooth) / (st.sum(dims) + self.smooth)
        return (1 - 2 * tp * ts / (tp + ts)).mean()
