"""Centreline objectives: differentiable clDice (fused C++ CPU morphology, PyTorch CUDA pooling)
and Skeleton Recall on precomputable tubed target skeletons."""
from __future__ import annotations

import math
from numbers import Integral
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exacta.functional.topology import SkeletonMethod, _skeleton_method, skeletonize
from exacta.losses.geometry import _geometry_input, _options
from exacta.losses.segmentation import (
    Activation, TargetFormat, _activate, _check_ignore_index, _check_index_target, _check_mask, _global_overlap,
    _one_hot, _overlap_mean, _upcast,
)
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


@torch.no_grad()
def skeleton_recall_target(target: Tensor, *, target_format: TargetFormat = "index",
                           union: bool | None = None, tube: int = 2,
                           method: SkeletonMethod | None = None,
                           ignore_index: int | None = None) -> Tensor:
    """Tubed target skeleton for :class:`SkeletonRecallLoss` (Kirchhoff et al., ECCV 2024).

    As the authors' nnU-Net transform does: the foreground (every label above 0) is thinned
    as one mask, the skeleton is dilated by ``tube`` steps of the face-adjacent neighbourhood
    (their ``iterate_structure(generate_binary_structure(ndim, 1), 2)`` for ``tube=2``, in
    voxel steps whatever the spacing), and each voxel of that tube keeps its target label; tube
    voxels on background become 0. ``union=False`` thins every class separately instead, which
    differs where classes touch. One-hot targets (B, C, *spatial) default to separate channels,
    as overlapping labels require; ``union=True`` there thins channels 1.. together and leaves
    channel 0 empty. ``method`` is the thinning of :func:`exacta.functional.skeletonize`;
    ``None`` uses scikit-image's default (Zhang in 2-D, Lee in 3-D), as the authors do.

    Index targets give a (B, *spatial) label map of the target's dtype; ``ignore_index`` voxels
    are background for thinning and 0 in the result. One-hot targets give a boolean
    (B, C, *spatial) array. Thinning runs on the CPU: compute targets in the data loader, after
    augmentation, as the authors do, rather than once per optimisation step on the GPU.
    """
    if isinstance(tube, bool) or not isinstance(tube, Integral) or tube < 0:
        raise ValueError("tube must be a nonnegative integer")
    if target_format not in ("index", "one-hot"):
        raise ValueError(f"target_format must be 'index' or 'one-hot', got {target_format!r}")
    index = target_format == "index"
    spatial = target.ndim - (1 if index else 2)
    if spatial not in (2, 3):
        raise ValueError("target must be (B, *spatial) indices or (B, C, *spatial) channels, 2-D or 3-D")
    method = _skeleton_method(method, spatial)
    ignore_index = _check_ignore_index(ignore_index, target_format)
    union = index if union is None else union
    from exacta.lesions import _dilate  # avoid an import cycle through exacta.functional

    def tubed(region: Tensor) -> Tensor:
        skeleton = skeletonize(region, method=method)
        return _dilate(skeleton, int(tube), 1) if tube else skeleton

    source = target.detach().cpu()
    if index:
        if target.dtype.is_floating_point or target.dtype.is_complex or target.dtype == torch.bool:
            raise TypeError("index targets must hold integer labels")
        labels = source.long()
        if ignore_index is not None:
            labels = torch.where(labels == ignore_index, 0, labels)
        if bool((labels < 0).any()):
            raise ValueError("index targets must be nonnegative (pass ignore_index for ignored labels)")
        out = torch.zeros_like(source)
        for b in range(source.shape[0]):
            if union:
                found = tubed(labels[b] > 0)
                out[b] = torch.where(found, labels[b], 0).to(source.dtype)
                continue
            for label in labels[b].unique().tolist():
                if label > 0:
                    region = labels[b] == label
                    out[b][tubed(region) & region] = label
        return out.to(target.device)
    if not bool(((source == 0) | (source == 1)).all()):
        raise ValueError("one-hot targets must be binary")
    masks = source.bool()
    out = torch.zeros_like(masks)
    for b in range(masks.shape[0]):
        if union:
            if masks.shape[1] < 2:
                raise ValueError("union=True needs a background channel 0 and foreground channels")
            out[b, 1:] = tubed(masks[b, 1:].any(0)) & masks[b, 1:]
            continue
        for c in range(masks.shape[1]):
            out[b, c] = tubed(masks[b, c]) & masks[b, c]
    return out.to(target.device)


class SkeletonRecallLoss(nn.Module):
    """Skeleton Recall loss (Kirchhoff et al., ECCV 2024): ``1 - mean_c R_c``, with the soft recall
    ``R_c = (Σ p_c·s_c + smooth) / (Σ s_c + smooth)`` of the tubed target skeleton ``s`` (see
    :func:`skeleton_recall_target`) by the predicted probabilities ``p``.

    It rewards covering thin structures end to end without thinning the prediction, so it is
    cheap on the GPU; combine it with a regional loss (the authors add it to Dice + CE with weight
    1). The authors' code returns ``-mean R``; this differs by the constant 1 only. Sums cover the
    spatial axes, and the batch with ``batch_dice=True``; the mean is over cases and the selected
    classes (background excluded by default, as the authors require for softmax outputs). A class
    without skeleton voxels has recall 1 and no gradient. ``mask`` and ``ignore_index`` exclude
    voxels from both sums; entirely excluded cases are left out of the mean. ``distributed=True``
    (with ``batch_dice``) sums over the global batch with DDP-correct gradients.

    ``forward(input, target, target_skeleton=None)`` thins ``target`` on the CPU when no
    precomputed ``target_skeleton`` (the output of :func:`skeleton_recall_target` for the same
    target) is given.
    """

    activation: Activation
    target_format: TargetFormat
    include_background: bool
    batch_dice: bool
    distributed: bool
    smooth: float
    ignore_index: int | None
    union: bool | None
    tube: int
    method: SkeletonMethod | None

    def __init__(self, *, activation: Activation = "softmax", target_format: TargetFormat = "index",
                 include_background: bool = False, batch_dice: bool = False, distributed: bool = False,
                 smooth: float = 1e-5, ignore_index: int | None = None, union: bool | None = None,
                 tube: int = 2, method: SkeletonMethod | None = None) -> None:
        super().__init__()
        _options(activation, target_format)
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be finite and positive, so that absent skeletons are defined")
        if distributed and not batch_dice:
            raise ValueError("distributed=True requires batch_dice=True")
        if isinstance(tube, bool) or not isinstance(tube, Integral) or tube < 0:
            raise ValueError("tube must be a nonnegative integer")
        if method not in (None, "zhang", "lee"):
            raise ValueError(f"method must be 'zhang', 'lee' or None, got {method!r}")
        self.activation, self.target_format = activation, target_format
        self.include_background, self.batch_dice, self.distributed = include_background, batch_dice, distributed
        self.smooth = smooth
        self.ignore_index = _check_ignore_index(ignore_index, target_format)
        self.union, self.tube, self.method = union, int(tube), method

    def forward(self, input: Tensor, target: Tensor, *, target_skeleton: Tensor | None = None,
                mask: Tensor | None = None) -> Tensor:
        valid = _check_mask(mask, input)
        input = _upcast(input)
        if self.target_format == "index":
            _check_index_target(target, input)
            if self.ignore_index is not None:
                observed = target.long() != self.ignore_index
                valid = observed if valid is None else valid & observed
        if target_skeleton is None:
            target_skeleton = skeleton_recall_target(
                target, target_format=self.target_format, union=self.union, tube=self.tube,
                method=self.method, ignore_index=self.ignore_index,
            )
        elif target_skeleton.shape != target.shape or target_skeleton.device != target.device:
            raise ValueError("target_skeleton must have the target's shape and device")
        if self.target_format == "index" and self.ignore_index is not None:
            target_skeleton = torch.where(target_skeleton.long() == self.ignore_index, 0, target_skeleton)
        truth = _one_hot(target_skeleton, input, self.target_format).to(input.dtype)
        if valid is not None:
            input = torch.where(valid[:, None], input, 0)  # sanitise before the activation
        probs, complement = _activate(self.activation, input)
        if valid is not None:
            probs = torch.where(valid[:, None], probs, 0)
            complement = torch.where(valid[:, None], complement, 0)
            truth = torch.where(valid[:, None], truth, 0)
        if not self.include_background:
            if input.shape[1] < 2:
                raise ValueError("include_background=False needs at least 2 channels")
            probs, complement, truth = probs[:, 1:], complement[:, 1:], truth[:, 1:]
        dims = tuple(range(2, probs.ndim))
        if self.batch_dice:
            dims = (0, *dims)
        # 1 - (covered + smooth) / (skeleton + smooth) = missed / (covered + missed + smooth), with
        # covered = Σ p·s, missed = Σ (1-p)·s and skeleton = covered + missed, sums of non-negative terms.
        covered, missed = (probs * truth).sum(dims), (complement * truth).sum(dims)
        counts = None if valid is None else valid.flatten(1).sum(1)
        if counts is not None and self.batch_dice:
            counts = counts.sum()
        if self.distributed:
            (covered, missed), counts = _global_overlap((covered, missed), counts, input)
        return _overlap_mean(missed, covered + self.smooth, counts).to(input.dtype)
