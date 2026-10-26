"""Class-distance Dice objectives using native PyTorch CPU/CUDA operations."""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

from exacta.losses.segmentation import (
    _check_ignore_index, _check_index_target, _check_index_values, _check_mask,
    _nonnegative_finite, _prepare_masked_input,
)


class GeneralizedWassersteinDiceLoss(nn.Module):
    """Generalized Wasserstein Dice for mutually exclusive labels (Fidon et al., 2017).

    ``input`` contains logits (B, C, *spatial), ``target`` integer labels (B, *spatial).
    ``dist_matrix`` is a fixed, finite, nonnegative, symmetric C x C ground-cost
    matrix with zero diagonal and a positive maximum. It is normalized by that
    maximum and saved as a buffer. Costs concern label confusions, not spatial
    distances. Triangle inequalities are not checked; zero costs between distinct
    labels are allowed, treating those confusions as cost-free.

    The default weighting gives foreground labels weight 1 and background (label
    0) weight 0. It requires every background/foreground cost to be maximal, as
    assumed by the original formulation. ``weighting_mode='GDL'`` weights all
    classes, including background, by 1 / (observed target count + 1), and also
    weights the error term (Tilborghs et al., 2020; MONAI's GDL convention).

    A shared nonnegative ``smooth`` is added to numerator and denominator.
    ``reduction='none'`` returns (B,); mean averages observed cases only. Ignored
    cases, empty inputs and a zero denominator return differentiable zero.
    ``mask`` and ``ignore_index`` follow the other index-target losses, including
    sanitizing excluded nonfinite logits before softmax. Float16/bfloat16 inputs
    compute in float32. Native autograd supports higher derivatives and
    ``torch.compile``. No cross-rank pooling or independent-label sigmoid is implied.
    """

    dist_matrix: Tensor

    def __init__(
        self, dist_matrix: Any, *, weighting_mode: Literal["default", "GDL"] = "default",
        smooth: float = 1e-5, reduction: Literal["none", "mean", "sum"] = "mean",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        if weighting_mode not in ("default", "GDL"):
            raise ValueError("weighting_mode must be 'default' or 'GDL'")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        _nonnegative_finite("smooth", smooth)
        raw = torch.as_tensor(dist_matrix)
        if raw.is_complex():
            raise TypeError("dist_matrix must contain real costs")
        matrix = torch.as_tensor(dist_matrix, dtype=torch.float64).detach().clone()
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("dist_matrix must be square C x C with at least two classes")
        if not bool(torch.isfinite(matrix).all()) or bool((matrix < 0).any()):
            raise ValueError("dist_matrix costs must be finite and nonnegative")
        if bool(matrix.diagonal().ne(0).any()) or not torch.equal(matrix, matrix.T):
            raise ValueError("dist_matrix must be symmetric with a zero diagonal")
        maximum = matrix.max()
        if not bool(maximum > 0):
            raise ValueError("dist_matrix must have a positive maximum")
        if weighting_mode == "default" and not bool(matrix[0, 1:].eq(maximum).all()):
            raise ValueError("default weighting requires maximal background/foreground distances (label 0)")
        self.register_buffer("dist_matrix", matrix / maximum)
        self.num_classes = matrix.shape[0]
        self.weighting_mode = weighting_mode
        self.smooth = float(smooth)
        self.reduction = reduction
        self.ignore_index = _check_ignore_index(ignore_index, "index")

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        mask = _check_mask(mask, input)
        input, target, valid = _prepare_masked_input(input, target, "index", self.ignore_index, mask)
        _check_index_target(target, input)
        if input.shape[1] != self.num_classes:
            raise ValueError("input channels must match dist_matrix size")
        compiling = torch.compiler.is_compiling()
        if not compiling:
            _check_index_values(target, self.num_classes)
        # Empty axes cannot be reshaped with an inferred dimension. Preserve the graph.
        if not target.numel():
            values = input.sum(tuple(range(1, input.ndim))) * 0
            return values if self.reduction == "none" else values.sum()
        shape = (input.shape[0], -1)
        labels = target.long().reshape(shape)
        observed = None if valid is None else valid.reshape(shape)
        safe_labels = labels.clamp(0, self.num_classes - 1) if compiling else labels
        matrix = self.dist_matrix.to(device=input.device, dtype=input.dtype)
        probs = input.softmax(1).flatten(2)
        # Keep voxels contiguous, matching the softmax reduction/backward layout.
        costs = matrix.T[:, safe_labels].permute(1, 0, 2)
        error = (probs * costs).sum(1)
        if observed is not None:
            error = torch.where(observed, error, 0)
        if self.weighting_mode == "GDL":
            # Integer counts avoid both a C-times-larger one-hot target and floating
            # atomic accumulation of class volumes on CUDA.
            counts = torch.zeros((input.shape[0], self.num_classes), dtype=torch.int64, device=input.device)
            counts = counts.scatter_add(1, safe_labels, torch.ones_like(labels) if observed is None else observed.long())
            weight = (counts.to(input.dtype) + 1).reciprocal().gather(1, safe_labels)
        else:
            weight = safe_labels.ne(0)
        if observed is not None:
            weight = torch.where(observed, weight, 0) if self.weighting_mode == "GDL" else weight & observed
        true_positive = (weight * (1 - error)).sum(1)
        all_error = (weight * error).sum(1) if self.weighting_mode == "GDL" else error.sum(1)
        numerator = 2 * true_positive + self.smooth
        denominator = 2 * true_positive + all_error + self.smooth
        if self.smooth > 0:
            values = 1 - numerator / denominator
        else:
            nonzero = denominator != 0
            values = torch.where(nonzero, 1 - numerator / torch.where(nonzero, denominator, 1), 0)
        # Eager execution validates labels above. Compiled execution must not
        # silently turn invalid labels into valid ones or index out of bounds.
        if compiling:
            in_range = (labels >= 0) & (labels < self.num_classes)
            values = torch.where(in_range.all(1), values, torch.nan)
        if self.reduction == "none":
            return values
        if self.reduction == "sum":
            return values.sum()
        if observed is None:
            return values.mean()
        return values.sum() / observed.any(1).sum().clamp_min(1)
