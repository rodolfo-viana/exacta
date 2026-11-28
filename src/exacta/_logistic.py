"""Logistic function and binary cross-entropy that stay accurate where ``sigma(z)`` rounds to one.

For ``z`` of 37 or more (binary64; 17 in binary32), ``sigma(z) = 1 / (1 + exp(-z))`` rounds to 1, so
every expression that subtracts from it or from one loses all its digits, although the result is
representable (``sigma(-40) = 4.2e-18``):

* ``torch.sigmoid`` differentiates as ``sigma * (1 - sigma)``, which is exactly 0 there;
* ``sigma(z) - g``, the gradient of the cross-entropy, is exactly 0 for ``g = 1``;
* ``F.binary_cross_entropy_with_logits`` computes ``(1 - g) z + softplus(-z)``; for ``g = 0`` and
  ``z = -40`` that is ``-40 + (40 + 4.2e-18)``, which is 0 instead of ``4.2e-18``.

Here every term is formed from ``sigma(z)`` and ``sigma(-z)``, each accurate on its own, and from sums
and products of non-negative numbers, so that the relative error stays a few units of roundoff for any
finite logit. Both functions are differentiable twice (their backward passes use the same functions), so
the PyTorch backend of the losses keeps supporting gradient penalties.
"""

from __future__ import annotations

import torch
from torch import Tensor


class _Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(z)
        return torch.sigmoid(z)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:  # type: ignore[override]
        (z,) = ctx.saved_tensors
        # sigma(z) * sigma(-z) rather than sigma * (1 - sigma): no cancellation when sigma rounds to 1.
        return grad * sigmoid(z) * sigmoid(-z)


def sigmoid(z: Tensor) -> Tensor:
    """``torch.sigmoid`` with the derivative ``sigma(z) * sigma(-z)``, accurate at any logit."""
    return _Sigmoid.apply(z)


class _SigmoidPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        ctx.save_for_backward(z)
        return torch.sigmoid(z), torch.sigmoid(-z)

    @staticmethod
    def backward(ctx, grad_p: Tensor, grad_q: Tensor) -> Tensor:  # type: ignore[override]
        (z,) = ctx.saved_tensors
        # q = 1 - p, so that d q = -d p.
        return (grad_p - grad_q) * sigmoid(z) * sigmoid(-z)


def sigmoid_pair(z: Tensor) -> tuple[Tensor, Tensor]:
    """``(sigma(z), sigma(-z))``: the probability and its complement ``1 - p``, each accurate at any logit."""
    if torch.compiler.is_compiling():  # two single-output Functions: as accurate, and the form dynamo has traced
        return sigmoid(z), sigmoid(-z)
    return _SigmoidPair.apply(z)


def _terms(z: Tensor, g: Tensor) -> Tensor:
    # softplus(z) - g z = log1p(exp(-|z|)) + (1 - g) z      for z >= 0
    #                   = log1p(exp(-|z|)) + g |z|          for z < 0
    # Every term is non-negative for g in [0, 1], so nothing cancels.
    return torch.log1p(torch.exp(-z.abs())) + torch.where(z >= 0, (1 - g) * z, -g * z)


class _BCEWithLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, g: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(z, g)
        return _terms(z, g)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor | None, Tensor | None]:  # type: ignore[override]
        z, g = ctx.saved_tensors
        d_z = d_g = None
        if ctx.needs_input_grad[0]:
            # (1 - g) sigma(z) - g sigma(-z) equals sigma(z) - g, with each term accurate on its own.
            d_z = grad * ((1 - g) * sigmoid(z) - g * sigmoid(-z))
        if ctx.needs_input_grad[1]:
            d_g = -grad * z
        return d_z, d_g


def bce_with_logits(z: Tensor, g: Tensor) -> Tensor:
    """Elementwise ``softplus(z) - g z``, for targets ``g`` in [0, 1] of the shape of ``z``.

    Same value as ``F.binary_cross_entropy_with_logits(z, g, reduction="none")`` up to rounding, with
    a relative error of a few roundoffs at any finite logit, and the gradients ``(1 - g) sigma(z) -
    g sigma(-z)`` with respect to the logits and ``-z`` with respect to the target.
    """
    if z.shape != g.shape:
        raise ValueError(f"target size ({tuple(g.shape)}) must match input size ({tuple(z.shape)})")
    return _BCEWithLogits.apply(z, g)
