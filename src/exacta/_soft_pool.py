"""Autograd adapter for fixed-neighbourhood C++ CPU morphology."""
from __future__ import annotations

import torch
from torch import Tensor
from exacta import _native


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, erosion: bool) -> Tensor:  # type: ignore[override]
        source = input.contiguous()
        output = torch.empty_like(source)
        indices = (torch.empty_like(source, dtype=torch.int64) if ctx.needs_input_grad[0]
                   else torch.empty(0, dtype=torch.int64))
        _native._C.soft_pool(source, output, indices, erosion, torch.get_num_threads())
        ctx.save_for_backward(indices)
        ctx.shape = input.shape
        return output

    @staticmethod
    def backward(ctx, gradient: Tensor):  # type: ignore[override]
        indices, = ctx.saved_tensors
        # PyTorch's native scatter-add also supports higher-order derivatives.
        out = gradient.new_zeros(indices.numel())
        out.scatter_add_(0, indices.flatten(), gradient.contiguous().flatten())
        return out.reshape(ctx.shape), None


def pool(input: Tensor, erosion: bool) -> Tensor:
    """First raster-neighbour subgradient at ties; image exterior excluded."""
    return _Pool.apply(input, erosion)
