"""Standard classification losses from native PyTorch.

These are direct re-exports, not extra wrappers or custom kernels: their shapes,
weighting, ignore_index, reductions and gradients follow torch.nn exactly.
"""
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, KLDivLoss, NLLLoss

__all__ = ["BCELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "KLDivLoss", "NLLLoss"]
