"""Stateless metric functions. Each returns a (B, C) array of per-case, per-class values."""

from exacta.functional.overlap import dice, iou
from exacta.functional.surface import hausdorff_distance, hd95

__all__ = ["dice", "hausdorff_distance", "hd95", "iou"]
