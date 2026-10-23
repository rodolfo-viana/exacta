"""Synthetic segmentations for tests and benchmarks."""

from __future__ import annotations

import numpy as np


def blobs(shape, num_classes: int, *, seed: int, n_blobs: int = 3, noise: float = 0.01) -> np.ndarray:
    """Label map of random ellipsoids per foreground class, with salt-and-pepper label noise."""
    rng = np.random.default_rng(seed)
    shape = tuple(shape)
    grid = np.indices(shape, dtype=np.float64)
    expand = (slice(None),) + (None,) * len(shape)
    labels = np.zeros(shape, dtype=np.int64)
    for c in range(1, num_classes):
        for _ in range(n_blobs):
            center = rng.uniform(0, shape)
            radii = rng.uniform(0.1, 0.3, len(shape)) * np.array(shape)
            inside = (((grid - center[expand]) / radii[expand]) ** 2).sum(0) <= 1
            labels[inside] = c
    flip = rng.random(shape) < noise
    labels[flip] = rng.integers(0, num_classes, flip.sum())
    return labels


def pair(
    shape, num_classes: int, *, seed: int, batch: int = 2, noise: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """A (prediction, target) batch of label maps that overlap partially."""
    target = np.stack([blobs(shape, num_classes, seed=seed + i, noise=noise) for i in range(batch)])
    pred = target.copy()
    rng = np.random.default_rng(seed + 1000)
    for i in range(batch):
        other = blobs(shape, num_classes, seed=seed + 2000 + i, n_blobs=1)
        region = rng.random(shape) < 0.3
        pred[i][region & (other > 0)] = other[region & (other > 0)]
        pred[i] = np.roll(pred[i], rng.integers(-2, 3, len(shape)), axis=tuple(range(len(shape))))
    return pred, target
