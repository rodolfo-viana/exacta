"""Generate the masks behind the lesion-wise reference fixtures (lesion_cases.npz).

    python tests/fixtures/make_lesion_cases.py

Run with any NumPy; the masks are stored, so the fixtures do not depend on the random stream.
Each target holds several ellipsoidal lesions; each prediction perturbs them the ways lesion
matching has to handle: shifted, split in two, merged with a neighbour, missed, and false
positives, with some lesions below the BraTS volume threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def _ellipsoid(shape, center, radii):
    grid = np.indices(shape, dtype=np.float64)
    column = (-1,) + (1,) * len(shape)
    return (((grid - np.reshape(center, column)) / np.reshape(radii, column)) ** 2).sum(0) <= 1


def binary_case(rng, shape):
    target = np.zeros(shape, bool)
    pred = np.zeros(shape, bool)
    for _ in range(rng.integers(3, 8)):
        center = rng.uniform(3, np.array(shape) - 3)
        radii = rng.uniform(1.0, 4.5, len(shape))
        lesion = _ellipsoid(shape, center, radii)
        target |= lesion
        action = rng.choice(["shift", "split", "miss", "grow", "keep"], p=[0.3, 0.2, 0.15, 0.15, 0.2])
        if action == "miss":
            continue
        if action == "shift":
            lesion = np.roll(lesion, tuple(rng.integers(-2, 3, len(shape))), axis=tuple(range(len(shape))))
        elif action == "split":
            cut = int(round(center[0]))
            lesion = lesion.copy()
            lesion[max(cut - 1, 0) : cut + 1] = False
        elif action == "grow":
            lesion = _ellipsoid(shape, center, radii + rng.uniform(0.5, 1.5))
        pred |= lesion
    for _ in range(rng.integers(0, 3)):  # false positives
        pred |= _ellipsoid(shape, rng.uniform(2, np.array(shape) - 2), rng.uniform(0.8, 2.5, len(shape)))
    return pred, target


def brats_case(rng, shape):
    """Label maps with BraTS labels: 1 (necrosis), 2 (oedema), 3 (enhancing)."""
    labels = []
    for _ in range(2):
        labels.append(np.zeros(shape, np.uint8))
    target, pred = labels
    for _ in range(rng.integers(2, 6)):
        center = rng.uniform(6, np.array(shape) - 6)
        radii = rng.uniform(1.5, 6.0, 3)
        for out, jitter in ((target, 0.0), (pred, 1.0)):
            if out is pred and rng.random() < 0.2:
                continue  # missed lesion
            c = center + rng.uniform(-jitter, jitter, 3)
            out[_ellipsoid(shape, c, radii * 1.6)] = 2
            out[_ellipsoid(shape, c, radii)] = 3
            out[_ellipsoid(shape, c, radii * 0.5)] = 1
    for _ in range(rng.integers(0, 3)):  # small false positives
        pred[_ellipsoid(shape, rng.uniform(4, np.array(shape) - 4), rng.uniform(0.8, 2.0, 3))] = 3
    return pred, target


def main() -> None:
    rng = np.random.default_rng(2026)
    arrays = {}
    for i in range(8):
        shape = (40, 44) if i < 4 else (22, 24, 26)
        pred, target = binary_case(rng, shape)
        arrays[f"binary{i}_pred"], arrays[f"binary{i}_target"] = pred, target
    for i in range(5):
        pred, target = brats_case(rng, (48, 52, 44))
        arrays[f"brats{i}_pred"], arrays[f"brats{i}_target"] = pred, target
    np.savez_compressed(HERE / "lesion_cases.npz", **arrays)


if __name__ == "__main__":
    main()
