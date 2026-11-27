"""The nearest-element search returns the minimum, over all elements, of the computed
point-to-element distance, also where its pruning margin has to widen: very thin triangles
(strongly anisotropic spacings), coordinates far larger than the elements, degenerate triangles."""

import itertools

import numpy as np
import pytest
import torch

from exacta import _native
from exacta.functional.surface import _mesh

pytestmark = pytest.mark.skipif(not _native.available(), reason="needs the compiled extension")


def _each_alone(points, elements):
    # One element per call is a tree of one leaf, where nothing is pruned: the kernel's own
    # distance to that element. NaN (a degenerate triangle) never becomes the nearest.
    each = torch.stack([_native.point_mesh_distances(points, element[None]) for element in elements])
    return torch.nan_to_num(each, nan=float("inf")).min(0).values


def _near_vertices(elements, rng, count=60):
    """Points just off random vertices and edge midpoints, at heights from 0 to 10 element sizes
    along the triangle's normal, nudged in the plane by a few hundred ulps of the coordinates."""
    elements = elements.numpy()
    picked = elements[rng.choice(len(elements), size=min(count, len(elements)), replace=False)]
    a, b, c = picked[:, 0], picked[:, 1], picked[:, 2]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    size = np.linalg.norm(b - a, axis=1, keepdims=True)
    ulp = np.abs(picked).max() * 2.0**-52
    points = []
    for base in (a, b, c, (a + b) / 2, (b + c) / 2):
        for height in (0.0, 1e-6, 1.0, 10.0):
            nudge = rng.normal(size=base.shape) * ulp * rng.uniform(0, 500, (len(base), 1))
            points.append(base + height * size * normal + nudge)
    return torch.from_numpy(np.concatenate(points))


@pytest.mark.parametrize("ratio", [1.0, 1e2, 1e4, 1e6])
def test_nearest_element_with_anisotropic_spacing(ratio):
    rng = np.random.default_rng(5)
    grid = np.indices((9, 8, 7), dtype=np.float64)
    target = ((grid - np.array([4.0, 3.5, 3.0]).reshape(3, 1, 1, 1)) ** 2).sum(0) <= 6
    pred = rng.random((9, 8, 7)) < 0.2
    for axes in sorted(set(itertools.permutations((1.0, 1.0, ratio)))):
        spacing = tuple(0.37 * s for s in axes)
        elements = _mesh(torch.from_numpy(target), spacing).elements
        points = torch.cat([_mesh(torch.from_numpy(pred), spacing).centres, _near_vertices(elements, rng)])
        assert torch.equal(_native.point_mesh_distances(points, elements), _each_alone(points, elements))


@pytest.mark.parametrize(
    "shape, ratio, offset",
    [
        ((6, 6, 6), 1e5, 0.0),  # the fixed margin covers the bound
        ((2, 2, 2), 2e5, 0.0),  # a few very thin triangles: the bound exceeds the fixed margin
        ((6, 6, 6), 1e5, 1e3),  # far from the origin: outside the bound's range, nothing pruned
        ((6, 6, 6), 1e2, 1e9),
    ],
)
def test_nearest_element_of_thin_triangles(shape, ratio, offset):
    rng = np.random.default_rng(6)
    target = rng.random(shape) < 0.3 if min(shape) > 2 else np.ones(shape, bool)
    spacing = (0.37, 0.37, 0.37 * ratio)
    elements = _mesh(torch.from_numpy(target), spacing).elements + offset * max(spacing)
    points = _near_vertices(elements, rng)
    assert torch.equal(_native.point_mesh_distances(points, elements), _each_alone(points, elements))


def test_nearest_element_with_degenerate_triangles():
    rng = np.random.default_rng(7)
    elements = rng.random((120, 3, 3)) * 10
    elements[::7, 2] = elements[::7, 0] + 0.5 * (elements[::7, 1] - elements[::7, 0])  # collinear
    elements[::11, 1] = elements[::11, 0]  # two vertices equal
    elements = torch.from_numpy(elements)
    points = torch.from_numpy(rng.random((300, 3)) * 12 - 1)
    assert torch.equal(_native.point_mesh_distances(points, elements), _each_alone(points, elements))
