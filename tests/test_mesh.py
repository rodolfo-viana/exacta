"""The ``surface="mesh"`` representation, checked against VTK and MeshMetrics."""

import math
import warnings

import numpy as np
import pytest
import torch

import exacta.functional as EF
from exacta import _native
from exacta.functional.surface import _mesh
from exacta.metrics import NSD


def _blobs(shape, rng, count=2):
    grid = np.indices(shape, dtype=np.float64)
    column = (-1,) + (1,) * len(shape)
    mask = np.zeros(shape, bool)
    for _ in range(count):
        centre = rng.uniform(4, np.array(shape) - 4)
        radii = rng.uniform(2, 6, len(shape))
        mask |= (((grid - centre.reshape(column)) / radii.reshape(column)) ** 2).sum(0) <= 1
    return mask


def _vtk_mesh(mask):
    """Elements of VTK's discrete marching cubes (flying edges in 2-D) on the padded mask, in
    voxel coordinates: what MeshMetrics meshes."""
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    ndim = mask.ndim
    padded = np.pad(mask.astype(np.uint8), 1)
    image = vtk.vtkImageData()
    image.SetDimensions(*padded.shape, *(1,) * (3 - ndim))
    image.GetPointData().SetScalars(numpy_to_vtk(padded.transpose().ravel(), deep=True))  # x fastest
    algorithm = vtk.vtkDiscreteMarchingCubes() if ndim == 3 else vtk.vtkDiscreteFlyingEdges2D()
    algorithm.SetValue(0, 1)
    algorithm.ComputeScalarsOff()
    algorithm.SetInputData(image)
    algorithm.Update()
    out = algorithm.GetOutput()
    points = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)[:, :ndim] - 1
    cells = out.GetPolys() if ndim == 3 else out.GetLines()
    ids, elements = vtk.vtkIdList(), []
    cells.InitTraversal()
    while cells.GetNextCell(ids):
        elements.append([points[ids.GetId(i)] for i in range(ids.GetNumberOfIds())])
    return np.array(elements)


def _rows(elements):
    """Elements as sorted rows of (centre, size), to compare meshes regardless of order."""
    centres = elements.mean(1)
    if elements.shape[-1] == 3:
        edges = elements[:, 1:] - elements[:, :1]
        sizes = 0.5 * np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    else:
        sizes = np.linalg.norm(elements[:, 1] - elements[:, 0], axis=1)
    rows = np.c_[centres, sizes]
    return rows[np.lexsort(rows.T[::-1])]


@pytest.mark.parametrize("shape", [(9, 10, 11), (20, 18, 16), (7, 7, 7), (30, 34), (40, 9)])
@pytest.mark.parametrize("density", [0.1, 0.4, 0.7])
def test_meshing_reproduces_vtk(shape, density):
    mask = np.random.default_rng(0).random(shape) < density
    elements = _mesh(torch.from_numpy(mask), (1.0,) * len(shape)).elements
    np.testing.assert_array_equal(_rows(elements.numpy()), _rows(_vtk_mesh(mask)))


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) * ab).sum(-1) / (ab**2).sum(), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _triangle_distance(points, a, b, c):
    """Exact distance to a triangle, computed differently from the kernel: the projection onto the
    plane when it falls inside (barycentric test), otherwise the nearest of the three edges."""
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    height = ((points - a) @ normal)
    projected = points - height[:, None] * normal
    basis = np.c_[b - a, c - a]
    uv = np.linalg.lstsq(basis, (projected - a).T, rcond=None)[0].T
    inside = (uv >= 0).all(1) & (uv.sum(1) <= 1)
    edges = np.min([_segment_distance(points, s, e) for s, e in ((a, b), (b, c), (c, a))], axis=0)
    return np.where(inside, np.abs(height), edges)


@pytest.mark.parametrize("dim", [2, 3])
def test_point_mesh_distances_match_brute_force(dim):
    rng = np.random.default_rng(1)
    elements = rng.random((300, dim, dim)) * 10
    points = rng.random((200, dim)) * 12 - 1
    got = _native.point_mesh_distances(torch.from_numpy(points), torch.from_numpy(elements)).numpy()
    distance = _triangle_distance if dim == 3 else _segment_distance
    expected = np.min([distance(points, *element) for element in elements], axis=0)
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    no_mesh = torch.zeros(0, dim, dim, dtype=torch.float64)
    assert np.isinf(_native.point_mesh_distances(torch.from_numpy(points), no_mesh).numpy()).all()


def _meshmetrics(ref, pred, spacing, tolerance):
    meshmetrics = pytest.importorskip("meshmetrics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm = meshmetrics.DistanceMetrics(verbose=False)
        dm.set_input(ref=ref, pred=pred, spacing=spacing)
        return dict(hd=dm.hd(), hd95=dm.hd(95), assd=dm.assd(), masd=dm.masd(), nsd=dm.nsd(tolerance))


def _ours(pred, target, spacing, tolerance, device):
    P, T = (torch.from_numpy(x)[None, None].to(device) for x in (pred, target))
    kw = dict(spacing=spacing, surface="mesh", input_format="one-hot")
    return dict(
        hd=EF.hausdorff_distance(P, T, **kw),
        hd95=EF.hd95(P, T, **kw),
        assd=EF.assd(P, T, **kw),
        masd=EF.assd(P, T, combine="mean", **kw),
        nsd=EF.nsd(P, T, tolerance=tolerance, **kw),
    )


@pytest.mark.parametrize(
    ("shape", "spacing"),
    [((20, 22, 24), (1.2, 0.9, 1.1)), ((18, 18, 18), (1.0, 1.0, 1.0)), ((16, 20, 12), (3.0, 0.8, 0.8))],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_meshmetrics_3d(device, shape, spacing, seed):
    rng = np.random.default_rng(seed)
    pred, target = _blobs(shape, rng), _blobs(shape, rng)
    expected = _meshmetrics(target, pred, spacing, tolerance=1.37)
    got = _ours(pred, target, spacing, 1.37, device)
    for name, value in got.items():
        assert float(value[0, 0]) == pytest.approx(expected[name], rel=1e-9), name


@pytest.mark.parametrize(("shape", "spacing"), [((30, 34), (0.7, 1.3)), ((40, 36), (1.0, 1.0))])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_meshmetrics_2d(device, shape, spacing, seed):
    # MeshMetrics measures 2-D distances to an extruded contour whose points VTK stores in float32,
    # so its values carry float32 rounding (about 1e-7); exacta measures the float64 contour.
    rng = np.random.default_rng(seed)
    pred, target = _blobs(shape, rng), _blobs(shape, rng)
    expected = _meshmetrics(target, pred, spacing, tolerance=1.37)
    got = _ours(pred, target, spacing, 1.37, device)
    for name, value in got.items():
        assert float(value[0, 0]) == pytest.approx(expected[name], rel=1e-6), name


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_2d_matches_a_float64_reference_on_vtk_contours(seed):
    rng = np.random.default_rng(seed)
    shape, spacing = (30, 34), np.array([0.7, 1.3])
    pred, target = _blobs(shape, rng), _blobs(shape, rng)
    segments = {name: _vtk_mesh(mask) * spacing for name, mask in (("pred", pred), ("target", target))}

    def directed(a, b):
        centres = (a[:, 0] + a[:, 1]) / 2
        ab = b[:, 1] - b[:, 0]
        t = np.clip(((centres[:, None] - b[None, :, 0]) * ab[None]).sum(-1) / (ab**2).sum(-1)[None], 0, 1)
        closest = b[None, :, 0] + t[..., None] * ab[None]
        lengths = np.linalg.norm(a[:, 1] - a[:, 0], axis=1)
        return np.sqrt(((centres[:, None] - closest) ** 2).sum(-1)).min(1), lengths

    d_pt, w_p = directed(segments["pred"], segments["target"])
    d_tp, w_t = directed(segments["target"], segments["pred"])
    expected = (np.dot(d_pt, w_p) + np.dot(d_tp, w_t)) / (w_p.sum() + w_t.sum())
    P, T = pred[None, None], target[None, None]
    got = EF.assd(P, T, spacing=tuple(spacing), surface="mesh", input_format="one-hot")
    assert float(got[0, 0]) == pytest.approx(expected, rel=1e-12)


def test_representations_differ():
    rng = np.random.default_rng(3)
    pred, target = _blobs((18, 20, 22), rng)[None, None], _blobs((18, 20, 22), rng)[None, None]
    kw = dict(spacing=(1.5, 0.8, 0.8), input_format="one-hot")
    values = {s: float(EF.assd(pred, target, surface=s, **kw)[0, 0]) for s in ("voxels", "surfels", "mesh")}
    assert len(set(values.values())) == 3


def test_empty_handling():
    empty = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    filled = empty.clone()
    filled[0, 2:5, 2:5, 2:5] = 1
    kw = dict(spacing=1, num_classes=2, surface="mesh")
    assert EF.hd95(empty, empty, **kw)[0, 1].isnan()
    assert EF.hd95(empty, filled, **kw)[0, 1] == math.inf
    assert EF.nsd(filled, empty, tolerance=1, **kw)[0, 1] == 0
    # Identical meshes share every element: distances are exactly 0, not the ~1e-16 that floating
    # point gives (and MeshMetrics reports), so NSD at tolerance 0 is 1.
    assert EF.hausdorff_distance(filled, filled, **kw)[0, 1] == 0
    assert EF.nsd(filled, filled, tolerance=0, **kw)[0, 1] == 1


def test_metric_class_accepts_mesh(device):
    rng = np.random.default_rng(4)
    pred = torch.from_numpy(_blobs((16, 18, 20), rng))[None, None].to(device)
    target = torch.from_numpy(_blobs((16, 18, 20), rng))[None, None].to(device)
    metric = NSD(tolerance=1.0, surface="mesh", input_format="one-hot")
    metric.update(pred, target, spacing=(1.5, 1.0, 1.0))
    kw = dict(spacing=(1.5, 1.0, 1.0), tolerance=1.0, surface="mesh", input_format="one-hot")
    expected = EF.nsd(pred, target, **kw)
    torch.testing.assert_close(metric.compute().per_case, expected)


@pytest.mark.parametrize("shape", [(18, 20, 22), (30, 34)])
def test_shared_elements_are_exactly_zero(shape):
    rng = np.random.default_rng(5)
    target = _blobs(shape, rng)
    pred = target.copy()
    pred[: shape[0] // 2] = _blobs(shape, rng)[: shape[0] // 2]  # agree on one half only
    spacing = (1.3, 0.7, 0.9)[: len(shape)]

    # Shared elements, counted independently on VTK's meshes: identical vertex sets.
    def keys(mask):
        return {tuple(sorted(map(tuple, element))) for element in _vtk_mesh(mask).tolist()}

    shared = keys(pred) & keys(target)
    assert shared  # the masks do agree somewhere
    from exacta.functional.surface import _mesh_distances

    pred_to_target, _ = _mesh_distances([(torch.from_numpy(pred), torch.from_numpy(target))], spacing, None)[0]
    d = pred_to_target.d
    assert int((d == 0).sum()) == len(shared & keys(pred))
    # elsewhere the kernel's distances are untouched
    elements_t = _mesh(torch.from_numpy(target), spacing).elements
    raw = _native.point_mesh_distances(_mesh(torch.from_numpy(pred), spacing).centres, elements_t)
    torch.testing.assert_close(d[d != 0], raw[d != 0], rtol=0, atol=0)
    assert raw[d == 0].max() < 1e-12  # what floating point would have said


def test_no_mesh_element_lies_on_its_block_boundary():
    # What makes the shared-element test exact: elements of different blocks never coincide.
    from exacta.functional._mesh_tables import SEGMENTS_2D, TRIANGLES_3D

    for table, d in ((TRIANGLES_3D, 3), (SEGMENTS_2D, 2)):
        for elements in table:
            for element in elements:
                vertices = [[int(x) for x in element[i * d : (i + 1) * d]] for i in range(d)]
                assert not any(all(v[a] == c for v in vertices) for a in range(d) for c in (0, 2))
