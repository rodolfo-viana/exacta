"""Every surface score is a fixed binary64 expression of the exact squared distances.

After the exact squared distances, a score takes square roots, forms element weights (surfel areas,
mesh areas and lengths), sorts, interpolates or accumulates, and compares with the NSD bound. Each of
these steps is a fixed sequence of correctly rounded operations that does not depend on the CPU's
vector instructions (PyTorch's kernels use fused multiply-adds under AVX2 and not under its default
kernels), on a BLAS library, on the C library's ``pow`` or on the NumPy version. These tests check
the expressions against independent evaluations in Python floats (which round every operation
separately on every platform) and NumPy, and that the scores do not change with the PyTorch kernel
set or with the number of threads.
"""

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import exacta.functional as EF
import reference
from exacta import _native
from exacta.functional import surface as S
from exacta.functional._surfel_tables import NORMALS_3D, surfel_measures

needs_native = pytest.mark.skipif(not _native.available(), reason="needs the compiled extension")


def _noise(shape, seed, density=0.5):
    return np.random.default_rng(seed).random(shape) < density


# --- Mesh element sizes -------------------------------------------------------------------------


def _triangle_area(v):
    """Half the norm of the cross product of the edges from the first vertex, every operation
    rounded once (Python floats): the expression documented in ``exacta_mesh_elements_f64``."""
    e1 = [v[1][i] - v[0][i] for i in range(3)]
    e2 = [v[2][i] - v[0][i] for i in range(3)]
    c0 = e1[1] * e2[2] - e1[2] * e2[1]
    c1 = e1[2] * e2[0] - e1[0] * e2[2]
    c2 = e1[0] * e2[1] - e1[1] * e2[0]
    return 0.5 * math.sqrt((c0 * c0 + c1 * c1) + c2 * c2)


def _segment_length(v):
    x, y = v[1][0] - v[0][0], v[1][1] - v[0][1]
    return math.sqrt(x * x + y * y)


@needs_native
@pytest.mark.parametrize(
    ("shape", "spacing"),
    [
        ((24, 24, 24), (0.7, 1.37, 0.3)),  # inexact coordinates: fused and unfused sizes differ
        ((20, 18, 22), (0.769531, 0.769531, 4.0)),
        ((20, 18, 22), tuple(float(np.float32(x)) for x in (1.25, 1.25, 1.37))),  # header-stored
        ((40, 36), (0.7, 1.3)),
        ((40, 36), (0.3, 0.3)),
    ],
)
def test_mesh_sizes_are_the_documented_expression(shape, spacing):
    mesh = S._mesh(torch.from_numpy(_noise(shape, 0)), spacing)
    area = _triangle_area if len(shape) == 3 else _segment_length
    expected = [area(element) for element in mesh.elements.tolist()]
    assert len(expected) > 1000
    assert mesh.sizes.tolist() == expected  # bit for bit


@needs_native
def test_mesh_sizes_do_not_use_pytorch_arithmetic(monkeypatch):
    # The sizes come from the compiled kernel: PyTorch's cross product and norm are never called,
    # since their rounding depends on which CPU kernels it dispatches to.
    def refuse(*args, **kwargs):
        raise AssertionError("mesh sizes must not call torch.linalg.cross or torch.linalg.norm")

    monkeypatch.setattr(torch.linalg, "cross", refuse)
    monkeypatch.setattr(torch.linalg, "norm", refuse)
    monkeypatch.setattr(torch.Tensor, "norm", refuse)
    S._mesh(torch.from_numpy(_noise((12, 12, 12), 1)), (0.7, 1.37, 0.3))


# --- Interpolated percentile --------------------------------------------------------------------


def _one_branch(d, q):
    """The interpolation Exacta used before: a + (b - a) * t for every t."""
    n = d.size
    pos = q / 100 * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return float(d[lo] + (d[hi] - d[lo]) * (pos - lo))


def test_linear_percentile_equals_numpy_on_many_vectors():
    rng = random.Random(0)
    mismatches, second_branch, one_branch_differs = 0, 0, 0
    levels = [95.0, 50.0, 99.0, 90.0, 0.0, 1.0, 37.5, 99.9, 5.0, 12.345]
    for trial in range(30000):
        n = rng.choice([1, 2, 3, 4, 5, 7, 8, 20, 99, 100, 101]) if trial % 3 == 0 else rng.randint(1, 3000)
        h = rng.choice([1.0, 0.64, 1.8769, 0.5917779, 0.09])
        d = np.array([math.sqrt(rng.randint(0, 400) * h) for _ in range(n)])
        q = rng.choice(levels) if trial % 2 else rng.uniform(0, 100)
        got = S._linear_percentile(np.sort(d), q)
        reference_value = float(np.percentile(d, q))
        mismatches += got != reference_value
        second_branch += (n - 1) * (q / 100) - math.floor((n - 1) * (q / 100)) >= 0.5
        one_branch_differs += _one_branch(np.sort(d), q) != reference_value
    assert mismatches == 0
    # the vectors exercise both branches of NumPy's lerp, and the old one-branch rule did differ
    assert second_branch > 5000
    assert one_branch_differs > 0


@pytest.mark.parametrize("q", [0.0, 25.0, 95.0, 99.9, 100.0])
def test_percentile_of_distances_equals_numpy(device, q):
    rng = np.random.default_rng(3)
    for n in (1, 2, 17, 1000, 4097):
        d = np.sqrt(rng.integers(0, 500, n) * 0.64)
        x = S._Distances(torch.from_numpy(d).to(device), None)
        assert S._percentile(x, q) == float(np.percentile(d, q))


def test_percentile_with_nan_is_nan_as_in_numpy():
    d = np.array([0.5, 1.5, np.nan, 2.5])
    assert math.isnan(S._linear_percentile(np.sort(d), 50.0))
    assert math.isnan(float(np.percentile(d, 50.0)))


def test_voxel_hd95_follows_medpy_bit_for_bit_where_the_distances_agree():
    # MedPy takes np.percentile of both directions pooled; its distances are SciPy's. Where they
    # are exacta's (they differ in one ulp at a few voxels per million), so is the HD95.
    compared = 0
    for seed in range(24):
        shape = (22, 26, 20) if seed % 3 else (60, 50)
        spacing = ((0.7, 1.37, 0.3) if seed % 2 else (0.8, 0.8, 3.0))[: len(shape)]
        p, t = _noise(shape, seed, 0.45), _noise(shape, seed + 100, 0.45)
        if len(shape) == 3:
            p, t = (_smooth(x) for x in (p, t))
        mine = S._voxel_distances([(torch.from_numpy(p), torch.from_numpy(t))], spacing, S._edt_function("auto", torch.device("cpu")))[0]
        theirs = reference.surface_distances(p, t, spacing)
        if not all(np.array_equal(a.d.numpy(), b) for a, b in zip(mine, theirs)):
            continue
        compared += 1
        expected = float(np.percentile(np.hstack(theirs), 95))
        got = EF.hd95(p[None, None], t[None, None], spacing=spacing, combine="pooled", input_format="one-hot")
        assert float(got[0, 0]) == expected
    assert compared >= 10


def _smooth(mask):
    """Blob-like mask from noise: a face-neighbour majority."""
    total = mask.astype(int)
    for axis in range(mask.ndim):
        for shift in (1, -1):
            total = total + np.roll(mask, shift, axis=axis)
    return total >= 4


# --- Nearest-rank percentile of weighted surfaces -----------------------------------------------


def _deepmind_rule(d, w, q):
    """DeepMind's compute_robust_hausdorff for one direction, in NumPy (the grouping of np.sum
    is that of NumPy 2.3 and later, and of every version for at most 8192 terms)."""
    order = np.lexsort((w, d))
    d, w = d[order], w[order]
    cumulative = np.cumsum(w) / np.sum(w)
    return float(d[min(int(np.searchsorted(cumulative, q / 100.0)), d.size - 1)])


@needs_native
def test_pairwise_sum_is_numpys_for_every_length():
    rng = np.random.default_rng(0)
    for n in [*range(0, 300), 1000, 4096, 8191, 8192]:
        for scale in (1.0, 1e-3, 1e3):
            w = rng.random(n) * scale
            assert _native.pairwise_sum(w) == float(np.sum(w))
            assert S._pairwise_sum(w.tolist()) == float(np.sum(w))


@needs_native
def test_pairwise_sum_does_not_depend_on_the_numpy_version_beyond_8192_terms():
    # np.sum sums in blocks of the iterator's buffer size before NumPy 2.3 (release notes,
    # "Changes to the main iterator and potential numerical changes"); the kernel fixes the grouping
    # of NumPy 2.3 and later, which the Python reference implements independently.
    rng = np.random.default_rng(1)
    for n in (8193, 20000, 100001, 300000):
        w = rng.random(n)
        assert _native.pairwise_sum(w) == S._pairwise_sum(w.tolist())
    mixed = np.array([1e16, 1.0, -1e16, 1.0, 3.5, -2.25, 1e-8] * 3000)  # cancellation
    assert _native.pairwise_sum(mixed) == S._pairwise_sum(mixed.tolist())


@needs_native
@pytest.mark.parametrize("n", [1, 2, 7, 20, 100, 1000, 5000, 8192])
@pytest.mark.parametrize("kind", ["random", "equal", "few_values"])
def test_nearest_rank_matches_the_deepmind_rule(n, kind):
    rng = np.random.default_rng(n)
    d = np.sort(rng.integers(0, 50, n) * 0.5)
    w = {"random": rng.random(n) + 0.1, "equal": np.full(n, 0.1), "few_values": rng.choice([0.125, 0.25, 0.5], n)}[kind]
    for q in (0.0, 1.0, 5.0, 25.0, 50.0, 80.0, 90.0, 95.0, 99.0, 99.9, 100.0):
        assert S._weighted_percentile(d, w, q) == _deepmind_rule(d, w, q), (kind, n, q)


@needs_native
def test_nearest_rank_with_equal_weights_hits_the_level_exactly(monkeypatch):
    # With equal weights the running fraction i / n can equal q / 100 up to rounding, and a one-ulp
    # difference in the total moves the index: the rule is part of the roundings.
    d = np.arange(20, dtype=np.float64)
    w = np.full(20, 0.1)
    expected = {q: _deepmind_rule(d, w, q) for q in (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0)}
    assert len(set(expected.values())) > 1
    for q, value in expected.items():
        assert S._weighted_percentile(d, w, q) == value
    # without the compiled extension the Python rule gives the same
    monkeypatch.setattr(S._native, "available", lambda: False)
    for q, value in expected.items():
        assert S._weighted_percentile(d, w, q) == value


@needs_native
def test_nearest_rank_python_and_native_agree_on_large_inputs():
    rng = np.random.default_rng(5)
    for n in (8193, 50000, 200001):
        w = rng.random(n) + 0.01
        for q in (50.0, 95.0, 99.0, 100.0):
            assert _native.nearest_rank(w, q / 100.0) == S._nearest_rank(w.tolist(), q / 100.0)


# --- Surfel tables ------------------------------------------------------------------------------


def _surface_areas_reference(spacing):
    """Independent evaluation with NumPy elementwise operations (each its own rounding): per code,
    the normals' scaled lengths added from the left."""
    s0, s1, s2 = spacing
    out = np.zeros(256)
    for code, normals in enumerate(NORMALS_3D):
        n = np.array(normals)
        v = np.stack([n[:, 0] * s1 * s2, n[:, 1] * s0 * s2, n[:, 2] * s0 * s1], 1)
        lengths = np.sqrt((v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + v[:, 2] * v[:, 2])
        total = 0.0
        for x in lengths:
            total += float(x)
        out[code] = total
    return out


@pytest.mark.parametrize("spacing", [(0.7, 1.37, 0.3), (2.5, 0.7, 0.9), (0.769531, 0.769531, 4.0), (1.0, 1.0, 1.0)])
def test_3d_surfel_areas_are_explicit_binary64_expressions(spacing):
    np.testing.assert_array_equal(surfel_measures(spacing), _surface_areas_reference(spacing))


def _spacings_where_pow_differs_from_the_square(count):
    """Spacings s for which the C library's pow(s, 2) is not the correctly rounded s * s (a handful
    per thousand with glibc; none where pow is correctly rounded for squares)."""
    rng = random.Random(0)
    found = []
    for _ in range(300000):
        s = rng.uniform(0.1, 5.0)
        if s**2 != s * s:
            found.append(s)
            if len(found) == count:
                break
    return found


def test_2d_contour_lengths_use_the_correctly_rounded_square():
    found = _spacings_where_pow_differs_from_the_square(60)
    if not found:
        pytest.skip("this C library's pow(s, 2) is the correctly rounded square")
    differs = 0
    for s0, s1 in zip(found, reversed(found)):
        diag = 0.5 * math.sqrt(s0 * s0 + s1 * s1)
        table = surfel_measures((s0, s1))
        assert table[0b1000] == diag  # one corner of the block inside
        assert table[0b1001] == 2 * diag  # opposite corners inside
        assert table[0b0011] == s1 and table[0b0101] == s0
        differs += diag != 0.5 * math.sqrt(s0**2 + s1**2)
    assert differs > 0  # pow would have changed some of them


def test_surfel_tables_do_not_call_blas(monkeypatch):
    # np.linalg.norm of a 3-vector is a BLAS dot product (OpenBLAS fuses its multiply-adds on some
    # CPUs and not on others); the table is built from explicit products and sums.
    def refuse(*args, **kwargs):
        raise AssertionError("surfel tables must not call BLAS")

    monkeypatch.setattr(np.linalg, "norm", refuse)
    monkeypatch.setattr(np, "dot", refuse)
    surfel_measures.__wrapped__((0.71, 1.29, 0.33))
    surfel_measures.__wrapped__((0.71, 1.29))


# --- NSD bound ----------------------------------------------------------------------------------


def test_nsd_bound_is_tolerance_times_one_plus_rtol():
    # fl(tau * fl(1 + rho)), as documented: not tau + tau * rho, which rounds differently.
    rng = random.Random(1)
    differs = 0
    for _ in range(2000):
        tau, rho = rng.uniform(0.1, 20.0), rng.choice([1e-6, 1e-9, 0.0, 1e-3])
        assert S._nsd_bounds(tau, range(1), rho) == [tau * (1.0 + rho)]
        differs += tau * (1.0 + rho) != tau + tau * rho
    assert differs > 0


# --- The scores do not depend on PyTorch's kernels, on threads or on the BLAS -------------------

_PROGRAM = textwrap.dedent(
    """
    import hashlib
    import json
    import numpy as np
    import torch
    import exacta.functional as EF
    from exacta.functional import surface as S

    capability = getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unknown")
    out = {"capability": capability(), "threads": torch.get_num_threads()}
    cases = (("3d", (22, 22, 22), (0.7, 1.37, 0.3), 11), ("3d_b", (16, 18, 14), (0.769531, 0.769531, 4.0), 12),
             ("2d", (48, 44), (0.7, 1.3), 13))
    for name, shape, spacing, seed in cases:
        rng = np.random.default_rng(seed)
        p, t = rng.random(shape) < 0.5, rng.random(shape) < 0.5
        P, T = torch.from_numpy(p)[None, None], torch.from_numpy(t)[None, None]
        kw = dict(spacing=spacing, input_format="one-hot")
        for surface in ("voxels", "surfels", "mesh"):
            sk = dict(kw, surface=surface)
            row = {}
            for combine in ("max", "pooled"):
                row["hd95_" + combine] = float(EF.hd95(P, T, combine=combine, **sk)[0, 0]).hex()
            row["hd_max"] = float(EF.hausdorff_distance(P, T, **sk)[0, 0]).hex()
            for combine in ("pooled", "mean"):
                row["assd_" + combine] = float(EF.assd(P, T, combine=combine, **sk)[0, 0]).hex()
            for tau in (1.0, 1.37):
                for rtol in (0.0, 1e-6):
                    row[f"nsd_{tau}_{rtol}"] = float(EF.nsd(P, T, tolerance=tau, tie_rtol=rtol, **sk)[0, 0]).hex()
            out[f"{name}/{surface}"] = row
        mesh = S._mesh(torch.from_numpy(p), spacing)
        for field in ("elements", "centres", "sizes"):
            out[f"{name}/mesh {field}"] = hashlib.sha256(getattr(mesh, field).numpy().tobytes()).hexdigest()
    print(json.dumps(out))
    """
)


def _run_program(**env):
    environment = {**os.environ, **env}
    flags = ["-S"] if sys.flags.no_site else []
    result = subprocess.run(
        [sys.executable, *flags, "-c", _PROGRAM], capture_output=True, text=True, env=environment, timeout=600
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    return {
        "native": _run_program(),
        "default kernels": _run_program(ATEN_CPU_CAPABILITY="default"),
        "1 thread": _run_program(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        "8 threads": _run_program(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8"),
    }


@needs_native
@pytest.mark.parametrize("other", ["default kernels", "1 thread", "8 threads"])
def test_scores_are_identical_across_pytorch_kernels_and_threads(runs, other):
    base, variant = runs["native"], runs[other]
    differing = [k for k in base if k not in ("capability", "threads") and base[k] != variant[k]]
    assert not differing, (
        f"{other} ({variant['capability']}, {variant['threads']} threads) differs from "
        f"native ({base['capability']}, {base['threads']} threads) in {differing}"
    )
