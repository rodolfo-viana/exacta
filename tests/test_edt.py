import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import ndimage

from exacta import _C, _native, _triton_edt
from exacta.functional.surface import _edt_sq_torch

GPU = torch.cuda.is_available() and _triton_edt.available()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")


def _triton(features, spacing):
    return _triton_edt.edt_sq(features.cuda(), spacing).cpu()


CPU_KERNELS = [_native.edt_sq, _edt_sq_torch]
KERNELS = CPU_KERNELS + [pytest.param(_triton, marks=needs_gpu)]

CASES = [
    ((50,), (0.7,)),
    ((31, 47), (1.0, 2.5)),
    ((17, 23, 29), (3.0, 0.8, 0.8)),
    ((5, 6, 7, 8), (1.0, 2.0, 0.5, 1.5)),
]


def _features(shape, density):
    features = np.random.default_rng(0).random(shape) < density
    features.flat[0] = True
    return features


@pytest.mark.parametrize(("shape", "spacing"), CASES)
@pytest.mark.parametrize("density", [0.001, 0.05, 0.5])
def test_native_matches_scipy(shape, spacing, density):
    features = _features(shape, density)
    expected = ndimage.distance_transform_edt(~features, sampling=spacing) ** 2
    got = _native.edt_sq(torch.from_numpy(features), spacing).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize(("shape", "spacing"), CASES)
@pytest.mark.parametrize("density", [0.001, 0.05, 0.5])
def test_torch_matches_scipy(device, shape, spacing, density):
    features = _features(shape, density)
    expected = ndimage.distance_transform_edt(~features, sampling=spacing) ** 2
    got = _edt_sq_torch(torch.from_numpy(features).to(device), spacing).cpu().numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-9)


def _brute_force(features: np.ndarray, spacing) -> np.ndarray:
    """min over feature voxels p of sum_i ((p_i - q_i) * h_i)^2, rounded and summed in axis
    order: the expression SciPy's distance_transform_edt evaluates for its chosen feature."""
    pts = np.argwhere(features)
    grid = np.indices(features.shape).reshape(features.ndim, -1).T
    total = 0.0
    for i, h in enumerate(spacing):
        total = total + ((pts[None, :, i] - grid[:, None, i]).astype(np.float64) * h) ** 2
    return total.min(1).reshape(features.shape)


@settings(max_examples=300, deadline=None)
@given(
    arrays(np.bool_, st.tuples(st.integers(1, 6), st.integers(1, 7), st.integers(1, 8))),
    st.tuples(*[st.floats(0.1, 5.0)] * 3),
)
def test_is_the_exact_floating_point_minimum(features, spacing):
    if not features.any():
        return
    expected = _brute_force(features, spacing)
    for edt_sq in CPU_KERNELS + ([_triton] if GPU else []):
        np.testing.assert_array_equal(edt_sq(torch.from_numpy(features), spacing).numpy(), expected)


# Lines on which the 0.1.0 envelope missed the rounded minimum by an ulp: parabolas c and c + 1
# whose breakpoint lies one tolerance right of the point where c ties with c + G and c + G + 1,
# G large, so that c + 1 survives as a zero-width tie and c + G is popped. Found by fuzzing the
# worst case of the exactness proof. Each case: n, h, bound, then (index, f) pairs; inf elsewhere.
FAR_TIES = [
    (749, 0.3, 36957.237702106751, ((70, 36948.507702104274), (71, 36957.237702106751),
                                     (728, 3785.3077021038771), (729, 3675.5977021038775))),
    (788, 0.6, 242165.36564522516, ((8, 242152.76564521194), (9, 242165.36564522516),
                                     (775, 40309.045645208796), (776, 39769.405645208812))),
    (794, 1.5726659790111748, 2484073.9947091881, ((67, 2483764.8349238811), (68, 2484073.9947091881),
                                                   (790, 1416221.2569848425), (791, 1412954.0563749289))),
    (1033, 1.1, 2574667.2310094885, ((6, 2574639.401009379), (7, 2574667.2310094885),
                                      (1028, 1340492.641009348), (1029, 1338047.2310093478))),
    (1065, 0.4, 177668.01497944829, ((34, 177653.77497943811), (35, 177668.01497944829),
                                      (1061, 23685.934979433048), (1062, 23371.534979433036))),
    (1131, 0.7, 925503.49714840332, ((138, 925407.94714835926), (139, 925503.49714840332),
                                      (1127, 541112.21714834159), (1128, 540238.54714834166))),
]


def _line_minimum(f: np.ndarray, h: float) -> np.ndarray:
    """Brute-force pass: min over p with finite f[p] of fl(fl(fl(h * (q - p))^2) + f[p])."""
    q = np.arange(f.shape[-1])
    dx = (q[:, None] - q[None, :]).astype(np.float64) * h
    return (dx * dx + f[..., None, :]).min(-1)


def _far_tie_line(n, pairs):
    f = np.full((1, n), np.inf)
    for index, value in pairs:
        f[0, index] = value
    return f


def _line_kernels():
    kernels = [_native.edt_lines]
    if GPU:
        kernels.append(lambda f, h, bound: _triton_edt.edt_lines(f.cuda(), h, bound).cpu())
    return kernels


@pytest.mark.parametrize(("n", "h", "bound", "pairs"), FAR_TIES)
def test_far_apart_ties_keep_the_rounded_minimum(n, h, bound, pairs):
    f = _far_tie_line(n, pairs)
    expected = _line_minimum(f, h)
    for edt_lines in _line_kernels():
        np.testing.assert_array_equal(edt_lines(torch.from_numpy(f), h, bound).numpy(), expected)


def test_lines_with_near_ties_are_exact():
    # Clusters of parabolas through one point, each raised by up to ~3000 eps * M: breakpoints
    # that tie within rounding error and others a tolerance or more apart.
    rng = np.random.default_rng(7)
    eps = np.finfo(np.float64).eps
    for n in (5, 17, 40, 96):
        for h in (0.3, 0.7, 1.1, 2.6457513110645907):
            f = np.full((400, n), np.inf)
            span = (h * n) ** 2
            for row in f:
                for _ in range(rng.integers(1, 4)):
                    x0 = rng.integers(n) * h if rng.random() < 0.6 else rng.random() * (n - 1) * h
                    level = rng.random() * span
                    p = rng.integers(n, size=rng.integers(3, 8))
                    value = level - (x0 - h * p) ** 2
                    value += rng.uniform(-1, 1, p.size) * 10 ** rng.uniform(-1, 3.5, p.size) * eps * (level + span)
                    keep = value >= 0
                    np.minimum.at(row, p[keep], value[keep])
                if np.isinf(row).all():
                    row[rng.integers(n)] = 0.0
            bound = float(f[np.isfinite(f)].max())
            expected = _line_minimum(f, h)
            for edt_lines in _line_kernels():
                np.testing.assert_array_equal(edt_lines(torch.from_numpy(f), h, bound).numpy(), expected)


def test_edt_lines_rejects_bad_arguments():
    f = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        _native.edt_lines(f, 0.0, 1.0)
    with pytest.raises(ValueError):
        _native.edt_lines(f, 1.0, math.inf)


def _edt_lines_checkers():
    """The Python wrapper, the bare extension function (no Python-side check) and, on a GPU, Triton."""
    def bare(f, h, bound):
        out = torch.empty_like(f)
        _C.edt_lines(f, out, h, bound)
        return out

    checkers = [_native.edt_lines, bare]
    if GPU:
        checkers.append(lambda f, h, bound: _triton_edt.edt_lines(f.cuda(), h, bound).cpu())
    return checkers


@pytest.mark.parametrize(
    "bad",
    [math.nan, -1.0, -1e-300, -math.inf, 5.0000000001],
    ids=["nan", "negative", "tiny-negative", "-inf", "above-bound"],
)
def test_edt_lines_rejects_inputs_outside_the_proof(bad):
    # The proof takes every finite input in [0, bound]; +inf marks samples without a parabola.
    f = torch.tensor([[0.0, math.inf, bad, 5.0]], dtype=torch.float64)
    for edt_lines in _edt_lines_checkers():
        with pytest.raises(ValueError):
            edt_lines(f, 1.0, 5.0)


def test_edt_lines_accepts_the_whole_closed_range():
    f = torch.tensor([[0.0, math.inf, 5.0, 2.5, -0.0], [math.inf] * 5], dtype=torch.float64)
    expected = _line_minimum(f.numpy(), 1.0)
    for edt_lines in _edt_lines_checkers():
        torch.testing.assert_close(edt_lines(f, 1.0, 5.0), torch.from_numpy(expected), rtol=0, atol=0)


@pytest.mark.parametrize(("h", "bound", "n"), [(1e-100, 1e250, 3), (1e-50, 1e260, 3), (1.0, 2e300, 3)])
def test_edt_lines_rejects_passes_beyond_the_guarded_range(h, bound, n):
    # (bound + (h n)^2) / h above 1e300: the breakpoints and the tolerance could overflow.
    f = torch.zeros(1, n, dtype=torch.float64)
    for edt_lines in _edt_lines_checkers():
        with pytest.raises(ValueError, match="too anisotropic"):
            edt_lines(f, h, bound)


def test_the_extension_enforces_the_range_guard_without_the_python_check():
    # exacta_edt_sq_f64 is the exported C interface: it must not rely on the Python wrapper.
    features = torch.zeros(3, 20_000, dtype=torch.uint8)
    out = torch.full(features.shape, 7.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="too anisotropic"):
        _C.edt_sq(features, out, [1e100, 1e-100])
    assert (out == 7.0).all()  # rejected before anything is written
    with pytest.raises(ValueError, match="too anisotropic"):
        _C.edt_sq(features.expand(2, 3, 20_000).contiguous(), torch.empty((2, 3, 20_000), dtype=torch.float64),
                  [1e100, 1e-100], batch_ndim=1)
    # a grid inside the limit is accepted: its last pass reaches squared distance / spacing = 4e280
    _C.edt_sq(features[:, :2000].contiguous(), torch.empty(3, 2000, dtype=torch.float64), [1e90, 1e-100])


def test_the_extension_and_the_python_check_agree_on_the_guarded_range():
    rng = np.random.default_rng(3)
    rejected = accepted = 0
    for _ in range(600):
        ndim = int(rng.integers(2, 4))
        shape = [int(n) for n in rng.integers(0, 9, ndim)]  # zeros give empty arrays: no work
        # A large first spacing and a second chosen near the limit: the squared extent of the
        # first axis over the second spacing is about 1e300, on either side of it.
        first = rng.uniform(97, 100)
        second = np.clip(2 * first + 2 * np.log10(max(shape[0] - 1, 1)) - 300 + rng.uniform(-0.5, 0.5), -100, 100)
        spacing = [float(10.0**first), float(10.0**second)] + [float(10.0 ** rng.uniform(-1, 1))] * (ndim - 2)
        features = torch.zeros(shape, dtype=torch.uint8)
        out = torch.empty(shape, dtype=torch.float64)
        try:
            _native.check_edt_extent(shape, spacing)
            python = "ok"
        except ValueError:
            python = "rejected"
        try:
            _C.edt_sq(features, out, spacing)
            kernel = "ok"
        except ValueError as error:
            assert "too anisotropic" in str(error)
            kernel = "rejected"
        assert kernel == python, (shape, spacing)
        rejected += python == "rejected"
        accepted += python == "ok"
    assert rejected > 20 and accepted > 20  # both sides of the limit were exercised


def test_lines_longer_than_the_proof_covers_are_rejected():
    limit = _native.EDT_MAX_LINE
    assert limit == 2**40
    # An array of this shape is empty, so nothing is allocated or computed.
    features = torch.zeros(limit + 1, 0, dtype=torch.uint8)
    with pytest.raises(ValueError, match="too anisotropic"):
        _C.edt_sq(features, torch.empty(features.shape, dtype=torch.float64), [1.0, 1.0])
    with pytest.raises(ValueError, match="too long"):
        _native.check_edt_extent(features.shape, (1.0, 1.0))
    _native.check_edt_extent((limit, 3), (1.0, 1.0))  # the limit itself is allowed


@needs_gpu
@pytest.mark.parametrize(
    ("shape", "spacing"), CASES + [((1, 1, 7), (1.0, 2.0, 0.5)), ((300, 7, 1), (0.5, 1.5, 2.0))]
)
@pytest.mark.parametrize("density", [0.001, 0.05, 0.5])
def test_triton_matches_native_bit_for_bit(shape, spacing, density):
    features = torch.from_numpy(_features(shape, density))
    assert torch.equal(_triton(features, spacing), _native.edt_sq(features, spacing))


# Non-dyadic spacings, where mathematically tied features round to different values. The
# isotropic 1.1 grid has many points where three parabolas meet, which the plain lower-envelope
# algorithm resolves to a value an ulp too high.
@pytest.mark.parametrize("spacing", [(1.1, 1.1, 1.1), (0.7, 0.9, 1.3), (2.0, 0.6, 0.6), (3.0, 0.8, 0.8)])
@pytest.mark.parametrize("density", [0.002, 0.02, 0.2])
def test_backends_agree_bit_for_bit(device, spacing, density):
    features = torch.from_numpy(_features((30, 36, 42), density))
    native = _native.edt_sq(features, spacing)
    assert torch.equal(_edt_sq_torch(features.to(device), spacing).cpu(), native)
    if device.type == "cuda" and GPU:
        assert torch.equal(_triton(features, spacing), native)


@pytest.mark.parametrize("edt_sq", KERNELS)
def test_no_features_is_inf(edt_sq):
    got = edt_sq(torch.zeros(4, 5, 6, dtype=torch.bool), (1.0, 1.0, 1.0))
    assert torch.isinf(got).all()


@pytest.mark.parametrize("edt_sq", KERNELS)
def test_all_features_is_zero(edt_sq):
    got = edt_sq(torch.ones(4, 5, dtype=torch.bool), (1.0, 2.0))
    assert (got == 0).all()


@pytest.mark.parametrize("edt_sq", KERNELS)
def test_empty_array(edt_sq):
    assert edt_sq(torch.zeros(0, 5, dtype=torch.bool), (1.0, 2.0)).shape == (0, 5)


def test_single_thread_matches_default():
    features = torch.rand(20, 30, 40) < 0.01
    single = torch.empty(features.shape, dtype=torch.float64)
    _C.edt_sq(features.to(torch.uint8), single, [1.0, 1.0, 1.0], num_threads=1)
    torch.testing.assert_close(single, _native.edt_sq(features, (1.0, 1.0, 1.0)), rtol=0, atol=0)


@pytest.mark.parametrize("spacing", [[1.0, 0.0], [1.0, -2.0], [1.0, math.inf], [1.0]])
def test_rejects_bad_spacing(spacing):
    features = torch.zeros(3, 4, dtype=torch.uint8)
    with pytest.raises(ValueError):
        _C.edt_sq(features, torch.empty(3, 4, dtype=torch.float64), spacing)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        _C.edt_sq(torch.zeros(3, 4, dtype=torch.uint8), torch.empty(4, 3, dtype=torch.float64), [1.0, 1.0])


@pytest.mark.parametrize(
    ("shape", "spacing"),
    [((3, 20, 21, 22), (1.5, 0.8, 0.7)), ((5, 31, 47), (1.0, 2.5)), ((2, 3, 9, 10, 11), (0.7, 0.9, 1.3))],
)
@pytest.mark.parametrize("edt_sq", KERNELS)
def test_batch_axes_transform_each_image_separately(edt_sq, shape, spacing):
    batch_ndim = len(shape) - len(spacing)
    features = torch.from_numpy(_features(shape, 0.05))
    features[(0,) * batch_ndim] = False  # an image without features is all inf
    expected = torch.stack(
        [_native.edt_sq(image, spacing) for image in features.reshape(-1, *shape[batch_ndim:])]
    ).reshape(shape)
    got = edt_sq(features, spacing, batch_ndim) if edt_sq is not _triton else _triton_edt.edt_sq(
        features.cuda(), spacing, batch_ndim
    ).cpu()
    assert torch.equal(got, expected)


@needs_gpu
def test_triton_scratch_scales_with_lines_not_blocks():
    # Three lines of 400 000: scratch for three lanes, not for a whole block of them.
    features = torch.from_numpy(_features((3, 400_000), 0.0001))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    got = _triton(features, (1.0, 0.5))
    peak = torch.cuda.max_memory_allocated() - base
    assert torch.equal(got, _native.edt_sq(features, (1.0, 0.5)))
    assert peak < 64 << 20, f"{peak / 2**20:.0f} MiB"


@pytest.mark.parametrize(
    "shape",
    [(1000, 1000, 1000), (2**31 - 2,), (2**31 - 32, 1), (2**40, 0), (0, 2**33), (3, 3)],
)
def test_triton_index_limit_accepts_what_int32_indices_cover(shape):
    _triton_edt.check_index_range(shape)


@pytest.mark.parametrize(
    ("shape", "batch_ndim"),
    [
        ((2**31 - 1,), 0),  # a line with more samples than int32 can index
        ((2**31 - 31, 1), 0),  # 2**31 - 31 lines in the pass along the last axis
        ((2, 2**31), 0),
        ((2**20, 2**12, 2**12), 1),  # batch axes count as lines: 2**32 of them
        ((2**31, 1, 5), 1),
    ],
)
def test_triton_index_limit_rejects_what_would_overflow_int32(shape, batch_ndim):
    with pytest.raises(ValueError, match="32-bit integers"):
        _triton_edt.check_index_range(shape, batch_ndim)


def test_triton_limit_is_checked_before_anything_is_allocated():
    # A zero-stride view of one byte: the check sees the shape only, so this costs no memory.
    features = torch.zeros(1, dtype=torch.uint8).expand(2**31 - 1, 1)
    if _triton_edt.available():
        with pytest.raises(ValueError, match="32-bit integers"):
            _triton_edt.edt_sq(features, (1.0, 1.0))
    else:
        with pytest.raises(RuntimeError, match="Triton is not installed"):
            _triton_edt.edt_sq(features, (1.0, 1.0))
    assert "32-bit" in (_triton_edt.edt_sq.__doc__ or "")  # the limit is documented


@needs_gpu
def test_triton_limit_is_checked_on_the_device_tensor():
    features = torch.zeros(1, dtype=torch.uint8, device="cuda").expand(2**31 - 1, 1)
    with pytest.raises(ValueError, match="32-bit integers"):
        _triton_edt.edt_sq(features, (1.0, 1.0))
    with pytest.raises(ValueError, match="32-bit integers"):
        _triton_edt.edt_lines(torch.zeros(1, dtype=torch.float64, device="cuda").expand(2, 2**31 - 1), 1.0, 1.0)


@pytest.mark.parametrize("edt_sq", [_native.edt_sq, pytest.param(_triton, marks=needs_gpu)])
def test_rejects_grids_anisotropic_enough_to_overflow(edt_sq):
    # The first axis contributes (2e100)^2 = 4e200 to the bound of the second pass, whose spacing
    # is 1e-100: squared distance / spacing would reach 4e300, past the guarded range.
    features = torch.zeros(3, 20_000, dtype=torch.bool)
    features[0, 0] = True
    with pytest.raises(ValueError, match="too anisotropic"):
        edt_sq(features, (1e100, 1e-100))


@pytest.mark.parametrize("spacing", [(1e100, 1e100), (1e-100, 1e-100), (1e100, 1.0), (1.0, 1e-100)])
def test_extreme_but_safe_spacings_are_exact(spacing):
    features = torch.from_numpy(_features((7, 9), 0.2))
    features[0, 0] = True
    expected = _edt_sq_torch(features, spacing)
    assert torch.equal(_native.edt_sq(features, spacing), expected)
    if GPU:
        assert torch.equal(_triton(features, spacing), expected)


def _compiled_kernels(jit):
    """Compiled variants of a Triton JIT function, or [] if this Triton keeps them elsewhere."""
    kernels = []
    for entry in getattr(jit, "device_caches", {}).values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        if isinstance(cache, dict):
            kernels += [k for k in cache.values() if hasattr(k, "asm")]
    return kernels


@needs_gpu
def test_triton_arithmetic_is_what_the_proof_assumes():
    # The exactness argument assumes binary64 operations rounded once each: no fused
    # multiply-add and IEEE division. In PTX, add/sub/mul with an explicit rounding modifier
    # (.rn) are never contracted into an fma by ptxas, and div.rn.f64 is correctly rounded.
    import re

    features = torch.from_numpy(_features((9, 11, 13), 0.05))
    _triton(features, (1.1, 0.7, 1.3))
    jits = (_triton_edt._first_pass, _triton_edt._edt_pass)
    kernels = [k for jit in jits for k in _compiled_kernels(jit)]
    if not kernels:
        pytest.skip("this Triton version does not expose compiled kernels")
    for kernel in kernels:
        ops = re.findall(r"\b(add|sub|mul|div|fma|mad)((?:\.[a-z0-9]+)*)\.f64\b", kernel.asm["ptx"])
        assert ops, "no binary64 arithmetic found in the PTX"
        for op, modifiers in ops:
            assert op not in ("fma", "mad"), f"fused {op}{modifiers}.f64 in the EDT kernel"
            assert "rn" in modifiers.split("."), f"{op}{modifiers}.f64 without .rn"
