"""Bit-level digest of a fixed set of Exacta results, to compare platforms and compilers.

``run`` evaluates deterministic computations (exact Euclidean distance transform, connected
components, skeletons and the surface metrics with every surface model) on inputs drawn with
NumPy's PCG64 stream and integer arithmetic, so the inputs are identical on every machine. It
writes the SHA-256 of each float64/int output and the values in hexadecimal. ``compare`` checks
that digest files written on different machines agree exactly. The CI workflow
``.github/workflows/portability.yml`` runs both across its platform matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

SEED = 20261126
SPACING = (2.5, 0.7, 0.9)


def _blobs(rng: Any, shape: tuple[int, ...], threshold: int) -> Any:
    """Blob-like binary volumes from integer noise: a 7-point box sum, thresholded."""
    import numpy as np

    noise = rng.integers(0, 256, size=shape, dtype=np.int64)
    total = noise.copy()
    for axis in range(noise.ndim - 1):
        for shift in (1, -1):
            total += np.roll(noise, shift, axis=axis + 1)
    return total > threshold


def _inputs() -> tuple[Any, Any]:
    import numpy as np
    import torch

    rng = np.random.default_rng(SEED)
    target = _blobs(rng, (2, 26, 30, 28), threshold=3 * 256 + 60)
    prediction = np.roll(target, (1, 0, 2), axis=(1, 2, 3))
    prediction ^= rng.random(target.shape) < 0.01
    return torch.from_numpy(prediction.astype(np.int64)), torch.from_numpy(target.astype(np.int64))


def _random_masks(shape: tuple[int, ...], seed: int) -> tuple[Any, Any]:
    """Two masks of independent noise (density 1/2): a surface of many small triangles whose areas
    are sensitive to fused multiply-adds (PCG64 draws the same numbers on every machine)."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random(shape) < 0.5), torch.from_numpy(rng.random(shape) < 0.5)


def _percentile_vectors(seed: int) -> list[tuple[Any, float]]:
    """Sorted vectors of grid-like distances (square roots of small integers times a spacing
    factor, so that many are equal) and the levels at which their percentile is taken."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i in range(400):
        small = (1, 2, 3, 8, 20, 99, 100, 101)
        n = int(rng.integers(1, 3000)) if i % 4 else small[int(rng.integers(0, len(small)))]
        factor = (1.0, 0.64, 1.8769, 0.5917779, 0.09)[int(rng.integers(0, 5))]
        d = np.sort(np.sqrt(rng.integers(0, 400, n) * factor))
        levels = (95.0, 50.0, 99.0, 90.0, 0.0, 1.0, 37.5, 99.9, 5.0)
        q = levels[int(rng.integers(0, len(levels)))] if i % 2 else 100.0 * float(rng.random())
        out.append((d, q))
    return out


def _surfel_tables() -> Any:
    """Surfel area/length tables of many spacings; the 2-D ones include spacings at which the C
    library's pow(s, 2) and the correctly rounded square differ."""
    import numpy as np

    from exacta.functional._surfel_tables import surfel_measures

    rng = np.random.default_rng(SEED + 5)
    tables = [surfel_measures(s) for s in ((0.7, 1.37, 0.3), (2.5, 0.7, 0.9), (1.0, 1.0, 1.0), (0.7, 1.3), (1.5, 0.8))]
    tables += [surfel_measures(tuple(float(x) for x in 0.1 + 4.9 * rng.random(3))) for _ in range(40)]
    tables += [surfel_measures(tuple(float(x) for x in 0.1 + 4.9 * rng.random(2))) for _ in range(4000)]
    return np.concatenate(tables)


def _weights_summary() -> Any:
    """Pairwise sums and nearest-rank indices of seeded weight vectors of many lengths."""
    import numpy as np

    from exacta import _native

    rng = np.random.default_rng(SEED + 6)
    out = []
    for n in (1, 2, 7, 8, 20, 127, 128, 129, 1000, 4097, 8191, 8192, 8193, 20000, 100001):
        for kind in range(3):
            w = (rng.random(n) + 0.1, np.full(n, 0.1), np.array([0.125, 0.25, 0.5])[rng.integers(0, 3, n)])[kind]
            out.append(_native.pairwise_sum(w))
            out += [float(_native.nearest_rank(w, q / 100.0)) for q in (5.0, 50.0, 90.0, 95.0, 99.0, 100.0)]
    return np.array(out)


def _digest(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    flat = array.astype(np.float64).ravel() if array.dtype.kind == "f" else array.ravel()
    values = [v.hex() if array.dtype.kind == "f" else str(int(v)) for v in flat.tolist()[:6]]
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "first_values": values,
    }


def _cases() -> dict[str, Callable[[], Any]]:
    import numpy as np
    import torch

    import exacta.components as components
    import exacta.functional as F
    from exacta import _native

    pred, target = _inputs()
    cases: dict[str, Callable[[], Any]] = {}
    features = (target[0] == 0).to(torch.uint8)
    cases["edt_sq"] = lambda: _native.edt_sq(features, SPACING)
    cases["edt_sq_batched"] = lambda: _native.edt_sq((target == 0).to(torch.uint8), SPACING, batch_ndim=1)
    cases["dice"] = lambda: F.dice(pred, target, num_classes=2)
    for surface in ("voxels", "surfels", "mesh"):
        kw: dict[str, Any] = {"spacing": SPACING, "num_classes": 2, "surface": surface}
        cases[f"hd_{surface}"] = partial(F.hausdorff_distance, pred, target, **kw)
        cases[f"hd95_{surface}"] = partial(F.hd95, pred, target, **kw)
        cases[f"assd_{surface}"] = partial(F.assd, pred, target, **kw)
        cases[f"nsd_{surface}"] = partial(F.nsd, pred, target, tolerance=1.5, **kw)
    # Arithmetic between the exact squared distances and the scores: each is a fixed binary64
    # expression, so these are the same bits whatever the CPU's vector instructions, the BLAS, the
    # C library, the PyTorch build or the NumPy version.
    from exacta.functional import surface as S

    spacing_3d, spacing_2d = (0.7, 1.37, 0.3), (0.7, 1.3)
    noise_3d, noise_2d = _random_masks((24, 24, 24), SEED + 1), _random_masks((40, 36), SEED + 2)
    cases["mesh_sizes_3d"] = lambda: S._mesh(noise_3d[0], spacing_3d).sizes.numpy()
    cases["mesh_centres_3d"] = lambda: S._mesh(noise_3d[0], spacing_3d).centres.numpy()
    cases["mesh_sizes_2d"] = lambda: S._mesh(noise_2d[0], spacing_2d).sizes.numpy()
    rng = np.random.default_rng(SEED + 4)
    blobs_3d = tuple(torch.from_numpy(_blobs(rng, (1, 26, 24, 22), threshold=3 * 256 + 40)[0]) for _ in range(2))
    blobs_2d = tuple(torch.from_numpy(_blobs(rng, (1, 44, 40), threshold=2 * 256 + 20)[0]) for _ in range(2))
    for label, (p, t), spacing in (
        ("noise_3d", noise_3d, spacing_3d), ("noise_2d", noise_2d, spacing_2d),
        ("blobs_3d", blobs_3d, spacing_3d), ("blobs_2d", blobs_2d, spacing_2d),
    ):
        kw = {"spacing": spacing, "input_format": "one-hot"}
        for surface in ("surfels", "mesh"):
            sk = {**kw, "surface": surface}
            cases[f"{label}_hd95_{surface}"] = partial(F.hd95, p[None, None], t[None, None], **sk)
            cases[f"{label}_assd_{surface}"] = partial(F.assd, p[None, None], t[None, None], **sk)
            cases[f"{label}_nsd_{surface}"] = partial(F.nsd, p[None, None], t[None, None], tolerance=1.0, **sk)
        cases[f"{label}_hd95_voxels_pooled"] = partial(F.hd95, p[None, None], t[None, None], combine="pooled", **kw)
    vectors = _percentile_vectors(SEED + 3)
    cases["percentile_linear"] = lambda: [S._linear_percentile(d, q) for d, q in vectors]
    cases["surfel_tables"] = _surfel_tables
    cases["pairwise_sum_nearest_rank"] = _weights_summary
    cases["hd95_voxels_torch_edt"] = lambda: F.hd95(pred, target, spacing=SPACING, num_classes=2, backend="torch")
    cases["label_6"] = lambda: components.label(target[0].bool(), connectivity=1)[0]
    cases["label_26"] = lambda: components.label(target[0].bool(), connectivity=3)[0]
    cases["skeleton_lee"] = lambda: F.skeletonize(target[0].bool(), method="lee")
    cases["cldice"] = lambda: F.cldice(pred, target, num_classes=2)
    cases["betti_6"] = lambda: F.betti_numbers(target, connectivity=1, num_classes=2)
    return cases


def run(out: Path) -> None:
    import numpy as np
    import torch

    import exacta

    results = {}
    for name, fn in _cases().items():
        value = fn()
        results[name] = _digest(value)
        print(f"{name:26s} {results[name]['sha256']}  {results[name]['first_values'][:2]}")
    environment = {
        "exacta": exacta.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "system": platform.platform(),
        "seed": SEED,
        "spacing": list(SPACING),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"environment": environment, "results": results}, indent=2, sort_keys=True))
    print(f"wrote {out}")


def compare(files: list[Path]) -> int:
    """Return 0 when every file has the same digests, 1 otherwise (printing the differences)."""
    loaded = {f: json.loads(f.read_text()) for f in files}
    names = sorted({n for d in loaded.values() for n in d["results"]})
    status = 0
    for f, d in loaded.items():
        env = d["environment"]
        print(f"{str(f.parent.name or f):40s} {env['machine']:8s} exacta {env['exacta']} torch {env['torch']}")
    for name in names:
        shas = {f: d["results"].get(name, {}).get("sha256") for f, d in loaded.items()}
        if len(set(shas.values())) != 1:
            status = 1
            print(f"DIFFERENT {name}")
            for f, d in loaded.items():
                print(f"    {f.parent.name:38s} {shas[f]} {d['results'].get(name, {}).get('first_values')}")
    print(f"{len(names)} results over {len(files)} platforms:", "identical" if status == 0 else "DIFFERENCES")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run").add_argument("--out", type=Path, default=Path("digest.json"))
    commands.add_parser("compare").add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        run(args.out)
    else:
        sys.exit(compare(args.files))


if __name__ == "__main__":
    main()
