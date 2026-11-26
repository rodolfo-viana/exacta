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
