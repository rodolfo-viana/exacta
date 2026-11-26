#!/usr/bin/env python3
"""Example driver for Exacta: a short tour of the public API on small synthetic inputs.

Run it with ``python examples/driver.py`` (CPU) or ``python examples/driver.py --device cuda``.
The driver draws no random numbers: every input is built from integer arithmetic, so the output
is the same on every machine. It prints values with ``repr``, which writes a float64 with all the
digits needed to read it back bit for bit, and it writes a short description of the environment
to standard error. Its standard output on CPU is stored in ``examples/driver_expected.txt``, and
``tests/test_example_driver.py`` compares the two.

What is compared, and how:

* Lines without a marker come from computations that Exacta rounds exactly or counts exactly
  (distance transform, Hausdorff, ASSD and NSD scores, overlap counts). They are the same bits on
  every platform, thread count and device; the test compares them as text.
* Lines that end with ``(approx)`` come from the losses, which run on PyTorch operations (CPU) or
  fused Triton kernels (GPU). Their last bits depend on the PyTorch build and on the device, so
  the test compares them to a relative tolerance of 1e-9.

Sections: (1) inputs, (2) exact distance transform against a brute-force oracle, (3) surface
metrics with the three surface models, (4) a tie-sensitive NSD tolerance, (5) presets that
reproduce the conventions of MONAI, MedPy, DeepMind's surface-distance and MeshMetrics, (6) a
stateful metric and the signature of a result, (7) losses with value and gradient norm.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

import exacta
import exacta.functional as EF
from exacta.conventions import parse_signature, signature
from exacta.losses import DiceCELoss, DiceLoss, distance_maps
from exacta.metrics import HD95

# Voxel size in millimetres, in the axis order of the arrays: anisotropic, and none of the three
# values is a dyadic rational, so products such as 3 * 0.4 are inexact in binary64.
SPACING = (1.2, 0.4, 0.4)
# NSD tolerance. Three voxels of 0.4 mm are 1.2 mm in exact arithmetic but 1.2000000000000002 in
# binary64, so surface elements that lie exactly on the tolerance are ties.
TOLERANCE = 1.2

# What each reference tool computes under its name, as arguments of the functions in
# exacta.functional (see the "Conventions" page of the documentation).
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "monai": {  # monai.metrics: voxel surfaces, HD95 as the maximum of the directional percentiles
        "hd95": {"surface": "voxels", "combine": "max"},
        "assd": {"surface": "voxels", "combine": "pooled"},
        "nsd": {"surface": "voxels"},  # float32 comparison: exact ties count as within, as the default
    },
    "medpy": {  # medpy.metric.binary 0.5.2: voxel surfaces, directions pooled
        "hd95": {"surface": "voxels", "combine": "pooled"},
        "assd": {"surface": "voxels", "combine": "pooled"},
    },
    "deepmind": {  # surface-distance: surfel areas; its ASSD pair has the two directed means
        "hd95": {"surface": "surfels", "combine": "max"},
        "assd": {"surface": "surfels", "combine": "mean"},
        "nsd": {"surface": "surfels", "tie_rtol": 0.0},  # compares in binary64, like SciPy
    },
    "meshmetrics": {  # MeshMetrics 1.x: triangle meshes; hd95, assd (pooled) and masd (mean)
        "hd95": {"surface": "mesh", "combine": "max"},
        "assd": {"surface": "mesh", "combine": "pooled"},
        "nsd": {"surface": "mesh"},
    },
}


def fmt(value: Any) -> str:
    """Text of a number, bool, string or sequence of them, with floats written by ``repr``."""
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(fmt(v) for v in value) + "]"
    return repr(value)


def emit(label: str, value: Any, *, approx: bool = False) -> None:
    print(f"{label}: {fmt(value)}{' (approx)' if approx else ''}")


def heading(title: str) -> None:
    print(f"\n# {title}")


def balls(shape: Sequence[int], centres: Sequence[Sequence[int]], radii: Sequence[int]) -> np.ndarray:
    """Union of discrete balls: voxels whose squared index distance to a centre is at most r**2."""
    grid = np.indices(shape)
    mask = np.zeros(shape, dtype=bool)
    for centre, radius in zip(centres, radii):
        mask |= sum((g - c) ** 2 for g, c in zip(grid, centre)) <= radius**2
    return mask


def make_cases() -> tuple[torch.Tensor, torch.Tensor]:
    """Two 22 x 26 x 24 cases. The target of both is a ball. The prediction of case 0 is that ball
    moved three voxels along the last axis (1.2 mm) plus a small separate ball; the prediction of
    case 1 is a larger ball with the same centre."""
    shape = (22, 26, 24)
    target = balls(shape, [(10, 13, 11)], [7])
    shifted = balls(shape, [(10, 13, 14), (3, 3, 3)], [7, 2])
    larger = balls(shape, [(10, 13, 11)], [8])
    target_batch = np.stack([target, target]).astype(np.int64)
    prediction_batch = np.stack([shifted, larger]).astype(np.int64)
    return torch.from_numpy(prediction_batch), torch.from_numpy(target_batch)


def brute_force_distance(features: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Distance of every voxel to the nearest feature voxel, by exhaustive search.

    The squared distance to a feature voxel p is the sum over axes, in axis order, of
    fl(fl((q_i - p_i) * h_i) ** 2), each product and sum rounded once; the oracle takes the
    minimum over p and then the correctly rounded square root.
    """
    points = np.argwhere(features)
    grid = np.indices(features.shape).reshape(features.ndim, -1).T
    total = np.zeros((grid.shape[0], points.shape[0]))
    for axis, h in enumerate(spacing):
        total = total + ((grid[:, None, axis] - points[None, :, axis]).astype(np.float64) * h) ** 2
    return np.sqrt(total.min(1)).reshape(features.shape)


def section_distance_transform(device: torch.device) -> bool:
    heading("2. Exact Euclidean distance transform (losses.distance_maps)")
    spacing = (1.3, 0.3, 0.7)
    mask = balls((5, 6, 7), [(2, 3, 3)], [2]) | balls((5, 6, 7), [(4, 0, 6)], [1])
    maps = distance_maps(torch.from_numpy(mask[None, None].astype(np.int64)).to(device), spacing=spacing)
    got = maps[0, 0].cpu().numpy()
    # Unsigned map: distance to the nearest background voxel inside the mask, to the nearest
    # foreground voxel outside it.
    oracle = np.where(mask, brute_force_distance(~mask, spacing), brute_force_distance(mask, spacing))
    emit("spacing", list(spacing))
    emit("dtype", str(maps.dtype))
    emit("shape", list(maps.shape))
    emit("maximum", float(got.max()))
    emit("sum (math.fsum)", math.fsum(got.ravel().tolist()))
    emit("distances along the first axis, column (3, 3)", got[:, 3, 3].tolist())
    equal = bool(np.array_equal(got, oracle))
    emit("equal to the brute-force oracle, bit for bit", equal)
    return equal


def section_surface_metrics(pred: torch.Tensor, target: torch.Tensor) -> None:
    heading("3. Surface metrics, one value per case (voxels, surfels, mesh)")
    emit("spacing", list(SPACING))
    for surface in ("voxels", "surfels", "mesh"):
        kw: dict[str, Any] = dict(spacing=SPACING, num_classes=2, include_background=False, surface=surface)
        emit(f"hd.{surface}", EF.hausdorff_distance(pred, target, **kw)[:, 0])
        emit(f"hd95.{surface}", EF.hd95(pred, target, **kw)[:, 0])
        emit(f"assd.{surface}", EF.assd(pred, target, **kw)[:, 0])
    counts = EF.confusion_counts(pred, target, num_classes=2, include_background=False)
    emit("dice", EF.dice(pred, target, num_classes=2, include_background=False)[:, 0])
    emit("iou", EF.iou(pred, target, num_classes=2, include_background=False)[:, 0])
    emit("true positives per case", counts.tp[:, 0])


def section_ties(pred: torch.Tensor, target: torch.Tensor) -> None:
    heading("4. NSD where distances tie with the tolerance")
    emit("3 * 0.4", 3 * 0.4)
    emit("tolerance", TOLERANCE)
    for surface in ("voxels", "surfels", "mesh"):
        kw: dict[str, Any] = dict(spacing=SPACING, num_classes=2, include_background=False,
                                  tolerance=TOLERANCE, surface=surface)
        default = EF.nsd(pred, target, **kw)[:, 0]
        strict = EF.nsd(pred, target, tie_rtol=0.0, **kw)[:, 0]
        emit(f"nsd.{surface}, default tie_rtol=1e-06", default)
        emit(f"nsd.{surface}, tie_rtol=0 (strict binary64)", strict)


def section_presets(pred: torch.Tensor, target: torch.Tensor) -> None:
    heading("5. Presets that reproduce other tools (case 0)")
    for tool, options in PRESETS.items():
        kw: dict[str, Any] = dict(spacing=SPACING, num_classes=2, include_background=False)
        hd95 = EF.hd95(pred, target, **kw, **options["hd95"])[0, 0]
        assd = EF.assd(pred, target, **kw, **options["assd"])[0, 0]
        emit(f"{tool}.hd95", float(hd95))
        emit(f"{tool}.assd", float(assd))
        if "nsd" in options:
            nsd = EF.nsd(pred, target, tolerance=TOLERANCE, **kw, **options["nsd"])[0, 0]
            emit(f"{tool}.nsd at {TOLERANCE}", float(nsd))


def section_result_and_signature(pred: torch.Tensor, target: torch.Tensor) -> bool:
    heading("6. A stateful metric, and the signature of a result")
    metric = HD95(num_classes=2, include_background=False, surface="surfels")
    metric.update(pred, target, spacing=SPACING)
    result = metric.compute()
    emit("HD95.per_case", result.per_case[:, 0])
    emit("HD95.per_class", result.per_class)
    emit("HD95.mean", float(result.mean))
    text = signature("nsd", tolerance=TOLERANCE, surface="mesh")
    name, options, version = parse_signature(text)
    emit("signature", text.rsplit("@", 1)[0])
    emit("signature version is the installed version", version == exacta.__version__)
    kw: dict[str, Any] = dict(spacing=SPACING, num_classes=2, include_background=False)
    direct = EF.nsd(pred, target, tolerance=TOLERANCE, surface="mesh", **kw)
    rebuilt = EF.evaluate(pred, target, metrics={name: options}, **kw)[name]
    same = bool(torch.equal(direct, rebuilt))
    emit("the signature recomputes the same bits", same)
    return same


def section_losses(device: torch.device) -> bool:
    heading("7. Losses: value and gradient norm")
    shape = (2, 3, 6, 7, 8)
    # Dyadic logits from integer arithmetic, and labels from a second integer formula.
    index = torch.arange(int(np.prod(shape)), dtype=torch.int64).reshape(shape)
    logits = (((index * 7 + index // 5) % 19 - 9).to(torch.float64) / 4).to(device).requires_grad_()
    spatial = torch.arange(2 * 6 * 7 * 8, dtype=torch.int64).reshape(2, 6, 7, 8)
    labels = ((spatial * 3 + spatial // 7) % 3).to(device)
    emit("logits", list(logits.shape))
    emit("label counts", torch.bincount(labels.flatten().cpu(), minlength=3).tolist())

    def run(loss_fn: torch.nn.Module) -> tuple[float, float]:
        logits.grad = None
        value = loss_fn(logits, labels)
        value.backward()
        assert logits.grad is not None
        return float(value.detach()), float(logits.grad.norm())

    dice = run(DiceLoss(include_background=False))
    dice_ce = run(DiceCELoss(include_background=False))
    emit("DiceLoss value", dice[0], approx=True)
    emit("DiceLoss gradient norm", dice[1], approx=True)
    emit("DiceCELoss value", dice_ce[0], approx=True)
    emit("DiceCELoss gradient norm", dice_ce[1], approx=True)
    cross_entropy = float(torch.nn.functional.cross_entropy(logits.detach(), labels))
    composed = abs(dice_ce[0] - dice[0] - cross_entropy) <= 1e-12
    emit("DiceCELoss is DiceLoss plus torch cross_entropy, to 1e-12", composed)
    return composed


def describe_environment(device: torch.device) -> None:
    """Environment of the run, on standard error so that standard output stays portable."""
    print(
        f"exacta {exacta.__version__}, torch {torch.__version__}, numpy {np.__version__}, "
        f"python {platform.python_version()}, {platform.platform()}, device {device}, "
        f"torch threads {torch.get_num_threads()}",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="device of the tensors (default: cpu)")
    device = torch.device(parser.parse_args(argv).device)
    describe_environment(device)

    heading("1. Inputs (no random numbers: integer arithmetic only)")
    pred, target = make_cases()
    emit("prediction, target shape", list(pred.shape))
    emit("foreground voxels, prediction", pred.sum((1, 2, 3)))
    emit("foreground voxels, target", target.sum((1, 2, 3)))
    pred, target = pred.to(device), target.to(device)

    checks = [section_distance_transform(device)]
    section_surface_metrics(pred, target)
    section_ties(pred, target)
    section_presets(pred, target)
    checks.append(section_result_and_signature(pred, target))
    checks.append(section_losses(device))

    heading("Summary")
    emit("all checks passed", all(checks))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
