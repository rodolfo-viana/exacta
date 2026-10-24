"""Time exacta's metrics against reference implementations on synthetic 3D segmentations.

    python benchmarks/bench_metrics.py --sizes 96 128 192 --classes 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import reference  # noqa: E402
from _data import pair  # noqa: E402

import exacta.functional as EF  # noqa: E402
from exacta import components  # noqa: E402

SLOW = 5.0  # seconds; slower methods run once instead of --repeat times


def timed(fn, device: str, repeat: int) -> float:
    def run() -> float:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    first = run()  # warm-up (CUDA context, caching allocator, OpenMP pool)
    if first > SLOW:
        return first
    return statistics.median(run() for _ in range(repeat))


def methods(pred, target, classes: int, spacing):
    p_np, t_np = pred.numpy(), target.numpy()
    cases = [
        ("Dice", "exacta", "cpu", lambda: EF.dice(pred, target, num_classes=classes)),
        ("Dice", "numpy reference", "cpu", lambda: reference.per_case(reference.dice, p_np, t_np, classes)),
        ("HD95", "exacta native", "cpu",
         lambda: EF.hd95(pred, target, spacing=spacing, num_classes=classes, backend="native")),
        ("HD95", "scipy reference", "cpu",
         lambda: reference.per_case(reference.hausdorff, p_np, t_np, classes, spacings=[spacing], percentile=95)),
        ("NSD voxels", "exacta native", "cpu",
         lambda: EF.nsd(pred, target, spacing=spacing, tolerance=1.0, num_classes=classes)),
        ("NSD surfels", "exacta native", "cpu",
         lambda: EF.nsd(pred, target, spacing=spacing, tolerance=1.0, num_classes=classes, surface="surfels")),
        ("HD95 mesh", "exacta native", "cpu",
         lambda: EF.hd95(pred, target, spacing=spacing, num_classes=classes, surface="mesh")),
    ]
    try:
        from meshmetrics import DistanceMetrics
    except ImportError:
        pass
    else:
        def meshmetrics_hd95():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for c in range(classes):
                    dm = DistanceMetrics(verbose=False)
                    dm.set_input(ref=t_np[0] == c, pred=p_np[0] == c, spacing=spacing)
                    dm.hd(95)

        cases.append(("HD95 mesh", "MeshMetrics", "cpu", meshmetrics_hd95))
    from scipy import ndimage

    fg = p_np[0] > 0
    full = ndimage.generate_binary_structure(3, 3)
    cases += [
        ("label (26-conn.)", "exacta native", "cpu", lambda: components.label(fg, connectivity=3)),
        ("label (26-conn.)", "scipy", "cpu", lambda: ndimage.label(fg, full)),
    ]
    try:
        import surface_distance
    except ImportError:
        pass
    else:
        def deepmind_nsd():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for c in range(classes):
                    s = surface_distance.compute_surface_distances(t_np[0] == c, p_np[0] == c, spacing)
                    surface_distance.compute_surface_dice_at_tolerance(s, 1.0)

        cases.append(("NSD surfels", "DeepMind", "cpu", deepmind_nsd))
    if torch.cuda.is_available():
        pc, tc = pred.cuda(), target.cuda()
        fg_cuda = pc[0] > 0
        cases += [
            ("Dice", "exacta", "cuda", lambda: EF.dice(pc, tc, num_classes=classes)),
            ("HD95", "exacta triton", "cuda",
             lambda: EF.hd95(pc, tc, spacing=spacing, num_classes=classes, backend="triton")),
            ("HD95", "exacta torch", "cuda",
             lambda: EF.hd95(pc, tc, spacing=spacing, num_classes=classes, backend="torch")),
            ("NSD voxels", "exacta triton", "cuda",
             lambda: EF.nsd(pc, tc, spacing=spacing, tolerance=1.0, num_classes=classes)),
            ("NSD surfels", "exacta triton", "cuda",
             lambda: EF.nsd(pc, tc, spacing=spacing, tolerance=1.0, num_classes=classes, surface="surfels")),
            ("label (26-conn.)", "exacta triton", "cuda",
             lambda: components.label(fg_cuda, connectivity=3)),
        ]
    try:
        import monai.metrics
    except ImportError:
        return cases

    one_hot = torch.nn.functional.one_hot(pred, classes).movedim(-1, 1)
    one_hot_t = torch.nn.functional.one_hot(target, classes).movedim(-1, 1)

    def monai_hd95():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monai.metrics.compute_hausdorff_distance(
                one_hot, one_hot_t, include_background=True, percentile=95, spacing=spacing
            )

    def monai_nsd():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monai.metrics.compute_surface_dice(
                one_hot, one_hot_t, [1.0] * classes, include_background=True, spacing=spacing
            )

    def monai_nsd_surfels():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monai.metrics.compute_surface_dice(
                one_hot, one_hot_t, [1.0] * classes, include_background=True, spacing=spacing, use_subvoxels=True
            )

    cases += [
        ("HD95", "MONAI", "cpu", monai_hd95),
        ("NSD voxels", "MONAI", "cpu", monai_nsd),
        ("NSD surfels", "MONAI", "cpu", monai_nsd_surfels),
        ("largest component", "exacta native", "cpu", lambda: components.largest(fg, connectivity=3)),
    ]
    if torch.cuda.is_available():
        cases.append(("largest component", "exacta triton", "cuda",
                      lambda: components.largest(fg_cuda, connectivity=3)))
    try:
        import skimage.measure  # noqa: F401  (MONAI's connected components need it)
    except ImportError:
        return cases
    from monai.transforms.utils import get_largest_connected_component_mask

    cases.append(("largest component", "MONAI", "cpu",
                  lambda: get_largest_connected_component_mask(torch.from_numpy(fg), connectivity=3)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 128, 192])
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    spacing = (1.5, 0.8, 0.8)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}, GPU {gpu}")
    print(f"batch 1, {args.classes} classes, spacing {spacing}, median of {args.repeat}\n")
    print("| volume | metric | implementation | device | time (ms) |")
    print("|---|---|---|---|---:|")
    for size in args.sizes:
        p_np, t_np = pair((size,) * 3, args.classes, seed=0, batch=1, noise=0.0)
        pred, target = torch.from_numpy(p_np), torch.from_numpy(t_np)
        for metric, impl, device, fn in methods(pred, target, args.classes, spacing):
            seconds = timed(fn, device, args.repeat)
            print(f"| {size}³ | {metric} | {impl} | {device} | {seconds * 1e3:,.1f} |", flush=True)


if __name__ == "__main__":
    main()
