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
        ("HD95", "exacta torch", "cpu",
         lambda: EF.hd95(pred, target, spacing=spacing, num_classes=classes, backend="torch")),
        ("HD95", "scipy reference", "cpu",
         lambda: reference.per_case(reference.hausdorff, p_np, t_np, classes, spacings=[spacing], percentile=95)),
    ]
    if torch.cuda.is_available():
        pc, tc = pred.cuda(), target.cuda()
        cases += [
            ("Dice", "exacta", "cuda", lambda: EF.dice(pc, tc, num_classes=classes)),
            ("HD95", "exacta torch", "cuda",
             lambda: EF.hd95(pc, tc, spacing=spacing, num_classes=classes, backend="torch")),
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

    cases.append(("HD95", "MONAI", "cpu", monai_hd95))
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
