"""Compare shared metric evaluation with separate calls on seeded 3D cuboids.

python benchmarks/bench_collection.py --device cuda --size 64 --repeat 10
python benchmarks/bench_collection.py --device cpu --size 64 --threads 16
"""

import argparse
import statistics
import time

import torch

from exacta import __version__, functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.size, args.threads, args.warmup, args.repeat) < 1:
        parser.error("size, threads, warmup and repeat must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    n = args.size
    target = torch.zeros(2, n, n, n, dtype=torch.long, device=args.device)
    target[:, n//8:n//2, n//8:n//2, n//8:n//2] = 1
    target[:, n//2:7*n//8, n//2:7*n//8, n//2:7*n//8] = 2
    pred = target.roll(2, dims=-1)
    spacing = (1.0, 0.7, 2.0)
    overlap = {name: {} for name in ("dice", "iou", "sensitivity", "precision", "specificity",
                                    "volume_similarity", "relative_volume_difference")}
    surface = {"hd95": {}, "assd": {}, "nsd": {"tolerance": 1.0}}
    hardware = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
    print(f"exacta={__version__}, torch={torch.__version__}, {hardware}, device={args.device}, "
          f"shape=2x{n}^3, classes=3, spacing={spacing}, threads={args.threads}, seed={args.seed}, "
          f"warmup={args.warmup}, repeat={args.repeat}")
    print("| metrics | separate ms | shared ms | speedup |")
    print("|---|---:|---:|---:|")
    for name, selection in (("counts", overlap), ("surfaces", surface), ("all", overlap | surface)):
        def separate():
            return {metric: getattr(F, metric)(pred, target, num_classes=3, include_background=False,
                    **({"spacing": spacing} if metric in surface else {}), **options)
                    for metric, options in selection.items()}

        def shared():
            return F.evaluate(pred, target, metrics=selection, spacing=spacing, num_classes=3,
                              include_background=False)

        gold, actual = separate(), shared()
        for metric in selection:
            torch.testing.assert_close(actual[metric], gold[metric], rtol=0, atol=0, equal_nan=True)
        times = []
        for fn in (separate, shared):
            for _ in range(args.warmup):
                fn()
            samples = []
            for _ in range(args.repeat):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                fn()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            times.append(statistics.median(samples))
        print(f"| {name} | {times[0]:.3f} | {times[1]:.3f} | {times[0]/times[1]:.2f}x |", flush=True)


if __name__ == "__main__":
    main()
