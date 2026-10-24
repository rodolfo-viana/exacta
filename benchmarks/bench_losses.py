"""Measure segmentation losses, forward plus backward, on identical seeded inputs.

    python benchmarks/bench_losses.py --classes 4 --sizes 128
    python benchmarks/bench_losses.py --device cpu --sizes 32 --backends torch compiled

Compilation and warm-up are excluded. CUDA timings synchronize before and after each step;
CPU timings use wall-clock time. CUDA extra memory is the peak allocation beyond the inputs
and the logits' gradient. No target gradients or higher-order derivatives are requested.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from exacta import __version__
from exacta.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss, GeneralizedDiceCELoss, GeneralizedDiceLoss, TverskyLoss

LOSSES = {"dice": DiceLoss, "dice_ce": DiceCELoss, "tversky": TverskyLoss, "focal": FocalLoss,
          "generalized_dice": GeneralizedDiceLoss, "dice_focal": DiceFocalLoss,
          "generalized_dice_ce": GeneralizedDiceCELoss}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float64": torch.float64}


class SeparateDiceFocal(torch.nn.Module):
    """Two existing auto-dispatched loss calls, to measure the value of fusion."""

    def __init__(self, *, activation, ignore_index):
        super().__init__()
        self.dice = DiceLoss(activation=activation, ignore_index=ignore_index)
        self.focal = FocalLoss(activation=activation, ignore_index=ignore_index)

    def forward(self, input, target, *, mask=None):
        return self.dice(input, target, mask=mask) + self.focal(input, target, mask=mask)


def step(loss_fn, logits, target, mask) -> None:
    logits.grad = None
    loss_fn(logits, target, mask=mask).backward()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(loss_fn, logits, target, mask=None, *, repeat: int, warmup: int) -> tuple[float, float | None]:
    for _ in range(warmup):
        step(loss_fn, logits, target, mask)
    logits.grad = None
    synchronize(logits.device)
    if logits.is_cuda:
        torch.cuda.reset_peak_memory_stats(logits.device)
        base = torch.cuda.memory_allocated(logits.device) + logits.numel() * logits.element_size()
    times = []
    for _ in range(repeat):
        synchronize(logits.device)
        start = time.perf_counter()
        step(loss_fn, logits, target, mask)
        synchronize(logits.device)
        times.append(time.perf_counter() - start)
    extra = (max(0, torch.cuda.max_memory_allocated(logits.device) - base) / 2**20
             if logits.is_cuda else None)
    logits.grad = None
    return statistics.median(times) * 1e3, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--classes", type=int, nargs="+", default=[4, 14])
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 160])
    parser.add_argument("--losses", choices=list(LOSSES), nargs="+", default=list(LOSSES))
    parser.add_argument("--backends", choices=["torch", "compiled", "triton", "separate"], nargs="+",
                        default=["torch", "compiled", "triton"])
    parser.add_argument("--dtypes", choices=list(DTYPES), nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--activation", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--class-weight", type=float, nargs="+", default=None,
                        help="CE class weights for dice_ce and generalized_dice_ce only")
    parser.add_argument("--ignore-fraction", type=float, default=0.0,
                        help="fraction assigned label -100; enables ignore_index when positive")
    parser.add_argument("--mask-fraction", type=float, default=0.0,
                        help="fraction independently excluded by a boolean mask")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda needs a CUDA device")
    if args.device == "cpu" and "triton" in args.backends:
        parser.error("use --backends torch compiled for CPU runs")
    if "separate" in args.backends and args.losses != ["dice_focal"]:
        parser.error("--backends separate requires --losses dice_focal")
    if min(args.batch, args.repeat, args.warmup, *args.classes, *args.sizes) < 1:
        parser.error("batch, repeat, warmup, classes and sizes must be positive")
    if any(not 0 <= fraction <= 1 for fraction in (args.ignore_fraction, args.mask_fraction)):
        parser.error("ignore-fraction and mask-fraction must be in [0, 1]")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"exacta {__version__}, torch {torch.__version__}, CUDA {torch.version.cuda}, {hardware}, "
          f"threads={torch.get_num_threads()}, seed={args.seed}, warmup={args.warmup}, repeat={args.repeat}")
    print(f"activation={args.activation}, ignore_fraction={args.ignore_fraction}, "
          f"mask_fraction={args.mask_fraction}, class_weight={args.class_weight}; "
          "remaining arguments use defaults; median forward + backward\n")
    print("| loss | logits | dtype | backend | included (%) | time (ms) | extra memory (MiB) |")
    print("|---|---|---|---|---:|---:|---:|")
    for dtype_name in args.dtypes:
        for classes in args.classes:
            for size in args.sizes:
                generator = torch.Generator().manual_seed(args.seed)
                logits = torch.randn(args.batch, classes, size, size, size, generator=generator,
                                     dtype=DTYPES[dtype_name]).to(device).requires_grad_()
                target = torch.randint(classes, (args.batch, size, size, size), generator=generator).to(device)
                ignore_index = -100 if args.ignore_fraction else None
                if ignore_index is not None:
                    ignored = (torch.rand(target.shape, generator=generator) < args.ignore_fraction).to(device)
                    target[ignored] = ignore_index
                    del ignored
                mask = ((torch.rand(target.shape, generator=generator) >= args.mask_fraction).to(device)
                        if args.mask_fraction else None)
                valid = target != -100
                if mask is not None:
                    valid &= mask
                included = valid.float().mean().item() * 100
                del valid
                for name in args.losses:
                    for backend in args.backends:
                        options = {}
                        if args.class_weight is not None and name in ("dice_ce", "generalized_dice_ce"):
                            options["class_weight"] = args.class_weight
                        if backend == "separate":
                            loss_fn = SeparateDiceFocal(activation=args.activation, ignore_index=ignore_index)
                        else:
                            loss_fn = LOSSES[name](backend="torch" if backend == "compiled" else backend,
                                                   activation=args.activation, ignore_index=ignore_index, **options)
                        if backend == "compiled":
                            torch._dynamo.reset()
                            loss_fn = torch.compile(loss_fn, fullgraph=True)
                        ms, mib = measure(loss_fn, logits, target, mask, repeat=args.repeat, warmup=args.warmup)
                        memory = "—" if mib is None else f"{mib:,.1f}"
                        shape = f"{args.batch}×{classes}×{size}³"
                        print(f"| {name} | {shape} | {dtype_name} | {backend} | {included:.2f} | "
                              f"{ms:,.3f} | {memory} |", flush=True)
                        del loss_fn


if __name__ == "__main__":
    main()
