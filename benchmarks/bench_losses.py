"""Time and measure the memory of exacta's losses, forward plus backward, on GPU.

    python benchmarks/bench_losses.py --classes 4 14 --sizes 128 192

"Extra memory" is the peak allocation during a training step beyond the logits and their
gradient, i.e. what the loss itself costs.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from exacta.losses import DiceCELoss


def step(loss_fn, logits, target) -> None:
    logits.grad = None
    loss_fn(logits, target).backward()


def measure(backend: str, dtype: torch.dtype, batch: int, classes: int, size: int, repeat: int):
    logits = torch.randn(batch, classes, size, size, size, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, classes, (batch, size, size, size), device="cuda")
    loss_fn = DiceCELoss(backend=backend)
    step(loss_fn, logits, target)  # warm-up: Triton compilation, caching allocator
    logits.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() + logits.numel() * logits.element_size()  # logits + gradient
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step(loss_fn, logits, target)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    extra = max(0, torch.cuda.max_memory_allocated() - base) / 2**20
    return statistics.median(times) * 1e3, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--classes", type=int, nargs="+", default=[4, 14])
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 160])
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}, DiceCELoss forward + backward\n")
    print("| logits | dtype | backend | time (ms) | extra memory (MiB) |")
    print("|---|---|---|---:|---:|")
    for dtype in (torch.float32, torch.bfloat16):
        for classes in args.classes:
            for size in args.sizes:
                for backend in ("torch", "triton"):
                    ms, mib = measure(backend, dtype, args.batch, classes, size, args.repeat)
                    shape = f"{args.batch}×{classes}×{size}³"
                    print(f"| {shape} | {str(dtype)[6:]} | {backend} | {ms:,.1f} | {mib:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
