"""Validated multilabel probability evaluation: fused versus eager/compiled PyTorch.

python benchmarks/bench_probability.py --device cuda --dim 3 --size 128 --output local/probability.json
python benchmarks/bench_probability.py --device cpu --dim 2 --size 512 --compiled --output local/probability-2d.json
"""
from contextlib import ExitStack
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from unittest.mock import patch

import torch

from exacta import __version__, _multilabel, _native, _triton_probability, functional as F
from bench_advanced import cpu_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dim", choices=[2, 3], type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.size, args.classes, args.batch, args.bins, args.threads, args.warmup, args.repeat) < 1:
        parser.error("sizes, bins, threads and repetition counts must be positive")
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.batch, args.classes, *((args.size,) * args.dim))
    logits = torch.randn(shape, generator=generator).to(args.device)
    probabilities = logits.sigmoid()
    target = (torch.rand(shape, generator=generator) > .7).to(args.device)
    mask = (torch.rand(shape, generator=generator) > .2).to(args.device)
    rows = []
    compiled = torch.compile(_multilabel.torch_statistics) if args.compiled else None

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    for name, function in (("brier", F.brier_score), ("nll", F.negative_log_likelihood),
                            ("ece", F.expected_calibration_error)):
        for from_logits, pred in ((False, probabilities), (True, logits)):
            expected = None
            for backend in (["auto", "torch", "compiled"] if args.compiled else ["auto", "torch"]):
                with ExitStack() as stack:
                    if backend != "auto":
                        stack.enter_context(patch.object(_native, "available", return_value=False))
                        stack.enter_context(patch.object(_triton_probability, "available", return_value=False))
                    if backend == "compiled":
                        stack.enter_context(patch.object(_multilabel, "torch_statistics", compiled))

                    def fn():
                        kw = dict(bins=args.bins) if name == "ece" else {}
                        return function(pred, target, multilabel=True, from_logits=from_logits, mask=mask, **kw)

                    actual = fn()
                    if expected is None:
                        expected = actual
                    else:
                        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12, equal_nan=True)
                    for _ in range(args.warmup):
                        fn()
                    sync()
                    initial = torch.cuda.memory_allocated() if args.device == "cuda" else 0
                    if args.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    samples = []
                    for _ in range(args.repeat):
                        sync()
                        start = time.perf_counter()
                        fn()
                        sync()
                        samples.append((time.perf_counter()-start)*1000)
                    peak = torch.cuda.max_memory_allocated()-initial if args.device == "cuda" else None
                    row = dict(metric=name, from_logits=from_logits, backend=backend,
                               median_ms=statistics.median(samples), samples_ms=samples,
                               peak_additional_bytes=peak)
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
    digest = hashlib.sha256()
    for path in sorted([*Path("src").rglob("*.py"), *Path("csrc").rglob("*.cpp"), *Path("csrc").rglob("*.h")]):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    report = dict(exacta=__version__, torch=torch.__version__, python=platform.python_version(),
                  hardware=torch.cuda.get_device_name() if args.device == "cuda" else cpu_model(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256=digest.hexdigest(), dtype="float32", target_dtype="bool",
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
