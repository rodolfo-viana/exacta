"""Measure shared hard-metric evaluation, including validation and GPU synchronization.

python benchmarks/bench_masked_metrics.py --device cpu --size 128 --threads 2
python benchmarks/bench_masked_metrics.py --device cuda --size 128 --output local/masked-gpu.json
Use --unmasked-only to compare the same workload against earlier installed releases.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from unittest.mock import patch

import torch

import exacta
from exacta import _native, _triton_counts, functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=171)
    parser.add_argument("--keep", type=float, default=.8)
    parser.add_argument("--unmasked-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.size, args.classes, args.batch, args.threads, args.warmup, args.repeat) < 1:
        parser.error("sizes, threads, warmup and repeat must be positive")
    if not 0 <= args.keep <= 1:
        parser.error("keep must lie in [0, 1]")
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.batch, args.size, args.size, args.size)
    target = torch.randint(args.classes, shape, generator=generator).to(args.device)
    pred = target.roll(1, -1)
    mask = (torch.rand(shape, generator=generator) < args.keep).to(args.device)
    ignored = torch.where(mask, target, -100)
    one_hot = [torch.nn.functional.one_hot(x, args.classes).movedim(-1, 1).bool().contiguous()
               for x in (pred, target)]
    names = ["dice", "iou", "sensitivity", "precision", "specificity", "volume_similarity",
             "relative_volume_difference"]
    cases = [("index_unmasked", pred, target, {}),
             ("one_hot_unmasked", *one_hot, {"input_format": "one-hot"})]
    if not args.unmasked_only:
        cases += [("index_mask", pred, target, {"mask": mask}),
                  ("index_ignore", pred, ignored, {"ignore_index": -100}),
                  ("index_both", pred, ignored, {"mask": mask, "ignore_index": -100}),
                  ("one_hot_mask", *one_hot, {"input_format": "one-hot", "mask": mask})]

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    rows = []
    for label, p, t, options in cases:
        expected = None
        for backend in ("auto", "torch"):
            with ExitStack() as stack:
                if backend == "torch":
                    stack.enter_context(patch.object(_native, "available", return_value=False))
                    stack.enter_context(patch.object(_triton_counts, "available", return_value=False))

                def fn():
                    return F.evaluate(p, t, metrics=names, num_classes=args.classes, **options)

                actual = fn()
                if expected is None:
                    expected = actual
                else:
                    for name in names:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, equal_nan=True)
                for _ in range(args.warmup):
                    fn()
                samples = []
                for _ in range(args.repeat):
                    sync()
                    start = time.perf_counter()
                    fn()
                    sync()
                    samples.append((time.perf_counter() - start) * 1000)
                peak = None
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    initial = torch.cuda.memory_allocated()
                    result = fn()
                    sync()
                    peak = torch.cuda.max_memory_allocated() - initial
                    del result
                row = dict(case=label, backend=backend, median_ms=statistics.median(samples),
                           min_ms=min(samples), max_ms=max(samples), samples_ms=samples,
                           peak_additional_bytes=peak)
                rows.append(row)
                print(f"{label:18s} {backend:5s} {row['median_ms']:9.3f} ms  peak={peak}", flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/exacta").rglob("*.py")) + sorted((root / "csrc").rglob("*.cpp"))
    sources += sorted((root / "csrc").rglob("*.h")) + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), exacta=exacta.__version__,
                  imported_from=exacta.__file__, python=platform.python_version(), torch=torch.__version__,
                  device=args.device, hardware=torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
                  platform=platform.platform(), source_sha256=digest.hexdigest(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, results=rows,
                  note="Source hash describes the checkout; imported_from identifies the measured installed package. "
                       "GPU atomic reductions use integer counts; timings vary with scheduling and clock state.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
