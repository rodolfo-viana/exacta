"""Reproducible timings for physical areas/volumes, voxel ranking and the added losses.

Loss timings include backward; metrics include validation. Exact versus binned
ranking has different semantics, so report both latency and score differences.
Compilation is excluded from timings. Run without concurrent tests/benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from exacta import __version__, functional as F, losses as L, metrics as M
from bench_advanced import cpu_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=881)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.size, args.batch, args.classes, args.threads, args.warmup, args.repeat) < 1 or args.bins < 2:
        parser.error("positive sizes/threads/repeats and bins>=2 required")
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.batch, args.classes, *([args.size]*args.dim))
    x = torch.randn(shape, generator=generator, device="cpu").to(args.device).requires_grad_()
    y = (torch.rand(shape, generator=generator) > .7).to(args.device)
    mask = (torch.rand((args.batch, *shape[2:]), generator=generator) > .2).to(args.device)
    prob = x.detach().sigmoid()
    rows = []

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def measure(name, fn, backward=False):
        def step():
            x.grad = None
            result = fn()
            if backward:
                result.backward()
        for _ in range(args.warmup):
            step()
        x.grad = None
        sync()
        initial = torch.cuda.memory_allocated() if args.device == "cuda" else 0
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.repeat):
            sync()
            start = time.perf_counter()
            step()
            sync()
            samples.append((time.perf_counter()-start)*1000)
        peak = torch.cuda.max_memory_allocated()-initial if args.device == "cuda" else None
        row = dict(name=name, median_ms=statistics.median(samples), samples_ms=samples,
                   peak_additional_bytes=peak)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
        x.grad = None

    for name, loss in [
        ("squared_dice_torch", L.DiceLoss(squared_pred=True, activation="sigmoid", target_format="one-hot", backend="torch")),
        ("squared_dice_auto", L.DiceLoss(squared_pred=True, activation="sigmoid", target_format="one-hot")),
        ("lovasz_hinge", L.LovaszHingeLoss(multilabel=True)),
    ]:
        expected = loss(x, y, mask=mask)
        grad = torch.autograd.grad(expected, x)[0]
        measure(name, lambda: loss(x, y, mask=mask), True)
        if args.compiled:
            torch._dynamo.reset()
            compiled = torch.compile(loss)
            actual = compiled(x, y, mask=mask)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, x)[0], grad)
            measure(name+"_compiled", lambda: compiled(x, y, mask=mask), True)
    scores = []
    for bins in (None, args.bins):
        name = "exact" if bins is None else f"bins{bins}"
        scores.append(F.voxel_ranking(prob, y, multilabel=True, mask=mask, bins=bins))
        measure("voxel_ap_"+name, lambda: F.average_precision(prob, y, multilabel=True, mask=mask, bins=bins))
        metric = M.VoxelRanking(multilabel=True, bins=bins)

        def update_compute():
            metric.reset()
            metric.update(prob, y, mask=mask)
            return metric.compute()
        measure("pooled_ranking_"+name, update_compute)
    differences = dict(ap_max_abs=float((scores[0].average_precision-scores[1].average_precision).abs().max()),
                       auroc_max_abs=float((scores[0].auroc-scores[1].auroc).abs().max()))
    p = prob > .5
    kind = "area" if args.dim == 2 else "volume"
    options = dict(input_format="one-hot", spacing=(.5, .8, 2)[:args.dim], mask=mask)
    names = (f"predicted_{kind}", f"target_{kind}", f"{kind}_error")
    shared = F.evaluate(p, y, metrics=names, **options)
    for name in names:
        torch.testing.assert_close(shared[name], getattr(F, name)(p, y, **options))
    measure(kind+"s_shared", lambda: F.evaluate(p, y, metrics=names, **options))
    measure(kind+"s_separate", lambda: [getattr(F, name)(p, y, **options) for name in names])
    digest = hashlib.sha256()
    for path in sorted([*Path("src").rglob("*.py"), *Path("csrc").rglob("*.cpp"), *Path("csrc").rglob("*.h")]):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    report = dict(exacta=__version__, torch=torch.__version__, python=platform.python_version(),
                  hardware=torch.cuda.get_device_name() if args.device == "cuda" else cpu_model(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256=digest.hexdigest(), dtype="float32", target_dtype="bool", differences=differences,
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
