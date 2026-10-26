"""Compare Wasserstein Dice forward/backward latency and CUDA allocated-memory peaks.

Metrics are synthetic implementation measurements, not model-quality evidence.
Run serially, without concurrent tests. Compilation and constructors are not timed.
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

from exacta import __version__
from exacta.losses import GeneralizedWassersteinDiceLoss
from bench_advanced import cpu_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=3)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--seed', type=int, default=879)
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--monai', action='store_true', help='Include the optional external reference.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.size, args.batch, args.threads, args.warmup, args.repeat) < 1 or args.classes < 2:
        parser.error('positive sizes/threads/repeats and classes>=2 required')
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.batch, args.classes, *([args.size] * args.dim))
    logits = torch.randn(shape, generator=generator).to(args.device).requires_grad_()
    target = torch.randint(args.classes, (args.batch, *shape[2:]), generator=generator).to(args.device)
    # Tree-like example: maximum background distance, lower foreground confusion cost.
    matrix = torch.full((args.classes, args.classes), .25, device=args.device)
    matrix[0, :], matrix[:, 0] = 1, 1
    matrix.fill_diagonal_(0)
    results = []
    reference_version = None

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def measure(name, loss):
        def step():
            logits.grad = None
            loss(logits, target).backward()
        for _ in range(args.warmup):
            step()
        logits.grad = None
        sync()
        initial = torch.cuda.memory_allocated() if args.device == 'cuda' else 0
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.repeat):
            sync()
            start = time.perf_counter()
            step()
            sync()
            samples.append((time.perf_counter() - start) * 1000)
        peak = torch.cuda.max_memory_allocated() - initial if args.device == 'cuda' else None
        row = dict(name=name, median_ms=statistics.median(samples), samples_ms=samples,
                   peak_additional_bytes=peak)
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'samples_ms'}), flush=True)
        logits.grad = None

    for mode in ('default', 'GDL'):
        exacta = GeneralizedWassersteinDiceLoss(matrix, weighting_mode=mode).to(args.device, dtype=logits.dtype)
        choices = [('exacta', exacta)]
        if args.monai:
            import monai
            reference_version = monai.__version__
            choices.append(('monai', monai.losses.GeneralizedWassersteinDiceLoss(matrix, weighting_mode=mode)))
        expected = exacta(logits, target)
        expected_grad = torch.autograd.grad(expected, logits)[0]
        for name, loss in choices:
            actual = loss(logits, target)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], expected_grad, rtol=1e-4, atol=1e-8)
            measure(f'{name}_{mode}', loss)
            if args.compiled:
                torch._dynamo.reset()
                compiled = torch.compile(loss, fullgraph=True)
                actual = compiled(logits, target)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], expected_grad, rtol=1e-4, atol=1e-8)
                measure(f'{name}_{mode}_compiled', compiled)
    digest = hashlib.sha256()
    for path in sorted(Path('src/exacta').rglob('*.py')):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    report = dict(exacta=__version__, monai=reference_version, torch=torch.__version__,
                  python=platform.python_version(), dtype=str(logits.dtype),
                  hardware=torch.cuda.get_device_name() if args.device == 'cuda' else cpu_model(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256=digest.hexdigest(), dist_matrix=matrix.cpu().tolist(),
                  config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
