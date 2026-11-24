"""Latency of the 0.0.1a6 segmentation additions: instances, thinning, Betti numbers, pooled
counts, Skeleton Recall and soft-label Dice.

Run serially; warmups and constructors are outside measured regions. Each comparison first
checks that the compared paths agree. Raw reports are local and excluded from distributions.
Synthetic timing does not measure segmentation quality.
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

import numpy as np
import torch

import exacta
from exacta import instances
from exacta.functional import betti_numbers, confusion_counts, dice, skeletonize
from exacta.losses import DiceCELoss, DiceLoss, SkeletonRecallLoss, skeleton_recall_target
from exacta.metrics import Dice
from bench_advanced import cpu_model


def nuclei(size: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Touching Voronoi nuclei of radius < 9 and a shifted, eroded prediction."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    seeds = np.zeros((size, size), np.int64)
    seeds.flat[rng.choice(size * size, count, replace=False)] = np.arange(1, count + 1)
    distance, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    target = np.where(distance < 9, seeds[iy, ix], 0)
    pred = np.roll(np.where(ndimage.binary_erosion(target > 0), target, 0), 2, axis=1)
    return np.where(pred > 0, pred + 5 * count, 0), target


def vessels(shape: tuple[int, ...], seed: int, classes: int = 3) -> np.ndarray:
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.int64)
    for c in range(1, classes):
        labels[ndimage.binary_dilation(rng.random(shape) < 0.002, iterations=3)] = c
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.repeat, args.warmup) < 1:
        parser.error('positive threads and repeats required')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    records = []

    def sync() -> None:
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def measure(name: str, fn, **extra) -> None:
        for _ in range(args.warmup):
            fn()
        sync()
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() if args.device == 'cuda' else 0
        elapsed = []
        for _ in range(args.repeat):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            elapsed.append((time.perf_counter() - start) * 1000)
        result = dict(name=name, median_ms=statistics.median(elapsed), samples_ms=elapsed, **extra,
                      peak_extra_bytes=torch.cuda.max_memory_allocated() - allocated if args.device == 'cuda' else None)
        records.append(result)
        print(name, round(result['median_ms'], 3), flush=True)

    # Instance maps: 1024 x 1024, 2000 touching nuclei.
    pred, target = nuclei(1024, 2000, args.seed)
    p, t = torch.from_numpy(pred)[None].to(device), torch.from_numpy(target)[None].to(device)
    reference = instances.evaluate(torch.from_numpy(pred)[None], torch.from_numpy(target)[None])
    got = instances.evaluate(p, t)
    for name in ('pq', 'aji', 'aji_plus', 'seg'):
        torch.testing.assert_close(getattr(got, name).cpu(), getattr(reference, name))
    measure('instances.evaluate 1024^2, 2000 nuclei', lambda: instances.evaluate(p, t))
    measure('instances.dsb2018 1024^2, 2000 nuclei', lambda: instances.dsb2018(p, t))

    # Thinning (CPU kernels; CUDA inputs round-trip through the host).
    from skimage.morphology import skeletonize as sk_skeletonize

    plane = vessels((1024, 1024), args.seed + 1) > 0
    mask = torch.from_numpy(plane).to(device)
    assert (skeletonize(mask, method='zhang').cpu().numpy() == sk_skeletonize(plane, method='zhang')).all()
    measure('skeletonize zhang 1024^2', lambda: skeletonize(mask, method='zhang'))
    if args.device == 'cpu':
        measure('scikit-image skeletonize zhang 1024^2', lambda: sk_skeletonize(plane, method='zhang'))

    # Betti numbers of a 3-D vessel volume, both connectivities.
    volume = torch.from_numpy(vessels((128, 128, 128), args.seed + 2)).to(device)
    for connectivity in (1, 3):
        measure(f'betti_numbers 128^3, 2 classes, connectivity {connectivity}',
                lambda c=connectivity: betti_numbers(volume[None], connectivity=c, num_classes=3, include_background=False))

    # Pooled versus case-averaged overlap: the same count pass.
    labels = torch.from_numpy(np.stack([vessels((96, 128, 128), args.seed + k, 4) for k in (3, 4)])).to(device)
    shifted = torch.roll(labels, 1, dims=1)
    for aggregate in ('cases', 'pooled'):
        metric = Dice(num_classes=4, aggregate=aggregate)
        measure(f'Dice update aggregate={aggregate} 2x96x128x128, 4 classes',
                lambda m=metric: m.update(shifted, labels))
    measure('confusion_counts 2x96x128x128, 4 classes', lambda: confusion_counts(shifted, labels, num_classes=4))
    measure('dice 2x96x128x128, 4 classes', lambda: dice(shifted, labels, num_classes=4))

    # Skeleton Recall: target construction (CPU) and loss forward/backward.
    seg = torch.from_numpy(np.stack([vessels((64, 64, 64), args.seed + k) for k in (5, 6)]))
    measure('skeleton_recall_target 2x64^3 (CPU thinning)', lambda: skeleton_recall_target(seg))
    skeleton = skeleton_recall_target(seg).to(device)
    seg = seg.to(device)
    logits = torch.randn((2, 3, 64, 64, 64), generator=torch.Generator().manual_seed(args.seed)).to(device).requires_grad_()
    for name, loss, kwargs in (('SkeletonRecallLoss', SkeletonRecallLoss(), dict(target_skeleton=skeleton)),
                               ('DiceCELoss', DiceCELoss(include_background=False), {})):
        def step(loss=loss, kwargs=kwargs) -> None:
            logits.grad = None
            loss(logits, seg, **kwargs).backward()
        measure(f'{name} forward_backward 2x3x64^3', step)

    # Soft-label Dice: semimetric (PyTorch) versus the product form (PyTorch and, on CUDA, fused).
    soft = torch.rand((2, 4, 96, 96, 96), generator=torch.Generator().manual_seed(args.seed + 7)).softmax(1).to(device)
    logits4 = torch.randn((2, 4, 96, 96, 96), generator=torch.Generator().manual_seed(args.seed + 8)).to(device).requires_grad_()
    variants = [('DiceLoss soft_label=True', DiceLoss(target_format='one-hot', soft_label=True)),
                ('DiceLoss product torch', DiceLoss(target_format='one-hot', backend='torch'))]
    if args.device == 'cuda':
        variants.append(('DiceLoss product fused', DiceLoss(target_format='one-hot')))
    for name, loss in variants:
        def step(loss=loss) -> None:
            logits4.grad = None
            loss(logits4, soft).backward()
        measure(f'{name} forward_backward 2x4x96^3', step)

    files = [*Path('src/exacta').rglob('*.py'), *Path('csrc').rglob('*.cpp'),
             *Path('csrc').rglob('*.h'), Path('CMakeLists.txt'), Path(__file__)]
    source = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    report = dict(arguments={**vars(args), 'output': str(args.output)}, version=exacta.__version__,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256=source, python=platform.python_version(), torch=torch.__version__,
                  cpu=cpu_model(), gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                  results=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
