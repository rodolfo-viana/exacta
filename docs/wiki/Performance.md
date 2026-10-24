# Performance and backends

Exacta uses compiled C++ CPU kernels for expensive metric passes and Triton for supported
CUDA paths. Losses use native PyTorch/autograd on CPU and fused Triton forward/backward on GPU.
Backend choice should be supported by measurements for the shapes and dtypes used in training.

Fused losses recompute activations in backward and retain inputs plus small statistics, avoiding
full-volume saved probability and one-hot intermediates. Mesh distances currently run on CPU,
including when input arrays reside on CUDA. Cross-device conversion costs matter in a pipeline.

## Reproducing measurements

Run benchmarks from a source checkout. Inputs are seeded, compilation is warmed up, and GPU
steps synchronize around timing. These commands measure forward plus backward:

```bash
python benchmarks/bench_losses.py --classes 4 --sizes 128 --dtypes float32 \
  --repeat 20 --warmup 5 --seed 0 --threads 16
python benchmarks/bench_losses.py --device cpu --classes 4 --sizes 64 --dtypes float32 \
  --backends torch compiled --repeat 20 --warmup 5 --seed 0 --threads 16
python benchmarks/bench_metrics.py --sizes 128 256 --only exacta
```

Local generalized-Dice measurements for version 0.0.1a2 use float32 softmax inputs, PyTorch 2.14.1,
CUDA 13.0, an RTX 5070 and a Ryzen 9 5900XT with 16 CPU threads. Medians of 20 steps after five
warm-up steps are:

| Device and shape | Eager PyTorch | Compiled PyTorch | Fused Triton |
|---|---:|---:|---:|
| GPU, `2×4×128³` | 4.184 ms | 3.521 ms | 0.938 ms |
| CPU, `2×4×64³` | 10.541 ms | 4.985 ms | Not applicable |

GPU additional peak allocations were 256 MiB, 128 MiB and under 0.1 MiB respectively, measured
beyond inputs and the logits' gradient. Scratch allocations can fit below that baseline.
These are synthetic local measurements, not guaranteed speedups on other hardware or complete
training pipelines. Record code version, shapes, classes, dtype, threads and environment.

## Masked losses

The loss benchmark supports `--ignore-fraction` and `--mask-fraction`. At code revision
[`d3ba67f`](https://github.com/rodolfo-viana/exacta/commit/d3ba67fbe8b24ec6b65bdc1f1fa3f813fcb6ad9c),
independently excluding 25% of labels and 20% through a boolean mask retained 60.01% of voxels
on the GPU input. On the same RTX 5070 with float32 `2×4×128³` logits, seed 0, five warm-up
steps and 20 measured forward/backward steps, median times were:

| Loss | Eager PyTorch | Compiled PyTorch | Fused Triton |
|---|---:|---:|---:|
| Dice | 5.741 ms | 2.479 ms | 1.030 ms |
| Dice + CE | 8.271 ms | 3.719 ms | 1.071 ms |
| Tversky | 8.757 ms | 2.558 ms | 1.068 ms |
| Focal | 9.869 ms | 2.605 ms | 0.728 ms |
| Generalized Dice | 5.760 ms | 2.620 ms | 1.158 ms |

On CPU, masked Dice + CE with `2×4×64³` logits took 23.432 ms eager and 11.087 ms compiled,
using 16 threads. The [complete measurement record](https://github.com/rodolfo-viana/exacta/blob/v0.0.1a3/benchmarks/results/masking-2026-10-24.md)
includes unmasked runs, allocation measurements, environment versions, commands and limitations.

## Compiled CPU losses

For repeated CPU training, compiling native PyTorch operations can reduce allocations and
dispatch overhead. Compilation has startup cost and should be excluded from steady-state
comparisons while still being considered for short runs:

```python
import torch
from exacta.losses import DiceCELoss

loss_fn = torch.compile(DiceCELoss(backend="torch"), fullgraph=True)
```

## Shared metrics and composed losses

At revision [`12af1d4`](https://github.com/rodolfo-viana/exacta/commit/12af1d4762ef03b5724b2603ca26f80051bf9f75),
float32 Dice–focal on the same RTX 5070, with `2×4×128³` logits, took 0.861 ms for combined
Triton forward/backward, 1.622 ms for separate fused Dice and focal calls, and 3.390 ms for
compiled native Torch. Additional GPU peak allocation was 0.0, 64.0 and 256.0 MiB respectively,
using the baseline definition above. CPU native Torch also benefited from compilation.

Shared evaluation of seven count metrics plus HD95, ASSD and NSD on two synthetic `128³`
cuboid cases took 6.714 ms versus 27.696 ms for separate CPU calls, and 5.659 ms versus
15.260 ms for separate CUDA calls. Spacing was anisotropic and background was excluded.
All figures are medians of 20 repetitions after five warm-ups, using 16 threads.

The [measurement record](https://github.com/rodolfo-viana/exacta/blob/v0.0.1a3/benchmarks/results/composition-collection-2026-10-24.md)
contains the full environment, code revision, CPU results, bfloat16 runs, masked class-weighted
CE combinations, exact commands and limitations. The scripts are `bench_losses.py` and
`bench_collection.py`. Distributed correctness tests do not establish multi-GPU throughput;
NCCL benchmarks require additional hardware.

Avoid running performance measurements alongside other workloads. Measure sparse and dense
targets, different class counts and representative MRI sizes before choosing an implementation.
