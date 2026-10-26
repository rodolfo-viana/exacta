# Performance and backends

Exacta uses compiled C++ CPU kernels for expensive metric passes and Triton for supported
CUDA paths. Losses use native PyTorch/autograd and, for supported CUDA objectives,
fused Triton forward/backward.
Backend choice should be supported by measurements for the shapes and dtypes used in training.

Fused losses recompute activations in backward and retain inputs plus small statistics, avoiding
full-volume saved probability and one-hot intermediates. Mesh distances currently run on CPU,
including when input arrays reside on CUDA. Cross-device conversion costs matter in a pipeline.

## Reproducing measurements

The benchmark scripts used for these published measurements remain available in the
`v0.0.1a3` source snapshot. Inputs are seeded, compilation is warmed up, and GPU steps
synchronize around timing. Clone that snapshot before running the commands below:

```bash
git clone --branch v0.0.1a3 https://github.com/rodolfo-viana/exacta.git exacta-benchmarks
cd exacta-benchmarks
python -m pip install -e ".[test]"
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

## Probability, ranking and loss measurements

The source includes `bench_probability.py` and `bench_completion.py`. Measurements
on 2026-10-26 used Python 3.14.4, PyTorch 2.14.1, Ryzen 9 5900XT (two threads) and RTX 5070,
float32 inputs of shape `2×3×64³`, three warmups and ten synchronized repetitions.
The tables report medians on synthetic inputs; compilation time is excluded. Probability
calculations use float64 arithmetic and an 80% observation mask. Loss timings include backward.

| Operation | CPU fused C++ | CPU eager PyTorch | CPU compiled PyTorch |
|---|---:|---:|---:|
| Multilabel Brier, probabilities | 1.892 ms | 18.446 ms | 7.219 ms |
| Multilabel NLL, probabilities | 6.810 ms | 38.126 ms | 11.034 ms |

For squared Dice on this GPU workload, compiling native PyTorch was fastest. Choose the
backend and compilation mode using representative shapes, not the implementation language:

| GPU forward/backward | Eager PyTorch | Compiled PyTorch | Eager fused Triton |
|---|---:|---:|---:|
| Squared Dice, sigmoid | 0.657 ms | 0.267 ms | 0.575 ms |
| Lovasz-hinge, independent labels | 1.666 ms | 0.865 ms | Uses PyTorch sorting |

Exact per-case voxel AP took 131.745 ms on CPU and 4.293 ms on GPU; 256-bin approximation
took 19.352 ms and 3.620 ms. This compares different ranking definitions, not interchangeable
implementations. Pooling updates can change both values and timing. Binned streaming retains
only histogram state; its current-batch working memory still scales with voxel count.

Three physical volume quantities evaluated together took 1.589 ms CPU / 0.423 ms GPU,
versus 4.644 ms / 0.853 ms for three separate calls. Their shared counts avoid repeated scans.
The implementations are included in the `v0.0.1a5` snapshot. Reproduce these measurements
from that source checkout:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python benchmarks/bench_probability.py --device cpu --size 64 --compiled --output local/probability-cpu.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python benchmarks/bench_completion.py --device cuda --size 64 --compiled --output local/completion-gpu.json
```

Both scripts support `--device cpu/cuda`, `--dim 2/3` and configurable sizes. Output records
raw samples, allocation measurements, configuration, environment and source hashes. Different
mask density, class counts, hardware and compiler versions can change the best path.
