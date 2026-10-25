# Advanced segmentation performance

These measurements cover the unreleased advanced segmentation APIs. They guide backend
choices on the tested hardware; they do not establish model-quality gains or universal
speedups. Loss timings include forward and backward; metric timings include validation
and device transfers. All numbers below are medians in milliseconds.

## Environment and method

Measured on 2026-10-24 with AMD Ryzen 9 5900XT 16-Core Processor, two CPU threads, and
NVIDIA GeForce RTX 5070. Python 3.14.4, PyTorch 2.14.1+cu130, CUDA
13.0, NumPy 2.5.4; float32 inputs, seed 723, 3 warmups and 10 measured
repeats. The target consists of smooth synthetic regions; logits are seeded normal noise.
Spacing is `(1,1,2)`, soft-skeleton iterations 10. Compilation and warmup are excluded.
CUDA timings synchronize before and after each step. Additional GPU memory is peak
allocation above inputs, existing caches and the input gradient, not total training memory.
Cached rows exclude map/skeleton construction. CPU runs use the OpenMP-enabled extension.

The raw JSON files retain configurations, timing ranges, memory, Git base revision, dirty
status and a SHA-256 of the Python/C++ implementation. The runs used the working tree
about to be committed; its implementation hash is:

`f63ec46ac0747cfe812d1045720358ef2c68440d870987152ade6cddf871c428`

## CPU selection

The `1×3×64³` workload favours eager native PyTorch for the sorting/overlap losses and
fused C++ morphology for soft skeletons. Compilation is not automatically faster on CPU.
`SoftClDiceLoss(backend="auto")` uses C++ in eager CPU execution; `backend="torch"`
provides the pooling reference path. Both keep PyTorch autograd.

| Operation | Native PyTorch | Selected C++ path | Compiled PyTorch |
|---|---:|---:|---:|
| Focal Tversky | 7.780 | — | 12.234 |
| Top-k CE | 5.979 | — | 8.166 |
| Lovasz-Softmax | 66.788 | — | 68.136 |
| Soft-clDice | 774.681 | 356.791 | 3570.959 |

Caching the target soft skeleton reduces the C++ loss to 222.146 ms.
For one `64³` hard mask, C++ Lee thinning takes 1.108 ms versus
5.162 ms for scikit-image 0.26.0's Lee implementation, about
4.7× faster on this mask.
Lee thinning remains on CPU even for CUDA metric inputs. C++ and Torch soft erosion
can select different valid subgradients at ties; values and gradients away from ties are checked.

## GPU selection

For `1×3×64³`, compiled PyTorch lowers sorting/pooling overhead substantially. Use
`torch.compile(LovaszSoftmaxLoss())` or `torch.compile(SoftClDiceLoss())` after measuring
your own shapes and accounting for first-call compilation cost. Soft-clDice compilation
also reduces peak additional allocation from about 697 MiB to 189 MiB here.

| Loss | Eager PyTorch | Compiled PyTorch |
|---|---:|---:|
| Top-k CE | 0.530 | 0.428 |
| Lovasz-Softmax | 1.559 | 0.523 |
| Soft-clDice | 9.937 | 2.295 |

Focal Tversky shows the size-dependent tradeoff clearly. The auto backend keeps the
existing fused Triton policy on CUDA. It is fastest for the larger many-class workload
and saves memory; compiled `backend="torch"` is faster for the small workload. There
is no runtime calibration benchmark or implicit compilation inside the loss constructor.

| Shape | Eager Torch | Compiled Torch | Fused Triton | Extra MiB: eager / compiled / fused |
|---|---:|---:|---:|---:|
| 1×3×64³ | 0.658 | 0.263 | 0.692 | 12.002 / 3.001 / 0.003 |
| 1×3×128³ | 1.869 | 0.505 | 0.746 | 96.002 / 24.001 / 0.003 |
| 2×14×128³ | 30.073 | 4.304 | 1.616 | 896.002 / 48.002 / 0.003 |

Wrapping the fused loss in `torch.compile` was slower and used more temporary memory
for the many-class workload than calling the fused loss directly; its raw measurements
are retained. GPU cached soft-clDice was also slightly slower than the compiled uncached
version in this run, so caching is not presented as a universal speedup.

## Cached geometry and evaluation

Target distance maps can be reused while geometry and spacing are unchanged. At
`1×3×128³` on CUDA, this avoids repeated exact target EDT work:

| Loss | Recompute target geometry | Reuse target map |
|---|---:|---:|
| Boundary | 5.763 | 1.179 |
| Hausdorff-DT | 11.554 | 7.074 |

The following `1×3×64³` evaluation timings include their normal validation and
aggregation. Hard clDice includes CPU skeletonization and transfers on the CUDA path.
AP/FROC uses a small number of smooth-region components; densely fragmented predictions
can cost much more and are not characterized by this benchmark.

| Evaluation | CPU | CUDA input |
|---|---:|---:|
| Brier | 2.274 | 0.404 |
| NLL | 2.181 | 0.379 |
| ECE | 3.951 | 0.641 |
| Reliability summary | 3.961 | 0.651 |
| Hard clDice | 5.903 | 4.581 |
| Lesion AP/FROC | 2.427 | 2.548 |

## Reproduction and validation

Run each benchmark separately, without concurrent tests or other training workloads.
The JSON artifacts are versioned alongside the benchmark script:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
python benchmarks/bench_advanced.py --device cpu --size 64 --compiled --output cpu64.json
python benchmarks/bench_advanced.py --device cuda --size 64 --compiled --output gpu64.json
python benchmarks/bench_advanced.py --device cuda --size 128 --compiled \
  --only focal_tversky topk_ce lovasz boundary hausdorff_dt --output gpu128.json
python benchmarks/bench_advanced.py --device cuda --size 128 --batch 2 --classes 14 \
  --compiled --only focal_tversky --output gpu128-c14.json
```

Raw results: [CPU64](../benchmarks/results/advanced-cpu64-2026-10-24.json),
[GPU64](../benchmarks/results/advanced-gpu64-2026-10-24.json),
[GPU128](../benchmarks/results/advanced-gpu128-2026-10-24.json), and
[GPU128, 14 classes](../benchmarks/results/advanced-gpu128-c14-2026-10-24.json).

Validation completed before recording these runs: the CPU/CUDA suite passed **2,947 tests**,
with 25 skipped. The installed source distribution passed **1,294 tests**, with 636 skipped,
on Python 3.12.15, Torch 2.4.1 CPU and NumPy 1.26.4. Type checks passed in both environments.
Coverage includes independent formulas, SciPy distances, MONAI Hausdorff-DT, frozen and
live scikit-image Lee references, first/second derivatives, masking, mixed precision,
compilation and distributed Gloo accumulation. One physical GPU was available;
multi-GPU NCCL remains unverified. GPU histogram reductions can differ in last-bit
rounding due to atomic summation order. These are synthetic implementation checks,
not MRI model experiments or clinical validation.
