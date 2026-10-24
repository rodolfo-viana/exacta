# Shared metrics and composed losses: 2026-10-24

These measurements test whether sharing volume passes reduces latency and saved intermediates.
They use synthetic inputs, not patient data or a trained model. They do not establish better
segmentation quality, whole-training speedups, or multi-GPU throughput.

## Environment and method

The measured implementation is Exacta `0.0.1a3.dev0`, commit
[`12af1d4`](https://github.com/rodolfo-viana/exacta/commit/12af1d4762ef03b5724b2603ca26f80051bf9f75).
Later fixes for disabled single-channel overlap and NumPy typing do not affect these configurations.
The environment was:

| Component | Value |
|---|---|
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB reported device memory |
| OS kernel | Linux 7.0.0-38-generic, x86-64 |
| Python | 3.14.4 |
| PyTorch / CUDA | 2.14.1+cu130 / 13.0 |
| NVIDIA driver | 580.178.04 |
| Triton | 3.8.0 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| Torch / native OpenMP threads | 16 / 16 |
| Seed | 0 |
| Timing | Median of 20 repetitions after 5 warm-up repetitions |

Loss timing covers forward and backward, including gradient allocation; CUDA synchronizes
before and after each timed step. Compilation and warm-up are excluded. Additional GPU peak
memory is measured beyond inputs and one logits-gradient buffer, so a reported 0.0 MiB does
**not** mean no allocation: scratch buffers can fit below that baseline. Measurements ran
sequentially after local correctness tests finished. Each configuration has one timed run;
no confidence intervals or cross-machine claims are made.

## Dice plus focal

Inputs were softmax logits `(2, 4, 128, 128, 128)` and index targets, with no exclusions.
Both coefficients were 1, focal gamma was 2, alpha was None, and overlap included background.
`separate` calls the existing auto-dispatched Dice and focal losses independently; on this
GPU both are fused Triton losses. This comparison isolates the benefit of combining their passes.

| dtype | implementation | forward + backward (ms) | additional GPU peak (MiB) |
|---|---|---:|---:|
| float32 | eager Torch composition | 13.425 | 688.0 |
| float32 | compiled Torch composition | 3.390 | 256.0 |
| float32 | combined Triton | 0.861 | 0.0 |
| float32 | separate Triton losses | 1.622 | 64.0 |
| bfloat16 | eager Torch composition | 13.951 | 784.0 |
| bfloat16 | compiled Torch composition | 2.318 | 256.0 |
| bfloat16 | combined Triton | 0.815 | 0.0 |
| bfloat16 | separate Triton losses | 1.214 | 32.0 |

Combining the GPU passes reduced float32 latency by 1.88× versus the two separate fused losses
and 3.94× versus compiled Torch on this input. This supports keeping the combined kernel.

```bash
python benchmarks/bench_losses.py --device cuda --classes 4 --sizes 128 \
  --losses dice_focal --backends torch compiled triton separate \
  --dtypes float32 bfloat16 --repeat 20 --warmup 5 --seed 0 --threads 16
```

## Weighted CE combinations with exclusions

These runs used float32 softmax logits `(2, 4, 128, 128, 128)`, index targets and CE weights
`[0.2, 0.5, 1.0, 1.0]`. Independent label exclusions of 25% and voxel-mask exclusions of 20%
left 60.01% of voxels included. Coefficients and other options retained their defaults.

| loss | implementation | forward + backward (ms) | additional GPU peak (MiB) |
|---|---|---:|---:|
| generalized Dice + CE | eager Torch | 8.398 | 324.0 |
| generalized Dice + CE | compiled Torch | 3.809 | 148.0 |
| generalized Dice + CE | combined Triton | 1.280 | 0.0 |
| Dice + CE | eager Torch | 8.257 | 324.0 |
| Dice + CE | compiled Torch | 3.759 | 148.0 |
| Dice + CE | combined Triton | 1.133 | 0.0 |

```bash
python benchmarks/bench_losses.py --device cuda --classes 4 --sizes 128 \
  --losses generalized_dice_ce dice_ce --class-weight 0.2 0.5 1.0 1.0 \
  --backends torch compiled triton --dtypes float32 --repeat 20 --warmup 5 \
  --seed 0 --threads 16 --ignore-fraction 0.25 --mask-fraction 0.2
```

## CPU compositions

On CPU, inputs were float32 softmax logits `(2, 4, 64, 64, 64)` with index targets, no
class weights and no exclusions. Compiling native PyTorch improved both measured objectives;
these data support retaining that path instead of adding an unmeasured custom CPU loss kernel.

| loss | eager Torch (ms) | compiled Torch (ms) |
|---|---:|---:|
| Dice + focal | 28.392 | 14.075 |
| generalized Dice + CE | 17.812 | 10.768 |

```bash
python benchmarks/bench_losses.py --device cpu --classes 4 --sizes 64 \
  --losses dice_focal generalized_dice_ce --backends torch compiled \
  --dtypes float32 --repeat 20 --warmup 5 --seed 0 --threads 16
```

## Shared metric evaluation

Inputs were two deterministic `128³` label maps containing two cuboid foreground classes
plus background, shifted two voxels between prediction and target. Background was excluded,
spacing was `(1.0, 0.7, 2.0)`, and surfaces used the voxel convention. The script checks exact
agreement with standalone calls before timing. Seven count-based metrics form `counts`;
HD95, ASSD and NSD with tolerance 1 form `surfaces`; `all` requests all ten together.

| device | metrics | separate calls (ms) | shared evaluation (ms) | speedup |
|---|---|---:|---:|---:|
| CPU | counts | 5.066 | 0.791 | 6.41× |
| CPU | surfaces | 18.179 | 7.087 | 2.57× |
| CPU | all | 27.696 | 6.714 | 4.12× |
| CUDA | counts | 2.313 | 0.734 | 3.15× |
| CUDA | surfaces | 12.849 | 5.237 | 2.45× |
| CUDA | all | 15.260 | 5.659 | 2.70× |

Small differences between the CPU `surfaces` and `all` shared measurements reflect separate
timing runs; they do not imply that adding count metrics makes surface extraction faster.
Other shapes, class counts and surface representations need their own measurements.

```bash
python benchmarks/bench_collection.py --device cpu --size 128 --threads 16 --repeat 20 --warmup 5
python benchmarks/bench_collection.py --device cuda --size 128 --threads 16 --repeat 20 --warmup 5
```

## Correctness and limits

Validation compares metric collections with standalone definitions, and composed losses with
independently composed native PyTorch objectives. Checks include float64 values and gradients,
finite differences, soft-target derivatives, mixed precision, strided inputs, ignored/nonfinite
voxels and full-graph compilation. DDP tests compare global losses and parameter gradients
with one concatenated batch, including uneven, empty and entirely ignored ranks.

CPU and CUDA distributed correctness was checked using two Gloo workers; CUDA workers shared
one GPU. The two-GPU NCCL test is present but could not run on this machine. These tests do
not measure NCCL scaling, network performance or end-to-end MRI training. There is no data
split or leakage analysis to report for these deterministic synthetic kernel benchmarks.
