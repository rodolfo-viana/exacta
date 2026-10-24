# Development

Feature work starts on short-lived branches off **`main`**. Push each branch, review its CI
and pull request, then merge it into `main`. Several merged features can be collected into
one release. These pages document public release `0.0.1a3`; `main` may contain newer changes.

Install a checkout and run the tests with:

```bash
git clone --branch main https://github.com/rodolfo-viana/exacta.git
cd exacta
git switch -c feature/descriptive-name
python -m pip install -e ".[test]"
python -m pytest -q
```

Rebuild after editing `csrc/`. Optional reference tests skip when their libraries are absent;
CUDA and multi-GPU tests require the corresponding hardware. A passing CPU-only run does not
establish GPU correctness. The README describes optional reference packages and isolated
fixture regeneration for Panoptica and BraTS.

Development CI runs on all branch pushes and pull requests targeting `main`. It tests
an installed source distribution against Python 3.10, Torch 2.4.1 and NumPy 1.x, and runs
type checks. A reusable wheel workflow builds manylinux_2_28 x86-64 wheels for Python 3.10,
3.11 and the 3.12 stable ABI, tests each installed wheel, then tests the stable-ABI wheel
on Python 3.13 and 3.14. The resulting artifacts can be downloaded from the workflow run.
CI has no GPU; CUDA and NCCL validation require separate hardware runs.

The publishing workflow reuses the wheel checks and uploads their artifacts with the sdist
only when a **GitHub Release is published**. Its version tag must identify a commit already
merged into `main` and match the source version. Branch pushes, merges, tag pushes and saved
release drafts do not publish to PyPI. Review the collected release notes before publishing.
After release, `main` continues toward the next version; the published tag preserves the snapshot.

## Contribution checks

Changes should state their mathematical definition, reduction and empty-input behavior.
Verify against an independent implementation and, for losses, finite-difference gradients.
Exercise CPU/GPU agreement, mixed precision, compilation and noncontiguous inputs. Benchmark
latency and memory whenever a volume pass changes. Higher-order derivatives require the
PyTorch loss backend.

The [segmentation roadmap](https://github.com/rodolfo-viana/exacta/blob/main/docs/segmentation-roadmap.md)
tracks remaining work. The [release procedure](https://github.com/rodolfo-viana/exacta/blob/main/docs/releasing.md)
is maintained separately from user-facing installation instructions.

## Maintaining these pages

Wiki sources live in [`docs/wiki`](https://github.com/rodolfo-viana/exacta/tree/main/docs/wiki)
on `main`. Update them alongside feature code and publish reviewed pages to the separate GitHub Wiki
repository. Keep unreleased features labeled until their release succeeds. See
[wiki maintenance](https://github.com/rodolfo-viana/exacta/blob/main/docs/wiki-maintenance.md).
