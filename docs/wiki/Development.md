# Development

Feature work belongs on **`dev`**. **`main` is reserved for releases**. The current development
version is `0.0.1a3.dev0`; the documented public release is `0.0.1a2`.

Install a checkout and run the tests with:

```bash
git clone --branch dev https://github.com/rodolfo-viana/exacta.git
cd exacta
python -m pip install -e ".[test]"
python -m pytest -q
```

Rebuild after editing `csrc/`. Optional reference tests skip when their libraries are absent;
CUDA and multi-GPU tests require the corresponding hardware. A passing CPU-only run does not
establish GPU correctness. The README describes optional reference packages and isolated
fixture regeneration for Panoptica and BraTS.

## Contribution checks

Changes should state their mathematical definition, reduction and empty-input behavior.
Verify against an independent implementation and, for losses, finite-difference gradients.
Exercise CPU/GPU agreement, mixed precision, compilation and noncontiguous inputs. Benchmark
latency and memory whenever a volume pass changes. Higher-order derivatives require the
PyTorch loss backend.

The [segmentation roadmap](https://github.com/rodolfo-viana/exacta/blob/dev/docs/segmentation-roadmap.md)
tracks remaining work. The [release procedure](https://github.com/rodolfo-viana/exacta/blob/dev/docs/releasing.md)
is maintained separately from user-facing installation instructions.

## Maintaining these pages

Wiki sources live in [`docs/wiki`](https://github.com/rodolfo-viana/exacta/tree/dev/docs/wiki)
on `dev`. Update them alongside code and publish reviewed pages to the separate GitHub Wiki
repository. Keep unreleased features labeled until their release succeeds. See
[wiki maintenance](https://github.com/rodolfo-viana/exacta/blob/dev/docs/wiki-maintenance.md).
