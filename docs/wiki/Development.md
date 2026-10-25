# Development

Use a source checkout to evaluate unreleased changes or contribute to Exacta. Start feature
work on a short-lived branch off **`main`**. These pages document public release `0.0.1a3`;
`main` may contain newer changes.

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
establish GPU correctness.

Optional reference packages can be installed with:

```bash
python -m pip install monai medpy scikit-image vtk SimpleITK \
  "git+https://github.com/google-deepmind/surface-distance.git" \
  "git+https://github.com/gasperpodobnik/MeshMetrics.git"
```

The Panoptica and BraTS reference fixtures require separate environments. Run
`bash tests/fixtures/regenerate.sh` from a source checkout with `uv` installed to regenerate
them. The script reports any differences from the committed reference values.

## Contribution checks

Changes should state their mathematical definition, reduction and empty-input behavior.
Verify against an independent implementation and, for losses, finite-difference gradients.
Exercise CPU/GPU agreement, mixed precision, compilation and noncontiguous inputs. Benchmark
latency and memory whenever a volume pass changes. Higher-order derivatives require the
PyTorch loss backend.
