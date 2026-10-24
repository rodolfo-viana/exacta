# Releasing Exacta

This document is for maintainers. `README.md` is also the package description on PyPI;
keep release procedures here and installation and usage instructions in the README.

The manual [publishing workflow](../.github/workflows/publish.yml) runs only from `main`.
It builds a source distribution, checks the metadata and requested version, installs that
artifact with PyTorch 2.4.1 and NumPy 1.x, and runs the tests before publishing. The reusable
Linux wheel workflow builds and repairs manylinux_2_28 x86-64 wheels for Python 3.10, 3.11
and the Python 3.12 stable ABI. Each wheel is tested after installation; the stable-ABI
wheel is also tested on Python 3.13 and 3.14. Publishing waits for all these jobs.
These wheels enter PyPI with the next release; 0.0.1a2 remains source-only.

## Branch policy

Ongoing development belongs on `dev`. `main` contains releases only; do not push ordinary
feature work there. Use Rodolfo Viana's Git identity for commits and SSH for pushes on both
branches. Development versions may use a `.devN` suffix and are not published by this workflow.

Development CI runs on `dev` pushes and pull requests targeting `dev` or `main`. It tests
an installed sdist with Python 3.10, Torch 2.4.1 and NumPy 1.x, checks types, and builds/tests
the same portable wheels. It never uploads to PyPI. Download wheel artifacts from the
workflow run for pre-release installation checks. CI is CPU-only; run the CUDA and NCCL
tests separately on suitable hardware before release. The minimum-Torch test jobs clear
the `CI` variable to avoid a Torch 2.4 internal decomposition assertion; full-graph
compilation tests remain enabled.

## Trusted Publisher configuration

The first release, `0.0.1a1`, registered `exacta` on PyPI. Subsequent releases use the
project's existing GitHub Trusted Publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project | `exacta` |
| GitHub owner | `rodolfo-viana` |
| Repository | `exacta` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Manage the publisher in the PyPI project's publishing settings. The workflow field is the
filename, without `.github/workflows/`. A pending publisher is only needed before a project's
first upload; do not create another one for a new release of this project.

In GitHub repository settings, the `pypi` environment should restrict deployments to `main`
and require the owner's review, with self-review allowed for owner-triggered releases.
Publishing uses short-lived OIDC credentials; no PyPI password or API-token repository secret
is needed. See [PyPI's Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Release procedure

Each release must use a new version. Prepare and review the source before running the workflow:

1. Prepare the release on `dev`. Update `__version__` in `src/exacta/__init__.py`, the workflow's
   default version, and the README's version and installation command. Update the Wiki sources
   in `docs/wiki`; remove development labels only for features included in this release.
2. Run the relevant tests, then build and validate the source distribution in a fresh output
   directory with `python -m build --sdist --outdir <output-directory>` and
   `python -m twine check --strict <output-directory>/*.tar.gz`. Inspect the README as it will
   appear on PyPI.
3. Commit and push the reviewed release preparation to `dev`. When the release is ready,
   merge it into `main` and push `main` using Rodolfo Viana's Git identity and SSH.
4. In GitHub **Actions → Publish to PyPI → Run workflow**, select `main` and enter the exact
   source version.
5. Review the build and test results, then approve the `pypi` deployment if prompted.
6. Verify the version, distribution files and rendered description on
   [the PyPI project page](https://pypi.org/project/exacta/).
   Publish the matching Wiki pages following [wiki maintenance](wiki-maintenance.md).
7. Resume work on `dev` with the next development version; keep `main` at the release.

PyPI stores a release's description at its first upload; editing GitHub's README or uploading
another file for the same version does not update it. Publish a new version to change the
project description. See [PyPI's metadata behavior](https://docs.pypi.org/api/json/#get-a-project).
