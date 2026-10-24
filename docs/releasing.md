# Releasing Exacta

This document is for maintainers. `README.md` is also the package description on PyPI;
keep release procedures here and installation and usage instructions in the README.

The manual [publishing workflow](../.github/workflows/publish.yml) runs only from `main`.
It builds a source distribution, checks the metadata and requested version, installs that
artifact with PyTorch 2.4.1 and NumPy 1.x, and runs the tests before publishing. Binary wheels
are not yet part of the release workflow.

## Branch policy

Ongoing development belongs on `dev`. `main` contains releases only; do not push ordinary
feature work there. Use Rodolfo Viana's Git identity for commits and SSH for pushes on both
branches. Development versions may use a `.devN` suffix and are not published by this workflow.

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
