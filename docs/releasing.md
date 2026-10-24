# Developing and releasing Exacta

Feature branches feed `main`; a release selects a tested commit from that history.
`main` can contain work for the next release. Publishing a **GitHub Release** is the explicit
step that starts PyPI publication. Branch pushes, merges, tag pushes and saved release drafts
only prepare the work; they do not publish a package.

This document is for maintainers. `README.md` is also the package description on PyPI;
keep release procedures here and installation and usage instructions in the README.

## Feature cycle

Start each change from current `main`, commit it on a short-lived branch, and push over SSH.
Use Rodolfo Viana's Git identity for commits, merge commits and annotated tags.

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/descriptive-name
# Implement, test and document the change, then commit selected files.
git push -u origin feature/descriptive-name
```

Use `fix/...` for fixes and `release/...` for release preparation when those names help.
Open a pull request into `main`, review the change and its CI results, then merge it. Record
user-visible changes under **Unreleased** in `CHANGELOG.md`. Several merged features can be
collected into one release; a feature merge does not need a version bump or a publication.
After a branch is fully merged, it can be deleted. There is no permanent `dev` branch.

## Validation

Development CI runs on all branch pushes and pull requests into `main`. It tests an installed
sdist with Python 3.10, Torch 2.4.1 and NumPy 1.x, and checks types. Installing the sdist compiles
the CPU extension needed by the tests. Development CI does not run the portable-wheel matrix
or publish to PyPI.

Portable Linux wheels are built, repaired and tested only for published GitHub Releases.
That matrix covers Python 3.10, 3.11 and the Python 3.12 stable ABI, with additional tests on
Python 3.13 and 3.14.

The [publishing workflow](../.github/workflows/publish.yml) handles `release: published`,
including alpha/beta/RC releases published from drafts. Its first job checks that the `v...`
tag matches the source version and checked-out commit, and that the commit is already in
`origin/main` history. Development and local versions are rejected. Only after this guard
passes do the source and wheel build/test jobs run; the upload waits for both.
See [GitHub's release event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release).

CI is CPU-only. Run CUDA and NCCL tests on suitable hardware before claiming those paths are
validated. Version 0.0.1a3 has CPU and single-GPU CUDA coverage with two Gloo ranks; multi-GPU
NCCL remains experimental and unverified. Minimum-Torch test jobs clear the `CI` variable to
avoid a Torch 2.4 internal decomposition assertion; full-graph compilation tests remain enabled.

## Release procedure

Choose the next version when the desired features have reached `main`. For example, the next
alpha after 0.0.1a3 is 0.0.1a4; the corresponding Git tag is `v0.0.1a4`.

1. Create `release/0.0.1a4` from `main`. Set `__version__` in `src/exacta/__init__.py` to
   `0.0.1a4` without `.dev0`. Move the collected changelog entries into a dated release section.
   Update the README's version/install command and the Wiki sources in `docs/wiki`.
2. Run relevant checks, build into a fresh directory with
   `python -m build --sdist --outdir <output-directory>`, and validate with
   `python -m twine check --strict <output-directory>/*.tar.gz`. Inspect the README that will
   appear on PyPI. Commit and push the preparation branch, review it, and merge into `main`.
3. Wait for CI on the selected `main` commit, then create an annotated tag on that exact commit
   and push the tag over SSH. Keep the tag fixed once it is published.
4. Open [GitHub Releases](https://github.com/rodolfo-viana/exacta/releases/new), select that
   existing tag, and draft the release notes. Use **Generate release notes** to collect merged
   PRs since the previous tag; `.github/release.yml` groups them by `enhancement`, `bug` and
   `documentation` labels, with a catch-all category. Reconcile the result with the changelog.
5. Mark alpha/beta/RC versions as **pre-release** and review the notes. Click **Publish release**
   when ready. This is the action that starts the PyPI workflow.
6. Follow **Actions → Publish to PyPI**, review the build/test results and approve the `pypi`
   environment if prompted. Verify the version, files and description on
   [PyPI](https://pypi.org/project/exacta/), then publish matching Wiki pages following
   [wiki maintenance](wiki-maintenance.md).
7. Start the next development version, such as `0.0.1a5.dev0`, in a small branch off `main`,
   reopen the **Unreleased** changelog section, and merge it through the same feature cycle.
   Published tags keep the release snapshot while `main` moves forward.

For the tag step, use the reviewed commit SHA recorded from the successful CI run:

```bash
# Set release_commit to that full commit SHA before running these commands.
git tag -a v0.0.1a4 "$release_commit" -m 'Exacta 0.0.1a4'
git push origin refs/tags/v0.0.1a4
```

The GitHub Release UI and draft behavior are described in
[GitHub's release guide](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
Generated PR notes complement the curated changelog; see
[GitHub's release-note configuration](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes).

## Trusted Publisher configuration

The existing PyPI Trusted Publisher keeps the same values; changing the release trigger does
not require a new publisher or a PyPI password/token.

| Field | Value |
|---|---|
| PyPI project | `exacta` |
| GitHub owner | `rodolfo-viana` |
| Repository | `exacta` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

If GitHub environment deployment restrictions are enabled, allow the release **tags** (`v*`).
A rule allowing only the `main` branch blocks tag-based release jobs. The workflow independently
checks main ancestry. Owner review can remain enabled with self-review allowed for owner releases.
Publishing uses short-lived OIDC credentials; see
[PyPI's Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Retries and existing releases

For a transient failure before upload, rerun the failed GitHub Actions jobs for the same
published release. Check PyPI before retrying a partial upload: existing filenames cannot be
replaced. If source changes are needed, prepare a new version rather than moving a published tag.
Do not publish another release merely to test the workflow.

PyPI stores a release's description at its first upload. Editing the README or uploading another
file for the same version does not update it; publish a new version to change the description.
See [PyPI's metadata behavior](https://docs.pypi.org/api/json/#get-a-project). Version 0.0.1a3
includes Linux wheels; 0.0.1a2 remains source-only.
