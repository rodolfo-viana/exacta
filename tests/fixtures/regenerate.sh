#!/usr/bin/env bash
# Regenerate the lesion-wise reference fixtures (panoptica.json, brats2023.json) in throwaway
# environments and report whether they changed.
#
# The reference tools cannot share exacta's environment: Panoptica 2.1.7 needs NumPy < 2.3, and
# the BraTS 2023 evaluation code needs its pinned requirements (pandas 1.3, NumPy 1.24) because it
# uses APIs that pandas 2 and NumPy 2 removed. The masks they are run on (lesion_cases.npz) are
# kept as they are. Needs uv and git.
#
#   tests/fixtures/regenerate.sh
#   PANOPTICA_VERSION=2.1.8 tests/fixtures/regenerate.sh      # try another Panoptica release
#   BRATS_COMMIT=<sha> tests/fixtures/regenerate.sh           # or another BraTS commit
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
uv="${UV:-uv}"
panoptica_version="${PANOPTICA_VERSION:-2.1.7}"
brats_commit="${BRATS_COMMIT:-43c905242b2eecf421d4ab2da7af8ece9777d322}"

echo "== Panoptica $panoptica_version"
"$uv" venv -q -p 3.12 "$work/panoptica"
# Panoptica publishes only a source distribution. Its dependencies (those of 2.1.7) are installed
# as wheels only: resolving them freely builds some from source, which is slow and fragile.
"$uv" pip install -q -p "$work/panoptica/bin/python" --only-binary :all: \
  "numpy<2.3" "connected-components-3d<4" "scipy<2" "rich<14" "scikit-image<1" \
  "ruamel.yaml<1" "plotly<6" "pandas<3" "typer<1" tqdm
"$uv" pip install -q -p "$work/panoptica/bin/python" --no-deps "panoptica==$panoptica_version"
"$work/panoptica/bin/python" "$here/make_panoptica_fixture.py"

echo "== BraTS-2023-Metrics $brats_commit"
git clone -q https://github.com/rachitsaluja/BraTS-2023-Metrics "$work/brats-src"
git -C "$work/brats-src" checkout -q "$brats_commit"
"$uv" venv -q -p 3.10 "$work/brats"
"$uv" pip install -q -p "$work/brats/bin/python" --only-binary :all: -r "$work/brats-src/requirements.txt"
"$work/brats/bin/python" "$here/make_brats_fixture.py" "$work/brats-src"

cd "$here"
if git diff --quiet -- panoptica.json brats2023.json; then
  echo "== Fixtures unchanged."
else
  git diff --stat -- panoptica.json brats2023.json
  echo "== Fixtures changed: review the diff and run pytest tests/test_lesions.py."
fi
