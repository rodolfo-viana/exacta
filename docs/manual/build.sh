#!/usr/bin/env bash
# Build docs/manual/userManual.pdf with latexmk (pdflatex). Needs a TeX Live with latexmk,
# listings, booktabs, longtable, hyperref and microtype. The PDF and the build files stay out of
# version control (see .gitignore).
#
#   docs/manual/build.sh          build the PDF
#   docs/manual/build.sh clean    remove the build products
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$here"

if [[ "${1:-}" == "clean" ]]; then
    latexmk -C userManual.tex
    rm -f version.tex
    exit 0
fi

# The version on the title page is the one of the source tree.
version="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$root/src/exacta/__init__.py")"
if [[ -z "$version" ]]; then
    echo "build.sh: cannot read __version__ from src/exacta/__init__.py" >&2
    exit 1
fi
printf '\\newcommand{\\exactaversion}{%s}\n' "$version" > version.tex

# -halt-on-error: a build with errors fails instead of leaving a partial PDF.
# A fixed date in the PDF keeps two builds of the same sources identical.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(date -d '2026-01-01' +%s)}"
export FORCE_SOURCE_DATE=1
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error userManual.tex
echo "built $here/userManual.pdf (version $version)"
