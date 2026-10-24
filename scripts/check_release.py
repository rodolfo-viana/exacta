"""Reject release tags that do not identify a release version merged into main."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess

from packaging.version import Version


def validate_release(ref: str, tag: str) -> str:
    """Validate a checked-out GitHub Release and return its canonical package version."""
    if not tag.startswith("v") or ref != f"refs/tags/{tag}":
        raise ValueError("A release must use a version tag such as v0.0.1a4")
    expected = tag[1:]
    version = Version(expected)
    if str(version) != expected or version.is_devrelease or version.local is not None:
        raise ValueError("Release tags require a canonical, non-development, public version")

    assignments = ast.parse(Path("src/exacta/__init__.py").read_text()).body
    versions = [ast.literal_eval(node.value) for node in assignments
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__"
                        for target in node.targets)]
    if versions != [expected]:
        raise ValueError("Release tag does not match the source __version__")

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.PIPE).strip()

    if git("rev-parse", "HEAD") != git("rev-parse", "--verify", f"{ref}^{{commit}}"):
        raise ValueError("Checked-out commit does not match the release tag")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/main"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise ValueError("Release commit must already be merged into origin/main; fetch full history")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    try:
        version = validate_release(args.ref, args.tag)
    except (ValueError, OSError, SyntaxError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release rejected: {error}\n")
    print(f"version={version}")


if __name__ == "__main__":
    main()
