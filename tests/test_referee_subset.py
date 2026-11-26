"""tests/referee_subset.txt names only tests that exist, so `pytest -m referee` cannot shrink silently.

The listing is checked statically here (no collection), and conftest.py checks it again when the
listed files are collected.
"""

from __future__ import annotations

import ast
from pathlib import Path

from conftest import REFEREE_SUBSET, referee_entries

ROOT = REFEREE_SUBSET.parents[1]


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_every_listed_file_and_test_exists():
    problems = []
    for entry in referee_entries():
        file, _, rest = entry.partition("::")
        path = ROOT / file
        if not path.is_file():
            problems.append(f"{entry}: no such file")
        elif rest and rest.split("[", 1)[0].split("::")[-1] not in _test_names(path):
            problems.append(f"{entry}: no such test")
    assert problems == []


def test_entries_are_unique_and_include_the_driver():
    entries = referee_entries()
    assert len(entries) == len(set(entries)) > 20
    assert "tests/test_example_driver.py" in entries
