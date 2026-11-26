from pathlib import Path

import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Node ids (a file, "file::test" or a full id) of the tests that `pytest -m referee` selects.
REFEREE_SUBSET = Path(__file__).with_name("referee_subset.txt")


@pytest.fixture(params=DEVICES)
def device(request) -> torch.device:
    return torch.device(request.param)


def referee_entries() -> list[str]:
    """Node ids listed in referee_subset.txt, without comments and blank lines."""
    lines = (line.split("#", 1)[0].strip() for line in REFEREE_SUBSET.read_text().splitlines())
    return [line for line in lines if line]


def _selects(entry: str, nodeid: str) -> bool:
    base = nodeid.split("[", 1)[0]
    return nodeid == entry or base == entry or base.startswith(entry + "::")


@pytest.hookimpl(tryfirst=True)  # before -m and -k deselect, so that unlisted tests are not counted
def pytest_collection_modifyitems(config, items):
    """Mark the tests listed in referee_subset.txt with ``referee``.

    An entry whose file was collected but that selects no test raises, so that renaming a test
    cannot silently remove it from the subset. Entries of files that were not collected (a run of
    some other file or directory) are ignored.
    """
    entries = referee_entries()
    matched = set()
    for item in items:
        for entry in entries:
            if _selects(entry, item.nodeid):
                item.add_marker(pytest.mark.referee)
                matched.add(entry)
    collected_files = {item.nodeid.split("::", 1)[0] for item in items}
    stale = [e for e in entries if e not in matched and e.split("::", 1)[0] in collected_files]
    if stale:
        raise pytest.UsageError("tests/referee_subset.txt lists tests that do not exist: " + ", ".join(stale))
