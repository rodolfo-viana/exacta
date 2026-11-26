"""The example driver (examples/driver.py) prints the output stored in examples/driver_expected.txt.

Lines without a marker come from computations that Exacta rounds or counts exactly, so they must
match as text (the same bits on every platform, thread count and device). Lines that end with
"(approx)" are losses, whose last bits depend on the PyTorch build; they match to a relative
tolerance. Regenerate the expected file after a deliberate change of the driver or of a value with
``python examples/driver.py 2> /dev/null > examples/driver_expected.txt`` and review the diff.
"""

from __future__ import annotations

import functools
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from exacta import _triton_edt

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DRIVER = EXAMPLES / "driver.py"
EXPECTED = EXAMPLES / "driver_expected.txt"
APPROX = " (approx)"
RTOL = 1e-9

GPU = torch.cuda.is_available() and _triton_edt.available()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")


@functools.lru_cache(maxsize=None)
def _run(threads: int | None, device: str = "cpu") -> str:
    """Standard output of ``python examples/driver.py`` with the given number of OpenMP threads."""
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(threads)
    # Search for modules as this interpreter does (an interpreter started with -S has no .pth files).
    command = [sys.executable, *(["-S"] if sys.flags.no_site else []), str(DRIVER), "--device", device]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"driver failed with status {done.returncode}:\n{done.stdout}\n{done.stderr}"
    return done.stdout


def _differences(actual: str, expected: str) -> list[str]:
    """Lines that differ: as text, or for approximate lines beyond the relative tolerance."""
    got, want = actual.splitlines(), expected.splitlines()
    if len(got) != len(want):
        return [f"{len(got)} lines instead of {len(want)}"]
    out = []
    for number, (a, b) in enumerate(zip(got, want), start=1):
        if a == b:
            continue
        if not (a.endswith(APPROX) and b.endswith(APPROX)):
            out.append(f"line {number}: {a!r} != {b!r}")
            continue
        (label_a, _, value_a), (label_b, _, value_b) = (x[: -len(APPROX)].partition(": ") for x in (a, b))
        close = label_a == label_b and _close(_numbers(value_a), _numbers(value_b))
        if not close:
            out.append(f"line {number}: {a!r} !~ {b!r}")
    return out


def _numbers(text: str) -> list[float]:
    """The floats of a value written as '1.5' or '[1.5 2.5]'."""
    return [float(word) for word in text.strip("[]").split()]


def _close(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=RTOL, abs_tol=0.0) for x, y in zip(a, b))


def test_expected_output_is_complete_and_marks_what_is_approximate():
    lines = EXPECTED.read_text().splitlines()
    approx = [line for line in lines if line.endswith(APPROX)]
    assert lines[-1] == "all checks passed: True"
    # Only the losses are approximate (PyTorch operations and fused kernels); the distance
    # transform and every surface score, with voxel, surfel and mesh weights, are exact lines.
    assert len(approx) == 4
    assert all(line.startswith(("DiceLoss", "DiceCELoss")) for line in approx)
    assert not any("nan" in line.split(": ", 1)[-1] for line in approx)


def test_driver_prints_the_expected_output():
    assert _differences(_run(None), EXPECTED.read_text()) == []


def test_driver_output_does_not_depend_on_the_thread_count():
    one, eight = _run(1), _run(8)
    assert _differences(one, EXPECTED.read_text()) == []
    assert _differences(eight, EXPECTED.read_text()) == []
    exact = [line for line in one.splitlines() if not line.endswith(APPROX)]
    assert exact == [line for line in eight.splitlines() if not line.endswith(APPROX)]


@needs_gpu
def test_driver_on_gpu_prints_the_same_exact_lines():
    assert _differences(_run(None, "cuda"), EXPECTED.read_text()) == []
