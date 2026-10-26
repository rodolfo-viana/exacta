"""A stalled child must fail with diagnostics instead of hanging the test runner."""
import time

import pytest
import torch.multiprocessing as mp

from _process import spawn


def _stall(rank):
    time.sleep(60)


def test_spawn_deadline_reports_stack_and_reaps_child(tmp_path):
    before = {p.pid for p in mp.active_children()}
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="Rank 0") as error:
        spawn(_stall, args=(), directory=tmp_path, nprocs=1, timeout=6)
    assert time.monotonic() - started < 20
    # Startup can consume the deadline on a heavily loaded runner. Either report
    # the entered worker's diagnostic file or explicitly identify bootstrap delay.
    assert "Worker entered" in str(error.value) or "did not reach its entry point" in str(error.value)
    assert {p.pid for p in mp.active_children()} == before
