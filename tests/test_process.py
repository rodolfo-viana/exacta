"""A stalled child must fail with diagnostics instead of hanging the test runner."""
import time

import pytest
import torch.multiprocessing as mp

from _process import destroy_process_group, spawn


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


def _gloo_teardown_worker(rank, directory):
    import os
    from datetime import timedelta
    from pathlib import Path

    import torch
    import torch.distributed as dist

    # Match small runners without changing the parent process's affinity.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    for iteration in range(20):
        dist.init_process_group("gloo", init_method=f"file://{Path(directory) / str(iteration)}",
                                rank=rank, world_size=2, timeout=timedelta(seconds=60))
        try:
            def gather():
                value = torch.tensor([0, -1])
                outputs = [torch.zeros_like(value) for _ in range(2)]
                dist.all_gather(outputs, value)
                return [item.tolist() for item in outputs]
            # Local tensor wrappers are released before group destruction, just as in
            # an empty metric compute. Gloo may still be releasing them on its worker.
            assert gather() == [[0, -1], [0, -1]]
        finally:
            destroy_process_group()


def test_gloo_teardown_after_temporary_collective_tensors(tmp_path):
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    spawn(_gloo_teardown_worker, args=(str(tmp_path),), directory=tmp_path / "workers", timeout=30)
