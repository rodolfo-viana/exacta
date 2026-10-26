"""Bound spawned test workers, including stalls outside distributed collectives."""
from __future__ import annotations

import faulthandler
import time
from pathlib import Path
from typing import Any, Callable

import torch.multiprocessing as mp


def _entry(rank: int, function: Callable[..., Any], args: tuple, directory: str, timeout: float) -> None:
    with (Path(directory) / f"worker-{rank}.trace").open("w") as trace:
        trace.write("Worker entered\n")
        trace.flush()
        faulthandler.dump_traceback_later(timeout / 2, repeat=True, file=trace)
        try:
            function(rank, *args)
        finally:
            faulthandler.cancel_dump_traceback_later()


def spawn(function: Callable[..., Any], *, args: tuple, directory: Path,
          nprocs: int = 2, timeout: float = 90) -> None:
    """Run all ranks or fail with worker traces and reap every process.

    Process-group timeouts cover collectives, but not interpreter startup, native
    code or process teardown. This deadline bounds the complete worker lifetime.
    """
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    context = mp.spawn(_entry, args=(function, args, str(directory), timeout), nprocs=nprocs, join=False)
    try:
        while not context.join(timeout=min(1., max(0., timeout - (time.monotonic() - started)))):
            if time.monotonic() - started >= timeout:
                traces = []
                for rank in range(nprocs):
                    path = directory / f"worker-{rank}.trace"
                    message = path.read_text() if path.exists() else "Worker did not reach its entry point."
                    traces.append(f"Rank {rank}:\n{message}")
                raise TimeoutError(f"distributed test workers exceeded {timeout}s\n" + "\n".join(traces))
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
