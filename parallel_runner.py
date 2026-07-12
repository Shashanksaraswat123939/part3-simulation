"""
parallel_runner.py — Part 3 Stage 9: parallel execution of independent
inner loops.

Spec ("Parallel Execution"): each candidate's inner loop is fully
independent; no communication between candidates except at evolutionary
selection steps. Wall-clock per iteration = single CFD + adjoint time given
sufficient cores.

Threads (not processes) are used: the expensive work — snappyHexMesh,
simpleFoam, the adjoint — runs in external subprocesses in production, so
the GIL is irrelevant, and threads avoid pickling φ grids. max_workers=1
gives fully serial, fully deterministic execution for tests and debugging.

One candidate's crash must never take down the batch: every task's
exception is captured and returned as a TaskFailure, and the caller decides
what to do (typically: log it, count the candidate as dead, continue).
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TaskFailure:
    """A task that raised instead of returning. Carries enough context to
    debug without re-running."""

    task_index: int
    exception_type: str
    message: str
    traceback_text: str


def run_candidates_parallel(
    tasks: Sequence[Callable[[], Any]],
    max_workers: int = 1,
) -> list:
    """Run zero-argument callables, preserving input order in the output.

    Args:
        tasks: callables (typically functools.partial over run_inner_loop).
        max_workers: 1 = serial deterministic; >1 = thread pool.

    Returns:
        list of results, same length/order as tasks. A raised exception in
        task i is replaced by TaskFailure(task_index=i, ...) — the exception
        NEVER propagates and NEVER cancels sibling tasks.

    Raises:
        ValueError for max_workers < 1 or a non-callable task (caught before
        any task runs, so a bad batch fails atomically).
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    for i, t in enumerate(tasks):
        if not callable(t):
            raise ValueError(f"task {i} is not callable: {t!r}")
    if not tasks:
        return []

    def _guarded(index: int, fn: Callable[[], Any]):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — isolation is the whole point
            return TaskFailure(
                task_index=index,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback_text=traceback.format_exc(),
            )

    if max_workers == 1:
        return [_guarded(i, t) for i, t in enumerate(tasks)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_guarded, i, t) for i, t in enumerate(tasks)]
        return [f.result() for f in futures]
