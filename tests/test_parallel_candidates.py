"""
test_parallel_candidates.py -- concurrent candidates, the last untested surface.

--max-workers > 1 has never run. Every real execution used max_workers=1, and
the mocked orchestrator tests do too, so the thread pool in wheelbase_sweep is
unexercised. It is the riskiest of the remaining unknowns: candidates run as
threads in ONE process, sharing module state and a case directory, and the
uuid4 run-dir naming in openfoam_case/openfoam_adjoint exists precisely because
someone anticipated a collision there.

These tests drive the real thread pool with mocked CFD, checking the things that
break under concurrency rather than the things that break in sequence:
  * every candidate's result is returned, none lost or duplicated;
  * results stay ordered by task index regardless of completion order;
  * one candidate failing is isolated, not fatal to its siblings;
  * concurrent candidates do not collide on run-directory names.
"""
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_P3 = _HERE.parent
_ROOT = _P3.parent
for _p in (_P3, _ROOT / "part1-simulation", _ROOT / "part2-simulation",
           _ROOT / "part1-simulation" / "sandbox"):
    sys.path.insert(0, str(_p))

from parallel_runner import TaskFailure, run_candidates_parallel  # noqa: E402

_passed = _failed = 0


def _run(t):
    global _passed, _failed
    try:
        t()
        print(f"PASS {t.__name__}")
        _passed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {t.__name__}: {exc!r}")
        _failed += 1


def test_every_candidate_returns_exactly_once():
    seen = []
    lock = threading.Lock()

    def make(i):
        def task():
            time.sleep(0.01 * ((7 - i) % 4))   # finish out of submission order
            with lock:
                seen.append(i)
            return i * 10
        return task

    results = run_candidates_parallel([make(i) for i in range(8)], max_workers=4)
    assert sorted(seen) == list(range(8)), f"tasks ran {sorted(seen)}"
    assert results == [i * 10 for i in range(8)], (
        f"results are not in task order: {results}")


def test_results_keep_task_order_not_completion_order():
    """A sweep ranks candidates by index; reordering would mislabel them."""
    def make(i, delay):
        return lambda: (time.sleep(delay), i)[1]

    # Task 0 finishes last on purpose.
    tasks = [make(0, 0.12), make(1, 0.01), make(2, 0.02), make(3, 0.01)]
    results = run_candidates_parallel(tasks, max_workers=4)
    assert results == [0, 1, 2, 3], f"order not preserved: {results}"


def test_one_failing_candidate_does_not_take_down_the_batch():
    def ok(i):
        return lambda: i

    def boom():
        raise RuntimeError("adjoint diverged")

    results = run_candidates_parallel([ok(0), boom, ok(2)], max_workers=3)
    assert results[0] == 0 and results[2] == 2, f"siblings lost: {results}"
    assert isinstance(results[1], TaskFailure), f"expected TaskFailure: {results[1]}"
    assert "adjoint diverged" in results[1].message
    assert results[1].task_index == 1
    assert results[1].traceback_text, "a failure with no traceback is undiagnosable"


def test_concurrent_run_directory_names_do_not_collide():
    """uuid4 run dirs exist so parallel candidates cannot share a case dir.

    hash(stl_path) % 10000 was the original scheme; under threads a collision
    would rmtree a live sibling's case mid-solve.
    """
    import openfoam_case as oc  # noqa: F401  -- import guard
    import inspect
    import openfoam_adjoint as oa

    for mod, fn in (("openfoam_case", oc.invoke), ("openfoam_adjoint", oa.invoke_adjoint)):
        src = inspect.getsource(fn)
        assert "uuid4" in src, (
            f"{mod} does not use uuid4 for its run dir; concurrent candidates "
            f"could collide and delete each other's case")

    names = set()
    lock = threading.Lock()

    def make_name():
        import os
        import uuid as _uuid
        n = f"run_{os.getpid()}_{_uuid.uuid4().hex[:12]}"
        with lock:
            names.add(n)
        return n

    run_candidates_parallel([make_name for _ in range(64)], max_workers=8)
    assert len(names) == 64, f"only {len(names)} unique run-dir names from 64 tasks"


def test_serial_and_parallel_agree():
    """max_workers=1 takes a different code path; it must give the same answer."""
    def make(i):
        return lambda: i * i

    tasks = [make(i) for i in range(12)]
    assert (run_candidates_parallel(tasks, max_workers=1)
            == run_candidates_parallel(tasks, max_workers=6)), (
        "serial and threaded paths disagree")


if __name__ == "__main__":
    for t in (test_every_candidate_returns_exactly_once,
              test_results_keep_task_order_not_completion_order,
              test_one_failing_candidate_does_not_take_down_the_batch,
              test_concurrent_run_directory_names_do_not_collide,
              test_serial_and_parallel_agree):
        _run(t)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
