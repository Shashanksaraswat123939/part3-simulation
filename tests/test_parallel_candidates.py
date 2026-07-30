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

    assert "uuid4" in inspect.getsource(oc.new_run_dir_name), (
        "run dir naming is no longer uuid4-based; concurrent candidates could "
        "collide and rmtree each other's case")
    for mod, fn in (("openfoam_case", oc.invoke), ("openfoam_adjoint", oa.invoke_adjoint)):
        assert "new_run_dir_name" in inspect.getsource(fn), (
            f"{mod} builds its run dir some other way than the shared helper")

    # Call PRODUCTION's naming, not a local re-implementation. The first
    # version of this test built `run_{pid}_{uuid4hex}` itself and asserted 64
    # were unique -- which tests the uuid stdlib, not this pipeline, and would
    # stay green if openfoam_case reverted to hash(stl_path) % 10000.
    names = set()
    lock = threading.Lock()

    def make_name():
        n = oc.new_run_dir_name()
        with lock:
            names.add(n)
        return n

    run_candidates_parallel([make_name for _ in range(64)], max_workers=8)
    assert len(names) == 64, (
        f"only {len(names)} unique run-dir names from 64 concurrent tasks; "
        f"colliding names mean one candidate rmtree's a live sibling's case")


def test_serial_and_parallel_agree():
    """max_workers=1 takes a different code path; it must give the same answer."""
    def make(i):
        return lambda: i * i

    tasks = [make(i) for i in range(12)]
    assert (run_candidates_parallel(tasks, max_workers=1)
            == run_candidates_parallel(tasks, max_workers=6)), (
        "serial and threaded paths disagree")




def test_a_failed_candidate_is_printed_not_just_collected(capsys=None):
    """TaskFailure.traceback_text must reach a human.

    parallel_runner captures a full traceback for every candidate that dies,
    and until 2026-07-30 nothing in the codebase read that field: the failure
    was appended to a list, carried to WResult, and never printed. A candidate
    could crash mid-sweep with a complete stack trace in hand and the only
    evidence was a missing record file -- which is exactly how a real run's
    d_halo=16 stopped after 3 of 6 iterations with no explanation anywhere.
    """
    import inspect
    import wheelbase_sweep as ws

    src = inspect.getsource(ws.optimize_single_w)
    assert "traceback_text" in src, (
        "the sweep collects TaskFailure but never reads its traceback; a "
        "candidate crash is invisible in the log")
    # and it must actually be PRINTED, not merely referenced. Comments are
    # stripped first: the explanation above the print is long enough that a
    # fixed character window lands inside it.
    code = " ".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    idx = code.index("traceback_text")
    assert "print(" in code[max(0, idx - 300):idx], (
        "traceback_text is referenced but not printed near the failure branch; "
        "collecting a traceback nobody reads is the bug this guards")


if __name__ == "__main__":
    # Collected by name; a hand-written list drops tests appended after it.
    _mod = sys.modules[__name__]
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        _run(getattr(_mod, _n))
    print("%d passed, %d failed" % (_passed, _failed))
    sys.exit(1 if _failed else 0)
