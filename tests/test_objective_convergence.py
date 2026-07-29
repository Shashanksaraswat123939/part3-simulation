from convergence import (
    REASON_DELTA_T,
    REASON_GATE_FAILURES,
    REASON_GRAD_NORM,
    ConvergenceTracker,
)
from objective_policy import (
    compose_penalized_time,
    final_ranking,
    search_ranking,
    select_build_candidate,
)
from optimizer_contract import CandidateOutcome, PenaltyInputs


def expect_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def candidate(candidate_id, raw, penalized, state="valid_simulated", reason=None):
    return CandidateOutcome(
        candidate_id=candidate_id,
        W_mm=130.0,
        x_front_mm=64.0,
        d_halo_mm=40.0,
        lifecycle_state=state,
        T_raw=raw,
        T_penalized=penalized,
        failure_reason=reason,
    )


def test_penalized_time_composition():
    penalties = PenaltyInputs(manufacturing_penalty_s=0.010, rule_margin_penalty_s=0.005)
    assert compose_penalized_time(1.250, penalties) == 1.265
    expect_raises(ValueError, compose_penalized_time, 0.0, penalties)
    expect_raises(ValueError, compose_penalized_time, float("nan"), penalties)


def test_search_ranking_and_final_ranking_are_different_by_design():
    fast_raw_with_penalty = candidate("a", 1.20, 1.40)
    slower_raw_clean = candidate("b", 1.25, 1.25)
    failed = candidate("z", None, None, state="CFD_failed", reason="bad mesh")

    assert [c.candidate_id for c in search_ranking([fast_raw_with_penalty, slower_raw_clean, failed])] == [
        "b",
        "a",
        "z",
    ]
    assert [c.candidate_id for c in final_ranking([fast_raw_with_penalty, slower_raw_clean, failed])] == [
        "a",
        "b",
    ]
    assert select_build_candidate([failed]) is None


def test_convergence_good_stop_reasons():
    # The delta-T criterion now needs N CONSECUTIVE small steps. This test used
    # to assert that ONE sub-threshold step converged, which is the behaviour
    # that stopped a real run after a single 0.372 ms delta against a +/-15 ms
    # noise floor -- convergence declared on noise.
    from optimizer_contract import INNER_CONVERGENCE_CONSECUTIVE
    delta = ConvergenceTracker(iteration_budget=10, gradient_norm_threshold=1.0e-9)
    assert not delta.update_success(1.5000, 1.0).stop
    status = None
    for i in range(INNER_CONVERGENCE_CONSECUTIVE):
        status = delta.update_success(1.5000 + (i + 1) * 5e-5, 1.0)
        if i < INNER_CONVERGENCE_CONSECUTIVE - 1:
            assert not status.stop, (
                f"stopped after {i + 1} small step(s); "
                f"{INNER_CONVERGENCE_CONSECUTIVE} are required")
    assert status.stop
    assert status.converged
    assert status.reason == REASON_DELTA_T

    grad = ConvergenceTracker(iteration_budget=10, gradient_norm_threshold=0.1)
    status = grad.update_success(1.5000, 0.01)
    assert status.stop
    assert status.converged
    assert status.reason == REASON_GRAD_NORM


def test_convergence_failure_stop_reason():
    tracker = ConvergenceTracker(iteration_budget=10, gradient_norm_threshold=1.0e-9)
    assert not tracker.update_failure().stop
    assert not tracker.update_failure().stop
    status = tracker.update_failure()
    assert status.stop
    assert not status.converged
    assert status.reason == REASON_GATE_FAILURES




def test_single_small_step_does_not_declare_convergence():
    """A production run stopped after ONE update on dT = 0.372 ms and reported
    converged=True, while the pipeline's drag noise is about +/-15 ms of race
    time. Under a noisy measurement a small delta by chance is likely, so a
    single-sample test fires at random."""
    from convergence import ConvergenceTracker, REASON_DELTA_T
    t = ConvergenceTracker(iteration_budget=50, gradient_norm_threshold=1e-12)
    t.update_success(3.2068, 1.0)
    s = t.update_success(3.20643, 1.0)          # dT = 0.37 ms
    assert not s.stop, "stopped on a single sub-threshold step"
    assert s.reason != REASON_DELTA_T


def test_consecutive_small_steps_do_declare_convergence():
    from convergence import ConvergenceTracker, REASON_DELTA_T
    from optimizer_contract import INNER_CONVERGENCE_CONSECUTIVE
    t = ConvergenceTracker(iteration_budget=50, gradient_norm_threshold=1e-12)
    t.update_success(3.2000, 1.0)
    last = None
    for i in range(INNER_CONVERGENCE_CONSECUTIVE):
        last = t.update_success(3.2000 + (i + 1) * 1e-6, 1.0)
    assert last.stop and last.converged and last.reason == REASON_DELTA_T, (
        f"expected convergence after {INNER_CONVERGENCE_CONSECUTIVE} small "
        f"steps, got {last}")


def test_a_big_step_resets_the_small_step_run():
    from convergence import ConvergenceTracker
    t = ConvergenceTracker(iteration_budget=50, gradient_norm_threshold=1e-12)
    t.update_success(3.2000, 1.0)
    t.update_success(3.2000 + 1e-6, 1.0)        # small
    t.update_success(3.3000, 1.0)               # big -- resets
    s = t.update_success(3.3000 + 1e-6, 1.0)    # small again, run length 1
    assert not s.stop, "the small-step run should have reset after a big step"


def test_budget_still_applies_while_waiting_for_small_steps():
    from convergence import ConvergenceTracker, REASON_BUDGET
    t = ConvergenceTracker(iteration_budget=3, gradient_norm_threshold=1e-12,
                           required_small_steps=99)
    t.update_success(3.2000, 1.0)
    t.update_success(3.2000 + 1e-6, 1.0)
    s = t.update_success(3.2000 + 2e-6, 1.0)
    assert s.stop and s.reason == REASON_BUDGET, (
        f"budget must still stop the loop while a small-step run accumulates, got {s}")


def test_gradient_norm_criterion_is_known_dead_not_assumed_live():
    """Pins a measured fact so nobody trusts a stop condition that cannot fire.

    The tracker stops when gradient_norm < DEFAULT_GRADIENT_NORM_THRESHOLD
    (1e-6), but it is fed the norm of the SCALAR objective sensitivities, which
    never approach zero -- dT/dmass alone is 18-29 s/kg whatever the shape.
    Measured 2026-07-28: 17.60 at the live operating point (1.76e+07x the
    threshold) and never below 6.03 across mass 48-300 g x drag 0.05-3 N.

    If someone later feeds this criterion the SHAPE gradient (the field that
    does vanish at an optimum), this test should fail and be replaced -- that
    would be the fix, not a regression.
    """
    import os, sys
    import numpy as np
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(_root, "part2-simulation"))
    from gradient_combiner import scalar_gradient_norm
    from optimizer_contract import DEFAULT_GRADIENT_NORM_THRESHOLD
    from race_objective import build_smooth_sheet_model
    from race_objective_adapter import race_value_and_grad_guarded

    model = build_smooth_sheet_model(
        os.path.join(_root, "part2-simulation", "co2_thrust_data.csv"))
    p = np.array([0.7149, 0.158598, 0.010, 1e-7, 1.0, 0.0292, 0.02, 0.10],
                 dtype=np.float64)
    norm = scalar_gradient_norm(race_value_and_grad_guarded(p, model)[2])
    assert norm > 1.0, f"scalar gradient norm collapsed to {norm}; re-check the objective"
    assert norm > 1e6 * DEFAULT_GRADIENT_NORM_THRESHOLD, (
        f"norm {norm} is now within reach of the {DEFAULT_GRADIENT_NORM_THRESHOLD} "
        f"threshold -- if the criterion has been made live, update this test")


if __name__ == "__main__":
    # Collected by name so an appended test can never be silently skipped --
    # the previous hand-written list sat ABOVE the tests added later, so they
    # were defined and never called.
    import sys as _sys
    _mod = _sys.modules[__name__]
    _fns = sorted(n for n in dir(_mod) if n.startswith("test_"))
    _passed = _failed = 0
    for _n in _fns:
        try:
            getattr(_mod, _n)()
            print(f"PASS {_n}")
            _passed += 1
        except Exception as _e:  # noqa: BLE001
            print(f"FAIL {_n}: {_e!r}")
            _failed += 1
    print(f"{_passed} passed, {_failed} failed")
    _sys.exit(1 if _failed else 0)
