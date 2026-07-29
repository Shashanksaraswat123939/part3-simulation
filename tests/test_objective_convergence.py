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


if __name__ == "__main__":
    test_penalized_time_composition()
    test_search_ranking_and_final_ranking_are_different_by_design()
    test_convergence_good_stop_reasons()
    test_convergence_failure_stop_reason()


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
