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
    delta = ConvergenceTracker(iteration_budget=10, gradient_norm_threshold=1.0e-9)
    assert not delta.update_success(1.5000, 1.0).stop
    status = delta.update_success(1.5005, 1.0)
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
