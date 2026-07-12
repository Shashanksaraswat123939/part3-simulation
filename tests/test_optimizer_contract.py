import math

from optimizer_contract import (
    CandidateOutcome,
    GradientWeights,
    OptimizerConfig,
    PenaltyInputs,
    validate_W,
    validate_d_halo,
)


def expect_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_wheelbase_and_halo_validation():
    validate_W(120.0)
    validate_W(140.0)
    expect_raises(ValueError, validate_W, 119.999)
    expect_raises(ValueError, validate_W, 140.001)

    validate_d_halo(0.0, 120.0)
    validate_d_halo(136.0, 120.0)
    expect_raises(ValueError, validate_d_halo, -0.01, 120.0)
    expect_raises(ValueError, validate_d_halo, 136.01, 120.0)


def test_config_and_weight_guards():
    GradientWeights(w_aero=1.0, w_mass=0.1, w_com=0.05, w_mfg=0.0)
    expect_raises(ValueError, GradientWeights, 0.0, 0.0, 0.0, 0.0)
    expect_raises(ValueError, GradientWeights, 1.0, -0.1, 0.0, 0.0)

    OptimizerConfig(
        rtc_validated_against_track_data=True,
        cfd_pipeline_validated_on_known_geometry=True,
        mu=0.08,
        wheel_moi_kg_m2=1.0e-6,
    )
    expect_raises(
        ValueError,
        OptimizerConfig,
        True,
        True,
        1.2,
        1.0e-6,
    )


def test_penalty_inputs_are_non_negative_and_finite():
    p = PenaltyInputs(manufacturing_penalty_s=0.002, rule_margin_penalty_s=0.003)
    assert math.isclose(p.total_s, 0.005)
    expect_raises(ValueError, PenaltyInputs, -0.001, 0.0)
    expect_raises(ValueError, PenaltyInputs, 0.0, float("nan"))


def test_candidate_lifecycle_consistency():
    ok = CandidateOutcome(
        candidate_id="ok",
        W_mm=130.0,
        d_halo_mm=40.0,
        lifecycle_state="valid_simulated",
        T_raw=1.25,
        T_penalized=1.30,
        failure_reason=None,
    )
    assert ok.is_fully_valid
    assert ok.ranking_time_s() == 1.30

    dead = CandidateOutcome(
        candidate_id="dead",
        W_mm=130.0,
        d_halo_mm=40.0,
        lifecycle_state="CFD_failed",
        T_raw=None,
        T_penalized=None,
        failure_reason="solver did not converge",
    )
    assert not dead.is_fully_valid
    assert dead.ranking_time_s() > 1000.0

    expect_raises(
        ValueError,
        CandidateOutcome,
        "bad_success",
        130.0,
        40.0,
        "valid_simulated",
        None,
        None,
        None,
    )
    expect_raises(
        ValueError,
        CandidateOutcome,
        "bad_failure",
        130.0,
        40.0,
        "CFD_failed",
        None,
        None,
        None,
    )


if __name__ == "__main__":
    test_wheelbase_and_halo_validation()
    test_config_and_weight_guards()
    test_penalty_inputs_are_non_negative_and_finite()
    test_candidate_lifecycle_consistency()
