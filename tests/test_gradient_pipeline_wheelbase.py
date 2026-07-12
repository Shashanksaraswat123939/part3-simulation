import numpy as np

from gradient_combiner import (
    calibrate_gradient_weights,
    combine_gradients,
    normalize_to_unit_rms,
    scalar_gradient_norm,
)
from optimizer_contract import GradientWeights
from pipeline_interface import PipelineBindings, validate_bindings
from wheelbase_sweep import coarse_w_values, refined_w_values


def expect_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def noop(*args, **kwargs):
    return None


def test_gradient_normalization_and_combination():
    g = np.array([3.0, 4.0])
    n = normalize_to_unit_rms(g)
    assert np.isclose(np.sqrt(np.mean(n * n)), 1.0)

    weights = GradientWeights(w_aero=1.0, w_mass=0.5, w_com=0.25, w_mfg=0.0)
    combined = combine_gradients(
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
        np.array([-3.0, -3.0]),
        np.array([0.0, 0.0]),
        weights,
    )
    assert combined.shape == (2,)
    assert np.allclose(combined, np.array([1.25, 1.25]))
    expect_raises(
        ValueError,
        combine_gradients,
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.array([1.0]),
        np.array([1.0]),
        weights,
    )


def test_gradient_weight_calibration_and_norm():
    def evaluate_T(inputs):
        return 2.0 * inputs["D20"] + 10.0 * inputs["m_total"] + 5.0 * inputs["h_com"]

    weights = calibrate_gradient_weights(
        evaluate_T=evaluate_T,
        base_inputs={"D20": 0.20, "m_total": 0.12, "h_com": 0.03},
        typical_changes={"D20": 0.01, "m_total": 0.005, "h_com": 0.002},
        mfg_typical_impact_s=0.01,
    )
    assert weights.w_mass == 1.0
    assert weights.w_aero > 0.0
    assert weights.w_com > 0.0
    assert weights.w_mfg > 0.0
    assert scalar_gradient_norm({"a": 3.0, "b": 4.0, "skip": None}) == 5.0


def test_pipeline_binding_validation():
    bindings = PipelineBindings(
        initialize_phi_fields=noop,
        warm_start_phi_fields=noop,
        perturb_phi_fields=noop,
        run_quality_gates=noop,
        compute_mass_report=noop,
        run_cfd=noop,
        evaluate_objective=noop,
        compute_adjoint_weight=noop,
        run_adjoint=noop,
        update_phi=noop,
        write_candidate_record=noop,
    )
    validate_bindings(bindings)

    bad = PipelineBindings(
        initialize_phi_fields=None,
        warm_start_phi_fields=noop,
        perturb_phi_fields=noop,
        run_quality_gates=noop,
        compute_mass_report=noop,
        run_cfd=noop,
        evaluate_objective=noop,
        compute_adjoint_weight=noop,
        run_adjoint=noop,
        update_phi=noop,
        write_candidate_record=noop,
    )
    expect_raises(TypeError, validate_bindings, bad)


def test_wheelbase_helper_ranges():
    coarse = coarse_w_values()
    assert len(coarse) == 21
    assert coarse[0] == 120.0
    assert coarse[-1] == 140.0

    refined = refined_w_values([120.0, 130.0, 140.0])
    assert 120.0 in refined
    assert 130.5 in refined
    assert 140.0 in refined
    assert min(refined) >= 120.0
    assert max(refined) <= 140.0


if __name__ == "__main__":
    test_gradient_normalization_and_combination()
    test_gradient_weight_calibration_and_norm()
    test_pipeline_binding_validation()
    test_wheelbase_helper_ranges()
