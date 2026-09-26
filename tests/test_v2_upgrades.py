"""Part 3 checks for the 2026-09-25 upgrade."""
import sys


def test_final_ranking_excludes_underweight_cars():
    from objective_policy import final_ranking
    from optimizer_contract import CandidateOutcome
    mk = lambda cid, t, m: CandidateOutcome(  # noqa: E731
        candidate_id=cid, W_mm=120.0, x_front_mm=46.0, d_halo_mm=30.0,
        lifecycle_state="valid_simulated", T_raw=t, T_penalized=t + 0.01,
        failure_reason=None, competition_mass_kg=m)
    ranked = final_ranking([mk("light", 1.40, 0.0459), mk("legal", 1.45, 0.0482),
                            mk("unknown", 1.50, None)])
    assert [c.candidate_id for c in ranked] == ["legal", "unknown"]


def test_defaults_are_the_measured_values():
    import optimizer_contract as oc
    assert oc.DEFAULT_WHEEL_MOI_KG_M2 == 1.39e-7
    assert oc.DEFAULT_ROLLING_MU == 0.010


def test_inner_loop_zeroes_mass_push_when_ballast_absorbs():
    import inspect
    import inner_loop
    src = inspect.getsource(inner_loop._run_single_iteration)
    assert 'ballast_regime", "none") == "absorbing"' in src
    assert '_update_grads["dT_dmass"] = 0.0' in src


def test_aero_body_steps_are_off_and_stop_the_loop():
    import dataclasses
    import inspect
    import inner_loop
    from optimizer_contract import OptimizerConfig
    f = {x.name: x.default for x in dataclasses.fields(OptimizerConfig)}
    assert f["aero_body_steps"] is False
    assert "if aero_phase and not config.aero_body_steps:" in inspect.getsource(inner_loop.run_inner_loop)


def test_stability_flag_is_json_safe_with_numpy_inputs():
    """Regression 2026-09-26: a numpy x_com made statically_stable np.bool_,
    json.dump refused it, and no CI record was ever written."""
    import json
    import numpy as np
    from stability_check import check_stability
    r = check_stability(total_mass_kg=np.float64(0.071), x_com_m=np.float64(0.06),
                        W_mm=120.3, prerequisites_met=False)
    json.dumps({"s": r.statically_stable})


if __name__ == "__main__":
    _mod = sys.modules[__name__]
    _fails = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)(); print("PASS", _n)
        except Exception as e:  # noqa: BLE001
            _fails += 1; print("FAIL", _n, "->", repr(e))
    print(f"{_fails} failed")
    sys.exit(1 if _fails else 0)
