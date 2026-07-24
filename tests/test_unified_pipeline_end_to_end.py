"""
test_unified_pipeline_end_to_end.py

Proves the claim: the whole Part 3 pipeline runs end-to-end on the UNIFIED
single field with the REAL race objective, and the ONLY thing stubbed is the
OpenFOAM binary. We mock exactly run_cfd + run_adjoint (what shells out to
OpenFOAM) and leave everything else real, then require a VALID candidate with a
real race time to come out.

Before the unified wiring, the same test on the four-grid path produced
`best is None` -- every iteration rejected at the geometry gates
(`rearpod: inaccessible area`), before CFD ever mattered.

Run at coarse spacing for speed; the properties under test are wiring, not
resolution.
"""
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for p in ("part1-simulation", "part2-simulation", "part3-simulation",
          "part1-simulation/sandbox"):
    sys.path.insert(0, str(_ROOT / p))
os.environ.setdefault("PART2_PATH", str(_ROOT / "part2-simulation"))

import numpy as np  # noqa: E402
import coarse  # noqa: E402
coarse.use_spacing(3.0)

# The real thrust CSV lives beside part2 (installed there as the default).
_CSV = str(_ROOT / "part2-simulation" / "co2_thrust_data.csv")
_FHW = dict(
    co2_cartridge_mass_kg=0.023, co2_cartridge_com=(0.20, 0.0, 0.035),
    rear_wing_mass_kg=0.005, rear_wing_com=(0.24, 0.0, 0.05),
    wheels_axles_mass_kg=0.015, wheels_axles_com=(0.11, 0.0, 0.015),
)


def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)


def _mocked_openfoam_bindings(td):
    """unified_bindings with ONLY run_cfd + run_adjoint faked."""
    import trimesh
    from pipeline_interface import unified_bindings, CFDOutcome, AdjointOutcome
    b = unified_bindings(_CSV, _FHW, td)

    def fake_cfd(stl_half_path):
        return CFDOutcome(D20=0.18, L=-0.02, Cm=0.005, A=0.009,
                          converged=True, residual_final=1e-5)

    def fake_adjoint(stl_half_path, w):
        m = trimesh.load(stl_half_path, process=False)
        return AdjointOutcome(sensitivity=np.full(len(m.vertices), -1e-4), half_mesh=m)

    return dataclasses.replace(b, run_cfd=fake_cfd, run_adjoint=fake_adjoint)


def test_half_car_stl_is_watertight_and_passes_part2():
    # The handshake to OpenFOAM: the half-car STL must be watertight with all
    # vertices y >= -1e-6, or Part 2's run_half_car_cfd rejects it.
    from unified_phi import (
        build_unified_geometry, enforce_symmetry, extract_half_surface,
    )
    from body_profile import build_car_body_field
    g = build_unified_geometry(130.0, 46.0, 20.0)
    g.phi.grid = build_car_body_field(g)
    g.phi.apply_hard_constraints()
    enforce_symmetry(g)
    half = extract_half_surface(g)
    assert half.is_watertight, "half-car STL is not watertight (OpenFOAM would reject)"
    assert half.vertices[:, 1].min() >= -1e-6, "half-car STL has y < -1e-6 vertices"
    assert len(half.split(only_watertight=False)) == 1, "half-car STL is fragmented"
    _pass("test_half_car_stl_is_watertight_and_passes_part2")


def test_full_pipeline_completes_a_valid_candidate_with_openfoam_mocked():
    from optimizer_contract import OptimizerConfig, GradientWeights
    from inner_loop import run_inner_loop
    with tempfile.TemporaryDirectory() as td:
        b = _mocked_openfoam_bindings(td)
        cfg = OptimizerConfig(
            rtc_validated_against_track_data=True,
            cfd_pipeline_validated_on_known_geometry=True,
            mu=0.01, wheel_moi_kg_m2=1e-7, iteration_budget=3,
        )
        gw = GradientWeights(w_aero=1.0, w_mass=0.3, w_com=0.3, w_mfg=0.1)
        grids = b.initialize_phi_fields(130.0, 46.0, 20.0, seed=42)
        res = run_inner_loop(b, cfg, "e2e", 130.0, 46.0, 20.0, grids, td, gw)

    assert res.best is not None, \
        f"no valid candidate produced (stop_reason={res.stop_reason}); the " \
        f"pipeline ran but every iteration failed -- the four-grid regression"
    assert res.best.lifecycle_state in ("valid_simulated", "geometry_repaired", "converged"), \
        f"unexpected best lifecycle {res.best.lifecycle_state}"
    assert res.best.T_raw is not None and res.best.T_raw > 0, \
        f"best candidate has no real race time: {res.best.T_raw}"
    # It must have reached the REAL objective, i.e. past gates + CFD.
    assert any(o.lifecycle_state in ("valid_simulated", "geometry_repaired", "converged")
               for o in res.history), "no iteration passed the geometry gates"
    _pass("test_full_pipeline_completes_a_valid_candidate_with_openfoam_mocked")


def test_update_phi_evolves_the_unified_field():
    # The adjoint + objective gradients must actually move the level set.
    import trimesh
    from optimizer_contract import GradientWeights
    with tempfile.TemporaryDirectory() as td:
        b = _mocked_openfoam_bindings(td)
        geom = b.initialize_phi_fields(130.0, 46.0, 20.0, seed=42)
        g0 = geom.phi.grid.copy()
        mr = b.compute_mass_report(geom)
        gate = b.run_quality_gates(geom, "c", td)
        assert gate.stl_half_path, "gate produced no half STL"
        hm = trimesh.load(gate.stl_half_path, process=False)
        grads = {"dT_dD20": 0.4, "dT_dmass": 15.0, "dT_dh_com": -2.0,
                 "dT_dx_com": 0.0, "dT_dL": 0.0}
        for _ in range(5):
            b.update_phi(geom, np.full(len(hm.vertices), -2e-4), hm, 0.5,
                         GradientWeights(1.0, 0.3, 0.3, 0.1), grads, mr)
        moved = float(np.abs(geom.phi.grid - g0).sum())
    assert moved > 0.0, "update_phi did not move the field (aero/objective inert)"
    _pass("test_update_phi_evolves_the_unified_field")


if __name__ == "__main__":
    fns = [f for f in dir(sys.modules[__name__]) if f.startswith("test_")]
    passed = failed = 0
    for name in sorted(fns):
        try:
            globals()[name]()
            passed += 1
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
