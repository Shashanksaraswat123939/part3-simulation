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
import json
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
        # Drag RESPONDS to the geometry. It used to be a hard-coded 0.18 for
        # every candidate, which meant the aero channel was inert in every test
        # that used this mock: the adjoint could return anything and the run
        # would behave identically, so nothing here could catch a regression in
        # the one coupling the project exists to study. A 25-iteration probe
        # run made that obvious -- D20 printed 0.18 on all 15 iterations while
        # the optimiser did pure mass minimisation.
        #
        # Frontal area x a fixed Cd, which is the dominant term for a bluff body
        # and cheap to evaluate: project the half-car onto x and double it. Not
        # a substitute for CFD, but it makes drag a real function of the shape
        # so the aero path is exercised rather than short-circuited.
        m = trimesh.load(stl_half_path, process=False)
        lo, hi = m.bounds
        area_half = float((hi[1] - lo[1]) * (hi[2] - lo[2]))
        A = max(2.0 * area_half, 1e-6)
        CD, Q = 0.45, 0.5 * 1.225 * 20.0 ** 2      # rho/2 * U^2 at the ref speed
        return CFDOutcome(D20=CD * Q * A, L=-0.02, Cm=0.005, A=A,
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


def test_decimation_restores_the_symmetry_plane():
    """Decimation runs AFTER extract_half_surface's clamp and moves vertices.

    Measured 2026-07-27: an aero-only iteration handed OpenFOAM a half-STL with
    a vertex at y = -1.000000e-6, exactly on Part 2's y >= -1e-6 tolerance edge,
    and the candidate died at the CFD gate. The clamp existed; it just ran
    before the operation that broke it.
    """
    import numpy as np
    import trimesh
    from pipeline_interface import _snap_symmetry_plane

    m = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    m.vertices[:, 1] += 0.5             # sit the box on y=0, like a real half
    m.vertices[0, 1] = -1.0e-6          # decimation-scale overshoot
    m.vertices[1, 1] = -3.0e-7
    out = _snap_symmetry_plane(m)
    assert (out.vertices[:, 1] >= 0.0).all(), "symmetry plane not restored"
    print("PASS test_decimation_restores_the_symmetry_plane")


def test_symmetry_snap_refuses_a_genuinely_broken_half():
    """Snapping must not paper over a half that is actually wrong."""
    import trimesh
    from pipeline_interface import _snap_symmetry_plane

    m = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    m.vertices[:, 1] += 0.5
    m.vertices[0, 1] = -5.0e-3          # 5 mm: not rounding
    try:
        _snap_symmetry_plane(m)
    except ValueError as exc:
        assert "below the y=0 symmetry plane" in str(exc)
        print("PASS test_symmetry_snap_refuses_a_genuinely_broken_half")
        return
    raise AssertionError("a 5 mm excursion was silently snapped")




def test_the_adjoint_sensitivity_field_is_saved_for_analysis():
    """The expensive product of each iteration must survive it.

    CandidateRecord has declared adjoint_sensitivity_field_path since the
    beginning, the serialiser writes it and the reader reads it, and nothing
    ever SET it -- every record carried "". So ~25 minutes of
    adjointOptimisationFoam per iteration was splatted onto the grid and
    dropped, and the field this project exists to analyse was never on disk.
    The phi snapshots say what the shape became; only this says why.

    Vertices must travel with it: sensitivity[i] belongs to vertices[i] and the
    array alone carries no indexing (see AdjointOutcome's docstring).
    """
    import inspect
    import numpy as np
    import inner_loop as il

    src = inspect.getsource(il._run_single_iteration)
    assert "_try_save_sensitivity" in src, (
        "the iteration does not save its adjoint sensitivity field")
    assert "adjoint_sensitivity_field_path" in src, (
        "the sensitivity path is not put into the candidate record")

    # Round-trip the saver itself on a realistic pair.
    import tempfile
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    n = 500
    sens = rng.normal(size=n)
    verts = rng.normal(size=(n, 3))
    adj = SimpleNamespace(sensitivity=sens,
                          half_mesh=SimpleNamespace(vertices=verts))
    with tempfile.TemporaryDirectory() as d:
        path = il._try_save_sensitivity(adj, "cand_iter0001", d)
        assert path, "saver returned no path on a valid pair"
        # `with`, because np.load holds the file open and Windows will not let
        # TemporaryDirectory remove a file that still has a handle on it.
        with np.load(path) as z:
            got_s, got_v = z["sensitivity"], z["vertices"]
        assert got_s.shape == (n,)
        assert got_v.shape == (n, 3)
        assert np.allclose(got_s, sens, atol=1e-6)
        assert np.allclose(got_v, verts, atol=1e-5)

        # A mismatched pair is unindexable; refuse rather than write nonsense.
        bad = SimpleNamespace(sensitivity=sens[:10],
                              half_mesh=SimpleNamespace(vertices=verts))
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            assert il._try_save_sensitivity(bad, "cand_iter0002", d) is None
        assert any("unindexable" in str(c.message) for c in caught), (
            "a length mismatch was saved silently")




def test_an_iteration_that_cannot_be_recorded_as_success_records_a_failure():
    """_run_single_iteration must ALWAYS produce a record.

    CandidateOutcome.__post_init__ validates lifecycle state, finiteness of
    T_raw/T_penalized, and T_penalized >= T_raw. That construction sat outside
    every try, so any of those raising meant neither the success path nor the
    failure helper wrote anything -- the exception left the function, the
    candidate died, and (before the sweep printed TaskFailure) nothing said why.

    That is what a real run did: d_halo=16 stopped after 3 of 6 iterations with
    iteration 3 finishing its CFD, adjoint and phi update and producing no
    record at all.
    """
    import inspect
    import inner_loop as il

    # Regex on the raw source. Index arithmetic does not work here: the
    # first `_try_write_record` in the function belongs to the `failure`
    # helper, so searching backwards from it finds that helper's
    # CandidateOutcome rather than the success one.
    import re
    src = inspect.getsource(il._run_single_iteration)
    guarded = re.search(
        r"try:\s*\n\s*outcome = CandidateOutcome\((?:.|\n)*?"
        r"except[^\n]*\n(?:.|\n)*?return failure\(", src)
    assert guarded, (
        "the success CandidateOutcome is not inside a try whose handler "
        "routes to `failure`; a validation error there escapes "
        "_run_single_iteration and the iteration vanishes without a "
        "record of any kind")




def test_a_whole_run_reaches_final_deliverables():
    """Nothing has ever reached the END of a run.

    Every real execution has died, been killed, or been invalidated partway
    through, so the tail of the pipeline -- convergence firing, warm-starting
    the next d_halo from a CARVED field, ranking, build-candidate selection,
    final_deliverables -- had never executed once. final_deliverables was tested
    nowhere at all, and select_build_candidate only against a synthetic failure.
    That is exactly where the remaining bugs would be: those paths only run on a
    converged car, and no car had ever converged.

    This drives the real geometry and the real inner loop with mocked CFD, over
    two d_halo values, at coarse spacing, in minutes rather than the ~10 hours
    the real solve takes. The CFD mock returns drag proportional to the
    bounding-box frontal area -- crude, but a real function of the shape, so the
    aero path is exercised rather than short-circuited. An earlier version of
    this docstring claimed that of a mock that returned a constant 0.18 N; it
    did not, and every test sharing the mock was running with a dead aero
    channel.
    """
    from optimizer_contract import OptimizerConfig, GradientWeights
    from orchestrator import SearchResult
    from objective_policy import final_ranking, select_build_candidate
    from wheelbase_sweep import run_d_halo_sweep

    with tempfile.TemporaryDirectory() as td:
        b = _mocked_openfoam_bindings(td)
        cfg = OptimizerConfig(
            rtc_validated_against_track_data=True,
            cfd_pipeline_validated_on_known_geometry=True,
            mu=0.01, wheel_moi_kg_m2=1e-7,
            iteration_budget=4, evolution_interval_iters=4, max_workers=1,
        )
        gw = GradientWeights(w_aero=1.0, w_mass=1.0, w_com=0.0, w_mfg=0.0)

        results = run_d_halo_sweep(
            bindings=b, config=cfg, d_halo_list=[18.0, 24.0],
            W_mm=130.0, x_front_mm=46.0, n_candidates=1, out_dir=td,
            gradient_weights=gw, n_evolution_rounds=1)

        assert len(results) == 2, f"sweep returned {len(results)} d_halo results"
        for r in results:
            assert r.best is not None, (
                f"d_halo produced no candidate at all; every iteration failed")

        pool = [o for r in results for o in r.all_outcomes if o.is_fully_valid]
        assert pool, "no fully valid candidate across the whole sweep"

        ranked = final_ranking(pool)
        build = select_build_candidate(pool)
        assert build is not None, "ranking produced no build candidate"
        assert ranked[0].T_raw <= ranked[-1].T_raw, "final_ranking is not sorted"

        # THE tail nobody had run: assemble the deliverables dict.
        sr = SearchResult(build_candidate=build, backup_ranking=list(ranked[1:3]),
                          coarse_results=results, refined_results=[])
        deliverables = sr.final_deliverables(td)

        for key in ("optimal_W_mm", "optimal_x_front_mm", "optimal_d_halo_mm",
                    "predicted_T_raw_s", "predicted_T_penalized_s",
                    "candidate_record_path", "robustness_status"):
            assert key in deliverables, f"final_deliverables is missing {key!r}"
        assert deliverables["predicted_T_raw_s"] > 0
        # An empty robustness list must say NOT RUN rather than read as clean.
        assert "NOT RUN" in deliverables["robustness_status"], (
            "no robustness_runner was passed, so the status must say so")

        # And it must be JSON-serialisable -- run_optimization writes it out.
        json.dumps(deliverables, default=str)


if __name__ == "__main__":
    # Collected by name; a hand-written call list silently drops every test
    # appended after it, which has already hidden several tests in this repo.
    _mod = sys.modules[__name__]
    _passed = _failed = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)()
            print("PASS " + _n)
            _passed += 1
        except Exception as _e:  # noqa: BLE001
            print("FAIL %s: %r" % (_n, _e))
            _failed += 1
    print("%d passed, %d failed" % (_passed, _failed))
    sys.exit(1 if _failed else 0)
