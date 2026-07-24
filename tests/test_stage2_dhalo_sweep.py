"""
test_stage2_dhalo_sweep.py — the Stage-2 d_halo sweep and the CFD-facing STL.

Guards the 2026-07-24 audit fixes:
  * d_halo sweep respects the STRICT upper bound (< W-34), so the sweep can
    never propose a geometry Part 1 rejects.
  * run_d_halo_sweep varies d_halo and holds W/x_front fixed (the whole point
    of the two-stage split — the old sweep varied W).
  * config.iteration_budget is honoured per round; it used to be silently
    replaced by evolution_interval_iters, so --iteration-budget did nothing.
  * the half-STL handed to OpenFOAM stays under the triangle budget AND still
    satisfies Part 2's hard contract (ASCII, watertight, right-half only,
    trimesh vertex order index-aligned with the raw STL vertex lines).
"""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_P3 = _HERE.parent
_ROOT = _P3.parent
for _p in (_P3, _ROOT / "part1-simulation", _ROOT / "part2-simulation",
           _ROOT / "part1-simulation" / "sandbox"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("PART2_PATH", str(_ROOT / "part2-simulation"))

_passed, _failed = 0, 0


def _run(fn):
    global _passed, _failed
    try:
        fn()
        print(f"PASS {fn.__name__}")
        _passed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {fn.__name__}: {exc}")
        _failed += 1


def test_d_halo_values_respect_the_strict_upper_bound():
    from optimizer_contract import validate_d_halo
    from wheelbase_sweep import d_halo_values
    for W in (120.0, 130.0, 140.0):
        vals = d_halo_values(W, n=6)
        assert len(vals) == 6, vals
        assert vals == sorted(vals)
        assert vals[0] == 0.0
        for d in vals:
            # Must not raise — this is Part 1's own gate.
            validate_d_halo(d, W)
        assert vals[-1] < W - 34.0, f"top sample {vals[-1]} not below {W - 34.0}"


def test_refined_d_halo_values_stay_legal():
    from optimizer_contract import validate_d_halo
    from wheelbase_sweep import refined_d_halo_values
    W = 130.0
    for d in refined_d_halo_values([0.0, 95.0], W):
        validate_d_halo(d, W)


def test_d_halo_values_rejects_a_degenerate_range():
    from wheelbase_sweep import d_halo_values
    try:
        d_halo_values(150.0)  # outside [120,140] -> validate_W raises
    except ValueError:
        return
    raise AssertionError("expected ValueError for an illegal W")


def test_sweep_varies_d_halo_and_holds_W_and_x_front_fixed():
    """The two-stage contract, checked against a fake inner loop."""
    import wheelbase_sweep as ws

    seen = []

    def fake_optimize(bindings, config, W_mm, x_front_mm, d_halo_mm, *a, **kw):
        seen.append((W_mm, x_front_mm, d_halo_mm))
        return ws.WResult(W_mm=W_mm, best=None, best_phi_grids=None)

    original = ws.optimize_single_w
    ws.optimize_single_w = fake_optimize
    try:
        ws.run_d_halo_sweep(
            bindings=object(), config=object(),
            d_halo_list=[0.0, 20.0, 40.0], W_mm=130.0, x_front_mm=46.0,
            n_candidates=1, out_dir="/tmp", gradient_weights=object(),
        )
    finally:
        ws.optimize_single_w = original

    assert [s[2] for s in seen] == [0.0, 20.0, 40.0], seen
    assert {s[0] for s in seen} == {130.0}, "W must not vary in a d_halo sweep"
    assert {s[1] for s in seen} == {46.0}, "x_front must not vary in a d_halo sweep"


def test_iteration_budget_is_not_silently_discarded():
    """--iteration-budget 1 must actually cap a round at 1 iteration."""
    import wheelbase_sweep as ws
    from optimizer_contract import OptimizerConfig

    captured = []

    def fake_inner(*a, **kw):
        captured.append(kw["config"].iteration_budget)
        return ws.InnerLoopResult(
            candidate_id="c", W_mm=130.0, x_front_mm=46.0, d_halo_mm=0.0,
            converged=False, stop_reason="budget", iterations_run=1, best=None,
        )

    class _B:
        def __getattr__(self, _n):
            return lambda *a, **k: {}

    original = ws.run_inner_loop
    ws.run_inner_loop = fake_inner
    try:
        ws.optimize_single_w(
            _B(),
            OptimizerConfig(
                rtc_validated_against_track_data=True,
                cfd_pipeline_validated_on_known_geometry=True,
                mu=0.01, wheel_moi_kg_m2=1e-7,
                iteration_budget=1, evolution_interval_iters=10,
            ),
            130.0, 46.0, 0.0, n_candidates=1, out_dir="/tmp",
            gradient_weights=object(), n_evolution_rounds=1,
        )
    finally:
        ws.run_inner_loop = original

    assert captured == [1], (
        f"round budget was {captured}, expected [1] — iteration_budget is being "
        "overridden by evolution_interval_iters again"
    )


def test_cfd_stl_is_under_budget_and_still_meets_part2_contract():
    """The decimated half-STL must stay a legal Part 2 input.

    Decimation is an optimisation, but it runs on the exact mesh OpenFOAM
    meshes, so it must not break watertightness, the y>=0 half-car rule, or the
    per-vertex ordering the adjoint sensitivity is index-aligned to.
    """
    import trimesh
    import cfd_wrapper as cw
    import openfoam_case as oc
    import unified_phi as up
    from pipeline_interface import STL_TRIANGLE_BUDGET, _decimate_for_cfd

    import coarse
    coarse.use_spacing(2.0)  # 2 mm: structural properties, not resolution ones

    geom = up.build_unified_geometry(130.0, 46.0, 20.0, init_mode="full", seed=0)
    up.enforce_symmetry(geom)
    raw = up.extract_half_surface(geom)

    # Force the decimation path: at 2 mm spacing the raw mesh is already small,
    # so budgeting at STL_TRIANGLE_BUDGET would exercise nothing. The first
    # version of this test did exactly that and passed while decimation was in
    # fact a silent no-op on the real production mesh.
    budget = max(len(raw.faces) // 4, 64)
    half = _decimate_for_cfd(raw, budget)
    assert len(half.faces) < len(raw.faces), (
        f"decimation did not reduce {len(raw.faces)} faces at budget {budget} — "
        "it must reduce or warn, never silently return the original"
    )
    assert half.is_watertight, "decimated mesh must stay watertight for Part 2"
    assert abs(half.volume - raw.volume) / raw.volume < 0.02, "volume drifted >2%"
    assert STL_TRIANGLE_BUDGET >= 60_000, "budget must stay above the hole threshold"
    with tempfile.TemporaryDirectory() as d:
        stl = os.path.join(d, "half.stl")
        half.export(stl, file_type="stl_ascii")
        cw._assert_watertight_stl(stl)          # ASCII + edge-manifold + y >= 0
        raw = oc.read_ascii_stl_vertices(stl)
        tm = trimesh.load(stl, process=False)
        assert len(tm.vertices) == len(raw), (
            f"trimesh sees {len(tm.vertices)} vertices, the STL has {len(raw)} "
            "vertex lines — the adjoint sensitivity would be misaligned"
        )
        assert min(v[1] for v in raw) >= -1e-6, "half-car STL must have y >= 0"


def test_ground_plane_sits_on_the_track_with_a_rolling_road():
    """domain_box's lowerWall must be the track, not 250 mm below it."""
    import openfoam_case as oc
    bounds = ((0.0001, 0.0, 0.0016), (0.233, 0.0356, 0.0647))
    box_min, box_max = oc.domain_box(bounds)
    assert box_min[2] == 0.0, f"lowerWall at z={box_min[2]}, expected the track (0.0)"
    assert box_min[1] == 0.0, "symmetry plane must sit on the centreline"

    ub_min, ub_max = oc.underbody_box(bounds)
    assert ub_max[2] > bounds[0][2], "underbody box must reach above the car floor"

    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "case"
        stl = Path(d) / "s.stl"
        _write_tetra(stl)
        oc.build_case(str(run), str(stl), oc.OpenFOAMRunConfig(resolution="coarse"))
        u = (run / "0" / "U").read_text()
        assert "lowerWall   { type fixedValue; value uniform (20.0 0 0); }" in u, (
            "lowerWall must be a rolling road carrying the freestream speed:\n" + u
        )
        snappy = (run / "system" / "snappyHexMeshDict").read_text()
        assert "underbody" in snappy, "underbody refinement region missing"
        assert "searchableBox" in snappy


def _write_tetra(path: Path) -> None:
    v = [(0, 0, 0.002), (0.01, 0, 0.002), (0, 0.01, 0.002), (0, 0, 0.012)]
    tris = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    out = ["solid car"]
    for a, b, c in tris:
        out += ["facet normal 0 0 0", "  outer loop"]
        out += [f"    vertex {v[i][0]} {v[i][1]} {v[i][2]}" for i in (a, b, c)]
        out += ["  endloop", "endfacet"]
    out.append("endsolid car")
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    for t in (
        test_d_halo_values_respect_the_strict_upper_bound,
        test_refined_d_halo_values_stay_legal,
        test_d_halo_values_rejects_a_degenerate_range,
        test_sweep_varies_d_halo_and_holds_W_and_x_front_fixed,
        test_iteration_budget_is_not_silently_discarded,
        test_ground_plane_sits_on_the_track_with_a_rolling_road,
        test_cfd_stl_is_under_budget_and_still_meets_part2_contract,
    ):
        _run(t)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
