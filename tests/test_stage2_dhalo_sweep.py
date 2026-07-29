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


def test_d_halo_values_respect_both_physical_bounds():
    from optimizer_contract import D_HALO_MIN_MM, validate_d_halo
    from wheelbase_sweep import d_halo_values
    for W in (120.0, 130.0, 140.0):
        vals = d_halo_values(W, n=6)
        assert len(vals) == 6, vals
        assert vals == sorted(vals)
        # Starts at the forward-most physical position (pocket front on the
        # front axle line), NOT 0 -- 0 would put the halo ahead of the axle.
        assert vals[0] == D_HALO_MIN_MM, f"{vals[0]} != {D_HALO_MIN_MM}"
        for d in vals:
            validate_d_halo(d, W)   # Part 1's gate, mirrored
        assert vals[-1] < W - 34.0, f"top sample {vals[-1]} not below {W - 34.0}"


def test_part1_and_part3_agree_on_the_d_halo_floor():
    """The two D_HALO_MIN_MM constants are duplicated by design; pin them."""
    import optimizer_contract as p3
    try:
        import geometry_contract as p1
    except ImportError:
        return  # Part 1 not importable here; the Part 1 suite covers its side
    assert p3.D_HALO_MIN_MM == p1.D_HALO_MIN_MM, (
        f"Part 3 says {p3.D_HALO_MIN_MM}, Part 1 says {p1.D_HALO_MIN_MM}"
    )


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


def test_prerequisite_gate_blocks_an_unvalidated_sweep():
    """The gate exists to stop a multi-day sweep whose numbers cannot mean
    anything, and had no test at all.

    It was previously dead: run_optimization.py hardcoded both flags True with
    the comment "set here so the wiring can be exercised", so nothing could
    ever trip it. They are CLI flags now, defaulting False. --smoke bypasses
    the orchestrator entirely (deliberately — it is a wiring check), so this is
    the only place the gate gets exercised.
    """
    from orchestrator import run_stage2_dhalo_search, PrerequisitesNotMet
    from optimizer_contract import OptimizerConfig

    for rtc, cfd, expect in ((False, False, True), (True, False, True),
                             (False, True, True)):
        cfg = OptimizerConfig(
            rtc_validated_against_track_data=rtc,
            cfd_pipeline_validated_on_known_geometry=cfd,
            mu=0.01, wheel_moi_kg_m2=1e-7,
        )
        try:
            # bindings=None / weights=None are fine: the gate must fire BEFORE
            # anything is used. If it does not, the TypeError from touching them
            # is itself the failure signal.
            run_stage2_dhalo_search(None, cfg, 130.0, 46.0,
                                    tempfile.gettempdir(), None)
        except PrerequisitesNotMet:
            assert expect, f"gate fired unexpectedly for rtc={rtc} cfd={cfd}"
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"expected PrerequisitesNotMet for rtc={rtc} cfd={cfd}, "
                f"got {type(exc).__name__}: {exc}") from exc
        raise AssertionError(
            f"no gate for rtc={rtc} cfd={cfd} — an unvalidated sweep would run")


def test_stage1_cargo_placement_reaches_the_built_geometry():
    """Stage 1's scored cargo choice must survive into every Stage-2 car.

    It did not: scalars_for_stage2() emitted cargo_x_start_m/cargo_flip, nothing
    downstream read them, and unified_bindings called build_unified_geometry
    without cargo_placement — so each Stage-2 car reverted to the geometric
    default and the fore-aft flip DOF was dead. Checked by building the SAME
    scalars with flip=False and flip=True and requiring the solid masks to
    differ; if the placement were being dropped, both builds would be identical.
    """
    import numpy as np
    import unified_phi as up
    from stage1_search import Stage1Point, Stage1Result

    import coarse
    coarse.use_spacing(3.0)

    W, xf, dh = 130.0, 46.0, 20.0
    base = up.build_unified_geometry(W, xf, dh, init_mode="full", seed=0)
    z_base = max(base.region.origin_m[2], 0.0025)
    x_start = 0.107

    # Two different x positions must produce different solid masks. If
    # cargo_placement were being dropped (the bug), both builds would fall back
    # to the geometric default and be byte-identical.
    masks = {}
    for xs in (x_start, x_start - 0.012):
        g = up.build_unified_geometry(
            W, xf, dh, init_mode="full", seed=0,
            cargo_placement={"x_start_m": xs, "z_base_m": z_base, "flip": False},
        )
        masks[xs] = g.phi.hard_mask_solid.copy()
    a, b = list(masks.values())
    assert not np.array_equal(a, b), (
        "two different cargo x positions produced identical solid masks — "
        "cargo_placement is being ignored by build_unified_geometry"
    )

    # And the erosion guard must REJECT a colliding placement rather than let
    # `hard_solid &= ~hard_air` delete it. Measured for real: at these scalars,
    # flip=True puts the wedge's wide 55 mm end at x=156..165 mm against a rear
    # axle at 176 mm and loses 17.8% of the mandatory T4.2 volume, while
    # flip=False at the same x_start loses none. find_cargo_placement screens
    # only the halo pocket, so nothing upstream catches this.
    try:
        up.build_unified_geometry(
            W, xf, dh, init_mode="full", seed=0,
            cargo_placement={"x_start_m": x_start, "z_base_m": z_base, "flip": True},
        )
    except ValueError as exc:
        assert "cargo" in str(exc).lower(), exc
    else:
        raise AssertionError(
            "flip=True at this placement erodes the mandatory cargo but was "
            "accepted — the CARGO_MAX_ERODED_FRACTION guard is not firing"
        )

    # And the handoff dict must be shaped for build_unified_geometry directly.
    # flip=False here: flip=True at these scalars is the colliding case asserted
    # above, and this check is about the SHAPE of the handoff dict, not placement
    # legality.
    pt = Stage1Point(W_mm=W, x_front_mm=xf, d_halo_mm=dh, T_proxy=1.0,
                     mass_kg=0.05, com_x_m=0.1, com_z_m=0.02,
                     cargo_x_start_m=x_start, cargo_flip=False, cargo_z_base_m=z_base)
    handoff = Stage1Result(best=pt).scalars_for_stage2()
    placement = handoff["cargo_placement"]
    assert set(placement) == {"x_start_m", "z_base_m", "flip"}, placement
    assert placement["flip"] is False and placement["z_base_m"] == z_base
    # Must be accepted verbatim by build_unified_geometry.
    up.build_unified_geometry(W, xf, dh, init_mode="full", seed=0,
                              cargo_placement=placement)


def test_launch_com_blends_the_propellant_and_excludes_it_from_totals():
    """The CO2 charge moves the COM while it is aboard, but is not in the dry
    totals -- that separation is what stops the mass double-count returning."""
    from pipeline_interface import MassReport

    dry = MassReport(total_mass_kg=0.048, com_x_m=0.1124, com_y_m=0.0,
                     com_z_m=0.0297, propellant_mass_kg=0.00787,
                     propellant_com=(0.2082, 0.0, 0.035))
    m, cx, _cy, cz = dry.launch_com()

    # Totals stay dry; only launch_com() sees the charge.
    assert dry.total_mass_kg == 0.048
    assert abs(m - (0.048 + 0.00787)) < 1e-12
    # Propellant sits aft and high, so it pulls the COM back and up.
    assert cx > dry.com_x_m and cz > dry.com_z_m
    assert 0.012 < cx - dry.com_x_m < 0.015, f"com_x shift {cx - dry.com_x_m}"
    # Exact mass-weighted blend.
    expect = (0.048 * 0.1124 + 0.00787 * 0.2082) / (0.048 + 0.00787)
    assert abs(cx - expect) < 1e-12
    # No charge -> launch_com is the dry COM exactly.
    empty = MassReport(total_mass_kg=0.048, com_x_m=0.1124, com_y_m=0.0, com_z_m=0.0297)
    assert empty.launch_com()[1] == 0.1124


def test_stability_gets_front_axle_origin_not_nose_origin():
    """check_stability's wheel-load formula needs COM measured from the front
    axle; mass_com_ingest reports it from the NOSE. Passing it unconverted
    reported 13.5%/86.5% front/rear where the truth is 48.9%/51.1%."""
    from stability_check import check_stability

    W_mm, x_front_mm, com_nose_m = 130.0, 46.0, 0.1124
    wrong = check_stability(total_mass_kg=0.048, x_com_m=com_nose_m, W_mm=W_mm)
    right = check_stability(total_mass_kg=0.048,
                            x_com_m=com_nose_m - x_front_mm / 1000.0, W_mm=W_mm)

    def front_frac(r):
        return r.static_W_front_N / (r.static_W_front_N + r.static_W_rear_N)

    assert front_frac(wrong) < 0.2, "sanity: the unconverted value is badly rear-biased"
    assert 0.45 < front_frac(right) < 0.55, (
        f"front-axle-origin COM should be near-balanced, got {front_frac(right):.3f}"
    )
    assert abs(front_frac(right) - front_frac(wrong)) > 0.3, (
        "the origin conversion must materially change the answer, or the test "
        "is not exercising it"
    )


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



def test_a_blacklisted_point_is_skipped_instead_of_re_run():
    """The failure memory is READ, not just written.

    Before this, every sweep recorded failures into FailureRegionMemory and
    nothing ever consulted it -- so the refined sweep, which steps 0.5 mm
    inside the 1.0 mm failure radius the coarse sweep had just filled, spent a
    full CFD budget re-running wheelbases that had already died three times.
    """
    import wheelbase_sweep as ws
    from evolutionary import FailureRegionMemory
    from optimizer_contract import CandidateOutcome

    mem = FailureRegionMemory(region_radius_mm=1.0, kill_threshold=3)
    for i in range(3):
        mem.record_failure(CandidateOutcome(
            candidate_id=f"dead{i}", W_mm=120.0, x_front_mm=46.0, d_halo_mm=30.0,
            lifecycle_state="geometry_rejected", T_raw=None, T_penalized=None,
            failure_reason="interior forced-air region"))

    seen = []
    real = ws.optimize_single_w

    def fake(bindings, config, W, xf, d, *a, **kw):
        seen.append(d)
        return ws.WResult(W_mm=W, best=None, all_outcomes=[], best_phi_grids=None)

    ws.optimize_single_w = fake
    try:
        ws.run_d_halo_sweep(
            bindings=None, config=None, d_halo_list=[20.0, 30.0, 40.0],
            W_mm=120.0, x_front_mm=46.0, n_candidates=1, out_dir=".",
            gradient_weights=None, failure_memory=mem, n_evolution_rounds=1,
        )
    finally:
        ws.optimize_single_w = real

    assert 30.0 not in seen, f"blacklisted d_halo=30 was still run: {seen}"
    assert seen == [20.0, 40.0], f"non-blacklisted values were skipped too: {seen}"


if __name__ == "__main__":
    # Collected by name. A hand-written call list silently drops every test
    # appended after it -- that bug has already hidden four tests in this repo.
    _mod = sys.modules[__name__]
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        _run(getattr(_mod, _n))
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
