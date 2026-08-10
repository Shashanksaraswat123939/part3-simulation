"""
inner_loop.py — Part 3 Stage 7: single-candidate inner φ optimization loop.

Implements the 15-step loop from 03_optimizer_workflow ("Inner Loop: φ Shape
Optimization") for ONE candidate at ONE wheelbase, with the failure-recovery
and lifecycle rules from Part 1's spec and the Candidate Lifecycle table.

Design decisions recorded here (and enforced by tests):

  * CFD health is a GATE. Part 2's run_half_car_cfd returns non-converged
    force values as ordinary numbers (converged=False on the health report,
    no exception). Nothing in Part 2 stops those numbers reaching the race
    objective — Part 2 audit finding N-5. Part 3 closes that hole: when
    config.require_cfd_convergence is True (default), a non-converged CFD
    run is treated as CFD_failed for that iteration.

  * Candidate records are written with iteration-suffixed IDs
    ("{candidate_id}_iter{k:04d}") so per-iteration history is never
    overwritten — Part 2's write_candidate_record silently overwrites
    same-ID JSON (audit finding N-9), so uniqueness is Part 3's job.

  * Objective exceptions (ValueError from Part 2's adapter guards, JAX
    failures, anything) map to "objective_failed", are logged with the
    exception text, and the candidate continues to its next iteration —
    consistent with the spec's failure table. The consecutive-failure
    counter handles chronic failure.

  * T_penalized = Part 2's COM-penalized time + Part 3 penalties
    (manufacturing from gate outcome, rule margin). See objective_policy.
"""

from __future__ import annotations

import os
import math
import traceback
import warnings

# Distinct record-write failure causes already warned about, so a broken record
# layer says so once rather than once per iteration.
_RECORD_WRITE_WARNED: set = set()
from dataclasses import dataclass, field
from typing import Callable, Optional

from convergence import ConvergenceTracker, REASON_BUDGET
from gradient_combiner import scalar_gradient_norm
from objective_policy import compose_penalized_time
from optimizer_contract import (
    CandidateOutcome,
    OptimizerConfig,
    GradientWeights,
    PenaltyInputs,
    validate_W,
    validate_x_front,
    validate_d_halo,
)
from pipeline_interface import PipelineBindings, validate_bindings


@dataclass
class IterationLog:
    iteration: int
    lifecycle_state: str
    T_raw: Optional[float]
    T_penalized: Optional[float]
    gradient_norm: Optional[float]
    failure_reason: Optional[str]
    record_path: Optional[str]
    # Drag and mass are carried separately because T_raw alone cannot validate
    # the adjoint. T_raw moves with BOTH the drag the adjoint minimises and the
    # mass the scalar gradient removes, so a rise or fall in it is unattributable.
    # A 2026-07-27 smoke run read a T_raw rise as an inverted adjoint sign; the
    # aero term was inert and the mass term was doing all of it. To test the
    # adjoint you need D20 at w_mass=0, which means D20 has to be in the log.
    D20: Optional[float] = None
    total_mass_kg: Optional[float] = None


@dataclass
class InnerLoopResult:
    candidate_id: str
    W_mm: float
    x_front_mm: float
    d_halo_mm: float
    converged: bool
    stop_reason: str
    iterations_run: int
    best: Optional[CandidateOutcome]
    history: list = field(default_factory=list)
    final_phi_grids: Optional[dict] = None


# Penalty provider: (GateOutcome, dT/dmass) -> Part 3 penalties. dT/dmass comes
# from the objective for THIS candidate, so a provider can price geometry in
# seconds without inventing a coefficient. Default is machinability_penalty.
PenaltyProvider = Callable[[object, float], PenaltyInputs]


_ZERO_PENALTY_WARNED = set()


def machinability_penalty(gate_outcome: object,
                          dT_dmass_s_per_kg: float) -> PenaltyInputs:
    """Charge unreachable surface at the objective's own mass sensitivity.

    Material behind a face the cutter cannot reach does not get removed, so the
    manufactured car is heavier than the simulated one. That is a race-time
    cost the objective already knows how to price: dT/dmass, which Part 2
    returns for every candidate. No new coefficient is introduced.

    The depth is MIN_RADIUS_M, the minimum machining radius. A cutter cannot
    approach an unreachable surface closer than its own radius, so one
    tool-radius layer under each blocked face is a floor on the material left
    behind -- and, being a property of the tool rather than of the mesh, it
    keeps the penalty independent of grid spacing. A penalty that moved with
    resolution would make Stage 1 and Stage 2 rank the same car differently.

        penalty_s = dT/dmass * (blocked_area * MIN_RADIUS_M * rho_body)

    On the 2026-08-10 car: 165.8 mm^2 blocked -> 0.085 g trapped -> ~1.5 ms
    against a ~1.2 s race. Small, because the car is nearly machinable; it
    grows with the defect rather than being a fixed fine.

    This replaces zero_penalties as the default. Read that function's docstring
    for what the gap cost: every candidate in the 2026-07-29 sweep was logged
    "geometry_repaired" while paying nothing for it.
    """
    area_mm2 = getattr(gate_outcome, "inaccessible_area_mm2", None) or 0.0
    if area_mm2 <= 0.0:
        return PenaltyInputs(manufacturing_penalty_s=0.0, rule_margin_penalty_s=0.0)
    from geometry_contract import DENSITY_BODY_KGM3, MIN_RADIUS_M
    trapped_kg = (area_mm2 * 1e-6) * MIN_RADIUS_M * DENSITY_BODY_KGM3
    # A negative dT/dmass means the objective currently wants MORE mass (the
    # T3.6 floor is pushing back out). Rewarding unmachinable geometry for that
    # would be perverse, so the penalty floors at zero rather than flipping.
    seconds = max(0.0, float(dT_dmass_s_per_kg)) * trapped_kg
    return PenaltyInputs(manufacturing_penalty_s=seconds,
                         rule_margin_penalty_s=0.0)


def zero_penalties(gate_outcome: object,
                   dT_dmass_s_per_kg: float = 0.0) -> PenaltyInputs:
    """Explicit zero penalties. NO LONGER THE DEFAULT -- see
    machinability_penalty, which prices blocked area through dT/dmass.

    Kept so a caller can deliberately switch the manufacturing term off (an
    aero-only study, a test isolating the objective), and because the warning
    below is still the right behaviour when someone does: it names the measured
    area being ignored rather than letting a silent 0.0 look like a clean car.

    History, because it is the reason the default changed: this was the default
    for the whole project. Every candidate in the 2026-07-29 sweep was logged
    "geometry_repaired" -- the accessibility check DID find unreachable surface
    -- while paying nothing for it, which is free rein to evolve an
    unmanufacturable shape. The stated reason for returning zero was that the
    magnitude was a calibration decision and a made-up coefficient would be
    worse than the gap. That reasoning was sound; what unblocked it was
    realising no new coefficient is needed, because dT/dmass already prices
    trapped material in seconds.
    """
    area = getattr(gate_outcome, "inaccessible_area_mm2", None)
    cid = getattr(gate_outcome, "stl_path", None) or "?"
    if area and cid not in _ZERO_PENALTY_WARNED:
        _ZERO_PENALTY_WARNED.add(cid)
        warnings.warn(
            f"{area:.1f} mm^2 of this candidate's surface is unreachable by the "
            f"cutter, and the manufacturing penalty applied for it is 0.0 s "
            f"because this run passed zero_penalties explicitly. Unmanufacturable "
            f"geometry is ranking exactly as well as manufacturable geometry. "
            f"Drop the override to get machinability_penalty, the default.",
            RuntimeWarning, stacklevel=2)
    return PenaltyInputs(manufacturing_penalty_s=0.0, rule_margin_penalty_s=0.0)



# T3.6 minimum mass, enforced in STAGE 2.
#
# The barrier lived only in Stage 1's proxy (bayesian_outer_search). Stage 2's
# objective is the real race time, which has no mass floor at all -- lighter is
# always faster, so nothing stopped it carving straight through the regulation
# minimum.
#
# That was harmless while Stage 2 started from the 150 g envelope and never got
# near the floor in its iteration budget. It stopped being harmless the moment
# Stage 1 began handing over a car ALREADY at the floor: the 2026-08-05 run
# seeded at 27.93 g of machined body, which is 46.93 g of competition mass
# against a 48 g minimum, and every further iteration would have made it more
# illegal.
#
# Measured on the COMPETITION mass -- T3.6 excludes the CO2 cartridge -- and
# applied two ways, because a penalty that only affects the score lets the
# optimiser keep walking downhill:
#   * as a penalty on T_penalized, so an underweight car ranks badly; and
#   * via t36_descent_gradient, which REPLACES dT/dmass so the shape update
#     pushes material back out.
# The second is what holds the line. It has to replace rather than add: an
# added barrier cancels against the physics term and the descent rests exactly
# where they cancel, which is always inside the illegal region.
#
# An earlier version of this comment said Stage 1's barrier "settles ~1.75 g
# under the floor, which is the discrete level-set equilibrium rather than a
# defect". Both halves were wrong. It settled under the floor because it
# SUBTRACTED the barrier from dT/dmass -- the same defect, since fixed there
# too -- and a car that comes to rest below the T3.6 minimum is illegal, which
# is a defect however it arises.
T36_MIN_COMPETITION_MASS_KG: float = 0.048
T36_BARRIER_WEIGHT: float = 100.0
# The DESCENT aims above the floor; the ranking penalty still measures against
# the floor itself. See t36_descent_gradient for why the target has to be
# strictly higher: the growth velocity decays to zero as the deficit closes, so
# aiming at the floor converges TO it from below and every candidate stays
# fractionally illegal. 0.5 g also covers machining tolerance.
T36_TARGET_MARGIN_KG: float = 0.0005
# Cartridge mass, excluded from T3.6. Imported lazily to avoid a Part-2 import
# at module load in environments that only exercise Part 3's fakes.
_T36_CARTRIDGE_KG: float = 0.023


def t36_descent_gradient(total_mass_kg: float) -> "float | None":
    """dT/dmass to USE while T3.6 is active, or None when it is not.

    This REPLACES the physics mass gradient rather than adding to it, and that
    distinction is the entire fix. A soft penalty gradient ADDED to dT/dmass
    can cancel against the physics term, and the descent parks exactly where it
    does -- which is always strictly INSIDE the illegal region, because at the
    floor the barrier contributes nothing while physics still says "lighter is
    faster". Solving physics + barrier = 0 for the barrier as first written
    (weight 100, floor 48 g) gives the resting mass:

        dT/dmass    5 s/kg  ->  settles 0.058 g under the floor
        dT/dmass   17 s/kg  ->  settles 0.196 g under
        dT/dmass   40 s/kg  ->  settles 0.461 g under
        dT/dmass  100 s/kg  ->  settles 1.152 g under

    No choice of weight fixes that. Raising it only shrinks the offset while
    stiffening the shape velocity, and the offset scales with a gradient whose
    magnitude is not known in advance. Replacing the gradient removes the
    cancellation outright: while the car is underweight the only mass signal is
    "add mass", so the descent is always pushed back toward legality.

    CAVEAT on how far that goes. The numbers above, and the resting masses in
    the tests, come from a scalar model where the mass step is proportional to
    the gradient. The real Hamilton-Jacobi step is CFL-limited -- it moves the
    surface by CFL x spacing whatever the gradient magnitude -- so the descent
    OSCILLATES about the target with an amplitude set by grid spacing rather
    than resting on it. Measured on Stage 1's proxy at 2 mm, where one step
    moves 1-2 g, replacing rather than subtracting took the overshoot from
    2.5-4.1 g under the floor to 0.85-2.3 g under: a clear improvement, and
    still illegal at that spacing. Stage 2 runs at 0.5 mm where a step moves
    ~4x less mass, so the residual should be a few tenths of a gram -- but that
    has NOT been measured end to end, and until it has, the built car's mass
    must be checked against T3.6 rather than assumed legal.

    The RANKING penalty stays additive and stays measured against the true
    floor -- see t36_mass_barrier. Penalty decides which candidate wins;
    this decides which way the shape moves. They are different jobs.
    """
    comp = total_mass_kg - _T36_CARTRIDGE_KG
    target = T36_MIN_COMPETITION_MASS_KG + T36_TARGET_MARGIN_KG
    if comp >= target:
        return None
    return -(2.0 * T36_BARRIER_WEIGHT * (target - comp)
             / T36_MIN_COMPETITION_MASS_KG ** 2)


def t36_mass_barrier(total_mass_kg: float) -> tuple[float, float]:
    """(penalty_seconds, d_penalty/d_mass) for the T3.6 minimum-mass floor.

    Zero at or above the floor; steep and one-sided below it. The penalty is
    what makes an underweight candidate rank badly. The gradient it returns is
    reported, not used for the descent -- t36_descent_gradient does that, for
    the cancellation reason documented there.
    """
    comp = total_mass_kg - _T36_CARTRIDGE_KG
    if comp >= T36_MIN_COMPETITION_MASS_KG:
        return 0.0, 0.0
    deficit = (T36_MIN_COMPETITION_MASS_KG - comp) / T36_MIN_COMPETITION_MASS_KG
    penalty = T36_BARRIER_WEIGHT * deficit * deficit
    d_penalty = -(2.0 * T36_BARRIER_WEIGHT
                  * (T36_MIN_COMPETITION_MASS_KG - comp)
                  / T36_MIN_COMPETITION_MASS_KG ** 2)
    return penalty, d_penalty


def _iter_candidate_id(candidate_id: str, iteration: int) -> str:
    return f"{candidate_id}_iter{iteration:04d}"


def run_inner_loop(
    bindings: PipelineBindings,
    config: OptimizerConfig,
    candidate_id: str,
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    initial_phi_grids: dict,
    out_dir: str,
    gradient_weights: GradientWeights,
    penalty_provider: PenaltyProvider = machinability_penalty,
) -> InnerLoopResult:
    """Run the full inner loop for one candidate.

    Args:
        bindings: pipeline handshake (real or injected fake).
        config: optimizer configuration.
        candidate_id: base ID; per-iteration records get _iterNNNN suffixes.
        W_mm, x_front_mm, d_halo_mm: outer-loop scalars; validated against
            legal ranges.
        initial_phi_grids: dict of the four φ grids (fresh, warm-started, or
            perturbed) — ownership transfers to the loop, which mutates them.
        out_dir: directory for candidate records and φ snapshots.
        gradient_weights: normalized-gradient weights (from calibration).
        penalty_provider: GateOutcome -> PenaltyInputs.

    Returns:
        InnerLoopResult. `best` is the best successful iteration's outcome
        by T_penalized (search metric), or None if no iteration succeeded.
        `final_phi_grids` are the loop's φ fields at stop (for warm-starting
        the next wheelbase or evolutionary perturbation).
    """
    validate_bindings(bindings)
    validate_W(W_mm)
    validate_x_front(x_front_mm, W_mm)
    validate_d_halo(d_halo_mm, W_mm)
    if not candidate_id or "/" in candidate_id or "\\" in candidate_id:
        raise ValueError(f"candidate_id must be a plain name, got {candidate_id!r}")

    tracker = ConvergenceTracker(
        iteration_budget=config.iteration_budget,
        gradient_norm_threshold=config.gradient_norm_threshold,
    )
    phi_grids = initial_phi_grids
    history: list[IterationLog] = []
    best: Optional[CandidateOutcome] = None
    stop_reason = REASON_BUDGET
    converged = False

    iteration = 0
    while True:
        iteration += 1
        iter_id = _iter_candidate_id(candidate_id, iteration)

        outcome, log, phi_snapshot_paths = _run_single_iteration(
            bindings, config, iter_id, W_mm, x_front_mm, d_halo_mm, phi_grids, out_dir,
            gradient_weights, penalty_provider, iteration,
        )
        history.append(log)
        # One line per iteration, unconditionally. Without it a run that ends
        # early is undiagnosable from the log: the 2026-07-30 split4 run stopped
        # d_halo=16 after 3 of its 6 iterations and the only evidence was a
        # missing record file and an STL timestamp. Records are written for
        # successes and most failures, but the loop's own view -- which
        # iteration, what state, what it decided next -- was never printed.
        print(f"[inner_loop] {candidate_id} iter {log.iteration}: "
              f"{log.lifecycle_state}"
              f"  T_raw={log.T_raw}  T_pen={log.T_penalized}"
              f"  D20={log.D20}  mass={log.total_mass_kg}"
              f"  record={'yes' if log.record_path else 'NO'}"
              + (f"  FAILED: {log.failure_reason}" if log.failure_reason else ""),
              flush=True)

        if outcome is not None and outcome.T_penalized is not None:
            if best is None or outcome.T_penalized < best.T_penalized:
                best = outcome
            status = tracker.update_success(
                outcome.T_penalized,
                log.gradient_norm if log.gradient_norm is not None else math.inf,
            )
        else:
            status = tracker.update_failure()

        if status.stop:
            stop_reason = status.reason
            converged = status.converged
            break

    # Promote the best outcome to "converged" when the loop converged (spec
    # lifecycle: 'converged — inner loop converged, best candidate saved').
    if converged and best is not None:
        best = CandidateOutcome(
            candidate_id=best.candidate_id,
            W_mm=best.W_mm,
            x_front_mm=best.x_front_mm,
            d_halo_mm=best.d_halo_mm,
            lifecycle_state="converged",
            T_raw=best.T_raw,
            T_penalized=best.T_penalized,
            failure_reason=None,
            phi_snapshot_paths=best.phi_snapshot_paths,
            record_path=best.record_path,
        )

    print(f"[inner_loop] {candidate_id} STOPPED after {len(history)} "
          f"iteration(s): {stop_reason}"
          f"  (budget was {config.iteration_budget})", flush=True)
    return InnerLoopResult(
        candidate_id=candidate_id,
        W_mm=W_mm,
        x_front_mm=x_front_mm,
        d_halo_mm=d_halo_mm,
        converged=converged,
        stop_reason=stop_reason,
        iterations_run=tracker.iterations,
        best=best,
        history=history,
        final_phi_grids=phi_grids,
    )


def _run_single_iteration(
    bindings: PipelineBindings,
    config: OptimizerConfig,
    iter_id: str,
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    phi_grids: dict,
    out_dir: str,
    gradient_weights: GradientWeights,
    penalty_provider: PenaltyProvider,
    iteration: int,
) -> tuple:
    """One pass of steps 2–14. Returns (CandidateOutcome | None, IterationLog,
    phi_snapshot_paths). Outcome is None on any failure path; the log always
    carries the lifecycle state and reason."""

    def failure(state: str, reason: str, snaps: dict) -> tuple:
        outcome = CandidateOutcome(
            candidate_id=iter_id, W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            lifecycle_state=state, T_raw=None, T_penalized=None,
            failure_reason=reason, phi_snapshot_paths=snaps,
        )
        # Failed candidates are recorded too — the spec is explicit that
        # failed runs are not discarded; φ snapshots + failure reason feed
        # the evolutionary layer's failure-region memory.
        record_path = _try_write_record(bindings, outcome)
        log = IterationLog(
            iteration=iteration, lifecycle_state=state, T_raw=None,
            T_penalized=None, gradient_norm=None, failure_reason=reason,
            record_path=record_path,
        )
        return None, log, snaps

    # Steps 2–5: hard constraints + extraction + gates (Part 1 owns all of it;
    # run_quality_gates enforces constraints, extracts, repairs, retries).
    # P3-3: unexpected exceptions (ImportError, AttributeError from K-5's halo
    # ValueError during grid construction, etc.) must map to a lifecycle state
    # and produce a record — never propagate as bare exceptions that kill the loop.
    try:
        gate = bindings.run_quality_gates(phi_grids, iter_id, out_dir)
    except Exception as exc:  # noqa: BLE001
        return failure(
            "geometry_rejected",
            f"run_quality_gates raised unexpected error: {exc}\n"
            f"{traceback.format_exc(limit=3)}",
            {},
        )
    snaps = dict(gate.phi_snapshot_paths)
    if gate.lifecycle_state not in ("valid_simulated", "geometry_repaired"):
        return failure(gate.lifecycle_state, gate.failure_reason or "gates failed", snaps)
    if not gate.stl_half_path:
        return failure(
            "geometry_rejected",
            "gates reported success but produced no half-car STL — Part 1 bug",
            snaps,
        )

    # Step 7 (done before CFD so a mass/COM contract violation is caught as
    # cheaply as possible): mass + COM from φ grids.
    try:
        mass_report = bindings.compute_mass_report(phi_grids)
    except Exception as exc:  # noqa: BLE001 — must map to lifecycle, never crash loop
        return failure("objective_failed", f"mass/COM ingestion failed: {exc}", snaps)

    # Step 6: CFD.
    try:
        cfd = bindings.run_cfd(gate.stl_half_path)
    except Exception as exc:  # noqa: BLE001
        return failure("CFD_failed", f"CFD failed: {exc}", snaps)
    if config.require_cfd_convergence and not cfd.converged:
        return failure(
            "CFD_failed",
            f"CFD did not converge (residual_final={cfd.residual_final}); "
            "forces untrusted, iteration rejected",
            snaps,
        )

    # COM is taken with the CO2 charge FULLY ABOARD, every iteration, and is
    # not migrated as the propellant empties (project owner, 2026-07-24).
    # Justification measured before adopting it: the true COM travels 13.5 mm
    # forward in com_x and 0.75 mm down in com_z over the first ~0.2 s of a
    # ~1.23 s run, and ignoring that migration entirely costs <= 0.11 ms of race
    # time -- an upper bound, holding the launch COM for the whole run, which is
    # exactly what this does. That is far below every other modelling error in
    # the stack, and it buys a single unambiguous COM per candidate.
    #
    # Note the deliberate asymmetry: m_total stays DRY because the objective
    # adds the propellant mass itself over time (car_mass_from_time). Mass is
    # modelled as time-varying; COM is not.
    _, _launch_com_x, _, _launch_com_z = mass_report.launch_com()

    # Step 8: race objective.
    try:
        objective = bindings.evaluate_objective(
            D20=cfd.D20, L=cfd.L,
            m_total=mass_report.total_mass_kg,
            h_com=_launch_com_z,
            x_com=_launch_com_x,
            mu=config.mu, wheel_moi=config.wheel_moi_kg_m2,
        )
    except Exception as exc:  # noqa: BLE001
        return failure("objective_failed", f"race objective failed: {exc}", snaps)

    penalties = penalty_provider(
        gate, float(objective.gradients.get("dT_dmass", 0.0)))
    T_penalized = compose_penalized_time(objective.T_com_penalized, penalties)

    # T3.6: penalise an underweight car AND push the shape back out. Without the
    # gradient half, an underweight candidate merely scores badly while the
    # descent keeps removing material.
    _t36_pen, _ = t36_mass_barrier(mass_report.total_mass_kg)
    if _t36_pen > 0.0:
        T_penalized += _t36_pen
        _comp_g = (mass_report.total_mass_kg - _T36_CARTRIDGE_KG) * 1000.0
        if iteration == 1 or iteration % 5 == 0:
            # Print the gradient the DESCENT actually uses, not the barrier's
            # own -- they differ (the descent aims at floor + margin), and a log
            # that reports a number the code does not act on is how most of the
            # bugs in this pipeline stayed hidden.
            _shown = t36_descent_gradient(mass_report.total_mass_kg) or 0.0
            print(f"[T3.6] {iter_id}: competition mass {_comp_g:.2f} g is under "
                  f"the {T36_MIN_COMPETITION_MASS_KG*1000:.0f} g floor; "
                  f"penalty {_t36_pen:.3f} s, dT/dmass {_shown:+.1f} s/kg "
                  f"(pushing material back out)", flush=True)
    # ⚠ THIS CANNOT TRIGGER CONVERGENCE, and it is recorded rather than trusted.
    #
    # The tracker stops when gradient_norm < DEFAULT_GRADIENT_NORM_THRESHOLD
    # (1e-6). But this is the norm of the SCALAR objective sensitivities
    # (dT/dD20, dT/dmass, ...), which never approach zero: a lighter car is
    # always faster, so dT/dmass alone sits at 18-29 s/kg whatever the shape.
    # Measured 2026-07-28 at the live operating point: norm = 17.60, i.e.
    # 1.76e+07x the threshold, and sweeping mass 48-300 g against drag
    # 0.05-3 N never brings it below 6.03 -- still 6e+06x above.
    #
    # So the gradient criterion is DEAD: one of the four documented stop
    # conditions can never fire. It is not harmful (budget, delta-T and gate
    # failures all work), but it looks like a safety net and is not one.
    #
    # The quantity that DOES vanish at a shape optimum is the SHAPE gradient
    # dT/dSurface -- the field phi_updater actually steps along. Making this
    # criterion real means returning that field's norm from
    # apply_adjoint_to_unified and threading it through update_phi to here,
    # rather than reusing the scalar norm because it was the number in scope.
    grad_norm = scalar_gradient_norm(objective.gradients)

    # Step 9: stability check — computed and recorded; static instability is
    # reported in the record, not used to kill (killing criteria are the
    # gates; stability thresholds are a final-selection concern per spec's
    # robustness section). Import here to keep the module import-light.
    from stability_check import check_stability
    # ORIGIN CONVERSION, do not remove. mass_report.com_x_m is in CAR
    # coordinates (x=0 at the nose tip, physics_contract.MOMENT_REFERENCE_POINT_M);
    # check_stability's wheel-load formula needs it measured FROM THE FRONT AXLE.
    # Passing the nose-origin value straight through (what this did until
    # 2026-07-24) reported 13.5%/86.5% front/rear at W=130/x_front=46 where the
    # truth is 48.9%/51.1% -- a 35-point error, because a COM 112 mm behind the
    # NOSE is only 66 mm behind the AXLE.
    # LAUNCH condition, not finish-line condition. Static stability is about
    # wheelie risk, which peaks at t=0 where thrust is highest -- and that is
    # exactly when the ~7.9 g CO2 charge is still aboard, sitting well aft at
    # the cartridge. It pulls com_x +13.5 mm rearward at W=130/x_front=46,
    # unloading the front axle by ~10 percentage points. Evaluating stability
    # on the dry (finish-line) COM flatters the car at the one instant it
    # matters. Worth <=0.11 ms of race time, which is why the OBJECTIVE keeps
    # using the dry COM; worth 10 points of front-axle load here.
    _launch_mass, _launch_com_x, _, _ = mass_report.launch_com()
    stability = check_stability(
        total_mass_kg=_launch_mass,
        x_com_m=_launch_com_x - (x_front_mm / 1000.0),
        W_mm=W_mm,
    )

    # Steps 10–14: adjoint → combine → velocity extension → HJ → reinit.
    # combine/extend/HJ/reinit all live inside Part 1's update_phi (spec
    # signature: update_phi(phi_grids, right_half_sensitivity,
    # right_half_mesh, dt, gradient_weights)).
    try:
        # Same COM convention as the objective above -- these two MUST agree,
        # or the adjoint weight w_D20 = dT/dD20 is differentiating a different
        # operating point than the one the objective value came from.
        w_D20 = bindings.compute_adjoint_weight(
            D20=cfd.D20, L=cfd.L,
            m_total=mass_report.total_mass_kg,
            h_com=_launch_com_z,
            x_com=_launch_com_x,
            mu=config.mu, wheel_moi=config.wheel_moi_kg_m2,
        )
        # run_adjoint returns the sensitivity together with the half-car mesh
        # that defines its vertex ordering (AdjointOutcome). Passing
        # gate.meshes here instead -- a dict[str, Trimesh] with no .vertices --
        # is what silently broke every phi update; see AdjointOutcome's
        # docstring.
        adjoint = bindings.run_adjoint(gate.stl_half_path, w_D20)
        # SAVE IT. The sensitivity field is the expensive product of this
        # iteration -- roughly 25 minutes of adjointOptimisationFoam -- and it
        # was splatted onto the grid and then dropped. CandidateRecord has
        # declared `adjoint_sensitivity_field_path` since the beginning, the
        # serialiser writes it and the reader reads it, and NOTHING has ever
        # set it: every record carries "".
        #
        # It is also the analysis output this project exists to produce. Without
        # the field you cannot see where drag sensitivity concentrates, compare
        # it across d_halo, or check the adjoint against a finite difference
        # after the fact -- the phi snapshots say what the shape became, never
        # why. ~6 MB compressed per iteration against the 154 MB STL already
        # written beside it.
        #
        # Vertices travel WITH it, for the reason AdjointOutcome's docstring
        # gives: sensitivity[i] belongs to half_mesh.vertices[i] and the array
        # alone carries no indexing.
        sens_path = _try_save_sensitivity(adjoint, iter_id, out_dir)
        # The gradients the SHAPE UPDATE sees carry the T3.6 barrier; the ones
        # recorded in the candidate record stay pure physics, so the reported
        # dT/dmass is still the race objective's own number.
        _update_grads = dict(objective.gradients)
        _t36_descent = t36_descent_gradient(mass_report.total_mass_kg)
        if _t36_descent is not None:
            # REPLACES, does not add -- an added barrier cancels against the
            # physics term and parks the car inside the illegal region.
            _update_grads["dT_dmass"] = _t36_descent
        bindings.update_phi(
            phi_grids, adjoint.sensitivity, adjoint.half_mesh, config.hj_dt,
            gradient_weights, _update_grads, mass_report,
        )
    except Exception as exc:  # noqa: BLE001
        # Adjoint/update failure after a successful objective: the T values
        # are real and recorded, but the candidate can't improve further this
        # iteration. Treat as objective_failed for THIS iteration's lifecycle
        # while still logging the measured times in the failure reason.
        reason = (
            f"adjoint/φ-update failed after successful objective "
            f"(T_raw={objective.T_raw:.6f}, T_pen={T_penalized:.6f}): {exc}\n"
            f"{traceback.format_exc(limit=3)}"
        )
        return failure("objective_failed", reason, snaps)

    # Guarded, because this function's contract is that EVERY iteration
    # produces a record -- success or failure -- and this construction was the
    # one statement left that could escape it. CandidateOutcome.__post_init__
    # validates: lifecycle state, T_raw/T_penalized present and finite, and
    # T_penalized >= T_raw. Any of those raising here means neither the success
    # path below nor `failure` above ever writes anything, the exception leaves
    # _run_single_iteration entirely, and the candidate dies.
    #
    # That is not hypothetical. On 2026-07-30 d_halo=16 stopped after 3 of its 6
    # iterations: iteration 3 completed its CFD, adjoint and phi update, then
    # produced no record at all, and the traceback went into a TaskFailure that
    # nothing read. (The COM penalty was checked as a suspect and cleared -- it
    # is a degree-4 fit with a clean zero minimum at the 30 mm target and never
    # goes negative, so T_penalized >= T_raw always holds from that direction.)
    #
    # Whatever it turns out to be, an iteration that cannot describe itself as a
    # success should record itself as a failure, not vanish.
    try:
        outcome = CandidateOutcome(
            candidate_id=iter_id, W_mm=W_mm, x_front_mm=x_front_mm,
            d_halo_mm=d_halo_mm,
            lifecycle_state=gate.lifecycle_state,
            T_raw=objective.T_raw, T_penalized=T_penalized,
            failure_reason=None, phi_snapshot_paths=snaps,
        )
    except Exception as exc:  # noqa: BLE001
        return failure(
            "objective_failed",
            f"iteration completed CFD, adjoint and the phi update but could not "
            f"be recorded as a success (T_raw={objective.T_raw!r}, "
            f"T_pen={T_penalized!r}, state={gate.lifecycle_state!r}): {exc}\n"
            f"{traceback.format_exc(limit=3)}",
            snaps)
    # Carry the CFD and mass results into the record. Without them a ranked
    # table shows race time and nothing else -- merge_results printed empty D20
    # and mass columns on the first real record, and those are exactly the two
    # numbers you want beside T_raw when deciding whether a ranking is credible
    # (a car that is faster because it is lighter is a different story from one
    # that is faster because it is slipperier). Both are in scope here; the
    # record just never asked for them.
    record_path = _try_write_record(
        bindings, outcome,
        extra={"stability_notes": stability.notes,
               "statically_stable": stability.statically_stable,
               # The objects, not dicts: the serialiser reads attributes.
               # The winning car's geometry must be recoverable from its own
               # record. stl_path was declared on CandidateRecord and
               # serialised, but never passed -- every record carried "".
               "stl_path": gate.stl_path or "",
               "inaccessible_area_mm2": gate.inaccessible_area_mm2,
               "adjoint_sensitivity_field_path": sens_path,
               "cfd_force_report": cfd,
               "mass_report": mass_report,
               "com_report": mass_report,
               "gradients": objective.gradients},
    )
    outcome = CandidateOutcome(
        candidate_id=outcome.candidate_id, W_mm=outcome.W_mm,
        x_front_mm=outcome.x_front_mm,
        d_halo_mm=outcome.d_halo_mm, lifecycle_state=outcome.lifecycle_state,
        T_raw=outcome.T_raw, T_penalized=outcome.T_penalized,
        failure_reason=None, phi_snapshot_paths=snaps, record_path=record_path,
    )
    log = IterationLog(
        iteration=iteration, lifecycle_state=gate.lifecycle_state,
        T_raw=objective.T_raw, T_penalized=T_penalized,
        gradient_norm=grad_norm, failure_reason=None, record_path=record_path,
        D20=cfd.D20, total_mass_kg=mass_report.total_mass_kg,
    )
    return outcome, log, snaps


def _try_save_sensitivity(adjoint, iter_id: str, out_dir: str) -> Optional[str]:
    """Persist the adjoint surface sensitivity and the vertices it indexes.

    Same rule as _try_write_record: saving an artefact must never take the loop
    down. A failed save costs the post-hoc analysis, not the optimisation.
    """
    try:
        import numpy as _np
        sens = _np.asarray(adjoint.sensitivity)
        mesh = getattr(adjoint, "half_mesh", None)
        verts = None if mesh is None else _np.asarray(mesh.vertices)
        if verts is not None and len(verts) != len(sens):
            # Not fatal here -- update_phi validates this properly -- but the
            # saved pair would be meaningless, so say so rather than write it.
            warnings.warn(
                f"{iter_id}: {len(sens)} sensitivity values against "
                f"{len(verts)} vertices; not saving an unindexable field.",
                RuntimeWarning, stacklevel=2)
            return None
        path = os.path.join(out_dir, f"sens_{iter_id}.npz")
        _np.savez_compressed(
            path, sensitivity=sens.astype(_np.float32),
            vertices=(_np.empty((0, 3), dtype=_np.float32) if verts is None
                      else verts.astype(_np.float32)))
        return path
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"{iter_id}: could not save the adjoint sensitivity field ({exc}). "
            f"The optimisation continues; the field is lost for analysis.",
            RuntimeWarning, stacklevel=2)
        return None


def _try_write_record(bindings: PipelineBindings, outcome: CandidateOutcome,
                      extra: Optional[dict] = None) -> Optional[str]:
    """Record-writing must never take the loop down. A failed write is
    logged into the returned path slot as None; the loop continues."""
    payload = {
        "candidate_id": outcome.candidate_id,
        "W_mm": outcome.W_mm,
        "x_front_mm": outcome.x_front_mm,
        "d_halo_mm": outcome.d_halo_mm,
        "lifecycle_state": outcome.lifecycle_state,
        "T_raw": outcome.T_raw,
        "T_penalized": outcome.T_penalized,
        "failure_reason": outcome.failure_reason,
        "phi_grid_snapshot_paths": outcome.phi_snapshot_paths,
    }
    if extra:
        payload.update(extra)
    try:
        return bindings.write_candidate_record(payload)
    except Exception as exc:  # noqa: BLE001 — persistence failure != physics failure
        # Still non-fatal, but no longer SILENT. This swallow hid a record layer
        # that never worked at all: the payload carried x_front_mm, which
        # CandidateRecord did not accept, so every write raised TypeError and
        # every record_path came back None. Candidate records ARE the output --
        # ranking across d_halo values and merging shards both read them -- so a
        # sweep could have run for days and produced nothing, reporting success
        # throughout. Warn once per distinct cause.
        global _RECORD_WRITE_WARNED
        key = f"{type(exc).__name__}: {exc}"[:200]
        if key not in _RECORD_WRITE_WARNED:
            _RECORD_WRITE_WARNED.add(key)
            warnings.warn(
                f"candidate record write FAILED and was skipped: {key}. "
                f"The optimiser will keep running, but it is producing no "
                f"records -- nothing downstream can rank or merge these results.",
                RuntimeWarning, stacklevel=2,
            )
        return None
