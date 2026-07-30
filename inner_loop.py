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


# Penalty provider: maps a GateOutcome to Part 3 penalties. Default is the
# honest zero-with-provenance provider; a real provider reads curvature /
# accessibility penalty data from Part 1 once Part 1 emits it.
PenaltyProvider = Callable[[object], PenaltyInputs]


_ZERO_PENALTY_WARNED = set()


def zero_penalties(gate_outcome: object) -> PenaltyInputs:
    """Explicit zero penalties. Used when Part 1 has not yet emitted
    manufacturing-penalty magnitudes for repaired-but-penalized geometry.
    Deliberately a named function so the choice shows up in code review —
    do NOT let this silently become the permanent behavior; the spec's
    'Accessibility failure (large) → assign manufacturing penalty, continue'
    path needs a real magnitude eventually.

    It became the permanent behaviour. No caller has ever passed a different
    provider, so compose_penalized_time has only ever added 0.0 and every
    candidate in the 2026-07-29 sweep was logged "geometry_repaired" -- meaning
    the accessibility check DID find unreachable surface -- while paying nothing
    for it. That is free rein to evolve an unmanufacturable shape, and it costs
    more now that the optimiser actually carves.

    Still returns zero, because the right magnitude is a calibration decision
    and inventing a coefficient here would be worse than the gap: a made-up
    penalty silently reranks candidates and looks principled. But it warns once
    per candidate with the measured area attached, so the gap is visible in the
    log rather than inferred from source.
    """
    area = getattr(gate_outcome, "inaccessible_area_mm2", None)
    cid = getattr(gate_outcome, "stl_path", None) or "?"
    if area and cid not in _ZERO_PENALTY_WARNED:
        _ZERO_PENALTY_WARNED.add(cid)
        warnings.warn(
            f"{area:.1f} mm^2 of this candidate's surface is unreachable by the "
            f"cutter, and the manufacturing penalty applied for it is 0.0 s. No "
            f"caller passes a penalty_provider, so unmanufacturable geometry "
            f"ranks exactly as well as manufacturable geometry. Set one before "
            f"trusting a final ranking.",
            RuntimeWarning, stacklevel=2)
    return PenaltyInputs(manufacturing_penalty_s=0.0, rule_margin_penalty_s=0.0)


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
    penalty_provider: PenaltyProvider = zero_penalties,
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

    penalties = penalty_provider(gate)
    T_penalized = compose_penalized_time(objective.T_com_penalized, penalties)
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
        bindings.update_phi(
            phi_grids, adjoint.sensitivity, adjoint.half_mesh, config.hj_dt,
            gradient_weights, objective.gradients, mass_report,
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

    outcome = CandidateOutcome(
        candidate_id=iter_id, W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
        lifecycle_state=gate.lifecycle_state,
        T_raw=objective.T_raw, T_penalized=T_penalized,
        failure_reason=None, phi_snapshot_paths=snaps,
    )
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
