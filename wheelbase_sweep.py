"""
wheelbase_sweep.py — Part 3 Stage 10: outer wheelbase sweep.

Spec ("Outer Loop: Wheelbase Sweep" + "Search Strategy"):

    W ∈ [120, 140] mm; coarse sweep 1 mm steps (21 values);
    refined sweep 0.5 mm steps near top candidates.
    For each W: rerun inner φ optimization (M candidates, evolutionary
    selection between them); save best valid candidate and its φ fields.
    Warm-starting: converged φ fields from W=N initialize W=N+1.

The per-W population logic:
    - M candidates seeded (fresh random φ, except candidate 0 which is
      warm-started from the previous W when available).
    - All M inner loops run (in parallel if configured).
    - Between evolutionary rounds: rank by T_penalized, kill bottom 50%,
      perturb survivors' φ fields to refill the population.
    - The number of evolutionary rounds is config-driven; each round runs
      each candidate's inner loop for up to `evolution_interval_iters`
      iterations (a budget slice), matching the spec's "after every N inner
      iterations" cadence without cross-thread synchronization complexity.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

from evolutionary import FailureRegionMemory, plan_evolution_step
from inner_loop import run_inner_loop, InnerLoopResult
from objective_policy import final_ranking
from optimizer_contract import (
    COARSE_W_STEP_MM,
    D_HALO_MIN_MM,
    D_HALO_PLACEMENT_MARGIN_MM,
    REFINED_W_STEP_MM,
    W_MAX_MM,
    W_MIN_MM,
    CandidateOutcome,
    GradientWeights,
    OptimizerConfig,
    validate_W,
    validate_x_front,
)
from parallel_runner import TaskFailure, run_candidates_parallel
from pipeline_interface import PipelineBindings


def coarse_w_values() -> list[float]:
    """120..140 mm inclusive at 1 mm steps — exactly 21 values."""
    n = int(round((W_MAX_MM - W_MIN_MM) / COARSE_W_STEP_MM)) + 1
    return [W_MIN_MM + i * COARSE_W_STEP_MM for i in range(n)]


def d_halo_values(W_mm: float, n: int = 6) -> list[float]:
    """`n` legal halo positions for a fixed wheelbase: [0, W-34) mm, exclusive.

    This is the Stage-2 sweep variable in the two-stage architecture
    (ARCHITECTURE.md §4): W and x_front are frozen by Stage 1's no-CFD Bayesian
    search, and each d_halo becomes a SEPARATE CAR run through the full inner
    φ loop, ranked by real race time. d_halo's payoff is aerodynamic (the
    halo→canister loft), which is exactly why it needs CFD per value and W
    does not.

    The upper bound is strict (`< W-34`), so the top sample is pulled just
    inside it rather than sitting on the excluded endpoint.
    """
    validate_W(W_mm)
    if n < 1:
        raise ValueError("n must be >= 1")
    upper = W_mm - D_HALO_PLACEMENT_MARGIN_MM
    if upper <= D_HALO_MIN_MM:
        raise ValueError(
            f"W={W_mm} mm leaves no legal d_halo range (upper bound {upper} mm)"
        )
    if n == 1:
        return [D_HALO_MIN_MM]
    span = upper - D_HALO_MIN_MM
    # Last sample at 99% of the span keeps it strictly below the exclusive bound.
    return [round(D_HALO_MIN_MM + span * 0.99 * i / (n - 1), 6) for i in range(n)]


def refined_d_halo_values(top_d_halos: list[float], W_mm: float,
                          window_mm: float = 4.0, n_per: int = 3) -> list[float]:
    """Refinement grid around the best d_halo values, clamped to [0, W-34)."""
    validate_W(W_mm)
    if not top_d_halos:
        raise ValueError("top_d_halos must not be empty")
    upper = W_mm - D_HALO_PLACEMENT_MARGIN_MM
    values = set()
    for d in top_d_halos:
        for k in range(-(n_per // 2), n_per // 2 + 1):
            candidate = d + k * (window_mm / max(n_per - 1, 1))
            if D_HALO_MIN_MM <= candidate < upper:
                values.add(round(candidate, 6))
    return sorted(values)


def refined_w_values(top_ws: list[float], step_mm: float = REFINED_W_STEP_MM) -> list[float]:
    """0.5 mm grid around each top W (±1 mm window), deduplicated, clamped
    to the legal range, ascending."""
    if not top_ws:
        raise ValueError("top_ws must not be empty")
    values = set()
    for w in top_ws:
        validate_W(w)
        for k in (-2, -1, 0, 1, 2):
            candidate = w + k * step_mm
            if W_MIN_MM <= candidate <= W_MAX_MM:
                values.add(round(candidate, 6))
    return sorted(values)


@dataclass
class WResult:
    """Outcome of the full multi-candidate optimization at one wheelbase."""

    W_mm: float
    best: Optional[CandidateOutcome]
    best_phi_grids: Optional[dict]
    all_outcomes: list = field(default_factory=list)
    task_failures: list = field(default_factory=list)


def optimize_single_w(
    bindings: PipelineBindings,
    config: OptimizerConfig,
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    n_candidates: int,
    out_dir: str,
    gradient_weights: GradientWeights,
    warm_start_grids: Optional[dict] = None,
    failure_memory: Optional[FailureRegionMemory] = None,
    n_evolution_rounds: int = 3,
    candidate_prefix: str = "cand",
) -> WResult:
    """Run the M-candidate evolutionary inner optimization at one W."""
    validate_W(W_mm)
    validate_x_front(x_front_mm, W_mm)
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    if n_evolution_rounds < 1:
        raise ValueError("n_evolution_rounds must be >= 1")

    # Seed the population. Candidate 0 warm-starts when fields are supplied.
    population: list[dict] = []
    for i in range(n_candidates):
        if i == 0 and warm_start_grids is not None:
            grids = bindings.warm_start_phi_fields(warm_start_grids, W_mm, x_front_mm, d_halo_mm)
        else:
            grids = bindings.initialize_phi_fields(
                W_mm, x_front_mm, d_halo_mm, seed=config.random_seed + i
            )
        population.append(grids)

    round_config = OptimizerConfig(
        rtc_validated_against_track_data=config.rtc_validated_against_track_data,
        cfd_pipeline_validated_on_known_geometry=config.cfd_pipeline_validated_on_known_geometry,
        mu=config.mu,
        wheel_moi_kg_m2=config.wheel_moi_kg_m2,
        # Per-round slice = min(evolution_interval_iters, iteration_budget).
        # It used to be evolution_interval_iters unconditionally, which silently
        # discarded config.iteration_budget — so run_optimization.py's
        # --iteration-budget flag did nothing at all, and a caller asking for a
        # 1-iteration smoke run still got 10.
        iteration_budget=min(config.evolution_interval_iters, config.iteration_budget),
        gradient_norm_threshold=config.gradient_norm_threshold,
        evolution_interval_iters=config.evolution_interval_iters,
        hj_dt=config.hj_dt,
        require_cfd_convergence=config.require_cfd_convergence,
        random_seed=config.random_seed,
        max_workers=config.max_workers,
    )

    all_outcomes: list[CandidateOutcome] = []
    task_failures: list[TaskFailure] = []
    latest: list[Optional[InnerLoopResult]] = [None] * n_candidates

    for round_idx in range(n_evolution_rounds):
        tasks = []
        for i, grids in enumerate(population):
            cid = f"{candidate_prefix}_W{W_mm:g}_c{i}_r{round_idx}"
            tasks.append(
                functools.partial(
                    run_inner_loop,
                    bindings=bindings,
                    config=round_config,
                    candidate_id=cid,
                    W_mm=W_mm,
                    x_front_mm=x_front_mm,
                    d_halo_mm=d_halo_mm,
                    initial_phi_grids=grids,
                    out_dir=out_dir,
                    gradient_weights=gradient_weights,
                )
            )
        results = run_candidates_parallel(tasks, max_workers=config.max_workers)

        round_outcomes: list[CandidateOutcome] = []
        for i, result in enumerate(results):
            if isinstance(result, TaskFailure):
                task_failures.append(result)
                # PRINT IT. parallel_runner captures a full traceback into
                # TaskFailure.traceback_text and, before this, nothing in the
                # codebase ever read that field -- the failure was appended to a
                # list, carried to WResult, and never surfaced. A candidate
                # could die mid-sweep with a complete stack trace in hand and
                # the only evidence in the log was a missing record file.
                #
                # That is exactly what happened on 2026-07-30: d_halo=16 stopped
                # after 3 of 6 iterations, iteration 3 finished its CFD, adjoint
                # and phi update and then wrote no record, and there was nothing
                # anywhere saying why. Both record-writing paths in
                # _run_single_iteration were accounted for, so the exception had
                # to have escaped the function entirely -- which is precisely
                # the case this swallows.
                print(f"[sweep] CANDIDATE FAILED: task {result.task_index} "
                      f"({candidate_prefix}_W{W_mm:g}_c{result.task_index}"
                      f"_r{round_idx}): {result.message}\n{result.traceback_text}",
                      flush=True)
                continue
            latest[i] = result
            if result.best is not None:
                round_outcomes.append(result.best)
                all_outcomes.append(result.best)
            else:
                # Whole inner slice failed; feed the failure memory from the
                # last logged failure.
                last = result.history[-1] if result.history else None
                if last is not None and failure_memory is not None:
                    outcome = CandidateOutcome(
                        candidate_id=f"{result.candidate_id}_dead",
                        W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
                        lifecycle_state=last.lifecycle_state,
                        T_raw=None, T_penalized=None,
                        failure_reason=last.failure_reason or "inner loop produced no success",
                    )
                    failure_memory.record_failure(outcome)

        if failure_memory is not None:
            for outcome in round_outcomes:
                failure_memory.record_failure(outcome)  # ignores successes

        # Evolutionary selection between rounds (skip after the last round).
        if round_idx == n_evolution_rounds - 1:
            break
        if round_outcomes:
            plan = plan_evolution_step(round_outcomes)
            survivor_ids = {c.candidate_id for c in plan.survivors}
            # Map survivor outcomes back to their population slots.
            survivor_slots = [
                i for i, r in enumerate(latest)
                if r is not None and r.best is not None
                and r.best.candidate_id in survivor_ids
            ]
            if survivor_slots:
                new_population: list[dict] = []
                for i in range(n_candidates):
                    if i in survivor_slots:
                        new_population.append(latest[i].final_phi_grids)
                    else:
                        src = survivor_slots[i % len(survivor_slots)]
                        perturbed = bindings.perturb_phi_fields(
                            latest[src].final_phi_grids,
                            seed=config.random_seed + 1000 * round_idx + i,
                            amplitude=0.10,
                        )
                        new_population.append(perturbed)
                population = new_population
        # If nothing succeeded this round, reseed everything fresh.
        else:
            population = [
                bindings.initialize_phi_fields(
                    W_mm, x_front_mm, d_halo_mm,
                    seed=config.random_seed + 5000 * round_idx + i,
                )
                for i in range(n_candidates)
            ]

    ranked = final_ranking(all_outcomes)
    best = ranked[0] if ranked else None
    best_grids = None
    if best is not None:
        for r in latest:
            if r is not None and r.best is not None and r.best.candidate_id == best.candidate_id:
                best_grids = r.final_phi_grids
                break
        if best_grids is None:
            # Best came from an earlier round whose grids were since evolved;
            # fall back to the best-ranked latest grids.
            live = [r for r in latest if r is not None and r.final_phi_grids is not None]
            best_grids = live[0].final_phi_grids if live else None

    return WResult(
        W_mm=W_mm, best=best, best_phi_grids=best_grids,
        all_outcomes=all_outcomes, task_failures=task_failures,
    )


def run_wheelbase_sweep(
    bindings: PipelineBindings,
    config: OptimizerConfig,
    w_values: list[float],
    x_front_mm: float,
    d_halo_mm: float,
    n_candidates: int,
    out_dir: str,
    gradient_weights: GradientWeights,
    failure_memory: Optional[FailureRegionMemory] = None,
    n_evolution_rounds: int = 3,
) -> list[WResult]:
    """Sweep the given W values in ascending order with warm-starting.

    The converged φ fields from each W seed candidate 0 of the next W (spec:
    "the converged φ fields from W=N are used to initialize W=N+1"). If a W
    produced no valid candidate, the next W starts fresh.

    x_front_mm and d_halo_mm are fixed for the whole sweep (Level 1 —
    Part 1's bayesian_outer_search.py — proposes the (W, x_front, d_halo)
    triple; this sweep executes the W dimension at the proposed x_front/d_halo).
    """
    if not w_values:
        raise ValueError("w_values must not be empty")
    results: list[WResult] = []
    warm: Optional[dict] = None
    for w in sorted(w_values):
        # The only READ of the failure memory. Everything above records into it
        # and nothing consulted it, so the refined sweep -- which runs at 0.5 mm
        # spacing around the top W values, i.e. well inside the 1.0 mm failure
        # radius the coarse sweep just populated -- happily re-spent a full CFD
        # budget on wheelbases that had already died three times.
        if failure_memory is not None and failure_memory.is_blacklisted(
            w, x_front_mm, d_halo_mm
        ):
            reasons = {r.failure_reason for r
                       in failure_memory.failures_near(w, x_front_mm, d_halo_mm)}
            print(f"[sweep] skipping W={w} mm: blacklisted after "
                  f"{len(failure_memory.failures_near(w, x_front_mm, d_halo_mm))} "
                  f"nearby failures ({'; '.join(sorted(filter(None, reasons)))})")
            continue
        result = optimize_single_w(
            bindings, config, w, x_front_mm, d_halo_mm, n_candidates, out_dir,
            gradient_weights, warm_start_grids=warm,
            failure_memory=failure_memory,
            n_evolution_rounds=n_evolution_rounds,
        )
        results.append(result)
        warm = result.best_phi_grids if result.best is not None else None
    return results


def run_d_halo_sweep(
    bindings: PipelineBindings,
    config: OptimizerConfig,
    d_halo_list: list[float],
    W_mm: float,
    x_front_mm: float,
    n_candidates: int,
    out_dir: str,
    gradient_weights: GradientWeights,
    failure_memory: Optional[FailureRegionMemory] = None,
    n_evolution_rounds: int = 3,
) -> list[WResult]:
    """Stage 2 of the two-stage architecture: sweep d_halo at FIXED W/x_front.

    ARCHITECTURE.md §4: "a separate car for each halo-canister distance, best
    race time wins". W and x_front come from Stage 1's no-CFD Bayesian search
    (part1-simulation/stage1_search.py) and are never reopened here.

    Deliberately reuses `optimize_single_w` unchanged — that function was
    already parameterised by all three scalars and only ever varied one, so the
    difference between the two sweeps is which one the outer loop steps. The
    returned WResult still carries `W_mm` (constant across this sweep); read
    `best.d_halo_mm` for the swept variable.

    Warm-starting carries φ from one halo position to the next, same as the W
    sweep — adjacent d_halo cars differ only in where the pocket sits.
    """
    if not d_halo_list:
        raise ValueError("d_halo_list must not be empty")
    validate_W(W_mm)
    validate_x_front(x_front_mm, W_mm)
    results: list[WResult] = []
    warm: Optional[dict] = None
    for d in sorted(d_halo_list):
        if failure_memory is not None and failure_memory.is_blacklisted(
            W_mm, x_front_mm, d
        ):
            near = failure_memory.failures_near(W_mm, x_front_mm, d)
            reasons = {r.failure_reason for r in near}
            print(f"[sweep] skipping d_halo={d} mm: blacklisted after "
                  f"{len(near)} nearby failures "
                  f"({'; '.join(sorted(filter(None, reasons)))})")
            continue
        result = optimize_single_w(
            bindings, config, W_mm, x_front_mm, d, n_candidates, out_dir,
            gradient_weights, warm_start_grids=warm,
            failure_memory=failure_memory,
            n_evolution_rounds=n_evolution_rounds,
            candidate_prefix=f"dhalo{d:g}",
        )
        results.append(result)
        warm = result.best_phi_grids if result.best is not None else None
    return results
