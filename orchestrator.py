"""
orchestrator.py — Part 3 Stage 12: full search strategy + final deliverables.

Spec ("Search Strategy", 03_optimizer_workflow):

    1. Validate RTC against track data                (prerequisite flag)
    2. Validate CFD pipeline on a known geometry      (prerequisite flag)
    3. Coarse W sweep: 1 mm steps, M=5 candidates/W, moderate budget
    4. Rank wheelbases by best valid T_raw
    5. Refined sweep: 0.5 mm around top 5 W values, M=10
    6. Robustness checks on the finalists
    7. Select build candidate
    8. Emit final deliverables

Steps 1-2 are experiments, not code — the orchestrator REFUSES to run
unless the config asserts both flags True. This is the loud gate the spec
demands; flipping the flags without doing the experiments is a human lie
the code cannot detect, but it will at least be a deliberate one.

Final deliverables (spec "Final Deliverables" list):
    optimal W, optimal d_halo, four converged φ fields (snapshot paths),
    full-car STL, predicted T_raw, robustness report, candidate records
    directory, backup ranking (2nd/3rd best).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from evolutionary import FailureRegionMemory
from objective_policy import final_ranking, select_build_candidate
from optimizer_contract import (
    CandidateOutcome,
    GradientWeights,
    OptimizerConfig,
)
from pipeline_interface import PipelineBindings, validate_bindings
from robustness import RobustnessReport
from wheelbase_sweep import (
    WResult,
    coarse_w_values,
    d_halo_values,
    refined_d_halo_values,
    refined_w_values,
    run_d_halo_sweep,
    run_wheelbase_sweep,
)


class PrerequisitesNotMet(RuntimeError):
    """Raised when the search is started before RTC/CFD validation."""


@dataclass
class SearchResult:
    build_candidate: Optional[CandidateOutcome]
    backup_ranking: list                     # 2nd, 3rd best CandidateOutcome
    coarse_results: list = field(default_factory=list)   # list[WResult]
    refined_results: list = field(default_factory=list)  # list[WResult]
    robustness_reports: list = field(default_factory=list)
    failure_memory: Optional[FailureRegionMemory] = None

    def final_deliverables(self, out_dir: str) -> dict:
        """Assemble the spec's final-deliverables dict. Raises RuntimeError
        if there is no build candidate — an empty search must fail loudly,
        never emit a deliverables report with holes."""
        if self.build_candidate is None:
            raise RuntimeError(
                "no build candidate: every candidate in the search failed. "
                "Inspect the candidate records and failure memory before "
                "re-running; do NOT ship a failure as a deliverable."
            )
        c = self.build_candidate
        return {
            "optimal_W_mm": c.W_mm,
            "optimal_x_front_mm": c.x_front_mm,
            "optimal_d_halo_mm": c.d_halo_mm,
            "converged_phi_snapshot_paths": dict(c.phi_snapshot_paths),
            "predicted_T_raw_s": c.T_raw,
            "predicted_T_penalized_s": c.T_penalized,
            "candidate_record_path": c.record_path,
            "candidate_records_dir": out_dir,
            "robustness_reports": [r.summary() for r in self.robustness_reports],
            "backup_ranking": [
                {"candidate_id": b.candidate_id, "W_mm": b.W_mm,
                 "T_raw_s": b.T_raw, "record_path": b.record_path}
                for b in self.backup_ranking
            ],
        }


def _require_prerequisites(config: OptimizerConfig) -> None:
    """The spec's steps 1-2 gate. Shared by both search entry points so a new
    entry point cannot accidentally skip it."""
    if not config.rtc_validated_against_track_data:
        raise PrerequisitesNotMet(
            "Search Strategy step 1 unmet: RTC has not been validated "
            "against real track data (config.rtc_validated_against_track_data "
            "is False). Run the physical validation first."
        )
    if not config.cfd_pipeline_validated_on_known_geometry:
        raise PrerequisitesNotMet(
            "Search Strategy step 2 unmet: CFD pipeline has not been "
            "validated on a known geometry "
            "(config.cfd_pipeline_validated_on_known_geometry is False)."
        )


def run_stage2_dhalo_search(
    bindings: PipelineBindings,
    config: OptimizerConfig,
    W_mm: float,
    x_front_mm: float,
    out_dir: str,
    gradient_weights: GradientWeights,
    n_d_halo: int = 6,
    robustness_runner: Optional[Callable[[CandidateOutcome], RobustnessReport]] = None,
    n_evolution_rounds: int = 3,
    n_finalists_for_robustness: int = 3,
    refine: bool = True,
) -> SearchResult:
    """Stage 2 of the two-stage architecture: sweep d_halo at FIXED W/x_front.

    This is the search `ARCHITECTURE.md` §4 actually describes, and the one
    `run_full_search` (below) is NOT: that one sweeps W, a leftover from before
    the two-stage split, and re-decides with expensive CFD a scalar Stage 1
    already chose cheaply from mass/COM. Use this for the two-stage flow; W and
    x_front arrive frozen from `part1-simulation/stage1_search.py`.

    Cost note, because it is easy to launch by accident: each d_halo value runs
    `n_candidates x n_evolution_rounds x min(evolution_interval_iters,
    iteration_budget)` inner iterations, and EVERY inner iteration is one
    forward CFD plus one adjoint. With the defaults (6 halos x 5 candidates x
    3 rounds x 10 iters) that is 900 solve-pairs. Size it deliberately.
    """
    _require_prerequisites(config)
    validate_bindings(bindings)

    failure_memory = FailureRegionMemory()

    coarse = run_d_halo_sweep(
        bindings, config, d_halo_values(W_mm, n_d_halo), W_mm, x_front_mm,
        n_candidates=config.coarse_candidates_per_w,
        out_dir=out_dir, gradient_weights=gradient_weights,
        failure_memory=failure_memory, n_evolution_rounds=n_evolution_rounds,
    )

    scored = sorted(
        (r.best.T_raw, r.best.d_halo_mm) for r in coarse
        if r.best is not None and r.best.T_raw is not None
    )
    top_d = [d for _, d in scored[: config.top_w_count]]

    refined: list[WResult] = []
    if refine and top_d:
        refined = run_d_halo_sweep(
            bindings, config, refined_d_halo_values(top_d, W_mm), W_mm, x_front_mm,
            n_candidates=config.refined_candidates_per_w,
            out_dir=out_dir, gradient_weights=gradient_weights,
            failure_memory=failure_memory, n_evolution_rounds=n_evolution_rounds,
        )

    pool: list[CandidateOutcome] = []
    for r in coarse + refined:
        pool.extend(o for o in r.all_outcomes if o.is_fully_valid)

    ranked = final_ranking(pool)
    build = select_build_candidate(pool)

    reports: list[RobustnessReport] = []
    if robustness_runner is not None:
        for finalist in ranked[:n_finalists_for_robustness]:
            reports.append(robustness_runner(finalist))

    return SearchResult(
        build_candidate=build,
        backup_ranking=list(ranked[1:3]),
        coarse_results=coarse,
        refined_results=refined,
        robustness_reports=reports,
        failure_memory=failure_memory,
    )


def run_full_search(
    bindings: PipelineBindings,
    config: OptimizerConfig,
    x_front_mm: float,
    d_halo_mm: float,
    out_dir: str,
    gradient_weights: GradientWeights,
    robustness_runner: Optional[Callable[[CandidateOutcome], RobustnessReport]] = None,
    n_evolution_rounds: int = 3,
    n_finalists_for_robustness: int = 3,
) -> SearchResult:
    """Run the complete Part 3 search (steps 3-8).

    Args:
        bindings: pipeline handshake.
        config: optimizer config; BOTH prerequisite flags must be True.
        x_front_mm: nose length / front axle position, fixed for this search
            (validated per-W inside the loops). Proposed by Level 1
            (Part 1's bayesian_outer_search.py) in the three-level structure.
        d_halo_mm: halo distance (validated per-W inside the loops).
        out_dir: candidate records root.
        gradient_weights: from gradient_combiner.calibrate_gradient_weights.
        robustness_runner: optional callable mapping a finalist outcome to a
            RobustnessReport (wraps robustness.run_robustness_checks with the
            real RTC + hooks). None → robustness step skipped LOUDLY: the
            SearchResult carries no reports and final_deliverables will show
            an empty list, which reviewers must treat as "not done".
        n_evolution_rounds: evolutionary rounds per W.
        n_finalists_for_robustness: how many top candidates get robustness
            checks.

    Raises:
        PrerequisitesNotMet before touching any pipeline code if the two
        validation flags are not both True.
    """
    _require_prerequisites(config)
    validate_bindings(bindings)

    failure_memory = FailureRegionMemory()

    # Step 3: coarse sweep.
    coarse = run_wheelbase_sweep(
        bindings, config, coarse_w_values(), x_front_mm, d_halo_mm,
        n_candidates=config.coarse_candidates_per_w,
        out_dir=out_dir, gradient_weights=gradient_weights,
        failure_memory=failure_memory, n_evolution_rounds=n_evolution_rounds,
    )

    # Step 4: rank wheelbases by best valid T_raw.
    scored: list[tuple[float, float]] = []  # (T_raw, W)
    for r in coarse:
        if r.best is not None and r.best.T_raw is not None:
            scored.append((r.best.T_raw, r.W_mm))
    scored.sort()
    top_ws = [w for _, w in scored[: config.top_w_count]]

    refined: list[WResult] = []
    if top_ws:
        # Step 5: refined sweep at 0.5 mm around the top W values.
        refined = run_wheelbase_sweep(
            bindings, config, refined_w_values(top_ws), x_front_mm, d_halo_mm,
            n_candidates=config.refined_candidates_per_w,
            out_dir=out_dir, gradient_weights=gradient_weights,
            failure_memory=failure_memory, n_evolution_rounds=n_evolution_rounds,
        )

    # Pool every successful outcome from both phases for final ranking.
    pool: list[CandidateOutcome] = []
    for r in coarse + refined:
        pool.extend(o for o in r.all_outcomes if o.is_fully_valid)

    ranked = final_ranking(pool)
    build = select_build_candidate(pool)
    backups = ranked[1:3]

    # Step 6: robustness on the finalists.
    reports: list[RobustnessReport] = []
    if robustness_runner is not None:
        for finalist in ranked[:n_finalists_for_robustness]:
            reports.append(robustness_runner(finalist))

    return SearchResult(
        build_candidate=build,
        backup_ranking=list(backups),
        coarse_results=coarse,
        refined_results=refined,
        robustness_reports=reports,
        failure_memory=failure_memory,
    )
