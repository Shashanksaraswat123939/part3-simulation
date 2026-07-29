"""
optimizer_contract.py — Part 3 Stage 1: shared optimizer constants and types.

Every other Part 3 module imports FROM this module and never re-derives
sweep ranges, convergence thresholds, lifecycle states, or penalty policy
locally. Mirrors the role physics_contract.py plays in Part 2 and
geometry_contract.py plays in Part 1.

Coordinate/unit conventions are inherited from Part 1/Part 2 and are NOT
redefined here:
    x = front to rear, y = centerline to outside, z = track upward
    SI internally (kg, m, N, s); W and d_halo are expressed in mm at this
    layer because the governing spec defines the outer-loop variables in mm.

Sources (verbatim from 03_optimizer_workflow spec):
    W ∈ [120, 140] mm, coarse sweep 1 mm steps (21 values),
    refined sweep 0.5 mm steps near top candidates,
    d_halo ∈ [0, W + 16] mm,
    inner-loop stop when |ΔT_penalized| < 1 ms OR gradient norm < threshold
    OR 3+ consecutive gate failures OR iteration budget exhausted,
    evolutionary step: kill bottom 50%, perturb top 50%,
    coarse: M=5 candidates per W; refined: M=10; top 5 wheelbase values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Outer-loop variable ranges (spec: "Optimization Variables")
# ---------------------------------------------------------------------------

W_MIN_MM: float = 120.0
W_MAX_MM: float = 140.0
COARSE_W_STEP_MM: float = 1.0
REFINED_W_STEP_MM: float = 0.5

# x_front bounds are W-dependent. Mirrors Part 1's geometry_contract.
# calibrate_x_front_bounds exactly (cross-check test verifies this) — added
# as a real outer-loop variable (was previously absent from Part 3 entirely,
# audit finding P3-2: "zero occurrences of x_front anywhere in
# part3-simulation" even though Part 1's geometry cannot be built without
# it). x_front is fixed per sweep run, threaded the same way d_halo_mm
# already is, not swept independently within Part 3 — Level 1 (Part 1's
# bayesian_outer_search.py) proposes (W, x_front, d_halo) triples; Part 3
# executes at the proposed values.
#
# Values updated 2026-07-20 in lockstep with Part 1. The old pair (61.0, 90.0)
# derived its floor from "the nose must fit the CO2 cartridge depth", which is
# false — the cartridge chamber is rear of Ref Plane A, not in the nose. Worse,
# that floor forced a nose overhang of >= 45 mm against T8.2's 40 mm maximum,
# so every candidate Part 3 accepted was illegal. The ceiling is now T8.2
# directly: nose overhang = x_front - 16 <= 40  ->  x_front <= 56.
# See geometry_contract.X_FRONT_MIN_MM for the full derivation and regs cites.
X_FRONT_MIN_MM: float = 36.0   # 16 mm Ref Plane A offset + 20 mm design min nose
X_FRONT_ABS_MAX_MM: float = 56.0   # 16 mm Ref Plane A offset + T8.2's 40 mm max


def calibrate_x_front_bounds(W_mm: float) -> tuple[float, float]:
    """Return (x_front_min_mm, x_front_max_mm) for the given wheelbase.
    Mirrors geometry_contract.calibrate_x_front_bounds byte-for-byte."""
    x_min = X_FRONT_MIN_MM
    x_max = min(X_FRONT_ABS_MAX_MM, 207.0 - W_mm)
    x_max = max(x_max, x_min + 1.0)
    return x_min, x_max


# Forward-most halo travel: pocket FRONT edge on the front axle line. Ref Plane
# A sits 16 mm ahead of the axle and d_halo is measured from it, so this is
# exactly 16 mm and is independent of W. Was 0.0, which allowed the halo up to
# 16 mm AHEAD of the front axle. MUST mirror
# geometry_contract.D_HALO_MIN_MM (Part 1) -- test_optimizer_contract
# cross-checks when Part 1 is importable.
D_HALO_MIN_MM: float = 16.0
# d_halo upper bound is STRICTLY less than W - 34 mm, derived from the halo
# pocket placement constraint in Part 1 (halo pocket length 50mm, Ref Plane A
# offset 16mm → pocket rear = ref_A + d_halo + 50mm must stay before rear axle).
# This replaces the stale W+16 value which allowed candidates Part 1 rejects —
# see audit finding K-5.  Mirrors geometry_contract._D_HALO_PLACEMENT_MARGIN_MM.
D_HALO_PLACEMENT_MARGIN_MM: float = 34.0   # 50mm pocket - 16mm Ref_A offset

# ---------------------------------------------------------------------------
# Inner-loop convergence (spec: "Convergence Criteria")
# ---------------------------------------------------------------------------

INNER_CONVERGENCE_DELTA_T_S: float = 1e-3       # |ΔT_penalized| < 1 ms
# ...and it must hold for this many CONSECUTIVE iterations before the inner loop
# calls itself converged.
#
# 1 ms is far below what this pipeline can resolve: measured CFD drag noise is
# about +/-15 ms of race time, so a single sub-millisecond delta says nothing.
# A production run on 2026-07-28 stopped after ONE update on dT = 0.372 ms and
# reported converged=True -- convergence declared on noise. Three in a row is
# very unlikely to happen by chance and costs only a couple of iterations when
# the objective genuinely has flattened.
INNER_CONVERGENCE_CONSECUTIVE: int = 3
DEFAULT_GRADIENT_NORM_THRESHOLD: float = 1e-6
MAX_CONSECUTIVE_GATE_FAILURES: int = 3          # "3+ consecutive iterations"
DEFAULT_ITERATION_BUDGET: int = 100

# ---------------------------------------------------------------------------
# Evolutionary search (spec: "Evolutionary Outer Search" / "Search Strategy")
# ---------------------------------------------------------------------------

EVOLUTION_KILL_FRACTION: float = 0.5
COARSE_CANDIDATES_PER_W: int = 5                # M=5 at 1 mm steps
REFINED_CANDIDATES_PER_W: int = 10              # M=10 at 0.5 mm steps
TOP_W_COUNT: int = 5                            # top 5 wheelbase values

# ---------------------------------------------------------------------------
# Failure penalty used ONLY for evolutionary ranking of dead candidates.
# Deliberately large but finite; 1e15-style sentinels poison float sums and
# make NaN bugs indistinguishable from failures. Never enters T_raw ranking:
# dead candidates are excluded from final ranking entirely.
# ---------------------------------------------------------------------------

FAILURE_PENALTY_S: float = 1.0e6

# ---------------------------------------------------------------------------
# Candidate lifecycle — MUST stay byte-identical to Part 1's
# geometry_contract.ALLOWED_LIFECYCLE_STATES and Part 2's
# candidate_record.ALLOWED_LIFECYCLE_STATES. test_integration cross-checks
# this when those packages are importable.
# ---------------------------------------------------------------------------

ALLOWED_LIFECYCLE_STATES = frozenset(
    {
        "valid_simulated",
        "geometry_repaired",
        "geometry_rejected",
        "rule_rejected",
        "machining_rejected",
        "CFD_failed",
        "objective_failed",
        "converged",
    }
)

# Lifecycle states in which a candidate's CFD/objective numbers exist and are
# meaningful. Everything else carries a failure penalty in search ranking and
# is excluded from final ranking.
SUCCESS_STATES = frozenset({"valid_simulated", "geometry_repaired", "converged"})

# Lifecycle states eligible for FINAL ranking (spec "Objective" section:
# "T_raw only, among fully valid candidates"). geometry_repaired counts as
# valid — it passed the gates after repair; the spec's failure-recovery table
# treats repaired candidates as continuing normally.
FINAL_RANKING_STATES = frozenset({"valid_simulated", "geometry_repaired", "converged"})


def validate_W(W_mm: float) -> None:
    """Raise ValueError unless W ∈ [120, 140] mm. Mirrors Part 1 semantics."""
    if not (isinstance(W_mm, (int, float)) and math.isfinite(W_mm)):
        raise ValueError(f"W_mm must be a finite number, got {W_mm!r}")
    if not (W_MIN_MM <= W_mm <= W_MAX_MM):
        raise ValueError(f"W_mm={W_mm} outside legal range [{W_MIN_MM}, {W_MAX_MM}] mm")


def validate_x_front(x_front_mm: float, W_mm: float) -> None:
    """Raise ValueError if x_front is outside its W-dependent bounds.
    Mirrors Part 1's geometry_contract.validate_x_front."""
    validate_W(W_mm)
    if not (isinstance(x_front_mm, (int, float)) and math.isfinite(x_front_mm)):
        raise ValueError(f"x_front_mm must be a finite number, got {x_front_mm!r}")
    x_min, x_max = calibrate_x_front_bounds(W_mm)
    if not (x_min <= x_front_mm <= x_max):
        raise ValueError(
            f"x_front_mm={x_front_mm} outside legal range [{x_min}, {x_max}] mm "
            f"for W={W_mm} mm"
        )


def validate_d_halo(d_halo_mm: float, W_mm: float) -> None:
    """Raise ValueError unless d_halo ∈ [0, W-34) mm (strict upper bound).

    Upper bound mirrors Part 1's geometry_contract.validate_d_halo: the halo
    pocket rear edge must stay strictly before the rear axle (pocket is 50mm long,
    Ref Plane A is 16mm ahead of front axle, so d_halo < W - 34mm).
    """
    validate_W(W_mm)
    if not (isinstance(d_halo_mm, (int, float)) and math.isfinite(d_halo_mm)):
        raise ValueError(f"d_halo_mm must be a finite number, got {d_halo_mm!r}")
    upper = W_mm - D_HALO_PLACEMENT_MARGIN_MM  # strict exclusive bound
    if not (D_HALO_MIN_MM <= d_halo_mm < upper):
        raise ValueError(
            f"d_halo_mm={d_halo_mm} outside legal range "
            f"[{D_HALO_MIN_MM}, {upper:.1f}) mm for W={W_mm} mm "
            f"(upper bound is W-34 mm, placement-derived)"
        )


def _require_finite(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


@dataclass(frozen=True)
class GradientWeights:
    """Weights for the four normalized gradient terms (spec: Gradient
    Combination). All dimensionless. Non-negative; at least one positive.

    These are hyperparameters. The spec requires their initial values to be
    set by sensitivity analysis (see gradient_combiner.calibrate_gradient_weights),
    not guessed — construct via that function unless you have a reason.
    """

    w_aero: float
    w_mass: float
    w_com: float
    w_mfg: float

    def __post_init__(self) -> None:
        for name in ("w_aero", "w_mass", "w_com", "w_mfg"):
            value = getattr(self, name)
            _require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.w_aero == 0.0 and self.w_mass == 0.0 and self.w_com == 0.0 and self.w_mfg == 0.0:
            raise ValueError("at least one gradient weight must be positive")


@dataclass(frozen=True)
class PenaltyInputs:
    """Part 3-owned additive penalties (spec Objective section):

        T_penalized = T_raw + COM_penalty            (from Part 2)
                    + manufacturing_penalty          (this type)
                    + rule_margin_penalty            (this type)

    Both fields are REQUIRED with no defaults: the caller must consciously
    supply 0.0 when a penalty source has produced nothing, rather than a
    penalty silently defaulting to zero and hiding an unwired data source.
    Units: seconds. Must be finite and non-negative (penalties never speed
    a car up).
    """

    manufacturing_penalty_s: float
    rule_margin_penalty_s: float

    def __post_init__(self) -> None:
        for name in ("manufacturing_penalty_s", "rule_margin_penalty_s"):
            value = getattr(self, name)
            _require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    @property
    def total_s(self) -> float:
        return self.manufacturing_penalty_s + self.rule_margin_penalty_s


@dataclass(frozen=True)
class OptimizerConfig:
    """Top-level configuration for the full search.

    Prerequisite flags (spec Search Strategy steps 1-2): the orchestrator
    refuses to start the full search unless both are True. This is a loud,
    deliberate gate — see orchestrator.run_full_search.
    """

    rtc_validated_against_track_data: bool
    cfd_pipeline_validated_on_known_geometry: bool
    mu: float
    wheel_moi_kg_m2: float
    iteration_budget: int = DEFAULT_ITERATION_BUDGET
    gradient_norm_threshold: float = DEFAULT_GRADIENT_NORM_THRESHOLD
    coarse_candidates_per_w: int = COARSE_CANDIDATES_PER_W
    refined_candidates_per_w: int = REFINED_CANDIDATES_PER_W
    top_w_count: int = TOP_W_COUNT
    evolution_interval_iters: int = 10          # "after every N inner iterations"
    hj_dt: float = 0.5
    require_cfd_convergence: bool = True
    random_seed: int = 0
    max_workers: int = 1

    def __post_init__(self) -> None:
        _require_finite("mu", self.mu)
        if not (0.0 <= self.mu <= 1.0):
            raise ValueError(f"mu={self.mu} outside physically sane range [0, 1]")
        _require_finite("wheel_moi_kg_m2", self.wheel_moi_kg_m2)
        if self.wheel_moi_kg_m2 < 0:
            raise ValueError(f"wheel_moi_kg_m2 must be non-negative, got {self.wheel_moi_kg_m2}")
        if self.iteration_budget < 1:
            raise ValueError("iteration_budget must be >= 1")
        if self.gradient_norm_threshold <= 0:
            raise ValueError("gradient_norm_threshold must be > 0")
        if self.evolution_interval_iters < 1:
            raise ValueError("evolution_interval_iters must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if not (self.hj_dt > 0 and math.isfinite(self.hj_dt)):
            raise ValueError(f"hj_dt must be a positive finite number, got {self.hj_dt}")


@dataclass(frozen=True)
class CandidateOutcome:
    """One candidate's outcome for ranking/selection. A thin, JSON-friendly
    view of Part 2's CandidateRecord — Part 3 never re-invents that record,
    it stores the path to it.

    T_raw / T_penalized are None exactly when the candidate never reached a
    successful objective evaluation. lifecycle/T consistency is ENFORCED here
    (a gap in Part 2's CandidateRecord — see the Part 2 audit): a success
    state must carry both times, a failure state must carry a failure_reason.
    """

    candidate_id: str
    W_mm: float
    x_front_mm: float
    d_halo_mm: float
    lifecycle_state: str
    T_raw: Optional[float]
    T_penalized: Optional[float]
    failure_reason: Optional[str]
    phi_snapshot_paths: dict = field(default_factory=dict)
    record_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lifecycle_state not in ALLOWED_LIFECYCLE_STATES:
            raise ValueError(
                f"lifecycle_state={self.lifecycle_state!r} not in "
                f"{sorted(ALLOWED_LIFECYCLE_STATES)}"
            )
        if self.lifecycle_state in SUCCESS_STATES:
            if self.T_raw is None or self.T_penalized is None:
                raise ValueError(
                    f"candidate {self.candidate_id!r} in success state "
                    f"{self.lifecycle_state!r} must carry T_raw and T_penalized"
                )
            _require_finite("T_raw", self.T_raw)
            _require_finite("T_penalized", self.T_penalized)
            if self.T_penalized < self.T_raw - 1e-12:
                raise ValueError(
                    f"T_penalized ({self.T_penalized}) < T_raw ({self.T_raw}); "
                    "penalties are additive and non-negative, this is a bug upstream"
                )
        else:
            if not self.failure_reason:
                raise ValueError(
                    f"candidate {self.candidate_id!r} in failure state "
                    f"{self.lifecycle_state!r} must carry a failure_reason"
                )

    @property
    def is_fully_valid(self) -> bool:
        return self.lifecycle_state in FINAL_RANKING_STATES

    def ranking_time_s(self) -> float:
        """Time used for EVOLUTIONARY/search ranking (T_penalized with
        failure penalty for dead candidates). NEVER use for final ranking."""
        if self.T_penalized is not None:
            return self.T_penalized
        return FAILURE_PENALTY_S
