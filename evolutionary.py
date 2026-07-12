"""
evolutionary.py — Part 3 Stage 8: evolutionary outer search primitives.

Spec ("Evolutionary Outer Search"):

    1. Initialize M φ fields per wheelbase (random within legal volumes)
    2. Run M inner loops in parallel
    3. After every N inner iterations:
         rank all M candidates by T_penalized
         kill bottom 50%
         perturb top 50% → M new candidates
    4. Repeat until outer convergence

    Perturbation: add smooth random noise to the converged φ field of a
    top candidate. Enough to escape the local valley, not enough to destroy
    the good geometry.

Plus the failure-history rule (Part 1 spec Rule 6 / Part 2 artifacts):
failed candidates keep their φ snapshots; the outer loop uses failure
history to avoid re-exploring regions that consistently produce bad
geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from objective_policy import search_ranking
from optimizer_contract import EVOLUTION_KILL_FRACTION, CandidateOutcome


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_survivors(
    candidates: Sequence[CandidateOutcome],
    kill_fraction: float = EVOLUTION_KILL_FRACTION,
) -> tuple[list[CandidateOutcome], list[CandidateOutcome]]:
    """Rank by T_penalized (failure penalty for dead candidates), keep the
    top (1 − kill_fraction), kill the rest.

    Returns (survivors, killed), both in rank order. With an odd population,
    the extra candidate SURVIVES (ceil) — killing more than the spec's 50%
    would shrink diversity faster than specified.

    Raises ValueError for an empty population or kill_fraction outside (0,1).
    """
    if not candidates:
        raise ValueError("cannot select survivors from an empty population")
    if not (0.0 < kill_fraction < 1.0):
        raise ValueError(f"kill_fraction must be in (0, 1), got {kill_fraction}")
    ranked = search_ranking(candidates)
    n_keep = math.ceil(len(ranked) * (1.0 - kill_fraction))
    n_keep = max(1, n_keep)
    return list(ranked[:n_keep]), list(ranked[n_keep:])


# ---------------------------------------------------------------------------
# Smooth φ perturbation
# ---------------------------------------------------------------------------


def _box_blur_3d(a: np.ndarray, passes: int) -> np.ndarray:
    """Cheap separable 3-point box blur, repeated. Repeated box blurs
    approach a Gaussian (central limit theorem) — smooth noise without a
    scipy dependency. Edges use reflected padding so smoothing doesn't
    drain amplitude at the boundary."""
    out = a
    for _ in range(passes):
        for axis in range(3):
            padded = np.concatenate(
                [np.take(out, [0], axis=axis), out, np.take(out, [-1], axis=axis)],
                axis=axis,
            )
            n = out.shape[axis]
            sl = [slice(None)] * 3
            sl_lo, sl_mid, sl_hi = list(sl), list(sl), list(sl)
            sl_lo[axis] = slice(0, n)
            sl_mid[axis] = slice(1, n + 1)
            sl_hi[axis] = slice(2, n + 2)
            out = (padded[tuple(sl_lo)] + padded[tuple(sl_mid)] + padded[tuple(sl_hi)]) / 3.0
    return out


def perturb_phi_array(
    phi: np.ndarray,
    seed: int,
    amplitude: float = 0.10,
    smoothing_passes: int = 4,
) -> np.ndarray:
    """Add smooth random noise to a φ array (spec: 'enough to escape the
    local valley, not enough to destroy the good geometry').

    amplitude is RELATIVE to the φ field's RMS: noise_rms = amplitude × φ_rms.
    Default 10% — a gentle nudge. The noise is white Gaussian passed through
    repeated box blurs, so its spatial scale is several cells: it moves the
    zero level set locally without introducing sub-resolution speckle that
    the 3.15 mm minimum-radius gate would immediately reject.

    Returns a NEW array (same dtype); the caller is responsible for
    re-applying hard constraints on the owning PhiGrid afterwards — hard
    masks are sacred (Part 1 Rule 4) and Part 3 cannot apply them itself.

    Raises ValueError for a non-3D array, non-finite input, amplitude
    outside (0, 1], or smoothing_passes < 1.
    """
    a = np.asarray(phi, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"phi must be a 3D array, got ndim={a.ndim}")
    if not np.all(np.isfinite(a)):
        raise ValueError("phi contains non-finite values")
    if not (0.0 < amplitude <= 1.0):
        raise ValueError(f"amplitude must be in (0, 1], got {amplitude}")
    if smoothing_passes < 1:
        raise ValueError("smoothing_passes must be >= 1")

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(a.shape)
    noise = _box_blur_3d(noise, smoothing_passes)
    noise_rms = float(np.sqrt(np.mean(noise * noise)))
    if noise_rms <= 1e-15:
        return a.astype(phi.dtype, copy=True)
    phi_rms = float(np.sqrt(np.mean(a * a)))
    scale = amplitude * (phi_rms if phi_rms > 1e-15 else 1.0) / noise_rms
    return (a + scale * noise).astype(phi.dtype)


# ---------------------------------------------------------------------------
# Failure-region memory
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    W_mm: float
    d_halo_mm: float
    lifecycle_state: str
    failure_reason: str
    phi_snapshot_paths: dict = field(default_factory=dict)


class FailureRegionMemory:
    """Remembers where in (W, d_halo) space candidates consistently die, so
    fresh initializations avoid re-exploring those regions.

    Scope note (honest): the spec says failure φ snapshots inform the outer
    loop. This implementation keys regions on the outer-loop scalars
    (W, d_halo) — the variables the outer loop actually controls when
    seeding — and RETAINS the φ snapshot paths on every record so a future,
    richer shape-space distance can be added without changing the storage.
    φ-space similarity itself is not implemented here.
    """

    def __init__(self, region_radius_mm: float = 1.0, kill_threshold: int = 3) -> None:
        if region_radius_mm <= 0:
            raise ValueError("region_radius_mm must be > 0")
        if kill_threshold < 1:
            raise ValueError("kill_threshold must be >= 1")
        self._radius = region_radius_mm
        self._threshold = kill_threshold
        self._records: list[FailureRecord] = []

    def record_failure(self, outcome: CandidateOutcome) -> None:
        """Store a failed candidate. Success states are ignored (not an
        error — callers can pipe every outcome through)."""
        if outcome.failure_reason is None:
            return
        self._records.append(
            FailureRecord(
                W_mm=outcome.W_mm,
                d_halo_mm=outcome.d_halo_mm,
                lifecycle_state=outcome.lifecycle_state,
                failure_reason=outcome.failure_reason,
                phi_snapshot_paths=dict(outcome.phi_snapshot_paths),
            )
        )

    def failures_near(self, W_mm: float, d_halo_mm: float) -> list[FailureRecord]:
        return [
            r for r in self._records
            if abs(r.W_mm - W_mm) <= self._radius
            and abs(r.d_halo_mm - d_halo_mm) <= self._radius
        ]

    def is_blacklisted(self, W_mm: float, d_halo_mm: float) -> bool:
        """True when >= kill_threshold failures cluster within region_radius
        of the query point — the outer loop should seed elsewhere."""
        return len(self.failures_near(W_mm, d_halo_mm)) >= self._threshold

    @property
    def records(self) -> list[FailureRecord]:
        return list(self._records)


# ---------------------------------------------------------------------------
# One evolutionary step over a population of inner-loop candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolutionPlan:
    """What to do next with a population: which candidates continue as-is
    (survivors) and, for each replacement slot, which survivor to perturb
    (round-robin over survivors, deterministic)."""

    survivors: list
    killed: list
    perturb_from: list  # survivor outcome to clone+perturb, one per killed slot


def plan_evolution_step(
    candidates: Sequence[CandidateOutcome],
    kill_fraction: float = EVOLUTION_KILL_FRACTION,
) -> EvolutionPlan:
    survivors, killed = select_survivors(candidates, kill_fraction)
    perturb_from = [survivors[i % len(survivors)] for i in range(len(killed))]
    return EvolutionPlan(survivors=survivors, killed=killed, perturb_from=perturb_from)
