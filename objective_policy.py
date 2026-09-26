"""
objective_policy.py — Part 3 Stage 3: objective composition and ranking.

Spec ("Objective" section, 03_optimizer_workflow):

    T_penalized = T_raw
                + COM_penalty           (Part 2 owns this; already inside
                                         the adapter's T_penalized output)
                + manufacturing_penalty (Part 3 adds, from Part 1 gate data)
                + rule_margin_penalty   (Part 3 adds, soft near-boundary)

    Final ranking: T_raw only, among fully valid candidates.
    Penalties guide the search. They do not redefine the competition
    objective.

This module is the ONLY place these two rankings are implemented. If any
other code sorts candidates, it calls into here.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from optimizer_contract import (
    FAILURE_PENALTY_S,
    CandidateOutcome,
    PenaltyInputs,
)


def compose_penalized_time(T_com_penalized: float, penalties: PenaltyInputs) -> float:
    """Add Part 3-owned penalties on top of Part 2's COM-penalized time.

    Args:
        T_com_penalized: seconds; T_raw + COM penalties, straight from
            Part 2's race_value_and_grad_guarded (its 2nd return value).
        penalties: manufacturing + rule-margin penalty seconds.

    Returns:
        T_penalized in seconds (the evolutionary/gradient objective).

    Raises:
        ValueError on non-finite T_com_penalized. Penalty validation lives
        in PenaltyInputs itself.
    """
    if not (isinstance(T_com_penalized, (int, float)) and math.isfinite(T_com_penalized)):
        raise ValueError(f"T_com_penalized must be finite, got {T_com_penalized!r}")
    if T_com_penalized <= 0:
        raise ValueError(
            f"T_com_penalized={T_com_penalized} s is non-positive; a 20 m race "
            "time cannot be <= 0 — upstream objective bug"
        )
    return T_com_penalized + penalties.total_s


def search_ranking(candidates: Sequence[CandidateOutcome]) -> list[CandidateOutcome]:
    """Evolutionary/search ranking: ascending T_penalized; dead candidates
    carry FAILURE_PENALTY_S and therefore sink to the bottom deterministically.
    Ties broken by candidate_id for reproducibility."""
    return sorted(candidates, key=lambda c: (c.ranking_time_s(), c.candidate_id))


def final_ranking(candidates: Sequence[CandidateOutcome]) -> list[CandidateOutcome]:
    """FINAL ranking per spec: T_raw only, among fully valid candidates.

    Candidates in failure states are excluded entirely — they are not
    "very slow cars", they are non-cars. Returns ascending by T_raw,
    ties broken by candidate_id.
    """
    valid = [c for c in candidates if c.is_fully_valid and c.T_raw is not None
             and not getattr(c, "is_underweight", False)]
    return sorted(valid, key=lambda c: (c.T_raw, c.candidate_id))


def select_build_candidate(candidates: Sequence[CandidateOutcome]) -> Optional[CandidateOutcome]:
    """The winner: fastest fully valid car by T_raw, or None if no candidate
    survived. Never falls back to ranking failures — an empty search result
    must surface as 'no build candidate', not as the least-broken failure."""
    ranked = final_ranking(candidates)
    return ranked[0] if ranked else None


def failure_penalty_for_state(lifecycle_state: str) -> float:
    """Uniform failure penalty for ranking dead candidates.

    Deliberately NOT graduated by failure type: the spec's evolutionary
    layer learns from failure *regions* via φ snapshots and failure reasons,
    not from pretending one failure mode is 'faster' than another.
    """
    del lifecycle_state
    return FAILURE_PENALTY_S
