"""
convergence.py — Part 3 Stage 5: inner-loop convergence criteria.

Spec ("Convergence Criteria", 03_optimizer_workflow) — stop inner loop when:

    |T_penalized(iter) - T_penalized(iter-1)| < 1 ms
    OR gradient norm < threshold
    OR candidate fails gates repeatedly (3+ consecutive iterations)
    OR iteration budget exhausted

The tracker is a small state machine so the inner loop stays readable and
the stopping logic stays testable in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from optimizer_contract import (
    INNER_CONVERGENCE_DELTA_T_S,
    MAX_CONSECUTIVE_GATE_FAILURES,
)

# Stop reasons — string enum, stable for logging/records.
REASON_DELTA_T = "delta_T_below_1ms"
REASON_GRAD_NORM = "gradient_norm_below_threshold"
REASON_GATE_FAILURES = "repeated_gate_failures"
REASON_BUDGET = "iteration_budget_exhausted"


@dataclass(frozen=True)
class ConvergenceStatus:
    stop: bool
    reason: Optional[str]
    converged: bool  # True only for the two "good" stops (ΔT / grad norm)


class ConvergenceTracker:
    """Feed it one update per inner iteration; it says when to stop.

    Semantics:
      - The ΔT criterion compares consecutive SUCCESSFUL objective
        evaluations only. A gate/CFD failure iteration does not reset the
        last-good T (a failed extraction says nothing about T's trend), but
        it does advance the iteration counter and the consecutive-failure
        counter.
      - A successful iteration resets the consecutive-failure counter.
      - Budget counts every iteration, success or failure.
    """

    def __init__(
        self,
        iteration_budget: int,
        gradient_norm_threshold: float,
        delta_t_threshold_s: float = INNER_CONVERGENCE_DELTA_T_S,
        max_consecutive_gate_failures: int = MAX_CONSECUTIVE_GATE_FAILURES,
    ) -> None:
        if iteration_budget < 1:
            raise ValueError("iteration_budget must be >= 1")
        if not (gradient_norm_threshold > 0 and math.isfinite(gradient_norm_threshold)):
            raise ValueError("gradient_norm_threshold must be positive finite")
        if not (delta_t_threshold_s > 0 and math.isfinite(delta_t_threshold_s)):
            raise ValueError("delta_t_threshold_s must be positive finite")
        if max_consecutive_gate_failures < 1:
            raise ValueError("max_consecutive_gate_failures must be >= 1")
        self._budget = iteration_budget
        self._grad_threshold = gradient_norm_threshold
        self._delta_t = delta_t_threshold_s
        self._max_fail = max_consecutive_gate_failures
        self._iterations = 0
        self._consecutive_failures = 0
        self._last_T: Optional[float] = None

    @property
    def iterations(self) -> int:
        return self._iterations

    def update_failure(self) -> ConvergenceStatus:
        """Record a gate/CFD/objective failure iteration."""
        self._iterations += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_fail:
            return ConvergenceStatus(stop=True, reason=REASON_GATE_FAILURES, converged=False)
        return self._budget_check()

    def update_success(self, T_penalized: float, gradient_norm: float) -> ConvergenceStatus:
        """Record a successful objective evaluation."""
        if not (isinstance(T_penalized, (int, float)) and math.isfinite(T_penalized)):
            raise ValueError(f"T_penalized must be finite, got {T_penalized!r}")
        if not (isinstance(gradient_norm, (int, float)) and math.isfinite(gradient_norm)
                and gradient_norm >= 0):
            raise ValueError(f"gradient_norm must be finite >= 0, got {gradient_norm!r}")
        self._iterations += 1
        self._consecutive_failures = 0

        if self._last_T is not None and abs(T_penalized - self._last_T) < self._delta_t:
            self._last_T = T_penalized
            return ConvergenceStatus(stop=True, reason=REASON_DELTA_T, converged=True)
        self._last_T = T_penalized

        if gradient_norm < self._grad_threshold:
            return ConvergenceStatus(stop=True, reason=REASON_GRAD_NORM, converged=True)

        return self._budget_check()

    def _budget_check(self) -> ConvergenceStatus:
        if self._iterations >= self._budget:
            return ConvergenceStatus(stop=True, reason=REASON_BUDGET, converged=False)
        return ConvergenceStatus(stop=False, reason=None, converged=False)
