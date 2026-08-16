"""
convergence.py — Part 3 Stage 5: inner-loop convergence criteria.

Spec ("Convergence Criteria", 03_optimizer_workflow) — stop inner loop when:

    |T_penalized(iter) - T_penalized(iter-1)| < 1 ms, for N CONSECUTIVE
        iterations (a single sub-ms delta is noise -- see update_success)
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
    INNER_CONVERGENCE_CONSECUTIVE,
    INNER_CONVERGENCE_DELTA_T_S,
    MAX_CONSECUTIVE_GATE_FAILURES,
)

# Stop reasons — string enum, stable for logging/records.
# Not "delta_T_below_1ms". The threshold moved to 15 ms
# (INNER_CONVERGENCE_DELTA_T_S) once the drag noise was measured at +/-15 ms of
# race time -- a 1 ms criterion could not fire above the noise floor -- and this
# label kept claiming 1 ms. It is what a converged run prints as its reason, and
# the first mocked run to actually reach convergence printed the wrong number.
# Nothing parses it, so naming it after the criterion rather than a stale
# constant costs nothing.
REASON_DELTA_T = "delta_T_below_threshold"
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
        required_small_steps: int = INNER_CONVERGENCE_CONSECUTIVE,
    ) -> None:
        if iteration_budget < 1:
            raise ValueError("iteration_budget must be >= 1")
        if not (gradient_norm_threshold > 0 and math.isfinite(gradient_norm_threshold)):
            raise ValueError("gradient_norm_threshold must be positive finite")
        if not (delta_t_threshold_s > 0 and math.isfinite(delta_t_threshold_s)):
            raise ValueError("delta_t_threshold_s must be positive finite")
        if max_consecutive_gate_failures < 1:
            raise ValueError("max_consecutive_gate_failures must be >= 1")
        if required_small_steps < 1:
            raise ValueError("required_small_steps must be >= 1")
        self._budget = iteration_budget
        self._last_metric = None
        self._grad_threshold = gradient_norm_threshold
        self._delta_t = delta_t_threshold_s
        self._max_fail = max_consecutive_gate_failures
        self._required_small_steps = required_small_steps
        self._iterations = 0
        self._consecutive_failures = 0
        self._consecutive_small_steps = 0
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

    def update_success(self, T_penalized: float, gradient_norm: float,
                       feasible: bool = True, metric: float | None = None,
                       metric_threshold: float | None = None) -> ConvergenceStatus:
        """Record a successful objective evaluation.

        feasible=False means the candidate breaks a hard rule right now (in
        practice the T3.6 minimum mass). Such a candidate CANNOT be converged,
        however still the objective is moving: the live 2026-08-10 sweep stopped
        every candidate after 4 of 25 iterations on deltas of 3.3, 1.9 and
        1.9 ms while the car sat at 47.30 g against a 48 g floor, and recorded
        the result as converged=True. The barrier was pushing mass back out the
        whole time; it just moves the objective slowly, because the HJ step is
        CFL-limited rather than proportional to the gradient. So "the objective
        stopped moving" and "the car is finished" are different statements, and
        only the second deserves the word converged.

        An infeasible iteration still counts toward the budget -- this blocks
        early success, it does not grant unlimited iterations.

        metric / metric_threshold: converge on something OTHER than T_penalized.
        Used by the aero-only phase, where the mass channel is switched off and
        T_pen barely moves -- a 2-3% drag gain is worth ~3 ms of race time, so a
        T_pen test cannot see the very thing being optimised. Passing D20 and a
        drag-noise threshold makes the criterion measure the quantity actually
        being descended. Both must be given together.
        """
        if not (isinstance(T_penalized, (int, float)) and math.isfinite(T_penalized)):
            raise ValueError(f"T_penalized must be finite, got {T_penalized!r}")
        if not (isinstance(gradient_norm, (int, float)) and math.isfinite(gradient_norm)
                and gradient_norm >= 0):
            raise ValueError(f"gradient_norm must be finite >= 0, got {gradient_norm!r}")
        self._iterations += 1
        self._consecutive_failures = 0

        # CONSECUTIVE small steps, not one.
        #
        # A single |dT| below threshold used to end the run and report
        # converged=True. Measured 2026-07-28: a production run stopped after
        # ONE update on dT = 0.372 ms, while the pipeline's drag noise is about
        # +/-15 ms of race time -- so that delta was indistinguishable from
        # zero, and stopping on it declared success on noise. Under a noisy
        # measurement a small delta by chance is likely, not rare, so a
        # single-sample test fires more or less at random.
        #
        # Requiring several in a row is the standard remedy and costs nothing
        # when the objective really has flattened: it just takes N iterations to
        # say so. This makes the criterion STRICTER, not looser -- it was
        # producing false convergence, which is the expensive direction.
        if not feasible:
            # Small steps while illegal say nothing about being finished; a run
            # of them must not accumulate toward convergence.
            self._consecutive_small_steps = 0
            self._last_T = T_penalized
            if self._iterations >= self._budget:
                return ConvergenceStatus(stop=True, reason=REASON_BUDGET,
                                         converged=False)
            return ConvergenceStatus(stop=False, reason="", converged=False)

        # Which quantity is being tested for flatness, and against what.
        if metric is not None and metric_threshold is not None:
            value, threshold, last = metric, metric_threshold, self._last_metric
            self._last_metric = metric
            self._last_T = T_penalized
        else:
            value, threshold, last = T_penalized, self._delta_t, self._last_T
            self._last_T = T_penalized
            self._last_metric = None

        if last is not None and abs(value - last) < threshold:
            self._consecutive_small_steps += 1
            if self._consecutive_small_steps >= self._required_small_steps:
                return ConvergenceStatus(stop=True, reason=REASON_DELTA_T,
                                         converged=True)
            # Budget still applies while we wait for the run of small steps --
            # returning a bare "keep going" here would let the loop overrun it.
            return self._budget_check()
        self._consecutive_small_steps = 0
        self._last_T = T_penalized

        if gradient_norm < self._grad_threshold:
            return ConvergenceStatus(stop=True, reason=REASON_GRAD_NORM, converged=True)

        return self._budget_check()

    def _budget_check(self) -> ConvergenceStatus:
        if self._iterations >= self._budget:
            return ConvergenceStatus(stop=True, reason=REASON_BUDGET, converged=False)
        return ConvergenceStatus(stop=False, reason=None, converged=False)
