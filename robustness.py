"""
robustness.py — Part 3 Stage 11: robustness checks on finalists.

Spec ("Robustness Checks"):

    surface machining tolerance: ±0.1 mm surface perturbation → T change?
    mass tolerance:              ±1 g                        → T change?
    COM tolerance:               ±1 mm                       → T change?
    mu variation:                ±0.005                      → T change?
    CO2 thrust variation:        ±5%                         → T change?
    CFD mesh refinement:         coarse vs fine              → D20 change?

    Robustness margins are reported alongside T_raw for the final ranking.
    A slightly slower but robust car beats a fragile mathematical winner.

Two of the six checks (surface perturbation, mesh refinement) need external
machinery (re-mesh + re-CFD). They are injected callables; when a caller
cannot supply one, that check's status is "not_run" with an explicit
reason — a check is never silently skipped, and a report with unrun checks
says so in aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

STATUS_RUN = "run"
STATUS_NOT_RUN = "not_run"

# Spec-mandated perturbation magnitudes.
MASS_TOL_KG = 0.001           # ±1 g
COM_TOL_M = 0.001             # ±1 mm
MU_TOL = 0.005                # ±0.005
THRUST_TOL_FRACTION = 0.05    # ±5%
SURFACE_TOL_MM = 0.1          # ±0.1 mm


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    delta_T_s: Optional[float]      # max |T(perturbed) − T(base)| over ± cases
    detail: str


@dataclass(frozen=True)
class RobustnessReport:
    candidate_id: str
    T_base_s: float
    checks: tuple
    all_checks_run: bool
    max_delta_T_s: Optional[float]

    def summary(self) -> str:
        lines = [f"Robustness report for {self.candidate_id} (T_base={self.T_base_s:.6f} s)"]
        for c in self.checks:
            if c.status == STATUS_RUN:
                lines.append(f"  {c.name}: ΔT = {c.delta_T_s:.6f} s")
            else:
                lines.append(f"  {c.name}: NOT RUN — {c.detail}")
        if not self.all_checks_run:
            lines.append("  WARNING: report is incomplete; unrun checks listed above.")
        return "\n".join(lines)


def _two_sided_delta(evaluate: Callable[[Mapping[str, float]], float],
                     base_inputs: Mapping[str, float],
                     key: str, delta: float, T_base: float) -> float:
    hi = dict(base_inputs); hi[key] = base_inputs[key] + delta
    lo = dict(base_inputs); lo[key] = base_inputs[key] - delta
    d_hi = abs(evaluate(hi) - T_base)
    d_lo = abs(evaluate(lo) - T_base)
    result = max(d_hi, d_lo)
    if not math.isfinite(result):
        raise ValueError(f"robustness check for {key!r} produced non-finite ΔT")
    return result


def run_robustness_checks(
    candidate_id: str,
    evaluate_T: Callable[[Mapping[str, float]], float],
    base_inputs: Mapping[str, float],
    surface_perturbation_T: Optional[Callable[[float], float]] = None,
    mesh_refinement_D20: Optional[Callable[[], tuple]] = None,
) -> RobustnessReport:
    """Run the six spec robustness checks.

    Args:
        candidate_id: for the report.
        evaluate_T: maps an input dict with keys 'm_total' (kg), 'h_com' (m),
            'mu', 'thrust_scale' (dimensionless multiplier, base 1.0), plus
            whatever else the wrapped RTC needs, to a race time in seconds.
            The thrust_scale hook is how ±5% thrust is applied — the caller's
            wrapper rescales the thrust model.
        base_inputs: baseline values for every key evaluate_T reads. Must
            include 'm_total', 'h_com', 'mu', 'thrust_scale'.
        surface_perturbation_T: optional callable(tolerance_mm) -> T seconds
            for the re-meshed ±0.1 mm surface. None → check not_run.
        mesh_refinement_D20: optional callable() -> (D20_coarse, D20_fine).
            None → check not_run.

    Returns:
        RobustnessReport with per-check status; never raises for missing
        optional hooks, always raises for a broken evaluate_T.
    """
    for key in ("m_total", "h_com", "mu", "thrust_scale"):
        if key not in base_inputs:
            raise ValueError(f"base_inputs missing required key {key!r}")
    T_base = evaluate_T(dict(base_inputs))
    if not math.isfinite(T_base):
        raise ValueError(f"baseline T is non-finite: {T_base!r}")

    checks: list[CheckResult] = []

    checks.append(CheckResult(
        name="mass_tolerance_pm_1g", status=STATUS_RUN,
        delta_T_s=_two_sided_delta(evaluate_T, base_inputs, "m_total", MASS_TOL_KG, T_base),
        detail=f"±{MASS_TOL_KG*1000:.0f} g on m_total",
    ))
    checks.append(CheckResult(
        name="com_tolerance_pm_1mm", status=STATUS_RUN,
        delta_T_s=_two_sided_delta(evaluate_T, base_inputs, "h_com", COM_TOL_M, T_base),
        detail=f"±{COM_TOL_M*1000:.0f} mm on h_com",
    ))
    checks.append(CheckResult(
        name="mu_variation_pm_0p005", status=STATUS_RUN,
        delta_T_s=_two_sided_delta(evaluate_T, base_inputs, "mu", MU_TOL, T_base),
        detail=f"±{MU_TOL} on mu",
    ))
    checks.append(CheckResult(
        name="thrust_variation_pm_5pct", status=STATUS_RUN,
        delta_T_s=_two_sided_delta(
            evaluate_T, base_inputs, "thrust_scale",
            THRUST_TOL_FRACTION * base_inputs["thrust_scale"], T_base,
        ),
        detail=f"±{THRUST_TOL_FRACTION*100:.0f}% thrust scale",
    ))

    if surface_perturbation_T is not None:
        T_pert = surface_perturbation_T(SURFACE_TOL_MM)
        if not math.isfinite(T_pert):
            raise ValueError("surface_perturbation_T produced non-finite T")
        checks.append(CheckResult(
            name="surface_machining_pm_0p1mm", status=STATUS_RUN,
            delta_T_s=abs(T_pert - T_base),
            detail=f"±{SURFACE_TOL_MM} mm surface perturbation via re-mesh",
        ))
    else:
        checks.append(CheckResult(
            name="surface_machining_pm_0p1mm", status=STATUS_NOT_RUN, delta_T_s=None,
            detail="no surface-perturbation hook supplied (needs re-mesh + re-CFD)",
        ))

    if mesh_refinement_D20 is not None:
        d_coarse, d_fine = mesh_refinement_D20()
        for label, value in (("coarse", d_coarse), ("fine", d_fine)):
            if not (isinstance(value, (int, float)) and math.isfinite(value)):
                raise ValueError(f"mesh_refinement_D20 {label} D20 non-finite: {value!r}")
        scale = max(abs(d_coarse), abs(d_fine), 1e-12)
        checks.append(CheckResult(
            name="cfd_mesh_refinement", status=STATUS_RUN,
            delta_T_s=abs(d_fine - d_coarse) / scale,  # relative D20 spread
            detail="relative D20 spread coarse vs fine (dimensionless)",
        ))
    else:
        checks.append(CheckResult(
            name="cfd_mesh_refinement", status=STATUS_NOT_RUN, delta_T_s=None,
            detail="no mesh-refinement hook supplied (needs coarse+fine CFD runs)",
        ))

    run_deltas = [c.delta_T_s for c in checks if c.status == STATUS_RUN]
    return RobustnessReport(
        candidate_id=candidate_id,
        T_base_s=T_base,
        checks=tuple(checks),
        all_checks_run=all(c.status == STATUS_RUN for c in checks),
        max_delta_T_s=max(run_deltas) if run_deltas else None,
    )
