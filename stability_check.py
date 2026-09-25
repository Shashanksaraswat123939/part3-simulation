"""
stability_check.py — Part 3 Stage 6: inner-loop step 9 (wheel loads, CoP margin).

Two tiers, matching the Rolling Friction Policy deferral in the Part 2 spec:

  Tier 1 — STATIC wheel loads. Pure rigid-body statics from mass and x_com;
  no aero, no calibration data needed. Always computed:

      W_front_static = m·g·(W − x_com)/W
      W_rear_static  = m·g·x_com/W          (moments about the axles;
                                             x_com measured from front axle)

  A candidate whose COM lies outside the wheelbase (negative load on either
  axle) is flagged unstable — it would tip on the track.

  Tier 2 — AERO-CORRECTED loads and CoP margin, using the spec's upgrade
  formula:

      W_front = (m·g·d_rear − Cm·q·A·L_ref) / W
      W_rear  = m·g − W_front − L

  Tier 2 is gated on the same three prerequisites as the per-wheel friction
  upgrade (mu fitted from track data, ballast COM experiment completed,
  Cm and L wired into the RTC). Until those are met, Tier 2 fields are None
  and `aero_checked` is False. Additionally, the pitching-moment reference
  length L_ref is currently a √A stand-in inside Part 2 (a flagged
  placeholder), so Tier 2 refuses to run against a stand-in convention —
  it raises loudly if invoked with the placeholder flag set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

GRAVITY_MPS2 = 9.81  # must match Part 2 physics_contract.GRAVITY_MPS2


@dataclass(frozen=True)
class StabilityReport:
    """Result of the stability check for one candidate iteration."""

    static_W_front_N: float
    static_W_rear_N: float
    statically_stable: bool
    aero_checked: bool
    aero_W_front_N: Optional[float]
    aero_W_rear_N: Optional[float]
    cop_margin_m: Optional[float]
    notes: str


def _require_positive(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def check_stability(
    total_mass_kg: float,
    x_com_m: float,
    W_mm: float,
    *,
    prerequisites_met: bool = False,
    Cm: Optional[float] = None,
    L_N: Optional[float] = None,
    A_m2: Optional[float] = None,
    q_ref_pa: Optional[float] = None,
    cm_ref_length_is_placeholder: bool = True,
) -> StabilityReport:
    """Run the stability check (inner-loop step 9).

    Args:
        total_mass_kg: full-car mass, kg.
        x_com_m: COM fore-aft position measured FROM THE FRONT AXLE, m.

            ⚠ This is NOT the mass_com_ingest convention, despite what this
            docstring claimed until 2026-07-24. Part 2 reports com_x_m in CAR
            coordinates with x=0 at the NOSE TIP
            (physics_contract.MOMENT_REFERENCE_POINT_M). Callers must subtract
            x_front: `x_com_m = mass_report.com_x_m - x_front_mm/1000`.
            inner_loop was passing the nose-origin value unconverted, which
            reported 13.5%/86.5% front/rear where the truth is 48.9%/51.1%.
        W_mm: wheelbase, mm.
        prerequisites_met: True only when all three Rolling Friction Policy
            prerequisites hold. Gate for Tier 2.
        Cm, L_N, A_m2, q_ref_pa: full-car aero values, required for Tier 2.
        cm_ref_length_is_placeholder: True while Part 2's Cm uses the √A
            stand-in reference length. Tier 2 REFUSES to run in that case,
            because the wheel-load formula's L_ref must match the Cm
            normalization or the moment term is silently wrong scale.

    Returns:
        StabilityReport. Tier 1 always populated; Tier 2 fields None unless
        prerequisites_met and the Cm convention is real.

    Raises:
        ValueError for invalid mass/wheelbase inputs.
        NotImplementedError (loud, per house Rule 8) if Tier 2 is requested
        while the Cm reference-length placeholder is still in force.
    """
    _require_positive("total_mass_kg", total_mass_kg)
    _require_positive("W_mm", W_mm)
    if not (isinstance(x_com_m, (int, float)) and math.isfinite(x_com_m)):
        raise ValueError(f"x_com_m must be finite, got {x_com_m!r}")

    W_m = W_mm / 1000.0
    mg = total_mass_kg * GRAVITY_MPS2
    w_front = mg * (W_m - x_com_m) / W_m
    w_rear = mg * x_com_m / W_m
    # bool(): a numpy x_com makes this np.bool_, which json.dump rejects
    statically_stable = bool(w_front > 0.0 and w_rear > 0.0)
    notes = "" if statically_stable else (
        f"COM outside wheelbase: x_com={x_com_m:.4f} m for W={W_m:.4f} m — "
        "negative static wheel load; candidate would tip"
    )

    if not prerequisites_met:
        return StabilityReport(
            static_W_front_N=w_front,
            static_W_rear_N=w_rear,
            statically_stable=statically_stable,
            aero_checked=False,
            aero_W_front_N=None,
            aero_W_rear_N=None,
            cop_margin_m=None,
            notes=(notes + " | " if notes else "")
            + "aero tier skipped: rolling-friction prerequisites unmet (by design)",
        )

    if cm_ref_length_is_placeholder:
        # ? UNRESOLVED: Part 2's Cm uses ref_length = sqrt(A_full) as an
        # explicit stand-in. The wheel-load moment term Cm·q·A·L_ref is only
        # meaningful when L_ref matches the convention Cm was normalized
        # with. Running Tier 2 against the stand-in would produce a moment
        # term of silently wrong scale.
        raise NotImplementedError(
            "? UNRESOLVED: aero-corrected wheel loads requested but Part 2's "
            "pitching-moment reference length is still the sqrt(A) stand-in. "
            "Confirm the real reference length (likely wheelbase W) in "
            "physics_contract.to_full_car, then pass "
            "cm_ref_length_is_placeholder=False."
        )

    for name, value in (("Cm", Cm), ("L_N", L_N), ("A_m2", A_m2), ("q_ref_pa", q_ref_pa)):
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Tier 2 stability requires finite {name}, got {value!r}")

    # Spec formula. d_rear = distance from COM to rear axle = W − x_com.
    # L_ref: once the placeholder is resolved this MUST equal the Cm
    # normalization length; the caller asserts that by clearing the flag.
    L_ref = W_m
    d_rear = W_m - x_com_m
    aero_w_front = (total_mass_kg * GRAVITY_MPS2 * d_rear - Cm * q_ref_pa * A_m2 * L_ref) / W_m
    aero_w_rear = mg - aero_w_front - L_N

    # Center-of-pressure margin: distance between COM and the aero load
    # centroid along x. CoP_x from the pitching moment about the reference
    # point: x_cop = x_com − M/(L) is ill-conditioned at small L; use the
    # front/rear load split instead: margin = (aero front-load fraction −
    # static front-load fraction) × W. Positive margin = aero shifts load
    # rearward less than statics (stable-ish); reported, not thresholded —
    # thresholds are a design decision to be set from track data.
    static_front_frac = w_front / mg
    total_aero_load = aero_w_front + aero_w_rear
    if abs(total_aero_load) < 1e-12:
        cop_margin = None
        notes = (notes + " | " if notes else "") + "total aero-corrected load ~0; CoP margin undefined"
    else:
        aero_front_frac = aero_w_front / total_aero_load
        cop_margin = (aero_front_frac - static_front_frac) * W_m

    if aero_w_front <= 0.0 or aero_w_rear <= 0.0:
        notes = (notes + " | " if notes else "") + (
            f"aero-corrected wheel load non-positive (front={aero_w_front:.4f} N, "
            f"rear={aero_w_rear:.4f} N) — pitch instability at speed"
        )

    return StabilityReport(
        static_W_front_N=w_front,
        static_W_rear_N=w_rear,
        statically_stable=statically_stable,
        aero_checked=True,
        aero_W_front_N=aero_w_front,
        aero_W_rear_N=aero_w_rear,
        cop_margin_m=cop_margin,
        notes=notes,
    )
