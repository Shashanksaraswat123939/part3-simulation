"""
gradient_combiner.py — Part 3 Stage 4: gradient combination + weight calibration.

Spec ("Gradient Combination", 03_optimizer_workflow):

    total_gradient = normalize(aero_gradient) × w_aero
                   + normalize(mass_gradient) × w_mass
                   + normalize(COM_gradient)  × w_com
                   + normalize(mfg_gradient)  × w_mfg

    Each gradient normalized to the same RMS magnitude before weighting.
    Weights are hyperparameters whose INITIAL values must come from
    sensitivity analysis, not guesses.

NOTE ON OWNERSHIP: Part 1's phi_updater.combine_gradients performs this same
math at φ-update time and is authoritative inside update_phi(). This module
exists for (a) weight calibration, (b) Part 3-side verification tests, and
(c) any caller combining SCALAR gradient summaries. The formulas are kept
byte-for-byte equivalent (unit-RMS normalization, 1e-12 floor) so the two
can never disagree; test_gradient_combiner locks this.

WARNING (recorded in the Part 2 audit, finding N-2): unit-RMS normalization
amplifies weak-but-nonzero gradients to parity with strong ones. A gradient
term built on placeholder physics (e.g. the current fabricated com_x
penalty) will be inflated to the same RMS as the real aero gradient. Keep
w_com small until the ballast experiment replaces the placeholder data, and
use calibrate_gradient_weights so weights reflect measured impact.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

import numpy as np

from optimizer_contract import GradientWeights

RMS_FLOOR = 1e-12


def normalize_to_unit_rms(gradient: np.ndarray) -> np.ndarray:
    """g / rms(g); zeros if rms < 1e-12 (matches Part 1 spec exactly)."""
    g = np.asarray(gradient, dtype=float)
    if not np.all(np.isfinite(g)):
        raise ValueError("gradient contains non-finite values")
    rms = math.sqrt(float(np.mean(g * g))) if g.size else 0.0
    if rms <= RMS_FLOOR:
        return np.zeros_like(g)
    return g / rms


def combine_gradients(
    aero_gradient: np.ndarray,
    mass_gradient: np.ndarray,
    com_gradient: np.ndarray,
    mfg_gradient: np.ndarray,
    weights: GradientWeights,
) -> np.ndarray:
    """Normalize-then-weight combination. All four arrays must share a shape.

    Raises ValueError on shape mismatch or non-finite input. Naive addition
    without normalization is forbidden by spec — different units and scales
    would bias the optimizer toward whichever term is largest (typically aero).
    """
    arrays = [np.asarray(a, dtype=float) for a in
              (aero_gradient, mass_gradient, com_gradient, mfg_gradient)]
    shape = arrays[0].shape
    for name, a in zip(("aero", "mass", "com", "mfg"), arrays):
        if a.shape != shape:
            raise ValueError(
                f"{name} gradient shape {a.shape} != aero gradient shape {shape}"
            )
    n_aero, n_mass, n_com, n_mfg = (normalize_to_unit_rms(a) for a in arrays)
    return (
        n_aero * weights.w_aero
        + n_mass * weights.w_mass
        + n_com * weights.w_com
        + n_mfg * weights.w_mfg
    )


def calibrate_gradient_weights(
    evaluate_T: Callable[[Mapping[str, float]], float],
    base_inputs: Mapping[str, float],
    typical_changes: Mapping[str, float],
    mfg_typical_impact_s: float,
) -> GradientWeights:
    """Set initial weights by sensitivity analysis, exactly per spec:

        run RTC with realistic D20, m, h_com ranges
        measure dT/dD20 × typical_D20_change
        measure dT/dm   × typical_m_change
        measure dT/dh_com × typical_h_com_change
        set weights so each term contributes comparably

    Args:
        evaluate_T: callable mapping an input dict (keys at least
            'D20', 'm_total', 'h_com', plus whatever else it needs held in
            base_inputs) to a race time in seconds. Typically wraps Part 2's
            adapter. Central finite differences are used — no JAX needed here,
            since this runs once, offline.
        base_inputs: realistic baseline values for every input evaluate_T reads.
        typical_changes: expected per-iteration design change magnitude for
            'D20' (N), 'm_total' (kg), 'h_com' (m). All must be > 0.
        mfg_typical_impact_s: expected seconds of manufacturing-penalty change
            per iteration. Manufacturing penalties don't flow through the RTC,
            so their impact scale must be supplied, not measured. Must be >= 0.

    Returns:
        GradientWeights scaled so the LARGEST impact term gets weight 1.0 and
        every other term's weight is (its impact / largest impact) — i.e.
        after unit-RMS normalization each term contributes proportionally to
        its measured real-world effect on T.
    """
    required = ("D20", "m_total", "h_com")
    for key in required:
        if key not in base_inputs:
            raise ValueError(f"base_inputs missing required key {key!r}")
        if key not in typical_changes:
            raise ValueError(f"typical_changes missing required key {key!r}")
        if not (typical_changes[key] > 0 and math.isfinite(typical_changes[key])):
            raise ValueError(f"typical_changes[{key!r}] must be positive finite")
    if not (mfg_typical_impact_s >= 0 and math.isfinite(mfg_typical_impact_s)):
        raise ValueError("mfg_typical_impact_s must be non-negative finite")

    impacts: dict[str, float] = {}
    for key in required:
        step = typical_changes[key] * 1e-2  # small FD step relative to typical change
        hi = dict(base_inputs); hi[key] = base_inputs[key] + step
        lo = dict(base_inputs); lo[key] = base_inputs[key] - step
        dT_dx = (evaluate_T(hi) - evaluate_T(lo)) / (2.0 * step)
        if not math.isfinite(dT_dx):
            raise ValueError(f"finite-difference dT/d{key} is non-finite")
        impacts[key] = abs(dT_dx) * typical_changes[key]

    impacts["mfg"] = mfg_typical_impact_s
    largest = max(impacts.values())
    if largest <= 0:
        raise ValueError(
            "all measured impacts are zero — evaluate_T is not responding to "
            "its inputs; check the baseline and the RTC wiring"
        )
    return GradientWeights(
        w_aero=impacts["D20"] / largest,
        w_mass=impacts["m_total"] / largest,
        w_com=impacts["h_com"] / largest,
        w_mfg=impacts["mfg"] / largest,
    )


def scalar_gradient_norm(gradients: Mapping[str, float]) -> float:
    """L2 norm over the objective's scalar gradient dict (dT_dD20 etc.),
    ignoring non-numeric entries (e.g. manufacturing_gradient=None).
    Used by the convergence tracker."""
    total = 0.0
    for value in gradients.values():
        if isinstance(value, (int, float)) and math.isfinite(value):
            total += float(value) ** 2
    return math.sqrt(total)
