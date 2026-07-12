"""
pipeline_interface.py — Part 3 Stage 2: the Part 1 / Part 2 handshake layer.

Part 3 never imports Part 1 or Part 2 modules directly from loop code.
All external calls go through a PipelineBindings object so that:

  1. The exact interface Part 3 depends on is written down in ONE place
     (mirroring "Section 16 — Part 2 Interface Contract" of the Part 1 spec).
  2. Tests can inject fakes and exercise the whole optimizer without
     OpenFOAM, JAX, trimesh, or the real φ grids.
  3. Every not-yet-wired integration fails LOUDLY (house Rule 8) instead
     of silently returning garbage.

real_bindings() constructs the production wiring against the actual
part1_geometry/ and part2_simulation/ packages.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional


# Names Part 1 uses for the four φ grids; Part 2's candidate record requires
# exactly these keys for phi_grid_snapshot_paths.
PHI_COMPONENT_NAMES = ("nose", "sidepod", "rearpod", "main_body")


@dataclass(frozen=True)
class GateOutcome:
    """Part 3's view of Part 1's quality_gates.GateResult."""

    lifecycle_state: str
    phi_snapshot_paths: dict
    stl_path: Optional[str]
    stl_half_path: Optional[str]
    failure_reason: Optional[str]
    meshes: Optional[dict] = None


@dataclass(frozen=True)
class CFDOutcome:
    """Part 3's view of one Part 2 CFD run, already converted to FULL-CAR
    values per the Half-Car CFD Contract. Part 3 never sees half-car forces.
    """

    D20: float
    L: float
    Cm: float
    A: float
    converged: bool
    residual_final: float


@dataclass(frozen=True)
class ObjectiveOutcome:
    """Part 3's view of one Part 2 race-objective evaluation.

    T_com_penalized = T_raw + COM penalties (what Part 2's adapter returns
    as T_penalized). Part 3 adds manufacturing/rule-margin penalties on top
    — see objective_policy.compose_penalized_time.
    gradients keys: dT_dD20, dT_dmass, dT_dh_com, dT_dx_com, dT_dL
    (all w.r.t. the COM-penalized time; units s/N, s/kg, s/m, s/m, s/N).
    """

    T_raw: float
    T_com_penalized: float
    gradients: dict


@dataclass(frozen=True)
class MassReport:
    """Part 3's view of Part 2's FullCarMassCOM."""

    total_mass_kg: float
    com_x_m: float
    com_y_m: float
    com_z_m: float


@dataclass
class PipelineBindings:
    """Every callable Part 3 needs from the rest of the system.

    All fields are REQUIRED (no defaults) so a partially wired binding set
    fails at construction, not mid-optimization.

    Signatures (duck-typed; validate_bindings checks presence/callability):

      initialize_phi_fields(W_mm, d_halo_mm, seed) -> dict[name, phi_grid]
          Fresh random-legal φ fields for all four components.
      warm_start_phi_fields(prev_phi_grids, W_mm, d_halo_mm) -> dict
          Remap converged fields from a neighboring W (spec: warm-starting).
      perturb_phi_fields(phi_grids, seed, amplitude) -> dict
          Smooth random perturbation of a top candidate (spec: evolutionary).
      run_quality_gates(phi_grids, candidate_id, out_dir) -> GateOutcome
      compute_mass_report(phi_grids) -> MassReport
          φ volume integrals + fixed hardware rollup (Part 1 S5 + Part 2 S2).
      run_cfd(stl_half_path) -> CFDOutcome
          Half-car CFD, converted to full-car values before returning.
      evaluate_objective(D20, L, m_total, h_com, x_com, mu, wheel_moi) ->
          ObjectiveOutcome
      compute_adjoint_weight(...same args...) -> float          (w_D20, s/N)
      run_adjoint(stl_half_path, objective_weight) -> sensitivity_field
          Right-half surface sensitivity dObjective/dSurface.
      update_phi(phi_grids, sensitivity_field, meshes, dt, weights,
                 objective_gradients, mass_report) -> None (in place)
      write_candidate_record(outcome_dict) -> record_path (str)
    """

    initialize_phi_fields: Callable[..., dict]
    warm_start_phi_fields: Callable[..., dict]
    perturb_phi_fields: Callable[..., dict]
    run_quality_gates: Callable[..., GateOutcome]
    compute_mass_report: Callable[..., MassReport]
    run_cfd: Callable[..., CFDOutcome]
    evaluate_objective: Callable[..., ObjectiveOutcome]
    compute_adjoint_weight: Callable[..., float]
    run_adjoint: Callable[..., Any]
    update_phi: Callable[..., None]
    write_candidate_record: Callable[..., str]


def validate_bindings(bindings: PipelineBindings) -> None:
    """Raise TypeError if any binding is missing or not callable."""
    for f in fields(PipelineBindings):
        value = getattr(bindings, f.name, None)
        if not callable(value):
            raise TypeError(
                f"PipelineBindings.{f.name} must be callable, got {value!r}"
            )


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------


def _add_sibling_packages_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    # Support both the spec/internal package names and the actual local repo
    # folder names used beside this repository.
    for pkg in ("part1_geometry", "part2_simulation", "part1-simulation", "part2-simulation"):
        p = os.path.join(root, pkg)
        if p not in sys.path:
            sys.path.insert(0, p)


def real_bindings(
    thrust_csv_path: str,
    fixed_hardware_kwargs: dict,
    out_dir: str,
) -> PipelineBindings:
    """Bind Part 3 to the real Part 1 + Part 2 code.

    Args:
        thrust_csv_path: CSV for the locked race objective's thrust model.
        fixed_hardware_kwargs: exact constructor kwargs for Part 2's
            FixedHardwareSpec (co2_cartridge_mass_kg=0.023, COMs, masses).
        out_dir: root directory for candidate records and snapshots.

    Any Part 1/Part 2 import failure raises ImportError with an explicit
    message naming the missing package — never a silent stub.
    """
    _add_sibling_packages_to_path()

    try:
        from quality_gates import run_quality_gates as p1_run_quality_gates  # noqa: F401
        from mass_com_calculator import compute_all_machined_components
        from phi_updater import update_phi as p1_update_phi
    except ImportError as exc:
        raise ImportError(
            "Part 3 real_bindings requires part1_geometry/ on the path "
            f"(quality_gates, mass_com_calculator, phi_updater). Cause: {exc}"
        ) from exc
    try:
        from cfd_wrapper import run_half_car_cfd, CFDRunError  # noqa: F401
        from mass_com_ingest import FixedHardwareSpec, ingest_mass_com
        from race_objective import build_smooth_sheet_model
        from race_objective_adapter import race_value_and_grad_guarded
        from adjoint_contract import compute_adjoint_objective_weight
        from candidate_record import CandidateRecord, write_candidate_record
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Part 3 real_bindings requires part2_simulation/ on the path. "
            f"Cause: {exc}"
        ) from exc

    fixed_hardware = FixedHardwareSpec(**fixed_hardware_kwargs)
    model = build_smooth_sheet_model(thrust_csv_path)

    def _params(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        # Locked-file PARAM_NAMES order:
        # [drag_20_n, car_weight_kg, mu, wheel_moi_kg_m2,
        #  time_coefficient(=1.0), com_height_m, lift_20_n, com_x_m]
        return np.array(
            [D20, m_total, mu, wheel_moi, 1.0, h_com, L, x_com],
            dtype=np.float64,
        )

    def initialize_phi_fields(W_mm, d_halo_mm, seed):
        # ? UNRESOLVED: Part 1 exposes PhiGrid.init(mode="sphere") per grid
        # but no single factory that builds all four grids for a given
        # (W, d_halo). When Part 1 lands that factory (bounding_volumes +
        # fixed_hardware + PhiGrid wiring), call it here. Until then this
        # binding must be supplied by the caller.
        raise NotImplementedError(
            "? UNRESOLVED: Part 1 does not yet expose a build-all-phi-grids "
            "factory for a given (W_mm, d_halo_mm). Wire it here in "
            "pipeline_interface.real_bindings.initialize_phi_fields once "
            "part1_geometry provides it, or inject a custom binding."
        )

    def warm_start_phi_fields(prev_phi_grids, W_mm, d_halo_mm):
        # ? UNRESOLVED: requires PhiGrid.remap(new_bv, new_hard_masks) plus
        # recomputed bounding volumes for the new W. Wire once the Part 1
        # factory above exists.
        raise NotImplementedError(
            "? UNRESOLVED: warm-start remap needs Part 1's bounding-volume "
            "recompute + PhiGrid.remap wiring for the new W. See "
            "pipeline_interface.real_bindings.warm_start_phi_fields."
        )

    def perturb_phi_fields(phi_grids, seed, amplitude):
        from evolutionary import perturb_phi_array  # local import, no cycle at module load
        out = {}
        for name, phi in phi_grids.items():
            new_phi = phi  # PhiGrid is mutated in place by design in Part 1
            new_phi.phi = perturb_phi_array(new_phi.phi, seed=seed, amplitude=amplitude)
            new_phi.apply_hard_constraints()
            out[name] = new_phi
        return out

    def run_quality_gates(phi_grids, candidate_id, run_out_dir):
        r = p1_run_quality_gates(phi_grids, candidate_id, run_out_dir)
        return GateOutcome(
            lifecycle_state=r.lifecycle_state,
            phi_snapshot_paths=dict(r.phi_snapshot_paths),
            stl_path=r.stl_path,
            stl_half_path=r.stl_half_path,
            failure_reason=r.failure_reason,
            meshes=r.meshes,
        )

    def compute_mass_report(phi_grids):
        machined = compute_all_machined_components(
            phi_grids["nose"], phi_grids["sidepod"],
            phi_grids["rearpod"], phi_grids["main_body"],
        )
        full = ingest_mass_com(machined, fixed_hardware)
        return MassReport(
            total_mass_kg=full.total_mass_kg,
            com_x_m=full.com_x_m,
            com_y_m=full.com_y_m,
            com_z_m=full.com_z_m,
        )

    def run_cfd(stl_half_path):
        half, health = run_half_car_cfd(stl_half_path)
        full = half.to_full_car()
        return CFDOutcome(
            D20=full.D20, L=full.L, Cm=full.Cm, A=full.A,
            converged=health.converged,
            residual_final=health.residual_final,
        )

    def evaluate_objective(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        p = _params(D20, L, m_total, h_com, x_com, mu, wheel_moi)
        T_raw, T_pen, grads = race_value_and_grad_guarded(p, model)
        return ObjectiveOutcome(T_raw=T_raw, T_com_penalized=T_pen, gradients=grads)

    def compute_adjoint_weight(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        p = _params(D20, L, m_total, h_com, x_com, mu, wheel_moi)
        return float(compute_adjoint_objective_weight(p, model))

    def run_adjoint(stl_half_path, objective_weight):
        # ? UNRESOLVED: OpenFOAM adjoint solver is not wired anywhere in the
        # project yet (Part 2 build report: cfd_wrapper pipeline placeholder).
        # This must run the adjoint case with Objective = w_D20 × D20 and
        # return the right-half surface sensitivity field.
        raise NotImplementedError(
            "? UNRESOLVED: OpenFOAM adjoint is not wired. run_adjoint must "
            "execute the adjoint case for the half-car STL and return "
            "dObjective/dSurface on the right-half mesh. See Part 2's "
            "cfd_wrapper placeholder and the Adjoint Objective Contract."
        )

    def update_phi(phi_grids, sensitivity_field, meshes, dt, weights,
                   objective_gradients, mass_report):
        p1_update_phi(
            phi_grids,
            right_half_sensitivity=sensitivity_field,
            right_half_mesh=meshes,
            dt=dt,
            gradient_weights={
                "w_aero": weights.w_aero,
                "w_mass": weights.w_mass,
                "w_com": weights.w_com,
                "w_mfg": weights.w_mfg,
            },
        )

    def write_record(outcome: dict) -> str:
        record = CandidateRecord(**outcome)
        return write_candidate_record(record, out_dir)

    bindings = PipelineBindings(
        initialize_phi_fields=initialize_phi_fields,
        warm_start_phi_fields=warm_start_phi_fields,
        perturb_phi_fields=perturb_phi_fields,
        run_quality_gates=run_quality_gates,
        compute_mass_report=compute_mass_report,
        run_cfd=run_cfd,
        evaluate_objective=evaluate_objective,
        compute_adjoint_weight=compute_adjoint_weight,
        run_adjoint=run_adjoint,
        update_phi=update_phi,
        write_candidate_record=write_record,
    )
    validate_bindings(bindings)
    return bindings
