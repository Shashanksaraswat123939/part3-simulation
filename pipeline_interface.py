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
import warnings
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
    # Surface area the cutter cannot reach, mm^2. None when not measured.
    #
    # extract_unified_surface computes this and run_quality_gates read it once,
    # to pick between the strings "geometry_repaired" and "valid_simulated", and
    # then dropped the number. So the one quantity that says HOW unmanufacturable
    # a candidate is never left the function that computed it: every record in
    # the 2026-07-29 sweep says "geometry_repaired" and none says by how much.
    # It matters more now that the optimiser actually carves -- an organic shape
    # has real accessibility problems where the starting envelope had few.
    inaccessible_area_mm2: Optional[float] = None


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
    # Peak-to-peak swing of streamwise force over the averaged window, as a
    # fraction of its mean. None when not measured.
    #
    # Part 2 measures this and its comment states the contract: "The signal is
    # reported instead, and warned about. Whether an unsteady force is fatal is
    # a policy decision for the caller." The caller handed that decision never
    # received the number -- CFDOutcome had no field for it, so it died at the
    # Part 2 -> Part 3 boundary and survived only as a warning in a worker
    # thread. It is the reproducibility marker for D20 (measured 18-29% against
    # a 5% limit), so without it a candidate record cannot say how much to trust
    # its own race time, and merge_results ranks on a number whose error bar was
    # deliberately computed and then dropped.
    force_oscillation: Optional[float] = None
    # Standard error of the MEAN streamwise force, and how much that mean is
    # still drifting, both as fractions of it. These are the numbers that say
    # whether a drag DELTA is measurable -- force_oscillation is peak-to-peak,
    # which one outlier sets and which does not shrink with a longer window.
    # Carried here for the same reason force_oscillation had to be: Part 2
    # computes them and warns, but the caller holding the ranking decision
    # never saw them, so they died at this boundary and survived only as a
    # warning in a worker thread.
    force_mean_stderr: Optional[float] = None
    force_drift: Optional[float] = None


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
    """Part 3's view of Part 2's FullCarMassCOM.

    total_mass_kg / com_*_m describe the car as it CROSSES THE LINE -- dry, all
    propellant spent. The CO2 charge is reported separately because it is only
    aboard at the start; race_objective owns its mass over time. Use
    `launch_com()` for anything evaluated at t=0.
    """

    total_mass_kg: float
    com_x_m: float
    com_y_m: float
    com_z_m: float
    propellant_mass_kg: float = 0.0
    propellant_com: tuple = (0.0, 0.0, 0.0)

    def launch_com(self) -> tuple:
        """(mass, com_x, com_y, com_z) with a full propellant charge aboard."""
        m = self.total_mass_kg + self.propellant_mass_kg
        if m <= 0:
            return (0.0, self.com_x_m, self.com_y_m, self.com_z_m)
        px, py, pz = self.propellant_com
        w, p = self.total_mass_kg, self.propellant_mass_kg
        return (m,
                (w * self.com_x_m + p * px) / m,
                (w * self.com_y_m + p * py) / m,
                (w * self.com_z_m + p * pz) / m)


@dataclass(frozen=True)
class AdjointOutcome:
    """One adjoint solve: the surface sensitivity AND the mesh indexing it.

    These two are inseparable and must travel together. `sensitivity[i]` is
    dObjective/dSurface at `half_mesh.vertices[i]`; the array alone carries no
    information about which point each value belongs to.

    Keeping them apart produced the pipeline's longest-lived silent failure:
    inner_loop had only the sensitivity, so it reached for the nearest
    mesh-shaped object in scope -- `gate.meshes`, a dict[str, Trimesh] keyed by
    component. A dict has no `.vertices`, so the phi update raised
    AttributeError on the first real iteration, the loop's broad `except`
    caught it, and every iteration was quietly downgraded to
    "objective_failed" with phi never updated once.

    half_mesh may be None only for test doubles whose update_phi ignores it.
    """

    sensitivity: Any
    half_mesh: Any


@dataclass
class PipelineBindings:
    """Every callable Part 3 needs from the rest of the system.

    All fields are REQUIRED (no defaults) so a partially wired binding set
    fails at construction, not mid-optimization.

    Signatures (duck-typed; validate_bindings checks presence/callability):

      initialize_phi_fields(W_mm, x_front_mm, d_halo_mm, seed) -> dict[name, phi_grid]
          Fresh random-legal φ fields for all four components.
      warm_start_phi_fields(prev_phi_grids, W_mm, x_front_mm, d_halo_mm) -> dict
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
      run_adjoint(stl_half_path, objective_weight) -> AdjointOutcome
          Right-half surface sensitivity dObjective/dSurface, together with
          the half-car mesh that defines its vertex ordering.
      update_phi(phi_grids, sensitivity_field, half_mesh, dt, weights,
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
        from phi_grid_factory import (
            build_phi_grids_for_candidate,
            warm_start_phi_grids as warm_start_phi_grids_impl,
        )
    except ImportError as exc:
        raise ImportError(
            "Part 3 real_bindings requires part1_geometry/ on the path "
            f"(quality_gates, mass_com_calculator, phi_updater, "
            f"phi_grid_factory). Cause: {exc}"
        ) from exc
    try:
        from cfd_wrapper import run_half_car_cfd, run_half_car_adjoint, CFDRunError  # noqa: F401
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

    def initialize_phi_fields(W_mm, x_front_mm, d_halo_mm, seed):
        phi_grids, _bv = build_phi_grids_for_candidate(
            W_mm, x_front_mm, d_halo_mm, seed=seed,
        )
        return phi_grids

    def warm_start_phi_fields(prev_phi_grids, W_mm, x_front_mm, d_halo_mm):
        # Scoped-down warm start: rebuilds fresh grids at the new geometry
        # rather than remapping prev_phi_grids' field values (see
        # phi_grid_factory.warm_start_phi_grids's docstring — true remap
        # needs a PhiGrid.remap() that resamples a signed-distance field
        # across a resized/re-origined grid without corrupting |grad phi|=1,
        # which does not exist yet and is a separate, nontrivial task).
        phi_grids, _bv = warm_start_phi_grids_impl(
            prev_phi_grids, W_mm, x_front_mm, d_halo_mm,
        )
        return phi_grids

    def perturb_phi_fields(phi_grids, seed, amplitude):
        from evolutionary import perturb_phi_array  # local import, no cycle at module load
        out = {}
        for name, phi in phi_grids.items():
            new_phi = phi  # PhiGrid is mutated in place by design in Part 1
            # P3-1(b): PhiGrid stores the array as `.grid`, not `.phi`.
            new_phi.grid = perturb_phi_array(new_phi.grid, seed=seed, amplitude=amplitude)
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
            propellant_mass_kg=full.propellant_mass_kg,
            propellant_com=tuple(full.propellant_com),
        )

    def run_cfd(stl_half_path):
        half, health = run_half_car_cfd(stl_half_path)
        full = half.to_full_car()
        return CFDOutcome(
            D20=full.D20, L=full.L, Cm=full.Cm, A=full.A,
            converged=health.converged,
            residual_final=health.residual_final,
            force_oscillation=health.force_oscillation,
            force_mean_stderr=getattr(health, "force_mean_stderr", None),
            force_drift=getattr(health, "force_drift", None),
        )

    def evaluate_objective(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        p = _params(D20, L, m_total, h_com, x_com, mu, wheel_moi)
        T_raw, T_pen, grads = race_value_and_grad_guarded(p, model)
        return ObjectiveOutcome(T_raw=T_raw, T_com_penalized=T_pen, gradients=grads)

    def compute_adjoint_weight(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        p = _params(D20, L, m_total, h_com, x_com, mu, wheel_moi)
        return float(compute_adjoint_objective_weight(p, model))

    def run_adjoint(stl_half_path, objective_weight):
        # Runs the ESI adjointOptimisationFoam case (see Part 2's
        # openfoam_adjoint.py) and returns dObjective/dSurface on the
        # right-half mesh, one scalar per vertex in the same order as
        # trimesh.load(stl_half_path).vertices -- exactly what p1_update_phi
        # (phi_updater.apply_adjoint_sensitivity_symmetric) requires as
        # right_half_sensitivity. objective_weight (w_D20, from
        # compute_adjoint_weight) and the project's ADJOINT_HALF_CAR_SCALING
        # are applied inside run_half_car_adjoint.
        #
        # The mesh is loaded HERE, beside the solve that defines the ordering,
        # so the sensitivity can never be paired with the wrong geometry
        # downstream. process=False keeps trimesh from merging or reordering
        # vertices, which would silently break the index alignment.
        import trimesh

        sensitivity = run_half_car_adjoint(stl_half_path, objective_weight)
        half_mesh = trimesh.load(stl_half_path, process=False)
        n_v, n_s = len(half_mesh.vertices), len(sensitivity)
        if n_v != n_s:
            raise ValueError(
                f"adjoint returned {n_s} sensitivities but {stl_half_path} has "
                f"{n_v} vertices; they must be index-aligned."
            )
        return AdjointOutcome(sensitivity=sensitivity, half_mesh=half_mesh)

    def update_phi(phi_grids, sensitivity_field, half_mesh, dt, weights,
                   objective_gradients, mass_report):
        # `half_mesh` MUST be the half-car STL mesh, not Part 1's per-component
        # mesh dict. run_adjoint returns one scalar per vertex of
        # trimesh.load(stl_half_path).vertices, and p1_update_phi pairs
        # sensitivity[i] with right_half_mesh.vertices[i] -- so the mesh handed
        # over has to be that exact object or the pairing is meaningless.
        #
        # This used to receive gate.meshes, a dict[str, Trimesh]. A dict has no
        # .vertices, so the very first real iteration raised AttributeError,
        # which inner_loop caught broadly and downgraded to "objective_failed".
        # The loop therefore never updated phi and never crashed loudly enough
        # to be noticed.
        p1_update_phi(
            phi_grids,
            right_half_sensitivity=sensitivity_field,
            right_half_mesh=half_mesh,
            dt=dt,
            gradient_weights={
                "w_aero": weights.w_aero,
                "w_mass": weights.w_mass,
                "w_com": weights.w_com,
                "w_mfg": weights.w_mfg,
            },
            # Previously accepted and then discarded, which zeroed the mass and
            # COM gradient channels no matter how w_mass/w_com were calibrated.
            objective_gradients=objective_gradients,
            mass_report=mass_report,
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


# Triangle budget for the half-car STL handed to OpenFOAM.
#
# Measured on a real production-spacing build (2026-07-24): marching cubes emits
# 1_057_912 triangles over 0.0467 m2 of half-car surface -- ~0.3 mm facets, a
# 235 MB ASCII file. The CFD mesh it feeds is 1.21 mm at "medium" resolution,
# i.e. ~32_000 surface cells. The STL was therefore ~16x FINER than any mesh
# built from it, buying nothing and costing: 1.67 GB of Python heap in
# _normalise_solid_name, 12.8 s to rewrite, 14.8 s to edge-manifold-check,
# 4.3 s per re-parse (it is parsed 4+ times per solve), snappy's triSurface
# search tree over a million facets, and surfaceFeatureExtract harvesting
# "features" off every marching-cubes staircase step.
#
# 120k triangles is ~0.63 mm facets on this car -- still finer than the 0.606 mm
# finest CFD cell at "medium" with underbody refinement, so nothing is lost.
# Measured: 235 MB -> ~27 MB, volume preserved to 0.01%.
#
# Why not lower: quadric decimation of a marching-cubes half-car OPENS the
# surface below ~120k faces (Euler number 2 -> 5 at 60k, i.e. holes), and Part 2
# hard-rejects a non-edge-manifold STL. _decimate_for_cfd backs off until the
# result is watertight rather than trusting this number blindly.
# ponytail: the rule of thumb is facet size <= finest CFD cell size; revisit if
# you move to "fine" resolution or raise underbody_refinement_level.
STL_TRIANGLE_BUDGET = 120_000

# Re-exported so the decimation warning can name the envelope it failed. One
# source of truth: surface_extraction owns these, set from real solver runs.
from surface_extraction import (  # noqa: E402
    MEASURED_SAFE_MIN_ANGLE_DEG, MEASURED_SAFE_SLIVER_FRACTION,
)


# How far below the symmetry plane a vertex may drift and still be treated as
# rounding rather than a broken half. Quadric decimation moves vertices to
# error-minimising positions, so a y=0 cap vertex lands microns either side; a
# millimetre-scale excursion would mean the half itself is wrong and must still
# fail loudly rather than be snapped into looking correct.
_SYMMETRY_SNAP_TOL_M = 1.0e-4


def _snap_symmetry_plane(mesh):
    """Re-impose y >= 0 after any operation that moves vertices.

    extract_half_surface already flattens sub-cell overshoot onto y=0, but
    DECIMATION RUNS AFTER IT and repositions vertices, which puts them back
    below the plane. Measured 2026-07-27: an aero-only iteration produced a
    vertex at y = -1.000000e-6 against Part 2's y >= -1e-6 contract, exactly on
    the tolerance edge, and the candidate died at the CFD gate with
    "Expected right-half only".

    This is the shared exit every CFD half-STL passes through, so the invariant
    is restored once here rather than at each caller.
    """
    import numpy as _np
    y = mesh.vertices[:, 1]
    below = y < 0.0
    if not below.any():
        return mesh
    worst = float(-y[below].min())          # largest excursion below the plane
    if worst > _SYMMETRY_SNAP_TOL_M:
        raise ValueError(
            f"half-car mesh has a vertex {worst * 1e3:.4f} mm below the y=0 "
            f"symmetry plane, far past the {_SYMMETRY_SNAP_TOL_M * 1e3:.4f} mm "
            f"rounding tolerance. This is a broken half, not decimation noise."
        )
    mesh.vertices[below, 1] = 0.0
    return mesh


def _within_measured_mesh_envelope(mesh) -> bool:
    """Is this mesh inside the envelope snappyHexMesh was actually measured on?

    The decimated STL is a DIFFERENT mesh from the one Part 1's
    _check_mesh_quality inspected, and it never passes back through that gate --
    so nothing checked it. Making decimation succeed (it had never accepted a
    result at all, see _repair_decimated) promptly handed OpenFOAM a mesh with
    min angle 0.08 deg and 3.71% of triangles under 10 deg, against a measured
    envelope of 8.6 deg and 0.01%. That is 371x the sliver fraction anything has
    been measured at, and it went out silently.

    Thresholds are imported, not copied: surface_extraction owns them, and they
    came from meshing real carved STLs on openfoam2412 (8.65-14.19 deg, 0-3
    slivers, "Detected 0 illegal faces" every time). Widen them there, with
    evidence from a solver run, never here.
    """
    import numpy as _np
    from surface_extraction import (MEASURED_SAFE_MIN_ANGLE_DEG,
                                    MEASURED_SAFE_SLIVER_FRACTION)
    try:
        worst_per_face = _np.degrees(mesh.face_angles).min(axis=1)
    except Exception:  # noqa: BLE001 -- cannot measure => cannot certify
        return False
    if not len(worst_per_face):
        return False
    min_angle = float(worst_per_face.min())
    sliver_fraction = float((worst_per_face < 10.0).mean())
    return (min_angle >= MEASURED_SAFE_MIN_ANGLE_DEG
            and sliver_fraction <= MEASURED_SAFE_SLIVER_FRACTION)


def _repair_decimated(reduced) -> None:
    """Close a decimated half-car in place. Order matters.

    Quadric decimation does not open holes here -- measured on the 81,688-face
    half-car, the 50% result had ZERO boundary loops yet is_watertight was
    False, because 4 degenerate (zero-area) triangles left edges shared by more
    than two faces. fill_holes on that mesh is a no-op: there is nothing to
    fill. Dropping the degenerate faces is what creates the holes, and only
    then does fill_holes close them.

    The previous version called fill_holes alone and so never repaired
    anything; every backoff target failed and OpenFOAM was handed the full
    325,204-triangle mesh, with the slow snappyHexMesh and high memory the
    warning describes.
    """
    try:
        import trimesh
        reduced.merge_vertices()
        reduced.update_faces(reduced.nondegenerate_faces())
        reduced.update_faces(reduced.unique_faces())
        reduced.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(reduced)
    except Exception:  # noqa: BLE001 -- decimation is an optimisation, never fatal
        pass


def _decimate_for_cfd(mesh, budget: int = STL_TRIANGLE_BUDGET, _max_backoffs: int = 3):
    """Reduce a marching-cubes surface to a CFD-appropriate triangle count.

    Decimation is an optimisation, so it must never fail a candidate — but it
    must also never SILENTLY do nothing, which is what the first version did:
    it discarded any non-watertight result and returned the original, so a
    1.06M-triangle STL sailed through unchanged with no signal. Back off toward
    a coarser reduction instead, and say so if none of them hold.

    Returns a watertight mesh, always: either a reduced one or the original.
    """
    import warnings

    n = len(mesh.faces)
    if n <= budget:
        return _snap_symmetry_plane(mesh)
    target = budget
    for _ in range(_max_backoffs):
        if target >= n:
            break
        try:
            reduced = mesh.simplify_quadric_decimation(face_count=target)
        except Exception as exc:  # noqa: BLE001 -- optional, never fatal
            warnings.warn(f"STL decimation unavailable ({exc}); using {n} faces.",
                          RuntimeWarning, stacklevel=2)
            return _snap_symmetry_plane(mesh)
        if reduced is not None and len(reduced.faces):
            if not reduced.is_watertight:
                _repair_decimated(reduced)
            if reduced.is_watertight and _within_measured_mesh_envelope(reduced):
                return _snap_symmetry_plane(reduced)
        # Backing off means keeping MORE detail. Note this is not reliably the
        # safer direction: measured on the 81,688-face half-car, 25% decimation
        # came back watertight and 50% did not. Gentler targets leave sliver
        # triangles the aggressive pass would have collapsed outright.
        target *= 2
    warnings.warn(
        f"STL decimation produced no mesh inside the measured-safe envelope at "
        f"any target up to {target}; handing OpenFOAM the full {n}-triangle mesh "
        f"({budget} budget). Expect slow snappyHexMesh and high memory.\n"
        f"This is a real limit, not a tuning knob: quadric decimation targets "
        f"face count, not angle quality. Measured on the 81,688-face half-car, "
        f"a 50% reduction gives min angle 0.00 deg with 3.8% of triangles under "
        f"10 deg, against an envelope of "
        f"{MEASURED_SAFE_MIN_ANGLE_DEG} deg / "
        f"{100.0 * MEASURED_SAFE_SLIVER_FRACTION:.2f}%. Neither hole repair, "
        f"vertex merging (3.8% -> 3.5%, and it breaks watertightness) nor Taubin "
        f"smoothing (fraction 3.8% -> 0.4% but min angle only reaches 2.3 deg, "
        f"and >=10 iterations push vertices 0.375 mm past the y=0 plane) closes "
        f"the gap. The routes to a smaller STL are a coarser extraction grid or "
        f"a quality-preserving decimator -- or mesh one of these on the solver "
        f"and widen the envelope in surface_extraction with the evidence.",
        RuntimeWarning, stacklevel=2,
    )
    return _snap_symmetry_plane(mesh)


def unified_bindings(
    thrust_csv_path: str,
    fixed_hardware_kwargs,          # dict, or callable(W_mm) -> dict
    out_dir: str,
    cfd_kwargs: Optional[dict] = None,
    adjoint_kwargs: Optional[dict] = None,
    stl_triangle_budget: int = STL_TRIANGLE_BUDGET,
    cargo_placement: Optional[dict] = None,
    seed_geometry=None,
) -> PipelineBindings:
    """Bind Part 3 to the UNIFIED single-field geometry + the real objective.

    Same CFD / race-objective / adjoint bindings as real_bindings, but the
    geometry pipeline is the single labelled level set (unified_phi), not the
    four-grid path. That fixes what the four-grid path could not:

      * geometry is ONE connected watertight body, so the quality gate passes
        (the four-grid path fails on tool accessibility and disconnected slabs);
      * the drag adjoint and the real objective gradients evolve one field, so
        material can move across former component seams;
      * `phi_grids` handed around the inner loop is a UnifiedGeometry object
        (the loop treats it opaquely, so this is contract-compatible).

    The only thing still stubbed after this is the OpenFOAM binary: run_cfd /
    run_adjoint shell out to it exactly as in real_bindings.

    cfd_kwargs / adjoint_kwargs are forwarded verbatim to Part 2's
    run_half_car_cfd / run_half_car_adjoint. Before 2026-07-24 both were called
    with ALL defaults, so `n_subdomains` was pinned to 1 (every solve
    single-core, on a 360 GB box), `keep_run_dir` to False (logs deleted on the
    failure path), and resolution/iteration caps were unreachable — which made
    a cheap smoke run impossible to configure. Typical smoke values:
        cfd_kwargs={"resolution": "coarse", "max_iterations": 300,
                    "n_subdomains": 8, "keep_run_dir": True}
        adjoint_kwargs={"resolution": "coarse", "primal_iters": 200,
                        "adjoint_iters": 200, "keep_run_dir": True}
    """
    _add_sibling_packages_to_path()
    cfd_kwargs = dict(cfd_kwargs or {})
    adjoint_kwargs = dict(adjoint_kwargs or {})

    try:
        from unified_phi import (
            build_unified_geometry, enforce_symmetry, extract_unified_surface,
            compute_mass_com, extract_half_surface, remap_geometry,
        )
        from phi_updater import apply_adjoint_to_unified
    except ImportError as exc:
        raise ImportError(
            "unified_bindings requires part1 unified_phi + phi_updater. "
            f"Cause: {exc}"
        ) from exc
    try:
        from cfd_wrapper import run_half_car_cfd, run_half_car_adjoint  # noqa: F401
        from mass_com_ingest import FixedHardwareSpec, ingest_mass_com
        from race_objective import build_smooth_sheet_model
        from race_objective_adapter import race_value_and_grad_guarded
        from adjoint_contract import compute_adjoint_objective_weight
        from candidate_record import CandidateRecord, write_candidate_record
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            f"unified_bindings requires part2_simulation/ on the path. Cause: {exc}"
        ) from exc

    # Geometry-dependent, and BOTH sweep axes move it. The rear axle sits at
    # x_front + W, so the rear wheel/axle COM follows W; the canister COM and
    # rear-wing COM come off bounding volumes that depend on d_halo. Building
    # one spec here froze the whole fixed-hardware layout at whatever (W,
    # d_halo) the caller happened to pass -- run_optimization pinned W=130 while
    # sweeping W, run_two_stage pinned d_halo=20.0 while sweeping d_halo. So
    # accept a callable with default_fixed_hardware_kwargs' own signature and
    # re-derive per geometry.
    if callable(fixed_hardware_kwargs):
        _build_fh = fixed_hardware_kwargs
    else:
        _build_fh = lambda *_a: fixed_hardware_kwargs  # noqa: E731
    _fh_cache: dict = {}

    def fixed_hardware_for(geom):
        # One spec per distinct (W, x_front, d_halo); a sweep visits a handful.
        key = (round(float(geom.W_mm), 6), round(float(geom.x_front_mm), 6),
               round(float(geom.d_halo_mm), 6))
        if key not in _fh_cache:
            _fh_cache[key] = FixedHardwareSpec(**_build_fh(*key))
        return _fh_cache[key]

    model = build_smooth_sheet_model(thrust_csv_path)
    # The propellant charge from THIS csv, not mass_com_ingest's nominal 8.00 g.
    # The two disagreed by 0.13 g: the mass rollup carried 8.00 g at the canister
    # COM while race_objective burned the csv's own 7.87 g sheet, so the launch
    # mass the objective saw and the launch mass the record reported were
    # different numbers. Small (~2 ms of race time) but free to get right.
    import jax.numpy as _jnp
    from race_objective import sheet_mass as _sheet_mass
    _propellant_kg = float(_sheet_mass(_jnp.float64(0.0), model)
                           - model.mass_sheet_final)
    _COMPONENT_KEYS = ("nose", "sidepod", "rearpod", "main_body")

    def _params(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        return np.array([D20, m_total, mu, wheel_moi, 1.0, h_com, L, x_com],
                        dtype=np.float64)

    # Stage 1 scores cargo position AND fore-aft flip against the real
    # com_x-aware race objective (stage1_search.make_race_objective_cargo_scorer)
    # — the proxy cannot rank cargo because it has no com_x term. That choice
    # only means something if Stage 2 BUILDS with it. Before this was threaded
    # through, build_unified_geometry was called without cargo_placement, so
    # every Stage-2 car silently reverted to the geometric default
    # (corridor-centre, wide-forward) and the flip DOF was dead.
    def initialize_phi_fields(W_mm, x_front_mm, d_halo_mm, seed):
        # SEEDED from Stage 1's carved field when one is supplied.
        #
        # Stage 1 carves every candidate 30 steps toward the T3.6 floor to rank
        # it, with no CFD at all, and that shape used to be discarded -- so
        # Stage 2 rebuilt from the 150 g envelope and re-descended the same
        # curve at roughly 55 minutes per iteration. It was affordable only
        # because redistancing was eroding the body and doing most of the
        # carving for free; with reinitialise_sdf fixed, the descent measures
        # ~150 iterations at production spacing, about six days per d_halo.
        #
        # Seeding also makes the cross-d_halo comparison FAIR, which the
        # warm-start chain does not: every d_halo now begins from the same
        # Stage-1 carve rather than from wherever the previous one finished. The
        # 2026-08-01 sweep ranked perfectly in sweep order with starting masses
        # of 149.4, 135.9, 126.3, 119.3 and 113.2 g, which measured carve depth
        # rather than halo position.
        if seed_geometry is not None:
            import copy as _copy
            geom = _copy.deepcopy(seed_geometry)
            geom = remap_geometry(geom, W_mm=W_mm, x_front_mm=x_front_mm,
                                  d_halo_mm=d_halo_mm,
                                  cargo_placement=cargo_placement)
            enforce_symmetry(geom)
            return geom
        geom = build_unified_geometry(W_mm, x_front_mm, d_halo_mm,
                                      init_mode="full", seed=seed,
                                      cargo_placement=cargo_placement)
        enforce_symmetry(geom)
        return geom

    def warm_start_phi_fields(prev_geom, W_mm, x_front_mm, d_halo_mm):
        # Actually warm-start. This used to rebuild a fresh envelope and throw
        # prev_geom away -- "a documented scope-down; a true phi remap across a
        # resized envelope is a separate task" -- except remap_geometry is that
        # task, it has been written and tested since, and it had no production
        # caller at all.
        #
        # It cost nothing while the optimiser was a no-op: a rebuild and a warm
        # start both produced the same 149 g envelope, because carving never
        # happened. Now that a step removes ~7 g, discarding the field means
        # every d_halo after the first restarts from the envelope and every CFD
        # solve spent carving the previous one is thrown away. With six
        # iterations per d_halo that is most of the run.
        #
        # Measured on a carved car at 1 mm, four steps in: 97.0% of the carved
        # mass survives a remap to d_halo=16 and 94.6% to d_halo=43.72, with
        # |grad phi| = 1.000, a watertight single-body extraction, and a further
        # update that still removes material. The residual few percent is the
        # halo pocket moving, which is the point of changing d_halo.
        #
        # Falls back to a rebuild rather than failing the candidate: a remap can
        # legitimately refuse (the cargo has to fit the new pocket), and losing
        # the warm start is much cheaper than losing the d_halo.
        # When a Stage-1 seed exists, warm-start from THAT rather than from the
        # previous d_halo. Chaining is what made the 2026-08-01 sweep's ranking
        # meaningless: each group inherited its predecessor's carve, so the
        # winner had simply been carving longest. Seeding from a common ancestor
        # keeps the cost saving and restores a fair comparison.
        _src = seed_geometry if seed_geometry is not None else prev_geom
        try:
            import copy as _copy
            geom = remap_geometry(_copy.deepcopy(_src),
                                  W_mm=W_mm, x_front_mm=x_front_mm,
                                  d_halo_mm=d_halo_mm,
                                  cargo_placement=cargo_placement)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"warm start to d_halo={d_halo_mm} could not remap the evolved "
                f"field ({type(exc).__name__}: {exc}); rebuilding a fresh "
                f"envelope instead. This candidate starts from scratch and the "
                f"CFD solves spent carving the previous one are lost.",
                RuntimeWarning, stacklevel=2)
            geom = build_unified_geometry(W_mm, x_front_mm, d_halo_mm,
                                          init_mode="full",
                                          cargo_placement=cargo_placement)
        enforce_symmetry(geom)
        return geom

    def perturb_phi_fields(geom, seed, amplitude):
        from evolutionary import perturb_phi_array
        geom.phi.grid = perturb_phi_array(geom.phi.grid, seed=seed, amplitude=amplitude)
        geom.phi.apply_hard_constraints()
        enforce_symmetry(geom)
        return geom

    def run_quality_gates(geom, candidate_id, run_out_dir):
        os.makedirs(run_out_dir, exist_ok=True)
        try:
            mesh, report = extract_unified_surface(geom, allow_inaccessible=True)
        except Exception as exc:  # noqa: BLE001
            return GateOutcome(
                lifecycle_state="geometry_rejected", phi_snapshot_paths={},
                stl_path=None, stl_half_path=None,
                failure_reason=f"unified extraction failed: {exc}", meshes=None,
            )
        stl_path = os.path.join(run_out_dir, f"{candidate_id}_full.stl")
        stl_half = os.path.join(run_out_dir, f"{candidate_id}_half.stl")
        mesh.export(stl_path, file_type="stl_ascii")
        # Decimate ONLY the half-STL — it is the one that goes to OpenFOAM.
        # The full STL is a deliverable/inspection artefact and keeps full
        # marching-cubes fidelity.
        half_mesh_cfd = _decimate_for_cfd(extract_half_surface(geom), stl_triangle_budget)
        half_mesh_cfd.export(stl_half, file_type="stl_ascii")
        # Report the quality of the mesh OpenFOAM actually receives.
        #
        # The triangle-quality gate in surface_extraction runs on the FULL
        # marching-cubes mesh. This decimated half is what goes to
        # snappyHexMesh, nothing measured it, and it is far worse -- measured on
        # real split6 output:
        #     dhalo16 iter1       120,000 tris  min angle 0.10 deg  6.44% < 10 deg
        #     dhalo43.72 iter6    240,000 tris  min angle 0.04 deg  0.90% < 10 deg
        # Both meshed and solved cleanly (converged=True, Mesh OK, 0 illegal
        # faces), as has every candidate for the life of the project.
        #
        # So the 10 deg gate is roughly two orders of magnitude stricter than
        # anything this pipeline has ever needed, AND it is checking a different
        # artefact than the one that matters. That is worth knowing rather than
        # acting on blind: the number is logged here per iteration so the
        # threshold can be set from evidence instead of assumption. Note the
        # second row also shows decimation backing off -- 240,000 triangles
        # against a 120,000 budget -- which _decimate_for_cfd does deliberately
        # when reduction would break watertightness.
        try:
            _a = np.degrees(half_mesh_cfd.face_angles)
            _a = _a[np.isfinite(_a) & (_a > 0)]
            print(f"[cfd-stl] {candidate_id}: {len(half_mesh_cfd.faces):,} tris "
                  f"(budget {stl_triangle_budget:,})  min angle {_a.min():.2f} deg  "
                  f"{100.0 * (_a < 10.0).mean():.2f}% below 10 deg  "
                  f"watertight={half_mesh_cfd.is_watertight}", flush=True)
        except Exception:  # noqa: BLE001 -- a report must never fail the gate
            pass
        snap = geom.phi.save(candidate_id, run_out_dir)
        # The record contract wants the four component keys; the unified field
        # is one file, so all four point at it (single-field snapshot).
        snaps = {k: snap for k in _COMPONENT_KEYS}
        # allow_inaccessible carries a manufacturing penalty rather than
        # rejecting -> a repaired-but-valid candidate, per the spec's failure
        # table (large accessibility failure = penalty, continue).
        # A per-COMPONENT dict {name: mm^2}, not a scalar -- the old
        # `if report["inaccessible_area_mm2"]` was testing a dict for
        # truthiness, which is why it never noticed. Summed for the record;
        # the breakdown stays in the extraction report.
        _areas = report["inaccessible_area_mm2"]
        inaccessible = float(sum(_areas.values())) if isinstance(_areas, dict)             else float(_areas or 0.0)
        state = "geometry_repaired" if inaccessible else "valid_simulated"
        return GateOutcome(
            lifecycle_state=state, phi_snapshot_paths=snaps,
            stl_path=stl_path, stl_half_path=stl_half,
            failure_reason=None, meshes={"car": mesh},
            inaccessible_area_mm2=inaccessible,
        )

    def compute_mass_report(geom):
        machined = compute_mass_com(geom)
        full = ingest_mass_com(machined, fixed_hardware_for(geom),
                               propellant_mass_kg=_propellant_kg)
        return MassReport(
            total_mass_kg=full.total_mass_kg, com_x_m=full.com_x_m,
            com_y_m=full.com_y_m, com_z_m=full.com_z_m,
            propellant_mass_kg=full.propellant_mass_kg,
            propellant_com=tuple(full.propellant_com),
        )

    def run_cfd(stl_half_path):
        half, health = run_half_car_cfd(stl_half_path, **cfd_kwargs)
        full = half.to_full_car()
        return CFDOutcome(D20=full.D20, L=full.L, Cm=full.Cm, A=full.A,
                          converged=health.converged,
                          residual_final=health.residual_final,
                          force_oscillation=health.force_oscillation,
                          force_mean_stderr=getattr(health, "force_mean_stderr", None),
                          force_drift=getattr(health, "force_drift", None))

    def evaluate_objective(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        p = _params(D20, L, m_total, h_com, x_com, mu, wheel_moi)
        T_raw, T_pen, grads = race_value_and_grad_guarded(p, model)
        return ObjectiveOutcome(T_raw=T_raw, T_com_penalized=T_pen, gradients=grads)

    def compute_adjoint_weight(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        p = _params(D20, L, m_total, h_com, x_com, mu, wheel_moi)
        return float(compute_adjoint_objective_weight(p, model))

    def run_adjoint(stl_half_path, objective_weight):
        import trimesh
        sensitivity = run_half_car_adjoint(stl_half_path, objective_weight, **adjoint_kwargs)
        half_mesh = trimesh.load(stl_half_path, process=False)
        if len(half_mesh.vertices) != len(sensitivity):
            raise ValueError(
                f"adjoint returned {len(sensitivity)} sensitivities but "
                f"{stl_half_path} has {len(half_mesh.vertices)} vertices."
            )
        return AdjointOutcome(sensitivity=sensitivity, half_mesh=half_mesh)

    def update_phi(geom, sensitivity_field, half_mesh, dt, weights,
                   objective_gradients, mass_report):
        apply_adjoint_to_unified(
            geom, sensitivity_field, half_mesh, dt,
            {"w_aero": weights.w_aero, "w_mass": weights.w_mass,
             "w_com": weights.w_com, "w_mfg": weights.w_mfg},
            objective_gradients, mass_report,
        )

    def write_record(outcome: dict) -> str:
        # Drop keys CandidateRecord does not declare rather than raising.
        #
        # `CandidateRecord(**outcome)` meant any new key in the inner loop's
        # payload broke persistence entirely, and inner_loop swallowed the
        # TypeError -- so the record layer was dead and silent. It failed first
        # on x_front_mm, then on stability_notes; both are now real fields, but
        # the pattern would repeat on the next addition.
        #
        # Unknown keys are reported once, because silently dropping data is the
        # failure mode this whole area already had.
        import dataclasses as _dc
        known = {f.name for f in _dc.fields(CandidateRecord)}
        extra_keys = set(outcome) - known
        if extra_keys:
            import warnings
            warnings.warn(
                f"candidate record: dropping unrecognised field(s) "
                f"{sorted(extra_keys)}. Add them to CandidateRecord if they "
                f"are worth persisting.", RuntimeWarning, stacklevel=2,
            )
        return write_candidate_record(
            CandidateRecord(**{k: v for k, v in outcome.items() if k in known}),
            out_dir,
        )

    bindings = PipelineBindings(
        initialize_phi_fields=initialize_phi_fields,
        warm_start_phi_fields=warm_start_phi_fields,
        perturb_phi_fields=perturb_phi_fields,
        run_quality_gates=run_quality_gates,
        compute_mass_report=compute_mass_report,
        run_cfd=run_cfd, evaluate_objective=evaluate_objective,
        compute_adjoint_weight=compute_adjoint_weight, run_adjoint=run_adjoint,
        update_phi=update_phi, write_candidate_record=write_record,
    )
    validate_bindings(bindings)
    return bindings
