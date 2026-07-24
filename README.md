# Part 3 Simulation - Optimizer Workflow

> 📐 **Whole-project architecture:** see [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
> Part 3 owns orchestration: the `PipelineBindings` handshake, the per-candidate
> inner φ loop (forward CFD + adjoint + Hamilton-Jacobi update), the outer sweep,
> and candidate records. ⚠️ The outer sweep currently sweeps **W** (grid); the
> target two-stage flow sweeps **d_halo** (see ARCHITECTURE.md §8).

This repository contains the Part 3 optimizer layer for the STEM Racing CFD workflow. It is designed to sit beside:

- `part1-simulation`: generative geometry, level-set fields, quality gates, STL export, mass/COM extraction.
- `part2-simulation`: CFD wrapper, mesh validation, race objective, adjoint/objective contracts, candidate records.

The intended local workspace layout is:

```text
NEW CFD/
  part1-simulation/
  part2-simulation/
  part3-simulation/
```

## What Part 3 Owns

Part 3 owns the search policy, not the physics itself. The main responsibilities are:

- enforce optimizer constants and candidate lifecycle states in `optimizer_contract.py`
- define the Part 1/Part 2 handshake in `pipeline_interface.py`
- combine normalized gradient terms and calibrate starting weights in `gradient_combiner.py`
- compose the penalized search objective while preserving raw final ranking in `objective_policy.py`
- run the inner optimization loop in `inner_loop.py`
- track convergence in `convergence.py`
- manage evolutionary survivor selection in `evolutionary.py`
- sweep wheelbase values in `wheelbase_sweep.py`
- run finalist robustness checks in `robustness.py`
- coordinate the full search and final deliverables in `orchestrator.py`

## Current Integration Status

This code is a contract-first optimizer scaffold. It compiles and its local smoke tests run without OpenFOAM, JAX, or the full Part 1/Part 2 pipeline. The real optimizer run still depends on production bindings from Part 1 and Part 2.

Important unresolved integration points:

- `pipeline_interface.real_bindings().initialize_phi_fields` is intentionally `NotImplementedError` until Part 1 exposes a single factory that builds all four phi grids for a given `(W_mm, d_halo_mm)`.
- `pipeline_interface.real_bindings().warm_start_phi_fields` is intentionally `NotImplementedError` until Part 1 exposes a remap/warm-start path between neighboring wheelbases.
- `run_full_search` refuses to start unless `rtc_validated_against_track_data=True` and `cfd_pipeline_validated_on_known_geometry=True`; those flags represent real experiments, not code-only checks.
- The folder path support now includes both spec-style names (`part1_geometry`, `part2_simulation`) and your current repo names (`part1-simulation`, `part2-simulation`).

## Main Entry Point

The top-level orchestration entry point is:

```python
from orchestrator import run_full_search
```

Typical flow:

1. Build or inject `PipelineBindings`.
2. Create an `OptimizerConfig`.
3. Calibrate `GradientWeights`.
4. Call `run_full_search(...)`.
5. Read `SearchResult.final_deliverables(out_dir)`.

The code deliberately fails loudly when the real pipeline is not wired. Do not replace those failures with dummy values; that would only make the optimizer produce confident garbage.

## Run Tests

From this folder:

```powershell
python run_all_tests.py
```

You can also run Python's syntax check:

```powershell
python -m compileall .
```

## Files

| File | Purpose |
|---|---|
| `optimizer_contract.py` | Shared constants, optimizer config, gradient weights, penalties, candidate outcome lifecycle validation |
| `pipeline_interface.py` | Single handshake layer between Part 3 and Part 1/Part 2 |
| `objective_policy.py` | Penalized search ranking vs raw final ranking |
| `gradient_combiner.py` | Unit-RMS gradient normalization and initial weight calibration |
| `convergence.py` | Inner-loop stopping state machine |
| `inner_loop.py` | Per-candidate optimizer loop |
| `evolutionary.py` | Kill/perturb survivor policy and failure-region memory |
| `wheelbase_sweep.py` | Coarse/refined wheelbase search helpers |
| `parallel_runner.py` | Parallel candidate execution wrapper |
| `robustness.py` | Finalist robustness checks |
| `stability_check.py` | Stability checks for candidate behavior |
| `orchestrator.py` | Full Part 3 search strategy and final deliverables |
| `PROJECT_AUDIT_ALL_PARTS.md` | Existing supplied audit notes across Parts 1-3 |

## Brutal Reality Check

Part 3 is not a magic optimizer yet. It is the control layer. If Part 1 exports broken geometry, or Part 2 accepts mocked/unconverged CFD, Part 3 can only rank bad data more efficiently. The production readiness blockers are the real Part 1/Part 2 interfaces, CFD validation, and physical calibration.
