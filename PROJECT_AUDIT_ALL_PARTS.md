# STEM Racing CFD Optimization — Full Project Audit (Parts 1–3)

> **⚠️ DATED 2026-07-12 — several findings are now resolved or superseded (banner
> added 2026-07-24). Read `../ARCHITECTURE.md` for the current state.** Notably:
> finding **N-20 ("no frontal-area computation exists anywhere")** is false —
> `compute_frontal_area_half` exists. The audit also predates: the two-stage
> architecture (Stage 1 no-CFD Bayesian over W+x_front; Stage 2 per-`d_halo`
> CFD sweep), the corrected `d_halo < W-34` bound, the virtual-cargo fore-aft
> flip, and the mandatory ballast-container void. It still describes the outer
> loop as a single 3-scalar BO with nested adjoint. Part 2's CFD/adjoint wrappers
> are now REAL (not the stubs some sections describe). See
> `../SESSION_CHANGES_2026-07-24.md`.

**Date:** 2026-07-12
**Scope:** Part 1 (generative geometry, under work), Part 2 (simulation setups, completed), Part 3 (optimizer, built in this session), and every interface between them.
**Method:** Static review of all retrievable source against the three governing MD specs and the Part 1 SPEC.txt interface contract (§16) and signature reference (§22). Where a finding was quantifiable, I reproduced it numerically in an isolated Python environment rather than trusting inspection — those findings carry reproduction numbers. Part 3 was built and its 24 tests executed to green in this session; its audit section is a self-audit and is deliberately the harshest.

Severity scale: **CRITICAL** (wrong physics or broken pipeline), **HIGH** (will bite during integration or corrupt results silently), **MEDIUM** (correctness or contract gap with contained blast radius), **LOW** (drift, hygiene, doc rot).

---

## 1. Executive summary

The project is architecturally sound: the contract-first layering (geometry_contract / physics_contract / optimizer_contract), the locked race objective with SHA-256 pinning, the 8-state candidate lifecycle, and the loud-placeholder rule are all genuinely good engineering discipline and are mostly followed.

The dangerous problems are not inside any one part. They sit **on the seams**:

1. **The STL handshake between Part 1 and Part 2 is broken as written.** Part 1 assembles STLs with trimesh, whose STL export defaults to **binary**; Part 2's CFD wrapper hard-rejects binary STL (ASCII-only parser). First real integration run dies at the door. (HIGH, §5.1)
2. **The half-car domain check is claimed but absent.** SPEC.txt §16 promises Part 2 validates the right-half STL ("all vertices y ≥ −1e-6"); no such check exists in `cfd_wrapper.py`. A full-car STL passed by mistake would sail through and get its forces **doubled** by `to_full_car()` — a silent 2× drag error, the worst kind. (HIGH, §5.2)
3. **Non-converged CFD results flow into the race objective unimpeded.** `run_half_car_cfd` returns unconverged forces as ordinary numbers (`converged=False` on a health report nothing consumes). Part 3 now closes this with a hard convergence gate (`require_cfd_convergence=True`), but Part 2 should not have left it open. (HIGH, §4, N-5)
4. **The lift gradient is computed and then dropped.** The adapter produces `dT_dL`, but `package_gradient_bundle` has no slot for it — the contract keys are exactly `{w_D20, dT_dmass, dT_dh_com, dT_dx_com, manufacturing_gradient}`. Lift sensitivity dies at the packaging boundary. Part 3 carries it separately in `ObjectiveOutcome`, but the Part 2 bundle contract needs the slot. (MEDIUM-HIGH, §4, N-8)
5. **Fabricated placeholder physics is not flagged as such.** The `com_x` penalty (k=0.001 s/m², target = front axle) is invented, violates the project's own Rule 8 (unknowns must be loud), and — worse — the unit-RMS gradient normalization in the combiner will **inflate that meaningless gradient to the same magnitude as the real aero gradient**. (HIGH in combination, §4, N-2)

Everything else is enumerated below. Part 2's internals are otherwise in good shape after the prior fix rounds: the locked objective's math checks out, the adapter guards are real, the mesh-validation injection fix is correct.

---

## 2. Coverage table

| Artifact | Reviewed | How |
|---|---|---|
| 01_generative_geometry.md | full | read from disk |
| 02_simulation_setups.md | full | read from disk |
| 03_optimizer_workflow.md | full | read from disk |
| Part 1 SPEC.txt §16, §22 | full | project knowledge |
| Part 1 code (quality_gates, phi_grid, mass_com_calculator, phi_updater, geometry_contract) | partial — signatures + retrieved chunks | project knowledge (Part 1 is under work; audited at contract level) |
| Part 2: physics_contract.py | full | project knowledge |
| Part 2: mass_com_ingest.py | full | project knowledge |
| Part 2: cfd_wrapper.py | full | project knowledge |
| Part 2: mesh_validation.py | near-full (solver-agreement / speed-sensitivity helpers partially retrieved) | project knowledge |
| Part 2: calibration.py | full | project knowledge |
| Part 2: race_objective.py (locked, SHA 6ed47bb6…) | full | project knowledge + numeric reproduction |
| Part 2: race_objective_adapter.py | full | project knowledge + numeric reproduction |
| Part 2: adjoint_contract.py | full | project knowledge |
| Part 2: candidate_record.py | full | project knowledge |
| Part 2: run_all_tests.py | **not retrieved** — D1-fix status unverified (see N-22) | — |
| Part 3 (13 modules + 4 test files) | full | written and executed this session |

Note: you mentioned the files live in a GitHub repo. Everything above came from the project knowledge snapshot. If the repo has moved past that snapshot, share the URL and I'll diff the audit against head.

---

## 3. Part 1 audit (spec + contract level — code is under work)

Part 1 was explicitly left aside for deep code review per your instruction, but its **contracts** are load-bearing for Parts 2–3, so those were audited:

**P1-1 (HIGH, interface).** `assemble_stl` via trimesh: trimesh's `export(file_obj, file_type="stl")` writes **binary** STL unless `file_type="stl_ascii"` is requested. Part 2 requires ASCII (see §5.1). One line in Part 1 fixes the whole seam — but it must be a *contractual* line, tested by the integration test, not an accident of a default.

**P1-2 (MEDIUM).** `run_quality_gates` promises `phi_snapshot_paths` "always populated" including on failure — good, Part 3's failure-region memory depends on it. But the GateResult contract does not pin *where* snapshots go relative to `out_dir`, and Part 2's candidate record requires exactly the four keys `nose/sidepod/rearpod/main_body`. Recommend the GateResult docstring pin both (key set and path layout) so Part 3 can rely on them without defensive re-checks.

**P1-3 (MEDIUM).** `update_phi(phi_grids, right_half_sensitivity, right_half_mesh, dt, gradient_weights)` — the signature takes a single sensitivity field, but the spec's gradient combination has four terms (aero, mass, COM, mfg), of which mass/COM/mfg are *scalar chain-rule* gradients through volume integrals, not surface fields. Where does `dT_dmass` enter `update_phi`? The signature has no slot for the objective's scalar gradients. Part 3's binding passes them anyway (extra args in its own wrapper) but the Part 1 signature must grow slots or define that mass/COM gradients are derived internally from φ. **This is the biggest unresolved design question in Part 1** and it blocks a correct first optimization run.

**P1-4 (LOW).** `combine_gradients` normalizes each gradient to unit RMS with a 1e-12 floor — matches spec. See N-2 for why this interacts badly with placeholder gradients.

**P1-5 (LOW, doc).** SPEC.txt §16 says COM-height violation costs "10^15 s"; Part 2's adapter now raises ValueError instead. Stale doc (also logged as N-17).

---

## 4. Part 2 deep audit — findings register

Status of prior audit rounds first: findings #1–#17 from PART2_AUDIT_CATEGORIZED are fixed in the current snapshot with two exceptions — #4 (dead CFD parameters) still open (now N-15), and the run_all_tests D1 fix unverifiable from the snapshot (N-22).

### New findings (this audit)

**N-1 (MEDIUM) — T_raw is contaminated by the clamp exactly at the optimum.** *Numerically verified.* The locked COM quartic (`polyfit` degree 4 through the 9 placeholder points) has an in-domain **negative dip**: minimum **−3.792e-05 s at δ = +0.411 mm**, negative over δ ∈ (0, 0.823] mm. The locked objective *adds* the raw (possibly negative) penalty into T_penalized; the adapter recovers T_raw by subtracting `max(penalty, 0)`. In the dip, it subtracts 0, so **T_raw inherits up to 37.9 µs of polyfit artifact** — precisely in the region a converged optimizer will sit (just above the 30 mm reference). Final ranking is T_raw-only, so this is direct ranking noise between finalists. Fix options: (a) subtract the *unclamped* penalty in the adapter (exact recovery), or (b) refit the penalty with a monotone/convex-constrained form once real ballast data exists. Given the file is locked, (a) is an adapter-side one-liner.

**N-2 (HIGH in combination) — the com_x penalty is fabricated physics, unflagged, and normalization will amplify it.** `k_x = 0.001 s/m²` with target `x_com = 0.0` — the **front axle** per the mass_com_ingest origin convention. Reproduced magnitudes: at a realistic x_com = 65 mm, penalty = 4.2 µs and gradient = 1.3e-4 s/m, pulling the COM *toward the front axle* for no physical reason. Alone it's noise. But the spec's combiner normalizes every gradient term to **unit RMS** before weighting — a meaningless-but-nonzero gradient direction gets inflated to parity with the real aero gradient inside the COM term. This violates the project's own Rule 8 (invented values must be loud). Fix: either raise `NotImplementedError` for com_x sensitivity until the ballast experiment lands, or set the term to exactly zero with a `? UNRESOLVED` comment, and keep `w_com` small until calibrated (Part 3's `calibrate_gradient_weights` exists for exactly this).

**N-3 (MEDIUM) — ADJOINT_HALF_CAR_SCALING = 0.5 is arguably inverted.** The objective differentiated by OpenFOAM is `0.5 · w_D20 · D20_half`, giving surface sensitivity `0.5 · w · ∂D20_half/∂S`. But the quantity the optimizer needs is `∂T/∂S` under *symmetric* application: `D20_full = 2·D20_half`, so a mirrored pair moving together gives `∂T/∂S_sym = w · 2 · ∂D20_half/∂S`. The implemented sensitivity is a factor **4** below that, or 2 below the per-half convention. Because Part 1's combiner RMS-normalizes, the *scale* washes out of the φ update — but it does **not** wash out of anything that consumes unnormalized magnitudes, e.g. sensitivity-based weight calibration or convergence-by-gradient-norm. Either adopt 2.0 with a derivation comment, or keep 0.5 and document loudly that adjoint magnitudes are convention-scaled and must never be compared to RTC gradient magnitudes.

**N-4 (HIGH) — claimed half-domain check is absent.** SPEC.txt §16: right-half STL "all vertices y ≥ −1e-6, Part 2 raises CFDRunError otherwise." `run_half_car_cfd` performs: existence → ASCII parse → watertight edge count → pipeline. **No y-coordinate check anywhere.** A full-car STL (an easy upstream mistake — Part 1 emits both `stl_path` and `stl_half_path`) passes silently, and `to_full_car()` doubles its already-full forces: **2× drag, invisible**. Fix: five lines in the STL parse loop; reject `min(y) < −1e-6` with a CFDRunError naming the offending vertex.

**N-5 (HIGH) — non-converged CFD is not a gate anywhere in Part 2.** `converged = residual ≤ 1e-3` is recorded on the health report and nothing consumes it; forces are returned normally. Part 3 now enforces `require_cfd_convergence=True` (a non-converged run becomes lifecycle `CFD_failed` for that iteration). Part 2 should still document that its own API is unguarded, or raise on request via a strict flag.

**N-6 = P1-1 (HIGH) — binary STL handshake break.** See §5.1.

**N-7 (LOW) — exact-float edge matching in the watertight check.** Edges are matched on exact vertex tuples. Marching-cubes output with consistent f-string formatting will match, but any upstream re-serialization that perturbs the 17th digit produces a false "not watertight". Robustness suggestion: quantize vertices (e.g. round to 1e-9 m) before edge counting.

**N-8 (MEDIUM-HIGH) — dT_dL is dropped at the packaging boundary.** The adapter returns `dT_dL` (lift-dependent friction is in the locked model), but `package_gradient_bundle`'s key set has no lift slot and `get_active_objective_weights` returns `w_L = 0` (correct for now) with no path to carry the RTC's lift sensitivity when `w_L` activates. Add the key now (value flowing, weight zero) so activating lift later is a weight change, not a contract change.

**N-9 (MEDIUM) — candidate records silently overwrite.** `write_candidate_record` with a repeated candidate_id overwrites the JSON — per-iteration history loss, contradicting "failed runs are not discarded". Part 3 works around it with `_iterNNNN` suffixes; Part 2 should still refuse or version on collision.

**N-10 (MEDIUM, Windows) — path-traversal guard misses drive-relative IDs.** The guard rejects `/`, `\\`, `.`, `..` — but `C:evil` contains none of those and on Windows resolves drive-relative, escaping `out_dir`. The project runs on Windows (build logs use PowerShell paths). Reject `:` and reserved device names (CON, NUL, …).

**N-11 (MEDIUM) — CandidateRecord has no lifecycle/value invariants.** `valid_simulated` with `T_raw=None`, or `CFD_failed` with `failure_reason=None`, both construct fine. Part 3's `CandidateOutcome` now enforces these invariants (success ⇒ both times present and `T_pen ≥ T_raw`; failure ⇒ reason present); mirroring them into Part 2's record would stop bad records at the source.

**N-12 (MEDIUM) — the COM-height hard guard turns a smooth landscape into a wall.** `COM_HEIGHT_FIT_RANGE_M = (0.018, 0.042)` raises ValueError. The guard is necessary (the quartic extrapolates to garbage: reproduced **+9.2 ms at h = 0.05 m** — positive, so clamping would not catch it). But a legitimately tall design gets `objective_failed` with **no gradient pointing back into range**. Recommend linear extension of the penalty beyond the fit range (continuous value + slope), keeping the hard raise only far outside (e.g. beyond ±2× the fit range).

**N-13 (MEDIUM) — two thrust-CSV schemas that cannot read each other.** `calibration.fit_thrust_surrogate` wants `time_s`/`thrust_N`; the locked objective wants `time (s)`/`force (N)`/`mass (kg)`. One physical CSV cannot feed both. Since the locked file wins, either retire calibration's fitter or make it a documented pre-processor emitting the locked schema.

**N-14 (MEDIUM) — the pitching-moment reference point is pinned nowhere.** `Cm` flows through the whole system, but no constant defines the moment reference point (CofR) the OpenFOAM `forceCoeffs` dict must use, and `ref_length = √A` is itself a flagged stand-in. When someone wires OpenFOAM they will pick a CofR silently. Add `MOMENT_REFERENCE_POINT_M` to physics_contract now, even with a `? UNRESOLVED` value. Part 3's Tier-2 stability check refuses to run while `cm_ref_length_is_placeholder=True` for exactly this reason.

**N-15 (MEDIUM, carried from prior #4) — the CFD wrapper's signature advertises dead parameters.** `reference_speed_mps, air_density_kgm3, max_iterations` are `del`'d on entry. The planned 5 m/s vs 20 m/s CdA-variation validation cannot be expressed through this API, and the `D20` field name cannot honestly hold a 5 m/s result. Either honor the parameters when OpenFOAM lands or remove them and add a separate `run_half_car_cfd_at_speed`.

**N-16 (LOW) — ingest_mass_com does not enforce the handshake it claims.** Component names (`nose/sidepod/rearpod/main_body`) and the sidepod-pair convention (`com_y = 0`) are stated as hard constraints in §16 but unchecked in code.

**N-17 (LOW, doc)** — §16's "violation = 10^15 s" is stale; the adapter raises. Update the doc.

**N-18 (LOW)** — mesh-independence spec says "currently 10 %, target 5 %"; code implements only the 5 % target with no representation of the interim tolerance.

**N-19 (LOW)** — T_raw is not purely physics: the locked model's `low_speed_penalty` smoothing term stays inside it. Sub-millisecond, but worth a comment where T_raw is defined, since final ranking is T_raw-only.

**N-20 (MEDIUM, forward-looking)** — no frontal-area computation exists anywhere in the project. `A` arrives from the (placeholder) CFD result dict; the spec says it comes from geometry projection. Someone must own `compute_frontal_area(mesh)` — natural home is Part 1's mesh utilities.

**N-21 (LOW)** — `race_objective` flips `jax.config.update("jax_enable_x64", True)` at import: a process-global side effect that can silently retrace any other JAX code in the process.

**N-22 (VERIFY)** — Part 2's `run_all_tests.py` could not be retrieved; whether the D1 fix (stderr surfacing, exit-code discipline) landed is unverified. Part 3's runner implements the full Rule-7 contract as reference.

---

## 5. Cross-part integration risks (the seams)

**5.1 Binary vs ASCII STL (P1-1/N-6, HIGH).** Part 1 → trimesh → binary STL by default; Part 2 → ASCII-only parser → hard reject. Fix in Part 1 (`file_type="stl_ascii"`), assert in the Part 1↔2 integration test by byte-sniffing the emitted file (`solid ` prefix).

**5.2 Missing y ≥ −1e-6 check (N-4, HIGH).** The one check that distinguishes a half-car from a full-car input does not exist. Until it does, every consumer of `to_full_car()` is one wrong path away from doubled forces.

**5.3 update_phi's missing scalar-gradient slots (P1-3, blocks first real run).** The Part 1 signature accepts only the surface sensitivity; mass/COM gradients from the RTC have no entry point. Decide the design (extra parameters vs internal derivation) before wiring the adjoint.

**5.4 Gradient bundle loses lift (N-8).** Add `dT_dL` to the bundle contract now.

**5.5 Cm convention (N-14).** Pin `MOMENT_REFERENCE_POINT_M` and the eventual real `ref_length` in physics_contract, version the convention in candidate records so pre/post-change records are never compared.

**5.6 Record ID discipline (N-9/N-10).** Part 3 emits `{candidate}_iterNNNN` IDs; Part 2 should reject collisions and drive-relative names so no other caller can regress this.

---

## 6. Part 3 — what was built, and its self-audit

**Delivered (13 modules + runner + 4 test files, 24 tests, all green):**

`optimizer_contract.py` (constants, `OptimizerConfig` with the two spec-prerequisite flags, `GradientWeights`, `PenaltyInputs` with no silent defaults, `CandidateOutcome` with enforced lifecycle/value invariants) · `pipeline_interface.py` (the full Part 1/2 handshake as injectable `PipelineBindings`, plus `real_bindings()` wiring the actual packages with loud `? UNRESOLVED` placeholders for the three genuinely unwired integrations: φ-field factory, warm-start remap, OpenFOAM adjoint) · `objective_policy.py` (T_penalized composition; **search ranking by T_penalized with finite 1e6 s failure penalty; final ranking strictly T_raw among valid states; failures never ranked as "slow cars"**) · `gradient_combiner.py` (unit-RMS combine mirroring Part 1's spec math byte-for-byte, plus `calibrate_gradient_weights` implementing the spec's sensitivity-analysis procedure via central differences) · `convergence.py` (all four spec stop criteria; failures don't reset the last-good T; successes reset the failure streak) · `stability_check.py` (Tier-1 static wheel loads always; Tier-2 aero-corrected loads + CoP margin gated on the three rolling-friction prerequisites AND refusing to run against the √A Cm stand-in) · `inner_loop.py` (the 15-step loop; **every failure maps to a lifecycle state, gets a record with φ snapshots, and never crashes the loop**; the CFD-convergence gate; iteration-suffixed record IDs) · `evolutionary.py` (kill-bottom-50 % with ceil-keeps-extra, smooth Gaussian-through-box-blur φ perturbation scaled to 10 % of φ RMS, failure-region memory keyed on (W, d_halo) retaining φ snapshot paths) · `parallel_runner.py` (order-preserving, per-task exception isolation, max_workers=1 fully deterministic) · `wheelbase_sweep.py` (21-value coarse grid, ±1 mm/0.5 mm refined grids, per-W evolutionary rounds, ascending-W warm-start chain) · `robustness.py` (all six spec checks; the two needing re-mesh/re-CFD are injected hooks that report **not_run with a reason** when absent — never silently skipped) · `orchestrator.py` (refuses to start unless both validation prerequisites are asserted; coarse → top-5 → refined → robustness → build-candidate selection; `final_deliverables` raises rather than shipping an empty search) · `run_all_tests.py` (Rule 7: exit 0/1/2, stderr surfaced verbatim).

**Self-audit — honest limitations:**

**P3-1.** Three bindings in `real_bindings()` are loud placeholders by necessity: `initialize_phi_fields` (Part 1 has no all-four-grids factory yet), `warm_start_phi_fields` (needs bounding-volume recompute + `PhiGrid.remap`), `run_adjoint` (OpenFOAM adjoint unwired project-wide). Part 3 cannot run end-to-end against reality until those land; it runs end-to-end against fakes today, which is what the integration test proves.

**P3-2.** The failure-region memory keys on (W, d_halo) only; φ-space similarity is stored (snapshot paths retained) but not yet used as a distance. The spec's ambition is bigger; the storage is future-proofed, the metric is not implemented.

**P3-3.** The per-W evolutionary cadence is implemented as budget slices (`evolution_interval_iters` per round) rather than a shared-clock interrupt across parallel loops — behaviorally equivalent to "after every N inner iterations" without cross-thread synchronization, but it is a reading of the spec, recorded here deliberately.

**P3-4.** `zero_penalties` is the default manufacturing/rule-margin provider because Part 1 does not yet emit penalty magnitudes for repaired geometry. It is a named function precisely so it shows up in review; it must not silently become permanent.

**P3-5.** Adjoint/φ-update failure *after* a successful objective evaluation is mapped to `objective_failed` with the measured times embedded in the failure reason — a pragmatic choice since the lifecycle enum has no "update_failed" state. If you want that state, it's an enum addition across all three parts.

---

## 7. Next steps — specific, ordered, with acceptance criteria

**Step 1 — Fix the two seam-breakers (½ day).**
(a) Part 1 `assemble_stl`: export ASCII (`file_type="stl_ascii"`); acceptance: integration test asserts the emitted half-STL starts with `solid ` and `run_half_car_cfd` parses it.
(b) Part 2 `cfd_wrapper`: add the y ≥ −1e-6 check inside the STL parse; acceptance: a mirrored full-car STL raises `CFDRunError` naming the min-y vertex; the existing half-car fixture still passes.

**Step 2 — Patch the objective seam (½ day).**
(a) Adapter: subtract the *unclamped* COM penalty when recovering T_raw (kills the 37.9 µs contamination, N-1); acceptance: a regression test at δ = +0.411 mm shows T_raw = T_pen − pen exactly.
(b) Neutralize the fabricated com_x term (zero + `? UNRESOLVED`, or NotImplementedError) until ballast data exists (N-2).
(c) Add `dT_dL` to `package_gradient_bundle` with `w_L` still 0 (N-8).
(d) Adapter-side: replace the hard COM-height raise with linear extension inside a widened soft band, hard raise only far outside (N-12).

**Step 3 — Resolve update_phi's gradient entry points (design decision, do before any adjoint work).** Choose: extend the Part 1 signature with `objective_scalar_gradients: dict` + `mass_report`, or specify that mass/COM field gradients are derived internally from φ. Write it into SPEC.txt §22 and mirror in Part 3's binding. Acceptance: the Part 3 `real_bindings.update_phi` wrapper compiles against the real signature with no `**kwargs` smuggling.

**Step 4 — Land the three Part 1 factories Part 3 is waiting on (the current `? UNRESOLVED` set).** `build_all_phi_grids(W_mm, d_halo_mm, seed)`, warm-start remap (`compute_bounding_volumes` for new W → `PhiGrid.remap`), and only then the OpenFOAM adjoint. Acceptance for each: replace the placeholder in `pipeline_interface.real_bindings`, and the Part 3 integration test gains a real-bindings variant that reaches the next placeholder instead of the current one.

**Step 5 — Contract hygiene sweep (1 day, mechanical).** N-9/N-10 (record collision + Windows IDs), N-11 (record invariants, copy from Part 3's `CandidateOutcome`), N-13 (retire or adapt calibration's thrust schema), N-14 (`MOMENT_REFERENCE_POINT_M` constant), N-16 (enforce component names + sidepod com_y), N-17/N-18 (doc updates), N-22 (verify Part 2's test runner against Rule 7 — compare with Part 3's).

**Step 6 — The two physical experiments the code is blocked on.** These are prerequisites the orchestrator *enforces*: RTC validation against a real track run (sets `rtc_validated_against_track_data=True` honestly) and CFD validation on a known geometry (sets the second flag). Also: the ballast COM experiment, which simultaneously unblocks the real COM penalty curve, the com_x term, and Tier-2 stability.

**Step 7 — Weight calibration, then first real search.** Run `gradient_combiner.calibrate_gradient_weights` against the validated RTC with your measured typical per-iteration changes; feed the resulting `GradientWeights` into `orchestrator.run_full_search` with a small budget smoke run (3 W values, M=2) before the full 21-value sweep.

---

## Appendix — reproduction snippets

COM quartic dip / extrapolation / com_x magnitudes: rebuild `polyfit(deg=4)` over the 9 locked data points, evaluate over δ ∈ [−12, 12] mm → min −3.792e-05 s at +0.411 mm, negative on (0, 0.823] mm; `pen(0.050) = +9.2e-03 s`; `k_x·x²` at 65 mm = 4.2e-06 s, gradient 1.3e-04 s/m. Adjoint factor derivation in N-3. All runnable stand-alone; none require JAX.
