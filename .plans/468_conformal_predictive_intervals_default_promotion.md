# Plan 468: Promote `conformal_predictive_intervals` to DEFAULT-ON

**Filed:** 2026-07-20
**Repo:** katgpt-rs (primary) + riir-ai (doc updates)
**Parent:** [Plan 340](340_conformal_predictive_intervals_primitive.md) — T1.14 deferral lift
**Status:** COMPLETE — T1-T7 all done (2026-07-20)
**Committed:** 2026-07-20 — katgpt-rs `562b9a79`, riir-ai `7f2d8a80b` + `d6448c870` + `9d0716e22` + `a12f23a44`
**Branch:** `develop` (per AGENTS.md)

## TL;DR

Promote the `conformal_predictive_intervals` feature from opt-in to DEFAULT-ON
in `katgpt-rs/crates/katgpt-core/Cargo.toml`. Plan 340 T1.14 deferred
promotion "pending a runtime consumer that demonstrably beats its simpler
heuristic counterpart." Four consumers were probed (Benches 562/563/564/565);
two PASSed — Bench 564 (MCTS collapse) + Bench 565 (Salience Tri-Gate,
**ΔF1 = +0.3145, 6.3× margin**). The deferral condition is satisfied.

Primitive-level G1–G4 GOAT already PASSed (Bench 340, 2026-06-30): coverage
[0.9445, 0.9493], interval_into H=1 642ns (≤ 1µs), 0 allocs, bit-reproducible.
Pure modelless (empirical-quantile calibration over a residual reservoir —
no training, no learned params). Zero runtime cost unless a caller
constructs `ConformalIntervalCalibrator`. Feature is `[]` (no new deps).

Consumer-level feature gates (`karc_conformal_width`, `salience_conformal_width`,
the four probe features in riir-engine) STAY opt-in — promoting the primitive
only removes the katgpt-core-level re-forward; consumers still choose.

## Background — why now

The primitive has been opt-in since 2026-06-30 (Plan 340 Phase 1). The Cargo.toml
comment said:

> Opt-in — primitive-level G1–G4 GOAT PASS, promotion to default-on deferred
> pending a runtime consumer that demonstrably beats its simpler heuristic
> counterpart. The Plan 508 curiosity-detector consumer (riir-ai Bench 562,
> 2026-07-19) FAILED G3. Other consumers (Sleep-Time, MCTS, Salience) remain open.

All four consumers are now closed:

| Consumer | Signal | Bench | Verdict | Margin |
|---|---|---|---|---|
| Curiosity FP filter (row 2) | L2 surprise scalar | 562 | G3 FAIL | wider than 5×EMA |
| Sleep-Time predictability (row 3) | realized curiosity | 563 | G3 FAIL | distribution-level summary loses cycle info |
| MCTS collapse τ (row 4) | per-NPC calibrated τ | 564 | **G3 PASS** | per-NPC σ-scaled threshold |
| **Salience Tri-Gate (row 5)** | **interval-width Delegate nudge** | **565** | **G3 PASS** | **ΔF1 = +0.3145 (6.3× gate margin), ΔFP = −0.8155** |

Plus Plan 513 (2026-07-20) fixed the width-definition semantic bug in
`KarcConformalSidecar::interval_width()` — the G3 PASS verdict was re-verified
bit-identical (F1=0.8724, FP=0.1845) under the corrected full-width definition.

The Cargo.toml language specified "a runtime consumer that demonstrably beats"
(singular). Two did. The condition is satisfied with margin.

## Tasks

- [x] **T1** — Add `conformal_predictive_intervals` to `default = [...]` in
      `katgpt-rs/crates/katgpt-core/Cargo.toml`. Update the feature comment
      to reflect promotion (cite Bench 565 as the trigger; note Plan 513
      width-fix vindication).
- [x] **T2** — Update the `pub mod conformal;` comment in
      `katgpt-rs/crates/katgpt-core/src/lib.rs` to reflect promotion
      (currently says "STAYS OPT-IN").
- [x] **T3** — Validate: `cargo check`, `cargo clippy --all-targets`,
      `cargo test -p katgpt-core --lib` under default features. The
      conformal module's own tests (32 unit + 8 integration) must still
      pass — they're now compiled by default. **DONE: 1720/1720 lib tests
      pass under default features (was 1688 + 37 new conformal tests = 1725,
      5 ignored). Clippy clean. Conformal coverage/reproducibility/alloc
      integration tests all pass.**
- [x] **T4** — Validate no-regression on `--all-features` and `--no-default-features`
      (the latter should still resolve — the feature is now in `default`, but
      `--no-default-features` strips default, so the module is opt-out).
      **DONE: both configs compile clean (merkle_root lesson check passes).**
- [x] **T5** — Update Bench 565 (`riir-ai/.benchmarks/565_conformal_salience_tri_gate_probe.md`)
      with a "Promotion landed" note pointing to Plan 468.
- [x] **T6** — Update Plan 340 promotion-decision section + Bench 340 promotion
      section with "Promotion landed 2026-07-20 per Plan 468" pointer.
- [x] **T7** — Commit katgpt-rs changes (`feat:` prefix), then riir-ai doc
      changes (`docs:` prefix). Per AGENTS.md commit rule.
      **DONE 2026-07-20:** katgpt-rs landed as `562b9a79`
      (`feat(conformal): promote conformal_predictive_intervals to DEFAULT-ON
      (Plan 468)`); riir-ai doc syncs landed as `7f2d8a80b` (Bench 565),
      `d6448c870` (super-goat status), `9d0716e22` + `a12f23a44`
      (feature-gate-audit stale-surface cleanup).

## Non-Goals

- ❌ NO change to consumer-level gates. `karc_conformal_width`,
  `salience_conformal_width`, the four probe features — all stay opt-in.
  Promoting the primitive removes the katgpt-core re-forward friction; it
  does not auto-enable any consumer.
- ❌ NO change to the `KarcConformalSidecar` wrapper. Bench 568 (per-channel
  probe) is closed; the scalar-L2 path is vindicated.
- ❌ NO change to the "Report the Floor" rule. The primitive IS the floor
  (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` m=1); making it
  default-on means every UQ-bearing primitive's GOAT gate has the floor
  available without an opt-in flag. This strengthens the rule.
- ❌ NO Bench 562/563/568 re-open. All four §1.3 consumers are closed; the
  arc is fully resolved per Research 165.

## Risk assessment

**Compile-time cost:** the `conformal/` module is ~1000 LOC across 6 files
(mod, ring, seasonal, metrics, karc_adapter, floor_harness). Adds ~1-2s to
clean katgpt-core builds. Acceptable.

**Binary size:** the module compiles in but is dead code unless a caller
constructs `ConformalIntervalCalibrator`. Linker will strip unused symbols
in release builds.

**Consumer compatibility:** the feature is `[]` (empty — gates only `pub mod`
+ re-exports). Existing consumers that explicitly enable
`katgpt-core/conformal_predictive_intervals` (riir-engine's `karc_conformal_width`,
the four probes, etc.) become redundant but harmless — enabling an already-on
feature is a no-op.

**No `cfg(not(...))` blocks.** Verified via grep: zero matches for
`cfg(not(feature = "conformal_predictive_intervals"))` in katgpt-rs. Promotion
cannot break any "feature-off" code path because none exist.

**Pattern match.** Many existing default-on primitives follow the same shape:
primitive-level G1-G4 PASS + modelless + zero runtime cost unless invoked.
See `tilr_invariant_subspace`, `manifold_erasure`, `ane_fused_chain`,
`poincare_navigator`, etc. The conformal primitive matches all criteria.

## GOAT gate (Plan 468)

This is a *promotion* plan, not a new primitive. The primitive-level GOAT
gate is Bench 340 (PASS). The runtime-consumer GOAT gate is Bench 565 (PASS,
ΔF1 = +0.3145). The gate that this plan itself must pass is **G3
no-regression**: default-feature build + test must produce identical results
to the pre-promotion opt-in build.

- **G1 (correctness)**: N/A — primitive correctness is Bench 340's domain.
- **G2 (perf)**: N/A — primitive perf is Bench 340's domain. The promotion
  itself adds zero runtime cost (the module compiles but is unused unless
  invoked).
- **G3 (no-regression)**: ✅ required — default-feature `cargo test` must
  pass with the conformal module now compiled by default. Specifically,
  the 39 conformal tests (24 Phase 1 + 7 Phase 2 + 8 integration) must
  still pass under default features, AND all other katgpt-core lib tests
  must be unaffected.
- **G4 (alloc-free)**: N/A — Bench 340 G3 already established zero allocs.

## Cross-references

- [Plan 340](340_conformal_predictive_intervals_primitive.md) — parent plan,
  T1.14 deferral language
- [Bench 340](../.benchmarks/340_conformal_goat.md) — primitive-level GOAT
  (G1-G4 PASS)
- [Bench 565](../../riir-ai/.benchmarks/565_conformal_salience_tri_gate_probe.md)
  — the runtime consumer win that triggers promotion (ΔF1 = +0.3145)
- [Bench 564](../../riir-ai/.benchmarks/564_conformal_mcts_collapse_probe.md)
  — the other runtime consumer win
- [Bench 562](../../riir-ai/.benchmarks/562_conformal_curiosity_fp_probe.md)
  — consumer FAIL (curiosity axis)
- [Bench 563](../../riir-ai/.benchmarks/563_conformal_sleep_time_predictability_probe.md)
  — consumer FAIL (sleep-time axis)
- [Bench 568](../../riir-ai/.benchmarks/568_per_channel_conformal_probe.md)
  — per-channel probe (MIXED, door closed for current consumer)
- [Plan 508](../../riir-ai/.plans/508_conformal_curiosity_false_positive_gate.md)
  — closed curiosity consumer; references the stale Cargo.toml comment
- [Plan 513](../../riir-ai/.plans/513_conformal_sidecar_width_definition_fix.md)
  — width-definition fix vindicating Bench 565's G3 PASS
- `Issue 010` — "Report the Floor" rule
  (removed per noise reduction; consolidated in `.benchmarks/010_report_the_floor_consolidated.md`)
