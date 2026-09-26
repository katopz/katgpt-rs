# Plan 610 — `arm_drift_alignment`: per-arm trajectory-aligned curiosity gate for cgsp

**Status:** PLANNED — opt-in feature `arm_drift_alignment`, promote to default only on GOAT pass. Research: `katgpt-rs/.research/591_Trajectory_Aligned_Curiosity.md` (arXiv:2609.30063 extraction); guide: `riir-ai/.research/389_Trajectory_Aligned_Curiosity_Guide.md`.

## Pinned shape (before code)

- New module `crates/katgpt-core/src/cgsp/arm_alignment.rs` (DerivativeCuriosity sibling; keeps files small), feature `arm_drift_alignment = []`.
- `TrajectoryAlignedCuriosity` implements `CuriosityConjecturer` by delegating sampling to an inner `PoolConjecturer` (the `DerivativeCuriosity` pattern) and scoring each sampled arm:
  `d = TemporalDerivativeKernel::observe(&pref_buf)` (per-dim fast−slow, already computed) → `rms_j = ema(|d_j|)` (fixed-size state) → `û_j = d_j/(rms_j+ε)` normalized → per candidate `r̃_k = sigmoid(β·|dot(ĝ_k, û)|)` with `simd_dot_f32`; candidates' `direction` normalized if non-unit.
- Exposes `last_alignment_scores()` (per-arm, for logging + G2/G3 readouts) and `last_interestingness()` (mean, compatibility with the existing telemetry).
- Zero-alloc: fixed-size arrays + caller scratch; no Solver call; bit-identical when the flag is off.
- Module docs must state the honest caveat: behavioral drift ↔ parameter movement is an assumption until the riir-train-side bridge correlation is measured (Plan 420 T3's checkpoints can host it).

## Tasks

- [ ] T1 — module + types + feature gate + `lib.rs` cfg + required-features rows for any new test/bench targets (repo-birth gate discipline); module doc with the pinned claim + signal-diff table from Research 591 **using the VERIFIED neighbor citations only** (AdaS = arXiv:2006.06587, verified 2026-09-26 — optimizer-side step sizing, never a selection gate; LESS 2402.04333 validation-anchored; RHO-LOSS 2202.03258 magnitude-only). The novelty-asserting docstring ships only with verified neighbors — no memory-cited IDs (verdict-review round-3 condition).
- [ ] T2 — the gate math: per-coordinate RMS preconditioner (EMA of |d_j|), drift normalization, per-arm projection + abs + sigmoid; `#[repr(transparent)]`-clean fixed state; unit tests for monotonicity in |cos|, scale-invariance (rescaling d_j must not move ranks).
- [ ] T3 — G1 mechanism gates (planted-truth): synthetic bandit with a planted drift axis — rank-AUC ≥ 0.8 for drifting-family vs stationary arms; **negative control**: pure-noise drift ⇒ flat score distribution across groups; **scale-invariance arm** (from T2, promoted to gate).
- [ ] T4 — G2 discrimination gate: `DerivativeCuriosity` (global norm) MUST FAIL T3's per-arm ranking fixture. If the incumbent passes, fix the fixture, never weaken the bar.
- [ ] T5 — G3 loop A/B (Bench-950 paired pattern): CGSP loop with (a) trajectory-aligned, (b) global-norm incumbent, (c) uniform sampling — matched budget, paired runs, bit-identical-when-off pin; primary readout cycles-to-recovery / reward-accumulation ratio with CIs; honest-null clause recorded if flat.
- [ ] T6 — G4 alloc/perf: counting allocator zero steady-state allocs; ns/decision ≤ 2× incumbent `observe_interestingness` (the added work is B dot products of dim d); `--release` gates only.
- [ ] T7 — docs: README feature row + count bump, `.research/591` status update with measured gate results, guide 389 P0/P1 checkbox updates; clippy `-D` at default + `--features arm_drift_alignment`; full gate script per AGENTS.md.

## Demotion clause

If G3 reads flat (aligned ≈ global-norm ≈ uniform at matched budget), record the negative in Research 591 with raw numbers, keep the feature opt-in, and mark the guide's P2 fusion as refuted-at-current-budget — do not re-tune silently and do not promote.
