# Issue 731 — Residual-gated early exit for the weight-tied looped forward (EqR action item 7.2)

**Status:** OPEN — T1 LANDED 2026-09-07 (`LoopResidualExit`, feature `cadence_gate`, default-off); T2 τ-calibration bench + T3 GOAT next (T3's G2 needs T2's calibrated τ); promote only on GOAT (G1 replay parity ✓ landed, G2 iteration cut pending, G4 alloc-free ✓ by construction).
**Date:** 2026-09-06
**Source:** arXiv:2605.21488 *Equilibrium Reasoners* (ICML 2026, Huang/Geng/Kolter) — re-audit addendum in [`.research/079_EqR_Equilibrium_Reasoners.md`](../.research/079_EqR_Equilibrium_Reasoners.md) §10. This is research note 079's action item 7.2 ("Residual Tracking in Loop Mode"), never landed.
**Lineage:** Plan 119 (✅ `Top1Converged` rollout **selection**, feature `eqr_convergence`) — this issue is the other half: loop **exit**.

## Gap

The weight-tied looped forward (`forward_looped`, LT2 — Plan 108) still runs a **fixed `loop_count`**. Convergence-gated exit ships only in pieces, none wired to the looped forward:

| Substrate | What it does | Consumed signal | Wired to looped forward? |
|---|---|---|---|
| `cp_hopfield/llg.rs` | velocity-tol early exit (Hopfield recall) | ‖flow‖ < tol | no (different loop) |
| `katgpt_core::convergence_cadence` (opt-in `cadence_gate`, Issue 720 / Research 529) | windowed ‖Δz‖ `Settled`/`Churning` classifier + damp/deliberate/restart escalation | newer/older half-window means of step size | **no consumer** |
| `gain_cost_halt` (Plan 304) | per-loop halt composition | gain vs cost curve | composed, not wired to the loop's own residual |
| `ResidualTracker` (Plan 119) | marginal-change residual ∥p_{d+1}−p_d∥₂ per rollout | state delta | selection only, never exit |
| `ResidualGate` (Issue 698) | conditional gate freeze at quiescence | cos(z_τ, z_{τ−1}) | freeze, not exit |

EqR's ACT runtime half: difficulty-adaptive iteration — easy inputs exit in 1–5 steps, hard ones keep unrolling; a learned halter cut avg NFE 17.4× for −0.8% accuracy at D=1024. The learned ACT head is training-track (and measured FAILED in `riir-train/.research/440_Sotaku_Late_State_Looped_Solver.md`); the **residual-threshold exit** is the modelless half and consumes signals this stack already computes.

## Tasks

- [x] T1 — Consume, don't re-deploy: thread `ResidualTracker` (or `ConvergenceCadence`) through `forward_looped`'s z-trajectory behind a feature flag (reuse `eqr_convergence` or gate alongside `cadence_gate`). Exit when the final-window mean residual < τ (window L=3 per EqR) OR the cadence verdict is `Settled`; never exit before a floor of D_min iterations; τ = config, default off (bit-identical to fixed-D when disabled). **DONE 2026-09-07 (this commit):** shipped as `LoopResidualExit` (katgpt-core `convergence_cadence.rs`, feature `cadence_gate`) — the ConvergenceCadence substrate chosen over ResidualTracker (fixed `[f32;4]` ring = the G4 zero-alloc contract; ResidualTracker's Vec grows per push). Two arms: magnitude (L=3 window mean < τ, INF-prefilled so a partial window cannot fire) OR shape (`Settled` verdict — the Research-440 magnitude-only trap guarded by construction: the absolute-floor shape arm rides alongside). Root forward `cadence_gate = ["katgpt-core/cadence_gate"]` (opt-in, DEFAULT-OFF); `forward_looped` takes the probe as a cfg-gated param slot (the Issue-035/gain_cost_halt precedent — no Config-field constructor ripple; τ/d_min are caller-owned probe config, recorded as the deviation from the "τ = config" wording). Gates: katgpt-core cadence unit tests 16/16 (floor, both arms, churning-never-fires, non-finite-never-fires) + `tests/issue_731_t1_residual_exit.rs` 3/3 e2e (G1 fed-but-never-firing ≡ None bit-identical ×27 tokens; fired-exit ≡ `elastic_loop_override` BIT-IDENTICAL at 4 and at d_min=8; the L=3 window timeline pinned) + katgpt-core default lib 1974/0 + clippy 0 at default and cadence_gate states + the issue_698_t4 halter-floors e2e still green under its own feature set.
- [ ] T2 — τ calibration + the Research-440 negative control: calibrate on the bench_119 harness and report the knee (median iterations vs task metric). **Raw residual magnitude is a known trap on looped checkpoints** — Research 440 measured RMS(‖F(h)−h‖) plateauing ≈0.63 while state RMS grows 24→706 (growing-denominator artifact) — so the exit must consume the *windowed decay shape* (`Settled` vs `Churning`), not magnitude alone. Include a magnitude-only arm, expected to fail, as the recorded control. **DESIGN CORRECTION + PRE-REGISTRATION (2026-09-07, committed BEFORE the run — the bench_119 harness measures DDTree rollout selection, a different loop with a different residual scale; calibrating the forward_looped probe's τ on it would be a category error):** the calibration harness is `tests/bench_731_t2_residual_calibration.rs` (this commit) on the T1 fixture convention (micro/seed-42/Uniform/AHLA, R_REF = 32), two stability arms — `None` (the natural-decay regime, the exit's target) and `InterLoopNorm` (the Issue-698-T4 measured step-plateau regime = the Research-440 control; realistic τ ≤ 10 must fire 0/27 there, asserted). Phase A = the depth→quality curve (mean cosine distance to the R_REF reference over 27 tokens, k grid 1..32); Phase B = the τ→fired-k table (τ log grid ×27 tokens, d_min = 4) + the exit ≡ elastic bit-equivalence re-checked at each calibration point. G2 GOAT input: a τ qualifies iff median fired k ≤ 16 at cosine distance ≤ the knee bound.
- [ ] T3 — GOAT gates: G1 halt-disabled replay bit-identical to fixed-D; G2 ≥2× median iteration cut at equal task metric; G4 zero steady-state allocation for the tracker; p99 worst-case depth reported alongside the median.
- [ ] T4 — Promotion path per feature-flag discipline: only on G1–G4 pass; fixed `loop_count` stays default until then. No loser to demote (the fixed count is the control arm, retained).

## Non-goals

- No learned halting head (Q-head halt measured FAILED, Research 440).
- No Anderson / root-solver acceleration (Research 035 disproved; EqR itself shapes landscapes instead of solving fixed points).
