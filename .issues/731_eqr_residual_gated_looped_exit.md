# Issue 731 — Residual-gated early exit for the weight-tied looped forward (EqR action item 7.2)

**Status:** OPEN — POC + bench behind an opt-in feature; promote only on GOAT (G1 replay parity, G2 iteration cut, G4 alloc-free).
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

- [ ] T1 — Consume, don't re-derive: thread `ResidualTracker` (or `ConvergenceCadence`) through `forward_looped`'s z-trajectory behind a feature flag (reuse `eqr_convergence` or gate alongside `cadence_gate`). Exit when the final-window mean residual < τ (window L=3 per EqR) OR the cadence verdict is `Settled`; never exit before a floor of D_min iterations; τ = config, default off (bit-identical to fixed-D when disabled).
- [ ] T2 — τ calibration + the Research-440 negative control: calibrate on the bench_119 harness and report the knee (median iterations vs task metric). **Raw residual magnitude is a known trap on looped checkpoints** — Research 440 measured RMS(‖F(h)−h‖) plateauing ≈0.63 while state RMS grows 24→706 (growing-denominator artifact) — so the exit must consume the *windowed decay shape* (`Settled` vs `Churning`), not magnitude alone. Include a magnitude-only arm, expected to fail, as the recorded control.
- [ ] T3 — GOAT gates: G1 halt-disabled replay bit-identical to fixed-D; G2 ≥2× median iteration cut at equal task metric; G4 zero steady-state allocation for the tracker; p99 worst-case depth reported alongside the median.
- [ ] T4 — Promotion path per feature-flag discipline: only on G1–G4 pass; fixed `loop_count` stays default until then. No loser to demote (the fixed count is the control arm, retained).

## Non-goals

- No learned halting head (Q-head halt measured FAILED, Research 440).
- No Anderson / root-solver acceleration (Research 035 disproved; EqR itself shapes landscapes instead of solving fixed points).
