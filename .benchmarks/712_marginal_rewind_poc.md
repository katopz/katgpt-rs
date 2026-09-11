# Bench 712 — `marginal_rewind` defend-wrong PoC (Issue 746 Row 2)

**Status:** PASS — Row 2 LIVES as opt-in `marginal_rewind`; Row 1 CLOSED (no
time-grid consumer; the schedule ships inside this PoC's harness only).
Feature-count claims bumped 589 → 590 across README/examples (docs_gate
count_features enforcement).

- **Primitive:** `crates/katgpt-core/src/marginal_rewind.rs` (feature
  `marginal_rewind`, opt-in) — calibrated noise-backtrack for interpolant
  flow states, Eq 18 of arXiv:2609.11801 minus the denoiser:
  `a = s/t`, `x̄_s = a·x_t + sqrt((1−s)² − (a−s)²)·ε'`, marginal identity
  `a²(1−t)² + (1−s)² − (a−s)² = (1−s)²`; γ-dial form `a = clip(1−γΔt)`,
  `s = a·t`.
- **Harness:** `crates/katgpt-core/tests/marginal_rewind_poc.rs` — 1-D
  bimodal commitment flow (modes ±1.5), nearest-mode denoiser surrogate,
  Euler-as-lerp with the paper's commitment schedule `r_i = Δt/(1−t_i)`
  (Row 1's math, exercised here per its no-consumer verdict), detection at
  `t_det ≈ 0.703`, K=8192 paired stuck states per regime, deterministic
  splitmix64 seeds.
- **Gates (pre-registered in the harness doc):** G0 telescoping annihilation
  + collapse cohort ≥ 60% · G1 calibrated ≥ 1.5× additive at equal injected
  budget · G2 ∃s: calibrated > hard restart at LOWER budget (collapse-prone
  regime). Unbiased regime = honesty axis (measured, not gated).

## Collapse-prone regime (prior `ε ~ N(−0.8, 1)` — the cgsp-style scenario; 6389/8192 = 78.0% collapsed)

| arm | s=0.55 (σ=0.385) | s=0.40 (σ=0.576) | s=0.25 (σ=0.743) | s=0.10 (σ=0.899) |
|---|---|---|---|---|
| calibrated | 0.3% | 9.2% | 26.1% | **41.7%** |
| additive (equal budget) | 0.0% | 1.1% | 4.0% | 7.3% |
| naive (full (1−s) noise) | 1.4% | 10.3% | 26.0% | 41.5% |
| hard restart (σ=0.950) | — | — | — | 20.9% |

- **G1: PASS — 5.71×** (best calibrated 41.7% vs best additive 7.3% at
  equal per-s budgets; every s-point dominates individually, 3–10×).
- **G2: PASS** — calibrated s=0.10: 41.7% @ σ=0.899 and s=0.25: 26.1% @
  σ=0.743 both dominate hard restart (20.9% @ σ=0.950) at LOWER budget.
  Restart inherits the same collapse-prone prior (the honest "re-run the
  system" model) — its 20.9% matches the analytic Φ(−0.8)·… ≈ 21.2%.

## Unbiased regime (honesty axis; `ε ~ N(0, 1)`, 48.9% collapsed)

| arm | s=0.40 | s=0.25 | s=0.10 | restart |
|---|---|---|---|---|
| calibrated | 11.3% | 27.8% | 42.4% | — |
| additive | 1.5% | 5.1% | 8.7% | — |
| **hard restart** | — | — | — | **50.3%** |

**Hard restart WINS when the prior is clean** (50.3% vs 42.4%) — exactly as
pre-registered: retained wrong-commitment is a pure liability when fresh
prior draws are unbiased. The consumer decision rule this produces:

> Rewind for collapse-prone environments (the trap is prior-favored — cgsp
> degenerate-priority collapse, stale-belief fog-of-war); hard-restart when
> the prior is clean. The γ-dial interpolates between the two policies.

## Honest findings

1. **The mechanism is the de-commit shrink, not the exact variance.**
   Naive full-variance resample (`a·x + (1−s)·ε`) ties calibrated at deep
   rewind (−0.1pp at s=0.25/0.10) and beats it slightly at shallow rewind
   (+1.1–1.2pp at s=0.55/0.40 — over-injection helps when under-noised).
   The identity's value is ON-MANIFOLD level statistics (level-s SNR is
   exactly what downstream re-integration expects; the q_sample cousin
   can't do this without the clean point), not raw escape power.
2. **Additive perturbation is nearly useless at matched budget** (7.3% best
   vs 41.7%) — the uncalibrated-noise class (`saddle_escape` kicks,
   `renoise_ce` probes) is the wrong tool for basin escape at fixed budget;
   its value remains scoring/detection, not recovery.
3. **Two harness bugs were caught by the measurement itself** (the defend-
   wrong discipline working): a first run integrated the pre-detection
   segment to t=1 (state fully committed, level bookkeeping wrong —
   calibrated read 23.9% at s=0.25 vs analytic 30.8%) and passed `+0.8`
   bias (the accidental mirror regime — prior leans CORRECT — where restart
   dominates at 79%; recorded as the honesty axis's mirror image). After
   the fixes, measured vs analytic agree within ~2–8% relative (the sim's
   nearest-mode estimator commits harder than the pure interpolant).
4. **Analytic expectations (pre-registered):** `P_cal(s) = 1 − Φ(s·M/(1−s))`
   (independent of `t_det`), `P_add(s) = 1 − Φ(t_det·M/√((1−t_det)²+σ²))`,
   `P_restart = Φ(b̄)` — measured values track all three.

## GOAT status

- **G1 (correctness):** 9 module tests — exact identity at `s == t`,
  marginal-variance MC (24,576 draws, 5 se bound), γ-dial monotonicity +
  `s = a·t` consistency, σ monotone-in-depth, no-NaN near the `s≈t` f32
  boundary, contract panics.
- **G2 (the quality claim):** this PoC — G1 5.71× / G2 dominate-restart
  gates asserted in-test.
- **G3 (no-regression):** feature-off compiles the module to nothing
  (`--no-default-features --features marginal_rewind` standalone check
  clean; default surface untouched).
- **G4 (alloc):** `tests/marginal_rewind_alloc_check.rs` — 0 allocs over
  8192 rewind calls after warmup (Copy struct plans + caller buffers).
- **G5 (determinism):** the module is pure ordered scalar arithmetic (no
  SIMD reassociation, no libm in the operator path); RNG policy stays with
  the caller.
- **Promotion:** owner-gated (the saddle_escape/525 precedent) — needs a
  live consumer (cgsp collapse recovery or stale-belief re-exploration)
  before default-on.

## Verdicts (Issue 746)

- **Row 2 (marginal-calibrated backtrack): LIVE** — ships as opt-in
  `marginal_rewind`; the calibrated form beats plain noise injection at
  equal magnitude by 5.71× (T1/T2 gates passed, row does NOT die).
- **Row 1 (anytime commitment schedule): CLOSED** — no consumer with a
  time-grid need (tf_loop `DampedEuler` is fixed-β damped iteration, no
  grid; `CommittedFieldBlend` is sigmoid-weighted, no grid;
  `set_diffusion_schedule` is reveal-time CDFs for token orderings —
  different family). The 20-line schedule is exercised by this PoC's
  harness (telescoping annihilation asserted in G0) and stays with the
  harness until a flow-integrating consumer materializes (AC-Prefix
  Issue-002 precedent: no consumer ⇒ dead code).
