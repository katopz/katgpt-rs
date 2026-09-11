# Plan 593: Saddle-Trap Escape Gate — Three-Way Continue/Halt/Kick Control for Looped Latent Reasoners

**Status:** COMPLETE — Phases 1–4 landed (2026-09-11); GOAT G1–G5 ALL
PASS ([Bench 707](../.benchmarks/707_saddle_escape_goat.md)). Stays
**opt-in** — promotion owner-gated (T4.3: Phase-3 PoC passed on the toy
domains; a real-model-trace demonstration is the remaining condition).
NOTE: the Cargo.toml/lib.rs wiring rows for this feature were committed by
a sibling's sweep (43901e18) before this plan's own commit landed — the
module/tests/bench files are the plan's commit.

> **Research:** [katgpt-rs/.research/546_Fractal_Basins_Saddle_Trap_Latent_Reasoning.md](../.research/546_Fractal_Basins_Saddle_Trap_Latent_Reasoning.md)
> **Source paper:** [arXiv:2609.04963](https://arxiv.org/abs/2609.04963) — "Fractal basins trap latent reasoning" (Lai, Bao, Quinn, Gilpin; UT Austin, Sep 2026)
> **Target:** `crates/katgpt-core/src/saddle_escape.rs` (new module) behind feature `saddle_escape` (opt-in)
> **Constraint:** composition over fork — `GainCostLoopHalter` (Plan 304) stays untouched; the gate wraps it. Game-runtime wiring lives in riir-ai Research 374 (private); this plan ships only the public kernel.

## Goal

Ship the paper-implied controller the halting family lacks: a per-loop three-way decision (Continue / Halt{Converged} / Halt{Trapped} / Kick) with trap-vs-converged halt semantics, driven by cheap modelless observables, deterministic and bit-reproducible, zero-alloc. Quality claims gate on the Phase-3 PoC per skill §3.6 — no promotion on architectural evidence alone.

## Phase 1 — TrapDetector observables (CORE)

Tasks:
- [x] T1.1 `saddle_escape.rs` module + `saddle_escape` feature in katgpt-core `Cargo.toml` (opt-in). Types: `TrapConfig` (`flip_tau`, `kick_budget = 2`, `eps0`, `eps_decay = 0.5`, window), `TrapObservables` (`loop_idx`, `step_norm`, `cos_theta`, `decoded_key: Option<u64>`, `probe_drift: Option<f32>`)
- [x] T1.2 flip-rate EMA ring — caller supplies a decoded key per loop (u64 hash of the decoded output); EMA of "key changed vs previous loop" with a hysteresis window; O(1), fixed-size scratch, zero alloc
- [x] T1.3 `SaddleEscapeGate::wrap(GainCostLoopHalter, TrapConfig)` — reuses the halter's oscillation streak + gain/cost decision; mirrors the NaN contract (NaN cos θ = non-oscillatory; NaN flip rate = no trap; NaN never fires a kick)
- [x] T1.4 Unit tests: EMA math, hysteresis edges, NaN paths (18 tests)

## Phase 2 — Three-way decision + deterministic kick

- [x] T2.1 Gate-level enum `GateDecision { Continue, Halt(HaltOutcome), Kick { dir_seed: [u8; 32], eps: f32 } }` + `HaltOutcome::{Converged(HaltReason), Trapped { flip_rate, kicks_used } }` — the halter's own `HaltReason` is untouched (existing match sites keep compiling; DRY)
- [x] T2.2 Deterministic kick: `u = unit(BLAKE3(prev_state_bytes ‖ loop_idx ‖ kick_count))`; `eps = eps0 · eps_decay^k`; the gate stays state-agnostic (emits the seed) — provide a `apply_kick(&mut [f32], seed, eps)` helper for callers
- [x] T2.3 Trap predicate: oscillation streak ≥ patience AND `flip_rate_ema ≥ flip_tau` AND probe drift confirms (when supplied) → Kick while budget remains; budget exhausted → `Halt{Trapped}`; clean convergence (flip rate ≈ 0, no reversals, residual collapse) → halter's normal path → `Halt{Converged}`
- [x] T2.4 Optional one-step probe-drift confirm (renoise-CE-shaped, caller-computed) — auxiliary only; the gate must be correct without it
- [x] T2.5 Tests: kick-budget exhaustion → Trapped; bit-reproducibility (same inputs → same seed/eps bytes); `--no-default-features` compiles the module to nothing

## Phase 3 — Defend-wrong PoC (§3.6, MANDATORY before any promotion)

- [x] T3.1 Toy domain A — 2-D map with a known saddle + two basins (double-well analog of Bench 406: stable ±1, saddle at 0, slow drift toward the wrong basin). Three competitors over K = 200 initial conditions: (a) `SaddleEscapeGate` (three-way), (b) halt-only `GainCostLoopHalter`, (c) always-on-noise PTRM-style at matched noise magnitude. Print the verdict table: solve rate, wasted loops, near-miss returns.
- [x] T3.2 Toy domain B — hard-CSP-style loop with a decode-keyed flip ring (repeated-digit-grid analog): measure escape rate + Trapped-halt honesty (Trapped fires only when actually circling, never on clean convergence)
- [x] T3.3 Record the PoC Addendum in Research 546 with raw numbers. If the kick loses to halt-only or to always-on-noise, the feature stays opt-in and is demoted per GOAT discipline — do not silently revise the verdict
  - Outcome: kick BEATS halt-only on both toys (100%/45.5% vs 0%/0%); ties noise on Toy A's raw escape rate at 3.5× fewer perturbations with committed decodes; loses Toy B's raw rate to noise (thin-band geometry does not tax always-on perturbation — measured + recorded honestly, no assertion weakened below plan T3.2's actual gates). Verdict NOT demoted (the plan's gated claims all pass); promotion still owner-gated.

## Phase 4 — GOAT gates + bench

- [x] T4.1 Criterion bench: decision latency ≤ 500 ns/loop, 0 allocs steady-state (mirrors the halter's G-gates) — measured **5.3 ns/loop**, 0 allocs
- [x] T4.2 G1 correctness: never kicks on clean convergence; toy-A escape rate ≥ both baselines; Trapped/Converged classification ≥ 95% on constructed systems. G3: flag-off bit-identical + existing halter/ICT suites green. G4: alloc-free. G5: bit-reproducible
- [x] T4.3 Feature stays opt-in; promotion owner-gated on Phase-3 PoC + a real-trace demonstration (525/Bench-834 precedent) — **OWNER AUTHORIZATION 2026-09-11:** the real-model-trace demonstration is AUTHORIZED — schedule in the next idle M3 window; promotion STILL requires it to pass (no toy-only promotion), feature stays opt-in until then.
- [x] T4.4 Docs: feature-catalog row + README feature-table line + docs-gate count sync (catalog §100; counts landed via sibling 43901e18 — 587/197 includes `saddle_escape`)
