# Plan 590: hmm_homeostasis — Setpoint Reachability Control + psafe-MOP Open Primitive

**Date:** 2026-09-10
**Research:** [katgpt-rs/.research/543_HMM_Homeostatic_Control_Deterministic_Drives.md](../.research/543_HMM_Homeostatic_Control_Deterministic_Drives.md)
**Source paper:** [arXiv:2609.07508](https://arxiv.org/abs/2609.07508) — Moreno-Bote, "Homeostasis Revisited and Reformulated Through Hidden Markov Model Control"
**Target:** `katgpt-rs/crates/katgpt-core/src/hmm_control/` (new module) + `katgpt-core/src/mop/` (config extension) + Cargo feature `hmm_homeostasis`; psafe rides the existing `mop_path_entropy` feature
**Status:** Active — Phase 1 not started
**Downstream:** riir-ai Research 370 (runtime guide); runtime plan opens when P0 merges

---

## Goal

Ship the paper's two exact, modelless homeostatic control primitives on the `mop/` substrate: (1) `HmmControlSolver` — finite-horizon setpoint reachability control via multiplicative backward messages, deterministic argmax policy, bounded [0,1] outputs, zero-alloc; (2) the psafe continuation-probability extension to `MopSolver` (bit-identical when off/`psafe≡1`). GOAT gate: G1 analytic parity on the paper's own T=2 divergence fixtures, G2 paper-Fig-2 behavioral reproduction (deterministic agent reaches the best location; entropy-regularized comparator does not), G3 psafe survival improvement without occupancy collapse, G4 zero-alloc + `psafe≡1` bit-identity to the shipped solver. No promote-to-default until the gate passes AND a runtime consumer wires it (riir-ai Plan 538-successor).

Not UQ-bearing: β messages are exact model-computed probabilities, not calibrated estimates over data — the conformal floor rule (Research 322) does not apply.

## Phase 1 — Setpoint Solver Skeleton (CORE)

### Tasks

- [ ] **T1.1** `src/hmm_control/` module skeleton (`mod.rs`, `types.rs`, `solve.rs`) behind `#[cfg(feature = "hmm_homeostasis")]`, mirroring `mop/` layout (module doc carries the paper equations + the classical-lineage honesty note from Research 543 §4.1).
- [ ] **T1.2** Types: `HmmConfig { horizon_t: u16, beta_floor: f32 }` (+ `HmmConfigError` validation: horizon ≥ 1, emission rows sum ≤ 1+ε); `HmmSolution<const N: usize> { beta: Vec<[f32; N]>, policy: Vec<[u8; N]> }`; `HmmScratch<const N: usize>` (double-buffer for messages — zero per-solve alloc, `MopScratch` pattern).
- [ ] **T1.3** `HmmControlSolver::solve(p, e, &mut scratch) -> HmmSolution` — backward pass Eqs. 14-15: `β_T = max_a e`, then per step `a*_t = argmax_a e·(P·β_{t+1})`, `β_t = e[a*]·(P·β_{t+1})[a*]`. One-hot fast path per row (reuse `mop`'s row_onehot pattern). Log-space guard NOT in Phase 1 (document the f32 underflow bound: β ≤ (max emission)^remaining-steps — underflow to 0 is semantically correct "no hope", not a bug; revisit only if a consumer needs ranking among tiny β).
- [ ] **T1.4** Policy extractor `optimal_action(&self, t, x) -> Option<u8>` (None at terminal/β=0 ties → lowest index, documented tie-break).
- [ ] **T1.5** Golden unit tests — the paper §3 T=2 fixture as a const arena: risky action reaches x*₂ (emission 1.0) with p=0.01, safe action reaches x′₂ (emission 0.5) with p=1.0. Assert: HMM picks safe; hand-computed control-as-inference pick (risky, Eq. 28) recorded in the comment; variational pick (Eq. 30/31 log-vs-sum divergence case) recorded. Structurally-different reference impl (direct recursion, no fast path) for parity — the Plan 573 T2.1 golden-parity discipline.
- [ ] **T1.6** Invariant tests: β ∈ [0,1] for random stochastic kernels + sub-stochastic rows; determinism (same inputs → identical policy bytes); partial observability shape (policy over observation classes `z = g(x)` stays deterministic — build the lifted kernel in the test, assert argmax per class).
- [ ] **T1.7** `benches/bench_hmm_control.rs` (harness = false + Instant pattern per repo bench convention): N ∈ {82, 256}, A ∈ {4, 16}, T ∈ {8, 32, 128}; report µs/solve + iterations-equivalent; alloc assertion (0 per solve).

## Phase 2 — psafe-MOP Extension

### Tasks

- [ ] **T2.1** `MopConfig.terminal_psafe: Option<PsafeTable>` where `PsafeTable<const N: usize, const A: usize>([[f32; A]; N])` — default `None`; validation: entries ∈ [0,1]. Doc comment: paper Eqs. 16-18, "continuation probability = natural discount" + the γ-absorption note (Research 543 §2.1 P2).
- [ ] **T2.2** Solve-path change: when the table is set, scale the per-action bootstrap exponent by `psafe[i][k]` inside the LSE argument (`(β/α)·H̄ + γ·psafe·Σ p·ln z`); psafe=0 ⇒ action contributes only its own entropy reward (no future). Document the chosen finite-horizon-free formulation (discounted-with-continuation) and its divergence contract: contraction holds for any γ<1 OR any state with psafe<1 reachable-and-unavoidable; add the γ=1 + all-psafe=1 degenerate case to `MopConfigError`.
- [ ] **T2.3** Backward-compat gate: `psafe ≡ 1` ⟹ bit-identical `MopSolution` to the current solver on all three existing arenas (golden parity tests extended, not replaced). `psafe = None` is the compile-and-run default (zero behavior change, `mop_path_entropy` consumers untouched).
- [ ] **T2.4** New arena: `ring_world_terminal(slip)` — `ring_world_noisy` + one terminal zone (psafe=0 off-ring, 1 on-ring) as a const fn in `arenas.rs`, so the G3 comparison is reproducible without caller setup.
- [ ] **T2.5** Invariant tests: psafe=0 actions are never chosen when a psafe>0 alternative exists with equal entropy reward; effective-discount claim (γ=1 + honest psafe terminates: sup |Δζ| → 0 on the terminal arena).

## Phase 3 — GOAT Gate + Defend-Wrong PoC

### Tasks

- [ ] **T3.1** G1 (analytic parity): T1.5 fixtures green at default features of the flag, `cargo test -p katgpt-core --features hmm_homeostasis --lib`; count-floor row added to the gate script for the new feature set (the `#![cfg]`-green-zero rule: required-features row + count pin).
- [ ] **T3.2** G2 (behavioral): 4-room gridworld variant with best-location emission 1.0 / elsewhere 0.95 + stay-noise p_s=0.5 (paper Fig. 2 settings); HMM agent reaches + holds the best location; comparator = softmax policy from the variational recursion (Eq. 10-11 with α=1 — implemented in the bench, NOT in the library); report p(y=yd) over 10⁴ episodes both arms; gate: HMM strictly higher.
- [ ] **T3.3** G3 (psafe gain): T2.4 arena — survival fraction + settled coverage for psafe-MOP vs plain MOP vs psafe≡1; gate: survival strictly improves, occupancy H(π*) above a documented floor (no orbit collapse — Bench 681's G8 metric pattern).
- [ ] **T3.4** G4 (perf + alloc): solve ≤ 1 ms at N=256, A=16, T=128 (release); 0 allocs/solve via CountingAllocator; psafe path within +25% of plain solve (documented bound — one extra fmul per LSE arg).
- [ ] **T3.5** Defend-wrong PoC in `riir-poc` (three competitors: HMM control / plain MOP / variational comparator; G2+G3 arenas; verdict table printed; `CARGO_TARGET_DIR=/tmp`); record raw numbers in Research 543 §5 as a PoC Addendum — refutes-or-defends, no silent revision.
- [ ] **T3.6** Gate-script rows: `test_gate.sh`/feature-guard row for `hmm_homeostasis` (count floor), bench doc rows (`.benchmarks/` per numbering discipline), README feature table + examples/README feature count sync (docs_gate `count_features.py`).

## Phase 4 — Promotion Decision (gated)

- [ ] **T4.1** On G1–G4 PASS + riir-ai consumer landed (Research 370 P1/P2): owner call on promote-to-default vs stay-opt-in (`mop_path_entropy` itself is opt-in — default outcome is stay-opt-in until the runtime pillar asks for default).
- [ ] **T4.2** Update Research 543 status line + Research 370 §7 P0 checkbox; add `.benchmarks/` record per numbering discipline.

## Honest risks (carry from Research 543 §8)

- Tabular only (zone-KG abstraction required); kernel quality is the ceiling.
- f32 β underflow at long horizons is semantically "no hope" — documented, revisit on consumer demand.
- psafe formulation choice (discounted-with-continuation vs paper's finite-horizon V_t arrays): Phase 2 ships the discounted form for substrate continuity; if G3's gain is ambiguous, re-run as finite-horizon before declaring the primitive weak.
