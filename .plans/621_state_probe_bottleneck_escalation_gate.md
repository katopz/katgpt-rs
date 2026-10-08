# Plan 621: State Probe + Bottleneck-Gated Escalation (`state_probe`, `escalation_probe_gate`)

> **Status:** Active — Phase 1 not started; feature-flagged, promotion follows the GOAT gates (never precedes them)
**Date:** 2026-10-08
**Research:** [katgpt-rs/.research/609_FlyBy_Execution_Knowledge_Bottleneck_Gate.md](../.research/609_FlyBy_Execution_Knowledge_Bottleneck_Gate.md)
**Source paper:** [arXiv:2609.34327](https://arxiv.org/abs/2609.34327) — FlyBy (KAIST 2026); decision layer only, zero training
**Target:** `crates/katgpt-core/src/state_probe.rs` (new) + `crates/katgpt-core/src/escalation_probe_gate.rs` (new) + Cargo features `state_probe`, `escalation_probe_gate` (implies `state_probe`)
**Cross-ref:** riir-train Plan 448 (trained-policy arm, SECONDARY) · riir-refine Issue 156 (rescue mining consumer)

---

## Goal

Ship the modelless decision layer of FlyBy: a probe kernel (`V̂`, Miller-Madow entropy, Wilson/LCB — **consuming `hint_regret::gate::wilson_score_ci`, no third CI copy**), a bottleneck classifier (LCB inequality + adaptive-N stopping), and a cost-aware escalation utility wired INTO the existing ESC seams (`modelless_cap` / `EscalateSpec` consumers) as composition, not replacement. The kernel takes plain ensemble outcomes (pass counts + answer histogram) — reflex-free by construction. **Super-GOAT promotion of the fusion is decided by measured G1 on the healer lane, never asserted.**

## Phase 1 — Probe kernel (CORE)

### Tasks

- [ ] **T1.1** `state_probe.rs`: `ProbeInput { pass_count: u32, n: u32, histogram: &[u16] }` → `ProbeEstimate { v_hat, h_mm, wilson_lo, wilson_hi }`; consume `hint_regret::gate::wilson_score_ci`; zero-alloc; feature `state_probe` (default-off).
- [ ] **T1.2** Miller-Madow fixture: uniform multinomial K=4/N=32 within tolerance of the exact expectation; monotone in added successes; BLAKE3-stable output bytes.
- [ ] **T1.3** G4 alloc gate: fixed arrays, `alloc_delta` pattern (bench_039 precedent).
- [ ] **T1.4** G2 micro-bench: bound math ≤1 µs/call at release.

## Phase 2 — Bottleneck classifier

### Tasks

- [ ] **T2.1** `classify()`: knowledge-like ⟺ `wilson_hi < ε`; adaptive-N stopping (interval straddles ε → continue; N_max cap); every classification flip disclosed **with its interval width**.
- [ ] **T2.2** G1 fixtures: planted p ∈ {0, 0.05, 0.2} — p=0 declared knowledge-like within N≤64 at δ; p=0.2 never; p=0.5 either-but-disclosed.
- [ ] **T2.3** Reclassification N-sweep test (N ∈ {8,16,32,64}): flips are MONOTONE (0→positive only), every flip reported with width. The paper's 28/76 is calibration context in Research 609 — never a fixture assertion.

## Phase 3 — Escalation gate wiring

### Tasks

- [ ] **T3.1** `escalation_probe_gate.rs` (feature `escalation_probe_gate`, implies `state_probe`): utility `U = I[pred-success]·(1 − λ·Ĉ)` over `{stay, escalate(d)}` — `gain_cost_halt` scissors law generalized; corpus-distance signal is CONSUMER-SUPPLIED (kernel stays reflex-free).
- [ ] **T3.2** Seam adapters: `modelless_cap` comparison (riir-refine consumer) + `EscalateSpec` margin (riir-instinct consumer) — composition, not replacement; the cap gate remains the fallback path.
- [ ] **T3.3** G3: probe-off / λ=0 → incumbent decision byte-identical.
- [ ] **T3.4** G1 truth table: a failing escalation never out-scores a cheaper success; utility monotone in λ (higher λ ⇒ weakly fewer escalations).

## Phase 4 — Healer-lane evaluation (the G1 that decides the label)

### Tasks

- [ ] **T4.1** Eval corpus: fix-miss states with BOTH classes present (planted execution-like + knowledge-like). **A gate that never fires FAILS G1** (never-escalate floor arm).
- [ ] **T4.2** Comparison table in every G1 run: incumbent (modelless_cap alone) vs probe-gated vs never-escalate floor → escalation rate, rescued fixes, spend.
- [ ] **T4.3** G2 cost line: probe's N evaluations ≤ **10%** of the gated decision's cost; breach → fallback to the incumbent decision (never silently eat the budget).
- [ ] **T4.4** Benchmark file + `.benchmarks/` record. **The Super-GOAT label is decided HERE**: probe-gated must beat the incumbent at matched spend.

## Phase 5 — Consumers + docs

### Tasks

- [ ] **T5.1** reflex harness lane `--probe-delta-ab` (offline paired certification that an intervention class moves value) — reflex-side, consumer only.
- [ ] **T5.2** riir-refine Issue 156 rescue predicate consumes `ProbeEstimate` (state-level b0/bd).
- [ ] **T5.3** README + `count_features` sync; cross-refs from Research 609.
- [ ] **T5.4** Promotion/demotion: GOAT pass + modelless gain → promote `escalation_probe_gate` per flag discipline; the cap-only incumbent is RETAINED as the G3 fallback, never deleted.

---

## GOAT gate summary

G1 correctness (truth tables + loud-zero baseline + two-class corpus) · G2 perf (≤1 µs bound math; ≤10% probe-cost budget with fallback) · G3 no-regression (probe-off / λ=0 byte-identical) · G4 alloc-free. **UQ note:** the probe emits a binomial CI — G1 arms assert its empirical coverage on fixtures; if a consumer treats the interval as calibrated UQ, the Report-the-Floor extension binds at that consumer's re-gate. Feature flags `state_probe` + `escalation_probe_gate` default-off until Phase 4's gates pass; demote-the-loser rule: the incumbent cap gate survives as fallback.
