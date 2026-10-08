# Plan 621: State Probe + Bottleneck-Gated Escalation (`state_probe`, `escalation_probe_gate`)

> **Status:** Active — Phase 1 (T1.1–T1.4) LANDED 2026-10-08 (opt-in `state_probe`, katgpt-core only; Phase 2 not started; feature-flagged, promotion follows the GOAT gates (never precedes them)
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

- [x] **T1.1** `state_probe.rs`: `ProbeInput { pass_count: u32, n: u32, histogram: &[u16] }` → `ProbeEstimate { v_hat, h_mm, wilson_lo, wilson_hi }`; consume `hint_regret::gate::wilson_score_ci`; zero-alloc; feature `state_probe` (default-off).
  - **LANDED 2026-10-08** — `crates/katgpt-core/src/state_probe.rs`; feature `state_probe = ["hint_regret"]` (implies the CI substrate's feature — the probe_guidance/dllm precedent; a build without it would compile the module to nothing). `probe()` + `miller_madow_entropy()` both `#[inline]`, borrows the caller's histogram, no collections. K′ = NONZERO bins (observed-support Miller-1955/Grassberger reading — doc'd; coincides with Research 609's alphabet form on the fixture's fully-populated case; empty slice = answers-not-tracked → h_mm 0). Wilson CONSUMED from `hint_regret::wilson_score_ci` (bit-identity pinned by test). NOT-calibrated-UQ caveat in the module doc.
- [x] **T1.2** Miller-Madow fixture: uniform multinomial K=4/N=32 within tolerance of the exact expectation; monotone in added successes; BLAKE3-stable output bytes.
  - **LANDED 2026-10-08** — `tests/state_probe_t1.rs` (9 tests; `#![cfg]` + `required-features` row — no green zero). ⚠ Fixture correction during landing: the plan's "within tolerance of the exact expectation" is about the MEAN over draws — a single all-equal histogram is the H_emp MAXIMUM (ln K + (K−1)/(2N)), not a typical draw; the fixture Monte-Carlos 20,000 seed-pinned splitmix64 draws (E[h_mm] within 5e-3 of ln 4) + a separate exact formula-identity pin on the all-equal draw (ln 4 + 3/64, 1e-12). Monotone-in-agreement (7-step strict decrease) + blake3 output-bytes pin (`dc33cba1…1601f`, to_bits canonical serialization) + substrate bit-identity + n=0/degenerate/single-pass edges + the K′-not-K support pin all green.
- [x] **T1.3** G4 alloc gate: fixed arrays, `alloc_delta` pattern (bench_039 precedent).
  - **LANDED 2026-10-08** — `tests/state_probe_t1_alloc_check.rs` (single-fn binary, the bench_576 convention): 10,000 probe calls, varying pass-count through a black_boxed base (defeats LLVM hoisting of the pure kernel), **0 allocs**; `assert_counter_is_live()` canary armed.
- [x] **T1.4** G2 micro-bench: bound math ≤1 µs/call at release.
  - **LANDED 2026-10-08** — `benches/bench_621_state_probe.rs` (`harness = false`, the bench_039 pattern — asserted bar, black-boxed inputs per call, cargo-bench release posture; no profile-cfg games). **167 ns/call median-of-10k** (probe() full kernel = the entropy scan; Wilson is ~free), **~6× under the 1 µs bar**. Box state: M3 Max, shared box (hyperthink T1 capture running at nice-19, sibling session building katgpt-attn) — the headroom makes the claim load-robust; cargo bench is release by construction.

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
