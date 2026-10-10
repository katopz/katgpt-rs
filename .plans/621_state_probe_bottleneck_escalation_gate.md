# Plan 621: State Probe + Bottleneck-Gated Escalation (`state_probe`, `escalation_probe_gate`)

> **Status:** Active — Phase 1 (T1.1–T1.4) LANDED 2026-10-08 (opt-in `state_probe`, katgpt-core only); **Phase 2 (T2.1–T2.3) + Phase 3 kernel (T3.1/T3.3/T3.4) LANDED 2026-10-10** (opt-in `escalation_probe_gate` implies `state_probe`; G3/G4 green, truth tables pinned); **Phase 4 healer-lane eval LANDED 2026-10-11** ([.benchmarks/621_escalation_eval_goat.md](../.benchmarks/621_escalation_eval_goat.md): matched-spend rescues 2.66× the incumbent at precision 1.00 vs 0.34–0.40, identical rescues at 40 % spend uncapped, G2 cost cell 350× under the bar — Super-GOAT ON THE DECLARED MODEL, promotion stays opt-in for the T5.2 consumer's re-gate); **T5.1 LANDED 2026-10-11** (reflex `2680159`, the `--probe-delta-ab` certification lane — real first read MOVES VALUE, settled @ n=128); T5.3 DONE 2026-10-11; T3.2 seam adapters are CONSUMER-side composition (refine Issue 156 + the reflex/instinct ESC seams — not this repo); remaining: T5.2 (refine rescue predicate, consumer) + T5.4 (promotion, gated on T5.2's real-telemetry re-gate)
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

- [x] **T2.1** `classify()`: knowledge-like ⟺ `wilson_hi < ε`; adaptive-N stopping (interval straddles ε → continue; N_max cap); every classification flip disclosed **with its interval width**.
  → **LANDED 2026-10-10** — `state_probe.rs` Phase-2 section: `classify()` (NaN bounds fail closed into Undetermined — never a decisive class from a corrupt interval), `AdaptiveClassifier` driver (caller-owned flip-log slice, zero-alloc; records every flip INCLUDING Undetermined→decisive — Undetermined IS the flip origin per T2.3 — with `at_n`/`from`/`to`/`interval_width`; a full log drops further flips, caller's sizing duty, pinned by test), `AdaptiveDecision::{Settled,Continue,CappedUndetermined}` (the cap falls back to the conservative non-knowledge-like reading).
- [x] **T2.2** G1 fixtures: planted p ∈ {0, 0.05, 0.2} — p=0 declared knowledge-like within N≤64 at δ; p=0.2 never; p=0.5 either-but-disclosed.
  → `tests/state_probe_t2_classify.rs` (6 tests): p=0 settles KL within N=64 with the flip DISCLOSED; p=0.05 never Productive (settle N is draw-dependent — E[k]=3.2@N=64 keeps wilson_hi above ε on a k=5 draw; the seed-robust properties pinned instead); p=0.2 never KL at N∈{8..128} and reaches Productive by 64/128; p=0.5 settles Productive through disclosed flips only.
- [x] **T2.3** Reclassification N-sweep test (N ∈ {8,16,32,64}): flips are MONOTONE (0→positive only), every flip reported with width. The paper's 28/76 is calibration context in Research 609 — never a fixture assertion.
  → the sweep test walks 4 planted states × the N grid: ≤1 flip per state, Undetermined is the ONLY flip origin (a decisive revert is the failure), disclosed count == observed count, every entry carries width > 0.

## Phase 3 — Escalation gate wiring

### Tasks

- [x] **T3.1** `escalation_probe_gate.rs` (feature `escalation_probe_gate`, implies `state_probe`): utility `U = I[pred-success]·(1 − λ·Ĉ)` over `{stay, escalate(d)}` — `gain_cost_halt` scissors law generalized; corpus-distance signal is CONSUMER-SUPPLIED (kernel stays reflex-free).
  → **LANDED 2026-10-10** — the multiplicative reward shape (Research 609 §2.4: cost multiplies on predicted success ONLY; the subtractive form is the recorded λ=0.2 cost-centering failure); `stay_utility` = I[¬KL] (Ĉ=0), `escalate_utility` = I[r_d≥r_min]·clamp(1−λ·Ĉ_d, 0, 1) (a discount, never negative), `gate()` = escalate iff KL ∧ some U_esc > 0, ties to the LOWEST index, stay otherwise. FlyBy's asymmetry is STRUCTURAL: failing escalation scores 0; a productive state's stay scores 1 ≥ any U_esc ≤ 1. NaN λ/prior/bounds fail closed to StayLocal.
- [-] **T3.2** Seam adapters: `modelless_cap` comparison (riir-refine consumer) + `EscalateSpec` margin (riir-instinct consumer) — composition, not replacement; the cap gate remains the fallback path.
  → DEFERRED to the consumer repos by design (T3.2 is composition IN the consumers; katgpt-rs is upstream of both and cannot host the seams). riir-refine Issue 156 owns the rescue-predicate adapter (T5.2's consumer); the instinct ESC seam is post-split Rethink-side for the encoder lane. The kernel-side λ=0 reduction the adapters compose WITH is pinned (T3.3).
- [x] **T3.3** G3: probe-off / λ=0 → incumbent decision byte-identical.
  → `lambda_zero_is_the_incumbent_pred_success_rule`: at λ=0 the cost multipliers strip to 1 and the gate equals the pred-success-only rule on a 3-case grid (KL→escalate, Undetermined→conservative stay, Productive→stay) + the bare-indicator utility pins. Probe-off is structural: the module compiles away without `state_probe` (the feature implication), and the default build carries neither (G3 default lib 2141 passed 0 failed, unchanged).
- [x] **T3.4** G1 truth table: a failing escalation never out-scores a cheaper success; utility monotone in λ (higher λ ⇒ weakly fewer escalations).
  → `failing_escalation_never_outscores_a_cheaper_success` (r_d < r_min ⇒ never escalates; productive stay out-scores a passing escalation) + `utility_monotone_in_lambda_weakly_fewer_escalations` (the pinned depth sequence [3,1,1,1,1,1,1] over λ ∈ {0..2}: λ=0's all-1.0 tie takes the lowest index, λ>0 re-ranks to the near corpus, discounted-out depths never return).

## Phase 4 — Healer-lane evaluation (the G1 that decides the label)

### Tasks

- [x] **T4.1** Eval corpus: fix-miss states with BOTH classes present (planted execution-like + knowledge-like). **A gate that never fires FAILS G1** (never-escalate floor arm).
  → **LANDED 2026-10-11** — `state_probe_eval.rs`: 150 planted states (60/60/30 knowledge/execution/productive, class-INTERLEAVED round-robin — a knowledge-first layout would hand the cap-only incumbent its best case by construction), seed-pinned splitmix64, common-random-number draws across arms; the scissors law declared FlyBy-faithful (escalation rescues ONLY the knowledge class — a class-agnostic expert rate would make triage worthless by construction). The floor law enforced by test (`corpus_carries_both_classes_and_the_floor_leaves_headroom`).
- [x] **T4.2** Comparison table in every G1 run: incumbent (modelless_cap alone) vs probe-gated vs never-escalate floor → escalation rate, rescued fixes, spend.
  → LANDED — `print_table` prints on every run; the frozen read: tight cap 93 vs 35 rescues (2.66×) at precision 1.00 vs 0.34; generous cap 98 = 98 rescues at 60 vs 150 units (40 % spend).
- [x] **T4.3** G2 cost line: probe's N evaluations ≤ **10%** of the gated decision's cost; breach → fallback to the incumbent decision (never silently eat the budget).
  → LANDED — `PROBE_BUDGET_FRACTION` + the in-arm guard (`budget_fallback` flag, loud); measured 0.048 vs the 6.0 line (125× headroom); the breach→fallback path TESTED with an absurd per-eval cost (`budget_breach_falls_back_to_the_incumbent_loudly`). Wall cell: 14.4 ns/state (High-Power-Mode disclosure in the record; 350× under the 5 µs bar).
- [x] **T4.4** Benchmark file + `.benchmarks/` record. **The Super-GOAT label is decided HERE**: probe-gated must beat the incumbent at matched spend.
  → **DECIDED: EARNED ON THE DECLARED MODEL** — [.benchmarks/621_escalation_eval_goat.md](../.benchmarks/621_escalation_eval_goat.md); the record carries the label's boundary (planted corpus, declared constants) and routes promotion to the T5.2 consumer's real-telemetry re-gate.

## Phase 5 — Consumers + docs

### Tasks

- [x] **T5.1** reflex harness lane `--probe-delta-ab` (offline paired certification that an intervention class moves value) — reflex-side, consumer only.
  → **LANDED 2026-10-11** — reflex `2680159` (feature `probe_delta_ab` → `katgpt-core/state_probe`; `src/harness/probe_delta.rs`): the FlyBy paired counterfactual ΔV protocol over the corpus-ab lane's frozen read — discordant pairs = the ensemble, win-rate = V̂, `classify(ε = 0.5)` = the sign test (MOVES VALUE / NO VALUE / UNDETERMINED), the adaptive prefix walk discloses `settled_at_n` + every flip with its width; the full frozen read is the certification of record. AUGMENTS the V5 paired-LB95 gate (byte-identical verdict, G3 377/0 default · 385/0 feature · clippy 0 ×3 postures). Real first read: massive_intent_en 11W/1L → MOVES VALUE, settled @ n=128. 8 module tests pin the geometry.
- [ ] **T5.2** riir-refine Issue 156 rescue predicate consumes `ProbeEstimate` (state-level b0/bd).
- [x] **T5.3** README + `count_features` sync; cross-refs from Research 609.
  → **DONE 2026-10-11** — `count_features.py` green (685 = measured, all claim sites match; the two features are default-off opt-ins, correctly outside the README's default-on listing); Research 609 row 3 carries the T5.1 landed hash; the T5.2 consumer cross-ref rides that landing's commit.
- [ ] **T5.4** Promotion/demotion: GOAT pass + modelless gain → promote `escalation_probe_gate` per flag discipline; the cap-only incumbent is RETAINED as the G3 fallback, never deleted.

---

## GOAT gate summary

G1 correctness (truth tables + loud-zero baseline + two-class corpus) · G2 perf (≤1 µs bound math; ≤10% probe-cost budget with fallback) · G3 no-regression (probe-off / λ=0 byte-identical) · G4 alloc-free. **UQ note:** the probe emits a binomial CI — G1 arms assert its empirical coverage on fixtures; if a consumer treats the interval as calibrated UQ, the Report-the-Floor extension binds at that consumer's re-gate. Feature flags `state_probe` + `escalation_probe_gate` default-off until Phase 4's gates pass; demote-the-loser rule: the incumbent cap gate survives as fallback.
