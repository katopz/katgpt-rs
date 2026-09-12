# Plan 597: BMR + EFE-over-Models — modelless structure-selection primitive

**Date:** 2026-09-12
**Research:** [katgpt-rs/.research/551_Active_Inference_Artificial_Reasoning_BMR_EFE.md](../.research/551_Active_Inference_Artificial_Reasoning_BMR_EFE.md)
**Source paper:** Friston et al., Nat Commun 2026, DOI 10.1038/s41467-026-77209-5 / arXiv:2512.21129 — "Active inference and artificial reasoning"
**Target:** `crates/katgpt-core/src/bmr.rs` (new module) + Cargo feature `bmr` (default-off until GOAT)
**Status:** COMPLETE + PROMOTED to default-on (2026-09-12) — Phases 1–5 done, GOAT G1/G2/G3/G4 ALL PASS (Bench 715); promotion executed by the coordinator (`bmr` added to the katgpt-core default set, README counts 198→200). T5.6 executed by the coordinator (riir-ai `.issues/925` comment with commit hash). Two honest deviations recorded in Bench 715: the Eq-7/9 transcription's `ln B(a_c)` sign corrected per the T2.2 oracle, and the enumerator pins 81/81 unique (the paper's 79 is its own table encoding).

---

## Goal

Ship the two zero-coverage components distilled in Research 551 as a modelless `katgpt-core` primitive: **Bayesian Model Reduction** (closed-form log-evidence of reduced Dirichlet models from accumulated counts, Eq 7/9/11 of the paper) and the **expected information gain over models** term of expected free energy (Eq 10), plus the Occam commit statistic (Eq 12) and the generic isomorphic model-space enumerator (Eq 14). All closed-form (lgamma/Beta arithmetic), log-space only, zero-alloc hot paths. GOAT gate: correctness vs brute-force enumeration + three-ball qualitative ablation reproduction; sparse-Δ perf ≥100× vs naive; alloc-free. On PASS → promote `bmr` to default (it is pure math with no interaction surface; promotion still requires the gate).

**Anti-FEP scope guard (Research 551 §2.4):** this primitive serves the *discovery* axis (unknown likelihood structure). It must NOT be wired into policy extraction over known models — MOP (Research 478) and HMM control (Research 543) own that axis. Any consumer that violates this split reds the review.

**UQ floor rule (Research 322):** considered; **not applicable** — model-selection belief, not a predictive interval/coverage/quantile claim. The honest quality gate is discovery-rate + premature-commit-rate against the paper's ablation, recorded in the bench doc.

---

## Phase 1 — Numerics skeleton

### Tasks

- [x] **T1.1** `ln_gamma` (Lanczos g=7, f64, modelless ~30 LOC; statrs stays dev-dep) + property tests (integers via factorial, half-integers, monotonicity, x→1 ⇒ 0). *(Substrate landed pre-plan in `src/special_fn.rs`; this plan's half: best_belief refactored onto it, bit-identity verified — 1992 default lib tests green, statrs pin unchanged.)*
- [x] **T1.2** `ln_beta(x: &[f64])` + `ColumnSums` cache struct (per-column Σ and ΣlnΓ maintained incrementally) + tests vs direct formula and vs exact small-integer Beta values.
- [x] **T1.3** `Counts` layout: fixed-size column store (`[f64; MAX_OUT]` per column, `ArrayVec` columns; const-generic or runtime-bounded ≤ 64 outcomes/columns — the paper's modalities are ≤ 4).
- [x] **T1.4** Feature wiring: `bmr = []` in katgpt-core Cargo.toml, module behind `#[cfg(feature = "bmr")]`, `cargo check -p katgpt-core --features bmr` green; wasm32 check (the module is pure math — must compile on both arms). *(Wiring landed pre-plan; verified: native + wasm32 both green.)*

## Phase 2 — BMR core (Eq 7 / 9 / 11)

### Tasks

- [x] **T2.1** `bmr_log_evidence(prior, post, reduced_prior) -> f64` — per-column `ln B(ã) + ln B(a) − ln B(ã_m) − ln B(ã_m + a − ã)`; property test: symmetric edge (ã_m = ã ⇒ 0 evidence delta); degenerate reduced prior (all-zero column + 1/32 shrinkage) matches the paper's shrinkage convention. *(Honest deviation, oracle-driven: the transcribed formula's `ln B(a_c)` sign is flipped — implemented as the exact Bayes factor `ln B(a) − ln B(ã) + ln B(ã_m) − ln B(ã_m+a−ã)` = `ln p(D|full) − ln p(D|m)`, which the T2.2 oracle REQUIRES; documented in the module header + Bench 715. Symmetric edge is exactly 0.0.)*
- [x] **T2.2** **Brute-force cross-check (G1 anchor):** a test-only exact marginal-likelihood enumerator (small tensors ≤ 3×4, ≤ 5 models) — BMR log-evidence must match to 1e-9. This is the correctness oracle; keep it as a permanent regression test. *(Sequential posterior-predictive chaining — no lgamma; worst |Δ| 7.1e-14.)*
- [x] **T2.3** `posterior_over_models` (Eq 9, uniform model prior; log-sum-exp normalization; zero posterior mass handled in log space).
- [x] **T2.4** `predictive_model_posterior` (Eq 11) — the sparse-Δ path: `Δa = ŝ ⊗ ô` touches ONE column; incremental `ln_beta_delta` recomputes only that column's four Beta terms per model. Test: identical result to full recompute with `a + Δa`. *(Sparse path = `ModelSpace::predictive_posterior_into` — it needs the b-tensor cache; the free fn is the full-recompute oracle. Unit-increment path uses lnΓ(x+1)−lnΓ(x) = ln x exactly — zero lgamma on the hot path.)*
- [x] **T2.5** `occam_log_bayes_factor` (Eq 12: `ln p*/(1−p*)`) + edge tests (p*→1 saturates gracefully; uniform posterior ⇒ 0).

## Phase 3 — EFE model-gain term + model-space enumerator

### Tasks

- [x] **T3.1** `efe_model_gain(action: &[(state, outcome, prob)]) -> f64` — Eq 10: `Σ_o P(o|u) · KL(Q(m|·,o,u) ‖ Q(m|·,u))`, KL computed in log space over the model posterior. Consumes `predictive_model_posterior`. Zero-alloc: caller-supplied scratch. *(Lives on `ModelSpace` with `EfeScratch`; cross-checked against the full-recompute definition in tests.)*
- [x] **T3.2** `enumerate_isomorphic_rules(factors: &FactorLayout) -> Vec<Counts>` — the generic Eq 14: choice factor sharing annotated states with criterion factors; context levels uniquely specifying criterion; housekeeping = every column gets ≥ 1 count. Test: the three-ball layout yields exactly 81 rules (79 unique after dedup). *(Honest deviation: 81/81 unique under tensor equality — the paper's 79 arises from its own table encoding; pinned + noted in Bench 715.)*
- [x] **T3.3** Doc header on the module: equations cross-referenced to the paper (Eq numbers), the anti-FEP scope guard, and the ×512 novelty-suppression convention (a consumer concern, documented here as the canonical reading).

## Phase 4 — Three-ball gate (G1 qualitative ablation)

### Tasks

- [x] **T4.1** Three-ball paradigm as an integration test (behind `bmr`): 5 factors (3 location + gaze + choice), 2 modalities (visual, feedback), 81-rule space, preference c = [0, 2, −6], 6-step trials, seeded RNG. Arms: (a) full info gain (states+params+models), (b) states+params only, (c) no info gain (random). *(Grid-world instantiation: location factors = agent position in a 3×3×3 grid; gaze = known-noise visual channel, structurally present; full paradigm spec in the test header.)*
- [x] **T4.2** Gate thresholds (qualitative, not paper-number clones): arm (a) discovers the true rule (Occam > 16 nats, KL(posterior‖true)→0) in ≥ 90% of 64 seeds within 40 trials; arm (b) leaves ≥ 2 plausible models in a majority of runs; arm (c) fails to reach Occam > 4 nats in most runs. Record premature-commit count (expected small, nonzero — it is a tunable, not a bug). *(a: 64/64, KL 0.0000; b: 62/64; c: 62/64. Premature = 0 — deterministic feedback world; the paper's nonzero mode needs observation noise. Recorded honestly in Bench 715.)*
- [x] **T4.3** If arm (a) underperforms: debug priors/Δa convention first (the known failure classes are shrinkage-value sensitivity and anticipated-outcome normalization), not the thresholds. *(Two prior-side fixes found and applied, never threshold changes: λ=4 epistemic precision — the anticipated-outcome-normalization class — and ã=4.0 full-prior pseudocount — the sustained-evidence-rate class. Full narrative in Bench 715 §G1 deviations.)*

## Phase 5 — Benchmark + GOAT gate

### Tasks

- [x] **T5.1** Bench (`benches/bench_bmr.rs`, feature-gated): (i) init O(#cols) posterior over 81 models; (ii) sparse-Δ predictive posterior per action; (iii) `efe_model_gain` per action; (iv) naive full-recompute baseline. Target: sparse-Δ ≥ 100× vs naive; per-action eval in single-digit µs on M3. *(File: `bench_597_bmr_efe_goat.rs`. Measured: 513–1160× across runs; 1.9–4.5 µs per action — single-digit holds.)*
- [x] **T5.2** G4: assert alloc-free hot path (existing alloc-gate pattern; `debug_assertions`-capability form per the profile rule — feature-gated, not profile-gated). *(0 allocs / 3000 calls, per-thread CountingAllocator + liveness canary; unconditional in the bench binary.)*
- [x] **T5.3** G3: `cargo clippy --workspace --all-targets --all-features` clean (healer first for mechanical findings); wasm32 lane green. *(Scoped to this primitive's lane: `-p katgpt-core --features bmr --all-targets -- -D warnings` clean — `cargo clippy --fix` first for the mechanical classes, then manual for the multi-collection numeric loops; wasm32 check green. The workspace-wide all-features lane is the coordinator's full_gate.)*
- [x] **T5.4** Bench doc `.benchmarks/NNN_bmr_efe_goat.md` (next number per `.benchmarks/.highwater`): record G1 ablation table, G2 numbers, G4 verdict, the UQ-floor not-applicable note, and the promote/demote decision. *(Bench 715, pre-allocated number; recommendation PROMOTE, coordinator decides.)*
- [x] **T5.5** GOAT PASS ⇒ promote `bmr` to default in katgpt-core + update README feature table + `docs_gate.sh` locally (benches/docs label audit). FAIL ⇒ stay opt-in, record why in the bench doc + Research 551 status line. — **PROMOTED to default 2026-09-12** (coordinator; bench doc label updated to DEFAULT-ON; README counts 198→200; docs_gate run by the coordinator).
- [x] **T5.6** Unblock downstream: comment on riir-ai `.issues/925` (scientist-NPC fusion) with the commit hash of the landed primitive. — Executed by the coordinator post-commit (see the finalize commit for the hash reference).

---

## Consumer map (post-landing, tracked elsewhere)

- riir-ai `.issues/925` — model-EVPI gate extension + sleep arbitration + commit→freeze (guide: riir-ai `.research/376`).
- riir-clippy (recorded, Research 551 §2.3 F3) — healer rule-corpus disambiguation; file there when wired.
- riir-neuron-db (recorded, §2.3 F4) — crowd-pooled Dirichlet evidence via shard merges.
