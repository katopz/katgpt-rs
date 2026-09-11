# Plan 597: BMR + EFE-over-Models — modelless structure-selection primitive

**Date:** 2026-09-12
**Research:** [katgpt-rs/.research/551_Active_Inference_Artificial_Reasoning_BMR_EFE.md](../.research/551_Active_Inference_Artificial_Reasoning_BMR_EFE.md)
**Source paper:** Friston et al., Nat Commun 2026, DOI 10.1038/s41467-026-77209-5 / arXiv:2512.21129 — "Active inference and artificial reasoning"
**Target:** `crates/katgpt-core/src/bmr.rs` (new module) + Cargo feature `bmr` (default-off until GOAT)
**Status:** Active — Phase 0 (filed; no code yet). Downstream: riir-ai guide `.research/376` + issue `.issues/925` (scientist-NPC fusion, blocked on this plan).

---

## Goal

Ship the two zero-coverage components distilled in Research 551 as a modelless `katgpt-core` primitive: **Bayesian Model Reduction** (closed-form log-evidence of reduced Dirichlet models from accumulated counts, Eq 7/9/11 of the paper) and the **expected information gain over models** term of expected free energy (Eq 10), plus the Occam commit statistic (Eq 12) and the generic isomorphic model-space enumerator (Eq 14). All closed-form (lgamma/Beta arithmetic), log-space only, zero-alloc hot paths. GOAT gate: correctness vs brute-force enumeration + three-ball qualitative ablation reproduction; sparse-Δ perf ≥100× vs naive; alloc-free. On PASS → promote `bmr` to default (it is pure math with no interaction surface; promotion still requires the gate).

**Anti-FEP scope guard (Research 551 §2.4):** this primitive serves the *discovery* axis (unknown likelihood structure). It must NOT be wired into policy extraction over known models — MOP (Research 478) and HMM control (Research 543) own that axis. Any consumer that violates this split reds the review.

**UQ floor rule (Research 322):** considered; **not applicable** — model-selection belief, not a predictive interval/coverage/quantile claim. The honest quality gate is discovery-rate + premature-commit-rate against the paper's ablation, recorded in the bench doc.

---

## Phase 1 — Numerics skeleton

### Tasks

- [ ] **T1.1** `ln_gamma` (Lanczos g=7, f64, modelless ~30 LOC; statrs stays dev-dep) + property tests (integers via factorial, half-integers, monotonicity, x→1 ⇒ 0).
- [ ] **T1.2** `ln_beta(x: &[f64])` + `ColumnSums` cache struct (per-column Σ and ΣlnΓ maintained incrementally) + tests vs direct formula and vs exact small-integer Beta values.
- [ ] **T1.3** `Counts` layout: fixed-size column store (`[f64; MAX_OUT]` per column, `ArrayVec` columns; const-generic or runtime-bounded ≤ 64 outcomes/columns — the paper's modalities are ≤ 4).
- [ ] **T1.4** Feature wiring: `bmr = []` in katgpt-core Cargo.toml, module behind `#[cfg(feature = "bmr")]`, `cargo check -p katgpt-core --features bmr` green; wasm32 check (the module is pure math — must compile on both arms).

## Phase 2 — BMR core (Eq 7 / 9 / 11)

### Tasks

- [ ] **T2.1** `bmr_log_evidence(prior, post, reduced_prior) -> f64` — per-column `ln B(ã) + ln B(a) − ln B(ã_m) − ln B(ã_m + a − ã)`; property test: symmetric edge (ã_m = ã ⇒ 0 evidence delta); degenerate reduced prior (all-zero column + 1/32 shrinkage) matches the paper's shrinkage convention.
- [ ] **T2.2** **Brute-force cross-check (G1 anchor):** a test-only exact marginal-likelihood enumerator (small tensors ≤ 3×4, ≤ 5 models) — BMR log-evidence must match to 1e-9. This is the correctness oracle; keep it as a permanent regression test.
- [ ] **T2.3** `posterior_over_models` (Eq 9, uniform model prior; log-sum-exp normalization; zero posterior mass handled in log space).
- [ ] **T2.4** `predictive_model_posterior` (Eq 11) — the sparse-Δ path: `Δa = ŝ ⊗ ô` touches ONE column; incremental `ln_beta_delta` recomputes only that column's four Beta terms per model. Test: identical result to full recompute with `a + Δa`.
- [ ] **T2.5** `occam_log_bayes_factor` (Eq 12: `ln p*/(1−p*)`) + edge tests (p*→1 saturates gracefully; uniform posterior ⇒ 0).

## Phase 3 — EFE model-gain term + model-space enumerator

### Tasks

- [ ] **T3.1** `efe_model_gain(action: &[(state, outcome, prob)]) -> f64` — Eq 10: `Σ_o P(o|u) · KL(Q(m|·,o,u) ‖ Q(m|·,u))`, KL computed in log space over the model posterior. Consumes `predictive_model_posterior`. Zero-alloc: caller-supplied scratch.
- [ ] **T3.2** `enumerate_isomorphic_rules(factors: &FactorLayout) -> Vec<Counts>` — the generic Eq 14: choice factor sharing annotated states with criterion factors; context levels uniquely specifying criterion; housekeeping = every column gets ≥ 1 count. Test: the three-ball layout yields exactly 81 rules (79 unique after dedup).
- [ ] **T3.3** Doc header on the module: equations cross-referenced to the paper (Eq numbers), the anti-FEP scope guard, and the ×512 novelty-suppression convention (a consumer concern, documented here as the canonical reading).

## Phase 4 — Three-ball gate (G1 qualitative ablation)

### Tasks

- [ ] **T4.1** Three-ball paradigm as an integration test (behind `bmr`): 5 factors (3 location + gaze + choice), 2 modalities (visual, feedback), 81-rule space, preference c = [0, 2, −6], 6-step trials, seeded RNG. Arms: (a) full info gain (states+params+models), (b) states+params only, (c) no info gain (random).
- [ ] **T4.2** Gate thresholds (qualitative, not paper-number clones): arm (a) discovers the true rule (Occam > 16 nats, KL(posterior‖true)→0) in ≥ 90% of 64 seeds within 40 trials; arm (b) leaves ≥ 2 plausible models in a majority of runs; arm (c) fails to reach Occam > 4 nats in most runs. Record premature-commit count (expected small, nonzero — it is a tunable, not a bug).
- [ ] **T4.3** If arm (a) underperforms: debug priors/Δa convention first (the known failure classes are shrinkage-value sensitivity and anticipated-outcome normalization), not the thresholds.

## Phase 5 — Benchmark + GOAT gate

### Tasks

- [ ] **T5.1** Bench (`benches/bench_bmr.rs`, feature-gated): (i) init O(#cols) posterior over 81 models; (ii) sparse-Δ predictive posterior per action; (iii) `efe_model_gain` per action; (iv) naive full-recompute baseline. Target: sparse-Δ ≥ 100× vs naive; per-action eval in single-digit µs on M3.
- [ ] **T5.2** G4: assert alloc-free hot path (existing alloc-gate pattern; `debug_assertions`-capability form per the profile rule — feature-gated, not profile-gated).
- [ ] **T5.3** G3: `cargo clippy --workspace --all-targets --all-features` clean (healer first for mechanical findings); wasm32 lane green.
- [ ] **T5.4** Bench doc `.benchmarks/NNN_bmr_efe_goat.md` (next number per `.benchmarks/.highwater`): record G1 ablation table, G2 numbers, G4 verdict, the UQ-floor not-applicable note, and the promote/demote decision.
- [ ] **T5.5** GOAT PASS ⇒ promote `bmr` to default in katgpt-core + update README feature table + `docs_gate.sh` locally (benches/docs label audit). FAIL ⇒ stay opt-in, record why in the bench doc + Research 551 status line.
- [ ] **T5.6** Unblock downstream: comment on riir-ai `.issues/925` (scientist-NPC fusion) with the commit hash of the landed primitive.

---

## Consumer map (post-landing, tracked elsewhere)

- riir-ai `.issues/925` — model-EVPI gate extension + sleep arbitration + commit→freeze (guide: riir-ai `.research/376`).
- riir-clippy (recorded, Research 551 §2.3 F3) — healer rule-corpus disambiguation; file there when wired.
- riir-neuron-db (recorded, §2.3 F4) — crowd-pooled Dirichlet evidence via shard merges.
