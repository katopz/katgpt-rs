# Bench 715 — BMR + EFE-over-models GOAT gate (Plan 597 / Research 551)

**Date:** 2026-09-12
**Primitive:** `katgpt-core` feature `bmr` — Bayesian Model Reduction (Eq 7/9), posterior over models (Eq 9), sparse-Δ predictive posterior (Eq 11), expected information gain over models (Eq 10), Occam commit statistic (Eq 12), generic isomorphic model-space enumerator (Eq 14). Source: Friston et al., "Active inference and artificial reasoning", Nat Commun 2026 (DOI 10.1038/s41467-026-77209-5 / arXiv:2512.21129).
**Files:** `src/bmr.rs` (+ shared `src/special_fn.rs` Lanczos ln_gamma substrate), `tests/three_ball_paradigm.rs`, `benches/bench_597_bmr_efe_goat.rs`.
**Machine:** M3 Max (Apple silicon), `cargo bench` (release profile).

**Status: GOAT G1/G2/G3/G4 ALL PASS — PROMOTED to default-on 2026-09-12** (the primitive is pure modelless math with no interaction surface; the plan's promotion criterion "modelless gain" is met on the evidence below).

---

## G1 — Correctness (modelless)

### G1-numerics (vs exact oracle)

The T2.2 brute-force anchor: BMR log-evidence against an exact marginal-likelihood oracle built from sequential posterior-predictive chaining (integer counts, **no lgamma anywhere** — structurally independent of the Beta-function machinery it validates). The identity checked: `ln p(D|full) − F(m) = ln p(D|m)` where `F` is the module's free energy.

| Check | Result | Target |
|---|---|---|
| BMR vs chained oracle (40 random tensors × ≤4 models, zeros exercising shrinkage) | worst \|Δ\| = **7.1e-14** | < 1e-9 ✅ |
| Symmetric edge (ã_m = ã) | **exactly 0.0** | 0 ✅ |
| All-zero reduced column ≡ all-(1/32) column (shrinkage convention) | bit-identical | ✅ |
| Sparse-Δ predictive vs naive full recompute (81 models × 27 sampled columns) | worst \|Δ\| = **7.2e-16** | < 1e-9 ✅ |
| Engine incremental F vs from-scratch `bmr_log_evidence` after random accumulates | < 1e-8 | ✅ |
| Sign pin (data-consistent reduced model ⇒ lower F ⇒ more posterior mass) | passes | ✅ |
| Occam one-hot saturation | 36.84 nats, finite | ✅ |

### G1-ablation — three-ball qualitative reproduction (plan T4.2)

Our instantiation (grid-world three-ball; see `tests/three_ball_paradigm.rs` header for the full paradigm spec): 27-cell grid (3 location factors), gaze factor (3 levels, known-noise visual modality — structurally present, uninformative here), choice factor (3 levels), feedback modality {none, reward, penalty} with log-preferences c = [0, 2, −6], 81-rule space via `enumerate_isomorphic_rules`, 6-step trials, 40 trials/run, 64 seeds/arm, fastrand-seeded. Gates are the plan's qualitative thresholds, not paper-number clones.

| Arm | Discovery (Occam > 16 ∧ argmax = true) | Mean KL(final ‖ true) | Mean Occam | Mean score | Gate |
|---|---|---|---|---|---|
| **(a) full info gain** (pragmatic + Eq-10 model gain, λ=4) | **64/64 (100%)** | **0.0000** | +36.84 | +415.8 | ≥ 90% ✅ |
| **(b) states+params only** | 2/64 (3%) | — | −3.06 | +13.2 | ≥2 plausible (posterior > 0.01) in 62/64 ✅ |
| **(c) no info gain (random)** | 0/64 | — | −0.53 | −7.8 | Occam ≤ 4 in 62/64 ✅ |

- Premature commits: **0 across all arms** (deterministic feedback world — the wrong-hypothesis evidence path cannot reach 16 nats; the paper's 1/64 premature-commit mode needs their observation noise. Recorded honestly as 0, a tunable, not a bug).
- The ablation reproduces the paper's qualitative separation: info-gain-over-models discovers in every seed with KL→0; states+params leaves ≥2 plausible in a majority (the paper's "fails to disambiguate two plausible hypotheses"); random never commits (the paper's "half never discover" — ours never, slightly stronger, because our barren-cell "none" outcome kills 3 hypotheses per visit whereas the paper's uninformative observations kill none).

### G1 honest deviations from the paper's numbers

1. **Rule count**: the plan pins "81 rules (79 unique after dedup)". Our enumerator yields **81 rules, all 81 distinct** under tensor equality. The paper's 79-unique arises from its own table encoding (not published in a form we could reconstruct); the plan's T4.2 rule — qualitative reproduction, not paper-number clones — covers this. Pinned as 81/81 in `enumerate_three_ball_yields_81_distinct_housekept_rules`.
2. **Formula sign correction**: Research 551's Eq-7/9 transcription carries the `ln B(a_c)` term's sign flipped, which contradicts the T2.2 oracle by the model-dependent amount `2·Σ_c ln B(ã_m,c)`. The corrected exact-Bayes-factor form (`F = ln p(D|full) − ln p(D|m)`, i.e. `ln B(a) − ln B(ã) + ln B(ã_m) − ln B(ã_m + a − ã)` per column) is implemented — the oracle (the plan's designated G1 anchor) outranks the transcription. Documented in the module header.
3. **Task tuning (plan T4.3 protocol — priors/convention debugged, never thresholds)**: two prior-side knobs were required, both documented in the test header: (i) epistemic precision λ=4 on the Eq-10 term — the a-priori outcome marginals {78/81 none, 1/81 reward, 2/81 penalty} make every fresh pick's expected preference ≈ −0.03, so λ=1 causes gaze-forever abstention (the anticipated-outcome-normalization failure class); (ii) full-prior pseudocount ã=4.0 — the sustained per-observation evidence rate scales as 2(ã−s)·ln N, so ã=1 saturates at ~8 nats (below the 16-nat commit line within the 240-step budget) while ã=4 clears it in ~70–120 picks. First-pick kill magnitude is ã-independent (~4.45 nats), so the search-phase dynamics are unchanged.

## G2 — Perf (81-model space, release, M3)

| Operation | Run 1 | Run 2 (post-clippy-fix) |
|---|---|---|
| (i) `ModelSpace::new` init (O(M×C)) | 989.9 µs | 1251.0 µs |
| (ii) sparse-Δ predictive posterior | **0.84 µs/action** | **1.92 µs/action** |
| (iii) `efe_model_gain` (3 branches) | **2.87 µs/action** | **4.54 µs/action** |
| (iv) naive full recompute | 969.1 µs/action | 983.8 µs/action |
| **Speedup (iv)/(ii)** | **1160×** | **513×** |

Targets: sparse-Δ ≥ 100× vs naive (**PASS** at 513–1160×), per-action single-digit µs (**PASS** — worst observed 4.54 µs). The sparse path does zero lgamma evaluations — a unit count increment reduces to `ln(x_row) − ln(Σx)` exactly via Γ(x+1) = xΓ(x); only the per-branch log-sum-exp remains (O(#models) exp/ln).

## G3 — No-regression

- Default-features `cargo test -p katgpt-core --lib`: **1992 passed, 0 failed, 7 ignored** — bit-identical count to pre-plan baseline (the 23 bmr module tests are feature-gated out here; the best_belief refactor through the shared `special_fn::ln_gamma` kernel is behavior-preserving — its G1 fixtures including the statrs reference pin pass unchanged).
- `cargo clippy -p katgpt-core --features bmr --all-targets -- -D warnings`: **clean**.
- `cargo test -p katgpt-core --no-default-features --features bmr --lib`: **351 passed** (isolation — bmr pulls no other feature).
- `cargo check -p katgpt-core --target wasm32-unknown-unknown --features bmr --lib`: **clean** (pure math + std + arrayvec).

## G4 — Alloc-free hot path

`predictive_posterior_into` + `efe_model_gain` + `accumulate`: **0 allocations over 1000×3 steady-state calls** (per-thread CountingAllocator, `assert_counter_is_live` canary — the Issue 714 pattern). Scratch is caller-supplied (`EfeScratch`, fixed `[f64; MAX_MODELS]` arrays); the gate is feature-unconditional in the bench binary (never `debug_assertions`-only, per the profile rule).

## UQ floor rule (Research 322)

Considered; **not applicable** — model-selection belief, not a predictive interval/coverage/quantile claim. The honest quality gate is the discovery-rate + premature-commit-rate table above, per the plan.

## Anti-FEP scope guard

Discovery axis ONLY — the module header carries the guard (never policy extraction over known models; MOP / HMM control own that axis; Research 551 §2.4). The ×512 novelty-suppression convention is documented in the header as the canonical consumer-side reading; the three-ball harness implements its freeze equivalent post-commit.

## Substrate note (T1.1)

One `ln_gamma` per crate: `best_belief`'s private Lanczos kernel was extracted to `src/special_fn.rs` (bit-identical body — same coefficients, same operation order) and `bmr` consumes the same kernel. Verified by the unchanged default-features test suite (1992/0) and best_belief's statrs-reference numerics pin.

## Promote/demote

**Recommendation: PROMOTE to default** (coordinator's call) — G1–G4 PASS; pure modelless closed-form math (lgamma/Beta arithmetic, zero deps, zero-cost-unless-invoked, zero-alloc hot paths, wasm32-clean); the plan's promotion criterion (modelless gain) is met. No demotion candidate exists (nothing was displaced).
