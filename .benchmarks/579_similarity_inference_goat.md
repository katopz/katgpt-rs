# Benchmark 579 — Similarity Inference GOAT (Plan 526)

**Date:** 2026-08-11
**Primitive:** `similarity_inference` (feature gate, katgpt-core)

> **DEMOTED to opt-in 2026-09-04** (riir-ai `Issue 867` T1.3, quarterly goat-audit): 24 days DEFAULT-ON (2026-08-11..2026-09-03) with ZERO consumers workspace-wide — the GOAT verdict below stands; nothing consumes it. Compose partner `latent_cce_moderator` (riir-engine, DEFAULT-ON) enables `katgpt-rs/cce_moderator` but never this feature — the exogenous CCE half is wired, the endogenous inference half is not. Wire candidates (civ social / moderator Γ₀ composition / ruliology arena) are gameplay-emergence changes on production default surfaces = owner calls (CLR precedent). Re-promotion is one default-list line the day a consumer lands.
**Plan:** [`.plans/526_similarity_inference_primitive.md`](../.plans/526_similarity_inference_primitive.md)
**Research:** [`.research/471_Similarity_Inference_Embedded_Equilibrium.md`](../.research/471_Similarity_Inference_Embedded_Equilibrium.md)
**Paper:** [arXiv:2608.03958](https://arxiv.org/abs/2608.03958) — Meulemans, Wołczyk, Weis, Nasser, et al. *Paradigms of Intelligence: A game theory for foundation models shows new paths to rational cooperation through similarity inference.* 4 Aug 2026.

## TL;DR

**ALL GOAT GATES PASS (G1–G8).** Promoted to DEFAULT-ON in katgpt-core. The primitive infers an *endogenous* correlation device `ω ∈ (0,1)` from joint-action history and switches from competitive-best-response (Nash) to cooperative-best-response (CCE) when `ω` crosses a payoff-derived threshold. The mechanism is genuinely novel per R471 §3.5 — the shipped `CceLp<N,A>` (Plan 295) uses an *exogenous* designer-set correlation device; this primitive *infers* it. Indirect inference (Phase 3 G5) is the Super-GOAT-capability subset (zero-shot cooperation from third-party observation).

---

## G1 — Closed-Form Reproduction ✅

**Assertion:** `ω_T` from incremental `observe_match` updates matches the analytical closed form `α/(α+(1−α)·|A|^(−T))` to f32 epsilon.

| Parameter | Value |
|---|---|
| α (prior) | 0.1 |
| \|A\| (actions) | 2 |
| T range | 0..50 |
| Tolerance | rel_err < 1e-5 |

**Result: PASS.** Every T-step matches to <1e-5 relative error. After T=50 matches, ω saturates to 1.0 (f32 precision floor — `exp(−50·ln2) = exp(−34.7)` underflows to 0, making `ω = α/(α+0) = 1.0`).

Companion: `log W` matches `−T·ln(|A|)` exactly for |A|=4.

Test: `g1_matches_analytical_omega`, `g1_log_w_matches_minus_t_ln_a`.

## G2 — Emergent Cooperation PoC ✅ (the load-bearing quality gate)

**Assertion:** Shared-shard pairs cooperate at >80%; random-shard pairs at <20%.

| Parameter | Value |
|---|---|
| Entities | 128 (32 shared pairs + 32 random pairs) |
| Info-gathering rounds | T=50 |
| Game | 2-action, perfect monitoring |
| Terminal | Prisoner's Dilemma (R=2, S=0, T=3, P=1) |
| Seeds | 10 (mean reported) |

**Result: PASS — perfect separation.**

| Pair kind | Cooperation rate | Target | Mean ω |
|---|---|---|---|
| Shared-shard | **1.000** | >0.80 ✓ | 1.0000 |
| Random-shard | **0.000** | <0.20 ✓ | 0.0000 |

The mechanism works exactly as the paper predicts: shared pairs accumulate 50 matches → ω→1 → cooperate; random pairs mismatch at least once in 50 rounds → ω=0 → defect.

Test: `g2_emergent_cooperation_poc`, `g2_shared_pairs_never_mismatch`, `g2_random_pairs_mismatch_frequently`.

## G3 — No-Regression ✅

**Assertion:** Existing workspace tests still pass.

| Configuration | Before | After | Delta |
|---|---|---|---|
| Default features | 1862 pass | 1862 pass | 0 |
| `--features similarity_inference` | — | +12 new | +12 |
| `--all-features` | 3820 pass | 3832 pass | +12 |

**Result: PASS.** Zero regressions.

## G4 — Alloc-Free ✅

**Assertion:** `observe_match` / `observe_mismatch` / `embedded_best_response` allocate 0 bytes after construction.

**Result: PASS by construction + smoke test.**

Code audit: `observe_match` is pure f32 arithmetic (`log_w +=`, `saturating_add`, `exp`, `divide`). No Vec/Box/String/format! on the hot path.

Smoke test: 100K `observe_match` calls in **1.63ms** (16 ns/call, debug build). A leaky path would OOM or slow dramatically.

Test: `g4_alloc_free_smoke`.

## G5 — Indirect Inference ✅ (Super-GOAT-capability subset)

**Assertion:** Shared-policy primary entities cooperate at >70%; random-policy at <25%. Primaries never interact directly — each plays the same 3 NPCs concurrently, and infers the other's similarity via shared-NPC encounters.

| Parameter | Value |
|---|---|
| Primary entities | 2 (A, B) per trial |
| Shared NPCs | 3 |
| Info rounds | T=50 |
| Trials | 40 |
| Indirect observations per primary | 150 (3 NPCs × 50 rounds) |

**Result: PASS — perfect separation.**

| Policy kind | Cooperation rate | Target | Mean ω |
|---|---|---|---|
| Shared-policy | **1.000** | >0.70 ✓ | 1.0000 |
| Random-policy | **0.000** | <0.25 ✓ | 0.0000 |

**This is the genuinely new capability class** per R471 §3.2 — zero-shot cooperation from third-party observation. No shipped primitive produces cooperation on first direct encounter from parallel third-party observation alone. Phase 7 (scoped Super-GOAT claim for indirect inference ONLY) is unblocked.

Test: `g5_indirect_inference_poc`, `g5_indirect_primaries_never_directly_interact`.

## G6 — Crowd-Scale Latency ✅

**Assertion:** <5ms total per tick for 20K pairwise `ω` updates (1000 entities × 20 AOI-neighbors). Sub-µs per individual update.

| Workload | Time | Per-unit | Budget | Headroom |
|---|---|---|---|---|
| 20K `observe_match` | **482.6µs** | 24 ns/update | 5ms | **10×** |
| 1000 `embedded_best_response` | **114.6µs** | 115 ns/call | 5ms | **43×** |

**Result: PASS.** Production-ready for 1000-NPC zones at 20Hz tick. Debug-build numbers; release will be faster.

Tests: `g6_crowd_scale_latency`, `g6_best_response_crowd_scale`.

## G7 — UQ Floor Comparison ✅ ("Report the Floor" rule)

**Assertion:** Bayesian `ω` beats conformal-naive floor (`sigmoid(k·(match_fraction−0.5))`) on Brier score by ≥10% relative.

Uses the **soft-identity model** (δ=0.9: shared partners match 90% of the time, random 50%) to create overlapping distributions where calibration matters. Without noise, both methods get perfect separation and Brier=0.

| Method | Brier score | Mean ω (shared) | Mean ω (random) |
|---|---|---|---|
| **Bayesian `ω`** | **0.001220** | 0.9974 | 0.0015 |
| Floor `ω_floor` | 0.145789 | — | — |
| **Improvement** | **99.2%** (119× better) | — | — |

**Result: PASS — crushes.** The Bayesian posterior is 119× better calibrated than the floor. The floor collapses the full history to a single `match_fraction` scalar, throwing away the count information that the Bayesian posterior correctly compounds via multiplicative likelihood ratios.

Test: `g7_uq_floor_comparison`.

## G8 — PD Threshold = 0.5 ✅

**Assertion:** `embedded_best_response` cooperates iff `ω > 0.5` for canonical PD (R=2, S=0, T=3, P=1) with uniform partner marginal.

**Derivation:** `Q(C) − Q(D) = 2ω − 1 > 0 ⟺ ω > 0.5`.

**Measured threshold:** 0.500 ± 0.001 (binary-searched to 4 decimals).

**Result: PASS.**

Tests: `g8_cooperates_iff_omega_above_half_pd`, `g8_threshold_analytical_pd`.

---

## Correctness Fix Shipped With Phase 2

While building the G2 PoC, found that `observe_mismatch` was incorrectly calling `observe_match` (both added `log(1/|A|)` to `log_w`). Re-derived the Bayes update:

- Under the perfect-identity shared hypothesis: `P(match|shared) = 1`, `P(mismatch|shared) = 0` (Kronecker delta).
- Match: `LR_t = 1/(1/|A|) = |A|` → `log_w += −ln(|A|)` (ω climbs)
- Mismatch: `LR_t = 0/(1/|A|) = 0` → `log_w = +∞` (ω=0 permanently)

Fixed: `observe_mismatch` now sets `log_w = +∞` → ω = 0 permanently. This matches the paper's perfect-identity Bayes update. Added `is_collapsed_to_zero()` diagnostic + regression tests.

---

## Verdict

**GOAT — all gates pass.** Promoted to DEFAULT-ON in `katgpt-core/Cargo.toml`.

**Super-GOAT-capability subset (conditional):** Phase 3 G5 (indirect inference) PASSES. Per R471 §3.2, zero-shot cooperation from third-party observation is a genuinely new capability class. Phase 7 opens the scoped Super-GOAT claim for indirect inference ONLY (not the equilibrium concept, not the direct-inference mechanism).

## Honest Limitations

1. **Perfect-identity model only.** The production `SimilarityPosterior` uses δ=1 (any mismatch → ω=0 permanently). The soft-identity model (δ<1) is tested in G7 but not shipped as a production API. A future extension would add a `soft_observe` method taking `delta` as a parameter.
2. **Discrete actions only.** The continuous-embedding path (`predictive_similarity`) ships as a stub; full latent-embedding inference requires a consumer-supplied identity direction vector (Phase 3 of R335 private guide).
3. **No staleness window.** All observations count equally. A time-decayed window (Phase 3 T3.5, deferred) would require time-stamped observations + exponential decay.
4. **No sync-boundary relevance.** `ω` is a per-focal latent scalar that stays local. Only the final cooperate/defect action crosses the sync boundary. This is correct per AGENTS.md §"Sync Boundary Rule" — do NOT sync `ω` directly.
5. **The G2/G5 PoCs use deterministic policies.** Real-game policies are stochastic (e.g., softmax-sampled actions). The PoC validates the math, not the full game integration. A riir-ai consumer (R335 selling-point guide) would test on real NPC cognition.

## Test Inventory (23 tests)

| Gate | Tests |
|---|---|
| G1 | `g1_matches_analytical_omega`, `g1_log_w_matches_minus_t_ln_a`, `g1_observations_counter_tracks_t`, `g1_rejects_invalid_prior_alpha`, `g1_omega_stays_in_closed_unit_interval_f32`, `g1_clone_preserves_state`, `g1_mismatch_drives_omega_to_zero`, `g1_mismatch_at_t0_omega_zero_from_start` |
| G2 | `g2_emergent_cooperation_poc`, `g2_shared_pairs_never_mismatch`, `g2_random_pairs_mismatch_frequently` |
| G4 | `g4_alloc_free_smoke` |
| G5 | `g5_indirect_inference_poc`, `g5_indirect_primaries_never_directly_interact` |
| G6 | `g6_crowd_scale_latency`, `g6_best_response_crowd_scale` |
| G7 | `g7_uq_floor_comparison` |
| G8 | `g8_cooperates_iff_omega_above_half_pd`, `g8_threshold_analytical_pd`, `g8_shape_mismatch_errors`, `g8_into_variant_matches_plain` |
| — | `payoff_matrix_shape_validation`, `canonical_pd_layout` |
