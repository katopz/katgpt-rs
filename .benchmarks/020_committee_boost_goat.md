# Benchmark 020: Committee Boost — GOAT Proofs

**Plan:** 132 — Committee Boost — Oracle-Gap Recovery, Debiasing, Budget Sizing
**Research:** 093 — Agentic Systems as Boosting Weak Reasoning Models
**Feature Gate:** `committee_boost = ["bt_rank", "bandit"]`
**Date:** 2026-05-29 (updated 2026-05-31 with GOAT benchmark T24)

---

## Test Results

### Phase 1: Oracle-Gap Recovery Metric (T1–T6)

| Test | Result |
|------|--------|
| `test_recovery_known_values` (Rec = 0.24/0.30 = 0.8) | ✅ PASS |
| `test_recovery_perfect` (Rec = 1.0) | ✅ PASS |
| `test_recovery_no_improvement` (Rec = 0.0) | ✅ PASS |
| `test_recovery_zero_gap` (None) | ✅ PASS |
| `test_failure_mode_coverage_limited` (Rec > 0.5) | ✅ PASS |
| `test_failure_mode_selection_limited` (Rec < 0.5) | ✅ PASS |
| `test_failure_mode_mixed` (Rec = 0.5) | ✅ PASS |
| `test_failure_mode_display` | ✅ PASS |
| `test_diagnostic_contains_recovery_pct` | ✅ PASS |
| `test_diagnostic_zero_gap` | ✅ PASS |

### Phase 2: Position-Swap Debiasing (T7–T11)

| Test | Result |
|------|--------|
| `test_identical_inputs_tie` | ✅ PASS |
| `test_symmetric_comparison_returns_tie` | ✅ PASS |
| `test_asymmetric_agreement_correct_winner` | ✅ PASS |
| `test_asymmetric_disagreement_tie` | ✅ PASS |
| `test_one_side_tie_produces_tie` | ✅ PASS |
| `test_tournament_consistent_ranking` | ✅ PASS |
| `test_tournament_pair_count` | ✅ PASS |
| `test_tournament_zero_candidates` | ✅ PASS |
| `test_tournament_single_candidate` | ✅ PASS |
| `test_debiased_compare_free_function` | ✅ PASS |

### Phase 3: Budget Sizing from Theory (T12–T18)

| Test | Result |
|------|--------|
| `test_budget_basic_sizing` | ✅ PASS |
| `test_budget_small_depth` | ✅ PASS |
| `test_budget_large_portfolio` | ✅ PASS |
| `test_budget_tighter_delta_needs_more` | ✅ PASS |
| `test_budget_high_alpha_fewer_proposers` | ✅ PASS |
| `test_budget_high_beta_fewer_critic` | ✅ PASS |
| `test_budget_high_sigma_fewer_comparisons` | ✅ PASS |
| `test_budget_deterministic` | ✅ PASS |
| `test_budget_equality` | ✅ PASS |
| `test_budget_reasonable_range` | ✅ PASS |
| `test_budget_error_display` | ✅ PASS |
| `test_reject_zero_depth` | ✅ PASS |
| `test_reject_zero_portfolio` | ✅ PASS |
| `test_reject_delta_out_of_range` | ✅ PASS |
| `test_reject_alpha_out_of_range` | ✅ PASS |
| `test_reject_beta_out_of_range` | ✅ PASS |
| `test_reject_sigma_out_of_range` | ✅ PASS |
| `test_validate_ok` | ✅ PASS |
| `test_validate_k_zero` | ✅ PASS |
| `test_validate_m_zero` | ✅ PASS |
| `test_validate_r_zero` | ✅ PASS |
| `test_total_role_calls_formula` | ✅ PASS |
| `test_total_role_calls_single_round` | ✅ PASS |
| `test_total_role_calls_zero_depth` | ✅ PASS |
| `test_total_role_calls_scales_with_depth` | ✅ PASS |
| `test_total_role_calls_depth_2` | ✅ PASS |
| `test_alpha_one_is_valid` | ✅ PASS |

### Phase 4: Blind-Spot Floor Estimation (T19–T23)

| Test | Result |
|------|--------|
| `test_saturation_at_0_8_gives_b_0_2` | ✅ PASS |
| `test_monotonic_increase_b_near_zero` | ✅ PASS |
| `test_single_point_b_is_one_minus_rate` | ✅ PASS |
| `test_empty_input_b_is_one` | ✅ PASS |
| `test_convergence_fit_empty` | ✅ PASS |
| `test_convergence_fit_single_point` | ✅ PASS |
| `test_convergence_fit_converged` | ✅ PASS |
| `test_convergence_fit_not_converged` | ✅ PASS |
| `test_convergence_rate_positive` | ✅ PASS |
| `test_diagnostic_diversify_proposers` | ✅ PASS |
| `test_diagnostic_increase_k` | ✅ PASS |
| `test_diagnostic_adequate` | ✅ PASS |
| `test_diagnostic_max_k_and_oracle` | ✅ PASS |
| `test_unsorted_input` | ✅ PASS |
| `test_flat_rates_converged` | ✅ PASS |

---

## Overall Status: ✅ GOAT 68/68 + 7/7 GOAT Benchmark PASS

All Phase 1–4 unit tests pass (68/68). Phase 5 GOAT benchmark (T24) also passes (7/7).

---

## GOAT Benchmark Results (T24)

Run: `cargo test --features committee_boost --test bench_committee_boost_goat -- --nocapture`

| Proof | Description | Result |
|-------|-------------|--------|
| G1 | Oracle-gap recovery: Rec within ±0.01 for 6 known cases | ✅ PASS |
| G2 | Debiased comparison: 100% Tie rate for biased comparator (45 pairs) | ✅ PASS |
| G2b | Debiasing catches lead-position bias (6 false rankings eliminated) | ✅ PASS |
| G3 | Budget sizing: Theorem 3 monotonicity + determinism (k=40, m=34, r=21) | ✅ PASS |
| G3b | Budget rejects all invalid parameters | ✅ PASS |
| G4 | Blind-spot floor: 8 cases verified (B estimation, convergence, diagnostics) | ✅ PASS |
| G5 | End-to-end: committee improves 29.8% over single-shot (≥5% target) | ✅ PASS |

Budget at paper parameters (L=10, δ=0.05, α=0.3, β=0.2, σ=0.4, |P_N|=2): k=40, m=34, r=21, total_role_calls=350,000.
