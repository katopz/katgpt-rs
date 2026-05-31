# Plan 132: Committee Boost — Oracle-Gap Recovery, Debiasing, Budget Sizing

> **Research:** [093 — Agentic Systems as Boosting Weak Reasoning Models](../.research/093_Boosting_Weak_Reasoning_Committee_Search.md)
> **Paper:** [arXiv:2605.14163](https://arxiv.org/pdf/2605.14163) — Verifier-backed committee search as inference-time boosting
> **Feature Gate:** `committee_boost` (**Opt-in**, requires GOAT proof before default-on promotion)
> **Depends on:** Plan 030 (Bandit), `bt_rank` feature (BtRank), speculative module (DDTree + verifier)
> **Status:** ✅ Complete (T1–T26) · GOAT 7/7 PASS

## Summary

Implement four diagnostics from the boosting committee paper that our architecture already supports conceptually but lacks as measurable metrics:

1. **Oracle-gap recovery** — `Rec = (p_system - p1) / (p_oracle - p1)` tells us whether failures are selection (low Rec → improve critic/comparator) or coverage (high Rec, low p_oracle → improve proposer diversity)
2. **Position-swap debiasing** — Compare pairs in both A/B orders; count win only if both agree. Eliminates lead-position bias in BtRank
3. **Budget sizing from theory** — Given (α₀, β₀, σ₀, L, δ), compute optimal (k, m, r) per paper's Theorem 3
4. **Blind-spot floor estimation** — Measure B = 1 - lim_{k→∞} p_oracle(k) to find the proposer diversity ceiling

The paper proves our DDTree + BtRank + ScreeningPruner stack IS the committee protocol Π_{k,m,r}. These additions make the theoretical guarantees **measurable and actionable**.

**Target: GOAT-proof oracle-gap recovery metric + debiased BtRank comparison + principled budget sizing.**

---

## Tasks

### Phase 1: Oracle-Gap Recovery Metric
- [x] **T1**: Create `src/pruners/committee_boost/mod.rs` — module index, re-exports, `#[cfg(feature = "committee_boost")]` gate
- [x] **T2**: Create `src/pruners/committee_boost/types.rs` — `OracleGapRecovery` struct with `p1`, `p_oracle`, `p_system` fields
- [x] **T3**: Implement `OracleGapRecovery::recovery()` — returns `(p_system - p1) / (p_oracle - p1)`, handles NaN/zero-gap
- [x] **T4**: Implement `OracleGapRecovery::failure_mode()` — returns enum `SelectionFailure | CoverageFailure | Mixed` based on recovery value
- [x] **T5**: Implement `OracleGapRecovery::diagnostic()` — human-readable breakdown: "Recovery=78.3%: selection recovers most latent capability; focus on proposer diversity for further gains"
- [x] **T6**: Unit tests: recovery() with known values (p1=0.5, p_oracle=0.8, p_system=0.74 → Rec=0.8), edge cases (p_oracle=p1 → NaN, p_system=p_oracle → 1.0)

### Phase 2: Position-Swap Debiasing
- [x] **T7**: Create `src/pruners/committee_boost/debiased_compare.rs` — `debiased_compare<F>(i, j, compare: &F) -> BtOutcome`
- [x] **T8**: Implement A/B swap logic: compare(i,j) and compare(j,i), map reversed result back, require agreement for win/loss, disagreement → Tie
- [x] **T9**: Implement `DebiasedComparator` struct wrapping comparison function with swap-debiasing
- [x] **T10**: Implement `DebiasedComparator::tournament()` — run debiased pairwise comparisons over all pairs, collect `Vec<BtComparison>` for `bt_fit()`
- [x] **T11**: Unit tests: symmetric comparison (same input → Tie), asymmetric with agreement → correct winner, asymmetric with disagreement → Tie

### Phase 3: Budget Sizing from Theory
- [x] **T12**: Create `src/pruners/committee_boost/budget.rs` — `CommitteeBudget` struct with `k`, `m`, `r` fields
- [x] **T13**: Implement `committee_budget(depth, delta, alpha, beta, sigma, portfolio_size) -> CommitteeBudget` per paper Theorem 3:
  - `k ≥ |P_N| × ⌈ln(2L/δ) / α₀⌉`
  - `m ≥ ⌈(1/2β₀) × ln(2k²L/δ)⌉`
  - `r ≥ ⌈(1/4σ₀²) × ln(2k²L/δ)⌉`
- [x] **T14**: Implement `total_role_calls(&self, depth: usize) -> usize` — O(L × (k + mk + rk²))
- [x] **T15**: Implement `CommitteeBudget::validate()` — sanity checks (k ≥ 1, m ≥ 1, r ≥ 1, alpha/beta/sigma in (0,1])
- [x] **T16**: Unit tests: sizing matches paper examples, total_role_calls formula, edge cases (very small/large parameters)

### Phase 4: Blind-Spot Floor Estimation
- [x] **T17**: Create `src/pruners/committee_boost/blind_spot.rs` — `BlindSpotEstimate` struct
- [x] **T18**: Implement `estimate_blind_spot_floor(oracle_rates: &[(usize, f64)]) -> f64` — B ≈ 1 - max(oracle_rates)
- [x] **T19**: Implement `fit_convergence(oracle_rates: &[(usize, f64)]) -> ConvergenceFit` — exponential fit to estimate saturation point
- [x] **T20**: Implement `coverage_diagnostic(oracle_rates: &[(usize, f64)]) -> CoverageDiagnostic` — report B, R_k residual, recommended action (diversify proposers vs increase k)
- [x] **T21**: Unit tests: saturation at 0.8 → B=0.2, monotonic increase → B near 0, single point → B = 1-rate

### Phase 5: Integration & GOAT Proof
- [x] **T22**: Add `committee_boost = ["bt_rank", "bandit"]` feature gate to `Cargo.toml`
- [x] **T23**: Add `#[cfg(feature = "committee_boost")] pub mod committee_boost;` to `src/pruners/mod.rs`
- [x] **T24**: Create `tests/bench_committee_boost_goat.rs` — GOAT proof benchmark:
  - (P1) Oracle-gap recovery: Rec within ±0.01 for 6 known cases ✅
  - (P2) Debiased comparison: 100% Tie rate for biased comparator ✅
  - (P3) Budget sizing: Theorem 3 monotonicity + determinism ✅
  - (P4) Blind-spot floor: 8 cases verified ✅
  - (P5) End-to-end committee ≥5% over single-shot ✅
- [x] **T25**: Add benchmark results to `.benchmarks/020_committee_boost_goat.md`
- [x] **T26**: Update `README.md` — add Committee Boost section under GOAT Proofs, reference Research 093

---

## Architecture

### Module Structure

```text
src/pruners/committee_boost/
├── mod.rs                    # Module index, re-exports
├── types.rs                  # OracleGapRecovery, FailureMode enum
├── debiased_compare.rs       # Position-swap debiasing for BtRank
├── budget.rs                 # CommitteeBudget, committee_budget() sizing
└── blind_spot.rs             # BlindSpotEstimate, coverage diagnostic
```

### Key Types

```rust
/// Oracle-gap recovery: how much latent capability the selector recovers.
#[derive(Debug, Clone)]
pub struct OracleGapRecovery {
    pub p1: f64,        // Single-shot (Pass@1)
    pub p_oracle: f64,  // Best-of-k with perfect selector
    pub p_system: f64,  // Deployed harness
}

/// Failure mode diagnosis from recovery fraction.
#[derive(Debug, Clone, PartialEq)]
pub enum FailureMode {
    /// Rec > 0.7: selector works, proposer needs diversity
    CoverageLimited,
    /// Rec < 0.3: selector struggles, improve critic/comparator
    SelectionLimited,
    /// 0.3 ≤ Rec ≤ 0.7: mixed, both need work
    Mixed,
}

/// Committee budget from theoretical sizing rules.
#[derive(Debug, Clone)]
pub struct CommitteeBudget {
    pub k: usize,  // Proposer width
    pub m: usize,  // Critic depth per candidate
    pub r: usize,  // Comparator votes per pair
}

/// Blind-spot floor estimate from oracle best-of-k curve.
#[derive(Debug, Clone)]
pub struct BlindSpotEstimate {
    pub blind_spot_floor: f64,  // B ≈ 1 - max(p_oracle)
    pub oracle_at_max_k: f64,   // p_oracle at largest k tested
    pub max_k: usize,           // Largest k tested
}
```

### Integration with Existing Stack

```text
DDTree (k branches)
  ↓ proposals
ScreeningPruner (m critic votes) ─── committee_boost::CommitteeBudget sizes m
  ↓ survivors
BtRank (r comparator votes) ─────── committee_boost::DebiasedComparator wraps comparison
  ↓ Copeland winner
ConstraintPruner (verifier) ──────── committee_boost::OracleGapRecovery measures recovery
  ↓
CommitteeBoost::diagnostic() ─────── reports Rec, failure mode, blind-spot floor
```

---

## GOAT Proof Criteria

| # | Proof | Metric | Pass Threshold |
|---|-------|--------|----------------|
| G1 | Oracle-gap recovery is computed correctly | Rec formula matches hand calculation | Rec within ±0.01 of expected |
| G2 | Debiased comparison eliminates position bias | Symmetric inputs → 100% Tie | Tie rate = 100% for identical inputs |
| G3 | Budget sizing matches paper Theorem 3 | k, m, r match paper examples | Exact match for paper parameters |
| G4 | Blind-spot floor estimates correctly | B ≈ 1 - max(oracle_rates) | B within ±0.05 of true floor |
| G5 | End-to-end: Committee protocol improves over single-shot | p_system > p1 with k > 1 | p_system ≥ p1 + 5% on Bomber |

---

## Estimated Scope

| Component | LOC | Complexity |
|-----------|-----|------------|
| types.rs (OracleGapRecovery + FailureMode) | ~80 | Low |
| debiased_compare.rs | ~60 | Low |
| budget.rs | ~70 | Low |
| blind_spot.rs | ~80 | Low |
| mod.rs + re-exports | ~30 | Trivial |
| GOAT benchmark test | ~120 | Medium |
| **Total** | **~440** | **Low** |

---

## References

- Paper: https://arxiv.org/pdf/2605.14163
- Research: `.research/093_Boosting_Weak_Reasoning_Committee_Search.md`
- Related plans: Plan 030 (Bandit), Plan 040 (BtRank), Plan 112 (SR²AM)
- Key code:
  - `src/pruners/bt_rank.rs` — BtRank (comparator)
  - `src/speculative/dd_tree.rs` — DDTree (proposer)
  - `src/speculative/types.rs` — ScreeningPruner (critic)