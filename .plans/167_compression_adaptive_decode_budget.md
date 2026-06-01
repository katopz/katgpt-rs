# Plan 055: Compression-Adaptive Decode Budget — PFlash Complexity Signal

**Date:** 2026-06-01
**Status:** 📋 Ready — all prerequisites exist
**Research:** R050 (PFlash Compression as Complexity Signal)
**Feature Gate:** `budget_adaptation` (default-OFF until GOAT proof)
**Cross-ref:** Plan 026 (Domain Inference Budget), Plan 057 (MTP Budget Propagation), riir-ai P179 (PFlash benchmarks)

---

## Goal

Use the prompt compression ratio (a free byproduct of prefill scoring) to dynamically scale DDTree budget per-prompt. Simple prompts → less search. Complex prompts → more search. Zero additional compute cost.

---

## Optimization Alignment

Per `.contexts/optimization.md`:
- ✅ "Profile first" — Plan 179 profiled PFlash vs Naive; we know where compression helps
- ✅ "Pre-compute values that don't change across samples" — compression ratio computed once per prompt
- ✅ "Don't: Recompute unchanged values" — ratio is static per prompt, derived from existing attention
- ✅ "Cache allocations" — no new allocations, just scaling an existing integer

---

## Phase 1: Budget Derivation Function

### Task 1 — Add `BudgetAdaptation` enum
- [ ] T1: Add `BudgetAdaptation` enum to `speculative/types.rs`
  ```rust
  pub enum BudgetAdaptation {
      Off,              // current behavior: fixed budget
      Compression,      // scale by compression ratio from attention scores
      Entropy,          // scale by first-marginal entropy
  }
  ```
- [ ] Add to `FlashPrefillConfig` or new `AdaptiveBudgetConfig` struct
- [ ] Default: `Off` (current behavior preserved)

### Task 2 — Implement `adaptive_tree_budget()` function
- [ ] T2: Add to `speculative/prefill.rs` or new module `speculative/budget.rs`
  ```rust
  /// Derive per-prompt tree_budget from base + complexity signal.
  /// Returns budget clamped to [base/2, base*2].
  pub fn adaptive_tree_budget(
      base_budget: usize,
      compression_ratio: f32,  // r ∈ (0, 1]: fraction of tokens that matter
      mode: BudgetAdaptation,
  ) -> usize {
      match mode {
          BudgetAdaptation::Off => base_budget,
          BudgetAdaptation::Compression => {
              // High r = complex → more budget. Low r = simple → less budget.
              let scale = 0.5 + 1.5 * compression_ratio; // f(0)=0.5, f(0.5)=1.25, f(1)=2.0
              ((base_budget as f32 * scale) as usize)
                  .max(base_budget / 2)
                  .min(base_budget * 2)
          }
          BudgetAdaptation::Entropy => {
              // TODO: derive from first-marginal entropy
              base_budget
          }
      }
  }
  ```
- [ ] Unit tests: verify clamping, verify scaling curve

### Task 3 — Extract compression ratio from existing scoring
- [ ] T3: Add `compression_ratio()` to `PrefillScorer` trait (or as free function)
  - Given attention scores and alpha threshold, compute: `r = count(score > alpha * max_score) / total`
  - This is already computed inside `block_select` — extract it as a return value
- [ ] Zero-alloc: return `f32` ratio alongside existing `Vec<usize>` block indices
- [ ] Or: compute from `BlockScores` without extra allocation

---

## Phase 2: Wiring into DDTree Dispatch

### Task 4 — Pass adaptive budget to `speculative_step`
- [ ] T4: Modify `speculative_step*` functions to accept `effective_budget: usize`
  - Currently uses `config.draft_lookahead` (derived from domain tree_budget)
  - Add optional override: if `budget_adaptation != Off`, use adaptive budget
- [ ] Backward compatible: default `effective_budget = None` → current behavior

### Task 5 — Wire into domain config
- [ ] T5: Add `budget_adaptation` field to domain config (TOML)
  ```toml
  [domain.inference]
  tree_budget = 2374
  budget_adaptation = "compression"  # or "entropy" or "off"
  ```
- [ ] Parse in domain config loader
- [ ] Pass through to speculative step dispatch

### Task 6 — Wire into DFlash marginals
- [ ] T6: Adjust `draft_lookahead` proportionally when budget changes
  - If adaptive_budget is 2× base, draft_lookahead scales ~1.4× (sqrt relationship)
  - If adaptive_budget is 0.5× base, draft_lookahead scales ~0.7×
  - Rationale: more tree_budget → more branches → more lookahead to fill them

---

## Phase 3: GOAT Proof

### Task 7 — Correctness: adaptive budget produces same acceptance pattern
- [ ] T7: `test_adaptive_budget_matches_fixed_at_midpoint`
  - When compression_ratio = 0.5 (medium complexity), adaptive budget ≈ base budget
  - Output tokens identical to fixed-budget run
- [ ] `test_adaptive_budget_clamped` — verify budget stays within [base/2, base*2]
- [ ] `test_adaptive_budget_off_matches_current` — Off mode = exact current behavior

### Task 8 — Benchmark: heterogeneous prompt complexity
- [ ] T8: Create test with prompts of varying complexity:
  - Simple prompt (boilerplate code): compression_ratio ≈ 0.05 → budget halved
  - Medium prompt (mixed): compression_ratio ≈ 0.40 → budget ~base
  - Complex prompt (dense logic): compression_ratio ≈ 0.80 → budget doubled
- [ ] Measure: tokens/second, acceptance rate, total decode time
- [ ] Expectation: simple prompts faster (less wasted search), complex prompts same or better (more search)

### Task 9 — Benchmark: no regression on fixed prompts
- [ ] T9: Run existing DDTree benchmarks with `budget_adaptation = Off` (control)
- [ ] Run same benchmarks with `budget_adaptation = Compression`
- [ ] Verify: no regression (within ±2% noise floor)

### Task 10 — Promotion decision
- [ ] T10: If T7-T9 pass → promote `budget_adaptation` to default-ON
- [ ] If T8 shows <5% gain → keep default-OFF, document as opt-in
- [ ] If T9 shows regression → investigate, keep default-OFF

---

## Success Criteria

| Gate | Criterion | Measurement |
|------|-----------|-------------|
| G1 | Correctness | Midpoint (r=0.5) produces same output as fixed budget |
| G2 | Clamping | Budget stays within [base/2, base*2] for all r ∈ [0, 1] |
| G3 | No regression | Fixed prompts: same tok/s ±2% |
| G4 | Gain | Heterogeneous prompts: ≥5% improvement in decode efficiency |
| G5 | Off = current | `budget_adaptation = "off"` is bit-identical to current behavior |

---

## Scope

- **IN:** BudgetAdaptation enum, adaptive_tree_budget(), wiring into domain config + DDTree dispatch
- **OUT:** Entropy mode implementation (T2 placeholder), PFlash compression changes (this doesn't change PFlash), GPU kernel changes (pure CPU logic)
