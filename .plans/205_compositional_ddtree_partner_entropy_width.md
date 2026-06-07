# Plan 205: Compositional DDTree Partner-Entropy Width

**Research:** 181 (Compositional Muon — Partner-Weighted Inference)
**Feature Gate:** `comp_width`
**Status:** Done
**Priority:** MEDIUM — clean improvement, ~50 LOC

---

## Motivation

Compositional Muon shows that when two functions compose (f∘g), controlling the composition's perturbation ‖Δ(f∘g)‖ is better than controlling each factor independently. In DDTree, the "composition" is:
- draft marginals × validator relevance = joint token score
- Currently: `PEAK_DOMINANCE_RATIO` (binary threshold) decides DDTree width
- CM's isotropic approximation says: rescale each factor by partner's "norm" (scalar)

The insight: entropy of the draft distribution is the "partner norm" for the validator, and vice versa. High-entropy draft = many tokens compete = validator needs more budget.

## Tasks

- [x] Add `compositional_width()` function in `crates/katgpt-core/src/mux/dd_tree.rs`
  - Takes: peaks (top-K logits), base width
  - Returns: scaled width as `usize`
  - Formula: `width = max(1, round(base * (entropy / max_entropy)))` — linear entropy→width mapping
  - Zero-alloc, branch-free inner loop
  
- [x] Feature-gate behind `comp_width` in `Cargo.toml` (both crate and workspace)
  
- [x] Replace `PEAK_DOMINANCE_RATIO` usage with continuous partner-entropy scaling
  - Updated `MuxDdTree::detect_width` in `dd_tree.rs` — feature-gated
  - Updated `MuxBfs::detect_width` in `bfs.rs` — delegates to `MuxDdTree::detect_width` when enabled
  
- [x] Add unit test: `comp_width_monotonic_with_entropy` — higher entropy gives >= width
  
- [x] Add unit test: `comp_width_zero_entropy_returns_min` — zero entropy gives width 1
  
- [ ] Add GOAT gate proof: benchmark before/after on multi-peak token distributions
  - Use existing DDTree test harness
  - Compare: fixed width vs PEAK_DOMINANCE_RATIO vs compositional_width
  - Expected: compositional_width matches or beats both in acceptance rate per total compute
  
- [ ] Add benchmark: overhead of entropy calculation (should be negligible — already computed)

## Implementation Notes

- The isotropic CM approximation for this is literally:
  ```
  s = (entropy / max_entropy + damping).recip().sqrt()
  width = base * s
  ```
  This is ONE division, ONE sqrt, ONE multiply. Zero-alloc, branch-free.

- Reference implementation: `.raw/comp-muon-release/src/whitening.py::isotropic_scale()`
  ```
  s = (||W||_F^2 / d_h + damping)^{-1/2}
  ```
  Replace `||W||_F^2 / d_h` with `entropy / max_entropy` (same normalization structure).

- Gauge correction (Fusion 3 from Research 181) is NOT in this plan. Profile first.

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| DDTree width control | Binary (peak/not-peak) | Continuous (partner-entropy scaled) |
| Multi-peak acceptance | Fixed width | Adaptive width |
| Overhead | 0 | ~3ns (one sqrt + one multiply) |
| Code change | — | ~50 lines |

## Why Not More

CM's core value (partner-whitened gradient updates) is training-only. The modelless path gets the *principle* (control the composition, not the factors) but not the mechanism (gradient whitening). The only high-ROI transfer is this scalar rescaling. Everything else in Research 181 is low priority.
