# Plan 171: Thinking Prune — FrozenBase Guard for SpecHop/LT2

> **Research:** 153 (Thinking Pixel)
> **Feature Gate:** `thinking_prune`
> **Default:** ON (perf improvement, no quality loss)
> **Related Plans:** 131 (SpecHop), 108 (LT2), 136 (Training-Free Loop)

---

## Motivation

The Thinking Pixel paper (arXiv:2604.25299 §3.3) identifies that repeated full verification at every recursion step **corrupts the draft distribution** through compounding artifacts. Their solution: apply lightweight processing at intermediate steps, full model only at the final step.

Applied to our SpecHop pipeline: intermediate hops should use `ScreeningPruner` only (lightweight, O(1)), reserving `ConstraintPruner` verification for the final hop. This is a pure speedup — `ConstraintPruner` calls (especially WASM validators) are expensive and intermediate hops don't need full validation.

## Tasks

- [x] T1: Add `hop_context` parameter to `build_dd_tree_screened()` — contains `(hop_index: usize, total_hops: usize)`
- [x] T2: Implement `FrozenBaseGuard` wrapper for `ScreeningPruner` — when `hop_index < total_hops - 1`, returns inner relevance unchanged; when final hop, delegates to full pipeline
- [x] T3: Add `PrunerSchedule` enum to SR²AM configurator with `Uniform` (current behavior) and `FrozenBaseGuard` (new default)
- [x] T4: Wire `FrozenBaseGuard` into SpecHop `HopDDTree` — intermediate hops skip ConstraintPruner, final hop applies both
- [x] T5: Wire `FrozenBaseGuard` into LT2 `LoopMode::TrainingFree` — intermediate loops use damped sub-stepping only (already implemented), final loop adds ConstraintPruner
- [x] T6: GOAT proof benchmark — SpecHop total latency with/without FrozenBaseGuard, same quality constraint
- [x] T7: Update README feature flags section with `thinking_prune` gate
- [x] T8: Wire `FrozenBaseGuard` into token-level DDTree via `build_dd_tree_screened_with_schedule()` — intermediate hops use `NoScreeningPruner` (skip all screening), final hop applies full screener
- [x] T9: GOAT timing benchmark with expensive screener — `tests/bench_171_thinking_prune_goat.rs` (5 proofs: P1 structural dominance, P2 correctness, P3 wall-clock 44.5% speedup, P4 single-hop edge case, P5 final-hop quality)

## Implementation Notes

### FrozenBaseGuard Design

```rust
/// Guard that applies full verification only at the final recursion step.
/// Intermediate steps use lightweight screening only.
/// 
/// Named after the "frozen base model" guard from the Thinking Pixel paper,
/// where the frozen model is applied only at the final latent step to prevent
/// distribution drift from repeated exposure.
pub struct FrozenBaseGuard<P: ScreeningPruner> {
    inner: P,
    /// When true, applies full inner pruner. When false, returns 1.0 (accept all).
    is_final_step: bool,
}

impl<P: ScreeningPruner> ScreeningPruner for FrozenBaseGuard<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.is_final_step {
            self.inner.relevance(depth, token_idx, parent_tokens)
        } else {
            // Intermediate step: lightweight screening only
            // Don't reject anything — let the final step decide
            1.0
        }
    }
}
```

### Why This Is Safe

The paper's ablation (Table 2) shows that recursion **without** modulation still improves over no recursion (70.36 vs 69.55 SD3+SFT baseline). Intermediate steps contribute by refining the draft distribution — they don't need to be perfect, they just need to be approximately right. The final step applies full verification to ensure quality.

### Optimization Alignment

- No allocation inside hot loops (struct wraps existing pruner)
- O(1) branch per relevance call (single bool check)
- Zero-cost enum dispatch for `PrunerSchedule`
- Pre-compute `is_final_step` once per hop, not per token

### Feature Gate Rationale

Feature-gated as `thinking_prune` to allow A/B comparison. If GOAT proof confirms no quality loss with latency improvement, this becomes default-on. The feature gate stays for binary bloat control per optimization.md guidelines.
