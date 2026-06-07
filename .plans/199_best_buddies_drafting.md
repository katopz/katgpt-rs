# Plan 199: Best Buddies Drafting — Mutual NN Filter for Speculative Decoding

**Date**: 2026-06-07
**Status**: Plan
**Research**: `.research/178_Rosetta_Neurons_Cross_Model_Alignment.md` (Section 2.1)
**GOAT Rank**: #1 (highest impact × lowest effort)

---

## Context

Rosetta Neurons paper shows that mutual nearest neighbors ("best buddies") between model activations produce reliable cross-model correspondences. We apply this to speculative decoding: filter draft model marginals to only tokens where draft and target models **agree bidirectionally** (draft top-K contains target's preference AND vice versa).

Current pipeline: draft model produces marginals → DDTree explores → target verifies.
New pipeline: draft model produces marginals → **BestBuddy filter** → DDTree explores smaller, higher-quality tree → target verifies → higher acceptance rate.

---

## Architecture

```rust
/// Best Buddies filter for speculative decoding.
///
/// Mines mutual agreement between draft and target model marginals.
/// Zero training — purely inference-time correlation.
pub trait BestBuddyAligner: Send + Sync {
    /// Compute mutual agreement score for a token position.
    /// Returns 1.0 if token is a best buddy (draft prefers it AND target prefers it),
    /// 0.0 otherwise. Continuous in between via EMA correlation.
    fn mutual_agreement(&self, draft_top_k: &[f32], target_top_k: &[f32]) -> f32;

    /// Batch version: compute alignment confidence for all positions.
    fn batch_alignment_confidence(
        &self,
        draft_logits: &[f32],    // [seq_len × vocab_size]
        target_logits: &[f32],   // [seq_len × vocab_size]
        results: &mut [f32],     // [seq_len]
    );
}
```

---

## Tasks

- [ ] Implement `pearson_correlation(a: &[f32], b: &[f32]) -> f32` in `katgpt-core/src/math.rs` (SIMD-friendly)
- [ ] Implement `best_buddies(corr_rows: &[&[f32]], k: usize) -> Vec<(usize, usize)>` in `katgpt-core/src/math.rs`
- [ ] Add `BestBuddyAligner` trait to `katgpt-core/src/traits.rs`
- [ ] Implement `MarginalBestBuddyAligner` in `katgpt-rs/src/speculative/` using `SpeculativeContext` marginals
- [ ] Integrate into `build_dd_tree_speculative`: filter marginals by mutual agreement before tree construction
- [ ] Add feature flag `best_buddies` (opt-in initially, default-on after GOAT proof)
- [ ] Write test: before/after speculative acceptance rate with and without best buddies
- [ ] Write benchmark: measure Pearson + mutual NN overhead per decode step
- [ ] GOAT gate: measure acceptance rate delta. If ≥ 5% improvement → default-on.
- [ ] Update README feature flags section

---

## Expected Performance

- **Pearson correlation**: O(V) per position, V = vocab_size. ~5μs for V=32K with SIMD.
- **Mutual NN**: O(K²) for K=5 top-K → 25 comparisons → negligible.
- **Net effect**: Smaller DDTree (fewer branches to explore) → faster decode → higher acceptance.
- **Overhead**: ~5μs per position added to speculative decode path.
- **Break-even**: Acceptance rate must improve by >2% to offset overhead (expected: 5-15%).

---

## CPU/GPU Auto-Route

- CPU path: SIMD Pearson correlation (portable, fast for single decode)
- GPU path: Batch Pearson across all positions in parallel (for batched inference)
- Router: Use existing `inference_router` — if batch_size > 8, route to GPU

---

## TL;DR

**Best Buddies Drafting** filters speculative decoding candidates to only mutual draft↔target agreements. Expect 5-15% acceptance rate improvement with ~5μs overhead per position. Feature-gated `best_buddies`, default-on after GOAT proof.
