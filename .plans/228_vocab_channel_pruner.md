# VocabChannel Pruner: ROTATE-Derived ConstraintPruner

**Plan:** 228
**Research:** 203_ROTATE_Vocabulary_Channel_Inference.md
**Feature Gate:** `vocab_channel_pruner`
**Status:** Plan
**GOAT Status:** GOAT — load-time weight decomposition → per-neuron token reachability → ConstraintPruner lookup

---

## Summary

At model load time, decompose MLP output weights into vocabulary channels using kurtosis-maximizing Householder reflections (ROTATE method). Build per-neuron token reachability maps. At inference time, use as a `ConstraintPruner` to reject unreachable tokens in DDTree speculative decoding.

**Expected gain:** 30-60% DDTree branch reduction, quality-neutral.

---

## Architecture

```
Load Time:
  Wout[l][i] → ROTATE → channels {v₁...v₅₀}
  Each channel → top-50 tokens → per-neuron reachability set
  Aggregate → per-layer reachability: {token_idx → neuron_count}

Inference Time:
  hidden state x → top-k active neurons → union of reachability sets
  → ConstraintPruner::is_valid(depth, token_idx, ...) = reachability.contains(token_idx)
```

---

## Tasks

### Phase 1: Core Infrastructure

- [ ] Implement `skewness()` function alongside existing `excess_kurtosis()` in `kurtosis_gate.rs`
- [ ] Implement Householder reflection: `R = I - 2*h*h^T / ||h||^2` as a pure function on `&[f32]`
- [ ] Implement vocabulary projection: `w @ U` for a single neuron weight vector (reuse lm_head from transformer.rs)
- [ ] Implement iterative token masking: given channel logits z, mask tokens where |z_i - μ| > k*σ

### Phase 2: ROTATE Decomposition Pipeline

- [ ] Implement `VocabChannelDecomposer` struct with configurable kurtosis threshold, regularization λ, learning rate η, max iterations
- [ ] Implement per-neuron channel discovery: optimize Householder h to maximize kurtosis(z) - λ*(1 - cos(v, w))
- [ ] Implement iterative multi-channel extraction with token masking between iterations
- [ ] Add `VocabChannel { direction: Vec<f32>, top_tokens: Vec<usize>, kurtosis: f32, skewness: f32 }` struct

### Phase 3: Reachability Map Builder

- [ ] Implement `VocabChannelMap` struct: per-layer, per-neuron token reachability
- [ ] Build reachability from channels: for each neuron, union of top-50 tokens from each channel
- [ ] Implement compact storage: `Vec<FixedSet<usize>>` per layer (fixed-size token sets, no HashMap)
- [ ] Add serialization/deserialization for the map (avoid recomputing on every load)

### Phase 4: ConstraintPruner Integration

- [ ] Implement `VocabChannelPruner` struct implementing `ConstraintPruner` trait
- [ ] `is_valid()`: look up active neurons from current hidden state, check token reachability
- [ ] `batch_is_valid()`: batch lookup for multiple tokens at same depth
- [ ] Integrate with DDTree: `build_dd_tree_pruned()` with VocabChannelPruner as additional constraint
- [ ] Feature gate behind `vocab_channel_pruner`

### Phase 5: Load-Time Pipeline Integration

- [ ] Add ROTATE decomposition to model loading path (after weights are loaded, before inference starts)
- [ ] Add `--vocab-channels` CLI flag to enable/disable
- [ ] Add timing metrics for load-time decomposition (should be < 30s for 8B model)
- [ ] Add cache: save decomposed channels to disk, skip recomputation if weights unchanged (BLAKE3 hash of weight bytes)

### Phase 6: Benchmarks & Tests

- [ ] Benchmark: load-time decomposition speed per layer (target: < 30s total for 8B)
- [ ] Benchmark: DDTree branch reduction with vs without VocabChannelPruner (target: 30-60%)
- [ ] Benchmark: inference throughput with vs without (target: no regression, ideally improvement)
- [ ] Test: round-trip — ROTATE channels reconstruct original weight with cos_sim > 0.95
- [ ] Test: reachability correctness — tokens in reachability set are actually promoted by the neuron
- [ ] Test: feature gate isolation — no binary bloat when feature is disabled
- [ ] Example: `vocab_channel_pruner_demo.rs` showing before/after DDTree stats

### Phase 7: GOAT Gate

- [ ] Add `vocab_channel_pruner_goat` feature flag for initial validation
- [ ] Run full benchmark suite with goat flag enabled
- [ ] Verify no quality regression on existing tests
- [ ] If GOAT: promote `vocab_channel_pruner_goat` → `vocab_channel_pruner` (remove goat suffix)
- [ ] If regression: demote to experimental, document why

---

## SOLID/DRY Compliance

- **S:** VocabChannelPruner implements ConstraintPruner trait — single responsibility (token reachability pruning)
- **O:** Open for extension — new channel discovery methods can plug in without changing the pruner
- **L:** VocabChannelPruner is substitutable for any ConstraintPruner
- **I:** Uses existing ConstraintPruner interface — no new trait needed
- **D:** Depends on abstraction (ConstraintPruner), not concrete DDTree implementation
- **DRY:** Reuses `excess_kurtosis()` from kurtosis_gate, reuses lm_head projection from transformer.rs

## Performance Constraints

- Load-time decomposition: < 30s for 8B model (parallelize across neurons with rayon)
- Per-inference pruner lookup: O(1) — just set membership check
- Storage: ~50 channels × 50 tokens × 14K neurons × 32 layers ≈ 1.1B entries ≈ 4.4GB — too large
  - Optimization: Only decompose top-10% most polysemantic neurons (low kurtosis = need disentanglement)
  - Optimization: Use roaring bitmap for token sets (compression)
  - Target: < 500MB total storage

## CPU/GPU Auto-Route

- Load-time decomposition: CPU (Householder optimization is small-scale, no GPU benefit)
- Inference-time pruning: CPU (lookup table, zero compute)
- No GPU kernel needed for this feature
