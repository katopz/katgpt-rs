# Plan 217: NextLat Belief-State Speculative Drafter

**Research**: R192 (NextLat Belief-State Latent Dynamics)
**Status**: Phase 0 COMPLETE (feature gate wired, 7 tests passing). Phase 1+ pending.
**Feature Gate**: `belief_drafter` (off by default until GOAT proof)
**Depends on**: Plan 055 (MTP Drafter infrastructure), Plan 195 (ThoughtFold), Plan 212 (Collapse-Aware Adaptive Thinking)

---

## Goal

Replace the separate draft model with a lightweight MLP that predicts next hidden states from `(h_t, x_{t+1})`, enabling variable-length self-speculative decoding at near-zero overhead. Distilled from arXiv:2511.05963 (NextLat).

## Why

- Current draft model requires full forward pass + separate KV cache → expensive
- NextLat proves a 3-layer MLP can draft variable-length sequences with 3.3× speedup
- MLP is ~2% of target model params, needs zero KV cache
- Belief-state quality improves ConstraintPruner effectiveness (better hidden states = better pruning)
- Compatible with existing DDTree + ConstraintPruner + ScreeningPruner pipeline

## Architecture

```
BeliefDrafter {
    mlp: LatentDynamicsMLP,  // 3-layer residual MLP, loaded from nextlat.bin or random init
    output_head: Linear,     // shared with target model's output projection
}

// Recursive drafting loop:
fn draft(drafter: &BeliefDrafter, h_t: &[f32], max_steps: usize, entropy_threshold: f32) -> Vec<(usize, f32)> {
    let mut drafts = Vec::new();
    let mut h = h_t;
    for _ in 0..max_steps {
        let logits = output_head.forward(h);
        let (token, entropy) = sample_with_entropy(logits);
        if entropy > entropy_threshold { break; }
        drafts.push((token, entropy));
        h = mlp.forward(h, embedding(token)); // residual: h + delta
    }
    drafts
}
```

### Integration with DDTree

1. Target model produces `h_t` at position t
2. BeliefDrafter drafts K tokens from `h_t` (variable length, entropy-gated)
3. DDTree branches from draft candidates
4. ConstraintPruner validates each branch (existing pipeline)
5. Target model verifies accepted branches (existing pipeline)

### LatentDynamicsMLP

```rust
// Input: LayerNorm(concat(h_t, emb(x_{t+1})))  -- shape [2 * n_embd]
// FC1: [2 * n_embd] → [n_embd], GELU
// FC2: [n_embd] → [n_embd], GELU
// FC3: [n_embd] → [n_embd]
// Output: h_{t+1} = h_t + FC3(GELU(FC2(GELU(FC1(LN(concat))))))
struct LatentDynamicsMLP {
    norm_weight: Vec<f32>,   // LayerNorm params
    norm_bias: Vec<f32>,
    fc1_weight: Vec<f32>,    // [n_embd, 2*n_embd]
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<f32>,    // [n_embd, n_embd]
    fc2_bias: Vec<f32>,
    fc3_weight: Vec<f32>,    // [n_embd, n_embd]
    fc3_bias: Vec<f32>,
}
```

For Config::micro (embd=16): MLP has ~1.5K params. For Config::bpe (embd=32): ~6K params.

---

## Tasks

### Phase 0: Types & MLP Forward
- [x] Add `belief_drafter` feature gate to `crates/katgpt-core/src/lib.rs` and `src/lib.rs`
- [x] Add `LatentDynamicsMLP` struct to `src/speculative/belief_drafter.rs`
- [x] Implement `forward(&self, h_t: &[f32], next_emb: &[f32]) -> Vec<f32>` with SIMD matmul
- [x] Implement `load_from_bin(path: &Path) -> Result<LatentDynamicsMLP>` for riir-ai export
- [x] Implement `random_init(config: &Config) -> LatentDynamicsMLP` for untrained mode
- [x] Unit test: MLP forward shape correctness for all config presets

### Phase 1: Belief Drafter Integration
- [ ] Add `BeliefDrafter` struct wrapping MLP + output head reference
- [ ] Implement `draft()` method with entropy-gated variable-length stopping
- [ ] Integrate into `SpeculativeGenerator` trait as new drafter variant
- [ ] Add `DecodeStage::BeliefDraft` variant to forward pipeline
- [ ] Add `belief_drafter_path` to `Config` / `InferenceOverrides`
- [ ] Integration test: belief drafter produces valid token sequences

### Phase 2: DDTree Fusion
- [ ] Wire BeliefDrafter output into DDTree branch initialization
- [ ] Add entropy-based draft length control (Plan 212 collapse-aware gate integration)
- [ ] Benchmark: belief drafter vs separate draft model vs MTP drafter
- [ ] Benchmark: variable-length vs fixed-length draft at micro scale
- [ ] Benchmark: MLP forward overhead measurement

### Phase 3: Belief-State Pruner
- [ ] Implement effective-rank computation on hidden states (SVD on recent buffer)
- [ ] Add `BeliefRankPruner` implementing `ScreeningPruner` trait
- [ ] Low rank → high confidence → accept draft; high rank → reject → deeper search
- [ ] Benchmark: pruning quality with/without belief-state signal

### Phase 4: Latent Transition Cache
- [ ] Implement `(h_t, x_{t+1})` → `ĥ_{t+1}` LRU cache using papaya HashMap
- [ ] Measure cache hit rate on game domain sequences
- [ ] Benchmark: cached vs uncached MLP forward

### Phase 5: GOAT Proof & Default-On
- [ ] GOAT proof test: belief drafter acceptance rate ≥ MTP drafter
- [ ] GOAT proof test: variable-length ≥ fixed-length speedup
- [ ] GOAT proof test: no perf regression on non-speculative path
- [ ] If all pass: flip `belief_drafter` to default-on
- [ ] Update README, docs, feature flag table

---

## Expected Performance

| Metric | Current (MTP Drafter) | Expected (Belief Drafter) | Reason |
|---|---|---|---|
| Draft forward cost | ~50K FLOPs (full forward) | ~1K FLOPs (MLP only) | No attention, no KV, single matmul |
| Draft KV cache | Full cache per branch | None | MLP is stateless |
| Draft length | Fixed (training horizon) | Variable (entropy-gated) | NextLat recursive composition |
| Acceptance rate | ~65-85% (4 tokens) | ~60-80% estimated | MLP is less accurate than full forward, but variable length compensates |
| Net speedup | ~1.5-2.0× | ~2.0-3.0× | Cheaper drafts + variable length |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| MLP quality too low without NextLat training | Random-init MLP still provides diversity; bandit adapts weights online |
| Entropy threshold tuning fragile | Use existing collapse-aware gate from Plan 212 as starting point |
| Effective rank computation too expensive | Compute on downsampled hidden state (every 8th position) |
| Cache thrashing for diverse inputs | Fixed-size LRU, papaya lock-free for concurrent access |

---

## Commercial Alignment

- **Engine (MIT):** BeliefDrafter MLP + DDTree integration + belief-state pruner
- **Fuel (SaaS):** Pre-trained `nextlat.bin` from riir-ai NextLat training
- **Flywheel:** Better belief states → better translations → better validators → better marketplace

TL;DR: Replace draft model with tiny MLP that recursively predicts next hidden states for variable-length speculative decoding. Phased: types → integration → DDTree fusion → pruner → cache → GOAT proof → default-on.
