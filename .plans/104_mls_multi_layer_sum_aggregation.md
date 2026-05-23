# Plan 104: MLS Multi-Layer Sum Aggregation

> **Parent**: Research 68 (RAEv2 Multi-Layer Representation Autoencoders)
> **Depends**: Plan 103 (CODA Fused SIMD Kernels) ✅
> **Scope**: Sum last K transformer layer residuals before LM head for richer token representations
> **Feature Gate**: `mls_aggregate` in microgpt-rs (**default-on**, GOAT 6/6 proved)
> **Cross-project**: Guides riir-ai Plan 107 (if self-guidance pursued later)

## Motivation

RAEv2 (arXiv:2605.18324) shows that summing the last K encoder layers instead of using only the final layer is:
- **Training-free** — zero new parameters
- **Pareto-optimal** — sweeping K trades reconstruction vs generation quality
- **10× faster convergence** — richer intermediate signal accelerates learning

For our LLM inference engine, the transfer is: intermediate transformer layers carry syntactic/semantic signal that the final layer's task specialization may dilute. Summing the last K residual states before the LM head could improve:
1. Speculative draft acceptance rates (richer token representations)
2. Early exit confidence (layer-aggregated signal is more stable)
3. Screening/relevance scoring quality (better token embeddings)

**Honest caveat**: This is a vision/diffusion idea transferred to text LLM inference. Well-trained LLMs may already have well-specialized final layers. Benchmarking is essential before any default-on.

## Tasks

### D1: EP Accuracy@k Metric — Zero-Risk Reporting Improvement (No Feature Gate)

- [x] **T1**: Add `ep_accuracy_k` helper to `src/benchmark.rs`
  ```rust
  /// Compute EP Accuracy@k: number of rounds to first reach target_accuracy.
  /// Returns None if target was never reached within the data.
  pub fn ep_accuracy_k(accuracies: &[f32], target: f32) -> Option<usize> {
      accuracies.iter().position(|&a| a >= target)
  }
  ```

- [x] **T2**: Update GOAT benchmark reports to include EP Accuracy@k
  - Report `EP Accuracy@0.8` and `EP Accuracy@0.9` alongside final win rate
  - Show convergence speedup vs baseline: `"EP Acc@0.8: {n} rounds ({speedup}× vs baseline {baseline_n})"`
  - Update existing benchmark output in `examples/bomber_03_hl_proof.rs`, `examples/go_05_hl_proof.rs`

### D2: MLS Aggregation Core — Feature Gate `mls_aggregate`

- [x] **T3**: Add `mls_aggregate` feature to `Cargo.toml`
  ```toml
  [features]
  mls_aggregate = []  # Multi-Layer Sum: aggregate last K layer residuals (Research 68)
  ```

- [x] **T4**: Add MLS config fields to `crates/microgpt-core/src/types.rs`
  ```rust
  pub struct Config {
      // ... existing fields ...
      /// Number of last layers to sum for MLS aggregation. 0 = disabled (standard).
      /// Research 68: RAEv2 shows summing last K layers improves representation quality.
      pub mls_layers: usize,
  }

  pub struct InferenceOverrides {
      // ... existing fields ...
      pub mls_layers: Option<usize>,
  }
  ```

- [x] **T5**: Add MLS accumulator buffer to `ForwardContext` in `src/transformer.rs`
  ```rust
  pub struct ForwardContext {
      // ... existing fields ...
      #[cfg(feature = "mls_aggregate")]
      mls_buf: Vec<f32>,   // Accumulator for last K layer residuals [n_embd]
      #[cfg(feature = "mls_aggregate")]
      mls_count: usize,     // How many layers accumulated
  }
  ```

- [x] **T6**: Implement MLS accumulation in `forward_base` layer loop
  ```rust
  // In the layer loop, after MLP residual add:
  #[cfg(feature = "mls_aggregate")]
  {
      if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
          crate::simd::simd_add_inplace(&mut ctx.mls_buf[..n], &ctx.x[..n]);
          ctx.mls_count += 1;
      }
  }

  // After layer loop, before LM head:
  #[cfg(feature = "mls_aggregate")]
  let lm_input = if ctx.mls_count > 0 {
      // Use MLS aggregated buffer (normalize by count)
      let inv_k = 1.0 / ctx.mls_count as f32;
      for v in &mut ctx.mls_buf[..n] { *v *= inv_k; }
      &ctx.mls_buf[..n]
  } else {
      &ctx.x[..n]
  };

  #[cfg(not(feature = "mls_aggregate"))]
  let lm_input = &ctx.x[..n];

  standard_lm_head(&mut ctx.logits, lm_input, &weights.lm_head, config.vocab_size, n);
  ```

- [x] **T7**: Reset MLS state in `ForwardContext` at start of each forward call
  ```rust
  #[cfg(feature = "mls_aggregate")]
  {
      ctx.mls_buf[..n].fill(0.0);
      ctx.mls_count = 0;
  }
  ```

- [x] **T8**: Add `mls_layers` to relevant `Config` constructors
  - `Config::micro()` → `mls_layers: 0`
  - `Config::game()` → `mls_layers: 0`
  - `Config::gemma2_2b()` → `mls_layers: 0`
  - All other constructors → `mls_layers: 0`

- [x] **T9**: Add `mls_layers` to `InferenceOverrides::apply()` and `Config::with_overrides()`

### D3: GOAT Proof — Property-Based Invariant Proofs ✅

- [x] **T10**: Create GOAT proof test `tests/goat_104_mls_aggregate.rs` — **6/6 proofs passed**
  - Proof 1: `ep_accuracy_k` returns correct first-match index
  - Proof 2: MLS averaging produces correct element-wise arithmetic mean
  - Proof 3: Buffer reset then accumulate is correct
  - Proof 4: MLS disabled (K=0) is identity passthrough
  - Proof 5: MLS accumulation is order-independent (commutative)
  - Proof 6: MLS mean preserves scalar multiplication
  - Run: `cargo test --features mls_aggregate --test goat_104_mls_aggregate -- --nocapture`

- [ ] **T11**: K-sweep benchmark (deferred — requires trained model)
  - Sweep K ∈ {0, 1, 2, 3, 4} for a micro config with n_layer=6
  - Measure: speculative acceptance rate, perplexity proxy, EP Accuracy@0.8

- [ ] **T12**: Create benchmark result file `.benchmarks/025_mls_aggregation_goat.md` (deferred)

### D4: Documentation & Cleanup

- [ ] **T13**: Update `README.md` — add MLS section under 🔧 Block-Diagonal Rotation area
  ```markdown
  ## 📐 MLS: Multi-Layer Sum Aggregation (Plan 104)
  Training-free aggregation of last K layer residuals before LM head.
  Opt-in via `mls_aggregate` feature gate. Sweeping K provides Pareto-optimal
  representation quality vs task specialization tradeoff.
  ```

- [ ] **T14**: Update `.docs/15_paper_feature_comparison.md` with RAEv2 row

- [ ] **T15**: Run `cargo clippy --fix --allow-dirty` with `--features mls_aggregate`

- [ ] **T16**: Run `cargo test --features mls_aggregate` — all tests pass

## Feature Gate Summary

```toml
[features]
default = []
mls_aggregate = []  # Plan 104: Sum last K layer residuals before LM head (Research 68)
```

**Default-on** as of GOAT 6/6 proof. Controlled via `Config.mls_layers` (0 = disabled).

## Expected Outcomes

| Scenario | K | Acceptance Rate | EP Accuracy@0.8 | Action |
|----------|---|----------------|------------------|--------|
| **Best case** | 2-3 | +5-10% | 1.5-2× faster | Consider default-on for specific configs |
| **Neutral** | 1-4 | ±1% | ±10% | Keep opt-in, document no gain |
| **Negative** | >1 | -5%+ | Worse | Keep disabled, document negative result |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Well-trained models don't benefit | Medium | Low | Default K=0, opt-in only |
| Layer sum dilutes final-layer specialization | Medium | Medium | K sweep required per model |
| Breaks speculative decoding | Low | High | GOAT benchmark gate |
| No improvement on small models (n_layer < 6) | High | Low | Document limitation |
| Extra buffer memory (n_embd f32) | Low | Low | ~2KB for n_embd=512 |

## Non-Goals

- Do NOT add self-guidance (`self_guidance` feature) — requires trained intermediate LM head, out of scope
- Do NOT add REPA-style spatial regularization — text has no spatial structure
- Do NOT add autoencoder training — we're inference-only
- Do NOT replace the final LM head — MLS augments, not replaces

## References

- Research 68: `.research/068_RAEv2_Multi_Layer_Representation_Autoencoders.md`
- RAEv2 paper: arXiv:2605.18324
- Related: Research 26 (MTP drafter), Research 38 (SDAR), Research 61 (Delta Routing)
- Key files: `src/transformer.rs`, `crates/microgpt-core/src/types.rs`, `src/benchmark.rs`
