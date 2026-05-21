# Plan 089: Tri-Mode Inference — Self-Speculation + Mode Switching

> **Research:** `.research/55_Nemotron_TriMode_Diffusion.md`
> **Paper:** Nemotron-Labs-Diffusion (NVIDIA 2026)
> **Depends on:** Plan 066 (D2F, ✅ complete), Plan 055 (MTP, ✅ complete)
> **Feature Gate:** `tri_mode` (opt-in, proofs required)

## Objective

Unify our existing AR, D2F diffusion, and speculative decoding into a **tri-mode inference pipeline** that can:
1. **AR mode**: Standard causal generation (already works)
2. **Diffusion mode**: Block-wise parallel denoising (D2F, already works)
3. **Self-speculation mode**: Diffusion drafts → AR verifies (NEW)

Plus: global loss averaging, trained sampler research, mode-adaptive switching.

## Why Now

The Nemotron paper validates our D2F architecture and proves self-speculation beats MTP (Eagle3) by 2.4-3.3×. We have all the pieces — just need orchestration.

## Feature Gate

All new code behind `tri_mode` feature flag in `microgpt-rs/Cargo.toml`:
```toml
tri_mode = ["dllm"]  # depends on dllm for D2F diffusion
```

This ensures zero impact on existing code. Self-speculation requires D2F as drafter.

---

## Tasks

### T1: Self-Speculation State Machine (microgpt-rs)
- [ ] Create `src/speculative/self_spec.rs` (feature-gated `tri_mode`)
- [ ] Define `SelfSpeculationState`:
  ```rust
  pub struct SelfSpeculationState {
      draft_width: usize,       // k tokens to draft
      verified_prefix: Vec<usize>, // committed tokens
      draft_tokens: Vec<usize>,    // k drafted tokens
      block_size: usize,           // D2F block size for drafting
  }
  ```
- [ ] Implement `self_speculate_step()`:
  1. Call `d2f_decode_block()` with mask tokens for positions [n..n+k] (DIFFUSION DRAFT)
  2. Run AR `forward()` on draft tokens with causal attention (AR VERIFY)
  3. Compare: accept longest prefix where draft[i] == AR prediction[i]
  4. Commit accepted + 1 (AR provides one extra token at first rejection)
  5. Return accepted count
- [ ] Implement `SelfSpeculation` as `SpeculativeVerifier` trait impl
- [ ] Test: self-speculation accepts ≥1 token per step (trivial case)
- [ ] Test: self-speculation terminates and produces valid token sequence
- [ ] Test: acceptance rate measurement on pattern data

### T2: Mode-Adaptive Decode Strategy (microgpt-rs)
- [ ] Extend `DecodeStrategy` enum in `speculative/types.rs`:
  ```rust
  pub enum DecodeStrategy {
      Autoregressive,
      Speculative,
      DiscreteDiffusion,
      SelfSpeculation, // NEW
  }
  ```
- [ ] Update `DecodeStrategy::recommend()` heuristic:
  ```
  if tri_mode enabled AND n_tokens >= block_size AND has_model → SelfSpeculation
  else if dllm enabled AND n_tokens >= block_size → DiscreteDiffusion
  else if has_draft_model → Speculative
  else → Autoregressive
  ```
- [ ] Feature-gate `SelfSpeculation` variant with `#[cfg(feature = "tri_mode")]`
- [ ] Test: recommend() returns correct strategy for each config

### T3: Self-Speculation Pipeline Integration (microgpt-rs)
- [ ] Wire `SelfSpeculation` into `speculative/mod.rs` behind `tri_mode` feature
- [ ] Integrate with `D2fContext` for zero-alloc buffer reuse (draft forward reuses KV)
- [ ] Integrate with `ConstraintPruner` — pruner restricts draft tokens at each denoising step
- [ ] Add `SelfSpecConfig` to `speculative/types.rs`:
  ```rust
  pub struct SelfSpecConfig {
      pub draft_width: usize,          // k (default: 8)
      pub block_size: usize,           // D2F block size (default: 16)
      pub denoising_steps: usize,      // per draft (default: 4)
      pub confidence_threshold: f32,   // D2F τ_conf (default: 0.9)
  }
  ```
- [ ] Test: end-to-end self-speculation decode on pattern data
- [ ] Test: self-speculation produces valid sequence matching AR ground truth

### T4: Global Loss Averaging (microgpt-rs dllm.rs)
- [ ] Update `masked_loss()` in `src/dllm.rs`:
  ```rust
  // BEFORE: per-sequence averaging
  // AFTER: global token averaging
  // L = (1/(N*L)) * Σ_n Σ_i ℓ_{n,i}
  ```
  where N=sequences, L=seq_length, only counting masked positions
- [ ] Add `LossAveraging` enum: `PerSequence`, `Global` (default `Global`)
- [ ] Test: global averaging produces different loss than per-sequence (when masking varies)
- [ ] Test: training with global averaging converges (re-train mini dLLM)
- [ ] Benchmark: compare convergence speed global vs per-sequence on pattern data

### T5: GOAT Proof — Self-Speculation vs MTP (microgpt-rs)
- [ ] Create `tests/test_self_speculation.rs` (feature-gated `tri_mode`)
- [ ] Proof 1: Self-speculation acceptance rate ≥ MTP acceptance rate on same model
  - Train mini model with D2F + AR capability
  - Run 1000 steps self-speculation vs 1000 steps MTP speculative
  - Measure: avg tokens accepted per step
- [ ] Proof 2: Self-speculation produces valid output
  - Decode 100 tokens with self-speculation
  - Verify: all tokens in valid range, no infinite loops, sequence terminates
- [ ] Proof 3: Mode switching works correctly
  - Start with AR, switch to SelfSpeculation, switch to Diffusion
  - Verify: seamless transition, KV cache preserved across switches
- [ ] Benchmark: throughput comparison AR vs Speculative vs SelfSpeculation vs D2F
- [ ] Record results in `.benchmarks/012_self_speculation_goat.md`

### T6: Trained Sampler Research (riir-gpu, lower priority)
- [ ] Design `DiffusionSampler` struct:
  ```rust
  pub struct DiffusionSampler {
      pca_proj: Vec<f32>,           // [144, n_embd]
      layers: [SamplerLayer; 4],    // d=384, 4-layer transformer
      head: Vec<f32>,               // [384, 1] sigmoid output
  }
  ```
- [ ] Collect denoising trajectories from D2F inference on trained model
  - Store: per-position features (144-dim) + binary label (correct/incorrect)
  - Target: ~20M trajectory steps
- [ ] Train sampler with binary cross-entropy on held-out split
- [ ] Evaluate: AUC on held-out trajectories
- [ ] Integrate into D2F denoising loop: replace confidence threshold with sampler prediction
- [ ] Benchmark: TPF improvement with trained sampler vs confidence threshold
- [ ] Feature gate: `tri_mode` depends on this being optional

### T7: LoRA Drafter Alignment (riir-gpu, research)
- [ ] Implement LK-hybrid distribution matching loss in `riir-gpu`:
  ```
  L_LK = λ · KL(p_verifier || q_drafter) + (1-λ) · TV(p, q)
  λ adaptive: exp(-η · α_j) where α_j = acceptance probability
  ```
- [ ] Implement active position masking: only accepted + first rejected
- [ ] LoRA target: o_proj only, rank=128, α=512
- [ ] Train on D2F draft → AR verify pairs
- [ ] Measure: acceptance rate improvement with vs without LoRA alignment
- [ ] Feature gate: `sdar_loss` in riir-gpu (already exists)

---

## Architecture After

```
speculative/
├── mod.rs              # pub mod self_spec (feature-gated tri_mode)
├── types.rs            # DecodeStrategy + SelfSpecConfig + SelfSpeculation variant
├── step.rs             # AR speculative step (unchanged)
├── d2f.rs              # D2F block decode (unchanged)
├── self_spec.rs        # NEW: Self-Speculation state machine
│   ├── SelfSpeculationState
│   ├── self_speculate_step()  # Diff draft → AR verify → prefix accept
│   └── impl SpeculativeVerifier for SelfSpeculation
├── verifier.rs         # SpeculativeVerifier trait (unchanged)
└── ...
```

## Dependency Graph

```
tri_mode (feature gate)
├── dllm (feature gate, already exists)
│   ├── d2f_decode_block() — diffusion drafter
│   ├── D2fContext — zero-alloc buffers
│   └── forward_block_causal_with() — block-causal forward
├── forward() — AR verifier (existing)
├── MultiLayerKVCache — shared KV across modes (existing)
└── ConstraintPruner — prunes draft tokens (existing)
```

## Estimated Effort

| Task | Lines | Effort | Depends On |
|------|-------|--------|-----------|
| T1: Self-speculation state machine | ~200 | 1-2 days | D2F (done) |
| T2: Mode-adaptive decode | ~50 | 0.5 days | T1 |
| T3: Pipeline integration | ~150 | 1-2 days | T1, T2 |
| T4: Global loss averaging | ~30 | 0.5 days | None |
| T5: GOAT proof | ~300 (tests) | 2-3 days | T1-T4 |
| T6: Trained sampler | ~500 | 5-7 days | T5 |
| T7: LoRA drafter alignment | ~400 | 5-7 days | T5, riir-gpu |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Self-speculation acceptance rate low | No speedup over AR | Fall back to AR/D2F mode; acceptance improves with model quality |
| D2F draft quality insufficient | Poor draft→verify alignment | Increase denoising steps; add LoRA alignment (T7) |
| Feature gate conflicts | Build failures | tri_mode → dllm dependency, tested in CI |
| Trained sampler overfits | No generalization | Use diverse trajectory data; AUC early stopping |
| No real model to test with | Can't validate at scale | Test with mini dLLM (Plan 066 proved this works) |

## What This Does NOT Do

- ❌ Does NOT train a full tri-mode model from scratch (1T-token pretraining)
- ❌ Does NOT implement dual-stream attention (training-only, we don't pretrain)
- ❌ Does NOT add VLM support (no vision encoder)
- ❌ Does NOT implement quadratic self-speculation (kernel complexity)
- ❌ Does NOT change existing AR or D2F code paths (feature-gated only)

## Success Criteria

1. ✅ Self-speculation mode produces valid token sequences
2. ✅ Self-speculation acceptance rate ≥ 1.0 tokens/step (trivially beating AR)
3. ✅ Mode switching works without KV cache corruption
4. ✅ Global loss averaging improves D2F training convergence
5. ✅ All new code behind `tri_mode` feature gate
6. ✅ Zero regression in existing AR/D2F/speculative benchmarks