# Plan 109: DMax Soft Parallel Decode — Hybrid Embedding D2F Enhancement

> **Research:** `.research/072_DMax_Aggressive_Parallel_Decoding_dLLMs.md`
> **Paper:** DMax: Aggressive Parallel Decoding for dLLMs (arXiv 2604.08302, NUS 2026)
> **Depends on:** Plan 066 (D2F ✅ complete), Plan 089 (Tri-Mode ✅ complete)
> **Feature Gate:** `dmax_spd` (opt-in, depends on `dllm`, proofs required)

## Objective

Enhance our existing D2F denoising pipeline with **Soft Parallel Decoding (SPD)** from the DMax paper. The core idea: replace binary mask/token transitions with **hybrid embeddings** that interpolate between predicted token embedding and mask embedding based on confidence. This carries uncertainty forward across denoising steps, enabling iterative self-refinement instead of one-shot commitment.

Additionally, add **contiguous prefix promotion** and **block convergence criteria** as inference heuristics that improve robustness under aggressive parallelism.

## Honest Assessment: What's Actually New

| Component | What We Already Have | Delta |
|---|---|---|
| D2F denoising loop | `d2f_decode_block()` in `d2f.rs` | ✅ Foundation exists |
| Confidence remasking | τ_conf threshold in `D2fDecodeConfig` | ✅ Binary only |
| Block-causal attention | `forward_block_causal_with()` in `dllm.rs` | ✅ No change needed |
| D2fContext zero-alloc | Flat buffers for logits, KV, embeddings | ✅ No change needed |
| ConstraintPruner integration | Pruner filters logits at each step | ✅ No change needed |
| D2F pipeline | `D2fPipeline` multi-block decode | ✅ No change needed |
| **Hybrid embedding construction** | **MISSING** | ❌ ~80 lines |
| **Contiguous prefix promotion** | **MISSING** | ❌ ~30 lines |
| **Block convergence check** | **MISSING** | ❌ ~20 lines |
| **SPD-aware D2fDecodeConfig** | **MISSING** | ❌ ~40 lines |

Total new code: ~170 lines in `d2f.rs`, all feature-gated behind `dmax_spd`.

## Feature Gate

```toml
# Cargo.toml
[features]
dmax_spd = ["dllm"]  # Soft Parallel Decoding, depends on D2F infrastructure
```

---

## Tasks

### T1: Hybrid Embedding Infrastructure — The Core Delta
- [ ] Add `SoftDecodeConfig` to `d2f.rs` (feature-gated `dmax_spd`):
  ```rust
  /// Configuration for DMax Soft Parallel Decoding.
  ///
  /// When enabled, decoded positions use hybrid embeddings (interpolation
  /// between predicted token embedding and mask embedding) instead of
  /// binary mask/token transitions. This carries uncertainty forward,
  /// enabling iterative self-refinement.
  #[cfg(feature = "dmax_spd")]
  #[derive(Clone, Debug)]
  pub struct SoftDecodeConfig {
      /// Enable hybrid embedding construction (default: true).
      pub use_hybrid_embeddings: bool,
      /// Decoding threshold τ_dec: promote positions with confidence > τ_dec (default: 0.5).
      pub decode_threshold: f32,
      /// Acceptance threshold τ_acc: block converges when all positions > τ_acc (default: 0.9).
      pub accept_threshold: f32,
      /// Enable contiguous prefix promotion rule (default: true).
      pub contiguous_prefix: bool,
      /// Enable consistency convergence check (default: true).
      pub consistency_check: bool,
  }
  ```
- [ ] Implement `SoftDecodeConfig::default()`, `::aggressive()`, `::conservative()` presets
- [ ] Implement `HybridEmbedding` helper struct:
  ```rust
  /// Hybrid embedding: soft interpolation between token and mask embeddings.
  ///
  /// h̃ = π * e_token + (1 - π) * e_mask
  /// h = h̃ / ||h̃||₂ * (π * ||e_token||₂ + (1 - π) * ||e_mask||₂)
  ///
  /// The renormalization prevents magnitude collapse from adding high-dim vectors.
  #[cfg(feature = "dmax_spd")]
  pub struct HybridEmbedding {
      /// Confidence π for the predicted token.
      pub confidence: f32,
      /// Predicted token id.
      pub token_id: usize,
  }

  impl HybridEmbedding {
      /// Construct hybrid embedding vector in-place.
      /// Writes into `out[dim]` slice, reads from `token_emb[dim]` and `mask_emb[dim]`.
      pub fn build(&self, token_emb: &[f32], mask_emb: &[f32], out: &mut [f32]) { ... }
  }
  ```
- [ ] Test: `HybridEmbedding::build()` produces valid normalized output
- [ ] Test: confidence=1.0 → output ≈ token_embedding
- [ ] Test: confidence=0.0 → output ≈ mask_embedding
- [ ] Test: confidence=0.5 → output is valid interpolation with correct norm

### T2: Contiguous Prefix Promotion
- [ ] Implement `contiguous_prefix_promote()`:
  ```rust
  /// DMax contiguous prefix promotion rule.
  ///
  /// Scan masked positions left-to-right. Promote the longest contiguous prefix
  /// where confidence > τ_dec. If none qualify, promote the leftmost position
  /// (ensure progress). This keeps the masked region contiguous.
  ///
  /// Returns: Vec of position indices to promote from mask→token.
  #[cfg(feature = "dmax_spd")]
  pub fn contiguous_prefix_promote(
      masked_positions: &[usize],
      confidences: &[f32],
      decode_threshold: f32,
  ) -> Vec<usize> { ... }
  ```
- [ ] Test: all positions above threshold → promote all
- [ ] Test: no positions above threshold → promote leftmost only
- [ ] Test: partial prefix → promote only prefix positions
- [ ] Test: gap in confidence → stop at first below-threshold position

### T3: Block Convergence Check
- [ ] Implement `BlockConvergence` enum and check function:
  ```rust
  /// Convergence status for a D2F decode block.
  #[cfg(feature = "dmax_spd")]
  #[derive(Clone, Debug, PartialEq)]
  pub enum BlockConvergence {
      /// Block has not converged, continue denoising.
      NotConverged,
      /// Block converged: all positions above acceptance threshold.
      ConfidenceConverged,
      /// Block converged: top-1 predictions unchanged for 2 consecutive steps.
      ConsistencyConverged,
  }

  /// Check if a block has converged using DMax criteria.
  ///
  /// Primary signal: consistency (unchanged top-1 across 2 steps).
  /// Secondary signal: confidence (all positions > τ_acc).
  /// Either criterion triggers convergence.
  #[cfg(feature = "dmax_spd")]
  pub fn check_block_convergence(
      current_top1: &[usize],
      prev_top1: Option<&[usize]>,
      confidences: &[f32],
      accept_threshold: f32,
  ) -> BlockConvergence { ... }
  ```
- [ ] Test: all confidences above threshold → ConfidenceConverged
- [ ] Test: top-1 unchanged from previous → ConsistencyConverged
- [ ] Test: neither condition met → NotConverged
- [ ] Test: prev_top1=None (first step) → falls through to confidence check

### T4: SPD-Enhanced D2F Denoising Loop
- [ ] Add `d2f_decode_block_soft()` to `d2f.rs` (feature-gated `dmax_spd`):
  ```rust
  /// DMax Soft Parallel Decoding — enhanced D2F block decode.
  ///
  /// Key differences from standard `d2f_decode_block()`:
  /// 1. Hybrid embeddings replace binary mask/token transitions
  /// 2. Contiguous prefix promotion for position selection
  /// 3. Block convergence check for early stopping
  ///
  /// **Important:** Best results with OPUT-trained models. May degrade quality
  /// on models trained with standard D2F loss only. See Research 072 Doubt 2.
  #[cfg(feature = "dmax_spd")]
  pub fn d2f_decode_block_soft(
      ctx: &mut D2fContext,
      weights: &TransformerWeights,
      config: &Config,
      pruner: &dyn ConstraintPruner,
      prompt: &[usize],
      block_start: usize,
      block_len: usize,
      soft_config: &SoftDecodeConfig,
      rng: &mut Rng,
  ) -> Vec<usize> { ... }
  ```
- [ ] Integration points:
  1. Get mask embedding from `weights.embedding_table[config.mask_token]`
  2. At each denoising step:
     - Forward pass via `forward_block_causal_with()` → logits
     - Apply `ConstraintPruner` to logits (existing logic)
     - Sample top-1 and confidence for each position
     - Compute hybrid embeddings via `HybridEmbedding::build()`
     - Apply contiguous prefix promotion for position selection
     - Check block convergence → early stop if converged
  3. Write hybrid embeddings into `ctx.x_norm` for next forward pass input
- [ ] Ensure `D2fContext` has space for previous-step top-1 tracking:
  - Add `prev_top1: Vec<usize>` field behind `#[cfg(feature = "dmax_spd")]`
  - Initialize in `D2fContext::new()` with zeros
- [ ] Test: `d2f_decode_block_soft()` produces valid token sequence
- [ ] Test: soft decode terminates (no infinite loop)
- [ ] Test: convergence check triggers early stop on easy inputs
- [ ] Test: hybrid embeddings flow correctly through forward pass

### T5: Integrate with D2fPipeline
- [ ] Add `SoftDecodeConfig` field to `D2fPipeline` behind `#[cfg(feature = "dmax_spd")]`:
  ```rust
  pub struct D2fPipeline<'a> {
      // ... existing fields ...
      #[cfg(feature = "dmax_spd")]
      soft_config: Option<SoftDecodeConfig>,
  }
  ```
- [ ] In `D2fPipeline::decode_all()`:
  - If `soft_config` is `Some`, call `d2f_decode_block_soft()` instead of `d2f_decode_block()`
  - Otherwise, use existing binary decode path
- [ ] Test: pipeline with SPD config uses soft decode
- [ ] Test: pipeline without SPD config uses existing binary decode (no regression)
- [ ] Test: multi-block SPD decode produces coherent output across blocks

### T6: DecodeStrategy Integration
- [ ] Extend `DecodeStrategy` recommendation heuristic:
  ```rust
  // In DecodeStrategy::recommend():
  // If dmax_spd feature enabled and n_tokens >= block_size:
  //   → DiscreteDiffusionSoft (new variant) instead of DiscreteDiffusion
  ```
- [ ] Add `DiscreteDiffusionSoft` variant to `DecodeStrategy` behind `#[cfg(feature = "dmax_spd")]`
- [ ] Test: recommend() selects soft decode when appropriate
- [ ] Test: config-driven switch between binary/soft D2F

### T7: GOAT Proof — SPD vs Binary D2F
- [ ] Create `tests/test_dmax_spd.rs` (feature-gated `dmax_spd`)
- [ ] **Proof 1: SPD maintains quality under aggressive parallelism**
  - Train mini dLLM with standard D2F loss (Plan 066 config)
  - Decode 100 blocks with binary D2F at τ=0.5 → measure accuracy
  - Decode 100 blocks with SPD at τ=0.5 → measure accuracy
  - Hypothesis: SPD ≥ binary accuracy (may be lower without OPUT, but should not collapse)
  - Record result honestly even if negative
- [ ] **Proof 2: Hybrid embeddings carry meaningful uncertainty**
  - During SPD decode, log confidence values at each step
  - Verify: confidence generally increases across steps (self-refinement)
  - Verify: low-confidence positions get revised more often
- [ ] **Proof 3: Convergence check saves forward passes**
  - Count total forward passes with fixed step count (e.g., T=8)
  - Count total forward passes with convergence check (max T=8, early stop)
  - Measure: forward passes saved + quality delta
- [ ] **Proof 4: Contiguous prefix improves quality at τ=0**
  - Decode with all-confident-promotion at τ=0 → measure accuracy
  - Decode with contiguous-prefix-promotion at τ=0 → measure accuracy
  - Hypothesis: contiguous prefix ≥ all-confident at τ=0
- [ ] Record all results in `.benchmarks/020_dmax_spd_goat.md`

### T8: OPUT Training Research (riir-gpu, deferred)
- [ ] Design `GpuOputTrainer` — on-policy rollout + L_pred loss
- [ ] Requires: forward pass without grad, sample predictions, second forward with grad
- [ ] Depends on riir-gpu D2F training pipeline (already exists from Plan 066)
- [ ] ~150 lines in `riir-gpu/src/dllm.rs` behind `dmax_oput` feature
- [ ] **Deferred until T1-T7 prove SPD has value at our scale**
- [ ] Key risk from Research 072 Doubt 2: SPD without OPUT may not work well

---

## Architecture After

```
speculative/
├── mod.rs                # pub mod d2f (unchanged)
├── types.rs              # DecodeStrategy + DiscreteDiffusionSoft variant (feature-gated)
├── step.rs               # AR speculative step (unchanged)
├── d2f.rs                # D2F block decode
│   ├── d2f_decode_block()          # Binary mask/token (unchanged)
│   ├── d2f_decode_block_soft()     # NEW: DMax SPD hybrid embeddings
│   ├── HybridEmbedding             # NEW: soft interpolation helper
│   ├── contiguous_prefix_promote() # NEW: left-to-right prefix scan
│   ├── check_block_convergence()   # NEW: consistency + confidence early stop
│   ├── SoftDecodeConfig            # NEW: SPD parameters
│   └── D2fPipeline                 # Extended with soft_config option
├── d2f_verifier.rs       # D2F drafter verifier (unchanged, Plan 089)
├── verifier.rs           # SpeculativeVerifier trait (unchanged)
└── ...

dllm.rs
├── D2fContext             # Extended with prev_top1 field (feature-gated)
├── forward_block_causal_with()  # Unchanged
└── ...existing infrastructure unchanged...
```

## Key Differences: SPD vs Binary D2F

| Aspect | Binary D2F (Plan 066) | DMax SPD (Plan 109) |
|---|---|---|
| Position state | Binary: mask OR token | Soft: hybrid embedding |
| Error recovery | None — committed token stays | Self-refinement — uncertainty propagates |
| Position promotion | All above τ_conf | Contiguous left-to-right prefix |
| Convergence | Fixed step count T | Consistency + confidence early stop |
| Requires OPUT training | No | Yes (for best results) |
| Risk | Low — proven | Medium — unproven at micro scale |

## Estimated Effort

| Task | Lines | Effort | Depends On |
|------|-------|--------|-----------|
| T1: Hybrid embedding | ~80 | 1 day | None |
| T2: Contiguous prefix | ~30 | 0.5 day | None |
| T3: Block convergence | ~20 | 0.5 day | None |
| T4: SPD denoising loop | ~80 | 1.5 days | T1, T2, T3 |
| T5: Pipeline integration | ~30 | 0.5 day | T4 |
| T6: DecodeStrategy | ~20 | 0.5 day | T5 |
| T7: GOAT proof | ~200 (tests) | 2 days | T1-T6 |
| T8: OPUT training | ~150 (deferred) | — | T7, riir-gpu |

**Total: ~6-7 days for T1-T7, T8 deferred**

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| SPD without OPUT causes quality drop | Negative GOAT result | Fallback to binary D2F, document finding |
| Hybrid embedding norm collapse | Forward pass instability | Renormalization (Eq. 10), test early |
| Contiguous prefix too restrictive at block_size=8 | Reduced parallelism | Make configurable, A/B test |
| D2fContext breaking changes | Regression in existing D2F | Feature-gate all new fields |
| prev_top1 tracking overhead | Memory/perf cost | Small Vec, only allocated when `dmax_spd` enabled |

## What This Does NOT Do

- ❌ Does NOT change existing `d2f_decode_block()` (binary path untouched)
- ❌ Does NOT require OPUT training (SPD works without it, just less effective)
- ❌ Does NOT change AR inference or speculative decoding
- ❌ Does NOT change modelless distillation strategies
- ❌ Does NOT add new GPU kernels (all CPU, micro scale)
- ❌ Does NOT replace D2F — it **enhances** it with soft embeddings

## Success Criteria

1. ✅ `HybridEmbedding` produces valid interpolated embeddings
2. ✅ `d2f_decode_block_soft()` produces valid token sequences
3. ✅ Convergence check reduces unnecessary forward passes
4. ✅ All new code behind `dmax_spd` feature gate
5. ✅ Zero regression in existing D2F/binary/AR benchmarks
6. ✅ GOAT proof results recorded honestly (even if negative)
7. ✅ Documentation updated: README, .docs, .research

## Conventional Commit Messages (when ready)

- `feat(d2f): add soft parallel decode (DMax SPD) with hybrid embeddings`
- `feat(d2f): add contiguous prefix promotion rule`
- `feat(d2f): add block convergence early stop`
- `test(d2f): add DMax SPD GOAT proofs`
- `docs: update README and .docs with DMax SPD`
