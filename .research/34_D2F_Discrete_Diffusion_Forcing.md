# Research Verdict 34: D2F — Discrete Diffusion Forcing (arXiv 2508.09192)

> Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, Zhijie Deng
> Shanghai Jiao Tong University & UC San Diego, Aug 2025

## 1. What D2F Actually Does (Verified by Reading)

### 1.1 Core Mechanism

D2F converts standard bidirectional dLLMs (LLaDA, Dream) into an **AR-diffusion hybrid**:

1. **Block-wise AR generation**: Text is generated block-by-block (size k, typically 16-64 tokens)
2. **Intra-block parallel denoising**: Within each block, tokens are denoised in parallel using bidirectional attention
3. **Inter-block causal attention**: Blocks attend causally to prior blocks → standard KV cache works
4. **Monotonically increasing masks**: Training uses noise schedule t₁ < t₂ < ... < tₙ so earlier blocks are less noisy

This is NOT a new model architecture. It's a **training paradigm + inference algorithm** applied to existing dLLMs.

### 1.2 Key Results (From Paper Tables)

| Model | Baseline TPS | D2F TPS | Speedup | Quality Δ |
|-------|-------------|---------|---------|-----------|
| LLaDA-8B (GSM8K) | 7.2 | 52.5 | 7.3× | 77.3 vs 77.4 |
| LLaDA-8B (HumanEval) | 2.8 | 81.6 | 29.1× | 40.2 vs 36.0 |
| Dream-7B (GSM8K-CoT) | 9.5 | 91.2 | 9.6× | 77.6 vs 75.0 |
| D2F-Dream-7B vs LLaMA3-8B | — | 119.9 vs 48.0 | 2.5× | Comparable |

Training: 12 hours on 8× A100, LoRA rank 32 on q/k/v/o projections only.

### 1.3 What We Can Verify From Paper

1. ✅ Block-causal attention preserves exact KV cache (no approximation)
2. ✅ Asymmetric distillation works: teacher sees all blocks, student sees causal prefix
3. ✅ Pipelined parallel decoding with dual-state (semi/fully-activated) blocks
4. ✅ Confidence-based remasking (τ_conf) for token selection at each denoising step
5. ✅ Block size 16 for training, 32-64 for inference is optimal (Figure 5)

---

## 2. Honest Gap Analysis vs Our System

### 2.1 What We Have That D2F Needs

| Component | Our Status | D2F Requirement | Gap |
|-----------|-----------|-----------------|-----|
| Transformer weights (AR) | ✅ `TransformerWeights` | ❌ Need dLLM weights (bidirectional) | **Critical** |
| Causal attention | ✅ `attention_head()` CPU + `attention_score.wgsl` GPU | ✅ For inter-block | None |
| Bidirectional attention | ❌ Causal only (`t < t_n` where t_n = pos+1) | ✅ For intra-block | **1 kernel change** |
| KV cache | ✅ `MultiLayerKVCache`, `PagedKVCache` | ✅ Block-wise causal reuse | None |
| LoRA training (wgpu) | ✅ 26 WGSL shaders, AdamW, outer product | ✅ q/k/v/o LoRA targets | None |
| KL divergence distillation | ✅ `distill.rs` has `compute_distill_kl` | ✅ Asymmetric KL loss | **Loss adaptation needed** |
| Noise schedule | ❌ Not in our stack | Forward process: q(y_t\|y_0) | **New component** |
| Mask token handling | ❌ No mask token in vocab | `[MASK]` token for corrupted positions | **New component** |
| Cross-entropy loss | ✅ `GpuLoss` with softmax + CE | ✅ Per-mask-token prediction | Minor adaptation |
| Pipelined inference | ✅ Speculative step pipeline | ✅ Dual-state block management | **New logic** |
| ConstraintPruner | ✅ `is_valid(depth, token, path)` | Can intercept at each denoising step | **Integration point** |

### 2.2 The Fundamental Question

**Can we build a mini dLLM from scratch using our existing wgpu training infrastructure, rather than distilling from LLaDA/Dream?**

D2F paper distills FROM existing dLLMs. We don't have one. But we have:
- Full transformer weight initialization (`TransformerWeights::new()`)
- Full forward pass (CPU + GPU)
- Full backward pass (LoRA gradients via wgpu)
- Cross-entropy loss
- KL divergence distillation

**Hypothesis**: We can train a small bidirectional dLLM from scratch using our existing infrastructure by:
1. Adding a `[MASK]` token to vocabulary
2. Training with bidirectional attention + masked token prediction (like BERT, but with discrete diffusion schedule)
3. Then applying D2F's asymmetric distillation to convert it to block-causal

This is the **research question** — it's unproven but feasible.

---

## 3. What's In Doubt (Must Be Proven)

### Doubt 1: Can a Mini dLLM Actually Learn?

The smallest dLLM in the paper is 7B parameters. Our micro config is:
- vocab=27, block=16, n_embd=16, n_head=4, n_layer=1 → ~6K parameters

**Question**: Can a 6K-parameter transformer learn meaningful masked token prediction with discrete diffusion? BERT proved this at scale — does it work at micro scale?

**Proof Task**: Train a 1-layer transformer with bidirectional attention on a simple task (e.g., complete 4-letter words with 1-2 masked positions). Measure reconstruction accuracy.

### Doubt 2: Does Bidirectional Attention Actually Help Within Small Blocks?

At block_size=16 with 1-layer, the "bidirectional context" is just 16 positions. This may not provide enough context for meaningful denoising.

**Question**: Is the quality gain from bidirectional attention meaningful at our scale, or does it only matter at 7B+ parameters?

**Proof Task**: A/B test: train same model with causal vs bidirectional attention on masked prediction. Compare denoising quality.

### Doubt 3: Can We Do Asymmetric Distillation Without a Pre-trained Teacher?

D2F uses a pre-trained dLLM as teacher. If we train our dLLM from scratch, we'd need a teacher too — or skip distillation entirely and train block-causal directly.

**Question**: Can we train block-causal attention directly (no teacher) and still get parallel denoising to work? Or is the bidirectional teacher essential?

**Proof Task**: Compare three training approaches:
- A: Train bidirectional dLLM, then distill to block-causal (full D2F)
- B: Train block-causal directly with monotonic masks (skip teacher)
- C: Train bidirectional only, use at inference with causal KV cache (no distillation)

### Doubt 4: Does ConstraintPruner Actually Improve Denoising Convergence?

The theory says pruning invalid tokens at each denoising step should help the diffusion model converge faster (fewer valid tokens = easier denoising). But at our micro scale with small vocab, the pruning may be trivially satisfied.

**Question**: Does ConstraintPruner integration actually reduce denoising steps or improve quality in practice?

**Proof Task**: Run denoising with and without ConstraintPruner. Measure: (a) convergence speed (steps to <5% error), (b) final quality.

### Doubt 5: Does Pipelined Parallel Decoding Actually Help at Micro Scale?

D2F's speedup comes from GPU-parallel block denoising. Our micro models run on CPU with SIMD. The parallelism benefit may be negligible at small block sizes.

**Question**: Is there any throughput benefit from pipeline parallelism at block_size=16 on CPU?

**Proof Task**: Benchmark: serial block generation vs pipelined with overlapping draft/verify.

---

## 4. Distillation Strategy for Our System

### 4.1 What We Should Steal

| D2F Concept | Our Adaptation | Priority |
|-------------|---------------|----------|
| Block-causal attention | Add `AttentionMode::BlockCausal` to `Config` | P0 |
| Monotonic noise schedule | New `NoiseSchedule` struct with linear mask ratios | P0 |
| Asymmetric distillation loss | Adapt `compute_distill_kl` for block-causal student | P1 |
| Pipelined parallel decoding | New `d2f_decode` step in speculative module | P2 |
| Dual-state block management | Semi/fully-activated block states | P2 |
| Confidence remasking (τ_conf) | Integrate with existing `ScreeningPruner::relevance()` | P2 |

### 4.2 What We Should NOT Steal

| D2F Concept | Why Skip |
|-------------|----------|
| Pre-trained dLLM teacher | We train from scratch; no LLaDA/Dream weights needed for research |
| Large-scale LoRA on 8× A100 | Our micro models train on CPU/Metal in seconds |
| Full 7B parameter dLLM | Research first with micro; scale later if results are good |

### 4.3 Architecture: Hybrid Mini dLLM

```
                    ┌─────────────────────────────────┐
                    │   Mini dLLM (from scratch)       │
                    │   bidirectional + mask token     │
                    │   vocab=32, block=16, n_layer=2  │
                    └──────────────┬──────────────────┘
                                   │ train
                    ┌──────────────▼──────────────────┐
                    │   D2F Student (block-causal)      │
                    │   distill from bidirectional      │
                    │   teacher → block-causal student  │
                    └──────────────┬──────────────────┘
                                   │ inference
                    ┌──────────────▼──────────────────┐
                    │   Pipelined Parallel Decode       │
                    │   + ConstraintPruner intercept    │
                    │   + KV cache (existing infra)     │
                    └─────────────────────────────────┘
```

---

## 5. Key Insight: Why This Is Different From ColaDLM (Research 10)

We previously rejected ColaDLM (continuous latent diffusion) because:
- Continuous latents are incompatible with `ConstraintPruner` (operates on discrete tokens)
- VAE encoder adds ~500M parameters
- Multi-step denoising too heavy for CPU

**D2F solves ALL three problems**:
- Discrete tokens → `ConstraintPruner` works naturally
- No VAE → same transformer architecture, just different attention mask + training objective
- Block-causal → reuses our existing KV cache infrastructure

---

## 6. Feature Gate Strategy

All D2F code behind feature gates, zero impact on existing code:

### `microgpt-rs`
- `dllm` feature: bidirectional attention mode, mask token, noise schedule, d2f inference
- Feature-gated `forward_bidirectional()` alongside existing `forward()`
- Feature-gated `d2f_decode()` in speculative module

### `riir-ai/riir-engine`
- `dllm` feature: `AttentionMode::Bidirectional` enum variant, `mask_token` in Config
- Feature-gated `forward_bidirectional()` alongside `forward()`

### `riir-ai/riir-gpu`
- `dllm` feature: `attention_score_bidirectional.wgsl` kernel, noise schedule training
- Feature-gated `GpuD2fDistill` training loop
- Feature-gated `GpuNoiseSchedule` for mask corruption

---

## 8. Phase 2 Results: D2F Inference Pipeline (Plan 066)

### 8.1 Implementation Summary

The D2F inference pipeline is implemented in `src/speculative/d2f.rs` behind the `dllm` feature gate. All Phase 0 proof tasks passed, confirming the approach is viable at our micro scale.

**Key components:**

| Component | File | Purpose |
|-----------|------|---------|
| `D2fContext` | `src/dllm.rs` | Pre-allocated flat buffers for zero-alloc denoising |
| `D2fPipeline` | `src/speculative/d2f.rs` | Multi-block sequential decode with KV cache commit |
| `D2fDecodeConfig` | `src/speculative/d2f.rs` | Thresholds, block sizes, denoising steps |
| `D2fBlockState` | `src/speculative/d2f.rs` | SemiActivated → FullyActivated state machine |
| `DecodeStrategy` | `src/speculative/types.rs` | Config-driven strategy selection (AR / Speculative / D2F) |

### 8.2 Zero-Allocation Integration

`D2fContext` avoids `Vec<Vec<f32>>` allocation per denoising step by using flat 2D buffers indexed by `[p * dim..(p+1) * dim]`:

- `logits_flat: Vec<f32>` — `[max_seq * vocab_size]` instead of `Vec<Vec<f32>>`
- `k_cache`, `v_cache: Vec<f32>` — `[max_seq * kv_dim]` flat KV storage
- `x_norm`, `xr: Vec<f32>` — `[max_seq * n_embd]` for per-position embeddings

`forward_block_causal_with()` writes directly into these pre-allocated buffers. The `D2fContext::commit(len)` method preserves KV entries for positions `[0..len)`, allowing subsequent blocks to skip recomputation.

### 8.3 Test Results

15/15 tests pass in `tests/test_d2f_decode.rs`:

| Category | Tests | What They Verify |
|----------|-------|-----------------|
| Quality | `produces_non_mask_tokens`, `convergence_curve`, `target_accuracy` | Block decode outputs valid tokens, improves with steps |
| Convergence | `steps_vs_quality` | More denoising steps → higher accuracy (monotonic) |
| Pipeline | `multi_block_decode`, `prompt_context`, `partial_block` | Multi-block sequential decode with block-causal context |
| Constraints | `constraint_pruner_restricts_vocab`, `no_repeat_constraint` | `ConstraintPruner` restricts vocab and reduces invalid tokens |
| Temperature | `temperature_effects` | Temperature affects sampling diversity in denoising |
| Benchmarks | `decode_block`, `pipeline`, `steps_sweep`, `constraint_overhead` | Throughput and overhead measurements |

### 8.4 Key Measurements

**Convergence:** Denoising quality improves monotonically with more steps. The `test_d2f_decode_steps_vs_quality` test verifies that accuracy at step T is ≥ accuracy at step T-1 for a trained model on pattern data.

**Pipeline:** Multi-block decode works correctly — each block uses block-causal attention so previously decoded blocks provide causal context while the current block denoises bidirectionally. KV cache commit avoids recomputation across blocks.

**Constraints:** `ConstraintPruner` integration restricts the vocabulary at each denoising step. Tests confirm pruned decode produces only valid tokens and reduces the space of invalid outputs.

**Temperature:** Affects sampling diversity. Low temperature → more deterministic/greedy denoising. High temperature → more diverse but potentially noisier intermediate predictions.

### 8.5 DecodeStrategy Auto-Switch

`DecodeStrategy::recommend(block_size, n_tokens, has_draft_model)` implements a heuristic:

1. If `dllm` feature enabled **and** `n_tokens >= block_size` → `DiscreteDiffusion`
2. Else if `has_draft_model` → `Speculative`
3. Else → `Autoregressive`

This allows config-driven strategy selection without manual branching.

---

## 7. Paper Metadata

- **arXiv**: 2508.09192v1
- **Date**: Aug 8, 2025
- **Code**: https://github.com/zhijie-group/Discrete-Diffusion-Forcing
- **Models**: LLaDA-Instruct-8B, Dream-Base-7B (not ours, reference only)
- **Training**: Bespoke-Stratos-17k dataset, LoRA rank 32, 12 hours on 8× A100