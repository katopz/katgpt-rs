# Research 55: Nemotron-Labs-Diffusion — Tri-Mode Language Model

> **Paper:** [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying AR, Diffusion, and Self-Speculation Decoding](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf) — Fu et al., NVIDIA 2026 (21 pages)
> **Date:** 2025-07
> **Related:** Plan 089 (tri-mode inference prep), Plan 066 (D2F), Research 34 (D2F), Research 26 (MTP), Research 38 (SDAR)
> **Supersedes:** None — extends D2F + MTP with tri-mode unification

## Executive Summary

Nemotron-Labs-Diffusion unifies AR, diffusion, and self-speculation decoding in a single model via joint AR-diffusion training. The key insight: **AR and diffusion losses are complementary, not competing** — AR provides left-to-right priors, diffusion enables parallel decoding. Self-speculation (diffusion drafts, AR verifies) outperforms MTP methods like Eagle3 with 2.4-3.3× higher acceptance rates.

**Why we care:** We already have D2F (block-causal diffusion), MTP drafter, and speculative decoding as separate systems. This paper shows how to **unify them** into a single tri-mode model. The self-speculation mode is particularly valuable — it's the missing link between our D2F diffusion and AR speculative decoding.

**Key results (8B Instruct):**
- AR mode: +0.86% accuracy over Qwen3-8B (63.61 vs 62.75 avg)
- Diffusion mode: 2.57× tokens-per-forward, +0.43% accuracy over Qwen3-8B
- Self-speculation (linear): 5.99× TPF with LoRA drafter alignment
- Self-speculation beats Eagle3 by 2.4× at batch-size-1 throughput on GB200
- SOL analysis: diffusion has 76.5% more potential than self-speculation

---

## Paper Core

### 1. Joint AR-Diffusion Training

**Training objective:**
```
L(θ) = L_AR(θ) + α · L_diff(θ)    where α = 0.3
```

**Two-stage training:**
1. Stage 1: Pure AR (α=0) — establishes left-to-right priors
2. Stage 2: Joint AR+Diffusion (α=0.3) — enables parallel decoding

**Key finding:** Both modes peak at α=0.3. No value in [0.1, 0.5] improves one at the expense of the other. The objectives are complementary.

### 2. Dual-Stream Attention Pattern

```
[Noisy stream: x̃_t] [Clean stream: x]
 Noisy→Noisy: bidirectional within block, causal across blocks
 Noisy→Clean: attends to clean prefix blocks x_{<b}
 Clean→Clean: STRICTLY CAUSAL (key difference from prior work)
```

The strictly causal clean stream enables computing AR loss on the same forward pass — no label leakage.

### 3. Three Inference Modes

**Mode 1: AR Decoding** — Standard left-to-right, best for high concurrency.

**Mode 2: Diffusion Decoding** — Block-wise parallel denoising with confidence-based sampling or trained sampler. Best for throughput at batch-size-1.

**Mode 3: Self-Speculation** — The killer mode:
1. Diffusion drafts k tokens in parallel (bidirectional attention)
2. AR verifies in second forward pass (causal attention)
3. Accept longest prefix matching AR prediction
4. LoRA on o_proj (rank 128, α=512, ~36M params) aligns drafter with verifier

### 4. Trained Sampler (Appendix A)

Lightweight 4-layer Transformer (d=384, ~4.8M params) that predicts whether top-1 prediction at each masked position is correct. Input: 144-dim features (PCA-compressed embeddings + distribution statistics). Shifts Pareto frontier: +1.3× TPF or +10.6% accuracy.

### 5. Speed-of-Light (SOL) Analysis

Measures upper bound of diffusion parallel decoding:
- SOL acceptance rate: 7.60× at block_size=32
- Current confidence-based sampling: ~3×
- Self-speculation real TPF: 3.41× (two forwards per cycle)
- SOL real TPF: 6.02× (single forward)
- **76.5% headroom** between self-speculation and diffusion SOL

### 6. Global Loss Averaging

Treat all tokens across batch equally (not per-sequence average). Critical for training stability with variable masking ratios. Reduces gradient variance from random noise levels.

### 7. LoRA Drafter Alignment

```
L = λ_KL · L_LK + λ_CE · L_CE    where λ_KL = λ_CE = 1

L_LK: LK-hybrid distribution matching on top-K union support
L_CE: Cross-entropy on verifier argmax
Active positions: accepted prefix + first rejected only
Temperature: τ=3.0 for distributions, τ=1.0 for CE
```

---

## Cross-Reference: What We Already Have

| Nemotron Component | Our Code | Status |
|---|---|---|
| Causal attention | `transformer.rs` `forward()` | ✅ Production |
| Bidirectional attention | `dllm.rs` `forward_bidirectional_positions()` | ✅ Plan 066 |
| Block-causal attention | `dllm.rs` `forward_block_causal_positions()` | ✅ Plan 066 |
| Noise schedule | `dllm.rs` `NoiseSchedule` | ✅ Plan 066 |
| D2F block decode | `speculative/d2f.rs` `d2f_decode_block()` | ✅ Plan 066 |
| ConstraintPruner | `speculative/types.rs` trait | ✅ Production |
| Speculative decoding (AR→AR) | `speculative/step.rs` | ✅ Production |
| MTP drafter | `transformer.rs` mtp_activation_proj | ✅ Plan 055 |
| LoRA training (wgpu) | `riir-gpu` full stack | ✅ Production |
| SDAR sigmoid gating | `pruners/sdar_gate.rs` modelless | ✅ Plan 072 |
| Gemma 2 model loading | `riir-engine/safetensors_loader.rs` | ✅ Plan 087 |
| KV cache | `MultiLayerKVCache`, `PagedKVCache` | ✅ Production |
| Draft→Verify→Accept pattern | `speculative/step.rs` `speculative_step_rollback()` | ✅ Production |
| `SpeculativeVerifier` trait | `verifier.rs` with `SimulatedVerifier`, `LeviathanVerifier` | ✅ Production |
| DDTree path extraction | `speculative/dd_tree.rs` | ✅ Production |
| KV cache snapshot/rollback | `MultiLayerKVCache::snapshot()/restore()` | ✅ Production |
| Prefix acceptance logic | `LeviathanVerifier::speculate()` | ✅ Production |
| **D2F Drafter Verifier** | **MISSING — new `SpeculativeVerifier` impl** | ❌ ~100 lines |
| **Dual-stream attention** | **MISSING** (training-only, not needed for inference) | — |
| **Trained sampler** | **MISSING** | ❌ Research |
| **LoRA drafter alignment loss** | **MISSING** | ❌ riir-gpu |
| **Global loss averaging** | **MISSING** | ❌ ~30 lines |

---

## Distillation Ideas for Our System

### D1: D2F Drafter Verifier (LOW EFFORT — Same Pattern, Different Drafter)

**Honest take:** The Nemotron "self-speculation" is NOT a new system. It's the same `Draft → Verify → Prefix Accept` pattern we already have in `LeviathanVerifier`. The only delta is **what does the drafting**:

| Verifier | Drafter | Verify Method | Our Code |
|---|---|---|---|
| `SimulatedVerifier` | DFlash (AR) | Simulated acceptance | ✅ `verifier.rs` |
| `LeviathanVerifier` | DFlash (AR) + MTP activation | Real p/q rejection, DDTree, KV rollback | ✅ `verifier.rs` |
| **`D2fDrafterVerifier`** | **D2F diffusion (parallel)** | **Real AR verify, prefix accept** | ❌ **~100 lines** |

The actual code is a new `impl SpeculativeVerifier for D2fDrafterVerifier` that:
1. Calls `d2f_decode_block()` instead of `dflash_predict()` for drafting
2. Calls existing `forward()` for verification (same as `LeviathanVerifier`)
3. Uses existing prefix acceptance logic (same as `LeviathanVerifier`)

This is a **variant**, not a new system. The `SpeculativeVerifier` trait already abstracts the orchestration.

**Proof task:** Compare D2F drafter acceptance rate vs AR drafter (DFlash/Leviathan) on same model.

### D2: Mode-Adaptive Decode Strategy (LOW EFFORT)

We already have `DecodeStrategy` enum (AR / Speculative / DiscreteDiffusion). Add SelfSpeculation mode:

```rust
pub enum DecodeStrategy {
    Autoregressive,
    Speculative,
    DiscreteDiffusion,
    SelfSpeculation,  // NEW: Diffusion drafts, AR verifies
}
```

Auto-switch heuristic:
- High concurrency → AR
- Low concurrency + has model → SelfSpeculation
- No model → DiscreteDiffusion (modelless)

### D3: Trained Confidence Sampler (MEDIUM VALUE, MEDIUM EFFORT)

Replace fixed confidence threshold with learned per-position correctness predictor:

```rust
pub struct DiffusionSampler {
    // 4-layer transformer, d=384
    layers: [SamplerLayer; 4],
    // 144-dim input: PCA embeddings + distribution stats
    pca_proj: Vec<f32>,  // [144, embed_dim]
}
```

**Input features per position:**
- Top-1 probability, margin, top-3 mass, entropy
- PCA-compressed semantic embedding of top-1 prediction

**Output:** P(correct | position features) ∈ [0, 1]

**Proof task:** Train sampler on D2F denoising trajectories. Measure TPF improvement.

### D4: Global Loss Averaging for D2F Training (LOW EFFORT, HIGH VALUE)

Our D2F training uses per-sample loss averaging. Switch to global:

```rust
// BEFORE (per-sequence):
L = (1/N) * Σ_n (1/L) * Σ_i ℓ_{n,i}

// AFTER (global):
L = (1/(N*L)) * Σ_n Σ_i ℓ_{n,i}
```

When masking ratios vary across samples, global averaging reduces gradient variance. The paper shows +2.12% accuracy improvement from this alone.

### D5: LoRA Drafter Alignment for Self-Speculation (HIGH VALUE, in riir-gpu)

Train LoRA on o_proj to align diffusion drafter with AR verifier:

```
Loss = LK_hybrid(top-K=200, λ adaptive) + CE(verifier_argmax)
Active positions: accepted + first rejected only
LoRA: rank=128, α=512 on o_proj only (~36M params, 0.4% backbone)
```

**Result:** +14-33% relative TPF gain across 3B/8B/14B scales.

---

## What's NOT Directly Applicable

| Nemotron Aspect | Why Not For Us (Yet) |
|---|---|
| 1T-token pretraining | We train tiny models or LoRA-finetune |
| 256× H100 training | Consumer GPU (Metal/wgpu) |
| VLM extension | No vision encoder in our stack |
| Quadratic self-speculation | FlexAttention not in wgpu, kernel cost high |
| Ministral3 base models | We use Gemma 2 or random init |
| SGLang deployment | We're a Rust library, not a serving framework |

---

## Verdict

### Can We Do Tri-Mode?

**YES, partially.** Here's the honest breakdown:

| Capability | Can We? | What's Needed |
|---|---|---|
| Load a tri-mode model (GGUF/safetensors) | ✅ Yes | Already have safetensors loader for Gemma 2 |
| AR mode inference | ✅ Yes | Already production |
| Diffusion mode inference | ✅ Yes | D2F pipeline complete (Plan 066) |
| Self-speculation (Diff→AR verify) | ⚠️ Partially | Need new orchestration code, not new kernels |
| Train tri-mode from scratch | ❌ No | Need 1T-token pretraining infrastructure |
| Joint AR-Diffusion training | ⚠️ LoRA only | riir-gpu can do LoRA on block-causal forward |
| Trained sampler | ⚠️ Research | Need trajectory collection + small model training |
| VLM tri-mode | ❌ No | No vision encoder |

### What We Should Steal (Priority Order)

1. **D2F drafter verifier** (P0) — new `SpeculativeVerifier` impl, ~100 lines, no new kernels
2. **Global loss averaging** (P1) — one-line change in `masked_loss()`, +2.12% accuracy
3. **Mode-adaptive decode** (P1) — extend existing `DecodeStrategy` enum with `SelfSpeculation` variant
4. **LoRA drafter alignment loss** (P2) — new training loss in riir-gpu
5. **Trained sampler** (P3) — research project, needs trajectory data

### What We Should NOT Steal

- Full pretraining pipeline (we LoRA-finetune, not pretrain)
- Quadratic self-speculation (kernel complexity not worth it)
- VLM asymmetric dual-stream (no vision encoder)

### Honest Assessment

The paper validates our D2F approach (Plan 066) — block-causal attention with bidirectional intra-block denoising is exactly what Nemotron uses. Our architecture is already aligned.

The missing piece is a **D2F drafter verifier** — a new `impl SpeculativeVerifier` that uses `d2f_decode_block()` instead of `dflash_predict()` for drafting. This is ~100 lines of new code, no new GPU kernels, and plugs directly into our existing `SpeculativeVerifier` trait. The draft→verify→accept pattern is identical to `LeviathanVerifier`; only the drafter backend changes.

The SOL analysis is the most exciting finding: **diffusion decoding has 76.5% headroom beyond self-speculation**. This means our D2F investment has long-term upside — better samplers will unlock more parallelism over time.

---

## References

- Nemotron-Labs-Diffusion: https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf
- D2F (our implementation): `.research/034_D2F_Discrete_Diffusion_Forcing.md`, Plan 066
- MTP Drafter (our implementation): `.research/026_Gemma_4_MTP_Multi_Token_Prediction.md`, Plan 055
- SDAR Gating (our implementation): `.research/038_SDAR_Self_Distilled_Agentic_RL.md`, Plan 072/073
- Block Diffusion: arXiv:2503.09573
- LK Losses: arXiv:2602.23881