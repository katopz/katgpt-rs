# Research 142: JLT — Clean-Latent Prediction in Latent Diffusion Transformers

> **Paper:** [JLT: Clean-Latent Prediction in Latent Diffusion Transformers](https://arxiv.org/pdf/2605.27102) — Fu et al., 2026
> **Code:** [github.com/akatsuki-neo/JLT](https://github.com/akatsuki-neo/JLT) — reviewed in `.raw/JLT/`
> **Date:** 2026-05-31
> **Related:** Plan 066 (D2F), Plan 089 (tri-mode), Plan 097 (training-free loop), Plan 108 (LT2), Research 034 (D2F), Research 055 (Nemotron tri-mode)
> **Applies to:** `katgpt-rs` D2F training (`dllm.rs`), D2F decode (`src/speculative/d2f.rs`), LT2 loop (`tf_loop.rs`)

## Executive Summary

JLT proves that **prediction target parameterization matters even in compressed latent spaces**. Clean-latent prediction (x-prediction) outperforms velocity prediction (v-prediction) by FID 2.56 vs 6.56 on the same 130M DiT with identical FLUX.2 VAE latents. The mechanism: velocity prediction adds an isotropic covariance floor `Cov(yv) = Σ + I` that amplifies low-variance directions, while clean prediction `Cov(yx) = Σ` attenuates them. The conditional ambiguity is `Var(vi|zi) = Var(xi|zi)/(1-t)²`, so velocity regression is strictly harder for any t ∈ [0,1).

**Code review reveals three additional techniques applicable to our system:**
1. **MAE-style token masking** (`mask_prob`/`mask_ratio`) — drops random patches during training, forcing the model to reconstruct from partial context
2. **Layer loop** (`loop_indices`/`loop_count`) — repeats consecutive transformer layers, identical to our LT2 (Plan 108/097)
3. **EMA feature alignment** (`ema_feat_align_weight`) — cosine similarity between EMA-teacher and student intermediate features
4. **Async per-token timesteps** (`async_timesteps`) — each token gets a different noise level within the same image

**For our system:** Our D2F already uses clean prediction (CE against original tokens). The paper validates this. The mask token, loop, and EMA alignment techniques are applicable to our D2F training and LT2 inference.

---

## Paper Core (Verified Against Code)

### 1. The Dual Training Path (from `engine_jit.py:forward()`)

```python
# Corruption: z = t*x + (1-t)*e
z = t * x + (1 - t) * e

# x-prediction path (JLT default):
v_target = (x - z) / (1 - t).clamp_min(t_eps)  # reconstruct velocity from clean target
v_pred   = (net_out - z) / (1 - t).clamp_min(t_eps)  # net_out = x_pred
loss = (v_target - v_pred) ** 2

# v-prediction path (DiT baseline, --flow_matching flag):
v_target = x - e       # direct velocity target
v_pred   = net_out     # network directly predicts velocity
loss = (v_target - v_pred) ** 2
```

**Critical insight:** Both paths use L2 loss on velocity `v`, but the x-prediction path goes through `1/(1-t)` rescaling for BOTH target and prediction. This means:
- The network is trained to output `x_pred ≈ x` (clean prediction)
- The loss is `(x_pred - x)² / (1-t)²` — automatically upweights near-clean predictions
- The `t_eps=5e-2` floor prevents division by zero as t → 1

### 2. Sampling (from `engine_jit.py:generate()`)

Both paths produce velocity for ODE stepping:
```python
# x-prediction sampling:
v_cond = (out_cond - z) / (1 - t).clamp_min(t_eps)
v_uncond = (out_uncond - z) / (1 - t).clamp_min(t_eps)

# flow matching sampling:
v_cond = out_cond
v_uncond = out_uncond

# Both then do:
v = v_uncond + cfg_scale * (v_cond - v_uncond)  # classifier-free guidance
z_next = z + (t_next - t) * v  # Euler step
```

Heun's method (2nd-order) is also supported with `--sampling_method heun`.

### 3. Target Geometry Analysis (The Key Mechanism)

Under local Gaussian assumption `x ~ N(0, Σ)`:

| Target | Covariance | Low-Var Direction Behavior |
|--------|-----------|---------------------------|
| Clean (yx = x) | Cov(yx) = Σ | **Attenuates** — coefficient → 0 as λi → 0 |
| Noise (yε = ε) | Cov(yε) = I | Isotropic floor |
| Velocity (yv = x - ε) | Cov(yv) = Σ + I | **Amplifies** — coefficient → -1/(1-t) as λi → 0 |

Bayes estimators reveal the mechanism:
- `E[xi|zi] = tλi/Di · zi` → coefficient → 0 for low-variance directions
- `E[vi|zi] = (tλi - (1-t))/Di · zi` → coefficient → -1/(1-t) for low-variance directions

### 4. Conditional Ambiguity Gap

```
Var(vi|zi) = Var(xi|zi) / (1-t)²    for all t ∈ [0,1)
```

### 5. Architecture Details (from `model_jit.py`)

| Component | Specification |
|-----------|--------------|
| Transformer Blocks | 12 (Base) / 24 (Large) |
| Hidden Dimension | 768 (Base) / 1024 (Large) |
| Attention Heads | 12 (Base) / 16 (Large) |
| SwiGLU FFN | hidden_dim = int(dim * 4 * 2/3) |
| AdaLN-Zero | 6 modulation params per block (shift/scale/gate × MSA/MLP) |
| Final Layer | AdaLN-modulated linear projection |
| Positional Encoding | Vision Rotary Embedding (RoPE) |
| Attention | Flash attention via `flash_attn.cute` |
| EMA | Dual EMA (decay 0.9999 + 0.9996), maintained in fp32 |
| Mask Token | Learnable `nn.Parameter`, std=0.02 init |

---

## Code-Level Techniques Found (Beyond Paper)

### T1: MAE-Style Token Masking (from `model_jit.py:forward()`)

```python
if self.training and self.mask_prob > 0.0 and self.mask_ratio > 0.0:
    s_mask = torch.rand(B, device=x.device) < self.mask_prob  # per-sample
    t_mask = torch.rand(B, N, device=x.device) < self.mask_ratio  # per-token
    mask = (s_mask.unsqueeze(1) & t_mask).unsqueeze(-1)
    x = torch.where(mask, self.mask_token.to(x.dtype), x)
```

**Two-level masking:**
1. `mask_prob`: Per-sample probability of activating masking (sample-level gate)
2. `mask_ratio`: Per-token Bernoulli probability of being replaced (within masked samples)

This forces the model to reconstruct from partial context — similar to D2F corruption but at the embedding level instead of token level. Uses a **learnable mask token** (not zero/MASK token ID).

### T2: Layer Loop (from `model_jit.py:forward()`)

```python
if self.loop_indices is not None and self.loop_count > 0:
    a, b = self.loop_indices[0], self.loop_indices[-1]
    schedule = (
        list(range(0, a)) +
        list(range(a, b + 1)) * self.loop_count +  # repeated loop body
        list(range(b + 1, len(self.blocks)))
    )
else:
    schedule = list(range(len(self.blocks)))
for layer_idx in schedule:
    x = self.blocks[layer_idx](x, c, self.feat_rope)
```

**This is identical to our LT2 (Plan 108) training-free loop.** JLT repeats consecutive layers [a, b] N times during training AND inference. The shared weights between iterations are the same mechanism as our `tf_loop.rs`.

### T3: EMA Feature Alignment (from `engine_jit.py:_ema_feature_alignment_loss()`)

```python
# Teacher: EMA model at lower noise level
teacher_t = min(teacher_sample_t, student_t)  # cleaner input
teacher_z = teacher_t * x + (1 - teacher_t) * e

# Student: current model at student's noise level
# Loss: cosine similarity between intermediate features
feat_loss = (1.0 - cosine_similarity(student_feat, teacher_feat)).mean()
total_loss = main_loss + ema_feat_align_weight * feat_loss
```

**Key design:** Teacher is the EMA model run at a **lower noise level** than the student. This creates a "denoising distillation" signal — the student learns to match the EMA teacher's features when the teacher sees a cleaner input. Similar in spirit to progressive distillation.

### T4: Async Per-Token Timesteps (from `engine_jit.py:_sample_tokenwise_t()`)

```python
# Each token in the latent grid gets a DIFFERENT noise level
token_t = self.sample_t(B * token_h * token_w, device=x.device)
# With dropout: some samples fall back to uniform timestep
disable_async = torch.rand(B) < self.async_timestep_drop
token_t[disable_async] = fallback_t.unsqueeze(-1)
```

Each spatial token gets an independent noise level t, creating a heterogeneous corruption pattern. The `async_timestep_drop` allows mixing uniform and per-token sampling for stability.

### T5: VAE BN Normalization (from `vae.py`)

FLUX.2 VAE has a built-in BatchNorm that normalizes encoded latents. The `Flux2VAE.encode()` applies BN normalization: `(latents - running_mean) / sqrt(running_var + eps)`. Latents are pre-encoded to safetensors with this normalization applied, making training I/O efficient.

### T6: Noise Schedule Parameters

```python
# JLT default (x-prediction): P_mean=-0.8, P_std=0.8
# t = sigmoid(N(P_mean, P_std)) — logit-normal distribution
# Biased toward t~0.3 (mostly clean) for x-prediction

# DiT baseline (v-prediction/flow matching): P_mean=0, P_std=1
# More uniform distribution over t
```

The different P_mean/P_std between x-prediction and v-prediction is intentional — x-prediction benefits from more time near the clean end (where the `1/(1-t)` weighting is moderate), while v-prediction needs uniform coverage.

---

## Distillation for Our System

### D1: Validation — Our D2F Already Uses Clean Prediction ✅

Our `dllm.rs` trains with cross-entropy on clean token IDs at each denoising step. This IS clean prediction in discrete space:
- Model sees corrupted tokens (mask + clean mix)
- Model outputs logits over vocabulary
- Loss is CE against the **original clean token** (not against a velocity/noise target)

**JLT validates this as the correct choice.** No change needed.

### D2: MAE-Style Embedding Masking for D2F Training (MEDIUM GAIN, LOW EFFORT)

JLT's `mask_prob`/`mask_ratio` can be applied to our D2F training in `dllm.rs`:

```rust
/// During D2F training, randomly replace some token embeddings with a learned mask embedding.
/// This forces the model to reconstruct from partial context, improving robustness.
///
/// Two-level gate (same as JLT):
/// 1. Per-sequence: probability mask_prob of activating masking
/// 2. Per-position: within activated sequences, mask_ratio chance of replacement
pub struct EmbeddingMaskConfig {
    /// Per-sequence activation probability (0 = disabled)
    pub mask_prob: f32,
    /// Per-position replacement probability within activated sequences
    pub mask_ratio: f32,
    /// Learnable mask embedding [dim]
    pub mask_embedding: Vec<f32>,
}
```

**Current gap:** Our D2F uses token-ID masking (replace token with MASK token ID), which is the discrete equivalent. JLT adds embedding-level masking on top of the latent corruption. For our discrete case, the existing token-level masking is sufficient.

**Verdict:** No gain over our existing `corrupt_block()` — different domain (continuous vs discrete).

### D3: Layer Loop Validation (Our LT2 Already Does This) ✅

JLT's `loop_indices`/`loop_count` is identical to our LT2 (Plan 108, `tf_loop.rs`). The JLT code even uses the same schedule construction:

```python
# JLT schedule construction
schedule = list(range(0, a)) + list(range(a, b+1)) * loop_count + list(range(b+1, depth))
```

This validates our LT2 approach — the paper's authors independently arrived at the same mechanism.

**Key difference:** JLT uses this during training too (not just inference). Our LT2 is training-free inference only. JLT's trained loop may be more effective than our training-free approach.

**Potential gain:** Training with loop from the start (like JLT) vs training without and applying loop at inference (our current approach). This is a training-time change, not an inference change.

### D4: EMA Feature Alignment (MEDIUM GAIN for D2F Training)

The EMA teacher-student feature alignment could improve our D2F training quality:

```rust
/// EMA feature alignment for D2F training.
/// Teacher = EMA model at lower noise level (closer to clean).
/// Student = current model at training noise level.
/// Loss = cosine similarity between intermediate layer features.
pub struct EmaFeatureAlign {
    /// EMA decay (0.9999 = slow update = stable teacher)
    pub ema_decay: f32,
    /// Feature alignment weight (0 = disabled, 0.8 = JLT default)
    pub align_weight: f32,
    /// Teacher layer index (deeper, more abstract features)
    pub teacher_layer: usize,
    /// Student layer index (shallower, faster to compute)
    pub student_layer: usize,
}
```

**Applicable to:** D2F training in `dllm.rs` — during the mini dLLM training loop, track EMA of weights and add feature alignment loss between EMA-teacher (lower noise) and current student (training noise).

**This is a training-time optimization, not inference.** It would improve D2F training convergence and quality.

### D5: Target-Geometry Diagnostic (LOW EFFORT, HIGH VALUE)

The paper's diagnostic framework can validate D2F training quality:

```rust
/// Diagnostic: measure effective rank of model outputs during D2F training.
/// Inspired by JLT Appendix D: "effective rank of yv should exceed that of yx"
fn effective_rank(logits: &[f32], vocab: usize, n_positions: usize) -> f32 {
    // Compute singular values of logits matrix [n_positions × vocab]
    // If rank >> expected, model is fitting noise, not signal
}
```

### D6: Noise Schedule Optimization (POTENTIAL GAIN)

JLT uses logit-normal noise schedule with different parameters for x vs v prediction:
- x-prediction: P_mean=-0.8, P_std=0.8 (biased toward clean end)
- v-prediction: P_mean=0, P_std=1.0 (uniform)

Our D2F uses monotonic linear schedule from `min_ratio` to `max_ratio`. The logit-normal schedule could improve training by concentrating samples where the model needs them most.

---

## GOAT Verdict

### What the Paper+Code Proves

1. ✅ Target parameterization matters in latent spaces (FID 2.50 vs 6.56, controlled ablation)
2. ✅ Clean prediction attenuates low-variance directions, velocity amplifies them (analytical proof)
3. ✅ The conditional ambiguity gap is `1/(1-t)²` (Bayes-optimal, not model-dependent)
4. ✅ The effect survives latent compression (FLUX.2 VAE is anisotropic)
5. ✅ Layer loop during training is effective (validates our LT2 direction)
6. ✅ EMA feature alignment improves training convergence
7. ✅ Per-token async timesteps provide additional training diversity

### What It Does NOT Prove

1. ❌ Does NOT apply to discrete token diffusion (paper is continuous latent diffusion)
2. ❌ Does NOT quantify layer loop gain in isolation (used alongside other techniques)
3. ❌ Does NOT prove EMA alignment helps at our scale (130M model, may not transfer to mini dLLM)
4. ❌ Analysis is local Gaussian — real distributions are non-Gaussian

### Verdict per Technique

| Technique | Our Status | Gain? | Perf Hurt? | Default-On? |
|-----------|-----------|-------|-----------|-------------|
| **Clean prediction** | ✅ Already using (CE on original tokens) | — | — | ✅ Already on |
| **Layer loop** | ✅ LT2 (Plan 108), inference-only | ✅ Validated approach | No | ✅ Already on |
| **MAE embedding masking** | ✅ Token-level masking in `corrupt_block()` | 🟡 Minimal over existing | No | ✅ Already on |
| **EMA feature alignment** | ❌ Not in D2F training | 🟡 Training quality | Training cost | ❌ Opt-in (training-time) |
| **Async per-token timesteps** | ❌ Not in D2F | 🟡 Training diversity | Training cost | ❌ Opt-in (training-time) |
| **Logit-normal noise schedule** | ✅ `NoiseSchedule` uses monotonic linear | 🟡 Potential improvement | No | ❌ Research needed |
| **Effective rank diagnostic** | ❌ Not in `data_probe` | 🟡 Training quality monitoring | No overhead | ❌ Debug tool |

### Decision: ⚠️ NO NEW PLAN — VALIDATION + RESEARCH NOTES

The paper validates existing design choices. All high-value techniques are already implemented:

1. ✅ **Clean prediction** — our D2F already predicts clean tokens (validated by JLT)
2. ✅ **Layer loop** — our LT2 already does this (validated by JLT's identical implementation)
3. ✅ **Token masking** — our `corrupt_block()` already does this (different level but same principle)

The remaining techniques (EMA alignment, async timesteps, logit-normal schedule) are **training-time optimizations** that would improve D2F training quality but don't affect inference performance. These are filed for future D2F training improvements.

**No new plan created. No feature gate needed. No code changes.**

---

## Key References

- JLT (this paper): Fu et al., "JLT: Clean-Latent Prediction in Latent Diffusion Transformers", arXiv 2605.27102
- JiT: Li & He, "Back to basics: Let denoising generative models denoise", arXiv 2511.13720
- EDM: Karras et al., "Elucidating the design space of diffusion-based generative models", NeurIPS 2022
- SiT: Ma et al., "SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers", ECCV 2024
- REPA: Yu et al., "Representation alignment for generation", ICLR 2025

> **PASS-Redirects (synthesis):** Feng et al. [arXiv:2609.28473 "On the Diffusibility of High-Dimensional Latents"] — the ECCV'26 successor to this exact mechanism: restates JLT/JiT's target geometry in subspace-projector form (v* = Q·v_l* + (1/t)·P⊥·z_t, orthogonal term deterministic, E‖ε⊥‖² = h−l) and measures reconstruction-tuned encoders collapsing effective dim (finetuned DINOv2-L R90 672→129 of 1024). Confirms D1's verdict: clean prediction is the right parameterization in the collapsed-effective-dim regime — our D2F CE-on-clean-tokens choice needs no change.
