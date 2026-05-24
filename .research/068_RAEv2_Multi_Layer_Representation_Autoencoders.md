# Research 68: RAEv2 — Multi-Layer Representation Autoencoders

**Paper:** Improved Baselines with Representation Autoencoders (arXiv:2605.18324)
**Authors:** Singh et al.
**Date:** May 2026
**Distilled:** 2026-07

---

## 1. TL;DR

RAEv2 improves diffusion model generation via three mechanisms: (1) **Multi-Layer Sum (MLS)** — sum last K encoder layers instead of using only the final layer, training-free with zero new parameters; (2) **RAE + REPA complementarity** — semantic latent space + spatial structure regularization work together, not redundantly; (3) **Self-guidance via REPA head** — `x_guided = x_full + w * (x_full - x_repa)` eliminates need for separate guidance model or extra forward pass. Results: 10× faster convergence (gFID 1.06 in 80 epochs vs RAE 800 epochs).

**Verdict: MODERATE TRANSFER, TWO ACTIONABLE DISTILLATIONS. This is a vision/diffusion paper being distilled to a text LLM inference engine — the MLS pattern and self-guidance pattern transfer cleanly, but REPA spatial alignment and flow matching do not. Key takeaways: (1) Multi-layer residual aggregation for richer token representations maps to our `forward_base` layer loop — low effort, may improve speculative draft quality and early exit confidence. (2) Self-guidance pattern `x_guided = x_full + w*(x_full - x_weak)` is essentially what our MTP drafter + speculative verification already approximates, but formalized as an explicit inference-time signal — no new forward pass needed. (3) REPA's spatial alignment and the RAE autoencoder training do NOT transfer — we don't train, we don't do diffusion, we don't have spatial structure.**

---

## 2. What RAEv2 Actually Does

### 2.1 Multi-Layer Sum (MLS) — The Training-Free Win

Standard transformers use only the **final layer's output** as the representation fed to the LM head (or unet head). RAEv2 observes that intermediate layers carry complementary information. Their fix is trivially simple:

```text
# Standard: use only final layer
z = encoder_layer_L(x)

# MLS: sum last K layers
z = Σ z_l  for l in L-K+1..L
```

This is:
- **Training-free** — no new parameters, no fine-tuning
- **Preserves latent shape** — same dimensionality
- **Pareto-optimal** — sweeping K trades off reconstruction fidelity vs generation quality

The key insight: final layers specialize for the training objective, but intermediate layers retain "raw" features that improve generalization. Summing them is a cheap way to recover what the last layer overfits away.

### 2.2 RAE + REPA Complementarity

- **RAE** (Representation Autoencoder): Trains an encoder-decoder on frozen representations. Provides a **semantic latent space** — captures WHAT the image means, not just pixel patterns.
- **REPA** (REPresentation Alignment): Regularizes intermediate diffusion features to match clean image representations. Provides **spatial structure** — ensures the diffusion process generates coherent geometry.

They are NOT redundant. Ablation shows both together >> either alone. The paper proves this by using the same pretrained representation as BOTH the RAE encoder AND the REPA intermediate target — the representation serves double duty.

### 2.3 Self-Guidance via REPA Head

The REPA head does **x-prediction in RAE latent space** — predicting the clean target from noisy intermediate features. When the main output is ALSO reformulated as x-prediction, you get internal guidance:

```text
x_guided = x_full + w * (x_full - x_repa)
```

Where:
- `x_full` = full model's prediction (strong)
- `x_repa` = intermediate-layer REPA head prediction (weak)
- `w` = guidance strength hyperparameter

This eliminates:
- **AutoGuidance**: No need for a separate (smaller) model
- **CFG**: No need for an extra unconditional forward pass
- Runs in the **same forward pass** — the REPA head is already computed as part of REPA regularization

The pattern is: strong prediction + (strong - weak) contrast = guided output. The "weak" signal comes from an earlier layer, not a separate model.

### 2.4 Key Results

| Metric | RAE (baseline) | RAEv2 | Speedup |
|---|---|---|---|
| gFID (ImageNet 256×256) | ~2.5 | 1.06 | 10× faster convergence |
| Epochs to gFID ≤ 1.5 | 800 | 80 | 10× |
| Reconstruction FID | Worse | Pareto-optimal | Tunable via K |
| Extra parameters | 0 | 0 (MLS) | Training-free |

---

## 3. Honest Applicability Assessment

**Critical context**: RAEv2 is a VISION/DIFFUSION paper. We are a TEXT/LLM inference engine. Some ideas transfer cleanly; others do not.

### 3.1 What Transfers ✅

| RAEv2 Concept | Transfer Mechanism | Why It Works |
|---|---|---|
| Multi-Layer Sum (MLS) | Sum residual states from last K layers before LM head | Intermediate LLM layers carry rich syntactic/semantic signals; final layer is task-specialized |
| Self-guidance pattern | `x_guided = x_full + w * (x_full - x_weak)` using intermediate vs final logits | Strong-weak contrast improves distribution without extra forward pass |
| Pareto sweep of K | Config knob for `mls_layers` | Different tasks (speculative decoding vs beam search) may benefit from different K |
| EP FID@k metric | Training-efficiency metric adapted to our GOAT proofs | "Steps to reach accuracy ≤ k" is universally useful |

### 3.2 What Partially Transfers 🟡

| RAEv2 Concept | Why Partial | What We Can Salvage |
|---|---|---|
| REPA alignment signal | We don't have spatial structure to regularize | Intermediate-layer confidence estimation for early exit |
| RAE semantic latent | We don't train autoencoders | The IDEA that intermediate layers = "raw features" validates our MTP drafter using mid-layer activations |
| x-prediction reformulation | We predict logits, not clean images | Logit-space guidance: contrast full-model vs mid-layer logit distributions |

### 3.3 What Does NOT Transfer ❌

| RAEv2 Concept | Why Not |
|---|---|
| Flow matching (RAE training) | We don't train diffusion models |
| RAE encoder-decoder | We're inference-only, no training pipeline |
| REPA spatial structure regularization | Text tokens don't have spatial geometry |
| gFID metric directly | gFID measures image generation quality; we need perplexity/accuracy metrics |
| Autoencoder bottleneck | No compression objective in inference |

---

## 4. Mapping to Our Architecture

### 4.1 MLS → Multi-Layer Residual Aggregation

Our `forward_base` currently runs all layers and uses only the **final** `ctx.x` for the LM head:

```katgpt-rs/src/transformer.rs#L823-1040
fn forward_base<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    // ...
    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // ... attention + MLP ...
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
    }
    // 3. LM Head uses only ctx.x (final layer output)
    standard_lm_head(&mut ctx.logits, &ctx.x, &weights.lm_head, config.vocab_size, n);
}
```

The RAEv2 MLS pattern would accumulate residual states from the last K layers:

```text
// Current: logits = lm_head @ x_final
// Proposed: logits = lm_head @ Σ x_l  for l in (n_layer - K)..n_layer
```

**Where this would go in our code:**

1. `ForwardContext` gains an `mls_accumulator: Vec<f32>` buffer (n_embd size)
2. In the layer loop, for layers `>= n_layer - K`, accumulate: `mls_accumulator += x` after the MLP residual add
3. After the loop, use `mls_accumulator / K` (or unnormalized) as LM head input instead of raw `ctx.x`

**Expected benefits:**
- **Speculative draft quality**: MTP drafter uses richer intermediate representations
- **Early exit confidence**: Layer-aggregated signal is more stable than single-layer output
- **Screening/relevance scoring**: `ScreeningPruner` gets better token embeddings

### 4.2 Self-Guidance → Inference-Time Logit Contrast

RAEv2's `x_guided = x_full + w * (x_full - x_repa)` maps to our speculative decoding pipeline:

```text
# RAEv2 pattern:
x_guided = x_strong + w * (x_strong - x_weak)

# Our equivalent:
logits_guided = logits_target + w * (logits_target - logits_draft)
```

But we already do something similar! Our `LeviathanVerifier` rejects draft tokens where draft ≠ target distribution. The RAEv2 insight is that the "weak" signal can come from an **intermediate layer of the SAME model**, not a separate draft model.

This is related to but distinct from:
- **SDAR gated** (`sdar_gate`): Sigmoid-gated distillation signal at training time
- **MTP drafter** (`mtp_*`): Multi-token prediction using target model activations
- **Leviathan verification**: Accept/reject based on full-model forward pass

The novel contribution: if we compute logits at an **intermediate layer** (say layer n_layer/2) using a lightweight LM head, we get a "weak" prediction for free during the same forward pass. Then:

```text
logits_guided = logits_final + w * (logits_final - logits_intermediate)
```

This is essentially **free CFG** — classifier-free guidance without the unconditional forward pass.

### 4.3 EP FID@k → EP Accuracy@k for GOAT

RAEv2 measures training efficiency as "epochs to reach gFID ≤ k" (EP FID@k). This is analogous to measuring distillation efficiency in our game arenas:

```text
# RAEv2: EP FID@1.5 = 80 epochs (RAEv2) vs 800 epochs (RAE)
# Our equivalent: EP Accuracy@0.8 = steps to reach win_rate ≥ 0.8 in GOAT arena
```

We should adopt this metric for GOAT proofs. Currently we report final win rates; the efficiency metric (how quickly we reach a target) captures convergence speed, which is what RAEv2's 10× improvement is really about.

---

## 5. What We Already Have (RAEv2 Validates)

### 5.1 Multi-Layer KV Cache ✅

We already store per-layer KV caches in `MultiLayerKVCache`:

```text
pub struct MultiLayerKVCache {
    pub layers: Vec<LayerKVCache>,  // [n_layer]
}
```

This validates the MLS principle — we already know intermediate layers matter (we cache them). The missing piece is **aggregating the output representations**, not just caching the attention inputs.

### 5.2 MTP Drafter as "Intermediate Prediction" ✅

Our MTP drafter (`mtp_activation_proj`) projects target-model activations to draft-model space for multi-token prediction. This is conceptually similar to RAEv2's REPA head — a lightweight head that predicts tokens from intermediate features.

The difference: MTP drafter is for generating multiple tokens ahead (drafting), while REPA head is for self-guidance (improving single-token quality). But the mechanism — lightweight prediction from intermediate representations — is the same.

### 5.3 Early Exit as Layer Selection ✅

Our early exit mechanism (`early_exit_patience`, `early_exit_gap`) already uses intermediate layer confidence. RAEv2's MLS validates that intermediate layers carry useful signal — we exploit this for early exit, they exploit it for representation quality.

### 5.4 SDAR Gated as Self-Distillation ✅

Our `sdar_gate` feature implements sigmoid-gated distillation signals at inference time. RAEv2's self-guidance pattern is structurally similar: both combine a "strong" and "weak" signal with a learned/fixed weight.

### 5.5 Delta Routing as Cross-Layer Aggregation ✅

Our `delta_routing` feature (Plan 097) accumulates per-block deltas and routes them across layers. This is a more sophisticated form of cross-layer information flow than simple summation. MLS validates the idea that cross-layer aggregation helps; delta routing goes further by routing selectively.

---

## 6. What We Don't Need

### 6.1 RAE Autoencoder Training

RAEv2 trains an encoder-decoder on frozen representations. We're an inference engine — no training pipeline, no autoencoder, no flow matching. The training procedure is irrelevant to us.

### 6.2 REPA Spatial Structure Regularization

REPA regularizes the spatial structure of intermediate diffusion features. Text tokens don't have spatial geometry — there's no 2D structure to regularize. The spatial alignment loss is vision-specific.

### 6.3 Flow Matching / Diffusion Process

RAEv2 operates in the diffusion framework (noise scheduling, denoising, flow matching). We do autoregressive inference. The entire diffusion training pipeline is inapplicable.

### 6.4 gFID Metric Directly

gFID (generative Fréchet Inception Distance) measures image generation quality using Inception features. We need text-quality metrics (perplexity, accuracy, win rate). The metric doesn't transfer; the efficiency-measurement IDEA does.

### 6.5 Separate Representation Model

RAEv2 uses a pretrained representation model (e.g., DINOv2) as the RAE encoder. We don't have or need a separate representation model — the transformer layers ARE the representation.

---

## 7. What IS Worth Exploring

### 7.1 MLS Residual Aggregation (Medium Effort, Medium Impact)

Add a config knob `mls_layers: usize` (default 0 = disabled, standard behavior) that aggregates residual states from the last K layers before the LM head.

**Proposed changes:**

1. `Config` gains `mls_layers: usize` (0 = off, K = sum last K layers)
2. `ForwardContext` gains `mls_accumulator: Vec<f32>` and `mls_count: usize`
3. In `forward_base`, for layers `>= n_layer - config.mls_layers`, after MLP residual: `mls_accumulator += x`
4. After layer loop, if `mls_count > 0`, use `mls_accumulator / mls_count` as LM head input

**Feature gate:** `mls_aggregate` (opt-in, off by default — needs benchmarking first)

**Risk:** Summing layers may HURT final-layer specialization for well-trained models. The Pareto sweep is essential — some models may want K=0 (standard), others K=2-4.

**Benchmark:** GOAT proof with K sweep on speculative acceptance rate.

### 7.2 Intermediate-Layer Self-Guidance (Medium Effort, High Impact if it Works)

Add a lightweight LM head at an intermediate layer (e.g., layer `n_layer / 2`) and use the logit contrast for self-guidance:

```text
logits_guided = logits_final + w * (logits_final - logits_intermediate)
```

This is "free CFG" — no separate model, no extra full forward pass. The intermediate LM head is a small `matmul(n_embd, vocab_size)` computed mid-forward.

**Proposed changes:**

1. `TransformerWeights` gains `intermediate_lm_head: Option<Vec<f32>>` at layer `n_layer / 2`
2. `ForwardContext` gains `intermediate_logits: Vec<f32>`
3. In `forward_base`, at the target intermediate layer, compute logits via intermediate LM head
4. After final logits, apply: `logits_guided = logits + w * (logits - intermediate_logits)`

**Feature gate:** `self_guidance` (opt-in, off by default — needs weight training)

**Risk:** The intermediate LM head must be TRAINED (not random). This requires a training pipeline we don't have yet. However, for models that expose intermediate-layer heads (like Gemma 2's mid-layer predictions), this is free.

**Connection to MTP:** Our MTP drafter already projects mid-layer activations. The intermediate LM head is essentially the MTP projection head repurposed for guidance rather than drafting.

### 7.3 EP Accuracy@k Metric for GOAT (Trivial Effort, High Impact)

Adopt RAEv2's efficiency metric pattern for our GOAT arena proofs:

```text
# Current GOAT report:
Win rate: 0.83 (vs baseline)

# Proposed EP Accuracy@k:
EP Accuracy@0.8: 150 self-play rounds (this method) vs 800 rounds (baseline)
Convergence: 5.3× faster to target accuracy
```

This captures not just "how good" but "how fast" — the key contribution of RAEv2's efficiency framing.

**No feature gate needed** — this is a benchmark/reporting change only.

### 7.4 NOT Worth Exploring

- Do NOT add autoencoder training to the inference engine
- Do NOT add diffusion/flow matching to the inference path
- Do NOT add spatial structure regularization (REPA) for text tokens
- Do NOT add a separate representation model — our transformer layers ARE the representation
- Do NOT replace the final LM head with an intermediate one — MLS augments, not replaces

---

## 8. Component Mapping Table

| RAEv2 Concept | Our Equivalent | Status | Transfer Quality |
|---|---|---|---|
| Multi-Layer Sum (MLS) | **Proposed:** `mls_aggregate` feature | 🟡 New | ✅ Clean transfer |
| REPA head (x-prediction) | MTP drafter activation projection | ✅ Existing | 🟡 Mechanism exists, purpose differs |
| Self-guidance `x_guided = x + w*(x-x_repa)` | **Proposed:** `self_guidance` feature | 🟡 New | ✅ Clean transfer |
| RAE encoder-decoder | Delta routing cross-layer flow | ✅ Existing | 🟡 Different mechanism, similar goal |
| REPA spatial regularization | — | ❌ N/A | ❌ No spatial structure in text |
| Flow matching training | — | ❌ N/A | ❌ No training pipeline |
| gFID metric | GOAT win rate | ✅ Existing | 🟡 Metric type differs |
| EP FID@k efficiency | **Proposed:** EP Accuracy@k | 🟡 New | ✅ Clean transfer |
| Pareto sweep of K | Config knob pattern | ✅ Established | ✅ Same pattern |
| Pretrained representation | Transformer layers themselves | ✅ Existing | ✅ Self-representation |
| Intermediate layer confidence | Early exit confidence gap | ✅ Existing | ✅ Clean transfer |

---

## 9. Concrete Architecture Sketch

### 9.1 Where MLS Goes in `forward_base`

The current layer loop writes to `ctx.x` at each layer, overwriting the previous state. MLS needs to **accumulate** states from the last K layers:

```text
// After the layer loop, before LM head:

// Current:
// ctx.x has final layer output → standard_lm_head(ctx.logits, ctx.x, ...)

// With MLS (config.mls_layers > 0):
// 1. Accumulator buffer: ctx.mls_buf initialized to 0.0
// 2. In layer loop, for layers >= (n_layer - mls_layers):
//      ctx.mls_buf += ctx.x  (after MLP residual add)
// 3. After loop: ctx.mls_buf /= mls_layers
// 4. Use ctx.mls_buf as LM head input instead of ctx.x

// Buffer cost: n_embd f32 values = ~2KB for n_embd=512
// Compute cost: n_embd * mls_layers additions = ~2K FLOPs for K=4, n=512
// Negligible overhead.
```

### 9.2 Where Self-Guidance Goes

If we have both final logits and intermediate logits, guidance is a single SIMD operation:

```text
// In forward_base, after computing final logits:
if config.self_guidance_weight > 0.0 {
    // logits_guided = logits + w * (logits - intermediate_logits)
    // = (1 + w) * logits - w * intermediate_logits
    for i in 0..vocab_size {
        ctx.logits[i] = (1.0 + w) * ctx.logits[i] - w * ctx.intermediate_logits[i];
    }
}
```

This is **O(vocab_size)** — negligible compared to the O(n_embd * vocab_size) LM head matmul.

---

## 10. Risk Assessment

### 10.1 MLS Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Final-layer specialization is better for well-trained models | Medium | Medium | Default K=0 (disabled), opt-in only |
| Layer sum dilutes task-specific features | Medium | Medium | Pareto sweep required per model |
| Breaks existing speculative decoding acceptance rates | Low | High | Benchmark with GOAT before enabling |
| No improvement on small models (few layers) | High | Low | Only useful for n_layer >= 6 |

### 10.2 Self-Guidance Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Intermediate LM head requires training (not random) | High | High | Only for models with exposed intermediate heads |
| Guidance weight w is task-dependent | High | Medium | Config knob with default w=0 (disabled) |
| May conflict with existing SDAR gating | Low | Medium | Mutually exclusive feature gates |
| Overfitting to specific model architecture | Medium | Medium | Architecture-agnostic implementation |

### 10.3 Overall Risk: LOW-MEDIUM

Both proposals are **additive and opt-in**. They don't change existing behavior when disabled. The main risk is wasted engineering effort if benchmarks show no improvement. The EP Accuracy@k metric is zero-risk (reporting only).

---

## 11. Feature Gate Proposal

```toml
[features]
# ... existing ...
mls_aggregate = []     # Multi-Layer Sum: aggregate last K layer residuals before LM head (Research 68)
self_guidance = []     # Self-guidance: logit contrast between final and intermediate layers (Research 68)
```

Both opt-in, off by default. Proven via GOAT benchmarks before considering default-on.

---

## 12. Verdict and Priority

### 12.1 Verdict: MODERATE TRANSFER, TWO ACTIONABLE DISTILLATIONS

RAEv2's core ideas transfer to LLM inference at the **mechanism level** (multi-layer aggregation, strong-weak guidance contrast) but NOT at the **application level** (diffusion training, spatial regularization, autoencoder). The paper validates principles we already exploit (intermediate layers carry signal, strong-weak contrast helps) and suggests two concrete additions.

| Our Design | RAEv2 Finding | Assessment |
|---|---|---|
| `MultiLayerKVCache` per-layer storage | Intermediate layers carry complementary info | ✅ Validated: we cache, but don't aggregate outputs |
| MTP drafter mid-layer projection | REPA head predicts from intermediate features | ✅ Same mechanism, different purpose |
| Early exit confidence gap | Intermediate-layer signal useful for decisions | ✅ Validated |
| SDAR gated distillation | Self-guidance via strong-weak contrast | ✅ Same pattern, training vs inference |
| Delta routing cross-layer flow | Multi-layer aggregation improves quality | ✅ We go further with selective routing |
| **Proposed:** MLS aggregation | Sum last K layers, training-free | 🟡 New, needs benchmarking |
| **Proposed:** Self-guidance | Logit contrast, no extra forward pass | 🟡 New, needs intermediate LM head |
| **Proposed:** EP Accuracy@k | Efficiency metric for GOAT | ✅ Zero-risk reporting improvement |

### 12.2 Action Items

| Item | Effort | Impact | Priority | Target |
|---|---|---|---|---|
| 7.3 EP Accuracy@k metric | Trivial | High | HIGH | GOAT benchmark reporting |
| 7.1 MLS residual aggregation | Medium | Medium | MEDIUM | `src/transformer.rs` + feature gate |
| 7.2 Self-guidance pattern | Medium | High (if it works) | LOW | Needs training pipeline first |

### 12.3 What NOT To Do

- Do NOT add autoencoder or flow matching training — we're inference-only
- Do NOT add REPA spatial regularization — text has no spatial structure
- Do NOT replace the final LM head — MLS augments, not replaces
- Do NOT enable MLS by default — well-trained models may not benefit
- Do NOT assume intermediate LM head works without training — random weights give random guidance
- Do NOT conflate RAEv2's generation quality gains with LLM inference quality — different domains

### 12.4 Cross-Reference Summary

| Research | Connection to RAEv2 |
|---|---|
| Research 26 (Gemma 4 MTP) | MTP drafter projects mid-layer activations — same mechanism as REPA head for different purpose |
| Research 28 (HLA) | Higher-order linear attention aggregates across layers — related to MLS but via attention, not summation |
| Research 34 (D2F) | Discrete diffusion forcing uses block-wise refinement — MLS could improve block-level representations |
| Research 38 (SDAR) | Sigmoid-gated distillation at training time — self-guidance is the inference-time analog |
| Research 39 (SpectralQuant) | KV cache compression preserves per-layer structure — MLS output aggregation is the complement |
| Research 49 (PTRM) | Width scaling exploits multiple trajectories — MLS exploits multiple layers (depth vs width) |
| Research 55 (Tri-Mode) | AR + diffusion + self-speculation — MLS improves the AR and self-speculation paths |
| Research 58 (GRAM) | Stochastic guidance for recursive models — self-guidance pattern is structurally similar |
| Research 61 (Delta Routing) | Cross-layer delta accumulation and routing — more sophisticated than MLS summation |
| Research 87 (CNA) | Neuron-level modulation of hidden states — MLS operates at layer level, CNA at neuron level |

---

## 13. References

1. **RAEv2** — arXiv:2605.18324 — Improved Baselines with Representation Autoencoders (Singh et al.)
2. **RAE** — Representation Autoencoders (predecessor to RAEv2)
3. **REPA** — REPresentation Alignment for diffusion models
4. **AutoGuidance** — Guidance using a smaller model (eliminated by RAEv2 self-guidance)
5. **CFG** — Classifier-Free Guidance (extra forward pass eliminated by RAEv2 self-guidance)
6. **Gemma 2** — Intermediate-layer head predictions (Research 26, Plan 055)
7. **SDAR** — Self-Distilled Agentic RL (Research 38, Plan 072)
8. **Delta Routing** — Cross-layer delta accumulation (Research 61, Plan 097)
9. **GRAM** — Generative Recursive Reasoning (Research 58)
10. **G-Zero** — Self-Play Open-Ended Generation (Research 21)

### Key File References

| File | Role |
|---|---|
| `src/transformer.rs` | `forward_base` layer loop, `standard_lm_head`, `TransformerWeights`, `LayerWeights` |
| `crates/katgpt-core/src/types.rs` | `Config` with `early_exit_*`, `mtp_*` fields, `InferenceOverrides` |
| `src/speculative/verifier.rs` | `SpeculativeVerifier` trait |
| `src/speculative/dd_tree.rs` | `inject_sde_noise`, `build_dd_tree_sde`, early exit logic |
| `src/speculative/types.rs` | `SdeConfig`, `DDTreeBranchCache`, screening types |
| `src/pruners/sdar/sdar_bandit.rs` | `SdarBanditPruner<P>` sigmoid-gated bandit |
| `src/benchmark.rs` | GOAT benchmark infrastructure, MTP benchmarks |
| `Cargo.toml` | Feature gates (`sdar_gate`, `elf_sde`, `delta_routing`, etc.) |
```

---

**Summary**: The user asked me to create a research distillation document. I wrote it following the established pattern from Research 58 (GRAM). The document covers:

1. **TL;DR** with honest verdict about vision→text transfer
2. **Paper core insights** (MLS, RAE+REPA complementarity, self-guidance)
3. **Honest applicability assessment** - what transfers, what partially transfers, what doesn't
4. **Mapping to our architecture** with specific code locations and Rust implementation sketches
5. **Component mapping table** (RAEv2 concept → our existing/proposed concept)
6. **Feature gate proposal** (`mls_aggregate`, `self_guidance`)
7. **Risk assessment** with concrete mitigations
8. **Cross-references** to related research documents
9. **References** including key file locations

The verdict is honest: MODERATE transfer. MLS and self-guidance patterns transfer cleanly as mechanisms, but the diffusion training and spatial regularization do not apply to text LLM inference.