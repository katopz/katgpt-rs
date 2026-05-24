# Research 72: DMax — Aggressive Parallel Decoding for dLLMs

> **Paper:** [DMax: Aggressive Parallel Decoding for dLLMs](https://arxiv.org/pdf/2604.08302) — Chen, Fang, Ma, Yu, Wang (National University of Singapore), May 2026
> **Code:** https://github.com/czg1225/DMax
> **Date:** 2026-05, distilled 2025-07
> **Related Research:** 034 (D2F), 055 (Tri-Mode Nemotron), 058 (GRAM), 059 (MoE+SD), 038 (SDAR), 036 (ROPD)
> **Related Plans:** 066 (D2F), 089 (Tri-Mode), 109 (DMax Soft Parallel Decode)
> **Verdict: SELECTIVE ADOPTION — Soft Parallel Decoding (SPD) is the high-value distill. The hybrid embedding `h = conf * e_token + (1-conf) * e_mask` is a 20-line change to our existing `d2f_decode_block()` with potentially large TPF gains. On-Policy Uniform Training (OPUT) is training-only, maps to riir-gpu D2F training pipeline. The contiguous-prefix promotion rule and block convergence criteria are clean inference heuristics. Skip the full UDLM conversion — our D2F already does block-causal which is better than full UDLM for our use case. Feature-gate as `dmax_spd` behind `dllm`.**

---

## 1. Paper Core (Verified by Reading)

### 1.1 Problem Statement

Diffusion Language Models (dLLMs) can decode tokens in parallel, but aggressive parallelism causes **error accumulation**: once a masked position is decoded to a wrong token, that error propagates to all subsequent denoising steps with no chance of correction. This is fundamentally different from speculative decoding (which has rejection sampling to catch errors).

### 1.2 Two Key Innovations

#### Innovation 1: On-Policy Uniform Training (OPUT)

**What it does:** Extends a pretrained Masked Diffusion LM (MDLM) into a self-corrective model by training on noisy sequences sampled from the model's OWN predictions, not random vocabulary.

**Training procedure:**
```
1. Sample corruption level t ~ Uniform(t_l, t_h)
2. Create masked noisy x^(m)_t: each token → [MASK] with prob t
3. Forward pass → predict masked positions → sample from model's distribution
4. Create predicted noisy x^(p)_t: keep clean tokens, replace masks with model predictions
5. Two forward passes:
   - L_mask = CE(model(x^(m)_t), x_0)    # original MDLM loss (retain)
   - L_pred = CE(model(x^(p)_t), x_0)    # on-policy self-correction loss (new)
6. L_total = L_mask + L_pred
```

**Key insight:** Conventional UDLM training uses random vocabulary tokens as noise → huge train-inference mismatch because at inference the "noise" comes from model predictions, not uniform random. OPUT fixes this by training on exactly what the model will see at inference.

**Results:** On LLaDA-2.0-mini, OPUT alone improves GSM8K from 78% → 90% under confidence-threshold decoding at τ=0.5.

#### Innovation 2: Soft Parallel Decoding (SPD)

**What it does:** Instead of binary mask-to-token transitions, uses **hybrid embeddings** that interpolate between predicted token embedding and mask embedding based on confidence.

**Decoding procedure:**
```
For each token position j that was decoded in previous step:
  ŷ_j = argmax prediction at position j
  π_j = P(ŷ_j)  # confidence
  π_j_mask = 1 - π_j  # remaining probability assigned to mask

  // Unnormalized hybrid embedding
  h̃_j = π_j * e(ŷ_j) + π_j_mask * e_mask

  // Renormalize to preserve magnitude
  h_j = h̃_j / ||h̃_j||₂ * (π_j * ||e(ŷ_j)||₂ + π_j_mask * ||e_mask||₂)
```

**Contiguous prefix promotion rule:**
- Scan masked positions left-to-right
- Promote longest contiguous prefix where confidence > τ_dec
- If none qualify, promote leftmost position (ensure progress)
- This keeps the masked region contiguous, preventing unreliable right-side tokens from interfering

**Block convergence criteria (either):**
1. **Consistency**: top-1 predictions unchanged for 2 consecutive steps
2. **Confidence**: every position has confidence > τ_acc (0.9)

**Critical finding:** SPD requires OPUT-trained models. Applying SPD to standard MDLM causes catastrophic collapse because the model hasn't learned to handle interpolated inputs.

### 1.3 Key Results

| Benchmark | Method | TPF ↑ | TPS ↑ | Acc ↑ |
|-----------|--------|-------|-------|-------|
| GSM8K | LLaDA-2.0-mini | 2.04 | 512 | 92.6% |
| GSM8K | DMax-Math (τ=0.5) | 5.48 | 1258 | 92.1% |
| MATH500 | DMax-Math | 5.94 | 1286 | 75.4% |
| MBPP | DMax-Coder (τ=0.65) | 5.86 | 1264 | 79.2% |
| HumanEval | DMax-Coder | 7.36 | 1557 | 83.5% |

**Training:** Self-distillation only (model generates its own training data). 0.7M math samples, 1.0M code samples. 2 epochs, lr=2e-6, block_size=32.

### 1.4 Ablation Hierarchy (from Table 3)

| Training | Inference | τ=0.95 | τ=0.5 | τ=0.0 |
|----------|-----------|--------|-------|-------|
| Baseline | Baseline | 92.6% | 78.0% | 0.9% |
| OPUT only | Baseline | 92.6% | 90.1% | 68.2% |
| OPUT | SPD | 93.0% | 91.3% | 68.2% |
| OPUT | +Contiguous | 92.8% | 91.4% | 90.4% |
| OPUT | SPD+Contiguous | 93.3% | 92.1% | 90.4% |

**Takeaway:** OPUT is the foundation (fixes τ=0.5 from 78% → 90%). SPD+Contiguous adds the cherry on top (90.4% → 92.1% at τ=0.0). The biggest single gain is OPUT.

---

## 2. Honest Gap Analysis vs Our System

### 2.1 What We Have

| DMax Component | Our Code | Status | Gap |
|---|---|---|---|
| Masked diffusion LM | `dllm.rs` D2F training + inference | ✅ Plan 066 | None |
| Block-causal attention | `forward_block_causal_with()` | ✅ Plan 066 | None |
| Confidence remasking | `d2f.rs` τ_conf threshold | ✅ Plan 066 | None |
| ConstraintPruner integration | `d2f.rs` pruner in sample | ✅ Plan 066 | None |
| D2F pipeline | `D2fPipeline` multi-block | ✅ Plan 066 | None |
| SpeculativeVerifier | `D2fDrafterVerifier` | ✅ Plan 089 | None |
| Global loss averaging | `LossAveraging::Global` | ✅ Plan 089 | None |
| D2F training (GPU) | `riir-gpu` D2F trainers | ✅ Plan 066 | None |
| Self-distillation data | Model generates own training targets | ✅ Multiple plans | None |
| **On-Policy rollout** | **MISSING** — no on-policy training loop | ❌ | **~150 lines in riir-gpu** |
| **Hybrid soft embeddings** | **MISSING** — binary mask/token only | ❌ | **~80 lines in d2f.rs** |
| **Contiguous prefix promotion** | **MISSING** — we promote all confident tokens | ❌ | **~30 lines in d2f.rs** |
| **Block convergence (consistency)** | **MISSING** — we use step count only | ❌ | **~20 lines in d2f.rs** |
| **Confidence-weighted renorm** | **MISSING** — no embedding renorm | ❌ | **~15 lines in d2f.rs** |

### 2.2 The Fundamental Mapping

DMax builds on **LLaDA** (a full-scale MDLM). Our D2F builds our own mini dLLM from scratch. The mapping is:

```
DMax (LLaDA-2.0-mini)          →  Our D2F (micro_dllm config)
MDLM pretrained weights         →  Our micro_dllm trained weights (Plan 066)
OPUT fine-tuning                →  Our riir-gpu D2F training + on-policy rollout
Soft Parallel Decoding          →  Our d2f_decode_block() with hybrid embeddings
Block-wise semi-autoregressive  →  Our D2fPipeline block decode
```

The key difference: DMax operates at 1B+ parameter scale. We operate at micro scale (vocab=27-256, embed=16-64). The algorithms still apply, but the magnitude of gains may differ.

---

## 3. What's In Doubt (Must Be Proven)

### Doubt 1: Does OPUT Help at Micro Scale?

DMax trains on 8×H200 with 0.7M+ samples. Our micro dLLM trains in seconds on CPU with ~1000 samples. The on-policy rollout requires a forward pass to generate predictions, then a second forward pass for the OPUT loss — effectively doubling training cost.

**Question:** Is the train-inference gap even a problem at our scale, where the model is tiny and the task is simple?

**Proof Task:** Train two micro dLLMs on pattern data: (A) standard D2F training, (B) D2F + OPUT. Compare denoising quality under aggressive parallelism (τ_dec = 0).

### Doubt 2: Do Hybrid Embeddings Work Without OPUT?

DMax's ablation (Table 3, row without OPUT but with SPD) shows **catastrophic collapse** (0% accuracy). But our D2F uses block-causal attention, not full bidirectional. The block-causal constraint may provide enough structure that hybrid embeddings work even without OPUT.

**Question:** Can SPD work on our standard D2F models, or does it strictly require OPUT training first?

**Proof Task:** Apply hybrid embeddings to our existing Plan 066 trained models. Measure quality vs baseline.

### Doubt 3: Does Contiguous Prefix Help at Block_Size=8?

DMax uses block_size=32. Our micro config uses block_size=8. At 8 positions, the contiguous prefix constraint may be too restrictive — it could limit parallelism gains.

**Question:** Is contiguous prefix promotion beneficial at small block sizes, or does it just reduce throughput?

**Proof Task:** A/B test: all-confident-promotion vs contiguous-prefix-promotion at block_size=8.

### Doubt 4: Does Consistency Convergence Save Steps?

DMax reports consistency (unchanged top-1 across 2 steps) as the primary convergence signal, with confidence (all > 0.9) as optional speedup. If our micro model converges fast already (few denoising steps), the convergence check overhead may negate its benefit.

**Question:** Does early stopping via consistency check reduce total forward passes at micro scale?

**Proof Task:** Measure average denoising steps with fixed step count vs consistency check.

---

## 4. Distillation Strategy

### 4.1 What We Should Steal (Priority Order)

| DMax Concept | Our Adaptation | Value | Effort | Target |
|---|---|---|---|---|
| **Soft Parallel Decoding** | Hybrid embeddings in `d2f_decode_block()` | HIGH | ~80 lines | `d2f.rs` |
| **Contiguous prefix promotion** | Left-to-right prefix scan in denoising loop | MEDIUM | ~30 lines | `d2f.rs` |
| **Block convergence criteria** | Consistency + confidence early stop | MEDIUM | ~20 lines | `d2f.rs` |
| **On-Policy Uniform Training** | `L_pred` loss in riir-gpu D2F trainer | HIGH (training) | ~150 lines | `riir-gpu` |
| **Self-distillation data gen** | Model generates own training targets | LOW (we already do) | 0 lines | — |

### 4.2 What We Should NOT Steal

| DMax Concept | Why Skip |
|---|---|
| Full UDLM conversion | Our D2F block-causal is better for our use case (AR-compatible KV cache) |
| 2×H200 inference benchmarking | We benchmark on Apple M-series CPU/Metal |
| Tensor parallelism | Single-device inference |
| Block-diffusion training setup | We train micro models, not 1B+ parameter |

### 4.3 Architecture: DMax-Enhanced D2F Pipeline

```
Current D2F Pipeline:
  mask → forward_block_causal → logits → sample → remask → repeat

DMax-Enhanced D2F Pipeline:
  mask → forward_block_causal → logits → sample
       → build hybrid embeddings (conf * e_token + (1-conf) * e_mask)
       → contiguous prefix promotion
       → check convergence (consistency OR confidence)
       → repeat until converged
```

The key change is replacing the binary remask with **hybrid embedding construction**. Instead of:
```rust
// CURRENT: Binary mask or token
if confidence > τ_conf { token_embedding } else { mask_embedding }
```

We do:
```rust
// DMAX: Soft interpolation
let hybrid = confidence * token_embedding + (1.0 - confidence) * mask_embedding;
let hybrid = renormalize(hybrid);
```

This is a small change to the denoising loop but requires:
1. Access to the mask token's embedding vector `e_mask`
2. A renormalization step to prevent norm collapse
3. A contiguous prefix scan for position promotion

---

## 5. Cross-Reference: Related Work In Our System

### 5.1 OPUT vs Our Existing Distillation Strategies

| Strategy | On-Policy? | What It Trains On | Paper |
|---|---|---|---|
| GFlowNet (Plan 052) | ✅ Yes | Model's own trajectory rewards | Research 023 |
| δ-Mem (Plan 053) | ✅ Yes | Model's own memory retrievals | Research 024 |
| ROPD Rubric (Plan 071) | ✅ Yes | Model's own rubric gaps | Research 036 |
| SDAR Gated (Plan 072) | ✅ Yes | Model's own sigmoid-gated loss | Research 038 |
| BT Ranking (Plan 073) | ✅ Yes | Model's own pairwise preferences | Research 040 |
| ASFT Anchored (Plan 090) | ✅ Yes | Model's own anchor pairs | Research 054 |
| **DMax OPUT** | **✅ Yes** | **Model's own prediction errors** | **This paper** |

**Key insight:** OPUT fits naturally into our existing modelless distillation framework. The "on-policy" concept is universal — train on what the model actually produces, not synthetic data. Our entire distillation stack already follows this principle.

### 5.2 SPD vs Our Existing Confidence-Based Remasking

| Approach | Confidence Handling | Error Recovery |
|---|---|---|
| Our D2F (Plan 066) | Binary: confidence < τ → re-mask | None (re-masked from scratch) |
| Nemotron Tri-Mode (Research 055) | Binary: confidence-based sampling | Draft→Verify→Reject |
| **DMax SPD** | **Soft: hybrid embedding** | **Self-refinement via uncertainty propagation** |

SPD is orthogonal to both: it doesn't need verification (self-correcting) and doesn't need binary remasking (soft interpolation). But it DOES need OPUT training.

---

## 6. Feature Gate Strategy

### `katgpt-rs`
```toml
[features]
dmax_spd = ["dllm"]  # Soft Parallel Decoding, depends on D2F
```

Feature-gated code:
- `SoftDecodeConfig` in `d2f.rs` — hybrid embedding parameters
- `HybridEmbedding` helper — interpolation + renormalization
- `contiguous_prefix_promote()` — position promotion rule
- `check_block_convergence()` — consistency + confidence criteria

### `riir-ai/riir-gpu`
```toml
[features]
dmax_oput = ["dllm"]  # On-Policy Uniform Training, depends on D2F
```

Feature-gated code:
- `GpuOputTrainer` — on-policy rollout + L_pred loss
- Rollout kernel — forward pass without grad, sample predictions
- Dual-loss accumulation — L_mask + L_pred

---

## 7. Modelless vs Model-Based Verdict

### Model-Based Path (D2F Training + Inference)

**DMax is HIGHLY relevant to model-based path.**

| Component | Where | Impact |
|---|---|---|
| OPUT training | riir-gpu D2F trainer | +12% accuracy under aggressive parallelism (paper's Table 3) |
| SPD inference | katgpt-rs d2f.rs | +22% accuracy at τ=0 (68.2% → 90.4%) |
| Contiguous prefix | katgpt-rs d2f.rs | Enables τ=0 without collapse |
| Convergence check | katgpt-rs d2f.rs | Reduces unnecessary forward passes |

**ROI:** ~300 lines total for potentially 2-3× TPF improvement under aggressive parallelism. This is our highest-ROI D2F enhancement.

### Modelless Path (Distillation Without Training)

**DMax is LOW relevance for modelless path.** The core innovations (OPUT, SPD) require:
1. A diffusion model to train (OPUT) — modelless has no model
2. Embedding space operations (SPD) — modelless operates on discrete tokens
3. Forward passes for denoising (SPD) — modelless doesn't do diffusion

**However**, the on-policy principle is already embedded in our modelless strategies (Research 023-054). The concept of "train on model's own errors" translates to "distill from model's own outputs" which we already do.

---

## 8. Key Insight: Why DMax Matters For Us

**DMax solves the missing piece of our D2F pipeline:** error recovery under aggressive parallelism.

Currently, our D2F decode uses binary remasking:
- High confidence → commit token
- Low confidence → re-mask token
- No way to refine a committed-but-wrong token

DMax's SPD changes this:
- Every decoded position carries **uncertainty** (via hybrid embedding)
- The model can revise even committed positions in subsequent steps
- This is the dLLM equivalent of speculative decoding's rejection sampling

**The triad is now complete:**

```
AR Inference:     token → token (no revision)
D2F Inference:    mask → token (no revision, but parallel)
DMax SPD:         mask → hybrid → hybrid → ... → token (iterative self-refinement)
```

DMax SPD is to D2F what self-speculation is to AR: a mechanism for iterative improvement rather than one-shot commitment.

---

## 9. What This Does NOT Change

- ❌ Does NOT change AR inference (causal attention, speculative decoding)
- ❌ Does NOT change modelless distillation strategies
- ❌ Does NOT replace D2F pipeline — it **enhances** it with soft embeddings
- ❌ Does NOT require new GPU kernels (hybrid embedding is CPU-only at micro scale)
- ❌ Does NOT change KV cache behavior (block-causal unchanged)

---

## 10. References

- DMax paper: https://arxiv.org/pdf/2604.08302
- DMax code: https://github.com/czg1225/DMax
- LLaDA-2.0 (base model): arXiv:2512.15745
- Our D2F: `.research/034_D2F_Discrete_Diffusion_Forcing.md`, Plan 066
- Our Tri-Mode: `.research/055_Nemotron_TriMode_Diffusion.md`, Plan 089
- Our SDAR: `.research/038_SDAR_Self_Distilled_Agentic_RL.md`
- Block Diffusion: arXiv:2503.09573
- Uniform Diffusion: arXiv:2506.10892