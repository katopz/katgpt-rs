# Research: Contrastive Neuron Attribution — Sparse MLP Circuit Discovery (53)

> Source: [Targeted Neuron Modulation via Contrastive Pair Search](https://arxiv.org/pdf/2605.12290) by Sam Herring, Jake Naviasky, Karan Malhotra (Nous Research), arXiv:2605.12290, May 2026
> Date: 2026-05 (paper), distilled 2025-07
> **Verdict: ADOPT — CNA maps directly to existing trait stack as a model-based ScreeningPruner. Feature gate `cna_steering` needed for forward-hook MLP activation capture. katgpt-rs is the correct home (model-level code, not riir-ai GPU training).**
> **Split verdict (synthesis):** Wei et al. [arXiv:2607.09185 "Causally Debiased Latent Action Model for Embodied Action Conditioned World Models"] — CD-LAM's action-centric contrastive loss (softplus on same-primitive/different-primitive pairs over a 12-way verb codebook) is a TRAINING-TIME organization of the latent space; CNA (here) is the modelless post-hoc DISCOVERY of contrastive structure. Different operations (training reshapes vs discovery identifies). **Training recipe → riir-train.** BUT CD-LAM's diagnostic framework (§III-B: three forward-pass purity checks) is modelless → **GAIN** (Research 460, Issue 194 — `LatentConfounderAudit`). CNA remains the closest modelless cousin for the contrastive *discovery* operation; CD-LAM's diagnostics are a complementary *purity audit* tool.

## TL;DR

Contrastive Neuron Attribution (CNA) identifies the **0.1% of MLP neurons** (post-ReLU hidden units) whose activations most distinguish contrastive prompt sets, using only forward passes — no gradients, no auxiliary training. Ablating these neurons reduces refusal rates by >50% while maintaining output quality >0.97 at all steering strengths. The key finding: alignment fine-tuning transforms pre-existing late-layer discrimination structure into a sparse, targetable behavioral gate.

**Why it matters for us**: Our `ScreeningPruner::relevance()` already accepts model-based signals. CNA is a new relevance scorer that captures per-neuron MLP activation differences. Our game domains (Bomber, Go, FFT, Monopoly) already have natural contrastive pairs (good/bad moves, winning/losing positions). The discovery phase is a ScreeningPruner; the ablation/modulation phase is a runtime hook on `ctx.hidden` in `forward_base`.

---

## Paper Core Contributions

### 1. Contrastive Neuron Attribution (CNA) Method

For each behavior, define positive (exhibits target) and negative (does not) prompt sets:

```
δℓj = (1/|P+|) Σ aℓj(x) − (1/|P−|) Σ aℓj(x)   (1)
```

Where `aℓj(x)` = activation of neuron j in layer ℓ on prompt x (post-ReLU, at last token position).

Select circuit `Ck = top-k{|δℓj|}` with k = 0.1% of total MLP activations.

**Three steps**:
1. Run all prompts through model, record MLP activations via forward hooks on `down_proj`
2. Compute per-neuron mean activation difference between positive and negative sets
3. Select top 0.1% neurons by absolute difference

### 2. Universal Neuron Filtering

Some neurons fire regardless of prompt content. Detect by running diverse prompts, flagging any neuron in top 0.1% for ≥80% of prompts. Exclude these from all discovered circuits.

### 3. Targeted Ablation

Multiply each circuit neuron's activation by scalar `m` at inference time:
- `m = 0`: ablate (zero out)
- `m = 1`: baseline (no change)
- `m > 1`: amplify

### Key Results

| Metric | CNA | CAA (residual-stream) |
|--------|-----|----------------------|
| Refusal reduction | >50% | comparable at moderate α |
| Output quality at max steering | >0.97 | <0.60 for 6/8 models |
| MMLU preservation | within 1 point | drops to near-zero |
| Neurons needed | 0.1% of MLP | full residual stream |

---

## Architecture Mapping: CNA → Our Trait Stack

### Phase 1: Discovery → ScreeningPruner

CNA's discovery phase is a **model-based ScreeningPruner** that computes relevance from MLP activation differences:

```text
Our Stack                          CNA Equivalent
─────────                          ──────────────
ScreeningPruner::relevance()  ←→   |δℓj| for discovered circuit
ConstraintPruner::is_valid()  ←→   Universal Neuron Filtering (≥80% flagging)
BanditPruner<P> Q-values      ←→   Online refinement of circuit weights
```

The discovery phase runs **before** inference as a calibration step. It produces a `CircuitMap` (layer → neuron indices → δ values) that the ScreeningPruner consults at inference time.

### Phase 2: Ablation → Forward Hook on ctx.hidden

In our `forward_base()` at `transformer.rs`, the MLP path is:

```text
ctx.xr2 = ctx.x                    // save residual
rmsnorm(ctx.x)                     // pre-MLP norm
matmul_relu(ctx.hidden, w1, x)     // ← CNA captures activations HERE (post-ReLU)
matmul(ctx.x, w2, ctx.hidden)      // down projection
simd_add(ctx.x, ctx.xr2)          // residual add
```

CNA ablation inserts between `matmul_relu` and `matmul(w2)`:

```text
matmul_relu(ctx.hidden, w1, x)
// CNA HOOK: for each (layer, neuron) in circuit: ctx.hidden[neuron] *= m
cna_modulate(ctx.hidden, layer_idx, &circuit)  // new
matmul(ctx.x, w2, ctx.hidden)
```

This is **zero-cost when disabled** (feature gate compiles out) and **O(k) when enabled** where k = 0.1% of mlp_hidden.

### Model-Based vs Modelless Spectrum

CNA fits naturally into our existing duality (Research 37):

| Our Component | Type | CNA Analog |
|---------------|------|------------|
| `ConstraintPruner` | Modelless | Universal Neuron Filtering |
| `NoScreeningPruner` (relevance=1.0) | Modelless | No circuit discovery |
| `BanditPruner<P>` | Modelless→model-based | Online circuit refinement |
| `DeltaBanditPruner` | Model-based bridge | δ signal from log-probs |
| **`CnaScreeningPruner`** (NEW) | **Model-based** | **Contrastive discovery** |
| **`cna_modulate()`** (NEW) | **Model-based** | **Targeted ablation** |

---

## Game Domain Contrastive Pairs

Our game domains provide natural contrastive pairs for CNA discovery — no need for harmful/benign prompt sets:

### Bomberman
- **Positive**: moves that place bomb near opponent, safe moves
- **Negative**: moves that walk into blast radius, stuck moves

### Go
- **Positive**: moves with high GoHeuristic score, tenuki at right time
- **Negative**: moves with low GoHeuristic score, self-atari

### FFT Tactics
- **Positive**: actions that kill enemy unit, heal low-HP ally
- **Negative**: actions that waste turn, heal full-HP unit

### Monopoly
- **Positive**: buy property when cash-rich, trade that completes set
- **Negative**: buy property when cash-poor, trade that breaks set

Each domain already has `StateHeuristic` implementations. CNA discovery would run forward passes on episodes tagged with high vs low heuristic scores, then identify the MLP neurons that distinguish them.

---

## Implementation Sketch

### Types (cna_types.rs)

```rust
/// A discovered neuron circuit from contrastive pair analysis.
/// Maps (layer_idx, neuron_idx) → δ value (mean activation difference).
pub struct CnaCircuit {
    /// Sparse set of (layer, neuron, delta) entries.
    /// Sorted by |delta| descending. Top 0.1% of total MLP activations.
    pub neurons: Vec<CnaNeuron>,

    /// Universal neurons filtered out (fired ≥80% across diverse prompts).
    pub universal_excluded: Vec<(usize, usize)>,

    /// Discovery metadata.
    pub n_positive: usize,
    pub n_negative: usize,
    pub total_mlp_activations: usize,  // n_layer * mlp_hidden
}

pub struct CnaNeuron {
    pub layer: usize,
    pub index: usize,     // index into ctx.hidden (post-ReLU)
    pub delta: f32,       // mean activation difference
}

/// Runtime modulation state.
pub struct CnaModulator {
    pub circuit: CnaCircuit,
    /// Per-neuron multiplier (0.0 = ablate, 1.0 = baseline, >1.0 = amplify).
    pub multiplier: f32,
}
```

### ScreeningPruner Implementation

```rust
pub struct CnaScreeningPruner {
    circuit: CnaCircuit,
}

impl ScreeningPruner for CnaScreeningPruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        // Relevance is derived from circuit membership —
        // if the current token would activate circuit neurons, relevance is high.
        // This is used during DDTree search to prefer tokens that engage the circuit.
        1.0 // placeholder — real impl checks activation overlap
    }
}
```

### Forward Hook

```rust
/// Modulate post-ReLU MLP activations according to discovered circuit.
/// Insert between matmul_relu and matmul(w2) in forward_base().
#[cfg(feature = "cna_steering")]
#[inline]
pub fn cna_modulate(hidden: &mut [f32], layer_idx: usize, modulator: &CnaModulator) {
    if modulator.multiplier == 1.0 {
        return; // baseline — no-op
    }
    for neuron in &modulator.circuit.neurons {
        if neuron.layer == layer_idx {
            hidden[neuron.index] *= modulator.multiplier;
        }
    }
}
```

### Feature Gate

```toml
cna_steering = ["bandit"]  # Contrastive Neuron Attribution — sparse MLP circuit discovery + modulation
```

Requires `bandit` because discovery uses the existing heuristic infrastructure (StateHeuristic, trial logging).

---

## Late-Layer Concentration

The paper finds discrimination neurons concentrate in the **final ~10% of layers**:

| Model | Top 3 Layers | Top ¼ Layers |
|-------|-------------|--------------|
| Llama-1B | 85-87% | 88-90% |
| Qwen-3B | 58-72% | 95-100% |

This means for our models:
- `Config::micro()` (n_layer=6): top 1 layer ≈ 85% of circuit
- `Config::gqa_draft()` (n_layer=12): top 2 layers ≈ 85% of circuit
- Production models (n_layer=32+): top 3-4 layers ≈ 85% of circuit

**Optimization**: Only capture activations from the last `ceil(n_layer * 0.15)` layers during discovery. Skip early layers entirely.

---

## Verdict: Why Adopt

1. **Trait alignment**: CNA discovery is a ScreeningPruner. CNA modulation is a forward hook. Both patterns already exist in our codebase.

2. **Game domain fit**: Our arenas already produce labeled episodes (win/loss, heuristic scores). These are natural contrastive pairs — no manual labeling needed.

3. **Sparse and cheap**: 0.1% of MLP activations is ~10-50 neurons for our model sizes. Modulation is O(k) per layer, negligible.

4. **Quality preservation**: Unlike CAA (residual-stream), CNA maintains output quality at all steering strengths. Our `sparse_mlp` feature already operates on the same activation tensor (`ctx.hidden`).

5. **Feature-gated**: All new code behind `cna_steering`. Zero cost when disabled.

6. **Cross-domain**: Same infrastructure works for refusal steering (LLM safety), game strategy steering (Bomber, Go), and any behavior that admits clean contrastive pairs.

### What NOT to do

- **Don't** implement SAE (Sparse Autoencoder) — the paper explicitly avoids this cost
- **Don't** implement RelP (Layer-wise Relevance Propagation) — requires linearization, not needed for contrastive approach
- **Don't** modify the residual stream — paper shows CAA degrades quality; CNA operates on individual neurons only
- **Don't** create a new trait — `ScreeningPruner` + `ConstraintPruner` already cover the interface

---

## riir-ai Impact

CNA is primarily a **katgpt-rs** feature (model-level forward pass hooks). riir-ai's role is:

1. **GPU kernel** (riir-gpu): If we want CNA on GPU models, the WGSL kernel needs a `modulate` pass on hidden activations. Trivial — one loop over sparse indices.

2. **Validator SDK** (riir-validator-sdk): A CNA-aware validator could check if model output was generated under circuit modulation. Useful for safety auditing.

3. **Prompt Router** (riir-router): Domain-specific circuits could be loaded per route. Already has `ExpertBundle` pattern.

No new riir-ai plan needed — these are extension points in existing plans.

---

## Relationship to Existing Research

| Research | Connection |
|----------|------------|
| 07_Screening_Absolute_Relevance | CNA is a model-based ScreeningPruner (contrastive activation scoring) |
| 08_Sakana_TwELL_Sparse_MLP | Both target MLP sparsity; CNA discovers which neurons, TwELL prunes weights |
| 21_G-Zero_Self-Play | G-Zero's Hint-δ is log-prob signal; CNA's δ is MLP activation signal |
| 37_REAP_Model-Based_Modelless_Duality | CNA is a new point on the model-based spectrum |
| 42_SP_KV_Self_Pruned_Key_Value_Attention | Both use forward-pass observations for runtime decisions |

---

## References

- Paper: https://arxiv.org/pdf/2605.12290
- Authors: Sam Herring, Jake Naviasky, Karan Malhotra (Nous Research)
- Code (planned): https://github.com/NousResearch/neural-steering
- Related: CAA (Contrastive Activation Addition), RelP (Layer-wise Relevance Propagation), SAE (Sparse Autoencoders)
- Our related plans: 021_screening_pruner, 022_sparse_mlp_twell, 049_g_zero_self_play, 086_simpletes_evaluation_driven_scaling

> **PASS-Redirects (synthesis):** Fermion Research "Mapping every neuron in Qwen3-8B" (fermionresearch.com, 2026-11-16/18) — the interventional push census + matched-random control battery is the causal-twin confirmation step this note's contrastive screen lacks; see katgpt-rs [Research 684](684_Fermion_Neuron_Census_and_Targeted_Edits.md) + riir-infer Issue 1241.