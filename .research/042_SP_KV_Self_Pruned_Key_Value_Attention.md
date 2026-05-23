# 14 — SP-KV: Self-Pruned Key-Value Attention Research

**Paper**: Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility
**Authors**: Szilvasy, Faysse, Lomeli, Douze, Mazaré, Cabannes, Yih, Jégou (Meta FAIR)
**Date**: May 2026 | **arXiv**: 2605.14037

---

## 1. Core Contribution

SP-KV learns **which KV pairs to write** into persistent cache by predicting their **future utility**.
Unlike post-hoc eviction (KVZap, H2O) that prune a frozen model's cache after the fact,
SP-KV trains the utility predictor **jointly** with the LLM via next-token prediction only.

**Key result**: 3–10× KV cache reduction with −0.2% average benchmark degradation (8.1B model).

---

## 2. Mechanism

### 2.1 Utility Predictor

Per key-head, per position utility prediction:

```
u_{s}^{l,k} = σ(f_θ^{l,k}(h_s^l))  ∈ (0, 1)
```

- `f_θ` = 2-layer MLP (SiLU activation)
- Input: hidden state `h_s^l` at layer `l`, position `s`
- Output: scalar utility per KV head (GQA: one gate per KV head, broadcast to query groups)
- Initialized with bias=5 → σ(5)≈0.993 (gates start nearly open)

### 2.2 Sliding Window + Gated Global Attention

```
g_{t,s} = 0         if z_s=1 OR in_window(t,s)
          -∞        otherwise
```

- **Local window** (default w=128): always available, no gating
- **Global attention**: only for positions where `z_s=1` (gated on)
- Combined mask: `B_{t,s} = M_causal(t,s) + g_{t,s}`

### 2.3 Training: Two Phases

**Phase 1 — Soft Gating** (first 75% of cosine decay):
- No thresholding; gate bias = `log(u_s)` (differentiable)
- u=1 → bias=0 (no effect), u=0 → bias=-∞ (masked out)
- Trained with vanilla next-token prediction loss only (no auxiliary loss)

**Phase 2 — Threshold-Aware Hard Gating** (last 25%):
- Freeze utility predictor weights
- Binarize gates: `z_s = 1[u_s ≥ τ]`
- Anneal over 500 steps: `ũ = (1-α)u + α·1[u≥τ]`, α ramps 0→1
- Enables block-skipping optimization in kernel

### 2.4 Inference

```
z_s = 1[u_s ≥ τ]    (binary keep/drop decision)
```

- Default threshold τ=0.5
- Sweeping τ ∈ [0.01, 0.99] controls sparsity-performance tradeoff at deployment
- No model weight changes needed — purely a threshold adjustment

---

## 3. Key Results (8.1B Model)

| Metric | Full Attention | SP-KV (τ=0.5) | Delta |
|--------|---------------|----------------|-------|
| Standard Benchmarks Avg | 0.548 | 0.546 | −0.2% |
| RULER 16k Avg | 0.750 | 0.748 | −0.3% |
| RULER 32k Avg | 0.635 | 0.610 | −3.9% |
| NIAH Single 1 (all ctx) | 1.000 | 1.000 | 0.0% |
| KV Gate Density | 100% | 33.7% | **66% sparsity** |
| Decode Speedup (bs=16) | 1.0× | 2.1–4.6× | |

**Density varies by task**:
- NIAH: 5–7% (extremely sparse — most context is irrelevant)
- GSM8k/MBPP: 18–25% (generative tasks, moderate sparsity)
- ARC/Winogrande: 40–50% (short-context, denser)
- RULER avg: 17–19%

---

## 4. Comparison with Prior Work

| Method | NLL Δ | Density | Training |
|--------|-------|---------|----------|
| StreamingLLM | +11.86% | 0% | Post-hoc |
| H2O | +3.26% | 20.45% | Post-hoc |
| ExpectedAttention | +3.95% | 20.70% | Post-hoc |
| KVZap (+4 sinks) | +1.23% | 20.15% | Post-hoc |
| **SP-KV (τ=0.5)** | **+0.08%** | **25.72%** | **Joint** |
| **SP-KV (τ=0.7)** | **+0.46%** | **11.44%** | **Joint** |

SP-KV dominates because the model **adapts its representations** to sparsity during training,
eliminating the train-test mismatch that plagues post-hoc methods.

---

## 5. Mapping to Our System

### 5.1 Existing Components

| SP-KV Concept | Our Equivalent | Status |
|---------------|---------------|--------|
| Utility Predictor (MLP) | `PFlash::AttentionScorer` | Prefill-only, not decode-time |
| Sliding window (w=128) | `FlashPrefillConfig` window+sinks | Prefill-only |
| Soft gate = log(u) as attention bias | `ScreeningPruner` blends R∈[0,1] into log-prob | Operates on tokens, not KV entries |
| Hard gate = threshold τ | TurboQuant (quantizes uniformly) | No selective eviction |
| Per-head sparsity patterns | Not present | Gap |
| Joint training (model + predictor) | δ-Mem trains associative memory jointly | For pruner relevance, not KV utility |

### 5.2 Attention Kernel Insertion Point

Our `attention_head()` in `transformer.rs` (L340-395):

```
// Current: score = dot(q, k) * scale
// SP-KV:   score = dot(q, k) * scale + gate_bias[s]
// where gate_bias = log(u_s) for training, 0 or -inf for inference
```

This is **one additive bias** added during Pass 1 (Q·K scoring). The rest of the softmax
and weighted accumulation remains unchanged.

### 5.3 Cache Write Insertion Point

Our `forward_base()` in `transformer.rs` (L617-626):

```rust
// Current: unconditional write
ptr::copy_nonoverlapping(ctx.k.as_ptr(), layer_cache.key.as_mut_ptr().add(pos_off), kvd);
ptr::copy_nonoverlapping(ctx.v.as_ptr(), layer_cache.value.as_mut_ptr().add(pos_off), kvd);

// SP-KV: conditional write based on utility
let utility = predictor.predict(&ctx.x, layer_idx, kv_head); // ∈ (0,1)
if utility >= tau || (pos as isize - t_n as isize).unsigned_abs() < window_size {
    // write to persistent cache
} else {
    // skip write — position stays zeroed
}
```

### 5.4 Where It Lives

```
microgpt-rs/src/
├── sp_kv/                          # NEW module
│   ├── mod.rs                      # Module index + re-exports
│   ├── types.rs                    # SpKvConfig, SpKvCache, UtilityPredictorWeights
│   ├── utility_predictor.rs        # 2-layer MLP: h → u ∈ (0,1) per KV head
│   ├── cache.rs                    # Sparse-write KV cache (gated append)
│   └── forward.rs                  # forward_sp_kv() with gate bias
├── transformer.rs                  # Add SpKvCache variant + forward_sp_kv dispatch
└── turboquant/                     # Orthogonal: quantize what SP-KV keeps
```

### 5.5 Composability with Existing Mechanisms

| Combination | Value | Notes |
|-------------|-------|-------|
| SP-KV + TurboQuant | High | SP-KV selects which KV to keep, TQ compresses what's kept |
| SP-KV + HLA | Medium | HLA layers can be the "local" in hybrid; SP-KV handles global |
| SP-KV + Raven | Low | Both address KV growth; Raven is O(1) slots, SP-KV is selective standard KV |
| SP-KV + PFlash | High | PFlash at prefill, SP-KV during decode — end-to-end sparse pipeline |
| SP-KV + Sparse MLP | Orthogonal | Different axes of sparsity (attention vs FFN) |

---

## 6. Training Protocol

### 6.1 Continual Pretraining (Paper's Method)

1. Train full-attention model with 140 tokens-per-parameter (TPP)
2. Branch: switch attention to SP-KV for additional 20 TPP with cosine decay
3. Phase 1 (0–75% of decay): soft gating with log(u) bias
4. Phase 2 (75–100% of decay): freeze predictor, binarize with annealing

### 6.2 Adapted for Our System

Our training infra is simpler (no distributed training), but the protocol maps directly:

1. Load pretrained weights (or train from scratch — also works, see Appendix C.5)
2. Add `UtilityPredictorWeights` per layer: 2-layer MLP, ~`2 × n_embd × hidden` params
3. Initialize gates open (bias=5)
4. Train with soft gating for most of the schedule
5. Optionally: TAHG phase for the last portion

### 6.3 From-Scratch Training

Paper Appendix C.5: training from scratch works and yields **higher sparsity** (15.8% vs 25.4%)
but slightly worse NLL (+0.1%). This is relevant for our micro configs.

---

## 7. Design Decisions & Ablations (From Paper)

### 7.1 What Works Best

| Design Choice | Best Setting | Ablation |
|---------------|-------------|----------|
| Predictor depth | 2-layer MLP | Linear predictor: 31% more density for same quality |
| Initial bias | 5.0 (σ≈0.993) | Bias=20: 67% more density (too open), Bias=1: −2.7% density |
| LR multiplier | 5× global LR | 1×: more density, 0.1×: 82% density (barely sparsifies) |
| Local window | 128 | w=1: +139% density, w=512: +8.3% density, −0.3% NLL |
| Auxiliary loss | Not needed | Optional density regularizer for fine control |
| Weight decay | 0.1 on predictor | No decay: +1.1% density, similar quality |

### 7.2 Frozen LLM Ablation

When freezing LLM and only training the predictor: **81.8% density** (barely sparsifies).
This confirms joint training is essential — the model must adapt representations to sparsity.

### 7.3 Hybrid Architecture Discovery

SP-KV density patterns reveal which heads/layers need global attention:
- Strategy D (18 densest heads by SP-KV utility): 68.44% coverage, +0.161% NLL
- Strategy A (3:1 CWM pattern): 8.28% coverage, +0.396% NLL
- Full SP-KV: 100% coverage, +0.074% NLL

**Implication for us**: SP-KV can guide which layers get HLA (local) vs SDPA (global).

---

## 8. Reference Implementation (PyTorch)

From paper Appendix I, adapted to Rust pseudocode:

```rust
struct UtilityPredictor {
    w1: Vec<f32>,  // [n_embd, hidden]
    w2: Vec<f32>,  // [hidden, n_kv_heads]
}

impl UtilityPredictor {
    fn forward(&self, h: &[f32]) -> Vec<f32> {
        // h: [n_embd] → hidden → n_kv_heads → sigmoid
        let hidden = matmul_silu(&self.w1, h);  // SiLU activation
        let logits = matmul(&self.w2, &hidden);
        sigmoid(&logits)  // u ∈ (0, 1)
    }
}

fn sp_kv_attention(q, k, v, utility, window_size, hard, tau) {
    let gate = if hard {
        // Inference: binary threshold
        utility.map(|u| if u >= tau { 0.0 } else { f32::NEG_INFINITY })
    } else {
        // Training: soft gate (log u)
        utility.map(|u| (u + 1e-8).ln())
    };

    // Build mask: causal + window + gate
    for t in 0..T {
        for s in 0..=t {
            let in_window = t - s < window_size;
            let bias = if in_window { 0.0 }
                       else if causal { gate[s] }
                       else { f32::NEG_INFINITY };
            scores[t][s] = dot(q[t], k[s]) * scale + bias;
        }
    }
    // Standard softmax + weighted sum
}
```

---

## 9. Novel Distillations for Our System

### 9.1 Sparse-Write Pipeline: PFlash → SP-KV → TurboQuant

```
Prefill:  PFlash compresses prompt (block-sparse selection)
Decode:   SP-KV selectively writes new KV pairs (utility gating)
Storage:  TurboQuant quantizes retained pairs (2-4 bits/coord)
```

End-to-end: only useful KV pairs are kept, and those are compressed.
This is a 3-stage pipeline that addresses KV growth at every phase.

### 9.2 NAS-Guided HLA/SDPA Hybrid Layout

Use SP-KV density patterns to decide:
- Heads with <10% average density → assign to HLA (local-only, O(1) memory)
- Heads with >40% average density → assign to SDPA (global attention)
- Middle heads → SP-KV gated SDPA

This gives a data-driven hybrid architecture instead of fixed interleaving.

### 9.3 Inference-Time Budget Control via τ

The threshold τ is a **deployment knob**:
- Memory-constrained: τ=0.7 (11% density, 9× compression)
- Quality-critical: τ=0.3 (50% density, 2× compression)
- Default: τ=0.5 (25% density, 4× compression)

No retraining needed — just change a config value.

---

## 10. Risks & Limitations

| Risk | Mitigation |
|------|-----------|
| Training instability with gates closing too fast | Initialize bias=5, LR multiplier=5, Bernoulli clipping |
| Long-context tasks (32k+) degrade more | Add RULER-style data to training mix (Table 6: −0.3% with data) |
| GPU kernel complexity for sparse decode | Block-skipping at 64-token granularity (paper's approach) |
| Our models are too small for learned sparsity | From-scratch training works at 1B scale with higher sparsity |
| Interaction with existing mechanisms | Feature-gate behind `sp_kv` flag, independent module |

---

## 11. References

- SP-KV: arXiv:2605.14037
- H2O: Zhang et al. (2023) — Heavy-hitter eviction
- KVZap: Jegou & Jeblick (2026) — Fast adaptive KV pruning
- StreamingLLM: Xiao et al. (2024b) — Attention sinks
- DMS: Łańcucki et al. (2025) — Retrofit eviction with distillation
- FlashAttention-3: Shah et al. (2024) — Fused attention kernel
- Our TurboQuant: arXiv:2504.19874 — KV quantization
- Our PFlash: Plan 048 — Block-sparse speculative prefill