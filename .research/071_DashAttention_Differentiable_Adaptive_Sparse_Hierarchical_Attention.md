# Research 68: DashAttention — Differentiable and Adaptive Sparse Hierarchical Attention

> **Paper:** [DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention](https://arxiv.org/pdf/2605.18753) — Huang, Gonçalves, Alvetreti, Li, Han, Ponti, Martins, Treviso (Tsinghua, IST Lisbon, CMU, etc.), May 2026
> **Code:** https://github.com/fasa-org/dash-attention
> **Date:** 2026-05, distilled 2025-07
> **Related Research:** 22 (Lighthouse Attention), 42 (SP-KV), 63 (OCTOPUS), 28 (HLA), 061 (Delta Attention Residuals)
> **Related Plans:** 104 (DashAttention Adaptive Sparse Attention)
> **Verdict: SELECTIVE ADOPTION — The α-entmax router + learned chunk summarization (Stage 0) distills cleanly into our SP-KV + PFlash pipeline. Key takeaway: replace top-k block selection with adaptive sparsity, add learned chunk summaries. GPU kernel work goes to riir-ai. Feature-gate as `dash_attn`.**

---

## TL;DR

DashAttention replaces fixed-budget top-k block routing (NSA, InfLLMv2) with **α-entmax** — an adaptively sparse transformation that selects a variable number of KV chunks per query. The entire hierarchy remains **end-to-end differentiable** via a prior-induced softmax (Stage 2). A 3-stage pipeline: (0) learned local chunk summarization, (1) α-entmax block routing, (2) prior-induced sparse softmax. Results: matches full attention at 75% sparsity, 3.36× speedup over FlashAttention-3 at 96K context, better Pareto frontier than NSA/InfLLMv2.

**For our stack:** The core algorithmic insight (adaptive sparsity via α-entmax instead of top-k) applies directly to our PFlash block selector and SP-KV utility predictor. The learned chunk summarization (Stage 0) replaces mean-pooling in our existing block scoring. The GPU kernel design goes to riir-ai. We do NOT need the full 3-stage pipeline for inference — we can adopt the α-entmax router as a drop-in replacement for top-k in our existing sparse attention path.

---

## Core Algorithm (3-Stage Pipeline)

### Stage 0: Local Chunk Summarization

Instead of mean pooling (MoBA, InfLLMv2) or MLP (NSA), DashAttention uses **learned local attention**:

```
k̄_c^(r) = Σ_{t∈C_c} softmax(⟨q̄^(r), k_t^(r)⟩ / √d_h) · k_t^(r)
```

- `q̄ ∈ R^{h_kv × d_h}` — learned per-head summary query, **initialized at zero**
- At init: softmax(0) = uniform → equivalent to mean pooling (smooth transition from pretrained)
- During training: learns to attend to informative keys within each chunk
- Chunk summaries are **cached** once computed — no recomputation at decode time
- Per KV-head `r` (GQA-aware)

**Key insight for us:** Our PFlash `BlockAttentionScorer` uses mean-K dot-product scoring. Adding a learnable summary query `q̄` is a ~`n_kv_head × head_dim` parameter addition that makes summarization expressive.

### Stage 1: α-entmax Block Routing

```
ŵ_i^(h) = α-entmax(γ · z̄_i^(h))  ∈ △_{⌊n/B⌋}
```

Where `z̄_i^(h) = ⟨q_i^(h), k̄_c^(r(h))⟩ / √d_h` are chunk-level logits.

Properties:
- **Adaptive sparsity**: the number of active chunks depends on the query, NOT a fixed k
- **α controls sparsity**: α=1 → softmax (dense), α=2 → sparsemax, α=1.5 is practical default
- **Fully differentiable**: unlike top-k, gradients flow through the routing weights
- **GQA-aware**: average entmax probabilities across query heads in same group before thresholding
- **Support**: `Ŝ_i = {c | w_{i,c} > 0}` determines which chunks to attend

**The α-entmax function:**
```
α-entmax(s)_i = [(α-1)·s_i - τ]_{+}^{1/(α-1)}
```
Where τ is a normalizing constant. Coordinates with `(α-1)·s_i ≤ τ` become exactly zero.

For α=1.5 (practical choice): reduces to quadratic operations, no exponentials/logarithms.

### Stage 2: Prior-Induced Sparse Softmax

The chunk-level entmax weights serve as a **prior** for token-level softmax:

```
g_σ(w_i)_j = λ_i · w'_{i,j}     if j ∈ R_i (routed)
           = (1-λ_i)/|D_i|       if j ∈ D_i (diagonal)
           = 0                    otherwise
```

The final attention is equivalent to adding a bias `d_{i,j}` to attention logits:

```
d_{i,j} = (log w_{i,c(j)} - μ_i) / σ   if j ∈ R_i
d_{i,j} = 0                              if j ∈ D_i

o_i = Σ_{j∈R_i∪D_i} exp(z_{i,j} + d_{i,j}) · v_j / Σ exp(...)
```

**Key insight:** This is just softmax attention with an additive routing bias — compatible with FlashAttention kernels! The routing bias `d_{i,j}` encodes the entmax prior.

### σ Controls Prior Strength

- σ → ∞: prior becomes uniform over routed support → standard softmax on selected chunks
- σ small: prior strongly biases attention toward highly-scored chunks
- Paper uses σ=10^6 (1B/3B) and σ=10^8 (8B) — effectively weakens the prior

---

## Key Theoretical Result: Non-Dispersion

**Theorem (from paper):** Softmax head aggregation in GQA is dispersive (entropy grows as log n). Entmax head aggregation is NOT dispersive when each head's support is O(n^β) for β < 1.

**Implication:** NSA and InfLLMv2 use softmax head aggregation before top-k → dispersion re-enters through aggregation. DashAttention uses entmax aggregation → preserves sparsity guarantee.

---

## Key Experimental Results

### RULER-16K (8B model, 75% sparsity)
| Method | SG1-3 | MK1-3 | MV-MQ | VT-FWE | QA1-2 | Avg |
|--------|-------|-------|-------|--------|-------|-----|
| FullAttn | 99.3 | 98.0 | 99.8 | 54.6 | 65.0 | 85.3 |
| NSA | 83.3 | 20.7 | 74.0 | 36.4 | 35.0 | 55.0 |
| InfLLMv2 | 100.0 | 78.0 | 97.0 | 49.0 | 62.0 | 78.9 |
| **DashAttention** | **100.0** | **94.0** | **99.5** | **54.8** | **63.0** | **83.6** |

DashAttention matches full attention at 75% sparsity on most tasks, dominates NSA and InfLLMv2 especially on multi-key retrieval (MK1-MK3).

### Speedup over FlashAttention-3 (decoding, 96K context)
| Sparsity | NSA | InfLLMv2 | DashAttention |
|----------|-----|----------|---------------|
| 75% | 0.80× | 1.73× | **1.96×** |
| 87.5% | 1.04× | 2.38× | **2.72×** |
| 93.75% | 1.34× | 3.10× | **3.36×** |

### Dynamic Sparsity Behavior
- Early layers: denser (need broad context)
- Middle layers: sparser (specialized)
- Late layers: moderate
- Automatically produces layer-wise budget allocation without explicit configuration

---

## Mapping to Our Stack

### What We Already Have

| DashAttention Concept | Our Implementation | Status |
|---|---|---|
| Chunk summarization (Stage 0) | PFlash `BlockAttentionScorer` uses mean-K | Mean pooling only, no learned query |
| Block scoring → selection (Stage 1) | `block_select()` with heuristic rules (sink+window+α) | Top-k style, fixed budget |
| Sparse attention on selected blocks (Stage 2) | `compress_prompt_blocks()` → target prefill | Works for prefill only |
| Per-head sparsity patterns | SP-KV `UtilityPredictor` per KV head | Decode-time utility, not routing |
| Additive attention bias | SP-KV `gate_bias` (log(u)) | Same pattern! SP-KV adds log(u), DashAttn adds d_{i,j} |
| Chunk cache | PFlash block-level scoring cache | Exists but mean-pooled, not learned |
| GQA awareness | Full GQA support in Config + forward | ✅ |

### What's New from DashAttention

| Innovation | Our Gap | Opportunity |
|---|---|---|
| **α-entmax adaptive routing** | We use fixed-budget top-k in PFlash | Replace `block_select()` top-k with α-entmax |
| **Learned chunk summary query q̄** | We use mean-K pooling | Add `head_cls` vector per KV head (paper calls it `head_cls`, ~`n_kv_head × head_dim` params) |
| **End-to-end differentiable routing** | PFlash selection is non-differentiable | For training in riir-ai, differentiable routing improves quality |
| **Prior-induced softmax bias** | SP-KV has gate bias, but no routing-to-attention prior | Compose: SP-KV utility × entmax routing → attention bias |
| **Layer-wise dynamic sparsity** | PFlash has fixed sparsity budget | Entmax naturally allocates more to early/dense layers |
| **Non-dispersive head aggregation** | GQA softmax aggregation | Entmax aggregation preserves sparsity in GQA groups |

---

## Distillation to Our Architecture

### D1: α-entmax Router — Replaces top-k in PFlash

```rust
/// α-entmax transformation (α=1.5 special case: quadratic, no exp/log)
fn entmax_1p5(scores: &[f32], threshold: &mut f32) -> Vec<f32> {
    // For α=1.5: p_i = max(0, 0.5*s_i - τ)^2
    // τ found by bisection such that Σ p_i = 1
    // Returns sparse routing weights (many exactly zero)
}

/// Adaptive block selection — replaces block_select() top-k
fn block_select_entmax(
    scores: &[f32],        // per-chunk scores
    alpha: f32,            // 1.5 default
    scaling_factor: f32,   // γ, controls sparsity level
) -> Vec<usize> {          // active chunk indices (variable count!)
    let scaled: Vec<f32> = scores.iter().map(|s| s * scaling_factor).collect();
    let mut tau = 0.0f32;
    let weights = entmax_1p5(&scaled, &mut tau);
    // Return indices where weight > 0
    weights.iter().enumerate()
        .filter(|(_, &w)| w > 0.0)
        .map(|(i, _)| i)
        .collect()
}
```

### D2: Learned Chunk Summary — Enhances PFlash Scorer

```rust
/// Per-head learned summary query (Stage 0)
/// Initialized at zero → starts as mean pooling
/// Trained to attend to informative keys within chunk
struct ChunkSummaryQuery {
    /// [n_kv_head × head_dim] — one query per KV head
    head_cls: Vec<f32>,
}

impl ChunkSummaryQuery {
    fn summarize_chunk(&self, keys: &[f32], kv_head: usize, head_dim: usize) -> Vec<f32> {
        // Local SDPA: softmax(q̄ · K_chunk / √d) · K_chunk
        // At init (q̄=0): softmax(0) = uniform → mean pooling
        // After training: weighted attention to informative keys
    }
}
```

### D3: Entmax Routing Bias — Enhances SP-KV Gate

```rust
/// Combined routing: SP-KV utility × entmax routing
fn compute_routing_bias(
    entmax_weights: &[f32],  // from Stage 1
    sp_kv_utility: &[f32],   // from utility predictor
    mu: f32,                  // mean log weight
    sigma: f32,               // prior strength (large = weak prior)
) -> Vec<f32> {
    // d_j = (log w_j - μ) / σ  for routed positions
    // d_j = 0                  for diagonal/window positions
    entmax_weights.iter().enumerate().map(|(j, &w)| {
        if w > 0.0 {
            (w.ln() - mu) / sigma
        } else {
            0.0  // not routed — handled by sparse mask
        }
    }).collect()
}
```

### D4: Non-dispersive GQA Aggregation

```rust
/// Entmax-based head aggregation for GQA groups
/// Replaces softmax aggregation that re-introduces dispersion
fn entmax_gqa_aggregate(
    per_head_scores: &[Vec<f32>],  // [n_query_heads][n_chunks]
    gqa_group_size: usize,
) -> Vec<f32> {
    // Average entmax probabilities (not logits) across query heads in same group
    // Remains sparse — entmax zeros propagate through averaging
    // Paper: w_i^(r) = Σ_{h∈G_r} ŵ_i^(h) / g_q
}
```

---

## Architecture Integration

### Where It Lives (katgpt-rs)

```
katgpt-rs/src/
├── sp_kv/                    # EXISTING — utility predictor, gated cache
├── speculative/
│   └── prefill.rs            # EXISTING — PFlash block scoring
└── dash_attn/                # NEW module
    ├── mod.rs                # Module index + re-exports
    ├── types.rs              # DashAttnConfig, ChunkSummaryQuery, EntmaxRouter
    ├── entmax.rs             # α-entmax kernel (α=1.5 special case)
    ├── chunk_summary.rs      # Learned local-attention summarization
    ├── routing.rs            # Entmax block routing + GQA aggregation
    └── forward.rs            # forward_dash_attn() — integrates with PFlash + SP-KV
```

### Where It Lives (riir-ai)

```
riir-ai/crates/riir-gpu/src/
└── dash_attn/                # NEW — GPU fused kernels
    ├── mod.rs
    ├── entmax_routing.wgsl   # Stage 1: entmax routing kernel
    ├── sparse_attn_bias.wgsl # Stage 2: masked FA with routing bias
    └── chunk_summary.wgsl    # Stage 0: local attention summarization
```

### Feature Gate

```toml
[features]
dash_attn = []        # α-entmax adaptive routing + learned chunk summaries
sp_kv = []            # Utility predictor (existing)
# Both can compose: dash_attn routing + sp_kv utility gating
```

---

## Composability with Existing Mechanisms

| Combination | Value | Notes |
|---|---|---|
| **DashAttn routing + SP-KV utility** | **High** | Entmax selects blocks, SP-KV gates individual KV writes within blocks |
| DashAttn + TurboQuant/SpectralQuant | High | Entmax selects blocks, quantization compresses selected KV |
| DashAttn + PFlash | **High** | Replace PFlash top-k with entmax — same flow, adaptive budget |
| DashAttn + HLA | Medium | HLA layers could use entmax routing for chunk-level attention |
| DashAttn + Raven | Low | Both address sparse attention; Raven is slot-based, entmax is block-based |
| DashAttn + OCTOPUS | High | Entmax routing + octahedral KV compression on selected blocks |
| DashAttn + Delta Routing | Orthogonal | Different axes (spatial vs depth) |

### Ideal Production Pipeline

```
Prefill:  PFlash with entmax routing (adaptive block selection)
          + learned chunk summaries (instead of mean-K)
Decode:   SP-KV utility gating (selective write)
          + entmax routing bias (additive to attention scores)
Storage:  SpectralQuant/OCTOPUS (compress retained KV)
Depth:    Delta routing (cross-layer info flow)
```

---

## Key Differences from NSA / InfLLMv2 (Our Baselines)

| Aspect | NSA | InfLLMv2 | DashAttention | Our PFlash |
|--------|-----|----------|---------------|------------|
| Chunk summary | MLP (extra params) | Mean pooling | Learned local attn | Mean-K dot |
| Block routing | Top-k (fixed k) | Top-k (fixed k) | α-entmax (adaptive) | Top-k (heuristic) |
| Differentiable | Compressed attn path | Not trainable Stage 1 | **Fully differentiable** | No (inference-only) |
| Head aggregation | Softmax (dispersive) | Softmax (dispersive) | **Entmax (non-dispersive)** | N/A |
| Prior strength | N/A | N/A | σ controls | N/A |
| Decode support | Yes | Yes | Yes | Prefill only |

---

## What We Should NOT Adopt

1. **Full 3-stage GPU pipeline** — Our CPU inference benefits from α-entmax algorithm, not Triton kernels. riir-ai can do WGSL kernels later.
2. **Training from scratch** — We do LoRA fine-tuning and inference. Continual pretraining with DashAttn is a riir-burner concern.
3. **σ tuning** — Paper uses σ=10^6-10^8 (essentially weakens prior to near-zero). For inference, just use the adaptive support from entmax without the prior bias.
4. **MLP chunk summarization** — Paper ablates against this; learned local attention is better and starts from mean pooling (zero-init).
5. **Bit-packed block masks** — Optimization for GPU kernels. On CPU, a Vec<usize> of active indices is fine.

---

## Risks & Limitations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| α-entmax adds compute vs top-k | Medium | α=1.5 reduces to quadratic ops; still O(n_chunks) not O(n_tokens) |
| Learned q̄ requires training | Low | Zero-init = mean pooling fallback; can run without training |
| Only validated at 1B-8B scale | Medium | Paper shows consistent results across 1B/3B/8B |
| Requires GQA (all experiments use GQA) | Low | We have full GQA support; SP-KV already handles this |
| Triton kernels are CUDA-only | High for riir-ai | We need WGSL/Metal reimplementation; riir-ai Plan 106 (cubecl) |
| Entmax threshold finding is iterative | Medium | α=1.5 has closed-form 2-pass algorithm; not expensive for ~256 chunks |

---

## Training Protocol (if needed for riir-ai)

1. Load pretrained model with softmax attention
2. Add `head_cls` vectors (zero-init) and set α=1.25
3. Continual pretrain with long-context data (16K)
4. Gradually increase α from 1.25 → 1.5
5. Short SFT stage
6. Inference with α=1.5 and softmax full attention for short contexts

---

## References

- DashAttention: arXiv 2605.18753
- α-entmax: Peters, Niculae, Martins (ACL 2019) — Sparse Sequence-to-Sequence Models
- AdaSplash-2: Gonçalves et al. (ICML 2026) — Faster differentiable sparse attention
- NSA: Yuan et al. (ACL 2025) — Native Sparse Attention
- InfLLM-v2: Zhao et al. (ICLR 2026) — Dense-sparse switchable attention
- MoBA: Lu et al. (NeurIPS 2025) — Mixture of Block Attention
- FlashAttention-3: Shah et al. (NeurIPS 2024)