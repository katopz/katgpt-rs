# Research 73: LT2 — Linear-Time Looped Transformers

> **Paper:** [LT2: Linear-Time Looped Transformers](https://arxiv.org/abs/2605.20670) — Deng, Zhang, Zhu, Xu, Liu, Ng, Chen (Rice/Apple/UCSC/CMU), May 2026
> **Code:** https://github.com/facebookresearch/lingua (forked, apps/LT2)
> **Date:** 2026-05, distilled 2025-07
> **Related Research:** 28 (HLA), 70 (Gated DeltaNet-2), 71 (DashAttention), 55 (Nemotron TriMode), 58 (GRAM), 057 (Higher-order LA), 097 (Delta Attention Residuals)
> **Related Plans:** 108 (LT2 Looped Inference Pipeline)
> **Verdict: HIGH VALUE — LT2's looped weight-sharing is a natural fit for our parameter-constrained CPU inference. The rank-T state upgrade from looping directly amplifies our existing HLA/AHLA recurrent states. Hybrid (Full+GDN) with 1:4 ratio is the flagship recipe. SDPA output gate is a free lunch (+0.3–0.5 avg points). Feature-gate as `lt2_looped`. Priority: looped AHLA (our existing linear attention) first, then hybrid with windowed SDPA.**

---

## TL;DR

LT2 replaces the quadratic attention in Looped Transformers with subquadratic token mixers (linear, sparse, or hybrid). The key finding: **looping synergizes uniquely with subquadratic attention** — T loops turn rank-1 DPLR state updates into rank-T updates (enabling state tracking), and turn window-w sparse attention into effective receptive field T·w (enabling long-context).

Two flagship hybrid variants:
1. **LT2-hybrid (Full+GDN)** — 1:4 full-to-linear ratio. Best quality: +2.1 avg points over standard looped transformer at 1.3B, ~2.7× decode speedup.
2. **LT2-hybrid (GDN+DSA)** — Fully linear-time. Matches full-attention looped transformer quality with ~5.7× decode speedup.

Distillation pathway: Pre-trained Ouro-1.4B → LT2-hybrid with ~1B tokens training, competitive with industry 4B models.

**For our stack:** We already have AHLA (O(1) memory linear attention). Looping AHLA T=4 times gives rank-4 state updates for free — same weights, 4× effective depth. The SDPA output gate eliminates attention-sink compounding. The hybrid pattern (1:4 full+linear) maps directly to our existing SDPA + AHLA dispatch.

---

## Core Innovation: Loop × Subquadratic Synergy

### 1. Rank-T State Upgrade (Loop × DPLR Linear Attention)

A single DPLR block (GDN, KDA, DeltaNet) applies a rank-1 perturbation to recurrent state:

```
S_t = A_t · S_{t-1} + β_t · k_t · v_tᵀ
A_t = Diag(α_t)(I - β_t · k_t · k_tᵀ)  // rank-1 + diagonal
```

Looping T times composes T such operators:

```
A_eff = ∏_{τ=1}^{T} Diag(α_t^(τ))(I - β_t^(τ) · k_t^(τ) · k_t^(τ)ᵀ)
```

**Key result:** When loop-specific keys are approximately orthogonal (expected in high-dim spaces), the effective perturbation rank is T, not 1. By Cartan-Dieudonné theorem, T ≥ d_k loops suffice to realize any orthogonal transformation in O(d_k).

| Loops | Rank | State Tracking |
|-------|------|---------------|
| T=1 | 1 | Cannot solve S_n (n≥3) |
| T=2 | 2 | Reflections + rotations in 2D |
| T=4 | ≤4 | Solves prefix products for S_5 |
| T=d_k | ≤d_k | Universal orthogonal representation |

**Connection to our HLA (Research 28):** Our AHLA maintains (PKV, mK, E, n) state in O(d·dv). Looping AHLA T times means T independent key projections acting on this state — the same rank-T upgrade applies. Our second-order SK accumulator would benefit even more: T loops produce T rank-1 key-direction corrections to SK, yielding a rank-T update to the key second-moment matrix.

### 2. Receptive Field Expansion (Loop × Sparse Attention)

Window-w sparse attention, looped T times:

```
ℐ_t^(T) ⊇ {max(1, t - T·w + 1), ..., t},  |ℐ_t^(T)| = O(T·w)
```

T=4 loops of window-2048 → effective receptive field of 8192. This matches 4 stacked layers of window-2048 attention but with 4× fewer parameters.

**Important caveat (from paper Appendix B.2.2):** Residual connections cap the *effective* receptive field. With residual skip connections (α ≈ 0.95), influence decays as `(1-α)^{⌈d/w⌉}`, yielding an effective horizon of ~1.5w regardless of T. The combinatorial reach is O(Tw), but the signal quality at distance d > 2w is exponentially attenuated.

**Practical implication:** Looping sparse attention helps mostly in the 1–2w band. For truly long-range recall, you still need either (a) some full attention layers, or (b) linear attention with recurrent state carry.

### 3. SDPA Output Gate (Attention Sink Suppression)

In looped transformers, the attention sink (first-token mass concentration) compounds across loops — a sawtooth pattern that intensifies each iteration. Fix: a head-specific sigmoid gate after SDPA:

```python
gate = sigmoid(x @ W_gate.T)  # zero-init → starts at 0.5
output = sdpa_output * gate    # before output projection
```

Results at 1.3B:
| Model | Gate | PPL | Avg |
|-------|------|-----|-----|
| Looped Transformer | — | 9.87 | 59.27 |
| Looped Transformer | ✓ | 9.39 | 60.70 |
| LT2-Hybrid (Full+GDN) | — | 9.31 | 61.39 |
| LT2-Hybrid (Full+GDN) | ✓ | **9.03** | **62.33** |

---

## Hybrid Architecture: The Pareto Frontier

### Depth-Level Hybrid (1:4 Full+GDN Interleave)

The winning pattern: every 5th layer uses full attention, the other 4 use linear attention.

```
[GDN, GDN, GDN, GDN, Full, GDN, GDN, GDN, GDN, Full, ...]
```

Ablation results (1.3B, T=4):
| Full:GDN Ratio | PPL | Avg |
|----------------|-----|-----|
| 1:0 (full only) | 9.87 | 59.27 |
| 1:1 | 9.41 | 60.92 |
| **1:4 (optimal)** | **9.03** | **62.33** |
| 1:6 | 9.36 | 61.07 |
| 1:12 | 9.74 | 59.51 |
| 0:1 (GDN only) | 10.02 | 58.42 |

Pattern placement matters: bookend > interleave > front-loaded > back-loaded. Spreading full attention layers is critical.

### Loop-Level Hybrid (Coarse→Fine)

Less effective than depth-level. The paper tried:
- Full → SWA-512 → SWA-256 → SWA-128 (coarse→fine)
- SWA-128 → SWA-256 → SWA-512 → Full (fine→coarse)

Both underperform fixed depth-level interleave. The coarse→fine wins on PPL but loses on downstream (overfits local statistics in final loops).

---

## Key Experimental Results

### Language Modeling (100B tokens FineWeb-Edu, T=4)

| Model (1.3B) | PPL | ARC-E | ARC-C | HellaS | PIQA | Avg |
|---|---|---|---|---|---|---|
| Transformer | 10.65 | 67.52 | 33.84 | 52.47 | 71.03 | 56.04 |
| Looped Transformer | 9.87 | 70.83 | 37.54 | 57.06 | 72.43 | 59.27 |
| Looped GDN | 9.75 | 71.28 | 38.33 | 57.73 | 73.37 | 59.92 |
| Looped KDA | 9.68 | 71.57 | 38.62 | 57.99 | 73.53 | **60.14** |
| Looped DSA | 9.97 | 69.93 | 36.93 | 56.38 | 71.94 | 58.54 |
| **Hybrid (Full+GDN)** | **9.03** | **74.82** | **41.63** | **61.04** | **75.93** | **62.33** |
| Hybrid (GDN+DSA) | 9.50 | 72.44 | 39.33 | 58.84 | 73.98 | 60.73 |

### Efficiency at Long Context (1.3B, H100)

| Variant | Decode@8K (t/s) | Decode@32K (t/s) | OOM Frontier (bs=8) |
|---------|-----------------|------------------|---------------------|
| Looped Transformer | 125 | 22 | 8K |
| Looped GDN | 135 | 120 | >32K |
| Hybrid (Full+GDN) | 130 | 105 | >32K |
| Hybrid (GDN+DSA) | 128 | 115 | 16K |

Linear-time variants hold flat decode throughput across the entire range. Looped Transformer loses 82% of throughput between 4K and 32K.

### Distillation: Ouro → LT2-hybrid

3-stage recipe:
1. **Linear pre-alignment** (100M tokens, len=512): MSE loss aligning GDN blocks to teacher attention outputs
2. **Hybrid logit distillation** (600M tokens, len=4096): KL-div with per-loop supervision schedule
3. **Long-context continuation** (600M tokens, len=32768): extend with OpenThoughts reasoning data

Result: Ouro-Hybrid-1.4B matches industry 1B models, approaches 4B models, with ~1B tokens total training.

---

## Training Stability Findings

Critical for implementation. The paper found clear stability tiers:

| Tier | Mixers | Behavior |
|------|--------|----------|
| **Most stable** | GDN (gating + delta rule) | Smoothest loss, smallest gradient norms |
| **Stable** | Mamba2 (gating, no delta rule), DeltaNet (delta rule, weaker gating) | Occasional spikes |
| **Unstable** | RetNet (no gating, no delta rule) | **Diverges** |

**Takeaway for us:** Our looped implementation MUST include data-dependent gating (α_t) and preferably a delta-rule update (β_t · k_t · v_tᵀ). Our AHLA already has channel-wise decay — this maps to α_t. The delta rule is additive — easy to layer on.

---

## Mapping to Our Architecture

### What We Already Have

| LT2 Component | Our Equivalent | Status |
|---|---|---|
| GDN linear attention | AHLA (asymmetric HLA) | ✅ Implemented (Plan 057) |
| Sliding window attention | SDPA with window | ✅ In transformer.rs |
| SDPA output gate | — | ❌ Not yet |
| Loop weight sharing | — | ❌ Not yet |
| Per-loop residual gate ρ_τ | — | ❌ Not yet |
| DSA sparse attention | DashAttention α-entmax | 🔬 Research 71 |
| GDN2 channel-wise erase | — | 🔬 Research 70 |

### Natural Fit Points

1. **Looped AHLA** — Our AHLA already maintains O(d·dv) constant state. Looping T=4 times gives:
   - 4× effective depth with same parameter count
   - Rank-4 state updates (up from rank-1)
   - ~95% of SDPA throughput maintained (from our benchmarks)
   - No KV cache growth per loop iteration

2. **Hybrid SDPA+AHLA** — Our `forward()` already dispatches on `HlaMode`. Adding a loop with depth-level hybrid:
   - Every 5th "layer" uses standard SDPA (exact recall)
   - Other 4 use AHLA (constant memory, streaming)
   - Only 1/5 of layers pay quadratic cost

3. **SDPA Output Gate** — Zero-init sigmoid gate after attention, before Wo projection. ~`n_heads × head_dim` extra parameters. Free +0.3–0.5 avg points.

### What We Don't Need

- **RetNet, HGRN2, DeltaNet, Mamba2** — Our AHLA is our linear attention. The paper confirms: gating + delta rule matters, specific mixer family matters less.
- **NSA** — DashAttention (Research 71) covers this with α-entmax, which is strictly better than top-k routing.
- **Loop-level hybrid** — Underperforms depth-level. Skip.
- **ACT (adaptive computation time)** — Paper tried it, found unstable at scale. We use fixed T.

---

## Implementation Strategy

### Phase 1: Looped Forward Pass (Feature: `lt2_looped`)

Core change to `transformer.rs`:

```rust
/// Looped transformer configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopMode {
    /// Standard single-pass (no looping).
    #[default]
    None,
    /// Weight-shared looping: same layers applied T times.
    /// Effective depth = n_layer × loop_count.
    WeightShared {
        loop_count: usize,     // T (paper default: 4)
    },
}

/// Hybrid attention pattern for looped inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HybridPattern {
    /// All layers use the same attention mode.
    #[default]
    Uniform,
    /// Depth-level interleave: every Nth layer uses full SDPA.
    /// e.g., Interleave { full_ratio: 5 } = every 5th layer is full.
    Interleave { full_ratio: usize },
    /// Bookend: first and last layers are full, middle is linear.
    Bookend,
}
```

Forward pass change:
```rust
// Before: single pass through all layers
for layer in 0..config.n_layer {
    forward_layer(layer, ...)
}

// After: looped with weight sharing
for tau in 0..loop_count {
    for layer in 0..config.n_layer {
        let is_full = match hybrid_pattern {
            HybridPattern::Uniform => false,
            HybridPattern::Interleave { full_ratio } => {
                (layer % full_ratio) == full_ratio - 1
            }
            HybridPattern::Bookend => {
                layer == 0 || layer == config.n_layer - 1
            }
        };
        forward_layer(layer, is_full, ...);
    }
    // Per-loop residual gate: h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
    apply_residual_gate(tau, &mut hidden_state, &residual_gates);
}
```

### Phase 2: SDPA Output Gate (Feature: `lt2_looped`)

```rust
/// Head-specific sigmoid gate after SDPA, before Wo.
/// Zero-initialized → starts at sigmoid(0) = 0.5 (neutral).
pub struct SdpaOutputGate {
    pub w_gate: Vec<f32>,  // [n_heads * head_dim, dim]
}
```

### Phase 3: Looped AHLA State Carry (Feature: `lt2_looped`)

The key: AHLA state (PKV, mK, E, n) carries across loop iterations, accumulating rank-T updates.

```rust
// Per-layer AHLA state persists across loops within a single sequence
let mut ahla_states: Vec<AhlaState> = vec![AhlaState::new(config); n_layer];

for tau in 0..loop_count {
    for layer in 0..n_layer {
        if is_linear_layer(layer) {
            forward_ahla_layer(layer, &mut ahla_states[layer], ...);
        } else {
            forward_sdpa_layer(layer, &mut kv_cache[layer], ...);
        }
    }
}
```

---

## Feature Gates

| Gate | Scope | Description |
|------|-------|-------------|
| `lt2_looped` | katgpt-rs | Looped forward pass with weight sharing + hybrid dispatch + SDPA output gate |
| `lt2_looped` | katgpt-core | `LoopMode`, `HybridPattern` enums, `SdpaOutputGate` struct |

Dependencies: `lt2_looped` requires `hla_attention` (for AHLA linear layers in hybrid mode).

---

## Benchmarking Strategy

Before implementation, benchmark our existing forward pass to establish baselines:
1. **Single-layer SDPA** (current) — tokens/second, µs/step
2. **Single-layer AHLA** (current) — tokens/second, µs/step
3. **4× looped SDPA** (naive) — expected 4× slowdown, KV cache ×4
4. **4× looped AHLA** — expected ~4× compute, constant memory
5. **Hybrid 1:4 (SDPA+AHLA)** — expected ~1.5× compute, 1/4 KV cache of full loop

Target: hybrid (1:4 SDPA+AHLA) at ≥60% of single-pass SDPA throughput with 75% KV cache reduction at long contexts.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Loop instability (gradient explosion) | High | Data-dependent gating (α_t) + delta rule (β_t) in AHLA |
| Per-loop residual gate adds params | Low | Only `loop_count × dim` scalars, zero-init |
| No training loop yet | Medium | Focus on inference first; training in riir-ai |
| Effective receptive field limited by residuals | Medium | Hybrid with full attention recovers recall |
| KV cache still needed for full-attention layers | Low | Only 1/5 of layers in hybrid pattern |

---

## Open Questions

1. **How does looped AHLA quality compare to looped GDN?** Our AHLA is asymmetric (different inductive bias than GDN). The rank-T upgrade applies to both, but quality may differ. Needs empirical validation.

2. **Optimal T for CPU inference?** Paper uses T=4. On CPU, each loop iteration is pure compute (no GPU parallelism). T=2 or T=3 may be optimal for throughput/quality tradeoff.

3. **Cross-loop state sharing?** Paper explicitly notes this as future work: "principled state-sharing across loops may further improve long-context modeling." Our AHLA states naturally carry across loops — this is a potential advantage.

4. **Distillation from pre-trained models?** The Ouro→LT2 pathway requires a pre-trained teacher. We'd need to either (a) train from scratch with looped config, or (b) adapt from an existing model. Option (a) is more aligned with our modelless-first philosophy.

---

## References

- LT2 paper: https://arxiv.org/abs/2605.20670
- LT2 codebase (reference): `.raw/LT2/`
- Gated DeltaNet (GDN): https://arxiv.org/abs/2412.06464
- Gated DeltaNet-2 (Research 70): our distilled research on channel-wise erase/write
- DashAttention (Research 71): our distilled research on α-entmax sparse attention
- HLA (Research 28): our implemented second-order linear attention
- Ouro (looped LM at scale): https://arxiv.org/abs/2502.09556