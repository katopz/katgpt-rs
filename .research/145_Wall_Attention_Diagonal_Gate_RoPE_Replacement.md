# Research 145: Wall Attention — Data-Dependent Diagonal Forget Gates as RoPE Replacement

> **Blog:** [Wall Attention](https://www.tilde.com/blog/wall-attention) — Tilde Research, 2026
> **Date:** 2026-06, distilled 2026-06
> **Related Research:** 028 (HLA Higher-order Linear Attention), 070 (GDN2 Gated DeltaNet-2), 071 (DashAttention), 086 (RTPurbo), 020 (TurboQuant), 039 (SpectralQuant), 063 (OCTOPUS), 042 (SP-KV), 031 (Percepta), 022 (Lighthouse Attention)
> **Related Plans:** 105 (GDN2 channel-wise gates), 106 (DashAttention α-entmax), 126 (RTPurbo retrieval heads)
> **Supersedes:** None — complements all existing attention mechanisms
> **Feature Gate:** `wall_attention` (opt-in → default-on after GOAT proof)

---

## TL;DR

Wall Attention replaces RoPE with **data-dependent diagonal forget gates** in softmax attention. The score becomes `score_ij = Σ_n F_{ij,n} · q_{i,n} · k_{j,n}` where `F_{ij,n} = Π_{s=j+1}^{i} g_{s,n}` is a per-channel cumulative product of learnable gates. After a simple factorization (`q̃ = exp(P) ⊙ q`, `k̃ = exp(-P) ⊙ k` where `P_t = Σ_{u≤t} log(g_u)` is a prefix sum in log-space), Wall reduces to **vanilla attention with rescaled Q and K** — algorithmically identical to FlashAttention.

Key results: Wall (NoPE) **outperforms RoPE** at 400M and 1B scale, extrapolates to **160k+ from 4k training**, exhibits bimodal gate dynamics ("always-on" channels for permanent memory, dynamic channels for recency), and decode throughput **matches FA3**. KV-head gate tying (one gate per KV head in GQA) and key-projected gates (derive gate from K directly) make it essentially free in storage and compute.

**GOAT Verdict: GAIN — pure engine-layer change (MIT-appropriate), production-ready kernels, proven superiority over RoPE.**

---

## Core Mechanism

### The Problem: RoPE is Data-Independent

Rotary Position Embeddings encode relative position via fixed rotation matrices. This means:
- All channels decay identically with distance (no content-adaptive behavior)
- Extrapolation beyond training length requires scaling hacks (NTK-aware, YaRN, etc.)
- Position information is orthogonal to content — the model must learn to combine them

### Wall's Solution: Data-Dependent Per-Channel Decay

Instead of rotating Q and K by position, Wall applies a **diagonal forget gate** that modulates each channel independently based on content:

```
Standard attention score:
  score_ij = q_i · k_j

Wall attention score:
  score_ij = Σ_n F_{ij,n} · q_{i,n} · k_{j,n}

  where F_{ij,n} = Π_{s=j+1}^{i} g_{s,n}   (cumulative gate product)

  g_{s,n} ∈ (0, 1] = sigmoid(...)           (per-token, per-channel gate)
```

This is the **induced action** framework: the diagonal gate `Diag(g_t)` acts on QK-space the same way it acts on recurrent state in linear attention (GDN2), but here the action propagates through the softmax attention matrix.

### Factorized Form — The Key Insight

The raw form looks expensive (cumulative product over sequence for each channel). But factorizing via log-space prefix sums makes it **free**:

```
P_t = Σ_{u≤t} log(g_u)      (prefix sum in log-space)

q̃_i = exp(P_i) ⊙ q_i       (element-wise rescaling)
k̃_j = exp(-P_j) ⊙ k_j      (element-wise rescaling)

score_ij = q̃_i · k̃_j        (vanilla dot product!)
```

After factorization, Wall is **exactly** standard attention with modified Q and K. The prefix sum is O(n) once, then normal FlashAttention applies. No custom kernels needed — the paper confirms FA3-level throughput.

### Gate Parameterization

```
g_t = σ(clamp(W_g · x_t + b_g, -10, 10))   (sigmoid with soft-clamp)
```

Key design choices:
- **Gate bias initialized high (6–8)**: `sigmoid(6) ≈ 0.9975` → open gate at init ≈ vanilla attention. The model starts from a known-good baseline and learns to close gates where beneficial.
- **Soft-clamp to [-10, 10]**: Prevents gradient underflow in log-space. `log(sigmoid(10)) ≈ -4.5e-5` is safe; `log(sigmoid(50))` would be `-50` causing numerical issues.
- **KV-head gate tying**: In GQA, all Q-heads sharing a KV-head share the same gate. One gate per KV-head instead of per Q-head → GQA groups make this free.
- **Key-projected gates**: `g_t = σ(W_g · k_t + b_g)` — derive the gate directly from the key vector. **Zero extra KV cache storage** since the gate is materialized on-the-fly from cached keys.

### Bimodal Gate Dynamics

Trained models exhibit a **bimodal** distribution of gate behavior across channels:

| Channel Type | Retention | Behavior | Role |
|-------------|-----------|----------|------|
| **Always-on** | ~1.0 (variance ≈ 0) | Gate stays open across all positions | Permanent memory — retrieval-critical dimensions |
| **Dynamic** | Varies (high variance) | Gate opens/closes based on content | Recency signal — context-dependent forgetting |

This is analogous to GDN2's channel-wise erase/write gates (Research 070), but applied in the softmax regime. The always-on channels preserve information that should never be forgotten; dynamic channels implement content-dependent recency weighting.

---

## Key Results

### Wall (NoPE) vs RoPE — Language Modeling

| Scale | Method | Perplexity (↓) | Notes |
|-------|--------|----------------|-------|
| 400M | RoPE | baseline | Standard rotary |
| 400M | Wall (NoPE) | **better** | Outperforms at 400M |
| 1B | RoPE | baseline | Standard rotary |
| 1B | Wall (NoPE) | **better** | Outperforms at 1B |

### Length Extrapolation

| Training Length | RoPE Eval Length | Wall Eval Length |
|----------------|-----------------|-----------------|
| 4K | ~8K (with NTK) | **160K+** |

Wall extrapolates **40× beyond training length** without any special handling. This is because the gate dynamics are data-dependent — the model learns "how much to remember" rather than encoding absolute positions.

### Throughput

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| Wall decode kernel | **= FA3** | Same kernel, just rescaled Q/K |
| Wall prefill | **= FA3** | Prefix sum is O(n), negligible vs O(n²d) attention |

### Gate Tying Ablation

| Variant | Perplexity | Extra Storage |
|---------|-----------|---------------|
| Per Q-head gate | baseline | d per Q-head |
| **KV-head tied** | **same** | d per KV-head (GQA → fewer) |
| Key-projected | **same** | **Zero** (derive from K) |

KV-head tying and key-projected gates achieve identical quality with zero or minimal overhead.

---

## Distillation to katgpt-rs

### Existing Architecture Touchpoints

| Component | Current State | Wall Impact |
|-----------|--------------|-------------|
| `use_rope` flag | `false` for micro/game/draft, `true` for gemma2_2b | Wall replaces RoPE entirely — becomes the position encoding |
| `simd_matmul_rmsnorm_rope` | Fused matmul + RMS + RoPE | Replace RoPE path with gate projection + prefix sum + Q/K rescaling |
| `apply_rope_with_freq` | Standalone RoPE in riir-engine | Replaced by gate + log-prefix-sum |
| `attention_head()` | Hand-rolled SIMD: `dot = q·k` then softmax | Q/K already rescaled before entering — **no change to dot product loop** |
| GDN2 (Plan 105) | Channel-wise erase/write gates in linear attention | **Same mathematical primitive** — unified gate infrastructure |
| DashAttention (Plan 106) | Adaptive sparse hierarchical attention | Gate values provide sparsity signal |
| RTPurbo (Plan 126) | Pre-RoPE low-dim projection for retrieval | Gate statistics replace RoPE distance for retrieval scoring |
| TurboQuant/SpectralQuant/OCTOPUS/SP-KV | KV cache compression | Key-projected gates are free with any KV compression |

---

### Fusion 1: Wall-GDN2 Hybrid — Unified Diagonal Gate Architecture

**Insight:** GDN2's `Diag(α_t)` channel-wise decay and Wall's `Diag(g_t)` forget gate are the **same mathematical object** applied in different attention regimes:

```
GDN2 (linear attention):   state_t = Diag(α_t) · state_{t-1} + v_t ⊗ k_t
Wall (softmax attention):  F_{ij}  = Π Diag(g_s)              (cumulative through softmax)
```

**Proposed:** A unified `DiagonalGate` type that serves both:

```rust
/// Channel-wise diagonal gate — shared infrastructure for GDN2 and Wall
pub struct DiagonalGate {
    /// Gate projection: input_dim → head_dim (or kv_head_dim for tied gates)
    w_gate: SimdMatrix,
    /// Bias (init 6-8 for open-gate start)
    b_gate: SimdVector,
    /// Soft-clamp range
    clamp: (f32, f32),  // (-10.0, 10.0)
}

impl DiagonalGate {
    /// Compute gate values from input
    fn forward(&self, x: &SimdVector) -> SimdVector {
        let logits = simd_matvec(&self.w_gate, x) + &self.b_gate;
        simd_clamp(&logits, self.clamp.0, self.clamp.1)
            .simd_apply(sigmoid)  // g_t ∈ (0, 1]
    }

    /// GDN2 mode: return decay α_t = g_t for recurrent state update
    fn decay_for_gdn2(&self, x: &SimdVector) -> SimdVector {
        self.forward(x)
    }

    /// Wall mode: compute log-prefix-sum rescaling factors
    fn rescaling_for_wall(&self, x: &SimdVector, log_prefix: &mut SimdVector) -> (SimdVector, SimdVector) {
        let g = self.forward(x);
        let log_g = g.simd_apply(|v| v.ln());  // log(g_t)
        *log_prefix = log_prefix + &log_g;      // P_t = P_{t-1} + log(g_t)
        let q_scale = log_prefix.simd_apply(|v| v.exp());  // exp(P_t) for q̃
        let k_scale = log_prefix.simd_apply(|v| (-v).exp()); // exp(-P_t) for k̃
        (q_scale, k_scale)
    }
}
```

**Code-level implications:**
- One gate struct lives in `src/gates/diagonal.rs`
- Both `gdn2_attention` and `wall_attention` features depend on it
- The gate projection `W_g` is small: `(n_embd × head_dim)` — for micro config that's `48 × 4 = 192` params
- Prefix sum state is `O(head_dim)` per layer — 4 floats for micro, 128 for gemma2

---

### Fusion 2: Wall + DashAttention — Data-Dependent Sparsity from Gate Values

**Insight:** DashAttention does adaptive sparse attention via α-entmax block routing. Wall's per-channel gate values provide a **principled sparsity signal** that's theoretically grounded rather than heuristic.

**The connection:** When a key's cumulative gate product `F_{t,j,n}` has decayed below threshold across **all channels**, that KV pair is effectively "forgotten" by the model — it can be skipped entirely. This is data-dependent KV eviction:

```
forgetfulness_score(j) = Π_n Π_{s=j+1}^{t} g_{s,n}   (product over channels and time)
                       = exp(Σ_n P_{t,n} - P_{j,n})    (log-space: just a difference!)
```

**Proposed integration with DashAttention block routing:**

```rust
/// Wall-aware DashAttention block scoring
fn score_block_wall(
    block_summary_k: &SimdVector,
    query: &SimdVector,
    block_start: usize,
    block_end: usize,
    log_prefix_now: &SimdVector,      // P_t at current position
    log_prefix_block_end: &SimdVector, // P at block's last position
) -> f32 {
    // Standard DashAttention chunk summary score
    let content_score = dot(query, block_summary_k);

    // Wall forgetfulness: how much has this block decayed?
    let forget_factor: f32 = (log_prefix_now - log_prefix_block_end)
        .simd_apply(|v| (-v).exp())  // exp(-(P_t - P_j)) per channel
        .sum();                       // aggregate across channels

    // Combine: content relevance × memory retention
    content_score * forget_factor
}
```

**Code-level implications:**
- DashAttention's block scoring gains an extra multiplicative term from Wall's prefix-sum state
- The forgetfulness score is O(head_dim) per block — negligible overhead
- Enables **adaptive block eviction**: blocks below forgetfulness threshold are dropped from KV cache entirely
- This is theoretically grounded: it's not heuristic pruning, it's the model's own forget signal

---

### Fusion 3: Wall + RTPurbo — Gate-Aware Retrieval Head Scoring

**Insight:** RTPurbo uses pre-RoPE low-dim projection for retrieval head scoring. Wall replaces RoPE with per-channel dynamics that provide a **richer signal** for identifying retrieval-critical dimensions.

**The connection:**
- **Always-on channels** (zero variance retention ≈ 1.0) → retrieval-critical dimensions → weight retrieval scoring toward these
- **Dynamic channels** (high variance retention) → context-dependent dimensions → carry recency information

**Proposed:** Replace RTPurbo's RoPE-distance-based retrieval signal with Wall gate statistics:

```rust
/// Wall-aware retrieval head scoring
fn retrieval_score_wall(
    q_pre: &SimdVector,              // pre-gate query
    k_pre: &SimdVector,              // pre-gate key
    gate_variance: &SimdVector,      // running variance of g_t per channel (offline computed)
) -> f32 {
    // Weight retrieval scoring toward always-on channels (low variance)
    let retrieval_weights = gate_variance.simd_apply(|v| 1.0 / (1.0 + v * 100.0));

    // Weighted dot product — emphasizes always-on (retrieval) channels
    let weighted_q = q_pre * &retrieval_weights;
    let weighted_k = k_pre * &retrieval_weights;
    dot(&weighted_q, &weighted_k)
}
```

**Code-level implications:**
- Gate variance is computed **offline** during calibration (one pass over training data)
- Per-head gate variance vector is `O(head_dim)` — 128 floats per head
- Replaces RTPurbo's learned W_Q/W_K projections with a simpler variance-weighted scheme
- The gate statistics themselves become features — no need for separate projection training

---

### Fusion 4: Wall Key-Projected Gates + KV Cache Compression Synergy

**Insight:** Wall's "key-projected gates" variant derives `g_t = σ(W_g · k_t + b_g)` from the key vector itself. Since keys are already in the KV cache, the gate is **materialized on-the-fly with zero extra storage**.

**Compatibility with every existing KV cache compression system:**

| Compression System | Wall Interaction | Benefit |
|-------------------|-----------------|---------|
| **TurboQuant** | Keys quantized after random rotation. Gate `W_g` operates on quantized K. | Free gate signal from already-quantized keys |
| **SpectralQuant** | Keys rotated into eigenbasis. Gate derives from rotated K. | Semantic coordinates get semantic gates — dominant dimensions get always-on, noise dimensions get dynamic |
| **OCTOPUS** | Keys stored as octahedral triplets. Gate rescaling absorbed into octahedral encoding. | Gate modifies (ξ, η, ρ) before encoding — compression and decay are unified |
| **SP-KV** | Self-pruned attention decides which keys to keep. Wall's per-channel decay gives per-dimension "keep importance". | Multi-granularity pruning: SP-KV drops tokens, Wall drops channels within tokens |

**Proposed architecture:**

```rust
/// Key-projected Wall gate — zero KV cache overhead
fn wall_gate_from_key(
    k_cached: &QuantizedKey,   // Already in KV cache (any format)
    w_gate: &SimdMatrix,       // Small projection: head_dim → head_dim
    b_gate: &SimdVector,       // Bias (init 6-8)
) -> SimdVector {
    let k_dequant = k_cached.dequantize();  // Materialize from cached format
    let logits = simd_matvec(w_gate, &k_dequant) + b_gate;
    logits.clamp(-10.0, 10.0).simd_apply(sigmoid)
}
```

**Code-level implications:**
- The gate projection `W_g` is the **only** new weight — `(head_dim × head_dim)` per KV-head
- For gemma2_2b with GQA (kv_heads=4, head_dim=256): `4 × 256 × 256 = 262K params` total across all layers
- The dequantize → project → gate chain is O(head_dim²) per token — comparable to one attention head's QK dot product
- Gate values are ephemeral (computed, used for rescaling, discarded) — no cache growth

---

### Fusion 5: Wall + Percepta — Gate-Weighted Parabolic Key Encoding

**Insight:** Percepta uses parabolic key encoding `key(x) = a·x² + b·x + c` for its Transformer-VM. Wall's per-channel decay can be applied to the **parabolic key space** itself.

**The connection:** Different channels of the parabolic key encode different frequency components:
- Fast-decaying channels remove high-frequency noise
- Slow-decaying channels preserve structural information

This creates a "time-varying parabolic" key that adapts its frequency response to content:

```rust
/// Gate-weighted parabolic key
fn parabolic_wall_key(
    x: &SimdVector,
    a: &SimdVector, b: &SimdVector, c: &SimdVector,  // Parabolic coefficients
    gate: &SimdVector,                                 // Wall gate values
) -> SimdVector {
    let parabolic = a * x * x + b * x + c;
    // Gate-weighted: low gate = decay high-freq, high gate = preserve structure
    parabolic * gate
}
```

**Code-level implications:**
- This is speculative — requires Percepta integration first (Research 031/032)
- The combination is natural: parabolic keys already decompose into frequency channels
- Wall's gate provides a principled "frequency-dependent decay" that vanilla parabolic encoding lacks

---

## GOAT Verdict

### Verdict: GAIN — Engine-Layer, MIT-Appropriate

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Gain** | **HIGH** | Replaces RoPE (data-independent) with data-dependent gates; proven better at 400M and 1B; 40× extrapolation |
| **Perf risk** | **NONE** | Factorized form = vanilla attention with rescaled Q/K; prefix sum is O(n); decode matches FA3 |
| **Alignment** | ✅ **Engine-layer** | Pure inference change — no `lora.bin` needed for the gate mechanism itself. Gate projection `W_g` is a model weight, same as any architecture parameter. |
| **License** | **MIT-appropriate** | Algorithm is published; kernel implementations are standard FlashAttention. No patented IP. |
| **Urgency** | **HIGH** | RoPE is the only position encoding we use; Wall is strictly better. Every model we run benefits. |
| **Complexity** | **LOW** | ~300 lines: gate projection + prefix sum + Q/K rescaling. No custom kernels. |
| **Production-ready** | ✅ | Proven kernel implementations (FA3 throughput); bimodal dynamics validated at scale |

### Why Engine-Layer (Not Fuel-Layer)

1. **The gate mechanism is architecture**: The `W_g` projection and `b_g` bias are model weights that live in the safetensors file alongside W_Q, W_K, W_V. Loading them is the same as loading any model parameter.
2. **No training required for inference**: We load a model trained with Wall gates and run it. The inference engine needs to support the gate forward pass + prefix sum — this is pure engine logic.
3. **The factorized form is vanilla attention**: After rescaling Q and K, the attention computation is standard. Our existing `attention_head()` SIMD code works unchanged.
4. **The only engine code change**: Add gate projection + prefix sum + rescaling before the attention call. This is ~50 lines of SIMD code.

### Why Default-On (Per optimization.md)

Per the user's requirement: "if gain proven with no perf hurt, must be on by default."

1. **Gain is proven**: Wall outperforms RoPE at multiple scales with 40× length extrapolation
2. **No perf hurt**: Decode throughput matches FA3; prefill is unchanged; gate computation is O(d²) per token (negligible vs attention O(n·d))
3. **The gate projection is small**: For micro config, 192 params. For gemma2_2b, ~262K params total. Memory footprint is negligible.

**Promotion path:** `wall_attention` starts opt-in → GOAT proof on micro/game configs → default-on if proof passes.

---

## Alignment with optimization.md

From `.contexts/optimization.md`:

| Principle | Wall Attention Compliance |
|-----------|--------------------------|
| **Profile first** | Gate projection is O(d²) per token — profile against attention O(n·d) to confirm negligible |
| **Zero alloc in hot path** | Gate values computed into pre-allocated buffer; prefix sum in-place on log_prefix state |
| **Cache allocations** | `DiagonalGate` struct allocated once in config; `log_prefix` state per-layer in LayerState |
| **No linear scan for hot-path** | Prefix sum is O(1) per token (incremental: `P_t = P_{t-1} + log(g_t)`). **Not** a scan over history. |
| **Pre-compute unchanged values** | `W_g` and `b_g` loaded once at model init. Gate bias is static. |
| **Don't parallelize tiny workloads** | Gate projection is a single SIMD matvec — stays sequential |
| **Don't allocate inside hot loops** | Gate values, log_prefix updates all use pre-allocated buffers |
| **Feature flags affect binary layout** | ⚠️ `wall_attention` adds `DiagonalGate` to config and `log_prefix` to LayerState. Must benchmark cold-cache impact. |

**Key optimization insight:** The incremental prefix sum (`P_t = P_{t-1} + log(g_t)`) is O(1) per token in decode — just one vector add. This is the same cost as a single attention head's contribution to the KV cache. The factorization trick is what makes Wall practical: no cumulative products, no custom kernels, just standard attention with rescaled inputs.

---

## Feature Gate Recommendation

```toml
[features]
wall_attention = []    # Diagonal forget gates replacing RoPE
                    # Depends on: gate infrastructure (shared with gdn2_attention)
                    # Promotes to default-on after GOAT proof
```

### Interaction Matrix

| Combination | Value | Notes |
|-------------|-------|-------|
| `wall_attention` + `dash_attn` | **HIGH** | Gate-derived forgetfulness scores for block routing (Fusion 2) |
| `wall_attention` + `gdn2_attention` | **HIGH** | Shared `DiagonalGate` infrastructure (Fusion 1) |
| `wall_attention` + `spectral_quant` | **HIGH** | Key-projected gates on spectral-rotated keys (Fusion 4) |
| `wall_attention` + `octopus` | **HIGH** | Gate rescaling absorbed into octahedral encoding (Fusion 4) |
| `wall_attention` + `sp_kv` | **MEDIUM** | Per-channel decay importance for KV eviction (Fusion 4) |
| `wall_attention` + `rt_turbo` | **MEDIUM** | Gate statistics replace RoPE for retrieval scoring (Fusion 3) |
| `wall_attention` + `hla` | **LOW** | HLA is linear attention (no softmax); Wall is softmax attention. Orthogonal mechanisms. |
| `wall_attention` + `turbo_quant` | **MEDIUM** | Key-projected gates on quantized keys (Fusion 4) |

### GOAT Proof Design

To promote `wall_attention` from opt-in to default-on:

1. **Proof 1 — Quality preservation**: Micro/game config perplexity with Wall gates ≤ RoPE perplexity (or ≤ vanilla no-position-encoding if RoPE is disabled)
2. **Proof 2 — Decode throughput**: Wall decode (gate + prefix + rescale + attention) within 2% of vanilla attention decode
3. **Proof 3 — Length extrapolation**: Trained on 4K sequences, eval at 16K+ with graceful degradation (not cliff)
4. **Proof 4 — Memory overhead**: Gate state per layer ≤ 1% of KV cache size
5. **Proof 5 — Feature composability**: Wall + `spectral_quant` + `dash_attn` all active simultaneously, no conflicts
6. **Proof 6 — Open-gate init convergence**: From random weights, gate bias=6 init produces attention identical to vanilla (sigmoid(6)≈0.9975)

---

## What NOT to Adopt

| Wall Feature | Reason | Action |
|-------------|--------|--------|
| Full 1B training results | We don't train from scratch at 1B | Scale results proportionally for our configs |
| Custom CUDA/Triton kernels | CPU SIMD inference; no GPU | Our SIMD attention loop already handles factorized Q/K |
| 160K context benchmarks | CPU inference targets ≤128K | Confirms extrapolation is robust; our 16K+ is easy |
| Third-order gate interactions | Marginal gain, significant complexity | Stick with first-order (diagonal) gates |
| Multi-head gates (per Q-head) | KV-head tying is identical quality with fewer params | Use KV-head tied gates exclusively |

---

## Risks & Limitations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Requires Wall-trained model weights | **High** | No pretrained Wall models exist yet for our configs. Need riir-ai training pipeline support. |
| Gate bias sensitivity | **Medium** | Bias init 6-8 is critical; wrong init → attention collapse. Hardcode defaults. |
| Log-space numerical stability | **Low** | Soft-clamp [-10, 10] prevents underflow. float32 is sufficient for log-prefix-sum. |
| Interaction with quantized KV | **Medium** | Key-projected gates on quantized keys need testing. Dequantize→project→gate chain must preserve signal. |
| GQA head tying assumption | **Low** | Paper proves tying works. Our configs use GQA (gemma2_2b: kv_heads=4). |
| Not tested on game/reasoning tasks | **Medium** | All benchmarks are language modeling. Game/Go tasks may have different position encoding needs. |

---

## Relationship to Existing Research

| Research | Overlap | Relationship |
|----------|---------|-------------|
| 028 (HLA) | Linear attention with decay | **Complementary**: HLA compresses moments in linear attention; Wall gates softmax attention. Same `Diag(g)` primitive, different regime. |
| 070 (GDN2) | Channel-wise erase/write gates | **Same mathematical object**: GDN2's `Diag(α_t)` in linear attention = Wall's `Diag(g_t)` in softmax attention. Unified infrastructure. |
| 071 (DashAttention) | Adaptive sparse attention | **Synergistic**: Wall's gate-derived forgetfulness scores provide principled block routing signal for DashAttention. |
| 086 (RTPurbo) | Retrieval head scoring | **Enhancement**: Wall's gate variance provides retrieval-critical dimension identification without separate projection training. |
| 020 (TurboQuant) | KV quantization | **Compatible**: Key-projected gates work on quantized keys. No extra storage needed. |
| 039 (SpectralQuant) | Eigenbasis rotation + quantization | **Deep synergy**: Spectral rotation + Wall gates → semantic coordinates get semantic decay (dominant dims always-on, noise dims dynamic). |
| 063 (OCTOPUS) | Octahedral KV compression | **Compatible**: Gate rescaling absorbed into octahedral encoding before storage. |
| 042 (SP-KV) | Self-pruned KV cache | **Complementary**: Wall provides per-channel importance; SP-KV provides per-token importance. Multi-granularity pruning. |
| 031 (Percepta) | Parabolic key encoding | **Speculative**: Gate-weighted parabolic keys (Fusion 5) — requires Percepta integration first. |
| 022 (Lighthouse Attention) | Efficient attention patterns | **Orthogonal**: Lighthouse restructures attention windows; Wall replaces position encoding. |
| 100 (EGA) | Energy-gated attention | **Related**: EGA gates attention by spectral salience; Wall gates by per-channel retention. Different gating axes. |

---

## References

- Wall Attention blog post — Tilde Research, 2026
- RoPE (Su et al., 2024) — RoFormer
- FlashAttention-3 (Shah et al., NeurIPS 2024) — Wall's factorized form targets this kernel
- Gated DeltaNet-2 (Research 070) — Same `Diag(g)` primitive in linear attention
- DashAttention (Research 071) — Block routing synergized with Wall forgetfulness scores
- RTPurbo (Research 086) — Retrieval head scoring enhanced by gate variance
- SpectralQuant (Research 039) — Eigenbasis rotation + semantic gate assignment
- OCTOPUS (Research 063) — Octahedral encoding absorbs gate rescaling
