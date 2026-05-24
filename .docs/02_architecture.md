# katgpt-rs: Core Architecture

## Overview
The transformer is a from-scratch GPT-2 style implementation. No frameworks — weights are `Vec<f32>`, ops are hand-written matmul/softmax/rmsnorm. Supports multi-layer, grouped-query attention (GQA), and zero-allocation inference.

## Config (`crates/katgpt-core/src/types.rs`, re-exported via `src/types.rs`)
```rust
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,     // max sequence length
    pub n_embd: usize,         // embedding dimension
    pub n_head: usize,         // number of attention Q heads
    pub head_dim: usize,       // dimension per head (n_embd / n_head)
    pub mlp_hidden: usize,     // MLP intermediate size
    pub n_layer: usize,        // number of transformer layers
    pub n_kv_head: usize,      // number of K/V heads (≤ n_head for GQA)
    pub bos_token: usize,
    pub temperature: f32,
    pub draft_lookahead: usize,
    pub tree_budget: usize,
    pub parallel_threshold: usize,  // skip rayon if n_embd ≤ this
    pub lora_rank: usize,           // LoRA adapter rank (Plan 008)
    pub lora_alpha: f32,            // LoRA scaling factor
    pub lora_dropout: f32,          // LoRA dropout probability
    pub lora_targets: Vec<String>,  // which projections to apply LoRA
    pub screening_threshold: f32,   // hard-trim cutoff for ScreeningPruner (Plan 021)
    pub sparse_threshold: f32,      // use sparse_mlp when alive ratio ≤ this (Plan 022)
    pub early_exit_patience: usize, // AutoTTS early exit patience (Plan 026)
    pub early_exit_gap: f32,        // AutoTTS early exit confidence gap
    // MTP Drafter thresholds (Plan 055: Gemma 4 MTP)
    pub mtp_activation_threshold: usize,    // enable MTP when n_embd >= this
    pub mtp_cluster_vocab_threshold: usize, // enable cluster LM head when vocab_size >= this
    pub mtp_shared_kv_prompt_threshold: usize, // enable shared KV for prompt when pos >= this
    pub mtp_cluster_size: usize,            // cluster size for round-robin vocab mapping
    // HLA Attention (Plan 057: Higher-order Linear Attention)
    pub hla_mode: HlaMode,                  // Standard, Hla, Ahla
    pub hla_normalize: bool,                // normalize HLA output
    pub hla_decay: f32,                     // decay factor for HLA state
    // D2F Discrete Diffusion Forcing (Plan 066)
    pub mask_token: usize,                  // mask token ID for dLLM
    pub attention_mode: AttentionMode,      // Causal, Bidirectional, BlockCausal, SpKv
    // SP-KV self-pruned KV attention (Plan 070)
    pub sp_kv_window: usize,               // sliding window size for SP-KV
    pub sp_kv_threshold: f32,              // gate threshold for SP-KV utility predictor
    pub sp_kv_predictor_hidden: usize,     // hidden dim for utility predictor MLP
    pub sp_kv_predictor_lr_mult: f32,      // learning rate multiplier for predictor
    // Gemma 2 architecture (Plan 087)
    pub model_arch: ModelArchitecture,      // Generic, Gemma2
    pub rms_norm_eps: f64,                  // epsilon for RMSNorm (1e-5 default, 1e-6 for Gemma2)
    pub rms_norm_offset: bool,              // add offset in RMSNorm (Gemma2: true)
    pub tied_embeddings: bool,              // share wte and lm_head (Gemma2: true)
    pub use_rope: bool,                     // rotary position embeddings (Gemma2: true)
    pub rope_theta: f32,                    // RoPE base frequency
    pub post_norm: bool,                    // post-attention norm (Gemma2: true)
    pub attn_logit_softcapping: f32,        // cap attention logits (Gemma2: 50.0)
    pub final_logit_softcapping: f32,       // cap final logits (Gemma2: 30.0)
    pub weight_dtype: WeightDtype,          // F32, F16, BF16
    // PTRM width scaling (Plan 083)
    pub width_rollouts: usize,              // number of parallel rollouts
    pub early_stop_threshold: f32,          // stop early when reward exceeds this
    // EqR Convergence Selection (Plan 119)
    pub convergence_selector: ConvergenceSelector, // rollout selection strategy
    // D2F block size for discrete diffusion forcing
    pub d2f_block_size: usize,              // block size for D2F diffusion
    // MLS Multi-Layer Sum aggregation (Plan 104: Research 68)
    pub mls_layers: usize,                  // number of last layers to aggregate (0 = disabled)
    // LT2 Looped Inference Pipeline (Plan 108, Research 73)
    pub loop_mode: LoopMode,                // None or WeightShared { loop_count }
    pub hybrid_pattern: HybridPattern,      // Uniform, Interleave, Bookend
    pub gated_attn: bool,                   // whether to use SDPA output gate
}
```
- All configs constructed via factory methods: `Config::micro()`, `Config::micro_lora()`, `Config::draft()`, `Config::game()`, `Config::game_go()`, `Config::gemma2_2b()`, `Config::micro_dllm()`, `Config::bpe()`, `Config::bpe_draft()`, `Config::small_target()`, `Config::gqa_draft()`
- Validation: `n_head % n_kv_head == 0`, `n_embd == n_head * head_dim`
- `kv_dim()` helper returns `n_kv_head * head_dim`

### Key Enums (`crates/katgpt-core/src/types.rs`)

```rust
#[repr(u8)]
pub enum ConvergenceSelector {
    BestQ,          // Highest cumulative relevance (default)
    MajorityVote,   // Most common path across rollouts (mode@K)
    Top1Converged,  // Smallest residual ∥p_{d+1} − p_d∥ (EqR proxy)
    BtRank,         // Pairwise Bradley-Terry ranking (requires `bt_rank` feature)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DashAttnConfig {
    pub chunk_size: usize,          // tile size for chunked attention
    pub alpha: f32,                 // mixing coefficient
    pub scaling_factor: f32,        // attention scale override
    pub sigma: f32,                 // smoothing parameter
    pub estimate_diagonal: bool,    // whether to estimate diagonal terms
}

#[repr(u8)]
pub enum DeltaRoutingMode {
    Off,           // No delta routing (standard layer-by-layer)
    DeltaBlock,    // Route accumulated block deltas
    DeltaAttnRes,  // Route attention residual deltas
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeltaRoutingConfig {
    pub mode: DeltaRoutingMode,     // routing mode
    pub block_size: usize,          // layers per block (default 4)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopMode {
    #[default]
    None,                                    // standard single-pass
    WeightShared { loop_count: usize },      // T-pass weight-shared loop
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HybridPattern {
    #[default]
    Uniform,                                  // all layers use same attention
    Interleave { full_ratio: usize },         // every Nth layer is full SDPA
    Bookend,                                  // first+last layers are full SDPA
}

#[derive(Clone, Debug)]
pub struct SdpaOutputGate {
    pub w_gate: Vec<f32>,    // [n_embd] sigmoid gate weights (zero-init)
}
// forward(&self, attn_out, n_embd) — applies sigmoid gate after SDPA

#[derive(Clone, Debug)]
pub struct ResidualGate {
    pub gates: Vec<f32>,     // [loop_count] per-loop learned gate ρ_τ (zero-init)
}
// new(loop_count, n_embd) — creates zero-init gates

// Feature-gated: `sr2am_configurator`
pub enum PlanningDecision {
    PlanNew,      // reset tree, full budget (high uncertainty)
    PlanExtend,   // keep tree, extend depth (moderate uncertainty)
    PlanSkip,     // skip tree search, direct sample (low uncertainty)
}

// Feature-gated: `sr2am_configurator`
pub struct ConfiguratorContext {
    pub domain: usize,        // domain index from bandit infrastructure
    pub entropy_bin: usize,   // coarse entropy bin: floor(entropy * 10.0), 0..9
}
```


```rust
#[repr(u8)]
pub enum HlaMode {
    Standard,  // SDPA with KV cache (default)
    Hla,       // Symmetric second-order linear attention — O(1) per-token memory
    Ahla,      // Asymmetric second-order linear attention — lower state cost
}

#[repr(u8)]
pub enum AttentionMode {
    Causal,       // Standard autoregressive (default)
    Bidirectional, // Attend to ALL positions — dLLM masked prediction
    BlockCausal,  // Bidirectional within block, causal across blocks — D2F student
    SpKv,         // Self-pruned key-value attention with learned utility (Plan 070)
    SpKvQuant,    // SP-KV + Quantized KV fusion (Plan 070 Phase 3, Task T12)
    DashAttn,     // Chunked linear attention (Research 077, DashAttnConfig)
}

#[repr(u8)]
pub enum ModelArchitecture {
    Generic,  // Default GPT-2 style
    Gemma2,   // Gemma 2 architecture (Plan 087)
}

#[repr(u8)]
pub enum WeightDtype {
    F32,   // Full precision (default)
    F16,   // Half precision
    BF16,  // Bfloat16
}
```

### InferenceOverrides (`crates/katgpt-core/src/types.rs`)

Runtime override fields that can be applied per-inference call without modifying the base `Config`:

```rust
pub struct InferenceOverrides {
    pub tree_budget: Option<usize>,
    pub temperature: Option<f32>,
    pub draft_lookahead: Option<usize>,
    pub parallel_threshold: Option<usize>,
    pub screening_threshold: Option<f32>,
    pub sparse_threshold: Option<f32>,
    pub early_exit_patience: Option<usize>,
    pub early_exit_gap: Option<f32>,
    // MTP Drafter overrides (Plan 055)
    pub mtp_activation_threshold: Option<usize>,
    pub mtp_cluster_vocab_threshold: Option<usize>,
    pub mtp_shared_kv_prompt_threshold: Option<usize>,
    pub mtp_cluster_size: Option<usize>,
    pub mtp_min_output_tokens: Option<usize>,  // skip MTP when remaining tokens < threshold (Plan 117 T15)
    pub mtp_cluster_topk: Option<usize>,       // compute logits for top-K clusters (Plan 117 T22)
    // SP-KV inference-time threshold knob (Plan 070)
    pub sp_kv_threshold: Option<f32>,
    // PTRM width scaling (Plan 083)
    pub width_rollouts: Option<usize>,
    pub early_stop_threshold: Option<f32>,
    // EqR Convergence Selection (Plan 119)
    pub convergence_selector: Option<ConvergenceSelector>,
    // MLS Multi-Layer Sum override (Plan 104)
    pub mls_layers: Option<usize>,
    // Drafter LoRA path (Plan 117: MTP LoRA Drafter)
    pub drafter_lora_path: Option<std::path::PathBuf>,
    // SR²AM horizon truncation override (Plan 112 T11)
    pub max_plan_horizon: Option<usize>,
}
```

Overrides are merged onto a base `Config` at inference time, allowing per-request parameter tuning without cloning or mutating the shared config.

### InferenceResult (`crates/katgpt-core/src/types.rs`)

Output of a single inference pass with reward signal for feedback loop:

```rust
pub struct InferenceResult {
    pub domain: String,
    pub reward: f32,
    pub tree_budget_used: usize,
    pub budget_level: u8,
    pub prompt_hash: u64,
    pub output: String,
    pub timestamp: i64,
    pub screened: bool,
    // Feature-gated: `sr2am_configurator` (Plan 112)
    pub planning_decision: Option<PlanningDecision>,  // SR²AM planning decision
    pub plan_horizon_used: usize,                     // actual horizon after entropy truncation
}
```

### QuantizedKVCache (`src/types.rs`)

Shared interface for quantized KV caches, katgpt-rs–specific (not in katgpt-core):

```rust
pub trait QuantizedKVCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]);
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]);
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    fn reset(&mut self);
    fn pos(&self) -> usize;
    fn set_pos(&mut self, pos: usize);
}
```

Enables `forward_quantized` to work with any compression backend (TurboQuant, SpectralQuant, or future methods).

## TransformerWeights (`transformer.rs`)
```rust
pub struct TransformerWeights {
    pub wte: Vec<f32>,              // [vocab_size, n_embd] — token embedding
    pub wpe: Vec<f32>,              // [block_size, n_embd] — position embedding
    pub lm_head: Vec<f32>,          // [vocab_size, n_embd] — output projection
    pub layers: Vec<LayerWeights>,  // per-layer weights (n_layer entries)
    pub mtp_activation_proj: Option<Vec<f32>>,  // MTP target activation projection (Plan 055)
    pub mtp_cluster_classifier: Option<Vec<f32>>, // MTP cluster classifier (Plan 055)
    pub mtp_cluster_map: Option<Vec<usize>>,     // MTP vocab cluster mapping (Plan 055)
}

pub struct LayerWeights {
    pub attn_wq: Vec<f32>,   // [n_embd, n_embd]
    pub attn_wk: Vec<f32>,   // [n_embd, kv_dim]
    pub attn_wv: Vec<f32>,   // [n_embd, kv_dim]
    pub attn_wo: Vec<f32>,   // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,    // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,    // [n_embd, mlp_hidden]
}
```
- Weight init: Kaiming-style `rng.normal() * sqrt(2 / (n_embd * n_layer))`
- Embedding init: `sqrt(2 / n_embd)`
- `TransformerWeights::new(config, rng)` creates all layers

## ForwardContext (`transformer.rs`)
Pre-allocated scratch buffers for zero-allocation forward passes:
```rust
pub struct ForwardContext {
    x: Vec<f32>,              // [n_embd] — hidden state (mutated in-place)
    q: Vec<f32>,              // [n_embd]
    k: Vec<f32>,              // [kv_dim]
    v: Vec<f32>,              // [kv_dim]
    attn_out: Vec<f32>,       // [n_embd]
    hidden: Vec<f32>,         // [mlp_hidden]
    xr: Vec<f32>,             // [n_embd] — residual buffer 1
    xr2: Vec<f32>,            // [n_embd] — residual buffer 2
    scores: Vec<f32>,         // [block_size] — attention scores
    logits: Vec<f32>,         // [vocab_size]
    pub hidden_state: Vec<f32>, // [n_embd] — snapshot before lm_head (for REST/Validator)
    // Feature-gated buffers (allocated once, zero runtime cost when unused):
    lora_buf: Vec<f32>,       // [rank] — LoRA intermediate (always allocated)
    // #[cfg(feature = "sparse_mlp")]
    active_indices: Vec<usize>, // [mlp_hidden] — alive neuron indices (Plan 022)
    // #[cfg(feature = "sparse_mlp")]
    active_values: Vec<f32>,    // [mlp_hidden] — alive neuron values (Plan 022)
    // MTP Drafter buffers (Plan 055)
    mtp_context_buf: Vec<f32>,    // MTP projection intermediate buffer
    // TurboQuant buffers
    tq_dequant_pos: Vec<f32>,     // dequantized KV for current position
    // Paged KV cache: pre-allocated flat buffers for attention computation
    paged_flat_key: Vec<f32>,     // [block_size * kv_dim]
    paged_flat_value: Vec<f32>,   // [block_size * kv_dim]
    // Raven: pre-allocated query buffer for per-head slot attention
    raven_query_buf: Vec<f32>,    // [kv_dim]
    // Quantized KV cache incremental dequant tracking
    dequant_pos: Vec<usize>,      // [n_layer] last dequantized position per layer
    // Delta routing (Plan 097, feature: `delta_routing`)
    block_deltas: Vec<Vec<f32>>,  // [n_blocks][n_embd] accumulated deltas per block
    delta_routing_logits: Vec<f32>, // [max_sources] routing logits temp buffer
    // CODA fused kernels (Plan 103, feature: `coda_fusion`)
    coda_partial_sums: Vec<f32>,  // [1] single-block RMS sum of squares
    // MLS Multi-Layer Sum (Plan 104, feature: `mls_aggregate`)
    mls_buf: Vec<f32>,            // [n_embd] accumulator for last K layer residuals
    mls_count: usize,             // how many layers accumulated
    // Tiled attention (Plan 115, feature: `tiled_attention`)
    tiled_q: Vec<f32>,            // [block_size × n_embd] repacked queries per head
    tiled_k: Vec<f32>,            // [block_size × kv_dim] repacked keys per kv group
    tiled_v: Vec<f32>,            // [block_size × kv_dim] repacked values per kv group
    tiled_out: Vec<f32>,          // [block_size × n_embd] tiled output before transpose
}
```
- Created once, reused across calls via `ctx.reset()`
- `hidden_state` is copied from `x` before lm_head projection — "free embedding" for vector search
- `lora_buf` avoids per-projection LoRA allocation; fused into `lora_apply()` in-place
- Sparse MLP buffers pack alive ReLU neurons for `sparse_matmul()` — only used when `alive_ratio ≤ sparse_threshold`

## MultiLayerKVCache (`transformer.rs`)
```rust
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
}
pub struct KVCache {
    pub key: Vec<f32>,    // [block_size, kv_dim]
    pub value: Vec<f32>,  // [block_size, kv_dim]
}
```
- One KVCache per layer
- `kv_dim = n_kv_head * head_dim` (may be < n_embd with GQA)
- `reset()` clears all layers
- `snapshot(pos, config)` → `KVSnapshot` (copies only filled slots `[0..pos*kv_dim]`)
- `restore(snapshot, config)` — rollback to earlier state

## Forward Pass (`transformer.rs`)

`forward()` is the **public API** — it delegates to internal `forward_base()` with feature-appropriate parameters:

```rust
// Public API — handles domain_latent feature gating internally
pub fn forward(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &mut [f32]  // logits

// Internal — called by forward(), forward_prefill(), and generate_with_prefill()
// Accepts optional LoRA adapter and domain latent
fn forward_base(
    ctx, weights, cache, token, pos, config,
    lora: Option<&LoraAdapter>,        // cfg: always available
    domain_latent: Option<&DomainLatent>,  // cfg(feature = "domain_latent")
) -> &mut [f32]
```

Pipeline (inside `forward_base`):
1. **Embedding**: `x = wte[token] + wpe[pos]`
2. **Layer loop** (n_layer iterations):
   a. RMSNorm → QKV projection (GQA: K/V use kv_group)
   b. Store K/V in per-layer cache at position `pos`
   c. Multi-head attention (fused: score → softmax → weighted value)
   d. Output projection + residual add
   e. RMSNorm → MLP (matmul_relu + matmul) + residual add
   f. *(domain_latent)* At layer `n_layer / 2`: inject `DomainLatent` into K/V
3. **Snapshot**: `hidden_state = x` (before lm_head)
4. **LM Head**: `logits = lm_head @ x`

### GQA (Grouped-Query Attention)
When `n_kv_head < n_head`, K/V heads are shared:
- `kv_group = q_head * n_kv_head / n_head`
- K/V projection outputs `kv_dim` instead of `n_embd`
- 4× KV cache reduction for `n_head=8, n_kv_head=2`

## Math Kernels (`crates/katgpt-core/src/types.rs`)
All hot-path kernels are `#[inline(always)]` with `unsafe get_unchecked`:
- `matmul(out, w, x, rows, cols)` — out = W @ x — SIMD-accelerated via `simd_dot_f32` (Plan 060)
- `matmul_relu(out, w, x, rows, cols)` — fused matmul + ReLU — SIMD-accelerated with fused ReLU zero-clamp (Plan 060)
- `sparse_matmul(out, w, x, rows, cols, active_indices, active_values)` — skip dead ReLU neurons (Plan 022)
- `softmax(x)` — in-place, one-pass exp+sum, uses `inv_sum` multiply
- `softmax_scaled(x, scale)` — scaled softmax for attention (divides by sqrt(head_dim) before exp)
- `rmsnorm(x)` — in-place, two-pass with `inv_rms` multiply
- `attention_head(...)` — fused: score → softmax → weighted value (avoids separate softmax write)
- `sample_token(logits, rng)` — categorical sampling
- `lora_apply(output, lora, input, lora_buf)` — in-place LoRA delta: `output += (α/r) × B @ (A @ input)`
- `gegelu(hidden, gate, up)` — GeGLU activation for Gemma 2 MLP: `GELU(gate) * up`
- `gegelu_tanh(hidden, gate, up)` — GeGLU with tanh approximation
- `rmsnorm_with_gamma(x, gamma)` — RMSNorm with learnable gain parameter
- `rmsnorm_with_gamma_eps(x, gamma, eps)` — RMSNorm with gain and custom epsilon

## SIMD Kernels (`crates/katgpt-core/src/simd.rs`, Plan 060)

Runtime SIMD detection and dispatch for hot-path operations:
- `SimdLevel` enum: `Scalar`, `Neon` (ARM), `Avx2` (x86_64)
- `simd_level()` — runtime detection of available SIMD level
- `simd_dot_f32(a, b, len)` — NEON `vfmaq_f32` / AVX2 `_mm256_mul_ps` dot product
- `simd_outer_product_acc(acc, a, b, m, n)` — rank-1 update for HLA SK, CQV, PKV
- `simd_matmul_rows(out, w, x, rows, cols)` — row-major matmul via SIMD dot
- `simd_matmul_relu_rows(out, w, x, rows, cols)` — SIMD matmul + fused ReLU clamp
- `simd_fused_decay_write(dst, decay, src, write)` — fused decay+write for HLA state update
- `maxsim_score(queries, documents, lq, ld, dim)` — MaxSim late-interaction scoring
- `maxsim_score_packed(queries, query_offsets, documents, doc_offsets, pair_q_ids, pair_d_ids, dim)` — batched MaxSim for packed representations
- `simd_add_into(dst, a, b)` — SIMD-accelerated element-wise vector add
- No dependencies — pure `core::arch::{aarch64, x86_64}` intrinsics
- Zero-cost dispatch: compile-time `#[cfg(target_arch)]` + runtime level check

## Additional Forward Variants (`transformer.rs`)

| Function | Description |
|----------|-------------|
| `forward_prefill(ctx, prefill, weights, cache, tokens, config, lora, domain_latent)` | Bidirectional prefill — all prompt tokens attend to all others (Plan 025) |
| `forward_paged(ctx, weights, paged_cache, token, pos, config, seq_idx)` | Paged KV cache forward — copy-on-write branch isolation |
| `forward_raven(ctx, weights, raven_cache, token, pos, config)` | Raven RSM forward — slot-based O(1) routing attention |
| `forward_turboquant(ctx, weights, tq_cache, token, pos, config)` | TurboQuant forward — bit-packed KV cache with dequantize-on-read |
| `forward_hla(ctx, weights, hla_cache, token, pos, config)` | Symmetric second-order HLA — O(d²) constant-state attention, SIMD-accelerated (Plan 057/060, `hla_attention`) |
| `forward_ahla(ctx, weights, ahla_cache, token, pos, config)` | Asymmetric AHLA — O(d·dv) constant-state attention, SIMD-accelerated (Plan 057/060, `hla_attention`) |
| `forward_with_domain_latent(ctx, weights, cache, token, pos, config, dl)` | Convenience wrapper — `forward_base` with domain latent only (no LoRA) |
| `forward_sp_kv(ctx, weights, sp_kv_cache, token, pos, config, predictors, bias)` | SP-KV self-pruned KV forward — utility-gated attention with learned predictor MLP (Plan 070, `sp_kv`) |
| `forward_looped(ctx, weights, cache, ahla_cache, token, pos, config, residual_gate, sdpa_gate)` | LT2 looped forward — weight-shared T-pass loop with hybrid SDPA+AHLA dispatch (Plan 108, `lt2_looped`) |
| `forward_coda(ctx, weights, cache, token, pos, config, lora, domain_latent)` | CODA-fused forward — single-pass SIMD kernels eliminate intermediate buffer writes (Plan 103, `coda_fusion`) |
| `forward_decode_stage(ctx, weights, cache, token, pos, config, stage)` | DecodeStage dispatch — routes to draft/target/coda based on stage enum |
| `depth_route(residual, sources, query_weight, norm_weight, logits_buf, n_embd)` | Delta routing — softmax-weighted blend of accumulated block deltas (Plan 097) |
| `depth_route_weights(sources, query_weight, norm_weight, n_embd)` | Returns routing weights without mutation (for analysis/logging) |

> **Plan 059 Note**: HLA is inference-only — SDPA→HLA distillation via LoRA shows KL divergence does NOT converge. HLA provides streaming O(1) attention for inference but cannot be trained to approximate SDPA outputs. Use DeltaMemoryState for facts/retrieval.

## LT2 Looped Forward Pass (`transformer.rs`, Plan 108)

Weight-shared T-pass loop: same layer weights applied T times, yielding effective depth T×n_layer with no extra parameters. Hybrid dispatch mixes SDPA (full attention) and AHLA (O(1) constant-state) layers per loop iteration.

```
Input: x = wte[token] + wpe[pos]
For τ = 1..T:
  Save prev_h = x
  For ℓ = 1..n_layer:
    is_full = match hybrid_pattern {
      Uniform    => true,
      Interleave{full_ratio:5} => (ℓ % 5) == 4,
      Bookend    => ℓ == 0 || ℓ == n_layer-1,
    }
    h' = h + Mixer_ℓ(h, is_full)    // AHLA or SDPA
    h  = h' + FFN_ℓ(h')             // shared FFN
    if gated_attn && is_full: h = SdpaOutputGate(h)  // sigmoid gate, zero-init
  h = h̃ + ρ_τ ⊙ prev_h             // per-loop residual gate (zero-init)
Output: lm_head(h)
```

**Key types** (`crates/katgpt-core/src/types.rs`):

| Type | Description |
|------|-------------|
| `LoopMode` | `None` (standard) or `WeightShared { loop_count: T }` |
| `HybridPattern` | `Uniform`, `Interleave { full_ratio }`, `Bookend` |
| `ResidualGate` | Per-loop learned gate ρ_τ — zero-init → first iteration is identity |
| `SdpaOutputGate` | Sigmoid gate after SDPA before Wo — zero-init → sigmoid(0) = 0.5 neutral |

**Memory scaling**: AHLA layers use O(d·dv) constant state (no growth with L or T). SDPA layers use O(L·d) KV cache (no growth with T). Hybrid 1:4 achieves ~95% throughput of pure SDPA T=4 with 80% constant-memory layers.

**Feature gate**: `lt2_looped = ["hla_attention"]` (default-on). GOAT: 11/11 proofs pass.

## MTP Projection (`transformer.rs`, Plan 055)

Multi-Token Prediction projection weights for draft model acceleration:
- `MtpProjection` — Projection weights for target-activation-based MTP drafting
- `project_target_activation()` — Projects hidden state to draft token logits
- `cluster_map_round_robin()` — Round-robin vocab cluster assignment
- `cluster_map_from_embeddings()` — Embedding-similarity-based cluster assignment
- Threshold-gated: features activate only when config thresholds are met (see `13_mtp_threshold_guide.md`)

## Generate (`transformer.rs`)
```rust
pub fn generate(ctx, cache, weights, config, rng, token, n_tokens) -> Vec<usize>
pub fn generate_into(ctx, cache, weights, config, rng, tokens, n_tokens)  // zero-alloc variant
pub fn generate_batch(ctx, cache, weights, config, rng, token, n_tokens, n_samples) -> Vec<Vec<usize>>
pub fn generate_with_prefill(
    ctx, prefill, cache, weights, config, rng,
    prompt_tokens, n_tokens,
    // Optional per-call overrides:
    lora_pair: Option<&LoraPair>,          // reader→writer LoRA switching
    domain_latent: Option<&DomainLatent>,  // mid-layer domain conditioning
) -> Vec<usize>
```
- Autoregressive: sample → feed back → repeat
- `generate_into` reuses pre-allocated buffers (zero-alloc hot path)
- `generate_batch` uses Rayon `par_iter` with per-worker contexts
- `generate_with_prefill` runs bidirectional prefill (reader LoRA) then switches to causal decode (writer LoRA), with optional domain latent injection
- `tokens_to_string(tokens, config)` — converts token IDs back to string via `id_to_vocab` lookup

## PagedKVCache (implemented, DDTree integration pending)
```rust
pub struct PagedKVCache {
    pages: Vec<Vec<f32>>,                    // pool of fixed-size pages
    layer_page_tables: Vec<Vec<Vec<usize>>>, // per-layer, per-sequence page indices
    free_pages: Vec<usize>,                  // reuse pool
    kv_dim: usize,
}
```
- Fixed `PAGE_SIZE = 16` tokens per page
- `fork(seq_idx, fork_at_pos)` — copy-on-write branch (shares prefix pages)
- Designed for DDTree branch exploration (each branch = one sequence)
- Deferred integration: currently DDTree uses flat `snapshot()/restore()` instead

## KVSnapshot
```rust
pub struct KVSnapshot {
    pub layers: Vec<KVLayerSnapshot>,
    pub pos: usize,
}
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,    // [pos, kv_dim]
    pub value: Vec<f32>,  // [pos, kv_dim]
}
```
- Cheap: copies only filled slots `[0..pos*kv_dim]` per layer
- Used in speculative rollback: snapshot before verify, restore on reject

## ScreeningPruner: Absolute Relevance (Plan 021)

Distilled from ["Screening Is Enough"](https://arxiv.org/abs/2604.01178) — upgrades binary pruning to **graded relevance**:

```rust
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}
```

Score formula: `blended = parent_score + ln(P_llm) + ln(R)`

| Relevance R | ln(R) | Effect |
|---|---|---|
| 1.0 | 0.0 | No penalty — perfect match |
| 0.5 | -0.69 | Soft penalty — mediocre match |
| 0.0 | -∞ | **Hard trim** — branch killed |

`ConstraintPruner` adapts via `BinaryScreeningPruner(pruner)` (R ∈ {0.0, 1.0}). `WasmPruner` implements `ScreeningPruner` natively — loads optional WASM `relevance` export (Q16.16 fixed-point), falls back to binary `is_valid` if missing.

`config.screening_threshold` (default `0.0`) controls hard-trim cutoff. Set `> 0.0` to aggressively trim low-relevance branches.

## Freeze/Thaw (`src/pruners/freeze.rs`, Plan 092)

Shared freeze/thaw disk I/O for `repr(C)` bandit knowledge structs. Zero-dependency binary persistence — raw `std::fs::write`/`read` on `repr(C)` data with magic bytes + version validation on load. No serde/bincode needed.

```rust
pub fn save_frozen<T>(path: &Path, data: &T) -> Result<(), String>
pub fn load_frozen<T>(path: &Path) -> Result<T, String>
```

### Key Fix: Per-Move Reward (Issue 065)

Initial freeze/thaw showed **negative** knowledge transfer (-3pp win rate). Root cause: binary game-end reward + low α=0.3 blended with per-move signal, causing all Q-values to converge to ~0.25 when losing 86% of games (no differentiation).

**Fix:** `HL_PER_MOVE_ALPHA = 1.0` (pure per-move reward, no game-end blending) + `HL_DELTA_AMPLIFICATION = 10.0` (amplifies raw heuristic delta ±0.01–0.06 → ±0.1–0.6).

Results (GoHL vs Validator, 100 rounds × 3 phases):
| Metric | Frozen | Baseline | Δ |
|--------|--------|----------|---|
| Win Rate | 25% | 14% | **+11pp ✅** |
| Avg Score | -13.3 | -16.8 | **+3.5 ✅** |

Q-values after learning:
- Corner: 0.80, Side: 0.64, Center: 0.74, Capture: 0.75, Defense: 0.40, Extend: 0.48, Influence: 0.59, Pass: 0.00
- 2× spread (Corner vs Defense) vs old flat ~0.25

Learning vs Random also verified: Q-values differentiate properly (spread > 0.1) with α=1.0, unlike old binary reward that collapsed all to ~0.85.

Run: `cargo run --example go_08_self_play_freeze --features go`

## SpeculativeVerifier (Strategy Pattern)

Based on [Algorithm 1 from Leviathan et al. 2022](https://arxiv.org/pdf/2211.17192) — the verification strategy is swappable via trait:

```rust
pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&mut self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

| Verifier | Availability | What it does |
|----------|--------------|--------------|
| `SimulatedVerifier` | always compiled | DFlash/AR draft → DDTree → simulated acceptance cap → bonus token from last marginal |
| `LeviathanVerifier` | always compiled | AR draft → target model p/q scoring → rejection sampling → residual distribution → bonus from target p(x). Proves Algorithm 1 works end-to-end. |
| `D2fDrafterVerifier` | `tri_mode` feature | D2F diffusion drafts in parallel (bidirectional within block) → AR verifies with causal attention (Plan 089: Tri-Mode "self-speculation") |

`SimulatedVerifier` is fast (no target model). `LeviathanVerifier` is the full Algorithm 1 — mathematically proven distribution-preserving, but needs large model asymmetry to be faster than pure AR.

## PPoT: Logit-Parameterized CPU Resampling (Plan 026 + 027)

After DFlash produces marginals and DDTree rejects all paths, PPoT identifies high-entropy positions in the saved marginals and resamples variant token sequences using **only CPU** — no additional GPU forward passes. Resampled paths are screened through `ScreeningPruner` for verification. This activates only on failure (zero overhead on success path).

Plan 027 extends baseline with TRT-inspired adaptive rescue: rejection memory (ring buffer of "don't" insights), per-sample strategy cycling across `TokenRule` variants, and self-consistency ranking for multi-valid variant selection. Knowledge accumulates within a generation session, biasing future resampling toward historically successful positions and rules.

```rust
pub enum TokenRule {
    Digit,      // prefer digit tokens
    Compare,    // prefer comparison operators
    Arithmetic, // prefer arithmetic operators
    Augment,    // prefer augmented assignment
    All,        // no preference
}
```

## Prompt Router: Batch-Level Domain Routing (Plan 023)

Inspired by [EMO: Pretraining Mixture of Experts for Emergent Modularity](https://arxiv.org/abs/2406.08732) — document-level routing constraints force experts to learn high-level semantic domains instead of syntax.

1. **Classify once** — `KeywordRouter` scores the prompt against domain keywords (V1, ~80% accuracy; embedding-based V2 via anyrag is planned)
2. **Select expert** — `ExpertRegistry` returns a `Box<dyn ScreeningPruner>` + optional LoRA path for the matched domain
3. **Lock for generation** — the selected `ScreeningPruner` is passed to `build_dd_tree_screened()`, preventing domain drift

```rust
let router = KeywordRouter::new(config.domain.clone());
let registry = ExpertRegistry::from_config(&config, pruner_dir);

let decision = router.route("solve this sudoku puzzle");
let expert = registry.get_expert(&decision.domain);
// expert.pruner is locked for the entire DDTree generation
```

Domains are defined in `domains.toml` — platform manages expert bundles via Web UI or MCP agent.

## Embedding Router: KV Cache Priming (Plan 024)

Extends keyword routing with **semantic embedding retrieval** from anyrag. When a user edits a known file, the system retrieves the most relevant document embedding, projects it to the draft model's hidden dimension, and injects it as KV cache priming context via `dflash_predict_conditioned_with`.

**Three-tier fallback** (graceful degradation when anyrag is unavailable):

```
1. Embedding search (POST /search/embedding)  ~200ms
   ↓ on failure
2. Domain classify (POST /classify/domain)     ~100ms
   ↓ on failure
3. KeywordRouter (local, no network)            <1ms
```

```rust
let router = EmbeddingRouter::new(
    embedding_config, domains, Box::new(TruncatePadProjector),
);

// Sync: delegates to KeywordRouter (no network)
let decision = router.route("fn validate_token(");

// Async: tries anyrag embedding search, falls back to keyword
let decision = router.route_async("fn validate_token(").await;

if let Some(embedding) = &decision.embedding {
    let projected = router.project_embedding(embedding, draft_config.n_embd);
    speculative_step_embedding_conditioned(&weights, &config, token, pos, &projected, &mut rng);
}
```

**Separation from target model conditioning:** `speculative_step_conditioned_with` uses the target model's hidden state (syntactic alignment). `speculative_step_embedding_conditioned` uses a retrieved embedding (semantic alignment). These are complementary signals.

## Bidirectional Prefill + Modality LoRA Switching (Plan 025)

Distilled from [ZAYA1-VL-8B Technical Report](https://arxiv.org/abs/2504.02268) — two production techniques adapted for the Python→Rust translation pipeline:

### 1. Bidirectional Prefill

During prefill, prompt tokens (Python code + anyRAG docs) attend to ALL other prompt tokens — no causal mask. Code is non-linear; a function body references a struct 3,000 tokens earlier. Generation tokens still use causal attention. Zero overhead on the decode hot path — prefill runs once per request.

### 2. Modality LoRA Switching

Load two LoRA adapters per domain — a `reader_lora` (active during prefill) and a `writer_lora` (active during decode). The switch is a reference swap at the prefill→decode boundary. Zero data movement.

```
  tokens[0..prompt_len]                    tokens[prompt_len..]
        │                                         │
   ┌────┴────┐                              ┌─────┴─────┐
   │ PREFILL │  bidirectional attention     │  DECODE   │  causal attention
   │         │  reader_lora active          │           │  writer_lora active
   └────┬────┘                              └─────┴─────┘
        │ KV cache populated                      │ generates tokens
        └──────────── shared KV cache ────────────┘
```

### LoraPair & PrefillContext

```rust
pub struct LoraPair {
    pub reader: Option<LoraAdapter>,  // active during bidirectional prefill
    pub writer: Option<LoraAdapter>,  // active during causal decode
}

pub struct PrefillContext {
    pub hidden: Vec<f32>,  // [prompt_len × n_embd] — pre-allocated once
}
```

Two-phase per layer (zero-copy):

| Phase | What | Buffers |
|-------|------|---------|
| A: KV Fill | Compute K/V for all positions → store in cache | Reuses `ForwardContext` per-position |
| B: Bidirectional Attend | Q attends to K/V[0..prompt_len] via `attention_head(t_n=prompt_len)` | `attention_head` unchanged — caller controls range |

```rust
let mut prefill = PrefillContext::new(&config);

// Bidirectional prefill with reader LoRA + optional domain latent
let logits = forward_prefill(&mut ctx, &mut prefill, &weights, &mut cache,
    &prompt_tokens, &config, lora_pair.reader.as_ref(), domain_latent);

// Causal decode — forward() delegates to forward_base(writer LoRA + domain latent)
let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
// Note: for explicit LoRA control during decode, use generate_with_prefill()
// which handles the reader→writer swap internally.
```

Domain config in `domains.toml`:
```toml
[[domain]]
name = "py2rs"
keywords = ["python", "rewrite", "translate"]
pruner = "syn_validator.wasm"
reader_lora = "python_reader.bin"   # active during bidirectional prefill
writer_lora = "rust_writer.bin"     # active during causal decode
```

## CODA Fusion Kernels (`crates/katgpt-core/src/coda.rs`, Plan 103)

CODA-inspired fused SIMD kernels that algebraically reparameterize matmul+residual+rmsnorm+activation into single-pass SIMD loops, eliminating intermediate buffer writes.

**Key identity (CODA §3.2.1):**
```
RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'
```

This delays the row-wise RMSNorm scale past the next GEMM.

```rust
#[repr(u8)]
pub enum GateActivation {
    Relu,        // max(0, x) — standard 2-layer MLP
    Silu,        // x * sigmoid(x) — LLaMA SwiGLU
    GegeluTanh,  // tanh-approx GELU — Gemma 2 GeGLU
    Gegelu,      // sigmoid-approx GELU — standard GeGLU
}
```

| Kernel | Description |
|--------|-------------|
| `simd_matmul_residual(out_d, out_o, partial_sums, w, x, residual, gamma, bias, rows, cols)` | Fused matmul + residual add + delayed RMSNorm (Plan 103 T3) |
| `compute_rstd(partial_sums, n, eps)` | Compute reciprocal standard deviation from partial sums |
| `simd_matmul_rmsnorm_swiglu(out, x, norm, w_gate, w_up, w_down, rstd, hidden_buf, n)` | Fused RMSNorm + SwiGLU MLP (SiLU activation) |
| `simd_matmul_rmsnorm_activation(out, rstd, hidden, activation, n)` | Apply delayed activation with rstd scaling |
| `simd_matmul_rmsnorm_rope(out, q_buf, k_buf, x, wq, wk, wv, rstd, pos, head_dim, n_heads, theta)` | Fused QKV projection + RoPE with delayed RMSNorm |

**Feature gate:** `coda_fusion`

**Buffer write savings per layer:** ~8 passes (baseline) → ~0 passes (CODA fused).

## Tiled Attention (`crates/katgpt-core/src/attention.rs`, Plan 115)

CPU SIMD tiled flash attention using online-softmax algorithm, adapted from ThunderKittens (Research 077). Processes Q in SIMD-width row tiles, K/V in column tiles — avoids materializing full N×N score matrix.

```
Tile sizes: BR=8 (query rows), BC=128 (key/value columns)
Threshold: tiled path activates when N > 128 (score matrix > L1 cache)
```

| Function | Description |
|----------|-------------|
| `tiled_attention_forward(q, k, v, output, seq_len, head_dim, scale)` | Single-head tiled attention with online-softmax |
| `tiled_attention_batched(q, k, v, output, batch, heads, seq_len, head_dim)` | Multi-head batched via rayon `par_chunks_mut` |

**Online-softmax algorithm (per query tile):**
1. Initialize: `o_tile = 0, max_tile = -inf, norm_tile = 0`
2. For each K/V tile: score → update running max → correction factor → exp → accumulate
3. Final normalize: `o_tile / norm_tile`

**Feature gate:** `tiled_attention`

## Consolidated Traits (`crates/katgpt-core/src/traits.rs`, Plan 107 Phase 0)

Shared traits for game AI and speculative decoding, consolidated from katgpt-rs and riir-engine to eliminate duplication. Both crates depend on `katgpt-core`, so moving traits here requires zero new dependency edges.

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
    fn batch_is_valid(&self, depth, candidates, parent_tokens, results); // default: per-item
}

pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

pub trait GameState: Clone {
    type Action: Clone;
    fn available_actions(&self, player_id: u8) -> Vec<Self::Action>;
    fn advance(&self, action: &Self::Action, player_id: u8) -> Self;
    fn is_terminal(&self) -> bool;
    fn reward(&self, player_id: u8) -> f32;
    fn tick(&self) -> u32;
}

pub trait StateHeuristic<S: GameState> {
    fn evaluate(&self, state: &S, player_id: u8) -> f32;
}

pub trait RolloutPolicy<S: GameState> {
    fn select(&mut self, state: &S, actions: &[S::Action], player_id: u8, rng: &mut Rng) -> usize;
}
```

| Struct | Trait | Description |
|--------|-------|-------------|
| `NoPruner` | `ConstraintPruner` | Allows all tokens (baseline) |
| `BinaryScreeningPruner<P>` | `ScreeningPruner` | Adapter: `ConstraintPruner` → binary `{0.0, 1.0}` relevance |
| `NoScreeningPruner` | `ScreeningPruner` | Returns 1.0 for everything (no penalty) |
| `RandomRolloutPolicy` | `RolloutPolicy` | Uniform random action selection |
| `ActionSpaceLog` | — | Per-tick branching factor metrics for analysis |

Re-exported from both `katgpt-core` and `katgpt-rs`.

LoRA application is fused in-place after each projection: `output += (α/r) × B @ (A @ input)`. Zero intermediate buffers — the delta accumulates directly into the output.