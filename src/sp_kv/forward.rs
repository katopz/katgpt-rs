//! SP-KV forward pass: gated attention with utility-based bias.
//!
//! `attention_head_gated()` is a drop-in replacement for `attention_head()` that adds
//! an optional per-position gate bias to Q·K scores before softmax.
//!
//! ## Gate Bias Semantics
//!
//! | Mode | bias[s] | Effect |
//! |------|---------|--------|
//! | `None` | — | Identical to `attention_head()` |
//! | Soft (training) | `log(u_s + ε)` | Smooth, differentiable masking |
//! | Hard (inference) | `0` or `-∞` | Binary retain/prune |
//! | TAHG (annealing) | Blended | Gradual hardening |
//!
//! Positions within the sliding window always get bias = 0 (unconditionally attended).

use crate::simd::simd_dot_f32;

/// Pre-allocated context for SP-KV forward passes.
///
/// Holds gate bias buffer and utility predictor scratch buffer.
/// Create once, reuse across calls. Zero alloc in hot path.
pub struct SpKvForwardContext {
    /// Gate bias buffer: [block_size]. Rebuilt every forward call.
    gate_bias: crate::sp_kv::types::GateBiasBuffer,
    /// Utility predictor scratch: [predictor_hidden]. Reused across layers.
    predictor_buf: Vec<f32>,
    /// Per-head utilities: [n_kv_heads]. Reused across layers.
    head_utilities: Vec<f32>,
    /// Flat dequantized key buffer: [block_size * kv_dim]. Used by quant fusion path.
    flat_keys: Vec<f32>,
    /// Flat dequantized value buffer: [block_size * kv_dim]. Used by quant fusion path.
    flat_values: Vec<f32>,
}

impl SpKvForwardContext {
    /// Create SP-KV forward context for the given model and SP-KV config.
    pub fn new(
        config: &crate::types::Config,
        sp_kv_config: &crate::sp_kv::types::SpKvConfig,
    ) -> Self {
        let hidden = sp_kv_config.predictor_hidden.max(16);
        let n_kv_heads = config.n_kv_head;
        Self {
            gate_bias: crate::sp_kv::types::GateBiasBuffer::new(config.block_size),
            predictor_buf: vec![0.0; hidden],
            head_utilities: vec![0.0; n_kv_heads],
            flat_keys: Vec::new(),
            flat_values: Vec::new(),
        }
    }

    /// Create context pre-allocated for quantized cache fusion.
    ///
    /// Flat KV buffers are sized `[block_size * kv_dim]` for dequantized attention.
    pub fn new_quant(
        block_size: usize,
        kv_dim: usize,
        predictor_hidden: usize,
        n_kv_heads: usize,
    ) -> Self {
        let block_kv = block_size * kv_dim;
        Self {
            gate_bias: crate::sp_kv::types::GateBiasBuffer::new(block_size),
            predictor_buf: vec![0.0; predictor_hidden],
            head_utilities: vec![0.0; n_kv_heads],
            flat_keys: vec![0.0; block_kv],
            flat_values: vec![0.0; block_kv],
        }
    }

    /// Ensure flat KV buffers are allocated for quant path.
    ///
    /// No-op if already sized (e.g. created via [`new_quant`]).
    /// Lazily allocates if created via [`new`] and then used for quant fusion.
    fn ensure_quant_bufs(&mut self, block_size: usize, kv_dim: usize) {
        let block_kv = block_size * kv_dim;
        if self.flat_keys.len() < block_kv {
            self.flat_keys.resize(block_kv, 0.0);
            self.flat_values.resize(block_kv, 0.0);
        }
    }
}

// ── Monomorphized bias providers for zero-overhead gate dispatch ──────────
//
// The primary source of gate bias overhead is the `Option<&[f32]>` match
// inside the hot Q·K scoring loop — one branch per position per head per layer.
// By specializing via a trait, the compiler generates two versions:
//   - NoBias:  identical machine code to baseline `attention_head()`
//   - GateBias: fused add with prune-skip, no per-iteration Option check

/// Trait for providing per-position attention bias.
///
/// Enables monomorphization: the compiler emits specialized code per provider,
/// eliminating `Option` branching in the hot Q·K loop.
///
/// | Provider   | Overhead vs baseline | Use case       |
/// |------------|---------------------|----------------|
/// | `NoBias`   | 0% (identical)      | Baseline       |
/// | `GateBias` | <1% (fused add)     | SP-KV decode   |
pub trait BiasProvider {
    /// Read the bias for position `t`. Returns 0.0 for no-bias providers.
    fn bias(&self, t: usize) -> f32;

    /// Whether this provider may produce `-inf` (pruned) positions.
    /// When `false`, the prune-skip branch is eliminated at compile time.
    fn may_prune(&self) -> bool {
        false
    }
}

/// No bias: compiles to identical machine code as baseline `attention_head()`.
pub struct NoBias;

impl BiasProvider for NoBias {
    #[inline(always)]
    fn bias(&self, _t: usize) -> f32 {
        0.0
    }
}

/// Gate bias from a precomputed slice: direct `get_unchecked` access.
///
/// The lifetime ties the borrow to the caller's bias buffer so no copy is needed.
pub struct GateBias<'a> {
    bias: &'a [f32],
}

impl<'a> GateBias<'a> {
    #[inline(always)]
    pub fn new(bias: &'a [f32]) -> Self {
        Self { bias }
    }
}

impl BiasProvider for GateBias<'_> {
    #[inline(always)]
    fn bias(&self, t: usize) -> f32 {
        unsafe { *self.bias.get_unchecked(t) }
    }

    #[inline(always)]
    fn may_prune(&self) -> bool {
        true
    }
}

/// Core attention head with generic bias — monomorphized for zero-overhead dispatch.
///
/// When `B = NoBias`: identical to `attention_head()` in `transformer.rs`.
/// When `B = GateBias`: adds `bias[t]` to each Q·K score, skips `-inf` positions.
///
/// ## Optimizations over naive `Option<&[f32]>`
///
/// 1. **Monomorphization** — `B` is known at compile time; no per-iteration match.
/// 2. **Prune skip** — when `bias[t] == -inf`, the expensive `simd_dot_f32` call
///    is skipped entirely (score → 0 after softmax, contributes nothing).
/// 3. **Direct indexing** — `GateBias` reads via `get_unchecked`; no `Option` unwrap.
///
/// ## SAFETY
///
/// Caller must ensure all indices are in bounds (same as `attention_head_gated`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn attention_head_core<B: BiasProvider>(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    bias: B,
) {
    // Pass 1: compute Q·K scores with bias, skip pruned positions
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let b = bias.bias(t);

        // Pruned position: skip expensive dot product — score → 0 after softmax.
        // For NoBias: may_prune() = false → entire block eliminated at compile time.
        if bias.may_prune() && b == f32::NEG_INFINITY {
            unsafe {
                *scores_buf.get_unchecked_mut(t) = f32::NEG_INFINITY;
            }
            continue;
        }

        let k_off = t * kv_dim + kv_group_offset;
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), hd);
            let k_slice = std::slice::from_raw_parts(key_cache.as_ptr().add(k_off), hd);
            simd_dot_f32(q_slice, k_slice, hd)
        };

        // Fused scale + bias. For NoBias, b=0.0 → compiler elides the add.
        let score = dot * scale + b;

        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }

    // Pass 2: exp(scores - max) and accumulate sum
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation
    let inv_sum = 1.0 / sum;
    for d in 0..hd {
        let mut val = 0.0f32;
        for t in 0..t_n {
            unsafe {
                val += *scores_buf.get_unchecked(t)
                    * inv_sum
                    * *value_cache.get_unchecked(t * kv_dim + kv_group_offset + d);
            }
        }
        unsafe {
            *attn_out.get_unchecked_mut(q_head_offset + d) = val;
        }
    }
}

/// Backward-compatible wrapper: dispatches to monomorphized core.
///
/// Preserves the original `Option<&[f32]>` API. The `Option` match happens
/// **once per call** (not per iteration), so overhead is negligible.
///
/// # Safety
///
/// Delegates to [`attention_head_core`] — same safety requirements:
/// all slice indices must be in bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn attention_head_gated(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    gate_bias: Option<&[f32]>,
) {
    unsafe {
        match gate_bias {
            Some(bias) => attention_head_core(
                q,
                key_cache,
                value_cache,
                attn_out,
                scores_buf,
                q_head_offset,
                kv_group_offset,
                kv_dim,
                hd,
                t_n,
                scale,
                GateBias::new(bias),
            ),
            None => attention_head_core(
                q,
                key_cache,
                value_cache,
                attn_out,
                scores_buf,
                q_head_offset,
                kv_group_offset,
                kv_dim,
                hd,
                t_n,
                scale,
                NoBias,
            ),
        }
    }
}

/// Build per-position gate biases for all positions up to `pos`.
///
/// Convenience function that dispatches to the appropriate mode:
/// - Soft: `log(u_s + ε)` outside window, `0` inside
/// - Hard: `0` if retained or in window, `-∞` otherwise
/// - TAHG: blended soft/hard with annealing
///
/// Writes into `buf.bias[..=pos]`. Positions after `pos` are left unchanged.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn build_gate_biases(
    buf: &mut crate::sp_kv::types::GateBiasBuffer,
    utilities: &[f32],
    retained: &[bool],
    pos: usize,
    window: usize,
    threshold: f32,
    mode: crate::sp_kv::types::SpKvGateMode,
) {
    use crate::sp_kv::types::SpKvGateMode;

    match mode {
        SpKvGateMode::Soft => buf.build_soft(utilities, pos, window),
        SpKvGateMode::Hard => buf.build_hard(utilities, retained, pos, window, threshold),
        SpKvGateMode::Tahg { step, total_steps } => {
            buf.build_tahg(utilities, pos, window, threshold, step, total_steps);
        }
    }
}

/// Check if a position is within the sliding window from the current decode position.
#[inline(always)]
pub fn is_in_window(current_pos: usize, source_pos: usize, window: usize) -> bool {
    current_pos.saturating_sub(source_pos) < window
}

/// SP-KV forward pass: utility-gated attention with conditional KV write.
///
/// Mirrors `forward_base()` from `transformer.rs` with three SP-KV modifications:
///
/// 1. **Utility prediction** (after RMSNorm, before QKV): 2-layer MLP predicts per-KV-head utility
/// 2. **Conditional KV write**: only store if `utility ≥ τ` or position is in sliding window
/// 3. **Gate-biased attention**: `attention_head_gated()` adds `log(u)` bias to Q·K scores
///
/// Pipeline composability: PFlash (prefill) → SP-KV (decode) → TurboQuant (storage).
///
/// # Arguments
///
/// * `ctx` — Pre-allocated forward context (from `transformer.rs`)
/// * `weights` — Transformer weights (embeddings, per-layer, LM head)
/// * `sp_cache` — SP-KV sparse KV cache (replaces `MultiLayerKVCache`)
/// * `predictors` — Per-layer utility predictor weights
/// * `sp_ctx` — SP-KV forward context (gate bias buffer, predictor scratch)
/// * `token` — Current input token index
/// * `pos` — Current sequence position
/// * `config` — Model configuration
/// * `lora` — Optional LoRA adapter
/// * `gate_mode` — Soft (training), Hard (inference), or TAHG (annealing)
/// * `domain_latent` — Optional domain latent for mid-layer injection (feature-gated)
///
/// # Returns
///
/// Mutable reference to logits buffer `ctx.logits`.
///
/// # When to Use
///
/// During **decode** (autoregressive generation) with SP-KV enabled.
/// Prefill should use standard `forward_base()` or PFlash.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[inline(always)]
pub fn forward_sp_kv<'a>(
    ctx: &'a mut crate::transformer::ForwardContext,
    weights: &crate::transformer::TransformerWeights,
    sp_cache: &mut crate::sp_kv::types::SpKvCache,
    predictors: &crate::sp_kv::types::SpKvPredictors,
    sp_ctx: &mut SpKvForwardContext,
    token: usize,
    pos: usize,
    config: &crate::types::Config,
    lora: Option<&crate::types::LoraAdapter>,
    gate_mode: crate::sp_kv::types::SpKvGateMode,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    use crate::types::{kv_dim, matmul, matmul_relu, rmsnorm};

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let n_kv = config.n_kv_head;
    // Copy scalar config fields to locals to avoid holding a reference across mutable borrows.
    let sp_predictor_hidden = sp_cache.config.predictor_hidden;
    let sp_window = sp_cache.config.window;
    let sp_threshold = sp_cache.config.threshold;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                + *weights.wpe.get_unchecked(pos_off_emb + i);
        }
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut sp_cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // ── SP-KV: Predict utility from hidden state (after RMSNorm, before QKV) ──
        // Zero-alloc: writes directly into pre-allocated head_utilities buffer.
        let predictor = &predictors.layers[layer_idx];
        crate::sp_kv::utility_predictor::predict_into(
            predictor,
            &ctx.x,
            n,
            sp_predictor_hidden,
            n_kv,
            &mut sp_ctx.predictor_buf,
            &mut sp_ctx.head_utilities,
        );

        // Aggregate per-head utilities to single scalar for cache write decision
        let pos_utility = crate::sp_kv::utility_predictor::aggregate_utilities(
            &sp_ctx.head_utilities,
            crate::sp_kv::utility_predictor::UtilityAggregation::Max,
        );

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            for i in 0..kvd {
                unsafe {
                    *ctx.k.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                    *ctx.v.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                }
            }
        }

        // ── SP-KV: Conditional KV write (skip positions with low utility) ──
        // Current position is always in window (distance 0 < window).
        layer_cache.write_gated(
            &ctx.k,
            &ctx.v,
            pos_utility,
            pos,
            true, // current pos always in window
            sp_threshold,
            kvd,
        );

        // ── SP-KV: Build gate biases for all past positions ──
        build_gate_biases(
            &mut sp_ctx.gate_bias,
            &layer_cache.utilities,
            &layer_cache.retained,
            pos,
            sp_window,
            sp_threshold,
            gate_mode,
        );

        // Multi-head attention with GQA + SP-KV gate bias
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);
        let t_n = pos + 1;

        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                // Direct monomorphized call — skips per-iteration Option dispatch
                attention_head_core(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                    GateBias::new(&sp_ctx.gate_bias.bias),
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // MLP w2: sparse when feature enabled and sparsity is high enough
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = crate::types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: standard matmul (clustered LM head not yet wired — TODO: expose as pub(crate))
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

// ---------------------------------------------------------------------------
// SP-KV + Quantized KV Fusion (Plan 070 Phase 3, Task T12)
// ---------------------------------------------------------------------------

/// SP-KV + Quantized KV fused forward: selective write + lossy quantize.
///
/// Two-stage KV compression pipeline, generic over any [`QuantizedKVCache`](crate::types::QuantizedKVCache) backend:
/// 1. **SP-KV selective write**: utility predictor decides which positions to retain
/// 2. **Quantized storage**: retained positions are compressed (f32 → 2-4 bits)
///
/// This is the maximal-compression decode path:
/// ```text
/// Prefill: PFlash (block-sparse token selection)
/// Decode:  SP-KV (selective write) → Quant cache (lossy quantize retained)
/// Result:  only useful KV pairs kept, those compressed to 2-4 bits/coord
/// ```
///
/// ## Architecture
///
/// Mirrors [`forward_sp_kv`] but replaces raw `SpKvCache` with [`SpKvQuantCache`]:
/// - `SpKvQuantCache.meta[]` tracks utilities + retained bitfield (which positions to keep)
/// - `SpKvQuantCache.quant` stores compressed KV for retained positions only
///
/// For retained positions: `quant.store_key/value()` quantizes in-place.
/// For pruned positions: no write — saving quantize compute and storage.
/// During attention: dequantize retained positions into flat buffer, then
/// `attention_head_core` with `GateBias` masking pruned positions to `-inf`.
///
/// ## Estimated Compression
///
/// At τ=0.5 (~30% density) + 3-bit quant: ~10.7 bits/position vs 32-bit baseline = **3× compression**.
/// At τ=0.7 (~11% density) + 3-bit quant: ~33 bits/position vs 32-bit baseline = **~29× compression**.
#[allow(clippy::too_many_arguments)]
pub fn forward_sp_kv_quant<'a, C: crate::types::QuantizedKVCache>(
    ctx: &'a mut crate::transformer::ForwardContext,
    weights: &crate::transformer::TransformerWeights,
    cache: &mut crate::sp_kv::types::SpKvQuantCache<C>,
    predictors: &crate::sp_kv::types::SpKvPredictors,
    sp_ctx: &mut SpKvForwardContext,
    token: usize,
    pos: usize,
    config: &crate::types::Config,
    lora: Option<&crate::types::LoraAdapter>,
    gate_mode: crate::sp_kv::types::SpKvGateMode,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    use crate::types::{kv_dim, matmul, matmul_relu, rmsnorm};

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let n_kv = config.n_kv_head;
    // Copy scalar config fields to avoid borrowing cache.config across mutable cache calls.
    let sp_predictor_hidden = cache.config.predictor_hidden;
    let sp_window = cache.config.window;
    let sp_threshold = cache.config.threshold;

    // Ensure flat dequant buffers are allocated
    sp_ctx.ensure_quant_bufs(config.block_size, kvd);

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                + *weights.wpe.get_unchecked(pos_off_emb + i);
        }
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // ── SP-KV: Predict utility from hidden state (after RMSNorm, before QKV) ──
        // Zero-alloc: writes directly into pre-allocated head_utilities buffer.
        let predictor = &predictors.layers[layer_idx];
        crate::sp_kv::utility_predictor::predict_into(
            predictor,
            &ctx.x,
            n,
            sp_predictor_hidden,
            n_kv,
            &mut sp_ctx.predictor_buf,
            &mut sp_ctx.head_utilities,
        );

        // Aggregate per-head utilities to single scalar for cache write decision
        let pos_utility = crate::sp_kv::utility_predictor::aggregate_utilities(
            &sp_ctx.head_utilities,
            crate::sp_kv::utility_predictor::UtilityAggregation::Max,
        );

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            for i in 0..kvd {
                unsafe {
                    *ctx.k.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                    *ctx.v.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                }
            }
        }

        // ── SP-KV + Quant: Conditional KV write (quantize only retained positions) ──
        // Current position is always in window (distance 0 < window).
        cache.write_gated(
            layer_idx,
            &ctx.k,
            &ctx.v,
            pos_utility,
            pos,
            true, // current pos always in window
            sp_threshold,
        );

        // ── SP-KV: Build gate biases for all past positions ──
        build_gate_biases(
            &mut sp_ctx.gate_bias,
            &cache.meta[layer_idx].utilities,
            &cache.meta[layer_idx].retained,
            pos,
            sp_window,
            sp_threshold,
            gate_mode,
        );

        // ── Quant: Dequantize retained positions into flat buffer ──
        // Non-retained positions are zeroed — masked by gate_bias = -inf during attention.
        cache.dequantize_retained_keys_into(layer_idx, pos, kvd, &mut sp_ctx.flat_keys);
        cache.dequantize_retained_values_into(layer_idx, pos, kvd, &mut sp_ctx.flat_values);

        // Multi-head attention with GQA + SP-KV gate bias on dequantized flat cache
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);
        let t_n = pos + 1;

        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                attention_head_core(
                    &ctx.q,
                    &sp_ctx.flat_keys,
                    &sp_ctx.flat_values,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                    GateBias::new(&sp_ctx.gate_bias.bias),
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // MLP w2: sparse when feature enabled and sparsity is high enough
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = crate::types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: standard matmul (clustered LM head not yet wired)
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline: attention_head_gated with gate_bias=None matches standard attention.
    /// Compares against a manual reference implementation.
    #[test]
    fn test_gated_none_matches_baseline() {
        let hd = 8;
        let kv_dim = 16; // 2 KV heads × 8 head_dim
        let t_n = 4;

        let q = vec![0.1; kv_dim];
        let key_cache = vec![0.2; t_n * kv_dim];
        let value_cache = vec![0.3; t_n * kv_dim];
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out_gated = vec![0.0; kv_dim];
        let mut scores_gated = vec![0.0; t_n];

        let mut attn_out_baseline = vec![0.0; kv_dim];
        let mut scores_baseline = vec![0.0; t_n];

        unsafe {
            // Gated with None
            attention_head_gated(
                &q,
                &key_cache,
                &value_cache,
                &mut attn_out_gated,
                &mut scores_gated,
                0,
                0,
                kv_dim,
                hd,
                t_n,
                scale,
                None,
            );

            // Manual baseline: same as gated but we compute manually
            for h_off in [0] {
                let mut max_s = f32::NEG_INFINITY;
                for t in 0..t_n {
                    let k_off = t * kv_dim + h_off;
                    let dot: f32 = q[h_off..h_off + hd]
                        .iter()
                        .zip(&key_cache[k_off..k_off + hd])
                        .map(|(&a, &b)| a * b)
                        .sum();
                    let score = dot * scale;
                    scores_baseline[t] = score;
                    if score > max_s {
                        max_s = score;
                    }
                }
                let sum: f32 = scores_baseline.iter().map(|s| (s - max_s).exp()).sum();
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let v: f32 = (0..t_n)
                        .map(|t| {
                            let exp = (scores_baseline[t] - max_s).exp();
                            exp * inv * value_cache[t * kv_dim + h_off + d]
                        })
                        .sum();
                    attn_out_baseline[h_off + d] = v;
                }
            }
        }

        for d in 0..hd {
            assert!(
                (attn_out_gated[d] - attn_out_baseline[d]).abs() < 1e-4,
                "Mismatch at d={d}: gated={gated}, baseline={baseline}",
                gated = attn_out_gated[d],
                baseline = attn_out_baseline[d],
            );
        }
    }

    /// Gate bias = -inf should zero out attention weight for that position.
    #[test]
    fn test_hard_gate_prunes_position() {
        let hd = 4;
        let kv_dim = 4;
        let t_n = 4;
        let scale = 1.0 / (hd as f32).sqrt();

        let q = vec![1.0; kv_dim];
        let key_cache = vec![1.0; t_n * kv_dim];
        let value_cache = vec![1.0; t_n * kv_dim];

        // Gate: keep positions 0,1,3; prune position 2
        let gate_bias = vec![0.0, 0.0, f32::NEG_INFINITY, 0.0];

        let mut attn_out = vec![0.0; kv_dim];
        let mut scores = vec![0.0; t_n];

        unsafe {
            attention_head_gated(
                &q,
                &key_cache,
                &value_cache,
                &mut attn_out,
                &mut scores,
                0,
                0,
                kv_dim,
                hd,
                t_n,
                scale,
                Some(&gate_bias),
            );
        }

        // All value_cache entries are 1.0, so attn_out[d] should be 1.0
        // (weighted average of 1.0s across non-pruned positions)
        for d in 0..hd {
            assert!(
                (attn_out[d] - 1.0).abs() < 1e-4,
                "Expected ~1.0 at d={d}, got {v}",
                v = attn_out[d],
            );
        }

        // Verify pruned position has zero attention weight
        // (positions 0,1,3 each get ~1/3 weight)
        let total_weight: f32 = scores[0] + scores[1] + scores[3];
        assert!(total_weight > 0.0, "Non-pruned weights should be positive");
        assert!(
            scores[2] < 1e-20,
            "Pruned position should have ~0 weight, got {w}",
            w = scores[2],
        );
    }

    /// Window positions should have zero bias regardless of utility.
    #[test]
    fn test_is_in_window() {
        assert!(is_in_window(10, 5, 8)); // 10-5=5 < 8
        assert!(is_in_window(10, 9, 8)); // 10-9=1 < 8
        assert!(!is_in_window(10, 2, 8)); // 10-2=8, NOT < 8
        assert!(is_in_window(5, 0, 128)); // 5-0=5 < 128
    }
}
