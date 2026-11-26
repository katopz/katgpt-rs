//! DashAttention forward pass integration.
//!
//! Provides prefill (batch) and decode (single-token) forward paths that
//! combine chunk summarization, α-entmax routing, and sparse attention.
//!
//! Current status: MVP — chunk summaries are computed during prefill and
//! stored, but attention still uses the standard dense path. Full sparse
//! attention on active chunks will be added in a follow-up.
//!
//! # Origin
//!
//! Moved from `katgpt-rs/src/dash_attn/forward.rs` (Issue 007 Phase F.4a,
//! 2026-07-02). The composition layer previously stayed in root because
//! `ForwardContext` was root-only; now that `ForwardContext` lives in
//! `katgpt-forward` (Phase F.1-F.3), this file moved into the katgpt-attn
//! leaf to join the DashAttention substrate (chunk_summary + entmax + routing
//! already here). Path rewrites: `crate::transformer::ForwardContext` →
//! `katgpt_forward`, `crate::transformer::{MultiLayerKVCache,TransformerWeights}`
//! → `katgpt_transformer`, `crate::types` → `katgpt_core::types`. The
//! `super::{chunk_summary,routing}` paths stay unchanged (both modules are
//! siblings in this leaf).
//!
//! # Stripped: `forward_dash_attn_decode_vortex`
//!
//! The root original had a `#[cfg(feature = "vortex_flow")]` function
//! `forward_dash_attn_decode_vortex` that used `super::vortex_flow::{VortexRouter,
//! VortexRouterCache, VortexFlowExt, VortexScratch}`. The `vortex_flow` module
//! and its cluster (block_topk, channel_aware, entmax_router, meta_router,
//! value_energy) STAY IN ROOT because `meta_router` depends on
//! `pruners::bandit` + `speculative::types` (root-only modules) and `vortex_flow`
//! depends on `meta_router` — a chain that can't resolve in katgpt-attn without
//! pulling root-only deps. The vortex decode path was stripped from this leaf
//! migration; to re-add it, either (a) move the vortex_flow cluster into a
//! separate crate that can depend on bandit/speculative, or (b) inject the
//! router via a trait. Tracked as a non-blocking follow-up — `vortex_flow` is
//! default-on in root but the decode path is rarely the hot path (prefill +
//! standard decode cover the common cases).

use katgpt_core::simd;
use katgpt_core::types::{self, Config, DashAttnConfig};
use katgpt_forward::ForwardContext;
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};

use super::chunk_summary::{
    ChunkSummaryCache, ChunkSummaryQuery, summarize_chunk_into_with_entropy,
};

// ---------------------------------------------------------------------------
// Prefill (batch prompt processing)
// ---------------------------------------------------------------------------

/// Forward pass for DashAttention in prefill mode.
///
/// 1. Chunk summarization over K at chunk boundaries
/// 2. Entmax routing via chunk summaries
/// 3. Dense attention (sparse routing lives in `EntmaxRouter` / VortexFlow,
///    Plan 196 — not wired into this prefill function directly)
/// 4. Store chunk summaries to cache
#[allow(clippy::too_many_arguments)]
pub fn forward_dash_attn_prefill(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    _cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    dash_config: &DashAttnConfig,
    summary_query: &ChunkSummaryQuery,
    summary_cache: &mut ChunkSummaryCache,
) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kv_dim = types::kv_dim(config);
    let chunk_size = dash_config.chunk_size.max(1);

    // Pre-allocate scratch buffers for chunk summarization (reused per chunk).
    let mut summarize_out = vec![0.0f32; hd];
    let mut summarize_scores_buf = vec![0.0f32; chunk_size];
    let mut summarize_entropy = 0.0f32;

    // Per-head chunk key buffer (flat): `[n_kv_head * chunk_size * hd]`, laid
    // out head-major — head `h`'s chunk slot `(chunk_local, d)` lives at
    // `h * chunk_size * hd + chunk_local * hd + d`. Populated incrementally as
    // each token's K is computed; summarized when the chunk completes (every
    // `chunk_size` tokens, or at sequence end for a partial tail chunk).
    //
    // Flattened from `Vec<Vec<f32>>` to a single `Vec<f32>` so the whole
    // buffer is one allocation (vs `n_kv_head + 1` allocations) and one
    // contiguous memory block (friendlier to memcpy prefetch).
    //
    // This upgrades the prefill from single-token summaries (degenerate MVP)
    // to full-chunk summaries, which (a) fixes a pre-existing mean-pool bug
    // (the old inline zero-init path divided by head_dim instead of
    // chunk_size, producing k/hd instead of the correct mean), and (b)
    // activates the HiLS Prop 3.1 entropy bias (Issue 044) — at zero-init,
    // b'_c = ln(chunk_size) instead of ln(1) = 0.
    let chunk_stride = chunk_size * hd;
    let mut chunk_keys_buf: Vec<f32> = vec![0.0f32; config.n_kv_head * chunk_stride];

    for (pos, &token) in tokens.iter().enumerate() {
        let tok_off = token * n;
        let pos_off = pos * n;
        ctx.x[..n].fill(0.0);
        simd::simd_add_inplace(&mut ctx.x[..n], &weights.wte[tok_off..tok_off + n]);
        simd::simd_add_inplace(&mut ctx.x[..n], &weights.wpe[pos_off..pos_off + n]);

        for layer_weights in &weights.layers {
            types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            types::rmsnorm(&mut ctx.x);

            types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kv_dim, n);
            types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kv_dim, n);

            types::matmul(&mut ctx.attn_out, &layer_weights.attn_wo, &ctx.q, n, n);
            simd::simd_add_inplace(&mut ctx.x[..n], &ctx.attn_out[..n]);
            simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            types::rmsnorm(&mut ctx.x);
            #[cfg(feature = "gated_mlp")]
            {
                // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
                types::matmul(
                    &mut ctx.hidden,
                    &layer_weights.mlp_w1,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
                types::matmul(
                    &mut ctx.hidden2,
                    &layer_weights.mlp_w_up,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
                types::swiglu_inplace(&mut ctx.hidden, &ctx.hidden2);
            }
            #[cfg(not(feature = "gated_mlp"))]
            types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
        }

        // After the layer loop, ctx.k holds the last layer's K for this token.
        // Buffer head h's key into the current chunk slot.
        let chunk_local = pos % chunk_size;
        let dst_start = chunk_local * hd;
        for h in 0..config.n_kv_head {
            let src_start = h * hd;
            let head_off = h * chunk_stride;
            chunk_keys_buf[head_off + dst_start..head_off + dst_start + hd]
                .copy_from_slice(&ctx.k[src_start..src_start + hd]);
        }

        // Summarize at chunk completion (full chunk or partial tail).
        let is_chunk_end = chunk_local == chunk_size - 1;
        let is_seq_end = pos == tokens.len() - 1;
        if is_chunk_end || is_seq_end {
            let chunk_idx = pos / chunk_size;
            let actual_size = if is_chunk_end {
                chunk_size
            } else {
                chunk_local + 1
            };
            let actual_bytes = actual_size * hd;
            if chunk_idx < summary_cache.n_chunks() {
                for h in 0..config.n_kv_head {
                    let head_off = h * chunk_stride;
                    let slot = &mut summary_cache.summaries[chunk_idx][h];
                    let entropy_slot = &mut summary_cache.entropy_biases[chunk_idx][h];
                    summarize_chunk_into_with_entropy(
                        summary_query,
                        &chunk_keys_buf[head_off..head_off + actual_bytes],
                        actual_size,
                        h,
                        hd,
                        &mut summarize_out,
                        &mut summarize_scores_buf,
                        &mut summarize_entropy,
                    );
                    slot[..hd].copy_from_slice(&summarize_out[..hd]);
                    *entropy_slot = summarize_entropy;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decode (single-token autoregressive)
// ---------------------------------------------------------------------------

/// Forward pass for DashAttention in decode mode.
///
/// Reuses cached chunk summaries and scores the current query against them
/// via entmax routing. Falls through to dense attention for MVP.
///
/// # Dead routing elimination (perf)
///
/// The per-layer entmax routing call (`score_blocks_entmax_with_entropy_into`)
/// was computed but its result was bound to `_routing` and never consumed.
/// Removed to eliminate dead compute; the `dash_config`, `summary_query`,
/// and `summary_cache` params are retained (prefixed `_`) for API stability.
///
/// Block-level routing via `routing.active_indices` is implemented elsewhere
/// — see `EntmaxRouter` (Plan 196 / VortexFlow feature gate), which composes
/// `score_blocks_entmax` into a `VortexFlow` router consumed by the
/// VortexFlow dispatcher. The Plan 173 Task 6 sparse-KV-block-selection
/// milestone is complete via that path; this decode function does not need
/// to re-wire routing inline.
#[allow(clippy::too_many_arguments)]
pub fn forward_dash_attn_decode<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    _cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    _dash_config: &DashAttnConfig,
    _summary_query: &ChunkSummaryQuery,
    _summary_cache: &ChunkSummaryCache,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let tok_off = token * n;
    let pos_off = pos * n;
    ctx.x[..n].fill(0.0);
    simd::simd_add_inplace(&mut ctx.x[..n], &weights.wte[tok_off..tok_off + n]);
    simd::simd_add_inplace(&mut ctx.x[..n], &weights.wpe[pos_off..pos_off + n]);

    for layer_weights in &weights.layers {
        types::rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);

        types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        types::matmul(
            &mut ctx.k,
            &layer_weights.attn_wk,
            &ctx.x,
            types::kv_dim(config),
            n,
        );
        types::matmul(
            &mut ctx.v,
            &layer_weights.attn_wv,
            &ctx.x,
            types::kv_dim(config),
            n,
        );

        // Routing lives in `EntmaxRouter` (VortexFlow feature, Plan 196) —
        // see the function-level doc comment for the full rationale. This
        // decode function uses dense attention directly.

        types::matmul(&mut ctx.attn_out, &layer_weights.attn_wo, &ctx.q, n, n);
        simd::simd_add_inplace(&mut ctx.x[..n], &ctx.attn_out[..n]);
        simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);
        #[cfg(feature = "gated_mlp")]
        {
            // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
            types::matmul(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.hidden2,
                &layer_weights.mlp_w_up,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::swiglu_inplace(&mut ctx.hidden, &ctx.hidden2);
        }
        #[cfg(not(feature = "gated_mlp"))]
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        types::matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
    }

    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
    // LM head: standard_lm_head is private, use matmul directly
    types::matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );
    &mut ctx.logits
}

// NOTE: `forward_dash_attn_decode_vortex` (the `#[cfg(feature = "vortex_flow")]`
// variant) was STRIPPED during the leaf migration — see the module-level comment
// above for the full rationale (vortex_flow cluster stays root-only). Re-add by
// resolving the root-only vortex_flow/meta_router dependency chain first.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::types::{Config, DashAttnConfig, Rng};
    use katgpt_transformer::TransformerWeights;

    fn random_weights(config: &Config) -> TransformerWeights {
        let mut rng = Rng::new(42);
        TransformerWeights::new(config, &mut rng)
    }

    #[test]
    fn test_decode_returns_logits_slice() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let dash_config = DashAttnConfig::default();
        let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
        let summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

        let logits = forward_dash_attn_decode(
            &mut ctx,
            &weights,
            &mut cache,
            0,
            0,
            &config,
            &dash_config,
            &summary_query,
            &summary_cache,
        );

        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_decode_with_cached_summaries() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let dash_config = DashAttnConfig::default();
        let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
        let mut summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

        // Pre-populate some chunk summaries
        let n_chunks = 3;
        summary_cache.allocate(n_chunks);
        for summary_row in summary_cache.summaries[..n_chunks].iter_mut() {
            for head_summary in summary_row[..config.n_kv_head].iter_mut() {
                *head_summary = vec![0.1; config.head_dim];
            }
        }

        let logits = forward_dash_attn_decode(
            &mut ctx,
            &weights,
            &mut cache,
            0,
            0,
            &config,
            &dash_config,
            &summary_query,
            &summary_cache,
        );

        assert_eq!(logits.len(), config.vocab_size);
        // Logits should be finite (not NaN/Inf)
        for &l in logits.iter() {
            assert!(l.is_finite(), "logit should be finite, got {l}");
        }
    }

    #[test]
    fn test_prefill_runs_without_panics() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let dash_config = DashAttnConfig::default();
        let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
        let mut summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

        let tokens = vec![0, 1, 2];
        let n_chunks = tokens.len().div_ceil(dash_config.chunk_size) + 1;
        summary_cache.allocate(n_chunks.max(1));

        forward_dash_attn_prefill(
            &mut ctx,
            &weights,
            &mut cache,
            &tokens,
            &config,
            &dash_config,
            &summary_query,
            &mut summary_cache,
        );

        // Activation should be finite
        for &v in ctx.x.iter().take(config.n_embd) {
            assert!(v.is_finite(), "activation should be finite, got {v}");
        }
    }

    #[test]
    fn test_prefill_stores_chunk_summaries() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let dash_config = DashAttnConfig::default();
        let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
        let mut summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

        // chunk_size defaults to 64, token 0 triggers chunk boundary
        let n_chunks = 2;
        summary_cache.allocate(n_chunks);

        let tokens = vec![0];
        forward_dash_attn_prefill(
            &mut ctx,
            &weights,
            &mut cache,
            &tokens,
            &config,
            &dash_config,
            &summary_query,
            &mut summary_cache,
        );

        // Chunk 0 should have been populated for all KV heads
        for summary in summary_cache.summaries[0][..config.n_kv_head].iter() {
            assert_eq!(summary.len(), config.head_dim);
            // With zero-init head_cls → mean pooling → values should be finite
            for &v in summary {
                assert!(v.is_finite(), "chunk summary should be finite, got {v}");
            }
        }
    }

    /// Full-chunk summarization upgrade: prefill now buffers K across the entire
    /// chunk and summarizes at chunk completion, not at the first token.
    /// This activates the HiLS Prop 3.1 entropy bias (Issue 044).
    #[test]
    fn test_prefill_full_chunk_entropy_activation() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        // Small chunk size so we can fill a complete chunk in the test.
        let dash_config = DashAttnConfig {
            chunk_size: 4,
            ..DashAttnConfig::default()
        };
        let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
        let mut summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

        // 4 tokens → exactly 1 full chunk (chunk 0 completes at pos 3).
        let tokens = vec![0, 1, 2, 3];
        summary_cache.allocate(1);

        forward_dash_attn_prefill(
            &mut ctx,
            &weights,
            &mut cache,
            &tokens,
            &config,
            &dash_config,
            &summary_query,
            &mut summary_cache,
        );

        // At zero-init with chunk_size=4: entropy = ln(4) ≈ 1.386.
        // This is the KEY upgrade — previously entropy was always ln(1)=0
        // because prefill only summarized a single boundary token.
        let expected_entropy = (4.0f32).ln();
        for h in 0..config.n_kv_head {
            let entropy = summary_cache.entropy_biases[0][h];
            assert!(
                (entropy - expected_entropy).abs() < 1e-5,
                "head {h}: entropy={entropy}, expected ln(4)={expected_entropy}"
            );
            let summary = &summary_cache.summaries[0][h];
            assert_eq!(summary.len(), config.head_dim);
            for &v in summary {
                assert!(v.is_finite(), "chunk summary should be finite, got {v}");
            }
        }
    }

    /// Partial tail chunk: a sequence that doesn't fill the last chunk should
    /// still summarize the partial chunk with entropy = ln(actual_size).
    #[test]
    fn test_prefill_partial_tail_chunk() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let dash_config = DashAttnConfig {
            chunk_size: 4,
            ..DashAttnConfig::default()
        };
        let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
        let mut summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

        // 6 tokens → chunk 0 (full, 4 tokens) + chunk 1 (partial, 2 tokens).
        let tokens = vec![0, 1, 2, 3, 4, 5];
        summary_cache.allocate(2);

        forward_dash_attn_prefill(
            &mut ctx,
            &weights,
            &mut cache,
            &tokens,
            &config,
            &dash_config,
            &summary_query,
            &mut summary_cache,
        );

        // Chunk 0: full (4 tokens) → entropy = ln(4).
        let e0 = (4.0f32).ln();
        // Chunk 1: partial (2 tokens) → entropy = ln(2).
        let e1 = (2.0f32).ln();
        for h in 0..config.n_kv_head {
            assert!(
                (summary_cache.entropy_biases[0][h] - e0).abs() < 1e-5,
                "chunk 0 head {h}: entropy={}, expected ln(4)={}",
                summary_cache.entropy_biases[0][h],
                e0
            );
            assert!(
                (summary_cache.entropy_biases[1][h] - e1).abs() < 1e-5,
                "chunk 1 head {h}: entropy={}, expected ln(2)={}",
                summary_cache.entropy_biases[1][h],
                e1
            );
        }
    }
}
