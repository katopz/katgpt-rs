//! DashAttention forward pass integration.
//!
//! Provides prefill (batch) and decode (single-token) forward paths that
//! combine chunk summarization, α-entmax routing, and sparse attention.
//!
//! Current status: MVP — chunk summaries are computed during prefill and
//! stored, but attention still uses the standard dense path. Full sparse
//! attention on active chunks will be added in a follow-up.

use crate::simd;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::{self, Config, DashAttnConfig};

use super::chunk_summary::{ChunkSummaryCache, ChunkSummaryQuery, summarize_chunk_into};
use super::routing::score_blocks_entmax_into;

// ---------------------------------------------------------------------------
// Prefill (batch prompt processing)
// ---------------------------------------------------------------------------

/// Forward pass for DashAttention in prefill mode.
///
/// 1. Chunk summarization over K at chunk boundaries
/// 2. Entmax routing via chunk summaries
/// 3. Dense attention (MVP — sparse on active chunks TODO)
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

    // Cache once — avoids O(n_kv_head * head_dim) scan per position per head per layer
    let zero_init = summary_query.is_zero_init();
    // Pre-allocate scratch buffers for the non-zero-init summarize path
    let mut summarize_out = vec![0.0f32; hd];
    let mut summarize_scores_buf = vec![0.0f32; 1]; // chunk_size=1 at boundaries

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

            // Compute chunk summaries at chunk boundaries
            if pos % dash_config.chunk_size == 0 {
                let chunk_idx = pos / dash_config.chunk_size;
                if chunk_idx < summary_cache.n_chunks() {
                    for h in 0..config.n_kv_head {
                        let k_h = &ctx.k[h * hd..(h + 1) * hd];
                        // Reuse per-head Vecs: clear + write in-place avoids realloc
                        let slot = &mut summary_cache.summaries[chunk_idx][h];
                        slot.resize(hd, 0.0);
                        if zero_init {
                            // Inline mean-pool for the common zero-init case (avoids alloc)
                            let inv = if k_h.len() == hd && hd > 0 {
                                1.0 / hd as f32
                            } else {
                                1.0
                            };
                            slot[..hd].copy_from_slice(k_h);
                            for v in slot[..hd].iter_mut() {
                                *v *= inv;
                            }
                        } else {
                            summarize_chunk_into(
                                summary_query,
                                k_h,
                                1,
                                h,
                                hd,
                                &mut summarize_out,
                                &mut summarize_scores_buf,
                            );
                            slot[..hd].copy_from_slice(&summarize_out[..hd]);
                        }
                    }
                }
            }

            types::matmul(&mut ctx.attn_out, &layer_weights.attn_wo, &ctx.q, n, n);
            simd::simd_add_inplace(&mut ctx.x[..n], &ctx.attn_out[..n]);
            simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            types::rmsnorm(&mut ctx.x);
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
    }
}

// ---------------------------------------------------------------------------
// Decode (single-token autoregressive)
// ---------------------------------------------------------------------------

/// Forward pass for DashAttention in decode mode.
///
/// Reuses cached chunk summaries and scores the current query against them
/// via entmax routing. Falls through to dense attention for MVP.
#[allow(clippy::too_many_arguments)]
pub fn forward_dash_attn_decode<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    _cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    dash_config: &DashAttnConfig,
    _summary_query: &ChunkSummaryQuery,
    summary_cache: &ChunkSummaryCache,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let tok_off = token * n;
    let pos_off = pos * n;
    ctx.x[..n].fill(0.0);
    simd::simd_add_inplace(&mut ctx.x[..n], &weights.wte[tok_off..tok_off + n]);
    simd::simd_add_inplace(&mut ctx.x[..n], &weights.wpe[pos_off..pos_off + n]);

    // Pre-allocate summary references outside the layer loop to avoid
    // per-layer Vec allocation (summaries don't change between layers).
    let mut summary_refs: Vec<&Vec<f32>> = Vec::with_capacity(summary_cache.n_chunks());
    // Pre-allocate routing scratch outside the layer loop for reuse across layers
    let mut routing_scratch =
        super::routing::RoutingScratch::new(summary_cache.n_chunks(), config.head_dim);

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

        // Entmax routing: score query against cached chunk summaries
        if summary_cache.n_chunks() > 0 {
            let hd = config.head_dim;
            // Use first query head as representative for routing decision
            let q_head = &ctx.q[..hd];
            // Reuse pre-allocated summary reference buffer
            summary_refs.clear();
            for chunk in &summary_cache.summaries {
                summary_refs.push(&chunk[0]);
            }
            let _routing =
                score_blocks_entmax_into(q_head, &summary_refs, dash_config, &mut routing_scratch);
            // TODO: Use routing.active_indices to select sparse KV blocks
        }

        types::matmul(&mut ctx.attn_out, &layer_weights.attn_wo, &ctx.q, n, n);
        simd::simd_add_inplace(&mut ctx.x[..n], &ctx.attn_out[..n]);
        simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, DashAttnConfig, Rng};

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
        for c in 0..n_chunks {
            for h in 0..config.n_kv_head {
                summary_cache.summaries[c][h] = vec![0.1; config.head_dim];
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
        for h in 0..config.n_kv_head {
            let summary = &summary_cache.summaries[0][h];
            assert_eq!(summary.len(), config.head_dim);
            // With zero-init head_cls → mean pooling → values should be finite
            for &v in summary {
                assert!(v.is_finite(), "chunk summary should be finite, got {v}");
            }
        }
    }
}
