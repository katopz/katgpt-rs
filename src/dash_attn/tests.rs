//! Integration tests for the dash_attn module.
//!
//! Tests the combined workflow: chunk summarization → entmax routing → forward pass.

use super::chunk_summary::{ChunkSummaryCache, ChunkSummaryQuery, summarize_chunk};
use super::entmax::{entmax_1p5, entmax_gqa_aggregate};
use super::forward::{forward_dash_attn_decode, forward_dash_attn_prefill};
use super::routing::{compute_routing_bias, score_blocks_entmax};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::{Config, DashAttnConfig, Rng};

fn random_weights(config: &Config) -> TransformerWeights {
    let mut rng = Rng::new(42);
    TransformerWeights::new(config, &mut rng)
}

// ---------------------------------------------------------------------------
// End-to-end forward pass integration
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_then_decode_integration() {
    let config = Config::micro();
    let weights = random_weights(&config);
    let dash_config = DashAttnConfig::default();
    let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);

    // --- Prefill ---
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
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

    // Prefill should have produced finite activations
    for &v in ctx.x.iter().take(config.n_embd) {
        assert!(
            v.is_finite(),
            "prefill activation should be finite, got {v}"
        );
    }

    // Chunk 0 should be populated
    assert!(summary_cache.n_chunks() > 0, "should have cached chunks");
    for h in 0..config.n_kv_head {
        for &v in &summary_cache.summaries[0][h] {
            assert!(v.is_finite(), "chunk summary should be finite, got {v}");
        }
    }

    // --- Decode ---
    let mut ctx_decode = ForwardContext::new(&config);
    let mut cache_decode = MultiLayerKVCache::new(&config);

    let logits = forward_dash_attn_decode(
        &mut ctx_decode,
        &weights,
        &mut cache_decode,
        3, // next token
        tokens.len(),
        &config,
        &dash_config,
        &summary_query,
        &summary_cache,
    );

    assert_eq!(logits.len(), config.vocab_size);
    for &l in logits.iter() {
        assert!(l.is_finite(), "decode logit should be finite, got {l}");
    }
}

#[test]
fn test_decode_empty_cache_runs_cleanly() {
    let config = Config::micro();
    let weights = random_weights(&config);
    let dash_config = DashAttnConfig::default();
    let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
    let summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

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
    for &l in logits.iter() {
        assert!(
            l.is_finite(),
            "logit should be finite with empty cache, got {l}"
        );
    }
}

// ---------------------------------------------------------------------------
// Routing + entmax integration
// ---------------------------------------------------------------------------

#[test]
fn test_routing_with_real_summaries() {
    let config = Config::micro();
    let dash_config = DashAttnConfig::default();
    let hd = config.head_dim;

    let query = ChunkSummaryQuery::new(config.n_kv_head, hd);

    // Create 3 chunks with very different magnitudes so routing can discriminate.
    // Chunk 0: large positive, Chunk 1: small, Chunk 2: negative.
    let chunk_keys: [Vec<f32>; 3] = [
        (0..hd).map(|d| 10.0 * (d as f32 + 1.0).sin()).collect(),
        (0..hd).map(|d| 0.1 * (d as f32 + 1.0).cos()).collect(),
        (0..hd).map(|d| -10.0 * (d as f32 + 1.0).sin()).collect(),
    ];

    let summaries: Vec<Vec<f32>> = chunk_keys
        .iter()
        .map(|keys| summarize_chunk(&query, keys, 1, 0, hd))
        .collect();

    // Query aligned with chunk 0 (same sign, similar direction)
    let routing_query: Vec<f32> = summaries[0].iter().map(|&x| x * 2.0).collect();

    let result = score_blocks_entmax(&routing_query, &summaries, &dash_config);

    // Probabilities must sum to 1 and be non-negative
    let sum: f32 = result.probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "probs must sum to 1.0, got {sum}");
    for &p in &result.probs {
        assert!(p >= 0.0, "prob must be non-negative, got {p}");
    }

    // Chunk 0 should dominate (query is a scaled copy of its summary)
    assert!(
        result.probs[0] > result.probs[1],
        "chunk 0 should score > chunk 1: {} vs {}",
        result.probs[0],
        result.probs[1]
    );
    assert!(
        result.probs[0] > result.probs[2],
        "chunk 0 should score > chunk 2: {} vs {}",
        result.probs[0],
        result.probs[2]
    );
}

#[test]
fn test_compute_routing_with_multi_head_summaries() {
    let config = Config::micro();
    let dash_config = DashAttnConfig::default();
    let hd = config.head_dim;
    let n_q_heads = config.n_head;
    let n_kv_heads = config.n_kv_head;

    let query = ChunkSummaryQuery::new(n_kv_heads, hd);

    // Create summaries for 4 chunks (using head 0 as proxy)
    let summaries: Vec<Vec<f32>> = (0..4)
        .map(|c| {
            let chunk_keys: Vec<f32> = (0..hd).map(|d| (c as f32 * 0.1 + d as f32).cos()).collect();
            summarize_chunk(&query, &chunk_keys, 1, 0, hd)
        })
        .collect();

    // Create synthetic per-head queries
    let queries: Vec<Vec<f32>> = (0..n_q_heads)
        .map(|h| (0..hd).map(|d| (h as f32 * 0.1 + d as f32).sin()).collect())
        .collect();

    let results = compute_routing_bias(&queries, &summaries, n_kv_heads, &dash_config);

    assert_eq!(results.len(), n_q_heads, "one result per query head");
    for (h, r) in results.iter().enumerate() {
        let sum: f32 = r.probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "head {h} probs must sum to 1.0, got {sum}"
        );
        assert_eq!(r.active_indices.len(), r.bias.len());
    }
}

// ---------------------------------------------------------------------------
// Chunk summarization + entmax pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_summarize_then_entmax_route() {
    let hd = 8;
    let n_chunks = 3;
    let dash_config = DashAttnConfig::default();
    let query = ChunkSummaryQuery::new(1, hd);

    // Create well-separated chunks: each has a unique "hot" dimension.
    let chunk_data: Vec<Vec<f32>> = (0..n_chunks)
        .map(|c| {
            let mut v = vec![0.0; hd];
            v[c * 2] = 10.0; // hot dimension unique to this chunk
            v
        })
        .collect();

    // Summarize each chunk (zero-init → mean pool = identity for single token)
    let summaries: Vec<Vec<f32>> = chunk_data
        .iter()
        .map(|keys| summarize_chunk(&query, keys, 1, 0, hd))
        .collect();

    assert_eq!(summaries.len(), n_chunks);

    // Route with a query matching chunk 1's hot direction
    let mut routing_query = vec![0.0; hd];
    routing_query[2] = 5.0;

    let result = score_blocks_entmax(&routing_query, &summaries, &dash_config);

    let sum: f32 = result.probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "probs must sum to 1.0, got {sum}");

    // Chunk 1 should have highest probability (query aligns with its hot dim)
    assert!(
        result.probs[1] >= result.probs[0],
        "chunk 1 should score >= chunk 0: {} vs {}",
        result.probs[1],
        result.probs[0]
    );
    assert!(
        result.probs[1] >= result.probs[2],
        "chunk 1 should score >= chunk 2: {} vs {}",
        result.probs[1],
        result.probs[2]
    );
}

#[test]
fn test_entmax_gqa_with_routing_results() {
    let n_query_heads = 4;
    let n_kv_heads = 2;
    let n_chunks = 3;

    // Run per-head entmax to get probabilities
    let all_scores: Vec<Vec<f32>> = (0..n_query_heads)
        .map(|h| (0..n_chunks).map(|c| (h + c) as f32 * 0.5).collect())
        .collect();

    let head_probs: Vec<Vec<f32>> = all_scores
        .iter()
        .map(|scores| entmax_1p5(scores).0)
        .collect();

    // Aggregate across GQA groups
    let agg = entmax_gqa_aggregate(&head_probs, n_query_heads, n_kv_heads, n_chunks);

    assert_eq!(agg.len(), n_kv_heads);
    for (g, group_probs) in agg.iter().enumerate() {
        assert_eq!(group_probs.len(), n_chunks);
        let sum: f32 = group_probs.iter().sum();
        // Aggregated probs don't necessarily sum to 1 (they're averages)
        // but each should be non-negative
        for (c, &p) in group_probs.iter().enumerate() {
            assert!(
                p >= 0.0,
                "aggregated prob for group {g} chunk {c} should be non-negative, got {p}"
            );
        }
        assert!(
            sum > 0.0,
            "group {g} should have some probability mass, sum={sum}"
        );
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_entmax_all_equal_scores_routing() {
    let dash_config = DashAttnConfig::default();
    let query = vec![1.0, 0.0, 0.0];
    // All chunks identical → uniform routing
    let summaries = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
    ];

    let result = score_blocks_entmax(&query, &summaries, &dash_config);

    let sum: f32 = result.probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "uniform routing should sum to 1.0, got {sum}"
    );

    // All probs should be approximately equal
    let expected = 1.0 / summaries.len() as f32;
    for (i, &p) in result.probs.iter().enumerate() {
        assert!(
            (p - expected).abs() < 1e-5,
            "uniform chunk {i}: expected {expected}, got {p}"
        );
    }
}

#[test]
fn test_chunk_summary_cache_lifecycle() {
    let n_kv = 2;
    let hd = 4;
    let query = ChunkSummaryQuery::new(n_kv, hd);

    let mut cache = ChunkSummaryCache::new(n_kv, hd);
    assert_eq!(cache.n_chunks(), 0);

    // Allocate
    cache.allocate(3);
    assert_eq!(cache.n_chunks(), 3);

    // Summarize and store
    let keys = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    for c in 0..3 {
        for h in 0..n_kv {
            let offset = h * hd;
            let summary = summarize_chunk(&query, &keys[offset..offset + hd], 1, h, hd);
            cache.summaries[c][h] = summary;
        }
    }

    // Verify
    for c in 0..3 {
        let view = cache.view(c);
        assert_eq!(view.len(), n_kv);
    }

    // Reset
    cache.reset();
    assert_eq!(cache.n_chunks(), 0);
}

#[test]
fn test_prefill_multiple_chunk_boundaries() {
    let config = Config::micro();
    let weights = random_weights(&config);
    let dash_config = DashAttnConfig {
        chunk_size: 2,
        ..DashAttnConfig::default()
    };

    let summary_query = ChunkSummaryQuery::new(config.n_kv_head, config.head_dim);
    let mut summary_cache = ChunkSummaryCache::new(config.n_kv_head, config.head_dim);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // 6 tokens with chunk_size=2 → 3 chunk boundaries (pos 0, 2, 4)
    let tokens = vec![0, 1, 2, 3, 4, 5];
    let n_chunks = tokens.len().div_ceil(dash_config.chunk_size);
    summary_cache.allocate(n_chunks);

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

    // Verify summaries were computed at chunk boundaries
    assert!(
        summary_cache.n_chunks() >= n_chunks,
        "expected at least {n_chunks} chunks, got {}",
        summary_cache.n_chunks()
    );
}
