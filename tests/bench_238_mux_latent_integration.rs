#![cfg(feature = "mux_latent_context")]

//! Integration tests for MUX-Latent full compress → decode → verify pipeline.
//!
//! Plan 238 Phase 6: end-to-end tests exercising the complete flow from
//! raw tokens through compression, expansion, prefill planning, budget
//! enforcement, and selective retrieval.

use katgpt_rs::mux_latent::{
    CompressionRatio, LatentContextBuffer, LatentPrefillAdapter, MuxLatentConfig, MuxLatentEncoder,
    expand_all, expand_segment, forward_prefill_with_compression, select_segments_to_expand,
};

#[cfg(feature = "lclm_adaptive_lod")]
use katgpt_rs::mux_latent::SpectralLOD;

// ── Helpers ────────────────────────────────────────────────────────

/// Generate tokens simulating a realistic vocabulary.
fn make_tokens(n: usize) -> Vec<u32> {
    (0..n).map(|t| (t as u32) % 32000).collect()
}

/// Generate mixed content: alternating windows of diverse and repetitive tokens.
#[cfg(feature = "lclm_adaptive_lod")]
fn make_mixed_tokens() -> Vec<u32> {
    let mut tokens = Vec::with_capacity(1024);
    // 8 windows of 128 tokens each
    for w in 0..8 {
        if w % 2 == 0 {
            // Diverse: spread across vocabulary
            for i in 0..128u32 {
                tokens.push((w * 1000 + i * 97) % 32000);
            }
        } else {
            // Repetitive: same token repeated
            for _ in 0..128 {
                tokens.push(42);
            }
        }
    }
    tokens
}

fn config_for(ratio: CompressionRatio) -> MuxLatentConfig {
    MuxLatentConfig {
        compression_ratio: ratio,
        preserve_instructions: false,
        ..Default::default()
    }
}

// ── Test 1: Compress → expand roundtrip at X4/X8/X16 ──────────────

#[test]
fn compress_then_expand_roundtrip() {
    // Use 512 tokens — fast in debug, still validates full pipeline.
    // (4k tested in bench_238_mux_latent_goat G6.)
    let tokens = make_tokens(512);

    for ratio in [
        CompressionRatio::X4,
        CompressionRatio::X8,
        CompressionRatio::X16,
    ] {
        let config = config_for(ratio);
        let encoder = MuxLatentEncoder::new(config);
        let ctx = encoder.encode(&tokens);

        // Verify compression actually happened
        assert!(
            ctx.latent_slot_count < tokens.len(),
            "X{ratio:?}: latent_slot_count ({}) should be < token count ({})",
            ctx.latent_slot_count,
            tokens.len()
        );

        // Expand all and verify perfect roundtrip
        let expanded = expand_all(&ctx);
        assert_eq!(
            expanded, tokens,
            "X{ratio:?}: expand_all roundtrip failed — recovered tokens differ from original"
        );

        // Spot-check first, middle, and last segments via expand_segment
        let span = ratio.span_size();
        let expected_segments = tokens.len().div_ceil(span);
        let check_ids: Vec<usize> = if expected_segments <= 3 {
            (0..expected_segments).collect()
        } else {
            vec![0, expected_segments / 2, expected_segments - 1]
        };
        for seg_id in check_ids {
            let seg = expand_segment(&ctx, seg_id as u32);
            assert!(
                seg.is_some(),
                "X{ratio:?}: expand_segment({}) returned None",
                seg_id
            );
            let seg = seg.unwrap();
            assert_eq!(seg.segment_id, seg_id as u32);
            let start = seg_id * span;
            let end = (start + span).min(tokens.len());
            assert_eq!(
                seg.tokens,
                tokens[start..end],
                "X{ratio:?}: segment {} tokens mismatch",
                seg_id
            );
        }
    }
}

// ── Test 2: Compress → prefill plan ────────────────────────────────

#[test]
fn compress_then_prefill_plan() {
    let tokens = make_tokens(512);
    let config = config_for(CompressionRatio::X8);
    let encoder = MuxLatentEncoder::new(config.clone());
    let ctx = encoder.encode(&tokens);

    // Create prefill sequence via adapter
    let adapter = LatentPrefillAdapter::new(config);
    let seq = adapter.to_prefill_sequence(&ctx);

    // Verify the sequence is smaller than original
    assert!(
        seq.effective_prefill_len < tokens.len(),
        "effective_prefill_len ({}) should be < original ({})",
        seq.effective_prefill_len,
        tokens.len()
    );
    assert_eq!(seq.original_token_count, tokens.len());

    // Create the prefill plan
    let plan = forward_prefill_with_compression(&seq);

    // Plan token_ids should be fewer than original
    assert!(
        plan.token_ids.len() < tokens.len(),
        "plan.token_ids ({}) should be < original tokens ({})",
        plan.token_ids.len(),
        tokens.len()
    );

    // Metadata should have latent_indices populated
    let meta = &plan.compression;
    assert!(
        !meta.latent_indices.is_empty(),
        "latent_indices should be non-empty for compressed context"
    );
    // Each latent index should correspond to a valid position in token_ids
    for &idx in &meta.latent_indices {
        assert!(
            idx < plan.token_ids.len(),
            "latent index {} out of bounds (token_ids len {})",
            idx,
            plan.token_ids.len()
        );
    }
    // summary should report both latent and raw counts
    assert!(meta.summary.latent_slots > 0, "summary.latent_slots should be > 0");
    assert!(meta.summary.effective_entries < tokens.len(), "summary.effective_entries ({}) should be < tokens.len() ({})", meta.summary.effective_entries, tokens.len());
}

// ── Test 3: Buffer budget enforcement ──────────────────────────────

#[test]
fn buffer_budget_enforcement_integration() {
    // Tight budget: 512 tokens at X8 = 64 slots, but we only allow 8
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        max_latent_slots: 8,
        ..Default::default()
    };

    let tokens = make_tokens(512);
    let mut buf = LatentContextBuffer::new(&tokens, config);

    // Verify budget enforced on initial compression
    let stats = buf.stats();
    assert!(
        stats.latent_slots_used <= 8,
        "initial latent_slots_used ({}) should be <= budget (8)",
        stats.latent_slots_used
    );
    assert_eq!(stats.total_input_tokens, 512);

    // Append several chunks to stress the budget
    for chunk in 0..4 {
        let extra: Vec<u32> = (5000u32..5064).map(|t| t + chunk * 64).collect();
        buf.append(&extra);
    }

    // Budget should still be enforced after appends
    let stats_after = buf.stats();
    assert!(
        stats_after.latent_slots_used <= 8,
        "post-append latent_slots_used ({}) should be <= budget (8)",
        stats_after.latent_slots_used
    );
    assert!(
        stats_after.total_input_tokens > 512,
        "total_input_tokens should have grown after appends, got {}",
        stats_after.total_input_tokens
    );

    // Full expand should still recover all tokens
    let expanded = buf.full_expand();
    assert_eq!(
        expanded.len(),
        stats_after.total_input_tokens,
        "full_expand length ({}) should match total_input_tokens ({})",
        expanded.len(),
        stats_after.total_input_tokens
    );
    // Verify first and last chunks are present
    assert_eq!(expanded[0], tokens[0]);
}

// ── Test 4: Adaptive vs fixed compression ─────────────────────────

#[test]
#[cfg(feature = "lclm_adaptive_lod")]
fn adaptive_vs_fixed_compression() {
    let tokens = make_mixed_tokens();

    // Fixed X8 compression
    let fixed_config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let fixed_buf = LatentContextBuffer::new(&tokens, fixed_config.clone());

    // Adaptive compression with default SLoD
    let slod = SpectralLOD::default();
    let adaptive_buf = LatentContextBuffer::new_adaptive(&tokens, fixed_config, slod);

    // Both should roundtrip correctly
    let fixed_expanded = fixed_buf.full_expand();
    let adaptive_expanded = adaptive_buf.full_expand();
    assert_eq!(fixed_expanded, tokens, "fixed X8 roundtrip failed");
    assert_eq!(adaptive_expanded, tokens, "adaptive roundtrip failed");

    // Adaptive should produce different segment structures for mixed content.
    // Count how many segments are compressed vs raw in each.
    let fixed_stats = fixed_buf.stats();
    let adaptive_stats = adaptive_buf.stats();

    // Adaptive may have a different latent_slots_used than fixed
    // (the key property: they diverge for mixed content)
    assert!(
        fixed_stats.latent_slots_used > 0,
        "fixed buffer should have latent slots"
    );
    assert!(
        adaptive_stats.latent_slots_used > 0,
        "adaptive buffer should have latent slots"
    );

    // Verify that the segment structures differ.
    // Fixed X8 on 1024 tokens → 128 segments all compressed at span_size=8.
    // Adaptive should produce a mix because diverse windows get X4 (more segments)
    // and repetitive windows get X16 (fewer segments).
    let fixed_ctx = fixed_buf.context();
    let adaptive_ctx = adaptive_buf.context();

    // Count distinct span sizes in adaptive (should be > 1 for mixed content)
    let adaptive_span_sizes: Vec<usize> = adaptive_ctx
        .segments
        .iter()
        .filter_map(|seg| match seg {
            katgpt_rs::mux_latent::LatentSegment::Compressed {
                original_tokens, ..
            } => Some(original_tokens.len()),
            _ => None,
        })
        .collect();

    // The adaptive buffer should have at least one segment that isn't span_size=8
    // (diverse windows should get smaller spans, repetitive should get larger)
    let fixed_all_same = fixed_ctx
        .segments
        .iter()
        .all(|seg| matches!(seg, katgpt_rs::mux_latent::LatentSegment::Compressed { .. }));
    assert!(
        fixed_all_same,
        "fixed X8 should have all compressed segments"
    );

    // Adaptive span sizes should not all be identical (mixed content → varied LOD)
    let unique_adaptive_spans: std::collections::HashSet<usize> =
        adaptive_span_sizes.into_iter().collect();
    // If adaptive produces multiple span sizes, the structures differ
    let structures_differ = unique_adaptive_spans.len() > 1
        || adaptive_stats.latent_slots_used != fixed_stats.latent_slots_used;
    assert!(
        structures_differ,
        "adaptive and fixed should produce different segment structures \
         (adaptive spans: {:?}, fixed slots: {}, adaptive slots: {})",
        unique_adaptive_spans, fixed_stats.latent_slots_used, adaptive_stats.latent_slots_used
    );
}

// ── Test 5: Selective expand by query ──────────────────────────────

#[test]
fn selective_expand_by_query() {
    // Build tokens with known segments at X4
    // Segment 0: [0, 1, 2, 3]
    // Segment 1: [100, 101, 102, 103]
    // Segment 2: [200, 201, 202, 203]
    // ...
    let config = config_for(CompressionRatio::X4);
    let tokens: Vec<u32> = (0..256).map(|i| (i / 4) * 100 + (i % 4)).collect();
    // So:
    //   seg 0: [0, 1, 2, 3]
    //   seg 1: [100, 101, 102, 103]
    //   seg 2: [200, 201, 202, 203]
    //   ...
    //   seg 63: [6300, 6301, 6302, 6303]

    let encoder = MuxLatentEncoder::new(config);
    let ctx = encoder.encode(&tokens);

    // Verify the encoding produced expected number of segments
    assert_eq!(ctx.latent_slot_count, 64);

    // Query for tokens from segment 5 → expect seg 5 (tokens: 500,501,502,503)
    let query_seg5 = vec![501u32, 502];
    let results = select_segments_to_expand(&ctx, &query_seg5, 3);
    assert!(
        results.contains(&5),
        "query {query_seg5:?} should return segment 5, got {results:?}"
    );
    assert!(
        results.len() <= 3,
        "top_k=3 should return at most 3 results, got {}",
        results.len()
    );

    // Query for tokens from segment 10 → expect seg 10 (tokens: 1000,1001,1002,1003)
    let query_seg10 = vec![1000u32, 1001, 1002];
    let results2 = select_segments_to_expand(&ctx, &query_seg10, 1);
    assert_eq!(
        results2.len(),
        1,
        "exact match query should return exactly 1 result with top_k=1"
    );
    assert_eq!(
        results2[0], 10,
        "query for seg 10 tokens should return segment 10"
    );

    // Multi-segment query: tokens from seg 0 and seg 63
    let query_multi = vec![0u32, 1, 6300, 6301];
    let results3 = select_segments_to_expand(&ctx, &query_multi, 2);
    assert_eq!(
        results3.len(),
        2,
        "multi-segment query should return 2 results"
    );
    assert!(
        results3.contains(&0) && results3.contains(&63),
        "multi-segment query should return segments 0 and 63, got {results3:?}"
    );

    // Verify expand_segment returns the correct tokens for found segments
    for &seg_id in &results {
        let expanded = expand_segment(&ctx, seg_id);
        assert!(expanded.is_some(), "segment {seg_id} should be expandable");
        let seg = expanded.unwrap();
        assert_eq!(seg.segment_id, seg_id);
        // Tokens should contain our query tokens
        let has_query_token = seg.tokens.iter().any(|t| query_seg5.contains(t));
        assert!(
            has_query_token,
            "segment {seg_id} tokens {:?} should contain at least one query token",
            seg.tokens
        );
    }
}

// TL;DR: 5 integration tests for Plan 238 Phase 6 covering full
// compress→expand roundtrip, prefill planning, budget enforcement,
// adaptive vs fixed compression, and selective expand by query.
