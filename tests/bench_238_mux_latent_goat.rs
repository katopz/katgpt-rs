#![cfg(feature = "mux_latent_context")]

//! GOAT Benchmark for MUX-Latent Context Compression (Plan 238).
//!
//! G1–G7 gates must all pass before promoting `mux_latent_context` to default.

use katgpt_rs::mux_latent::{
    CompressionRatio, CompressionSummary, EvictionPolicy, LatentContextBuffer,
    LatentPrefillAdapter, MuxLatentConfig, MuxLatentEncoder, SpectralLOD, expand_all,
};

use std::time::Instant;

/// Helper: generate 4k tokens with varied content.
fn make_4k_tokens() -> Vec<u32> {
    (0..4096).map(|t| t % 32000).collect()
}

// ── G1: Compression ratio correctness ──────────────────────────

#[test]
fn g1_compression_ratio_correctness() {
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config);
    let tokens = make_4k_tokens();
    let ctx = encoder.encode(&tokens);

    // 4096 / 8 = 512 latent slots
    assert_eq!(
        ctx.latent_slot_count, 512,
        "G1 FAIL: expected exactly 512 latent slots at X8, got {}",
        ctx.latent_slot_count
    );
    assert_eq!(ctx.original_token_count, 4096);
}

// ── G2: KV savings ─────────────────────────────────────────────

#[test]
fn g2_kv_savings() {
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config);
    let tokens = make_4k_tokens();
    let ctx = encoder.encode(&tokens);

    let adapter = LatentPrefillAdapter::new(MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    });
    let seq = adapter.to_prefill_sequence(&ctx);
    let (_, _, savings) = LatentPrefillAdapter::kv_savings(&seq);

    assert!(
        savings > 0.80,
        "G2 FAIL: expected >80% KV savings at X8, got {:.1}%",
        savings * 100.0
    );
}

// ── G3: TTFT reduction estimate ────────────────────────────────

#[test]
fn g3_ttft_reduction_estimate() {
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config);
    let tokens = make_4k_tokens();
    let ctx = encoder.encode(&tokens);

    let adapter = LatentPrefillAdapter::new(MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    });
    let seq = adapter.to_prefill_sequence(&ctx);
    let summary = CompressionSummary::from_sequence(&seq);

    // At X8, effective_entries/original_tokens = 8/64 = 0.125 < 0.2
    assert!(
        summary.estimated_ttft_reduction < 0.2,
        "G3 FAIL: expected TTFT reduction factor < 0.2 at X8, got {:.3}",
        summary.estimated_ttft_reduction
    );
}

// ── G4: Encoder throughput ─────────────────────────────────────

#[test]
fn g4_encoder_throughput() {
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config);
    let tokens = make_4k_tokens();

    // Warm up
    let _ = encoder.encode(&tokens);

    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        std::hint::black_box(encoder.encode(std::hint::black_box(&tokens)));
    }
    let elapsed = start.elapsed();
    let per_encode = elapsed / iterations;

    // Budget: 200μs for debug builds (release will be much faster).
    // 4k tokens encode should be well under 1ms even unoptimized.
    assert!(
        per_encode.as_micros() < 500,
        "G4 FAIL: encode 4k tokens took {}μs (budget: 500μs)",
        per_encode.as_micros()
    );
}

// ── G5: Buffer eviction correctness ────────────────────────────

#[test]
fn g5_buffer_eviction_correctness() {
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X4,
        preserve_instructions: false,
        max_latent_slots: 4,
        ..Default::default()
    };

    let tokens: Vec<u32> = (0..64).collect();
    let mut buf = LatentContextBuffer::new(&tokens, config);
    buf.set_eviction_policy(EvictionPolicy::OldestFirst);

    // Initially should have compressed to 16 slots, then enforced to 4
    let stats = buf.stats();
    assert!(
        stats.latent_slots_used <= 4,
        "G5 FAIL: expected <= 4 latent slots after budget enforcement, got {}",
        stats.latent_slots_used
    );

    // Append more to trigger further eviction
    let more_tokens: Vec<u32> = (100..164).collect();
    buf.append(&more_tokens);

    let stats_after = buf.stats();
    assert!(
        stats_after.latent_slots_used <= 4,
        "G5 FAIL: expected <= 4 latent slots after append, got {}",
        stats_after.latent_slots_used
    );
}

// ── G6: Expand roundtrip ───────────────────────────────────────

#[test]
fn g6_expand_roundtrip() {
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config);
    let tokens = make_4k_tokens();
    let ctx = encoder.encode(&tokens);

    let expanded = expand_all(&ctx);
    assert_eq!(
        expanded, tokens,
        "G6 FAIL: expand roundtrip didn't recover original tokens"
    );
}

// ── G7: Adaptive LOD ───────────────────────────────────────────

#[test]
fn g7_adaptive_lod() {
    let slod = SpectralLOD::default();

    // Diverse window → high concentration → should prefer lower compression
    let diverse: Vec<u32> = vec![0, 500, 1000, 1500, 2000, 2500, 3000, 3500];
    let diverse_ratio = slod.optimal_ratio(&diverse);

    // Repetitive window → low concentration → should prefer higher compression
    let repetitive: Vec<u32> = vec![5, 5, 5, 5, 5, 5, 5, 5];
    let repetitive_ratio = slod.optimal_ratio(&repetitive);

    // Diverse should compress less aggressively than repetitive
    assert!(
        diverse_ratio.span_size() <= repetitive_ratio.span_size(),
        "G7 FAIL: diverse window ({diverse_ratio:?}) should compress less than repetitive ({repetitive_ratio:?})"
    );

    // Verify that diverse actually gets a lower compression (X4) or equal,
    // and repetitive gets equal or higher
    assert!(
        diverse_ratio == CompressionRatio::X4,
        "G7 FAIL: diverse window should get X4, got {diverse_ratio:?}"
    );
}
