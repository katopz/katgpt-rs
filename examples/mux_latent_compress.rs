//! MUX-Latent Context Compression example (Plan 238).
//!
//! Demonstrates compressing a 4k token sequence at different compression ratios,
//! printing compression ratio, KV savings, estimated TTFT reduction,
//! and adaptive LOD with mixed content.

#![cfg(feature = "mux_latent_context")]

use katgpt_rs::mux_latent::{
    CompressionRatio, CompressionSummary, LatentContextBuffer, LatentPrefillAdapter,
    MuxLatentConfig, MuxLatentEncoder, SpectralLOD, forward_prefill_with_compression,
};

fn main() {
    println!("=== MUX-Latent Context Compression Demo (Plan 238) ===\n");

    // Generate a 4k token sequence (simulating a long prompt)
    let tokens: Vec<u32> = (0..4096).map(|t| t % 32000).collect();
    println!("Input: {} tokens\n", tokens.len());

    // --- Fixed compression ratios ---
    for ratio in [
        CompressionRatio::X4,
        CompressionRatio::X8,
        CompressionRatio::X16,
    ] {
        let config = MuxLatentConfig {
            compression_ratio: ratio,
            preserve_instructions: true,
            ..Default::default()
        };

        let encoder = MuxLatentEncoder::new(config.clone());
        let ctx = encoder.encode(&tokens);

        let adapter = LatentPrefillAdapter::new(config.clone());
        let seq = adapter.to_prefill_sequence(&ctx);
        let summary = CompressionSummary::from_sequence(&seq);

        // Build the prefill plan
        let plan = forward_prefill_with_compression(&seq);

        println!("── {ratio:?} Compression ──");
        println!(
            "  Latent slots: {}  |  Raw tokens: {}",
            summary.latent_slots, summary.raw_tokens
        );
        println!(
            "  Effective prefill entries: {} / {} ({:.1}% reduction)",
            summary.effective_entries,
            summary.original_tokens,
            summary.kv_savings * 100.0
        );
        println!(
            "  KV savings: {:.1}%  |  TTFT reduction factor: {:.3}",
            summary.kv_savings * 100.0,
            summary.estimated_ttft_reduction
        );
        println!(
            "  Prefill plan token IDs: {} (vs {} original)\n",
            plan.token_ids.len(),
            tokens.len()
        );
    }

    // --- Adaptive LOD ---
    println!("── Adaptive LOD (SpectralLOD) ──");
    let slod = SpectralLOD::default();

    let diverse: Vec<u32> = vec![0, 500, 1000, 1500, 2000, 2500, 3000, 3500];
    let repetitive: Vec<u32> = vec![5, 5, 5, 5, 5, 5, 5, 5];
    let medium: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];

    for (label, window) in [
        ("diverse", &diverse),
        ("repetitive", &repetitive),
        ("medium", &medium),
    ] {
        let concentration = slod.energy_concentration(window);
        let ratio = slod.optimal_ratio(window);
        println!("  {label:12}: concentration={concentration:.3} -> {ratio:?}");
    }
    println!();

    // --- Buffer with budget enforcement ---
    println!("── LatentContextBuffer with Budget ──");
    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: true,
        max_latent_slots: 128,
        ..Default::default()
    };

    let buf = LatentContextBuffer::new(&tokens, config);
    let stats = buf.stats();
    println!("  Input tokens: {}", stats.total_input_tokens);
    println!(
        "  Latent slots used: {} / budget {}",
        stats.latent_slots_used, stats.latent_slot_budget
    );
    println!("  Raw segments: {}", stats.raw_segment_count);
    println!("  Memory savings: {:.1}%", stats.memory_savings * 100.0);

    // Verify expand roundtrip
    let expanded = buf.full_expand();
    let roundtrip_ok = expanded == tokens;
    println!(
        "  Expand roundtrip: {}",
        if roundtrip_ok { "✓ PASS" } else { "✗ FAIL" }
    );
    println!();

    println!("=== Demo Complete ===");
}
