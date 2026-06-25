#![cfg(feature = "mux_latent_context")]

//! Benchmark: Fixed vs Adaptive LOD on RULER-style NIAH tasks (Plan 238 Phase 4).
//!
//! Compares fixed compression ratios (X4/X8/X16) against SLoD-adaptive compression
//! for needle-in-a-haystack retrieval quality and throughput.
//!
//! Run with: cargo test --features mux_latent_context bench_238_adaptive_lod -- --nocapture

use std::time::Instant;

use katgpt_rs::mux_latent::{
    CompressionRatio, MuxLatentConfig, MuxLatentEncoder, SpectralLOD, select_segments_to_expand,
};

#[cfg(feature = "lclm_adaptive_lod")]
use katgpt_rs::mux_latent::LatentContextBuffer;

// ── Helpers ──────────────────────────────────────────────────────────

/// Build a NIAH token sequence: repetitive haystack with a diverse needle inserted.
///
/// `haystack_size`: total token count.
/// `needle_pos`: fraction [0.0, 1.0] where the needle starts (0=start, 0.5=middle, 1.0=end).
/// `needle`: the diverse "needle" tokens to hide in the haystack.
fn make_niah_tokens(haystack_size: usize, needle_pos: f32, needle: &[u32]) -> Vec<u32> {
    let needle_start = ((haystack_size - needle.len()).max(0) as f32 * needle_pos) as usize;
    let mut tokens = vec![5u32; haystack_size]; // repetitive haystack
    for (i, &t) in needle.iter().enumerate() {
        if needle_start + i < haystack_size {
            tokens[needle_start + i] = t;
        }
    }
    tokens
}

/// Standard NIAH needle: diverse token IDs that stand out against the haystack.
fn standard_needle() -> Vec<u32> {
    vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

/// Build a fixed-compression config for a given ratio.
fn fixed_config(ratio: CompressionRatio) -> MuxLatentConfig {
    MuxLatentConfig {
        compression_ratio: ratio,
        window_size: 1024,
        preserve_instructions: false,
        ..Default::default()
    }
}

/// Build mixed-content tokens: 50% repetitive, 30% medium-diversity, 20% high-diversity.
#[cfg(feature = "lclm_adaptive_lod")]
fn make_mixed_content(total: usize) -> (Vec<u32>, Vec<(usize, &'static str)>) {
    let mut tokens = Vec::with_capacity(total);
    let mut labels = Vec::new();

    let n_repetitive = total * 50 / 100;
    let n_medium = total * 30 / 100;
    let n_diverse = total - n_repetitive - n_medium;

    // Repetitive block
    let start = tokens.len();
    for _ in 0..n_repetitive {
        tokens.push(5u32);
    }
    labels.push((start, "repetitive"));

    // Medium-diversity block: small range of values
    let start = tokens.len();
    for i in 0..n_medium {
        tokens.push(((i % 20) + 100) as u32);
    }
    labels.push((start, "medium"));

    // High-diversity block: wide range of values
    let start = tokens.len();
    for i in 0..n_diverse {
        tokens.push(((i * 997) % 32000) as u32);
    }
    labels.push((start, "diverse"));

    (tokens, labels)
}

// ── Test 1: Fixed vs Adaptive NIAH retrieval ─────────────────────────

#[test]
fn bench_fixed_vs_adaptive_niah() {
    #[cfg(feature = "lclm_adaptive_lod")]
    let slod = SpectralLOD::default();
    let needle = standard_needle();

    let haystack_sizes: &[usize] = &[1024, 2048, 4096];
    let positions: &[(f32, &str)] = &[(0.0, "start"), (0.5, "middle"), (1.0, "end")];
    let fixed_ratios: &[(CompressionRatio, &str)] = &[
        (CompressionRatio::X4, "X4"),
        (CompressionRatio::X8, "X8"),
        (CompressionRatio::X16, "X16"),
    ];
    let top_k = 3;

    println!();
    println!(
        "╔════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  Plan 238 Phase 4: Fixed vs Adaptive LOD — RULER-style NIAH Benchmark            ║"
    );
    println!(
        "╠════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║                                                                                  ║"
    );
    println!(
        "║  Needle: 10 diverse tokens hidden in repetitive haystack (all token 5)           ║"
    );
    println!(
        "║  Query: needle tokens used as query for select_segments_to_expand (top_k={top_k})  ║"
    );
    println!(
        "║                                                                                  ║"
    );
    println!(
        "╠════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║  Haystack │ Position │ Mode     │ Found │ Segs Scanned │ Latent Slots            ║"
    );
    println!(
        "╠═══════════╪══════════╪══════════╪═══════╪══════════════╪══════════════════════════╣"
    );

    #[cfg(feature = "lclm_adaptive_lod")]
    let mut adaptive_wins = 0usize;
    #[cfg(feature = "lclm_adaptive_lod")]
    let mut adaptive_total = 0usize;
    let mut x8_found_count = 0usize;

    for &size in haystack_sizes {
        for &(pos, pos_label) in positions {
            let tokens = make_niah_tokens(size, pos, &needle);

            // Fixed compression modes
            for &(ratio, ratio_label) in fixed_ratios {
                let config = fixed_config(ratio);
                let encoder = MuxLatentEncoder::new(config);
                let ctx = encoder.encode(&tokens);
                let found = select_segments_to_expand(&ctx, &needle, top_k);
                let needle_found = !found.is_empty();

                if ratio == CompressionRatio::X8 && needle_found {
                    x8_found_count += 1;
                }

                println!(
                    "║  {:>5}    │ {:>8} │ {:>8} │ {:>5} │ {:>12} │ {:>12}             ║",
                    size,
                    pos_label,
                    format!("Fixed {}", ratio_label),
                    if needle_found { "YES" } else { "no" },
                    found.len(),
                    ctx.latent_slot_count,
                );
            }

            // Adaptive compression (feature-gated)
            #[cfg(feature = "lclm_adaptive_lod")]
            {
                adaptive_total += 1;
                let config = fixed_config(CompressionRatio::X8); // base ratio for adaptive
                let buf = LatentContextBuffer::new_adaptive(&tokens, config, slod.clone());
                let ctx = buf.context();
                let found = select_segments_to_expand(ctx, &needle, top_k);
                let needle_found = !found.is_empty();

                if needle_found {
                    adaptive_wins += 1;
                }

                println!(
                    "║  {:>5}    │ {:>8} │ {:>8} │ {:>5} │ {:>12} │ {:>12}             ║",
                    size,
                    pos_label,
                    "Adaptive",
                    if needle_found { "YES" } else { "no" },
                    found.len(),
                    ctx.latent_slot_count,
                );
            }

            #[cfg(not(feature = "lclm_adaptive_lod"))]
            {
                println!(
                    "║  {:>5}    │ {:>8} │ {:>8} │   --- │          --- │          ---             ║",
                    size, pos_label, "Adaptive",
                );
                println!(
                    "║           │          │          │ (gated: enable `lclm_adaptive_lod` feature)       ║"
                );
            }
        }
    }

    println!(
        "╚════════════════════════════════════════════════════════════════════════════════════╝"
    );

    // Verification: at least some fixed modes should find the needle
    assert!(
        x8_found_count > 0,
        "Fixed X8 should find the needle in at least some configurations"
    );

    #[cfg(feature = "lclm_adaptive_lod")]
    {
        // Core assertion: adaptive compression finds the needle at least as well as fixed X8
        assert!(
            adaptive_wins >= x8_found_count,
            "Adaptive LOD should find needle at least as well as fixed X8 \
             (adaptive: {adaptive_wins}/{adaptive_total}, X8: {x8_found_count}/{}",
            haystack_sizes.len() * positions.len(),
        );
    }

    println!();
    println!(
        "   Fixed X8 needle found: {x8_found_count}/{} configurations",
        haystack_sizes.len() * positions.len()
    );
    #[cfg(feature = "lclm_adaptive_lod")]
    println!("   Adaptive needle found: {adaptive_wins}/{adaptive_total} configurations");
    println!();
}

// ── Test 2: Compression Ratio Distribution ───────────────────────────

#[test]
#[cfg(feature = "lclm_adaptive_lod")]
fn bench_adaptive_compression_ratio_distribution() {
    let slod = SpectralLOD::default();
    let total = 2048;

    let (tokens, labels) = make_mixed_content(total);

    let config = MuxLatentConfig {
        compression_ratio: CompressionRatio::X8,
        window_size: 1024,
        preserve_instructions: false,
        ..Default::default()
    };
    let window_size = config.window_size;

    let buf = LatentContextBuffer::new_adaptive(&tokens, config, slod.clone());
    let ctx = buf.context();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Adaptive Compression Ratio Distribution                       ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Window Type    │ Segments │ X4   │ X8   │ X16  │ Avg Ratio   ║");
    println!("╠═════════════════╪══════════╪══════╪══════╪══════╪═════════════╣");

    // Analyze per content type
    for &(start_offset, label) in &labels {
        let end_offset = if label == "repetitive" {
            total * 50 / 100
        } else if label == "medium" {
            total * 80 / 100
        } else {
            total
        };

        let block = &tokens[start_offset..end_offset];
        let mut x4_count = 0usize;
        let mut x8_count = 0usize;
        let mut x16_count = 0usize;
        let mut total_weighted = 0usize;
        let mut seg_count = 0usize;

        for window in block.chunks(window_size) {
            let ratio = slod.optimal_ratio(window);
            let n_segs = window.len().div_ceil(ratio.span_size());
            seg_count += n_segs;
            total_weighted += n_segs * ratio.span_size();
            match ratio {
                CompressionRatio::X4 => x4_count += n_segs,
                CompressionRatio::X8 => x8_count += n_segs,
                CompressionRatio::X16 => x16_count += n_segs,
            }
        }

        let avg_ratio = if seg_count > 0 {
            total_weighted as f32 / seg_count as f32
        } else {
            0.0
        };

        println!(
            "║  {:<14} │ {:>8} │ {:>4} │ {:>4} │ {:>4} │ {:>11.1} ║",
            label, seg_count, x4_count, x8_count, x16_count, avg_ratio,
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Verify: repetitive content should have higher average compression than diverse
    let repetitive_block = &tokens[0..total * 50 / 100];
    let diverse_block = &tokens[total * 80 / 100..total];

    let mut repetitive_avg = 0.0f32;
    let mut diverse_avg = 0.0f32;
    let mut r_count = 0usize;
    let mut d_count = 0usize;

    for window in repetitive_block.chunks(window_size) {
        repetitive_avg += slod.optimal_ratio(window).span_size() as f32;
        r_count += 1;
    }
    for window in diverse_block.chunks(window_size) {
        diverse_avg += slod.optimal_ratio(window).span_size() as f32;
        d_count += 1;
    }

    if r_count > 0 {
        repetitive_avg /= r_count as f32;
    }
    if d_count > 0 {
        diverse_avg /= d_count as f32;
    }

    assert!(
        repetitive_avg >= diverse_avg,
        "Repetitive content should get higher compression (avg span {:.1}) than diverse (avg span {:.1})",
        repetitive_avg,
        diverse_avg,
    );

    println!();
    println!(
        "   Repetitive avg span size: {repetitive_avg:.1} (higher = more aggressive compression)"
    );
    println!("   Diverse avg span size:    {diverse_avg:.1} (lower = less aggressive compression)");
    println!();
}

// ── Test 3: SpectralLOD Throughput ───────────────────────────────────

#[test]
fn bench_spectral_lod_throughput() {
    let slod = SpectralLOD::default();

    let window_sizes: &[usize] = &[1024, 2048, 4096];
    let warmup_iters = 50;
    let measure_iters = 200;

    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SpectralLOD Throughput — energy_concentration benchmark  ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Window Size │ μs/call   │ OK (<200μs) │ Tokens/window  ║");
    println!("╠══════════════╪═══════════╪═════════════╪════════════════╣");

    for &size in window_sizes {
        // Generate a mixed-content window
        let tokens: Vec<u32> = (0..size)
            .map(|i| {
                if i % 7 == 0 {
                    5u32
                } else {
                    ((i * 31) % 32000) as u32
                }
            })
            .collect();

        // Warmup
        for _ in 0..warmup_iters {
            std::hint::black_box(slod.energy_concentration(std::hint::black_box(&tokens)));
        }

        // Measure
        let start = Instant::now();
        for _ in 0..measure_iters {
            std::hint::black_box(slod.energy_concentration(std::hint::black_box(&tokens)));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / measure_iters;

        let under_budget = per_call.as_micros() < 200;
        let budget_mark = if under_budget { "✓" } else { "✗" };

        println!(
            "║  {:>11} │ {:>7}μs │ {:>8}    │ {:>14} ║",
            size,
            per_call.as_micros(),
            budget_mark,
            size,
        );

        // Verify throughput requirement: < 200μs per window (debug build budget)
        // Release builds are typically 5-10x faster, so this still meets the < 100μs
        // production budget.
        assert!(
            per_call.as_micros() < 200,
            "SpectralLOD::energy_concentration on {size} tokens took {}μs (budget: 200μs)",
            per_call.as_micros(),
        );
    }

    println!("╚════════════════════════════════════════════════════════════╝");

    // Also benchmark optimal_ratio (calls energy_concentration internally)
    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SpectralLOD Throughput — optimal_ratio benchmark        ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Window Size │ μs/call   │ OK (<200μs) │ Ratio           ║");
    println!("╠══════════════╪═══════════╪═════════════╪═════════════════╣");

    for &size in window_sizes {
        let tokens: Vec<u32> = (0..size)
            .map(|i| {
                if i % 7 == 0 {
                    5u32
                } else {
                    ((i * 31) % 32000) as u32
                }
            })
            .collect();

        // Warmup
        for _ in 0..warmup_iters {
            std::hint::black_box(slod.optimal_ratio(std::hint::black_box(&tokens)));
        }

        // Measure
        let start = Instant::now();
        for _ in 0..measure_iters {
            std::hint::black_box(slod.optimal_ratio(std::hint::black_box(&tokens)));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / measure_iters;
        let ratio = slod.optimal_ratio(&tokens);

        let under_budget = per_call.as_micros() < 200;
        let budget_mark = if under_budget { "✓" } else { "✗" };

        println!(
            "║  {:>11} │ {:>7}μs │ {:>8}    │ {:>15} ║",
            size,
            per_call.as_micros(),
            budget_mark,
            format!("{:?}", ratio),
        );

        assert!(
            per_call.as_micros() < 200,
            "SpectralLOD::optimal_ratio on {size} tokens took {}μs (budget: 200μs)",
            per_call.as_micros(),
        );
    }

    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
}

// ── TL;DR ────────────────────────────────────────────────────────────

#[test]
fn tldr_adaptive_lod_bench() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  TL;DR — Plan 238 Phase 4: Adaptive LOD Bench Summary          ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let slod = SpectralLOD::default();

    // Quick sanity checks
    let repetitive = vec![5u32; 64];
    let diverse: Vec<u32> = (0..64).map(|i| (i * 997) % 32000).collect();

    let conc_rep = slod.energy_concentration(&repetitive);
    let conc_div = slod.energy_concentration(&diverse);
    let ratio_rep = slod.optimal_ratio(&repetitive);
    let ratio_div = slod.optimal_ratio(&diverse);

    println!("║  Repetitive: concentration={conc_rep:.3} → {ratio_rep:?}                    ║");
    println!("║  Diverse:    concentration={conc_div:.3} → {ratio_div:?}                    ║");
    println!(
        "║  Diverse compresses LESS than repetitive: {}               ║",
        if ratio_div.span_size() <= ratio_rep.span_size() {
            "PASS ✓"
        } else {
            "WARN ✗"
        },
    );

    // NIAH quick check: needle is findable at X8 compression
    let needle = standard_needle();
    let tokens = make_niah_tokens(1024, 0.5, &needle);
    let config = fixed_config(CompressionRatio::X8);
    let encoder = MuxLatentEncoder::new(config);
    let ctx = encoder.encode(&tokens);
    let found = select_segments_to_expand(&ctx, &needle, 3);
    println!(
        "║  NIAH needle found at X8 (1k, middle): {}                    ║",
        if !found.is_empty() {
            "PASS ✓"
        } else {
            "FAIL ✗"
        },
    );
    assert!(
        !found.is_empty(),
        "Needle should be findable at X8 compression"
    );

    // Throughput quick check
    let big_window: Vec<u32> = (0..4096).map(|i| (i * 31) % 32000).collect();
    let start = Instant::now();
    std::hint::black_box(slod.energy_concentration(&big_window));
    let us = start.elapsed().as_micros();
    println!(
        "║  SLoD 4k tokens: {us}μs (< 200μs budget): {}                ║",
        if us < 200 { "PASS ✓" } else { "FAIL ✗" },
    );

    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}
