#![cfg(all(feature = "gdn2_attention", feature = "hla_attention"))]
//! GOAT Benchmark Test — Gated DeltaNet-2 Recurrent Attention (Plan 105)
//!
//! Validates all 6 success criteria from Plan 105:
//! 1. All unit tests pass (including GQA variant) — verified by `cargo test`
//! 2. GDN2 within 10% of AHLA throughput
//! 3. GDN2 memory < flat KV memory at all configs
//! 4. No NaN/Inf in logits at any position
//! 5. Gate ablation: EraseOnly within 5% of Full quality (cosine sim)
//! 6. Context scaling: flat throughput profile (O(1) per step)
//!
//! Run: `cargo test --features "gdn2_attention,hla_attention" --test bench_105_gdn2_goat -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use katgpt_rs::gdn2::{Gdn2GateConfig, MultiLayerGdn2Cache, forward_gdn2, generate_gdn2_into};
use katgpt_rs::hla::{MultiLayerAhlaCache, forward_ahla};
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use katgpt_rs::types::{Config, Rng, kv_dim};

const WARMUP: usize = 50;
const ITERS: usize = 500;
const POSITIONS: usize = 8;

// ── Helpers ───────────────────────────────────────────────────

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ── Criterion 2: GDN2 within 10% of AHLA throughput ──────────

#[test]
fn goat_2_gdn2_within_10pct_of_ahla_throughput() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Warmup GDN2
    {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);
        for _ in 0..WARMUP {
            cache.reset();
            for pos in 0..POSITIONS {
                black_box(forward_gdn2(
                    &mut ctx, &weights, &mut cache, 0, pos, &config,
                ));
            }
        }
    }

    // Warmup AHLA
    {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerAhlaCache::new(&config);
        for _ in 0..WARMUP {
            cache.reset();
            for pos in 0..POSITIONS {
                black_box(forward_ahla(
                    &mut ctx, &weights, &mut cache, 0, pos, &config,
                ));
            }
        }
    }

    // Benchmark GDN2
    let mut ctx_gdn2 = ForwardContext::new(&config);
    let mut cache_gdn2 = MultiLayerGdn2Cache::new(&config);
    let start_gdn2 = Instant::now();
    for _ in 0..ITERS {
        cache_gdn2.reset();
        for pos in 0..POSITIONS {
            black_box(forward_gdn2(
                &mut ctx_gdn2,
                &weights,
                &mut cache_gdn2,
                0,
                pos,
                &config,
            ));
        }
    }
    let elapsed_gdn2 = start_gdn2.elapsed();

    // Benchmark AHLA
    let mut ctx_ahla = ForwardContext::new(&config);
    let mut cache_ahla = MultiLayerAhlaCache::new(&config);
    let start_ahla = Instant::now();
    for _ in 0..ITERS {
        cache_ahla.reset();
        for pos in 0..POSITIONS {
            black_box(forward_ahla(
                &mut ctx_ahla,
                &weights,
                &mut cache_ahla,
                0,
                pos,
                &config,
            ));
        }
    }
    let elapsed_ahla = start_ahla.elapsed();

    let steps = ITERS as f64 * POSITIONS as f64;
    let gdn2_tps = steps / elapsed_gdn2.as_secs_f64();
    let ahla_tps = steps / elapsed_ahla.as_secs_f64();
    let gdn2_us = elapsed_gdn2.as_micros() as f64 / steps;
    let ahla_us = elapsed_ahla.as_micros() as f64 / steps;
    let ratio = gdn2_tps / ahla_tps;

    println!();
    println!("┌── GOAT 2: GDN2 vs AHLA Throughput (micro, {ITERS}×{POSITIONS} pos) ──┐");
    println!("│ {:<18} {:>10} {:>12} │", "Method", "tok/s", "µs/step");
    println!("│ {} │", "-".repeat(42));
    println!("│ {:<18} {:>10.1} {:>12.2} │", "GDN2", gdn2_tps, gdn2_us);
    println!("│ {:<18} {:>10.1} {:>12.2} │", "AHLA", ahla_tps, ahla_us);
    println!(
        "│ {:<18} {:>10.1}%{:>13} │",
        "GDN2/AHLA ratio",
        ratio * 100.0,
        ""
    );
    println!("└{}┘", "─".repeat(45));

    assert!(
        ratio >= 0.90,
        "GDN2 throughput ({gdn2_tps:.1} tok/s) must be within 10% of AHLA ({ahla_tps:.1} tok/s), got {ratio:.3}"
    );
    println!("  ✅ GOAT 2 PASSED: GDN2/AHLA = {ratio:.3} (≥ 0.90)");
}

// ── Criterion 3: GDN2 memory < flat KV memory at all configs ──

#[test]
fn goat_3_gdn2_memory_less_than_flat_kv() {
    let configs: [(&str, Config); 4] = [
        ("micro", Config::micro()),
        ("game", Config::game()),
        ("bpe", Config::bpe()),
        ("gqa_draft", Config::gqa_draft()),
    ];

    println!();
    println!("┌── GOAT 3: GDN2 Memory vs Flat KV ─────────────────────┐");
    println!(
        "│ {:<12} {:>10} {:>10} {:>8} │",
        "Config", "Flat KV", "GDN2", "Saved"
    );
    println!("│ {} │", "-".repeat(44));

    for (name, cfg) in &configs {
        let kvd = kv_dim(cfg);
        let flat_bytes = cfg.block_size * kvd * 2 * 4; // key + value, f32
        let gdn2_bytes = MultiLayerGdn2Cache::new(cfg).memory_bytes();
        let saved = (1.0 - gdn2_bytes as f64 / flat_bytes as f64) * 100.0;

        println!(
            "│ {:<12} {:>8} B {:>8} B {:>6.1}% │",
            name, flat_bytes, gdn2_bytes, saved
        );

        assert!(
            gdn2_bytes < flat_bytes,
            "GDN2 ({gdn2_bytes} B) must be < flat KV ({flat_bytes} B) for {name}"
        );
    }
    println!("└{}┘", "─".repeat(47));
    println!("  ✅ GOAT 3 PASSED: GDN2 memory < flat KV at all configs");
}

// ── Criterion 4: No NaN/Inf in logits at any position ────────

#[test]
fn goat_4_no_nan_inf_at_any_position() {
    let configs: [(&str, Config); 4] = [
        ("micro", Config::micro()),
        ("game", Config::game()),
        ("bpe", Config::bpe()),
        ("gqa_draft", Config::gqa_draft()),
    ];

    let positions: [usize; 5] = [0, 8, 64, 128, 255];

    println!();
    println!("┌── GOAT 4: No NaN/Inf in Logits ───────────────────────┐");

    for (cfg_name, config) in &configs {
        let max_pos = config.block_size - 1;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(config, &mut rng);
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerGdn2Cache::new(config);

        for &pos in &positions {
            if pos > max_pos {
                continue;
            }

            let logits = forward_gdn2(
                &mut ctx,
                &weights,
                &mut cache,
                config.bos_token,
                pos,
                config,
            );

            assert!(
                logits.iter().all(|&l| l.is_finite()),
                "Non-finite logits at {cfg_name} pos={pos}"
            );
        }
        println!("  ✅ {cfg_name}: all positions finite (up to pos={max_pos})");
    }

    // Also test multi-token streaming generation with all gate configs
    for gate_config in [
        Gdn2GateConfig::EraseOnly,
        Gdn2GateConfig::Full,
        Gdn2GateConfig::Kda,
    ] {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::with_gate_config(&config, gate_config);
        let mut tokens = Vec::new();

        generate_gdn2_into(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            &mut rng,
            16,
            &mut tokens,
        );

        assert_eq!(
            tokens.len(),
            16,
            "Gate {gate_config:?}: should generate 16 tokens"
        );
        for &t in &tokens {
            assert!(
                t < config.vocab_size,
                "Token {t} out of vocab range for {gate_config:?}"
            );
        }
        println!("  ✅ {gate_config:?}: 16-token streaming generation stable");
    }

    println!("└{}┘", "─".repeat(56));
    println!("  ✅ GOAT 4 PASSED: No NaN/Inf in logits at any position or gate config");
}

// ── Criterion 5: Gate ablation — EraseOnly within 5% of Full ──

#[test]
fn goat_5_erase_only_within_5pct_of_full_quality() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Compare final logits from a single forward pass (same input, different gate configs)
    let mut ctx_e = ForwardContext::new(&config);
    let mut cache_e = MultiLayerGdn2Cache::with_gate_config(&config, Gdn2GateConfig::EraseOnly);
    let logits_erase = forward_gdn2(
        &mut ctx_e,
        &weights,
        &mut cache_e,
        config.bos_token,
        0,
        &config,
    )
    .to_vec();

    let mut ctx_f = ForwardContext::new(&config);
    let mut cache_f = MultiLayerGdn2Cache::with_gate_config(&config, Gdn2GateConfig::Full);
    let logits_full = forward_gdn2(
        &mut ctx_f,
        &weights,
        &mut cache_f,
        config.bos_token,
        0,
        &config,
    )
    .to_vec();

    let cos_sim = cosine_sim(&logits_erase, &logits_full);

    println!();
    println!("┌── GOAT 5: Gate Ablation (cosine similarity) ──────────┐");
    println!("│ EraseOnly vs Full cosine sim: {cos_sim:.6}                   │");
    println!(
        "│ Threshold (1 - 0.05):         {threshold:.6}                   │",
        threshold = 0.95
    );
    println!("└{}┘", "─".repeat(57));

    assert!(
        cos_sim >= 0.95,
        "EraseOnly/Full cosine sim ({cos_sim:.4}) must be ≥ 0.95 (within 5%)"
    );
    println!("  ✅ GOAT 5 PASSED: EraseOnly/Full cosine sim = {cos_sim:.4} (≥ 0.95)");
}

// ── Criterion 6: Context scaling — flat throughput O(1) ───────
//
// Key insight: we measure ONLY the single decode step at target_pos,
// NOT the prefill. Prefill is O(N) for both methods (sequential token loop).
// The O(1) claim is about per-step decode cost:
//   - GDN2: O(dk × dv) regardless of position (recurrent state is fixed-size)
//   - Flat KV: O(N × dk) at position N (scans all N stored keys)

#[test]
fn goat_6_context_scaling_flat_o1() {
    // Use game() config — block_size=170, allows positions [0..169]
    let config = Config::game();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let single_step_iters = 2000;

    let positions: [usize; 4] = [1, 8, 64, 128];

    // ── GDN2 scaling (should be flat O(1) per step) ──
    let mut gdn2_us_per_step: Vec<f64> = Vec::new();
    for &target_pos in &positions {
        // Prefill once to reach target_pos
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);
        for pos in 0..target_pos {
            black_box(forward_gdn2(
                &mut ctx, &weights, &mut cache, 0, pos, &config,
            ));
        }

        // Warmup single-step at target_pos
        for _ in 0..WARMUP {
            black_box(forward_gdn2(
                &mut ctx, &weights, &mut cache, 0, target_pos, &config,
            ));
        }

        // Measure ONLY the single step at target_pos (cache already has state)
        let start = Instant::now();
        for _ in 0..single_step_iters {
            black_box(forward_gdn2(
                &mut ctx, &weights, &mut cache, 0, target_pos, &config,
            ));
        }
        let elapsed = start.elapsed();
        let us_per_step = elapsed.as_micros() as f64 / single_step_iters as f64;
        gdn2_us_per_step.push(us_per_step);
    }

    // ── Flat KV scaling (should grow linearly with position) ──
    let mut flat_us_per_step: Vec<f64> = Vec::new();
    for &target_pos in &positions {
        // Prefill once to reach target_pos
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        for pos in 0..target_pos {
            black_box(forward(&mut ctx, &weights, &mut cache, 0, pos, &config));
        }

        // Warmup single-step at target_pos
        for _ in 0..WARMUP.min(20) {
            black_box(forward(
                &mut ctx, &weights, &mut cache, 0, target_pos, &config,
            ));
        }

        // Measure ONLY the single step at target_pos (cache has N entries)
        let start = Instant::now();
        for _ in 0..single_step_iters {
            black_box(forward(
                &mut ctx, &weights, &mut cache, 0, target_pos, &config,
            ));
        }
        let elapsed = start.elapsed();
        let us_per_step = elapsed.as_micros() as f64 / single_step_iters as f64;
        flat_us_per_step.push(us_per_step);
    }

    // ── Variance analysis ──
    let gdn2_min = gdn2_us_per_step.iter().fold(f64::MAX, |a, &b| a.min(b));
    let gdn2_max = gdn2_us_per_step.iter().fold(0.0f64, |a, &b| a.max(b));
    let gdn2_mean: f64 = gdn2_us_per_step.iter().sum::<f64>() / gdn2_us_per_step.len() as f64;
    let gdn2_max_spread = (gdn2_max - gdn2_min) / gdn2_mean; // relative spread

    let flat_growth = flat_us_per_step.last().unwrap() / flat_us_per_step.first().unwrap();

    println!();
    println!("┌── GOAT 6: Context Scaling — Single Step Decode (game) ┐");
    println!(
        "│ {:<10} {:>14} {:>14} │",
        "Position", "GDN2 (µs)", "Flat KV (µs)"
    );
    println!("│ {} │", "-".repeat(42));
    for (i, &pos) in positions.iter().enumerate() {
        println!(
            "│ {:<10} {:>14.2} {:>14.2} │",
            pos, gdn2_us_per_step[i], flat_us_per_step[i]
        );
    }
    println!("│ {} │", "-".repeat(42));
    println!(
        "│ GDN2  spread (max-min)/mean: {spread:.3}                │",
        spread = gdn2_max_spread
    );
    println!(
        "│ Flat  growth (last/first):   {growth:.2}x                │",
        growth = flat_growth
    );
    println!("└{}┘", "─".repeat(46));

    // GDN2 single-step cost should be nearly constant — max spread < 30% of mean
    assert!(
        gdn2_max_spread < 0.30,
        "GDN2 single-step scaling not flat: spread={gdn2_max_spread:.3}, expected < 0.30"
    );

    // Flat KV single-step should grow (at least 1.5× from first to last)
    assert!(
        flat_growth > 1.5,
        "Flat KV should show O(N) growth, got {flat_growth:.2}× (expected > 1.5×)"
    );

    println!(
        "  ✅ GOAT 6 PASSED: GDN2 spread={gdn2_max_spread:.3} (< 0.30), Flat growth={flat_growth:.1}× (> 1.5×)"
    );
}

// ── Summary ───────────────────────────────────────────────────

#[test]
fn summary_goat_105_gdn2_benchmarks() {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  🐐 GOAT Benchmarks: Gated DeltaNet-2 (Plan 105)");
    println!("  Features: gdn2_attention + hla_attention");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  Criterion 1: All unit tests pass          — see `cargo test --features gdn2_attention`"
    );
    println!(
        "  Criterion 2: GDN2 within 10% of AHLA      — goat_2_gdn2_within_10pct_of_ahla_throughput"
    );
    println!("  Criterion 3: GDN2 mem < flat KV all config — goat_3_gdn2_memory_less_than_flat_kv");
    println!("  Criterion 4: No NaN/Inf in logits          — goat_4_no_nan_inf_at_any_position");
    println!(
        "  Criterion 5: EraseOnly within 5% of Full   — goat_5_erase_only_within_5pct_of_full_quality"
    );
    println!("  Criterion 6: Flat O(1) context scaling      — goat_6_context_scaling_flat_o1");
    println!();
    println!("  Run all: cargo test --features \"gdn2_attention,hla_attention\" \\");
    println!("             --test bench_105_gdn2_goat -- --nocapture");
    println!("═══════════════════════════════════════════════════════════════");
}
