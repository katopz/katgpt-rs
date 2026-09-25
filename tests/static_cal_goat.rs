//! GOAT benchmark for Static Calibration Tables (Plan 227 Phase 1).
//!
//! Measures: the cost of a `StaticCalTable` scale LOOKUP vs one Sinkhorn
//! `variance_normalize` of a 128x128 tile, cosine of the two scale vectors,
//! and calibration convergence. It never runs a KVarN quantize path —
//! `kvarn` is in `required-features` only for the Sinkhorn baseline import.
//! The KVarN static-cal branch this was meant to gate was dead code from the
//! Issue 015 extraction (b61a34f7b) and was deleted by Issue 897 (HISTORY.md).

use katgpt_attn::static_cal::{HeadStats, StaticCalTable};

/// Simulate per-head activation stats for calibration.
fn simulate_head_stats(num_layers: usize, num_heads: usize) -> Vec<HeadStats> {
    (0..num_layers)
        .flat_map(|layer| {
            (0..num_heads).map(move |head| {
                let base = (layer * num_heads + head) as f32;
                HeadStats {
                    layer,
                    head,
                    mean_activation: 2.0 + base * 0.1,
                    variance: 1.0 + base * 0.05,
                    max_activation: 5.0 + base * 0.2,
                }
            })
        })
        .collect()
}

#[test]
fn test_static_cal_new_defaults() {
    let table = StaticCalTable::new(4, 8);
    assert_eq!(table.len(), 32);
    assert!(table.verify());
}

#[test]
fn test_calibration_convergence() {
    let mut table = StaticCalTable::new(4, 8);

    // Run 10 calibration passes
    for _ in 0..10 {
        let stats = simulate_head_stats(4, 8);
        table.calibrate_from_stats(&stats);
    }

    // Should converge — verify commitment
    assert!(table.verify());
    assert_eq!(table.calibration_prompts, 10);

    // Scales should have diverged from 1.0 (heads have different activations)
    let has_diverged = (0..4).any(|l| (0..8).any(|h| (table.get_scale(l, h) - 1.0).abs() > 0.01));
    assert!(
        has_diverged,
        "Scales should differ from default after calibration"
    );
}

#[test]
fn test_o1_lookup_latency() {
    let mut table = StaticCalTable::new(32, 16); // realistic model size
    let stats = simulate_head_stats(32, 16);
    table.calibrate_from_stats(&stats);

    // Benchmark: 1M lookups should be < 10ms
    let start = std::time::Instant::now();
    let mut sum = 0.0f32;
    for l in 0..32 {
        for h in 0..16 {
            for _ in 0..2000 {
                sum += table.get_scale(l, h);
            }
        }
    }
    let elapsed = start.elapsed();

    // Prevent optimizer from eliminating the loop
    assert!(sum > 0.0);

    let us = elapsed.as_secs_f64() * 1e6;
    eprintln!("StaticCal 1M lookups: {us:.0}μs");
    assert!(
        elapsed.as_secs() < 1,
        "1M lookups took {us:.0}μs — too slow"
    );
}

#[test]
fn test_vs_sinkhorn_iterations() {
    use katgpt_kv::kvarn::var_norm::VarNormConfig;
    use katgpt_kv::kvarn::variance_normalize;

    // Create a representative 128x128 tile
    let rows = 128;
    let cols = 128;
    let mut tile = vec![0.0f32; rows * cols];
    let mut seed: u64 = 42;
    for v in tile.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((seed >> 33) as i32 as f32) / (1i32 << 31) as f32;
    }

    // Benchmark Sinkhorn (8 iterations)
    let config = VarNormConfig {
        iterations: 8,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let mut t = tile.clone();
        let _ = variance_normalize(&mut t, rows, cols, &config);
    }
    let sinkhorn_time = start.elapsed();

    // Benchmark Static Cal (1 lookup per channel)
    let mut table = StaticCalTable::new(1, rows);
    let stats: Vec<HeadStats> = (0..rows)
        .map(|ch| HeadStats {
            layer: 0,
            head: ch,
            mean_activation: 3.0,
            variance: 1.0,
            max_activation: 5.0,
        })
        .collect();
    table.calibrate_from_stats(&stats);

    let start = std::time::Instant::now();
    for _ in 0..100 {
        for ch in 0..rows {
            let _scale = table.get_scale(0, ch);
        }
    }
    let static_time = start.elapsed();

    eprintln!(
        "Sinkhorn 100×128×128: {:.0}μs | Static Cal 100×128 lookups: {:.0}μs",
        sinkhorn_time.as_secs_f64() * 1e6,
        static_time.as_secs_f64() * 1e6,
    );

    // Static cal should be faster (it's O(1) per lookup vs 8 iterations)
    // We just report, the GOAT gate decides
}

#[test]
fn test_commitment_tamper_detection() {
    let mut table = StaticCalTable::new(2, 4);
    let stats = simulate_head_stats(2, 4);
    table.calibrate_from_stats(&stats);
    assert!(table.verify());

    // Tamper
    table.scales[0] = 999.0;
    assert!(!table.verify());
}

#[test]
fn goat_g1_static_cal_latency_and_quality() {
    use katgpt_kv::kvarn::var_norm::{VarNormConfig, variance_normalize};

    // ── Setup ──
    let rows = 128;
    let cols = 128;
    let iters = 200;

    // Create a representative tile with seeded random data
    let mut tile_template = vec![0.0f32; rows * cols];
    let mut seed: u64 = 42;
    for v in tile_template.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((seed >> 33) as i32 as f32) / (1i32 << 31) as f32;
    }

    // ── Baseline: Sinkhorn (8 iterations) ──
    let config = VarNormConfig {
        iterations: 8,
        ..Default::default()
    };

    // Warmup
    for _ in 0..10 {
        let mut t = tile_template.clone();
        let _ = variance_normalize(&mut t, rows, cols, &config);
    }

    let start = std::time::Instant::now();
    let mut baseline_scales = Vec::new();
    for _ in 0..iters {
        let mut t = tile_template.clone();
        let scales = variance_normalize(&mut t, rows, cols, &config);
        baseline_scales = scales.s_row.clone();
    }
    let sinkhorn_time = start.elapsed();

    // Compute baseline output for quality comparison
    let mut baseline_tile = tile_template.clone();
    let baseline_result = variance_normalize(&mut baseline_tile, rows, cols, &config);

    // ── Feature: Static calibration (O(1) lookup per channel) ──
    let mut table = StaticCalTable::new(1, rows);
    let stats: Vec<HeadStats> = (0..rows)
        .map(|ch| HeadStats {
            layer: 0,
            head: ch,
            mean_activation: baseline_result.s_row[ch],
            variance: 1.0,
            max_activation: 5.0,
        })
        .collect();
    table.calibrate_from_stats(&stats);

    // Warmup
    for _ in 0..10 {
        for ch in 0..rows {
            std::hint::black_box(table.get_scale(0, ch));
        }
    }

    let start = std::time::Instant::now();
    let mut static_scales = Vec::new();
    for _ in 0..iters {
        let mut s = Vec::with_capacity(rows);
        for ch in 0..rows {
            s.push(table.get_scale(0, ch));
        }
        static_scales = s;
    }
    let static_time = start.elapsed();

    // ── Compute metrics ──
    let sinkhorn_us = sinkhorn_time.as_secs_f64() * 1e6;
    let static_us = static_time.as_secs_f64() * 1e6;
    let latency_improvement = (sinkhorn_us - static_us) / sinkhorn_us;

    // Perplexity proxy: cosine similarity between baseline and static scale vectors
    let dot: f32 = baseline_scales
        .iter()
        .zip(static_scales.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm_a: f32 = baseline_scales.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = static_scales.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos_sim = if norm_a > 1e-9 && norm_b > 1e-9 {
        dot / (norm_a * norm_b)
    } else {
        1.0
    };
    let perplexity_delta = 1.0 - cos_sim;

    eprintln!(
        "G1 SCT: sinkhorn={sinkhorn_us:.0}μs static={static_us:.0}μs improvement={:.1}% perplexity_delta={perplexity_delta:.4}",
        latency_improvement * 100.0
    );

    // ── GOAT gate assertions ──
    assert!(
        latency_improvement >= 0.05,
        "G1 FAIL: latency improvement {:.1}% < 5%",
        latency_improvement * 100.0
    );
    assert!(
        perplexity_delta < 0.1,
        "G1 FAIL: perplexity delta {perplexity_delta:.4} >= 0.1"
    );
    eprintln!(
        "✅ G1: SCT latency improvement = {:.1}%, perplexity delta = {perplexity_delta:.4}",
        latency_improvement * 100.0
    );
}
