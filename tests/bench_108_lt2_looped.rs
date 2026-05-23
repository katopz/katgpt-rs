#![cfg(feature = "lt2_looped")]
//! Benchmarks for LT2 Looped Inference Pipeline (Plan 108)
//!
//! Phase 0 baseline benchmarks:
//! - T0: Single-pass SDPA forward baseline
//! - T1: Single-pass AHLA forward baseline
//! - T2: Naive 4× looped SDPA (shows O(T) scaling problem)
//!
//! Run: `cargo test --features lt2_looped --test bench_108_lt2_looped -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use microgpt_rs::hla::MultiLayerAhlaCache;
use microgpt_rs::hla::forward_ahla;
use microgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use microgpt_rs::types::{Config, HlaMode, Rng, kv_dim};

// ── Constants ─────────────────────────────────────────────────

const WARMUP: usize = 5;
const ITERS: usize = 20;
const POSITIONS: usize = 8;

// ── Helpers ───────────────────────────────────────────────────

fn make_micro_sdpa() -> Config {
    let mut config = Config::micro();
    config.hla_mode = HlaMode::Standard;
    config
}

fn make_micro_ahla() -> Config {
    let mut config = Config::micro();
    config.hla_mode = HlaMode::Ahla;
    config
}

fn print_table_header(label: &str) {
    println!(
        "\n┌── {label} (micro, {WARMUP}+{ITERS}×{POSITIONS} pos) ──────────────────────────────┐"
    );
    println!(
        "│ {:<24} {:>10} {:>12} {:>14} │",
        "Method", "tok/s", "µs/step", "mem/layer (B)"
    );
    println!("│ {} │", "─".repeat(62));
}

fn print_table_row(label: &str, tps: f64, us: f64, mem: usize) {
    println!("│ {:<24} {:>10.1} {:>12.2} {:>14} │", label, tps, us, mem);
}

fn print_table_footer() {
    println!("└──────────────────────────────────────────────────────────────────────┘");
}

// ── T0: Benchmark single-pass SDPA forward baseline ──────────

#[test]
fn bench_forward_baseline() {
    let config = make_micro_sdpa();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Warmup
    for _ in 0..WARMUP {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        for pos in 0..POSITIONS {
            let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        for pos in 0..POSITIONS {
            black_box(forward(&mut ctx, &weights, &mut cache, 0, pos, &config));
        }
    }
    let elapsed = start.elapsed();

    let steps = ITERS as f64 * POSITIONS as f64;
    let tps = steps / elapsed.as_secs_f64();
    let us = elapsed.as_micros() as f64 / steps;

    // Memory per layer: block_size × kv_dim × 2 (key+value) × 4 (f32)
    let kvd = kv_dim(&config);
    let mem_per_layer = config.block_size * kvd * 2 * 4;

    print_table_header("T0: SDPA Forward Baseline");
    print_table_row("forward (flat KV)", tps, us, mem_per_layer);
    print_table_footer();

    println!("   → Baseline SDPA: {tps:.0} tok/s, {us:.2} µs/step, {mem_per_layer} B/layer");
}

// ── T1: Benchmark single-pass AHLA forward baseline ──────────

#[test]
fn bench_ahla_baseline() {
    let config = make_micro_ahla();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Warmup
    for _ in 0..WARMUP {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerAhlaCache::new(&config);
        for pos in 0..POSITIONS {
            let _ = forward_ahla(&mut ctx, &weights, &mut cache, 0, pos, &config);
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerAhlaCache::new(&config);
        for pos in 0..POSITIONS {
            black_box(forward_ahla(
                &mut ctx, &weights, &mut cache, 0, pos, &config,
            ));
        }
    }
    let elapsed = start.elapsed();

    let steps = ITERS as f64 * POSITIONS as f64;
    let tps = steps / elapsed.as_secs_f64();
    let us = elapsed.as_micros() as f64 / steps;

    // AHLA memory per layer (constant, no growth with sequence length)
    let ahla_mem = MultiLayerAhlaCache::new(&config).memory_bytes() / config.n_layer;

    print_table_header("T1: AHLA Forward Baseline");
    print_table_row("forward_ahla (constant)", tps, us, ahla_mem);
    print_table_footer();

    println!("   → AHLA constant state: {tps:.0} tok/s, {us:.2} µs/step, {ahla_mem} B/layer");
}

// ── T2: Benchmark naive 4× looped SDPA ────────────────────────
//
// Demonstrates the O(T) scaling problem: calling forward 4× with
// accumulating KV cache shows linear slowdown per loop iteration.

#[test]
fn bench_naive_loop() {
    let sdpa_config = make_micro_sdpa();
    let ahla_config = make_micro_ahla();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&sdpa_config, &mut rng);

    let kvd = kv_dim(&sdpa_config);
    let flat_mem = sdpa_config.block_size * kvd * 2 * 4;
    let ahla_mem = MultiLayerAhlaCache::new(&ahla_config).memory_bytes() / ahla_config.n_layer;

    print_table_header("T2: Naive Loop vs Single Pass");

    // ── Single-pass SDPA baseline ──
    for _ in 0..WARMUP {
        let mut ctx = ForwardContext::new(&sdpa_config);
        let mut cache = MultiLayerKVCache::new(&sdpa_config);
        for pos in 0..POSITIONS {
            let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &sdpa_config);
        }
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let mut ctx = ForwardContext::new(&sdpa_config);
        let mut cache = MultiLayerKVCache::new(&sdpa_config);
        for pos in 0..POSITIONS {
            black_box(forward(
                &mut ctx,
                &weights,
                &mut cache,
                0,
                pos,
                &sdpa_config,
            ));
        }
    }
    let elapsed_sdpa = start.elapsed();

    let steps = ITERS as f64 * POSITIONS as f64;
    let sdpa_tps = steps / elapsed_sdpa.as_secs_f64();
    let sdpa_us = elapsed_sdpa.as_micros() as f64 / steps;
    print_table_row("SDPA T=1 (baseline)", sdpa_tps, sdpa_us, flat_mem);

    // ── Naive 4× looped SDPA ──
    // Simulates looping by calling forward 4× at same position with growing KV cache.
    // This demonstrates the O(T) slowdown: attention scans T×positions entries.
    for _ in 0..WARMUP {
        let mut ctx = ForwardContext::new(&sdpa_config);
        let mut cache = MultiLayerKVCache::new(&sdpa_config);
        for pos in 0..POSITIONS {
            for _loop in 0..4 {
                let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &sdpa_config);
            }
        }
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let mut ctx = ForwardContext::new(&sdpa_config);
        let mut cache = MultiLayerKVCache::new(&sdpa_config);
        for pos in 0..POSITIONS {
            for _loop in 0..4 {
                black_box(forward(
                    &mut ctx,
                    &weights,
                    &mut cache,
                    0,
                    pos,
                    &sdpa_config,
                ));
            }
        }
    }
    let elapsed_loop = start.elapsed();

    let loop_steps = ITERS as f64 * POSITIONS as f64 * 4.0;
    let loop_tps = loop_steps / elapsed_loop.as_secs_f64();
    let loop_us = elapsed_loop.as_micros() as f64 / loop_steps;
    print_table_row("SDPA naive T=4 (4× fwd)", loop_tps, loop_us, flat_mem * 4);

    // ── Single-pass AHLA ──
    for _ in 0..WARMUP {
        let mut ctx = ForwardContext::new(&ahla_config);
        let mut cache = MultiLayerAhlaCache::new(&ahla_config);
        for pos in 0..POSITIONS {
            let _ = forward_ahla(&mut ctx, &weights, &mut cache, 0, pos, &ahla_config);
        }
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let mut ctx = ForwardContext::new(&ahla_config);
        let mut cache = MultiLayerAhlaCache::new(&ahla_config);
        for pos in 0..POSITIONS {
            black_box(forward_ahla(
                &mut ctx,
                &weights,
                &mut cache,
                0,
                pos,
                &ahla_config,
            ));
        }
    }
    let elapsed_ahla = start.elapsed();

    let ahla_tps = steps / elapsed_ahla.as_secs_f64();
    let ahla_us = elapsed_ahla.as_micros() as f64 / steps;
    print_table_row("AHLA T=1 (constant)", ahla_tps, ahla_us, ahla_mem);

    print_table_footer();

    let slow_ratio = sdpa_tps / loop_tps;
    println!("   → Naive T=4 loop is {slow_ratio:.1}× slower than T=1 SDPA");
    println!("   → This motivates hybrid SDPA+AHLA dispatch (constant memory AHLA layers)");
}
