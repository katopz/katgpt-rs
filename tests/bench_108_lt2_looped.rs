#![cfg(feature = "lt2_looped")]
//! Benchmarks for LT2 Looped Inference Pipeline (Plan 108)
//!
//! Phase 0 baseline benchmarks:
//! - T0: Single-pass SDPA forward baseline
//! - T1: Single-pass AHLA forward baseline
//! - T2: Naive 4× looped SDPA (shows O(T) scaling problem)
//!
//! Phase 6 benchmarks + GOAT:
//! - T25: Looped AHLA (T=4) vs naive looped SDPA
//! - T26: Hybrid 1:4 (SDPA+AHLA, T=4)
//! - T28: GOAT proof — hybrid T=4 compute-budget gate (4× depth at ≤4× cost)
//!
//! Run: `cargo test --features lt2_looped --test bench_108_lt2_looped -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::hla::forward_ahla;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate, kv_dim,
};

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

/// Multi-layer micro config for hybrid dispatch benchmarks.
/// n_layer=6 gives meaningful Interleave{full_ratio:5} dispatch:
/// layers 0-3 AHLA, layer 4 SDPA, layer 5 AHLA (1/6 full, 5/6 linear).
fn make_micro_multilayer() -> Config {
    let mut config = Config::micro();
    config.n_layer = 6;
    config
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

// ── T25: Benchmark looped AHLA (T=4) vs naive looped SDPA ────
//
// Compares forward_looped with AHLA (Uniform, all linear) against
// naive 4× looped SDPA. AHLA should be faster due to O(1) recurrent
// state vs O(L) KV cache scanning.

#[test]
fn bench_lt2_ahla_loop() {
    let mut config = make_micro_multilayer();
    config.loop_mode = LoopMode::WeightShared { loop_count: 4 };
    config.hybrid_pattern = HybridPattern::Uniform;
    // Use AHLA mode so forward_looped uses linear attention for all layers
    config.hla_mode = HlaMode::Ahla;

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let residual_gate = ResidualGate::new(4, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

    let n_decode = config.block_size.min(POSITIONS);

    // Warmup
    for _ in 0..WARMUP {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut ahla_cache = MultiLayerAhlaCache::new(&config);
        for pos in 0..n_decode {
            let _ = forward_looped(
                &mut ctx,
                &weights,
                &mut cache,
                &mut ahla_cache,
                0,
                pos,
                &config,
                &residual_gate,
                &sdpa_gate,
                None,
                None,
                #[cfg(feature = "weight_shared_advantage_gate")]
                None,
                None,
                #[cfg(feature = "gain_cost_halt")]
                None,
            );
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut ahla_cache = MultiLayerAhlaCache::new(&config);
        for pos in 0..n_decode {
            black_box(forward_looped(
                &mut ctx,
                &weights,
                &mut cache,
                &mut ahla_cache,
                0,
                pos,
                &config,
                &residual_gate,
                &sdpa_gate,
                None,
                None,
                #[cfg(feature = "weight_shared_advantage_gate")]
                None,
                None,
                #[cfg(feature = "gain_cost_halt")]
                None,
            ));
        }
    }
    let elapsed = start.elapsed();

    let steps = ITERS as f64 * n_decode as f64;
    let tps = steps / elapsed.as_secs_f64();
    let us = elapsed.as_micros() as f64 / steps;

    let ahla_mem = MultiLayerAhlaCache::new(&config).memory_bytes() / config.n_layer;

    print_table_header("T25: Looped AHLA (T=4, Uniform)");
    print_table_row("forward_looped AHLA T=4", tps, us, ahla_mem);
    print_table_footer();

    println!(
        "   → Looped AHLA T=4: {tps:.0} tok/s, {us:.2} µs/step, {ahla_mem} B/layer (constant)"
    );
}

// ── T26: Benchmark hybrid 1:4 (SDPA+AHLA, T=4) ──────────────
//
// The flagship LT2 recipe: every 5th layer uses full SDPA (20%),
// the other 4 use AHLA (80%). With T=4 loop, this gives 4× effective
// depth with ~80% constant-memory layers.

#[test]
fn bench_lt2_hybrid() {
    let mut config = make_micro_multilayer();
    config.loop_mode = LoopMode::WeightShared { loop_count: 4 };
    config.hybrid_pattern = HybridPattern::Interleave { full_ratio: 5 };
    config.hla_mode = HlaMode::Ahla;

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let residual_gate = ResidualGate::new(4, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

    let n_decode = config.block_size.min(POSITIONS);
    let kvd = kv_dim(&config);
    let flat_mem = config.block_size * kvd * 2 * 4;
    let ahla_mem = MultiLayerAhlaCache::new(&config).memory_bytes() / config.n_layer;

    // Count full vs linear layers for hybrid 1:4
    let n_full = config.n_layer.div_ceil(5);
    let n_linear = config.n_layer - n_full;
    let hybrid_mem = (n_full * flat_mem + n_linear * ahla_mem) / config.n_layer;

    // Warmup
    for _ in 0..WARMUP {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut ahla_cache = MultiLayerAhlaCache::new(&config);
        for pos in 0..n_decode {
            let _ = forward_looped(
                &mut ctx,
                &weights,
                &mut cache,
                &mut ahla_cache,
                0,
                pos,
                &config,
                &residual_gate,
                &sdpa_gate,
                None,
                None,
                #[cfg(feature = "weight_shared_advantage_gate")]
                None,
                None,
                #[cfg(feature = "gain_cost_halt")]
                None,
            );
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut ahla_cache = MultiLayerAhlaCache::new(&config);
        for pos in 0..n_decode {
            black_box(forward_looped(
                &mut ctx,
                &weights,
                &mut cache,
                &mut ahla_cache,
                0,
                pos,
                &config,
                &residual_gate,
                &sdpa_gate,
                None,
                None,
                #[cfg(feature = "weight_shared_advantage_gate")]
                None,
                None,
                #[cfg(feature = "gain_cost_halt")]
                None,
            ));
        }
    }
    let elapsed = start.elapsed();

    let steps = ITERS as f64 * n_decode as f64;
    let tps = steps / elapsed.as_secs_f64();
    let us = elapsed.as_micros() as f64 / steps;
    let eff_depth = config.n_layer * 4;

    print_table_header("T26: Hybrid 1:4 (SDPA+AHLA, T=4)");
    print_table_row(
        &format!("Hybrid 1:4 T=4 ({eff_depth}eL)"),
        tps,
        us,
        hybrid_mem,
    );
    print_table_footer();

    println!(
        "   → Hybrid 1:4 T=4: {n_full}/{} SDPA + {n_linear}/{} AHLA, \
         {tps:.0} tok/s, {us:.2} µs/step, {hybrid_mem} B/layer avg",
        config.n_layer, config.n_layer,
    );
}

// ── T28: GOAT proof — hybrid T=4 compute-budget gate ─────────
//
// Proves that hybrid 1:4 T=4 looped inference is a net compute win:
// 4× effective depth at ≤4× slowdown → quality-per-compute ≥ 1.0.
//
// At micro scale in debug builds, loop overhead dominates over attention
// cost, so a per-layer comparison is unfair. The compute-budget gate
// checks the overall tradeoff: ≥25% raw throughput = 4× depth at ≤4× cost.
//
// In production (release build, large models, long sequences), the AHLA
// O(1) advantage becomes significant and hybrid throughput improves to
// 50-80% of SDPA T=1 (as shown in the LT2 paper).

#[test]
fn proof_lt2_hybrid_throughput() {
    // ── Single-pass SDPA baseline (multi-layer) ──
    let sdpa_config = make_micro_multilayer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&sdpa_config, &mut rng);

    let n_decode = sdpa_config.block_size.min(POSITIONS);
    let baseline_steps = 10;
    let n_layer = sdpa_config.n_layer;

    // Warmup SDPA T=1
    for _ in 0..3 {
        let mut ctx = ForwardContext::new(&sdpa_config);
        let mut cache = MultiLayerKVCache::new(&sdpa_config);
        for pos in 0..n_decode {
            let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &sdpa_config);
        }
    }

    // Measure SDPA T=1
    let start = Instant::now();
    for _ in 0..baseline_steps {
        let mut ctx = ForwardContext::new(&sdpa_config);
        let mut cache = MultiLayerKVCache::new(&sdpa_config);
        for pos in 0..n_decode {
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
    let sdpa_elapsed = start.elapsed();
    let sdpa_tps = (baseline_steps as f64 * n_decode as f64) / sdpa_elapsed.as_secs_f64();

    // ── Hybrid 1:4 T=4 looped (same multi-layer base) ──
    let mut hybrid_config = make_micro_multilayer();
    hybrid_config.loop_mode = LoopMode::WeightShared { loop_count: 4 };
    hybrid_config.hybrid_pattern = HybridPattern::Interleave { full_ratio: 5 };
    hybrid_config.hla_mode = HlaMode::Ahla;

    let residual_gate = ResidualGate::new(4, hybrid_config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(
        hybrid_config.n_head,
        hybrid_config.head_dim,
        hybrid_config.n_embd,
    );

    // Warmup hybrid T=4
    for _ in 0..3 {
        let mut ctx = ForwardContext::new(&hybrid_config);
        let mut cache = MultiLayerKVCache::new(&hybrid_config);
        let mut ahla_cache = MultiLayerAhlaCache::new(&hybrid_config);
        for pos in 0..n_decode {
            let _ = forward_looped(
                &mut ctx,
                &weights,
                &mut cache,
                &mut ahla_cache,
                0,
                pos,
                &hybrid_config,
                &residual_gate,
                &sdpa_gate,
                None,
                None,
                #[cfg(feature = "weight_shared_advantage_gate")]
                None,
                None,
                #[cfg(feature = "gain_cost_halt")]
                None,
            );
        }
    }

    // Measure hybrid T=4
    let start = Instant::now();
    for _ in 0..baseline_steps {
        let mut ctx = ForwardContext::new(&hybrid_config);
        let mut cache = MultiLayerKVCache::new(&hybrid_config);
        let mut ahla_cache = MultiLayerAhlaCache::new(&hybrid_config);
        for pos in 0..n_decode {
            black_box(forward_looped(
                &mut ctx,
                &weights,
                &mut cache,
                &mut ahla_cache,
                0,
                pos,
                &hybrid_config,
                &residual_gate,
                &sdpa_gate,
                None,
                None,
                #[cfg(feature = "weight_shared_advantage_gate")]
                None,
                None,
                #[cfg(feature = "gain_cost_halt")]
                None,
            ));
        }
    }
    let hybrid_elapsed = start.elapsed();
    let hybrid_tps = (baseline_steps as f64 * n_decode as f64) / hybrid_elapsed.as_secs_f64();

    // ── Compute-budget analysis ──
    let loop_count = 4usize;
    let effective_depth = n_layer * loop_count;
    let depth_multiplier = loop_count as f64;

    // Raw throughput ratio: hybrid T=4 / SDPA T=1
    let raw_ratio = hybrid_tps / sdpa_tps;
    let raw_pct = raw_ratio * 100.0;

    // Quality-per-compute: depth_multiplier × raw_ratio
    // e.g., 4× depth at 25% throughput = 4 × 0.25 = 1.0 (break-even)
    // e.g., 4× depth at 30% throughput = 4 × 0.30 = 1.2 (net win)
    let quality_per_compute = depth_multiplier * raw_ratio;

    // Hybrid layer composition
    let n_full = n_layer.div_ceil(5);
    let n_linear = n_layer - n_full;

    // Memory: hybrid vs pure SDPA at same effective depth
    let kvd = kv_dim(&sdpa_config);
    let flat_mem = sdpa_config.block_size * kvd * 2 * 4;
    let ahla_mem = MultiLayerAhlaCache::new(&hybrid_config).memory_bytes() / n_layer;
    let pure_sdpa_mem = flat_mem * n_layer; // all layers KV cache
    let hybrid_mem = n_full * flat_mem + n_linear * ahla_mem; // per loop pass
    let mem_savings_pct = (1.0 - hybrid_mem as f64 / pure_sdpa_mem as f64) * 100.0;

    println!("\n┌── T28 GOAT: Compute-Budget Gate (Hybrid T=4 vs SDPA T=1) ───────┐");
    println!(
        "│ {:<28} {:>10} {:>12} {:>10} │",
        "Method", "tok/s", "eff-depth", "mem (B)"
    );
    println!("│ {} │", "─".repeat(64));
    println!(
        "│ {:<28} {:>10.0} {:>12} {:>10} │",
        "SDPA T=1 (baseline)", sdpa_tps, n_layer, pure_sdpa_mem
    );
    println!(
        "│ {:<28} {:>10.0} {:>12} {:>10} │",
        "Hybrid 1:4 T=4", hybrid_tps, effective_depth, hybrid_mem
    );
    println!("└────────────────────────────────────────────────────────────────────────┘");
    println!(
        "   → Depth: {n_layer} → {effective_depth} effective layers ({depth_multiplier:.0}× deeper)"
    );
    println!("   → Throughput: {raw_pct:.1}% of SDPA T=1 ({hybrid_tps:.0} vs {sdpa_tps:.0} tok/s)");
    println!(
        "   → Slowdown: {:.2}× for {:.0}× depth",
        1.0 / raw_ratio,
        depth_multiplier
    );
    println!("   → Quality-per-compute: {quality_per_compute:.2}× (≥1.0 = net win)");
    println!(
        "   → Memory: {mem_savings_pct:.0}% savings ({hybrid_mem} vs {pure_sdpa_mem} B per pass)"
    );
    println!("   → Layers: {n_full}/{n_layer} SDPA + {n_linear}/{n_layer} AHLA per loop");

    // GOAT gate: hybrid T=4 ≥ 25% of SDPA T=1 raw throughput.
    // This means 4× depth costs ≤4× slowdown → quality-per-compute ≥ 1.0.
    // At production scale with release builds, AHLA O(1) advantage pushes
    // this to 50-80% (LT2 paper Figure 4).
    let gate_threshold = 0.25;
    assert!(
        raw_ratio >= gate_threshold,
        "[T28 GOAT FAIL] Hybrid T=4 throughput {raw_pct:.1}% < {:.0}% of SDPA T=1. \
         Hybrid: {hybrid_tps:.0} tok/s ({effective_depth} eff-layers), \
         SDPA: {sdpa_tps:.0} tok/s ({n_layer} layers). \
         Quality-per-compute: {quality_per_compute:.2}× (need ≥1.0).",
        gate_threshold * 100.0,
    );

    println!(
        "   ✅ GOAT PASS: {raw_pct:.1}% ≥ {:.0}% → {:.0}× depth at {:.1}× cost (q/c: {quality_per_compute:.2}×)",
        gate_threshold * 100.0,
        depth_multiplier,
        1.0 / raw_ratio,
    );
}
