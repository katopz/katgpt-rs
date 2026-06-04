//! Bench 176: Trigger Gate + Inference Router Performance (Plan 176)
//!
//! Benchmarks:
//! - TriggerGate evaluate() overhead
//! - TriggerGate record_inference() overhead
//! - InferenceRouter forward() overhead (CPU tier)
//! - InferenceRouter under simulated load (tier transitions)
//!
//! Run with:
//!   cargo test --test bench_176_trigger_gate --release -- --nocapture

use std::time::Instant;

use katgpt_rs::inference_router::InferenceRouter;
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use katgpt_rs::trigger_gate::{TriggerGate, TriggerGateConfig};
use katgpt_rs::types::{Config, Rng};

/// Fast gate config for benchmarks (tiny min interval so tier changes fire quickly).
fn fast_gate_config() -> TriggerGateConfig {
    TriggerGateConfig {
        gpu_activate_qps: 10_000.0,
        ane_activate_qps: 100_000.0,
        hysteresis_factor: 0.7,
        queue_depth_trigger: 100,
        latency_p99_trigger_us: 5000,
        min_tier_change_interval_ms: 10,
    }
}

/// Micro model fixtures for forward-pass benchmarks.
fn micro_fixtures() -> (
    Config,
    TransformerWeights,
    ForwardContext,
    MultiLayerKVCache,
) {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let ctx = ForwardContext::new(&config);
    let cache = MultiLayerKVCache::new(&config);
    (config, weights, ctx, cache)
}

#[test]
fn bench_176_gate_evaluate_overhead() {
    let gate = TriggerGate::new(fast_gate_config(), true, true);

    // Warmup
    for _ in 0..100 {
        let _ = gate.evaluate();
    }

    // Benchmark
    let n_iters = 10_000;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = gate.evaluate();
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

    println!(
        "Bench 176: TriggerGate evaluate() overhead: {:.3} µs/call",
        us_per_call
    );
    assert!(
        us_per_call < 1.0,
        "TriggerGate evaluate() too slow: {us_per_call:.3} µs/call (expected < 1 µs)"
    );
}

#[test]
fn bench_176_gate_record_inference_overhead() {
    let gate = TriggerGate::new(fast_gate_config(), true, true);

    // Warmup
    for _ in 0..100 {
        gate.record_inference(100);
    }

    // Benchmark
    let n_iters = 10_000;
    let start = Instant::now();
    for _ in 0..n_iters {
        gate.record_inference(100);
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

    println!(
        "Bench 176: TriggerGate record_inference() overhead: {:.3} µs/call",
        us_per_call
    );
    assert!(
        us_per_call < 0.5,
        "TriggerGate record_inference() too slow: {us_per_call:.3} µs/call (expected < 0.5 µs)"
    );
}

#[test]
fn bench_176_router_forward_cpu() {
    let (config, weights, mut ctx, mut cache) = micro_fixtures();
    let mut router = InferenceRouter::new(fast_gate_config(), Config::micro(), false, false);

    let block_size = config.block_size;

    // Warmup
    let _ = router.forward(&mut ctx, &weights, &mut cache, 0, 0);

    // --- Baseline: raw transformer::forward ---
    let n_iters = 1000;
    let start = Instant::now();
    for i in 0..n_iters {
        let pos = i % block_size;
        if pos == 0 && i > 0 {
            cache.reset();
        }
        let _ = katgpt_rs::transformer::forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
    }
    let baseline_elapsed = start.elapsed();
    let baseline_us = baseline_elapsed.as_secs_f64() * 1e6 / n_iters as f64;

    // Reset cache for router benchmark
    cache.reset();

    // --- Routed: InferenceRouter::forward ---
    let start = Instant::now();
    for i in 0..n_iters {
        let pos = i % block_size;
        if pos == 0 && i > 0 {
            cache.reset();
        }
        let _ = router.forward(&mut ctx, &weights, &mut cache, 0, pos);
    }
    let routed_elapsed = start.elapsed();
    let routed_us = routed_elapsed.as_secs_f64() * 1e6 / n_iters as f64;

    let overhead_us = routed_us - baseline_us;
    let overhead_pct = (overhead_us / baseline_us) * 100.0;

    println!(
        "Bench 176: InferenceRouter forward() (CPU tier): {:.2} µs/call (baseline: {:.2} µs, overhead: {:.2} µs / {:.1}%)",
        routed_us, baseline_us, overhead_us, overhead_pct
    );

    // Overhead should be < 20% of baseline (router adds evaluate() + timing + routing logic).
    assert!(
        overhead_pct < 20.0,
        "Router overhead too high: {overhead_pct:.1}% (baseline {baseline_us:.2} µs, routed {routed_us:.2} µs)"
    );
}

#[test]
fn bench_176_router_under_load() {
    let (config, weights, mut ctx, mut cache) = micro_fixtures();
    let block_size = config.block_size;
    let vocab_size = config.vocab_size;

    let mut router = InferenceRouter::new(fast_gate_config(), Config::micro(), true, true);

    // Simulate high QPS: run forward in a tight loop with queue depth above threshold.
    // Keep pos within block_size bounds and reset cache when wrapping.
    let n_iters = 2000;

    let start = Instant::now();
    for i in 0..n_iters {
        let pos = i % block_size;
        let token = i % vocab_size;
        if pos == 0 && i > 0 {
            cache.reset();
        }
        // Simulate queue pressure to encourage tier promotion.
        router.record_queue_depth(200);
        let _ = router.forward(&mut ctx, &weights, &mut cache, token, pos);
    }
    let elapsed = start.elapsed();

    let stats = router.stats();
    let us_per_call = elapsed.as_secs_f64() * 1e6 / n_iters as f64;
    let throughput = n_iters as f64 / elapsed.as_secs_f64();

    println!(
        "Bench 176: Router under simulated load ({} iters): {:.2} µs/call, {:.0} calls/sec",
        n_iters, us_per_call, throughput
    );
    println!(
        "           tier_transitions={}, total_inferences={}, estimated_qps={:.1}, tier={}",
        stats.tier_transitions, stats.total_inferences, stats.estimated_qps, stats.current_tier
    );

    // Verify the router tracked all inferences.
    assert_eq!(
        stats.total_inferences, n_iters as u64,
        "router should have recorded all {} inferences",
        n_iters
    );

    // Throughput should be reasonable (> 1000 calls/sec for micro model).
    assert!(
        throughput > 1000.0,
        "throughput too low: {throughput:.0} calls/sec"
    );
}

#[test]
fn bench_176_router_forward_batch() {
    let (config, weights, mut ctx, mut cache) = micro_fixtures();
    let block_size = config.block_size;
    let vocab_size = config.vocab_size;
    let mut router = InferenceRouter::new(fast_gate_config(), Config::micro(), false, false);

    // --- Baseline: sequential forward calls ---
    let batch_size = 8;
    let n_batches = 500;
    let start = Instant::now();
    for b in 0..n_batches {
        for i in 0..batch_size {
            let pos = (b * batch_size + i) % block_size;
            if pos == 0 && (b * batch_size + i) > 0 {
                cache.reset();
            }
            let _ = katgpt_rs::transformer::forward(
                &mut ctx,
                &weights,
                &mut cache,
                i % vocab_size,
                pos,
                &config,
            );
        }
    }
    let baseline_elapsed = start.elapsed();
    let baseline_us = baseline_elapsed.as_secs_f64() * 1e6 / (n_batches * batch_size) as f64;

    cache.reset();

    // --- Routed batch ---
    let start = Instant::now();
    for b in 0..n_batches {
        let offset = b * batch_size;
        let batch: Vec<(usize, usize)> = (0..batch_size)
            .map(|i| {
                let pos = (offset + i) % block_size;
                (i % vocab_size, pos)
            })
            .collect();
        let _ = router.forward_batch(&mut ctx, &weights, &mut cache, &batch);
    }
    let batch_elapsed = start.elapsed();
    let batch_us = batch_elapsed.as_secs_f64() * 1e6 / (n_batches * batch_size) as f64;

    let overhead_pct = ((batch_us - baseline_us) / baseline_us) * 100.0;

    println!(
        "Bench 176: forward_batch (batch_size={batch_size}): {:.2} µs/token (baseline: {:.2} µs, overhead: {:.1}%)",
        batch_us, baseline_us, overhead_pct
    );

    // Batch overhead should be < 15% (single evaluate() for entire batch).
    assert!(
        overhead_pct < 15.0,
        "Batch overhead too high: {overhead_pct:.1}%"
    );

    let stats = router.stats();
    assert_eq!(
        stats.total_inferences,
        (n_batches * batch_size) as u64,
        "batch router should track all inferences"
    );
}
