//! Routing / MoE benchmarks.
//!
//! Measures throughput of Raven slot routing primitives and cross-layer delta
//! routing, covering the "Route" feature dimension from the Paper Feature
//! Comparison Matrix.

use super::{BenchCategory, BenchResult};
use crate::transformer::{raven_compute_router, raven_readout, raven_update};
use crate::types::{Config, Rng};
use std::time::Instant;

/// Benchmark the Routing / MoE feature dimension.
///
/// Tests:
/// - Raven slot router computation (top-k softmax routing)
/// - Raven slot memory update (gated decay-write)
/// - Raven readout (attention over fixed slot memory)
/// - Delta routing (cross-layer residual routing simulation, feature-gated)
///
/// Returns BenchResult entries tagged with `feature_dim = "Route"`.
pub fn bench_routing(_config: &Config) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let warmup = 100;
    let iters = 5_000;

    println!("\n🧭 Routing / MoE...");
    println!("   ({iters} iterations, {warmup} warmup)");

    let draft_config = Config::draft();
    let kv_dim = crate::types::kv_dim(&draft_config);
    let num_slots: usize = 16;
    let top_k: usize = 4;

    let mut rng = Rng::new(42);

    // ── T1: Raven slot router ──
    {
        let raw_logits: Vec<f32> = (0..num_slots).map(|i| (i as f32 * 0.37).sin()).collect();

        for _ in 0..warmup {
            let _ = raven_compute_router(&raw_logits, top_k);
        }

        let start = Instant::now();
        for _ in 0..iters {
            let _ = raven_compute_router(&raw_logits, top_k);
        }
        let elapsed = start.elapsed();
        let tp = iters as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        results.push(BenchResult {
            label: format!("Raven router (slots={num_slots}, top_k={top_k})"),
            throughput: tp,
            time_per_step_us: us,
            avg_acceptance_len: 0.0,
            color: (138, 43, 226), // blue violet
            category: BenchCategory::Routing,
            feature_dim: "Route".into(),
        });
    }

    // ── T2: Raven slot update ──
    {
        let mut keys: Vec<f32> = (0..num_slots * kv_dim).map(|_| rng.normal()).collect();
        let mut values: Vec<f32> = (0..num_slots * kv_dim).map(|_| rng.normal()).collect();
        let new_key: Vec<f32> = (0..kv_dim).map(|_| rng.normal()).collect();
        let new_value: Vec<f32> = (0..kv_dim).map(|_| rng.normal()).collect();
        let r_t: Vec<f32> = {
            let logits: Vec<f32> = (0..num_slots).map(|i| (i as f32 * 0.37).sin()).collect();
            raven_compute_router(&logits, top_k)
        };
        let forget_rate = -1.0f32;

        for _ in 0..warmup {
            raven_update(
                &mut keys,
                &mut values,
                &new_key,
                &new_value,
                &r_t,
                forget_rate,
                num_slots,
                kv_dim,
            );
        }

        let start = Instant::now();
        for _ in 0..iters {
            raven_update(
                &mut keys,
                &mut values,
                &new_key,
                &new_value,
                &r_t,
                forget_rate,
                num_slots,
                kv_dim,
            );
        }
        let elapsed = start.elapsed();
        let tp = iters as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        results.push(BenchResult {
            label: format!("Raven update (slots={num_slots}, kv_dim={kv_dim})"),
            throughput: tp,
            time_per_step_us: us,
            avg_acceptance_len: 0.0,
            color: (255, 140, 0), // dark orange
            category: BenchCategory::Routing,
            feature_dim: "Route".into(),
        });
    }

    // ── T3: Raven readout ──
    {
        let query: Vec<f32> = (0..kv_dim).map(|_| rng.normal()).collect();
        let keys: Vec<f32> = (0..num_slots * kv_dim).map(|_| rng.normal()).collect();
        let values: Vec<f32> = (0..num_slots * kv_dim).map(|_| rng.normal()).collect();

        for _ in 0..warmup {
            let _ = raven_readout(&query, &keys, &values, num_slots, kv_dim);
        }

        let start = Instant::now();
        for _ in 0..iters {
            let _ = raven_readout(&query, &keys, &values, num_slots, kv_dim);
        }
        let elapsed = start.elapsed();
        let tp = iters as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        results.push(BenchResult {
            label: format!("Raven readout (slots={num_slots}, kv_dim={kv_dim})"),
            throughput: tp,
            time_per_step_us: us,
            avg_acceptance_len: 0.0,
            color: (0, 206, 209), // dark turquoise
            category: BenchCategory::Routing,
            feature_dim: "Route".into(),
        });
    }

    // ── T4: Delta routing (cross-layer residual) ──
    #[cfg(feature = "delta_routing")]
    {
        let layer_a: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let layer_b: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).cos()).collect();
        let mut output = vec![0.0f32; kv_dim];

        for _ in 0..warmup {
            delta_routing_op(&layer_a, &layer_b, &mut output);
        }

        let start = Instant::now();
        for _ in 0..iters {
            delta_routing_op(&layer_a, &layer_b, &mut output);
        }
        let elapsed = start.elapsed();
        let tp = iters as f64 / elapsed.as_secs_f64();
        let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

        results.push(BenchResult {
            label: format!("Delta routing (dim={kv_dim})"),
            throughput: tp,
            time_per_step_us: us,
            avg_acceptance_len: 0.0,
            color: (220, 20, 60), // crimson
            category: BenchCategory::Routing,
            feature_dim: "Route".into(),
        });
    }

    // Print summary
    println!("\n   {:<40} {:>12} {:>12}", "Method", "ops/s", "μs/op");
    println!("   {}", "-".repeat(66));
    for r in &results {
        println!(
            "   {:<40} {:>12.0} {:>12.2}",
            r.label, r.throughput, r.time_per_step_us,
        );
    }

    results
}

/// Cross-layer residual delta with sigmoid gating.
///
/// Computes: `output[i] = sigmoid(delta[i]) * delta[i]`
/// where `delta = layer_b - layer_a`.
#[cfg(feature = "delta_routing")]
fn delta_routing_op(layer_a: &[f32], layer_b: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        let delta = layer_b[i] - layer_a[i];
        let gate = 1.0 / (1.0 + (-delta).exp()); // sigmoid
        output[i] = gate * delta;
    }
}
