//! Diffusion / Denoising benchmarks.
//!
//! Covers the "Diff" feature dimension from the Paper Feature Comparison Matrix:
//! - D2F block decode: single-block denoising throughput (feature-gated: `dllm`)
//! - D2F pipeline: multi-block sequential pipeline throughput (feature-gated: `dllm`)
//! - Confidence thresholding: token masking loop throughput (always available)

use super::{BenchCategory, BenchResult};
use std::time::Instant;

#[cfg(feature = "dllm")]
use crate::dllm::D2fContext;
#[cfg(feature = "dllm")]
use crate::speculative::d2f::{D2fDecodeConfig, D2fPipeline, d2f_decode_block_with};
#[cfg(feature = "dllm")]
use crate::speculative::types::{NoPruner as ConstraintPruner, NoScreeningPruner};
#[cfg(feature = "dllm")]
use crate::transformer::TransformerWeights;
#[cfg(feature = "dllm")]
use crate::types::{Config, Rng};

// ── D2F block decode benchmark ───────────────────────────────────

/// Benchmark single D2F block decoding (mask → forward → sample → remask).
///
/// Uses `Config::micro()` with `D2fDecodeConfig::speed()` (4 denoise steps,
/// block_size=8). Measures tokens/sec and μs/block.
#[cfg(feature = "dllm")]
fn bench_d2f_block_decode() -> BenchResult {
    let mut rng = Rng::new(42);
    let config = Config::micro();
    let weights = TransformerWeights::new(&config, &mut rng);
    let decode_config = D2fDecodeConfig::speed();
    let pruner = ConstraintPruner;
    let warmup = 10;
    let iters = 100;

    // Pre-allocate context for zero-alloc variant
    let mut dctx = D2fContext::new(&config);

    for _ in 0..warmup {
        d2f_decode_block_with(
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &pruner,
            &NoScreeningPruner,
            &mut rng,
        );
    }

    let start = Instant::now();
    for _ in 0..iters {
        d2f_decode_block_with(
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &pruner,
            &NoScreeningPruner,
            &mut rng,
        );
    }
    let elapsed = start.elapsed();

    let block_size = decode_config.block_size as f64;
    let tokens_per_sec = iters as f64 * block_size / elapsed.as_secs_f64();
    let us_per_block = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    BenchResult {
        label: "D2F block decode".into(),
        throughput: tokens_per_sec,
        time_per_step_us: us_per_block,
        avg_acceptance_len: block_size,
        color: (147, 112, 219), // medium purple
        category: BenchCategory::Diffusion,
        feature_dim: "Diff".into(),
    }
}

// ── D2F pipeline benchmark ───────────────────────────────────────

/// Benchmark full D2F pipeline: sequential multi-block decoding.
///
/// Uses `Config::micro()` with `D2fDecodeConfig::speed()`, decoding 32 tokens
/// (4 blocks × 8 block_size). Measures tokens/sec and μs/pipeline-run.
#[cfg(feature = "dllm")]
fn bench_d2f_pipeline() -> BenchResult {
    let mut rng = Rng::new(42);
    let config = Config::micro();
    let weights = TransformerWeights::new(&config, &mut rng);
    let decode_config = D2fDecodeConfig::speed();
    let pruner = ConstraintPruner;
    let total_len = 32; // 4 blocks × block_size=8
    let warmup = 10;
    let iters = 100;

    for _ in 0..warmup {
        let pipeline = D2fPipeline::new(&config, decode_config, total_len);
        pipeline.decode_all(&weights, &pruner, &NoScreeningPruner, &mut rng);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let pipeline = D2fPipeline::new(&config, decode_config, total_len);
        pipeline.decode_all(&weights, &pruner, &NoScreeningPruner, &mut rng);
    }
    let elapsed = start.elapsed();

    let tokens_per_sec = iters as f64 * total_len as f64 / elapsed.as_secs_f64();
    let us_per_run = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    BenchResult {
        label: "D2F pipeline (4 blocks)".into(),
        throughput: tokens_per_sec,
        time_per_step_us: us_per_run,
        avg_acceptance_len: total_len as f64,
        color: (186, 85, 211), // medium orchid
        category: BenchCategory::Diffusion,
        feature_dim: "Diff".into(),
    }
}

// ── Confidence thresholding benchmark ────────────────────────────

/// Benchmark confidence-based token masking.
///
/// Generates 256 random logits, applies softmax, then masks tokens below
/// a confidence threshold of 0.5. Measures thresholds/sec throughput.
fn bench_confidence_thresholding() -> BenchResult {
    let warmup = 100;
    let iters = 5_000;
    let dim = 256;
    let threshold = 0.5f32;

    // Pre-generate random logits (deterministic seed)
    let mut rng = fastrand::Rng::with_seed(42);
    let base_logits: Vec<f32> = (0..dim).map(|_| rng.f32() * 10.0 - 5.0).collect();

    // Inline softmax + mask loop (avoids pulling in transformer types)
    let apply_threshold = |logits: &mut [f32], thresh: f32| -> usize {
        // Softmax (SIMD batch)
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        katgpt_core::simd::simd_add_scalar_inplace(logits, -max_val);
        katgpt_core::simd::simd_exp_inplace(logits);
        let sum = katgpt_core::simd::simd_sum_f32(logits);
        let inv_sum = 1.0 / sum;
        katgpt_core::simd::simd_scale_inplace(logits, inv_sum);
        // Count masked tokens (probability below threshold)
        logits.iter().filter(|&&p| p < thresh).count()
    };

    // Pre-allocate reusable buffer to avoid per-iteration allocation
    let mut base_logits_buf = base_logits.clone();

    // Warmup
    for _ in 0..warmup {
        base_logits_buf.copy_from_slice(&base_logits);
        let _ = apply_threshold(&mut base_logits_buf, threshold);
    }

    // Bench
    let start = Instant::now();
    for _ in 0..iters {
        base_logits_buf.copy_from_slice(&base_logits);
        let _ = apply_threshold(&mut base_logits_buf, threshold);
    }
    let elapsed = start.elapsed();

    let tp = iters as f64 / elapsed.as_secs_f64();
    let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    BenchResult {
        label: "Confidence threshold (256)".into(),
        throughput: tp,
        time_per_step_us: us,
        avg_acceptance_len: 0.0,
        color: (255, 182, 193), // light pink
        category: BenchCategory::Diffusion,
        feature_dim: "Diff".into(),
    }
}

// ── Public entry point ───────────────────────────────────────────

/// Run all Diffusion / Denoising benchmarks.
///
/// D2F benchmarks are feature-gated behind `dllm`. Confidence thresholding
/// is always available.
pub fn bench_diffusion() -> Vec<BenchResult> {
    // Up to 3 results: d2f_block, d2f_pipeline, confidence_thresholding.
    let mut results = Vec::with_capacity(3);

    println!("\n🔬 Diffusion / Denoising Benchmarks...");

    // D2F block decode (feature-gated)
    #[cfg(feature = "dllm")]
    {
        println!("   D2F block decode (10 warmup, 100 iters)...");
        results.push(bench_d2f_block_decode());

        println!("   D2F pipeline (10 warmup, 100 iters)...");
        results.push(bench_d2f_pipeline());
    }

    // Confidence thresholding (always available)
    {
        println!("   Confidence thresholding (1_000 warmup, 50_000 iters)...");
        results.push(bench_confidence_thresholding());
    }

    // Print summary table
    println!("\n   {:<30} {:>12} {:>12}", "Method", "tok/s", "μs/step");
    println!("   {}", "-".repeat(56));
    for r in &results {
        println!(
            "   {:<30} {:>12.0} {:>12.2}",
            r.label, r.throughput, r.time_per_step_us,
        );
    }

    results
}
