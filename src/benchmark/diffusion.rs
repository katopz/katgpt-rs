//! Diffusion / Denoising benchmarks.
//!
//! Covers the "Diff" feature dimension from the Paper Feature Comparison Matrix:
//! - D2F block decode: single-block denoising throughput (feature-gated: `dllm`)
//! - D2F pipeline: multi-block sequential pipeline throughput (feature-gated: `dllm`)
//! - Confidence thresholding: token masking loop throughput (always available)
//! - AR-ness × w cross-tab: decode-order statistics over a w-sweep
//!   (feature-gated: `decode_order_metrics`, Plan 602 T3.1)

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

// ── Plan 602 T3.1: AR-ness × w cross-tab ────────────────────────

/// The measured explanation axis for the Plans 379–384 SW-SetDLM w=0.5 win
/// (0.71 NLL): train ONE set-causal model under the SW-SetDLM default
/// (w=0.5, k=1.0 — the winner), then sweep the INFERENCE schedule width w
/// and cross-tab the decode-order statistics (ALR/AGR over the T1.3 π
/// logs) against NELBO, NFE, and convergence. The paper's regime claim
/// (arXiv:2609.20751): the winning hybrid schedules decode globally
/// left-to-right (high AGR) while locally going out-of-order (ALR ≈
/// 0.6–0.7) — this table is the axis that makes the w-choice measurable
/// instead of tuned.
///
/// Deterministic (seeded training + greedy temperature-0 decode); the
/// printed cross-tab is the deliverable, the per-w `BenchResult` rows carry
/// it into the harness summary. Run with `--release` for numbers worth
/// recording (the profile is part of the claim).
#[cfg(feature = "decode_order_metrics")]
pub fn bench_ar_ness_w_sweep() -> Vec<BenchResult> {
    use crate::dllm::{
        PositionOffsetSchedule, evaluate_set_causal_nelbo, generate_pattern_dataset,
        train_mini_set_causal,
    };
    use crate::speculative::set_diffusion::{
        CpuSetCausalForward, SetDiffusionConfig, set_diffusion_decode,
    };
    use katgpt_core::order_to_gen_steps;

    // L=8/V=8 — the validated GOAT-fixture shape (src/speculative/
    // set_diffusion.rs tests + Bench 809 Cell A both converge here; L=16
    // diverges to NaN at this lr — measured, not assumed).
    let seq_len = 8usize;
    let config = crate::types::Config::micro_dllm();

    // One training run under the SW-SetDLM default (w=0.5) — the sweep
    // measures train/infer schedule mismatch, exactly the Plans 379–384
    // question. Alternating-pattern data: bidirectionally predictable, so
    // every schedule CAN converge and the mismatch axis is isolated.
    let mut train_rng = crate::types::Rng::new(602);
    let train_data = generate_pattern_dataset(&mut train_rng, 100, seq_len, 8);
    let test_data = generate_pattern_dataset(&mut train_rng, 20, seq_len, 8);
    let sw_default = PositionOffsetSchedule::default();
    let (weights, _) = train_mini_set_causal(
        &config,
        &train_data,
        &test_data,
        300,
        0.01,
        &sw_default,
        602,
    );

    let forward = CpuSetCausalForward {
        weights: &weights,
        config: &config,
    };
    let decode_config = SetDiffusionConfig {
        mask_token: config.mask_token,
        vocab_size: config.vocab_size,
        denoise_steps: 8,
        confidence_threshold: 0.5,
        temperature: 0.0, // greedy — the only randomness is the ordering
        #[cfg(feature = "entropy_bounded_commit")]
        eb_gamma: -1.0,
        #[cfg(feature = "entropy_bounded_commit")]
        eb_max_commit: usize::MAX,
    };

    let ws = [0.1f32, 0.3, 0.5, 0.7, 0.9, 1.0];
    let n_decodes = 16usize;
    let mut order_rng = crate::types::Rng::new(6021);
    let mut decode_rng = crate::types::Rng::new(6022);
    let mut results = Vec::with_capacity(ws.len() + 1);

    // NOTE (axis semantics, from the decode loop): eligibility is CUMULATIVE
    // (`gen_step <= current_step`), so every permutation schedule decodes
    // with singleton outer steps — the w axis acts through the ORDER's
    // correlation with position (w→0.1 ≈ AR, w=1.0 ≈ uniform random). The
    // true parallel endpoint is the separate MDLM arm (all-zero gen steps:
    // everything eligible at once → ALR = 0 by the ties-discordant contract,
    // NFE ≈ denoise_steps) — appended below with the Bench-809 degeneracy
    // caveat on its NELBO (clean-token identity copy).
    println!("\n   AR-ness × w cross-tab (Plan 602 T3.1, trained at w=0.5):");
    println!(
        "   {:>6} {:>7} {:>7} {:>9} {:>5} {:>6} {:>10}",
        "w", "ALR", "AGR", "NELBO", "NFE", "conv", "us/decode"
    );
    println!("   {}", "-".repeat(56));

    for &w in &ws {
        let sched = PositionOffsetSchedule::new(w);
        let mut alr_sum = 0.0f32;
        let mut agr_sum = 0.0f32;
        let mut alr_n = 0usize;
        let mut agr_n = 0usize;
        let mut nfe_sum = 0usize;
        let mut converged_n = 0usize;
        let start = std::time::Instant::now();
        for _ in 0..n_decodes {
            let order = sched.sample_order_with(seq_len, || order_rng.uniform());
            let gen_steps = order_to_gen_steps(&order);
            let result =
                set_diffusion_decode(&forward, &decode_config, &[], &gen_steps, &mut decode_rng);
            // NaN statistics (nothing committed) are excluded, never averaged.
            let alr = result.local_ar_ness();
            let agr = result.global_ar_ness();
            if alr.is_finite() {
                alr_sum += alr;
                alr_n += 1;
            }
            if agr.is_finite() {
                agr_sum += agr;
                agr_n += 1;
            }
            nfe_sum += result.forward_passes;
            converged_n += result.converged as usize;
        }
        let elapsed = start.elapsed();
        let alr = alr_sum / alr_n.max(1) as f32;
        let agr = agr_sum / agr_n.max(1) as f32;
        let nfe = nfe_sum as f64 / n_decodes as f64;
        let us_per_decode = elapsed.as_secs_f64() * 1_000_000.0 / n_decodes as f64;

        // Quality axis: mean NELBO of the trained model under THIS schedule
        // (one sampled ordering per test sequence — the training-matched
        // estimator).
        let mut nelbo_rng = crate::types::Rng::new(6023);
        let nelbo =
            evaluate_set_causal_nelbo(&weights, &test_data, &config, &sched, &mut nelbo_rng);

        println!(
            "   {:>6.2} {:>7.3} {:>7.3} {:>9.4} {:>5.1} {:>4}/{} {:>10.1}",
            w, alr, agr, nelbo, nfe, converged_n, n_decodes, us_per_decode
        );
        results.push(BenchResult {
            label: format!("AR-ness w={w:.2} (ALR {alr:.2}/AGR {agr:.2})"),
            throughput: (n_decodes * seq_len) as f64 / elapsed.as_secs_f64(),
            time_per_step_us: us_per_decode,
            avg_acceptance_len: 0.0,
            color: (72, 209, 204), // medium turquoise
            category: BenchCategory::Diffusion,
            feature_dim: "Diff".into(),
        });
    }

    // MDLM arm — the parallel endpoint (all-zero gen steps: every position
    // eligible in outer step 0; commits are pass-0 ties → ALR/AGR → 0 by
    // the ties-discordant contract; NFE collapses to the inner budget).
    // Its NELBO is the Bench-809 degenerate floor (clean-token all-zero
    // gen steps = the trivial identity copy — REPORTED, never compared as
    // quality).
    {
        use katgpt_core::mdlm_gen_steps;
        let gen_steps = mdlm_gen_steps(seq_len);
        let mut alr_sum = 0.0f32;
        let mut agr_sum = 0.0f32;
        let mut alr_n = 0usize;
        let mut agr_n = 0usize;
        let mut nfe_sum = 0usize;
        let mut converged_n = 0usize;
        let start = std::time::Instant::now();
        for _ in 0..n_decodes {
            let result =
                set_diffusion_decode(&forward, &decode_config, &[], &gen_steps, &mut decode_rng);
            let alr = result.local_ar_ness();
            let agr = result.global_ar_ness();
            if alr.is_finite() {
                alr_sum += alr;
                alr_n += 1;
            }
            if agr.is_finite() {
                agr_sum += agr;
                agr_n += 1;
            }
            nfe_sum += result.forward_passes;
            converged_n += result.converged as usize;
        }
        let elapsed = start.elapsed();
        let alr = alr_sum / alr_n.max(1) as f32;
        let agr = agr_sum / agr_n.max(1) as f32;
        let nfe = nfe_sum as f64 / n_decodes as f64;
        let us_per_decode = elapsed.as_secs_f64() * 1_000_000.0 / n_decodes as f64;
        // Degenerate-floor NELBO via the all-zero reveal law.
        let mut nelbo_rng = crate::types::Rng::new(6024);
        let nelbo = crate::dllm::evaluate_set_causal_nelbo_with_gen_steps(
            &weights,
            &test_data,
            &config,
            &mut |_len, _tokens, _rng| mdlm_gen_steps(seq_len),
            &mut nelbo_rng,
        );
        println!(
            "   {:>6} {:>7.3} {:>7.3} {:>9.4} {:>5.1} {:>4}/{} {:>10.1}",
            "mdlm", alr, agr, nelbo, nfe, converged_n, n_decodes, us_per_decode
        );
        println!(
            "   (mdlm NELBO = the degenerate identity-copy floor — Bench 809; never a quality claim)"
        );
        results.push(BenchResult {
            label: format!("AR-ness mdlm (ALR {alr:.2}/AGR {agr:.2})"),
            throughput: (n_decodes * seq_len) as f64 / elapsed.as_secs_f64(),
            time_per_step_us: us_per_decode,
            avg_acceptance_len: 0.0,
            color: (72, 209, 204),
            category: BenchCategory::Diffusion,
            feature_dim: "Diff".into(),
        });
    }
    results
}

// ── Public entry point ───────────────────────────────────────────

/// Run all Diffusion / Denoising benchmarks.
///
/// D2F benchmarks are feature-gated behind `dllm`. Confidence thresholding
/// is always available.
pub fn bench_diffusion() -> Vec<BenchResult> {
    // Up to 3 results: d2f_block, d2f_pipeline, confidence_thresholding
    // (+ the Plan 602 T3.1 w-sweep rows when decode_order_metrics is on).
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

    // AR-ness × w cross-tab (Plan 602 T3.1, feature-gated)
    #[cfg(feature = "decode_order_metrics")]
    {
        println!("   AR-ness × w cross-tab (trains 1 set-causal model, sweeps 6 w values)...");
        results.extend(bench_ar_ness_w_sweep());
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
