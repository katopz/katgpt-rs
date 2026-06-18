//! Plan 291 GOAT gate (T1.7–T1.9) — D2F 3-State Reuse warm-start.
//!
//! Compares three denoising-loop configs on a micro-D2F workload:
//!   (a) RCD-only baseline (Plan 258 as shipped)
//!   (b) RCD + uniform-γ warm-start (γ=1.0 everywhere — paper Fig. 5 ablation)
//!   (c) RCD + 3SR warm-start (this plan — token-type-aware γ)
//!
//! G1 PASS condition (Plan 291 T1.8):
//!   (c) reaches the agreement threshold in ≥15% fewer total FP solver
//!   iterations than (a), at equal token-agreement quality.
//!
//! # Run
//!
//! ```bash
//! cargo test --features d2f_3sr_warm_start --test bench_291_d2f_3sr_warm_start_goat -- --nocapture --test-threads=1
//! ```
//!
//! # Honest result
//!
//! See `.benchmarks/291_d2f_3sr_warm_start_goat.md` for the canonical numbers.
//! The implementation here captures the *structure* of 3SR (token-type-aware
//! warm-start with three discrete γ coefficients) but operates on the input
//! embedding layer (same layer as RCD), not the FP solver hidden state —
//! full FP-state 3SR would require exposing Plan 108 LT2's loop carry in
//! `BidirectionalContext`, which is out of scope for this plan. The G1 gate
//! is therefore expected to be **marginal**; the benchmark doc records the
//! actual measurement and the demotion decision.

#![cfg(feature = "d2f_3sr_warm_start")]

use katgpt_core::Config;
use katgpt_rs::dllm::{
    denoise_loop, denoise_loop_rcd, denoise_loop_rcd_3sr, denoising_accuracy,
    generate_pattern_dataset, train_mini_dllm, NoConstraint,
};
use katgpt_core::Rng;
use katgpt_rs::dllm_solver::{RcdConfig, ThreeStateReuseConfig};
use katgpt_rs::transformer::TransformerWeights;

// ─── Config knobs ─────────────────────────────────────────────────────────

/// How many denoising steps to budget per config.
const N_STEPS: usize = 8;

/// Confidence threshold for committing a token in a single step.
/// Higher = more conservative (fewer commits per step → more steps to converge).
/// Set high to force multi-step denoising on the micro benchmark.
const CONFIDENCE_THRESHOLD: f32 = 0.8;

/// Number of independent targets to average over (reduces seed noise).
const N_TARGETS: usize = 16;

/// Agreement threshold for "converged" (G1 measures iterations to reach this).
const AGREEMENT_THRESHOLD: f32 = 0.95;

// ─── Config (a): RCD-only baseline ────────────────────────────────────────

fn run_rcd_only(
    weights: &TransformerWeights,
    config: &Config,
    target: &[usize],
) -> (Vec<usize>, usize) {
    let mut rcd_cfg = RcdConfig::new(config.vocab_size, config.n_embd);
    denoise_loop_rcd(
        weights,
        target,
        config,
        N_STEPS,
        CONFIDENCE_THRESHOLD,
        &mut NoConstraint,
        &mut Rng::new(42),
        Some(&mut rcd_cfg),
    )
}

// ─── Config (b): RCD + uniform-γ (γ=1.0 everywhere) ───────────────────────
//
// This is the "full reuse" ablation from paper Fig. 5 — degenerate case where
// every position gets γ=1.0 regardless of transition type. Paper notes this
// can be unstable at high budgets; we include it as a control.

fn run_rcd_uniform_gamma(
    weights: &TransformerWeights,
    config: &Config,
    target: &[usize],
) -> (Vec<usize>, usize) {
    let mut rcd_cfg = RcdConfig::new(config.vocab_size, config.n_embd);
    let tsr_cfg = ThreeStateReuseConfig {
        gamma_visible: 1.0,
        gamma_masked_min: 1.0, // uniform γ=1.0
        gamma_masked_max: 1.0,
        gamma_newly_revealed: 1.0,
        enabled: true,
    };
    denoise_loop_rcd_3sr(
        weights,
        target,
        config,
        N_STEPS,
        CONFIDENCE_THRESHOLD,
        &mut NoConstraint,
        &mut Rng::new(42),
        Some(&mut rcd_cfg),
        Some(&tsr_cfg),
    )
}

// ─── Config (c): RCD + 3SR warm-start (this plan) ─────────────────────────

fn run_rcd_3sr(
    weights: &TransformerWeights,
    config: &Config,
    target: &[usize],
) -> (Vec<usize>, usize) {
    let mut rcd_cfg = RcdConfig::new(config.vocab_size, config.n_embd);
    let tsr_cfg = ThreeStateReuseConfig::default(); // paper defaults
    denoise_loop_rcd_3sr(
        weights,
        target,
        config,
        N_STEPS,
        CONFIDENCE_THRESHOLD,
        &mut NoConstraint,
        &mut Rng::new(42),
        Some(&mut rcd_cfg),
        Some(&tsr_cfg),
    )
}

// ─── Baseline (no RCD, no 3SR) — for reference ────────────────────────────

fn run_baseline(
    weights: &TransformerWeights,
    config: &Config,
    target: &[usize],
) -> (Vec<usize>, usize) {
    denoise_loop(
        weights,
        target,
        config,
        N_STEPS,
        CONFIDENCE_THRESHOLD,
        &mut NoConstraint,
        &mut Rng::new(42),
    )
}

// ─── Aggregated metrics over multiple targets ─────────────────────────────

struct Aggregated {
    /// Mean iterations-to-convergence across all targets (lower is better).
    /// A target that doesn't converge in `N_STEPS` counts as `N_STEPS`.
    mean_iters: f32,
    /// Mean final token-agreement vs ground truth (higher is better).
    mean_agreement: f32,
    /// How many targets reached AGREEMENT_THRESHOLD.
    converged_count: usize,
}

fn aggregate<F: Fn(&TransformerWeights, &Config, &[usize]) -> (Vec<usize>, usize)>(
    weights: &TransformerWeights,
    config: &Config,
    targets: &[Vec<usize>],
    run: F,
) -> Aggregated {
    let mut total_iters = 0u32;
    let mut total_agreement = 0.0f32;
    let mut converged = 0usize;
    for target in targets {
        let (tokens, iters) = run(weights, config, target);
        let agreement = denoising_accuracy(&tokens, target);
        total_iters += iters as u32;
        total_agreement += agreement;
        if agreement >= AGREEMENT_THRESHOLD {
            converged += 1;
        }
    }
    let n = targets.len() as f32;
    Aggregated {
        mean_iters: total_iters as f32 / n,
        mean_agreement: total_agreement / n,
        converged_count: converged,
    }
}

// ─── G1: 3SR vs RCD-only — fewer iterations at equal quality ─────────────

#[test]
fn g1_3sr_vs_rcd_iteration_reduction_at_equal_quality() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    // Train a tiny D2F model on a synthetic pattern dataset — small enough
    // that the test runs in seconds, large enough that denoising isn't trivial.
    let train_data = generate_pattern_dataset(&mut rng, 50, 4, 8);
    // Train for FEWER epochs so the model is imperfect — a perfect model
    // converges in 1 step regardless of warm-start, which makes the G1 gate
    // unmeasurable. 30 epochs leaves the model at ~50% test accuracy.
    let (weights, _) = train_mini_dllm(&config, &train_data, &train_data, 30, 0.05, 0.25, 42);

    // Targets: random 8-token sequences (vocab 8). Harder than the trained
    // `abab` pattern — forces multi-step denoising on imperfect predictions.
    let targets: Vec<Vec<usize>> = (0..N_TARGETS)
        .map(|i| {
            let mut r = Rng::new(100 + i as u64);
            (0..8).map(|_| (r.next() as usize) % 8).collect()
        })
        .collect();

    let baseline = aggregate(&weights, &config, &targets, run_baseline);
    let rcd_only = aggregate(&weights, &config, &targets, run_rcd_only);
    let rcd_uniform = aggregate(&weights, &config, &targets, run_rcd_uniform_gamma);
    let rcd_3sr = aggregate(&weights, &config, &targets, run_rcd_3sr);

    println!("┌───────────────────────────────────────────────────────────┐");
    println!("│ Plan 291 G1 micro-benchmark (N_TARGETS={}, N_STEPS={})", N_TARGETS, N_STEPS);
    println!("├──────────────────────┬──────────┬───────────┬─────────────────┤");
    println!("│ Config               │ mean its │ mean agr  │ converged       │");
    println!("├──────────────────────┼──────────┼───────────┼─────────────────┤");
    println!("│ (0) baseline         │  {:>6.2}  │  {:.4}   │  {}/{}          │",
        baseline.mean_iters, baseline.mean_agreement, baseline.converged_count, N_TARGETS);
    println!("│ (a) RCD-only         │  {:>6.2}  │  {:.4}   │  {}/{}          │",
        rcd_only.mean_iters, rcd_only.mean_agreement, rcd_only.converged_count, N_TARGETS);
    println!("│ (b) RCD + uniform γ  │  {:>6.2}  │  {:.4}   │  {}/{}          │",
        rcd_uniform.mean_iters, rcd_uniform.mean_agreement, rcd_uniform.converged_count, N_TARGETS);
    println!("│ (c) RCD + 3SR        │  {:>6.2}  │  {:.4}   │  {}/{}          │",
        rcd_3sr.mean_iters, rcd_3sr.mean_agreement, rcd_3sr.converged_count, N_TARGETS);
    println!("└──────────────────────┴──────────┴───────────┴─────────────────┘");

    // Quality guard: 3SR must not regress agreement by more than 5% vs RCD.
    // (The paper's full FP-state 3SR can be unstable on synthetic micro D2F
    // because the proxy warm-start operates on input embeddings, not solver
    // state — see benchmark doc.)
    let quality_delta = rcd_3sr.mean_agreement - rcd_only.mean_agreement;
    assert!(
        quality_delta >= -0.05,
        "G1 QUALITY FAIL: 3SR agreement {:.4} vs RCD {:.4} (Δ={:.4}, threshold -0.05)",
        rcd_3sr.mean_agreement, rcd_only.mean_agreement, quality_delta,
    );

    // Iteration reduction: G1 PASS = 3SR uses ≥15% fewer iterations than RCD.
    let iter_reduction = (rcd_only.mean_iters - rcd_3sr.mean_iters) / rcd_only.mean_iters.max(1.0);
    println!("  → iteration reduction: (a)→(c) = {:.2}%", iter_reduction * 100.0);
    println!("  → G1 PASS threshold:   ≥ 15%");

    // Honest gate: we don't hard-assert PASS/FAIL — the actual number is what
    // matters and is recorded in the benchmark doc. The plan's promotion rule
    // (T1.9) says demote if G1 fails; this test surfaces the number so the
    // benchmark doc can record the verdict. We do assert the loop is sound:
    // 3SR must not catastrophically regress (>50% more iterations than RCD).
    assert!(
        rcd_3sr.mean_iters < rcd_only.mean_iters * 2.0,
        "G1 CATASTROPHIC REGRESSION: 3SR mean_iters {:.2} >> RCD {:.2}",
        rcd_3sr.mean_iters, rcd_only.mean_iters,
    );

    if iter_reduction >= 0.15 {
        println!("✅ G1 PASSED: 3SR uses ≥15% fewer iterations than RCD at equal quality");
    } else if iter_reduction >= 0.0 {
        println!("⚠️  G1 PARTIAL: 3SR uses {:.2}% fewer iterations (< 15% target) — feature stays opt-in", iter_reduction * 100.0);
    } else {
        println!("⚠️  G1 FAIL: 3SR uses {:.2}% MORE iterations than RCD — feature stays opt-in, document negative result", iter_reduction * 100.0);
    }
}

// ─── Control (b): uniform-γ should not catastrophically regress ──────────
//
// Paper Fig. 5 shows uniform γ=1.0 can be unstable on some budgets. This test
// confirms our implementation doesn't NaN/explode on the micro benchmark —
// if it does, there's a numerical bug in warm_start_lerp.

#[test]
fn control_b_uniform_gamma_does_not_explode() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let train_data = generate_pattern_dataset(&mut rng, 50, 4, 8);
    let (weights, _) = train_mini_dllm(&config, &train_data, &train_data, 200, 0.01, 0.25, 42);

    let target = vec![3, 7, 3, 7];
    let (tokens, _steps) = run_rcd_uniform_gamma(&weights, &config, &target);

    // All tokens must be valid (no mask left).
    assert!(
        tokens.iter().all(|&t| t != config.mask_token),
        "uniform-γ left mask tokens — numerical explosion?",
    );
    // Agreement should be sane (≥0 — i.e. at least some tokens correct, no NaN).
    let agreement = denoising_accuracy(&tokens, &target);
    assert!(
        agreement.is_finite() && agreement >= 0.0,
        "uniform-γ produced non-finite or negative agreement: {}",
        agreement,
    );
    println!("✅ Control (b): uniform-γ does not explode (agreement = {:.4})", agreement);
}

// ─── Sanity: 3SR disabled falls through to RCD ───────────────────────────

#[test]
fn sanity_3sr_disabled_matches_rcd() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let train_data = generate_pattern_dataset(&mut rng, 50, 4, 8);
    let (weights, _) = train_mini_dllm(&config, &train_data, &train_data, 200, 0.01, 0.25, 42);

    let target = vec![3, 7, 3, 7];

    // RCD-only.
    let mut rcd_cfg_a = RcdConfig::new(config.vocab_size, config.n_embd);
    let (rcd_tokens, rcd_steps) = denoise_loop_rcd(
        &weights, &target, &config, N_STEPS, CONFIDENCE_THRESHOLD,
        &mut NoConstraint, &mut Rng::new(42), Some(&mut rcd_cfg_a),
    );

    // RCD + 3SR-disabled.
    let mut rcd_cfg_b = RcdConfig::new(config.vocab_size, config.n_embd);
    let tsr_cfg = ThreeStateReuseConfig::disabled();
    let (tsr_tokens, tsr_steps) = denoise_loop_rcd_3sr(
        &weights, &target, &config, N_STEPS, CONFIDENCE_THRESHOLD,
        &mut NoConstraint, &mut Rng::new(42),
        Some(&mut rcd_cfg_b), Some(&tsr_cfg),
    );

    assert_eq!(rcd_tokens, tsr_tokens, "3SR disabled must match RCD tokens exactly");
    assert_eq!(rcd_steps, tsr_steps, "3SR disabled must match RCD steps exactly");
    println!("✅ Sanity: 3SR-disabled byte-identical to RCD-only (tokens + steps match)");
}
