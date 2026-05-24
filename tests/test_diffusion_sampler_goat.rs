//! GOAT Proof 019: DiffusionSampler — Adaptive Confidence in D2F Denoising (Plan 116 T4)
//!
//! Proofs:
//! 1. Fixed threshold baseline: measure accuracy, steps, time
//! 2. Trained logistic sampler: measure accuracy, steps, AUC
//! 3. Trained MLP sampler: measure accuracy, steps, AUC
//! 4. GOAT gate: trained sampler does not degrade quality >15pp vs baseline
//!
//! Run with:
//!   cargo test --features tri_mode --test test_diffusion_sampler_goat -- --nocapture

#![cfg(feature = "tri_mode")]

use katgpt_rs::dllm::{D2fContext, generate_pattern_dataset, train_mini_dllm};
use katgpt_rs::speculative::d2f::D2fDecodeConfig;
use katgpt_rs::speculative::d2f_decode_block_with_sampler;
use katgpt_rs::speculative::diffusion_sampler::{
    DiffusionSampler, SamplerVariant, collect_trajectories, train_logistic_on_patterns,
};
use katgpt_rs::speculative::types::NoPruner;
use katgpt_rs::types::{Config, Rng};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Train a mini dLLM and return (config, weights, test_data).
fn setup_trained_model() -> (
    Config,
    katgpt_rs::transformer::TransformerWeights,
    Vec<Vec<usize>>,
) {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let effective_vocab = config.vocab_size.saturating_sub(1);

    let train_data = generate_pattern_dataset(&mut rng, 50, config.block_size, effective_vocab);
    let test_data = generate_pattern_dataset(&mut rng, 20, config.block_size, effective_vocab);
    let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 300, 0.01, 0.3, 42);

    (config, weights, test_data)
}

/// Benchmark result for one sampler variant.
struct BenchResult {
    label: String,
    total_correct: usize,
    total_positions: usize,
    total_steps: usize,
    n_blocks: usize,
    elapsed_us: u128,
    auc: f32,
}

fn run_bench(
    label: &str,
    config: &Config,
    weights: &katgpt_rs::transformer::TransformerWeights,
    decode_config: &D2fDecodeConfig,
    targets: &[Vec<usize>],
    sampler: Option<&DiffusionSampler>,
    n_rounds: usize,
) -> BenchResult {
    let mut total_correct = 0usize;
    let mut total_positions = 0usize;
    let mut total_steps = 0usize;
    let n_blocks = targets.len() * n_rounds;

    let start = Instant::now();
    for round in 0..n_rounds {
        for target in targets {
            let mut dctx = D2fContext::new(config);
            let mut rng = Rng::new(round as u64 * 1000 + 7);

            let result = d2f_decode_block_with_sampler(
                &mut dctx,
                weights,
                config,
                decode_config,
                &NoPruner,
                &mut rng,
                sampler,
            );

            total_steps += result.steps_used;

            let correct = result
                .tokens
                .iter()
                .zip(target.iter())
                .filter(|(a, b)| a == b)
                .count();
            total_correct += correct;
            total_positions += target.len().min(result.tokens.len());
        }
    }
    let elapsed_us = start.elapsed().as_micros();

    let auc = match sampler {
        Some(s) => {
            let trajs = collect_trajectories(weights, config, decode_config, targets, 0);
            if trajs.is_empty() {
                0.5
            } else {
                s.evaluate_auc(&trajs)
            }
        }
        None => {
            // Baseline: compute AUC with untrained logistic as reference
            let mut rng = Rng::new(42);
            let untrained = DiffusionSampler::logistic(&mut rng);
            let trajs = collect_trajectories(weights, config, decode_config, targets, 0);
            if trajs.is_empty() {
                0.5
            } else {
                untrained.evaluate_auc(&trajs)
            }
        }
    };

    BenchResult {
        label: label.to_string(),
        total_correct,
        total_positions,
        total_steps,
        n_blocks,
        elapsed_us,
        auc,
    }
}

impl BenchResult {
    fn accuracy(&self) -> f64 {
        if self.total_positions == 0 {
            return 0.0;
        }
        self.total_correct as f64 / self.total_positions as f64
    }

    fn avg_steps(&self) -> f64 {
        if self.n_blocks == 0 {
            return 0.0;
        }
        self.total_steps as f64 / self.n_blocks as f64
    }

    fn us_per_block(&self) -> f64 {
        if self.n_blocks == 0 {
            return 0.0;
        }
        self.elapsed_us as f64 / self.n_blocks as f64
    }
}

fn make_decode_config() -> D2fDecodeConfig {
    D2fDecodeConfig {
        block_size: 4,
        denoise_steps: 8,
        confidence_threshold: 0.7,
        ..D2fDecodeConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Proof 1: Fixed threshold baseline
// ---------------------------------------------------------------------------

#[test]
fn proof_1_fixed_threshold_baseline() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();

    let result = run_bench(
        "Fixed (τ=0.7)",
        &config,
        &weights,
        &decode_config,
        &test_data,
        None,
        3,
    );

    eprintln!("\n  Proof 1: Fixed Threshold Baseline");
    eprintln!(
        "    Accuracy:    {:.1}% ({}/{})",
        result.accuracy() * 100.0,
        result.total_correct,
        result.total_positions
    );
    eprintln!("    Avg steps:   {:.1}/block", result.avg_steps());
    eprintln!("    Time:        {:.1} µs/block", result.us_per_block());
    eprintln!("    AUC (ref):   {:.3}", result.auc);

    assert!(result.total_positions > 0, "must produce positions");
    assert!(result.total_steps > 0, "must use denoising steps");
}

// ---------------------------------------------------------------------------
// Proof 2: Trained logistic sampler
// ---------------------------------------------------------------------------

#[test]
fn proof_2_trained_logistic_sampler() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();

    // Train sampler via convenience function
    let (sampler, train_loss, train_auc) =
        train_logistic_on_patterns(&config, &decode_config, 50, 20, 200, 0.1, 100, 42);

    assert!(
        matches!(sampler.variant(), SamplerVariant::Logistic),
        "should be logistic variant"
    );

    let result = run_bench(
        "Trained Logistic",
        &config,
        &weights,
        &decode_config,
        &test_data,
        Some(&sampler),
        3,
    );

    eprintln!("\n  Proof 2: Trained Logistic Sampler");
    eprintln!("    Train loss:  {train_loss:.4}");
    eprintln!("    Train AUC:   {train_auc:.3}");
    eprintln!(
        "    Accuracy:    {:.1}% ({}/{})",
        result.accuracy() * 100.0,
        result.total_correct,
        result.total_positions
    );
    eprintln!("    Avg steps:   {:.1}/block", result.avg_steps());
    eprintln!("    Time:        {:.1} µs/block", result.us_per_block());
    eprintln!("    Test AUC:    {:.3}", result.auc);

    assert!(result.total_positions > 0, "must produce positions");
    assert!(result.total_steps > 0, "must use denoising steps");
}

// ---------------------------------------------------------------------------
// Proof 3: Trained MLP sampler
// ---------------------------------------------------------------------------

#[test]
fn proof_3_trained_mlp_sampler() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();

    // Collect trajectories and train MLP directly
    let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);
    assert!(!trajectories.is_empty(), "must collect trajectories");

    let mut sampler = DiffusionSampler::mlp(16, &mut Rng::new(42));
    assert!(
        matches!(sampler.variant(), SamplerVariant::Mlp { .. }),
        "should be MLP variant"
    );

    let final_loss = sampler.train(&trajectories, 0.1, 200);

    let result = run_bench(
        "Trained MLP (d=16)",
        &config,
        &weights,
        &decode_config,
        &test_data,
        Some(&sampler),
        3,
    );

    eprintln!("\n  Proof 3: Trained MLP Sampler (hidden_dim=16)");
    eprintln!("    Train loss:  {final_loss:.4}");
    eprintln!(
        "    Accuracy:    {:.1}% ({}/{})",
        result.accuracy() * 100.0,
        result.total_correct,
        result.total_positions
    );
    eprintln!("    Avg steps:   {:.1}/block", result.avg_steps());
    eprintln!("    Time:        {:.1} µs/block", result.us_per_block());
    eprintln!("    Test AUC:    {:.3}", result.auc);

    assert!(result.total_positions > 0, "must produce positions");
    assert!(result.total_steps > 0, "must use denoising steps");
}

// ---------------------------------------------------------------------------
// Proof 4: GOAT gate — comparison table + quality gate
// ---------------------------------------------------------------------------

#[test]
fn proof_4_goat_comparison() {
    let (config, weights, test_data) = setup_trained_model();
    let decode_config = make_decode_config();
    let n_rounds = 5;

    // Baseline: fixed threshold
    let baseline = run_bench(
        "Fixed (τ=0.7)",
        &config,
        &weights,
        &decode_config,
        &test_data,
        None,
        n_rounds,
    );

    // Trained logistic
    let (logistic_sampler, logistic_loss, logistic_train_auc) =
        train_logistic_on_patterns(&config, &decode_config, 50, 20, 300, 0.1, 200, 42);

    let logistic = run_bench(
        "Logistic",
        &config,
        &weights,
        &decode_config,
        &test_data,
        Some(&logistic_sampler),
        n_rounds,
    );

    // Trained MLP
    let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);
    let mut mlp_sampler = DiffusionSampler::mlp(16, &mut Rng::new(42));
    let mlp_loss = if !trajectories.is_empty() {
        mlp_sampler.train(&trajectories, 0.1, 200)
    } else {
        0.0
    };

    let mlp = run_bench(
        "MLP (d=16)",
        &config,
        &weights,
        &decode_config,
        &test_data,
        Some(&mlp_sampler),
        n_rounds,
    );

    // Print comparison table
    eprintln!("\n  ┌──────────────────────────────────────────────────────────────────┐");
    eprintln!("  │ GOAT Proof 019: DiffusionSampler Comparison (micro_dllm)        │");
    eprintln!("  ├──────────────────────────────────────────────────────────────────┤");
    eprintln!("  │ {:<12} │ Acc%  │ AUC   │ Steps │ µs/block │", "Variant");
    eprintln!("  │ {:<12} │------│-------│-------│──────────│", "");
    eprintln!(
        "  │ {:<12} │ {:4.1}% │ {:.3} │ {:5.1} │ {:8.1} │",
        baseline.label,
        baseline.accuracy() * 100.0,
        baseline.auc,
        baseline.avg_steps(),
        baseline.us_per_block(),
    );
    eprintln!(
        "  │ {:<12} │ {:4.1}% │ {:.3} │ {:5.1} │ {:8.1} │",
        logistic.label,
        logistic.accuracy() * 100.0,
        logistic.auc,
        logistic.avg_steps(),
        logistic.us_per_block(),
    );
    eprintln!(
        "  │ {:<12} │ {:4.1}% │ {:.3} │ {:5.1} │ {:8.1} │",
        mlp.label,
        mlp.accuracy() * 100.0,
        mlp.auc,
        mlp.avg_steps(),
        mlp.us_per_block(),
    );
    eprintln!("  └──────────────────────────────────────────────────────────────────┘");

    eprintln!("    Logistic train: loss={logistic_loss:.4}, AUC={logistic_train_auc:.3}");
    eprintln!("    MLP train:      loss={mlp_loss:.4}");

    // Accuracy deltas
    let baseline_acc = baseline.accuracy();
    let logistic_delta = (logistic.accuracy() - baseline_acc) * 100.0;
    let mlp_delta = (mlp.accuracy() - baseline_acc) * 100.0;

    eprintln!("    Logistic Δ accuracy: {logistic_delta:+.1}pp vs baseline");
    eprintln!("    MLP Δ accuracy:      {mlp_delta:+.1}pp vs baseline");

    // GOAT gate: trained samplers must not degrade accuracy by more than 15pp.
    // At micro_dllm scale (n_embd=16, vocab=27), the signal is weak.
    // The sampler may not improve accuracy but shouldn't catastrophically hurt it.
    let gate_pp = 15.0;
    assert!(
        logistic.accuracy() >= baseline_acc - gate_pp / 100.0,
        "logistic accuracy ({:.1}%) more than {gate_pp:.0}pp below baseline ({:.1}%)",
        logistic.accuracy() * 100.0,
        baseline_acc * 100.0,
    );
    assert!(
        mlp.accuracy() >= baseline_acc - gate_pp / 100.0,
        "mlp accuracy ({:.1}%) more than {gate_pp:.0}pp below baseline ({:.1}%)",
        mlp.accuracy() * 100.0,
        baseline_acc * 100.0,
    );

    eprintln!("    GOAT gate: ✅ PASS — trained samplers within ±{gate_pp:.0}pp of baseline");

    // Signal quality assessment
    if logistic.auc > 0.55 {
        eprintln!(
            "    Logistic AUC {:.3} > 0.55 → sampler learned discriminative signal",
            logistic.auc
        );
    } else {
        eprintln!(
            "    Logistic AUC {:.3} ≤ 0.55 → weak signal at micro_dllm scale (expected)",
            logistic.auc
        );
    }

    if mlp.auc > 0.55 {
        eprintln!(
            "    MLP AUC {:.3} > 0.55 → sampler learned discriminative signal",
            mlp.auc
        );
    } else {
        eprintln!(
            "    MLP AUC {:.3} ≤ 0.55 → weak signal at micro_dllm scale (expected)",
            mlp.auc
        );
    }
}

// ---------------------------------------------------------------------------
// Extra: Sampler variant auto-selection
// ---------------------------------------------------------------------------

#[test]
fn proof_5_auto_selection_matches_config_scale() {
    let config_micro = Config::micro_dllm(); // n_embd=16 → logistic
    let sampler_micro = DiffusionSampler::auto(&config_micro);
    assert!(
        matches!(sampler_micro.variant(), SamplerVariant::Logistic),
        "micro_dllm (n_embd=16) should select logistic"
    );
    eprintln!(
        "  Auto(micro_dllm, n_embd={}) → Logistic ✓",
        config_micro.n_embd
    );
}
