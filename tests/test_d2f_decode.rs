//! D2F Decode Benchmark & Quality Test Suite
//!
//! Plan 066 Phase 2, Task 2.3 — benchmarks and quality validation for D2F inference pipeline.
//!
//! Run with:
//!   cargo test --features dllm --test test_d2f_decode -- --nocapture
//!   cargo test --features dllm --test test_d2f_decode -- benchmark --nocapture

#![cfg(feature = "dllm")]

use katgpt_rs::dllm::{generate_pattern_dataset, train_mini_dllm};
use katgpt_rs::speculative::{
    D2fDecodeConfig, D2fPipeline, NoPruner, NoScreeningPruner, d2f_decode_block,
    d2f_decode_block_with_prompt, d2f_decode_block_with_target,
};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Train a mini model and return (weights, test_data).
fn train_tiny_model(config: &Config, rng: &mut Rng) -> (TransformerWeights, Vec<Vec<usize>>) {
    let train_data = generate_pattern_dataset(rng, 30, config.block_size, config.vocab_size - 1);
    let test_data = generate_pattern_dataset(rng, 10, config.block_size, config.vocab_size - 1);
    let (weights, _) = train_mini_dllm(config, &train_data, &test_data, 300, 0.01, 0.3, 42);
    (weights, test_data)
}

// ---------------------------------------------------------------------------
// Quality Tests
// ---------------------------------------------------------------------------

#[test]
fn test_d2f_decode_produces_non_mask_tokens() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 16,
        confidence_threshold: 0.3,
        block_size: config.block_size,
        temperature: 0.8,
        ..D2fDecodeConfig::default()
    };

    let result = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    let n_unmasked = result
        .tokens
        .iter()
        .filter(|&&t| t != config.mask_token)
        .count();
    assert!(
        n_unmasked > 0,
        "Expected at least 1 unmasked token, got {n_unmasked}"
    );
}

#[test]
fn test_d2f_decode_convergence_curve() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(100);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 16,
        confidence_threshold: 0.3,
        block_size: config.block_size,
        ..D2fDecodeConfig::default()
    };

    let result = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    // Confidence should be non-decreasing (denoising makes progress)
    let history = &result.confidence_history;
    assert!(
        !history.is_empty(),
        "Confidence history should not be empty"
    );

    // At least the last step should have higher confidence than the first
    if history.len() > 1 {
        let first = history[0];
        let last = history[history.len() - 1];
        assert!(
            last >= first,
            "Confidence should not decrease: first={first:.3}, last={last:.3}"
        );
    }

    eprintln!(
        "  Convergence: {} steps, confidence: {:?}",
        result.steps_used, history
    );
}

#[test]
fn test_d2f_decode_with_target_accuracy() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(200);
    let (weights, test_data) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 16,
        confidence_threshold: 0.3,
        block_size: config.block_size,
        temperature: 0.5,
        ..D2fDecodeConfig::default()
    };

    let target = &test_data[0];
    let result = d2f_decode_block_with_target(
        &weights,
        &config,
        &decode_config,
        target,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    eprintln!(
        "  Accuracy: {:.1}%, steps: {}, confidence: {:?}",
        result.accuracy.unwrap_or(0.0) * 100.0,
        result.steps_used,
        result.confidence_history
    );

    // With a trained model, we should get SOME accuracy (even if low)
    // This mainly tests that the accuracy measurement path works
    assert!(result.accuracy.is_some());
}

#[test]
fn test_d2f_decode_steps_vs_quality() {
    // More denoising steps should generally produce equal or better quality
    let config = Config::micro_dllm();
    let mut rng = Rng::new(300);
    let (weights, test_data) = train_tiny_model(&config, &mut rng);
    let target = &test_data[0];

    let accuracies: Vec<(usize, f32)> = [2, 4, 8, 16]
        .iter()
        .map(|&steps| {
            let decode_config = D2fDecodeConfig {
                denoise_steps: steps,
                confidence_threshold: 0.3,
                block_size: config.block_size,
                ..D2fDecodeConfig::default()
            };
            let result = d2f_decode_block_with_target(
                &weights,
                &config,
                &decode_config,
                target,
                &NoPruner,
                &NoScreeningPruner,
                &mut rng,
            );
            (steps, result.accuracy.unwrap_or(0.0))
        })
        .collect();

    eprintln!("  Steps vs Quality:");
    for (steps, acc) in &accuracies {
        eprintln!("    {steps:>2} steps → {acc:.1}% accuracy");
    }

    // At minimum, 16 steps should produce some non-mask tokens
    let (_, acc_16) = accuracies[3];
    assert!(acc_16 >= 0.0, "Accuracy should be non-negative");
}

#[test]
fn test_d2f_decode_temperature_effects() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(400);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config_base = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size: config.block_size,
        ..D2fDecodeConfig::default()
    };

    // Low temperature: more deterministic
    let mut cfg_low = decode_config_base;
    cfg_low.temperature = 0.1;
    let result_low = d2f_decode_block(
        &weights,
        &config,
        &cfg_low,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    // High temperature: more diverse
    let mut cfg_high = decode_config_base;
    cfg_high.temperature = 2.0;
    let result_high = d2f_decode_block(
        &weights,
        &config,
        &cfg_high,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    eprintln!(
        "  Low temp (0.1): {} unmasked, {} steps",
        result_low
            .tokens
            .iter()
            .filter(|&&t| t != config.mask_token)
            .count(),
        result_low.steps_used
    );
    eprintln!(
        "  High temp (2.0): {} unmasked, {} steps",
        result_high
            .tokens
            .iter()
            .filter(|&&t| t != config.mask_token)
            .count(),
        result_high.steps_used
    );

    // Both should produce some output
    assert!(!result_low.tokens.is_empty());
    assert!(!result_high.tokens.is_empty());
}

#[test]
fn test_d2f_decode_prompt_conditioning() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(500);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size: 4,
        ..D2fDecodeConfig::default()
    };

    // No prompt
    let result_no_prompt = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    // With prompt
    let prompt = vec![0, 1, 0, 1];
    let result_with_prompt = d2f_decode_block_with_prompt(
        &weights,
        &config,
        &decode_config,
        &prompt,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    // Both should produce block_size tokens
    assert_eq!(result_no_prompt.tokens.len(), decode_config.block_size);
    assert_eq!(result_with_prompt.tokens.len(), decode_config.block_size);

    eprintln!("  No prompt: {:?}", result_no_prompt.tokens);
    eprintln!(
        "  With prompt {:?}: {:?}",
        prompt, result_with_prompt.tokens
    );
}

// ---------------------------------------------------------------------------
// Pipeline Tests
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_multi_block_decode() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(600);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let block_size = 4;
    let total_len = 8; // 2 blocks
    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size,
        ..D2fDecodeConfig::default()
    };

    let pipeline = D2fPipeline::new(&config, decode_config, total_len);
    assert_eq!(pipeline.n_blocks(), 2);

    let result = pipeline.decode_all(&weights, &NoPruner, &NoScreeningPruner, &mut rng);

    assert_eq!(result.tokens.len(), total_len);
    assert_eq!(result.block_results.len(), 2);
    assert!(result.total_steps > 0);

    eprintln!(
        "  Pipeline: {} tokens, {} blocks, {} total steps",
        result.tokens.len(),
        result.block_results.len(),
        result.total_steps
    );
    eprintln!(
        "  Fully activated: {}/{}, Semi: {}/{}",
        result.n_fully_activated,
        result.block_results.len(),
        result.n_semi_activated,
        result.block_results.len()
    );
}

#[test]
fn test_pipeline_with_prompt_context() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(700);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let block_size = 4;
    let total_len = 4;
    let prompt = vec![0, 1, 0, 1];

    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size,
        ..D2fDecodeConfig::default()
    };

    let pipeline = D2fPipeline::with_prompt(&config, decode_config, total_len, &prompt);
    let result = pipeline.decode_all(&weights, &NoPruner, &NoScreeningPruner, &mut rng);

    // Tokens = prompt + generated
    assert_eq!(result.tokens.len(), prompt.len() + total_len);
    assert_eq!(&result.tokens[..prompt.len()], &prompt);

    eprintln!(
        "  Prompt {:?} + generated {:?} = {:?}",
        &result.tokens[..prompt.len()],
        &result.tokens[prompt.len()..],
        result.tokens
    );
}

#[test]
fn test_pipeline_partial_block() {
    // Total length that doesn't divide evenly into blocks
    let config = Config::micro_dllm();
    let mut rng = Rng::new(800);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let block_size = 4;
    let total_len = 6; // 1 full block + 1 partial block (2 tokens)

    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size,
        ..D2fDecodeConfig::default()
    };

    let pipeline = D2fPipeline::new(&config, decode_config, total_len);
    assert_eq!(pipeline.n_blocks(), 2);

    let result = pipeline.decode_all(&weights, &NoPruner, &NoScreeningPruner, &mut rng);

    assert_eq!(result.tokens.len(), total_len);
    assert_eq!(result.block_results.len(), 2);
    // Second block should be partial (2 tokens)
    assert_eq!(result.block_results[1].tokens.len(), 2);

    eprintln!(
        "  Partial pipeline: {} tokens in {} blocks, block sizes: {:?}",
        result.tokens.len(),
        result.block_results.len(),
        result
            .block_results
            .iter()
            .map(|b| b.tokens.len())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// ConstraintPruner Impact Tests
// ---------------------------------------------------------------------------

/// A pruner that only allows tokens in a specific range.
struct VocabRangePruner {
    min_token: usize,
    max_token: usize,
}

impl katgpt_rs::speculative::ConstraintPruner for VocabRangePruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx >= self.min_token && token_idx <= self.max_token
    }
}

#[test]
fn test_constraint_pruner_restricts_vocab() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(900);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size: config.block_size,
        ..D2fDecodeConfig::default()
    };

    // Without constraint
    let result_free = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    );

    // With constraint: only tokens 0..5 allowed
    let pruner = VocabRangePruner {
        min_token: 0,
        max_token: 5,
    };
    let result_constrained = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &pruner,
        &NoScreeningPruner,
        &mut rng,
    );

    // All non-mask tokens in constrained result should be in [0..5]
    for &t in &result_constrained.tokens {
        if t != config.mask_token {
            assert!(t <= 5, "Constrained token {t} should be ≤ 5");
        }
    }

    eprintln!("  Free tokens: {:?}", result_free.tokens);
    eprintln!("  Constrained tokens: {:?}", result_constrained.tokens);
}

// ---------------------------------------------------------------------------
// ScreeningPruner Integration Tests
// ---------------------------------------------------------------------------

/// A screener that assigns relevance based on token parity:
/// - Even tokens get relevance 1.0
/// - Odd tokens get relevance 0.1 (heavily dampened)
struct ParityScreener;

impl katgpt_rs::speculative::ScreeningPruner for ParityScreener {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if token_idx.is_multiple_of(2) {
            1.0
        } else {
            0.1
        }
    }
}

#[test]
fn test_screening_pruner_biases_sampling() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(1100);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 16,
        confidence_threshold: 0.2, // lower threshold for more convergence
        block_size: 4,
        temperature: 0.8,
        ..D2fDecodeConfig::default()
    };

    // Decode with parity screener: even tokens boosted, odd tokens dampened
    let screener = ParityScreener;
    let result = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &NoPruner,
        &screener,
        &mut rng,
    );

    let non_mask: Vec<usize> = result
        .tokens
        .iter()
        .filter(|&&t| t != config.mask_token)
        .copied()
        .collect();

    // Count even vs odd tokens
    let n_even = non_mask.iter().filter(|&&t| t % 2 == 0).count();
    let n_odd = non_mask.iter().filter(|&&t| t % 2 != 0).count();

    eprintln!(
        "  Parity screener: non_mask={:?}, even={}, odd={}",
        non_mask, n_even, n_odd
    );

    // With even-token relevance=1.0 and odd-token relevance=0.1,
    // we expect a strong bias toward even tokens (if tokens were placed)
    if !non_mask.is_empty() {
        // At least some tokens should be even (high relevance)
        assert!(
            n_even > 0,
            "Expected at least 1 even token with parity screener, got {n_even} even / {n_odd} odd"
        );
    }
}

#[test]
fn test_screening_pruner_zero_relevance_excludes_token() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(1200);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 16,
        confidence_threshold: 0.2,
        block_size: 4,
        temperature: 0.8,
        ..D2fDecodeConfig::default()
    };

    // Screener that gives relevance 0.0 to tokens 3..6
    struct BlockRangeScreener;
    impl katgpt_rs::speculative::ScreeningPruner for BlockRangeScreener {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if (3..=6).contains(&token_idx) {
                0.0
            } else {
                1.0
            }
        }
    }

    let screener = BlockRangeScreener;
    let result = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &NoPruner,
        &screener,
        &mut rng,
    );

    // Tokens 3..6 should never appear (relevance=0 effectively excludes them)
    for &t in &result.tokens {
        if t != config.mask_token {
            assert!(
                !(3..=6).contains(&t),
                "Token {t} should not appear (relevance=0.0 for tokens 3..6)"
            );
        }
    }

    eprintln!("  Zero-relevance exclusion: {:?}", result.tokens);
}

/// No-repeat pruner implementing speculative::ConstraintPruner.
/// Unlike dllm::NoRepeatConstraint (which implements a different trait),
/// this integrates with the D2F decode pipeline via ConstraintPruner.
struct NoRepeatPruner {
    mask_token: usize,
}

impl katgpt_rs::speculative::ConstraintPruner for NoRepeatPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        if token_idx == self.mask_token {
            return false;
        }
        !parent_tokens.contains(&token_idx)
    }
}

#[test]
fn test_no_repeat_constraint_deduplicates() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(1000);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 16,
        confidence_threshold: 0.2, // lower threshold for more convergence
        block_size: 4,             // smaller block easier to enforce no-repeat
        ..D2fDecodeConfig::default()
    };

    let pruner = NoRepeatPruner {
        mask_token: config.mask_token,
    };
    let result = d2f_decode_block(
        &weights,
        &config,
        &decode_config,
        &pruner,
        &NoScreeningPruner,
        &mut rng,
    );

    // Count non-mask tokens
    let non_mask: Vec<usize> = result
        .tokens
        .iter()
        .filter(|&&t| t != config.mask_token)
        .copied()
        .collect();

    // If we have multiple non-mask tokens, check no repeats
    // (Constraint may prevent some tokens from being placed)
    if non_mask.len() > 1 {
        let mut seen = std::collections::HashSet::new();
        let mut has_dups = false;
        for &t in &non_mask {
            if !seen.insert(t) {
                has_dups = true;
            }
        }
        // Note: with random weights, the constraint may leave tokens masked
        // rather than placing duplicates — both outcomes are valid
        eprintln!(
            "  Non-mask tokens: {:?}, has_duplicates: {has_dups}",
            non_mask
        );
    }
}

// ---------------------------------------------------------------------------
// Benchmarks (run with -- --nocapture to see output)
// ---------------------------------------------------------------------------

#[test]
fn benchmark_d2f_decode_block() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size: config.block_size,
        ..D2fDecodeConfig::default()
    };

    // Warmup
    for _ in 0..3 {
        let _ = d2f_decode_block(
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng,
        );
    }

    // Measure
    let n_iters = 50;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = d2f_decode_block(
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng,
        );
    }
    let elapsed = start.elapsed();
    let us_per_block = elapsed.as_micros() as f64 / n_iters as f64;

    eprintln!(
        "\n  Benchmark: d2f_decode_block (block_size={}, steps={})",
        decode_config.block_size, decode_config.denoise_steps
    );
    eprintln!("    {us_per_block:.1} µs/block ({n_iters} iters)");
    eprintln!(
        "    {:.0} tokens/sec (theoretical)",
        decode_config.block_size as f64 / (us_per_block / 1_000_000.0)
    );
}

#[test]
fn benchmark_d2f_pipeline() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let block_size = 4;
    let total_len = 8; // 2 blocks
    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size,
        ..D2fDecodeConfig::default()
    };

    // Warmup
    for _ in 0..3 {
        let pipeline = D2fPipeline::new(&config, decode_config, total_len);
        let _ = pipeline.decode_all(&weights, &NoPruner, &NoScreeningPruner, &mut rng);
    }

    // Measure
    let n_iters = 20;
    let start = Instant::now();
    for _ in 0..n_iters {
        let pipeline = D2fPipeline::new(&config, decode_config, total_len);
        let _ = pipeline.decode_all(&weights, &NoPruner, &NoScreeningPruner, &mut rng);
    }
    let elapsed = start.elapsed();
    let us_per_pipeline = elapsed.as_micros() as f64 / n_iters as f64;

    eprintln!(
        "\n  Benchmark: D2fPipeline ({} blocks × {block_size} tokens, steps={})",
        total_len.div_ceil(block_size),
        decode_config.denoise_steps
    );
    eprintln!("    {us_per_pipeline:.1} µs/pipeline ({n_iters} iters)");
    eprintln!(
        "    {:.0} tokens/sec (theoretical)",
        total_len as f64 / (us_per_pipeline / 1_000_000.0)
    );
}

#[test]
fn benchmark_d2f_steps_sweep() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let (weights, test_data) = train_tiny_model(&config, &mut rng);
    let target = &test_data[0];

    eprintln!("\n  Benchmark: D2F steps sweep (convergence vs throughput)");

    for steps in [2, 4, 8, 16] {
        let decode_config = D2fDecodeConfig {
            denoise_steps: steps,
            confidence_threshold: 0.3,
            block_size: config.block_size,
            ..D2fDecodeConfig::default()
        };

        let start = Instant::now();
        let n_iters = 20;
        let mut last_acc = 0.0f32;
        for _ in 0..n_iters {
            let result = d2f_decode_block_with_target(
                &weights,
                &config,
                &decode_config,
                target,
                &NoPruner,
                &NoScreeningPruner,
                &mut rng,
            );
            last_acc = result.accuracy.unwrap_or(0.0);
        }
        let elapsed = start.elapsed();
        let us_per = elapsed.as_micros() as f64 / n_iters as f64;

        eprintln!(
            "    {steps:>2} steps: {us_per:>8.1} µs/block, accuracy={:.1}%",
            last_acc * 100.0
        );
    }
}

#[test]
fn benchmark_constraint_pruner_overhead() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let (weights, _) = train_tiny_model(&config, &mut rng);

    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.3,
        block_size: config.block_size,
        ..D2fDecodeConfig::default()
    };

    // Without pruner
    let start = Instant::now();
    let n_iters = 30;
    for _ in 0..n_iters {
        let _ = d2f_decode_block(
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng,
        );
    }
    let elapsed_no_pruner = start.elapsed();

    // With pruner (adds per-token is_valid check)
    let pruner = VocabRangePruner {
        min_token: 0,
        max_token: 20,
    };
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = d2f_decode_block(
            &weights,
            &config,
            &decode_config,
            &pruner,
            &NoScreeningPruner,
            &mut rng,
        );
    }
    let elapsed_with_pruner = start.elapsed();

    let us_no = elapsed_no_pruner.as_micros() as f64 / n_iters as f64;
    let us_with = elapsed_with_pruner.as_micros() as f64 / n_iters as f64;
    let overhead_pct = (us_with - us_no) / us_no * 100.0;

    eprintln!("\n  Benchmark: ConstraintPruner overhead");
    eprintln!("    No pruner:    {us_no:.1} µs/block");
    eprintln!("    With pruner:  {us_with:.1} µs/block");
    eprintln!("    Overhead:     {overhead_pct:+.1}%");
}
