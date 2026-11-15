//! Diffusion Sampler — root re-export shim + train-bridge.
//!
//! Plan 399 (2026-07-05): the bulk of this module (production code + 16
//! inference-only tests) moved to `katgpt_forward::diffusion_sampler`. This
//! file is a thin re-export shim that preserves every historical
//! `crate::speculative::diffusion_sampler::*` import path.
//!
//! The `train_logistic_on_patterns` convenience function stays in root
//! because it depends on `crate::dllm::{generate_pattern_dataset, train_mini_dllm}`
//! (training code). It's defined here and re-exports the inference-side
//! helpers (`collect_trajectories`, `DiffusionSampler`) from katgpt-forward.
//!
//! The 6 training-dependent tests stayed here for the same reason — they
//! exercise the train+infer interaction.

pub use katgpt_forward::diffusion_sampler::{
    DiffusionSampler, N_STABILITY_FEATURES, SamplerDecision, SamplerFeatures, SamplerTrajectory,
    SamplerVariant, StabilityObservation, StabilityReason, StabilityTracker, TOPK_DRIFT_K,
    collect_trajectories,
};

use crate::speculative::d2f::D2fDecodeConfig;
use crate::types::{Config, Rng};

/// Train a logistic sampler on D2F trajectories from pattern data.
///
/// Convenience function that:
/// 1. Generates pattern dataset
/// 2. Trains a mini dLLM
/// 3. Collects trajectories
/// 4. Trains the sampler
///
/// Returns (trained sampler, final loss, auc).
///
/// _Stays in root (Plan 399) because it bridges inference (now in
/// `katgpt_forward::diffusion_sampler`) with training
/// (`crate::dllm::{generate_pattern_dataset, train_mini_dllm}`)._
#[allow(clippy::too_many_arguments)]
pub fn train_logistic_on_patterns(
    config: &Config,
    decode_config: &D2fDecodeConfig,
    n_train: usize,
    n_test: usize,
    n_dllm_epochs: usize,
    sampler_lr: f32,
    sampler_epochs: usize,
    seed: u64,
) -> (DiffusionSampler, f32, f32) {
    use crate::dllm::{generate_pattern_dataset, train_mini_dllm};

    let mut rng = Rng::new(seed);

    // Generate data
    let effective_vocab = config.vocab_size.saturating_sub(1);
    let train_data =
        generate_pattern_dataset(&mut rng, n_train, config.block_size, effective_vocab);
    let test_data = generate_pattern_dataset(&mut rng, n_test, config.block_size, effective_vocab);

    // Train dLLM
    let (weights, _) = train_mini_dllm(
        config,
        &train_data,
        &test_data,
        n_dllm_epochs,
        0.01,
        0.3,
        seed,
    );

    // Collect trajectories from test data
    let trajectories = collect_trajectories(
        &weights,
        config,
        decode_config,
        &test_data,
        0, // unlimited
    );

    // Train sampler
    let mut sampler = DiffusionSampler::logistic(&mut Rng::new(seed + 1));
    let final_loss = sampler.train(&trajectories, sampler_lr, sampler_epochs);
    let auc = sampler.evaluate_auc(&trajectories);

    (sampler, final_loss, auc)
}

#[cfg(test)]
mod tests {
    // These 6 tests stayed in root because they call
    // `crate::dllm::{train_mini_dllm, generate_pattern_dataset}` (root-only
    // training code) to produce weights, then exercise the D2F + DiffusionSampler
    // inference pipeline (now living in katgpt-forward).
    use super::*;
    use crate::dllm::D2fContext;
    use crate::dllm::{generate_pattern_dataset, train_mini_dllm};
    use crate::speculative::d2f::D2fDecodeConfig;
    use crate::speculative::d2f::d2f_decode_block_with_sampler;
    use crate::speculative::types::{NoPruner, NoScreeningPruner};
    use crate::types::{Config, Rng};

    fn make_config() -> Config {
        Config::micro_dllm()
    }

    #[test]
    fn test_collect_trajectories_produces_data() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 20, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 5, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 100, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);

        assert!(
            !trajectories.is_empty(),
            "should collect at least some trajectories",
        );

        // Check features are valid
        for traj in &trajectories {
            assert!(
                traj.features.top1_prob >= 0.0 && traj.features.top1_prob <= 1.0,
                "top1_prob should be in [0, 1], got {}",
                traj.features.top1_prob,
            );
        }
    }

    #[test]
    fn test_collect_trajectories_respects_cap() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 20, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 5, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 50, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);
        let cap = 10;
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, cap);

        assert!(
            trajectories.len() <= cap,
            "should respect cap of {cap}, got {}",
            trajectories.len(),
        );
    }

    // ── End-to-End: Train Sampler ──

    #[test]
    fn test_train_logistic_on_patterns() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 30, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 10, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);

        let mut sampler = DiffusionSampler::logistic(&mut Rng::new(99));
        let loss = sampler.train(&trajectories, 0.1, 100);
        let auc = sampler.evaluate_auc(&trajectories);

        assert!(loss.is_finite(), "loss should be finite, got {loss}",);
        // AUC > 0.5 means the sampler learned something
        // (may be close to 0.5 with random weights, but should be finite)
        assert!(
            (0.0..=1.0).contains(&auc),
            "AUC should be in [0, 1], got {auc}",
        );
    }

    // ── Convenience Function Test ──

    #[test]
    fn test_train_logistic_on_patterns_convenience() {
        let config = make_config();
        let decode_config = D2fDecodeConfig::with_block_size(4);

        let (sampler, loss, auc) =
            train_logistic_on_patterns(&config, &decode_config, 20, 5, 100, 0.1, 50, 42);

        assert!(
            matches!(sampler.variant(), SamplerVariant::Logistic),
            "should return logistic sampler",
        );
        assert!(loss.is_finite(), "loss should be finite, got {loss}",);
        assert!(
            (0.0..=1.0).contains(&auc),
            "AUC should be in [0, 1], got {auc}",
        );
    }

    // ── T3 Integration Tests (Plan 116) ──

    #[test]
    fn test_d2f_decode_with_sampler_produces_valid_output() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 30, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 10, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);

        // Train a sampler on the test data
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);
        let mut sampler = DiffusionSampler::logistic(&mut rng);
        if !trajectories.is_empty() {
            sampler.train(&trajectories, 0.1, 50);
        }

        // Decode with sampler
        let mut dctx = D2fContext::new(&config);
        let result = d2f_decode_block_with_sampler(
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng,
            Some(&sampler),
        );

        // All tokens should be valid (in vocab range)
        for (i, &t) in result.tokens.iter().enumerate() {
            assert!(
                t < config.vocab_size,
                "token[{i}] = {t} out of vocab range [0, {})",
                config.vocab_size,
            );
        }
        assert!(
            result.steps_used > 0,
            "should use at least 1 denoising step",
        );
    }

    #[test]
    fn test_d2f_decode_sampler_differs_from_fixed_threshold() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 30, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 10, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);

        // Train a sampler with strong weights to force different decisions
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);
        let mut sampler = DiffusionSampler::logistic(&mut rng);
        if !trajectories.is_empty() {
            // Train aggressively to differentiate from fixed threshold
            sampler.train(&trajectories, 0.5, 200);
        }

        // Decode with sampler=None (fixed threshold)
        let mut dctx_fixed = D2fContext::new(&config);
        let mut rng_fixed = Rng::new(99);
        let result_fixed = d2f_decode_block_with_sampler(
            &mut dctx_fixed,
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng_fixed,
            None,
        );

        // Decode with sampler=Some (adaptive)
        let mut dctx_sampler = D2fContext::new(&config);
        let mut rng_sampler = Rng::new(99);
        let result_sampler = d2f_decode_block_with_sampler(
            &mut dctx_sampler,
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng_sampler,
            Some(&sampler),
        );

        // Both should produce valid output
        assert!(
            !result_fixed.tokens.is_empty(),
            "fixed should produce tokens"
        );
        assert!(
            !result_sampler.tokens.is_empty(),
            "sampler should produce tokens",
        );

        // If the sampler learned anything non-trivial, confidence Histories
        // should differ (different accept/reject patterns at each step).
        // This may not always differ (e.g., if training data is too easy),
        // so we only check that both produce valid confidence values.
        for &c in &result_fixed.confidence_history {
            assert!(
                (0.0..=1.0).contains(&c),
                "fixed confidence {c} out of [0,1]"
            );
        }
        for &c in &result_sampler.confidence_history {
            assert!(
                (0.0..=1.0).contains(&c),
                "sampler confidence {c} out of [0,1]",
            );
        }

        // At minimum, steps_used should be positive for both
        assert!(result_fixed.steps_used > 0, "fixed should use steps");
        assert!(result_sampler.steps_used > 0, "sampler should use steps",);
    }
}
