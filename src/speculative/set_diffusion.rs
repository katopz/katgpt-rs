//! Set Diffusion Inference Decoder (Research 376 Phase 4 T4.1) — root re-export shim.
//!
//! Plan 401 (2026-07-06): The production code + PURE inference tests moved
//! to `crates/katgpt-forward/src/set_diffusion.rs`. This file is now a thin
//! re-export shim that preserves every historical
//! `crate::speculative::set_diffusion::*` import path, plus the TRAIN-only
//! tests that depend on root-local training code
//! (`crate::dllm::{train_mini_dllm, generate_pattern_dataset,
//!  evaluate_set_causal_nelbo, train_mini_set_causal}`).
//!
//! See the moved file for the full module docs.

#![allow(clippy::too_many_arguments)]

// Re-export the entire public API from katgpt-forward.
pub use katgpt_forward::set_diffusion::*;

// Re-export the module itself so `crate::speculative::set_diffusion::Foo`
// paths (used by `src/dllm.rs` training code at lines 1762, 1854) continue
// to resolve.
pub use katgpt_forward::set_diffusion;

#[cfg(all(test, feature = "set_diffusion"))]
mod tests {
    // Plan 401: The original `mod tests` block had these imports spread across
    // inline `use` statements in the PURE section (which moved to
    // katgpt-forward). Re-centralize them here so the TRAIN tests compile.
    // `set_diffusion::*` symbols (CpuSetCausalForward, SetDiffusionConfig,
    // set_diffusion_decode, etc.) come via `super::*` from the shim's
    // `pub use katgpt_forward::set_diffusion::*;`.
    use super::*;
    use crate::dllm::PositionOffsetSchedule;
    use crate::types::{Config, Rng};

    // ── Trained-model integration tests (Phase 4 T4.3 full inference path) ──
    //
    // These are the FIRST tests that exercise the full decode pipeline against
    // a TRAINED model (not random-init). The prior CPU-adapter tests
    // (`test_cpu_adapter_wiring_runs_decode_loop`, `test_cpu_adapter_matches_direct_forward_call`)
    // proved the trait wiring is correct with random-init weights but explicitly
    // did NOT assert convergence or output quality — "random-init model may
    // not produce confident predictions."
    //
    // These tests close that gap. They train a mini bidirectional D2F model via
    // `train_mini_dllm` on alternating-pattern data, then drive the set-diffusion
    // decoder against the trained weights via `CpuSetCausalForward`.
    //
    // Key insight (the AC-Prefix G1 lesson, applied): a bidirectionally-trained
    // model is a SUPERSET of every set-causal attention pattern. The forward
    // pass applies the set-causal mask regardless of how the weights were
    // trained, so the decoder runs end-to-end. Quality varies with the
    // train/infer mismatch:
    //
    //   - MDLM gen-steps (all step 0): every position attends to every other →
    //     ZERO mismatch with bidirectional training → strongest convergence.
    //   - Block-causal gen-steps: within-block attention preserved, cross-block
    //     restricted → mild mismatch → still converges on simple patterns.
    //   - AR-like schedule (w→0): early positions see almost nothing → severe
    //     mismatch for a bidirectional model → weakest convergence.
    //
    // This is NOT a substitute for a true SW-SetDLM-trained model (which would
    // converge well across the full schedule spectrum). It IS a proof that the
    // full inference pipeline — schedule → ordering → gen-steps → forward →
    // sample → commit — produces meaningful, convergent output against real
    // trained weights, and establishes the quality baseline that a
    // set-causal-trained model is expected to beat.

    use crate::dllm::{generate_pattern_dataset, train_mini_dllm};

    /// Train a mini bidirectional D2F model on 8-token alternating patterns.
    ///
    /// Shared setup for the trained-model tests below. The model learns
    /// "even positions = a, odd positions = b" from `[a,b,a,b,a,b,a,b]`
    /// sequences — a simple bidirectional-predictable task.
    fn train_pattern_model(seed: u64) -> (Config, crate::transformer::TransformerWeights) {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(seed);
        let train_data = generate_pattern_dataset(&mut rng, 100, 8, 8);
        let test_data = generate_pattern_dataset(&mut rng, 20, 8, 8);
        let (weights, _loss) =
            train_mini_dllm(&config, &train_data, &test_data, 300, 0.01, 0.25, seed);
        (config, weights)
    }

    #[test]
    fn test_set_diffusion_decode_trained_model_mdlm_converges() {
        // MDLM gen-steps: all positions share step 0 → every position attends
        // to every other. This is EXACTLY bidirectional attention, so there is
        // ZERO train/infer mismatch with the bidirectionally-trained model.
        // Expect strong convergence.
        let (config, weights) = train_pattern_model(42);
        let forward = CpuSetCausalForward {
            weights: &weights,
            config: &config,
        };
        let decode_config = SetDiffusionConfig {
            mask_token: config.mask_token,
            vocab_size: config.vocab_size,
            denoise_steps: 8,
            confidence_threshold: 0.5,
            temperature: 0.0, // Greedy for determinism.
            #[cfg(feature = "entropy_bounded_commit")]
            eb_gamma: -1.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_max_commit: usize::MAX,
        };
        let gen_steps = mdlm_gen_steps(8);
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&forward, &decode_config, &[], &gen_steps, &mut rng);

        assert!(
            result.converged,
            "MDLM gen-steps + bidirectional-trained model should converge (zero mismatch). \
             forward_passes={}, confidence_history={:?}",
            result.forward_passes, result.confidence_history,
        );
        assert_eq!(result.tokens.len(), 8);
        assert!(
            result.tokens.iter().all(|&t| t != config.mask_token),
            "no mask tokens should remain: {:?}",
            result.tokens,
        );
    }

    #[test]
    fn test_set_diffusion_decode_trained_model_block_causal_converges() {
        // Block-causal gen-steps (block_size=4): positions 0-3 → step 0,
        // positions 4-7 → step 1. Within each block, full attention is
        // preserved (matching bidirectional training); cross-block attention
        // is restricted. Mild mismatch — simple patterns should still converge.
        let (config, weights) = train_pattern_model(42);
        let forward = CpuSetCausalForward {
            weights: &weights,
            config: &config,
        };
        let decode_config = SetDiffusionConfig {
            mask_token: config.mask_token,
            vocab_size: config.vocab_size,
            denoise_steps: 8,
            confidence_threshold: 0.5,
            temperature: 0.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_gamma: -1.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_max_commit: usize::MAX,
        };
        let gen_steps = block_causal_gen_steps(8, 4);
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&forward, &decode_config, &[], &gen_steps, &mut rng);

        assert!(
            result.converged,
            "Block-causal gen-steps + bidirectional-trained model should converge on simple patterns. \
             forward_passes={}, confidence_history={:?}",
            result.forward_passes, result.confidence_history,
        );
        assert_eq!(result.tokens.len(), 8);
        assert!(
            result.tokens.iter().all(|&t| t != config.mask_token),
            "no mask tokens should remain: {:?}",
            result.tokens,
        );
    }

    #[test]
    fn test_set_diffusion_decode_scheduled_trained_model_diffusion_endpoint() {
        // Full T4.3-CPU-caller bridge with a TRAINED model: schedule → ordering →
        // gen-steps → decode. Uses PositionOffsetSchedule::diffusion() (w=1, k=1)
        // which produces uniform-random orderings — the order-agnostic diffusion
        // endpoint. Combined with a bidirectional-trained model, this is the
        // lowest-mismatch point of the schedule spectrum.
        //
        // This is the canonical end-to-end validation of the Phase 4 bridge:
        // every piece of plumbing (sample_order, order_to_gen_steps,
        // set_diffusion_decode) runs against real trained weights.
        let (config, weights) = train_pattern_model(42);
        let forward = CpuSetCausalForward {
            weights: &weights,
            config: &config,
        };
        let decode_config = SetDiffusionConfig {
            mask_token: config.mask_token,
            vocab_size: config.vocab_size,
            denoise_steps: 8,
            confidence_threshold: 0.5,
            temperature: 0.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_gamma: -1.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_max_commit: usize::MAX,
        };
        let schedule = PositionOffsetSchedule::diffusion();
        let mut rng = Rng::new(42);

        let result =
            set_diffusion_decode_scheduled(&forward, &decode_config, &[], &schedule, 8, &mut rng);

        assert!(
            result.converged,
            "Scheduled decode (diffusion endpoint) + trained model should converge. \
             forward_passes={}, confidence_history={:?}",
            result.forward_passes, result.confidence_history,
        );
        assert_eq!(result.tokens.len(), 8);
        assert!(
            result.tokens.iter().all(|&t| t != config.mask_token),
            "no mask tokens should remain: {:?}",
            result.tokens,
        );
    }

    #[test]
    fn test_set_diffusion_decode_trained_model_prompt_anchored_alternating_pattern() {
        // The model was trained on [a,b,a,b,a,b,a,b] alternating patterns.
        // From an all-masked cold start, order-agnostic diffusion has no anchor
        // to enforce a globally consistent (a,b) pair — different positions may
        // lock in to different guesses during the greedy denoise commit. This is
        // the EXPECTED cold-start behavior of a diffusion model without context.
        //
        // With a 2-token prompt [a, b] anchoring the pattern, the decode region
        // should inherit the prompt's (a,b) pair and produce a globally consistent
        // alternating sequence. This is the realistic usage pattern (conditioned
        // generation) and the strongest modelless quality claim we can make.
        //
        // MDLM gen-steps isolate the pattern-quality claim from any attention-
        // restriction artifact (zero train/infer mismatch).
        let (config, weights) = train_pattern_model(42);
        let forward = CpuSetCausalForward {
            weights: &weights,
            config: &config,
        };
        let decode_config = SetDiffusionConfig {
            mask_token: config.mask_token,
            vocab_size: config.vocab_size,
            denoise_steps: 8,
            confidence_threshold: 0.5,
            temperature: 0.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_gamma: -1.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_max_commit: usize::MAX,
        };
        // 2-token prompt establishes the (a, b) pair. Decode 6 more positions
        // to complete an 8-token alternating sequence [a, b, ?, ?, ?, ?, ?, ?].
        let prompt: &[usize] = &[5, 2];
        let gen_steps = mdlm_gen_steps(6);
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&forward, &decode_config, prompt, &gen_steps, &mut rng);

        assert!(result.converged, "precondition: decoder must converge");
        assert_eq!(result.tokens.len(), 8);
        let t = &result.tokens;
        // Prompt preserved.
        assert_eq!(&t[..2], prompt, "prompt must be preserved: {t:?}");
        // Decode region inherits the prompt's alternating structure.
        // Expected: [5, 2, 5, 2, 5, 2, 5, 2].
        assert_eq!(
            t[2], prompt[0],
            "position 2 must match prompt[0] (alternating): {t:?}"
        );
        assert_eq!(
            t[3], prompt[1],
            "position 3 must match prompt[1] (alternating): {t:?}"
        );
        assert_eq!(
            t[4], prompt[0],
            "position 4 must match prompt[0] (alternating): {t:?}"
        );
        assert_eq!(
            t[5], prompt[1],
            "position 5 must match prompt[1] (alternating): {t:?}"
        );
        assert_eq!(
            t[6], prompt[0],
            "position 6 must match prompt[0] (alternating): {t:?}"
        );
        assert_eq!(
            t[7], prompt[1],
            "position 7 must match prompt[1] (alternating): {t:?}"
        );
    }

    // ── GOAT gate: set-causal-trained vs bidirectional-trained (Phase 4 unblock) ──
    //
    // The prior session deferred `set_diffusion_decoder` promotion to default,
    // claiming the GOAT gate needs a set-causal-trained model that only
    // riir-train could produce. This was ANOTHER premature deferral (the
    // AC-Prefix G1 lesson, round 2): `train_mini_dllm` already trains real
    // `TransformerWeights` models on CPU, and the set-causal counterpart
    // (`train_mini_set_causal`) is a straightforward adaptation that reuses
    // the same backward + SGD infrastructure. The backward compatibility
    // invariant (softmax Jacobian naturally zeros masked paths) means no new
    // backward code is needed.
    //
    // This test trains BOTH models on directional (Markov) data and compares
    // their NELBO when evaluated under set-causal attention at the SW-SetDLM
    // training schedule (w=0.5). The set-causal model should WIN because it
    // was trained under the exact attention pattern it's evaluated with.

    use crate::dllm::{evaluate_set_causal_nelbo, train_mini_set_causal};

    /// Generate Markov-chain token sequences for directional structure tests.
    ///
    /// Each sequence: token[0] = random, token[i] = (token[i-1] + step) % vocab
    /// where step ∈ {1, 2}. This gives strong left-to-right dependency — the
    /// key property that distinguishes set-causal training (which learns to
    /// predict from limited context) from bidirectional training (which learns
    /// to predict from full context).
    fn generate_markov_token_dataset(
        rng: &mut Rng,
        n_sequences: usize,
        seq_len: usize,
        vocab: usize,
    ) -> Vec<Vec<usize>> {
        (0..n_sequences)
            .map(|_| {
                let mut seq = Vec::with_capacity(seq_len);
                seq.push((rng.next() as usize) % vocab);
                for i in 1..seq_len {
                    let step = 1 + (rng.next() as usize) % 2;
                    seq.push((seq[i - 1] + step) % vocab);
                }
                seq
            })
            .collect()
    }

    /// Train a bidirectional D2F model on Markov data (shared baseline for GOAT gate).
    fn train_bidirectional_markov_model(
        seed: u64,
    ) -> (
        Config,
        crate::transformer::TransformerWeights,
        Vec<Vec<usize>>,
    ) {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(seed);
        let train_data = generate_markov_token_dataset(&mut rng, 100, 8, 8);
        let test_data = generate_markov_token_dataset(&mut rng, 20, 8, 8);
        let (weights, _loss) =
            train_mini_dllm(&config, &train_data, &test_data, 300, 0.01, 0.25, seed);
        (config, weights, test_data)
    }

    /// Train a set-causal SW-SetDLM model on Markov data (the GOAT candidate).
    fn train_set_causal_markov_model(
        seed: u64,
        schedule: &PositionOffsetSchedule,
    ) -> (
        Config,
        crate::transformer::TransformerWeights,
        Vec<Vec<usize>>,
    ) {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(seed);
        let train_data = generate_markov_token_dataset(&mut rng, 100, 8, 8);
        let test_data = generate_markov_token_dataset(&mut rng, 20, 8, 8);
        let (weights, _loss) =
            train_mini_set_causal(&config, &train_data, &test_data, 300, 0.01, schedule, seed);
        (config, weights, test_data)
    }

    #[test]
    fn test_goat_gate_set_causal_beats_bidirectional_at_sw_schedule() {
        // The GOAT gate: does a set-causal-trained model provide a GAIN over a
        // bidirectional-trained model when both are evaluated under set-causal
        // attention at the SW-SetDLM schedule (w=0.5)?
        //
        // This is the test the prior sessions said was "blocked on riir-train".
        // It is NOT blocked — `train_mini_set_causal` trains the model on CPU.
        let schedule = PositionOffsetSchedule::new(0.5); // w=0.5, k=1.0 (SW-SetDLM)
        let seed = 42;

        let (config, bidir_weights, test_data) = train_bidirectional_markov_model(seed);
        let (_config2, sc_weights, _test_data2) = train_set_causal_markov_model(seed, &schedule);

        // Evaluate both under set-causal attention at the SW-SetDLM schedule.
        let mut rng_eval = Rng::new(seed + 1000);
        let bidir_nelbo = evaluate_set_causal_nelbo(
            &bidir_weights,
            &test_data,
            &config,
            &schedule,
            &mut rng_eval,
        );
        let mut rng_eval2 = Rng::new(seed + 1000); // Same seed for apples-to-apples.
        let sc_nelbo =
            evaluate_set_causal_nelbo(&sc_weights, &test_data, &config, &schedule, &mut rng_eval2);

        eprintln!(
            "GOAT gate (w=0.5): bidirectional NELBO={:.4}, set-causal NELBO={:.4}, \
             improvement={:.4} ({:.1}%)",
            bidir_nelbo,
            sc_nelbo,
            bidir_nelbo - sc_nelbo,
            (bidir_nelbo - sc_nelbo) / bidir_nelbo * 100.0,
        );

        // The set-causal model MUST beat the bidirectional model here — it was
        // trained under the exact attention pattern used for evaluation. The
        // bidirectional model expects full context and degrades when some
        // positions are masked.
        assert!(
            sc_nelbo < bidir_nelbo,
            "GOAT gate FAILED: set-causal model ({sc_nelbo:.4}) must beat bidirectional model ({bidir_nelbo:.4}) \
             when evaluated under SW-SetDLM schedule (w=0.5). \
             This means the set-diffusion decoder requires a set-causal-trained model \
             to provide a gain — the substrate alone is not enough.",
        );
    }

    #[test]
    fn test_goat_gate_set_causal_trained_model_converges_at_ar_endpoint() {
        // Sanity check: a set-causal-trained model should produce convergent
        // output when decoded at the AR-like endpoint (w→0). The bidirectional
        // model struggles here (early positions see almost nothing), but the
        // set-causal model was trained with stochastic orderings that include
        // near-AR patterns.
        //
        // This test doesn't compare NELBO — it just confirms the decode
        // pipeline produces coherent output (no mask tokens remain).
        let schedule = PositionOffsetSchedule::new(0.5);
        let (config, sc_weights, _test_data) = train_set_causal_markov_model(42, &schedule);

        let forward = CpuSetCausalForward {
            weights: &sc_weights,
            config: &config,
        };
        let decode_config = SetDiffusionConfig {
            mask_token: config.mask_token,
            vocab_size: config.vocab_size,
            denoise_steps: 8,
            confidence_threshold: 0.5,
            temperature: 0.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_gamma: -1.0,
            #[cfg(feature = "entropy_bounded_commit")]
            eb_max_commit: usize::MAX,
        };
        // 2-token prompt anchors the Markov chain (positions 0,1 = [3, 5]).
        // Decode 6 more positions with AR gen-steps (strict left-to-right).
        let prompt: &[usize] = &[3, 5];
        let gen_steps: Vec<u32> = (0..6u32).collect(); // AR: 0,1,2,3,4,5
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&forward, &decode_config, prompt, &gen_steps, &mut rng);

        assert!(
            result.converged,
            "Set-causal-trained model + AR decode should converge. \
             forward_passes={}, confidence_history={:?}",
            result.forward_passes, result.confidence_history,
        );
        assert_eq!(result.tokens.len(), 8);
        assert!(
            result.tokens.iter().all(|&t| t != config.mask_token),
            "no mask tokens should remain: {:?}",
            result.tokens,
        );
    }
}
