//! GOAT Proof — D2F Drafter Verifier (Plan 089, Tri-Mode)
//!
//! Proofs:
//! 1. D2F drafter produces ≥1 token per step
//! 2. D2F drafter terminates, valid sequence
//! 3. Mode switching works (AR → SelfSpeculation → D2F)
//! 4. Acceptance rate measurement (benchmark-style)
//!
//! Run with:
//!   cargo test --features tri_mode --test test_d2f_verifier -- --nocapture
//!   cargo test --features tri_mode --test test_d2f_verifier -- benchmark --nocapture

#![cfg(feature = "tri_mode")]

use katgpt_rs::speculative::d2f::D2fDecodeConfig;
use katgpt_rs::speculative::d2f_verifier::D2fDrafterVerifier;
use katgpt_rs::speculative::types::DecodeStrategy;
use katgpt_rs::speculative::verifier::SpeculativeVerifier;
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Proof 1: D2F drafter acceptance rate ≥ 1 token
// ---------------------------------------------------------------------------

#[test]
fn proof_1_d2f_drafter_produces_at_least_one_token() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

    let d2f_config = D2fDecodeConfig {
        block_size: 4,
        ..D2fDecodeConfig::speed()
    };
    let draft_width = 4;
    let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, draft_width);

    // Run multiple speculation steps — each must return ≥1 token
    let n_steps = 10;
    for step in 0..n_steps {
        let accepted = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            step,
            &mut Rng::new(step as u64 * 7 + 13),
        );
        assert!(
            !accepted.is_empty(),
            "Step {step}: speculate must return at least 1 token, got 0"
        );
        assert!(
            accepted.len() <= draft_width + 1,
            "Step {step}: accepted {} tokens but max is {}",
            accepted.len(),
            draft_width + 1,
        );
        eprintln!("  Step {step}: accepted {} tokens", accepted.len());
    }
}

// ---------------------------------------------------------------------------
// Proof 2: D2F drafter produces valid output (terminates, tokens in range)
// ---------------------------------------------------------------------------

#[test]
fn proof_2_d2f_drafter_produces_valid_sequence() {
    let config = Config::micro_dllm();
    let vocab_size = config.vocab_size;
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

    let d2f_config = D2fDecodeConfig {
        block_size: 4,
        ..D2fDecodeConfig::speed()
    };
    let draft_width = 4;
    let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, draft_width);

    // Run speculation loop for ~20 steps — verify termination and valid tokens
    let n_steps = 20;
    let mut all_tokens: Vec<usize> = Vec::new();
    let mut total_accepted = 0usize;

    for step in 0..n_steps {
        let accepted = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0, // pos=0: verifier resets cache each call, pos must stay < block_size
            &mut Rng::new(step as u64 * 3 + 7),
        );

        // Verify all tokens are in valid range [0, vocab_size)
        for (i, &tok) in accepted.iter().enumerate() {
            assert!(
                tok < vocab_size,
                "Step {step}, token {i}: token {tok} out of range [0, {vocab_size})"
            );
            all_tokens.push(tok);
        }

        total_accepted += accepted.len();
    }

    assert!(
        !all_tokens.is_empty(),
        "Must produce at least some tokens across {n_steps} steps"
    );

    eprintln!(
        "  Proof 2: {n_steps} steps, {total_accepted} total tokens, all in [0, {vocab_size})"
    );
}

// ---------------------------------------------------------------------------
// Proof 3: Mode switching — DecodeStrategy::recommend() returns correct values
// ---------------------------------------------------------------------------

#[test]
fn proof_3_mode_switching_recommend() {
    // (block_size, n_tokens, has_draft_model) → expected strategy
    //
    // Decode strategy priority (order matters, dmax_spd is default-on):
    //   1. has_draft_model && n_tokens >= block_size → SelfSpeculation (tri_mode)
    //   2. n_tokens >= block_size                     → DiscreteDiffusionSoft (dmax_spd)
    //   3. n_tokens >= block_size                     → DiscreteDiffusion (dllm, fallback)
    //   4. has_draft_model                            → Speculative
    //   5. else                                       → Autoregressive

    let cases: Vec<(usize, usize, bool, DecodeStrategy)> = vec![
        // Case 1: No draft model, enough tokens → DiscreteDiffusionSoft (dmax_spd default-on)
        (4, 8, false, DecodeStrategy::DiscreteDiffusionSoft),
        // Case 2: Has draft model, enough tokens → SelfSpeculation (tri-mode wins)
        (4, 8, true, DecodeStrategy::SelfSpeculation),
        // Case 3: Has draft model, NOT enough tokens → Speculative
        (16, 4, true, DecodeStrategy::Speculative),
        // Case 4: No draft model, NOT enough tokens → Autoregressive
        (16, 4, false, DecodeStrategy::Autoregressive),
    ];

    for (block_size, n_tokens, has_draft, expected) in &cases {
        let recommended = DecodeStrategy::recommend(*block_size, *n_tokens, *has_draft);
        assert_eq!(
            recommended, *expected,
            "recommend({block_size}, {n_tokens}, {has_draft}) = {recommended:?}, expected {expected:?}"
        );
        eprintln!("  recommend({block_size}, {n_tokens}, {has_draft}) → {recommended:?} ✓");
    }
}

// ---------------------------------------------------------------------------
// Proof 4: Acceptance rate measurement (benchmark-style, untrained model)
// ---------------------------------------------------------------------------

#[test]
fn proof_4_acceptance_rate_untrained() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

    let draft_width = 4;
    let d2f_config = D2fDecodeConfig {
        block_size: draft_width,
        ..D2fDecodeConfig::speed()
    };

    let n_steps = 30;
    let mut total_accepted = 0usize;
    let mut accepted_counts: Vec<usize> = Vec::with_capacity(n_steps);

    // Warmup (3 iterations)
    {
        let mut verifier =
            D2fDrafterVerifier::new(&target_weights, &config, d2f_config, draft_width);
        for i in 0..3 {
            let _ = verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0, // pos=0: must stay < block_size
                &mut Rng::new(i as u64),
            );
        }
    }

    // Measure
    let start = Instant::now();
    {
        let mut verifier =
            D2fDrafterVerifier::new(&target_weights, &config, d2f_config, draft_width);
        for step in 0..n_steps {
            let accepted = verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0, // pos=0: must stay < block_size
                &mut Rng::new(step as u64 * 11 + 31),
            );
            total_accepted += accepted.len();
            accepted_counts.push(accepted.len());
        }
    }
    let elapsed = start.elapsed();

    let avg_accepted = total_accepted as f64 / n_steps as f64;
    let us_per_step = elapsed.as_micros() as f64 / n_steps as f64;

    eprintln!("\n  Proof 4: D2F Drafter Verifier Acceptance Rate (untrained model)");
    eprintln!("    Draft width: {draft_width}");
    eprintln!("    Steps: {n_steps}");
    eprintln!("    Total tokens: {total_accepted}");
    eprintln!("    Avg tokens/step: {avg_accepted:.2} / {draft_width}+1 max");
    eprintln!("    Time: {us_per_step:.1} µs/step");
    eprintln!(
        "    Accepted counts: {:?}",
        &accepted_counts[..accepted_counts.len().min(10)]
    );

    // With untrained model, acceptance is low but we always get ≥1 token
    assert!(
        avg_accepted >= 1.0,
        "Avg acceptance must be ≥1.0, got {avg_accepted:.2}"
    );

    // Theoretical throughput
    let tokens_per_sec = avg_accepted / (us_per_step / 1_000_000.0);
    eprintln!("    Theoretical throughput: {tokens_per_sec:.0} tokens/sec");
}

// ---------------------------------------------------------------------------
// Extra: Determinism check
// ---------------------------------------------------------------------------

#[test]
fn test_d2f_verifier_deterministic() {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

    let d2f_config = D2fDecodeConfig {
        block_size: 4,
        ..D2fDecodeConfig::speed()
    };

    let r1 = {
        let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);
        verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0,
            &mut Rng::new(100),
        )
    };

    let r2 = {
        let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);
        verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0,
            &mut Rng::new(100),
        )
    };

    assert_eq!(r1, r2, "same seed must produce identical output");
    eprintln!("  Determinism: {r1:?} == {r2:?} ✓");
}
