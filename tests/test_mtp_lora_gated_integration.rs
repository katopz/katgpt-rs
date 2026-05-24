//! Plan 117 T34: MTP LoRA Drafter + Output-Length Gate + Top-K Integration Tests
//!
//! Verifies LoRA drafter + output-length gate + Top-K cluster selection
//! all compose correctly when used together.
//!
//! Run: `cargo test --test test_mtp_lora_gated_integration -- --nocapture`

#![cfg(feature = "dllm")]

use katgpt_rs::speculative::{DrafterLoraWeights, LeviathanVerifier, SpeculativeVerifier};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};

// ── T34: Output-length gating + LoRA drafter ──────────────────
//
// When remaining_capacity < mtp_min_output_tokens, the verifier
// should return exactly 1 token even with LoRA drafter attached.

#[test]
fn test_lora_output_length_gate_skips_short_sequences() {
    let draft_config = Config::draft();
    let mut target_config = Config::micro_dllm();
    target_config.mtp_min_output_tokens = 100; // Very high threshold → gating active

    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    // Attach LoRA drafter — gating should still take effect
    let lora = DrafterLoraWeights::zeros(
        &draft_config,
        draft_config.lora_rank,
        draft_config.lora_alpha,
    );
    let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config)
        .with_drafter_lora(lora, &draft_config);

    // At pos=0, remaining_capacity = block_size(16) - 0 = 16 < 100 → gated
    let accepted = verifier.speculate(
        &draft_weights,
        &draft_config,
        target_config.bos_token,
        0,
        &mut Rng::new(100),
    );

    // Gating should return exactly 1 token (no MTP), even with LoRA drafter
    assert_eq!(
        accepted.len(),
        1,
        "gating should return exactly 1 token when output too short, got {}",
        accepted.len()
    );
    assert!(
        accepted[0] < target_config.vocab_size,
        "token should be valid vocab index"
    );
}

// ── T34: Top-K cluster selection + LoRA drafter ───────────────
//
// With different mtp_cluster_topk values, LoRA drafter should
// still produce valid tokens (in vocab range).

#[test]
fn test_lora_topk_produces_valid_tokens() {
    for topk in [1usize, 2, 4] {
        let draft_config = Config::draft();
        let mut target_config = Config::micro_dllm();
        target_config.mtp_min_output_tokens = 1; // Low threshold → MTP active
        target_config.mtp_cluster_topk = topk;

        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

        let lora = DrafterLoraWeights::zeros(
            &draft_config,
            draft_config.lora_rank,
            draft_config.lora_alpha,
        );
        let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config)
            .with_drafter_lora(lora, &draft_config);

        let accepted = verifier.speculate(
            &draft_weights,
            &draft_config,
            target_config.bos_token,
            0,
            &mut Rng::new(100),
        );

        assert!(!accepted.is_empty(), "topk={topk}: must produce tokens");
        for &t in &accepted {
            assert!(
                t < target_config.vocab_size,
                "topk={topk}: token {t} out of range"
            );
        }
    }
}

// ── T34: Composition — all features active simultaneously ──────
//
// Output-length gating + Top-K cluster selection + LoRA drafter
// all active together should still produce valid output.

#[test]
fn test_lora_composition_all_features() {
    let draft_config = Config::draft();
    let mut target_config = Config::micro_dllm();
    target_config.mtp_min_output_tokens = 1; // Enable MTP
    target_config.mtp_cluster_topk = 2; // Top-2 clusters

    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    let lora = DrafterLoraWeights::zeros(
        &draft_config,
        draft_config.lora_rank,
        draft_config.lora_alpha,
    );
    let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config)
        .with_drafter_lora(lora, &draft_config);

    // Run multiple positions to exercise composition
    for pos in 0..4 {
        let accepted = verifier.speculate(
            &draft_weights,
            &draft_config,
            target_config.bos_token,
            pos,
            &mut Rng::new(100 + pos as u64),
        );
        assert!(!accepted.is_empty(), "pos={pos}: must produce tokens");
        for &t in &accepted {
            assert!(
                t < target_config.vocab_size,
                "pos={pos}: token {t} out of range"
            );
        }
    }
}

// ── T34: LoRA drafter without gating (MTP active) ─────────────
//
// When mtp_min_output_tokens is low, LoRA drafter path should
// produce valid tokens and may return more than 1 token.

#[test]
fn test_lora_mtp_active_produces_valid_tokens() {
    let draft_config = Config::draft();
    let mut target_config = Config::micro_dllm();
    target_config.mtp_min_output_tokens = 1; // Low threshold → MTP active

    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    let lora = DrafterLoraWeights::zeros(
        &draft_config,
        draft_config.lora_rank,
        draft_config.lora_alpha,
    );
    let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config)
        .with_drafter_lora(lora, &draft_config);

    // Run many iterations to verify stability
    let mut saw_multi = false;
    for seed in 0..50u64 {
        let accepted = verifier.speculate(
            &draft_weights,
            &draft_config,
            target_config.bos_token,
            0,
            &mut Rng::new(seed),
        );
        assert!(
            !accepted.is_empty(),
            "seed={seed}: should always return at least 1 token"
        );
        for &t in &accepted {
            assert!(
                t < target_config.vocab_size,
                "seed={seed}: token {t} out of vocab range"
            );
        }
        if accepted.len() > 1 {
            saw_multi = true;
        }
    }
    assert!(
        saw_multi,
        "with MTP enabled (low threshold) + LoRA, should see multi-token results at least once"
    );
}
