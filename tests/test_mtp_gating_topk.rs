//! Plan 117: Output-Length Gating + Top-K Cluster Selection Tests
//!
//! T18-T19: Output-length gating prevents MTP overhead on short texts.
//! T29-T32: Top-K cluster selection for clustered LM head.
//!
//! Run: `cargo test --test test_mtp_gating_topk -- --nocapture`

use katgpt_rs::speculative::{LeviathanVerifier, SpeculativeVerifier};
use katgpt_rs::transformer::{
    TransformerWeights, cluster_map_round_robin, forward, select_topk_indices,
};
use katgpt_rs::types::{Config, Rng};

// ── T18: Output-length gating disables MTP on short output ─────
//
// When remaining_capacity < mtp_min_output_tokens, the verifier
// should return exactly 1 token (no speculative drafting).

#[test]
fn test_mtp_min_output_tokens_disables_short() {
    let mut config = Config::micro();
    config.mtp_min_output_tokens = 100; // Very high threshold

    let draft_config = Config::draft();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    let mut verifier = LeviathanVerifier::new(&target_weights, &config, &draft_config);

    // At pos=0, remaining_capacity = block_size(16) - 0 = 16 < 100 → gated
    let accepted = verifier.speculate(
        &draft_weights,
        &draft_config,
        config.bos_token,
        0,
        &mut Rng::new(100),
    );

    // Gating should return exactly 1 token (no MTP)
    assert_eq!(
        accepted.len(),
        1,
        "gating should return exactly 1 token when output too short, got {}",
        accepted.len()
    );
    assert!(
        accepted[0] < config.vocab_size,
        "token should be valid vocab index"
    );
}

// ── T19: Output-length gating enables MTP on long output ───────
//
// When remaining_capacity >= mtp_min_output_tokens, MTP speculative
// decoding is active. The verifier may return 1 or more tokens.

#[test]
fn test_mtp_min_output_tokens_enables_long() {
    let mut config = Config::micro();
    config.mtp_min_output_tokens = 1; // Low threshold → MTP active

    let draft_config = Config::draft();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

    // Run many iterations and verify we sometimes get > 1 token (MTP active)
    let mut saw_multi = false;
    for seed in 0..100u64 {
        let mut verifier = LeviathanVerifier::new(&target_weights, &config, &draft_config);
        let accepted = verifier.speculate(
            &draft_weights,
            &draft_config,
            config.bos_token,
            0,
            &mut Rng::new(seed),
        );
        assert!(
            !accepted.is_empty(),
            "should always return at least 1 token"
        );
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of vocab range");
        }
        if accepted.len() > 1 {
            saw_multi = true;
        }
    }
    assert!(
        saw_multi,
        "with MTP enabled (low threshold), should see multi-token results at least once"
    );
}

// ── T32: select_topk_indices correctness ────────────────────────
//
// Verify that select_topk_indices returns the correct top-K indices
// sorted by score descending.

#[test]
fn test_select_topk_indices_correctness() {
    // Simple case: 5 elements, K=3
    let scores = [0.5, 0.9, 0.1, 0.7, 0.3];
    let topk = select_topk_indices(&scores, 3);

    assert_eq!(topk.len(), 3, "should return exactly K indices");
    // Top 3 by score: idx 1 (0.9), idx 3 (0.7), idx 0 (0.5)
    assert_eq!(topk[0], 1, "highest score at index 1");
    assert_eq!(topk[1], 3, "second highest at index 3");
    assert_eq!(topk[2], 0, "third highest at index 0");
}

#[test]
fn test_select_topk_indices_k_exceeds_len() {
    let scores = [1.0, 2.0, 3.0];
    let topk = select_topk_indices(&scores, 10);

    // K clamped to scores.len()
    assert_eq!(topk.len(), 3, "K should be clamped to scores length");
    assert_eq!(topk[0], 2, "highest at index 2");
    assert_eq!(topk[1], 1, "second at index 1");
    assert_eq!(topk[2], 0, "third at index 0");
}

#[test]
fn test_select_topk_indices_k_equals_1() {
    let scores = [0.1, 0.5, 0.3, 0.8, 0.2];
    let topk = select_topk_indices(&scores, 1);

    assert_eq!(topk.len(), 1, "K=1 should return 1 index");
    assert_eq!(topk[0], 3, "argmax at index 3 (score 0.8)");
}

#[test]
fn test_select_topk_indices_empty() {
    let scores: [f32; 0] = [];
    let topk = select_topk_indices(&scores, 3);
    assert!(topk.is_empty(), "empty scores should return empty result");
}

#[test]
fn test_select_topk_indices_k_zero() {
    let scores = [1.0, 2.0, 3.0];
    let topk = select_topk_indices(&scores, 0);
    assert!(topk.is_empty(), "K=0 should return empty result");
}

// ── T29: clustered_lm_head Top-K equals Top-1 when K=1 ──────────
//
// When K=1, top-K cluster selection should produce identical output
// to the old single-cluster argmax behavior.

#[test]
fn test_clustered_lm_head_topk_equals_top1_when_k1() {
    let mut config = Config::bpe();
    config.mtp_cluster_topk = 1; // Force top-1
    config.mtp_cluster_vocab_threshold = 1; // Always activate clustered path

    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);

    let cluster_map = cluster_map_round_robin(config.vocab_size, config.mtp_cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * config.n_embd)
        .map(|_| rng.normal())
        .collect();
    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map);

    let mut ctx = katgpt_rs::transformer::ForwardContext::new(&config);
    let mut cache = katgpt_rs::transformer::MultiLayerKVCache::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

    // With K=1, exactly one cluster should have finite logits
    let cluster_size = config.mtp_cluster_size;
    let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
    let inf_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();

    assert!(
        finite_count <= cluster_size,
        "K=1 should have at most cluster_size ({cluster_size}) finite logits, got {finite_count}"
    );
    assert!(
        inf_count > 0,
        "K=1 should have some -inf logits (non-cluster tokens)"
    );
    assert_eq!(
        finite_count + inf_count,
        config.vocab_size,
        "finite + inf should equal vocab_size"
    );
}

// ── T30: clustered_lm_head Top-K covers more tokens ─────────────
//
// With K=4, we should get at least K * cluster_size finite logits
// (assuming distinct clusters are selected).

#[test]
fn test_clustered_lm_head_topk_covers_more_tokens() {
    let mut config = Config::bpe();
    config.mtp_cluster_topk = 4; // Select 4 clusters
    config.mtp_cluster_vocab_threshold = 1; // Always activate clustered path

    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);

    let cluster_map = cluster_map_round_robin(config.vocab_size, config.mtp_cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * config.n_embd)
        .map(|_| rng.normal())
        .collect();
    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map);

    let mut ctx = katgpt_rs::transformer::ForwardContext::new(&config);
    let mut cache = katgpt_rs::transformer::MultiLayerKVCache::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

    let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
    let inf_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();

    // K=4 should cover at least 4 * cluster_size tokens
    let min_expected = config.mtp_cluster_topk * config.mtp_cluster_size.min(config.vocab_size);
    assert!(
        finite_count >= min_expected.min(config.vocab_size),
        "K=4 should cover at least {min_expected} tokens, got {finite_count}"
    );
    assert!(
        inf_count > 0,
        "with K=4 < num_clusters, should still have some -inf logits"
    );
}

// ── T31: Top-K covers all clusters when K >= num_clusters ───────
//
// When K >= number of clusters, all clusters are selected,
// meaning all tokens get finite logits (no -inf).

#[test]
fn test_clustered_lm_head_topk_all_clusters_when_k_ge_num_clusters() {
    let mut config = Config::bpe();
    // bpe has vocab=4096, cluster_size=512 → num_clusters=8
    config.mtp_cluster_topk = 100; // Way more than num_clusters
    config.mtp_cluster_vocab_threshold = 1; // Always activate clustered path

    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);

    let cluster_map = cluster_map_round_robin(config.vocab_size, config.mtp_cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * config.n_embd)
        .map(|_| rng.normal())
        .collect();
    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map);

    let mut ctx = katgpt_rs::transformer::ForwardContext::new(&config);
    let mut cache = katgpt_rs::transformer::MultiLayerKVCache::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

    // K >= num_clusters → all tokens should have finite logits
    let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
    let inf_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();

    assert_eq!(
        finite_count, config.vocab_size,
        "K >= num_clusters should compute all tokens"
    );
    assert_eq!(inf_count, 0, "K >= num_clusters should have no -inf logits");
}
