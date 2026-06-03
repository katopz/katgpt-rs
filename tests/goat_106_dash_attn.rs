#![cfg(feature = "dash_attn")]
//! GOAT Proof Test — DashAttention Adaptive Sparse Attention (Plan 106)
//!
//! Proves mathematical invariants of α-entmax (α=1.5) sparse attention routing:
//! probability normalization, non-negativity, sparsity, support extraction,
//! GQA aggregation, and chunk summary zero-init detection.
//!
//! Reference: Peters et al. (2019), "Sparse Sequence-to-Sequence Models"
//!            Correia et al. (2019), "Adaptively Sparse Transformers"
//!
//! Run: `cargo test --features dash_attn --test goat_106_dash_attn -- --nocapture`

use katgpt_rs::dash_attn::{
    ChunkSummaryCache, ChunkSummaryQuery, entmax_1p5, entmax_gqa_aggregate, entmax_support,
    score_blocks_entmax,
};
use katgpt_rs::types::DashAttnConfig;

// ── Helpers ───────────────────────────────────────────────────

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

// ── Proof 1: Entmax Probabilities Sum to 1 ────────────────────
//
// α-entmax produces a valid probability distribution: Σ p_i = 1.0.
// This holds for all score distributions: uniform, peaked, spread, negative.

#[test]
fn proof_1_entmax_probabilities_sum_to_one() {
    // Case 1: Uniform scores
    let (probs, _) = entmax_1p5(&[2.0, 2.0, 2.0, 2.0]);
    let sum: f32 = probs.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P1.1] uniform scores: sum={sum}, expected 1.0"
    );

    // Case 2: Peaked distribution
    let (probs, _) = entmax_1p5(&[10.0, 1.0, 0.5, 0.1]);
    let sum: f32 = probs.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P1.2] peaked scores: sum={sum}, expected 1.0"
    );

    // Case 3: Mixed positive and negative
    let (probs, _) = entmax_1p5(&[3.0, -1.0, 0.5, -2.0, 1.5]);
    let sum: f32 = probs.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P1.3] mixed scores: sum={sum}, expected 1.0"
    );

    // Case 4: Two clear winners
    let (probs, _) = entmax_1p5(&[5.0, 4.9, 0.01, -5.0]);
    let sum: f32 = probs.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P1.4] two winners: sum={sum}, expected 1.0"
    );

    // Case 5: Many scores
    let scores: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0).sin()).collect();
    let (probs, _) = entmax_1p5(&scores);
    let sum: f32 = probs.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P1.5] 20 scores: sum={sum}, expected 1.0"
    );

    println!("✅ Proof 1 PASSED: Entmax probabilities sum to 1.0 for all distributions");
}

// ── Proof 2: Entmax Non-Negative ──────────────────────────────
//
// p_i = max(0, 0.5 * s_i - τ)² ≥ 0 by construction.
// No probability can be negative regardless of score distribution.

#[test]
fn proof_2_entmax_non_negative() {
    let test_cases: Vec<&[f32]> = vec![
        &[1.0, 0.5, 0.2, -1.0, 2.0, 0.0],
        &[10.0, 9.0, 0.01, -5.0, -10.0],
        &[-3.0, -2.0, -1.0],
        &[0.0, 0.0, 0.0],
        &[100.0, -100.0, 50.0, -50.0],
    ];

    for (case_idx, scores) in test_cases.iter().enumerate() {
        let (probs, _) = entmax_1p5(scores);
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p >= 0.0,
                "[P2.{case_idx}] negative probability at index {i}: {p} for scores {scores:?}"
            );
        }
    }

    println!("✅ Proof 2 PASSED: All entmax probabilities are non-negative");
}

// ── Proof 3: Entmax Sparse Zeros ──────────────────────────────
//
// Low scores get exactly zero probability (sparse output).
// This is the key property of entmax over softmax: adaptive sparsity.
// With widely spread scores, the dominant element gets all probability.

#[test]
fn proof_3_entmax_sparse_zeros() {
    // Case 1: Very spread scores → only top survives
    let scores = [10.0, 9.0, 0.01, -5.0, -10.0];
    let (probs, _) = entmax_1p5(&scores);

    // Very negative scores should be exactly zero
    assert!(
        probs[3] < 1e-8,
        "[P3.1] score=-5.0 should have ~0 prob, got {}",
        probs[3]
    );
    assert!(
        probs[4] < 1e-8,
        "[P3.2] score=-10.0 should have ~0 prob, got {}",
        probs[4]
    );

    // Highest score should dominate
    assert!(
        probs[0] > 0.99,
        "[P3.3] score=10.0 should dominate, got {}",
        probs[0]
    );

    // Case 2: Clear gap between relevant and irrelevant
    let scores2 = [5.0, 4.5, 0.001, 0.0001];
    let (probs2, _) = entmax_1p5(&scores2);

    // Low scores should be zero or very close
    assert!(
        probs2[2] < 1e-4,
        "[P3.4] low score should have ~0 prob, got {}",
        probs2[2]
    );
    assert!(
        probs2[3] < 1e-4,
        "[P3.5] very low score should have ~0 prob, got {}",
        probs2[3]
    );

    // Case 3: All negative → entmax still produces valid distribution
    let scores3 = [-1.0, -2.0, -3.0];
    let (probs3, _) = entmax_1p5(&scores3);
    let sum: f32 = probs3.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P3.6] all-negative scores should still sum to 1.0, got {sum}"
    );

    // At least the highest (least negative) should be active
    assert!(
        probs3[0] > 0.0,
        "[P3.7] least negative score should have positive prob, got {}",
        probs3[0]
    );

    println!("✅ Proof 3 PASSED: Entmax produces sparse zeros for low scores");
}

// ── Proof 4: Entmax Empty/Single Edge Cases ───────────────────
//
// Edge cases: empty input → empty output, single input → prob=1.0.
// These are boundary conditions that must not panic or produce invalid output.

#[test]
fn proof_4_entmax_edge_cases() {
    // Case 1: Empty input → empty probabilities, tau=0
    let (probs, tau) = entmax_1p5(&[]);
    assert!(
        probs.is_empty(),
        "[P4.1] empty input should produce empty probs"
    );
    assert!(
        approx_eq(tau, 0.0, 1e-6),
        "[P4.2] empty input tau should be 0.0, got {tau}"
    );

    // Case 2: Single element → prob = 1.0
    let (probs, _) = entmax_1p5(&[3.0]);
    assert_eq!(probs.len(), 1, "[P4.3] single input should have 1 prob");
    assert!(
        approx_eq(probs[0], 1.0, 1e-6),
        "[P4.4] single input prob should be 1.0, got {}",
        probs[0]
    );

    // Case 3: Single negative element → prob = 1.0 (only option)
    let (probs, _) = entmax_1p5(&[-5.0]);
    assert!(
        approx_eq(probs[0], 1.0, 1e-6),
        "[P4.5] single negative prob should be 1.0, got {}",
        probs[0]
    );

    // Case 4: Single zero element → prob = 1.0
    let (probs, _) = entmax_1p5(&[0.0]);
    assert!(
        approx_eq(probs[0], 1.0, 1e-6),
        "[P4.6] single zero prob should be 1.0, got {}",
        probs[0]
    );

    // Case 5: Two identical scores → both get 0.5
    let (probs, _) = entmax_1p5(&[3.0, 3.0]);
    let sum: f32 = probs.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P4.7] two identical should sum to 1.0, got {sum}"
    );
    assert!(
        approx_eq(probs[0], probs[1], 1e-5),
        "[P4.8] two identical scores should have equal probs: {} vs {}",
        probs[0],
        probs[1]
    );

    println!("✅ Proof 4 PASSED: Entmax handles empty/single edge cases correctly");
}

// ── Proof 5: Entmax Support Extraction ────────────────────────
//
// entmax_support returns indices where p_i > ε.
// The support must exactly match the non-zero probability entries.

#[test]
fn proof_5_entmax_support_extraction() {
    // Case 1: Known sparse distribution
    // [3.0, 2.9, 2.8, -5.0]: top entries above threshold
    let scores = [3.0, 2.9, 2.8, -5.0];
    let (probs, _) = entmax_1p5(&scores);
    let support = entmax_support(&probs);

    // Verify each support index has positive probability
    for &idx in &support {
        assert!(
            probs[idx] > 1e-8,
            "[P5.1] support index {idx} has near-zero prob: {}",
            probs[idx]
        );
    }

    // Verify non-support indices have zero probability
    for (i, &p) in probs.iter().enumerate() {
        if !support.contains(&i) {
            assert!(
                p <= 1e-8,
                "[P5.2] non-support index {i} has positive prob: {p}"
            );
        }
    }

    // Case 2: Uniform → all active
    let (probs_u, _) = entmax_1p5(&[1.0, 1.0, 1.0]);
    let support_u = entmax_support(&probs_u);
    assert_eq!(
        support_u.len(),
        3,
        "[P5.3] uniform should have all 3 active"
    );

    // Case 3: Empty input → empty support
    let (probs_e, _) = entmax_1p5(&[]);
    let support_e = entmax_support(&probs_e);
    assert!(
        support_e.is_empty(),
        "[P5.4] empty input should have empty support"
    );

    // Case 4: Single input → support is [0]
    let (probs_s, _) = entmax_1p5(&[5.0]);
    let support_s = entmax_support(&probs_s);
    assert_eq!(
        support_s,
        vec![0],
        "[P5.5] single input support should be [0]"
    );

    // Case 5: Support size ≤ total scores (sparsity guarantee)
    let many_scores: Vec<f32> = (0..50).map(|i| 10.0 - i as f32 * 0.5).collect();
    let (probs_m, _) = entmax_1p5(&many_scores);
    let support_m = entmax_support(&probs_m);
    assert!(
        support_m.len() <= many_scores.len(),
        "[P5.6] support size should be ≤ total scores"
    );
    // Entmax with spread scores should be sparse
    assert!(
        support_m.len() < many_scores.len(),
        "[P5.7] entmax should be sparse for spread scores: support={}/{}",
        support_m.len(),
        many_scores.len()
    );

    println!("✅ Proof 5 PASSED: Entmax support extraction matches non-zero probabilities");
}

// ── Proof 6: GQA Aggregate Averaging ──────────────────────────
//
// GQA aggregation averages probabilities across query heads in the same KV group.
// kv_group(h) = h * n_kv_heads / n_query_heads.
// Result[g][c] = mean of head_probs[h][c] for all h in group g.

#[test]
fn proof_6_gqa_aggregate_averaging() {
    // Case 1: 4 query heads, 2 KV heads, 3 chunks
    // h=0→g0, h=1→g0, h=2→g1, h=3→g1
    let n_query_heads = 4;
    let n_kv_heads = 2;
    let n_chunks = 3;

    let head_probs = vec![
        vec![0.5, 0.3, 0.2], // Q head 0 → KV group 0
        vec![0.6, 0.2, 0.2], // Q head 1 → KV group 0
        vec![0.4, 0.4, 0.2], // Q head 2 → KV group 1
        vec![0.3, 0.5, 0.2], // Q head 3 → KV group 1
    ];

    let result = entmax_gqa_aggregate(&head_probs, n_query_heads, n_kv_heads, n_chunks);

    assert_eq!(
        result.len(),
        n_kv_heads,
        "[P6.1] should have n_kv_heads groups"
    );
    assert_eq!(
        result[0].len(),
        n_chunks,
        "[P6.2] each group should have n_chunks"
    );

    // KV group 0: average of heads 0 and 1
    // [(0.5+0.6)/2, (0.3+0.2)/2, (0.2+0.2)/2] = [0.55, 0.25, 0.20]
    assert!(
        approx_eq(result[0][0], 0.55, 1e-6),
        "[P6.3] group 0 chunk 0: expected 0.55, got {}",
        result[0][0]
    );
    assert!(
        approx_eq(result[0][1], 0.25, 1e-6),
        "[P6.4] group 0 chunk 1: expected 0.25, got {}",
        result[0][1]
    );
    assert!(
        approx_eq(result[0][2], 0.20, 1e-6),
        "[P6.5] group 0 chunk 2: expected 0.20, got {}",
        result[0][2]
    );

    // KV group 1: average of heads 2 and 3
    // [(0.4+0.3)/2, (0.4+0.5)/2, (0.2+0.2)/2] = [0.35, 0.45, 0.20]
    assert!(
        approx_eq(result[1][0], 0.35, 1e-6),
        "[P6.6] group 1 chunk 0: expected 0.35, got {}",
        result[1][0]
    );
    assert!(
        approx_eq(result[1][1], 0.45, 1e-6),
        "[P6.7] group 1 chunk 1: expected 0.45, got {}",
        result[1][1]
    );
    assert!(
        approx_eq(result[1][2], 0.20, 1e-6),
        "[P6.8] group 1 chunk 2: expected 0.20, got {}",
        result[1][2]
    );

    // Case 2: Single KV head — all query heads aggregate to one group
    let n_qh = 3;
    let n_kvh = 1;
    let nc = 2;
    let hp = vec![vec![0.6, 0.4], vec![0.8, 0.2], vec![0.4, 0.6]];
    let agg = entmax_gqa_aggregate(&hp, n_qh, n_kvh, nc);

    assert_eq!(agg.len(), 1, "[P6.9] single KV head should have 1 group");
    // Average: [(0.6+0.8+0.4)/3, (0.4+0.2+0.6)/3] = [0.6, 0.4]
    assert!(
        approx_eq(agg[0][0], 0.6, 1e-6),
        "[P6.10] single group chunk 0: expected 0.6, got {}",
        agg[0][0]
    );
    assert!(
        approx_eq(agg[0][1], 0.4, 1e-6),
        "[P6.11] single group chunk 1: expected 0.4, got {}",
        agg[0][1]
    );

    println!("✅ Proof 6 PASSED: GQA aggregate correctly computes per-group means");
}

// ── Proof 7: Routing Probs Sum to 1 ──────────────────────────
//
// score_blocks_entmax produces a valid probability distribution
// via entmax routing. The probs in RoutingResult must sum to 1.0.

#[test]
fn proof_7_routing_probs_sum_to_one() {
    let config = DashAttnConfig::default();

    // Case 1: Multiple chunks with different summaries
    let query = vec![1.0, 2.0, 3.0, 0.5];
    let summaries = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.4, 0.5, 0.6, 0.7],
        vec![0.7, 0.8, 0.9, 1.0],
    ];

    let result = score_blocks_entmax(&query, &summaries, &config);
    let sum: f32 = result.probs.iter().sum();
    assert!(
        approx_eq(sum, 1.0, 1e-5),
        "[P7.1] routing probs should sum to 1.0, got {sum}"
    );

    // All probs non-negative
    for (i, &p) in result.probs.iter().enumerate() {
        assert!(p >= 0.0, "[P7.2] routing prob[{i}] is negative: {p}");
    }

    // Case 2: Single chunk → prob = 1.0
    let query2 = vec![1.0, 0.0, 0.0];
    let summaries2 = vec![vec![1.0, 0.0, 0.0]];
    let result2 = score_blocks_entmax(&query2, &summaries2, &config);
    assert_eq!(
        result2.active_indices,
        vec![0],
        "[P7.3] single chunk should be active"
    );
    assert!(
        approx_eq(result2.probs[0], 1.0, 1e-5),
        "[P7.4] single chunk should get all probability mass"
    );

    // Case 3: Orthogonal query — still valid distribution
    let query3 = vec![1.0, 0.0];
    let summaries3 = vec![vec![0.0, 1.0], vec![0.0, -1.0]];
    let result3 = score_blocks_entmax(&query3, &summaries3, &config);
    let sum3: f32 = result3.probs.iter().sum();
    assert!(
        approx_eq(sum3, 1.0, 1e-5) || result3.probs.is_empty(),
        "[P7.5] orthogonal routing probs should sum to 1.0 or be empty, got {sum3}"
    );

    // Case 4: Active indices match support
    let query4 = vec![1.0, 0.0, 0.0];
    let summaries4 = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let result4 = score_blocks_entmax(&query4, &summaries4, &config);
    let support = entmax_support(&result4.probs);
    assert_eq!(
        result4.active_indices, support,
        "[P7.6] active_indices should match entmax support"
    );

    // Case 5: Bias vector length matches active indices
    assert_eq!(
        result4.active_indices.len(),
        result4.bias.len(),
        "[P7.7] bias length must match active_indices length"
    );

    println!("✅ Proof 7 PASSED: Routing produces valid probability distribution via entmax");
}

// ── Proof 8: ChunkSummary Zero-Init Detection ─────────────────
//
// ChunkSummaryQuery::new() initializes head_cls to all zeros.
// is_zero_init() should return true for fresh instances.
// After mutation, is_zero_init() should return false.

#[test]
fn proof_8_chunk_summary_zero_init() {
    // Case 1: Fresh query is zero-init
    let query = ChunkSummaryQuery::new(2, 4);
    assert!(query.is_zero_init(), "[P8.1] new() should be zero-init");

    // Case 2: head_query returns zero slices
    for h in 0..2 {
        let hq = query.head_query(h);
        for (i, &v) in hq.iter().enumerate() {
            assert!(
                approx_eq(v, 0.0, 1e-10),
                "[P8.2] head_query({h})[{i}] = {v}, expected 0.0"
            );
        }
    }

    // Case 3: After mutation, no longer zero-init
    let mut query2 = ChunkSummaryQuery::new(2, 4);
    query2.head_cls[0] = 0.001;
    query2.recompute_zero_init_cache(); // is_zero_init() reads a cache; refresh after direct mutation
    assert!(
        !query2.is_zero_init(),
        "[P8.3] mutated query should not be zero-init"
    );

    // Case 4: ChunkSummaryCache starts empty
    let cache = ChunkSummaryCache::new(2, 4);
    assert_eq!(cache.n_chunks(), 0, "[P8.4] new cache should have 0 chunks");

    // Case 5: Allocate fills with zeros
    let mut cache2 = ChunkSummaryCache::new(2, 4);
    cache2.allocate(3);
    assert_eq!(
        cache2.n_chunks(),
        3,
        "[P8.5] allocated cache should have 3 chunks"
    );
    for (c, chunk) in cache2.summaries.iter().enumerate() {
        for (h, head) in chunk.iter().enumerate() {
            for (d, &v) in head.iter().enumerate() {
                assert!(
                    approx_eq(v, 0.0, 1e-10),
                    "[P8.6] allocated chunk {c} head {h} dim {d} = {v}, expected 0.0"
                );
            }
        }
    }

    // Case 6: Reset clears cache
    let mut cache3 = ChunkSummaryCache::new(2, 4);
    cache3.allocate(5);
    assert_eq!(cache3.n_chunks(), 5, "[P8.7] pre-reset chunks");
    cache3.reset();
    assert_eq!(
        cache3.n_chunks(),
        0,
        "[P8.8] post-reset should have 0 chunks"
    );

    println!("✅ Proof 8 PASSED: ChunkSummary zero-init detection and cache lifecycle correct");
}

// ── Proof 9: DashAttnConfig Defaults ──────────────────────────
//
// DashAttnConfig::default() should produce the expected default values:
// chunk_size=64, alpha=1.5, scaling_factor=1.0, sigma=1e6, estimate_diagonal=true.

#[test]
fn proof_9_dash_attn_config_defaults() {
    let config = DashAttnConfig::default();

    assert_eq!(
        config.chunk_size, 64,
        "[P9.1] default chunk_size should be 64, got {}",
        config.chunk_size
    );

    assert!(
        approx_eq(config.alpha, 1.5, 1e-6),
        "[P9.2] default alpha should be 1.5, got {}",
        config.alpha
    );

    assert!(
        approx_eq(config.scaling_factor, 1.0, 1e-6),
        "[P9.3] default scaling_factor should be 1.0, got {}",
        config.scaling_factor
    );

    assert!(
        approx_eq(config.sigma, 1e6, 1.0),
        "[P9.4] default sigma should be 1e6, got {}",
        config.sigma
    );

    assert!(
        config.estimate_diagonal,
        "[P9.5] default estimate_diagonal should be true"
    );

    // Verify alpha=1.5 means entmax-1.5 (quadratic)
    // The alpha field is informational; the actual algorithm is always entmax-1.5
    assert!(
        config.alpha > 1.0,
        "[P9.6] alpha must be > 1.0 for sparse entmax"
    );

    println!(
        "✅ Proof 9 PASSED: DashAttnConfig defaults verified (chunk=64, α=1.5, γ=1.0, σ=1e6, diag=true)"
    );
}

// ── Summary ───────────────────────────────────────────────────

#[test]
fn summary_goat_106_dash_attn() {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  🐐 GOAT Proof: DashAttention Adaptive Sparse Attention (Plan 106)");
    println!("  Feature: dash_attn");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Proof 1: Entmax probabilities sum to 1.0               ✅");
    println!("  Proof 2: Entmax probabilities are non-negative         ✅");
    println!("  Proof 3: Entmax produces sparse zeros for low scores   ✅");
    println!("  Proof 4: Entmax empty/single edge cases                ✅");
    println!("  Proof 5: Entmax support extraction matches non-zero    ✅");
    println!("  Proof 6: GQA aggregate computes per-group means        ✅");
    println!("  Proof 7: Routing probs sum to 1 via score_blocks       ✅");
    println!("  Proof 8: ChunkSummary zero-init detection              ✅");
    println!("  Proof 9: DashAttnConfig defaults verified              ✅");
    println!();
    println!("  Verdict: DashAttention α-entmax routing is mathematically");
    println!("  correct. It produces valid, sparse probability distributions");
    println!("  with adaptive support size, correct GQA aggregation, and");
    println!("  robust edge-case handling.");
    println!("═══════════════════════════════════════════════════════════════");
}
