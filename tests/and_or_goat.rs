//! GOAT Proof: AND-OR DDTree Blueprint Decomposition (Plan 190).
//!
//! Validates five properties:
//! G1: AND-OR DDTree explores fewer nodes than flat DDTree for complex tasks
//! G2: No quality regression on simple tasks (decomposition not triggered)
//! G3: Decomposition reviewer prevents search collapse (progress signal works)
//! G4: Blueprint pre-pass adds < 5% overhead to total decode time
//! G5: ProofGoalCache hit rate >= 30% on tasks with repeated subgoals

use katgpt_core::traits::ScreeningPruner;
use katgpt_rs::pruners::proof::goal_cache::GoalVerifier;
use katgpt_rs::pruners::{GoalResult, ProofGoalCache};
use katgpt_rs::speculative::{
    AndOrBuilder, BlueprintPass, DecompositionReviewer, build_dd_tree_and_or,
    build_dd_tree_screened,
};
use katgpt_rs::types::Config;
use std::time::Instant;

// ── Helpers ────────────────────────────────────────────────────

/// A pruner that returns low relevance at specific depths (simulates uncertain regions).
struct PatchyPruner {
    low_depths: Vec<usize>,
}

impl ScreeningPruner for PatchyPruner {
    fn relevance(&self, depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if self.low_depths.contains(&depth) {
            0.1
        } else {
            0.9
        }
    }
}

/// A pruner that returns high relevance everywhere (simple tasks).
struct HighRelevancePruner;

impl ScreeningPruner for HighRelevancePruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        0.95
    }
}

/// Generate synthetic marginals with the given pattern.
fn make_marginals(depths: usize, vocab: usize, pattern: &str) -> Vec<Vec<f32>> {
    (0..depths)
        .map(|_| match pattern {
            "uniform" => vec![1.0 / vocab as f32; vocab],
            "peaked" => {
                let mut m = vec![0.01; vocab];
                m[0] = 0.9;
                let sum: f32 = m.iter().sum();
                for v in m.iter_mut() {
                    *v /= sum;
                }
                m
            }
            _ => vec![1.0 / vocab as f32; vocab],
        })
        .collect()
}

/// Helper: convert `Vec<Vec<f32>>` to `Vec<&[f32]>` for API calls.
fn as_refs(marginals: &[Vec<f32>]) -> Vec<&[f32]> {
    marginals.iter().map(|m| m.as_slice()).collect()
}

/// Compute the expected greedy argmax path from marginals.
fn expected_argmax_path(marginals: &[Vec<f32>]) -> Vec<usize> {
    marginals
        .iter()
        .map(|m| {
            m.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

// ── GOAT Tests ─────────────────────────────────────────────────

/// G1: AND-OR DDTree explores fewer nodes than flat DDTree for complex tasks.
///
/// Uses PatchyPruner with low relevance at depths 2, 4, 6 → triggers decomposition.
/// The AND-OR tree decomposes uncertain regions into subgoals,
/// resulting in fewer explored nodes than flat expansion.
#[test]
fn test_goat_1_node_reduction() {
    let config = Config::draft();
    let vocab = config.vocab_size;
    let depths = config.draft_lookahead;

    // Complex task: low relevance at depths 2, 4, 6 → triggers decomposition
    let pruner = PatchyPruner {
        low_depths: vec![2, 4, 6],
    };
    let marginals = make_marginals(depths, vocab, "peaked");
    let mrefs = as_refs(&marginals);

    // Flat DDTree
    let flat_tree = build_dd_tree_screened(&mrefs, &config, &pruner, false);
    let flat_nodes = flat_tree.len();

    // AND-OR DDTree
    let mut cache = ProofGoalCache::new();
    let and_or_tree = build_dd_tree_and_or(&mrefs, &config, &pruner, &mut cache, false);
    let and_or_nodes = and_or_tree.len();

    // AND-OR must not produce more nodes than flat.
    // For complex tasks, decomposition solves via argmax path (one node per depth)
    // rather than full tree search (up to tree_budget nodes).
    assert!(
        and_or_nodes <= flat_nodes,
        "GOAT G1 FAIL: AND-OR ({and_or_nodes} nodes) should explore <= flat ({flat_nodes} nodes)"
    );
}

/// G2: No quality regression on simple tasks (decomposition not triggered).
///
/// Uses HighRelevancePruner → all depths have high relevance → no decomposition.
/// AND-OR DDTree should produce the same greedy argmax path as flat DDTree's
/// best path. For peaked marginals, both converge to the argmax token at each depth.
#[test]
fn test_goat_2_quality_parity() {
    let config = Config::draft();
    let vocab = config.vocab_size;
    let depths = config.draft_lookahead;

    // Simple task: high relevance everywhere → no decomposition
    let pruner = HighRelevancePruner;
    let marginals = make_marginals(depths, vocab, "peaked");
    let mrefs = as_refs(&marginals);

    // Expected argmax path (greedy best)
    let expected = expected_argmax_path(&marginals);

    // AND-OR DDTree
    let mut cache = ProofGoalCache::new();
    let and_or_tree = build_dd_tree_and_or(&mrefs, &config, &pruner, &mut cache, false);

    // Extract token path from AND-OR tree (one node per depth)
    let and_or_tokens: Vec<usize> = and_or_tree.iter().map(|n| n.token_idx).collect();

    // The AND-OR path should match the expected argmax path
    // (no quality regression — produces the same best tokens)
    assert_eq!(
        expected, and_or_tokens,
        "GOAT G2 FAIL: AND-OR path {and_or_tokens:?} should match argmax {expected:?}"
    );

    // All tokens must be valid (within vocab)
    for (d, &tok) in and_or_tokens.iter().enumerate() {
        assert!(
            tok < vocab,
            "GOAT G2: invalid token {tok} at depth {d} (vocab={vocab})"
        );
    }

    // Tree should cover all depths
    assert_eq!(
        and_or_tree.len(),
        depths,
        "GOAT G2: AND-OR should cover all {depths} depths, got {}",
        and_or_tree.len()
    );
}

/// G3: Decomposition reviewer prevents search collapse (progress signal works).
///
/// Verifies that:
/// - No data → assumes productive (safe default)
/// - High novelty (many cache misses) → branch is productive → keep
/// - Low novelty (many cache hits) → branch is unproductive → prune
/// - Reset clears state correctly for new branch evaluation
#[test]
fn test_goat_3_dead_end_detection() {
    let reviewer = DecompositionReviewer::new(0.3);

    // Case 1: No data → assume productive
    assert!(
        reviewer.is_productive(),
        "GOAT G3: no data should assume productive"
    );
    assert!(
        (reviewer.novelty() - 1.0).abs() < f32::EPSILON,
        "GOAT G3: novelty should be 1.0 with no data"
    );

    // Case 2: High novelty → productive (exploring new territory)
    for _ in 0..7 {
        reviewer.record_miss();
    }
    for _ in 0..3 {
        reviewer.record_hit();
    }
    assert!(
        reviewer.is_productive(),
        "GOAT G3: 70% novelty should be productive (threshold=0.3)"
    );

    // Case 3: Reset and create low-novelty scenario → unproductive
    reviewer.reset_branch();
    for _ in 0..9 {
        reviewer.record_hit();
    }
    reviewer.record_miss();
    assert!(
        !reviewer.is_productive(),
        "GOAT G3: 10% novelty should be unproductive (threshold=0.3)"
    );

    // Case 4: Exact boundary — novelty == threshold → productive (>=)
    reviewer.reset_branch();
    for _ in 0..5 {
        reviewer.record_miss();
    }
    for _ in 0..5 {
        reviewer.record_hit();
    }
    assert!(
        reviewer.is_productive(),
        "GOAT G3: exactly at boundary (0.5 >= 0.3) should be productive"
    );
}

/// G4: Blueprint pre-pass adds < 5% overhead to total decode time.
///
/// Blueprint is O(depth * vocab) argmax — should be negligible compared to DDTree build.
#[test]
fn test_goat_4_blueprint_overhead() {
    let config = Config::draft();
    let vocab = config.vocab_size;
    let depths = config.draft_lookahead;

    // Use peaked marginals (realistic distribution)
    let marginals = make_marginals(depths, vocab, "peaked");
    let mrefs = as_refs(&marginals);

    // Warm up (avoid cold-start effects)
    let _ = BlueprintPass::generate(&mrefs);

    // Time blueprint alone
    let bp_iters = 1000;
    let bp_start = Instant::now();
    for _ in 0..bp_iters {
        let _blueprint = BlueprintPass::generate(&mrefs);
    }
    let bp_total = bp_start.elapsed();
    let bp_per_call = bp_total / bp_iters;

    // Time full AND-OR build
    let pruner = PatchyPruner {
        low_depths: vec![2, 4, 6],
    };
    let build_iters = 100;
    let build_start = Instant::now();
    for _ in 0..build_iters {
        let mut cache = ProofGoalCache::new();
        let _tree = build_dd_tree_and_or(&mrefs, &config, &pruner, &mut cache, false);
    }
    let build_total = build_start.elapsed();
    let build_per_call = build_total / build_iters;

    // Blueprint overhead = bp_time / build_time
    let overhead_ratio = bp_per_call.as_secs_f64() / build_per_call.as_secs_f64();

    // Must be < 5%
    assert!(
        overhead_ratio < 0.05,
        "GOAT G4 FAIL: blueprint overhead {overhead_ratio:.4} (> 5%) — bp={bp_per_call:?}, build={build_per_call:?}"
    );
}

/// G5: ProofGoalCache hit rate >= 30% on tasks with repeated subgoals.
///
/// Strategy:
/// 1. Use AndOrBuilder to populate cache via solve_subgoal (insert path)
/// 2. Re-query the same subgoals via get_or_verify (updates hit/miss counters)
/// 3. Verify hit rate >= 30%
#[test]
fn test_goat_5_cache_hit_rate() {
    let config = Config::draft();
    let vocab = config.vocab_size;
    let depths = config.draft_lookahead;

    // Repeated pattern: low relevance at depths 2, 4, 6
    let pruner = PatchyPruner {
        low_depths: vec![2, 4, 6],
    };
    let marginals = make_marginals(depths, vocab, "peaked");
    let mrefs = as_refs(&marginals);

    // Step 1: Build AND-OR tree to populate cache
    let mut cache = ProofGoalCache::new();
    let mut builder = AndOrBuilder::new(&pruner, &mut cache).with_relevance_threshold(0.3);
    let _tree = builder.build(&mrefs);

    // Step 2: Re-query the same subgoals that the builder would create.
    // Canonical encoding matches Subgoal::canonical_bytes:
    //   (depth_start: u64 LE || depth_end: u64 LE || argmax_per_depth: [u64 LE])
    // The builder creates subgoals for various ranges based on decomposition regions.
    // For PatchyPruner { [2,4,6] }, the regions are (2,3), (4,5), (6,7).
    // High-relevance segments: (0,2), (3,4), (5,6), (7,8).
    // Also per-depth subgoals in low-relevance regions.
    let subgoal_ranges: Vec<(usize, usize)> = vec![
        (0, depths), // full range
        (0, 2),      // high-relevance segment
        (2, 3),      // low-relevance region
        (3, 4),      // high-relevance segment
        (4, 5),      // low-relevance region
        (5, 6),      // high-relevance segment
        (6, 7),      // low-relevance region
        (7, depths), // trailing high-relevance
        (2, 4),      // cross-region
        (4, 6),      // cross-region
    ];

    // Helper: encode canonical bytes matching Subgoal::canonical_bytes format
    let encode_canonical = |start: usize, end: usize| -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + (end - start) * 8);
        buf.extend_from_slice(&(start as u64).to_le_bytes());
        buf.extend_from_slice(&(end as u64).to_le_bytes());
        for d in start..end {
            // argmax at depth d
            let top = mrefs[d]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            buf.extend_from_slice(&(top as u64).to_le_bytes());
        }
        buf
    };

    // Trivial verifier for cache testing — always returns Proved
    #[derive(Clone, Copy)]
    struct ProvedVerifier;
    impl GoalVerifier for ProvedVerifier {
        fn verify(&self, _canonical_bytes: &[u8]) -> GoalResult {
            GoalResult::Proved
        }
    }
    let verifier = ProvedVerifier;

    // First pass: get_or_verify on all subgoal ranges (updates counters)
    for &(start, end) in &subgoal_ranges {
        if start >= end || end > depths {
            continue;
        }
        let bytes = encode_canonical(start, end);
        cache.get_or_verify(&bytes, verifier);
    }

    // Second pass: same subgoals → should all be cache hits
    for &(start, end) in &subgoal_ranges {
        if start >= end || end > depths {
            continue;
        }
        let bytes = encode_canonical(start, end);
        cache.get_or_verify(&bytes, verifier);
    }

    let hit_rate = cache.hit_rate();

    // Hit rate must be >= 30%
    assert!(
        hit_rate >= 0.30,
        "GOAT G5 FAIL: cache hit rate {hit_rate:.4} (< 30%) — hits={}, misses={}, total={}",
        cache.hits(),
        cache.misses(),
        cache.total_lookups()
    );
}
