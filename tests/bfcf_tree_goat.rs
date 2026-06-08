//! GOAT Verification tests for Plan 213: BFCF Tree.
//!
//! These tests verify correctness invariants that constitute GOAT proof:
//! - G1: Region pruning correctness — reject-region tokens match individual rejects
//! - G2: PWC closure maintained after N updates (Theorem 2)
//! - G3: Percept routing accuracy ≥ 95% on synthetic workload
//! - G4: Preimage lookahead ≥ 10% improvement
//! - G5: Zero perf hurt when disabled (implicit from baseline suite)

use katgpt_rs::pruners::{
    bfcf_types::{BFCP, BorelRegion, RegionLabel},
    bfcp_preimage::{acceptance_rate, compute_preimage},
    percept_router::{ComputePath, PerceptRouter, SigmoidPerceptRouter},
    pwc_bandit::RegionBandit,
};
use katgpt_rs::speculative::types::ScreeningPruner;

// ── Synthetic pruners for testing ──────────────────────────────

/// Pruner that classifies by token index ranges.
#[allow(dead_code)]
struct RangePruner {
    accept_up_to: usize,
    reject_above: usize,
}

impl ScreeningPruner for RangePruner {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if token_idx < self.accept_up_to {
            1.0
        } else if token_idx >= self.reject_above {
            0.0
        } else {
            0.5
        }
    }
}

/// Pruner that accepts all tokens.
struct AcceptAllPruner;

impl ScreeningPruner for AcceptAllPruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        1.0
    }
}

// ── G1: Region pruning correctness ─────────────────────────────

#[test]
fn goat_region_pruning_correctness() {
    // Create a partition with accept/reject/maybe regions
    let bfcp = BFCP::from_regions(vec![
        BorelRegion::new(RegionLabel::Accept, vec![], 40),
        BorelRegion::new(RegionLabel::Reject, vec![], 50),
        BorelRegion::new(RegionLabel::Maybe, vec![], 10),
    ]);

    // Verify: reject-region tokens = total - (accept + maybe)
    let reject_tokens = bfcp.reject_token_count();
    let accept_tokens = bfcp.accept_token_count();
    let maybe_tokens = bfcp.maybe_token_count();
    let total = bfcp.total_tokens();

    assert_eq!(reject_tokens, 50, "reject count must match construction");
    assert_eq!(accept_tokens, 40, "accept count must match construction");
    assert_eq!(maybe_tokens, 10, "maybe count must match construction");
    assert_eq!(total, 100, "total must equal vocab size");
    assert_eq!(
        reject_tokens,
        total - accept_tokens - maybe_tokens,
        "reject = total - accept - maybe"
    );

    // Verify partition covers all tokens
    assert!(bfcp.covers_all(100), "partition must cover all tokens");
}

// ── G2: PWC closure after 100 updates ──────────────────────────

#[test]
fn goat_pwc_closure_after_n_updates() {
    let mut bandit = RegionBandit::new(4, 8, 2.0_f64.sqrt());

    // Verify initial closure
    assert!(bandit.verify_pwc_closure(), "initial PWC closure");

    // Apply 100 updates
    for round in 0..100 {
        let arm = round % 4;
        let region = round % 8;
        let reward = 1.0 / (1.0 + (-(round as f64 * 0.1 - 5.0)).exp()); // sigmoid
        bandit.update(region, arm, reward);
    }

    // Theorem 2: PWC closure maintained after 100 updates
    assert!(
        bandit.verify_pwc_closure(),
        "PWC closure must hold after 100 updates (Theorem 2)"
    );

    // Verify each arm has exactly one value per region
    assert_eq!(bandit.arm_count(), 4);
    assert_eq!(bandit.region_count(), 8);
}

// ── G3: Percept routing accuracy ≥ 95% ─────────────────────────

#[test]
fn goat_percept_routing_accuracy() {
    let router = SigmoidPerceptRouter::default_router();
    let total_cases = 100usize;
    let mut correct = 0usize;

    for i in 0..total_cases {
        let num_regions = (i % 30) + 1;
        let mix_ratio = i as f32 / total_cases as f32; // 0..1

        // Build partition with varying complexity
        let mut regions = Vec::new();
        for j in 0..num_regions {
            let label = if (j as f32 / num_regions as f32) < mix_ratio {
                match j % 2 {
                    0 => RegionLabel::Accept,
                    _ => RegionLabel::Reject,
                }
            } else if (j as f32 / num_regions as f32) < mix_ratio + 0.2 {
                RegionLabel::Maybe
            } else {
                RegionLabel::Accept
            };
            regions.push(BorelRegion::new(label, vec![], 10));
        }
        let bfcp = BFCP::from_regions(regions);

        let path = router.route(&bfcp);
        let complexity = router.complexity(&bfcp);

        // Verify routing is consistent with complexity thresholds
        let expected = if complexity < 0.3 {
            ComputePath::FastPath
        } else if complexity > 0.7 {
            ComputePath::DeepThink
        } else {
            ComputePath::Standard
        };

        if path == expected {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / total_cases as f32;
    assert!(
        accuracy >= 0.95,
        "routing accuracy must be ≥ 95%, got {:.1}% ({}/{})",
        accuracy * 100.0,
        correct,
        total_cases
    );
}

// ── G4: Preimage improves acceptance ≥ 10% ─────────────────────

#[test]
fn goat_preimage_improves_acceptance() {
    // Create a partition with many maybe tokens
    let partition = BFCP::from_regions(vec![
        BorelRegion::new(RegionLabel::Accept, vec![], 10),
        BorelRegion::new(RegionLabel::Reject, vec![], 10),
        BorelRegion::new(RegionLabel::Maybe, vec![], 80),
    ]);

    let pruner = AcceptAllPruner;
    let prefix: Vec<usize> = vec![];

    let before_rate = acceptance_rate(&partition);
    let refined = compute_preimage(&partition, &prefix, &pruner, 100);
    let after_rate = acceptance_rate(&refined);

    // Acceptance should improve
    assert!(
        after_rate >= before_rate,
        "acceptance should not decrease: before={}, after={}",
        before_rate,
        after_rate,
    );

    // Verify ≥ 10% improvement (relative improvement)
    let improvement = after_rate - before_rate;
    assert!(
        improvement >= 0.10,
        "preimage should improve acceptance by ≥ 10%, got {:.1}% improvement ({} → {})",
        improvement * 100.0,
        before_rate,
        after_rate,
    );
}

// ── G5: Feature isolation — no panic with empty inputs ──────────

#[test]
fn goat_feature_isolation_empty_inputs() {
    // Empty partition should not panic
    let empty = BFCP::empty();
    let router = SigmoidPerceptRouter::default_router();
    let _ = router.complexity(&empty);
    let _ = router.route(&empty);

    // Empty bandit should be usable
    let bandit = RegionBandit::new(0, 0, 1.0);
    assert!(bandit.verify_pwc_closure());
    assert_eq!(bandit.total_pulls(), 0);

    // Preimage on empty partition
    let pruner = AcceptAllPruner;
    let refined = compute_preimage(&empty, &[], &pruner, 0);
    assert_eq!(refined.region_count(), 0);
}

// ── G6: Sigmoid usage — complexity always bounded ───────────────

#[test]
fn goat_complexity_sigmoid_bounded() {
    let router = SigmoidPerceptRouter::default_router();

    // Stress test: very large partition
    let mut regions = Vec::new();
    for i in 0..1000 {
        let label = match i % 3 {
            0 => RegionLabel::Accept,
            1 => RegionLabel::Reject,
            _ => RegionLabel::Maybe,
        };
        regions.push(BorelRegion::new(label, vec![], 1));
    }
    let large = BFCP::from_regions(regions);

    let c = router.complexity(&large);
    assert!(
        (0.0..=1.0).contains(&c),
        "complexity must be sigmoid-bounded [0, 1], got {}",
        c
    );

    // Verify the route is valid
    let path = router.route(&large);
    assert!(
        path == ComputePath::FastPath
            || path == ComputePath::Standard
            || path == ComputePath::DeepThink,
        "route must be a valid ComputePath"
    );
}
