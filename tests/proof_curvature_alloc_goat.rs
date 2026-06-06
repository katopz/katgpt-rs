//! GOAT Proof Tests for Curvature-Influence Allocation Bandit (Plan 183)
//!
//! 6/6 proofs required for GOAT qualification.
//! Run: `cargo test --test proof_curvature_alloc_goat`

use katgpt_rs::pruners::{CurvatureInfluenceScorer, CurvatureWeightedBudget, EosProxyScorer};

// ═══════════════════════════════════════════════════════════════
// G1: EosProxyScorer initializes with zero influence for all groups
// ═══════════════════════════════════════════════════════════════

#[test]
fn g1_scorer_initializes_zero_influence() {
    let num_groups = 7;
    let scorer = EosProxyScorer::new(num_groups, 0.1);
    assert_eq!(scorer.num_groups(), num_groups);

    for k in 0..num_groups {
        let inf = scorer.curvature_influence(k);
        assert!(
            inf.abs() < 1e-6,
            "Group {k} should have zero influence at init, got {inf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// G2: After updating persistence and alignment, influence ∈ [0, 1]
// ═══════════════════════════════════════════════════════════════

#[test]
fn g2_influence_in_bounds_after_updates() {
    let mut scorer = EosProxyScorer::new(5, 0.3);

    // Feed diverse data
    for k in 0..5 {
        scorer.update_persistence(k, 0.1 * (k as f32 + 1.0));
        scorer.update_alignment(k, &[0.1, 0.7, 0.15, 0.05]);
    }

    // Additional updates to create variance
    for _ in 0..5 {
        scorer.update_persistence(0, 0.9);
        scorer.update_persistence(4, 0.1);
    }

    for k in 0..5 {
        let inf = scorer.curvature_influence(k);
        assert!(
            (0.0..=1.0).contains(&inf),
            "Influence for group {k} should be in [0,1], got {inf}"
        );
    }

    // Out-of-bounds group returns 0
    assert!(
        (scorer.curvature_influence(99) - 0.0).abs() < 1e-6,
        "Out-of-bounds group should return 0"
    );
}

// ═══════════════════════════════════════════════════════════════
// G3: CurvatureWeightedBudget allocation sums to total_budget
// ═══════════════════════════════════════════════════════════════

#[test]
fn g3_budget_allocation_sums_to_total() {
    let scorer = EosProxyScorer::new(5, 0.1);
    let budget = CurvatureWeightedBudget::new();

    for total in [50, 100, 997, 1] {
        let alloc = budget.allocate(total, 5, &scorer);
        let sum: usize = alloc.iter().sum();
        assert_eq!(sum, total, "Allocation should sum to {total}, got {sum}");
    }

    // Edge case: zero budget
    let alloc = budget.allocate(0, 5, &scorer);
    assert!(alloc.is_empty(), "Zero budget should return empty");

    // Edge case: zero depth
    let alloc = budget.allocate(100, 0, &scorer);
    assert!(alloc.is_empty(), "Zero depth should return empty");
}

// ═══════════════════════════════════════════════════════════════
// G4: High-influence positions get more budget
// ═══════════════════════════════════════════════════════════════

#[test]
fn g4_high_influence_gets_more_budget() {
    let mut scorer = EosProxyScorer::new(4, 0.5);

    // Make group 3 have high influence
    for _ in 0..20 {
        scorer.update_persistence(3, 0.95);
    }
    scorer.update_alignment(3, &[0.97, 0.01, 0.01, 0.01]);

    // Group 0 has moderate influence
    for _ in 0..5 {
        scorer.update_persistence(0, 0.3);
    }
    scorer.update_alignment(0, &[0.25, 0.25, 0.25, 0.25]); // Uniform → low alignment

    // Groups 1 and 2 have zero influence (no updates)
    scorer.update_alignment(1, &[0.25, 0.25, 0.25, 0.25]);
    scorer.update_alignment(2, &[0.25, 0.25, 0.25, 0.25]);

    let budget = CurvatureWeightedBudget {
        floor_ratio: 0.1,
        max_boost: 2.0, // Allow significant variation
    };
    let alloc = budget.allocate(100, 4, &scorer);

    assert!(
        alloc[3] > alloc[1],
        "Group 3 (high influence) should get more than group 1 (low), got {} vs {}",
        alloc[3],
        alloc[1]
    );
    assert!(
        alloc[3] > alloc[2],
        "Group 3 (high influence) should get more than group 2 (low), got {} vs {}",
        alloc[3],
        alloc[2]
    );
}

// ═══════════════════════════════════════════════════════════════
// G5: Floor guarantee — every position gets at least floor_ratio allocation
// ═══════════════════════════════════════════════════════════════

#[test]
fn g5_floor_guarantee() {
    let mut scorer = EosProxyScorer::new(8, 0.3);
    // Make one group dominate
    for _ in 0..30 {
        scorer.update_persistence(5, 0.99);
    }
    scorer.update_alignment(5, &[0.98, 0.005, 0.005, 0.005, 0.005]);

    let budget = CurvatureWeightedBudget {
        floor_ratio: 0.15,
        max_boost: 0.5,
    };
    let alloc = budget.allocate(200, 8, &scorer);

    // Every position should get at least 1 (floor_ratio guarantees a minimum weight)
    for (i, &a) in alloc.iter().enumerate() {
        assert!(
            a > 0,
            "Position {i} should get at least 1 token (floor guarantee), got {a}"
        );
    }

    let sum: usize = alloc.iter().sum();
    assert_eq!(sum, 200, "Allocation should still sum to total");
}

// ═══════════════════════════════════════════════════════════════
// G6: Concentration computation from scores is correct
// ═══════════════════════════════════════════════════════════════

#[test]
fn g6_concentration_computation() {
    let mut scorer = EosProxyScorer::new(3, 0.5);

    // Fully concentrated: one score massively dominates (use raw logits)
    scorer.update_alignment(0, &[100.0, 0.0, 0.0, 0.0]);
    // Perfectly uniform
    scorer.update_alignment(1, &[0.25, 0.25, 0.25, 0.25]);
    // Moderately concentrated
    scorer.update_alignment(2, &[10.0, 1.0, 0.5, 0.5]);

    // Concentrated → high alignment (close to 1)
    assert!(
        scorer.alignment(0) > 0.5,
        "Concentrated scores should yield high alignment, got {}",
        scorer.alignment(0)
    );

    // Uniform → low alignment (close to 0)
    assert!(
        scorer.alignment(1) < 0.5,
        "Uniform scores should yield low alignment, got {}",
        scorer.alignment(1)
    );

    // Moderate → between concentrated and uniform
    assert!(
        scorer.alignment(2) > scorer.alignment(1) && scorer.alignment(2) < scorer.alignment(0),
        "Moderate concentration should be between uniform and concentrated: got {}",
        scorer.alignment(2)
    );
}
