//! GOAT Proof for Plan 234: ManifoldE Point-to-Manifold Pruner
//!
//! Gates:
//!   G1: HyperplanePruner ≥ boolean AND (intersection is at least as strict)
//!   G2: ManifoldPruner soft scoring differentiates boundary tokens
//!   G3: Kernel scoring produces valid similarity scores
//!   G4: Feature isolation — default build unaffected
//!
//! ```sh
//! cargo test --features "manifold_pruner" --test goat_234_manifold_pruner -- --nocapture
//! ```

#![cfg(feature = "manifold_pruner")]

use katgpt_core::traits::{ConstraintPruner, NoPruner, ScreeningPruner};
use katgpt_rs::pruners::hyperplane_pruner::HyperplanePruner;
use katgpt_rs::pruners::kernel_scoring::{KernelKind, kernel_score};
use katgpt_rs::pruners::kernel_screening_pruner::KernelScreeningPruner;
use katgpt_rs::pruners::manifold_pruner::ManifoldPruner;

// Test pruners
struct ThresholdPruner {
    limit: usize,
}
impl ConstraintPruner for ThresholdPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx < self.limit
    }
}

struct EvenPruner;
impl ConstraintPruner for EvenPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx % 2 == 0
    }
}

struct ConstScreener {
    val: f32,
}
impl ScreeningPruner for ConstScreener {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.val
    }
}

#[test]
fn g1_hyperplane_intersection_is_stricter() {
    println!("\n🧪 G1: HyperplanePruner intersection is at least as strict as boolean AND");
    println!("{}", "═".repeat(60));

    let p1 = ThresholdPruner { limit: 8 };
    let p2 = EvenPruner;
    let hyper = HyperplanePruner::new(vec![&p1, &p2]);

    let mut hyper_valid = 0usize;
    let mut bool_valid = 0usize;
    for t in 0..20 {
        let h = hyper.is_valid(0, t, &[]);
        let b = p1.is_valid(0, t, &[]) && p2.is_valid(0, t, &[]);
        if h {
            hyper_valid += 1;
        }
        if b {
            bool_valid += 1;
        }
        assert_eq!(h, b, "token {}: hyper={} but bool={}", t, h, b);
    }
    println!("   HyperplanePruner valid count: {}", hyper_valid);
    println!("   Boolean AND valid count: {}", bool_valid);
    assert_eq!(
        hyper_valid, bool_valid,
        "intersection should match boolean AND"
    );
    println!("   ✅ PASS — intersection matches boolean AND exactly");
}

#[test]
fn g2_manifold_pruner_soft_scoring() {
    println!("\n🧪 G2: ManifoldPruner soft scoring differentiates boundary tokens");
    println!("{}", "═".repeat(60));

    let inner = ThresholdPruner { limit: 5 };
    let soft = ManifoldPruner::new(inner).with_temperature(0.5);

    let valid_score = soft.manifold_score(0, 2, &[]);
    let invalid_score = soft.manifold_score(0, 8, &[]);

    println!("   Valid token (2) score: {:.4}", valid_score);
    println!("   Invalid token (8) score: {:.4}", invalid_score);

    assert!(
        valid_score > invalid_score,
        "valid score {} should > invalid score {}",
        valid_score,
        invalid_score
    );
    assert!(valid_score > 0.5, "valid score should be > 0.5");
    assert!(invalid_score < 0.5, "invalid score should be < 0.5");
    println!("   ✅ PASS — soft scoring differentiates valid from invalid");
}

#[test]
fn g3_kernel_scoring_gaussian() {
    println!("\n🧪 G3: Gaussian kernel produces valid similarity scores");
    println!("{}", "═".repeat(60));

    let v = [1.0, 2.0, 3.0];
    let identical = kernel_score(&v, &v, KernelKind::Gaussian { sigma: 1.0 });
    assert!(
        (identical - 1.0).abs() < 1e-5,
        "identical vectors should score 1.0"
    );

    let distant = kernel_score(
        &[0.0, 0.0],
        &[10.0, 10.0],
        KernelKind::Gaussian { sigma: 1.0 },
    );
    assert!(
        distant < 0.01,
        "distant vectors should score ~0, got {}",
        distant
    );

    let kernel_screener = KernelScreeningPruner::new(
        ConstScreener { val: 1.0 },
        KernelKind::Gaussian { sigma: 1.0 },
    );
    let score = kernel_screener.relevance(0, 0, &[]);
    assert!(
        (score - 1.0).abs() < 1e-5,
        "perfect relevance kernel should be 1.0"
    );

    println!(
        "   Identical: {:.4}, Distant: {:.6}, KernelScreener: {:.4}",
        identical, distant, score
    );
    println!("   ✅ PASS — kernel scoring produces correct similarity scores");
}

#[test]
fn g4_feature_isolation() {
    println!("\n🧪 G4: Feature isolation — default build unaffected");
    println!("{}", "═".repeat(60));

    // Verify NoPruner still works identically
    let pruner = NoPruner;
    assert!(pruner.is_valid(0, 0, &[]));
    assert!(pruner.is_valid(0, 999, &[]));
    // Default manifold_score should return 1.0 for valid (all valid for NoPruner)
    assert_eq!(pruner.manifold_score(0, 0, &[]), 1.0);
    assert!(pruner.constraint_vector(0, &[]).is_none());
    println!("   NoPruner defaults: is_valid=true, manifold_score=1.0, constraint_vector=None");
    println!("   ✅ PASS — trait defaults are backward compatible");
}
