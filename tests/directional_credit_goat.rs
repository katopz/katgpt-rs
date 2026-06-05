//! GOAT Proof: Direction-Adaptive Credit — Entropy-Bifurcated Pruning (Plan 184).
//!
//! Validates six properties:
//! G1: EntropyBifurcatedPruner with low top-1 prob returns higher relevance than raw inner
//! G2: EntropyBifurcatedPruner with high top-1 prob returns same relevance as inner
//! G3: Threshold boundary works correctly
//! G4: Relax factor scaling is correct
//! G5: Delegates correctly to inner pruner for non-fork tokens
//! G6: EntropyState updates correctly via update_entropy()

use katgpt_rs::pruners::{EntropyBifurcatedPruner, EntropyState};
use katgpt_rs::speculative::types::ScreeningPruner;

// ── Helpers ────────────────────────────────────────────────────

/// A simple pruner that returns a fixed relevance.
#[derive(Debug, Clone)]
struct FixedPruner {
    relevance_val: f32,
}

impl ScreeningPruner for FixedPruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.relevance_val
    }
}

fn make_pruner(val: f32, threshold: f32, relax: f32) -> EntropyBifurcatedPruner<FixedPruner> {
    EntropyBifurcatedPruner::new(FixedPruner { relevance_val: val }, threshold, relax)
}

// ── GOAT Tests ─────────────────────────────────────────────────

/// G1: Low top-1 prob returns higher relevance than raw inner.
#[test]
fn goat_g1_forking_scales_relevance_up() {
    let mut pruner = make_pruner(0.5, 0.5, 0.3);
    pruner.update_entropy(0.2); // below threshold → Forking
    let rel = pruner.relevance(0, 0, &[]);
    let expected = 0.5 * (1.0 + 0.3);
    assert!(
        (rel - expected).abs() < 1e-6,
        "expected {expected}, got {rel}"
    );
    assert!(rel > 0.5, "relaxed relevance should exceed base");
}

/// G2: High top-1 prob returns same relevance as inner.
#[test]
fn goat_g2_scaffolding_returns_inner_relevance() {
    let mut pruner = make_pruner(0.5, 0.5, 0.3);
    pruner.update_entropy(0.9); // above threshold → Scaffolding
    let rel = pruner.relevance(0, 0, &[]);
    assert!((rel - 0.5).abs() < 1e-6, "expected 0.5, got {rel}");
}

/// G3: Threshold boundary — exactly at threshold is scaffolding, below is forking.
#[test]
fn goat_g3_threshold_boundary() {
    let mut pruner = make_pruner(1.0, 0.5, 0.5);

    // Exactly at threshold → Scaffolding (not strictly less than)
    pruner.update_entropy(0.5);
    assert_eq!(pruner.state(), EntropyState::Scaffolding);
    assert!((pruner.relevance(0, 0, &[]) - 1.0).abs() < 1e-6);

    // Just below threshold → Forking
    pruner.update_entropy(0.4999);
    assert_eq!(pruner.state(), EntropyState::Forking);
    assert!((pruner.relevance(0, 0, &[]) - 1.5).abs() < 1e-6);
}

/// G4: Relax factor scaling is mathematically correct.
#[test]
fn goat_g4_relax_factor_scaling() {
    let mut pruner = make_pruner(0.8, 0.5, 0.25);
    pruner.update_entropy(0.1); // Forking
    let rel = pruner.relevance(0, 0, &[]);
    let expected = 0.8 * 1.25;
    assert!(
        (rel - expected).abs() < 1e-6,
        "expected {expected}, got {rel}"
    );
}

/// G5: Delegates correctly to inner pruner for non-fork tokens (scaffolding path).
#[test]
fn goat_g5_delegation_to_inner() {
    let mut pruner = make_pruner(0.7, 0.5, 0.3);
    pruner.update_entropy(0.8); // Scaffolding
    let rel = pruner.relevance(3, 42, &[1, 2, 3]);
    assert!(
        (rel - 0.7).abs() < 1e-6,
        "should delegate to inner, got {rel}"
    );
}

/// G6: EntropyState transitions correctly via update_entropy().
#[test]
fn goat_g6_entropy_state_transitions() {
    let mut pruner = make_pruner(1.0, 0.5, 0.3);

    // Default state
    assert_eq!(pruner.state(), EntropyState::Scaffolding);

    // Low top-1 → Forking
    pruner.update_entropy(0.1);
    assert_eq!(pruner.state(), EntropyState::Forking);

    // High top-1 → Scaffolding
    pruner.update_entropy(0.9);
    assert_eq!(pruner.state(), EntropyState::Scaffolding);

    // Back to low → Forking
    pruner.update_entropy(0.3);
    assert_eq!(pruner.state(), EntropyState::Forking);
}
