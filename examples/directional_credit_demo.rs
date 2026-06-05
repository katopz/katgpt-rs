//! Direction-Adaptive Credit — Entropy-Bifurcated Pruning Demo
//!
//! Shows how entropy-bifurcated screening treats high-entropy (forking) tokens
//! differently from low-entropy (scaffolding) tokens.
//!
//! Run: `cargo run --example directional_credit_demo --features directional_credit`
//!
//! Demonstrates:
//! - EntropyBifurcatedPruner wrapping a base pruner
//! - SelfDrivenTokenTracker for RLRT exploration signals
//! - Before/after: uniform vs entropy-bifurcated screening

#[cfg(feature = "directional_credit")]
use katgpt_rs::pruners::{EntropyBifurcatedPruner, SelfDrivenTokenTracker};
#[cfg(feature = "directional_credit")]
use katgpt_rs::speculative::types::ScreeningPruner;

// A simple fixed relevance pruner for demo purposes.
#[cfg(feature = "directional_credit")]
#[derive(Debug, Clone)]
struct FixedRelevancePruner {
    base: f32,
}

#[cfg(feature = "directional_credit")]
impl ScreeningPruner for FixedRelevancePruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.base
    }
}

#[cfg(feature = "directional_credit")]
fn main() {
    println!("=== Direction-Adaptive Credit Demo ===\n");

    let base_relevance = 0.5f32;
    let threshold = 0.5f32;
    let relax = 0.3f32;

    let inner = FixedRelevancePruner {
        base: base_relevance,
    };
    let mut pruner = EntropyBifurcatedPruner::new(inner, threshold, relax);

    println!("Base relevance: {base_relevance}");
    println!("Top-1 threshold: {threshold}");
    println!("Relax factor: {relax}\n");

    // Simulate token positions with different entropy levels
    let top1_probs = [0.9, 0.1, 0.8, 0.15, 0.95];
    println!("--- Entropy-Bifurcated Screening ---");
    for (i, &top1) in top1_probs.iter().enumerate() {
        pruner.update_entropy(top1);
        let rel = pruner.relevance(i, 0, &[]);
        let state = pruner.state();
        println!("Position {i}: top1={top1:.2} → {state:?} → relevance={rel:.3}");
    }

    // SelfDrivenTokenTracker demo
    println!("\n--- Self-Driven Token Tracker ---");
    let mut tracker = SelfDrivenTokenTracker::new(5, 0.2);

    // Record parent's top-1 choices
    let parent_choices = [3, 7, 1, 5, 2];
    let child_choices = [3, 4, 1, 8, 2]; // positions 1 and 3 diverge

    for (d, &choice) in parent_choices.iter().enumerate() {
        tracker.record_parent(d, choice);
    }

    for (d, &choice) in child_choices.iter().enumerate() {
        let driven = tracker.check_self_driven(d, choice);
        let bonus = tracker.bonus(d);
        println!(
            "Depth {d}: parent={}->{choice} | self_driven={driven} | bonus={bonus:.2}",
            parent_choices[d]
        );
    }

    println!("\n=== Demo Complete ===");
    println!("Key insight: High-entropy (low top-1) tokens get relaxed screening,");
    println!("enabling more exploration at forking points while keeping scaffolding tight.");
}

#[cfg(not(feature = "directional_credit"))]
fn main() {
    eprintln!("This example requires the `directional_credit` feature.");
    eprintln!("Run: cargo run --example directional_credit_demo --features directional_credit");
}
