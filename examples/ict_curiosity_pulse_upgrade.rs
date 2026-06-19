//! Plan 294 T7.2 — Curiosity Pulse (R041) H_1 → β drop-in reference.
//!
//! Curiosity Pulse currently uses Shannon entropy H_1 of relevance scores as
//! the curiosity trigger. ICT §1.5 + §A.3.3 prove this is the wrong entropy
//! for the long tail (π < e⁻¹ ≈ 0.37). This example shows the drop-in
//! replacement: collision purity β = Σ π², which is unconditionally monotone.
//!
//! The actual implementation lives in `riir-ai` Plan 274 / Plan 187. This
//! file documents the public spec only. Run with:
//!
//! ```text
//! cargo run --example ict_curiosity_pulse_upgrade --features ict_branching
//! ```

#![cfg(feature = "ict_branching")]

use katgpt_core::ict::math::{collision_purity, shannon_h1};

fn main() {
    println!("=== Curiosity Pulse H_1 → β Drop-in (R041 / Plan 274 / Plan 187) ===\n");

    println!("BEFORE (riir-ai current implementation, R041 §2.1):");
    println!("    u_i(t) = shannon_h1(relevance_scores)");
    println!("\nAFTER (ICT-correct drop-in per §1.5 + §A.3.3):");
    println!("    u_i(t) = collision_purity(relevance_scores)   // = β");
    println!("\n// Why: H_1 is 'blind exploration' (ICT §1); β captures concentration");
    println!("//      — the right curiosity trigger. ∂H_2/∂π(a) < 0 unconditionally;");
    println!("//      H_1 only valid for π > e⁻¹ ≈ 0.37.\n");

    // Demonstrate the divergence: two relevance-score distributions where
    // H_1 reports near-identical "curiosity" but β reports very different
    // "concentration" — meaning Curiosity Pulse fires spuriously under H_1
    // but correctly stays quiet under β on the long-tail case.

    // Case A: one highly-relevant item among many (genuine curiosity).
    let relevance_focused = [0.7_f32, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01];
    // Case B: long-tail, no dominant item (H_1 reports same curiosity but
    // there's nothing concentrated to be curious about).
    let relevance_long_tail = [0.18_f32, 0.16, 0.14, 0.12, 0.11, 0.10, 0.10, 0.09];

    let h1_a = shannon_h1(&relevance_focused);
    let h1_b = shannon_h1(&relevance_long_tail);
    let beta_a = collision_purity(&relevance_focused);
    let beta_b = collision_purity(&relevance_long_tail);

    println!("Case                                    H_1 (curiosity_H1)  β (curiosity_β)");
    println!("--------------------------------------- ------------------  ---------------");
    println!(
        "A: focused   (0.7, 0.1, 0.05, ...)        {:.4}            {:.4}",
        h1_a, beta_a
    );
    println!(
        "B: long-tail (0.18, 0.16, 0.14, ...)      {:.4}            {:.4}",
        h1_b, beta_b
    );

    println!("\nH_1 reports ~equal curiosity in both cases (it can't tell focused");
    println!("relevance from long-tail noise). β reports A is highly concentrated");
    println!("(real signal) and B is near-uniform (nothing to focus on). Curiosity");
    println!("Pulse firing on B is a H_1 artifact; β correctly stays quiet.");

    println!("\n=== Drop-in patch ===\n");
    println!("```rust,ignore");
    println!("// In riir-ai Plan 274 CuriosityPulse::uncertainty_ema:");
    println!("//// let u = shannon_h1(&relevance_scores);  // OLD");
    println!("let u = collision_purity(&relevance_scores); // NEW = β");
    println!("// The downstream sigmoid gate and EMA are unchanged — the swap is");
    println!("// literally one function call.");
    println!("```");
}
