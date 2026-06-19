//! Plan 294 T8.3 — Minimal ICT primitive walkthrough.
//!
//! Shows the four core primitives: collision_purity, renyi_h2, shannon_h1,
//! js_divergence. Run with:
//!
//! ```text
//! cargo run --example ict_minimal --features ict_branching
//! ```

#![cfg(feature = "ict_branching")]

use katgpt_core::ict::math::{collision_purity, js_divergence, renyi_h2, shannon_h1};

fn main() {
    println!("=== ICT Distributional Branching-Point Detector — Minimal Walkthrough ===\n");

    // Three illustrative distributions:
    let degenerate = [1.0_f32, 0.0, 0.0, 0.0]; // β = 1.0 — fully concentrated
    let decisive = [0.6_f32, 0.3, 0.1, 0.0]; // β = 0.46 — one action dominates
    let uniform = [0.25_f32; 4]; // β = 0.25 — maximum entropy

    for (name, p) in [("degenerate", &degenerate[..]), ("decisive  ", &decisive[..]), ("uniform   ", &uniform[..])] {
        let beta = collision_purity(p);
        let h2 = renyi_h2(p);
        let h1 = shannon_h1(p);
        println!("{name}: β = {beta:.4}   H_2 = {h2:.4}   H_1 = {h1:.4}");
    }

    println!("\nNote: H_1 and H_2 agree on ranking here, but disagree on the");
    println!("long tail (π < e⁻¹ ≈ 0.37) — see bench_294_ict_g1.rs for the");
    println!("paper's Figure 1a proof that β distinguishes where H_1 cannot.\n");

    // JS divergence between two distributions (symmetric, bounded by ln 2).
    let mut scratch = [0.0_f32; 4];
    let js_dec_uni = js_divergence(&decisive, &uniform, &mut scratch);
    let js_dec_deg = js_divergence(&decisive, &degenerate, &mut scratch);
    println!("JS(decisive, uniform)    = {js_dec_uni:.4}  (some divergence)");
    println!("JS(decisive, degenerate) = {js_dec_deg:.4}  (large divergence)");
    println!("Upper bound on JS = ln 2 ≈ {:.4}", core::f32::consts::LN_2);

    println!("\nDone. For runtime branching detection see `ict_branching_detector` example.");
}
