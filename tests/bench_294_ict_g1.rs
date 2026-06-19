//! Plan 294 Phase 2 — GOAT Gate G1: ICT distributional discrimination.
//!
//! Paper-proof that β (collision purity) distinguishes distributions where
//! H₁ (Shannon entropy) cannot. This is the *capability* proof — it always
//! passes because it's a mathematical fact (ICT §1.5, Figure 1a).
//!
//! ## The construction
//!
//! `p_A = [0.5, 0.5, 0, 0, 0, 0]` has H₁ = ln 2 ≈ 0.693 and β = 0.5.
//!
//! To construct `p_B` with **identical H₁** but **different β**, we use a
//! 3-outcome family `p_B(t) = (t, (1−t)/2, (1−t)/2, 0, 0, 0)` and solve
//! `H₁(p_B(t)) = ln 2` for `t ∈ (0, 1)`. H₁(t) is strictly decreasing in
//! `t` (concentration), `H₁(0.5) ≈ 1.04` and `H₁(0.9) ≈ 0.40`, so a unique
//! root exists near `t ≈ 0.77`. The resulting β ≈ 0.62, clearly ≠ 0.5.
//! This is the paper's Figure 1a bifurcation in textbook form: H₁ sees
//! identical "uncertainty", β sees concentration vs spread.
//!
//! ## Run
//!
//! ```text
//! cargo test --features ict_branching --test bench_294_ict_g1 -- --nocapture
//! ```

#![cfg(feature = "ict_branching")]

use katgpt_core::ict::{
    branching::is_critical_branching,
    math::{collision_purity, shannon_h1},
};

const TOL: f32 = 1e-3;

/// Solve `H₁(p_B(t)) = ln 2` for `t` where `p_B(t) = (t, (1−t)/2, (1−t)/2, 0, 0, 0)`.
/// Returns `t`. Uses bisection — H₁(t) is strictly decreasing in `t` over
/// `(0, 1)`, so the root is unique. The other 5 entries follow from `t`.
fn solve_equal_h1_three_outcome() -> f32 {
    let target = core::f32::consts::LN_2;
    let h1_of_t = |t: f32| -> f32 {
        let q = (1.0 - t) / 2.0;
        // H_1 = -t ln t - 2 q ln q (q appears twice).
        let mut h = 0.0_f32;
        if t > 0.0 {
            h -= t * t.ln();
        }
        if q > 0.0 {
            h -= 2.0 * q * q.ln();
        }
        h
    };
    // Bisect t ∈ (0.5, 0.95). h1_of_t(0.5) ≈ 1.04 > ln 2, h1_of_t(0.95) ≈ 0.23 < ln 2.
    let mut lo = 0.5_f32;
    let mut hi = 0.95_f32;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let h = h1_of_t(mid);
        if h > target {
            lo = mid; // need larger t to reduce H
        } else {
            hi = mid; // need smaller t to grow H
        }
    }
    0.5 * (lo + hi)
}

// ──────────────────────────────────────────────────────────────────────────
// G1 — Distributional discrimination (paper proof of capability)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn g1_distributional_discrimination() {
    // p_A: 50/50 split, H_1 = ln 2, β = 0.5.
    let p_a = [0.5_f32, 0.5, 0.0, 0.0, 0.0, 0.0];
    let beta_a = collision_purity(&p_a);
    let h1_a = shannon_h1(&p_a);

    // p_B: 3-outcome (t, (1−t)/2, (1−t)/2, 0, 0, 0) with same H_1.
    let t = solve_equal_h1_three_outcome();
    let q = (1.0 - t) / 2.0;
    let p_b = [t, q, q, 0.0, 0.0, 0.0];
    let beta_b = collision_purity(&p_b);
    let h1_b = shannon_h1(&p_b);

    println!("\n=== G1 — ICT distributional discrimination (paper Fig 1a) ===");
    println!("p_A = [0.5, 0.5, 0, 0, 0, 0]                H_1 = {h1_a:.6}, β = {beta_a:.6}");
    println!(
        "p_B = [{t:.4}, {q:.4}, {q:.4}, 0, 0, 0]   H_1 = {h1_b:.6}, β = {beta_b:.6}"
    );
    println!("ΔH_1 = {:.2e}   Δβ = {:.4}", (h1_a - h1_b).abs(), (beta_a - beta_b).abs());

    // (1) H_1 cannot distinguish — both have ln 2 to within tolerance.
    assert!(
        (h1_a - h1_b).abs() < TOL,
        "G1 FAIL: H_1 should match (both ln 2), got h1_a={h1_a}, h1_b={h1_b}, |Δ|={}",
        (h1_a - h1_b).abs()
    );

    // (2) β DOES distinguish — paper claim is that Σ π² captures concentration
    // that H_1 averages away.
    assert!(
        (beta_a - beta_b).abs() > 0.05,
        "G1 FAIL: β should differ by > 0.05 (paper claim), got beta_a={beta_a}, beta_b={beta_b}"
    );

    // (3) Sanity: p_B sums to 1.
    let sum_b: f32 = p_b.iter().sum();
    assert!((sum_b - 1.0).abs() < 1e-4, "p_B must sum to 1, got {sum_b}");

    println!(
        "G1 PASS: β distinguishes [{beta_a:.4} vs {beta_b:.4}] where H_1 cannot [{h1_a:.4} vs {h1_b:.4}]"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Bifurcation regimes for `is_critical_branching`
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn regime_h_collapse() {
    // Collapse regime: π(a*) = 0.9 > β = 0.82 → policy dominated by one
    // action → NOT a branching point. Returns false.
    let result = is_critical_branching(0.9, 0.82, 0.05);
    assert!(
        !result,
        "regime_h_collapse: π=0.9 > β=0.82 should NOT be critical (collapse), got {result}"
    );
    println!("regime_h_collapse PASS: π > β (collapse) → not critical");
}

#[test]
fn regime_l_explosion() {
    // Explosion regime: π(a*) = 0.05 < β = 0.5 → near-uniform noise → NOT
    // a branching point. Returns false.
    let result = is_critical_branching(0.05, 0.5, 0.05);
    assert!(
        !result,
        "regime_l_explosion: π=0.05 < β=0.5 should NOT be critical (explosion), got {result}"
    );
    println!("regime_l_explosion PASS: π < β (explosion) → not critical");
}

#[test]
fn critical_branching() {
    // Critical branching: π(a*) = 0.5 = β, η = 0.05 → |Δ| = 0 < η → TRUE.
    let result = is_critical_branching(0.5, 0.5, 0.05);
    assert!(
        result,
        "critical_branching: π=0.5 = β=0.5 within η=0.05 should be critical, got {result}"
    );
    println!("critical_branching PASS: |π − β| < η → critical (paper Theorem 3.1)");
}

// ──────────────────────────────────────────────────────────────────────────
// Sanity: paper-claimed monotonicity of β
// ──────────────────────────────────────────────────────────────────────────

/// ICT §A.2.5: ∂β/∂π(a) = 2π(a) > 0 unconditionally. Verify numerically —
/// concentrating any probability mass strictly increases β.
#[test]
fn beta_is_monotone_in_concentration() {
    let uniform = [0.25_f32; 4]; // β = 0.25
    let half_concentrated = [0.5_f32, 0.5 / 3.0, 0.5 / 3.0, 0.5 / 3.0];
    let fully_concentrated = [1.0_f32, 0.0, 0.0, 0.0]; // β = 1.0
    let b_u = collision_purity(&uniform);
    let b_h = collision_purity(&half_concentrated);
    let b_f = collision_purity(&fully_concentrated);
    assert!(b_u < b_h, "uniform β={b_u} < half β={b_h}");
    assert!(b_h < b_f, "half β={b_h} < full β={b_f}");
    println!("β monotone in concentration: uniform={b_u:.4} < half={b_h:.4} < full={b_f:.4}");
}
