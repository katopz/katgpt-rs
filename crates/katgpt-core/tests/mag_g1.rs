//! MAG G1 — mining correctness gate (Plan 418 Phase 2 T2.1).
//!
//! Verifies that `mine_direction` recovers a known mean-shift to cos ≥ 0.99,
//! and `mine_contrast_direction` recovers a known inter-class contrast to
//! cos ≥ 0.95, on synthetic data in ℝ^64.

use katgpt_core::mag::{mine_contrast_direction, mine_direction};

const D: usize = 64;

/// Box-Muller Gaussian (deterministic given the rng state).
#[inline]
fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Generate a d-dim sample from N(center, sigma²·I).
fn sample_gaussian(rng: &mut fastrand::Rng, center: &[f32], sigma: f32, out: &mut [f32]) {
    for j in 0..D {
        out[j] = center[j] + sigma * gaussian(rng);
    }
}

#[test]
fn g1_mine_direction_recovers_known_shift() {
    let mut rng = fastrand::Rng::with_seed(0xA100_1000);

    // Known unit-norm shift direction: all dims equal.
    let mut v = [0.0_f32; D];
    let inv_sqrt_d = 1.0 / (D as f32).sqrt();
    for x in v.iter_mut() {
        *x = inv_sqrt_d;
    }
    let shift_scale = 2.0;

    let n = 100;
    let mut with = vec![[0.0_f32; D]; n];
    let mut without = vec![[0.0_f32; D]; n];

    let zero = [0.0_f32; D];
    for i in 0..n {
        sample_gaussian(&mut rng, &zero, 1.0, &mut without[i]);
        for j in 0..D {
            with[i][j] = without[i][j] + shift_scale * v[j];
        }
    }

    let dir = mine_direction(&with, &without).expect("mine_direction should succeed");

    // Cosine of recovered direction vs true direction.
    let mut dot = 0.0;
    for (j, vj) in v.iter().enumerate() {
        dot += dir.as_slice()[j] * vj;
    }
    // Both are unit-norm, so cosine = dot.
    let cos = dot; // dir is unit-norm, v is unit-norm
    assert!(
        cos >= 0.99,
        "G1 FAIL: mine_direction recovered direction with cos = {cos:.6} < 0.99"
    );
    println!("G1 mine_direction: cos = {cos:.6} (≥ 0.99) ✓");
}

#[test]
fn g1_mine_contrast_recovers_known_class_direction() {
    let mut rng = fastrand::Rng::with_seed(0xA100_1001);

    // Two clusters in ℝ^64. μ₁ = [+2, 0, ...], μ₂ = [−2, 0, ...].
    // Inter-class direction = μ₁ − μ₂ = [+4, 0, ...] → normalized = [+1, 0, ...].
    let mut mu1 = [0.0_f32; D];
    let mut mu2 = [0.0_f32; D];
    mu1[0] = 2.0;
    mu2[0] = -2.0;
    let sigma = 1.0;

    let n_per = 200;
    let mut positive = vec![[0.0_f32; D]; n_per]; // class 2 (μ₂)
    let mut negative = vec![[0.0_f32; D]; n_per]; // class 1 (μ₁)

    for i in 0..n_per {
        sample_gaussian(&mut rng, &mu1, sigma, &mut negative[i]);
        sample_gaussian(&mut rng, &mu2, sigma, &mut positive[i]);
    }

    let dir = mine_contrast_direction(&positive, &negative).expect("mine_contrast_direction");

    // Expected: mean(negative) − mean(positive) ≈ μ₁ − μ₂ = [+4, 0, ...]
    // → normalized = [+1, 0, ...]
    // So cosine with [1, 0, 0, ...] should be ≥ 0.95.
    let expected = {
        let mut e = [0.0_f32; D];
        e[0] = 1.0;
        e
    };
    let mut dot = 0.0;
    for (j, e) in expected.iter().enumerate() {
        dot += dir.as_slice()[j] * e;
    }
    assert!(
        dot >= 0.95,
        "G1 FAIL: mine_contrast_direction recovered direction with cos = {dot:.6} < 0.95"
    );
    println!("G1 mine_contrast_direction: cos = {dot:.6} (≥ 0.95) ✓");
}
