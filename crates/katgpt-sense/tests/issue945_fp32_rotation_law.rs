#![cfg(feature = "phase_belief")]
//! Issue 945 P5 — the fp32-rotation law + reduced-precision canary
//! (Research 623, arXiv:2610.18174 numerics law): a rotation must be
//! computed in f32 REGARDLESS of storage dtype. At bf16 the per-step
//! rotation stops being orthogonal — `cos²+sin² = 1 ± 2⁻⁸` — and the state
//! norm random-walks ~13% over 1000 steps. [`DampedPhaseBlock`] is the
//! rotation-carrying primitive today; its `step` is f32-internal by
//! construction (precomputed f32 `cos θ`/`sin θ` folded with single-rounding
//! `mul_add`).
//!
//! Two arms, and the second is the loud-zero defense:
//! 1. **Law arm** — the SHIPPED f32 primitive over 1000 free steps drifts
//!    under a small relative bound.
//! 2. **Canary arm** — a reduced-precision emulation (bf16-rounded rotation
//!    and state, RNE on the low 16 bits) reproduces the paper's ~13% drift.
//!    If the canary CANNOT reproduce the drift, the emulation is wrong; if
//!    the law arm ever shows canary-class drift, the law is BROKEN. Either
//!    failure reds this file.

use katgpt_sense::phase_belief::DampedPhaseBlock;

/// IEEE-754 round-to-nearest-even truncation of an f32 to bf16 precision
/// (round the low 16 mantissa/exponent bits away). The test-only emulation
/// of "reduced-precision rotation".
fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// One rotation step entirely in reduced precision: state AND rotation
/// constants AND every product/sum rounded — the law-violating counterfactual.
fn bf16_step(z: [f32; 2], rho: f32, cos_t: f32, sin_t: f32) -> [f32; 2] {
    let (c, s, r) = (bf16_round(cos_t), bf16_round(sin_t), bf16_round(rho));
    let rc = bf16_round(r * c);
    let rs = bf16_round(r * s);
    let x = bf16_round(z[0]);
    let y = bf16_round(z[1]);
    [
        bf16_round(bf16_round(rc * x) - bf16_round(rs * y)),
        bf16_round(bf16_round(rs * x) + bf16_round(rc * y)),
    ]
}

fn norm(z: [f32; 2]) -> f32 {
    (z[0] * z[0] + z[1] * z[1]).sqrt()
}

const RHO: f32 = 0.999;
const THETA: f32 = 0.13;
const STEPS: u32 = 1000;

/// Law arm: the shipped f32 primitive stays within a small relative drift of
/// the closed-form `ρⁿ` over 1000 steps.
#[test]
fn f32_rotation_norm_drift_stays_bounded_over_1000_steps() {
    let mut block = DampedPhaseBlock::new(RHO, THETA).expect("valid ρ, θ");
    block.step([1.0, 0.0]);
    for _ in 1..STEPS {
        block.step([0.0, 0.0]);
    }
    let final_norm = block.magnitude();
    let expected = RHO.powi(STEPS as i32);
    let drift = (final_norm - expected) / expected;
    println!("qrr p5 f32 law arm: relative drift {drift:.3e} over {STEPS} steps");
    assert!(
        drift.abs() < 0.02,
        "f32 rotation drifted {drift:.3e} — beyond the 2% law bound"
    );
}

/// Canary arm: the SAME recurrence in reduced precision reproduces the
/// paper's ~13%-class random walk — proving the failure mode the f32 law
/// exists to prevent is real (the loud-zero defense for the law arm).
#[test]
fn reduced_precision_rotation_reproduces_the_paper_drift() {
    let mut z = [1.0_f32, 0.0_f32];
    for _ in 0..STEPS {
        z = bf16_step(z, RHO, THETA.cos(), THETA.sin());
    }
    let expected = RHO.powi(STEPS as i32);
    let drift = (norm(z) - expected) / expected;
    println!("qrr p5 bf16 canary: relative drift {drift:.3e} over {STEPS} steps");
    assert!(
        drift.abs() > 0.05,
        "the reduced-precision canary drifted only {drift:.3e} — the emulation no longer \
         reproduces the paper's failure mode; the law arm's 2% bound is unproven"
    );
}

/// The contrast, explicitly: the f32 arm must drift orders less than the
/// canary — the law and its counterfactual must never approach each other.
#[test]
fn f32_arm_drifts_far_less_than_the_reduced_precision_arm() {
    let measure_f32 = |steps: u32| {
        let mut block = DampedPhaseBlock::new(RHO, THETA).expect("valid ρ, θ");
        block.step([1.0, 0.0]);
        for _ in 1..steps {
            block.step([0.0, 0.0]);
        }
        (block.magnitude() - RHO.powi(steps as i32)).abs()
    };
    let measure_bf16 = |steps: u32| {
        let mut z = [1.0_f32, 0.0_f32];
        for _ in 0..steps {
            z = bf16_step(z, RHO, THETA.cos(), THETA.sin());
        }
        (norm(z) - RHO.powi(steps as i32)).abs()
    };
    let f32_drift = measure_f32(STEPS);
    let bf16_drift = measure_bf16(STEPS);
    println!("qrr p5 contrast: f32 {f32_drift:.3e} vs bf16 {bf16_drift:.3e}");
    assert!(
        bf16_drift > 5.0 * f32_drift.max(f32::EPSILON),
        "the two arms must not approach: f32 {f32_drift:.3e}, bf16 {bf16_drift:.3e}"
    );
}
