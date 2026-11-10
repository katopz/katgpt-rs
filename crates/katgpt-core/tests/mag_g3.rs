//! MAG G3 — reconstruction error sanity gate (Plan 418 Phase 2 T2.3).
//!
//! Verifies the ϵ_Q diagnostic behaves correctly:
//! - Perfectly linear shift → ϵ_Q ≈ 0.0
//! - Zero shift → ϵ_Q = 1.0 (by convention)
//! - Overshoot (predicted shift > actual shift) → ϵ_Q > 1.0

use katgpt_core::mag::{mine_direction, reconstruction_error};

const D: usize = 64;

#[inline]
fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

#[test]
fn g3_linear_shift_recon_error_near_zero() {
    let mut rng = fastrand::Rng::with_seed(0xA300_0001);

    // Unit-norm shift direction.
    let mut v = [0.0_f32; D];
    let inv_sqrt_d = 1.0 / (D as f32).sqrt();
    for x in v.iter_mut() {
        *x = inv_sqrt_d;
    }
    let shift_scale = 2.0;
    let alpha = shift_scale; // alpha·direction = shift_scale · v = the actual shift

    let n = 100;
    let mut with = vec![[0.0_f32; D]; n];
    let mut without = vec![[0.0_f32; D]; n];
    for i in 0..n {
        for j in 0..D {
            without[i][j] = gaussian(&mut rng);
            with[i][j] = without[i][j] + shift_scale * v[j];
        }
    }

    let (recon, cos) = reconstruction_error(&with, &without, &v, alpha).unwrap();

    println!("G3 linear shift: ϵ_Q = {recon:.8}, cos = {cos:.6}");
    assert!(
        recon < 1e-5,
        "G3 FAIL: linear shift should give ϵ_Q ≈ 0, got {recon:.8}"
    );
    assert!(
        cos > 0.9999,
        "G3 FAIL: linear shift cos should be ≈ 1.0, got {cos:.6}"
    );
}

#[test]
fn g3_zero_shift_recon_error_is_one() {
    let mut rng = fastrand::Rng::with_seed(0xA300_0002);

    let v = {
        let mut v = [0.0_f32; D];
        v[0] = 1.0;
        v
    };
    let alpha = 1.0;

    let n = 100;
    let mut data = vec![[0.0_f32; D]; n];
    for row in data.iter_mut() {
        for val in row.iter_mut() {
            *val = gaussian(&mut rng);
        }
    }
    // with == without (zero shift).
    let with = data.clone();

    let (recon, _cos) = reconstruction_error(&with, &data, &v, alpha).unwrap();

    println!("G3 zero shift: ϵ_Q = {recon:.6} (expected 1.0 by convention)");
    assert!(
        (recon - 1.0).abs() < 1e-6,
        "G3 FAIL: zero shift should give ϵ_Q = 1.0 by convention, got {recon:.6}"
    );
}

#[test]
fn g3_overshoot_recon_error_gt_one() {
    let mut rng = fastrand::Rng::with_seed(0xA300_0003);

    // Unit-norm shift direction.
    let mut v = [0.0_f32; D];
    let inv_sqrt_d = 1.0 / (D as f32).sqrt();
    for x in v.iter_mut() {
        *x = inv_sqrt_d;
    }
    let shift_scale = 1.0;
    // Overshoot: actual shift = v, predicted = alpha · v = 3v.
    let alpha = 3.0;

    let n = 100;
    let mut with = vec![[0.0_f32; D]; n];
    let mut without = vec![[0.0_f32; D]; n];
    for i in 0..n {
        for j in 0..D {
            without[i][j] = gaussian(&mut rng);
            with[i][j] = without[i][j] + shift_scale * v[j];
        }
    }

    let (recon, cos) = reconstruction_error(&with, &without, &v, alpha).unwrap();

    println!("G3 overshoot (α=3×): ϵ_Q = {recon:.6}, cos = {cos:.6}");
    assert!(
        recon > 1.0,
        "G3 FAIL: overshoot should give ϵ_Q > 1.0, got {recon:.6}"
    );
}

#[test]
fn g3_mine_then_recon_roundtrip() {
    // Bonus: mine the direction from the same data, then check ϵ_Q is low.
    // This tests that mine_direction + reconstruction_error are consistent.
    let mut rng = fastrand::Rng::with_seed(0xA300_0004);

    let mut v = [0.0_f32; D];
    let inv_sqrt_d = 1.0 / (D as f32).sqrt();
    for x in v.iter_mut() {
        *x = inv_sqrt_d;
    }
    let shift_scale = 2.0;

    let n = 100;
    let mut with = vec![[0.0_f32; D]; n];
    let mut without = vec![[0.0_f32; D]; n];
    for i in 0..n {
        for j in 0..D {
            without[i][j] = gaussian(&mut rng);
            with[i][j] = without[i][j] + shift_scale * v[j];
        }
    }

    let dir = mine_direction(&with, &without).unwrap();
    // The mined direction is unit-norm. The actual shift is shift_scale * v.
    // calibrate_alpha would give alpha = shift_scale (for unit-norm dir).
    let (recon, cos) = reconstruction_error(&with, &without, dir.as_slice(), shift_scale).unwrap();

    println!("G3 roundtrip: ϵ_Q = {recon:.8}, cos = {cos:.6} (mined direction, calibrated alpha)");
    assert!(
        recon < 1e-4,
        "G3 FAIL: mined direction with calibrated alpha should give ϵ_Q ≈ 0, got {recon:.8}"
    );
}
