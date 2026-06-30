//! Conformal bit-reproducibility test — GOAT gate G4 (Plan 340).
//!
//! Two `ConformalIntervalCalibrator` instances with identical
//! `(residual_pool, m, alpha, h, decay_config, orientation)` MUST produce
//! byte-identical `PredictiveInterval` bounds (verified via `f32::to_bits`).
//!
//! Required for quorum commitment downstream — if two nodes with the same
//! residual observations produce different interval bounds, the LatCal
//! sync-boundary story breaks.
//!
//! Varies `α ∈ {0.01, 0.05, 0.1, 0.2}` and `h ∈ {1, 8, 24}`.

use katgpt_core::{
    ConformalIntervalCalibrator, DecayUnit, PredictiveInterval, ResidualMode, SeasonalPoolForecaster,
};

/// Build a calibrator, push a deterministic residual stream, return it.
fn build_calibrator() -> ConformalIntervalCalibrator<SeasonalPoolForecaster> {
    let forecaster = SeasonalPoolForecaster::new(64, 12, 0.01, 0.5);
    let mut cal = ConformalIntervalCalibrator::new(
        forecaster,
        1,
        24, // max_h
        12, // m
        128, // capacity
        0.02, // exp_lambda
        DecayUnit::Step,
        ResidualMode::HStep,
        true, // orientation
    );
    // Push a deterministic stream of residuals. Use a simple LCG.
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    for t in 0..200 {
        // LCG step.
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (((state >> 33) as f32) / (1u64 << 31) as f32) * 2.0 - 1.0; // [-1, 1]
        let forecast = 0.0_f32; // forecast is constant for this reproducibility test
        cal.update_residual(r, forecast, 0, 1);
        // Step the tick every push so recency weights differ per residual.
        for _ in 0..t % 5 {
            cal.step();
        }
    }
    cal
}

#[test]
fn g4_identical_configs_produce_identical_bounds() {
    let mut a = build_calibrator();
    let mut b = build_calibrator();

    let mut iva = PredictiveInterval::new(0.0, 0.0, 0.0, 0.0);
    let mut ivb = PredictiveInterval::new(0.0, 0.0, 0.0, 0.0);
    for &alpha in &[0.01_f32, 0.05, 0.1, 0.2] {
        for &h in &[1_usize, 8, 24] {
            a.interval_into(0, h, alpha, &mut iva);
            b.interval_into(0, h, alpha, &mut ivb);
            assert_eq!(
                iva.lower.to_bits(),
                ivb.lower.to_bits(),
                "alpha={} h={} lower mismatch: {} vs {}",
                alpha,
                h,
                iva.lower,
                ivb.lower
            );
            assert_eq!(
                iva.point.to_bits(),
                ivb.point.to_bits(),
                "alpha={} h={} point mismatch: {} vs {}",
                alpha,
                h,
                iva.point,
                ivb.point
            );
            assert_eq!(
                iva.upper.to_bits(),
                ivb.upper.to_bits(),
                "alpha={} h={} upper mismatch: {} vs {}",
                alpha,
                h,
                iva.upper,
                ivb.upper
            );
            assert_eq!(iva.alpha.to_bits(), ivb.alpha.to_bits());
        }
    }
}

#[test]
fn g4_reproducible_across_pool_states() {
    // Two calibrators that receive the SAME residual sequence (in the same
    // order) but with different intermediate `interval_into` reads should
    // still produce identical bounds — `interval_into` is a pure read.
    let mk = || {
        let forecaster = SeasonalPoolForecaster::new(32, 1, 0.0, 0.0);
        let mut cal = ConformalIntervalCalibrator::new(
            forecaster, 1, 1, 1, 64, 0.0, DecayUnit::Step, ResidualMode::HStep, false,
        );
        for i in 0..50 {
            let r = (i as f32) * 0.1 - 2.5;
            cal.update_residual(r, 0.0, 0, 1);
        }
        cal
    };
    let mut a = mk();
    let mut b = mk();

    // Interleave reads on `a` between pushes (no, we already pushed all).
    // Instead: read from `a` many times — the bounds should be stable.
    let mut iv1 = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
    let mut iv2 = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
    for _ in 0..10 {
        a.interval_into(0, 1, 0.05, &mut iv1);
    }
    b.interval_into(0, 1, 0.05, &mut iv2);
    assert_eq!(iv1.lower.to_bits(), iv2.lower.to_bits());
    assert_eq!(iv1.upper.to_bits(), iv2.upper.to_bits());
}

#[test]
fn g4_sample_predictive_distribution_deterministic_with_seed() {
    let forecaster = SeasonalPoolForecaster::new(32, 1, 0.0, 0.0);
    let mut cal = ConformalIntervalCalibrator::new(
        forecaster, 1, 1, 1, 64, 0.0, DecayUnit::Step, ResidualMode::HStep, false,
    );
    for i in 0..30 {
        let r = (i as f32) * 0.2 - 3.0;
        cal.update_residual(r, 0.0, 0, 1);
    }
    let mut rng_a = fastrand::Rng::with_seed(42);
    let mut rng_b = fastrand::Rng::with_seed(42);
    let a = cal.sample_predictive_distribution(0, 1, 50, &mut rng_a);
    let b = cal.sample_predictive_distribution(0, 1, 50, &mut rng_b);
    assert_eq!(a.len(), b.len());
    for (xa, xb) in a.iter().zip(b.iter()) {
        assert_eq!(xa.to_bits(), xb.to_bits(), "sample mismatch");
    }
}
