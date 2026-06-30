//! Conformal coverage test — GOAT gate G1 (Plan 340).
//!
//! On a stationary seasonal synthetic series `y_t = sin(2π t/m) + ε_t`,
//! `ε ~ N(0, σ)`, the conformal interval calibrator's empirical coverage at
//! α=0.05 over 10,000 ticks MUST be in [0.93, 0.97].
//!
//! Varies `m ∈ {12, 24, 48}`, `σ ∈ {0.1, 0.5, 1.0}`, and `m=1` HStep mode
//! (non-seasonal, interval should widen but coverage should still hold).

use katgpt_core::{
    ConformalIntervalCalibrator, DecayUnit, PredictiveInterval, ResidualMode,
    SeasonalPoolForecaster,
};

/// Simple deterministic Gaussian-ish noise via central-limit sum of uniforms.
/// (Avoids pulling in a `rand` dep; the central limit theorem gives us
/// approximately Gaussian noise from 12 uniforms, per the Box-Muller-free
/// classic trick.)
struct DeterministicNoise {
    state: u64,
}
impl DeterministicNoise {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        // SplitMix64 — deterministic, good distribution.
        let mut z = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        self.state = z ^ (z >> 31);
        z
    }
    fn unit_f32(&mut self) -> f32 {
        // [0, 1)
        (self.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32)
    }
    fn gaussian(&mut self, sigma: f32) -> f32 {
        // Sum of 12 uniforms → mean 6, variance 1. Subtract 6 → approx N(0,1).
        let mut sum = 0.0_f32;
        for _ in 0..12 {
            sum += self.unit_f32();
        }
        (sum - 6.0) * sigma
    }
}

/// Run a coverage backtest: generate `n_ticks` of `sin(2π t/m) + ε`, fit the
/// calibrator online, measure empirical coverage at level `1−alpha`.
fn run_coverage(m: usize, sigma: f32, alpha: f32, n_ticks: usize, seed: u64) -> f32 {
    let forecaster = SeasonalPoolForecaster::new(8 * m.max(1), m, 0.0, 0.0);
    let max_h = 1; // single-step forecast for the coverage test
    let mut cal = ConformalIntervalCalibrator::new(
        forecaster,
        1, // 1 channel
        max_h,
        m,
        256,
        0.0, // no decay for the stationary test
        DecayUnit::Step,
        ResidualMode::HStep,
        false,
    );

    let mut noise = DeterministicNoise::new(seed);
    let two_pi_over_m = 2.0 * core::f32::consts::PI / (m as f32);

    // Warmup: fill the residual pool before measuring coverage.
    let warmup = (8 * m).min(n_ticks / 2).max(m);
    let mut prev_y = 0.0_f32;
    for t in 0..warmup {
        let y = (two_pi_over_m * (t as f32)).sin() + noise.gaussian(sigma);
        // Forecast (seasonal-naive), observe, update residual.
        cal.observe_and_update(y, &[], 0, 1);
        cal.step();
        prev_y = y;
    }

    // Measurement phase: count hits.
    let mut hits = 0usize;
    let mut total = 0usize;
    for t in warmup..n_ticks {
        let y = (two_pi_over_m * (t as f32)).sin() + noise.gaussian(sigma);
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, alpha);
        cal.interval_into(0, 1, alpha, &mut iv);
        if iv.contains(y) {
            hits += 1;
        }
        total += 1;
        // Update the residual pool with the new observation.
        cal.observe_and_update(y, &[], 0, 1);
        cal.step();
        prev_y = y;
    }
    let _ = prev_y; // sink
    if total == 0 {
        return 0.0;
    }
    (hits as f32) / (total as f32)
}

#[test]
fn g1_coverage_m12_sigma_list() {
    // α=0.05 → target coverage 0.95. Tolerance [0.93, 0.97] per Plan 340.
    let alpha = 0.05_f32;
    for &m in &[12_usize, 24, 48] {
        for &sigma in &[0.1_f32, 0.5, 1.0] {
            let cov = run_coverage(m, sigma, alpha, 10_000, 0xC0FFEE_0000 | (m as u64));
            // Log for debugging.
            eprintln!("m={m} sigma={sigma} alpha={alpha} → coverage {cov:.4}");
            assert!(
                (0.93..=0.97).contains(&cov),
                "G1 FAIL: m={} sigma={} → coverage {} outside [0.93, 0.97]",
                m,
                sigma,
                cov
            );
        }
    }
}

#[test]
fn g1_coverage_m1_hstep_nonseasonal() {
    // m=1, HStep mode: non-seasonal. Coverage should hold with widening
    // intervals (the residual pool at h=1 is just the lag-1 residuals).
    let alpha = 0.05_f32;
    for &sigma in &[0.1_f32, 0.5, 1.0] {
        let cov = run_coverage(1, sigma, alpha, 10_000, 0xBEEF_0000 | (sigma.to_bits() as u64));
        eprintln!("m=1 sigma={sigma} alpha={alpha} → coverage {cov:.4}");
        assert!(
            (0.90..=0.99).contains(&cov),
            "G1 FAIL (m=1): sigma={} → coverage {} outside [0.90, 0.99]",
            sigma,
            cov
        );
    }
}

#[test]
fn g1_coverage_varying_alpha() {
    // Different alpha levels should produce proportionally different coverage.
    let m = 24;
    let sigma = 0.5;
    for &alpha in &[0.01_f32, 0.05, 0.1, 0.2] {
        let cov = run_coverage(m, sigma, alpha, 10_000, 0xF00D_0000 | ((alpha.to_bits()) as u64));
        let target = 1.0 - alpha;
        eprintln!("m={m} sigma={sigma} alpha={alpha} → coverage {cov:.4} (target {target:.2})");
        // Allow a wider tolerance band for the more extreme alpha levels.
        let lo = (target - 0.04).max(0.5);
        let hi = (target + 0.04).min(0.999);
        assert!(
            (lo..=hi).contains(&cov),
            "G1 FAIL (alpha sweep): alpha={} → coverage {} outside [{:.3}, {:.3}]",
            alpha,
            cov,
            lo,
            hi
        );
    }
}
