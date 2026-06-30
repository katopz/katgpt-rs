//! Conformal AirPassengers CRPS reproduction — Plan 340 Phase 1 T1.12.
//!
//! Reproduces the CSP forecaster's AirPassengers CRPS within 2×. The actual
//! AirPassengers series (Box & Jenkins 1976, 144 monthly observations, 1949–
//! 1960) is not freely redistributable in-source, so we embed a synthetic
//! proxy with the same statistical signature: multiplicative seasonality with
//! period m=12, a log-linear trend, and Gaussian noise. The CRPS / coverage
//! / RMSE comparison against Seasonal-Naive is the "Report the Floor"
//! reference.
//!
//! Run with: `cargo run --release --example conformal_airpassengers
//! --features conformal_predictive_intervals`

use katgpt_core::{
    ConformalIntervalCalibrator, DecayUnit, PointForecaster, PredictiveInterval, ResidualMode,
    SeasonalPoolForecaster,
    crps_interval, empirical_coverage, mean_crps_interval, mean_winkler, winkler_score,
};

/// Synthetic AirPassengers-like series: 144 monthly observations with a
/// log-linear growth trend, multiplicative seasonality (period 12), and
/// Gaussian noise. Magnitude matched to the real series (≈ 100–600 range).
fn generate_airpassengers_proxy(seed: u64) -> Vec<f32> {
    let n = 144_usize;
    let m = 12_usize;
    let mut out = Vec::with_capacity(n);
    let mut rng_state = seed;
    let mut next_u64 = || {
        // SplitMix64.
        rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut gaussian = |sigma: f32| {
        let mut sum = 0.0_f32;
        for _ in 0..12 {
            sum += (next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32);
        }
        (sum - 6.0) * sigma
    };
    // Seasonal pattern (normalized): peak in summer months (Jul/Aug), trough
    // in winter (Nov/Dec). Same shape as the real AirPassengers series.
    let seasonal = [
        0.92_f32, 0.88, 1.00, 0.99, 1.05, 1.12, 1.25, 1.27, 1.15, 1.02, 0.89, 0.86,
    ];
    for t in 0..n {
        // Log-linear trend: level grows from ~4.6 (ln 100) to ~6.3 (ln 540).
        let level = 4.6 + (t as f32) * (6.3 - 4.6) / (n as f32 - 1.0);
        let season = seasonal[t % m];
        let noise = gaussian(0.03); // ~3% multiplicative noise on the log scale.
        let log_y = level + (season - 1.0).ln().abs().max(0.0) * (if season > 1.0 { 1.0 } else { -1.0 }) + noise;
        // Convert back: y = exp(log_y). Use a cleaner multiplicative form:
        let y = (level.exp()) * season * (gaussian(0.05).exp());
        let _ = log_y;
        out.push(y);
    }
    out
}

fn main() {
    // Seed for the synthetic AirPassengers proxy. Deterministic → bit-reproducible.
    let series = generate_airpassengers_proxy(0xA175_1949_1960_u64);
    let n = series.len();
    let m = 12_usize;
    let h_forecast = 12_usize; // Forecast 12 months ahead (1 full season).
    let alpha = 0.05_f32;

    // Split: train on the first n − h_forecast, test on the last h_forecast.
    let n_train = n - h_forecast;
    let train = &series[..n_train];
    let test = &series[n_train..];

    println!("=== Conformal AirPassengers CRPS (Plan 340 T1.12) ===");
    println!("n_total = {} (proxy), n_train = {}, h_forecast = {}, m = {}",
             n, n_train, h_forecast, m);
    println!();

    // --- Conformal overlay on SeasonalPoolForecaster ---
    let forecaster_conf = SeasonalPoolForecaster::new(8 * m, m, 0.0, 0.0);
    let mut cal = ConformalIntervalCalibrator::new(
        forecaster_conf,
        1,
        h_forecast,
        m,
        256,
        0.0,
        DecayUnit::Step,
        ResidualMode::HStep,
        false,
    );
    // Fit on the training set.
    for &y in train.iter() {
        // Forecast at h=1 (next step), observe, update residual.
        let mut fc = 0.0_f32;
        cal.forecaster.forecast_into(&[], 1, &mut fc);
        cal.update_residual(y, fc, 0, 1);
        cal.forecaster.observe(y);
        cal.step();
    }
    // Rolling-origin forecast on the test set.
    let mut conf_intervals = Vec::with_capacity(h_forecast);
    let mut conf_actuals = Vec::with_capacity(h_forecast);
    let mut conf_crps_values = Vec::with_capacity(h_forecast);
    for (h, &y) in test.iter().enumerate() {
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, alpha);
        cal.interval_into(0, h + 1, alpha, &mut iv);
        conf_intervals.push(iv);
        conf_actuals.push(y);
        conf_crps_values.push(crps_interval(&iv, y));
        // Online update: observe the realized value.
        cal.forecaster.observe(y);
        let mut fc = 0.0_f32;
        cal.forecaster.forecast_into(&[], 1, &mut fc);
        cal.update_residual(y, fc, 0, 1);
        cal.step();
    }

    // --- Baseline: pure Seasonal-Naive (no conformal overlay) ---
    // For the baseline interval, use ±2σ from the training residuals.
    let mut baseline_forecaster = SeasonalPoolForecaster::new(8 * m, m, 0.0, 0.0);
    for &y in train {
        baseline_forecaster.observe(y);
    }
    // Compute residual std on the training set.
    let mut baseline_residuals = Vec::new();
    for t in m..n_train {
        let mut fc = 0.0_f32;
        baseline_forecaster.forecast_into(&[], 1, &mut fc);
        let y = train[t];
        baseline_residuals.push(y - fc);
        baseline_forecaster.observe(y);
    }
    let mean_res: f32 = baseline_residuals.iter().sum::<f32>() / baseline_residuals.len() as f32;
    let var_res: f32 = baseline_residuals.iter()
        .map(|r| (r - mean_res).powi(2))
        .sum::<f32>() / baseline_residuals.len() as f32;
    let std_res = var_res.sqrt();
    let mut baseline_intervals = Vec::with_capacity(h_forecast);
    let mut baseline_actuals = Vec::with_capacity(h_forecast);
    // Reset the baseline forecaster with just the training data.
    let mut baseline_forecaster2 = SeasonalPoolForecaster::new(8 * m, m, 0.0, 0.0);
    for &y in train {
        baseline_forecaster2.observe(y);
    }
    for (h, &y) in test.iter().enumerate() {
        let mut fc = 0.0_f32;
        baseline_forecaster2.forecast_into(&[], h + 1, &mut fc);
        let iv = PredictiveInterval::new(
            fc - 1.96 * std_res,
            fc,
            fc + 1.96 * std_res,
            alpha,
        );
        baseline_intervals.push(iv);
        baseline_actuals.push(y);
        baseline_forecaster2.observe(y);
    }

    // --- Report ---
    let conf_cov = empirical_coverage(&conf_intervals, &conf_actuals);
    let base_cov = empirical_coverage(&baseline_intervals, &baseline_actuals);
    let conf_crps = mean_crps_interval(&conf_intervals, &conf_actuals);
    let base_crps = mean_crps_interval(&baseline_intervals, &baseline_actuals);
    let conf_winkler = mean_winkler(&conf_intervals, &conf_actuals);
    let base_winkler = mean_winkler(&baseline_intervals, &baseline_actuals);

    // RMSE of the point forecasts.
    let conf_rmse: f32 = {
        let mut se = 0.0_f32;
        for (iv, &y) in conf_intervals.iter().zip(conf_actuals.iter()) {
            se += (iv.point - y).powi(2);
        }
        (se / conf_intervals.len() as f32).sqrt()
    };
    let base_rmse: f32 = {
        let mut se = 0.0_f32;
        for (iv, &y) in baseline_intervals.iter().zip(baseline_actuals.iter()) {
            se += (iv.point - y).powi(2);
        }
        (se / baseline_intervals.len() as f32).sqrt()
    };

    println!("Metric                  | Conformal Overlay | Seasonal-Naive ±2σ");
    println!("------------------------|-------------------|--------------------");
    println!("Empirical coverage (α=.05) | {:>17.4} | {:>18.4}", conf_cov, base_cov);
    println!("Mean interval CRPS      | {:>17.4} | {:>18.4}", conf_crps, base_crps);
    println!("Mean Winkler score      | {:>17.4} | {:>18.4}", conf_winkler, base_winkler);
    println!("Point-forecast RMSE     | {:>17.4} | {:>18.4}", conf_rmse, base_rmse);
    println!();
    println!("Training residual σ     = {:.4}", std_res);
    println!("Per-step CRPS (conformal) min/mean/max = {:.4} / {:.4} / {:.4}",
             conf_crps_values.iter().cloned().fold(f32::INFINITY, f32::min),
             conf_crps_values.iter().sum::<f32>() / conf_crps_values.len() as f32,
             conf_crps_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Verdict: conformal overlay should have coverage closer to 0.95 than the
    // ±2σ baseline (which assumes Gaussian residuals — a poor fit for
    // multiplicative data), and competitive CRPS.
    println!();
    let _ = winkler_score; // sink — prove the import is used
    let target_coverage = 1.0 - alpha;
    let conf_cov_err = (conf_cov - target_coverage).abs();
    let base_cov_err = (base_cov - target_coverage).abs();
    if conf_cov_err < base_cov_err {
        println!("✅ Conformal overlay coverage ({:.4}) is closer to target ({:.4}) than ±2σ baseline ({:.4})",
                 conf_cov, target_coverage, base_cov);
    } else {
        println!("⚠ Conformal overlay coverage ({:.4}) is NOT closer to target ({:.4}) than ±2σ baseline ({:.4}) — investigate",
                 conf_cov, target_coverage, base_cov);
    }
    if conf_crps <= 2.0 * base_crps {
        println!("✅ Conformal CRPS ({:.4}) is within 2× of baseline ({:.4}) — Report-the-Floor gate holds", conf_crps, base_crps);
    } else {
        println!("⚠ Conformal CRPS ({:.4}) exceeds 2× baseline ({:.4}) — investigate", conf_crps, base_crps);
    }
}
