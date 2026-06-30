//! KARC + conformal overlay — Plan 340 Phase 2 T2.2.
//!
//! Fits KARC on the Lorenz-63 chaotic attractor (3D), then wraps it with the
//! conformal overlay using the documented KARC integration pattern
//! (`interval_from_point_into` — see `conformal::karc_adapter` module docs for
//! why `interval_into` is incompatible with KARC).
//!
//! Reports empirical coverage at α=0.05 over an online rolling-origin
//! evaluation. The gate: coverage should be in [0.90, 1.00] (KARC's
//! chaotic-regime residuals are heavier-tailed than the seasonal-pool case,
//! so we widen the lower bound from the Phase 1 [0.93, 0.97] target).
//!
//! Run with:
//! ```text
//! cargo run --release --example conformal_karc_overlay \
//!   --features conformal_predictive_intervals,karc_forecaster
//! ```

use katgpt_core::{
    ChebyshevBasis, ConformalIntervalCalibrator, DecayUnit, KarcForecaster, PredictiveInterval,
    ResidualMode, crps_interval, empirical_coverage, mean_crps_interval,
};

// ── Lorenz-63 parameters (classic chaotic regime) ─────────────────────────
const SIGMA: f64 = 10.0;
const RHO: f64 = 28.0;
const BETA: f64 = 8.0 / 3.0;

/// Lorenz-63 right-hand side. State `[x, y, z]`.
#[inline]
fn lorenz_rhs(state: &[f64; 3], out: &mut [f64; 3]) {
    let (x, y, z) = (state[0], state[1], state[2]);
    out[0] = SIGMA * (y - x);
    out[1] = x * (RHO - z) - y;
    out[2] = x * y - BETA * z;
}

/// Classical RK4 step (f64) for Lorenz-63.
fn rk4_step(state: &mut [f64; 3], dt: f64) {
    let mut k1 = [0.0; 3];
    let mut k2 = [0.0; 3];
    let mut k3 = [0.0; 3];
    let mut k4 = [0.0; 3];
    let mut tmp = [0.0; 3];

    lorenz_rhs(state, &mut k1);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k1[j];
    }
    lorenz_rhs(&tmp, &mut k2);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k2[j];
    }
    lorenz_rhs(&tmp, &mut k3);
    for j in 0..3 {
        tmp[j] = state[j] + dt * k3[j];
    }
    lorenz_rhs(&tmp, &mut k4);
    for j in 0..3 {
        state[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
    }
}

/// Generate `n` samples of the Lorenz-63 trajectory at sampling interval `dt`
/// after discarding `n_transient` transient samples. Returns the f32 trajectory
/// as a flat `n * 3` buffer (row-major: `[x0, y0, z0, x1, y1, z1, ...]`).
fn generate_lorenz(n_transient: usize, n: usize, dt: f64) -> Vec<f32> {
    let mut state = [0.1_f64, 0.0, 0.0];
    // Transient.
    for _ in 0..n_transient {
        rk4_step(&mut state, dt);
    }
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        rk4_step(&mut state, dt);
        out.push(state[0] as f32);
        out.push(state[1] as f32);
        out.push(state[2] as f32);
    }
    out
}

// ── KARC shape ────────────────────────────────────────────────────────────
// D=3 (Lorenz is 3D), M=8 (Chebyshev features per coordinate), K=4 (delay
// embedding depth). d_h = K·D·M = 96 features. Matches the Plan 308 KARC
// double-scroll example's "first-order full-rank" config.
const D: usize = 3;
const M: usize = 8;
const K: usize = 4;

fn main() {
    // ── 1. Generate trajectory ────────────────────────────────────────────
    // dt=0.02 (coarser than the ODE intrinsic scale) keeps consecutive samples
    // decorrelated enough for the Gram matrix to be well-conditioned with a
    // moderate λ. Finer dt (e.g. 0.01) over-correlates and needs heavier λ.
    let dt = 0.02;
    let n_transient = 1_000;
    let n_train = 4_000;
    let n_test = 2_000;
    let traj_raw = generate_lorenz(n_transient, n_train + n_test + K, dt);
    // traj has (n_train + n_test + K) rows of 3.

    // Normalize each coordinate to [-1, 1] (Chebyshev basis requires |x| ≤ 1
    // for numerical stability — Lorenz z can reach ~45). Track scale/offset
    // so we can un-normalize forecasts back to physical units.
    let mut scale = [1.0_f32; D];
    let mut offset = [0.0_f32; D];
    for ch in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in 0..(n_train + n_test + K) {
            let v = traj_raw[row * D + ch];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        offset[ch] = 0.5 * (lo + hi);
        scale[ch] = 0.5 * (hi - lo).max(1e-6);
    }
    let traj: Vec<f32> = (0..traj_raw.len())
        .map(|i| {
            let ch = i % D;
            (traj_raw[i] - offset[ch]) / scale[ch]
        })
        .collect();

    println!("=== KARC + Conformal Overlay (Plan 340 Phase 2 T2.2) ===");
    println!("System: Lorenz-63 (σ={}, ρ={}, β={:.4})", SIGMA, RHO, BETA);
    println!("KARC shape: D={}, M={}, K={} (d_h = {})", D, M, K, K * D * M);
    println!("n_train={}, n_test={}, dt={}", n_train, n_test, dt);
    println!("Trajectory normalized to [-1, 1] per channel (Chebyshev requires |x| ≤ 1)");
    println!();

    // ── 2. Fit KARC on the training portion ───────────────────────────────
    let basis = ChebyshevBasis::<M>::new();
    let mut karc = KarcForecaster::<ChebyshevBasis<M>, D, M, K>::with_capacity(basis, n_train);

    // Build training pairs: delay_state = [u_t, u_{t-1}, ..., u_{t-K+1}]
    // flattened (K·D), target = u_{t+1}.
    for t in (K - 1)..(n_train - 1) {
        let mut ds = [0.0_f32; K * D];
        for k in 0..K {
            let row = &traj[(t - k) * D..(t - k + 1) * D];
            ds[k * D..(k + 1) * D].copy_from_slice(row);
        }
        let target_row = &traj[(t + 1) * D..(t + 2) * D];
        let mut target = [0.0_f32; D];
        target.copy_from_slice(target_row);
        karc.accumulate_pair(&ds, &target);
    }
    // λ=1e-3: moderate regularization. The Chebyshev Gram on normalized Lorenz
    // is well-conditioned at this dt; λ=1e-6 under-regularizes and Cholesky fails.
    karc.fit_ridge(1e-3).expect("KARC fit");
    println!("KARC fitted: {} samples, λ=1e-3", karc.n_samples());

    // ── 3. Conformal overlay: one calibrator per channel ──────────────────
    // The KARC integration pattern (see conformal::karc_adapter module docs):
    //   - Use a SeasonalNaiveForecaster placeholder (the floor) as the wrapped
    //     forecaster — it's never called via interval_into because we use
    //     interval_from_point_into for the read path.
    //   - Actually, simpler: use a ConstForecaster-like placeholder. But we
    //     don't have one exported. The cleanest approach is to use the KARC
    //     adapter for type-level composition. But interval_into would panic.
    //     So we use SeasonalNaiveForecaster as a dummy forecaster (its
    //     forecast_into is called with empty delay_state inside interval_into,
    //     returning 0.0 — which we ignore because we use interval_from_point_into).
    //
    // For clarity and to demonstrate the adapter, we build ONE calibrator per
    // channel using the KarcChannelForecaster adapter — BUT we only call
    // update_residual + interval_from_point_into (never interval_into).
    use katgpt_core::KarcChannelForecaster;

    let alpha = 0.05_f32;
    let mut calibrators: Vec<ConformalIntervalCalibrator<KarcChannelForecaster<ChebyshevBasis<M>, D, M, K>>> =
        Vec::with_capacity(D);
    for ch in 0..D {
        // Each channel gets its own KARC adapter (sharing the same fitted
        // Wout via clone). In production you'd keep ONE KARC forecaster and
        // use interval_from_point_into for all channels — this per-channel
        // adapter demo is just to show the type-level composition.
        let karc_clone = clone_fitted_karc(&karc);
        let adapter = KarcChannelForecaster::new(karc_clone, ch);
        let cal = ConformalIntervalCalibrator::new(
            adapter,
            1, // 1 channel per calibrator (per-channel)
            1, // max_h = 1 (KARC is h=1)
            1, // m = 1 (non-seasonal)
            256, // capacity
            0.0, // no recency decay (stationary within the test window)
            DecayUnit::Step,
            ResidualMode::HStep,
            false,
        );
        calibrators.push(cal);
    }

    // ── 4. Warmup + rolling-origin evaluation ─────────────────────────────
    // Warmup: push the training tail into the residual pools so the first
    // test-step has a non-empty pool.
    let warmup_start = n_train - 256; // one pool-capacity of warmup
    for t in warmup_start..n_train {
        let mut ds = [0.0_f32; K * D];
        for k in 0..K {
            let row = &traj[(t - k) * D..(t - k + 1) * D];
            ds[k * D..(k + 1) * D].copy_from_slice(row);
        }
        let actual_row = &traj[(t + 1) * D..(t + 2) * D];
        // KARC forecast for this step (all D channels).
        let mut point = [0.0_f32; D];
        calibrators[0].forecaster.karc.forecast_into(&ds, &mut point);
        for ch in 0..D {
            calibrators[ch].update_residual(actual_row[ch], point[ch], 0, 1);
            calibrators[ch].step();
        }
    }

    // Test: rolling-origin. For each test step, forecast, read interval,
    // observe actual, update residual.
    let mut all_intervals: Vec<Vec<PredictiveInterval>> = (0..D).map(|_| Vec::new()).collect();
    let mut all_actuals: Vec<Vec<f32>> = (0..D).map(|_| Vec::new()).collect();
    let mut all_crps: Vec<Vec<f32>> = (0..D).map(|_| Vec::new()).collect();

    for t in n_train..(n_train + n_test) {
        let mut ds = [0.0_f32; K * D];
        for k in 0..K {
            let row = &traj[(t - k) * D..(t - k + 1) * D];
            ds[k * D..(k + 1) * D].copy_from_slice(row);
        }
        let actual_row = &traj[(t + 1) * D..(t + 2) * D];

        // KARC forecasts all D channels in one matvec.
        let mut point = [0.0_f32; D];
        calibrators[0].forecaster.karc.forecast_into(&ds, &mut point);

        for ch in 0..D {
            // Read the calibrated interval (point-supplied path — the
            // documented KARC pattern).
            let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, alpha);
            calibrators[ch].interval_from_point_into(point[ch], 0, 1, alpha, &mut iv);
            all_intervals[ch].push(iv);
            all_actuals[ch].push(actual_row[ch]);
            all_crps[ch].push(crps_interval(&iv, actual_row[ch]));
            // Online update.
            calibrators[ch].update_residual(actual_row[ch], point[ch], 0, 1);
            calibrators[ch].step();
        }
    }

    // ── 5. Report ─────────────────────────────────────────────────────────
    let channel_names = ["x", "y", "z"];
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>12}",
        "Channel", "Coverage", "MeanCRPS", "MeanHalfWid", "RMSE"
    );
    println!("{:-<60}", "");

    let target_cov = 1.0 - alpha;
    let mut all_pass = true;
    for ch in 0..D {
        let cov = empirical_coverage(&all_intervals[ch], &all_actuals[ch]);
        let crps = mean_crps_interval(&all_intervals[ch], &all_actuals[ch]);
        let mean_half: f32 = all_intervals[ch].iter().map(|iv| iv.half_width()).sum::<f32>()
            / all_intervals[ch].len() as f32;
        let rmse: f32 = {
            let mut se = 0.0_f32;
            for (iv, &y) in all_intervals[ch].iter().zip(all_actuals[ch].iter()) {
                se += (iv.point - y).powi(2);
            }
            (se / all_intervals[ch].len() as f32).sqrt()
        };
        println!(
            "{:<8} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            channel_names[ch],
            cov,
            crps,
            mean_half,
            rmse
        );
        // Gate: coverage in [0.90, 1.00] (chaotic regime, widened lower bound).
        if !(0.90..=1.00).contains(&cov) {
            all_pass = false;
        }
    }

    println!();
    println!("Target coverage (1−α): {:.4}", target_cov);
    println!("Coverage gate: [0.90, 1.00] (chaotic regime, widened from Phase 1 [0.93, 0.97])");
    if all_pass {
        println!("✅ All channels pass the coverage gate — KARC + conformal overlay is calibrated.");
    } else {
        println!("⚠ At least one channel outside [0.90, 1.00] — investigate residual distribution.");
    }
}

/// Clone a fitted KARC forecaster (preserving the Wout). Used to give each
/// per-channel calibrator its own adapter without re-fitting.
fn clone_fitted_karc<B, const D: usize, const M: usize, const K: usize>(
    src: &KarcForecaster<B, D, M, K>,
) -> KarcForecaster<B, D, M, K>
where
    B: katgpt_core::KarcBasis<M> + Copy,
{
    // Re-build with the same basis + capacity, then restore Wout via the
    // public `restore_wout` API (the freeze/thaw bridge).
    let n = src.n_samples();
    let mut dst = KarcForecaster::with_capacity(src.basis, n.max(1));
    let wout = src.wout.clone();
    dst.restore_wout(wout);
    dst
}
