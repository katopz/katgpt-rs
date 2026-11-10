//! VFD (Velocity-Field Disagreement) UQ conformal floor comparison.
//!
//! Plan 432 Phase 2 T2.2 — the make-or-break GOAT gate per Issue 010
//! ("Report the Floor" rule). Tests whether VFD-calibrated prediction
//! intervals beat the canonical conformal-naive floor
//! (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with m=1)
//! on:
//!   (a) AR(1) corpus — same as Plan 376 Phase 6.
//!   (b) 1D bimodal flow-matching toy — simplified from the paper's 2D setup
//!       (Appendix C.1) because the floor harness handles scalar trajectories.
//!
//! ## The VFD-bearing forecaster design
//!
//! VFD produces a scalar disagreement score `u_e(y) ∈ [0, +∞)`, NOT samples or
//! intervals. To compare it as a UQ primitive, we construct a VFD-calibrated
//! interval forecaster:
//!
//! ```text
//!   point_forecast = ensemble_mean_prediction(y_t)
//!   total_variance  = σ_ale² + λ · max(u_e(y_t), 0)
//!   interval        = [point ± z(α) · sqrt(total_variance)]
//! ```
//!
//! where:
//! - `σ_ale` is the training residual std (aleatoric uncertainty).
//! - `λ` is a scaling factor, calibrated modellessly via grid search on
//!   training CRPS. The grid INCLUDES λ=0 (pure ensemble baseline), so VFD
//!   can only help or tie — never regress below the ensemble.
//! - `z(α)` is the Gaussian quantile (z_{1-α/2}).
//!
//! The velocity fields are closures conditioned on the current observation
//! `y_t`, so `u_e(y_t)` varies per-target (high disagreement where members
//! diverge).
//!
//! ## What the test reports
//!
//! For each corpus, the test prints:
//! 1. VFD (optimal λ) vs floor — the GOAT gate comparison.
//! 2. The optimal λ value — if λ ≈ 0, VFD isn't contributing.
//! 3. VFD (λ=0, pure ensemble baseline) vs floor — isolates the point-forecast
//!    advantage from VFD's epistemic contribution.
//!
//! ## Reproduction
//!
//! ```sh
//! cargo test -p katgpt-core \
//!   --features velocity_field_disagreement,conformal_predictive_intervals \
//!   --test velocity_field_disagreement_uq_floor -- --nocapture --ignored
//! ```

use katgpt_core::conformal::{
    PredictiveInterval, PredictiveOutput, TrajectoryCorpus, UqPrimitiveUnderTest,
    run_floor_comparison,
};
use katgpt_core::velocity_field_disagreement::{VfdScratch, vfd_score_into};
use katgpt_core::velocity_field_ensemble::{ClosureField, Schedule, VelocityField};

// ── Constants ─────────────────────────────────────────────────────────────

/// AR(1) parameters (matches Plan 376 Phase 6 corpus).
const AR1_PHI: f32 = 0.7;
const AR1_SIGMA: f32 = 0.5;
const N_TRAIN: usize = 200;
const N_TEST: usize = 200;
const SEED: u64 = 0x1234_5678_9ABC_DEF0;

/// VFD hyperparameters (paper defaults: M=2, N_s=10, B=5).
const VFD_N_STEPS: usize = 10;
const VFD_BATCH: usize = 5;

/// Grid of λ values for modelless calibration. Includes 0 (pure ensemble
/// baseline) so VFD can only help or tie.
const LAMBDA_GRID: &[f32] = &[0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0];

/// Gaussian z-quantile for α=0.05 (95% interval): z_{0.975} ≈ 1.96.
const Z_975: f32 = 1.959_964;

// ── Deterministic RNG (SplitMix64 — matches floor_harness) ────────────────

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Approximate standard normal via central-limit sum of 12 uniforms.
    fn gaussian(&mut self, sigma: f32) -> f32 {
        let mut sum = 0.0_f32;
        for _ in 0..12 {
            sum += (self.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32);
        }
        (sum - 6.0) * sigma
    }
}

// ── Corpus generators ─────────────────────────────────────────────────────

/// Generate AR(1) series: `x_{t+1} = φ·x_t + ε`, `ε ~ N(0, σ²)`.
fn generate_ar1(n: usize, phi: f32, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut series = Vec::with_capacity(n);
    series.push(rng.gaussian(sigma)); // x_0 ~ N(0, σ²) (stationary)
    for _ in 1..n {
        let prev = *series.last().unwrap();
        let next = phi * prev + rng.gaussian(sigma);
        series.push(next);
    }
    series
}

/// Generate 1D bimodal markov-switching series:
/// `x_{t+1} = μ_{s_t} + ε`, where `s_t ∈ {0,1}`, `μ_0 = +mode`, `μ_1 = -mode`,
/// `ε ~ N(0, σ²)`, and the regime switches with probability `switch_prob` per step.
fn generate_bimodal(n: usize, mode: f32, sigma: f32, switch_prob: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut series = Vec::with_capacity(n);
    let mut regime: u8 = if rng.next_u64().is_multiple_of(2) {
        0
    } else {
        1
    };
    for _ in 0..n {
        let mu = if regime == 0 { mode } else { -mode };
        series.push(mu + rng.gaussian(sigma));
        // Regime switch?
        let u = (rng.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32);
        if u < switch_prob {
            regime = 1 - regime;
        }
    }
    series
}

// ── Least-squares AR(1) coefficient estimator ─────────────────────────────

/// Estimate AR(1) coefficient φ via least squares: `φ̂ = Σ(x_t·x_{t+1}) / Σ(x_t²)`.
fn estimate_ar1_phi(series: &[f32]) -> f32 {
    let n = series.len() - 1;
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for t in 0..n {
        num += series[t] * series[t + 1];
        den += series[t] * series[t];
    }
    if den < 1e-10 {
        return 0.0;
    }
    num / den
}

/// Compute residual std of AR(1) with given φ: `σ = std(x_{t+1} - φ·x_t)`.
fn ar1_residual_std(series: &[f32], phi: f32) -> f32 {
    let n = series.len() - 1;
    let mut residuals = Vec::with_capacity(n);
    for t in 0..n {
        residuals.push(series[t + 1] - phi * series[t]);
    }
    let mean = residuals.iter().sum::<f32>() / n as f32;
    let var = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n as f32;
    var.sqrt().max(1e-3)
}

// ── VFD interval adapter ──────────────────────────────────────────────────

/// Configuration for the VFD-calibrated interval forecaster.
///
/// The adapter computes a VFD score per observation, then constructs a
/// Gaussian interval whose variance = `σ_ale² + λ·u_e(y)`.
struct VfdIntervalAdapter {
    /// The two members' AR(1) coefficients (φ_0, φ_1). Used to construct
    /// velocity fields conditioned on the current observation.
    phi: [f32; 2],
    /// Base aleatoric scale (training residual std).
    sigma_ale: f32,
    /// Epistemic scaling factor (calibrated on training via grid search).
    lambda: f32,
    /// Current observed state (last `observe` value).
    current_y: f32,
    /// VFD scratch (reused across predict_next calls — zero-alloc).
    vfd_scratch: VfdScratch<2, 1>,
    /// RNG for VFD's sample_normal (Monte Carlo initial conditions + SDE steps).
    vfd_rng: SplitMix64,
    /// VFD hyperparams.
    n_steps: usize,
    batch: usize,
    schedule: Schedule,
}

impl VfdIntervalAdapter {
    /// Construct with member coefficients, aleatoric scale, and lambda.
    fn new(phi: [f32; 2], sigma_ale: f32, lambda: f32) -> Self {
        Self {
            phi,
            sigma_ale,
            lambda,
            current_y: 0.0,
            vfd_scratch: VfdScratch::new(),
            vfd_rng: SplitMix64::new(SEED.wrapping_add(42)),
            n_steps: VFD_N_STEPS,
            batch: VFD_BATCH,
            schedule: Schedule::Linear,
        }
    }

    /// Compute the VFD score at the current observation `self.current_y`.
    ///
    /// The two velocity fields are closures conditioned on `self.current_y`:
    ///   member i: `v_i(x) = phi_i * current_y - x`
    /// This is the flow-matching marginal velocity for a linear-Gaussian
    /// conditional path toward target `phi_i * current_y`. The disagreement
    /// between members is `(phi_0 - phi_1)² * current_y²`, which varies
    /// per-target — high when |current_y| is large.
    fn compute_vfd_score(&mut self) -> f32 {
        let y = self.current_y;
        let phi0 = self.phi[0];
        let phi1 = self.phi[1];

        // Member 0: drift toward phi_0 * y.
        let f0 = ClosureField::new(0, move |x: &[f32], out: &mut [f32; 1]| {
            out[0] = phi0 * y - x[0];
        });
        // Member 1: drift toward phi_1 * y.
        let f1 = ClosureField::new(1, move |x: &[f32], out: &mut [f32; 1]| {
            out[0] = phi1 * y - x[0];
        });
        let fields: [&dyn VelocityField<1>; 2] = [&f0, &f1];

        vfd_score_into(
            &fields,
            self.schedule,
            self.n_steps,
            self.batch,
            &mut self.vfd_scratch,
            &mut || self.vfd_rng.gaussian(1.0),
        )
    }

    /// Ensemble mean point forecast: `(phi_0 + phi_1)/2 * current_y`.
    fn point_forecast(&self) -> f32 {
        0.5 * (self.phi[0] + self.phi[1]) * self.current_y
    }

    /// Predictive interval at level α, using VFD-calibrated variance.
    fn predictive_interval(&mut self, alpha: f32) -> PredictiveInterval {
        let point = self.point_forecast();
        let vfd_score = self.compute_vfd_score();
        let total_var = self.sigma_ale.powi(2) + self.lambda * vfd_score.max(0.0);
        let total_std = total_var.max(1e-10).sqrt();
        PredictiveInterval::new(
            point - Z_975 * total_std,
            point,
            point + Z_975 * total_std,
            alpha,
        )
    }
}

impl UqPrimitiveUnderTest for VfdIntervalAdapter {
    fn name(&self) -> &str {
        "VFD (2 linear closure members, VFD-scaled Gaussian interval)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        PredictiveOutput::from_interval(self.predictive_interval(0.05))
    }

    fn observe(&mut self, y: f32) {
        self.current_y = y;
    }
}

// ── Bimodal VFD adapter (fixed attractors, no φ estimation) ───────────────

/// VFD adapter for the bimodal toy: two members with FIXED attractors (+mode,
/// -mode). The disagreement is constant (doesn't depend on y), so VFD
/// provides a constant variance inflation — testing whether uniform epistemic
/// widening beats adaptive conformal calibration.
struct VfdBimodalAdapter {
    /// The two attractor targets: [+mode, -mode].
    attractors: [f32; 2],
    /// Base aleatoric scale (training residual std).
    sigma_ale: f32,
    /// Epistemic scaling factor.
    lambda: f32,
    /// Current observed state.
    current_y: f32,
    /// VFD scratch.
    vfd_scratch: VfdScratch<2, 1>,
    /// RNG for VFD.
    vfd_rng: SplitMix64,
    n_steps: usize,
    batch: usize,
    schedule: Schedule,
}

impl VfdBimodalAdapter {
    fn new(attractors: [f32; 2], sigma_ale: f32, lambda: f32) -> Self {
        Self {
            attractors,
            sigma_ale,
            lambda,
            current_y: 0.0,
            vfd_scratch: VfdScratch::new(),
            vfd_rng: SplitMix64::new(SEED.wrapping_add(99)),
            n_steps: VFD_N_STEPS,
            batch: VFD_BATCH,
            schedule: Schedule::Linear,
        }
    }

    /// Compute VFD score. Members: `v_i(x) = attractors[i] - x`.
    /// Disagreement = `(attractors[0] - attractors[1])²` — constant.
    fn compute_vfd_score(&mut self) -> f32 {
        let a0 = self.attractors[0];
        let a1 = self.attractors[1];
        let f0 = ClosureField::new(0, move |x: &[f32], out: &mut [f32; 1]| {
            out[0] = a0 - x[0];
        });
        let f1 = ClosureField::new(1, move |x: &[f32], out: &mut [f32; 1]| {
            out[0] = a1 - x[0];
        });
        let fields: [&dyn VelocityField<1>; 2] = [&f0, &f1];
        vfd_score_into(
            &fields,
            self.schedule,
            self.n_steps,
            self.batch,
            &mut self.vfd_scratch,
            &mut || self.vfd_rng.gaussian(1.0),
        )
    }

    /// Ensemble mean point forecast: `(attractors[0] + attractors[1]) / 2`.
    /// For symmetric attractors (+mode, -mode), this is 0.
    fn point_forecast(&self) -> f32 {
        0.5 * (self.attractors[0] + self.attractors[1])
    }

    fn predictive_interval(&mut self, alpha: f32) -> PredictiveInterval {
        let point = self.point_forecast();
        let vfd_score = self.compute_vfd_score();
        let total_var = self.sigma_ale.powi(2) + self.lambda * vfd_score.max(0.0);
        let total_std = total_var.max(1e-10).sqrt();
        PredictiveInterval::new(
            point - Z_975 * total_std,
            point,
            point + Z_975 * total_std,
            alpha,
        )
    }
}

impl UqPrimitiveUnderTest for VfdBimodalAdapter {
    fn name(&self) -> &str {
        "VFD (bimodal fixed attractors, VFD-scaled Gaussian interval)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        PredictiveOutput::from_interval(self.predictive_interval(0.05))
    }

    fn observe(&mut self, y: f32) {
        self.current_y = y;
    }
}

// ── λ calibration (modelless grid search on training CRPS) ────────────────

/// Compute mean interval-CRPS for a series of (prediction, actual) pairs.
fn mean_crps(predictions: &[f32], actuals: &[f32], interval_half_width: f32) -> f32 {
    let n = predictions.len();
    if n == 0 {
        return f32::INFINITY;
    }
    let mut total = 0.0_f32;
    for (pred, actual) in predictions.iter().zip(actuals.iter()) {
        let lower = pred - interval_half_width;
        let upper = pred + interval_half_width;
        let width = upper - lower;
        let crps = if *actual < lower {
            width + 2.0 * (lower - *actual)
        } else if *actual > upper {
            width + 2.0 * (*actual - upper)
        } else {
            width
        };
        total += crps;
    }
    total / n as f32
}

/// Calibrate λ for the AR(1) VFD adapter via grid search on training CRPS.
///
/// For each λ in LAMBDA_GRID, computes the VFD score at each training point,
/// constructs the interval, and measures mean CRPS. Returns the best λ.
///
/// This is modelless (grid search, no gradient descent).
fn calibrate_lambda_ar1(phi: [f32; 2], sigma_ale: f32, train_series: &[f32]) -> (f32, f32) {
    let n = train_series.len() - 1;

    // Pre-compute VFD scores for each training point y_t.
    // VFD for linear fields is analytic: u_e(y) = (phi_0 - phi_1)²·y² · C
    // where C = (1/N_s)·Σ_ℓ κ_{s_ℓ} depends only on schedule + n_steps.
    // We compute C numerically to avoid schedule-specific formulas.
    let c_factor = compute_kappa_sum_factor(Schedule::Linear, VFD_N_STEPS);
    let phi_diff_sq = (phi[0] - phi[1]).powi(2);

    let vfd_scores: Vec<f32> = train_series
        .iter()
        .take(n)
        .map(|&y| phi_diff_sq * y * y * c_factor)
        .collect();

    // For each λ, compute mean CRPS.
    let mut best_lambda = 0.0_f32;
    let mut best_crps = f32::INFINITY;
    for &lambda in LAMBDA_GRID {
        let mut total_crps = 0.0_f32;
        for t in 0..n {
            let y = train_series[t];
            let actual = train_series[t + 1];
            let point = 0.5 * (phi[0] + phi[1]) * y;
            let total_var = sigma_ale.powi(2) + lambda * vfd_scores[t];
            let half_width = Z_975 * total_var.max(1e-10).sqrt();
            let lower = point - half_width;
            let upper = point + half_width;
            let width = 2.0 * half_width;
            let crps = if actual < lower {
                width + 2.0 * (lower - actual)
            } else if actual > upper {
                width + 2.0 * (actual - upper)
            } else {
                width
            };
            total_crps += crps;
        }
        let mean_crps = total_crps / n as f32;
        if mean_crps < best_crps {
            best_crps = mean_crps;
            best_lambda = lambda;
        }
    }
    (best_lambda, best_crps)
}

/// Calibrate λ for the bimodal VFD adapter via grid search.
fn calibrate_lambda_bimodal(
    attractors: [f32; 2],
    sigma_ale: f32,
    train_series: &[f32],
) -> (f32, f32) {
    let n = train_series.len() - 1;
    let c_factor = compute_kappa_sum_factor(Schedule::Linear, VFD_N_STEPS);
    let disagreement_sq = (attractors[0] - attractors[1]).powi(2);
    let vfd_score = disagreement_sq * c_factor; // constant for all y

    let point = 0.5 * (attractors[0] + attractors[1]); // constant (0 for symmetric)

    let mut best_lambda = 0.0_f32;
    let mut best_crps = f32::INFINITY;
    for &lambda in LAMBDA_GRID {
        let total_var = sigma_ale.powi(2) + lambda * vfd_score;
        let half_width = Z_975 * total_var.max(1e-10).sqrt();
        let predictions: Vec<f32> = vec![point; n];
        let actuals: Vec<f32> = train_series[1..].to_vec();
        let mean_crps_val = mean_crps(&predictions, &actuals, half_width);
        if mean_crps_val < best_crps {
            best_crps = mean_crps_val;
            best_lambda = lambda;
        }
    }
    (best_lambda, best_crps)
}

/// Compute the κ_s Riemann sum factor: `C = (1/N_s) · Σ_{ℓ=0}^{N_s-1} κ_{s_ℓ}`
/// where `s_ℓ = ℓ/N_s`. For constant-disagreement fields, VFD = disagreement² · C.
///
/// This is extracted from the VFD algorithm's normalization. It depends only
/// on the schedule and n_steps, not on the fields or observation.
fn compute_kappa_sum_factor(schedule: Schedule, n_steps: usize) -> f32 {
    let delta_s = 1.0 / n_steps as f32;
    let mut kappa_sum = 0.0_f32;
    for l in 0..n_steps {
        let s_l = l as f32 * delta_s;
        // κ_s = beta / (alpha * gamma) — same formula as kappa_s() in the module.
        let (alpha, beta) = schedule.alpha_beta(s_l);
        let gamma = schedule.gamma(s_l);
        kappa_sum += beta / (alpha * gamma);
    }
    kappa_sum / n_steps as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// AR(1) corpus: VFD-calibrated intervals vs conformal floor.
///
/// Two members are ridge-fit (via least-squares φ estimation) on disjoint
/// training halves. The VFD score varies with y² (high when |y| is large).
/// λ is grid-search-calibrated on training CRPS (includes λ=0 baseline).
#[test]
#[ignore = "prints a VFD-vs-conformal-floor comparison table (0 assertions, 41 println); run with --release --ignored"]
fn vfd_vs_floor_ar1() {
    // Generate full corpus.
    let full = generate_ar1(N_TRAIN + N_TEST, AR1_PHI, AR1_SIGMA, SEED);
    let train = &full[..N_TRAIN];
    let test_corpus = &full[N_TRAIN..];

    // Split training into two halves; estimate φ for each member.
    let half = N_TRAIN / 2;
    let phi_0 = estimate_ar1_phi(&train[..half]);
    let phi_1 = estimate_ar1_phi(&train[half..]);
    let phi = [phi_0, phi_1];

    // Estimate σ_ale from full training residuals (using the mean φ).
    let mean_phi = 0.5 * (phi_0 + phi_1);
    let sigma_ale = ar1_residual_std(train, mean_phi);

    println!("=== AR(1) VFD Floor Comparison Setup ===");
    println!("True φ = {AR1_PHI}, σ = {AR1_SIGMA}");
    println!("Member 0 φ̂ = {phi_0:.6} (fit on first {half} pairs)");
    println!(
        "Member 1 φ̂ = {phi_1:.6} (fit on last {} pairs)",
        N_TRAIN - half
    );
    println!("|φ_0 - φ_1| = {:.6}", (phi_0 - phi_1).abs());
    println!("σ_ale = {sigma_ale:.6}");
    println!();

    // Calibrate λ on training.
    let (best_lambda, train_crps) = calibrate_lambda_ar1(phi, sigma_ale, train);
    println!("λ calibration (grid search on training CRPS):");
    println!("  Best λ = {best_lambda}, training CRPS = {train_crps:.6}");
    println!();

    // Run floor comparison with optimal λ.
    let corpus = TrajectoryCorpus::from_slice(
        format!("ar1_phi{AR1_PHI}_sigma{AR1_SIGMA}_n{N_TEST}"),
        test_corpus,
        32,
    );

    println!("--- VFD (λ={best_lambda}) vs floor ---");
    let mut adapter = VfdIntervalAdapter::new(phi, sigma_ale, best_lambda);
    let report = run_floor_comparison(
        &mut adapter,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    report.pretty_print();
    println!("Primitive wins? {}", report.primitive_wins());
    println!();

    // Also run with λ=0 (pure ensemble baseline) to isolate VFD's contribution.
    println!("--- VFD (λ=0, pure ensemble baseline) vs floor ---");
    let mut baseline = VfdIntervalAdapter::new(phi, sigma_ale, 0.0);
    let baseline_report = run_floor_comparison(
        &mut baseline,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    baseline_report.pretty_print();
    println!();

    // Summary.
    println!("=== AR(1) Summary ===");
    println!(
        "VFD (λ={best_lambda}): CRPS={:.4}, Winkler={:.4}, Cov={:.4}",
        report.primitive.mean_crps_interval,
        report.primitive.mean_winkler,
        report.primitive.coverage
    );
    println!(
        "VFD (λ=0):      CRPS={:.4}, Winkler={:.4}, Cov={:.4}",
        baseline_report.primitive.mean_crps_interval,
        baseline_report.primitive.mean_winkler,
        baseline_report.primitive.coverage
    );
    println!(
        "Floor:          CRPS={:.4}, Winkler={:.4}, Cov={:.4}",
        report.floor.mean_crps_interval, report.floor.mean_winkler, report.floor.coverage
    );
    let vfd_helps =
        report.primitive.mean_crps_interval < baseline_report.primitive.mean_crps_interval;
    let vfd_verdict = if vfd_helps { "HELPS" } else { "does NOT help" };
    println!("\nVFD epistemic scaling {vfd_verdict} (λ* CRPS < λ=0 CRPS)");
    println!("Overall verdict (λ={best_lambda}): {:?}", report.overall);
}

/// Bimodal corpus: VFD-calibrated intervals vs conformal floor.
///
/// Two members with fixed attractors (+2, -2). The ground truth is a
/// markov-switching process between the two modes. VFD score is constant
/// (constant disagreement between attractors).
#[test]
#[ignore = "prints a VFD-vs-conformal-floor comparison table (0 assertions, 41 println); run with --release --ignored"]
fn vfd_vs_floor_bimodal() {
    let mode = 2.0_f32;
    let sigma = 0.5_f32;
    let switch_prob = 0.05_f32;

    let full = generate_bimodal(N_TRAIN + N_TEST, mode, sigma, switch_prob, SEED);
    let train = &full[..N_TRAIN];
    let test_corpus = &full[N_TRAIN..];

    let attractors = [mode, -mode];

    // σ_ale: residual std of the training series around the ensemble mean (0).
    let train_mean = train.iter().sum::<f32>() / train.len() as f32;
    let train_var =
        train.iter().map(|x| (x - train_mean).powi(2)).sum::<f32>() / train.len() as f32;
    let sigma_ale = train_var.sqrt().max(1e-3);

    println!("=== Bimodal VFD Floor Comparison Setup ===");
    println!("Mode = ±{mode}, σ = {sigma}, switch_prob = {switch_prob}");
    println!("Attractors: [+{mode}, -{mode}]");
    println!("σ_ale = {sigma_ale:.6}");
    println!("Ensemble mean point forecast = 0.0 (constant)");
    println!();

    // Calibrate λ.
    let (best_lambda, train_crps) = calibrate_lambda_bimodal(attractors, sigma_ale, train);
    println!("λ calibration (grid search on training CRPS):");
    println!("  Best λ = {best_lambda}, training CRPS = {train_crps:.6}");
    println!();

    let corpus = TrajectoryCorpus::from_slice(
        format!("bimodal_mode{mode}_sigma{sigma}_switch{switch_prob}_n{N_TEST}"),
        test_corpus,
        32,
    );

    println!("--- VFD (λ={best_lambda}) vs floor ---");
    let mut adapter = VfdBimodalAdapter::new(attractors, sigma_ale, best_lambda);
    let report = run_floor_comparison(
        &mut adapter,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    report.pretty_print();
    println!("Primitive wins? {}", report.primitive_wins());
    println!();

    // λ=0 baseline.
    println!("--- VFD (λ=0, pure ensemble baseline) vs floor ---");
    let mut baseline = VfdBimodalAdapter::new(attractors, sigma_ale, 0.0);
    let baseline_report = run_floor_comparison(
        &mut baseline,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    baseline_report.pretty_print();
    println!();

    println!("=== Bimodal Summary ===");
    println!(
        "VFD (λ={best_lambda}): CRPS={:.4}, Winkler={:.4}, Cov={:.4}",
        report.primitive.mean_crps_interval,
        report.primitive.mean_winkler,
        report.primitive.coverage
    );
    println!(
        "VFD (λ=0):      CRPS={:.4}, Winkler={:.4}, Cov={:.4}",
        baseline_report.primitive.mean_crps_interval,
        baseline_report.primitive.mean_winkler,
        baseline_report.primitive.coverage
    );
    println!(
        "Floor:          CRPS={:.4}, Winkler={:.4}, Cov={:.4}",
        report.floor.mean_crps_interval, report.floor.mean_winkler, report.floor.coverage
    );
    let vfd_helps =
        report.primitive.mean_crps_interval < baseline_report.primitive.mean_crps_interval;
    let vfd_verdict = if vfd_helps { "HELPS" } else { "does NOT help" };
    println!("\nVFD epistemic scaling {vfd_verdict} (λ* CRPS < λ=0 CRPS)");
    println!("Overall verdict (λ={best_lambda}): {:?}", report.overall);
}
