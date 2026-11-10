//! Velocity-Field Ensemble UQ conformal floor comparison (Plan 376 Phase 6,
//! Issue 038).
//!
//! Runs the velocity-field ensemble + stochastic interpolator against the
//! canonical conformal-naive floor (`ConformalIntervalCalibrator<
//! SeasonalNaiveForecaster>` with m=1) on an AR(1) corpus. Per the "Report the
//! Floor" rule (Issue 010), if the ensemble cannot beat the floor on CRPS /
//! coverage / Winkler, it cannot make a UQ claim.
//!
//! **Honest expectation:** the ensemble likely LOSES. The ensemble's induced
//! distribution is Gaussian by construction (we add Gaussian noise to the
//! deterministic drift). The conformal floor uses non-parametric empirical
//! quantile calibration, which is tighter on non-Gaussian data and at least as
//! good on Gaussian data. This test confirms or refutes that expectation.
//!
//! ## The adapter
//!
//! `VfeForecastAdapter` wraps a pre-fit `VelocityFieldEnsemble` as a
//! `UqPrimitiveUnderTest`. Each `predict_next` call:
//! 1. Evaluates the ensemble at the current observed state `x_t`.
//! 2. Generates M samples by adding Gaussian noise calibrated to the training
//!    residual std.
//! 3. Returns the samples as `PredictiveOutput::from_samples(samples)`.
//!
//! `observe` updates the current state. The ensemble is fit OFFLINE on a
//! training prefix; it does NOT refit online (this is the static-fit regime).
//!
//! ## Output
//!
//! Prints a verdict table. Capture to `.benchmarks/376_uq_floor.md` via:
//! ```sh
//! cargo test -p katgpt-core \
//!   --test velocity_field_ensemble_uq_floor -- --nocapture --ignored
//! ```

use katgpt_core::conformal::{
    PredictiveOutput, TrajectoryCorpus, UqPrimitiveUnderTest, run_floor_comparison,
};
use katgpt_core::velocity_field_ensemble::{
    ClosureField, EnsembleFitScratch, VelocityFieldEnsemble,
};

// ── Test corpus constants ─────────────────────────────────────────────────

/// AR(1) parameter: `x_{t+1} = φ·x_t + ε`, `ε ~ N(0, σ²)`.
const PHI: f32 = 0.7;
const SIGMA: f32 = 0.5;
const N_TRAIN: usize = 200;
const N_TEST: usize = 200;
const SEED: u64 = 0x1234_5678_9ABC_DEF0;

/// Number of samples the ensemble generates per prediction (for empirical
/// quantile → interval conversion in the floor harness).
const M_SAMPLES: usize = 64;

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
    fn gaussian(&mut self, sigma: f32) -> f32 {
        let mut sum = 0.0_f32;
        for _ in 0..12 {
            sum += (self.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32);
        }
        (sum - 6.0) * sigma
    }
}

// ── AR(1) corpus generation ───────────────────────────────────────────────

fn generate_ar1(n: usize, phi: f32, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut series = Vec::with_capacity(n);
    series.push(rng.gaussian(sigma)); // x_0 ~ N(0, σ²) (stationary dist)
    for _ in 1..n {
        let prev = *series.last().unwrap();
        let next = phi * prev + rng.gaussian(sigma);
        series.push(next);
    }
    series
}

// ── VFE forecast adapter ──────────────────────────────────────────────────

/// The velocity-field ensemble configured as a 1D forecaster.
///
/// Fields:
/// - Two linear closure fields with different slopes (regression basis).
/// - Pre-fit on N_TRAIN AR(1) pairs where target = x_{t+1} - x_t (drift).
struct VfeForecastAdapter {
    #[allow(clippy::type_complexity)] // generic ensemble type is inherent to the adapter
    ensemble: VelocityFieldEnsemble<ClosureField<1, fn(&[f32], &mut [f32; 1])>, 2, 1>,
    eval_scratch: [f32; 1],
    /// Estimated residual std from training (calibrates the Gaussian noise
    /// added to the drift prediction).
    noise_sigma: f32,
    /// Current observed state (last `observe` value).
    current_x: f32,
    /// RNG for sample generation (deterministic seed for reproducibility).
    rng: SplitMix64,
}

impl VfeForecastAdapter {
    fn new() -> Self {
        // Generate training corpus.
        fn field_const(_x: &[f32], out: &mut [f32; 1]) {
            out[0] = 1.0;
        }

        let train = generate_ar1(N_TRAIN, PHI, SIGMA, SEED);

        // Build training pairs: (x_t, drift=x_{t+1}-x_t).
        // The "state" seen by the field is the scalar x_t (length-1 slice).
        let x_refs: Vec<Vec<f32>> = (0..N_TRAIN - 1).map(|t| vec![train[t]]).collect();
        let drift_refs: Vec<Vec<f32>> = (0..N_TRAIN - 1)
            .map(|t| vec![train[t + 1] - train[t]])
            .collect();
        let x_slices: Vec<&[f32]> = x_refs.iter().map(|v| v.as_slice()).collect();
        let drift_slices: Vec<&[f32]> = drift_refs.iter().map(|v| v.as_slice()).collect();

        // Two closure fields: b_0(x) = x (identity), b_1(x) = 1.0 (constant).
        // These span the AR(1) drift space: drift = (φ-1)·x + ε ≈ a·x + b.
        fn field_identity(x: &[f32], out: &mut [f32; 1]) {
            out[0] = x[0];
        }
        let fields = [
            ClosureField::<1, _>::new(0, field_identity as fn(&[f32], &mut [f32; 1])),
            ClosureField::<1, _>::new(1, field_const as fn(&[f32], &mut [f32; 1])),
        ];

        let mut ensemble = VelocityFieldEnsemble::<_, 2, 1>::new(fields);
        let mut fit_scratch = EnsembleFitScratch::<2, 1>::new();
        ensemble.fit_into(&x_slices, &drift_slices, 1e-3, &mut fit_scratch);

        // Estimate residual std from training residuals (drift - predicted_drift).
        let mut residuals: Vec<f32> = Vec::with_capacity(N_TRAIN - 1);
        let mut eval_scratch = [0.0f32; 1];
        let mut predicted = [0.0f32; 1];
        for t in 0..N_TRAIN - 1 {
            ensemble.eval_into(&[train[t]], &mut predicted, &mut eval_scratch);
            let actual_drift = train[t + 1] - train[t];
            residuals.push(actual_drift - predicted[0]);
        }
        let mean_resid = residuals.iter().sum::<f32>() / residuals.len() as f32;
        let var = residuals
            .iter()
            .map(|r| (r - mean_resid).powi(2))
            .sum::<f32>()
            / residuals.len() as f32;
        let noise_sigma = var.sqrt().max(1e-3);

        Self {
            ensemble,
            eval_scratch,
            noise_sigma,
            current_x: 0.0,
            rng: SplitMix64::new(SEED.wrapping_add(1)),
        }
    }
}

impl UqPrimitiveUnderTest for VfeForecastAdapter {
    fn name(&self) -> &str {
        "VFE (2 linear closure fields, static-fit, Gaussian-noise sampler)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        // Evaluate drift at current_x.
        let mut drift = [0.0f32; 1];
        self.ensemble
            .eval_into(&[self.current_x], &mut drift, &mut self.eval_scratch);

        // Generate M samples: x_pred = current_x + drift + noise_sigma · ξ.
        let mut samples = Vec::with_capacity(M_SAMPLES);
        for _ in 0..M_SAMPLES {
            let xi = self.rng.gaussian(self.noise_sigma);
            samples.push(self.current_x + drift[0] + xi);
        }
        PredictiveOutput::from_samples(samples)
    }

    fn observe(&mut self, y: f32) {
        self.current_x = y;
    }
}

// ── The benchmark itself ──────────────────────────────────────────────────

/// Run the VFE-vs-floor comparison. Marked `#[ignore]` so it doesn't run on
/// default `cargo test` — invoke explicitly with `--ignored` (the test prints
/// a verdict table that should be captured to `.benchmarks/376_uq_floor.md`).
#[test]
#[ignore = "prints a VFE-vs-conformal-floor comparison table; run with --release --ignored"]
fn vfe_vs_conformal_floor_ar1() {
    // Build the test corpus: N_TRAIN + N_TEST steps of AR(1).
    let full = generate_ar1(N_TRAIN + N_TEST, PHI, SIGMA, SEED);
    // Skip the first N_TRAIN (used for fitting inside the adapter constructor).
    let test_corpus = &full[N_TRAIN..];

    let corpus = TrajectoryCorpus::from_slice(
        format!("ar1_phi{PHI}_sigma{SIGMA}_n{N_TEST}").as_str(),
        test_corpus,
        32, // warmup: seed the floor's residual pool before scoring
    );

    let mut adapter = VfeForecastAdapter::new();
    let report = run_floor_comparison(
        &mut adapter,
        &corpus.values,
        0.05, // alpha = 0.05 (95% interval)
        corpus.recommended_warmup,
        &corpus.name,
    );

    report.pretty_print();

    // The verdict is informational, not gating (the primitive makes NO UQ claim
    // today per Issue 038). We assert only that the comparison ran and produced
    // a verdict — the actual win/loss is captured in the benchmark doc.
    println!("\nPrimitive wins? {}", report.primitive_wins());
    println!("Not applicable?  {}", report.is_not_applicable());
}
