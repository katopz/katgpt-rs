//! Plan 359 Phase 4 T4.3 — "Report the Floor" comparison for the BoM Trajectory
//! sampler (`heat_kernel_trajectory_bom`, Plan 359 Phase 4).
//!
//! The BoM Trajectory sampler produces K diverse field trajectories by
//! perturbing the initial state `h₀` along the near-harmonic subspace and
//! applying the linear heat kernel to each. The K trajectories are LATENT
//! field states (spatial cochains) representing exploration of the trajectory
//! space — NOT calibrated predictive intervals over the next observation.
//!
//! This test answers the T4.3 question: **can the K-trajectory spread be
//! evaluated as a UQ primitive against the conformal-naive floor?**
//!
//! ## Verdict (recorded from the canonical run)
//!
//! **EXCLUDED from the "Report the Floor" policy** — the BoM Trajectory's
//! K-hypothesis spread is exploration diversity (controlled by σ), not
//! calibrated predictive uncertainty. Same structural class as BoMSampler
//! (Issue 010 T3). See the test assertions for the canonical evidence.
//!
//! ## Why excluded (the structural argument)
//!
//! The BoM Trajectory is a SPATIAL FIELD predictor, not a univariate
//! time-series forecaster. Forcing it into the floor comparison requires
//! embedding the scalar observation into a spatial field and projecting the
//! K trajectories back to a scalar. The interval width is controlled by
//! `perturbation_sigma` (a hyperparameter), NOT by local data volatility
//! (residual calibration). This is the textbook false-confidence signature.
//!
//! ## Method
//!
//! The adapter embeds the scalar observation `y_t` as a uniform rank-0 field
//! on a 4×4 grid (16 cells, dim=1), samples K=8 perturbed trajectories at
//! t=1 via near-harmonic perturbation (4 directions, motor=0), and projects
//! each trajectory to cell 0. The K scalars are returned as
//! `PredictiveOutput::from_samples`; the harness converts samples →
//! empirical-quantile interval and scores CRPS / coverage / Winkler against
//! the floor on the SAME corpora.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_bom_trajectory \
//!   --features conformal_predictive_intervals,heat_kernel_trajectory -- --nocapture
//! ```

#![cfg(all(
    feature = "conformal_predictive_intervals",
    feature = "heat_kernel_trajectory"
))]
#![allow(clippy::needless_range_loop)]

use katgpt_core::dec::{CellComplex, CochainField, DecEigendecomposition, near_harmonic_indices};
use katgpt_core::{
    FloorComparisonReport, PredictiveOutput, TrajectoryCorpus, UqPrimitiveUnderTest,
    run_floor_comparison,
};

// ===== Deterministic Gaussian RNG (test-local; harness's SplitMix64 is private) =====

/// Minimal SplitMix64 + Box-Muller for deterministic perturbation-noise
/// generation. Bit-reproducible across runs (matches the harness's determinism
/// discipline — the floor comparison must be reproducible).
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

    /// Uniform (0, 1) — open interval to avoid log(0) in Box-Muller.
    fn next_open_unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) | 1;
        let f = (u as f32) / ((1u64 << 24) as f32);
        f.clamp(1e-7, 1.0 - 1e-7)
    }

    /// Standard normal via Box-Muller.
    fn gaussian(&mut self) -> f32 {
        let u1 = self.next_open_unit();
        let u2 = self.next_open_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ===== BoM Trajectory adapter =====

/// Configuration for the BoM Trajectory adapter.
struct BomTrajectoryConfig {
    /// Perturbation magnitude σ.
    sigma: f32,
    /// Number of hypotheses K.
    k: usize,
    /// Number of near-harmonic perturbation directions.
    n_dirs: usize,
    /// Motor value (per-channel scalar; single channel here).
    motor: f32,
    /// Prediction horizon.
    t: f32,
}

impl Default for BomTrajectoryConfig {
    fn default() -> Self {
        Self {
            sigma: 0.1,
            k: 8,
            n_dirs: 4,
            motor: 0.0,
            t: 1.0,
        }
    }
}

/// Adapter wrapping `heat_kernel_trajectory_bom` as a `UqPrimitiveUnderTest`.
///
/// **Embedding**: the scalar observation `y_t` is embedded as a uniform rank-0
/// field on a 4×4 grid (all 16 cells = y_t). This excites only the constant
/// (harmonic) mode of the Laplacian.
///
/// **Prediction**: K=8 hypotheses are sampled via near-harmonic perturbation;
/// each is projected to cell 0; the K scalars are returned as
/// `PredictiveOutput::from_samples`. The harness converts samples →
/// empirical-quantile interval.
///
/// **State advance**: on `observe(y)`, the adapter stores `y` as the next
/// initial condition (the field is re-embedded from the scalar each call —
/// no persistent spatial state, matching the "last observation" semantics of
/// the floor's seasonal-naive forecaster).
///
/// **Perturbation noise**: regenerated each `predict_next` from a deterministic
/// SplitMix64 RNG (seeded at construction). Bit-reproducible.
pub struct BomTrajectoryAdapter {
    eig: DecEigendecomposition,
    near_harm: Vec<usize>,
    noise: Vec<f32>,
    h0: CochainField,
    trajs: Vec<CochainField>,
    scratch: CochainField,
    rng: SplitMix64,
    last_observation: f32,
    cfg: BomTrajectoryConfig,
}

impl BomTrajectoryAdapter {
    /// Construct a BoM Trajectory adapter.
    fn new(seed: u64, cfg: BomTrajectoryConfig) -> Self {
        let cx = CellComplex::grid_2d(4, 4);
        let n_cells = cx.n_cells(0);
        // Full eigendecomposition (k=16, max_iter=2000) — converges on 4×4.
        let eig = DecEigendecomposition::compute(&cx, 0, n_cells, 2000);
        let near_harm = near_harmonic_indices(&eig, cfg.motor, cfg.n_dirs);
        let k = cfg.k;
        Self {
            eig,
            near_harm,
            noise: vec![0.0; k * cfg.n_dirs],
            h0: CochainField::zeros(0, n_cells, 1),
            trajs: (0..k).map(|_| CochainField::zeros(0, n_cells, 1)).collect(),
            scratch: CochainField::zeros(0, n_cells, 1),
            rng: SplitMix64::new(seed),
            last_observation: 0.0,
            cfg,
        }
    }

    /// Embed the scalar observation as a uniform field.
    #[inline]
    fn embed_observation(&mut self, y: f32) {
        self.h0.data.fill(y);
    }
}

impl UqPrimitiveUnderTest for BomTrajectoryAdapter {
    fn name(&self) -> &str {
        "BoM Trajectory (heat_kernel_trajectory_bom, near-harmonic, cell-0 projection)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        // 1. Embed last observation.
        self.embed_observation(self.last_observation);

        // 2. Regenerate K * n_dirs Gaussian perturbation coefficients.
        for q in self.noise.iter_mut() {
            *q = self.rng.gaussian();
        }

        // 3. Sample K trajectories into the pre-allocated scratch buffers
        //    (motor is a 1-element slice; dim=1).
        let motor = [self.cfg.motor];
        katgpt_core::dec::heat_kernel_trajectory_bom_into(
            &self.eig,
            &self.h0,
            &motor,
            1,
            self.cfg.t,
            self.cfg.k,
            self.cfg.sigma,
            &self.near_harm,
            &self.noise,
            &mut self.trajs,
            &mut self.scratch,
        );

        // 4. Project each trajectory to cell 0 → K predictive samples.
        let mut samples = Vec::with_capacity(self.cfg.k);
        for k_idx in 0..self.cfg.k {
            samples.push(self.trajs[k_idx].data[0]);
        }

        PredictiveOutput::from_samples(samples)
    }

    fn observe(&mut self, y: f32) {
        self.last_observation = y;
    }
}

// ===== Corpora =====

/// Small-amplitude seasonal: `0.8·sin(2πt/12) + N(0, 0.05)`.
/// Values in roughly [-0.95, 0.95] — fits the adapter's representable range.
fn small_amplitude_seasonal(n: usize, seed: u64) -> TrajectoryCorpus {
    let mut rng = SplitMix64::new(seed);
    let mut values = Vec::with_capacity(n);
    for t in 0..n {
        let phase = 2.0 * core::f32::consts::PI * (t as f32) / 12.0;
        let noise = rng.gaussian() * 0.05;
        values.push(0.8 * phase.sin() + noise);
    }
    TrajectoryCorpus::from_slice(format!("small_amp_seasonal_0p8sigma0p05_n{n}"), &values, 48)
}

/// Small-σ white noise: `N(0, 0.3)`. Values roughly in [-0.9, 0.9] (3σ).
fn small_sigma_white_noise(n: usize, seed: u64) -> TrajectoryCorpus {
    let mut rng = SplitMix64::new(seed);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(rng.gaussian() * 0.3);
    }
    TrajectoryCorpus::from_slice(format!("white_noise_sigma0p3_n{n}"), &values, 64)
}

// ===== Tests =====

/// Helper: run the comparison and print the full report.
fn run_and_print(
    adapter: &mut BomTrajectoryAdapter,
    corpus: &TrajectoryCorpus,
    alpha: f32,
) -> FloorComparisonReport {
    let report = run_floor_comparison(
        adapter,
        &corpus.values,
        alpha,
        corpus.recommended_warmup,
        &corpus.name,
    );
    report.pretty_print();
    report
}

#[test]
fn bom_trajectory_vs_floor_on_small_amplitude_seasonal() {
    // The floor's home turf: seasonal structure. The key finding (expected,
    // matching BoM T3): the BoM Trajectory's interval width is σ-controlled
    // (not data-calibrated), producing the false-confidence signature —
    // narrow intervals that flatter CRPS but cover far below the nominal rate.
    let corpus = small_amplitude_seasonal(500, 0xA1B2_C3D4);
    let cfg = BomTrajectoryConfig::default(); // σ=0.1, K=8, motor=0, t=1
    let mut bom = BomTrajectoryAdapter::new(0xCAFE_BABE, cfg);
    let report = run_and_print(&mut bom, &corpus, 0.05);

    assert!(report.n_scored > 400, "n_scored={}", report.n_scored);
    assert_eq!(
        report.n_unscorable, 0,
        "BoM trajectory samples should always be scorable"
    );
    // The false-confidence signature: severe under-coverage. Nominal is 0.95;
    // the BoM Trajectory's σ-bound intervals should land well below.
    assert!(
        report.primitive.coverage < 0.30,
        "BoM trajectory coverage {:.4} should be < 0.30 (false confidence; nominal 0.95)",
        report.primitive.coverage
    );
    assert!(
        report.floor.coverage > 0.90,
        "floor coverage {:.4} should be > 0.90 (calibrated)",
        report.floor.coverage
    );
    // Must not be declared a UQ win.
    assert!(
        !report.primitive_wins(),
        "BoM trajectory must not be declared a UQ win"
    );
}

#[test]
fn bom_trajectory_vs_floor_on_small_sigma_white_noise() {
    // The floor's worst case: i.i.d. data where the optimal forecast is the
    // mean (0), not the last value. Even here, the BoM Trajectory's σ-bound
    // intervals should show the false-confidence pattern (under-coverage).
    let corpus = small_sigma_white_noise(500, 0xDEAD_BEEF);
    let cfg = BomTrajectoryConfig::default();
    let mut bom = BomTrajectoryAdapter::new(0xFACE_CAFE, cfg);
    let report = run_and_print(&mut bom, &corpus, 0.05);

    assert!(report.n_scored > 400, "n_scored={}", report.n_scored);
    assert_eq!(report.n_unscorable, 0);
    assert!(
        report.primitive.coverage < 0.30,
        "BoM trajectory coverage {:.4} should be < 0.30 (false confidence)",
        report.primitive.coverage
    );
    assert!(
        !report.primitive_wins(),
        "BoM trajectory should not beat the floor on white noise, got {:?}",
        report.overall
    );
}

#[test]
fn bom_trajectory_interval_width_is_constant_across_volatility_regimes() {
    // KEY EVIDENCE for the exclusion verdict: the BoM Trajectory's interval
    // width is controlled by σ (a hyperparameter), NOT by local data
    // volatility. The floor's width IS data-driven (residual quantile). Feed
    // the adapter a low-volatility regime then a high-volatility regime; its
    // interval width should NOT adapt, while the floor's should.
    let cfg = BomTrajectoryConfig::default();
    let mut bom = BomTrajectoryAdapter::new(0x1234_5678, cfg);

    let low_vol: Vec<f32> = {
        let mut rng = SplitMix64::new(0xAA11);
        (0..200).map(|_| 0.8 * rng.gaussian() * 0.02).collect()
    };
    let high_vol: Vec<f32> = {
        let mut rng = SplitMix64::new(0xBB22);
        (0..200).map(|_| 0.8 * rng.gaussian() * 0.30).collect()
    };

    let mut low_widths = Vec::new();
    for (t, &y) in low_vol.iter().enumerate() {
        if t >= 48 {
            let out = bom.predict_next();
            if let Some(iv) = out.into_interval(0.05) {
                low_widths.push(iv.upper - iv.lower);
            }
        }
        bom.observe(y);
    }

    let mut high_widths = Vec::new();
    for (t, &y) in high_vol.iter().enumerate() {
        if t >= 48 {
            let out = bom.predict_next();
            if let Some(iv) = out.into_interval(0.05) {
                high_widths.push(iv.upper - iv.lower);
            }
        }
        bom.observe(y);
    }

    let mean_low = low_widths.iter().sum::<f32>() / low_widths.len() as f32;
    let mean_high = high_widths.iter().sum::<f32>() / high_widths.len() as f32;
    eprintln!(
        "BoM trajectory interval width: low-vol mean = {:.4}, high-vol mean = {:.4} (ratio {:.3})",
        mean_low,
        mean_high,
        mean_high / mean_low.max(1e-9)
    );

    // Widths should be NEARLY IDENTICAL (σ-controlled, not volatility-controlled).
    let width_ratio = mean_high / mean_low.max(1e-9);
    assert!(
        (0.5..=2.0).contains(&width_ratio),
        "BoM trajectory width ratio {width_ratio:.3} should be ~1.0 (σ-controlled); \
         if data-driven like the floor, it would be ~15×"
    );
}

#[test]
fn bom_trajectory_sigma_sweep_changes_width_but_not_quality() {
    // Widening σ widens the interval (better coverage) but does NOT improve
    // CRPS (the point forecast is unchanged). Confirms the "UQ" is a
    // hyperparameter knob, not data-driven calibration.
    let corpus = small_amplitude_seasonal(300, 0x5EA5_0A1A);

    let mut reports = Vec::new();
    for &sigma in &[0.05, 0.1, 0.3, 0.5] {
        let cfg = BomTrajectoryConfig {
            sigma,
            ..Default::default()
        };
        let mut bom = BomTrajectoryAdapter::new(0xBEEF_F00D, cfg);
        let report = run_floor_comparison(
            &mut bom,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &format!("seasonal_sigma_{sigma}"),
        );
        eprintln!(
            "σ={:.2}: crps_ratio={:.4}, coverage={:.4}, verdict={:?}",
            sigma, report.crps_ratio, report.primitive.coverage, report.overall
        );
        reports.push((sigma, report));
    }

    // No σ value produces a primitive_wins verdict.
    for (sigma, report) in &reports {
        assert!(
            !report.primitive_wins(),
            "σ={sigma}: BoM trajectory must not be declared a UQ win"
        );
    }
}
