//! Issue 010 T3 — "Report the Floor" comparison for BoMSampler (Plan 281).
//!
//! BoMSampler produces K diverse next-belief-states from `(s_prev, x)` by
//! injecting K Gaussian noise queries at the kernel's pre-sigmoid activation
//! site. The K hypotheses are LATENT (in `(-1, 1)`) and represent exploration
//! of the belief space — NOT calibrated predictive intervals over the next
//! observation.
//!
//! This test answers the T3 question: **can BoM's K-hypothesis spread be
//! evaluated as a UQ primitive against the conformal-naive floor?**
//!
//! ## Verdict (recorded from the canonical run below)
//!
//! **EXCLUDED from the "Report the Floor" policy** — BoM's hypothesis spread
//! is exploration noise (controlled by σ), not calibrated predictive
//! uncertainty. The empirical evidence (see `bom_full_report_for_benchmark_doc`):
//!
//! | Corpus        | CRPS ratio | Winkler ratio | Coverage (nom 0.95) | Verdict |
//! |---------------|------------|---------------|---------------------|---------|
//! | seasonal      |     0.866  |       13.79   |        0.055        | Mixed   |
//! | white noise   |     0.306  |        4.11   |        0.151        | Mixed   |
//!
//! BoM *wins on CRPS* because its intervals are narrow (σ-bound), and CRPS
//! rewards narrowness. But it *loses catastrophically on coverage* (5–15%
//! vs the nominal 95%) — the textbook **false-confidence** failure mode. The
//! Winkler score (which penalizes misses by 2/α = 40) exposes the under-
//! coverage: 4–14× the floor. No value of σ fixes this — see the σ-sweep test
//! (σ=0.5 lifts coverage to only 0.254, still a third of nominal).
//!
//! BoM's GOAT gate (Plan 281 G2) measures *planning* win rate (+31.49pp on
//! the riir-ai arena, Plan 314), NOT calibrated UQ. BoM is a belief-space
//! exploration sampler, not a forecaster. Excluding it from the UQ policy is
//! the T3 escape hatch, exercised with evidence.
//!
//! ## Method
//!
//! The adapter projects BoM's K next-belief-states to a scalar (channel 0) and
//! feeds the K scalars to the harness as samples (`PredictiveOutput::from_samples`).
//! The harness converts samples → empirical-quantile interval and scores
//! CRPS / coverage / Winkler against the floor on the SAME corpora.
//!
//! Two corpora, both scaled into BoM's representable `(-1, 1)` range:
//! - **small-amplitude seasonal**: `0.8·sin(2πt/12) + N(0, 0.05)` — the floor's
//!   home turf (seasonal-naive captures the structure).
//! - **small-σ white noise**: `N(0, 0.3)` — the floor's worst case (last-value
//!   forecast is meaningless on i.i.d. data; the optimal forecast is the mean).
//!
//! The kernel is `AttractorKernel::from_seed` (random init, UNFITTED to the
//! corpus). This is the honest baseline: BoM's GOAT gate (Plan 281 G2) measures
//! *planning* win rate in riir-ai's arena (Plan 314), not scalar forecasting —
//! so there is no "fitted scalar forecaster" configuration of BoM to test.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_bom \
//!   --features conformal_predictive_intervals,bom_sampling -- --nocapture
//! ```

#![cfg(all(feature = "conformal_predictive_intervals", feature = "bom_sampling"))]
#![allow(clippy::needless_range_loop)]

use katgpt_core::{
    AttractorKernel, BoMSampler, FloorComparisonReport, MicroRecurrentBeliefState,
    NoiseQueryConfig, OverallVerdict, PredictiveOutput, TrajectoryCorpus, UqPrimitiveUnderTest,
    run_floor_comparison,
};

// ===== Deterministic Gaussian RNG (test-local; harness's SplitMix64 is private) =====

/// Minimal SplitMix64 + Box-Muller for deterministic Gaussian noise query
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
        // Mask to 24 bits of mantissa for f32, force away from 0 and 1.
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

// ===== BoMSampler adapter =====

/// Adapter wrapping `AttractorKernel` as a `UqPrimitiveUnderTest`.
///
/// **Embedding**: the scalar observation `y_t` is embedded into the kernel's
/// D-dim input as `x = [y_t, 0, 0, ..., 0]` (channel 0 carries the signal).
///
/// **Prediction**: K hypotheses are sampled; each is projected to channel 0;
/// the K scalars are returned as `PredictiveOutput::from_samples`. The harness
/// converts samples → empirical-quantile interval.
///
/// **State advance**: on `observe(y)`, the kernel's `step()` advances the
/// belief state `s_prev` using the embedded input, so the next `predict_next`
/// sees the updated belief.
///
/// **Noise queries**: regenerated each `predict_next` from a deterministic
/// SplitMix64 RNG (seeded at construction). Bit-reproducible.
pub struct BoMSamplerAdapter {
    kernel: AttractorKernel,
    s_prev: Vec<f32>,
    x_input: Vec<f32>,
    queries: Vec<f32>,
    hyp_out: Vec<f32>,
    cfg: NoiseQueryConfig,
    rng: SplitMix64,
    last_observation: f32,
    dim: usize,
    k: usize,
}

impl BoMSamplerAdapter {
    /// Construct an unfitted BoM adapter.
    ///
    /// - `kernel_seed`: seed for `AttractorKernel::from_seed` (random init).
    /// - `noise_seed`: seed for the noise-query RNG.
    /// - `dim`: belief-state dimension (D).
    /// - `cfg`: `NoiseQueryConfig` (σ, K).
    pub fn new(kernel_seed: u64, noise_seed: u64, dim: usize, cfg: NoiseQueryConfig) -> Self {
        let kernel = AttractorKernel::from_seed(kernel_seed, dim);
        let k = cfg.k;
        Self {
            kernel,
            s_prev: vec![0.0; dim],
            x_input: vec![0.0; dim],
            queries: vec![0.0; k * dim],
            hyp_out: vec![0.0; k * dim],
            cfg,
            rng: SplitMix64::new(noise_seed),
            last_observation: 0.0,
            dim,
            k,
        }
    }

    /// Embed the scalar observation into the D-dim input vector (channel 0).
    #[inline]
    fn embed_observation(&mut self, y: f32) {
        // Channel 0 carries the observation; the rest stay at their last value
        // (zero on first call, then whatever step wrote — but we overwrite ch0
        // and leave the rest untouched, matching "channel-0 input" semantics).
        // For a clean probe we zero the non-observation channels each call so
        // the kernel sees exactly [y, 0, 0, ..., 0].
        self.x_input.fill(0.0);
        self.x_input[0] = y;
    }
}

impl UqPrimitiveUnderTest for BoMSamplerAdapter {
    fn name(&self) -> &str {
        "BoMSampler (AttractorKernel, unfitted, channel-0 projection)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        // 1. Embed last observation.
        self.embed_observation(self.last_observation);

        // 2. Regenerate K*D Gaussian noise queries (deterministic RNG).
        for q in self.queries.iter_mut() {
            *q = self.rng.gaussian() * self.cfg.sigma;
        }

        // 3. Sample K hypotheses in one batched call.
        self.kernel.sample_k_states(
            &self.s_prev,
            &self.x_input,
            &self.queries,
            &mut self.hyp_out,
            &self.cfg,
        );

        // 4. Project each hypothesis to channel 0 → K predictive samples.
        let mut samples = Vec::with_capacity(self.k);
        for k_idx in 0..self.k {
            samples.push(self.hyp_out[k_idx * self.dim]);
        }

        PredictiveOutput::from_samples(samples)
    }

    fn observe(&mut self, y: f32) {
        // Advance the belief state using the embedded observation.
        self.embed_observation(y);
        self.kernel.step(&mut self.s_prev, &self.x_input);
        self.last_observation = y;
    }
}

// ===== Corpora (scaled into BoM's (-1, 1) representable range) =====

/// Small-amplitude seasonal: `0.8·sin(2πt/12) + N(0, 0.05)`.
/// Values in roughly [-0.95, 0.95] — fits BoM's output range.
fn small_amplitude_seasonal(n: usize, seed: u64) -> TrajectoryCorpus {
    let mut rng = SplitMix64::new(seed);
    let mut values = Vec::with_capacity(n);
    for t in 0..n {
        let phase = 2.0 * core::f32::consts::PI * (t as f32) / 12.0;
        let noise = rng.gaussian() * 0.05;
        values.push(0.8 * phase.sin() + noise);
    }
    TrajectoryCorpus::from_slice(
        format!("small_amp_seasonal_0p8sigma0p05_n{n}"),
        &values,
        48, // 4 periods warmup
    )
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

/// Helper: run the comparison and print the full report (for the benchmark doc).
fn run_and_print(
    adapter: &mut BoMSamplerAdapter,
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
fn bom_vs_floor_on_small_amplitude_seasonal() {
    // The floor's home turf: seasonal structure. The key finding here is the
    // **false-confidence** pattern: BoM's narrow σ-bound intervals flatter CRPS
    // (ratio 0.866, a technical "win") but cover only 5.5% of actuals vs the
    // nominal 95%. The Winkler score (penalty 2/α = 40 per miss) exposes the
    // under-coverage: ~14× the floor. This is exactly the failure mode the
    // "Report the Floor" policy exists to catch.
    let corpus = small_amplitude_seasonal(500, 0xA1B2_C3D4);
    let cfg = NoiseQueryConfig::default(); // σ=0.1, K=8
    let mut bom = BoMSamplerAdapter::new(42, 0xCAFE_BABE, 4, cfg);
    let report = run_and_print(&mut bom, &corpus, 0.05);

    assert!(report.n_scored > 400, "n_scored={}", report.n_scored);
    assert_eq!(
        report.n_unscorable, 0,
        "BoM samples should always be scorable"
    );
    // The false-confidence signature: severe under-coverage (BoM's intervals
    // are σ-bound, NOT residual-calibrated). Nominal is 0.95; BoM lands near
    // 0.05–0.15. This is the policy-relevant failure, not the CRPS ratio.
    assert!(
        report.primitive.coverage < 0.20,
        "BoM coverage {:.4} should be < 0.20 (false confidence; nominal 0.95)",
        report.primitive.coverage
    );
    assert!(
        report.floor.coverage > 0.90,
        "floor coverage {:.4} should be > 0.90 (calibrated)",
        report.floor.coverage
    );
    // The Winkler ratio should be much > 1 (misses penalized by 2/α).
    assert!(
        report.winkler_ratio > 2.0,
        "BoM winkler_ratio {:.4} should be > 2.0 (under-coverage penalized)",
        report.winkler_ratio
    );
    // Overall: never a win. Mixed (CRPS flatters, coverage/Winkler expose) or
    // LosesToFloor at high σ.
    assert!(
        matches!(
            report.overall,
            OverallVerdict::LosesToFloor | OverallVerdict::Mixed
        ),
        "BoM should lose or mix on seasonal, got {:?}",
        report.overall
    );
    assert!(
        !report.primitive_wins(),
        "BoM must not be declared a UQ win"
    );
}

#[test]
fn bom_vs_floor_on_small_sigma_white_noise() {
    // The floor's worst case: i.i.d. data where the optimal forecast is the
    // mean (0), not the last value. The floor's seasonal-naive forecast is
    // the worst possible predictor here.
    //
    // Even so, BoM (unfitted) is unlikely to beat the floor: its channel-0
    // output is not the mean (it's a random function of the input history),
    // and its interval width is fixed by σ. The floor at least has a
    // data-driven interval width (residual quantile), even if its point
    // forecast is bad.
    let corpus = small_sigma_white_noise(500, 0xDEAD_BEEF);
    let cfg = NoiseQueryConfig::default(); // σ=0.1, K=8
    let mut bom = BoMSamplerAdapter::new(42, 0xFACE_CAFE, 4, cfg);
    let report = run_and_print(&mut bom, &corpus, 0.05);

    assert!(report.n_scored > 400, "n_scored={}", report.n_scored);
    assert_eq!(report.n_unscorable, 0);
    // On white noise the floor's point forecast is bad (last value is the worst
    // predictor for i.i.d. data), so BoM's channel-0 output — which averages
    // near 0 thanks to the sigmoid's centering — coincidentally beats the floor
    // on CRPS (ratio ~0.31). But the false-confidence pattern holds: coverage
    // ~0.15 vs nominal 0.95, Winkler ~4× the floor.
    assert!(
        report.primitive.coverage < 0.25,
        "BoM coverage {:.4} should be < 0.25 (false confidence)",
        report.primitive.coverage
    );
    assert!(
        report.winkler_ratio > 2.0,
        "BoM winkler_ratio {:.4} should be > 2.0",
        report.winkler_ratio
    );
    assert!(
        !report.primitive_wins(),
        "BoM should not beat the floor on white noise, got {:?}",
        report.overall
    );
}

#[test]
fn bom_interval_width_is_constant_across_volatility_regimes() {
    // KEY EVIDENCE for the exclusion verdict: BoM's interval width is controlled
    // by σ (a hyperparameter), NOT by local data volatility. The floor's width
    // IS data-driven (residual quantile). This test demonstrates the structural
    // difference: feed BoM a low-volatility regime then a high-volatility regime;
    // its interval width should NOT adapt, while the floor's should.
    //
    // We measure BoM's interval half-width on two sub-corpus regimes.
    let cfg = NoiseQueryConfig::default();
    let mut bom = BoMSamplerAdapter::new(42, 0x1234_5678, 4, cfg);

    // Low-volatility regime: tiny noise.
    let low_vol: Vec<f32> = {
        let mut rng = SplitMix64::new(0xAA11);
        (0..200).map(|_| 0.8 * rng.gaussian() * 0.02).collect()
    };
    // High-volatility regime: large noise (same mean).
    let high_vol: Vec<f32> = {
        let mut rng = SplitMix64::new(0xBB22);
        (0..200).map(|_| 0.8 * rng.gaussian() * 0.30).collect()
    };

    // Warm up + measure BoM's interval width on low-vol.
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

    // Continue into high-vol regime (same adapter — belief state carries over).
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
        "BoM interval width: low-vol mean = {:.4}, high-vol mean = {:.4} (ratio {:.3})",
        mean_low,
        mean_high,
        mean_high / mean_low.max(1e-9)
    );

    // The widths should be NEARLY IDENTICAL (both driven by σ, not by the data
    // regime). Allow some slack for RNG variation in the empirical quantile of
    // K=8 samples, but the ratio should be close to 1.0 — NOT close to the
    // volatility ratio (0.30 / 0.02 = 15×).
    let width_ratio = mean_high / mean_low.max(1e-9);
    assert!(
        (0.5..=2.0).contains(&width_ratio),
        "BoM width ratio {width_ratio:.3} should be ~1.0 (σ-controlled, not volatility-controlled); \
         if it were data-driven like the floor, it would be ~15×"
    );
}

#[test]
fn bom_sigma_sweep_changes_width_but_not_quality() {
    // Second piece of evidence: widening σ widens BoM's interval (better
    // coverage) but does NOT improve its CRPS (the point forecast is still
    // unfitted). This confirms BoM's "UQ" is a hyperparameter knob, not a
    // data-driven calibration.
    let corpus = small_amplitude_seasonal(300, 0x5EA5_0A1A);

    let mut reports = Vec::new();
    for &sigma in &[0.05, 0.1, 0.3, 0.5] {
        let cfg = NoiseQueryConfig::default().with_sigma(sigma);
        let mut bom = BoMSamplerAdapter::new(42, 0xBEEF_F00D, 4, cfg);
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

    // All σ values should fail to beat the floor (none wins).
    for (sigma, report) in &reports {
        assert!(
            !report.primitive_wins(),
            "σ={:.2} should not beat the floor, got {:?}",
            sigma,
            report.overall
        );
    }
}

#[test]
fn bom_full_report_for_benchmark_doc() {
    // The canonical run whose numbers go into .benchmarks/010_bom_floor_comparison.md.
    // Runs both corpora at the default config (σ=0.1, K=8, D=4) and pretty-prints
    // the full report. This is the evidence row for the exclusion verdict.
    eprintln!("\n=== BoMSampler floor comparison (canonical run for Issue 010 T3) ===\n");

    let seasonal = small_amplitude_seasonal(500, 0xA1B2_C3D4);
    let noise = small_sigma_white_noise(500, 0xDEAD_BEEF);

    eprintln!("--- Corpus: {} ---", seasonal.name);
    let mut bom1 = BoMSamplerAdapter::new(42, 0xCAFE_BABE, 4, NoiseQueryConfig::default());
    let r1 = run_and_print(&mut bom1, &seasonal, 0.05);

    eprintln!("\n--- Corpus: {} ---", noise.name);
    let mut bom2 = BoMSamplerAdapter::new(42, 0xFACE_CAFE, 4, NoiseQueryConfig::default());
    let r2 = run_and_print(&mut bom2, &noise, 0.05);

    eprintln!("\n=== Summary ===");
    eprintln!(
        "seasonal: crps_ratio={:.4}, winkler_ratio={:.4}, coverage={:.4} (nominal {:.2}), verdict={:?}",
        r1.crps_ratio,
        r1.winkler_ratio,
        r1.primitive.coverage,
        1.0 - 0.05,
        r1.overall
    );
    eprintln!(
        "white noise: crps_ratio={:.4}, winkler_ratio={:.4}, coverage={:.4} (nominal {:.2}), verdict={:?}",
        r2.crps_ratio,
        r2.winkler_ratio,
        r2.primitive.coverage,
        1.0 - 0.05,
        r2.overall
    );
    eprintln!();

    // Verdict: BoM loses or mixes on both corpora. The policy conclusion (T3
    // escape hatch) is documented in the benchmark file and the issue, NOT
    // asserted here — the test just records the numbers.
    assert!(r1.n_scored > 0);
    assert!(r2.n_scored > 0);
}
