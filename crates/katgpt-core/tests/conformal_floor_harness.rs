//! Issue 010 T2 — "Report the Floor" comparison harness integration tests.
//!
//! These tests exercise the harness end-to-end on multiple scenarios and serve
//! as the canonical usage example for T3–T7 adapter authors (BoMSampler,
//! Sleep-Time Anticipator, Best-Belief Beta Selector, Alien Sampler).
//!
//! ## What's covered
//!
//! 1. **Sanity**: floor-vs-floor ties.
//! 2. **Decisive win**: a true oracle (peeks at next value) beats the floor.
//! 3. **Decisive loss**: an over-wide primitive loses to the floor.
//! 4. **Samples-only path**: a primitive producing samples (no interval) is
//!    scored via the empirical-quantile conversion.
//! 5. **NotApplicable path**: a primitive producing no output is excluded.
//! 6. **Realistic win**: a mean-forecaster beats the floor on white noise
//!    (where the optimal forecast is the mean, not the last value — the floor
//!    is worst-case here).
//! 7. **Multi-corpus sweep**: the same primitive evaluated across multiple
//!    corpora (the standard "Report the Floor" evaluation protocol).
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_harness \
//!   --features conformal_predictive_intervals
//! ```

#![cfg(feature = "conformal_predictive_intervals")]

use katgpt_core::{
    FloorAdapter, OverallVerdict, PredictiveInterval, PredictiveOutput,
    TrajectoryCorpus, UqPrimitiveUnderTest, run_floor_comparison,
};

// ===== Test primitives (mirror the unit-test helpers, but exposed here as the
//       canonical adapter pattern for T3-T7 authors to copy). =====

/// A TRUE oracle — peeks at the upcoming corpus value. Decisively beats the
/// floor. NOT a realistic primitive (no real forecaster sees the future); used
/// only to validate the harness can detect a win.
pub struct TrueOracle<'a> {
    pub corpus: &'a [f32],
    pub step: usize,
}

impl<'a> UqPrimitiveUnderTest for TrueOracle<'a> {
    fn name(&self) -> &str { "true-oracle" }
    fn predict_next(&mut self) -> PredictiveOutput {
        let next_y = self.corpus.get(self.step).copied().unwrap_or(0.0);
        let eps = 1e-6;
        PredictiveOutput::from_interval(PredictiveInterval::new(
            next_y - eps, next_y, next_y + eps, 0.05,
        ))
    }
    fn observe(&mut self, _y: f32) { self.step += 1; }
}

/// An over-wide primitive — always predicts ±10. Decisively loses to the
/// floor on CRPS/Winkler.
pub struct OverWide;

impl UqPrimitiveUnderTest for OverWide {
    fn name(&self) -> &str { "over-wide (±10)" }
    fn predict_next(&mut self) -> PredictiveOutput {
        PredictiveOutput::from_interval(PredictiveInterval::new(-10.0, 0.0, 10.0, 0.05))
    }
    fn observe(&mut self, _y: f32) {}
}

/// A mean-tracking primitive — predicts the running mean with a ±2σ interval.
/// On white noise, this BEATS the floor (whose seasonal-naive forecast is the
/// last observation, the worst possible predictor for i.i.d. data). On seasonal
/// data, this LOSES to the floor (the mean misses the seasonal structure).
pub struct MeanTracker {
    pub n: f32,
    pub mean: f32,
    pub m2: f32, // sum of squared deviations (Welford)
}

impl Default for MeanTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MeanTracker {
    pub fn new() -> Self { Self { n: 0.0, mean: 0.0, m2: 0.0 } }
}

impl UqPrimitiveUnderTest for MeanTracker {
    fn name(&self) -> &str { "mean-tracker (running mean ± 2σ)" }
    fn predict_next(&mut self) -> PredictiveOutput {
        let sigma = if self.n > 1.0 {
            (self.m2 / (self.n - 1.0)).sqrt()
        } else {
            1.0
        };
        let half_width = 2.0 * sigma.max(1e-6);
        PredictiveOutput::from_interval(PredictiveInterval::new(
            self.mean - half_width,
            self.mean,
            self.mean + half_width,
            0.05,
        ))
    }
    fn observe(&mut self, y: f32) {
        // Welford's online mean + variance.
        self.n += 1.0;
        let delta = y - self.mean;
        self.mean += delta / self.n;
        let delta2 = y - self.mean;
        self.m2 += delta * delta2;
    }
}

/// A samples-only primitive — produces 41 samples around the last value, no
/// explicit interval. Tests the samples → interval conversion.
pub struct SamplesAroundLast {
    pub last_y: f32,
}

impl UqPrimitiveUnderTest for SamplesAroundLast {
    fn name(&self) -> &str { "samples-around-last" }
    fn predict_next(&mut self) -> PredictiveOutput {
        let mut samples = Vec::with_capacity(41);
        for i in 0..41u32 {
            let offset = (i as f32 - 20.0) * 0.05;
            samples.push(self.last_y + offset);
        }
        PredictiveOutput::from_samples(samples)
    }
    fn observe(&mut self, y: f32) { self.last_y = y; }
}

/// An empty primitive — produces nothing. Excluded via NotApplicable.
pub struct Empty;

impl UqPrimitiveUnderTest for Empty {
    fn name(&self) -> &str { "empty" }
    fn predict_next(&mut self) -> PredictiveOutput { PredictiveOutput::empty() }
    fn observe(&mut self, _y: f32) {}
}

// ===== Tests =====

#[test]
fn floor_vs_floor_ties_on_seasonal() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 500, 0xA1B2_C3D4);
    let mut prim = FloorAdapter::new(0.05);
    let report = run_floor_comparison(
        &mut prim,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    assert!(report.n_scored > 400, "n_scored={}", report.n_scored);
    assert!(
        (report.crps_ratio - 1.0).abs() < 0.05,
        "floor-vs-floor crps_ratio {} should be ~1.0",
        report.crps_ratio
    );
    // Floor-vs-floor should tie (or trivially beat itself by RNG noise).
    assert!(
        matches!(report.overall, OverallVerdict::TiesFloor | OverallVerdict::BeatsFloor),
        "floor-vs-floor should tie or beat, got {:?}",
        report.overall
    );
}

#[test]
fn true_oracle_beats_floor_on_seasonal() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 300, 0xFACE_CAFE);
    let mut oracle = TrueOracle { corpus: &corpus.values, step: 0 };
    let report = run_floor_comparison(
        &mut oracle,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    assert!(report.n_scored > 200);
    assert!(
        report.crps_ratio < 0.01,
        "true-oracle crps_ratio {} should be < 0.01",
        report.crps_ratio
    );
    assert_eq!(report.overall, OverallVerdict::BeatsFloor);
    assert!(report.primitive_wins());
}

#[test]
fn overwide_loses_to_floor_on_seasonal() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 300, 0xDEAD_BEEF);
    let mut wide = OverWide;
    let report = run_floor_comparison(
        &mut wide,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    assert!(report.crps_ratio > 5.0, "over-wide crps_ratio {}", report.crps_ratio);
    assert_eq!(report.overall, OverallVerdict::LosesToFloor);
}

#[test]
fn samples_only_primitive_is_scorable_on_seasonal() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 300, 0xBA5E_BA11);
    let mut samp = SamplesAroundLast { last_y: 0.0 };
    let report = run_floor_comparison(
        &mut samp,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    assert_eq!(report.n_unscorable, 0, "samples-only should be scorable");
    assert!(report.n_scored > 200);
    // Samples-around-last is essentially seasonal-naive with a fixed spread;
    // it should roughly tie the floor (both predict near the last value).
    assert!(
        (report.crps_ratio - 1.0).abs() < 0.5,
        "samples-only crps_ratio {} should be near 1.0 (±0.5)",
        report.crps_ratio
    );
}

#[test]
fn empty_primitive_is_not_applicable() {
    let corpus = TrajectoryCorpus::white_noise(1.0, 100, 0x1234_5678);
    let mut empty = Empty;
    let report = run_floor_comparison(
        &mut empty,
        &corpus.values,
        0.05,
        10,
        &corpus.name,
    );
    assert_eq!(report.n_scored, 0);
    assert!(report.is_not_applicable());
}

#[test]
fn mean_tracker_beats_floor_on_white_noise() {
    // KEY TEST: on i.i.d. white noise, the optimal forecast is the MEAN, not
    // the last value. The floor (seasonal-naive = last value) is worst-case
    // here. The mean-tracker should decisively beat the floor.
    let corpus = TrajectoryCorpus::white_noise(1.0, 1000, 0xCAFE_BABE);
    let mut mt = MeanTracker::new();
    let report = run_floor_comparison(
        &mut mt,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    assert!(report.n_scored > 800);
    // Mean-tracker CRPS should be ~half the floor's (variance of the mean
    // shrinks as 1/n, while the floor's residual variance stays at ~2σ²).
    assert!(
        report.crps_ratio < 0.9,
        "mean-tracker crps_ratio {} should be < 0.9 on white noise",
        report.crps_ratio
    );
    assert!(
        matches!(report.overall, OverallVerdict::BeatsFloor | OverallVerdict::Mixed),
        "mean-tracker should beat or mix on white noise, got {:?}",
        report.overall
    );
}

#[test]
fn mean_tracker_loses_to_floor_on_seasonal() {
    // Counterpart: on seasonal data, the mean-tracker misses the seasonal
    // structure entirely (it predicts the global mean for every step). The
    // floor (seasonal-naive) captures the structure and should win.
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.1, 500, 0x5EA5_0A1A);
    let mut mt = MeanTracker::new();
    let report = run_floor_comparison(
        &mut mt,
        &corpus.values,
        0.05,
        corpus.recommended_warmup,
        &corpus.name,
    );
    // The mean-tracker's residual includes the full seasonal swing (amplitude
    // ~2.0 since sin ranges [-1,1]), so its interval must be wide (~±4) to
    // cover. The floor's residual is just the noise (σ=0.1), so its interval
    // is narrow (~±0.4). Floor CRPS << mean-tracker CRPS.
    assert!(
        report.crps_ratio > 2.0,
        "mean-tracker crps_ratio {} should be > 2.0 on seasonal data (floor wins)",
        report.crps_ratio
    );
}

#[test]
fn multi_corpus_sweep_reports_consistent_verdicts() {
    // The standard "Report the Floor" protocol: evaluate the same primitive
    // across multiple corpora. A primitive that beats on ALL corpora is a
    // robust win; one that beats on some but loses on others is mixed.
    let corpora = vec![
        TrajectoryCorpus::stationary_seasonal(12, 0.5, 400, 0x1111_1111),
        TrajectoryCorpus::stationary_seasonal(24, 0.5, 400, 0x2222_2222),
        TrajectoryCorpus::white_noise(1.0, 400, 0x3333_3333),
    ];

    let mut wins = 0;
    let mut losses = 0;
    let mut _ties = 0;
    for corpus in &corpora {
        let mut mt = MeanTracker::new();
        let report = run_floor_comparison(
            &mut mt,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        match report.overall {
            OverallVerdict::BeatsFloor => wins += 1,
            OverallVerdict::LosesToFloor => losses += 1,
            OverallVerdict::TiesFloor => _ties += 1,
            _ => {}
        }
    }
    // Mean-tracker beats on white noise, loses on seasonal. Expect at least
    // one win and at least one loss across the sweep.
    assert!(wins >= 1, "should beat on at least one corpus (white noise)");
    assert!(losses >= 1, "should lose on at least one corpus (seasonal)");
}

#[test]
fn report_pretty_print_survives_all_verdict_types() {
    // Smoke: pretty_print should not panic on any verdict variant.
    let corpora = [
        TrajectoryCorpus::stationary_seasonal(12, 0.5, 200, 0xAA11),
        TrajectoryCorpus::white_noise(1.0, 200, 0xBB22),
    ];

    // Oracle → BeatsFloor
    let corpus = &corpora[0];
    let mut oracle = TrueOracle { corpus: &corpus.values, step: 0 };
    let r1 = run_floor_comparison(&mut oracle, &corpus.values, 0.05, corpus.recommended_warmup, &corpus.name);
    r1.pretty_print();
    assert_eq!(r1.overall, OverallVerdict::BeatsFloor);

    // Over-wide → LosesToFloor
    let mut wide = OverWide;
    let r2 = run_floor_comparison(&mut wide, &corpus.values, 0.05, corpus.recommended_warmup, &corpus.name);
    r2.pretty_print();
    assert_eq!(r2.overall, OverallVerdict::LosesToFloor);

    // Empty → NotApplicable
    let mut empty = Empty;
    let r3 = run_floor_comparison(&mut empty, &corpus.values, 0.05, 10, &corpus.name);
    r3.pretty_print();
    assert!(r3.is_not_applicable());
}

#[test]
fn floor_adapter_alpha_propagates_to_intervals() {
    // The floor adapter's alpha should match the alpha passed to
    // run_floor_comparison. Verify by checking the interval's alpha field.
    let corpus = TrajectoryCorpus::white_noise(1.0, 200, 0xA1A_A111);
    let mut floor = FloorAdapter::new(0.10); // 90% interval
    // Warm up.
    for &y in corpus.values.iter().take(100) {
        floor.observe(y);
    }
    let out = floor.predict_next();
    let iv = out.interval.expect("floor should produce interval");
    assert!(
        (iv.alpha - 0.10).abs() < 1e-6,
        "floor interval alpha {} should be 0.10",
        iv.alpha
    );
}
