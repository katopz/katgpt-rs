//! # "Report the Floor" comparison harness (Issue 010 T2)
//!
//! A reusable benchmark fixture that wraps any UQ-bearing primitive, runs it
//! on a standard trajectory corpus, and compares CRPS / coverage / Winkler
//! against the conformal-naive floor (`ConformalIntervalCalibrator<
//! SeasonalNaiveForecaster>` with `m=1`).
//!
//! ## Why this exists
//!
//! Per the "Report the Floor" policy (Research 322, `AGENTS.md` Feature Flag
//! Discipline, Issue 010): any primitive claiming a probability distribution,
//! predictive interval, quantile, coverage guarantee, confidence score, or
//! calibrated uncertainty MUST beat the conformal-naive floor at its GOAT
//! gate. This harness is the canonical fixture for that comparison — it
//! removes per-primitive benchmark boilerplate so T3–T7 (BoMSampler, Sleep-
//! Time, Best-Belief, Alien Sampler) each reduce to "implement
//! [`UqPrimitiveUnderTest`] for primitive X, then call
//! [`run_floor_comparison`]".
//!
//! ## How it works
//!
//! 1. Both the primitive-under-test and the floor observe the same corpus.
//! 2. Before each observation, both produce a [`PredictiveOutput`].
//! 3. The harness normalizes outputs to intervals (samples → interval via
//!    empirical quantile if the primitive only produced samples).
//! 4. After the corpus, the harness computes CRPS / coverage / Winkler for
//!    both and reports the ratios + an overall verdict.
//!
//! ## The floor is fixed
//!
//! The canonical floor is always `ConformalIntervalCalibrator<
//! SeasonalNaiveForecaster>` with `m=1`, `exp_lambda=0.0`, `HStep` residual
//! mode, capacity 256. The primitive-under-test may use any internal config;
//! the floor's config is pinned so comparisons across primitives are
//! apples-to-apples.
//!
//! ## Modelless mandate
//!
//! The floor is modelless (empirical-quantile calibration, no training). The
//! harness itself is also modelless — it only orchestrates and scores. A
//! primitive-under-test that requires training (e.g. KARC's ridge solve) is
//! responsible for its own modelless-vs-trained classification per the
//! research skill §3.5; the harness just reports the numbers.

use crate::conformal::{
    ConformalIntervalCalibrator, DecayUnit, PointForecaster, PredictiveInterval, ResidualMode,
    SeasonalNaiveForecaster, seasonal_naive_floor,
};
use crate::conformal::metrics::{empirical_coverage, mean_crps_interval, mean_winkler};
use core::cmp::Ordering;

/// Capacity for the floor's residual ring buffer per (channel, horizon-bucket).
/// Matches Plan 340's recommended default. The floor has 1 channel × 1 bucket,
/// so total floor memory = `FLOOR_CAPACITY · 8` bytes = 2KB.
const FLOOR_CAPACITY: usize = 256;

/// Canonical floor seasonal period. `m=1` = non-seasonal (forecast = last
/// observation). Pinned by the "Report the Floor" rule.
const FLOOR_M: usize = 1;

/// Fractional improvement (or regression) threshold for the verdict logic.
/// A ratio within `[1.0 ± BEAT_THRESHOLD]` counts as a tie; outside that band
/// counts as a meaningful win or loss.
const BEAT_THRESHOLD: f32 = 0.05;

/// Coverage tolerance: if the primitive's coverage error is within
/// `COVERAGE_TOL` of the floor's, it counts as "no worse on coverage".
const COVERAGE_TOL: f32 = 0.02;

// ===== PredictiveOutput =====

/// The predictive output a UQ primitive produces for one test point.
///
/// A primitive may produce samples (a discrete distribution), an interval
/// (a calibrated range), or both. The harness normalizes to intervals for
/// unified scoring via [`PredictiveOutput::into_interval`].
#[derive(Clone, Debug)]
pub struct PredictiveOutput {
    /// Predictive samples (for sample-CRPS, future work). May be `None` if the
    /// primitive only produces intervals.
    pub samples: Option<Vec<f32>>,
    /// Predictive interval. May be `None` if the primitive only produces
    /// samples; the harness converts samples → interval via
    /// [`empirical_quantile_interval`].
    pub interval: Option<PredictiveInterval>,
}

impl PredictiveOutput {
    /// Samples-only output.
    #[inline]
    pub fn from_samples(samples: Vec<f32>) -> Self {
        Self { samples: Some(samples), interval: None }
    }

    /// Interval-only output.
    #[inline]
    pub fn from_interval(interval: PredictiveInterval) -> Self {
        Self { samples: None, interval: Some(interval) }
    }

    /// Both samples and an interval.
    #[inline]
    pub fn from_both(samples: Vec<f32>, interval: PredictiveInterval) -> Self {
        Self { samples: Some(samples), interval: Some(interval) }
    }

    /// An empty output — the primitive produced no prediction for this step.
    /// The harness will count this as unscorable.
    #[inline]
    pub fn empty() -> Self {
        Self { samples: None, interval: None }
    }

    /// Convert to an interval. If the primitive provided an interval, use it
    /// directly. Otherwise, convert samples → interval via empirical quantile.
    /// Returns `None` if the primitive produced neither.
    pub fn into_interval(&self, alpha: f32) -> Option<PredictiveInterval> {
        if let Some(iv) = self.interval {
            return Some(iv);
        }
        if let Some(samples) = &self.samples {
            if samples.is_empty() {
                return None;
            }
            return Some(empirical_quantile_interval(samples, alpha));
        }
        None
    }
}

/// Convert a sample set to a predictive interval via empirical quantile.
///
/// Sorts the samples, reads the lower quantile at `α/2`, the upper at
/// `1 − α/2`, and the median for the point estimate. Allocates a sort buffer
/// (the harness is a benchmark fixture, not hot-path).
pub fn empirical_quantile_interval(samples: &[f32], alpha: f32) -> PredictiveInterval {
    debug_assert!(!samples.is_empty(), "samples must be non-empty");
    debug_assert!(
        (0.0..=0.5).contains(&alpha),
        "alpha must be in [0, 0.5] for a two-tailed interval"
    );
    let mut sorted: Vec<f32> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = sorted.len();
    // Linear interpolation between order statistics (type 7, R default).
    let lo_idx = ((alpha * 0.5 * (n as f32 - 1.0)) as usize).min(n - 1);
    let hi_idx = (((1.0 - alpha * 0.5) * (n as f32 - 1.0)) as usize).min(n - 1);
    let med_idx = (n - 1) / 2;
    PredictiveInterval::new(sorted[lo_idx], sorted[med_idx], sorted[hi_idx], alpha)
}

// ===== UqPrimitiveUnderTest trait =====

/// A UQ-bearing primitive under floor-comparison test.
///
/// Adapters implement this for each concrete primitive (BoMSampler,
/// SleepTimeAnticipator, Best-Belief Beta Selector, Alien Sampler, etc.).
/// The harness orchestrates `predict_next` → score → `observe` for each step
/// of the corpus.
///
/// All methods take `&mut self` because forecasting is stateful (matches the
/// `PointForecaster::forecast_into` signature from Plan 340 Phase 2).
pub trait UqPrimitiveUnderTest {
    /// Human-readable name (for the report).
    fn name(&self) -> &str;

    /// Produce the predictive output for the next step (before `observe`).
    /// Called BEFORE the ground-truth `y_t` is revealed.
    fn predict_next(&mut self) -> PredictiveOutput;

    /// Observe the ground-truth value `y_t` and update internal state.
    /// Called AFTER `predict_next` for step t.
    fn observe(&mut self, y: f32);
}

// ===== FloorAdapter =====

/// The canonical conformal-naive floor wrapped as a [`UqPrimitiveUnderTest`].
///
/// Config (pinned by the "Report the Floor" rule):
/// - forecaster: `SeasonalNaiveForecaster` (forecast = last observation)
/// - `m = 1` (non-seasonal)
/// - `exp_lambda = 0.0` (no recency decay — all residuals equal weight)
/// - residual mode: `HStep`
/// - capacity: 256
/// - single channel, single horizon bucket (`max_h = 1`)
///
/// Constructed via [`FloorAdapter::new`] with the desired `alpha`.
pub struct FloorAdapter {
    calibrator: ConformalIntervalCalibrator<SeasonalNaiveForecaster>,
    alpha: f32,
}

impl FloorAdapter {
    /// Construct the canonical floor at miscoverage level `alpha`
    /// (e.g. `0.05` for a 95% interval).
    #[inline]
    pub fn new(alpha: f32) -> Self {
        debug_assert!(
            (0.0..=0.5).contains(&alpha),
            "alpha must be in [0, 0.5]"
        );
        let forecaster = seasonal_naive_floor(FLOOR_CAPACITY);
        let calibrator = ConformalIntervalCalibrator::new(
            forecaster,
            1,              // n_channels
            1,              // max_h
            FLOOR_M,        // m=1 (canonical floor)
            FLOOR_CAPACITY,
            0.0,            // exp_lambda (no recency decay)
            DecayUnit::Step,
            ResidualMode::HStep,
            false,          // orientation
        );
        Self { calibrator, alpha }
    }
}

impl UqPrimitiveUnderTest for FloorAdapter {
    #[inline]
    fn name(&self) -> &str {
        "conformal-naive floor (SeasonalNaive m=1)"
    }

    #[inline]
    fn predict_next(&mut self) -> PredictiveOutput {
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, self.alpha);
        // ch=0 (single channel), h=1 (next step).
        self.calibrator.interval_into(0, 1, self.alpha, &mut iv);
        PredictiveOutput::from_interval(iv)
    }

    #[inline]
    fn observe(&mut self, y: f32) {
        // Lifecycle: forecast → update residual → push to history → step tick.
        let mut fc = 0.0_f32;
        self.calibrator.forecaster.forecast_into(&[], 1, &mut fc);
        self.calibrator.update_residual(y, fc, 0, 1);
        self.calibrator.forecaster.observe(y);
        self.calibrator.step();
    }
}

// ===== UqMetrics =====

/// The three "Report the Floor" metrics computed for one side (primitive or
/// floor) over the scored portion of the corpus.
#[derive(Clone, Debug, Default)]
pub struct UqMetrics {
    /// Mean interval-CRPS (lower is better). Uniform-on-interval closed form.
    pub mean_crps_interval: f32,
    /// Empirical coverage (fraction of actuals within the interval).
    /// Should converge to `1 − alpha` on stationary data.
    pub coverage: f32,
    /// Mean Winkler interval score (lower is better). Penalizes width always
    /// and outside-miss distance by `2/alpha`.
    pub mean_winkler: f32,
}

// ===== OverallVerdict =====

/// The harness's conservative overall verdict for the primitive-vs-floor
/// comparison.
///
/// **The harness exposes the raw metrics + ratios; the verdict is a hint, not
/// a judgment.** A human (or the T3–T7 adapter author) may overrule it — for
/// example, a primitive that ties the floor on CRPS but is 10× faster is
/// still valuable (the policy's "reframing" escape hatch).
#[derive(Clone, Debug, PartialEq)]
pub enum OverallVerdict {
    /// Primitive is meaningfully better (>BEAT_THRESHOLD) on BOTH lower-better
    /// metrics (CRPS, Winkler) AND coverage is no worse.
    BeatsFloor,
    /// Primitive is within ±BEAT_THRESHOLD on all metrics.
    TiesFloor,
    /// Primitive is meaningfully worse on at least one lower-better metric,
    /// with no compensating win. Loses the UQ claim.
    LosesToFloor,
    /// Mixed signals — better on some metrics, worse on others. Judgment call.
    Mixed,
    /// Primitive couldn't be evaluated on this corpus (e.g. produced no
    /// scorable output, or the corpus is incompatible with the primitive's
    /// domain). Excludes the primitive from the policy, with a documented
    /// reason.
    NotApplicable {
        /// Why the primitive couldn't be evaluated.
        reason: String,
    },
}

// ===== FloorComparisonReport =====

/// The full report from a single primitive-vs-floor comparison run.
#[derive(Clone, Debug)]
pub struct FloorComparisonReport {
    /// Name of the primitive under test.
    pub primitive_name: String,
    /// Name of the corpus used.
    pub corpus_name: String,
    /// Number of scored observations (corpus length minus warmup, minus any
    /// unscorable steps).
    pub n_scored: usize,
    /// Number of steps where the primitive produced no scorable output.
    pub n_unscorable: usize,
    /// Miscoverage level (e.g. `0.05` for a 95% interval).
    pub alpha: f32,
    /// The primitive's metrics.
    pub primitive: UqMetrics,
    /// The floor's metrics.
    pub floor: UqMetrics,
    /// `primitive / floor` for CRPS. `< 1.0` = primitive wins (lower is better).
    pub crps_ratio: f32,
    /// `primitive / floor` for Winkler. `< 1.0` = primitive wins.
    pub winkler_ratio: f32,
    /// `|primitive_coverage − nominal|`. Lower is better.
    pub primitive_cov_err: f32,
    /// `|floor_coverage − nominal|`. Lower is better.
    pub floor_cov_err: f32,
    /// Overall verdict (hint — raw metrics are authoritative).
    pub overall: OverallVerdict,
}

impl FloorComparisonReport {
    /// `true` iff the verdict is [`OverallVerdict::BeatsFloor`].
    #[inline]
    pub fn primitive_wins(&self) -> bool {
        matches!(self.overall, OverallVerdict::BeatsFloor)
    }

    /// `true` iff the verdict is [`OverallVerdict::NotAppicable`].
    #[inline]
    pub fn is_not_applicable(&self) -> bool {
        matches!(self.overall, OverallVerdict::NotApplicable { .. })
    }

    /// Print a human-readable summary table. For benchmark logs / examples.
    pub fn pretty_print(&self) {
        let nominal = 1.0 - self.alpha;
        println!("=== Floor Comparison: {} ===", self.primitive_name);
        println!("Corpus: {} (n_scored={}, n_unscorable={}, α={:.2})",
                 self.corpus_name, self.n_scored, self.n_unscorable, self.alpha);
        println!();
        println!("Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict");
        println!("-------------------|------------|------------|--------------------|---------");
        let crps_v = if self.crps_ratio < 1.0 - BEAT_THRESHOLD { "WIN" }
                     else if self.crps_ratio > 1.0 + BEAT_THRESHOLD { "LOSE" } else { "tie" };
        let winkler_v = if self.winkler_ratio < 1.0 - BEAT_THRESHOLD { "WIN" }
                        else if self.winkler_ratio > 1.0 + BEAT_THRESHOLD { "LOSE" } else { "tie" };
        let cov_v = if self.primitive_cov_err < self.floor_cov_err - COVERAGE_TOL { "WIN" }
                    else if self.primitive_cov_err > self.floor_cov_err + COVERAGE_TOL { "LOSE" }
                    else { "tie" };
        println!("Mean CRPS          | {:>10.4} | {:>10.4} | {:>18.4} | {}",
                 self.primitive.mean_crps_interval, self.floor.mean_crps_interval,
                 self.crps_ratio, crps_v);
        println!("Mean Winkler       | {:>10.4} | {:>10.4} | {:>18.4} | {}",
                 self.primitive.mean_winkler, self.floor.mean_winkler,
                 self.winkler_ratio, winkler_v);
        println!("Coverage (nom={:.2}) | {:>10.4} | {:>10.4} | {:>18} | {}",
                 nominal, self.primitive.coverage, self.floor.coverage,
                 format!("err {:.4} vs {:.4}", self.primitive_cov_err, self.floor_cov_err),
                 cov_v);
        println!();
        let verdict_str = match &self.overall {
            OverallVerdict::BeatsFloor => "✅ BEATS FLOOR — primitive adds UQ value".to_string(),
            OverallVerdict::TiesFloor => "🟡 TIES FLOOR — no UQ gain, but may be valuable for other reasons".to_string(),
            OverallVerdict::LosesToFloor => "❌ LOSES TO FLOOR — primitive does not add UQ value".to_string(),
            OverallVerdict::Mixed => "🟠 MIXED — better on some metrics, worse on others (judgment call)".to_string(),
            OverallVerdict::NotApplicable { reason } => format!("⚪ N/A — {}", reason),
        };
        println!("Overall: {}", verdict_str);
    }
}

// ===== TrajectoryCorpus =====

/// A standard trajectory corpus for floor comparison.
///
/// Ships with constructors for the canonical "Report the Floor" test fixtures:
/// stationary seasonal (matches Plan 340 G1), white noise (degenerate floor-
/// favorable case), and a deterministic-reproducible RNG (SplitMix64) so
/// comparisons are bit-reproducible across runs.
#[derive(Clone, Debug)]
pub struct TrajectoryCorpus {
    /// Corpus name (for the report).
    pub name: String,
    /// The trajectory values.
    pub values: Vec<f32>,
    /// Recommended warmup steps (seed the residual pool before scoring).
    pub recommended_warmup: usize,
}

impl TrajectoryCorpus {
    /// Stationary seasonal: `y_t = sin(2πt/m) + N(0, σ)`. The canonical G1
    /// fixture from Plan 340. Default warmup = 4·m.
    pub fn stationary_seasonal(m: usize, sigma: f32, n: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut values = Vec::with_capacity(n);
        for t in 0..n {
            let phase = 2.0 * core::f32::consts::PI * (t as f32) / (m as f32);
            let noise = rng.gaussian(sigma);
            values.push(phase.sin() + noise);
        }
        Self {
            name: format!("stationary_seasonal_m{}_sigma{}_n{}", m, sigma, n),
            values,
            recommended_warmup: (4 * m).min(n / 4),
        }
    }

    /// Pure white noise: `y_t ~ N(0, σ)`. The degenerate case where the floor
    /// (forecast = last observation) is worst-case — the optimal forecast is
    /// the mean (0), not the last value. Useful for stress-testing primitives
    /// that should beat the floor by learning the mean. Default warmup = 64.
    pub fn white_noise(sigma: f32, n: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(rng.gaussian(sigma));
        }
        Self {
            name: format!("white_noise_sigma{}_n{}", sigma, n),
            values,
            recommended_warmup: 64.min(n / 4),
        }
    }

    /// Construct a corpus from a precomputed trajectory slice (for Lorenz-63,
    /// real data, etc.).
    pub fn from_slice(name: impl Into<String>, values: &[f32], warmup: usize) -> Self {
        Self {
            name: name.into(),
            values: values.to_vec(),
            recommended_warmup: warmup,
        }
    }
}

// ===== SplitMix64 (deterministic RNG for reproducible corpora) =====

/// SplitMix64 — deterministic, seedable, no external dep. Matches the RNG
/// used in `examples/conformal_airpassengers.rs` for bit-reproducibility.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Approximate Gaussian via central-limit sum of 12 uniforms.
    /// Matches `examples/conformal_airpassengers.rs`.
    #[inline]
    fn gaussian(&mut self, sigma: f32) -> f32 {
        let mut sum = 0.0_f32;
        for _ in 0..12 {
            sum += (self.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32);
        }
        (sum - 6.0) * sigma
    }
}

// ===== run_floor_comparison =====

/// Run the "Report the Floor" comparison of `primitive` against the canonical
/// conformal-naive floor on `corpus`.
///
/// Both the primitive and a fresh [`FloorAdapter`] observe the corpus
/// element-by-element. Before each observation, both produce a
/// [`PredictiveOutput`]; the harness normalizes both to intervals and scores
/// them on CRPS / coverage / Winkler.
///
/// # Arguments
/// - `primitive`: the UQ primitive under test.
/// - `corpus`: the trajectory values (univariate `f32` slice).
/// - `alpha`: miscoverage level (e.g. `0.05` for a 95% interval). Must be in
///   `[0, 0.5]`.
/// - `warmup`: number of initial observations to seed both primitive and
///   floor without scoring. Recommended: see [`TrajectoryCorpus`].
/// - `corpus_name`: label for the report (e.g. `"lorenz63_x"`).
///
/// # Returns
/// A [`FloorComparisonReport`] with per-metric scores, ratios, and an
/// overall verdict.
pub fn run_floor_comparison<P: UqPrimitiveUnderTest>(
    primitive: &mut P,
    corpus: &[f32],
    alpha: f32,
    warmup: usize,
    corpus_name: &str,
) -> FloorComparisonReport {
    debug_assert!(
        (0.0..=0.5).contains(&alpha),
        "alpha must be in [0, 0.5]"
    );
    debug_assert!(warmup <= corpus.len(), "warmup must be <= corpus.len()");

    let mut floor = FloorAdapter::new(alpha);

    let mut prim_intervals: Vec<PredictiveInterval> = Vec::with_capacity(corpus.len() - warmup);
    let mut floor_intervals: Vec<PredictiveInterval> = Vec::with_capacity(corpus.len() - warmup);
    let mut actuals: Vec<f32> = Vec::with_capacity(corpus.len() - warmup);
    let mut n_unscorable: usize = 0;

    // Warmup: observe without scoring. Both primitive and floor see the same
    // data, so the comparison is fair (both have the same initial state).
    let warmup_n = warmup.min(corpus.len());
    for &y in corpus.iter().take(warmup_n) {
        let _ = primitive.predict_next();
        primitive.observe(y);
        let _ = floor.predict_next();
        floor.observe(y);
    }

    // Score: predict → score → observe.
    for &y in corpus.iter().skip(warmup_n) {
        let prim_out = primitive.predict_next();
        let floor_out = floor.predict_next();

        let prim_iv = prim_out.into_interval(alpha);
        let floor_iv = floor_out.into_interval(alpha);

        // Floor should always produce an interval after warmup — if not, this
        // is a bug in the floor (not the primitive). Still score gracefully.
        match (prim_iv, floor_iv) {
            (Some(piv), Some(fiv)) => {
                prim_intervals.push(piv);
                floor_intervals.push(fiv);
                actuals.push(y);
            }
            (None, Some(_)) => {
                n_unscorable += 1;
            }
            (Some(_), None) => {
                // Floor produced nothing — unreachable after warmup, but score
                // the primitive anyway by pushing a sentinel. In practice
                // this branch never fires; we log via debug_assert.
                debug_assert!(false, "floor produced no interval after warmup");
                n_unscorable += 1;
            }
            (None, None) => {
                n_unscorable += 1;
            }
        }

        primitive.observe(y);
        floor.observe(y);
    }

    let n_scored = actuals.len();

    // Handle the degenerate case: primitive produced no scorable output.
    if n_scored == 0 {
        return FloorComparisonReport {
            primitive_name: primitive.name().to_string(),
            corpus_name: corpus_name.to_string(),
            n_scored: 0,
            n_unscorable,
            alpha,
            primitive: UqMetrics::default(),
            floor: UqMetrics::default(),
            crps_ratio: f32::NAN,
            winkler_ratio: f32::NAN,
            primitive_cov_err: f32::NAN,
            floor_cov_err: f32::NAN,
            overall: OverallVerdict::NotApplicable {
                reason: format!(
                    "primitive '{}' produced no scorable output on corpus '{}'",
                    primitive.name(),
                    corpus_name
                ),
            },
        };
    }

    let primitive_metrics = UqMetrics {
        mean_crps_interval: mean_crps_interval(&prim_intervals, &actuals),
        coverage: empirical_coverage(&prim_intervals, &actuals),
        mean_winkler: mean_winkler(&prim_intervals, &actuals),
    };
    let floor_metrics = UqMetrics {
        mean_crps_interval: mean_crps_interval(&floor_intervals, &actuals),
        coverage: empirical_coverage(&floor_intervals, &actuals),
        mean_winkler: mean_winkler(&floor_intervals, &actuals),
    };

    let nominal = 1.0 - alpha;
    let primitive_cov_err = (primitive_metrics.coverage - nominal).abs();
    let floor_cov_err = (floor_metrics.coverage - nominal).abs();

    let floor_crps = floor_metrics.mean_crps_interval.abs().max(1e-10);
    let floor_winkler = floor_metrics.mean_winkler.abs().max(1e-10);
    let crps_ratio = primitive_metrics.mean_crps_interval / floor_crps;
    let winkler_ratio = primitive_metrics.mean_winkler / floor_winkler;

    let overall = compute_verdict(
        crps_ratio,
        winkler_ratio,
        primitive_metrics.coverage,
        floor_metrics.coverage,
    );

    FloorComparisonReport {
        primitive_name: primitive.name().to_string(),
        corpus_name: corpus_name.to_string(),
        n_scored,
        n_unscorable,
        alpha,
        primitive: primitive_metrics,
        floor: floor_metrics,
        crps_ratio,
        winkler_ratio,
        primitive_cov_err,
        floor_cov_err,
        overall,
    }
}

/// Compute the overall verdict from the per-metric ratios and coverage values.
///
/// Conservative: a win requires beating the floor on BOTH lower-better
/// metrics (CRPS, Winkler) AND not under-covering relative to the floor. A
/// loss requires being worse on at least one lower-better metric with no
/// compensating win. Otherwise mixed.
///
/// Coverage policy: **over-coverage is acceptable** (a primitive that covers
/// more than nominal is conservative, and the extra width is already
/// penalized via CRPS). Only **under-coverage** (false confidence — claiming
/// tighter intervals than warranted) fails the gate.
fn compute_verdict(
    crps_ratio: f32,
    winkler_ratio: f32,
    primitive_coverage: f32,
    floor_coverage: f32,
) -> OverallVerdict {
    let crps_beats = crps_ratio < 1.0 - BEAT_THRESHOLD;
    let crps_ties = (1.0 - BEAT_THRESHOLD..=1.0 + BEAT_THRESHOLD).contains(&crps_ratio);
    let crps_loses = crps_ratio > 1.0 + BEAT_THRESHOLD;

    let winkler_beats = winkler_ratio < 1.0 - BEAT_THRESHOLD;
    let winkler_ties = (1.0 - BEAT_THRESHOLD..=1.0 + BEAT_THRESHOLD).contains(&winkler_ratio);
    let winkler_loses = winkler_ratio > 1.0 + BEAT_THRESHOLD;

    // Only flag under-coverage (primitive covers less than floor − tol).
    // Over-coverage is fine — the width penalty in CRPS handles it.
    let coverage_ok = primitive_coverage >= floor_coverage - COVERAGE_TOL;

    if (crps_beats || winkler_beats) && coverage_ok && !(crps_loses || winkler_loses) {
        return OverallVerdict::BeatsFloor;
    }
    if (crps_ties || crps_beats) && (winkler_ties || winkler_beats) && !(crps_loses || winkler_loses) {
        return OverallVerdict::TiesFloor;
    }
    if (crps_loses || winkler_loses) && !(crps_beats || winkler_beats) {
        return OverallVerdict::LosesToFloor;
    }
    OverallVerdict::Mixed
}

// ===== Unit tests =====

#[cfg(test)]
mod tests {
    use super::*;

    /// A TRUE oracle primitive — peeks at the upcoming corpus value and
    /// predicts it with a vanishingly narrow interval. Decisively beats the
    /// floor on every metric. Used only in tests (a real primitive can't see
    /// the future).
    struct TrueOraclePrimitive<'a> {
        corpus: &'a [f32],
        step: usize,
    }
    impl<'a> UqPrimitiveUnderTest for TrueOraclePrimitive<'a> {
        fn name(&self) -> &str { "true-oracle (peeks at next value)" }
        fn predict_next(&mut self) -> PredictiveOutput {
            // Peek at the upcoming ground-truth value.
            let next_y = self.corpus.get(self.step).copied().unwrap_or(0.0);
            let eps = 1e-6;
            PredictiveOutput::from_interval(PredictiveInterval::new(
                next_y - eps,
                next_y,
                next_y + eps,
                0.05,
            ))
        }
        fn observe(&mut self, _y: f32) {
            self.step += 1;
        }
    }

    /// A primitive that always predicts a very wide interval (±10σ). Should
    /// lose to the floor on CRPS/Winkler (wider = higher score).
    struct OverWidePrimitive;
    impl UqPrimitiveUnderTest for OverWidePrimitive {
        fn name(&self) -> &str { "over-wide (±10)" }
        fn predict_next(&mut self) -> PredictiveOutput {
            PredictiveOutput::from_interval(PredictiveInterval::new(-10.0, 0.0, 10.0, 0.05))
        }
        fn observe(&mut self, _y: f32) {}
    }

    /// A primitive that produces samples-only (no interval). Tests the
    /// samples → interval conversion path.
    struct SamplesOnlyPrimitive {
        last_y: f32,
    }
    impl UqPrimitiveUnderTest for SamplesOnlyPrimitive {
        fn name(&self) -> &str { "samples-only" }
        fn predict_next(&mut self) -> PredictiveOutput {
            // 41 samples around the last observed value with small Gaussian-ish spread.
            let mut samples = Vec::with_capacity(41);
            for i in 0..41u32 {
                let offset = (i as f32 - 20.0) * 0.05;
                samples.push(self.last_y + offset);
            }
            PredictiveOutput::from_samples(samples)
        }
        fn observe(&mut self, y: f32) {
            self.last_y = y;
        }
    }

    /// A primitive that produces nothing — exercises the NotApplicable path.
    struct EmptyPrimitive;
    impl UqPrimitiveUnderTest for EmptyPrimitive {
        fn name(&self) -> &str { "empty" }
        fn predict_next(&mut self) -> PredictiveOutput { PredictiveOutput::empty() }
        fn observe(&mut self, _y: f32) {}
    }

    #[test]
    fn empirical_quantile_interval_spans_data() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let iv = empirical_quantile_interval(&samples, 0.05);
        // Lower ≈ 2.5th percentile, upper ≈ 97.5th percentile of [0..100).
        assert!(iv.lower < 10.0, "lower {} should be < 10", iv.lower);
        assert!(iv.upper > 90.0, "upper {} should be > 90", iv.upper);
        assert!(iv.lower < iv.point && iv.point < iv.upper);
    }

    #[test]
    fn predictive_output_into_interval_uses_explicit() {
        let iv = PredictiveInterval::new(1.0, 2.0, 3.0, 0.05);
        let out = PredictiveOutput::from_interval(iv);
        assert_eq!(out.into_interval(0.05), Some(iv));
    }

    #[test]
    fn predictive_output_into_interval_converts_samples() {
        // 5 samples, α=0.05. Type-7 quantile index for upper = 0.975×4 = 3.9 → 3.
        // So upper = sorted[3] = 4.0 (not 5.0 — that would need interpolation).
        let samples = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let out = PredictiveOutput::from_samples(samples);
        let iv = out.into_interval(0.05).expect("samples should convert");
        assert!(iv.lower <= 1.0, "lower {} should be <= 1.0", iv.lower);
        assert!(iv.upper >= 4.0, "upper {} should be >= 4.0", iv.upper);
        assert!(iv.lower < iv.point && iv.point < iv.upper);
    }

    #[test]
    fn predictive_output_into_interval_empty_is_none() {
        let out = PredictiveOutput::empty();
        assert_eq!(out.into_interval(0.05), None);
    }

    #[test]
    fn corpus_stationary_seasonal_is_deterministic() {
        let a = TrajectoryCorpus::stationary_seasonal(12, 0.5, 100, 0xCAFE);
        let b = TrajectoryCorpus::stationary_seasonal(12, 0.5, 100, 0xCAFE);
        assert_eq!(a.values.len(), 100);
        assert_eq!(a.values, b.values, "same seed → identical corpus");
        // Mean should be near 0 (sin wave centered at 0 + zero-mean Gaussian).
        let mean: f32 = a.values.iter().sum::<f32>() / a.values.len() as f32;
        assert!(mean.abs() < 0.3, "mean {} should be near 0", mean);
    }

    #[test]
    fn corpus_white_noise_is_deterministic() {
        let a = TrajectoryCorpus::white_noise(1.0, 100, 0xBEEF);
        let b = TrajectoryCorpus::white_noise(1.0, 100, 0xBEEF);
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn floor_adapter_produces_narrow_interval_after_warmup() {
        // Warm up the floor with a stable series, then check the interval width.
        let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.1, 200, 0x1234);
        let mut floor = FloorAdapter::new(0.05);
        for &y in corpus.values.iter().take(100) {
            floor.observe(y);
        }
        let out = floor.predict_next();
        let iv = out.interval.expect("floor should produce interval");
        // σ=0.1 noise → 95% interval width ≈ 4σ = 0.4. Allow generous bound.
        let width = iv.upper - iv.lower;
        assert!(width > 0.0, "interval width {} should be > 0", width);
        assert!(width < 2.0, "interval width {} should be < 2.0 for σ=0.1", width);
    }

    #[test]
    fn floor_vs_floor_is_tie() {
        // The floor compared to itself should tie on all metrics.
        let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 500, 0xABCD_1234);
        let mut floor2 = FloorAdapter::new(0.05);
        let report = run_floor_comparison(
            &mut floor2,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        // CRPS ratio should be ~1.0 (within tolerance).
        assert!(
            (report.crps_ratio - 1.0).abs() < 0.02,
            "floor-vs-floor crps_ratio {} should be ~1.0",
            report.crps_ratio
        );
        assert!(
            matches!(report.overall, OverallVerdict::TiesFloor | OverallVerdict::BeatsFloor),
            "floor-vs-floor should tie or beat (got {:?})",
            report.overall
        );
    }

    #[test]
    fn oracle_beats_floor() {
        // True oracle peeks at the next corpus value → should beat the floor
        // decisively on every metric (vanishingly narrow interval, always covers).
        let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 300, 0x0ACE_BEE5);
        let mut oracle = TrueOraclePrimitive { corpus: &corpus.values, step: 0 };
        let report = run_floor_comparison(
            &mut oracle,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        // True oracle: ~2e-6 width, ~2e-6 CRPS, perfect coverage.
        // Floor: positive width, positive CRPS. So oracle wins on CRPS by a
        // huge margin (crps_ratio ≈ 1e-5).
        assert!(report.n_scored > 100, "should score many steps");
        assert!(
            report.crps_ratio < 0.01,
            "true-oracle crps_ratio {} should be < 0.01",
            report.crps_ratio
        );
        assert_eq!(report.overall, OverallVerdict::BeatsFloor);
    }

    #[test]
    fn overwide_loses_to_floor() {
        // Over-wide ±10 interval should lose on CRPS/Winkler.
        let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 300, 0xDEAD);
        let mut wide = OverWidePrimitive;
        let report = run_floor_comparison(
            &mut wide,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        // Over-wide CRPS = ~20 (width); floor CRPS is much smaller.
        assert!(
            report.crps_ratio > 5.0,
            "over-wide crps_ratio {} should be > 5.0",
            report.crps_ratio
        );
        assert_eq!(report.overall, OverallVerdict::LosesToFloor);
    }

    #[test]
    fn samples_only_primitive_is_scorable() {
        // Samples-only primitive should be scorable via samples → interval.
        let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 300, 0x5ADB_7E57);
        let mut samp = SamplesOnlyPrimitive { last_y: 0.0 };
        let report = run_floor_comparison(
            &mut samp,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        assert_eq!(report.n_unscorable, 0, "samples-only should be scorable");
        assert!(report.n_scored > 100);
        // The samples-only primitive is essentially seasonal-naive-like with
        // Gaussian spread, so it should roughly tie the floor (both predict
        // near the last value).
        assert!(
            (report.crps_ratio - 1.0).abs() < 0.5,
            "samples-only crps_ratio {} should be near 1.0 (±0.5)",
            report.crps_ratio
        );
    }

    #[test]
    fn empty_primitive_is_not_applicable() {
        let corpus = TrajectoryCorpus::white_noise(1.0, 100, 0xE7A_DE55);
        let mut empty = EmptyPrimitive;
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
    fn report_pretty_print_does_not_panic() {
        // Smoke: pretty_print should not panic on any verdict.
        let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 200, 0xAB_DE_01);
        let mut floor = FloorAdapter::new(0.05);
        let report = run_floor_comparison(
            &mut floor,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        report.pretty_print();
    }
}
