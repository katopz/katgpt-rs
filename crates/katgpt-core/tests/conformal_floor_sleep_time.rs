//! Issue 010 T4 — "Report the Floor" comparison for Sleep-Time Query
//! Anticipator (Plan 334/341).
//!
//! The Sleep-Time Anticipator produces a **predictability score** `p_i ∈ [0,1]`
//! per anticipated-query direction, via the modelless `DotPredictabilityScorer`
//! `p = sigmoid(α·dot(c, dir) + β)`. The T4 question: **is this predictability
//! score a calibrated UQ signal, or is it an uncalibrated gate heuristic?**
//!
//! ## Comparison angle (from Issue 010 T4)
//!
//! > "predictability scores from the anticipator vs interval-width from the
//! > floor. Both should correlate with actual forecast difficulty; the one
//! > with higher correlation wins."
//!
//! Two complementary evaluations:
//!
//! 1. **Calibration (CRPS / coverage / Winkler)** — derive a predictive
//!    interval from the anticipator's predictability (`width = z·scale·(1−p_best)`),
//!    feed it to the floor harness. If `p_best` doesn't track difficulty, the
//!    derived interval is miscalibrated and loses coverage/Winkler (the BoM
//!    false-confidence signature).
//! 2. **Correlation with difficulty** — collect raw `(1−p_best)` (inverse
//!    predictability = claimed difficulty) and the floor's interval width
//!    alongside actual difficulty `|y_t − ŷ_t|`, compute Pearson r. The floor
//!    is calibrated on exactly these residuals, so it SHOULD correlate; the
//!    question is whether the uncalibrated anticipator score does too.
//!
//! ## Method
//!
//! - **Context embedding** `c = [y_{t−1}, y_{t−2}, y_{t−3}, y_{t−4}]` (D=4 delay
//!   embedding — the last 4 observations).
//! - **Direction set** (K=4): fixed query classes capturing common patterns:
//!   - `[+1, 0, 0, 0]` — "recent level"
//!   - `[+1, +1, 0, 0]` — "two-back level"
//!   - `[+1, −1, 0, 0]` — "recent trend" (first difference)
//!   - `[0, 0, +1, −1]` — "older trend"
//! - **Point forecast** = last observation (matches the floor's seasonal-naive
//!   with m=1, so both points are identical — the comparison isolates the
//!   interval-calibration question, not the point-forecast question).
//! - **Interval width** = `z_{α/2} · residual_scale · (1 − p_best + ε)`,
//!   where `residual_scale` = empirical std of one-step residuals over warmup
//!   (the SAME information the floor's residual pool sees), `z_{α/2}` = the
//!   normal two-tailed quantile (1.95996 for α=0.05), `p_best` = max
//!   predictability across K directions, `ε = 0.05` floor to avoid zero-width.
//!
//! This is the honest framing: the anticipator claims `p` measures
//! predictability. If `p` correlates with actual difficulty, `(1−p)·scale`
//! is calibrated. If not (like BoM's σ), coverage collapses and Winkler
//! explodes. No per-corpus tuning of α/β — the scorer uses the paper default
//! (α=1.0, β=0.0) for all corpora.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_sleep_time \
//!   --features conformal_predictive_intervals,sleep_time_anticipation -- --nocapture
//! ```

#![cfg(all(feature = "conformal_predictive_intervals", feature = "sleep_time_anticipation"))]
#![allow(clippy::needless_range_loop)]

use katgpt_core::{
    AnticipatedQueryDir, DotPredictabilityScorer, FloorComparisonReport, IdentityFunctorOp,
    PredictiveInterval, PredictiveOutput, SleepTimeAnticipator,
    SleepTimeScratch, TrajectoryCorpus, UqPrimitiveUnderTest, run_floor_comparison,
};

/// Delay-embedding dimension (context window length).
const D: usize = 4;
/// Number of anticipated-query directions.
const K: usize = 4;
/// Normal two-tailed quantile at α=0.05 (Φ⁻¹(0.975)).
const Z_025: f32 = 1.959964;
/// ε floor on interval width — avoids zero-width intervals when p_best → 1.
const WIDTH_EPS: f32 = 0.05;

// ===== SleepTimeAnticipatorAdapter =====

/// Adapter wrapping `SleepTimeAnticipator<D, K, IdentityFunctorOp,
/// DotPredictabilityScorer>` as a `UqPrimitiveUnderTest`.
///
/// **Context**: `c = [y_{t−1}, y_{t−2}, y_{t−3}, y_{t−4}]` — the last D
/// observations (delay embedding).
///
/// **Prediction**: run `anticipate()` to score K directions, take `p_best` =
/// max predictability. Point forecast = last observation. Interval width =
/// `z_{α/2} · scale · (1 − p_best + ε)`.
///
/// **State advance**: on `observe(y)`, shift the delay-embedding window left
/// and append `y` at position 0.
pub struct SleepTimeAnticipatorAdapter {
    anticipator: SleepTimeAnticipator<D, K, IdentityFunctorOp, DotPredictabilityScorer>,
    dirs: [AnticipatedQueryDir<D>; K],
    scratch: SleepTimeScratch<D>,
    /// Delay-embedding window: `window[0]` = most recent, `window[D−1]` = oldest.
    window: [f32; D],
    /// Empirical residual std (from warmup). Scales the anticipator's
    /// `(1−p)` to the data's volatility, the same way the floor's residual
    /// pool does.
    residual_scale: f32,
    /// Last point forecast (= last observation).
    last_point: f32,
    /// α for interval construction.
    alpha: f32,
    /// Number of observations seen so far (for warmup detection).
    n_seen: usize,
    /// Warmup length (residual_scale computed from observations 1..warmup).
    warmup: usize,
    /// Accumulator for one-step residuals during warmup (for scale estimate).
    warmup_residuals: Vec<f32>,
    /// Whether residual_scale has been finalized.
    scale_finalized: bool,
}

impl SleepTimeAnticipatorAdapter {
    /// Construct an unfitted adapter.
    ///
    /// - `scorer_alpha`: DotPredictabilityScorer sharpness (paper default 1.0).
    /// - `scorer_beta`: DotPredictabilityScorer bias (paper default 0.0).
    /// - `alpha`: miscoverage level for interval construction (e.g. 0.05).
    /// - `warmup`: number of warmup observations used to estimate
    ///   `residual_scale` (the one-step residual std). Must match the harness
    ///   warmup so both primitive and floor see the same seed data.
    pub fn new(scorer_alpha: f32, scorer_beta: f32, alpha: f32, warmup: usize) -> Self {
        // Fixed direction set — common query classes on a delay embedding.
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0, 0.0, 0.0]), // recent level
            AnticipatedQueryDir::new([1.0, 1.0, 0.0, 0.0]), // two-back level
            AnticipatedQueryDir::new([1.0, -1.0, 0.0, 0.0]), // recent trend
            AnticipatedQueryDir::new([0.0, 0.0, 1.0, -1.0]), // older trend
        ];
        let anticipator = SleepTimeAnticipator {
            op: IdentityFunctorOp,
            scorer: DotPredictabilityScorer::new(scorer_alpha, scorer_beta),
            budgets: [100; K],
            tau: 0.5,
            beta: 4.0,
        };
        Self {
            anticipator,
            dirs,
            scratch: SleepTimeScratch::new(),
            window: [0.0; D],
            residual_scale: 1.0,
            last_point: 0.0,
            alpha,
            n_seen: 0,
            warmup,
            warmup_residuals: Vec::with_capacity(warmup),
            scale_finalized: false,
        }
    }

    /// Shift the window left and append `y` at position 0.
    #[inline]
    fn push_observation(&mut self, y: f32) {
        // window[i] = y_{t−i}; shift so window[0] = newest.
        for i in (1..D).rev() {
            self.window[i] = self.window[i - 1];
        }
        self.window[0] = y;
    }

    /// Finalize the residual scale from warmup residuals (population std).
    /// Called once, at the end of warmup.
    fn finalize_scale(&mut self) {
        if self.warmup_residuals.is_empty() {
            self.residual_scale = 1.0;
        } else {
            let n = self.warmup_residuals.len() as f32;
            let mean: f32 = self.warmup_residuals.iter().sum::<f32>() / n;
            let var: f32 = self.warmup_residuals.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f32>() / n;
            self.residual_scale = var.sqrt().max(1e-6);
        }
        self.scale_finalized = true;
    }
}

impl UqPrimitiveUnderTest for SleepTimeAnticipatorAdapter {
    fn name(&self) -> &str {
        "SleepTimeAnticipator (DotPredictabilityScorer, unfitted, delay-embed D=4)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        // After warmup, residual_scale is fixed. Point = last observation.
        let point = self.last_point;

        // If scale not finalized yet (still in warmup), emit a wide interval
        // so the harness doesn't score garbage. The harness skips warmup
        // anyway, but predict_next is called once before the first scored
        // observe — defensive.
        if !self.scale_finalized {
            let wide = PredictiveInterval::new(point - 100.0, point, point + 100.0, self.alpha);
            return PredictiveOutput::from_interval(wide);
        }

        // Score all K directions, take p_best = max predictability.
        let artifact = self.anticipator.anticipate(&self.window, &self.dirs, &mut self.scratch);
        let mut p_best: f32 = 0.0;
        for slot in &artifact.slots {
            if slot.predictability > p_best {
                p_best = slot.predictability;
            }
        }
        // Clamp p_best to [0,1] defensively (sigmoid output should already be).
        let p_best = p_best.clamp(0.0, 1.0);

        // Width = z_{α/2} · scale · (1 − p_best + ε).
        // High p_best → narrow interval (confident). Low p_best → wide.
        let half_width = Z_025 * self.residual_scale * (1.0 - p_best + WIDTH_EPS);
        let lower = point - half_width;
        let upper = point + half_width;
        let iv = PredictiveInterval::new(lower, point, upper, self.alpha);
        PredictiveOutput::from_interval(iv)
    }

    fn observe(&mut self, y: f32) {
        // During warmup, accumulate one-step residuals for the scale estimate.
        // The residual at step t = |y_t − y_{t−1}| (persistent-forecast error —
        // the same quantity the floor's m=1 residual pool sees).
        if !self.scale_finalized && self.n_seen > 0 {
            let residual = y - self.window[0];
            self.warmup_residuals.push(residual);
        }

        self.push_observation(y);
        self.last_point = y;
        self.n_seen += 1;

        // Finalize scale at the end of warmup.
        if !self.scale_finalized && self.n_seen >= self.warmup {
            self.finalize_scale();
        }
    }
}

// ===== Corpora =====

/// Standard seasonal: `sin(2πt/12) + N(0, 0.1)`.
fn seasonal_m12(n: usize, seed: u64) -> TrajectoryCorpus {
    TrajectoryCorpus::stationary_seasonal(12, 0.1, n, seed)
}

/// White noise: `N(0, 0.5)`.
fn white_noise(n: usize, seed: u64) -> TrajectoryCorpus {
    TrajectoryCorpus::white_noise(0.5, n, seed)
}

/// Regime-switching: alternating blocks of seasonal (predictable) and
/// random-walk (unpredictable). This corpus has GENUINE variation in
/// forecast difficulty — the key test for the correlation angle.
fn regime_switching(n: usize, seed: u64) -> TrajectoryCorpus {
    use rng::SplitMix64;
    let mut rng = SplitMix64::new(seed);
    let mut values = Vec::with_capacity(n);
    let block = 40; // each regime lasts 40 steps
    let mut t_in_block = 0;
    let mut seasonal_regime = true;
    let mut last_y = 0.0_f32;
    for _ in 0..n {
        let y = if seasonal_regime {
            let phase = 2.0 * core::f32::consts::PI * (t_in_block as f32) / 12.0;
            phase.sin() + rng.gaussian(0.1)
        } else {
            // Random walk: y_t = y_{t-1} + N(0, 0.3)
            last_y + rng.gaussian(0.3)
        };
        values.push(y);
        last_y = y;
        t_in_block += 1;
        if t_in_block >= block {
            t_in_block = 0;
            seasonal_regime = !seasonal_regime;
        }
    }
    TrajectoryCorpus {
        name: format!("regime_switching_block{}_n{}", block, n),
        values,
        recommended_warmup: 80,
    }
}

// ===== Pearson correlation helper =====

/// Pearson correlation coefficient between two slices.
/// Returns 0.0 if either slice has zero variance (degenerate).
fn pearson_r(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

// ===== Difficulty-correlation harness (T4 specific) =====

/// Run both the anticipator and the floor on a corpus, collecting per-step:
/// - anticipator `(1 − p_best)` (claimed difficulty)
/// - floor interval width (calibrated difficulty proxy)
/// - actual difficulty `|y_t − ŷ_t^{naive}|` (ground truth = |y_t − y_{t−1}|)
///
/// Returns `(r_anticipator, r_floor)` — Pearson r of each signal with actual
/// difficulty. The floor is calibrated on exactly these residuals, so r_floor
/// should be high; the question is whether r_anticipator is comparable.
fn difficulty_correlation(corpus: &[f32], warmup: usize) -> (f32, f32) {
    use katgpt_core::FloorAdapter;

    let alpha = 0.05_f32;
    let mut prim = SleepTimeAnticipatorAdapter::new(1.0, 0.0, alpha, warmup);
    let mut floor = FloorAdapter::new(alpha);

    // Warmup.
    for &y in corpus.iter().take(warmup) {
        let _ = prim.predict_next();
        prim.observe(y);
        let _ = floor.predict_next();
        floor.observe(y);
    }

    let mut prim_signals: Vec<f32> = Vec::with_capacity(corpus.len() - warmup);
    let mut floor_signals: Vec<f32> = Vec::with_capacity(corpus.len() - warmup);
    let mut difficulties: Vec<f32> = Vec::with_capacity(corpus.len() - warmup);

    for &y in corpus.iter().skip(warmup) {
        let prim_out = prim.predict_next();
        let floor_out = floor.predict_next();

        // Actual difficulty = |y_t − y_{t−1}| (one-step innovation magnitude).
        // y_{t−1} = corpus[i−1], y_t = y (the value about to be observed).
        let prev_y = if warmup > 0 {
            // corpus index of the previous observation = current index − 1.
            // We're iterating skip(warmup), so the previous is corpus[i+warmup−1].
            // Track via prim.last_point which was set to the previous y on the
            // last observe().
            prim.last_point
        } else {
            0.0
        };

        if let (Some(piv), Some(fiv)) = (prim_out.into_interval(alpha), floor_out.into_interval(alpha)) {
            // Anticipator signal = half-width / (z · scale) ≈ (1 − p_best + ε).
            // We invert back to the predictability-derived difficulty proxy.
            // half_width = piv.upper - piv.point; the difficulty signal is
            // half_width itself (wider = harder), which already captures
            // (1 − p_best) · scale. Use half_width directly.
            let prim_half = (piv.upper - piv.point).abs();
            let floor_half = (fiv.upper - fiv.point).abs();
            prim_signals.push(prim_half);
            floor_signals.push(floor_half);
            difficulties.push((y - prev_y).abs());
        }

        prim.observe(y);
        floor.observe(y);
    }

    let r_prim = pearson_r(&prim_signals, &difficulties);
    let r_floor = pearson_r(&floor_signals, &difficulties);
    (r_prim, r_floor)
}

// ===== Test-local deterministic RNG (matches harness's SplitMix64) =====

/// Minimal SplitMix64 + central-limit Gaussian for reproducible corpora.
/// Bit-compatible with the harness's private SplitMix64 (same constants,
/// same sum-of-12-uniforms Gaussian approximation).
mod rng {
    pub struct SplitMix64 {
        pub state: u64,
    }
    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        pub fn gaussian(&mut self, sigma: f32) -> f32 {
            let mut sum = 0.0_f32;
            for _ in 0..12 {
                self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                sum += (z >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32);
            }
            (sum - 6.0) * sigma
        }
    }
}

// ===== Tests =====

fn run_and_print(
    prim: &mut SleepTimeAnticipatorAdapter,
    corpus: &TrajectoryCorpus,
    alpha: f32,
) -> FloorComparisonReport {
    let report = run_floor_comparison(prim, &corpus.values, alpha, corpus.recommended_warmup, &corpus.name);
    report.pretty_print();
    report
}

#[test]
fn anticipator_vs_floor_on_seasonal() {
    let corpus = seasonal_m12(600, 0xA1);
    let mut prim = SleepTimeAnticipatorAdapter::new(1.0, 0.0, 0.05, corpus.recommended_warmup);
    let report = run_and_print(&mut prim, &corpus, 0.05);

    println!("\n[seasonal] Anticipator residual_scale = {:.4}", prim.residual_scale);

    // The anticipator's predictability is an UNCALIBRATED heuristic.
    // On a stationary seasonal series the floor is near-optimal, so we
    // EXPECT the anticipator to lose on coverage/Winkler. We assert only
    // that the test RAN and produced a scorable report (not that it wins).
    assert!(report.n_scored > 0, "must score some steps");
    assert!(report.primitive.coverage >= 0.0 && report.primitive.coverage <= 1.0);
}

#[test]
fn anticipator_vs_floor_on_white_noise() {
    let corpus = white_noise(600, 0xB2);
    let mut prim = SleepTimeAnticipatorAdapter::new(1.0, 0.0, 0.05, corpus.recommended_warmup);
    let report = run_and_print(&mut prim, &corpus, 0.05);

    println!("\n[white_noise] Anticipator residual_scale = {:.4}", prim.residual_scale);

    assert!(report.n_scored > 0, "must score some steps");
}

#[test]
fn anticipator_vs_floor_on_regime_switching() {
    let corpus = regime_switching(800, 0xC3);
    let mut prim = SleepTimeAnticipatorAdapter::new(1.0, 0.0, 0.05, corpus.recommended_warmup);
    let report = run_and_print(&mut prim, &corpus, 0.05);

    println!("\n[regime_switching] Anticipator residual_scale = {:.4}", prim.residual_scale);

    assert!(report.n_scored > 0, "must score some steps");
}

#[test]
fn anticipator_difficulty_correlation_on_regime_switching() {
    // T4's headline test: does the anticipator's predictability correlate
    // with actual forecast difficulty? The regime-switching corpus has
    // genuine difficulty variation (seasonal blocks vs random-walk blocks).
    let corpus = regime_switching(800, 0xC3);
    let warmup = corpus.recommended_warmup;
    let (r_prim, r_floor) = difficulty_correlation(&corpus.values, warmup);

    println!("\n=== Difficulty Correlation (regime_switching) ===");
    println!("  Anticipator (1−p_best)-derived half-width vs |Δy|:  r = {:.4}", r_prim);
    println!("  Floor half-width vs |Δy|:                            r = {:.4}", r_floor);
    println!("  Ratio (prim/floor):                                  {:.4}", r_prim / r_floor.max(1e-6));

    // The floor is calibrated on |Δy|, so r_floor should be high.
    // The anticipator's predictability is an uncalibrated heuristic; we
    // DOCUMENT whatever correlation it has (the honest finding) without
    // asserting a threshold — the point of T4 is to MEASURE, not to gate.
    assert!(r_prim.is_finite(), "correlation must be finite");
    assert!(r_floor.is_finite(), "correlation must be finite");
}

#[test]
fn anticipator_difficulty_correlation_on_seasonal() {
    let corpus = seasonal_m12(600, 0xA1);
    let warmup = corpus.recommended_warmup;
    let (r_prim, r_floor) = difficulty_correlation(&corpus.values, warmup);

    println!("\n=== Difficulty Correlation (seasonal) ===");
    println!("  Anticipator half-width vs |Δy|:  r = {:.4}", r_prim);
    println!("  Floor half-width vs |Δy|:        r = {:.4}", r_floor);

    assert!(r_prim.is_finite());
    assert!(r_floor.is_finite());
}

#[test]
fn anticipator_full_report_for_benchmark_doc() {
    // The canonical evidence run — prints all three corpora + correlations
    // for the `.benchmarks/010_sleep_time_floor_comparison.md` writeup.
    println!("\n\n========================================");
    println!("=== Sleep-Time Anticipator Floor Comparison (Issue 010 T4) ===");
    println!("========================================\n");

    for (name, corpus) in [
        ("seasonal_m12", seasonal_m12(600, 0xA1)),
        ("white_noise", white_noise(600, 0xB2)),
        ("regime_switching", regime_switching(800, 0xC3)),
    ] {
        println!("\n--- Corpus: {} ---", name);
        let mut prim =
            SleepTimeAnticipatorAdapter::new(1.0, 0.0, 0.05, corpus.recommended_warmup);
        let _report = run_and_print(&mut prim, &corpus, 0.05);

        let (r_prim, r_floor) =
            difficulty_correlation(&corpus.values, corpus.recommended_warmup);
        println!("  Difficulty correlation: anticipator r = {:.4}, floor r = {:.4}",
                 r_prim, r_floor);
    }
}
