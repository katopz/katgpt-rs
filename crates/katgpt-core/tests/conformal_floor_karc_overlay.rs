//! Issue 010 T7 — "Report the Floor" comparison for the KARC + conformal overlay
//! (Plan 308 + Plan 340 Phase 2).
//!
//! The KARC + conformal overlay is the composition
//! `ConformalIntervalCalibrator<KarcChannelForecaster<..>>`. It produces
//! coverage-guaranteed predictive intervals by feeding KARC's point forecast
//! into the conformal residual-pool machinery. This is the canonical
//! "UQ-bearing primitive" pattern documented in
//! `examples/conformal_karc_overlay.rs` + `conformal::karc_adapter`.
//!
//! ## Why this test exists
//!
//! Per AGENTS.md "Feature Flag Discipline" / Issue 010's "Report the Floor"
//! rule, every UQ-bearing primitive must benchmark against the
//! conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`
//! with `m=1`) on CRPS / coverage / Winkler. The other three grandfathered
//! UQ primitives — BoMSampler (T3), Sleep-Time Anticipator (T4), Best-Belief
//! Beta Selector (T5) — each shipped a floor test in the Issue 010 cycle.
//! KARC+overlay was the remaining grandfathered primitive without one. This
//! test closes that gap.
//!
//! ## Comparison angle
//!
//! Unlike BoM (where σ controls width, not data → EXCLUDED as UQ), KARC+overlay
//! is structurally similar to the floor: both wrap a point forecaster in the
//! SAME conformal math. The only difference is the wrapped forecaster
//! (KARC's reservoir vs. seasonal-naive's last-observation anchor). So the
//! question reduces to: **does KARC's point forecast produce tighter
//! conformal intervals than seasonal-naive's, at the same coverage?**
//!
//! ## Honest verdict (recorded from the canonical run)
//!
//! | Corpus | K | CRPS ratio | Winkler ratio | Coverage (nom 0.95) | Verdict |
//! |---|---|---|---|---|---|
//! | stationary_seasonal m=12, σ=0.5 | 4 | 5.74 | 21.73 | 0.916 vs 0.939 | **LosesToFloor** |
//! | Lorenz-x dt=0.02 (chaotic) | 4 | 0.0047 | 0.0059 | 0.932 vs 0.943 | **BeatsFloor** |
//! | stationary_seasonal m=12, σ=0.5 | 12 | 20.26 | 36.97 | 0.911 vs 0.939 | **LosesToFloor (worse)** |
//! | Lorenz-x dt=0.02 (chaotic) | 12 | 0.0018 | 0.0023 | 0.933 vs 0.943 | **BeatsFloor (better)** |
//!
//! **KARC+overlay is a chaotic-regime specialist, not a universal UQ
//! improvement.** On stationary seasonal data (period 12) KARC LOSES
//! regardless of K — the prior session's hypothesis that "K=4 is too
//! shallow for period-12" was **refuted by the K=12 measurement**: K=12
//! loses WORSE (CRPS ratio 5.74 → 20.26), not better. The scope-limit is
//! **structural** (KARC's Chebyshev basis + ridge-fit architecture doesn't
//! fit periodic data), not parametric in K. Meanwhile K=12 IMPROVES KARC's
//! chaotic performance (CRPS ratio 0.0047 → 0.0018, ~2.6× tighter) — more
//! delay context helps on chaotic signals and hurts on periodic ones.
//!
//! ### K-sweep follow-up (this file's `*_k12` tests, measured 2026-07-20)
//!
//! The original T7 verdict stated "K=4 too shallow for period-12" as a
//! hypothesis. The K-sweep tests verify that claim empirically by re-running
//! the seasonal comparison with K=12 (matching the period). The measurement
//! REFUTES the hypothesis:
//!   - **K=12 on seasonal: LosesToFloor WORSE (CRPS 5.74 → 20.26)** → the
//!     scope-limit is structural. KARC's basis/reservoir architecture doesn't
//!     fit periodic data regardless of delay depth. More context makes the
//!     over-fit worse, not better.
//!   - **K=12 on Lorenz: BeatsFloor HARDER (CRPS 0.0047 → 0.0018)** → more
//!     delay context genuinely helps on chaotic signals. KARC benefits from
//!     longer memory of recent trajectory curvature.
//!
//! Production guidance: KARC+overlay is a chaotic-regime specialist. Pick K
//! based on how much trajectory memory the chaotic signal benefits from
//! (larger K = tighter intervals on chaotic, but no help on periodic). For
//! periodic/seasonal data, use the floor (`SeasonalNaiveForecaster`)
//! directly — KARC cannot match it regardless of K.
//!
//! The take-away for the "Report the Floor" policy: KARC+overlay is
//! **IN-SCOPE** as a UQ primitive (it produces calibrated intervals, the
//! coverage doesn't break) but **SCOPE-LIMITED** to chaotic regimes where
//! KARC's delay embedding captures the signal's structure. The Plan 308 GOAT
//! gate (chaotic Lorenz double-scroll) remains the right canonical validation
//! target; stationary-seasonal is out-of-scope regardless of K.
//!
//! ## Method
//!
//! The adapter `KarcOverlayAdapter` holds ONE pre-fitted KARC forecaster
//! (D=1, M=8, K=4) wrapped in a `ConformalIntervalCalibrator<
//! KarcChannelForecaster<..>>`. KARC is fit once on the warmup corpus slice
//! (the same data the floor's residual pool sees during warmup) — the
//! comparison is honest: both sides get the same warmup.
//!
//! Per tick:
//! 1. Build the delay state from the adapter's recent-observation window.
//! 2. KARC forecasts the next point.
//! 3. The calibrator reads the conformal interval around that point
//!    (`interval_from_point_into` — the documented KARC pattern).
//! 4. After the ground truth is revealed, `update_residual` pushes the
//!    residual and `step()` advances the calibrator's tick.
//!
//! ## Run
//!
//! ```bash
//! cargo test -p katgpt-core --test conformal_floor_karc_overlay \
//!   --features conformal_predictive_intervals,karc_forecaster -- --nocapture
//! ```

#![allow(clippy::needless_range_loop)]

use katgpt_core::{
    ChebyshevBasis, ConformalIntervalCalibrator, DecayUnit, FloorComparisonReport,
    KarcChannelForecaster, KarcForecaster, OverallVerdict, PointForecaster, PredictiveInterval,
    PredictiveOutput, ResidualMode, TrajectoryCorpus, UqPrimitiveUnderTest, run_floor_comparison,
};

// ── KARC shape ────────────────────────────────────────────────────────────
// D=1 (scalar trajectory — matches the floor's 1-channel layout exactly),
// M=8 (Chebyshev features per coordinate — enough expressivity to capture
// Lorenz-x's smooth-but-chaotic dynamics). K (delay-embedding depth) is
// generic per-adapter — the original T7 tests use K=4 (d_h=32 features);
// the K-sweep follow-up tests use K=12 to match period-12 seasonal cycles.
const D: usize = 1;
const M: usize = 8;

/// The canonical K=4 (original T7 measurement). Used by the original tests.
pub const K4: usize = 4;

/// K=12 (K-sweep follow-up). Matches the period-12 seasonal cycle, letting
/// KARC's delay embedding see a full period of context.
pub const K12: usize = 12;

/// Capacity of the conformal residual ring buffer. Matches the canonical
/// floor config (FLOOR_CAPACITY = 256).
const POOL_CAPACITY: usize = 256;

/// α for the conformal interval (95% coverage).
const ALPHA: f32 = 0.05;

// ===== Lorenz-63 (deterministic f64 trajectory generator) ==================
//
// Used to construct a chaotic 1D corpus the floor cannot easily forecast
// (its anchor = last value misses the curvature). KARC's delay embedding
// should capture the structure → smaller residuals → tighter intervals.

const LORENZ_SIGMA: f64 = 10.0;
const LORENZ_RHO: f64 = 28.0;
const LORENZ_BETA: f64 = 8.0 / 3.0;

#[inline]
fn lorenz_rhs(state: &[f64; 3], out: &mut [f64; 3]) {
    let (x, y, z) = (state[0], state[1], state[2]);
    out[0] = LORENZ_SIGMA * (y - x);
    out[1] = x * (LORENZ_RHO - z) - y;
    out[2] = x * y - LORENZ_BETA * z;
}

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

/// Generate `n` samples of the Lorenz-63 x-coordinate after discarding
/// `n_transient` transient samples. The sampling interval dt=0.02 is coarse
/// enough to keep consecutive samples decorrelated (well-conditioned Gram).
fn generate_lorenz_x(n_transient: usize, n: usize, dt: f64) -> Vec<f32> {
    let mut state = [0.1_f64, 0.0, 0.0];
    for _ in 0..n_transient {
        rk4_step(&mut state, dt);
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        rk4_step(&mut state, dt);
        out.push(state[0] as f32);
    }
    out
}

// ===== KarcOverlayAdapter ==================================================

/// Adapter wrapping a pre-fitted KARC + conformal overlay as a
/// `UqPrimitiveUnderTest`. Generic over K (the delay-embedding depth) so
/// the same adapter logic supports both the canonical K=4 measurement and
/// the K-sweep follow-up (K=12, etc.). D and M are file-level consts
/// (D=1, M=8) — only K varies in this test suite.
pub struct KarcOverlayAdapter<const K: usize> {
    /// KARC wrapped in the conformal calibrator. The calibrator's wrapped
    /// forecaster is `KarcChannelForecaster<..>` (channel 0 of D=1).
    calibrator: ConformalIntervalCalibrator<KarcChannelForecaster<ChebyshevBasis<M>, D, M, K>>,
    /// Recent-observation window: `window[0]` = y_{t−1} (most recent),
    /// `window[K-1]` = y_{t−K}. Used to build the K·D delay state.
    window: [f32; K],
    /// Pre-allocated delay-state buffer (length K·D). Reused per forecast —
    /// zero allocations on the hot path.
    delay_state: Vec<f32>,
    /// Last point forecast KARC produced (used for residual update).
    last_point: f32,
    /// α for the conformal interval.
    alpha: f32,
    /// Whether the adapter has seen enough observations to fill the delay
    /// window. Before this, the interval is undefined → adapter emits a
    /// wide placeholder (harness skips warmup ticks anyway).
    warmed_up: bool,
    /// Number of observations seen.
    n_seen: usize,
}

impl<const K: usize> KarcOverlayAdapter<K> {
    /// Construct an adapter with a PRE-FITTED KARC forecaster.
    ///
    /// The KARC is fit on the `warmup_corpus` slice (the same observations
    /// the harness will feed to `observe()` during warmup). This is the
    /// honest comparison: both KARC and the floor see the same warmup data.
    /// The alternative (fit incrementally as observations arrive) is also
    /// valid but more complex and not what production callers do — they fit
    /// once on a training slice and use the fitted model in production.
    pub fn new_fitted(warmup_corpus: &[f32], alpha: f32) -> Self {
        let basis = ChebyshevBasis::<M>::new();
        let mut karc =
            KarcForecaster::<ChebyshevBasis<M>, D, M, K>::with_capacity(basis, warmup_corpus.len());

        // Build training pairs: for each t ∈ [K-1, n-1], delay_state =
        // [y_t, y_{t-1}, ..., y_{t-K+1}] flattened (K·D values, here D=1 so
        // just K values), target = y_{t+1}.
        //
        // (We use a `Vec` for `ds` rather than a stack array because K is a
        // const generic — stable Rust doesn't allow `K * D` in array lengths.
        // This is test-only warmup-fit code; the production hot path stays
        // zero-alloc — see `KarcForecaster::with_capacity`'s pre-allocated
        // buffers.)
        if warmup_corpus.len() > K {
            let mut ds = vec![0.0_f32; K * D];
            for t in (K - 1)..(warmup_corpus.len() - 1) {
                for k in 0..K {
                    ds[k * D..(k + 1) * D].copy_from_slice(&warmup_corpus[t - k..=t - k]);
                }
                let target = [warmup_corpus[t + 1]];
                karc.accumulate_pair(&ds, &target);
            }
        }
        // λ=1e-3 matches the example; the Lorenz-x Gram is well-conditioned
        // at dt=0.02 with this regularization. Fit failure → the adapter
        // still constructs but forecasts return 0.0 (KARC's "not fitted"
        // fallback). The test would then fail loudly on the first assertion.
        let _ = karc.fit_ridge(1e-3);

        let adapter = KarcChannelForecaster::new(karc, 0);
        let calibrator = ConformalIntervalCalibrator::new(
            adapter,
            1, // n_channels (matches D=1)
            1, // max_h (KARC is h=1)
            1, // m=1 (non-seasonal — matches the floor)
            POOL_CAPACITY,
            0.0, // exp_lambda=0 — equal weight, matches floor
            DecayUnit::Step,
            ResidualMode::HStep,
            false, // orientation
        );

        Self {
            calibrator,
            window: [0.0; K],
            delay_state: vec![0.0_f32; K * D],
            last_point: 0.0,
            alpha,
            warmed_up: false,
            n_seen: 0,
        }
    }

    /// Shift the window left and append `y` at position 0.
    #[inline]
    fn push_observation(&mut self, y: f32) {
        for i in (1..K).rev() {
            self.window[i] = self.window[i - 1];
        }
        self.window[0] = y;
        self.n_seen += 1;
        if self.n_seen >= K {
            self.warmed_up = true;
        }
    }

    /// Build the delay state from the current window into `self.delay_state`.
    /// Layout: `[window[0], window[1], ..., window[K-1]]` (K values, since D=1).
    #[inline]
    fn build_delay_state(&mut self) {
        for k in 0..K {
            self.delay_state[k * D..(k + 1) * D].copy_from_slice(&self.window[k..=k]);
        }
    }
}

impl<const K: usize> UqPrimitiveUnderTest for KarcOverlayAdapter<K> {
    fn name(&self) -> &str {
        // Allocate a per-call String. This is a cold API call (once per test run,
        // for logging) — the hot `predict_next` / `observe` paths stay zero-alloc.
        // We leak the String to get a `&'static str` so the trait signature stays
        // simple; tests don't churn this enough to matter.
        //
        // (Alternative: use a `&'static str` via `concat!`, but `K` is generic —
        // can't `concat!` a const generic. `format!` + `Box::leak` is the
        // cleanest escape hatch for one-off cold-path formatting.)
        let s: String =
            format!("KARC+overlay (Chebyshev M={M}, K={K}, D={D}; pre-fitted on warmup)");
        Box::leak(s.into_boxed_str())
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        // During warmup (not enough observations to fill the delay window),
        // emit a wide interval so the harness's scoring doesn't see garbage.
        // The harness skips warmup ticks anyway, but predict_next is called
        // once before the first scored observe — defensive.
        if !self.warmed_up {
            let wide = PredictiveInterval::new(-1e6, 0.0, 1e6, self.alpha);
            return PredictiveOutput::from_interval(wide);
        }

        // Build delay state, run KARC forecast.
        self.build_delay_state();
        let mut point = 0.0_f32;
        // KARC's PointForecaster impl via the adapter: takes &mut self,
        // writes one f32 to `point`. The wrapped KarcChannelForecaster
        // projects channel 0 of the D=1 output.
        self.calibrator
            .forecaster
            .forecast_into(&self.delay_state, 1, &mut point);
        self.last_point = point;

        // Read the conformal interval around that point. The calibrator's
        // residual pool is per-channel × per-horizon-bucket; for KARC we use
        // channel=0, h=1 (the only configured channel/horizon).
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, self.alpha);
        self.calibrator
            .interval_from_point_into(point, 0, 1, self.alpha, &mut iv);
        PredictiveOutput::from_interval(iv)
    }

    fn observe(&mut self, y: f32) {
        // Update the residual pool with the realized observation, then push
        // the observation into the delay window and advance the calibrator's
        // tick. Order matters: the residual uses `last_point` (the forecast
        // produced in the preceding predict_next), so we must update the
        // residual BEFORE pushing y into the window (which would change what
        // the NEXT forecast sees, not the residual we're scoring now).
        if self.warmed_up {
            self.calibrator.update_residual(y, self.last_point, 0, 1);
            self.calibrator.step();
        }
        self.push_observation(y);
    }
}

// ===== Tests ===============================================================

/// Sanity: adapter constructs + forecasts without panic after enough warmup.
#[test]
fn adapter_predicts_after_warmup() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.1, 200, 0x1234);
    let warmup = corpus.recommended_warmup;
    let mut adapter = KarcOverlayAdapter::<K4>::new_fitted(&corpus.values[..warmup], ALPHA);

    // Feed the warmup observations through observe() so the window fills.
    for &y in &corpus.values[..warmup] {
        let _ = adapter.predict_next();
        adapter.observe(y);
    }

    // After warmup, predict_next should produce a finite interval.
    let out = adapter.predict_next();
    let iv = out.interval.expect("interval after warmup");
    assert!(iv.lower.is_finite(), "lower finite");
    assert!(iv.upper.is_finite(), "upper finite");
    assert!(iv.upper >= iv.lower, "upper >= lower");
}

/// Sanity: the adapter produces bit-identical intervals when run twice on the
/// same corpus. The floor comparison requires determinism.
#[test]
fn adapter_is_deterministic() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.1, 300, 0xABCD);
    let warmup = corpus.recommended_warmup;

    let run_once = || -> Vec<PredictiveInterval> {
        let mut adapter = KarcOverlayAdapter::<K4>::new_fitted(&corpus.values[..warmup], ALPHA);
        for &y in &corpus.values[..warmup] {
            let _ = adapter.predict_next();
            adapter.observe(y);
        }
        let mut out = Vec::new();
        for &y in &corpus.values[warmup..] {
            let p = adapter.predict_next();
            out.push(p.interval.expect("interval"));
            adapter.observe(y);
        }
        out
    };

    let a = run_once();
    let b = run_once();
    assert_eq!(a.len(), b.len());
    for (i, (xa, xb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            (xa.lower.to_bits(), xa.upper.to_bits()),
            (xb.lower.to_bits(), xb.upper.to_bits()),
            "interval {i} differs across runs"
        );
    }
}

/// Comparison 1: stationary seasonal corpus (K=4 baseline).
///
/// The floor's seasonal-naive anchor (forecast = last value) is near-optimal
/// for a stationary seasonal signal with period m=12. KARC's Chebyshev basis
/// with ridge-fit doesn't fit periodic data — the point forecast
/// systematically overshoots/undershoots at curvature changes, so the
/// conformal overlay's intervals are wider than the floor's.
///
/// **Measured verdict (K=4): KARC+overlay LOSES on this corpus**
/// (CRPS ratio 5.74, coverage 0.916 vs floor 0.939). This is a real finding,
/// not a bug — KARC+overlay is a chaotic-regime specialist, not a universal
/// UQ improvement. The conformal overlay faithfully calibrates around
/// whatever point forecast KARC produces; when the point forecast is worse
/// than the floor's anchor, the overlay's intervals are wider (the overlay
/// doesn't lie about uncertainty).
///
/// **K-sweep follow-up** (see `floor_comparison_stationary_seasonal_k12`
/// below): the prior "K=4 too shallow for period-12" hypothesis was REFUTED —
/// K=12 (matching the period) loses WORSE (CRPS ratio 20.26), not better.
/// The scope-limit is structural, not parametric in K.
///
/// Gate: we ACCEPT LosesToFloor here as a documented scope limitation.
/// The test asserts the weaker but still load-bearing properties:
/// - Coverage stays within 0.10 of the floor (no calibration break)
/// - The verdict is one of the scored outcomes (not NotApplicable)
#[test]
fn floor_comparison_stationary_seasonal() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 800, 0xCAFE_BABE);
    let warmup = corpus.recommended_warmup;

    let mut adapter = KarcOverlayAdapter::<K4>::new_fitted(&corpus.values[..warmup], ALPHA);
    let report: FloorComparisonReport =
        run_floor_comparison(&mut adapter, &corpus.values, ALPHA, warmup, &corpus.name);

    println!("── KARC+overlay vs floor on {} ──", corpus.name);
    println!("{report:.?}");

    // Sanity: at least 100 scored steps.
    assert!(report.n_scored > 100, "n_scored = {}", report.n_scored);

    // Hard gate: KARC must NOT lose to the floor on coverage. The floor is
    // calibrated by construction; KARC's residual pool uses the same math.
    // A coverage regression > 0.10 means a bug in the adapter wiring (likely
    // the residual-push ordering or delay-state construction). KARC losing
    // on CRPS/Winkler is acceptable (scope limitation); losing on coverage
    // is not (would indicate broken calibration).
    let floor_cov = report.floor.coverage;
    let prim_cov = report.primitive.coverage;
    let cov_delta = (prim_cov - floor_cov).abs();
    assert!(
        cov_delta < 0.10,
        "coverage delta too large: floor={floor_cov:.4}, prim={prim_cov:.4}, |Δ|={cov_delta:.4}"
    );

    // Verdict: any of these is acceptable on seasonal. LosesToFloor is the
    // expected honest verdict (KARC's Chebyshev basis doesn't fit periodic
    // data — see the K=12 K-sweep test below for the structural confirmation).
    // BeatsFloor would mean KARC found structure we didn't expect; that's
    // also fine. NotApplicable shouldn't happen (adapter always emits
    // intervals post-warmup).
    assert!(
        matches!(
            report.overall,
            OverallVerdict::BeatsFloor
                | OverallVerdict::TiesFloor
                | OverallVerdict::Mixed
                | OverallVerdict::LosesToFloor
        ),
        "KARC+overlay verdict on seasonal: {:?} — expected one of the scored outcomes",
        report.overall
    );
}

/// Comparison 2: Lorenz-63 x-coordinate (chaotic).
///
/// The floor's anchor (last value) cannot capture the chaotic curvature.
/// KARC's reservoir delay-embedding SHOULD produce smaller residuals →
/// tighter intervals at the same coverage. This is KARC's home turf.
///
/// Gate: KARC should BEAT the floor on CRPS and Winkler. A TIE is
/// acceptable (KARC may not have enough capacity at M=8, K=4 to fully
/// capture Lorenz). A LOSES verdict would be a significant negative
/// finding worth recording.
#[test]
fn floor_comparison_lorenz_x() {
    // Generate the chaotic trajectory + wrap as a corpus.
    let n_total = 2_000;
    let n_transient = 1_000;
    let traj = generate_lorenz_x(n_transient, n_total, 0.02);
    // Warmup: needs to be ≥ POOL_CAPACITY (256) so the residual pool fills,
    // AND large enough to fit KARC (the constructor uses corpus[..warmup]).
    let warmup = 600;
    let corpus = TrajectoryCorpus::from_slice("lorenz_x_dt0.02", &traj, warmup);

    let mut adapter = KarcOverlayAdapter::<K4>::new_fitted(&corpus.values[..warmup], ALPHA);
    let report: FloorComparisonReport =
        run_floor_comparison(&mut adapter, &corpus.values, ALPHA, warmup, &corpus.name);

    println!("── KARC+overlay vs floor on {} ──", corpus.name);
    println!("{report:.?}");

    assert!(report.n_scored > 500, "n_scored = {}", report.n_scored);

    // Same coverage invariant: KARC's overlay uses the same conformal math
    // as the floor, so coverage should match within noise. A coverage gap
    // > 0.10 indicates a wiring bug.
    let floor_cov = report.floor.coverage;
    let prim_cov = report.primitive.coverage;
    let cov_delta = (prim_cov - floor_cov).abs();
    assert!(
        cov_delta < 0.10,
        "coverage delta too large: floor={floor_cov:.4}, prim={prim_cov:.4}, |Δ|={cov_delta:.4}"
    );

    // Verdict: we EXPECT BeatsFloor or at worst TiesFloor here. LOSES would
    // be a real negative finding (KARC's reservoir failing on its showcase
    // chaotic system). Mixed is acceptable if CRPS wins but Winkler ties.
    assert!(
        matches!(
            report.overall,
            OverallVerdict::BeatsFloor | OverallVerdict::TiesFloor | OverallVerdict::Mixed
        ),
        "KARC+overlay verdict on Lorenz-x: {:?} — expected Beats/Ties/Mixed (KARC's home turf)",
        report.overall
    );

    // Soft check: KARC's mean CRPS should be ≤ floor's mean CRPS + 10%.
    // (Roughly: KARC's reservoir should at least not catastrophically
    // under-perform on its canonical chaotic target.)
    let prim_crps = report.primitive.mean_crps_interval;
    let floor_crps = report.floor.mean_crps_interval;
    assert!(
        prim_crps <= floor_crps * 1.10,
        "KARC CRPS {prim_crps:.6} > floor CRPS {floor_crps:.6} × 1.10 — significant regression on Lorenz"
    );
}

// ===== K-sweep follow-up: K=12 tests (RUN 2026-07-20) ====================
//
// The original T7 measurement found KARC+overlay with K=4 LOSES on
// stationary_seasonal (period 12). The post-hoc explanation was "K=4 is
// too shallow to capture a period-12 cycle". The K=12 tests below verify
// that explanation empirically by re-running with K matching the period.
//
// MEASURED RESULT (recorded in the file header verdict table):
//   - K=12 on seasonal: LosesToFloor WORSE (CRPS ratio 5.74 → 20.26)
//     → REFUTES the "K=4 too shallow" hypothesis. The scope-limit is
//       structural: KARC's Chebyshev basis + ridge-fit doesn't fit periodic
//       data regardless of delay depth. More context → more overfitting
//       capacity → worse point forecast at curvature changes.
//   - K=12 on Lorenz: BeatsFloor HARDER (CRPS ratio 0.0047 → 0.0018)
//     → more delay context genuinely helps on chaotic signals (longer
//       trajectory memory captures more of the attractor dynamics).
//
// Production guidance: KARC+overlay is a chaotic-regime specialist. Pick K
// by how much trajectory memory the chaotic signal benefits from. For
// periodic/seasonal data, use the floor (`SeasonalNaiveForecaster`)
// directly — KARC cannot match it regardless of K.

/// K=12 on stationary_seasonal (period 12): the headline K-sweep measurement.
///
/// **Measured verdict: LosesToFloor WORSE than K=4** (CRPS ratio 20.26 vs
/// K=4's 5.74; Winkler ratio 36.97 vs K=4's 21.73; coverage 0.911 vs floor
/// 0.939 — still calibrated, no false-confidence signature). This REFUTES
/// the prior session's "K=4 too shallow for period-12" hypothesis: matching
/// K to the period makes KARC LOSE WORSE on seasonal, not better. The
/// scope-limit is **structural** — KARC's Chebyshev basis + ridge-fit
/// architecture doesn't fit periodic data regardless of delay depth. More
/// delay context gives the ridge fit more overfitting capacity, amplifying
/// the systematic point-forecast errors at curvature changes.
///
/// Gate: we ACCEPT LosesToFloor here (measurement, not a pass/fail gate).
/// The hard gates are the same as the K=4 tests — coverage stays within
/// 0.10 of floor (no calibration break), the verdict is one of the scored
/// outcomes (not NotApplicable).
#[test]
fn floor_comparison_stationary_seasonal_k12() {
    let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 800, 0xCAFE_BABE);
    let warmup = corpus.recommended_warmup;

    let mut adapter = KarcOverlayAdapter::<K12>::new_fitted(&corpus.values[..warmup], ALPHA);
    let report: FloorComparisonReport =
        run_floor_comparison(&mut adapter, &corpus.values, ALPHA, warmup, &corpus.name);

    println!("── KARC+overlay (K=12) vs floor on {} ──", corpus.name);
    println!("{report:.?}");

    // Sanity: at least 100 scored steps.
    assert!(report.n_scored > 100, "n_scored = {}", report.n_scored);

    // Hard gate: coverage must stay calibrated (same invariant as K=4 tests).
    let floor_cov = report.floor.coverage;
    let prim_cov = report.primitive.coverage;
    let cov_delta = (prim_cov - floor_cov).abs();
    assert!(
        cov_delta < 0.10,
        "coverage delta too large: floor={floor_cov:.4}, prim={prim_cov:.4}, |Δ|={cov_delta:.4}"
    );

    // Accept any scored verdict — this is a measurement, not a gate.
    assert!(
        matches!(
            report.overall,
            OverallVerdict::BeatsFloor
                | OverallVerdict::TiesFloor
                | OverallVerdict::Mixed
                | OverallVerdict::LosesToFloor
        ),
        "KARC+overlay (K=12) verdict on seasonal: {:?} — expected one of the scored outcomes",
        report.overall
    );

    // Log the CRPS ratio so the test output records the measurement. This is
    // the load-bearing number for the scope-limit question.
    let prim_crps = report.primitive.mean_crps_interval;
    let floor_crps = report.floor.mean_crps_interval;
    let crps_ratio = if floor_crps > 1e-12 {
        prim_crps / floor_crps
    } else {
        f32::NAN
    };
    println!(
        "K=12 stationary_seasonal CRPS ratio (prim/floor): {crps_ratio:.4} (K=4 baseline was 5.74)"
    );
}

/// K=12 on Lorenz-x (chaotic): sanity check that deeper delay IMPROVES
/// KARC's chaotic-regime performance (the inverse of the seasonal case).
///
/// **Measured verdict: BeatsFloor HARDER than K=4** (CRPS ratio 0.0018 vs
/// K=4's 0.0047, ~2.6× tighter; coverage 0.933 vs floor 0.943 — calibrated).
/// More delay context genuinely helps on chaotic signals: longer trajectory
/// memory captures more of the Lorenz attractor's structure. This is the
/// mirror image of the seasonal result — same architecture, same K change,
/// opposite effect because the signal class is different.
///
/// Gate: K=12 BEATS the floor on Lorenz (as expected). The test would only
/// FAIL if K=12 catastrophically regressed — e.g. over-fitting the warmup
/// slice and failing to generalize. The measured result confirms the
/// chaotic-regime specialist role documented in the file header.
#[test]
fn floor_comparison_lorenz_x_k12() {
    let n_total = 2_000;
    let n_transient = 1_000;
    let traj = generate_lorenz_x(n_transient, n_total, 0.02);
    let warmup = 600;
    let corpus = TrajectoryCorpus::from_slice("lorenz_x_dt0.02", &traj, warmup);

    let mut adapter = KarcOverlayAdapter::<K12>::new_fitted(&corpus.values[..warmup], ALPHA);
    let report: FloorComparisonReport =
        run_floor_comparison(&mut adapter, &corpus.values, ALPHA, warmup, &corpus.name);

    println!("── KARC+overlay (K=12) vs floor on {} ──", corpus.name);
    println!("{report:.?}");

    assert!(report.n_scored > 500, "n_scored = {}", report.n_scored);

    // Same coverage invariant.
    let floor_cov = report.floor.coverage;
    let prim_cov = report.primitive.coverage;
    let cov_delta = (prim_cov - floor_cov).abs();
    assert!(
        cov_delta < 0.10,
        "coverage delta too large: floor={floor_cov:.4}, prim={prim_cov:.4}, |Δ|={cov_delta:.4}"
    );

    // Verdict: still expect Beats/Ties/Mixed on Lorenz with K=12. A LOSES
    // would be a significant negative finding.
    assert!(
        matches!(
            report.overall,
            OverallVerdict::BeatsFloor | OverallVerdict::TiesFloor | OverallVerdict::Mixed
        ),
        "KARC+overlay (K=12) verdict on Lorenz-x: {:?} — expected Beats/Ties/Mixed",
        report.overall
    );

    // Log the CRPS ratio.
    let prim_crps = report.primitive.mean_crps_interval;
    let floor_crps = report.floor.mean_crps_interval;
    let crps_ratio = if floor_crps > 1e-12 {
        prim_crps / floor_crps
    } else {
        f32::NAN
    };
    println!("K=12 Lorenz-x CRPS ratio (prim/floor): {crps_ratio:.4} (K=4 baseline was 0.0047)");
}
