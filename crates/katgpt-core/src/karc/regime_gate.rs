//! KARC Regime Gate — closed-form residual-MSE mux between KARC and
//! Seasonal forecasters (Plan 556 Phase 1; revised to MSE 2026-07-20 after
//! Plan 514 Phase 1 integration surfaced a variance-only failure mode).
//!
//! # The problem this fixes
//!
//! KARC is a chaotic-regime specialist (see `.benchmarks/010_report_the_floor_consolidated.md`
//! T7 K-sweep, 2026-07-20): on Lorenz-x chaotic signals KARC+conformal-overlay
//! beats the seasonal-naive floor at CRPS ratio 0.0047 (K=4) → 0.0018 (K=12),
//! but on stationary seasonal data KARC *loses* at CRPS ratio 5.74 (K=4) →
//! 20.26 (K=12) — *worse* with more delay context. The K-sweep refuted the
//! "K=4 too shallow" hypothesis: the scope-limit is **structural** (KARC's
//! Chebyshev/Fourier basis + ridge-fit doesn't fit periodic data regardless
//! of K).
//!
//! The fix is not to make KARC fit periodic data — it is to **route each
//! per-NPC trajectory to the right forecaster for its current regime**. This
//! gate does that with zero training: it compares rolling residual MSE
//! (mean squared error, = variance + bias²) of the two forecasters and picks
//! the lower-MSE one, with a sigmoid confidence to avoid flip-flop.
//!
//! # Why MSE, not variance
//!
//! The original Plan 556 spec said "rolling residual variance". The first
//! runtime integration (Plan 514 Phase 1) exposed the failure mode: a
//! consistently-biased forecaster has variance 0 but a large error. The gate
//! would prefer the biased forecaster over a more-accurate-but-volatile one —
//! categorically wrong for the regime-mux use case. MSE = Var(r) + (E[r])²
//! penalizes both bias and dispersion; for the regime-mux question "which
//! forecaster has smaller error on this regime?", MSE is the right metric.
//! Variance alone is insufficient.
//!
//! # Algorithm
//!
//! Each tick, both KARC and Seasonal produce a point forecast; after the
//! ground truth is observed the caller computes two residuals and pushes
//! them to [`KarcRegimeGate::observe_residuals`]. The gate maintains a
//! [`WelfordVariance`] accumulator per forecaster. On [`decide`](KarcRegimeGate::decide):
//!
//! ```text
//! if n < min_pool:           return Seasonal  (cold-start floor)
//! else:                      pick argmin(MSE_karc, MSE_seasonal)
//!                            confidence = sigmoid(β · |MSE_high − MSE_low|)
//! ```
//!
//! Cold-start defaults to `Seasonal` because `SeasonalNaiveForecaster` is the
//! canonical conformal-naive floor (Plan 340, Issue 010 "Report the Floor").
//! The gate never claims to beat the floor until it has empirical evidence.
//!
//! # Modelless
//!
//! Pure empirical-statistics + sigmoid. No training, no learned gate weights,
//! no gradient descent. Cold-start is a single constant (`min_pool`); the
//! sigmoid inverse temperature `β` is a designer hyperparameter, not learned.
//!
//! # Latent-only sync boundary
//!
//! `KarcRegimeGate` holds per-NPC local state (two variance accumulators +
//! `min_pool` + `β`). The resulting `RegimeVerdict` is consumed locally by
//! the runtime to pick which forecast to apply; the verdict itself does not
//! cross `SyncBlock`. Per the AGENTS.md latent-vs-raw rule, only the chosen
//! 5-scalar emotion projection crosses sync — the gate's internal state never
//! does.
//!
//! # Allocation
//!
//! Zero. `WelfordVariance` is `(count, mean, M2)` = 3 fields. `KarcRegimeGate`
//! holds two `WelfordVariance` + two `usize`/`f32` config fields. All methods
//! are stack-only.
//!
//! # Plan 556 GOAT gate
//!
//! - **G1**: on Lorenz-63 residuals → ≥95% ticks route to KARC; on stationary
//!   seasonal (period=12) residuals → ≥95% route to Seasonal.
//! - **G2**: `decide()` ≤ 50 ns/call (pure Welford + sigmoid + branch).
//! - **G3**: enabling `karc_regime_gate` does not perturb `karc_forecaster`
//!   forecasts (the gate is a pure consumer of residual signals; verified by
//!   `tests/conformal_karc_no_regression.rs` which already enforces this
//!   composition pattern).
//! - **G4**: 0 allocs/100 calls on the hot path.
//! - **G1-extension (MSE revision)**: a consistently-biased forecaster
//!   (variance 0, large mean) must NOT win over a less-biased one. The
//!   `seasonal_low_residual_routes_to_seasonal` regression test in
//!   `riir-engine::karc_bridge::regime_mux` enforces this end-to-end.
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/556_karc_mitigations_open_primitives.md` (Phase 1)
//! - **Bench:** `katgpt-rs/.benchmarks/010_report_the_floor_consolidated.md` §T7
//!   (the structural scope-limit finding that motivates this gate).
//! - **Floor:** `crates/katgpt-core/src/conformal/seasonal.rs::SeasonalNaiveForecaster`
//!   (the conformal-naive floor; cold-start target).
//! - **Runtime integration:** `riir-ai/.plans/514_karc_mitigations_runtime.md`
//!   Phase 1 (per-NPC regime mux wiring).

#[cfg(any(test, feature = "estimators"))]
mod imp {
    // The Welford accumulator moved to the crate-root `welford` module
    // (Plan: reflex 008 / Issue 055 substrate pass, 2026-09-30) so a second
    // substrate consumer reaches it without the `karc_regime_gate` feature
    // chain. Re-exported here — the historical path resolves unchanged and
    // there is exactly one definition.
    pub use crate::welford::WelfordVariance;

    /// Which forecaster the gate currently prefers. `#[repr(u8)]` so the
    /// verdict stays sync-friendly if a downstream consumer wants to commit
    /// the routing history (latent-only — the verdict itself never enters
    /// `SyncBlock`).
    ///
    /// Named `KarcRegime` (not `Regime`) to avoid collision with
    /// `mean_field_regime::Regime` at the crate root re-export.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(u8)]
    pub enum KarcRegime {
        /// KARC delay-basis ridge (chaotic-regime specialist).
        Karc = 0,
        /// Seasonal-naive floor (periodic-regime specialist; cold-start default).
        Seasonal = 1,
    }

    /// Gate verdict returned by [`KarcRegimeGate::decide`].
    ///
    /// `confidence ∈ [0.5, 1.0]` — `sigmoid(β · |ΔMSE|)`; 0.5 at a tie, →1.0
    /// as the MSE gap widens. Never below 0.5 by construction (the gate
    /// reports the winning side's confidence, not a directional sign).
    ///
    /// The fields hold MSE (variance + bias²), not variance alone — see the
    /// module-level "Why MSE, not variance" section for the rationale. The
    /// field names use `mse_*` (not `sigma_sq_*`) to make the semantics
    /// unambiguous at the call site.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct RegimeVerdict {
        /// Which forecaster to apply this tick.
        pub preferred: KarcRegime,
        /// `sigmoid(β · |MSE_high − MSE_low|)`, in `[0.5, 1.0]`.
        pub confidence: f32,
        /// KARC's current MSE vs zero target (or `f32::INFINITY` if undefined).
        pub mse_karc: f32,
        /// Seasonal's current MSE vs zero target (or `f32::INFINITY` if undefined).
        pub mse_seas: f32,
        /// Number of accumulated residual pairs.
        pub n: usize,
    }

    /// Closed-form residual-MSE regime gate.
    ///
    /// Holds two [`WelfordVariance`] accumulators (one per forecaster). The
    /// caller pushes both residuals per tick via [`observe_residuals`]; the
    /// runtime then calls [`decide`] to pick which forecast to apply.
    ///
    /// [`observe_residuals`]: KarcRegimeGate::observe_residuals
    /// [`decide`]: KarcRegimeGate::decide
    #[derive(Clone, Copy, Debug)]
    pub struct KarcRegimeGate {
        karc_var: WelfordVariance,
        seas_var: WelfordVariance,
        /// Cold-start floor: until `n >= min_pool`, [`decide`] always returns
        /// [`KarcRegime::Seasonal`] (the floor). Default 16.
        min_pool: usize,
        /// Sigmoid inverse temperature. Default 8.0 — picks a moderate
        /// transition: at `|ΔMSE| = 0.5`, confidence ≈ 0.92; at `|ΔMSE| = 0.1`,
        /// confidence ≈ 0.69.
        beta: f32,
    }

    impl Default for KarcRegimeGate {
        #[inline]
        fn default() -> Self {
            Self::new()
        }
    }

    impl KarcRegimeGate {
        /// New gate with default config (`min_pool = 16`, `β = 8.0`).
        #[inline]
        pub const fn new() -> Self {
            Self {
                karc_var: WelfordVariance::new(),
                seas_var: WelfordVariance::new(),
                min_pool: 16,
                beta: 8.0,
            }
        }

        /// New gate with custom config.
        #[inline]
        pub const fn with_config(min_pool: usize, beta: f32) -> Self {
            Self {
                karc_var: WelfordVariance::new(),
                seas_var: WelfordVariance::new(),
                min_pool,
                beta,
            }
        }

        /// Current number of accumulated residual pairs (the matched-pair
        /// count — `min(karc_var.n, seas_var.n)` because NaN rejection can
        /// desync the two counters).
        #[inline]
        pub const fn n(&self) -> usize {
            // `usize::min` is not yet stable as a const trait method
            // (rust-lang/rust#143874) — use the explicit comparison.
            let k = self.karc_var.n();
            let s = self.seas_var.n();
            if k <= s { k } else { s }
        }

        /// Push a residual pair. Either residual may be NaN (no observation
        /// for that forecaster this tick) — NaN is silently rejected by the
        /// accumulator.
        #[inline]
        pub fn observe_residuals(&mut self, karc_residual: f32, seasonal_residual: f32) {
            self.karc_var.observe(karc_residual);
            self.seas_var.observe(seasonal_residual);
        }

        /// Reset both accumulators (e.g. on NPC freeze/thaw).
        #[inline]
        pub fn reset(&mut self) {
            self.karc_var.reset();
            self.seas_var.reset();
        }

        /// Current gate verdict.
        ///
        /// **Cold-start:** until `n >= min_pool`, returns `Regime::Seasonal`
        /// with `confidence = 0.5` (the floor — no claim until evidence).
        ///
        /// **Steady-state:** picks the lower-MSE forecaster; confidence
        /// is `sigmoid(β · |MSE_high − MSE_low|)` ∈ `[0.5, 1.0]`.
        ///
        /// **Edge case:** if both MSEs are undefined (n < 1), returns
        /// `Seasonal` at `confidence = 0.5`.
        ///
        /// **MSE vs variance:** uses MSE (= variance + bias²) — not variance
        /// alone — so a consistently-biased forecaster with variance 0 still
        /// gets a large MSE and the gate correctly routes away from it.
        #[inline]
        pub fn decide(&self) -> RegimeVerdict {
            // Cold-start floor.
            if self.n() < self.min_pool {
                return RegimeVerdict {
                    preferred: KarcRegime::Seasonal,
                    confidence: 0.5,
                    mse_karc: self.karc_var.mse().unwrap_or(f32::INFINITY),
                    mse_seas: self.seas_var.mse().unwrap_or(f32::INFINITY),
                    n: self.n(),
                };
            }

            // Both MSEs must be defined (n >= min_pool >= 1).
            let mse_karc = self.karc_var.mse().unwrap_or(f32::INFINITY);
            let mse_seas = self.seas_var.mse().unwrap_or(f32::INFINITY);

            let (preferred, gap) = if mse_karc <= mse_seas {
                (KarcRegime::Karc, mse_seas - mse_karc)
            } else {
                (KarcRegime::Seasonal, mse_karc - mse_seas)
            };

            // sigmoid(β · gap) ∈ (0.5, 1.0]. Use the standard logistic:
            //   sigmoid(x) = 1 / (1 + exp(-x))
            // gap >= 0 always (abs-difference), so x = β·gap >= 0 and
            // sigmoid ∈ [0.5, 1.0).
            let x = self.beta * gap;
            let confidence = 1.0 / (1.0 + (-x).exp());

            RegimeVerdict {
                preferred,
                confidence,
                mse_karc,
                mse_seas,
                n: self.n(),
            }
        }
    }
}

#[cfg(feature = "estimators")]
pub use imp::{KarcRegime, KarcRegimeGate, RegimeVerdict, WelfordVariance};

#[cfg(test)]
mod tests {
    use super::imp::*;
    // Local alias so the tests can use the short name `Regime` even though
    // the public type is `KarcRegime` (to avoid colliding with
    // `mean_field_regime::Regime` at the crate root).
    type Regime = KarcRegime;

    /// Helper: build a Lorenz-63-style residual stream where KARC's residuals
    /// are smaller than Seasonal's (KARC is the better forecaster for chaotic
    /// signals). Returns `(karc_residuals, seas_residuals)` vectors.
    fn lorenz_like_residuals(n: usize) -> (Vec<f32>, Vec<f32>) {
        use std::f32::consts::TAU;
        // KARC tracks the chaotic signal well → small residuals.
        // Seasonal-naive assumes periodicity → larger residuals.
        let mut k = Vec::with_capacity(n);
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 * 0.05;
            // Pseudo-chaotic: sum of incommensurate sinusoids + drift.
            let signal = (t * 1.1).sin() + (t * 2.7).cos() * 0.7 + t * 0.01;
            let karc_residual = (t * 0.3).sin() * 0.02; // tiny
            let seas_residual = signal * 0.5; // large (seasonal-naive can't follow chaos)
            k.push(karc_residual);
            s.push(seas_residual);
        }
        // Make sure variance(karc) << variance(seas) — sanity-anchor the test.
        let _ = TAU; // suppress unused-import warning if std::f32::consts::TAU never read
        (k, s)
    }

    /// Helper: stationary seasonal residuals — Seasonal-naive is the better
    /// forecaster here (KARC's basis can't fit the periodicity per Bench 010).
    fn seasonal_like_residuals(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut k = Vec::with_capacity(n);
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            // KARC residuals: structured periodicity the ridge can't fit →
            // medium-magnitude residuals.
            let karc_residual =
                ((i as f32 / 12.0).floor() * 0.0).exp() * (i as f32 * 0.5).sin() * 0.3;
            // Seasonal residuals: small (period-12, seasonal-naive at m=12 nails it).
            let seas_residual = (i as f32 * 0.001).sin() * 0.01;
            k.push(karc_residual);
            s.push(seas_residual);
        }
        (k, s)
    }

    #[test]
    fn welford_basic() {
        let mut w = WelfordVariance::new();
        assert_eq!(w.n(), 0);
        assert!(w.variance().is_none());

        w.observe(1.0);
        assert_eq!(w.n(), 1);
        assert!(w.variance().is_none()); // need >= 2

        w.observe(2.0);
        w.observe(3.0);
        assert_eq!(w.n(), 3);
        // Sample variance of [1,2,3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / (3-1) = 1.0
        let v = w.variance().unwrap();
        assert!((v - 1.0).abs() < 1e-5, "variance = {v}");
    }

    #[test]
    fn welford_nan_safe() {
        let mut w = WelfordVariance::new();
        w.observe(f32::NAN);
        assert_eq!(w.n(), 0, "NaN rejected");
        w.observe(1.0);
        w.observe(2.0);
        assert_eq!(w.n(), 2);
        w.observe(f32::NAN); // mid-stream NaN
        assert_eq!(w.n(), 2, "NaN rejected mid-stream");
    }

    #[test]
    fn welford_reset() {
        let mut w = WelfordVariance::new();
        w.observe(1.0);
        w.observe(2.0);
        assert_eq!(w.n(), 2);
        w.reset();
        assert_eq!(w.n(), 0);
        assert!(w.variance().is_none());
    }

    #[test]
    fn gate_cold_start_returns_seasonal() {
        let mut g = KarcRegimeGate::with_config(16, 8.0);
        // Push fewer than min_pool residuals — gate must always return Seasonal.
        for i in 0..10 {
            g.observe_residuals(i as f32 * 0.01, i as f32 * 0.5);
        }
        let v = g.decide();
        assert_eq!(v.preferred, Regime::Seasonal, "cold-start → Seasonal");
        assert_eq!(v.confidence, 0.5, "cold-start confidence = 0.5");
        assert_eq!(v.n, 10);
    }

    #[test]
    fn gate_routes_lorenz_to_karc() {
        // Plan 556 G1: on Lorenz-like residuals, gate routes ≥95% to KARC.
        let (karc_res, seas_res) = lorenz_like_residuals(500);
        let mut g = KarcRegimeGate::with_config(16, 8.0);
        let mut karc_votes = 0usize;
        let total = karc_res.len();
        for (kr, sr) in karc_res.iter().copied().zip(seas_res.iter().copied()) {
            g.observe_residuals(kr, sr);
            let v = g.decide();
            if v.preferred == Regime::Karc {
                karc_votes += 1;
            }
        }
        let karc_frac = karc_votes as f32 / total as f32;
        assert!(
            karc_frac >= 0.95,
            "G1 Lorenz → KARC fraction = {karc_frac:.3}, expected ≥ 0.95"
        );
    }

    #[test]
    fn gate_routes_seasonal_to_seasonal() {
        // Plan 556 G1: on stationary seasonal residuals, gate routes ≥95%
        // to Seasonal.
        let (karc_res, seas_res) = seasonal_like_residuals(500);
        let mut g = KarcRegimeGate::with_config(16, 8.0);
        let mut seas_votes = 0usize;
        let total = karc_res.len();
        for (kr, sr) in karc_res.iter().copied().zip(seas_res.iter().copied()) {
            g.observe_residuals(kr, sr);
            let v = g.decide();
            if v.preferred == Regime::Seasonal {
                seas_votes += 1;
            }
        }
        let seas_frac = seas_votes as f32 / total as f32;
        assert!(
            seas_frac >= 0.95,
            "G1 Seasonal → Seasonal fraction = {seas_frac:.3}, expected ≥ 0.95"
        );
    }

    #[test]
    fn gate_confidence_at_tie_is_half() {
        // Plan 556: at variance tie, confidence ≈ 0.5 (sigmoid(0) = 0.5).
        let mut g = KarcRegimeGate::with_config(2, 8.0);
        // Push identical residual streams → variance tie.
        for i in 0..20 {
            let r = (i as f32) * 0.1;
            g.observe_residuals(r, r);
        }
        let v = g.decide();
        assert!(
            (v.confidence - 0.5).abs() < 1e-3,
            "tie confidence = {}, expected ~0.5",
            v.confidence
        );
        // Tie-break goes to whichever is "lower" — Karc wins ties due to `<=`.
        assert_eq!(v.preferred, Regime::Karc, "tie-break to Karc");
    }

    #[test]
    fn gate_confidence_grows_with_gap() {
        // Larger variance gap → higher confidence.
        let mut g_small = KarcRegimeGate::with_config(2, 8.0);
        let mut g_large = KarcRegimeGate::with_config(2, 8.0);
        for i in 0..20 {
            let t = i as f32 * 0.1;
            // Small-gap stream: |MSE_karc − MSE_seas| ≈ 0.1
            g_small.observe_residuals(t.sin() * 0.1, t.sin() * 0.15);
            // Large-gap stream: |MSE_karc − MSE_seas| ≈ 5.0
            g_large.observe_residuals(t.sin() * 0.1, t.sin() * 2.0);
        }
        let v_small = g_small.decide();
        let v_large = g_large.decide();
        assert!(
            v_large.confidence > v_small.confidence,
            "larger gap → higher confidence ({} > {})",
            v_large.confidence,
            v_small.confidence
        );
    }

    #[test]
    fn gate_no_flip_flop_at_borderline() {
        // Sigmoid smoothing: at near-tie, the gate should not flip-flop
        // every tick. Stream where variances slowly cross — the gate's
        // preferred should change at most a few times, not every tick.
        let mut g = KarcRegimeGate::with_config(2, 8.0);
        let mut flips = 0usize;
        let mut last_preferred: Option<Regime> = None;
        for i in 0..200 {
            let t = i as f32 * 0.1;
            // KARC residual grows slowly, seasonal shrinks slowly.
            let karc_res = t.sin() * (0.1 + i as f32 * 0.005);
            let seas_res = t.sin() * (0.5 - i as f32 * 0.002);
            g.observe_residuals(karc_res, seas_res);
            let v = g.decide();
            if let Some(prev) = last_preferred
                && prev != v.preferred
            {
                flips += 1;
            }
            last_preferred = Some(v.preferred);
        }
        // Expect few flips (sigmoid smoothing + Welford inertia) — not 200.
        assert!(
            flips < 50,
            "expected few flips due to sigmoid smoothing, got {flips}/200"
        );
    }

    #[test]
    fn gate_decide_is_in_idempotent() {
        // `decide()` is a pure read — calling it twice returns identical
        // verdicts. Required for the runtime to safely re-read after a
        // boundary event without changing state.
        let mut g = KarcRegimeGate::with_config(2, 8.0);
        for i in 0..10 {
            g.observe_residuals(i as f32 * 0.1, i as f32 * 0.2);
        }
        let v1 = g.decide();
        let v2 = g.decide();
        assert_eq!(v1, v2, "decide() is read-only");
    }

    #[test]
    fn gate_reset_clears_state() {
        let mut g = KarcRegimeGate::with_config(2, 8.0);
        for i in 0..10 {
            g.observe_residuals(i as f32 * 0.1, i as f32 * 0.2);
        }
        assert!(g.n() > 0);
        g.reset();
        assert_eq!(g.n(), 0);
        // After reset, decide() returns cold-start Seasonal.
        let v = g.decide();
        assert_eq!(v.preferred, KarcRegime::Seasonal);
    }

    #[test]
    fn gate_nan_residuals_are_no_op() {
        let mut g = KarcRegimeGate::with_config(2, 8.0);
        g.observe_residuals(f32::NAN, f32::NAN);
        assert_eq!(g.n(), 0, "NaN pair is no-op");
        // One side NaN, the other real — only the real side accumulates.
        g.observe_residuals(0.5, f32::NAN);
        // The "matched-pair" count uses min of the two, so n=0 here.
        assert_eq!(g.n(), 0, "mismatched NaN pair → n=0 (no quorum)");
    }
}
