//! `tether` — closed-form outcome-fit estimator blend (Issue 675).
//!
//! Source: arXiv:2608.16739 "Le Critique: Privileged Value Functions for LLM
//! Reinforcement Learning" (the TETHER baseline), distilled at
//! `riir-train/.research/426_Le_Critique_PVF_TETHER.md`. Public-tier math
//! (forecast combination, Bates–Granger lineage); the primitive's value is
//! the **never-worse guarantee + lag/EMA contract encoded as an API shape**,
//! not new math.
//!
//! ## The defect this cures
//!
//! The stack blends two estimators in several places with a **fixed or
//! threshold rule**, never fit against realized outcomes: a fixed `alpha` in
//! `DualLeoMixer` (Bench 553: fixed α + harmful student measured −100%), a
//! fixed confidence threshold in `KvRoutingConfig::blend_factor`, and a
//! hand-pinned `W_EVO`/`W_RATE` pair in riir-clippy's selection blend. A
//! fixed weight cannot express that the two axes carry different amounts of
//! information at different points of a stream's lifecycle.
//!
//! ## The mechanism
//!
//! Blend `b(ρ) = (1−ρ)·p1 + ρ·p2`, fit ρ per window by ordinary least squares
//! against realized outcomes:
//!
//! ```text
//! A_B = R − p1            (the residual the p1-only baseline leaves)
//! Δ   = p2 − p1           (the direction the p2 endpoint moves in)
//! ρ*  = clip[0,1]( Σ A_B·Δ / Σ Δ² )
//! ```
//!
//! with the **exact in-sample guarantee** `SSE(ρ*) ≤ min(SSE(p1-only),
//! SSE(p2-only))` — unconditional, because SSE is convex in ρ and both
//! endpoints (ρ = 0, ρ = 1) are feasible points of the clipped domain.
//! [`fit_rho`] + [`sse`] make that assertable. EMA smoothing
//! `ρ_k = d·ρ_{k−1} + (1−d)·ρ̂_k` bounds how far one noisy window can move
//! the applied weight.
//!
//! ## The lag law, encoded as API shape
//!
//! Fitting ρ on a window and applying it to the SAME window makes the blend a
//! function of its own realized returns — bias by construction (the paper's
//! admissibility violation). So [`TetherBlend`] makes same-window application
//! **unrepresentable**: [`observe`](TetherBlend::observe) only ever touches
//! the accumulators, and [`rho`](TetherBlend::rho) only ever reads the value
//! published by a *closed* window. The window closes inside `observe` AFTER
//! the sample is folded in, so the new ρ is visible only to later
//! selections. There is no method that applies an open window's fit;
//! [`open_rho_hat`](TetherBlend::open_rho_hat) is telemetry only and consumed
//! by nothing on this type.
//!
//! ## Admissibility vocabulary (doc-level rule)
//!
//! Which outcome streams may lawfully feed the fit:
//!
//! - **Own-future** outcomes of the decisions the blend influences:
//!   **inadmissible** (the lag law exists for exactly this).
//! - **Independent-instance sibling** outcomes (another agent's realized
//!   return under the same estimator pair): **admissible** — not caused by
//!   this consumer's selection.
//! - **Contested shared encounters** (outcomes whose value both instances
//!   influenced): **violate** admissibility — neither side's outcome is
//!   exogenous to the other. Do not feed them to the fit.
//!
//! The Monte-Carlo fixture pair in `tests` pins the rule concretely: the mean
//! advantage is preserved under an admissible conditioning signal and
//! measurably biased under a leaking one.
//!
//! ## HAZARD 1 — Report the Floor (Issue 010 rule)
//!
//! Blending a UQ primitive WITH the conformal-naive floor does **not**
//! discharge the promotion gate — the primitive itself must beat the floor
//! unblended. Runtime blending is orthogonal to the promotion gate; a
//! floor+noise blend must not be citable as "beats the floor".
//!
//! ## HAZARD 2 — ρ\* minimizes a PREDICTION loss (measured negative on a
//! ranking consumer, Bench 042)
//!
//! riir-clippy ran the full A/B (riir-clippy `.benchmarks/042`, commit
//! `5494dbe`): the in-sample guarantee held exactly on real recorded
//! streams, ρ drift was large and reproducible, alloc 0, bit-exact
//! determinism — and the consumer's **ranking** metric got significantly
//! WORSE. ρ\* is a gain only where the consumer's metric IS prediction
//! error. A consumer whose metric is a ranking or an argmax (selection,
//! routing, retrieval order) can get a strictly better SSE and a strictly
//! worse outcome — especially when one endpoint is a *deliberately biased*
//! estimator (a conservative lower bound is a bad predictor and a good
//! ranker, so the fit correctly downweights it and incorrectly loses the
//! ranking). Fit ρ against the consumer's own metric, or ship a
//! ranking-aware objective. Known-good consumer class: value-prediction
//! baselines (riir-train `loss_grpo`, Plan 345). Known-negative consumer:
//! selection blends (Bench 042, kept as the reproducible artifact).
//!
//! ## G4 (allocation)
//!
//! Scalar accumulators only — no heap, ever, no warm-up phase to amortize.
//!
//! tether — closed-form outcome-fit estimator blend (Issue 675, Research 426
//! via arXiv:2608.16739 "Le Critique" TETHER baseline): ρ* per window by OLS
//! against realized outcomes with the exact in-sample never-worse guarantee,
//! the lag law encoded as API shape (same-window application unrepresentable),
//! EMA smoothing, an explained-variance accumulator + control-variate gate,
//! and the horizon-decay LUT λ = c^(1/L). Two documented hazards in-source:
//! Report-the-Floor (blending does not discharge the promotion gate) and
//! prediction-vs-ranking (measured NEGATIVE on a ranking consumer, Bench 042
//! — fit ρ against the consumer's own metric). Consumers: riir-train
//! loss_grpo TETHER baseline (Plan 345). Opt-in; G1–G4 PASS (Bench 670).

/// Default initial mixing weight: **0.0** — no evidence yet, so do not blend.
///
/// The cold-start behaviour is therefore "p1 only until the first window
/// closes", the no-regression-by-construction choice for a fresh primitive.
/// Consumers with a historical prior pin it via [`TetherBlend::with_params`]
/// (riir-clippy's selection blend pins its legacy `0.4`; Plan 345 pins
/// `ρ₀ = 0` with EV-gated movement).
pub const DEFAULT_RHO: f32 = 0.0;

/// Default EMA retention on the per-window fit (`d` in the paper's
/// `ρ_k = d·ρ_{k−1} + (1−d)·ρ̂_k`).
///
/// 0.95 is the paper's own smoothing constant: a single noisy window moves
/// the applied ρ by 5% of its distance, so a starved window cannot swing a
/// decision.
pub const DEFAULT_EMA_DECAY: f32 = 0.95;

/// Default observations per window.
///
/// The window is the fit's sample size, so it trades tracking speed against
/// fit variance. 32 is the generic default; consumers pin their own by
/// cadence (riir-clippy's heal loop uses 16 for several windows per run;
/// Plan 345's warmup uses 20 updates). A window must contain at least a
/// handful of `p1 ≠ p2` disagreements or the degenerate guard holds the
/// previous ρ forever.
pub const DEFAULT_WINDOW: u32 = 32;

/// `ΣΔ²` floor below which a window is **degenerate** — the two endpoints
/// agreed on every sample in it, so the data contains no information about
/// their mixing weight and the OLS quotient is `0/0`.
///
/// The guard holds the previous ρ rather than resetting it: worst case the
/// adaptive mode converges to fixed-blend behaviour, never worse.
pub const DEGENERATE_EPS: f64 = 1e-9;

/// Telemetry snapshot of a [`TetherBlend`] — the ρ-drift signature, which is
/// part of the gate rather than a debugging nicety: a ρ that never moves
/// means the mechanism is inert on this stream, and that is a finding, not a
/// pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetherStats {
    /// The applied (published, EMA-smoothed) mixing weight.
    pub rho: f32,
    /// The last CLOSED window's raw OLS fit, before EMA smoothing. `None`
    /// until a window closes.
    pub last_rho_hat: Option<f32>,
    /// Observations folded into the currently OPEN window.
    pub open_observations: u32,
    /// Windows closed so far.
    pub windows_closed: u32,
    /// How many of those were degenerate (`ΣΔ² < ε` — previous ρ held).
    pub degenerate_windows: u32,
}

/// Online closed-form fit of the blend weight ρ (three accumulators + EMA).
///
/// See the module docs for the mechanism, the in-sample guarantee, the lag
/// law this type's shape enforces, and the two hazards. Semantics are
/// validated end-to-end by a real consumer (riir-clippy Bench 042).
#[derive(Debug, Clone, Copy)]
pub struct TetherBlend {
    /// `Σ A_B·Δ` over the open window.
    num: f64,
    /// `Σ Δ²` over the open window.
    den: f64,
    /// Observations in the open window.
    obs: u32,
    /// The applied weight — written ONLY by [`Self::close_window`].
    rho: f32,
    /// Last closed window's raw fit (telemetry).
    last_rho_hat: Option<f32>,
    decay: f32,
    window: u32,
    windows_closed: u32,
    degenerate_windows: u32,
}

impl Default for TetherBlend {
    fn default() -> Self {
        Self::new()
    }
}

impl TetherBlend {
    /// Fresh blend at [`DEFAULT_RHO`] with the default decay + window.
    #[must_use]
    pub fn new() -> Self {
        Self::with_params(DEFAULT_RHO, DEFAULT_EMA_DECAY, DEFAULT_WINDOW)
    }

    /// Fresh blend with explicit parameters.
    ///
    /// `rho0` is clamped to `[0, 1]`, `decay` to `[0, 1]`, and `window` is
    /// floored at 1 — a zero-length window would close on every observation
    /// and fit ρ from a single sample.
    #[must_use]
    pub fn with_params(rho0: f32, decay: f32, window: u32) -> Self {
        Self {
            num: 0.0,
            den: 0.0,
            obs: 0,
            rho: rho0.clamp(0.0, 1.0),
            last_rho_hat: None,
            decay: decay.clamp(0.0, 1.0),
            window: if window == 0 { 1 } else { window },
            windows_closed: 0,
            degenerate_windows: 0,
        }
    }

    /// The applied mixing weight — fit on CLOSED windows only (the lag law).
    #[must_use]
    pub const fn rho(&self) -> f32 {
        self.rho
    }

    /// Blend the two estimator endpoints at the applied ρ:
    /// `(1−ρ)·p1 + ρ·p2`.
    #[must_use]
    pub fn blend(&self, p1: f32, p2: f32) -> f32 {
        (1.0 - self.rho) * p1 + self.rho * p2
    }

    /// Fold one realized outcome into the open window.
    ///
    /// `r` is the realized return, `p1` / `p2` the two endpoint predictions
    /// **as they were at decision time** — passing post-outcome values would
    /// leak the return into its own baseline.
    ///
    /// Returns `true` iff this observation closed the window (and therefore
    /// republished ρ for *subsequent* decisions).
    pub fn observe(&mut self, r: f32, p1: f32, p2: f32) -> bool {
        let a_b = f64::from(r - p1);
        let delta = f64::from(p2 - p1);
        self.num += a_b * delta;
        self.den += delta * delta;
        self.obs += 1;
        // Close AFTER folding: the sample can never influence the ρ that
        // selected it.
        let boundary = self.obs >= self.window;
        if boundary {
            self.close_window();
        }
        boundary
    }

    /// Close the open window: fit ρ̂, EMA it into the applied ρ, reset the
    /// accumulators.
    ///
    /// Called automatically by [`Self::observe`] at the window boundary;
    /// public for consumers that want to flush a short tail (a run ending
    /// mid-window).
    pub fn close_window(&mut self) {
        self.windows_closed += 1;
        if self.den < DEGENERATE_EPS {
            // Degenerate: the endpoints never disagreed, so nothing was
            // learned. Hold the previous ρ — the fixed-blend fallback.
            self.degenerate_windows += 1;
        } else {
            let rho_hat = clip_unit(self.num / self.den);
            self.last_rho_hat = Some(rho_hat);
            self.rho = self.decay * self.rho + (1.0 - self.decay) * rho_hat;
        }
        self.num = 0.0;
        self.den = 0.0;
        self.obs = 0;
    }

    /// The OPEN window's raw fit — **telemetry only**, never applied.
    ///
    /// Exposed so the drift signature can be watched at sub-window
    /// resolution; `None` while the open window is degenerate. Applying this
    /// value would be the same-window bias the type exists to prevent, which
    /// is why no method on this type consumes it.
    #[must_use]
    pub fn open_rho_hat(&self) -> Option<f32> {
        if self.den < DEGENERATE_EPS {
            return None;
        }
        Some(clip_unit(self.num / self.den))
    }

    /// Telemetry snapshot (the ρ-drift signature).
    #[must_use]
    pub const fn stats(&self) -> TetherStats {
        TetherStats {
            rho: self.rho,
            last_rho_hat: self.last_rho_hat,
            open_observations: self.obs,
            windows_closed: self.windows_closed,
            degenerate_windows: self.degenerate_windows,
        }
    }
}

/// Batch OLS fit of ρ over a recorded `(r, p1, p2)` stream — the same closed
/// form [`TetherBlend`] accumulates online, for tests and offline analysis.
///
/// Returns the caller-supplied fallback (`prev`) on a degenerate stream,
/// mirroring the online guard.
#[must_use]
pub fn fit_rho(samples: &[(f32, f32, f32)], prev: f32) -> f32 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for &(r, p1, p2) in samples {
        let a_b = f64::from(r - p1);
        let delta = f64::from(p2 - p1);
        num += a_b * delta;
        den += delta * delta;
    }
    if den < DEGENERATE_EPS {
        return prev;
    }
    clip_unit(num / den)
}

/// `clip[0,1]` of the OLS quotient, narrowed to f32.
///
/// The clip is the paper's, not a safety net: ρ outside `[0, 1]` would be an
/// extrapolation past both endpoints, which the never-worse guarantee does
/// not cover. Truncation is intentional — the accumulators are f64 for
/// numerical stability, the applied weight is an f32 blend coefficient.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn clip_unit(q: f64) -> f32 {
    q.clamp(0.0, 1.0) as f32
}

/// In-sample sum of squared errors of the blend at `rho` over a recorded
/// stream — the quantity the never-worse guarantee is stated in.
#[must_use]
pub fn sse(rho: f32, samples: &[(f32, f32, f32)]) -> f64 {
    samples
        .iter()
        .map(|&(r, p1, p2)| {
            let b = f64::from(1.0 - rho) * f64::from(p1) + f64::from(rho) * f64::from(p2);
            let e = f64::from(r) - b;
            e * e
        })
        .sum()
}

/// One-pass explained-variance accumulator: `EV = 1 − Var(R−V)/Var(R)`.
///
/// The telemetry pair that gates ρ movement in the paper (cold-head guard:
/// do not let ρ chase a value endpoint that explains nothing). Welford-style
/// stable channels for `R` and for the residual `E = R − V` — five running
/// scalars, zero heap, one pass over the stream.
///
/// `ev()` returns `None` while `Var(R) = 0` (constant outcomes — EV is
/// undefined, there is nothing to explain).
#[derive(Debug, Clone, Copy, Default)]
pub struct EvAccumulator {
    n: f64,
    mean_r: f64,
    m2_r: f64,
    mean_e: f64,
    m2_e: f64,
}

impl EvAccumulator {
    /// Fresh accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            n: 0.0,
            mean_r: 0.0,
            m2_r: 0.0,
            mean_e: 0.0,
            m2_e: 0.0,
        }
    }

    /// Fold one realized outcome `r` and the value endpoint's prediction `v`.
    pub fn observe(&mut self, r: f32, v: f32) {
        let e = f64::from(r - v);
        let r = f64::from(r);
        self.n += 1.0;
        // Welford update, channel R.
        let dr = r - self.mean_r;
        self.mean_r += dr / self.n;
        self.m2_r += dr * (r - self.mean_r);
        // Welford update, channel E = R − V.
        let de = e - self.mean_e;
        self.mean_e += de / self.n;
        self.m2_e += de * (e - self.mean_e);
    }

    /// Observations folded so far.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // n is integer-valued by construction
    pub const fn count(&self) -> u64 {
        self.n as u64
    }

    /// Sample variance of the realized outcomes (`None` before 2 samples).
    #[must_use]
    pub fn var_r(&self) -> Option<f64> {
        if self.n < 2.0 {
            return None;
        }
        Some(self.m2_r / (self.n - 1.0))
    }

    /// Sample variance of the residual `R − V` (`None` before 2 samples).
    #[must_use]
    pub fn var_e(&self) -> Option<f64> {
        if self.n < 2.0 {
            return None;
        }
        Some(self.m2_e / (self.n - 1.0))
    }

    /// Explained variance `1 − Var(R−V)/Var(R)` (`None` while undefined:
    /// fewer than 2 samples, or `Var(R) = 0`).
    #[must_use]
    pub fn ev(&self) -> Option<f32> {
        let var_r = self.var_r()?;
        let var_e = self.var_e()?;
        if var_r < f64::EPSILON {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        let v = 1.0 - var_e / var_r;
        Some(v as f32)
    }
}

/// The control-variate gate, batch form:
/// `Var(R−V) < Var(R)` iff `2·Cov(R,V) > Var(V)`.
///
/// Uses the variance identity `2·Cov(R,V) = Var(R) + Var(V) − Var(R−V)` so a
/// single pass computing the three variances two-pass style (f64) decides
/// both sides; the fixture asserts the two comparisons agree — they always
/// do, exactly, because the identity is an equality.
///
/// Returns `None` on a degenerate batch (fewer than 2 samples or zero
/// variance in every channel — no statement is meaningful).
#[must_use]
pub fn control_variate_improves(samples: &[(f32, f32)]) -> Option<bool> {
    let n = samples.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let (mean_r, mean_v, mean_e) = samples
        .iter()
        .fold((0.0f64, 0.0f64, 0.0f64), |(sr, sv, se), &(r, v)| {
            (sr + f64::from(r), sv + f64::from(v), se + f64::from(r - v))
        });
    let (mean_r, mean_v, mean_e) = (mean_r / nf, mean_v / nf, mean_e / nf);
    let mut var_r = 0.0;
    let mut var_v = 0.0;
    let mut var_e = 0.0;
    for &(r, v) in samples {
        let (r, v) = (f64::from(r), f64::from(v));
        let e = r - v;
        var_r += (r - mean_r) * (r - mean_r);
        var_v += (v - mean_v) * (v - mean_v);
        var_e += (e - mean_e) * (e - mean_e);
    }
    let (var_r, var_v, var_e) = (var_r / (nf - 1.0), var_v / (nf - 1.0), var_e / (nf - 1.0));
    if var_r < f64::EPSILON || var_v < f64::EPSILON {
        return None;
    }
    let direct = var_e < var_r;
    #[cfg(debug_assertions)]
    {
        // The identity 2Cov = VarR + VarV − VarE is an equality; the two
        // comparisons must agree (boundary rounding aside).
        let via_cov = var_v < var_r + var_v - var_e;
        assert_eq!(direct, via_cov);
    }
    Some(direct)
}

/// Horizon decay: the per-step retention λ with `λ^L = c` — "retain
/// fraction `c` after horizon `L`".
///
/// ```text
/// λ = c^(1/L)   (one exp/log path, f64 internal)
/// ```
///
/// The paper's near-1 finding: `0.4^(1/8192) ≈ 0.99989` beats λ = 1.0 at the
/// 8k horizon — per-step discounting tuned to the horizon leaves more signal
/// than either no decay or a coarse step discount.
///
/// Contract: `c ∈ (0, 1]`, `horizon ≥ 1` (`0` is treated as `1`).
/// `c = 1` yields λ = 1 for any horizon (no decay); `c = 0` yields λ = 0
/// (retain nothing after any positive horizon — mathematically right, almost
/// certainly not what a caller wants; see HAZARD notes in the module docs
/// for why a silent default would be worse).
#[must_use]
pub fn horizon_decay(c: f32, horizon: u32) -> f32 {
    debug_assert!((0.0..=1.0).contains(&c), "c is a retain fraction in (0, 1]");
    let l = f64::from(horizon.max(1));
    #[allow(clippy::cast_possible_truncation)]
    let lam = ((c as f64).ln() / l).exp();
    lam as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic splitmix64 — every stochastic fixture in this module
    /// draws from a seeded instance so G1 (bit-identical repeats) holds by
    /// construction.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
        }

        /// Uniform in `[-1, 1)`.
        fn next_unit(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // 53-bit mantissa → uniform in [0, 1), remapped to [-1, 1).
            let u01 = (z >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * u01 - 1.0
        }

        /// Bernoulli(p).
        fn bern(&mut self, p: f64) -> bool {
            let u01 = (self.next_unit() + 1.0) * 0.5;
            u01 < p
        }
    }

    // ---- T1: closed-form fit fixtures -------------------------------------

    /// The closed form never loses to a 1001-point grid over [0, 1].
    #[test]
    fn closed_form_matches_grid_argmin() {
        let mut rng = Rng::new(6751);
        let mut samples = Vec::with_capacity(400);
        for _ in 0..400 {
            let u = rng.next_unit();
            let r = (0.55 * u + 0.15 * rng.next_unit()) as f32;
            let p1 = 0.05 * rng.next_unit() as f32;
            let p2 = (0.45 * u + 0.2 * rng.next_unit()) as f32;
            samples.push((r, p1, p2));
        }
        let rho_star = fit_rho(&samples, DEFAULT_RHO);
        let mut grid_best = f64::INFINITY;
        let mut grid_arg = 0.0f32;
        for i in 0..=1000 {
            let rho = i as f32 / 1000.0;
            let s = sse(rho, &samples);
            if s < grid_best {
                grid_best = s;
                grid_arg = rho;
            }
        }
        assert!(
            sse(rho_star, &samples) <= grid_best + 1e-12,
            "closed form rho*={rho_star} SSE {} > grid argmin rho={grid_arg} SSE {grid_best}",
            sse(rho_star, &samples),
        );
    }

    /// The exact in-sample never-worse guarantee, both endpoints, on random
    /// and adversarial streams.
    #[test]
    fn in_sample_never_worse_than_either_endpoint() {
        let streams: Vec<Vec<(f32, f32, f32)>> = vec![
            // Random mixed stream.
            {
                let mut rng = Rng::new(6752);
                (0..250)
                    .map(|_| {
                        (
                            if rng.bern(0.5) { 1.0 } else { 0.0 },
                            0.3,
                            if rng.bern(0.6) { 0.7 } else { 0.2 },
                        )
                    })
                    .collect()
            },
            // Anti-complementary: p2 strictly dominates (ρ* should clip to 1).
            (0..50)
                .map(|i| {
                    let r = 1.0;
                    let p1 = r - 0.2;
                    let p2 = r - 0.05 - (i % 7) as f32 * 0.001;
                    (r, p1, p2)
                })
                .collect(),
            // Complementary symmetric errors (ρ* should be 0.5).
            (0..50)
                .map(|i| {
                    let r = if i % 2 == 0 { 1.0f32 } else { 0.0 };
                    (r, r - 0.1, r + 0.1)
                })
                .collect(),
        ];
        for samples in &streams {
            let rho_star = fit_rho(samples, DEFAULT_RHO);
            let sse_star = sse(rho_star, samples);
            assert!(
                sse_star <= sse(0.0, samples) + 1e-9,
                "rho*={rho_star} loses to p1-only",
            );
            assert!(
                sse_star <= sse(1.0, samples) + 1e-9,
                "rho*={rho_star} loses to p2-only",
            );
        }
    }

    /// Error-regime shapes: complementary → 0.5, anti-complementary → 1.0
    /// (clipped), identical → degenerate guard holds the previous ρ.
    #[test]
    fn error_regimes_shape_the_fit() {
        // Complementary: errors symmetric around the outcome.
        let comp: Vec<(f32, f32, f32)> = (0..64)
            .map(|i| {
                let r = if i % 2 == 0 { 1.0 } else { 0.0 };
                (r, r - 0.1, r + 0.1)
            })
            .collect();
        assert!((fit_rho(&comp, 0.0) - 0.5).abs() < 1e-6);

        // Anti-complementary: both endpoints err the same sign, p2 less so.
        let anti: Vec<(f32, f32, f32)> = (0..64).map(|_| (1.0, 0.8, 0.95)).collect();
        assert_eq!(fit_rho(&anti, 0.0), 1.0);

        // Identical endpoints: degenerate, previous ρ returned.
        let same: Vec<(f32, f32, f32)> = (0..64)
            .map(|i| {
                let p = if i % 3 == 0 { 0.6 } else { 0.2 };
                (if i % 2 == 0 { 1.0 } else { 0.0 }, p, p)
            })
            .collect();
        assert_eq!(fit_rho(&same, 0.7), 0.7);
    }

    /// Identical outcome streams give bit-identical state (G1).
    #[test]
    fn identical_streams_are_bit_identical() {
        let mut rng = Rng::new(6753);
        let stream: Vec<(f32, f32, f32)> = (0..200)
            .map(|_| {
                (
                    if rng.bern(0.5) { 1.0 } else { 0.0 },
                    0.4,
                    (0.5 + 0.2 * rng.next_unit()) as f32,
                )
            })
            .collect();
        let mut a = TetherBlend::with_params(0.3, 0.9, 16);
        let mut b = TetherBlend::with_params(0.3, 0.9, 16);
        for &(r, p1, p2) in &stream {
            a.observe(r, p1, p2);
            b.observe(r, p1, p2);
        }
        assert_eq!(a.stats(), b.stats());
        assert_eq!(a.rho().to_bits(), b.rho().to_bits());
    }

    // ---- T2: lag law + EMA fixtures ---------------------------------------

    /// ρ is frozen inside a window; it moves only at a close, and the
    /// closing sample cannot influence the ρ that selected it.
    #[test]
    fn rho_is_frozen_inside_a_window() {
        let mut t = TetherBlend::with_params(0.0, 0.5, 8);
        // Strongly p2-favouring stream; if any same-window leak existed the
        // blend would start moving mid-window.
        for _ in 0..7 {
            let closed = t.observe(1.0, 0.0, 1.0);
            assert!(!closed, "window must not close before 8 observations");
            assert_eq!(t.rho(), 0.0, "rho frozen at the initial value");
            assert_eq!(t.blend(0.25, 0.75), 0.25, "blend uses the frozen rho");
        }
        assert!(t.open_rho_hat().is_some(), "telemetry sees the open fit");
        let closed = t.observe(1.0, 0.0, 1.0);
        assert!(closed, "8th observation closes the window");
        assert!(t.rho() > 0.0, "rho republished for SUBSEQUENT decisions");
        // And the open window starts over: rho visible to the next selection
        // stays the published one until the next close.
        t.observe(1.0, 0.0, 1.0);
        let published = t.rho();
        assert_eq!(
            t.blend(0.25, 0.75),
            (1.0 - published) * 0.25 + published * 0.75
        );
    }

    /// Known-answer EMA: constant ρ̂ windows converge geometrically, with
    /// exact pinned values (replayed in f32 = bit-identical).
    #[test]
    fn ema_converges_geometrically_on_constant_fits() {
        let d = 0.75f32;
        let mut t = TetherBlend::with_params(0.0, d, 4);
        // p2 perfectly predicts a constant-1 stream → every window's ρ̂ = 1.
        for _ in 0..8 {
            for _ in 0..4 {
                t.observe(1.0, 0.5, 1.0);
            }
        }
        // Replay the identical f32 recurrence.
        let mut expected = 0.0f32;
        for _ in 0..8 {
            expected = d * expected + (1.0 - d) * 1.0;
        }
        assert_eq!(t.rho().to_bits(), expected.to_bits());
        // Geometric bound: after k windows, |ρ_k − 1| ≤ d^k.
        assert!((1.0 - t.rho()) <= d.powi(8) + 1e-7);
    }

    /// A degenerate window holds the previous ρ (never resets to the cold
    /// start) — the fixed-blend fallback.
    #[test]
    fn degenerate_window_holds_the_previous_rho() {
        let mut t = TetherBlend::with_params(0.2, 0.9, 4);
        // One informative window to move ρ.
        for &(r, p1, p2) in &[(1.0f32, 0.0f32, 1.0f32); 4] {
            t.observe(r, p1, p2);
        }
        let moved = t.rho();
        assert!(moved > 0.2);
        // Then a degenerate window: p1 == p2 everywhere.
        for _ in 0..4 {
            t.observe(1.0, 0.5, 0.5);
        }
        assert_eq!(t.rho(), moved, "degenerate window holds rho");
        assert_eq!(t.stats().degenerate_windows, 1);
    }

    // ---- T3: EvAccumulator + control-variate fixtures ---------------------

    /// One-pass Welford matches a two-pass reference on a random stream.
    #[test]
    fn ev_one_pass_matches_two_pass_reference() {
        let mut rng = Rng::new(6754);
        let mut samples = Vec::with_capacity(1000);
        let mut acc = EvAccumulator::new();
        for _ in 0..1000 {
            let u = rng.next_unit();
            let r = (0.5 * u + 0.1 * rng.next_unit()) as f32;
            let v = (0.4 * u + 0.25 * rng.next_unit()) as f32;
            samples.push((r, v));
            acc.observe(r, v);
        }
        // Two-pass reference in f64.
        let n = samples.len() as f64;
        let mean_r = samples.iter().map(|&(r, _)| f64::from(r)).sum::<f64>() / n;
        let mean_e = samples.iter().map(|&(r, v)| f64::from(r - v)).sum::<f64>() / n;
        let var_r = samples
            .iter()
            .map(|&(r, _)| (f64::from(r) - mean_r).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let var_e = samples
            .iter()
            .map(|&(r, v)| (f64::from(r - v) - mean_e).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let acc_var_r = acc.var_r().expect("var_r defined at n=1000");
        let acc_var_e = acc.var_e().expect("var_e defined at n=1000");
        assert!((acc_var_r - var_r).abs() < 1e-12 * var_r);
        assert!((acc_var_e - var_e).abs() < 1e-12 * var_e);
        let ev = acc.ev().expect("ev defined");
        let ref_ev = 1.0 - var_e / var_r;
        assert!(
            (f64::from(ev) - ref_ev).abs() < 1e-6,
            "ev={ev} ref={ref_ev}"
        );
        assert_eq!(acc.count(), 1000);
    }

    /// EV degenerate cases: constant outcomes → undefined, not nonsense.
    #[test]
    fn ev_undefined_on_constant_outcomes() {
        let mut acc = EvAccumulator::new();
        assert!(acc.ev().is_none());
        acc.observe(1.0, 0.5);
        assert!(acc.ev().is_none(), "n < 2");
        let mut flat = EvAccumulator::new();
        for _ in 0..10 {
            flat.observe(1.0, 0.3);
        }
        assert!(flat.ev().is_none(), "Var(R) = 0 → EV undefined");
    }

    /// Control-variate iff: an informative V improves, noise does not, and
    /// the identity-based comparison always agrees with the direct one.
    #[test]
    fn control_variate_gate_decides_correctly() {
        let mut rng = Rng::new(6755);
        // Informative V: r = 0.6·u + noise, v = 0.5·u + less noise.
        let mut informative = Vec::with_capacity(800);
        for _ in 0..800 {
            let u = rng.next_unit();
            informative.push((
                (0.6 * u + 0.1 * rng.next_unit()) as f32,
                (0.5 * u + 0.05 * rng.next_unit()) as f32,
            ));
        }
        assert_eq!(control_variate_improves(&informative), Some(true));
        // Pure noise V.
        let mut noise = Vec::with_capacity(800);
        for _ in 0..800 {
            noise.push((
                (0.6 * rng.next_unit()) as f32,
                (0.6 * rng.next_unit()) as f32,
            ));
        }
        assert_eq!(control_variate_improves(&noise), Some(false));
        // Degenerate batches.
        assert_eq!(control_variate_improves(&[]), None);
        assert_eq!(control_variate_improves(&[(1.0, 0.5)]), None);
    }

    // ---- T4: horizon decay fixtures ---------------------------------------

    /// λ^L == c within 1e-9 (f64 round trip) + the paper's near-1 spot value.
    #[test]
    fn horizon_decay_round_trips() {
        for &(c, l) in &[
            (0.4f32, 8192u32),
            (0.5, 1024),
            (0.9, 128),
            (0.25, 7),
            (0.999, 1_000_000),
        ] {
            let lam_f32 = horizon_decay(c, l);
            // The f64 closed form the f32 LUT value is truncated from.
            let lam_f64 = (f64::from(c).ln() / f64::from(l)).exp();
            let round = lam_f64.powi(l as i32);
            assert!(
                (round - f64::from(c)).abs() < 1e-9,
                "lambda^{l} = {round} != c = {c}"
            );
            assert!((f64::from(lam_f32) - lam_f64).abs() < 1e-6);
        }
        // The paper's finding: 0.4^(1/8192) ≈ 0.99989.
        let lam = horizon_decay(0.4, 8192);
        assert!((lam - 0.9998882).abs() < 1e-5, "lam = {lam}");
        // Edges.
        assert_eq!(horizon_decay(1.0, 4096), 1.0);
        assert_eq!(horizon_decay(0.4, 1), 0.4);
    }

    /// Out-of-sample holdout on a stationary process: the EMA-blended ρ,
    /// published before each holdout window, beats BOTH fixed endpoints on
    /// the holdout aggregate.
    #[test]
    fn out_of_sample_holdout_beats_fixed_endpoints() {
        // Stationary joint distribution: truth u-driven, p1 biased to 0,
        // p2 informative but shrunk + noisy → interior optimum. Each sample
        // is scored under the ρ published BEFORE it was folded (the lag law
        // holds in the fixture too); warmup windows' contributions are
        // discarded so the cold start is not the mechanism under test.
        const WARMUP_WINDOWS: u32 = 8;

        let mut rng = Rng::new(6757);
        let mut t = TetherBlend::with_params(0.0, 0.9, 64);
        let mut sse_blend = 0.0f64;
        let mut sse_p1 = 0.0f64;
        let mut sse_p2 = 0.0f64;
        let mut w = 0u32;
        let mut in_window = 0u32;
        for _ in 0..(64 * 40) {
            let u = rng.next_unit();
            // p2 informative but noisy enough that the optimum is clearly
            // interior (ρ_opt = Cov/Var ≈ 0.6) — the blend must beat BOTH
            // endpoints out of sample, with a margin MC noise cannot flip.
            let r = (0.6 * u + 0.1 * rng.next_unit()) as f32;
            let p1 = 0.0f32;
            let p2 = (0.5 * u + 0.5 * rng.next_unit()) as f32;
            let rho_in_force = t.rho(); // BEFORE folding this sample
            let e = f64::from(r) - f64::from((1.0 - rho_in_force) * p1 + rho_in_force * p2);
            sse_blend += e * e;
            let e1 = f64::from(r - p1);
            sse_p1 += e1 * e1;
            let e2 = f64::from(r - p2);
            sse_p2 += e2 * e2;
            t.observe(r, p1, p2);
            in_window += 1;
            if in_window == 64 {
                if w < WARMUP_WINDOWS {
                    // Discard warmup: the cold start is not under test.
                    sse_blend = 0.0;
                    sse_p1 = 0.0;
                    sse_p2 = 0.0;
                }
                in_window = 0;
                w += 1;
            }
        }
        assert!(
            sse_blend < sse_p1,
            "blend SSE {sse_blend} must beat p1-only {sse_p1}"
        );
        assert!(
            sse_blend < sse_p2,
            "blend SSE {sse_blend} must beat p2-only {sse_p2}"
        );
    }

    /// Regime drift: when the optimal ρ jumps mid-stream, the EMA tracks it
    /// (moves toward the new regime, stays bounded away from the old one).
    #[test]
    fn ema_tracks_regime_drift() {
        let mut t = TetherBlend::with_params(0.0, 0.8, 32);
        // Phase 1: p2 useless (pure noise) → ρ̂ ≈ 0.
        let mut rng = Rng::new(6758);
        for _ in 0..(32 * 6) {
            let r = if rng.bern(0.5) { 1.0f32 } else { 0.0 };
            let p1 = 0.5f32;
            let p2 = 0.5f32 + 0.4 * rng.next_unit() as f32;
            t.observe(r, p1, p2);
        }
        let rho_before = t.rho();
        assert!(
            rho_before < 0.2,
            "noise endpoint → rho stays low ({rho_before})"
        );
        // Phase 2: p2 becomes near-perfect → ρ̂ ≈ 1.
        for _ in 0..(32 * 10) {
            let r = if rng.bern(0.5) { 1.0f32 } else { 0.0 };
            let p2 = if r > 0.5 { 0.95f32 } else { 0.05 };
            t.observe(r, 0.5, p2);
        }
        let rho_after = t.rho();
        assert!(
            rho_after > rho_before + 0.3,
            "EMA must track the drift: {rho_before} → {rho_after}"
        );
        assert!(
            rho_after > 0.6,
            "tracking approaches the new optimum ({rho_after})"
        );
    }

    // ---- T5: admissibility Monte-Carlo fixture pair ------------------------

    /// Mean advantage is preserved under an admissible conditioning signal
    /// (independent sibling outcomes) and measurably biased under a leaking
    /// one (the signal carries the return it claims to predict).
    #[test]
    fn admissibility_preserves_mean_advantage_leak_biases_it() {
        let mut rng = Rng::new(6759);
        let n = 4000;
        // Base stream: r Bernoulli(0.5), a mediocre pair of endpoints.
        let mut rs = Vec::with_capacity(n);
        for _ in 0..n {
            rs.push(if rng.bern(0.5) { 1.0f32 } else { 0.0 });
        }
        let adv = |r: f32| r - 0.5; // advantage vs a constant 0.5 baseline
        let full_mean = rs.iter().map(|&r| adv(r)).sum::<f32>() / n as f32;

        // Admissible z: independent fair coin — the z=1 subpopulation is an
        // unbiased sample of the full one.
        let mut admissible_mean = 0.0f32;
        let mut z1 = 0u32;
        for &r in &rs {
            if rng.bern(0.5) {
                admissible_mean += adv(r);
                z1 += 1;
            }
        }
        admissible_mean /= z1 as f32;
        assert!(
            (admissible_mean - full_mean).abs() < 0.05,
            "admissible z: |{admissible_mean} − {full_mean}| must be MC noise"
        );

        // Leaking z: z fires with prob 0.9 when r = 1, 0.3 when r = 0 — the
        // conditioned mean is shifted by construction.
        let mut leaking_mean = 0.0f32;
        let mut l1 = 0u32;
        for &r in &rs {
            let p = if r > 0.5 { 0.9 } else { 0.3 };
            if rng.bern(p) {
                leaking_mean += adv(r);
                l1 += 1;
            }
        }
        leaking_mean /= l1 as f32;
        assert!(
            (leaking_mean - full_mean).abs() > 0.1,
            "leaking z must measurably bias the mean ({leaking_mean} vs {full_mean})"
        );
    }
}

#[cfg(all(test, debug_assertions))]
mod alloc_gate {
    use super::*;

    /// G4: 0 allocations across 10 000 observations, window closes, EMA
    /// publishes, telemetry snapshots, EV reads, and decay lookups — the
    /// whole primitive surface, steady state.
    #[test]
    fn steady_state_is_alloc_free() {
        let mut t = TetherBlend::with_params(0.2, 0.95, 64);
        let mut ev = EvAccumulator::new();
        for i in 0..64 {
            let r = if i % 3 == 0 { 1.0 } else { 0.0 };
            t.observe(r, 0.4, 0.6);
            ev.observe(r, 0.5);
        }
        crate::alloc::reset_alloc_stats();
        for i in 0..10_000 {
            let r = if i % 3 == 0 { 1.0 } else { 0.0 };
            t.observe(r, 0.4, 0.6);
            ev.observe(r, 0.5);
            let b = t.blend(0.4, 0.6);
            let open = t.open_rho_hat();
            let lam = horizon_decay(0.4, 1024);
            let e = ev.ev();
            std::hint::black_box((b, open, lam, e));
        }
        t.close_window();
        std::hint::black_box(t.stats());
        let (count, bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(
            (count, bytes),
            (0, 0),
            "tether must be alloc-free in steady state"
        );
    }
}
