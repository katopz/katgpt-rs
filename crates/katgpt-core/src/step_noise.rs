//! `step_noise` — modelless thrash monitor + DDR gate (Issue 975).
//!
//! Source: riir-train Research 485 (arXiv:2603.06009, "Preventing Learning
//! Stagnation in PPO by Scaling to 1 Million Parallel Environments"). The
//! PPO wrapper is incidental; the extractable content is a **step-vs-noise
//! control law** for any anchored, evidence-driven update process: plateaus
//! and thrash arise when the update step is large relative to evidence
//! noise; the signature is HIGH update magnitude + FLAT objective (not slow
//! learning). All closed-form arithmetic over observable scalars — zero
//! gradients, zero training.
//!
//! # The five primitives
//!
//! 1. [`EwmaCom`] — center-of-mass EWMA. `α = 1/(1+N)` gives the exact
//!    closed-form lag law `lag = (1−α)/α = N` steps (unit-tested against
//!    the numeric kernel mean).
//! 2. [`half_split_dispersion`] — `1 − cosine` between the two half-window
//!    update directions. A labeled noise PROXY (gradient-disagreement
//!    proxy), never claimed as a gradient norm.
//! 3. [`StepNoiseMonitor::observe`] — 4 streaming EMAs + regime
//!    classification. O(N)/tick, fixed `[f32; N]` arrays, ZERO allocation.
//! 4. [`Regime`] — five regimes with disjoint typed remedies
//!    ([`Regime::remedy`]).
//! 5. [`ddr`] + [`min_ddr_at`] — evidence-per-divergence with the
//!    zero-divergence INFINITY law, and the sigmoid DDR-floor ramp (the
//!    paper's rising-optimum law).
//!
//! # Regime classification — sigmoid everywhere, softmax banned
//!
//! Three sigmoid gates over the streamed scalars (house law: sigmoid ONLY,
//! never softmax — no cross-class normalization exists here):
//!
//! - `progress   = σ(β_p·(slope − p0))` — belief the objective rises
//! - `churn_hot  = σ(β_c·(churn − c0))` — belief the updates are large
//! - `regressing = σ(β_w·(−slope − w0))` — belief the objective falls
//!
//! Thrash is the literal composition `σ_churn·(1 − σ_progress)` (high
//! magnitude × flat objective); Stuck is its MIRROR
//! `(1 − σ_churn)·(1 − σ_progress)` — the missing high/low-norm pair
//! around `guided_width::StagnationGate`'s low-norm stuck drive (the
//! Issue-975 substrate table: step_noise's Thrash arm is the mirror the
//! gate lacks; nothing replaces the gate).
//!
//! Precedence is fixed and documented: a sustained NEGATIVE slope is
//! [`Regime::CommittedWrong`] even at high churn (revert, don't damp);
//! thrash beats stuck (both share a flat objective — the magnitude axis
//! decides); the quiet floor falls to [`Regime::Converged`].
//!
//! # Stuck ↔ Converged discriminator (T1 convention)
//!
//! At monitor granularity Stuck and Converged share observables (quiet +
//! flat). T1's convention: cumulative travel `Σ‖Δstate‖` above
//! `StepNoiseConfig::travel_tol` reads Converged (the run WENT somewhere
//! and settled); at-or-below reads Stuck (it never left the anchor). T2's
//! fixture bench measures the real boundary (the drift archetype is the
//! measured Stuck-or-Learning case; only the Thrash exclusion is
//! load-bearing).
//!
//! # Interop (consumed, never forked)
//!
//! [`regime_from_stagnation_counter`] (compiled when
//! `guided_width_rollouts` is also on) derives a Regime FROM a consumer's
//! existing `guided_width::StagnationGate` counter: where the gate's OWN
//! `sigma(w_stuck)` drive sits at/above half of its sanitised ceiling, the
//! gate's counter asserts Stuck and WINS (the gate is the authority on its
//! own axis); otherwise the monitor's verdict stands. The
//! progressive_mcgs binary no-improvement counters are untouched — they
//! cannot separate Stuck from Thrash (both read "no improvement"); that
//! separation is precisely this module's graded EWMA slope.
//!
//! # NaN / INF contract
//!
//! - [`half_split_dispersion`]: zero or non-finite input norms → exactly
//!   `0.0` (no evidence of disagreement; never poisons the monitor's EMAs).
//! - [`ddr`]: `divergence ≤ 0` (and NaN — no usable divergence) →
//!   `+INFINITY`, the paper's evidence-per-nothing law. `+inf` divergence
//!   reads `0.0` (finite evidence per infinite divergence).
//! - [`min_ddr_at`]: any non-finite schedule input → `0.0` (a broken
//!   schedule must never RAISE a floor); `w ≤ 0` degenerates to a step at
//!   `mid`.
//! - The monitor's EMAs are structurally finite given finite inputs; a NaN
//!   state/objective poisons its own EMAs (the saddle_escape caller-bug
//!   precedent — documented, not silently repaired). NaN-poisoned reads
//!   classify as [`Regime::Converged`] (every gate comparison reads
//!   false), so the no-op remedy fires — a poisoned monitor never
//!   actuates damp/explore/revert.
//!
//! # Latent vs raw
//!
//! Every scalar here is per-process local (think-brain side). Nothing
//! crosses a sync boundary; a consumer that exports anything exports the
//! [`Regime`] verdict (a u8), never a latent vector (Issue 975 Notes).
//!
//! # Scope
//!
//! ISSUE-975 STATUS (2026-10-31): the PoC is COMPLETE — G1–G4 all PASS
//! (six-archetype fixture classification on DEFAULT calibration; both
//! causal toggle arms incl. the DualPoolBandit anti-circularity arm;
//! 28.3 ns/observe at N=8 vs the ≤100 ns bar; zero steady-state alloc).
//! T1 landed at `349e07ee3`, T2–T4 at `48fd0be4d`; the issue file itself is
//! removed per noise-reduction (katgpt-rs HISTORY.md Issue 975 carries the
//! record). Consumer follow-ups: riir-ai Issue 1143 (engine bridge third
//! verdict row), riir-refine Issue 190 (elo_k_eff + readout), riir-dao
//! Issue 009 (advisory evidence_scaled_budget). Stays opt-in
//! (`step_noise`) — consumer promotion rides each filed issue's own gates.
//!
//! NOTE: `Regime` is deliberately NOT re-exported at the crate root —
//! the mean-field regime classifier's `Regime`
//! (`feature = "mean_field_regime"`) already owns that root name; use
//! `katgpt_core::step_noise::Regime` (the `saddle_escape::GateDecision`
//! precedent). Everything else re-exports at the root when the feature
//! is on.

use crate::simd::fast_sigmoid;

// ─────────────────────────────────────────────────────────────────────
// EMA core (one implementation; both the const-generic sugar and the
// monitor's configured-COM EMAs ride it)
// ─────────────────────────────────────────────────────────────────────

/// Streaming EMA state — `value += α·(x − value)`. Fixed size, zero
/// allocation, no warmup logic (callers own warmup policy).
#[derive(Clone, Copy, Debug)]
struct EmaState {
    value: f32,
    alpha: f32,
    ticks: u32,
}

impl EmaState {
    /// New EMA with the given smoothing factor `α ∈ (0, 1]`.
    #[inline]
    fn new(alpha: f32) -> Self {
        Self {
            value: 0.0,
            alpha,
            ticks: 0,
        }
    }

    /// Feed one observation, return the post-observation value.
    #[inline]
    fn observe(&mut self, x: f32) -> f32 {
        self.ticks = self.ticks.saturating_add(1);
        self.value += self.alpha * (x - self.value);
        self.value
    }
}

// ─────────────────────────────────────────────────────────────────────
// Primitive 1 — center-of-mass EWMA
// ─────────────────────────────────────────────────────────────────────

/// Center-of-mass EWMA: `α = 1/(1 + COM)` with `COM = N`, so the
/// closed-form lag law is exact:
///
/// ```text
/// lag = (1 − α)/α = N steps
/// ```
///
/// (the mean age of the input under the filter weights `w_k = α(1−α)^k`
/// is `Σ k·w_k = (1−α)/α` — unit-tested against the numeric kernel).
/// Fixed size, zero allocation, O(1) per observation.
///
/// `N = 0` is the identity tracker (`α = 1`, the value IS the last
/// observation, lag 0).
#[derive(Clone, Copy, Debug)]
pub struct EwmaCom<const N: usize> {
    ema: EmaState,
}

impl<const N: usize> EwmaCom<N> {
    /// New tracker at the type-level center of mass `N`.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            ema: EmaState::new(1.0 / (N as f32 + 1.0)),
        }
    }

    /// The smoothing factor `α = 1/(1+N)`.
    #[inline]
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.ema.alpha
    }

    /// Closed-form lag `(1−α)/α` — equals `N` exactly for this
    /// parameterization.
    #[inline]
    #[must_use]
    pub fn lag_steps(&self) -> f32 {
        (1.0 - self.ema.alpha) / self.ema.alpha
    }

    /// Feed one observation, return the post-observation value.
    ///
    /// NaN/inf input poisons the tracker (the saddle_escape caller-bug
    /// precedent — the caller's contract, not silently repaired here).
    #[inline]
    pub fn observe(&mut self, x: f32) -> f32 {
        self.ema.observe(x)
    }

    /// Current smoothed value.
    #[inline]
    #[must_use]
    pub fn value(&self) -> f32 {
        self.ema.value
    }

    /// Observations seen (telemetry).
    #[inline]
    #[must_use]
    pub fn ticks(&self) -> u32 {
        self.ema.ticks
    }

    /// Reset the value/ticks; keeps `α`.
    #[inline]
    pub fn reset(&mut self) {
        self.ema = EmaState::new(self.ema.alpha);
    }
}

impl<const N: usize> Default for EwmaCom<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Primitive 2 — the labeled noise proxy
// ─────────────────────────────────────────────────────────────────────

/// `1 − cosine(a, b)` between the two half-window update directions —
/// the DISPERSION of recent update direction.
///
/// ⚠ **This is a noise PROXY**, the monitor's stand-in for
/// gradient-disagreement. It is never claimed as a gradient norm; the
/// paper's signal is gradient noise, and this is the pre-commit observable
/// cheap enough to stream per tick (Issue 975: "labeled, not claimed").
///
/// Guards (all return exactly `0.0` — no evidence of disagreement, and a
/// proxy must never poison the monitor's EMAs):
/// - a zero direction on either side (undefined cosine),
/// - any non-finite input or intermediate (NaN / ±inf norms).
///
/// The cosine is clamped to `[-1, 1]`, so the result is always in
/// `[0, 2]`: `0.0` = perfectly aligned, `1.0` = orthogonal, `2.0` =
/// exactly opposed. Zero allocation, O(N).
#[inline]
#[must_use]
pub fn half_split_dispersion(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 || !denom.is_finite() || !dot.is_finite() {
        return 0.0;
    }
    let cos = (dot / denom).clamp(-1.0, 1.0);
    1.0 - cos
}

// ─────────────────────────────────────────────────────────────────────
// Primitive 5 — DDR: evidence per divergence + the sigmoid floor ramp
// ─────────────────────────────────────────────────────────────────────

/// Evidence per unit divergence from the anchor: `n_evidence / divergence`.
///
/// The paper's control knob (Research 485 §4): evidence-per-divergence is
/// INFINITE when divergence is zero — `divergence ≤ 0` returns
/// `+INFINITY`. NaN divergence folds into the same arm (no usable
/// divergence); `+inf` divergence reads `0.0` (finite evidence per
/// infinite divergence). Zero allocation, O(1).
#[inline]
#[must_use]
pub fn ddr(n_evidence: u32, divergence: f32) -> f32 {
    if divergence.is_nan() || divergence <= 0.0 {
        return f32::INFINITY;
    }
    n_evidence as f32 / divergence
}

/// Sigmoid ramp of the DDR floor with the cumulative evidence budget
/// (the paper's rising-optimum law):
///
/// ```text
/// min_ddr_at(f) = lo + (hi − lo)·σ((f − mid)/w)
/// ```
///
/// Monotone non-decreasing for `hi ≥ lo`; `mid` is the inflection (the
/// floor reads `lo + (hi−lo)/2` there), `w` the sigmoid width. The
/// endpoints are consumer-calibrated (Issue 975 primitive 5).
///
/// Guards:
/// - any non-finite input → `0.0` (a broken schedule must never RAISE a
///   floor — the disarmed floor),
/// - `w ≤ 0` (or non-finite) degenerates to a STEP at `mid`:
///   `f ≥ mid → hi`, else `lo`.
///
/// Zero allocation, O(1).
#[inline]
#[must_use]
pub fn min_ddr_at(budget_frac: f32, lo: f32, hi: f32, mid: f32, w: f32) -> f32 {
    if !(lo.is_finite() && hi.is_finite() && mid.is_finite() && budget_frac.is_finite()) {
        return 0.0;
    }
    if !(w.is_finite() && w > 0.0) {
        // Step degenerate: exact threshold at the inflection.
        return if budget_frac >= mid { hi } else { lo };
    }
    let ramp = fast_sigmoid((budget_frac - mid) / w);
    lo + (hi - lo) * ramp
}

// ─────────────────────────────────────────────────────────────────────
// Primitive 4 — Regime + typed remedies
// ─────────────────────────────────────────────────────────────────────

/// The five-regime verdict of a [`StepNoiseMonitor`].
///
/// Disjoint remedies (typed in [`Remedy`] — a `match` over [`Regime`]
/// cannot forget one):
///
/// | Regime | Signature | Remedy |
/// |---|---|---|
/// | [`Learning`](Regime::Learning) | objective rising | no-op |
/// | [`Thrash`](Regime::Thrash) | high churn × flat objective | damp + gather |
/// | [`Stuck`](Regime::Stuck) | low churn × flat objective, never traveled | explore / replan |
/// | [`Converged`](Regime::Converged) | quiet after travel (or all gates quiet) | no-op |
/// | [`CommittedWrong`](Regime::CommittedWrong) | objective falling steadily | revert |
///
/// Thrash and Stuck are the magnitude-axis mirror pair around
/// `guided_width::StagnationGate`'s low-norm drive; `Stuck` FEEDS that
/// gate (interop: [`regime_from_stagnation_counter`]) — nothing replaces
/// it. [`Converged`] and [`Learning`] are both no-op remedies, so a cold
/// monitor (insufficient evidence) is labeled [`Learning`] — the least-
/// wrong no-op label; [`StepNoiseMonitor::is_warm`] distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Regime {
    /// Objective rising — keep going.
    Learning = 0,
    /// High update magnitude, flat objective — the anti-thrash gate's
    /// whole point. Remedy: DAMP the step size and GATHER evidence.
    Thrash = 1,
    /// Low update magnitude, flat objective, still near the anchor —
    /// the classic stuck arm. Remedy: EXPLORE / replan (this is the arm
    /// `guided_width::StagnationGate` escalates on).
    Stuck = 2,
    /// Quiet after travel — settled. No-op.
    Converged = 3,
    /// Objective falling steadily — the process is optimizing into a
    /// worse region. Remedy: REVERT.
    CommittedWrong = 4,
}

/// The remedy vocabulary behind [`Regime::remedy`] — disjoint by
/// construction, so consumers act on exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Remedy {
    /// Do nothing (Learning / Converged).
    NoOp,
    /// Suppress injection / shrink the effective step until evidence
    /// accumulates (Thrash).
    DampAndGather,
    /// Escalate exploration or re-plan through the host FSM (Stuck).
    ExploreReplan,
    /// Undo the recent committed direction (CommittedWrong).
    Revert,
}

impl Regime {
    /// The disjoint remedy for this regime.
    #[inline]
    #[must_use]
    pub fn remedy(self) -> Remedy {
        match self {
            Regime::Learning | Regime::Converged => Remedy::NoOp,
            Regime::Thrash => Remedy::DampAndGather,
            Regime::Stuck => Remedy::ExploreReplan,
            Regime::CommittedWrong => Remedy::Revert,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Primitive 3 — the monitor (config, read, state machine)
// ─────────────────────────────────────────────────────────────────────

/// Calibration of the regime classifier and the monitor's smoothing.
///
/// ⚠ The midpoints (`p0`, `c0`, `w0`) live in the CALLER's units —
/// objective delta per tick and state-norm movement per tick. The
/// defaults assume roughly unit-scale states and objectives (the
/// `StagnationGate::DEFAULT` posture: size to the host's scale). Slopes
/// (`β_*`) are sigmoid steepnesses; `τ_*` are firing floors in `[0, 1]`.
/// These are T1 starting points — the T2 fixture bench measures the real
/// boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepNoiseConfig {
    /// Progress midpoint: slope above this reads as rising (objective
    /// units / tick).
    pub p0: f32,
    /// Progress sigmoid steepness.
    pub beta_p: f32,
    /// Churn midpoint: movement above this reads as large (state-norm
    /// units / tick).
    pub c0: f32,
    /// Churn sigmoid steepness.
    pub beta_c: f32,
    /// Regressing midpoint on `−slope`: sustained fall beyond this fires
    /// CommittedWrong (objective units / tick).
    pub w0: f32,
    /// Regressing sigmoid steepness.
    pub beta_w: f32,
    /// CommittedWrong firing floor on the regressing gate.
    pub tau_wrong: f32,
    /// Thrash firing floor on `σ_churn·(1 − σ_progress)`.
    pub tau_thrash: f32,
    /// Stuck firing floor on `(1 − σ_churn)·(1 − σ_progress)`.
    pub tau_stuck: f32,
    /// Learning firing floor on `σ_progress`.
    pub tau_learn: f32,
    /// Cumulative-travel tolerance separating Stuck (≤) from Converged
    /// (>) — state-norm units, summed over the run.
    pub travel_tol: f32,
    /// Center of mass of the monitor's four EMAs (ticks). `α = 1/(1+com)`.
    pub ema_com: f32,
    /// Ticks before the classifier is trusted (a fresh monitor reads
    /// `Regime::Learning` — the no-op label — until then).
    pub warmup: u32,
    /// Half-window `H` of the dispersion accumulator: the two half-window
    /// update directions are summed over `H` ticks each (super-window
    /// `2H`), then compared by [`half_split_dispersion`]. Clamped ≥ 1.
    pub half_window: u16,
}

impl StepNoiseConfig {
    /// Calibration starting point (unit-scale posture — retune `p0`/`c0`/
    /// `w0`/`travel_tol` to the host's scales):
    /// `p0 1e-3, β_p 200, c0 1e-2, β_c 100, w0 1e-3, β_w 200,
    /// τ_wrong 0.60, τ_thrash 0.30, τ_stuck 0.30, τ_learn 0.70,
    /// travel_tol 1e-2, ema_com 8, warmup 16, half_window 4`.
    pub const DEFAULT: Self = Self {
        p0: 1e-3,
        beta_p: 200.0,
        c0: 1e-2,
        beta_c: 100.0,
        w0: 1e-3,
        beta_w: 200.0,
        tau_wrong: 0.60,
        tau_thrash: 0.30,
        tau_stuck: 0.30,
        tau_learn: 0.70,
        travel_tol: 1e-2,
        ema_com: 8.0,
        warmup: 16,
        half_window: 4,
    };
}

impl Default for StepNoiseConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One `observe()` readout: the four streamed scalars + the regime.
///
/// - `churn` — EWMA of `‖Δstate‖` per tick (the update magnitude).
/// - `progress_slope` — EWMA of `Δobjective` per tick (flat ≈ 0).
/// - `dispersion` — EWMA of the half-split cosine dispersion (the
///   labeled noise PROXY; `0.0` until the first super-window completes).
/// - `ddr` — evidence per unit divergence (`ddr(evidence, divergence
///   EMA)`); `+INFINITY` while the process hugs its anchor — the paper's
///   law, not a bug.
/// - `regime` — the classified [`Regime`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepNoiseRead {
    /// EWMA of the per-tick state movement norm.
    pub churn: f32,
    /// EWMA of the per-tick objective delta.
    pub progress_slope: f32,
    /// EWMA of the half-window direction dispersion (noise proxy).
    pub dispersion: f32,
    /// Evidence per unit divergence from the anchor (INF-guarded).
    pub ddr: f32,
    /// The classified regime.
    pub regime: Regime,
}

/// Live thrash/stuck/learning monitor over an `N`-dim anchored update
/// process.
///
/// Feeds `observe(&state, objective)` once per tick; internally maintains
/// FOUR streaming EMAs (churn, objective slope, dispersion, anchor
/// divergence) + the half-window direction accumulators + cumulative
/// travel — every one a fixed `[f32; N]` array or scalar, ZERO heap
/// allocation, O(N) per tick. The classification is pure sigmoid
/// arithmetic (never softmax).
///
/// Cold behavior: until `cfg.warmup` ticks the regime is
/// [`Regime::Learning`] (the no-op label — see [`Regime`] docs) and
/// [`StepNoiseMonitor::is_warm`] reports `false`.
///
/// NaN state/objective poisons the owning EMAs (caller's bug, documented
/// at the module level); a poisoned read classifies
/// [`Regime::Converged`] — the no-op remedy — never damp/explore/revert.
#[derive(Clone, Copy, Debug)]
pub struct StepNoiseMonitor<const N: usize> {
    cfg: StepNoiseConfig,
    half: u16,
    churn: EmaState,
    slope: EmaState,
    disp: EmaState,
    divergence: EmaState,
    prev: [f32; N],
    anchor: [f32; N],
    acc_a: [f32; N],
    acc_b: [f32; N],
    travel: f32,
    phase: u32,
    have_prev: bool,
    have_obj: bool,
    prev_obj: f32,
    evidence: u32,
}

impl<const N: usize> StepNoiseMonitor<N> {
    /// New monitor over a config (the first `observe` sets the anchor).
    #[must_use]
    pub fn new(cfg: StepNoiseConfig) -> Self {
        let com = if cfg.ema_com.is_finite() && cfg.ema_com >= 0.0 {
            cfg.ema_com
        } else {
            0.0
        };
        let alpha = 1.0 / (com + 1.0);
        Self {
            cfg,
            half: cfg.half_window.max(1),
            churn: EmaState::new(alpha),
            slope: EmaState::new(alpha),
            disp: EmaState::new(alpha),
            divergence: EmaState::new(alpha),
            prev: [0.0; N],
            anchor: [0.0; N],
            acc_a: [0.0; N],
            acc_b: [0.0; N],
            travel: 0.0,
            phase: 0,
            have_prev: false,
            have_obj: false,
            prev_obj: 0.0,
            evidence: 0,
        }
    }

    /// Feed one tick: the post-update `state`, the post-update
    /// `objective`. Returns the read (see [`StepNoiseRead`]). O(N), zero
    /// allocation.
    pub fn observe(&mut self, state: &[f32; N], objective: f32) -> StepNoiseRead {
        // Per-tick state delta + its norm (first tick: no delta).
        let mut delta = [0.0f32; N];
        let mut norm2 = 0.0f32;
        if self.have_prev {
            for (d, (s, p)) in delta.iter_mut().zip(state.iter().zip(self.prev.iter())) {
                *d = s - p;
                norm2 += *d * *d;
            }
        }
        let movement = norm2.sqrt();
        self.churn.observe(movement);
        self.travel += movement;

        // Divergence from the anchor (the first tick re-anchors).
        if !self.have_prev {
            self.anchor = *state;
        }
        let mut div2 = 0.0f32;
        for (s, a) in state.iter().zip(self.anchor.iter()) {
            let v = s - a;
            div2 += v * v;
        }
        self.divergence.observe(div2.sqrt());

        // Half-window direction accumulation; at the super-window
        // boundary, disperse the two half-sums and rotate.
        if self.phase < self.half as u32 {
            for (t, d) in self.acc_a.iter_mut().zip(delta.iter()) {
                *t += *d;
            }
        } else {
            for (t, d) in self.acc_b.iter_mut().zip(delta.iter()) {
                *t += *d;
            }
        }
        self.phase += 1;
        let super_window = 2 * self.half as u32;
        if self.phase >= super_window {
            let raw = half_split_dispersion(&self.acc_a, &self.acc_b);
            self.disp.observe(raw);
            for (a, b) in self.acc_a.iter_mut().zip(self.acc_b.iter()) {
                *a = *b;
            }
            for b in self.acc_b.iter_mut() {
                *b = 0.0;
            }
            self.phase = 0;
        }

        // Objective slope (first tick: no delta).
        if self.have_obj {
            self.slope.observe(objective - self.prev_obj);
        }
        self.prev_obj = objective;
        self.have_obj = true;

        // Rotate state, count evidence.
        self.prev = *state;
        self.have_prev = true;
        self.evidence = self.evidence.saturating_add(1);

        let regime = if self.is_warm() {
            classify_regime(self.churn.value, self.slope.value, self.travel, &self.cfg)
        } else {
            // Cold: insufficient evidence — the documented no-op label.
            Regime::Learning
        };

        StepNoiseRead {
            churn: self.churn.value,
            progress_slope: self.slope.value,
            dispersion: self.disp.value,
            ddr: ddr(self.evidence, self.divergence.value),
            regime,
        }
    }

    /// `true` once `cfg.warmup` ticks of evidence exist (the regime is
    /// trustworthy).
    #[inline]
    #[must_use]
    pub fn is_warm(&self) -> bool {
        self.evidence >= self.cfg.warmup
    }

    /// Observations consumed (the DDR evidence count).
    #[inline]
    #[must_use]
    pub fn evidence(&self) -> u32 {
        self.evidence
    }

    /// Cumulative travel `Σ‖Δstate‖` (the Stuck↔Converged discriminator
    /// input).
    #[inline]
    #[must_use]
    pub fn travel(&self) -> f32 {
        self.travel
    }

    /// The active config.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &StepNoiseConfig {
        &self.cfg
    }

    /// Full reset (the next `observe` re-anchors at the new state).
    pub fn reset(&mut self) {
        let alpha = self.churn.alpha;
        self.churn = EmaState::new(alpha);
        self.slope = EmaState::new(alpha);
        self.disp = EmaState::new(alpha);
        self.divergence = EmaState::new(alpha);
        self.prev = [0.0; N];
        self.anchor = [0.0; N];
        self.acc_a = [0.0; N];
        self.acc_b = [0.0; N];
        self.travel = 0.0;
        self.phase = 0;
        self.have_prev = false;
        self.have_obj = false;
        self.prev_obj = 0.0;
        self.evidence = 0;
    }
}

/// The regime classifier — pure sigmoid arithmetic over the streamed
/// scalars (module docs carry the full precedence rationale).
///
/// Precedence: CommittedWrong (sustained fall — even at high churn,
/// revert beats damp) → Thrash (`σ_churn·(1 − σ_progress)`, the issue's
/// literal composition) → Stuck/Converged (the mirror; travel decides)
/// → Learning → the quiet floor Converged.
///
/// NaN-poisoned scalars read `Converged` (every comparison false → the
/// final arm): the no-op remedy, never an actuated one.
#[inline]
#[must_use]
fn classify_regime(churn: f32, progress_slope: f32, travel: f32, cfg: &StepNoiseConfig) -> Regime {
    let progress = fast_sigmoid(cfg.beta_p * (progress_slope - cfg.p0));
    let churn_hot = fast_sigmoid(cfg.beta_c * (churn - cfg.c0));
    let regressing = fast_sigmoid(cfg.beta_w * (-progress_slope - cfg.w0));

    if regressing >= cfg.tau_wrong {
        return Regime::CommittedWrong;
    }
    if churn_hot * (1.0 - progress) >= cfg.tau_thrash {
        return Regime::Thrash;
    }
    if (1.0 - churn_hot) * (1.0 - progress) >= cfg.tau_stuck {
        return if travel <= cfg.travel_tol {
            Regime::Stuck
        } else {
            Regime::Converged
        };
    }
    if progress >= cfg.tau_learn {
        return Regime::Learning;
    }
    Regime::Converged
}

// ─────────────────────────────────────────────────────────────────────
// Interop — Regime::Stuck derivable FROM the guided_width counter
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "guided_width_rollouts")]
use crate::guided_width::StagnationGate;

/// Derive a [`Regime`] for a consumer that ALREADY tracks the
/// `guided_width::StagnationGate` counter state (`w_stuck`: consecutive
/// steps with update norm below the gate's `stuck_tol`).
///
/// The mapping never re-counts and never re-derives the gate's drive: it
/// CONSUMES the shipped `sigma(w_stuck)` — at/above half of the gate's
/// sanitised ceiling (its own midpoint), the counter asserts Stuck and
/// WINS; a zeroed gate (`sigma_max = 0`, the kill switch) asserts nothing
/// and the monitor's verdict stands. Below the midpoint the counter is
/// silent — it cannot separate Stuck from Thrash (both are "no
/// improvement" to a counter) — so the monitor's graded verdict stands.
///
/// This is interop, not replacement: [`Regime::Stuck`] feeds the gate
/// where a consumer tracks no counter, and the counter feeds the regime
/// where it does. Compiled only with `guided_width_rollouts` also on.
#[cfg(feature = "guided_width_rollouts")]
#[must_use]
pub fn regime_from_stagnation_counter(
    w_stuck: u16,
    gate: &StagnationGate,
    read: StepNoiseRead,
) -> Regime {
    let smax = gate.sigma_max_sanitised();
    if smax > 0.0 && gate.sigma(w_stuck) >= 0.5 * smax {
        return Regime::Stuck;
    }
    read.regime
}

// ─────────────────────────────────────────────────────────────────────
// Tests — lag law, DDR known answers, gate quadrants
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EwmaCom: the lag law ─────────────────────────────────────────

    #[test]
    fn lag_law_closed_form_equals_com() {
        assert!((EwmaCom::<1>::new().lag_steps() - 1.0).abs() < 1e-5);
        assert!((EwmaCom::<4>::new().lag_steps() - 4.0).abs() < 1e-5);
        assert!((EwmaCom::<8>::new().lag_steps() - 8.0).abs() < 1e-5);
        assert!((EwmaCom::<32>::new().lag_steps() - 32.0).abs() < 1e-4);
    }

    #[test]
    fn lag_law_kernel_mean_matches_closed_form() {
        // Filter weights w_k = α(1−α)^k; mean input age = Σ k·w_k =
        // (1−α)/α = COM. Feed a unit impulse, accumulate the response
        // mean in f64; a 1e-2 bound absorbs f32 kernel drift.
        const N: usize = 8;
        const STEPS: u32 = 4000;
        let mut e = EwmaCom::<N>::new();
        let mut mass = 0.0f64;
        let mut moment = 0.0f64;
        for k in 0..STEPS {
            let y = if k == 0 {
                e.observe(1.0)
            } else {
                e.observe(0.0)
            } as f64;
            mass += y;
            moment += k as f64 * y;
        }
        let mean_lag = moment / mass;
        assert!(
            (mean_lag - N as f64).abs() < 1e-2,
            "kernel mean lag {mean_lag} vs closed form {N}"
        );
    }

    #[test]
    fn lag_law_step_target_crosses_half_at_the_analytic_point() {
        // Constant 1.0 target from zero state: y_t = 1 − (1−α)^t, so the
        // 0.5 crossing lands at t_half = ln(0.5)/ln(1−α) (≈ 5.884 for
        // COM 8). Assert the observed crossing is within one step.
        const N: usize = 8;
        let mut e = EwmaCom::<N>::new();
        let alpha = 1.0f32 / (N as f32 + 1.0);
        let t_half = (0.5f64.ln() / (1.0 - f64::from(alpha)).ln()) as f32;
        let mut crossing: Option<u32> = None;
        for k in 0..64 {
            if e.observe(1.0) >= 0.5 {
                crossing = Some(k);
                break;
            }
        }
        let c = crossing.expect("step response must cross 0.5") as f32;
        assert!(
            (c - t_half).abs() <= 1.0,
            "crossing at {c}, analytic t_half {t_half}"
        );
    }

    #[test]
    fn zero_com_is_the_identity_tracker() {
        let mut e = EwmaCom::<0>::new();
        assert_eq!(e.lag_steps(), 0.0);
        assert_eq!(e.alpha(), 1.0);
        assert_eq!(e.observe(0.25), 0.25);
        assert_eq!(e.observe(-0.75), -0.75);
    }

    // ── half_split_dispersion: proxy known answers ───────────────────

    #[test]
    fn dispersion_known_answers() {
        let a = [1.0, 0.0, 0.0];
        assert_eq!(half_split_dispersion(&a, &a), 0.0); // aligned
        assert_eq!(half_split_dispersion(&a, &[-1.0, 0.0, 0.0]), 2.0); // opposed
        assert!((half_split_dispersion(&a, &[0.0, 2.0, 0.0]) - 1.0).abs() < 1e-6); // orthogonal
        assert_eq!(half_split_dispersion(&a, &[0.0; 3]), 0.0); // zero direction
        assert_eq!(half_split_dispersion(&[f32::NAN; 3], &a), 0.0); // NaN
        assert_eq!(half_split_dispersion(&[f32::INFINITY; 3], &a), 0.0); // inf
        assert_eq!(half_split_dispersion(&a, &[f32::NAN, 0.0, 0.0]), 0.0);
    }

    // ── ddr: the paper's law, known answers ──────────────────────────

    #[test]
    fn ddr_known_answer_vectors() {
        assert_eq!(ddr(100, 0.5), 200.0); // hand-computed
        assert_eq!(ddr(100, 0.0), f32::INFINITY); // the zero-divergence law
        assert_eq!(ddr(7, -1.0), f32::INFINITY); // negative divergence: same arm
        assert_eq!(ddr(100, f32::NAN), f32::INFINITY); // no usable divergence
        assert_eq!(ddr(100, f32::INFINITY), 0.0); // finite evidence / infinite divergence
        assert_eq!(ddr(0, 2.0), 0.0); // no evidence yet
        assert_eq!(ddr(1, 1.0), 1.0);
    }

    // ── min_ddr_at: the sigmoid ramp ─────────────────────────────────

    #[test]
    fn min_ddr_at_midpoint_and_saturated_endpoints() {
        let (lo, hi, mid, w) = (10.0f32, 100.0f32, 0.5f32, 0.1f32);
        // At the inflection the ramp is exactly half.
        let at_mid = min_ddr_at(mid, lo, hi, mid, w);
        assert!((at_mid - 55.0).abs() < 1e-5, "{at_mid} vs 55");
        // 41w past either side saturates fast_sigmoid to exactly lo/hi.
        assert_eq!(min_ddr_at(mid + 41.0 * w, lo, hi, mid, w), hi);
        assert_eq!(min_ddr_at(mid - 41.0 * w, lo, hi, mid, w), lo);
    }

    #[test]
    fn min_ddr_at_is_monotone() {
        let (lo, hi, mid, w) = (10.0f32, 100.0f32, 0.5f32, 0.1f32);
        let mut prev = -f32::INFINITY;
        for step in 0..=200 {
            let f = mid - 2.0 * w + 4.0 * w * (step as f32 / 200.0);
            let v = min_ddr_at(f, lo, hi, mid, w);
            assert!(v >= prev, "ramp must be non-decreasing: {v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn min_ddr_at_guards() {
        let (lo, hi, mid, w) = (10.0f32, 100.0f32, 0.5f32, 0.1f32);
        // Broken schedule → the disarmed floor, never a raised one.
        assert_eq!(min_ddr_at(f32::NAN, lo, hi, mid, w), 0.0);
        assert_eq!(min_ddr_at(0.5, f32::NAN, hi, mid, w), 0.0);
        assert_eq!(min_ddr_at(0.5, lo, f32::INFINITY, mid, w), 0.0);
        // w ≤ 0 degenerates to the exact step.
        assert_eq!(min_ddr_at(0.49, lo, hi, mid, 0.0), lo);
        assert_eq!(min_ddr_at(0.5, lo, hi, mid, 0.0), hi);
        assert_eq!(min_ddr_at(0.51, lo, hi, mid, -1.0), hi);
    }

    // ── Regime remedies ──────────────────────────────────────────────

    #[test]
    fn remedies_are_disjoint_and_typed() {
        assert_eq!(Regime::Learning.remedy(), Remedy::NoOp);
        assert_eq!(Regime::Thrash.remedy(), Remedy::DampAndGather);
        assert_eq!(Regime::Stuck.remedy(), Remedy::ExploreReplan);
        assert_eq!(Regime::Converged.remedy(), Remedy::NoOp);
        assert_eq!(Regime::CommittedWrong.remedy(), Remedy::Revert);
    }

    // ── Monitor fixtures ─────────────────────────────────────────────

    /// Deterministic thrash delta: pseudo-random magnitude/sign per
    /// (tick, coord) — non-periodic in the aggregate (the oscillation
    /// archetype is T2's fixture; this one exercises the magnitude axis).
    fn thrash_delta(t: u32, i: usize) -> f32 {
        let t = t as usize;
        let v = ((t * 7 + i * 3 + t * t) % 7) as f32 / 3.5 - 1.0;
        0.5 * v
    }

    /// Drive `ticks` observations with per-tick deterministic deltas and
    /// a linear objective; returns the LAST read. The leading observe
    /// establishes the baseline.
    fn drive<const N: usize>(
        m: &mut StepNoiseMonitor<N>,
        state: &mut [f32; N],
        objective: &mut f32,
        d_obj: f32,
        ticks: u32,
        delta_at: impl Fn(u32) -> [f32; N],
    ) -> StepNoiseRead {
        let mut read = m.observe(state, *objective);
        for t in 0..ticks {
            let d = delta_at(t);
            for (s, d) in state.iter_mut().zip(d.iter()) {
                *s += *d;
            }
            *objective += d_obj;
            read = m.observe(state, *objective);
        }
        read
    }

    #[test]
    fn monitor_thrash_high_churn_flat_objective() {
        const N: usize = 4;
        let mut m = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        let mut state = [0.0f32; N];
        let mut obj = 1.0f32;
        let read = drive(&mut m, &mut state, &mut obj, 0.0, 40, |t| {
            core::array::from_fn(|i| thrash_delta(t, i))
        });
        assert!(m.is_warm());
        assert_eq!(read.regime, Regime::Thrash, "read: {read:?}");
        assert!(read.churn > 0.5, "churn should read hot: {}", read.churn);
        assert!(read.progress_slope.abs() < 1e-6, "flat objective");
        assert!(read.dispersion.is_finite() && read.dispersion >= 0.0);
        assert_eq!(read.regime.remedy(), Remedy::DampAndGather);
    }

    #[test]
    fn monitor_learning_progress_beats_the_flat_arms() {
        const N: usize = 4;
        let mut m = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        let mut state = [0.0f32; N];
        let mut obj = 0.0f32;
        let read = drive(&mut m, &mut state, &mut obj, 0.01, 40, |_| [0.05; N]);
        assert!(m.is_warm());
        assert_eq!(read.regime, Regime::Learning, "read: {read:?}");
        assert!(read.progress_slope > 0.005);
        assert_eq!(read.regime.remedy(), Remedy::NoOp);
    }

    #[test]
    fn monitor_stuck_low_churn_near_the_anchor() {
        const N: usize = 4;
        let mut m = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        let mut state = [0.0f32; N];
        let mut obj = 1.0f32;
        let read = drive(&mut m, &mut state, &mut obj, 0.0, 40, |_| [1e-5; N]);
        assert!(m.is_warm());
        assert_eq!(read.regime, Regime::Stuck, "read: {read:?}");
        assert!(read.churn < 1e-3);
        assert!(m.travel() <= m.config().travel_tol);
        assert_eq!(read.regime.remedy(), Remedy::ExploreReplan);
    }

    #[test]
    fn monitor_converged_quiet_after_travel() {
        const N: usize = 4;
        let mut m = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        let mut state = [0.0f32; N];
        let mut obj = 1.0f32;
        // Phase 1: real movement (travel accumulates well past the tol).
        drive(&mut m, &mut state, &mut obj, 0.0, 20, |t| {
            core::array::from_fn(|i| {
                if (t + i as u32).is_multiple_of(2) {
                    0.5
                } else {
                    -0.5
                }
            })
        });
        // Phase 2: freeze — the churn EMA decays, the regime settles.
        let read = drive(&mut m, &mut state, &mut obj, 0.0, 50, |_| [0.0; N]);
        assert!(m.travel() > m.config().travel_tol);
        assert_eq!(read.regime, Regime::Converged, "read: {read:?}");
        assert_eq!(read.regime.remedy(), Remedy::NoOp);
    }

    #[test]
    fn monitor_committed_wrong_sustained_fall_wins_over_churn() {
        const N: usize = 4;
        let mut m = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        let mut state = [0.0f32; N];
        let mut obj = 1.0f32;
        // Steady steps AND a sustained fall: precedence must pick Revert.
        let read = drive(&mut m, &mut state, &mut obj, -0.01, 40, |_| [0.005; N]);
        assert!(m.is_warm());
        assert_eq!(read.regime, Regime::CommittedWrong, "read: {read:?}");
        assert!(read.progress_slope < -0.005);
        assert_eq!(read.regime.remedy(), Remedy::Revert);
    }

    #[test]
    fn monitor_cold_reads_the_noop_label_and_ddr_is_infinite_at_birth() {
        const N: usize = 4;
        let mut m = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        assert!(!m.is_warm());
        let mut state = [0.3, -0.2, 0.1, 0.5];
        let read = m.observe(&state, 1.0);
        assert_eq!(read.regime, Regime::Learning, "cold = the no-op label");
        assert_eq!(read.ddr, f32::INFINITY, "hugging the anchor: the law");
        assert_eq!(read.churn, 0.0);
        assert_eq!(read.progress_slope, 0.0);
        assert_eq!(read.dispersion, 0.0);
        // Same state again: still zero divergence → still infinite DDR.
        let read = m.observe(&state, 1.0);
        assert_eq!(read.ddr, f32::INFINITY);
        // Movement away from the anchor: DDR turns finite.
        for s in state.iter_mut() {
            *s += 0.5;
        }
        let read = m.observe(&state, 1.0);
        assert!(read.ddr.is_finite(), "diverged anchor → finite DDR");
        assert!(read.ddr > 0.0);
    }

    #[test]
    fn monitor_fixed_shape_is_stack_embeddable() {
        // The G4 zero-alloc canary is T4's lane; the T1 claim is the
        // SHAPE: no heap anywhere, every array fixed — so the monitor
        // stays small enough to embed by value.
        assert!(size_of::<StepNoiseMonitor<8>>() < 1024);
        assert!(size_of::<StepNoiseRead>() <= 5 * size_of::<f32>() + 8);
    }

    #[test]
    fn monitor_reset_reanchors() {
        const N: usize = 4;
        let mut m = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        let mut state = [1.0f32; N];
        let mut obj = 1.0f32;
        drive(&mut m, &mut state, &mut obj, 0.01, 30, |_| [0.1; N]);
        assert!(m.is_warm());
        m.reset();
        assert_eq!(m.evidence(), 0);
        assert!(!m.is_warm());
        assert_eq!(m.travel(), 0.0);
        let read = m.observe(&state, obj);
        assert_eq!(read.regime, Regime::Learning, "cold again after reset");
    }

    // ── Interop: derivable FROM the guided_width counter ─────────────

    #[test]
    #[cfg(feature = "guided_width_rollouts")]
    fn regime_is_derivable_from_the_gate_counter_state() {
        use crate::guided_width::StagnationGate;
        let gate = StagnationGate::DEFAULT; // σ_max 0.25, α 2, w₀ 2
        let thrash_read = StepNoiseRead {
            churn: 1.0,
            progress_slope: 0.0,
            dispersion: 0.5,
            ddr: 1.0,
            regime: Regime::Thrash,
        };
        // At the gate's own midpoint (w = w₀) the counter asserts stuck
        // and WINS over the monitor's thrash verdict.
        assert_eq!(
            regime_from_stagnation_counter(2, &gate, thrash_read),
            Regime::Stuck
        );
        // Above it, likewise.
        assert_eq!(
            regime_from_stagnation_counter(5, &gate, thrash_read),
            Regime::Stuck
        );
        // Below the midpoint the counter is silent; the monitor's graded
        // verdict (which CAN separate Thrash from Stuck) stands.
        assert_eq!(
            regime_from_stagnation_counter(1, &gate, thrash_read),
            Regime::Thrash
        );
        // A zeroed gate is disarmed — it asserts nothing.
        let dead = StagnationGate {
            sigma_max: 0.0,
            ..StagnationGate::DEFAULT
        };
        assert_eq!(
            regime_from_stagnation_counter(9, &dead, thrash_read),
            Regime::Thrash
        );
        // The silent-counter path passes the monitor verdict through
        // unchanged for every regime.
        for regime in [
            Regime::Learning,
            Regime::Thrash,
            Regime::Stuck,
            Regime::Converged,
            Regime::CommittedWrong,
        ] {
            let read = StepNoiseRead {
                churn: 0.0,
                progress_slope: 0.0,
                dispersion: 0.0,
                ddr: 1.0,
                regime,
            };
            assert_eq!(regime_from_stagnation_counter(0, &gate, read), regime);
        }
    }
}
