//! Risk-controlled dual-threshold compute exit (Plan 575, Research 494 —
//! "Conformal Thinking: Risk Control for Reasoning on a Compute Budget",
//! Wang et al., arXiv:2602.03814, ICML 2026).
//!
//! The distilled modelless question this module answers: *when may a compute
//! loop stop thinking?* Every shipped halt mechanism in this stack answers it
//! with a hand-set constant (Plan 304's τ=1.0, FPRM's 0.1, DEQ's 0.05, MCTS's
//! fixed 512) — a threshold is a budget hyperparameter wearing a different
//! unit, and nothing certifies the error it buys. This primitive converts the
//! question into an **interpretable risk budget with a finite-sample
//! guarantee**: an upper threshold (stop-when-confident, risks false
//! positives), a parametric lower threshold — a squeezed-sigmoid confidence
//! schedule `λ−(t) = σ(c(ωt − sB), l, u)` that stops hopeless instances early
//! (stop-when-not-progressing) — and an offline UCB/Hoeffding calibrator
//! (`Risk̂ + √(ln(1/δ)/2n) ≤ ε`) that turns a labeled validation set into
//! thresholds certified at the caller's ε, δ. Modelless throughout: two
//! comparisons + one house [`crate::sigmoid`] on the hot path, distribution-
//! free statistics at calibration time.
//!
//! # Composition
//!
//! - [`DualExitPolicy`] — the runtime decision (T1.1/T1.2): `exit()` is 2
//!   comparisons + 1 squeezed sigmoid, zero-alloc, `#[inline]`.
//! - Losses (T1.3): [`fp_loss`] (Eq. 8), [`farsighted_loss`] (Eq. 9),
//!   [`regret_loss`] (Eq. 10), [`past_wrongness`] (Eq. 11) — all bounded
//!   [0,1], pure fns over labeled trajectories.
//! - [`calibrate_into`] (T1.4/T1.5): the UCB calibrator + two-step decoupled
//!   selection (λ+ at ε+, then `{c,s,l}` at ε− conditioned), efficiency-loss
//!   argmin among feasible pairs, monotonicity verification with loud refusal.
//! - [`PiGePcMonitor`] (T1.6): the App. C distribution-shift tripwire —
//!   disarms the lower threshold when it filters more correct than incorrect
//!   instances (`p_i < p_c`); upper-only mode remains guaranteed.
//!
//! # Honesty notes (inherited from the paper's own caveats)
//!
//! - **Monotonicity of risk in (λ+, c) is assumed by the paper, not
//!   proven.** This module verifies it empirically per calibration and
//!   REFUSES the violating span (grid points past the first violation are
//!   excluded from selection; the violation is reported). A flat curve
//!   (risk identically 0 or constant) is trivially monotone and passes.
//! - **Two-step decoupling trades rigor for practice** (λ+ selected before
//!   the schedule, no joint certificate) — recorded as accepted, per the plan.
//! - **Infeasible grids fall back conservatively, they do not silently
//!   certify.** When no grid point satisfies `Risk̂ + ucb ≤ ε`, the returned
//!   policy is the most conservative grid point and `fell_back` is set: the
//!   guarantee could not be established at this (ε, δ, n); the caller is
//!   told, not protected by accident.
//! - The Hoeffding term is the [0,1]-bounded-loss specialization
//!   `√(ln(1/δ)/(2n))`, kept local (a one-liner) rather than depending on
//!   the `hint_regret` feature — that module's [`hoeffding_half_width`
//!   cousin](crate::hint_regret::hoeffding_half_width) carries a
//!   `ReturnBounds` range parameter this module does not need.
//!
//! Zero-allocation steady state: `exit()` is pure arithmetic;
//! `calibrate_into` reuses caller-owned [`CalibrateScratch`] (per-grid-point
//! curves, `resize`-grown once, `clear`-reused thereafter).
//!
//! risk_control_exit — modelless dual-threshold compute-exit (Plan 575,
//! Research 494, "Conformal Thinking" arXiv:2602.03814): stop-when-confident
//! upper threshold + parametric stop-when-not-progressing lower schedule,
//! offline UCB/Hoeffding calibration with guaranteed realized exit-risk,
//! efficiency-loss selection among feasible pairs, App. C disarm tripwire.
//! Opt-in pending the Bench 681 GOAT.

use crate::sigmoid;

// ──────────────────────────────────────────────────────────────────────────
// T1.1 — the runtime decision
// ──────────────────────────────────────────────────────────────────────────

/// Per-tick exit decision (the plan's three-variant contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitVerdict {
    /// Neither threshold fired — keep computing.
    Continue = 0,
    /// Upper exit: `s̃t ≥ λ+` — confident success, commit the answer now.
    Commit = 1,
    /// Lower exit: `s̃t ≤ λ−(t)` — confident failure, abandon early.
    Abandon = 2,
}

/// Terminal state of a policy run over a whole trajectory (the trace-level
/// companion to [`ExitVerdict`]: a trajectory that never fires either
/// threshold runs to budget exhaustion and commits its final answer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TerminalVerdict {
    /// Exited via the upper threshold at `tick`.
    Commit = 0,
    /// Exited via the lower threshold at `tick`.
    Abandon = 1,
    /// Never fired — consumed the full budget; the answer is the final step.
    Exhausted = 2,
}

/// Squeezed sigmoid `σ(z, l, u) = (u−l)·σ(z) + l` — the paper's Eq. 7
/// schedule kernel, built on the house sigmoid (never softmax).
#[inline]
pub fn squeezed_sigmoid(z: f32, l: f32, u: f32) -> f32 {
    (u - l) * sigmoid(z) + l
}

/// The dual-threshold exit policy (paper §3).
///
/// Fields: `lambda_plus` = upper (stop-when-confident) threshold; `{c, s, l,
/// u}` parameterize the lower schedule `λ−(t) = σ(c(ωt − s·B), l, u)` where
/// `ωt` = compute consumed at step t and `B` = total budget. The instance
/// must raise confidence **on a schedule** to earn the right to keep
/// reasoning: `s̃t ≤ λ−(t)` means not progressing enough → abandon.
///
/// Invariant (asserted at construction, mutual exclusivity by construction):
/// `0 ≤ l ≤ u < lambda_plus`, `c ≥ 0`. Because `λ−(t) < u < λ+` for every
/// t, the two exits can never fire on the same tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualExitPolicy {
    /// Upper threshold `λ+` — stop when `s̃t ≥ λ+`.
    pub lambda_plus: f32,
    /// Schedule steepness `c` (rate per budget unit; `c → 0` = constant).
    pub c: f32,
    /// Schedule center `s` (the schedule's midpoint sits at `ωt = s·B`).
    pub s: f32,
    /// Schedule floor `l` — the lowest value `λ−` can approach.
    pub l: f32,
    /// Schedule ceiling `u` — the highest value `λ−` can approach
    /// (must stay strictly below `lambda_plus`).
    pub u: f32,
}

impl DualExitPolicy {
    /// Constructs the policy, asserting the mutual-exclusivity invariant
    /// (`0 ≤ l ≤ u < λ+`, `c ≥ 0`, all finite). A violation is a caller
    /// wiring bug — fail loud at construction, not silently at exit time.
    #[inline]
    pub const fn new(lambda_plus: f32, c: f32, s: f32, l: f32, u: f32) -> Self {
        // const-friendly checks would need const_float_ops; the debug_assert
        // form covers the intended dev-time catching.
        debug_assert!(
            l >= 0.0 && u >= l && lambda_plus > u,
            "mutual exclusivity: 0 <= l <= u < lambda_plus"
        );
        debug_assert!(c >= 0.0, "schedule steepness c must be >= 0");
        Self {
            lambda_plus,
            c,
            s,
            l,
            u,
        }
    }

    /// The lower (stop-when-not-progressing) threshold at progress `ωt`
    /// out of budget `B`: `λ−(t) = σ(c(ωt − s·B), l, u)` (paper Eq. 7).
    #[inline]
    pub fn lambda_minus(&self, omega_t: u32, budget: u32) -> f32 {
        let z = self.c * (omega_t as f32 - self.s * budget as f32);
        squeezed_sigmoid(z, self.l, self.u)
    }

    /// The exit decision at one tick — 2 comparisons + 1 squeezed sigmoid,
    /// zero-alloc. Check order is upper-first (the confident exit is the
    /// common case and short-circuits the schedule evaluation).
    #[inline]
    pub fn exit(&self, s_tilde: f32, omega_t: u32, budget: u32) -> ExitVerdict {
        if s_tilde >= self.lambda_plus {
            ExitVerdict::Commit
        } else if s_tilde <= self.lambda_minus(omega_t, budget) {
            ExitVerdict::Abandon
        } else {
            ExitVerdict::Continue
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T1.2 — schedule shape presets (paper Eq. 12–13)
// ──────────────────────────────────────────────────────────────────────────

impl DualExitPolicy {
    /// **Linear** shape (paper Eq. 12): `s = 0.5`, `c·B ≪ 1` — the schedule
    /// stays near its midpoint with a gentle ramp across the horizon.
    /// `u = λ+/2` keeps the mutual-exclusivity invariant by construction.
    #[inline]
    pub fn linear(lambda_plus: f32, budget: u32) -> Self {
        Self::new(
            lambda_plus,
            0.1 / budget.max(1) as f32,
            0.5,
            0.0,
            0.5 * lambda_plus,
        )
    }

    /// **Exponential** shape: `s > 1` puts the sigmoid center beyond the
    /// horizon — within `[0, B]` only the convex rising limb is visible.
    #[inline]
    pub fn exponential(lambda_plus: f32, budget: u32) -> Self {
        Self::new(
            lambda_plus,
            8.0 / budget.max(1) as f32,
            1.5,
            0.0,
            0.5 * lambda_plus,
        )
    }

    /// **Log** shape: `s < 0` puts the sigmoid center before the horizon —
    /// within `[0, B]` only the concave saturating limb is visible.
    #[inline]
    pub fn log(lambda_plus: f32, budget: u32) -> Self {
        Self::new(
            lambda_plus,
            8.0 / budget.max(1) as f32,
            -0.5,
            0.0,
            0.5 * lambda_plus,
        )
    }

    /// **Constant** shape: `c → 0` collapses the schedule to its midpoint
    /// `(u−l)/2 + l` for every t. No budget dependence.
    #[inline]
    pub fn constant(lambda_plus: f32) -> Self {
        Self::new(lambda_plus, 0.0, 0.5, 0.0, 0.5 * lambda_plus)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T1.3 — losses (paper Eq. 8–11), all bounded [0, 1]
// ──────────────────────────────────────────────────────────────────────────

/// A policy's terminal event over one trajectory: what happened, and when.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExitTrace {
    /// What terminated the trajectory.
    pub verdict: TerminalVerdict,
    /// 0-indexed step at which the trajectory ended (always `< T`; an
    /// [`TerminalVerdict::Exhausted`] trace carries `tick = T − 1`).
    pub tick: usize,
}

/// Runs a policy over one confidence trajectory: the first tick at which
/// either threshold fires wins (the paper's `τ = min{t : s̃t ≥ λ+ ∨ s̃t ≤
/// λ−(t)}`); if neither fires the trace is `Exhausted` at `T − 1`.
///
/// Progress mapping: step t consumes `ωt = t + 1` of `budget = T` (the
/// schedule consumes normalized progress; consumers whose token/step counts
/// differ call [`DualExitPolicy::exit`] directly with their own ω).
#[inline]
pub fn run_policy(policy: &DualExitPolicy, s_tilde: &[f32]) -> ExitTrace {
    let t_len = s_tilde.len();
    debug_assert!(t_len >= 1, "trajectory must have at least one step");
    let budget = t_len.max(1) as u32;
    for (t, &s) in s_tilde.iter().enumerate() {
        match policy.exit(s, (t + 1) as u32, budget) {
            ExitVerdict::Commit => {
                return ExitTrace {
                    verdict: TerminalVerdict::Commit,
                    tick: t,
                };
            }
            ExitVerdict::Abandon => {
                return ExitTrace {
                    verdict: TerminalVerdict::Abandon,
                    tick: t,
                };
            }
            ExitVerdict::Continue => {}
        }
    }
    ExitTrace {
        verdict: TerminalVerdict::Exhausted,
        tick: t_len - 1,
    }
}

/// **Eq. 8 — false-positive loss of the upper threshold**, per instance:
/// `I[exited via λ+] · I[f_τ ≠ y*]` ∈ {0, 1}. Lower exits and budget
/// exhaustion contribute 0 (exhaustion is not a threshold failure; the
/// lower exit is charged by [`farsighted_loss`]).
#[inline]
pub fn fp_loss(correct: &[bool], trace: ExitTrace) -> f32 {
    debug_assert!(trace.tick < correct.len(), "tick out of range");
    f32::from(trace.verdict == TerminalVerdict::Commit && !correct[trace.tick])
}

/// **Eq. 9 — farsighted false-negative loss of the lower threshold**, per
/// instance: `I[exited via λ−] · (Σ_{k≥τ} I[f_k = y*]) / (T − τ)` ∈ [0, 1].
///
/// Farsighted = the sum checks ALL future solutions, so abandoning an
/// instance that would have been solved at any later step costs more than
/// abandoning a hopeless one (0-indexed ticks; the denominator counts the
/// steps `τ..T`, matching the paper's `T − t + 1` under 1-indexing).
#[inline]
pub fn farsighted_loss(correct: &[bool], trace: ExitTrace) -> f32 {
    debug_assert!(trace.tick < correct.len(), "tick out of range");
    if trace.verdict != TerminalVerdict::Abandon {
        return 0.0;
    }
    let t_len = correct.len();
    let future = correct[trace.tick..].iter().filter(|&&c| c).count() as f32;
    let denom = (t_len - trace.tick).max(1) as f32;
    future / denom
}

/// First tick at which the answer was correct (`t'` in Eq. 10), if any.
#[inline]
fn first_correct(correct: &[bool]) -> Option<usize> {
    correct.iter().position(|&c| c)
}

/// **Eq. 10 — normalized regret (efficiency loss of committing too late)**:
/// `J+ = max(0, τ − t') / T` — compute wasted after the first correct
/// answer. 0 when the answer was never correct within the horizon (nothing
/// was wasted — there was nothing to waste).
#[inline]
pub fn regret_loss(correct: &[bool], trace: ExitTrace) -> f32 {
    let t_len = correct.len();
    debug_assert!(trace.tick < t_len, "tick out of range");
    match first_correct(correct) {
        Some(tp) if trace.tick > tp => (trace.tick - tp) as f32 / t_len.max(1) as f32,
        _ => 0.0,
    }
}

/// **Eq. 11 — past wrongness (efficiency loss of exiting too early)**:
/// `J− = Σ_{k ≤ τ} I[f_k ≠ y*] / T` — the wrong-answer density up to the
/// exit tick.
#[inline]
pub fn past_wrongness(correct: &[bool], trace: ExitTrace) -> f32 {
    let t_len = correct.len();
    debug_assert!(trace.tick < t_len, "tick out of range");
    let wrong = correct[..=trace.tick].iter().filter(|&&c| !c).count() as f32;
    wrong / t_len.max(1) as f32
}

// ──────────────────────────────────────────────────────────────────────────
// T1.4 — the UCB calibrator
// ──────────────────────────────────────────────────────────────────────────

/// Hoeffding UCB half-width for the mean of `n` i.i.d. [0,1]-bounded losses
/// at confidence `1 − δ`: `√(ln(1/δ)/(2n))`. Local one-liner — the
/// [0,1] specialization of the `hint_regret::hoeffding_half_width` neighbor
/// (which carries a `ReturnBounds` range this module does not need); kept
/// local so this module never depends on the `hint_regret` feature.
#[inline]
pub fn ucb_half_width(n: u32, delta: f32) -> f32 {
    debug_assert!(delta > 0.0 && delta < 1.0, "delta must be in (0,1)");
    if n == 0 {
        return f32::MAX;
    }
    ((1.0f64 / delta as f64).ln() / (2.0 * n as f64)).sqrt() as f32
}

/// Expected monotonicity direction of the empirical risk curve vs the
/// hyperparameter being swept (the paper ASSUMES monotonicity; this module
/// verifies it empirically and refuses the violating span).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MonotoneDir {
    /// Risk weakly increases with the hyperparameter.
    Increasing = 0,
    /// Risk weakly decreases with the hyperparameter (e.g. FP risk vs λ+).
    Decreasing = 1,
}

/// The first monotonicity violation found in an empirical risk curve:
/// `index` is the LAST VALID grid position (the span `> index` is refused).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonotoneViolation {
    /// Grid index after which the curve can no longer be trusted.
    pub index: usize,
    /// Risk at `index`.
    pub prev: f32,
    /// Risk at `index + 1` (the violating step).
    pub next: f32,
}

/// Verifies an empirical risk curve is weakly monotone in `dir` within
/// `tol` (tolerance absorbs binomial sampling noise — the recommended scale
/// is half the UCB correction). Returns the FIRST violation, if any.
pub fn verify_monotone(values: &[f32], dir: MonotoneDir, tol: f32) -> Option<MonotoneViolation> {
    for i in 0..values.len().saturating_sub(1) {
        let (prev, next) = (values[i], values[i + 1]);
        let violates = match dir {
            MonotoneDir::Increasing => next < prev - tol,
            MonotoneDir::Decreasing => next > prev + tol,
        };
        if violates {
            return Some(MonotoneViolation {
                index: i,
                prev,
                next,
            });
        }
    }
    None
}

/// One labeled compute trajectory for calibration: per-step confidence and
/// per-step correctness (`I[f_k = y*]`). The budget is the slice length.
#[derive(Debug, Clone, Copy)]
pub struct TrajectorySample<'a> {
    /// Confidence trajectory `s̃` (one value per step).
    pub s_tilde: &'a [f32],
    /// Per-step correctness indicator `I[f_k = y*]`.
    pub correct: &'a [bool],
}

impl<'a> TrajectorySample<'a> {
    /// Constructs a sample, asserting slice-length agreement (a mismatch is
    /// a caller wiring bug).
    #[inline]
    pub fn new(s_tilde: &'a [f32], correct: &'a [bool]) -> Self {
        debug_assert_eq!(
            s_tilde.len(),
            correct.len(),
            "s_tilde/correct length mismatch"
        );
        Self { s_tilde, correct }
    }

    /// Horizon `T` (number of steps).
    #[inline]
    pub fn horizon(&self) -> usize {
        self.s_tilde.len()
    }
}

/// First tick at which `s̃ ≥ λ+` (upper-only crossing — the Eq. 8 exit).
#[inline]
fn upper_exit_tick(s_tilde: &[f32], lambda_plus: f32) -> Option<usize> {
    s_tilde.iter().position(|&s| s >= lambda_plus)
}

/// Empirical FP risk (Eq. 8 mean) of an upper-only policy at `lambda_plus`
/// over the sample set — the step-1 calibration quantity.
pub fn empirical_upper_risk(samples: &[TrajectorySample<'_>], lambda_plus: f32) -> f32 {
    if samples.is_empty() {
        return f32::MAX;
    }
    let mut loss = 0.0f64;
    for smp in samples {
        // Never crossed λ+ — not a threshold failure (contributes 0).
        if let Some(t) = upper_exit_tick(smp.s_tilde, lambda_plus) {
            loss += f64::from(!smp.correct[t]);
        }
    }
    (loss / samples.len() as f64) as f32
}

/// Empirical farsighted FN risk (Eq. 9 mean) of the full dual policy over
/// the sample set — the step-2 calibration quantity (conditioned on the
/// already-selected λ+).
pub fn empirical_lower_risk(samples: &[TrajectorySample<'_>], policy: &DualExitPolicy) -> f32 {
    if samples.is_empty() {
        return f32::MAX;
    }
    let mut loss = 0.0f64;
    for smp in samples {
        let trace = run_policy(policy, smp.s_tilde);
        loss += farsighted_loss(smp.correct, trace) as f64;
    }
    (loss / samples.len() as f64) as f32
}

/// Mean normalized compute `mean((τ+1)/T)` consumed by the policy across
/// the sample set — the efficiency objective for selection among feasible
/// candidates (lower = cheaper; the paper picks the most efficient
/// feasible signal/threshold pair).
pub fn mean_normalized_compute(samples: &[TrajectorySample<'_>], policy: &DualExitPolicy) -> f32 {
    if samples.is_empty() {
        return f32::MAX;
    }
    let mut acc = 0.0f64;
    for smp in samples {
        let trace = run_policy(policy, smp.s_tilde);
        acc += (trace.tick + 1) as f64 / smp.horizon().max(1) as f64;
    }
    (acc / samples.len() as f64) as f32
}

/// Lower-schedule grid entry `{c, s, l, u}`. Caller contract: the grid is
/// sorted by `c` ascending (the FN-risk-vs-c monotonicity check sweeps it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScheduleParams {
    /// Schedule steepness.
    pub c: f32,
    /// Schedule center.
    pub s: f32,
    /// Schedule floor.
    pub l: f32,
    /// Schedule ceiling.
    pub u: f32,
}

impl ScheduleParams {
    /// Freezes the params into a full policy at the given upper threshold.
    #[inline]
    pub fn into_policy(self, lambda_plus: f32) -> DualExitPolicy {
        DualExitPolicy::new(lambda_plus, self.c, self.s, self.l, self.u)
    }
}

/// Calibration knobs (paper §4): the two risk budgets, the confidence
/// level, and the grid-correction / monotonicity-tolerance switches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrateConfig {
    /// Upper-threshold risk budget `ε+` (FP risk).
    pub epsilon_plus: f32,
    /// Lower-schedule risk budget `ε−` (farsighted FN risk).
    pub epsilon_minus: f32,
    /// Confidence level `δ` (per-grid-point failure probability).
    pub delta: f32,
    /// Union-bound the grid: use `δ / (|G+| + |G−|)` instead of `δ`
    /// (the multiple-comparison variant for wide grids).
    pub delta_over_grid: bool,
    /// Monotonicity tolerance as a fraction of the UCB correction —
    /// empirical non-monotonicity within `tol_scale · ucb` is binomial
    /// noise, larger steps refuse the span.
    pub monotone_tol_scale: f32,
}

impl CalibrateConfig {
    /// The paper's default configuration: simple δ (no grid correction),
    /// monotonicity tolerance at half the UCB correction.
    pub fn new(epsilon_plus: f32, epsilon_minus: f32, delta: f32) -> Self {
        Self {
            epsilon_plus,
            epsilon_minus,
            delta,
            delta_over_grid: false,
            monotone_tol_scale: 0.5,
        }
    }
}

/// Caller-owned calibration scratch — per-grid-point risk/efficiency
/// curves, grown once and reused (zero-alloc steady state).
#[derive(Debug, Clone, Default)]
pub struct CalibrateScratch {
    upper_risks: Vec<f32>,
    lower_risks: Vec<f32>,
    computes: Vec<f32>,
}

impl CalibrateScratch {
    /// Empty scratch (grows on first use).
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-sized for the given grid lengths (the warm-loop form — the G4
    /// gate constructs once, reuses across calibrations).
    pub fn with_capacity(upper_len: usize, lower_len: usize) -> Self {
        Self {
            upper_risks: Vec::with_capacity(upper_len),
            lower_risks: Vec::with_capacity(lower_len),
            computes: Vec::with_capacity(lower_len),
        }
    }

    /// Grows capacity (never shrinks) to cover the given grid lengths.
    pub fn reserve(&mut self, upper_len: usize, lower_len: usize) {
        self.upper_risks
            .reserve(upper_len.saturating_sub(self.upper_risks.capacity()));
        self.lower_risks
            .reserve(lower_len.saturating_sub(self.lower_risks.capacity()));
        self.computes
            .reserve(lower_len.saturating_sub(self.computes.capacity()));
    }
}

/// The calibration result: the selected policy plus everything the caller
/// needs to audit the certificate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibratedPolicy {
    /// The selected dual-threshold policy.
    pub policy: DualExitPolicy,
    /// Grid index (into the upper grid) of the selected λ+.
    pub upper_index: usize,
    /// Grid index (into the lower grid) of the selected schedule.
    pub lower_index: usize,
    /// True when BOTH steps certified a feasible point (`Risk̂ + ucb ≤ ε`).
    pub feasible: bool,
    /// True when either step found nothing feasible and fell back to the
    /// most conservative grid point — the guarantee could NOT be
    /// established at this (ε, δ, n); the caller is told, not protected
    /// by accident.
    pub fell_back: bool,
    /// Empirical FP risk of the selected λ+ on the validation set.
    pub fp_risk_hat: f32,
    /// UCB correction applied at step 1.
    pub fp_ucb: f32,
    /// Empirical farsighted FN risk of the selected schedule.
    pub fn_risk_hat: f32,
    /// UCB correction applied at step 2 (same n; differs only under
    /// `delta_over_grid` when the grids have different lengths).
    pub fn_ucb: f32,
    /// Mean normalized compute of the selected policy (the efficiency
    /// objective's value at the selection).
    pub mean_normalized_compute: f32,
    /// Validation-set size the certificate is built on.
    pub n: u32,
    /// First monotonicity violation in the UPPER risk curve, if any (the
    /// span past it was refused — excluded from selection).
    pub upper_monotonicity: Option<MonotoneViolation>,
    /// First monotonicity violation in the LOWER risk curve, if any.
    pub lower_monotonicity: Option<MonotoneViolation>,
}

/// The effective δ for one calibration step under the grid-correction
/// switch.
#[inline]
fn effective_delta(cfg: &CalibrateConfig, grid_len: usize) -> f32 {
    if cfg.delta_over_grid {
        cfg.delta / grid_len.max(1) as f32
    } else {
        cfg.delta
    }
}

/// Two-step decoupled UCB calibration (paper Algorithm 1), zero-alloc in
/// the steady state via caller-owned `scratch`.
///
/// **Step 1** sweeps the upper grid (ascending λ+): empirical FP risk
/// (which should DECREASE in λ+) + UCB ≤ ε+ selects the *smallest* feasible
/// λ+ (most aggressive = most savings among certified points).
/// **Step 2** conditions on that λ+ and sweeps the lower grid (ascending
/// c): empirical farsighted FN risk (measured direction verified against
/// [`MonotoneDir::Increasing`]) + UCB ≤ ε− filters feasibility, then the
/// *minimum mean normalized compute* among feasible points wins.
///
/// Refusals: a monotonicity violation in either curve refuses the span past
/// it (those grid points are excluded from selection); no feasible point
/// falls back to the most conservative grid point with `fell_back = true`.
pub fn calibrate_into(
    samples: &[TrajectorySample<'_>],
    cfg: &CalibrateConfig,
    upper_grid: &[f32],
    lower_grid: &[ScheduleParams],
    scratch: &mut CalibrateScratch,
) -> CalibratedPolicy {
    let n = samples.len() as u32;
    let delta_plus = effective_delta(cfg, upper_grid.len() + lower_grid.len());
    let delta_minus = delta_plus; // one union over both grids (documented)
    let ucb_plus = ucb_half_width(n, delta_plus);
    let ucb_minus = ucb_half_width(n, delta_minus);
    let tol_plus = cfg.monotone_tol_scale * ucb_plus;
    let tol_minus = cfg.monotone_tol_scale * ucb_minus;

    // ── Step 1: λ+ at ε+ (FP risk, decreasing in λ+) ─────────────────────
    let upper_risks = &mut scratch.upper_risks;
    upper_risks.clear();
    upper_risks.resize(upper_grid.len(), 0.0);
    for (i, &lp) in upper_grid.iter().enumerate() {
        upper_risks[i] = empirical_upper_risk(samples, lp);
    }
    let upper_mono = verify_monotone(upper_risks, MonotoneDir::Decreasing, tol_plus);
    let upper_trusted = upper_mono.map_or(upper_grid.len(), |v| v.index + 1);

    let mut upper_index = None;
    for (i, &risk) in upper_risks.iter().enumerate().take(upper_trusted) {
        if risk + ucb_plus <= cfg.epsilon_plus {
            upper_index = Some(i);
            break;
        }
    }
    let (upper_index, upper_feasible) = match upper_index {
        Some(i) => (i, true),
        // Conservative fallback: the largest trusted λ+ (fewest exits).
        None => (upper_trusted.saturating_sub(1), false),
    };
    let lambda_plus = upper_grid[upper_index];

    // ── Step 2: {c,s,l} at ε−, conditioned on λ+ (FN risk vs c) ─────────
    let lower_risks = &mut scratch.lower_risks;
    let computes = &mut scratch.computes;
    lower_risks.clear();
    lower_risks.resize(lower_grid.len(), 0.0);
    computes.clear();
    computes.resize(lower_grid.len(), 0.0);
    for (i, params) in lower_grid.iter().enumerate() {
        let policy = params.into_policy(lambda_plus);
        lower_risks[i] = empirical_lower_risk(samples, &policy);
        computes[i] = mean_normalized_compute(samples, &policy);
    }
    let lower_mono = verify_monotone(lower_risks, MonotoneDir::Increasing, tol_minus);
    let lower_trusted = lower_mono.map_or(lower_grid.len(), |v| v.index + 1);

    let mut lower_index = None;
    let mut best_compute = f32::INFINITY;
    for i in 0..lower_trusted {
        if lower_risks[i] + ucb_minus <= cfg.epsilon_minus && computes[i] < best_compute {
            best_compute = computes[i];
            lower_index = Some(i);
        }
    }
    let (lower_index, lower_feasible) = match lower_index {
        Some(i) => (i, true),
        // Conservative fallback: the smallest c (gentlest schedule =
        // fewest abandons = lowest FN risk).
        None => (0, false),
    };

    CalibratedPolicy {
        policy: lower_grid[lower_index].into_policy(lambda_plus),
        upper_index,
        lower_index,
        feasible: upper_feasible && lower_feasible,
        fell_back: !upper_feasible || !lower_feasible,
        fp_risk_hat: upper_risks[upper_index],
        fp_ucb: ucb_plus,
        fn_risk_hat: lower_risks[lower_index],
        fn_ucb: ucb_minus,
        mean_normalized_compute: computes[lower_index],
        n,
        upper_monotonicity: upper_mono,
        lower_monotonicity: lower_mono,
    }
}

/// Allocating convenience wrapper over [`calibrate_into`] — the cold path.
/// Warm loops must reuse a [`CalibrateScratch`] (the G4 gate).
pub fn calibrate(
    samples: &[TrajectorySample<'_>],
    cfg: &CalibrateConfig,
    upper_grid: &[f32],
    lower_grid: &[ScheduleParams],
) -> CalibratedPolicy {
    let mut scratch = CalibrateScratch::with_capacity(upper_grid.len(), lower_grid.len());
    calibrate_into(samples, cfg, upper_grid, lower_grid, &mut scratch)
}

// ──────────────────────────────────────────────────────────────────────────
// T1.6 — the App. C distribution-shift tripwire
// ──────────────────────────────────────────────────────────────────────────

/// Rolling `p_i ≥ p_c` monitor (paper App. C): counts lower-exit **rights**
/// (the abandoned instance was genuinely unsolvable — it was an *incorrect*
/// one being filtered, contributing to `p_i`) vs **wrongs** (the abandoned
/// instance was solvable — a *correct* one being filtered, contributing to
/// `p_c`) over a fixed window, and disarms the lower threshold when the
/// filter takes more correct than incorrect instances (`p_i < p_c`) — the
/// regime where the paper proves the upper-threshold guarantee breaks.
/// Upper-only mode (never consulting the lower threshold) remains
/// guaranteed, which is exactly what [`Self::apply_guard`] degrades to.
///
/// Fixed-capacity ring, zero heap, `Copy`-free incremental counts.
pub struct PiGePcMonitor<const W: usize = 64> {
    ring: [bool; W],
    head: usize,
    rights: u32,
    wrongs: u32,
    filled: u32,
    min_n: u32,
}

impl<const W: usize> PiGePcMonitor<W> {
    /// New monitor with the given disarm sample floor (`n ≥ min_n` before
    /// the tripwire may fire — a 2-observation window must not disarm).
    #[inline]
    pub fn new(min_n: u32) -> Self {
        Self {
            ring: [false; W],
            head: 0,
            rights: 0,
            wrongs: 0,
            filled: 0,
            min_n,
        }
    }

    /// Records one lower-exit outcome: `was_right = true` when the abandoned
    /// instance was genuinely unsolvable (a correct filtration), `false`
    /// when it was solvable (a wrong filtration — the farsighted-loss case).
    pub fn record(&mut self, was_right: bool) {
        if self.filled == W as u32 {
            // Evict the oldest observation before overwriting its slot.
            let evicted = self.ring[self.head];
            if evicted {
                self.rights = self.rights.saturating_sub(1);
            } else {
                self.wrongs = self.wrongs.saturating_sub(1);
            }
        } else {
            self.filled += 1;
        }
        self.ring[self.head] = was_right;
        if was_right {
            self.rights += 1;
        } else {
            self.wrongs += 1;
        }
        self.head = (self.head + 1) % W;
    }

    /// Observations currently in the window.
    #[inline]
    pub fn n(&self) -> u32 {
        self.filled
    }

    /// `p_i` — the proportion of lower-exited instances that were
    /// INCORRECT (genuinely unsolvable): the correct filtrations' share.
    /// 0.5 when the window is empty (uninformative, never disarms alone).
    #[inline]
    pub fn p_i(&self) -> f32 {
        if self.filled == 0 {
            return 0.5;
        }
        self.rights as f32 / self.filled as f32
    }

    /// `p_c` — the proportion of lower-exited instances that were CORRECT
    /// (solvable, wrongly abandoned): the wrong filtrations' share.
    #[inline]
    pub fn p_c(&self) -> f32 {
        if self.filled == 0 {
            return 0.5;
        }
        self.wrongs as f32 / self.filled as f32
    }

    /// The App. C tripwire: fire (disarm the lower threshold) once the
    /// window is full enough AND the filter takes more correct than
    /// incorrect instances (`p_i < p_c`).
    #[inline]
    pub fn should_disarm(&self) -> bool {
        self.filled >= self.min_n && self.p_i() < self.p_c()
    }

    /// Applies the guard to a per-tick verdict: an [`ExitVerdict::Abandon`]
    /// becomes [`ExitVerdict::Continue`] while disarmed (the upper-only
    /// degradation that keeps the guarantee); every other verdict passes
    /// through untouched.
    #[inline]
    pub fn apply_guard(&self, v: ExitVerdict) -> ExitVerdict {
        if self.should_disarm() && v == ExitVerdict::Abandon {
            ExitVerdict::Continue
        } else {
            v
        }
    }
}

impl<const W: usize> Default for PiGePcMonitor<W> {
    /// Default monitor: window W, disarm floor 32.
    #[inline]
    fn default() -> Self {
        Self::new(32)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T1.7 — unit tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const B: u32 = 32;

    // ── Schedule shapes (T1.2 / Eq. 12–13) ──────────────────────────────

    #[test]
    fn constant_preset_is_exactly_the_midpoint_everywhere() {
        let p = DualExitPolicy::constant(0.9);
        let mid = (p.u - p.l) / 2.0 + p.l;
        for t in 1..=B {
            assert!(
                (p.lambda_minus(t, B) - mid).abs() < 1e-6,
                "constant schedule must be flat (t={t})"
            );
        }
        // c = 0 exactly.
        assert_eq!(p.c, 0.0);
    }

    #[test]
    fn linear_preset_is_a_gentle_near_midpoint_ramp() {
        let p = DualExitPolicy::linear(0.9, B);
        let mid = (p.u - p.l) / 2.0 + p.l;
        let lo = p.lambda_minus(1, B);
        let hi = p.lambda_minus(B, B);
        // c·B = 0.1 ≪ 1: the whole schedule stays within ~2.5% of midpoint.
        assert!((lo - mid).abs() < 0.025 && (hi - mid).abs() < 0.025);
        // Weakly increasing across the horizon (a ramp, not a cliff).
        assert!(hi >= lo);
        assert!(p.s == 0.5, "linear preset centers at s = 0.5");
    }

    #[test]
    fn exponential_preset_is_convex_rising() {
        let p = DualExitPolicy::exponential(0.9, B);
        let at = |frac: f32| p.lambda_minus((frac * B as f32) as u32, B);
        let a = at(0.25);
        let b = at(0.5);
        let c = at(0.75);
        // Convex (accelerating): second half-step gains more than first.
        assert!(
            (c - b) > (b - a),
            "exponential must be convex: quarters {a:.4} {b:.4} {c:.4}"
        );
        assert!(p.s > 1.0, "exponential preset needs s > 1");
    }

    #[test]
    fn log_preset_is_concave_saturating() {
        let p = DualExitPolicy::log(0.9, B);
        let at = |frac: f32| p.lambda_minus((frac * B as f32) as u32, B);
        let a = at(0.25);
        let b = at(0.5);
        let c = at(0.75);
        // Concave (decelerating): second half-step gains less than first.
        assert!(
            (c - b) < (b - a),
            "log must be concave: quarters {a:.4} {b:.4} {c:.4}"
        );
        assert!(p.s < 0.0, "log preset needs s < 0");
    }

    #[test]
    fn lambda_plus_dominates_the_schedule_everywhere() {
        // Mutual exclusivity by construction: λ−(t) < u < λ+ for all t.
        for p in [
            DualExitPolicy::linear(0.9, B),
            DualExitPolicy::exponential(0.9, B),
            DualExitPolicy::log(0.9, B),
            DualExitPolicy::constant(0.9),
        ] {
            assert!(p.u < p.lambda_plus);
            for t in 1..=B {
                let lm = p.lambda_minus(t, B);
                assert!(
                    lm < p.lambda_plus,
                    "λ−({t}) = {lm} must stay below λ+ = {}",
                    p.lambda_plus
                );
            }
        }
    }

    // ── Exit verdict boundaries (T1.1) ──────────────────────────────────

    #[test]
    fn exit_verdict_boundaries_and_mutual_exclusivity() {
        let p = DualExitPolicy::new(0.9, 12.0 / B as f32, 0.5, 0.0, 0.6);
        // Upper boundary: exactly at λ+ commits.
        assert_eq!(p.exit(0.9, 16, B), ExitVerdict::Commit);
        assert_eq!(p.exit(1.0, 16, B), ExitVerdict::Commit);
        // Lower boundary: exactly at λ− abandons (compute it first).
        let lm = p.lambda_minus(16, B);
        assert_eq!(p.exit(lm, 16, B), ExitVerdict::Abandon);
        // Between the two: continue.
        let mid = (p.lambda_plus + lm) / 2.0;
        assert_eq!(p.exit(mid, 16, B), ExitVerdict::Continue);
        // Mutual exclusivity at the same tick: a value ≥ λ+ NEVER abandons
        // (λ+ > λ− everywhere), and a value ≤ λ− NEVER commits.
        assert_ne!(p.exit(p.lambda_plus, 16, B), ExitVerdict::Abandon);
        assert_ne!(p.exit(lm, 16, B), ExitVerdict::Commit);
    }

    #[test]
    fn squeezed_sigmoid_range_and_anchors() {
        // σ(z, l, u) lives in (l, u) and hits the midpoint at z = 0.
        assert!((squeezed_sigmoid(0.0, 0.0, 0.6) - 0.3).abs() < 1e-6);
        assert!(squeezed_sigmoid(-40.0, 0.0, 0.6) <= 1e-4 + 0.0);
        assert!(squeezed_sigmoid(40.0, 0.0, 0.6) >= 0.6 - 1e-4);
        // Squeeze width respected at a mid argument.
        assert!((squeezed_sigmoid(1.0, 0.2, 0.8) - (0.6 * sigmoid(1.0) + 0.2)).abs() < 1e-6);
    }

    // ── Losses (T1.3 / Eq. 8–11) ────────────────────────────────────────

    #[test]
    fn fp_loss_counts_only_wrong_upper_exits() {
        let correct = [false, false, true, true];
        // Commit at tick 2 where correct → 0.
        assert_eq!(
            fp_loss(
                &correct,
                ExitTrace {
                    verdict: TerminalVerdict::Commit,
                    tick: 2
                }
            ),
            0.0
        );
        // Commit at tick 1 where wrong → 1.
        assert_eq!(
            fp_loss(
                &correct,
                ExitTrace {
                    verdict: TerminalVerdict::Commit,
                    tick: 1
                }
            ),
            1.0
        );
        // Abandon / exhausted never count (other losses own them).
        assert_eq!(
            fp_loss(
                &correct,
                ExitTrace {
                    verdict: TerminalVerdict::Abandon,
                    tick: 1
                }
            ),
            0.0
        );
        assert_eq!(
            fp_loss(
                &correct,
                ExitTrace {
                    verdict: TerminalVerdict::Exhausted,
                    tick: 3
                }
            ),
            0.0
        );
    }

    #[test]
    fn farsighted_loss_weights_future_correctness() {
        // 8 steps; correct at steps {5, 6}; abandon at tick 2.
        let correct = [false, false, false, false, false, true, true, false];
        let trace = ExitTrace {
            verdict: TerminalVerdict::Abandon,
            tick: 2,
        };
        // Future-correct from tick 2: steps 2..8 → {5,6} → 2 correct;
        // denominator T − τ = 6 → 2/6.
        assert!((farsighted_loss(&correct, trace) - 2.0 / 6.0).abs() < 1e-6);
        // Abandon at the last step: slice [7..8) = [false] → 0.
        let last = ExitTrace {
            verdict: TerminalVerdict::Abandon,
            tick: 7,
        };
        assert!((farsighted_loss(&correct, last) - 0.0).abs() < 1e-6);
        // Inclusive exit step: abandon at tick 6 → slice [6..8) = [t, f]
        // → 1 correct of 2 (denominator T − τ = 2).
        let at6 = ExitTrace {
            verdict: TerminalVerdict::Abandon,
            tick: 6,
        };
        assert!((farsighted_loss(&correct, at6) - 0.5).abs() < 1e-6);
        // Abandon at tick 5 → slice [5..8) = [t, t, f] → 2/3.
        let at5 = ExitTrace {
            verdict: TerminalVerdict::Abandon,
            tick: 5,
        };
        assert!((farsighted_loss(&correct, at5) - 2.0 / 3.0).abs() < 1e-6);
        // Never-abandon traces carry zero.
        let commit = ExitTrace {
            verdict: TerminalVerdict::Commit,
            tick: 0,
        };
        assert_eq!(farsighted_loss(&correct, commit), 0.0);
    }

    #[test]
    fn regret_loss_is_wasted_compute_after_first_correct() {
        let correct = [false, false, true, true, false];
        // First correct at t'=2; exit at τ=4 → (4−2)/5.
        let t4 = ExitTrace {
            verdict: TerminalVerdict::Commit,
            tick: 4,
        };
        assert!((regret_loss(&correct, t4) - 2.0 / 5.0).abs() < 1e-6);
        // Exit exactly at first correct → 0.
        let t2 = ExitTrace {
            verdict: TerminalVerdict::Commit,
            tick: 2,
        };
        assert_eq!(regret_loss(&correct, t2), 0.0);
        // Never correct → 0 (nothing to waste).
        let never = [false, false, false];
        assert_eq!(
            regret_loss(
                &never,
                ExitTrace {
                    verdict: TerminalVerdict::Exhausted,
                    tick: 2
                }
            ),
            0.0
        );
    }

    #[test]
    fn past_wrongness_counts_wrong_steps_up_to_exit() {
        let correct = [false, true, false, true, true];
        // Exit at τ=2: wrong steps in 0..=2 → {0, 2} → 2/5.
        let t2 = ExitTrace {
            verdict: TerminalVerdict::Commit,
            tick: 2,
        };
        assert!((past_wrongness(&correct, t2) - 2.0 / 5.0).abs() < 1e-6);
        // Exit at τ=0: 1/5.
        let t0 = ExitTrace {
            verdict: TerminalVerdict::Abandon,
            tick: 0,
        };
        assert!((past_wrongness(&correct, t0) - 1.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn run_policy_first_firing_wins_and_exhaustion_is_terminal() {
        // Upper fires at tick 3; lower schedule low enough to never fire.
        let p = DualExitPolicy::new(0.9, 1e-6, 0.5, 0.0, 0.1);
        let s = [0.2, 0.4, 0.6, 0.95, 0.99];
        assert_eq!(
            run_policy(&p, &s),
            ExitTrace {
                verdict: TerminalVerdict::Commit,
                tick: 3
            }
        );
        // Nothing fires → exhausted at the last tick.
        let s2 = [0.2, 0.3, 0.4];
        assert_eq!(
            run_policy(&p, &s2),
            ExitTrace {
                verdict: TerminalVerdict::Exhausted,
                tick: 2
            }
        );
        // Lower fires when confidence stalls below the rising schedule.
        let p2 = DualExitPolicy::new(0.95, 40.0 / B as f32, 0.5, 0.0, 0.9);
        let stall = vec![0.62f32; B as usize];
        let tr = run_policy(&p2, &stall);
        assert_eq!(tr.verdict, TerminalVerdict::Abandon);
        assert!(
            tr.tick < B as usize - 1,
            "stalled run must abandon before budget"
        );
    }

    // ── Hoeffding bound numerics (T1.4) ─────────────────────────────────

    #[test]
    fn ucb_half_width_matches_the_closed_form() {
        // sqrt(ln(1/δ)/(2n)): n=100, δ=0.05 → sqrt(2.9957/200) = 0.12237.
        assert!((ucb_half_width(100, 0.05) - 0.12237).abs() < 1e-4);
        // n=40, δ=0.05 → sqrt(2.9957/80) = 0.19358.
        assert!((ucb_half_width(40, 0.05) - 0.19358).abs() < 1e-4);
        // 1/sqrt(n) shape: quadrupling n halves the width.
        let h1 = ucb_half_width(64, 0.05);
        let h4 = ucb_half_width(256, 0.05);
        assert!((h1 / h4 - 2.0).abs() < 1e-5);
        // Empty set is uninformative, never a certificate.
        assert_eq!(ucb_half_width(0, 0.05), f32::MAX);
    }

    // ── Monotonicity refusal (T1.4) ─────────────────────────────────────

    #[test]
    fn verify_monotone_detects_and_passes() {
        use MonotoneDir::{Decreasing, Increasing};
        // Clean decreasing curve passes.
        assert!(verify_monotone(&[0.5, 0.4, 0.3], Decreasing, 0.01).is_none());
        // Non-monotone within tolerance passes (sampling noise).
        assert!(verify_monotone(&[0.5, 0.4, 0.45, 0.3], Decreasing, 0.1).is_none());
        // Beyond tolerance refuses, reporting the first violating step.
        let v = verify_monotone(&[0.5, 0.4, 0.45, 0.3], Decreasing, 0.01).unwrap();
        assert_eq!(v.index, 1);
        assert!((v.prev - 0.4).abs() < 1e-6 && (v.next - 0.45).abs() < 1e-6);
        // Increasing direction mirrored.
        assert!(verify_monotone(&[0.1, 0.2, 0.3], Increasing, 0.01).is_none());
        let v = verify_monotone(&[0.1, 0.3, 0.2], Increasing, 0.01).unwrap();
        assert_eq!(v.index, 1);
        // Flat curves are trivially monotone in both directions.
        assert!(verify_monotone(&[0.0; 8], Decreasing, 1e-6).is_none());
        assert!(verify_monotone(&[0.0; 8], Increasing, 1e-6).is_none());
        // Single-point grids carry no steps to violate.
        assert!(verify_monotone(&[0.3], Decreasing, 0.0).is_none());
    }

    #[test]
    fn calibrate_refuses_non_monotone_upper_span() {
        // Empirical FP risk by λ+ ∈ {0.55, 0.65, 0.75, 0.85} over these
        // archetypes (25 copies each, n = 100 → ucb(100, 0.05) = 0.122,
        // small enough for interior feasibility):
        //   A1 s=[.58,.58,.58] all-f: crosses 0.55 at a WRONG step only.
        //   A2 s=[.60,.60,.78] c=[t,f,f]: crosses 0.55 CORRECT (tick 0),
        //       then 0.75 WRONG (tick 2).
        //   A3 s=[.50,.56,.76] c=[f,t,f]: crosses 0.55 correct (tick 1),
        //       0.75 wrong (tick 2).
        //   A4 s=[.40,.40,.40]: never crosses.
        //   risks: 0.55 → A1 → 0.25; 0.65 → A2+A3 → 0.5 (RISE: violation);
        //   0.75 → A2+A3 → 0.5; 0.85 → 0.0 (looks attractive but sits PAST
        //   the violation → must be EXCLUDED from selection).
        let a1 = ([0.58f32, 0.58, 0.58], [false, false, false]);
        let a2 = ([0.6f32, 0.60, 0.78], [true, false, false]);
        let a3 = ([0.5f32, 0.56, 0.76], [false, true, false]);
        let a4 = ([0.4f32, 0.40, 0.40], [false, false, false]);
        let mut samples = Vec::with_capacity(100);
        for (s, c) in [&a1, &a2, &a3, &a4] {
            for _ in 0..25 {
                samples.push(TrajectorySample::new(s, c));
            }
        }
        let grid = [0.55f32, 0.65, 0.75, 0.85];
        let lower = [ScheduleParams {
            c: 0.0,
            s: 0.5,
            l: 0.0,
            u: 0.3,
        }];
        let cfg = CalibrateConfig::new(0.4, 0.5, 0.05);
        let mut scratch = CalibrateScratch::new();
        let out = calibrate_into(&samples, &cfg, &grid, &lower, &mut scratch);
        let v = out
            .upper_monotonicity
            .expect("risk rising with λ+ (0.25 → 0.5) must be refused");
        assert_eq!(v.index, 0, "violation is the 0.25 → 0.5 rise");
        assert!((v.prev - 0.25).abs() < 1e-6 && (v.next - 0.5).abs() < 1e-6);
        // Trusted prefix = grid[0] alone; it is feasible (0.25 + 0.122 ≤
        // 0.4) so it is the selection. The risk-0 point at index 3 would
        // ALSO be feasible — the refusal is what excludes it.
        assert_eq!(
            out.upper_index, 0,
            "selection confined to the trusted prefix"
        );
        assert!(out.feasible);
        assert!((out.fp_risk_hat - 0.25).abs() < 1e-6);

        // Contrast: dropping the bump archetype (A3) gives risks
        // [0.33, 0.33, 0.33, 0] — cleanly decreasing → no refusal.
        let mut clean = Vec::with_capacity(75);
        for (s, c) in [&a1, &a2, &a4] {
            for _ in 0..25 {
                clean.push(TrajectorySample::new(s, c));
            }
        }
        let out2 = calibrate_into(&clean, &cfg, &grid, &lower, &mut scratch);
        assert!(
            out2.upper_monotonicity.is_none(),
            "clean curve must not refuse"
        );
    }

    #[test]
    fn calibrate_falls_back_conservatively_when_infeasible() {
        // ε+ = 0 with any positive empirical risk → nothing feasible → the
        // largest λ+ (most conservative) with fell_back = true.
        let samples = [
            TrajectorySample::new(&[0.5, 0.9], &[false, false]), // commits wrong at 0.9
            TrajectorySample::new(&[0.5, 0.5], &[false, false]),
        ];
        let grid = [0.6f32, 0.7, 0.8, 0.95];
        let lower = [ScheduleParams {
            c: 0.0,
            s: 0.5,
            l: 0.0,
            u: 0.3,
        }];
        let cfg = CalibrateConfig::new(0.0, 0.5, 0.05);
        let out = calibrate(&samples, &cfg, &grid, &lower);
        assert!(out.fell_back);
        assert!(!out.feasible);
        assert_eq!(out.upper_index, grid.len() - 1, "fallback = largest λ+");
        assert!((out.policy.lambda_plus - 0.95).abs() < 1e-6);
    }

    #[test]
    fn two_step_interior_selection() {
        // Owned trajectories so the sample slices can borrow them.
        #[derive(Clone)]
        struct Owned {
            s: Vec<f32>,
            c: Vec<bool>,
        }
        let owned = [
            Owned {
                s: vec![0.5, 0.62, 0.62, 0.62],
                c: vec![false, false, false, false],
            }, // crosses 0.55 wrong (tick 1)
            Owned {
                s: vec![0.5, 0.72, 0.72, 0.72],
                c: vec![false, false, false, false],
            }, // crosses 0.55 AND 0.7 wrong
            Owned {
                s: vec![0.5, 0.5, 0.82, 0.82],
                c: vec![false, false, false, true],
            }, // crosses 0.8 at a CORRECT step only
            Owned {
                s: vec![0.5, 0.5, 0.5, 0.5],
                c: vec![false, false, false, false],
            }, // stalls — the lower-exit case
        ];
        // Empirical FP risk by λ+ (n = 4 archetypes): 0.55 → 2/4 (first two
        // commit wrong at tick 1), 0.7 → 1/4 (only the 0.72 instance),
        // 0.85 → 0 (the 0.82 instance crosses 0.85? No — 0.82 < 0.85 → it
        // never commits; risk 0), 0.95 → 0. Curve [0.5, 0.25, 0, 0] —
        // cleanly decreasing ✓. ucb(n=32, δ=0.05) = sqrt(2.9957/64) =
        // 0.2163; ε+ = 0.25 → feasible needs risk ≤ 0.0337 → indices 2, 3
        // (risk 0); smallest feasible = index 2 (0.85) — INTERIOR (index 3
        // also feasible but more conservative).
        let mut samples = Vec::with_capacity(32);
        for o in &owned {
            for _ in 0..8 {
                samples.push(TrajectorySample::new(&o.s, &o.c));
            }
        }
        let grid = [0.55f32, 0.7, 0.85, 0.95];
        // Lower grid (c-ascending): the flat schedule (c=0, u=0.3) never
        // reaches the stalled instance's 0.5 confidence; the steep one
        // (c=4, u=0.55) crosses it at tick 2 — z = 4·(t+1 − 2) → σ(4)·0.55
        // = 0.54 ≥ 0.5 → Abandon at tick 2, saving 1 of 4 steps.
        let lower = [
            ScheduleParams {
                c: 0.0,
                s: 0.5,
                l: 0.0,
                u: 0.3,
            },
            ScheduleParams {
                c: 4.0,
                s: 0.5,
                l: 0.0,
                u: 0.55,
            },
        ];
        let cfg = CalibrateConfig::new(0.25, 0.5, 0.05);
        let out = calibrate(&samples, &cfg, &grid, &lower);
        assert!(
            out.feasible,
            "ε+=0.25 with ucb 0.216 admits the risk-0 point"
        );
        assert_eq!(out.upper_index, 2, "smallest feasible λ+ = 0.85 (risk 0)");
        assert!((out.policy.lambda_plus - 0.85).abs() < 1e-6);
        assert!((out.fp_risk_hat - 0.0).abs() < 1e-6);
        // Step 2 must also be feasible (FN risk 0 on this population: no
        // solvable instance stalls under the schedule) and pick the
        // cheaper schedule among feasible points.
        assert!(
            out.lower_index == 1,
            "the steeper schedule saves compute on the stalled instance"
        );
    }

    // ── Tripwire (T1.6 / App. C) ────────────────────────────────────────

    #[test]
    fn tripwire_disarms_when_filter_takes_more_correct_than_incorrect() {
        let mut m = PiGePcMonitor::<8>::new(4);
        // 3 rights (genuinely unsolvable filtered) + 1 wrong.
        for r in [true, true, true, false] {
            m.record(r);
        }
        assert_eq!(m.n(), 4);
        assert!((m.p_i() - 0.75).abs() < 1e-6);
        assert!((m.p_c() - 0.25).abs() < 1e-6);
        assert!(!m.should_disarm(), "p_i > p_c — keep the lower threshold");
        // Flip the mix: 1 right + 3 wrongs.
        let mut m2 = PiGePcMonitor::<8>::new(4);
        for r in [false, false, false, true] {
            m2.record(r);
        }
        assert!(m2.should_disarm(), "p_i < p_c — disarm");
        // The guard maps Abandon → Continue while disarmed, passes the rest.
        assert_eq!(m2.apply_guard(ExitVerdict::Abandon), ExitVerdict::Continue);
        assert_eq!(m2.apply_guard(ExitVerdict::Commit), ExitVerdict::Commit);
        assert_eq!(m2.apply_guard(ExitVerdict::Continue), ExitVerdict::Continue);
        // Undisarmed monitor passes Abandon through.
        assert_eq!(m.apply_guard(ExitVerdict::Abandon), ExitVerdict::Abandon);
    }

    #[test]
    fn tripwire_respects_the_sample_floor() {
        let mut m = PiGePcMonitor::<8>::new(4);
        // 3 wrongs only — p_i = 0 < p_c = 0.75, but n = 3 < min_n = 4.
        for _ in 0..3 {
            m.record(false);
        }
        assert!(!m.should_disarm(), "below the disarm floor");
        m.record(false);
        assert!(m.should_disarm(), "at the floor with p_i < p_c");
    }

    #[test]
    fn tripwire_ring_eviction_keeps_counts_consistent() {
        let mut m = PiGePcMonitor::<4>::new(4);
        // Fill with rights, then push wrongs through — counts must track
        // the live window, not the history.
        for _ in 0..4 {
            m.record(true);
        }
        assert_eq!((m.rights, m.wrongs), (4, 0));
        for _ in 0..4 {
            m.record(false);
        }
        assert_eq!((m.rights, m.wrongs), (0, 4));
        assert_eq!(m.n(), 4);
        // Mixed eviction: R W W R over a full window.
        for r in [true, false, false, true] {
            m.record(r);
        }
        assert_eq!((m.rights, m.wrongs), (2, 2));
        assert!((m.p_i() - 0.5).abs() < 1e-6);
        assert!(!m.should_disarm(), "p_i == p_c is not < — keep armed");
    }

    // ── Calibration zero-alloc surface (mirrors the G4 binary) ─────────

    #[test]
    fn calibrate_into_reuses_scratch_without_growing() {
        let owned: Vec<(Vec<f32>, Vec<bool>)> = (0..16)
            .map(|i| {
                let s: Vec<f32> = (0..8).map(|t| 0.3 + 0.05 * (t + i % 3) as f32).collect();
                let c: Vec<bool> = (0..8).map(|t| t >= 6 && i % 2 == 0).collect();
                (s, c)
            })
            .collect();
        let samples: Vec<TrajectorySample<'_>> = owned
            .iter()
            .map(|(s, c)| TrajectorySample::new(s, c))
            .collect();
        let grid = [0.6f32, 0.7, 0.8];
        let lower = [
            ScheduleParams {
                c: 0.0,
                s: 0.5,
                l: 0.0,
                u: 0.3,
            },
            ScheduleParams {
                c: 1.0,
                s: 0.5,
                l: 0.0,
                u: 0.3,
            },
        ];
        let cfg = CalibrateConfig::new(0.2, 0.2, 0.05);
        let mut scratch = CalibrateScratch::with_capacity(grid.len(), lower.len());
        let _ = calibrate_into(&samples, &cfg, &grid, &lower, &mut scratch);
        let cap_before = (
            scratch.upper_risks.capacity(),
            scratch.lower_risks.capacity(),
            scratch.computes.capacity(),
        );
        for _ in 0..8 {
            let out = calibrate_into(&samples, &cfg, &grid, &lower, &mut scratch);
            assert!(out.n == 16);
        }
        assert_eq!(
            cap_before,
            (
                scratch.upper_risks.capacity(),
                scratch.lower_risks.capacity(),
                scratch.computes.capacity()
            ),
            "steady-state calibration must not grow the scratch"
        );
    }
}
