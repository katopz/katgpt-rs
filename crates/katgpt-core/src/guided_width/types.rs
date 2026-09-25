//! Configuration, caller-owned scratch, and report types for the guided
//! width rollouts (Issue 895). Decoupled from the operators per the
//! `types.rs` convention.

use crate::saddle_escape::FlipDetector;
use crate::simd::fast_sigmoid;
use crate::speculative::qmc::{SOBOL_MAX_DIM, SobolQmc};

use super::table::{DirectionPosterior, DirectionTable};

/// Upper bound on parallel branches per decision (stack-array bound for the
/// scorer and the returned-set selector). Callers asking for more are
/// clamped, never allocated for.
pub const MAX_BRANCHES: usize = 32;

/// Upper bound on the rows a [`DirectionTable`] may carry.
pub const MAX_DIRECTIONS: usize = 32;

// ─────────────────────────────────────────────────────────────────────
// T2 — stagnation-gated variance
// ─────────────────────────────────────────────────────────────────────

/// Stagnation-gated exploration budget (T2):
///
/// ```text
/// σ_t = σ_max · sigmoid(α · (w_stuck − w₀))
/// ```
///
/// `w_stuck` counts consecutive steps whose deterministic update norm fell
/// below `stuck_tol` (reset to 0 on any larger step). A progressing branch
/// gets `≈ σ_max·sigmoid(−α w₀)`; a stuck one ramps toward `σ_max`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StagnationGate {
    /// Ceiling of the per-step exploration magnitude. `0` is the kill switch.
    pub sigma_max: f32,
    /// Sigmoid slope on the stuck count.
    pub alpha: f32,
    /// Stuck count at which σ_t reaches `σ_max / 2`.
    pub w0: f32,
    /// Update-norm threshold below which a step counts as stuck.
    pub stuck_tol: f32,
}

impl StagnationGate {
    /// `σ_max 0.25, α 2, w₀ 2, stuck_tol 1e-2` — the Bench 898 pre-stated
    /// configuration (size `sigma_max` and `stuck_tol` to the host's scale).
    pub const DEFAULT: Self = Self {
        sigma_max: 0.25,
        alpha: 2.0,
        w0: 2.0,
        stuck_tol: 1e-2,
    };

    /// `σ_max` sanitised: non-finite or non-positive ⇒ exactly `0.0` (the
    /// kill switch). NaN never becomes noise.
    #[inline]
    pub fn sigma_max_sanitised(&self) -> f32 {
        if self.sigma_max.is_finite() && self.sigma_max > 0.0 {
            self.sigma_max
        } else {
            0.0
        }
    }

    /// `σ_t` for a branch that has been stuck for `w_stuck` steps. Always
    /// finite and in `[0, σ_max]`; any NaN in the config yields `0.0`.
    #[inline]
    pub fn sigma(&self, w_stuck: u16) -> f32 {
        let smax = self.sigma_max_sanitised();
        if smax == 0.0 {
            return 0.0;
        }
        let s = smax * fast_sigmoid(self.alpha * (w_stuck as f32 - self.w0));
        if s.is_finite() { s } else { 0.0 }
    }
}

impl Default for StagnationGate {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ─────────────────────────────────────────────────────────────────────
// T3 — decode-free latent scorer config
// ─────────────────────────────────────────────────────────────────────

/// Weights of the decode-free latent scorer (T3). Every term is a sigmoid of
/// a latent quantity; the combination is a RANKING signal only — never a
/// calibrated probability (a confidence consumer must pass the Report-the-
/// Floor conformal gate first).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentValueConfig {
    /// Weight of the self-consistency term `mean_{j≠i} σ(β(cos(h_i,h_j) − τ))`.
    pub w_consistency: f32,
    /// Slope of the self-consistency kernel.
    pub beta: f32,
    /// Cosine at which two hypotheses count as half-agreeing.
    pub tau: f32,
    /// Weight of the convergence term `2σ(−r_i / r_scale) ∈ (0, 1]`.
    pub w_residual: f32,
    /// Residual scale (size to the step norm the host calls converged).
    pub residual_scale: f32,
    /// Weight of the optional frozen-direction term `σ(β_d · cos(h_i, d))`.
    pub w_direction: f32,
    /// Slope of the frozen-direction term.
    pub beta_direction: f32,
}

impl LatentValueConfig {
    /// `w_c 1, β 20, τ 0.9, w_r 1, r_scale 1e-2, w_d 1, β_d 4`.
    pub const DEFAULT: Self = Self {
        w_consistency: 1.0,
        beta: 20.0,
        tau: 0.9,
        w_residual: 1.0,
        residual_scale: 1e-2,
        w_direction: 1.0,
        beta_direction: 4.0,
    };
}

impl Default for LatentValueConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ─────────────────────────────────────────────────────────────────────
// T6 — trap-kill-reallocate config
// ─────────────────────────────────────────────────────────────────────

/// Trap handling for noisy branches (T6), consuming `saddle_escape`'s
/// [`FlipDetector`] over the pre-decode [`super::belief_key`] and
/// `apply_kick` as the kick. The NEW behaviour is only the last line: a
/// branch whose kick budget is spent is KILLED and its remaining step budget
/// re-spent on a fresh hypothesis in the same slot (width, not cooldown).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrapReallocConfig {
    /// Flip-rate EMA at or above which a branch counts as trapped.
    pub flip_tau: f32,
    /// FlipDetector window (hysteresis).
    pub window: u16,
    /// Kicks per branch before it is killed.
    pub kick_budget: u8,
    /// First kick magnitude (absolute).
    pub eps0: f32,
    /// Geometric kick decay.
    pub eps_decay: f32,
    /// Probe confirmation threshold: when a probe is supplied its signal
    /// must be `>= probe_tau` (NaN never confirms) — e.g. the codifferential
    /// circulation signal of the cochain host.
    pub probe_tau: f32,
    /// Re-spend a killed branch's remaining budget on a fresh hypothesis.
    /// `false` = the branch simply stops (budget freed, not reallocated).
    pub respawn: bool,
}

impl TrapReallocConfig {
    /// `flip_tau 0.5, window 4, kick_budget 1, eps0 0.1, decay 0.5,
    /// probe_tau 0.5, respawn true`.
    pub const DEFAULT: Self = Self {
        flip_tau: 0.5,
        window: 4,
        kick_budget: 1,
        eps0: 0.1,
        eps_decay: 0.5,
        probe_tau: 0.5,
        respawn: true,
    };
}

impl Default for TrapReallocConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ─────────────────────────────────────────────────────────────────────
// Driver config
// ─────────────────────────────────────────────────────────────────────

/// Configuration of one guided-width decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuidedWidthConfig {
    /// Parallel hypotheses N (clamped to [`MAX_BRANCHES`] and the scratch
    /// capacity). `≤ 1` is the kill switch.
    pub n_branches: usize,
    /// Steps per hypothesis K. Every branch spends exactly K step calls;
    /// the equal-compute depth baseline is one trajectory of N·K.
    pub k_steps: usize,
    /// T2 exploration budget.
    pub gate: StagnationGate,
    /// Initial Sobol spread, as a multiple of `σ_max` (so `σ_max = 0` zeroes
    /// it — the kill switch covers init diversity too).
    pub init_spread: f32,
    /// Guidance gain κ: the μ≠0 term is `κ·σ·s·d_j` (at init with `σ_max`,
    /// per step with `σ_t`).
    pub guidance_gain: f32,
    /// T3 scorer weights.
    pub score: LatentValueConfig,
    /// T6 trap handling (`None` = off).
    pub trap: Option<TrapReallocConfig>,
    /// BLAKE3 seed material for every draw in the decision.
    pub seed: [u8; 32],
}

impl GuidedWidthConfig {
    /// N=8, K=16, [`StagnationGate::DEFAULT`], spread 1, κ 1, default scorer,
    /// no trap handling, zero seed.
    pub const DEFAULT: Self = Self {
        n_branches: 8,
        k_steps: 16,
        gate: StagnationGate::DEFAULT,
        init_spread: 1.0,
        guidance_gain: 1.0,
        score: LatentValueConfig::DEFAULT,
        trap: None,
        seed: [0u8; 32],
    };

    /// `true` when this config takes the incumbent path verbatim.
    #[inline]
    pub fn is_kill_switch(&self) -> bool {
        self.n_branches <= 1 || self.gate.sigma_max_sanitised() == 0.0
    }
}

impl Default for GuidedWidthConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ─────────────────────────────────────────────────────────────────────
// Hooks
// ─────────────────────────────────────────────────────────────────────

/// A trap-confirmation probe over a branch's last deterministic update.
pub type TrapProbe<'a> = &'a mut dyn FnMut(&[f32]) -> f32;

/// The μ≠0 guidance source (T5): a frozen direction table plus its runtime
/// Beta posterior (a latent overlay, not part of the frozen commitment).
#[derive(Clone, Copy)]
pub struct Guidance<'a> {
    /// Frozen, BLAKE3-committed direction table.
    pub table: &'a DirectionTable,
    /// Per-direction success/failure counts.
    pub posterior: &'a DirectionPosterior,
    /// ε of the `best_belief_score` lower bound used to rank directions.
    pub epsilon: f32,
    /// A direction whose sign-bias is at least this is used with its
    /// learned sign by every branch; below it, branches alternate ±.
    pub bias_tau: f32,
}

/// Optional inputs of one decision. `Hooks::default()` is the table-absent,
/// probe-free, frozen-direction-free call.
#[derive(Default)]
pub struct Hooks<'a> {
    /// μ≠0 guidance (`None` ⇒ isotropic+transversal, bit-identical).
    pub guidance: Option<Guidance<'a>>,
    /// Frozen direction for the scorer's third term (`None` ⇒ term off).
    pub frozen_direction: Option<&'a [f32]>,
    /// Trap-confirmation probe over the branch's last deterministic update
    /// (e.g. the codifferential circulation signal). `None` ⇒ flip rate alone.
    pub probe: Option<TrapProbe<'a>>,
}

// ─────────────────────────────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────────────────────────────

/// What one decision did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RolloutReport {
    /// Index of the selected branch (`0` = the deterministic branch).
    pub best: usize,
    /// Its latent value (`NaN` on the kill-switch path — nothing scored).
    pub best_value: f32,
    /// Direction `(j, sign)` the selected branch was committed to, if any —
    /// the caller's posterior update key.
    pub direction: Option<(u16, i8)>,
    /// Deterministic step calls spent (the compute budget; `≤ N·K`).
    pub step_evals: u32,
    /// Escape kicks applied.
    pub kicks: u16,
    /// Branches killed and re-spawned (budget reallocated to width).
    pub respawns: u16,
    /// Branches killed without respawn (budget freed).
    pub killed: u16,
    /// `true` when the kill-switch (incumbent) path ran.
    pub incumbent: bool,
}

// ─────────────────────────────────────────────────────────────────────
// Scratch
// ─────────────────────────────────────────────────────────────────────

/// Caller-owned scratch for [`super::guided_width_rollouts`]. Allocate once
/// with [`Self::with_capacity`]; every decision up to that capacity is
/// allocation-free.
pub struct GuidedWidthScratch {
    pub(crate) n_max: usize,
    pub(crate) d: usize,
    /// `n × d` branch states (row-major, branch-major).
    pub(crate) states: Vec<f32>,
    /// Per-step deterministic update `u_t − h_{t−1}` (len d).
    pub(crate) delta: Vec<f32>,
    /// ε draw buffer (len d).
    pub(crate) noise: Vec<f32>,
    /// Sobol draw buffer (`n × d`).
    pub(crate) sobol: Vec<f32>,
    /// Cached Sobol source (`min(d, SOBOL_MAX_DIM)` dims), reseeded per
    /// decision — built once here because its construction allocates.
    pub(crate) sobol_src: SobolQmc,
    /// Last deterministic update norm per branch.
    pub(crate) residual: Vec<f32>,
    /// Latent values per branch.
    pub(crate) values: Vec<f32>,
    /// Consecutive stuck steps per branch.
    pub(crate) w_stuck: Vec<u16>,
    /// Committed direction per branch.
    pub(crate) dir: Vec<Option<(u16, i8)>>,
    /// Flip detectors per branch (T6).
    pub(crate) flips: Vec<FlipDetector>,
    /// Kicks spent per branch.
    pub(crate) kicks: Vec<u8>,
    /// Respawn generation per branch (seeds a fresh hypothesis).
    pub(crate) generation: Vec<u16>,
    /// Branch still stepping (a killed, non-respawned branch stops).
    pub(crate) alive: Vec<bool>,
    /// Farthest-point workspaces (T4 returned set).
    pub(crate) min_dist: Vec<f32>,
    pub(crate) is_selected: Vec<bool>,
    /// Branch count of the last decision (for [`super::diverse_set_into`]).
    pub(crate) last_n: usize,
}

impl GuidedWidthScratch {
    /// Scratch for up to `n_max` branches (clamped to [`MAX_BRANCHES`]) of
    /// dimension `d`.
    pub fn with_capacity(n_max: usize, d: usize) -> Self {
        let n = n_max.clamp(1, MAX_BRANCHES);
        Self {
            n_max: n,
            d,
            states: vec![0.0; n * d],
            delta: vec![0.0; d],
            noise: vec![0.0; d],
            sobol: vec![0.0; n * d],
            sobol_src: SobolQmc::new_multi(0, d.clamp(1, SOBOL_MAX_DIM)),
            residual: vec![0.0; n],
            values: vec![0.0; n],
            w_stuck: vec![0; n],
            dir: vec![None; n],
            flips: vec![FlipDetector::new(1); n],
            kicks: vec![0; n],
            generation: vec![0; n],
            alive: vec![true; n],
            min_dist: Vec::with_capacity(n),
            is_selected: Vec::with_capacity(n),
            last_n: 0,
        }
    }

    /// Branch capacity.
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    /// State dimension.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Final state of branch `b` of the last decision.
    pub fn branch(&self, b: usize) -> &[f32] {
        &self.states[b * self.d..(b + 1) * self.d]
    }

    /// Branch count of the last decision (`0` after the kill-switch path).
    pub fn last_branches(&self) -> usize {
        self.last_n
    }

    /// Latent values of the last decision (`last_branches()` entries).
    pub fn values(&self) -> &[f32] {
        &self.values[..self.last_n]
    }

    /// Final deterministic-update residuals of the last decision.
    pub fn residuals(&self) -> &[f32] {
        &self.residual[..self.last_n]
    }

    /// Committed direction of branch `b` in the last decision.
    pub fn branch_direction(&self, b: usize) -> Option<(u16, i8)> {
        self.dir[b]
    }
}
