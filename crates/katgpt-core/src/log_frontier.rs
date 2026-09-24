//! Log-frontier budget tracker — the modelless extraction of MAttr's
//! `AdaptiveLogK` schedule controller (upstream `schedules.py`, not in the
//! paper text; Research 584 / Issue 879).
//!
//! A single scalar `k_max_log` — the log of the budget ceiling — adapted by
//! **sign steps** against a scalar accuracy target. The tracker consumes
//! NO ranking, no scores, no gradients: only `acc: f32`. That is the whole
//! reason it is modelless — MAttr uses it to pick the matryoshka budget
//! ladder during RL, we use it anywhere a budget dial must track a quality
//! signal (KV `DensityBudget` ladder boundaries, `thermal_lod` attention_k
//! tier elbows, and the k-supervision lane riir-clippy Issue 133 wants for
//! `rule_embed` — randomized-budget sampling is the other half of this
//! feature family, see [`crate::exact_mass_admit`]).
//!
//! # Protocol
//!
//! - [`sample`](LogFrontier::sample) draws the next budget: log-uniform over
//!   `[1, k_max)` from a caller-supplied unit uniform `u01 ∈ [0, 1)` — the
//!   caller owns the randomness, so the tracker is deterministic and
//!   seed-free. Every `probe_every`-th draw (cadence derived from
//!   `probe_frac`, e.g. 0.25 → every 4th) is instead a **probe at exactly
//!   `k_max`** and arms the controller.
//! - [`observe`](LogFrontier::observe) feeds the outcome of the most recent
//!   draw. Only probe outcomes move the controller: `acc < target` →
//!   `+lr` in log space (the ceiling must grow), `acc > target` → `−lr`
//!   (the ceiling can shrink), `== target` → hold. Non-probe observes are
//!   no-ops — a log-uniform sample says nothing about the ceiling. `k_max_log`
//!   is clamped to `[ln floor, ln total]`.
//!
//! Zero-alloc, `Copy`, no deps. Distilled from the upstream controller's
//! published behavior; no code copied (the upstream repo carries no
//! license).

/// Deterministic log-space budget controller over a scalar accuracy signal
/// (MAttr `AdaptiveLogK`, modelless extraction).
///
/// ```ignore
/// use katgpt_core::log_frontier::LogFrontier;
///
/// let mut lf = LogFrontier::new(1024, 0.90, 0.05, 0.25, 8);
/// let k = lf.sample(0.37);          // log-uniform in [1, 1024)
/// // ... run the budgeted thing, measure accuracy ...
/// lf.observe(0.85);                 // moves ONLY if `k` was a probe
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogFrontier {
    k_max_log: f32,
    floor_log: f32,
    ceil_log: f32,
    target: f32,
    lr: f32,
    probe_every: u32,
    step: u32,
    armed: bool,
}

impl LogFrontier {
    /// Build the tracker.
    ///
    /// - `total` — upper budget bound (`k_max` starts here, permissive);
    ///   `total ≥ 2` (a one-element frontier has no log scale).
    /// - `target` — the accuracy the probe budget must hold.
    /// - `lr` — sign-step size in log space.
    /// - `probe_frac` — fraction of draws that probe at exactly `k_max`
    ///   (`0.25` → every 4th draw). `0` disarms the controller entirely
    ///   (sample still works; observe is a permanent no-op). `≥ 1` probes
    ///   every draw.
    /// - `floor` — lower budget bound, clamped into `[1, total]`.
    pub fn new(total: usize, target: f32, lr: f32, probe_frac: f32, floor: usize) -> Self {
        debug_assert!(total >= 2, "total must be >= 2 for a log frontier");
        debug_assert!(lr.is_finite() && lr > 0.0, "lr must be a positive finite step");
        debug_assert!(target.is_finite(), "target must be finite");
        let floor = floor.clamp(1, total);
        let ceil_log = (total as f32).ln();
        let floor_log = (floor as f32).ln();
        let probe_every = if probe_frac <= 0.0 {
            0
        } else if probe_frac >= 1.0 {
            1
        } else {
            ((1.0f32 / probe_frac).round() as u32).max(1)
        };
        Self {
            k_max_log: ceil_log,
            floor_log,
            ceil_log,
            target,
            lr,
            probe_every,
            step: 0,
            armed: false,
        }
    }

    /// Current budget ceiling `k_max = exp(k_max_log)`.
    #[inline]
    pub fn k_max(&self) -> f32 {
        self.k_max_log.exp()
    }

    /// Draw the next budget. Pure function of `(state, u01)` — deterministic
    /// by construction (the caller owns the randomness).
    ///
    /// Non-probe draws: `k = exp(u01 · k_max_log) ∈ [1, k_max)`.
    /// Probe draws (every `probe_every`-th call): `k = k_max` exactly, and
    /// the controller arms for the next [`observe`](Self::observe).
    pub fn sample(&mut self, u01: f32) -> f32 {
        self.step = self.step.wrapping_add(1);
        let probe = self.probe_every > 0 && self.step.is_multiple_of(self.probe_every);
        self.armed = probe;
        if probe {
            self.k_max_log.exp()
        } else {
            let u = if !(0.0..1.0).contains(&u01) {
                // Clamp caller noise into [0, 1): the draw must stay a
                // valid log-uniform sample.
                u01.clamp(0.0, 1.0 - f32::EPSILON)
            } else {
                u01
            };
            (u * self.k_max_log).exp()
        }
    }

    /// Feed the outcome of the most recent draw (an accuracy in the
    /// tracker's own units — only compared against `target`).
    ///
    /// Moves `k_max_log` by `±lr` **iff** the last [`sample`](Self::sample)
    /// was a probe; clamped to `[ln floor, ln total]`. Returns the new
    /// `k_max` (convenient for logging; `floor`/`ceil` saturation is
    /// visible as `k_max` stopping at `floor`/`total`).
    pub fn observe(&mut self, acc: f32) -> f32 {
        if !self.armed {
            return self.k_max();
        }
        self.armed = false;
        let dir = if acc < self.target {
            1.0
        } else if acc > self.target {
            -1.0
        } else {
            0.0
        };
        self.k_max_log = (self.k_max_log + dir * self.lr).clamp(self.floor_log, self.ceil_log);
        self.k_max()
    }
}
