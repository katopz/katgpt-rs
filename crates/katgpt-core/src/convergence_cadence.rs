//! `ConvergenceCadence` — windowed update-magnitude outcome classifier
//! (Issue 720 T1; source: Research 529, HRM mechanistic dissection,
//! Rodrigues & Kang, "Dissecting Hierarchical Reasoning Models", ICML 2026
//! Mech Interp Workshop — Finding 4, n=93/107 runs).
//!
//! # The signal
//!
//! On an iterative refinement loop, the **windowed trajectory of update
//! magnitudes** classifies the run's outcome long before the final step:
//!
//! | window tail | solved | failed |
//! |---|---|---|
//! | ‖Δz‖ at step 7-8 | **0.30** (decayed to fixed point) | **1.46** (~4.9× — plateaued HIGH) |
//! | consecutive-state cos | → 0.998 | stalls ~0.97 |
//! | consecutive-update cos | ≈ 0 | ≈ 0 (rotational churn) |
//!
//! Solved runs **decay**; failed runs **plateau high**. The shape — not any
//! single value — is the classifier.
//!
//! # What this is NOT (the signal-diff, pinned from the issue)
//!
//! - **NOT a halt signal.** [`crate::gain_cost_halt::GainCostLoopHalter`]
//!   (Plan 304 / Research 282) consumes step size ‖Δh‖ for HALT only
//!   (decay = concavity stop, growth = expansion stop). Halting ≠
//!   classification: a plateau-high run eventually halts, but the caller
//!   cannot tell "nothing left to gain" from "stuck churning — escalate".
//!   This probe is the **outcome read** the halter lacks.
//! - **NOT a novelty signal.** `DerivativeCuriosity` / `TemporalDerivativeKernel`
//!   (Plan 277) reads churn as *interesting* (explore). Here churn is
//!   *failing* (abstain/escalate). Different axis, same deltas.
//! - **NOT anti-cheat / NOT a sync surface.** Think-brain telemetry only
//!   (AGENTS.md domain rules); never crosses a SyncBlock.
//!
//! # The three laws (Research 529 §Paper × R35 × 717 × 304 — recorded here
//! per Issue 720 T4b so every consumer inherits them)
//!
//! 1. **Absolute update magnitude, never relative.** Relative residuals are
//!    a growing-denominator trap on non-fixed-point recurrences (R35's
//!    negative half; Issue 717 T6). Both verdicts here are gated by
//!    **absolute floors**; the decay ratio only discriminates *shape*
//!    between them.
//! 2. **Windowed trajectory shape, not a single-step threshold.** Plateau
//!    vs decay is a shape over a window (two half-window means), not one
//!    observation.
//! 3. **Tangential-first before radial damping.** Successive updates are
//!    near-orthogonal (cos_updates ≈ 0) — plateau churn is *rotational*,
//!    so when this probe flags Churning, scale the tangential component
//!    (Issue 717 T4) before damping the radial one.
//!
//! # Consumers (Issue 720)
//!
//! - `GainCostLoopHalter` callers — halt + outcome escalation arm.
//! - Issue 717 T3/T4 — cadence is the degradation DETECTOR its damping
//!   knob lacked ("don't damp unless inference already degrades").
//! - Per-NPC belief loops (`evolve_belief`) — settled → early-commit;
//!   churning → deliberate (riir-mmorpg-examples Issue 054 L2, T3).
//! - Consolidation-side sibling: riir-neuron-db `can_freeze` validates the
//!   same convergence finding at measure time; this probe makes it a live
//!   predictor.
//!
//! # Zero-alloc contract (G4)
//!
//! Fixed `[f32; K]` ring, `Copy` verdicts, O(1) `push`/`classify`. No heap,
//! no slices of caller state — the caller feeds the norm it already computed
//! (e.g. [`crate::gain_cost_halt::step_size`]'s return value).
//!
//! # Non-finite input policy
//!
//! A non-finite norm is a pathological signal and is counted as HIGH (it
//! feeds the plateau run and can only push the verdict toward
//! [`CadenceVerdict::Churning`]). A non-finite window mean classifies as
//! [`CadenceVerdict::Churning`] — loud, never a silent `Settled`.

/// Verdict of a full cadence window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CadenceVerdict {
    /// Healthy window — decaying shape or absolutely-low magnitude.
    /// Semantics for callers: safe to commit / halt normally. NOT a claim
    /// that the answer is correct — only that the loop is not churning.
    Settled {
        /// Mean update magnitude over the newer half-window.
        mag: f32,
    },
    /// Plateaued-high window — the run is churning. Escalate per consumer:
    /// damp (Issue 717, tangential-first), deliberate (NPC think loop),
    /// restart-with-new-conjecture (CGSP).
    Churning {
        /// Mean update magnitude over the newer half-window.
        mag: f32,
        /// Length of the trailing run of samples at or above
        /// [`CadenceConfig::plateau_floor`] (capped at the window length).
        /// How long the loop has been stuck high.
        plateau_len: u32,
    },
}

/// Thresholds for [`ConvergenceCadence`]. All magnitudes are ABSOLUTE
/// (law 1) and must be calibrated to the caller's update scale — the
/// Research 529 HRM values are one calibration point, not a universal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceConfig {
    /// Absolute floor for "stuck high". Newer-half mean at or above this
    /// AND not decaying ⇒ [`CadenceVerdict::Churning`].
    /// Research 529 calibration point: failed runs plateau at 1.46.
    pub plateau_floor: f32,
    /// Absolute ceiling for "settled low". Newer-half mean at or below
    /// this settles regardless of shape (a flat tiny-magnitude window is
    /// quiescent, not churning — law 1 dominates).
    /// Research 529 calibration point: solved runs end at 0.30.
    pub settle_floor: f32,
    /// Maximum newer/older half-window mean ratio still counted as
    /// "decaying" (shape law 2). At or above ⇒ no decay evidence.
    /// Paper: solved ≈ 0.30 ratio by step 7-8; failed ≈ 1.0 (plateau).
    pub decay_ratio_max: f32,
}

impl Default for CadenceConfig {
    fn default() -> Self {
        Self {
            // Between the paper's solved endpoint (0.30) and failed
            // plateau (1.46): a mid-scale starting point for callers.
            plateau_floor: 1.0,
            settle_floor: 0.5,
            decay_ratio_max: 0.5,
        }
    }
}

/// Windowed update-magnitude cadence probe. Zero-alloc, O(1) per step.
///
/// Generic over the window length `K` (default 16 — the paper's signal is
/// readable by step 7-8, so two 8-sample half-windows see it). `K` must be
/// even and `>= 4` (enforced at construction).
///
/// The caller feeds one update magnitude per refinement step (‖Δh‖ from
/// [`crate::gain_cost_halt::step_size`], ‖Δbelief‖ from a leaky-integrator
/// step, …) via [`push`](Self::push), then reads
/// [`classify`](Self::classify).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceCadence<const K: usize = 16> {
    ring: [f32; K],
    /// Next write slot (circular).
    head: usize,
    /// Samples observed so far (saturates at K).
    filled: usize,
    /// Trailing run of samples at/above `plateau_floor` (capped at K).
    trailing_high: u32,
    config: CadenceConfig,
}

impl<const K: usize> ConvergenceCadence<K> {
    const _SHAPE_GUARD: () = assert!(

        K >= 4 && K.is_multiple_of(2),

        "ConvergenceCadence window K must be even and >= 4 (two half-windows)"

    );

    /// New probe with the default [`CadenceConfig`].
    #[inline]
    pub fn new() -> Self {
        Self::with_config(CadenceConfig::default())
    }

    /// New probe with caller-calibrated thresholds.
    #[inline]
    pub fn with_config(config: CadenceConfig) -> Self {
        debug_assert!(config.settle_floor <= config.plateau_floor,
            "settle_floor must be <= plateau_floor (a window cannot be both settled-low and stuck-high)");
        debug_assert!(config.decay_ratio_max > 0.0, "decay_ratio_max must be positive");
        Self {
            ring: [0.0; K],
            head: 0,
            filled: 0,
            trailing_high: 0,
            config,
        }
    }

    /// Feed one update magnitude (caller-computed ‖Δ‖). O(1), zero-alloc.
    ///
    /// Non-finite values are pathological and count as HIGH (see the
    /// module-level non-finite policy).
    #[inline]
    pub fn push(&mut self, norm: f32) {
        let high = !norm.is_finite() || norm >= self.config.plateau_floor;
        self.trailing_high = if high {
            (self.trailing_high + 1).min(K as u32)
        } else {
            0
        };
        self.ring[self.head] = norm;
        self.head = (self.head + 1) % K;
        self.filled = (self.filled + 1).min(K);
    }

    /// Classify the window once it is full. `None` before K samples.
    ///
    /// Verdict (evaluated on the two half-window means, chronological):
    ///
    /// 1. newer mean `<= settle_floor` → `Settled` (absolute-low dominates;
    ///    law 1).
    /// 2. newer mean `>= plateau_floor` AND newer/older ratio
    ///    `>= decay_ratio_max` → `Churning` (stuck high, no decay evidence;
    ///    laws 1 + 2). The zero/tiny-older-mean degenerate is guarded: with
    ///    no older evidence the ratio reads 1.0 (no decay), so an absolute
    ///    jump out of stillness classifies on magnitude alone.
    /// 3. otherwise → `Settled` (decaying shape, or the gray band between
    ///    the floors — a non-committal window is never called Churning;
    ///    escalation should need evidence).
    ///
    /// A non-finite window mean classifies as `Churning` (pathological
    /// signal = failure signal).
    #[inline]
    pub fn classify(&self) -> Option<CadenceVerdict> {
        if self.filled < K {
            return None;
        }
        let half = K / 2;
        // Chronological order: oldest..newest. `head` is the OLDEST slot
        // once the ring is full (it was just overwritten by the newest at
        // push time... no — `head` points at the slot the NEXT push will
        // overwrite, which holds the oldest sample).
        let (mut older_sum, mut newer_sum) = (0.0_f32, 0.0_f32);
        for i in 0..K {
            let v = self.ring[(self.head + i) % K];
            if i < half {
                older_sum += v;
            } else {
                newer_sum += v;
            }
        }
        let older = older_sum / half as f32;
        let newer = newer_sum / half as f32;
        if !older.is_finite() || !newer.is_finite() {
            return Some(CadenceVerdict::Churning {
                mag: newer,
                plateau_len: self.trailing_high,
            });
        }
        if newer <= self.config.settle_floor {
            return Some(CadenceVerdict::Settled { mag: newer });
        }
        // Decay ratio with the degenerate-denominator guard (law 1: the
        // ratio is shape evidence only; absence of older evidence reads as
        // no decay).
        let ratio = if older > f32::EPSILON {
            newer / older
        } else {
            1.0
        };
        if newer >= self.config.plateau_floor && ratio >= self.config.decay_ratio_max {
            return Some(CadenceVerdict::Churning {
                mag: newer,
                plateau_len: self.trailing_high,
            });
        }
        Some(CadenceVerdict::Settled { mag: newer })
    }

    /// Trailing run of samples at/above `plateau_floor` (capped at K).
    /// Available before the window fills — useful as an early tripwire.
    #[inline]
    pub fn plateau_len(&self) -> u32 {
        self.trailing_high
    }

    /// Samples observed so far.
    #[inline]
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Clear all state (new run, same config).
    #[inline]
    pub fn reset(&mut self) {
        self.ring = [0.0; K];
        self.head = 0;
        self.filled = 0;
        self.trailing_high = 0;
    }
}

impl<const K: usize> Default for ConvergenceCadence<K> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ── Issue 731 T1 — the loop residual-exit probe ─────────────────────────

/// Residual-gated early exit for a weight-tied looped forward (Issue 731 T1;
/// EqR action item 7.2 — arXiv:2605.21488 Equilibrium Reasoners).
///
/// The probe consumes the loop's per-iteration step norm ‖h_τ − h_{τ−1}‖ and
/// decides EXIT when EITHER arm fires (never before [`LoopResidualExit::d_min`]
/// completed iterations):
///
/// 1. **magnitude arm** — the mean of the last `L = 3` step norms drops below
///    the configured `tau` (EqR's window-L residual exit);
/// 2. **shape arm** — the underlying [`ConvergenceCadence`] window classifies
///    [`CadenceVerdict::Settled`] (the Research-529 decay-shape read).
///
/// Research-440 guard (the growing-denominator trap) — **CORRECTED by Issue
/// 731 T6; the original claim was false.** What T1 claimed: "the magnitude arm
/// is ABSOLUTE and the shape arm rides alongside it, so a loop whose state
/// norm grows while its step norm plateaus can never exit spuriously, because
/// arm 2's absolute floors gate the verdict." That reasoning does not hold,
/// because **the arms are OR'd, not AND'd**: arm 2 can fire alone, and arm
/// 2's rule-3 decay fall-through (`newer/older < decay_ratio_max` ⇒
/// `Settled`) carries NO absolute floor at all — law 2 is scale-invariant by
/// design. On the K = 4 window a half-window is 2 samples, so a single 2-vs-2
/// DIP inside a high plateau reads as decay and no calibration of the
/// absolute floors can suppress it. Measured, held-out fixture seed 1003
/// (T6): the InterLoopNorm negative control fired 40 times over 8 τ × 27
/// inputs, identically from `settle_floor` 0.5 down to 1e-5.
///
/// What actually guards the trap, therefore, is **three** things, and the
/// third is the T6 fix: (a) the magnitude arm's absolute τ; (b) the shape
/// arm's absolute floors, calibrated per consumer via
/// [`Self::with_cadence_config`] — these govern rules 1 and 2 only; and (c)
/// [`Self::with_shape_persistence`] — the shape arm requires
/// [`DEFAULT_SHAPE_PERSISTENCE`] CONSECUTIVE `Settled` windows, which is what
/// distinguishes a dip from a decay and is orthogonal to calibration.
/// A magnitude-ONLY arm is the T2 negative control, expected to fail
/// calibration; it is not what this probe ships.
///
/// Zero-alloc contract (G4): fixed `[f32; 3]` + `[f32; 4]` windows plus two
/// `u32` counters, no heap.
/// Non-finite norms never satisfy arm 1 (NaN comparisons are false) and count
/// HIGH in the cadence — pathological input can delay an exit, never force
/// one.
///
/// The earliest observable exit is after **2** completed iterations via the
/// magnitude arm: the first step norm needs two loop states (τ=0 has no
/// predecessor). The shape arm cannot fire before the K = 4 window fills AND
/// [`DEFAULT_SHAPE_PERSISTENCE`] consecutive `Settled` windows accumulate.
/// Default consecutive-`Settled` windows the shape arm requires (Issue 731
/// T6 fix-forward). See [`LoopResidualExit::with_shape_persistence`] for the
/// measurement that set it.
#[cfg(feature = "cadence_gate")]
pub const DEFAULT_SHAPE_PERSISTENCE: u32 = 2;

#[cfg(feature = "cadence_gate")]
#[derive(Debug, Clone)]
pub struct LoopResidualExit {
    tau: f32,
    d_min: usize,
    /// Last L = 3 step norms (EqR's magnitude window), ring order — pre-filled
    /// with INFINITY so a partially-filled window cannot fire.
    window: [f32; 3],
    /// The shape classifier (law-2 arm).
    cadence: ConvergenceCadence<4>,
    /// Step norms observed so far (= completed iterations − 1).
    seen: usize,
    /// Completed-iteration count at the moment the probe fired.
    fired_at: Option<usize>,
    /// Consecutive `Settled` classifications the SHAPE arm requires before it
    /// may exit (Issue 731 T6 fix-forward; `1` = the pre-T6 behavior).
    shape_persistence: u32,
    /// Current run of consecutive `Settled` classifications.
    settled_run: u32,
}

#[cfg(feature = "cadence_gate")]
impl LoopResidualExit {
    /// `tau` = the absolute magnitude threshold on the 3-window mean;
    /// `d_min` = the completed-iteration floor (values < 2 clamp to 2 — the
    /// earliest observable exit). Shape arm runs on the default
    /// [`CadenceConfig`] (HRM-scale floors — Research 529's ‖Δh‖ calibration).
    #[inline]
    pub fn new(tau: f32, d_min: usize) -> Self {
        Self::with_cadence_config(tau, d_min, CadenceConfig::default())
    }

    /// Consumer-calibrated constructor (Issue 731 T5; riir-ai Issue 881
    /// fix-forward): `config` recalibrates the shape arm's ABSOLUTE floors to
    /// the consuming loop's residual scale. The default floors are the
    /// Research 529 HRM calibration point (‖Δh‖ plateaus 0.30–1.46) — on a
    /// loop whose step norms live at a different scale the defaults classify
    /// every full window `Settled` via rule 1 (`newer ≤ settle_floor`) and
    /// degenerate the exit to fire at exactly `d_min` (measured on the CCE
    /// crowd-batch loop, ‖Δρ‖₁ plateau ≈ 2e-2: riir-ai Bench 873).
    ///
    /// Calibration rule: put `plateau_floor` AT/BELOW the loop's churn
    /// plateau and `settle_floor` clearly BELOW it — a window sitting on the
    /// plateau then reads `Churning` (keep iterating), and only genuine
    /// decay through the band reaches `Settled`.
    ///
    /// **Scope (corrected by Issue 731 T6):** these floors govern
    /// [`ConvergenceCadence::classify`]'s rules 1 and 2 ONLY. Rule 3 — the
    /// decay fall-through — is scale-invariant by design (law 2) and carries
    /// no absolute threshold, so calibration alone does NOT make the
    /// Research-440 guard config-invariant: a transient 2-vs-2 dip inside a
    /// high plateau fires the shape arm under EVERY calibration (measured
    /// from `settle_floor` 0.5 down to 1e-5 on held-out fixture seed 1003).
    /// [`Self::with_shape_persistence`] is the arm that closes that gap, and
    /// it is on by default.
    #[inline]
    pub fn with_cadence_config(tau: f32, d_min: usize, config: CadenceConfig) -> Self {
        Self {
            tau,
            d_min: d_min.max(2),
            window: [f32::INFINITY; 3],
            cadence: ConvergenceCadence::with_config(config),
            seen: 0,
            fired_at: None,
            shape_persistence: DEFAULT_SHAPE_PERSISTENCE,
            settled_run: 0,
        }
    }

    /// Override the shape arm's consecutive-`Settled` requirement (Issue 731
    /// T6). `1` recovers the pre-T6 single-window behavior EXACTLY and is
    /// retained as the recorded control arm; `0` clamps to `1`.
    ///
    /// Why the default is [`DEFAULT_SHAPE_PERSISTENCE`], measured: the shape
    /// arm's rule-3 decay fall-through
    /// (`newer/older < decay_ratio_max` ⇒ `Settled`) is scale-invariant by
    /// design — law 2 carries NO absolute floor — so on a K = 4 window a
    /// single 2-vs-2 dip inside a HIGH plateau reads as decay and no
    /// calibration of the absolute floors can suppress it. Measured on
    /// held-out fixture seed 1003 (Issue 731 T6): the InterLoopNorm negative
    /// control fired 40 times over 8 τ × 27 inputs, identically at
    /// `settle_floor` 0.5 down to 1e-5, and only `decay_ratio_max ≤ 0.3`
    /// suppressed it — i.e. the Research-440 trap, on the arm the T1 record
    /// claimed was "guarded by construction". Requiring the decay to PERSIST
    /// across consecutive windows is orthogonal to calibration and is what
    /// distinguishes a dip from a decay.
    #[inline]
    #[must_use]
    pub fn with_shape_persistence(mut self, consecutive: u32) -> Self {
        self.shape_persistence = consecutive.max(1);
        self
    }

    /// Feed one iteration's step norm ‖h_τ − h_{τ−1}‖; `true` = exit now.
    /// O(1), zero-alloc. Both windows are fed on EVERY observation (the
    /// criterion is evaluated over the trailing window including pre-floor
    /// iterations); only the EXIT is floored by `d_min`.
    #[inline]
    pub fn observe(&mut self, step_norm: f32) -> bool {
        self.seen += 1;
        // Arm 1 — magnitude: mean of the last L = 3 norms < tau. The window
        // pre-fills with INFINITY so a partially-filled window cannot fire
        // (mean stays +∞ until all three slots are real).
        self.window[self.seen % 3] = step_norm;
        let mean = self.window.iter().sum::<f32>() / 3.0;
        let magnitude_exit = mean < self.tau;
        // Arm 2 — shape: the cadence verdict is Settled (decayed or
        // absolutely-low; classify needs the full K = 4 window).
        self.cadence.push(step_norm);
        let settled = matches!(self.cadence.classify(), Some(CadenceVerdict::Settled { .. }));
        // Persistence (Issue 731 T6): a `None` (window not yet full) or a
        // `Churning` breaks the run — only CONSECUTIVE Settled windows count,
        // so a transient dip inside a plateau cannot read as decay.
        self.settled_run = if settled { self.settled_run + 1 } else { 0 };
        let shape_exit = self.settled_run >= self.shape_persistence;
        // The floor: never EXIT before d_min completed iterations (the k-th
        // observation arrives after k+1 completed iterations).
        if self.seen + 1 < self.d_min {
            return false;
        }
        let exit = magnitude_exit || shape_exit;
        if exit {
            self.fired_at = Some(self.seen + 1);
        }
        exit
    }

    /// The completed-iteration floor.
    #[inline]
    pub fn d_min(&self) -> usize {
        self.d_min
    }

    /// Completed iterations when the probe fired (None = never fired).
    #[inline]
    pub fn fired_at_iteration(&self) -> Option<usize> {
        self.fired_at
    }

    /// The shape arm's consecutive-`Settled` requirement.
    #[inline]
    pub fn shape_persistence(&self) -> u32 {
        self.shape_persistence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Research 529 solved-shape: high start, decay to ~0.30 by step 8.
    fn solved_shape() -> [f32; 16] {
        [
            3.0, 2.2, 1.7, 1.3, 1.0, 0.8, 0.62, 0.48, // older half: decaying
            0.40, 0.36, 0.33, 0.31, 0.30, 0.30, 0.30, 0.30, // newer half: ~0.30
        ]
    }

    /// Research 529 failed-shape: plateau high at ~1.46.
    fn failed_shape() -> [f32; 16] {
        [1.46; 16]
    }

    #[test]
    fn decaying_window_classifies_settled() {
        let mut c = ConvergenceCadence::<16>::new();
        for n in solved_shape() {
            c.push(n);
        }
        let v = c.classify().expect("window full");
        match v {
            CadenceVerdict::Settled { mag } => {
                assert!(
                    (mag - 0.325).abs() < 1e-4,
                    "newer-half mean should be ~0.325, got {mag}"
                );
            }
            other => panic!("solved shape must classify Settled, got {other:?}"),
        }
    }

    #[test]
    fn plateau_high_classifies_churning_with_plateau_len() {
        let mut c = ConvergenceCadence::<16>::new();
        for n in failed_shape() {
            c.push(n);
        }
        let v = c.classify().expect("window full");
        match v {
            CadenceVerdict::Churning { mag, plateau_len } => {
                assert!((mag - 1.46).abs() < 1e-4, "mag should be ~1.46, got {mag}");
                assert_eq!(plateau_len, 16, "every sample at/above the floor");
            }
            other => panic!("failed shape must classify Churning, got {other:?}"),
        }
    }

    #[test]
    fn insufficient_window_returns_none() {
        let mut c = ConvergenceCadence::<16>::new();
        for i in 0..15 {
            c.push(1.0);
            assert_eq!(c.classify(), None, "classify before {i} samples");
        }
        c.push(1.0);
        assert!(c.classify().is_some(), "full window classifies");
    }

    /// Law 1: a flat TINY window is quiescent (Settled), never Churning —
    /// the shape says "plateau" but the absolute magnitude says "settled".
    #[test]
    fn absolute_low_overrides_plateau_shape() {
        let mut c = ConvergenceCadence::<16>::new();
        for _ in 0..16 {
            c.push(0.01);
        }
        let v = c.classify().expect("window full");
        assert!(
            matches!(v, CadenceVerdict::Settled { .. }),
            "flat tiny window must settle (ratio 1.0 but absolute-low), got {v:?}"
        );
    }

    /// Degenerate denominator: jump out of an all-zero older half must not
    /// produce Inf/NaN or a bogus decay classification.
    #[test]
    fn zero_older_half_is_guarded() {
        let mut c = ConvergenceCadence::<16>::new();
        for _ in 0..8 {
            c.push(0.0);
        }
        for _ in 0..8 {
            c.push(2.0);
        }
        let v = c.classify().expect("window full");
        match v {
            CadenceVerdict::Churning { mag, .. } => {
                assert_eq!(mag, 2.0, "newer-half mean");
            }
            other => panic!("jump out of stillness at high magnitude must churn, got {other:?}"),
        }
    }

    /// Non-vacuity (issue T2's gate shape, at probe level): a shuffled
    /// mix of high and low norms must NOT read Settled — escalation needs
    /// a genuinely quiet window.
    #[test]
    fn shuffled_cadence_does_not_read_settled() {
        let mut c = ConvergenceCadence::<16>::new();
        // Deterministic interleave: high, low, high, low, …
        for i in 0..16 {
            c.push(if i % 2 == 0 { 2.0 } else { 0.01 });
        }
        let v = c.classify().expect("window full");
        assert!(
            matches!(v, CadenceVerdict::Churning { .. }),
            "shuffled churn must not classify Settled, got {v:?}"
        );
        // Newer-half mean = (2.0 + 0.01) / 2 = 1.005 ≥ plateau_floor 1.0;
        // older half same ⇒ ratio 1.0 ≥ decay_ratio_max.
    }

    /// G1: same input sequence ⇒ bit-identical verdicts (pure f32 math,
    /// no RNG, no wall-clock).
    #[test]
    fn determinism_bit_identical() {
        let mut a = ConvergenceCadence::<16>::new();
        let mut b = ConvergenceCadence::<16>::new();
        for i in 0..16 {
            let n = 1.5 - 0.05 * i as f32;
            a.push(n);
            b.push(n);
        }
        assert_eq!(a.classify(), b.classify());
        // Mixed sequence too.
        let mut a2 = a;
        let mut b2 = b;
        for (i, n) in [0.9f32, 1.7, 0.2, 2.4].into_iter().enumerate() {
            a2.push(n + i as f32 * 1e-3);
            b2.push(n + i as f32 * 1e-3);
        }
        assert_eq!(a2.classify(), b2.classify());
    }

    /// G4: push + classify are alloc-free (counters via the crate's own
    /// test TrackingAllocator — see Issue 721 for why this static exists).
    /// The counters are `debug_assertions`-only by design (alloc.rs), so the
    /// test must gate to match — a `--release` test build has `cfg(test)` on
    /// but `debug_assertions` off (the debug_release_profile_axis T1 class).
    #[cfg(debug_assertions)]
    #[test]
    fn g4_alloc_free_hot_path() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        let mut c = ConvergenceCadence::<16>::new();
        // Warm + fill once outside the measurement.
        for i in 0..16 {
            c.push(1.0 + 0.1 * i as f32);
        }
        let _ = c.classify();
        reset_alloc_stats();
        for i in 0..1000 {
            c.push(1.0 + 0.001 * (i % 7) as f32);
            let _ = c.classify();
        }
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(count, 0, "push+classify must be zero-alloc, saw {count} allocs");
    }

    #[test]
    fn reset_clears_state() {
        let mut c = ConvergenceCadence::<16>::new();
        for _ in 0..16 {
            c.push(2.0);
        }
        assert!(c.classify().is_some());
        c.reset();
        assert_eq!(c.classify(), None);
        assert_eq!(c.plateau_len(), 0);
        assert_eq!(c.filled(), 0);
    }

    /// Early tripwire: plateau_len is observable before the window fills.
    #[test]
    fn plateau_len_observable_before_full_window() {
        let mut c = ConvergenceCadence::<16>::new();
        for _ in 0..5 {
            c.push(1.5);
        }
        assert_eq!(c.plateau_len(), 5);
        assert_eq!(c.classify(), None, "no verdict before a full window");
        c.push(0.1);
        assert_eq!(c.plateau_len(), 0, "a low sample breaks the run");
    }

    /// Law-3 doc pin at the API surface: the Churning payload exposes the
    /// magnitude but NOT a radial-damping recommendation — rotation-first
    /// is the caller's policy (cos_updates ≈ 0 in Research 529).
    #[test]
    fn churning_payload_is_magnitude_and_run_only() {
        let mut c = ConvergenceCadence::<8>::new();
        for _ in 0..8 {
            c.push(2.0);
        }
        let v = c.classify().expect("K=8 window full");
        assert!(matches!(
            v,
            CadenceVerdict::Churning { plateau_len: 8, .. }
        ));
    }

    /// Custom config: the same window reclassifies when the caller's
    /// floors move. A fully-flat window at 0.30 (the paper's solved
    /// endpoint) is `Settled` under the default floors (0.30 ≤ settle_floor
    /// 0.5 — absolute-low dominates) but `Churning` under tight floors
    /// (0.30 ≥ plateau_floor 0.25, ratio 1.0 = no decay evidence).
    /// Deliberate: calibration is the caller's contract.
    #[test]

    fn custom_config_changes_verdict() {

        let flat_tail = [0.30_f32; 16];

        let mut default_cfg = ConvergenceCadence::<16>::new();
        for n in flat_tail {
            default_cfg.push(n);
        }
        assert!(matches!(
            default_cfg.classify(),
            Some(CadenceVerdict::Settled { .. })
        ));

        let tight = CadenceConfig {

            plateau_floor: 0.25,

            settle_floor: 0.1,

            decay_ratio_max: 0.5,

        };

        let mut tight_cfg = ConvergenceCadence::<16>::with_config(tight);

        for n in flat_tail {
            tight_cfg.push(n);
        }
        assert!(matches!(
            tight_cfg.classify(),
            Some(CadenceVerdict::Churning { .. })
        ));
    }

    // ── Issue 731 T1 — LoopResidualExit ─────────────────────────────────

    #[cfg(feature = "cadence_gate")]
    #[test]
    fn residual_exit_floor_holds_and_magnitude_arm_fires_on_tiny_window() {
        // d_min = 4: the window is fed from the first observation, but no
        // EXIT may fire before completed iteration 4. With a trivially-low
        // tau the magnitude arm fires the moment both are satisfiable: the
        // 3rd observation (completed 4 = d_min, window just filled).
        let mut p = LoopResidualExit::new(1e-3, 4);
        assert_eq!(p.d_min(), 4);
        assert!(!p.observe(1e-6), "completed 2 < d_min");
        assert!(!p.observe(1e-6), "completed 3 < d_min");
        assert!(p.observe(1e-6), "completed 4 = d_min, window full of tiny norms → magnitude arm fires");
        assert_eq!(p.fired_at_iteration(), Some(4));
    }

    #[cfg(feature = "cadence_gate")]
    #[test]
    fn residual_exit_shape_arm_fires_on_decaying_sequence() {
        // Research-529 solved shape: decay to ~0.30 — Settled by shape even
        // with tau = 0 (magnitude arm can never fire on tau = 0; norms ≥ 0).
        // Issue 731 T6: the shape arm now requires
        // `DEFAULT_SHAPE_PERSISTENCE` CONSECUTIVE Settled windows, so a
        // genuinely-decaying sequence fires one observation later than it did
        // pre-T6 — the sequence is EXTENDED by one settled step rather than
        // the assertion loosened (the loop really is settled at 0.30).
        let mut p = LoopResidualExit::new(0.0, 2);
        let seq = [3.0, 0.4, 0.30, 0.30, 0.30];
        let fired: Vec<bool> = seq.iter().map(|n| p.observe(*n)).collect();
        assert!(
            !fired[0] && !fired[1] && !fired[2],
            "cadence window not yet full (K = 4 needs 4 pushes)"
        );
        assert!(!fired[3], "obs 4 is the FIRST Settled window — persistence 2 refuses it alone");
        assert!(fired[4], "obs 5 is the second CONSECUTIVE Settled window → shape arm fires");
        assert_eq!(p.fired_at_iteration(), Some(6));
        // The pre-T6 single-window behavior, retained as the control arm.
        let mut p1 = LoopResidualExit::new(0.0, 2).with_shape_persistence(1);
        let fired1: Vec<bool> = seq.iter().map(|n| p1.observe(*n)).collect();
        assert!(fired1[3], "persistence 1 recovers the pre-T6 first-full-window fire");
    }

    #[cfg(feature = "cadence_gate")]
    #[test]
    fn residual_exit_never_fires_on_churning_high_plateau() {
        // Failed shape: plateau HIGH — neither arm may fire (tau tiny, shape
        // classifies Churning).
        let mut p = LoopResidualExit::new(1e-3, 2);
        for i in 0..64 {
            assert!(!p.observe(1.46), "iteration {} must not exit", i + 2);
        }
        assert!(p.fired_at_iteration().is_none());
    }

    #[cfg(feature = "cadence_gate")]
    #[test]
    fn residual_exit_nonfinite_never_forces_an_exit() {
        let mut p = LoopResidualExit::new(1e-3, 2);
        for _ in 0..32 {
            assert!(!p.observe(f32::NAN), "NaN window mean → arm 1 false; cadence reads HIGH → Churning");
        }
        assert!(p.fired_at_iteration().is_none());
    }

    // ── Issue 731 T5 — the floor-calibration seam (`with_cadence_config`) ──

    #[cfg(feature = "cadence_gate")]
    #[test]
    fn with_cadence_config_default_matches_new_bit_identical() {
        // G3-style: the seam with the DEFAULT config is the same probe as
        // `new` — identical fired results and fired_at over a mixed sequence.
        let seq = [3.0, 0.4, 0.30, 0.30, 1.2, 1.4, 1.5, 0.2, 0.1, 0.1];
        let mut a = LoopResidualExit::new(0.05, 3);
        let mut b = LoopResidualExit::with_cadence_config(0.05, 3, CadenceConfig::default());
        for (i, &n) in seq.iter().enumerate() {
            assert_eq!(
                a.observe(n),
                b.observe(n),
                "observation {i}: default-config seam diverged from new",
            );
        }
        assert_eq!(a.fired_at_iteration(), b.fired_at_iteration());
    }

    #[cfg(feature = "cadence_gate")]
    #[test]
    fn calibrated_floors_refuse_low_scale_churning_plateau() {
        // The Bench 873 mechanism (riir-ai Issue 881): CCE ‖Δρ‖₁ jumps to
        // ~4e-1 then plateaus at ~2e-2. DEFAULT floors (0.5/1.0) classify
        // that window Settled via rule 1 → false-positive exit at d_min;
        // the CALIBRATED floors (settle 1e-3, plateau 1e-2) read Churning
        // and the probe never fires.
        let seq = [4e-1, 3e-2, 2e-2, 2.1e-2, 2e-2, 1.9e-2, 2e-2, 2.1e-2];
        let mut default_probe = LoopResidualExit::new(1e-4, 8);
        let fired: Vec<bool> = seq.iter().map(|&n| default_probe.observe(n)).collect();
        // The floor gate holds through completed 7; the first full window
        // (obs 7 = completed 8) is rule-1 Settled (newer ≈ 1.95e-2 ≤ 0.5)
        // → false-positive exit at d_min. (fired_at keeps overwriting on
        // every later Settled window, so assert the FIRST fire.)
        assert!(fired[..6].iter().all(|&f| !f), "floor gate: no exit before completed 8");
        assert!(
            fired[6],
            "default floors false-positive at the low-scale plateau (the 881 mechanism)"
        );
        assert!(default_probe.fired_at_iteration().is_some());
        let calibrated = CadenceConfig {
            settle_floor: 1e-3,
            plateau_floor: 1e-2,
            decay_ratio_max: 0.5,
        };
        let mut p = LoopResidualExit::with_cadence_config(1e-4, 8, calibrated);
        for (i, &n) in seq.iter().enumerate() {
            assert!(!p.observe(n), "calibrated probe must not exit at obs {}", i + 1);
        }
        // Keep iterating well past the fixture length — still Churning.
        for i in 0..64 {
            assert!(!p.observe(2e-2), "calibrated probe must not exit (churn {}) ", i);
        }
        assert!(p.fired_at_iteration().is_none());
    }

    // ── Issue 731 T6 — the shape arm's persistence requirement ──────────

    /// The T6 finding, as a unit test: a transient 2-vs-2 DIP inside a high
    /// plateau reaches `classify`'s rule-3 decay fall-through and reads
    /// `Settled` — the Research-440 trap on the arm the T1 record claimed was
    /// "guarded by construction". Both directions asserted: persistence 1
    /// (pre-T6) FIRES on the dip, the shipped default REFUSES it, and the
    /// plateau that follows never fires.
    #[cfg(feature = "cadence_gate")]
    #[test]
    fn shape_persistence_refuses_a_transient_dip_inside_a_high_plateau() {
        // Default floors (settle 0.5, plateau 1.0, decay_ratio_max 0.5).
        // Chronological windows: [2,2,2,2] Churning · [2,2,2,.8] Churning ·
        // [2,2,.8,.8] older 2.0 / newer 0.8 → rule 1 no, rule 2 no (0.8 <
        // plateau 1.0) → rule 3 SETTLED · [2,.8,.8,2] Churning again.
        let dip = [2.0_f32, 2.0, 2.0, 2.0, 0.8, 0.8, 2.0, 2.0, 2.0, 2.0];

        let mut pre_t6 = LoopResidualExit::new(0.0, 2).with_shape_persistence(1);
        let fired: Vec<bool> = dip.iter().map(|&n| pre_t6.observe(n)).collect();
        assert!(
            fired[5],
            "the pre-T6 single-window shape arm false-positives on the dip — this IS the T6 defect"
        );

        let mut shipped = LoopResidualExit::new(0.0, 2);
        for (i, &n) in dip.iter().enumerate() {
            assert!(
                !shipped.observe(n),
                "persistence {} must refuse the dip at obs {}",
                DEFAULT_SHAPE_PERSISTENCE,
                i + 1
            );
        }
        assert_eq!(shipped.fired_at_iteration(), None);
    }

    /// The dip is floor-INVARIANT: rule 3 carries no absolute threshold, so
    /// recalibrating the T5 seam's floors arbitrarily low does NOT suppress
    /// it — only persistence (or `decay_ratio_max`) does. This is why the T5
    /// calibration rule alone was insufficient (measured on held-out fixture
    /// seed 1003: identical fires from settle_floor 0.5 down to 1e-5).
    #[cfg(feature = "cadence_gate")]
    #[test]
    fn a_dip_defeats_every_absolute_floor_calibration() {
        let dip = [2.0_f32, 2.0, 2.0, 2.0, 0.8, 0.8];
        for settle in [0.5_f32, 0.05, 1e-3, 1e-5] {
            let cfg = CadenceConfig {
                settle_floor: settle,
                plateau_floor: settle * 2.0,
                decay_ratio_max: 0.5,
            };
            let mut p = LoopResidualExit::with_cadence_config(0.0, 2, cfg)
                .with_shape_persistence(1);
            let fired: Vec<bool> = dip.iter().map(|&n| p.observe(n)).collect();
            assert!(
                fired[5],
                "settle_floor {settle}: the dip still reaches rule 3 — absolute floors cannot gate a ratio"
            );
            // And the shipped persistence refuses it at every calibration.
            let mut shipped = LoopResidualExit::with_cadence_config(0.0, 2, cfg);
            assert!(
                dip.iter().all(|&n| !shipped.observe(n)),
                "settle_floor {settle}: persistence must refuse the dip under every calibration"
            );
        }
    }

    /// `with_shape_persistence(1)` is the recorded control arm and `0` clamps
    /// to it; the default constructors ship `DEFAULT_SHAPE_PERSISTENCE`.
    #[cfg(feature = "cadence_gate")]
    #[test]
    fn shape_persistence_accessor_and_clamp() {
        assert_eq!(
            LoopResidualExit::new(1.0, 4).shape_persistence(),
            DEFAULT_SHAPE_PERSISTENCE
        );
        assert_eq!(
            LoopResidualExit::with_cadence_config(1.0, 4, CadenceConfig::default())
                .shape_persistence(),
            DEFAULT_SHAPE_PERSISTENCE
        );
        assert_eq!(
            LoopResidualExit::new(1.0, 4).with_shape_persistence(0).shape_persistence(),
            1,
            "0 clamps to 1 — a shape arm that can never fire is not a config, it is a bug"
        );
        assert_eq!(
            LoopResidualExit::new(1.0, 4).with_shape_persistence(5).shape_persistence(),
            5
        );
    }

    /// Persistence must not weaken the MAGNITUDE arm: an absolute L = 3
    /// window mean below τ still exits at `d_min` with no shape evidence.
    #[cfg(feature = "cadence_gate")]
    #[test]
    fn persistence_does_not_gate_the_magnitude_arm() {
        let mut p = LoopResidualExit::new(1e-3, 4);
        assert!(!p.observe(1e-6));
        assert!(!p.observe(1e-6));
        assert!(p.observe(1e-6), "magnitude arm is not subject to shape persistence");
        assert_eq!(p.fired_at_iteration(), Some(4));
    }

    #[cfg(feature = "cadence_gate")]
    #[test]
    fn calibrated_floors_admit_high_scale_settling() {
        // The mirror calibration class: a loop whose settled plateau sits at
        // norms FAR ABOVE the HRM band (converged, flat, residual ≈ 8).
        // DEFAULT floors (plateau 1.0) classify the flat window Churning
        // forever; calibrated floors (settle 5, plateau 15) put the plateau
        // mid-band → the shipped fall-through reads Settled and the probe
        // fires. (The decay-ratio arm is scale-invariant by design — law 2 —
        // so DECAYING high-scale windows settle under both configs; the flat
        // plateau is the discriminating shape.)
        let mut default_probe = LoopResidualExit::new(0.0, 4);
        for i in 0..20 {
            default_probe.observe(8.0);
            assert_eq!(
                default_probe.fired_at_iteration(),
                None,
                "default floors must keep refusing a flat high-scale plateau (Churning) at obs {i}"
            );
        }
        let calibrated = CadenceConfig {
            settle_floor: 5.0,
            plateau_floor: 15.0,
            decay_ratio_max: 0.5,
        };
        // Issue 731 T6: extended by one settled step — the shape arm needs
        // two CONSECUTIVE Settled windows, so the fire moves from the 4th to
        // the 5th observation (completed 5 → 6). The plateau is genuinely
        // flat, so the second window is Settled too.
        let seq = [40.0, 20.0, 8.0, 8.0, 8.0];
        let mut p = LoopResidualExit::with_cadence_config(0.0, 4, calibrated);
        let fired: Vec<bool> = seq.iter().map(|&n| p.observe(n)).collect();
        assert!(!fired[3], "the FIRST Settled window alone does not satisfy persistence 2");
        assert!(fired[4], "calibrated shape arm fires on the second consecutive Settled window");
        // fired_at records COMPLETED iterations (seen + 1): the 5th
        // observation arrives after 6 completed iterations.
        assert_eq!(p.fired_at_iteration(), Some(6));
    }
}
