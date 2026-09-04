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
}
