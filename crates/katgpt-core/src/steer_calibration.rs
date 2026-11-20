//! steer_calibration — closed-form steering-strength calibration (Issue 1235
//! row (a); Research 683, arXiv:2602.22600 App. D, the ACE generation
//! intervention controller transferred to a DECISION margin).
//!
//! The steering consumers ([`crate::latent_steering`] Plan 309,
//! [`crate::subspace_steering`] Plan 412, `region_subspace` Plan 416) apply a
//! pre-discovered direction with a HAND-SET strength α. A fixed α overshoots
//! states that were already near the decision boundary and undershoots states
//! far from it. This module replaces the hand α with a three-step controller:
//!
//! ```text
//!   gate   : relevance_mass ≥ mass_threshold  and  m0 < m_target   (else skip, s = 0)
//!   probe  : g  = (m(s + s0·u) − m0) / s0                            (one local-gain probe)
//!   steer  : s* = clamp((m_target − m0) / g, 0, s_max)                (minimal margin crossing)
//! ```
//!
//! `m(·)` is the CONSUMER's decision margin (e.g. top-1 minus runner-up score
//! of a scored decision) — passed in as a closure, so "the consumer names its
//! margin" is a call-site argument, never baked here. `m_target`,
//! `mass_threshold`, `s0` and `s_max` are consumer configuration too: this
//! module owns no magic thresholds.
//!
//! # Honest scope
//!
//! - One probe = a LINEAR model of the margin along `u`. On a linear margin
//!   `s*` lands exactly on `m_target`; on a curved margin the residual is the
//!   secant error at `s0` (Bench gate `bench_1235_steer_calibration_goat`
//!   measures it, never assumes it).
//! - A direction that does not RAISE the margin (`g ≤ min_gain`) refuses
//!   loud ([`SteerCalibError::NoGain`]) — never a silent zero or a negative
//!   strength.
//! - [`SubspaceSteeringField`](crate::subspace_steering::SubspaceSteeringField)
//!   requires `α ∈ [0, 1]`; pick `s_max ≤ 1` when building one from `s*`.
//!
//! # Determinism + allocation
//!
//! The probe state is built with the same `mul_add` SAXPY as
//! `SubspaceSteeringField::apply` (K=1 parity), on a stack copy. Zero heap
//! allocations; the only non-determinism is whatever the caller's margin
//! closure carries.

/// Consumer configuration for [`calibrate_strength`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteerCalibConfig {
    /// Probe strength `s0 > 0` for the local-gain estimate.
    pub probe: f32,
    /// The margin the steer aims to reach (the minimal crossing).
    pub m_target: f32,
    /// Strength cap `s_max > 0` (the safety bound; ≤ 1 for a steering field).
    pub s_max: f32,
    /// Smallest gain accepted as "the direction raises the margin" (`≥ 0`).
    pub min_gain: f32,
    /// Task-relevance mass gate: intervene only at or above this mass.
    pub mass_threshold: f32,
}

/// Why the controller chose not to intervene (`s = 0`, state untouched).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SkipReason {
    /// `relevance_mass < mass_threshold` — the decision is not task-relevant.
    BelowMass,
    /// `m0 ≥ m_target` — the margin already crosses; steering would overshoot.
    AlreadyAtTarget,
}

/// The controller's decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SteerDecision {
    /// No intervention.
    Skip { reason: SkipReason, m0: f32 },
    /// Intervene with strength `s_star` along the direction.
    Steer {
        /// The calibrated strength, in `[0, s_max]`.
        s_star: f32,
        /// Margin before steering.
        m0: f32,
        /// Probed local gain `dm/ds`.
        gain: f32,
        /// `true` when `s_max` clipped the closed-form strength (the target
        /// will NOT be reached — disclosed, never hidden).
        capped: bool,
    },
}

impl SteerDecision {
    /// The strength to apply (`0.0` on a skip).
    #[inline]
    #[must_use]
    pub fn strength(&self) -> f32 {
        match *self {
            Self::Skip { .. } => 0.0,
            Self::Steer { s_star, .. } => s_star,
        }
    }
}

/// Refuse-loud errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SteerCalibError {
    /// A config field is non-finite or out of range (`probe ≤ 0`, `s_max ≤ 0`,
    /// `min_gain < 0`).
    BadConfig,
    /// The state, direction, mass or a margin read is non-finite.
    NonFinite,
    /// The direction does not raise the margin (`gain ≤ min_gain`).
    NoGain { gain: f32 },
}

impl SteerCalibConfig {
    fn validate(&self) -> Result<(), SteerCalibError> {
        let finite = self.probe.is_finite()
            && self.m_target.is_finite()
            && self.s_max.is_finite()
            && self.min_gain.is_finite()
            && self.mass_threshold.is_finite();
        if !finite || self.probe <= 0.0 || self.s_max <= 0.0 || self.min_gain < 0.0 {
            return Err(SteerCalibError::BadConfig);
        }
        Ok(())
    }
}

/// `out = state + s·dir` with the `SubspaceSteeringField::apply` SAXPY
/// (`mul_add`) — K=1 bit parity with the field the consumer applies.
#[inline]
#[must_use]
pub fn steered<const D: usize>(state: &[f32; D], dir: &[f32; D], s: f32) -> [f32; D] {
    let mut out = *state;
    for j in 0..D {
        out[j] = s.mul_add(dir[j], out[j]);
    }
    out
}

/// Calibrate the steering strength along `dir` for one decision state.
///
/// `relevance_mass` is the consumer's task-mass read (e.g. a block-energy
/// share); `margin` is the consumer's decision margin. Calls `margin` at most
/// twice (`m0`, then the probe). Zero-alloc.
pub fn calibrate_strength<const D: usize>(
    state: &[f32; D],
    dir: &[f32; D],
    relevance_mass: f32,
    mut margin: impl FnMut(&[f32; D]) -> f32,
    cfg: &SteerCalibConfig,
) -> Result<SteerDecision, SteerCalibError> {
    cfg.validate()?;
    if !relevance_mass.is_finite()
        || state.iter().any(|v| !v.is_finite())
        || dir.iter().any(|v| !v.is_finite())
    {
        return Err(SteerCalibError::NonFinite);
    }
    let m0 = margin(state);
    if !m0.is_finite() {
        return Err(SteerCalibError::NonFinite);
    }
    if relevance_mass < cfg.mass_threshold {
        return Ok(SteerDecision::Skip {
            reason: SkipReason::BelowMass,
            m0,
        });
    }
    if m0 >= cfg.m_target {
        return Ok(SteerDecision::Skip {
            reason: SkipReason::AlreadyAtTarget,
            m0,
        });
    }
    let m1 = margin(&steered(state, dir, cfg.probe));
    if !m1.is_finite() {
        return Err(SteerCalibError::NonFinite);
    }
    let gain = (m1 - m0) / cfg.probe;
    if !gain.is_finite() || gain <= cfg.min_gain {
        return Err(SteerCalibError::NoGain { gain });
    }
    let raw = (cfg.m_target - m0) / gain;
    let capped = raw > cfg.s_max;
    Ok(SteerDecision::Steer {
        s_star: raw.clamp(0.0, cfg.s_max),
        m0,
        gain,
        capped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: SteerCalibConfig = SteerCalibConfig {
        probe: 0.05,
        m_target: 0.2,
        s_max: 1.0,
        min_gain: 1e-6,
        mass_threshold: 0.1,
    };

    fn lin(s: &[f32; 4]) -> f32 {
        2.0 * s[0] - 0.5
    }

    #[test]
    fn linear_margin_lands_on_target() {
        let dir = [1.0, 0.0, 0.0, 0.0];
        let state = [0.0; 4];
        let d = calibrate_strength(&state, &dir, 1.0, lin, &CFG).unwrap();
        let s = d.strength();
        assert!((s - 0.35).abs() < 1e-5, "{d:?}");
        assert!((lin(&steered(&state, &dir, s)) - 0.2).abs() < 1e-5);
    }

    #[test]
    fn gates_skip() {
        let dir = [1.0, 0.0, 0.0, 0.0];
        let d = calibrate_strength(&[0.0; 4], &dir, 0.0, lin, &CFG).unwrap();
        assert!(matches!(
            d,
            SteerDecision::Skip {
                reason: SkipReason::BelowMass,
                ..
            }
        ));
        let d = calibrate_strength(&[1.0, 0.0, 0.0, 0.0], &dir, 1.0, lin, &CFG).unwrap();
        assert!(matches!(
            d,
            SteerDecision::Skip {
                reason: SkipReason::AlreadyAtTarget,
                ..
            }
        ));
        assert_eq!(d.strength(), 0.0);
    }

    #[test]
    fn cap_is_disclosed() {
        let dir = [1.0, 0.0, 0.0, 0.0];
        let d = calibrate_strength(&[-5.0, 0.0, 0.0, 0.0], &dir, 1.0, lin, &CFG).unwrap();
        assert!(matches!(d, SteerDecision::Steer { capped: true, s_star, .. } if s_star == 1.0));
    }

    #[test]
    fn refuses_no_gain_and_bad_input() {
        let ortho = [0.0, 1.0, 0.0, 0.0];
        assert!(matches!(
            calibrate_strength(&[0.0; 4], &ortho, 1.0, lin, &CFG),
            Err(SteerCalibError::NoGain { .. })
        ));
        let neg = [-1.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            calibrate_strength(&[0.0; 4], &neg, 1.0, lin, &CFG),
            Err(SteerCalibError::NoGain { .. })
        ));
        let bad = SteerCalibConfig { probe: 0.0, ..CFG };
        assert_eq!(
            calibrate_strength(&[0.0; 4], &neg, 1.0, lin, &bad),
            Err(SteerCalibError::BadConfig)
        );
        assert_eq!(
            calibrate_strength(&[f32::NAN, 0.0, 0.0, 0.0], &neg, 1.0, lin, &CFG),
            Err(SteerCalibError::NonFinite)
        );
    }
}
