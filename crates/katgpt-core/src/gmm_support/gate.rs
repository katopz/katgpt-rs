//! The two-density support gate + its EMA smoother (Plan 618 T4/T5).
//!
//! `ℓ(x) = log Φ_pos(x) − log Φ_neg(x)` over a diagonal-GMM pair (the
//! positive density fit on the suite/adapter's own training distribution,
//! the negative on a generic reference corpus — capturing distribution
//! WIDTH, not identity). Adapter/answer applies iff `ℓ(x) > 0`; the
//! confidence bridge is `sigmoid(ℓ/τ)` — **sigmoid, never softmax** (the
//! house law; a two-density ratio is one scalar, not a categorical).
//!
//! # Unfitted = closed-always
//!
//! The [`CorpusDistanceGate`](crate::distance_abstain::CorpusDistanceGate)
//! empty-corpus precedent: a gate with no densities has no coverage, and
//! `log_ratio` reads `−∞` (every query closed, every confidence 0) rather
//! than refusing — callers gate on [`is_fitted`](SupportGate::is_fitted)
//! when they need to distinguish.

use super::mixture::DiagGmm;

/// Errors returned by [`SupportGate::new`] and [`GateSmoother::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GateError {
    /// `tau <= 0` or non-finite (the confidence bridge scale).
    InvalidTau,
    /// `alpha` outside `(0, 1]` or non-finite (the EMA fresh-signal
    /// weight).
    InvalidAlpha,
}

/// Two-density support gate over projected points `R^E`.
///
/// `mid`/scale knobs are consumer-pinned; `DEFAULT_TAU` is POC-scale (the
/// `contrastive_scope` precedent — named as such, never a universal
/// optimum).
#[derive(Debug, Clone)]
pub struct SupportGate<const E: usize, const K: usize> {
    pos: Option<DiagGmm<E, K>>,
    neg: Option<DiagGmm<E, K>>,
    /// Confidence bridge scale `τ > 0` (sigmoid temperature).
    tau: f32,
}

/// POC-scale default sigmoid temperature (consumers re-pin).
pub const DEFAULT_TAU: f32 = 1.0;

impl<const E: usize, const K: usize> SupportGate<E, K> {
    /// Build a fitted gate from a density pair. Both mixtures must already
    /// be valid (they are, by `DiagGmm` construction).
    pub fn new(pos: DiagGmm<E, K>, neg: DiagGmm<E, K>, tau: f32) -> Result<Self, GateError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(GateError::InvalidTau);
        }
        Ok(Self {
            pos: Some(pos),
            neg: Some(neg),
            tau,
        })
    }

    /// The closed-always gate (the empty-corpus precedent). Every
    /// `log_ratio` reads `−∞`; `open` is `false`; `confidence` is `0`.
    pub fn unfitted(tau: f32) -> Result<Self, GateError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(GateError::InvalidTau);
        }
        Ok(Self {
            pos: None,
            neg: None,
            tau,
        })
    }

    /// `true` iff both densities are present.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.pos.is_some() && self.neg.is_some()
    }

    /// The two-density log-ratio `ℓ(x)`, or `−∞` when unfitted.
    ///
    /// If exactly one density is present (a consumer mid-wiring), the
    /// gate still reads closed — a ratio needs both sides.
    #[inline]
    #[must_use]
    pub fn log_ratio(&self, x: &[f32; E]) -> f32 {
        match (&self.pos, &self.neg) {
            (Some(pos), Some(neg)) => pos.loglik_mixture(x) - neg.loglik_mixture(x),
            _ => f32::NEG_INFINITY,
        }
    }

    /// The hard admission predicate: `ℓ(x) > 0` (the paper's gate; a
    /// fixed threshold on the ratio — the `contrastive_scope` shape).
    #[inline]
    #[must_use]
    pub fn open(&self, x: &[f32; E]) -> bool {
        self.log_ratio(x) > 0.0
    }

    /// Confidence bridge `sigmoid(ℓ(x) / τ)` — the scalar a calibrated
    /// consumer composes. Unfitted reads 0 (never NaN).
    #[inline]
    #[must_use]
    pub fn confidence(&self, x: &[f32; E]) -> f32 {
        crate::exact_sigmoid(self.log_ratio(x) / self.tau)
    }

    /// The positive density (the fit on the gate's own training
    /// distribution), when present.
    #[must_use]
    pub fn pos(&self) -> Option<&DiagGmm<E, K>> {
        self.pos.as_ref()
    }

    /// The reference density (shared per shape class at the consumer's
    /// discretion — the paper's `-43% gate count` mechanism), when
    /// present.
    #[must_use]
    pub fn neg(&self) -> Option<&DiagGmm<E, K>> {
        self.neg.as_ref()
    }

    /// The sigmoid temperature in use.
    #[must_use]
    pub fn tau(&self) -> f32 {
        self.tau
    }
}

/// Per-stream EMA smoother over the gate signal (Plan 618 T5):
/// `s ← α·g + (1−α)·s`, deciding at 0.5.
///
/// Plain runtime state — never synced, never a weight, never committed
/// (per-stream latent state, the `FkStepper` weights-only precedent for
/// what may NOT be resampled/reconstructed).
#[derive(Debug, Clone, Copy)]
pub struct GateSmoother {
    alpha: f32,
    state: f32,
}

impl GateSmoother {
    /// `alpha ∈ (0, 1]` is the fresh-signal weight; `state` starts at the
    /// neutral 0.5 (no evidence either way).
    ///
    /// # Errors
    ///
    /// [`GateError::InvalidAlpha`] if `alpha` is outside `(0, 1]`.
    pub fn new(alpha: f32) -> Result<Self, GateError> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err(GateError::InvalidAlpha);
        }
        Ok(Self { alpha, state: 0.5 })
    }

    /// Push a gate signal `g ∈ [0, 1]` (the binary `1[ℓ>0]` or the
    /// confidence scalar) and return the smoothed state.
    #[inline]
    pub fn update(&mut self, g: f32) -> f32 {
        self.state = self.alpha * g + (1.0 - self.alpha) * self.state;
        self.state
    }

    /// The smoothed state.
    #[inline]
    #[must_use]
    pub fn state(&self) -> f32 {
        self.state
    }

    /// The smoothed decision at the default 0.5 threshold.
    #[inline]
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.state > 0.5
    }

    /// The smoothed decision at a caller threshold.
    #[inline]
    #[must_use]
    pub fn decides(&self, threshold: f32) -> bool {
        self.state > threshold
    }
}
