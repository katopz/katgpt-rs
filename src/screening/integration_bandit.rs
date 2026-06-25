//! Bandit arm-selection integration for the algorithmic-probability sampler
//! (Plan 305 T3.2). Biases arm selection by `sigmoid(-α·K̃(arm) - β)`.
//! Zero-cost when `bandit_k_prior` is off.
//!
//! # Why adapter-only (not a new `BanditStrategy` variant)
//!
//! `katgpt-rs/src/pruners/bandit.rs` has a `BanditStrategy` enum with 10+
//! variants (Ucb1, EpsilonGreedy, ThompsonSampling, VarianceEpsilon,
//! Rpucg, RandOptAdaptive, SafePhased, CurvatureInfluence, ...) each with
//! its own match arm in `BanditPruner::arm_bandit_score`,
//! `BanditSession::select_arm`, etc. Adding a `KPrior` variant would
//! require new arms in every match — an invasive change to a 2057-line file.
//!
//! Instead, this module ships a `KPriorBandit<K>` wrapper that provides a
//! per-arm log-prior. The caller (their UCB1 / Thompson / ε-greedy policy)
//! adds this log-prior to their existing arm score:
//!
//! ```text
//!   arm_score_with_prior(a) = arm_score_base(a) + arm_log_prior(a)
//! ```
//!
//! The wrapper is decoupled from the bandit flavour. It does NOT implement
//! a bandit policy itself — it just provides the prior. This keeps the
//! trait composable with every strategy in `bandit.rs` without touching
//! the enum.
//!
//! # RNG note
//!
//! `CompressionPriorSampler::sample_ix` uses `fastrand::Rng` (same RNG as
//! the existing bandit code via `crate::types::Rng`). No new `rand` dep is
//! introduced. This module does not call `sample_ix` — it only exposes
//! `arm_log_prior` (deterministic, no RNG), leaving the categorical draw
//! to the caller's policy.

use crate::screening::complexity_prior::{ComplexityProxy, CompressionPriorSampler};

/// Bandit prior that biases arm selection toward low-K arms.
///
/// Wraps a `CompressionPriorSampler<K>` and produces per-arm log-priors that
/// the caller's bandit policy (UCB, Thompson, ε-greedy, etc.) can add to its
/// arm scores. The wrapper itself does NOT implement a bandit policy — it
/// just provides the prior. This keeps the trait decoupled from the many
/// bandit flavours in `katgpt-rs/src/pruners/bandit.rs`.
///
/// # Usage
///
/// ```rust,ignore
/// let prior = KPriorBandit::new(CompressionPriorSampler::default());
/// // In the caller's arm-scoring loop:
/// for arm in 0..num_arms {
///     let arm_bytes: &[u8] = encode_arm(arm);
///     arm_score[arm] += prior.arm_log_prior(arm_bytes);
/// }
/// ```
///
/// Lower-K arms get higher log-priors, so the policy pulls them more often.
/// This is the algorithmic-probability prior (Research 284 / Plan 305).
#[derive(Debug, Clone, Copy)]
pub struct KPriorBandit<K: ComplexityProxy> {
    sampler: CompressionPriorSampler<K>,
}

impl<K: ComplexityProxy> KPriorBandit<K> {
    /// Construct from a sampler.
    #[inline]
    #[must_use]
    pub const fn new(sampler: CompressionPriorSampler<K>) -> Self {
        Self { sampler }
    }

    /// Borrow the inner sampler (for `sample_ix` / `top_k` access).
    #[inline]
    pub const fn sampler(&self) -> &CompressionPriorSampler<K> {
        &self.sampler
    }

    /// Per-arm log-prior. Caller adds this to the bandit's arm score.
    ///
    /// `arm_bytes` is the byte encoding of the arm (caller's responsibility —
    /// the wrapper is agnostic to how arms are encoded). Returns
    /// `-α·K̃(arm_bytes) - β`.
    ///
    /// Zero allocation, branch-free. `#[inline]` so the caller's arm-scoring
    /// loop fuses with the bandit's own score computation.
    #[inline]
    pub fn arm_log_prior(&self, arm_bytes: &[u8]) -> f32 {
        self.sampler.log_prob(arm_bytes)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::complexity_prior::RleComplexity;

    #[test]
    fn arm_log_prior_delegates_to_sampler() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let bandit = KPriorBandit::new(sampler);
        let arm_bytes = [10u8, 20, 30, 40];
        assert_eq!(
            bandit.arm_log_prior(&arm_bytes),
            bandit.sampler().log_prob(&arm_bytes),
            "arm_log_prior must equal sampler.log_prob(arm_bytes)"
        );
    }

    #[test]
    fn low_k_arm_preferred_over_high_k() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let bandit = KPriorBandit::new(sampler);
        // Low-K arm: all-zero bytes → RLE K̃ ≈ 0.031.
        let low_k_arm = [0u8; 64];
        // High-K arm: alternating → RLE K̃ = 2.0.
        let mut high_k_arm = [0u8; 64];
        for (i, b) in high_k_arm.iter_mut().enumerate() {
            *b = if i % 2 == 0 { 255 } else { 0 };
        }
        let lp_low = bandit.arm_log_prior(&low_k_arm);
        let lp_high = bandit.arm_log_prior(&high_k_arm);
        assert!(
            lp_low > lp_high,
            "low-K arm should have higher prior: lp_low={lp_low}, lp_high={lp_high}"
        );
    }

    #[test]
    fn sampler_accessor_returns_same_ref() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let bandit = KPriorBandit::new(sampler);
        // The accessor must return a reference to the wrapped sampler —
        // calling log_prob through it must match the direct call.
        let arm_bytes = [1u8, 2, 3];
        assert_eq!(
            bandit.sampler().log_prob(&arm_bytes),
            bandit.arm_log_prior(&arm_bytes),
            "sampler() accessor must return the same sampler used internally"
        );
        // Sanity: alpha/beta from the wrapped sampler are visible.
        assert_eq!(bandit.sampler().alpha, 1.0);
        assert_eq!(bandit.sampler().beta, 0.0);
    }
}
