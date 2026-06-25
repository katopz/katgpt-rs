//! MCTS expansion-prior integration for the algorithmic-probability sampler
//! (Plan 305 T3.1). Biases child expansion by `sigmoid(-α·K̃(child) - β)`
//! instead of uniform. Zero-cost when `mcts_k_prior` is off.
//!
//! # Why adapter-only (not direct wiring into `mcts.rs`)
//!
//! The MCTS at `katgpt-rs/src/pruners/game_state/mcts.rs` is generic over
//! `S: GameState` with an opaque `S::Action` type, and `expand_and_rollout`
//! picks a uniform-random unexpanded action via:
//!
//! ```rust,ignore
//! let pick = rng.usize(0..node.unexpanded.len());
//! let action_idx = node.unexpanded.swap_remove(pick);
//! ```
//!
//! To wire in a K-prior here would require (a) threading a `CompressionPriorSampler<K>`
//! through `mcts_search_impl` / `select_inline` / `expand_and_rollout` as a new
//! generic parameter, (b) defining a game-agnostic `Action -> &[u8]` encoding
//! hook on `GameState`, and (c) gating the whole thing behind `mcts_k_prior`
//! without changing the existing function signatures. That's an invasive
//! refactor of a 1044-line hot-path file.
//!
//! Per the plan, this module ships the trait + impls as an **adapter seam** —
//! a caller wires it in by (1) encoding each unexpanded action as a byte
//! slice, (2) calling `CompressionPriorSampler::sample_ix(&candidates, scratch, rng)`
//! to get the K-prior-weighted pick, (3) using that index as the new `pick`.
//! The trait documents the contract; the caller provides the encoding.
//!
//! # Safety guarantee
//!
//! Because the sampler is `sigmoid(-α·K̃ - β)`-weighted with `α ≥ 0`, it is
//! **never worse than uniform** on the expected rank of sampled candidates
//! (Plan 305 Phase 1 `test_sampler_never_worse_than_uniform`, Research 284).
//! On games whose optimum is low-K (simple tactics), the sampler reaches it
//! exponentially faster (Levin-search variant).

use crate::screening::complexity_prior::{ComplexityProxy, CompressionPriorSampler};

/// Expansion prior applied to MCTS child candidates.
///
/// Given a `CompressionPriorSampler<K>` and a candidate's byte encoding, returns
/// an unnormalised log-probability. Higher = preferred.
///
/// - [`UniformExpansion`] returns `0.0` for all candidates (no preference —
///   byte-identical to pre-Plan-305 behaviour).
/// - [`KPriorExpansion`] returns `sampler.log_prob(candidate)`, biasing tree
///   growth toward low-K children per the algorithmic-probability prior
///   (Research 284 / Plan 305).
///
/// # Integration pattern
///
/// The caller's MCTS expansion loop:
///
/// ```rust,ignore
/// // Before (uniform):
/// let pick = rng.usize(0..node.unexpanded.len());
///
/// // After (K-prior, behind `mcts_k_prior`):
/// // 1. Build byte encodings of each unexpanded action (game-specific).
/// let candidates: Vec<&[u8]> = node.unexpanded.iter()
///     .map(|&a| encode_action(&leaf_actions[a]).as_slice())
///     .collect();
/// let mut scratch = vec![0.0; candidates.len()];
/// let pick = sampler.sample_ix(&candidates, &mut scratch, rng);
/// ```
///
/// `sample_ix` is the recommended entry point — it folds the prior into a
/// categorical draw in zero allocations. The trait's `log_prior` is exposed
/// for callers who want to inspect / mix the prior with other signals.
pub trait MctsExpansionPrior {
    /// Return an unnormalised log-probability for the candidate. Higher =
    /// preferred. Uniform returns `0.0` for all candidates.
    fn log_prior<K: ComplexityProxy>(
        &self,
        sampler: &CompressionPriorSampler<K>,
        candidate: &[u8],
    ) -> f32;
}

/// Uniform expansion — no preference (byte-identical to pre-Plan-305 behaviour).
///
/// Returning a constant `0.0` log-prior makes every candidate equally likely
/// after `sample_ix` normalises, exactly reproducing the uniform-random pick
/// the MCTS code used before Plan 305.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformExpansion;

impl MctsExpansionPrior for UniformExpansion {
    #[inline]
    fn log_prior<K: ComplexityProxy>(
        &self,
        _sampler: &CompressionPriorSampler<K>,
        _candidate: &[u8],
    ) -> f32 {
        0.0
    }
}

/// K-prior expansion — biases toward low-complexity children.
///
/// Delegates to `CompressionPriorSampler::log_prob`, returning
/// `-α·K̃(candidate) - β`. Lower-K candidates get higher (less negative)
/// log-priors, so `sample_ix` draws them more often.
#[derive(Debug, Clone, Copy, Default)]
pub struct KPriorExpansion;

impl MctsExpansionPrior for KPriorExpansion {
    #[inline]
    fn log_prior<K: ComplexityProxy>(
        &self,
        sampler: &CompressionPriorSampler<K>,
        candidate: &[u8],
    ) -> f32 {
        sampler.log_prob(candidate)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::complexity_prior::RleComplexity;

    #[test]
    fn uniform_returns_zero_for_any_candidate() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let prior = UniformExpansion;
        // Low-K: all-zero bytes (highly RLE-compressible).
        let low_k = [0u8; 64];
        // High-K: alternating bytes (8 runs of length 1 → K̃ ≈ 2.0).
        let mut high_k = [0u8; 64];
        for (i, b) in high_k.iter_mut().enumerate() {
            *b = if i % 2 == 0 { 255 } else { 0 };
        }
        assert_eq!(prior.log_prior(&sampler, &low_k), 0.0);
        assert_eq!(prior.log_prior(&sampler, &high_k), 0.0);
    }

    #[test]
    fn k_prior_returns_higher_for_low_k() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let prior = KPriorExpansion;
        // Low-K: all-zero → RLE K̃ = 2/64 ≈ 0.031.
        let low_k = [0u8; 64];
        // High-K: alternating → RLE K̃ = 128/64 = 2.0.
        let mut high_k = [0u8; 64];
        for (i, b) in high_k.iter_mut().enumerate() {
            *b = if i % 2 == 0 { 255 } else { 0 };
        }
        let lp_low = prior.log_prior(&sampler, &low_k);
        let lp_high = prior.log_prior(&sampler, &high_k);
        assert!(
            lp_low > lp_high,
            "low-K candidate should have higher log_prior: lp_low={lp_low}, lp_high={lp_high}"
        );
    }

    #[test]
    fn k_prior_log_prob_delegates_to_sampler() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let prior = KPriorExpansion;
        let candidate = [1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            prior.log_prior(&sampler, &candidate),
            sampler.log_prob(&candidate),
            "KPriorExpansion must return exactly sampler.log_prob(candidate)"
        );
    }
}
