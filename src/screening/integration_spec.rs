//! Speculative drafter integration for the algorithmic-probability sampler
//! (Plan 305 T3.3). Re-ranks draft tokens by K-prior. Zero-cost when
//! `spec_k_prior` is off.
//!
//! # Why post-drafting re-ranker (not a drafter wrapper)
//!
//! `katgpt-rs/src/speculative/` and `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs`
//! (the R256 cousin) define many drafter flavours: NF-Flow, Domino, DFlash,
//! Echo, Compression, Dendritic, ... each with its own trait surface. Defining
//! a single wrapping trait would either (a) over-constrain the drafter API or
//! (b) require a new generic parameter on every drafter. Too invasive.
//!
//! Instead, this module ships `KPriorDrafter<K>` as a **post-drafting
//! re-ranker**: the caller's existing drafter produces a top-K list of
//! (token, score) pairs, then this hook adds `log_prior` to each score (in
//! log-space, equivalent to multiplying by `exp(log_prior)`) and the caller
//! re-sorts. Low-K drafts are boosted; high-K drafts are demoted.
//!
//! Composes cleanly with `CompressionDrafter` (R256) and `DendriticGate`
//! (R260): both produce `(token, score)` pairs the caller can pass to
//! [`KPriorDrafter::rerank`].
//!
//! # Zero allocation
//!
//! `rerank` takes caller-provided `scores: &mut [f32]` and `scratch: &mut [f32]`
//! buffers. The function does NOT allocate — it writes log-priors into
//! `scratch` and adds them to `scores` in place. The caller is responsible
//! for sorting `scores` after the call (sorting would require an index
//! permutation, which is an allocation we avoid).

use crate::screening::complexity_prior::{ComplexityProxy, CompressionPriorSampler};

/// Drafter hook that re-ranks candidate drafts by K-prior.
///
/// See the module docs for why this is a post-drafting re-ranker rather than
/// a drafter wrapper.
#[derive(Debug, Clone, Copy)]
pub struct KPriorDrafter<K: ComplexityProxy> {
    sampler: CompressionPriorSampler<K>,
}

impl<K: ComplexityProxy> KPriorDrafter<K> {
    /// Construct from a sampler.
    #[inline]
    #[must_use]
    pub const fn new(sampler: CompressionPriorSampler<K>) -> Self {
        Self { sampler }
    }

    /// Borrow the inner sampler.
    #[inline]
    pub const fn sampler(&self) -> &CompressionPriorSampler<K> {
        &self.sampler
    }

    /// Re-rank drafts in place by K-prior.
    ///
    /// For each draft `i`, computes `log_prior_i = sampler.log_prob(draft_i)`
    /// and adds it to `scores[i]` (log-space multiplication by `exp(log_prior)`).
    /// Low-K drafts (high `log_prior`) get boosted; high-K drafts demoted.
    ///
    /// # Arguments
    ///
    /// - `drafts_bytes`: byte encoding of each draft (caller's responsibility).
    /// - `scores`: per-draft log-score from the base drafter. **Modified in
    ///   place**: `scores[i] += log_prior_i`.
    /// - `scratch`: caller-provided scratch buffer for log-priors. Length must
    ///   equal `drafts_bytes.len()`. Written but not read back by the caller
    ///   (useful for debugging / logging the prior contribution).
    ///
    /// # Zero allocation
    ///
    /// No heap allocation. The caller sorts `scores` descending after this
    /// call (sorting would require an index permutation, which we avoid).
    ///
    /// # Panics
    ///
    /// Debug-build assert: `drafts_bytes.len() == scores.len() == scratch.len()`.
    #[inline]
    pub fn rerank(&self, drafts_bytes: &[&[u8]], scores: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(
            drafts_bytes.len(),
            scores.len(),
            "rerank: drafts_bytes.len()={} must equal scores.len()={}",
            drafts_bytes.len(),
            scores.len()
        );
        debug_assert_eq!(
            drafts_bytes.len(),
            scratch.len(),
            "rerank: drafts_bytes.len()={} must equal scratch.len()={}",
            drafts_bytes.len(),
            scratch.len()
        );
        for (i, &d) in drafts_bytes.iter().enumerate() {
            scratch[i] = self.sampler.log_prob(d);
            scores[i] += scratch[i];
        }
        // Caller sorts `scores` descending; we don't sort here to keep the
        // function zero-allocation (sort would need an index permutation).
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::complexity_prior::RleComplexity;

    #[test]
    fn rerank_adds_log_prior_to_scores() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let drafter = KPriorDrafter::new(sampler);
        // Three drafts with distinct byte patterns.
        let d0: &[u8] = &[0, 0, 0, 0]; // low-K
        let d1: &[u8] = &[1, 2, 3, 4];
        let d2: &[u8] = &[255, 0, 255, 0]; // higher-K
        let drafts: [&[u8]; 3] = [d0, d1, d2];
        // Baseline scores from the (mock) base drafter.
        let mut scores = [0.0f32, 0.0, 0.0];
        let mut scratch = [0.0f32, 0.0, 0.0];
        drafter.rerank(&drafts, &mut scores, &mut scratch);
        // Verify scores[i] == log_prob(draft_i) exactly (baseline was 0.0).
        for (i, &d) in drafts.iter().enumerate() {
            assert_eq!(
                scores[i],
                sampler.log_prob(d),
                "scores[{i}] must equal log_prob(draft_{i}) after rerank"
            );
            assert_eq!(scratch[i], sampler.log_prob(d));
        }
    }

    #[test]
    fn low_k_draft_boosted() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let drafter = KPriorDrafter::new(sampler);
        // Two drafts with EQUAL base scores; after rerank, low-K must end up
        // with the higher score.
        let low_k: &[u8] = &[0u8; 64]; // RLE K̃ ≈ 0.031
        let mut high_k = [0u8; 64];
        for (i, b) in high_k.iter_mut().enumerate() {
            *b = if i % 2 == 0 { 255 } else { 0 };
        }
        let high_k_ref: &[u8] = &high_k; // RLE K̃ = 2.0
        let drafts: [&[u8]; 2] = [low_k, high_k_ref];
        let mut scores = [1.0f32, 1.0]; // equal base
        let mut scratch = [0.0f32, 0.0];
        drafter.rerank(&drafts, &mut scores, &mut scratch);
        assert!(
            scores[0] > scores[1],
            "low-K draft (index 0) should have higher score after rerank: s0={}, s1={}",
            scores[0],
            scores[1]
        );
    }

    #[test]
    fn empty_drafts_noop() {
        let sampler = CompressionPriorSampler::<RleComplexity>::default();
        let drafter = KPriorDrafter::new(sampler);
        let drafts: [&[u8]; 0] = [];
        let mut scores: [f32; 0] = [];
        let mut scratch: [f32; 0] = [];
        // Must not panic; scores / scratch remain empty.
        drafter.rerank(&drafts, &mut scores, &mut scratch);
        assert!(scores.is_empty());
        assert!(scratch.is_empty());
    }
}
