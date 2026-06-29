//! ResidPruner — constraint-filtered residual injection (Plan 258).
//!
//! Before injecting RCD residuals into the denoising loop, verifies that
//! the top-K tokens in the residual distribution are pruner-valid.
//! If none pass the constraint pruner → skip injection, use standard mask embedding.
//! This prevents noise injection from nonsensical context.

#![allow(clippy::needless_range_loop)]

use katgpt_core::ConstraintPruner;

/// Residual pruner that wraps a `ConstraintPruner` to filter implausible residuals.
///
/// Before computing the full codebook-weighted residual Δ_i, checks whether
/// any of the top-K tokens in the marginal distribution pass the constraint.
/// If none pass → the residual would be nonsensical, so we skip injection.
///
/// This is a zero-allocation filter: reuses caller-provided scratch buffers.
pub struct ResidPruner<'a> {
    /// Reference to the underlying constraint pruner.
    pruner: &'a dyn ConstraintPruner,
    /// Number of top tokens to check (default: 5).
    top_k: usize,
}

impl<'a> ResidPruner<'a> {
    /// Create a new ResidPruner wrapping the given constraint pruner.
    pub fn new(pruner: &'a dyn ConstraintPruner) -> Self {
        Self { pruner, top_k: 5 }
    }

    /// Create with custom top-K check depth.
    pub fn with_top_k(pruner: &'a dyn ConstraintPruner, top_k: usize) -> Self {
        Self {
            pruner,
            top_k: top_k.max(1),
        }
    }

    /// Check whether residual injection should proceed for the given position.
    ///
    /// Extracts top-K token indices from `marginals`, checks each against the
    /// constraint pruner. Returns `true` if at least one top-K token is valid.
    ///
    /// `scratch_indices` is a caller-provided buffer of at least `top_k` elements.
    pub fn should_inject(
        &self,
        position: usize,
        marginals: &[f32],
        current_tokens: &[usize],
        scratch_indices: &mut [(usize, f32)],
    ) -> bool {
        let k = self.top_k.min(scratch_indices.len()).min(marginals.len());
        if k == 0 {
            return false;
        }

        // Find top-K indices from marginals (partial sort via insertion)
        let mut filled = 0;
        for (idx, &prob) in marginals.iter().enumerate() {
            if filled < k {
                // Insert into sorted position
                let pos = scratch_indices[..filled].partition_point(|&(_, p)| p >= prob);
                // Shift right
                for j in (pos + 1..=filled.min(k - 1)).rev() {
                    scratch_indices[j] = scratch_indices[j - 1];
                }
                scratch_indices[pos] = (idx, prob);
                filled = filled.saturating_add(1).min(k);
            } else if prob > scratch_indices[k - 1].1 {
                // Replace the smallest
                scratch_indices[k - 1] = (idx, prob);
                // Bubble up to maintain sorted order
                let mut j = k - 1;
                while j > 0 && scratch_indices[j].1 > scratch_indices[j - 1].1 {
                    scratch_indices.swap(j, j - 1);
                    j -= 1;
                }
            }
        }

        // Check if ANY top-K token passes the constraint pruner
        for i in 0..filled {
            let (token_idx, _) = scratch_indices[i];
            if self.pruner.is_valid(position, token_idx, current_tokens) {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pruner that only allows even token IDs.
    struct EvenOnlyPruner;

    impl ConstraintPruner for EvenOnlyPruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx.is_multiple_of(2)
        }

        fn batch_is_valid(
            &self,
            depth: usize,
            candidates: &[usize],
            parent_tokens: &[usize],
            results: &mut [bool],
        ) {
            for (i, &c) in candidates.iter().enumerate() {
                results[i] = self.is_valid(depth, c, parent_tokens);
            }
        }
    }

    /// A pruner that rejects everything.
    struct RejectAllPruner;

    impl ConstraintPruner for RejectAllPruner {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            false
        }

        fn batch_is_valid(
            &self,
            _depth: usize,
            _candidates: &[usize],
            _parent_tokens: &[usize],
            results: &mut [bool],
        ) {
            results.fill(false);
        }
    }

    #[test]
    fn test_resid_pruner_allows_when_valid_token_in_top_k() {
        let pruner = EvenOnlyPruner;
        let resid = ResidPruner::new(&pruner);
        // Token 0 (even) has highest probability
        let marginals = vec![0.5, 0.3, 0.2];
        let mut scratch = [(0usize, 0.0f32); 5];
        assert!(resid.should_inject(0, &marginals, &[], &mut scratch));
    }

    #[test]
    fn test_resid_pruner_rejects_when_no_valid_in_top_k() {
        let pruner = EvenOnlyPruner;
        let resid = ResidPruner::with_top_k(&pruner, 3);
        // Top 3 tokens: 1, 3, 5 — all odd
        let mut marginals = vec![0.0f32; 10];
        marginals[1] = 0.4;
        marginals[3] = 0.3;
        marginals[5] = 0.2;
        let mut scratch = [(0usize, 0.0f32); 5];
        assert!(!resid.should_inject(0, &marginals, &[], &mut scratch));
    }

    #[test]
    fn test_resid_pruner_rejects_all() {
        let pruner = RejectAllPruner;
        let resid = ResidPruner::new(&pruner);
        let marginals = vec![0.5, 0.3, 0.2];
        let mut scratch = [(0usize, 0.0f32); 5];
        assert!(!resid.should_inject(0, &marginals, &[], &mut scratch));
    }

    #[test]
    fn test_resid_pruner_empty_marginals() {
        let pruner = EvenOnlyPruner;
        let resid = ResidPruner::new(&pruner);
        let marginals: Vec<f32> = vec![];
        let mut scratch = [(0usize, 0.0f32); 5];
        assert!(!resid.should_inject(0, &marginals, &[], &mut scratch));
    }
}
