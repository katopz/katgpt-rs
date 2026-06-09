//! BFCP Region Pruner — skip reject regions, sample accept regions (Plan 213).
//!
//! Partitions logit space into BFCP regions by evaluating ScreeningPruner
//! relevance for each token, then groups tokens by label into BorelRegion
//! instances. Reject regions are skipped entirely — O(regions) instead of
//! O(vocab_size).

use crate::speculative::types::ScreeningPruner;

use super::bfcf_types::{BFCP, BorelRegion, RegionLabel};

// ── Thresholds ─────────────────────────────────────────────────

const ACCEPT_THRESHOLD: f32 = 0.7;
const REJECT_THRESHOLD: f32 = 0.3;

// ── BFCPPruner ─────────────────────────────────────────────────

/// Region-level pruner built from a ScreeningPruner partition.
///
/// Constructs a BFCP by classifying each token via the inner pruner's
/// relevance score, grouping tokens by label into convex regions.
pub struct BFCPPruner {
    partition: BFCP,
    /// Token IDs that passed screening (accept + maybe).
    kept_tokens: Vec<usize>,
    /// Number of reject-region tokens (work saved).
    skip_count: usize,
}

impl BFCPPruner {
    /// Build a BFCPPruner by partitioning tokens via the inner pruner.
    ///
    /// Each token gets a label:
    /// - `relevance >= 0.7` → Accept
    /// - `relevance <= 0.3` → Reject
    /// - otherwise → Maybe
    ///
    /// Tokens are grouped into up to 3 BorelRegion instances (one per label).
    pub fn from_logits(pruner: &dyn ScreeningPruner, _logits: &[f32], vocab_size: usize) -> Self {
        let mut accept_tokens = Vec::with_capacity(vocab_size / 2);
        let mut reject_tokens = Vec::with_capacity(vocab_size / 2);
        let mut maybe_tokens = Vec::with_capacity(vocab_size / 4);

        for token_idx in 0..vocab_size {
            let relevance = pruner.relevance(0, token_idx, &[]);
            match classify(relevance) {
                RegionLabel::Accept => {
                    accept_tokens.push(token_idx);
                }
                RegionLabel::Reject => {
                    reject_tokens.push(token_idx);
                }
                RegionLabel::Maybe => {
                    maybe_tokens.push(token_idx);
                }
            }
        }

        let skip_count = reject_tokens.len();

        // Build regions — one per label with non-zero tokens.
        let mut regions = Vec::with_capacity(3);
        if !accept_tokens.is_empty() {
            regions.push(BorelRegion {
                constraints: Vec::new(),
                label: RegionLabel::Accept,
                token_count: accept_tokens.len(),
                boundary_precision: 0.0,
            });
        }
        if !reject_tokens.is_empty() {
            regions.push(BorelRegion {
                constraints: Vec::new(),
                label: RegionLabel::Reject,
                token_count: reject_tokens.len(),
                boundary_precision: 0.0,
            });
        }
        if !maybe_tokens.is_empty() {
            regions.push(BorelRegion {
                constraints: Vec::new(),
                label: RegionLabel::Maybe,
                token_count: maybe_tokens.len(),
                boundary_precision: 0.0,
            });
        }

        let mut kept_tokens = Vec::with_capacity(accept_tokens.len() + maybe_tokens.len());
        kept_tokens.extend_from_slice(&accept_tokens);
        kept_tokens.extend_from_slice(&maybe_tokens);

        let partition = BFCP::from_regions(regions);

        Self {
            partition,
            kept_tokens,
            skip_count,
        }
    }

    /// Token IDs that survive screening (accept + maybe regions).
    pub fn pruned_token_ids(&self) -> Vec<usize> {
        self.kept_tokens.clone()
    }

    /// Number of tokens in reject regions — work saved by region-level skip.
    pub fn skip_count(&self) -> usize {
        self.skip_count
    }

    /// Reference to the underlying BFCP partition.
    pub fn partition(&self) -> &BFCP {
        &self.partition
    }
}

/// Classify a relevance score into a region label.
#[inline]
fn classify(relevance: f32) -> RegionLabel {
    if relevance >= ACCEPT_THRESHOLD {
        RegionLabel::Accept
    } else if relevance <= REJECT_THRESHOLD {
        RegionLabel::Reject
    } else {
        RegionLabel::Maybe
    }
}

// ── ScreeningPruner impl ──────────────────────────────────────

impl ScreeningPruner for BFCPPruner {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        match self.kept_tokens.binary_search(&token_idx) {
            Ok(_) => {
                // Token is in kept set — check which bucket by scanning regions.
                // For Phase 1 we return a fixed value based on label.
                for region in &self.partition.regions {
                    if region.label == RegionLabel::Accept && token_idx < region.token_count {
                        return 1.0;
                    }
                }
                0.5 // Maybe region
            }
            Err(_) => 0.0, // Reject region
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy pruner that returns fixed relevance values based on token index.
    struct SyntheticPruner {
        /// relevance for each token index.
        scores: Vec<f32>,
    }

    impl ScreeningPruner for SyntheticPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.scores[token_idx]
        }
    }

    #[test]
    fn test_from_logits_partition() {
        let scores = vec![0.9, 0.8, 0.1, 0.2, 0.5];
        let pruner = SyntheticPruner { scores };
        let logits = vec![1.0; 5];
        let bfcp = BFCPPruner::from_logits(&pruner, &logits, 5);

        let partition = bfcp.partition();
        assert!(partition.covers_all(5));
        // tokens 0,1 → Accept (2), tokens 2,3 → Reject (2), token 4 → Maybe (1)
        assert_eq!(
            partition
                .accept_regions()
                .map(|r| r.token_count)
                .sum::<usize>(),
            2
        );
        assert_eq!(
            partition
                .reject_regions()
                .map(|r| r.token_count)
                .sum::<usize>(),
            2
        );
        assert_eq!(
            partition
                .maybe_regions()
                .map(|r| r.token_count)
                .sum::<usize>(),
            1
        );
    }

    #[test]
    fn test_pruned_tokens_skip_reject() {
        let scores = vec![0.9, 0.1, 0.5, 0.2, 0.8];
        let pruner = SyntheticPruner { scores };
        let logits = vec![1.0; 5];
        let bfcp = BFCPPruner::from_logits(&pruner, &logits, 5);

        let pruned = bfcp.pruned_token_ids();
        // Accept: 0, 4. Maybe: 2. Reject: 1, 3.
        assert!(pruned.contains(&0));
        assert!(pruned.contains(&2));
        assert!(pruned.contains(&4));
        assert!(!pruned.contains(&1));
        assert!(!pruned.contains(&3));
    }

    #[test]
    fn test_skip_count() {
        let scores = vec![0.9, 0.1, 0.5, 0.2, 0.8];
        let pruner = SyntheticPruner { scores };
        let logits = vec![1.0; 5];
        let bfcp = BFCPPruner::from_logits(&pruner, &logits, 5);

        // tokens 1 (0.1) and 3 (0.2) are reject
        assert_eq!(bfcp.skip_count(), 2);
    }

    #[test]
    fn test_screening_pruner_delegate() {
        let scores = vec![0.9, 0.1, 0.5];
        let pruner = SyntheticPruner { scores };
        let logits = vec![1.0; 3];
        let bfcp = BFCPPruner::from_logits(&pruner, &logits, 3);

        // kept_tokens should be sorted: [0, 2] (accept=0, maybe=2)
        let rel0 = bfcp.relevance(0, 0, &[]);
        let rel1 = bfcp.relevance(0, 1, &[]);
        let rel2 = bfcp.relevance(0, 2, &[]);

        // Token 0 is in kept set (accept) → should return nonzero
        assert!(rel0 > 0.0);
        // Token 1 is rejected → 0.0
        assert_eq!(rel1, 0.0);
        // Token 2 is in kept set (maybe) → should return nonzero
        assert!(rel2 > 0.0);
    }
}
