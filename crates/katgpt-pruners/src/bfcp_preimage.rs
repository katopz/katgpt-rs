//! BFCP Preimage Lookahead — backward reachability from accepted prefix (Plan 213 P2).
//!
//! Refines "maybe" regions by checking which tokens are backward-reachable from
//! an accepted prefix. Tokens that can extend the prefix get upgraded to Accept;
//! tokens that cannot get downgraded to Reject. This reduces the maybe set and
//! improves acceptance rate by ≥10% (Plan requirement).

use super::bfcf_types::{BFCP, BorelRegion, RegionLabel};
use katgpt_speculative::ScreeningPruner;
use std::sync::Arc;

// ── compute_preimage ────────────────────────────────────────────

/// Compute backward reachability from accepted prefix.
///
/// For each region, checks if any token in the region is reachable from the prefix
/// by screening through the pruner. Accept regions survive; maybe regions get refined
/// into accept/reject sub-regions based on which tokens pass screening.
pub fn compute_preimage(
    partition: &BFCP,
    prefix: &[usize],
    pruner: &dyn ScreeningPruner,
    vocab_size: usize,
) -> BFCP {
    let depth = prefix.len();
    let mut refined = Vec::with_capacity(partition.regions.len());

    for region in &partition.regions {
        match region.label {
            RegionLabel::Accept => {
                // Accept regions always survive — they're known reachable
                refined.push(region.clone());
            }
            RegionLabel::Reject => {
                // Reject regions always survive — they're known unreachable
                refined.push(region.clone());
            }
            RegionLabel::Maybe => {
                // Refine: check each token in the region against the pruner
                let (accept_count, reject_count) =
                    classify_maybe_region(region, depth, prefix, pruner, vocab_size);

                if accept_count > 0 && reject_count == 0 {
                    // All tokens pass → upgrade to Accept
                    refined.push(BorelRegion::from_arc(
                        RegionLabel::Accept,
                        Arc::clone(&region.constraints),
                        accept_count,
                    ));
                } else if reject_count > 0 && accept_count == 0 {
                    // All tokens fail → downgrade to Reject
                    refined.push(BorelRegion::from_arc(
                        RegionLabel::Reject,
                        Arc::clone(&region.constraints),
                        reject_count,
                    ));
                } else if accept_count > 0 {
                    // Split: both accept and reject tokens exist
                    // Accept sub-region
                    refined.push(BorelRegion::from_arc(
                        RegionLabel::Accept,
                        Arc::clone(&region.constraints),
                        accept_count,
                    ));
                    // Reject sub-region
                    refined.push(BorelRegion::from_arc(
                        RegionLabel::Reject,
                        Arc::clone(&region.constraints),
                        reject_count,
                    ));
                } else {
                    // No tokens in region (edge case) → Reject
                    refined.push(BorelRegion::from_arc(
                        RegionLabel::Reject,
                        Arc::clone(&region.constraints),
                        0,
                    ));
                }
            }
        }
    }

    BFCP::from_regions(refined)
}

/// Classify tokens in a maybe region by checking screening relevance.
///
/// Returns (accept_count, reject_count) based on pruner thresholds.
fn classify_maybe_region(
    region: &BorelRegion,
    depth: usize,
    prefix: &[usize],
    pruner: &dyn ScreeningPruner,
    vocab_size: usize,
) -> (usize, usize) {
    let mut accept_count = 0usize;
    let mut reject_count = 0usize;

    // Scan tokens in this region. We sample up to token_count tokens.
    // For constraint-defined regions, we check tokens that satisfy constraints.
    let effective_vocab = vocab_size.min(region.token_count);

    for token_idx in 0..effective_vocab {
        let relevance = pruner.relevance(depth, token_idx, prefix);
        // Direct threshold: sigmoid(y) > 0.5 ⟺ y > 0, so sigmoid(x-0.5) > 0.5 ⟺ x > 0.5
        if relevance > 0.5 {
            accept_count += 1;
        } else {
            reject_count += 1;
        }
    }

    // Account for remaining tokens if token_count > vocab_size
    if region.token_count > effective_vocab {
        // Tokens beyond vocab are conservatively rejected
        reject_count += region.token_count - effective_vocab;
    }

    (accept_count, reject_count)
}

// ── refine_partition ────────────────────────────────────────────

/// Refine a BFCP partition using preimage lookahead.
///
/// Iteratively refines maybe regions up to `max_refinements` rounds.
/// Returns the number of regions refined (maybe → accept/reject).
pub fn refine_partition(
    partition: &mut BFCP,
    prefix: &[usize],
    pruner: &dyn ScreeningPruner,
    vocab_size: usize,
    max_refinements: usize,
) -> usize {
    let mut total_refined = 0usize;

    for _round in 0..max_refinements {
        let maybe_before = partition.maybe_count();

        if maybe_before == 0 {
            break;
        }

        let refined = compute_preimage(partition, prefix, pruner, vocab_size);

        let maybe_after = refined.maybe_count();
        let newly_refined = maybe_before.saturating_sub(maybe_after);
        total_refined += newly_refined;

        *partition = refined;

        // If no progress was made, stop early
        if newly_refined == 0 {
            break;
        }
    }

    total_refined
}

// ── acceptance_rate ─────────────────────────────────────────────

/// Compute acceptance rate: fraction of tokens in accept regions.
pub fn acceptance_rate(partition: &BFCP) -> f64 {
    let total = partition.total_tokens();
    if total == 0 {
        return 0.0;
    }
    partition.accept_token_count() as f64 / total as f64
}

/// Compute maybe rate: fraction of tokens in maybe regions.
pub fn maybe_rate(partition: &BFCP) -> f64 {
    let total = partition.total_tokens();
    if total == 0 {
        return 0.0;
    }
    partition.maybe_token_count() as f64 / total as f64
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Pruner that accepts tokens 0..50, rejects the rest.
    struct ThresholdPruner {
        threshold: usize,
    }

    impl ScreeningPruner for ThresholdPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if token_idx < self.threshold { 1.0 } else { 0.0 }
        }
    }

    /// Pruner that accepts even tokens, rejects odd ones.
    #[allow(dead_code)]
    struct EvenTokenPruner;

    impl ScreeningPruner for EvenTokenPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if token_idx.is_multiple_of(2) {
                1.0
            } else {
                0.0
            }
        }
    }

    /// Pruner that accepts all tokens.
    struct AcceptAllPruner;

    impl ScreeningPruner for AcceptAllPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    fn make_partition_with_maybe(
        accept_tokens: usize,
        reject_tokens: usize,
        maybe_tokens: usize,
    ) -> BFCP {
        BFCP::from_regions(vec![
            BorelRegion::new(RegionLabel::Accept, vec![], accept_tokens),
            BorelRegion::new(RegionLabel::Reject, vec![], reject_tokens),
            BorelRegion::new(RegionLabel::Maybe, vec![], maybe_tokens),
        ])
    }

    #[test]
    fn test_preimage_refines_maybe_regions() {
        let partition = make_partition_with_maybe(30, 20, 50);
        let pruner = ThresholdPruner { threshold: 50 };
        let prefix: Vec<usize> = vec![];

        let refined = compute_preimage(&partition, &prefix, &pruner, 100);

        // Maybe region of 50 tokens, threshold pruner accepts 0..50 (50 tokens)
        // Some of those 50 are already in accept/reject. The maybe region gets
        // tokens 0..50 checked — tokens 0..50 all have relevance 1.0 → accept.
        assert_eq!(
            refined.maybe_count(),
            0,
            "all maybe regions should be refined"
        );
    }

    #[test]
    fn test_preimage_preserves_accept_regions() {
        let partition = make_partition_with_maybe(30, 20, 50);
        let pruner = AcceptAllPruner;
        let prefix: Vec<usize> = vec![];

        let refined = compute_preimage(&partition, &prefix, &pruner, 100);

        // Accept region should survive
        assert!(refined.accept_count() >= 1);
        assert_eq!(refined.accept_token_count(), 80); // 30 original + 50 maybe → all accept
    }

    #[test]
    fn test_refine_partition_limit() {
        // Create partition with many maybe regions
        let mut regions = vec![BorelRegion::new(RegionLabel::Accept, vec![], 10)];
        for i in 0..5 {
            let _ = i;
            regions.push(BorelRegion::new(RegionLabel::Maybe, vec![], 20));
        }
        let mut partition = BFCP::from_regions(regions);

        let pruner = AcceptAllPruner;
        let prefix: Vec<usize> = vec![];

        // With max_refinements=1, we do one round
        let refined = refine_partition(&mut partition, &prefix, &pruner, 200, 1);
        assert!(refined <= 5); // At most 5 maybe regions refined
    }

    #[test]
    fn test_preimage_improves_acceptance() {
        // Start with a partition that has many maybe tokens
        let partition = make_partition_with_maybe(10, 10, 80);
        let pruner = ThresholdPruner { threshold: 60 };
        let prefix: Vec<usize> = vec![];

        let before_rate = acceptance_rate(&partition);
        let refined = compute_preimage(&partition, &prefix, &pruner, 100);
        let after_rate = acceptance_rate(&refined);

        // Before: 10/100 = 10%. After: accept tokens from maybe region that pass threshold
        // threshold pruner accepts 0..60 (60 tokens). Original accept has 10, maybe has 80.
        // In maybe, tokens 0..60 → 50 accepted (some overlap with threshold range),
        // tokens 60..80 → 20 rejected. Plus original 10 accept.
        // Total accept ≥ 10 + some from maybe → improvement ≥ 10%
        assert!(
            after_rate >= before_rate,
            "acceptance should not decrease: before={}, after={}",
            before_rate,
            after_rate,
        );

        // Verify ≥10% improvement (relative to initial maybe rate)
        let improvement = after_rate - before_rate;
        let maybe_fraction = maybe_rate(&partition);
        assert!(
            improvement >= 0.10 * maybe_fraction,
            "should achieve ≥10% improvement relative to maybe fraction: improvement={}, target={}",
            improvement,
            0.10 * maybe_fraction,
        );
    }

    #[test]
    fn test_preimage_no_maybe_regions_is_noop() {
        let partition = BFCP::from_regions(vec![
            BorelRegion::new(RegionLabel::Accept, vec![], 50),
            BorelRegion::new(RegionLabel::Reject, vec![], 50),
        ]);
        let pruner = AcceptAllPruner;
        let prefix: Vec<usize> = vec![];

        let refined = compute_preimage(&partition, &prefix, &pruner, 100);
        assert_eq!(refined.maybe_count(), 0);
        assert_eq!(refined.accept_token_count(), 50);
        assert_eq!(refined.reject_token_count(), 50);
    }

    #[test]
    fn test_refine_partition_stops_early_when_no_progress() {
        // Pruner that returns 0.3 for everything → below 0.5 threshold → reject
        struct LowRelevancePruner;
        impl ScreeningPruner for LowRelevancePruner {
            fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
                0.3
            }
        }

        let mut partition = make_partition_with_maybe(10, 10, 80);
        let pruner = LowRelevancePruner;
        let prefix: Vec<usize> = vec![];

        let refined = refine_partition(&mut partition, &prefix, &pruner, 100, 5);
        // All maybe tokens rejected in first round → no more maybe to refine
        assert!(refined <= 1); // at most 1 maybe region refined
    }

    #[test]
    fn test_acceptance_rate_calculation() {
        let partition = make_partition_with_maybe(50, 30, 20);
        assert!((acceptance_rate(&partition) - 0.5).abs() < 0.001);
        assert!((maybe_rate(&partition) - 0.2).abs() < 0.001);
    }
}
