//! DynamicRankPruner — GATv2-Inspired Static Ranking Detection & Correction (Plan 232)
//!
//! GATv2 (ICLR 2022) proves that GAT computes static attention: the ranking of attention
//! scores is unconditioned on the query node. In katgpt-rs, BanditPruner has the same problem:
//! Q-values are indexed by arm only, not by (arm, parent_context).
//!
//! This module provides:
//! 1. `static_ranking_diagnostic()` — detects whether a ScreeningPruner produces static ranking
//! 2. `DynamicRankPruner<P>` — wrapper that corrects static ranking with prefix-aware deltas
//!
//! Feature-gated under `dynamic_rank`. Zero overhead when inner pruner is already dynamic.

use std::sync::atomic::{AtomicBool, Ordering};

use katgpt_core::traits::ScreeningPruner;

// ── Diagnostic ──────────────────────────────────────────────────

/// Result of static ranking diagnosis.
#[derive(Debug, Clone)]
pub struct StaticRankingReport {
    /// Whether the inner pruner produces static (context-invariant) ranking.
    pub is_static: bool,
    /// How much the argsort varies across contexts. 0.0 = perfectly static, 1.0 = perfectly dynamic.
    pub ranking_entropy: f32,
    /// Number of parent contexts sampled.
    pub sample_count: usize,
}

/// Diagnose whether a ScreeningPruner produces static ranking.
///
/// Method: sample N parent contexts, compute argsort of relevance for each context at each depth,
/// measure how much the argsort varies. If argsort is invariant → static.
///
/// We use Kendall tau distance between ranking pairs as the entropy measure.
pub fn static_ranking_diagnostic(
    pruner: &dyn ScreeningPruner,
    vocab_size: usize,
    max_depth: usize,
    sample_count: usize,
) -> StaticRankingReport {
    // For each depth, collect argsort vectors from different parent contexts
    let mut total_entropy = 0.0f32;
    let mut depth_count = 0usize;

    for depth in 0..max_depth {
        // Generate sample_count different parent contexts
        let mut rankings: Vec<Vec<usize>> = Vec::with_capacity(sample_count);

        for sample in 0..sample_count {
            // Create a synthetic parent context
            let parent: Vec<usize> = (0..depth)
                .map(|d| (sample * 7 + d * 13 + 42) % vocab_size)
                .collect();

            // Skip depth 0 (empty parent, all rankings identical)
            if depth == 0 {
                continue;
            }

            // Compute relevance scores for all tokens
            let mut scored: Vec<(usize, f32)> = (0..vocab_size)
                .map(|t| (t, pruner.relevance(depth, t, &parent)))
                .collect();

            // Sort by relevance descending to get ranking
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let ranking: Vec<usize> = scored.iter().map(|(t, _)| *t).collect();
            rankings.push(ranking);
        }

        if rankings.len() < 2 {
            continue;
        }

        // Compute average pairwise Kendall tau distance
        let (pair_count, total_distance) = pairwise_kendall_tau(&rankings);

        if pair_count > 0 {
            total_entropy += total_distance / pair_count as f32;
            depth_count += 1;
        }
    }

    let avg_entropy = if depth_count > 0 {
        total_entropy / depth_count as f32
    } else {
        0.0
    };

    // Static if entropy is very low (argsort barely changes across contexts)
    StaticRankingReport {
        is_static: avg_entropy < 0.05, // 5% threshold — almost no ranking variation
        ranking_entropy: avg_entropy,
        sample_count,
    }
}

/// Sum of pairwise normalized Kendall tau distances across all ranking pairs.
fn pairwise_kendall_tau(rankings: &[Vec<usize>]) -> (usize, f32) {
    let mut pair_count = 0usize;
    let mut total_distance = 0.0f32;

    for i in 0..rankings.len() {
        for j in (i + 1)..rankings.len() {
            let tau = kendall_tau_normalized(&rankings[i], &rankings[j]);
            total_distance += tau;
            pair_count += 1;
        }
    }

    (pair_count, total_distance)
}

/// Normalized Kendall tau distance between two rankings.
/// Returns 0.0 if identical, 1.0 if completely reversed.
/// Uses O(n²) pairwise comparison (fine for diagnostic, not hot path).
fn kendall_tau_normalized(a: &[usize], b: &[usize]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }

    // Build position maps: item → position in each ranking
    let mut pos_a = vec![0usize; n];
    let mut pos_b = vec![0usize; n];
    for i in 0..n {
        if a[i] < n {
            pos_a[a[i]] = i;
        }
        if b[i] < n {
            pos_b[b[i]] = i;
        }
    }

    // Count discordant pairs: pairs of items (p, q) where
    // p comes before q in ranking a but after q in ranking b
    let mut discordant = 0usize;
    let mut total = 0usize;

    for p in 0..n {
        for q in (p + 1)..n {
            // In ranking a, is p before q?
            let a_before = pos_a[p] < pos_a[q];
            // In ranking b, is p before q?
            let b_before = pos_b[p] < pos_b[q];
            if a_before != b_before {
                discordant += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        0.0
    } else {
        discordant as f32 / total as f32
    }
}

// ── DynamicRankPruner ──────────────────────────────────────────

/// Wrapper that detects and corrects static ranking in ScreeningPruner.
///
/// GATv2 insight: if argsort(relevance(token_j)) is invariant across
/// different parent contexts, the inner pruner is "static" — it cannot
/// discriminate between contexts. This wrapper:
/// 1. Diagnoses static ranking by comparing argsort across sampled parents
/// 2. If static: applies a context-dependent correction (prefix hash → delta)
/// 3. If dynamic: passes through unchanged (zero overhead)
pub struct DynamicRankPruner<P: ScreeningPruner> {
    inner: P,
    /// Diagnosis result (lazily computed).
    diagnosed: AtomicBool,
    is_static: AtomicBool,
    /// Correction table: fnv1a(parent_tokens) → per-token correction delta.
    /// Only populated if is_static.
    corrections: papaya::HashMap<u64, Vec<f32>>,
    /// Vocab size for correction vector allocation.
    vocab_size: usize,
}

impl<P: ScreeningPruner> DynamicRankPruner<P> {
    pub fn new(inner: P, vocab_size: usize) -> Self {
        Self {
            inner,
            diagnosed: AtomicBool::new(false),
            is_static: AtomicBool::new(false),
            corrections: papaya::HashMap::new(),
            vocab_size,
        }
    }

    /// Run the diagnostic and cache the result.
    /// Called once, on first relevance() call that has parent context.
    fn run_diagnostic(&self, max_depth: usize) {
        if self.diagnosed.load(Ordering::Relaxed) {
            return;
        }

        let report = static_ranking_diagnostic(
            &self.inner,
            self.vocab_size,
            max_depth.min(8), // Cap diagnostic depth
            10,               // 10 sample contexts
        );

        self.is_static.store(report.is_static, Ordering::Relaxed);
        self.diagnosed.store(true, Ordering::Relaxed);
    }

    /// Compute a hash of the parent token prefix for correction lookup.
    fn prefix_hash(parent: &[usize]) -> u64 {
        // Fast FNV-1a for small arrays
        let mut hash: u64 = 0xcbf29ce484222325;
        for &v in parent {
            hash ^= v as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Record a correction for a given parent context.
    /// Called after observing reward feedback.
    pub fn record_correction(&self, parent: &[usize], token_idx: usize, delta: f32) {
        if !self.is_static.load(Ordering::Relaxed) {
            return;
        }

        let hash = Self::prefix_hash(parent);
        let corrections = self.corrections.pin();
        if let Some(v) = corrections.get(&hash) {
            // Copy-on-write upsert: clone, mutate, re-insert
            let mut updated = v.clone();
            if token_idx < updated.len() {
                updated[token_idx] += delta * 0.01; // Small learning rate
            }
            corrections.insert(hash, updated);
        } else {
            let mut v = vec![0.0; self.vocab_size];
            if token_idx < v.len() {
                v[token_idx] = delta * 0.01;
            }
            corrections.insert(hash, v);
        }
    }

    /// Get the correction for a token given parent context.
    #[inline]
    fn get_correction(&self, parent: &[usize], token_idx: usize) -> f32 {
        if !self.is_static.load(Ordering::Relaxed) {
            return 0.0;
        }

        let hash = Self::prefix_hash(parent);
        let corrections = self.corrections.pin();
        if let Some(vec) = corrections.get(&hash)
            && token_idx < vec.len() {
                return vec[token_idx];
            }
        0.0
    }

    /// Check if the inner pruner has been diagnosed as static.
    pub fn is_static(&self) -> Option<bool> {
        if self.diagnosed.load(Ordering::Relaxed) {
            Some(self.is_static.load(Ordering::Relaxed))
        } else {
            None
        }
    }
}

impl<P: ScreeningPruner> ScreeningPruner for DynamicRankPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Lazy diagnosis on first call with parent context
        if !self.diagnosed.load(Ordering::Relaxed) && !parent_tokens.is_empty() {
            self.run_diagnostic(depth + 4); // Look ahead a few depths
        }

        let base = self.inner.relevance(depth, token_idx, parent_tokens);

        // Fast path: if not static, zero overhead
        if !self.is_static.load(Ordering::Relaxed) {
            return base;
        }

        // Slow path: apply context-dependent correction
        let correction = self.get_correction(parent_tokens, token_idx);
        (base + correction).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pruner that ignores parent context entirely (static ranking).
    struct StaticPruner;

    impl ScreeningPruner for StaticPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            // Pure function of token_idx — always the same ranking regardless of parent
            1.0 / (token_idx as f32 + 1.0)
        }
    }

    /// A pruner that uses parent context (dynamic ranking).
    struct DynamicPruner;

    impl ScreeningPruner for DynamicPruner {
        fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
            let context = parent_tokens.first().copied().unwrap_or(0);
            // Ranking depends on context — different parents produce different orderings
            1.0 / ((token_idx ^ context) as f32 + depth as f32 + 1.0)
        }
    }

    #[test]
    fn static_pruner_detected_as_static() {
        let pruner = StaticPruner;
        let report = static_ranking_diagnostic(&pruner, 20, 4, 5);
        assert!(
            report.is_static,
            "StaticPruner should be detected as static"
        );
        assert!(
            report.ranking_entropy < 0.05,
            "StaticPruner entropy should be < 0.05, got {}",
            report.ranking_entropy
        );
    }

    #[test]
    fn dynamic_pruner_not_detected_as_static() {
        let pruner = DynamicPruner;
        let report = static_ranking_diagnostic(&pruner, 20, 4, 5);
        assert!(
            !report.is_static,
            "DynamicPruner should NOT be detected as static"
        );
        assert!(
            report.ranking_entropy > 0.05,
            "DynamicPruner entropy should be > 0.05, got {}",
            report.ranking_entropy
        );
    }

    #[test]
    fn dynamic_rank_pruner_passthrough_when_dynamic() {
        let pruner = DynamicRankPruner::new(DynamicPruner, 20);

        // First call triggers diagnosis
        let r = pruner.relevance(1, 5, &[3, 7]);
        let expected = 1.0 / ((5 ^ 3) as f32 + 1.0 + 1.0);
        assert!(
            (r - expected).abs() < 1e-6,
            "Should pass through to inner pruner, got {} expected {}",
            r,
            expected
        );

        // Should be diagnosed as NOT static
        assert_eq!(pruner.is_static(), Some(false));
    }

    #[test]
    fn dynamic_rank_pruner_applies_correction_when_static() {
        let pruner = DynamicRankPruner::new(StaticPruner, 10);

        // First call triggers diagnosis
        let r0 = pruner.relevance(1, 3, &[0, 1]);

        // Should be diagnosed as static
        assert_eq!(pruner.is_static(), Some(true));

        // Record a positive correction for token 3 with parent [0, 1]
        pruner.record_correction(&[0, 1], 3, 5.0);

        // Same query should now be boosted
        let r1 = pruner.relevance(1, 3, &[0, 1]);
        assert!(
            r1 > r0,
            "After correction, relevance should increase: {} vs {}",
            r1,
            r0
        );
    }

    #[test]
    fn kendall_tau_identical_rankings() {
        let r = vec![0, 1, 2, 3, 4];
        assert_eq!(kendall_tau_normalized(&r, &r), 0.0);
    }

    #[test]
    fn kendall_tau_reversed_rankings() {
        let a = vec![0, 1, 2, 3, 4];
        let b = vec![4, 3, 2, 1, 0];
        let tau = kendall_tau_normalized(&a, &b);
        assert!(
            tau > 0.9,
            "Reversed rankings should have high tau, got {}",
            tau
        );
    }

    #[test]
    fn prefix_hash_deterministic() {
        let a = vec![1, 2, 3];
        let b = vec![3, 2, 1];
        assert_eq!(
            DynamicRankPruner::<StaticPruner>::prefix_hash(&a),
            DynamicRankPruner::<StaticPruner>::prefix_hash(&a)
        );
        assert_ne!(
            DynamicRankPruner::<StaticPruner>::prefix_hash(&a),
            DynamicRankPruner::<StaticPruner>::prefix_hash(&b)
        );
    }
}
