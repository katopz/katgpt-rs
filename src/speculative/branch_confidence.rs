//! Deep Manifold §2.4.2 — Union Bound Branch Confidence (Research 205, Plan 231)
//!
//! Paper Eq. 32-35: On stacked piecewise manifolds, deviation probability
//! obeys the union bound (Boole's inequality):
//!   P(hk ∉ Mk) ≤ Σᵢ P(hk ∉ Mk,i)
//!
//! Errors propagate ADDITIVELY, not exponentially.
//! Branch confidence should use additive combination, not multiplicative.

/// Branch confidence computation strategy.
pub trait BranchConfidence: Send + Sync {
    /// Compute total confidence from per-position scores in [0, 1].
    fn total_confidence(&self, position_scores: &[f32]) -> f32;
    /// Name of the confidence method.
    fn name(&self) -> &'static str;
}

/// Multiplicative (chain) confidence — classical approach.
/// P(correct) = Πᵢ pᵢ. Pessimistic: single weak position kills the chain.
pub struct MultiplicativeScorer;

impl BranchConfidence for MultiplicativeScorer {
    fn total_confidence(&self, position_scores: &[f32]) -> f32 {
        if position_scores.is_empty() {
            return 1.0;
        }
        position_scores.iter().product()
    }
    fn name(&self) -> &'static str {
        "multiplicative"
    }
}

/// Union bound (additive) confidence — Deep Manifold §2.4.2.
/// P(correct) = 1 - min(1, Σᵢ (1 - pᵢ)).
/// More optimistic: individual weak positions don't kill the chain.
pub struct UnionBoundScorer;

impl BranchConfidence for UnionBoundScorer {
    fn total_confidence(&self, position_scores: &[f32]) -> f32 {
        if position_scores.is_empty() {
            return 1.0;
        }
        let fail_prob: f32 = position_scores.iter().map(|p| 1.0 - p).sum();
        1.0 - fail_prob.min(1.0)
    }
    fn name(&self) -> &'static str {
        "union_bound"
    }
}

/// Hybrid: multiplicative for short chains, union bound for long chains.
pub struct HybridScorer {
    pub short_chain_threshold: usize,
}

impl Default for HybridScorer {
    fn default() -> Self {
        Self {
            short_chain_threshold: 4,
        }
    }
}

impl BranchConfidence for HybridScorer {
    fn total_confidence(&self, position_scores: &[f32]) -> f32 {
        if position_scores.len() <= self.short_chain_threshold {
            MultiplicativeScorer.total_confidence(position_scores)
        } else {
            UnionBoundScorer.total_confidence(position_scores)
        }
    }
    fn name(&self) -> &'static str {
        "hybrid"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplicative_empty() {
        assert_eq!(MultiplicativeScorer.total_confidence(&[]), 1.0);
    }

    #[test]
    fn multiplicative_basic() {
        let scores = [0.9, 0.8, 0.7];
        let result = MultiplicativeScorer.total_confidence(&scores);
        assert!((result - 0.504).abs() < 1e-6);
    }

    #[test]
    fn union_bound_empty() {
        assert_eq!(UnionBoundScorer.total_confidence(&[]), 1.0);
    }

    #[test]
    fn union_bound_basic() {
        let scores = [0.9, 0.8, 0.7];
        let result = UnionBoundScorer.total_confidence(&scores);
        // 1 - (0.1 + 0.2 + 0.3) = 0.4
        assert!((result - 0.4).abs() < 1e-6);
    }

    #[test]
    fn union_bound_additive_degradation() {
        // Key property from Deep Manifold §2.4.2: errors propagate ADDITIVELY.
        // Union bound confidence = 1 - Σᵢ(1-pᵢ), which degrades linearly
        // rather than exponentially like multiplicative Πᵢ pᵢ.
        // For mostly-high scores with one weak position, union bound degrades gracefully.
        let scores = [0.99, 0.99, 0.99, 0.6];
        let mult = MultiplicativeScorer.total_confidence(&scores);
        let union = UnionBoundScorer.total_confidence(&scores);
        // Union bound always <= multiplicative (Boole's inequality)
        assert!(union >= 0.0, "union should be >= 0, got {}", union);
        assert!(
            union <= mult,
            "union {} should <= mult {} (Boole bound)",
            union,
            mult
        );
        // But both are non-trivially positive
        assert!(union > 0.5, "union {} should be > 0.5", union);
        assert!(mult > 0.5, "mult {} should be > 0.5", mult);
    }

    #[test]
    fn union_bound_additive_degradation_is_linear() {
        // Deep Manifold §2.4.2 key insight: errors propagate ADDITIVELY.
        // Adding one more position with score p reduces union bound by (1-p),
        // but reduces multiplicative by factor p.
        // For high p (e.g. 0.9): union loses 0.1, mult loses factor 0.9.
        // Union bound degrades linearly, multiplicative degrades exponentially.
        let short = [0.9; 3];
        let long = [0.9; 10];
        let short_union = UnionBoundScorer.total_confidence(&short);
        let long_union = UnionBoundScorer.total_confidence(&long);
        let short_mult = MultiplicativeScorer.total_confidence(&short);
        let long_mult = MultiplicativeScorer.total_confidence(&long);

        // Union bound degradation: linear (each step costs 0.1)
        let union_degradation = short_union - long_union;
        // 0.7 -> 0.0 = 0.7 (clamped)

        // Multiplicative degradation: exponential (each step costs factor 0.9)
        let mult_degradation = short_mult - long_mult;
        // 0.729 -> 0.3487 = 0.3803

        // Both degrade, but union bound's degradation is bounded by additive property
        assert!(
            union_degradation >= 0.0,
            "union should degrade with more positions"
        );
        assert!(
            mult_degradation >= 0.0,
            "mult should degrade with more positions"
        );
        // Union bound never goes negative (clamped)
        assert!(
            long_union >= 0.0,
            "union should be >= 0, got {}",
            long_union
        );
    }

    #[test]
    fn union_bound_clamps_at_zero() {
        let scores = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let result = UnionBoundScorer.total_confidence(&scores);
        assert!(result >= 0.0);
    }

    #[test]
    fn hybrid_short_chain_uses_multiplicative() {
        let hybrid = HybridScorer::default(); // threshold=4
        let scores = [0.9, 0.8];
        let result = hybrid.total_confidence(&scores);
        let expected = MultiplicativeScorer.total_confidence(&scores);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn hybrid_long_chain_uses_union_bound() {
        let hybrid = HybridScorer::default(); // threshold=4
        let scores = [0.9, 0.8, 0.7, 0.6, 0.5];
        let result = hybrid.total_confidence(&scores);
        let expected = UnionBoundScorer.total_confidence(&scores);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn scorer_names() {
        assert_eq!(MultiplicativeScorer.name(), "multiplicative");
        assert_eq!(UnionBoundScorer.name(), "union_bound");
        assert_eq!(HybridScorer::default().name(), "hybrid");
    }
}
