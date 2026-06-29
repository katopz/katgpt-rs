//! Deep Manifold Part 2 — Fixed-Point Residual Scoring (Research 51)
//!
//! Paper Eq. 23: f(x) - x = e(x), minimize e(x)
//! Our HintDelta already computes this as log-prob shift.
//! This trait makes residual tracking explicit and composable.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "deep_manifold")]`.
//! Feature: `deep_manifold = []` in `Cargo.toml`.

// ── Trait ─────────────────────────────────────────────────────

/// Fixed-point residual scorer for candidate evaluation.
///
/// In the Deep Manifold framework, inference is boundary-conditioned
/// fixed-point iteration on stacked piecewise manifolds. The residual
/// ‖f(x) - x‖ measures distance from equilibrium — how far a candidate
/// is from its stable fixed point.
///
/// Our HintDelta (G-Zero Plan 049) instantiates this:
///   δ = (1/T) Σ [log πG(at|q,h,a<t) - log πG(at|q,a<t)]
/// where δ ≈ 0 means the generator is at equilibrium.
pub trait ManifoldResidual: Send + Sync {
    /// Compute fixed-point residual between candidate and base logits.
    ///
    /// Returns ‖candidate - base‖ — the distance from the base
    /// distribution's equilibrium. Lower = closer to fixed point.
    fn residual(&self, candidate: &[f32], base: &[f32]) -> f32;

    /// Check if residual is below convergence threshold.
    ///
    /// Paper §2.5.1 Eq. 50: convergence when ρ(J_fi) < 1 and sup‖δt‖ < ∞.
    /// We approximate with L2 residual < tolerance.
    fn is_converged(&self, residual: f32, tolerance: f32) -> bool {
        residual < tolerance
    }

    /// Compute per-position residuals for fine-grained analysis.
    ///
    /// Useful for identifying which tokens are far from equilibrium
    /// vs which have already converged (intrinsic pathway analysis).
    fn per_position_residual(&self, candidate: &[f32], base: &[f32]) -> Vec<f32> {
        candidate
            .iter()
            .zip(base.iter())
            .map(|(c, b)| (c - b).powi(2))
            .collect()
    }
}

// ── L2 Residual Scorer ────────────────────────────────────────

/// L2 norm residual scorer — standard Euclidean distance.
///
/// Paper §2.3.2: Lagrangian energy E(θ) = ∫ ‖fθ(x) - x‖² dμ
pub struct L2ResidualScorer {
    /// Convergence tolerance (default: 1e-4, matching Attractor paper ε)
    pub tolerance: f32,
}

impl Default for L2ResidualScorer {
    fn default() -> Self {
        Self { tolerance: 1e-4 }
    }
}

impl ManifoldResidual for L2ResidualScorer {
    fn residual(&self, candidate: &[f32], base: &[f32]) -> f32 {
        let sum_sq: f32 = candidate
            .iter()
            .zip(base.iter())
            .map(|(c, b)| (c - b).powi(2))
            .sum();
        sum_sq.sqrt()
    }

    fn is_converged(&self, residual: f32, tolerance: f32) -> bool {
        residual < tolerance
    }
}

// ── KL Residual Scorer ────────────────────────────────────────

/// KL-divergence residual scorer — distributional distance.
///
/// For probability distributions (after softmax), KL divergence
/// measures how much information is lost when using candidate
/// to approximate base. This is the paper's §7.6 KL coupling.
pub struct KlResidualScorer {
    pub tolerance: f32,
}

impl Default for KlResidualScorer {
    fn default() -> Self {
        Self { tolerance: 0.01 }
    }
}

impl ManifoldResidual for KlResidualScorer {
    fn residual(&self, candidate: &[f32], base: &[f32]) -> f32 {
        candidate
            .iter()
            .zip(base.iter())
            .filter(|(_, b)| **b > 1e-10)
            .map(|(c, b)| {
                let c_safe = c.max(1e-10);
                c_safe * (c_safe / b).ln()
            })
            .sum()
    }
}

// ── Composite Scorer ──────────────────────────────────────────

/// Composite scorer combining residual with relevance.
///
/// Paper §5.5 Learning Triangle:
///   Composite = Φ_arch ∘ ∂Ω_train ∘ M_data
///
/// This combines manifold residual (architecture quality)
/// with ScreeningPruner relevance (domain fitness).
pub struct ResidualRelevanceScorer<R: ManifoldResidual> {
    pub residual_scorer: R,
    /// Weight for residual vs relevance (0.0 = pure relevance, 1.0 = pure residual)
    pub residual_weight: f32,
}

impl<R: ManifoldResidual> ResidualRelevanceScorer<R> {
    /// Create a new composite scorer with the given residual scorer and weight.
    pub fn new(residual_scorer: R, residual_weight: f32) -> Self {
        Self {
            residual_scorer,
            residual_weight,
        }
    }

    /// Compute blended score from residual and relevance.
    ///
    /// Low residual (near fixed-point) → high normalized score.
    /// Combined with relevance using configurable weight.
    pub fn blended_score(&self, residual: f32, relevance: f32) -> f32 {
        // Invert: low residual = high score
        let normalized_residual = 1.0 / (1.0 + residual);
        let w = self.residual_weight;
        w * normalized_residual + (1.0 - w) * relevance
    }

    /// Compute blended score directly from candidate/base logits and relevance.
    pub fn score(&self, candidate: &[f32], base: &[f32], relevance: f32) -> f32 {
        let residual = self.residual_scorer.residual(candidate, base);
        self.blended_score(residual, relevance)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn l2_residual_identical_vectors() {
        let scorer = L2ResidualScorer::default();
        let v = vec![1.0, 2.0, 3.0];
        let residual = scorer.residual(&v, &v);
        assert!(
            approx_eq(residual, 0.0, 1e-6),
            "identical vectors should have zero residual, got {residual}"
        );
    }

    #[test]
    fn l2_residual_known_distance() {
        let scorer = L2ResidualScorer::default();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        // ‖a - b‖ = ‖(1,-1,0)‖ = sqrt(2)
        let residual = scorer.residual(&a, &b);
        assert!(
            approx_eq(residual, 2.0f32.sqrt(), 1e-4),
            "expected sqrt(2), got {residual}"
        );
    }

    #[test]
    fn l2_is_converged() {
        let scorer = L2ResidualScorer { tolerance: 0.01 };
        assert!(scorer.is_converged(0.001, scorer.tolerance));
        assert!(!scorer.is_converged(0.1, scorer.tolerance));
    }

    #[test]
    fn l2_per_position_residual() {
        let scorer = L2ResidualScorer::default();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 3.0, 1.0];
        let pp = scorer.per_position_residual(&a, &b);
        assert_eq!(pp.len(), 3);
        assert!(approx_eq(pp[0], 0.0, 1e-6), "pos 0: identical");
        assert!(approx_eq(pp[1], 1.0, 1e-6), "pos 1: (2-3)^2 = 1");
        assert!(approx_eq(pp[2], 4.0, 1e-6), "pos 2: (3-1)^2 = 4");
    }

    #[test]
    fn kl_residual_identical_distributions() {
        let scorer = KlResidualScorer::default();
        let v = vec![0.25, 0.25, 0.25, 0.25];
        let residual = scorer.residual(&v, &v);
        assert!(
            approx_eq(residual, 0.0, 1e-6),
            "identical distributions should have zero KL, got {residual}"
        );
    }

    #[test]
    fn kl_residual_positive_for_different_distributions() {
        let scorer = KlResidualScorer::default();
        let a = vec![0.5, 0.5];
        let b = vec![0.9, 0.1];
        let residual = scorer.residual(&a, &b);
        assert!(
            residual > 0.0,
            "different distributions should have positive KL, got {residual}"
        );
    }

    #[test]
    fn blended_score_pure_relevance() {
        let scorer = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 0.0);
        let score = scorer.blended_score(10.0, 0.8);
        assert!(
            approx_eq(score, 0.8, 1e-6),
            "pure relevance weight should return relevance, got {score}"
        );
    }

    #[test]
    fn blended_score_pure_residual() {
        let scorer = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 1.0);
        let score = scorer.blended_score(0.0, 0.0);
        // residual=0 → normalized=1/(1+0)=1.0, weight=1.0
        assert!(
            approx_eq(score, 1.0, 1e-6),
            "zero residual pure weight should give 1.0, got {score}"
        );
    }

    #[test]
    fn blended_score_balanced() {
        let scorer = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 0.5);
        // residual=1.0 → normalized=1/(1+1)=0.5
        // 0.5*0.5 + 0.5*0.8 = 0.25 + 0.40 = 0.65
        let score = scorer.blended_score(1.0, 0.8);
        assert!(
            approx_eq(score, 0.65, 1e-4),
            "balanced blend should be 0.65, got {score}"
        );
    }

    #[test]
    fn composite_score_end_to_end() {
        let scorer = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 0.5);
        let candidate = vec![0.5, 0.5];
        let base = vec![0.5, 0.5];
        let score = scorer.score(&candidate, &base, 0.9);
        // residual=0 → normalized=1.0, blend=0.5*1.0 + 0.5*0.9 = 0.95
        assert!(
            approx_eq(score, 0.95, 1e-4),
            "identical candidate/base should blend with relevance, got {score}"
        );
    }
}
