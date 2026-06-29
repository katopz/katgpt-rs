//! Deep Manifold Part 2 — Federated Boundary Alignment (Research 51, §7.6)
//!
//! Paper Eq. 163-164: Cross-model KL coupling replaces gradient exchange.
//!   q₋ᵢ(·|x) = Σⱼ≠ᵢ αᵢⱼ pθⱼ(·|x)
//!   θ*ᵢ = argmin [ℓ(θᵢ) + λ·KL(pθᵢ ‖ q₋ᵢ)]
//!
//! Each local expert aligns to the ensemble of other experts,
//! producing coherent global manifold without centralized aggregation.
//!
//! Feature-gated behind `federation` (Plan 085).

// ── Trait ─────────────────────────────────────────────────────

/// Federated boundary alignment between domain experts.
///
/// In the Deep Manifold framework, each domain expert is a local
/// manifold piece. Boundary alignment ensures these pieces form
/// a coherent global structure through KL coupling — no data exchange,
/// no privacy concern.
pub trait BoundaryAlignment: Send + Sync {
    /// Compute KL divergence between local expert and ensemble.
    ///
    /// Paper §7.6: This is the boundary misalignment measure.
    /// Lower KL = better aligned to global manifold.
    fn kl_divergence(&self, local: &[f32], ensemble: &[f32]) -> f32;

    /// Compute coupling weight for a domain relative to neighbors.
    ///
    /// Domains with higher coupling weight should prioritize alignment.
    /// Weight can be derived from bandit Q-values (high-uncertainty domains
    /// need more alignment) or domain similarity.
    fn coupling_weight(&self, domain: &str, neighbors: &[&str]) -> f32;

    /// Compute the federated boundary penalty for training.
    ///
    /// Paper Eq. 164: L_total = L_base + λ·KL(pθᵢ ‖ q₋ᵢ)
    /// This returns the λ·KL term to add to the base loss.
    fn boundary_penalty(&self, local: &[f32], ensemble: &[f32], lambda: f32) -> f32 {
        lambda * self.kl_divergence(local, ensemble)
    }
}

// ── Implementations ───────────────────────────────────────────

/// Simple KL-based boundary aligner using symmetric KL.
pub struct KlBoundaryAligner {
    /// Regularization for KL computation (prevents log(0))
    pub epsilon: f32,
}

impl Default for KlBoundaryAligner {
    fn default() -> Self {
        Self { epsilon: 1e-10 }
    }
}

impl KlBoundaryAligner {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }
}

impl BoundaryAlignment for KlBoundaryAligner {
    fn kl_divergence(&self, local: &[f32], ensemble: &[f32]) -> f32 {
        let kl_forward: f32 = local
            .iter()
            .zip(ensemble.iter())
            .map(|(l, e)| {
                let l_safe = l.max(self.epsilon);
                let e_safe = e.max(self.epsilon);
                l_safe * (l_safe / e_safe).ln()
            })
            .sum();

        let kl_reverse: f32 = ensemble
            .iter()
            .zip(local.iter())
            .map(|(e, l)| {
                let e_safe = e.max(self.epsilon);
                let l_safe = l.max(self.epsilon);
                e_safe * (e_safe / l_safe).ln()
            })
            .sum();

        // Symmetric KL (Jensen-Shannon proxy)
        (kl_forward + kl_reverse) / 2.0
    }

    fn coupling_weight(&self, domain: &str, _neighbors: &[&str]) -> f32 {
        // Default: uniform coupling. Domain-specific weights can be
        // learned from bandit Q-values in a real deployment.
        let _ = domain;
        1.0
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
    fn kl_divergence_identical_distributions_is_zero() {
        let aligner = KlBoundaryAligner::default();
        let p = vec![0.25_f32, 0.25, 0.25, 0.25];
        let kl = aligner.kl_divergence(&p, &p);
        assert!(
            approx_eq(kl, 0.0, 1e-6),
            "identical distributions should have KL=0, got {kl}"
        );
    }

    #[test]
    fn kl_divergence_symmetric() {
        let aligner = KlBoundaryAligner::default();
        let p = vec![0.5_f32, 0.5];
        let q = vec![0.3_f32, 0.7];
        let kl_pq = aligner.kl_divergence(&p, &q);
        let kl_qp = aligner.kl_divergence(&q, &p);
        // Symmetric KL should be identical in both directions
        assert!(
            approx_eq(kl_pq, kl_qp, 1e-6),
            "symmetric KL should be equal: KL(p||q)={kl_pq} vs KL(q||p)={kl_qp}"
        );
    }

    #[test]
    fn kl_divergence_non_negative() {
        let aligner = KlBoundaryAligner::default();
        let p = vec![0.1_f32, 0.4, 0.3, 0.2];
        let q = vec![0.25_f32, 0.25, 0.25, 0.25];
        let kl = aligner.kl_divergence(&p, &q);
        assert!(kl >= 0.0, "KL divergence must be non-negative, got {kl}");
    }

    #[test]
    fn boundary_penalty_scales_with_lambda() {
        let aligner = KlBoundaryAligner::default();
        let local = vec![0.5_f32, 0.5];
        let ensemble = vec![0.3_f32, 0.7];

        let penalty_1 = aligner.boundary_penalty(&local, &ensemble, 1.0);
        let penalty_2 = aligner.boundary_penalty(&local, &ensemble, 2.0);

        assert!(
            approx_eq(penalty_2, 2.0 * penalty_1, 1e-5),
            "penalty should scale linearly with lambda: {penalty_2} vs 2*{penalty_1}"
        );
    }

    #[test]
    fn coupling_weight_default_is_one() {
        let aligner = KlBoundaryAligner::default();
        let w = aligner.coupling_weight("bomber", &["go", "fft"]);
        assert!(
            approx_eq(w, 1.0, 1e-6),
            "default coupling weight should be 1.0, got {w}"
        );
    }

    #[test]
    fn kl_divergence_handles_zeros() {
        let aligner = KlBoundaryAligner::default();
        let p = vec![0.0_f32, 1.0];
        let q = vec![0.5_f32, 0.5];
        // Should not panic — epsilon handles zeros
        let kl = aligner.kl_divergence(&p, &q);
        assert!(
            kl.is_finite(),
            "KL should be finite with zero entries, got {kl}"
        );
    }
}
