//! Bregman potentials for the primal-dual iterator (Plan 295 Phase 2).
//!
//! The primal step of `CcePrimalDual` solves:
//!
//! ```text
//! ρⁿ⁺¹ = argmin_{ρ ∈ Δ}  Γ₀(ρ) + λⁿ · ER(ρ) + (1/η) · Dψ(ρ, ρⁿ)
//! ```
//!
//! where `Dψ(ρ, σ) = ψ(ρ) − ψ(σ) − ⟨∇ψ(σ), ρ − σ⟩` is the Bregman divergence
//! induced by potential `ψ`. Different choices of `ψ` give different update
//! rules:
//!
//! - `Euclidean` (`ψ = ½‖·‖²`): projected gradient descent.
//! - `Kl` (`ψ = Σ ρ log ρ`): entropic mirror descent (Phase 3 follow-up).

use crate::cce::types::OccupationMeasure;

/// Bregman potential — defines the proximal geometry for the primal update.
pub trait BregmanPotential<const N: usize, const A: usize> {
    /// Bregman divergence `Dψ(ρ, σ) = ψ(ρ) − ψ(σ) − ⟨∇ψ(σ), ρ − σ⟩`.
    fn divergence(&self, rho: &OccupationMeasure<N, A>, sigma: &OccupationMeasure<N, A>) -> f32;

    /// Gradient of the potential `∇ψ(ρ)`. Length = `N·A`.
    fn gradient(&self, rho: &OccupationMeasure<N, A>) -> Vec<f32>;
}

/// Euclidean potential `ψ(ρ) = ½·‖ρ‖²`. Gives projected gradient descent.
///
/// `Dψ(ρ, σ) = ½·‖ρ − σ‖²`, `∇ψ(ρ) = ρ`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Euclidean;

impl<const N: usize, const A: usize> BregmanPotential<N, A> for Euclidean {
    fn divergence(
        &self,
        rho: &OccupationMeasure<N, A>,
        sigma: &OccupationMeasure<N, A>,
    ) -> f32 {
        rho.entries
            .iter()
            .zip(sigma.entries.iter())
            .map(|(&r, &s)| {
                let d = r - s;
                0.5 * d * d
            })
            .sum()
    }

    fn gradient(&self, rho: &OccupationMeasure<N, A>) -> Vec<f32> {
        rho.entries.clone()
    }
}

/// KL-divergence potential `ψ(ρ) = Σ ρ log ρ`. Gives entropic mirror descent.
///
/// `Dψ(ρ, σ) = Σ ρ · log(ρ/σ)` (generalized KL). Not yet wired into the
/// primal-dual iterator; reserved for Phase 3 follow-up.
#[derive(Debug, Default, Clone, Copy)]
pub struct Kl;

impl<const N: usize, const A: usize> BregmanPotential<N, A> for Kl {
    fn divergence(
        &self,
        rho: &OccupationMeasure<N, A>,
        sigma: &OccupationMeasure<N, A>,
    ) -> f32 {
        let mut d = 0.0;
        for (&r, &s) in rho.entries.iter().zip(sigma.entries.iter()) {
            if r > 1e-12 {
                d += r * (r / s.max(1e-12)).ln();
            }
        }
        d
    }

    fn gradient(&self, rho: &OccupationMeasure<N, A>) -> Vec<f32> {
        // ∇ψ(ρ) = log(ρ) + 1.
        rho.entries
            .iter()
            .map(|&v| v.max(1e-12).ln() + 1.0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_divergence_is_half_squared_norm() {
        let rho = OccupationMeasure::<2, 2>::from_entries_trusted(vec![0.4, 0.1, 0.3, 0.2]);
        let sigma = OccupationMeasure::<2, 2>::from_entries_trusted(vec![0.25, 0.25, 0.25, 0.25]);
        let d = Euclidean.divergence(&rho, &sigma);
        // ½ · [(0.15)² + (-0.15)² + (0.05)² + (-0.05)²]
        //   = ½ · [0.0225 + 0.0225 + 0.0025 + 0.0025] = ½ · 0.05 = 0.025.
        assert!((d - 0.025).abs() < 1e-6, "got {d}");
    }

    #[test]
    fn euclidean_gradient_is_identity() {
        let rho = OccupationMeasure::<2, 1>::from_entries_trusted(vec![0.7, 0.3]);
        let g = Euclidean.gradient(&rho);
        assert_eq!(g, vec![0.7, 0.3]);
    }

    #[test]
    fn euclidean_divergence_zero_on_self() {
        let rho = OccupationMeasure::<3, 1>::from_entries_trusted(vec![0.5, 0.3, 0.2]);
        assert!(Euclidean.divergence(&rho, &rho).abs() < 1e-9);
    }

    #[test]
    fn kl_divergence_nonneg_and_zero_on_self() {
        let rho = OccupationMeasure::<2, 1>::from_entries_trusted(vec![0.6, 0.4]);
        assert!(Kl.divergence(&rho, &rho).abs() < 1e-5);

        let sigma = OccupationMeasure::<2, 1>::from_entries_trusted(vec![0.5, 0.5]);
        let d = Kl.divergence(&rho, &sigma);
        // 0.6·ln(0.6/0.5) + 0.4·ln(0.4/0.5) = 0.6·ln(1.2) + 0.4·ln(0.8)
        //   ≈ 0.6·0.18232 + 0.4·(-0.22314) ≈ 0.10939 - 0.08926 ≈ 0.02014.
        assert!(d > 0.0, "KL should be positive, got {d}");
        assert!((d - 0.02014).abs() < 1e-3, "got {d}");
    }
}
