//! Trait definitions for the Alien Sampler primitive.
//!
//! See [`crate::alien_sampler`] for the full module doc + paper citation.
//!
//! Both traits are generic over the atom type `V` — the primitive never
//! inspects `V`, it only hands slices of `V` to the caller-supplied scorer.
//! Typical `V` choices:
//! - `f32`: each candidate is `Vec<f32>` (a dense embedding); the scorer
//!   computes dot-products / cosines directly. This is the
//!   [`super::median_top_m::MedianTopMAvailability`] default.
//! - A richer atom type (token id + weight, KG-triple, …): the scorer
//!   implements whatever projection makes sense. This crate ships only the
//!   `V = f32` default scorer; richer `V` is the consumer's job (riir-ai
//!   Plan 312+).
//!
//! Reference: Plan 311 (T1.2), Research 293, arXiv:2603.01092 §1.4.

/// Score how **coherent** a candidate atom-set is.
///
/// Higher = more coherent (more internally consistent, more on-personality,
/// higher Guide score, …). The interpretation of "coherent" is the caller's;
/// the sampler only requires that the score be a finite `f32` and that the
/// same atom-set yields the same score across calls (determinism).
///
/// # Determinism
/// Implementations MUST be deterministic: same `atoms` ⇒ same `f32`, no RNG,
/// no thread-local state, no clock. This is required for replay / sync /
/// audit.
///
/// # NaN
/// Implementations SHOULD NOT return NaN. If they do, the sampler guards
/// downstream (NaN is treated as `-∞` for ranking, i.e. ranked last), but the
/// caller will see a non-NaN score in the output via the z-score pass — NaN
/// inputs propagate as NaN z-scores, which is usually a bug. Prefer to clamp
/// or panic at the scorer boundary.
///
/// Reference: Plan 311 (T1.2).
pub trait CoherenceScorer<V> {
    /// Score the coherence of `atoms`. Higher = more coherent.
    fn coherence(&self, atoms: &[V]) -> f32;
}

/// Score how **available** a candidate is to the reference community.
///
/// Higher = MORE available (more community support / more represented in the
/// community bank). **The sampler negates this internally** to produce an
/// "unavailability" / "alien-ness" signal: candidates that are LESS available
/// to the community rank higher (are more "alien").
///
/// This sign convention is load-bearing — the paper's `Fβ = (1−β)·zC + β·zU`
/// uses `zU` (unavailability), but the *scorer* reports availability because
/// that's the natural quantity for the community-bank implementation to
/// compute (cosine similarity against a bank = "how available is this to the
/// community"). The sampler's job is to flip the sign and z-score.
///
/// # Determinism
/// Same contract as [`CoherenceScorer`]: deterministic, no RNG, no
/// thread-local state.
///
/// # NaN
/// Same contract as [`CoherenceScorer`]: avoid returning NaN.
///
/// Reference: Plan 311 (T1.2).
pub trait AvailabilityScorer<V> {
    /// Score the availability of `atoms` to the reference community.
    /// Higher = more available (sampler negates).
    fn availability(&self, atoms: &[V]) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial coherence scorer: sums the atoms (assumes they're f32).
    /// Used by the trait-level doc test and as a reference for downstream
    /// unit tests.
    struct SumCoherence;

    impl CoherenceScorer<f32> for SumCoherence {
        fn coherence(&self, atoms: &[f32]) -> f32 {
            atoms.iter().sum()
        }
    }

    /// Trivial availability scorer: returns a constant. Useful for testing
    /// the `β=0` coherence-only path (constant availability ⇒ z-score is 0 ⇒
    /// fusion is pure coherence).
    struct ConstAvailability(f32);

    impl AvailabilityScorer<f32> for ConstAvailability {
        fn availability(&self, _atoms: &[f32]) -> f32 {
            self.0
        }
    }

    #[test]
    fn sum_coherence_works() {
        let s = SumCoherence;
        assert!((s.coherence(&[1.0, 2.0, 3.0]) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn const_availability_is_constant() {
        let a = ConstAvailability(0.42);
        assert!((a.availability(&[1.0, 2.0]) - 0.42).abs() < 1e-6);
        assert!((a.availability(&[9.9]) - 0.42).abs() < 1e-6);
    }
}
