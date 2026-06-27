//! Predictability scoring (Plan 334 Phase 1 T1.3).
//!
//! `PredictabilityScorer` answers: "given context `c`, how predictable is a
//! query of class `dir`?" Returns `p ∈ [0,1]`. Higher `p` = more predictable
//! = more sleep-time compute warranted for this direction.
//!
//! # Why a trait, not a function
//!
//! The "right" predictability measure is domain-dependent:
//! - **Default (this file)**: `p = sigmoid(α · dot(c, dir) + β)` — a baseline
//!   that captures "alignment between context and query direction".
//! - **Curiosity inversion (riir-ai Plan 341)**: `p = 1 − sigmoid(curiosity)`
//!   where curiosity = `||c − karc_forecast(c)||`. High-curiosity contexts
//!   are *un*predictable, so they should NOT get sleep-time compute.
//!
//! The trait lets riir-ai swap scorers without touching the anticipator.
//!
//! # Sigmoid not softmax (AGENTS.md)
//!
//! Every gate in this module is a single-scalar `sigmoid`. There is no
//! `softmax` symbol — predictability is per-direction, not a distribution
//! over directions.

use crate::simd::{fast_sigmoid, simd_dot_f32};
use crate::sleep_time::types::AnticipatedQueryDir;

/// Scores how predictable a query of class `dir` is from context `c`.
///
/// Returns `p ∈ [0,1]`. Higher = more predictable = more sleep-time compute
/// warranted.
///
/// Implementations MUST be:
/// - **Modelless** — no training, no backprop. Closed-form algebra only.
/// - **Deterministic** — same `(c, dir)` always yields the same `p`.
/// - **Zero-allocation** — the wake-time path scans K scorers per query; any
///   allocation here breaks the G5 zero-alloc gate.
pub trait PredictabilityScorer<const D: usize> {
    fn predictability(&self, c: &[f32; D], dir: &AnticipatedQueryDir<D>) -> f32;
}

/// Default scorer: `p = sigmoid(α · dot(c, dir) + β)`.
///
/// - `alpha` = sharpness (higher = sharper gate).
/// - `beta` = bias (higher = pre-compute by default).
///
/// This is a baseline, not a claim — the trait lets consumers swap in any
/// scorer (curiosity inversion, KL-divergence, etc.) without touching the
/// anticipator.
#[derive(Clone, Copy, Debug)]
pub struct DotPredictabilityScorer {
    /// Sharpness. Paper-style default: α = 1.0.
    pub alpha: f32,
    /// Bias. Paper-style default: β = 0.0 (pre-compute iff dot > 0).
    pub beta: f32,
}

impl Default for DotPredictabilityScorer {
    #[inline]
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

impl DotPredictabilityScorer {
    /// Construct with explicit α, β.
    #[inline]
    pub const fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }
}

impl<const D: usize> PredictabilityScorer<D> for DotPredictabilityScorer {
    #[inline]
    fn predictability(&self, c: &[f32; D], dir: &AnticipatedQueryDir<D>) -> f32 {
        // simd_dot_f32 over fixed-size arrays — zero-alloc, vectorizable.
        let dot = simd_dot_f32(c, &dir.direction, D);
        fast_sigmoid(self.alpha * dot + self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictability_is_in_unit_interval() {
        let s = DotPredictabilityScorer::new(2.0, -1.0);
        let dir = AnticipatedQueryDir::new([1.0; 4]);
        // Sweep c across a wide range — p must stay in [0,1].
        for &scale in &[-10.0f32, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let c = [scale; 4];
            let p = s.predictability(&c, &dir);
            assert!(
                (0.0..=1.0).contains(&p),
                "p={} out of [0,1] at scale {}",
                p,
                scale
            );
        }
    }

    #[test]
    fn aligned_context_is_more_predictable() {
        let s = DotPredictabilityScorer::default();
        let dir = AnticipatedQueryDir::new([1.0, 0.0, 0.0, 0.0]);
        let aligned = [1.0, 0.0, 0.0, 0.0];
        let orthogonal = [0.0, 1.0, 0.0, 0.0];
        let opposite = [-1.0, 0.0, 0.0, 0.0];
        let p_aligned = s.predictability(&aligned, &dir);
        let p_ortho = s.predictability(&orthogonal, &dir);
        let p_opp = s.predictability(&opposite, &dir);
        // sigmoid monotone in dot product → aligned > orthogonal > opposite.
        assert!(
            p_aligned > p_ortho,
            "aligned ({}) should beat orthogonal ({})",
            p_aligned,
            p_ortho
        );
        assert!(
            p_ortho > p_opp,
            "orthogonal ({}) should beat opposite ({})",
            p_ortho,
            p_opp
        );
    }

    #[test]
    fn predictability_is_deterministic() {
        let s = DotPredictabilityScorer::default();
        let dir = AnticipatedQueryDir::new([0.5, 0.5, 0.5, 0.5]);
        let c = [1.0; 4];
        let p1 = s.predictability(&c, &dir);
        let p2 = s.predictability(&c, &dir);
        assert_eq!(p1.to_bits(), p2.to_bits(), "deterministic");
    }

    #[test]
    fn beta_shifts_the_gate_threshold() {
        let dir = AnticipatedQueryDir::new([1.0, 0.0]);
        let c = [0.0, 0.0]; // dot = 0
        let s_neutral = DotPredictabilityScorer::new(1.0, 0.0);
        let s_precompute = DotPredictabilityScorer::new(1.0, 2.0);
        let p_neutral = s_neutral.predictability(&c, &dir);
        let p_precompute = s_precompute.predictability(&c, &dir);
        // sigmoid(0) = 0.5; sigmoid(2) ≈ 0.88.
        assert!((p_neutral - 0.5).abs() < 1e-6);
        assert!(p_precompute > 0.8, "beta=2 should push p well above 0.8");
    }
}
