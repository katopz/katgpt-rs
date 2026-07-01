//! `HlaProjectionGuide` — reference `QualityGuide` impl (Plan 274 T1.3).
//!
//! Score combines:
//! - Relevance: `sigmoid(λ · dot(candidate, target))`
//! - Elegance:  `sigmoid(−α · structural_complexity(candidate))`
//!
//! Final: `relevance · elegance` ∈ `[0, 1]`. Uses sigmoid only (no softmax).

use crate::cgsp::traits::QualityGuide;
use crate::cgsp::types::{sigmoid, Direction, Target};

// ── ComplexityWeights ─────────────────────────────────────────────────────

/// Generic, game-agnostic weights for the structural-complexity penalty.
///
/// Defaults `(0.4, 0.3, 03)` come from the SGS paper's "relevance × elegance
/// × non-redundancy" rubric. Higher values penalize the corresponding axis
/// more aggressively.
#[derive(Clone, Copy, Debug)]
pub struct ComplexityWeights {
    /// Penalty per disjunction-like coordinate flip (sign-change count).
    pub disjunction: f32,
    /// Penalty per unit of L1 length above `nominal_length`.
    pub length: f32,
    /// Penalty for redundancy (max coordinate magnitude relative to L2 norm).
    pub redundancy: f32,
    /// Reference length used by the length penalty. Defaults to `1.0`
    /// (unit-norm directions). Lengths above this are penalized.
    pub nominal_length: f32,
}

impl Default for ComplexityWeights {
    fn default() -> Self {
        Self {
            disjunction: 0.4,
            length: 0.3,
            redundancy: 0.3,
            nominal_length: 1.0,
        }
    }
}

// ── structural_complexity ─────────────────────────────────────────────────

/// Compute the structural complexity of a direction vector.
///
/// Generic, game-agnostic proxy for the SGS paper's "elegance" rubric. The
/// same idea — penalize candidates that are long, redundant, or full of
/// sign-changes — applies whether the direction vector encodes a theorem
/// proof structure or a game-state curiosity vector.
///
/// # Returns
///
/// Non-negative `f32`. Higher = more complex = lower elegance score.
#[inline]
pub fn structural_complexity(candidate: &Direction, weights: &ComplexityWeights) -> f32 {
    let coords = candidate.coords.as_slice();
    if coords.is_empty() {
        return 0.0;
    }
    // ── Disjunction: count of sign changes ───────────────────────────────
    let mut sign_changes = 0u32;
    let mut prev_sign: i8 = 0;
    for &c in coords {
        let s = if c > 0.0 {
            1
        } else if c < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 && prev_sign != 0 && s != prev_sign {
            sign_changes += 1;
        }
        if s != 0 {
            prev_sign = s;
        }
    }
    let disjunction_score = sign_changes as f32 / coords.len().max(1) as f32;

    // ── Length: L2 norm above nominal ────────────────────────────────────
    let l2: f32 = coords.iter().map(|c| c * c).sum::<f32>().sqrt();
    let length_excess = (l2 - weights.nominal_length).max(0.0);

    // ── Redundancy: max-coord share of L2 norm ──────────────────────────
    let max_abs = coords.iter().fold(0.0f32, |a, &c| a.max(c.abs()));
    let redundancy_ratio = if l2 > 1e-9 {
        (max_abs * max_abs) / (l2 * l2)
    } else {
        0.0
    };
    // Redundancy is high when one coordinate dominates. Subtract 1/len so
    // a uniform direction has zero redundancy penalty.
    let n = coords.len() as f32;
    let redundancy_excess = (redundancy_ratio - 1.0 / n).max(0.0);

    weights.disjunction * disjunction_score
        + weights.length * length_excess
        + weights.redundancy * redundancy_excess
}

// ── HlaProjectionGuide ────────────────────────────────────────────────────

/// Reference `QualityGuide` combining relevance + elegance.
///
/// Score = `sigmoid(λ · dot(c, t)) · sigmoid(−α · complexity(c))`.
pub struct HlaProjectionGuide {
    /// λ — relevance sharpness (higher = sharper preference for target-aligned).
    pub lambda: f32,
    /// α — complexity penalty sharpness (higher = sharper preference for simple).
    pub alpha: f32,
    /// Complexity weights.
    pub weights: ComplexityWeights,
}

impl HlaProjectionGuide {
    /// Build with `lambda` (relevance) and `alpha` (complexity penalty).
    pub fn new(lambda: f32, alpha: f32, weights: ComplexityWeights) -> Self {
        Self {
            lambda,
            alpha,
            weights,
        }
    }
}

impl QualityGuide for HlaProjectionGuide {
    #[inline]
    fn score(&self, target: &Target, candidate: &Direction) -> f32 {
        let dot = target.direction.dot(candidate);
        let relevance = sigmoid(self.lambda * dot);
        let complexity = structural_complexity(candidate, &self.weights);
        let elegance = sigmoid(-self.alpha * complexity);
        relevance * elegance
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, axis: usize) -> Direction {
        let mut coords = vec![0.0f32; dim];
        coords[axis.min(dim.saturating_sub(1))] = 1.0;
        Direction { coords }
    }

    #[test]
    fn score_is_in_unit_interval() {
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let target = Target::new(unit(4, 0));
        for axis in 0..4 {
            let candidate = unit(4, axis);
            let s = guide.score(&target, &candidate);
            assert!(s.is_finite(), "score NaN");
            assert!((0.0..=1.0).contains(&s), "score out of [0,1]: {s}");
        }
    }

    #[test]
    fn score_monotone_in_dot_product() {
        // For a fixed target, score should increase as candidate moves into
        // alignment with the target.
        let guide = HlaProjectionGuide::new(4.0, 0.0, ComplexityWeights::default());
        let target = Target::new(unit(2, 0));
        let orthogonal = unit(2, 1); // dot = 0
        let aligned = unit(2, 0); // dot = 1
        let s_orth = guide.score(&target, &orthogonal);
        let s_aligned = guide.score(&target, &aligned);
        assert!(
            s_aligned > s_orth,
            "aligned must outscore orthogonal: {s_aligned} vs {s_orth}"
        );
    }

    #[test]
    fn score_monotone_decreasing_in_complexity() {
        // For two candidates with the same dot-product, the one with higher
        // structural complexity should score lower.
        let guide = HlaProjectionGuide::new(0.0, 4.0, ComplexityWeights::default());
        let target = Target::new(unit(4, 0));

        // Simple: single unit coord.
        let simple = unit(4, 0);
        // Complex: many sign flips and high length.
        let mut complex_coords = vec![1.0f32, -1.0, 1.0, -1.0];
        // Scale up to inflate length penalty too.
        for c in complex_coords.iter_mut() {
            *c *= 2.0;
        }
        let complex = Direction { coords: complex_coords };

        let s_simple = guide.score(&target, &simple);
        let s_complex = guide.score(&target, &complex);
        assert!(
            s_simple > s_complex,
            "simple must outscore complex: {s_simple} vs {s_complex}"
        );
    }

    #[test]
    fn uses_sigmoid_not_softmax() {
        // Numerical gradient sign check: sigmoid derivative is always
        // positive in x, while softmax depends on the whole vector.
        // Verify that increasing the dot-product always increases the
        // relevance portion (with complexity held at zero).
        let guide = HlaProjectionGuide::new(2.0, 0.0, ComplexityWeights::default());
        let target = Target::new(unit(2, 0));

        let mut prev = -f32::INFINITY;
        for v in [-2.0f32, -1.0, 0.0, 1.0, 2.0] {
            let candidate = Direction { coords: vec![v, 0.0] };
            let s = guide.score(&target, &candidate);
            assert!(s > prev, "score should monotonically increase: {prev} -> {s}");
            prev = s;
        }
    }

    #[test]
    fn structural_complexity_zero_for_uniform_unit() {
        // A perfectly spread direction (all coords equal magnitude) has
        // zero sign changes, length = nominal, and zero redundancy (after
        // subtracting 1/n baseline). Build one explicitly.
        let weights = ComplexityWeights::default();
        let n = 4usize;
        let val = 1.0 / (n as f32).sqrt();
        let uniform = Direction { coords: vec![val; n] };
        let c = structural_complexity(&uniform, &weights);
        assert!(c.abs() < 1e-5, "uniform unit should have ~0 complexity, got {c}");
    }
}
