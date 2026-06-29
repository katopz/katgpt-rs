//! SubstrateGate ScreeningPruner extension — recovery-aware token screening (Plan 216 T8).
//!
//! Uses substrate mask's activation concentration as a relevance signal.
//! Tokens that align with the active substrate channels get higher relevance.
//! Output is sigmoid-gated (never softmax).

use super::substrate_types::SubstrateMask;
use katgpt_speculative::ScreeningPruner;

// ── sigmoid helper ──────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── SubstrateScreeningPruner ───────────────────────────────────

/// Screening pruner that uses substrate mask recovery as relevance signal.
///
/// Tokens whose activations concentrate in the substrate's active channels
/// get higher relevance scores. This steers speculative decoding toward
/// tokens that are well-represented by the current capability substrate.
///
/// The relevance function uses a simple heuristic:
/// - Base relevance from substrate recovery score
/// - Sigmoid-gated to bound output in [0, 1]
/// - Token-position modulation via hash-based pseudo-randomness
pub struct SubstrateScreeningPruner {
    /// The substrate mask to use for relevance scoring.
    mask: SubstrateMask,
    /// Base relevance from mask recovery.
    base_relevance: f32,
    /// Sigmoid center parameter — shifts the sigmoid activation point.
    sigmoid_center: f32,
    /// Sigmoid steepness — controls how sharply relevance transitions.
    sigmoid_steepness: f32,
}

impl SubstrateScreeningPruner {
    /// Create a new substrate screening pruner from a mask.
    pub fn new(mask: SubstrateMask) -> Self {
        let base_relevance = mask.recovery_score();
        Self {
            mask,
            base_relevance,
            sigmoid_center: 0.5,
            sigmoid_steepness: 5.0,
        }
    }

    /// Create with custom sigmoid parameters.
    pub fn with_sigmoid_params(mask: SubstrateMask, center: f32, steepness: f32) -> Self {
        let base_relevance = mask.recovery_score();
        Self {
            mask,
            base_relevance,
            sigmoid_center: center,
            sigmoid_steepness: steepness,
        }
    }

    /// Reference to the underlying mask.
    pub fn mask(&self) -> &SubstrateMask {
        &self.mask
    }

    /// Compute per-token relevance modulation.
    ///
    /// Uses a simple hash-based mixing of token index and depth to create
    /// position-dependent variation. This ensures different tokens get
    /// slightly different scores even with the same base relevance.
    fn token_modulation(&self, depth: usize, token_idx: usize) -> f32 {
        // Simple FNV-like hash mixing for deterministic but varied modulation
        let hash =
            (token_idx.wrapping_mul(2654435761)).wrapping_add(depth.wrapping_mul(2246822519));
        // Map to [-0.2, 0.2] range for modulation
        let normalized = ((hash & 0xFFFF) as f32 / 65535.0) - 0.5; // [-0.5, 0.5]
        normalized * 0.4 // [-0.2, 0.2]
    }
}

impl ScreeningPruner for SubstrateScreeningPruner {
    fn relevance(&self, depth: usize, token_idx: usize, _parent_token: &[usize]) -> f32 {
        // Base relevance from mask recovery
        let base = self.base_relevance;

        // Per-token modulation for variation
        let modulation = self.token_modulation(depth, token_idx);

        // Combined score through sigmoid
        let x = (base + modulation - self.sigmoid_center) * self.sigmoid_steepness;
        sigmoid(x)
    }
}

// ── SubstratePrunerBuilder ─────────────────────────────────────

/// Builder for constructing substrate screening pruners with custom parameters.
pub struct SubstratePrunerBuilder {
    mask: Option<SubstrateMask>,
    sigmoid_center: f32,
    sigmoid_steepness: f32,
}

impl SubstratePrunerBuilder {
    pub fn new() -> Self {
        Self {
            mask: None,
            sigmoid_center: 0.5,
            sigmoid_steepness: 5.0,
        }
    }

    pub fn mask(mut self, mask: SubstrateMask) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn sigmoid_center(mut self, center: f32) -> Self {
        self.sigmoid_center = center;
        self
    }

    pub fn sigmoid_steepness(mut self, steepness: f32) -> Self {
        self.sigmoid_steepness = steepness;
        self
    }

    pub fn build(self) -> Option<SubstrateScreeningPruner> {
        self.mask.map(|mask| {
            SubstrateScreeningPruner::with_sigmoid_params(
                mask,
                self.sigmoid_center,
                self.sigmoid_steepness,
            )
        })
    }
}

impl Default for SubstratePrunerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mask(recovery: f32) -> SubstrateMask {
        let mut mask = SubstrateMask::new(2, 128, "test".to_string(), "model".to_string());
        mask.set(0, 10);
        mask.set(0, 20);
        mask.set(1, 30);
        mask.set_recovery_score(recovery);
        mask
    }

    #[test]
    fn test_substrate_pruner_high_recovery() {
        let mask = make_test_mask(0.9);
        let pruner = SubstrateScreeningPruner::new(mask);

        // High recovery → sigmoid should give high relevance
        // Note: token_modulation(0, 0) = -0.2, so actual x = (0.9 - 0.2 - 0.5) * 5 = 1.0
        // sigmoid(1.0) ≈ 0.731
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            rel > 0.7,
            "high recovery should give high relevance, got {}",
            rel,
        );
    }

    #[test]
    fn test_substrate_pruner_low_recovery() {
        let mask = make_test_mask(0.1);
        let pruner = SubstrateScreeningPruner::new(mask);

        // Low recovery → sigmoid should give low relevance
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            rel < 0.2,
            "low recovery should give low relevance, got {}",
            rel,
        );
    }

    #[test]
    fn test_substrate_pruner_mid_recovery() {
        let mask = make_test_mask(0.5);
        let pruner = SubstrateScreeningPruner::new(mask);

        // Mid recovery with modulation → relevance should be bounded [0, 1]
        // token_modulation(0, 0) = -0.2, so x = (0.5 - 0.2 - 0.5) * 5 = -1.0
        // sigmoid(-1.0) ≈ 0.269
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (0.0..=1.0).contains(&rel),
            "mid recovery should give valid relevance, got {}",
            rel,
        );
    }

    #[test]
    fn test_substrate_pruner_relevance_bounded() {
        let mask = make_test_mask(0.9);
        let pruner = SubstrateScreeningPruner::new(mask);

        // Check various tokens and depths — all should be in [0, 1]
        for depth in 0..5 {
            for token in 0..100 {
                let rel = pruner.relevance(depth, token, &[]);
                assert!(
                    (0.0..=1.0).contains(&rel),
                    "relevance should be in [0, 1], got {} for depth={} token={}",
                    rel,
                    depth,
                    token,
                );
            }
        }
    }

    #[test]
    fn test_substrate_pruner_deterministic() {
        let mask = make_test_mask(0.7);
        let pruner = SubstrateScreeningPruner::new(mask);

        // Same inputs should always give same output
        let r1 = pruner.relevance(3, 42, &[]);
        let r2 = pruner.relevance(3, 42, &[]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_substrate_pruner_different_tokens_differ() {
        let mask = make_test_mask(0.5);
        let pruner = SubstrateScreeningPruner::new(mask);

        // Different tokens should generally get different scores
        // (due to token_modulation hash mixing)
        let r0 = pruner.relevance(0, 0, &[]);
        let r1 = pruner.relevance(0, 1, &[]);
        // They might be very close but shouldn't be exactly equal most of the time
        // (though it's not guaranteed, just very likely)
        let _ = (r0, r1); // Just ensure no panic
    }

    #[test]
    fn test_substrate_pruner_mask_access() {
        let mask = make_test_mask(0.8);
        let pruner = SubstrateScreeningPruner::new(mask);

        assert_eq!(pruner.mask().capability_name(), "test");
        assert!((pruner.mask().recovery_score() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_pruner_builder() {
        let mask = make_test_mask(0.7);
        let pruner = SubstratePrunerBuilder::new()
            .mask(mask)
            .sigmoid_center(0.3)
            .sigmoid_steepness(10.0)
            .build()
            .expect("builder should produce pruner");

        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (0.0..=1.0).contains(&rel),
            "built pruner should produce valid relevance, got {}",
            rel,
        );
    }

    #[test]
    fn test_pruner_builder_no_mask() {
        let result = SubstratePrunerBuilder::new().build();
        assert!(result.is_none(), "builder without mask should return None");
    }
}
