//! Entropy-bifurcated screening pruner — direction-adaptive credit (Plan 184).
//!
//! High-entropy "forking" tokens get relaxed screening; low-entropy "scaffolding"
//! tokens get tight screening. This wrapper routes screening through entropy-aware
//! relaxation without requiring model access — purely based on top-1 probability.
//!
//! **Feature gate:** `directional_credit`

use crate::speculative::types::ScreeningPruner;

/// Entropy state for the current decoding position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EntropyState {
    /// Low entropy — tight screening (use inner relevance as-is).
    #[default]
    Scaffolding,
    /// High entropy — relaxed screening (scale relevance up).
    Forking,
}

/// Wrapper that routes screening through entropy-aware relaxation.
///
/// When the top-1 probability from marginals is below `top1_threshold` (high entropy = forking),
/// relevance is scaled by `(1.0 + relax_factor)` (> 1.0 = more permissive).
/// When top-1 probability is above threshold (low entropy = scaffolding),
/// relevance is used as-is (tight screening).
///
/// # Usage
///
/// ```ignore
/// use katgpt_rs::pruners::EntropyBifurcatedPruner;
/// use katgpt_rs::speculative::types::ScreeningPruner;
///
/// let inner = MyPruner::new();
/// let mut pruner = EntropyBifurcatedPruner::new(inner, 0.5, 0.3);
///
/// // Before relevance queries, update entropy state from top-1 prob:
/// pruner.update_entropy(0.2); // low top-1 → Forking (relaxed)
/// let rel = pruner.relevance(0, 0, &[]); // scaled up
///
/// pruner.update_entropy(0.9); // high top-1 → Scaffolding (tight)
/// let rel = pruner.relevance(0, 0, &[]); // as-is
/// ```
#[derive(Debug, Clone)]
pub struct EntropyBifurcatedPruner<P: ScreeningPruner> {
    /// Inner pruner that provides base relevance.
    pub inner: P,
    /// Top-1 probability threshold. Below = "fork" (relaxed). Default: 0.5.
    pub top1_threshold: f32,
    /// Relaxation factor applied at forks. Default: 0.3 (scales relevance UP).
    pub relax_factor: f32,
    /// Current entropy state — updated via `update_entropy()`.
    state: EntropyState,
}

impl<P: ScreeningPruner> EntropyBifurcatedPruner<P> {
    /// Create a new entropy-bifurcated pruner.
    ///
    /// # Arguments
    /// * `inner` — Base pruner providing relevance scores
    /// * `top1_threshold` — Top-1 probability below which tokens are considered "forking"
    /// * `relax_factor` — Scale factor applied to relevance at fork points (relevance *= 1.0 + relax_factor)
    pub fn new(inner: P, top1_threshold: f32, relax_factor: f32) -> Self {
        Self {
            inner,
            top1_threshold,
            relax_factor,
            state: EntropyState::Scaffolding,
        }
    }

    /// Update entropy state from top-1 probability of current position.
    ///
    /// Call before relevance queries for each token position.
    pub fn update_entropy(&mut self, top1_prob: f32) {
        self.state = match top1_prob < self.top1_threshold {
            true => EntropyState::Forking,
            false => EntropyState::Scaffolding,
        };
    }

    /// Get the current entropy state.
    pub fn state(&self) -> EntropyState {
        self.state
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Access the inner pruner mutably.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
}

impl<P: ScreeningPruner> ScreeningPruner for EntropyBifurcatedPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let base = self.inner.relevance(depth, token_idx, parent_tokens);
        match self.state {
            EntropyState::Scaffolding => base,
            EntropyState::Forking => base * (1.0 + self.relax_factor),
        }
    }
}

/// Tracks DDTree top-1 changes from parent → child nodes.
/// "Self-driven" tokens (top-1 changed from parent) get exploration bonus.
/// This is the RLRT signal: tokens that diverge from parent's choice are
/// exploring new territory, so they should get bonus exploration budget.
#[cfg(feature = "directional_credit")]
#[derive(Debug, Clone)]
pub struct SelfDrivenTokenTracker {
    /// Parent top-1 token per depth.
    parent_top1: Vec<usize>,
    /// Whether the current position was self-driven (top-1 changed).
    self_driven: Vec<bool>,
    /// Exploration bonus multiplier for self-driven tokens.
    pub exploration_bonus: f32,
}

#[cfg(feature = "directional_credit")]
impl SelfDrivenTokenTracker {
    pub fn new(max_depth: usize, exploration_bonus: f32) -> Self {
        Self {
            parent_top1: vec![0; max_depth],
            self_driven: vec![false; max_depth],
            exploration_bonus,
        }
    }

    /// Record parent's top-1 choice at depth.
    pub fn record_parent(&mut self, depth: usize, top1_token: usize) {
        if depth < self.parent_top1.len() {
            self.parent_top1[depth] = top1_token;
        }
    }

    /// Check if child's top-1 differs from parent's at depth.
    /// Returns true if self-driven (top-1 changed).
    pub fn check_self_driven(&mut self, depth: usize, child_top1: usize) -> bool {
        if depth < self.parent_top1.len() && depth < self.self_driven.len() {
            let driven = child_top1 != self.parent_top1[depth];
            self.self_driven[depth] = driven;
            driven
        } else {
            false
        }
    }

    /// Get exploration bonus if self-driven at depth.
    #[inline]
    pub fn bonus(&self, depth: usize) -> f32 {
        if depth < self.self_driven.len() && self.self_driven[depth] {
            self.exploration_bonus
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple pruner that returns a fixed relevance.
    #[derive(Debug, Clone)]
    struct FixedPruner {
        relevance_val: f32,
    }

    impl ScreeningPruner for FixedPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.relevance_val
        }
    }

    fn make_pruner(val: f32, threshold: f32, relax: f32) -> EntropyBifurcatedPruner<FixedPruner> {
        EntropyBifurcatedPruner::new(FixedPruner { relevance_val: val }, threshold, relax)
    }

    // G1: Low top-1 prob returns higher relevance than raw inner
    #[test]
    fn test_forking_scales_relevance_up() {
        let mut pruner = make_pruner(0.5, 0.5, 0.3);
        pruner.update_entropy(0.2); // below threshold → Forking
        let rel = pruner.relevance(0, 0, &[]);
        let expected = 0.5 * (1.0 + 0.3);
        assert!(
            (rel - expected).abs() < 1e-6,
            "expected {expected}, got {rel}"
        );
        assert!(rel > 0.5, "relaxed relevance should exceed base");
    }

    // G2: High top-1 prob returns same relevance as inner
    #[test]
    fn test_scaffolding_returns_inner_relevance() {
        let mut pruner = make_pruner(0.5, 0.5, 0.3);
        pruner.update_entropy(0.9); // above threshold → Scaffolding
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 0.5).abs() < 1e-6, "expected 0.5, got {rel}");
    }

    // G3: Threshold boundary works correctly
    #[test]
    fn test_threshold_boundary() {
        let mut pruner = make_pruner(1.0, 0.5, 0.5);

        // Exactly at threshold → Scaffolding (not < threshold)
        pruner.update_entropy(0.5);
        assert_eq!(pruner.state(), EntropyState::Scaffolding);
        assert!((pruner.relevance(0, 0, &[]) - 1.0).abs() < 1e-6);

        // Just below threshold → Forking
        pruner.update_entropy(0.4999);
        assert_eq!(pruner.state(), EntropyState::Forking);
        assert!((pruner.relevance(0, 0, &[]) - 1.5).abs() < 1e-6);
    }

    // G4: Relax factor scaling is correct
    #[test]
    fn test_relax_factor_scaling() {
        let mut pruner = make_pruner(0.8, 0.5, 0.25);
        pruner.update_entropy(0.1); // Forking
        let rel = pruner.relevance(0, 0, &[]);
        let expected = 0.8 * 1.25;
        assert!(
            (rel - expected).abs() < 1e-6,
            "expected {expected}, got {rel}"
        );
    }

    // G5: Delegates correctly to inner pruner for non-fork tokens
    #[test]
    fn test_delegation_to_inner() {
        let mut pruner = make_pruner(0.7, 0.5, 0.3);
        pruner.update_entropy(0.8); // Scaffolding
        let rel = pruner.relevance(3, 42, &[1, 2, 3]);
        assert!(
            (rel - 0.7).abs() < 1e-6,
            "should delegate to inner, got {rel}"
        );
    }

    // G6: EntropyState updates correctly via update_entropy()
    #[test]
    fn test_entropy_state_transitions() {
        let mut pruner = make_pruner(1.0, 0.5, 0.3);

        // Default state
        assert_eq!(pruner.state(), EntropyState::Scaffolding);

        // Low top-1 → Forking
        pruner.update_entropy(0.1);
        assert_eq!(pruner.state(), EntropyState::Forking);

        // High top-1 → Scaffolding
        pruner.update_entropy(0.9);
        assert_eq!(pruner.state(), EntropyState::Scaffolding);

        // Back to low → Forking
        pruner.update_entropy(0.3);
        assert_eq!(pruner.state(), EntropyState::Forking);
    }
}
