//! FrozenBaseGuard — Lightweight Screening at Intermediate Steps (Plan 171).
//!
//! Named after the "frozen base model" guard from the Thinking Pixel paper
//! (arXiv:2604.25299 §3.3), where the frozen model is applied only at the final
//! latent step to prevent distribution drift from repeated exposure.
//!
//! Applied to our SpecHop/LT2 pipeline: intermediate hops/loops use `ScreeningPruner`
//! only (lightweight, O(1)), reserving full verification for the final step.
//! Intermediate steps return relevance 1.0 (accept all) — the final step applies
//! the full inner pruner pipeline.
//!
//! **Feature gate:** `thinking_prune`
//!
//! # Safety Argument
//!
//! The paper's ablation (Table 2) shows recursion *without* modulation still improves
//! over no recursion (70.36 vs 69.55 SD3+SFT baseline). Intermediate steps contribute
//! by refining the draft distribution — they don't need to be perfect. The final step
//! applies full verification to ensure quality.
//!
//! # Performance
//!
//! - No allocation inside hot loops (struct wraps existing pruner)
//! - O(1) branch per relevance call (single bool check)
//! - Pre-compute `is_final_step` once per hop, not per token

use katgpt_core::ScreeningPruner;

/// Guard that applies full verification only at the final recursion step.
///
/// Intermediate steps accept everything (relevance 1.0). The final step delegates
/// to the inner pruner for full screening.
///
/// # Type Parameter
///
/// `P`: Any type implementing [`ScreeningPruner`]. Typically `BanditPruner`,
/// `FlowPruner`, or composed pruners.
///
/// # Usage
///
/// ```ignore
/// use katgpt_rs::pruners::FrozenBaseGuard;
/// use katgpt_rs::speculative::ScreeningPruner;
///
/// let inner = MyPruner::new();
/// // Intermediate hop (hop 0 of 3 total):
/// let guard = FrozenBaseGuard::new(inner, false);
/// assert_eq!(guard.relevance(0, 0, &[]), 1.0); // accept all
///
/// // Final hop (hop 2 of 3 total):
/// let guard = FrozenBaseGuard::new(inner, true);
/// guard.relevance(0, 0, &[]); // delegates to inner.relevance()
/// ```
#[derive(Debug, Clone)]
pub struct FrozenBaseGuard<P: ScreeningPruner> {
    inner: P,
    /// When true, applies full inner pruner. When false, returns 1.0 (accept all).
    is_final_step: bool,
}

impl<P: ScreeningPruner> FrozenBaseGuard<P> {
    /// Create a new guard wrapping `inner`.
    ///
    /// - `is_final_step = true`: delegates to `inner.relevance()` (full screening)
    /// - `is_final_step = false`: returns 1.0 (accept all, lightweight)
    #[inline]
    pub fn new(inner: P, is_final_step: bool) -> Self {
        Self {
            inner,
            is_final_step,
        }
    }

    /// Create a guard for an intermediate step (accept-all mode).
    #[inline]
    pub fn intermediate(inner: P) -> Self {
        Self {
            inner,
            is_final_step: false,
        }
    }

    /// Create a guard for the final step (full screening).
    #[inline]
    pub fn final_step(inner: P) -> Self {
        Self {
            inner,
            is_final_step: true,
        }
    }

    /// Create a guard by computing `is_final_step` from hop position.
    ///
    /// `is_final_step = (hop_index == total_hops.saturating_sub(1))`
    #[inline]
    pub fn from_hop_context(inner: P, hop_index: usize, total_hops: usize) -> Self {
        Self {
            inner,
            is_final_step: hop_index >= total_hops.saturating_sub(1),
        }
    }

    /// Access the inner pruner.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Whether this guard applies full screening.
    #[inline]
    pub fn is_final_step(&self) -> bool {
        self.is_final_step
    }
}

impl<P: ScreeningPruner> ScreeningPruner for FrozenBaseGuard<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.is_final_step {
            self.inner.relevance(depth, token_idx, parent_tokens)
        } else {
            // Intermediate step: lightweight screening only.
            // Don't reject anything — let the final step decide.
            1.0
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

    #[test]
    fn test_intermediate_returns_one() {
        let inner = FixedPruner { relevance_val: 0.5 };
        let guard = FrozenBaseGuard::intermediate(inner);
        assert_eq!(guard.relevance(0, 0, &[]), 1.0);
        assert_eq!(guard.relevance(5, 10, &[1, 2, 3]), 1.0);
    }

    #[test]
    fn test_final_step_delegates() {
        let inner = FixedPruner { relevance_val: 0.3 };
        let guard = FrozenBaseGuard::final_step(inner);
        assert_eq!(guard.relevance(0, 0, &[]), 0.3);
        assert_eq!(guard.relevance(3, 7, &[1]), 0.3);
    }

    #[test]
    fn test_new_constructor() {
        let inner = FixedPruner { relevance_val: 0.7 };
        let intermediate = FrozenBaseGuard::new(inner.clone(), false);
        let final_step = FrozenBaseGuard::new(inner, true);
        assert_eq!(intermediate.relevance(0, 0, &[]), 1.0);
        assert_eq!(final_step.relevance(0, 0, &[]), 0.7);
    }

    #[test]
    fn test_from_hop_context_first_of_three() {
        let inner = FixedPruner { relevance_val: 0.5 };
        // hop 0 of 3 → intermediate
        let guard = FrozenBaseGuard::from_hop_context(inner, 0, 3);
        assert!(!guard.is_final_step());
        assert_eq!(guard.relevance(0, 0, &[]), 1.0);
    }

    #[test]
    fn test_from_hop_context_last_of_three() {
        let inner = FixedPruner { relevance_val: 0.5 };
        // hop 2 of 3 → final
        let guard = FrozenBaseGuard::from_hop_context(inner, 2, 3);
        assert!(guard.is_final_step());
        assert_eq!(guard.relevance(0, 0, &[]), 0.5);
    }

    #[test]
    fn test_from_hop_context_single_hop() {
        let inner = FixedPruner { relevance_val: 0.5 };
        // hop 0 of 1 → final
        let guard = FrozenBaseGuard::from_hop_context(inner, 0, 1);
        assert!(guard.is_final_step());
        assert_eq!(guard.relevance(0, 0, &[]), 0.5);
    }

    #[test]
    fn test_from_hop_context_zero_hops() {
        let inner = FixedPruner { relevance_val: 0.5 };
        // 0 hops → saturating_sub makes final
        let guard = FrozenBaseGuard::from_hop_context(inner, 0, 0);
        assert!(guard.is_final_step());
    }

    #[test]
    fn test_inner_accessor() {
        let inner = FixedPruner {
            relevance_val: 0.42,
        };
        let guard = FrozenBaseGuard::intermediate(inner);
        assert!((guard.inner().relevance_val - 0.42).abs() < 1e-6);
    }
}
