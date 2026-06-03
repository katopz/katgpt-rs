//! Budget adaptation integration helpers (Plan 167 Phase 2, T4+T6).
//!
//! Provides the glue between `adaptive_tree_budget()` and the speculative
//! decoding dispatch layer. Callers compute an effective budget from the
//! compression ratio, then use it when building DDTree.
//!
//! # Usage
//! ```ignore
//! use crate::speculative::budget_compat::{effective_tree_budget, scaled_draft_lookahead};
//!
//! let effective = effective_tree_budget(base_budget, ratio, mode);
//! let lookahead = scaled_draft_lookahead(base_lookahead, effective, base_budget);
//! // Use `effective` and `lookahead` when calling build_dd_tree or TreeBuilder
//! ```

use crate::speculative::types::BudgetAdaptation;

/// Compute effective tree budget for the current prompt.
///
/// If budget adaptation is enabled (Compression or Entropy mode), derives
/// the effective budget from the base budget and compression ratio.
/// Otherwise returns the base budget unchanged.
///
/// # Arguments
/// * `base_budget` — domain-config tree_budget
/// * `compression_ratio` — r ∈ (0,1] from PFlash block selection
/// * `mode` — adaptation mode from FlashPrefillConfig
///
/// # Example
/// ```ignore
/// let effective = effective_tree_budget(2374, 0.3, config.budget_adaptation);
/// // Use `effective` when calling build_dd_tree or build_dd_tree_screened
/// ```
pub fn effective_tree_budget(
    base_budget: usize,
    compression_ratio: f32,
    mode: BudgetAdaptation,
) -> usize {
    #[cfg(feature = "budget_adaptation")]
    {
        crate::speculative::budget::adaptive_tree_budget(base_budget, compression_ratio, mode)
    }
    #[cfg(not(feature = "budget_adaptation"))]
    {
        let _ = (compression_ratio, mode);
        base_budget
    }
}

/// Scale draft_lookahead proportionally when budget changes.
///
/// Uses sqrt relationship: if budget doubles, lookahead scales by ~1.4×.
/// If budget halves, lookahead scales by ~0.7×.
/// Rationale: more tree_budget → more branches → more lookahead to fill them.
///
/// # Arguments
/// * `base_lookahead` — current draft_lookahead from domain config
/// * `effective_budget` — adapted budget from `effective_tree_budget`
/// * `base_budget` — original domain-config tree_budget
pub fn scaled_draft_lookahead(
    base_lookahead: usize,
    effective_budget: usize,
    base_budget: usize,
) -> usize {
    if base_budget == 0 {
        return base_lookahead;
    }
    let ratio = (effective_budget as f64) / (base_budget as f64);
    let scaled = (base_lookahead as f64 * ratio.sqrt()) as usize;
    scaled.max(1).min(base_lookahead * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_lookahead_identity() {
        // Same budget → same lookahead
        assert_eq!(scaled_draft_lookahead(5, 100, 100), 5);
    }

    #[test]
    fn test_scaled_lookahead_double_budget() {
        // 2× budget → ~1.4× lookahead (sqrt(2) ≈ 1.414)
        let result = scaled_draft_lookahead(5, 200, 100);
        assert_eq!(result, 7); // 5 * 1.414 ≈ 7.07 → 7
    }

    #[test]
    fn test_scaled_lookahead_half_budget() {
        // 0.5× budget → ~0.7× lookahead (sqrt(0.5) ≈ 0.707)
        let result = scaled_draft_lookahead(5, 50, 100);
        assert_eq!(result, 3); // 5 * 0.707 ≈ 3.53 → 3
    }

    #[test]
    fn test_scaled_lookahead_min_one() {
        // Even with tiny budget, lookahead is at least 1
        let result = scaled_draft_lookahead(5, 1, 100);
        assert!(result >= 1);
    }

    #[test]
    fn test_scaled_lookahead_max_double() {
        // Lookahead never exceeds 2× base
        let result = scaled_draft_lookahead(5, 10000, 100);
        assert!(result <= 10);
    }

    #[test]
    fn test_scaled_lookahead_zero_base_budget() {
        // Edge case: zero base budget → return base lookahead
        assert_eq!(scaled_draft_lookahead(5, 100, 0), 5);
    }

    #[test]
    fn test_effective_tree_budget_off_returns_base() {
        let result = effective_tree_budget(100, 0.5, BudgetAdaptation::Off);
        assert_eq!(result, 100);
    }

    #[test]
    fn test_effective_tree_budget_entropy_adapts() {
        let result = effective_tree_budget(100, 1.5, BudgetAdaptation::Entropy);
        // H=1.5: half of threshold → scale = 0.5 + 1.5*0.5 = 1.25 → 125
        assert_eq!(result, 125);
    }

    #[cfg(feature = "budget_adaptation")]
    #[test]
    fn test_effective_tree_budget_compression_adapts() {
        let result = effective_tree_budget(100, 0.5, BudgetAdaptation::Compression);
        // scale = 0.5 + 1.5*0.5 = 1.25 → 125
        assert_eq!(result, 125);
    }
}
