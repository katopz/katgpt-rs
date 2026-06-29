//! MGPO sampling weight + budget allocation (Plan 284 T2.5).
//!
//! Distilled from Research 255's MGPO (Margin-Guided Policy Optimization) arm,
//! stripped of all training. This is the modelless inference-time sampling
//! primitive: given a per-seed success rate `p`, weight future sampling toward
//! the calibration boundary `p = 0.5`.
//!
//! # Math
//!
//! ```text
//! w(p) = exp(-γ * |2p - 1|)
//! ```
//!
//! - `p = 0.5` → `w = exp(0) = 1.0` (peak — calibration boundary, max entropy).
//! - `p = 0.0` → `w = exp(-γ)` (too hard — model never succeeds).
//! - `p = 1.0` → `w = exp(-γ)` (saturated — model always succeeds).
//!
//! The caller maintains an EMA `p` per sampling seed and recomputes `w(p)`
//! each cycle. [`allocate_budget`] then distributes a fixed total sample
//! budget proportionally to `w`.
//!
//! # No softmax
//!
//! This is `exp(-γ|x|)` (Laplace-kernel decay), NOT softmax. Softmax would
//! normalize across seeds and destroy per-seed independence: a seed at `p=0.5`
//! should always get weight 1.0 regardless of how many other seeds exist.

use katgpt_core::simd::simd_sum_f32;

/// MGPO sampling weight: `exp(-γ * |2p - 1|)`.
///
/// Peaks at `p = 0.5` (maximum-entropy / calibration boundary), decays toward
/// `p = 0` (too hard) and `p = 1` (saturated). The caller maintains an EMA `p`
/// per sampling seed.
///
/// # Arguments
///
/// * `p` — per-seed success rate (EMA), typically in `[0, 1]`.
/// * `gamma` — decay exponent `γ` (from [`crate::clr::ClrConfig::gamma_mgpo`]).
///
/// # Returns
///
/// Weight in `(0, 1]`. `p = 0.5` always returns `1.0` regardless of `γ`.
#[inline]
pub fn mgpo_sampling_weight(p: f32, gamma: f32) -> f32 {
    (-gamma * (2.0 * p - 1.0).abs()).exp()
}

/// Distribute a fixed total sample budget proportionally to per-seed weights.
///
/// Proportional allocation: each seed gets `floor(w_i / Σ * total_budget)`.
/// The remainder (from flooring) goes to the last seed so the result sums to
/// exactly `total_budget`.
///
/// # Arguments
///
/// * `weights` — per-seed MGPO weights (from [`mgpo_sampling_weight`]).
///   Non-negative; need not be normalized.
/// * `total_budget` — total samples to distribute. Must be `> 0` when
///   `weights` is non-empty.
///
/// # Returns
///
/// `Vec<usize>` of length `weights.len()` summing to `total_budget`. The last
/// element absorbs the flooring remainder.
///
/// # Panics
///
/// Panics if `total_budget == 0` while `weights` is non-empty.
///
/// # Allocation
///
/// Allocates the returned `Vec<usize>` (output only — runs once per sampling
/// cycle, not per token).
pub fn allocate_budget(weights: &[f32], total_budget: usize) -> Vec<usize> {
    assert!(
        total_budget > 0 || weights.is_empty(),
        "allocate_budget: total_budget must be > 0 when weights non-empty"
    );
    if weights.is_empty() {
        return Vec::new();
    }
    let sum: f32 = simd_sum_f32(weights);
    let inv = 1.0 / sum;
    let mut counts: Vec<usize> = Vec::with_capacity(weights.len());
    let mut allocated = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        if i + 1 == weights.len() {
            // Last seed absorbs the remainder so the total is exact.
            counts.push(total_budget.saturating_sub(allocated));
        } else {
            let c = (w * inv * total_budget as f32).floor() as usize;
            counts.push(c);
            allocated += c;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_at_p_half() {
        let gamma = 2.0;
        let w_peak = mgpo_sampling_weight(0.5, gamma);
        let w_zero = mgpo_sampling_weight(0.0, gamma);
        let w_one = mgpo_sampling_weight(1.0, gamma);
        assert!((w_peak - 1.0).abs() < 1e-6, "peak should be 1.0, got {}", w_peak);
        assert!(w_zero < w_peak);
        assert!(w_one < w_peak);
        assert!((w_zero - w_one).abs() < 1e-6, "w(0) should equal w(1)");
    }

    #[test]
    fn known_value_at_p_zero_gamma_two() {
        // w(0) = exp(-2 * |-1|) = exp(-2) ≈ 0.1353.
        let w = mgpo_sampling_weight(0.0, 2.0);
        assert!((w - (-2.0f32).exp()).abs() < 1e-6, "got {}", w);
    }

    #[test]
    fn allocate_budget_sums_exact() {
        let weights = [1.0f32, 0.5, 0.25, 0.125];
        let total = 100;
        let alloc = allocate_budget(&weights, total);
        assert_eq!(alloc.len(), weights.len());
        assert_eq!(alloc.iter().sum::<usize>(), total, "must sum to total");
        assert!(alloc[0] > alloc[3], "seed 0 should get more than seed 3");
    }

    #[test]
    fn allocate_budget_uniform() {
        let weights = [1.0f32, 1.0, 1.0, 1.0];
        let total = 40;
        let alloc = allocate_budget(&weights, total);
        assert_eq!(alloc.iter().sum::<usize>(), total);
    }

    #[test]
    fn allocate_budget_empty() {
        let alloc = allocate_budget(&[], 0);
        assert!(alloc.is_empty());
    }

    #[test]
    #[should_panic(expected = "total_budget must be > 0")]
    fn allocate_budget_panics_on_zero_budget() {
        let _ = allocate_budget(&[1.0, 1.0], 0);
    }
}
