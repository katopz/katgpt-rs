//! Offline sensitivity measurement — STUB interface (Plan 271 Phase 3, T3.4).
//!
//! The real measurement tool runs a model on a calibration dataset, sweeps
//! compression ratios per head, and records the quality delta. That requires
//! GPU training infrastructure and lives in `riir-ai` (see plan Out of Scope).
//!
//! This module provides a synthetic stub so the solver can be tested and the
//! downstream schedule pipeline can be exercised end-to-end without a real
//! measurement run. The stub curves mimic realistic head diversity: some
//! heads are nearly flat (insensitive), others are steep (sensitive).

use super::curve::HeadSensitivityCurve;

/// Standard ratio grid used by the stub. Real measurement should use the
/// same grid for consistency (or expose it as a parameter).
pub const STUB_RATIOS: &[f32] = &[0.05, 0.1, 0.25, 0.5, 0.75, 1.0];

/// Generate synthetic sensitivity curves for testing.
///
/// Returns `num_heads` curves, one per head. The synthetic pattern:
/// - Heads with even `head_id` are "sensitive" — quality drops steeply as
///   the ratio shrinks. Modeled as `delta = (1 - r)^1.5 * 0.8`.
/// - Heads with odd `head_id` are "flat" — quality barely changes. Modeled
///   as `delta = (1 - r) * 0.1`.
///
/// This produces a realistic mix where the solver should allocate more
/// budget to even-indexed heads.
///
/// # Real implementation
/// Replace this with a call to the `riir-ai` measurement pipeline:
/// ```ignore
/// pub fn measure_sensitivity(
///     model: &Model,
///     dataset: &Dataset,
///     ratios: &[f32],
/// ) -> Vec<HeadSensitivityCurve> { ... }
/// ```
pub fn measure_sensitivity_stub(num_heads: usize) -> Vec<HeadSensitivityCurve> {
    (0..num_heads)
        .map(|head_id| {
            let deltas: Vec<f32> = STUB_RATIOS
                .iter()
                .map(|&r| synthetic_delta(head_id, r))
                .collect();
            HeadSensitivityCurve::new(head_id, STUB_RATIOS.to_vec(), deltas)
        })
        .collect()
}

/// Synthetic quality delta for a head at a given ratio.
///
/// Even heads are sensitive (steep curve), odd heads are flat. The exact
/// shape is arbitrary but monotonic in `r` (smaller `r` → larger delta).
#[inline]
fn synthetic_delta(head_id: usize, r: f32) -> f32 {
    let one_minus_r = (1.0f32 - r).max(0.0);
    if head_id.is_multiple_of(2) {
        // Sensitive head.
        (one_minus_r.powf(1.5) * 0.8).min(1.0)
    } else {
        // Flat head.
        one_minus_r * 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_returns_correct_count() {
        let curves = measure_sensitivity_stub(8);
        assert_eq!(curves.len(), 8);
        for (i, c) in curves.iter().enumerate() {
            assert_eq!(c.head_id, i);
            assert_eq!(c.ratios, STUB_RATIOS);
            assert_eq!(c.deltas.len(), STUB_RATIOS.len());
        }
    }

    #[test]
    fn test_stub_deltas_monotonic() {
        // delta should be non-increasing as ratio increases (less compaction
        // → less quality loss).
        let curves = measure_sensitivity_stub(4);
        for c in &curves {
            for w in c.deltas.windows(2) {
                assert!(
                    w[0] >= w[1] - 1e-6,
                    "delta should be non-increasing in ratio: head={} deltas={:?}",
                    c.head_id,
                    c.deltas
                );
            }
        }
    }

    #[test]
    fn test_stub_even_heads_more_sensitive() {
        let curves = measure_sensitivity_stub(4);
        // At r=0.1, even heads should have larger delta than odd heads.
        let r = 0.1f32;
        for pair in curves.chunks(2) {
            let even = &pair[0];
            let odd = &pair[1];
            assert!(
                even.interpolate(r) > odd.interpolate(r),
                "even head {} should be more sensitive than odd head {} at r={}: {} vs {}",
                even.head_id,
                odd.head_id,
                r,
                even.interpolate(r),
                odd.interpolate(r)
            );
        }
    }

    #[test]
    fn test_stub_zero_heads() {
        let curves = measure_sensitivity_stub(0);
        assert_eq!(curves.len(), 0);
    }

    #[test]
    fn test_stub_at_full_ratio_zero_delta() {
        // At r=1.0, every head should have ~0 quality loss (no compaction).
        let curves = measure_sensitivity_stub(4);
        for c in &curves {
            let delta = c.interpolate(1.0);
            assert!(delta.abs() < 1e-6, "delta at r=1.0 should be 0, got {}", delta);
        }
    }
}
