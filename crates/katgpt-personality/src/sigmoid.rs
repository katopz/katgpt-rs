//! Numerically stable sigmoid helper (Plan 297 T1.5).
//!
//! Per AGENTS.md: sigmoid is mandated over softmax for projections onto
//! learned direction vectors. The branching implementation below avoids
//! overflow for large `|x|` — the negative branch computes `e^x / (1 + e^x)`
//! instead of `1 / (1 + e^{-x})`, so `e^{-x}` never overflows for `x → -∞`.
//!
//! # Stability
//!
//! - `x ≥ 0`: `1 / (1 + e^{-x})` — `e^{-x} ∈ (0, 1]`, no overflow.
//! - `x < 0`: `e^x / (1 + e^x)` — `e^x ∈ (0, 1)`, no overflow.
//! - `|x| > ~18`: result saturates to `0.0` or `1.0` in f32 (correct).
//! - `|x| > 88`: `e^{±88}` would overflow f32, but the branching avoids it.

/// Numerically stable scalar sigmoid: `σ(x) = 1 / (1 + e^{-x})`.
///
/// Branching on the sign of `x` avoids `e^{-x}` overflow for large negative
/// `x`. The result is in `(0, 1)` for all finite inputs, and saturates to
/// `0.0` / `1.0` for `|x| > ~18` (correct limit behaviour).
///
/// # Examples
///
/// ```
/// # use katgpt_core::personality_composition::sigmoid;
/// assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
/// assert!(sigmoid(100.0) > 0.999999);
/// assert!(sigmoid(-100.0) < 1e-6);
/// assert!(sigmoid(1000.0).is_finite());  // no overflow
/// ```
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Vectorized sigmoid: `out[i] = σ(x[i])`.
///
/// Elementwise application of [`sigmoid`]. The caller is responsible for
/// ensuring `out.len() >= x.len()`; the function processes `x.len()` elements.
///
/// # Panics (debug)
///
/// In debug builds, panics if `out.len() < x.len()`.
pub fn sigmoid_into(x: &[f32], out: &mut [f32]) {
    debug_assert!(
        out.len() >= x.len(),
        "out too short: {} < {}",
        out.len(),
        x.len()
    );
    for (src, dst) in x.iter().zip(out.iter_mut()) {
        *dst = sigmoid(*src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_stable_for_extreme_inputs() {
        // Large positive → saturates near 1, no overflow.
        let large_pos = sigmoid(1000.0);
        assert!(large_pos.is_finite());
        assert!(large_pos <= 1.0);
        assert!(large_pos > 0.9999);

        // Large negative → saturates near 0, no overflow.
        let large_neg = sigmoid(-1000.0);
        assert!(large_neg.is_finite());
        assert!(large_neg >= 0.0);
        assert!(large_neg < 1e-4);
    }

    #[test]
    fn sigmoid_monotone() {
        // σ(x) is strictly increasing.
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            assert!(sigmoid(x + 0.01) > sigmoid(x), "not monotone at {x}");
        }
    }

    #[test]
    fn sigmoid_into_matches_scalar() {
        let input = vec![-100.0, -1.0, 0.0, 1.0, 100.0];
        let mut out = vec![0.0; input.len()];
        sigmoid_into(&input, &mut out);
        for (src, dst) in input.iter().zip(out.iter()) {
            assert!((sigmoid(*src) - dst).abs() < 1e-6);
        }
    }
}
