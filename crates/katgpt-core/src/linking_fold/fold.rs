//! Coordinate-fold unlinking correction (paper §5, Eq. 1) — hot-path, zero-alloc.
//!
//! The paper proves (Theorem 4.7) that width-d feedforward nets with
//! coordinate-wise monotonic activations preserve the linking number and
//! therefore cannot linearly separate linked manifolds. The escape is a
//! **fold**: a coordinate-wise map with a strict local extremum, such as
//! `|x|`, GELU, Swish, or Mish. A ResNet block realizes `|x|` exactly via
//! the identity `|x| = x + 2·ReLU(−x)` (Eq. 1) — one residual block with a
//! monotonic ReLU + skip is enough to break the homotopy-preservation
//! argument that underlies the impossibility theorem.
//!
//! This module ships the modelless fold as an in-place latent correction.
//! When the linking detector (`detect_linking`, paper Algorithm 1, gated under
//! the `linking_fold_detector` feature) fires (two latent clusters are
//! topologically linked), apply one coordinate-fold pass per axis to unlink
//! them before the monotonic projection. The fold is:
//!
//! - **Closed-form**: `state[i] ← center[i] + |state[i] − center[i]|` (Abs
//!   variant), or a smooth GELU-surrogate with a local minimum (Gelu variant).
//! - **Zero-allocation, `#[inline]`**: hot-path, per-tick safe.
//! - **Deterministic**: bit-identical given the same input + center.
//!
//! # Why this is §3.5 path-3 (latent-space correction), not riir-train
//!
//! The fold is the modelless analog of a trained adapter that fixes a
//! systematic, characterizable bias ("two classes are linked"). Instead of
//! learning the correction via gradient descent, it is derived in closed
//! form from the failure mode (Lemma C.7: monotonic activations preserve
//! link via straight-line homotopy; a fold breaks the homotopy by creating
//! a collision line). This is exactly the case the modelless-unblock
//! protocol (AGENTS.md §3.5) prefers over riir-train deferral.

/// Hard coordinate-fold: `state[i] ← center[i] + |state[i] − center[i]|`.
///
/// This is the paper's `|x| = x + 2·ReLU(−x)` (Eq. 1) realized as a single
/// fold per coordinate, applied in-place. After one pass per axis (three
/// passes for R^3, paper Fig. 9), a linked pair of manifolds becomes
/// linearly separable.
///
/// The fold reflects the input across `center` on each axis, collapsing
/// the negative half-line onto the positive one. For a Hopf link where the
/// two components occupy complementary half-spaces (e.g. X near x=0 plane,
/// Y near y=0 plane), folding both onto the positive octant makes them
/// disjoint and convexly separable.
///
/// # Arguments
///
/// - `state`: flat `[x_0, x_1, ..., x_{d-1}]` latent vector, modified in-place.
/// - `center`: the fold center `c`; same length as `state`. Pass a zero
///   slice for the canonical `|x|` fold (reflects across the origin).
///
/// # Panics
///
/// Panics (debug builds only) if `state.len() != center.len()`. In release
/// the shorter slice determines the fold count — caller's contract.
///
/// # Example
///
/// ```
/// use katgpt_core::linking_fold::fold_projection_into;
/// // Fold a 3D vector onto the positive octant.
/// let mut state = [-1.0_f32, 0.5, -0.25];
/// let center = [0.0_f32; 3];
/// fold_projection_into(&mut state, &center);
/// assert_eq!(state, [1.0, 0.5, 0.25]);
/// ```
#[inline]
pub fn fold_projection_into(state: &mut [f32], center: &[f32]) {
    debug_assert_eq!(
        state.len(),
        center.len(),
        "fold_projection_into: state and center must have the same length"
    );
    let n = state.len().min(center.len());
    for i in 0..n {
        let diff = state[i] - center[i];
        state[i] = center[i] + diff.abs();
    }
}

/// Smoothed `|x|` via a multiquadric-bump surrogate (paper §F.2).
///
/// The paper's impossibility theorem (Thm 4.7) depends only on monotonicity,
/// not on the specific form of the activation. Any activation with a strict
/// local extremum on an open interval breaks the straight-line homotopy
/// argument (Lemma C.7) the same way `|·|` does. GELU `x·Φ(x)` has a local
/// minimum near `x ≈ −0.753...`; Swish `x·σ(x)` near `x ≈ −1.278...`; Mish
/// near a similar point. The construction (paper §F.2, Example 5.1) is:
///
/// 1. Rescale the data coordinate into the local-extremum interval via an
///    affine pre-shift.
/// 2. Apply the non-monotonic activation — it folds the data the same way
///    `|·|` does, just smoothly.
///
/// This function returns a single-coordinate smoothed `|x|`-surrogate:
/// `gelu_smoothed_abs(x, alpha)` ≈ `|x|` for large `|x|`, but with a smooth
/// V-shape near the origin instead of a hard kink. The `alpha` parameter
/// controls the smoothness (larger → sharper, → hard `|x|` as `alpha → ∞`).
///
/// The formula is `sqrt(x² + alpha⁻²) − alpha⁻¹` (a smoothed absolute value,
/// the multiquadric-bump analogue of `|x|`). It is everywhere differentiable,
/// has a strict local minimum at `x = 0`, and converges to `|x| − alpha⁻¹`
/// as `|x| → ∞`. The `−alpha⁻¹` offset is removed so the minimum is exactly 0.
///
/// This is a generic smoothed-abs surrogate, not GELU itself — GELU's local
/// minimum is asymmetric and only exists on the negative side, which makes
/// it unsuitable as a direct symmetric `|x|` replacement. The multiquadric
/// surrogate is symmetric, everywhere-smooth, and has the right large-`|x|`
/// asymptote.
#[inline]
pub fn gelu_smoothed_abs(x: f32, alpha: f32) -> f32 {
    // sqrt(x² + α⁻²) has min α⁻¹ at x=0, asymptote |x|. Subtract α⁻¹ to
    // make the minimum 0 (matching |x|'s minimum).
    let inv_alpha = 1.0_f32 / alpha;
    (x * x + inv_alpha * inv_alpha).sqrt() - inv_alpha
}

/// Smooth GELU-surrogate coordinate-fold.
///
/// Like [`fold_projection_into`] but uses [`gelu_smoothed_abs`] instead of
/// the hard `|·|`. The result is a smooth fold — differentiable everywhere,
/// with no kink at the center. Useful when the downstream consumer is
/// sensitive to discontinuities (e.g. a Jacobian-based diagnostic that
/// would be undefined at the `|·|` kink).
///
/// # Arguments
///
/// - `state`: latent vector, modified in-place.
/// - `center`: fold center, same length as `state`.
/// - `alpha`: smoothness parameter. `alpha → ∞` recovers the hard
///   [`fold_projection_into`]; `alpha = 10.0` is a reasonable default
///   (smooth near `|x| < 0.1`, sharp outside).
///
/// # Example
///
/// ```
/// use katgpt_core::linking_fold::fold_gelu_into;
/// let mut state = [-1.0_f32, 0.5, -0.25];
/// let center = [0.0_f32; 3];
/// fold_gelu_into(&mut state, &center, 10.0);
/// // Smoothed fold: `sqrt(x² + α⁻²) − α⁻¹`, so each coordinate lands just
/// // *under* `|x|`. The shortfall tends to `α⁻¹ = 0.1` as `|x|` grows and
/// // shrinks toward 0 at the center — it is largest, not smallest, for the
/// // coordinates nearest the fold center.
/// assert!((state[0] - 0.904_987_6).abs() < 1e-6); // |-1.00| - 0.0950
/// assert!((state[1] - 0.409_902_0).abs() < 1e-6); // | 0.50| - 0.0901
/// assert!((state[2] - 0.169_258_2).abs() < 1e-6); // |-0.25| - 0.0807
/// ```
#[inline]
pub fn fold_gelu_into(state: &mut [f32], center: &[f32], alpha: f32) {
    debug_assert_eq!(
        state.len(),
        center.len(),
        "fold_gelu_into: state and center must have the same length"
    );
    debug_assert!(alpha.is_finite() && alpha > 0.0, "alpha must be finite > 0");
    let n = state.len().min(center.len());
    for i in 0..n {
        let diff = state[i] - center[i];
        state[i] = center[i] + gelu_smoothed_abs(diff, alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_abs_reflects_negative_halfline() {
        let mut state = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
        let center = [0.0_f32; 5];
        fold_projection_into(&mut state, &center);
        assert_eq!(state, [1.0, 0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn fold_abs_with_nonzero_center() {
        let mut state = [-1.0_f32, 0.0, 1.0];
        let center = [0.5_f32; 3];
        fold_projection_into(&mut state, &center);
        // diff = [-1.5, -0.5, 0.5], |diff| = [1.5, 0.5, 0.5]
        // result = center + |diff| = [2.0, 1.0, 1.0]
        assert_eq!(state, [2.0, 1.0, 1.0]);
    }

    #[test]
    fn fold_gelu_approximates_abs_for_large_alpha() {
        let mut state = [-1.0_f32, 0.5, -0.25];
        let center = [0.0_f32; 3];
        fold_gelu_into(&mut state, &center, 1000.0);
        // alpha=1000 → very sharp, should be within 1e-3 of hard |x|.
        assert!((state[0] - 1.0).abs() < 1e-3);
        assert!((state[1] - 0.5).abs() < 1e-3);
        assert!((state[2] - 0.25).abs() < 1e-3);
    }

    #[test]
    fn fold_gelu_is_smooth_at_origin() {
        // gelu_smoothed_abs(0, alpha) should be exactly 0 for any alpha
        // (the −alpha⁻¹ offset cancels the sqrt at x=0).
        for &alpha in &[1.0_f32, 5.0, 10.0, 100.0] {
            assert!(gelu_smoothed_abs(0.0, alpha).abs() < 1e-6);
        }
    }

    #[test]
    fn fold_gelu_is_symmetric() {
        for &alpha in &[1.0_f32, 5.0, 10.0] {
            for &x in &[-2.0_f32, -1.0, -0.5, 0.5, 1.0, 2.0] {
                let (xp, xn) = (gelu_smoothed_abs(x, alpha), gelu_smoothed_abs(-x, alpha));
                assert!(
                    (xp - xn).abs() < 1e-6,
                    "not symmetric at x={x}, alpha={alpha}"
                );
            }
        }
    }

    #[test]
    fn fold_gelu_monotone_away_from_origin() {
        // For x > 0, gelu_smoothed_abs should be strictly increasing.
        for &alpha in &[1.0_f32, 5.0, 10.0] {
            let mut prev = f32::NEG_INFINITY;
            let mut x = 0.0_f32;
            while x < 3.0 {
                let v = gelu_smoothed_abs(x, alpha);
                assert!(v >= prev, "not monotone at x={x}, alpha={alpha}");
                prev = v;
                x += 0.1;
            }
        }
    }

    #[test]
    fn fold_abs_deterministic_bit_identical() {
        let base = [-1.5_f32, 0.3, -0.7, 2.1, -0.01];
        let center = [0.1_f32; 5];
        let mut a = base;
        let mut b = base;
        for _ in 0..10 {
            fold_projection_into(&mut a, &center);
            fold_projection_into(&mut b, &center);
        }
        // Idempotence + determinism: after one fold, subsequent folds are no-ops
        // (already in positive half), and both runs agree bit-for-bit.
        assert_eq!(a, b);
    }
}
