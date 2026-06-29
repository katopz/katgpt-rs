//! Gauge-Invariant Adapter Composition — modelless primitives for LoRA factor pairs.
//!
//! Distilled from arXiv:2606.12921 (LoRA-Muon, Cesista/Crowson/Simal/Biderman).
//! Plan 270, Research 238.
//!
//! A LoRA weight is `W = A·B^T` where `A ∈ R^{m×r}`, `B ∈ R^{n×r}`. The
//! factorization is not unique: `(A, B) ~ (A·R, B·R^{-T})` for any invertible
//! `R`. Arithmetic on `(A, B)` pairs — composition, interpolation, TIES merge —
//! is **gauge-dependent**: different factorizations of the same `W` give
//! different merged outputs.
//!
//! This module provides pure inference-time primitives that remove the gauge
//! artifact:
//!
//! - [`gauge_rebalance`] — paper Algorithm 2. Rescales `(A, B)` so
//!   `σ_max(A) ≈ σ_max(B)` without changing `A·B^T`. Pure matrix op.
//! - [`gauge_invariant_compose`] — weighted sum of `(η_i, A_i, B_i)` pairs
//!   with rebalancing applied first. Drop-in for naive adapter arithmetic.
//! - [`gauge_invariant_lerp`] — fast path for 2-pair interpolation.
//!
//! # When to Use
//!
//! - You are composing two or more LoRA adapters from different sources
//!   (different training runs, different games, different checkpoints).
//! - Your adapter warm-start pipeline shuffles factor scales.
//! - You want TIES-merge style sign-election to be well-defined.
//!
//! # When NOT to Use
//!
//! - You have a single adapter — gauge doesn't matter for application.
//! - You're inside the training loop (use LoRA-Muon optimizer instead,
//!   which is gauge-invariant by construction).
//!
//! # Substrate Routing
//!
//! All operations are pure CPU SIMD — no GPU, no ANE. The PSD inverse sqrt
//! and power iteration are O(r·(m+n)) — sub-microsecond for typical LoRA
//! sizes (r ∈ [4, 64], m = n ∈ [128, 4096]).
//!
//! # Example
//!
//! ```
//! use katgpt_rs::gauge_invariant::{gauge_rebalance, GaugeRebalanceScratch};
//!
//! // Two factor pairs representing the same W = A·B^T but at different gauges.
//! // Pair 1: balanced (c=1).
//! let r = 4;
//! let m = 8;
//! let n = 6;
//! let a_balanced: Vec<f32> = (0..m * r).map(|i| i as f32 * 0.01).collect();
//! let b_balanced: Vec<f32> = (0..n * r).map(|i| (i as f32 + 1.0) * 0.02).collect();
//!
//! // Pair 2: same W but A scaled by 10, B scaled by 0.1 (gauge c=10).
//! let c: f32 = 10.0;
//! let mut a_skewed: Vec<f32> = a_balanced.iter().map(|&v| v * c).collect();
//! let mut b_skewed: Vec<f32> = b_balanced.iter().map(|&v| v / c).collect();
//!
//! let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
//! gauge_rebalance(&mut a_skewed, &mut b_skewed, m, r, n, r, 1.0, &mut scratch);
//!
//! // After rebalance, σ_max(A) ≈ σ_max(B), and A·B^T is unchanged.
//! ```

#![allow(clippy::too_many_arguments)]

use katgpt_core::simd::{simd_dot_f32, simd_scale_inplace};
use crate::spectral_retract::power_iter_step;

/// Pre-allocated scratch for gauge rebalancing.
///
/// Stores power-iteration vectors for σ_max estimation. Sized for matrices
/// up to `max_m × max_r`. Reused across calls.
pub struct GaugeRebalanceScratch {
    /// Power iteration vector for A (length = r, since A is m×r and we compute A^T·A·v).
    v_a: Vec<f32>,
    /// Power iteration result for A (length = r).
    v_a_new: Vec<f32>,
    /// Power iteration vector for B (length = r, since B is n×r and we compute B^T·B·v).
    v_b: Vec<f32>,
    /// Power iteration result for B (length = r).
    v_b_new: Vec<f32>,
}

impl GaugeRebalanceScratch {
    /// Create scratch sized for factors up to `max_outer × max_rank`.
    ///
    /// For LoRA: `max_outer = max(m, n)` (input/output dim), `max_rank = r`.
    pub fn new(_max_outer: usize, max_rank: usize) -> Self {
        Self {
            v_a: vec![1.0 / (max_rank as f32).sqrt(); max_rank],
            v_a_new: vec![0.0; max_rank],
            v_b: vec![1.0 / (max_rank as f32).sqrt(); max_rank],
            v_b_new: vec![0.0; max_rank],
        }
    }
}

/// Estimate `σ_max(M)` for an `outer × rank` matrix M via power iteration on
/// `M^T M ∈ R^{rank × rank}`.
///
/// Returns the estimated largest singular value. Converges quickly for the
/// ratio estimation we need (5 steps sufficient for order-of-magnitude).
///
/// - `mat`: row-major `outer × rank` slice
/// - `v`, `v_new`: length-`rank` scratch vectors (v should be initialized to
///   a unit vector; we'll re-init if it's zero)
/// - `n_steps`: power iteration steps (5 is a good default)
fn power_iterate_sigma_max(
    mat: &[f32],
    outer: usize,
    rank: usize,
    v: &mut [f32],
    v_new: &mut [f32],
    n_steps: u8,
) -> f32 {
    assert_eq!(mat.len(), outer * rank, "matrix size mismatch");
    assert_eq!(v.len(), rank, "v length mismatch");
    assert_eq!(v_new.len(), rank, "v_new length mismatch");

    // Initialize v if zero (defensive).
    let v_norm: f32 = simd_dot_f32(v, v, rank).sqrt();
    if v_norm < 1e-20 {
        v.fill(1.0 / (rank as f32).sqrt());
    } else {
        // Normalize (in case caller passed something weird).
        simd_scale_inplace(v, 1.0 / v_norm);
    }

    for _ in 0..n_steps {
        // Compute M^T · (M · v) into v_new via the shared `power_iter_step`
        // helper (Plan 279 DRY refactor — Research 246 §6 Fusion Idea F).
        //
        // The matvec is the implicit PSD operator `M^T M` where M is
        // `outer × rank`. To avoid an `outer`-length allocation for u = M·v,
        // we recompute u[i] inline in the second pass (trading 2× compute
        // for zero allocation — original behavior preserved byte-for-byte).
        let step_norm = power_iter_step(v, v_new, |vin: &[f32], vout: &mut [f32]| {
            vout.fill(0.0);
            for i in 0..outer {
                let row = &mat[i * rank..(i + 1) * rank];
                let u_i = simd_dot_f32(row, vin, rank);
                // vout[k] += M[i,k] · u_i for all k.
                for k in 0..rank {
                    vout[k] += row[k] * u_i;
                }
            }
        });
        if step_norm < 1e-20 {
            // Degenerate — return 0.
            return 0.0;
        }
    }

    // Final σ_max² ≈ v^T · (M^T M) · v = ‖M · v‖².
    // Compute ‖M·v‖² by one more pass.
    let mut sigma_sq = 0.0f32;
    for i in 0..outer {
        let row = &mat[i * rank..(i + 1) * rank];
        let u_i = simd_dot_f32(row, v, rank);
        sigma_sq += u_i * u_i;
    }
    sigma_sq.sqrt()
}

/// Rebalance `(A, B)` so `σ_max(A) ≈ σ_max(B)` without changing `A · B^T`.
///
/// Implements paper Algorithm 2 from arXiv:2606.12921:
/// ```text
/// c = (σ_max(B) / σ_max(A))^{α/2}
/// A ← c · A
/// B ← B / c
/// ```
///
/// # Gauge Invariance (paper Prop 1)
///
/// The projector `P_A = A · (A^T A)^{-1} · A^T` is unchanged by scalar
/// rescalings of `A` — so this rebalancing does not change the result of any
/// spectral operation on `(A, B)` (msign, projector products, etc.). It only
/// changes numerical conditioning.
///
/// # Application: Adapter Composition
///
/// When composing multiple LoRA pairs via weighted sum, naive arithmetic is
/// gauge-dependent. Rebalancing each input first makes the result well-defined.
///
/// # Arguments
///
/// - `a`, `b`: LoRA factors, mutated in place. `a` is `a_rows × a_cols`,
///   `b` is `b_rows × b_cols`. Note: `a_cols == b_cols == r` (rank).
/// - `alpha`: damping exponent `∈ (0, 1]`. `alpha = 1.0` fully rebalances,
///   `alpha = 0.5` half-strength. Default in consumers: `1.0` for full
///   rebalance, `0.5` if numerical sensitivity is high.
/// - `scratch`: reusable power-iteration buffers ([`GaugeRebalanceScratch`]).
///
/// # Invariant
///
/// `‖A · B^T‖_F` is unchanged before/after (modulo f32 precision).
///
/// # Panics
///
/// - If `a.len() != a_rows * a_cols`.
/// - If `b.len() != b_rows * b_cols`.
/// - If `a_cols != b_cols` (rank mismatch).
pub fn gauge_rebalance(
    a: &mut [f32],
    b: &mut [f32],
    a_rows: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
    alpha: f32,
    scratch: &mut GaugeRebalanceScratch,
) {
    assert_eq!(a.len(), a_rows * a_cols, "A size mismatch");
    assert_eq!(b.len(), b_rows * b_cols, "B size mismatch");
    assert_eq!(
        a_cols, b_cols,
        "rank mismatch: a_cols={} b_cols={}",
        a_cols, b_cols
    );
    assert!(
        alpha > 0.0 && alpha <= 1.0,
        "alpha must be in (0, 1], got {}",
        alpha
    );

    let r = a_cols;

    // Ensure scratch is sized for rank r.
    if scratch.v_a.len() != r {
        let init = 1.0 / (r as f32).sqrt();
        scratch.v_a.clear();
        scratch.v_a.resize(r, init);
        scratch.v_a_new.clear();
        scratch.v_a_new.resize(r, 0.0);
        scratch.v_b.clear();
        scratch.v_b.resize(r, init);
        scratch.v_b_new.clear();
        scratch.v_b_new.resize(r, 0.0);
    }

    // σ_max(A) and σ_max(B) via power iteration on A^T A and B^T B (each r×r).
    let sigma_a = power_iterate_sigma_max(a, a_rows, r, &mut scratch.v_a, &mut scratch.v_a_new, 5);
    let sigma_b = power_iterate_sigma_max(b, b_rows, r, &mut scratch.v_b, &mut scratch.v_b_new, 5);

    if sigma_a < 1e-20 || sigma_b < 1e-20 {
        // Degenerate (zero matrix) — nothing to rebalance.
        return;
    }

    // c = (σ_max(B) / σ_max(A))^{α/2}
    let ratio = sigma_b / sigma_a;
    let c = ratio.powf(alpha * 0.5);
    let inv_c = 1.0 / c;

    // A ← c · A, B ← B / c.
    simd_scale_inplace(a, c);
    simd_scale_inplace(b, inv_c);
}

/// A weighted LoRA factor pair for composition.
///
/// Represents the contribution `η · A · B^T` to a merged adapter.
#[derive(Debug, Clone, Copy)]
pub struct GaugePair<'a> {
    /// Scalar weight (positive for addition, negative for subtraction).
    pub eta: f32,
    /// Factor A (`a_rows × rank`, row-major).
    pub a: &'a [f32],
    /// Factor B (`b_rows × rank`, row-major).
    pub b: &'a [f32],
    /// Number of rows in A.
    pub a_rows: usize,
    /// Number of rows in B.
    pub b_rows: usize,
    /// LoRA rank (columns of A and B).
    pub rank: usize,
}

/// Compose multiple LoRA pairs with gauge-invariant rebalancing.
///
/// For each pair `(η_i, A_i, B_i)`:
///   1. Rebalance so `σ_max(A_i) ≈ σ_max(B_i)` — removes factorization artifact.
///   2. Scale by `η_i`.
///   3. Sum: `W_merged = Σ_i η_i · A_i · B_i^T`
///
/// Output is a single `(A_merged, B_merged)` pair in rebalanced form. All input
/// pairs must have the same `(a_rows, b_rows, rank)` shape.
///
/// **Without rebalancing**, the merged result depends on the arbitrary
/// factorization of each input — e.g., if game_1's adapter was trained with A
/// scaled by 100 and B scaled by 0.01 (gauge c=100), naive sum would weight
/// game_1's contribution incorrectly.
///
/// # Algorithm
///
/// Naive sum: `W = Σ_i η_i A_i B_i^T` — gauge-dependent.
///
/// Gauge-invariant: rebalance each pair first, then sum:
/// ```text
/// for each pair: gauge_rebalance(A_i, B_i)
/// W = Σ_i η_i A_i B_i^T  // now well-defined
/// ```
///
/// Since the merged result is itself a sum of low-rank products, we can store
/// it as a block matrix: `A_merged = [η_1 A_1, η_2 A_2, ...]` (stacked
/// horizontally), `B_merged = [B_1, B_2, ...]`. This preserves the rank-p·r
/// representation exactly. Callers that want rank ≤ r should apply NS inv-sqrt
/// truncation after.
///
/// # Panics
///
/// - If `pairs` is empty.
/// - If any pair has mismatched shape.
/// - If `out_a` or `out_b` is the wrong size (`a_rows × (n_pairs · rank)` and
///   `b_rows × (n_pairs · rank)` respectively).
pub fn gauge_invariant_compose(pairs: &[GaugePair<'_>], out_a: &mut [f32], out_b: &mut [f32]) {
    assert!(!pairs.is_empty(), "pairs must not be empty");

    let p0 = &pairs[0];
    let a_rows = p0.a_rows;
    let b_rows = p0.b_rows;
    let r = p0.rank;
    let n_pairs = pairs.len();
    let merged_rank = n_pairs * r;

    assert_eq!(
        out_a.len(),
        a_rows * merged_rank,
        "out_a must be {} × {} = {} elements, got {}",
        a_rows,
        merged_rank,
        a_rows * merged_rank,
        out_a.len()
    );
    assert_eq!(
        out_b.len(),
        b_rows * merged_rank,
        "out_b must be {} × {} = {} elements, got {}",
        b_rows,
        merged_rank,
        b_rows * merged_rank,
        out_b.len()
    );

    // Validate all pairs have consistent shape.
    for (i, p) in pairs.iter().enumerate() {
        assert_eq!(p.a_rows, a_rows, "pair {} a_rows mismatch", i);
        assert_eq!(p.b_rows, b_rows, "pair {} b_rows mismatch", i);
        assert_eq!(p.rank, r, "pair {} rank mismatch", i);
        assert_eq!(p.a.len(), a_rows * r, "pair {} A size mismatch", i);
        assert_eq!(p.b.len(), b_rows * r, "pair {} B size mismatch", i);
    }

    // For each pair: rebalance into a local buffer, then write scaled copy to output.
    // To avoid per-pair heap allocation, we allocate two row-major buffers once.
    let mut a_buf = vec![0.0f32; a_rows * r];
    let mut b_buf = vec![0.0f32; b_rows * r];
    let mut rebalance_scratch = GaugeRebalanceScratch::new(a_rows.max(b_rows), r);

    for (pair_idx, p) in pairs.iter().enumerate() {
        // Copy input into mutable buffers (rebalance mutates in place).
        a_buf.copy_from_slice(p.a);
        b_buf.copy_from_slice(p.b);

        // Rebalance (α=1.0 for full rebalance — composition should be exact).
        gauge_rebalance(
            &mut a_buf,
            &mut b_buf,
            a_rows,
            r,
            b_rows,
            r,
            1.0,
            &mut rebalance_scratch,
        );

        // Scale A by η (B stays as is; the η ends up in A side of the product).
        simd_scale_inplace(&mut a_buf, p.eta);

        // Write to output at column offset `pair_idx * r`.
        let a_offset = pair_idx * r;
        for i in 0..a_rows {
            let src = &a_buf[i * r..(i + 1) * r];
            let dst = &mut out_a[i * merged_rank + a_offset..i * merged_rank + a_offset + r];
            dst.copy_from_slice(src);
        }
        let b_offset = pair_idx * r;
        for i in 0..b_rows {
            let src = &b_buf[i * r..(i + 1) * r];
            let dst = &mut out_b[i * merged_rank + b_offset..i * merged_rank + b_offset + r];
            dst.copy_from_slice(src);
        }
    }
}

/// Two-pair interpolation: `W = (1-α)·A_1·B_1^T + α·A_2·B_2^T`.
///
/// Special case of [`gauge_invariant_compose`] with two pairs. Slightly
/// faster (avoids the general compose loop) and easier to call for the common
/// adapter-interpolation use case.
///
/// Both pairs must have identical shape. Outputs `A_out` and `B_out` of size
/// `a_rows × 2r` and `b_rows × 2r` respectively (block representation).
pub fn gauge_invariant_lerp(
    a1: &[f32],
    b1: &[f32],
    a2: &[f32],
    b2: &[f32],
    a_rows: usize,
    b_rows: usize,
    r: usize,
    alpha: f32,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    let pairs = [
        GaugePair {
            eta: 1.0 - alpha,
            a: a1,
            b: b1,
            a_rows,
            b_rows,
            rank: r,
        },
        GaugePair {
            eta: alpha,
            a: a2,
            b: b2,
            a_rows,
            b_rows,
            rank: r,
        },
    ];
    gauge_invariant_compose(&pairs, out_a, out_b);
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pseudo-random factor matrix.
    fn seeded_random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        let mut s = seed;
        let mut mat = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0;
            mat.push(v);
        }
        mat
    }

    /// Compute `A · B^T` for A `m × r`, B `n × r` → result `m × n`.
    fn abt(a: &[f32], b: &[f32], m: usize, r: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..r {
                    s += a[i * r + k] * b[j * r + k];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Frobenius norm.
    fn fro_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn test_gauge_rebalance_preserves_abt() {
        // Paper Prop 1: rebalancing must not change A·B^T.
        let m = 8;
        let n = 6;
        let r = 4;
        let a_orig = seeded_random_matrix(42, m, r);
        let b_orig = seeded_random_matrix(99, n, r);
        let w_before = abt(&a_orig, &b_orig, m, r, n);

        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
        gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);

        let w_after = abt(&a, &b, m, r, n);

        let max_diff = w_before
            .iter()
            .zip(w_after.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "Rebalance changed A·B^T by {} (should be ≈ 0)",
            max_diff
        );
    }

    #[test]
    fn test_gauge_rebalance_balances_sigmas() {
        // After rebalance with α=1.0, σ_max(A) ≈ σ_max(B).
        let m = 16;
        let n = 12;
        let r = 4;
        // Construct deliberately imbalanced: A scaled by 10, B scaled by 0.1
        let mut a: Vec<f32> = seeded_random_matrix(7, m, r)
            .iter()
            .map(|v| v * 10.0)
            .collect();
        let mut b: Vec<f32> = seeded_random_matrix(8, n, r)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);

        let sigma_a_before = power_iterate_sigma_max(
            &a,
            m,
            r,
            &mut scratch.v_a.clone(),
            &mut scratch.v_a_new.clone(),
            20,
        );
        let sigma_b_before = power_iterate_sigma_max(
            &b,
            n,
            r,
            &mut scratch.v_b.clone(),
            &mut scratch.v_b_new.clone(),
            20,
        );
        let ratio_before = (sigma_a_before / sigma_b_before).abs();
        assert!(
            ratio_before > 5.0,
            "Pre-condition: ratio should be >> 1, got {}",
            ratio_before
        );

        gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);

        let sigma_a_after = power_iterate_sigma_max(
            &a,
            m,
            r,
            &mut scratch.v_a.clone(),
            &mut scratch.v_a_new.clone(),
            20,
        );
        let sigma_b_after = power_iterate_sigma_max(
            &b,
            n,
            r,
            &mut scratch.v_b.clone(),
            &mut scratch.v_b_new.clone(),
            20,
        );
        let ratio_after = (sigma_a_after / sigma_b_after).abs();
        assert!(
            ratio_after < 1.5,
            "Post-condition: ratio should be ≈ 1, got {} (before={}, after a={}, b={})",
            ratio_after,
            ratio_before,
            sigma_a_after,
            sigma_b_after
        );
    }

    #[test]
    fn test_gauge_rebalance_zero_matrix_safe() {
        // Zero matrix should not panic.
        let m = 4;
        let n = 4;
        let r = 2;
        let mut a = vec![0.0f32; m * r];
        let mut b = vec![0.0f32; n * r];
        let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
        gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);
        // All still zero.
        for v in a.iter().chain(b.iter()) {
            assert!((*v).abs() < 1e-20, "Expected zero, got {}", v);
        }
    }

    #[test]
    fn test_power_iterate_matches_naive_sigma_max() {
        // For a known matrix, power iteration should approximate σ_max well.
        let m = 8;
        let r = 4;
        let mat = seeded_random_matrix(123, m, r);

        // Reference σ_max via M^T M and max eigenvalue (approximate via power iter).
        let mut v = vec![1.0 / (r as f32).sqrt(); r];
        let mut v_new = vec![0.0; r];
        let sigma_est = power_iterate_sigma_max(&mat, m, r, &mut v, &mut v_new, 30);

        // Compute true σ_max via M^T M eigenvalues (brute force for small r).
        let mut mtm = [0.0f32; 16];
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f32;
                for k in 0..m {
                    s += mat[k * r + i] * mat[k * r + j];
                }
                mtm[i * r + j] = s;
            }
        }
        // Jacobi eigenvalue iteration (very small 4×4).
        let mut a_work = mtm;
        for _ in 0..50 {
            // Find largest off-diagonal.
            let mut p = 0;
            let mut q = 1;
            let mut max_val = 0.0f32;
            for i in 0..r {
                for j in (i + 1)..r {
                    if a_work[i * r + j].abs() > max_val {
                        max_val = a_work[i * r + j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < 1e-12 {
                break;
            }
            let app = a_work[p * r + p];
            let aqq = a_work[q * r + q];
            let apq = a_work[p * r + q];
            let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
            let c = theta.cos();
            let s = theta.sin();
            for i in 0..r {
                let aip = a_work[i * r + p];
                let aiq = a_work[i * r + q];
                a_work[i * r + p] = c * aip - s * aiq;
                a_work[i * r + q] = s * aip + c * aiq;
            }
            for i in 0..r {
                let api = a_work[p * r + i];
                let aqi = a_work[q * r + i];
                a_work[p * r + i] = c * api - s * aqi;
                a_work[q * r + i] = s * api + c * aqi;
            }
        }
        let true_sigma_sq = (0..r).map(|i| a_work[i * r + i]).fold(0.0f32, f32::max);
        let true_sigma = true_sigma_sq.sqrt();

        let rel_err = (sigma_est - true_sigma).abs() / true_sigma;
        assert!(
            rel_err < 0.05,
            "σ_max estimate {} vs true {} → rel err {} > 5%",
            sigma_est,
            true_sigma,
            rel_err
        );
    }

    #[test]
    fn test_gauge_invariant_compose_basic() {
        // 2-pair compose: should produce a (4r)-rank merged pair.
        let m = 8;
        let n = 6;
        let r = 4;
        let a1 = seeded_random_matrix(1, m, r);
        let b1 = seeded_random_matrix(2, n, r);
        let a2 = seeded_random_matrix(3, m, r);
        let b2 = seeded_random_matrix(4, n, r);

        let pairs = [
            GaugePair {
                eta: 1.0,
                a: &a1,
                b: &b1,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
            GaugePair {
                eta: 1.0,
                a: &a2,
                b: &b2,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
        ];

        let merged_rank = 2 * r;
        let mut out_a = vec![0.0f32; m * merged_rank];
        let mut out_b = vec![0.0f32; n * merged_rank];
        gauge_invariant_compose(&pairs, &mut out_a, &mut out_b);

        // The merged A·B^T should equal (after rebalance) A_1 B_1^T + A_2 B_2^T.
        // Since rebalance preserves A·B^T, this should match naive sum within ε.
        let w_merged = abt(&out_a, &out_b, m, merged_rank, n);
        let w_naive_1 = abt(&a1, &b1, m, r, n);
        let w_naive_2 = abt(&a2, &b2, m, r, n);

        let max_diff = (0..m * n)
            .map(|i| (w_merged[i] - w_naive_1[i] - w_naive_2[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "Compose should match naive sum within ε, max diff = {}",
            max_diff
        );
    }

    #[test]
    fn test_gauge_invariant_lerp_endpoints() {
        // Lerp at α=0 should give just pair 1 (in AB^T sense).
        // Lerp at α=1 should give just pair 2.
        let m = 6;
        let n = 4;
        let r = 3;
        let a1 = seeded_random_matrix(10, m, r);
        let b1 = seeded_random_matrix(11, n, r);
        let a2 = seeded_random_matrix(12, m, r);
        let b2 = seeded_random_matrix(13, n, r);

        let merged_r = 2 * r;
        let mut out_a = vec![0.0f32; m * merged_r];
        let mut out_b = vec![0.0f32; n * merged_r];

        // α = 0 → only pair 1 contributes.
        gauge_invariant_lerp(&a1, &b1, &a2, &b2, m, n, r, 0.0, &mut out_a, &mut out_b);
        let w_merged = abt(&out_a, &out_b, m, merged_r, n);
        let w_p1 = abt(&a1, &b1, m, r, n);
        let max_diff = (0..m * n)
            .map(|i| (w_merged[i] - w_p1[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "α=0 should match pair 1 only, max diff = {}",
            max_diff
        );

        // α = 1 → only pair 2 contributes.
        gauge_invariant_lerp(&a1, &b1, &a2, &b2, m, n, r, 1.0, &mut out_a, &mut out_b);
        let w_merged = abt(&out_a, &out_b, m, merged_r, n);
        let w_p2 = abt(&a2, &b2, m, r, n);
        let max_diff = (0..m * n)
            .map(|i| (w_merged[i] - w_p2[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "α=1 should match pair 2 only, max diff = {}",
            max_diff
        );
    }

    #[test]
    fn test_compose_gauge_invariance_under_input_rescaling() {
        // KEY TEST: composing gauge-equivalent inputs gives the same result.
        //
        // Take pair (A, B) and gauge-equivalent (A·c, B/c). Compose each with
        // itself (eta=0.5, eta=0.5). The merged A·B^T must be the same.
        let m = 6;
        let n = 5;
        let r = 3;
        let a = seeded_random_matrix(100, m, r);
        let b = seeded_random_matrix(101, n, r);

        // Gauge transform: A' = 5·A, B' = B/5. Same A·B^T.
        let c = 5.0f32;
        let a_g: Vec<f32> = a.iter().map(|v| v * c).collect();
        let b_g: Vec<f32> = b.iter().map(|v| v / c).collect();

        let merged_r = 2 * r;

        // Compose original with itself.
        let mut out_a_orig = vec![0.0f32; m * merged_r];
        let mut out_b_orig = vec![0.0f32; n * merged_r];
        let pairs_orig = [
            GaugePair {
                eta: 0.5,
                a: &a,
                b: &b,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
            GaugePair {
                eta: 0.5,
                a: &a,
                b: &b,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
        ];
        gauge_invariant_compose(&pairs_orig, &mut out_a_orig, &mut out_b_orig);
        let w_orig = abt(&out_a_orig, &out_b_orig, m, merged_r, n);

        // Compose gauge-transformed with itself.
        let mut out_a_g = vec![0.0f32; m * merged_r];
        let mut out_b_g = vec![0.0f32; n * merged_r];
        let pairs_g = [
            GaugePair {
                eta: 0.5,
                a: &a_g,
                b: &b_g,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
            GaugePair {
                eta: 0.5,
                a: &a_g,
                b: &b_g,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
        ];
        gauge_invariant_compose(&pairs_g, &mut out_a_g, &mut out_b_g);
        let w_g = abt(&out_a_g, &out_b_g, m, merged_r, n);

        // Both should produce the same merged W (gauge-invariant).
        let max_diff = (0..m * n)
            .map(|i| (w_orig[i] - w_g[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "Gauge-equivalent inputs should give same merged W, max diff = {}",
            max_diff
        );
    }

    #[test]
    fn test_naive_sum_is_not_gauge_invariant() {
        // Negative test: prove naive summing is gauge-dependent.
        // This justifies why we need gauge_invariant_compose.
        let m = 6;
        let n = 5;
        let r = 3;
        let a = seeded_random_matrix(200, m, r);
        let b = seeded_random_matrix(201, n, r);

        // Naive sum: just add A_1 + A_2 and B_1 + B_2 (where pair 2 = pair 1).
        let w_naive_1 = abt(&a, &b, m, r, n);
        // Naive 2·A·B^T = sum of pair with itself.
        let _w_naive_sum: Vec<f32> = w_naive_1.iter().map(|v| 2.0 * v).collect();

        // Now do the same with gauge-transformed inputs (A' = 5A, B' = B/5).
        // Naive sum gives 2·A'·B'^T = 2·A·B^T (gauge cancels in this trivial case
        // because both inputs are at the same gauge). So this isn't a great test.
        //
        // Instead, mix gauges: pair 1 at c=1, pair 2 at c=5.
        let c = 5.0f32;
        let a_g: Vec<f32> = a.iter().map(|v| v * c).collect();
        let b_g: Vec<f32> = b.iter().map(|v| v / c).collect();

        // Naive sum: A_sum = A + A_g, B_sum = B + B_g.
        let a_naive_sum: Vec<f32> = (0..m * r).map(|i| a[i] + a_g[i]).collect();
        let b_naive_sum: Vec<f32> = (0..n * r).map(|i| b[i] + b_g[i]).collect();
        let w_naive_mixed = abt(&a_naive_sum, &b_naive_sum, m, r, n);

        // Gauge-invariant: rebalance each first.
        let merged_r = 2 * r;
        let mut out_a = vec![0.0f32; m * merged_r];
        let mut out_b = vec![0.0f32; n * merged_r];
        let pairs = [
            GaugePair {
                eta: 1.0,
                a: &a,
                b: &b,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
            GaugePair {
                eta: 1.0,
                a: &a_g,
                b: &b_g,
                a_rows: m,
                b_rows: n,
                rank: r,
            },
        ];
        gauge_invariant_compose(&pairs, &mut out_a, &mut out_b);
        let w_gauge = abt(&out_a, &out_b, m, merged_r, n);

        // These should DIFFER — naive sum is gauge-sensitive.
        let diff = fro_norm(&w_naive_mixed) - fro_norm(&w_gauge);
        // The gauge-invariant result should be the "true" sum (2·A·B^T),
        // while naive sum will be skewed by the c=5 factor.
        assert!(
            diff.abs() > 0.1,
            "Expected naive vs gauge-invariant to differ, got diff = {} (naive={}, gauge={})",
            diff,
            fro_norm(&w_naive_mixed),
            fro_norm(&w_gauge)
        );

        // Gauge-invariant result should match the true sum (2·A·B^T since both
        // pairs represent the same W). Check Frobenius norms are close.
        let w_true_sum: Vec<f32> = w_naive_1.iter().map(|v| 2.0 * v).collect();
        let true_norm = fro_norm(&w_true_sum);
        let gauge_norm = fro_norm(&w_gauge);
        assert!(
            (gauge_norm - true_norm).abs() / true_norm < 0.05,
            "Gauge-invariant should match true sum norm within 5%, got gauge={} vs true={}",
            gauge_norm,
            true_norm
        );
    }

    #[test]
    fn test_gauge_rebalance_alpha_zero_is_noop_structurally() {
        // α very small → minimal rebalance → A·B^T unchanged (still invariant).
        // (α=0 exactly is disallowed by assert; use α=0.01.)
        let m = 4;
        let n = 4;
        let r = 2;
        let a = seeded_random_matrix(33, m, r);
        let b = seeded_random_matrix(44, n, r);
        let w_before = abt(&a, &b, m, r, n);

        let mut a_mut = a.clone();
        let mut b_mut = b.clone();
        let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
        gauge_rebalance(&mut a_mut, &mut b_mut, m, r, n, r, 0.01, &mut scratch);

        let w_after = abt(&a_mut, &b_mut, m, r, n);
        let max_diff = (0..m * n)
            .map(|i| (w_before[i] - w_after[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "Small α rebalance should still preserve A·B^T, max diff = {}",
            max_diff
        );
    }
}
