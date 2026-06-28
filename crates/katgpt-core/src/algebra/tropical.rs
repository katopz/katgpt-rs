//! Tropical `(max, +)` semiring primitives for modelless inference.
//!
//! Distilled from Smets, *Mathematics of Neural Networks* (arXiv:2403.04807),
//! Chapter 3 §3.5 — see Research 321 §1.2 for the full distillation.
//!
//! # The `(max, +)` semiring
//!
//! The tropical (max-plus) semiring is `ℝ_max = (ℝ ∪ {−∞}, max, +)` with
//! additive identity `𝟘 = −∞` and multiplicative identity `𝟙 = 0`. It is
//! idempotent (`a ⊕ a = a`) and commutative. Every latent aggregation
//! primitive shipped today operates over the standard `(ℝ, +, ·)` ring
//! (dot products, matvecs, sigmoid projections). The `(max, +)` algebra is
//! a genuinely different aggregation substrate — a *worst-case* / *bottleneck*
//! algebra rather than an *average* / *expected* algebra.
//!
//! # Textbook identities
//!
//! - **ReLU is tropically affine.** `ReLU(x) = max(x, 0) = x ⊕ 0` in `ℝ_max`
//!   (Smets Example 3.49). A ReLU NN alternates between two semirings: `(ℝ,+,·)`
//!   for the matmul, `(max,+)` for the activation.
//! - **Max-pool is tropical convolution** with a window kernel
//!   `(Tf)(y) = sup_{x ∈ y+S} f(x)` (Smets Example 3.55).
//! - **Tropical matvec** `(W ⊗ x)_i = max_j (W[i,j] + x[j])` is the `(max,+)`
//!   analog of the standard matvec `(W · x)_i = Σ_j W[i,j] · x[j]`. It selects
//!   the *most-aligned* feature pair per output row rather than averaging.
//!
//! # DEC wrappers
//!
//! The three DEC wrappers below give "worst-case" variants of the linear
//! `d` / `δ` / `line_integral` operators:
//!
//! - [`tropical_exterior_derivative`] — for each (k+1)-cell, the max of the
//!   `+1`-signed boundary k-cell values (the forward endpoint), rather than
//!   the signed sum (gradient). Semantically: "the value at the forward
//!   boundary" rather than "the difference across the boundary".
//! - [`tropical_codifferential`] — max of `+1`-signed boundary contributions
//!   in the k→k−1 direction.
//! - [`tropical_line_integral`] — the bottleneck (max) edge weight along a
//!   path, rather than the total (sum) work.
//!
//! # Performance
//!
//! `max` + `+` reductions, zero allocation in the `_into` variants. The inner
//! column/channel loops are unrolled 4-wide to help LLVM auto-vectorize,
//! mirroring the pattern in `dec::operators::exterior_derivative_into`.

#[cfg(feature = "dec_operators")]
use crate::dec::{CellComplex, CochainField, MAX_RANK};

// ---------------------------------------------------------------------------
// Core primitives — (max, +) matvec and dot
// ---------------------------------------------------------------------------

/// Tropical matvec into a caller-provided buffer: `(W ⊗ x)_i = max_j (W[i,j] + x[j])`.
///
/// `w_row_major` is `n_rows × n_cols` in row-major order. `out` must be at
/// least `n_rows` long. The buffer is filled with `f32::NEG_INFINITY` (the
/// additive identity of `(max, +)`) before accumulation — the caller does NOT
/// need to pre-clear.
///
/// Zero allocations. Dispatches to a NEON / scalar-specialised inner loop.
/// All paths use **multiple independent accumulators** (NEON: four
/// `float32x4_t`; scalar: four `f32`) to hide `f32::max` latency — a single
/// serial `acc = acc.max(...)` chain is latency-bound and 4–9× slower than
/// `simd_matvec` on the same shape (Plan 337 Phase 3 T3.3 G2 finding). The
/// 4-accumulator tree reduction mirrors `simd_dot_f32`'s pattern in
/// `katgpt-types/src/simd/dot.rs`.
#[inline]
pub fn tropical_matvec_into(
    w_row_major: &[f32],
    x: &[f32],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(
        w_row_major.len() >= n_rows * n_cols,
        "tropical_matvec_into: w.len={} < n_rows*n_cols={}",
        w_row_major.len(),
        n_rows * n_cols
    );
    debug_assert!(
        x.len() >= n_cols,
        "tropical_matvec_into: x.len={} < n_cols={}",
        x.len(),
        n_cols
    );
    debug_assert!(
        out.len() >= n_rows,
        "tropical_matvec_into: out.len={} < n_rows={}",
        out.len(),
        n_rows
    );

    // Additive identity of (max, +) is −∞, NOT 0.0.
    out[..n_rows].fill(f32::NEG_INFINITY);

    if n_cols == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        for (i, out_slot) in out.iter_mut().enumerate().take(n_rows) {
            // stride math: row_off = i * n_cols
            let row_off = i * n_cols;
            unsafe { *out_slot = neon_tropical_row_max_sum(&w_row_major[row_off..], x, n_cols) };
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (i, out_slot) in out.iter_mut().enumerate().take(n_rows) {
            // stride math: row_off = i * n_cols
            let row_off = i * n_cols;
            *out_slot = scalar_tropical_row_max_sum(&w_row_major[row_off..], x, n_cols);
        }
    }
}

/// Scalar (max, +) row reduction with 4 independent accumulators.
///
/// Mirrors `scalar_dot_f32`'s 4-accumulator pattern from
/// `katgpt-types/src/simd/dot.rs`: a single-accumulator `acc = acc.max(...)`
/// chain is latency-bound on `f32::max` (~1–4 cycles per op). Four parallel
/// accumulators keep the FP add pipeline full.
///
/// On `aarch64` this is unused (the NEON path dispatches instead) but kept
/// compiled as the portable reference + non-aarch64 fallback.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn scalar_tropical_row_max_sum(w_row: &[f32], x: &[f32], n: usize) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 4];
    let chunks = n / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] = acc[0].max(*w_row.get_unchecked(i) + *x.get_unchecked(i));
            acc[1] = acc[1].max(*w_row.get_unchecked(i + 1) + *x.get_unchecked(i + 1));
            acc[2] = acc[2].max(*w_row.get_unchecked(i + 2) + *x.get_unchecked(i + 2));
            acc[3] = acc[3].max(*w_row.get_unchecked(i + 3) + *x.get_unchecked(i + 3));
        }
        i += 4;
    }
    let mut m = acc[0].max(acc[1]).max(acc[2]).max(acc[3]);
    while i < n {
        unsafe {
            m = m.max(*w_row.get_unchecked(i) + *x.get_unchecked(i));
        }
        i += 1;
    }
    m
}

/// NEON `(max, +)` row reduction — 4 independent `float32x4_t` accumulators.
///
/// Mirrors `neon_dot_f32` from `katgpt-types/src/simd/dot.rs`:
/// - 4 × `float32x4_t` = 16 lanes in flight per outer iteration (hides
///   `vmaxq_f32` latency, ~2–3 cycles on Cortex-A / Apple Silicon).
/// - `vaddq_f32` does the tropical "product" (the `+`), `vmaxq_f32` does the
///   tropical "sum" (the `max`).
/// - Horizontal max reduce via `vmaxvq_f32` at the end.
///
/// # Safety
/// Caller must guarantee `w_row` and `x` each have at least `n` readable
/// `f32`s. The 16-element main loop reads in 4-wide chunks; the tail is
/// handled by the 4-wide and scalar fallbacks.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_tropical_row_max_sum(w_row: &[f32], x: &[f32], n: usize) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32};

    unsafe {
        let neg_inf = f32::NEG_INFINITY;
        let mut acc0 = vdupq_n_f32(neg_inf);
        let mut acc1 = vdupq_n_f32(neg_inf);
        let mut acc2 = vdupq_n_f32(neg_inf);
        let mut acc3 = vdupq_n_f32(neg_inf);
        let mut i = 0usize;
        let chunks16 = n / 16;

        for _ in 0..chunks16 {
            acc0 = vmaxq_f32(
                acc0,
                vaddq_f32(
                    vld1q_f32(w_row.as_ptr().add(i)),
                    vld1q_f32(x.as_ptr().add(i)),
                ),
            );
            acc1 = vmaxq_f32(
                acc1,
                vaddq_f32(
                    vld1q_f32(w_row.as_ptr().add(i + 4)),
                    vld1q_f32(x.as_ptr().add(i + 4)),
                ),
            );
            acc2 = vmaxq_f32(
                acc2,
                vaddq_f32(
                    vld1q_f32(w_row.as_ptr().add(i + 8)),
                    vld1q_f32(x.as_ptr().add(i + 8)),
                ),
            );
            acc3 = vmaxq_f32(
                acc3,
                vaddq_f32(
                    vld1q_f32(w_row.as_ptr().add(i + 12)),
                    vld1q_f32(x.as_ptr().add(i + 12)),
                ),
            );
            i += 16;
        }

        let acc01 = vmaxq_f32(acc0, acc1);
        let acc23 = vmaxq_f32(acc2, acc3);
        let mut merged = vmaxq_f32(acc01, acc23);
        let mut m = vmaxvq_f32(merged);

        let remaining_4 = (n - i) / 4;
        for _ in 0..remaining_4 {
            merged = vmaxq_f32(
                merged,
                vaddq_f32(
                    vld1q_f32(w_row.as_ptr().add(i)),
                    vld1q_f32(x.as_ptr().add(i)),
                ),
            );
            i += 4;
        }
        m = m.max(vmaxvq_f32(merged));

        while i < n {
            m = m.max(*w_row.get_unchecked(i) + *x.get_unchecked(i));
            i += 1;
        }
        m
    }
}

/// Tropical dot product into a caller-provided scalar:
/// `max_j (a[j] + b[j])`.
///
/// `*out` is set to `f32::NEG_INFINITY` before accumulation. Zero allocation.
#[inline]
pub fn tropical_dot_into(a: &[f32], b: &[f32], out: &mut f32, n: usize) {
    debug_assert!(
        a.len() >= n,
        "tropical_dot_into: a.len={} < n={}",
        a.len(),
        n
    );
    debug_assert!(
        b.len() >= n,
        "tropical_dot_into: b.len={} < n={}",
        b.len(),
        n
    );

    *out = f32::NEG_INFINITY;

    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let off = c * 4;
        *out = (*out).max(a[off] + b[off]);
        *out = (*out).max(a[off + 1] + b[off + 1]);
        *out = (*out).max(a[off + 2] + b[off + 2]);
        *out = (*out).max(a[off + 3] + b[off + 3]);
    }
    for d in 0..remainder {
        let off = chunks * 4 + d;
        *out = (*out).max(a[off] + b[off]);
    }
}

/// Allocating wrapper for [`tropical_matvec_into`] — convenience for cold
/// paths and tests. Returns a fresh `Vec<f32>` of length `n_rows`.
#[inline]
pub fn tropical_matvec(w: &[f32], x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; n_rows];
    tropical_matvec_into(w, x, &mut out, n_rows, n_cols);
    out
}

// ---------------------------------------------------------------------------
// DEC wrappers — tropical (max, +) variants of d / δ / line_integral
// ---------------------------------------------------------------------------

/// Tropical exterior derivative: for each (k+1)-cell, the max of the
/// `+1`-signed boundary k-cell values.
///
/// Mirrors [`crate::dec::exterior_derivative`] but swaps the inner reduction
/// from `Σ sign·input` to `max(input)` over the `+1`-signed (head) boundary
/// cells only. Signed `+1` → include the input value; signed `−1` → exclude
/// (contribute `−∞`, the additive identity). Semantically this gives "the
/// value at the forward boundary" rather than "the signed difference".
///
/// For a rank-0 vertex cochain, `tropical_exterior_derivative` gives, for each
/// edge, the value at the head vertex — a directed max-aggregation.
#[cfg(feature = "dec_operators")]
#[inline]
pub fn tropical_exterior_derivative(cx: &CellComplex, input: &CochainField) -> CochainField {
    let k = input.rank;
    assert!(
        k < MAX_RANK,
        "tropical_exterior_derivative: rank {k} has no dₖ (max rank is {MAX_RANK})"
    );

    let target_rank = k + 1;
    let n_output = cx.n_cells(target_rank);
    let dim = input.dim;
    let mut output = CochainField::zeros(target_rank, n_output, dim);

    debug_assert_eq!(
        output.data.len(),
        n_output * dim,
        "tropical_exterior_derivative: output buffer size mismatch"
    );

    // Fill with additive identity of (max, +) = −∞ (NOT 0.0).
    output.data.fill(f32::NEG_INFINITY);

    let entries = cx.boundary_entries(k);

    let chunks = dim / 4;
    let remainder = dim % 4;

    // boundary_entries(k) triplets: (k-cell, (k+1)-cell, sign).
    // d_trop maps k-cells (src) to (k+1)-cells (dst): output[dst] = max(output[dst], input[src]) for sign>0.
    for &(src_cell, dst_cell, sign) in entries {
        if sign <= 0 {
            continue; // signed −1 → exclude (contribute −∞, no-op for max)
        }
        let src_start = src_cell * dim;
        let dst_start = dst_cell * dim;

        for c in 0..chunks {
            let off = c * 4;
            let v0 = input.data[src_start + off];
            let v1 = input.data[src_start + off + 1];
            let v2 = input.data[src_start + off + 2];
            let v3 = input.data[src_start + off + 3];
            output.data[dst_start + off] = output.data[dst_start + off].max(v0);
            output.data[dst_start + off + 1] = output.data[dst_start + off + 1].max(v1);
            output.data[dst_start + off + 2] = output.data[dst_start + off + 2].max(v2);
            output.data[dst_start + off + 3] = output.data[dst_start + off + 3].max(v3);
        }
        for d in 0..remainder {
            let off = chunks * 4 + d;
            output.data[dst_start + off] =
                output.data[dst_start + off].max(input.data[src_start + off]);
        }
    }

    output
}

/// Tropical codifferential: for each (k−1)-cell, the max of the
/// `+1`-signed boundary k-cell values.
///
/// Mirrors [`crate::dec::codifferential`] but swaps the inner reduction
/// from `Σ sign·input` to `max(input)` over the `+1`-signed boundary cells.
#[cfg(feature = "dec_operators")]
#[inline]
pub fn tropical_codifferential(cx: &CellComplex, input: &CochainField) -> CochainField {
    let k = input.rank;
    assert!(
        k > 0,
        "tropical_codifferential: rank {k} has no δₖ (rank must be > 0)"
    );

    let target_rank = k - 1;
    let n_output = cx.n_cells(target_rank);
    let dim = input.dim;
    let mut output = CochainField::zeros(target_rank, n_output, dim);

    debug_assert_eq!(
        output.data.len(),
        n_output * dim,
        "tropical_codifferential: output buffer size mismatch"
    );

    output.data.fill(f32::NEG_INFINITY);

    // codifferential_into iterates boundary_entries(k-1) and interprets
    // triplets as (dst=(k-1)-cell, src=k-cell, sign).
    let entries = cx.boundary_entries(k - 1);

    let chunks = dim / 4;
    let remainder = dim % 4;

    for &(dst_cell, src_cell, sign) in entries {
        if sign <= 0 {
            continue;
        }
        let src_start = src_cell * dim;
        let dst_start = dst_cell * dim;

        for c in 0..chunks {
            let off = c * 4;
            let v0 = input.data[src_start + off];
            let v1 = input.data[src_start + off + 1];
            let v2 = input.data[src_start + off + 2];
            let v3 = input.data[src_start + off + 3];
            output.data[dst_start + off] = output.data[dst_start + off].max(v0);
            output.data[dst_start + off + 1] = output.data[dst_start + off + 1].max(v1);
            output.data[dst_start + off + 2] = output.data[dst_start + off + 2].max(v2);
            output.data[dst_start + off + 3] = output.data[dst_start + off + 3].max(v3);
        }
        for d in 0..remainder {
            let off = chunks * 4 + d;
            output.data[dst_start + off] =
                output.data[dst_start + off].max(input.data[src_start + off]);
        }
    }

    output
}

/// Tropical line integral: the bottleneck (max) edge weight along a path.
///
/// Mirrors [`crate::dec::line_integral`] but replaces the sum with a max.
/// Returns `f32::NEG_INFINITY` for paths shorter than 2 vertices. Consecutive
/// vertex pairs not connected by an edge contribute nothing (the edge lookup
/// fails silently). Sign handling is preserved: traversal along orientation
/// contributes `+field[e]`, traversal against contributes `−field[e]`; the max
/// picks the largest signed contribution.
///
/// Semantically: the "worst edge" on the path (for threat/latency bottleneck
/// analysis) rather than the "total work".
#[cfg(feature = "dec_operators")]
#[inline]
pub fn tropical_line_integral(cx: &CellComplex, edge_field: &CochainField, path: &[u32]) -> f32 {
    if path.len() < 2 {
        return f32::NEG_INFINITY;
    }

    debug_assert_eq!(
        edge_field.rank, 1,
        "tropical_line_integral: edge_field must be rank-1 (edge) cochain, got rank {}",
        edge_field.rank
    );
    debug_assert_eq!(
        edge_field.dim, 1,
        "tropical_line_integral: edge_field must be dim=1 (scalar per edge), got dim {}",
        edge_field.dim
    );

    let entries = cx.boundary_entries(0);
    let mut total = f32::NEG_INFINITY;

    for window in path.windows(2) {
        let a = window[0] as usize;
        let b = window[1] as usize;
        if a == b {
            continue;
        }

        // B₁ entries from grid_2d are paired: (tail, e, −1), (head, e, +1).
        // Iterate pairs to find the edge connecting a and b.
        for pair in entries.chunks_exact(2) {
            let (v0, e0, _s0) = pair[0];
            let (v1, e1, _s1) = pair[1];
            debug_assert_eq!(e0, e1, "B₁ entries must be paired by edge index");

            if (v0 == a && v1 == b) || (v0 == b && v1 == a) {
                // Found edge e connecting a and b.
                // Contribution = field[e] · sign(b, e):
                //   b is head (sign=+1) → traversal along orientation → +field
                //   b is tail (sign=−1) → traversal against orientation → −field
                let sign_b = if v0 == b { pair[0].2 } else { pair[1].2 };
                let contribution = sign_b as f32 * edge_field.scalar(e0);
                total = total.max(contribution);
                break;
            }
        }
    }

    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Phase 1 unit tests ────────────────────────────────────────────────

    #[test]
    fn tropical_matvec_matches_definition() {
        // W = [[1,2,3],[4,5,6]], x = [10,20,30]
        // Row 0: max(1+10, 2+20, 3+30) = max(11,22,33) = 33
        // Row 1: max(4+10, 5+20, 6+30) = max(14,25,36) = 36
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [10.0f32, 20.0, 30.0];
        let out = tropical_matvec(&w, &x, 2, 3);
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 33.0).abs() < 1e-6,
            "row 0: expected 33, got {}",
            out[0]
        );
        assert!(
            (out[1] - 36.0).abs() < 1e-6,
            "row 1: expected 36, got {}",
            out[1]
        );
    }

    #[test]
    fn tropical_dot_is_max_sum() {
        // max(1+10, 2+20, 3+30) = max(11,22,33) = 33
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        let mut out = 0.0f32;
        tropical_dot_into(&a, &b, &mut out, 3);
        assert!((out - 33.0).abs() < 1e-6, "expected 33, got {}", out);
    }

    #[test]
    fn neg_inf_identity() {
        // W = [−∞, −∞], x = [5, 7] → out = [−∞] (1×2 case)
        let w = [f32::NEG_INFINITY, f32::NEG_INFINITY];
        let x = [5.0f32, 7.0];
        let out = tropical_matvec(&w, &x, 1, 2);
        assert_eq!(out.len(), 1);
        assert!(
            out[0].is_infinite() && out[0].is_sign_negative(),
            "expected −∞, got {}",
            out[0]
        );
    }

    #[test]
    fn relu_is_tropical_affine() {
        // tropical_dot([0,0], [x,0]) = max(0+x, 0+0) = max(x, 0) = ReLU(x)
        // Positive x: ReLU(3.7) = 3.7
        let w_pos = [0.0f32, 0.0];
        let x_pos = [3.7f32, 0.0];
        let mut out_pos = 0.0f32;
        tropical_dot_into(&w_pos, &x_pos, &mut out_pos, 2);
        assert!(
            (out_pos - 3.7).abs() < 1e-6,
            "ReLU(3.7): expected 3.7, got {}",
            out_pos
        );

        // Negative x: ReLU(−2.1) = 0
        let x_neg = [-2.1f32, 0.0];
        let mut out_neg = 0.0f32;
        tropical_dot_into(&w_pos, &x_neg, &mut out_neg, 2);
        assert!(
            (out_neg - 0.0).abs() < 1e-6,
            "ReLU(−2.1): expected 0.0, got {}",
            out_neg
        );
    }

    #[test]
    fn dim_zero_noop() {
        // tropical_matvec_into(&[], &[], &mut [], 0, 0) is a no-op (no panic).
        let mut out: [f32; 0] = [];
        tropical_matvec_into(&[], &[], &mut out, 0, 0);
        // If we reach here, no panic occurred.
    }

    #[test]
    fn non_contiguous_strides_smoke() {
        // Smoke test: allocating wrapper matches _into on a 3×3 case.
        let w = [
            1.0f32, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
        ];
        let x = [0.1f32, 0.2, 0.3];

        let out_alloc = tropical_matvec(&w, &x, 3, 3);

        let mut out_into = [0.0f32; 3];
        tropical_matvec_into(&w, &x, &mut out_into, 3, 3);

        for i in 0..3 {
            assert!(
                (out_alloc[i] - out_into[i]).abs() < 1e-6,
                "row {}: alloc={} vs into={}",
                i,
                out_alloc[i],
                out_into[i]
            );
        }

        // Verify hand-computed values:
        // Row 0: max(1.1, 2.2, 3.3) = 3.3
        // Row 1: max(4.1, 5.2, 6.3) = 6.3
        // Row 2: max(7.1, 8.2, 9.3) = 9.3
        assert!(
            (out_into[0] - 3.3).abs() < 1e-6,
            "row 0: expected 3.3, got {}",
            out_into[0]
        );
        assert!(
            (out_into[1] - 6.3).abs() < 1e-6,
            "row 1: expected 6.3, got {}",
            out_into[1]
        );
        assert!(
            (out_into[2] - 9.3).abs() < 1e-6,
            "row 2: expected 9.3, got {}",
            out_into[2]
        );
    }

    // ─── Phase 2 DEC wrapper tests ─────────────────────────────────────────
    //
    // These tests require `dec_operators` (always-on when `tropical_algebra`
    // is on, since tropical_algebra = ["dec_operators"]).

    #[cfg(feature = "dec_operators")]
    #[test]
    fn tropical_d_of_constant_is_zero_or_infty() {
        // grid_2d(3,3): 9 vertices, 12 edges.
        // Rank-0 vertex cochain, constant value 5.0, dim=1.
        // Under tropical max, d(constant) = 5.0 on every edge (the max of
        // boundary head contributions = 5.0 since the head vertex has 5.0).
        // This is DIFFERENT from linear d(constant) = 0.
        let cx = CellComplex::grid_2d(3, 3);
        let n_vertices = cx.n_cells(0);
        let n_edges = cx.n_cells(1);
        assert_eq!(n_vertices, 9);
        assert!(n_edges >= 1);

        let mut input = CochainField::zeros(0, n_vertices, 1);
        input.data.fill(5.0);

        let output = tropical_exterior_derivative(&cx, &input);

        assert_eq!(output.rank, 1);
        assert_eq!(output.dim, 1);
        assert_eq!(output.data.len(), n_edges);

        // Every edge output should be exactly 5.0 (the constant value at its
        // head vertex — since all vertices are 5.0, the max of head
        // contributions is 5.0).
        for (i, &v) in output.data.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-6,
                "edge {}: expected 5.0, got {}",
                i,
                v
            );
        }
    }

    #[cfg(feature = "dec_operators")]
    #[test]
    fn tropical_line_integral_is_bottleneck() {
        // grid_2d(3,3): 9 vertices, 12 edges.
        // Rank-1 edge field, dim=1, values (i * 0.5) for each edge.
        // Path v0→v3→v4 traverses edges 6 (v0→v3) and 2 (v3→v4).
        //   edge 6 value = 6*0.5 = 3.0
        //   edge 2 value = 2*0.5 = 1.0
        // Tropical (max): max(3.0, 1.0) = 3.0  (bottleneck)
        // Linear (sum): 3.0 + 1.0 = 4.0  (total work)
        let cx = CellComplex::grid_2d(3, 3);
        let n_edges = cx.n_cells(1);
        assert_eq!(n_edges, 12);

        let mut edge_field = CochainField::zeros(1, n_edges, 1);
        for (i, v) in edge_field.data.iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }

        // Path v0→v3→v4 (both edges traversed tail→head, sign +1).
        let path: Vec<u32> = vec![0, 3, 4];
        let tropical_result = tropical_line_integral(&cx, &edge_field, &path);
        assert!(
            (tropical_result - 3.0).abs() < 1e-6,
            "tropical: expected 3.0 (bottleneck), got {}",
            tropical_result
        );
        // Sanity: linear would give 4.0 — prove we're NOT doing sum.
        let linear_result = crate::dec::line_integral(&cx, &edge_field, &path);
        assert!(
            (linear_result - 4.0).abs() < 1e-6,
            "linear: expected 4.0 (sum), got {}",
            linear_result
        );
    }

    #[cfg(feature = "dec_operators")]
    #[test]
    fn tropical_exterior_derivative_includes_all_boundary_cells() {
        // grid_2d(3,3): 9 vertices, 12 edges.
        // Vertex 8 (bottom-right corner) has value 10.0, all others −100.0.
        // Vertex 8 is the HEAD of edges 5 (v7→v8) and 11 (v5→v8), both with
        // sign +1. With the sign>0 inclusion rule, these two edges should
        // output 10.0. Edges whose head is NOT vertex 8 output ≤ −100.0.
        //
        // NOTE: The plan originally specified vertex 0, but vertex 0 is ALWAYS
        // a tail (sign −1) in grid_2d, so its value is excluded by the sign>0
        // rule. Vertex 8 is the symmetric corner that IS a head of 2 edges,
        // testing the same structural property (corner → 2 incident edges).
        let cx = CellComplex::grid_2d(3, 3);
        let n_vertices = cx.n_cells(0);
        let n_edges = cx.n_cells(1);
        assert_eq!(n_vertices, 9);
        assert_eq!(n_edges, 12);

        let mut input = CochainField::zeros(0, n_vertices, 1);
        input.data.fill(-100.0);
        input.data[8] = 10.0;

        let output = tropical_exterior_derivative(&cx, &input);
        assert_eq!(output.rank, 1);
        assert_eq!(output.data.len(), n_edges);

        // Count edges with output == 10.0 (incident to vertex 8 as head).
        let mut count_10 = 0;
        for (i, &v) in output.data.iter().enumerate() {
            if (v - 10.0).abs() < 1e-6 {
                count_10 += 1;
            } else {
                // All other edges should output ≤ −100.0 (from their head vertex).
                assert!(
                    v <= -100.0 + 1e-6,
                    "edge {}: expected ≤ −100.0, got {}",
                    i,
                    v
                );
            }
        }
        // Vertex 8 is the bottom-right corner → 2 incident edges as head.
        assert!(
            count_10 >= 2,
            "expected ≥2 edges with output 10.0 (vertex 8 head-incident), got {}",
            count_10
        );
    }
}
