//! Research-flavored ML primitives — sigmoid margin loss, retrieval margin,
//! Gram matrix, entropy, coincidence score, and various norm/distance kernels.
//!
//! Each pub fn here is feature-gated (`sigmoid_margin`, `dual_gram_pca`, etc.)
//! and dispatches to NEON/AVX2/scalar impls via the same pattern as `dot.rs`.

// ── Sigmoid Margin Loss + Retrieval Diagnostic (Research 123, Plan 157) ────

/// Numerically stable softplus: log(1 + exp(x)).
///
/// For x > 20: returns x (avoids exp overflow).
/// For x < -20: returns exp(x) ≈ 0 (avoids log(1+0) precision loss).
use super::*;

// `horizontal_sum_256` lives in `super::horizontal`, so `use super::*` doesn't
// reach it. Some call sites in this file use the fully-qualified path
// (`super::horizontal::horizontal_sum_256`); this import lets the others use
// the bare name to match the other simd submodules.
#[cfg(target_arch = "x86_64")]
use super::horizontal::horizontal_sum_256;

#[cfg(feature = "sigmoid_margin")]
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SigLIP-style sigmoid margin loss: softplus(t · (score − b) · sign).
///
/// For each (query, doc) pair in the [N × n] score matrix:
///   - positive pairs (adjacency[i,j] = 1): loss = softplus(−t·(score−b)), pushes score above bias
///   - negative pairs (adjacency[i,j] = 0): loss = softplus(+t·(score−b)), pushes score below bias
///
/// Global minimizers coincide with max-margin embeddings (Prop 7, arXiv 2605.23556).
/// The paper proves this loss achieves O(k log n) dimension scaling vs InfoNCE's O(n^{1/3}).
///
/// # Arguments
/// - `scores`:      `[N × n]` dot-product score matrix (row-major)
/// - `adjacency`:   `[N × n]` binary adjacency (positive pairs = 1.0, negative = 0.0)
/// - `temperature`: learnable temperature (init 1.0)
/// - `bias`:        learnable bias (init 0.0)
/// - `n_rows`:      number of queries (N)
/// - `n_cols`:      number of documents (n)
///
/// # Returns
/// Mean loss across all pairs.
///
/// # Feature flag
/// `sigmoid_margin` — Plan 157
#[cfg(feature = "sigmoid_margin")]
#[inline]
pub fn sigmoid_margin_loss(
    scores: &[f32],
    adjacency: &[f32],
    temperature: f32,
    bias: f32,
    n_rows: usize,
    n_cols: usize,
) -> f32 {
    debug_assert_eq!(scores.len(), n_rows * n_cols);
    debug_assert_eq!(adjacency.len(), n_rows * n_cols);

    let total_elements = n_rows * n_cols;
    let mut total = 0.0f32;
    for idx in 0..total_elements {
        let score = unsafe { *scores.get_unchecked(idx) };
        let adj = unsafe { *adjacency.get_unchecked(idx) };
        // For positive pairs (adj=1): loss = softplus(-t·(score−b)), minimized when score → +∞
        // For negative pairs (adj=0): loss = softplus(+t·(score−b)), minimized when score → −∞
        // Branchless: 1.0 - 2.0 * (adj > 0.5) → -1.0 if positive, +1.0 if negative
        let sign = 1.0 - 2.0 * f32::from(adj > 0.5);
        let x = temperature * (score - bias) * sign;
        total += softplus(x);
    }
    total / total_elements as f32
}

/// Compute retrieval margin: 0.5 × (min_pos_score − max_neg_score).
///
/// For each query embedding u_i with positive set P_i given by `neighborhoods`:
///   pos_min = min_{j ∈ P_i} dot(u_i, v_j)
///   neg_max = max_{j ∉ P_i} dot(u_i, v_j)
///   margin_i = 0.5 * (pos_min − neg_max)
///
/// Returns (global_min_pos, global_max_neg, global_margin) across all queries.
///
/// # Arguments
/// - `queries`:       `[N × dim]` row-major query embeddings
/// - `documents`:     `[n × dim]` row-major document embeddings
/// - `neighborhoods`: `[N × k]` positive pair indices (flat, row-major)
/// - `dim`:           embedding dimension
/// - `n_queries`:     number of queries (N)
/// - `n_docs`:        number of documents (n)
/// - `k`:            neighborhood size
///
/// # Feature flag
/// `sigmoid_margin` — Plan 157
#[cfg(feature = "sigmoid_margin")]
#[inline]
pub fn compute_retrieval_margin(
    queries: &[f32],
    documents: &[f32],
    neighborhoods: &[usize],
    dim: usize,
    n_queries: usize,
    n_docs: usize,
    k: usize,
) -> (f32, f32, f32) {
    debug_assert!(queries.len() >= n_queries * dim);
    debug_assert!(documents.len() >= n_docs * dim);
    debug_assert!(neighborhoods.len() >= n_queries * k);

    let mut global_pos_min = f32::INFINITY;
    let mut global_neg_max = f32::NEG_INFINITY;

    // Generation-counter bitmap: avoids per-query `fill(0)` memset.
    // Instead of zeroing `n_docs` bytes per query (n_queries passes), we tag each
    // slot with the current query index and compare — O(1) per query, no memset.
    let mut pos_gen: Vec<u32> = vec![0; n_docs];

    for i in 0..n_queries {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let cur_gen = (i + 1) as u32;

        // Build positive set and compute min positive score in one pass.
        let pos_start = i * k;
        let mut pos_min = f32::INFINITY;
        for &idx in &neighborhoods[pos_start..pos_start + k] {
            if idx < n_docs {
                pos_gen[idx] = cur_gen;
                let d_row = &documents[idx * dim..(idx + 1) * dim];
                let dot = simd_dot_f32(q_row, d_row, dim);
                pos_min = pos_min.min(dot);
            }
        }

        // max negative score (docs not in pos_set). Generation check replaces memset.
        let mut neg_max = f32::NEG_INFINITY;
        for j in 0..n_docs {
            if pos_gen[j] == cur_gen {
                continue;
            }
            let d_row = &documents[j * dim..(j + 1) * dim];
            let dot = simd_dot_f32(q_row, d_row, dim);
            neg_max = neg_max.max(dot);
        }

        global_pos_min = global_pos_min.min(pos_min);
        global_neg_max = global_neg_max.max(neg_max);
    }

    let margin = 0.5 * (global_pos_min - global_neg_max);
    (global_pos_min, global_neg_max, margin)
}

/// Theoretical O(k log n) dimension sufficiency bound from arXiv 2605.23556.
///
/// Returns the minimum embedding dimension theoretically sufficient
/// for near-optimal retrieval margin, given query sparsity k and corpus size n.
///
/// Theorem 1.4: d = O(k · log n) is sufficient.
/// Theorem 1.5: d = O(k · log(n/k)) is also necessary → tight bound.
///
/// Uses a conservative constant factor of 1.5 to provide a practical upper bound.
/// For k ≤ 0 or n ≤ 1, returns 1 (trivial case).
///
/// # Feature flag
/// `sigmoid_margin` — Plan 157
#[cfg(feature = "sigmoid_margin")]
pub fn dim_sufficiency_bound(k: usize, n: usize) -> usize {
    if k == 0 || n <= 1 {
        return 1;
    }
    // d = ceil(1.5 * k * ln(n))  — conservative O(k log n) bound
    let bound = 1.5 * (k as f64) * (n as f64).ln();
    bound.ceil() as usize
}

// ── SIMD Sum of Squares (Issue 075) ──────────────────────────

/// SIMD-accelerated sum of squares: `Σ x[i]²`.
/// More efficient than `simd_dot_f32(x, x, len)` — loads data once instead of twice.
#[inline(always)]
pub fn simd_sum_sq(x: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sum_sq(x, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sum_sq(x, len) }
        } else {
            scalar_sum_sq(x, len)
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_simd128_sum_sq(x, len) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_sum_sq(x, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_sum_sq(x: &[f32], len: usize) -> f32 {
    // 4 independent accumulators — FMA latency bound on a single accumulator.
    // mul_add preserves single-rounding parity with the SIMD path.
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            let v0 = *x.get_unchecked(i);
            let v1 = *x.get_unchecked(i + 1);
            let v2 = *x.get_unchecked(i + 2);
            let v3 = *x.get_unchecked(i + 3);
            acc[0] = v0.mul_add(v0, acc[0]);
            acc[1] = v1.mul_add(v1, acc[1]);
            acc[2] = v2.mul_add(v2, acc[2]);
            acc[3] = v3.mul_add(v3, acc[3]);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < len {
        unsafe {
            let v = *x.get_unchecked(i);
            sum = v.mul_add(v, sum);
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sum_sq(x: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks4 = len / 16;

        for _ in 0..chunks4 {
            let v0 = vld1q_f32(x.as_ptr().add(i));
            acc0 = vfmaq_f32(acc0, v0, v0);
            let v1 = vld1q_f32(x.as_ptr().add(i + 4));
            acc1 = vfmaq_f32(acc1, v1, v1);
            let v2 = vld1q_f32(x.as_ptr().add(i + 8));
            acc2 = vfmaq_f32(acc2, v2, v2);
            let v3 = vld1q_f32(x.as_ptr().add(i + 12));
            acc3 = vfmaq_f32(acc3, v3, v3);
            i += 16;
        }

        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        let mut acc_rem = vdupq_n_f32(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            let v = vld1q_f32(x.as_ptr().add(i));
            acc_rem = vfmaq_f32(acc_rem, v, v);
            i += 4;
        }
        sum += vaddvq_f32(acc_rem);

        while i < len {
            let v = *x.get_unchecked(i);
            sum += v * v;
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_sum_sq(x: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps};

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        let chunks2 = len / 16;

        for _ in 0..chunks2 {
            let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
            i += 16;
        }

        let mut sum = horizontal_sum_256(_mm256_add_ps(acc0, acc1));

        // Accumulate remaining 8-element chunks, reduce once
        let mut acc_rem = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            acc_rem = _mm256_fmadd_ps(v, v, acc_rem);
            i += 8;
        }
        sum += horizontal_sum_256(acc_rem);

        while i < len {
            let v = *x.get_unchecked(i);
            sum += v * v;
            i += 1;
        }

        sum
    }
}

// ── WASM SIMD128 sum-of-squares (Plan 008 Step 7a — port from riir-engine) ──

/// WASM SIMD128 sum of squares — 4-wide f32 multiply-accumulate of `x*x`.
///
/// Uses 4 independent accumulators (4 lanes each = 16 elements per outer
/// iter) to hide the mul→add latency chain, mirroring the NEON kernel's
/// unroll factor. Ported from the proven riir-engine `simd::wasm32::simd_sum_sq`
/// (Plan 286 T6/T7) so that `katgpt_core::simd::simd_sum_sq` is no longer
/// scalar-bound on WASM targets.
///
/// Compile-time gated by `target_feature = "simd128"`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_sum_sq(x: &[f32], len: usize) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load};

    unsafe {
        let n = x.len().min(len);
        let simd_len = n / 4 * 4;
        let mut acc = f32x4_splat(0.0);

        let mut i = 0;
        while i < simd_len {
            let v = v128_load(x.as_ptr().add(i) as *const _);
            acc = f32x4_add(acc, f32x4_mul(v, v));
            i += 4;
        }

        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        while i < n {
            let v = *x.get_unchecked(i);
            sum += v * v;
            i += 1;
        }
        sum
    }
}

// ── SIMD Sum-x² + Sum-x⁴ fused (Plan 306 T7.4 revisit) ─────

/// SIMD-accelerated fused sum of squares and sum of quartics:
/// returns `(Σ x[i]², Σ x[i]⁴)` in a single sweep.
///
/// Used by `depth_invariance::classify_chain` to compute magnitude
/// (`sqrt(Σx²)`) and participation-ratio flatness `((Σx²)² / (d·Σx⁴))`
/// in one pass over each timestep's `h_t` slice, instead of the old
/// scalar `mul_add` loop. Plan 306 Phase 6 T7.4 SIMD follow-up.
///
/// Empty slice returns `(0.0, 0.0)`.
#[inline(always)]
pub fn simd_sum_sq_quartic(x: &[f32]) -> (f32, f32) {
    if x.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sum_sq_quartic(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sum_sq_quartic(x) }
        } else {
            scalar_sum_sq_quartic(x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_sum_sq_quartic(x)
    }
}

/// Scalar reference for [`simd_sum_sq_quartic`]. `pub(super)` so the
/// `simd::tests` module can use it as the truth via `use super::*`.
///
/// 4 independent accumulators per quantity — FMA latency bound on a
/// single accumulator. `mul_add` preserves single-rounding parity with
/// the SIMD path.
#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_sum_sq_quartic(x: &[f32]) -> (f32, f32) {
    let len = x.len();
    let mut sq = [0.0f32; 4];
    let mut qu = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            let v0 = *x.get_unchecked(i);
            let v1 = *x.get_unchecked(i + 1);
            let v2 = *x.get_unchecked(i + 2);
            let v3 = *x.get_unchecked(i + 3);
            let x2_0 = v0 * v0;
            let x2_1 = v1 * v1;
            let x2_2 = v2 * v2;
            let x2_3 = v3 * v3;
            // x² accumulation.
            sq[0] += x2_0;
            sq[1] += x2_1;
            sq[2] += x2_2;
            sq[3] += x2_3;
            // x⁴ = x² · x² — fused mul-add for single rounding.
            qu[0] = x2_0.mul_add(x2_0, qu[0]);
            qu[1] = x2_1.mul_add(x2_1, qu[1]);
            qu[2] = x2_2.mul_add(x2_2, qu[2]);
            qu[3] = x2_3.mul_add(x2_3, qu[3]);
        }
        i += 4;
    }
    let mut sum_sq = sq.iter().sum::<f32>();
    let mut sum_quartic = qu.iter().sum::<f32>();
    while i < len {
        unsafe {
            let v = *x.get_unchecked(i);
            let x2 = v * v;
            sum_sq += x2;
            sum_quartic = x2.mul_add(x2, sum_quartic);
        }
        i += 1;
    }
    (sum_sq, sum_quartic)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sum_sq_quartic(x: &[f32]) -> (f32, f32) {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32,
    };

    unsafe {
        let mut sq0 = vdupq_n_f32(0.0);
        let mut sq1 = vdupq_n_f32(0.0);
        let mut sq2 = vdupq_n_f32(0.0);
        let mut sq3 = vdupq_n_f32(0.0);
        let mut qu0 = vdupq_n_f32(0.0);
        let mut qu1 = vdupq_n_f32(0.0);
        let mut qu2 = vdupq_n_f32(0.0);
        let mut qu3 = vdupq_n_f32(0.0);

        let len = x.len();
        let mut i = 0;
        let chunks4 = len / 16;

        for _ in 0..chunks4 {
            let v0 = vld1q_f32(x.as_ptr().add(i));
            let x2_0 = vmulq_f32(v0, v0);
            sq0 = vaddq_f32(sq0, x2_0);
            qu0 = vfmaq_f32(qu0, x2_0, x2_0);

            let v1 = vld1q_f32(x.as_ptr().add(i + 4));
            let x2_1 = vmulq_f32(v1, v1);
            sq1 = vaddq_f32(sq1, x2_1);
            qu1 = vfmaq_f32(qu1, x2_1, x2_1);

            let v2 = vld1q_f32(x.as_ptr().add(i + 8));
            let x2_2 = vmulq_f32(v2, v2);
            sq2 = vaddq_f32(sq2, x2_2);
            qu2 = vfmaq_f32(qu2, x2_2, x2_2);

            let v3 = vld1q_f32(x.as_ptr().add(i + 12));
            let x2_3 = vmulq_f32(v3, v3);
            sq3 = vaddq_f32(sq3, x2_3);
            qu3 = vfmaq_f32(qu3, x2_3, x2_3);

            i += 16;
        }

        let mut sum_sq = vaddvq_f32(vaddq_f32(vaddq_f32(sq0, sq1), vaddq_f32(sq2, sq3)));
        let mut sum_quartic = vaddvq_f32(vaddq_f32(vaddq_f32(qu0, qu1), vaddq_f32(qu2, qu3)));

        // Remaining 4-element chunks.
        let mut sq_rem = vdupq_n_f32(0.0);
        let mut qu_rem = vdupq_n_f32(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            let v = vld1q_f32(x.as_ptr().add(i));
            let x2 = vmulq_f32(v, v);
            sq_rem = vaddq_f32(sq_rem, x2);
            qu_rem = vfmaq_f32(qu_rem, x2, x2);
            i += 4;
        }
        sum_sq += vaddvq_f32(sq_rem);
        sum_quartic += vaddvq_f32(qu_rem);

        // Scalar tail.
        while i < len {
            let v = *x.get_unchecked(i);
            let x2 = v * v;
            sum_sq += x2;
            sum_quartic += x2 * x2;
            i += 1;
        }

        (sum_sq, sum_quartic)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_sum_sq_quartic(x: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps,
    };

    unsafe {
        let mut sq0 = _mm256_setzero_ps();
        let mut sq1 = _mm256_setzero_ps();
        let mut qu0 = _mm256_setzero_ps();
        let mut qu1 = _mm256_setzero_ps();

        let len = x.len();
        let mut i = 0;
        let chunks2 = len / 16;

        for _ in 0..chunks2 {
            let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
            let x2_0 = _mm256_mul_ps(v0, v0);
            sq0 = _mm256_add_ps(sq0, x2_0);
            qu0 = _mm256_fmadd_ps(x2_0, x2_0, qu0);

            let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            let x2_1 = _mm256_mul_ps(v1, v1);
            sq1 = _mm256_add_ps(sq1, x2_1);
            qu1 = _mm256_fmadd_ps(x2_1, x2_1, qu1);

            i += 16;
        }

        let mut sum_sq = super::horizontal::horizontal_sum_256(_mm256_add_ps(sq0, sq1));
        let mut sum_quartic = super::horizontal::horizontal_sum_256(_mm256_add_ps(qu0, qu1));

        // Remaining 8-element chunks.
        let mut sq_rem = _mm256_setzero_ps();
        let mut qu_rem = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let x2 = _mm256_mul_ps(v, v);
            sq_rem = _mm256_add_ps(sq_rem, x2);
            qu_rem = _mm256_fmadd_ps(x2, x2, qu_rem);
            i += 8;
        }
        sum_sq += super::horizontal::horizontal_sum_256(sq_rem);
        sum_quartic += super::horizontal::horizontal_sum_256(qu_rem);

        // Scalar tail.
        while i < len {
            let v = *x.get_unchecked(i);
            let x2 = v * v;
            sum_sq += x2;
            sum_quartic += x2 * x2;
            i += 1;
        }

        (sum_sq, sum_quartic)
    }
}

// ── SIMD Sum-|x| (Issue 120) ───────────────────────────────

/// SIMD-accelerated sum of absolute values: `Σ |x[i]|`.
#[inline(always)]
pub fn simd_sum_abs_f32(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sum_abs_f32(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sum_abs_f32(x) }
        } else {
            scalar_sum_abs_f32(x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_sum_abs_f32(x)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_sum_abs_f32(x: &[f32]) -> f32 {
    // 4 independent accumulators — addition latency bound on a single accumulator.
    let mut acc = [0.0f32; 4];
    let chunks = x.len() / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] += (*x.get_unchecked(i)).abs();
            acc[1] += (*x.get_unchecked(i + 1)).abs();
            acc[2] += (*x.get_unchecked(i + 2)).abs();
            acc[3] += (*x.get_unchecked(i + 3)).abs();
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < x.len() {
        unsafe {
            sum += (*x.get_unchecked(i)).abs();
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sum_abs_f32(x: &[f32]) -> f32 {
    use core::arch::aarch64::{vabsq_f32, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32};
    unsafe {
        // 4 independent accumulators (16 elements per outer iter) — matches
        // `neon_sum_f32` / `neon_dot_f32` and keeps the FADD pipeline full.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let len = x.len();
        let chunks4 = len / 16;
        for _ in 0..chunks4 {
            acc0 = vaddq_f32(acc0, vabsq_f32(vld1q_f32(x.as_ptr().add(i))));
            acc1 = vaddq_f32(acc1, vabsq_f32(vld1q_f32(x.as_ptr().add(i + 4))));
            acc2 = vaddq_f32(acc2, vabsq_f32(vld1q_f32(x.as_ptr().add(i + 8))));
            acc3 = vaddq_f32(acc3, vabsq_f32(vld1q_f32(x.as_ptr().add(i + 12))));
            i += 16;
        }
        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        // Remainder (4-element chunks).
        let mut acc_rem = vdupq_n_f32(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            acc_rem = vaddq_f32(acc_rem, vabsq_f32(vld1q_f32(x.as_ptr().add(i))));
            i += 4;
        }
        sum += vaddvq_f32(acc_rem);

        while i < len {
            sum += (*x.get_unchecked(i)).abs();
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_sum_abs_f32(x: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_and_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    };
    unsafe {
        // Mask to clear the sign bit: AND with this yields |x|
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        // 4 independent accumulators (32 elements per outer iter) — matches
        // `avx2_sum_f32` / `avx2_dot_f32` and keeps the FADD pipeline full.
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        let len = x.len();
        let chunks4 = len / 32;
        for _ in 0..chunks4 {
            acc0 = _mm256_add_ps(
                acc0,
                _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(i)), abs_mask),
            );
            acc1 = _mm256_add_ps(
                acc1,
                _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), abs_mask),
            );
            acc2 = _mm256_add_ps(
                acc2,
                _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(i + 16)), abs_mask),
            );
            acc3 = _mm256_add_ps(
                acc3,
                _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(i + 24)), abs_mask),
            );
            i += 32;
        }
        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = horizontal_sum_256(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ));

        // Remainder (8-element chunks).
        let mut acc_rem = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            acc_rem = _mm256_add_ps(acc_rem, _mm256_and_ps(v, abs_mask));
            i += 8;
        }
        sum += horizontal_sum_256(acc_rem);
        while i < len {
            sum += (*x.get_unchecked(i)).abs();
            i += 1;
        }
        sum
    }
}

// ── SIMD Distance² (Issue 076) ───────────────────────────────

/// SIMD-accelerated squared distance: `Σ (a[i] - b[i])²`.
/// Computes the elementwise difference, squares, and sums in one pass.
#[inline]
pub fn simd_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dist_sq(a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_dist_sq(a, b, len) }
        } else {
            scalar_dist_sq(a, b, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_dist_sq(a, b, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    // 4 independent accumulators — FMA latency bound on a single accumulator.
    // mul_add preserves single-rounding parity with the SIMD path.
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            let d0 = *a.get_unchecked(i) - *b.get_unchecked(i);
            let d1 = *a.get_unchecked(i + 1) - *b.get_unchecked(i + 1);
            let d2 = *a.get_unchecked(i + 2) - *b.get_unchecked(i + 2);
            let d3 = *a.get_unchecked(i + 3) - *b.get_unchecked(i + 3);
            acc[0] = d0.mul_add(d0, acc[0]);
            acc[1] = d1.mul_add(d1, acc[1]);
            acc[2] = d2.mul_add(d2, acc[2]);
            acc[3] = d3.mul_add(d3, acc[3]);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < len {
        unsafe {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum = diff.mul_add(diff, sum);
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vsubq_f32,
    };

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks2 = len / 8;

        for _ in 0..chunks2 {
            let d0 = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            let d1 = vsubq_f32(
                vld1q_f32(a.as_ptr().add(i + 4)),
                vld1q_f32(b.as_ptr().add(i + 4)),
            );
            acc1 = vfmaq_f32(acc1, d1, d1);
            i += 8;
        }

        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));

        if i + 4 <= len {
            let d = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            sum += vaddvq_f32(vmulq_f32(d, d));
            i += 4;
        }

        while i < len {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += diff * diff;
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_sub_ps,
    };

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        let chunks2 = len / 16;

        for _ in 0..chunks2 {
            let d0 = _mm256_sub_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            );
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            i += 16;
        }

        let mut sum = horizontal_sum_256(_mm256_add_ps(acc0, acc1));

        let mut acc_rem = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let d = _mm256_sub_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            );
            acc_rem = _mm256_fmadd_ps(d, d, acc_rem);
            i += 8;
        }
        sum += horizontal_sum_256(acc_rem);

        while i < len {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += diff * diff;
            i += 1;
        }

        sum
    }
}

// ── SIMD L-∞ Distance (Issue 003 / riir-neuron-db) ───────────

/// SIMD-accelerated L-infinity distance: `max_i |a[i] - b[i]|`.
///
/// Computes the elementwise absolute difference and horizontal max in one
/// pass. Used by `select_diverse_subset`'s `argmax_pair` seed (O(n²) pairwise
/// scan) and the greedy-fill inner loop. The dispatch shape mirrors
/// [`simd_dist_sq`] / [`simd_sum_abs_f32`]: 4-accumulator scalar fallback,
/// NEON `vmaxq_f32` + `vmaxvq_f32` reduce, AVX2 `_mm256_max_ps` +
/// `horizontal_max_256` reduce.
///
/// # Numerical note
///
/// `|x|` is computed as a sign-bit AND (`x & 0x7fff_ffff`), which is
/// bit-identical to `f32::abs()` for all finite inputs (and matches the
/// SIMD `fabs` intrinsic). The horizontal max is associative/commutative for
/// finite `f32`, so the reduction order does not affect the result — this
/// makes the SIMD output bit-identical to the scalar reference on all finite
/// inputs (verified by `simd::tests::*l_inf_distance*`). NaN inputs follow
/// IEEE 754 `max` propagation semantics (`_mm_max_ps` / `vmaxq_f32` are
/// total-ordered max-of-magnitudes on x86/ARM, matching the scalar `if a > b`
/// path for the non-NaN case).
#[inline]
pub fn simd_l_inf_distance_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_l_inf_distance_f32(a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_l_inf_distance_f32(a, b, len) }
        } else {
            scalar_l_inf_distance_f32(a, b, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_l_inf_distance_f32(a, b, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_l_inf_distance_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    // 4 independent max-accumulators to break the latency chain of a single
    // `max` (cmp+select on most FPUs). Mirrors the 4-acc shape of
    // `scalar_dist_sq` / `scalar_sum_abs_f32`. Output is bit-identical to the
    // reference `l_inf_distance` in katgpt-core's diversity::temp.
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            let d0 = (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
            let d1 = (*a.get_unchecked(i + 1) - *b.get_unchecked(i + 1)).abs();
            let d2 = (*a.get_unchecked(i + 2) - *b.get_unchecked(i + 2)).abs();
            let d3 = (*a.get_unchecked(i + 3) - *b.get_unchecked(i + 3)).abs();
            if d0 > acc[0] {
                acc[0] = d0;
            }
            if d1 > acc[1] {
                acc[1] = d1;
            }
            if d2 > acc[2] {
                acc[2] = d2;
            }
            if d3 > acc[3] {
                acc[3] = d3;
            }
        }
        i += 4;
    }
    let mut max_abs = acc.iter().copied().fold(0.0f32, f32::max);
    while i < len {
        unsafe {
            let d = (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        i += 1;
    }
    max_abs
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_l_inf_distance_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vabdq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32};

    unsafe {
        // Single 4-wide accumulator. NEON vmaxq is single-cycle / low-latency,
        // so the 4-accumulator chain (used for latency-bound FMA in dot/sum)
        // is unnecessary here. We still unroll by 4 vectors (16 elements) to
        // amortize loop overhead.
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks16 = len / 16;
        for _ in 0..chunks16 {
            // vabdq_f32: per-lane |a - b| in one intrinsic.
            let d0 = vabdq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            let d1 = vabdq_f32(
                vld1q_f32(a.as_ptr().add(i + 4)),
                vld1q_f32(b.as_ptr().add(i + 4)),
            );
            let d2 = vabdq_f32(
                vld1q_f32(a.as_ptr().add(i + 8)),
                vld1q_f32(b.as_ptr().add(i + 8)),
            );
            let d3 = vabdq_f32(
                vld1q_f32(a.as_ptr().add(i + 12)),
                vld1q_f32(b.as_ptr().add(i + 12)),
            );
            acc = vmaxq_f32(acc, d0);
            acc = vmaxq_f32(acc, d1);
            acc = vmaxq_f32(acc, d2);
            acc = vmaxq_f32(acc, d3);
            i += 16;
        }

        // 4-wide remainder (128-bit ops keep one intrinsic set).
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            let d = vabdq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            acc = vmaxq_f32(acc, d);
            i += 4;
        }

        // Horizontal max across the 4 lanes.
        let mut max_abs = vmaxvq_f32(acc);

        // Scalar tail (1–3 trailing elements).
        while i < len {
            let diff = (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            i += 1;
        }

        max_abs
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_l_inf_distance_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{
        _mm256_and_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_sub_ps,
    };

    unsafe {
        // Sign-bit mask: AND with this clears the sign bit → |x|.
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        // Single 8-wide accumulator. AVX2 `_mm256_max_ps` is single-cycle /
        // 4-cycle latency (Skylake/Kaby Lake/Zen2+); a single acc keeps the
        // throughput-bound max pipeline saturated without the 4-acc
        // dependency-chain-breaking used for latency-bound FMA in dot/sum.
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        let chunks32 = len / 32;
        for _ in 0..chunks32 {
            // Unrolled by 4 vectors (32 elements) to amortize loop overhead.
            let d0 = _mm256_and_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                ),
                abs_mask,
            );
            let d1 = _mm256_and_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                ),
                abs_mask,
            );
            let d2 = _mm256_and_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                ),
                abs_mask,
            );
            let d3 = _mm256_and_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                ),
                abs_mask,
            );
            acc = _mm256_max_ps(acc, d0);
            acc = _mm256_max_ps(acc, d1);
            acc = _mm256_max_ps(acc, d2);
            acc = _mm256_max_ps(acc, d3);
            i += 32;
        }

        // 8-wide remainder.
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let d = _mm256_and_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                ),
                abs_mask,
            );
            acc = _mm256_max_ps(acc, d);
            i += 8;
        }

        // Horizontal max across the 8 lanes (shared helper in horizontal.rs).
        let mut max_abs = super::horizontal::horizontal_max_256(acc);

        // Scalar tail (1–7 trailing elements).
        while i < len {
            let diff = (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            i += 1;
        }

        max_abs
    }
}

// ── SIMD Fused Subtract-Accumulate (Issue 071) ──────────────

/// SIMD-accelerated fused subtract-accumulate: `dst[i] += a[i] - b[i]`.
/// Single-pass operation for MLS and delta routing accumulation.
#[inline(always)]
pub fn simd_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_sub_acc(dst, a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_fused_sub_acc(dst, a, b, len) }
        } else {
            scalar_fused_sub_acc(dst, a, b, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_fused_sub_acc(dst, a, b, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    for i in 0..len {
        unsafe {
            *dst.get_unchecked_mut(i) += *a.get_unchecked(i) - *b.get_unchecked(i);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32, vsubq_f32};

    unsafe {
        let mut i = 0;
        let chunks = len / 4;

        for _ in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vd = vld1q_f32(dst.as_ptr().add(i));
            let result = vaddq_f32(vd, vsubq_f32(va, vb));
            vst1q_f32(dst.as_mut_ptr().add(i), result);
            i += 4;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};

    unsafe {
        let mut i = 0;
        let chunks = len / 8;

        for _ in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vd = _mm256_loadu_ps(dst.as_ptr().add(i));
            let result = _mm256_add_ps(vd, _mm256_sub_ps(va, vb));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), result);
            i += 8;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }
}

/// Fused scale-accumulate: `dst[i] += scale * src[i]` for `len` elements.
///
/// Used in attention value accumulation where a scalar weight broadcasts across a value row.
/// NEON: fused via `vfmaq_f32`. AVX2: fused via `_mm256_fmadd_ps`.
#[inline(always)]
pub fn simd_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_scale_acc(dst, src, scale, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_fused_scale_acc(dst, src, scale, len) }
        } else {
            scalar_fused_scale_acc(dst, src, scale, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_fused_scale_acc(dst, src, scale, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    for i in 0..len {
        unsafe {
            // FMA: dst[i] = scale * src[i] + dst[i] (single rounding).
            let s = *src.get_unchecked(i);
            *dst.get_unchecked_mut(i) = scale.mul_add(s, *dst.get_unchecked(i));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

    unsafe {
        let scale_vec = vdupq_n_f32(scale);
        let mut i = 0;
        let chunks = len / 4;

        for _ in 0..chunks {
            let s = vld1q_f32(src.as_ptr().add(i));
            let d = vld1q_f32(dst.as_ptr().add(i));
            let acc = vfmaq_f32(d, s, scale_vec);
            vst1q_f32(dst.as_mut_ptr().add(i), acc);
            i += 4;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += scale * *src.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    unsafe {
        let scale_vec = _mm256_set1_ps(scale);
        let mut i = 0;
        let chunks = len / 8;

        for _ in 0..chunks {
            let s = _mm256_loadu_ps(src.as_ptr().add(i));
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let acc = _mm256_fmadd_ps(scale_vec, s, d);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), acc);
            i += 8;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += scale * *src.get_unchecked(i);
            i += 1;
        }
    }
}

/// SIMD-accelerated Gram matrix computation: G = X·Xᵀ where X is (seq_len × d_h).
///
/// For each pair (i, j), G[i*seq_len + j] = dot(X_row_i, X_row_j).
/// Only computes upper triangle, then mirrors to lower triangle.
/// Uses existing `simd_dot_f32` for each dot product.
///
/// `x` is row-major (seq_len × d_h), `gram_out` is row-major (seq_len × seq_len).
#[inline]
pub fn simd_gram_f32(x: &[f32], seq_len: usize, d_h: usize, gram_out: &mut [f32]) {
    debug_assert_eq!(x.len(), seq_len * d_h, "x length mismatch");
    debug_assert_eq!(
        gram_out.len(),
        seq_len * seq_len,
        "gram_out length mismatch"
    );

    for i in 0..seq_len {
        let row_i = &x[i * d_h..(i + 1) * d_h];
        let i_row_off = i * seq_len;

        // Diagonal: sum of squares (one load instead of two)
        let diag = simd_sum_sq(row_i, d_h);
        gram_out[i_row_off + i] = diag;

        // Upper triangle: j > i
        for j in (i + 1)..seq_len {
            let row_j = &x[j * d_h..(j + 1) * d_h];
            let val = simd_dot_f32(row_i, row_j, d_h);
            gram_out[i_row_off + j] = val;
            gram_out[j * seq_len + i] = val; // mirror to lower triangle
        }
    }
}

// ── Entropy & Coincidence for DendriticGate (Plan 260) ──────────

/// SIMD-accelerated entropy computation from log-probabilities.
///
/// Computes `entropy = -Σ p·log(p)` where `p = exp(logprobs[i])` normalized.
///
/// **Single-pass** over `logprobs`: accumulates both `Z = Σexp(logp)` and
/// `Σ exp(logp)·logp` in the same loop, then applies the identity
/// `H = log(Z) - Σ(e·logp)/Z`. This halves the number of `exp()` calls vs the
/// old two-pass formulation (saves ~128K exp() calls on a 128K vocab) and removes
/// the per-element branch that previously blocked auto-vectorization.
///
/// NaN-safe: when `logp = -∞`, `exp(logp) = 0` and `0 · (-∞) = NaN`; the
/// `if e > 0.0` guard turns this into a branch-free `select`, contributing 0.
/// This matches the old `if p_norm > 0.0` semantics exactly.
///
/// When `logprobs` is empty, returns 0.0.
#[inline]
pub fn entropy_f32(logprobs: &[f32]) -> f32 {
    if logprobs.is_empty() {
        return 0.0;
    }

    let len = logprobs.len();
    let chunks = len / 4;
    let remainder = len % 4;

    // Single pass: accumulate Z = Σexp(logp) and S = Σ exp(logp)·logp together.
    // The `if e > 0.0 { e * logp } else { 0.0 }` form compiles to a branch-free
    // select (cmov), enabling LLVM to vectorize the whole loop body.
    let mut sum_exp = 0.0f32;
    let mut sum_exp_logp = 0.0f32;
    for i in 0..chunks {
        let base = i * 4;
        let l0 = logprobs[base];
        let l1 = logprobs[base + 1];
        let l2 = logprobs[base + 2];
        let l3 = logprobs[base + 3];
        let e0 = l0.exp();
        let e1 = l1.exp();
        let e2 = l2.exp();
        let e3 = l3.exp();
        sum_exp += e0 + e1 + e2 + e3;
        // Branch-free guard: 0·(-∞) = NaN in IEEE 754; select 0 when e == 0
        // (which happens iff logp = -∞). Equivalent to old `if p_norm > 0.0`.
        sum_exp_logp += if e0 > 0.0 { e0 * l0 } else { 0.0 };
        sum_exp_logp += if e1 > 0.0 { e1 * l1 } else { 0.0 };
        sum_exp_logp += if e2 > 0.0 { e2 * l2 } else { 0.0 };
        sum_exp_logp += if e3 > 0.0 { e3 * l3 } else { 0.0 };
    }
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        let l = logprobs[idx];
        let e = l.exp();
        sum_exp += e;
        sum_exp_logp += if e > 0.0 { e * l } else { 0.0 };
    }

    if sum_exp <= f32::EPSILON {
        return 0.0;
    }

    // H = log(Z) - S / Z   (identity: H = log(Z) - Σ p_norm · logp)
    let entropy = sum_exp.ln() - sum_exp_logp / sum_exp;
    entropy.max(0.0)
}

/// Compute coincidence score: agreement between top-K candidates and parent path.
///
/// Counts how many of the top-K candidate tokens appear in the parent path
/// within the coincidence window (last `window` tokens of parent_path).
/// Returns `agreement_count / window_size` ∈ [0, 1].
///
/// When either slice is empty, returns 0.0.
#[inline]
pub fn coincidence_score(top_k: &[usize], parent_path: &[usize], window: usize) -> f32 {
    if top_k.is_empty() || parent_path.is_empty() || window == 0 {
        return 0.0;
    }

    let window_start = parent_path.len().saturating_sub(window);
    let window_slice = &parent_path[window_start..];
    let effective_window = window_slice.len() as f32;

    if effective_window <= 0.0 {
        return 0.0;
    }

    let mut agreement = 0usize;
    // Small window → linear scan is fine (window typically ≤ 8)
    for &token in top_k {
        if window_slice.contains(&token) {
            agreement += 1;
        }
    }

    agreement as f32 / effective_window
}
