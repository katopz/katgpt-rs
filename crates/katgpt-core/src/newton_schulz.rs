//! Newton-Schulz Orthogonalization + Muon Momentum (Plan 152, Research 114).
//!
//! Converts any matrix to its nearest orthogonal factor via 5 fixed-point
//! iterations. Generic building block for Muon-family optimizers.
//!
//! The Newton-Schulz cubic iteration:
//! ```text
//!   X = G / ||G||_F
//!   for 5 iters: A = X @ X^T; X = a*X + (b*A + c*A@A) @ X
//! ```
//! Constants a=3.4445, b=-4.7750, c=2.0315 from the AMUSE paper —
//! converges for singular values in [0, 1].

#![allow(clippy::needless_range_loop)]

/// Newton-Schulz coefficients (converges for σ ∈ [0, 1]).
const A: f32 = 3.4445;
const B: f32 = -4.7750;
const C: f32 = 2.0315;

// ── Matrix helpers ──────────────────────────────────────────────

/// Transpose `rows × cols` matrix stored row-major from `src` into `dst`.
/// Processes 4 rows at a time for better auto-vectorization.
#[inline]
fn transpose(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    let mut r = 0;
    while r + 4 <= rows {
        for c in 0..cols {
            let sr0 = r * cols + c;
            let sr1 = (r + 1) * cols + c;
            let sr2 = (r + 2) * cols + c;
            let sr3 = (r + 3) * cols + c;
            dst[c * rows + r] = src[sr0];
            dst[c * rows + r + 1] = src[sr1];
            dst[c * rows + r + 2] = src[sr2];
            dst[c * rows + r + 3] = src[sr3];
        }
        r += 4;
    }
    // Handle remaining rows
    while r < rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
        r += 1;
    }
}

/// Compute `A = X * X^T` for an `m × n` matrix X, producing an `m × m` result.
/// Uses SIMD dot products and exploits symmetry (upper triangle + mirror).
#[inline]
fn matmul_xtx(x: &[f32], m: usize, n: usize, a: &mut [f32]) {
    for i in 0..m {
        let row_i = &x[i * n..(i + 1) * n];
        // Diagonal
        a[i * m + i] = crate::simd::simd_dot_f32(row_i, row_i, n);
        // Upper triangle + mirror
        for j in (i + 1)..m {
            let row_j = &x[j * n..(j + 1) * n];
            let dot = crate::simd::simd_dot_f32(row_i, row_j, n);
            a[i * m + j] = dot;
            a[j * m + i] = dot;
        }
    }
}

/// Compute `R = A * X` where A is `m × m` and X is `m × n`, result is `m × n`.
/// Transposes X for contiguous inner-loop access, then uses SIMD dot products.
/// Caller provides `xt_buf` (`m * n` elements) to avoid per-call allocation.
#[inline]
fn matmul_ax(a: &[f32], x: &[f32], m: usize, n: usize, r: &mut [f32], xt_buf: &mut [f32]) {
    // Transpose X: columns become contiguous rows in xt (n × m).
    transpose(x, m, n, xt_buf);

    // r[i,j] = dot(a_row_i, xt_col_j) = dot(&a[i*m..], &xt[j*m..], m)
    for i in 0..m {
        let a_row = &a[i * m..(i + 1) * m];
        let r_row = &mut r[i * n..(i + 1) * n];
        for j in 0..n {
            r_row[j] = crate::simd::simd_dot_f32(a_row, &xt_buf[j * m..(j + 1) * m], m);
        }
    }
}

/// Frobenius norm of a flat matrix.
#[inline]
fn frobenius_norm(m: &[f32]) -> f32 {
    crate::simd::simd_sum_sq(m, m.len()).sqrt()
}

/// Grow a Vec to `new_len` without zeroing the new tail.
///
/// The caller must guarantee that the new elements will be fully written
/// before being read (e.g., in the Newton-Schulz iteration loop where every
/// buffer element is overwritten each iteration).
///
/// Avoids the O(n) memset that `Vec::resize()` performs on the new tail.
#[allow(clippy::uninit_vec)]
#[inline]
fn grow_no_zero(v: &mut Vec<f32>, new_len: usize) {
    if v.len() >= new_len {
        return;
    }
    v.reserve(new_len - v.len());
    // SAFETY: We reserved enough capacity above, and the new elements
    // will be overwritten before being read (all NS scratch buffers are
    // fully written each iteration via copy_from_slice or matmul output).
    unsafe {
        v.set_len(new_len);
    }
}

// ── Public API ───────────────────────────────────────────────────

/// Newton-Schulz N-iteration orthogonalization (generalized).
///
/// Converts matrix `G` (`rows × cols`, row-major) to its nearest orthogonal
/// factor `X` via `n_iters` Newton-Schulz cubic iterations:
/// ```text
///   X = G / ||G||_F
///   for n_iters: A = X @ X^T; X = a*X + (b*A + c*A@A) @ X
/// ```
/// where `a = 3.4445`, `b = -4.7750`, `c = 2.0315`.
///
/// Use `n_iters = 5` for NanoGPT-scale, 7 for intermediate, 10 for DeepSeek-V4.
/// See `spectral_budget::ns_depth_for_sigma()` for modelless depth selection.
///
/// For non-square matrices where `rows > cols`, the input is transposed,
/// orthogonalized, and transposed back.
///
/// `out` must have `rows * cols` elements.
#[cfg(feature = "newton_schulz")]
pub fn newton_schulz_n(g: &[f32], rows: usize, cols: usize, out: &mut [f32], n_iters: u8) {
    assert_eq!(g.len(), rows * cols, "input matrix size mismatch");
    assert_eq!(out.len(), rows * cols, "output buffer size mismatch");
    assert!(n_iters >= 1, "n_iters must be >= 1, got {n_iters}");

    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<NewtonSchulzScratch> = RefCell::new(NewtonSchulzScratch::new(0, 0));
    }
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        newton_schulz_n_into(g, rows, cols, out, &mut s, n_iters);
    });
}

/// Newton-Schulz 5-iteration orthogonalization.
///
/// Converts matrix `G` (`rows × cols`, row-major) to its nearest orthogonal
/// factor `X` via:
/// ```text
///   X = G / ||G||_F
///   for 5 iters: A = X @ X^T; X = a*X + (b*A + c*A@A) @ X
/// ```
/// where `a = 3.4445`, `b = -4.7750`, `c = 2.0315`.
///
/// For non-square matrices where `rows > cols`, the input is transposed,
/// orthogonalized, and transposed back.
///
/// `out` must have `rows * cols` elements.
#[cfg(feature = "newton_schulz")]
pub fn newton_schulz5(g: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    newton_schulz_n(g, rows, cols, out, 5);
}

/// Muon optimizer update: momentum + Newton-Schulz orthogonalization + scaling.
///
/// For hot paths (training loops), prefer [`muon_update_into`] which avoids
/// per-call heap allocations when processing non-square matrices.
///
/// Updates the momentum buffer: `m = β * m + grad`, then computes
/// `update = newton_schulz5(m) * scaling` where `scaling = 1.0 / (rows as f32)`.
///
/// `grad` and `momentum` must have `rows * cols` elements.
/// `out` receives the orthogonalized update (`rows * cols` elements).
#[cfg(feature = "newton_schulz")]
pub fn muon_update(
    grad: &[f32],
    momentum: &mut [f32],
    beta: f32,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(grad.len(), rows * cols, "grad size mismatch");
    assert_eq!(momentum.len(), rows * cols, "momentum size mismatch");
    assert_eq!(out.len(), rows * cols, "out size mismatch");

    // Momentum accumulation: m = β*m + g — single fused FMA pass
    // (was 2 passes: scale_inplace + add_inplace).
    crate::simd::simd_fused_decay_write(momentum, beta, grad, 1.0);

    // Orthogonalize the momentum
    newton_schulz5(momentum, rows, cols, out);

    // Scale by 1/max(rows, cols) — standard Muon scaling
    let scale = 1.0 / (rows.max(cols) as f32);
    crate::simd::simd_scale_inplace(out, scale);
}

/// Zero-alloc variant of [`muon_update`].
/// Pass a pre-allocated [`NewtonSchulzScratch`] to avoid per-call heap allocations.
#[cfg(feature = "newton_schulz")]
pub fn muon_update_into(
    grad: &[f32],
    momentum: &mut [f32],
    beta: f32,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    scratch: &mut NewtonSchulzScratch,
) {
    assert_eq!(grad.len(), rows * cols, "grad size mismatch");
    assert_eq!(momentum.len(), rows * cols, "momentum size mismatch");
    assert_eq!(out.len(), rows * cols, "out size mismatch");

    // Momentum accumulation: m = β*m + g — single fused FMA pass
    // (was 2 passes: scale_inplace + add_inplace).
    crate::simd::simd_fused_decay_write(momentum, beta, grad, 1.0);

    // Orthogonalize the momentum (zero-alloc)
    newton_schulz5_into(momentum, rows, cols, out, scratch);

    // Scale by 1/max(rows, cols) — standard Muon scaling
    let scale = 1.0 / (rows.max(cols) as f32);
    crate::simd::simd_scale_inplace(out, scale);
}

// ── Zero-alloc API ───────────────────────────────────────────────

/// Pre-allocated scratch buffers for Newton-Schulz, reused across calls.
pub struct NewtonSchulzScratch {
    x: Vec<f32>,
    a_mat: Vec<f32>,
    b_mat: Vec<f32>,
    bx: Vec<f32>,
    xt_buf: Vec<f32>,
    /// Also used for non-square transpose temporaries
    gt_buf: Vec<f32>,
    ort_buf: Vec<f32>,
}

impl NewtonSchulzScratch {
    /// Create scratch space sized for matrices up to `max_m × max_n`.
    pub fn new(max_m: usize, max_n: usize) -> Self {
        let mn = max_m * max_n;
        let mm = max_m * max_m;
        Self {
            x: vec![0.0; mn],
            a_mat: vec![0.0; mm],
            b_mat: vec![0.0; mm],
            bx: vec![0.0; mn],
            xt_buf: vec![0.0; mn],
            gt_buf: vec![0.0; mn],
            ort_buf: vec![0.0; mn],
        }
    }

    /// Ensure internal buffers are large enough for `m × n`.
    ///
    /// Uses growth-only reallocation: reserves capacity and sets length without
    /// zeroing (the new tail will be overwritten before use). This avoids O(n)
    /// memset on every capacity increase during the Newton-Schulz iteration loop.
    pub fn ensure_capacity(&mut self, m: usize, n: usize) {
        let mn = m * n;
        let mm = m * m;
        grow_no_zero(&mut self.x, mn);
        grow_no_zero(&mut self.a_mat, mm);
        grow_no_zero(&mut self.b_mat, mm);
        grow_no_zero(&mut self.bx, mn);
        grow_no_zero(&mut self.xt_buf, mn);
        grow_no_zero(&mut self.gt_buf, mn);
        grow_no_zero(&mut self.ort_buf, mn);
    }
}

// ── Newton-Schulz Inverse Square Root for PSD matrices ─────────────
//
// Plan 270 (LoRA-Muon distillation). Paper Algorithm 4 from arXiv:2606.12921.
// Computes P^{-1/2} for a positive-semidefinite r×r matrix P via polynomial
// Newton-Schulz iteration with damping γ=1.001 and ε=1e-5 regularization.
//
// Used by:
//   - LoRA-Muon optimizer (riir-ai Plan 299) for Gram matrices A^T A, B^T B
//   - Gauge-invariant adapter composition (Plan 270) — projector-form updates
//   - Future Muon-family LoRA training (HTMuon, CM-LoRA evolution)
//
// Coefficients from paper Table 2 (7 iterations converge stably for σ ∈ [0,1]).

/// Inverse square root coefficients (paper Table 2, r=2 specialization).
const INV_SQRT_COEFFS: [(f32, f32, f32); 7] = [
    (7.424_865_7, -18.395_817, 12.896_721),
    (3.487_725_5, -2.330_043_6, 0.440_469_2),
    (2.776_608_5, -2.070_643_2, 0.463_022_62),
    (1.991_314_2, -1.373_936_7, 0.387_593_5),
    (1.875_463_7, -1.250_515_2, 0.375_051_53),
    (1.874_999, -1.249_998_1, 0.374_999_08),
    (1.875, -1.25, 0.375),
];

const INV_SQRT_GAMMA: f32 = 1.001;
const INV_SQRT_EPS: f32 = 1e-5;

/// Pre-allocated scratch for PSD inverse square root.
///
/// Sized for matrices up to `max_r × max_r`. Reused across calls to avoid
/// per-step heap allocations in training hot loops.
pub struct InvSqrtScratch {
    p_a: Vec<f32>,
    p_b: Vec<f32>,
    p_k_sq: Vec<f32>,
    w_mat: Vec<f32>,
    x_mat: Vec<f32>,
    xw: Vec<f32>,
    pw2: Vec<f32>,
    w_sq: Vec<f32>,
    /// Transpose of the right operand for `matmul_nn`. Lets the inner dot
    /// product read Bᵀ row-major (contiguous) instead of striding through B
    /// column-by-column.
    bt_buf: Vec<f32>,
}

impl InvSqrtScratch {
    pub fn new(max_r: usize) -> Self {
        let rr = max_r * max_r;
        Self {
            p_a: vec![0.0; rr],
            p_b: vec![0.0; rr],
            p_k_sq: vec![0.0; rr],
            w_mat: vec![0.0; rr],
            x_mat: vec![0.0; rr],
            xw: vec![0.0; rr],
            pw2: vec![0.0; rr],
            w_sq: vec![0.0; rr],
            bt_buf: vec![0.0; rr],
        }
    }

    pub fn ensure_capacity(&mut self, r: usize) {
        let rr = r * r;
        grow_no_zero(&mut self.p_a, rr);
        grow_no_zero(&mut self.p_b, rr);
        grow_no_zero(&mut self.p_k_sq, rr);
        grow_no_zero(&mut self.w_mat, rr);
        grow_no_zero(&mut self.x_mat, rr);
        grow_no_zero(&mut self.xw, rr);
        grow_no_zero(&mut self.pw2, rr);
        grow_no_zero(&mut self.w_sq, rr);
        grow_no_zero(&mut self.bt_buf, rr);
    }
}

/// Newton-Schulz inverse square root for a symmetric PSD `r × r` matrix.
///
/// Computes `P^{-1/2}` via paper Algorithm 4. The input `p` is symmetrized
/// defensively; output is symmetric `r × r`.
#[cfg(feature = "newton_schulz")]
pub fn ns_inv_sqrt_psd(p: &[f32], r: usize, out: &mut [f32], n_iters: u8) {
    assert_eq!(p.len(), r * r, "input PSD matrix size mismatch");
    assert_eq!(out.len(), r * r, "output buffer size mismatch");
    assert!(n_iters >= 1, "n_iters must be >= 1, got {n_iters}");

    use std::cell::RefCell;
    thread_local! {
        static INV_SQRT_SCRATCH: RefCell<InvSqrtScratch> =
            RefCell::new(InvSqrtScratch::new(0));
    }
    INV_SQRT_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        ns_inv_sqrt_psd_into(p, r, out, &mut s, n_iters);
    });
}

/// Zero-alloc variant of [`ns_inv_sqrt_psd`].
#[cfg(feature = "newton_schulz")]
pub fn ns_inv_sqrt_psd_into(
    p: &[f32],
    r: usize,
    out: &mut [f32],
    scratch: &mut InvSqrtScratch,
    n_iters: u8,
) {
    assert_eq!(p.len(), r * r, "input PSD matrix size mismatch");
    assert_eq!(out.len(), r * r, "output buffer size mismatch");
    assert!(n_iters >= 1, "n_iters must be >= 1, got {n_iters}");
    let rr = r * r;
    scratch.ensure_capacity(r);

    let t = frobenius_norm(&p[..rr]);
    if t < 1e-20 {
        out[..rr].fill(0.0);
        return;
    }
    let inv_t = 1.0 / t;
    let inv_sqrt_t = 1.0 / t.sqrt();

    // p_a = P/t + ε·I  (symmetrize defensively). This is P_0.
    {
        let p_norm = &mut scratch.p_a[..rr];
        for i in 0..r {
            for j in 0..r {
                let val = 0.5 * (p[i * r + j] + p[j * r + i]);
                p_norm[i * r + j] = val * inv_t;
            }
            p_norm[i * r + i] += INV_SQRT_EPS;
        }
    }

    // X_0 = I.
    let x_mat = &mut scratch.x_mat[..rr];
    x_mat.fill(0.0);
    for i in 0..r {
        x_mat[i * r + i] = 1.0;
    }

    let gamma = INV_SQRT_GAMMA;
    let inv_gamma = 1.0 / gamma;
    let inv_gamma3 = 1.0 / (gamma * gamma * gamma);
    let inv_gamma5 = inv_gamma3 / (gamma * gamma);

    let mut p_in_a = true;
    let n_iters_clamped = (n_iters as usize).min(INV_SQRT_COEFFS.len());

    for k in 0..n_iters_clamped {
        let (a_k, b_k, c_k) = INV_SQRT_COEFFS[k];

        let (p_cur, p_next) = if p_in_a {
            (&scratch.p_a[..rr], &mut scratch.p_b[..rr])
        } else {
            (&scratch.p_b[..rr], &mut scratch.p_a[..rr])
        };

        // P_k² → p_k_sq.
        matmul_symmetric(p_cur, r, &mut scratch.p_k_sq[..rr]);
        let p_sq_buf = &scratch.p_k_sq[..rr];

        // W_k = (a_k/γ)·I + (b_k/γ³)·P_k + (c_k/γ⁵)·P_k².
        {
            let w_mat = &mut scratch.w_mat[..rr];
            let a_term = a_k * inv_gamma;
            let b_term = b_k * inv_gamma3;
            let c_term = c_k * inv_gamma5;
            for i in 0..r {
                for j in 0..r {
                    let p_ij = p_cur[i * r + j];
                    let psq_ij = p_sq_buf[i * r + j];
                    let identity = if i == j { 1.0 } else { 0.0 };
                    w_mat[i * r + j] = a_term * identity + b_term * p_ij + c_term * psq_ij;
                }
            }
        }

        // X_{k+1} = X_k · W_k.
        matmul_nn(
            x_mat,
            &scratch.w_mat[..rr],
            r,
            &mut scratch.xw[..rr],
            &mut scratch.bt_buf[..rr],
        );
        x_mat.copy_from_slice(&scratch.xw[..rr]);

        if k + 1 < n_iters_clamped {
            matmul_symmetric(&scratch.w_mat[..rr], r, &mut scratch.w_sq[..rr]);
            matmul_nn(
                p_cur,
                &scratch.w_sq[..rr],
                r,
                &mut scratch.pw2[..rr],
                &mut scratch.bt_buf[..rr],
            );
            for i in 0..r {
                for j in 0..r {
                    let v = 0.5 * (scratch.pw2[i * r + j] + scratch.pw2[j * r + i]);
                    p_next[i * r + j] = v;
                }
            }
            p_in_a = !p_in_a;
        }
    }

    crate::simd::simd_scale_inplace(x_mat, inv_sqrt_t);
    out[..rr].copy_from_slice(x_mat);
}

#[inline]
fn matmul_symmetric(a: &[f32], r: usize, c: &mut [f32]) {
    for i in 0..r {
        let a_row_i = &a[i * r..(i + 1) * r];
        c[i * r + i] = crate::simd::simd_dot_f32(a_row_i, a_row_i, r);
        for j in (i + 1)..r {
            let a_row_j = &a[j * r..(j + 1) * r];
            let v = crate::simd::simd_dot_f32(a_row_i, a_row_j, r);
            c[i * r + j] = v;
            c[j * r + i] = v;
        }
    }
}

/// Compute C = A · B for `r × r` matrices. Transposes B into `bt_buf` first
/// so the inner dot product reads both operands row-major (contiguous) and can
/// hit the SIMD dot kernel; the naive column-walk through B thrashes the cache
/// on anything that doesn't fit in L1.
#[inline]
fn matmul_nn(a: &[f32], b: &[f32], r: usize, c: &mut [f32], bt_buf: &mut [f32]) {
    transpose(b, r, r, bt_buf);
    for i in 0..r {
        let a_row_i = &a[i * r..(i + 1) * r];
        let c_row_i = &mut c[i * r..(i + 1) * r];
        for j in 0..r {
            // bt_buf row j == B column j, stored contiguously.
            c_row_i[j] = crate::simd::simd_dot_f32(a_row_i, &bt_buf[j * r..(j + 1) * r], r);
        }
    }
}

/// Newton-Schulz N-iteration with pre-allocated scratch buffers (zero-alloc after first call).
#[cfg(feature = "newton_schulz")]
pub fn newton_schulz_n_into(
    g: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    scratch: &mut NewtonSchulzScratch,
    n_iters: u8,
) {
    assert_eq!(g.len(), rows * cols, "input matrix size mismatch");
    assert_eq!(out.len(), rows * cols, "output buffer size mismatch");
    assert!(n_iters >= 1, "n_iters must be >= 1, got {n_iters}");

    if rows > cols {
        let cr = cols * rows;
        scratch.ensure_capacity(cols, rows);
        transpose(g, rows, cols, &mut scratch.gt_buf[..cr]);
        {
            let NewtonSchulzScratch {
                x,
                a_mat,
                b_mat,
                bx,
                xt_buf,
                gt_buf,
                ort_buf,
            } = scratch;
            newton_schulz_n_square_into_raw(
                &gt_buf[..cr],
                cols,
                rows,
                &mut ort_buf[..cr],
                x,
                a_mat,
                b_mat,
                bx,
                xt_buf,
                n_iters,
            );
        }
        transpose(&scratch.ort_buf[..cr], cols, rows, out);
        return;
    }

    scratch.ensure_capacity(rows, cols);
    newton_schulz_n_square_into(g, rows, cols, out, scratch, n_iters);
}

/// Newton-Schulz 5-iteration with pre-allocated scratch buffers.
/// Equivalent to `newton_schulz_n_into(g, rows, cols, out, scratch, 5)`.
#[cfg(feature = "newton_schulz")]
pub fn newton_schulz5_into(
    g: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    scratch: &mut NewtonSchulzScratch,
) {
    newton_schulz_n_into(g, rows, cols, out, scratch, 5);
}

/// Core Newton-Schulz N-iteration with scratch reuse.
#[inline]
fn newton_schulz_n_square_into(
    g: &[f32],
    m: usize,
    n: usize,
    out: &mut [f32],
    scratch: &mut NewtonSchulzScratch,
    n_iters: u8,
) {
    let mn = m * n;
    let mm = m * m;
    let NewtonSchulzScratch {
        x,
        a_mat,
        b_mat,
        bx,
        xt_buf,
        ..
    } = scratch;
    newton_schulz_n_square_into_raw(
        g,
        m,
        n,
        out,
        &mut x[..mn],
        &mut a_mat[..mm],
        &mut b_mat[..mm],
        &mut bx[..mn],
        &mut xt_buf[..mn],
        n_iters,
    );
}

/// Raw implementation operating on individual scratch slices.
#[allow(clippy::too_many_arguments)]
fn newton_schulz_n_square_into_raw(
    g: &[f32],
    m: usize,
    n: usize,
    out: &mut [f32],
    x: &mut [f32],
    a_mat: &mut [f32],
    b_mat: &mut [f32],
    bx: &mut [f32],
    xt_buf: &mut [f32],
    n_iters: u8,
) {
    let mn = m * n;
    let mm = m * m;

    let norm = frobenius_norm(g);
    if norm < 1e-12 {
        out.fill(0.0);
        return;
    }
    let inv_norm = 1.0 / norm;
    x[..mn].copy_from_slice(&g[..mn]);
    crate::simd::simd_scale_inplace(&mut x[..mn], inv_norm);

    let a_mat = &mut a_mat[..mm];
    let b_mat = &mut b_mat[..mm];
    let bx = &mut bx[..mn];
    let xt_buf = &mut xt_buf[..mn];

    for _ in 0..n_iters {
        matmul_xtx(x, m, n, a_mat);
        // B = B·A + C·A², where A is symmetric (from X·Xᵀ), so A² is symmetric.
        // Exploit symmetry: compute upper triangle + mirror instead of full matmul.
        // No transpose needed since A is symmetric (A^T = A).
        for i in 0..m {
            let a_row_i = &a_mat[i * m..(i + 1) * m];
            // Diagonal
            let a2_ii = crate::simd::simd_dot_f32(a_row_i, a_row_i, m);
            b_mat[i * m + i] = B * a_row_i[i] + C * a2_ii;
            // Upper triangle + mirror
            for j in (i + 1)..m {
                let a_col_j = &a_mat[j * m..(j + 1) * m];
                let a2_ij = crate::simd::simd_dot_f32(a_row_i, a_col_j, m);
                let val = B * a_row_i[j] + C * a2_ij;
                b_mat[i * m + j] = val;
                b_mat[j * m + i] = val;
            }
        }
        matmul_ax(b_mat, x, m, n, bx, xt_buf);
        // X = A*X + BX — single fused FMA pass (was 2 passes).
        // Runs n_iters times per NS call → halves memory traffic in the
        // innermost Newton-Schulz iteration loop.
        crate::simd::simd_fused_decay_write(&mut x[..mn], A, &bx[..mn], 1.0);
    }

    out.copy_from_slice(&x[..mn]);
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple pseudo-random matrix with a fixed seed.
    fn seeded_random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        let mut s = seed;
        let mut mat = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            // xorshift64
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Map to [-1, 1]
            let v = ((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0;
            mat.push(v);
        }
        mat
    }

    /// Max absolute error between X @ X^T and the identity matrix.
    fn orthogonality_error(x: &[f32], m: usize, n: usize) -> f32 {
        let mut a = vec![0.0f32; m * m];
        matmul_xtx(x, m, n, &mut a);
        let mut max_err = 0.0f32;
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (a[i * m + j] - expected).abs();
                max_err = max_err.max(err);
            }
        }
        max_err
    }

    /// Max off-diagonal absolute value of X @ X^T.
    fn max_off_diagonal(x: &[f32], m: usize, n: usize) -> f32 {
        let mut a = vec![0.0f32; m * m];
        matmul_xtx(x, m, n, &mut a);
        let mut max_od = 0.0f32;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    max_od = max_od.max(a[i * m + j].abs());
                }
            }
        }
        max_od
    }

    #[test]
    fn test_newton_schulz_8x8_orthogonal() {
        // Newton-Schulz produces approximately orthogonal output.
        // After 5 iterations with the tuned coefficients, singular values
        // converge to [0.68, 1.12] (Keller Jordan's Muon blog).
        let g = seeded_random_matrix(42, 8, 8);
        let mut out = vec![0.0f32; 64];
        newton_schulz5(&g, 8, 8, &mut out);

        let off_diag = max_off_diagonal(&out, 8, 8);
        assert!(
            off_diag < 0.5,
            "off-diagonal should be small, max = {off_diag}"
        );

        let err = orthogonality_error(&out, 8, 8);
        assert!(
            err < 0.5,
            "X @ X^T should be approximately I, max error = {err}"
        );
    }

    #[test]
    fn test_newton_schulz_nonsquare_transpose() {
        // 12 × 6 matrix (rows > cols) — should transpose, orthogonalize, transpose back
        let g = seeded_random_matrix(99, 12, 6);
        let mut out = vec![0.0f32; 72];
        newton_schulz5(&g, 12, 6, &mut out);

        // Result is 12 × 6. Check X^T @ X ≈ I_6 (since cols < rows)
        // out is 12×6, compute X^T @ X (6×6)
        let mut xt_x = [0.0f32; 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut sum = 0.0f32;
                for k in 0..12 {
                    sum += out[k * 6 + i] * out[k * 6 + j];
                }
                xt_x[i * 6 + j] = sum;
            }
        }
        let mut max_err = 0.0f32;
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((xt_x[i * 6 + j] - expected).abs());
            }
        }
        // Newton-Schulz produces approximately orthogonal output.
        // After 5 iterations with the tuned coefficients, singular values
        // converge to [0.68, 1.12] (Keller Jordan's Muon blog).
        assert!(
            max_err < 0.5,
            "X^T @ X should be approximately I, max error = {max_err}"
        );
    }

    #[test]
    fn test_newton_schulz_identity_stays_orthogonal() {
        // 4 × 4 identity matrix — output should remain approximately orthogonal.
        // The Newton-Schulz coefficients don't perfectly preserve identity,
        // but the result should have small off-diagonal and diagonal near 1.
        let mut g = vec![0.0f32; 16];
        for i in 0..4 {
            g[i * 4 + i] = 1.0;
        }
        let mut out = vec![0.0f32; 16];
        newton_schulz5(&g, 4, 4, &mut out);

        let off_diag = max_off_diagonal(&out, 4, 4);
        assert!(
            off_diag < 0.5,
            "Identity output should have small off-diagonal, max = {off_diag}"
        );
    }

    #[test]
    fn test_newton_schulz5_into_matches_original() {
        // The zero-alloc API should produce identical results to the original.
        let g = seeded_random_matrix(42, 8, 8);
        let mut out_alloc = vec![0.0f32; 64];
        let mut out_scratch = vec![0.0f32; 64];

        newton_schulz5(&g, 8, 8, &mut out_alloc);

        let mut scratch = NewtonSchulzScratch::new(8, 8);
        newton_schulz5_into(&g, 8, 8, &mut out_scratch, &mut scratch);

        for i in 0..64 {
            assert!(
                (out_alloc[i] - out_scratch[i]).abs() < 1e-6,
                "Mismatch at index {i}: alloc={}, scratch={}",
                out_alloc[i],
                out_scratch[i]
            );
        }
    }

    #[test]
    fn test_newton_schulz5_into_nonsquare() {
        let g = seeded_random_matrix(99, 12, 6);
        let mut out_alloc = vec![0.0f32; 72];
        let mut out_scratch = vec![0.0f32; 72];

        newton_schulz5(&g, 12, 6, &mut out_alloc);

        let mut scratch = NewtonSchulzScratch::new(12, 6);
        newton_schulz5_into(&g, 12, 6, &mut out_scratch, &mut scratch);

        for i in 0..72 {
            assert!(
                (out_alloc[i] - out_scratch[i]).abs() < 1e-6,
                "Mismatch at index {i}: alloc={}, scratch={}",
                out_alloc[i],
                out_scratch[i]
            );
        }
    }

    #[test]
    fn test_muon_update_orthogonal_output() {
        let grad = seeded_random_matrix(77, 8, 8);
        let mut momentum = vec![0.0f32; 64];
        let mut out = vec![0.0f32; 64];
        muon_update(&grad, &mut momentum, 0.9, 8, 8, &mut out);

        let off_diag = max_off_diagonal(&out, 8, 8);
        assert!(
            off_diag < 0.2,
            "Muon output should have small off-diagonal, max = {off_diag}"
        );
    }

    #[test]
    fn test_muon_momentum_accumulation() {
        // Same gradient applied 3 times with β=0.9 → increasing momentum magnitude
        let grad = seeded_random_matrix(33, 4, 4);
        let mut momentum = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 16];

        let mut norms = Vec::new();
        for _ in 0..3 {
            muon_update(&grad, &mut momentum, 0.9, 4, 4, &mut out);
            // Track the momentum buffer norm (before orthogonalization in next step)
            let mom_norm: f32 = momentum.iter().map(|v| v * v).sum::<f32>().sqrt();
            norms.push(mom_norm);
        }

        // Momentum magnitude should be strictly increasing over 3 steps
        assert!(
            norms[1] > norms[0] && norms[2] > norms[1],
            "Momentum should accumulate: norms = {norms:?}"
        );
    }

    // ── Newton-Schulz inverse square root tests (Plan 270) ────────────

    fn seeded_random_psd(seed: u64, r: usize) -> Vec<f32> {
        let m = seeded_random_matrix(seed, r, r);
        let mut p = vec![0.0f32; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f32;
                for k in 0..r {
                    s += m[i * r + k] * m[j * r + k];
                }
                p[i * r + j] = s;
            }
        }
        for i in 0..r {
            p[i * r + i] += 0.1;
        }
        p
    }

    fn inv_sqrt_roundtrip_error(inv_sqrt: &[f32], p: &[f32], r: usize) -> f32 {
        let mut tmp = vec![0.0f32; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f32;
                for k in 0..r {
                    s += inv_sqrt[i * r + k] * p[k * r + j];
                }
                tmp[i * r + j] = s;
            }
        }
        let mut max_err = 0.0f32;
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f32;
                for k in 0..r {
                    s += tmp[i * r + k] * inv_sqrt[k * r + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((s - expected).abs());
            }
        }
        max_err
    }

    #[test]
    fn test_ns_inv_sqrt_identity_matrix() {
        let r = 4;
        let p: Vec<f32> = (0..r * r)
            .map(|idx| if idx % (r + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut out = vec![0.0f32; r * r];
        ns_inv_sqrt_psd(&p, r, &mut out, 7);
        let err = inv_sqrt_roundtrip_error(&out, &p, r);
        assert!(err < 0.1, "P=I roundtrip max err = {err}");
    }

    #[test]
    fn test_ns_inv_sqrt_scaled_identity() {
        let r = 3;
        let mut p = vec![0.0f32; r * r];
        for i in 0..r {
            p[i * r + i] = 4.0;
        }
        let mut out = vec![0.0f32; r * r];
        ns_inv_sqrt_psd(&p, r, &mut out, 7);
        for i in 0..r {
            assert!(
                (out[i * r + i] - 0.5).abs() < 0.05,
                "P=4I diag {} = {}",
                i,
                out[i * r + i]
            );
        }
    }

    #[test]
    fn test_ns_inv_sqrt_random_psd() {
        for seed in [42u64, 99, 777, 1234] {
            for r in [2usize, 4, 8, 16] {
                let p = seeded_random_psd(seed + r as u64, r);
                let mut out = vec![0.0f32; r * r];
                ns_inv_sqrt_psd(&p, r, &mut out, 7);
                let err = inv_sqrt_roundtrip_error(&out, &p, r);
                assert!(err < 0.05, "seed={seed} r={r}: roundtrip err = {err}");
            }
        }
    }

    #[test]
    fn test_ns_inv_sqrt_matches_alloc_vs_scratch() {
        let r = 8;
        let p = seeded_random_psd(42, r);
        let mut out_alloc = vec![0.0f32; r * r];
        let mut out_scratch = vec![0.0f32; r * r];
        ns_inv_sqrt_psd(&p, r, &mut out_alloc, 7);
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out_scratch, &mut scratch, 7);
        for i in 0..r * r {
            assert!(
                (out_alloc[i] - out_scratch[i]).abs() < 1e-6,
                "Mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_ns_inv_sqrt_output_symmetric() {
        let r = 6;
        let p = seeded_random_psd(2024, r);
        let mut out = vec![0.0f32; r * r];
        ns_inv_sqrt_psd(&p, r, &mut out, 7);
        for i in 0..r {
            for j in (i + 1)..r {
                let diff = (out[i * r + j] - out[j * r + i]).abs();
                assert!(diff < 1e-4, "Asymmetric at ({i},{j}): diff = {diff}");
            }
        }
    }

    #[test]
    fn test_ns_inv_sqrt_no_nan_inf() {
        let r = 4;
        let mut p = seeded_random_psd(13, r);
        p[0] = 1e6;
        p[r + 1] = 1e-3;
        let mut out = vec![0.0f32; r * r];
        ns_inv_sqrt_psd(&p, r, &mut out, 7);
        for v in &out {
            assert!(v.is_finite(), "Got non-finite value: {v}");
        }
    }
}
