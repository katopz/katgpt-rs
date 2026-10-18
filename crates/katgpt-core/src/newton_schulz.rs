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
//!
//! Newton-Schulz orthogonalization + Muon momentum (Plan 152, Research 114,
//! GOAT 25/25 Bench 050). Pure substrate — self-contained f32 linear algebra,
//! zero crate:: deps, zero external deps. Extracted from katgpt-rs/src/ per
//! Issue 355 Phase 1a. Re-exported by katgpt-rs root so historical
//! `katgpt_rs::newton_schulz::*` paths continue to resolve.

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
/// Uses the blocked 4-column kernel for small n (L1-resident) and falls back
/// to per-dot `simd_dot_f32` for large n. Exploits symmetry (upper triangle + mirror).
#[inline]
fn matmul_xtx(x: &[f32], m: usize, n: usize, a: &mut [f32]) {
    let use_blocked = (16..=256).contains(&n);
    for i in 0..m {
        let row_i = &x[i * n..(i + 1) * n];
        // Diagonal
        a[i * m + i] = crate::simd::simd_dot_f32(row_i, row_i, n);
        if use_blocked {
            // Upper triangle + mirror, blocked 8 at a time
            let j_start = i + 1;
            let j_count = m.saturating_sub(j_start);
            let j_blocks = j_count / 8;

            for jb in 0..j_blocks {
                let j = j_start + jb * 8;
                let rj_rows: [&[f32]; 8] = [
                    &x[j * n..(j + 1) * n],
                    &x[(j + 1) * n..(j + 2) * n],
                    &x[(j + 2) * n..(j + 3) * n],
                    &x[(j + 3) * n..(j + 4) * n],
                    &x[(j + 4) * n..(j + 5) * n],
                    &x[(j + 5) * n..(j + 6) * n],
                    &x[(j + 6) * n..(j + 7) * n],
                    &x[(j + 7) * n..(j + 8) * n],
                ];

                let (s0, s1, s2, s3, s4, s5, s6, s7) = blocked_dot8(row_i, &rj_rows, n);
                a[i * m + j] = s0;
                a[i * m + j + 1] = s1;
                a[i * m + j + 2] = s2;
                a[i * m + j + 3] = s3;
                a[i * m + j + 4] = s4;
                a[i * m + j + 5] = s5;
                a[i * m + j + 6] = s6;
                a[i * m + j + 7] = s7;
                a[j * m + i] = s0;
                a[(j + 1) * m + i] = s1;
                a[(j + 2) * m + i] = s2;
                a[(j + 3) * m + i] = s3;
                a[(j + 4) * m + i] = s4;
                a[(j + 5) * m + i] = s5;
                a[(j + 6) * m + i] = s6;
                a[(j + 7) * m + i] = s7;
            }

            // Remaining columns (scalar tail — blocked_dot4 not used to avoid
            // FMA accumulation order differences on ill-conditioned inputs)
            for j in (j_start + j_blocks * 8)..m {
                let row_j = &x[j * n..(j + 1) * n];
                let dot = crate::simd::simd_dot_f32(row_i, row_j, n);
                a[i * m + j] = dot;
                a[j * m + i] = dot;
            }
        } else {
            // Upper triangle + mirror (scalar fallback for large n)
            for j in (i + 1)..m {
                let row_j = &x[j * n..(j + 1) * n];
                let dot = crate::simd::simd_dot_f32(row_i, row_j, n);
                a[i * m + j] = dot;
                a[j * m + i] = dot;
            }
        }
    }
}

/// Compute `R = A * X` where A is `m × m` and X is `m × n`, result is `m × n`.
/// Transposes X for contiguous inner-loop access, then uses the blocked matmul
/// kernel to process 8 output columns per A-row load (small m only).
/// Caller provides `xt_buf` (`m * n` elements) to avoid per-call allocation.
#[inline]
fn matmul_ax(a: &[f32], x: &[f32], m: usize, n: usize, r: &mut [f32], xt_buf: &mut [f32]) {
    // Transpose X: columns become contiguous rows in xt (n × m).
    transpose(x, m, n, xt_buf);

    // Use blocked kernel for small m (L1-resident). The input X is normalized
    // to ||X||_F = 1 by the caller (newton_schulz_n_square_into_raw), so all
    // eigenvalues are in [0, 1] — the blocked FMA order is numerically safe.
    if (16..=256).contains(&m) {
        matmul_at_bt_blocked(a, xt_buf, m, n, m, r);
    } else {
        // r[i,j] = dot(a_row_i, xt_col_j) = dot(&a[i*m..], &xt[j*m..], m)
        for i in 0..m {
            let a_row = &a[i * m..(i + 1) * m];
            let r_row = &mut r[i * n..(i + 1) * n];
            for j in 0..n {
                r_row[j] = crate::simd::simd_dot_f32(a_row, &xt_buf[j * m..(j + 1) * m], m);
            }
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
    use std::cell::RefCell;

    assert_eq!(g.len(), rows * cols, "input matrix size mismatch");
    assert_eq!(out.len(), rows * cols, "output buffer size mismatch");
    assert!(n_iters >= 1, "n_iters must be >= 1, got {n_iters}");
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
/// `update = newton_schulz5(m) * √max(rows, cols)` — the RMS-1.0 step scale:
/// the pre-LR update has RMS ≈ 1.0 at every shape, because the NS5 output of
/// a full-rank `[rows, cols]` input has RMS `√(1/max(rows, cols))` (Moonlight,
/// arXiv:2502.16982, Lemma 1). The Muon family's three conventions — Keller
/// Jordan `√max(1, A/B)` (update RMS `√(1/D)` at square D×D), Moonlight
/// `0.2·√max(A,B)` (RMS 0.2, matched to AdamW), and this crate's `√max(A,B)`
/// (RMS 1.0, the same family as riir-train `muon_step_dense`) — differ only
/// by constants the learning rate absorbs; a d-dependent deviation is NOT
/// absorbable that way (riir-train Bench 492). Issue 851: was
/// `1/max(rows, cols)` — a d-dependent D× deviation from all three.
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

    // Scale by √max(rows, cols) — the RMS-1.0 step scale (Issue 851): the NS5
    // output RMS is √(1/max(rows, cols)), so the product has RMS ≈ 1.0 at
    // every shape (the Muon family convention; was 1/max — d-dependent).
    let scale = (rows.max(cols) as f32).sqrt();
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

    // Scale by √max(rows, cols) — the RMS-1.0 step scale (Issue 851): the NS5
    // output RMS is √(1/max(rows, cols)), so the product has RMS ≈ 1.0 at
    // every shape (the Muon family convention; was 1/max — d-dependent).
    let scale = (rows.max(cols) as f32).sqrt();
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
/// Regularization added to the diagonal of the normalized PSD matrix before
/// the NS iteration. Pushes zero/near-zero eigenvalues away from the
/// convergence-basin edge [0,1].
///
/// Issue 043 (Approach D resolved): the original 1e-5 is sufficient for the
/// blocked `blocked_dot8` kernel. The Plan 421 divergence was caused by the
/// `blocked_dot4` tail handler (since removed), not by `blocked_dot8` proper.
/// The blocked path in `matmul_symmetric`/`matmul_nn` uses `blocked_dot8` for
/// 8-aligned blocks and `simd_dot_f32` for the tail — the same pattern as
/// `matmul_xtx` in `newton_schulz5`, which has always been numerically safe.
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
    use std::cell::RefCell;

    assert_eq!(p.len(), r * r, "input PSD matrix size mismatch");
    assert_eq!(out.len(), r * r, "output buffer size mismatch");
    assert!(n_iters >= 1, "n_iters must be >= 1, got {n_iters}");
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
            // Row pre-slicing hoists three `i * r + j` multiplies and three
            // bounds checks per element out of the inner loop. `(i == j) as u32
            // as f32` yields exactly the same 1.0/0.0 the `if` produced, so the
            // expression — and its left-to-right evaluation order — is
            // unchanged (bit-identical), but the inner loop is now branch-free
            // and vectorizable.
            for i in 0..r {
                let p_row = &p_cur[i * r..(i + 1) * r];
                let psq_row = &p_sq_buf[i * r..(i + 1) * r];
                let w_row = &mut w_mat[i * r..(i + 1) * r];
                for j in 0..r {
                    let identity = (i == j) as u32 as f32;
                    w_row[j] = a_term * identity + b_term * p_row[j] + c_term * psq_row[j];
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
            // Symmetrization touched every off-diagonal pair twice, computing
            // `0.5 * (pw2[ij] + pw2[ji])` for one slot and
            // `0.5 * (pw2[ji] + pw2[ij])` for its mirror. IEEE-754 addition of
            // two operands is commutative (`a + b` and `b + a` round to the same
            // bits for all finite inputs), so both slots always received the
            // identical value — computing it once and writing both halves the
            // work. `pw2` is also bound once instead of re-projecting the struct
            // field twice per element.
            let pw2 = &scratch.pw2[..rr];
            for i in 0..r {
                let i_row = i * r;
                p_next[i_row + i] = 0.5 * (pw2[i_row + i] + pw2[i_row + i]);
                for j in (i + 1)..r {
                    let v = 0.5 * (pw2[i_row + j] + pw2[j * r + i]);
                    p_next[i_row + j] = v;
                    p_next[j * r + i] = v;
                }
            }
            p_in_a = !p_in_a;
        }
    }

    crate::simd::simd_scale_inplace(x_mat, inv_sqrt_t);
    out[..rr].copy_from_slice(x_mat);
}

/// Compute C = A · Aᵀ for a symmetric `r × r` matrix A (so C = A²).
/// Exploits symmetry: only computes the upper triangle + mirrors to lower.
///
/// Uses `blocked_dot8` for the upper triangle when `16 ≤ r ≤ 256` (L1-resident
/// working set), falling back to per-dot `simd_dot_f32` otherwise. The blocked
/// path is numerically safe: `blocked_dot8` is used for 8-aligned blocks and
/// `simd_dot_f32` for the tail — the same pattern as `matmul_xtx` in
/// `newton_schulz5` (Issue 043 Approach D resolved, 2026-07-10).
#[inline]
fn matmul_symmetric(a: &[f32], r: usize, c: &mut [f32]) {
    let use_blocked = (16..=256).contains(&r);
    for i in 0..r {
        let a_row_i = &a[i * r..(i + 1) * r];
        // Diagonal — always uses simd_dot_f32 (single dot, no blocking benefit)
        c[i * r + i] = crate::simd::simd_dot_f32(a_row_i, a_row_i, r);

        if use_blocked {
            // Upper triangle + mirror, blocked 8 at a time
            let j_start = i + 1;
            let j_count = r.saturating_sub(j_start);
            let j_blocks = j_count / 8;

            for jb in 0..j_blocks {
                let j = j_start + jb * 8;
                let rj_rows: [&[f32]; 8] = [
                    &a[j * r..(j + 1) * r],
                    &a[(j + 1) * r..(j + 2) * r],
                    &a[(j + 2) * r..(j + 3) * r],
                    &a[(j + 3) * r..(j + 4) * r],
                    &a[(j + 4) * r..(j + 5) * r],
                    &a[(j + 5) * r..(j + 6) * r],
                    &a[(j + 6) * r..(j + 7) * r],
                    &a[(j + 7) * r..(j + 8) * r],
                ];

                let (s0, s1, s2, s3, s4, s5, s6, s7) = blocked_dot8(a_row_i, &rj_rows, r);
                c[i * r + j] = s0;
                c[i * r + j + 1] = s1;
                c[i * r + j + 2] = s2;
                c[i * r + j + 3] = s3;
                c[i * r + j + 4] = s4;
                c[i * r + j + 5] = s5;
                c[i * r + j + 6] = s6;
                c[i * r + j + 7] = s7;
                c[j * r + i] = s0;
                c[(j + 1) * r + i] = s1;
                c[(j + 2) * r + i] = s2;
                c[(j + 3) * r + i] = s3;
                c[(j + 4) * r + i] = s4;
                c[(j + 5) * r + i] = s5;
                c[(j + 6) * r + i] = s6;
                c[(j + 7) * r + i] = s7;
            }

            // Remaining columns (scalar tail — blocked_dot4 not used to avoid
            // FMA accumulation order differences on ill-conditioned inputs)
            for j in (j_start + j_blocks * 8)..r {
                let a_row_j = &a[j * r..(j + 1) * r];
                let v = crate::simd::simd_dot_f32(a_row_i, a_row_j, r);
                c[i * r + j] = v;
                c[j * r + i] = v;
            }
        } else {
            // Upper triangle + mirror (scalar fallback for large r)
            for j in (i + 1)..r {
                let a_row_j = &a[j * r..(j + 1) * r];
                let v = crate::simd::simd_dot_f32(a_row_i, a_row_j, r);
                c[i * r + j] = v;
                c[j * r + i] = v;
            }
        }
    }
}

/// Compute C = A · B for `r × r` matrices. Transposes B into `bt_buf` first
/// so the inner dot product reads both operands row-major (contiguous) and can
/// hit the SIMD dot kernel; the naive column-walk through B thrashes the cache
/// on anything that doesn't fit in L1.
///
/// Uses `matmul_at_bt_blocked` (8-wide blocked NEON kernel) when `16 ≤ r ≤ 256`,
/// falling back to per-dot `simd_dot_f32` otherwise. Numerically safe — same
/// blocked pattern as `matmul_xtx` in `newton_schulz5` (Issue 043 Approach D
/// resolved, 2026-07-10).
#[inline]
fn matmul_nn(a: &[f32], b: &[f32], r: usize, c: &mut [f32], bt_buf: &mut [f32]) {
    transpose(b, r, r, bt_buf);
    if (16..=256).contains(&r) {
        matmul_at_bt_blocked(a, bt_buf, r, r, r, c);
    } else {
        for i in 0..r {
            let a_row_i = &a[i * r..(i + 1) * r];
            let c_row_i = &mut c[i * r..(i + 1) * r];
            for j in 0..r {
                // bt_buf row j == B column j, stored contiguously.
                c_row_i[j] = crate::simd::simd_dot_f32(a_row_i, &bt_buf[j * r..(j + 1) * r], r);
            }
        }
    }
}

/// Blocked matmul kernel: C = A · Bᵀ where A is `m×k`, Bᵀ is `k×n` (row j of
/// Bᵀ = column j of B). Processes 8 output columns per A-row load.
///
/// Uses NEON on aarch64, and a 4-accumulator scalar fallback elsewhere.
/// The block size of 8 provides better amortization of the A-row load and
/// horizontal reduction overhead than a block size of 4.
#[inline]
fn matmul_at_bt_blocked(a: &[f32], bt: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    const BLOCK: usize = 8;
    let j_blocks = n / BLOCK;

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];

        // Process BLOCK output columns at a time.
        for jb in 0..j_blocks {
            let j = jb * BLOCK;
            // Gather BLOCK Bᵀ rows (each length k).
            let bt_rows: [&[f32]; BLOCK] = [
                &bt[j * k..(j + 1) * k],
                &bt[(j + 1) * k..(j + 2) * k],
                &bt[(j + 2) * k..(j + 3) * k],
                &bt[(j + 3) * k..(j + 4) * k],
                &bt[(j + 4) * k..(j + 5) * k],
                &bt[(j + 5) * k..(j + 6) * k],
                &bt[(j + 6) * k..(j + 7) * k],
                &bt[(j + 7) * k..(j + 8) * k],
            ];

            let (s0, s1, s2, s3, s4, s5, s6, s7) = blocked_dot8(a_row, &bt_rows, k);
            c_row[j] = s0;
            c_row[j + 1] = s1;
            c_row[j + 2] = s2;
            c_row[j + 3] = s3;
            c_row[j + 4] = s4;
            c_row[j + 5] = s5;
            c_row[j + 6] = s6;
            c_row[j + 7] = s7;
        }

        // Handle remaining columns (n not divisible by BLOCK).
        for j in (j_blocks * BLOCK)..n {
            c_row[j] = crate::simd::simd_dot_f32(a_row, &bt[j * k..(j + 1) * k], k);
        }
    }
}

/// Compute 8 dot products sharing the same `a` operand.
/// Returns `(dot(a,b0), dot(a,b1), ..., dot(a,b7))`.
///
/// This is the 8-wide extension of `blocked_dot4`, processing 8 output values
/// per A-row load. On NEON, uses 8 accumulator registers (well within the 32
/// NEON register file) to hide FMA pipeline latency.
#[inline(always)]
fn blocked_dot8(
    a: &[f32],
    bs: &[&[f32]; 8],
    len: usize,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { blocked_dot8_neon(a, bs, len) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        blocked_dot8_scalar(a, bs, len)
    }
}

/// NEON implementation of `blocked_dot8`: 8 dot products sharing operand `a`.
///
/// Uses 8 NEON accumulators (one per dot), processing 4 elements per FMA.
/// The 8 independent accumulators fully hide the ~4-cycle FMA pipeline latency
/// (8 FMAs per 4-cycle window = 2 FMA/cycle throughput). Register usage:
/// 8 acc + 1 a_temp + 1 b_temp = 10 live regs (within 32).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn blocked_dot8_neon(
    a: &[f32],
    bs: &[&[f32]; 8],
    len: usize,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    // 8 accumulators as individual variables (not array) to help the compiler
    // keep them in NEON registers rather than spilling to the stack.
    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };
    let mut acc4 = unsafe { vdupq_n_f32(0.0) };
    let mut acc5 = unsafe { vdupq_n_f32(0.0) };
    let mut acc6 = unsafe { vdupq_n_f32(0.0) };
    let mut acc7 = unsafe { vdupq_n_f32(0.0) };

    let mut i = 0;
    let chunks = len / 4;
    for _ in 0..chunks {
        let av = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        acc0 = unsafe { vfmaq_f32(acc0, av, vld1q_f32(bs[0].as_ptr().add(i))) };
        acc1 = unsafe { vfmaq_f32(acc1, av, vld1q_f32(bs[1].as_ptr().add(i))) };
        acc2 = unsafe { vfmaq_f32(acc2, av, vld1q_f32(bs[2].as_ptr().add(i))) };
        acc3 = unsafe { vfmaq_f32(acc3, av, vld1q_f32(bs[3].as_ptr().add(i))) };
        acc4 = unsafe { vfmaq_f32(acc4, av, vld1q_f32(bs[4].as_ptr().add(i))) };
        acc5 = unsafe { vfmaq_f32(acc5, av, vld1q_f32(bs[5].as_ptr().add(i))) };
        acc6 = unsafe { vfmaq_f32(acc6, av, vld1q_f32(bs[6].as_ptr().add(i))) };
        acc7 = unsafe { vfmaq_f32(acc7, av, vld1q_f32(bs[7].as_ptr().add(i))) };
        i += 4;
    }

    let s0 = unsafe { vaddvq_f32(acc0) };
    let s1 = unsafe { vaddvq_f32(acc1) };
    let s2 = unsafe { vaddvq_f32(acc2) };
    let s3 = unsafe { vaddvq_f32(acc3) };
    let s4 = unsafe { vaddvq_f32(acc4) };
    let s5 = unsafe { vaddvq_f32(acc5) };
    let s6 = unsafe { vaddvq_f32(acc6) };
    let s7 = unsafe { vaddvq_f32(acc7) };

    // Handle remaining elements (len not divisible by 4).
    let mut s0 = s0;
    let mut s1 = s1;
    let mut s2 = s2;
    let mut s3 = s3;
    let mut s4 = s4;
    let mut s5 = s5;
    let mut s6 = s6;
    let mut s7 = s7;
    while i < len {
        let av = unsafe { *a.get_unchecked(i) };
        s0 += av * unsafe { *bs[0].get_unchecked(i) };
        s1 += av * unsafe { *bs[1].get_unchecked(i) };
        s2 += av * unsafe { *bs[2].get_unchecked(i) };
        s3 += av * unsafe { *bs[3].get_unchecked(i) };
        s4 += av * unsafe { *bs[4].get_unchecked(i) };
        s5 += av * unsafe { *bs[5].get_unchecked(i) };
        s6 += av * unsafe { *bs[6].get_unchecked(i) };
        s7 += av * unsafe { *bs[7].get_unchecked(i) };
        i += 1;
    }

    (s0, s1, s2, s3, s4, s5, s6, s7)
}

/// Scalar fallback for `blocked_dot8` (non-NEON targets).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn blocked_dot8_scalar(
    a: &[f32],
    bs: &[&[f32]; 8],
    len: usize,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let mut s = [0.0f32; 8];
    for i in 0..len {
        let av = a[i];
        for j in 0..8 {
            s[j] = av.mul_add(bs[j][i], s[j]);
        }
    }
    (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
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
        // Uses blocked_dot8 to process 8 j-values per a_row_i load (small m only).
        let use_blocked_a2 = (16..=256).contains(&m);
        for i in 0..m {
            let a_row_i = &a_mat[i * m..(i + 1) * m];
            // Diagonal
            let a2_ii = crate::simd::simd_dot_f32(a_row_i, a_row_i, m);
            b_mat[i * m + i] = B * a_row_i[i] + C * a2_ii;
            if use_blocked_a2 {
                // Upper triangle + mirror, blocked 8 at a time
                let j_start = i + 1;
                let j_count = m.saturating_sub(j_start);
                let j_blocks = j_count / 8;

                for jb in 0..j_blocks {
                    let j = j_start + jb * 8;
                    let aj_rows: [&[f32]; 8] = [
                        &a_mat[j * m..(j + 1) * m],
                        &a_mat[(j + 1) * m..(j + 2) * m],
                        &a_mat[(j + 2) * m..(j + 3) * m],
                        &a_mat[(j + 3) * m..(j + 4) * m],
                        &a_mat[(j + 4) * m..(j + 5) * m],
                        &a_mat[(j + 5) * m..(j + 6) * m],
                        &a_mat[(j + 6) * m..(j + 7) * m],
                        &a_mat[(j + 7) * m..(j + 8) * m],
                    ];

                    let (a2_0, a2_1, a2_2, a2_3, a2_4, a2_5, a2_6, a2_7) =
                        blocked_dot8(a_row_i, &aj_rows, m);
                    let v0 = B * a_row_i[j] + C * a2_0;
                    let v1 = B * a_row_i[j + 1] + C * a2_1;
                    let v2 = B * a_row_i[j + 2] + C * a2_2;
                    let v3 = B * a_row_i[j + 3] + C * a2_3;
                    let v4 = B * a_row_i[j + 4] + C * a2_4;
                    let v5 = B * a_row_i[j + 5] + C * a2_5;
                    let v6 = B * a_row_i[j + 6] + C * a2_6;
                    let v7 = B * a_row_i[j + 7] + C * a2_7;
                    b_mat[i * m + j] = v0;
                    b_mat[i * m + j + 1] = v1;
                    b_mat[i * m + j + 2] = v2;
                    b_mat[i * m + j + 3] = v3;
                    b_mat[i * m + j + 4] = v4;
                    b_mat[i * m + j + 5] = v5;
                    b_mat[i * m + j + 6] = v6;
                    b_mat[i * m + j + 7] = v7;
                    b_mat[j * m + i] = v0;
                    b_mat[(j + 1) * m + i] = v1;
                    b_mat[(j + 2) * m + i] = v2;
                    b_mat[(j + 3) * m + i] = v3;
                    b_mat[(j + 4) * m + i] = v4;
                    b_mat[(j + 5) * m + i] = v5;
                    b_mat[(j + 6) * m + i] = v6;
                    b_mat[(j + 7) * m + i] = v7;
                }

                // Remaining columns (scalar tail — blocked_dot4 not used to avoid
                // FMA accumulation order differences on ill-conditioned inputs)
                for j in (j_start + j_blocks * 8)..m {
                    let a_col_j = &a_mat[j * m..(j + 1) * m];
                    let a2_ij = crate::simd::simd_dot_f32(a_row_i, a_col_j, m);
                    let val = B * a_row_i[j] + C * a2_ij;
                    b_mat[i * m + j] = val;
                    b_mat[j * m + i] = val;
                }
            } else {
                // Scalar fallback for large m
                for j in (i + 1)..m {
                    let a_col_j = &a_mat[j * m..(j + 1) * m];
                    let a2_ij = crate::simd::simd_dot_f32(a_row_i, a_col_j, m);
                    let val = B * a_row_i[j] + C * a2_ij;
                    b_mat[i * m + j] = val;
                    b_mat[j * m + i] = val;
                }
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
        mat.shrink_to_fit();
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

        // Scale-invariant orthogonality (Issue 851): the step scale is now
        // √max(rows, cols), so an absolute Gram threshold would test the
        // scale, not the geometry. Normalize the Gram by its mean diagonal
        // first — the ratio is the orthogonality statement.
        let mut gram = vec![0.0f32; 64];
        matmul_xtx(&out, 8, 8, &mut gram);
        let mean_diag = (0..8).map(|i| gram[i * 8 + i]).sum::<f32>() / 8.0;
        let mut off_ratio = 0.0f32;
        for i in 0..8 {
            for j in 0..8 {
                if i != j {
                    off_ratio = off_ratio.max((gram[i * 8 + j] / mean_diag).abs());
                }
            }
        }
        assert!(
            off_ratio < 0.35,
            "Muon output Gram off-diagonal/mean-diagonal = {off_ratio}"
        );
    }

    #[test]
    fn test_muon_update_rms_one_invariant() {
        // Issue 851 regression: the pre-LR update has RMS ≈ 1.0 at EVERY
        // shape — the √max(rows, cols) step scale. NS5 maps singular values
        // into ≈[0.68, 1.12] (Bench 050), so RMS(out) = √(mean sv²) lands in
        // that band — not exactly 1.0. The old 1/max scale put it at
        // max(rows, cols)^{-3/2} instead: d-dependent, the exact class
        // riir-train Bench 492 measured as un-absorbable by an LR grid.
        for (seed, rows, cols) in [(11u64, 64, 64), (12, 128, 64), (13, 64, 128)] {
            let grad = seeded_random_matrix(seed, rows, cols);
            let mut momentum = vec![0.0f32; rows * cols];
            let mut out = vec![0.0f32; rows * cols];
            muon_update(&grad, &mut momentum, 0.9, rows, cols, &mut out);
            let sum_sq: f32 = out.iter().map(|v| v * v).sum();
            let rms = (sum_sq / (rows * cols) as f32).sqrt();
            assert!(
                (0.68..=1.12).contains(&rms),
                "{rows}×{cols}: update RMS {rms} outside the NS5 band [0.68, 1.12]"
            );
        }
    }

    #[test]
    fn test_muon_momentum_accumulation() {
        // Same gradient applied 3 times with β=0.9 → increasing momentum magnitude
        let grad = seeded_random_matrix(33, 4, 4);
        let mut momentum = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 16];

        let mut norms = Vec::with_capacity(3);
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
                "Mismatch at {i}"
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
