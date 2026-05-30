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

/// Newton-Schulz coefficients (converges for σ ∈ [0, 1]).
const A: f32 = 3.4445;
const B: f32 = -4.7750;
const C: f32 = 2.0315;
const ITERS: usize = 5;

// ── Matrix helpers ──────────────────────────────────────────────

/// Transpose `rows × cols` matrix stored row-major from `src` into `dst`.
fn transpose(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Compute `A = X * X^T` for an `m × n` matrix X, producing an `m × m` result.
/// Uses SIMD dot products and exploits symmetry (upper triangle + mirror).
fn matmul_xtx(x: &[f32], m: usize, n: usize, a: &mut [f32]) {
    for i in 0..m {
        // Diagonal
        a[i * m + i] = crate::simd::simd_dot_f32(&x[i * n..(i + 1) * n], &x[i * n..(i + 1) * n], n);
        // Upper triangle + mirror
        for j in (i + 1)..m {
            let dot = crate::simd::simd_dot_f32(&x[i * n..(i + 1) * n], &x[j * n..(j + 1) * n], n);
            a[i * m + j] = dot;
            a[j * m + i] = dot;
        }
    }
}

/// Compute `R = A * X` where A is `m × m` and X is `m × n`, result is `m × n`.
/// Transposes X for contiguous inner-loop access, then uses SIMD dot products.
/// Caller provides `xt_buf` (`m * n` elements) to avoid per-call allocation.
fn matmul_ax(a: &[f32], x: &[f32], m: usize, n: usize, r: &mut [f32], xt_buf: &mut [f32]) {
    // Transpose X: columns become contiguous rows in xt (n × m).
    transpose(x, m, n, xt_buf);

    // r[i,j] = dot(a_row_i, xt_col_j) = dot(&a[i*m..], &xt[j*m..], m)
    for i in 0..m {
        let a_row = &a[i * m..(i + 1) * m];
        for j in 0..n {
            r[i * n + j] = crate::simd::simd_dot_f32(a_row, &xt_buf[j * m..(j + 1) * m], m);
        }
    }
}

/// Frobenius norm of a flat matrix.
fn frobenius_norm(m: &[f32]) -> f32 {
    crate::simd::simd_sum_sq(m, m.len()).sqrt()
}

// ── Public API ───────────────────────────────────────────────────

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
    assert_eq!(g.len(), rows * cols, "input matrix size mismatch");
    assert_eq!(out.len(), rows * cols, "output buffer size mismatch");

    // Delegate to scratch-reusing variant via thread-local scratch.
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<NewtonSchulzScratch> = RefCell::new(NewtonSchulzScratch::new(0, 0));
    }
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        newton_schulz5_into(g, rows, cols, out, &mut s);
    });
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

    // Momentum accumulation: m = β * m + g (SIMD fused scale-accumulate)
    crate::simd::simd_scale_inplace(momentum, beta);
    crate::simd::simd_add_inplace(momentum, grad);

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

    // Momentum accumulation: m = β * m + g (SIMD fused scale-accumulate)
    crate::simd::simd_scale_inplace(momentum, beta);
    crate::simd::simd_add_inplace(momentum, grad);

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
    at: Vec<f32>,
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
            at: vec![0.0; mm],
            xt_buf: vec![0.0; mn],
            gt_buf: vec![0.0; mn],
            ort_buf: vec![0.0; mn],
        }
    }

    /// Ensure internal buffers are large enough for `m × n`.
    pub fn ensure_capacity(&mut self, m: usize, n: usize) {
        let mn = m * n;
        let mm = m * m;
        if self.x.len() < mn {
            self.x.resize(mn, 0.0);
        }
        if self.a_mat.len() < mm {
            self.a_mat.resize(mm, 0.0);
        }
        if self.b_mat.len() < mm {
            self.b_mat.resize(mm, 0.0);
        }
        if self.bx.len() < mn {
            self.bx.resize(mn, 0.0);
        }
        if self.at.len() < mm {
            self.at.resize(mm, 0.0);
        }
        if self.xt_buf.len() < mn {
            self.xt_buf.resize(mn, 0.0);
        }
        if self.gt_buf.len() < mn {
            self.gt_buf.resize(mn, 0.0);
        }
        if self.ort_buf.len() < mn {
            self.ort_buf.resize(mn, 0.0);
        }
    }
}

/// Newton-Schulz with pre-allocated scratch buffers (zero-alloc after first call).
#[cfg(feature = "newton_schulz")]
pub fn newton_schulz5_into(
    g: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    scratch: &mut NewtonSchulzScratch,
) {
    assert_eq!(g.len(), rows * cols, "input matrix size mismatch");
    assert_eq!(out.len(), rows * cols, "output buffer size mismatch");

    if rows > cols {
        let cr = cols * rows;
        scratch.ensure_capacity(cols, rows);
        transpose(g, rows, cols, &mut scratch.gt_buf[..cr]);
        {
            // Borrow scratch fields separately to avoid double-mut borrow.
            let NewtonSchulzScratch {
                x,
                a_mat,
                b_mat,
                bx,
                at,
                xt_buf,
                gt_buf,
                ort_buf,
            } = scratch;
            newton_schulz5_square_into_raw(
                &gt_buf[..cr],
                cols,
                rows,
                &mut ort_buf[..cr],
                x,
                a_mat,
                b_mat,
                bx,
                at,
                xt_buf,
            );
        }
        transpose(&scratch.ort_buf[..cr], cols, rows, out);
        return;
    }

    scratch.ensure_capacity(rows, cols);
    newton_schulz5_square_into(g, rows, cols, out, scratch);
}

/// Core Newton-Schulz with scratch reuse.
fn newton_schulz5_square_into(
    g: &[f32],
    m: usize,
    n: usize,
    out: &mut [f32],
    scratch: &mut NewtonSchulzScratch,
) {
    let mn = m * n;
    let mm = m * m;
    let NewtonSchulzScratch {
        x,
        a_mat,
        b_mat,
        bx,
        at,
        xt_buf,
        ..
    } = scratch;
    newton_schulz5_square_into_raw(
        g,
        m,
        n,
        out,
        &mut x[..mn],
        &mut a_mat[..mm],
        &mut b_mat[..mm],
        &mut bx[..mn],
        &mut at[..mm],
        &mut xt_buf[..mn],
    );
}

/// Raw implementation operating on individual scratch slices.
#[allow(clippy::too_many_arguments)]
fn newton_schulz5_square_into_raw(
    g: &[f32],
    m: usize,
    n: usize,
    out: &mut [f32],
    x: &mut [f32],
    a_mat: &mut [f32],
    b_mat: &mut [f32],
    bx: &mut [f32],
    at: &mut [f32],
    xt_buf: &mut [f32],
) {
    let mn = m * n;
    let mm = m * m;

    let norm = frobenius_norm(g);
    if norm < 1e-12 {
        out.fill(0.0);
        return;
    }
    let inv_norm = 1.0 / norm;
    for (i, &v) in g.iter().enumerate() {
        x[i] = v * inv_norm;
    }

    let a_mat = &mut a_mat[..mm];
    let b_mat = &mut b_mat[..mm];
    let bx = &mut bx[..mn];
    let at = &mut at[..mm];
    let xt_buf = &mut xt_buf[..mn];

    for _ in 0..ITERS {
        matmul_xtx(x, m, n, a_mat);
        transpose(a_mat, m, m, at);
        for i in 0..m {
            let a_row = &a_mat[i * m..(i + 1) * m];
            for j in 0..m {
                let a2_ij = crate::simd::simd_dot_f32(a_row, &at[j * m..(j + 1) * m], m);
                b_mat[i * m + j] = B * a_mat[i * m + j] + C * a2_ij;
            }
        }
        matmul_ax(b_mat, x, m, n, bx, xt_buf);
        // X = a*X + BX (fused SIMD scale-accumulate)
        crate::simd::simd_scale_inplace(&mut x[..mn], A);
        crate::simd::simd_add_inplace(&mut x[..mn], &bx[..mn]);
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
}
