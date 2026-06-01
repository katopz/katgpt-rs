//! River-Valley Diagnostic Metrics (Plan 152, Research 114).
//!
//! Modelless diagnostics that reveal why training is (or isn't) converging:
//! - **Subspace ratios**: dominant vs bulk gradient alignment
//! - **Effective rank**: entropy-based rank measure of a weight matrix
//! - **Update cosine similarity**: smoothness of the optimization trajectory
//!
//! No external dependencies. Pure scalar arithmetic.

#![allow(clippy::empty_line_after_doc_comments)]
// ── Helpers ──────────────────────────────────────────────────────

/// Dot product of two vectors.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::simd_dot_f32(a, b, a.len())
}

/// L2 norm of a vector.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

// ── Public API ───────────────────────────────────────────────────

/// Compute dominant/bulk subspace alignment ratios.
///
/// Given a gradient vector `g` and the top-k dominant Hessian eigenvectors
/// `U_k`, computes:
/// - `r_dom = ||U_k^T @ g|| / ||g||` — fraction of gradient in dominant subspace
/// - `r_bulk = sqrt(1 - r_dom^2)` — fraction in bulk/remaining subspace
///
/// Satisfies `r_dom^2 + r_bulk^2 = 1.0`.
///
/// `dominant_eigvecs` is a slice of k eigenvectors, each of same length as
/// `gradient`.
#[cfg(feature = "river_valley")]
pub fn subspace_ratios(gradient: &[f32], dominant_eigvecs: &[Vec<f32>]) -> (f32, f32) {
    let g_norm_sq = dot(gradient, gradient);
    if g_norm_sq < 1e-24 || dominant_eigvecs.is_empty() {
        return (0.0, 1.0);
    }

    // Project gradient onto dominant subspace: projections[k] = u_k^T @ g
    let proj_norm_sq: f32 = dominant_eigvecs
        .iter()
        .map(|u| {
            let coeff = dot(u, gradient);
            coeff * coeff
        })
        .sum();

    let r_dom = (proj_norm_sq / g_norm_sq).sqrt();
    let r_dom_clamped = r_dom.min(1.0); // numerical safety
    let r_bulk = (1.0 - r_dom_clamped * r_dom_clamped).sqrt();

    (r_dom_clamped, r_bulk)
}

/// Zero-alloc variant of [`effective_rank`].
///
/// `gram_buf` must have `>= min(rows, cols)^2` elements.
/// `eigenvalues_buf` must have `>= min(rows, cols)` elements.
/// `scratch_buf` must have `>= min(rows, cols)^2` elements.
#[cfg(feature = "river_valley")]
pub fn effective_rank_into(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    gram_buf: &mut [f32],
    eigenvalues_buf: &mut [f32],
    scratch_buf: &mut [f32],
) -> f32 {
    assert_eq!(matrix.len(), rows * cols, "matrix size mismatch");

    // Compute Gram matrix M^T @ M (n × n) or M @ M^T (rows × rows),
    // whichever is smaller.
    if rows <= cols {
        // G = M @ M^T  (rows × rows) — exploit symmetry
        gram_buf[..rows * rows].fill(0.0);
        for i in 0..rows {
            // Diagonal
            gram_buf[i * rows + i] = crate::simd::simd_dot_f32(
                &matrix[i * cols..(i + 1) * cols],
                &matrix[i * cols..(i + 1) * cols],
                cols,
            );
            // Upper triangle + mirror
            for j in (i + 1)..rows {
                let dot = crate::simd::simd_dot_f32(
                    &matrix[i * cols..(i + 1) * cols],
                    &matrix[j * cols..(j + 1) * cols],
                    cols,
                );
                gram_buf[i * rows + j] = dot;
                gram_buf[j * rows + i] = dot;
            }
        }
        erank_from_gram_into(&gram_buf[..rows * rows], rows, eigenvalues_buf, scratch_buf)
    } else {
        // G = M^T @ M  (cols × cols)
        // Accumulate outer products row-by-row for SIMD-friendly contiguous access:
        // G[i,j] = Σ_k M[k,i]·M[k,j] = Σ_k (row_k)_i · (row_k)_j
        gram_buf[..cols * cols].fill(0.0);
        for k in 0..rows {
            let row = &matrix[k * cols..(k + 1) * cols];
            for i in 0..cols {
                let a = row[i];
                for j in i..cols {
                    gram_buf[i * cols + j] += a * row[j];
                }
            }
        }
        // Symmetrize
        for i in 0..cols {
            for j in (i + 1)..cols {
                gram_buf[j * cols + i] = gram_buf[i * cols + j];
            }
        }
        erank_from_gram_into(&gram_buf[..cols * cols], cols, eigenvalues_buf, scratch_buf)
    }
}

/// Effective rank of a matrix (entropy of normalized singular values).
///
/// Computes `erank = exp(H)` where `H = -Σ σ̃_i log(σ̃_i)` and
/// `σ̃_i = σ_i / Σ σ_j` are the normalized eigenvalues of `M^T @ M`.
///
/// Uses Jacobi eigenvalue iteration on the Gram matrix to compute all
/// eigenvalues of `M^T @ M` directly (O(n²m) for n ≤ m).
#[cfg(feature = "river_valley")]
pub fn effective_rank(matrix: &[f32], rows: usize, cols: usize) -> f32 {
    assert_eq!(matrix.len(), rows * cols, "matrix size mismatch");
    let n = rows.min(cols);
    let mut gram_buf = vec![0.0f32; n * n];
    let mut eigenvalues_buf = vec![0.0f32; n];
    let mut scratch_buf = vec![0.0f32; n * n];
    effective_rank_into(
        matrix,
        rows,
        cols,
        &mut gram_buf,
        &mut eigenvalues_buf,
        &mut scratch_buf,
    )
}

/// Zero-alloc variant of `erank_from_gram` — caller provides eigenvalues and scratch buffers.
fn erank_from_gram_into(
    gram: &[f32],
    n: usize,
    eigenvalues: &mut [f32],
    scratch: &mut [f32],
) -> f32 {
    eigenvalues[..n].fill(0.0);
    scratch[..n * n].fill(0.0);
    jacobi_eigenvalues_into(gram, n, eigenvalues, scratch);

    let sum_ev: f32 = eigenvalues[..n].iter().sum();
    if sum_ev < 1e-12 {
        return 0.0;
    }

    let mut entropy = 0.0f32;
    for &ev in &eigenvalues[..n] {
        if ev > 1e-12 {
            let p = ev / sum_ev;
            entropy -= p * p.ln();
        }
    }

    entropy.exp()
}

/// Allocating wrapper — prefer `erank_from_gram_into` in hot paths.
#[allow(dead_code)]
fn erank_from_gram(gram: &[f32], n: usize) -> f32 {
    let mut eigenvalues = vec![0.0f32; n];
    let mut scratch = vec![0.0f32; n * n];
    erank_from_gram_into(gram, n, &mut eigenvalues, &mut scratch)
}

/// Jacobi eigenvalue algorithm — returns eigenvalues of an n×n symmetric matrix.
///
/// Runs a fixed number of sweeps (sufficient for convergence on small matrices).
///
/// Writes eigenvalues into `out[..n]`. Caller provides `scratch[..n*n]` to avoid
/// per-call allocation.
fn jacobi_eigenvalues_into(mat: &[f32], n: usize, out: &mut [f32], scratch: &mut [f32]) {
    if n == 0 {
        return;
    }
    if n == 1 {
        out[0] = mat[0];
        return;
    }

    // Work on a copy in caller-provided scratch
    scratch[..n * n].copy_from_slice(&mat[..n * n]);
    let a = &mut scratch[..n * n];

    // Jacobi rotations: 20 sweeps is plenty for small diagnostic matrices
    let sweeps = 20;
    for _ in 0..sweeps {
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }

                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // Rotation angle (branch-free via copysign)
                let tau = (aqq - app) / (2.0 * apq);
                let t = 1.0f32.copysign(tau) / (tau.abs() + (1.0 + tau * tau).sqrt());

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update off-diagonal — skip p/q explicitly to avoid wasted masking math.
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let new_rp = c * arp - s * arq;
                    let new_rq = s * arp + c * arq;
                    a[r * n + p] = new_rp;
                    a[p * n + r] = new_rp;
                    a[r * n + q] = new_rq;
                    a[q * n + r] = new_rq;
                }
            }
        }
    }

    // Eigenvalues are on the diagonal
    for i in 0..n {
        out[i] = a[i * n + i].max(0.0);
    }
}

/// Allocating wrapper — prefer `jacobi_eigenvalues_into` in hot paths.
#[allow(dead_code)]
fn jacobi_eigenvalues(mat: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut scratch = vec![0.0f32; n * n];
    jacobi_eigenvalues_into(mat, n, &mut out, &mut scratch);
    out
}

/// Average cosine similarity between consecutive updates.
///
/// Computes `mean(cos(Δ_t, Δ_{t+1}))` for all consecutive pairs.
/// A value near 1.0 means smooth trajectory; near 0.0 means erratic.
///
/// `updates` is a slice of update vectors (all must have the same dimension).
#[cfg(feature = "river_valley")]
pub fn update_cosine_similarity(updates: &[Vec<f32>]) -> f32 {
    if updates.len() < 2 {
        return 1.0;
    }

    let mut total = 0.0f32;
    let mut count = 0usize;

    // Rolling norm: compute each norm once, reuse as prev in next iteration.
    // Eliminates Vec allocation vs. collecting all norms upfront.
    let mut prev_norm = l2_norm(&updates[0]);
    for i in 0..(updates.len() - 1) {
        let curr_norm = l2_norm(&updates[i + 1]);
        if prev_norm < 1e-12 || curr_norm < 1e-12 {
            prev_norm = curr_norm;
            continue;
        }
        let cos = dot(&updates[i], &updates[i + 1]) / (prev_norm * curr_norm);
        total += cos;
        count += 1;
        prev_norm = curr_norm;
    }

    if count == 0 {
        1.0
    } else {
        total / count as f32
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subspace_ratios_known_subspace() {
        // 5-dim gradient, 2 dominant eigenvectors spanning first 2 dims
        let gradient = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let e1 = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0, 0.0, 0.0];

        let (r_dom, r_bulk) = subspace_ratios(&gradient, &[e1, e2]);

        // Projection onto first 2 dims: (1,1) with norm sqrt(2)
        // Total gradient norm: sqrt(5)
        // r_dom = sqrt(2/5) ≈ 0.632
        // r_bulk = sqrt(3/5) ≈ 0.775
        let expected_r_dom = (2.0_f32 / 5.0_f32).sqrt();
        let expected_r_bulk = (3.0_f32 / 5.0_f32).sqrt();

        assert!(
            (r_dom - expected_r_dom).abs() < 1e-5,
            "r_dom = {r_dom}, expected {expected_r_dom}"
        );
        assert!(
            (r_bulk - expected_r_bulk).abs() < 1e-5,
            "r_bulk = {r_bulk}, expected {expected_r_bulk}"
        );

        // Pythagorean: r_dom^2 + r_bulk^2 = 1
        let pythag = r_dom * r_dom + r_bulk * r_bulk;
        assert!(
            (pythag - 1.0).abs() < 1e-5,
            "r_dom^2 + r_bulk^2 = {pythag}, expected 1.0"
        );
    }

    #[test]
    fn test_effective_rank_identity() {
        // 4 × 4 identity → all singular values = 1 → effective rank = 4
        let mut identity = vec![0.0f32; 16];
        for i in 0..4 {
            identity[i * 4 + i] = 1.0;
        }
        let erank = effective_rank(&identity, 4, 4);

        assert!(
            (erank - 4.0).abs() < 0.1,
            "effective rank of 4×4 identity should be ≈ 4.0, got {erank}"
        );
    }

    #[test]
    fn test_effective_rank_rank_deficient() {
        // 3×3 matrix with rank 1: all rows are [1, 2, 3]
        let mat = vec![
            1.0, 2.0, 3.0, //
            1.0, 2.0, 3.0, //
            1.0, 2.0, 3.0,
        ];
        let erank = effective_rank(&mat, 3, 3);

        // Rank-1 → erank ≈ 1.0
        assert!(
            (erank - 1.0).abs() < 0.1,
            "effective rank of rank-1 matrix should be ≈ 1.0, got {erank}"
        );
    }

    #[test]
    fn test_update_cosine_similarity_constant_direction() {
        // All updates in the same direction → cosine similarity = 1.0
        let updates = vec![
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![3.0, 0.0, 0.0],
        ];
        let cos = update_cosine_similarity(&updates);
        assert!(
            (cos - 1.0).abs() < 1e-6,
            "constant direction cosine should be 1.0, got {cos}"
        );
    }

    #[test]
    fn test_update_cosine_similarity_opposite_direction() {
        // Alternating directions → cosine similarity ≈ -1.0
        let updates = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        let cos = update_cosine_similarity(&updates);
        assert!(
            (cos - (-1.0)).abs() < 1e-6,
            "opposite direction cosine should be -1.0, got {cos}"
        );
    }
}
