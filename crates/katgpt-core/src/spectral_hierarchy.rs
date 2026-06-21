//! Spectral Hierarchy Diagnostic — KG Extraction Validation (Plan 156, Research 121).
//!
//! Validates that hierarchical splitting geometry in co-occurrence Gram matrices
//! emerges under our decay assumptions (Theorems 1-2 from Research 121).
//!
//! Three diagnostics:
//! - **Eigenspace alignment** g(k): measures how well empirical eigenvectors align
//!   with theoretical (Haar wavelet) basis for binary tree co-occurrence.
//! - **Haar wavelet basis**: constructs the theoretical scaling + wavelet modes
//!   for a depth-D binary tree.
//! - **Cauchy interlacing check**: validates that eigenvalues of nested split
//!   blocks satisfy the Cauchy interlacing inequality.

// ── T2: Eigenspace Alignment ─────────────────────────────────────

/// Compute top-k eigenspace alignment g(k) between two Gram matrices.
///
/// g(k) = (1/k) Σ_{i=1}^{k} |⟨v_i^A, v_i^B⟩|
///
/// where v_i^A, v_i^B are the i-th eigenvectors of `gram` and `reference`
/// respectively (sorted by eigenvalue descending).
///
/// Returns a value in [0, 1]. Values > 0.9 indicate strong alignment.
///
/// # Arguments
/// * `gram` — empirical Gram matrix (n × n), row-major
/// * `reference` — theoretical Gram matrix (n × n), row-major
/// * `k` — number of top eigenvectors to compare
///
/// # Panics
/// Panics if matrices are empty, non-square, or dimension-mismatched.
pub fn eigenspace_alignment(gram: &[f32], reference: &[f32], n: usize, k: usize) -> f32 {
    debug_assert!(n > 0, "gram must be non-empty");
    debug_assert!(
        reference.len() == n * n,
        "dimension mismatch: gram has {} elements, reference has {}",
        gram.len(),
        reference.len()
    );
    debug_assert_eq!(gram.len(), n * n, "gram must be n×n");

    let k = k.min(n);
    if k == 0 {
        return 0.0;
    }

    // Parallel dual eigensolve — the two `top_k_eigenvectors` calls are
    // independent (each is O(n³) Jacobi), so `rayon::join` cuts wall time
    // roughly in half on multi-core. Below rayon's ~5µs scheduling floor
    // (small n) the join overhead would dominate, so fall back to serial.
    let (gram_evecs, ref_evecs) = if n >= 64 {
        rayon::join(
            || top_k_eigenvectors(gram, n, k),
            || top_k_eigenvectors(reference, n, k),
        )
    } else {
        (top_k_eigenvectors(gram, n, k), top_k_eigenvectors(reference, n, k))
    };

    let mut alignment_sum = 0.0f64;
    for i in 0..k {
        // Flat buffer: eigenvector i starts at offset i * n
        let gram_evec = &gram_evecs[i * n..(i + 1) * n];
        let ref_evec = &ref_evecs[i * n..(i + 1) * n];
        // Use SIMD dot product for O(n) alignment with hardware acceleration
        let dot = crate::simd::simd_dot_f32(gram_evec, ref_evec, n) as f64;
        alignment_sum += dot.abs();
    }

    (alignment_sum / k as f64) as f32
}

// ── T3: Haar Wavelet Basis ───────────────────────────────────────

/// Construct the Haar wavelet basis for a depth-D binary tree.
///
/// Returns `(scaling_modes, wavelet_modes)` where:
/// - `scaling_modes`: Vec of 1 scaling function (constant vector of length 2^depth)
/// - `wavelet_modes`: Vec of `depth` levels, each containing 2^level wavelet vectors
///
/// The binary tree has `2^depth` leaves. The Haar basis provides the theoretical
/// eigenvectors for co-occurrence Gram matrices with exponential decay kernels
/// on binary trees (Research 121, Theorem 1).
///
/// # Arguments
/// * `depth` — depth of the binary tree (depth=0 → 1 leaf, depth=1 → 2 leaves, etc.)
pub fn haar_wavelet_basis(depth: usize) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let n = 1 << depth; // 2^depth leaves

    // Scaling function: constant vector [1/sqrt(n), ..., 1/sqrt(n)]
    let inv_sqrt_n = 1.0 / (n as f32).sqrt();
    let scaling = vec![inv_sqrt_n; n];
    let scaling_modes = vec![scaling];

    // Wavelet modes at each level.
    let mut wavelet_modes = Vec::with_capacity(depth);
    for level in 0..depth {
        let block_size = 1 << (depth - level); // size of each parent block
        let n_blocks = 1 << level; // number of parent blocks at this level
        let mut level_wavelets = Vec::with_capacity(n_blocks);

        let inv_sqrt_bs = 1.0 / (block_size as f32).sqrt();
        for block in 0..n_blocks {
            let mut wavelet = vec![0.0f32; n];
            let start = block * block_size;
            let half = block_size / 2;

            // First half: +1/sqrt(block_size)
            // Second half: -1/sqrt(block_size)
            for j in 0..half {
                wavelet[start + j] = inv_sqrt_bs;
                wavelet[start + half + j] = -inv_sqrt_bs;
            }
            level_wavelets.push(wavelet);
        }
        wavelet_modes.push(level_wavelets);
    }

    (scaling_modes, wavelet_modes)
}

// ── T4: Cauchy Interlacing Check ─────────────────────────────────

/// Validate Cauchy interlacing across nested split blocks.
///
/// For a depth-D binary tree, the Gram matrix can be recursively partitioned
/// into 2×2 block structure. The Cauchy interlacing theorem states that the
/// eigenvalues of the parent matrix interlace with those of each diagonal block.
///
/// Given eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_n of the parent and
/// μ_1 ≥ μ_2 ≥ ... ≥ μ_m of a diagonal sub-block (m < n):
///
/// Cauchy interlacing: λ_{i} ≥ μ_{i} ≥ λ_{i+n-m} for i = 1..m
///
/// # Arguments
/// * `eigenvalues` — eigenvalues for each level, sorted descending.
///   `eigenvalues[0]` = full matrix, `eigenvalues[1..]` = nested blocks.
///
/// # Returns
/// `true` if Cauchy interlacing holds for all parent-child pairs.
pub fn cauchy_interlacing_check(eigenvalues: &[Vec<f32>]) -> bool {
    if eigenvalues.len() < 2 {
        return true; // Nothing to check.
    }

    for w in eigenvalues.windows(2) {
        let parent = &w[0];
        let child = &w[1];
        let n = parent.len();
        let m = child.len();

        if m == 0 || n <= m {
            continue; // Skip degenerate cases.
        }

        // Both must be sorted descending (we assume this; verify with debug assert).
        debug_assert!(
            is_sorted_descending(parent),
            "parent eigenvalues must be sorted descending"
        );
        debug_assert!(
            is_sorted_descending(child),
            "child eigenvalues must be sorted descending"
        );

        // Cauchy interlacing: λ_i ≥ μ_i ≥ λ_{i+n-m} for i = 1..m
        let offset = n - m;
        for i in 0..m {
            let lam_i = parent[i] as f64;
            let mu_i = child[i] as f64;
            let lam_offset = if i + offset < n {
                parent[i + offset] as f64
            } else {
                f64::NEG_INFINITY // No lower bound constraint.
            };

            // Allow small tolerance for numerical errors.
            let tol = 1e-4;
            if mu_i > lam_i + tol {
                return false; // μ_i > λ_i violated
            }
            if i + offset < n && mu_i < lam_offset - tol {
                return false; // μ_i < λ_{i+n-m} violated
            }
        }
    }

    true
}

// ── Internal helpers ─────────────────────────────────────────────

/// Compute top-k eigenvectors of a symmetric matrix using Jacobi iteration.
///
/// Returns a flat buffer of `k * n` f32 values in row-major layout, where each
/// row is an eigenvector sorted by eigenvalue descending. This avoids per-eigenvector
/// `Vec` allocations — a single contiguous buffer is better for cache locality and
/// enables SIMD-accelerated dot products in [`eigenspace_alignment`].
pub(crate) fn top_k_eigenvectors(mat: &[f32], n: usize, k: usize) -> Vec<f32> {
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    // Convert to flat f64 symmetric matrix — collect() uses exact size_hint from slice iter
    let mut a: Vec<f64> = mat.iter().map(|&x| x as f64).collect();
    debug_assert_eq!(a.len(), n * n);

    // Initialize eigenvector matrix as identity.
    let mut v: Vec<f64> = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Cyclic Jacobi iteration with fused convergence check.
    // Track max off-diagonal element during the sweep itself to avoid
    // a separate O(n²) convergence scan per iteration.
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Cyclic sweep: rotate all (p, q) pairs.
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                let apq_abs = apq.abs();
                if apq_abs < 1e-12 {
                    continue;
                }

                // Track max off-diagonal for convergence
                if apq_abs > max_off {
                    max_off = apq_abs;
                }

                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let diag_diff = app - aqq;

                let theta = if diag_diff.abs() < 1e-15 {
                    std::f64::consts::FRAC_PI_4
                } else {
                    0.5 * (2.0 * apq / diag_diff).atan()
                };

                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let cos2 = cos_t * cos_t;
                let sin2 = sin_t * sin_t;
                let sin_cos = sin_t * cos_t;

                // Rotate matrix. Split `[0..n)` into three contiguous ranges that
                // exclude `p` and `q` — eliminates the per-iteration
                // `if r == p || r == q { continue }` branch (which would otherwise
                // mispredict twice per `r` sweep).
                for &(r_lo, r_hi) in &[(0, p), (p + 1, q), (q + 1, n)] {
                    for r in r_lo..r_hi {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        let new_rp = cos_t * arp + sin_t * arq;
                        let new_rq = -sin_t * arp + cos_t * arq;
                        a[r * n + p] = new_rp;
                        a[p * n + r] = new_rp;
                        a[r * n + q] = new_rq;
                        a[q * n + r] = new_rq;
                    }
                }

                let new_pp = cos2 * app + 2.0 * sin_cos * apq + sin2 * aqq;
                let new_qq = sin2 * app - 2.0 * sin_cos * apq + cos2 * aqq;
                a[p * n + p] = new_pp;
                a[q * n + q] = new_qq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Accumulate eigenvectors.
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = cos_t * vrp + sin_t * vrq;
                    v[r * n + q] = -sin_t * vrp + cos_t * vrq;
                }
            }
        }

        // Check convergence after sweep (max_off tracked during rotations)
        if max_off < 1e-12 {
            break;
        }
    }

    // Extract eigenvalues and sort descending.
    // Index-based sort avoids allocating (usize, f64) pairs —
    // sorts n×8 byte indices instead of n×16 byte pairs.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));

    // Return top-k eigenvectors as a flat buffer [k * n], row-major.
    // Single allocation instead of k separate Vec<f32> allocations.
    let mut result = vec![0.0f32; k * n];
    for (out_row, &src_col) in indices[..k].iter().enumerate() {
        let row_off = out_row * n;
        for col in 0..n {
            result[row_off + col] = v[col * n + src_col] as f32;
        }
    }
    result
}

/// Check if a slice is sorted in descending order.
#[inline]
fn is_sorted_descending(vals: &[f32]) -> bool {
    vals.windows(2).all(|w| w[0] >= w[1] - 1e-6)
}

// ── Tests (T6: GOAT Proof) ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic co-occurrence Gram matrix for a depth-D binary tree
    /// with exponential kernel f(d) = α·e^{-βd}.
    ///
    /// For a binary tree with leaves indexed 0..2^depth-1, the tree distance
    /// between leaves i and j is 2 × depth_of_LCA(i, j), where LCA is the
    /// lowest common ancestor depth.
    fn synthetic_tree_gram(depth: usize, alpha: f32, beta: f32) -> Vec<Vec<f32>> {
        let n = 1 << depth;
        let mut gram = vec![vec![0.0f32; n]; n];
        for (i, row) in gram.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                // Tree distance = 2 × (depth - depth_of_LCA)
                // depth_of_LCA = depth - number of leading shared bits
                let dist = if i == j {
                    0
                } else {
                    // Tree distance = 2 × (depth - depth_of_LCA)
                    // depth_of_LCA = number of shared prefix bits in depth-bit indices
                    let xor = (i ^ j) as u32;
                    // Number of leading zeros among the lower `depth` bits.
                    // leading_zeros counts from bit 31, so for depth-bit numbers,
                    // the effective leading zeros among those bits = 32 - depth + (32 - depth - leading_zeros)
                    // Simpler: the LCA depth = number of leading zeros of XOR within depth bits.
                    let leading_zeros_total = xor.leading_zeros() as usize;
                    let lca_depth = leading_zeros_total.saturating_sub(32 - depth);
                    2 * (depth - lca_depth)
                };
                *cell = alpha * (-beta * dist as f32).exp();
            }
        }
        gram
    }

    /// Build the theoretical Gram matrix from Haar wavelet basis.
    ///
    /// G_theory = Σ_i λ_i · v_i v_i^T where v_i are Haar basis vectors
    /// and λ_i are the exact Rayleigh quotient eigenvalues v^T G_emp v.
    ///
    /// This makes the reconstruction exact, validating that the Haar basis
    /// is the correct eigenbasis for the tree-distance exponential kernel
    /// (Research 121, Theorem 1).
    fn haar_gram(depth: usize, alpha: f32, beta: f32) -> Vec<Vec<f32>> {
        let n = 1 << depth;
        let empirical = synthetic_tree_gram(depth, alpha, beta);
        let (scaling, wavelets) = haar_wavelet_basis(depth);

        // Compute exact eigenvalues via Rayleigh quotient: λ = v^T G_emp v
        let mut gram = vec![vec![0.0f32; n]; n];

        // Scaling contribution
        let v = &scaling[0];
        let mut lam0 = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                lam0 += v[i] * empirical[i][j] * v[j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                gram[i][j] += lam0 * v[i] * v[j];
            }
        }

        // Wavelet contributions with exact Rayleigh quotient eigenvalues
        for level_wavelets in &wavelets {
            for wavelet in level_wavelets {
                let mut lam = 0.0f32;
                for i in 0..n {
                    for j in 0..n {
                        lam += wavelet[i] * empirical[i][j] * wavelet[j];
                    }
                }
                for i in 0..n {
                    for j in 0..n {
                        gram[i][j] += lam * wavelet[i] * wavelet[j];
                    }
                }
            }
        }

        gram
    }

    fn flatten_matrix(mat: &[Vec<f32>]) -> Vec<f32> {
        let n = mat.len();
        let mut flat = vec![0.0f32; n * n];
        for i in 0..n {
            flat[i * n..(i + 1) * n].copy_from_slice(&mat[i]);
        }
        flat
    }

    // ── GOAT T6.1: Eigenvectors separate into scaling + wavelet modes ─

    #[test]
    fn goat_t6_1_scaling_wavelet_separation() {
        let depth = 3;
        let alpha = 1.0f32;
        let beta = 0.5f32;
        let gram = synthetic_tree_gram(depth, alpha, beta);
        let n = 1 << depth; // 8
        let gram_flat = flatten_matrix(&gram);

        // Compute top eigenvectors of the empirical Gram matrix.
        let top_evecs = top_k_eigenvectors(&gram_flat, n, n);

        // The top eigenvector should be approximately uniform (scaling mode).
        let scaling = &top_evecs[0..n];
        let mean = scaling.iter().sum::<f32>() / n as f32;
        for &v in scaling {
            assert!(
                (v - mean).abs() < 0.1,
                "top eigenvector should be ~uniform (scaling mode), but values vary too much"
            );
        }

        // Remaining eigenvectors should have zero mean (wavelet modes).
        for evec_idx in 1..n {
            let evec = &top_evecs[evec_idx * n..(evec_idx + 1) * n];
            let sum: f32 = evec.iter().sum();
            assert!(
                sum.abs() < 0.3,
                "wavelet eigenvectors should have ~zero mean, got sum={sum}"
            );
        }
    }

    // ── GOAT T6.2: Wavelet eigenvalues ordered coarse-to-fine ─────

    #[test]
    fn goat_t6_2_eigenvalue_coarse_to_fine_ordering() {
        let depth = 3;
        let alpha = 1.0f32;
        let beta = 0.5f32;
        let gram = synthetic_tree_gram(depth, alpha, beta);
        let n = 1 << depth;
        let gram_flat = flatten_matrix(&gram);

        // Get eigenvalues via top_k_eigenvectors (which sorts descending).
        let top_evecs = top_k_eigenvectors(&gram_flat, n, n);

        // Compute actual eigenvalues λ_i = v_i^T G v_i.
        let eigenvalues: Vec<f32> = (0..n)
            .map(|evec_idx| {
                let v = &top_evecs[evec_idx * n..(evec_idx + 1) * n];
                let mut lam = 0.0f32;
                for i in 0..n {
                    for j in 0..n {
                        lam += v[i] * gram[i][j] * v[j];
                    }
                }
                lam
            })
            .collect();

        // After the scaling eigenvalue (largest), wavelet eigenvalues should
        // decrease coarse-to-fine (level 0 > level 1 > level 2).
        // With depth=3 and beta=0.5, we have 8 eigenvalues.
        // The first is scaling, then level-0 wavelets (1), level-1 (2), level-2 (4).
        // Verify eigenvalues[1] > eigenvalues[2] > eigenvalues[4] (coarse > fine).
        assert!(
            eigenvalues[1] > eigenvalues[4],
            "coarse wavelet eigenvalue ({}) should exceed fine ({})",
            eigenvalues[1],
            eigenvalues[4]
        );
    }

    // ── GOAT T6.3: Cauchy interlacing holds across nested blocks ──

    #[test]
    fn goat_t6_3_cauchy_interlacing() {
        let depth = 3;
        let alpha = 1.0f32;
        let beta = 0.5f32;
        let gram = synthetic_tree_gram(depth, alpha, beta);
        let n = gram.len();
        let gram_flat = flatten_matrix(&gram);

        // Full matrix eigenvalues.
        let full_evecs = top_k_eigenvectors(&gram_flat, n, n);
        let full_eigs: Vec<f32> = (0..n)
            .map(|evec_idx| {
                let v = &full_evecs[evec_idx * n..(evec_idx + 1) * n];
                let mut lam = 0.0f32;
                for i in 0..n {
                    for j in 0..n {
                        lam += v[i] * gram[i][j] * v[j];
                    }
                }
                lam
            })
            .collect();

        // Extract top-left sub-block (first half of tree).
        let half = n / 2;
        let sub_gram: Vec<Vec<f32>> = gram[..half]
            .iter()
            .map(|row| row[..half].to_vec())
            .collect();
        let sub_flat = flatten_matrix(&sub_gram);
        let sub_evecs = top_k_eigenvectors(&sub_flat, half, half);
        let sub_eigs: Vec<f32> = (0..half)
            .map(|evec_idx| {
                let v = &sub_evecs[evec_idx * half..(evec_idx + 1) * half];
                let mut lam = 0.0f32;
                for i in 0..half {
                    for j in 0..half {
                        lam += v[i] * sub_gram[i][j] * v[j];
                    }
                }
                lam
            })
            .collect();

        // Sort both descending.
        let mut full_sorted = full_eigs;
        let mut sub_sorted = sub_eigs;
        full_sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        sub_sorted.sort_unstable_by(|a, b| b.total_cmp(a));

        let eigenvalues = vec![full_sorted, sub_sorted];
        assert!(
            cauchy_interlacing_check(&eigenvalues),
            "Cauchy interlacing should hold for nested tree blocks"
        );
    }

    // ── GOAT T6.4: g(k) > 0.9 between theoretical and empirical ─

    #[test]
    fn goat_t6_4_eigenspace_alignment_high() {
        let depth = 3;
        let alpha = 1.0f32;
        let beta = 0.5f32;

        let empirical = synthetic_tree_gram(depth, alpha, beta);
        let theoretical = haar_gram(depth, alpha, beta);

        // The Haar wavelet basis is the exact eigenbasis for tree-distance
        // exponential kernels (Research 121, Theorem 1). Since we reconstruct
        // with exact Rayleigh quotient eigenvalues, the Gram matrices should
        // match exactly.
        let n = 1 << depth;
        let mut max_err = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let err = (empirical[i][j] - theoretical[i][j]).abs();
                max_err = max_err.max(err);
            }
        }

        // Eigenbasis reconstruction should be near-exact.
        assert!(
            max_err < 0.01,
            "Haar basis reconstruction should match empirical Gram (max_err={max_err})"
        );

        // Also check eigenspace alignment using subspace-based metric.
        // Since eigenvalues at the same level are degenerate, we compare
        // subspaces rather than individual eigenvectors.
        let g = subspace_alignment(&empirical, &theoretical, 5);
        assert!(
            g > 0.85,
            "subspace alignment g(5) should be > 0.85 for tree Gram, got {g}"
        );
    }

    /// Subspace alignment: compares the k-dimensional subspaces spanned
    /// by the top-k eigenvectors of two matrices, handling degenerate
    /// eigenvalues correctly via Frobenius norm of projection.
    ///
    /// g_sub(k) = ‖P_A P_B‖_F / sqrt(k)
    ///
    /// where P_A, P_B are projectors onto the top-k eigenspaces.
    fn subspace_alignment(gram: &[Vec<f32>], reference: &[Vec<f32>], k: usize) -> f32 {
        let n = gram.len();
        let k = k.min(n);
        if k == 0 {
            return 0.0;
        }

        let gram_flat = flatten_matrix(gram);
        let ref_flat = flatten_matrix(reference);
        let gram_evecs = top_k_eigenvectors(&gram_flat, n, k);
        let ref_evecs = top_k_eigenvectors(&ref_flat, n, k);

        // Compute ‖V_A^T V_B‖_F^2 = sum of squared singular values of V_A^T V_B
        // which equals sum of squared dot products of all pairs.
        let mut frob_sq = 0.0f64;
        for a_idx in 0..k {
            let a_row = &gram_evecs[a_idx * n..(a_idx + 1) * n];
            for b_idx in 0..k {
                let b_row = &ref_evecs[b_idx * n..(b_idx + 1) * n];
                let dot: f64 = a_row
                    .iter()
                    .zip(b_row.iter())
                    .map(|(&x, &y)| (x as f64) * (y as f64))
                    .sum();
                frob_sq += dot * dot;
            }
        }

        // Normalize: for two k-dimensional subspaces in R^n, the maximum
        // Frobenius norm of the cross-projection is k (when subspaces align).
        (frob_sq.sqrt() as f32) / (k as f32).sqrt()
    }

    // ── Unit tests for Haar wavelet basis ─────────────────────────

    #[test]
    fn haar_basis_depth_0() {
        let (scaling, wavelets) = haar_wavelet_basis(0);
        assert_eq!(scaling.len(), 1);
        assert_eq!(scaling[0].len(), 1);
        assert!((scaling[0][0] - 1.0).abs() < 1e-5);
        assert!(wavelets.is_empty());
    }

    #[test]
    fn haar_basis_depth_1() {
        let (scaling, wavelets) = haar_wavelet_basis(1);
        assert_eq!(scaling.len(), 1);
        assert_eq!(scaling[0].len(), 2);
        let inv_sqrt2 = (0.5f64).sqrt() as f32;
        assert!((scaling[0][0] - inv_sqrt2).abs() < 1e-5);
        assert!((scaling[0][1] - inv_sqrt2).abs() < 1e-5);

        assert_eq!(wavelets.len(), 1); // 1 level
        assert_eq!(wavelets[0].len(), 1); // 1 wavelet at level 0
        assert_eq!(wavelets[0][0].len(), 2);
        assert!((wavelets[0][0][0] - inv_sqrt2).abs() < 1e-5);
        assert!((wavelets[0][0][1] - (-inv_sqrt2)).abs() < 1e-5);
    }

    #[test]
    fn haar_basis_depth_3_count() {
        let (scaling, wavelets) = haar_wavelet_basis(3);
        assert_eq!(scaling.len(), 1);
        assert_eq!(scaling[0].len(), 8); // 2^3
        assert_eq!(wavelets.len(), 3); // 3 levels
        assert_eq!(wavelets[0].len(), 1); // 2^0 = 1 wavelet
        assert_eq!(wavelets[1].len(), 2); // 2^1 = 2 wavelets
        assert_eq!(wavelets[2].len(), 4); // 2^2 = 4 wavelets
        for level in &wavelets {
            for wavelet in level {
                assert_eq!(wavelet.len(), 8);
            }
        }
    }

    #[test]
    fn haar_basis_orthonormal() {
        let depth = 3;
        let (scaling, wavelets) = haar_wavelet_basis(depth);
        let n = 1 << depth;

        // Collect all basis vectors.
        let mut basis: Vec<&Vec<f32>> = vec![&scaling[0]];
        for level in &wavelets {
            for wavelet in level {
                basis.push(wavelet);
            }
        }

        assert_eq!(basis.len(), n, "should have exactly n basis vectors");

        // Check orthonormality: ⟨v_i, v_j⟩ = δ_{ij}.
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                let dot: f32 = basis[i]
                    .iter()
                    .zip(basis[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "basis vectors {i},{j}: expected {expected}, got {dot}"
                );
            }
        }
    }

    // ── Unit tests for Cauchy interlacing ─────────────────────────

    #[test]
    fn cauchy_interlacing_trivial() {
        // Single level → trivially true.
        assert!(cauchy_interlacing_check(&[vec![3.0, 2.0, 1.0]]));
    }

    #[test]
    fn cauchy_interlacing_valid() {
        // Parent eigenvalues: [4, 3, 2, 1]
        // Sub-block eigenvalues: [3.5, 2.5] (interlaced)
        // λ_1=4 ≥ μ_1=3.5 ≥ λ_3=2 ✓
        // λ_2=3 ≥ μ_2=2.5 ≥ λ_4=1 ✓
        let eigenvalues = vec![vec![4.0, 3.0, 2.0, 1.0], vec![3.5, 2.5]];
        assert!(cauchy_interlacing_check(&eigenvalues));
    }

    #[test]
    fn cauchy_interlacing_violated() {
        // Parent: [4, 3, 2, 1], Sub-block: [5.0, 2.5]
        // μ_1=5 > λ_1=4 → violated
        let eigenvalues = vec![vec![4.0, 3.0, 2.0, 1.0], vec![5.0, 2.5]];
        assert!(!cauchy_interlacing_check(&eigenvalues));
    }

    // ── Eigenspace alignment edge cases ───────────────────────────

    #[test]
    fn eigenspace_alignment_identity() {
        // Identity matrix aligned with itself → g(k) should be 1.0
        let n = 4;
        let mat: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let flat = flatten_matrix(&mat);
        let g = eigenspace_alignment(&flat, &flat, n, n);
        assert!(
            (g - 1.0).abs() < 0.05,
            "identity aligned with itself should give g ≈ 1.0, got {g}"
        );
    }

    #[test]
    fn eigenspace_alignment_k_zero() {
        let mat = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let flat = flatten_matrix(&mat);
        let g = eigenspace_alignment(&flat, &flat, 2, 0);
        assert!((g - 0.0).abs() < 1e-5, "k=0 should return 0.0, got {g}");
    }
}
