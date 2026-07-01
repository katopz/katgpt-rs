//! Random rotation and projection matrices for TurboQuant.
//!
//! Generates deterministic orthogonal matrices via QR decomposition
//! (modified Gram-Schmidt) for random rotation, and i.i.d. Gaussian
//! matrices for QJL residual estimation.

/// Generate a random orthogonal matrix via QR decomposition.
///
/// Uses modified Gram-Schmidt for numerical stability.
/// Deterministic from seed for reproducibility across runs.
pub fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = katgpt_core::types::Rng::new(seed);
    let mut mat = vec![0.0f32; dim * dim];
    for val in mat.iter_mut() {
        *val = rng.normal();
    }

    // QR decomposition via modified Gram-Schmidt (column-major for processing)
    let mut q = vec![0.0f32; dim * dim];
    // Use a single flat buffer for column vectors instead of per-column Vec allocation.
    // Column data stored at [col * dim + row] (column-major layout).
    let mut v_flat = vec![0.0f32; dim * dim];

    // Fill column vectors from row-major random matrix
    for col in 0..dim {
        for row in 0..dim {
            v_flat[col * dim + row] = mat[row * dim + col];
        }
    }

    for i in 0..dim {
        // Subtract projections onto previous columns
        for j in 0..i {
            // q is column-major: q[col * dim + row]. Original dot was q[k*dim+j] * v_flat[i*dim+k]
            // = Q[j,k] * V[k,i] = (Q @ V)[j,i]. Can't directly use row-major simd_dot.
            let dot: f32 = (0..dim).map(|k| q[k * dim + j] * v_flat[i * dim + k]).sum();
            for k in 0..dim {
                v_flat[i * dim + k] -= dot * q[k * dim + j];
            }
        }

        // Normalize column i to get q_i
        let norm = katgpt_core::simd::simd_sum_sq(&v_flat[i * dim..i * dim + dim], dim).sqrt();
        if norm > 1e-8 {
            for k in 0..dim {
                q[k * dim + i] = v_flat[i * dim + k] / norm;
            }
        }
    }

    q
}

/// Generate QJL projection matrix (i.i.d. N(0,1) entries).
///
/// Used for residual estimation in online quantization quality tracking.
/// Seed is offset from the rotation seed to ensure independence.
pub fn generate_qjl_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = katgpt_core::types::Rng::new(seed.wrapping_add(42));
    let mut mat = vec![0.0f32; dim * dim];
    for val in mat.iter_mut() {
        *val = rng.normal();
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_is_orthogonal() {
        let dim = 8;
        let q = generate_rotation_matrix(dim, 42);

        // Q^T * Q should approximate identity
        for i in 0..dim {
            for j in 0..dim {
                let dot: f32 = (0..dim).map(|k| q[k * dim + i] * q[k * dim + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 0.1,
                    "Q^T*Q[{i}][{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_rotation_deterministic() {
        let q1 = generate_rotation_matrix(16, 123);
        let q2 = generate_rotation_matrix(16, 123);
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let dim = 16;
        let q = generate_rotation_matrix(dim, 77);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let original_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let rotated: Vec<f32> = (0..dim)
            .map(|i| (0..dim).map(|j| q[j * dim + i] * v[j]).sum())
            .collect();
        let rotated_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (original_norm - rotated_norm).abs() < 0.01,
            "norm changed: {original_norm} -> {rotated_norm}"
        );
    }

    #[test]
    fn test_qjl_matrix_variance() {
        let dim = 32;
        let s = generate_qjl_matrix(dim, 99);
        let mean: f32 = s.iter().sum::<f32>() / s.len() as f32;
        let var: f32 = s.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / s.len() as f32;
        // Should be approximately 1.0 (N(0,1) entries)
        assert!((var - 1.0).abs() < 0.5, "QJL variance {var}, expected ~1.0");
    }
}
