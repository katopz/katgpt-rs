//! Rotation transforms for SpectralQuant — spectral (calibrated).
//!
//! Spectral rotation uses learned eigenvectors from covariance analysis.
//! `RandomRotation` (TurboQuant baseline/fallback) is gated behind the `turboquant` feature.

// ---------------------------------------------------------------------------
// Vendored `generate_rotation_matrix` (Issue 015 Phase 2)
// ---------------------------------------------------------------------------
//
// Inlined from `katgpt-rs/src/turboquant/rotation.rs` so that katgpt-spectral
// can be consumed standalone (turboquant is not its own crate yet). The
// function is 50 lines and depends only on `katgpt_core::types::Rng` +
// `katgpt_core::simd::simd_sum_sq`, both available here. When turboquant is
// eventually extracted into its own crate, this should be replaced by a dep.

/// Generate a random orthogonal matrix via QR decomposition (modified Gram-Schmidt).
/// Deterministic from seed for reproducibility across runs.
#[allow(dead_code)] // gated by `turboquant` feature at call site; always compiled for crate-internal reuse
fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = katgpt_core::types::Rng::new(seed);
    let mut mat = vec![0.0f32; dim * dim];
    for val in mat.iter_mut() {
        *val = rng.normal();
    }

    let mut q = vec![0.0f32; dim * dim];
    // Column data stored at [col * dim + row] (column-major layout).
    let mut v_flat = vec![0.0f32; dim * dim];

    for col in 0..dim {
        for row in 0..dim {
            v_flat[col * dim + row] = mat[row * dim + col];
        }
    }

    for i in 0..dim {
        for j in 0..i {
            let dot: f32 = (0..dim).map(|k| q[k * dim + j] * v_flat[i * dim + k]).sum();
            for k in 0..dim {
                v_flat[i * dim + k] -= dot * q[k * dim + j];
            }
        }
        let norm = katgpt_core::simd::simd_sum_sq(&v_flat[i * dim..i * dim + dim], dim).sqrt();
        if norm > 1e-8 {
            for k in 0..dim {
                q[k * dim + i] = v_flat[i * dim + k] / norm;
            }
        }
    }
    q
}

/// Data-driven orthogonal rotation using calibrated eigenvectors.
///
/// Forward: x_hat = V^T @ x  (project into spectral basis)
/// Inverse: x = V @ x_hat   (reconstruct original basis)
///
/// Stores both `eigenvectors` (V, row-major) and `eigenvectors_t` (V^T,
/// row-major) so both forward and inverse paths use contiguous-access
/// SIMD row dot-products via `simd_matmul_rows`. The transpose is computed
/// once at construction time.
pub struct SpectralRotation {
    eigenvectors: Vec<f32>, // (head_dim × head_dim), row-major (V)
    /// Transpose of `eigenvectors` (V^T), row-major. Precomputed once so that
    /// the forward rotation `out = V^T @ x` uses `simd_matmul_rows`.
    eigenvectors_t: Vec<f32>,
    head_dim: usize,
}

impl SpectralRotation {
    pub fn new(eigenvectors: Vec<f32>, head_dim: usize) -> Self {
        assert_eq!(
            eigenvectors.len(),
            head_dim * head_dim,
            "eigenvectors must be head_dim × head_dim"
        );
        // Precompute V^T so both rotate() and unrotate() use contiguous-row
        // SIMD dot products instead of strided column gathers.
        let mut eigenvectors_t = vec![0.0f32; head_dim * head_dim];
        for i in 0..head_dim {
            for j in 0..head_dim {
                eigenvectors_t[j * head_dim + i] = eigenvectors[i * head_dim + j];
            }
        }
        Self {
            eigenvectors,
            eigenvectors_t,
            head_dim,
        }
    }

    /// Forward rotation: out = V^T @ x.
    ///
    /// Uses the precomputed `eigenvectors_t` (V^T stored row-major) so the
    /// computation reduces to `out[j] = dot(V^T_row_j, x)` — a contiguous-access
    /// SIMD row dot-product via `simd_matmul_rows`.
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.head_dim);
        assert_eq!(out.len(), self.head_dim);
        katgpt_core::simd::simd_matmul_rows(
            out,
            &self.eigenvectors_t,
            x,
            self.head_dim,
            self.head_dim,
        );
    }

    /// Inverse rotation: out = V @ x.
    ///
    /// Uses SIMD-accelerated row dot-products via `simd_matmul_rows` for
    /// ~4-8× speedup over scalar on NEON/AVX2 targets.
    pub fn unrotate(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.head_dim);
        assert_eq!(out.len(), self.head_dim);
        katgpt_core::simd::simd_matmul_rows(out, &self.eigenvectors, x, self.head_dim, self.head_dim);
    }
}

/// Deterministic random orthogonal rotation (TurboQuant baseline/fallback).
///
/// Uses the same QR decomposition as `generate_rotation_matrix` but wraps
/// it in a struct for interface consistency with `SpectralRotation`.
#[cfg(feature = "turboquant")]
pub struct RandomRotation {
    /// Per-(layer, head) rotation matrices, flattened.
    /// Indexed as rotations[layer * n_heads + head].
    rotations: Vec<f32>,
    head_dim: usize,
    n_layers: usize,
    n_heads: usize,
}

#[cfg(feature = "turboquant")]
impl RandomRotation {
    /// Generate all per-(layer, head) rotation matrices upfront.
    pub fn new(head_dim: usize, n_layers: usize, n_heads: usize, global_seed: u64) -> Self {
        let total = n_layers * n_heads;
        let mut rotations = Vec::with_capacity(total * head_dim * head_dim);
        for idx in 0..total {
            let seed = global_seed.wrapping_add(idx as u64 * 7919);
            let mat = generate_rotation_matrix(head_dim, seed);
            rotations.extend_from_slice(&mat);
        }
        Self {
            rotations,
            head_dim,
            n_layers,
            n_heads,
        }
    }

    /// Forward rotation for a specific (layer, head).
    #[allow(clippy::needless_range_loop)]
    pub fn rotate(&self, x: &[f32], layer_idx: usize, head_idx: usize, out: &mut [f32]) {
        let mat = self.get_matrix(layer_idx, head_idx);
        // out = mat^T @ x (same convention as SpectralRotation)
        for j in 0..self.head_dim {
            let mut sum = 0.0f32;
            for i in 0..self.head_dim {
                sum += mat[i * self.head_dim + j] * x[i];
            }
            out[j] = sum;
        }
    }

    /// Inverse rotation for a specific (layer, head).
    #[allow(clippy::needless_range_loop)]
    pub fn unrotate(&self, x: &[f32], layer_idx: usize, head_idx: usize, out: &mut [f32]) {
        let mat = self.get_matrix(layer_idx, head_idx);
        // out = mat @ x
        for i in 0..self.head_dim {
            let mut sum = 0.0f32;
            for j in 0..self.head_dim {
                sum += mat[i * self.head_dim + j] * x[j];
            }
            out[i] = sum;
        }
    }

    fn get_matrix(&self, layer_idx: usize, head_idx: usize) -> &[f32] {
        assert!(
            layer_idx < self.n_layers,
            "layer_idx {layer_idx} >= n_layers {}",
            self.n_layers
        );
        assert!(
            head_idx < self.n_heads,
            "head_idx {head_idx} >= n_heads {}",
            self.n_heads
        );
        let offset = (layer_idx * self.n_heads + head_idx) * self.head_dim * self.head_dim;
        &self.rotations[offset..offset + self.head_dim * self.head_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_rotation_roundtrip() {
        let dim = 8;
        // Create a simple orthogonal matrix (identity)
        let mut eigen = vec![0.0f32; dim * dim];
        for i in 0..dim {
            eigen[i * dim + i] = 1.0;
        }
        let rot = SpectralRotation::new(eigen, dim);

        let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let mut rotated = vec![0.0; dim];
        let mut recovered = vec![0.0; dim];

        rot.rotate(&x, &mut rotated);
        rot.unrotate(&rotated, &mut recovered);

        for (i, (orig, rec)) in x.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-4,
                "roundtrip failed at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_spectral_rotation_with_known_matrix() {
        let dim = 2;
        // 2D rotation by 45°: V = [[cos45, -sin45], [sin45, cos45]]
        let c = std::f32::consts::FRAC_PI_4.cos();
        let s = std::f32::consts::FRAC_PI_4.sin();
        let eigen = vec![c, -s, s, c];
        let rot = SpectralRotation::new(eigen, dim);

        let x = vec![1.0, 0.0];
        let mut rotated = vec![0.0; dim];
        rot.rotate(&x, &mut rotated);

        // V^T @ [1,0] = [cos45, -sin45]
        assert!(
            (rotated[0] - c).abs() < 1e-4,
            "rotated[0] should be cos45, got {}",
            rotated[0]
        );
        assert!(
            (rotated[1] - (-s)).abs() < 1e-4,
            "rotated[1] should be -sin45, got {}",
            rotated[1]
        );
    }

    #[cfg(feature = "turboquant")]
    #[test]
    fn test_random_rotation_roundtrip() {
        let dim = 16;
        let rot = RandomRotation::new(dim, 2, 4, 42);

        let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).cos()).collect();
        let mut rotated = vec![0.0; dim];
        let mut recovered = vec![0.0; dim];

        rot.rotate(&x, 0, 0, &mut rotated);
        rot.unrotate(&rotated, 0, 0, &mut recovered);

        for (i, (orig, rec)) in x.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 0.1,
                "roundtrip failed at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[cfg(feature = "turboquant")]
    #[test]
    fn test_random_rotation_preserves_norm() {
        let dim = 16;
        let rot = RandomRotation::new(dim, 1, 1, 77);

        let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut rotated = vec![0.0; dim];
        rot.rotate(&x, 0, 0, &mut rotated);
        let rotated_norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (norm - rotated_norm).abs() < 0.1,
            "norm changed: {norm} -> {rotated_norm}"
        );
    }
}
