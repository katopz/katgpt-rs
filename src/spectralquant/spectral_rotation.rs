//! Rotation transforms for SpectralQuant — spectral (calibrated).
//!
//! Spectral rotation uses learned eigenvectors from covariance analysis.
//! `RandomRotation` (TurboQuant baseline/fallback) is gated behind the `turboquant` feature.

/// Data-driven orthogonal rotation using calibrated eigenvectors.
///
/// Forward: x_hat = V^T @ x  (project into spectral basis)
/// Inverse: x = V @ x_hat   (reconstruct original basis)
pub struct SpectralRotation {
    eigenvectors: Vec<f32>, // (head_dim × head_dim), row-major
    head_dim: usize,
}

impl SpectralRotation {
    pub fn new(eigenvectors: Vec<f32>, head_dim: usize) -> Self {
        assert_eq!(
            eigenvectors.len(),
            head_dim * head_dim,
            "eigenvectors must be head_dim × head_dim"
        );
        Self {
            eigenvectors,
            head_dim,
        }
    }

    /// Forward rotation: out = V^T @ x.
    /// V is stored row-major, V^T[j][i] = V[i][j].
    /// out[j] = Σ_i V[i][j] * x[i]
    ///
    /// Implemented as transpose-and-accumulate: iterate over rows of V,
    /// scaling each row by x[i] and accumulating into `out`. This gives
    /// contiguous reads from V and contiguous writes to out, which is
    /// SIMD-friendly and cache-friendly.
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.head_dim);
        assert_eq!(out.len(), self.head_dim);
        out.fill(0.0);
        let hd = self.head_dim;
        for i in 0..hd {
            let xi = x[i];
            let row = &self.eigenvectors[i * hd..i * hd + hd];
            // out[j] += row[j] * xi — contiguous for both read and write
            for j in 0..hd {
                unsafe {
                    *out.get_unchecked_mut(j) += *row.get_unchecked(j) * xi;
                }
            }
        }
    }

    /// Inverse rotation: out = V @ x.
    /// out[i] = Σ_j V[i][j] * x[j]
    ///
    /// Uses SIMD-accelerated row dot-products via `simd_matmul_rows` for
    /// ~4-8× speedup over scalar on NEON/AVX2 targets.
    pub fn unrotate(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.head_dim);
        assert_eq!(out.len(), self.head_dim);
        crate::simd::simd_matmul_rows(
            out,
            &self.eigenvectors,
            x,
            self.head_dim,
            self.head_dim,
        );
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
            let mat = crate::turboquant::rotation::generate_rotation_matrix(head_dim, seed);
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
