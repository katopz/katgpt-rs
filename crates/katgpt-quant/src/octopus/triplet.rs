//! Triplet decomposition for OCTOPUS KV cache compression.
//!
//! After random rotation, coordinate triplets t_i ∈ R³ are decomposed into:
//! - **Norm** ρ_i = ||t_i||₂ — has Beta(3/2, (d-3)/2) marginal
//! - **Direction** n_i = t_i/ρ_i ∈ S² — encoded via octahedral map
//!
//! The triplet norm concentrates near √(3/d) as d grows, meaning
//! direction errors dominate — motivating the non-uniform (b+1, b-1) bit split.

/// A single triplet: norm + unit direction on S².
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Triplet {
    /// L2 norm of the triplet vector (ρ).
    pub norm: f32,
    /// Unit direction vector on S² (x, y, z).
    pub dir: [f32; 3],
}

impl Triplet {
    /// Create a zero triplet (norm=0, direction along z-axis).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            norm: 0.0,
            dir: [0.0, 0.0, 1.0],
        }
    }

    /// Create a triplet from a 3-element slice.
    ///
    /// Returns `None` if the slice has zero norm (degenerate triplet).
    pub fn from_slice(v: &[f32]) -> Option<Self> {
        debug_assert!(v.len() == 3, "triplet slice must have exactly 3 elements");
        let x = v[0];
        let y = v[1];
        let z = v[2];
        let norm = (x * x + y * y + z * z).sqrt();
        if norm < 1e-10 {
            return None;
        }
        Some(Self {
            norm,
            dir: [x / norm, y / norm, z / norm],
        })
    }

    /// Reconstruct the 3-element vector from norm and direction.
    #[must_use]
    pub fn to_vec(&self) -> [f32; 3] {
        [
            self.norm * self.dir[0],
            self.norm * self.dir[1],
            self.norm * self.dir[2],
        ]
    }
}

/// Decompose a d-dimensional vector into ⌈d/3⌉ triplets.
///
/// If d is not divisible by 3, the last triplet is zero-padded.
/// Zero-norm blocks produce `Triplet::zero()` entries.
///
/// # Panics
/// Panics if `vec` is empty.
#[must_use]
pub fn decompose(vec: &[f32]) -> Vec<Triplet> {
    assert!(!vec.is_empty(), "cannot decompose empty vector");

    let d = vec.len();
    let n_triplets = d.div_ceil(3);
    let mut triplets = Vec::with_capacity(n_triplets);

    for i in 0..n_triplets {
        let base = i * 3;
        let x = vec.get(base).copied().unwrap_or(0.0);
        let y = vec.get(base + 1).copied().unwrap_or(0.0);
        let z = vec.get(base + 2).copied().unwrap_or(0.0);

        let norm = (x * x + y * y + z * z).sqrt();
        if norm < 1e-10 {
            triplets.push(Triplet::zero());
        } else {
            triplets.push(Triplet {
                norm,
                dir: [x / norm, y / norm, z / norm],
            });
        }
    }

    triplets
}

/// Decompose a d-dimensional vector into a pre-allocated buffer (zero-alloc hot path).
///
/// Clears and fills `out` with ⌈d/3⌉ triplets.
/// Equivalent to [`decompose`] but avoids allocation.
///
/// # Panics
/// Panics if `vec` is empty.
pub fn decompose_into(vec: &[f32], out: &mut Vec<Triplet>) {
    assert!(!vec.is_empty(), "cannot decompose empty vector");

    let d = vec.len();
    let n = d.div_ceil(3);
    out.clear();
    out.reserve(n);

    for i in 0..n {
        let base = i * 3;
        let x = vec.get(base).copied().unwrap_or(0.0);
        let y = vec.get(base + 1).copied().unwrap_or(0.0);
        let z = vec.get(base + 2).copied().unwrap_or(0.0);

        let norm = (x * x + y * y + z * z).sqrt();
        if norm < 1e-10 {
            out.push(Triplet::zero());
        } else {
            out.push(Triplet {
                norm,
                dir: [x / norm, y / norm, z / norm],
            });
        }
    }
}

/// Recompose triplets back into a d-dimensional vector.
///
/// The output vector has length `triplets.len() * 3`, which may be
/// longer than the original d if zero-padding was applied. The caller
/// should truncate to the original dimension.
#[must_use]
pub fn recompose(triplets: &[Triplet]) -> Vec<f32> {
    let d = triplets.len() * 3;
    let mut vec = vec![0.0f32; d];

    for (i, t) in triplets.iter().enumerate() {
        let base = i * 3;
        vec[base] = t.norm * t.dir[0];
        vec[base + 1] = t.norm * t.dir[1];
        vec[base + 2] = t.norm * t.dir[2];
    }

    vec
}

/// Recompose triplets into a pre-allocated buffer (zero-alloc hot path).
///
/// Writes `min(triplets.len() * 3, out.len())` elements.
/// Remaining elements (if out is larger) are zeroed.
///
/// # Panics
/// Panics if `out` is shorter than `triplets.len() * 3`.
pub fn recompose_into(triplets: &[Triplet], out: &mut [f32]) {
    let expected_len = triplets.len() * 3;
    assert!(
        out.len() >= expected_len,
        "output buffer too short: got {}, need {expected_len}",
        out.len()
    );

    for (i, t) in triplets.iter().enumerate() {
        let base = i * 3;
        out[base] = t.norm * t.dir[0];
        out[base + 1] = t.norm * t.dir[1];
        out[base + 2] = t.norm * t.dir[2];
    }
}

/// Number of triplets for a given dimension d.
///
/// Equivalent to `⌈d/3⌉`.
#[must_use]
#[inline]
pub fn n_triplets(d: usize) -> usize {
    d.div_ceil(3)
}

/// Count how many triplets have non-zero norm.
#[must_use]
pub fn count_nonzero(triplets: &[Triplet]) -> usize {
    triplets.iter().filter(|t| t.norm >= 1e-10).count()
}

/// Compute the sum of squared norms across all triplets.
///
/// This equals ||v||² for the original vector v.
#[must_use]
pub fn sum_squared_norms(triplets: &[Triplet]) -> f32 {
    triplets.iter().map(|t| t.norm * t.norm).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_recompose_roundtrip() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let triplets = decompose(&v);
        let reconstructed = recompose(&triplets);
        assert_eq!(reconstructed.len(), 6);
        for i in 0..6 {
            assert!(
                (reconstructed[i] - v[i]).abs() < 1e-6,
                "mismatch at [{i}]: got {}, expected {}",
                reconstructed[i],
                v[i]
            );
        }
    }

    #[test]
    fn test_decompose_recompose_dim128() {
        let d = 128;
        let v: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
        let triplets = decompose(&v);
        assert_eq!(triplets.len(), n_triplets(d)); // ⌈128/3⌉ = 43

        let reconstructed = recompose(&triplets);
        // Recomposed vector is 43*3 = 129 elements, first 128 match original
        for i in 0..d {
            assert!(
                (reconstructed[i] - v[i]).abs() < 1e-5,
                "mismatch at [{i}]: got {}, expected {}",
                reconstructed[i],
                v[i]
            );
        }
        // 129th element should be 0 (zero-pad)
        assert!((reconstructed[128]).abs() < 1e-10);
    }

    #[test]
    fn test_decompose_zero_padding() {
        // d=5: last triplet is [v[3], v[4], 0.0]
        let v: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let triplets = decompose(&v);
        assert_eq!(triplets.len(), 2);

        // First triplet: [1, 0, 0] → norm=1, dir=[1,0,0]
        assert!((triplets[0].norm - 1.0).abs() < 1e-6);
        assert!((triplets[0].dir[0] - 1.0).abs() < 1e-6);

        // Second triplet: [0, 0, 0] → zero triplet
        assert!(triplets[1].norm < 1e-10);
    }

    #[test]
    fn test_decompose_dim3() {
        let v: Vec<f32> = vec![3.0, 4.0, 0.0];
        let triplets = decompose(&v);
        assert_eq!(triplets.len(), 1);
        assert!((triplets[0].norm - 5.0).abs() < 1e-5);
        assert!((triplets[0].dir[0] - 0.6).abs() < 1e-5);
        assert!((triplets[0].dir[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_decompose_unit_vector() {
        let v: Vec<f32> = vec![1.0, 0.0, 0.0];
        let triplets = decompose(&v);
        assert_eq!(triplets.len(), 1);
        assert!((triplets[0].norm - 1.0).abs() < 1e-6);
        assert!((triplets[0].dir[0] - 1.0).abs() < 1e-6);
        assert!(triplets[0].dir[1].abs() < 1e-6);
        assert!(triplets[0].dir[2].abs() < 1e-6);
    }

    #[test]
    fn test_recompose_into() {
        let triplets = decompose(&[1.0, 2.0, 3.0]);
        let mut out = vec![0.0f32; 3];
        recompose_into(&triplets, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_recompose_into_buffer_too_short() {
        let triplets = decompose(&[1.0, 2.0, 3.0]);
        let mut out = vec![0.0f32; 2]; // too short
        recompose_into(&triplets, &mut out);
    }

    #[test]
    fn test_triplet_zero() {
        let t = Triplet::zero();
        assert_eq!(t.norm, 0.0);
        let v = t.to_vec();
        // Zero norm → zero vector regardless of direction
        assert!(v[0].abs() < 1e-10);
        assert!(v[1].abs() < 1e-10);
        assert!(v[2].abs() < 1e-10);
    }

    #[test]
    fn test_triplet_from_slice() {
        let t = Triplet::from_slice(&[3.0, 4.0, 0.0]).expect("non-zero");
        assert!((t.norm - 5.0).abs() < 1e-5);
        assert!((t.dir[0] - 0.6).abs() < 1e-5);
        assert!((t.dir[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_from_zero_slice() {
        assert!(Triplet::from_slice(&[0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_n_triplets() {
        assert_eq!(n_triplets(3), 1);
        assert_eq!(n_triplets(4), 2);
        assert_eq!(n_triplets(6), 2);
        assert_eq!(n_triplets(7), 3);
        assert_eq!(n_triplets(128), 43);
        assert_eq!(n_triplets(64), 22);
        assert_eq!(n_triplets(256), 86);
    }

    #[test]
    fn test_count_nonzero() {
        let v: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let triplets = decompose(&v);
        assert_eq!(count_nonzero(&triplets), 1);
    }

    #[test]
    fn test_sum_squared_norms() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected_sq_norm: f32 = v.iter().map(|x| x * x).sum();
        let triplets = decompose(&v);
        let actual = sum_squared_norms(&triplets);
        assert!(
            (actual - expected_sq_norm).abs() < 1e-4,
            "sum squared norms: got {actual}, expected {expected_sq_norm}"
        );
    }

    #[test]
    fn test_direction_is_unit() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
        let triplets = decompose(&v);
        for (i, t) in triplets.iter().enumerate() {
            if t.norm >= 1e-10 {
                let dir_norm =
                    (t.dir[0] * t.dir[0] + t.dir[1] * t.dir[1] + t.dir[2] * t.dir[2]).sqrt();
                assert!(
                    (dir_norm - 1.0).abs() < 1e-5,
                    "triplet {i} direction not unit: norm={dir_norm}"
                );
            }
        }
    }

    #[test]
    fn test_norm_preservation() {
        // Decomposing and recomposing should preserve total squared norm
        let d = 128;
        let v: Vec<f32> = (0..d).map(|i| (i as f32 * 0.13).cos()).collect();
        let original_sq: f32 = v.iter().map(|x| x * x).sum();

        let triplets = decompose(&v);
        let reconstructed = recompose(&triplets);
        let recon_sq: f32 = reconstructed[..d].iter().map(|x| x * x).sum();

        assert!(
            (recon_sq - original_sq).abs() < 1e-3,
            "norm not preserved: original={original_sq}, recon={recon_sq}"
        );
    }
}
