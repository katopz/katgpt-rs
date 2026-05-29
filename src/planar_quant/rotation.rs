//! 2D Givens rotation for PlanarQuant KV cache.
//!
//! Each adjacent pair of elements is rotated by an independent angle θ.
//! Forward:  (cos θ · v0 - sin θ · v1, sin θ · v0 + cos θ · v1)
//! Inverse:  (cos θ · v0 + sin θ · v1, -sin θ · v0 + cos θ · v1)
//!
//! Cost: 4 FMAs per pair, 2·n_groups FMAs for full vector.
//! For d=128: 256 FMAs vs TurboQuant's 16,384.

/// Apply 2D rotation to a pair of values.
#[inline]
pub fn rot2_apply(cos_sin: &[f32; 2], v0: f32, v1: f32) -> (f32, f32) {
    let (c, s) = (cos_sin[0], cos_sin[1]);
    (c * v0 - s * v1, s * v0 + c * v1)
}

/// Apply inverse 2D rotation (transpose = negate sin).
#[inline]
pub fn rot2_inverse(cos_sin: &[f32; 2], v0: f32, v1: f32) -> (f32, f32) {
    let (c, s) = (cos_sin[0], cos_sin[1]);
    (c * v0 + s * v1, -s * v0 + c * v1)
}

/// Generate random 2D rotation parameters (cos θ, sin θ) per group.
///
/// n_groups = ceil(kv_dim / 2). Each group gets an independent random angle.
/// Deterministic from seed.
pub fn generate_givens_rotations(n_groups: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut rng = crate::types::Rng::new(seed);
    (0..n_groups)
        .map(|_| {
            let angle = rng.uniform() * std::f32::consts::TAU;
            [angle.cos(), angle.sin()]
        })
        .collect()
}

/// Apply full vector 2D rotation: rotate all pairs.
///
/// `rotations` has ceil(dim/2) entries.
/// `input` has `dim` elements.
/// `output` must have `dim` elements (pre-allocated).
#[inline]
pub fn apply_rotation(rotations: &[[f32; 2]], input: &[f32], output: &mut [f32]) {
    let n_groups = rotations.len();
    let dim = input.len();
    debug_assert_eq!(output.len(), dim);
    let full_groups = dim / 2;
    // Fast path: full pairs (no bounds checks needed)
    for g in 0..full_groups {
        let (r0, r1) = rot2_apply(&rotations[g], input[g * 2], input[g * 2 + 1]);
        output[g * 2] = r0;
        output[g * 2 + 1] = r1;
    }
    // Odd element: last group with only one element
    if !dim.is_multiple_of(2) && full_groups < n_groups {
        let (r0, _) = rot2_apply(&rotations[full_groups], input[full_groups * 2], 0.0);
        output[full_groups * 2] = r0;
    }
}

/// Apply full vector inverse 2D rotation: inverse-rotate all pairs.
#[inline]
pub fn apply_inverse_rotation(rotations: &[[f32; 2]], input: &[f32], output: &mut [f32]) {
    let n_groups = rotations.len();
    let dim = input.len();
    debug_assert_eq!(output.len(), dim);
    let full_groups = dim / 2;
    // Fast path: full pairs (no bounds checks needed)
    for g in 0..full_groups {
        let (r0, r1) = rot2_inverse(&rotations[g], input[g * 2], input[g * 2 + 1]);
        output[g * 2] = r0;
        output[g * 2 + 1] = r1;
    }
    // Odd element: last group with only one element
    if !dim.is_multiple_of(2) && full_groups < n_groups {
        let (r0, _) = rot2_inverse(&rotations[full_groups], input[full_groups * 2], 0.0);
        output[full_groups * 2] = r0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rot2_preserves_norm() {
        let cs = [0.6f32, 0.8f32]; // cos, sin (norm = 1)
        let v0 = 3.0f32;
        let v1 = 4.0f32;
        let (r0, r1) = rot2_apply(&cs, v0, v1);
        let orig_norm = (v0 * v0 + v1 * v1).sqrt();
        let rot_norm = (r0 * r0 + r1 * r1).sqrt();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-5,
            "norm not preserved: {orig_norm} vs {rot_norm}"
        );
    }

    #[test]
    fn test_rot2_roundtrip() {
        let cs = [0.6f32, 0.8f32];
        let v0 = 2.5f32;
        let v1 = -1.3f32;
        let (r0, r1) = rot2_apply(&cs, v0, v1);
        let (o0, o1) = rot2_inverse(&cs, r0, r1);
        assert!((o0 - v0).abs() < 1e-5, "v0 roundtrip failed: {o0} vs {v0}");
        assert!((o1 - v1).abs() < 1e-5, "v1 roundtrip failed: {o1} vs {v1}");
    }

    #[test]
    fn test_full_rotation_roundtrip() {
        let dim: usize = 128;
        let n_groups = dim.div_ceil(2);
        let rots = generate_givens_rotations(n_groups, 42);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut rotated = vec![0.0f32; dim];
        let mut recovered = vec![0.0f32; dim];

        apply_rotation(&rots, &input, &mut rotated);
        apply_inverse_rotation(&rots, &rotated, &mut recovered);

        for i in 0..dim {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-5,
                "roundtrip failed at [{i}]: {} vs {}",
                input[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_rotation_preserves_vector_norm() {
        let dim: usize = 64;
        let n_groups = dim.div_ceil(2);
        let rots = generate_givens_rotations(n_groups, 77);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let mut rotated = vec![0.0f32; dim];

        apply_rotation(&rots, &input, &mut rotated);

        let orig_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rot_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-3,
            "vector norm changed: {orig_norm} vs {rot_norm}"
        );
    }

    #[test]
    fn test_generation_deterministic() {
        let r1 = generate_givens_rotations(32, 42);
        let r2 = generate_givens_rotations(32, 42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_rotation_unit_circle() {
        // Verify cos²θ + sin²θ = 1 for each generated rotation
        let rots = generate_givens_rotations(64, 123);
        for (i, &[c, s]) in rots.iter().enumerate() {
            let norm_sq = c * c + s * s;
            assert!(
                (norm_sq - 1.0).abs() < 1e-6,
                "rotation [{i}] not unit: cos²+sin² = {norm_sq}"
            );
        }
    }

    #[test]
    fn test_odd_dim_roundtrip() {
        // Odd dimension: last group has only one element (padded with 0).
        // Buffers must be padded to even length for full pair rotation.
        let dim = 7;
        let padded = (dim + 1) & !1; // 8
        let n_groups = padded / 2;
        let rots = generate_givens_rotations(n_groups, 55);
        let mut input = vec![0.0f32; padded];
        input[..dim].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let mut rotated = vec![0.0f32; padded];
        let mut recovered = vec![0.0f32; padded];

        apply_rotation(&rots, &input, &mut rotated);
        apply_inverse_rotation(&rots, &rotated, &mut recovered);

        for i in 0..dim {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-5,
                "odd dim roundtrip failed at [{i}]: {} vs {}",
                input[i],
                recovered[i]
            );
        }
    }
}
