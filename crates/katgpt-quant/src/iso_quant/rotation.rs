//! Quaternion 4D rotation for IsoQuant KV cache.
//!
//! Hamilton product: q = (w, x, y, z), 16 FMAs per multiply.
//! Full rotation: q_L * v * conj(q_R) = 2 Hamilton products = 32 FMAs per group.
//! Fast rotation:  q_L * v = 1 Hamilton product = 16 FMAs per group.
//!
//! For d=128: Full = 32 groups × 32 FMAs = 1024 FMAs.
//!            Fast = 32 groups × 16 FMAs = 512 FMAs.
//! vs TurboQuant's 16,384 FMAs.

use katgpt_core::types::Rng;

/// Quaternion multiply (Hamilton product): 16 FMAs.
/// a, b: (w, x, y, z).
#[inline]
pub fn quat_multiply(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z).
#[inline]
pub fn quat_conjugate(q: &[f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Generate random unit quaternions via normalized Gaussian.
/// n quaternions, deterministic from seed.
pub fn generate_unit_quaternions(n: usize, seed: u64) -> Vec<[f32; 4]> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let w = rng.normal();
            let x = rng.normal();
            let y = rng.normal();
            let z = rng.normal();
            let norm = (w * w + x * x + y * y + z * z).sqrt().max(1e-8);
            [w / norm, x / norm, y / norm, z / norm]
        })
        .collect()
}

/// Apply full quaternion sandwich rotation: q_L * v * conj(q_R).
/// For 4D: treat the 4D vector as a quaternion directly.
#[inline]
pub fn quat_sandwich_forward(q_l: &[f32; 4], v: &[f32; 4], q_r: &[f32; 4]) -> [f32; 4] {
    let temp = quat_multiply(q_l, v);
    quat_multiply(&temp, &quat_conjugate(q_r))
}

/// Apply inverse quaternion sandwich: conj(q_L) * v * q_R.
#[inline]
pub fn quat_sandwich_inverse(q_l: &[f32; 4], v: &[f32; 4], q_r: &[f32; 4]) -> [f32; 4] {
    let temp = quat_multiply(&quat_conjugate(q_l), v);
    quat_multiply(&temp, q_r)
}

/// Apply fast (left-only) forward rotation: q_L * v.
#[inline]
pub fn quat_left_forward(q_l: &[f32; 4], v: &[f32; 4]) -> [f32; 4] {
    quat_multiply(q_l, v)
}

/// Apply fast (left-only) inverse rotation: conj(q_L) * v.
#[inline]
pub fn quat_left_inverse(q_l: &[f32; 4], v: &[f32; 4]) -> [f32; 4] {
    quat_multiply(&quat_conjugate(q_l), v)
}

/// Apply full vector quaternion rotation (all groups).
///
/// `q_left` has ceil(dim/4) entries.
/// For full mode, `q_right` must be Some with same length.
/// `input` has `dim` elements, `output` must have `dim` elements.
#[inline]
pub fn apply_rotation(
    q_left: &[[f32; 4]],
    q_right: Option<&[[f32; 4]]>,
    input: &[f32],
    output: &mut [f32],
) {
    let dim = input.len();
    debug_assert_eq!(output.len(), dim);
    let _n_groups = q_left.len();
    let n_full_groups = dim / 4;

    match q_right {
        Some(qr) => {
            // Main loop: no bounds checks for full groups
            for g in 0..n_full_groups {
                let base = g * 4;
                let v = [
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                ];
                let r = quat_sandwich_forward(&q_left[g], &v, &qr[g]);
                output[base] = r[0];
                output[base + 1] = r[1];
                output[base + 2] = r[2];
                output[base + 3] = r[3];
            }
            // Tail: partial last group (zero-padded)
            if !dim.is_multiple_of(4) {
                let g = n_full_groups;
                let base = g * 4;
                let n = dim - base;
                let mut v = [0.0f32; 4];
                v[..n].copy_from_slice(&input[base..base + n]);
                let r = quat_sandwich_forward(&q_left[g], &v, &qr[g]);
                output[base..base + n].copy_from_slice(&r[..n]);
            }
        }
        None => {
            // stride math: `g` indexes both q_left[g] and base = g*4 into input/output
            #[allow(clippy::needless_range_loop)]
            for g in 0..n_full_groups {
                let base = g * 4;
                let v = [
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                ];
                let r = quat_left_forward(&q_left[g], &v);
                output[base] = r[0];
                output[base + 1] = r[1];
                output[base + 2] = r[2];
                output[base + 3] = r[3];
            }
            if !dim.is_multiple_of(4) {
                let g = n_full_groups;
                let base = g * 4;
                let n = dim - base;
                let mut v = [0.0f32; 4];
                v[..n].copy_from_slice(&input[base..base + n]);
                let r = quat_left_forward(&q_left[g], &v);
                output[base..base + n].copy_from_slice(&r[..n]);
            }
        }
    }
}

/// Apply full vector inverse quaternion rotation.
#[inline]
pub fn apply_inverse_rotation(
    q_left: &[[f32; 4]],
    q_right: Option<&[[f32; 4]]>,
    input: &[f32],
    output: &mut [f32],
) {
    let dim = input.len();
    debug_assert_eq!(output.len(), dim);
    let _n_groups = q_left.len();
    let n_full_groups = dim / 4;

    match q_right {
        Some(qr) => {
            for g in 0..n_full_groups {
                let base = g * 4;
                let v = [
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                ];
                let r = quat_sandwich_inverse(&q_left[g], &v, &qr[g]);
                output[base] = r[0];
                output[base + 1] = r[1];
                output[base + 2] = r[2];
                output[base + 3] = r[3];
            }
            if !dim.is_multiple_of(4) {
                let g = n_full_groups;
                let base = g * 4;
                let n = dim - base;
                let mut v = [0.0f32; 4];
                v[..n].copy_from_slice(&input[base..base + n]);
                let r = quat_sandwich_inverse(&q_left[g], &v, &qr[g]);
                output[base..base + n].copy_from_slice(&r[..n]);
            }
        }
        None => {
            // stride math: `g` indexes both q_left[g] and base = g*4 into input/output
            #[allow(clippy::needless_range_loop)]
            for g in 0..n_full_groups {
                let base = g * 4;
                let v = [
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                ];
                let r = quat_left_inverse(&q_left[g], &v);
                output[base] = r[0];
                output[base + 1] = r[1];
                output[base + 2] = r[2];
                output[base + 3] = r[3];
            }
            if !dim.is_multiple_of(4) {
                let g = n_full_groups;
                let base = g * 4;
                let n = dim - base;
                let mut v = [0.0f32; 4];
                v[..n].copy_from_slice(&input[base..base + n]);
                let r = quat_left_inverse(&q_left[g], &v);
                output[base..base + n].copy_from_slice(&r[..n]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_multiply_identity() {
        let identity = [1.0f32, 0.0, 0.0, 0.0];
        let q = [0.5f32, 0.5, 0.5, 0.5]; // unit quaternion
        let result = quat_multiply(&identity, &q);
        for i in 0..4 {
            assert!(
                (result[i] - q[i]).abs() < 1e-5,
                "identity mul failed at {i}"
            );
        }
    }

    #[test]
    fn test_quat_conjugate() {
        let q = [1.0f32, 2.0, 3.0, 4.0];
        let c = quat_conjugate(&q);
        assert_eq!(c, [1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_unit_quaternion_norm() {
        let quats = generate_unit_quaternions(100, 42);
        for (i, q) in quats.iter().enumerate() {
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "quat {i} norm = {norm}");
        }
    }

    #[test]
    fn test_full_sandwich_roundtrip() {
        let q_l = [0.5f32, 0.5, 0.5, 0.5];
        let q_r = [
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
        ];
        let v = [1.0f32, 2.0, 3.0, 4.0];

        let rotated = quat_sandwich_forward(&q_l, &v, &q_r);
        let recovered = quat_sandwich_inverse(&q_l, &rotated, &q_r);

        for i in 0..4 {
            assert!(
                (v[i] - recovered[i]).abs() < 1e-4,
                "roundtrip failed at [{i}]: {} vs {}",
                v[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_fast_roundtrip() {
        let q_l = [0.5f32, 0.5, 0.5, 0.5];
        let v = [1.0f32, 2.0, 3.0, 4.0];

        let rotated = quat_left_forward(&q_l, &v);
        let recovered = quat_left_inverse(&q_l, &rotated);

        for i in 0..4 {
            assert!(
                (v[i] - recovered[i]).abs() < 1e-4,
                "roundtrip failed at [{i}]: {} vs {}",
                v[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_full_vector_rotation_roundtrip() {
        let dim: usize = 128;
        let n_groups = dim.div_ceil(4);
        let q_left = generate_unit_quaternions(n_groups, 42);
        let q_right = generate_unit_quaternions(n_groups, 43);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut rotated = vec![0.0f32; dim];
        let mut recovered = vec![0.0f32; dim];

        apply_rotation(&q_left, Some(&q_right), &input, &mut rotated);
        apply_inverse_rotation(&q_left, Some(&q_right), &rotated, &mut recovered);

        for i in 0..dim {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-3,
                "roundtrip failed at [{i}]: {} vs {}",
                input[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_fast_vector_rotation_roundtrip() {
        let dim: usize = 64;
        let n_groups = dim.div_ceil(4);
        let q_left = generate_unit_quaternions(n_groups, 99);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();
        let mut rotated = vec![0.0f32; dim];
        let mut recovered = vec![0.0f32; dim];

        apply_rotation(&q_left, None, &input, &mut rotated);
        apply_inverse_rotation(&q_left, None, &rotated, &mut recovered);

        for i in 0..dim {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-3,
                "roundtrip failed at [{i}]: {} vs {}",
                input[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let dim: usize = 64;
        let n_groups = dim.div_ceil(4);
        let q_left = generate_unit_quaternions(n_groups, 77);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let mut rotated = vec![0.0f32; dim];

        apply_rotation(&q_left, None, &input, &mut rotated);

        let orig_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rot_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (orig_norm - rot_norm).abs() / orig_norm < 0.01,
            "vector norm changed: {orig_norm} vs {rot_norm}"
        );
    }

    #[test]
    fn test_generation_deterministic() {
        let q1 = generate_unit_quaternions(32, 42);
        let q2 = generate_unit_quaternions(32, 42);
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_different_seeds_differ() {
        let q1 = generate_unit_quaternions(32, 42);
        let q2 = generate_unit_quaternions(32, 43);
        assert_ne!(q1, q2);
    }

    #[test]
    fn test_non_multiple_of_4() {
        // dim = 10 → 3 groups (12 slots, last 2 zero-padded).
        //
        // Convention (matches `planar_quant::rotation::test_odd_dim_roundtrip`):
        // when the vector length is not a multiple of the group size, the CALLER
        // is responsible for padding the buffer up to the next multiple of the
        // group size. The rotation functions then treat the padded group as a
        // normal full group and the roundtrip is exact, including the padded
        // zeros. Attempting to roundtrip on an unpadded buffer is lossy by
        // construction — the forward rotation mixes the trailing real elements
        // into the (discarded) pad slots, so the inverse cannot recover them
        // from the truncated buffer.
        let dim: usize = 10;
        let padded = (dim + 3) & !3; // next multiple of 4 = 12
        assert_eq!(padded, 12);
        let n_groups = padded / 4;
        assert_eq!(n_groups, 3);

        let q_left = generate_unit_quaternions(n_groups, 55);
        let q_right = generate_unit_quaternions(n_groups, 56);

        let mut input = vec![0.0f32; padded];
        input[..dim].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let mut rotated = vec![0.0f32; padded];
        let mut recovered = vec![0.0f32; padded];

        apply_rotation(&q_left, Some(&q_right), &input, &mut rotated);
        apply_inverse_rotation(&q_left, Some(&q_right), &rotated, &mut recovered);

        // Real elements (indices 0..dim) must roundtrip to full f32 precision.
        for i in 0..dim {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-4,
                "roundtrip failed at [{i}]: {} vs {}",
                input[i],
                recovered[i]
            );
        }
        // Padded tail (indices dim..padded) started as zero and must remain
        // zero after the roundtrip — this is what makes the padded-buffer
        // convention invertible.
        for i in dim..padded {
            assert!(
                recovered[i].abs() < 1e-4,
                "padded tail not preserved at [{i}]: {}",
                recovered[i]
            );
        }
    }
}
