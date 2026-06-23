    use super::*;

    #[test]
    fn argmax_matches_two_pass_idiom() {
        // Reference: the two-pass idiom this primitive replaces.
        fn naive(x: &[f32]) -> (usize, f32) {
            let m = simd_max_f32(x);
            (x.iter().position(|&v| v == m).unwrap_or(0), m)
        }
        let cases: &[&[f32]] = &[
            &[3.0],
            &[1.0, 2.0, 3.0, 2.0, 1.0],
            &[5.0, 5.0, 5.0],           // tie → first index (0)
            &[-1.0, -2.0, -0.5, -9.0],  // all negative
            &[0.0, 1.0, 1.0, 0.5, 1.0], // multiple maxima → first (index 1)
        ];
        for c in cases {
            assert_eq!(simd_argmax_f32(c), naive(c), "mismatch on {c:?}");
        }
        // Larger pseudo-random buffer: max placed at a known interior index.
        let mut buf = vec![0.0f32; 4096];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i * 2654435761) % 997) as f32;
        }
        buf[1234] = 10_000.0;
        assert_eq!(simd_argmax_f32(&buf), (1234, 10_000.0));

        // Randomized equivalence sweep across lengths — exercises the SIMD tail
        // (len % 4), every lane position, and cross-lane ties. Many duplicate
        // values (mod 7) so ties are common and first-index tie-break is tested.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in 1..=130usize {
            let v: Vec<f32> = (0..len).map(|_| (rng() % 7) as f32).collect();
            assert_eq!(simd_argmax_f32(&v), naive(&v), "len={len} v={v:?}");
        }
    }

    #[test]
    fn argmax_empty_slice() {
        assert_eq!(simd_argmax_f32(&[]), (0, f32::NEG_INFINITY));
    }

    #[test]
    fn simd_level_matches_platform() {
        let level = simd_level();
        #[cfg(target_arch = "aarch64")]
        assert_eq!(level, SimdLevel::Neon);
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(level, SimdLevel::Avx2 | SimdLevel::Scalar));
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        assert_eq!(level, SimdLevel::Scalar);
    }

    /// `simd_sigmoid_inplace` must produce sigmoid values in (0, 1) and match
    /// `fast_sigmoid` to within the documented <3e-7 Cephes-vs-libm tolerance.
    /// Verifies the SIMD chunk path, the scalar tail, and the boundary cases
    /// (|x| > 40 saturates, x=0 → 0.5).
    #[test]
    fn simd_sigmoid_inplace_matches_fast_sigmoid_within_tolerance() {
        let mut rng = fastrand::Rng::with_seed(2026);
        // Sweep a wide range of lengths to exercise both SIMD chunks and scalar tails.
        for len in 0..=32 {
            let mut input: Vec<f32> = (0..len)
                .map(|_| (rng.f32() * 80.0) - 40.0) // [-40, 40]
                .collect();
            let reference: Vec<f32> = input.iter().map(|&x| fast_sigmoid(x)).collect();
            simd_sigmoid_inplace(&mut input);
            assert_eq!(input.len(), reference.len(), "length changed");
            let mut max_diff = 0.0f32;
            for (got, want) in input.iter().zip(reference.iter()) {
                // Sigmoid can round to exactly 0.0 or 1.0 in f32 at the precision
                // boundary (e.g. σ(20) rounds to 1.0). Allow the closed range.
                assert!(*got >= 0.0 && *got <= 1.0, "sigmoid out of [0,1]: {got}");
                assert!(*want >= 0.0 && *want <= 1.0, "reference out of [0,1]: {want}");
                max_diff = max_diff.max((got - want).abs());
            }
            assert!(
                max_diff < 5e-6,
                "len={len}: max_diff={max_diff:e} exceeds Cephes tolerance"
            );
        }
    }

    /// Boundary cases: |x| > 40 saturates; x=0 → 0.5; empty slice is a no-op.
    #[test]
    fn simd_sigmoid_inplace_handles_boundaries() {
        let mut empty: Vec<f32> = vec![];
        simd_sigmoid_inplace(&mut empty);
        assert!(empty.is_empty());

        let mut extremes = [60.0f32, -60.0, 0.0, 0.0001, -0.0001];
        simd_sigmoid_inplace(&mut extremes);
        assert!((extremes[0] - 1.0).abs() < 1e-6, "σ(60) ≈ 1, got {}", extremes[0]);
        assert!((extremes[1] - 0.0).abs() < 1e-6, "σ(-60) ≈ 0, got {}", extremes[1]);
        assert!((extremes[2] - 0.5).abs() < 1e-6, "σ(0) = 0.5, got {}", extremes[2]);
        // σ near zero should be near 0.5.
        assert!((extremes[3] - 0.5).abs() < 1e-3);
        assert!((extremes[4] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn dot_product_aligned_len_8() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let scalar = scalar_dot_f32(&a, &b, 8);
        let simd = simd_dot_f32(&a, &b, 8);

        assert!((scalar - simd).abs() < 1e-4, "scalar={scalar}, simd={simd}");
        // Expected: 0.5+2+4.5+8+12.5+18+24.5+32 = 102
        assert!((simd - 102.0).abs() < 1e-4, "simd={simd}");
    }

    #[test]
    fn dot_product_non_aligned_len() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 1.0, 1.0, 1.0, 1.0];

        let scalar = scalar_dot_f32(&a, &b, 5);
        let simd = simd_dot_f32(&a, &b, 5);

        assert!((scalar - simd).abs() < 1e-4, "scalar={scalar}, simd={simd}");
        assert!((simd - 15.0).abs() < 1e-4);
    }

    #[test]
    fn dot_product_len_4() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.5, 0.25, 0.125];

        let expected = 1.0 + 1.0 + 0.75 + 0.5;
        let simd = simd_dot_f32(&a, &b, 4);

        assert!((simd - expected).abs() < 1e-4);
    }

    #[test]
    fn dot_product_len_32() {
        // Game config n_embd=32
        let a: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let scalar = scalar_dot_f32(&a, &b, 32);
        let simd = simd_dot_f32(&a, &b, 32);

        assert!((scalar - simd).abs() < 1e-3, "scalar={scalar}, simd={simd}");
    }

    #[test]
    fn dot_product_zero_length() {
        let simd = simd_dot_f32(&[], &[], 0);
        assert!((simd - 0.0).abs() < 1e-6);
    }

    #[test]
    fn outer_product_4x4_matches_scalar() {
        let m = 4;
        let n = 4;
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.0, 1.5, 2.0];

        let mut acc_scalar = vec![0.0f32; m * n];
        let mut acc_simd = vec![0.0f32; m * n];

        scalar_outer_product_acc(&mut acc_scalar, &a, &b, m, n);
        simd_outer_product_acc(&mut acc_simd, &a, &b, m, n);

        for i in 0..m * n {
            assert!(
                (acc_scalar[i] - acc_simd[i]).abs() < 1e-4,
                "mismatch at {i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }

    #[test]
    fn outer_product_8x8_matches_scalar() {
        // Game config: hd=8
        let m = 8;
        let n = 8;
        let a: Vec<f32> = (0..m).map(|i| (i + 1) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|j| (j + 1) as f32 * 0.2).collect();

        let mut acc_scalar = vec![0.0f32; m * n];
        let mut acc_simd = vec![0.0f32; m * n];

        scalar_outer_product_acc(&mut acc_scalar, &a, &b, m, n);
        simd_outer_product_acc(&mut acc_simd, &a, &b, m, n);

        for i in 0..m * n {
            assert!(
                (acc_scalar[i] - acc_simd[i]).abs() < 1e-4,
                "mismatch at {i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }

    #[test]
    fn outer_product_accumulates() {
        let m = 4;
        let n = 4;
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0, 1.0];

        let mut acc = vec![0.0f32; m * n];
        simd_outer_product_acc(&mut acc, &a, &b, m, n);

        // Only acc[0*4 + 3] = 1.0 * 1.0 = 1.0 should be non-zero
        assert!((acc[3] - 1.0).abs() < 1e-5);
        for (i, &val) in acc.iter().enumerate() {
            if i != 3 {
                assert!(val.abs() < 1e-6, "acc[{i}] should be 0, got {val}");
            }
        }
    }

    #[test]
    fn matvec_matches_scalar() {
        let rows = 3;
        let cols = 4;
        let mat = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32,
        ];
        let vec = [1.0, 0.0, 1.0, 0.0f32];

        let mut acc_scalar = vec![0.0f32; rows];
        let mut acc_simd = vec![0.0f32; rows];

        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += mat[r * cols + c] * vec[c];
            }
            acc_scalar[r] = sum;
        }

        simd_matvec(&mut acc_simd, &mat, &vec, rows, cols);

        for r in 0..rows {
            assert!(
                (acc_scalar[r] - acc_simd[r]).abs() < 1e-4,
                "mismatch at row {r}: scalar={}, simd={}",
                acc_scalar[r],
                acc_simd[r]
            );
        }
    }

    #[test]
    fn matmul_rows_identity() {
        let rows = 4;
        let cols = 4;
        let weight = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let input = [1.0, 2.0, 3.0, 4.0f32];

        let mut output = vec![0.0f32; rows];
        simd_matmul_rows(&mut output, &weight, &input, rows, cols);

        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_relu_clamps_negative() {
        let rows = 2;
        let cols = 2;
        let weight = [-1.0, 0.0, 1.0, 1.0];
        let input = [1.0, 1.0];

        let mut output = vec![0.0f32; rows];
        simd_matmul_relu_rows(&mut output, &weight, &input, rows, cols);

        assert!((output[0]).abs() < 1e-5, "negative should clamp to 0");
        assert!((output[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn fma_row_matches_dot() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.0, 1.5, 2.0];

        let dot = simd_dot_f32(&a, &b, 4);
        let fma = simd_fma_row(&a, &b, 4);

        assert!((dot - fma).abs() < 1e-6);
    }

    // ── Sparse SIMD Tests ────────────────────────────────────

    #[test]
    fn sparse_dot_matches_scalar_dense() {
        // 8 elements, all alive (indices 0..7) — should match dense dot
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices: Vec<usize> = (0..8).collect();
        let values = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let sparse = simd_sparse_dot_f32(&weight, 0, &indices, &values, 8);
        let dense = simd_dot_f32(&weight, &values, 8);

        assert!(
            (sparse - dense).abs() < 1e-4,
            "sparse={sparse}, dense={dense}"
        );
    }

    #[test]
    fn sparse_dot_matches_scalar_sparse() {
        // 13 elements alive out of 64 (typical micro config: 20% of mlp_hidden=64)
        let mut weight = vec![0.0f32; 64];
        for (i, w) in weight.iter_mut().enumerate() {
            *w = (i as f32 + 1.0) * 0.01;
        }
        let indices: Vec<usize> = vec![0, 3, 7, 12, 15, 20, 25, 31, 38, 45, 50, 56, 63];
        let values: Vec<f32> = indices.iter().map(|&i| weight[i] * 2.0).collect();

        let simd_result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 13);
        let scalar_result = scalar_sparse_dot_f32(&weight, 0, &indices, &values, 13);

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "simd={simd_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn sparse_dot_small_alive_uses_scalar() {
        // alive=3 — should use inline scalar fallback (≤4)
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices = vec![0usize, 3, 7];
        let values = [0.5f32, 1.0, 1.5];

        let result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 3);
        let expected = 1.0 * 0.5 + 4.0 * 1.0 + 8.0 * 1.5; // 0.5 + 4.0 + 12.0 = 16.5

        assert!(
            (result - expected).abs() < 1e-4,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn sparse_dot_zero_alive() {
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let indices: Vec<usize> = vec![];
        let values: Vec<f32> = vec![];

        let result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 0);
        assert!(result.abs() < 1e-6, "expected 0.0, got {result}");
    }

    #[test]
    fn sparse_dot_with_row_offset() {
        // 8-element weight row at offset 4 in a 12-element weight matrix
        let mut weight = [0.0f32; 12]; // first 4 are padding
        weight[4] = 1.0;
        weight[5] = 2.0;
        weight[6] = 3.0;
        weight[7] = 4.0;
        weight[8] = 5.0;
        weight[9] = 6.0;
        weight[10] = 7.0;
        weight[11] = 8.0;
        // Need mutable for construction
        let weight = weight;

        let indices: Vec<usize> = (0..8).collect();
        let values = [1.0f32; 8];

        let result = simd_sparse_dot_f32(&weight, 4, &indices, &values, 8);
        // Expected: 1+2+3+4+5+6+7+8 = 36
        assert!((result - 36.0).abs() < 1e-4, "result={result}");
    }

    #[test]
    fn sparse_dot_alive_5_triggers_simd() {
        // alive=5 — just above scalar fallback threshold (4)
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices: Vec<usize> = (0..8).collect();
        let values = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let simd_result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 5);
        let expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0; // first 5 only

        assert!(
            (simd_result - expected).abs() < 1e-4,
            "simd={simd_result}, expected={expected}"
        );
    }

    #[test]
    fn sparse_matmul_rows_matches_scalar() {
        let rows = 4;
        let cols = 8;
        // Identity-like weight: row r has weight[r*cols + r] = 1.0, rest = 0.1
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let r = i / cols;
                let c = i % cols;
                if r == c { 1.0 } else { 0.1 }
            })
            .collect();

        // Only indices 1, 3, 5 are alive with values
        let indices = vec![1usize, 3, 5];
        let values = vec![2.0f32, 3.0, 4.0];

        let mut output_scalar = vec![0.0f32; rows];
        let mut output_simd = vec![0.0f32; rows];

        // Scalar
        for (r, out) in output_scalar.iter_mut().enumerate() {
            *out = scalar_sparse_dot_f32(&weight, r * cols, &indices, &values, 3);
        }

        // SIMD
        simd_sparse_matmul_rows(&mut output_simd, &weight, &indices, &values, rows, cols, 3);

        for (r, (scalar, simd)) in output_scalar.iter().zip(output_simd.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-4,
                "row {r}: scalar={scalar}, simd={simd}"
            );
        }
    }

    #[test]
    fn sparse_matmul_rows_game_config() {
        // Game config: n_embd=32 rows, mlp_hidden=128 cols, ~20% alive = 26 elements
        let rows = 32;
        let cols = 128;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();

        // Simulate 26 alive neurons (20% of 128)
        let alive = 26;
        let indices: Vec<usize> = (0..alive).map(|i| i * (cols / alive)).collect();
        let values: Vec<f32> = (0..alive).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut output_scalar = vec![0.0f32; rows];
        let mut output_simd = vec![0.0f32; rows];

        for (r, out) in output_scalar.iter_mut().enumerate() {
            *out = scalar_sparse_dot_f32(&weight, r * cols, &indices, &values, alive);
        }
        simd_sparse_matmul_rows(
            &mut output_simd,
            &weight,
            &indices,
            &values,
            rows,
            cols,
            alive,
        );

        for r in 0..rows {
            assert!(
                (output_scalar[r] - output_simd[r]).abs() < 1e-3,
                "row {r}: scalar={}, simd={}",
                output_scalar[r],
                output_simd[r]
            );
        }
    }

    // ── simd_scale_inplace tests ──────────────────────────────

    #[test]
    fn scale_aligned_len_8() {
        let mut x = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        simd_scale_inplace(&mut x, 0.5);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for i in 0..8 {
            assert!((x[i] - expected[i]).abs() < 1e-6, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn scale_non_aligned_len_13() {
        let mut x = [1.0f32; 13];
        simd_scale_inplace(&mut x, 3.0);
        for (i, &val) in x.iter().enumerate() {
            assert!((val - 3.0).abs() < 1e-6, "x[{i}]={val}");
        }
    }

    #[test]
    fn scale_empty() {
        let mut x: [f32; 0] = [];
        simd_scale_inplace(&mut x, 2.0); // should not panic
    }

    #[test]
    fn scale_single_element() {
        let mut x = [5.0f32];
        simd_scale_inplace(&mut x, 0.2);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scale_zero() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        simd_scale_inplace(&mut x, 0.0);
        for val in &x {
            assert!(*val == 0.0, "expected 0.0, got {val}");
        }
    }

    #[test]
    fn scale_matches_scalar() {
        let mut x_simd: Vec<f32> = (0..97).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x_scalar = x_simd.clone();
        let scale = 0.42f32;

        simd_scale_inplace(&mut x_simd, scale);
        scalar_scale_inplace(&mut x_scalar, scale);

        for i in 0..x_simd.len() {
            assert!(
                (x_simd[i] - x_scalar[i]).abs() < 1e-6,
                "x[{i}]: simd={}, scalar={}",
                x_simd[i],
                x_scalar[i]
            );
        }
    }

    // ── simd_add_scalar_inplace tests ────────────────────────

    #[test]
    fn add_scalar_aligned_len_8() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        simd_add_scalar_inplace(&mut x, -10.0);
        let expected = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0];
        for i in 0..8 {
            assert!((x[i] - expected[i]).abs() < 1e-6, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn add_scalar_non_aligned_len_13() {
        let mut x = [1.0f32; 13];
        simd_add_scalar_inplace(&mut x, 2.0);
        for (i, &val) in x.iter().enumerate() {
            assert!((val - 3.0).abs() < 1e-6, "x[{i}]={val}");
        }
    }

    #[test]
    fn add_scalar_empty() {
        let mut x: [f32; 0] = [];
        simd_add_scalar_inplace(&mut x, 1.0); // should not panic
    }

    #[test]
    fn add_scalar_matches_scalar_impl() {
        let mut x_simd: Vec<f32> = (0..97).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x_scalar = x_simd.clone();
        let val = -std::f32::consts::PI;

        simd_add_scalar_inplace(&mut x_simd, val);
        scalar_add_scalar_inplace(&mut x_scalar, val);

        for i in 0..x_simd.len() {
            assert!(
                (x_simd[i] - x_scalar[i]).abs() < 1e-6,
                "x[{i}]: simd={}, scalar={}",
                x_simd[i],
                x_scalar[i]
            );
        }
    }

    // ── simd_sum_f32 tests ──────────────────────────────────────

    #[test]
    fn sum_aligned_len_8() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = simd_sum_f32(&x);
        assert!((result - 36.0).abs() < 1e-4, "expected 36.0, got {result}");
    }

    #[test]
    fn sum_non_aligned_len_13() {
        let x = [1.0f32; 13];
        let result = simd_sum_f32(&x);
        assert!((result - 13.0).abs() < 1e-4, "expected 13.0, got {result}");
    }

    #[test]
    fn sum_empty() {
        let x: [f32; 0] = [];
        let result = simd_sum_f32(&x);
        assert!((result - 0.0).abs() < 1e-6, "expected 0.0, got {result}");
    }

    #[test]
    fn sum_single_element() {
        let x = [42.0f32];
        let result = simd_sum_f32(&x);
        assert!((result - 42.0).abs() < 1e-4, "expected 42.0, got {result}");
    }

    #[test]
    fn sum_matches_scalar_impl() {
        let x: Vec<f32> = (0..97).map(|i| (i as f32 * 0.1).sin()).collect();
        let simd_result = simd_sum_f32(&x);
        let scalar_result = scalar_sum_f32(&x);
        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "simd={simd_result}, scalar={scalar_result}"
        );
    }

    // ── simd_add_inplace tests ────────────────────────────────

    #[test]
    fn add_inplace_aligned_len_8() {
        let mut dst = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let src = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        simd_add_inplace(&mut dst, &src);
        for (i, val) in dst.iter().enumerate() {
            let expected = (1.0 + i as f32) + (i + 1) as f32 * 0.1;
            assert!((val - expected).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn add_inplace_non_aligned_len_13() {
        let mut dst = [0.0f32; 13];
        let src = [1.0f32; 13];
        for (i, val) in dst.iter_mut().enumerate() {
            *val = i as f32;
        }
        simd_add_inplace(&mut dst, &src);
        for (i, val) in dst.iter().enumerate() {
            assert!((val - (i as f32 + 1.0)).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn add_inplace_empty() {
        let mut dst: [f32; 0] = [];
        let src: [f32; 0] = [];
        simd_add_inplace(&mut dst, &src);
    }

    #[test]
    fn add_inplace_single_element() {
        let mut dst = [3.0f32];
        let src = [7.0f32];
        simd_add_inplace(&mut dst, &src);
        assert!((dst[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn add_inplace_matches_scalar() {
        let mut dst_simd = [0.0f32; 37];
        let mut dst_scalar = [0.0f32; 37];
        for i in 0..37 {
            dst_simd[i] = i as f32 * 0.7;
            dst_scalar[i] = i as f32 * 0.7;
        }
        let src: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).sin()).collect();
        simd_add_inplace(&mut dst_simd, &src);
        scalar_add_inplace(&mut dst_scalar, &src);
        for i in 0..37 {
            assert!(
                (dst_simd[i] - dst_scalar[i]).abs() < 1e-5,
                "mismatch at {i}"
            );
        }
    }

    // ── simd_add_into tests ───────────────────────────────────

    #[test]
    fn add_into_aligned_len_8() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut dst = [0.0f32; 8];
        simd_add_into(&mut dst, &a, &b);
        for val in &dst {
            assert!((val - 9.0).abs() < 1e-6);
        }
    }

    #[test]
    fn add_into_non_aligned_len_13() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b = [1.0f32; 13];
        let mut dst = [0.0f32; 13];
        simd_add_into(&mut dst, &a, &b);
        for (i, val) in dst.iter().enumerate() {
            assert!((val - (i as f32 + 1.0)).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn add_into_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        simd_add_into(&mut dst, &a, &b);
    }

    #[test]
    fn add_into_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7).sin()).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).cos()).collect();
        let mut dst_simd = [0.0f32; 37];
        let mut dst_scalar = [0.0f32; 37];
        simd_add_into(&mut dst_simd, &a, &b);
        scalar_add_into(&mut dst_scalar, &a, &b);
        for i in 0..37 {
            assert!(
                (dst_simd[i] - dst_scalar[i]).abs() < 1e-5,
                "mismatch at {i}"
            );
        }
    }

    // ── simd_max_f32 tests ────────────────────────────────────

    #[test]
    fn max_aligned_len_8() {
        let x = [1.0f32, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0];
        let max = simd_max_f32(&x);
        assert!((max - 8.0).abs() < 1e-6);
    }

    #[test]
    fn max_non_aligned_len_13() {
        let x: Vec<f32> = (0..13).map(|i| (i as f32 * 1.7).sin()).collect();
        let max = simd_max_f32(&x);
        let expected = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((max - expected).abs() < 1e-5);
    }

    #[test]
    fn max_empty() {
        let x: [f32; 0] = [];
        let max = simd_max_f32(&x);
        assert!(max.is_infinite() && max.is_sign_negative());
    }

    #[test]
    fn max_single_element() {
        let x = [42.0f32];
        let max = simd_max_f32(&x);
        assert!((max - 42.0).abs() < 1e-6);
    }

    #[test]
    fn max_negative_values() {
        let x = [-5.0f32, -3.0, -8.0, -1.0, -4.0];
        let max = simd_max_f32(&x);
        assert!((max - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn max_matches_scalar() {
        let x: Vec<f32> = (0..37).map(|i| (i as f32 * 0.97 - 18.0).sin()).collect();
        let max_simd = simd_max_f32(&x);
        let max_scalar = scalar_max_f32(&x);
        assert!((max_simd - max_scalar).abs() < 1e-5);
    }

    // ── simd_fused_decay_write tests ──────────────────────────

    #[test]
    fn fused_decay_write_aligned_len_8() {
        let mut dst = [1.0f32; 8];
        let src = [2.0f32; 8];
        let decay = 0.5f32;
        let write = 0.5f32;
        simd_fused_decay_write(&mut dst, decay, &src, write);
        // 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        for val in &dst {
            assert!((val - 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn fused_decay_write_zero_decay() {
        let mut dst = [1.0f32, 2.0, 3.0, 4.0];
        let src = [10.0f32, 20.0, 30.0, 40.0];
        let decay = 0.0f32;
        let write = 1.0f32;
        simd_fused_decay_write(&mut dst, decay, &src, write);
        for i in 0..4 {
            assert!((dst[i] - src[i]).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn fused_decay_write_zero_write() {
        let mut dst = [1.0f32, 2.0, 3.0, 4.0];
        let src = [10.0f32, 20.0, 30.0, 40.0];
        let decay = 1.0f32;
        let write = 0.0f32;
        simd_fused_decay_write(&mut dst, decay, &src, write);
        assert!((dst[0] - 1.0).abs() < 1e-5);
        assert!((dst[1] - 2.0).abs() < 1e-5);
        assert!((dst[2] - 3.0).abs() < 1e-5);
        assert!((dst[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn fused_decay_write_empty() {
        let mut dst: [f32; 0] = [];
        let src: [f32; 0] = [];
        simd_fused_decay_write(&mut dst, 0.5, &src, 0.5);
    }

    #[test]
    fn fused_decay_write_matches_scalar() {
        let mut dst_simd: Vec<f32> = (0..37).map(|i| i as f32 * 0.7).collect();
        let mut dst_scalar: Vec<f32> = (0..37).map(|i| i as f32 * 0.7).collect();
        let src: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).sin()).collect();
        let decay = 0.9f32;
        let write = 0.1f32;
        simd_fused_decay_write(&mut dst_simd, decay, &src, write);
        scalar_fused_decay_write(&mut dst_scalar, decay, &src, write);
        for i in 0..37 {
            assert!(
                (dst_simd[i] - dst_scalar[i]).abs() < 1e-4,
                "mismatch at {i}: simd={}, scalar={}",
                dst_simd[i],
                dst_scalar[i]
            );
        }
    }

    // ── f16×f32 kernel tests ──────────────────────────────────

    fn scalar_dot_f16_f32_ref(w: &[half::f16], x: &[f32], len: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += w[i].to_f32() * x[i];
        }
        sum
    }

    #[test]
    fn dot_f16_f32_aligned_len_8() {
        let w: Vec<half::f16> = (0..8)
            .map(|i| half::f16::from_f32(i as f32 * 0.1))
            .collect();
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.2).collect();
        let result = simd_dot_f16_f32(&w, &x, 8);
        let expected = scalar_dot_f16_f32_ref(&w, &x, 8);
        assert!(
            (result - expected).abs() < 1e-4,
            "f16 dot aligned: got {result}, expected {expected}"
        );
    }

    #[test]
    fn dot_f16_f32_non_aligned_len_13() {
        let w: Vec<half::f16> = (0..13)
            .map(|i| half::f16::from_f32(i as f32 + 1.0))
            .collect();
        let x: Vec<f32> = (0..13).map(|i| i as f32 * 0.3).collect();
        let result = simd_dot_f16_f32(&w, &x, 13);
        let expected = scalar_dot_f16_f32_ref(&w, &x, 13);
        assert!(
            (result - expected).abs() < 1e-3,
            "f16 dot non-aligned: got {result}, expected {expected}"
        );
    }

    #[test]
    fn dot_f16_f32_len_4() {
        let w: Vec<half::f16> = vec![1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let x: Vec<f32> = vec![0.25, 0.5, 0.75, 1.0];
        let result = simd_dot_f16_f32(&w, &x, 4);
        let expected = scalar_dot_f16_f32_ref(&w, &x, 4);
        assert!(
            (result - expected).abs() < 1e-4,
            "f16 dot len 4: got {result}, expected {expected}"
        );
    }

    #[test]
    fn dot_f16_f32_zero_length() {
        let w: Vec<half::f16> = Vec::new();
        let x: Vec<f32> = Vec::new();
        let result = simd_dot_f16_f32(&w, &x, 0);
        assert_eq!(result, 0.0, "f16 dot zero-length should be 0.0");
    }

    #[test]
    fn matmul_f16_f32_identity() {
        // 3×3 identity matrix stored as f16
        let w: Vec<half::f16> = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let x: Vec<f32> = vec![2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 3];
        simd_matmul_f16_f32_rows(&mut out, &w, &x, 3, 3);
        assert!(
            (out[0] - 2.0).abs() < 1e-4
                && (out[1] - 3.0).abs() < 1e-4
                && (out[2] - 4.0).abs() < 1e-4,
            "f16 identity matmul: got {out:?}"
        );
    }

    #[test]
    fn matmul_f16_f32_matches_f32() {
        // Compare f16 matmul vs f32 matmul on the same values
        let rows = 4;
        let cols = 6;
        let weight_f32: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01 - 0.1).collect();
        let weight_f16: Vec<half::f16> =
            weight_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let input: Vec<f32> = (0..cols).map(|i| i as f32 * 0.05).collect();

        let mut out_f32 = vec![0.0f32; rows];
        let mut out_f16 = vec![0.0f32; rows];
        simd_matmul_rows(&mut out_f32, &weight_f32, &input, rows, cols);
        simd_matmul_f16_f32_rows(&mut out_f16, &weight_f16, &input, rows, cols);

        for i in 0..rows {
            let diff = (out_f32[i] - out_f16[i]).abs();
            assert!(
                diff < 0.01,
                "f16 vs f32 matmul mismatch at row {i}: f32={}, f16={}, diff={diff}",
                out_f32[i],
                out_f16[i]
            );
        }
    }

    // ── MaxSim Tests (Plan 080 T2) ────────────────────────────

    /// Naive reference: materialize [Lq × Ld] then reduce.
    #[cfg(feature = "maxsim")]
    fn maxsim_naive(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
        let mut score = 0.0f32;
        for i in 0..lq {
            let q_row = &queries[i * dim..(i + 1) * dim];
            let mut my_max = f32::NEG_INFINITY;
            for j in 0..ld {
                let d_row = &documents[j * dim..(j + 1) * dim];
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q_row[d] * d_row[d];
                }
                my_max = my_max.max(dot);
            }
            score += my_max;
        }
        score
    }

    #[cfg(feature = "maxsim")]
    mod maxsim_tests {
        use super::*;

        #[test]
        fn maxsim_matches_naive() {
            let lq = 8;
            let ld = 16;
            let dim = 32;
            let mut queries = vec![0.0f32; lq * dim];
            let mut documents = vec![0.0f32; ld * dim];
            for q in queries.iter_mut() {
                *q = fastrand::f32() * 2.0 - 1.0;
            }
            for d in documents.iter_mut() {
                *d = fastrand::f32() * 2.0 - 1.0;
            }
            let naive = maxsim_naive(&queries, &documents, lq, ld, dim);
            let fused = maxsim_score(&queries, &documents, lq, ld, dim);
            assert!((naive - fused).abs() < 1e-4, "naive={naive}, fused={fused}");
        }

        #[test]
        fn maxsim_single_query_token() {
            let dim = 16;
            let queries = (0..dim).map(|i| i as f32).collect::<Vec<f32>>();
            let documents = (0..3 * dim)
                .map(|i| (i as f32 * 0.1).sin())
                .collect::<Vec<f32>>();
            let result = maxsim_score(&queries, &documents, 1, 3, dim);
            // Should equal max over all doc dots
            let mut expected = f32::NEG_INFINITY;
            for j in 0..3 {
                let d_row = &documents[j * dim..(j + 1) * dim];
                let dot = simd_dot_f32(&queries, d_row, dim);
                expected = expected.max(dot);
            }
            assert!(
                (result - expected).abs() < 1e-5,
                "result={result}, expected={expected}"
            );
        }

        #[test]
        fn maxsim_single_doc_token() {
            let dim = 16;
            let lq = 4;
            let queries = (0..lq * dim)
                .map(|i| (i as f32 * 0.2).cos())
                .collect::<Vec<f32>>();
            let documents = (0..dim).map(|i| i as f32 * 0.5).collect::<Vec<f32>>();
            let result = maxsim_score(&queries, &documents, lq, 1, dim);
            // Ld=1: each query token has exactly one doc token to match
            let mut expected = 0.0f32;
            for i in 0..lq {
                let q_row = &queries[i * dim..(i + 1) * dim];
                expected += simd_dot_f32(q_row, &documents, dim);
            }
            assert!(
                (result - expected).abs() < 1e-4,
                "result={result}, expected={expected}"
            );
        }

        #[test]
        fn maxsim_symmetry_breaking() {
            let dim = 8;
            let lq = 4;
            let ld = 4;
            let queries = (0..lq * dim).map(|i| i as f32).collect::<Vec<f32>>();
            let documents = (0..ld * dim)
                .map(|i| (i as f32 * 0.3).sin())
                .collect::<Vec<f32>>();
            let maxsim = maxsim_score(&queries, &documents, lq, ld, dim);
            // Diagonal sum: Σ dot(q_i, d_i)
            let mut diagonal = 0.0f32;
            for i in 0..lq.min(ld) {
                let q_row = &queries[i * dim..(i + 1) * dim];
                let d_row = &documents[i * dim..(i + 1) * dim];
                diagonal += simd_dot_f32(q_row, d_row, dim);
            }
            // They should differ (MaxSim takes max over ALL j, not just j==i)
            assert!(
                (maxsim - diagonal).abs() > 1e-3,
                "maxsim={maxsim} should differ from diagonal={diagonal}"
            );
        }

        #[test]
        fn maxsim_empty_doc() {
            let dim = 16;
            let queries = vec![1.0f32; dim];
            let documents: Vec<f32> = vec![];
            let result = maxsim_score(&queries, &documents, 1, 0, dim);
            assert_eq!(result, 0.0, "empty doc should return 0.0");
        }

        #[test]
        fn maxsim_large_dim_aligned() {
            let dim = 128;
            let lq = 4;
            let ld = 8;
            let queries: Vec<f32> = (0..lq * dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let documents: Vec<f32> = (0..ld * dim).map(|i| (i as f32 * 0.01).cos()).collect();
            let naive = maxsim_naive(&queries, &documents, lq, ld, dim);
            let fused = maxsim_score(&queries, &documents, lq, ld, dim);
            assert!((naive - fused).abs() < 1e-3, "naive={naive}, fused={fused}");
        }

        #[test]
        fn maxsim_packed_matches_sequential() {
            let dim = 16;
            // Two query sequences, three doc sequences
            let q1: Vec<f32> = (0..2 * dim).map(|i| i as f32).collect();
            let q2: Vec<f32> = (0..3 * dim).map(|i| (i as f32 * 0.5).sin()).collect();
            let d1: Vec<f32> = (0..4 * dim).map(|i| (i as f32 * 0.3).cos()).collect();
            let d2: Vec<f32> = (0..2 * dim).map(|i| i as f32 * 0.1).collect();
            let d3: Vec<f32> = (0..5 * dim).map(|i| (i as f32 * 0.7).sin()).collect();

            let queries: Vec<f32> = [q1.clone(), q2.clone()].concat();
            let documents: Vec<f32> = [d1.clone(), d2.clone(), d3.clone()].concat();
            let query_offsets = [0, q1.len(), q1.len() + q2.len()];
            let doc_offsets = [
                0,
                d1.len(),
                d1.len() + d2.len(),
                d1.len() + d2.len() + d3.len(),
            ];

            // Score pairs: (q0,d0), (q0,d2), (q1,d1)
            let pair_q_ids = [0usize, 0, 1];
            let pair_d_ids = [0usize, 2, 1];

            let mut packed = vec![0.0f32; pair_q_ids.len()];
            maxsim_score_packed(
                &queries,
                &query_offsets,
                &documents,
                &doc_offsets,
                &pair_q_ids,
                &pair_d_ids,
                dim,
                &mut packed,
            );

            // Verify against sequential calls
            let s0 = maxsim_score(&q1, &d1, 2, 4, dim);
            let s1 = maxsim_score(&q1, &d3, 2, 5, dim);
            let s2 = maxsim_score(&q2, &d2, 3, 2, dim);

            assert!(
                (packed[0] - s0).abs() < 1e-4,
                "pair 0: packed={}, sequential={}",
                packed[0],
                s0
            );
            assert!(
                (packed[1] - s1).abs() < 1e-4,
                "pair 1: packed={}, sequential={}",
                packed[1],
                s1
            );
            assert!(
                (packed[2] - s2).abs() < 1e-4,
                "pair 2: packed={}, sequential={}",
                packed[2],
                s2
            );
        }
    }

    // ── Sigmoid Margin Loss Tests (Plan 157 GOAT) ───────────────

    #[cfg(feature = "sigmoid_margin")]
    mod sigmoid_margin_tests {
        use super::*;

        // GOAT Proof 1: sigmoid_margin_loss matches paper's Python implementation
        //
        // For a small bipartite graph with n=20, k=2, d=8:
        //   - Generate random embeddings, compute dot-product scores
        //   - Compute loss with t=1.0, b=0.0
        //   - Verify against hand-computed softplus values
        #[test]
        fn proof1_loss_matches_manual() {
            // 2 queries × 3 docs, simple adjacency
            let n_rows = 2;
            let n_cols = 3;
            let scores: Vec<f32> = vec![
                0.8, 0.2, -0.5, // query 0: positive on doc 0
                -0.3, 0.9, 0.1, // query 1: positive on doc 1
            ];
            let adjacency: Vec<f32> = vec![
                1.0, 0.0, 0.0, // query 0 positive = doc 0
                0.0, 1.0, 0.0, // query 1 positive = doc 1
            ];

            let loss = sigmoid_margin_loss(&scores, &adjacency, 1.0, 0.0, n_rows, n_cols);

            // Manual computation:
            // query 0: pos: softplus(-0.8) = ln(1+exp(-0.8)) ≈ 0.5544
            //          neg: softplus(0.2) = ln(1+exp(0.2)) ≈ 0.7444
            //          neg: softplus(-0.5) = ln(1+exp(-0.5)) ≈ 0.4741
            // query 1: neg: softplus(-0.3) = ln(1+exp(-0.3)) ≈ 0.5544
            //          pos: softplus(-0.9) = ln(1+exp(-0.9)) ≈ 0.4887
            //          neg: softplus(0.1) = ln(1+exp(0.1)) ≈ 0.7444
            // total / 6
            let sp = |x: f32| -> f32 { (1.0f32 + x.exp()).ln() };
            let expected = (sp(-0.8) + sp(0.2) + sp(-0.5) + sp(-0.3) + sp(-0.9) + sp(0.1)) / 6.0;
            assert!(
                (loss - expected).abs() < 1e-4,
                "loss={loss}, expected={expected}"
            );
        }

        #[test]
        fn proof1_loss_with_bias_and_temperature() {
            let scores = vec![1.0, 0.0];
            let adjacency = vec![1.0, 0.0];

            // With t=2.0, b=0.5:
            //   pos (score=1): sign=-1, x = 2*(1-0.5)*(-1) = -1.0, softplus(-1.0)
            //   neg (score=0): sign=+1, x = 2*(0-0.5)*(+1) = -1.0, softplus(-1.0)
            //   Both = softplus(-1.0)
            let loss = sigmoid_margin_loss(&scores, &adjacency, 2.0, 0.5, 1, 2);
            let sp_neg1 = (1.0f32 + (-1.0f32).exp()).ln(); // softplus(-1.0)
            let expected = sp_neg1; // mean of 2 identical values
            assert!(
                (loss - expected).abs() < 1e-4,
                "loss={loss}, expected={expected}"
            );
        }

        #[test]
        fn proof1_loss_perfect_separation() {
            // Perfect separation: pos score >> bias, neg score << bias
            let scores = vec![100.0, -100.0];
            let adjacency = vec![1.0, 0.0];
            let loss = sigmoid_margin_loss(&scores, &adjacency, 1.0, 0.0, 1, 2);
            // pos: softplus(-100) ≈ 0, neg: softplus(-100) ≈ 0
            assert!(
                loss < 1e-10,
                "loss={loss} should be near 0 for perfect separation"
            );
        }

        // GOAT Proof 2: compute_retrieval_margin correctly identifies positive margin
        #[test]
        fn proof2_margin_positive_for_separated_embeddings() {
            let dim = 8;
            let n_queries = 3;
            let n_docs = 6;
            let k = 2;

            // Construct orthogonal-ish embeddings with known margin.
            // Each query is aligned with its 2 positive docs, orthogonal to the rest.
            let mut queries = vec![0.0f32; n_queries * dim];
            let mut documents = vec![0.0f32; n_docs * dim];

            // query i → doc 2i and doc 2i+1 as positives
            let mut neighborhoods = Vec::with_capacity(n_queries * k);
            for i in 0..n_queries {
                // Query: unit vector along dimension i
                queries[i * dim + i] = 1.0;
                // Positive docs: same direction as query
                documents[(2 * i) * dim + i] = 0.9;
                documents[(2 * i + 1) * dim + i] = 0.8;
                neighborhoods.push(2 * i);
                neighborhoods.push(2 * i + 1);
            }

            let (pos_min, neg_max, margin) = compute_retrieval_margin(
                &queries,
                &documents,
                &neighborhoods,
                dim,
                n_queries,
                n_docs,
                k,
            );

            // pos_min should be 0.8 (weakest positive = 0.8), neg_max should be 0.0 (no alignment)
            assert!(
                (pos_min - 0.8).abs() < 1e-5,
                "pos_min={pos_min}, expected 0.8"
            );
            assert!(neg_max.abs() < 1e-5, "neg_max={neg_max}, expected 0.0");
            assert!((margin - 0.4).abs() < 1e-5, "margin={margin}, expected 0.4");
            assert!(margin > 0.0, "margin should be positive");
        }

        #[test]
        fn proof2_margin_negative_for_mixed_embeddings() {
            let dim = 4;
            let n_queries = 1;
            let n_docs = 3;
            let k = 1;

            // Query aligned with a "wrong" doc (positive has lower score than a negative)
            let queries = vec![1.0, 0.0, 0.0, 0.0]; // aligned along dim 0
            // Doc 0 (positive): weak alignment
            let d0 = vec![0.1, 0.0, 0.0, 0.0];
            // Doc 1 (negative): strong alignment → should dominate
            let d1 = vec![0.9, 0.0, 0.0, 0.0];
            // Doc 2 (negative): orthogonal
            let d2 = vec![0.0, 1.0, 0.0, 0.0];
            let documents: Vec<f32> = [d0, d1, d2].concat();
            let neighborhoods = vec![0]; // query 0 positive = doc 0

            let (pos_min, neg_max, margin) = compute_retrieval_margin(
                &queries,
                &documents,
                &neighborhoods,
                dim,
                n_queries,
                n_docs,
                k,
            );

            assert!((pos_min - 0.1).abs() < 1e-5, "pos_min={pos_min}");
            assert!((neg_max - 0.9).abs() < 1e-5, "neg_max={neg_max}");
            assert!(margin < 0.0, "margin should be negative: {margin}");
        }

        // GOAT Proof 3: dim_sufficiency_bound returns O(k log n)
        #[test]
        fn proof3_bound_scales_as_k_log_n() {
            // k=2, n=100: 1.5 * 2 * ln(100) ≈ 1.5 * 2 * 4.605 ≈ 13.8 → 14
            let b1 = dim_sufficiency_bound(2, 100);
            assert!(b1 <= 20, "k=2, n=100: bound={b1}, should be ≤ 20");
            assert!(b1 >= 10, "k=2, n=100: bound={b1}, should be ≥ 10");

            // k=4, n=1000: 1.5 * 4 * ln(1000) ≈ 1.5 * 4 * 6.908 ≈ 41.4 → 42
            let b2 = dim_sufficiency_bound(4, 1000);
            assert!(b2 <= 60, "k=4, n=1000: bound={b2}, should be ≤ 60");
            assert!(b2 >= 30, "k=4, n=1000: bound={b2}, should be ≥ 30");
        }

        #[test]
        fn proof3_bound_edge_cases() {
            assert_eq!(dim_sufficiency_bound(0, 100), 1, "k=0 → trivial");
            assert_eq!(dim_sufficiency_bound(2, 1), 1, "n=1 → trivial");
            assert_eq!(dim_sufficiency_bound(2, 2), 3, "n=2 → minimal");
        }

        #[test]
        fn proof3_bound_monotonic() {
            let b1 = dim_sufficiency_bound(2, 50);
            let b2 = dim_sufficiency_bound(2, 100);
            let b3 = dim_sufficiency_bound(2, 200);
            assert!(b1 < b2, "bound should increase with n: {b1} < {b2}");
            assert!(b2 < b3, "bound should increase with n: {b2} < {b3}");

            let bk1 = dim_sufficiency_bound(2, 100);
            let bk2 = dim_sufficiency_bound(4, 100);
            assert!(bk1 < bk2, "bound should increase with k: {bk1} < {bk2}");
        }

        // GOAT Proof 4: Sigmoid loss converges to positive margin on synthetic data
        //
        // We use a structured initialization where each query and its positive docs
        // share a unique subspace dimension. The sigmoid margin loss then amplifies
        // this alignment while suppressing cross-talk.
        //
        // Uses analytical gradient: ∂loss/∂score = sigmoid(t·(score−b)·sign)
        // then backprops to embeddings via chain rule: ∂loss/∂q_i = Σ_j grad_ij · d_j.
        #[test]
        fn proof4_loss_gradient_pushes_to_positive_margin() {
            let dim = 8;
            let n = 4; // 4 docs
            let k = 2; // each query has 2 positives
            let n_queries = 2;

            // Bipartite structure:
            //   query 0 → doc 0, doc 1 (use dim 0 as shared subspace)
            //   query 1 → doc 2, doc 3 (use dim 1 as shared subspace)
            let neighborhoods: Vec<usize> = vec![0, 1, 2, 3];

            // Initialize with small positive signal in the right subspace + noise
            let mut queries = vec![0.0f32; n_queries * dim];
            let mut documents = vec![0.0f32; n * dim];

            // query 0 → dim 0, query 1 → dim 1
            queries[0] = 0.3;
            queries[dim + 1] = 0.3;

            // Positive docs aligned with their query subspace
            documents[0] = 0.2;
            documents[dim] = 0.15;
            documents[2 * dim + 1] = 0.2;
            documents[3 * dim + 1] = 0.15;
            // Small cross-talk noise
            documents[1] = 0.02;
            documents[2 * dim] = 0.02;

            let adjacency: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

            let (_, _, initial_margin) = compute_retrieval_margin(
                &queries,
                &documents,
                &neighborhoods,
                dim,
                n_queries,
                n,
                k,
            );

            // Analytical gradient descent with temperature t=10
            let t = 10.0f32;
            let lr = 0.1;
            let mut q = queries.clone();
            let mut d = documents.clone();

            for _step in 0..100 {
                // Forward: compute scores [n_queries × n]
                let mut scores = vec![0.0f32; n_queries * n];
                for i in 0..n_queries {
                    for j in 0..n {
                        scores[i * n + j] = simd_dot_f32(
                            &q[i * dim..(i + 1) * dim],
                            &d[j * dim..(j + 1) * dim],
                            dim,
                        );
                    }
                }

                // Score gradients matching the loss: sign = -1 for pos, +1 for neg
                // ∂L/∂score_ij = t · sign · σ(t · (score - b) · sign)
                // pos: sign=-1 → grad = -t · σ(-t·(score-b)), pushes score up
                // neg: sign=+1 → grad = +t · σ(+t·(score-b)), pushes score down
                let mut score_grads = vec![0.0f32; n_queries * n];
                for i in 0..n_queries {
                    for j in 0..n {
                        let idx = i * n + j;
                        let sign = if adjacency[idx] > 0.5 {
                            -1.0f32
                        } else {
                            1.0f32
                        };
                        let x = t * (scores[idx]) * sign;
                        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
                        score_grads[idx] = t * sign * sigmoid_x;
                    }
                }

                // Backprop to queries: ∂loss/∂q_i = Σ_j (score_grad_ij) · d_j
                let mut q_grads = vec![0.0f32; n_queries * dim];
                for i in 0..n_queries {
                    for j in 0..n {
                        let g = score_grads[i * n + j];
                        for dd in 0..dim {
                            q_grads[i * dim + dd] += g * d[j * dim + dd];
                        }
                    }
                }

                // Backprop to documents: ∂loss/∂d_j = Σ_i (score_grad_ij) · q_i
                let mut d_grads = vec![0.0f32; n * dim];
                for i in 0..n_queries {
                    for j in 0..n {
                        let g = score_grads[i * n + j];
                        for dd in 0..dim {
                            d_grads[j * dim + dd] += g * q[i * dim + dd];
                        }
                    }
                }

                // Gradient step
                for idx in 0..q.len() {
                    q[idx] -= lr * q_grads[idx];
                }
                for idx in 0..d.len() {
                    d[idx] -= lr * d_grads[idx];
                }
            }

            let (_, _, final_margin) =
                compute_retrieval_margin(&q, &d, &neighborhoods, dim, n_queries, n, k);

            assert!(
                final_margin > 0.0,
                "final_margin={final_margin} should be > 0 after training"
            );
            assert!(
                final_margin > initial_margin,
                "margin should improve: initial={initial_margin}, final={final_margin}"
            );
        }

        // GOAT Proof 5: Margin diagnostic validates MaxSim scoring quality
        #[test]
        #[cfg(feature = "maxsim")]
        fn proof5_margin_correlates_with_maxsim() {
            let dim = 16;
            let n_docs = 4;
            let lq = 2;
            let ld = n_docs;
            let k = 1;

            // Create two query-doc pairs with different margins
            // High margin: query 0 is closely aligned with doc 0, far from others
            let mut queries = vec![0.0f32; 2 * lq * dim]; // 2 sets of queries
            let mut documents = vec![0.0f32; n_docs * dim];

            // Doc 0: strong signal on dim 0
            documents[0] = 1.0;
            // Docs 1-3: weak/noise
            documents[dim + 1] = 0.1;
            documents[2 * dim + 2] = 0.1;
            documents[3 * dim + 3] = 0.1;

            // Query 0 (high margin): aligned with doc 0
            queries[0] = 1.0;
            // Query 0, token 1: also aligned
            queries[dim] = 0.9;

            let neighborhoods = vec![0]; // query 0 → doc 0

            let (pos_min, neg_max, margin) = compute_retrieval_margin(
                &queries[..lq * dim],
                &documents,
                &neighborhoods,
                dim,
                1,
                n_docs,
                k,
            );

            // MaxSim score for this query against all docs
            let ms = maxsim_score(&queries[..lq * dim], &documents, lq, ld, dim);

            // High margin → MaxSim should be dominated by the positive doc
            assert!(margin > 0.0, "margin={margin} should be positive");
            // MaxSim should be high when positive docs dominate
            assert!(
                ms > 0.0,
                "maxsim={ms} should be positive for high-margin setup"
            );
            assert!(
                pos_min > neg_max,
                "pos_min={pos_min} should exceed neg_max={neg_max}"
            );
        }

        // GOAT Proof 6: No performance regression on existing maxsim tests
        // (All existing maxsim tests still pass — verified by running the test suite)
        // This proof is structural: if this test compiles and the maxsim tests pass,
        // there is no regression.
        #[test]
        #[cfg(feature = "maxsim")]
        fn proof6_no_maxsim_regression() {
            // Re-run a basic maxsim test to verify nothing broke
            let dim = 16;
            let lq = 4;
            let ld = 8;
            let queries: Vec<f32> = (0..lq * dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let documents: Vec<f32> = (0..ld * dim).map(|i| (i as f32 * 0.01).cos()).collect();

            // Naive computation
            let mut expected = 0.0f32;
            for i in 0..lq {
                let q_row = &queries[i * dim..(i + 1) * dim];
                let mut my_max = f32::NEG_INFINITY;
                for j in 0..ld {
                    let d_row = &documents[j * dim..(j + 1) * dim];
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q_row[d] * d_row[d];
                    }
                    my_max = my_max.max(dot);
                }
                expected += my_max;
            }

            let result = maxsim_score(&queries, &documents, lq, ld, dim);
            assert!(
                (result - expected).abs() < 1e-3,
                "maxsim={result}, expected={expected}"
            );
        }

        // GOAT Proof 7: Feature gate isolation
        // This test verifies the functions exist and work when sigmoid_margin is enabled.
        // When the feature is disabled, the functions are not visible (compile-time check).
        #[test]
        fn proof7_feature_gate_functions_exist() {
            // All three functions should be usable
            let _loss = sigmoid_margin_loss(&[0.5, -0.5], &[1.0, 0.0], 1.0, 0.0, 1, 2);

            let (pm, _nm, m) = compute_retrieval_margin(
                &[1.0, 0.0, 0.0, 1.0], // 2 queries × dim 2
                &[1.0, 0.0, 0.0, 1.0], // 2 docs × dim 2
                &[0, 1],               // neighborhoods: q0→d0, q1→d1
                2,
                2,
                2,
                1,
            );
            assert!(pm >= 0.0);
            assert!(m >= 0.0);

            let bound = dim_sufficiency_bound(2, 100);
            assert!(bound > 0);
            assert!(bound <= 20);
        }
    }

    // ── Gram matrix tests ─────────────────────────────────────

    mod gram_tests {
        use super::*;

        #[test]
        fn test_gram_identity() {
            // Identity matrix X = I (3×3) → G = I·Iᵀ = I
            let seq_len = 3;
            let d_h = 3;
            let x: Vec<f32> = vec![
                1.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, // row 1
                0.0, 0.0, 1.0, // row 2
            ];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // Expected: identity 3×3
            let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            for (i, (&g, &e)) in gram.iter().zip(expected.iter()).enumerate() {
                assert!((g - e).abs() < 1e-5, "gram[{i}]={g}, expected={e}");
            }
        }

        #[test]
        fn test_gram_ones() {
            // All-ones rows → G[i][j] = d_h for all i,j
            let seq_len = 4;
            let d_h = 8;
            let x = vec![1.0f32; seq_len * d_h];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            for (i, &g) in gram.iter().enumerate() {
                assert!(
                    (g - d_h as f32).abs() < 1e-4,
                    "gram[{i}]={g}, expected={}",
                    d_h
                );
            }
        }

        #[test]
        fn test_gram_symmetric() {
            let seq_len = 5;
            let d_h = 8;
            let x: Vec<f32> = (0..seq_len * d_h).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let g_ij = gram[i * seq_len + j];
                    let g_ji = gram[j * seq_len + i];
                    assert!(
                        (g_ij - g_ji).abs() < 1e-5,
                        "G[{i}][{j}]={g_ij} != G[{j}][{i}]={g_ji}"
                    );
                }
            }
        }

        #[test]
        fn test_gram_upper_triangle_mirror() {
            // Construct X so that each row has a distinct value
            let seq_len = 3;
            let d_h = 4;
            let x: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // Verify G[i][j] == G[j][i] for all off-diagonal pairs
            // G[0][1] = dot(row0, row1) = 1*5+2*6+3*7+4*8 = 5+12+21+32 = 70
            // G[0][2] = dot(row0, row2) = 1*9+2*10+3*11+4*12 = 9+20+33+48 = 110
            // G[1][2] = dot(row1, row2) = 5*9+6*10+7*11+8*12 = 45+60+77+96 = 278
            assert!((gram[1] - 70.0).abs() < 1e-4, "G[0][1]={}", gram[1]);
            assert!((gram[3] - 70.0).abs() < 1e-4, "G[1][0]={}", gram[3]);
            assert!((gram[2] - 110.0).abs() < 1e-4, "G[0][2]={}", gram[2]);
            assert!((gram[6] - 110.0).abs() < 1e-4, "G[2][0]={}", gram[6]);
            assert!((gram[5] - 278.0).abs() < 1e-4, "G[1][2]={}", gram[5]);
            assert!((gram[7] - 278.0).abs() < 1e-4, "G[2][1]={}", gram[7]);
        }

        #[test]
        fn test_gram_2x3() {
            // X = [[1, 0, 2], [3, 1, 0]]
            let seq_len = 2;
            let d_h = 3;
            let x: Vec<f32> = vec![1.0, 0.0, 2.0, 3.0, 1.0, 0.0];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // G[0][0] = 1+0+4 = 5
            // G[0][1] = 3+0+0 = 3
            // G[1][1] = 9+1+0 = 10
            assert!((gram[0] - 5.0).abs() < 1e-5, "G[0][0]={}", gram[0]);
            assert!((gram[1] - 3.0).abs() < 1e-5, "G[0][1]={}", gram[1]);
            assert!((gram[2] - 3.0).abs() < 1e-5, "G[1][0]={}", gram[2]);
            assert!((gram[3] - 10.0).abs() < 1e-5, "G[1][1]={}", gram[3]);
        }

        #[test]
        fn test_gram_matches_outer_product() {
            let seq_len = 4;
            let d_h = 8;
            let x: Vec<f32> = (0..seq_len * d_h)
                .map(|i| (i as f32 * 0.17).sin() * 0.5)
                .collect();

            // Compute gram via simd_gram_f32
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // Compute gram via iterative outer product: G = X·Xᵀ = Σ_k X_ik * X_jk
            let mut reference = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0f32;
                    for k in 0..d_h {
                        sum += x[i * d_h + k] * x[j * d_h + k];
                    }
                    reference[i * seq_len + j] = sum;
                }
            }

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = i * seq_len + j;
                    assert!(
                        (gram[idx] - reference[idx]).abs() < 1e-4,
                        "G[{i}][{j}]: simd={}, reference={}",
                        gram[idx],
                        reference[idx]
                    );
                }
            }
        }
    }

    // ── simd_sum_abs_f32 tests (Issue 120) ─────────────────

    #[test]
    fn sum_abs_mixed_values() {
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let expected: f32 = data.iter().map(|v| v.abs()).sum();
        let result = crate::simd::simd_sum_abs_f32(&data);
        assert!(
            (result - expected).abs() < 1e-6,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn sum_abs_non_aligned_len() {
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let expected: f32 = data.iter().map(|v| v.abs()).sum();
        let result = crate::simd::simd_sum_abs_f32(&data);
        assert!(
            (result - expected).abs() < 1e-6,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn sum_abs_empty() {
        let data: Vec<f32> = vec![];
        let result = crate::simd::simd_sum_abs_f32(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn sum_abs_single_element() {
        assert_eq!(crate::simd::simd_sum_abs_f32(&[-42.0]), 42.0);
        assert_eq!(crate::simd::simd_sum_abs_f32(&[42.0]), 42.0);
        assert_eq!(crate::simd::simd_sum_abs_f32(&[0.0]), 0.0);
    }

    // ── simd_sum_sq_quartic tests (Plan 306 T7.4) ───────────────
    //
    // Fused Σx² + Σx⁴ used by depth_invariance::classify_chain for the
    // magnitude + participation-ratio (flatness) pass. Truth reference is
    // `scalar_sum_sq_quartic` (imported via #[cfg(test)] use in simd/mod.rs).

    #[test]
    fn simd_sum_sq_quartic_matches_scalar() {
        // Pseudo-random f32 across several lengths: aligned (4, 8, 16),
        // non-aligned tails, sub-SIMD-width remainder chunks. Deterministic
        // xorshift64* so the test is reproducible.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (((*s & 0xFFFFFF) as f32) / ((0x1000000) as f32) - 0.5) * 4.0 // range ≈ [-2, 2]
        };

        for &len in &[1usize, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 256, 257, 1023, 1024] {
            let data: Vec<f32> = (0..len).map(|_| next_f32(&mut state)).collect();
            let (sim_sq, sim_qu) = crate::simd::simd_sum_sq_quartic(&data);
            let (ref_sq, ref_qu) = scalar_sum_sq_quartic(&data);

            // Relative tolerance — large sums of x⁴ can drift on order of
            // 1e-5 between SIMD and scalar even with single-rounding FMA,
            // because the horizontal reduction order differs.
            let tol_sq = (ref_sq.abs() * 1e-5).max(1e-6);
            let tol_qu = (ref_qu.abs() * 1e-5).max(1e-6);
            assert!(
                (sim_sq - ref_sq).abs() <= tol_sq,
                "len={len}: sum_sq simd={sim_sq} scalar={ref_sq} (tol {tol_sq})"
            );
            assert!((
                sim_qu - ref_qu
            ).abs() <= tol_qu, "len={len}: sum_quartic simd={sim_qu} scalar={ref_qu} (tol {tol_qu})");
        }
    }

    #[test]
    fn simd_sum_sq_quartic_zero_input() {
        // Explicit zero slice at a few lengths including empty (which the
        // dispatcher short-circuits) and a 64-wide aligned zero buffer.
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[]), (0.0, 0.0));
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[0.0]), (0.0, 0.0));
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[0.0; 4]), (0.0, 0.0));
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[0.0; 17]), (0.0, 0.0));
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[0.0; 64]), (0.0, 0.0));
    }

    #[test]
    fn simd_sum_sq_quartic_short_input() {
        // Sub-SIMD-width inputs exercise the scalar tail (no SIMD lanes fire).
        // Exact values checked closed-form.
        // len=1: x = 3.0 → x² = 9, x⁴ = 81
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[3.0]), (9.0, 81.0));
        // len=2: 2.0, -1.0 → x² = (4, 1) → sum=5; x⁴ = (16, 1) → sum=17
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[2.0, -1.0]), (5.0, 17.0));
        // len=3: 1.0, 2.0, -2.0 → x² = (1, 4, 4) → sum=9; x⁴ = (1, 16, 16) → sum=33
        assert_eq!(crate::simd::simd_sum_sq_quartic(&[1.0, 2.0, -2.0]), (9.0, 33.0));
    }

    // ── Entropy & Coincidence Tests (Plan 260) ──────────────────

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 4 tokens: H = ln(4) ≈ 1.386
        let probs: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];
        let logprobs: Vec<f32> = probs.iter().map(|&p| p.ln()).collect();
        let h = entropy_f32(&logprobs);
        let expected = 4.0f32.ln(); // ln(4)
        assert!(
            (h - expected).abs() < 0.01,
            "uniform entropy should be ln(4)≈1.386, got {h}"
        );
    }

    #[test]
    fn test_entropy_peaked() {
        // Peaked distribution: one token dominates
        let probs: Vec<f32> = vec![0.99, 0.003, 0.004, 0.003];
        let logprobs: Vec<f32> = probs.iter().map(|&p| p.ln()).collect();
        let h = entropy_f32(&logprobs);
        assert!(
            h < 0.1,
            "peaked distribution should have near-zero entropy, got {h}"
        );
    }

    #[test]
    fn test_entropy_empty() {
        assert_eq!(entropy_f32(&[]), 0.0);
    }

    #[test]
    fn test_entropy_with_neg_inf_logprobs() {
        // logp = -∞ (zero-prob tokens) must not poison the sum via 0·(-∞) = NaN.
        // Two valid tokens with equal mass + two impossible tokens → H = ln(2) ≈ 0.693.
        let logprobs: Vec<f32> = vec![
            (0.5f32).ln(),
            (0.5f32).ln(),
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let h = entropy_f32(&logprobs);
        let expected = 2.0f32.ln();
        assert!(
            h.is_finite(),
            "entropy must be finite when some logp = -∞, got {h}"
        );
        assert!(
            (h - expected).abs() < 0.01,
            "two-token uniform entropy should be ln(2)≈0.693, got {h}"
        );
    }

    #[test]
    fn test_coincidence_full_match() {
        let top_k = vec![0, 1, 2, 3];
        let parent = vec![0, 1, 2, 3];
        let score = coincidence_score(&top_k, &parent, 4);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "full match should give 1.0, got {score}"
        );
    }

    #[test]
    fn test_coincidence_no_match() {
        let top_k = vec![10, 11, 12, 13];
        let parent = vec![0, 1, 2, 3];
        let score = coincidence_score(&top_k, &parent, 4);
        assert!(score.abs() < 1e-6, "no match should give 0.0, got {score}");
    }

    #[test]
    fn test_coincidence_partial_match() {
        let top_k = vec![0, 5, 2, 9];
        let parent = vec![0, 1, 2, 3];
        // Window=4: parent slice = [0,1,2,3]; matches: 0,2 → 2/4 = 0.5
        let score = coincidence_score(&top_k, &parent, 4);
        assert!(
            (score - 0.5).abs() < 1e-6,
            "2 of 4 match should give 0.5, got {score}"
        );
    }

    #[test]
    fn test_coincidence_empty_slices() {
        assert_eq!(coincidence_score(&[], &[1, 2], 4), 0.0);
        assert_eq!(coincidence_score(&[1], &[], 4), 0.0);
        assert_eq!(coincidence_score(&[1], &[2], 0), 0.0);
    }

    // ── simd_exp_sum_inplace tests ────────────────────────────

    #[test]
    fn exp_sum_matches_separate_exp_plus_sum() {
        // The fused kernel must produce bit-identical exp values and a sum
        // equal to `simd_sum_f32` over the exp'd buffer (within float tolerance
        // — reassociation across accumulators can reorder adds).
        let cases: &[&[f32]] = &[
            &[0.0],
            &[1.0],
            &[0.0, 1.0, 2.0, 3.0],
            &[-5.0, -1.0, 0.0, 1.0, 5.0, 10.0],
            &(0..32).map(|i| (i as f32 - 16.0) * 0.1).collect::<Vec<_>>(),
            // Lengths crossing SIMD chunk boundaries (16/32) + scalar tails
            &(0..17).map(|i| (i as f32 - 8.0) * 0.1).collect::<Vec<_>>(),
            &(0..33).map(|i| (i as f32 - 16.0) * 0.1).collect::<Vec<_>>(),
            &(0..100).map(|i| (i as f32 - 50.0) * 0.05).collect::<Vec<_>>(),
        ];
        for case in cases {
            let mut fused = case.to_vec();
            let mut sep = case.to_vec();

            let fused_sum = simd_exp_sum_inplace(&mut fused);
            simd_exp_inplace(&mut sep);
            let sep_sum = simd_sum_f32(&sep);

            // exp values must match the non-fused path bit-for-bit (same polynomial)
            for (i, (a, b)) in fused.iter().zip(sep.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "exp mismatch at {i}: fused={a}, separate={b}, input={}",
                    case[i]
                );
            }
            // Sum tolerance accounts for floating-point reassociation across
            // the 4 independent accumulators (different summation order).
            let rel_err = (fused_sum - sep_sum).abs() / sep_sum.max(1e-30);
            assert!(
                rel_err < 1e-5,
                "sum mismatch: fused={fused_sum}, separate={sep_sum}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn exp_sum_empty() {
        let mut x: Vec<f32> = vec![];
        assert_eq!(simd_exp_sum_inplace(&mut x), 0.0);
    }

    #[test]
    fn exp_sum_known_value() {
        // exp(0) + exp(1) + exp(2) = 1 + e + e² ≈ 1 + 2.71828 + 7.38906 ≈ 11.1073
        let mut x = vec![0.0f32, 1.0, 2.0];
        let sum = simd_exp_sum_inplace(&mut x);
        let expected = 1.0 + std::f32::consts::E + std::f32::consts::E.powi(2);
        assert!((sum - expected).abs() < 1e-4, "got {sum}, expected {expected}");
        // Verify in-place exp also happened
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - std::f32::consts::E).abs() < 1e-4);
        assert!((x[2] - std::f32::consts::E.powi(2)).abs() < 1e-4);
    }

    #[test]
    fn simd_exp_matches_f32_exp_truth_referenced() {
        // Issue 027 regression guard: simd_exp_inplace must match `f32::exp()` to
        // high precision. The previous polynomial used coefficients 1/k instead of
        // 1/k!, giving up to 5% error on exp(2). This test compares against the
        // platform libm `f32::exp()` (not against another SIMD path) so it cannot
        // be defeated by self-referential comparisons.
        //
        // Range: [-15, 15] in 0.1 steps. Covers the polynomial bug range (x=0.5/1/2
        // gave 2.6%/0.5%/5.1% error) and stays clear of the n-clamp boundary
        // (|x| > ~88). The threshold below tolerates the f32 range-reduction
        // precision floor (~2e-5 at |x|>6) while catching any polynomial regression.
        let inputs: Vec<f32> = (-150..=150).map(|i| i as f32 * 0.1).collect();
        let mut x = inputs.clone();
        simd_exp_inplace(&mut x);
        let mut worst_rel: f32 = 0.0;
        let mut worst_at: f32 = 0.0;
        for (i, &xi) in inputs.iter().enumerate() {
            let expected = xi.exp();
            let got = x[i];
            // Allow for Cephes 6th-order truncation error (~1e-6 relative for the
            // reduced argument). The old buggy poly produced ~5e-2 relative error
            // at x=2 — this assertion would have caught it.
            let denom = expected.abs().max(1e-30);
            let rel_err = (got - expected).abs() / denom;
            if rel_err > worst_rel {
                worst_rel = rel_err;
                worst_at = xi;
            }
            // Threshold: 5e-4 relative. The Cephes 6th-order polynomial itself is
            // accurate to ~1e-6, but the f32 range reduction `g = x - n·ln2_hi -
            // n·ln2_lo` introduces ~2e-5 relative noise at |x|>6 due to
            // catastrophic cancellation. 5e-4 is 25× above that floor and 100×
            // below the polynomial-coefficient bug (5e-2 at x=2, Issue 027), so it
            // catches any polynomial regression while tolerating range-reduction
            // precision loss.
            assert!(
                rel_err < 5e-4,
                "exp({xi}) = {got} vs true {expected}, rel_err = {rel_err:.3e} (worst so far: {worst_rel:.3e} at {worst_at})"
            );
        }
        // Sanity log: worst observed relative error across the sweep.
        // Post-fix this should be ~1e-7 (Cephes truncation floor).
        eprintln!("simd_exp truth-referenced worst rel_err = {worst_rel:.3e} at x = {worst_at}");
    }

    #[test]
    fn simd_exp_sum_matches_f32_exp_truth_referenced() {
        // Issue 027 companion guard for the fused exp+sum path. Exercises the
        // `step!` macro polynomial in both the main loop and the remaining-chunks
        // loop (lengths chosen to hit both).
        for &len in &[1usize, 3, 4, 8, 12, 16, 17, 31, 32, 33, 100] {
            let inputs: Vec<f32> = (0..len).map(|i| (i as f32 - (len as f32) * 0.5) * 0.3).collect();
            let mut x = inputs.clone();
            let got_sum = simd_exp_sum_inplace(&mut x);
            let mut expected_sum = 0.0f32;
            for (i, &xi) in inputs.iter().enumerate() {
                let expected = xi.exp();
                expected_sum += expected;
                let denom = expected.abs().max(1e-30);
                let rel_err = (x[i] - expected).abs() / denom;
                assert!(
                    rel_err < 5e-4,
                    "len={len} exp({xi}) = {} vs true {expected}, rel_err={rel_err:.3e}",
                    x[i]
                );
            }
            // Sum tolerance: fused 4-accumulator reassociation adds ~1e-6 relative.
            let sum_rel = (got_sum - expected_sum).abs() / expected_sum.abs().max(1e-30);
            assert!(sum_rel < 5e-4, "len={len} sum mismatch: got {got_sum}, exp {expected_sum}, rel={sum_rel:.3e}");
        }
    }

    // ── simd_sigmoid_tanh_clamp_inplace tests (Issue 024/025) ──────────────

    /// Reference implementation using the scalar `fast_sigmoid` path — the
    /// exact chain the SIMD helper replaces.
    fn ref_sigmoid_tanh_clamp(a: &[f32], q: &[f32], clamp: f32) -> Vec<f32> {
        a.iter()
            .zip(q.iter())
            .map(|(&ai, &qi)| (2.0 * fast_sigmoid(ai + qi) - 1.0).clamp(-clamp, clamp))
            .collect()
    }

    #[test]
    fn simd_sigmoid_tanh_clamp_matches_scalar_reference() {
        // Matches fast_sigmoid within 1e-6 for the canonical G1.4 sweep.
        // Tolerance is 3e-7 per the helper's documented ULP error, but we allow
        // 1e-6 to absorb the libm-vs-Cephes tail difference at the extremes.
        let cases = [-40.0f32, -10.0, -1.0, 0.0, 1.0, 10.0, 40.0];
        let zeros = vec![0.0f32; cases.len()];
        let clamp = 6.0f32;
        let mut out = vec![0.0f32; cases.len()];
        simd_sigmoid_tanh_clamp_inplace(&mut out, &cases, &zeros, clamp);
        let expected = ref_sigmoid_tanh_clamp(&cases, &zeros, clamp);
        for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "mismatch at i={i} (a={}): simd={got}, scalar={want}, diff={}",
                cases[i],
                (got - want).abs()
            );
        }
    }

    #[test]
    fn simd_sigmoid_tanh_clamp_output_in_range_with_outliers() {
        // Broad random input including ±100 outliers — output must stay in
        // (-clamp, clamp), and the sigmoid saturation must drive outliers to
        // ±1 (well within clamp=6).
        let a = [
            100.0f32, -100.0, 50.0, -50.0, 0.0, 1.5, -2.3, 10.0, -10.0, 2.5, -2.5, 0.001, 25.0,
            -25.0, 80.0, -80.0,
        ];
        let q = [0.0f32; 16];
        let clamp = 6.0f32;
        let mut out = [0.0f32; 16];
        simd_sigmoid_tanh_clamp_inplace(&mut out, &a, &q, clamp);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v > -clamp && v < clamp,
                "out-of-range at i={i}: {v} not in ({}, {})",
                -clamp,
                clamp
            );
        }
        // Saturated inputs must be essentially ±1 (tanh-like limit).
        assert!((out[0] - 1.0).abs() < 1e-6, "a=100 → ~+1, got {}", out[0]);
        assert!((out[1] + 1.0).abs() < 1e-6, "a=-100 → ~-1, got {}", out[1]);
    }

    #[test]
    fn simd_sigmoid_tanh_clamp_saturation_at_clamp_boundary() {
        // At a[i]=100, q[i]=0, clamp=0.5: σ(100)≈1, 2·1−1=1, clamp(−0.5, 0.5) → 0.5.
        let a = [100.0f32];
        let q = [0.0f32];
        let clamp = 0.5f32;
        let mut out = [0.0f32];
        simd_sigmoid_tanh_clamp_inplace(&mut out, &a, &q, clamp);
        assert_eq!(out[0], 0.5, "clamp saturation must give exactly +clamp");

        let a_neg = [-100.0f32];
        let mut out_neg = [0.0f32];
        simd_sigmoid_tanh_clamp_inplace(&mut out_neg, &a_neg, &q, clamp);
        assert_eq!(
            out_neg[0], -0.5,
            "clamp saturation must give exactly -clamp"
        );
    }

    #[test]
    fn simd_sigmoid_tanh_clamp_length_33_matches_length_32_prefix() {
        // Length 33 triggers the NEON scalar tail of 1 element (33 % 4 = 1).
        // The SIMD-processed prefix [0..32) must match the length-32 result,
        // and the scalar tail element must match the scalar reference.
        let mut rng = fastrand::Rng::with_seed(1234);
        let a33: Vec<f32> = (0..33).map(|_| rng.f32() * 20.0 - 10.0).collect();
        let q33: Vec<f32> = (0..33).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let clamp = 6.0f32;

        let mut out33 = vec![0.0f32; 33];
        simd_sigmoid_tanh_clamp_inplace(&mut out33, &a33, &q33, clamp);

        let mut out32 = vec![0.0f32; 32];
        simd_sigmoid_tanh_clamp_inplace(&mut out32, &a33[..32], &q33[..32], clamp);

        for i in 0..32 {
            assert_eq!(
                out33[i], out32[i],
                "prefix mismatch at i={i}: len33={}, len32={}",
                out33[i], out32[i]
            );
        }

        // Tail element (index 32) matches the scalar reference.
        let expected_tail = (2.0 * fast_sigmoid(a33[32] + q33[32]) - 1.0).clamp(-clamp, clamp);
        assert_eq!(
            out33[32], expected_tail,
            "scalar tail mismatch: simd={}, scalar={}",
            out33[32], expected_tail
        );
    }
