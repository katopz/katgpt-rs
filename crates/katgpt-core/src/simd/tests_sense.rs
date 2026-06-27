    use super::*;
    use crate::types::TernaryDir;

    #[test]
    fn test_ternary_dot_matches_scalar() {
        let state = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let dir = TernaryDir {
            pos_bits: 0b0101, // indices 0, 2 are positive
            neg_bits: 0b1000, // index 3 is negative
            row_scale: 0.5,
        };
        let result = simd_ternary_dot_f32(&state, &dir);
        // +1*1.0 + 0*2.0 + 1*3.0 + (-1)*4.0 = 0.0, scaled by 0.5 = 0.0
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_dot_all_positive() {
        let state = [1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dir = TernaryDir {
            pos_bits: 0b11,
            neg_bits: 0,
            row_scale: 1.0,
        };
        let result = simd_ternary_dot_f32(&state, &dir);
        assert!((result - 3.0).abs() < 1e-6);
    }

    // ── project_ternary_simd tests (Plan 008 Step 7c — ported from riir-engine) ──
    // These run the scalar fallback on native and the WASM SIMD128 SWAR kernel
    // on `wasm32 +simd128`. The scalar path is the bit-exact reference; the
    // WASM path was proven equivalent in Plan 286 T6.
    #[cfg(feature = "plasma_path")]
    #[test]
    fn test_project_ternary_simd_all_positive() {
        let result =
            unsafe { project_ternary_simd(&[1.0f32; 16], &[0xFFu8; 2], &[0x00u8; 2], 16) };
        assert_eq!(result, 16.0);
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn test_project_ternary_simd_all_negative() {
        let result =
            unsafe { project_ternary_simd(&[1.0f32; 16], &[0x00u8; 2], &[0xFFu8; 2], 16) };
        assert_eq!(result, -16.0);
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn test_project_ternary_simd_mixed() {
        // bits: 0x0F = lower 4 set, 0xF0 = upper 4 set
        let result = unsafe { project_ternary_simd(&[1.0f32; 8], &[0x0Fu8], &[0xF0u8], 8) };
        assert_eq!(result, 0.0); // 4×(+1) + 4×(-1) = 0
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn test_project_ternary_simd_zeros() {
        let result = unsafe { project_ternary_simd(&[1.0f32; 8], &[0x00u8], &[0x00u8], 8) };
        assert_eq!(result, 0.0);
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn test_project_ternary_simd_4_element_remainder() {
        // n=12: 8-element chunk + 4-element remainder. Confirms the remainder path.
        let input = [1.0f32; 12];
        let pos_bits = [0xFFu8; 2]; // 16 bits all positive
        let neg_bits = [0x00u8; 2];
        let result = unsafe { project_ternary_simd(&input, &pos_bits, &neg_bits, 12) };
        assert_eq!(result, 12.0);
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn test_project_ternary_simd_scalar_tail() {
        // n=11: 8-element chunk + 3-element scalar tail.
        let input = [1.0f32; 11];
        let pos_bits = [0xFFu8; 2];
        let neg_bits = [0x00u8; 2];
        let result = unsafe { project_ternary_simd(&input, &pos_bits, &neg_bits, 11) };
        assert_eq!(result, 11.0);
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn test_project_ternary_simd_64_dim_varied() {
        // The benchmark dimension: 64 elements, 8 bytes per bit-plane.
        // Pattern: alternate +1, -1, 0, 0 across all 64 lanes.
        let input: Vec<f32> = (0..64).map(|i| (i as f32) + 1.0).collect();
        let mut pos_bits = vec![0u8; 8];
        let mut neg_bits = vec![0u8; 8];
        // Reference scalar computation
        let mut expected = 0.0f32;
        for i in 0..64 {
            let byte = i / 8;
            let bit = i % 8;
            if i % 3 == 0 {
                pos_bits[byte] |= 1 << bit;
                expected += input[i];
            } else if i % 3 == 1 {
                neg_bits[byte] |= 1 << bit;
                expected -= input[i];
            }
            // i % 3 == 2 → zero weight, no contribution
        }
        let result = unsafe { project_ternary_simd(&input, &pos_bits, &neg_bits, 64) };
        assert!((result - expected).abs() < 1e-4, "got {result}, expected {expected}");
    }
