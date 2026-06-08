//! GOAT benchmark for Channel SIMD Alignment (Plan 227 Phase 5).
//!
//! Measures: SIMD throughput (ops/sec), padding overhead, cache-line alignment.

use katgpt_rs::channel_simd::AlignedWeightMatrix;

#[test]
fn test_matvec_throughput() {
    // Realistic size: 512×512 weight matrix
    let dim = 512;
    let rows: Vec<Vec<f32>> = (0..dim)
        .map(|r| {
            (0..dim)
                .map(|c| ((r * dim + c) as f32 * 0.001 - 0.256).sin())
                .collect()
        })
        .collect();

    let mat = AlignedWeightMatrix::from_rows(&rows);
    let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Warmup
    for _ in 0..10 {
        let _ = mat.matvec(&x);
    }

    let start = std::time::Instant::now();
    let iters = 1000;
    let mut result = vec![0.0f32; dim];
    for _ in 0..iters {
        mat.matvec_into(&x, &mut result);
    }
    let elapsed = start.elapsed();

    let us = elapsed.as_secs_f64() * 1e6;
    let us_per_matvec = us / iters as f64;
    eprintln!("Aligned matvec {dim}×{dim}: {us_per_matvec:.1}μs each ({iters} iters in {us:.0}μs)");

    // Sanity: should be < 10ms per matvec for 512×512
    assert!(
        elapsed.as_secs() < 10,
        "matvec too slow: {us_per_matvec:.1}μs"
    );
    assert!(!result.iter().all(|&v| v == 0.0));
}

#[test]
fn test_padding_overhead_acceptable() {
    for dim in [64, 128, 256, 512, 1024] {
        let rows: Vec<Vec<f32>> = vec![vec![1.0; dim]];
        let mat = AlignedWeightMatrix::from_rows(&rows);
        let overhead = mat.padding_overhead();

        eprintln!("dim={dim}: padding overhead={:.1}%", overhead * 100.0);

        // Padding should be < 100% (i.e., at most doubles the size)
        assert!(
            overhead < 1.0,
            "padding overhead too high for dim={dim}: {:.1}%",
            overhead * 100.0
        );
    }
}

#[test]
fn test_alignment_correctness() {
    let rows: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    let mat = AlignedWeightMatrix::from_rows(&rows);

    let x = vec![1.0, 0.0, 0.0, 0.0];
    let y = mat.matvec(&x);

    assert!((y[0] - 1.0).abs() < 1e-6);
    assert!((y[1] - 5.0).abs() < 1e-6);
}

#[test]
fn test_quantize_dequantize_roundtrip() {
    let original = vec![1.5, -2.3, 0.0, 4.7, -1.1, 3.2, 0.5, -0.8];
    let rows = vec![original.clone()];
    let mut mat = AlignedWeightMatrix::from_rows(&rows);

    // Quantize
    mat.quantize_row(0, &original);

    // Dequantize
    let mut recovered = vec![0.0; original.len()];
    mat.dequantize_row(0, &mut recovered);

    for (a, b) in original.iter().zip(recovered.iter()) {
        assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {a} vs {b}");
    }
}

#[test]
fn test_from_ternary() {
    // Simple 2×8 ternary weight matrix
    let rows = 2;
    let cols = 8;
    let blocks64 = 1;

    let mut pos_bits = vec![0u64; rows * blocks64];
    let mut neg_bits = vec![0u64; rows * blocks64];
    let row_scale = vec![1.0f32; rows];

    // Row 0: [+1, -1, 0, +1, -1, 0, +1, -1]
    pos_bits[0] = (1u64 << 0) | (1u64 << 3) | (1u64 << 6); // cols 0, 3, 6 are +1
    neg_bits[0] = (1u64 << 1) | (1u64 << 4) | (1u64 << 7); // cols 1, 4, 7 are -1

    // Row 1: all zeros
    pos_bits[1] = 0;
    neg_bits[1] = 0;

    let mat =
        AlignedWeightMatrix::from_ternary(&pos_bits, &neg_bits, &row_scale, rows, cols, blocks64);

    assert_eq!(mat.num_rows, 2);
    assert_eq!(mat.row_dim, 8);

    // Check row 0 values
    let row = mat.row(0);
    assert!((row[0] - 1.0).abs() < 1e-6, "col 0 should be +1");
    assert!((row[1] - (-1.0)).abs() < 1e-6, "col 1 should be -1");
    assert!((row[2] - 0.0).abs() < 1e-6, "col 2 should be 0");
    assert!((row[3] - 1.0).abs() < 1e-6, "col 3 should be +1");

    // Row 1 should be all zeros
    for c in 0..cols {
        assert!(
            (mat.row(1)[c] - 0.0).abs() < 1e-6,
            "row 1 col {c} should be 0"
        );
    }
}

#[test]
fn test_matvec_into_vs_matvec() {
    let rows: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let mat = AlignedWeightMatrix::from_rows(&rows);
    let x = vec![2.0, 3.0];

    let y1 = mat.matvec(&x);
    let mut y2 = vec![0.0; 2];
    mat.matvec_into(&x, &mut y2);

    for (a, b) in y1.iter().zip(y2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_vs_unaligned_throughput() {
    let dim = 256;
    let data: Vec<Vec<f32>> = (0..dim)
        .map(|r| {
            (0..dim)
                .map(|c| ((r * dim + c) as f32 * 0.001).sin())
                .collect()
        })
        .collect();

    let aligned = AlignedWeightMatrix::from_rows(&data);
    let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Warmup
    for _ in 0..10 {
        let _ = aligned.matvec(&x);
    }

    // Aligned matvec
    let iters = 5000;
    let start = std::time::Instant::now();
    let mut out = vec![0.0f32; dim];
    for _ in 0..iters {
        aligned.matvec_into(&x, &mut out);
    }
    let aligned_time = start.elapsed();

    // Unaligned (naive) matvec
    let start = std::time::Instant::now();
    let mut out2 = vec![0.0f32; dim];
    for _ in 0..iters {
        for (r, row) in data.iter().enumerate() {
            out2[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
    }
    let unaligned_time = start.elapsed();

    let aligned_us = aligned_time.as_secs_f64() * 1e6 / iters as f64;
    let unaligned_us = unaligned_time.as_secs_f64() * 1e6 / iters as f64;

    eprintln!(
        "dim={dim}: aligned={aligned_us:.1}μs vs unaligned={unaligned_us:.1}μs (ratio={:.2}x)",
        unaligned_us / aligned_us
    );

    // Results should be identical
    for (a, b) in out.iter().zip(out2.iter()) {
        assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
    }
}
