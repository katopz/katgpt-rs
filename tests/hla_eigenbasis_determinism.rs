//! Issue 001 — HLA Windowed Eigenbasis Recovery determinism test.
//!
//! The primitive's seed is the deterministic `1/sqrt(D)` vector (no RNG), and
//! the only cross-platform variability is the SIMD reduction order inside
//! `simd_dot_f32` / `simd_outer_product_acc`. This test verifies the
//! **same-machine, same-binary** determinism contract (two calls → bit-identical
//! outputs) and documents the **cross-platform** verification protocol.
//!
//! # Cross-platform verification protocol (G2 full claim)
//!
//! The issue's G2 gate requires bit-identical eigenvalues across `x86_64`,
//! `aarch64`, and `wasm32`. This cannot be checked from a single test binary —
//! it requires building the crate per target and diffing the output. To verify:
//!
//! ```bash
//! # 1. Build a small harness binary that runs recover_eigenbasis_from_window_fast
//! #    on a fixed window and prints the eigenvalues' bits to stdout.
//! # 2. Build for each target:
//! cargo build --release --features hla_eigenbasis_recovery --target x86_64-apple-darwin
//! cargo build --release --features hla_eigenbasis_recovery --target aarch64-apple-darwin
//! cargo build --release --features hla_eigenbasis_recovery --target wasm32-unknown-unknown
//! # 3. Run each, diff the bit patterns. 0 diffs = G2 cross-platform PASS.
//! ```
//!
//! The determinism surface is the same one `stable_rank_update_into`
//! (`katgpt-core/src/data_probe.rs`) already relies on for its cross-platform
//! claim — `simd_dot_f32` / `simd_outer_product_acc` dispatch to NEON / AVX2 /
//! wasm32-simd128 / scalar, and the scalar fallback uses `f32::mul_add` (single
//! rounding) to match the SIMD FMA path. If `stable_rank_update_into` is
//! cross-platform bit-identical on this host, so is this primitive.

#![cfg(feature = "hla_eigenbasis_recovery")]

use katgpt_rs::hla_eigenbasis::{
    EigenbasisScratch, EigenbasisTracker, recover_eigenbasis_from_window_fast,
};
use std::hint::black_box;

/// Fixed canonical test window — same inputs across every run of this test on
/// every platform. 128 ticks × 8 dims, deterministic structure (rank-leaning).
fn canonical_window() -> Vec<f32> {
    let (t, d) = (128usize, 8usize);
    let mut w = vec![0.0_f32; t * d];
    let mut s = 0x1234_5678_9abc_def0u64;
    for r in 0..t {
        let dom = r % 3;
        for j in 0..d {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = if j == dom { 1.0 } else { 0.0 };
            let noise = (s >> 33) as f32 / (1u64 << 31) as f32 * 0.02;
            w[r * d + j] = base + noise;
        }
    }
    w
}

#[test]
fn fast_path_is_deterministic_within_binary() {
    // Two independent recoveries on the same input must produce bit-identical
    // eigenvectors and eigenvalues. This is the within-binary half of G2.
    let window = canonical_window();
    let (t, d, k) = (128usize, 8usize, 4usize);

    let mut a = vec![0.0; d * k];
    let mut la = vec![0.0; k];
    let mut b = vec![0.0; d * k];
    let mut lb = vec![0.0; k];
    let mut sa = EigenbasisScratch::new();
    let mut sb = EigenbasisScratch::new();

    recover_eigenbasis_from_window_fast(
        black_box(&window), t, d, black_box(&mut a), black_box(&mut la), &mut sa, k, 5,
    );
    recover_eigenbasis_from_window_fast(
        black_box(&window), t, d, black_box(&mut b), black_box(&mut lb), &mut sb, k, 5,
    );

    for i in 0..a.len() {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "eigvec bit mismatch at {i}");
    }
    for i in 0..la.len() {
        assert_eq!(la[i].to_bits(), lb[i].to_bits(), "eigval bit mismatch at {i}");
    }
}

#[test]
fn tracker_is_deterministic_within_binary() {
    // The incremental tracker path must also be deterministic given identical
    // push order.
    let window = canonical_window();
    let (t, d, k) = (128usize, 8usize, 4usize);

    let run = || -> (Vec<f32>, Vec<f32>) {
        let mut tr = EigenbasisTracker::new(t, d);
        for r in 0..t {
            tr.push_tick(&window[r * d..(r + 1) * d]);
        }
        let mut ev = vec![0.0; d * k];
        let mut el = vec![0.0; k];
        tr.recover(&mut ev, &mut el, k, 5);
        (ev, el)
    };

    let (a, la) = run();
    let (b, lb) = run();
    for i in 0..a.len() {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "tracker eigvec bit mismatch at {i}");
    }
    for i in 0..la.len() {
        assert_eq!(la[i].to_bits(), lb[i].to_bits(), "tracker eigval bit mismatch at {i}");
    }
}

#[test]
fn seed_is_independent_of_input_scale() {
    // Doubling the window scales eigenvalues by 4 (G = W^T W; (2W)^T(2W) = 4G)
    // but must NOT change the eigenvector directions (they are scale-invariant).
    // This guards against any input-dependent seeding sneaking in.
    let window = canonical_window();
    let (t, d, k) = (128usize, 8usize, 2usize);
    let doubled: Vec<f32> = window.iter().map(|x| 2.0 * x).collect();

    let mut a = vec![0.0; d * k];
    let mut la = vec![0.0; k];
    let mut b = vec![0.0; d * k];
    let mut lb = vec![0.0; k];
    let mut sa = EigenbasisScratch::new();
    let mut sb = EigenbasisScratch::new();
    recover_eigenbasis_from_window_fast(&window, t, d, &mut a, &mut la, &mut sa, k, 5);
    recover_eigenbasis_from_window_fast(&doubled, t, d, &mut b, &mut lb, &mut sb, k, 5);

    // Eigenvalues: doubled window → 4× the eigenvalues.
    for i in 0..k {
        let ratio = lb[i] / la[i];
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "scale test: eigval ratio {} != 4.0 (la={}, lb={})",
            ratio,
            la[i],
            lb[i]
        );
    }
    // Eigenvectors: |cos| between corresponding directions ≈ 1 (sign may flip).
    for col in 0..k {
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for row in 0..d {
            dot += a[row * k + col] * b[row * k + col];
            na += a[row * k + col] * a[row * k + col];
            nb += b[row * k + col] * b[row * k + col];
        }
        let cos = (dot / (na.sqrt() * nb.sqrt())).abs();
        assert!(cos > 0.999, "scale test: col {col} direction cos {cos} < 0.999");
    }
}
