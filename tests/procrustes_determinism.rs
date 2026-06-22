//! Determinism test for `orthogonal_procrustes` (Issue 001 G3).
//!
//! Verifies that the SAME anchor pair produces a BIT-IDENTICAL rotation
//! matrix across repeated calls. The cross-platform aspect (x86_64 vs
//! aarch64 vs wasm32) requires running this test on each target — the
//! invariant this test enforces locally is "same input → same output",
//! which is the precondition for cross-platform determinism (if it
//! fails locally, cross-platform bit-identity is impossible).
//!
//! Run:
//! ```bash
//! cargo test --features orthogonal_procrustes --test procrustes_determinism
//! ```

#![cfg(feature = "orthogonal_procrustes")]

use katgpt_rs::procrustes::{
    orthogonal_procrustes, ProcrustesConfig, ProcrustesScratch,
};

/// Deterministic xorshift32 PRNG.
fn seeded_anchors(seed: u32, n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * d];
    let mut state = seed.max(1);
    for v in out.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *v = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
    out
}

#[test]
fn g3_bit_identical_output_across_repeated_calls() {
    // Same input → same R, bit-identical, across multiple calls + scratch
    // instances. This is the precondition for cross-platform determinism
    // (if this fails, the algorithm has internal non-determinism, e.g.
    // thread-local state, RDSEED, etc.).
    let cases: &[(usize, usize, u32, u32)] = &[
        (32, 4, 1, 100),
        (128, 8, 7, 200),
        (256, 16, 42, 999),
        (512, 32, 1234, 5678),
    ];

    for &(n, d, seed_a, seed_b) in cases {
        let a = seeded_anchors(seed_a, n, d);
        let b = seeded_anchors(seed_b, n, d);
        let mut r1 = vec![0.0_f32; d * d];
        let mut r2 = vec![0.0_f32; d * d];
        let mut s1 = ProcrustesScratch::new(n, d);
        let mut s2 = ProcrustesScratch::new(n, d);

        let cfg = ProcrustesConfig::default();
        let rep1 = orthogonal_procrustes(&a, &b, n, d, &mut r1, &mut s1, &cfg).unwrap();
        let rep2 = orthogonal_procrustes(&a, &b, n, d, &mut r2, &mut s2, &cfg).unwrap();

        assert_eq!(r1, r2, "G3 FAIL: n={}, d={}: rotation matrix differs between calls", n, d);
        // Reports should also be bit-identical.
        assert_eq!(rep1.n, rep2.n);
        assert_eq!(rep1.d, rep2.d);
        assert!(rep1.residual.to_bits() == rep2.residual.to_bits(),
                "G3 FAIL: residual bits differ: {} vs {}", rep1.residual, rep2.residual);
        assert!(rep1.m_norm.to_bits() == rep2.m_norm.to_bits(),
                "G3 FAIL: m_norm bits differ: {} vs {}", rep1.m_norm, rep2.m_norm);
    }
}

#[test]
fn g3_deterministic_with_zero_seed_guards() {
    // The PRNG has a fixed point at 0. Verify the algorithm doesn't
    // propagate NaN/Inf if anchors happen to be all-zero (it should
    // return DegenerateAnchors, not produce a non-deterministic result).
    let n = 16;
    let d = 4;
    let a = vec![0.0_f32; n * d];
    let b = vec![0.0_f32; n * d];
    let mut r = vec![0.0_f32; d * d];
    let mut scratch = ProcrustesScratch::new(n, d);
    let cfg = ProcrustesConfig::default();

    let result = orthogonal_procrustes(&a, &b, n, d, &mut r, &mut scratch, &cfg);
    assert!(result.is_err(), "expected DegenerateAnchors error");
    // Determinism: same input always errors. No NaN leakage.
    for v in &r {
        assert!(v.is_finite() || *v == 0.0, "buffer should not contain NaN/Inf after error");
    }
}

#[test]
fn g3_scratch_reuse_does_not_break_determinism() {
    // Same scratch reused across DIFFERENT inputs should still give the
    // same R for each input as if a fresh scratch were used.
    let n = 64;
    let d = 4;
    let a = seeded_anchors(11, n, d);
    let b = seeded_anchors(22, n, d);

    let mut r_reused = vec![0.0_f32; d * d];
    let mut r_fresh = vec![0.0_f32; d * d];
    let mut scratch_reused = ProcrustesScratch::new(n, d);

    // First call: warm up the scratch with different data.
    let a_warmup = seeded_anchors(99, n, d);
    let b_warmup = seeded_anchors(88, n, d);
    let mut r_warmup = vec![0.0_f32; d * d];
    let _ = orthogonal_procrustes(&a_warmup, &b_warmup, n, d, &mut r_warmup, &mut scratch_reused, &ProcrustesConfig::default());

    // Second call: this is what we compare.
    let _ = orthogonal_procrustes(&a, &b, n, d, &mut r_reused, &mut scratch_reused, &ProcrustesConfig::default());

    // Fresh scratch + same input.
    let mut scratch_fresh = ProcrustesScratch::new(n, d);
    let _ = orthogonal_procrustes(&a, &b, n, d, &mut r_fresh, &mut scratch_fresh, &ProcrustesConfig::default());

    assert_eq!(r_reused, r_fresh,
               "G3 FAIL: scratch reuse changed the result — internal state leaked");
}
