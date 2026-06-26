//! Plan 330 — Analytic Lattice GOAT gate (katgpt-core half, math-only).
//!
//! Pure-math GOAT tests for the analytic_lattice primitives. NO `GpuFuture`
//! — the runtime/ASOC tests (G4 latency, G1b non-blocking contract, G1c stash
//! refresh, G1d prev-tick join) are DEFERRED to riir-engine
//! (`riir-ai/crates/riir-engine/tests/analytic_lattice_runtime_goat.rs`),
//! which is the only layer with both `katgpt-core` AND `riir-gpu-async` in
//! scope (Phase 1b — separate task).
//!
//! # Gates covered here (math only)
//!
//! - **G1** determinism: `compose_chain` is bit-identical for same inputs.
//! - **G2** ranking: `direction_vector_decode` ranking matches brute-force
//!   reference within cos ≥ 0.95; `batch_compose_chain` matches per-player
//!   `compose_chain` within Frobenius ≤ 1e-6.
//! - **G3** associativity: `(A×B)×C ≈ A×(B×C)` within Frobenius ≤ 1e-5.
//! - **G6** spectral audit: known-good ≤ 5% spurious, known-bad > 5%.
//!
//! # Gates in the SIBLING test binary `analytic_lattice_alloc_check`
//!
//! - **G5** zero-alloc: `TrackingAllocator` shows 0 allocs after warmup for
//!   `compose_chain_into`, `batch_compose_chain_into`, and
//!   `direction_vector_decode`. Separated to avoid the global allocator
//!   picking up allocations from other tests in this binary running in
//!   parallel.
//!
//! # Gates DEFERRED to riir-engine (need GpuFuture)
//!
//! - **G4** latency: plasma-draft path < 100ns, hot-join path < 1µs,
//!   batched N=64 ≥ 4× vs naive.
//! - **G1b** non-blocking contract: `ComposerTick::poll` returns
//!   `Ready(stale_draft)` on `Poll::Pending`.
//! - **G1c** stash refresh: stale_draft refreshed every poll.
//! - **G1d** prev-tick join: emits reflection event on late completion.

#![cfg(feature = "analytic_lattice")]

use katgpt_core::analytic_lattice::{
    AuditReport, LatticeVector, TransportOperator, batch_compose_chain,
    batch_compose_chain_into, compose_chain, compose_chain_into,
    direction_vector_decode, spectral_audit,
};

// NOTE: the G5 zero-alloc gate lives in a SEPARATE test binary
// (`tests/analytic_lattice_alloc_check.rs`) because the `CountingAllocator`
// global would pick up allocations from other tests in this binary if they
// ran in parallel. The math-only gates (G1, G2, G3, G6) don't need the
// allocator and live here.

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_2x2(a: f32, b: f32, c: f32, d: f32) -> TransportOperator {
    TransportOperator::from_row_major(2, vec![a, b, c, d]).unwrap()
}

/// Build an operator that is diagonal in the DCT-II basis.
/// `C = Σ_m λ_m · φ_m φ_m^T` — zero cross-mode coupling by construction.
fn make_dct2_diagonal_operator(k: usize, eigenvalues: &[f32]) -> TransportOperator {
    let m = eigenvalues.len().min(k);
    let mut basis = vec![0.0f32; m * k];
    let denom = 2.0 * k as f32;
    for mode in 0..m {
        let alpha = if mode == 0 {
            (1.0 / k as f32).sqrt()
        } else {
            (2.0 / k as f32).sqrt()
        };
        for j in 0..k {
            let phase = std::f32::consts::PI * (mode as f32) * (2 * j + 1) as f32 / denom;
            basis[mode * k + j] = alpha * phase.cos();
        }
    }
    let mut data = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0f32;
            for mode in 0..m {
                s += eigenvalues[mode] * basis[mode * k + i] * basis[mode * k + j];
            }
            data[i * k + j] = s;
        }
    }
    TransportOperator::from_row_major(k, data).unwrap()
}

// ── G1: Determinism ───────────────────────────────────────────────────────

#[test]
fn g1_compose_chain_is_bit_identical() {
    let a = make_2x2(0.6, 0.1, 0.2, 0.5);
    let b = make_2x2(0.3, 0.4, 0.7, 0.2);
    let c = make_2x2(0.5, 0.3, 0.1, 0.6);

    let run1 = compose_chain(&[a.clone(), b.clone(), c.clone()]).unwrap();
    let run2 = compose_chain(&[a, b, c]).unwrap();

    // Bit-identical: same inputs → same f32 bits (not just approximately equal).
    for (x, y) in run1.as_slice().iter().zip(run2.as_slice().iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "G1 FAIL: compose_chain not bit-identical ({} vs {})",
            x,
            y
        );
    }
}

#[test]
fn g1_compose_chain_into_is_bit_identical() {
    let a = make_2x2(0.7, 0.2, 0.1, 0.8);
    let b = make_2x2(0.4, 0.5, 0.6, 0.3);

    let mut scratch1 = Vec::new();
    let mut out1 = TransportOperator::zeros(2);
    let mut scratch2 = Vec::new();
    let mut out2 = TransportOperator::zeros(2);

    compose_chain_into(&[a.clone(), b.clone()], &mut scratch1, &mut out1).unwrap();
    compose_chain_into(&[a, b], &mut scratch2, &mut out2).unwrap();

    for (x, y) in out1.as_slice().iter().zip(out2.as_slice().iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "G1 FAIL: compose_chain_into not bit-identical");
    }
}

#[test]
fn g1_direction_vector_decode_is_bit_identical() {
    let state = LatticeVector::<8>::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let dir = LatticeVector::<8>::new([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
    let a = direction_vector_decode(&state, &dir, 1.5);
    let b = direction_vector_decode(&state, &dir, 1.5);
    assert_eq!(a.to_bits(), b.to_bits(), "G1 FAIL: decode not bit-identical");
}

// ── G2: Ranking preservation ──────────────────────────────────────────────

#[test]
fn g2_decoder_ranking_matches_reference_cos_ge_095() {
    // 100 random states × fixed direction, verify SIMD decode ranking matches
    // brute-force reference within cos ≥ 0.95.
    use katgpt_core::simd::fast_sigmoid;

    let direction = LatticeVector::<8>::new([
        0.31, -0.42, 0.55, 0.19, -0.67, 0.83, -0.11, 0.47,
    ]);

    // Deterministic pseudo-random states (LCG — no external rand dep).
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (1u64 << 31) as f32 * 2.0 - 1.0
    };

    let mut simd_scores: Vec<f32> = Vec::with_capacity(100);
    let mut ref_scores: Vec<f32> = Vec::with_capacity(100);

    for _ in 0..100 {
        let state = LatticeVector::<8>::new([
            rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng(),
        ]);
        simd_scores.push(direction_vector_decode(&state, &direction, 1.0));

        // Reference: brute-force dot / N, then sigmoid.
        let z = state
            .as_slice()
            .iter()
            .zip(direction.as_slice())
            .map(|(s, d)| s * d)
            .sum::<f32>()
            / 8.0;
        ref_scores.push(fast_sigmoid(z));
    }

    // Cosine similarity between the two score vectors.
    let dot: f32 = simd_scores.iter().zip(ref_scores.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = simd_scores.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = ref_scores.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (norm_a * norm_b);

    assert!(
        cos >= 0.95,
        "G2 FAIL: decoder ranking cos {} < 0.95",
        cos
    );
}

#[test]
fn g2_batch_compose_matches_naive_frobenius_le_1e6() {
    // For 100 random (prefix, suffix_i) sets, batched output matches per-player
    // compose_chain within Frobenius ≤ 1e-6.
    let mut seed: u64 = 42;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (1u64 << 31) as f32 * 0.8 - 0.4 // range [-0.4, 0.4)
    };

    let k = 4;
    let mut max_err = 0.0f32;

    for _ in 0..100 {
        // Build a random 2-op prefix + random suffix.
        let prefix: Vec<TransportOperator> = (0..2)
            .map(|_| {
                let data: Vec<f32> = (0..k * k).map(|_| rng()).collect();
                TransportOperator::from_row_major(k, data).unwrap()
            })
            .collect();

        // 3 random suffixes (players).
        let suffixes_owned: Vec<Vec<TransportOperator>> = (0..3)
            .map(|_| {
                let data: Vec<f32> = (0..k * k).map(|_| rng()).collect();
                vec![TransportOperator::from_row_major(k, data).unwrap()]
            })
            .collect();
        let suffixes: Vec<&[TransportOperator]> =
            suffixes_owned.iter().map(|s| s.as_slice()).collect();

        // Batched.
        let mut batched = vec![
            TransportOperator::zeros(k),
            TransportOperator::zeros(k),
            TransportOperator::zeros(k),
        ];
        let mut scratch = Vec::new();
        batch_compose_chain(&prefix, &suffixes, &mut batched, &mut scratch).unwrap();

        // Naive per-player.
        for (i, suffix) in suffixes.iter().enumerate() {
            let mut chain: Vec<&TransportOperator> = prefix.iter().collect();
            chain.extend(suffix.iter());
            let naive = compose_chain(&chain.iter().cloned().cloned().collect::<Vec<_>>()).unwrap();

            let err: f32 = batched[i]
                .as_slice()
                .iter()
                .zip(naive.as_slice())
                .map(|(b, n)| (b - n).abs())
                .sum();
            max_err = max_err.max(err);
        }
    }

    assert!(
        max_err < 1e-6,
        "G2 FAIL: batch vs naive max Frobenius err {} >= 1e-6",
        max_err
    );
}

#[test]
fn g2_batch_compose_into_matches_typed() {
    // The raw-slice variant must agree with the typed variant on a shared case.
    let k = 3;
    let n = 4;

    // Build prefix = identity (so typed batch gives suffix × I = suffix).
    let prefix_typed = vec![TransportOperator::identity(k)];
    let suffixes_data: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..k * k)
                .map(|j| (i as f32 * 0.1 + j as f32 * 0.01))
                .collect()
        })
        .collect();
    let suffixes_typed: Vec<TransportOperator> = suffixes_data
        .iter()
        .map(|d| TransportOperator::from_row_major(k, d.clone()).unwrap())
        .collect();
    let suffix_slices: Vec<&[TransportOperator]> =
        suffixes_typed.iter().map(|s| std::slice::from_ref(s)).collect();

    let mut typed_out = vec![TransportOperator::zeros(k); n];
    let mut scratch = Vec::new();
    batch_compose_chain(&prefix_typed, &suffix_slices, &mut typed_out, &mut scratch).unwrap();

    // Raw-slice variant.
    let prefix_flat: Vec<f32> = (0..k * k)
        .map(|i| if i % (k + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let suffixes_flat: Vec<f32> = suffixes_data.iter().flatten().copied().collect();
    let mut raw_out = vec![0.0f32; n * k * k];
    batch_compose_chain_into(&prefix_flat, &suffixes_flat, &mut raw_out, k, n);

    for i in 0..n {
        for j in 0..k * k {
            let t = typed_out[i].as_slice()[j];
            let r = raw_out[i * k * k + j];
            assert!(
                (t - r).abs() < 1e-6,
                "G2 FAIL: typed[{}][{}] = {} vs raw = {}",
                i, j, t, r
            );
        }
    }
}

// ── G3: Associativity ─────────────────────────────────────────────────────

#[test]
fn g3_associativity_frobenius_le_1e5() {
    // (A×B)×C ≈ A×(B×C) within Frobenius ≤ 1e-5.
    // Use well-conditioned small-norm operators (entries in [-0.5, 0.5]).
    let mut seed: u64 = 12345;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (1u64 << 31) as f32 * 1.0 - 0.5
    };

    let k = 4;
    let mut mk = || -> TransportOperator {
        let data: Vec<f32> = (0..k * k).map(|_| rng()).collect();
        TransportOperator::from_row_major(k, data).unwrap()
    };

    let mut max_err = 0.0f32;
    for _ in 0..50 {
        let a = mk();
        let b = mk();
        let c = mk();

        // (A×B)×C
        let ab = compose_chain(&[a.clone(), b.clone()]).unwrap();
        let left = compose_chain(&[ab, c.clone()]).unwrap();

        // A×(B×C)
        let bc = compose_chain(&[b, c]).unwrap();
        let right = compose_chain(&[a, bc]).unwrap();

        let err: f32 = left
            .as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(l, r)| (l - r).abs())
            .sum();
        max_err = max_err.max(err);
    }

    assert!(
        max_err < 1e-5,
        "G3 FAIL: associativity max Frobenius err {} >= 1e-5",
        max_err
    );
}

// ── G6: Spectral audit ────────────────────────────────────────────────────

#[test]
fn g6_known_good_composite_le_5pct_spurious() {
    // A DCT-II-diagonal composite (clean spectral transport) should have
    // spurious coupling ≤ 5%.
    let a = make_dct2_diagonal_operator(8, &[1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65]);
    let b = make_dct2_diagonal_operator(8, &[0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]);
    let composite = compose_chain(&[a, b]).unwrap();
    let report: AuditReport = spectral_audit(&composite);

    assert!(
        report.spurious_ratio <= 0.05,
        "G6 FAIL: known-good spurious {} > 5%",
        report.spurious_ratio
    );
}

#[test]
fn g6_known_bad_random_gt_5pct_spurious() {
    // A dense random operator should have spurious coupling > 5%.
    let mut seed: u64 = 31415;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (1u64 << 31) as f32 * 2.0 - 1.0
    };

    let k = 8;
    let data: Vec<f32> = (0..k * k).map(|_| rng()).collect();
    let op = TransportOperator::from_row_major(k, data).unwrap();
    let report = spectral_audit(&op);

    assert!(
        report.spurious_ratio > 0.05,
        "G6 FAIL: known-bad spurious {} <= 5% (should be higher)",
        report.spurious_ratio
    );
}
