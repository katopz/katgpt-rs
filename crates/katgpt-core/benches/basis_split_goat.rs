//! basis_split — GOAT gate benchmark (Plan 625 Phase 2).
//!
//! Groups:
//! - **`g1_planted_recovery`** — planted-block recovery by principal angles at
//!   d ∈ {16, 64, 256}; recovery asserted (all cosines > cos(1e-3)) so a
//!   regression FAILS the bench, not just slows it.
//! - **`g2_split_latency`** — full split (accumulate + whitener + eigensolve +
//!   gate + basis) wall-clock at d=64/N=2048 (the T2.2 bar: < 10 ms at
//!   --release; quote `bench_preflight` PROVENANCE beside the number).
//! - **`adversarial_gates`** — the canaries: shuffled pairs and degenerate
//!   pairs must emit `Unidentified` (a gate that never reds certifies
//!   nothing), and the T0.3 TILR-composition row runs the same fixture
//!   through the substrate's `discover_invariant_subspace` for the A/B
//!   baseline.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features basis_split --bench basis_split_goat
//! ```
//!
//! # Feature gate
//!
//! Requires `basis_split` (Phase 1 primitive) + `subspace_phase_gate`
//! (the TILR comparison row + thin_svd for the principal angles).

#![cfg(all(feature = "basis_split", feature = "subspace_phase_gate"))]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::basis_split::{
    PairDifferenceMoment, Verdict, Whitening, project_invariant_into, split,
};
use katgpt_core::linalg::symmetric_eig::SymmetricEigScratch;
use katgpt_core::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

/// Deterministic LCG + Box–Muller (the unseeded-global-RNG gate).
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

/// Planted world: K invariant dims (difference exactly 0), D−K variant dims
/// standard normal, axis-aligned coordinates.
fn planted_differences(d: usize, k: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            let mut delta = vec![0.0; d];
            for (i, dv) in delta.iter_mut().enumerate() {
                if i >= k {
                    *dv = rng.next_gauss();
                }
            }
            delta
        })
        .collect()
}

fn identity_sigma(d: usize) -> Vec<f64> {
    let mut s = vec![0.0; d * d];
    for i in 0..d {
        s[i * d + i] = 1.0;
    }
    s
}

/// Principal-angle cosines between the true invariant block (first k
/// coordinate axes) and the complement of the variant basis: the singular
/// values of `(I − BᵀB)·V_true`.
fn principal_angle_cosines(basis: &[f64], k: usize, d: usize, true_k: usize) -> Vec<f32> {
    let mut proj = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut dot = 0.0;
            for row in 0..k {
                dot += basis[row * d + i] * basis[row * d + j];
            }
            proj[i * d + j] = if i == j { 1.0 - dot } else { -dot };
        }
    }
    let mut pv = vec![0.0f32; d * true_k];
    for i in 0..d {
        for j in 0..true_k {
            let mut acc = 0.0f32;
            for t in 0..d {
                acc += (proj[i * d + t] as f32) * pv_axis(t, j);
            }
            pv[i * true_k + j] = acc;
        }
    }
    let mut work = SvdScratch::with_capacity(true_k, d);
    let mut result = SvdResultScratch::with_capacity(d, true_k);
    thin_svd_into(&pv, d, true_k, &mut result, &mut work);
    result.singular_values().to_vec()
}

/// True invariant basis entry [t, j] (first `true_k` coordinate axes).
fn pv_axis(t: usize, j: usize) -> f32 {
    if t == j { 1.0 } else { 0.0 }
}

/// One full G1 recovery check at (d, k); panics on failure (the bench FAILS,
/// not just slows).
fn assert_planted_recovery(d: usize, k: usize, n: usize) {
    let diffs = planted_differences(d, k, n, 0x6255_6101 + d as u64);
    let mut m = vec![0.0; d * d];
    {
        let mut acc = PairDifferenceMoment::new(&mut m, d).unwrap();
        for delta in &diffs {
            acc.accumulate(delta).unwrap();
        }
        acc.finish();
    }
    let mut eigvals = vec![0.0; d];
    let mut eigvecs = vec![0.0; d * d];
    let mut basis = vec![0.0; (d - k) * d];
    let mut sym = SymmetricEigScratch::new();
    let mut jacobi = vec![0.0; d * d];
    let mut m_scr = vec![0.0; d * d];
    let mut whitener = vec![0.0; d * d];
    let sigma = identity_sigma(d);
    let (verdict, report) = split(
        &m,
        d,
        Some(Whitening::TotalCovariance(&sigma)),
        Some(d - k),
        10.0,
        &mut eigvals,
        &mut eigvecs,
        &mut basis,
        &mut sym,
        &mut jacobi,
        &mut m_scr,
        &mut whitener,
    )
    .unwrap();
    assert!(
        matches!(verdict, Verdict::SufficientVariation { .. }),
        "d={d} k={k}: planted world must split, got {verdict:?} (gap {})",
        report.gap_ratio
    );
    let cosines = principal_angle_cosines(&basis, d - k, d, k);
    let min_cos = cosines.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        min_cos > 1e-3f32.cos(),
        "d={d}: G1 principal-angle recovery FAIL: min cos {min_cos}"
    );
}

// ── G1: planted recovery at three dimensions (asserted, not timed) ────────────

fn bench_g1_planted_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("basis_split/g1_planted_recovery");
    for (d, k) in [(16usize, 4usize), (64, 16), (256, 64)] {
        group.bench_function(format!("d{d}_k{k}"), |b| {
            b.iter(|| assert_planted_recovery(black_box(d), black_box(k), 2048))
        });
    }
    group.finish();
}

// ── G2: split latency at d=64/N=2048 (the < 10 ms bar, release-measured) ──────

fn bench_g2_split_latency(c: &mut Criterion) {
    let diffs = planted_differences(64, 16, 2048, 0x6255_6102);
    let sigma = identity_sigma(64);
    let mut m = vec![0.0; 64 * 64];
    let mut eigvals = vec![0.0; 64];
    let mut eigvecs = vec![0.0; 64 * 64];
    let mut basis = vec![0.0; 48 * 64];
    let mut sym = SymmetricEigScratch::new();
    let mut jacobi = vec![0.0; 64 * 64];
    let mut m_scr = vec![0.0; 64 * 64];
    let mut whitener = vec![0.0; 64 * 64];
    let mut bx = vec![0.0; 48];
    let mut proj = vec![0.0; 64];
    let mut group = c.benchmark_group("basis_split/g2_split_latency");
    group.bench_function("d64_n2048_full", |b| {
        b.iter(|| {
            // Full pipeline per iteration: accumulate → finish → split → project.
            let mut acc = PairDifferenceMoment::new(&mut m, 64).unwrap();
            for delta in &diffs {
                acc.accumulate(delta).unwrap();
            }
            acc.finish();
            let (verdict, _report) = split(
                &m,
                64,
                Some(Whitening::TotalCovariance(&sigma)),
                Some(48),
                10.0,
                &mut eigvals,
                &mut eigvecs,
                &mut basis,
                &mut sym,
                &mut jacobi,
                &mut m_scr,
                &mut whitener,
            )
            .unwrap();
            assert!(matches!(verdict, Verdict::SufficientVariation { .. }));
            // One held-out projection per iteration (the armed read path).
            project_invariant_into(&basis, 48, 64, &diffs[0], &mut bx, &mut proj).unwrap();
        })
    });
    group.finish();
}

// ── adversarial canaries + the TILR composition row ───────────────────────────

fn bench_adversarial_and_tilr(c: &mut Criterion) {
    let mut group = c.benchmark_group("basis_split/adversarial_gates");

    // Shuffled values: the gate MUST refuse (asserted per iteration).
    let mut shuffled = planted_differences(64, 16, 2048, 0x6255_6103);
    let mut rng = Lcg(0x5100_0000_6255_6104);
    for delta in shuffled.iter_mut() {
        for _ in 0..delta.len() {
            let i = (rng.next_f64() * delta.len() as f64) as usize % delta.len();
            let j = (rng.next_f64() * delta.len() as f64) as usize % delta.len();
            delta.swap(i, j);
        }
    }
    let m_shuffled = {
        let mut m = vec![0.0; 64 * 64];
        let mut acc = PairDifferenceMoment::new(&mut m, 64).unwrap();
        for delta in &shuffled {
            acc.accumulate(delta).unwrap();
        }
        acc.finish();
        m
    };
    let sigma = identity_sigma(64);
    group.bench_function("shuffled_pairs_refuse", |b| {
        b.iter(|| {
            let mut eigvals = vec![0.0; 64];
            let mut eigvecs = vec![0.0; 64 * 64];
            let mut basis = vec![0.0; 48 * 64];
            let mut sym = SymmetricEigScratch::new();
            let mut jacobi = vec![0.0; 64 * 64];
            let mut m_scr = vec![0.0; 64 * 64];
            let mut whitener = vec![0.0; 64 * 64];
            let (verdict, _) = split(
                black_box(&m_shuffled),
                64,
                Some(Whitening::TotalCovariance(&sigma)),
                Some(48),
                10.0,
                &mut eigvals,
                &mut eigvecs,
                &mut basis,
                &mut sym,
                &mut jacobi,
                &mut m_scr,
                &mut whitener,
            )
            .unwrap();
            assert!(matches!(verdict, Verdict::Unidentified(_)));
        })
    });

    // Degenerate (all-identical): the gate MUST refuse.
    let degenerate = vec![vec![1.0f64; 64]; 64];
    let m_deg = {
        let mut m = vec![0.0; 64 * 64];
        let mut acc = PairDifferenceMoment::new(&mut m, 64).unwrap();
        for delta in &degenerate {
            acc.accumulate(delta).unwrap();
        }
        acc.finish();
        m
    };
    group.bench_function("degenerate_pairs_refuse", |b| {
        b.iter(|| {
            let mut eigvals = vec![0.0; 64];
            let mut eigvecs = vec![0.0; 64 * 64];
            let mut basis = vec![0.0; 48 * 64];
            let mut sym = SymmetricEigScratch::new();
            let mut jacobi = vec![0.0; 64 * 64];
            let mut m_scr = vec![0.0; 64 * 64];
            let mut whitener = vec![0.0; 64 * 64];
            let (verdict, _) = split(
                black_box(&m_deg),
                64,
                Some(Whitening::TotalCovariance(&sigma)),
                Some(48),
                10.0,
                &mut eigvals,
                &mut eigvecs,
                &mut basis,
                &mut sym,
                &mut jacobi,
                &mut m_scr,
                &mut whitener,
            )
            .unwrap();
            assert!(matches!(verdict, Verdict::Unidentified(_)));
        })
    });

    // The T0.3 substrate-comparison row: the SAME fixture through TILR's
    // discover_invariant_subspace (f32, allocating, τ-rank) — the baseline the
    // new path is weighed against (G1 exactness was T0.3's verdict; this row
    // prices the latency).
    let f32_diffs: Vec<Vec<f32>> = planted_differences(64, 16, 2048, 0x6255_6101 + 64)
        .iter()
        .map(|dv| dv.iter().map(|&v| v as f32).collect())
        .collect();
    let f32_refs: Vec<&[f32]> = f32_diffs.iter().map(|d| d.as_slice()).collect();
    group.bench_function("tilr_composition_d64", |b| {
        b.iter(|| {
            let (basis, r) =
                katgpt_core::tilr::discover_invariant_subspace(black_box(&f32_refs), 0.99)
                    .expect("planted fixture discovers");
            assert_eq!(r, 48);
            black_box(&basis);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_g1_planted_recovery,
    bench_g2_split_latency,
    bench_adversarial_and_tilr,
);
criterion_main!(benches);
