#![cfg(feature = "sst_project")]
//! Issue 958 — the G1 correctness battery for `sst_certified`.
//!
//! Arms (the issue's GOAT sketch):
//! 1. **Subspace agreement vs the SVD oracle** — the fitted right-side
//!    protected basis' projector matches the top-k right singular subspace
//!    of `thin_svd_into` (one-sided Jacobi, the substrate primitive the
//!    module consumes rather than forks) to f32 tolerance.
//! 2. **Anatomy vs the brute-force two-sided projection** — the stored-basis
//!    path agrees with materialized projectors at a larger random shape.
//! 3. **Rank-deficiency guard** — a reference with a zero column block
//!    refuses with `RankDeficientAnchor` (never silently shrinks).
//! 4. **Tail-capacity guard** — a σ-visible over-request refuses.
//! 5. **The isotropic-rounding negative** — dense-project-then-quantize
//!    FAILS the certifier at a bit budget where the factored arm PASSES
//!    (the check is not vacuous; the paper's composition-order law in
//!    miniature).
//! 6. **Tolerance + shape hygiene** — BadTolerance and ShapeMismatch
//!    refuse loud.
//!
//! The inline `mod tests` in `sst_certified.rs` carries the fixture-level
//! arms (digest invariance, planted-leak refusal, compose, stale anchor);
//! this battery carries the independent-oracle arms.

use katgpt_core::sst_certified::{
    SideBasis, SstError, SstRefusal, SstScratch, TailFactor, fit_anchor,
};
use katgpt_core::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

/// Deterministic LCG (the unseeded-global-RNG gate).
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

fn gauss(seed: u64, m: usize, n: usize) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..m * n).map(|_| rng.next_gauss() as f32).collect()
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|v| v * v).sum();
    if den > 0.0 {
        num.sqrt() / den.sqrt()
    } else {
        num
    }
}

const M: usize = 24;
const N: usize = 48;
const FREE: usize = 6;

fn fixture() -> (Vec<f32>, katgpt_core::sst_certified::SstAnchor) {
    let w = gauss(0x958_0001, M, N);
    let anchor = fit_anchor(&w, M, N, FREE).expect("fixture anchor");
    (w, anchor)
}

/// Arm 1 — the fitted protected subspace IS the SVD's top-k right singular
/// subspace (projector-level comparison: rotation-invariant, the only
/// solver-agnostic form).
#[test]
fn g1_fitted_basis_agrees_with_thin_svd_oracle() {
    let (w, anchor) = fixture();
    let mut result = SvdResultScratch::with_capacity(M, N);
    let mut work = SvdScratch::with_capacity(N, M);
    thin_svd_into(&w, M, N, &mut result, &mut work);
    assert!(
        result.len() >= anchor.k,
        "oracle must return the full spectrum"
    );

    // Oracle protected projector: Σ over the top-k right singular vectors.
    let mut p_oracle = vec![0.0f32; N * N];
    for j in 0..anchor.k {
        let v = result.right_singular_vector(j);
        for i in 0..N {
            for l in 0..N {
                p_oracle[i * N + l] += v[i] * v[l];
            }
        }
    }
    // Fitted projector: the wide side stores Protected{V_k}.
    let v_k = match &anchor.right {
        SideBasis::Protected { basis } => basis,
        other => panic!("wide side must store Protected, got {other:?}"),
    };
    let mut p_fitted = vec![0.0f32; N * N];
    for j in 0..anchor.k {
        for i in 0..N {
            for l in 0..N {
                p_fitted[i * N + l] += v_k[i * anchor.k + j] * v_k[l * anchor.k + j];
            }
        }
    }
    assert!(
        rel_err(&p_fitted, &p_oracle) < 1e-4,
        "fitted protected subspace vs thin_svd oracle: {}",
        rel_err(&p_fitted, &p_oracle)
    );
}

/// Arm 2 — the stored-basis anatomy agrees with brute-force materialized
/// projectors on random deltas (dense and factored forms).
#[test]
fn g1_anatomy_matches_brute_force_oracle() {
    let (_w, anchor) = fixture();
    let (m, n, k) = (anchor.m, anchor.n, anchor.k);

    // Materialize both protected projectors from the stored bases.
    let pl = match &anchor.left {
        SideBasis::Complement { basis } => {
            let f = basis.len() / m;
            let bbt = bb_t(basis, m, f);
            sub_eye(&bbt, m)
        }
        SideBasis::Protected { basis } => bb_t(basis, m, k),
    };
    let pr = match &anchor.right {
        SideBasis::Complement { basis } => {
            let f = basis.len() / n;
            let bbt = bb_t(basis, n, f);
            sub_eye(&bbt, n)
        }
        SideBasis::Protected { basis } => bb_t(basis, n, k),
    };

    for seed in [11u64, 12, 13] {
        let delta = gauss(seed, M, N);
        // Brute force: ‖P_lΔ‖² + ‖(I−P_l)ΔP_r‖² — the same two-positive-terms
        // identity, computed through materialized projectors.
        let pld = mat_mul(&pl, &delta, M, M, N);
        let term1: f32 = pld.iter().map(|v| v * v).sum();
        let tl = sub_eye(&pl, M);
        let tld = mat_mul(&tl, &delta, M, M, N);
        let tld_pr = mat_mul(&tld, &pr, M, N, N);
        let term2: f32 = tld_pr.iter().map(|v| v * v).sum();
        let brute = term1 + term2;
        let total: f32 = delta.iter().map(|v| v * v).sum();

        let mut s = SstScratch::default();
        let a = anchor.anatomy_into(&delta, &mut s).expect("anatomy");
        assert!(
            (a.protected_energy - brute).abs() < 1e-3 * total.max(1e-9),
            "seed {seed}: anatomy {} vs brute {brute}",
            a.protected_energy
        );

        // Factored form of the same delta: Δ = A·Bᵀ via its own rank-N SVD
        // is overkill — use the trivial full-rank factorization
        // A = Δ, B = I_N (r = N), exercising the r×r Gram path exactly.
        let eye_n: Vec<f32> = {
            let mut e = vec![0.0f32; N * N];
            for i in 0..N {
                e[i * N + i] = 1.0;
            }
            e
        };
        let af = anchor
            .anatomy_factored_into(&delta, &eye_n, N, &mut s)
            .expect("factored anatomy");
        assert!(
            (af.protected_energy - brute).abs() < 1e-2 * total.max(1e-9),
            "seed {seed}: factored anatomy {} vs brute {brute}",
            af.protected_energy
        );
        assert!((af.total_energy - total).abs() < 1e-2 * total.max(1e-9));
    }
}

/// Arm 3 — a rank-deficient reference refuses loud (the σ floor guard);
/// the available count is reported, never silently shrunk.
#[test]
fn g1_rank_deficient_anchor_refuses() {
    // 12×16 with columns 8..16 held at zero → rank ≤ 8; free_min = 2 asks
    // for k = 10 protected directions — only ~8 exist above any sane floor.
    let mut w = gauss(21, 12, 16);
    for row in 0..12 {
        for col in 8..16 {
            w[row * 16 + col] = 0.0;
        }
    }
    match fit_anchor(&w, 12, 16, 2) {
        Err(SstError::RankDeficientAnchor {
            requested,
            available,
            ..
        }) => {
            assert_eq!(requested, 10);
            assert!(
                available <= 10 && (6..=10).contains(&available),
                "available {available}"
            );
        }
        other => panic!("rank-deficient reference must refuse, got {other:?}"),
    }
}

/// Arm 4 — a σ-visible over-request refuses with the capacity named.
#[test]
fn g1_tail_capacity_refusal() {
    let (w, anchor) = fixture();
    let delta = gauss(22, M, N);
    // The visible tail is ≤ min−k = 18; ask for more.
    let mut out = TailFactor::default();
    let mut report = katgpt_core::sst_certified::SstFactorReport::default();
    let mut s = SstScratch::default();
    match anchor.factor_tail_into(
        &w,
        &delta,
        FREE,
        anchor.f_narrow + 1,
        &mut out,
        &mut report,
        &mut s,
    ) {
        Err(SstError::TailCapacityExceeded {
            requested,
            available,
            ..
        }) => {
            assert!(requested > available);
        }
        other => panic!("over-request must refuse, got {other:?}"),
    }
}

/// Arm 5 — THE isotropic-rounding negative: at the same per-element noise
/// grid, dense-project-then-quantize FAILS the certifier while the factored
/// arm PASSES. The paper's composition-order law, in miniature.
#[test]
fn g1_isotropic_rounding_negative_dense_fails_factored_passes() {
    let (w, anchor) = fixture();
    let (m, n) = (anchor.m, anchor.n);

    // A tail-confined delta (planted through the materialized projectors).
    let (pl, pr) = materialized_protected(&anchor);
    let ilt = sub_eye(&pl, m);
    let irt = sub_eye(&pr, n);
    let tail = two_sided(&gauss(31, M, N), &ilt, &irt, M, N);

    // The FACTORED arm: factor → quantize the payload → decode → certify.
    let (factor, _) = anchor.factor_tail(&w, &tail, FREE, FREE).expect("factor");
    // Choose the grid so the payload error is VISIBLE (≈2.5% relative): the
    // same relative noise budget both arms must survive. The dense arm's
    // isotropic share must clear the 1e-3 bar DECISIVELY (leak ∝ step² —
    // measured ≈ 6e-3 at this grid), or the negative proves nothing.
    let c_max = factor.c.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let step = 0.05f32 * c_max;
    let q = factor.quantized(step).expect("quantize");
    let mut decoded_factored = vec![0.0f32; M * N];
    let mut s = SstScratch::default();
    q.decode_into(&mut decoded_factored, &mut s);
    let cert = anchor
        .certify_decoded_tail(&decoded_factored, 1e-3)
        .expect("factored arm must PASS at the shared noise budget");
    assert!(cert.protected_fraction < 1e-4);

    // The DENSE arm: project-then-quantize the SAME delta elementwise on
    // the same grid, decode, certify → must REFUSE (rounding noise is
    // isotropic; its protected share is not confinable).
    let mut decoded_dense = tail.clone();
    for v in decoded_dense.iter_mut() {
        *v = (*v / step).round() * step;
    }
    match anchor.certify_decoded_tail(&decoded_dense, 1e-3) {
        Err(SstRefusal::LeakAboveTolerance {
            protected_fraction, ..
        }) => {
            assert!(
                protected_fraction > 1e-3,
                "dense rounding leak {protected_fraction} must clear the bar to prove non-vacuity"
            );
        }
        other => panic!(
            "dense-project-then-quantize must FAIL the certifier (the check is not vacuous), got {other:?}"
        ),
    }
}

/// Arm 6 — tolerance and shape hygiene refuse loud.
#[test]
fn g1_tolerance_and_shape_hygiene() {
    let (_w, anchor) = fixture();
    let delta = gauss(41, M, N);
    assert_eq!(
        anchor.certify_decoded_tail(&delta, 1e-7),
        Err(SstRefusal::BadTolerance(1e-7))
    );
    assert_eq!(
        anchor.certify_decoded_tail(&delta, 2.0),
        Err(SstRefusal::BadTolerance(2.0))
    );
    assert!(matches!(
        anchor.certify_decoded_tail(&delta[..10], 1e-2),
        Err(SstRefusal::ShapeMismatch { .. })
    ));
}

// -- shared brute-force helpers ------------------------------------------------

fn bb_t(b: &[f32], dim: usize, f: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0f32;
            for t in 0..f {
                acc += b[i * f + t] * b[j * f + t];
            }
            out[i * dim + j] = acc;
        }
    }
    out
}

fn sub_eye(bbt: &[f32], dim: usize) -> Vec<f32> {
    let mut e = vec![0.0f32; dim * dim];
    for i in 0..dim {
        e[i * dim + i] = 1.0;
    }
    e.iter().zip(bbt.iter()).map(|(a, b)| a - b).collect()
}

fn mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += a[i * k + t] * b[t * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn two_sided(x: &[f32], pl: &[f32], pr: &[f32], m: usize, n: usize) -> Vec<f32> {
    let t = mat_mul(pl, x, m, m, n);
    mat_mul(&t, pr, m, n, n)
}

fn materialized_protected(anchor: &katgpt_core::sst_certified::SstAnchor) -> (Vec<f32>, Vec<f32>) {
    let pl = match &anchor.left {
        SideBasis::Complement { basis } => {
            let f = basis.len() / anchor.m;
            sub_eye(&bb_t(basis, anchor.m, f), anchor.m)
        }
        SideBasis::Protected { basis } => bb_t(basis, anchor.m, anchor.k),
    };
    let pr = match &anchor.right {
        SideBasis::Complement { basis } => {
            let f = basis.len() / anchor.n;
            sub_eye(&bb_t(basis, anchor.n, f), anchor.n)
        }
        SideBasis::Protected { basis } => bb_t(basis, anchor.n, anchor.k),
    };
    (pl, pr)
}
