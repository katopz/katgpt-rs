#![cfg(feature = "regime_probe")]
//! Issue 740 T1/T2/T4 — golden vectors + determinism gates for the regime
//! probes (katgpt-core, feature `regime_probe`).
//!
//! 1. **T4 golden vectors** — the Gardner capacity LUT (`kappa_max`,
//!    `basin_radius_bound`) against direct bisection inversion of the same
//!    closed forms, relative error < 1e-6 (the issue's target).
//! 2. **T1 known answers** — uniform categorical ⇒ `ln V`; one-hot ⇒ 0;
//!    kernel entropy vs direct `Σ p·ln p` in f64; batch ≡ scalar.
//! 3. **T2/T3 bit-determinism** — same inputs ⇒ bit-identical BLAKE3
//!    artifacts (×2 runs, fresh objects).
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features regime_probe --test regime_probe_golden
//! ```

#![allow(clippy::float_cmp)]

use katgpt_core::regime_probe::{
    BasinScratch, BasinReport, FrozenRenovator, basin_probe_into, basin_radius_bound,
    basin_radius_from_kappa, conditional_entropies_into, conditional_entropy_nats,
    entropy_gap, gamma_capacity, kappa_max, kappa_max_bisection, phi_cdf, phi_pdf,
};

// ── T4: Gardner LUT golden vectors ──────────────────────────────────────────

#[test]
fn t4_lut_matches_direct_bisection_rel_err_below_1e_6() {
    // Sweep the whole useful load domain; a few points just above the
    // saturation cap and just below the κ = 0 boundary exercise the clamps.
    let mut worst_k = 0.0f64;
    let mut worst_k_at = 0.0f64;
    let mut worst_r = 0.0f64;
    let mut worst_r_at = 0.0f64;
    let mut gamma = 0.031;
    while gamma < 2.0 {
        let k_lut = kappa_max(gamma);
        let k_ref = kappa_max_bisection(gamma);
        // Relative error against the direct inversion. Where κ_ref is
        // effectively 0 (γ within 1e-9 of 2), fall back to absolute error.
        if k_ref > 1e-3 {
            let rel = (k_lut - k_ref).abs() / k_ref;
            if rel > worst_k {
                worst_k = rel;
                worst_k_at = gamma;
            }
            assert!(
                rel < 1e-6,
                "κ_max rel err {rel:e} at γ={gamma}: lut={k_lut} bisect={k_ref}"
            );
        } else {
            assert!(
                (k_lut - k_ref).abs() < 1e-6,
                "κ_max abs err at γ={gamma}: lut={k_lut} bisect={k_ref}"
            );
        }
        // The ρ bound through the LUT must agree with the bound computed
        // from the directly-inverted margin.
        let rho_lut = basin_radius_bound(gamma);
        let rho_ref = basin_radius_from_kappa(k_ref);
        if rho_ref > 1e-6 {
            let rel = (rho_lut - rho_ref).abs() / rho_ref;
            if rel > worst_r {
                worst_r = rel;
                worst_r_at = gamma;
            }
            assert!(
                rel < 1e-5,
                "ρ bound rel err {rel:e} at γ={gamma}: lut={rho_lut} bisect={rho_ref}"
            );
        }
        gamma += 0.0097; // incommensurate with the grid spacing on purpose
    }
    println!(
        "worst κ_max rel err {worst_k:.2e} at γ={worst_k_at:.4}; worst ρ_bound rel err {worst_r:.2e} at γ={worst_r_at:.4}"
    );
}

#[test]
fn t4_bisection_actually_solves_the_equation() {
    // The reference must satisfy γ_c(κ_max(γ)) = γ to bisection tolerance.
    for &gamma in &[0.05, 0.2, 0.5, 1.0, 1.5, 1.9] {
        let k = kappa_max_bisection(gamma);
        let back = gamma_capacity(k);
        let rel = (back - gamma).abs() / gamma;
        assert!(rel < 1e-9, "bisection self-check at γ={gamma}: got {back}, rel {rel:e}");
    }
}

#[test]
fn t4_known_capacity_points() {
    // γ_c(0) = 2 exactly, so κ_max(2) = 0 and the bound is 0 there.
    assert_eq!(kappa_max(2.0), 0.0);
    assert_eq!(basin_radius_bound(2.0), 0.0);
    // Above the γ domain there is no capacity at any margin.
    assert_eq!(basin_radius_bound(5.0), 0.0);
    // Below the grid cap the bound saturates at 1.
    assert_eq!(basin_radius_bound(0.001), 1.0);
    // A mid-domain spot value: κ_max(1.0) inverts γ_c(κ) = 1, i.e.
    // (1+κ²)Φ(κ) + κφ(κ) = 1 — κ ≈ 0.4712. Tolerance is loose (1e-2)
    // because this pins the closed forms against hand arithmetic, not the
    // interpolation (the golden sweep above pins that at 1e-6 vs bisection).
    let k = kappa_max(1.0);
    assert!((k - 0.4712).abs() < 1e-2, "κ_max(1.0) = {k}, expected ≈ 0.4712");
    assert!((phi_cdf(0.0) - 0.5).abs() < 1e-15);
    assert!((phi_pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-15);
}

#[test]
fn t4_saturation_and_monotonicity() {
    let mut gamma = 0.05;
    let mut prev = basin_radius_bound(gamma);
    while gamma < 1.99 {
        gamma += 0.05;
        let b = basin_radius_bound(gamma);
        assert!(b <= prev, "ρ bound must not rise with load at γ={gamma}");
        assert!((0.0..=1.0).contains(&b), "bound out of range at γ={gamma}: {b}");
        prev = b;
    }
}

// ── T1: conditional-entropy known answers ───────────────────────────────────

#[test]
fn t1_uniform_is_ln_v() {
    for v in &[2usize, 8, 64, 1024] {
        let logits = vec![0.0f32; *v];
        let h = conditional_entropy_nats(&logits);
        let want = (*v as f32).ln();
        assert!(
            (h - want).abs() < 1e-5,
            "uniform H over {v}: got {h}, want {want}"
        );
    }
}

#[test]
fn t1_one_hot_is_zero() {
    for pos in &[0usize, 7, 63] {
        let mut logits = vec![-100.0f32; 64];
        logits[*pos] = 100.0;
        let h = conditional_entropy_nats(&logits);
        assert!(h.abs() < 1e-4, "one-hot[{pos}] H = {h}");
    }
}

#[test]
fn t1_matches_direct_f64_reference() {
    // Deterministic pseudo-random logits (no RNG substrate in the fixture).
    let logits: Vec<f32> = (0..511)
        .map(|i| ((i as u64).wrapping_mul(6364136223846793005) >> 33) as f32 / 8.0 % 20.0 - 10.0)
        .collect();
    // Direct f64 softmax entropy.
    let max = logits.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    let want: f64 = logits
        .iter()
        .map(|&x| {
            let p = (x as f64 - max).exp() / z;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();
    let got = conditional_entropy_nats(&logits) as f64;
    assert!(
        (got - want).abs() < 1e-3,
        "kernel entropy {got} vs direct {want}"
    );
}

#[test]
fn t1_batch_equals_scalar_and_mean_matches() {
    let positions = 33;
    let vocab = 17;
    let logits: Vec<f32> = (0..positions * vocab)
        .map(|i| ((i * 37) % 23) as f32 / 3.0 - 3.0)
        .collect();
    let mut out = vec![0.0f32; positions];
    conditional_entropies_into(&logits, positions, vocab, &mut out);
    let mut sum = 0.0f32;
    for p in 0..positions {
        let scalar =
            conditional_entropy_nats(&logits[p * vocab..(p + 1) * vocab]);
        assert_eq!(out[p], scalar, "row {p}: batch vs scalar must be bit-equal");
        sum += scalar;
    }
    let mean = katgpt_core::regime_probe::mean_conditional_entropy(&logits, positions, vocab);
    assert_eq!(mean, sum / positions as f32);
}

// ── T2/T3: bit-determinism gates (×2 runs, fresh objects) ───────────────────

#[test]
fn t2_gap_artifact_bit_identical_across_runs() {
    let a: Vec<f32> = (0..128).map(|i| (i % 9) as f32 * 0.31).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i * 5) % 13) as f32 * 0.17).collect();
    let r1 = entropy_gap(&a, &b);
    let r2 = entropy_gap(&a, &b);
    assert_eq!(r1, r2, "reports must be content-equal");
    assert_eq!(r1.artifact, r2.artifact, "artifacts must be bit-identical");
}

/// Scripted renovator: one-hot at the reference answer (an "oracle basin").
struct Oracle {
    answer: Vec<usize>,
    alphabet: usize,
}
impl FrozenRenovator for Oracle {
    fn len(&self) -> usize {
        self.answer.len()
    }
    fn alphabet(&self) -> usize {
        self.alphabet
    }
    fn posterior_into(&self, i: usize, _x: &[usize], out: &mut [f32]) {
        out.fill(0.0);
        out[self.answer[i]] = 1.0;
    }
}

#[test]
fn t3_basin_artifact_bit_identical_across_runs() {
    let original: Vec<usize> = (0..96).map(|i| (i * 7) % 5).collect();
    let ren = Oracle { answer: original.clone(), alphabet: 5 };
    let mut s1 = BasinScratch::new(ren.len(), 5);
    let mut r1 = BasinReport::default();
    basin_probe_into(&ren, &original, 0.3, 3, 0x5EED_7402, &mut s1, &mut r1);
    // The rerun REUSES the same scratch (capacity-reuse must not perturb the
    // artifact) and reuses r2 as a fresh report.
    let mut r2 = BasinReport::default();
    basin_probe_into(&ren, &original, 0.3, 3, 0x5EED_7402, &mut s1, &mut r2);
    assert_eq!(r1.artifact, r2.artifact);
    assert_eq!(r1.final_state, r2.final_state);
    assert_eq!(r1.recovery_rate, 1.0, "oracle renovator must fully recover");
}
