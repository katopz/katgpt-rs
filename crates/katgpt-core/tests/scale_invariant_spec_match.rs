//! Spec-match test for the ScaleInvariant Lean 4 theorems (Plan 622 Phase 6).
//!
//! Mirrors the Lean spec at `.proofs/KatgptProof/ScaleInvariant/`:
//!
//! - **`Basic.lean`** — the schedule closed forms (`aSq`, `mOf`, the
//!   `(a₀², m₀) = (1, 0)` boundary pin forcing α = β = e^(1/2)), the per-key
//!   mass profile `E[e^{L_t}] = α/(t/τ+1)`, and the harmonic-sum interval
//!   bound `scale_invariant_decade_mass_bounded` (App. D):
//!   `ατ·log((TΔ+1+τ)/(T+1+τ)) ≤ Σ_{t=T+1}^{TΔ} mass ≤ ατ·log((TΔ+τ)/(T+τ))`.
//! - **`Sigmoid.lean`** — the tilt transfer: `E[σ(L_t)] = α·c(a_t)/(t/τ+1)`
//!   with `c(a) = E[σ(−L′)]`, `L′ ~ N(1, a²)`, and the halving
//!   `(t/τ+1)·E[σ(L_t)] → α/2`. The Lean side packages the Gaussian-integral
//!   facts as hypotheses (the `Hope/Basic.lean` precedent); THIS test
//!   validates them numerically — adaptive quadrature carries no hypotheses.
//! - **`Dominates.lean`** — the comparator ordering: ALiBi-shaped
//!   (`C·q^t`) and dilution-shaped (`C·(1+t)^{−κ}`) decade masses collapse
//!   to 0 while the schedule holds the T-independent floor
//!   `ατ·log((Δ+1+τ)/(2+τ))`.
//!
//! Deterministic grids only (no RNG — the Phase-5 probe law). If this test
//! fails, the Rust schedule drifted from the Lean spec.
//!
//! Run: `cargo test --features scale_invariant_attn --test scale_invariant_spec_match`
//!
//! Cross-references:
//! - Plan: `.plans/622_scale_invariant_attn.md` (Phase 6)
//! - Research: `.research/610_Scale_Invariant_Attention.md`
//! - Lean proofs: `.proofs/KatgptProof/ScaleInvariant/{Basic,Sigmoid,Dominates}.lean`
//! - Rust implementation: `crates/katgpt-core/src/scale_invariant.rs`

#![cfg(feature = "scale_invariant_attn")]
#![cfg(test)]

use katgpt_core::scale_invariant::{gaussian_expectation, ScaleInvariantLut};

// ── Lean spec mirrors (f64 — closer to the ℝ contract than f32) ─────────

/// `alpha` (Lean `ScaleInvariant.alpha`): α = e^(1/2), forced by the
/// boundary pin.
fn alpha() -> f64 {
    (0.5f64).exp()
}

/// `aSq t τ = 2·log(t/τ + 1) + 1` (Lean `ScaleInvariant.aSq`).
fn a_sq(t: f64, tau: f64) -> f64 {
    2.0 * (t / tau + 1.0).ln() + 1.0
}

/// `mOf t τ = 1 − aSq t τ` (Lean `ScaleInvariant.mOf`).
fn m_of(t: f64, tau: f64) -> f64 {
    1.0 - a_sq(t, tau)
}

/// `perKeyMass t τ = exp(mOf + aSq/2)` (Lean `ScaleInvariant.perKeyMass`).
fn per_key_mass(t: f64, tau: f64) -> f64 {
    (m_of(t, tau) + a_sq(t, tau) / 2.0).exp()
}

/// The decade mass Σ_{t=T+1}^{TΔ} perKeyMass (Lean `ScaleInvariant.decadeMass`,
/// ℕ indices).
fn decade_mass(t_lo: u64, t_hi: u64, tau: f64) -> f64 {
    (t_lo..=t_hi).map(|t| per_key_mass(t as f64, tau)).sum()
}

/// The Lean lower bound: ατ·log((TΔ+1+τ)/(T+1+τ)).
fn decade_lower(t: f64, d: f64, tau: f64) -> f64 {
    alpha() * tau * ((t * d + 1.0 + tau) / (t + 1.0 + tau)).ln()
}

/// The Lean upper bound: ατ·log((TΔ+τ)/(T+τ)).
fn decade_upper(t: f64, d: f64, tau: f64) -> f64 {
    alpha() * tau * ((t * d + tau) / (t + tau)).ln()
}

/// The T-independent Def-3.1 floor: ατ·log((Δ+1+τ)/(2+τ))
/// (Lean `ScaleInvariant.decadeMassFloor`).
fn decade_floor(d: f64, tau: f64) -> f64 {
    alpha() * tau * ((d + 1.0 + tau) / (2.0 + tau)).ln()
}

// ── The schedule matches the Lean closed forms ──────────────────────────

/// The built LUT pairs equal the Lean closed forms `(aSq.sqrt, mOf)` to
/// f32-cast precision, and the boundary pin `(a₀², m₀) = (1, 0)` is exact.
#[test]
fn spec_schedule_matches_lean_closed_forms() {
    let tau = 10.0_f64;
    let lut = ScaleInvariantLut::build(tau as f32, 4096);
    for &t in &[0.0_f64, 1.0, 7.0, 10.0, 63.0, 100.0, 1000.0, 4095.0] {
        let (a, m) = lut.pair(t as usize);
        let a_ref = a_sq(t, tau).sqrt() as f32;
        let m_ref = m_of(t, tau) as f32;
        assert!(
            (a - a_ref).abs() <= 1e-6 * a_ref.abs().max(1.0),
            "a_t drift: t={t} got {a} want {a_ref}"
        );
        assert!(
            (m - m_ref).abs() <= 1e-6 * m_ref.abs().max(1.0),
            "m_t drift: t={t} got {m} want {m_ref}"
        );
    }
    // Boundary pins are EXACT (Lean `aSq_boundary`, `mOf_boundary`).
    let (a0, m0) = lut.pair(0);
    assert_eq!(a0, 1.0, "a₀ must be exactly 1");
    assert_eq!(m0, 0.0, "m₀ must be exactly 0 (identity at distance 0)");
}

/// The per-key mass profile equals `α/(t/τ+1)` (Lean `perKeyMass_eq_alpha`)
/// and is strictly antitone in distance (the harmonic profile's shape).
#[test]
fn spec_per_key_mass_harmonic_profile() {
    let tau = 10.0_f64;
    let mut prev = f64::INFINITY;
    for &t in &[0.0_f64, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0] {
        let got = per_key_mass(t, tau);
        let want = alpha() / (t / tau + 1.0);
        assert!(
            (got - want).abs() <= 1e-9 * want.abs(),
            "perKeyMass drift: t={t} got {got} want {want}"
        );
        assert!(got < prev, "perKeyMass must be strictly antitone in t");
        prev = got;
    }
}

// ── Theorem 1: the harmonic-sum interval bounds ─────────────────────────

/// **Lean `scale_invariant_decade_mass_bounded` (∃-check).** The f64 decade
/// sum stays inside the harmonic interval bounds at every sampled
/// `(T, Δ, τ)`, and inside the T-independent band: ≤ ατ·log Δ always, and
/// ≥ the positive floor for Δ ≥ 2 (Lean `decadeMass_le_logDelta`,
/// `decadeMass_ge_floor`, `decadeMassFloor_pos`).
#[test]
fn spec_decade_mass_interval_bounds() {
    for &(t, d) in &[(1u64, 2u64), (7, 2), (100, 10), (1000, 10), (5000, 2)] {
        for &tau in &[1.0_f64, 10.0, 100.0] {
            let sum = decade_mass(t + 1, t * d, tau);
            let lo = decade_lower(t as f64, d as f64, tau);
            let hi = decade_upper(t as f64, d as f64, tau);
            let band = alpha() * tau * (d as f64).ln();
            assert!(
                lo <= sum + 1e-9 * hi.abs(),
                "lower bound violated: T={t} Δ={d} τ={tau}: {lo} > {sum}"
            );
            assert!(
                sum <= hi + 1e-9 * hi.abs(),
                "upper bound violated: T={t} Δ={d} τ={tau}: {sum} > {hi}"
            );
            assert!(
                sum <= band + 1e-9 * band.abs(),
                "band violated: T={t} Δ={d} τ={tau}: {sum} > ατ·lnΔ = {band}"
            );
            if d >= 2 {
                let floor = decade_floor(d as f64, tau);
                assert!(floor > 0.0, "floor must be positive: Δ={d} τ={tau}");
                assert!(
                    sum >= floor - 1e-9 * floor.abs(),
                    "floor violated: T={t} Δ={d} τ={tau}: {sum} < {floor}"
                );
            }
        }
    }
}

// ── Theorem 2: the sigmoid tilt transfer ────────────────────────────────

/// **Lean `sigmoid_transfer_halves_constant` + `sigmoid_transfer_gap`
/// (numeric validation of the hypothesized Gaussian-integral facts).**
///
/// Validates the full tilt assembly with the adaptive quadrature
/// (`gaussian_expectation` — the shipped Phase-2 instrument; Gauss–Hermite
/// misses the σ transition once a ≳ 5 and MC has no tail support):
///
/// 1. the tilt identity `E[σ(L_t)] = mass_t · c(a_t)` to quadrature
///    precision (the Lean `htilt` hypothesis),
/// 2. the exact gap `|(t/τ+1)·E[σ(L_t)] − α/2| = α·|c(a_t) − 1/2|`
///    (Lean `sigmoid_transfer_gap` — the explicit ε(t)),
/// 3. the halving: the constant drifts monotonically from σ(−1) toward 1/2
///    (the Lean `hlim` hypothesis, dominated convergence).
#[test]
fn spec_tilt_transfer_identity_and_halving() {
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let tau = 10.0_f64;
    let mut prev_c = 0.0_f64;
    for &t in &[1.0_f64, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0] {
        let a = a_sq(t, tau).sqrt();
        let m = m_of(t, tau);
        // E[σ(L_t)], L_t ~ N(m, a²) — the sigmoid arm's per-key mass.
        let e_sigma = gaussian_expectation(m, a, sig);
        // c(a) = E[σ(−L′)], L′ ~ N(1, a²) — the tilt-transfer constant.
        let c = gaussian_expectation(1.0, a, |z| sig(-z));
        // (1) the tilt identity (Lean `htilt`): Eσ = mass · c, mass = e^{m+a²/2}.
        let lhs = (t / tau + 1.0) * e_sigma;
        let rhs = alpha() * c;
        assert!(
            (lhs - rhs).abs() <= 1e-8 * rhs.abs(),
            "tilt identity drift: t={t}: {lhs} vs {rhs}"
        );
        // (2) the exact gap ε(t) (Lean `sigmoid_transfer_gap`).
        let gap_lhs = (lhs - alpha() / 2.0).abs();
        let gap_rhs = alpha() * (c - 0.5).abs();
        assert!(
            (gap_lhs - gap_rhs).abs() <= 1e-8 * gap_rhs.abs().max(1e-300),
            "gap identity drift: t={t}: {gap_lhs} vs {gap_rhs}"
        );
        // (3) the monotone drift toward the halved constant, from below.
        assert!(
            c < 0.5,
            "c must approach 1/2 from below (t={t}: {c})"
        );
        if t > 1.0 {
            assert!(c > prev_c, "c must increase monotonically (t={t})");
        }
        prev_c = c;
    }
    // The drift is O(1/√log t) — SLOW by design (a_t² grows only
    // logarithmically; the plan's recorded expected behavior). At t = 10⁵
    // (a ≈ 4.4) the constant is ≈ 0.42 — within 0.1 of the halved limit.
    let a_big = a_sq(100_000.0, tau).sqrt();
    let c_big = gaussian_expectation(1.0, a_big, |z| sig(-z));
    assert!(
        (c_big - 0.5).abs() < 0.1,
        "c must approach 1/2 (got {c_big} at t=10⁵)"
    );
}

// ── Theorem 3: the comparator ordering ──────────────────────────────────

/// **Lean `scale_invariant_dominates_ssmax_on_decade_mass` (∃-check).**
/// The schedule holds the floor while both comparator families collapse:
/// the ALiBi exponential (C·q^t, q = e^{−0.1}) and the dilution polynomial
/// (C·(1+t)^{−κ}, κ = 2) decade masses shrink toward zero across the ladder
/// while the schedule's floor stays fixed.
#[test]
fn spec_dominates_comparators() {
    let tau = 10.0_f64;
    let d = 2.0_f64;
    let floor = decade_floor(d, tau);
    assert!(floor > 0.0);

    // The schedule's decade mass sits above the floor at every T.
    for &(t, dd) in &[(1u64, 2u64), (10, 2), (100, 2), (1000, 2)] {
        let si = decade_mass(t + 1, t * dd, tau);
        assert!(
            si >= floor - 1e-9 * floor.abs(),
            "si floor violated: T={t}: {si} < {floor}"
        );
    }

    // ALiBi comparator: per-key mass C·q^t (q = e^{−κ} realizes m_t = −κt).
    // The window holds T terms of ~q^{2T} — the geometric decay beats the
    // count only past T ≈ 30 (measured: T=10 → 2.21, T=100 → 4.3e-4), so
    // the monotone ladder starts at T = 100.
    let c1 = 1.0_f64;
    let kappa = 0.1_f64;
    let q = (-kappa).exp();
    let mut prev_alibi = f64::INFINITY;
    for &t in &[1u64, 10, 100, 1000] {
        let mass: f64 = ((t + 1)..=(t * 2))
            .map(|k| c1 * q.powi(k as i32))
            .sum();
        let bound = c1 * q.powi((t + 1) as i32) / (1.0 - q);
        assert!(
            mass <= bound,
            "ALiBi tail bound violated: T={t}: {mass} > {bound}"
        );
        if t >= 100 {
            assert!(mass < prev_alibi, "ALiBi decade mass must shrink (T={t})");
        }
        prev_alibi = mass;
    }
    assert!(
        prev_alibi < floor,
        "ALiBi comparator must collapse below the schedule's floor"
    );

    // Dilution comparator: per-key mass C·(1+t)^{−κ} (κ = 2), the
    // polynomial shape SSMax's dilution bound leaves behind.
    let c2 = 1.0_f64;
    let k2 = 2.0_f64;
    let mut prev_pow = f64::INFINITY;
    for &t in &[1u64, 10, 100, 1000] {
        let mass: f64 = ((t + 1)..=(t * 2))
            .map(|k| c2 / (1.0 + k as f64).powf(k2))
            .sum();
        let bound = c2 * (d) * (t as f64).powf(1.0 - k2);
        assert!(
            mass <= bound + 1e-15,
            "dilution tail bound violated: T={t}: {mass} > {bound}"
        );
        assert!(mass < prev_pow, "dilution decade mass must shrink");
        prev_pow = mass;
    }
    assert!(
        prev_pow < floor,
        "dilution comparator must collapse below the schedule's floor"
    );
}

// ── Sentinel: the Lean proof directory integrity ─────────────────────────

/// The Lean 4 proof files for ScaleInvariant exist. If they are deleted, the
/// theorem trio would rely only on the empirical G1 tests.
#[test]
fn proofs_directory_exists() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for f in [
        "Basic.lean",
        "Sigmoid.lean",
        "Dominates.lean",
        "SpecTests.lean",
    ] {
        let p = repo_root.join(format!(".proofs/KatgptProof/ScaleInvariant/{f}"));
        assert!(p.exists(), "Lean spec file missing: {}", p.display());
    }
}
