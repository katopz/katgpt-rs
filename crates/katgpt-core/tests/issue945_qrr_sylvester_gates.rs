#![cfg(feature = "qrr_convergence")]
//! Issue 945 P6 gates — Sylvester attribution + α-intervention replay
//! (report-only offline lane; Research 623 §1, Theorem 3.2 + the α
//! protocol). Fixtures are CONSTRUCTED systems: the labels are the
//! construction truth (spectra, couplings), never the instrument under
//! test.
//!
//! Arms:
//! - **solve** — the Kronecker solve reproduces `PS − AP = B_c` to 1e-6 on
//!   a spectrally disjoint system; an exact spectral overlap refuses
//!   (`None`) — the uniqueness precondition surfacing as a verdict, not an
//!   error.
//! - **attribution** — rate inheritance: at `r = 0.999` over a `ρ(A) = 0.5`
//!   complement the coupled share dominates the intrinsic share at the
//!   t = 200 horizon (ratio > 10), and the t = 0 decomposition identity
//!   (`total = ‖e₀‖`) holds.
//! - **α replay** — `α = 1` is the identity arm (deviation < 1e-9), `α = ½`
//!   predicts just as exactly in the linear instrument, `α = 0` zeroes the
//!   coupled drive (the measured response collapses to the intrinsic-only
//!   prediction and the intervened split's coupled term is EXACTLY zero),
//!   and a corrupted coupling measured against a stale `P` blows the
//!   prediction past tolerance — the first-class recorded-negative path,
//!   `matches = false`.

use katgpt_core::qrr_convergence::{
    ALPHA_REPLAY_TOL, SylvesterBlocks, alpha_replay_check, energy_split, replay_probe_state,
    solve_sylvester,
};

/// Deterministic SplitMix64 — no global RNG (the workspace RNG-pinning law;
/// same generator as the P1/P3 gates file).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = self.0;
        let x = z ^ (z >> 30);
        let x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let x = x ^ (x >> 27);
        let x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        let x = x ^ (x >> 31);
        // Top 24 bits → [0, 1) → [−½, ½).
        ((x >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }
}

/// Damped-spiral block `r·e^{±iω}` as a real 2×2.
fn spiral(r: f32, omega: f32) -> [[f32; 2]; 2] {
    let (c, s) = (r * omega.cos(), r * omega.sin());
    [[c, -s], [s, c]]
}

/// The fixture: `d = 2 + 4` — the oscillatory spiral block, a diagonal
/// stable complement `A = diag(0.5, 0.4, 0.3, 0.25)` (ρ(A) = 0.5, real
/// spectrum, hence disjoint from any complex spiral), and a deterministic
/// bounded coupling into the complement rows only (the `U` rows stay zero).
fn fixture(r: f32, omega: f32, seed: u64) -> SylvesterBlocks {
    let k = 4;
    let diag = [0.5_f32, 0.4, 0.3, 0.25];
    let a: Vec<Vec<f32>> = (0..k)
        .map(|i| (0..k).map(|j| if i == j { diag[i] } else { 0.0 }).collect())
        .collect();
    let mut lcg = Lcg(seed);
    let mut b = vec![vec![0.0_f32; 2]; k + 2];
    for row in b.iter_mut().skip(2) {
        row[0] = lcg.next_f32();
        row[1] = lcg.next_f32();
    }
    let u: Vec<Vec<f32>> = (0..k + 2)
        .map(|i| vec![f32::from(i == 0), f32::from(i == 1)])
        .collect();
    SylvesterBlocks {
        s: spiral(r, omega),
        a,
        b,
        u,
    }
}

/// (a) The Kronecker solve reproduces the defining equation on a
/// spectrally disjoint system: ‖PS − AP − B_c‖_F < 1e-6 (f64; the
/// elimination itself lands at ~1e-14 — the bar is the spec's, not the
/// achievable floor).
#[test]
fn p_solves_the_sylvester_equation() {
    let fx = fixture(0.98, 0.3, 0x9E37_79B9);
    let p = solve_sylvester(&fx).expect("spectrally disjoint system must solve");
    let k = fx.a.len();
    let mut sq = 0.0_f64;
    for i in 0..k {
        for j in 0..2 {
            let ps = p[i][0] * f64::from(fx.s[0][j]) + p[i][1] * f64::from(fx.s[1][j]);
            let ap: f64 = (0..k).map(|l| f64::from(fx.a[i][l]) * p[l][j]).sum();
            let r = ps - ap - f64::from(fx.b[2 + i][j]);
            sq += r * r;
        }
    }
    let fro = sq.sqrt();
    assert!(fro < 1e-6, "‖PS − AP − B_c‖_F = {fro:e} must be < 1e-6");
}

/// (b) The uniqueness precondition surfacing as a verdict: planting S's
/// exact complex pair into A makes the Kronecker system singular
/// (`σ(A) ∩ σ(S) ≠ ∅`), and the solve refuses with `None`.
#[test]
fn singular_spectral_overlap_returns_none() {
    let mut fx = fixture(0.98, 0.3, 0x9E37_79B9);
    fx.a[0][0] = fx.s[0][0];
    fx.a[0][1] = fx.s[0][1];
    fx.a[1][0] = fx.s[1][0];
    fx.a[1][1] = fx.s[1][1];
    fx.a[2][2] = 0.5;
    fx.a[3][3] = 0.3;
    assert_eq!(solve_sylvester(&fx), None, "σ(A) ∩ σ(S) ≠ ∅ must refuse");
}

/// (c) Rate inheritance (Theorem 3.2's shape): with the oscillatory rate
/// r = 0.999 over a ρ(A) = 0.5 complement, the coupled share dominates the
/// intrinsic share at the t = 200 horizon. The t = 0 identity is pinned
/// first (the energy split's sanity arm).
#[test]
fn slow_mode_dominates_a_coupled_stall() {
    let fx = fixture(0.999, 0.3, 0xABC_DEF0);
    let p = solve_sylvester(&fx).expect("spectrally disjoint system must solve");
    let (u0, e0) = replay_probe_state(fx.a.len());
    let e0_norm: f64 = e0
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let at_zero = energy_split(&fx, &p, &u0, &e0, 0);
    assert!(
        (at_zero.total - e0_norm).abs() < 1e-9,
        "t=0 split total {} vs ‖e₀‖ {} must agree (the decomposition identity)",
        at_zero.total,
        e0_norm
    );
    let split = energy_split(&fx, &p, &u0, &e0, 200);
    assert!(
        split.intrinsic > 0.0,
        "intrinsic share must not underflow to a faked zero"
    );
    assert!(split.coupled > 0.0, "coupled share must be live at r=0.999");
    let ratio = split.coupled / split.intrinsic;
    assert!(
        ratio > 10.0,
        "rate inheritance: coupled {} vs intrinsic {} (ratio {ratio})",
        split.coupled,
        split.intrinsic
    );
}

/// (d) α = 0 zeroes the coupled drive: the replay's measured response
/// collapses to the intrinsic-only prediction (at t = 200 that is ~0), the
/// α = 1 response still carries the r = 0.999 mode, and the intervened
/// split's coupled term is EXACTLY zero.
#[test]
fn alpha_zero_replay_zeroes_the_coupled_response() {
    let fx = fixture(0.999, 0.3, 0xABC_DEF0);
    let p = solve_sylvester(&fx).expect("spectrally disjoint system must solve");
    let (_u0, e0) = replay_probe_state(fx.a.len());
    let v1 = alpha_replay_check(&fx, &p, 1.0, 200);
    let v0 = alpha_replay_check(&fx, &p, 0.0, 200);
    assert!(v0.matches, "the linear instrument's α=0 identity must hold");
    assert!(
        v0.deviation < 1e-9,
        "α=0 deviation {} must be roundoff",
        v0.deviation
    );
    assert!(
        v0.measured < 1e-6,
        "α=0 measured {} must be intrinsic-only (≈0 at t=200)",
        v0.measured
    );
    assert!(
        v1.measured > 10.0 * v0.measured.max(1e-12),
        "α=1 still carries the r=0.999 mode: {} vs intrinsic-only {}",
        v1.measured,
        v0.measured
    );
    let split = energy_split(&fx, &p, &[0.0, 0.0], &e0, 200);
    assert_eq!(split.coupled, 0.0, "zero drive ⇒ exactly-zero coupled term");
}

/// (e) α = 1 is the identity arm: measured vs closed form agree to f64
/// roundoff (≪ 1e-9) — the causal check can only fail for a real reason,
/// never from its own arithmetic. The α = ½ projection predicts just as
/// exactly in the linear instrument.
#[test]
fn alpha_one_replay_is_the_identity() {
    let fx = fixture(0.98, 0.3, 0x9E37_79B9);
    let p = solve_sylvester(&fx).expect("spectrally disjoint system must solve");
    let one = alpha_replay_check(&fx, &p, 1.0, 64);
    assert!(
        one.deviation < 1e-9,
        "α=1 identity deviation {} must be f64 roundoff",
        one.deviation
    );
    assert!(one.matches);
    assert!(one.measured.is_finite() && one.measured > 0.0);
    let half = alpha_replay_check(&fx, &p, 0.5, 64);
    assert!(
        half.deviation < 1e-9,
        "α=½ deviation {} must also be roundoff",
        half.deviation
    );
    assert!(half.matches);
}

/// (f) The recorded-negative path: solve P against the true coupling,
/// corrupt the coupling, replay — the stale attribution's prediction misses
/// the measured response by orders of magnitude above tolerance and the
/// verdict records `matches = false`.
#[test]
fn a_corrupted_coupling_fails_the_prediction_match() {
    let fx = fixture(0.98, 0.3, 0x9E37_79B9);
    let p = solve_sylvester(&fx).expect("true system must solve");
    let mut corrupted = fx.clone();
    corrupted.b[3][1] += 0.5; // complement-row coupling drift
    let v = alpha_replay_check(&corrupted, &p, 1.0, 64);
    assert!(
        v.deviation > ALPHA_REPLAY_TOL,
        "a corrupted coupling must blow the prediction: {} vs tol {ALPHA_REPLAY_TOL}",
        v.deviation
    );
    assert!(
        !v.matches,
        "the recorded negative must surface as matches=false"
    );
}
