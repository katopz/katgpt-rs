//! Issue 945 P6 — Sylvester attribution + α-intervention replay, REPORT
//! INSTRUMENT (Research 623, arXiv:2610.18174; feature `qrr_convergence`).
//!
//! Builds the coupled-stall fixture from the gate family (a damped spiral
//! r = 0.999, ω = 0.3 over a ρ(A) = 0.5 complement, deterministically
//! coupled), solves the Sylvester attribution, prints the complement energy
//! split, the three α replays (α ∈ {1, ½, 0}), and the recorded-negative
//! probe, then prints ONE verdict line.
//!
//! Report-only: prints to stdout, exits 0, writes nothing.
//!
//! Run:
//! `cargo run -p katgpt-core --features qrr_convergence --example issue945_qrr_sylvester_report`

use katgpt_core::qrr_convergence::{
    ALPHA_REPLAY_TOL, SylvesterBlocks, alpha_replay_check, energy_split, replay_probe_state,
    solve_sylvester,
};

fn main() {
    // ── the stall fixture (mirrors the gate (c) construction) ──
    let (r, omega, rho_a) = (0.999_f32, 0.3_f32, 0.5_f32);
    let blocks = stall_fixture(r, omega, rho_a);
    let k = blocks.a.len();
    let d = k + 2;
    let horizon = 200_usize;

    println!("== Issue 945 P6 — Sylvester attribution + alpha-intervention (report-only) ==");
    println!(
        "system: d={d} (oscillatory spiral r={r} ω={omega} + complement k={k}, ρ(A)={rho_a}), deterministically coupled"
    );

    let Some(p) = solve_sylvester(&blocks) else {
        println!(
            "sylvester solve: REFUSED (σ(A) ∩ σ(S) overlap — the decomposition is not unique)"
        );
        println!("P6 verdict: NEGATIVE (solve refused — spectral overlap)");
        return;
    };
    println!(
        "sylvester solve: unique (P is {k}×2 from the {}×{} Kronecker system)",
        2 * k,
        2 * k
    );

    // ── the complement energy split at the stall horizon ──
    let (u0, e0) = replay_probe_state(k);
    let split = energy_split(&blocks, &p, &u0, &e0, horizon);
    let at_zero = energy_split(&blocks, &p, &u0, &e0, 0);
    let e0_norm: f64 = e0
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let ratio = split.coupled / split.intrinsic;
    println!(
        "energy split @ t={horizon}: coupled={:.6} intrinsic={:.3e} total={:.6} (coupled/intrinsic={ratio:.3e})",
        split.coupled, split.intrinsic, split.total
    );
    println!(
        "t=0 sanity: ‖e₀‖={:.6} vs split total @0={:.6} (the decomposition identity)",
        e0_norm, at_zero.total
    );

    // ── the three α replays ──
    println!("alpha replay (horizon {horizon}):");
    let mut all_match = true;
    for alpha in [1.0_f32, 0.5, 0.0] {
        let v = alpha_replay_check(&blocks, &p, alpha, horizon);
        all_match &= v.matches;
        println!(
            "  α={alpha:.3}: measured={:.6} closed={:.6} deviation={:.3e} matches={}",
            v.measured, v.closed_form, v.deviation, v.matches
        );
    }

    // ── the recorded-negative probe: corrupted coupling vs the stale P ──
    let mut corrupted = blocks.clone();
    corrupted.b[3][1] += 0.5;
    let negative = alpha_replay_check(&corrupted, &p, 1.0, horizon);
    println!(
        "recorded-negative probe (corrupted coupling vs stale P): deviation={:.3e} matches={} (expected: no match)",
        negative.deviation, negative.matches
    );

    // ── the verdict ──
    let identity = alpha_replay_check(&blocks, &p, 1.0, horizon);
    let t0_ok = (at_zero.total - e0_norm).abs() < 1e-9;
    if all_match
        && t0_ok
        && identity.deviation < 1e-9
        && ratio > 10.0
        && !negative.matches
        && negative.deviation > ALPHA_REPLAY_TOL
    {
        println!("P6 verdict: ATTRIBUTION MATCHES");
    } else {
        println!(
            "P6 verdict: NEGATIVE (identity deviation {:.3e}, α-replays all match: {all_match}, split ratio {ratio:.3e}, negative probe fires: {})",
            identity.deviation, !negative.matches
        );
    }
}

/// The coupled-stall fixture — the same construction the gates use
/// (tests/issue945_qrr_sylvester_gates.rs): spiral block, diagonal stable
/// complement, SplitMix64-coupled complement rows, `[I₂; 0]` basis.
fn stall_fixture(r: f32, omega: f32, rho_a: f32) -> SylvesterBlocks {
    let (c, s) = (r * omega.cos(), r * omega.sin());
    let s_block = [[c, -s], [s, c]];
    let k = 4;
    let diag = [rho_a, 0.4, 0.3, 0.25];
    let a: Vec<Vec<f32>> = (0..k)
        .map(|i| (0..k).map(|j| if i == j { diag[i] } else { 0.0 }).collect())
        .collect();
    let mut state = 0xABC_DEF0_1234_5678_u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = state;
        let x = z ^ (z >> 30);
        let x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let x = x ^ (x >> 27);
        let x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        let x = x ^ (x >> 31);
        ((x >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    };
    let mut b = vec![vec![0.0_f32; 2]; k + 2];
    for row in b.iter_mut().skip(2) {
        row[0] = next();
        row[1] = next();
    }
    let u: Vec<Vec<f32>> = (0..k + 2)
        .map(|i| vec![f32::from(i == 0), f32::from(i == 1)])
        .collect();
    SylvesterBlocks {
        s: s_block,
        a,
        b,
        u,
    }
}
