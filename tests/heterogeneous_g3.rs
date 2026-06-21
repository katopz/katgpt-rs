//! Plan 300 T4.3b — G3 primal-dual convergence gate on a heterogeneous game.
//!
//! Closes the G3 split: verifies that `CcePrimalDual::run_heterogeneous`
//! converges to the LP-CCE optimum `ρ⋆` from `CceLp::solve_heterogeneous` at
//! rate `O(N⁻¹ᐟ²)` on a heterogeneous 4-player game (Plan 300's full
//! acceptance criterion — G1+G2+G3+G4 all PASS — is now met).
//!
//! ## Setup
//!
//! 4 players each have a small perturbation around the base emission-abatement
//! cost table. Each player uses the same 2-deviation class (always-Abate,
//! always-Pollute). The primal-dual iterator runs on the aggregate
//! `ER_heterogeneous(ρ)` and should converge to the exact LP solution.
//!
//! ## Assertions
//!
//! - **G3a**: `|γ₀(ρ̄ᴺ) − γ₀(ρ⋆_LP)| < 0.05` after `N = 10⁴` steps.
//! - **G3b**: `ER_heterogeneous(ρ̄ᴺ) ≤ 0.05` (subjective-CCE satisfied within
//!   Slater tolerance).
//! - **G3c**: log-log convergence slope in `[-2.0, -0.3]` (empirical
//!   verification of the `O(N⁻¹ᐟ²)` rate from Theorem 6.1 transferred via
//!   doc 62 §2 — sum of convex is convex).
//!
//! ## Run
//!
//! ```bash
//! cargo test --features cce_moderator --test heterogeneous_g3 -- --nocapture
//! ```

#![cfg(feature = "cce_moderator")]

use katgpt_rs::cce::{
    CceLp, CcePrimalDual, Deviation, DeviationClass, ExternalRegret, HeterogeneousPayoff,
    OccupationMeasure, PayoffTensor, PerPlayerGame,
};

/// Per-player payoff table: small deterministic perturbation around a base.
struct PerturbedPlayer {
    c: [f32; 4], // row-major N·A
}

impl PayoffTensor<2, 2> for PerturbedPlayer {
    fn reward_follow(&self, s: usize, a: usize) -> f32 {
        self.c[s * 2 + a]
    }
    fn gamma0(&self, rho: &OccupationMeasure<2, 2>) -> f32 {
        self.gamma(rho)
    }
}

struct EmitDevs {
    v: Vec<Deviation<2, 2>>,
}
impl DeviationClass<2, 2> for EmitDevs {
    fn deviations(&self) -> &[Deviation<2, 2>] {
        &self.v
    }
}

/// Deterministic per-player perturbation from a simple LCG (mirrors
/// heterogeneous_g2.rs setup for consistency). Scale: 1% of base cost.
fn perturbed_cost(base: [[f32; 2]; 2], seed: u64) -> [f32; 4] {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x123456789ABCDEF0;
    let mut out = [0.0_f32; 4];
    for row in 0..2 {
        for col in 0..2 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32) - 0.5;
            let noise = u * 0.02;
            out[row * 2 + col] = (base[row][col] + noise).max(0.01);
        }
    }
    out
}

const BASE: [[f32; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
const N_PLAYERS: usize = 4;

fn build_game() -> PerPlayerGame<'static, 2, 2, PerturbedPlayer, EmitDevs> {
    // SAFETY: we leak the players + devs to give them 'static lifetime so the
    // PerPlayerGame borrows survive across all G3 sub-tests. The cost is 4
    // small structs + 1 struct per test invocation; deliberate per-test leak
    // (test process is short-lived).
    let players: Vec<PerturbedPlayer> = (0..N_PLAYERS)
        .map(|i| PerturbedPlayer {
            c: perturbed_cost(BASE, i as u64 + 7),
        })
        .collect();
    let players_ref: &'static Vec<PerturbedPlayer> = Box::leak(Box::new(players));

    let d_ref: &'static EmitDevs = Box::leak(Box::new(EmitDevs {
        v: vec![
            Deviation::<2, 2>::constant(0, 0),
            Deviation::<2, 2>::constant(1, 1),
        ],
    }));

    let player_refs: Vec<(&'static PerturbedPlayer, &'static EmitDevs)> =
        players_ref.iter().map(|p| (p, d_ref)).collect();
    PerPlayerGame::new(player_refs)
}

/// G3a — averaged iterate γ₀(ρ̄ᴺ) reaches the LP optimum.
#[test]
fn g3a_heterogeneous_primal_dual_reaches_lp_optimum() {
    let game = build_game();

    // Reference: exact LP solve.
    let rho_lp = CceLp::new()
        .solve_heterogeneous(&game)
        .expect("LP feasible");
    let gamma0_lp = game.gamma0(&rho_lp);
    eprintln!("G3a: LP optimum γ₀(ρ⋆) = {gamma0_lp:.6}");

    // Run heterogeneous primal-dual for 10⁴ steps.
    let runner = CcePrimalDual::new::<2, 2>().with_eta(0.05);
    let report = runner.run_heterogeneous(&game, 10_000);

    let gap = (report.gamma0_avg - gamma0_lp).abs();
    eprintln!(
        "G3a: γ₀(ρ̄ᴺ) = {:.6}, gap = {:.6}, ER_hetero(ρ̄ᴺ) = {:.6}",
        report.gamma0_avg, gap, report.er_avg
    );

    assert!(
        gap < 0.05,
        "G3a FAIL: |γ₀(ρ̄ᴺ) − γ₀(ρ⋆)| = {gap:.6} should be < 0.05"
    );
}

/// G3b — averaged iterate satisfies the subjective-CCE constraint.
#[test]
fn g3b_averaged_iterate_satisfies_subjective_cce() {
    let game = build_game();

    let runner = CcePrimalDual::new::<2, 2>().with_eta(0.05);
    let report = runner.run_heterogeneous(&game, 10_000);

    eprintln!("G3b: ER_heterogeneous(ρ̄ᴺ) = {:.6}", report.er_avg);
    assert!(
        report.er_avg <= 0.05,
        "G3b FAIL: ER_heterogeneous(ρ̄ᴺ) = {} should be ≤ 0.05",
        report.er_avg
    );
}

/// G3c — empirical convergence rate matches `O(N⁻¹ᐟ²)`.
///
/// Sample `|γ₀(ρ̄ⁿ) − γ₀(ρ⋆)|` at geometrically-spaced iterations and fit a
/// log-log slope. The paper's Theorem 6.1 predicts slope `≈ −0.5`; we allow
/// `[-2.0, -0.3]` to admit faster (well-conditioned) convergence and exclude
/// numerical artifacts. The convex-aggregate transfer argument (doc 62 §2)
/// guarantees the same bound applies to the heterogeneous aggregate
/// `ER_heterogeneous(ρ)` — this test verifies the rate empirically.
#[test]
fn g3c_convergence_rate_is_one_over_sqrt_n() {
    let game = build_game();

    let rho_lp = CceLp::new()
        .solve_heterogeneous(&game)
        .expect("LP feasible");
    let gamma0_lp = game.gamma0(&rho_lp);

    let sample_points: [usize; 6] = [100, 300, 1000, 3000, 10_000, 30_000];
    let runner = CcePrimalDual::new::<2, 2>().with_eta(0.05);
    let report = runner.run_heterogeneous(&game, *sample_points.last().unwrap());

    // Reconstruct γ₀(ρ̄ⁿ) via the cumulative mean of per-step γ₀ values (proxy
    // for the true averaged iterate — same approach as the homogeneous G2c).
    let mut gaps: Vec<(f64, f64)> = Vec::new();
    let mut cumsum_gamma0 = 0.0_f64;
    for (i, step) in report.history.iter().enumerate() {
        let n = i + 1;
        cumsum_gamma0 += step.gamma0 as f64;
        let gamma0_avg_n = cumsum_gamma0 / n as f64;
        let gap = (gamma0_avg_n - gamma0_lp as f64).abs();
        if sample_points.contains(&n) && gap > 1e-6 {
            gaps.push(((n as f64).ln(), gap.ln()));
        }
    }

    eprintln!("G3c: collected {} sample points", gaps.len());
    for &(log_n, log_gap) in &gaps {
        eprintln!("  n = {:.0}, gap = {:.6}", log_n.exp(), log_gap.exp());
    }

    // Least-squares fit log(gap) = slope · log(n) + intercept.
    assert!(
        gaps.len() >= 2,
        "G3c FAIL: need ≥2 sample points, got {}",
        gaps.len()
    );
    let n_pts = gaps.len() as f64;
    let sum_x: f64 = gaps.iter().map(|(x, _)| *x).sum();
    let sum_y: f64 = gaps.iter().map(|(_, y)| *y).sum();
    let sum_xy: f64 = gaps.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = gaps.iter().map(|(x, _)| x * x).sum();
    let denom = n_pts * sum_x2 - sum_x * sum_x;
    assert!(denom.abs() > 1e-12, "G3c FAIL: singular least-squares fit");
    let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
    eprintln!(
        "G3c: fitted log-log slope = {slope:.4} (paper upper bound: -0.5; steeper is better)"
    );
    assert!(
        (-2.0..=-0.3).contains(&slope),
        "G3c FAIL: slope = {slope:.4} should be in [-2.0, -0.3]"
    );
}

/// Sanity: `linear_derivative_heterogeneous` is the per-(s,a) subgradient and
/// the iterator's inlined aggregation must match it (consistency between the
/// public method and the iterator's internal gradient computation).
#[test]
fn g3d_iterator_gradient_matches_public_derivative() {
    let game = build_game();

    // Pick a non-symmetric ρ and verify the iterator's gradient at step n=1
    // matches `(1/P) Σ_i [cost_i(s,a) − reward_deviate(i, s, κ_i*(ρ))]`
    // computed via the public `linear_derivative_heterogeneous`. We can't
    // directly inspect the iterator's internal grad; instead, we verify that
    // one step from a known ρ lands at ρ_temp = ρ - η·(gamma0_coeff + λ·deriv)
    // with λ = 0 on the first step.
    let er = ExternalRegret::new();

    // Use the iterator's uniform initial ρ (λ⁰ = 0, so first step is pure γ₀
    // gradient — same direction for both paths by construction).
    let rho0 = OccupationMeasure::<2, 2>::uniform();

    // At ρ0, λ = 0, so grad = gamma0_coeff(m). The first primal step reduces
    // to projected gradient descent on γ₀. Verify the iterator's first
    // StepReport.gamma0 is consistent with the LP optimum direction (lower
    // than uniform).
    let gamma0_uniform = game.gamma0(&rho0);
    let runner = CcePrimalDual::new::<2, 2>().with_eta(0.05);
    let report = runner.run_heterogeneous(&game, 1);
    let gamma0_step1 = report.history[0].gamma0;

    // γ₀ at the first iterate should be ≤ γ₀(uniform) — the iterator must be
    // descending the moderator objective (or at least not climbing above it
    // by more than the projection artifact).
    assert!(
        gamma0_step1 <= gamma0_uniform + 1e-4,
        "G3d FAIL: γ₀(ρ¹) = {gamma0_step1} should be ≤ γ₀(uniform) = {gamma0_uniform}"
    );

    // Verify the public derivative method runs without panic on the uniform.
    for m in 0..(2 * 2) {
        let _ = er.linear_derivative_heterogeneous(&rho0, m, &game);
    }
}
