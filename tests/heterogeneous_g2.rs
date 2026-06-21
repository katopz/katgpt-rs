//! Plan 300 Phase 4 — G2 regret-transfer gate on synthetic heterogeneous CWMs.
//!
//! Generates an 8-player game where each player's payoff tensor is a small
//! Gaussian perturbation around a base emission-style table. Runs
//! `solve_heterogeneous` and verifies `er_heterogeneous(ρ⋆) ≤ 1e-3`.
//!
//! The regret bound `ER(ρ̄_T) ≤ O(T⁻¹ᐟ²)` transfers from the homogeneous case
//! (doc 62 §2). With the LP solved exactly (not via primal-dual iteration),
//! the regret at the optimum should be machine-precision-zero — this gate
//! verifies that property as the prerequisite for the runtime primal-dual
//! path (G3, follow-up).
//!
//! ## Run
//!
//! ```bash
//! cargo test --features cce_moderator --test heterogeneous_g2 -- --nocapture
//! ```

#![cfg(feature = "cce_moderator")]

use katgpt_rs::cce::{
    CceLp, Deviation, DeviationClass, ExternalRegret, OccupationMeasure, PayoffTensor,
    PerPlayerGame,
};

/// Per-player payoff table: each entry is a small perturbation around a base.
struct PerturbedPlayer {
    /// Cost table `c[state][action]`. Length `N*A`, row-major.
    c: Vec<f32>,
}

impl PayoffTensor<2, 2> for PerturbedPlayer {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        self.c[state * 2 + action]
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

/// Deterministic per-player perturbation from a simple LCG (avoids pulling in
/// a `rand` dep just for this test). Perturbation scale: 1% of the base cost.
fn perturbed_cost(base: [[f32; 2]; 2], seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x123456789ABCDEF0;
    let mut out = Vec::with_capacity(4);
    for row in &base {
        for &val in row {
            // LCG step → uniform-ish in [-0.01, 0.01].
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32) - 0.5; // [-0.5, 0.5)
            let noise = u * 0.02; // [-0.01, 0.01)
            out.push((val + noise).max(0.01)); // keep positive
        }
    }
    out
}

#[test]
fn g2_regret_transfer_8_players() {
    const BASE: [[f32; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
    const N_PLAYERS: usize = 8;

    // Build 8 players, each with its own perturbed cost table.
    let players_tables: Vec<PerturbedPlayer> = (0..N_PLAYERS)
        .map(|i| PerturbedPlayer {
            c: perturbed_cost(BASE, i as u64 + 1),
        })
        .collect();

    // Shared deviation class (constant deviations).
    let d = EmitDevs {
        v: vec![
            Deviation::<2, 2>::constant(0, 0),
            Deviation::<2, 2>::constant(1, 1),
        ],
    };

    // Borrow each (table, devs) pair and build the game.
    let player_refs: Vec<(&PerturbedPlayer, &EmitDevs)> =
        players_tables.iter().map(|p| (p, &d)).collect();
    let game = PerPlayerGame::new(player_refs);

    // Solve.
    let rho_star = CceLp::new()
        .solve_heterogeneous(&game)
        .expect("G2: 8-player heterogeneous LP feasible");

    // Verify ρ⋆ is a valid subjective-CCE for every player's deviation class.
    assert!(
        CceLp::new().is_heterogeneous_cce(&rho_star, &game, 1e-3),
        "G2: ρ⋆ must satisfy subjective-CCE constraints (ε = 1e-3)"
    );

    // Regret bound check: er_heterogeneous must be ≤ 1e-3 at the LP optimum.
    let er = ExternalRegret::new().er_heterogeneous(&rho_star, &game);
    assert!(
        er <= 1e-3,
        "G2 FAIL: heterogeneous regret {er} should be ≤ 1e-3 at the LP optimum"
    );

    eprintln!("G2 PASS — 8-player heterogeneous regret = {er:.2e} (≤ 1e-3)");
}

/// Larger population (16 players) to verify the regret transfer holds at
/// moderate scale (still small enough for BFS enumeration).
#[test]
fn g2_regret_transfer_16_players() {
    const BASE: [[f32; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
    const N_PLAYERS: usize = 16;

    let players_tables: Vec<PerturbedPlayer> = (0..N_PLAYERS)
        .map(|i| PerturbedPlayer {
            c: perturbed_cost(BASE, i as u64 + 100),
        })
        .collect();

    let d = EmitDevs {
        v: vec![
            Deviation::<2, 2>::constant(0, 0),
            Deviation::<2, 2>::constant(1, 1),
        ],
    };

    let player_refs: Vec<(&PerturbedPlayer, &EmitDevs)> =
        players_tables.iter().map(|p| (p, &d)).collect();
    let game = PerPlayerGame::new(player_refs);

    let rho_star = CceLp::new()
        .solve_heterogeneous(&game)
        .expect("G2: 16-player heterogeneous LP feasible");

    assert!(
        CceLp::new().is_heterogeneous_cce(&rho_star, &game, 1e-3),
        "G2: ρ⋆ must satisfy subjective-CCE constraints (16 players)"
    );

    let er = ExternalRegret::new().er_heterogeneous(&rho_star, &game);
    assert!(
        er <= 1e-3,
        "G2 FAIL: 16-player heterogeneous regret {er} should be ≤ 1e-3"
    );

    eprintln!("G2 PASS — 16-player heterogeneous regret = {er:.2e} (≤ 1e-3)");
}
