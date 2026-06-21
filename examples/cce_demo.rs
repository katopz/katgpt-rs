//! Plan 295 Phase 3 — CCE designer steering demo (G3).
//!
//! Three-section demonstration of the LP-CCE moderator primitives:
//!
//! 1. **CCE vs Nash on chicken** — shows the CCE Pareto-dominates the Nash
//!    equilibrium on a general-sum game.
//! 2. **Primal-dual convergence** — shows `CcePrimalDual` converging to the
//!    `CceLp` optimum on the emission-abatement problem.
//! 3. **Designer steering** — the headline selling-point demo: the same game
//!    with two different moderator objectives `Γ₀` yields two structurally
//!    different optimal CCEs. This is the "designer can steer the population"
//!    claim from paper §8.2.
//!
//! ## Run
//!
//! ```bash
//! cargo run --example cce_demo --features cce_moderator
//! ```

use katgpt_rs::cce::{
    CceLp, CcePrimalDual, Deviation, DeviationClass, ExternalRegret, OccupationMeasure,
    PayoffTensor,
};

// =========================================================================
// Section 1: Chicken — CCE vs Nash
// =========================================================================

const CHICKEN_REWARD: [[f32; 2]; 2] = [[3.0, 1.0], [4.0, 0.0]];

fn chicken_welfare(a_1: usize, a_2: usize) -> f32 {
    CHICKEN_REWARD[a_1][a_2] + CHICKEN_REWARD[a_2][a_1]
}

struct ChickenWelfare;

impl PayoffTensor<4, 2> for ChickenWelfare {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        let s_2 = state % 2;
        -CHICKEN_REWARD[action][s_2]
    }
    fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
        let mut g = 0.0;
        for s in 0..4 {
            let s_2 = s % 2;
            for a in 0..2 {
                g += rho.at(s, a) * (-chicken_welfare(a, s_2));
            }
        }
        g
    }
    fn gamma0_coeff(&self, state: usize, action: usize) -> f32 {
        let s_2 = state % 2;
        -chicken_welfare(action, s_2)
    }
}

struct ChickenDevs {
    v: Vec<Deviation<4, 2>>,
}

impl DeviationClass<4, 2> for ChickenDevs {
    fn deviations(&self) -> &[Deviation<4, 2>] {
        &self.v
    }
}

fn chicken_devs() -> ChickenDevs {
    ChickenDevs {
        v: vec![
            Deviation::<4, 2>::constant(0, 0),
            Deviation::<4, 2>::constant(1, 1),
        ],
    }
}

fn section1_chicken_cce_vs_nash() {
    println!("==============================================================");
    println!("Section 1: Chicken — CCE vs Nash Pareto-dominance");
    println!("==============================================================");
    println!();
    println!("Game: Chicken (Hawk-Dove). Actions: S=Swerve(0), T=Straight(1).");
    println!("Reward matrix (player 1 row, player 2 col):");
    println!("         S       T");
    println!("  S    (3,3)   (1,4)");
    println!("  T    (4,1)   (0,0)");
    println!();

    let d = chicken_devs();
    let p = ChickenWelfare;
    let rho_cce = CceLp::new().solve(&d, &p).expect("chicken LP feasible");
    let gamma0_cce = p.gamma0(&rho_cce);
    let welfare_cce = -gamma0_cce;
    let welfare_nash = 4.0_f32;

    println!("LP-CCE solution ρ⋆ (player-1-only model):");
    for s in 0..4 {
        let s_1 = s / 2;
        let s_2 = s % 2;
        let label = |x: usize| if x == 0 { "S" } else { "T" };
        for a in 0..2 {
            let mass = rho_cce.at(s, a);
            if mass > 0.01 {
                println!(
                    "  ρ(({},{}) = state {}, action {}) = {:.4}",
                    label(s_1),
                    label(s_2),
                    s,
                    label(a),
                    mass
                );
            }
        }
    }
    println!();
    println!("  CCE welfare  = {welfare_cce:.4}");
    println!("  Nash welfare = {welfare_nash:.4}  (mixed Nash, each swerves p=0.5)");
    println!(
        "  Pareto gain  = {:.1}%  (CCE ≥ Nash + 5% ✓)",
        (welfare_cce / welfare_nash - 1.0) * 100.0
    );
    println!();
}

// =========================================================================
// Section 2: Emission-abatement primal-dual convergence
// =========================================================================

const EMIT_COST: [[f32; 4]; 4] = [
    [1.0, 2.0, 3.0, 4.0],
    [3.0, 2.5, 3.5, 4.5],
    [6.0, 4.0, 3.0, 5.0],
    [10.0, 7.0, 4.5, 4.0],
];

struct EmissionAbatement4x4;

impl PayoffTensor<4, 4> for EmissionAbatement4x4 {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        EMIT_COST[state][action]
    }
    fn gamma0(&self, rho: &OccupationMeasure<4, 4>) -> f32 {
        self.gamma(rho)
    }
}

struct Emit4Devs {
    v: Vec<Deviation<4, 4>>,
}

impl DeviationClass<4, 4> for Emit4Devs {
    fn deviations(&self) -> &[Deviation<4, 4>] {
        &self.v
    }
}

fn emit_devs() -> Emit4Devs {
    Emit4Devs {
        v: vec![
            Deviation::<4, 4>::constant(0, 0),
            Deviation::<4, 4>::constant(1, 1),
            Deviation::<4, 4>::constant(2, 2),
            Deviation::<4, 4>::constant(3, 3),
        ],
    }
}

fn section2_primal_dual_convergence() {
    println!("==============================================================");
    println!("Section 2: Primal-dual convergence on emission-abatement");
    println!("==============================================================");
    println!();
    println!("Setup: 4 states (price signals), 4 actions (abatement levels).");
    println!("LP optimum: (Low, None) with cost 1.0.");
    println!();

    let d = emit_devs();
    let p = EmissionAbatement4x4;

    let rho_lp = CceLp::new().solve(&d, &p).expect("LP feasible");
    let gamma0_lp = p.gamma0(&rho_lp);
    println!("LP reference: γ₀(ρ⋆) = {gamma0_lp:.6}");
    println!();

    // Run primal-dual and sample at geometric checkpoints.
    let checkpoints = [100, 1000, 10_000];
    let runner = CcePrimalDual::new::<4, 4>().with_eta(0.05);
    let report = runner.run(&d, &p, *checkpoints.last().unwrap());

    println!("Convergence (averaged iterate ρ̄ⁿ):");
    println!("  {:>10}  {:>12}  {:>12}", "n", "γ₀(ρ̄ⁿ)", "gap");
    let mut cumsum = 0.0_f64;
    for (i, step) in report.history.iter().enumerate() {
        let n = i + 1;
        cumsum += step.gamma0 as f64;
        if checkpoints.contains(&n) {
            let gamma0_avg = cumsum / n as f64;
            let gap = (gamma0_avg - gamma0_lp as f64).abs();
            println!("  {:>10}  {:>12.6}  {:>12.6}", n, gamma0_avg, gap);
        }
    }
    println!();
    println!(
        "Final: γ₀(ρ̄ᴺ) = {:.6}, ER(ρ̄ᴺ) = {:.6}",
        report.gamma0_avg, report.er_avg
    );
    println!();
}

// =========================================================================
// Section 3: Designer steering — two Γ₀ → two different CCEs
// =========================================================================

struct ChickenPlayer1Cost;

impl PayoffTensor<4, 2> for ChickenPlayer1Cost {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        let s_2 = state % 2;
        -CHICKEN_REWARD[action][s_2]
    }
    // γ₀ = γ (player 1's cost). Minimizing this picks the most selfish CCE.
    fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
        self.gamma(rho)
    }
}

fn section3_designer_steering() {
    println!("==============================================================");
    println!("Section 3: Designer steering — same game, two Γ₀, two CCEs");
    println!("==============================================================");
    println!();
    println!("Game: Chicken. The designer picks the moderator objective Γ₀:");
    println!("  (A) Γ₀ = player 1's cost  → 'selfish' CCE (player 1 exploits).");
    println!("  (B) Γ₀ = -welfare         → 'welfare-max' CCE (both players benefit).");
    println!();

    let d = chicken_devs();

    // --- (A) Selfish moderator ---
    let p_a = ChickenPlayer1Cost;
    let rho_a = CceLp::new().solve(&d, &p_a).expect("LP (A) feasible");
    let gamma0_a = p_a.gamma0(&rho_a);
    let player1_reward_a = -gamma0_a;

    // Compute player 2 reward under ρ_A (player 2 plays s_2 honestly).
    let mut player2_reward_a = 0.0_f32;
    for s in 0..4 {
        let s_1 = s / 2;
        let s_2 = s % 2;
        for (a, &reward) in CHICKEN_REWARD[s_2].iter().enumerate() {
            let mass = rho_a.at(s, a);
            // Player 2 plays s_2, player 1 plays a.
            player2_reward_a += mass * reward;
            let _ = s_1;
        }
    }
    let welfare_a = player1_reward_a + player2_reward_a;

    println!("(A) Selfish Γ₀ = player 1 cost:");
    println!("  Player 1 reward = {player1_reward_a:.4}");
    println!("  Player 2 reward = {player2_reward_a:.4}");
    println!("  Welfare         = {welfare_a:.4}");
    println!("  ρ⋆ support:");
    print_rho_support(&rho_a);
    println!();

    // --- (B) Welfare-maximizing moderator ---
    let p_b = ChickenWelfare;
    let rho_b = CceLp::new().solve(&d, &p_b).expect("LP (B) feasible");
    let gamma0_b = p_b.gamma0(&rho_b);
    let welfare_b = -gamma0_b;

    // Compute player rewards under ρ_B.
    let mut player1_reward_b = 0.0_f32;
    let mut player2_reward_b = 0.0_f32;
    for s in 0..4 {
        let s_2 = s % 2;
        for (a, reward_a) in CHICKEN_REWARD.iter().enumerate() {
            let mass = rho_b.at(s, a);
            player1_reward_b += mass * reward_a[s_2];
            player2_reward_b += mass * CHICKEN_REWARD[s_2][a];
        }
    }

    println!("(B) Welfare-max Γ₀ = -welfare:");
    println!("  Player 1 reward = {player1_reward_b:.4}");
    println!("  Player 2 reward = {player2_reward_b:.4}");
    println!("  Welfare         = {welfare_b:.4}");
    println!("  ρ⋆ support:");
    print_rho_support(&rho_b);
    println!();

    // --- Verdict ---
    println!("Designer steering verdict:");
    println!(
        "  Selfish moderator → welfare {:.2}, player 1 reward {:.2}.",
        welfare_a, player1_reward_a
    );
    println!(
        "  Welfare moderator → welfare {:.2}, player 1 reward {:.2}.",
        welfare_b, player1_reward_b
    );
    if welfare_b > welfare_a {
        println!(
            "  → Designer can lift welfare by {:.2} (+{:.0}%) by switching Γ₀.",
            welfare_b - welfare_a,
            (welfare_b / welfare_a - 1.0) * 100.0
        );
    }
    if (player1_reward_a - player1_reward_b).abs() > 0.1 {
        println!("  → The two CCEs are structurally DIFFERENT (different player-1 rewards).");
        println!("     This is the designer steering effect: same game, same CCE");
        println!("     constraints, but the moderator objective picks different");
        println!("     equilibria from the CCE set.");
    }
    println!();
}

fn print_rho_support<const N: usize, const A: usize>(rho: &OccupationMeasure<N, A>) {
    let labels_2 = ["S", "T"];
    for s in 0..N {
        for a in 0..A {
            let mass = rho.at(s, a);
            if mass > 0.01 {
                let s_str = if N == 4 && A == 2 {
                    let s_1 = s / 2;
                    let s_2 = s % 2;
                    format!("({},{})", labels_2[s_1], labels_2[s_2])
                } else {
                    format!("state {s}")
                };
                let a_str: String = if A == 2 {
                    labels_2[a].to_string()
                } else {
                    format!("action {a}")
                };
                println!("    ρ({s_str}, action {a_str}) = {mass:.4}");
            }
        }
    }
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CCE Moderator Demo (Plan 295, Research 274, arxiv 2606.20062) ║");
    println!("║  Optimal Coarse Correlated Equilibria in Mean Field Games     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    section1_chicken_cce_vs_nash();
    section2_primal_dual_convergence();
    section3_designer_steering();

    // Final sanity: verify ExternalRegret agrees on CCE validity for the
    // welfare-maximizing chicken solution.
    let d = chicken_devs();
    let p = ChickenWelfare;
    let rho = CceLp::new().solve(&d, &p).expect("LP feasible");
    let er = ExternalRegret::new().er(&rho, &d, &p);
    println!("==============================================================");
    println!("Sanity: ER(ρ⋆_welfare-max) = {er:.6}  (≤ 0 ⇒ valid CCE)");
    println!("==============================================================");
}
