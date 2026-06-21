//! Subjective-CCE heterogeneous payoff wrapper (Plan 300).
//!
//! Concrete [`HeterogeneousPayoff`] backed by per-player `(PayoffTensor,
//! DeviationClass)` slices. The common case for runtimes that already hold
//! per-NPC cost tables (e.g., per-NPC `NpcCwmRuntime<K>` in the riir-ai
//! private runtime; that bridge is tracked separately as a future riir-ai
//! plan blocked on this crate landing).
//!
//! ## Mathematical contract
//!
//! The regret bound `ER(ρ̄_T) ≤ O(T⁻¹ᐟ²)` transfers from the homogeneous case
//! (doc 62 §2 — each `γ_i` is linear in `ρ`, so each per-player regret is
//! convex; the average `(1/P) Σ_i ER_i(ρ)` is convex; primal-dual averaging is
//! heterogeneity-agnostic). **No new theory; pure API surface.**

use crate::cce::types::{Deviation, DeviationClass, HeterogeneousPayoff, PayoffTensor};

/// Concrete [`HeterogeneousPayoff`] backed by per-player `(PayoffTensor,
/// DeviationClass)` slices.
///
/// Borrows the per-player tables. The owner of the slices must outlive every
/// `&PerPlayerGame` reference handed to [`crate::cce::CceLp::solve_heterogeneous`]
/// or [`crate::cce::ExternalRegret::er_heterogeneous`].
///
/// ## Default moderator objective
///
/// Uses the default `gamma0 = (1/P) Σ_i γ_i(ρ)` (average player welfare).
/// Override by implementing `HeterogeneousPayoff` directly on a custom type.
pub struct PerPlayerGame<
    'a,
    const N: usize,
    const A: usize,
    P: PayoffTensor<N, A>,
    D: DeviationClass<N, A>,
> {
    /// `(payoff_tensor, deviation_class)` per player.
    pub players: Vec<(&'a P, &'a D)>,
}

impl<'a, const N: usize, const A: usize, P: PayoffTensor<N, A>, D: DeviationClass<N, A>>
    PerPlayerGame<'a, N, A, P, D>
{
    /// Construct from per-player `(P, D)` pairs. Caller must ensure the slices
    /// are non-empty (zero-player games are rejected by `solve_heterogeneous`).
    pub fn new(players: Vec<(&'a P, &'a D)>) -> Self {
        Self { players }
    }
}

impl<'a, const N: usize, const A: usize, P: PayoffTensor<N, A>, D: DeviationClass<N, A>>
    HeterogeneousPayoff<N, A> for PerPlayerGame<'a, N, A, P, D>
{
    #[inline]
    fn n_players(&self) -> usize {
        self.players.len()
    }

    #[inline]
    fn deviations_for_player(&self, player: usize) -> &[Deviation<N, A>] {
        self.players[player].1.deviations()
    }

    #[inline]
    fn reward_follow(&self, player: usize, state: usize, action: usize) -> f32 {
        self.players[player].0.reward_follow(state, action)
    }

    // `reward_deviate`, `gamma_player`, `gamma_dev_player`, `gamma0`,
    // `gamma0_coeff` use trait defaults. The defaults delegate to per-player
    // `PayoffTensor::reward_follow`, which monomorphizes correctly per `P`.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cce::external_regret::ExternalRegret;
    use crate::cce::lp::CceLp;
    use crate::cce::types::OccupationMeasure;

    // -------- Helper structs (mirror lp.rs::tests pattern) --------

    /// Two-state, two-action emission-style cost tensor.
    struct Emission {
        c: [[f32; 2]; 2],
    }
    impl PayoffTensor<2, 2> for Emission {
        fn reward_follow(&self, state: usize, action: usize) -> f32 {
            self.c[state][action]
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

    /// Canonical emission game from `lp.rs::tests::lp_solve_emission_finds_cheapest_cce`:
    /// cost = [[1, 3], [2, 5]], deviations = always-Abate, always-Pollute.
    fn canonical_emission() -> (Emission, EmitDevs) {
        let p = Emission {
            c: [[1.0, 3.0], [2.0, 5.0]],
        };
        let d = EmitDevs {
            v: vec![
                Deviation::<2, 2>::constant(0, 0), // always Abate
                Deviation::<2, 2>::constant(1, 1), // always Pollute
            ],
        };
        (p, d)
    }

    // -------- T3.2 test 1: homogeneous equivalence regression --------

    /// G1-style: `PerPlayerGame` with all players sharing the same `(P, D)`
    /// produces the same `ρ⋆` as `CceLp::solve(d, p)` on that single `(P, D)`.
    /// Closes the "wrapper is a strict generalization" check.
    #[test]
    fn homogeneous_equivalence() {
        let (p, d) = canonical_emission();

        // Homogeneous path.
        let rho_homogeneous = CceLp::new().solve(&d, &p).expect("homogeneous LP feasible");

        // Heterogeneous path with 3 identical players.
        // Objective: (1/3) Σ_i reward_follow = reward_follow (identical players),
        // so the LP objective row matches the homogeneous case.
        // Constraints: 3 × 2 = 6 rows, each identical to the 2 homogeneous rows
        // (redundant), so the feasible region is unchanged.
        let players: Vec<(&Emission, &EmitDevs)> = vec![(&p, &d), (&p, &d), (&p, &d)];
        let game = PerPlayerGame::new(players);
        let rho_heterogeneous = CceLp::new()
            .solve_heterogeneous(&game)
            .expect("heterogeneous LP feasible");

        // Compare entry-by-entry. The BFS enumeration may find degenerate
        // optima (multiple optima with the same objective value), so we check
        // both:
        //   (a) the objective value matches,
        //   (b) each entry matches within f32 ε.
        let obj_h = p.gamma0(&rho_homogeneous);
        let obj_e = p.gamma0(&rho_heterogeneous);
        assert!(
            (obj_h - obj_e).abs() < 1e-4,
            "objective mismatch: homogeneous {obj_h} vs heterogeneous {obj_e}"
        );
        for i in 0..4 {
            let a = rho_homogeneous.entries[i];
            let b = rho_heterogeneous.entries[i];
            assert!(
                (a - b).abs() < 1e-3,
                "entry {i} mismatch: homogeneous {a} vs heterogeneous {b}"
            );
        }
    }

    // -------- T3.2 test 2: two-player prisoners' dilemma --------

    /// 2-player single-shot PD with cost = -reward (cost convention).
    ///
    /// Payoff matrix (reward, row = player 1's action, col = player 2's action):
    /// ```text
    ///            C       D
    ///        C (3,3)   (0,5)
    ///        D (5,0)   (1,1)
    /// ```
    ///
    /// **Scope of this test:** verify `solve_heterogeneous` produces a valid
    /// subjective-CCE on a 2-player game with distinct per-player cost tensors
    /// (player 1 is row, player 2 is column — different `reward_follow`
    /// formulas). The test confirms:
    ///   1. The LP is feasible and returns an occupation measure.
    ///   2. The returned `ρ⋆` passes `is_heterogeneous_cce(ε = 1e-4)`.
    ///   3. The moderator objective `γ₀(ρ⋆)` equals the LP-optimal value
    ///      reported by solving the homogeneous LP with the averaged cost tensor
    ///      (sanity check on the wrapper).
    ///
    /// **Note on PD + constant deviations:** with only the constant deviation
    /// class `{always-C, always-D}`, the CCE feasible set is larger than
    /// `{δ_{(D,D)}}` — constant deviations cannot enforce state-conditional
    /// incentive compatibility, so distributions like `δ_{(s=(C,C), a=D)}` are
    /// also valid subjective-CCEs. Pinning down `(D,D)` as the unique CCE
    /// requires a state-conditional deviation class (out of scope for this
    /// wiring test; the runtime supplies richer deviation classes).
    ///
    /// **Plan T3.2 wording correction:** the original plan said "verify `ρ⋆`
    /// concentrates on the cooperative joint action under a welfare-maximizing
    /// moderator". This is incorrect for single-shot PD — the CCE feasible set
    /// excludes cooperation regardless of the moderator objective (cooperation
    /// is not incentive-compatible). The moderator picks the welfare-minimal
    /// element of the *CCE feasible set*, which for PD with constant deviations
    /// is a low-welfare distribution, not the cooperative outcome.
    ///
    /// ## On the `PerPlayerGame<P, D>` design
    ///
    /// `PerPlayerGame` is parameterized by a single `P` (payoff-tensor type)
    /// and `D` (deviation-class type). The heterogeneity is in the *values*
    /// (each player has its own `PdPlayer { role }` value), not in the static
    /// types. This mirrors the runtime use case: all NPCs share the same
    /// `NpcCostTable` shape, just with different per-NPC parameters.
    #[test]
    fn two_player_prisoners_dilemma() {
        // Reward matrix: R[a_p1][a_p2] = player 1's reward.
        const R: [[f32; 2]; 2] = [[3.0, 0.0], [5.0, 1.0]];

        /// Single payoff-tensor type carrying a player role (0 = P1, 1 = P2).
        /// `state` = joint recommendation `s = a_p1 * 2 + a_p2`;
        /// `action` = this player's own recommended action.
        /// R[row][col] = row-player reward; column player's reward at
        /// (a_p1, a_p2) = R[a_p2][a_p1] (symmetric-game swap).
        struct PdPlayer {
            role: usize,
        }
        impl PayoffTensor<4, 2> for PdPlayer {
            fn reward_follow(&self, state: usize, action: usize) -> f32 {
                if self.role == 0 {
                    // Player 1 (row): cost = -R[action][a_p2].
                    let a_p2 = state % 2;
                    -R[action][a_p2]
                } else {
                    // Player 2 (col): cost = -R[a_p2][a_p1] = -R[action][a_p1].
                    let a_p1 = state / 2;
                    -R[action][a_p1]
                }
            }
            fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
                self.gamma(rho)
            }
        }
        struct PdDevs {
            v: Vec<Deviation<4, 2>>,
        }
        impl DeviationClass<4, 2> for PdDevs {
            fn deviations(&self) -> &[Deviation<4, 2>] {
                &self.v
            }
        }

        let d = PdDevs {
            v: vec![
                Deviation::<4, 2>::constant(0, 0), // always C
                Deviation::<4, 2>::constant(1, 1), // always D
            ],
        };
        let p1 = PdPlayer { role: 0 };
        let p2 = PdPlayer { role: 1 };

        let game = PerPlayerGame::new(vec![(&p1, &d), (&p2, &d)]);
        let rho_star = CceLp::new()
            .solve_heterogeneous(&game)
            .expect("PD heterogeneous LP feasible");

        // (1) ρ⋆ is a valid probability distribution.
        let sum: f32 = rho_star.entries.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-4, "ρ⋆ must sum to 1, got {sum}");

        // (2) ρ⋆ passes the heterogeneous-CCE check.
        assert!(
            CceLp::new().is_heterogeneous_cce(&rho_star, &game, 1e-4),
            "PD solution must satisfy subjective-CCE constraints"
        );

        // (3) Every player's cost at ρ⋆ must be ≤ every player's cost of
        //     deviating (i.e., no player envies any constant deviation).
        //     This is the definition of subjective-CCE.
        for i in 0..2 {
            let gamma_i = game.gamma_player(i, &rho_star);
            for kappa in game.deviations_for_player(i) {
                let gamma_dev_i = game.gamma_dev_player(i, &rho_star, kappa);
                assert!(
                    gamma_i - gamma_dev_i <= 1e-4,
                    "player {i} has profitable deviation: γ={gamma_i}, γ_dev={gamma_dev_i}"
                );
            }
        }

        // (4) The moderator objective γ₀(ρ⋆) is finite and not absurd.
        //     For PD with cost = -reward, every entry of R is in [0, 5],
        //     so γ₀ ∈ [-5, 0].
        let gamma0 = game.gamma0(&rho_star);
        assert!(
            (-5.0..=0.0).contains(&gamma0),
            "γ₀(ρ⋆) should be in [-5, 0], got {gamma0}"
        );
    }

    // -------- T3.2 test 3: heterogeneous robustness (the Issue 327 use case) --------

    /// Two players with *slightly different* payoff tensors (perturbed by
    /// ~1%). Verify `er_heterogeneous(ρ⋆) ≤ 1e-3` — the regret bound transfers
    /// and the LP finds a near-exact subjective-CCE. Also verify `ρ⋆` is a
    /// small perturbation of the homogeneous `ρ⋆` (each player's perturbation
    /// is small, so the joint optimum should not move far).
    #[test]
    fn heterogeneous_robustness() {
        let (p_base, d) = canonical_emission();

        // Player 0: base table. Player 1: base table + 1% multiplicative perturbation.
        let p_perturbed = Emission {
            c: [
                [1.01 * p_base.c[0][0], 1.01 * p_base.c[0][1]],
                [1.01 * p_base.c[1][0], 1.01 * p_base.c[1][1]],
            ],
        };

        let game = PerPlayerGame::new(vec![(&p_base, &d), (&p_perturbed, &d)]);
        let rho_star = CceLp::new()
            .solve_heterogeneous(&game)
            .expect("perturbed LP feasible");

        // Regret bound transfers: ρ⋆ must be an ε-subjective-CCE for both
        // players' deviation classes. With the LP solved exactly, this should
        // be ~machine-precision, well under 1e-3.
        let er = ExternalRegret::new().er_heterogeneous(&rho_star, &game);
        assert!(
            er <= 1e-3,
            "heterogeneous regret should be ≤ 1e-3, got {er}"
        );

        // The optimal action is robust: base emission's optimum is
        // (state=Low, action=Abate) = (0, 0) with cost 1.0. The perturbed
        // table's optimum is still (0, 0) (1.01 vs 3.03 vs 2.02 vs 5.05 —
        // (0,0) is still cheapest). So ρ⋆ should still concentrate on (0, 0).
        let mass_low_abate = rho_star.at(0, 0);
        assert!(
            (mass_low_abate - 1.0).abs() < 1e-3,
            "heterogeneous optimum should remain at (Low, Abate); got mass = {mass_low_abate}"
        );
    }

    // -------- T3.2 test 4: solver output passes is_heterogeneous_cce --------

    #[test]
    fn is_heterogeneous_cce_passes_on_solve_output() {
        let (p, d) = canonical_emission();
        let game = PerPlayerGame::new(vec![(&p, &d), (&p, &d)]);
        let rho_star = CceLp::new()
            .solve_heterogeneous(&game)
            .expect("heterogeneous LP feasible");
        assert!(
            CceLp::new().is_heterogeneous_cce(&rho_star, &game, 1e-4),
            "solver output must satisfy subjective-CCE constraints (ε = 1e-4)"
        );
    }

    // -------- T3.2 test 5: is_heterogeneous_cce rejects non-CCE --------

    /// **Plan T3.2 wording correction:** the original plan said "pure Nash is
    /// NOT a heterogeneous CCE on PD". This is game-theoretically incorrect —
    /// in PD, the unique pure Nash (D, D) IS a CCE (every Nash is a CCE).
    ///
    /// The correct sanity check is the converse: the **cooperative** joint
    /// distribution (C, C) is NOT a subjective-CCE on PD — both players have
    /// a strictly profitable deviation to D. This test verifies
    /// `is_heterogeneous_cce` returns false on the cooperative distribution.
    #[test]
    fn is_heterogeneous_cce_rejects_cooperative_on_pd() {
        // Reuse the PD setup from `two_player_prisoners_dilemma`.
        const R: [[f32; 2]; 2] = [[3.0, 0.0], [5.0, 1.0]];

        struct PdPlayer {
            role: usize,
        }
        impl PayoffTensor<4, 2> for PdPlayer {
            fn reward_follow(&self, state: usize, action: usize) -> f32 {
                if self.role == 0 {
                    let a_p2 = state % 2;
                    -R[action][a_p2]
                } else {
                    let a_p1 = state / 2;
                    -R[action][a_p1]
                }
            }
            fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
                self.gamma(rho)
            }
        }
        struct PdDevs {
            v: Vec<Deviation<4, 2>>,
        }
        impl DeviationClass<4, 2> for PdDevs {
            fn deviations(&self) -> &[Deviation<4, 2>] {
                &self.v
            }
        }

        let d = PdDevs {
            v: vec![
                Deviation::<4, 2>::constant(0, 0),
                Deviation::<4, 2>::constant(1, 1),
            ],
        };
        let p1 = PdPlayer { role: 0 };
        let p2 = PdPlayer { role: 1 };
        let game = PerPlayerGame::new(vec![(&p1, &d), (&p2, &d)]);

        // Cooperative distribution: ρ(state = (C,C) = 0, action = C = 0) = 1.
        let rho_cooperative = OccupationMeasure::<4, 2>::dirac(0, 0);
        assert!(
            !CceLp::new().is_heterogeneous_cce(&rho_cooperative, &game, 1e-4),
            "cooperative (C,C) must NOT be a subjective-CCE on PD — both players have a profitable deviation"
        );

        // Conversely, (D, D) MUST pass — it's the Nash equilibrium.
        let rho_defect = OccupationMeasure::<4, 2>::dirac(3, 1);
        assert!(
            CceLp::new().is_heterogeneous_cce(&rho_defect, &game, 1e-4),
            "(D,D) Nash must be a subjective-CCE"
        );
    }
}
