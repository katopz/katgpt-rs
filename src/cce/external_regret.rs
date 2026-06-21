//! External-regret functional on a finite deviation class (Plan 295 Phase 1).
//!
//! `ER(ρ) = max_{κ ∈ D} (Γ(ρ) − Γ_dev(ρ, κ))`.
//!
//! With the **cost** convention used in this module (see [`crate::cce::PayoffTensor`]):
//! - `ER = 0` at a Nash equilibrium (marginal CCE).
//! - `ER < 0` at a strict CCE (every deviation strictly worse).
//! - `ER > 0` is NOT a CCE — a profitable deviation exists.
//!
//! The linear derivative (Lemma 6.5) of `ER` w.r.t. `ρ[m]` at a point where
//! the maximizer `κ*` is unique:
//!
//! ```text
//! ∂ER/∂ρ[m] = cost_follow(s, a) − cost_deviate(s, κ*(ρ))
//! ```
//!
//! where `m = (s, a)` flat. Used by the primal-dual iterator (Phase 2) for the
//! primal gradient step.

use crate::cce::types::{
    Deviation, DeviationClass, HeterogeneousPayoff, OccupationMeasure, PayoffTensor,
};

/// Closed-form external-regret functional on a finite deviation class `D`.
///
/// Stateless — every method takes `&D` and `&P` as input. The type is
/// parameterized only at the method level (`<const N, const A, D, P>`) so the
/// same instance can be reused across games of different sizes.
#[derive(Debug, Default)]
pub struct ExternalRegret;

impl ExternalRegret {
    /// Construct a stateless regret evaluator.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// `ER(ρ) = max_{κ ∈ D} (γ(ρ) − γ_dev(ρ, κ))`.
    ///
    /// Returns `f32::NEG_INFINITY` if `D` is empty.
    pub fn er<const N: usize, const A: usize, D: DeviationClass<N, A>, P: PayoffTensor<N, A>>(
        &self,
        rho: &OccupationMeasure<N, A>,
        d: &D,
        p: &P,
    ) -> f32 {
        let gamma = p.gamma(rho);
        let mut best = f32::NEG_INFINITY;
        for kappa in d.deviations() {
            let val = gamma - p.gamma_dev(rho, kappa);
            if val > best {
                best = val;
            }
        }
        best
    }

    /// Argmax `κ* = argmax_{κ ∈ D} (γ(ρ) − γ_dev(ρ, κ))` — the most profitable
    /// deviation (largest cost reduction by deviating).
    ///
    /// Returns `None` if `D` is empty.
    pub fn best_deviation<
        'a,
        const N: usize,
        const A: usize,
        D: DeviationClass<N, A>,
        P: PayoffTensor<N, A>,
    >(
        &self,
        rho: &OccupationMeasure<N, A>,
        d: &'a D,
        p: &P,
    ) -> Option<&'a Deviation<N, A>> {
        let gamma = p.gamma(rho);
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx: Option<usize> = None;
        for (i, kappa) in d.deviations().iter().enumerate() {
            let val = gamma - p.gamma_dev(rho, kappa);
            match best_idx {
                None => {
                    best_val = val;
                    best_idx = Some(i);
                }
                Some(_) if val > best_val => {
                    best_val = val;
                    best_idx = Some(i);
                }
                _ => {}
            }
        }
        best_idx.map(|i| &d.deviations()[i])
    }

    /// Check Assumption 6.2 (unique maximizer up to `ε`).
    ///
    /// Sort the per-deviation values `γ(ρ) − γ_dev(ρ, κ)` in descending order
    /// and verify that the top-2 gap exceeds `ε`. Returns `true` if `D` has
    /// fewer than 2 deviations (vacuously unique).
    pub fn is_unique_maximizer<
        const N: usize,
        const A: usize,
        D: DeviationClass<N, A>,
        P: PayoffTensor<N, A>,
    >(
        &self,
        rho: &OccupationMeasure<N, A>,
        d: &D,
        p: &P,
        eps: f32,
    ) -> bool {
        let devs = d.deviations();
        if devs.len() < 2 {
            return true;
        }
        let gamma = p.gamma(rho);
        let mut values: Vec<f32> = devs.iter().map(|k| gamma - p.gamma_dev(rho, k)).collect();
        // Sort descending so values[0] is the max.
        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        (values[0] - values[1]) > eps
    }

    /// Linear derivative `∂ER/∂ρ[m]` (Lemma 6.5) at flat index `m = (s, a)`.
    ///
    /// Equals `cost_follow(s, a) − cost_deviate(s, κ*(ρ))` where `κ*(ρ)` is
    /// the best deviation at `ρ`. Requires `D` non-empty; assumes `κ*` is
    /// unique (otherwise this is one element of the subgradient).
    pub fn linear_derivative<
        const N: usize,
        const A: usize,
        D: DeviationClass<N, A>,
        P: PayoffTensor<N, A>,
    >(
        &self,
        rho: &OccupationMeasure<N, A>,
        m_flat: usize,
        d: &D,
        p: &P,
    ) -> f32 {
        let (s, a) = OccupationMeasure::<N, A>::unflat_index(m_flat);
        let kappa_star = self
            .best_deviation(rho, d, p)
            .expect("linear_derivative: deviation class must be non-empty");
        p.reward_follow(s, a) - p.reward_deviate(s, kappa_star)
    }

    /// Linear derivative `∂ER_heterogeneous/∂ρ[m]` at flat index `m = (s, a)`
    /// (Plan 300 T4.3b).
    ///
    /// Equals `(1/P) Σ_i [cost_i(s, a) − reward_deviate(i, s, κ_i*(ρ))]`
    /// where `κ_i*(ρ)` is player `i`'s best deviation at `ρ`. This is the
    /// subgradient of the convex aggregate `ER_heterogeneous(ρ)` — sum of
    /// per-player subgradients, each of which is a single-valued selection
    /// from `∂(γ_i − γ_dev_i(·, κ_i*))`.
    ///
    /// Players with empty deviation classes contribute 0 (vacuous CCE).
    /// Requires `P ≥ 1`; returns 0 if `P == 0`.
    ///
    /// **Cost note:** recomputes the best deviation per player on every call.
    /// When called inside a per-`(s, a)` loop (as in `CcePrimalDual`), this
    /// is `O(N·A·P·|D_i|·N·A)` per iteration — wasteful. The primal-dual
    /// iterator instead caches best deviations once per step and inlines the
    /// per-index aggregation. This public method exists for testing and for
    /// callers who evaluate only one (s, a) per step.
    pub fn linear_derivative_heterogeneous<
        const N: usize,
        const A: usize,
        H: HeterogeneousPayoff<N, A>,
    >(
        &self,
        rho: &OccupationMeasure<N, A>,
        m_flat: usize,
        game: &H,
    ) -> f32 {
        let (s, a) = OccupationMeasure::<N, A>::unflat_index(m_flat);
        let p = game.n_players();
        if p == 0 {
            return 0.0;
        }
        let inv_p = 1.0 / p as f32;
        let mut g = 0.0_f32;
        for i in 0..p {
            // Find player i's best deviation κ_i*(ρ).
            let gamma_i = game.gamma_player(i, rho);
            let mut best_val = f32::NEG_INFINITY;
            let mut best_kappa: Option<&Deviation<N, A>> = None;
            for kappa in game.deviations_for_player(i) {
                let val = gamma_i - game.gamma_dev_player(i, rho, kappa);
                if val > best_val {
                    best_val = val;
                    best_kappa = Some(kappa);
                }
            }
            // Subgradient component: cost_i(s, a) − reward_deviate(i, s, κ_i*).
            if let Some(kappa) = best_kappa {
                g += game.reward_follow(i, s, a) - game.reward_deviate(i, s, kappa);
            }
            // Empty deviation class: contributes 0 (vacuous).
        }
        g * inv_p
    }

    /// Heterogeneous external regret (Plan 300 T2.3).
    ///
    /// `ER_hetero(ρ) = (1/P) Σ_i max_{κ ∈ D_i} (γ_i(ρ) − γ_dev_i(ρ, κ))`.
    ///
    /// Convex by construction (sum of convex per-player regrets). The
    /// `O(T⁻¹ᐟ²)` convergence bound transfers from the homogeneous case
    /// (doc 62 §2). Returns `f32::NEG_INFINITY` only if `P = 0`; returns
    /// `0.0` if every player has an empty deviation class (vacuous CCE).
    pub fn er_heterogeneous<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        &self,
        rho: &OccupationMeasure<N, A>,
        game: &H,
    ) -> f32 {
        let p = game.n_players();
        if p == 0 {
            return f32::NEG_INFINITY;
        }
        let inv_p = 1.0 / p as f32;
        let mut total = 0.0_f32;
        for i in 0..p {
            let gamma_i = game.gamma_player(i, rho);
            let mut best_for_i = f32::NEG_INFINITY;
            for kappa in game.deviations_for_player(i) {
                let val = gamma_i - game.gamma_dev_player(i, rho, kappa);
                if val > best_for_i {
                    best_for_i = val;
                }
            }
            // Empty deviation class → vacuous: this player contributes 0.
            if best_for_i == f32::NEG_INFINITY {
                best_for_i = 0.0;
            }
            total += best_for_i;
        }
        total * inv_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cce::types::{OccupationMeasureError, StateSpace};

    // ------- Test scaffolding: 3-action single-agent recommendation problem -------
    //
    // Setup: state `s` is the recommended action (S = A = {0, 1, 2}). Costs are
    // negative rewards: cost(s, a) = -R[a] where R = [1.0, 2.0, 3.0]. The cost
    // does NOT depend on `s` — the "state" is purely the mediator's signal.
    //
    // Honest mediator: ρ lives on the diagonal `ρ((s,s)) = μ(s)`.
    //   γ(ρ) = -Σ_s μ(s)·R[s]
    //   γ_dev(ρ, κ) = -Σ_s μ(s)·R[κ(s)]
    //
    // Deviation class: identity (follow) + 3 constant deviations.

    /// Linear cost tensor `cost(s, a) = -rewards[a]`.
    struct LinearCost {
        rewards: [f32; 3],
    }

    impl PayoffTensor<3, 3> for LinearCost {
        fn reward_follow(&self, _state: usize, action: usize) -> f32 {
            -self.rewards[action]
        }
        fn gamma0(&self, rho: &OccupationMeasure<3, 3>) -> f32 {
            self.gamma(rho)
        }
    }

    /// Deviation class with 4 deviations: identity + 3 constant.
    struct ThreeActionDevs {
        devs: Vec<Deviation<3, 3>>,
    }

    impl DeviationClass<3, 3> for ThreeActionDevs {
        fn deviations(&self) -> &[Deviation<3, 3>] {
            &self.devs
        }
    }

    fn make_dev_class() -> ThreeActionDevs {
        ThreeActionDevs {
            devs: vec![
                Deviation::<3, 3>::identity(0),
                Deviation::<3, 3>::constant(1, 0),
                Deviation::<3, 3>::constant(2, 1),
                Deviation::<3, 3>::constant(3, 2),
            ],
        }
    }

    fn linear_cost() -> LinearCost {
        LinearCost {
            rewards: [1.0, 2.0, 3.0],
        }
    }

    // -------- T1.4 canonical tests --------

    /// `test_er_zero_on_nash` — Dirac on the Nash point (always recommend and
    /// play the best action `a*`) yields `ER = 0`.
    #[test]
    fn test_er_zero_on_nash() {
        // Nash: ρ = δ_{(2,2)} — always recommend action 2 (reward 3, best).
        let rho = OccupationMeasure::<3, 3>::dirac(2, 2);
        let d = make_dev_class();
        let p = linear_cost();
        let val = ExternalRegret::new().er(&rho, &d, &p);

        // γ(ρ) = -R[2] = -3.
        // Best deviation: any κ with κ(2)[2] = 1 (identity or always-2) gives
        // γ_dev = -3. Others are worse (less negative). ER = -3 - (-3) = 0.
        assert!(
            val.abs() < 1e-6,
            "ER should be 0 at Nash (Dirac on best action), got {val}"
        );
    }

    /// `test_er_positive_off_nash` — perturb ρ off the Nash point so a
    /// profitable deviation appears, hence `ER > 0`.
    #[test]
    fn test_er_positive_off_nash() {
        // ρ = 0.5 on (0,0) + 0.5 on (2,2) — split between worst and best action.
        let mut e = vec![0.0; 9];
        e[OccupationMeasure::<3, 3>::flat_index(0, 0)] = 0.5;
        e[OccupationMeasure::<3, 3>::flat_index(2, 2)] = 0.5;
        let rho = OccupationMeasure::<3, 3>::new(e).unwrap();
        let d = make_dev_class();
        let p = linear_cost();
        let val = ExternalRegret::new().er(&rho, &d, &p);

        // γ(ρ) = -(0.5·R[0] + 0.5·R[2]) = -(0.5 + 1.5) = -2.0.
        // Best deviation: always-2 → γ_dev = -(0.5·R[2] + 0.5·R[2]) = -3.0.
        // ER = γ - γ_dev = -2.0 - (-3.0) = +1.0 > 0 (always-2 is profitable).
        assert!(
            val > 0.5,
            "ER should be > 0 off-Nash (profitable deviation), got {val}"
        );
        assert!(
            (val - 1.0).abs() < 1e-6,
            "ER should be exactly +1.0, got {val}"
        );
    }

    /// `test_unique_maximizer_strictly_convex` — for the off-Nash ρ above, the
    /// best deviation (always-2) is unique: top-2 gap = 1.0 > ε.
    #[test]
    fn test_unique_maximizer_strictly_convex() {
        let mut e = vec![0.0; 9];
        e[OccupationMeasure::<3, 3>::flat_index(0, 0)] = 0.5;
        e[OccupationMeasure::<3, 3>::flat_index(2, 2)] = 0.5;
        let rho = OccupationMeasure::<3, 3>::new(e).unwrap();
        let d = make_dev_class();
        let p = linear_cost();
        let er = ExternalRegret::new();

        // Per-deviation values (γ - γ_dev):
        //   identity:   γ_dev = -2.0   → 0.0
        //   always-0:   γ_dev = -1.0   → -1.0
        //   always-1:   γ_dev = -2.0   → 0.0
        //   always-2:   γ_dev = -3.0   → +1.0   ← best
        // Top-2 gap = 1.0 - 0.0 = 1.0.
        assert!(
            er.is_unique_maximizer(&rho, &d, &p, 0.1),
            "best deviation should be unique (gap = 1.0 > 0.1)"
        );

        // Sanity: with ε larger than the gap, uniqueness fails.
        assert!(
            !er.is_unique_maximizer(&rho, &d, &p, 2.0),
            "with ε=2.0 (> gap 1.0), uniqueness should fail"
        );
    }

    /// `test_linear_derivative_matches_fd` — verify `∂ER/∂ρ[m]` via central
    /// finite differences on the simplex.
    ///
    /// Because `ρ` must sum to 1, we use a **paired perturbation**: bump `m`
    /// up by `ε` and `m_partner` down by `ε`. The simplex FD then estimates
    /// `∂ER/∂ρ[m] − ∂ER/∂ρ[m_partner]`, which matches the difference of
    /// analytic derivatives.
    #[test]
    fn test_linear_derivative_matches_fd() {
        // ρ = 0.4 on (0,0), 0.3 on (1,1), 0.3 on (2,2). Off-Nash, stable κ*.
        let mut e = vec![0.0; 9];
        e[OccupationMeasure::<3, 3>::flat_index(0, 0)] = 0.4;
        e[OccupationMeasure::<3, 3>::flat_index(1, 1)] = 0.3;
        e[OccupationMeasure::<3, 3>::flat_index(2, 2)] = 0.3;
        let rho = OccupationMeasure::<3, 3>::new(e).unwrap();
        let d = make_dev_class();
        let p = linear_cost();
        let er = ExternalRegret::new();

        // Confirm κ* is unique and is always-2 (so analytic is well-defined).
        let kappa_star = er.best_deviation(&rho, &d, &p).expect("non-empty D");
        assert_eq!(kappa_star.id, 3, "κ* should be the always-2 deviation");

        let eps = 1e-4_f32;
        let partner = OccupationMeasure::<3, 3>::flat_index(0, 0);

        for m in 0..9 {
            if m == partner {
                continue;
            }
            // Central FD on the simplex: ρ ± ε·(e_m - e_partner).
            let mut e_plus = rho.entries.clone();
            e_plus[m] += eps;
            e_plus[partner] -= eps;
            let rho_plus = OccupationMeasure::<3, 3>::from_entries_trusted(e_plus);

            let mut e_minus = rho.entries.clone();
            e_minus[m] -= eps;
            e_minus[partner] += eps;
            let rho_minus = OccupationMeasure::<3, 3>::from_entries_trusted(e_minus);

            let er_plus = er.er(&rho_plus, &d, &p);
            let er_minus = er.er(&rho_minus, &d, &p);
            let fd = (er_plus - er_minus) / (2.0 * eps);

            // Analytic: (∂ER/∂ρ[m]) - (∂ER/∂ρ[partner])
            //   = [cost_follow(m) - cost_deviate(m, κ*)]
            //     - [cost_follow(partner) - cost_deviate(partner, κ*)]
            let analytic_m = er.linear_derivative(&rho, m, &d, &p);
            let analytic_partner = er.linear_derivative(&rho, partner, &d, &p);
            let analytic = analytic_m - analytic_partner;

            assert!(
                (fd - analytic).abs() < 1e-2,
                "m={m}: FD={fd:.6}, analytic={analytic:.6}, diff={:.2e}",
                (fd - analytic).abs()
            );
        }
    }

    // -------- Extra sanity tests --------

    #[test]
    fn empty_deviation_class_returns_neg_infinity() {
        struct Empty;
        impl DeviationClass<3, 3> for Empty {
            fn deviations(&self) -> &[Deviation<3, 3>] {
                &[]
            }
        }
        let rho = OccupationMeasure::<3, 3>::uniform();
        let p = linear_cost();
        let val = ExternalRegret::new().er(&rho, &Empty, &p);
        assert!(
            val.is_infinite() && val.is_sign_negative(),
            "empty D → -∞, got {val}"
        );
        assert!(
            ExternalRegret::new()
                .best_deviation(&rho, &Empty, &p)
                .is_none()
        );
    }

    #[test]
    fn strict_cce_has_negative_er() {
        // Strict CCE: every deviation strictly worse than following.
        // ρ = Dirac on (2, 2) with a deviation class that does NOT include
        // identity or always-2 (so following strictly beats every κ ∈ D).
        struct WorstDevs {
            v: Vec<Deviation<3, 3>>,
        }
        impl DeviationClass<3, 3> for WorstDevs {
            fn deviations(&self) -> &[Deviation<3, 3>] {
                &self.v
            }
        }
        let d = WorstDevs {
            v: vec![
                Deviation::<3, 3>::constant(0, 0),
                Deviation::<3, 3>::constant(1, 1),
                // Note: no identity, no always-2.
            ],
        };
        let rho = OccupationMeasure::<3, 3>::dirac(2, 2);
        let p = linear_cost();
        let val = ExternalRegret::new().er(&rho, &d, &p);
        // γ(ρ) = -3.
        // Best deviation (smallest γ_dev): always-1 → γ_dev = -2.
        // ER = -3 - (-2) = -1 < 0 (strict CCE w.r.t. this restricted D).
        assert!(
            (val - (-1.0)).abs() < 1e-6,
            "strict CCE should have ER = -1, got {val}"
        );
    }

    #[test]
    fn best_deviation_picks_argmax() {
        let mut e = vec![0.0; 9];
        e[OccupationMeasure::<3, 3>::flat_index(0, 0)] = 0.5;
        e[OccupationMeasure::<3, 3>::flat_index(2, 2)] = 0.5;
        let rho = OccupationMeasure::<3, 3>::new(e).unwrap();
        let d = make_dev_class();
        let p = linear_cost();
        let kappa = ExternalRegret::new()
            .best_deviation(&rho, &d, &p)
            .expect("non-empty D");
        // Best deviation is always-2 (id = 3).
        assert_eq!(kappa.id, 3);
    }

    /// Sanity: an invalid occupation measure is rejected by `new`. We compile
    /// this against `OccupationMeasureError` to ensure the error surface is
    /// reachable from this module too.
    #[test]
    fn invalid_occupation_measure_errors() {
        let err = OccupationMeasure::<2, 2>::new(vec![0.5, 0.5, 0.5, 0.5]).unwrap_err();
        assert!(matches!(err, OccupationMeasureError::NotNormalized { .. }));
    }

    /// Marker types compile and are zero-sized.
    #[test]
    fn marker_types_are_zst() {
        assert_eq!(core::mem::size_of::<StateSpace<4>>(), 0);
        assert_eq!(core::mem::size_of::<crate::cce::types::ActionSpace<4>>(), 0);
    }

    // ---------------------------------------------------------------------------
    // Canonical examples — required by Plan 295 Phase 1 exit criterion:
    // "ExternalRegret is correct on the 3 canonical examples
    //  (RPS, chicken, emission-abatement discrete)."
    // ---------------------------------------------------------------------------

    /// **RPS (rock-paper-scissors).** Symmetric zero-sum 2-player game.
    ///
    /// State = joint recommendation `(s_1, s_2) ∈ {R,P,S}²` (N=9), action =
    /// your play `a_1 ∈ {R,P,S}` (A=3). Honest mediator: ρ lives on the
    /// `a_1 = s_1` slice. Opponent follows: plays `s_2`. Cost = -payoff
    /// (reward matrix from player 1's perspective).
    ///
    /// Test: the mixed Nash (both uniform) has `ER = 0`. RPS has no
    /// Pareto-dominant CCE (zero-sum), so uniform is also the optimal CCE.
    #[test]
    fn canonical_rps_nash_has_zero_er() {
        // Reward matrix: R[i][j] = player 1 payoff when p1 plays i, p2 plays j.
        // R = Rock, P = Paper, S = Scissors. RPS cycle.
        // Index mapping: 0=R, 1=P, 2=S.
        const R: [[f32; 3]; 3] = [
            [0.0, -1.0, 1.0], // R vs R/P/S
            [1.0, 0.0, -1.0], // P vs R/P/S
            [-1.0, 1.0, 0.0], // S vs R/P/S
        ];

        /// State index for player 1's CCE problem: (s_1, s_2) → s_1·3 + s_2.
        /// Action: a_1 ∈ {0,1,2}. cost((s_1,s_2), a_1) = -R[a_1][s_2].
        struct Rps;
        impl PayoffTensor<9, 3> for Rps {
            fn reward_follow(&self, state: usize, action: usize) -> f32 {
                let s_2 = state % 3;
                -R[action][s_2]
            }
            fn gamma0(&self, rho: &OccupationMeasure<9, 3>) -> f32 {
                self.gamma(rho)
            }
        }

        /// Deviation class: 3 constant deviations (always R, always P, always S).
        /// For honest mediator, identity deviation = follow is implicit (any
        /// honest ρ has ρ(s, a_1) = μ(s)·δ(a_1 = s_1), so playing what's
        /// recommended = playing the dominant action under uniform opponent).
        struct RpsDevs {
            v: Vec<Deviation<9, 3>>,
        }
        impl DeviationClass<9, 3> for RpsDevs {
            fn deviations(&self) -> &[Deviation<9, 3>] {
                &self.v
            }
        }
        let d = RpsDevs {
            v: vec![
                Deviation::<9, 3>::constant(0, 0), // always R
                Deviation::<9, 3>::constant(1, 1), // always P
                Deviation::<9, 3>::constant(2, 2), // always S
            ],
        };
        let p = Rps;

        // Mixed Nash: both players uniform over {R,P,S} independently.
        // Honest mediator with μ(s_1, s_2) = (1/9) for every pair.
        // ρ((s_1, s_2), a_1) = (1/9)·δ(a_1 = s_1).
        let mut e = vec![0.0; 27];
        for s_1 in 0..3 {
            for s_2 in 0..3 {
                let state = s_1 * 3 + s_2;
                e[state * 3 + s_1] = 1.0 / 9.0;
            }
        }
        let rho = OccupationMeasure::<9, 3>::new(e).unwrap();

        // γ(ρ) = -(1/9)·Σ_{s_1,s_2} R[s_1][s_2] = -(1/9)·0 = 0 (zero-sum symmetric).
        assert!(p.gamma(&rho).abs() < 1e-6, "RPS uniform Nash γ = 0");

        // Every constant deviation: γ_dev(ρ, κ_c) = -(1/9)·Σ_{s_1,s_2} R[c][s_2]
        //   = -(1/3)·Σ_{s_2} R[c][s_2] = -(1/3)·0 = 0 for any c.
        // ER = max_c (0 - 0) = 0. Uniform Nash is a marginal CCE for RPS.
        let val = ExternalRegret::new().er(&rho, &d, &p);
        assert!(
            val.abs() < 1e-6,
            "RPS mixed Nash should have ER = 0 (marginal CCE), got {val}"
        );
    }

    /// **Chicken (Hawk-Dove).** 2-player general-sum game with a
    /// Pareto-dominant CCE.
    ///
    /// State = joint recommendation `(s_1, s_2) ∈ {S,T}²` (N=4), action =
    /// your play `a_1 ∈ {S,T}` (A=2). Cost = -payoff. Reward matrix
    /// (player 1 row, player 2 col):
    /// ```text
    ///         S     T
    ///    S   (3,3) (1,4)
    ///    T   (4,1) (0,0)
    /// ```
    ///
    /// Test 1: mixed Nash (each swerves with prob 0.5) has `ER = 0`.
    /// Test 2: the strict CCE `0.5·(S,T) + 0.5·(T,S)` has `ER < 0` (every
    /// constant deviation strictly worse than following).
    #[test]
    fn canonical_chicken_nash_zero_and_cce_strict() {
        // R[a_1][s_2] = player 1 payoff. Index: 0=S, 1=T.
        const R: [[f32; 2]; 2] = [[3.0, 1.0], [4.0, 0.0]];

        struct Chicken;
        impl PayoffTensor<4, 2> for Chicken {
            fn reward_follow(&self, state: usize, action: usize) -> f32 {
                let s_2 = state % 2;
                -R[action][s_2]
            }
            fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
                self.gamma(rho)
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
        let d = ChickenDevs {
            v: vec![
                Deviation::<4, 2>::constant(0, 0), // always S
                Deviation::<4, 2>::constant(1, 1), // always T
            ],
        };
        let p = Chicken;
        let er = ExternalRegret::new();

        // --- Test 1: mixed Nash (each swerves with prob 0.5 independently) ---
        // Honest mediator with μ(s_1, s_2) = (1/4) for every pair.
        // ρ((s_1,s_2), a_1) = (1/4)·δ(a_1 = s_1).
        let mut e_nash = vec![0.0; 8];
        for s_1 in 0..2 {
            for s_2 in 0..2 {
                let state = s_1 * 2 + s_2;
                e_nash[state * 2 + s_1] = 0.25;
            }
        }
        let rho_nash = OccupationMeasure::<4, 2>::new(e_nash).unwrap();
        let er_nash = er.er(&rho_nash, &d, &p);
        assert!(
            er_nash.abs() < 1e-6,
            "chicken mixed Nash should have ER = 0, got {er_nash}"
        );

        // --- Test 2: strict CCE = 0.5·(S,T) + 0.5·(T,S) ---
        // Honest mediator: ρ((S,T), S) = 0.5 (player 1 swerves), ρ((T,S), T) = 0.5.
        // State encoding: (s_1,s_2): (S,T) = (0,1) → state 1; (T,S) = (1,0) → state 2.
        let mut e_cce = vec![0.0; 8];
        // state 1 = (s_1=0, s_2=1), recommend a_1=0 (S):
        e_cce[2] = 0.5;
        // state 2 = (s_1=1, s_2=0), recommend a_1=1 (T):
        e_cce[5] = 0.5;
        let rho_cce = OccupationMeasure::<4, 2>::new(e_cce).unwrap();
        // γ(ρ_CCE) = -(0.5·R[S][T] + 0.5·R[T][S]) = -(0.5·1 + 0.5·4) = -2.5.
        assert!((p.gamma(&rho_cce) - (-2.5)).abs() < 1e-6);
        // γ_dev(always-S) = -(0.5·R[S][T] + 0.5·R[S][S]) = -(0.5·1 + 0.5·3) = -2.0.
        // γ_dev(always-T) = -(0.5·R[T][T] + 0.5·R[T][S]) = -(0.5·0 + 0.5·4) = -2.0.
        // ER = max(-2.5 - (-2.0), -2.5 - (-2.0)) = max(-0.5, -0.5) = -0.5 < 0.
        // Strict CCE: every constant deviation strictly worse than following.
        let er_cce = er.er(&rho_cce, &d, &p);
        assert!(
            er_cce < -1e-6,
            "chicken strict CCE should have ER < 0, got {er_cce}"
        );
        assert!(
            (er_cce - (-0.5)).abs() < 1e-6,
            "expected ER = -0.5, got {er_cce}"
        );

        // Welfare comparison (for future G1 benchmark): CCE welfare = 5.0
        // (both players' rewards sum to 5 in each of (S,T) and (T,S)).
        // Mixed Nash welfare = 2·2 = 4. CCE strictly Pareto-dominates.
        let welfare_cce = -p.gamma(&rho_cce) * 2.0; // symmetric players.
        let welfare_nash = -p.gamma(&rho_nash) * 2.0;
        assert!(
            welfare_cce > welfare_nash + 0.5,
            "chicken CCE welfare {welfare_cce} should beat Nash {welfare_nash}"
        );
    }

    /// **Emission-abatement discrete (paper §8.2).** Simplified 1-firm
    /// 2-action abatement game.
    ///
    /// State = market price signal `s ∈ {Low, High}` (N=2). Action =
    /// abatement level `a ∈ {Abate, Pollute}` (A=2). Cost = emission +
    /// abatement cost (firm minimizes).
    ///
    /// Test: at the optimal pure-strategy profile (always Abate), the firm
    /// has no incentive to deviate to Pollute — `ER ≤ 0` (CCE satisfied).
    #[test]
    fn canonical_emission_abatement_cce_satisfied() {
        // cost(s, a):
        //   Low price:  Abate = 1.0, Pollute = 3.0 (low price → cheap pollution but emission cost dominates)
        //   High price: Abate = 2.0, Pollute = 5.0 (high price → expensive pollution)
        // Firm minimizes cost. Abate is dominant in both states.
        struct Emission;
        impl PayoffTensor<2, 2> for Emission {
            fn reward_follow(&self, state: usize, action: usize) -> f32 {
                const C: [[f32; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
                C[state][action]
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
        let d = EmitDevs {
            v: vec![
                Deviation::<2, 2>::constant(0, 0), // always Abate
                Deviation::<2, 2>::constant(1, 1), // always Pollute
            ],
        };
        let p = Emission;
        let er = ExternalRegret::new();

        // Optimal CCE: mediator always recommends Abate, both states equally likely.
        // Honest mediator: ρ((s, Abate)) = 0.5 for s ∈ {Low, High}.
        let rho = OccupationMeasure::<2, 2>::new(vec![0.5, 0.0, 0.5, 0.0]).unwrap();
        // γ(ρ) = 0.5·1 + 0.5·2 = 1.5 (expected cost when always Abate).
        assert!((p.gamma(&rho) - 1.5).abs() < 1e-6);
        // Deviation to always-Pollute: γ_dev = 0.5·3 + 0.5·5 = 4.0 > γ(ρ).
        // ER = max(1.5 - 1.5, 1.5 - 4.0) = max(0.0, -2.5) = 0.0 (marginal CCE).
        let val = er.er(&rho, &d, &p);
        assert!(
            val.abs() < 1e-6,
            "emission CCE should have ER = 0 (no profitable deviation), got {val}"
        );

        // Off-CCE: mediator recommends Pollute half the time.
        let rho_bad = OccupationMeasure::<2, 2>::new(vec![0.0, 0.5, 0.5, 0.0]).unwrap();
        // γ(ρ_bad) = 0.5·3 + 0.5·2 = 2.5. γ_dev(always-Abate) = 0.5·1 + 0.5·2 = 1.5.
        // ER = max(2.5 - 1.5, 2.5 - 5.0) = max(1.0, -2.5) = 1.0 > 0 (profitable deviation).
        let val_bad = er.er(&rho_bad, &d, &p);
        assert!(
            val_bad > 0.5,
            "off-CCE ρ should have ER > 0 (always-Abate profitable), got {val_bad}"
        );
    }

    /// Plan 300 T4.3b: `linear_derivative_heterogeneous` matches the
    /// homogeneous `linear_derivative` when the game is a single-player wrapper
    /// around the same `(P, D)`. Sanity check that the per-player subgradient
    /// aggregation reduces correctly to the homogeneous case at `P = 1`.
    #[test]
    fn linear_derivative_heterogeneous_matches_homogeneous_at_one_player() {
        use crate::cce::heterogeneous::PerPlayerGame;

        struct Emission;
        impl PayoffTensor<2, 2> for Emission {
            fn reward_follow(&self, s: usize, a: usize) -> f32 {
                const C: [[f32; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
                C[s][a]
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
        let d = EmitDevs {
            v: vec![
                Deviation::<2, 2>::constant(0, 0),
                Deviation::<2, 2>::constant(1, 1),
            ],
        };
        let p = Emission;
        let rho = OccupationMeasure::<2, 2>::new(vec![0.4, 0.1, 0.3, 0.2]).unwrap();

        let game = PerPlayerGame::new(vec![(&p, &d)]);
        let er = ExternalRegret::new();

        // For each flat index m, the heterogeneous derivative (with P = 1,
        // single player) must equal the homogeneous derivative.
        for m in 0..(2 * 2) {
            let homo = er.linear_derivative(&rho, m, &d, &p);
            let hetero = er.linear_derivative_heterogeneous(&rho, m, &game);
            assert!(
                (homo - hetero).abs() < 1e-6,
                "m = {m}: homo = {homo:.6}, hetero = {hetero:.6} (must match at P = 1)"
            );
        }
    }
}
