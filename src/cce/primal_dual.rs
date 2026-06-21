//! Bregman primal-dual iterator for LP-CCE (Plan 295 Phase 2).
//!
//! Implements Algorithm 1 from Campi, Cannerozzi, Tzouanas 2026 (arxiv
//! 2606.20062): a primal-dual scheme that converges to the optimal CCE at
//! rate `O(N⁻¹ᐟ²)` on the averaged iterates.
//!
//! ## Algorithm (Euclidean Bregman potential)
//!
//! ```text
//! ρ⁰ = uniform, λ⁰ = 0
//! for n = 1, 2, ...:
//!     # Primal: projected gradient step on L(ρ, λⁿ⁻¹) = γ₀(ρ) + λⁿ⁻¹ · ER(ρ)
//!     grad[m]  = gamma0_coeff(m) + λⁿ⁻¹ · linear_derivative(ρⁿ⁻¹, m)
//!     ρ_temp   = ρⁿ⁻¹ - η_n · grad
//!     ρⁿ       = project_onto_simplex(ρ_temp)
//!
//!     # Dual: sub-gradient ascent on the CCE constraint ER(ρ) ≤ 0
//!     λⁿ       = max(0, λⁿ⁻¹ + (1/√n) · ER(ρⁿ))
//!
//!     # Averaged iterate (this is what converges)
//!     ρ̄ⁿ       = ((n-1)/n) · ρ̄ⁿ⁻¹ + (1/n) · ρⁿ
//! ```
//!
//! ## Convergence
//!
//! The averaged iterate `ρ̄ᴺ` satisfies `|γ₀(ρ̄ᴺ) − γ₀(ρ⋆)| = O(N⁻¹ᐟ²)` and
//! `ER(ρ̄ᴺ) ≤ O(N⁻¹ᐟ²)` (Theorem 6.1). Phase 2 test G2 verifies this on the
//! emission-abatement example.

use crate::cce::bregman::Euclidean;
use crate::cce::external_regret::ExternalRegret;
use crate::cce::types::{
    Deviation, DeviationClass, HeterogeneousPayoff, OccupationMeasure, PayoffTensor,
};

/// Per-step diagnostic reported by [`CcePrimalDual::step`].
#[derive(Debug, Clone)]
pub struct StepReport {
    /// Iteration index (1-based).
    pub n: usize,
    /// Primal variable `ρⁿ` after projection.
    pub rho: Vec<f32>,
    /// Dual variable `λⁿ` after the update.
    pub lambda: f32,
    /// External regret at the new `ρⁿ` (positive = constraint violated).
    pub er: f32,
    /// Moderator objective `γ₀(ρⁿ)`.
    pub gamma0: f32,
    /// Step size used at this iteration.
    pub eta: f32,
}

/// Bregman primal-dual iterator for the LP-CCE problem.
///
/// Generic over the Bregman potential; Phase 2 ships the Euclidean potential
/// (projected gradient descent). The KL potential (entropic mirror descent)
/// is a Phase 3 follow-up.
pub struct CcePrimalDual {
    /// Dual variable `λ ≥ 0`.
    pub lambda: f32,
    /// Current primal iterate `ρⁿ`.
    pub rho: Vec<f32>,
    /// Averaged primal iterate `ρ̄ⁿ`.
    pub rho_avg: Vec<f32>,
    /// Iteration counter (1-based after the first `step`).
    pub n_iter: usize,
    /// Primal step size `η`.
    pub eta: f32,
    /// Euclidean Bregman potential (Phase 2 only; KL reserved for Phase 3).
    _potential: Euclidean,
}

impl CcePrimalDual {
    /// Initialize the iterator at the uniform distribution with `λ⁰ = 0`.
    pub fn new<const N: usize, const A: usize>() -> Self {
        let na = N * A;
        let p = 1.0 / na as f32;
        let rho = vec![p; na];
        let rho_avg = rho.clone();
        Self {
            lambda: 0.0,
            rho,
            rho_avg,
            n_iter: 0,
            eta: 0.1,
            _potential: Euclidean,
        }
    }

    /// Override the primal step size `η`.
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.eta = eta;
        self
    }

    /// Override the initial primal iterate `ρ⁰`.
    pub fn with_initial_rho(mut self, rho: Vec<f32>) -> Self {
        debug_assert!(!rho.is_empty(), "initial rho must be non-empty");
        self.rho = rho;
        self.rho_avg = self.rho.clone();
        self
    }

    /// Override the initial dual variable `λ⁰`.
    pub fn with_initial_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.max(0.0);
        self
    }

    /// One primal-dual iteration. Updates `self.rho`, `self.rho_avg`,
    /// `self.lambda`, `self.n_iter` in place and returns a diagnostic report.
    pub fn step<const N: usize, const A: usize, D: DeviationClass<N, A>, P: PayoffTensor<N, A>>(
        &mut self,
        d: &D,
        p: &P,
    ) -> StepReport {
        let er_eval = ExternalRegret::new();
        let na = N * A;
        self.n_iter += 1;
        let n = self.n_iter;

        // --- Primal: projected gradient step on γ₀(ρ) + λ · ER(ρ) ---
        // Build the current ρ view for ExternalRegret calls.
        let rho_view = OccupationMeasure::<N, A>::from_entries_trusted(self.rho.clone());

        // Compute gradient: grad[m] = gamma0_coeff(m) + λ · linear_derivative(m).
        let mut grad = vec![0.0_f32; na];
        for s in 0..N {
            for a in 0..A {
                let m = s * A + a;
                let g0 = p.gamma0_coeff(s, a);
                let er_deriv = er_eval.linear_derivative(&rho_view, m, d, p);
                grad[m] = g0 + self.lambda * er_deriv;
            }
        }

        // Gradient step: ρ_temp = ρ - η · grad.
        let eta = self.eta;
        let rho_temp: Vec<f32> = self
            .rho
            .iter()
            .zip(grad.iter())
            .map(|(&r, &g)| r - eta * g)
            .collect();

        // Project onto the simplex.
        let rho_new = project_onto_simplex(&rho_temp);

        // --- Dual: λⁿ = max(0, λⁿ⁻¹ + (1/√n) · ER(ρⁿ)) ---
        let rho_new_view = OccupationMeasure::<N, A>::from_entries_trusted(rho_new.clone());
        let er_new = er_eval.er(&rho_new_view, d, p);
        let dual_step = 1.0 / (n as f32).sqrt();
        self.lambda = (self.lambda + dual_step * er_new).max(0.0);

        // --- Averaged iterate: ρ̄ⁿ = ((n-1)/n) · ρ̄ⁿ⁻¹ + (1/n) · ρⁿ ---
        let w_old = (n - 1) as f32 / n as f32;
        let w_new = 1.0 / n as f32;
        for (rho_avg_i, &rho_new_i) in self.rho_avg.iter_mut().zip(rho_new.iter()) {
            *rho_avg_i = w_old * *rho_avg_i + w_new * rho_new_i;
        }

        // Commit.
        let gamma0_new = p.gamma0(&rho_new_view);
        self.rho = rho_new;

        StepReport {
            n,
            rho: self.rho.clone(),
            lambda: self.lambda,
            er: er_new,
            gamma0: gamma0_new,
            eta,
        }
    }

    /// Run `n_steps` iterations and return the convergence report with
    /// averaged iterate + per-step history.
    pub fn run<const N: usize, const A: usize, D: DeviationClass<N, A>, P: PayoffTensor<N, A>>(
        mut self,
        d: &D,
        p: &P,
        n_steps: usize,
    ) -> ConvergenceReportRaw<N, A> {
        let mut history = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            history.push(self.step::<N, A, D, P>(d, p));
        }

        let rho_avg = OccupationMeasure::<N, A>::from_entries_trusted(self.rho_avg.clone());
        let gamma0_avg = p.gamma0(&rho_avg);
        let er_avg = ExternalRegret::new().er(&rho_avg, d, p);

        ConvergenceReportRaw {
            rho_avg,
            history,
            gamma0_avg,
            er_avg,
        }
    }

    /// One primal-dual iteration on a heterogeneous player population
    /// (Plan 300 T4.3b).
    ///
    /// Identical algorithm to [`step`](CcePrimalDual::step) but with the
    /// per-player subgradient oracle. Caches the best deviation `κ_i*(ρ)` for
    /// each player once per step (O(P · |D| · N · A) work), then aggregates
    /// `grad[m] = gamma0_coeff(m) + λ · (1/P) Σ_i [cost_i(s,a) − reward_deviate(i, s, κ_i*)]`
    /// for each (s, a) index. Heterogeneity-agnostic averaging follows the
    /// homogeneous recipe unchanged (Theorem 6.1 applies to the convex
    /// aggregate `ER_heterogeneous`; doc 62 §2).
    pub fn step_heterogeneous<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        &mut self,
        game: &H,
    ) -> StepReport {
        let na = N * A;
        self.n_iter += 1;
        let n = self.n_iter;

        // Build the current ρ view.
        let rho_view = OccupationMeasure::<N, A>::from_entries_trusted(self.rho.clone());

        // --- Subgradient oracle: pick each player's best deviation at ρⁿ⁻¹ ---
        // The best deviation κ_i*(ρ) maximizes (γ_i(ρ) − γ_dev_i(ρ, κ)) over
        // κ ∈ D_i. Cache the pick so we don't recompute it per (s, a) index.
        let n_players = game.n_players();
        let mut best_kappas: Vec<Option<&Deviation<N, A>>> = Vec::with_capacity(n_players);
        for i in 0..n_players {
            let gamma_i = game.gamma_player(i, &rho_view);
            let mut best_val = f32::NEG_INFINITY;
            let mut best_kappa: Option<&Deviation<N, A>> = None;
            for kappa in game.deviations_for_player(i) {
                let val = gamma_i - game.gamma_dev_player(i, &rho_view, kappa);
                if val > best_val {
                    best_val = val;
                    best_kappa = Some(kappa);
                }
            }
            best_kappas.push(best_kappa);
        }

        // --- Primal: projected gradient step on γ₀(ρ) + λ · ER_hetero(ρ) ---
        let inv_p = 1.0 / n_players.max(1) as f32;
        let mut grad = vec![0.0_f32; na];
        for s in 0..N {
            for a in 0..A {
                let m = s * A + a;
                let g0 = game.gamma0_coeff(s, a);
                let mut er_deriv = 0.0_f32;
                for (i, kappa_opt) in best_kappas.iter().copied().enumerate() {
                    if let Some(kappa) = kappa_opt {
                        er_deriv += game.reward_follow(i, s, a) - game.reward_deviate(i, s, kappa);
                    }
                    // Empty deviation class: player contributes 0.
                }
                grad[m] = g0 + self.lambda * er_deriv * inv_p;
            }
        }

        // Gradient step: ρ_temp = ρ - η · grad.
        let eta = self.eta;
        let rho_temp: Vec<f32> = self
            .rho
            .iter()
            .zip(grad.iter())
            .map(|(&r, &g)| r - eta * g)
            .collect();

        // Project onto the simplex.
        let rho_new = project_onto_simplex(&rho_temp);

        // --- Dual: λⁿ = max(0, λⁿ⁻¹ + (1/√n) · ER_hetero(ρⁿ)) ---
        let rho_new_view = OccupationMeasure::<N, A>::from_entries_trusted(rho_new.clone());
        let er_new = ExternalRegret::new().er_heterogeneous(&rho_new_view, game);
        let dual_step = 1.0 / (n as f32).sqrt();
        self.lambda = (self.lambda + dual_step * er_new).max(0.0);

        // --- Averaged iterate: ρ̄ⁿ = ((n-1)/n) · ρ̄ⁿ⁻¹ + (1/n) · ρⁿ ---
        let w_old = (n - 1) as f32 / n as f32;
        let w_new = 1.0 / n as f32;
        for (rho_avg_i, &rho_new_i) in self.rho_avg.iter_mut().zip(rho_new.iter()) {
            *rho_avg_i = w_old * *rho_avg_i + w_new * rho_new_i;
        }

        // Commit.
        let gamma0_new = game.gamma0(&rho_new_view);
        self.rho = rho_new;

        StepReport {
            n,
            rho: self.rho.clone(),
            lambda: self.lambda,
            er: er_new,
            gamma0: gamma0_new,
            eta,
        }
    }

    /// Run `n_steps` heterogeneous iterations and return the convergence
    /// report (Plan 300 T4.3b).
    ///
    /// `er_avg` carries `ER_heterogeneous(ρ̄ᴺ)` (the averaged-iterate regret).
    /// Convergence bound: `|γ₀(ρ̄ᴺ) − γ₀(ρ⋆)| ≤ O(N⁻¹ᐟ²)` and
    /// `ER_heterogeneous(ρ̄ᴺ) ≤ O(N⁻¹ᐟ²)` — transfers from the homogeneous
    /// case by convexity of the aggregate.
    pub fn run_heterogeneous<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        mut self,
        game: &H,
        n_steps: usize,
    ) -> ConvergenceReportRaw<N, A> {
        let mut history = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            history.push(self.step_heterogeneous::<N, A, H>(game));
        }

        let rho_avg = OccupationMeasure::<N, A>::from_entries_trusted(self.rho_avg.clone());
        let gamma0_avg = game.gamma0(&rho_avg);
        let er_avg = ExternalRegret::new().er_heterogeneous(&rho_avg, game);

        ConvergenceReportRaw {
            rho_avg,
            history,
            gamma0_avg,
            er_avg,
        }
    }
}

/// Concretely-typed convergence report (carries the `N, A` const generics
/// that the trait-level `ConvergenceReport` cannot).
#[derive(Debug, Clone)]
pub struct ConvergenceReportRaw<const N: usize, const A: usize> {
    /// Averaged iterate `ρ̄ᴺ`.
    pub rho_avg: OccupationMeasure<N, A>,
    /// Per-iteration diagnostics.
    pub history: Vec<StepReport>,
    /// `γ₀(ρ̄ᴺ)`.
    pub gamma0_avg: f32,
    /// `ER(ρ̄ᴺ)`.
    pub er_avg: f32,
}

/// Euclidean projection onto the probability simplex `{x ≥ 0, Σx = 1}`.
///
/// Implements the sort-based algorithm of Wang & Carreira-Perpiñán (2013):
/// `O(d log d)` for a `d`-dimensional vector. Returns `x_i = max(0, v_i - θ)`
/// where `θ = (Σ_{j≤k} v_j - 1) / k` for the largest `k` with `v_(k) > θ`
/// (`v_(k)` is the `k`-th largest entry).
fn project_onto_simplex(v: &[f32]) -> Vec<f32> {
    let d = v.len();
    if d == 0 {
        return Vec::new();
    }
    // Sort descending.
    let mut sorted: Vec<f32> = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));

    // Find θ = the last (Σ_{j≤k} v_j - 1) / k for which v_(k) > θ_k.
    let mut cumsum = 0.0_f32;
    let mut theta = 0.0_f32;
    for (i, &val) in sorted.iter().enumerate() {
        cumsum += val;
        let k = (i + 1) as f32;
        let theta_k = (cumsum - 1.0) / k;
        if val > theta_k {
            theta = theta_k;
        }
    }

    // Project: x_i = max(0, v_i - θ).
    v.iter().map(|&x| (x - theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cce::types::Deviation;

    #[test]
    fn project_onto_simplex_preserves_sum() {
        let v = vec![0.3, -0.5, 0.8, 0.1, 0.7];
        let x = project_onto_simplex(&v);
        let sum: f32 = x.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        for &xi in &x {
            assert!(xi >= -1e-7, "negative entry {xi}");
        }
    }

    #[test]
    fn project_onto_simplex_uniform_input() {
        let v = vec![0.25; 4];
        let x = project_onto_simplex(&v);
        for &xi in &x {
            assert!((xi - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn project_onto_simplex_already_feasible() {
        let v = vec![0.5, 0.3, 0.2];
        let x = project_onto_simplex(&v);
        for (xi, &vi) in x.iter().zip(v.iter()) {
            assert!((xi - vi).abs() < 1e-5);
        }
    }

    #[test]
    fn project_onto_simplex_negative_entries_clamped() {
        let v = vec![-1.0, -1.0, 5.0];
        let x = project_onto_simplex(&v);
        // The single large entry should absorb most mass.
        assert!(x[2] > 0.9, "x[2] = {}", x[2]);
        assert!(x[0] < 0.1 && x[1] < 0.1);
    }

    /// Smoke test: the primal-dual iterator converges toward the LP optimum
    /// on the emission-abatement problem.
    #[test]
    fn primal_dual_converges_to_emission_optimum() {
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
                Deviation::<2, 2>::constant(0, 0),
                Deviation::<2, 2>::constant(1, 1),
            ],
        };
        let p = Emission;

        // LP optimum: γ₀ = 1.0 (all mass on (Low, Abate)).
        let lp_opt = crate::cce::lp::CceLp::new()
            .solve(&d, &p)
            .expect("LP feasible");
        let lp_gamma0 = p.gamma0(&lp_opt);

        // Run primal-dual for 5000 steps.
        let runner = CcePrimalDual::new::<2, 2>().with_eta(0.05);
        let report = runner.run(&d, &p, 5000);

        // Averaged iterate should be close to LP optimum.
        assert!(
            (report.gamma0_avg - lp_gamma0).abs() < 0.1,
            "γ₀(ρ̄) = {} should be close to LP γ₀ = {}",
            report.gamma0_avg,
            lp_gamma0
        );

        // ER(ρ̄) should be near zero (CCE satisfied within tolerance).
        assert!(
            report.er_avg <= 0.1,
            "ER(ρ̄) = {} should be ≤ 0.1",
            report.er_avg
        );
    }

    /// Plan 300 T4.3b smoke test: the heterogeneous primal-dual iterator
    /// converges toward the same optimum as `CceLp::solve_heterogeneous` on
    /// a 4-player perturbed emission-abatement game.
    #[test]
    fn primal_dual_heterogeneous_converges() {
        use crate::cce::heterogeneous::PerPlayerGame;

        struct Player {
            c: [f32; 4],
        }
        impl PayoffTensor<2, 2> for Player {
            fn reward_follow(&self, s: usize, a: usize) -> f32 {
                self.c[s * 2 + a]
            }
            fn gamma0(&self, rho: &OccupationMeasure<2, 2>) -> f32 {
                self.gamma(rho)
            }
        }
        struct Devs {
            v: Vec<Deviation<2, 2>>,
        }
        impl DeviationClass<2, 2> for Devs {
            fn deviations(&self) -> &[Deviation<2, 2>] {
                &self.v
            }
        }
        let d = Devs {
            v: vec![
                Deviation::<2, 2>::constant(0, 0),
                Deviation::<2, 2>::constant(1, 1),
            ],
        };
        // 4 players, each a small perturbation around the same base table.
        let players: Vec<Player> = (0..4)
            .map(|i| Player {
                c: [1.0 + i as f32 * 0.01, 3.0, 2.0, 5.0],
            })
            .collect();
        let player_refs: Vec<(&Player, &Devs)> = players.iter().map(|p| (p, &d)).collect();
        let game = PerPlayerGame::new(player_refs);

        // Reference: exact LP solve.
        let rho_lp = crate::cce::lp::CceLp::new()
            .solve_heterogeneous(&game)
            .expect("LP feasible");
        let gamma0_lp = game.gamma0(&rho_lp);

        // Run heterogeneous primal-dual for 10⁴ steps.
        let runner = CcePrimalDual::new::<2, 2>().with_eta(0.05);
        let report = runner.run_heterogeneous(&game, 10_000);

        let gap = (report.gamma0_avg - gamma0_lp).abs();
        assert!(
            gap < 0.1,
            "heterogeneous primal-dual gap {gap:.4} should be < 0.1 (γ₀(ρ̄) = {}, γ₀(ρ⋆_LP) = {})",
            report.gamma0_avg,
            gamma0_lp
        );
        assert!(
            report.er_avg <= 0.1,
            "heterogeneous ER(ρ̄) = {} should be ≤ 0.1",
            report.er_avg
        );
    }
}
