//! LP-CCE solver — active-set LP over finite occupation measures (Plan 295 Phase 2).
//!
//! Solves the linear program:
//!
//! ```text
//! minimize   γ₀(ρ) = Σ_{s,a} ρ(s,a) · gamma0_coeff(s, a)
//! subject to Σ_{s,a} ρ(s,a) = 1                           (probability simplex)
//!            γ(ρ) ≤ γ_dev(ρ, κ)   for every κ ∈ D         (CCE constraints)
//!            ρ(s, a) ≥ 0           for every (s, a)        (non-negativity)
//! ```
//!
//! ## Method
//!
//! **Basic-feasible-solution (BFS) enumeration** — for each subset of `m`
//! variables (where `m` = number of equality constraints), solve the `m × m`
//! linear system, check non-negativity of the full solution, and keep the
//! best objective. This is exact for small problems (`N·A + |D| ≤ ~25`) and
//! avoids a from-scratch simplex implementation.
//!
//! For Phase 2's emission-abatement test (`N = 4, A = 4, |D| = 4`), this is
//! `C(20, 5) = 15504` candidates — runs in milliseconds.
//!
//! ## Standard form
//!
//! Slack variables convert the `|D|` CCE inequalities to equalities. Total
//! variables: `N·A + |D|`. Total constraints: `1 + |D|` (one for the simplex
//! sum, one per CCE constraint after slack conversion).

use crate::cce::external_regret::ExternalRegret;
use crate::cce::types::{
    DeviationClass, HeterogeneousPayoff, OccupationMeasure, PayoffTensor, TransitionKernel,
};

/// LP solver error.
#[derive(Debug)]
pub enum CceLpError {
    /// No ρ satisfies the CCE constraints (e.g., deviation class is too rich
    /// and excludes every distribution).
    Infeasible,
    /// The moderator objective is unbounded below over the feasible set.
    /// Should not happen for valid CCE LPs (feasible set is compact).
    Unbounded,
    /// Numerical failure (singular constraint submatrix).
    NumericalError(&'static str),
}

/// LP-CCE solver. Stateless — `solve` takes the deviation class and payoff
/// tensor by reference.
#[derive(Debug, Default)]
pub struct CceLp;

impl CceLp {
    pub fn new() -> Self {
        Self
    }

    /// Solve the LP-CCE problem: find `ρ⋆ = argmin_{ρ ∈ CCE} γ₀(ρ)`.
    ///
    /// Returns the optimal occupation measure, or an error if the LP is
    /// infeasible / unbounded / numerically degenerate.
    pub fn solve<const N: usize, const A: usize, D: DeviationClass<N, A>, P: PayoffTensor<N, A>>(
        &self,
        d: &D,
        p: &P,
    ) -> Result<OccupationMeasure<N, A>, CceLpError> {
        let na = N * A;
        let devs = d.deviations();
        let nd = devs.len();

        // Total variables: ρ[0..na] + s[0..nd] (slacks for CCE constraints).
        let n_vars = na + nd;
        // Total equality constraints: 1 (sum) + nd (CCE with slacks).
        let n_cons = 1 + nd;

        if n_cons == 0 || n_cons > n_vars {
            return Err(CceLpError::Infeasible);
        }

        // Build constraint matrix A (n_cons × n_vars) and RHS b (n_cons).
        let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
        let mut rhs = vec![0.0_f64; n_cons];

        // Row 0: Σ ρ = 1.
        mat[0][..na].fill(1.0);
        rhs[0] = 1.0;

        // Rows 1..=nd: for each κ, g_κ · ρ + s_κ = 0
        //   where g_κ(s,a) = cost(s,a) − reward_deviate(s, κ).
        for (k, kappa) in devs.iter().enumerate() {
            for s in 0..N {
                for a in 0..A {
                    let j = s * A + a;
                    let g = p.reward_follow(s, a) as f64 - p.reward_deviate(s, kappa) as f64;
                    mat[1 + k][j] = g;
                }
            }
            mat[1 + k][na + k] = 1.0; // slack column
            rhs[1 + k] = 0.0;
        }

        // Objective coefficients: γ₀(ρ) = Σ ρ(s,a) · gamma0_coeff(s,a).
        // Slack variables have zero objective.
        let mut obj = vec![0.0_f64; n_vars];
        for s in 0..N {
            for a in 0..A {
                obj[s * A + a] = p.gamma0_coeff(s, a) as f64;
            }
        }

        // Auto-select solver: BFS enumeration for small LPs (exact, fast),
        // two-phase primal simplex for large LPs (Plan 572).
        let best_rho_entries = solve_lp_auto(&mat, &rhs, &obj, n_vars, na);
        match best_rho_entries {
            Some(rho_entries) => {
                // Final normalization to exactly sum = 1 (within f32 tolerance).
                let sum: f32 = rho_entries.iter().map(|&v| v as f32).sum();
                let inv = if sum > 1e-9 { 1.0 / sum } else { 1.0 };
                let entries_f32: Vec<f32> = rho_entries.iter().map(|&v| (v as f32) * inv).collect();
                Ok(OccupationMeasure::from_entries_trusted(entries_f32))
            }
            None => Err(CceLpError::Infeasible),
        }
    }

    /// Verify that `ρ` is a CCE: `ER(ρ) ≤ ε`.
    ///
    /// Uses [`ExternalRegret::er`]. Recall the cost convention:
    /// `ER(ρ) = max_κ (γ(ρ) − γ_dev(ρ, κ))`, and `ER ≤ 0` is the CCE condition.
    /// With `ε > 0`, we accept small violations (Slater tolerance).
    pub fn is_cce<
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
        ExternalRegret::new().er(rho, d, p) <= eps
    }

    /// Solve the transition-kernel-constrained CCE LP (Plan 569).
    ///
    /// Extends [`solve`](Self::solve) with `N-1` balance-equation rows
    /// enforcing stationary MDP consistency:
    ///
    /// ```text
    /// ν(s') = Σ_{s,a} ρ(s,a) · P(s'|s,a)     for s' = 0..N-1
    /// ```
    ///
    /// This closes the free-state-distribution artifact (Issue 574 T4 PASS):
    /// the CCE can no longer freely choose the state distribution to exploit
    /// favorable `(state, action)` pairs. On a 2-state MDP with
    /// action-dependent transitions, the constrained CCE recovers the exact
    /// true MDP optimum (residual gap = 0.0).
    ///
    /// **Scope:** closes the artifact for games with action-dependent
    /// transitions. For games with state-independent transitions, the balance
    /// equation reduces to a marginal constraint (ν = stationary distribution
    /// of P) which Issue 573 T4a proved is insufficient.
    ///
    /// **Complexity:** `C(N·A + |D|, 1 + |D| + N-1) · m³` where `m = 1 + |D| +
    /// N-1`. Still exact for `N·A + |D| ≤ ~25`.
    pub fn solve_with_dynamics<
        const N: usize,
        const A: usize,
        D: DeviationClass<N, A>,
        P: PayoffTensor<N, A>,
        K: TransitionKernel<N, A>,
    >(
        &self,
        d: &D,
        p: &P,
        kernel: &K,
    ) -> Result<OccupationMeasure<N, A>, CceLpError> {
        let na = N * A;
        let devs = d.deviations();
        let nd = devs.len();
        // N-1 independent balance rows (the Nth is implied by normalization).
        let n_balance = N.saturating_sub(1);

        // Total variables: ρ[0..na] + s[0..nd] (slacks for CCE constraints).
        let n_vars = na + nd;
        // Total equality constraints: 1 (sum) + nd (CCE) + n_balance.
        let n_cons = 1 + nd + n_balance;

        if n_cons == 0 || n_cons > n_vars {
            return Err(CceLpError::Infeasible);
        }

        // Build constraint matrix A (n_cons × n_vars) and RHS b (n_cons).
        let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
        let mut rhs = vec![0.0_f64; n_cons];

        // Row 0: Σ ρ = 1.
        mat[0][..na].fill(1.0);
        rhs[0] = 1.0;

        // Rows 1..=nd: for each κ, g_κ · ρ + s_κ = 0.
        for (k, kappa) in devs.iter().enumerate() {
            for s in 0..N {
                for a in 0..A {
                    let j = s * A + a;
                    let g = p.reward_follow(s, a) as f64 - p.reward_deviate(s, kappa) as f64;
                    mat[1 + k][j] = g;
                }
            }
            mat[1 + k][na + k] = 1.0;
            rhs[1 + k] = 0.0;
        }

        // Rows (1+nd)..(1+nd+n_balance): balance equations.
        // For each s' in 0..N-1:
        //   Σ_{s,a} ρ(s,a)·P(s'|s,a) − Σ_a ρ(s',a) = 0
        for s_prime in 0..n_balance {
            let row = 1 + nd + s_prime;
            // Inflow: Σ_{s,a} ρ(s,a) · P(s'|s,a)
            for s in 0..N {
                for a in 0..A {
                    let j = s * A + a;
                    mat[row][j] += kernel.transition(s, a, s_prime) as f64;
                }
            }
            // Outflow: − Σ_a ρ(s',a) = −ν(s')
            for a in 0..A {
                let j = s_prime * A + a;
                mat[row][j] -= 1.0;
            }
            rhs[row] = 0.0;
        }

        // Objective coefficients: γ₀(ρ) = Σ ρ(s,a) · gamma0_coeff(s,a).
        let mut obj = vec![0.0_f64; n_vars];
        for s in 0..N {
            for a in 0..A {
                obj[s * A + a] = p.gamma0_coeff(s, a) as f64;
            }
        }

        // Auto-select solver: BFS for small LPs, simplex for large (Plan 572).
        let best_rho_entries = solve_lp_auto(&mat, &rhs, &obj, n_vars, na);
        match best_rho_entries {
            Some(rho_entries) => {
                let sum: f32 = rho_entries.iter().map(|&v| v as f32).sum();
                let inv = if sum > 1e-9 { 1.0 / sum } else { 1.0 };
                let entries_f32: Vec<f32> = rho_entries.iter().map(|&v| (v as f32) * inv).collect();
                Ok(OccupationMeasure::from_entries_trusted(entries_f32))
            }
            None => Err(CceLpError::Infeasible),
        }
    }

    /// Solve the subjective-CCE LP for a heterogeneous player population
    /// (Plan 300).
    ///
    /// Builds `Σ_i |D_i|` constraint rows. Each row `(i, κ)` uses player `i`'s
    /// own cost tensor: `g_κ(s,a) = cost_i(s,a) − reward_deviate(i, s, κ)`.
    /// Returns the optimal occupation measure under the moderator objective
    /// `γ₀(ρ)`.
    ///
    /// Regret bound `ER(ρ̄_T) ≤ O(T⁻¹ᐟ²)` transfers from the homogeneous case
    /// (doc 62 §2 — sum of convex is convex). No new theory; pure wrapper.
    pub fn solve_heterogeneous<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        &self,
        game: &H,
    ) -> Result<OccupationMeasure<N, A>, CceLpError> {
        let na = N * A;
        let n_players = game.n_players();
        if n_players == 0 {
            return Err(CceLpError::Infeasible);
        }

        // Count total deviations: Σ_i |D_i|.
        let mut total_devs = 0usize;
        for i in 0..n_players {
            total_devs += game.deviations_for_player(i).len();
        }

        // Total variables: ρ[0..na] + s[0..total_devs] (slacks for each
        // (player, κ) constraint).
        let n_vars = na + total_devs;
        // Total equality constraints: 1 (sum) + total_devs.
        let n_cons = 1 + total_devs;

        if n_cons == 0 || n_cons > n_vars {
            return Err(CceLpError::Infeasible);
        }

        // Build constraint matrix A (n_cons × n_vars) and RHS b (n_cons).
        let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
        let mut rhs = vec![0.0_f64; n_cons];

        // Row 0: Σ ρ = 1.
        mat[0][..na].fill(1.0);
        rhs[0] = 1.0;

        // Rows 1..: for each (player, κ) pair, g_{i,κ} · ρ + s_{i,κ} = 0
        //   where g_{i,κ}(s,a) = cost_i(s,a) − reward_deviate(i, s, κ).
        let mut row = 1usize;
        for i in 0..n_players {
            for kappa in game.deviations_for_player(i) {
                for s in 0..N {
                    for a in 0..A {
                        let j = s * A + a;
                        let g = game.reward_follow(i, s, a) as f64
                            - game.reward_deviate(i, s, kappa) as f64;
                        mat[row][j] = g;
                    }
                }
                mat[row][na + (row - 1)] = 1.0; // slack column
                rhs[row] = 0.0;
                row += 1;
            }
        }
        debug_assert_eq!(row, n_cons, "constraint row count mismatch");

        // Objective coefficients: γ₀(ρ) = Σ ρ(s,a) · gamma0_coeff(s,a).
        let mut obj = vec![0.0_f64; n_vars];
        for s in 0..N {
            for a in 0..A {
                obj[s * A + a] = game.gamma0_coeff(s, a) as f64;
            }
        }

        // Auto-select solver: BFS for small LPs, simplex for large (Plan 572).
        let best_rho_entries = solve_lp_auto(&mat, &rhs, &obj, n_vars, na);
        match best_rho_entries {
            Some(rho_entries) => {
                // Final normalization to exactly sum = 1 (within f32 tolerance).
                let sum: f32 = rho_entries.iter().map(|&v| v as f32).sum();
                let inv = if sum > 1e-9 { 1.0 / sum } else { 1.0 };
                let entries_f32: Vec<f32> = rho_entries.iter().map(|&v| (v as f32) * inv).collect();
                Ok(OccupationMeasure::from_entries_trusted(entries_f32))
            }
            None => Err(CceLpError::Infeasible),
        }
    }

    /// Verify that `ρ` is a subjective-CCE: for every player `i` and every
    /// `κ ∈ D_i`, `γ_i(ρ) ≤ γ_dev_i(ρ, κ) + ε`. Early-exit on first
    /// violation.
    pub fn is_heterogeneous_cce<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        &self,
        rho: &OccupationMeasure<N, A>,
        game: &H,
        epsilon: f32,
    ) -> bool {
        for i in 0..game.n_players() {
            let gamma_i = game.gamma_player(i, rho);
            for kappa in game.deviations_for_player(i) {
                let gamma_dev_i = game.gamma_dev_player(i, rho, kappa);
                if gamma_i - gamma_dev_i > epsilon {
                    return false;
                }
            }
        }
        true
    }

    /// Constraint-generation solver for heterogeneous CCE (Plan 572).
    ///
    /// Starts with no deviation constraints (just the sum-to-1 constraint),
    /// solves the relaxed LP, finds the most-violated deviation constraint via
    /// the external-regret separation oracle, adds it, and re-solves. Converges
    /// in `O(|active set|)` iterations — typically far fewer than `Σ_i |D_i|`
    /// because only the binding deviations end up active.
    ///
    /// This is the production path for large heterogeneous games (e.g., the
    /// N=9, A=9 two-player RPS case from Issue 575) where the full LP has too
    /// many variables for BFS enumeration.
    ///
    /// Returns the same shape as [`solve_heterogeneous`]. The result passes
    /// [`is_heterogeneous_cce`] at convergence.
    pub fn solve_heterogeneous_cg<
        const N: usize,
        const A: usize,
        H: HeterogeneousPayoff<N, A>,
    >(
        &self,
        game: &H,
    ) -> Result<OccupationMeasure<N, A>, CceLpError> {
        self.solve_heterogeneous_cg_with_tolerance(game, 1e-6)
    }

    /// Constraint-generation solver with explicit convergence tolerance.
    ///
    /// `epsilon` is the violation threshold below which a deviation constraint
    /// is considered satisfied (not added to the active set).
    pub fn solve_heterogeneous_cg_with_tolerance<
        const N: usize,
        const A: usize,
        H: HeterogeneousPayoff<N, A>,
    >(
        &self,
        game: &H,
        epsilon: f64,
    ) -> Result<OccupationMeasure<N, A>, CceLpError> {
        let na = N * A;
        let n_players = game.n_players();
        if n_players == 0 {
            return Err(CceLpError::Infeasible);
        }

        // Upper bound on iterations: every (player, κ) pair could become active.
        let max_total_devs: usize = (0..n_players).map(|i| game.deviations_for_player(i).len()).sum();
        let max_iters = max_total_devs + 1;

        // Active constraints: (player_idx, deviation_idx_within_player) pairs.
        let mut active: Vec<(usize, usize)> = Vec::new();

        // Objective coefficients (constant across iterations).
        let mut obj_full = vec![0.0_f64; na];
        for s in 0..N {
            for a in 0..A {
                obj_full[s * A + a] = game.gamma0_coeff(s, a) as f64;
            }
        }

        for _iter in 0..max_iters {
            // Build the relaxed LP with the current active set.
            // Variables: ρ[0..na] + s[0..active.len()].
            let n_active = active.len();
            let n_vars = na + n_active;
            let n_cons = 1 + n_active;
            let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
            let mut rhs = vec![0.0_f64; n_cons];

            // Row 0: Σ ρ = 1.
            mat[0][..na].fill(1.0);
            rhs[0] = 1.0;

            // Rows 1..=n_active: each active (player, κ) constraint.
            for (k, &(player, dev_idx)) in active.iter().enumerate() {
                let kappa = &game.deviations_for_player(player)[dev_idx];
                for s in 0..N {
                    for a in 0..A {
                        let j = s * A + a;
                        let g = game.reward_follow(player, s, a) as f64
                            - game.reward_deviate(player, s, kappa) as f64;
                        mat[1 + k][j] = g;
                    }
                }
                mat[1 + k][na + k] = 1.0; // slack
                rhs[1 + k] = 0.0;
            }

            // Objective: γ₀ over ρ only (slacks have zero objective).
            let mut obj = vec![0.0_f64; n_vars];
            obj[..na].copy_from_slice(&obj_full);

            // Solve the relaxed LP.
            let rho_entries = solve_lp_auto(&mat, &rhs, &obj, n_vars, na)
                .ok_or(CceLpError::Infeasible)?;

            // Convert to OccupationMeasure for the separation oracle.
            let sum: f32 = rho_entries.iter().map(|&v| v as f32).sum();
            let inv = if sum > 1e-9 { 1.0 / sum } else { 1.0 };
            let entries_f32: Vec<f32> =
                rho_entries.iter().map(|&v| (v as f32) * inv).collect();
            let rho = OccupationMeasure::from_entries_trusted(entries_f32);

            // Separation oracle: find the most-violated (player, κ) not yet active.
            let mut worst: Option<(usize, usize, f64)> = None; // (player, dev_idx, violation)
            for i in 0..n_players {
                let gamma_i = game.gamma_player(i, &rho);
                for (di, kappa) in game.deviations_for_player(i).iter().enumerate() {
                    // Skip if already active.
                    if active.contains(&(i, di)) {
                        continue;
                    }
                    let gamma_dev_i = game.gamma_dev_player(i, &rho, kappa);
                    let violation = (gamma_i - gamma_dev_i) as f64;
                    if violation > epsilon {
                        match worst {
                            None => worst = Some((i, di, violation)),
                            Some((_, _, w)) if violation > w => worst = Some((i, di, violation)),
                            _ => {}
                        }
                    }
                }
            }

            if let Some((i, di, _v)) = worst {
                    active.push((i, di));
                } else {
                    // No violated constraint found — converged.
                    return Ok(rho);
                }
        }

        // Exhausted iteration budget without convergence — numerical failure.
        Err(CceLpError::NumericalError("constraint generation did not converge"))
    }
}

// -------- Internal helpers --------

/// BFS candidate count threshold: LPs with more than this many BFS candidates
/// use the two-phase primal simplex instead of exhaustive enumeration. 50_000
/// candidates is ~1ms of BFS work on a modern CPU; larger LPs benefit from the
/// simplex's pivot-based search (Plan 572).
const BFS_CANDIDATE_CUTOFF: u128 = 50_000;

/// Auto-select between BFS enumeration and the two-phase primal simplex based
/// on the LP size (Plan 572). For small LPs (`C(n_vars, n_cons) ≤ cutoff`),
/// BFS is exact and fast. For large LPs, the simplex is used instead.
///
/// Both solvers return the same shape: `Option<Vec<f64>>` of the first `na`
/// entries (the `ρ` variables). The caller normalizes to f32.
fn solve_lp_auto(
    mat: &[Vec<f64>],
    rhs: &[f64],
    obj: &[f64],
    n_vars: usize,
    na: usize,
) -> Option<Vec<f64>> {
    let n_cons = mat.len();
    if n_cons == 0 {
        return None;
    }
    // Estimate C(n_vars, n_cons). If it exceeds the cutoff, use simplex.
    if binomial_exceeds(n_vars, n_cons, BFS_CANDIDATE_CUTOFF) {
        crate::cce::simplex::solve_simplex(mat, rhs, obj, n_vars, na)
    } else {
        enumerate_bfs(mat, rhs, obj, n_vars, na)
    }
}

/// Check whether `C(n, k)` exceeds `threshold` without computing the full
/// binomial (avoids overflow on large values). Returns `true` as soon as the
/// running product surpasses `threshold`.
fn binomial_exceeds(n: usize, k: usize, threshold: u128) -> bool {
    if k > n {
        return false; // C(n, k) = 0, doesn't exceed anything
    }
    let k = k.min(n - k); // symmetry: C(n, k) = C(n, n-k)
    if k == 0 {
        return 1 > threshold;
    }
    let mut result: u128 = 1;
    for i in 0..k {
        // result = result * (n - i) / (i + 1)
        // To avoid overflow, multiply then divide. If result ever exceeds
        // threshold, return early.
        result = result.saturating_mul((n - i) as u128);
        result /= (i + 1) as u128;
        if result > threshold {
            return true;
        }
    }
    result > threshold
}

/// Shared BFS-enumeration loop (Plan 300 T2.1).
///
/// For each subset of `n_cons` columns (where `n_cons = mat.len()` = number of
/// equality constraints), solve the `n_cons × n_cons` linear system
/// `A[:, combo] · x = b`, scatter into the full `n_vars` solution vector,
/// check non-negativity, and keep the minimum-objective feasible candidate.
///
/// Returns the best `ρ` entries (the first `na` slots) or `None` if no BFS is
/// feasible. The caller is responsible for f32 normalization on output.
///
/// `mat`: `n_cons × n_vars` constraint matrix. `rhs`: `n_cons` vector.
/// `obj`: `n_vars` objective coefficients. `na`: count of `ρ` variables
/// (slacks come after and have zero objective).
fn enumerate_bfs(
    mat: &[Vec<f64>],
    rhs: &[f64],
    obj: &[f64],
    n_vars: usize,
    na: usize,
) -> Option<Vec<f64>> {
    let n_cons = mat.len();
    if n_cons == 0 || n_cons > n_vars {
        return None;
    }

    let mut best_obj_val = f64::INFINITY;
    let mut best_rho_entries: Option<Vec<f64>> = None;
    let mut x = vec![0.0_f64; n_vars];

    let mut combo: Vec<usize> = (0..n_cons).collect();
    loop {
        if let Some(x_basic) = solve_square_system(mat, rhs, &combo) {
            // Scatter into the full solution vector (zero the others).
            x.fill(0.0);
            for (i, &col) in combo.iter().enumerate() {
                x[col] = x_basic[i];
            }

            // Feasibility: all variables ≥ -tol.
            const NEG_TOL: f64 = -1e-7;
            if x.iter().all(|&v| v >= NEG_TOL) {
                // Clamp tiny negatives to zero.
                for xi in x.iter_mut() {
                    if *xi < 0.0 {
                        *xi = 0.0;
                    }
                }

                // Renormalize ρ entries (guard against tiny drift).
                let sum_rho: f64 = x[..na].iter().copied().sum();
                if sum_rho > 1e-9 {
                    let inv = 1.0 / sum_rho;
                    for xi in x[..na].iter_mut() {
                        *xi *= inv;
                    }
                }

                let obj_val: f64 = x[..na]
                    .iter()
                    .zip(obj[..na].iter())
                    .map(|(&xi, &ci)| xi * ci)
                    .sum();

                if obj_val < best_obj_val {
                    best_obj_val = obj_val;
                    best_rho_entries = Some(x[..na].to_vec());
                }
            }
        }

        if !next_combination(&mut combo, n_vars) {
            break;
        }
    }

    best_rho_entries
}

/// Solve the `n × n` linear system `A[:, combo] · x = b` via Gaussian
/// elimination with partial pivoting. Returns `None` if the submatrix is
/// singular.
fn solve_square_system(mat: &[Vec<f64>], rhs: &[f64], combo: &[usize]) -> Option<Vec<f64>> {
    let n = combo.len();
    // Build augmented matrix [B | b].
    let mut aug = vec![vec![0.0_f64; n + 1]; n];
    for row in 0..n {
        for (col, &var) in combo.iter().enumerate() {
            aug[row][col] = mat[row][var];
        }
        aug[row][n] = rhs[row];
    }

    // Forward elimination with partial pivoting.
    for pivot in 0..n {
        // Find the row with max abs value in column `pivot`.
        let mut max_row = pivot;
        let mut max_val = aug[pivot][pivot].abs();
        for (row_off, aug_row) in aug[(pivot + 1)..n].iter().enumerate() {
            let val = aug_row[pivot].abs();
            if val > max_val {
                max_val = val;
                max_row = pivot + 1 + row_off;
            }
        }
        if max_val < 1e-12 {
            return None; // singular
        }
        if max_row != pivot {
            aug.swap(pivot, max_row);
        }

        // Eliminate below.
        let pivot_val = aug[pivot][pivot];
        for row in (pivot + 1)..n {
            let factor = aug[row][pivot] / pivot_val;
            if factor == 0.0 {
                continue;
            }
            // Safe disjoint borrow: pivot < row, so split_at_mut(row) puts
            // `pivot` in the left part and `row` at right[0].
            let (left, right) = aug.split_at_mut(row);
            let aug_pivot_row = &left[pivot];
            let aug_row = &mut right[0];
            for (aug_row_col, &aug_pivot_col) in aug_row[pivot..=n]
                .iter_mut()
                .zip(aug_pivot_row[pivot..=n].iter())
            {
                *aug_row_col -= factor * aug_pivot_col;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        let diag = aug[i][i];
        if diag.abs() < 1e-12 {
            return None;
        }
        x[i] = s / diag;
    }
    Some(x)
}

/// Advance `combo` to the next combination of `combo.len()` items from `0..n`.
/// Returns `false` when the last combination has been reached.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    if k == 0 {
        return false;
    }
    // Find the rightmost index that can be incremented.
    let mut i = k as isize - 1;
    while i >= 0 {
        if combo[i as usize] < n - k + i as usize {
            combo[i as usize] += 1;
            // Reset the tail.
            for j in (i as usize + 1)..k {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
        i -= 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cce::types::Deviation;

    #[test]
    fn next_combination_enumerates_all() {
        let mut combo = vec![0, 1, 2];
        let n = 5;
        let mut count = 1; // initial combo counts
        while next_combination(&mut combo, n) {
            count += 1;
        }
        // C(5, 3) = 10.
        assert_eq!(count, 10);
    }

    #[test]
    fn solve_square_system_identity() {
        let mat = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let rhs = vec![3.0, 5.0, 7.0];
        let combo = vec![0, 1, 2];
        let x = solve_square_system(&mat, &rhs, &combo).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-9);
        assert!((x[1] - 5.0).abs() < 1e-9);
        assert!((x[2] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn solve_square_system_singular_returns_none() {
        let mat = vec![vec![1.0, 2.0], vec![2.0, 4.0]]; // rank 1
        let rhs = vec![1.0, 2.0];
        let combo = vec![0, 1];
        assert!(solve_square_system(&mat, &rhs, &combo).is_none());
    }

    /// LP solver on the chicken game: with `γ₀ = γ` (player 1's cost), the
    /// LP correctly finds the most selfish CCE — player 1 always plays T
    /// against an opponent playing S, yielding cost -4.
    #[test]
    fn lp_solve_chicken_finds_minimum_cost_cce() {
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

        let rho_star = CceLp::new().solve(&d, &p).expect("chicken LP feasible");

        // Sanity: ρ⋆ is a valid CCE.
        assert!(
            CceLp::new().is_cce(&rho_star, &d, &p, 1e-4),
            "LP solution must be a CCE"
        );

        // The minimum-cost CCE for player 1 is (T,S): player 1 plays T,
        // opponent plays S. cost = -R[T][S] = -4.
        let gamma0 = p.gamma0(&rho_star);
        assert!(
            (gamma0 - (-4.0)).abs() < 1e-3,
            "expected γ₀ = -4 (T,S), got {gamma0}"
        );
    }

    /// LP solver on chicken with **welfare-based** `γ₀`: the moderator
    /// minimizes negative welfare. **Note**: this test models only player 1's
    /// CCE constraints (the deviation class D contains only player 1's
    /// deviations). The resulting optimum may exploit player 2 — full
    /// game CCE requires adding player 2's deviation constraints
    /// (riir-ai Plan 325 scope).
    #[test]
    fn lp_solve_chicken_welfare_player1_only() {
        const R: [[f32; 2]; 2] = [[3.0, 1.0], [4.0, 0.0]];

        struct ChickenWelfare;
        impl PayoffTensor<4, 2> for ChickenWelfare {
            fn reward_follow(&self, state: usize, action: usize) -> f32 {
                let s_2 = state % 2;
                -R[action][s_2]
            }
            // γ₀ = negative welfare. Player 1 plays `action`, player 2 plays
            // `s_2` (assumed honest). Welfare = R[action][s_2] + R[s_2][action]
            // (symmetric game: player 2's reward at (a_1, a_2) = R[a_2][a_1]).
            fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
                let mut g = 0.0;
                for s in 0..4 {
                    let s_2 = s % 2;
                    for (a, r_a) in R.iter().enumerate() {
                        let welfare_cost = -(r_a[s_2] + R[s_2][a]);
                        g += rho.at(s, a) * welfare_cost;
                    }
                }
                g
            }
            fn gamma0_coeff(&self, state: usize, action: usize) -> f32 {
                let s_2 = state % 2;
                -(R[action][s_2] + R[s_2][action])
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
        let p = ChickenWelfare;

        let rho_star = CceLp::new()
            .solve(&d, &p)
            .expect("chicken welfare LP feasible");
        assert!(CceLp::new().is_cce(&rho_star, &d, &p, 1e-4));

        // Player-1-only optimum: ρ(state 0 = (S,S), action S) + ρ(state 1 =
        // (S,T), action S) with equal mass. Player 1 always plays S (never
        // deviates). Welfare = 0.5·6 + 0.5·5 = 5.5 (cost -5.5). This is a
        // valid player-1 CCE but NOT a player-2 CCE (player 2 wants to
        // deviate from T in state (S,T)).
        let gamma0 = p.gamma0(&rho_star);
        assert!(
            (gamma0 - (-5.5)).abs() < 1e-3,
            "expected γ₀ = -5.5 (player-1-only welfare), got {gamma0}"
        );
    }

    /// LP solver on the emission-abatement problem (simplified, no dynamics):
    /// the optimal CCE concentrates all mass on `(Low, Abate)` — the cheapest
    /// state-action pair — with cost 1.0.
    #[test]
    fn lp_solve_emission_finds_cheapest_cce() {
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

        let rho_star = CceLp::new().solve(&d, &p).expect("emission LP feasible");
        assert!(CceLp::new().is_cce(&rho_star, &d, &p, 1e-4));

        // Without dynamics, the mediator concentrates on the cheapest pair:
        // (Low=0, Abate=0) with cost 1.0.
        let gamma0 = p.gamma0(&rho_star);
        assert!(
            (gamma0 - 1.0).abs() < 1e-3,
            "expected γ₀ = 1.0, got {gamma0}"
        );

        // ρ⋆ should put all mass on (state=Low, action=Abate).
        let mass_low_abate = rho_star.at(0, 0);
        assert!(
            (mass_low_abate - 1.0).abs() < 1e-3,
            "mass(Low,Abate) = {mass_low_abate}"
        );
    }

    // ── Plan 569: transition-kernel-constrained CCE ──────────────────────────

    /// MDP game for the transition-kernel PoC (Plan 569 / Issue 574).
    /// 2 states {LOW, HIGH}, 2 actions {WAIT, INVEST}, action-dependent
    /// transitions. The unconstrained CCE exploits the free state distribution
    /// (all mass on (HIGH, WAIT), γ₀ = 0); the constrained CCE recovers the
    /// true MDP optimum (always WAIT, γ₀ = 5/6).
    mod transition_kernel {
        use super::*;
        use crate::cce::types::TransitionKernel;

        const LOW: usize = 0;
        const HIGH: usize = 1;
        const WAIT: usize = 0;
        const INVEST: usize = 1;

        /// Transition kernel P(s'|s,a).
        const TRANSITION: [[[f64; 2]; 2]; 2] = [
            [[0.9, 0.1], [0.4, 0.6]], // s = LOW
            [[0.5, 0.5], [0.8, 0.2]], // s = HIGH
        ];

        /// Cost: cost(s, a) — minimize.
        const COST: [[f64; 2]; 2] = [[1.0, 3.0], [0.0, 2.0]];

        struct MdpGame;
        impl PayoffTensor<2, 2> for MdpGame {
            fn reward_follow(&self, state: usize, action: usize) -> f32 {
                COST[state][action] as f32
            }
            fn gamma0(&self, _rho: &OccupationMeasure<2, 2>) -> f32 {
                0.0 // gamma0_coeff (default = reward_follow) handles the objective
            }
        }

        struct MdpKernel;
        impl TransitionKernel<2, 2> for MdpKernel {
            fn transition(&self, state: usize, action: usize, next_state: usize) -> f32 {
                TRANSITION[state][action][next_state] as f32
            }
        }

        struct MdpDevs {
            v: Vec<Deviation<2, 2>>,
        }
        impl DeviationClass<2, 2> for MdpDevs {
            fn deviations(&self) -> &[Deviation<2, 2>] {
                &self.v
            }
        }
        fn devs() -> MdpDevs {
            MdpDevs {
                v: vec![
                    Deviation::<2, 2>::constant(0, WAIT),
                    Deviation::<2, 2>::constant(1, INVEST),
                ],
            }
        }

        /// G1: unconstrained CCE exploits the free state distribution.
        #[test]
        fn g1a_unconstrained_artifact() {
            let d = devs();
            let p = MdpGame;
            let rho = CceLp::new().solve(&d, &p).expect("unconstrained feasible");
            // Artifact: all mass on (HIGH, WAIT), γ₀ = 0.
            assert!((rho.at(HIGH, WAIT) - 1.0).abs() < 0.05, "expected all mass on (HIGH, WAIT)");
            let mut gamma0 = 0.0_f64;
            for (s, cost_s) in COST.iter().enumerate() {
                for (a, &cost_sa) in cost_s.iter().enumerate() {
                    gamma0 += rho.at(s, a) as f64 * cost_sa;
                }
            }
            assert!(gamma0.abs() < 0.05, "unconstrained γ₀ should be ≈ 0 (artifact), got {gamma0}");
        }

        /// G1: constrained CCE matches the true MDP optimum (5/6).
        #[test]
        fn g1b_constrained_matches_true_optimum() {
            let d = devs();
            let p = MdpGame;
            let k = MdpKernel;
            let rho = CceLp::new()
                .solve_with_dynamics(&d, &p, &k)
                .expect("constrained feasible");

            let mut gamma0 = 0.0_f64;
            for (s, cost_s) in COST.iter().enumerate() {
                for (a, &cost_sa) in cost_s.iter().enumerate() {
                    gamma0 += rho.at(s, a) as f64 * cost_sa;
                }
            }
            // True MDP optimum: always WAIT, γ₀ = 5/6 ≈ 0.833.
            assert!(
                (gamma0 - 5.0 / 6.0).abs() < 0.05,
                "constrained γ₀ should be ≈ 5/6 (artifact closed), got {gamma0}"
            );

            // ρ should match the optimal policy (always WAIT).
            assert!((rho.at(LOW, WAIT) - 5.0 / 6.0).abs() < 0.05);
            assert!((rho.at(HIGH, WAIT) - 1.0 / 6.0).abs() < 0.05);
        }

        /// G1: the constrained ρ is still a valid CCE (no over-restriction).
        #[test]
        fn g1c_constrained_is_valid_cce() {
            let d = devs();
            let p = MdpGame;
            let k = MdpKernel;
            let rho = CceLp::new()
                .solve_with_dynamics(&d, &p, &k)
                .expect("constrained feasible");
            assert!(
                CceLp::new().is_cce(&rho, &d, &p, 1e-4),
                "constrained ρ must still be a valid CCE"
            );
        }

        /// G1: balance equation is satisfied for the constrained solution.
        #[allow(clippy::needless_range_loop)] // 3D TRANSITION[s][a][s'] indexing needs numeric indices
        #[test]
        fn g1d_balance_equation_satisfied() {
            let d = devs();
            let p = MdpGame;
            let k = MdpKernel;
            let rho = CceLp::new()
                .solve_with_dynamics(&d, &p, &k)
                .expect("constrained feasible");

            // Check ν(s') = Σ_{s,a} ρ(s,a)·P(s'|s,a) for each state.
            for s_prime in 0..2 {
                let mut marginal = 0.0_f64;
                for a in 0..2 {
                    marginal += rho.at(s_prime, a) as f64;
                }
                let mut inflow = 0.0_f64;
                for s in 0..2 {
                    for a in 0..2 {
                        inflow += rho.at(s, a) as f64 * TRANSITION[s][a][s_prime];
                    }
                }
                assert!((marginal - inflow).abs() < 1e-4, "balance violated for s'={s_prime}: ν={marginal}, inflow={inflow}");
            }
        }
    }

    // ═══ Plan 572: constraint-generation wrapper tests ═══

    /// Constraint generation must match the direct solve on the emission game.
    #[test]
    fn cg_matches_direct_solve_on_emission() {
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

        // Wrap as a 1-player heterogeneous game for the CG path.
        use crate::cce::heterogeneous::PerPlayerGame;
        let game = PerPlayerGame::new(vec![(&p, &d)]);

        let rho_cg = CceLp::new()
            .solve_heterogeneous_cg(&game)
            .expect("CG feasible");
        assert!(
            CceLp::new().is_heterogeneous_cce(&rho_cg, &game, 1e-4),
            "CG result must be a valid CCE"
        );

        // Compare γ₀ with direct solve.
        let rho_direct = CceLp::new()
            .solve_heterogeneous(&game)
            .expect("direct feasible");
        let g_cg = game.gamma0(&rho_cg);
        let g_direct = game.gamma0(&rho_direct);
        assert!(
            (g_cg - g_direct).abs() < 1e-3,
            "CG γ₀={g_cg} must match direct γ₀={g_direct}"
        );
    }

    /// Constraint generation must match the direct solve on a 2-player game.
    /// Uses the same payoff tensor for both players (homogeneous) to fit the
    /// `PerPlayerGame<P, D>` shape — the test verifies CG mechanics, not game
    /// semantics.
    #[test]
    fn cg_matches_direct_solve_on_pd() {
        const R: [[f32; 2]; 2] = [[3.0, 0.0], [5.0, 1.0]];
        struct Pd;
        impl PayoffTensor<2, 2> for Pd {
            fn reward_follow(&self, _s: usize, a: usize) -> f32 {
                -R[a][0]
            }
            fn gamma0(&self, rho: &OccupationMeasure<2, 2>) -> f32 {
                self.gamma(rho)
            }
        }
        struct TwoDevs {
            v: Vec<Deviation<2, 2>>,
        }
        impl DeviationClass<2, 2> for TwoDevs {
            fn deviations(&self) -> &[Deviation<2, 2>] {
                &self.v
            }
        }
        use crate::cce::heterogeneous::PerPlayerGame;

let d = TwoDevs {
            v: vec![
                Deviation::<2, 2>::constant(0, 0),
                Deviation::<2, 2>::constant(1, 1),
            ],
        };
        let p = Pd;
        let game = PerPlayerGame::new(vec![(&p, &d), (&p, &d)]);

        let rho_cg = CceLp::new()
            .solve_heterogeneous_cg(&game)
            .expect("CG feasible");
        let rho_direct = CceLp::new()
            .solve_heterogeneous(&game)
            .expect("direct feasible");
        assert!(
            CceLp::new().is_heterogeneous_cce(&rho_cg, &game, 1e-4),
            "CG result must be a valid CCE"
        );
        let g_cg = game.gamma0(&rho_cg);
        let g_direct = game.gamma0(&rho_direct);
        assert!(
            (g_cg - g_direct).abs() < 1e-3,
            "CG γ₀={g_cg} must match direct γ₀={g_direct}"
        );
    }
}
