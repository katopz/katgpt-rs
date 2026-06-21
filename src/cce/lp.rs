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
use crate::cce::types::{DeviationClass, HeterogeneousPayoff, OccupationMeasure, PayoffTensor};

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
        for val in &mut mat[0][..na] {
            *val = 1.0;
        }
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

        // Enumerate BFS via the shared helper (Plan 300 T2.1: DRY with
        // `solve_heterogeneous`).
        let best_rho_entries = enumerate_bfs(&mat, &rhs, &obj, n_vars, na);
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
        for val in &mut mat[0][..na] {
            *val = 1.0;
        }
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

        let best_rho_entries = enumerate_bfs(&mat, &rhs, &obj, n_vars, na);
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
}

// -------- Internal helpers --------

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
            for xi in x.iter_mut() {
                *xi = 0.0;
            }
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
}
