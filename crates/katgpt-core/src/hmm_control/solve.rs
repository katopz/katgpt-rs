//! HMM homeostatic control solver — the paper's exact backward-message
//! recursion (Plan 590 / Research 543; paper arXiv:2609.07508, Eq. 14-15).
//!
//! ```text
//! a*_t(x) = argmax_a  e[t][x][a] · Σ_j p[x][a][j] · β_{t+1}[j]
//! β_t(x)  = e[t][x][a*_t(x)] · Σ_j p[x][a*_t(x)][j] · β_{t+1}[j]
//! β_t(x)  = P(y_{t:T} = y_d | x)        ← a probability, bounded [0, 1]
//! ```
//!
//! where `e[t][x][a] = P(y_t = y_d | x, a)` is the per-action emission.
//!
//! # Why deterministic, why multiplicative
//!
//! The objective `max_π P(y = y_d over all t)` is LINEAR in each π_t over
//! a simplex → the optimum is at a vertex → the optimal policy is strictly
//! deterministic (the paper's first main theorem). The messages multiply
//! probabilities instead of summing rewards (max-product on the (max, ×)
//! semiring): one low-emission step multiplies the whole future down —
//! there is no discount factor, the horizon T replaces γ.
//!
//! # Risk sensitivity
//!
//! Unlike control-as-inference (which follows the best successor
//! regardless of transition mass — the paper's "wishful thinking"), the
//! argmax weighs `p(x'|x,a) · β(x')`: a tiny-probability path to a perfect
//! state contributes its probability, not its existence. The
//! `golden_parity_paper_t2_fixture` test pins exactly this divergence.
//!
//! # Modelless + sync boundary
//!
//! Pure deterministic arithmetic on caller-owned arrays — no allocation,
//! no RNG, no training, no softmax (the policy is an argmax, not a
//! categorical sample). Nothing crosses a sync boundary; β is a local
//! per-state scalar.
//!
//! # UQ floor ("Report the Floor") — N/A
//!
//! β is an exact model-computed probability, not a calibrated uncertainty
//! estimate over data. The conformal-naive floor does not apply.
//!
//! # References
//!
//! - Research: `katgpt-rs/.research/543_HMM_Homeostatic_Control_Deterministic_Drives.md`
//! - Private runtime guide: `riir-ai/.research/370_per_npc_homeostatic_drive_control_guide.md`
//! - Classical lineage (honesty note): the recursion is stochastic
//!   shortest-path reachability (Bertsekas; probabilistic model checking);
//!   the paper's contribution is the homeostatic framing + determinism
//!   theorem, ours is the two-mode fusion on the tabular substrate.

use super::types::HmmSolution;
use crate::tabular_kernel::{DENSE_ROW, row_dot, row_onehot};

/// Strictly-deterministic argmax over an action row — ties break to the
/// LOWEST index (strict `>` keeps the first maximum).
#[inline]
fn argmax_lowest<const A: usize>(vals: &[f32; A]) -> u16 {
    let mut best = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (k, &v) in vals.iter().enumerate() {
        if v > max {
            max = v;
            best = k;
        }
    }
    best as u16
}

/// The HMM homeostatic control operator (paper Eqs. 14-15).
///
/// # Example — two-step reachability (paper §3 shape)
///
/// ```
/// use katgpt_core::hmm_control::{HmmControlSolver, invariant_emission};
///
/// // States: 0 = start, 1 = perfect (x*2), 2 = decent (x'2), 3 = dead.
/// // Actions: 0 = risky (0.01 → perfect, 0.99 → dead), 1 = safe (1.0 → decent).
/// const N: usize = 4;
/// const A: usize = 2;
/// const T: usize = 2;
/// let mut p = [[[0.0f32; N]; A]; N];
/// // Start: risky vs safe.
/// p[0][0][1] = 0.01;
/// p[0][0][3] = 0.99;
/// p[0][1][2] = 1.0;
/// // Perfect and decent absorb; dead absorbs.
/// for s in [1usize, 2, 3] {
///     p[s][0][s] = 1.0;
///     p[s][1][s] = 1.0;
/// }
/// // Emissions: perfect 1.0, decent 0.5, others 0 (dead) / 0.5 (start).
/// let mut e0 = [[0.5f32; A]; N];
/// e0[1] = [1.0; A];
/// e0[2] = [0.5; A];
/// e0[3] = [0.0; A];
/// let e = invariant_emission::<N, A, T>(&e0);
/// let sol = HmmControlSolver::<N, A, T>::new().solve(&p, &e);
/// // The safe action wins: 0.5·1.0·0.5 = 0.25 beats risky 0.5·0.01·1.0 = 0.005
/// // — control-as-inference would pick risky (only it can reach the perfect
/// // state), which is exactly the paper's wishful-thinking divergence.
/// assert_eq!(sol.optimal_action(0, 0), 1);
/// assert!((sol.beta[0][0] - 0.25).abs() < 1e-6);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct HmmControlSolver<const N: usize, const A: usize, const T: usize>;

impl<const N: usize, const A: usize, const T: usize> HmmControlSolver<N, A, T> {
    /// Construct the solver. The recursion has no knobs — the horizon is
    /// the const generic `T`, and β messages are exact probabilities
    /// (no floor, no discount).
    pub fn new() -> Self {
        Self
    }

    /// Run the exact backward-message recursion (paper Eqs. 14-15).
    ///
    /// - `p`: frozen transition kernel `[N][A][N]`. Rows may be
    ///   sub-stochastic — a mass leak is unmodelled termination and lowers
    ///   β (the honest semantics; no explicit dead state is required).
    /// - `e`: per-step emission tables `e[t][x][a] = P(y_t = y_d | x, a)`
    ///   — independent probabilities per action, NOT distributions over a.
    ///
    /// Panics only on index-inversion bugs (const shapes guarantee the
    /// rest). Input validation is [`super::types::validate_tables`] —
    /// opt-in, off the hot path.
    pub fn solve(
        &self,
        p: &[[[f32; N]; A]; N],
        e: &[[[f32; A]; N]; T],
    ) -> HmmSolution<N, A, T> {
        debug_assert!(T >= 1, "horizon must be at least one step");
        let mut solution = HmmSolution::<N, A, T> {
            beta: [[0.0f32; N]; T],
            policy: [[0u16; N]; T],
        };

        // One-hot scan per kernel row — reused across all T steps (the
        // kernel is frozen; only β changes).
        let mut onehot = [[DENSE_ROW; A]; N];
        for (i, row_i) in p.iter().enumerate() {
            for (k, row_ik) in row_i.iter().enumerate() {
                onehot[i][k] = row_onehot(row_ik);
            }
        }

        // Terminal step (t = T, stored at index T-1): β = max_a e; the
        // policy is the argmax (all-equal rows tie-break to index 0).
        // Two-buffer discipline: `next` holds the full β_{t+1} vector while
        // `cur` is filled; states must never read a half-updated vector
        // (successor cycles would read β_t values as if they were β_{t+1}).
        let last = T - 1;
        let mut next = [0.0f32; N];
        for (i, e_row) in e[last].iter().enumerate() {
            next[i] = row_max(e_row);
            solution.beta[last][i] = next[i];
            solution.policy[last][i] = argmax_lowest(e_row);
        }

        // Backward sweep t = T-1 .. 1 (stored indices T-2 .. 0):
        //   val[i][k] = e[t][i][k] · Σ_j p[i][k][j] · β_{t+1}[j]
        //   β_t[i]    = max_k val[i][k];  a*_t(i) = argmax_k val[i][k]
        for t in (0..last).rev() {
            let e_t = &e[t];
            let mut cur = [0.0f32; N];
            for (i, p_i) in p.iter().enumerate() {
                let e_row = &e_t[i];
                // All-equal-zero emission short-circuit: every action's
                // value is 0 regardless of successors (skip the dots —
                // large dead regions are common: e.g. emission 0 off-goal).
                if row_max(e_row) == 0.0 {
                    solution.beta[t][i] = 0.0;
                    solution.policy[t][i] = 0;
                    continue;
                }
                let mut best_val = f32::NEG_INFINITY;
                let mut best_k = 0usize;
                for (k, p_ik) in p_i.iter().enumerate() {
                    let succ = row_dot(p_ik, &next, onehot[i][k], N);
                    let val = e_row[k] * succ;
                    if val > best_val {
                        best_val = val;
                        best_k = k;
                    }
                }
                solution.beta[t][i] = best_val;
                solution.policy[t][i] = best_k as u16;
                cur[i] = best_val;
            }
            next = cur;
        }
        solution
    }
}

/// Row max (the terminal step's β) — strict `>` keeps the first maximum,
/// matching [`argmax_lowest`]'s tie-break.
#[inline]
fn row_max<const A: usize>(vals: &[f32; A]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in vals.iter() {
        if v > max {
            max = v;
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm_control::types::{HmmInputError, invariant_emission, validate_tables};

    /// Splitmix64 PRNG (repo-standard test fixture).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
        }
    }

    /// Random (sub)stochastic kernel: `sparsity`-support rows normalized
    /// to `mass` (1.0 = stochastic, < 1.0 = sub-stochastic with a leak).
    fn random_kernel<const N: usize, const A: usize>(seed: u64, sparsity: usize, mass: f32)
     -> [[[f32; N]; A]; N] {
        let mut rng = Rng::new(seed);
        let mut p = [[[0.0f32; N]; A]; N];
        for (i, p_i) in p.iter_mut().enumerate() {
            for p_ik in p_i.iter_mut() {
                let mut total = 0.0f32;
                for (j, pj) in p_ik.iter_mut().enumerate() {
                    if j % sparsity == (i + 7) % sparsity {
                        let w = 0.1 + rng.uniform();
                        *pj = w;
                        total += w;
                    }
                }
                if total == 0.0 {
                    p_ik[i] = mass;
                } else {
                    let scale = mass / total;
                    for pj in p_ik.iter_mut() {
                        *pj *= scale;
                    }
                }
            }
        }
        p
    }

    /// Random emission table with a few guaranteed-positive entries so β
    /// is not trivially all-zero.
    fn random_emission<const N: usize, const A: usize, const T: usize>(
        seed: u64,
    ) -> [[[f32; A]; N]; T] {
        let mut rng = Rng::new(seed);
        let mut e0 = [[0.0f32; A]; N];
        for (i, row) in e0.iter_mut().enumerate() {
            for (k, ek) in row.iter_mut().enumerate() {
                // ~50% of entries positive, in [0, 1].
                *ek = if (i + k) % 2 == 0 {
                    rng.uniform()
                } else {
                    0.0
                };
            }
        }
        invariant_emission::<N, A, T>(&e0)
    }

    // ── T1.5 — the paper §3 T=2 golden fixture ──────────────────────────

    #[test]
    fn golden_parity_paper_t2_fixture() {
        // States: 0 = start x₁, 1 = perfect x*₂, 2 = decent x′₂, 3 = dead.
        // Actions: 0 = risky a₁, 1 = safe a′₁.
        const N: usize = 4;
        const A: usize = 2;
        const T: usize = 2;
        let mut p = [[[0.0f32; N]; A]; N];
        // Paper §3: a₁ is the only action that can reach x*₂ (the max-
        // emission successor) — and its remaining mass dies.
        p[0][0][1] = 0.01;
        p[0][0][3] = 0.99;
        // a′₁ leads with probability one to the decent state.
        p[0][1][2] = 1.0;
        for s in [1usize, 2, 3] {
            p[s][0][s] = 1.0;
            p[s][1][s] = 1.0;
        }
        let mut e0 = [[0.5f32; A]; N];
        e0[1] = [1.0; A]; // p(y₂ = yd | x*₂) = 1.0
        e0[2] = [0.5; A];
        e0[3] = [0.0; A];
        let e = invariant_emission::<N, A, T>(&e0);

        let solver = HmmControlSolver::<N, A, T>::new();
        let sol = solver.solve(&p, &e);

        // Hand-computed: step-2 messages β₂ = [0.5, 1.0, 0.5, 0.0].
        assert!((sol.beta[1][0] - 0.5).abs() < 1e-6);
        assert_eq!(sol.beta[1][1], 1.0);
        assert!((sol.beta[1][2] - 0.5).abs() < 1e-6);
        assert_eq!(sol.beta[1][3], 0.0);
        // Step-1 values: risky = 0.5·(0.01·1.0 + 0.99·0.0) = 0.005;
        // safe = 0.5·(1.0·0.5) = 0.25 → SAFE wins.
        assert_eq!(sol.optimal_action(0, 0), 1, "safe action must win");
        assert!((sol.beta[0][0] - 0.25).abs() < 1e-6);
        // The wishful-thinking contrast (paper Eq. 28): control-as-inference
        // in the α→∞ limit follows ONLY the max-emission successor and would
        // pick the risky action (it is the only route to x*₂, regardless of
        // the 0.01 mass). HMM control weighs the mass — pinned here as the
        // divergence the paper proves.
        //
        // Absorbing states: policy defined (tie-break index 0), β = emission.
        assert_eq!(sol.optimal_action(0, 1), 0);
        assert_eq!(sol.beta[0][3], 0.0);

        // Structural-difference parity: naive reference recursion (no
        // one-hot fast path, plain sequential accumulation).
        let (beta_ref, pol_ref) = reference_t2::<N, A, T>(&p, &e0);
        for t in 0..T {
            for i in 0..N {
                assert!(
                    (sol.beta[t][i] - beta_ref[t][i]).abs() <= 1e-6 * beta_ref[t][i].abs().max(1.0),
                    "beta mismatch at t={t} i={i}: {} vs {}",
                    sol.beta[t][i],
                    beta_ref[t][i]
                );
                assert_eq!(sol.policy[t][i], pol_ref[t][i]);
            }
        }
    }

    /// Structurally-different reference: plain backward recursion, dense
    /// sequential dots (no fast path, no SIMD). Tolerance parity on dense
    /// rows (SIMD lane order vs sequential is not bit-associative); the
    /// policy must match exactly (same argmax semantics).
    fn reference_t2<const N: usize, const A: usize, const T: usize>(
        p: &[[[f32; N]; A]; N],
        e0: &[[f32; A]; N],
    ) -> ([[f32; N]; T], [[u16; N]; T]) {
        let mut beta = [[0.0f32; N]; T];
        let mut policy = [[0u16; N]; T];
        for (i, row) in e0.iter().enumerate() {
            let mut max = f32::NEG_INFINITY;
            let mut best = 0usize;
            for (k, &ek) in row.iter().enumerate() {
                if ek > max {
                    max = ek;
                    best = k;
                }
            }
            beta[T - 1][i] = max;
            policy[T - 1][i] = best as u16;
        }
        for t in (0..T - 1).rev() {
            for i in 0..N {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_k = 0usize;
                for k in 0..A {
                    let mut succ = 0.0f32;
                    for j in 0..N {
                        succ += p[i][k][j] * beta[t + 1][j];
                    }
                    let val = e0[i][k] * succ;
                    if val > best_val {
                        best_val = val;
                        best_k = k;
                    }
                }
                beta[t][i] = best_val;
                policy[t][i] = best_k as u16;
            }
        }
        (beta, policy)
    }

    // ── T1.6 — invariants ───────────────────────────────────────────────

    #[test]
    fn invariants_beta_bounded_and_deterministic() {
        const N: usize = 48;
        const A: usize = 6;
        const T: usize = 16;
        let solver = HmmControlSolver::<N, A, T>::new();

        for (seed, mass) in [(11u64, 1.0f32), (12, 1.0), (13, 0.9), (14, 0.5)] {
            let p = random_kernel::<N, A>(seed, 4, mass);
            let e = random_emission::<N, A, T>(seed + 100);
            let sol = solver.solve(&p, &e);
            for t in 0..T {
                for i in 0..N {
                    let b = sol.beta[t][i];
                    assert!(
                        (0.0..=1.0).contains(&b),
                        "β out of [0,1] at t={t} i={i} (mass={mass}): {b}"
                    );
                    assert!(sol.policy[t][i] < A as u16);
                }
            }
            // Determinism: a second solve is byte-identical.
            let sol2 = solver.solve(&p, &e);
            for t in 0..T {
                assert_eq!(sol.beta[t], sol2.beta[t], "β not deterministic t={t}");
                assert_eq!(sol.policy[t], sol2.policy[t], "policy not deterministic t={t}");
            }
        }
    }

    #[test]
    fn sub_stochastic_rows_lower_beta() {
        // Mass leak = unmodelled termination → β strictly smaller than the
        // stochastic twin (the no-explicit-dead-state semantics).
        const N: usize = 16;
        const A: usize = 3;
        const T: usize = 8;
        let p_full = random_kernel::<N, A>(21, 3, 1.0);
        let p_leak = random_kernel::<N, A>(21, 3, 0.85);
        let e = random_emission::<N, A, T>(77);
        let solver = HmmControlSolver::<N, A, T>::new();
        let s_full = solver.solve(&p_full, &e);
        let s_leak = solver.solve(&p_leak, &e);
        let mut strictly_smaller = 0usize;
        for t in 0..T {
            for i in 0..N {
                assert!(s_leak.beta[t][i] <= s_full.beta[t][i] + 1e-6);
                if s_full.beta[t][i] > 1e-3 && s_leak.beta[t][i] < s_full.beta[t][i] - 1e-6 {
                    strictly_smaller += 1;
                }
            }
        }
        assert!(strictly_smaller > 0, "leak must strictly lower β somewhere");
    }

    #[test]
    fn zero_emission_short_circuit_matches_dense_path() {
        // The all-zero-emission short-circuit must agree with the general
        // path: β = 0, policy = tie-break artifact 0.
        const N: usize = 5;
        const A: usize = 2;
        const T: usize = 3;
        let mut p = [[[0.0f32; N]; A]; N];
        for i in 0..N {
            p[i][0][(i + 1) % N] = 1.0;
            p[i][1][(i + 4) % N] = 1.0;
        }
        let mut e0 = [[0.7f32; A]; N];
        e0[2] = [0.0; A]; // one dead-emission state
        let e = invariant_emission::<N, A, T>(&e0);
        let sol = HmmControlSolver::<N, A, T>::new().solve(&p, &e);
        for t in 0..T {
            assert_eq!(sol.beta[t][2], 0.0);
            assert_eq!(sol.policy[t][2], 0);
        }
    }

    #[test]
    fn survives_partial_observability_lift() {
        // Observation classes merge hidden states; the class-conditional
        // (lifted) kernel is the policy-visible transition law. The paper's
        // content: linearity in π survives the restriction to class-
        // conditional policies → the lifted solve is still an exact vertex
        // optimum of ITS problem, and deterministic. The lifted problem is
        // a coarser control problem — its optimum can sit strictly below
        // every hidden member's value (information loss), so the assertion
        // is a closed-form hand solve of the lifted tables, not an
        // envelope claim.
        const N: usize = 4; // hidden: 0,2 → class 0; 1,3 → class 1
        const C: usize = 2;
        const A: usize = 2;
        const T: usize = 4;
        // Hidden kernel: each state moves CW with a=0, CCW with a=1.
        let mut p = [[[0.0f32; N]; A]; N];
        for i in 0..N {
            p[i][0][(i + 1) % N] = 1.0;
            p[i][1][(i + 3) % N] = 1.0;
        }
        // Emission: states 0, 1, 3 good (0.9), state 2 bad (0.2). So
        // class 0 = {0, 2} → average 0.55; class 1 = {1, 3} → 0.9.
        let mut e0 = [[0.2f32; A]; N];
        e0[0] = [0.9; A];
        e0[1] = [0.9; A];
        e0[3] = [0.9; A];
        // Lift: class transition = 0.5·(member1 row) + 0.5·(member2 row);
        // class emission = 0.5·(member1) + 0.5·(member2).
        let members: [[usize; 2]; C] = [[0, 2], [1, 3]];
        let mut pl = [[[0.0f32; C]; A]; C];
        let mut el = [[0.0f32; A]; C];
        for (c, mem) in members.iter().enumerate() {
            for k in 0..A {
                for (frac, &m) in [0.5f32, 0.5].iter().zip(mem.iter()) {
                    el[c][k] += frac * e0[m][k];
                    for (c2, mem2) in members.iter().enumerate() {
                        for &m2 in mem2 {
                            pl[c][k][c2] += frac * p[m][k][m2];
                        }
                    }
                }
            }
        }
        let lifted =
            HmmControlSolver::<C, A, T>::new().solve(&pl, &invariant_emission::<C, A, T>(&el));

        // Hand solve of the lifted tables: every action from class0 lands
        // in class1 (0.5/0.5) and vice versa, so both actions tie at every
        // step (tie-break → index 0) and the messages alternate:
        //   β₄ = [0.55, 0.9]  β₃ = [0.55·0.9, 0.9·0.55] = [0.495, 0.495]
        //   β₂ = [0.27225, 0.4455]
        //   β₁ = [0.245025, 0.245025]
        let cases: [(usize, usize, f32); 6] = [
            (3, 0, 0.55),
            (3, 1, 0.9),
            (2, 0, 0.495),
            (2, 1, 0.495),
            (1, 1, 0.4455),
            (0, 0, 0.245_025),
        ];
        for (t, c, expected) in cases {
            assert!(
                (lifted.beta[t][c] - expected).abs() < 1e-6,
                "lifted β[{t}][{c}] = {} vs hand {expected}",
                lifted.beta[t][c]
            );
        }
        // The theorem's content at this scale: the class-conditional policy
        // is a deterministic vertex choice (here: all ties → index 0).
        for t in 0..T {
            for c in 0..C {
                assert_eq!(lifted.policy[t][c], 0);
            }
        }
    }

    // ── T1.2 — input validation ─────────────────────────────────────────

    #[test]
    fn validate_tables_rejects_bad_input() {
        const N: usize = 2;
        const A: usize = 2;
        const T: usize = 1;
        let mut p = [[[0.0f32; N]; A]; N];
        p[0][0][0] = 0.5;
        p[0][0][1] = 0.5;
        p[0][1][0] = 1.0;
        p[1][0][1] = 1.0;
        p[1][1][1] = 1.0;
        let e0 = [[0.5f32; A]; N];
        let e = invariant_emission::<N, A, T>(&e0);
        assert_eq!(validate_tables(&p, &e), Ok(()));

        // Over-stochastic row.
        let mut p_bad = p;
        p_bad[0][0][0] = 0.9;
        assert_eq!(
            validate_tables(&p_bad, &e),
            Err(HmmInputError::TransitionRowOverStochastic)
        );

        // Negative transition element (also catches NaN via the same arm).
        let mut p_neg = p;
        p_neg[1][1][0] = -0.1;
        p_neg[1][1][1] = 1.1;
        assert_eq!(
            validate_tables(&p_neg, &e),
            Err(HmmInputError::TransitionNegativeOrNan)
        );

        // Emission out of [0, 1].
        let mut e_bad0 = e0;
        e_bad0[0][0] = 1.5;
        assert_eq!(
            validate_tables(&p, &invariant_emission::<N, A, T>(&e_bad0)),
            Err(HmmInputError::EmissionNotAProbability)
        );

        // NaN emission.
        let mut e_nan0 = e0;
        e_nan0[1][1] = f32::NAN;
        assert_eq!(
            validate_tables(&p, &invariant_emission::<N, A, T>(&e_nan0)),
            Err(HmmInputError::EmissionNotAProbability)
        );
    }
}
