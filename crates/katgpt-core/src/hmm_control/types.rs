//! HMM homeostatic control types — solution + input contract (Plan 590 /
//! Research 543; paper arXiv:2609.07508, Moreno-Bote 2026).
//!
//! Leaf-clean: plain arrays + scalars. No game/chain/shard/emotion vocabulary.

use core::fmt;

/// The exact homeostatic-control solution (paper Eqs. 14-15).
///
/// - `beta[t][x] = P(y_{t:T} = y_d | x_t = x)` under the optimal policy —
///   a **probability**, bounded `[0, 1]` by construction (products and
///   convex combinations of probabilities). `beta[0]` is the step-1
///   message (the one a caller reads for the current state);
///   `beta[T-1]` the terminal step. Long horizons with small emissions
///   underflow to `0.0` — semantically correct ("no hope from this
///   state"), not a numerical defect.
/// - `policy[t][x] = a*_t(x)` — the strictly deterministic optimal action
///   (the paper's linearity-in-π theorem: the objective is linear in each
///   π_t over a simplex, so the optimum is at a vertex). Ties break to
///   the LOWEST action index (documented determinism contract). States
///   with `beta[t][x] == 0` have no meaningful action — the stored index
///   is the tie-break artifact; callers gate on `beta` when it matters.
///
/// Storage is `T` arrays (not `T+1`): the recursion has exactly `T`
/// decision steps, each with one message + one action.
#[derive(Clone, Debug)]
pub struct HmmSolution<const N: usize, const A: usize, const T: usize> {
    pub beta: [[f32; N]; T],
    pub policy: [[u16; N]; T],
}

impl<const N: usize, const A: usize, const T: usize> HmmSolution<N, A, T> {
    /// The optimal action at step `t` (0-indexed: `0` = the first decision
    /// step, `T-1` = the terminal step) from state `x`. Always defined —
    /// the solver is total; see the tie-break contract on [`Self::policy`].
    #[inline]
    pub fn optimal_action(&self, t: usize, x: usize) -> u16 {
        self.policy[t][x]
    }
}

/// Input-contract violation from [`validate_tables`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HmmInputError {
    /// A transition kernel element is negative or NaN.
    TransitionNegativeOrNan,
    /// A transition row sums to more than `1 + eps` (not (sub)stochastic).
    /// Sub-stochastic rows are VALID (mass leak = unmodelled termination —
    /// it lowers β, which is the honest semantics).
    TransitionRowOverStochastic,
    /// An emission element is negative, NaN, or `> 1`.
    EmissionNotAProbability,
}

impl fmt::Display for HmmInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            HmmInputError::TransitionNegativeOrNan => {
                "transition kernel has a negative or NaN element"
            }
            HmmInputError::TransitionRowOverStochastic => {
                "transition row sums to more than 1 + eps (not (sub)stochastic)"
            }
            HmmInputError::EmissionNotAProbability => {
                "emission element is not a probability in [0, 1]"
            }
        };
        f.write_str(msg)
    }
}

impl std::error::Error for HmmInputError {}

/// Emission semantics note (differs from a distribution over actions):
/// `e[t][x][a] = P(y_t = y_d | x_t = x, a_t = a)` is an independent
/// probability PER action — rows do NOT sum to 1 and are not normalized.
/// Two actions can both carry `P = 0.9`. Validation is elementwise.
///
/// The hot solve path does NOT call this (zero-overhead contract); callers
/// who want checked input call it once per frozen table.
///
/// NOTE: the `!(x >= bound)` forms are deliberate NaN discipline (the
/// `MopConfig::validate` precedent) — every comparison must reject NaN;
/// the `partial_cmp` rewrite clippy suggests would silently admit NaN
/// through the incomparable arm.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn validate_tables<const N: usize, const A: usize, const T: usize>(
    p: &[[[f32; N]; A]; N],
    e: &[[[f32; A]; N]; T],
) -> Result<(), HmmInputError> {
    for row3 in p.iter() {
        for row in row3.iter() {
            let mut total = 0.0f32;
            for &pj in row.iter() {
                // NaN discipline: `!(x >= 0)` rejects NaN, -inf, negatives.
                if !(pj >= 0.0) {
                    return Err(HmmInputError::TransitionNegativeOrNan);
                }
                total += pj;
            }
            if total > 1.0 + 1e-4 {
                return Err(HmmInputError::TransitionRowOverStochastic);
            }
        }
    }
    for step in e.iter() {
        for row in step.iter() {
            for &ek in row.iter() {
                if !(ek >= 0.0) || ek > 1.0 {
                    return Err(HmmInputError::EmissionNotAProbability);
                }
            }
        }
    }
    Ok(())
}

/// Time-invariant emission table helper: replicate one per-(x, a) emission
/// row across all `T` steps (the common case — the drive's setpoint
/// predicate does not change with the step index).
pub fn invariant_emission<const N: usize, const A: usize, const T: usize>(
    e0: &[[f32; A]; N],
) -> [[[f32; A]; N]; T] {
    let mut e = [[[0.0f32; A]; N]; T];
    for step in e.iter_mut() {
        *step = *e0;
    }
    e
}
