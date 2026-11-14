//! Issue 1211 — the descent-search policy extension: K independent chains
//! with a lowest-cost pick, abstain-on-budget, and the fitted-level
//! outcome gate. Research 670 (CRAM, arXiv:2611.17309) distilled onto the
//! shipped local-search driver — an EXTENSION, never a second solver: the
//! orchestration below calls [`crate::solve`] per chain and reads the
//! [`StopReason`] each chain reports. No solver or types surface changes;
//! the feature-off build is byte-identical to the incumbent.
//!
//! ## The three deltas (issue 1211)
//!
//! 1. **K chains + lowest-cost pick** — CRAM's multi-solution axis: run
//!    `chains` independent searches from deterministic derived seeds (the
//!    golden-ratio stride below) and keep the lowest folded objective.
//!    Chain diversity comes from the seeded tie-break trajectories (the
//!    driver's first-improvement + permutation tie-breaks diverge under
//!    different seeds); chains share `problem.initial` — a randomized-
//!    start arm would change `MinimizeMovement` semantics and is a
//!    follow-up behind its own flag, not this lane.
//! 2. **Abstain-on-budget** — a chain that stopped on a BUDGET limit
//!    ([`StopReason::MoveLimit`] / [`StopReason::EvalLimit`] /
//!    [`StopReason::SweepLimit`]) did not converge; with
//!    [`DescentConfig::abstain_on_budget`] armed the outcome is
//!    [`DescentOutcome::Abstained`] — "no answer" instead of a confident
//!    wrong (CRAM measured 0% wrong-halts; 51 budget-exhaustions in
//!    1,238,358 searches). Disarmed (the default) the lowest endpoint is
//!    returned regardless — the incumbent shape.
//! 3. **Fitted-level outcome gate** — [`DescentConfig::stop_level`] is the
//!    caller's fitted cost threshold (calibration stays caller-side — the
//!    stack's quantile/conformal doctrine). A chain AT/BELOW the level
//!    ends the descent immediately with
//!    [`DescentStopped::LevelReached`]; the outcome's claim is
//!    compute-shape (abstain/budget), never accuracy — the compile VERIFY
//!    step (the consumer's) owns correctness. A level never reached
//!    falls through to the budget read; a deliberately mis-fitted level
//!    therefore yields ABSTAINS, never wrong emits (the issue's
//!    falsification arm).
//!
//! **Determinism (G1):** the chain seeds derive from the master seed by a
//! fixed stride — same `(problem, seed, cfg, limits)` ⇒ byte-identical
//! outcome on every node. **Allocation:** per-chain the driver's own
//! buffers; this module adds O(1) beyond the K `Solution`s it returns
//! one of.
//!
//! **Default posture:** feature-off builds compile this module to
//! nothing; with the feature on but `DescentConfig::default()` (1 chain,
//! no abstain, no level) the outcome is the incumbent `solve()` result
//! byte-identically — the tests pin it.

use crate::types::{InvalidProblem, Problem, Seed, Solution, StopReason};

/// Why a descent emitted [`DescentOutcome::Abstained`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstainReason {
    /// The best endpoint's chain exhausted its move budget.
    MoveLimit,
    /// The best endpoint's chain exhausted its evaluation budget.
    EvalLimit,
    /// The best endpoint's chain exhausted its sweep budget.
    SweepLimit,
}

impl AbstainReason {
    fn from_stop(reason: StopReason) -> Option<Self> {
        match reason {
            StopReason::MoveLimit => Some(Self::MoveLimit),
            StopReason::EvalLimit => Some(Self::EvalLimit),
            StopReason::SweepLimit => Some(Self::SweepLimit),
            _ => None,
        }
    }
}

/// Why a descent stopped searching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescentStopped {
    /// A chain's folded objective reached the fitted level — the descent
    /// stopped at the first chain that crossed it.
    LevelReached,
    /// Every chain ran to its own natural or budget stop.
    ChainsExhausted,
}

/// The descent outcome — the abstention-first answer shape (decision_wire
/// law: abstention is a first-class answer, never an error).
#[derive(Debug, Clone)]
pub enum DescentOutcome {
    /// The lowest-cost endpoint (and, when the level gate fired, the
    /// chain that crossed it).
    Solved {
        best: Solution,
        stopped: DescentStopped,
    },
    /// No chain converged: the best endpoint is reported as EVIDENCE,
    /// never as an answer — the caller treats this as "no answer"
    /// (the P104 posture: unverified output is never certified).
    Abstained {
        best: Solution,
        reason: AbstainReason,
        chains_run: usize,
    },
}

/// The descent configuration (issue 1211's three deltas; `default()` is
/// the byte-identical incumbent shape).
#[derive(Debug, Clone)]
pub struct DescentConfig {
    /// K independent chains (CRAM's multi-solution axis). 1 = the
    /// incumbent single search.
    pub chains: usize,
    /// The caller's fitted cost level; the first chain whose folded
    /// objective is AT/BELOW it ends the descent. `None` = off.
    pub stop_level: Option<i64>,
    /// Emit [`DescentOutcome::Abstained`] instead of a budget-cut best.
    pub abstain_on_budget: bool,
    /// CRAM's random-start axis: chains ≥ 1 begin from a deterministic
    /// hashed scatter of the objects instead of the shared authored
    /// start (chain 0 keeps `problem.initial` — the incumbent's exact
    /// trajectory — so `chains = 1` stays byte-identical even armed).
    /// Measured finding (issue 1211's probe): seed-only chains are a
    /// NO-OP on deterministic index-order-scan landscapes — identical
    /// trajectories unless a tie-break fires — so the multi-solution
    /// axis NEEDS start diversity. ⚠ Start-anchored specs
    /// ([`Spec::MinimizeMovement`] counts moves from `initial`,
    /// [`Spec::AvoidMoving`] pins it) get their objective SHIFTED by a
    /// scattered start — arm this only for start-independent landscapes
    /// (Capacity/Balance/affinity specs), the CRAM cell-to-value class.
    pub randomized_starts: bool,
}

impl Default for DescentConfig {
    fn default() -> Self {
        Self {
            chains: 1,
            stop_level: None,
            abstain_on_budget: false,
            randomized_starts: false,
        }
    }
}

impl DescentConfig {
    /// Chain `i`'s deterministic seed: the master seed advanced by the
    /// golden-ratio stride, one stride per chain — a fixed, documented,
    /// splitmix-free derivation (`same seed ⇒ same chains ⇒
    /// byte-identical outcome`). Chain 0 rides the master seed itself, so
    /// `chains = 1` is literally the incumbent `solve()` call.
    fn chain_seed(master: Seed, i: usize) -> Seed {
        Seed(
            master
                .0
                .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64)),
        )
    }

    /// Chain `i`'s start state: the authored `initial` for chain 0; for
    /// chains ≥ 1 under `randomized_starts`, a deterministic hashed
    /// scatter (splitmix-style mix of `(seed_i, object)` → container,
    /// fixed constants, no shared state). Pure function — G1 holds.
    fn chain_start(
        &self,
        problem: &Problem,
        master: Seed,
        i: usize,
    ) -> Vec<crate::types::ContainerId> {
        if i == 0 || !self.randomized_starts {
            return problem.initial.clone();
        }
        let nc = problem.num_containers.max(1) as u64;
        let seed = Self::chain_seed(master, i).0;
        problem
            .initial
            .iter()
            .enumerate()
            .map(|(o, _)| {
                let mut z = seed ^ (o as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) % nc) as crate::types::ContainerId
            })
            .collect()
    }
}

/// Run the K-chain descent (issue 1211). Deterministic for a given
/// `(problem, seed, cfg, limits)`.
///
/// Chains run sequentially, each a full [`crate::solve`] over the SAME
/// problem and limits with its derived seed. The first chain at/below
/// `stop_level` ends the descent (`DescentStopped::LevelReached`);
/// otherwise the lowest-cost endpoint answers — or abstains, when
/// `abstain_on_budget` is armed and that endpoint's chain stopped on a
/// budget limit.
pub fn solve_descent(
    problem: &Problem,
    seed: Seed,
    cfg: &DescentConfig,
    limits: &crate::types::Limits,
) -> Result<DescentOutcome, InvalidProblem> {
    let k = cfg.chains.max(1);
    let mut best: Option<Solution> = None;
    for i in 0..k {
        let chain_seed = DescentConfig::chain_seed(seed, i);
        let outcome = if cfg.randomized_starts && i > 0 {
            let mut p = problem.clone();
            p.initial = cfg.chain_start(problem, seed, i);
            crate::solve(&p, chain_seed, limits)?
        } else {
            crate::solve(problem, chain_seed, limits)?
        };
        let solution = outcome;
        let crossed = cfg
            .stop_level
            .is_some_and(|level| solution.folded_objective <= level);
        let better = best
            .as_ref()
            .is_none_or(|b| solution.folded_objective < b.folded_objective);
        if better {
            best = Some(solution);
        }
        if crossed {
            // The fitted level is the caller's calibration; reaching it
            // ends the descent — the budget under it is deliberately not
            // spent (the compute-shape claim).
            return Ok(DescentOutcome::Solved {
                best: best.expect("just assigned"),
                stopped: DescentStopped::LevelReached,
            });
        }
    }
    let best = best.expect("chains.max(1) guarantees at least one solve");
    if cfg.abstain_on_budget
        && let Some(reason) = AbstainReason::from_stop(best.stopped_by)
    {
        return Ok(DescentOutcome::Abstained {
            best,
            reason,
            chains_run: k,
        });
    }
    Ok(DescentOutcome::Solved {
        best,
        stopped: DescentStopped::ChainsExhausted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::types::Limits;

    /// G1 pin: the default config is the incumbent `solve()` result
    /// byte-identically (chain 0 rides the master seed).
    #[test]
    fn default_config_is_the_incumbent_solve() {
        let problem = fixtures::perfect_balance(7, 4, 25, 6);
        let incumbent = crate::solve(&problem, Seed(7), &Limits::default()).unwrap();
        let outcome = solve_descent(
            &problem,
            Seed(7),
            &DescentConfig::default(),
            &Limits::default(),
        )
        .unwrap();
        let DescentOutcome::Solved { best, stopped } = outcome else {
            panic!("default config must answer");
        };
        assert_eq!(stopped, DescentStopped::ChainsExhausted);
        assert_eq!(
            best.assignment.to_bytes(),
            incumbent.assignment.to_bytes(),
            "1-chain descent must be the incumbent byte-identically"
        );
        assert_eq!(best.folded_objective, incumbent.folded_objective);
    }

    /// Determinism: same inputs ⇒ byte-identical outcome (the G1 law at
    /// the orchestration level, K > 1).
    #[test]
    fn descent_is_deterministic_across_chains() {
        let problem = fixtures::perfect_balance(9, 5, 25, 6);
        let cfg = DescentConfig {
            chains: 4,
            ..DescentConfig::default()
        };
        let a = solve_descent(&problem, Seed(11), &cfg, &Limits::default()).unwrap();
        let b = solve_descent(&problem, Seed(11), &cfg, &Limits::default()).unwrap();
        let DescentOutcome::Solved { best: ba, .. } = a else {
            panic!("fixture must answer");
        };
        let DescentOutcome::Solved { best: bb, .. } = b else {
            panic!("fixture must answer");
        };
        assert_eq!(ba.assignment.to_bytes(), bb.assignment.to_bytes());
        assert_eq!(ba.folded_objective, bb.folded_objective);
    }

    /// The level gate: a fitted level at/below the reachable objective
    /// fires `LevelReached` and returns an endpoint AT/BELOW the level —
    /// the "never wrong relative to the level" property by construction.
    #[test]
    fn level_gate_fires_only_at_or_below_the_fitted_level() {
        let problem = fixtures::perfect_balance(7, 4, 25, 6);
        let incumbent = crate::solve(&problem, Seed(3), &Limits::default()).unwrap();
        // A level the search reaches (its own objective): fires at chain 0.
        let cfg = DescentConfig {
            chains: 3,
            stop_level: Some(incumbent.folded_objective),
            ..DescentConfig::default()
        };
        let outcome = solve_descent(&problem, Seed(3), &cfg, &Limits::default()).unwrap();
        let DescentOutcome::Solved { best, stopped } = outcome else {
            panic!("a reached level must answer");
        };
        assert_eq!(stopped, DescentStopped::LevelReached);
        assert!(best.folded_objective <= incumbent.folded_objective);
        // A level NOTHING reaches: falls through to ChainsExhausted (and,
        // with abstain armed — the falsification arm — ABSTAINS, never a
        // wrong emit).
        let impossible = DescentConfig {
            chains: 2,
            stop_level: Some(i64::MIN / 2),
            abstain_on_budget: true,
            ..DescentConfig::default()
        };
        let outcome = solve_descent(&problem, Seed(3), &impossible, &Limits::default()).unwrap();
        match outcome {
            // Either the chain converged naturally (a legal answer — the
            // fixture's natural stop is NOT a budget artifact) or it
            // abstained; BOTH are never-wrong shapes. The wrong shape
            // would be Solved{LevelReached} for a level nothing reached.
            DescentOutcome::Solved { stopped, .. } => {
                assert_eq!(stopped, DescentStopped::ChainsExhausted);
                // natural convergence on this fixture: not an abstain case
            }
            DescentOutcome::Abstained { reason, .. } => {
                assert!(matches!(
                    reason,
                    AbstainReason::MoveLimit | AbstainReason::EvalLimit | AbstainReason::SweepLimit
                ));
            }
        }
    }

    /// Abstain-on-budget: a chain whose budget exhausts (tiny
    /// max_evaluations) MUST abstain when armed and MUST answer when
    /// disarmed — the same inputs, the flag is the only difference.
    #[test]
    fn abstain_on_budget_answers_only_when_disarmed() {
        let problem = fixtures::perfect_balance(12, 6, 25, 6);
        let tiny = Limits {
            max_evaluations: 40,
            sideways_budget: 0,
            ..Limits::default()
        };
        // Disarmed: the budget-cut best is returned (the incumbent shape).
        let cfg_off = DescentConfig {
            chains: 1,
            abstain_on_budget: false,
            ..DescentConfig::default()
        };
        let outcome = solve_descent(&problem, Seed(5), &cfg_off, &tiny).unwrap();
        let DescentOutcome::Solved { best, .. } = outcome else {
            panic!("disarmed abstain must answer");
        };
        assert_eq!(best.stopped_by, StopReason::EvalLimit);
        // Armed: the SAME budget-cut search abstains.
        let cfg_on = DescentConfig {
            chains: 1,
            abstain_on_budget: true,
            ..DescentConfig::default()
        };
        let outcome = solve_descent(&problem, Seed(5), &cfg_on, &tiny).unwrap();
        let DescentOutcome::Abstained {
            best: evidence,
            reason,
            chains_run,
        } = outcome
        else {
            panic!("an EvalLimit endpoint under abstain_on_budget must abstain");
        };
        assert_eq!(reason, AbstainReason::EvalLimit);
        assert_eq!(chains_run, 1);
        assert_eq!(evidence.stopped_by, StopReason::EvalLimit);
        assert_eq!(
            evidence.assignment.to_bytes(),
            best.assignment.to_bytes(),
            "the abstained evidence IS the best endpoint"
        );
    }

    /// K chains never regress below the incumbent: best-of-K ≤ every
    /// single chain (and this fixture's K > 1 improves or ties chain 0).
    #[test]
    fn best_of_k_never_loses_to_chain_zero() {
        let problem = fixtures::perfect_balance(11, 5, 25, 6);
        let single = crate::solve(&problem, Seed(9), &Limits::default()).unwrap();
        let cfg = DescentConfig {
            chains: 5,
            ..DescentConfig::default()
        };
        let outcome = solve_descent(&problem, Seed(9), &cfg, &Limits::default()).unwrap();
        let DescentOutcome::Solved { best, .. } = outcome else {
            panic!("fixture must answer");
        };
        assert!(
            best.folded_objective <= single.folded_objective,
            "best-of-K {} must not lose to the single search {}",
            best.folded_objective,
            single.folded_objective
        );
    }
}
