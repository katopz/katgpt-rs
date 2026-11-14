//! Issue 1211 — the descent-extension proof battery (pre-registered in the
//! issue; the gate rows it names, in order):
//!
//! 1. **Planted-optimum fixture with adversarial local minima** — a
//!    landscape with two basins: a shallow decoy basin the incumbent
//!    single search settles into from `initial`, and a deeper planted
//!    optimum a differently-seeded chain reaches. The gate row:
//!    **chains+stop ≥ incumbent accuracy at equal total move budget**
//!    (K chains × budget B vs ONE chain × budget K·B).
//! 2. **Falsification arm** — a deliberately mis-fitted stop threshold
//!    MUST yield abstains, never wrong emits; a green run that emits a
//!    wrong answer FAILS the fixture.
//! 3. **Never-wrong relative to the level** — every emitted endpoint
//!    under an armed stop level is at/below that level, by construction.
//!
//! Per-family reporting (never pooled) = separate `#[test]`s; seeded +
//! deterministic fold order = fixed seeds asserted.

#![cfg(feature = "descent")]

use katgpt_assign::descent::{DescentConfig, DescentOutcome, DescentStopped, solve_descent};
use katgpt_assign::types::{ContainerId, Limits, Problem, Seed, Spec};
use katgpt_assign::{StopReason, solve};

/// A two-basin landscape: N unit objects all start packed in container 0
/// (over capacity by exactly one object — the shallow decoy), capacity
/// permits them spread over `num_containers`; the GOAL layer is
/// MinimizeMovement (weight W_move) fighting a strong Balance pull
/// (weight W_bal). The shallow basin = "move the one overflow object,
/// leave the rest" (low movement, high imbalance); the planted optimum =
/// "spread evenly" (higher movement, near-zero imbalance). With
/// W_bal·imbalance > W_move·(spread moves − 1), the even spread is the
/// planted optimum while the one-move repair is an adversarial local
/// minimum: every single-object move from it is accepted only if it
/// improves balance enough to cover its movement cost — the decoy's
/// neighbors are all downhill or level, so first-improvement descent
/// stalls there unless the tie-break trajectory (seeded) walks the
/// plateau out.
fn two_basin(n: usize, containers: usize, w_bal: i64, w_move: i64) -> Problem {
    Problem {
        num_objects: n,
        num_containers: containers,
        num_dims: 1,
        demands: vec![1; n],
        initial: vec![0 as ContainerId; n],
        specs: vec![
            Spec::capacity(0, (n - 1) as i64), // over capacity by exactly 1
            Spec::Balance {
                dim: 0,
                weight: w_bal,
            },
            Spec::MinimizeMovement { weight: w_move },
        ],
    }
}

/// The landscape probe (kept as the record — printed, asserted softly):
/// (a) SEED-ONLY chains are a NO-OP on this landscape class (identical
/// index-order-scan trajectories; best-of-K == single at every w_move —
/// the issue's diversity premise needs STARTS, not seeds); (b) with
/// randomized starts the packed-start decoy basin (obj w·16+480, the
/// "one container absorbs the spill" repair) vs the even-spread planted
/// optimum (obj w·20) are BOTH real, and best-of-4 lands the optimum at
/// every w_move. The incumbent single search NEVER leaves the decoy.
#[test]
fn landscape_probe_prints_the_basin_shape() {
    for w_move in [1i64, 3, 8] {
        let problem = two_basin(24, 6, 60, w_move);
        let single = solve(&problem, Seed(21), &Limits::default()).unwrap();
        let seed_only = DescentConfig {
            chains: 4,
            ..DescentConfig::default()
        };
        let DescentOutcome::Solved { best: s, .. } =
            solve_descent(&problem, Seed(21), &seed_only, &Limits::default()).unwrap()
        else {
            panic!("must answer");
        };
        let randomized = DescentConfig {
            chains: 4,
            randomized_starts: true,
            ..DescentConfig::default()
        };
        let DescentOutcome::Solved { best: r, .. } =
            solve_descent(&problem, Seed(21), &randomized, &Limits::default()).unwrap()
        else {
            panic!("must answer");
        };
        println!(
            "[probe] w_move={w_move}: single obj {} · seed-only best-of-4 obj {} · randomized best-of-4 obj {} (moves {}, viol {})",
            single.folded_objective,
            s.folded_objective,
            r.folded_objective,
            r.assignment
                .containers
                .iter()
                .zip(problem.initial.iter())
                .filter(|(a, b)| a != b)
                .count(),
            r.violation_root,
        );
        // (a) the seed-only no-op finding (the honest negative).
        assert_eq!(
            s.folded_objective, single.folded_objective,
            "seed-only chains must remain a no-op on this landscape (the recorded finding)"
        );
        // (b) the planted optimum is real and start-reachable.
        assert!(
            r.folded_objective < single.folded_objective,
            "randomized starts must escape the decoy basin at w_move={w_move}"
        );
        assert_eq!(r.violation_root, 0, "the optimum is feasible");
    }
}

/// The equal-total-move-budget A/B over the two-basin family (the
/// issue's gate row — accuracy at equal budget, this test IS the family
/// row): the incumbent ONE search with the WHOLE budget (K·B) vs K
/// randomized-start chains at B each, lowest pick.
#[test]
fn chains_beat_the_single_search_at_equal_total_budget_on_the_two_basin_landscape() {
    let (n, containers, w_bal, w_move) = (24usize, 6usize, 60i64, 3i64);
    let problem = two_basin(n, containers, w_bal, w_move);
    let per_chain = Limits {
        max_sweeps: 40,
        sideways_budget: 20_000,
        ..Limits::default()
    };

    // Incumbent posture: ONE search with the WHOLE budget (K·B).
    let mut big = per_chain.clone();
    big.max_sweeps *= 4; // 4 chains' worth of sweeps
    let incumbent = solve(&problem, Seed(21), &big).unwrap();

    // The challenger: K = 4 randomized-start chains at the per-chain
    // budget, lowest pick.
    let cfg = DescentConfig {
        chains: 4,
        randomized_starts: true,
        ..DescentConfig::default()
    };
    let outcome = solve_descent(&problem, Seed(21), &cfg, &per_chain).unwrap();
    let DescentOutcome::Solved { best, stopped } = outcome else {
        panic!("disarmed abstain must answer");
    };
    assert_eq!(stopped, DescentStopped::ChainsExhausted);

    // The gate row: chains+stop ≥ incumbent accuracy at equal budget.
    assert!(
        best.folded_objective <= incumbent.folded_objective,
        "best-of-K {} must not lose to the single search {} at equal total budget",
        best.folded_objective,
        incumbent.folded_objective
    );
    // The basin record — the fixture is only meaningful if the arms
    // actually DIFFER: the packed-start incumbent sits in the decoy
    // basin (movement-dominated), the chains reach the planted optimum.
    println!(
        "[two-basin] incumbent obj {} · best-of-4 obj {} (viol {}) — the decoy-to-optimum gap is the multi-solution axis",
        incumbent.folded_objective, best.folded_objective, best.violation_root,
    );
    assert_eq!(best.violation_root, 0);
}

/// The falsification arm: a deliberately mis-fitted (too-tight) stop
/// threshold under abstain-on-budget yields ABSTAIN or a natural
/// convergence — NEVER a `LevelReached` emit (nothing can reach the
/// level) and never a budget-artifact answer. A green run emitting a
/// wrong-shaped answer FAILS the fixture.
#[test]
fn misfitted_stop_threshold_yields_abstains_never_wrong_emits() {
    let problem = two_basin(24, 6, 60, 1);
    let impossible_level = i64::MIN / 2; // nothing reaches this
    let cfg = DescentConfig {
        chains: 3,
        stop_level: Some(impossible_level),
        abstain_on_budget: true,
        ..DescentConfig::default()
    };
    let outcome = solve_descent(&problem, Seed(31), &cfg, &Limits::default()).unwrap();
    match outcome {
        // Legal: every chain converged naturally (not a budget artifact).
        DescentOutcome::Solved { stopped, .. } => {
            assert_eq!(
                stopped,
                DescentStopped::ChainsExhausted,
                "a LevelReached emit for an unreachable level is the WRONG shape"
            );
        }
        // Legal: at least the best chain hit its budget → abstain.
        DescentOutcome::Abstained { reason, .. } => match reason {
            katgpt_assign::descent::AbstainReason::MoveLimit
            | katgpt_assign::descent::AbstainReason::EvalLimit
            | katgpt_assign::descent::AbstainReason::SweepLimit => {}
        },
    }
}

/// Never-wrong relative to the level: under an ARMED reachable level,
/// every emitted endpoint is at/below the level — across several seeds
/// and chain counts (the emission contract, not a per-seed accident).
#[test]
fn every_emitted_endpoint_respects_the_armed_level() {
    let problem = two_basin(24, 6, 60, 1);
    // A generous level (reachable) — any emission must be ≤ it.
    let level = 40_000i64;
    for seed in [2u64, 7, 13, 41] {
        for chains in [1usize, 3, 6] {
            let cfg = DescentConfig {
                chains,
                stop_level: Some(level),
                abstain_on_budget: true,
                ..DescentConfig::default()
            };
            let outcome = solve_descent(&problem, Seed(seed), &cfg, &Limits::default()).unwrap();
            match outcome {
                DescentOutcome::Solved { best, stopped } => {
                    assert!(
                        best.folded_objective <= level,
                        "seed {seed} chains {chains}: emitted {} above the level {level}",
                        best.folded_objective
                    );
                    if stopped == DescentStopped::LevelReached {
                        // The gate fired: SOME chain crossed it.
                        assert!(best.folded_objective <= level);
                    }
                }
                // Abstain is the other legal shape (nothing reached the
                // level and a chain hit its budget).
                DescentOutcome::Abstained { .. } => {}
            }
        }
    }
}

/// Byte-identity posture pin: feature on, default config — the outcome IS
/// the incumbent solve() (the extension is inert by default).
#[test]
fn default_config_is_byte_identical_to_the_incumbent() {
    let problem = two_basin(16, 4, 60, 1);
    let incumbent = solve(&problem, Seed(4), &Limits::default()).unwrap();
    let outcome = solve_descent(
        &problem,
        Seed(4),
        &DescentConfig::default(),
        &Limits::default(),
    )
    .unwrap();
    let DescentOutcome::Solved { best, .. } = outcome else {
        panic!("default config must answer");
    };
    assert_eq!(best.assignment.to_bytes(), incumbent.assignment.to_bytes());
    // And the incumbent's own stop surface is untouched.
    assert!(matches!(
        incumbent.stopped_by,
        StopReason::NoImprovingMove | StopReason::MoveLimit | StopReason::EvalLimit
    ));
}
