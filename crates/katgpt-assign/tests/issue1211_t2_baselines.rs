//! Issue 1211 T2 — the GOAT gate rows vs the OUTSIDE baselines (the
//! pre-registered T2 deliverable; T1's incumbent gate rows live in
//! `issue1211_descent.rs`).
//!
//! Arms on identical instances, judged on ONE yardstick — the TRUE folded
//! objective via [`katgpt_assign::eval_assignment`] (movement measured
//! against the authored `initial`):
//!
//! - `incumbent` — the single search with the WHOLE budget (K·B sweeps).
//! - `descent`   — K = 4 chains at B each, lowest pick. two_basin points
//!   run the `randomized_starts` posture (the T1 challenger); the
//!   movement-anchored consumer family runs the seed-only posture per the
//!   documented caveat (a scattered start shifts MinimizeMovement's
//!   baseline — the flag's own doc forbids arming it on start-anchored
//!   landscapes). Because the outcome's fold of a randomized chain was
//!   measured against the SCATTERED start, the gate rows here re-judge
//!   every descent answer with `eval_assignment` — the caveat-proof form.
//! - `ffd`       — first-fit-decreasing construction (the Plan-620 named
//!   outside baseline; Bench 924: local dominates FFD 3–47×).
//! - `greedy`    — least-loaded construction (balance-first; the second
//!   outside baseline — bounds what a 10-line constructive heuristic
//!   reaches without search).
//!
//! Gate rows (per family point, never pooled):
//! 1. descent(true fold) ≤ incumbent — T1's equal-total-budget row,
//!    re-pinned on the true yardstick.
//! 2. descent(true fold) ≤ ffd — the outside-baseline row.
//! 3. descent(true fold) ≤ greedy — the second outside-baseline row,
//!    SCOPED to the consumer class (measured, below).
//! 4. baseline-fitted LEVEL never emits above its level — `stop_level` =
//!    the FFD fold, seed-only chains, abstain armed; a LevelReached
//!    outcome is always at/below the level (the never-wrong row
//!    composed with an outside-baseline calibration); an Abstained or
//!    natural ChainsExhausted outcome is the honest fall-through.
//!
//! **The measured T2 finding (the honest negative, pinned in
//! `two_basin_randomized_losses_to_greedy_at_scale_are_pinned`):** the
//! unit-demand two-basin family is CONSTRUCTIVE-CLOSED — the balance
//! greedy IS the optimum (even spread), so no search lane can beat it
//! there: descent == greedy == optimum at n=24, and the randomized
//! posture LOSES at n ≥ 48 (243 vs 126 at 48/8) — its chains stop early
//! on the SCATTERED yardstick (the documented `randomized_starts`
//! caveat, measured on the true fold) and carry residual imbalance the
//! greedy never has. The greedy row is therefore asserted only on the
//! consumer family, where initial-placement knowledge exists and the
//! lane's value proposition lives; on the synthetic family the row is
//! printed + disclosed, and the loss is PINNED so a mechanism change
//! that flips it is forced to re-read this record.
//!
//! Wall time and sweeps consumed are PRINTED records only — never
//! asserted (the deterministic fold order is the assertion axis; a
//! quoted latency number carries its own box state in the bench record).

#![cfg(feature = "descent")]

use katgpt_assign::descent::{DescentConfig, DescentOutcome, DescentStopped, solve_descent};
use katgpt_assign::fixtures::{
    first_fit_decreasing, greedy_least_loaded, shard_topology, two_basin,
};
use katgpt_assign::types::{Limits, Seed};
use katgpt_assign::{eval_assignment, solve};

/// K for every descent arm (the T1 challenger posture).
const K: usize = 4;
/// Per-chain sweep budget B; the incumbent arm gets K·B (equal total
/// budget on the sweeps axis).
const PER_CHAIN_SWEEPS: u64 = 40;
const PER_CHAIN_SIDEWAYS: u64 = 20_000;

fn per_chain_limits() -> Limits {
    Limits {
        max_sweeps: PER_CHAIN_SWEEPS,
        sideways_budget: PER_CHAIN_SIDEWAYS,
        ..Limits::default()
    }
}

fn incumbent_limits() -> Limits {
    Limits {
        max_sweeps: PER_CHAIN_SWEEPS * K as u64,
        sideways_budget: PER_CHAIN_SIDEWAYS * K as u64,
        ..Limits::default()
    }
}

/// The true fold of an arm's answer: eval_assignment re-judges any
/// assignment against the authored `initial` — the one yardstick every
/// row shares (a randomized chain's outcome fold was measured against
/// its scattered start and is reported only as provenance).
fn true_fold(problem: &katgpt_assign::Problem, assignment: &[u32]) -> (i64, i64) {
    let (folded, violation, _) = eval_assignment(problem, assignment).unwrap();
    (folded, violation)
}

fn true_fold_of_sol(problem: &katgpt_assign::Problem, sol: &katgpt_assign::Solution) -> (i64, i64) {
    true_fold(problem, &sol.assignment.containers)
}

/// One family point: run every arm, print the row, assert the gate rows.
/// `greedy_gate` scopes gate row 3 to the consumer class (the measured
/// finding above — the constructive-closed synthetic family is printed,
/// never asserted, against greedy).
fn family_rows(
    label: &str,
    problem: &katgpt_assign::Problem,
    randomized_descent: bool,
    greedy_gate: bool,
) {
    let seed = Seed(21);

    // incumbent: ONE search, the whole budget.
    let t = std::time::Instant::now();
    let incumbent = solve(problem, seed, &incumbent_limits()).unwrap();
    let incumbent_ms = t.elapsed().as_secs_f64() * 1e3;
    let (incumbent_fold, incumbent_viol) = true_fold_of_sol(problem, &incumbent);

    // descent: K chains at per-chain budget, lowest pick.
    let cfg = DescentConfig {
        chains: K,
        randomized_starts: randomized_descent,
        ..DescentConfig::default()
    };
    let t = std::time::Instant::now();
    let outcome = solve_descent(problem, seed, &cfg, &per_chain_limits()).unwrap();
    let descent_ms = t.elapsed().as_secs_f64() * 1e3;
    let DescentOutcome::Solved { best, stopped } = outcome else {
        panic!("disarmed abstain must answer");
    };
    let (descent_fold, descent_viol) = true_fold_of_sol(problem, &best);

    // outside baselines: constructive, evaluated on the same fold.
    let ffd_assignment = first_fit_decreasing(problem);
    let (ffd_fold, ffd_viol, _) = eval_assignment(problem, &ffd_assignment).unwrap();
    let greedy_assignment = greedy_least_loaded(problem);
    let (greedy_fold, greedy_viol, _) = eval_assignment(problem, &greedy_assignment).unwrap();

    // the baseline-fitted LEVEL arm: answer only at/below the FFD fold;
    // seed-only chains (the caveat law — a fitted level compares folds on
    // ONE yardstick, so the arm may not shift baselines mid-descent).
    let level_cfg = DescentConfig {
        chains: K,
        stop_level: Some(ffd_fold),
        abstain_on_budget: true,
        randomized_starts: false,
    };
    let level_outcome = solve_descent(problem, seed, &level_cfg, &per_chain_limits()).unwrap();
    let (level_verdict, level_fold) = match &level_outcome {
        DescentOutcome::Solved {
            best,
            stopped: DescentStopped::LevelReached,
        } => ("level", Some(best.folded_objective)),
        DescentOutcome::Solved { best, .. } => ("natural", Some(best.folded_objective)),
        DescentOutcome::Abstained { best, .. } => ("abstain", Some(best.folded_objective)),
    };

    println!(
        "[{label}] incumbent fold={incumbent_fold} viol={incumbent_viol} ({incumbent_ms:.1}ms, sweeps {}) · \
         descent fold={descent_fold} viol={descent_viol} ({descent_ms:.1}ms, {stopped:?}, sweeps {}) · \
         ffd fold={ffd_fold} viol={ffd_viol} · greedy fold={greedy_fold} viol={greedy_viol} · \
         level[{level_verdict}] vs ffd {level_fold:?}",
        incumbent.sweeps, best.sweeps,
    );

    // Gate row 1 — T1's equal-total-budget row on the true yardstick.
    assert!(
        descent_fold <= incumbent_fold,
        "[{label}] descent true fold {descent_fold} > incumbent {incumbent_fold} at equal total budget"
    );
    // Gate row 2 — the outside baseline (FFD).
    assert!(
        descent_fold <= ffd_fold,
        "[{label}] descent true fold {descent_fold} > FFD {ffd_fold}"
    );
    // Gate row 3 — the second outside baseline (balance-greedy), scoped
    // to the consumer class (the measured constructive-closure finding
    // keeps the synthetic family's row a printed disclosure).
    if greedy_gate {
        assert!(
            descent_fold <= greedy_fold,
            "[{label}] descent true fold {descent_fold} > greedy {greedy_fold}"
        );
    }
    // Feasibility never regresses below the constructive arms.
    assert!(
        descent_viol <= ffd_viol.max(greedy_viol),
        "[{label}] descent violation {descent_viol} > baselines ({ffd_viol}, {greedy_viol})"
    );
    // Gate row 4 — the baseline-fitted level never emits above its level.
    if let DescentOutcome::Solved {
        best,
        stopped: DescentStopped::LevelReached,
    } = level_outcome
    {
        assert!(
            best.folded_objective <= ffd_fold,
            "[{label}] LevelReached emitted {} above the fitted FFD level {ffd_fold}",
            best.folded_objective
        );
    }
}

/// The two-basin family (the T1 fixture): the w_move sweep + two scale
/// points. `randomized_descent = true` — the T1 challenger posture — with
/// every gate row judged on the TRUE fold (the caveat-proof form; the
/// outcome's scattered-start fold is provenance, never a gate input).
/// `greedy_gate = false` — the constructive-closure finding: the greedy
/// row is a printed disclosure here, and the scale loss is pinned in
/// `two_basin_randomized_losses_to_greedy_at_scale_are_pinned`.
#[test]
fn two_basin_family_rows_vs_outside_baselines() {
    for (n, c, w_move) in [
        (24usize, 6usize, 1i64),
        (24, 6, 3),
        (24, 6, 8),
        (48, 8, 3),
        (96, 12, 3),
    ] {
        let problem = two_basin(n, c, 60, w_move);
        family_rows(
            &format!("two_basin n={n} c={c} wm={w_move}"),
            &problem,
            true,
            false,
        );
    }
}

/// The PINNED NEGATIVE (the T2 measured finding): on the unit-demand
/// two-basin family the balance greedy IS the optimum, and the randomized
/// posture loses to it at scale — its chains stop early on the scattered
/// yardstick (the `randomized_starts` caveat, measured on the true fold)
/// and carry residual imbalance the constructive arm never has. If this
/// assertion ever flips to ≤, the caveat mechanism changed — re-read the
/// bench record before trusting either direction.
#[test]
fn two_basin_randomized_losses_to_greedy_at_scale_are_pinned() {
    for (n, c) in [(48usize, 8usize), (96, 12)] {
        let problem = two_basin(n, c, 60, 3);
        let cfg = DescentConfig {
            chains: K,
            randomized_starts: true,
            ..DescentConfig::default()
        };
        let DescentOutcome::Solved { best, .. } =
            solve_descent(&problem, Seed(21), &cfg, &per_chain_limits()).unwrap()
        else {
            panic!("disarmed abstain must answer");
        };
        let (descent_fold, _) = true_fold_of_sol(&problem, &best);
        let greedy_assignment = greedy_least_loaded(&problem);
        let (greedy_fold, _, _) = eval_assignment(&problem, &greedy_assignment).unwrap();
        println!("[pinned-negative n={n} c={c}] descent {descent_fold} > greedy {greedy_fold}");
        assert!(
            descent_fold > greedy_fold,
            "n={n} c={c}: descent {descent_fold} no longer loses to greedy {greedy_fold} — the randomized-starts caveat mechanism changed; re-read the T2 record"
        );
    }
}

/// The consumer family (riir-chain Issue-164 shard topology, Bench 924's
/// shape): 2 dims, block-skewed demands, round-robin hand initial,
/// movement weight 3. Movement-anchored ⇒ the seed-only descent posture
/// (the randomized-start caveat's own law). Bench 924 measured the BASE
/// solver dominating FFD 3–47× on this class; these rows re-pin that
/// domination for the EXTENSION — and this is where gate row 3 (greedy)
/// is asserted: initial placement carries information here, the lane's
/// value proposition lives on this class.
#[test]
fn shard_topology_rows_vs_outside_baselines() {
    let problem = shard_topology(7, 64, 8);
    family_rows("shard_topology 64x8", &problem, false, true);
}
