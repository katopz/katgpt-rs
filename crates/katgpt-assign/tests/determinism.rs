//! G1 determinism gate (Plan 620): same input + same seed ⇒ byte-identical
//! `Assignment` across runs AND across input-shape perturbations that must
//! not change the answer (spec declaration order). Also pins that a
//! different seed is *allowed* to explore differently (both stay valid).
//!
//! The mechanism this gates: no HashMap iteration anywhere in the solve
//! path (dense arrays + seeded tie permutation + unique-key unstable
//! sorts), integer-only objective arithmetic, and Max/Sum value semantics
//! that are order-free.

use katgpt_assign::fixtures;
use katgpt_assign::{Limits, Seed, solve};

fn bytes_of(problem: &katgpt_assign::Problem, seed: u64) -> Vec<u8> {
    solve(problem, Seed(seed), &Limits::default())
        .unwrap_or_else(|e| panic!("solve failed: {e}"))
        .assignment
        .to_bytes()
}

#[test]
fn same_seed_same_bytes_across_runs() {
    // Multiple shapes: balance-heavy, random multi-dim, repair.
    let cases: Vec<katgpt_assign::Problem> = vec![
        fixtures::perfect_balance(1, 5, 15, 4),
        fixtures::random_instance(2, 60, 6, 2, 30),
        fixtures::overpacked_one_bin(4, 10, 3),
    ];
    for (i, problem) in cases.iter().enumerate() {
        let a = bytes_of(problem, 7);
        let b = bytes_of(problem, 7);
        let c = bytes_of(problem, 7);
        assert_eq!(a, b, "case {i}: run 1 vs run 2 differ");
        assert_eq!(b, c, "case {i}: run 2 vs run 3 differ");
    }
}

#[test]
fn spec_declaration_order_does_not_change_the_result() {
    // The DAG value semantics (Max/Sum over i64) are order-free, and move
    // generation never iterates spec-dependent structures — permuting the
    // spec list must compile to a value-identical solve.
    let problem = fixtures::random_instance(3, 50, 5, 2, 25);
    let specs = problem.specs.clone();
    let mut rotated_p = problem.clone();
    rotated_p.specs = {
        let mut r = specs[1..].to_vec();
        r.push(specs[0].clone());
        r
    };
    assert_ne!(
        problem.specs, rotated_p.specs,
        "fixture must actually rotate"
    );
    assert_eq!(bytes_of(&problem, 11), bytes_of(&rotated_p, 11));
}

#[test]
fn different_seeds_stay_valid() {
    let problem = fixtures::random_instance(4, 40, 4, 1, 20);
    for seed in [0u64, 1, 2, 999] {
        let solution = solve(&problem, Seed(seed), &Limits::default()).unwrap();
        assert_eq!(solution.violation_root, 0, "seed {seed}: infeasible");
    }
}

#[test]
fn solver_never_worsens_the_folded_objective() {
    // The never-worse guard: final folded <= initial folded, on every
    // fixture family (including the infeasible one — repair pressure
    // must still be monotone).
    let cases: Vec<katgpt_assign::Problem> = vec![
        fixtures::random_instance(5, 80, 8, 2, 40),
        fixtures::perfect_balance(6, 4, 30, 3),
        fixtures::overpacked_one_bin(3, 7, 4),
    ];
    for (i, problem) in cases.iter().enumerate() {
        let initial = problem.initial.clone();
        let (initial_folded, _, _) =
            katgpt_assign::eval_assignment(problem, &initial).expect("initial eval");
        let solution = solve(problem, Seed(13), &Limits::default()).unwrap();
        assert!(
            solution.folded_objective <= initial_folded,
            "case {i}: folded {} > initial {}",
            solution.folded_objective,
            initial_folded,
        );
    }
}

#[test]
fn infeasible_instance_reduces_violation_monotonically() {
    // Total demand >> total capacity: no feasible assignment exists. The
    // solver must still reduce the violation root vs initial and never
    // report a fabricated "feasible".
    let mut problem = fixtures::random_instance(8, 30, 3, 1, 10);
    // Tighten every capacity limit to a quarter of its slack value.
    for spec in problem.specs.iter_mut() {
        if let katgpt_assign::Spec::Capacity { limit, .. } = spec {
            *limit = (*limit / 4).max(1);
        }
    }
    let (initial_folded, initial_violation, _) =
        katgpt_assign::eval_assignment(&problem, &problem.initial).unwrap();
    assert!(initial_violation > 0, "fixture must start infeasible");
    let solution = solve(&problem, Seed(21), &Limits::default()).unwrap();
    assert!(
        solution.folded_objective <= initial_folded,
        "infeasible case worsened: {} > {}",
        solution.folded_objective,
        initial_folded
    );
    assert!(solution.violation_root > 0, "fabricated feasibility");
}
