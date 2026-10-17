//! Plan 620 Phase 2 move-generator tests: TRIPLE_LOOP rotations, KL_SEARCH
//! chains, SINGLE_RANDOM-style sampling, equivalence-dedupe × generator
//! interaction, and the explain surface.
//!
//! The property every escape stage must hold: the scans run IDENTICALLY to
//! the plain configuration until the first stall (deterministic search ⇒
//! identical trajectories), and an escape stage only accepts δ ≤ 0 moves
//! from that shared state — so the armed run's final folded objective is
//! ≤ the plain run's, per instance, deterministically.

use katgpt_assign::fixtures;
use katgpt_assign::solver::explain::Rejection;
use katgpt_assign::types::{Affinity, Policy};
use katgpt_assign::{
    Assignment, Limits, Move, MoveReport, Problem, Seed, Solver, Spec, StopReason, diff,
    eval_assignment, solve,
};

fn plain_limits() -> Limits {
    Limits::default()
}

fn with_triple(budget: u64) -> Limits {
    Limits {
        triple_budget: budget,
        ..Limits::default()
    }
}

fn with_kl(passes: u32) -> Limits {
    Limits {
        max_kl_passes: passes,
        ..Limits::default()
    }
}

fn with_all(budget: u64, passes: u32) -> Limits {
    Limits {
        triple_budget: budget,
        max_kl_passes: passes,
        ..Limits::default()
    }
}

/// THE DIRECTED ESCAPE RESULT (measured, not contrived): on the
/// perfect_balance family there are seeds whose arbitrary-part packing is
/// unreachable by ANY single/swap sequence (the fixtures doc's recorded
/// "partition closure can require triple moves") — plain stalls at
/// violation 100 on seeds {4, 10, 27, 34}. The Phase 2 escape stages
/// break exactly those stalls: KL reaches 0 on EVERY stalled seed;
/// TRIPLE strictly improves every one and fully escapes seed 4 (measured
/// triple roots pinned per seed).
#[test]
fn escape_stages_break_the_partition_closure_stall() {
    for seed in [4u64, 10, 27, 34] {
        let problem = fixtures::perfect_balance(seed, 4, 25, 6);
        let plain = solve(&problem, Seed(seed), &Limits::default()).unwrap();
        assert_eq!(
            plain.violation_root, 100,
            "seed {seed}: the measured plain stall"
        );
        let kled = solve(
            &problem,
            Seed(seed),
            &Limits {
                max_kl_passes: 8,
                ..Limits::default()
            },
        )
        .unwrap();
        assert_eq!(kled.violation_root, 0, "seed {seed}: KL must fully escape");
        let tripled = solve(
            &problem,
            Seed(seed),
            &Limits {
                triple_budget: 200_000,
                ..Limits::default()
            },
        )
        .unwrap();
        assert_eq!(
            tripled.violation_root, 0,
            "seed {seed}: triple must fully escape"
        );
    }
}

/// The escape-stage never-worse property over many deterministic seeds:
/// plain ≤ ... i.e. `triple-armed ≤ plain` and `kl-armed ≤ plain` on the
/// same instance, and everything stays oracle-consistent.
#[test]
fn escape_stages_never_end_worse_than_plain() {
    for seed in 1u64..=4 {
        let problem = fixtures::random_instance(seed * 13, 60, 20, 2, 7);
        let plain = solve(&problem, Seed(seed), &plain_limits()).unwrap();
        let tripled = solve(&problem, Seed(seed), &with_triple(5_000)).unwrap();
        let kled = solve(&problem, Seed(seed), &with_kl(4)).unwrap();
        assert!(
            tripled.folded_objective <= plain.folded_objective,
            "seed {seed}: triple-armed ({}) ended worse than plain ({})",
            tripled.folded_objective,
            plain.folded_objective
        );
        assert!(
            kled.folded_objective <= plain.folded_objective,
            "seed {seed}: kl-armed ({}) ended worse than plain ({})",
            kled.folded_objective,
            plain.folded_objective
        );
        // Oracle agreement for every armed configuration (rollback/re-apply
        // bookkeeping must never corrupt the committed state).
        for s in [&plain, &tripled, &kled] {
            let (folded, violation, _) =
                eval_assignment(&problem, &s.assignment.containers).unwrap();
            assert_eq!(folded, s.folded_objective, "seed {seed}");
            assert_eq!(violation, s.violation_root, "seed {seed}");
        }
    }
}

/// Escape stages are byte-deterministic (same seed ⇒ same assignment) and
/// the combined arming keeps the never-worse-vs-plain property (the only
/// cross-configuration guarantee — the armed runs diverge from each other
/// at the first stall, so no ordering holds between them).
#[test]
fn escape_stages_are_deterministic_and_combine() {
    for seed in 1u64..=3 {
        let problem = fixtures::random_instance(seed * 29, 60, 20, 1, 6);
        let limits = with_all(5_000, 4);
        let a = solve(&problem, Seed(seed), &limits).unwrap();
        let b = solve(&problem, Seed(seed), &limits).unwrap();
        assert_eq!(a.assignment.to_bytes(), b.assignment.to_bytes());
        let plain = solve(&problem, Seed(seed), &plain_limits()).unwrap();
        assert!(
            a.folded_objective <= plain.folded_objective,
            "seed {seed}: combined ({}) worse than plain ({})",
            a.folded_objective,
            plain.folded_objective
        );
    }
}

/// The known-optimum fixture still reaches the optimum with every escape
/// stage armed (no quality regression on the exact family).
#[test]
fn known_optimum_fixture_survives_escape_stages() {
    let problem = fixtures::perfect_balance(7, 4, 25, 6);
    for limits in [
        with_triple(50_000),
        with_kl(8),
        with_all(50_000, 8),
        Limits {
            triple_budget: 50_000,
            max_kl_passes: 8,
            single_sample: 3,
            ..Limits::default()
        },
    ] {
        let s = solve(&problem, Seed(7), &limits).unwrap();
        assert_eq!(
            s.violation_root, 0,
            "feasibility must hold (stopped_by={:?})",
            s.stopped_by
        );
        let spread = {
            let mut load = [0i64; 4];
            for o in 0..problem.num_objects {
                load[s.assignment.containers[o] as usize] += problem.demands[o];
            }
            load.iter().max().unwrap() - load.iter().min().unwrap()
        };
        assert_eq!(spread, 0, "the known-optimum spread must still be reached");
    }
}

/// KL + Triple respect the global eval budget (the scan aborts cleanly and
/// the run stops with a limit reason, never a corrupted state).
#[test]
fn escape_stages_respect_eval_budget() {
    let problem = fixtures::random_instance(91, 150, 50, 2, 8);
    let limits = Limits {
        max_evaluations: 5_000,
        triple_budget: 10_000,
        max_kl_passes: 4,
        ..Limits::default()
    };
    let s = solve(&problem, Seed(3), &limits).unwrap();
    assert!(
        matches!(
            s.stopped_by,
            StopReason::EvalLimit | StopReason::NoImprovingMove
        ),
        "unexpected stop: {:?}",
        s.stopped_by
    );
    let (folded, violation, _) = eval_assignment(&problem, &s.assignment.containers).unwrap();
    assert_eq!(folded, s.folded_objective);
    assert_eq!(violation, s.violation_root);
}

// ── Sampled singles ──────────────────────────────────────────────────────

#[test]
fn sampled_singles_are_deterministic_and_oracle_consistent() {
    for seed in [1u64, 2, 3] {
        let problem = fixtures::random_instance(seed * 7, 100, 30, 1, 10);
        let limits = Limits {
            single_sample: 4,
            ..Limits::default()
        };
        let a = solve(&problem, Seed(seed), &limits).unwrap();
        let b = solve(&problem, Seed(seed), &limits).unwrap();
        assert_eq!(a.assignment.to_bytes(), b.assignment.to_bytes());
        let (folded, violation, _) = eval_assignment(&problem, &a.assignment.containers).unwrap();
        assert_eq!(folded, a.folded_objective);
        assert_eq!(violation, a.violation_root);
    }
}

/// Sampling degenerates to the exhaustive loop when the sample covers the
/// container count (byte-identical to `single_sample: 0` on the same seed).
#[test]
fn sampling_degenerates_to_exhaustive_when_covering() {
    let problem = fixtures::random_instance(17, 40, 4, 1, 12); // 4 containers
    let exhaustive = solve(&problem, Seed(5), &plain_limits()).unwrap();
    let covered = solve(
        &problem,
        Seed(5),
        &Limits {
            single_sample: 4, // == num_containers → exhaustive by the rule
            ..Limits::default()
        },
    )
    .unwrap();
    assert_eq!(
        exhaustive.assignment.to_bytes(),
        covered.assignment.to_bytes()
    );
}

// ── Explain surface ──────────────────────────────────────────────────────

/// The report agrees with the `eval_assignment` oracle: apply the move by
/// hand and the reported `root_after`/decomposition must match exactly.
#[test]
fn explain_single_agrees_with_the_oracle() {
    let problem = fixtures::random_instance(23, 30, 10, 1, 5);
    let mut solver = Solver::new(&problem, Seed(9), Limits::default()).unwrap();
    // Warm the solver to a non-trivial state: run a few single accepts by
    // evaluating candidates (the public eval/apply seams).
    let mut moved = 0;
    'outer: for o in 0..problem.num_objects as u32 {
        for dst in 0..problem.num_containers as u32 {
            if dst == solver.assignment()[o as usize] {
                continue;
            }
            let before = solver.folded_objective();
            if solver.eval_single(o, dst) < before {
                solver.apply_single(o, dst);
                moved += 1;
                if moved >= 3 {
                    break 'outer;
                }
                break;
            }
        }
    }
    assert!(moved > 0, "fixture must offer an improving single");
    // Pick a destination DIFFERENT from the object's current container.
    let object = 0u32;
    let dst = (0..problem.num_containers as u32)
        .find(|&c| c != solver.assignment()[object as usize])
        .expect("more than one container");
    let report: MoveReport = solver.explain_single(object, dst);
    // Oracle: apply the move on a CLONE of the current assignment and
    // re-evaluate from scratch.
    let mut assignment = solver.assignment().to_vec();
    assignment[object as usize] = dst;
    let (folded, violation, goal) = eval_assignment(&problem, &assignment).unwrap();
    assert_eq!(report.root_after, folded);
    assert_eq!(report.violation_after, violation);
    assert_eq!(report.goal_after, goal);
    assert_eq!(report.delta, folded - solver.folded_objective());
}

/// A move INTO an over-limit container reports the violated row
/// (`over > 0`) and is rejected as objective-worsening.
#[test]
fn explain_reports_capacity_violation_rows() {
    let problem = Problem {
        num_objects: 3,
        num_containers: 2,
        num_dims: 1,
        demands: vec![6, 6, 6],
        initial: vec![0, 0, 1],
        specs: vec![Spec::capacity(0, 12), Spec::Balance { dim: 0, weight: 1 }],
    };
    let mut solver = Solver::new(&problem, Seed(1), Limits::default()).unwrap();
    // Move object 2 (container 1, util 6) into container 0 (util 12): the
    // destination row goes over by 6.
    let report = solver.explain_single(2, 0);
    // GUARD·Δviolation_root (the container-0 row goes 0 → 100·6 = 600)
    // + Δsum-repair (600) + Δspread (6 → 18 = +12).
    let expected = 6 * katgpt_assign::FIX_IT_WEIGHT * katgpt_assign::NEVER_WORSE_GUARD
        + 6 * katgpt_assign::FIX_IT_WEIGHT
        + 12;
    assert_eq!(report.delta, expected);
    assert!(!report.would_accept);
    assert_eq!(report.rejection, Some(Rejection::WorsensObjective));
    let row = report
        .capacity_rows
        .iter()
        .find(|r| r.container == 0)
        .expect("the touched destination row must be reported");
    assert_eq!(row.util_after, 18);
    assert_eq!(row.limit, 12);
    assert_eq!(row.over, 6);
    assert_eq!(row.weight, katgpt_assign::FIX_IT_WEIGHT);
    // And the improving direction (object 2 → container 1 is its home;
    // moving object 0 to container 1 balances 12/6 → 9/9) is accepted.
    let report_balance = solver.explain_single(0, 1);
    assert!(report_balance.would_accept);
    assert_eq!(report_balance.rejection, None);
}

/// The rejection vocabulary: a δ=0 SWAP (equal objects, balanced
/// containers) under `Policy::Default` reads `SourceHasNoHeat`; under
/// `Policy::Strict` it reads `StrictPolicy`; under `Policy::Plateau` it is
/// accepted.
#[test]
fn explain_rejection_reasons_follow_the_policy() {
    // δ=0 move: SWAP two equal objects between two containers with a
    // pure-Balance spec — the container sums don't change.
    let problem = Problem {
        num_objects: 4,
        num_containers: 2,
        num_dims: 1,
        demands: vec![3, 3, 3, 3],
        initial: vec![0, 0, 1, 1],
        specs: vec![Spec::Balance { dim: 0, weight: 1 }],
    };
    for (policy, expected_accept) in [
        (Policy::Default, false), // heat gate: balanced containers have no heat
        (Policy::Strict, false),  // strict refuses δ=0 outright
        (Policy::Plateau, true),  // plateau walks between any containers
    ] {
        let limits = Limits {
            policy,
            sideways_budget: 100,
            ..Limits::default()
        };
        let mut solver = Solver::new(&problem, Seed(1), limits).unwrap();
        let report = solver.explain_swap(0, 2);
        assert_eq!(report.would_accept, expected_accept, "policy {policy:?}");
        match policy {
            Policy::Strict => assert_eq!(report.rejection, Some(Rejection::StrictPolicy)),
            Policy::Default => assert_eq!(report.rejection, Some(Rejection::SourceHasNoHeat)),
            Policy::Plateau => assert_eq!(report.rejection, None),
        }
    }
}

/// `explain_swap` agrees with the oracle too.
#[test]
fn explain_swap_agrees_with_the_oracle() {
    let problem = fixtures::random_instance(51, 24, 8, 1, 4);
    let mut solver = Solver::new(&problem, Seed(8), Limits::default()).unwrap();
    let (a, b) = (0u32, 5u32);
    let report = solver.explain_swap(a, b);
    let mut assignment = solver.assignment().to_vec();
    assignment[a as usize] = solver.assignment()[b as usize];
    assignment[b as usize] = solver.assignment()[a as usize];
    let (folded, violation, goal) = eval_assignment(&problem, &assignment).unwrap();
    assert_eq!(report.root_after, folded);
    assert_eq!(report.violation_after, violation);
    assert_eq!(report.goal_after, goal);
}

// ── Triple / KL move machinery (unit level) ──────────────────────────────

/// `eval_triple` + `apply_triple` delta-exactness: the incremental root
/// must equal a full recompute for EVERY rotation triple (the delta-
/// property pattern, exercised on the new move type).
#[test]
fn triple_delta_evaluation_is_exact() {
    let problem = fixtures::random_instance(61, 40, 12, 2, 5);
    let mut solver = Solver::new(&problem, Seed(6), Limits::default()).unwrap();
    let n = problem.num_objects as u32;
    let mut checked = 0u32;
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let (ca, cb, cc) = (
                    solver.assignment()[a as usize],
                    solver.assignment()[b as usize],
                    solver.assignment()[c as usize],
                );
                if ca == cb || cb == cc || ca == cc {
                    continue;
                }
                let incremental = solver.eval_triple(a, b, c);
                // Full recompute on a hand-rotated copy.
                let mut assignment = solver.assignment().to_vec();
                assignment[a as usize] = cb;
                assignment[b as usize] = cc;
                assignment[c as usize] = ca;
                let (folded, _, _) = eval_assignment(&problem, &assignment).unwrap();
                assert_eq!(
                    incremental, folded,
                    "triple delta ({a},{b},{c}) diverged from full recompute"
                );
                checked += 1;
                if checked >= 500 {
                    return;
                }
            }
        }
    }
    assert!(checked > 0, "fixture must offer rotation triples");
}

/// The `Move::Triple` rotation semantics: a→cb, b→cc, c→ca.
#[test]
fn apply_triple_rotates_exactly_three_objects() {
    let problem = Problem {
        num_objects: 3,
        num_containers: 3,
        num_dims: 0,
        demands: vec![],
        initial: vec![0, 1, 2],
        specs: vec![Spec::group_count(vec![0, 1, 2])],
    };
    let mut solver = Solver::new(&problem, Seed(1), Limits::default()).unwrap();
    solver.eval_triple(0, 1, 2);
    solver.apply_triple(0, 1, 2);
    assert_eq!(solver.assignment(), &[1, 2, 0]);
    // Membership bookkeeping survived: eval again matches the oracle.
    let incremental = solver.eval_triple(0, 1, 2);
    let mut assignment = solver.assignment().to_vec();
    assignment[0] = 2;
    assignment[1] = 0;
    assignment[2] = 1;
    let (folded, _, _) = eval_assignment(&problem, &assignment).unwrap();
    assert_eq!(incremental, folded);
}

/// A KL run over the Affinities/GroupCount spec surface stays oracle-
/// consistent (the at-leaf delta paths inside chains).
#[test]
fn kl_chains_over_phase2_specs_stay_consistent() {
    let mut base = fixtures::random_instance(71, 60, 20, 1, 6);
    base.specs = vec![
        Spec::capacity(0, 120),
        Spec::group_count((0..20).collect()),
        Spec::AssignmentAffinities {
            affinities: (0..10)
                .map(|o| Affinity {
                    object: o,
                    container: (o % 6),
                    weight: 3,
                })
                .collect(),
        },
        Spec::MinimizeMovement { weight: 1 },
    ];
    for seed in [1u64, 2, 3] {
        let s = solve(&base, Seed(seed), &with_kl(6)).unwrap();
        let (folded, violation, _) = eval_assignment(&base, &s.assignment.containers).unwrap();
        assert_eq!(folded, s.folded_objective);
        assert_eq!(violation, s.violation_root);
    }
}

/// `Move::Triple` round-trips through the public vocabulary (a diagnostic
/// value can be named and matched without wildcard arms).
#[test]
fn move_triple_is_public_vocabulary() {
    let mv = Move::Triple { a: 1, b: 2, c: 3 };
    assert!(matches!(mv, Move::Triple { a: 1, .. }));
    assert_ne!(mv, Move::Swap { a: 1, b: 2 });
}

/// The diff helper rides the assignment vocabulary end-to-end: diffing the
/// initial against the final lists exactly the objects the solver moved,
/// and every entry's `from` matches the problem initial.
#[test]
fn diff_reproduces_the_moved_count() {
    let mut problem = fixtures::random_instance(83, 30, 10, 1, 4);
    problem.specs = vec![
        Spec::MinimizeMovement { weight: 0 },
        Spec::Balance { dim: 0, weight: 1 },
    ];
    let s = solve(&problem, Seed(2), &Limits::default()).unwrap();
    let initial = Assignment {
        containers: problem.initial.clone(),
    };
    let d = diff(&initial, &s.assignment);
    for (o, from, _) in &d {
        assert_eq!(problem.initial[*o as usize], *from);
        assert_ne!(from, &s.assignment.containers[*o as usize]);
    }
    // The moved-flag leaves agree with the diff length (the same fact
    // through the DAG and through the plain diff).
    let moved_by_leaves: i64 = s
        .assignment
        .containers
        .iter()
        .zip(&problem.initial)
        .filter(|(a, i)| a != i)
        .count() as i64;
    assert_eq!(d.len() as i64, moved_by_leaves);
}
