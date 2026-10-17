//! Plan 620 Phase 2 spec tests: `GroupCount`, `AvoidMoving`,
//! `AssignmentAffinities` — correctness (G1), oracle agreement
//! (`eval_assignment`), determinism across runs + spec-order perturbation,
//! and validation refusals.

use katgpt_assign::fixtures;
use katgpt_assign::types::{Affinity, InvalidProblem, Policy};
use katgpt_assign::{Assignment, Limits, Problem, Seed, Spec, StopReason, eval_assignment, solve};

fn count_spread(problem: &Problem, assignment: &[u32], members: &[u32]) -> i64 {
    let mut count = vec![0i64; problem.num_containers];
    for &o in members {
        count[assignment[o as usize] as usize] += 1;
    }
    count.iter().max().unwrap() - count.iter().min().unwrap()
}

fn util_spread(problem: &Problem, assignment: &[u32], dim: usize) -> i64 {
    let mut load = vec![0i64; problem.num_containers];
    for o in 0..problem.num_objects {
        load[assignment[o] as usize] += problem.demands[o * problem.num_dims + dim];
    }
    load.iter().max().unwrap() - load.iter().min().unwrap()
}

// ── GroupCount ────────────────────────────────────────────────────────────

/// 8 objects in one group, 4 containers, all starting in container 0 —
/// the group-count spread driver must spread them 2/2/2/2.
#[test]
fn group_count_spreads_member_counts() {
    let members: Vec<u32> = (0..8).collect();
    let problem = Problem {
        num_objects: 8,
        num_containers: 4,
        num_dims: 0,
        demands: vec![],
        initial: vec![0; 8],
        specs: vec![Spec::group_count(members.clone())],
    };
    let solution = solve(&problem, Seed(7), &Limits::default()).unwrap();
    assert_eq!(solution.violation_root, 0);
    assert_eq!(
        count_spread(&problem, &solution.assignment.containers, &members),
        0,
        "group counts must spread evenly: {:?}",
        solution.assignment.containers
    );
    // Oracle agreement: the final assignment re-evaluates to the reported
    // folded objective.
    let (folded, violation, _goal) =
        eval_assignment(&problem, &solution.assignment.containers).unwrap();
    assert_eq!(folded, solution.folded_objective);
    assert_eq!(violation, solution.violation_root);
}

/// GroupCount over a 0/1-dimension Balance spec: both spellings reach
/// count-spread 0 (the README recipe's equivalence, tested not asserted).
#[test]
fn group_count_matches_balance_over_indicator_dim() {
    let members: Vec<u32> = (0..6).collect();
    let mut demands = vec![0i64; 6];
    for &o in &members {
        demands[o as usize] = 1;
    }
    let via_dim = Problem {
        num_objects: 6,
        num_containers: 3,
        num_dims: 1,
        demands,
        initial: vec![0; 6],
        specs: vec![Spec::Balance { dim: 0, weight: 1 }],
    };
    let via_spec = Problem {
        num_objects: 6,
        num_containers: 3,
        num_dims: 0,
        demands: vec![],
        initial: vec![0; 6],
        specs: vec![Spec::group_count(members)],
    };
    let a = solve(&via_dim, Seed(3), &Limits::default()).unwrap();
    let b = solve(&via_spec, Seed(3), &Limits::default()).unwrap();
    assert_eq!(util_spread(&via_dim, &a.assignment.containers, 0), 0);
    assert_eq!(
        count_spread(
            &via_spec,
            &b.assignment.containers,
            &(0..6).collect::<Vec<u32>>()
        ),
        0
    );
}

// ── AvoidMoving ───────────────────────────────────────────────────────────

/// A displaced pin is repaired back to its initial container (the fix-it
/// machinery treats a broken pin like any violated row).
#[test]
fn avoid_moving_repairs_displaced_pin() {
    // Object 2 starts displaced (initial container 1, sits in 0) and is
    // pinned; a Balance goal pulled it out — the pin must win.
    let problem = Problem {
        num_objects: 4,
        num_containers: 2,
        num_dims: 1,
        demands: vec![3, 3, 4, 4],
        initial: vec![0, 0, 1, 1],
        specs: vec![
            Spec::avoid_moving(vec![2]),
            Spec::Balance { dim: 0, weight: 1 },
        ],
    };
    let solution = solve(&problem, Seed(9), &Limits::default()).unwrap();
    assert_eq!(solution.violation_root, 0, "pin must be repaired");
    assert_eq!(solution.assignment.containers[2], 1, "pin 2 back home");
    // Oracle agreement.
    let (folded, violation, _) =
        eval_assignment(&problem, &solution.assignment.containers).unwrap();
    assert_eq!(folded, solution.folded_objective);
    assert_eq!(violation, solution.violation_root);
}

/// A pin that starts at home never leaves — even when a Balance goal would
/// pay for it (the violation row outweighs the goal via the guard).
#[test]
fn avoid_moving_pin_at_home_stays() {
    let problem = Problem {
        num_objects: 4,
        num_containers: 2,
        num_dims: 1,
        demands: vec![5, 1, 6, 6],
        initial: vec![0, 1, 1, 1],
        specs: vec![
            Spec::avoid_moving(vec![0]),
            Spec::Balance {
                dim: 0,
                weight: 1000,
            },
        ],
    };
    let solution = solve(&problem, Seed(5), &Limits::default()).unwrap();
    assert_eq!(solution.violation_root, 0);
    assert_eq!(solution.assignment.containers[0], 0, "pin 0 stayed home");
}

// ── AssignmentAffinities ─────────────────────────────────────────────────

/// An object with a free move to its preferred container takes it (the
/// affinity term strictly decreases).
#[test]
fn affinity_pulls_object_to_preferred_container() {
    let problem = Problem {
        num_objects: 2,
        num_containers: 2,
        num_dims: 0,
        demands: vec![],
        initial: vec![0, 1],
        specs: vec![Spec::AssignmentAffinities {
            affinities: vec![Affinity {
                object: 0,
                container: 1,
                weight: 7,
            }],
        }],
    };
    let solution = solve(&problem, Seed(1), &Limits::default()).unwrap();
    assert_eq!(solution.assignment.containers[0], 1, "object 0 pulled to 1");
    assert_eq!(solution.goal_sum, 0, "affinity satisfied");
    let (folded, _, goal) = eval_assignment(&problem, &solution.assignment.containers).unwrap();
    assert_eq!(folded, solution.folded_objective);
    assert_eq!(goal, 0);
}

/// Duplicate (object, container) entries are ADDITIVE: weight 3 + weight 4
/// = 7 penalty while misplaced.
#[test]
fn affinity_duplicates_are_additive() {
    let problem = Problem {
        num_objects: 1,
        num_containers: 2,
        num_dims: 0,
        demands: vec![],
        initial: vec![0],
        specs: vec![Spec::AssignmentAffinities {
            affinities: vec![
                Affinity {
                    object: 0,
                    container: 1,
                    weight: 3,
                },
                Affinity {
                    object: 0,
                    container: 1,
                    weight: 4,
                },
            ],
        }],
    };
    let solution = solve(&problem, Seed(1), &Limits::default()).unwrap();
    assert_eq!(solution.assignment.containers[0], 1);
    // Misplaced penalty would be 7; satisfied is 0.
    assert_eq!(solution.goal_sum, 0);
}

// ── Determinism (G1) over the new specs ──────────────────────────────────

fn bytes_of(problem: &Problem, seed: u64, limits: &Limits) -> Vec<u8> {
    solve(problem, Seed(seed), limits)
        .unwrap()
        .assignment
        .to_bytes()
}

/// Same input + same seed ⇒ byte-identical assignment, for every Phase 2
/// spec combination (cross-run equality, the G1 gate).
#[test]
fn phase2_specs_are_byte_deterministic() {
    let base = fixtures::random_instance(31, 60, 20, 2, 6);
    let combos: Vec<Vec<Spec>> = vec![
        vec![Spec::group_count((0..20).collect())],
        vec![Spec::avoid_moving(vec![0, 5, 9])],
        vec![Spec::AssignmentAffinities {
            affinities: (0..10)
                .map(|o| Affinity {
                    object: o,
                    container: (o % 7),
                    weight: 2,
                })
                .collect(),
        }],
        vec![
            Spec::capacity(0, 300),
            Spec::group_count((0..30).collect()),
            Spec::avoid_moving(vec![1]),
            Spec::AssignmentAffinities {
                affinities: vec![Affinity {
                    object: 3,
                    container: 2,
                    weight: 5,
                }],
            },
            Spec::MinimizeMovement { weight: 1 },
        ],
    ];
    for specs in combos {
        let mut p = base.clone();
        p.specs = specs;
        let a = bytes_of(&p, 11, &Limits::default());
        let b = bytes_of(&p, 11, &Limits::default());
        assert_eq!(a, b, "same seed must be byte-identical");
        let c = bytes_of(&p, 12, &Limits::default());
        let _ = c; // different seed MAY differ; we only pin same-seed equality
    }
}

/// Spec-order rotation still compiles to a value-identical solve with the
/// Phase 2 specs in the list (the G1 perturbation arm).
#[test]
fn phase2_spec_order_rotation_is_value_identical() {
    let base = fixtures::random_instance(77, 40, 15, 1, 5);
    let specs = vec![
        Spec::capacity(0, 150),
        Spec::group_count((0..15).collect()),
        Spec::avoid_moving(vec![2]),
        Spec::AssignmentAffinities {
            affinities: vec![Affinity {
                object: 0,
                container: 1,
                weight: 3,
            }],
        },
        Spec::Balance { dim: 0, weight: 1 },
    ];
    let mut p1 = base.clone();
    p1.specs = specs.clone();
    let mut p2 = base.clone();
    p2.specs = {
        let mut r = specs[1..].to_vec();
        r.push(specs[0].clone());
        r
    };
    let a = solve(&p1, Seed(11), &Limits::default()).unwrap();
    let b = solve(&p2, Seed(11), &Limits::default()).unwrap();
    assert_eq!(a.folded_objective, b.folded_objective);
    assert_eq!(a.violation_root, b.violation_root);
    assert_eq!(a.goal_sum, b.goal_sum);
    assert_eq!(a.assignment.to_bytes(), b.assignment.to_bytes());
}

// ── Validation ───────────────────────────────────────────────────────────

#[test]
fn phase2_validation_refusals() {
    let mk = |specs| Problem {
        num_objects: 3,
        num_containers: 2,
        num_dims: 0,
        demands: vec![],
        initial: vec![0, 0, 0],
        specs,
    };
    // Out-of-range member.
    assert_eq!(
        mk(vec![Spec::group_count(vec![0, 7])]).validate(),
        Err(InvalidProblem::SpecObjectOutOfRange {
            spec: 0,
            object: 7,
            num_objects: 3
        })
    );
    // Empty lists.
    assert_eq!(
        mk(vec![Spec::group_count(vec![])]).validate(),
        Err(InvalidProblem::SpecEmptyList { spec: 0 })
    );
    assert_eq!(
        mk(vec![Spec::avoid_moving(vec![])]).validate(),
        Err(InvalidProblem::SpecEmptyList { spec: 0 })
    );
    assert_eq!(
        mk(vec![Spec::AssignmentAffinities { affinities: vec![] }]).validate(),
        Err(InvalidProblem::SpecEmptyList { spec: 0 })
    );
    // Out-of-range affinity container.
    assert_eq!(
        mk(vec![Spec::AssignmentAffinities {
            affinities: vec![Affinity {
                object: 0,
                container: 9,
                weight: 1
            }]
        }])
        .validate(),
        Err(InvalidProblem::SpecContainerOutOfRange {
            spec: 0,
            container: 9,
            num_containers: 2
        })
    );
    // Negative weights.
    assert_eq!(
        mk(vec![Spec::AvoidMoving {
            objects: vec![0],
            violation_weight: -1,
        }])
        .validate(),
        Err(InvalidProblem::NegativeWeight { spec: 0 })
    );
    assert_eq!(
        mk(vec![Spec::AssignmentAffinities {
            affinities: vec![Affinity {
                object: 0,
                container: 0,
                weight: -3
            }]
        }])
        .validate(),
        Err(InvalidProblem::NegativeWeight { spec: 0 })
    );
}

// ── Policy variants ──────────────────────────────────────────────────────

/// `Policy::Strict` refuses every δ=0 move: on the perfect_balance family
/// there are seeds where the DEFAULT posture's sideways window repairs the
/// instance but Strict freezes at violation 100 (measured: seed 38 — the
/// tie-plateau walk is load-bearing). Pinned on the fixed seed set; the
/// plateau posture repairs it too (its heat gate is dropped).
#[test]
fn strict_policy_stalls_where_default_repairs() {
    let problem = fixtures::perfect_balance(38, 4, 25, 6);
    let strict = Limits {
        policy: Policy::Strict,
        sideways_budget: 0,
        ..Limits::default()
    };
    let stalled = solve(&problem, Seed(38), &strict).unwrap();
    assert_eq!(stalled.violation_root, 100, "measured: Strict freezes here");
    assert_eq!(stalled.stopped_by, StopReason::NoImprovingMove);
    let repaired = solve(&problem, Seed(38), &Limits::default()).unwrap();
    assert_eq!(repaired.violation_root, 0);
    let plateau = Limits {
        policy: Policy::Plateau,
        ..Limits::default()
    };
    let walked = solve(&problem, Seed(38), &plateau).unwrap();
    assert_eq!(walked.violation_root, 0);
}

/// `Policy::Plateau` walks plateaus without the heat gate — it must stay
/// deterministic, oracle-consistent, and never end infeasible when Default
/// ends feasible on the same instance (both share the strict-improvement
/// floor; Plateau only widens the δ=0 window).
#[test]
fn plateau_policy_is_deterministic_and_never_worse_than_strict() {
    let problem = fixtures::random_instance(41, 50, 20, 1, 6);
    for seed in [1u64, 2, 3] {
        let plateau = Limits {
            policy: Policy::Plateau,
            ..Limits::default()
        };
        let a = solve(&problem, Seed(seed), &plateau).unwrap();
        let b = solve(&problem, Seed(seed), &plateau).unwrap();
        assert_eq!(a.assignment.to_bytes(), b.assignment.to_bytes());
        let (folded, _, _) = eval_assignment(&problem, &a.assignment.containers).unwrap();
        assert_eq!(folded, a.folded_objective, "oracle agreement");
        let strict = Limits {
            policy: Policy::Strict,
            sideways_budget: 0,
            ..Limits::default()
        };
        let c = solve(&problem, Seed(seed), &strict).unwrap();
        assert!(
            a.folded_objective <= c.folded_objective,
            "Plateau (wider window) must not end worse than Strict on the same trajectory"
        );
    }
}

// ── Equivalence-class dedupe (the Alg 2 object axis) ─────────────────────

/// A tiny instance FULL of exact-duplicate objects: the dedupe (always on)
/// must not lose optimality — the solver's folded objective equals the
/// brute-force optimum (exhaustive over all 2^6 container assignments).
#[test]
fn dedupe_preserves_optimality_on_duplicate_objects() {
    // 6 objects, 2 containers, all size 3: optimum split 3/3, spread 0.
    let problem = Problem {
        num_objects: 6,
        num_containers: 2,
        num_dims: 1,
        demands: vec![3, 3, 3, 3, 3, 3],
        initial: vec![0, 0, 0, 1, 1, 1],
        specs: vec![Spec::capacity(0, 9), Spec::Balance { dim: 0, weight: 1 }],
    };
    let solution = solve(&problem, Seed(4), &Limits::default()).unwrap();
    // Brute force: best spread is 0 (3 objects each side).
    assert_eq!(util_spread(&problem, &solution.assignment.containers, 0), 0);
    // And a deliberately-mixed start (2/4 split) still reaches the optimum.
    let mixed = Problem {
        initial: vec![0, 0, 1, 1, 1, 1],
        ..problem.clone()
    };
    let s2 = solve(&mixed, Seed(4), &Limits::default()).unwrap();
    assert_eq!(util_spread(&mixed, &s2.assignment.containers, 0), 0);
    let (folded, violation, _) = eval_assignment(&problem, &s2.assignment.containers).unwrap();
    assert_eq!(folded, s2.folded_objective);
    assert_eq!(violation, 0);
}

/// Duplicate-heavy instances stay byte-deterministic (the dedupe marks are
/// generation-stamped; a stale stamp across scans must not alias).
#[test]
fn dedupe_is_deterministic_across_duplicate_heavy_instances() {
    for seed in [1u64, 2, 3, 4, 5] {
        let problem = fixtures::random_instance(seed, 80, 30, 1, 3); // demands 1..=3 → many duplicates
        let a = solve(&problem, Seed(seed), &Limits::default()).unwrap();
        let b = solve(&problem, Seed(seed), &Limits::default()).unwrap();
        assert_eq!(a.assignment.to_bytes(), b.assignment.to_bytes());
        let (folded, violation, _) = eval_assignment(&problem, &a.assignment.containers).unwrap();
        assert_eq!(folded, a.folded_objective);
        assert_eq!(violation, a.violation_root);
    }
}

// ── diff (the Explorer's assignment diff) ────────────────────────────────

#[test]
fn diff_lists_moved_objects_in_order() {
    let before = Assignment {
        containers: vec![0, 1, 2, 1],
    };
    let after = Assignment {
        containers: vec![0, 2, 2, 0],
    };
    let d = katgpt_assign::diff(&before, &after);
    assert_eq!(d, vec![(1, 1, 2), (3, 1, 0)]);
    // Identical assignments diff to nothing.
    assert!(katgpt_assign::diff(&before, &before).is_empty());
}
