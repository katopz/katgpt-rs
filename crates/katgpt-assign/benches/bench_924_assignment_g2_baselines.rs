//! Bench 924 — katgpt-assign G2 outside baselines (Plan 620 Phase 1).
//!
//! Not a `criterion` bench (zero-dep posture): a plain `main()` that runs
//! the fixture families at target sizes and prints one table row per
//! (fixture, method) with quality (folded objective, violation root,
//! spread, movement, max-util) and wall time. The verdict lives in
//! `.benchmarks/924_assignment_g2_baselines.md` together with the box-state
//! provenance line (the repo's latency-claim rule — a number without its
//! box state is not a measurement).
//!
//! Methods compared on identical instances:
//! - `initial`  — do nothing (the repair starting point)
//! - `ffd`      — first-fit-decreasing construction (the plan's named
//!   outside baseline; workspace precedent riir-rag packer)
//! - `local`    — this solver (singles + swaps, hot ordering, default
//!   limits)
//!
//! Run: `cargo run --release -p katgpt-assign --bench bench_924_assignment_g2_baselines`

use std::time::Instant;

use katgpt_assign::fixtures;
use katgpt_assign::{Limits, Seed, eval_assignment, solve};

struct Row {
    fixture: &'static str,
    method: &'static str,
    folded: i64,
    violation: i64,
    spread: i64,
    moved: i64,
    max_util: i64,
    wall_ms: f64,
    evals: u64,
}

fn spread_of(problem: &katgpt_assign::Problem, assignment: &[u32], dim: usize) -> i64 {
    let mut load = vec![0i64; problem.num_containers];
    for o in 0..problem.num_objects {
        load[assignment[o] as usize] += problem.demands[o * problem.num_dims + dim];
    }
    load.iter().max().unwrap() - load.iter().min().unwrap()
}

fn max_util_of(problem: &katgpt_assign::Problem, assignment: &[u32], dim: usize) -> i64 {
    let mut load = vec![0i64; problem.num_containers];
    for o in 0..problem.num_objects {
        load[assignment[o] as usize] += problem.demands[o * problem.num_dims + dim];
    }
    load.into_iter().max().unwrap()
}

fn moved_of(problem: &katgpt_assign::Problem, assignment: &[u32]) -> i64 {
    assignment
        .iter()
        .zip(&problem.initial)
        .filter(|(a, i)| a != i)
        .count() as i64
}

fn evaluate(problem: &katgpt_assign::Problem, assignment: &[u32]) -> (i64, i64, i64) {
    eval_assignment(problem, assignment).expect("oracle")
}

fn run_case(rows: &mut Vec<Row>, name: &'static str, problem: &katgpt_assign::Problem, seed: u64) {
    // initial
    {
        let t = Instant::now();
        let (folded, violation, _) = evaluate(problem, &problem.initial);
        rows.push(Row {
            fixture: name,
            method: "initial",
            folded,
            violation,
            spread: spread_of(problem, &problem.initial, 0),
            moved: 0,
            max_util: max_util_of(problem, &problem.initial, 0),
            wall_ms: t.elapsed().as_secs_f64() * 1e3,
            evals: 0,
        });
    }
    // FFD
    {
        let t = Instant::now();
        let ffd = fixtures::first_fit_decreasing(problem);
        let (folded, violation, _) = evaluate(problem, &ffd);
        rows.push(Row {
            fixture: name,
            method: "ffd",
            folded,
            violation,
            spread: spread_of(problem, &ffd, 0),
            moved: moved_of(problem, &ffd),
            max_util: max_util_of(problem, &ffd, 0),
            wall_ms: t.elapsed().as_secs_f64() * 1e3,
            evals: 0,
        });
    }
    // local search
    {
        let t = Instant::now();
        let solution = solve(problem, Seed(seed), &Limits::default()).unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        rows.push(Row {
            fixture: name,
            method: "local",
            folded: solution.folded_objective,
            violation: solution.violation_root,
            spread: spread_of(problem, &solution.assignment.containers, 0),
            moved: moved_of(problem, &solution.assignment.containers),
            max_util: max_util_of(problem, &solution.assignment.containers, 0),
            wall_ms: wall,
            evals: solution.moves_evaluated,
        });
    }
}

fn run_case_with_limits(
    rows: &mut Vec<Row>,
    name: &'static str,
    problem: &katgpt_assign::Problem,
    seed: u64,
    method: &'static str,
    limits: &Limits,
) {
    let t = Instant::now();
    let solution = solve(problem, Seed(seed), limits).unwrap();
    let wall = t.elapsed().as_secs_f64() * 1e3;
    rows.push(Row {
        fixture: name,
        method,
        folded: solution.folded_objective,
        violation: solution.violation_root,
        spread: spread_of(problem, &solution.assignment.containers, 0),
        moved: moved_of(problem, &solution.assignment.containers),
        max_util: max_util_of(problem, &solution.assignment.containers, 0),
        wall_ms: wall,
        evals: solution.moves_evaluated,
    });
}

fn main() {
    println!("bench 924 — katgpt-assign G2 outside baselines (Plan 620 Phase 1 + Phase 2)");
    println!();

    let mut rows = Vec::new();

    // Perfect-balance family (known optimum: spread 0, violation 0).
    run_case(
        &mut rows,
        "balance_5c_b20",
        &fixtures::perfect_balance(7, 5, 20, 6),
        7,
    );
    run_case(
        &mut rows,
        "balance_20c_b12",
        &fixtures::perfect_balance(8, 20, 12, 4),
        8,
    );

    // Random multi-dim family at target sizes.
    run_case(
        &mut rows,
        "rand_200x10",
        &fixtures::random_instance(11, 200, 10, 2, 40),
        11,
    );
    run_case(
        &mut rows,
        "rand_1000x20",
        &fixtures::random_instance(12, 1000, 20, 2, 40),
        12,
    );
    run_case(
        &mut rows,
        "rand_5000x50",
        &fixtures::random_instance(13, 5000, 50, 2, 60),
        13,
    );
    run_case(
        &mut rows,
        "rand_20000x100",
        &fixtures::random_instance(14, 20000, 100, 2, 80),
        14,
    );
    run_case(
        &mut rows,
        "rand_20000x200_4d",
        &fixtures::random_instance(15, 20000, 200, 4, 60),
        15,
    );

    // Repair family (movement-dominated).
    run_case(
        &mut rows,
        "repair_10c_L1000_e50",
        &fixtures::overpacked_one_bin(10, 1000, 50),
        16,
    );

    // riir-chain shard topology (Issue 164 — the consumer's cluster shape:
    // hand round-robin vs FFD rebuild vs solver rebalance).
    run_case(
        &mut rows,
        "shard_256x8",
        &fixtures::shard_topology(21, 256, 8),
        21,
    );
    run_case(
        &mut rows,
        "shard_1024x16",
        &fixtures::shard_topology(22, 1024, 16),
        22,
    );

    // ── Phase 2: escape stages on the partition-closure stall family ────
    // The measured plain stall (singles+swaps cannot reach feasibility);
    // the escape stages break it (Plan 620 Phase 2 record).
    for seed in [4u64, 10, 27, 34] {
        let name = "balance_stall";
        let problem = fixtures::perfect_balance(seed, 4, 25, 6);
        run_case_with_limits(&mut rows, name, &problem, seed, "local", &Limits::default());
        run_case_with_limits(
            &mut rows,
            name,
            &problem,
            seed,
            "local+3cyc",
            &Limits {
                triple_budget: 200_000,
                ..Limits::default()
            },
        );
        run_case_with_limits(
            &mut rows,
            name,
            &problem,
            seed,
            "local+kl",
            &Limits {
                max_kl_passes: 8,
                ..Limits::default()
            },
        );
    }

    // ── Phase 2: the new spec vocabulary at a mid size ──────────────────
    {
        let mut p = fixtures::random_instance(31, 500, 20, 2, 40);
        p.specs = vec![
            katgpt_assign::Spec::capacity(0, 6_000),
            katgpt_assign::Spec::group_count((0..100).collect()),
            katgpt_assign::Spec::avoid_moving(vec![0, 50, 250]),
            katgpt_assign::Spec::AssignmentAffinities {
                affinities: (0..50)
                    .map(|o| katgpt_assign::Affinity {
                        object: o,
                        container: o % 20,
                        weight: 5,
                    })
                    .collect(),
            },
            katgpt_assign::Spec::MinimizeMovement { weight: 1 },
        ];
        run_case_with_limits(
            &mut rows,
            "phase2_specs_500x20",
            &p,
            31,
            "local",
            &Limits::default(),
        );
        run_case_with_limits(
            &mut rows,
            "phase2_specs_500x20",
            &p,
            31,
            "local+all",
            &Limits {
                triple_budget: 20_000,
                max_kl_passes: 4,
                single_sample: 8,
                ..Limits::default()
            },
        );
    }

    println!(
        "{:<24} {:<12} {:>12} {:>10} {:>8} {:>7} {:>10} {:>9} {:>10}",
        "fixture",
        "method",
        "folded",
        "violation",
        "spread",
        "moved",
        "max_util",
        "wall_ms",
        "evals"
    );
    for r in &rows {
        println!(
            "{:<24} {:<12} {:>12} {:>10} {:>8} {:>7} {:>10} {:>9.2} {:>10}",
            r.fixture,
            r.method,
            r.folded,
            r.violation,
            r.spread,
            r.moved,
            r.max_util,
            r.wall_ms,
            r.evals
        );
    }

    println!();
    println!("Box state: record scripts/bench_preflight.sh PROVENANCE beside these numbers");
    println!("in .benchmarks/924_assignment_g2_baselines.md (the latency-claim rule).");
}
