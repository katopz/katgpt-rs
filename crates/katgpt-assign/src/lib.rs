//! # katgpt-assign
//!
//! Spec-driven **constrained assignment solver** — place every object into
//! exactly one container such that constraint specs hold and goal specs are
//! optimized. Distilled from Meta's Rebalancer (OSDI'24; Research 607,
//! Plan 620) into a modelless, integer-deterministic, zero-dependency
//! primitive.
//!
//! - **Modeling**: a closed spec vocabulary (`Capacity`, `Balance`,
//!   `MinimizeMovement`, plus the Phase 2 `GroupCount`, `AvoidMoving`,
//!   `AssignmentAffinities` — each self-contained, so `Problem` never
//!   changes shape) over dense `u32` object/container indices and `i64`
//!   dimension values, compiled into ONE expression DAG of linear size
//!   (`Lookup` leaves shared per `(dim, container)` / `(object,
//!   container)` + `Sum`/`Max`/`Affine` internal nodes, per-node cached
//!   values).
//! - **Search**: strict-improvement local search (single + swap moves,
//!   first-improvement, hot-container ordering from leaf-side potentials,
//!   deterministic seeded tie-breaks, move/eval/sweep limits) plus the
//!   Phase 2 escape stages — TRIPLE_LOOP 3-cycle rotations and KL_SEARCH
//!   big-neighborhood chains (both budget-armed, default-off),
//!   SINGLE_RANDOM-style destination sampling, acceptance-policy variants
//!   (`Policy::Default`/`Strict`/`Plateau`), and equivalence-class
//!   candidate dedupe (the paper's Alg 2 object axis: same signature ⇒
//!   same delta, one representative evaluated).
//! - **Explain**: `explain_single`/`explain_swap` report WHY a move would
//!   or would not be accepted (the paper's Explorer role); `diff` shows
//!   what changed between two assignments.
//! - **Delta evaluation**: a candidate move touches a few leaves;
//!   recomputation runs bottom-up over reached nodes only. Integer sums
//!   are exact — no segment tree, no drift (Research 607 §1).
//! - **Integer determinism (G1)**: no floats anywhere in the objective
//!   path; same input + same seed ⇒ **byte-identical `Assignment` on
//!   every node** — the raw-domain requirement for anything crossing
//!   `SyncBlock → ChainConsensus` (the primary consumer, riir-chain Issue
//!   164, does exactly that).
//!
//! Paper defaults kept: broken-constraint fix-it weight 100, never-worse
//! guard 10000 (see [`types`]). Simulated annealing is a recorded negative
//! in-source and deliberately absent.
//!
//! ## Quick start
//!
//! ```
//! use katgpt_assign::fixtures;
//! use katgpt_assign::{solve, Limits, Seed};
//!
//! let problem = fixtures::perfect_balance(7, 4, 25, 6);
//! let solution = solve(&problem, Seed(7), &Limits::default()).unwrap();
//! assert_eq!(solution.violation_root, 0); // feasible
//! ```
//!
//! ## Phase 2 vocabulary (Plan 620)
//!
//! Three self-contained specs (each carries its own member/pair lists —
//! `Problem` never changes shape), two escape-stage move generators, an
//! equivalence-class candidate dedupe, acceptance policies, and the
//! explain surface. All arms default OFF / DEFAULT — the default posture
//! is byte-identical to Phase 1.
//!
//! ```
//! use katgpt_assign::fixtures;
//! use katgpt_assign::{solve, Affinity, Limits, Policy, Seed, Spec};
//!
//! // 8 objects pinned into one group; two of them must not move; object 3
//! // prefers container 2. GroupCount spreads the counts, the escape stages
//! // arm on stall.
//! let problem = fixtures::perfect_balance(7, 4, 25, 6);
//! let mut p = problem.clone();
//! p.specs.push(Spec::group_count((0..10).collect()));
//! p.specs.push(Spec::avoid_moving(vec![0, 1]));
//! p.specs.push(Spec::AssignmentAffinities {
//!     affinities: vec![Affinity { object: 3, container: 2, weight: 5 }],
//! });
//! let limits = Limits {
//!     triple_budget: 200_000, // TRIPLE_LOOP rotations at stalls
//!     max_kl_passes: 8,       // KL_SEARCH chains at stalls
//!     single_sample: 4,       // SINGLE_RANDOM-style dst draws
//!     policy: Policy::Default,
//!     ..Limits::default()
//! };
//! let solution = solve(&p, Seed(7), &limits).unwrap();
//! assert_eq!(solution.violation_root, 0);
//! ```
//!
//! ## Feature posture
//!
//! `katgpt-core` re-exports this crate as `katgpt_core::assign` behind the
//! opt-in `assignment` feature (the `katgpt-dec` → `katgpt_core::dec`
//! precedent). The feature stays opt-in until the GOAT gate
//! (G1+G2 vs outside baselines) passes AND a real consumer wires it
//! (riir-chain Issue 164 minimum) — see Plan 620's verdict protocol.

#![deny(rustdoc::broken_intra_doc_links)]

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
pub mod alloc;
pub mod expr;
pub mod fixtures;
pub mod rng;
pub mod solver;
mod specs;
pub mod types;

pub use solver::{
    Move, Solver, eval_assignment,
    explain::{CapacityRowRead, MoveReport, Rejection, diff},
    solve,
};
pub use types::{
    Affinity, Assignment, ContainerId, FIX_IT_WEIGHT, InvalidProblem, Limits, NEVER_WORSE_GUARD,
    Policy, Problem, Seed, Solution, Spec, StopReason,
};
