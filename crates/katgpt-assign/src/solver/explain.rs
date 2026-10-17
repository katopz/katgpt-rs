//! The `explain` surface (Plan 620 Phase 2 — the paper's Explorer role,
//! distilled to a report-only Rust API: "the majority of engineering time
//! is solver-behavior debugging").
//!
//! Two halves:
//! - [`Solver::explain_single`] / [`Solver::explain_swap`] — evaluate a
//!   hypothetical move and report WHY it would or would not be accepted:
//!   the folded-objective decomposition after the move, the policy
//!   rejection reason (if any), the capacity rows whose containers the
//!   move touches (post-move utilization vs limit), and the movement
//!   counters.
//! - [`diff`] — the assignment-diff half: which objects moved between two
//!   assignments (the Explorer's between-runs view).
//!
//! Report-only: explain calls never mutate solver state (they borrow the
//! same delta machinery the scans use, so a following `eval_*`/scan simply
//! overwrites the scratch). Not allocation-free — they are diagnostics,
//! never the hot loop.

use crate::expr::{ExprNode, NodeId};
use crate::types::{Assignment, ContainerId, ObjectId, Policy};

use super::{Move, Solver};

/// Post-move read of one capacity violation row whose container the
/// explained move touches (`over` = `max(0, util_after − limit)`, the raw
/// row; the Max root folds `weight · over`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityRowRead {
    /// Index into `Problem::specs`.
    pub spec: usize,
    pub container: ContainerId,
    pub util_after: i64,
    pub limit: i64,
    pub over: i64,
    /// The row's violation weight (the Max root sees `weight · over`).
    pub weight: i64,
}

/// Why a non-improving move was rejected under the current policy
/// (`None` when the move would be accepted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// `delta > 0` — the move worsens the folded objective.
    WorsensObjective,
    /// `delta == 0` and `Policy::Strict` — sideways moves are off.
    StrictPolicy,
    /// `delta == 0` and the sideways budget is spent.
    SidewaysBudgetExhausted,
    /// `delta == 0` and the move exactly undoes the last sideways move
    /// (the ping-pong guard).
    ReversalGuard,
    /// `delta == 0`, `Policy::Default`, and the source container has no
    /// heat to drain (sideways shuffles between balanced containers only
    /// manufacture fresh imbalance — the measured Phase 1 gate).
    SourceHasNoHeat,
}

/// The full report for one hypothetical move.
#[derive(Debug, Clone)]
pub struct MoveReport {
    /// Folded-objective delta (`root_after − root_before`).
    pub delta: i64,
    pub root_after: i64,
    /// Max-folded hard-constraint violation after the move (0 = feasible).
    pub violation_after: i64,
    /// Goal sum after the move.
    pub goal_after: i64,
    /// Would the current policy + budgets accept this move right now?
    pub would_accept: bool,
    /// The policy-level rejection reason (when not accepted).
    pub rejection: Option<Rejection>,
    /// Capacity rows whose container the move touches, post-move.
    pub capacity_rows: Vec<CapacityRowRead>,
    /// Total moved-object count before/after (what `MinimizeMovement` and
    /// `AvoidMoving` read).
    pub moved_before: i64,
    pub moved_after: i64,
}

impl<'p> Solver<'p> {
    /// Explain a hypothetical `Single` move of `object` to `dst`.
    pub fn explain_single(&mut self, object: ObjectId, dst: ContainerId) -> MoveReport {
        let src = self.assignment[object as usize];
        let candidate = Move::Single { object, dst };
        self.build_single_deltas(object, dst);
        self.explain_common(candidate, src)
    }

    /// Explain a hypothetical `Swap` of `a` and `b` (the "source" for the
    /// sideways heat gate is `a`'s container — the swap's primary anchor).
    pub fn explain_swap(&mut self, a: ObjectId, b: ObjectId) -> MoveReport {
        let src = self.assignment[a as usize];
        let candidate = Move::Swap { a, b };
        self.build_swap_deltas(a, b);
        self.explain_common(candidate, src)
    }

    fn explain_common(&mut self, candidate: Move, src: ContainerId) -> MoveReport {
        let root_before = self.folded_objective();
        let (violation_before, goal_before) = self.objective_parts();
        let moved_before = self.moved_count();
        // The candidate evaluation (the scratch overlay carries the
        // post-move values for every reached node).
        self.compiled
            .dag
            .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        let root_after = self.last_new_root();
        let delta = root_after - root_before;
        // Ancestor subroots are always reached by any leaf change; an
        // EMPTY leaf-delta list (a zero-demand move with no movement/at
        // spec) leaves the root untouched.
        let dirty_empty = self.leaf_deltas.is_empty();
        let (violation_after, goal_after) = if dirty_empty {
            (violation_before, goal_before)
        } else {
            (
                self.overlay_value(self.compiled.violation_root),
                self.overlay_value(self.compiled.goal_root),
            )
        };
        let moved_after = self.moved_count_after();
        // Capacity rows: every row whose container one of the touched
        // util leaves names — post-move utilization read from the overlay
        // when the row's leaf was touched, else the committed value.
        let b = self.problem.num_containers;
        let mut touched: Vec<ContainerId> = Vec::new();
        for &(leaf, _) in &self.leaf_deltas {
            if let ExprNode::LeafUtil { slot } = self.compiled.dag.node(leaf) {
                touched.push((slot % b as u32) as ContainerId);
            }
        }
        touched.sort_unstable();
        touched.dedup();
        let capacity_rows: Vec<CapacityRowRead> = self
            .compiled
            .capacity_rows
            .iter()
            .filter(|row| touched.contains(&row.container))
            .map(|row| {
                let util_after = self.overlay_value(row.util_leaf);
                CapacityRowRead {
                    spec: row.spec,
                    container: row.container,
                    util_after,
                    limit: row.limit,
                    over: (util_after - row.limit).max(0),
                    weight: row.weight,
                }
            })
            .collect();
        // The policy verdict (mirror of the scan-side predicate, without
        // consuming any budget — read-only).
        let (would_accept, rejection) = if delta < 0 {
            (true, None)
        } else if delta > 0 {
            (false, Some(Rejection::WorsensObjective))
        } else {
            match self.limits.policy {
                Policy::Strict => (false, Some(Rejection::StrictPolicy)),
                _ => {
                    if self.sideways_remaining == 0 {
                        (false, Some(Rejection::SidewaysBudgetExhausted))
                    } else if self.blocked_inverse == Some(candidate) {
                        (false, Some(Rejection::ReversalGuard))
                    } else if self.limits.policy == Policy::Default
                        && self.heat_of_container(src) <= 0
                    {
                        (false, Some(Rejection::SourceHasNoHeat))
                    } else {
                        (true, None)
                    }
                }
            }
        };
        MoveReport {
            delta,
            root_after,
            violation_after,
            goal_after,
            would_accept,
            rejection,
            capacity_rows,
            moved_before,
            moved_after,
        }
    }

    /// Overlay-aware read: the node's post-evaluation value if the last
    /// delta evaluation reached it, else its committed value.
    fn overlay_value(&self, node: NodeId) -> i64 {
        if self.scratch.is_reached(node) {
            self.scratch.new_value(node)
        } else {
            self.values[node as usize]
        }
    }

    /// Total moved-object count against the committed state.
    fn moved_count(&self) -> i64 {
        self.assignment
            .iter()
            .zip(&self.problem.initial)
            .filter(|(a, i)| a != i)
            .count() as i64
    }

    /// Total moved-object count against the POST-move state (moved flags
    /// in the overlay when reached, else committed).
    fn moved_count_after(&self) -> i64 {
        if self.compiled.moved_leaf.is_empty() {
            return self.moved_count();
        }
        let mut total = 0i64;
        for o in 0..self.problem.num_objects {
            total += self.overlay_value(self.compiled.moved_leaf[o]);
        }
        total
    }
}

/// Assignment diff (the Explorer's between-runs view): every object whose
/// container differs, as `(object, from, to)` in ascending object order.
/// Both assignments must address the same problem (debug-asserted).
pub fn diff(before: &Assignment, after: &Assignment) -> Vec<(ObjectId, ContainerId, ContainerId)> {
    debug_assert_eq!(
        before.containers.len(),
        after.containers.len(),
        "diff: assignments address different problems"
    );
    before
        .containers
        .iter()
        .zip(&after.containers)
        .enumerate()
        .filter_map(|(o, (&a, &b))| (a != b).then_some((o as ObjectId, a, b)))
        .collect()
}
