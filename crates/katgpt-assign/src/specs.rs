//! Spec → DAG compilation (the "MIP-free modeling layer" of Research 607).
//!
//! Compiles the Phase 1 spec vocabulary into ONE expression DAG whose root
//! is the folded objective:
//!
//! ```text
//! root = Sum[ Affine(NEVER_WORSE_GUARD)(violation_root), goal_root ]
//! violation_root = Max over (capacity spec, container) of
//!                  w_row · max(0, util(dim, b) − limit)
//! goal_root = Sum of Balance spread terms + MinimizeMovement terms
//! ```
//!
//! Leaves are shared per `(referenced dim, container)` — two specs on the
//! same dimension read the same leaf node, so a move's delta evaluation
//! touches each affected utilization exactly once regardless of how many
//! specs observe it.
//!
//! Construction is deterministic by enumeration order (referenced dims
//! sorted ascending, containers ascending, specs in declaration order) —
//! no HashMap anywhere, and the Max/Sum value semantics are order-free
//! (exact integer addition is associative and commutative), so a permuted
//! spec list compiles to a differently-shaped but value-identical DAG.
//! That invariance is pinned by the G1 determinism test.

use crate::expr::{Dag, NodeId};
use crate::types::{Affinity, NEVER_WORSE_GUARD, Problem, Spec};

/// The compiled problem: the DAG plus the solver's leaf addressing.
#[derive(Debug, Clone)]
pub struct Compiled {
    pub dag: Dag,
    /// Referenced dimension indices, sorted ascending.
    pub referenced_dims: Vec<usize>,
    /// `util_leaf[ref_dim_idx * num_containers + b]` → leaf NodeId.
    pub util_leaf: Vec<NodeId>,
    /// `moved_leaf[o]` → leaf NodeId (EMPTY when no movement spec — the
    /// leaf family is only allocated when something reads it).
    pub moved_leaf: Vec<NodeId>,
    /// Placement-leaf CSR (Phase 2): object `o`'s at-leaves are
    /// `at_nodes[at_off[o]..at_off[o+1]]`, the i-th naming container
    /// `at_items[at_off[o] + i]`. Empty when no Phase 2 spec reads them.
    pub at_off: Vec<u32>,
    pub at_items: Vec<u32>,
    pub at_nodes: Vec<NodeId>,
    /// Max-folded hard-constraint violation subroot (a `Const(0)` node
    /// when there are no capacity specs).
    pub violation_root: NodeId,
    /// Goal sum subroot (a `Const(0)` node when there are no goal specs).
    pub goal_root: NodeId,
    /// Capacity-row metadata for the `explain` surface: one entry per
    /// `(capacity spec, container)` violation row, in compilation order.
    pub capacity_rows: Vec<CapacityRowInfo>,
}

/// One capacity violation row's addressing (`explain` reads this to report
/// per-row post-move utilizations without re-walking the DAG).
#[derive(Debug, Clone, Copy)]
pub struct CapacityRowInfo {
    pub spec: usize,
    pub container: u32,
    pub util_leaf: NodeId,
    pub limit: i64,
    pub weight: i64,
}

/// Which dims does any spec read? Sorted ascending, deduped.
fn referenced_dims(problem: &Problem) -> Vec<usize> {
    let mut dims: Vec<usize> = problem
        .specs
        .iter()
        .filter_map(|s| match *s {
            Spec::Capacity { dim, .. } | Spec::Balance { dim, .. } => Some(dim),
            Spec::MinimizeMovement { .. } => None,
            Spec::GroupCount { .. }
            | Spec::AvoidMoving { .. }
            | Spec::AssignmentAffinities { .. } => None,
        })
        .collect();
    dims.sort_unstable();
    dims.dedup();
    dims
}

/// Compile a validated problem into the folded-objective DAG.
pub fn compile(problem: &Problem) -> Compiled {
    let b = problem.num_containers;
    let dims = referenced_dims(problem);
    let mut dag = Dag::builder();

    // ── Utilization leaves: one per (referenced dim, container), shared.
    // Pushed first (lowest node ids) in a fixed dense order.
    let mut util_leaf = Vec::with_capacity(dims.len() * b);
    for &dim in &dims {
        let slot = (dim * b) as u32;
        for container in 0..b {
            let leaf = dag.push_leaf_util(slot + container as u32);
            util_leaf.push(leaf);
        }
    }

    // ── Moved leaves: only when a movement spec reads them (the Phase 1
    // goal AND the Phase 2 AvoidMoving hard pin share the family).
    let wants_movement = problem
        .specs
        .iter()
        .any(|s| matches!(s, Spec::MinimizeMovement { .. } | Spec::AvoidMoving { .. }));
    let mut moved_leaf = Vec::new();
    if wants_movement {
        for o in 0..problem.num_objects {
            moved_leaf.push(dag.push_leaf_moved(o as u32));
        }
    }

    // ── At (placement) leaves: every distinct (object, container) pair the
    // Phase 2 specs read, deduped + sorted for a deterministic node order
    // and the CSR layout. AssignmentAffinities contributes its entries;
    // GroupCount contributes member × every-container pairs (the spread
    // reads each member's count in each container).
    let mut at_pairs: Vec<(u32, u32)> = Vec::new();
    for spec in &problem.specs {
        match spec {
            Spec::AssignmentAffinities { affinities } => {
                for a in affinities {
                    at_pairs.push((a.object, a.container));
                }
            }
            Spec::GroupCount { members, .. } => {
                for &o in members {
                    for c in 0..b as u32 {
                        at_pairs.push((o, c));
                    }
                }
            }
            _ => {}
        }
    }
    at_pairs.sort_unstable();
    at_pairs.dedup();
    let mut at_off = vec![0u32; problem.num_objects + 1];
    let mut at_items: Vec<u32> = Vec::with_capacity(at_pairs.len());
    let mut at_nodes: Vec<NodeId> = Vec::with_capacity(at_pairs.len());
    for &(o, c) in &at_pairs {
        at_off[o as usize + 1] += 1;
        at_items.push(c);
        at_nodes.push(dag.push_leaf_at(o, c));
    }
    for i in 0..problem.num_objects {
        at_off[i + 1] += at_off[i];
    }

    // ── Capacity rows → Max-folded violation root.
    // Per (spec, container): w · max(0, util − limit) — expressed as
    // Max[ Affine(1, −limit)(leaf), Const(0) ], then Affine(w, 0).
    let mut rows: Vec<NodeId> = Vec::new();
    let zero = dag.push_const(0);
    let mut capacity_rows: Vec<CapacityRowInfo> = Vec::new();
    for (s, spec) in problem.specs.iter().enumerate() {
        if let Spec::Capacity {
            dim,
            limit,
            violation_weight,
        } = *spec
        {
            let ref_idx = dims.binary_search(&dim).expect("dim is referenced");
            let base = ref_idx * b;
            for container in 0..b {
                let leaf = util_leaf[base + container];
                let shifted = dag.push_affine(1, -limit, leaf);
                let floored = dag.push_max(&[shifted, zero]);
                let row = if violation_weight == 1 {
                    floored
                } else {
                    dag.push_affine(violation_weight, 0, floored)
                };
                rows.push(row);
                capacity_rows.push(CapacityRowInfo {
                    spec: s,
                    container: container as u32,
                    util_leaf: leaf,
                    limit,
                    weight: violation_weight,
                });
            }
        }
    }
    // ── AvoidMoving rows: Σ moved[pinned] (each pinned object contributes
    // its 0/1 moved flag) — one row per spec, scaled by the violation
    // weight, folded into the same Max root (the fix-it machinery repairs
    // a displaced pin exactly like an over-capacity container).
    for spec in &problem.specs {
        if let Spec::AvoidMoving {
            objects,
            violation_weight,
        } = spec
        {
            let mut pinned: Vec<u32> = objects.clone();
            pinned.sort_unstable();
            pinned.dedup();
            let pins: Vec<NodeId> = pinned.iter().map(|&o| moved_leaf[o as usize]).collect();
            let sum = dag.push_sum(&pins);
            let row = if *violation_weight == 1 {
                sum
            } else {
                dag.push_affine(*violation_weight, 0, sum)
            };
            rows.push(row);
        }
    }
    let violation_root = if rows.is_empty() {
        dag.push_const(0)
    } else {
        dag.push_max(&rows)
    };
    // Secondary repair pressure: the SUM of all violation rows rides the
    // goal side at weight 1. When feasible it is identically zero (no
    // effect on goal semantics); when infeasible it makes plateau descent
    // MONOTONE — the Max fold alone cannot distinguish which of two
    // tied-at-worst containers to drain (measured: perfect_balance stalls
    // at a two-way tie with strict-only acceptance). Every accepted move
    // strictly decreases the sum, so cycles are impossible.
    let sum_violation_term = if rows.is_empty() {
        None
    } else {
        Some(dag.push_sum(&rows))
    };

    // ── Goal terms.
    let mut terms: Vec<NodeId> = Vec::new();
    for spec in &problem.specs {
        match spec {
            Spec::Capacity { .. } | Spec::AvoidMoving { .. } => {}
            Spec::Balance { dim, weight } => {
                // spread = max_b util − min_b util
                //        = Max[leaves] + Max[Affine(−1)(leaves)]
                let ref_idx = dims.binary_search(dim).expect("dim is referenced");
                let base = ref_idx * b;
                let leaves: Vec<NodeId> = (0..b).map(|c| util_leaf[base + c]).collect();
                let max_util = dag.push_max(&leaves);
                let neg_leaves: Vec<NodeId> =
                    leaves.iter().map(|&l| dag.push_affine(-1, 0, l)).collect();
                let neg_min = dag.push_max(&neg_leaves); // = −min(util)
                let spread = dag.push_sum(&[max_util, neg_min]);
                let term = if *weight == 1 {
                    spread
                } else {
                    dag.push_affine(*weight, 0, spread)
                };
                terms.push(term);
            }
            Spec::MinimizeMovement { weight } => {
                let term = if *weight == 1 {
                    dag.push_sum(&moved_leaf)
                } else {
                    let total = dag.push_sum(&moved_leaf);
                    dag.push_affine(*weight, 0, total)
                };
                terms.push(term);
            }
            Spec::GroupCount { members, weight } => {
                // Per-container member COUNT = Σ at(o, b) over the (deduped,
                // sorted) members; spread = max_b count − min_b count — the
                // Balance shape over counts instead of utilizations.
                let mut group: Vec<u32> = members.clone();
                group.sort_unstable();
                group.dedup();
                let mut counts: Vec<NodeId> = Vec::with_capacity(b);
                let mut neg_counts: Vec<NodeId> = Vec::with_capacity(b);
                for c in 0..b as u32 {
                    let pins: Vec<NodeId> = group
                        .iter()
                        .map(|&o| {
                            at_node_of(&at_off, &at_items, &at_nodes, o, c)
                                .expect("validated: at-pair exists")
                        })
                        .collect();
                    let cnt = dag.push_sum(&pins);
                    counts.push(cnt);
                    neg_counts.push(dag.push_affine(-1, 0, cnt));
                }
                let max_cnt = dag.push_max(&counts);
                let neg_min_cnt = dag.push_max(&neg_counts);
                let spread = dag.push_sum(&[max_cnt, neg_min_cnt]);
                let term = if *weight == 1 {
                    spread
                } else {
                    dag.push_affine(*weight, 0, spread)
                };
                terms.push(term);
            }
            Spec::AssignmentAffinities { affinities } => {
                // entry (o, b, w) contributes w while o is OUTSIDE b:
                // term = w · (1 − at(o,b)) = Affine(−w, w, at_leaf).
                for Affinity {
                    object,
                    container,
                    weight,
                } in affinities
                {
                    let leaf = at_node_of(&at_off, &at_items, &at_nodes, *object, *container)
                        .expect("validated: at-pair exists");
                    terms.push(dag.push_affine(-*weight, *weight, leaf));
                }
            }
        }
    }
    // The secondary repair term joins the goal sum (it is identically 0
    // at feasibility, so it never distorts goal semantics).
    if let Some(term) = sum_violation_term {
        terms.push(term);
    }
    let goal_root = if terms.is_empty() {
        dag.push_const(0)
    } else {
        dag.push_sum(&terms)
    };

    // ── Folded root: GUARD · violation + goal.
    let guarded = dag.push_affine(NEVER_WORSE_GUARD, 0, violation_root);
    let root = dag.push_sum(&[guarded, goal_root]);

    let dag = dag.finish(root);
    Compiled {
        dag,
        referenced_dims: dims,
        util_leaf,
        moved_leaf,
        at_off,
        at_items,
        at_nodes,
        violation_root,
        goal_root,
        capacity_rows,
    }
}

/// The at-leaf NodeId for `(object, container)`, found in the CSR (binary
/// search over the object's sorted container slice).
pub fn at_node_of(
    at_off: &[u32],
    at_items: &[u32],
    at_nodes: &[NodeId],
    object: u32,
    container: u32,
) -> Option<NodeId> {
    let s = at_off[object as usize] as usize;
    let e = at_off[object as usize + 1] as usize;
    let slice = &at_items[s..e];
    let idx = slice.binary_search(&container).ok()?;
    Some(at_nodes[s + idx])
}
