//! UCT child selection (paper Eq. 3).
//!
//! The classical MCTS selection rule, with one twist: the exploration
//! constant `c(t)` is piecewise-decayed from `c_0 = √2` to `c_min = 0.5`
//! over `[switch_start, switch_end]` (paper §3.2.2, Table 4).
//!
//! ```text
//! UCT(i) = Q_i + c(t) · √( ln(N_v + 1) / (N_i + ε) )
//! ```
//!
//! where `Q_i` is the child's average reward, `N_v` the parent's visit count,
//! `N_i` the child's visit count, and `ε > 0` a smoothing constant.
//!
//! # When to Use
//!
//! Called from the [`EntropyGatedScheduler`](crate::progressive_mcgs::scheduler::EntropyGatedScheduler)
//! in [`SelectMode::Uct`](crate::progressive_mcgs::scheduler::SelectMode::Uct).
//! In [`SelectMode::Elite`](crate::progressive_mcgs::scheduler::SelectMode::Elite),
//! the scheduler samples directly via
//! [`elite_sample`](crate::progressive_mcgs::scheduler::EntropyGatedScheduler::elite_sample),
//! bypassing UCT entirely.

use crate::progressive_mcgs::graph::ProgressiveMcgs;
use crate::progressive_mcgs::types::{NodeId, UCT_EPSILON};

/// Compute the time-varying exploration constant `c(t_norm)`.
///
/// Piecewise-linear decay from `c_0` at `t_norm ≤ switch_start` down to
/// `c_min` at `t_norm ≥ switch_end`.
///
/// This mirrors the [`EntropyGatedScheduler::w`](crate::progressive_mcgs::scheduler::EntropyGatedScheduler::w)
/// schedule shape — both are piecewise-linear over the same window.
#[inline]
#[must_use]
pub fn exploration_constant(
    t_norm: f32,
    c_0: f32,
    c_min: f32,
    switch_start: f32,
    switch_end: f32,
) -> f32 {
    if t_norm <= switch_start {
        c_0
    } else if t_norm >= switch_end {
        c_min
    } else {
        let span = switch_end - switch_start;
        let s = (t_norm - switch_start) / span;
        c_0 + s * (c_min - c_0)
    }
}

/// Select the child of `parent` with the highest UCT score (paper Eq. 3).
///
/// Returns `None` if `parent` has no children.
///
/// # Allocation Discipline
///
/// Zero-allocation. Iterates children via `graph.children(parent)` (a slice)
/// and tracks the argmax in registers.
///
/// # Tie-Breaking
///
/// Ties are broken by lowest `NodeId` (insertion order) for deterministic
/// replay.
#[inline]
#[must_use]
pub fn uct_select_child<N: Clone>(
    graph: &ProgressiveMcgs<N>,
    parent: NodeId,
    exploration_constant: f32,
) -> Option<NodeId> {
    let children = graph.children(parent);
    if children.is_empty() {
        return None;
    }

    let parent_visits = graph.visits(parent) as f32;
    let ln_parent = if parent_visits + 1.0 > 1.0 {
        (parent_visits + 1.0).ln()
    } else {
        0.0
    };

    let mut best_id = children[0];
    let mut best_score = f32::NEG_INFINITY;

    for &child in children {
        let child_visits = graph.visits(child) as f32;
        let q = graph.q_value(child);
        // UCT = Q + c · √(ln(N_v + 1) / (N_i + ε))
        // For unvisited children, exploration term dominates → strongly preferred.
        let exploration = exploration_constant * (ln_parent / (child_visits + UCT_EPSILON)).sqrt();
        let score = q + exploration;
        if score > best_score {
            best_score = score;
            best_id = child;
        }
    }
    Some(best_id)
}

/// Descend from `root` to a leaf by repeatedly applying [`uct_select_child`].
///
/// Stops when the current node has no children. Returns the leaf [`NodeId`].
///
/// Used by the search orchestrator in [`SelectMode::Uct`](crate::progressive_mcgs::scheduler::SelectMode::Uct).
#[inline]
#[must_use]
pub fn uct_descend_to_leaf<N: Clone>(
    graph: &ProgressiveMcgs<N>,
    root: NodeId,
    exploration_constant: f32,
) -> NodeId {
    let mut current = root;
    while let Some(child) = uct_select_child(graph, current, exploration_constant) {
        current = child;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progressive_mcgs::types::{BranchId, Reward};

    #[test]
    fn exploration_constant_at_boundaries() {
        let c = exploration_constant(0.0, 1.414, 0.5, 0.5, 0.7);
        assert!((c - 1.414).abs() < 1e-3);

        let c = exploration_constant(1.0, 1.414, 0.5, 0.5, 0.7);
        assert!((c - 0.5).abs() < 1e-3);

        let c = exploration_constant(0.6, 1.414, 0.5, 0.5, 0.7);
        // midpoint: 1.414 + 0.5·(0.5 - 1.414) = 1.414 - 0.457 = 0.957
        assert!((c - 0.957).abs() < 1e-2);
    }

    #[test]
    fn exploration_constant_monotonic() {
        let mut prev = exploration_constant(0.0, 1.414, 0.5, 0.5, 0.7);
        for i in 1..=100 {
            let t = i as f32 / 100.0;
            let c = exploration_constant(t, 1.414, 0.5, 0.5, 0.7);
            assert!(c <= prev + 1e-6, "c(t) not monotonic at t={t}");
            prev = c;
        }
    }

    #[test]
    fn uct_prefers_unvisited_children() {
        // With high exploration constant, unvisited children should be picked
        // over visited-but-mediocre ones.
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        let visited_child = g.expand_primary(root, 1, BranchId(0));
        let unvisited_child = g.expand_primary(root, 2, BranchId(0));

        // Give visited_child a mediocre reward.
        g.backprop(visited_child, Reward::Failure);

        // With c = √2, unvisited should be strongly preferred.
        let picked = uct_select_child(&g, root, 1.414).unwrap();
        assert_eq!(
            picked, unvisited_child,
            "UCT should prefer unvisited child, got {picked:?}"
        );
    }

    #[test]
    fn uct_exploits_high_q_with_low_c() {
        // With low exploration constant, the high-Q child should be picked.
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        let good_child = g.expand_primary(root, 1, BranchId(0));
        let bad_child = g.expand_primary(root, 2, BranchId(0));

        g.backprop(good_child, Reward::Breakthrough); // Q ≈ 2
        g.backprop(bad_child, Reward::Failure);       // Q = -1

        let picked = uct_select_child(&g, root, 0.1).unwrap();
        assert_eq!(picked, good_child);
    }

    #[test]
    fn uct_no_children_returns_none() {
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        assert!(uct_select_child(&g, root, 1.414).is_none());
    }

    #[test]
    fn uct_descend_to_leaf_stops_at_leaf() {
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        let c1 = g.expand_primary(root, 1, BranchId(0));
        let c2 = g.expand_primary(c1, 2, BranchId(0));
        // c2 has no children — should be the leaf.
        let leaf = uct_descend_to_leaf(&g, root, 1.414);
        assert_eq!(leaf, c2);
    }
}
