//! Reference-set builders for the four expansion operators.
//!
//! Paper §3.2.2 / Appendix B. These are *pure functions* over the graph —
//! they return reference sets (slices of [`NodeId`]) that the consumer's
//! proposer uses as additional context when constructing a new node's payload.
//!
//! # Allocation Discipline
//!
//! All builders return `Vec<NodeId>` (not `SmallVec`, to avoid the dep).
//! They are called from `expand_*()`, which is allowed to allocate
//! (one call per decision, not per token). The hot-path `select()` and
//! `observe()` methods do not call into this module.

use crate::progressive_mcgs::graph::ProgressiveMcgs;
use crate::progressive_mcgs::types::{BranchId, NodeId};

/// Default `k` for intra-branch history (paper doesn't specify; we use 3
/// to match `MAX_REFS_PER_NODE` — the per-node reference cap).
pub const DEFAULT_INTRA_BRANCH_K: usize = 3;

/// Default `N` for cross-branch top-N (paper Table 4: `elite_topk = 3`).
pub const DEFAULT_CROSS_BRANCH_N: usize = 3;

/// Default top-per-branch count for multi-branch aggregation.
pub const DEFAULT_AGG_PER_BRANCH: usize = 1;

/// **Intra-branch evolution** — reference the last-k ancestor nodes within
/// the same branch (paper Eq. 13).
///
/// Walks the `primary_parent` chain from `node` upward, collecting up to `k`
/// ancestors. Returns them in child→parent order (most-recent first).
///
/// This is the most critical operator per paper Table 5 — removing it costs
/// −33pp medal rate on the 9-task ablation subset.
#[must_use]
pub fn intra_branch_history<N: Clone>(
    graph: &ProgressiveMcgs<N>,
    node: NodeId,
    k: usize,
) -> Vec<NodeId> {
    let mut out = Vec::with_capacity(k);
    let mut current = graph.parent(node);
    while let Some(p) = current {
        if out.len() >= k {
            break;
        }
        // Only follow primary parent within the same branch.
        if graph.branch_of(p) != graph.branch_of(node) {
            break;
        }
        out.push(p);
        current = graph.parent(p);
    }
    out
}

/// **Cross-branch reference** — reference the top-N nodes globally by Q-value
/// (paper Eq. 14).
///
/// Returns nodes from branches OTHER than `current_branch`, sorted by
/// descending Q-value, capped at `n`. Used when a branch stagnates and
/// wants to learn from stronger solutions discovered elsewhere.
///
/// # Algorithm
///
/// 1. Collect all node ids whose branch ≠ `current_branch`.
/// 2. Sort by descending Q-value.
/// 3. Take top `n`.
///
/// For large graphs, this is O(V log V). Consumers in latency-sensitive
/// contexts should cache the result between calls.
#[must_use]
pub fn cross_branch_top_n<N: Clone>(
    graph: &ProgressiveMcgs<N>,
    current_branch: BranchId,
    n: usize,
) -> Vec<NodeId> {
    if n == 0 || graph.is_empty() {
        return Vec::new();
    }
    // Collect (NodeId, Q-value) pairs from foreign branches.
    let mut candidates: Vec<(NodeId, f32)> = graph
        .node_ids()
        .filter(|&id| graph.branch_of(id) != current_branch)
        .map(|id| (id, graph.q_value(id)))
        .collect();
    // Partial sort — top-N by Q-value descending.
    // For small n and small graphs, sort_by is fine; for large graphs,
    // a min-heap of size n would be faster.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.into_iter().take(n).map(|(id, _)| id).collect()
}

/// **Multi-branch aggregation** — reference the union of top trajectories per
/// branch (paper Eq. 15).
///
/// For each branch in the graph, take the top `per_branch` nodes by Q-value.
/// Return the union. Used when global stagnation trips and the system wants
/// to spawn a fresh branch that synthesizes the best of all current branches.
///
/// # Algorithm
///
/// 1. Discover all branch ids by scanning node_ids.
/// 2. For each branch, collect its nodes and sort by Q-value.
/// 3. Take top `per_branch` from each.
/// 4. Concatenate into a single Vec.
#[must_use]
pub fn multi_branch_aggregate<N: Clone>(
    graph: &ProgressiveMcgs<N>,
    per_branch: usize,
) -> Vec<NodeId> {
    if graph.is_empty() || per_branch == 0 {
        return Vec::new();
    }
    // Discover branches.
    let max_branch = graph
        .node_ids()
        .map(|id| graph.branch_of(id))
        .filter(|b| *b != BranchId::NONE)
        .map(|b| b.idx())
        .max()
        .unwrap_or(0);
    let n_branches = max_branch + 1;

    let mut out = Vec::with_capacity(n_branches * per_branch);
    for branch_idx in 0..n_branches {
        let branch = BranchId(branch_idx as u32);
        let mut candidates: Vec<(NodeId, f32)> = graph
            .node_ids()
            .filter(|&id| graph.branch_of(id) == branch)
            .map(|id| (id, graph.q_value(id)))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.extend(candidates.into_iter().take(per_branch).map(|(id, _)| id));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progressive_mcgs::types::Reward;

    fn build_two_branch_graph() -> ProgressiveMcgs<u32> {
        // root → a1 → a2 (branch 0)
        // root → b1 → b2 (branch 1)
        // a2 has best Q-value in branch 0, b2 in branch 1.
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        let a1 = g.expand_primary(root, 1, BranchId(0));
        let a2 = g.expand_primary(a1, 2, BranchId(0));
        let b1 = g.expand_primary(root, 10, BranchId(1));
        let b2 = g.expand_primary(b1, 11, BranchId(1));

        // Drive rewards: branch 0 weaker than branch 1.
        g.backprop(a1, Reward::Neutral);
        g.backprop(a2, Reward::Progress);
        g.backprop(b1, Reward::Neutral);
        g.backprop(b2, Reward::Breakthrough);
        g
    }

    #[test]
    fn intra_branch_walks_up() {
        let g = build_two_branch_graph();
        // a2's ancestors within branch 0: a1, root.
        let hist = intra_branch_history(&g, NodeId(2), 3);
        assert_eq!(hist, vec![NodeId(1), NodeId(0)]);
    }

    #[test]
    fn intra_branch_stops_at_branch_boundary() {
        let g = build_two_branch_graph();
        // Even with k=10, we stop when branch changes.
        let hist = intra_branch_history(&g, NodeId(2), 10);
        assert_eq!(hist.len(), 2);
    }

    #[test]
    fn intra_branch_root_empty() {
        let g = build_two_branch_graph();
        let hist = intra_branch_history(&g, NodeId(0), 3);
        assert!(hist.is_empty());
    }

    #[test]
    fn cross_branch_picks_top_foreign() {
        let g = build_two_branch_graph();
        // From branch 0, the top foreign node should be b2 (Breakthrough).
        let refs = cross_branch_top_n(&g, BranchId(0), 2);
        assert!(refs.contains(&NodeId(4)), "expected b2 (NodeId 4) in refs: {refs:?}");
        // All refs should be from branch 1.
        for r in &refs {
            assert_eq!(g.branch_of(*r), BranchId(1));
        }
    }

    #[test]
    fn cross_branch_empty_n() {
        let g = build_two_branch_graph();
        assert!(cross_branch_top_n(&g, BranchId(0), 0).is_empty());
    }

    #[test]
    fn cross_branch_empty_graph() {
        let g: ProgressiveMcgs<u32> = ProgressiveMcgs::new(100, 3);
        assert!(cross_branch_top_n(&g, BranchId(0), 3).is_empty());
    }

    #[test]
    fn multi_branch_aggregate_returns_one_per_branch_default() {
        let g = build_two_branch_graph();
        let agg = multi_branch_aggregate(&g, 1);
        // Branch 0 best = a2 (NodeId 2), branch 1 best = b2 (NodeId 4).
        // Both have Q ≈ 1.0 (Progress) / 2.0 (Breakthrough) respectively.
        // b2 should definitely be in the aggregate (highest Q globally).
        assert!(
            agg.contains(&NodeId(4)),
            "expected b2 (NodeId 4, highest Q) in aggregate: {agg:?}"
        );
        // Aggregate should have exactly 1 node per branch = 2 total.
        assert_eq!(agg.len(), 2, "expected 2 nodes (1 per branch), got {}: {agg:?}", agg.len());
        // All aggregate nodes should be valid.
        for id in &agg {
            assert!(id.0 < g.len() as u32);
        }
    }

    #[test]
    fn multi_branch_aggregate_multi_per_branch() {
        let g = build_two_branch_graph();
        let agg = multi_branch_aggregate(&g, 2);
        // Should contain top-2 from each branch = 4 nodes total.
        assert_eq!(agg.len(), 4);
    }
}
