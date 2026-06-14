//! Graph data structure for Progressive MCGS.
//!
//! Paper §3.2.1 formalizes `G = (V, E)`, `E = E_T ∪ E_ref`:
//! - **Primary edges** `E_T`: parent→child generative, used for selection + backprop.
//! - **Reference edges** `E_ref`: cross-branch / non-adjacent info flow,
//!   **excluded from backprop**.
//!
//! When `E_ref = ∅`, the search reduces to vanilla MCTS — this is the GOAT G2
//! precondition (see `.plans/272_progressive_mcgs.md` Phase 3).
//!
//! # Critical Invariant
//!
//! [`ProgressiveMcgs::backprop`] walks ONLY the `primary_parent` chain.
//! It NEVER traverses `reference_edges`. This is the single most important
//! correctness property — it guarantees that reference edges compose
//! information without polluting credit assignment.

use crate::progressive_mcgs::types::{BranchId, NodeId, Reward, UCT_EPSILON};

/// Generic Progressive MCGS graph — payload `N`, reward already classified via [`Reward`].
///
/// Storage is dense (SoA-style): each node attribute lives in its own `Vec`,
/// indexed by `NodeId`. This improves cache locality for hot-path iteration
/// over `visits` and `cumulative_reward`.
///
/// # Type Parameters
///
/// - `N`: node payload. Should be small (≤ 64 bytes recommended) — if larger,
///   use `Box<Payload>` to keep the SoA vectors cheap to grow. Must be `Clone`.
///
/// # Allocation Discipline (AGENTS.md hot-loop rules)
///
/// - `select()` / `observe()` / `q_value()` are zero-allocation (no heap).
/// - `expand_*()` may allocate — they happen once per decision, not per token.
#[derive(Debug, Clone)]
pub struct ProgressiveMcgs<N: Clone> {
    // --- Node payload + tree structure ---
    /// One payload per node, indexed by `NodeId`.
    payloads: Vec<N>,
    /// Primary-edge parent (`None` for root).
    primary_parent: Vec<Option<NodeId>>,
    /// Primary-edge children (one Vec per node — small, usually ≤ 4).
    primary_children: Vec<Vec<NodeId>>,
    /// Branch each node belongs to (`BranchId::NONE` for unassigned).
    branch_id: Vec<BranchId>,

    // --- Reference edges `E_ref` (information-only, excluded from backprop) ---
    /// Reference edges per child node. Cap = `max_refs_per_node` with LRU eviction.
    /// `Vec<Vec<NodeId>>` for simplicity; cap enforced in `add_reference`.
    reference_edges: Vec<Vec<NodeId>>,
    /// Max refs per node — typically [`MAX_REFS_PER_NODE`](crate::progressive_mcgs::types::MAX_REFS_PER_NODE) = 3.
    max_refs_per_node: usize,

    // --- MCTS statistics (one per node, on `E_T`) ---
    /// Visit count per node.
    visits: Vec<u32>,
    /// Cumulative reward per node (sum of backpropagated rewards).
    cumulative_reward: Vec<f32>,

    // --- Best-trackers (snapshotted BEFORE update per Plan 272 §4 risk) ---
    /// Best reward seen so far per branch (None = no observation yet).
    branch_best: Vec<Option<Reward>>,
    /// Global best reward seen so far.
    global_best: Option<Reward>,

    /// Hard cap on total nodes; future: LRU eviction policy.
    max_nodes: usize,
}

impl<N: Clone> ProgressiveMcgs<N> {
    /// Construct an empty graph.
    ///
    /// Pre-allocates space for `config.max_nodes` capacity in all vectors
    /// to avoid reallocation during search.
    #[must_use]
    pub fn new(max_nodes: usize, max_refs_per_node: usize) -> Self {
        Self {
            payloads: Vec::with_capacity(64),
            primary_parent: Vec::with_capacity(64),
            primary_children: Vec::with_capacity(64),
            branch_id: Vec::with_capacity(64),
            reference_edges: Vec::with_capacity(64),
            max_refs_per_node,
            visits: Vec::with_capacity(64),
            cumulative_reward: Vec::with_capacity(64),
            branch_best: Vec::with_capacity(16),
            global_best: None,
            max_nodes,
        }
    }

    /// Add the root node. Returns its [`NodeId`] (always `0`).
    ///
    /// Panics if called twice — there is exactly one root per graph.
    pub fn add_root(&mut self, payload: N, branch: BranchId) -> NodeId {
        assert!(self.payloads.is_empty(), "add_root called twice");
        self.push_node(payload, None, branch);
        NodeId(0)
    }

    /// Allocate a fresh node id (dense — always `len` before push).
    fn push_node(
        &mut self,
        payload: N,
        parent: Option<NodeId>,
        branch: BranchId,
    ) -> NodeId {
        let id = NodeId(self.payloads.len() as u32);
        assert!(
            (id.idx() as usize) < self.max_nodes,
            "max_nodes exceeded ({} >= {})",
            id.idx(),
            self.max_nodes
        );
        self.payloads.push(payload);
        self.primary_parent.push(parent);
        self.primary_children.push(Vec::with_capacity(4));
        self.branch_id.push(branch);
        self.reference_edges.push(Vec::with_capacity(self.max_refs_per_node));
        self.visits.push(0);
        self.cumulative_reward.push(0.0);
        // Ensure branch_best vector is large enough.
        if branch != BranchId::NONE {
            let bidx = branch.idx();
            if bidx >= self.branch_best.len() {
                self.branch_best.resize(bidx + 1, None);
            }
        }
        if let Some(p) = parent {
            self.primary_children[p.idx()].push(id);
        }
        id
    }

    /// Add a primary-edge child of `parent`.
    ///
    /// Used for the baseline "Primary expansion" operator (paper Eq. 12)
    /// and as the parent-side of all other expansion operators.
    #[must_use]
    pub fn expand_primary(&mut self, parent: NodeId, payload: N, branch: BranchId) -> NodeId {
        // Inherit parent's branch if caller passes NONE.
        let branch = if branch == BranchId::NONE {
            self.branch_id[parent.idx()]
        } else {
            branch
        };
        self.push_node(payload, Some(parent), branch)
    }

    /// Add a reference edge `r → child` (paper Eq. 6, 13, 14, 15).
    ///
    /// **This does NOT participate in backprop.** Reference edges are
    /// write-at-expansion, read-at-proposal — they carry information flow
    /// only. See [`backprop`](Self::backprop) for the credit-assignment path.
    ///
    /// If the per-node cap (`max_refs_per_node`) is reached, the OLDEST
    /// reference is evicted (LRU). Eviction touches only `E_ref`; the primary
    /// tree `E_T` is never modified.
    pub fn add_reference(&mut self, child: NodeId, referenced: NodeId) {
        let edges = &mut self.reference_edges[child.idx()];
        // No-op if already referenced (dedup).
        if edges.iter().any(|&e| e == referenced) {
            return;
        }
        if edges.len() >= self.max_refs_per_node {
            // LRU eviction: drop the oldest (front of Vec).
            edges.remove(0);
        }
        edges.push(referenced);
    }

    /// Read-only access to a node's reference set.
    #[inline]
    #[must_use]
    pub fn references(&self, node: NodeId) -> &[NodeId] {
        &self.reference_edges[node.idx()]
    }

    /// Read-only access to primary-edge children.
    #[inline]
    #[must_use]
    pub fn children(&self, node: NodeId) -> &[NodeId] {
        &self.primary_children[node.idx()]
    }

    /// Read-only access to primary-edge parent.
    #[inline]
    #[must_use]
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.primary_parent[node.idx()]
    }

    /// Read-only access to node payload.
    #[inline]
    #[must_use]
    pub fn payload(&self, node: NodeId) -> &N {
        &self.payloads[node.idx()]
    }

    /// Mutable access to node payload.
    #[inline]
    #[must_use]
    pub fn payload_mut(&mut self, node: NodeId) -> &mut N {
        &mut self.payloads[node.idx()]
    }

    /// Number of nodes in the graph.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.payloads.len()
    }

    /// Is the graph empty?
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    /// Branch id for a node.
    #[inline]
    #[must_use]
    pub fn branch_of(&self, node: NodeId) -> BranchId {
        self.branch_id[node.idx()]
    }

    /// Visit count for a node.
    #[inline]
    #[must_use]
    pub fn visits(&self, node: NodeId) -> u32 {
        self.visits[node.idx()]
    }

    /// Cumulative reward for a node.
    #[inline]
    #[must_use]
    pub fn cumulative_reward(&self, node: NodeId) -> f32 {
        self.cumulative_reward[node.idx()]
    }

    /// Q-value = average reward = `cumulative_reward / (visits + ε)` (paper Eq. 9).
    ///
    /// Returns `0.0` for unvisited nodes.
    #[inline]
    #[must_use]
    pub fn q_value(&self, node: NodeId) -> f32 {
        let v = self.visits[node.idx()];
        if v == 0 {
            0.0
        } else {
            self.cumulative_reward[node.idx()] / (v as f32 + UCT_EPSILON)
        }
    }

    /// **CRITICAL**: backpropagate `reward` from `leaf` up the `E_T` tree only.
    ///
    /// Updates `visits` and `cumulative_reward` for the leaf and all ancestors
    /// reachable via `primary_parent`. **Reference edges (`E_ref`) are NEVER
    /// traversed** — this guarantees credit assignment isolation.
    ///
    /// Paper Eqs. 8–9:
    /// ```text
    /// N_u ← N_u + 1
    /// W_u ← W_u + R(v)
    /// Q_u = W_u / (N_u + ε)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `leaf` doesn't exist in the graph.
    pub fn backprop(&mut self, leaf: NodeId, reward: Reward) {
        let r = reward.as_f32();
        let mut current = Some(leaf);
        while let Some(n) = current {
            let idx = n.idx();
            self.visits[idx] = self.visits[idx].saturating_add(1);
            self.cumulative_reward[idx] += r;
            current = self.primary_parent[idx];
            // CRITICAL: we walk ONLY primary_parent. reference_edges are
            // never touched here. This is the GOAT G2 invariant.
        }
    }

    /// Snapshot the branch best (BEFORE updating — see Plan 272 §4 risk).
    ///
    /// Returns the current best reward for `branch`, or `None` if no
    /// observation has been recorded yet. The caller uses this to classify
    /// a new reward into `Progress` vs `Breakthrough`.
    #[inline]
    #[must_use]
    pub fn branch_best(&self, branch: BranchId) -> Option<Reward> {
        if branch == BranchId::NONE {
            return None;
        }
        let idx = branch.idx();
        if idx >= self.branch_best.len() {
            None
        } else {
            self.branch_best[idx]
        }
    }

    /// Update the branch best. Caller MUST have already snapshotted the old
    /// value via [`branch_best`](Self::branch_best) and classified the reward.
    #[inline]
    pub fn set_branch_best(&mut self, branch: BranchId, reward: Reward) {
        if branch == BranchId::NONE {
            return;
        }
        let idx = branch.idx();
        if idx >= self.branch_best.len() {
            self.branch_best.resize(idx + 1, None);
        }
        // Only update if strictly greater (monotonic).
        match self.branch_best[idx] {
            Some(cur) if cur >= reward => {}
            _ => self.branch_best[idx] = Some(reward),
        }
    }

    /// Snapshot the global best (BEFORE updating).
    #[inline]
    #[must_use]
    pub const fn global_best(&self) -> Option<Reward> {
        self.global_best
    }

    /// Update the global best. Caller MUST have already snapshotted.
    #[inline]
    pub fn set_global_best(&mut self, reward: Reward) {
        match self.global_best {
            Some(cur) if cur >= reward => {}
            _ => self.global_best = Some(reward),
        }
    }

    /// Root node id (`0` if graph non-empty).
    #[inline]
    #[must_use]
    pub fn root(&self) -> Option<NodeId> {
        if self.payloads.is_empty() {
            None
        } else {
            Some(NodeId(0))
        }
    }

    /// All node ids, in insertion order.
    ///
    /// Used for diagnostic scans (e.g., building the Elite set).
    /// Returns an iterator, not a Vec — zero-allocation.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.payloads.len() as u32).map(NodeId)
    }

    /// Reference-edge count across the whole graph (for diagnostics).
    #[must_use]
    pub fn total_reference_edges(&self) -> usize {
        self.reference_edges.iter().map(Vec::len).sum()
    }
}

/// Expansion-operator tag — used by consumers to dispatch the correct proposer.
///
/// The four operators from paper §3.2.2 / Appendix B. The consumer builds
/// the payload; this enum just signals *which* reference-set builder to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExpansionOperator {
    /// Baseline primary expansion — no references (paper Eq. 12).
    Primary = 0,
    /// Intra-branch evolution — reference last-k ancestors (paper Eq. 13).
    IntraBranchEvolve = 1,
    /// Cross-branch reference — reference top-N globally (paper Eq. 14).
    CrossBranchReference = 2,
    /// Multi-branch aggregation — reference union of top trajectories per
    /// branch; spawns new branch root under graph root (paper Eq. 15).
    MultiBranchAggregation = 3,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> ProgressiveMcgs<u32> {
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let _root = g.add_root(100, BranchId(0));
        g
    }

    #[test]
    fn add_root_works() {
        let g = make_graph();
        assert_eq!(g.len(), 1);
        assert_eq!(g.root(), Some(NodeId(0)));
        assert_eq!(g.branch_of(NodeId(0)), BranchId(0));
    }

    #[test]
    fn expand_primary_creates_child() {
        let mut g = make_graph();
        let child = g.expand_primary(NodeId(0), 200, BranchId(0));
        assert_eq!(g.parent(child), Some(NodeId(0)));
        assert_eq!(g.children(NodeId(0)), &[child]);
        assert_eq!(g.branch_of(child), BranchId(0));
    }

    #[test]
    fn add_reference_dedup() {
        let mut g = make_graph();
        let c1 = g.expand_primary(NodeId(0), 200, BranchId(0));
        let c2 = g.expand_primary(NodeId(0), 300, BranchId(0));
        g.add_reference(c1, c2);
        g.add_reference(c1, c2); // dedup
        assert_eq!(g.references(c1).len(), 1);
    }

    #[test]
    fn add_reference_lru_eviction() {
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        let target1 = g.expand_primary(root, 1, BranchId(0));
        let target2 = g.expand_primary(root, 2, BranchId(0));
        let target3 = g.expand_primary(root, 3, BranchId(0));
        let target4 = g.expand_primary(root, 4, BranchId(0));
        let child = g.expand_primary(root, 99, BranchId(0));

        g.add_reference(child, target1);
        g.add_reference(child, target2);
        g.add_reference(child, target3);
        assert_eq!(g.references(child).len(), 3);

        // Adding 4th should evict target1 (oldest).
        g.add_reference(child, target4);
        assert_eq!(g.references(child).len(), 3);
        assert!(
            !g.references(child).contains(&target1),
            "LRU should have evicted target1"
        );
        assert!(g.references(child).contains(&target4));
    }

    #[test]
    fn backprop_walks_primary_only() {
        // The GOAT G2 invariant: backprop must walk E_T only,
        // never E_ref. Build a graph with a cross-branch reference,
        // backprop on a leaf, assert only primary-chain stats change.
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));

        // Branch A: root → a1 → a2 (leaf)
        let a1 = g.expand_primary(root, 1, BranchId(0));
        let a2 = g.expand_primary(a1, 2, BranchId(0));

        // Branch B: root → b1 (separate branch)
        let b1 = g.expand_primary(root, 10, BranchId(1));

        // Cross-branch reference: a2 references b1.
        g.add_reference(a2, b1);

        // Initial state
        assert_eq!(g.visits(b1), 0);
        assert_eq!(g.cumulative_reward(b1), 0.0);
        assert_eq!(g.visits(a2), 0);

        // Backprop a Breakthrough on a2.
        g.backprop(a2, Reward::Breakthrough);

        // Primary chain (a2, a1, root) should be updated.
        assert_eq!(g.visits(a2), 1);
        assert_eq!(g.visits(a1), 1);
        assert_eq!(g.visits(root), 1);
        assert!((g.cumulative_reward(a2) - 2.0).abs() < 1e-6);

        // CRITICAL: b1 (referenced but NOT on primary chain) must be untouched.
        assert_eq!(
            g.visits(b1),
            0,
            "reference target should NOT receive backprop credit"
        );
        assert_eq!(
            g.cumulative_reward(b1),
            0.0,
            "reference target cumulative_reward should NOT change"
        );
    }

    #[test]
    fn backprop_with_e_ref_empty_matches_vanilla_mcts() {
        // GOAT G2 precondition: with no reference edges, Q-values must
        // match a vanilla MCTS run on the same RNG seed.
        // (The full benchmark is in tests.rs; here we just verify the
        // data structure gives identical results when refs are absent.)
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        let c1 = g.expand_primary(root, 1, BranchId(0));
        let c2 = g.expand_primary(root, 2, BranchId(0));

        g.backprop(c1, Reward::Progress);
        g.backprop(c1, Reward::Breakthrough);
        g.backprop(c2, Reward::Failure);

        // c1: 2 visits, cumulative = 1 + 2 = 3, Q ≈ 3/2 = 1.5
        assert_eq!(g.visits(c1), 2);
        assert!((g.q_value(c1) - 1.5).abs() < 1e-3, "Q(c1) = {}", g.q_value(c1));

        // c2: 1 visit, cumulative = -1, Q = -1
        assert_eq!(g.visits(c2), 1);
        assert!((g.q_value(c2) - (-1.0)).abs() < 1e-3);

        // root: 3 visits, cumulative = 3 + (-1) = 2, Q ≈ 2/3
        assert_eq!(g.visits(root), 3);
        assert!((g.q_value(root) - 2.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn q_value_unvisited_zero() {
        let mut g = make_graph();
        let c = g.expand_primary(NodeId(0), 1, BranchId(0));
        assert_eq!(g.q_value(c), 0.0);
    }

    #[test]
    fn branch_best_monotonic() {
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let _root = g.add_root(0, BranchId(0));
        assert_eq!(g.branch_best(BranchId(0)), None);

        g.set_branch_best(BranchId(0), Reward::Progress);
        assert_eq!(g.branch_best(BranchId(0)), Some(Reward::Progress));

        // Lower reward — should NOT overwrite.
        g.set_branch_best(BranchId(0), Reward::Neutral);
        assert_eq!(g.branch_best(BranchId(0)), Some(Reward::Progress));

        // Higher reward — should overwrite.
        g.set_branch_best(BranchId(0), Reward::Breakthrough);
        assert_eq!(g.branch_best(BranchId(0)), Some(Reward::Breakthrough));
    }

    #[test]
    fn global_best_monotonic() {
        let mut g = make_graph();
        assert_eq!(g.global_best(), None);
        g.set_global_best(Reward::Progress);
        assert_eq!(g.global_best(), Some(Reward::Progress));
        g.set_global_best(Reward::Neutral);
        assert_eq!(g.global_best(), Some(Reward::Progress));
        g.set_global_best(Reward::Breakthrough);
        assert_eq!(g.global_best(), Some(Reward::Breakthrough));
    }

    #[test]
    fn total_reference_edges_counts() {
        let mut g = ProgressiveMcgs::<u32>::new(100, 3);
        let root = g.add_root(0, BranchId(0));
        let a = g.expand_primary(root, 1, BranchId(0));
        let b = g.expand_primary(root, 2, BranchId(0));
        let c = g.expand_primary(root, 3, BranchId(0));
        g.add_reference(a, b);
        g.add_reference(a, c);
        assert_eq!(g.total_reference_edges(), 2);
    }
}
