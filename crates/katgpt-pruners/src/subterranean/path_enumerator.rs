//! PathEnumerator — exhaustive acyclic path enumeration through procedure graphs.
//!
//! Paper: arXiv:2605.22502 — travel=86 paths, zoom=60 paths, insurance=2,381 paths.
//! Uses DFS with visited tracking to enumerate all unique acyclic paths
//! from start node to any terminal node.

use std::collections::HashSet;

use crate::subterranean::ProcedureGraph;
use crate::subterranean::types::Trajectory;

// ── PathEnumerator ─────────────────────────────────────────────

/// Enumerates all unique acyclic paths through a procedure graph using DFS.
///
/// Paper §3: "each unique path through the procedure graph corresponds to
/// a distinct scenario the model must learn to handle."
pub struct PathEnumerator<'a, G: ProcedureGraph> {
    graph: &'a G,
    max_depth: usize,
}

impl<'a, G: ProcedureGraph> PathEnumerator<'a, G> {
    /// Create a new path enumerator with a safety depth limit.
    ///
    /// `max_depth` prevents exponential blowup on large or cyclic graphs.
    /// Paper uses 39 as the longest path observed.
    pub fn new(graph: &'a G, max_depth: usize) -> Self {
        Self { graph, max_depth }
    }

    /// Access the underlying procedure graph.
    pub fn graph(&self) -> &'a G {
        self.graph
    }

    /// Enumerate all acyclic paths from start to any terminal node.
    ///
    /// Returns an empty vector if no terminal nodes are reachable or
    /// if all paths exceed `max_depth`.
    pub fn enumerate(&self) -> Vec<Trajectory<G::NodeId>> {
        let start = self.graph.start_node();
        let terminals: HashSet<G::NodeId> = self.graph.terminal_nodes().iter().copied().collect();

        let mut results = Vec::new();
        let mut current_path = Trajectory::from_start(start);
        let mut visited = HashSet::new();
        visited.insert(start);

        self.dfs_enumerate(
            start,
            &terminals,
            &mut visited,
            &mut current_path,
            &mut results,
        );

        results
    }

    /// Count paths without materializing them (for cost estimation).
    pub fn count_paths(&self) -> usize {
        // Fast path: just count without building trajectory strings
        let start = self.graph.start_node();
        let terminals: HashSet<G::NodeId> = self.graph.terminal_nodes().iter().copied().collect();

        let mut visited = HashSet::new();
        visited.insert(start);

        self.dfs_count(start, &terminals, &mut visited, 0)
    }

    /// Sample `n` random paths using inverse-length weighting.
    ///
    /// Shorter paths receive higher weight (paper: paths vary 4–39 turns).
    /// Returns fewer than `n` paths if the graph has fewer unique paths.
    pub fn sample(&self, n: usize, rng: &mut fastrand::Rng) -> Vec<Trajectory<G::NodeId>> {
        let all_paths = self.enumerate();

        match all_paths.is_empty() {
            true => Vec::new(),
            false => {
                let requested = n.min(all_paths.len());
                self.weighted_sample(&all_paths, requested, rng)
            }
        }
    }

    // ── Internal helpers ───────────────────────────────────────

    /// DFS recursive enumeration with cycle detection.
    fn dfs_enumerate(
        &self,
        current: G::NodeId,
        terminals: &HashSet<G::NodeId>,
        visited: &mut HashSet<G::NodeId>,
        path: &mut Trajectory<G::NodeId>,
        results: &mut Vec<Trajectory<G::NodeId>>,
    ) {
        // Terminal reached — record this path
        if terminals.contains(&current) {
            results.push(path.clone());
            // Don't return early: terminal nodes might have outgoing edges
            // in some procedure graphs (e.g., game_over → restart)
        }

        // Depth limit exceeded — stop exploring
        if path.path.len() >= self.max_depth {
            return;
        }

        // Explore all outgoing edges
        let edges = self.graph.edges_from(current);
        for (next, condition) in edges {
            // Skip already-visited nodes (acyclic constraint)
            if visited.contains(next) {
                continue;
            }

            let cond = condition.as_ref().map(|c| format!("{c:?}"));
            path.push(*next, cond);
            visited.insert(*next);

            self.dfs_enumerate(*next, terminals, visited, path, results);

            // Backtrack
            path.path.pop();
            path.conditions_met.pop();
            visited.remove(next);
        }
    }

    /// DFS count-only variant — avoids allocating trajectories.
    fn dfs_count(
        &self,
        current: G::NodeId,
        terminals: &HashSet<G::NodeId>,
        visited: &mut HashSet<G::NodeId>,
        depth: usize,
    ) -> usize {
        let mut count = 0;

        // Terminal reached — count this path
        if terminals.contains(&current) {
            count += 1;
        }

        // Depth limit exceeded
        if depth >= self.max_depth {
            return count;
        }

        let edges = self.graph.edges_from(current);
        for (next, _) in edges {
            if visited.contains(next) {
                continue;
            }

            visited.insert(*next);
            count += self.dfs_count(*next, terminals, visited, depth + 1);
            visited.remove(next);
        }

        count
    }

    /// Weighted random sampling with inverse-length weighting.
    ///
    /// Weight = 1 / (path_length ^ alpha), where alpha controls
    /// the preference for shorter paths. alpha=1.0 means inverse length.
    fn weighted_sample(
        &self,
        paths: &[Trajectory<G::NodeId>],
        n: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<Trajectory<G::NodeId>> {
        let alpha: f64 = 1.0;
        let weights: Vec<f64> = paths
            .iter()
            .map(|t| {
                let len = t.path.len().max(1) as f64;
                1.0 / len.powf(alpha)
            })
            .collect();

        let mut sampled = Vec::with_capacity(n);
        let mut remaining_indices: Vec<usize> = (0..paths.len()).collect();

        for _ in 0..n {
            match remaining_indices.is_empty() {
                true => break,
                false => {
                    let rem_weights: Vec<f64> =
                        remaining_indices.iter().map(|&i| weights[i]).collect();
                    let rem_total: f64 = rem_weights.iter().sum();

                    let pick = self.weighted_pick(&rem_weights, rem_total, rng);
                    sampled.push(paths[remaining_indices[pick]].clone());
                    // Remove picked index to avoid duplicates
                    remaining_indices.swap_remove(pick);
                }
            }
        }

        sampled
    }

    /// Pick an index from weighted choices using cumulative distribution.
    fn weighted_pick(&self, weights: &[f64], total: f64, rng: &mut fastrand::Rng) -> usize {
        let threshold = rng.f64() * total;
        let mut cumulative = 0.0;

        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                return i;
            }
        }

        // Fallback: last index (handles floating-point edge cases)
        weights.len().saturating_sub(1)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal linear graph: A -> B -> C(terminal).
    /// Should yield exactly 1 path.
    struct LinearGraph;

    impl ProcedureGraph for LinearGraph {
        type NodeId = u32;
        type Condition = String;

        fn start_node(&self) -> Self::NodeId {
            0
        }
        fn terminal_nodes(&self) -> &[Self::NodeId] {
            &[2]
        }
        fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
            static EMPTY: [(u32, Option<String>); 0] = [];
            static A_EDGES: [(u32, Option<String>); 1] = [(1, None)];
            static B_EDGES: [(u32, Option<String>); 1] = [(2, None)];
            match node {
                0 => &A_EDGES,
                1 => &B_EDGES,
                _ => &EMPTY,
            }
        }
        fn node_count(&self) -> usize {
            3
        }
        fn edge_count(&self) -> usize {
            2
        }
        fn node_label(&self, node: Self::NodeId) -> &str {
            match node {
                0 => "A",
                1 => "B",
                2 => "C",
                _ => "?",
            }
        }
    }

    #[test]
    fn test_linear_graph_single_path() {
        let graph = LinearGraph;
        let enumerator = PathEnumerator::new(&graph, 100);
        let paths = enumerator.enumerate();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].path, vec![0, 1, 2]);
        assert_eq!(enumerator.count_paths(), 1);
    }

    /// Diamond graph: A -> B, A -> C, B -> D(term), C -> D(term).
    /// Should yield 2 paths.
    struct DiamondGraph;

    impl ProcedureGraph for DiamondGraph {
        type NodeId = u32;
        type Condition = String;

        fn start_node(&self) -> Self::NodeId {
            0
        }
        fn terminal_nodes(&self) -> &[Self::NodeId] {
            &[3]
        }
        fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
            static EMPTY: [(u32, Option<String>); 0] = [];
            static A_EDGES: [(u32, Option<String>); 2] = [(1, None), (2, None)];
            static B_EDGES: [(u32, Option<String>); 1] = [(3, None)];
            static C_EDGES: [(u32, Option<String>); 1] = [(3, None)];
            match node {
                0 => &A_EDGES,
                1 => &B_EDGES,
                2 => &C_EDGES,
                _ => &EMPTY,
            }
        }
        fn node_count(&self) -> usize {
            4
        }
        fn edge_count(&self) -> usize {
            4
        }
        fn node_label(&self, node: Self::NodeId) -> &str {
            match node {
                0 => "A",
                1 => "B",
                2 => "C",
                3 => "D",
                _ => "?",
            }
        }
    }

    #[test]
    fn test_diamond_graph_two_paths() {
        let graph = DiamondGraph;
        let enumerator = PathEnumerator::new(&graph, 100);
        let paths = enumerator.enumerate();

        assert_eq!(paths.len(), 2);
        assert_eq!(enumerator.count_paths(), 2);

        // Verify both paths end at terminal
        for p in &paths {
            assert_eq!(*p.end().unwrap(), 3);
            assert_eq!(*p.start().unwrap(), 0);
        }
    }

    /// Cyclic graph: A -> B -> C -> A (cycle), B -> D(terminal).
    /// Should still enumerate 1 path (acyclic constraint prevents A->B->C->A).
    struct CyclicGraph;

    impl ProcedureGraph for CyclicGraph {
        type NodeId = u32;
        type Condition = String;

        fn start_node(&self) -> Self::NodeId {
            0
        }
        fn terminal_nodes(&self) -> &[Self::NodeId] {
            &[3]
        }
        fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
            static EMPTY: [(u32, Option<String>); 0] = [];
            static A_EDGES: [(u32, Option<String>); 1] = [(1, None)];
            static B_EDGES: [(u32, Option<String>); 2] = [(3, None), (2, None)];
            static C_EDGES: [(u32, Option<String>); 1] = [(0, None)];
            match node {
                0 => &A_EDGES,
                1 => &B_EDGES,
                2 => &C_EDGES,
                _ => &EMPTY,
            }
        }
        fn node_count(&self) -> usize {
            4
        }
        fn edge_count(&self) -> usize {
            4
        }
        fn node_label(&self, node: Self::NodeId) -> &str {
            match node {
                0 => "A",
                1 => "B",
                2 => "C",
                3 => "D",
                _ => "?",
            }
        }
    }

    #[test]
    fn test_cyclic_graph_acyclic_paths_only() {
        let graph = CyclicGraph;
        let enumerator = PathEnumerator::new(&graph, 100);
        let paths = enumerator.enumerate();

        // Only A -> B -> D should be found (B -> C -> A is blocked by visited)
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].path, vec![0, 1, 3]);
        assert_eq!(enumerator.count_paths(), 1);
    }

    #[test]
    fn test_max_depth_limits_paths() {
        let graph = LinearGraph;
        // Depth 1 means we can only go from A(0) to B(1), not reach C(2)
        let enumerator = PathEnumerator::new(&graph, 1);
        let paths = enumerator.enumerate();

        // Path A -> B -> C requires depth 2; with max_depth=1, no terminal reachable
        assert!(paths.is_empty());
    }

    #[test]
    fn test_sample_returns_correct_count() {
        let graph = DiamondGraph;
        let enumerator = PathEnumerator::new(&graph, 100);
        let mut rng = fastrand::Rng::with_seed(42);

        let samples = enumerator.sample(2, &mut rng);
        assert_eq!(samples.len(), 2);

        // Each sample must be a valid path
        for s in &samples {
            assert_eq!(*s.start().unwrap(), 0);
            assert_eq!(*s.end().unwrap(), 3);
        }
    }

    #[test]
    fn test_sample_fewer_than_requested() {
        let graph = LinearGraph;
        let enumerator = PathEnumerator::new(&graph, 100);
        let mut rng = fastrand::Rng::with_seed(42);

        // Only 1 path exists, requesting 5 should return 1
        let samples = enumerator.sample(5, &mut rng);
        assert_eq!(samples.len(), 1);
    }

    /// Multi-terminal graph: A -> B(term1), A -> C(term2).
    /// Should yield 2 paths (one per terminal).
    struct MultiTerminalGraph;

    impl ProcedureGraph for MultiTerminalGraph {
        type NodeId = u32;
        type Condition = String;

        fn start_node(&self) -> Self::NodeId {
            0
        }
        fn terminal_nodes(&self) -> &[Self::NodeId] {
            &[1, 2]
        }
        fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
            static EMPTY: [(u32, Option<String>); 0] = [];
            static A_EDGES: [(u32, Option<String>); 2] = [(1, None), (2, None)];
            match node {
                0 => &A_EDGES,
                _ => &EMPTY,
            }
        }
        fn node_count(&self) -> usize {
            3
        }
        fn edge_count(&self) -> usize {
            2
        }
        fn node_label(&self, node: Self::NodeId) -> &str {
            match node {
                0 => "A",
                1 => "B",
                2 => "C",
                _ => "?",
            }
        }
    }

    #[test]
    fn test_multi_terminal_graph() {
        let graph = MultiTerminalGraph;
        let enumerator = PathEnumerator::new(&graph, 100);
        let paths = enumerator.enumerate();

        assert_eq!(paths.len(), 2);
        assert_eq!(enumerator.count_paths(), 2);
    }
}
