//! WorkflowLattice — sigmoid-graded relevance scoring (Plan 223, Phase 4).
//!
//! Builds a lattice of predicates with implication ordering.
//! Sigmoid-graded satisfaction incrementally propagates across DDTree levels.

use crate::speculative::types::ScreeningPruner;

/// A node in the workflow lattice with sigmoid-graded satisfaction.
#[derive(Clone, Debug)]
pub struct PredicateNode {
    /// Unique identifier for this predicate.
    pub id: u32,
    /// Current satisfaction score [0, 1] via sigmoid.
    pub satisfaction: f32,
    /// Implication ordering: predicates that must be satisfied before this one.
    pub implies: Vec<u32>,
}

impl PredicateNode {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            satisfaction: 0.0,
            implies: Vec::new(),
        }
    }

    /// Update satisfaction using sigmoid grading.
    /// raw ∈ (-∞, +∞) → sigmoid → [0, 1]
    pub fn update_satisfaction(&mut self, raw_score: f32) {
        self.satisfaction = 1.0 / (1.0 + (-raw_score).exp());
    }
}

/// Workflow lattice for incremental satisfaction tracking.
#[derive(Clone, Debug, Default)]
pub struct WorkflowLattice {
    /// Predicate nodes indexed by id.
    pub nodes: Vec<PredicateNode>,
}

impl WorkflowLattice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate node to the lattice.
    pub fn add_node(&mut self, node: PredicateNode) {
        let id = node.id as usize;
        if id >= self.nodes.len() {
            self.nodes.resize(id + 1, PredicateNode::new(0));
        }
        self.nodes[id] = node;
    }

    /// Compute overall lattice satisfaction.
    /// Uses geometric mean of all node satisfactions (not softmax).
    pub fn overall_satisfaction(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let log_sum: f32 = self
            .nodes
            .iter()
            .filter(|n| n.satisfaction > 1e-10)
            .map(|n| n.satisfaction.ln())
            .sum();
        let n = self.nodes.iter().filter(|n| n.satisfaction > 1e-10).count();
        if n == 0 {
            return 0.0;
        }
        (log_sum / n as f32).exp()
    }

    /// Propagate satisfaction from parent to child predicates.
    /// If parent is unsatisfied, child satisfaction is reduced.
    pub fn propagate(&mut self) {
        let mut new_sats: Vec<f32> = self.nodes.iter().map(|n| n.satisfaction).collect();
        for node in &self.nodes {
            for &implies_id in &node.implies {
                let implies_id = implies_id as usize;
                if implies_id < new_sats.len() {
                    // Child satisfaction is bounded by parent satisfaction
                    let parent_sat = self.nodes[implies_id].satisfaction;
                    let child_idx = node.id as usize;
                    if child_idx < new_sats.len() {
                        new_sats[child_idx] = new_sats[child_idx].min(parent_sat);
                    }
                }
            }
        }
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.satisfaction = new_sats[i];
        }
    }

    /// Incremental DDTree-level propagation.
    ///
    /// Takes a list of `(predicate_id, raw_score)` pairs from a DDTree BFS level,
    /// updates each predicate node's satisfaction via `update_satisfaction()`,
    /// then propagates satisfaction through the lattice.
    /// Returns the overall satisfaction after propagation.
    pub fn propagate_level(&mut self, level_results: &[(u32, f32)]) -> f32 {
        for &(pred_id, raw_score) in level_results {
            let idx = pred_id as usize;
            if idx < self.nodes.len() {
                self.nodes[idx].update_satisfaction(raw_score);
            }
        }
        self.propagate();
        self.overall_satisfaction()
    }
}

// ── ScreeningPruner impl ──────────────────────────────────────────

#[cfg(feature = "workflow_lattice")]
impl ScreeningPruner for WorkflowLattice {
    #[inline]
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.overall_satisfaction()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_grading() {
        let mut node = PredicateNode::new(0);
        node.update_satisfaction(0.0);
        assert!((node.satisfaction - 0.5).abs() < 0.01);
        node.update_satisfaction(5.0);
        assert!(node.satisfaction > 0.9);
        node.update_satisfaction(-5.0);
        assert!(node.satisfaction < 0.1);
    }

    #[test]
    fn test_overall_satisfaction() {
        let mut lattice = WorkflowLattice::new();
        let mut n1 = PredicateNode::new(0);
        n1.satisfaction = 0.8;
        let mut n2 = PredicateNode::new(1);
        n2.satisfaction = 0.6;
        lattice.add_node(n1);
        lattice.add_node(n2);
        let overall = lattice.overall_satisfaction();
        assert!(overall > 0.0 && overall <= 1.0);
    }

    #[test]
    fn test_propagation() {
        let mut lattice = WorkflowLattice::new();
        let mut parent = PredicateNode::new(0);
        parent.satisfaction = 0.3;
        let mut child = PredicateNode::new(1);
        child.satisfaction = 0.9;
        child.implies = vec![0]; // depends on parent
        lattice.add_node(parent);
        lattice.add_node(child);
        lattice.propagate();
        // Child satisfaction should be bounded by parent
        assert!(lattice.nodes[1].satisfaction <= 0.3 + 0.01);
    }

    /// Test: lattice satisfaction incrementally builds across DDTree levels.
    ///
    /// P0 (root), P1 (implies P0), P2 (implies P0).
    /// Level 0: P0 gets raw_score=3.0 → sigmoid ≈ 0.95
    /// Level 1: P1 gets raw_score=2.0 → sigmoid ≈ 0.88
    /// Level 2: P2 gets raw_score=-1.0 → sigmoid ≈ 0.27
    /// After level 2: P2 satisfaction = min(0.27, P0_satisfaction)
    /// Overall satisfaction should decrease across levels as more constraints are checked.
    #[test]
    fn test_incremental_ddtree_levels() {
        let mut lattice = WorkflowLattice::new();

        // P0: root predicate
        let mut p0 = PredicateNode::new(0);
        p0.implies = vec![];

        // P1: implies P0
        let mut p1 = PredicateNode::new(1);
        p1.implies = vec![0];

        // P2: implies P0
        let mut p2 = PredicateNode::new(2);
        p2.implies = vec![0];

        lattice.add_node(p0);
        lattice.add_node(p1);
        lattice.add_node(p2);

        // Level 0: P0 gets high satisfaction
        let sat_l0 = lattice.propagate_level(&[(0, 3.0)]);
        let p0_sat = lattice.nodes[0].satisfaction;
        assert!(
            p0_sat > 0.9,
            "P0 satisfaction should be ~0.95, got {p0_sat}"
        );

        // Level 1: P1 gets medium satisfaction
        let sat_l1 = lattice.propagate_level(&[(1, 2.0)]);
        let p1_sat = lattice.nodes[1].satisfaction;
        assert!(
            p1_sat > 0.8,
            "P1 satisfaction should be ~0.88, got {p1_sat}"
        );
        // P1 implies P0, so P1 should be bounded by P0 (both are high, so no clamp)
        assert!(p1_sat <= p0_sat + 0.01);

        // Level 2: P2 gets low satisfaction
        let sat_l2 = lattice.propagate_level(&[(2, -1.0)]);
        let p2_sat = lattice.nodes[2].satisfaction;
        let expected_p2 = 1.0 / (1.0 + 1.0_f32.exp()); // sigmoid(-1.0)
        assert!(
            (p2_sat - expected_p2).abs() < 0.01,
            "P2 raw sigmoid should be ~0.27, got {p2_sat}"
        );
        // P2 implies P0 → P2 = min(0.27, 0.95) = 0.27
        assert!(p2_sat <= p0_sat + 0.01);

        // Overall satisfaction should decrease across levels
        // as more (and less satisfied) constraints are added
        assert!(
            sat_l0 > sat_l1,
            "sat after level 0 ({sat_l0}) should > sat after level 1 ({sat_l1})"
        );
        assert!(
            sat_l1 > sat_l2,
            "sat after level 1 ({sat_l1}) should > sat after level 2 ({sat_l2})"
        );
    }

    /// Benchmark: WorkflowLattice relevance overhead vs NoScreeningPruner.
    ///
    /// 10 predicates in a chain, 10k relevance() calls.
    /// Asserts overhead < 500ns per call.
    #[test]
    fn test_bench_lattice_vs_noop() {
        use std::time::Instant;

        // Build a 10-predicate chain
        let mut lattice = WorkflowLattice::new();
        for i in 0..10u32 {
            let mut node = PredicateNode::new(i);
            if i > 0 {
                node.implies = vec![i - 1];
            }
            node.update_satisfaction(1.0); // moderate satisfaction
            lattice.add_node(node);
        }
        lattice.propagate();

        const ITERS: usize = 10_000;

        // Warm up
        for _ in 0..100 {
            let _ = lattice.overall_satisfaction();
        }

        // Benchmark lattice overall_satisfaction (the hot path in relevance)
        let start = Instant::now();
        let mut sum = 0.0_f32;
        for _ in 0..ITERS {
            sum += lattice.overall_satisfaction();
        }
        let lattice_duration = start.elapsed();
        assert!(sum > 0.0, "prevent optimizer from eliding: {sum}");

        // Benchmark NoScreeningPruner equivalent (just return 1.0)
        let start = Instant::now();
        let mut sum2 = 0.0_f32;
        for _ in 0..ITERS {
            sum2 += 1.0_f32; // NoScreeningPruner::relevance is just 1.0
        }
        let noop_duration = start.elapsed();
        assert!(sum2 > 0.0, "prevent optimizer from eliding: {sum2}");

        let lattice_ns = lattice_duration.as_nanos() as f64 / ITERS as f64;
        let noop_ns = noop_duration.as_nanos() as f64 / ITERS as f64;
        let overhead_ns = lattice_ns - noop_ns;

        // Overhead should be < 500ns per call
        assert!(
            overhead_ns < 500.0,
            "lattice overhead {overhead_ns:.1}ns exceeds 500ns budget (lattice={lattice_ns:.1}ns, noop={noop_ns:.1}ns)"
        );
    }
}
