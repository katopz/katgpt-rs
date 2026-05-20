//! SimpleTES evaluation-driven scaling loop (Plan 086).
//!
//! Feature-gated under `tes_loop` (requires `bandit`).
//!
//! Implements the RPUCG (Rooted Propagation UCB on Graph) selection strategy
//! from SimpleTES (arXiv:2604.19341). The key insight: evaluation-driven loops
//! with simple policies beat frontier models by organizing test-time compute
//! as (C, L, K, Φ) — global width, refinement depth, local sample size, and
//! proposal constructor.

#[cfg(feature = "tes_loop")]
use std::cmp::Ordering;
#[cfg(feature = "tes_loop")]
use std::collections::HashSet;

#[cfg(feature = "tes_loop")]
use crate::speculative::types::{TesConfig, TesNode};

// ── Trait ───────────────────────────────────────────────────────

/// Core trait for the TES evaluation loop.
///
/// Implementors provide the evaluation function; the trait provides
/// default RPUCG selection and value propagation.
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────┐
/// │ TesLoop<C, L, K, Φ>                          │
/// │                                               │
/// │  C trajectories × L steps × K candidates      │
/// │  Φ = RPUCG (graph-based UCB)                  │
/// │                                               │
/// │  Per-step: BanditPruner (existing)             │
/// │  Per-trajectory: RPUCG propagation (this)      │
/// │  Across-trajectories: TrajectoryPruner (arena) │
/// └─────────────────────────────────────────────┘
/// ```
#[cfg(feature = "tes_loop")]
pub trait TesLoop: Send + Sync {
    /// Get the TES configuration.
    fn config(&self) -> &TesConfig;

    /// Total evaluation budget: C × L × K.
    fn budget(&self) -> usize {
        self.config().budget()
    }

    /// Select `count` inspirations from history using RPUCG greedy selection.
    ///
    /// Greedy by `propagated_value`, excluding one-hop neighbors for diversity
    /// (SimpleTES Section 3.3). This ensures selected inspirations cover
    /// distinct regions of the solution graph.
    ///
    /// Returns indices into `history`.
    fn select_inspirations(&self, history: &[TesNode], count: usize) -> Vec<usize> {
        if history.is_empty() || count == 0 {
            return Vec::new();
        }

        let mut selected: Vec<usize> = Vec::with_capacity(count.min(history.len()));
        let mut excluded: HashSet<usize> = HashSet::new();

        while selected.len() < count {
            let best = history
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected.contains(i) && !excluded.contains(i))
                .max_by(|(_, a), (_, b)| {
                    a.propagated_value
                        .partial_cmp(&b.propagated_value)
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(i, _)| i);

            match best {
                Some(idx) => {
                    selected.push(idx);
                    // Exclude one-hop neighbors for diversity
                    excluded.insert(idx);
                    if let Some(parent) = history[idx].parent_idx {
                        excluded.insert(parent);
                    }
                    for (child_idx, node) in history.iter().enumerate() {
                        if node.parent_idx == Some(idx) {
                            excluded.insert(child_idx);
                        }
                    }
                }
                None => break,
            }
        }

        selected
    }

    /// Backpropagate values through the evaluation graph.
    ///
    /// Updates `propagated_value` on each node:
    /// `U_i = max(r_i, γ · max(U_child_j for j in children(i)))`
    ///
    /// Must be called after scores are updated. Processes in reverse index
    /// order so children are visited before parents (assuming children have
    /// higher indices than parents).
    fn update_propagated_values(&self, history: &mut [TesNode], gamma: f32) {
        for i in (0..history.len()).rev() {
            let own_score = history[i].score;

            let max_child_value = history
                .iter()
                .filter(|node| node.parent_idx == Some(i))
                .map(|node| node.propagated_value)
                .fold(0.0f32, f32::max);

            history[i].propagated_value = own_score.max(gamma * max_child_value);
        }
    }

    /// Compute RPUCG score for a single node.
    ///
    /// `score_i = U_i + λ · √(1 + |S|) / (1 + n_i)`
    ///
    /// Where:
    /// - `U_i` = propagated value (max of own score and discounted children)
    /// - `λ` = exploration weight
    /// - `|S|` = total visits across all nodes
    /// - `n_i` = visits to node i
    fn rpucg_score(&self, node: &TesNode, total_visits: usize, lambda: f32) -> f32 {
        let exploration =
            lambda * ((1.0 + total_visits as f32) / (1.0 + node.visit_count as f32)).sqrt();
        node.propagated_value + exploration
    }

    /// Select top-k nodes by RPUCG score, excluding one-hop neighbors.
    ///
    /// Unlike `select_inspirations` which uses only propagated_value for ranking,
    /// this method uses the full RPUCG formula with exploration bonus.
    /// Use this for bandit-guided selection, `select_inspirations` for greedy.
    fn select_rpucg(&self, history: &[TesNode], count: usize, lambda: f32) -> Vec<usize> {
        if history.is_empty() || count == 0 {
            return Vec::new();
        }

        let total_visits: usize = history.iter().map(|n| n.visit_count).sum();

        let mut selected: Vec<usize> = Vec::with_capacity(count.min(history.len()));
        let mut excluded: HashSet<usize> = HashSet::new();

        while selected.len() < count {
            let best = history
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected.contains(i) && !excluded.contains(i))
                .max_by(|(_, a), (_, b)| {
                    let sa = self.rpucg_score(a, total_visits, lambda);
                    let sb = self.rpucg_score(b, total_visits, lambda);
                    sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
                })
                .map(|(i, _)| i);

            match best {
                Some(idx) => {
                    selected.push(idx);
                    excluded.insert(idx);
                    if let Some(parent) = history[idx].parent_idx {
                        excluded.insert(parent);
                    }
                    for (child_idx, node) in history.iter().enumerate() {
                        if node.parent_idx == Some(idx) {
                            excluded.insert(child_idx);
                        }
                    }
                }
                None => break,
            }
        }

        selected
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(feature = "tes_loop")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bandit::BanditStrategy;
    use crate::speculative::types::{TesConfig, TesNode};

    /// Minimal TesLoop implementor for testing.
    struct MockTesLoop {
        config: TesConfig,
    }

    impl TesLoop for MockTesLoop {
        fn config(&self) -> &TesConfig {
            &self.config
        }
    }

    fn mock_loop() -> MockTesLoop {
        MockTesLoop {
            config: TesConfig::default(),
        }
    }

    #[test]
    fn tes_budget_default() {
        let tl = mock_loop();
        // 32 × 100 × 16 = 51_200
        assert_eq!(tl.budget(), 51_200);
    }

    #[test]
    fn tes_budget_custom() {
        let tl = MockTesLoop {
            config: TesConfig {
                global_width: 4,
                refinement_depth: 10,
                local_sample_size: 8,
                bandit_strategy: BanditStrategy::Rpucg {
                    gamma: 0.9,
                    lambda: 0.5,
                },
            },
        };
        assert_eq!(tl.budget(), 320);
    }

    #[test]
    fn select_inspirations_empty_history() {
        let tl = mock_loop();
        let result = tl.select_inspirations(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn select_inspirations_zero_count() {
        let tl = mock_loop();
        let nodes = vec![TesNode::new(vec![1], None)];
        let result = tl.select_inspirations(&nodes, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn select_inspirations_single_node() {
        let tl = mock_loop();
        let mut node = TesNode::new(vec![1], None);
        node.propagated_value = 0.9;
        let result = tl.select_inspirations(&[node], 1);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn select_inspirations_picks_highest_value() {
        let tl = mock_loop();
        let nodes = vec![
            {
                let mut n = TesNode::new(vec![1], None);
                n.propagated_value = 0.3;
                n
            },
            {
                let mut n = TesNode::new(vec![2], None);
                n.propagated_value = 0.9;
                n
            },
            {
                let mut n = TesNode::new(vec![3], None);
                n.propagated_value = 0.6;
                n
            },
        ];
        let result = tl.select_inspirations(&nodes, 1);
        assert_eq!(result, vec![1]); // Index of 0.9 value
    }

    #[test]
    fn select_inspirations_excludes_one_hop_neighbors() {
        let tl = mock_loop();
        // Node 0 (root) → Node 1 (child), Node 2 (child)
        // Node 3 (independent)
        let nodes = vec![
            {
                let mut n = TesNode::new(vec![0], None);
                n.propagated_value = 0.8;
                n
            },
            {
                let mut n = TesNode::new(vec![1], Some(0));
                n.propagated_value = 0.9;
                n
            },
            {
                let mut n = TesNode::new(vec![2], Some(0));
                n.propagated_value = 0.7;
                n
            },
            {
                let mut n = TesNode::new(vec![3], None);
                n.propagated_value = 0.5;
                n
            },
        ];
        // Select 2: node 1 (0.9) first → excludes node 1 (self) + node 0 (parent).
        // One-hop exclusion does NOT exclude siblings (node 2 has parent=0, but
        // node 0's children are excluded only when node 0 is the *selected* node).
        // Remaining: node 2 (0.7) and node 3 (0.5). Node 2 wins by value.
        let result = tl.select_inspirations(&nodes, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1); // Highest value
        assert_eq!(result[1], 2); // Sibling not excluded, higher than node 3
    }

    #[test]
    fn update_propagated_values_leaf_only() {
        let tl = mock_loop();
        let mut nodes = vec![{
            let mut n = TesNode::new(vec![0], None);
            n.score = 0.5;
            n
        }];
        tl.update_propagated_values(&mut nodes, 0.8);
        // Leaf: propagated_value = max(0.5, 0.8 * 0.0) = 0.5
        assert!((nodes[0].propagated_value - 0.5).abs() < 1e-6);
    }

    #[test]
    fn update_propagated_values_child_beats_parent() {
        let tl = mock_loop();
        let mut nodes = vec![
            {
                let mut n = TesNode::new(vec![0], None);
                n.score = 0.3;
                n
            },
            {
                let mut n = TesNode::new(vec![1], Some(0));
                n.score = 0.9;
                n
            },
        ];
        tl.update_propagated_values(&mut nodes, 0.8);
        // Node 1 (leaf): propagated = max(0.9, 0) = 0.9
        assert!((nodes[1].propagated_value - 0.9).abs() < 1e-6);
        // Node 0 (parent): propagated = max(0.3, 0.8 * 0.9) = max(0.3, 0.72) = 0.72
        assert!((nodes[0].propagated_value - 0.72).abs() < 1e-6);
    }

    #[test]
    fn update_propagated_values_parent_score_wins() {
        let tl = mock_loop();
        let mut nodes = vec![
            {
                let mut n = TesNode::new(vec![0], None);
                n.score = 0.9;
                n
            },
            {
                let mut n = TesNode::new(vec![1], Some(0));
                n.score = 0.3;
                n
            },
        ];
        tl.update_propagated_values(&mut nodes, 0.5);
        // Node 1: propagated = max(0.3, 0) = 0.3
        assert!((nodes[1].propagated_value - 0.3).abs() < 1e-6);
        // Node 0: propagated = max(0.9, 0.5 * 0.3) = max(0.9, 0.15) = 0.9
        assert!((nodes[0].propagated_value - 0.9).abs() < 1e-6);
    }

    #[test]
    fn rpucg_score_unvisited_high_exploration() {
        let tl = mock_loop();
        let node = TesNode::new(vec![1], None); // visit_count = 0, propagated_value = 0.0
        let score = tl.rpucg_score(&node, 100, 1.0);
        // λ * √((1 + 100) / (1 + 0)) = √101 ≈ 10.05
        assert!(score > 10.0);
    }

    #[test]
    fn rpucg_score_visited_lower_exploration() {
        let tl = mock_loop();
        let mut node = TesNode::new(vec![1], None);
        node.visit_count = 50;
        node.propagated_value = 0.7;
        let score = tl.rpucg_score(&node, 100, 1.0);
        // 0.7 + 1.0 * √(101 / 51) ≈ 0.7 + 1.408 ≈ 2.108
        assert!(score > 1.5 && score < 2.5);
    }

    #[test]
    fn select_rpucg_prefers_unvisited() {
        let tl = mock_loop();
        let nodes = vec![
            {
                let mut n = TesNode::new(vec![1], None);
                n.propagated_value = 0.9;
                n.visit_count = 100;
                n
            },
            {
                let mut n = TesNode::new(vec![2], None);
                n.propagated_value = 0.1;
                n.visit_count = 0; // Unvisited → huge exploration bonus
                n
            },
        ];
        let result = tl.select_rpucg(&nodes, 1, 1.0);
        assert_eq!(result, vec![1]); // Unvisited wins due to exploration bonus
    }
}
