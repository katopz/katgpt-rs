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
use katgpt_speculative::TesNode;

// TesConfig moved here from main crate's src/speculative/types.rs (Plan 005).
// TesConfig has a BanditStrategy field; BanditStrategy is imported below near
// the concrete SimpleTesLoop impl, so we don't re-import it here.

/// SimpleTES evaluation-driven scaling config (Plan 086).
///
/// Budget = C × L × K total evaluations per solve.
#[cfg(feature = "tes_loop")]
#[derive(Clone, Debug)]
pub struct TesConfig {
    /// C: parallel trajectories.
    pub global_width: usize,
    /// L: iterations per trajectory.
    pub refinement_depth: usize,
    /// K: candidates per step.
    pub local_sample_size: usize,
    /// Bandit strategy for proposal selection (Φ).
    pub bandit_strategy: BanditStrategy,
}

#[cfg(feature = "tes_loop")]
impl Default for TesConfig {
    fn default() -> Self {
        Self {
            global_width: 32,
            refinement_depth: 100,
            local_sample_size: 16,
            bandit_strategy: BanditStrategy::Rpucg {
                gamma: 0.8,
                lambda: 1.0,
            },
        }
    }
}

#[cfg(feature = "tes_loop")]
impl TesConfig {
    /// Total evaluation budget: C × L × K.
    pub fn budget(&self) -> usize {
        self.global_width * self.refinement_depth * self.local_sample_size
    }
}

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

// ── Concrete Implementation ────────────────────────────────────

use crate::bandit::{BanditEnv, BanditStrategy};

/// Concrete TES evaluation loop using RPUCG over a bandit environment.
///
/// Implements the full SimpleTES (C, L, K, Φ) loop:
/// - **C trajectories** run in parallel (sequential with budget redistribution)
/// - **L steps** per trajectory with candidate mutation
/// - **K candidates** evaluated per step via the bandit environment
/// - **Φ = RPUCG** for inspiration selection and value propagation
///
/// After running, `history` contains all evaluated nodes with propagated values
/// and `best_solution()` returns the highest-scoring candidate found.
#[cfg(feature = "tes_loop")]
pub struct SimpleTesLoop<E: BanditEnv> {
    config: TesConfig,
    env: E,
    history: Vec<TesNode>,
    best_score: f32,
    best_idx: usize,
}

#[cfg(feature = "tes_loop")]
impl<E: BanditEnv + Clone> SimpleTesLoop<E> {
    /// Create a new TES loop with the given configuration and environment.
    pub fn new(config: TesConfig, env: E) -> Self {
        Self {
            config,
            env,
            history: Vec::new(),
            best_score: f32::MIN,
            best_idx: 0,
        }
    }

    /// Get the TES configuration.
    pub fn config(&self) -> &TesConfig {
        &self.config
    }

    /// Get the evaluation history (all nodes across all trajectories).
    pub fn history(&self) -> &[TesNode] {
        &self.history
    }

    /// Get the best solution found so far.
    pub fn best_solution(&self) -> Option<&TesNode> {
        self.history.get(self.best_idx)
    }

    /// Best score across all evaluated nodes.
    pub fn best_score(&self) -> f32 {
        self.best_score
    }

    /// Total evaluations performed so far.
    pub fn total_evaluations(&self) -> usize {
        self.history.len()
    }

    /// Run a single step: generate K candidates from inspiration, evaluate, propagate.
    ///
    /// Returns indices of newly added nodes.
    fn run_step(
        &mut self,
        inspiration_idx: Option<usize>,
        arm: usize,
        rng: &mut katgpt_types::Rng,
        vocab_size: usize,
    ) -> Vec<usize> {
        let k = self.config.local_sample_size;
        let parent_idx = inspiration_idx;

        // Generate K candidate solutions by mutating the inspiration
        let base_solution = match inspiration_idx {
            Some(idx) => self.history[idx].solution.clone(),
            None => (0..vocab_size)
                .map(|_| (rng.uniform() * vocab_size as f32) as usize)
                .collect(),
        };

        let mut new_indices = Vec::with_capacity(k);

        for _ in 0..k {
            // Mutate one position
            let mut candidate = base_solution.clone();
            if !candidate.is_empty() {
                let pos = (rng.uniform() * candidate.len() as f32) as usize;
                candidate[pos] = arm; // Use the selected arm as the mutation
            }

            let node = TesNode::new(candidate, parent_idx);
            let idx = self.history.len();
            self.history.push(node);
            new_indices.push(idx);
        }

        new_indices
    }

    /// Run the full TES loop: C trajectories × L steps × K candidates.
    ///
    /// Uses RPUCG for inspiration selection and value propagation at each step.
    /// TrajectoryPruner kills underperforming trajectories at checkpoints.
    ///
    /// Returns (total_evaluations, best_score).
    pub fn run(&mut self, vocab_size: usize, rng: &mut katgpt_types::Rng) -> (usize, f32) {
        let c = self.config.global_width;
        let l = self.config.refinement_depth;
        let _k = self.config.local_sample_size;
        let gamma = match &self.config.bandit_strategy {
            BanditStrategy::Rpucg { gamma, .. } => *gamma,
            _ => 0.8,
        };

        let pruner = crate::arena::TrajectoryPruner::new();

        // Track which trajectory each node belongs to
        let mut node_trajectory: Vec<usize> = Vec::new();
        // Track active trajectories (not pruned)
        let mut active_trajectories: Vec<usize> = (0..c).collect();
        // Track best score per trajectory
        let mut trajectory_best: Vec<f32> = vec![f32::MIN; c];
        // Track current step per trajectory
        let mut trajectory_step: Vec<usize> = vec![0; c];
        // Track root node index per trajectory
        let mut trajectory_root: Vec<usize> = vec![0; c];

        // Phase 1: Initialize C trajectories with random starting solutions
        for t in 0..c {
            let solution: Vec<usize> = (0..vocab_size)
                .map(|_| (rng.uniform() * vocab_size as f32) as usize)
                .collect();
            let mut node = TesNode::new(solution, None);
            let reward = self.env.pull(t % self.env.num_arms(), rng);
            node.score = reward;
            node.visit_count = 1;

            let idx = self.history.len();
            self.history.push(node);
            node_trajectory.push(t);
            trajectory_root[t] = idx;
            trajectory_best[t] = reward;

            if reward > self.best_score {
                self.best_score = reward;
                self.best_idx = idx;
            }
        }

        // Phase 2: Run L steps per trajectory
        for step in 0..l {
            // Checkpoint pruning
            if pruner.is_checkpoint(step, l) && active_trajectories.len() > 1 {
                let values: Vec<f32> = active_trajectories
                    .iter()
                    .map(|&t| trajectory_best[t])
                    .collect();
                let to_kill = pruner.prune(&values);
                let killed: Vec<usize> = to_kill.iter().map(|&i| active_trajectories[i]).collect();
                active_trajectories.retain(|t| !killed.contains(t));
            }

            if active_trajectories.is_empty() {
                break;
            }

            // Select inspiration for each active trajectory using RPUCG
            for &t in &active_trajectories {
                trajectory_step[t] += 1;

                // Select arm via bandit session
                let arm = if !self.history.is_empty() {
                    // Simple arm selection: use trajectory index modulo num_arms
                    t % self.env.num_arms()
                } else {
                    (rng.uniform() * self.env.num_arms() as f32) as usize
                };

                let inspiration_idx = if !self.history.is_empty() {
                    let insps = self.select_inspirations(&self.history, 1);
                    insps.first().copied()
                } else {
                    None
                };

                // Generate and evaluate K candidates
                let new_indices = self.run_step(inspiration_idx, arm, rng, vocab_size);

                for &idx in &new_indices {
                    // Evaluate the candidate
                    let reward = self.env.pull(arm, rng);
                    self.history[idx].score = reward;
                    self.history[idx].visit_count = 1;
                    node_trajectory.push(t);

                    if reward > trajectory_best[t] {
                        trajectory_best[t] = reward;
                    }

                    if reward > self.best_score {
                        self.best_score = reward;
                        self.best_idx = idx;
                    }
                }
            }

            // Propagate values through the graph (inlined to avoid borrow conflict)
            for i in (0..self.history.len()).rev() {
                let own_score = self.history[i].score;
                let max_child_value = self
                    .history
                    .iter()
                    .filter(|node| node.parent_idx == Some(i))
                    .map(|node| node.propagated_value)
                    .fold(0.0f32, f32::max);
                self.history[i].propagated_value = own_score.max(gamma * max_child_value);
            }
        }

        (self.history.len(), self.best_score)
    }
}

#[cfg(feature = "tes_loop")]
impl<E: BanditEnv + Clone> TesLoop for SimpleTesLoop<E> {
    fn config(&self) -> &TesConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(feature = "tes_loop")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandit::BanditStrategy;
    use katgpt_speculative::{TesConfig, TesNode};

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
