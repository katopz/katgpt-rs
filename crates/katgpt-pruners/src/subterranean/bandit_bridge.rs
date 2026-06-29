//! BanditBridge — Convert procedure trajectories to BanditSession training data.
//!
//! Plan 110, T8: Integration — PathSampler → BanditPruner conversion.
//!
//! Maps graph path decisions into bandit arm-pull episodes where:
//! - Each step in a trajectory is a decision point
//! - Arms = valid outgoing edges at that node
//! - Optimal arm = the edge actually taken in the trajectory
//! - Reward = 1.0 for correct choice, 0.0 otherwise
//!
//! This replaces the manual template proposer with structured graph-based data,
//! enabling bandit training directly from procedure graph topology.

use crate::bandit::{BanditEnv, BanditEvent, BanditResult, BanditSession, BanditStrategy};
use crate::subterranean::ProcedureGraph;
use crate::subterranean::types::Trajectory;
use katgpt_types::Rng;

// ── DecisionPoint ──────────────────────────────────────────────

/// A single decision point extracted from a trajectory.
///
/// Represents one "arm pull" opportunity: given a node with N outgoing
/// edges, which one was chosen? This maps directly to a bandit episode.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionPoint<NodeId> {
    /// The node where the decision was made.
    pub from_node: NodeId,
    /// Valid next nodes (arms) available at this decision point.
    pub arms: Vec<NodeId>,
    /// The node actually chosen (optimal arm).
    pub chosen: NodeId,
    /// Index of the chosen node in the `arms` vector.
    pub chosen_index: usize,
}

impl<NodeId: Copy + Eq + std::fmt::Debug> DecisionPoint<NodeId> {
    /// Whether the chosen index is valid (within bounds of arms).
    pub fn is_valid(&self) -> bool {
        self.chosen_index < self.arms.len() && self.arms[self.chosen_index] == self.chosen
    }

    /// Branching factor at this decision point.
    pub fn branching_factor(&self) -> usize {
        self.arms.len()
    }

    /// Number of alternative (non-chosen) actions.
    pub fn alternative_count(&self) -> usize {
        self.arms.len().saturating_sub(1)
    }
}

// ── DecisionPointEnv ───────────────────────────────────────────

/// Bandit environment wrapping a single decision point.
///
/// Each arm represents one of the valid outgoing edges. The optimal arm
/// is the one matching the trajectory's actual choice. Pulling the optimal
/// arm yields reward 1.0; any other arm yields 0.0.
pub struct DecisionPointEnv {
    num_arms: usize,
    optimal_arm: usize,
}

impl DecisionPointEnv {
    /// Create a deterministic env for a single decision.
    pub fn new(num_arms: usize, optimal_arm: usize) -> Self {
        Self {
            num_arms,
            optimal_arm,
        }
    }

    /// Create from a decision point.
    pub fn from_decision_point<N: Copy + Eq + std::fmt::Debug>(dp: &DecisionPoint<N>) -> Self {
        Self {
            num_arms: dp.arms.len(),
            optimal_arm: dp.chosen_index,
        }
    }
}

impl BanditEnv for DecisionPointEnv {
    fn pull(&self, arm: usize, _rng: &mut Rng) -> f32 {
        match arm == self.optimal_arm {
            true => 1.0,
            false => 0.0,
        }
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        match arm == self.optimal_arm {
            true => 1.0,
            false => 0.0,
        }
    }

    fn optimal_reward(&self) -> f32 {
        1.0
    }

    fn num_arms(&self) -> usize {
        self.num_arms
    }

    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}

// ── TrajectoryBanditSummary ────────────────────────────────────

/// Summary of bandit conversion results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrajectoryBanditSummary {
    /// Number of trajectories processed.
    pub trajectory_count: usize,
    /// Total decision points extracted across all trajectories.
    pub total_decision_points: usize,
    /// Average branching factor across all decision points.
    pub avg_branching_factor: f64,
    /// Distribution of branching factors: (factor, count).
    pub branching_histogram: Vec<(usize, usize)>,
}

impl TrajectoryBanditSummary {
    /// Build a summary from extracted decision points.
    ///
    /// Uses `arms.len()` directly instead of `branching_factor()` to avoid
    /// requiring `NodeId: Copy + Eq + Debug` bounds on this method.
    pub fn from_decision_points<NodeId>(
        points: &[DecisionPoint<NodeId>],
        trajectory_count: usize,
    ) -> Self {
        let total_decision_points = points.len();
        let avg_branching_factor = match total_decision_points {
            0 => 0.0,
            _ => {
                points.iter().map(|dp| dp.arms.len()).sum::<usize>() as f64
                    / total_decision_points as f64
            }
        };

        let mut histogram = std::collections::HashMap::new();
        for dp in points {
            *histogram.entry(dp.arms.len()).or_insert(0usize) += 1;
        }
        let mut branching_histogram: Vec<(usize, usize)> = histogram.into_iter().collect();
        branching_histogram.sort_by_key(|(factor, _)| *factor);

        Self {
            trajectory_count,
            total_decision_points,
            avg_branching_factor,
            branching_histogram,
        }
    }
}

// ── Conversion Functions ───────────────────────────────────────

/// Extract all decision points from a single trajectory.
///
/// Each step where the trajectory chose among multiple edges produces
/// one `DecisionPoint`. Terminal nodes produce no decision points.
pub fn extract_decision_points<G: ProcedureGraph>(
    trajectory: &Trajectory<G::NodeId>,
    graph: &G,
) -> Vec<DecisionPoint<G::NodeId>>
where
    G::NodeId: Copy + Eq,
{
    let mut points = Vec::new();

    for i in 0..trajectory.path.len().saturating_sub(1) {
        let current = trajectory.path[i];
        let edges = graph.edges_from(current);
        let arms: Vec<G::NodeId> = edges.iter().map(|(next, _)| *next).collect();

        // Skip if no alternatives (dead end or terminal)
        if arms.is_empty() {
            continue;
        }

        let chosen = trajectory.path[i + 1];

        // Find the index of the chosen arm
        let chosen_index = match arms.iter().position(|&a| a == chosen) {
            Some(idx) => idx,
            None => continue, // Trajectory took an edge not in graph — skip
        };

        points.push(DecisionPoint {
            from_node: current,
            arms,
            chosen,
            chosen_index,
        });
    }

    points
}

/// Extract decision points from multiple trajectories.
pub fn extract_all_decision_points<G: ProcedureGraph>(
    trajectories: &[Trajectory<G::NodeId>],
    graph: &G,
) -> Vec<DecisionPoint<G::NodeId>>
where
    G::NodeId: Copy + Eq,
{
    let mut all_points = Vec::new();
    for trajectory in trajectories {
        all_points.extend(extract_decision_points(trajectory, graph));
    }
    all_points
}

/// Convert trajectories into BanditSessions for training.
///
/// Creates one `BanditSession` per decision point, using `DecisionPointEnv`.
/// Each session runs `episodes_per_point` episodes with the given strategy.
///
/// Returns `(sessions_results, summary)` where:
/// - `sessions_results` contains `(events, result)` for each session
/// - `summary` provides aggregate statistics
pub fn graph_trajectories_to_sessions<G: ProcedureGraph>(
    trajectories: &[Trajectory<G::NodeId>],
    graph: &G,
    strategy: BanditStrategy,
    episodes_per_point: usize,
    rng: &mut Rng,
) -> (
    Vec<(Vec<BanditEvent>, BanditResult)>,
    TrajectoryBanditSummary,
)
where
    G::NodeId: Copy + Eq + std::fmt::Debug,
{
    let points = extract_all_decision_points(trajectories, graph);
    let summary = TrajectoryBanditSummary::from_decision_points(&points, trajectories.len());

    let mut results = Vec::with_capacity(points.len());

    for dp in &points {
        let env = DecisionPointEnv::from_decision_point(dp);
        let session = BanditSession::new(env, strategy.clone());
        let session_result = session.run(episodes_per_point, rng);
        results.push(session_result);
    }

    (results, summary)
}

/// Convert trajectories into BanditSessions with shared RNG seed per session.
///
/// Useful for reproducible training. Each session gets its own RNG seeded
/// from the base seed plus decision point index.
pub fn graph_trajectories_to_sessions_seeded<G: ProcedureGraph>(
    trajectories: &[Trajectory<G::NodeId>],
    graph: &G,
    strategy: BanditStrategy,
    episodes_per_point: usize,
    base_seed: u64,
) -> (
    Vec<(Vec<BanditEvent>, BanditResult)>,
    TrajectoryBanditSummary,
)
where
    G::NodeId: Copy + Eq + std::fmt::Debug,
{
    let points = extract_all_decision_points(trajectories, graph);
    let summary = TrajectoryBanditSummary::from_decision_points(&points, trajectories.len());

    let mut results = Vec::with_capacity(points.len());

    for (i, dp) in points.iter().enumerate() {
        let env = DecisionPointEnv::from_decision_point(dp);
        let session = BanditSession::new(env, strategy.clone());
        let mut local_rng = Rng::new(base_seed.wrapping_add(i as u64));
        let session_result = session.run(episodes_per_point, &mut local_rng);
        results.push(session_result);
    }

    (results, summary)
}

/// Create a single aggregated BanditSession for all decision points.
///
/// Unlike `graph_trajectories_to_sessions` which creates one session per
/// decision point, this creates a single session with arms = union of all
/// unique edges across the graph. This is useful for global bandit training.
pub fn graph_to_global_session<G: ProcedureGraph<NodeId = u32>>(
    graph: &G,
    strategy: BanditStrategy,
    episodes: usize,
    rng: &mut Rng,
) -> (Vec<BanditEvent>, BanditResult) {
    // Collect all unique edges as arms
    let mut optimal_arm = 0usize;

    // Enumerate all edges, pick first edge leading to terminal as "optimal"
    for node_id in 0..graph.node_count() {
        let node = node_id as u32;
        for (next, _) in graph.edges_from(node) {
            if graph.terminal_nodes().contains(next) && optimal_arm == 0 {
                // Mark this edge as optimal (first terminal-reaching edge found)
                optimal_arm = 1; // Will be set properly below
            }
        }
    }

    // Count total arms
    let mut num_arms = 0usize;
    let mut first_terminal_arm = 0usize;
    for node_id in 0..graph.node_count() {
        let node = node_id as u32;
        for (next, _) in graph.edges_from(node) {
            if graph.terminal_nodes().contains(next) && first_terminal_arm == 0 {
                first_terminal_arm = num_arms;
            }
            num_arms += 1;
        }
    }

    let env = GlobalProcedureEnv {
        num_arms,
        optimal_arm: first_terminal_arm,
    };

    let session = BanditSession::new(env, strategy);
    session.run(episodes, rng)
}

// ── GlobalProcedureEnv ─────────────────────────────────────────

/// Bandit environment for the entire graph (all edges as arms).
struct GlobalProcedureEnv {
    num_arms: usize,
    optimal_arm: usize,
}

impl BanditEnv for GlobalProcedureEnv {
    fn pull(&self, arm: usize, _rng: &mut Rng) -> f32 {
        match arm == self.optimal_arm {
            true => 1.0,
            false => 0.0,
        }
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        match arm == self.optimal_arm {
            true => 1.0,
            false => 0.0,
        }
    }

    fn optimal_reward(&self) -> f32 {
        1.0
    }

    fn num_arms(&self) -> usize {
        self.num_arms
    }

    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal linear graph: A(0) -> B(1) -> C(2, terminal).
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
            static A: [(u32, Option<String>); 1] = [(1, None)];
            static B: [(u32, Option<String>); 1] = [(2, None)];
            match node {
                0 => &A,
                1 => &B,
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
    fn test_extract_decision_points_linear() {
        let graph = LinearGraph;
        let mut trajectory = Trajectory::from_start(0u32);
        trajectory.push(1, None);
        trajectory.push(2, None);

        let points = extract_decision_points(&trajectory, &graph);

        // Two steps: 0->1, 1->2
        assert_eq!(points.len(), 2);

        // Step 0: at A, chose B (index 0 among [B])
        assert_eq!(points[0].from_node, 0);
        assert_eq!(points[0].arms, vec![1]);
        assert_eq!(points[0].chosen, 1);
        assert_eq!(points[0].chosen_index, 0);

        // Step 1: at B, chose C (index 0 among [C])
        assert_eq!(points[1].from_node, 1);
        assert_eq!(points[1].arms, vec![2]);
        assert_eq!(points[1].chosen, 2);
    }

    #[test]
    fn test_decision_point_is_valid() {
        let dp = DecisionPoint {
            from_node: 0u32,
            arms: vec![1, 2],
            chosen: 1,
            chosen_index: 0,
        };
        assert!(dp.is_valid());
        assert_eq!(dp.branching_factor(), 2);
        assert_eq!(dp.alternative_count(), 1);
    }

    #[test]
    fn test_decision_point_invalid_index() {
        let dp = DecisionPoint {
            from_node: 0u32,
            arms: vec![1, 2],
            chosen: 2,
            chosen_index: 0, // Wrong index
        };
        assert!(!dp.is_valid());
    }

    /// Diamond graph: A(0) -> B(1), A(0) -> C(2), both -> D(3, terminal).
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
            static A: [(u32, Option<String>); 2] = [(1, None), (2, None)];
            static B: [(u32, Option<String>); 1] = [(3, None)];
            static C: [(u32, Option<String>); 1] = [(3, None)];
            match node {
                0 => &A,
                1 => &B,
                2 => &C,
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
    fn test_extract_decision_points_diamond() {
        let graph = DiamondGraph;

        // Path: A -> B -> D
        let mut traj = Trajectory::from_start(0u32);
        traj.push(1, None);
        traj.push(3, None);

        let points = extract_decision_points(&traj, &graph);

        assert_eq!(points.len(), 2);

        // At A: arms = [B, C], chose B (index 0)
        assert_eq!(points[0].from_node, 0);
        assert_eq!(points[0].arms, vec![1, 2]);
        assert_eq!(points[0].chosen, 1);
        assert_eq!(points[0].chosen_index, 0);

        // At B: arms = [D], chose D (index 0)
        assert_eq!(points[1].from_node, 1);
        assert_eq!(points[1].arms, vec![3]);
    }

    #[test]
    fn test_extract_all_decision_points() {
        let graph = DiamondGraph;

        let mut traj1 = Trajectory::from_start(0u32);
        traj1.push(1, None);
        traj1.push(3, None);

        let mut traj2 = Trajectory::from_start(0u32);
        traj2.push(2, None);
        traj2.push(3, None);

        let points = extract_all_decision_points(&[traj1, traj2], &graph);

        // 2 trajectories × 2 decision points each = 4
        assert_eq!(points.len(), 4);
    }

    #[test]
    fn test_decision_point_env_optimal() {
        let env = DecisionPointEnv::new(3, 1);

        assert_eq!(env.num_arms(), 3);
        assert_eq!(env.optimal_arm(), 1);
        assert_eq!(env.optimal_reward(), 1.0);
        assert_eq!(env.expected_reward(0), 0.0);
        assert_eq!(env.expected_reward(1), 1.0);
        assert_eq!(env.expected_reward(2), 0.0);
    }

    #[test]
    fn test_decision_point_env_pull() {
        let env = DecisionPointEnv::new(3, 1);
        let mut rng = Rng::new(42);

        // Optimal arm always returns 1.0
        assert_eq!(env.pull(1, &mut rng), 1.0);

        // Non-optimal arms always return 0.0
        assert_eq!(env.pull(0, &mut rng), 0.0);
        assert_eq!(env.pull(2, &mut rng), 0.0);
    }

    #[test]
    fn test_graph_trajectories_to_sessions() {
        let graph = DiamondGraph;

        let mut traj = Trajectory::from_start(0u32);
        traj.push(1, None);
        traj.push(3, None);

        let mut rng = Rng::new(42);
        let (results, summary) =
            graph_trajectories_to_sessions(&[traj], &graph, BanditStrategy::Ucb1, 10, &mut rng);

        // 2 decision points → 2 sessions
        assert_eq!(results.len(), 2);
        assert_eq!(summary.trajectory_count, 1);
        assert_eq!(summary.total_decision_points, 2);

        // Each session should have found the optimal arm
        for (_events, result) in &results {
            assert_eq!(result.total_episodes, 10);
            assert!(
                result.total_reward > 0.0,
                "Session should have positive reward"
            );
        }
    }

    #[test]
    fn test_graph_trajectories_to_sessions_seeded() {
        let graph = DiamondGraph;

        let mut traj = Trajectory::from_start(0u32);
        traj.push(1, None);
        traj.push(3, None);

        let (results1, _) = graph_trajectories_to_sessions_seeded(
            &[traj.clone()],
            &graph,
            BanditStrategy::Ucb1,
            10,
            42,
        );

        let (results2, _) =
            graph_trajectories_to_sessions_seeded(&[traj], &graph, BanditStrategy::Ucb1, 10, 42);

        // Same seed should produce same results
        assert_eq!(results1.len(), results2.len());
        for ((_, r1), (_, r2)) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.total_reward, r2.total_reward);
        }
    }

    #[test]
    fn test_summary_branching_histogram() {
        let points = vec![
            DecisionPoint {
                from_node: 0u32,
                arms: vec![1, 2],
                chosen: 1,
                chosen_index: 0,
            },
            DecisionPoint {
                from_node: 1u32,
                arms: vec![3],
                chosen: 3,
                chosen_index: 0,
            },
            DecisionPoint {
                from_node: 2u32,
                arms: vec![4, 5, 6],
                chosen: 5,
                chosen_index: 1,
            },
        ];

        let summary = TrajectoryBanditSummary::from_decision_points(&points, 1);

        assert_eq!(summary.total_decision_points, 3);
        assert_eq!(summary.trajectory_count, 1);

        // Average: (2 + 1 + 3) / 3 = 2.0
        assert!((summary.avg_branching_factor - 2.0).abs() < 0.001);

        // Histogram: 1→1, 2→1, 3→1
        assert_eq!(summary.branching_histogram, vec![(1, 1), (2, 1), (3, 1)]);
    }

    #[test]
    fn test_summary_empty() {
        let summary = TrajectoryBanditSummary::from_decision_points::<u32>(&[], 0);
        assert_eq!(summary.total_decision_points, 0);
        assert_eq!(summary.avg_branching_factor, 0.0);
    }

    #[test]
    fn test_global_session() {
        let graph = LinearGraph;
        let mut rng = Rng::new(42);

        let (_events, result) = graph_to_global_session(&graph, BanditStrategy::Ucb1, 20, &mut rng);

        assert_eq!(result.total_episodes, 20);
        assert!(result.total_reward > 0.0);
    }

    #[test]
    fn test_extract_decision_points_terminal_only() {
        let graph = LinearGraph;

        // Single-node trajectory (just the start)
        let traj = Trajectory::from_start(0u32);

        let points = extract_decision_points(&traj, &graph);
        assert!(
            points.is_empty(),
            "Single-node trajectory has no decision points"
        );
    }

    #[test]
    fn test_extract_decision_points_dead_end() {
        /// Graph with a dead-end node.
        struct DeadEndGraph;

        impl ProcedureGraph for DeadEndGraph {
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
                static A: [(u32, Option<String>); 1] = [(1, None)];
                // Node 1 has no edges (dead end, not terminal)
                match node {
                    0 => &A,
                    _ => &EMPTY,
                }
            }
            fn node_count(&self) -> usize {
                3
            }
            fn edge_count(&self) -> usize {
                1
            }
            fn node_label(&self, node: Self::NodeId) -> &str {
                match node {
                    0 => "A",
                    1 => "B",
                    _ => "?",
                }
            }
        }

        let graph = DeadEndGraph;

        // Path goes to dead end
        let mut traj = Trajectory::from_start(0u32);
        traj.push(1, None);

        let points = extract_decision_points(&traj, &graph);

        // One decision point at A, but B has no outgoing edges
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].from_node, 0);
    }
}
