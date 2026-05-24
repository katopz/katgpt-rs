//! Subterranean Procedure Compilation (Plan 110, Research 074).
//!
//! Compiles procedural workflows into model weights by representing procedures
//! as directed graphs, enumerating paths, and generating structured training data.
//!
//! Paper: arXiv:2605.22502 — "Compiling Workflows into Weights"
//! Key finding: 87–98% of frontier quality at 128–462× lower cost.
//!
//! # Architecture
//!
//! ```text
//! ProcedureGraph (directed graph)     ← Declarative, offline
//!     ↓ enumerate_paths()
//!     ↓ sample_trajectories()
//! PathSampler (training data gen)     ← Structured data from paths
//!     ↓ Vec<Trajectory>
//!     ↓ to_bandit_sessions()
//! BanditPruner (online learning)      ← Existing: learns from trajectories
//!     ↓ freeze()
//! FrozenBandit (compiled knowledge)   ← Existing: no orchestrator needed
//! ```
//!
//! # Feature Gate
//!
//! `subterranean = ["bandit"]` — requires the `bandit` feature.
//!
//! # Usage
//!
//! ```ignore
//! use katgpt_rs::pruners::subterranean::{
//!     ProcedureGraph, PathEnumerator, PathSampler, ProcedureCostModel,
//! };
//!
//! // 1. Define or use an existing procedure graph
//! let graph = BomberProcedure::new();
//!
//! // 2. Enumerate paths
//! let enumerator = PathEnumerator::new(&graph, 50);
//! let paths = enumerator.enumerate();
//!
//! // 3. Generate training samples
//! let mut sampler = PathSampler::new(&graph, 50, 42);
//! let samples = sampler.generate_samples(100, &SampleFilter::All);
//!
//! // 4. Estimate cost
//! let cost = ProcedureCostModel::from_graph(&graph, 50);
//! println!("Cost ratio: {:.0}×", cost.unwrap().cost_ratio_vs_in_context());
//! ```

pub mod bandit_bridge;
pub mod bomber_procedure;
pub mod cost_model;
pub mod game_bridge;
pub mod go_procedure;
pub mod path_enumerator;
pub mod path_sampler;
pub mod training_mode;
pub mod types;

// ── Re-exports ─────────────────────────────────────────────────

pub use bandit_bridge::{
    DecisionPoint, TrajectoryBanditSummary, extract_all_decision_points, extract_decision_points,
    graph_to_global_session, graph_trajectories_to_sessions, graph_trajectories_to_sessions_seeded,
};
pub use bomber_procedure::{BomberNode, BomberProcedure};
pub use cost_model::{ComplexityTier, ProcedureCostModel};
pub use game_bridge::{BridgeError, NodeStateMapping, ProcedureGameState, TrajectoryValidator};
pub use go_procedure::{GoNode, GoProcedure};
pub use path_enumerator::PathEnumerator;
pub use path_sampler::{PathSampler, Sample, SampleFilter};
pub use training_mode::{SubterraneanTrainingMode, TrainingBudget};
pub use types::{ProcedureEdge, ProcedureNode, Trajectory};

// ── ProcedureGraph Trait ───────────────────────────────────────

use std::fmt;
use std::hash::Hash;

/// Directed graph representation of a procedural workflow.
///
/// Paper: F = (N, E, n₀, T) where N=nodes, E=edges with conditions,
/// n₀=start, T=terminal nodes.
///
/// Implementors define the topology of a procedure (game rules, validation
/// pipelines, workflow logic) as a directed graph. Paths through the graph
/// represent distinct scenarios the model must learn to handle.
///
/// # Type Parameters
///
/// - `NodeId`: Identifier for graph nodes (typically `u32` or an enum)
/// - `Condition`: Optional edge condition type
///
/// # Example
///
/// ```ignore
/// struct MyProcedure;
///
/// impl ProcedureGraph for MyProcedure {
///     type NodeId = u32;
///     type Condition = String;
///
///     fn start_node(&self) -> Self::NodeId { 0 }
///     fn terminal_nodes(&self) -> &[Self::NodeId] { &[3] }
///     fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
///         // ...
///     }
///     fn node_count(&self) -> usize { 4 }
///     fn edge_count(&self) -> usize { 3 }
///     fn node_label(&self, node: Self::NodeId) -> &str { "..." }
/// }
/// ```
pub trait ProcedureGraph {
    /// Identifier type for graph nodes.
    ///
    /// Must be `Copy` for efficient path enumeration, `Eq + Hash` for
    /// visited-set tracking, and `Debug` for logging.
    type NodeId: Copy + Eq + Hash + fmt::Debug;

    /// Condition type for guarded edges.
    ///
    /// Represents the condition under which an edge transition fires.
    /// Use `()` for unconditional graphs or `String` for descriptive conditions.
    type Condition: fmt::Debug;

    /// The entry point of the procedure graph.
    ///
    /// All valid paths start from this node.
    fn start_node(&self) -> Self::NodeId;

    /// Terminal (sink) nodes that end a procedure execution.
    ///
    /// Paths that reach any terminal node are considered complete.
    /// The returned slice must be non-empty for path enumeration to work.
    fn terminal_nodes(&self) -> &[Self::NodeId];

    /// Outgoing edges from the given node.
    ///
    /// Each edge is a tuple of `(target_node, optional_condition)`.
    /// Returns an empty slice for nodes with no outgoing edges (sinks).
    fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)];

    /// Total number of nodes in the graph.
    fn node_count(&self) -> usize;

    /// Total number of edges in the graph.
    fn edge_count(&self) -> usize;

    /// Human-readable label for a node.
    ///
    /// Used for logging, path descriptions, and training sample generation.
    fn node_label(&self, node: Self::NodeId) -> &str;
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the trait is object-safe for basic usage.
    /// (Note: generic associated types prevent full dyn usage, but
    /// concrete implementations should compile.)
    struct MinimalGraph;

    impl ProcedureGraph for MinimalGraph {
        type NodeId = u32;
        type Condition = String;

        fn start_node(&self) -> Self::NodeId {
            0
        }
        fn terminal_nodes(&self) -> &[Self::NodeId] {
            &[1]
        }
        fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
            static EMPTY: [(u32, Option<String>); 0] = [];
            static A: [(u32, Option<String>); 1] = [(1, None)];
            match node {
                0 => &A,
                _ => &EMPTY,
            }
        }
        fn node_count(&self) -> usize {
            2
        }
        fn edge_count(&self) -> usize {
            1
        }
        fn node_label(&self, node: Self::NodeId) -> &str {
            match node {
                0 => "start",
                1 => "end",
                _ => "?",
            }
        }
    }

    #[test]
    fn test_trait_compiles() {
        let graph = MinimalGraph;
        assert_eq!(graph.start_node(), 0);
        assert_eq!(graph.terminal_nodes(), &[1]);
        assert_eq!(graph.edges_from(0).len(), 1);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.node_label(0), "start");
        assert_eq!(graph.node_label(1), "end");
    }

    #[test]
    fn test_minimal_graph_path_enumeration() {
        let graph = MinimalGraph;
        let enumerator = PathEnumerator::new(&graph, 10);
        let paths = enumerator.enumerate();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].path, vec![0, 1]);
        assert_eq!(enumerator.count_paths(), 1);
    }

    #[test]
    fn test_re_exports_accessible() {
        // Verify all key types are accessible through the module
        let _node = ProcedureNode::start(0, "test");
        let _edge = ProcedureEdge::new(0, 1);
        let _traj: Trajectory<u32> = Trajectory::new();

        let mode = SubterraneanTrainingMode::default();
        assert!(mode.is_full_finetune());

        let tier = ComplexityTier::Simple;
        assert!(format!("{tier}").contains("Simple"));
    }

    #[test]
    fn test_bomber_procedure_compiles() {
        let graph = BomberProcedure::new();
        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_go_procedure_compiles() {
        let graph = GoProcedure::new_9x9();
        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);
        assert_eq!(graph.board_size(), 9);
    }

    #[test]
    fn test_cost_model_from_bomber() {
        let graph = BomberProcedure::new();
        let model = ProcedureCostModel::from_graph(&graph, 50);

        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.cost_ratio_vs_in_context() > 0.0);
    }

    #[test]
    fn test_bandit_bridge_from_graph() {
        let graph = BomberProcedure::new();
        let enumerator = PathEnumerator::new(&graph, 20);
        let paths = enumerator.enumerate();

        let points = extract_all_decision_points(&paths, &graph);
        assert!(!points.is_empty(), "Should have decision points");
    }
}
