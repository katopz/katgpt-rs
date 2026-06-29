//! BomberProcedure — Bomberman game as a procedure graph (Plan 110, T5).
//!
//! Represents Bomberman game flow as a directed graph where nodes are game
//! phases (start, moving, placing_bomb, waiting, explosion, etc.) and edges
//! are valid transitions conditioned on game events.
//!
//! GOAT proof targets:
//! - Path enumeration completes in <100ms for 12×12 grid
//! - Enumerated paths cover >90% of self-play game trajectories
//! - Cost model predicts correctly for Bomber complexity

use crate::subterranean::ProcedureGraph;
use crate::subterranean::types::{ProcedureEdge, ProcedureNode};

// ── Type alias ─────────────────────────────────────────────────

/// Adjacency list entry: (target_node_id, optional_condition).
type EdgeEntry = (u32, Option<String>);

/// Adjacency list: per-node list of outgoing edges.
type AdjacencyList = Vec<Vec<EdgeEntry>>;

// ── BomberNode ─────────────────────────────────────────────────

/// Node identifiers for the Bomberman procedure graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BomberNode {
    /// Game initialization phase.
    Start,
    /// Player is navigating the grid.
    Moving,
    /// Player places a bomb at current position.
    PlacingBomb,
    /// Waiting for bomb timer to expire.
    Waiting,
    /// Bomb detonation in progress.
    Explosion,
    /// Chain reaction from blast propagation.
    ChainReaction,
    /// Checking which players survived the blast.
    CheckAlive,
    /// Player picked up a power-up.
    PowerUp,
    /// Player is actively dodging blast radius.
    Evading,
    /// Terminal: exactly one survivor remains.
    GameOver,
    /// Terminal: no survivors (mutual elimination).
    Draw,
}

impl BomberNode {
    /// Numeric ID used as the graph node index.
    pub fn id(self) -> u32 {
        match self {
            Self::Start => 0,
            Self::Moving => 1,
            Self::PlacingBomb => 2,
            Self::Waiting => 3,
            Self::Explosion => 4,
            Self::ChainReaction => 5,
            Self::CheckAlive => 6,
            Self::PowerUp => 7,
            Self::Evading => 8,
            Self::GameOver => 9,
            Self::Draw => 10,
        }
    }

    /// Human-readable label for this node.
    pub fn label(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Moving => "moving",
            Self::PlacingBomb => "placing_bomb",
            Self::Waiting => "waiting",
            Self::Explosion => "explosion",
            Self::ChainReaction => "chain_reaction",
            Self::CheckAlive => "check_alive",
            Self::PowerUp => "power_up",
            Self::Evading => "evading",
            Self::GameOver => "game_over",
            Self::Draw => "draw",
        }
    }

    /// All BomberNode variants in canonical order.
    pub fn all() -> &'static [Self] {
        &[
            Self::Start,
            Self::Moving,
            Self::PlacingBomb,
            Self::Waiting,
            Self::Explosion,
            Self::ChainReaction,
            Self::CheckAlive,
            Self::PowerUp,
            Self::Evading,
            Self::GameOver,
            Self::Draw,
        ]
    }

    /// Lookup a node by its numeric ID.
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Start),
            1 => Some(Self::Moving),
            2 => Some(Self::PlacingBomb),
            3 => Some(Self::Waiting),
            4 => Some(Self::Explosion),
            5 => Some(Self::ChainReaction),
            6 => Some(Self::CheckAlive),
            7 => Some(Self::PowerUp),
            8 => Some(Self::Evading),
            9 => Some(Self::GameOver),
            10 => Some(Self::Draw),
            _ => None,
        }
    }

    /// Whether this node is terminal (no outgoing edges in game flow).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::GameOver | Self::Draw)
    }
}

// ── BomberProcedure ────────────────────────────────────────────

/// Bomberman game represented as a procedure graph.
///
/// Nodes encode discrete game phases; edges encode valid transitions
/// with optional conditions describing what triggers the transition.
///
/// Graph topology (11 nodes, ~20 edges):
/// ```text
/// Start ──► Moving ──► PlacingBomb ──► Moving
///   │          │    └─► Waiting ──► Explosion ──► ChainReaction ──┐
///   │          │                                     ◄────────────┤
///   │          ├─► Evading ──► Moving                             │
///   │          ├─► PowerUp ──► Moving                CheckAlive ──┤
///   │          ├─► GameOver                                    │  │
///   │          └─► PlacingBomb ──► Evading                      │  │
///   └─► PlacingBomb                    CheckAlive ──► Moving ───┘
///                                  CheckAlive ──► PowerUp
///                                  CheckAlive ──► GameOver
///                                  CheckAlive ──► Draw
///                                  PowerUp ──► PlacingBomb
///                                  Evading ──► GameOver
/// ```
pub struct BomberProcedure {
    nodes: Vec<ProcedureNode>,
    adjacency: Vec<Vec<(u32, Option<String>)>>,
    edge_count: usize,
}

impl BomberProcedure {
    /// Build the Bomberman procedure graph.
    ///
    /// Constructs all nodes and edges based on the canonical game flow.
    /// The graph is immutable after construction.
    pub fn new() -> Self {
        let nodes = Self::build_nodes();
        let (adjacency, edge_count) = Self::build_adjacency();
        Self {
            nodes,
            adjacency,
            edge_count,
        }
    }

    /// Get a reference to the underlying node descriptors.
    pub fn nodes(&self) -> &[ProcedureNode] {
        &self.nodes
    }

    /// Get all edges in the graph.
    pub fn edges(&self) -> Vec<ProcedureEdge> {
        let mut edges = Vec::with_capacity(self.edge_count);
        for (from_id, targets) in self.adjacency.iter().enumerate() {
            for (to_id, condition) in targets {
                edges.push(ProcedureEdge {
                    from: from_id as u32,
                    to: *to_id,
                    condition: condition.clone(),
                });
            }
        }
        edges
    }

    // ── Construction helpers ────────────────────────────────────

    fn build_nodes() -> Vec<ProcedureNode> {
        BomberNode::all()
            .iter()
            .map(|&node| ProcedureNode {
                id: node.id(),
                label: node.label().to_string(),
                is_start: matches!(node, BomberNode::Start),
                is_terminal: node.is_terminal(),
            })
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn build_adjacency() -> (AdjacencyList, usize) {
        let mut adj = vec![Vec::new(); BomberNode::all().len()];
        let mut count = 0usize;

        let add = |adj: &mut Vec<Vec<(u32, Option<String>)>>,
                   count: &mut usize,
                   from: BomberNode,
                   to: BomberNode,
                   cond: Option<&str>| {
            adj[from.id() as usize].push((to.id(), cond.map(String::from)));
            *count += 1;
        };

        // Start → Moving (game begins)
        add(
            &mut adj,
            &mut count,
            BomberNode::Start,
            BomberNode::Moving,
            None,
        );

        // Start → PlacingBomb (aggressive opening)
        add(
            &mut adj,
            &mut count,
            BomberNode::Start,
            BomberNode::PlacingBomb,
            Some("aggressive_start"),
        );

        // Moving → Moving (continue exploring)
        add(
            &mut adj,
            &mut count,
            BomberNode::Moving,
            BomberNode::Moving,
            Some("explore"),
        );

        // Moving → PlacingBomb (player places bomb)
        add(
            &mut adj,
            &mut count,
            BomberNode::Moving,
            BomberNode::PlacingBomb,
            Some("player_action"),
        );

        // Moving → Evading (threat detected)
        add(
            &mut adj,
            &mut count,
            BomberNode::Moving,
            BomberNode::Evading,
            Some("threat_detected"),
        );

        // Moving → PowerUp (found power-up on ground)
        add(
            &mut adj,
            &mut count,
            BomberNode::Moving,
            BomberNode::PowerUp,
            Some("power_up_found"),
        );

        // Moving → GameOver (walked into blast radius)
        add(
            &mut adj,
            &mut count,
            BomberNode::Moving,
            BomberNode::GameOver,
            Some("eliminated"),
        );

        // PlacingBomb → Moving (flee after placing)
        add(
            &mut adj,
            &mut count,
            BomberNode::PlacingBomb,
            BomberNode::Moving,
            Some("flee"),
        );

        // PlacingBomb → Waiting (stay and wait for explosion)
        add(
            &mut adj,
            &mut count,
            BomberNode::PlacingBomb,
            BomberNode::Waiting,
            Some("bomb_timer_start"),
        );

        // PlacingBomb → Evading (need to dodge own bomb blast)
        add(
            &mut adj,
            &mut count,
            BomberNode::PlacingBomb,
            BomberNode::Evading,
            Some("own_bomb_threat"),
        );

        // Waiting → Explosion (timer expires)
        add(
            &mut adj,
            &mut count,
            BomberNode::Waiting,
            BomberNode::Explosion,
            Some("bomb_timer"),
        );

        // Explosion → ChainReaction (blast hits another bomb)
        add(
            &mut adj,
            &mut count,
            BomberNode::Explosion,
            BomberNode::ChainReaction,
            Some("blast_propagation"),
        );

        // Explosion → CheckAlive (assess damage from blast)
        add(
            &mut adj,
            &mut count,
            BomberNode::Explosion,
            BomberNode::CheckAlive,
            Some("blast_complete"),
        );

        // ChainReaction → Explosion (further chain triggers)
        add(
            &mut adj,
            &mut count,
            BomberNode::ChainReaction,
            BomberNode::Explosion,
            Some("chain_trigger"),
        );

        // ChainReaction → CheckAlive (chain ends, assess damage)
        add(
            &mut adj,
            &mut count,
            BomberNode::ChainReaction,
            BomberNode::CheckAlive,
            Some("chain_complete"),
        );

        // CheckAlive → Moving (survivors continue)
        add(
            &mut adj,
            &mut count,
            BomberNode::CheckAlive,
            BomberNode::Moving,
            Some("survived"),
        );

        // CheckAlive → PowerUp (blast revealed a power-up)
        add(
            &mut adj,
            &mut count,
            BomberNode::CheckAlive,
            BomberNode::PowerUp,
            Some("power_up_revealed"),
        );

        // CheckAlive → GameOver (exactly one survivor)
        add(
            &mut adj,
            &mut count,
            BomberNode::CheckAlive,
            BomberNode::GameOver,
            Some("last_survivor"),
        );

        // CheckAlive → Draw (all eliminated simultaneously)
        add(
            &mut adj,
            &mut count,
            BomberNode::CheckAlive,
            BomberNode::Draw,
            Some("mutual_elimination"),
        );

        // PowerUp → Moving (continue with power-up)
        add(
            &mut adj,
            &mut count,
            BomberNode::PowerUp,
            BomberNode::Moving,
            Some("power_up_applied"),
        );

        // PowerUp → PlacingBomb (gained extra bomb capacity)
        add(
            &mut adj,
            &mut count,
            BomberNode::PowerUp,
            BomberNode::PlacingBomb,
            Some("extra_bomb"),
        );

        // Evading → Moving (escape successful)
        add(
            &mut adj,
            &mut count,
            BomberNode::Evading,
            BomberNode::Moving,
            Some("escape_success"),
        );

        // Evading → GameOver (didn't escape blast in time)
        add(
            &mut adj,
            &mut count,
            BomberNode::Evading,
            BomberNode::GameOver,
            Some("escape_failed"),
        );

        (adj, count)
    }
}

impl Default for BomberProcedure {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcedureGraph for BomberProcedure {
    type NodeId = u32;
    type Condition = String;

    fn start_node(&self) -> Self::NodeId {
        BomberNode::Start.id()
    }

    fn terminal_nodes(&self) -> &[Self::NodeId] {
        static TERMINALS: [u32; 2] = [
            9,  // GameOver
            10, // Draw
        ];
        &TERMINALS
    }

    fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
        match self.adjacency.get(node as usize) {
            Some(edges) => edges,
            None => &[],
        }
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.edge_count
    }

    fn node_label(&self, node: Self::NodeId) -> &str {
        match BomberNode::from_id(node) {
            Some(n) => n.label(),
            None => "unknown",
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subterranean::path_enumerator::PathEnumerator;

    #[test]
    fn test_bomber_procedure_graph_structure() {
        let graph = BomberProcedure::new();

        // 11 nodes total
        assert_eq!(graph.node_count(), 11);

        // 23 edges total
        assert_eq!(graph.edge_count(), 23);

        // Start node
        assert_eq!(graph.start_node(), BomberNode::Start.id());

        // Terminal nodes
        assert_eq!(graph.terminal_nodes(), &[9, 10]);

        // Labels
        assert_eq!(graph.node_label(0), "start");
        assert_eq!(graph.node_label(9), "game_over");
        assert_eq!(graph.node_label(10), "draw");
    }

    #[test]
    fn test_bomber_edges_from_start() {
        let graph = BomberProcedure::new();

        let start_edges = graph.edges_from(BomberNode::Start.id());
        assert_eq!(start_edges.len(), 2);

        let targets: Vec<u32> = start_edges.iter().map(|(t, _)| *t).collect();
        assert!(targets.contains(&BomberNode::Moving.id()));
        assert!(targets.contains(&BomberNode::PlacingBomb.id()));
    }

    #[test]
    fn test_bomber_terminal_nodes_have_outgoing_edges() {
        // GameOver has no outgoing edges in our model
        let graph = BomberProcedure::new();

        let game_over_edges = graph.edges_from(BomberNode::GameOver.id());
        assert!(
            game_over_edges.is_empty(),
            "GameOver should have no outgoing edges"
        );

        let draw_edges = graph.edges_from(BomberNode::Draw.id());
        assert!(draw_edges.is_empty(), "Draw should have no outgoing edges");
    }

    #[test]
    fn test_bomber_moving_has_many_exits() {
        let graph = BomberProcedure::new();

        let moving_edges = graph.edges_from(BomberNode::Moving.id());
        // Moving has: self-loop, PlacingBomb, Evading, PowerUp, GameOver = 5
        assert_eq!(moving_edges.len(), 5);
    }

    #[test]
    fn test_bomber_check_alive_branching() {
        let graph = BomberProcedure::new();

        let check_edges = graph.edges_from(BomberNode::CheckAlive.id());
        // CheckAlive → Moving, PowerUp, GameOver, Draw = 4
        assert_eq!(check_edges.len(), 4);
    }

    #[test]
    fn test_bomber_path_enumeration() {
        let graph = BomberProcedure::new();
        let enumerator = PathEnumerator::new(&graph, 50);
        let paths = enumerator.enumerate();

        // Should find multiple paths to terminal nodes
        assert!(
            !paths.is_empty(),
            "Bomber graph should have at least one path"
        );

        // Every path must start at Start and end at a terminal
        for path in &paths {
            assert_eq!(path.path[0], BomberNode::Start.id());
            let last = *path.end().unwrap();
            assert!(
                last == BomberNode::GameOver.id() || last == BomberNode::Draw.id(),
                "Path must end at terminal, got node {last}"
            );
        }

        // Count must match
        assert_eq!(enumerator.count_paths(), paths.len());
    }

    #[test]
    fn test_bomber_path_count_reasonable() {
        let graph = BomberProcedure::new();
        let enumerator = PathEnumerator::new(&graph, 50);
        let count = enumerator.count_paths();

        // With 11 nodes and 23 edges, expect somewhere between 20-200 paths
        assert!(count > 10, "Expected >10 paths, got {count}");
        assert!(count < 500, "Expected <500 paths, got {count}");
    }

    #[test]
    fn test_bomber_node_enum_roundtrip() {
        for &node in BomberNode::all() {
            let id = node.id();
            let recovered = BomberNode::from_id(id);
            assert_eq!(recovered, Some(node));
        }
    }

    #[test]
    fn test_bomber_edges_method() {
        let graph = BomberProcedure::new();
        let edges = graph.edges();

        assert_eq!(edges.len(), graph.edge_count());

        // Verify all edges reference valid nodes
        for edge in &edges {
            assert!(
                BomberNode::from_id(edge.from).is_some(),
                "Invalid from node: {}",
                edge.from
            );
            assert!(
                BomberNode::from_id(edge.to).is_some(),
                "Invalid to node: {}",
                edge.to
            );
        }
    }

    #[test]
    fn test_bomber_chain_reaction_cycle_detected() {
        let graph = BomberProcedure::new();
        let enumerator = PathEnumerator::new(&graph, 50);
        let paths = enumerator.enumerate();

        // Paths should NOT contain cycles (Explosion <-> ChainReaction)
        for path in &paths {
            let mut visited = std::collections::HashSet::new();
            for &node in &path.path {
                assert!(
                    visited.insert(node),
                    "Cycle detected: node {node} appears twice in path"
                );
            }
        }
    }

    #[test]
    fn test_bomber_cost_model() {
        use crate::subterranean::cost_model::ProcedureCostModel;

        let graph = BomberProcedure::new();
        let model = ProcedureCostModel::from_graph(&graph, 50);

        assert!(model.is_some());
        let m = model.unwrap();
        assert_eq!(m.node_count, 11);
        assert!(
            m.path_count > 10,
            "Should have >10 paths, got {}",
            m.path_count
        );

        // Cost ratio should be in paper's ballpark
        let ratio = m.cost_ratio_vs_in_context();
        assert!(ratio > 50.0, "Cost ratio should be >50×, got {ratio}");
    }
}
