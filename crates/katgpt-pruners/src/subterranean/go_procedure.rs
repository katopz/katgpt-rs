//! GoProcedure — Go game as a procedure graph (Plan 110, T6).
//!
//! Represents Go game flow as a directed graph where nodes are game phases
//! (opening, midgame, endgame, scoring, resigned) and edges are valid
//! transitions conditioned on board state.
//!
//! GOAT proof targets:
//! - 9×9 path enumeration completes in <1s
//! - Path count grows with game complexity
//! - Cost model shows increasing advantage with board size

use crate::subterranean::ProcedureGraph;
use crate::subterranean::types::{ProcedureEdge, ProcedureNode};

// ── Type alias ─────────────────────────────────────────────────

/// Adjacency list entry: (target_node_id, optional_condition).
type EdgeEntry = (u32, Option<String>);

/// Adjacency list: per-node list of outgoing edges.
type AdjacencyList = Vec<Vec<EdgeEntry>>;

// ── GoNode ─────────────────────────────────────────────────────

/// Node identifiers for the Go procedure graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GoNode {
    /// Game initialization, first stone placement.
    Start,
    /// Opening phase — corner and side star point approaches.
    Opening,
    /// Midgame — fighting, invasions, running battles.
    Midgame,
    /// Endgame — territory boundaries being finalized.
    Endgame,
    /// Ko rule dispute — immediate recapture prohibited.
    KoFight,
    /// Semeai (capturing race) — both groups in atari.
    CaptureRace,
    /// Dead stone removal and territory counting.
    Scoring,
    /// Terminal: one player resigned.
    Resigned,
    /// Terminal: Black wins by score margin.
    BlackWins,
    /// Terminal: White wins by score margin.
    WhiteWins,
    /// Terminal: Jigo (exact tie after komi).
    Draw,
}

impl GoNode {
    /// Numeric ID used as the graph node index.
    pub fn id(self) -> u32 {
        match self {
            Self::Start => 0,
            Self::Opening => 1,
            Self::Midgame => 2,
            Self::Endgame => 3,
            Self::KoFight => 4,
            Self::CaptureRace => 5,
            Self::Scoring => 6,
            Self::Resigned => 7,
            Self::BlackWins => 8,
            Self::WhiteWins => 9,
            Self::Draw => 10,
        }
    }

    /// Human-readable label for this node.
    pub fn label(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Opening => "opening",
            Self::Midgame => "midgame",
            Self::Endgame => "endgame",
            Self::KoFight => "ko_fight",
            Self::CaptureRace => "capture_race",
            Self::Scoring => "scoring",
            Self::Resigned => "resigned",
            Self::BlackWins => "black_wins",
            Self::WhiteWins => "white_wins",
            Self::Draw => "draw",
        }
    }

    /// All GoNode variants in canonical order.
    pub fn all() -> &'static [Self] {
        &[
            Self::Start,
            Self::Opening,
            Self::Midgame,
            Self::Endgame,
            Self::KoFight,
            Self::CaptureRace,
            Self::Scoring,
            Self::Resigned,
            Self::BlackWins,
            Self::WhiteWins,
            Self::Draw,
        ]
    }

    /// Lookup a node by its numeric ID.
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Start),
            1 => Some(Self::Opening),
            2 => Some(Self::Midgame),
            3 => Some(Self::Endgame),
            4 => Some(Self::KoFight),
            5 => Some(Self::CaptureRace),
            6 => Some(Self::Scoring),
            7 => Some(Self::Resigned),
            8 => Some(Self::BlackWins),
            9 => Some(Self::WhiteWins),
            10 => Some(Self::Draw),
            _ => None,
        }
    }

    /// Whether this node is terminal (no outgoing edges in game flow).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::BlackWins | Self::WhiteWins | Self::Draw)
    }
}

// ── GoProcedure ────────────────────────────────────────────────

/// Go game represented as a procedure graph.
///
/// Nodes encode discrete game phases; edges encode valid transitions
/// with optional conditions describing what triggers the phase change.
///
/// Graph topology (11 nodes, ~28 edges):
/// ```text
/// Start ──► Opening ──► Midgame ──► Endgame ──► Scoring ──► BlackWins
///              │           │           │                    ├─► WhiteWins
///              │           │           │                    └─► Draw
///              │           │           ├─► CaptureRace ──► Scoring
///              │           │           └─► Resigned ──────► BlackWins
///              │           ├─► KoFight ──┐                └─► WhiteWins
///              │           │             ├─► Midgame
///              │           │             └─► Endgame
///              │           ├─► CaptureRace ──► Midgame
///              │           │                ├─► Endgame
///              │           │                └─► Scoring
///              │           └─► Resigned
///              ├─► CaptureRace ──► Midgame
///              │                 ├─► Endgame
///              │                 └─► Scoring
///              └─► Resigned
/// ```
pub struct GoProcedure {
    nodes: Vec<ProcedureNode>,
    adjacency: Vec<Vec<(u32, Option<String>)>>,
    edge_count: usize,
    board_size: usize,
}

impl GoProcedure {
    /// Build the Go procedure graph for a given board size.
    ///
    /// The graph topology is identical regardless of board size, but
    /// `board_size` is stored as metadata for cost model estimation.
    /// Use 9 for tractable path enumeration; 19 for full game complexity.
    pub fn new(board_size: usize) -> Self {
        let nodes = Self::build_nodes();
        let (adjacency, edge_count) = Self::build_adjacency();
        Self {
            nodes,
            adjacency,
            edge_count,
            board_size,
        }
    }

    /// Convenience constructor for 9×9 board (tractable for enumeration).
    pub fn new_9x9() -> Self {
        Self::new(9)
    }

    /// Convenience constructor for 19×19 board (full game).
    pub fn new_19x19() -> Self {
        Self::new(19)
    }

    /// Board size this procedure was constructed for.
    pub fn board_size(&self) -> usize {
        self.board_size
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
        GoNode::all()
            .iter()
            .map(|&node| ProcedureNode {
                id: node.id(),
                label: node.label().to_string(),
                is_start: matches!(node, GoNode::Start),
                is_terminal: node.is_terminal(),
            })
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn build_adjacency() -> (AdjacencyList, usize) {
        let mut adj = vec![Vec::new(); GoNode::all().len()];
        let mut count = 0usize;

        let add = |adj: &mut Vec<Vec<(u32, Option<String>)>>,
                   count: &mut usize,
                   from: GoNode,
                   to: GoNode,
                   cond: Option<&str>| {
            adj[from.id() as usize].push((to.id(), cond.map(String::from)));
            *count += 1;
        };

        // Start → Opening (first stone placed)
        add(&mut adj, &mut count, GoNode::Start, GoNode::Opening, None);

        // Opening → Opening (continue joseki/fuseki)
        add(
            &mut adj,
            &mut count,
            GoNode::Opening,
            GoNode::Opening,
            Some("continue_joseki"),
        );

        // Opening → Midgame (major fight begins)
        add(
            &mut adj,
            &mut count,
            GoNode::Opening,
            GoNode::Midgame,
            Some("fight_starts"),
        );

        // Opening → CaptureRace (early capturing race from invasion)
        add(
            &mut adj,
            &mut count,
            GoNode::Opening,
            GoNode::CaptureRace,
            Some("early_invasion"),
        );

        // Opening → Resigned (rare: early resignation after blunder)
        add(
            &mut adj,
            &mut count,
            GoNode::Opening,
            GoNode::Resigned,
            Some("opening_blunder"),
        );

        // Midgame → Midgame (continue fighting)
        add(
            &mut adj,
            &mut count,
            GoNode::Midgame,
            GoNode::Midgame,
            Some("continue_fight"),
        );

        // Midgame → Endgame (territories mostly settled)
        add(
            &mut adj,
            &mut count,
            GoNode::Midgame,
            GoNode::Endgame,
            Some("territories_settled"),
        );

        // Midgame → KoFight (ko situation arises)
        add(
            &mut adj,
            &mut count,
            GoNode::Midgame,
            GoNode::KoFight,
            Some("ko_situation"),
        );

        // Midgame → CaptureRace (semeai starts)
        add(
            &mut adj,
            &mut count,
            GoNode::Midgame,
            GoNode::CaptureRace,
            Some("semeai_starts"),
        );

        // Midgame → Resigned (resignation during fight)
        add(
            &mut adj,
            &mut count,
            GoNode::Midgame,
            GoNode::Resigned,
            Some("resign_fight"),
        );

        // Endgame → Endgame (continue boundary plays)
        add(
            &mut adj,
            &mut count,
            GoNode::Endgame,
            GoNode::Endgame,
            Some("boundary_play"),
        );

        // Endgame → Scoring (both players pass)
        add(
            &mut adj,
            &mut count,
            GoNode::Endgame,
            GoNode::Scoring,
            Some("both_pass"),
        );

        // Endgame → CaptureRace (late capturing race)
        add(
            &mut adj,
            &mut count,
            GoNode::Endgame,
            GoNode::CaptureRace,
            Some("late_semeai"),
        );

        // Endgame → Resigned (resignation after losing endgame)
        add(
            &mut adj,
            &mut count,
            GoNode::Endgame,
            GoNode::Resigned,
            Some("lost_endgame"),
        );

        // KoFight → Midgame (ko resolved, continue fighting)
        add(
            &mut adj,
            &mut count,
            GoNode::KoFight,
            GoNode::Midgame,
            Some("ko_resolved"),
        );

        // KoFight → Endgame (ko resolved, game near end)
        add(
            &mut adj,
            &mut count,
            GoNode::KoFight,
            GoNode::Endgame,
            Some("ko_resolved_late"),
        );

        // CaptureRace → Midgame (race resolved, continue fighting)
        add(
            &mut adj,
            &mut count,
            GoNode::CaptureRace,
            GoNode::Midgame,
            Some("race_won"),
        );

        // CaptureRace → Endgame (race resolved, into endgame)
        add(
            &mut adj,
            &mut count,
            GoNode::CaptureRace,
            GoNode::Endgame,
            Some("race_settled"),
        );

        // CaptureRace → Scoring (race ends game)
        add(
            &mut adj,
            &mut count,
            GoNode::CaptureRace,
            GoNode::Scoring,
            Some("race_decisive"),
        );

        // Scoring → BlackWins
        add(
            &mut adj,
            &mut count,
            GoNode::Scoring,
            GoNode::BlackWins,
            Some("black_ahead"),
        );

        // Scoring → WhiteWins
        add(
            &mut adj,
            &mut count,
            GoNode::Scoring,
            GoNode::WhiteWins,
            Some("white_ahead"),
        );

        // Scoring → Draw (jigo)
        add(
            &mut adj,
            &mut count,
            GoNode::Scoring,
            GoNode::Draw,
            Some("jigo"),
        );

        // Resigned → BlackWins (white resigned)
        add(
            &mut adj,
            &mut count,
            GoNode::Resigned,
            GoNode::BlackWins,
            Some("white_resigned"),
        );

        // Resigned → WhiteWins (black resigned)
        add(
            &mut adj,
            &mut count,
            GoNode::Resigned,
            GoNode::WhiteWins,
            Some("black_resigned"),
        );

        (adj, count)
    }
}

impl Default for GoProcedure {
    fn default() -> Self {
        Self::new_9x9()
    }
}

impl ProcedureGraph for GoProcedure {
    type NodeId = u32;
    type Condition = String;

    fn start_node(&self) -> Self::NodeId {
        GoNode::Start.id()
    }

    fn terminal_nodes(&self) -> &[Self::NodeId] {
        static TERMINALS: [u32; 3] = [
            8,  // BlackWins
            9,  // WhiteWins
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
        match GoNode::from_id(node) {
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
    fn test_go_procedure_graph_structure() {
        let graph = GoProcedure::new_9x9();

        // 11 nodes total
        assert_eq!(graph.node_count(), 11);

        // 24 edges total
        assert_eq!(graph.edge_count(), 24);

        // Start node
        assert_eq!(graph.start_node(), GoNode::Start.id());

        // Terminal nodes
        assert_eq!(graph.terminal_nodes(), &[8, 9, 10]);

        // Board size stored correctly
        assert_eq!(graph.board_size(), 9);

        // Labels
        assert_eq!(graph.node_label(0), "start");
        assert_eq!(graph.node_label(8), "black_wins");
        assert_eq!(graph.node_label(9), "white_wins");
        assert_eq!(graph.node_label(10), "draw");
    }

    #[test]
    fn test_go_edges_from_start() {
        let graph = GoProcedure::new_9x9();

        let start_edges = graph.edges_from(GoNode::Start.id());
        assert_eq!(start_edges.len(), 1); // Start → Opening only

        let (target, cond) = &start_edges[0];
        assert_eq!(*target, GoNode::Opening.id());
        assert!(cond.is_none());
    }

    #[test]
    fn test_go_opening_has_many_exits() {
        let graph = GoProcedure::new_9x9();

        let opening_edges = graph.edges_from(GoNode::Opening.id());
        // Opening → Opening, Midgame, CaptureRace, Resigned = 4
        assert_eq!(opening_edges.len(), 4);
    }

    #[test]
    fn test_go_midgame_branching() {
        let graph = GoProcedure::new_9x9();

        let midgame_edges = graph.edges_from(GoNode::Midgame.id());
        // Midgame → Midgame, Endgame, KoFight, CaptureRace, Resigned = 5
        assert_eq!(midgame_edges.len(), 5);
    }

    #[test]
    fn test_go_terminal_nodes_have_no_outgoing() {
        let graph = GoProcedure::new_9x9();

        assert!(graph.edges_from(GoNode::BlackWins.id()).is_empty());
        assert!(graph.edges_from(GoNode::WhiteWins.id()).is_empty());
        assert!(graph.edges_from(GoNode::Draw.id()).is_empty());
    }

    #[test]
    fn test_go_scoring_branching() {
        let graph = GoProcedure::new_9x9();

        let scoring_edges = graph.edges_from(GoNode::Scoring.id());
        // Scoring → BlackWins, WhiteWins, Draw = 3
        assert_eq!(scoring_edges.len(), 3);
    }

    #[test]
    fn test_go_resigned_branching() {
        let graph = GoProcedure::new_9x9();

        let resigned_edges = graph.edges_from(GoNode::Resigned.id());
        // Resigned → BlackWins, WhiteWins = 2
        assert_eq!(resigned_edges.len(), 2);
    }

    #[test]
    fn test_go_path_enumeration() {
        let graph = GoProcedure::new_9x9();
        let enumerator = PathEnumerator::new(&graph, 50);
        let paths = enumerator.enumerate();

        assert!(!paths.is_empty(), "Go graph should have at least one path");

        // Every path must start at Start and end at a terminal
        for path in &paths {
            assert_eq!(path.path[0], GoNode::Start.id());
            let last = *path.end().unwrap();
            assert!(
                last == GoNode::BlackWins.id()
                    || last == GoNode::WhiteWins.id()
                    || last == GoNode::Draw.id(),
                "Path must end at terminal, got node {last}"
            );
        }

        // Count must match
        assert_eq!(enumerator.count_paths(), paths.len());
    }

    #[test]
    fn test_go_path_count_reasonable() {
        let graph = GoProcedure::new_9x9();
        let enumerator = PathEnumerator::new(&graph, 50);
        let count = enumerator.count_paths();

        // With 11 nodes and 24 edges, expect somewhere between 20-300 paths
        assert!(count > 10, "Expected >10 paths, got {count}");
        assert!(count < 1000, "Expected <1000 paths, got {count}");
    }

    #[test]
    fn test_go_node_enum_roundtrip() {
        for &node in GoNode::all() {
            let id = node.id();
            let recovered = GoNode::from_id(id);
            assert_eq!(recovered, Some(node));
        }
    }

    #[test]
    fn test_go_edges_method() {
        let graph = GoProcedure::new_9x9();
        let edges = graph.edges();

        assert_eq!(edges.len(), graph.edge_count());

        // Verify all edges reference valid nodes
        for edge in &edges {
            assert!(
                GoNode::from_id(edge.from).is_some(),
                "Invalid from node: {}",
                edge.from
            );
            assert!(
                GoNode::from_id(edge.to).is_some(),
                "Invalid to node: {}",
                edge.to
            );
        }
    }

    #[test]
    fn test_go_no_cycles_in_paths() {
        let graph = GoProcedure::new_9x9();
        let enumerator = PathEnumerator::new(&graph, 50);
        let paths = enumerator.enumerate();

        // Paths should NOT contain cycles (Opening↔Opening, Midgame↔Midgame, etc.)
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
    fn test_go_cost_model() {
        use crate::subterranean::cost_model::ProcedureCostModel;

        let graph = GoProcedure::new_9x9();
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

    #[test]
    fn test_go_19x19_same_topology() {
        let graph_9 = GoProcedure::new(9);
        let graph_19 = GoProcedure::new(19);

        // Same graph topology, different board size metadata
        assert_eq!(graph_9.node_count(), graph_19.node_count());
        assert_eq!(graph_9.edge_count(), graph_19.edge_count());
        assert_eq!(graph_9.board_size(), 9);
        assert_eq!(graph_19.board_size(), 19);
    }

    #[test]
    fn test_go_ko_fight_exits() {
        let graph = GoProcedure::new_9x9();

        let ko_edges = graph.edges_from(GoNode::KoFight.id());
        // KoFight → Midgame, Endgame = 2
        assert_eq!(ko_edges.len(), 2);
    }

    #[test]
    fn test_go_capture_race_exits() {
        let graph = GoProcedure::new_9x9();

        let race_edges = graph.edges_from(GoNode::CaptureRace.id());
        // CaptureRace → Midgame, Endgame, Scoring = 3
        assert_eq!(race_edges.len(), 3);
    }

    #[test]
    fn test_go_default_is_9x9() {
        let graph = GoProcedure::default();
        assert_eq!(graph.board_size(), 9);
    }
}
