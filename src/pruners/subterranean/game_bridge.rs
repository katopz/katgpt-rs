//! ProcedureGameState — bridge between declarative ProcedureGraph and runtime GameState.
//!
//! Allows generating training data from graph traversal, then validating
//! at runtime via GameState. The bridge maps graph nodes to concrete game
//! states and vice versa.
//!
//! Plan 110, T7: Integration — ProcedureGraph ↔ existing GameState trait bridge.

use std::fmt;

use crate::pruners::subterranean::ProcedureGraph;

// ── BridgeError ────────────────────────────────────────────────

/// Errors that can occur during graph↔state mapping.
#[derive(Debug, Clone)]
pub enum BridgeError {
    /// No graph node corresponds to the given game state.
    NoMatchingNode(String),
    /// No game state corresponds to the given graph node.
    NoMatchingState(String),
    /// Round-trip consistency check failed.
    RoundTripFailed {
        original_node: String,
        recovered_node: String,
    },
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMatchingNode(desc) => write!(f, "No matching node for state: {desc}"),
            Self::NoMatchingState(desc) => write!(f, "No matching state for node: {desc}"),
            Self::RoundTripFailed {
                original_node,
                recovered_node,
            } => write!(f, "Round-trip failed: {original_node} != {recovered_node}"),
        }
    }
}

impl std::error::Error for BridgeError {}

// ── NodeStateMapping ───────────────────────────────────────────

/// A single mapping between a graph node and a game state description.
/// Used for validation and debugging of the bridge.
#[derive(Debug, Clone)]
pub struct NodeStateMapping<NodeId, StateDesc> {
    /// The graph node identifier.
    pub node: NodeId,
    /// A human-readable description of the corresponding game state.
    pub state_description: StateDesc,
    /// Whether this node is a terminal state.
    pub is_terminal: bool,
}

// ── ProcedureGameState ─────────────────────────────────────────

/// Bridge trait connecting declarative ProcedureGraph with runtime GameState.
///
/// Implementors provide bidirectional mapping between graph nodes and game
/// states, enabling:
/// - Training data generation from graph path traversal
/// - Runtime validation of graph-predicted trajectories
/// - Round-trip consistency checking
///
/// # Type Parameters
/// - `NodeId`: Graph node identifier (from `ProcedureGraph`)
/// - `State`: Concrete game state type
/// - `Condition`: Edge condition type (from `ProcedureGraph`)
///
/// # Example
///
/// ```ignore
/// use katgpt_rs::pruners::subterranean::game_bridge::ProcedureGameState;
///
/// struct MyGameBridge { /* ... */ }
///
/// impl ProcedureGraph for MyGameBridge {
///     type NodeId = u32;
///     type Condition = String;
///     // ... implement trait methods
/// }
///
/// impl ProcedureGameState for MyGameBridge {
///     type State = MyGameState;
///     type StateDescriptor = String;
///
///     fn node_to_state(&self, node: Self::NodeId) -> Option<Self::State> {
///         // Convert graph node to game state
///     }
///
///     fn state_to_node(&self, state: &Self::State) -> Option<Self::NodeId> {
///         // Convert game state to graph node
///     }
/// }
/// ```
pub trait ProcedureGameState: ProcedureGraph {
    /// The concrete game state type this bridge maps to.
    type State;

    /// A human-readable descriptor for game states (for logging/debugging).
    type StateDescriptor: fmt::Debug + Clone;

    /// Map a graph node to its corresponding game state.
    ///
    /// Returns `None` if the node doesn't have a direct state mapping
    /// (e.g., abstract phase nodes that don't correspond to a single state).
    fn node_to_state(&self, node: Self::NodeId) -> Option<Self::State>;

    /// Map a game state to the closest corresponding graph node.
    ///
    /// Returns `None` if the state doesn't match any known graph phase.
    fn state_to_node(&self, state: &Self::State) -> Option<Self::NodeId>;

    /// Describe a game state in human-readable form.
    ///
    /// Useful for logging and debugging trajectory mismatches.
    fn describe_state(&self, state: &Self::State) -> Self::StateDescriptor;

    /// Validate round-trip consistency for a terminal node.
    ///
    /// Terminal nodes should map to terminal states, and the round-trip
    /// `node → state → node` should be identity for terminal nodes.
    fn validate_terminal_roundtrip(&self, node: Self::NodeId) -> Result<(), BridgeError>
    where
        Self::NodeId: fmt::Debug + Eq,
    {
        let state = self
            .node_to_state(node)
            .ok_or_else(|| BridgeError::NoMatchingState(format!("{node:?}")))?;

        let desc = self.describe_state(&state);
        let recovered = self
            .state_to_node(&state)
            .ok_or_else(|| BridgeError::NoMatchingNode(format!("{desc:?}")))?;

        match recovered == node {
            true => Ok(()),
            false => Err(BridgeError::RoundTripFailed {
                original_node: format!("{node:?}"),
                recovered_node: format!("{recovered:?}"),
            }),
        }
    }

    /// Get all node↔state mappings for this bridge.
    ///
    /// Default implementation iterates all terminal nodes.
    /// Override for custom mapping sets.
    fn mappings(&self) -> Vec<NodeStateMapping<Self::NodeId, Self::StateDescriptor>>
    where
        Self::NodeId: Copy,
    {
        let mut result = Vec::new();

        for &terminal in self.terminal_nodes() {
            if let Some(state) = self.node_to_state(terminal) {
                result.push(NodeStateMapping {
                    node: terminal,
                    state_description: self.describe_state(&state),
                    is_terminal: true,
                });
            }
        }

        result
    }
}

// ── TrajectoryValidator ────────────────────────────────────────

/// Validates that a procedure graph trajectory is consistent with
/// runtime game state transitions.
pub struct TrajectoryValidator<'a, B: ProcedureGameState> {
    bridge: &'a B,
}

impl<'a, B: ProcedureGameState> TrajectoryValidator<'a, B> {
    /// Create a new validator for the given bridge.
    pub fn new(bridge: &'a B) -> Self {
        Self { bridge }
    }

    /// Validate that each step in a trajectory is a valid game transition.
    ///
    /// Returns `Ok(())` if all steps are valid, or the first error encountered.
    pub fn validate_trajectory(&self, path: &[B::NodeId]) -> Result<(), BridgeError>
    where
        B::NodeId: fmt::Debug + Eq + Copy,
    {
        match path.len() {
            0 => Ok(()),
            1 => {
                // Single-node path: just verify it's a valid node
                let _ = self
                    .bridge
                    .node_to_state(path[0])
                    .ok_or_else(|| BridgeError::NoMatchingState(format!("{:?}", path[0])))?;
                Ok(())
            }
            _ => {
                for window in path.windows(2) {
                    let from = window[0];
                    let to = window[1];

                    // Check that the edge exists in the graph
                    let edges = self.bridge.edges_from(from);
                    let edge_exists = edges.iter().any(|(next, _)| *next == to);

                    match edge_exists {
                        true => {}
                        false => {
                            return Err(BridgeError::NoMatchingState(format!(
                                "No edge from {:?} to {:?}",
                                from, to
                            )));
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Validate all terminal nodes have consistent round-trip mappings.
    ///
    /// Returns the number of terminal nodes that passed validation,
    /// or the first error encountered.
    pub fn validate_all_terminals(&self) -> Result<usize, BridgeError>
    where
        B::NodeId: fmt::Debug + Eq + Copy,
    {
        let terminals = self.bridge.terminal_nodes();
        let mut validated = 0;

        for &terminal in terminals {
            self.bridge.validate_terminal_roundtrip(terminal)?;
            validated += 1;
        }

        Ok(validated)
    }

    /// Map a trajectory of node IDs to a sequence of state descriptions.
    ///
    /// Returns `None` for any node that doesn't have a state mapping.
    pub fn trajectory_to_descriptions(&self, path: &[B::NodeId]) -> Vec<Option<B::StateDescriptor>>
    where
        B::NodeId: Copy,
    {
        path.iter()
            .map(|&node| {
                self.bridge
                    .node_to_state(node)
                    .map(|state| self.bridge.describe_state(&state))
            })
            .collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal bridge for testing: 3-node linear graph with string states.
    struct TestBridge;

    impl ProcedureGraph for TestBridge {
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
                0 => "start",
                1 => "middle",
                2 => "end",
                _ => "?",
            }
        }
    }

    #[derive(Debug, Clone)]
    struct TestState {
        phase: String,
    }

    impl ProcedureGameState for TestBridge {
        type State = TestState;
        type StateDescriptor = String;

        fn node_to_state(&self, node: Self::NodeId) -> Option<Self::State> {
            match node {
                0 => Some(TestState {
                    phase: "start".into(),
                }),
                1 => Some(TestState {
                    phase: "middle".into(),
                }),
                2 => Some(TestState {
                    phase: "end".into(),
                }),
                _ => None,
            }
        }

        fn state_to_node(&self, state: &Self::State) -> Option<Self::NodeId> {
            match state.phase.as_str() {
                "start" => Some(0),
                "middle" => Some(1),
                "end" => Some(2),
                _ => None,
            }
        }

        fn describe_state(&self, state: &Self::State) -> Self::StateDescriptor {
            state.phase.clone()
        }
    }

    #[test]
    fn test_bridge_node_to_state() {
        let bridge = TestBridge;

        let state = bridge.node_to_state(0).unwrap();
        assert_eq!(state.phase, "start");

        let state = bridge.node_to_state(2).unwrap();
        assert_eq!(state.phase, "end");

        assert!(bridge.node_to_state(99).is_none());
    }

    #[test]
    fn test_bridge_state_to_node() {
        let bridge = TestBridge;

        let node = bridge.state_to_node(&TestState {
            phase: "start".into(),
        });
        assert_eq!(node, Some(0));

        let node = bridge.state_to_node(&TestState {
            phase: "end".into(),
        });
        assert_eq!(node, Some(2));

        let node = bridge.state_to_node(&TestState {
            phase: "unknown".into(),
        });
        assert!(node.is_none());
    }

    #[test]
    fn test_bridge_round_trip() {
        let bridge = TestBridge;

        for id in 0..3u32 {
            let state = bridge.node_to_state(id).unwrap();
            let recovered = bridge.state_to_node(&state).unwrap();
            assert_eq!(recovered, id, "Round-trip failed for node {id}");
        }
    }

    #[test]
    fn test_bridge_validate_terminal_roundtrip() {
        let bridge = TestBridge;
        let result = bridge.validate_terminal_roundtrip(2);
        assert!(
            result.is_ok(),
            "Terminal node 2 should round-trip: {result:?}"
        );
    }

    #[test]
    fn test_bridge_mappings() {
        let bridge = TestBridge;
        let mappings = bridge.mappings();

        assert_eq!(mappings.len(), 1); // One terminal node
        assert_eq!(mappings[0].node, 2);
        assert_eq!(mappings[0].state_description, "end");
        assert!(mappings[0].is_terminal);
    }

    #[test]
    fn test_validator_valid_trajectory() {
        let bridge = TestBridge;
        let validator = TrajectoryValidator::new(&bridge);

        let result = validator.validate_trajectory(&[0, 1, 2]);
        assert!(result.is_ok(), "Valid trajectory should pass: {result:?}");
    }

    #[test]
    fn test_validator_invalid_edge() {
        let bridge = TestBridge;
        let validator = TrajectoryValidator::new(&bridge);

        // No edge from 0 to 2 directly
        let result = validator.validate_trajectory(&[0, 2]);
        assert!(result.is_err(), "Invalid edge should fail");
    }

    #[test]
    fn test_validator_empty_path() {
        let bridge = TestBridge;
        let validator = TrajectoryValidator::new(&bridge);

        let result = validator.validate_trajectory(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validator_single_node() {
        let bridge = TestBridge;
        let validator = TrajectoryValidator::new(&bridge);

        let result = validator.validate_trajectory(&[1]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validator_all_terminals() {
        let bridge = TestBridge;
        let validator = TrajectoryValidator::new(&bridge);

        let count = validator.validate_all_terminals().unwrap();
        assert_eq!(count, 1); // One terminal node
    }

    #[test]
    fn test_validator_trajectory_descriptions() {
        let bridge = TestBridge;
        let validator = TrajectoryValidator::new(&bridge);

        let descs = validator.trajectory_to_descriptions(&[0, 1, 2]);
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0], Some("start".to_string()));
        assert_eq!(descs[1], Some("middle".to_string()));
        assert_eq!(descs[2], Some("end".to_string()));
    }

    #[test]
    fn test_bridge_error_display() {
        let err = BridgeError::NoMatchingNode("test_state".to_string());
        assert!(format!("{err}").contains("test_state"));

        let err = BridgeError::NoMatchingState("node_42".to_string());
        assert!(format!("{err}").contains("node_42"));

        let err = BridgeError::RoundTripFailed {
            original_node: "A".to_string(),
            recovered_node: "B".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("A") && msg.contains("B"));
    }

    /// Bridge with intentional mismatch for error testing.
    struct BrokenBridge;

    impl ProcedureGraph for BrokenBridge {
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

    impl ProcedureGameState for BrokenBridge {
        type State = TestState;
        type StateDescriptor = String;

        fn node_to_state(&self, node: Self::NodeId) -> Option<Self::State> {
            match node {
                0 => Some(TestState {
                    phase: "start".into(),
                }),
                1 => Some(TestState {
                    phase: "end".into(),
                }),
                _ => None,
            }
        }

        fn state_to_node(&self, state: &Self::State) -> Option<Self::NodeId> {
            // Intentional mismatch: "end" maps to node 0 instead of 1
            match state.phase.as_str() {
                "start" => Some(0),
                "end" => Some(0), // BUG: should be 1
                _ => None,
            }
        }

        fn describe_state(&self, state: &Self::State) -> Self::StateDescriptor {
            state.phase.clone()
        }
    }

    #[test]
    fn test_broken_bridge_roundtrip_fails() {
        let bridge = BrokenBridge;
        let result = bridge.validate_terminal_roundtrip(1);
        assert!(result.is_err(), "Broken round-trip should fail");

        match result {
            Err(BridgeError::RoundTripFailed {
                original_node,
                recovered_node,
            }) => {
                assert_eq!(original_node, "1");
                assert_eq!(recovered_node, "0");
            }
            other => panic!("Expected RoundTripFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_broken_bridge_validator_terminals_fails() {
        let bridge = BrokenBridge;
        let validator = TrajectoryValidator::new(&bridge);
        let result = validator.validate_all_terminals();
        assert!(result.is_err());
    }
}
