//! Core types for Subterranean Procedure Compilation (Plan 110).
//!
//! Declarative representation of procedural workflows as directed graphs.
//! Paper: arXiv:2605.22502 — "Compiling Workflows into Weights"

use std::fmt;

// ── ProcedureNode ──────────────────────────────────────────────

/// A node in a procedure graph representing a discrete game state or phase.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcedureNode {
    pub id: u32,
    pub label: String,
    pub is_terminal: bool,
    pub is_start: bool,
}

impl ProcedureNode {
    /// Create a start node.
    pub fn start(id: u32, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            is_terminal: false,
            is_start: true,
        }
    }

    /// Create a terminal node.
    pub fn terminal(id: u32, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            is_terminal: true,
            is_start: false,
        }
    }

    /// Create an intermediate (non-start, non-terminal) node.
    pub fn intermediate(id: u32, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            is_terminal: false,
            is_start: false,
        }
    }
}

impl fmt::Display for ProcedureNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.is_start, self.is_terminal) {
            (true, false) => write!(f, "[START] {} (id={})", self.label, self.id),
            (false, true) => write!(f, "[TERM] {} (id={})", self.label, self.id),
            _ => write!(f, "{} (id={})", self.label, self.id),
        }
    }
}

// ── ProcedureEdge ──────────────────────────────────────────────

/// A directed edge in a procedure graph with an optional transition condition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcedureEdge {
    pub from: u32,
    pub to: u32,
    pub condition: Option<String>,
}

impl ProcedureEdge {
    /// Unconditional edge between two nodes.
    pub fn new(from: u32, to: u32) -> Self {
        Self {
            from,
            to,
            condition: None,
        }
    }

    /// Conditional edge that fires only when the condition is met.
    pub fn conditional(from: u32, to: u32, condition: impl Into<String>) -> Self {
        Self {
            from,
            to,
            condition: Some(condition.into()),
        }
    }
}

impl fmt::Display for ProcedureEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.condition {
            Some(cond) => write!(f, "{} -> {} [{}]", self.from, self.to, cond),
            None => write!(f, "{} -> {}", self.from, self.to),
        }
    }
}

// ── Trajectory ─────────────────────────────────────────────────

/// A path through a procedure graph, recording visited nodes and conditions met.
#[derive(Debug, Clone)]
pub struct Trajectory<NodeId> {
    /// Ordered sequence of node IDs visited along the path.
    pub path: Vec<NodeId>,
    /// Condition (if any) that triggered each transition.
    /// Length is `path.len() - 1` (one per edge traversed).
    pub conditions_met: Vec<Option<String>>,
}

impl<NodeId> Trajectory<NodeId> {
    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self {
            path: Vec::new(),
            conditions_met: Vec::new(),
        }
    }

    /// Create a trajectory from a single starting node.
    pub fn from_start(start: NodeId) -> Self
    where
        NodeId: Copy,
    {
        Self {
            path: vec![start],
            conditions_met: Vec::new(),
        }
    }

    /// Push a node onto the path with an optional condition.
    pub fn push(&mut self, node: NodeId, condition: Option<String>)
    where
        NodeId: Copy,
    {
        self.path.push(node);
        self.conditions_met.push(condition);
    }

    /// Number of steps (edges) in this trajectory.
    pub fn step_count(&self) -> usize {
        match self.path.len() {
            0 => 0,
            n => n - 1,
        }
    }

    /// Whether this trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }

    /// The first node in this trajectory, if any.
    pub fn start(&self) -> Option<&NodeId> {
        self.path.first()
    }

    /// The last node in this trajectory, if any.
    pub fn end(&self) -> Option<&NodeId> {
        self.path.last()
    }
}

impl<NodeId> Default for Trajectory<NodeId> {
    fn default() -> Self {
        Self::new()
    }
}
