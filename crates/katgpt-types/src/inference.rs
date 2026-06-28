//! Inference result + proposer/gate types.

#[cfg(feature = "sr2am_configurator")]
use super::PlanningDecision;

// ---------------------------------------------------------------------------
// InferenceResult
// ---------------------------------------------------------------------------

/// Output of a single inference pass, with reward signal for feedback loop.
///
/// Fields ordered by descending alignment to minimize padding:
/// u64/i64/usize/String (8-byte) → f32 (4-byte) → Option<#[repr(u8)]> (2-byte) → u8/bool (1-byte).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    // --- 8-byte aligned ---
    /// Input prompt hash (for dedup, not stored).
    pub prompt_hash: u64,
    /// Timestamp (Uuid v7 prefix).
    pub timestamp: i64,
    /// Number of nodes explored in DDTree.
    pub tree_budget_used: usize,
    /// Actual planning horizon used this turn (after entropy truncation, Plan 112 T13).
    #[cfg(feature = "sr2am_configurator")]
    pub plan_horizon_used: usize,
    /// Domain that handled this inference.
    pub domain: String,
    /// Generated output text.
    pub output: String,

    // --- 4-byte aligned ---
    /// Best-path reward (max relevance score from WasmPruner).
    pub reward: f32,

    // --- 2-byte aligned (Option<#[repr(u8)] enum>) ---
    /// SR²AM configurator planning decision for this turn (Plan 112).
    #[cfg(feature = "sr2am_configurator")]
    pub planning_decision: Option<PlanningDecision>,

    // --- 1-byte fields (tail-packed) ---
    /// Was this result screened out (reward below threshold)?
    pub screened: bool,
    /// Inference budget level (0=cheap, 1=moderate, 2=expensive).
    pub budget_level: u8,
}

// ---------------------------------------------------------------------------
// Data Gate — Self-Play Stability (Plan 111, Research 075)
// ---------------------------------------------------------------------------

/// Discriminator for different self-play task types.
#[cfg(feature = "data_gate")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskType {
    /// Python code output prediction
    CodeIO,
    /// DSL expression evaluation
    DslExpr,
    /// Game action (Bomber, Go, FFT, Monopoly)
    GameAction,
    /// Open-ended generation
    OpenEnded,
}

/// A task proposed by the self-play proposer, before solver evaluation.
#[cfg(feature = "data_gate")]
#[derive(Debug, Clone)]
pub struct ProposerTask {
    /// Task identifier for diagnostics.
    pub id: usize,
    /// The problem/query text.
    pub query: String,
    /// Optional code or DSL expression to execute.
    pub program: Option<String>,
    /// Optional input for the program.
    pub program_input: Option<String>,
    /// Task type discriminator.
    pub task_type: TaskType,
}

/// Gate admission decision.
///
/// Decides whether a proposer-generated task should enter the training pool
/// BEFORE the solver attempts it.
#[cfg(feature = "data_gate")]
#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Task passes the gate — admitted to training pool.
    Admit,
    /// Task rejected with reason.
    Reject(String),
}

/// Task-level admission gate for self-play training pool.
///
/// Decides whether a proposer-generated task should enter the training pool
/// BEFORE the solver attempts it. This is the binding constraint for self-play
/// stability (Survive or Collapse, Pu et al. 2026).
///
/// Key finding: a strict gate is sufficient for stability under every reward
/// variant; no reward variant is sufficient once the gate is removed.
#[cfg(feature = "data_gate")]
pub trait DataGate {
    /// Admit or reject a proposed task.
    ///
    /// Returns `GateDecision::Admit` if the task passes the gate,
    /// `GateDecision::Reject(reason)` if not.
    fn admit(&self, task: &ProposerTask) -> GateDecision;

    /// Current leak rate ε (fraction of failed tasks admitted).
    /// ε=0 means strict gate (optimal). ε=1 means gate off (collapse).
    fn leak_rate(&self) -> f32;
}
