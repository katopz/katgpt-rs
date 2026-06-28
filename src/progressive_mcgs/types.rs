//! Core types for Progressive MCGS.
//!
//! Paper §3.1–§3.2 formalizes the search as a directed graph
//! `G = (V, E)` with `E = E_T ∪ E_ref`. This module ships the typed
//! handles for nodes, branches, edges, and the three-level reward signal.

/// Maximum number of reference edges per node.
///
/// Per Plan 272 §4 (risks): cap `E_ref` per node at K=3 with LRU eviction
/// to prevent reference-edge explosion. Eviction touches only `E_ref`,
/// never `E_T`.
pub const MAX_REFS_PER_NODE: usize = 3;

/// Default exploration constant `c_0` for UCT (paper §3.2.2).
///
/// Equal to `√2` — use `std::f32::consts::SQRT_2` if you need the exact value.
pub const DEFAULT_C_0: f32 = std::f32::consts::SQRT_2;

/// Default lower bound `c_min` for UCT exploration constant (paper Table 4).
pub const DEFAULT_C_MIN: f32 = 0.5;

/// Smoothing constant `ε` for UCT denominator (paper Eq. 3).
pub const UCT_EPSILON: f32 = 1e-6;

/// Dense node identifier — local to a single [`ProgressiveMcgs`](super::graph::ProgressiveMcgs) graph instance.
///
/// Uses `u32` (not `Uuid`) because:
/// (a) nodes are local to one graph, no cross-graph identity needed;
/// (b) `u32` enables dense `Vec<N>` storage indexed by `NodeId`;
/// (c) 4 bytes vs 16 bytes — better cache locality for hot loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Sentinel for "no node" — used as root's parent.
    pub const NONE: NodeId = NodeId(u32::MAX);

    #[inline]
    #[must_use]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Branch identifier — distinct type from [`NodeId`] to prevent accidental cross-use.
///
/// A "branch" is a personality lineage: nodes sharing a root-to-leaf path in
/// `E_T`. Used for branch-level stagnation detection and per-branch best tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BranchId(pub u32);

impl BranchId {
    /// Sentinel for "no branch" — used for the root before assignment.
    pub const NONE: BranchId = BranchId(u32::MAX);

    #[inline]
    #[must_use]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for BranchId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Edge kind — distinguishes credit-assignment edges from information-flow edges.
///
/// Per paper §3.2.1, `E = E_T ∪ E_ref`. This enum tags edges so that
/// backprop can assert it only walks `E_T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EdgeKind {
    /// Primary edge — parent→child generative. Participates in selection + backprop.
    Primary = 0,
    /// Reference edge — cross-branch / non-adjacent info flow.
    /// Excluded from backprop; read-only at proposal construction.
    Reference = 1,
}

/// Three-level reward signal (paper Eq. 7).
///
/// Clean credit assignment distinguishing failed runs, feasible but
/// non-improving attempts, and actual improvements. Mapped to `f32`
/// for compatibility with vanilla MCTS Q-value math.
///
/// # Note on Non-Stationarity
///
/// `Breakthrough` means "refreshes branch best". The branch best evolves
/// during search, so the same raw outcome can be `Progress` early and
/// `Breakthrough` later (or vice versa). Per Plan 272 §4 risk, callers MUST
/// snapshot `branch_best` *before* updating it when classifying rewards.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(i8)]
pub enum Reward {
    /// Execution failed / no valid metric. Maps to `-1.0`.
    Failure = -1,
    /// Succeeded but did not improve branch best. Maps to `+1.0`.
    Neutral = 0,
    /// Goal Q-value improved but didn't refresh branch best. Maps to `+1.0`.
    Progress = 1,
    /// Succeeded AND refreshed branch best. Maps to `+2.0`.
    /// Rare — triggers reference-edge candidacy in the consumer.
    Breakthrough = 2,
}

impl Reward {
    /// Map reward to `f32` for Q-value math (paper Eqs. 8–9).
    #[inline]
    #[must_use]
    pub const fn as_f32(self) -> f32 {
        match self {
            Self::Failure => -1.0,
            // Paper treats "feasible but non-improving" as +1; we collapse
            // Neutral and Progress into +1.0 for backprop compatibility.
            // The semantic distinction (Neutral vs Progress) is preserved
            // for stagnation detection — see `StagnationGate::observe_expansion`.
            Self::Neutral | Self::Progress => 1.0,
            Self::Breakthrough => 2.0,
        }
    }

    /// Returns `true` iff this reward counts as an "improvement" for
    /// stagnation-counter purposes. `Progress` and `Breakthrough` both qualify;
    /// `Failure` and `Neutral` do not.
    #[inline]
    #[must_use]
    pub const fn is_improvement(self) -> bool {
        matches!(self, Self::Progress | Self::Breakthrough)
    }

    /// Returns `true` iff this reward refreshes the branch best
    /// (only `Breakthrough` qualifies).
    #[inline]
    #[must_use]
    pub const fn is_breakthrough(self) -> bool {
        matches!(self, Self::Breakthrough)
    }
}

/// Top-level configuration for a [`ProgressiveMcgs`](super::graph::ProgressiveMcgs) search instance.
///
/// All hyperparameters are public + override-able; defaults match paper
/// Table 4. Callers in shorter-budget contexts (e.g., 20Hz game ticks)
/// should override `stagnation_branch_threshold` and
/// `stagnation_global_threshold` to tick-count equivalents.
#[derive(Debug, Clone)]
pub struct ProgressiveMcgsConfig {
    /// Maximum number of nodes before eviction kicks in.
    /// Default: 100_000 (paper uses 500 expansions, but our graphs grow wider).
    pub max_nodes: usize,
    /// Maximum reference edges per node (cap on `E_ref` per node, LRU-evicted).
    /// Default: [`MAX_REFS_PER_NODE`] = 3.
    pub max_refs_per_node: usize,
    /// UCT exploration constant at `t_norm = 0`. Default `√2`.
    pub uct_c0: f32,
    /// UCT exploration constant floor at `t_norm ≥ switch_end`. Default `0.5`.
    pub uct_c_min: f32,
    /// Stagnation threshold per branch (non-improving expansions).
    /// Default 3 — paper value, tuned for 500-step budget. Rescale for tick budgets.
    pub stagnation_branch_threshold: u32,
    /// Stagnation threshold globally (steps without global-best refresh).
    /// Default 6 — paper value.
    pub stagnation_global_threshold: u32,
    /// Minimum UCT-selection probability `w_min` (paper Eq. 4, default 0.2).
    pub entropy_w_min: f32,
    /// Normalized progress at which entropy decay begins (paper default 0.5).
    pub entropy_switch_start: f32,
    /// Normalized progress at which entropy decay saturates (paper default 0.7).
    pub entropy_switch_end: f32,
    /// Top-K nodes considered in Elite-Guided exploitation (paper default 3).
    pub elite_topk: usize,
}

impl Default for ProgressiveMcgsConfig {
    #[inline]
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_refs_per_node: MAX_REFS_PER_NODE,
            uct_c0: DEFAULT_C_0,
            uct_c_min: DEFAULT_C_MIN,
            stagnation_branch_threshold: 3,
            stagnation_global_threshold: 6,
            entropy_w_min: 0.2,
            entropy_switch_start: 0.5,
            entropy_switch_end: 0.7,
            elite_topk: 3,
        }
    }
}

impl ProgressiveMcgsConfig {
    /// Validate config values; returns `Err` with a human-readable message if invalid.
    ///
    /// Call this once at construction; cheap relative to search loop.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_refs_per_node == 0 {
            return Err("max_refs_per_node must be ≥ 1");
        }
        if self.entropy_w_min < 0.0 || self.entropy_w_min > 1.0 {
            return Err("entropy_w_min must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&self.entropy_switch_start) {
            return Err("entropy_switch_start must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&self.entropy_switch_end) {
            return Err("entropy_switch_end must be in [0, 1]");
        }
        if self.entropy_switch_end < self.entropy_switch_start {
            return Err("entropy_switch_end must be ≥ entropy_switch_start");
        }
        if self.elite_topk == 0 {
            return Err("elite_topk must be ≥ 1");
        }
        if self.stagnation_branch_threshold == 0 {
            return Err("stagnation_branch_threshold must be ≥ 1");
        }
        if self.stagnation_global_threshold == 0 {
            return Err("stagnation_global_threshold must be ≥ 1");
        }
        Ok(())
    }
}
