//! Shared traits for game AI and speculative decoding.
//!
//! Consolidated from katgpt-rs and riir-engine to eliminate duplication.
//! Both crates depend on `katgpt-core`, so moving traits here requires
//! zero new dependency edges.
//!
//! # Traits
//!
//! - [`ConstraintPruner`] — hard structural validity for DDTree branches
//! - [`ScreeningPruner`] — graded semantic relevance for speculative decoding
//! - [`GameState`] — forward model for what-if game simulation
//! - [`StateHeuristic`] — pluggable evaluation for non-terminal states
//! - [`RolloutPolicy`] — pluggable action selection for MCTS rollouts
//!
//! # Companion Structs
//!
//! - [`NoPruner`] — allows all tokens (baseline)
//! - [`BinaryScreeningPruner`] — adapter: ConstraintPruner → ScreeningPruner
//! - [`NoScreeningPruner`] — returns 1.0 for everything
//! - [`RandomRolloutPolicy`] — uniform random action selection
//! - [`ActionSpaceLog`] — per-tick branching factor metrics

use std::fmt;

use fastrand::Rng;

// ── ConstraintPruner ────────────────────────────────────────────

/// Trait for pruning drafted tokens against deterministic constraints.
///
/// The Deterministic Validator concept: before the target model verifies drafted
/// branches, a rules engine prunes invalid ones. This prevents the DDTree
/// from wasting budget on branches that can never be accepted.
///
/// Without pruner: DDTree explores ALL high-probability tokens.
/// With pruner:    DDTree explores only VALID high-probability tokens.
pub trait ConstraintPruner: Send + Sync {
    /// Check if `token_idx` at the given `depth` is valid, given the
    /// tokens placed at earlier depths in this path.
    ///
    /// `parent_tokens[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    ///
    /// Returns `false` to prune (reject) this branch.
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;

    /// Validate multiple token candidates at the same depth in a single call.
    ///
    /// Writes results into `results`: `results[i] = is_valid(depth, candidates[i], parent_tokens)`.
    /// Implementations can override this to amortize lock acquisition and setup costs
    /// across all candidates (e.g., single mutex lock + fuel reset for WASM).
    ///
    /// Default implementation calls `is_valid` per-item.
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        for i in 0..len {
            results[i] = self.is_valid(depth, candidates[i], parent_tokens);
        }
    }
}

/// No-op pruner: allows all tokens (original DDTree behavior).
pub struct NoPruner;

impl ConstraintPruner for NoPruner {
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        true
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[usize],
        _parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        results[..len].fill(true);
    }
}

// ── ScreeningPruner ─────────────────────────────────────────────

/// Graded relevance pruner replacing binary valid/invalid with continuous score.
///
/// Distilled from "Screening Is Enough" (arXiv:2604.01178).
/// Returns `R ∈ [0.0, 1.0]` which is blended into log-prob space:
/// - `1.0` = perfect match, no penalty (`ln(1.0) = 0.0`)
/// - `0.5` = mediocre match, soft penalty (`ln(0.5) ≈ -0.69`)
/// - `0.0` = hard rejection / trim (`ln(0.0) = -∞`)
///
/// This subsumes [`ConstraintPruner`] as the special case `R ∈ {0.0, 1.0}`.
/// Use [`BinaryScreeningPruner`] adapter to bridge between them.
///
/// # Ownership Boundary with ConstraintPruner (Plan 029, Task 7)
///
/// Single parser ownership: `ConstraintPruner` and `ScreeningPruner` make
/// **independent** decisions and must not compete for the same judgment:
///
/// - **`ConstraintPruner`** = hard structural validity (syntax, brackets, keywords).
///   Returns `bool`. Owns the decision: "is this token *syntactically* legal here?"
///
/// - **`ScreeningPruner`** = graded semantic relevance (domain fit, topic match).
///   Returns `f32` in `[0.0, 1.0]`. Owns the decision: "is this token *semantically*
///   relevant to the current domain?"
///
/// - **[`BinaryScreeningPruner`]** adapter = bridge only, zero additional logic.
///   Converts [`ConstraintPruner::is_valid()`] → `{0.0, 1.0}` relevance.
///
/// Both may prune the same token for different reasons — that's fine.
/// Both must NOT claim ownership of the same decision type — that's a bug.
pub trait ScreeningPruner: Send + Sync {
    /// Returns the absolute relevance of taking this token given the path.
    ///
    /// `parent_tokens[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

/// Adapter: wraps any [`ConstraintPruner`] as a [`ScreeningPruner`] with binary relevance.
/// - `is_valid() == true` → relevance 1.0 (no penalty)
/// - `is_valid() == false` → relevance 0.0 (hard trim)
///
/// Use this to pass a [`ConstraintPruner`] where a [`ScreeningPruner`] is expected.
/// We use an explicit adapter instead of a blanket impl to avoid conflicts
/// with types that implement [`ConstraintPruner`] but need a custom [`ScreeningPruner`].
pub struct BinaryScreeningPruner<P>(pub P);

impl<P: ConstraintPruner + Send + Sync> ScreeningPruner for BinaryScreeningPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        match self.0.is_valid(depth, token_idx, parent_tokens) {
            true => 1.0,
            false => 0.0,
        }
    }
}

/// No-op screener: returns 1.0 for everything (no penalty, no trimming).
pub struct NoScreeningPruner;

impl ScreeningPruner for NoScreeningPruner {
    #[inline]
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        1.0
    }
}

// ── GameState ───────────────────────────────────────────────────

/// Forward model trait — any game state that supports what-if simulation.
///
/// Implementors must be cheaply cloneable snapshots (~KB, not MB).
/// The arena converts its internal state → snapshot once per tick,
/// then search algorithms work entirely on snapshots.
///
/// # Type Parameters
/// - `Action`: the move type for this game domain
///
/// # Required Methods
/// - `available_actions`: legal moves for a player
/// - `advance`: pure successor state (no mutation)
/// - `is_terminal`: game-over check
/// - `reward`: terminal value for a player
/// - `tick`: current turn number
pub trait GameState: Clone {
    /// Move type for this game domain (e.g., `BomberAction`, `fft::Action`).
    type Action: Clone;

    /// Legal actions for `player_id` in current state.
    fn available_actions(&self, player_id: u8) -> Vec<Self::Action>;

    /// Fill `buf` with legal actions for `player_id`, clearing it first.
    ///
    /// Default implementation calls [`available_actions()`](Self::available_actions)
    /// and moves items into `buf`. Override to avoid intermediate allocation.
    fn available_actions_into(&self, player_id: u8, buf: &mut Vec<Self::Action>) {
        buf.clear();
        buf.extend(self.available_actions(player_id));
    }

    /// Apply action, return successor state. Does NOT mutate `self`.
    fn advance(&self, action: &Self::Action, player_id: u8) -> Self;

    /// Is the game over?
    fn is_terminal(&self) -> bool;

    /// Terminal reward for `player_id` (higher = better, typically 0..1).
    fn reward(&self, player_id: u8) -> f32;

    /// Current tick/turn number.
    fn tick(&self) -> u32;

    /// Number of legal actions for `player_id`.
    ///
    /// Default implementation calls [`available_actions().len()`](Self::available_actions).
    /// Override if you can compute this cheaper than building the full vec.
    fn action_space_size(&self, player_id: u8) -> usize {
        self.available_actions(player_id).len()
    }
}

// ── StateHeuristic ──────────────────────────────────────────────

/// Pluggable heuristic for evaluating non-terminal states.
///
/// Used by search algorithms (MCTS rollouts, RHEA fitness) when
/// [`GameState::is_terminal()`] is false but we need a numeric evaluation.
///
/// Domain-specific heuristics beat generic search (STRATEGA finding),
/// so each game provides its own implementation.
pub trait StateHeuristic<S: GameState> {
    /// Evaluate state for `player_id`. Higher = better.
    fn evaluate(&self, state: &S, player_id: u8) -> f32;
}

// ── RolloutPolicy ───────────────────────────────────────────────

/// Pluggable rollout policy for MCTS.
///
/// Replaces hardcoded random selection with informed action choice.
/// The default [`RandomRolloutPolicy`] preserves existing behavior.
///
/// # Implementors
/// - [`RandomRolloutPolicy`]: uniform random (baseline)
/// - `BanditRolloutPolicy<S>`: ε-greedy guided by bandit Q-values (riir-engine)
pub trait RolloutPolicy<S: GameState> {
    /// Select an action index from `actions` during MCTS rollout.
    ///
    /// # Arguments
    /// * `state` — current rollout state
    /// * `actions` — available actions for `player_id`
    /// * `player_id` — which player is acting
    /// * `rng` — RNG for stochastic policies
    ///
    /// # Returns
    /// Index into `actions` (0..actions.len()).
    fn select(&mut self, state: &S, actions: &[S::Action], player_id: u8, rng: &mut Rng) -> usize;
}

/// Uniform random rollout policy — baseline, identical to original MCTS behavior.
///
/// Every action has equal probability. Use this as a control group when
/// comparing against informed rollout policies.
pub struct RandomRolloutPolicy;

impl<S: GameState> RolloutPolicy<S> for RandomRolloutPolicy {
    #[inline]
    fn select(
        &mut self,
        _state: &S,
        actions: &[S::Action],
        _player_id: u8,
        rng: &mut Rng,
    ) -> usize {
        rng.usize(0..actions.len())
    }
}

// ── ActionSpaceLog ──────────────────────────────────────────────

/// Per-tick action space metrics for branching factor analysis.
///
/// Tracks how the action space evolves across ticks — useful for:
/// - Validating search budget vs branching factor
/// - Detecting game phases (opening → midgame → endgame)
/// - Comparing action space across game domains
#[derive(Clone, Debug, Default)]
pub struct ActionSpaceLog {
    /// (tick, player_id, action_count) entries.
    entries: Vec<(u32, u8, usize)>,
}

impl ActionSpaceLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record action space size for a player at the current tick.
    pub fn record<S: GameState>(&mut self, state: &S, player_id: u8) {
        self.entries
            .push((state.tick(), player_id, state.action_space_size(player_id)));
    }

    /// Total number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the log empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Average action space size across all entries.
    pub fn avg_action_space(&self) -> f32 {
        match self.entries.is_empty() {
            true => 0.0,
            false => {
                self.entries.iter().map(|&(_, _, n)| n as f32).sum::<f32>()
                    / self.entries.len() as f32
            }
        }
    }

    /// Average action space size for a specific player.
    pub fn avg_action_space_for(&self, player_id: u8) -> f32 {
        let filtered: Vec<_> = self
            .entries
            .iter()
            .filter(|&&(_, pid, _)| pid == player_id)
            .collect();
        match filtered.is_empty() {
            true => 0.0,
            false => {
                filtered.iter().map(|&&(_, _, n)| n as f32).sum::<f32>() / filtered.len() as f32
            }
        }
    }

    /// Peak (maximum) action space size recorded.
    pub fn peak_action_space(&self) -> usize {
        self.entries.iter().map(|&(_, _, n)| n).max().unwrap_or(0)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl fmt::Display for ActionSpaceLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.entries.is_empty() {
            true => write!(f, "ActionSpaceLog(empty)"),
            false => write!(
                f,
                "ActionSpaceLog(entries={}, avg={:.1}, peak={})",
                self.entries.len(),
                self.avg_action_space(),
                self.peak_action_space()
            ),
        }
    }
}
