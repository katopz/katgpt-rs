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

    /// Propagate semantic state with a new token. Returns true if all predicates hold.
    /// Default: no-op (delegates to is_valid for backward compat).
    fn propagate(&mut self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> bool {
        true // no-op by default
    }

    /// Soft validity score: how close is this token to the constraint boundary?
    /// Returns 1.0 for valid, 0.0 for invalid by default.
    /// Override for soft scoring (ManifoldE point-to-manifold, Plan 234).
    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        match self.is_valid(depth, token_idx, parent_tokens) {
            true => 1.0,
            false => 0.0,
        }
    }

    /// Returns the constraint as a half-space (normal vector, threshold) if available.
    /// Default: None (fall back to is_valid/manifold_score).
    fn constraint_vector(&self, _depth: usize, _parent_tokens: &[usize]) -> Option<(&[f32], f32)> {
        None
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

    fn propagate(&mut self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> bool {
        true
    }
}

impl DominoPruner for NoPruner {}

// ── DominoPruner (Plan 197, Research 177) ──────────────────────

/// Prefix-conditioned constraint correction for Domino causal correction.
///
/// Extends [`ConstraintPruner`] with a secondary correction pass that uses
/// the *specific prefix path* to refine validity decisions. This enables:
/// - **False positive elimination**: base says valid but prefix context says invalid
/// - **False negative recovery**: base says invalid but prefix context says valid (rare)
///
/// Default impl returns `base_valid` (no-op) — opt-in correction only.
///
/// # Implementors
///
/// - `SudokuPruner`: checks row/col/box constraints given the specific prefix path
/// - `SynPruner`: checks Rust syntax validity given preceding tokens
///   (e.g., `if` must be followed by `{` or condition)
pub trait DominoPruner: ConstraintPruner {
    /// Refine the base validity decision using prefix context.
    ///
    /// # Arguments
    /// * `depth` — current tree depth (position in the sequence)
    /// * `token` — candidate token index
    /// * `prefix` — tokens placed at earlier depths in this path
    /// * `base_valid` — result from the base [`ConstraintPruner::is_valid()`] check
    ///
    /// # Returns
    ///
    /// Corrected validity: `true` = keep, `false` = prune.
    #[inline]
    fn causal_correction(
        &self,
        _depth: usize,
        _token: usize,
        _prefix: &[usize],
        base_valid: bool,
    ) -> bool {
        base_valid
    }
}

// ── CompletionHorizon (Plan 207, Research 183) ──────────────────

/// Admissible "distance to a complete, valid output" for budget-aware pruning.
///
/// Extends [`ConstraintPruner`] with the **shortest-accepting-distance** `d(s)`:
/// a lower bound on how many *additional* tokens are needed, from the automaton
/// state reached after placing `token_idx` at `depth` (following `parent_tokens`),
/// to reach a complete & valid output. One precomputed integer per state powers
/// three things at once (Research 183):
///
/// - **(A) budget-aware masking** — prune any token whose successor cannot still
///   complete within the remaining token budget (TRUNCPROOF guarantee). This is
///   the property `build_dd_tree_lodestar` relies on.
/// - **(B) jump-ahead** — deterministic singular spans can be emitted in one step.
/// - **(C) A\* ordering + termination** — `d` is a monotone, admissible heuristic.
///
/// The novel equivalence (no single source states it): *min-completion-length ≡
/// remaining lattice height ≡ an admissible A\* heuristic.*
///
/// Default impls return `0` (no horizon info) so every existing
/// [`ConstraintPruner`] keeps working unchanged and pays **zero** overhead —
/// Lodestar is pure opt-in.
///
/// # Contract
///
/// `min_completion_distance` MUST be *admissible* (never overestimate the true
/// remaining length) for the budget guarantee to hold. Return [`u32::MAX`] to
/// signal "no valid completion reachable" — callers treat it as a hard prune.
pub trait CompletionHorizon: ConstraintPruner {
    /// Admissible lower bound on additional tokens needed to reach a complete,
    /// valid output *after* placing `token_idx` at `depth` following
    /// `parent_tokens` (the tokens at depths `0..depth`).
    ///
    /// Returns `0` by default (no horizon ⇒ budget masking is a no-op).
    #[inline]
    fn min_completion_distance(
        &self,
        _depth: usize,
        _token_idx: usize,
        _parent_tokens: &[usize],
    ) -> u32 {
        0
    }

    /// Length of the deterministic singular-path span starting at the state
    /// reached after `parent_tokens` — `0` means the next step is a real branch.
    /// Enables jump-ahead. Default `0`.
    #[inline]
    fn singular_span_len(&self, _depth: usize, _parent_tokens: &[usize]) -> u32 {
        0
    }
}

/// `NoPruner` has no horizon — budget masking is a no-op (returns 0).
impl CompletionHorizon for NoPruner {}

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

// ── Collapse-Aware Adaptive Thinking (Plan 212) ────────────────────

/// Monitors the token stream during reasoning and triggers early exit when
/// reasoning collapse is detected (hesitation patterns, repetitive tokens).
///
/// Modelless inference-time feature behind `collapse_aware_thinking` feature gate.
/// Integrates with `ThinkingController` to cut wasted budget on degenerate traces.
#[cfg(feature = "collapse_aware_thinking")]
pub trait CollapseDetector: Send + Sync {
    /// Returns `true` if the current trace exhibits collapse symptoms.
    fn check_collapse(&mut self, token_id: u32, position: usize) -> bool;

    /// Reset internal state between traces. May update self-tuning parameters.
    fn reset(&mut self);

    /// Number of hesitation tokens observed in the current trace.
    fn hesitation_count(&self) -> u32;

    /// Current collapse threshold τ.
    fn threshold(&self) -> u32;
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
    /// **Override this** if you can compute the count without building the full action Vec.
    /// The default implementation calls [`available_actions()`](Self::available_actions)
    /// which allocates — expensive in tight MCTS loops.
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
///
/// Per-player aggregate tracked during insert for O(1) reads.
#[derive(Clone, Debug, Default)]
struct PlayerAgg {
    // OPT: usize first avoids 4 bytes of padding between f32 and usize
    count: usize,
    sum: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ActionSpaceLog {
    /// (tick, player_id, action_count) entries.
    /// Field order: usize (8B) → u32 (4B) → u8 (1B) = 16B vs 24B with (u32, u8, usize).
    entries: Vec<(usize, u32, u8)>,
    /// Per-player running aggregates for O(1) avg_action_space_for().
    /// Indexed by player_id (u8, so max 256 entries). Lazy-initialized on first record.
    player_aggs: Vec<PlayerAgg>,
    /// Running peak across all entries, tracked during record() for O(1) peak_action_space().
    peak: usize,
}

impl ActionSpaceLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty log with pre-allocated capacity.
    /// Use this in hot paths (e.g., game rollouts) where the approximate
    /// number of entries is known upfront.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            player_aggs: Vec::new(),
            peak: 0,
        }
    }

    /// Record action space size for a player at the current tick.
    /// Maintains per-player aggregates for O(1) avg_action_space_for().
    pub fn record<S: GameState>(&mut self, state: &S, player_id: u8) {
        let n = state.action_space_size(player_id);
        let pid = player_id as usize;
        // Extend player_aggs if needed (at most 256 entries)
        if self.player_aggs.len() <= pid {
            self.player_aggs
                .resize(pid + 1, PlayerAgg { sum: 0.0, count: 0 });
        }
        self.player_aggs[pid].sum += n as f32;
        self.player_aggs[pid].count += 1;
        if n > self.peak {
            self.peak = n;
        }
        self.entries.push((n, state.tick(), player_id));
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
                self.entries.iter().map(|&(n, _, _)| n as f32).sum::<f32>()
                    / self.entries.len() as f32
            }
        }
    }

    /// Average action space size for a specific player.
    /// O(1) via pre-tracked per-player aggregates — no linear scan.
    pub fn avg_action_space_for(&self, player_id: u8) -> f32 {
        let pid = player_id as usize;
        let agg = match self.player_aggs.get(pid) {
            Some(a) if a.count > 0 => a,
            _ => return 0.0,
        };
        agg.sum / agg.count as f32
    }

    /// Peak (maximum) action space size recorded.
    /// O(1) via running peak tracked during record().
    pub fn peak_action_space(&self) -> usize {
        self.peak
    }

    /// Clear all entries and reset per-player aggregates.
    pub fn clear(&mut self) {
        self.entries.clear();
        for agg in &mut self.player_aggs {
            agg.sum = 0.0;
            agg.count = 0;
        }
        self.peak = 0;
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

// ── LEO All-Goals Traits (Plan 155) ──────────────────────────────
//
// LEO (Learn Everything All at Once) outputs Q-values for ALL goals
// simultaneously instead of conditioning on a single goal (UVFA-style).
// Ref: Matthews et al. (2026) "Goal-Conditioned Agents that Learn Everything
//   All at Once", ICML 2026. arXiv:2605.23551
//   https://github.com/MichaelTMatthews/purejaxgcrl
//
// Feature gates:
//   leo_all_goals — LeoHead + AllGoalsUpdate + sigmoid_bounded_q
//   dual_leo      — + DualLeoMixer + AutocurriculumSampler

// ── LEO All-Goals Trait Framework (Plan 155) ────────────────────
//
// Architecture note — Batch Renormalization (BatchRenorm):
//
// The reference JAX implementation uses BatchRenorm (not standard BatchNorm
// or LayerNorm) for network stability with highly off-policy data. Key params:
//   - r_max = 3 (maximum correction ratio)
//   - d_max = 5 (maximum correction difference)
//   - warmup = 1000 steps (gradual correction ramp-up)
//
// BatchRenorm constrains the running-statistics correction to prevent
// divergence when training on highly off-policy replay data — critical
// for LEO's all-goals Q-learning where the same network processes
// experiences from many different goal-conditioned policies.
//
// Implementors SHOULD use BatchRenorm (or an equivalent constrained
// normalization) in their LeoHead network architectures. Standard
// BatchNorm is insufficient; LayerNorm is acceptable but may underperform.
//
// Ref: Ioffe (2017) "Batch Renormalization" — arXiv:1702.03275
// Ref: Matthews et al. (2026) "Goal-Conditioned Agents that Learn Everything
//   All at Once", ICML 2026. arXiv:2605.23551 — Section 5.1

/// Bound Q-value estimates with sigmoid to prevent divergence.
///
/// CRITICAL: Without this, LEO's Q-values frequently diverge due to
/// highly off-policy updates (paper Section 5.1).
///
/// Maps raw Q ∈ (-∞, +∞) → bounded Q ∈ (0, 1).
#[cfg(feature = "leo_all_goals")]
#[inline]
pub fn sigmoid_bounded_q(raw_q: f32) -> f32 {
    1.0 / (1.0 + (-raw_q).exp())
}

/// All-goals Q-value output head (LEO architecture).
///
/// Instead of conditioning on a goal (UVFA-style), this outputs Q-values
/// for ALL goals simultaneously: Q(s) → R^{G×A}.
///
/// Ref: Matthews et al. (2026) "Goal-Conditioned Agents that Learn Everything
///   All at Once", ICML 2026. arXiv:2605.23551
#[cfg(feature = "leo_all_goals")]
pub trait LeoHead {
    /// Compute Q-values for all goals × all actions from state.
    /// Returns `[goals * actions]` flattened (row-major: goal-major).
    fn all_goals_q(&self, state: &[f32]) -> Vec<f32>;

    /// Number of goals in the output head.
    fn goal_count(&self) -> usize;

    /// Number of discrete actions per goal.
    fn action_count(&self) -> usize;

    /// Extract Q-values for a specific goal by indexing into the flat output.
    fn q_for_goal<'a>(&self, all_q: &'a [f32], goal: usize) -> &'a [f32] {
        let start = goal * self.action_count();
        &all_q[start..start + self.action_count()]
    }
}

/// Vectorized all-goals Bellman update.
///
/// L = (R(s') + γ · max_a' Q(a'|s') - Q(a|s))²
///
/// Where R(s') ∈ R^G is the reward vector across ALL goals.
/// Single forward pass updates all |G| Q-value heads simultaneously.
#[cfg(feature = "leo_all_goals")]
pub trait AllGoalsUpdate {
    /// Compute all-goals TD target.
    ///
    /// - `rewards`: `[goals]` — R(s', g) for all g
    /// - `next_q`: `[goals][actions]` — Q(s', a', g) for all g, a
    /// - Returns: `[goals]` — TD target per goal
    fn td_target(&self, rewards: &[f32], next_q: &[Vec<f32>], gamma: f32) -> Vec<f32> {
        rewards
            .iter()
            .zip(next_q.iter())
            .map(|(&r, q_next)| {
                let max_q = q_next.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                r + gamma * max_q
            })
            .collect()
    }

    /// Compute all-goals TD loss (MSE) averaged across goals.
    ///
    /// - `predicted`: `[goals]` each containing chosen-action Q-values
    /// - `target`: `[goals]` TD targets
    fn loss(predicted: &[Vec<f32>], target: &[f32]) -> f32 {
        predicted
            .iter()
            .zip(target.iter())
            .map(|(q_pred, &q_tgt)| {
                let chosen = q_pred[0]; // first action as chosen (caller should index correctly)
                0.5 * (chosen - q_tgt).powi(2)
            })
            .sum::<f32>()
            / predicted.len().max(1) as f32
    }

    /// Compute all-goals Q(λ) TD target with eligibility traces.
    ///
    /// Multi-step TD with trace decay parameter λ:
    ///   G(g) = R(g) + γ · [λ · G_next(g) + (1-λ) · max_a' Q(s',a',g)] · (1-done(g))
    ///
    /// - `rewards`: `[goals]` — R(s', g) for all g
    /// - `next_q_max`: `[goals]` — max_a' Q(s', a', g) for all g (pre-computed)
    /// - `next_lambda_return`: `[goals]` — G_next(g) λ-return from next timestep (0 for last step)
    /// - `done`: `[goals]` — whether goal g is terminal
    /// - `gamma`: discount factor
    /// - `lambda`: trace decay (0 = one-step TD, 1 = Monte Carlo)
    /// - Returns: `[goals]` — Q(λ) target per goal
    fn td_target_lambda(
        &self,
        rewards: &[f32],
        next_q_max: &[f32],
        next_lambda_return: &[f32],
        done: &[bool],
        gamma: f32,
        lambda: f32,
    ) -> Vec<f32> {
        rewards
            .iter()
            .zip(next_q_max.iter())
            .zip(next_lambda_return.iter())
            .zip(done.iter())
            .map(|(((&r, &q_max), &g_next), &d)| {
                if d {
                    r
                } else {
                    r + gamma * (lambda * g_next + (1.0 - lambda) * q_max)
                }
            })
            .collect()
    }
}

/// Acting mode for dual LEO mixing.
///
/// Controls how LEO (teacher) and UVFA (student) Q-values are combined
/// for action selection. From JAX `DUAL_LEO_ACTING_MODE` config.
#[cfg(feature = "dual_leo")]
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActingMode {
    /// Linear combination: Q = (1-α)·Q_UVFA + α·Q_LEO.
    /// Default mode, sweep winner on Craftax.
    #[default]
    Lc,
    /// LEO-only ablation: Q = Q_LEO[:,:,g].
    LeoOnly,
    /// UVFA-only ablation: Q = Q_UVFA[:,g].
    UvfaOnly,
    /// Optimistic combining: Q = max(Q_LEO, Q_UVFA).
    Max,
    /// Pessimistic combining: Q = min(Q_LEO, Q_UVFA).
    Min,
}

/// Schedule for the mixing coefficient α over training.
#[cfg(feature = "dual_leo")]
#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum AlphaSchedule {
    /// Constant α throughout training (default).
    /// Sweep uses 0.3 with `anneal_lc_leo=false`.
    Fixed(f32),
    /// Linearly anneal α from `start` to `end` over training.
    /// From JAX: coef = p * end + (1-p) * start, where p = step/total_steps.
    LinearAnneal { start: f32, end: f32 },
}

#[cfg(feature = "dual_leo")]
impl Default for AlphaSchedule {
    fn default() -> Self {
        AlphaSchedule::Fixed(0.3)
    }
}

/// Behavioral cloning target source for BC regularization (PPO variant).
#[cfg(feature = "dual_leo")]
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BcTarget {
    /// Follow LEO's greedy (argmax) action.
    #[default]
    Argmax,
}

/// Behavioral cloning regularization config (Dual LEO PPO variant).
///
/// The UVFA student's policy is regularized toward LEO's argmax action
/// early in training, then the BC coefficient decays to 0.
///
/// From `dual_leo_ppo.py`: bc_coef_policy=0.1, bc_coef_value=0.0,
/// bc_policy_target="argmax", anneal_bc=true.
#[cfg(feature = "dual_leo")]
#[derive(Clone, Copy, Debug)]
pub struct BcConfig {
    /// PPO policy regularization coefficient toward LEO's action.
    /// Default: 0.1
    pub policy_coef: f32,
    /// Value function BC coefficient. Default: 0.0 (disabled in sweep).
    pub value_coef: f32,
    /// Which action to use as BC target.
    pub target: BcTarget,
    /// Whether to anneal BC coefficient to 0 over training.
    pub anneal: bool,
}

#[cfg(feature = "dual_leo")]
impl Default for BcConfig {
    fn default() -> Self {
        Self {
            policy_coef: 0.1,
            value_coef: 0.0,
            target: BcTarget::Argmax,
            anneal: true,
        }
    }
}

/// Dual LEO mixing between teacher (LEO) and student (UVFA).
///
/// Q_combined(g) = α·Q_LEO(s,a,g) + (1-α)·Q_UVFA(s,a,g)
///
/// α controls modelless→model trust transfer:
/// - High α: trust LEO teacher (modelless, broad)
/// - Low α: trust UVFA student (model-based, precise)
#[cfg(feature = "dual_leo")]
pub trait DualLeoMixer {
    /// Mix LEO and UVFA Q-values for acting on a specific goal.
    fn mix(&self, q_leo: &[f32], q_uvfa: &[f32], alpha: f32) -> Vec<f32> {
        q_leo
            .iter()
            .zip(q_uvfa.iter())
            .map(|(&ql, &qu)| alpha * ql + (1.0 - alpha) * qu)
            .collect()
    }

    /// Mix LEO and UVFA Q-values into a pre-allocated buffer, avoiding allocation.
    fn mix_into(&self, out: &mut [f32], q_leo: &[f32], q_uvfa: &[f32], alpha: f32) {
        for (o, (&ql, &qu)) in out.iter_mut().zip(q_leo.iter().zip(q_uvfa.iter())) {
            *o = alpha * ql + (1.0 - alpha) * qu;
        }
    }

    /// Default α = 0.3 (from paper sweep on Craftax).
    fn default_alpha(&self) -> f32 {
        0.3
    }

    /// Which acting mode to use. Default: Lc (sweep winner).
    fn acting_mode(&self) -> ActingMode {
        ActingMode::Lc
    }

    /// Alpha schedule over training. Default: Fixed(0.3).
    fn alpha_schedule(&self) -> AlphaSchedule {
        AlphaSchedule::Fixed(self.default_alpha())
    }

    /// Resolve alpha for the current training progress (0.0..=1.0).
    fn alpha_at_progress(&self, progress: f32) -> f32 {
        match self.alpha_schedule() {
            AlphaSchedule::Fixed(a) => a,
            AlphaSchedule::LinearAnneal { start, end } => {
                progress.clamp(0.0, 1.0) * end + (1.0 - progress.clamp(0.0, 1.0)) * start
            }
        }
    }

    /// Combine Q-values using the configured acting mode.
    fn combine(&self, q_leo: &[f32], q_uvfa: &[f32], alpha: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; q_leo.len()];
        self.combine_into(&mut out, q_leo, q_uvfa, alpha);
        out
    }

    /// Zero-alloc variant of [`combine`]: writes into a pre-allocated buffer.
    fn combine_into(&self, out: &mut [f32], q_leo: &[f32], q_uvfa: &[f32], alpha: f32) {
        match self.acting_mode() {
            ActingMode::Lc => self.mix_into(out, q_leo, q_uvfa, alpha),
            ActingMode::LeoOnly => out.copy_from_slice(q_leo),
            ActingMode::UvfaOnly => out.copy_from_slice(q_uvfa),
            ActingMode::Max => {
                for (o, (&ql, &qu)) in out.iter_mut().zip(q_leo.iter().zip(q_uvfa.iter())) {
                    *o = ql.max(qu);
                }
            }
            ActingMode::Min => {
                for (o, (&ql, &qu)) in out.iter_mut().zip(q_leo.iter().zip(q_uvfa.iter())) {
                    *o = ql.min(qu);
                }
            }
        }
    }

    /// BC regularization config (Dual LEO PPO variant). Default: no BC.
    fn bc_config(&self) -> Option<BcConfig> {
        None
    }
}

/// Goal sampling from previously observed goals only.
///
/// "We sample goals only from goals observed at least once in the past,
/// to prevent completely out-of-reach goals being sampled."
/// — Matthews et al. (2026), ICML 2026, arXiv:2605.23551
#[cfg(feature = "dual_leo")]
pub trait AutocurriculumSampler {
    /// Sample a goal uniformly from previously observed goals.
    fn sample_goal(&self, rng: &mut Rng) -> usize;

    /// Mark a goal as observed (first time seen in any trajectory).
    fn observe_goal(&mut self, goal: usize);

    /// Number of unique goals observed so far.
    fn observed_count(&self) -> usize;

    /// Total goals in the goal set.
    fn total_goal_count(&self) -> usize;

    /// Update observed goals from a batch of observations.
    ///
    /// Checks which goals match any observation in the batch (union matching).
    /// Returns the updated boolean mask over all goals.
    ///
    /// From JAX `get_goals_seen()`: a goal is "seen" if any obs in the batch
    /// matches the goal's observation pattern (match_sum > 0).
    ///
    /// - `obs_batch`: batch of observation vectors
    /// - `all_goals`: all goal observation patterns `[goals][features]`
    /// - `current_mask`: current `[goals]` boolean mask (true = seen)
    /// - Returns: updated `[goals]` boolean mask
    fn update_goals_seen(
        &self,
        obs_batch: &[Vec<f32>],
        all_goals: &[Vec<f32>],
        current_mask: &[bool],
    ) -> Vec<bool> {
        let mut mask = current_mask.to_vec();
        // Pre-compute goal norms once (avoid redundant recomputation per obs).
        let norm_goals: Vec<f32> = all_goals
            .iter()
            .map(|goal_obs| goal_obs.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();
        for obs in obs_batch {
            let norm_obs: f32 = obs.iter().map(|x| x * x).sum::<f32>().sqrt();
            for (g, goal_obs) in all_goals.iter().enumerate() {
                if g < mask.len() && !mask[g] {
                    // Union matching: normalized cosine-like similarity.
                    // Threshold > 0.9 ensures only near-exact matches count.
                    // JAX uses binary match_sum > 0 on discretized observations.
                    let dot: f32 = obs.iter().zip(goal_obs.iter()).map(|(o, gi)| o * gi).sum();
                    let norm_goal = norm_goals[g];
                    let denom = norm_obs * norm_goal;
                    if denom > 0.0 && dot / denom > 0.9 {
                        mask[g] = true;
                    }
                }
            }
        }
        mask
    }

    /// Number of goals completed in the current episode.
    /// Enables "first return then explore" — after achieving one goal,
    /// immediately sample another. Resets to 0 on episode end.
    fn goals_completed_this_episode(&self) -> usize {
        0
    }

    /// Whether to only sample from previously seen goals.
    /// From JAX `ONLY_SAMPLE_FROM_SEEN_GOALS` config flag.
    fn only_sample_from_seen(&self) -> bool {
        true
    }
}

// ── SpeculativeGenerator (Plan 193 Phase 1) ──────────────────

/// Generic trait for speculative generation of candidates.
///
/// Decouples the output format (tokens, actions, etc.) from the generation
/// mechanism. Any domain implementing `generate() + validate()` can reuse
/// the same DDTree + pruning + routing infrastructure.
///
/// This is the Generator Contract pattern: open for extension (new domains),
/// closed for modification.
pub trait SpeculativeGenerator {
    /// Input condition type (e.g., token context, game state).
    type Condition;
    /// Output type (e.g., logit vector, game action).
    type Output;
    /// Error type.
    type Error;

    /// Generate candidate outputs given a condition.
    fn generate(
        &mut self,
        condition: &Self::Condition,
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error>;

    /// Batch variant for GPU amortization.
    #[inline]
    fn generate_batch(
        &mut self,
        conditions: &[Self::Condition],
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Vec<Self::Output>>, Self::Error> {
        conditions.iter().map(|c| self.generate(c, rng)).collect()
    }
}

/// Typed constraint pruner for generic outputs.
///
/// Extends ConstraintPruner concept to typed output validation.
pub trait GenerativeConstraintPruner<Output>: Send + Sync {
    /// Returns true if the output passes all constraints.
    fn is_valid(&self, output: &Output) -> bool;

    /// Batch variant for amortization.
    #[inline]
    fn batch_is_valid(&self, outputs: &[Output]) -> Vec<bool> {
        outputs.iter().map(|o| self.is_valid(o)).collect()
    }
}

// ── Game Trace + Partial Scoring (Plan 191) ──────────────────────

/// Minimal game trace for partial scoring.
///
/// Captures episode statistics needed for graduated reward computation.
/// Lightweight: stack-friendly, no heap allocations in the struct itself.
#[cfg(feature = "partial_scoring")]
#[derive(Clone, Debug, Default)]
pub struct GameTrace {
    /// Ticks survived before termination.
    pub survival_ticks: u32,
    /// Opponents eliminated.
    pub kills: u32,
    /// Total actions taken during the episode.
    pub actions_taken: u32,
    /// Maximum possible ticks (episode budget).
    pub max_ticks: u32,
    /// Final reward from the game engine (binary: win=1.0, loss=0.0).
    pub final_reward: f64,
}

/// Graduated reward scorer for game episodes.
///
/// Replaces binary win/loss with continuous [0.0, 1.0] reward,
/// giving bandit algorithms richer signal for faster convergence.
///
/// Ref: FrontierCS (human 95 vs best model 29 on open-ended problems).
#[cfg(feature = "partial_scoring")]
pub trait PartialScorer: Send + Sync {
    /// Compute graduated score from game trace.
    fn partial_score(&self, trace: &GameTrace) -> f32;

    /// Breakdown of per-criteria scores.
    fn score_breakdown(&self, trace: &GameTrace) -> Vec<(&'static str, f32)> {
        vec![("total", self.partial_score(trace))]
    }
}

// ── Game Config + Problem Mutation (Plan 191) ────────────────────

/// Generic game configuration for mutation.
///
/// Represents the tunable parameters of a game arena.
/// Each field can be perturbed by a [`ProblemMutator`] to create
/// harder variants for open-ended problem evolution.
#[cfg(feature = "problem_mutator")]
#[derive(Clone, Debug)]
pub struct GameConfig {
    /// Grid/board size (e.g., 9 for 9x9, 15 for 15x15).
    pub grid_size: u32,
    /// Number of opponents/NPCs.
    pub opponent_count: u32,
    /// Maximum steps per episode.
    pub max_steps: u32,
    /// Weight for survival objective in scoring.
    pub survival_weight: f32,
    /// Weight for kill/objective in scoring.
    pub kill_weight: f32,
}

#[cfg(feature = "problem_mutator")]
impl Default for GameConfig {
    fn default() -> Self {
        Self {
            grid_size: 9,
            opponent_count: 1,
            max_steps: 200,
            survival_weight: 0.5,
            kill_weight: 0.5,
        }
    }
}

/// Mutation kind for game configs.
#[cfg(feature = "problem_mutator")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MutationKind {
    /// Shift objective weights (e.g., survival vs kills).
    GoalReweight,
    /// Reduce action space or add constraints.
    ConstrainOutputs,
    /// Vary input parameters (grid size, opponent count).
    GeneralizeInputs,
}

/// A mutated game config with estimated difficulty delta.
#[cfg(feature = "problem_mutator")]
#[derive(Clone, Debug)]
pub struct MutantConfig {
    /// Estimated difficulty increase over seed config.
    pub difficulty_delta: f32,
    /// Which mutation strategy was applied.
    pub mutation_kind: MutationKind,
    /// Human-readable description of the mutation.
    pub description: String,
}

/// Trait for mutating game configs into harder variants.
///
/// Implements the FrontierSmith closed→open problem synthesis:
/// take a base game config and produce progressively harder variants.
/// The arena scheduler feeds these mutated configs to rounds.
#[cfg(feature = "problem_mutator")]
pub trait ProblemMutator: Send + Sync {
    /// Mutate a seed config into variants.
    fn mutate(&self, seed: &GameConfig) -> Vec<MutantConfig>;
}

// ── Best Buddies Drafting (Plan 199) ────────────────────────────

/// Pearson correlation coefficient between two f32 slices.
///
/// SIMD-friendly: single pass, no allocation.
/// Returns 0.0 if either slice has zero variance.
#[inline]
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "pearson_correlation: slices must have equal length"
    );
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f32;
    let sum_a: f32 = a.iter().copied().sum();
    let sum_b: f32 = b.iter().copied().sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = var_a.sqrt() * var_b.sqrt();
    match denom < 1e-12 {
        true => 0.0,
        false => cov / denom,
    }
}

/// Best buddies: mutual nearest neighbors from correlation rows.
///
/// Returns pairs `(i, j)` where row `i`'s best match is row `j` AND row `j`'s
/// best match is row `i`. Results sorted by correlation magnitude (descending),
/// truncated to top-`k`.
pub fn best_buddies(corr_rows: &[&[f32]], k: usize) -> Vec<(usize, usize)> {
    let n = corr_rows.len();

    // Build best-match index for each row
    let mut best_for: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let row = corr_rows[i];
        if row.is_empty() {
            continue;
        }
        let mut best_j = 0;
        let mut best_corr = f32::NEG_INFINITY;
        for (j, &val) in row.iter().enumerate() {
            if val > best_corr {
                best_corr = val;
                best_j = j;
            }
        }
        best_for[i] = Some(best_j);
    }

    // Mutual agreement: i→j AND j→i
    let mut buddies: Vec<(usize, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let j = match best_for[i] {
            Some(j) => j,
            None => continue,
        };
        // j must be within bounds and point back to i
        if j >= n {
            continue;
        }
        match best_for[j] {
            Some(k_idx) if k_idx == i => {
                // Avoid double-counting: only push when i < j
                if i < j {
                    buddies.push((i, j));
                }
            }
            _ => {}
        }
    }

    // Sort by correlation magnitude (descending), keep top-k
    buddies.sort_by(|a, b| {
        let corr_a = corr_rows[a.0][a.1];
        let corr_b = corr_rows[b.0][b.1];
        corr_b
            .partial_cmp(&corr_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    buddies.truncate(k);
    buddies
}

/// Best Buddies filter for speculative decoding.
///
/// Mines mutual agreement between draft and target model marginals.
/// Zero training — purely inference-time correlation.
pub trait BestBuddyAligner: Send + Sync {
    /// Compute mutual agreement score for a token position.
    ///
    /// Returns 1.0 if token is a best buddy (draft prefers it AND target prefers it),
    /// 0.0 otherwise. Continuous in between via EMA correlation.
    fn mutual_agreement(&self, draft_top_k: &[f32], target_top_k: &[f32]) -> f32;

    /// Batch version: compute alignment confidence for all positions.
    ///
    /// `draft_logits` and `target_logits` are `[seq_len × vocab_size]` (flat),
    /// `results` receives `[seq_len]` confidence scores.
    fn batch_alignment_confidence(
        &self,
        draft_logits: &[f32],
        target_logits: &[f32],
        results: &mut [f32],
    );
}

// ── LEO Tests (Plan 155, T7) ────────────────────────────────────

mod tests_leo {
    #[allow(unused_imports)]
    use super::*;

    // -- T5: sigmoid_bounded_q --

    #[test]
    #[cfg(feature = "leo_all_goals")]
    fn test_sigmoid_bounded_q_bounds() {
        // Raw Q = 0 → sigmoid(0) = 0.5
        assert!((sigmoid_bounded_q(0.0) - 0.5).abs() < 1e-6);
        // Large positive → approaches 1.0
        assert!(sigmoid_bounded_q(10.0) > 0.99);
        // Large negative → approaches 0.0
        assert!(sigmoid_bounded_q(-10.0) < 0.01);
        // Symmetry
        assert!((sigmoid_bounded_q(1.0) + sigmoid_bounded_q(-1.0) - 1.0).abs() < 1e-6);
    }

    // -- T1: LeoHead default q_for_goal --

    /// Minimal LeoHead impl for testing.
    #[allow(dead_code)]
    struct DummyLeoHead {
        goals: usize,
        actions: usize,
    }

    #[cfg(feature = "leo_all_goals")]
    impl LeoHead for DummyLeoHead {
        fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
            vec![0.5; self.goals * self.actions]
        }
        fn goal_count(&self) -> usize {
            self.goals
        }
        fn action_count(&self) -> usize {
            self.actions
        }
    }

    #[test]
    #[cfg(feature = "leo_all_goals")]
    fn test_leo_head_q_for_goal() {
        let head = DummyLeoHead {
            goals: 3,
            actions: 4,
        };
        let state = vec![0.0; 8];
        let all_q = head.all_goals_q(&state);
        assert_eq!(all_q.len(), 12); // 3 goals × 4 actions

        let q0 = head.q_for_goal(&all_q, 0);
        assert_eq!(q0.len(), 4);
        assert_eq!(q0, &[0.5; 4]);

        let q2 = head.q_for_goal(&all_q, 2);
        assert_eq!(q2.len(), 4);
    }

    // -- T3: AllGoalsUpdate td_target + loss --

    #[allow(dead_code)]
    struct Updater;
    #[cfg(feature = "leo_all_goals")]
    impl AllGoalsUpdate for Updater {}

    #[test]
    #[cfg(feature = "leo_all_goals")]
    fn test_all_goals_td_target() {
        let upd = Updater;
        let rewards = vec![1.0, 0.0, 0.5]; // 3 goals
        let next_q = vec![
            vec![0.1, 0.2], // goal 0: max = 0.2
            vec![0.3, 0.5], // goal 1: max = 0.5
            vec![0.0, 0.1], // goal 2: max = 0.1
        ];
        let gamma = 0.99;
        let targets = upd.td_target(&rewards, &next_q, gamma);
        assert_eq!(targets.len(), 3);
        assert!((targets[0] - (1.0 + 0.99 * 0.2)).abs() < 1e-5);
        assert!((targets[1] - (0.0 + 0.99 * 0.5)).abs() < 1e-5);
        assert!((targets[2] - (0.5 + 0.99 * 0.1)).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "leo_all_goals")]
    fn test_all_goals_loss() {
        let predicted = vec![vec![0.8], vec![0.2], vec![0.5]];
        let target = vec![1.0, 0.0, 0.5];
        let loss = <Updater as AllGoalsUpdate>::loss(&predicted, &target);
        // (0.8-1.0)² = 0.04, (0.2-0.0)² = 0.04, (0.5-0.5)² = 0.0
        // MSE = (0.04 + 0.04 + 0.0) / 2 / 3 = 0.01333...
        assert!((loss - 0.5 * (0.04 + 0.04 + 0.0) / 3.0).abs() < 1e-6);
    }

    // -- T2: DualLeoMixer --

    #[allow(dead_code)]
    struct Mixer;
    #[cfg(feature = "dual_leo")]
    impl DualLeoMixer for Mixer {}

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_dual_leo_mix() {
        let mixer = Mixer;
        let q_leo = vec![0.4, 0.6, 0.2];
        let q_uvfa = vec![0.1, 0.9, 0.3];
        let alpha = 0.3;
        let mixed = mixer.mix(&q_leo, &q_uvfa, alpha);
        // 0.3*0.4 + 0.7*0.1 = 0.19
        assert!((mixed[0] - 0.19).abs() < 1e-6);
        // 0.3*0.6 + 0.7*0.9 = 0.81
        assert!((mixed[1] - 0.81).abs() < 1e-6);
        // 0.3*0.2 + 0.7*0.3 = 0.27
        assert!((mixed[2] - 0.27).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_dual_leo_default_alpha() {
        let mixer = Mixer;
        assert!((mixer.default_alpha() - 0.3).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_acting_mode_default() {
        assert_eq!(Mixer.acting_mode(), ActingMode::Lc);
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_acting_mode_combine_lc() {
        let mixer = Mixer;
        let q_leo = vec![0.4, 0.6];
        let q_uvfa = vec![0.1, 0.9];
        let combined = mixer.combine(&q_leo, &q_uvfa, 0.3);
        // Same as mix: 0.3*0.4 + 0.7*0.1 = 0.19, 0.3*0.6 + 0.7*0.9 = 0.81
        assert!((combined[0] - 0.19).abs() < 1e-6);
        assert!((combined[1] - 0.81).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_alpha_schedule_fixed() {
        assert!(matches!(Mixer.alpha_schedule(), AlphaSchedule::Fixed(0.3)));
        assert!((Mixer.alpha_at_progress(0.0) - 0.3).abs() < 1e-6);
        assert!((Mixer.alpha_at_progress(0.5) - 0.3).abs() < 1e-6);
        assert!((Mixer.alpha_at_progress(1.0) - 0.3).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_alpha_schedule_linear_anneal() {
        struct AnnealingMixer;
        impl DualLeoMixer for AnnealingMixer {
            fn alpha_schedule(&self) -> AlphaSchedule {
                AlphaSchedule::LinearAnneal {
                    start: 1.0,
                    end: 0.0,
                }
            }
        }
        let m = AnnealingMixer;
        assert!((m.alpha_at_progress(0.0) - 1.0).abs() < 1e-6);
        assert!((m.alpha_at_progress(0.5) - 0.5).abs() < 1e-6);
        assert!((m.alpha_at_progress(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_bc_config_default() {
        assert!(Mixer.bc_config().is_none());
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_bc_config_values() {
        let bc = BcConfig::default();
        assert!((bc.policy_coef - 0.1).abs() < 1e-6);
        assert!((bc.value_coef - 0.0).abs() < 1e-6);
        assert_eq!(bc.target, BcTarget::Argmax);
        assert!(bc.anneal);
    }

    // -- T4: AutocurriculumSampler --

    #[allow(dead_code)]
    struct SimpleAutocurriculum {
        observed: Vec<bool>,
    }

    #[cfg(feature = "dual_leo")]
    impl SimpleAutocurriculum {
        #[allow(dead_code)]
        fn new(total: usize) -> Self {
            Self {
                observed: vec![false; total],
            }
        }
    }

    #[cfg(feature = "dual_leo")]
    impl AutocurriculumSampler for SimpleAutocurriculum {
        fn sample_goal(&self, rng: &mut Rng) -> usize {
            let observed: Vec<_> = self
                .observed
                .iter()
                .enumerate()
                .filter(|&(_, &o)| o)
                .map(|(i, _)| i)
                .collect();
            observed[rng.usize(0..observed.len())]
        }

        fn observe_goal(&mut self, goal: usize) {
            if goal < self.observed.len() {
                self.observed[goal] = true;
            }
        }

        fn observed_count(&self) -> usize {
            self.observed.iter().filter(|&&o| o).count()
        }

        fn total_goal_count(&self) -> usize {
            self.observed.len()
        }
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_autocurriculum_observe_and_count() {
        let mut ac = SimpleAutocurriculum::new(5);
        assert_eq!(ac.observed_count(), 0);
        assert_eq!(ac.total_goal_count(), 5);

        ac.observe_goal(2);
        ac.observe_goal(4);
        assert_eq!(ac.observed_count(), 2);

        // Duplicate observe doesn't change count
        ac.observe_goal(2);
        assert_eq!(ac.observed_count(), 2);
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_autocurriculum_sample_from_observed() {
        let mut ac = SimpleAutocurriculum::new(10);
        ac.observe_goal(3);
        ac.observe_goal(7);
        ac.observe_goal(9);

        let mut rng = Rng::new();
        // Sample many times — should only get 3, 7, or 9
        for _ in 0..100 {
            let g = ac.sample_goal(&mut rng);
            assert!(g == 3 || g == 7 || g == 9, "sampled unobserved goal: {g}");
        }
    }

    // -- T9d: Q(λ) tests --

    #[test]
    #[cfg(feature = "leo_all_goals")]
    fn test_td_target_lambda_no_done() {
        let upd = Updater;
        let rewards = vec![1.0, 0.0];
        let next_q_max = vec![0.5, 0.3];
        let next_lambda_return = vec![0.0, 0.0]; // last step, no future λ-return
        let done = vec![false, false];
        // lambda=0: standard TD
        let targets =
            upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 0.0);
        assert!((targets[0] - (1.0 + 0.99 * 0.5)).abs() < 1e-5);
        assert!((targets[1] - (0.0 + 0.99 * 0.3)).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "leo_all_goals")]
    fn test_td_target_lambda_with_done() {
        let upd = Updater;
        let rewards = vec![1.0, 0.5];
        let next_q_max = vec![0.5, 0.3];
        let next_lambda_return = vec![0.0, 0.0];
        let done = vec![true, false];
        let targets =
            upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 0.5);
        // done[0] = true → target = reward = 1.0
        assert!((targets[0] - 1.0).abs() < 1e-5);
        // done[1] = false, lambda=0.5 → r + γ*(0.5*0.0 + 0.5*0.3) = 0.5 + 0.99*0.15
        assert!((targets[1] - (0.5 + 0.99 * 0.15)).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "leo_all_goals")]
    fn test_td_target_lambda_with_future_return() {
        let upd = Updater;
        let rewards = vec![0.0];
        let next_q_max = vec![0.2];
        let next_lambda_return = vec![1.0]; // future λ-return accumulated
        let done = vec![false];
        // lambda=1.0: pure MC → r + γ * 1.0 * g_next = 0 + 0.99 * 1.0
        let targets_mc =
            upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 1.0);
        assert!((targets_mc[0] - 0.99).abs() < 1e-5);
        // lambda=0.0: one-step TD → r + γ * q_max = 0 + 0.99 * 0.2
        let targets_td =
            upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 0.0);
        assert!((targets_td[0] - (0.99 * 0.2)).abs() < 1e-5);
    }

    // -- T9e: AutocurriculumSampler refinements --

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_update_goals_seen() {
        let ac = SimpleAutocurriculum::new(3);
        let obs_batch = vec![vec![1.0, 0.0, 0.0]];
        let all_goals = vec![
            vec![1.0, 0.0, 0.0], // matches obs
            vec![0.0, 1.0, 0.0], // no match
            vec![0.0, 0.0, 1.0], // no match
        ];
        let current_mask = vec![false; 3];
        let updated = ac.update_goals_seen(&obs_batch, &all_goals, &current_mask);
        assert!(updated[0], "goal 0 should be seen");
        assert!(!updated[1], "goal 1 should not be seen");
        assert!(!updated[2], "goal 2 should not be seen");
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_update_goals_seen_union() {
        let ac = SimpleAutocurriculum::new(3);
        let obs_batch = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let all_goals = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let current_mask = vec![false; 3];
        let updated = ac.update_goals_seen(&obs_batch, &all_goals, &current_mask);
        assert!(updated[0], "goal 0 should be seen");
        assert!(updated[1], "goal 1 should be seen");
        assert!(!updated[2], "goal 2 should not be seen");
    }

    #[test]
    #[cfg(feature = "dual_leo")]
    fn test_autocurriculum_default_methods() {
        let ac = SimpleAutocurriculum::new(5);
        assert_eq!(ac.goals_completed_this_episode(), 0);
        assert!(ac.only_sample_from_seen());
    }
}

// ── SpeculativeGenerator trait-bound test (Plan 193 Phase 1) ──

#[cfg(test)]
mod tests_spec_gen {
    use super::*;

    /// Mock implementation to verify trait compiles and is object-safe.
    struct MockGen;

    impl SpeculativeGenerator for MockGen {
        type Condition = ();
        type Output = usize;
        type Error = ();

        fn generate(
            &mut self,
            _condition: &Self::Condition,
            _rng: &mut fastrand::Rng,
        ) -> Result<Vec<Self::Output>, Self::Error> {
            Ok(vec![1, 2, 3])
        }
    }

    /// Mock pruner to verify GenerativeConstraintPruner compiles.
    struct MockPruner;

    impl GenerativeConstraintPruner<usize> for MockPruner {
        fn is_valid(&self, output: &usize) -> bool {
            *output > 0
        }
    }

    #[test]
    fn test_speculative_generator_trait_bounds() {
        let mut generator = MockGen;
        let mut rng = fastrand::Rng::new();
        let result = generator.generate(&(), &mut rng).unwrap();
        assert_eq!(result, vec![1, 2, 3]);

        // Batch uses default impl
        let batch = generator.generate_batch(&[(), (), ()], &mut rng).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_generative_constraint_pruner_trait_bounds() {
        let pruner = MockPruner;
        assert!(pruner.is_valid(&1));
        assert!(!pruner.is_valid(&0));

        let results = pruner.batch_is_valid(&[0, 1, 2]);
        assert_eq!(results, vec![false, true, true]);
    }
}

// ── Best Buddies Tests (Plan 199) ───────────────────────────────

#[cfg(test)]
mod tests_best_buddies {
    use super::*;

    #[test]
    fn test_pearson_perfect_correlation() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 4.0, 6.0, 8.0, 10.0]; // b = 2*a
        let corr = pearson_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 1e-6, "expected 1.0, got {corr}");
    }

    #[test]
    fn test_pearson_anti_correlation() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 8.0, 6.0, 4.0, 2.0]; // b = -2*a + 12
        let corr = pearson_correlation(&a, &b);
        assert!((corr + 1.0).abs() < 1e-6, "expected -1.0, got {corr}");
    }

    #[test]
    fn test_pearson_zero_correlation() {
        let a = [1.0, -1.0, 1.0, -1.0];
        let b = [1.0, 1.0, -1.0, -1.0]; // orthogonal
        let corr = pearson_correlation(&a, &b);
        assert!(corr.abs() < 1e-6, "expected ~0.0, got {corr}");
    }

    #[test]
    fn test_pearson_zero_variance() {
        let a = [3.0, 3.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let corr = pearson_correlation(&a, &b);
        assert_eq!(corr, 0.0, "zero variance should return 0.0");
    }

    #[test]
    fn test_pearson_empty() {
        let corr = pearson_correlation(&[], &[]);
        assert_eq!(corr, 0.0, "empty slices should return 0.0");
    }

    #[test]
    fn test_best_buddies_simple() {
        // Row 0 best match → 1, Row 1 best match → 0 → mutual pair (0,1)
        // Row 2 best match → 0, but Row 0's best is 1 → not mutual
        let row0: &[f32] = &[0.1, 0.9, 0.2];
        let row1: &[f32] = &[0.8, 0.1, 0.3];
        let row2: &[f32] = &[0.7, 0.2, 0.1];
        let rows: Vec<&[f32]> = vec![row0, row1, row2];
        let buddies = best_buddies(&rows, 10);
        assert_eq!(buddies, vec![(0, 1)]);
    }

    #[test]
    fn test_best_buddies_top_k() {
        let row0: &[f32] = &[0.1, 0.9];
        let row1: &[f32] = &[0.8, 0.1];
        let rows: Vec<&[f32]> = vec![row0, row1];
        // k=1 should truncate to 1 result
        let buddies = best_buddies(&rows, 1);
        assert_eq!(buddies.len(), 1);
        assert_eq!(buddies[0], (0, 1));
    }

    #[test]
    fn test_best_buddies_no_mutual() {
        // Row 0 → 1, Row 1 → 2, Row 2 → 0 (cycle, no mutual)
        let row0: &[f32] = &[0.1, 0.9, 0.2];
        let row1: &[f32] = &[0.1, 0.2, 0.9];
        let row2: &[f32] = &[0.9, 0.1, 0.2];
        let rows: Vec<&[f32]> = vec![row0, row1, row2];
        let buddies = best_buddies(&rows, 10);
        assert!(buddies.is_empty(), "cycle should produce no mutual pairs");
    }
}
