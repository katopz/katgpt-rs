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
    /// Single-pass accumulation — zero allocation.
    pub fn avg_action_space_for(&self, player_id: u8) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for &(_, pid, n) in &self.entries {
            if pid == player_id {
                sum += n as f32;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
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

// ── LEO All-Goals Traits (Plan 155) ──────────────────────────────
//
// LEO (Learn Everything All at Once) outputs Q-values for ALL goals
// simultaneously instead of conditioning on a single goal (UVFA-style).
// Ref: Matthews et al. (2026) "Learn Everything All at Once"
//
// Feature gates:
//   leo_all_goals — LeoHead + AllGoalsUpdate + sigmoid_bounded_q
//   dual_leo      — + DualLeoMixer + AutocurriculumSampler

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
/// Ref: Matthews et al. (2026) "Learn Everything All at Once"
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

    /// Default α = 0.3 (from paper sweep on Craftax).
    fn default_alpha(&self) -> f32 {
        0.3
    }
}

/// Goal sampling from previously observed goals only.
///
/// "We sample goals only from goals observed at least once in the past,
/// to prevent completely out-of-reach goals being sampled."
/// — Matthews et al. (2026)
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
}

// ── LEO Tests (Plan 155, T7) ────────────────────────────────────

#[cfg(test)]
mod tests_leo {
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

    // -- T4: AutocurriculumSampler --

    struct SimpleAutocurriculum {
        observed: Vec<bool>,
    }

    #[cfg(feature = "dual_leo")]
    impl SimpleAutocurriculum {
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
}
