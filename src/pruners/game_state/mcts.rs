//! Generic MCTS (Monte Carlo Tree Search) for any `GameState`.
//!
//! Uses UCB1 selection + pluggable rollout policies. Operates on any `GameState` —
//! game-agnostic. Follows STRATEGA's simplification: only the current
//! player's actions are explored; opponent turns are skipped.
//!
//! # Algorithm
//! 1. **Select**: UCB1 down the tree (only our actions), tracking state inline
//! 2. **Expand**: add one child (our action) — random expansion order
//! 3. **Rollout**: simulate with rollout policy until depth limit or terminal
//! 4. **Backpropagate**: reward from heuristic/terminal state
//!
//! # Rollout Policies (Plan 067 — NFSP/MCTS Duality)
//! - [`RandomRolloutPolicy`]: uniform random baseline (default, backward-compatible)
//! - [`BanditRolloutPolicy`]: ε-greedy guided by bandit Q-values (requires `bandit` feature)
//!
//! Budget is measured in `advance()` calls during expansion + rollout.
//! Selection state tracking (tree walk) is not counted — it's overhead, not search.

use std::marker::PhantomData;

use fastrand::Rng;

use super::{GameState, RandomRolloutPolicy, RolloutPolicy, StateHeuristic};

#[cfg(feature = "bandit")]
use crate::pruners::bandit::BanditStats;

/// UCB1 exploration constant. sqrt(2) is standard; tuned lower for games
/// with high branching factor where exploitation matters more.
const UCB1_C: f32 = 1.414;

/// Maximum tree nodes before stopping. Prevents unbounded memory growth.
const MAX_TREE_SIZE: usize = 10_000;

// ── Tree Node ──────────────────────────────────────────────────

/// A single node in the MCTS search tree.
///
/// Uses index-based parent/child links into a flat `Vec<MCTSNode>`
/// for cache-friendly traversal. Action indices refer to the parent
/// node's `available_actions()` list — the inline state tracker
/// maintains the correct action list at each level.
struct MCTSNode {
    /// Action index that led to this node (None for root).
    action_index: Option<usize>,
    /// Parent node index (None for root).
    parent: Option<usize>,
    /// Child node indices.
    children: Vec<usize>,
    /// Accumulated reward from backpropagation.
    total_reward: f32,
    /// Number of visits through this node.
    visits: usize,
    /// Indices of actions not yet expanded into children.
    unexpanded: Vec<usize>,
}

impl MCTSNode {
    fn new_root(action_count: usize) -> Self {
        Self {
            action_index: None,
            parent: None,
            children: Vec::with_capacity(action_count),
            total_reward: 0.0,
            visits: 0,
            unexpanded: (0..action_count).collect(),
        }
    }

    fn new_child(action_index: usize, parent: usize, action_count: usize) -> Self {
        Self {
            action_index: Some(action_index),
            parent: Some(parent),
            children: Vec::with_capacity(action_count),
            total_reward: 0.0,
            visits: 0,
            unexpanded: (0..action_count).collect(),
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.unexpanded.is_empty()
    }
}

// ── MCTS Search — Core Implementation ─────────────────────────

/// Core MCTS implementation with pluggable rollout policy.
///
/// Shared by [`mcts_search`] (backward-compatible) and [`mcts_search_informed`].
/// The heuristic is passed as a closure for flexibility — callers can wrap
/// [`StateHeuristic`] or use a plain function.
#[allow(clippy::too_many_arguments)]
fn mcts_search_impl<S: GameState>(
    state: &S,
    player_id: u8,
    budget: usize,
    rollout_depth: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
    policy: &mut dyn RolloutPolicy<S>,
    rng: &mut Rng,
) -> S::Action {
    let actions = state.available_actions(player_id);
    assert!(!actions.is_empty(), "mcts_search: no available actions");

    // Single action — no search needed
    if actions.len() == 1 {
        return actions[0].clone();
    }

    // Initialize tree with root node
    let mut nodes = Vec::with_capacity(256);
    nodes.push(MCTSNode::new_root(actions.len()));

    let mut fm_calls = 0usize;

    while fm_calls < budget && nodes.len() < MAX_TREE_SIZE {
        // Each iteration consumes at least 1 budget unit (prevents infinite
        // loop when repeatedly hitting terminal nodes without expansion).
        fm_calls += 1;

        // ── 1. Selection: walk tree, tracking state inline ──────
        let (leaf_idx, leaf_state, leaf_actions) = select_inline(&nodes, state, player_id);

        // ── 2. Expand + Rollout, or Terminal ────────────────────
        let (eval_idx, reward) = if leaf_state.is_terminal() {
            // Terminal leaf — use terminal reward
            (leaf_idx, leaf_state.reward(player_id))
        } else if !nodes[leaf_idx].is_fully_expanded() {
            // Expand one action from the leaf
            expand_and_rollout(
                &mut nodes,
                leaf_idx,
                &leaf_state,
                &leaf_actions,
                player_id,
                rollout_depth,
                heuristic,
                policy,
                rng,
                &mut fm_calls,
                budget,
            )
        } else {
            // Fully expanded leaf with no children (edge case)
            let reward = rollout(
                &leaf_state,
                player_id,
                rollout_depth,
                heuristic,
                policy,
                rng,
                &mut fm_calls,
                budget,
            );
            (leaf_idx, reward)
        };

        // ── 3. Backpropagate ────────────────────────────────────
        backpropagate(&mut nodes, eval_idx, reward);
    }

    // ── 4. Select best action by visit count ────────────────────
    let root = &nodes[0];
    if root.children.is_empty() {
        // No search performed (budget=0) — fallback to first action
        return actions[0].clone();
    }

    let best_child = root
        .children
        .iter()
        .copied()
        .max_by_key(|&ci| nodes[ci].visits)
        .expect("root children non-empty");

    let best_action_idx = nodes[best_child].action_index.unwrap();
    actions[best_action_idx].clone()
}

// ── MCTS Search — Public API ──────────────────────────────────

/// Run MCTS search with UCB1 selection + random rollouts.
///
/// Backward-compatible API. Uses [`RandomRolloutPolicy`] internally.
///
/// # Arguments
/// * `state` — current game state snapshot
/// * `player_id` — which player to optimize for
/// * `budget` — max `advance()` calls during expansion + rollout
/// * `rollout_depth` — max ticks per random rollout
/// * `heuristic` — evaluation function for non-terminal states
/// * `rng` — random number generator for rollouts
///
/// # Returns
/// Best action found within budget (most visited root child).
///
/// # Panics
/// Panics if no actions are available.
pub fn mcts_search<S: GameState>(
    state: &S,
    player_id: u8,
    budget: usize,
    rollout_depth: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
    rng: &mut Rng,
) -> S::Action {
    let mut policy = RandomRolloutPolicy;
    mcts_search_impl(
        state,
        player_id,
        budget,
        rollout_depth,
        heuristic,
        &mut policy,
        rng,
    )
}

/// Run MCTS with informed rollout policy and structured heuristic.
///
/// Plan 067 (NFSP/MCTS Duality): wire backward signal (bandit Q-values)
/// into forward search (MCTS rollouts) for informed action selection.
///
/// # Arguments
/// * `state` — current game state snapshot
/// * `player_id` — which player to optimize for
/// * `budget` — max `advance()` calls during expansion + rollout
/// * `rollout_depth` — max ticks per rollout
/// * `heuristic` — structured heuristic for non-terminal evaluation
/// * `policy` — rollout policy for action selection during simulation
/// * `rng` — random number generator
///
/// # Returns
/// Best action found within budget (most visited root child).
///
/// # Example
/// ```ignore
/// use katgpt_rs::pruners::game_state::{mcts_search_informed, RandomRolloutPolicy, StateHeuristic};
///
/// struct MyHeuristic;
/// impl StateHeuristic<MyState> for MyHeuristic {
///     fn evaluate(&self, state: &MyState, player_id: u8) -> f32 { 0.5 }
/// }
///
/// let mut policy = RandomRolloutPolicy;
/// let heuristic = MyHeuristic;
/// let action = mcts_search_informed(&state, 0, 200, 10, &heuristic, &mut policy, &mut rng);
/// ```
pub fn mcts_search_informed<S: GameState>(
    state: &S,
    player_id: u8,
    budget: usize,
    rollout_depth: usize,
    heuristic: &dyn StateHeuristic<S>,
    policy: &mut dyn RolloutPolicy<S>,
    rng: &mut Rng,
) -> S::Action {
    let h = |s: &S, pid: u8| heuristic.evaluate(s, pid);
    mcts_search_impl(state, player_id, budget, rollout_depth, &h, policy, rng)
}

// ── Selection ──────────────────────────────────────────────────

/// Walk the tree from root, tracking state inline.
///
/// Returns `(leaf_index, leaf_state, leaf_actions)` where:
/// - `leaf_index` is the node to expand or evaluate
/// - `leaf_state` is the game state at that node
/// - `leaf_actions` are the available actions at that state
///
/// State tracking calls to `advance()` are NOT counted toward budget
/// (tree walk overhead, not search).
fn select_inline<S: GameState>(
    nodes: &[MCTSNode],
    root_state: &S,
    player_id: u8,
) -> (usize, S, Vec<S::Action>) {
    let mut idx = 0;
    let mut state = root_state.clone();
    let mut actions = state.available_actions(player_id);

    loop {
        let node = &nodes[idx];

        // Terminal or not fully expanded → this is our leaf
        if state.is_terminal() || !node.is_fully_expanded() {
            return (idx, state, actions);
        }

        // Fully expanded but no children → edge case, stop here
        if node.children.is_empty() {
            return (idx, state, actions);
        }

        // Fully expanded with children → select best child by UCB1
        let parent_visits = node.visits.max(1); // Guard against ln(0)
        let best_child = node
            .children
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = ucb1_score(nodes[a].total_reward, nodes[a].visits, parent_visits);
                let sb = ucb1_score(nodes[b].total_reward, nodes[b].visits, parent_visits);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("children non-empty");

        // Advance state to the selected child using parent's action list
        let action_idx = nodes[best_child].action_index.unwrap();
        state = state.advance(&actions[action_idx], player_id);
        actions = state.available_actions(player_id);
        idx = best_child;
    }
}

// ── Expansion + Rollout ───────────────────────────────────────

/// Expand one action from the leaf node and run a rollout from the child.
///
/// The expansion order (which unexpanded action to try) remains random.
/// Only the rollout uses the pluggable [`RolloutPolicy`].
///
/// Returns `(child_index, reward)`.
#[allow(clippy::too_many_arguments)]
fn expand_and_rollout<S: GameState>(
    nodes: &mut Vec<MCTSNode>,
    leaf_idx: usize,
    leaf_state: &S,
    leaf_actions: &[S::Action],
    player_id: u8,
    rollout_depth: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
    policy: &mut dyn RolloutPolicy<S>,
    rng: &mut Rng,
    fm_calls: &mut usize,
    budget: usize,
) -> (usize, f32) {
    // Pick a random unexpanded action (expansion order is random)
    let node = &mut nodes[leaf_idx];
    let pick = rng.usize(0..node.unexpanded.len());
    let action_idx = node.unexpanded.swap_remove(pick);
    let action = &leaf_actions[action_idx];

    // Advance to child state (1 FM call)
    let child_state = leaf_state.advance(action, player_id);
    *fm_calls += 1;

    // Create child node
    let child_actions_len = child_state.available_actions(player_id).len();
    let child_idx = nodes.len();
    nodes.push(MCTSNode::new_child(action_idx, leaf_idx, child_actions_len));
    nodes[leaf_idx].children.push(child_idx);

    // Rollout from child state
    let reward = if child_state.is_terminal() {
        child_state.reward(player_id)
    } else {
        rollout(
            &child_state,
            player_id,
            rollout_depth,
            heuristic,
            policy,
            rng,
            fm_calls,
            budget,
        )
    };

    (child_idx, reward)
}

/// Run a rollout from the given state using the provided policy.
///
/// Selects actions via [`RolloutPolicy`] until depth limit, terminal,
/// or budget exhausted. Returns terminal reward or heuristic evaluation.
#[allow(clippy::too_many_arguments)]
fn rollout<S: GameState>(
    state: &S,
    player_id: u8,
    max_depth: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
    policy: &mut dyn RolloutPolicy<S>,
    rng: &mut Rng,
    fm_calls: &mut usize,
    budget: usize,
) -> f32 {
    let mut current = state.clone();

    for _ in 0..max_depth {
        if *fm_calls >= budget || current.is_terminal() {
            break;
        }

        let actions = current.available_actions(player_id);
        if actions.is_empty() {
            break;
        }

        let pick = policy.select(&current, &actions, player_id, rng);
        current = current.advance(&actions[pick], player_id);
        *fm_calls += 1;
    }

    match current.is_terminal() {
        true => current.reward(player_id),
        false => heuristic(&current, player_id),
    }
}

// ── Backpropagation ────────────────────────────────────────────

/// Backpropagate reward from a node to the root.
fn backpropagate(nodes: &mut [MCTSNode], mut idx: usize, reward: f32) {
    loop {
        nodes[idx].visits += 1;
        nodes[idx].total_reward += reward;
        idx = match nodes[idx].parent {
            Some(p) => p,
            None => break,
        };
    }
}

/// Compute UCB1 score for a child node.
///
/// `total_reward` = accumulated reward, `visits` = visit count,
/// `parent_visits` = parent's visit count.
/// Returns `f32::INFINITY` for unvisited nodes (exploration priority).
#[inline]
fn ucb1_score(total_reward: f32, visits: usize, parent_visits: usize) -> f32 {
    match visits {
        0 => f32::INFINITY,
        _ => {
            let exploit = total_reward / visits as f32;
            let explore = UCB1_C * (parent_visits as f32).ln().sqrt() / (visits as f32).sqrt();
            exploit + explore
        }
    }
}

// ── Bandit Rollout Policy ─────────────────────────────────────

/// ε-greedy rollout policy guided by bandit Q-values.
///
/// Plan 067 (NFSP/MCTS Duality): wires backward signal (bandit Q-values
/// accumulated across episodes) into forward search (MCTS rollouts).
/// This is the AlphaZero pattern, but modelless — no neural net, just
/// bandit statistics.
///
/// # Type Parameters
/// * `S` — game state type
/// * `F` — action-to-index mapping closure
///
/// # Action Mapping
/// The `action_to_index` closure maps `S::Action` → bandit arm index.
/// For [`BomberAction`](crate::pruners::bomber::BomberAction), use
/// `|a: &BomberAction| a.as_usize()`.
///
/// # Exploration (ε)
/// - `ε = 0.0`: pure exploit (always pick highest Q-value action)
/// - `ε = 0.2`: 80% exploit, 20% explore (good default)
/// - `ε = 1.0`: pure random (equivalent to [`RandomRolloutPolicy`])
#[cfg(feature = "bandit")]
pub struct BanditRolloutPolicy<'a, S, F>
where
    S: GameState,
    F: Fn(&S::Action) -> usize,
{
    /// Bandit statistics with accumulated Q-values.
    stats: &'a BanditStats,
    /// Exploration probability ∈ [0.0, 1.0].
    epsilon: f32,
    /// Maps game actions to bandit arm indices.
    action_to_index: F,
    /// Marker for the game state type (no ownership implied).
    _phantom: PhantomData<fn() -> S>,
}

#[cfg(feature = "bandit")]
impl<'a, S, F> BanditRolloutPolicy<'a, S, F>
where
    S: GameState,
    F: Fn(&S::Action) -> usize,
{
    /// Create a new bandit-guided rollout policy.
    ///
    /// # Arguments
    /// * `stats` — bandit statistics with accumulated Q-values from prior episodes
    /// * `epsilon` — exploration probability (0.0 = pure exploit, 1.0 = pure random)
    /// * `action_to_index` — maps game actions to bandit arm indices
    ///
    /// # Panics
    /// Panics if `epsilon` is not in `[0.0, 1.0]`.
    pub fn new(stats: &'a BanditStats, epsilon: f32, action_to_index: F) -> Self {
        assert!(
            (0.0..=1.0).contains(&epsilon),
            "epsilon must be in [0.0, 1.0], got {epsilon}"
        );
        Self {
            stats,
            epsilon,
            action_to_index,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "bandit")]
impl<S, F> RolloutPolicy<S> for BanditRolloutPolicy<'_, S, F>
where
    S: GameState,
    F: Fn(&S::Action) -> usize,
{
    fn select(
        &mut self,
        _state: &S,
        actions: &[S::Action],
        _player_id: u8,
        rng: &mut Rng,
    ) -> usize {
        // ε-greedy: explore with probability ε, exploit with probability (1-ε)
        if rng.usize(0..1000) < (self.epsilon * 1000.0) as usize {
            // Explore: random action
            rng.usize(0..actions.len())
        } else {
            // Exploit: pick action with highest bandit Q-value
            let mut best_idx = 0;
            let mut best_q = f32::NEG_INFINITY;
            for (i, action) in actions.iter().enumerate() {
                let arm = (self.action_to_index)(action);
                let q = self.stats.q_value(arm);
                if q > best_q {
                    best_q = q;
                    best_idx = i;
                }
            }
            best_idx
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test Doubles ───────────────────────────────────────────

    /// Minimal 2-action game: both actions lead to terminal states.
    /// true → reward 1.0 (win), false → reward 0.0 (lose).
    #[derive(Clone)]
    struct TwoActionState {
        acted: bool,
        chose_win: bool,
    }

    impl GameState for TwoActionState {
        type Action = bool;

        fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
            match self.acted {
                true => vec![], // terminal, no actions
                false => vec![false, true],
            }
        }

        fn advance(&self, action: &Self::Action, _player_id: u8) -> Self {
            Self {
                acted: true,
                chose_win: *action,
            }
        }

        fn is_terminal(&self) -> bool {
            self.acted
        }

        fn reward(&self, _player_id: u8) -> f32 {
            if self.chose_win { 1.0 } else { 0.0 }
        }

        fn tick(&self) -> u32 {
            self.acted as u32
        }
    }

    /// Multi-step game: each "true" action accumulates 0.1 bonus.
    /// Terminal after `max_tick` steps.
    #[derive(Clone)]
    struct DeepState {
        tick: u32,
        max_tick: u32,
        cumulative: f32,
    }

    impl GameState for DeepState {
        type Action = bool;

        fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
            if self.is_terminal() {
                vec![]
            } else {
                vec![false, true]
            }
        }

        fn advance(&self, action: &Self::Action, _player_id: u8) -> Self {
            let bonus = if *action { 0.1 } else { 0.0 };
            Self {
                tick: self.tick + 1,
                max_tick: self.max_tick,
                cumulative: self.cumulative + bonus,
            }
        }

        fn is_terminal(&self) -> bool {
            self.tick >= self.max_tick
        }

        fn reward(&self, _player_id: u8) -> f32 {
            self.cumulative
        }

        fn tick(&self) -> u32 {
            self.tick
        }
    }

    /// Closure-based heuristic adapter for `mcts_search_informed` tests.
    struct FnHeuristic<F>(F);

    impl<S: GameState, F: Fn(&S, u8) -> f32> StateHeuristic<S> for FnHeuristic<F> {
        fn evaluate(&self, state: &S, player_id: u8) -> f32 {
            (self.0)(state, player_id)
        }
    }

    // ── UCB1 Tests ─────────────────────────────────────────────

    #[test]
    fn ucb1_unvisited_is_infinite() {
        let score = ucb1_score(0.0, 0, 10);
        assert!(score.is_infinite());
    }

    #[test]
    fn ucb1_visited_is_finite() {
        let score = ucb1_score(1.0, 10, 100);
        assert!(score.is_finite());
    }

    #[test]
    fn ucb1_more_visits_less_explore() {
        let few = ucb1_score(0.5, 5, 100);
        let many = ucb1_score(0.5, 50, 100);
        assert!(
            few > many,
            "fewer visits should have higher exploration bonus: {few} vs {many}"
        );
    }

    #[test]
    fn ucb1_higher_reward_higher_score() {
        let low = ucb1_score(0.2, 10, 100);
        let high = ucb1_score(0.8, 10, 100);
        assert!(
            high > low,
            "higher reward should have higher UCB1 score: {high} vs {low}"
        );
    }

    // ── MCTS Search Tests (backward-compatible API) ────────────

    #[test]
    fn mcts_finds_winning_action() {
        let state = TwoActionState {
            acted: false,
            chose_win: false,
        };
        let mut rng = Rng::with_seed(42);
        let action = mcts_search(&state, 0, 500, 10, &|_s, _p| 0.5, &mut rng);
        assert!(action, "should find the winning action (true)");
    }

    #[test]
    fn mcts_single_action_returns_immediately() {
        #[derive(Clone)]
        struct OneAction;

        impl GameState for OneAction {
            type Action = u8;

            fn available_actions(&self, _pid: u8) -> Vec<u8> {
                vec![42]
            }

            fn advance(&self, _a: &u8, _pid: u8) -> Self {
                Self
            }

            fn is_terminal(&self) -> bool {
                true
            }

            fn reward(&self, _pid: u8) -> f32 {
                1.0
            }

            fn tick(&self) -> u32 {
                0
            }
        }

        let state = OneAction;
        let mut rng = Rng::with_seed(42);
        let action = mcts_search(&state, 0, 100, 10, &|_, _| 0.5, &mut rng);
        assert_eq!(action, 42);
    }

    #[test]
    fn mcts_completes_within_budget() {
        let state = DeepState {
            tick: 0,
            max_tick: 100,
            cumulative: 0.0,
        };
        let mut rng = Rng::with_seed(42);
        let _ = mcts_search(&state, 0, 50, 10, &|_, _| 0.5, &mut rng);
        // Should complete without hanging (budget=50 limits iterations)
    }

    #[test]
    fn mcts_prefers_better_heuristic() {
        #[derive(Clone)]
        struct BiasedState {
            last_action: Option<bool>,
        }

        impl GameState for BiasedState {
            type Action = bool;

            fn available_actions(&self, _pid: u8) -> Vec<bool> {
                vec![false, true]
            }

            fn advance(&self, a: &bool, _pid: u8) -> Self {
                Self {
                    last_action: Some(*a),
                }
            }

            fn is_terminal(&self) -> bool {
                self.last_action.is_some()
            }

            fn reward(&self, _pid: u8) -> f32 {
                match self.last_action {
                    Some(true) => 1.0,
                    Some(false) => 0.0,
                    None => 0.5,
                }
            }

            fn tick(&self) -> u32 {
                if self.last_action.is_some() { 1 } else { 0 }
            }
        }

        let state = BiasedState { last_action: None };
        let mut rng = Rng::with_seed(42);
        let action = mcts_search(
            &state,
            0,
            200,
            5,
            &|s: &BiasedState, _| match s.last_action {
                Some(true) => 0.9,
                Some(false) => 0.1,
                None => 0.5,
            },
            &mut rng,
        );
        assert!(
            action,
            "MCTS should prefer the action with better heuristic"
        );
    }

    #[test]
    fn mcts_deep_state_find_good_policy() {
        let state = DeepState {
            tick: 0,
            max_tick: 5,
            cumulative: 0.0,
        };
        let mut rng = Rng::with_seed(42);
        let action = mcts_search(
            &state,
            0,
            500,
            10,
            &|s: &DeepState, _| s.cumulative / 5.0,
            &mut rng,
        );
        assert!(action, "should prefer the rewarding action in deep state");
    }

    // ── Backpropagation Tests ──────────────────────────────────

    #[test]
    fn backpropagate_updates_chain() {
        let mut nodes = vec![
            MCTSNode::new_root(2),
            MCTSNode::new_child(0, 0, 2),
            MCTSNode::new_child(1, 1, 2),
        ];
        backpropagate(&mut nodes, 2, 1.0);
        assert_eq!(nodes[2].visits, 1);
        assert!((nodes[2].total_reward - 1.0).abs() < f32::EPSILON);
        assert_eq!(nodes[1].visits, 1);
        assert!((nodes[1].total_reward - 1.0).abs() < f32::EPSILON);
        assert_eq!(nodes[0].visits, 1);
        assert!((nodes[0].total_reward - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn backpropagate_accumulates() {
        let mut nodes = vec![MCTSNode::new_root(2), MCTSNode::new_child(0, 0, 2)];
        backpropagate(&mut nodes, 1, 1.0);
        backpropagate(&mut nodes, 1, 0.5);
        assert_eq!(nodes[1].visits, 2);
        assert!((nodes[1].total_reward - 1.5).abs() < f32::EPSILON);
        assert_eq!(nodes[0].visits, 2);
        assert!((nodes[0].total_reward - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn backpropagate_root_only() {
        let mut nodes = vec![MCTSNode::new_root(2)];
        backpropagate(&mut nodes, 0, 0.7);
        assert_eq!(nodes[0].visits, 1);
        assert!((nodes[0].total_reward - 0.7).abs() < f32::EPSILON);
    }

    // ── Informed MCTS Tests ────────────────────────────────────

    #[test]
    fn mcts_informed_with_random_finds_winning_action() {
        let state = TwoActionState {
            acted: false,
            chose_win: false,
        };
        let mut rng = Rng::with_seed(42);
        let mut policy = RandomRolloutPolicy;
        let heuristic = FnHeuristic(|_s: &TwoActionState, _p: u8| 0.5f32);
        let action = mcts_search_informed(&state, 0, 500, 10, &heuristic, &mut policy, &mut rng);
        assert!(
            action,
            "informed search with random policy should find winning action"
        );
    }

    #[test]
    fn mcts_informed_with_random_deep_state() {
        let state = DeepState {
            tick: 0,
            max_tick: 5,
            cumulative: 0.0,
        };
        let mut rng = Rng::with_seed(42);
        let mut policy = RandomRolloutPolicy;
        let heuristic = FnHeuristic(|s: &DeepState, _| s.cumulative / 5.0);
        let action = mcts_search_informed(&state, 0, 500, 10, &heuristic, &mut policy, &mut rng);
        assert!(
            action,
            "informed search should prefer rewarding action in deep state"
        );
    }

    // ── Bandit Rollout Policy Tests ────────────────────────────

    #[cfg(feature = "bandit")]
    #[test]
    fn bandit_rollout_exploits_high_q() {
        let mut stats = BanditStats::new(2);
        // Arm 0 (false) has high Q, arm 1 (true) has low Q
        for _ in 0..100 {
            stats.update(0, 1.0);
        }
        for _ in 0..100 {
            stats.update(1, 0.0);
        }

        let action_to_index = |action: &bool| if *action { 1 } else { 0 };
        let mut policy = BanditRolloutPolicy::new(&stats, 0.0, action_to_index);

        let state = TwoActionState {
            acted: false,
            chose_win: false,
        };
        let actions = state.available_actions(0);
        let mut rng = Rng::with_seed(42);

        // With ε=0, should always select arm 0 (false, highest Q)
        for _ in 0..50 {
            let idx = policy.select(&state, &actions, 0, &mut rng);
            assert_eq!(idx, 0, "ε=0 should always exploit highest Q action");
        }
    }

    #[cfg(feature = "bandit")]
    #[test]
    fn bandit_rollout_explores_with_epsilon() {
        let mut stats = BanditStats::new(2);
        stats.update(0, 1.0);
        stats.update(1, 0.0);

        let action_to_index = |action: &bool| if *action { 1 } else { 0 };
        let mut policy = BanditRolloutPolicy::new(&stats, 1.0, action_to_index);

        let state = TwoActionState {
            acted: false,
            chose_win: false,
        };
        let actions = state.available_actions(0);
        let mut rng = Rng::with_seed(42);

        let mut found_explore = false;
        for _ in 0..100 {
            let idx = policy.select(&state, &actions, 0, &mut rng);
            if idx == 1 {
                found_explore = true;
                break;
            }
        }
        assert!(found_explore, "ε=1.0 should explore (select non-best arm)");
    }

    #[cfg(feature = "bandit")]
    #[test]
    fn bandit_rollout_balances_exploit_explore() {
        let mut stats = BanditStats::new(2);
        for _ in 0..50 {
            stats.update(0, 0.9);
        }
        for _ in 0..50 {
            stats.update(1, 0.1);
        }

        let action_to_index = |action: &bool| if *action { 1 } else { 0 };
        let mut policy = BanditRolloutPolicy::new(&stats, 0.3, action_to_index);

        let state = TwoActionState {
            acted: false,
            chose_win: false,
        };
        let actions = state.available_actions(0);
        let mut rng = Rng::with_seed(42);

        let mut exploit_count = 0usize;
        let total = 1000;
        for _ in 0..total {
            let idx = policy.select(&state, &actions, 0, &mut rng);
            if idx == 0 {
                exploit_count += 1;
            }
        }

        let exploit_rate = exploit_count as f32 / total as f32;
        // With ε=0.3, expect ~70% exploit. Allow wide tolerance for randomness.
        assert!(
            exploit_rate > 0.5 && exploit_rate < 0.9,
            "exploit rate should be around 70% (got {exploit_rate:.2}), ε=0.3"
        );
    }

    #[cfg(feature = "bandit")]
    #[test]
    fn mcts_informed_with_bandit_finds_winning_action() {
        let mut stats = BanditStats::new(2);
        // Teach the bandit that true (arm 1) is rewarding
        for _ in 0..100 {
            stats.update(1, 1.0);
        }
        for _ in 0..100 {
            stats.update(0, 0.0);
        }

        let action_to_index = |action: &bool| if *action { 1 } else { 0 };
        let mut policy = BanditRolloutPolicy::new(&stats, 0.2, action_to_index);

        let state = TwoActionState {
            acted: false,
            chose_win: false,
        };
        let mut rng = Rng::with_seed(42);
        let heuristic = FnHeuristic(|_s: &TwoActionState, _p: u8| 0.5f32);

        let action = mcts_search_informed(&state, 0, 500, 10, &heuristic, &mut policy, &mut rng);
        assert!(
            action,
            "bandit-guided MCTS should find the winning action (true)"
        );
    }

    #[cfg(feature = "bandit")]
    #[test]
    #[should_panic(expected = "epsilon must be in [0.0, 1.0]")]
    fn bandit_rollout_rejects_invalid_epsilon() {
        let stats = BanditStats::new(2);
        let action_to_index = |action: &bool| if *action { 1 } else { 0 };
        let _policy: BanditRolloutPolicy<'_, TwoActionState, _> =
            BanditRolloutPolicy::new(&stats, 1.5, action_to_index);
    }
}
