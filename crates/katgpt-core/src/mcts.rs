//! Generic MCTS (Monte Carlo Tree Search) for any [`GameState`].
//!
//! Substrate extraction (Plan 008 Step 6, 2026-06-28): the pure game-agnostic
//! MCTS algorithm — `MCTSNode`, `mcts_search`, `mcts_search_informed`, and all
//! internal helpers — moved here verbatim from `katgpt-rs/src/pruners/game_state/mcts.rs`.
//! Composition that needs root-only types (`BanditRolloutPolicy`, which depends
//! on `crate::pruners::bandit::BanditStats`) stays in the root crate.
//!
//! # Algorithm
//! 1. **Select**: UCB1 down the tree (only our actions), tracking state inline
//! 2. **Expand**: add one child (our action) — random expansion order
//! 3. **Rollout**: simulate with rollout policy until depth limit or terminal
//! 4. **Backpropagate**: reward from heuristic/terminal state
//!
//! Budget is measured in `advance()` calls during expansion + rollout.
//! Selection state tracking (tree walk) is not counted — it's overhead, not search.
//!
//! # Traits
//! Operates on [`GameState`], [`RolloutPolicy`], and [`StateHeuristic`] from
//! [`crate::traits`]. Any crate can `cargo add katgpt-core` and run MCTS over
//! its own `GameState` implementation.

use arrayvec::ArrayVec;
use fastrand::Rng;

use crate::traits::{GameState, RandomRolloutPolicy, RolloutPolicy, StateHeuristic};

/// UCB1 exploration constant. sqrt(2) is standard; tuned lower for games
/// with high branching factor where exploitation matters more.
const UCB1_C: f32 = 1.414;

/// Maximum tree nodes before stopping. Prevents unbounded memory growth.
const MAX_TREE_SIZE: usize = 10_000;

/// Maximum number of unexpanded actions per node (ArrayVec capacity).
/// Must accommodate the highest branching factor across all game domains.
/// Bomber=6, Grid=5, Raid=~9. 16 provides comfortable headroom.
///
/// Ported from riir-engine (Plan 008 Phase 2.6, 2026-06-28): switching
/// `children`/`unexpanded` from `Vec<usize>` to `ArrayVec<usize, MAX_UNEXPANDED>`
/// eliminates per-node heap allocation — a genuine hot-path win because the
/// tree allocates one node per MCTS iteration. Bit-identical values; only the
/// backing storage changes. Callers whose `available_actions()` can exceed 16
/// will hit the `assert!` in `new_root`/`new_child` — bump this const if a new
/// game domain needs more headroom.
const MAX_UNEXPANDED: usize = 16;

// ── Tree Node ──────────────────────────────────────────────────

/// A single node in the MCTS search tree.
///
/// Uses index-based parent/child links into a flat `Vec<MCTSNode>`
/// for cache-friendly traversal. Action indices refer to the parent
/// node's `available_actions()` list — the inline state tracker
/// maintains the correct action list at each level.
///
/// Fields ordered by size/alignment (u32 → usize → Vec → Option) to minimize padding.
pub(crate) struct MCTSNode {
    /// Accumulated reward from backpropagation.
    total_reward: f32,
    /// Number of visits through this node.
    visits: usize,
    /// Action index that led to this node (None for root).
    action_index: Option<usize>,
    /// Parent node index (None for root).
    parent: Option<usize>,
    /// Child node indices. Stack-allocated (no heap) — capacity MAX_UNEXPANDED.
    children: ArrayVec<usize, { MAX_UNEXPANDED }>,
    /// Indices of actions not yet expanded into children. Stack-allocated.
    unexpanded: ArrayVec<usize, { MAX_UNEXPANDED }>,
}

impl MCTSNode {
    fn new_root(action_count: usize) -> Self {
        assert!(
            action_count <= MAX_UNEXPANDED,
            "MCTSNode::new_root: action_count ({action_count}) exceeds unexpanded capacity ({MAX_UNEXPANDED})"
        );
        Self {
            total_reward: 0.0,
            visits: 0,
            action_index: None,
            parent: None,
            children: ArrayVec::new(),
            unexpanded: (0..action_count).collect(),
        }
    }

    fn new_child(action_index: usize, parent: usize, action_count: usize) -> Self {
        assert!(
            action_count <= MAX_UNEXPANDED,
            "MCTSNode::new_child: action_count ({action_count}) exceeds unexpanded capacity ({MAX_UNEXPANDED})"
        );
        Self {
            total_reward: 0.0,
            visits: 0,
            action_index: Some(action_index),
            parent: Some(parent),
            children: ArrayVec::new(),
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
    // Pre-allocate action buffers — reused across all MCTS iterations to avoid
    // per-call Vec allocation. Capacity 8 covers most board-game action spaces.
    let mut action_buf = Vec::with_capacity(8);
    let mut rollout_buf = Vec::with_capacity(8);

    state.available_actions_into(player_id, &mut action_buf);
    assert!(!action_buf.is_empty(), "mcts_search: no available actions");

    // Single action — no search needed
    if action_buf.len() == 1 {
        return action_buf[0].clone();
    }

    // Initialize tree with root node
    let root_action_count = action_buf.len();
    let mut nodes = Vec::with_capacity(256);
    nodes.push(MCTSNode::new_root(root_action_count));

    let mut fm_calls = 0usize;

    while fm_calls < budget && nodes.len() < MAX_TREE_SIZE {
        // Each iteration consumes at least 1 budget unit (prevents infinite
        // loop when repeatedly hitting terminal nodes without expansion).
        fm_calls += 1;

        // ── 1. Selection: walk tree, tracking state inline ──────
        let (leaf_idx, leaf_state) = select_inline(&nodes, state, player_id, &mut action_buf);

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
                &action_buf,
                player_id,
                rollout_depth,
                heuristic,
                policy,
                rng,
                &mut fm_calls,
                budget,
                &mut rollout_buf,
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
                &mut rollout_buf,
            );
            (leaf_idx, reward)
        };

        // ── 3. Backpropagate ────────────────────────────────────
        backpropagate(&mut nodes, eval_idx, reward);
    }

    // ── 4. Select best action by visit count ────────────────────
    // Re-fetch root actions for best-action lookup (action_buf still holds them).
    state.available_actions_into(player_id, &mut action_buf);

    let root = &nodes[0];
    if root.children.is_empty() {
        // No search performed (budget=0) — fallback to first action
        return action_buf[0].clone();
    }

    let best_child = root
        .children
        .iter()
        .copied()
        .max_by_key(|&ci| nodes[ci].visits)
        .expect("root children non-empty");

    let best_action_idx = nodes[best_child].action_index.unwrap();
    action_buf[best_action_idx].clone()
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
/// use katgpt_core::mcts::mcts_search_informed;
/// use katgpt_core::traits::{RandomRolloutPolicy, StateHeuristic};
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
    action_buf: &mut Vec<S::Action>,
) -> (usize, S) {
    let mut idx = 0;
    let mut state = root_state.clone();
    state.available_actions_into(player_id, action_buf);

    loop {
        let node = &nodes[idx];

        // Terminal or not fully expanded → this is our leaf
        if state.is_terminal() || !node.is_fully_expanded() {
            return (idx, state);
        }

        // Fully expanded but no children → edge case, stop here
        if node.children.is_empty() {
            return (idx, state);
        }

        // Fully expanded with children → select best child by UCB1.
        // Pre-compute ln(parent_visits) once — reused across all child comparisons
        // in this iteration (avoids redundant `.ln()` per child).
        let ln_parent = (node.visits.max(1) as f32).ln();
        let best_child = node
            .children
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = ucb1_score_cached(nodes[a].total_reward, nodes[a].visits, ln_parent);
                let sb = ucb1_score_cached(nodes[b].total_reward, nodes[b].visits, ln_parent);
                sa.total_cmp(&sb)
            })
            .expect("children non-empty");

        // Advance state to the selected child using parent's action list
        let action_idx = nodes[best_child].action_index.unwrap();
        state = state.advance(&action_buf[action_idx], player_id);
        state.available_actions_into(player_id, action_buf);
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
    rollout_buf: &mut Vec<S::Action>,
) -> (usize, f32) {
    // Pick a random unexpanded action (expansion order is random)
    let node = &mut nodes[leaf_idx];
    let pick = rng.usize(0..node.unexpanded.len());
    let action_idx = node.unexpanded.swap_remove(pick);
    let action = &leaf_actions[action_idx];

    // Advance to child state (1 FM call)
    let child_state = leaf_state.advance(action, player_id);
    *fm_calls += 1;

    // Create child node — use action_space_size to avoid allocating
    let child_actions_len = child_state.action_space_size(player_id);
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
            rollout_buf,
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
    action_buf: &mut Vec<S::Action>,
) -> f32 {
    let mut current = state.clone();

    for _ in 0..max_depth {
        if *fm_calls >= budget || current.is_terminal() {
            break;
        }

        current.available_actions_into(player_id, action_buf);
        if action_buf.is_empty() {
            break;
        }

        let pick = policy.select(&current, action_buf, player_id, rng);
        current = current.advance(&action_buf[pick], player_id);
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
///
/// Note: the hot path (`select_inline`) uses [`ucb1_score_cached`] which
/// pre-computes `ln(parent_visits)`. This scalar form is retained for tests
/// and as the reference implementation.
#[cfg(test)]
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

/// UCB1 with pre-computed `ln(parent_visits)` to avoid redundant computation
/// per child in the selection loop. Used by `select_inline` — mathematically
/// identical to `ucb1_score`; just hoists the `.ln()` call out of the
/// per-child comparison closure. Ported from riir-engine (Plan 008 Phase 2.6).
#[inline]
fn ucb1_score_cached(total_reward: f32, visits: usize, ln_parent: f32) -> f32 {
    if visits == 0 {
        return f32::INFINITY;
    }
    let exploit = total_reward / visits as f32;
    let explore = UCB1_C * ln_parent.sqrt() / (visits as f32).sqrt();
    exploit + explore
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
}
