//! Generic MCTS (Monte Carlo Tree Search) for any `GameState`.
//!
//! Uses UCB1 selection + random rollouts. Operates on any `GameState` —
//! game-agnostic. Follows STRATEGA's simplification: only the current
//! player's actions are explored; opponent turns are skipped.
//!
//! # Algorithm
//! 1. **Select**: UCB1 down the tree (only our actions), tracking state inline
//! 2. **Expand**: add one child (our action)
//! 3. **Rollout**: random actions until depth limit or terminal
//! 4. **Backpropagate**: reward from heuristic/terminal state
//!
//! Budget is measured in `advance()` calls during expansion + rollout.
//! Selection state tracking (tree walk) is not counted — it's overhead, not search.

use fastrand::Rng;

use super::GameState;

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

// ── MCTS Search ────────────────────────────────────────────────

/// Run MCTS search with UCB1 selection + random rollouts.
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

/// Expand one action from the leaf node and run a rollout from the child.
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
    rng: &mut Rng,
    fm_calls: &mut usize,
    budget: usize,
) -> (usize, f32) {
    // Pick a random unexpanded action
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
            rng,
            fm_calls,
            budget,
        )
    };

    (child_idx, reward)
}

/// Run a random rollout from the given state.
///
/// Picks random actions for `player_id` until depth limit, terminal,
/// or budget exhausted. Returns terminal reward or heuristic evaluation.
fn rollout<S: GameState>(
    state: &S,
    player_id: u8,
    max_depth: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
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

        let pick = rng.usize(0..actions.len());
        current = current.advance(&actions[pick], player_id);
        *fm_calls += 1;
    }

    match current.is_terminal() {
        true => current.reward(player_id),
        false => heuristic(&current, player_id),
    }
}

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

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test Doubles ────────────────────────────────────────────

    /// Two-action state: action 0 → reward 1.0, action 1 → reward 0.0.
    /// Always terminal after one action.
    #[derive(Clone)]
    struct TwoActionState {
        chosen: Option<usize>,
    }

    impl GameState for TwoActionState {
        type Action = usize;

        fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
            vec![0, 1]
        }

        fn advance(&self, action: &Self::Action, _player_id: u8) -> Self {
            Self {
                chosen: Some(*action),
            }
        }

        fn is_terminal(&self) -> bool {
            self.chosen.is_some()
        }

        fn reward(&self, _player_id: u8) -> f32 {
            match self.chosen {
                Some(0) => 1.0,
                Some(1) => 0.0,
                None => 0.5,
                _ => 0.0,
            }
        }

        fn tick(&self) -> u32 {
            self.chosen.is_some() as u32
        }
    }

    /// Multi-step state with configurable depth. Actions are 0 and 1.
    /// Action 0 gives +0.01 per tick, action 1 gives -0.01 per tick.
    #[derive(Clone)]
    struct DeepState {
        tick: u32,
        max_tick: u32,
        cumulative: f32,
    }

    impl GameState for DeepState {
        type Action = u8;

        fn available_actions(&self, _player_id: u8) -> Vec<u8> {
            vec![0, 1]
        }

        fn advance(&self, action: &u8, _player_id: u8) -> Self {
            let delta = match *action {
                0 => 0.01,
                _ => -0.01,
            };
            Self {
                tick: self.tick + 1,
                max_tick: self.max_tick,
                cumulative: self.cumulative + delta,
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

    // ── UCB1 Tests ─────────────────────────────────────────────

    #[test]
    fn ucb1_unvisited_is_infinite() {
        assert!(ucb1_score(0.0, 0, 100).is_infinite());
    }

    #[test]
    fn ucb1_visited_is_finite() {
        let score = ucb1_score(5.0, 10, 100);
        assert!(score.is_finite());
        assert!(score > 0.0);
    }

    #[test]
    fn ucb1_more_visits_less_explore() {
        let few = ucb1_score(5.0, 5, 100);
        let many = ucb1_score(5.0, 50, 100);
        assert!(few > many, "few visits should have higher explore bonus");
    }

    #[test]
    fn ucb1_higher_reward_higher_score() {
        let low = ucb1_score(1.0, 10, 100);
        let high = ucb1_score(5.0, 10, 100);
        assert!(high > low, "higher reward should have higher UCB1 score");
    }

    // ── MCTS Search Tests ──────────────────────────────────────

    #[test]
    fn mcts_finds_winning_action() {
        let state = TwoActionState { chosen: None };
        let mut rng = Rng::with_seed(42);

        let action = mcts_search(
            &state,
            0,
            200, // budget
            1,   // rollout depth
            &|_s: &TwoActionState, _pid: u8| 0.5,
            &mut rng,
        );

        assert_eq!(action, 0, "MCTS should find action 0 (reward=1.0)");
    }

    #[test]
    fn mcts_single_action_returns_immediately() {
        #[derive(Clone)]
        struct OneAction;

        impl GameState for OneAction {
            type Action = usize;
            fn available_actions(&self, _pid: u8) -> Vec<usize> {
                vec![42]
            }
            fn advance(&self, _: &usize, _pid: u8) -> Self {
                OneAction
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

        let mut rng = Rng::with_seed(42);
        let action = mcts_search(&OneAction, 0, 100, 5, &|_, _| 0.5, &mut rng);
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

        // Small budget — should complete quickly
        let action = mcts_search(
            &state,
            0,
            50,
            10,
            &|s: &DeepState, _pid: u8| s.cumulative,
            &mut rng,
        );

        // Should return a valid action (0 or 1)
        assert!(action <= 1, "action should be 0 or 1, got {action}");
    }

    #[test]
    fn mcts_prefers_better_heuristic() {
        // Action 0 leads to states with heuristic=1.0
        // Action 1 leads to states with heuristic=0.0
        #[derive(Clone)]
        struct BiasedState {
            last_action: Option<u8>,
        }

        impl GameState for BiasedState {
            type Action = u8;

            fn available_actions(&self, _pid: u8) -> Vec<u8> {
                vec![0, 1]
            }

            fn advance(&self, action: &u8, _pid: u8) -> Self {
                Self {
                    last_action: Some(*action),
                }
            }

            fn is_terminal(&self) -> bool {
                self.last_action.is_some()
            }

            fn reward(&self, _pid: u8) -> f32 {
                match self.last_action {
                    Some(0) => 1.0,
                    Some(1) => 0.0,
                    None => 0.5,
                    _ => 0.0,
                }
            }

            fn tick(&self) -> u32 {
                self.last_action.is_some() as u32
            }
        }

        let state = BiasedState { last_action: None };
        let mut rng = Rng::with_seed(123);

        let action = mcts_search(
            &state,
            0,
            100,
            1,
            &|_s: &BiasedState, _pid: u8| 0.5,
            &mut rng,
        );

        assert_eq!(action, 0, "should prefer action 0 (reward=1.0)");
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
            5,
            &|s: &DeepState, _pid: u8| s.cumulative,
            &mut rng,
        );

        // Action 0 gives +0.01/tick, action 1 gives -0.01/tick
        // With enough budget, MCTS should discover action 0 is better
        assert_eq!(
            action, 0,
            "MCTS should prefer action 0 (positive cumulative)"
        );
    }

    // ── Backpropagation Tests ──────────────────────────────────

    #[test]
    fn backpropagate_updates_chain() {
        let mut nodes = vec![
            MCTSNode::new_root(2),
            MCTSNode::new_child(0, 0, 0),
            MCTSNode::new_child(1, 1, 0),
        ];
        nodes[0].children.push(1);
        nodes[1].children.push(2);

        backpropagate(&mut nodes, 2, 0.8);

        assert_eq!(nodes[2].visits, 1);
        assert!((nodes[2].total_reward - 0.8).abs() < f32::EPSILON);
        assert_eq!(nodes[1].visits, 1);
        assert!((nodes[1].total_reward - 0.8).abs() < f32::EPSILON);
        assert_eq!(nodes[0].visits, 1);
        assert!((nodes[0].total_reward - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn backpropagate_accumulates() {
        let mut nodes = vec![MCTSNode::new_root(1), MCTSNode::new_child(0, 0, 0)];
        nodes[0].children.push(1);

        backpropagate(&mut nodes, 1, 0.5);
        backpropagate(&mut nodes, 1, 0.3);

        assert_eq!(nodes[1].visits, 2);
        assert!((nodes[1].total_reward - 0.8).abs() < f32::EPSILON);
        assert_eq!(nodes[0].visits, 2);
        assert!((nodes[0].total_reward - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn backpropagate_root_only() {
        let mut nodes = vec![MCTSNode::new_root(2)];

        backpropagate(&mut nodes, 0, 1.0);

        assert_eq!(nodes[0].visits, 1);
        assert!((nodes[0].total_reward - 1.0).abs() < f32::EPSILON);
    }
}
