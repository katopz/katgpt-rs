//! MCTS composition layer — `BanditRolloutPolicy` only.
//!
//! Substrate extraction (Plan 008 Step 6, 2026-06-28): the pure game-agnostic
//! MCTS algorithm (`MCTSNode`, `mcts_search`, `mcts_search_informed`, UCB1
//! helpers, all internal helpers + their tests) moved to
//! [`katgpt_core::mcts`]. This file is a thin re-export shim plus the
//! composition layer that needs a root-only type:
//! - [`BanditRolloutPolicy`] — depends on [`BanditStats`] from
//!   [`crate::pruners::bandit`] (root-only). Stays here. Behind `bandit`
//!   feature so it compiles out when the bandit machinery isn't needed.
//!
//! All existing call sites (`crate::pruners::game_state::mcts_search`,
//! `crate::pruners::game_state::mcts_search_informed`,
//! `crate::pruners::game_state::BanditRolloutPolicy`) resolve unchanged via
//! the re-exports.

use std::marker::PhantomData;

use fastrand::Rng;

use katgpt_core::traits::{GameState, RolloutPolicy};

#[cfg(feature = "bandit")]
use crate::pruners::bandit::BanditStats;

// ── Substrate re-exports ────────────────────────────────────────────────
//
// `mcts_search` and `mcts_search_informed` are the public algorithm entry
// points. Moved verbatim to katgpt-core; resolved through this shim so the
// existing `crate::pruners::game_state::mcts::*` import paths keep working.
pub use katgpt_core::mcts::{mcts_search, mcts_search_informed};

// ── Bandit Rollout Policy (composition) ─────────────────────────────────

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
///
/// [`RandomRolloutPolicy`]: katgpt_core::traits::RandomRolloutPolicy
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

// ── Tests ──────────────────────────────────────────────────────────────
//
// All substrate tests moved to katgpt-core/src/mcts.rs (14 tests, all green).
// Below are the composition-only tests that need `BanditStats` (root-only).

#[cfg(test)]
#[cfg(feature = "bandit")]
mod tests {
    use super::*;
    use katgpt_core::traits::GameState;

    /// Minimal 2-action game used by bandit tests.
    #[derive(Clone)]
    struct TwoActionState {
        acted: bool,
        chose_win: bool,
    }

    impl GameState for TwoActionState {
        type Action = bool;

        fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
            match self.acted {
                true => vec![],
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

    #[test]
    fn mcts_informed_with_bandit_finds_winning_action() {
        use katgpt_core::traits::StateHeuristic;

        /// Local closure heuristic adapter (mirror of the private one in core tests).
        struct FnHeuristic<F>(F);
        impl<S: GameState, F: Fn(&S, u8) -> f32> StateHeuristic<S> for FnHeuristic<F> {
            fn evaluate(&self, state: &S, player_id: u8) -> f32 {
                (self.0)(state, player_id)
            }
        }

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

    #[test]
    #[should_panic(expected = "epsilon must be in [0.0, 1.0]")]
    fn bandit_rollout_rejects_invalid_epsilon() {
        let stats = BanditStats::new(2);
        let action_to_index = |action: &bool| if *action { 1 } else { 0 };
        let _policy: BanditRolloutPolicy<'_, TwoActionState, _> =
            BanditRolloutPolicy::new(&stats, 1.5, action_to_index);
    }
}
