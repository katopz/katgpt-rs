//! GameState Forward Model — generic trait for what-if game simulation.
//!
//! Distilled from STRATEGA framework (Plan 056, Research 27):
//! - `GameState` trait: forward model API for any game domain
//! - `StateHeuristic` trait: pluggable evaluation for non-terminal states
//! - `RolloutPolicy` trait: pluggable action selection for MCTS rollouts
//! - `ActionSpaceLog`: per-tick branching factor metrics
//!
//! Design: snapshot-based — implementors are lightweight `Clone` structs,
//! NOT wrappers around `bevy_ecs::World` (which isn't `Clone`).

// ── Re-exported from katgpt-core (Plan 107 Phase 0) ─────────────
// GameState, StateHeuristic, RolloutPolicy, RandomRolloutPolicy, ActionSpaceLog
// consolidated into katgpt-core/src/traits.rs to eliminate duplication with riir-engine.

pub use katgpt_core::traits::{
    ActionSpaceLog, GameState, RandomRolloutPolicy, RolloutPolicy, StateHeuristic,
};

// ── Submodules ─────────────────────────────────────────────────

// `bomber_state` moved back to the main katgpt-rs crate (src/pruners/bomber/
// bomber_state.rs). It's tightly coupled to the bomber module which stays in
// the main crate (depends on inference_router / transformer re-exports).
// The main crate re-exports BomberState via its `crate::pruners::*` shim.

mod mcts;

pub use mcts::mcts_search;
pub use mcts::mcts_search_informed;

#[cfg(feature = "bandit")]
pub use mcts::BanditRolloutPolicy;

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GameState for testing the trait and ActionSpaceLog.
    #[derive(Clone)]
    struct DummyState {
        tick: u32,
        terminal: bool,
    }

    impl GameState for DummyState {
        type Action = u8;

        fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
            match self.terminal {
                true => vec![],
                false => vec![0, 1, 2],
            }
        }

        fn advance(&self, _action: &Self::Action, _player_id: u8) -> Self {
            let new_tick = self.tick + 1;
            Self {
                tick: new_tick,
                terminal: new_tick >= 5,
            }
        }

        fn is_terminal(&self) -> bool {
            self.terminal
        }

        fn reward(&self, player_id: u8) -> f32 {
            match self.terminal {
                true => 1.0,
                false => player_id as f32 * 0.1,
            }
        }

        fn tick(&self) -> u32 {
            self.tick
        }
    }

    #[test]
    fn action_space_log_records_entries() {
        let state = DummyState {
            tick: 0,
            terminal: false,
        };
        let mut log = ActionSpaceLog::new();

        log.record(&state, 0);
        log.record(&state, 1);

        assert_eq!(log.len(), 2);
        assert!((log.avg_action_space() - 3.0).abs() < f32::EPSILON);
        assert_eq!(log.peak_action_space(), 3);
    }

    #[test]
    fn action_space_log_per_player() {
        let state = DummyState {
            tick: 0,
            terminal: false,
        };
        let mut log = ActionSpaceLog::new();

        log.record(&state, 0);
        log.record(&state, 1);

        assert!((log.avg_action_space_for(0) - 3.0).abs() < f32::EPSILON);
        assert!((log.avg_action_space_for(1) - 3.0).abs() < f32::EPSILON);
        assert!((log.avg_action_space_for(99)).abs() < f32::EPSILON);
    }

    #[test]
    fn action_space_log_terminal_state() {
        let state = DummyState {
            tick: 10,
            terminal: true,
        };
        let mut log = ActionSpaceLog::new();

        log.record(&state, 0);

        assert_eq!(log.peak_action_space(), 0);
    }

    #[test]
    fn action_space_log_display() {
        let mut log = ActionSpaceLog::new();
        assert_eq!(format!("{log}"), "ActionSpaceLog(empty)");

        let state = DummyState {
            tick: 0,
            terminal: false,
        };
        log.record(&state, 0);
        let display = format!("{log}");
        assert!(display.contains("entries=1"));
        assert!(display.contains("avg=3.0"));
        assert!(display.contains("peak=3"));
    }

    #[test]
    fn action_space_log_clear() {
        let state = DummyState {
            tick: 0,
            terminal: false,
        };
        let mut log = ActionSpaceLog::new();
        log.record(&state, 0);
        assert!(!log.is_empty());

        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn dummy_state_advance_increments_tick() {
        let state = DummyState {
            tick: 3,
            terminal: false,
        };
        let next = state.advance(&0u8, 0);
        assert_eq!(next.tick(), 4);
    }

    #[test]
    fn dummy_state_becomes_terminal_at_limit() {
        let state = DummyState {
            tick: 4,
            terminal: false,
        };
        let next = state.advance(&0u8, 0);
        assert!(next.is_terminal());
    }

    #[test]
    fn dummy_state_terminal_has_no_actions() {
        let state = DummyState {
            tick: 10,
            terminal: true,
        };
        assert!(state.available_actions(0).is_empty());
        assert_eq!(state.action_space_size(0), 0);
    }
}
