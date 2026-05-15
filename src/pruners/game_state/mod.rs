//! GameState Forward Model — generic trait for what-if game simulation.
//!
//! Distilled from STRATEGA framework (Plan 056, Research 27):
//! - `GameState` trait: forward model API for any game domain
//! - `StateHeuristic` trait: pluggable evaluation for non-terminal states
//! - `ActionSpaceLog`: per-tick branching factor metrics
//!
//! Design: snapshot-based — implementors are lightweight `Clone` structs,
//! NOT wrappers around `bevy_ecs::World` (which isn't `Clone`).

use std::fmt;

// ── GameState Trait ────────────────────────────────────────────

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
    /// Default implementation calls `available_actions().len()`.
    /// Override if you can compute this cheaper than building the full vec.
    fn action_space_size(&self, player_id: u8) -> usize {
        self.available_actions(player_id).len()
    }
}

// ── StateHeuristic Trait ───────────────────────────────────────

/// Pluggable heuristic for evaluating non-terminal states.
///
/// Used by search algorithms (MCTS rollouts, RHEA fitness) when
/// `is_terminal()` is false but we need a numeric evaluation.
///
/// Domain-specific heuristics beat generic search (STRATEGA finding),
/// so each game provides its own implementation.
pub trait StateHeuristic<S: GameState> {
    /// Evaluate state for `player_id`. Higher = better.
    fn evaluate(&self, state: &S, player_id: u8) -> f32;
}

// ── ActionSpaceLog ─────────────────────────────────────────────

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

// ── Submodules ─────────────────────────────────────────────────

#[cfg(feature = "bomber")]
mod bomber_state;

#[cfg(feature = "bomber")]
pub use bomber_state::{BombSnapshot, BomberHeuristic, BomberState, PlayerSnapshot};

mod mcts;

pub use mcts::mcts_search;

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
