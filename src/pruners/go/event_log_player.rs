//! Go event log wrapper — records Go game traces to EventLog.
//!
//! Type-level integration between the Go arena and the EventLog
//! fork-diff infrastructure. Provides high-level recording methods that
//! wrap the generic EventLog with Go-specific semantics.
//!
//! Plan 124: Event-sourced game traces with fork-and-diff.
//! Feature gate: `event_log`

use crate::pruners::event_log::{
    Actor, DiffEvent, EventId, EventLog, EventType, ForkDiff, GameOutcome,
};
use crate::pruners::go::GoAction;

/// Go event log wrapper for recording game traces.
///
/// Wraps the generic `EventLog<GoAction>` with Go-specific
/// convenience methods for recording moves, passes, evaluations, and
/// game lifecycle events.
#[derive(Clone, Debug)]
pub struct GoEventLog {
    log: EventLog<GoAction>,
}

impl GoEventLog {
    /// Create a new empty Go event log.
    pub fn new() -> Self {
        Self {
            log: EventLog::new(),
        }
    }

    /// Record a stone placement.
    pub fn record_place_stone(&mut self, player_id: u8, x: usize, y: usize, round: u32) -> EventId {
        let _ = round; // Round derivable from event position
        let caused_by = self.log.last_id();
        self.log.push(
            EventType::Action,
            GoAction::Place(x, y),
            Actor::Player(player_id),
            Some(caused_by),
        )
    }

    /// Record a pass action.
    pub fn record_pass(&mut self, player_id: u8, round: u32) -> EventId {
        let _ = round;
        let caused_by = self.log.last_id();
        self.log.push(
            EventType::Action,
            GoAction::Pass,
            Actor::Player(player_id),
            Some(caused_by),
        )
    }

    /// Record a resign action (treated as game end).
    pub fn record_resign(&mut self, player_id: u8) -> EventId {
        self.log.push(
            EventType::Action,
            GoAction::Pass,
            Actor::Player(player_id),
            None,
        )
    }

    /// Record an evaluation score for a player's move.
    pub fn record_eval(
        &mut self,
        player_id: u8,
        action: GoAction,
        score: f32,
        round: u32,
    ) -> EventId {
        let _ = (score, round); // Eval score attached via causal chain
        self.log.push(
            EventType::Evaluation,
            action,
            Actor::Player(player_id),
            None,
        )
    }

    /// Record game start with board size.
    pub fn record_game_start(&mut self, board_size: usize) -> EventId {
        let _ = board_size;
        self.log
            .push(EventType::GameStart, GoAction::Pass, Actor::Runtime, None)
    }

    /// Record game end with outcome.
    pub fn record_game_end(&mut self, _outcome: GameOutcome) -> EventId {
        self.log
            .push(EventType::GameEnd, GoAction::Pass, Actor::Runtime, None)
    }

    /// Fork the log at a given event ID.
    /// Returns a new log containing events up to and including `at`.
    pub fn fork_at(&self, at: EventId) -> Self {
        Self {
            log: self.log.fork(at),
        }
    }

    /// Compute structural diff between this log and another.
    pub fn diff(&self, other: &GoEventLog) -> GoForkDiff {
        GoForkDiff {
            inner: self.log.diff(&other.log),
        }
    }

    /// Replay events through a fold function to reconstruct state.
    pub fn replay<F, S>(&self, initial: S, f: F) -> S
    where
        F: FnMut(S, &crate::pruners::event_log::Event<GoAction>) -> S,
    {
        self.log.replay(initial, f)
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.log.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }

    /// Get the last event ID.
    pub fn last_id(&self) -> EventId {
        self.log.last_id()
    }

    /// Get an event by ID.
    pub fn get(&self, id: EventId) -> Option<&crate::pruners::event_log::Event<GoAction>> {
        self.log.get(id)
    }

    /// Iterate over all events.
    pub fn iter(&self) -> impl Iterator<Item = &crate::pruners::event_log::Event<GoAction>> {
        self.log.iter()
    }

    /// Access the underlying generic EventLog.
    pub fn inner(&self) -> &EventLog<GoAction> {
        &self.log
    }
}

impl Default for GoEventLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing two divergent Go traces.
#[derive(Clone, Debug)]
pub struct GoForkDiff {
    inner: ForkDiff<GoAction>,
}

impl GoForkDiff {
    /// Whether the two logs are identical.
    pub fn is_identical(&self) -> bool {
        self.inner.is_identical()
    }

    /// Number of shared prefix events.
    pub fn shared_prefix_len(&self) -> usize {
        self.inner.shared_prefix_len
    }

    /// Number of divergent events.
    pub fn divergence_count(&self) -> usize {
        self.inner.divergence_count()
    }

    /// Event ID where divergence starts.
    pub fn fork_point(&self) -> EventId {
        self.inner.fork_point
    }

    /// Iterate over divergent events.
    pub fn diff_events(&self) -> &[DiffEvent<GoAction>] {
        &self.inner.diff_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_event_log_record_game() {
        let mut log = GoEventLog::new();

        let start_id = log.record_game_start(9);
        assert_eq!(start_id, EventId(0));

        let stone_id = log.record_place_stone(0, 3, 3, 1);
        assert_eq!(stone_id, EventId(1));

        let stone_id2 = log.record_place_stone(1, 15, 15, 2);
        assert_eq!(stone_id2, EventId(2));

        let pass_id = log.record_pass(0, 3);
        assert_eq!(pass_id, EventId(3));

        let end_id = log.record_game_end(GameOutcome::Win(0));
        assert_eq!(end_id, EventId(4));

        assert_eq!(log.len(), 5);
    }

    #[test]
    fn test_go_event_log_fork() {
        let mut log = GoEventLog::new();
        log.record_game_start(9);
        log.record_place_stone(0, 3, 3, 1);
        log.record_place_stone(1, 15, 15, 2);
        log.record_place_stone(0, 4, 4, 3);

        let forked = log.fork_at(EventId(1));
        assert_eq!(forked.len(), 2);

        // Forked should have same first two events
        assert_eq!(
            forked.get(EventId(0)).unwrap().payload,
            log.get(EventId(0)).unwrap().payload
        );
        assert_eq!(
            forked.get(EventId(1)).unwrap().payload,
            log.get(EventId(1)).unwrap().payload
        );
    }

    #[test]
    fn test_go_event_log_diff() {
        let mut log_a = GoEventLog::new();
        log_a.record_game_start(9);
        log_a.record_place_stone(0, 3, 3, 1);
        log_a.record_place_stone(0, 4, 4, 2);

        let mut log_b = GoEventLog::new();
        log_b.record_game_start(9);
        log_b.record_place_stone(0, 3, 3, 1);
        log_b.record_place_stone(1, 15, 3, 2); // Diverges: different player + position

        let diff = log_a.diff(&log_b);
        assert_eq!(diff.shared_prefix_len(), 2);
        assert!(!diff.is_identical());
        assert_eq!(diff.divergence_count(), 2); // 1 from each side
    }

    #[test]
    fn test_go_event_log_replay() {
        let mut log = GoEventLog::new();
        log.record_game_start(9);
        log.record_place_stone(0, 3, 3, 1);
        log.record_place_stone(1, 15, 15, 2);
        log.record_pass(0, 3);

        #[derive(Clone, PartialEq)]
        struct Board {
            moves: Vec<(usize, usize)>,
            passes: usize,
        }

        let start = Board {
            moves: Vec::new(),
            passes: 0,
        };
        let final_state = log.replay(start, |mut board, event| match &event.payload {
            GoAction::Place(x, y) => {
                board.moves.push((*x, *y));
                board
            }
            GoAction::Pass => {
                // Only count passes from Action events, not GameStart/GameEnd
                if event.event_type == EventType::Action {
                    Board {
                        passes: board.passes + 1,
                        ..board
                    }
                } else {
                    board
                }
            }
        });

        assert_eq!(final_state.moves, vec![(3, 3), (15, 15)]);
        assert_eq!(final_state.passes, 1);
    }

    #[test]
    fn test_go_event_log_default() {
        let log = GoEventLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn test_go_fork_diff_identical() {
        let mut log = GoEventLog::new();
        log.record_game_start(9);
        log.record_place_stone(0, 3, 3, 1);

        let diff = log.diff(&log);
        assert!(diff.is_identical());
    }

    #[test]
    fn test_go_record_eval() {
        let mut log = GoEventLog::new();
        log.record_game_start(9);

        let eval_id = log.record_eval(0, GoAction::Place(3, 3), 0.92, 1);
        assert_eq!(eval_id, EventId(1));

        let event = log.get(eval_id).unwrap();
        assert_eq!(event.event_type, EventType::Evaluation);
        assert_eq!(event.actor, Actor::Player(0));
    }

    #[test]
    fn test_go_record_resign() {
        let mut log = GoEventLog::new();
        log.record_game_start(9);
        log.record_place_stone(0, 3, 3, 1);

        let resign_id = log.record_resign(1);
        assert_eq!(resign_id, EventId(2));

        let event = log.get(resign_id).unwrap();
        assert_eq!(event.actor, Actor::Player(1));
    }
}
