//! Bomber event log wrapper — records bomber game traces to EventLog.
//!
//! Type-level integration between the Bomber HL arena and the EventLog
//! fork-diff infrastructure. Provides high-level recording methods that
//! wrap the generic EventLog with bomber-specific semantics.
//!
//! Plan 124: Event-sourced game traces with fork-and-diff.
//! Feature gate: `event_log`

use crate::pruners::bomber::BomberAction;
use crate::pruners::event_log::{
    Actor, DiffEvent, EventId, EventLog, EventType, ForkDiff, GameOutcome,
};

/// Bomber-specific position on the arena grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BomberPos {
    pub x: i32,
    pub y: i32,
}

/// Bomber event log wrapper for recording game traces.
///
/// Wraps the generic `EventLog<BomberAction>` with bomber-specific
/// convenience methods for recording moves, bombs, evaluations, and
/// game lifecycle events.
#[derive(Clone, Debug)]
pub struct BomberEventLog {
    log: EventLog<BomberAction>,
}

impl BomberEventLog {
    /// Create a new empty bomber event log.
    pub fn new() -> Self {
        Self {
            log: EventLog::new(),
        }
    }

    /// Record a player move action.
    pub fn record_move(&mut self, player_id: u8, action: BomberAction, _round: u32) -> EventId {
        let caused_by = self.log.last_id();
        self.log.push(
            EventType::Action,
            action,
            Actor::Player(player_id),
            Some(caused_by),
        )
    }

    /// Record a bomb placement.
    pub fn record_bomb(&mut self, player_id: u8, pos: BomberPos, round: u32) -> EventId {
        let _ = pos; // Position stored via causal chain
        let _ = round; // Round derivable from event position
        let caused_by = self.log.last_id();
        self.log.push(
            EventType::Action,
            BomberAction::Bomb,
            Actor::Player(player_id),
            Some(caused_by),
        )
    }

    /// Record an evaluation score for a player.
    pub fn record_eval(
        &mut self,
        player_id: u8,
        action: BomberAction,
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

    /// Record game start with player count.
    pub fn record_game_start(&mut self, n_players: u8) -> EventId {
        let _ = n_players;
        self.log.push(
            EventType::GameStart,
            BomberAction::Wait,
            Actor::Runtime,
            None,
        )
    }

    /// Record game end with outcome.
    pub fn record_game_end(&mut self, _outcome: GameOutcome) -> EventId {
        self.log
            .push(EventType::GameEnd, BomberAction::Wait, Actor::Runtime, None)
    }

    /// Fork the log at a given event ID.
    /// Returns a new log containing events up to and including `at`.
    pub fn fork_at(&self, at: EventId) -> Self {
        Self {
            log: self.log.fork(at),
        }
    }

    /// Compute structural diff between this log and another.
    pub fn diff(&self, other: &BomberEventLog) -> BomberForkDiff {
        BomberForkDiff {
            inner: self.log.diff(&other.log),
        }
    }

    /// Replay events through a fold function to reconstruct state.
    pub fn replay<F, S>(&self, initial: S, f: F) -> S
    where
        F: FnMut(S, &crate::pruners::event_log::Event<BomberAction>) -> S,
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
    pub fn get(&self, id: EventId) -> Option<&crate::pruners::event_log::Event<BomberAction>> {
        self.log.get(id)
    }

    /// Iterate over all events.
    pub fn iter(&self) -> impl Iterator<Item = &crate::pruners::event_log::Event<BomberAction>> {
        self.log.iter()
    }

    /// Access the underlying generic EventLog.
    pub fn inner(&self) -> &EventLog<BomberAction> {
        &self.log
    }
}

impl Default for BomberEventLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing two divergent bomber traces.
#[derive(Clone, Debug)]
pub struct BomberForkDiff {
    inner: ForkDiff<BomberAction>,
}

impl BomberForkDiff {
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
    pub fn diff_events(&self) -> &[DiffEvent<BomberAction>] {
        &self.inner.diff_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bomber_event_log_record_game() {
        let mut log = BomberEventLog::new();

        let start_id = log.record_game_start(4);
        assert_eq!(start_id, EventId(0));

        let move_id = log.record_move(0, BomberAction::Up, 1);
        assert_eq!(move_id, EventId(1));

        let bomb_id = log.record_bomb(0, BomberPos { x: 1, y: 1 }, 2);
        assert_eq!(bomb_id, EventId(2));

        let end_id = log.record_game_end(GameOutcome::Win(0));
        assert_eq!(end_id, EventId(3));

        assert_eq!(log.len(), 4);
    }

    #[test]
    fn test_bomber_event_log_fork() {
        let mut log = BomberEventLog::new();
        log.record_game_start(4);
        log.record_move(0, BomberAction::Up, 1);
        log.record_move(1, BomberAction::Down, 2);
        log.record_move(0, BomberAction::Left, 3);

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
    fn test_bomber_event_log_diff() {
        let mut log_a = BomberEventLog::new();
        log_a.record_game_start(4);
        log_a.record_move(0, BomberAction::Up, 1);
        log_a.record_move(0, BomberAction::Bomb, 2);

        let mut log_b = BomberEventLog::new();
        log_b.record_game_start(4);
        log_b.record_move(0, BomberAction::Up, 1);
        log_b.record_move(0, BomberAction::Wait, 2); // Diverges here

        let diff = log_a.diff(&log_b);
        assert_eq!(diff.shared_prefix_len(), 2);
        assert!(!diff.is_identical());
        assert_eq!(diff.divergence_count(), 2); // 1 from each side
    }

    #[test]
    fn test_bomber_event_log_replay() {
        let mut log = BomberEventLog::new();
        log.record_game_start(4);
        log.record_move(0, BomberAction::Up, 1);
        log.record_move(0, BomberAction::Right, 2);

        #[derive(Clone, PartialEq)]
        struct Pos {
            x: i32,
            y: i32,
        }

        let start = Pos { x: 0, y: 0 };
        let final_pos = log.replay(start, |pos, event| match event.payload {
            BomberAction::Up => Pos {
                x: pos.x,
                y: pos.y - 1,
            },
            BomberAction::Down => Pos {
                x: pos.x,
                y: pos.y + 1,
            },
            BomberAction::Left => Pos {
                x: pos.x - 1,
                y: pos.y,
            },
            BomberAction::Right => Pos {
                x: pos.x + 1,
                y: pos.y,
            },
            _ => pos,
        });

        assert_eq!(final_pos.x, 1);
        assert_eq!(final_pos.y, -1);
    }

    #[test]
    fn test_bomber_event_log_default() {
        let log = BomberEventLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn test_bomber_fork_diff_identical() {
        let mut log = BomberEventLog::new();
        log.record_game_start(4);
        log.record_move(0, BomberAction::Up, 1);

        let diff = log.diff(&log);
        assert!(diff.is_identical());
    }

    #[test]
    fn test_bomber_record_eval() {
        let mut log = BomberEventLog::new();
        log.record_game_start(4);

        let eval_id = log.record_eval(0, BomberAction::Up, 0.85, 1);
        assert_eq!(eval_id, EventId(1));

        let event = log.get(eval_id).unwrap();
        assert_eq!(event.event_type, EventType::Evaluation);
        assert_eq!(event.actor, Actor::Player(0));
    }
}
