//! Event-sourced game traces with fork-and-diff capability.
//!
//! Provides append-only event logs that serve as the source of truth for game state.
//! State is always a fold over the log, enabling:
//! - Deterministic replay from any event log
//! - Cheap forking at any event boundary
//! - Structural diff between divergent traces
//! - Content-addressed evaluation caching
//!
//! # Architecture
//!
//! ```text
//! EventLog<A>       — Append-only event sequence
//! ├── EvalCache     — Content-addressed evaluation results (blake3)
//! └── ForkDiff<A>   — Structural comparison of divergent traces
//! ```
//!
//! Plan 124: Event-sourced game traces with fork-and-diff.
//! Feature gate: `event_log`

use std::collections::HashMap;
use std::fmt::Debug;

// ── Types ───────────────────────────────────────────────────────

/// Monotonic event identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u64);

impl EventId {
    /// First event ID.
    pub const ZERO: Self = Self(0);
}

/// Type of event in the trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventType {
    /// Game started.
    GameStart,
    /// Player action taken.
    Action,
    /// Evaluation/heuristic computed.
    Evaluation,
    /// Bandit/arm update.
    BanditUpdate,
    /// Heuristic fired.
    HeuristicFire,
    /// Reward signal emitted.
    RewardSignal,
    /// Game ended.
    GameEnd,
}

/// Who produced this event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actor {
    /// A player action (player ID).
    Player(u8),
    /// A named heuristic.
    Heuristic(&'static str),
    /// Bandit/RL system.
    Bandit,
    /// External model.
    Model,
    /// Runtime/infrastructure.
    Runtime,
}

/// Outcome of a completed game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameOutcome {
    /// Player won (player ID).
    Win(u8),
    /// Game ended in draw.
    Draw,
    /// Game didn't complete.
    Incomplete,
}

/// Single event in the trace.
#[derive(Clone, Debug)]
pub struct Event<A: Clone + Debug> {
    /// Monotonic event ID.
    pub id: EventId,
    /// Type of event.
    pub event_type: EventType,
    /// Game-specific payload.
    pub payload: A,
    /// Who produced this event.
    pub actor: Actor,
    /// Which event caused this one (causal chain).
    pub caused_by: Option<EventId>,
}

/// Append-only event log for game traces.
/// Source of truth — game state is always a fold over this log.
#[derive(Clone, Debug)]
pub struct EventLog<A: Clone + Debug> {
    /// All events in order.
    events: Vec<Event<A>>,
}

impl<A: Clone + Debug> EventLog<A> {
    /// Create a new empty event log.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Append an event, auto-assigning the next monotonic ID.
    pub fn push(
        &mut self,
        event_type: EventType,
        payload: A,
        actor: Actor,
        caused_by: Option<EventId>,
    ) -> EventId {
        let id = EventId(self.events.len() as u64);
        self.events.push(Event {
            id,
            event_type,
            payload,
            actor,
            caused_by,
        });
        id
    }

    /// Number of events in the log.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get event by ID.
    pub fn get(&self, id: EventId) -> Option<&Event<A>> {
        self.events.get(id.0 as usize)
    }

    /// Iterate over all events.
    pub fn iter(&self) -> impl Iterator<Item = &Event<A>> {
        self.events.iter()
    }

    /// Last event ID (or ZERO if empty).
    pub fn last_id(&self) -> EventId {
        self.events.last().map(|e| e.id).unwrap_or(EventId::ZERO)
    }

    /// Fork the log at a given event: clone prefix up to (including) `at`.
    /// Returns a new log that shares the prefix events.
    pub fn fork(&self, at: EventId) -> Self {
        let prefix_len = (at.0 as usize + 1).min(self.events.len());
        Self {
            events: self.events[..prefix_len].to_vec(),
        }
    }

    /// Compute structural diff between this log and another.
    /// Returns divergence information starting from the first different event.
    pub fn diff(&self, other: &EventLog<A>) -> ForkDiff<A>
    where
        A: PartialEq,
    {
        let shared = self
            .events
            .iter()
            .zip(other.events.iter())
            .take_while(|(a, b)| a.payload == b.payload && a.event_type == b.event_type)
            .count();

        let fork_point = if shared < self.events.len() {
            self.events[shared].id
        } else {
            EventId(shared as u64)
        };

        let mut diff_events = Vec::new();

        // Events only in self after fork
        for event in &self.events[shared..] {
            diff_events.push(DiffEvent::ParentOnly(event.clone()));
        }

        // Events only in other after fork
        for event in &other.events[shared..] {
            diff_events.push(DiffEvent::ForkOnly(event.clone()));
        }

        ForkDiff {
            fork_point,
            shared_prefix_len: shared,
            diff_events,
        }
    }

    /// Replay events through a fold function to reconstruct state.
    pub fn replay<F, S>(&self, initial: S, f: F) -> S
    where
        F: FnMut(S, &Event<A>) -> S,
    {
        self.events.iter().fold(initial, f)
    }
}

impl<A: Clone + Debug> Default for EventLog<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing two divergent traces.
#[derive(Clone, Debug)]
pub struct ForkDiff<A: Clone + Debug> {
    /// Event ID where divergence starts.
    pub fork_point: EventId,
    /// Number of shared prefix events.
    pub shared_prefix_len: usize,
    /// Divergent events from both logs.
    pub diff_events: Vec<DiffEvent<A>>,
}

impl<A: Clone + Debug> ForkDiff<A> {
    /// Whether the two logs are identical.
    pub fn is_identical(&self) -> bool {
        self.diff_events.is_empty()
    }

    /// Number of divergent events.
    pub fn divergence_count(&self) -> usize {
        self.diff_events.len()
    }
}

/// A single event in a diff between two traces.
#[derive(Clone, Debug)]
pub enum DiffEvent<A: Clone + Debug> {
    /// Event exists only in the parent trace.
    ParentOnly(Event<A>),
    /// Event exists only in the forked trace.
    ForkOnly(Event<A>),
}

/// Content-addressed evaluation cache.
/// Same game state hash → cached score, no re-evaluation.
pub struct EvalCache {
    entries: HashMap<[u8; 32], CachedEval>,
}

/// A cached evaluation result.
#[derive(Clone, Debug)]
pub struct CachedEval {
    /// Cached score.
    pub score: f32,
    /// Depth of evaluation.
    pub depth: usize,
    /// Which event produced this evaluation.
    pub provenance: EventId,
}

impl EvalCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Look up a cached evaluation by state hash.
    pub fn get(&self, hash: &[u8; 32]) -> Option<&CachedEval> {
        self.entries.get(hash)
    }

    /// Insert a cached evaluation.
    pub fn insert(&mut self, hash: [u8; 32], score: f32, depth: usize, provenance: EventId) {
        self.entries.insert(
            hash,
            CachedEval {
                score,
                depth,
                provenance,
            },
        );
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit rate: hits / (hits + misses).
    pub fn hit_rate(&self, hits: usize, total: usize) -> f32 {
        match total {
            0 => 0.0,
            _ => hits as f32 / total as f32,
        }
    }
}

impl Default for EvalCache {
    fn default() -> Self {
        Self::new()
    }
}
