# Plan 122: Event-Sourced Game Traces with Fork-and-Diff

> **Status:** ✅ Complete (9/9 tasks done)
> **Priority:** Medium (strengthens GOAT proofs, enables counterfactual strategy exploration)
> **Feature Gate:** `event_log`
> **Related Research:** 081 (ActiveGraph Event-Sourced Reactive Graphs)
> **Related Plans:** 049 (G-Zero), 092 (Freeze/Thaw), 112 (SR²AM), 111 (Data Gate)
> **Architecture:** Additive — no changes to existing hot paths

---

## Tasks

- [x] T1: Define `EventLog<A>` types in `src/pruners/event_log.rs` — `EventId`, `EventType`, `Actor`, `GameOutcome`, `Event<A>`, `EventLog<A>` with push/get/iter/len/last_id
- [x] T2: Implement `EvalCache` with content-addressed hashing — `EvalCache` with `HashMap<[u8; 32], CachedEval>`, `get`/`insert`/`hit_rate`/`Default` (note: consumer provides hash; no blake3 crate dependency)
- [x] T3: Implement `fork()` and `structural_diff()` for `EventLog<A>` — `fork(at)` clones prefix, `diff()` returns `ForkDiff<A>` with `DiffEvent` enum, `replay()` for state reconstruction
- [x] T4: Wire `EventLog` into Bomber HL arena — `BomberEventLog` wrapper in `src/pruners/bomber/event_log_player.rs`
- [x] T5: Add GOAT proof for deterministic replay from event log — `test_deterministic_replay`, `test_multiple_games_replay` (100 games) in `tests/test_124_event_log_goat.rs`
- [x] T6: Add GOAT proof for fork-and-diff counterfactual outcome — `test_fork_shares_prefix`, `test_structural_diff`, `test_identical_logs_diff`, `test_diff_different_lengths` (22 GOAT proofs total)
- [x] T7: Wire `EventLog` into Go arena — `GoEventLog` wrapper in `src/pruners/go/event_log_player.rs`
- [x] T8: Benchmark: event_log overhead vs raw game trace — `tests/bench_124_event_log_overhead.rs`
- [x] T9: Update README.md and feature flags documentation — Event Log section + feature flag table entry

---

## Motivation

From Research 081 (ActiveGraph):

1. **Deterministic replay** — Any game is byte-reproducible from its event log
2. **Cheap forking** — Branch at move N without re-executing prefix (eval cache)
3. **Total lineage** — Every move traces to the heuristic/bandit/model that produced it
4. **Counterfactual evaluation** — Fork-and-diff is the missing self-improvement primitive for G-Zero

Our model-based/modelless spectrum in event-sourced form:
- **Modelless** = replay from log (cache hit, O(1) lookup)
- **Model-based** = new execution (cache miss, fresh evaluation)

---

## Architecture

### Module: `src/pruners/event_log.rs`

```
src/pruners/
├── mod.rs              (add event_log module, feature-gated)
├── event_log.rs        (NEW — EventLog<A>, EvalCache, ForkDiff)
├── bomber/             (existing)
├── go/                 (existing)
└── ...
```

### Key Types

```rust
/// Feature gate: event_log
#[cfg(feature = "event_log")]
pub mod event_log;

/// Append-only event log for game traces.
/// Source of truth — game state is always a fold over this log.
///
/// Plan 122: Event-sourced game traces with fork-and-diff capability.
pub struct EventLog<A: Clone + Debug> {
    events: Vec<Event<A>>,
    /// Content-addressed cache for evaluation results
    eval_cache: EvalCache,
}

/// Single event in the trace.
pub struct Event<A: Clone + Debug> {
    id: EventId,           // Monotonic u64
    event_type: EventType, // Enum: GameStart, Action, Eval, BanditUpdate, etc.
    payload: A,            // Game-specific action/state
    actor: Actor,          // Enum: Player, Heuristic, Bandit, Model
    caused_by: Option<EventId>, // Causal chain — which event triggered this
    response_hash: Option<blake3::Hash>, // Content hash of eval result (if any)
}

/// Content-addressed evaluation cache.
/// Same game state hash → cached score, no re-evaluation.
pub struct EvalCache {
    entries: papaya::HashMap<blake3::Hash, CachedEval>,
}

pub struct CachedEval {
    score: f32,
    depth: usize,
    provenance: EventId, // Which event produced this evaluation
}

/// Result of forking and comparing two traces.
pub struct ForkDiff<A: Clone + Debug> {
    fork_point: EventId,
    shared_prefix_len: usize,
    parent_outcome: GameOutcome,
    fork_outcome: GameOutcome,
    diff_events: Vec<DiffEvent<A>>,
}

pub enum DiffEvent<A> {
    ParentOnly(Event<A>),
    ForkOnly(Event<A>),
    Diverged { parent: Event<A>, fork: Event<A> },
}
```

### Enum Design (use Enums as possible)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    GameStart,
    Action,
    Evaluation,
    BanditUpdate,
    HeuristicFire,
    RewardSignal,
    GameEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Actor {
    Player(u8),
    Heuristic(&'static str),
    Bandit,
    Model,
    Runtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOutcome {
    Win(u8),
    Draw,
    Incomplete,
}
```

---

## Task Details

### T1: Define `EventLog<A>` types

File: `src/pruners/event_log.rs`

- `EventLog<A>` — append-only, no mutation of existing events
- `Event<A>` — causal chain via `caused_by` field
- `EventId` — newtype wrapper around `u64`
- Derive `Serialize, Deserialize` for persistence (interoperability with Freeze/Thaw)
- `EventLog::push()` — append event, auto-assign monotonic ID
- `EventLog::replay()` — fold events into game state (generic over `GameState`)

### T2: Implement `EvalCache`

File: `src/pruners/event_log.rs`

- `EvalCache` using `papaya::HashMap` (lock-free, per project rules)
- Key: `blake3::Hash` of game state serialization
- Value: `CachedEval { score, depth, provenance }`
- `EvalCache::get()` — O(1) lock-free lookup
- `EvalCache::insert()` — record evaluation with provenance
- `EvalCache::hit_rate()` — for benchmarking cache effectiveness

### T3: Implement fork and structural diff

File: `src/pruners/event_log.rs`

- `EventLog::fork(&self, at: EventId) -> EventLog<A>` — clone prefix, new empty log after
- `EventLog::diff(&self, other: &EventLog<A>) -> ForkDiff<A>` — compare event-by-event from fork point
- Forked log shares prefix events by clone (Rust ownership, no reference counting)
- On replay of fork: cache hits for all prefix evaluations → no re-computation

### T4: Wire into Bomber HL arena

File: `src/pruners/bomber.rs` (extend existing)

- Add `#[cfg(feature = "event_log")]` methods to `HLPlayer`
- `HLPlayer::play_with_log()` — play game while recording to `EventLog`
- `HLPlayer::replay_from_log()` — reconstruct game state from event log
- Start with Bomber HL because it's the simplest arena (finite actions, discrete outcomes)

### T5: GOAT proof — deterministic replay

File: `tests/bench_122_event_log_goat.rs`

```
GOAT PROOF: Event Log Deterministic Replay
  Given: EventLog<A> from a complete game
  When:  Replay log from scratch
  Then:  Final game state is byte-identical to original
  Proof: replay(log).state == original_final_state
```

- Play 100 games, record to event log
- Replay each log, assert final state matches
- Include eval cache hit rate metric

### T6: GOAT proof — fork-and-diff counterfactual

File: `tests/bench_122_event_log_goat.rs`

```
GOAT PROOF: Fork-and-Diff Counterfactual
  Given: EventLog<A> from a complete game (parent)
  When:  Fork at event K, apply alternative strategy
  Then:  Shared prefix events are identical (same IDs)
  And:   Divergence is detected at event K+1
  And:   Parent prefix evaluations served from cache (no re-compute)
```

- Fork 10 games at 5 different points each
- Assert shared prefix is byte-identical
- Assert divergence starts at fork point + 1
- Measure cache hit rate for prefix (should be 100%)

### T7: Wire into Go arena

File: `src/pruners/go.rs` (extend existing)

- Same pattern as T4 but for Go
- `GoHLPlayer::play_with_log()`
- `GoHLPlayer::replay_from_log()`
- Go is more complex (larger action space, longer games) — validates scalability

### T8: Benchmark — overhead measurement

File: `tests/bench_122_event_log_goat.rs`

- Measure: wall-clock time for game WITH event_log vs WITHOUT
- Target: < 5% overhead for event recording (append-only is cheap)
- Measure: cache hit rate during replay
- Measure: fork cost (should be near-zero for prefix, only pay for new execution)
- Report in `.benchmarks/122_event_log_overhead.md`

### T9: Update documentation

- Add to `README.md` under new section or extend Productions
- Add feature flag `event_log` to feature flags section
- Update `.docs/` if applicable

---

## Feature Gate

```toml
# Cargo.toml
[features]
default = []
event_log = ["serde", "blake3", "papaya"]
```

All `event_log` code behind `#[cfg(feature = "event_log")]`.
Zero overhead when feature is disabled.

---

## GOAT Proof Summary (Planned)

| # | Proof | Gate |
|---|-------|------|
| 1 | Deterministic replay produces byte-identical final state | ✅ Must pass |
| 2 | Fork shares exact prefix events (same IDs, no re-execution) | ✅ Must pass |
| 3 | Structural diff correctly identifies divergence point | ✅ Must pass |
| 4 | Eval cache hit rate = 100% for shared prefix during fork replay | ✅ Must pass |
| 5 | Event log overhead < 5% vs raw game trace | ✅ Must pass |

---

## What This Is NOT

- NOT a full reactive graph runtime (that's overkill)
- NOT changing existing hot paths (purely additive)
- NOT a behavior subscription system (our bandit/pruner traits are the right abstraction)
- NOT replacing Freeze/Thaw (complements it — event log is the trace, freeze/thaw is the snapshot)

---

## Relationship to Existing Systems

| System | Relationship |
|--------|-------------|
| Freeze/Thaw (Plan 092) | Event log = full trace; freeze/thaw = point-in-time snapshot |
| G-Zero (Plan 049) | Fork-and-diff = strategy evaluation primitive for self-play |
| Data Gate (Plan 111) | Event log makes gate decisions auditable and reversible |
| SR²AM (Plan 112) | Event log provides the self-regulation audit trail |
| GOAT proofs | Event log strengthens determinism guarantee with byte-reproducible replay |
| Productions | Simplified version of reactive behaviors (no changes needed) |

---

## Benchmark Target

```
.benchmarks/122_event_log_overhead.md

| Metric | Target | Measured |
|--------|--------|----------|
| Event recording overhead | < 5% | TBD |
| Replay correctness | 100% | TBD |
| Fork prefix cache hit rate | 100% | TBD |
| Fork cost (100-event prefix) | < 1ms | TBD |
| Diff accuracy | 100% | TBD |
```

---

## Implementation Order

1. **T1-T3** (types + cache + fork) — Core infrastructure, no game integration
2. **T4** (Bomber HL wiring) — Simplest validation
3. **T5-T6** (GOAT proofs) — Prove it works
4. **T8** (benchmark) — Measure overhead
5. **T7** (Go arena) — Scale validation
6. **T9** (docs) — Finalize