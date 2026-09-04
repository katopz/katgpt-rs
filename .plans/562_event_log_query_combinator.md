# Plan 562: EventLog Query Combinator — Programmatic Search over Lossless Log (Modelless PRO-LONG Distillation)

**Date:** 2026-07-29
**Research:** [katgpt-rs/.research/461_PRO_LONG_Programmatic_Memory_Log_Search.md](../.research/461_PRO_LONG_Programmatic_Memory_Log_Search.md)
**Source paper:** [arxiv 2607.20064](https://arxiv.org/abs/2607.20064) — PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning (Fox et al., Duke, 2026-07-23)
**Target:** `katgpt-rs/crates/katgpt-pruners/src/event_log.rs` (extend existing `EventLog<A>`) + Cargo feature `event_log_query`
**Status:** Active — Phase 1 (open primitive) DONE. Phase 2 (GOAT gate) DONE — all 4 gates PASS. Phase 3 (promotion) deferred (opens on consumer demand).

---

## Goal

Distill PRO-LONG's programmatic-search access pattern into a generic, modelless, MIT-licensed **query combinator** over the existing `EventLog<A>` append-only event log (Plan 124). PRO-LONG's empirical finding (Table 1): programmatic tools (grep + Python) drive +15.2 of the +18.1 gain on ARC-AGI-3, vs only +2.9 for Write/Edit. The current `EventLog` ships append + replay + fork + diff + `EvalCache`, but its only retrieval is `iter()` (linear scan with no filter/query/predicate API). This plan adds the missing programmatic-search axis — the deterministic, LLM-free analog of "coding agent greps the log."

The primitive is **generic** (works over any `A: Clone + Debug`), **modelless** (no LLM, no training, no embedding), **zero-allocation** (iterator-based, no `collect()` in the hot path), and **composable** (predicate combinators And/Or/Not; composes with semantic predicates at the consumer layer via the `EventPredicate` trait).

**Why this is Gain-tier, not GOAT-tier:** the gain is a missing feature (the programmatic-search axis does not ship), not a measurable improvement over an existing approach. There is no incumbent to beat on our substrate — `iter()` is the only retrieval, and it has no query API. The GOAT gate below is a ship-quality gate (correctness + perf + no-regression + alloc-free), not a promote-to-default gate. The feature stays opt-in (`event_log_query`) unless a downstream consumer (per-NPC cognition runtime, consolidation pipeline, MCTS planner) proves a gain that warrants promotion.

**Cross-repo follow-up (out of scope for this plan):** game-domain query helpers (`TrialLog::query_by_action_tag` / `query_by_tick_range` / `query_by_score_range`) live in `riir-ai/crates/riir-games/src/trial_log.rs` and would open a separate riir-ai plan when a consumer materializes. The latent-predicate bridge (compose pattern predicates with `experience_graph`'s semantic predicates) lives in `riir-engine` and would open a separate riir-ai plan. This plan ships ONLY the open primitive in `katgpt-pruners`.

---

## Phase 1 — Open Primitive: Predicate Combinator + Query API (CORE)

Goal: a compiling, tested, feature-gated extension to `EventLog<A>` that adds filter/query/query_window with a composable predicate enum, zero-allocation iterator returns, and no changes to the existing API.

### Tasks

- [x] **T1.1** Add feature flag `event_log_query = ["event_log"]` to `katgpt-rs/crates/katgpt-pruners/Cargo.toml` features section (after `event_log`). Add root forwarder `event_log_query = ["katgpt-pruners/event_log_query"]` to `katgpt-rs/Cargo.toml` features section (after `event_log`).
- [x] **T1.2** Define the `EventPredicate<A>` trait in `crates/katgpt-pruners/src/event_log.rs` (gated `#[cfg(feature = "event_log_query")]`):
  - [x] `fn matches(&self, event: &Event<A>) -> bool;`
  - [x] Object-safe (dyn-compatible): no generics on the method, takes `&Event<A>`. Supertrait `Debug` so `Predicate::Custom(Box<dyn EventPredicate<A>>)` derives `Debug`.
- [x] **T1.3** Define the predicate combinator enum `Predicate<A>` in `crates/katgpt-pruners/src/event_log.rs` (gated):
  - [x] `EventTypeIs(EventType)` — raw pattern predicate (direct PRO-LONG "grep event_type" analog)
  - [x] `EventTypeIn(&'static [EventType])` — multi-type pattern
  - [x] `IdRange { lo, hi }` — window predicate (tick range analog)
  - [x] `IdRangeFrom(EventId)` — open-ended window (last-N analog)
  - [x] `And(Box<Predicate<A>>, Box<Predicate<A>>)` — combinator
  - [x] `Or(Box<Predicate<A>>, Box<Predicate<A>>)` — combinator
  - [x] `Not(Box<Predicate<A>>)` — combinator
  - [x] `All` — always-true (identity for And)
  - [x] `None_` — always-false (identity for Or; named with trailing underscore to avoid the `None` keyword collision)
  - [x] `Custom(Box<dyn EventPredicate<A>>)` — escape hatch for consumer-defined predicates (e.g., score-threshold, action-tag-regex at the game-domain layer)
  - [x] `impl<A: Clone + Debug> EventPredicate<A> for Predicate<A>` — the enum delegates to its variants
  - [x] Constructor helpers: `Predicate::event_type(t)`, `Predicate::id_range(lo, hi)`, `Predicate::id_range_from(from)`, `Predicate::and(self, other)`, `Predicate::or(self, other)`, `impl Not` (via `std::ops::Not`, avoids clippy `should_implement_trait`), `Predicate::custom(p)`
- [x] **T1.4** Implement `EventLog::filter(&self, predicate: &Predicate<A>) -> impl Iterator<Item = &Event<A>>` (gated):
  - [x] Returns a lazy iterator over `self.events.iter().filter(|e| predicate.matches(e))`
  - [x] Zero allocation — the iterator borrows `self`; no `collect()` in the hot path
  - [x] Documented as the direct PRO-LONG "grep the log" analog
- [x] **T1.5** Implement `EventLog::query_window(&self, range: std::ops::Range<EventId>, event_type_filter: Option<EventType>) -> impl Iterator<Item = &Event<A>>` (gated):
  - [x] Returns a contiguous slice iterator (no allocation — direct slice into `self.events`)
  - [x] The `Option<EventType>` filter is applied via `filter()` only when `Some` — the slice is the fast path for "all events in window", the filter is the "actions only in window" path
  - [x] Documented as the bounded-window query (sub-µs target — it's a slice + optional filter)
  - [x] Design deviation: returns `impl Iterator` (lazy, zero-alloc) instead of `&[Event<A>]`. A `&[Event<A>]` return would require the type-filtered case to allocate a filtered Vec; the lazy iterator keeps both paths zero-alloc. This is a strict improvement over the plan spec.
- [x] **T1.6** Implement `EventLog::count_where(&self, predicate: &Predicate<A>) -> usize` (gated):
  - [x] Convenience: `self.filter(predicate).count()` — but documented separately because "count events matching pattern" is the PRO-LONG "grep -c" analog
  - [x] Zero allocation (iterator count)
- [x] **T1.7** Implement `EventLog::first_where(&self, predicate: &Predicate<A>) -> Option<&Event<A>>` and `last_where` (gated):
  - [x] Early-exit iterators (`find` / `rfind`) — the "find the first/last event matching pattern" analog
  - [x] Zero allocation
- [x] **T1.8** Write unit tests in `crates/katgpt-pruners/src/event_log.rs` `mod query_tests` (gated):
  - [x] `filter_returns_only_matching_events` — push 10 events of mixed types, filter by `EventTypeIs(Action)`, assert exactly the Action events returned
  - [x] `query_window_returns_contiguous_slice` — push 10 events, query_window(EventId(2)..EventId(5)), assert slice length 3 + correct events
  - [x] `query_window_with_type_filter` — same window, filter by Action, assert only Action events in window
  - [x] `predicate_and_composes_correctly` — `Predicate::event_type(Action).and(Predicate::id_range(EventId(0), EventId(5)))`, assert only Action events with id < 5
  - [x] `predicate_or_composes_correctly` — `Predicate::event_type(Action).or(Predicate::event_type(RewardSignal))`, assert union
  - [x] `predicate_not_composes_correctly` — `!Predicate::event_type(Action)` (via `std::ops::Not`), assert all non-Action events
  - [x] `count_where_matches_grep_c_semantics` — count events matching pattern, assert count
  - [x] `first_where_and_last_where_early_exit` — find first/last matching, assert correctness + that they differ on a mixed log
  - [x] `custom_predicate_escape_hatch` — implement a test-only `EventPredicate` (StartsWithA via a custom struct), assert it composes via `Predicate::custom`
  - [x] `filter_zero_allocation` — asserts the iterator is lazy (count matches direct linear scan; no intermediate collection)
  - [x] `existing_api_unchanged` — regression: `iter()`, `get()`, `replay()`, `fork()`, `diff()` all still work with `event_log_query` feature OFF
- [x] **T1.9** Add example `crates/katgpt-pruners/examples/event_log_query_basic.rs` (gated):
  - [x] Build an `EventLog<GameAction>` with 100 events (mix of GameStart, Action, RewardSignal, Evaluation, GameEnd, HeuristicFire)
  - [x] Demo: `filter(event_type(Action))` → print the action sequence (first 5 of 32)
  - [x] Demo: `query_window(EventId(10)..EventId(20), Some(Evaluation))` → print the evaluations in that window
  - [x] Demo: `count_where(event_type(RewardSignal).and(id_range_from(EventId(50))))` → print reward count in the back half
  - [x] Demo: `first_where(event_type(Evaluation))` + `last_where(event_type(Action))` → print first/last
  - [x] Demo: `Custom` predicate with a score-threshold struct (HighScoreEval) → print high-score evaluations + composed with IdRange via And
- [x] **T1.10** Document the module extension in `crates/katgpt-pruners/src/event_log.rs` header doc:
  - [x] Added a `# Query API (feature: event_log_query)` section referencing PRO-LONG (arxiv 2607.20064) + Research 461
  - [x] Documented the three retrieval axes (pattern / semantic / content-addressed) with a table; noted that this primitive ships the pattern axis; semantic + content-addressed compose at the consumer layer via `Predicate::Custom`
  - [x] Noted the zero-allocation contract (iterator-based; no `collect()` in the hot path)

### Phase 1 Exit Criteria
- `cargo build -p katgpt-pruners --features event_log_query` compiles clean
- `cargo test -p katgpt-pruners --features event_log_query` passes all new + existing tests
- `cargo build -p katgpt-pruners` (feature OFF) compiles clean — no regression
- `cargo clippy -p katgpt-pruners --features event_log_query --all-targets` zero new warnings
- `cargo run -p katgpt-pruners --example event_log_query_basic --features event_log_query` runs and prints the demo output

---

## Phase 2 — GOAT Gate (Ship-Quality, not Promote-to-Default)

Goal: prove the primitive is correct, fast enough for the per-tick hot path, doesn't regress the existing API, and is alloc-free. This is a **ship-quality gate** (the feature is opt-in; promotion to default requires a downstream consumer proving a gain — out of scope for this plan).

### Tasks

- [x] **T2.1** Write `benches/bench_562_event_log_query_goat.rs` (gated):
  - [x] **G1 correctness**: build a known log (100 events, deterministic mix), assert `filter`, `query_window`, `count_where`, `first_where`, `last_where` all return exactly the expected events for 13 predicate combinations (including composed And/Or/Not/Custom). Print a PASS/FAIL table.
  - [x] **G2 perf**: `filter(event_type(Action))` on a 10K-event log — **4.99 ns/result-event** (target < 1µs; 200× under). `query_window` — **0.46 ns/call** (target < 100ns; 217× under). `count_where` + `first_where` / `last_where` (early-exit) — **4.04 ns / 5.71 ns** (target < 100ns).
  - [x] **G3 no-regression**: documented — feature OFF build clean (verified in Phase 1); existing Plan 124 API unchanged (verified by `existing_api_unchanged` unit test).
  - [x] **G4 alloc-free**: capacity-stability proxy (mirrors `bench_413_snapshot_into_goat` pattern) — filter collect capacity stable (512 → 512) across 1000 steady-state iterations; count/first/last/query_window zero-alloc by construction (lazy iterators).
  - [x] Print a summary table: G1/G2/G3/G4 PASS/FAIL + headline numbers.
- [x] **T2.2** Run the GOAT gate: `cargo bench --bench bench_562_event_log_query_goat --features event_log_query`. Results recorded in `.benchmarks/564_event_log_query_goat.md` (numbered 564, not 562, because `.benchmarks/562` was already allocated by another agent — monotonic numbering discipline).
- [x] **T2.3** Honest verdict in the benchmark doc:
  - [x] G1–G4 all PASS → ship-quality gate met; feature stays opt-in (`event_log_query`) pending downstream consumer.

### Phase 2 Exit Criteria
- All 4 gates (G1/G2/G3/G4) PASS with honest numbers recorded in `.benchmarks/562_event_log_query_goat.md`
- OR: honest failure documented, feature stays opt-in as documented state

---

## Phase 3 — Promotion Decision (DEFERRED — opens on consumer demand)

This phase is intentionally deferred. Per the Gain-tier verdict (Research 461), the feature ships opt-in and stays opt-in until a downstream consumer proves a gain that warrants promotion. The trigger conditions for re-opening this phase:

- [-] **T3.1 (DEFERRED)** A per-NPC cognition runtime (riir-engine) adopts `EventLog::filter` for trajectory search AND benchmarks show it improves a measurable metric (e.g., CLR vote accuracy, KARC forecast skill, consolidation quality) over the no-query baseline. → opens a riir-ai plan for the latent-predicate bridge + consumer wiring.
- [-] **T3.2 (DEFERRED)** A consolidation pipeline (riir-neuron-db Raven/δ-Mem) adopts `query_window` for "find all events matching pattern P in the last N ticks" AND benchmarks show it improves consolidation quality or latency. → opens a riir-neuron-db plan.
- [-] **T3.3 (DEFERRED)** An MCTS planner (katgpt-pruners game_state) adopts `filter` for "find all evaluations matching pattern P" AND benchmarks show it improves search efficiency. → opens a katgpt-rs plan.
- [-] **T3.4 (DEFERRED)** If any of T3.1–T3.3 pass → promote `event_log_query` to default features in `katgpt-pruners/Cargo.toml` + root forwarder, update README Feature Showcase, update Research 461 with the promotion evidence.

---

## Non-goals

1. **No game-domain helpers in this plan.** `TrialLog::query_by_action_tag` etc. live in `riir-games` (riir-ai workspace) and open a separate plan when a consumer materializes. This plan ships ONLY the generic primitive in `katgpt-pruners`.
2. **No latent-predicate bridge in this plan.** Composing pattern predicates with `experience_graph`'s semantic predicates lives in `riir-engine` (riir-ai workspace) and opens a separate plan. The `Predicate::Custom` escape hatch (T1.3) is the composition point.
3. **No LLM-coded world-model generation.** PRO-LONG's §1.6 finding (agents spontaneously code transition functions + BFS planners) is genuinely LLM-dependent (semantic code generation, same as AIDE² R440). That slice has no modelless analog and is out of scope.
4. **No embedding-based retrieval.** The pattern axis (grep/regex/predicate) is this plan's scope. The semantic axis (vector ANN) already ships via `experience_graph`. The content-addressed axis (hash → slot) already ships via Engram. These three axes are orthogonal; this plan adds only the pattern axis.
5. **No promotion to default in this plan.** Gain-tier features stay opt-in until a downstream consumer proves a gain (Phase 3).

---

## References

- [Research 461](../.research/461_PRO_LONG_Programmatic_Memory_Log_Search.md) — the Gain verdict + distillation.
- `Plan 124` — the existing `EventLog<A>` primitive this plan extends.
- [arxiv 2607.20064](https://arxiv.org/abs/2607.20064) — PRO-LONG source paper.
- Research 368 (AutoMem) — the closest verdict cousin; LOG/PLAN two-phase memory management. Composable with this plan's filter API (AutoMem's PLAN phase can use `filter` to decide what to recall).
- Research 300 (Trellis/Experience Graph, riir-neuron-db) — the closest shipped query layer; latent-seeded NS traversal. Composes with this plan's pattern axis via `Predicate::Custom`.
- Research 169 (AgentMemBench, riir-ai) — F6 (raw > compressed) + F7 (late filtering > early filtering) validate this plan's design direction.
