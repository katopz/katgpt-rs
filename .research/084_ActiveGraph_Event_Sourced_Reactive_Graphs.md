# Research 081: ActiveGraph — Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems

**Paper:** [The Log is the Agent: Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems](https://arxiv.org/abs/2605.21997)
**Author:** Yohei Nakajima (Untapped Capital, BabyAGI creator)
**Date:** 2026-05-22, reviewed 2026-05-24
**Related Research:** 037 (REAP Model-Based/Modelless), 060 (MeMo Memory), 021 (G-Zero Self-Play), 075 (Data Gate), 076 (SR²AM)
**Related Plans:** 049 (G-Zero), 092 (Freeze/Thaw), 094 (MeMo Reflections), 112 (SR²AM Configurator)

---

## Summary

ActiveGraph inverts the conventional agent architecture: instead of bolting logging onto a conversation loop, the **append-only event log IS the agent**. The working graph is a deterministic projection (fold) of that log. Behaviors—functions, classes, LLM-backed routines—react to graph changes and emit new events. No orchestrator threads state between steps; coordination happens entirely through the shared graph.

Three properties emerge from this single design decision:
1. **Deterministic replay** — any run is byte-reproducible from its log (via content-addressed cache of model/tool responses)
2. **Cheap forking** — branch at any event without re-executing the shared prefix (model responses served from cache)
3. **Total lineage** — every artifact traces to the behavior, event, and model call that produced it

**This is a systems/architecture paper. No empirical task-performance benchmarks are reported.**

---

## Core Architecture

### Event Log (Source of Truth)

Every state change is an event: `{id, type, payload, actor, caused_by, timestamp}`. Event types include:
- `pack.loaded` — behavior bundle initialization
- `goal.created` — user objective
- `behavior.started/completed` — behavior lifecycle
- `object.created` — typed object with provenance
- `relation.created` — typed edge between objects
- `llm.requested/responded` — model calls (content-hashed)
- `tool.requested/responded` — tool calls (content-hashed)
- `patch.applied` — object mutation

Graph state is **never mutated directly** — it's always a fold over the event log.

### Reactive Behaviors

A behavior declares:
- **Subscription** — event type + optional predicate + graph-shape pattern (Cypher subset)
- **Body** — fires when subscription matches, receives triggering event + graph view + context handle

Four body forms: plain function, class (configurable), LLM-backed routine, relation-behavior (logic on typed edges).

Coordination is emergent: behavior A's output matches behavior B's subscription. No orchestrator, no workflow script.

### Content-Addressed Response Cache

Key mechanism for deterministic replay:
- Model responses keyed on hash of entire request (system message, user messages, model ID, tool definitions, output schema)
- Tool responses keyed on tool name + deterministic hash of arguments
- On replay: matching hash → serve from cache, no new model call
- Forks share prefix events literally (same event IDs), so prefix model calls are free

### Determinism Contract

Behavior bodies must be deterministic functions of inputs. Not statically enforced — violations surface at replay:
- No `random()`, wall-clock time, or fresh UUIDs directly
- No I/O outside framework primitives
- No mutable global state

**Critical nuance:** LLM-backed behaviors are NOT deterministic at first execution. The contract applies to **replay**, where recorded responses are served from cache.

### Fork + Structural Diff

1. Fork copies parent events up to cutoff → independent log after that
2. Shared prefix served from cache (no re-execution)
3. Structural diff compares object/relation/patch topology between runs
4. Enables counterfactual "what if" evaluation without re-paying for history

---

## Distillation to Our Model-Based/Modelless Architecture

### Mapping: ActiveGraph Concepts ↔ Our Stack

| ActiveGraph | Our Stack | Gap |
|-------------|-----------|-----|
| Append-only event log | Game traces (ad-hoc Vec<Action>) | **No formal event log** |
| Deterministic replay | GOAT proofs (benchmarks) | **No runtime replay** |
| Content-addressed cache | Freeze/Thaw serialization | **No response caching** |
| Fork + structural diff | — | **Not implemented** |
| Total lineage/provenance | — | **Not implemented** |
| Reactive behaviors | Productions system | Simplified version exists |
| Graph projection | — | **No graph layer** |
| Subscriptions (pattern matching) | BanditPruner Q-values | Different abstraction level |

### Where Event-Sourcing Would Add Value

#### Tier 1: Game Arena Traces (HIGH — Directly Strengthens G-Zero)

Our game arenas (Bomber, Go, FFT) produce traces but they're not event-sourced. Event-sourced traces would provide:

- **Deterministic replay** of any game from its event log
- **Fork at move N** to explore counterfactual strategies ("what if I placed stone at D4 instead of C3?")
- **Full lineage** from game outcome back to individual move decisions
- **Content-addressed cache** for heuristic evaluations (same board state → cached score)

This directly strengthens G-Zero self-play:
- Phase 1 (modelless): Event log captures heuristic evaluations, δ signals, bandit updates
- Phase 2 (model-based): Event log captures LoRA forward pass results, DPO preference pairs
- Fork-and-diff becomes a **cheap strategy evaluation primitive** — propose new heuristic, fork at event K, run forward, diff outcomes

**This IS our model-based/modelless spectrum in event-sourced form:**
- Modelless = replay from log (no new computation)
- Model-based = new execution (forward pass, fresh evaluation)

#### Tier 2: GOAT Proof Replay Cache (MEDIUM — Strengthens Determinism Guarantee)

Our GOAT proofs already assert determinism via benchmarks. Content-addressed caching would:
- Record model/evaluator responses during proof execution
- Enable byte-reproducible proof replay without re-running computation
- Provide audit trail: "this GOAT proof result came from these exact computation steps"

But: our proofs are already deterministic (fixed seeds, pure functions). The value is in **auditability**, not correctness.

#### Tier 3: Self-Improving Loop Forking (MEDIUM — Enables §7 Affordance)

The paper's §7 describes fork-and-diff as a self-improvement evaluation primitive. This maps to our:
- Freeze/Thaw (Plan 092) — serializes bandit state
- Self-improving loop (Plan 048) — proposes and evaluates changes

Event-sourced freeze/thaw would enable:
- Fork a frozen bandit at any training step
- Apply heuristic change in fork
- Run fork forward, diff outcomes against parent
- **No re-execution of shared training history**

This is conceptually aligned with our Data Gate (Plan 111) — the gate decides which data enters training. Event-sourcing would make that decision auditable and reversible.

#### Tier 4: Full Reactive Graph (LOW — Over-Engineering)

Implementing the full ActiveGraph runtime (behaviors, subscriptions, Cypher patterns) would be overkill for our current stack. Our productions system already provides reactive rule firing in a simpler form.

---

## Concrete Extraction Ideas

### Idea 1: `EventLog<T>` for Game Traces

```rust
/// Append-only event log for game traces.
/// Each event carries causal chain for lineage.
pub struct EventLog<A: Action> {
    events: Vec<Event<A>>,
}

pub struct Event<A: Action> {
    id: u64,
    event_type: EventType,
    payload: A,
    actor: Actor,
    caused_by: Option<u64>, // parent event
    timestamp: u64,
    /// Content hash of any evaluation result
    response_hash: Option<blake3::Hash>,
}
```

- Fork = clone events[0..k], new independent log after
- Replay = fold events into game state
- Diff = compare object/relation topology between two logs

### Idea 2: Content-Addressed Evaluation Cache

```rust
/// Cache evaluation results by content hash of input.
/// Same board state (same hash) → cached score, no re-evaluation.
pub struct EvalCache {
    entries: papaya::HashMap<blake3::Hash, CachedEval>,
}

pub struct CachedEval {
    score: f32,
    depth: usize,
    provenance: EventId, // which event produced this
}
```

- During replay/fork: matching hash → serve cached score
- Strengthens GOAT proof determinism guarantee
- Natural fit with our blake3 requirement

### Idea 3: Fork-and-Diff Strategy Explorer

```rust
/// Fork a game trace at event k, apply alternative strategy.
/// Shared prefix is replayed from cache (no re-evaluation).
pub fn fork_and_diff<A: Action>(
    parent: &EventLog<A>,
    fork_at: u64,
    alt_strategy: impl Fn(&GameState, &Event<A>) -> A,
) -> ForkDiff<A> {
    // 1. Clone prefix
    // 2. Apply alt_strategy from fork point
    // 3. Compare outcomes
    // 4. Return structural diff
}
```

This is the **self-improvement primitive** from paper §7 applied to our game arenas.

---

## Model-Based vs Modelless Mapping (Extended)

| Dimension | Modelless (Replay) | Model-Based (New Execution) |
|-----------|-------------------|---------------------------|
| **Source** | Event log (recorded) | Forward pass (fresh) |
| **Cost** | O(1) cache lookup | O(n) model compute |
| **Determinism** | Guaranteed (recorded) | Non-deterministic |
| **Use case** | Audit, replay, shared prefix | Exploration, evaluation |
| **Our analog** | Freeze/Thaw (frozen bandit) | G-Zero Phase 2 (LoRA update) |
| **ActiveGraph** | Cache hit on replay | Cache miss on new execution |

The spectrum is the same as REAP (Research 037):
- REAP `frequency` = modelless routing signal = event log replay
- REAP `reap` = model-based saliency = fresh forward pass
- ActiveGraph cache hit = modelless = deterministic
- ActiveGraph cache miss = model-based = new LLM call

---

## What We Should NOT Take

1. **Full reactive graph runtime** — Overkill. Our productions system is simpler and sufficient.
2. **Cypher-pattern subscriptions** — Our bandit/pruner trait stack is already the right abstraction for our token-level decisions.
3. **Blackboard architecture revival** — Interesting theoretically, but our trait composition is more Rust-idiomatic.
4. **Python runtime** — Paper's reference implementation is Python. Our Rust stack needs a leaner design.

---

## Verdict

**CONCEPTUAL ALIGNMENT — Event-sourcing strengthens existing abstractions. New `event_log` feature gate for game arena fork-and-diff.**

### What the paper validates:
- Our Freeze/Thaw pipeline is conceptually correct (serialized state = event snapshot)
- Our GOAT proof determinism is the right goal (paper makes it a first-class property)
- Our model-based/modelless spectrum is the same pattern, applied at different granularity

### What we should extract:
1. **`EventLog<A>` trait** — Append-only game trace with causal chain (feature-gated)
2. **Content-addressed eval cache** — blake3-hashed evaluation results for deterministic replay
3. **Fork-and-diff** — Strategy exploration primitive for G-Zero self-play

### What we should NOT extract:
- Full reactive graph runtime
- Behavior subscription system
- Cypher pattern matching

### Risk assessment:
- **Low risk** — Event-sourcing is additive, doesn't change existing hot paths
- **Feature gate** `event_log` keeps it opt-in, no perf regression for non-users
- **No empirical gain claimed** — Paper explicitly says "we do not report that ActiveGraph improves task accuracy"
- Value is in **auditability, replay, and counterfactual evaluation** — hard to benchmark but useful for GOAT proofs

### Recommended plan scope:
- Phase 1: `EventLog<A>` + `EvalCache` (feature-gated, test-only initially)
- Phase 2: Fork-and-diff for one arena (Bomber HL, simplest)
- Phase 3: GOAT proof integration (byte-reproducible proof replay)

---

## References

- Paper: https://arxiv.org/abs/2605.21997
- Code: https://github.com/yoheinakajima/activegraph (Apache-2.0)
- BabyAGI lineage: https://github.com/yoheinakajima/babyagi
- Related: MemGPT/Letta, Zep/Graphiti, Mem0, Hindsight (memory-as-substrate)
- Our related: Research 037 (REAP model-based/modelless), 021 (G-Zero), 060 (MeMo), 075 (Data Gate)