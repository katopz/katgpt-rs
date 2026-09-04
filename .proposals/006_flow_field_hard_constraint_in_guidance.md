# Proposal 006 — Flow Field as Hard Constraint in Guidance (Not Soft Cost in PIBT)

Status: **REJECTED** (2026-07-18, Issue 182 Phase 4 GOAT FAIL — see "Verdict" addendum at bottom)
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: Plan 440 Issue 149/150 (existing `GridFlowField` soft-cost seam) × Plan 453 (one-step LaCAM escalation) × Issue 546 diagnostic (riir-ai, P95 deadlock-chain = 8)
Related: `Issue 182` (the implementation plan — CLOSED with REVERT verdict; removed per noise-reduction rule, measurement record preserved in [Benchmark 440 §Follow-up attempt](../.benchmarks/440_lllg_paper_repro_goat.md#tldr) below), Issue 546 (riir-ai, removed per noise-reduction rule) (DEFERRED — the diagnostic that motivates this), [Benchmark 440](../.benchmarks/440_lllg_paper_repro_goat.md) (3/4 PASS), [Benchmark 453](../.benchmarks/453_lacam_escalation_goat.md) (one-step shipped)

## TL;DR

The existing `GridFlowField` (Issue 149 + Issue 150) assigns one-way directions to
corridor cells, then penalizes counter-flow moves via a `flow_mismatch` cost term
**at position 2 of PIBT's lexicographic tuple** `⟨guidance_mismatch, flow_mismatch, goal_dist, hindrance, ε⟩`.
On ht_chantry the flow field has **near-zero effect on throughput** (Benchmark 440 §Issue 150:
throughput change −0.6% to +1% across 4 maps). Three independent root causes:

1. **A\* guidance does not consult the flow field.** The A\* search
   (`SpaceTimeGuidance::astar_for_agent`, `local_guidance.rs:459-555`) expands neighbors
   using cost `g + 1 + α·collision_count` + heuristic `bfs_dist`. The flow field is never
   queried during path planning. So agents are *routed* through corridors in the wrong
   direction before PIBT even runs.
2. **`flow_mismatch` is position 2 in the cost tuple, behind `guidance_mismatch`.** When two
   agents meet head-on in a corridor, both have A\*-preferred forward moves
   (`guidance_mismatch = 0`), so `flow_mismatch` is never reached as a tiebreaker. Both choose
   forward → deadlock.
3. **All corridors assigned `sign = +1`.** Every corridor in the map becomes one-way in the
   positive direction. Any agent whose goal is in the negative direction has no
   flow-legal corridor to use. This is structurally impossible to satisfy on any real map.

This proposal: lift the flow field from a **PIBT-level soft cost** to a **guidance-level hard
constraint**. Concretely:

- **(A) Bi-directional corridor pairing.** In a 2-wide corridor, the two cells form a
  two-lane highway: cell A is `sign = +1`, cell B is `sign = −1`. Agents going positive use
  lane A; agents going negative use lane B. 1-wide corridors alternate by segment parity
  (segment N is `+`, N+1 is `−`) — these remain one-way but form directionally-balanced
  corridors over the map. The "all corridors `+1`" rule is replaced.
- **(B) Flow-respecting A\*.** The A\* guidance search refuses to expand a neighbor when
  entering that neighbor would violate its flow direction (a hard pruner, not a soft cost).
  Agents are *routed around* opposing-flow corridors from the start. This closes root cause #1.
- **(C) Demote `flow_mismatch` from the cost tuple.** Once A\* respects flow, the
  `flow_mismatch` cost term becomes redundant — PIBT's preferred next cell already complies.
  We keep the term at position 4 (after `hindrance`) as a defensive tiebreak for the rare
  case where PIBT must pick a non-preferred cell (constraint-tree forced assignment in
  LaCAM escalation). This closes root cause #2.

The combination closes the three root causes simultaneously. The expected outcome is
**ht_chantry G1 throughput ≥ 0.30** (the deferred Issue 546 target) with **no regression on
the 3 currently-PASS maps** (open maps have zero corridors, so (A)/(B)/(C) are all no-ops
there — paper-faithful ordering preserved).

**This is a `katgpt-rs` design.** It is modelless, closed-form, deterministic, and zero-cost
on open maps. It does not depend on training, gradient descent, or any riir-train work.

## The problem this solves

### The diagnostic that motivates this (Issue 546, riir-ai, 2026-07-18)

The previous session wrote
[`examples/ht_chantry_deadlock_chain_diagnostic.rs`](../crates/katgpt-core/examples/ht_chantry_deadlock_chain_diagnostic.rs)
(commit `2a8c378d`) and produced a **decisive verdict**:

| Metric | Value |
|---|---|
| Throughput (this seed) | 5.098 completions/step (ratio ~0.30) |
| Fast-path ticks (zero stuck) | 0/500 (0.0%) — ht_chantry is systemically congested |
| **P95 max-cluster-size** | **8** |
| P99 max-cluster-size | 9 |
| Max observed | 11 |
| Depth-2 coverage | 12.8% of stuck ticks |
| Depth-3 coverage | 36.4% of stuck ticks |

The "max-cluster-size" is the size of the largest weakly-connected component in the per-tick
blocking graph (edge A~B if B's cell is a passable neighbor of A's). It is a direct proxy
for the depth-K a multi-step LaCAM constraint tree would need to resolve that tick's
deadlocks. **On ht_chantry, P95 = 8** — multi-step LaCAM at the paper's suggested depths
(2-3) covers <40% of stuck ticks. Closing 95% requires depth ≥ 8, which is computationally
intractable.

Issue 546 is **DEFERRED** with the recommendation that "the fix must come from a different
layer: flow-field / guidance layer." This proposal is that fix.

### Why the current `GridFlowField` has near-zero effect (root cause analysis)

I traced the three root causes listed in the TL;DR through the code:

**Root cause #1 — A\* does not consult flow.** Verified in
`crates/katgpt-core/src/multi_agent_path/local_guidance.rs:497-555`. The neighbor expansion
loop:

```rust
for neighbor in self.neighbors_of(&pos) {
    let new_depth = depth + 1;
    let key = (neighbor.clone(), new_depth);
    if scratch.closed.contains(&key) { continue; }
    let h = bfs.get(&neighbor).copied().unwrap_or(f32::MAX);
    if h == f32::MAX { continue; }
    let chi = self.collision_count(&neighbor, depth as usize);
    let tentative_g = g + 1.0 + alpha * chi as f32;
    let known_g = scratch.g_score.get(&key).copied().unwrap_or(f32::MAX);
    if tentative_g < known_g { /* push to open */ }
}
```

There is **no `flow.mismatch(&pos, &neighbor)` term**, no pruner, no cost weight. The A\*
path is computed as if the flow field does not exist. So when an agent's goal is on the
other side of a `sign = +1` corridor and the agent must traverse it in the `−` direction to
reach the goal, A\* happily routes through it.

**Root cause #2 — `flow_mismatch` is at position 2 in the cost tuple.** Verified in
`pibt.rs:163-185`:

```rust
pub(super) fn lexicographic_cmp(&self, other: &Self) -> Ordering {
    self.guidance_mismatch
        .cmp(&other.guidance_mismatch)
        .then_with(|| self.flow_mismatch.cmp(&other.flow_mismatch))
        .then_with(|| /* goal_dist */)
        .then_with(|| /* hindrance */)
        .then_with(|| /* epsilon */)
}
```

When agent A and agent B meet head-on in a corridor:
- A's A\* says "forward" (guidance is correct directionally), `guidance_mismatch(forward) = 0`
- A's flow field says "don't go forward" (corridor sign is +, agent wants −), `flow_mismatch(forward) = 1`
- A's alternative is "wait": `guidance_mismatch(wait) = 1` (not preferred), `flow_mismatch(wait) = 0`

The forward candidate wins because `guidance_mismatch = 0 < 1`. The flow_mismatch tiebreak
is never consulted. **Both agents in a head-on corridor choose forward → deadlock.**

Root cause #2 was suspected in the Benchmark 440 §"cost-term tiebreak position is too weak"
note. This proposal confirms and fixes it.

**Root cause #3 — All corridors are `sign = +1`.** Verified in `flow.rs:248-260` and
`flow.rs:295-307`: every corridor cell is constructed with `sign: 1`. This means:

- An agent at corridor position `x=5` whose goal is at `x=0` must travel in the `−`
  direction. Every corridor cell along the way has `sign = +1`. Every step is a flow
  violation.
- There is no alternative corridor in the `−` direction. The agent has no flow-legal path.

This is structurally impossible to satisfy. Even if (B) made A\* respect flow perfectly,
many agents would have no reachable goal. Root cause #3 must be fixed first — bi-directional
flow assignment is a prerequisite for (B) to be feasible.

## Design

### (A) Bi-directional corridor pairing

#### 2-wide corridors: two-lane highways

For a 2-wide horizontal corridor (cells `(x, y)` and `(x, y+1)` flanked by walls at
`(x, y-1)` and `(x, y+2)`):

- `(x, y)` gets `sign = +1` (positive lane)
- `(x, y+1)` gets `sign = −1` (negative lane)

Agents going +x use the top lane; agents going −x use the bottom lane. They pass each other
without deadlock. Symmetric for 2-wide vertical corridors.

**Coverage on real maps** (from Benchmark 440 §Issue 150):

| Map | 2-wide cells | Directionally usable (post-fix) |
|---|---|---|
| empty-48-48 | 0 | n/a (open map — no effect) |
| random-64-64-10 | 182 | 182 (was 0) |
| warehouse | 4920 | 4920 (was 0) |
| ht_chantry | 102 | 102 (was 0) |

Today: 0 directionally-usable cells because all are `+1`. After (A): all 2-wide cells become
directionally usable.

#### 1-wide corridors: segment parity

A 1-wide corridor cannot form a two-lane highway (only one cell wide). The compromise:
**segment-alternating direction**. Walk each 1-wide corridor as a maximal chain between two
junctions. Number the corridors in BFS order from the map centroid. Even-indexed corridors
get `sign = +1`, odd-indexed get `sign = −1`. This creates a coarse directional balance:
agents going `+` use even-indexed corridors, agents going `−` use odd-indexed ones, taking
slightly longer paths but avoiding head-on deadlocks.

**Risk**: on ht_chantry (only 8 1-wide cells per Benchmark 440), this affects <0.2% of the
map. The 2-wide fix dominates.

**Fallback**: if a 1-wide corridor segment is the *only* path between two regions (a true
topological bottleneck detected by articulation-point analysis), assign it `sign = 0` (dual:
both directions allowed, accept the deadlock risk). This prevents creating unreachable goals.

### (B) Flow-respecting A\*

In `SpaceTimeGuidance::astar_for_agent` and `astar_for_agent_flat`, add a hard pruner in the
neighbor expansion loop:

```rust
for neighbor in self.neighbors_of(&pos) {
    // ... existing checks ...

    // NEW: flow-respecting pruner.
    if self.flow_field.mismatch(&pos, &neighbor) == 1 {
        continue; // Hard skip — this neighbor's flow direction forbids entry.
    }

    // ... existing cost + push ...
}
```

This is a **1-line check**. The flow field already exists and is passed to PIBT; we just
also pass it to guidance. The signature change:

- `SpaceTimeGuidance::new(...)` → `SpaceTimeGuidance::new(...).with_flow_field(flow)` (mirror
  the existing `LifelongLaCam::with_flow_field` API for symmetry)
- `LifelongLaCam::tick` already has the flow field; pass it through to guidance's
  `compute_guidance` via the existing `set_warm_start`-style setter (a new
  `set_flow_field` method on the `LocalGuidanceSource` trait, default no-op)

**Why hard pruner, not soft cost?** A soft cost (e.g., `tentative_g += β · mismatch`) would
still route through the corridor when the alternative is too long — exactly the failure mode
we're trying to prevent. A hard pruner guarantees flow-consistent paths exist or the search
fails gracefully (best-effort path). The graceful-failure path is already implemented: the
expansion cap returns the closest-to-goal partial path.

**Reachability guarantee.** (A) ensures bi-directional corridors exist on every 2-wide
passage and every 1-wide chain has a `sign = 0` fallback at true bottlenecks. So an
agent always has a flow-legal path to any goal. Worst case: longer path (more detours).

### (C) Demote `flow_mismatch` in the cost tuple

Once (B) lands, A\* never produces a `Φ[i][0]` that violates flow. So PIBT's preferred move
always has `flow_mismatch = 0`. The cost term is still useful in two narrow cases:

1. **LaCAM constraint-tree forced assignment.** When the constraint tree forces an agent to a
   non-preferred cell (Plan 453's escalation), that cell may violate flow. Keep
   `flow_mismatch` as a tiebreak to prefer the flow-consistent alternative.
2. **Agent density pressure.** When PIBT must defer (an occupied preferred cell), the
   fallback candidates may include flow-violating moves. `flow_mismatch` orders them.

Move `flow_mismatch` from position 2 to position 4 (after `hindrance`):

```rust
// OLD: guidance → flow → goal_dist → hindrance → ε
// NEW: guidance → goal_dist → hindrance → flow → ε
pub(super) fn lexicographic_cmp(&self, other: &Self) -> Ordering {
    self.guidance_mismatch.cmp(&other.guidance_mismatch)
        .then_with(|| /* goal_dist */)
        .then_with(|| /* hindrance */)
        .then_with(|| self.flow_mismatch.cmp(&other.flow_mismatch))
        .then_with(|| /* epsilon */)
}
```

This is a 1-line reorder. The semantic change: guidance, goal distance, and hindrance
dominate flow — flow is now purely a defensive tiebreak, not a primary signal.

## Modelless check

This proposal is **entirely modelless** per the `katgpt-rs/AGENTS.md` mandate:

- (A) is closed-form topology analysis (neighbor counting + parity assignment).
- (B) is a boolean check on a precomputed table.
- (C) is a tuple reorder.

No training. No gradient descent. No weights. No backprop. The only mutations to the flow
field are at *construction time* (bi-directional assignment) — at runtime the field is
read-only.

## GOAT gate (what the plan must prove)

The plan that lands this proposal must run:

- **G1** (throughput): ht_chantry-real ≥ 0.30 (close Issue 546); empty/random/warehouse ≥ 0.30 (no regression).
- **G3** (no-regression): all 1601+ lib tests pass; collision-freedom gates G6c/G-col stay perfect.
- **G4** (latency): median ≤ 100ms at 800 agents on the 3 non-maze maps; ht_chantry ≤ 500ms.
  The hard pruner **reduces** A\* expansion count (fewer neighbors to consider), so this
  should pass trivially.
- **G-flow** (flow-specific gate, new): on ht_chantry, the fraction of stuck-tick
  max-cluster-sizes ≥ 4 should drop by ≥ 50% vs the Issue 546 diagnostic baseline. This is
  the *mechanism* gate — proving the redesign actually prevents corridor queues, not just
  papers over them.

If G1 ht_chantry fails (still < 0.30) but G-flow passes (cluster sizes drop), the proposal
is *architecturally correct but insufficient* — we accept the marginal fail as the honest
steady-state (Issue 546 path 2). If G1 ht_chantry fails AND G-flow fails, the proposal is
wrong and we revert.

## Risks

- **(A) 1-wide corridor parity may regress maps that worked.** The even/odd split is
  arbitrary; a map with corridors mostly oriented one way could see agents routed through
  longer detours. Mitigation: the `sign = 0` fallback at articulation points guarantees
  reachability; the G3 no-regression gate catches throughput drops on the 3 passing maps.
- **(B) Hard pruner may make some paths unreachable.** If (A) has a bug and a corridor
  segment has no bi-directional counterpart, agents with goals across that segment cannot
  reach them. Mitigation: A\*'s existing best-effort fallback (closest-to-goal partial path)
  prevents hard failure; the G3 gate catches widespread unreachability.
- **(C) Demotion may regress if A\* and flow disagree in subtle cases.** If a future change
  makes A\* produce flow-violating paths (e.g., a new guidance source that ignores flow),
  demoting `flow_mismatch` means PIBT won't catch it. Mitigation: keep the term in the tuple
  (just at lower priority); add a debug-mode assertion that A\*-preferred cells have
  `flow_mismatch == 0`.
- **Test surface.** The flow field is currently tested with 11 tests (per README §Plan 440).
  The redesign requires new tests for bi-directional assignment, parity, fallback, and
  flow-respecting A\*. Estimated +20 tests.

## What is explicitly NOT in scope

- **Dynamic flow reversal** (time-varying flow based on observed traffic). This is a future
  extension — (A) is static topology only. Dynamic reversal risks non-determinism (would
  break replay).
- **Flow-balanced global assignment** (LP / min-cost-flow to optimize direction assignment).
  (A)'s local parity rule is a heuristic approximation. Global optimization is a future
  enhancement if (A)+(B)+(C) prove insufficient.
- **Multi-step LaCAM** (Issue 546 original plan). DEFERRED per the diagnostic. This proposal
  is the alternative path.

## References

- `Issue 182` — implementation plan (CLOSED with REVERT verdict; removed per noise-reduction rule)
- Issue 546 (riir-ai, removed per noise-reduction rule) — DEFERRED, motivates this proposal
- [Issue 546 diagnostic](../crates/katgpt-core/examples/ht_chantry_deadlock_chain_diagnostic.rs) (commit `2a8c378d`) — the P95=8 result
- [Benchmark 440](../.benchmarks/440_lllg_paper_repro_goat.md) — LLLG paper reproduction (3/4 PASS)
- [Benchmark 453](../.benchmarks/453_lacam_escalation_goat.md) — one-step LaCAM escalation (shipped)
- [Plan 440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md) — LLLG substrate
- [Plan 453](../.plans/453_bounded_one_step_lacam_escalation.md) — one-step LaCAM
- [Research 424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md) — LLLG paper distillation
- [Research 441](../.research/441_lacam_constraint_tree_distillation.md) — LaCAM constraint tree distillation
- `crates/katgpt-core/src/multi_agent_path/flow.rs` — current `GridFlowField` impl
- `crates/katgpt-core/src/multi_agent_path/local_guidance.rs:459-555` — the A\* search that ignores flow
- `crates/katgpt-core/src/multi_agent_path/pibt.rs:148-186` — the cost tuple with `flow_mismatch` at position 2

---

## Verdict (2026-07-18, post-implementation)

**Status flipped: draft → REJECTED.** Issue 182 Phases 1-3 were implemented,
bench_440 re-run, and the deadlock-chain diagnostic re-run. The result is a
**negative on the target metric** (ht_chantry G1) AND a **negative on the
mechanism metric** (P95 cluster size). Per the proposal's own P5.3 rule, this
is the REVERT case.

### What was measured

**bench_440 G1 (800 agents, 300 steps, seed=42):**

| Map | Pre-Proposal | Post-Proposal | Delta | Gate |
|---|---|---|---|---|
| empty-48-48 | 18.52 (0.68) | 18.75 (0.69) | +1.2% | PASS (no corridors, no change expected) |
| random-64-64-10 | 14.56 (0.69) | 13.98 (0.66) | **-4.0%** | PASS but **regression** (needs investigation) |
| warehouse | 7.33 (0.41) | **7.84 (0.44)** | **+7.0%** | Improved but still FAIL (target 0.5) |
| ht_chantry | 4.66 (0.27) | 4.70 (0.28) | +0.9% (noise) | **FAIL** (target ≥0.30) |

**ht_chantry deadlock-chain diagnostic (800 agents, 500 steps, seed=42):**

| Metric | Pre-Proposal (commit `2a8c378d`) | Post-Proposal | Delta |
|---|---|---|---|
| Throughput | **5.098** | **4.824** | **-5.4% regression** |
| P95 max-cluster-size | **8** | **15** | **+87.5% (WORSE)** |
| P99 max-cluster-size | 9 | 18 | +100% (WORSE) |
| Max observed | 11 | 20 | +82% (WORSE) |

### Why the proposal's hypothesis failed

The proposal predicted that bi-directional corridors + A\* hard pruner would
close ht_chantry G1 by eliminating corridor deadlocks. The mechanism IS
working on warehouse (+7% throughput — 4920 2-wide corridor cells, dense
aisle structure, exactly the proposal's target topology). But ht_chantry's
corridor topology is **not what the proposal assumed**:

1. **The 8 1-wide corridor cells are all articulation points** (single-passage
   bottlenecks) — they get the `sign = 0` fallback (P1.4), so the proposal's
   mechanism doesn't apply to them. No flow enforcement on the maze's narrowest
   passages.
2. **The 102 2-wide corridor cells get bi-directional pairing**, but they're
   short segments between larger open regions. The A\* hard pruner forces
   agents into the correct lane (+1 lane for +travel, -1 lane for -travel),
   which **lengthens paths** when an agent must enter a 2-wide corridor in
   the direction its current lane doesn't serve. The longer paths compound:
   more steps in transit = more congestion at corridor approaches = larger
   stuck clusters.
3. **The deadlock-chain P95 went UP from 8 to 15.** This is the smoking gun:
   the A\* pruner is causing agents to bunch up at the 2-wide corridor
   approaches, creating larger deadlock chains than before. The proposal's
   mechanism is **trading deadlock count for cluster size** — net negative on
   ht_chantry.

### Where the proposal's mechanism DOES work

The warehouse result (+7% throughput) is real and reproducible. Warehouse has
4920 2-wide corridor cells (50% of passable cells) forming long, regularly-
spaced aisles — exactly the topology the proposal targeted. On such maps:

- The bi-directional two-lane highway eliminates head-on deadlocks in aisles.
- The A\* hard pruner routes agents correctly without significant path
  lengthening (aisles are long, so the lane constraint adds ≤1 step).
- Net throughput improves.

But warehouse is still below its G1 target (0.44 vs 0.5), so even the
positive case doesn't clear the GOAT gate. The mechanism is **architecturally
sound for dense-aisle maps but insufficient for the gate**.

### Action: REVERT

All Phase 1-3 code changes were reverted (multi_agent_path/{flow.rs,
local_guidance.rs, mod.rs, pibt.rs, tests.rs}). The proposal and issue files
remain as a record of the negative result. The ht_chantry G1 gap (0.27-0.28
vs 0.30 target) **remains open** and is now a documented steady-state fail
awaiting a different approach.

The warehouse +7% data point is preserved in this verdict for any future
proposal that targets dense-aisle maps specifically. A future proposal could
revive the mechanism behind a feature flag (e.g. `flow_bidirectional`) that
consumers opt into per-map — but that's out of scope here.

### Lessons

1. **Mechanism correctness ≠ target metric improvement.** The proposal's
   three root causes were all correctly identified, and the three phases each
closed one. The mechanism IS doing what it was designed to do. But the target
   map's topology interacted with the mechanism in an unexpected way
   (path-lengthening compounds into larger clusters), producing a net
   negative on the target metric.
2. **Always re-run the mechanism diagnostic, not just the throughput bench.**
   The bench_440 ht_chantry result (+0.9%) looked like noise-level
   improvement. The diagnostic revealed the mechanism was actually **causing
   larger deadlock chains** (P95 8→15) — the opposite of its design goal. The
   diagnostic is the more sensitive instrument.
3. **Sparse-corridor maps are not dense-corridor maps.** The proposal's
   mental model was "corridors cause deadlocks; fix the corridors → fix the
   deadlocks." That holds when corridors are the dominant topology (warehouse,
   50% corridor cells). It fails when corridors are a minor feature
   (ht_chantry, 1.5% corridor cells) and the dominant topology is open regions
   with sparse bottlenecks. Different topologies need different mechanisms.
