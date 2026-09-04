# Benchmark 440: LLLG Paper Reproduction GOAT Gate

**Date:** 2026-07-15 (updated Issue 150)
**Plan:** [440](../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md)
**Research:** [424](../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md)
**Paper:** [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, AAAI 2026
**Issue:** `150` — 2-wide corridor detection (mechanism correct, still near-zero effect — the cost-term tiebreak position is too weak)
**Prior issue:** `149` — 1-wide corridor detection (near-zero effect due to corridor-definition mismatch)
**Prior issue:** `148` — real MovingAI benchmark maps (map-fidelity hypothesis test)
**Prior issue:** `147` — ht_chantry connectivity fix (10× improvement) + counter-flow Guided-PIBT (negative result)
**Prior issue:** `144` — swap technique (negative result, infrastructure-only)
**Prior issue:** `143` — LaCAM escalation (greedy PIBT + priority shuffle retry)
**Prior issue:** `142` — full space-time A* upgrade
**Prior issue:** `140` — PIBT PI + warm-start investigation

---

## TL;DR

**G3 (no-regression): PASS. G4 (latency): PASS. G1 (throughput): PARTIAL
(2/4 real maps). G2 (congestion): FAIL (warm-start not consumable — confirmed
by Issue 142 even with full A*).**

**Issue 150 finding (2-wide corridor detection):** Broadened the corridor
 detector to also catch 2-wide passages (pairs of adjacent passable cells
 flanked by walls). The 2-wide detection finds **significantly more corridors**:

- `ht_chantry`: 8 → **110** corridor cells (13.75× more)
- `warehouse`: 0 → **4920** corridor cells (50% of passable cells!)
- `random`: 63 → **245** (3.9× more)

But throughput is **still unchanged** (within noise) on all 4 maps. The
`flow_mismatch` cost term is a tiebreak at position 2 in the lexicographic tuple —
it only fires when `guidance_mismatch` is tied. Even with 4920 corridor cells on
warehouse, the term's effect is negligible because the guidance source
(space-time A*) already steers agents well, and the flow tiebreak rarely changes
the selected candidate. The gap is **not corridor detection** — it's the
**strength of the flow enforcement mechanism**.

| Map | Issue 149 (1-wide only) | Issue 150 (1+2-wide) | Corridors | Change |
|---|---|---|---|---|
| empty-48-48 | 18.52 (0.68) | **18.52 (0.68)** | 0 → 0 | identical |
| random-64-64-10 | 14.65 (0.69) | **14.56 (0.69)** | 63 → 245 | -0.6% (noise) |
| warehouse | 7.34 (0.41) | **7.33 (0.41)** | 0 → 4920 | -0.1% (noise) |
| ht_chantry | 4.61 (0.27) | **4.66 (0.27)** | 8 → 110 | +1% (noise) |

**Follow-up attempt — Proposal 006 (2026-07-18, REJECTED):** Lifting the flow
field from a PIBT-level soft cost to a guidance-level hard constraint was
implemented in three phases (bi-directional corridor pairing + flow-respecting
A\* + demoted `flow_mismatch` cost position), measured, then **reverted**.
ht_chantry throughput stayed flat (+0.9%, noise) AND ht_chantry deadlock-chain
P95 went from 8 → 15 (WORSE); the mechanism DID work on warehouse (+7%) but
that's not enough to clear either map's gate. Full measurement record + REVERT
verdict live in [`Proposal 006 §Verdict`](../.proposals/006_flow_field_hard_constraint_in_guidance.md#verdict-2026-07-18-post-implementation);
the implementation issue (182) was removed per noise-reduction rule. The
ht_chantry G1 gap (0.27–0.28 vs 0.30 target) remains a documented steady-state
fail awaiting a different approach.

---

## Issue 150 results (2026-07-15) — 2-wide corridor detection

### What was done

Broadened `GridFlowField::from_map` to detect both 1-wide and 2-wide corridors.
A 2-wide corridor is a pair of adjacent passable cells flanked by walls on both
sides perpendicular to the adjacency axis. Each cell in the pair gets the same
`FlowDirection` (sign=+1).

Added a `width: u8` field to `FlowDirection` (1 or 2) for diagnostics. Added
`corridor_1wide_count()` and `corridor_2wide_count()` diagnostic methods. 7 new
unit tests covering 2-wide horizontal/vertical detection, flow mismatch, 3-wide
exclusion, junction exclusion, open-map no-regression, and mixed 1+2-wide
coexistence.

### Result: corridor detection now covers real maps, but throughput unchanged

The 2-wide detector finds **dramatically more corridors** on real game maps:

| Map | 1-wide (Issue 149) | 2-wide (Issue 150) | Total | Coverage |
|---|---|---|---|---|
| empty-48-48 | 0 | 0 | 0 | 0.0% |
| random-64-64-10 | 63 | 182 | 245 | 6.6% |
| warehouse | 0 | **4920** | **4920** | **50.3%** |
| ht_chantry | 8 | 102 | 110 | 1.5% |

**Warehouse now has 4920 corridor cells (50% of passable cells!)** — the entire
aisle structure is 2-wide. ht_chantry went from 8 to 110 corridor cells.

But throughput is **still unchanged** (within noise) on all 4 maps:

| Map | Issue 149 | Issue 150 | Change |
|---|---|---|---|
| empty-48-48 | 18.52 (0.68) | **18.52 (0.68)** | identical |
| random-64-64-10 | 14.65 (0.69) | **14.56 (0.69)** | -0.6% (noise) |
| warehouse | 7.34 (0.41) | **7.33 (0.41)** | -0.1% (noise) |
| ht_chantry | 4.61 (0.27) | **4.66 (0.27)** | +1% (noise) |

### Root cause: the tiebreak position is too weak

The `flow_mismatch` cost term sits at **position 2** in the 5-tuple:

```text
⟨ guidance_mismatch, flow_mismatch, goal_dist, hindrance, ε ⟩
```

It only breaks ties between candidates with the same `guidance_mismatch` (0 or 1).
In practice, the guidance source (space-time A* on collision-count cost) steers
agents well enough that `guidance_mismatch` is almost always 0 for the preferred
move. When guidance_mismatch is 0 for multiple candidates (e.g., wait + move
forward are both guidance-consistent), the flow term does influence selection —
but this rarely changes the final collision-free choice because the
collision-checking loop tries candidates in sorted order and the first
collision-free one wins regardless of flow.

**The bottleneck is not corridor detection — it's the enforcement mechanism.**
The flow direction is a soft hint, not a hard constraint. To actually improve
throughput on warehouse/ht_chantry, the flow direction needs to influence the
**guidance source** (space-time A* should prefer routes aligned with the flow
field), not just the PIBT tiebreak.

### GOAT gate status (unchanged)

| Gate | Status | Detail |
|---|---|---|
| **G1** | **PARTIAL 2/4** | empty 0.68 ✅, random 0.69 ✅, warehouse 0.41 ❌, ht_chantry 0.27 ❌. Unchanged. |
| **G2** | **FAIL** | Warm-start non-consumable (ratio 1.00). Unchanged. |
| **G3** | **PASS** | 1611 tests pass (7 new 2-wide tests). Clippy clean. |
| **G4** | **PASS** | 225ms median at 1000 agents (<500ms target). |

**Promotion decision: KEEP OPT-IN** (unchanged).

---

## Issue 149 results (2026-07-15) — Guided-PIBT flow direction assignment

### What was done

Implemented the `FlowField<P>` trait + `GridFlowField` impl in a new `flow.rs`
module. The flow field detects 1-wide corridor cells (cells with exactly 2
passable neighbors on opposite sides) and assigns a canonical one-way direction
(+1 sign: right for horizontal corridors, down for vertical). A new
`flow_mismatch` cost term was inserted into the PIBT candidate tuple at
position 2 (between `guidance_mismatch` and `goal_dist`):

```text
⟨ guidance_mismatch, flow_mismatch, goal_dist, hindrance, ε ⟩
```

The `LifelongLaCam` orchestrator gained a `.with_flow_field()` builder.
11 unit tests cover corridor detection, direction assignment, flow mismatch
computation, and orchestrator integration.

### Result: mechanism correct, near-zero effect on real maps

The flow direction assignment mechanism is **correctly implemented and tested**
(proven by 11 unit tests on synthetic 1-wide corridors), but it has **near-zero
effect** on the real benchmark maps. The root cause is a **corridor-definition
mismatch**: the detector catches strict 1-wide passages, but real game-map
corridors are 2-wide or wider.

| Map | Passable cells | Corridor cells (detected) | Corridor % | Throughput change |
|---|---|---|---|---|
| empty-48-48 | 2304 | 0 | 0.0% | identical |
| random-64-64-10 | 3687 | 63 | 1.7% | +2% (noise) |
| warehouse | 9776 | 0 | 0.0% | identical |
| ht_chantry | 7461 | **8** | **0.1%** | +2% (noise) |

The real ht_chantry (Dragon Age: Origins) has only **8 strict 1-wide corridor
cells** — game-map corridors are wider. The warehouse has **0 corridors** (wide
aisles between shelf blocks). The flow_mismatch term fires on almost no cells,
so throughput is unchanged.

### No regression on open maps

The "safe promotion" design was verified: on open maps (empty/random), the flow
field has 0 or few corridors, so `flow_mismatch` is almost always 0, and the
cost tuple degenerates to the paper-faithful ordering. Empty-48-48 throughput
is bit-identical (18.52 → 18.52).

### What this means for the roadmap

The flow field mechanism is **correct but insufficient** for real game maps.
The corridor definition needs broadening to catch 2-wide passages (the actual
topology of game-map corridors like ht_chantry). This is a more complex
detection problem:

- A 2-wide horizontal corridor: two adjacent rows of passable cells, both
  flanked by walls above/below. Each cell has 3 passable neighbors (left, right,
  and the paired row), so the strict "exactly 2 neighbors" detector misses them.
- Detection needs to identify **passage width** — groups of cells where the
  passage is narrow (≤ 2 wide) relative to the surrounding open area.

The `FlowField` pluggable seam is shipped and ready for consumers who want to
implement a broader corridor detector. The default `GridFlowField` handles
the 1-wide case correctly.

### GOAT gate status (unchanged)

| Gate | Status | Detail |
|---|---|---|
| **G1** | **PARTIAL 2/4** | empty 0.68 ✅, random 0.69 ✅, warehouse 0.41 ❌, ht_chantry 0.27 ❌. Unchanged. |
| **G2** | **FAIL** | Warm-start non-consumable (ratio 1.00). Unchanged. |
| **G3** | **PASS** | 1601 tests pass (11 new flow field tests). Clippy clean. |
| **G4** | **PASS** | 467ms median at 1000 agents (<500ms target). |

---

## Issue 148 results (2026-07-15) — real MovingAI benchmark maps

### What was done

Downloaded the real MovingAI MAPF benchmark maps from
https://movingai.com/benchmarks/mapf/mapf-map.zip for all 4 paper scenarios:
`empty-48-48.map`, `random-64-64-10.map`, `warehouse-10-20-10-2-2.map`,
`ht_chantry.map`. Embedded them in `benches/data/*.map` and added a
`GridMap::from_movingai(text)` parser (reusable leaf primitive, with 5 unit
tests in `tests.rs`). The G1 gate now runs against the real maps; the
synthetic approximations are kept as diagnostic comparisons.

### Map topology comparison

| Map | Synthetic dims | Real dims | Real passable | Real corridor% |
|---|---|---|---|---|
| empty-48-48 | 48×48 / 2304 | 48×48 / 2304 | 2304 (100%) | 0.2% |
| random-64-64-10 | 64×64 / 4096 | 64×64 / 4096 | 3687 (90%) | 6.5% |
| warehouse | 63×45 / 2835 | **170×84 / 14280** | **9776 (68.5%)** | **0.0%** |
| ht_chantry | 71×53 / 3763 | **162×141 / 22842** | **7461 (32.7%)** | **2.9%** |

The synthetic warehouse and ht_chantry were both dramatically smaller than
the real maps. The real warehouse has 5× more passable cells; the real
ht_chantry has 2.5× more. Both real maps are single connected components
(verified via flood-fill).

### Result: map-fidelity hypothesis partially confirmed

**ht_chantry (the Issue 147 hypothesis):**
- Synthetic ratio 0.09 → Real ratio **0.27** (3× improvement).
- The synthetic's 5.9% corridor density (vs real 2.9%) was capping throughput
  at ~1.5; the real map's more open topology allows 4.51.
- **But 0.27 is still below the 0.30 PASS threshold.** A genuine ~4×
  algorithmic gap remains (4.51 vs paper ~17). Guided-PIBT (full flow
  direction assignment) is now genuinely warranted — this is no longer
  deferrable on map-fidelity grounds.

**warehouse (unexpected finding):**
- Synthetic ratio 0.42 → Real ratio **0.41** (essentially unchanged).
- The real warehouse is 5× larger (9776 vs 1971 passable cells) with wider
  aisles, yet throughput is identical. **Warehouse congestion is not
  size-limited — it's a genuine algorithmic limit on shelf-aisle topology.**
- Diagnostic: real warehouse max_stops=8 (agents rarely stuck), but
  throughput 7.34 means agents complete ~7 tasks/step vs paper's ~18. The
  gap is task-completion rate, not deadlock.

**empty/random (sanity check):**
- empty-48-48: identical (synthetic is exact by construction).
- random-64-64-10: -1.4% (17 fewer passable cells in real map; negligible).

### GOAT gate status (updated)

| Gate | Status | Detail |
|---|---|---|
| **G1** | **PARTIAL 2/4** | empty 0.68 ✅, random 0.68 ✅, warehouse 0.41 ❌, ht_chantry 0.27 ❌ (MARGINAL). All 4 maps now real MovingAI files. |
| **G2** | **FAIL** | Warm-start non-consumable (ratio 1.00). Unchanged. |
| **G3** | **PASS** | 1590 tests pass (5 new parser tests). Clippy clean. |
| **G4** | **PASS** | 222ms median at 1000 agents (<500ms target). |

### What this changes for next steps

1. **Full Guided-PIBT is now un-deferrable for ht_chantry.** The map-fidelity
   excuse is exhausted — the real map confirms a genuine ~4× algorithmic gap
   on maze topology. The `CounterFlowHindrance` infrastructure from Issue 147
   is available but needs to be promoted from the 3rd PIBT tiebreak to a
   higher-priority cost term (ahead of `goal_dist`), changing the algorithm's
   character. This is the algorithmic upgrade the paper recommends for mazes.

2. **Warehouse needs a different fix than size.** The 5× larger real warehouse
   gave zero improvement, ruling out map-size scaling as the cause. The gap
   is task-completion rate on shelf-aisle structure — likely needs intersection
   reservation or shelf-aware goal assignment, not global routing.

3. **riir-ai/489 fusion remains pragmatic.** The substrate works correctly on
   2/4 real maps (open + random topologies). The warehouse/maze gaps are
   documented algorithmic limits, not runtime bugs. Consumers with open/random
   topologies can consume the substrate as-is.

---

## Issue 147 results (2026-07-15) — ht_chantry connectivity fix + counter-flow Guided-PIBT

### Root cause: map generator created 37 disconnected components

The prior ht_chantry throughput of 0.01 (Issues 140–144) was misdiagnosed as
severe bottleneck congestion requiring Guided-PIBT. Diagnostic
(`crates/katgpt-core/examples/ht_chantry_diagnostic.rs`) revealed the `ht_chantry_approx` map
generator created **37 disconnected components**:

- Full-width horizontal walls (every 8 rows) with two 2-wide gaps each.
- Full-height vertical walls (every 10 columns) with one 2-wide gap each.
- At wall intersections, the combined walls sealed off regions.
- Only **24% of passable cells** were in the largest component.
- Agents placed in small components could never reach their goals (BFS
  returned `f32::MAX` → agent waited forever).

### Fix: `ensure_connected` post-processing

Added `ensure_connected()` to `ht_chantry_approx` in the bench. After
generating the maze walls, flood-fill to label components, then iteratively
punch holes (remove wall cells) to merge smaller components into the largest
one. Only **36 wall cells** were removed (0.9% of the map) to achieve full
connectivity. The maze character is preserved (177 corridor cells, 5.9%
corridor density).

### Result: 10× throughput improvement

| Metric | Before (Issue 144) | After (Issue 147) | Change |
|---|---|---|---|
| Throughput (800 agents) | 0.15 | **1.47** | **+880%** |
| Ratio vs paper (~17) | 0.01 | **0.09** | 10× |
| Completions (300 steps) | 3 | **442** | 147× |
| Max stops/cell | 372 | **372** | unchanged |

### Remaining gap analysis (1.47 vs paper ~17)

A config sweep (`crates/katgpt-core/examples/ht_chantry_config_sweep.rs`) tested:

1. **w_Φ sweep** (5, 10, 15, 20): w_Φ=10 gives +16% (1.00→1.16 at 200
   agents). w_Φ=15+ shows diminishing returns (1.06, regresses).
2. **α sweep** (0.0, 0.5, 1.0): α=0 (pure global BFS, no collision
   avoidance) does NOT help — throughput stays at 0.96 and max_stops
   INCREASES (78→124). The collision avoidance IS useful.
3. **Density scaling** (50–600 agents): per-agent throughput drops
   monotonically (0.0074 → 0.0023), confirming corridor saturation.
   Total throughput plateaus at ~1.4 above 400 agents.

**Conclusion:** the remaining gap is **map fidelity**, not algorithmic. Our
synthetic maze has extreme bottlenecks (177 corridor cells) that physically
limit throughput to ~1.4 regardless of routing algorithm. The paper's real
ht_chantry (MovingAI benchmark) likely has more/wider corridors.

### Counter-flow Guided-PIBT (negative result)

Implemented `CounterFlowHindrance` in `hindrance.rs`: a hindrance estimator
that penalizes agents whose goal direction is anti-aligned with nearby
siblings (head-on corridor approach). Tested at γ = 0, 1, 2, 5, 10.

**Result: zero improvement at all γ values.** Throughput and max_stops are
identical to baseline (1.00, 78 at 200 agents). Root cause: the hindrance
term is the **3rd PIBT tiebreak** — it only matters when guidance_mismatch
AND goal_dist are tied. On maze maps, agents almost always have a clear best
candidate, so the hindrance tiebreak rarely fires. Same lesson as the swap
technique (Issue 144): modifying a low-priority tiebreak doesn't change
behavior.

The `CounterFlowHindrance` infrastructure is kept for consumers who can
promote it to a higher-priority term via a custom PIBT cost tuple.

### GOAT gate status (updated)

| Gate | Status | Detail |
|---|---|---|
| **G1** | **PARTIAL 3/4** | empty 0.68 ✅, random 0.69 ✅, warehouse 0.42 ✅, ht_chantry 0.09 ❌ (improved from 0.01, still below 0.15 MARGINAL) |
| **G2** | **FAIL** | Warm-start non-consumable (ratio 1.00). Unchanged. |
| **G3** | **PASS** | 1585 tests pass. Clippy clean. |
| **G4** | **PASS** | 225ms median at 1000 agents (<500ms target). |

### Next steps

1. **Download real MovingAI ht_chantry map** — the synthetic approximation's
   extreme bottlenecks are not representative. A fair comparison requires
   the actual benchmark map.
2. **riir-ai/489 fusion** — the substrate works correctly on 3/4 maps. The
   ht_chantry limitation is a known benchmark constraint, not a runtime bug.
3. **Full Guided-PIBT (flow direction assignment)** — would require
   promoting counter-flow awareness to a higher-priority PIBT cost term
   (ahead of goal_dist), which changes the algorithm's character. Deferred
   until the real map confirms the algorithmic gap.

---

## Issue 144 results (2026-07-15) — swap technique (negative result)

### What changed

1. **Swap technique implemented** in `pibt.rs`: `detect_swap_backers` scans
   for head-on corridor deadlocks (agent i wants j's cell, j wants i's cell)
   and marks the lower-priority agent as a "backer". The backer uses reverse
   scoring ⟨0, −dist(v, g_i), ε⟩ to back up (move to the cell farthest from
   its goal), clearing the path for the higher-priority forward agent.

2. **`greedy_pibt_pass` gained a `swap_backers` parameter**: backers are
   processed first (partition reorder) and use reversed candidate scoring.
   When the backer set is empty, behavior is identical to Issue 143.

3. **2 new tests**: `test_swap_resolves_head_on_corridor` (passing-bay
   scenario verifies the swap mechanism works mechanically),
   `test_swap_no_collision_in_wide_corridor` (4 agents in 2-wide corridor,
   verifies no vertex/edge collisions).

### Three benchmark configurations tested

1. **Always-on** (swap in initial pass, all maps): REGRESSED all maps.
   The swap triggers on every head-on conflict, forcing one agent to back up.
   On open maps this is wasteful (agents could sidestep); on warehouse it's
   catastrophic (forced back-ups in aisles compound). This mirrors the
   recursive-PIBT finding (Issue 143): forcing agents to move away from
   goals hurts lifelong MAPF throughput.

2. **Gated escalation** (swap fires only when ≥ 20 stuck agents, result
   kept only if fewer stuck than baseline): open maps restored (no
   regression), but warehouse still regressed to 0.24. The swap reduces
   stuck-agent count (agents moved via back-up) but not throughput — the
   backed-up agents are now farther from their goals.

3. **Infrastructure-only** (swap NOT wired into `pibt_step`): no regression,
   no improvement. Back to Issue 143 baseline exactly.

### Why the swap doesn't help ht_chantry

The swap technique targets **1-wide corridor deadlocks** (Okumura's
warehouse-20-40-10-2-1). In a 1-wide corridor, two agents facing head-on have
no sidestep option — one must back up. The swap detects this and resolves it.

Our ht_chantry approximation uses **2-wide corridors** (the generator creates
2-cell-wide gaps in wall segments). In a 2-wide corridor, two agents facing
head-on can sidestep past each other — the A* guidance routes them around.
The swap pattern (i wants j's cell, j wants i's cell) rarely fires because
agents naturally take different cells in the 2-wide passage.

### Why the swap hurts warehouse

The warehouse map has 1-wide aisles between shelf blocks. Agents in aisles do
face head-on conflicts, and the swap fires. But in lifelong MAPF, forcing an
agent to back up (move away from its goal) has a cascading cost: the agent
must re-traverse the aisle later, blocking other agents. Over 300 ticks, this
cascading back-up pattern severely reduces sustained throughput (0.42 → 0.24).

The stuck-agent count metric (used to decide whether to keep the swap result)
is misleading here: the swap reduces stuck agents (they moved via back-up) but
the moves are in the wrong direction. The right metric (throughput) can only
be measured over many ticks, not within a single PIBT step.

---

### What changed

1. **LaCAM escalation** added to `pibt.rs`: when the greedy PIBT produces
   ≥ 20 stuck agents (systemic congestion), retry with shuffled priority
   orderings (up to 2 retries). The stuck agents are elevated to high
   priority, and non-stuck agents are randomly perturbed. The result with
   the fewest stuck agents is returned.

2. **Recursive PIBT tested and REJECTED.** The full recursive priority
   inheritance (agent i evicts undecided agent j from its cell, recursively)
   was implemented and benchmarked. Result: **throughput collapses -92%**
   (empty-48-48: 18.6 → 1.5). The eviction forces agents to move away from
   their goals, creating cascading stalls. The greedy PIBT — which lets
   agents compromise by taking their next-best cell — has dramatically
   higher collective throughput in the lifelong MAPF setting. The recursive
   variant is right for one-shot MAPF (finding ANY solution), wrong for
   lifelong MAPF (sustained throughput).

3. **Stuck-agent threshold** (MIN_STUCK_FOR_RETRY = 20): prevents retry
   overhead on open maps where occasional stuck agents (1-5) resolve
   naturally next tick. Retries only fire on genuinely congested maps.

### Latency analysis

The LaCAM retry adds overhead on dense maps:
- **800 agents, empty/random**: median 60-70ms (retries rarely trigger — too
  few stuck agents). Comparable to pre-retry baseline (~88ms).
- **800 agents, warehouse**: median 134ms (retries trigger — warehouse has
  systemic shelf-aisle congestion). The +8.3% throughput gain justifies the
  cost.
- **1000 agents, empty-48-48 (G4)**: median 234ms (retries trigger at 43%
  density). Under the 500ms target but above the 100ms stretch goal.

### ht_chantry — why local retry doesn't help

The maze topology creates head-on corridor conflicts: two agents meet in a
1-wide passage, both want to pass. Neither can move forward (vertex
conflict), neither can wait (the other is blocking), and backing up requires
one agent to reverse direction — which the greedy PIBT doesn't consider (it
prefers goal-directed moves). The priority shuffle doesn't help because the
issue isn't WHO goes first, it's that SOMEONE must back up, and no priority
ordering makes that happen with local decisions.

The fix is **global routing** (Guided-PIBT from the paper): pre-compute
flow directions for corridors and route agents accordingly. This is a
significantly larger implementation and is the paper's own recommended
approach for long-corridor maps (caveat #1).

---

## Issue 142 results (2026-07-15) — full space-time A*

Issue 142 replaced the greedy rollout with a proper priority-queue space-time
A* and fixed the broken multi-round refinement. This is the upgrade that Issue
140 identified as the real blocker.

### What changed

1. **`astar_for_agent` rewritten** from greedy rollout to proper A* over
   `(position, depth)` state space with BFS-distance heuristic. Priority queue
   (BinaryHeap), g/h/f scores, came_from path reconstruction. The A* has w-step
   lookahead with proper cost accumulation — it can plan multi-step detours
   around collisions.

2. **Multi-round refinement fixed** (unrecord/re-record). Previously each
   round called `clear_occupancy()`, making rounds no-ops (agent 0 always saw
   an empty map). Now each agent unrecords its previous path before recomputing,
   so round 1 agent 0 sees round-0 paths of agents 1..n-1.

3. **Dead code removed**: `step_cost_bfs` and `cycle_penalty` free functions
   (greedy rollout helpers) deleted. `collision_count` `#[allow(dead_code)]`
   removed (now used by the A*).

### Warm-start consumption — TRIED AND CONFIRMED HARMFUL

Occupancy-seeding with warm-start forecasts was implemented and benchmarked:

| Config | empty-48-48 throughput |
|---|---|
| A* without warm-start seeding | **18.60** |
| A* with warm-start seeding | 14.73 |

Seeding HURTS by 21%. The forecast is invalidated when PIBT deviates from the
guidance (common on dense maps), creating misleading phantom collision
constraints that the A* routes around. This confirms Issue 140's finding but
with the full A* — the problem isn't the greedy rollout, it's that warm-start
forecasts are too stale on dense maps without LaCAM escalation.

**Decision:** warm-start data is consumed (taken/cleared) but NOT seeded into
the occupancy. LllgPi = LllgEmpty with the current implementation. Positive
warm-start consumption likely requires LaCAM escalation (Phase 5) to keep
forecasts accurate.

---

## Issue 140 investigation results (2026-07-15)

The Issue 140 analysis identified two blocking items for G1/G2 full pass:
1. PIBT priority inheritance upgrade
2. Warm-start integration

Both were implemented, benchmarked, and found to be **blocked by a deeper
architectural gap**: the greedy guidance rollout.

### PIBT priority inheritance — implemented, benchmarked, REVERTED

The full recursive PIBT with priority inheritance was implemented (~200 lines
of recursive backtracking in `pibt.rs`):

- `pibt_recursive` function: when agent `i` wants cell `u` occupied by
  undecided agent `j`, recursively push `j` to move before committing.
- `in_chain: HashSet<usize>` for cycle prevention.
- `MAX_PIBT_DEPTH = 64` cap for pathological recursion.
- `blocked_cell: Option<&P>` swap prevention (pushed agent can't take
  pusher's cell).

**Benchmark result:** Throughput COLLAPSED on all maps:
- empty-48-48: 17.32 → 0.47 (ratio 0.63 → 0.02)
- random-64-64-10: 13.15 → 0.36
- All maps: mean_stops 170-300 out of 300 steps (agents barely moving)

**Root cause:** The recursive push is too conservative for dense maps
without LaCAM-level search escalation. In the paper's design, PIBT is
combined with LaCAM — when PIBT fails to resolve a conflict, LaCAM
escalates to a full search. Without LaCAM, the recursive push requires
occupants to vacate before committing, causing cascading stalls. The greedy
PIBT (take first collision-free candidate, let later agents adapt) has
higher throughput in the lifelong MAPF setting without LaCAM.

**Decision:** Reverted to the greedy PIBT (Phase 2 code). The recursive PIBT
is deferred until LaCAM is added (Phase 5). Documented in `pibt.rs` module
docs.

### Warm-start integration — infrastructure landed, consumption DEFERRED

The warm-start integration was plumbed end-to-end:

- `set_warm_start(Vec<Vec<P>>)` method added to `LocalGuidanceSource` trait
  (default no-op).
- `SpaceTimeGuidance` stores the data in a new `warm_start: Option<Vec<Vec<P>>>`
  field.
- `LifelongLaCam::tick` calls `guidance.set_warm_start(warm)` before
  `compute_guidance`.
- Solution recording fixed: prepends the executed PIBT action so suffix
  extraction correctly skips the executed step (T2.6 complete).

**Consumption attempt 1 (occupancy seeding, all paths):**
Throughput COLLAPSED: empty-48-48 17.32 → 0.47. The warm-start forecast
seeded ALL agents' paths into the occupancy map, causing agents to avoid
forecast cells (including their own forecast — self-collision). G2 "passed"
(ratio 0.48) but G1 was destroyed.

**Consumption attempt 2 (occupancy seeding, self-collision removal):**
Throughput still collapsed. Removing each agent's own forecast before
processing helped slightly but the forecast from OTHER agents still created
too many collision constraints for the greedy rollout.

**Consumption attempt 3 (weak bias, -0.5 per matching step):**
No effect on grid maps. BFS distances are integers (1.0 apart); the 0.5
bonus is too weak to break ties that don't exist. LllgPi = LllgEmpty.

**Root cause:** The paper's warm-start is designed for **full space-time A***,
where it provides an initial bound for A* pruning. The greedy rollout doesn't
benefit from warm-start — it always picks the locally-best step, ignoring the
forecast. Occupancy-seeding makes the greedy rollout MORE conservative (avoiding
forecast cells), which hurts throughput. Weak bias is too weak to matter on
integer-distance grids.

**Decision:** Warm-start infrastructure is in place (stored, consumed one-shot)
but NOT consumed by the greedy rollout. The `compute_guidance` method clears
the data (preventing stale leaks) but doesn't seed the occupancy. Consumption
is deferred to the full A* upgrade. LllgPi and LllgEmpty produce identical
results with the greedy rollout.

---

## G1 — Throughput (correctness)

**800 agents, 300 steps, 4 maps.** (Updated Issue 142 — full space-time A*)

| Map | Our throughput | Paper target | Ratio | Verdict |
|---|---|---|---|---|
| empty-48-48 | 18.63 | 27.3 | 0.68 | **PASS** |
| random-64-64-10 | 14.57 | 21.1 | 0.69 | **PASS** |
| warehouse-10-20-10-2-2 | 6.99 | 18.0 | 0.39 | **PASS** |
| ht_chantry-approx | 0.15 | 17.0 | 0.01 | **FAIL** |

**Pass criterion:** ratio ≥ 0.30 (within reasonable range, system works).

**Improvement vs Issue 140 (greedy rollout):**
- empty: 0.63 → 0.68 (+7.6%)
- random: 0.62 → 0.69 (+10.8%)
- warehouse: 0.35 → 0.39 (+11.0%) — **crossed the 0.30 threshold**
- ht_chantry: 0.01 → 0.01 (marginal)

**Analysis:**
- 3/4 maps now PASS (was 2/4). The A* with BFS-distance heuristic and
  multi-round refinement improves throughput across the board.
- ht_chantry FAILS because the maze topology with narrow bottlenecks requires
  LaCAM-level search escalation. The w_Φ=5 window can't see far enough through
  the maze to plan detours. This is the paper's known limitation (caveat #1:
  long one-cell-wide corridors).

**Root cause of ht_chantry failure:** The maze structure creates long
one-cell-wide corridors where agents meet head-on. Without LaCAM escalation
(which can reorder agents globally), the greedy PIBT can't resolve these
conflicts. Priority inheritance alone doesn't help (Issue 140 showed it
*collapses* throughput without LaCAM). This is Phase 5 work.

---

## G2 — Congestion mitigation

**empty-48-48, 1000 agents, 100 steps. LLLG_Π vs LllgEmpty baseline.**
(Updated Issue 142)

| Scheme | max_stops/cell | mean_stops | throughput |
|---|---|---|---|
| LLLG_Π | 56 | 6.1 | 18.60 |
| LllgEmpty | 56 | 6.1 | 18.60 |

**Ratio: 1.00 (identical). FAIL.**

**Root cause:** Issue 142 confirmed (with the full A*) that occupancy-seeding
with warm-start forecasts HURTS throughput. The forecast is invalidated when
PIBT deviates from the guidance. The data is consumed (cleared) but NOT seeded
into the occupancy, so LllgPi = LllgEmpty.

**Fix required:** LaCAM escalation (Phase 5) to keep PIBT deviations rare, so
warm-start forecasts stay accurate. Alternatively, a different warm-start
consumption method (e.g. soft bias, f-bound pruning) might work but was not
found effective on integer-distance grids (Issue 140 attempt 3).

---

## G3 — No-regression

```
cargo clippy -p katgpt-core --all-features --lib: clean
cargo test -p katgpt-core --lib: 1556/1556 pass
cargo test -p katgpt-core --features multi_agent_path --lib: 22/22 pass
```

**PASS.** Existing tests unaffected.

---

## G4 — Latency

**empty-48-48, 1000 agents, 100 steps.** (Updated Issue 142)

| Metric | Value |
|---|---|
| Median per-tick | 87.83 ms |
| Max per-tick | 285.36 ms |
| Paper (M1 Ultra) | 210–260 ms |
| Issue 140 (greedy) | 82.17 ms |

**PASS (target < 500ms). Stretch < 100ms: PASS.**

The A* is ~7% slower than the greedy rollout (87.83ms vs 82.17ms) due to the
priority queue and hash map overhead, but still well within the stretch goal
and 2.4× faster than the paper's M1 Ultra result. The A* explores a narrow
cone around the BFS gradient thanks to the admissible heuristic.

---

## Honest caveats

1. **Maps are synthetic approximations.** We don't have the exact MovingAI
   MAPF benchmark map files for warehouse-10-20-10-2-2 and ht_chantry.

2. **PIBT priority inheritance requires LaCAM.** The recursive push is too
   conservative without LaCAM-level escalation. See Issue 140 investigation.

3. **Warm-start requires full A*.** The greedy rollout can't consume the
   warm-start forecast. Occupancy-seeding collapses throughput; weak bias is
   too weak on integer-distance grids.

4. **Agent count is 800 not 1000.** Paper throughput targets are at 1000
   agents. We run at 800.

5. **The ht_chantry approximation is more extreme than the real map.**

---

## What was accomplished (Issue 142)

1. **Full space-time A* landed.** Replaced the greedy rollout with a proper
   priority-queue A* over `(position, depth)` state space with BFS-distance
   heuristic. The A* has w-step lookahead and can plan multi-step detours.

2. **Multi-round refinement fixed.** The broken `clear_occupancy()`-each-round
   pattern (making rounds no-ops) replaced with unrecord/re-record: each agent
   removes its previous path before recomputing, so round 1+ actually improves
   on round 0.

3. **Throughput improved on all 4 maps.** 3/4 maps now pass G1 (was 2/4).
   Warehouse crossed the 0.30 threshold. ht_chantry remains at 0.01 (needs
   LaCAM).

4. **Warm-start consumption confirmed harmful.** Occupancy-seeding with
   warm-start forecasts was implemented, benchmarked, and found to HURT
   throughput (18.60 → 14.73 on empty-48-48). The forecast is invalidated when
   PIBT deviates from the guidance. Confirmed with the full A* that this is
   not a greedy-rollout-specific problem — it's a forecast-accuracy problem
   that likely needs LaCAM to fix.

---

## Next steps

1. **LaCAM escalation** (Phase 5) — when PIBT fails to resolve a conflict,
   escalate to LaCAM-level search. This is the single blocker for both ht_chantry
   (G1) and warm-start consumption (G2). With LaCAM:
   - ht_chantry: LaCAM resolves corridor conflicts that PIBT can't.
   - Warm-start: LaCAM keeps PIBT deviations rare, so forecasts stay accurate.
   - PIBT priority inheritance: recursive push works with LaCAM fallback.

2. **Download real MovingAI maps** — for exact paper reproduction. The
   synthetic approximations may differ from the real warehouse/ht_chantry
   topology.

3. **Phase 3** (Fusion hooks documentation) — document the four pluggable seams
   (`CostFn`, `LocalGuidanceSource`, `WarmStartScheme`, `HindranceEstimator`)
   with stub examples.

4. **riir-ai/489** (private runtime fusion) — can now consume `multi_agent_path`
   via the four seams. Works correctly on open/moderate maps (3/4 G1 maps pass);
   the consumer should be aware of ht_chantry/maze limitations.

The substrate is functional and well-performing on the 3/4 maps it handles.
The ht_chantry failure and G2 warm-start non-consumption both point to the same
fix: LaCAM escalation (Phase 5).

---

## Promotion Decision (Phase 5, T5.1/T5.2)

**Decision: KEEP OPT-IN.** Reaffirms the T2.6 decision recorded in the TL;DR above.

### T5.1 Considerations (all four weighed)

1. **Modelless?** ✅ Yes — the substrate is entirely heuristic (PIBT greedy
   selection + BFS distance field + warm-start suffix reuse + blocking-count
   hindrance). No training, no backprop, no gradient descent. **Promotion is
   *allowed* by AGENTS.md's modelless mandate** — the rule permits default-on
   for modelless gains.

2. **Heavy / leaf-clean?** ❌ Multi-agent pathfinding is **not** a leaf-clean
   primitive. The substrate is ~1000 LOC across 8 files (mod, config, pibt,
   local_guidance, warm_start, hindrance, position, tests) plus the bench
   harness. Consumers that don't need crowd pathfinding would pay the compile
   cost. Keeping it opt-in avoids bloating the default build — mirrors the
   `cgsp` (Plan 274) and `induced_cwm` (Plan 296) precedent for heavier
   substrates.

3. **GOAT gate status?** ❌ **Not fully passed.** G3 (no-regression) and G4
   (latency) PASS, but G1 (throughput) is only PARTIAL (2/4 maps) and G2
   (congestion) FAILS. The modelless mandate's promotion rule requires a
   *modelless gain* — a perf gain on a biased/incorrect result is explicitly
   NOT a modelless gain (AGENTS.md Feature Flag Discipline). G1's warehouse/maze
   failures mean the substrate produces measurably wrong answers on those map
   classes; promoting a 2/4-correct primitive to default-on would violate the
   quality-gate rule even though the primitive itself is modelless.

4. **Super-GOAT claim validated?** ❌ **No.** The Super-GOAT selling point
   rests on the riir-ai/489 fusion gates G5–G7 (HLA projection per-NPC
   personality modulation, Crowd MCGS physical layer, P350 multi-agent
   closure). Those gates have **not run yet**. Promoting the substrate to
   default before the fusion is validated is premature — if G5–G7 fail, the
   substrate stays a standalone pathfinder with no Super-GOAT upside, and the
   default-on promotion would have been wasted build cost.

### Rationale (why opt-in is correct now)

The substrate is shipped, documented (Phase 3 fusion hooks with compile-checked
rustdoc examples), and available to any consumer that wants it via the
`multi_agent_path` feature flag. Promotion to default-on is deferred until
**both** of these hold:

- **G1/G2 unblocked** — the Phase 5 full space-time A* upgrade (replacing the
  greedy rollout) is the single change that unblocks both gates: A* benefits
  from warm-start (G2) and produces better paths through warehouse/maze
  corridors (G1). The latency budget has ample headroom (82ms current vs 500ms
  target).
- **Super-GOAT validated** — riir-ai/489 G5–G7 pass, confirming the HLA ×
  Crowd MCGS × P350 fusion produces emergent crowd coordination beyond what
  either primitive alone achieves.

Until then, the substrate is opt-in and the Super-GOAT claim is conditional.
