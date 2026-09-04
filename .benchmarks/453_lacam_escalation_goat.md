# Benchmark 453: Bounded One-Step LaCAM Escalation GOAT Gate

**Date:** 2026-07-15
**Plan:** [453](../.plans/453_bounded_one_step_lacam_escalation.md)
**Research:** [441](../.research/441_lacam_constraint_tree_distillation.md)
**Feature:** `lacam_escalation = ["multi_agent_path"]`
**Bench:** `crates/katgpt-core/benches/bench_453_lacam_escalation_goat.rs` (G6c + latency sweep)
**Bench:** `crates/katgpt-core/benches/bench_440_lllg_paper_repro.rs` (G1 throughput, re-run with `--features lacam_escalation`)

## TL;DR

The LaCAM constraint tree **passes both collision-freedom gates (G6c, G-col)
perfectly** and dramatically improves ht_chantry throughput (0.01 → 0.28, 28×).
However, G1 throughput still marginally FAILS on ht_chantry (0.28 < 0.30)
because the constraint tree resolves single-step collisions but cannot resolve
multi-step corridor deadlocks. The G-PI gate (no throughput collapse) PASSES —
the constraint tree makes recursive PIBT safe, confirming the Plan 453 hypothesis.

**Overall: G6c ✅ PASS, G-col ✅ PASS, G-PI ✅ PASS, G1 ⚠ 3/4 maps PASS (ht_chantry marginal), G4 ✅ PASS.**

Promotion decision: **defer** — the constraint tree is a genuine improvement
(collision-freedom + ht_chantry throughput), but G1 needs multi-step LaCAM to
fully pass. The `lacam_escalation` feature stays opt-in. See Phase 5 §T5.3.

---

## Critical Fix During Benchmarking: The Threshold Gate

**Root cause discovered during T3.1:** the initial G6c run showed 90% vertex
collisions (G6c = 0.100, FAIL). Investigation revealed the `pibt_step` function
uses `MIN_STUCK_FOR_RETRY = 20` to gate the escalation — meaning the LaCAM
constraint tree only fired when 20+ agents were stuck simultaneously. With 60
agents in a bottleneck, most collision ticks had fewer than 20 stuck agents, so
the constraint tree never activated.

**Fix:** added a cfg-gated override in `pibt_step` that lowers the threshold to
1 when `lacam_escalation` is ON. The constraint tree is designed to resolve
even a single stuck agent collision-free via recursive PIBT — unlike the legacy
shuffled retry which needed 20+ stuck agents to justify its overhead. The greedy
fast path (zero stuck agents) is unaffected, preserving open-map throughput.

**After the fix:** G6c = 1.000 (100% collision-free), G-col vertex rate = 0.0%.

This is documented in `pibt.rs` lines 263-289.

---

## G6c — Collision-freedom delta

**60 agents, 20×20 grid, 6-cell bottleneck gap, 200 ticks.**

Bench: `bench_453_lacam_escalation_goat.rs`

| Variant | Collision-free | Vertex coll | Edge coll | First coll tick |
|---|---|---|---|---|
| **LaCAM (constraint tree + recursive PIBT)** | **200/200 (100.0%)** | **0/200 (0.0%)** | **0/200 (0.0%)** | **None** |
| Naive (no planner — Stern et al. 2019) | 0/200 (0.0%) | 200/200 (100.0%) | 198/200 (99.0%) | 0 |

**G6c delta = 1.000 − 0.000 = 1.000 → PASS (≥ 0.50).**

The constraint tree achieves perfect collision-freedom on the bottleneck
scenario. Every tick produces a collision-free joint action — zero vertex
collisions, zero edge collisions, across all 200 ticks. This is the LaCAM
guarantee working as designed: the constraint tree + recursive PIBT resolves
every stuck agent via priority inheritance, without the all-wait fallback.

**Comparison to pre-fix baseline (riir-ai Issue 516, greedy PIBT without
constraint tree):** 37.5% collision-free (G6c = 0.360). The constraint tree
improves collision-freedom from 37.5% → 100.0% — a 62.5pp improvement.

---

## G-col — Vertex collision rate (NEW gate)

**Same G6c scenario.**

| Metric | Value | Target | Verdict |
|---|---|---|---|
| Vertex collision rate | 0.0% | ≤ 10% | **PASS** |

The constraint tree eliminates vertex collisions entirely on this scenario.
The recursive PIBT's priority inheritance mechanism ensures no two agents ever
occupy the same cell.

---

## G1 — Throughput (800 agents, 4 maps, 500 steps)

Bench: `bench_440_lllg_paper_repro.rs` compiled with `--features lacam_escalation`

| Map | LaCAM ON throughput | LaCAM OFF throughput | Paper | ON ratio | OFF ratio | Verdict |
|---|---|---|---|---|---|---|
| empty-48-48 | 18.75 | 18.63 | 27.3 | **0.69** | 0.68 | **PASS** |
| random-64-64-10 | 14.54 | 14.57 | 21.1 | **0.69** | 0.69 | **PASS** |
| warehouse-10-20-10-2-2 | 7.27 | 6.99 | 18.0 | **0.40** | 0.39 | **PASS** |
| ht_chantry | 4.80 | 0.15 | 17.0 | **0.28** | 0.01 | **FAIL** |

**Pass criterion:** ratio ≥ 0.30.

**Key findings:**

1. **ht_chantry dramatically improved:** 0.01 → 0.28 (28× improvement, +17pp
   ratio). The constraint tree's recursive PIBT unblocks corridor deadlocks
   that greedy PIBT cannot resolve. The throughput went from "system broken"
   to "system works, just not optimal."

2. **ht_chantry still FAILS the 0.30 threshold** (0.28 < 0.30, marginal). The
   one-step constraint tree resolves single-tick collisions but cannot plan
   multi-step detours through the maze. This is the expected limitation
   documented in Plan 453 §4.3 — full multi-step LaCAM is needed for maze
   throughput parity.

3. **Open maps unchanged:** empty-48-48 and random-64-64-10 are statistically
   identical (±0.01). The constraint tree fires only when stuck agents exist;
   on open maps the greedy fast path handles 99%+ of ticks. This confirms G-PI.

4. **warehouse marginally improved:** 0.39 → 0.40 (+0.01). The warehouse's
   2-wide corridors create some stuck agents that the constraint tree resolves,
   but the improvement is small.

**G1 verdict:** empty (0.69 ✅), random (0.69 ✅), warehouse (0.40 ✅),
ht_chantry (0.28 ❌). **3/4 maps PASS.** ht_chantry is marginal (0.28 vs the
0.30 threshold — just 2pp under).

---

## G-PI — No throughput collapse (the Issue 140/143 guard)

**The critical gate: does the constraint tree prevent the throughput collapse
that recursive PIBT caused in Issues 140/143?**

| Metric | Issue 140 (recursive PI, no constraint tree) | Plan 453 (recursive PI + constraint tree) | Target |
|---|---|---|---|
| empty-48-48 ratio | 0.02 (collapsed) | **0.69** | ≥ 0.60 |

**G-PI: PASS.** The constraint tree makes recursive PIBT safe. Throughput on
open maps is preserved (0.69 vs the Issue 140 collapse to 0.02). This confirms
the Plan 453 thesis: the constraint tree is the "missing half" that makes
recursive PI viable.

---

## G4 — Latency

**60 agents (G6c scenario):**

| max_nodes | Median (µs) | Max (µs) | Collision-free % |
|---|---|---|---|
| 100 | 39.9 | 62.7 | 100.0% |
| 500 | 39.6 | 68.8 | 100.0% |
| 1000 | 39.9 | 57.8 | 100.0% |
| 5000 | 39.0 | 67.2 | 100.0% |

**800 agents (G1 throughput benchmark):**

| Map | Median (ms) | Max (ms) |
|---|---|---|
| empty-48-48 | 14.33 | 204.78 |
| random-64-64-10 | 16.15 | 367.41 |
| warehouse | 17.56 | 898.58 |
| ht_chantry | 18.83 | 752.16 |

**G4 verdict: PASS.** Median per-tick ≤ 500ms on all maps (stretch goal ≤ 100ms
met on all but ht_chantry at 18.83ms — well under 100ms). The constraint tree
adds minimal overhead on open maps (14ms median at 800 agents) and acceptable
overhead on congested maps (~18ms median).

**Latency sweep finding:** the median latency is flat across all `max_nodes`
values (~40µs at 60 agents). This means the constraint tree finds a
collision-free config quickly (within the first few nodes) and rarely exhausts
its budget. The `max_nodes` parameter has negligible effect on latency at this
scale — the constraint tree converges fast.

---

## G3 — No-regression

```
cargo test -p katgpt-core --features lacam_escalation --lib:  1616/1616 pass
cargo test -p katgpt-core --features multi_agent_path --lib:  1611/1611 pass
cargo clippy -p katgpt-core --features lacam_escalation:       clean
cargo clippy -p katgpt-core --features multi_agent_path:       clean
```

**PASS.** No regressions. The threshold fix (MIN_STUCK_FOR_LACAM_GATE = 1 when
ON, = 20 when OFF) preserves the legacy behavior exactly when the feature is OFF.

---

## Honest caveats

1. **ht_chantry is marginal (0.28 vs 0.30).** The constraint tree improved it
   28× (from 0.01), but it's still 2pp under the threshold. This is the
   expected one-step limitation — multi-step LaCAM would close this gap.

2. **The latency sweep at 60 agents doesn't show budget sensitivity.** The
   constraint tree converges within the first few nodes at this scale. A sweep
   at 800 agents might show different behavior, but that benchmark is
   prohibitively slow to run 4× (each G1 run takes ~5 minutes).

3. **The max per-tick latency at 800 agents is high** (up to 898ms on
   warehouse). This is from the constraint tree firing on a congested tick
   where many agents are stuck. The default `time_budget_us = 5000` should cap
   this, but the budget check every 64 nodes means a single burst can exceed
   the budget before the check fires. This is a latency-tail risk, not a median
   risk.

4. **The threshold fix (1 vs 20) is aggressive.** Lowering the gate to 1 means
   the constraint tree fires on every tick with any stuck agent. On open maps
   this is fine (few stuck agents, constraint tree resolves quickly), but on
   highly congested maps it could add overhead. The G1 data shows this overhead
   is acceptable (median 14-19ms at 800 agents), but it should be monitored.

---

## GOAT Gate Summary

| Gate | Component | Threshold | Result | Verdict |
|---|---|---|---|---|
| **G6c** | Collision-freedom delta | ≥ 0.50 | **1.000** | ✅ PASS |
| **G-col** | Vertex collision rate | ≤ 10% | **0.0%** | ✅ PASS |
| **G1** | Throughput (4 maps ≥ 0.30) | ≥ 0.30 each | empty 0.69, random 0.69, warehouse 0.40, ht_chantry 0.28 | ⚠ 3/4 PASS (ht_chantry marginal) |
| **G-PI** | No throughput collapse | ≥ 0.60 on empty | **0.69** | ✅ PASS |
| **G3** | No-regression | all tests + clippy | 1616/1616 + clean | ✅ PASS |
| **G4** | Latency | median ≤ 500ms | 14-19ms at 800 agents | ✅ PASS |

---

## Phase 5 Promotion Decision

**T5.3 applies:** G1 FAILS on ht_chantry (0.28 < 0.30, marginal). LaCAM
one-step can resolve single-tick collisions but cannot plan multi-step detours
through the maze. The collision-freedom improvement (G6c, G-col) stands on its
own — the constraint tree genuinely fixes the vertex collision problem.

**Decision: stay opt-in.** The `lacam_escalation` feature is a genuine
improvement over the legacy shuffled retry:
- Collision-freedom: 37.5% → 100% (G6c scenario)
- ht_chantry throughput: 0.01 → 0.28 (28×)
- No throughput collapse (G-PI PASS)

But it doesn't meet the full G1 gate. Defer to a future multi-step LaCAM plan
(the full high-level configuration search) for ht_chantry throughput parity.
The one-step constraint tree is the correct foundation — multi-step LaCAM
builds on top of it.

**Issue 154 status:** RESOLVED — the vertex collision problem is eliminated
by the constraint tree (G-col = 0.0%). Issue 154 is closed as "fixed by
Plan 453" (the issue file was removed per the noise-reduction rule; the
resolution is recorded in `.plans/453_*` and this benchmark). The remaining
throughput gap is a different problem (multi-step planning, not collision-
freedom) and is deferred to a future multi-step LaCAM plan.

## Reproduction

```bash
# G6c + latency sweep (this bench)
CARGO_TARGET_DIR=/tmp/453_phase3 cargo bench -p katgpt-core \
    --features lacam_escalation --bench bench_453_lacam_escalation_goat -- --nocapture

# G1 throughput (bench_440, compiled with lacam_escalation ON)
CARGO_TARGET_DIR=/tmp/453_phase3 cargo bench -p katgpt-core \
    --features lacam_escalation --bench bench_440_lllg_paper_repro -- --nocapture

# Tests + clippy (G3)
CARGO_TARGET_DIR=/tmp/453_phase3 cargo test -p katgpt-core --features lacam_escalation --lib
CARGO_TARGET_DIR=/tmp/453_phase3 cargo clippy -p katgpt-core --features lacam_escalation --lib
```

## TL;DR

G6c = 1.000 (PASS), G-col = 0.0% (PASS), G-PI = 0.69 (PASS), G1 = 3/4 maps
(ht_chantry marginal at 0.28), G4 = 14-19ms median (PASS). The constraint tree
fixes collisions perfectly and improves ht_chantry 28×, but ht_chantry still
marginally fails G1 (0.28 < 0.30) because one-step LaCAM can't plan multi-step
maze detours. Stay opt-in; defer to multi-step LaCAM for full G1 parity.

---

## Addendum: Multi-Step LaCAM + Flow-Field Attempts (Issue 546, riir-ai)

**Status: BOTH paths SHIPPED-with-FAIL or REVERTED. ht_chantry G1 gap (0.27-0.28
vs 0.30 target) is the accepted honest steady-state floor for the current
architecture.** Recorded here 2026-07-19 from the now-removed
`riir-ai/.issues/546_lacam_multistep_escalation_ht_chantry.md` per the
noise-reduction rule.

### Diagnostic (commit `2a8c378d`)

`katgpt-rs/crates/katgpt-core/examples/ht_chantry_deadlock_chain_diagnostic.rs`
measured the per-tick max-cluster-size distribution on ht_chantry-real (162×141,
7461 passable, 800 agents, 500 steps, seed=42, `lacam_escalation` ON):

| Metric | Value |
|---|---|
| Throughput (this seed) | 5.098 (~0.30 ratio, consistent with bench's 4.80) |
| Fast-path ticks (zero stuck) | 0/500 (0.0%) — ht_chantry is systemically congested |
| P95 max-cluster-size | **8** |
| P99 max-cluster-size | **9** |
| Max observed | **11** |
| Depth-2 coverage | 12.8% of stuck ticks |
| Depth-3 coverage | 36.4% of stuck ticks |

Paper suggests depth 2-3 is "usually sufficient" for multi-step LaCAM. On
ht_chantry that covers only 36% of stuck ticks. Covering 95% needs depth ≥ 8
— computationally intractable (combinatorial blow-up) outside the real-time
MAPF budget.

### Path 1: Multi-step LaCAM (SHIPPED, +0.6% marginal)

Shipped behind `EscalationBudget::multistep_default()` (max_depth = 8) +
`LifelongLaCam::with_escalation_budget()`. Key innovation vs paper-faithful:
`target_stuck_agents: bool` — constraint tree iterates over STUCK agents
(computed by greedy PIBT) instead of all agents in priority order. At depth K,
constrain stuck agent `stuck_order[K]` to a neighbor cell. T1-T6 of the
reopened plan all shipped; T7 (this document update) replaces the separate
benchmark file.

A/B measurement (`ht_chantry_multistep_ab`, ht_chantry-real, 800 agents, 200
steps, seed=42):

| Metric | Default (Plan 453) | Multistep (Issue 546) | Delta |
|---|---|---|---|
| Throughput | **4.49** | **4.51** | +0.025 (+0.6%) |
| Completions | 898 | 903 | +5 |
| Median tick | 17.68ms | 18.13ms | +0.45ms (+2.5%) |
| G1 (≥5.10) | FAIL | FAIL | unchanged |

**Verdict: multistep ships with marginal improvement.** The +0.6% throughput
is real (5 deadlocks resolved that weren't before), but does not materially
close the G1 gap. The diagnostic was correct in mechanism but overstated the
impact — even with stuck-agent targeting and depth-8, the fundamental corridor-
queue structure on ht_chantry cannot be unwound by constraint-tree search alone.
No revert — the work is useful for combined-strategy experiments and future
maze-class maps.

### Path 2: Flow-field hard constraint (REVERTED, Proposal 006 REJECTED)

Attempted via [Proposal 006](../.proposals/006_flow_field_hard_constraint_in_guidance.md)
+ `Issue 182`. Phases
1-3 (bi-directional corridors + A\* hard pruner + cost-tuple demotion) were
implemented and measured.

Result:
- ht_chantry throughput: **+0.9% (noise — flat)** — FAIL
- ht_chantry deadlock-chain P95: **8 → 15 (WORSE)** — FAIL
- warehouse throughput: +7% (mechanism DID work on warehouse but insufficient)

Per Issue 182 Phase 5 P5.3, code was REVERTED. Proposal 006 marked REJECTED.
See Proposal 006 §Verdict for the full measurement table and analysis.

### Bottom line

The ht_chantry G1 gap (0.27-0.28 vs 0.30 target) is a **documented steady-state
fail**. Both attempted closures (multi-step LaCAM, flow-field redesign) have
been measured and found insufficient. The gap is accepted as the honest floor
for the current architecture. A fundamentally different approach (e.g. dynamic
flow reversal, global LP-based direction assignment, or a different planner
class entirely) would be needed to close it — none are in scope today.

**Feature status:** `lacam_escalation` (one-step, Plan 453) stays opt-in. The
multi-step extension (`EscalationBudget::multistep_default()`) is available
for combined-strategy experiments but provides no G1 gain alone. The
flow-field hard constraint was reverted (Proposal 006 REJECTED).
