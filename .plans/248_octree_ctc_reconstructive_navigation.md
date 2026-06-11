# Plan 248: OctreeCTC — Reconstructive Memory Navigation

**Date:** 2026-06-11
**Research:** `.research/216_MRAgent_Reconstructive_Memory_Graph.md`
**GOAT Status:** ⏳ Pending GOAT proof
**Feature Gate:** `octree_ctc` (default-OFF until GOAT proof)
**Depends On:** Existing `SenseModule`, `NpcBrain`, `SenseBandit` (all ✅ complete)

---

## Goal

Implement multi-step active reconstruction over KG-Latent-Octree, replacing single-shot `NpcBrain::project()` with iterative HLA-state-aware navigation. Modelless: entropy bandit + dot-product + sigmoid, no LLM.

## Tasks

### Phase 1: Core Types
- [ ] Create `ReconstructionState` struct in `katgpt-core/src/sense/reconstruction.rs`
  - `hla: [f32; 8]` — evolving HLA state
  - `active_nodes: ArrayVec<OctreeNodeId, 8>` — Z(t) active set
  - `accumulated: Vec<KgTriple>` — H(t) evidence
  - `step: u8` — current traversal step
  - `max_steps: u8` — budget (default 3)
  - `entropy_threshold: f32` — early stop threshold
- [ ] Create `OctreeNodeId` newtype (`u32` morton code)
- [ ] Create `TraversalAction` enum: `Forward { tag_idx: u8 }`, `Reverse { content_idx: u8 }`, `Halt`
- [ ] Create `ReconstructionConfig` with serde support
  - `max_steps: u8` (default 3)
  - `hla_learning_rate: f32` (default 0.1)
  - `entropy_threshold: f32` (default 0.05)
  - `lod_adaptive: bool` (default true)

### Phase 2: Reconstruction Loop
- [ ] Implement `ReconstructionState::expand()` — traverse octree children from active nodes
- [ ] Implement `ReconstructionState::route()` — entropy-gated bandit selection (reuse `SenseBandit`)
- [ ] Implement `ReconstructionState::accumulate()` — collect KG triples from selected content
- [ ] Implement `ReconstructionState::evolve_hla()` — bridge function: accumulated triples → HLA update
  - Must be: dot-product projection + sigmoid (per AGENTS.md)
  - Must be: zero-allocation, clamp to valid range
  - Max delta per step bounded by `hla_learning_rate`
- [ ] Implement `ReconstructionState::sufficient()` — entropy-based early stopping
- [ ] Implement `ReconstructionState::reconstruct()` — main loop combining above methods

### Phase 3: NpcBrain Integration
- [ ] Add `reconstruct()` method to `NpcBrain` (behind `octree_ctc` feature gate)
- [ ] Existing `project_all()` remains default behavior (backward compat)
- [ ] Add `project_reconstruct()` that uses `ReconstructionState` internally
- [ ] Wire `SenseBandit` trial logging for reconstruction steps

### Phase 4: SIMD Optimization
- [ ] Batch `expand()` across multiple active nodes using SIMD
- [ ] Batch `evolve_hla()` dot-product using existing SIMD infrastructure
- [ ] Benchmark: ensure <200ns per reconstruction cycle (3 steps)

### Phase 5: GOAT Proof
- [ ] Create `examples/octree_ctc_demo.rs` showing before/after:
  - Before: `NpcBrain::project_all()` single-shot
  - After: `NpcBrain::project_reconstruct()` multi-step
  - Metric: KG triple recall (ground truth vs recovered)
- [ ] Create `tests/octree_ctc_recall_test.rs`:
  - Multi-hop query: "Which enemies are near ally X?" (requires 2-hop traversal)
  - Measure recall improvement ≥ 20% vs passive projection
- [ ] Run benchmark: latency per tick < 200ns for 3-step reconstruction
- [ ] If GOAT passes → promote to default feature
- [ ] If GOAT fails → demote, document why, keep as opt-in

### Phase 6: CPU/GPU Auto-Route
- [ ] Add reconstruction budget threshold: if latency > 500ns, reduce max_steps
- [ ] Add SIMD/SISD path selection based on active node count
- [ ] Add ANE consideration: reconstruction maps well to Neural Engine matrix ops

---

## Architecture Decision Records

### ADR-1: Why Not LLM Routing?
MRAgent uses LLM for `f_select` and `f_route`. We cannot — game tick budget is 16ms, LLM call is 100ms+. Entropy-gated bandit provides deterministic, sub-microsecond routing that converges from `SenseBandit` trials.

### ADR-2: Why max_steps=3?
MRAgent shows diminishing returns after 3-4 turns (Figure 6a). Single-hop/temporal queries converge by turn 3. Multi-hop needs 3-5. Default 3 balances recall vs latency. Configurable via `ReconstructionConfig`.

### ADR-3: HLA Evolution Stability
HLA state update is clamped: `hla[i] = clamp(hla[i] + lr * delta, -1.0, 1.0)`. Sigmoid bridge ensures bounded output. No softmax. Per AGENTS.md latent→raw bridge rules.

---

## File Map

```
katgpt-core/src/
├── sense/
│   ├── reconstruction.rs    ← NEW: ReconstructionState + loop
│   ├── brain.rs              ← MODIFIED: add project_reconstruct()
│   ├── bandit.rs             ← REUSED: entropy-gated selection
│   └── ...
├── types.rs                  ← MODIFIED: OctreeNodeId, TraversalAction
└── ...

katgpt-rs/
├── examples/
│   └── octree_ctc_demo.rs    ← NEW: before/after demo
├── tests/
│   └── octree_ctc_recall_test.rs ← NEW: GOAT proof test
└── ...
```

---

## TL;DR

Add iterative HLA-evolving reconstruction to `NpcBrain` behind `octree_ctc` feature gate. 6 phases: types → loop → integration → SIMD → GOAT proof → auto-route. Target: 25%+ multi-hop recall improvement, <200ns latency. If GOAT passes, promote to default.
