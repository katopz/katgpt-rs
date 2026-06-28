# Plan 248: OctreeCTC — Reconstructive Memory Navigation

> **📍 Migration note (2026-06-28, Issue 007 Phase C follow-up):** The
> example + test referenced below (`examples/octree_ctc_demo.rs`,
> `tests/octree_ctc_recall_test.rs`) moved from this repo (katgpt-rs) to
> `riir-ai/crates/riir-engine/`. They construct `NpcBrain` which is now
> private NPC runtime IP. The reconstruction substrate they consume
> (`katgpt_core::sense::reconstruction`) stays public. The bench referenced
> below (`crates/katgpt-core/benches/reconstruction_bench.rs`) ALSO moved to
> `riir-engine/benches/reconstruction_bench.rs`. Historical task records
> below reflect the original locations.

**Date:** 2026-06-11
**Research:** `.research/216_MRAgent_Reconstructive_Memory_Graph.md`
**Status:** ✅ COMPLETE — all 6 phases done, promoted to default
**Feature Gate:** `octree_ctc` → `sense_composition` (default-ON since GOAT PASS)
**Reference:** arXiv:2606.06036
**Depends On:** Existing `SenseModule`, `NpcBrain`, `SenseBandit` (all ✅ complete)

---

## Goal

Implement multi-step active reconstruction over KG-Latent-Octree, replacing single-shot `NpcBrain::project()` with iterative HLA-state-aware navigation. Modelless: entropy bandit + dot-product + sigmoid, no LLM.

## GOAT Result

**PASS** — promoted to default feature.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Multi-hop recall improvement | ≥ 20% | ≥ 20% (5 steps, lr=0.3) | ✅ PASS |
| Latency per 3-step cycle | < 200ns | 93.2ns (scalar) | ✅ PASS |
| Per-step expand (matvec) | — | 2.1ns (3.6× vs scalar 8.9ns) | ✅ WIN |
| Tests | all pass | 18/18 | ✅ PASS |

5/5 GOAT tests: `single_hop_recall_improvement`, `multi_hop_recall_improvement`, `recall_threshold_met`, `reconstruction_converges`, `hla_stays_bounded`.

---

## Tasks

### Phase 1: Core Types — DONE (commit e8f05926)
- [x] `ReconstructionState` struct: `hla: [f32; 8]`, `active_nodes`, `evidence: TripleEvidence`, `step: u8`, config
- [x] `OctreeNodeId` newtype (`u32` morton code) with depth/parent/child
- [x] `TraversalAction` enum: `Forward { tag_idx }`, `Reverse { content_idx }`, `Halt`
- [x] `ReconstructionConfig`: `max_steps` (3), `hla_learning_rate` (0.1), `entropy_threshold` (0.05), `lod_adaptive` (true), `max_hla_delta` (0.3)

### Phase 2: Reconstruction Loop — DONE (commit e8f05926)
- [x] `expand()` — project all modules with current HLA
- [x] `route()` — entropy-gated selection above mean activation
- [x] `accumulate()` — collect KG triples from selected modules
- [x] `evolve_hla()` — bridge function: dot-product + sigmoid, clamp `[-1, 1]`, zero-alloc
- [x] `sufficient()` — entropy-based early stopping
- [x] `reconstruct()` — main loop

### Phase 3: NpcBrain Integration — DONE
- [x] `NpcBrain::reconstruct()` behind `sense_composition` feature
- [x] `project_all()` remains default (backward compat)
- [x] `SenseModule::project_reconstruction()` wrapper
- [x] `NpcBrain::project_reconstruct()` with `ReconstructionState`
- [x] `SenseBandit` trial logging wired

### Phase 4: SIMD Optimization — DONE (commits aeb41fa6, 8f476361, b39a434c)
- [x] `expand_simd()` — vectorized ternary projection (scalar faster at 6×8, available for scaling)
- [x] `evolve_hla_simd()` — SIMD fused sub-scale (net-negative at 8 elements, scalar auto-unroll wins)
- [x] `ProjectionWeights` — pre-computed `[6×8]` weight matrix from ternary dirs, compute once per brain config
- [x] `expand_with_weights()` — single `simd_matmul_rows` replaces 6× `module.project()` (**3.6× faster**)
- [x] `reconstruct_with_weights()` — production path with pre-computed weights
- [x] `BatchProjectionWeights` + `expand_batch()` — multi-entity batch API
- [x] Benchmark: <200ns ✅ (93.2ns scalar)

**Benchmark (NEON, Apple Silicon):**

| Path | Full cycle | Per-step expand |
|------|-----------|-----------------|
| Scalar `reconstruct()` | **93.2 ns** | 8.9 ns |
| SIMD `reconstruct_simd()` | 107.5 ns | 8.9 ns |
| Matvec `reconstruct_with_weights()` | 111.8 ns | **2.1 ns** |
| Multi-entity matvec (N=1..32) | — | **2.1 ns** per entity |

**GOAT verdict:** Scalar wins full cycle (LLVM tight-loop). Matvec wins expand (3.6×). Production: cache `ProjectionWeights` per brain config.

### Phase 5: GOAT Proof — DONE
- [x] `examples/octree_ctc_demo.rs` — before/after demo
- [x] `tests/octree_ctc_recall_test.rs` — multi-hop recall ≥ 20%
- [x] Benchmark: 93.2ns < 200ns ✅
- [x] GOAT PASS → promoted to default feature

### Phase 6: Auto-Route — DONE
- [x] `ReconstructionConfig::with_adaptive_budget()` — reduces `max_steps` if >500ns
- [x] `simd_beneficial()` — checks SIMD level + workload size
- [x] `reconstruct_auto()` — auto-selects scalar vs SIMD path
- [x] ANE consideration: `expand_with_weights()` is hook point for Metal backend

---

## Architecture Decision Records

### ADR-1: Why Not LLM Routing?
MRAgent uses LLM for `f_select` and `f_route`. We cannot — game tick budget is 16ms, LLM call is 100ms+. Entropy-gated bandit provides deterministic, sub-microsecond routing.

### ADR-2: Why max_steps=3?
MRAgent shows diminishing returns after 3-4 turns (Figure 6a). Default 3 balances recall vs latency. Configurable via `ReconstructionConfig`.

### ADR-3: HLA Evolution Stability
`hla[i] = clamp(hla[i] + lr * delta, -1.0, 1.0)`. Sigmoid bridge, no softmax. Per AGENTS.md latent→raw rules.

### ADR-4: SIMD vs Scalar at 8 Elements
NEON setup overhead (vld1q + vfmaq + vaddvq ≈ 5ns) exceeds compute savings for 8 f32 ops (≈1ns scalar auto-unrolled by LLVM). Scalar wins at 6 modules × 8 dim. Pre-computed matvec (`expand_with_weights`) wins by amortizing setup across all 6 modules in one call.

---

## File Map

```
katgpt-core/src/
├── sense/
│   ├── reconstruction.rs    ← NEW: ReconstructionState, ProjectionWeights, BatchProjectionWeights
│   ├── brain.rs              ← MODIFIED: project_reconstruct(), reconstruct_into()
│   ├── bandit.rs             ← REUSED: entropy-gated selection
│   └── ...
├── types.rs                  ← MODIFIED: OctreeNodeId, TraversalAction
└── ...

katgpt-rs/
├── crates/katgpt-core/benches/
│   └── reconstruction_bench.rs  ← NEW: full cycle, per-step, multi-entity batch
├── examples/
│   └── octree_ctc_demo.rs       ← NEW: before/after demo
├── tests/
│   └── octree_ctc_recall_test.rs ← NEW: GOAT proof (5 tests)
└── ...
```

---

## TL;DR

Iterative HLA-evolving reconstruction for `NpcBrain`. 6 phases complete. GOAT PASS (≥20% recall, 93.2ns < 200ns). Promoted to default feature. Production path: cache `ProjectionWeights` per brain config, use `reconstruct_with_weights()` per entity for 3.6× faster expand.
