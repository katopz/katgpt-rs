# Plan 251: DEC Operators & Cell Complex Inference Infrastructure

**Date:** 2026-06
**Status:** 🟢 ACTIVE
**Research:** 219 (Topological Neural Operators → DEC Inference)
**Depends On:** None (foundational)
**Blocks:** riir-ai Plan 277 (DEC LoRA Training)

---

## Overview

Implement Discrete Exterior Calculus (DEC) operators on cell complexes as zero-alloc, SIMD-accelerated inference-time operators for game spatial reasoning. This replaces ad-hoc gradient/flow computations with structured, conservation-guaranteed alternatives based on the TNO paper (arXiv:2606.09806).

The key insight: topology determines WHERE information flows (fixed), learning determines HOW features are mixed. For modelless inference, we only need the fixed part.

---

## Tasks

### Phase 1: Core Cell Complex & Cochain Types

- [x] T1: Create `src/dec/` module with feature gate `dec_operators`
- [x] T2: Implement `CellComplex` struct (vertices, edges, faces, volumes with incidence)
- [x] T3: Implement `CochainField` typed cochain on cell complex
- [x] T4: Implement `BoundaryMatrix` (sparse signed incidence matrix Bₖ as triplets)
- [x] T5: Write tests: cell complex construction, cochain assignment, incidence correctness
- [x] T6: Verify `BₖBₖ₊₁ = 0` (boundary-of-boundary is zero) for all test complexes

### Phase 2: DEC Operators (d, δ, Δ)

- [x] T7: Implement `exterior_derivative(rank)` → dₖ = Bₖ₊₁ᵀ (sparse matmul)
- [x] T8: Implement `codifferential(rank)` → δₖ = Mₖ₋₁⁻¹ Bₖ Mₖ (identity Hodge star)
- [x] T9: Implement `hodge_laplacian(rank)` → Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ
- [ ] T10: Implement `hodge_star(rank)` → Mₖ (metric/mass matrix) — deferred: uniform grid uses identity
- [ ] T11: Add SIMD acceleration for sparse matrix-vector multiply (DEC ops)
- [x] T12: Write tests: gradient of constant = 0, curl of gradient = 0, div of curl = 0
- [x] T13: Implement optimized `graph_laplacian()` with scratch buffer (zero intermediate alloc)

### Phase 3: Hodge Decomposition

- [x] T14: Implement `hodge_decompose()` → (exact, harmonic, coexact) projection
- [x] T15: Implement `betti_numbers()` → count zero eigenvalues of Δₖ
- [x] T16: Implement `harmonic_projector()` → P_harm = projection onto ker(Δₖ)
- [ ] T17: Extend `spectral_hierarchy.rs` with Hodge spectrum computation
- [x] T18: Write tests: decomposition reconstructs original, components orthogonal
- [ ] T19: Write benchmark: Hodge decomposition on game-sized maps (256×256)

### Phase 4: Game Integration (DecFlowField)

- [x] T20: Implement `DecFlowField` — Hodge-decomposed navigation field
- [x] T21: Implement exact channel navigation (gradient of distance potential)
- [x] T22: Implement coexact channel navigation (patrol/circulation behavior)
- [x] T23: Implement harmonic channel navigation (topologically guaranteed routes)
- [x] T24: Bridge `DecFlowField` → existing `FlowField` API for backward compat
- [x] T25: Write arena proof: DEC flow vs naive gradient on Bomber map
- [x] T26: Write benchmark: DEC navigation vs LeoPotentialGrid::gradient()

### Phase 5: DEC Pruner Features & Dirichlet Extension

- [x] T27: Extend `dirichlet.rs` with Hodge energy computation
- [x] T28: Add DEC-based features to `ScreeningPruner::relevance()` 
- [x] T29: Add `HodgeResidual` pruner signal (constraint satisfaction metric)
- [x] T30: Write GOAT gate: `dec_operators` feature flag, A/B test vs naive

### Phase 6: CPU/SIMD/GPU Auto-Route

- [ ] T31: Implement adaptive backend selection (CPU/SIMD/GPU) based on cochain size
- [ ] T32: Add threshold-based routing: n < 1K → CPU, 1K-10K → SIMD, >10K → GPU
- [ ] T33: Write benchmark: backend selection overhead vs compute savings

---

## Architecture

```
src/dec/
├── mod.rs              — Module root, feature gate
├── types.rs            — CellComplex, CochainField, BoundaryMatrix
├── operators.rs        — dₖ, δₖ, Δₖ, Hodge star Mₖ
├── hodge.rs            — Hodge decomposition, Betti numbers
├── flow.rs             — DecFlowField (Hodge-decomposed navigation)
├── simd.rs             — SIMD-accelerated sparse matmul
└── bench.rs            — Benchmarks vs naive alternatives
```

## Constraints

- Zero allocation in hot loop (pre-compute incidence matrices, reuse scratch buffers)
- `CochainField` uses fixed-size arrays where possible, Vec::with_capacity otherwise
- SIMD for sparse matmul (4 or 8 element chunks for auto-vectorization)
- Feature gate `dec_operators` — opt-in, default off until GOAT proof
- All DEC ops must satisfy `dₖ₊₁ ∘ dₖ = 0` exactly (no soft penalty)
- Files < 2048 lines each

## GOAT Gate

- Feature flag: `dec_operators`
- A/B test: `DecFlowField` vs `LeoPotentialGrid::gradient()` in Bomber arena
- Promote to default if: navigation quality ≥ naive + conservation guarantee
- Demote if: overhead > 20% with no quality improvement

## Validation

- [ ] All DEC identity tests pass (curl(grad)=0, div(curl)=0)
- [ ] Hodge decomposition reconstructs original cochain
- [ ] Arena proof shows DEC navigation quality
- [ ] Benchmark shows acceptable overhead vs naive gradient
- [ ] GOAT gate configured, can run with and without `dec_operators`
