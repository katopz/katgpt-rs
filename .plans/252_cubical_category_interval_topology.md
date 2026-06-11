# Plan 252: Cubical Category Interval Topology for Inference

**Date:** 2026-06
**Status:** COMPLETE (All phases T1-T31 done).
**Research:** 220 (Convenient Category of Cubes)
**Depends On:** Plan 251 (DEC Operators), Plan 195 (ThoughtFold)
**Blocks:** riir-ai Plan 278 (Operadic LoRA Composition)

## Overview

Research 220 (arXiv:2503.13663 "A Convenient Category of Cubes") identified three novel fusions from the paper's mathematical framework:

1. **IntervalPruner** — enforces convexity (interval-closure) of valid token sets during speculative decoding. The paper's interval object I gives a categorical notion of "betweenness" — we apply this to logit masks so that valid token ranges are contiguous (no "Swiss cheese" patterns that cause branching waste).

2. **CubicalNerve** — extracts CAT(0) cubical complexes from game zone posets for deterministic NPC navigation. The paper's cubical nerve functor ⊞ sends a distributive meet-semilattice L to a cubical set ⊞[L] whose geometric realization is a CAT(0) cube complex — this guarantees unique geodesics (shortest paths) for navigation.

3. **LatticeOpernad** — canonical AND/OR composition for ConstraintPruner expressions via the distributive lattice word problem. The paper's operadic structure on cubes gives canonical composition of face/degeneracy maps — we dualize this to canonical AND/OR composition of pruner constraints, eliminating redundant evaluations.

## Tasks

### Phase 1: IntervalPruner (No DEC dependency)
- [x] T1: Create `src/interval_pruner/` module with feature gate `interval_pruner`
- [x] T2: Implement `IntervalPruner` trait extending `ConstraintPruner` with `is_interval_closed()` and `close_intervals()`
- [x] T3: Implement interval detection on logit mask: find contiguous valid token ranges, detect "Swiss cheese" patterns
- [x] T4: Implement interval closure: merge nearby valid ranges (with configurable gap threshold)
- [x] T5: Wire into DDTree: when `interval_pruner` feature is on, apply interval closure before branching
- [x] T6: Write test: interval closure eliminates scattered rejects without over-accepting
- [x] T7: Write test: DDTree with IntervalPruner produces fewer rejected tokens than without

### Phase 2: LatticeOpernad — Canonical AND/OR for Pruners (No DEC dependency)
- [x] T8: Create `src/lattice_operad/` module with feature gate `lattice_operad`
- [x] T9: Implement `PrunerExpr` enum: Atom(usize), And(Box<Self>, Box<Self>), Or(Box<Self>, Box<Self>)
- [x] T10: Implement canonicalize() using distributive lattice word problem (absorption, idempotency, distributivity)
- [x] T11: Implement eval() that evaluates PrunerExpr against a token using composed pruner results
- [x] T12: Implement compose() that takes two PrunerExprs and AND/OR them, then canonicalize
- [x] T13: Wire into DDTree: when `lattice_operand` feature is on, compose pruner results via PrunerExpr instead of ad-hoc AND
- [x] T14: Write test: (A AND B) OR (A AND C) canonicalizes to A AND (B OR C)
- [x] T15: Write test: composition of 4+ pruners via PrunerExpr matches per-token AND but faster on batch

### Phase 3: CubicalNerve — CAT(0) from Game Zone Poset (Unblocked — Plan 251 Complete)
- [x] T16: Implement `DistributiveMeetSemilattice` trait: partial order with distributive meet
- [x] T17: Implement `cubical_nerve()` construction: L → ⊞[L] cubical set from distributive meet-semilattice
- [x] T18: Implement `cat0_geodesic()` on ⊞[L]: compute unique shortest path on CAT(0) complex
- [x] T19: Bridge `cat0_geodesic()` → `DecFlowField::exact_channel()` for navigation
- [x] T20: Implement cache: compute ⊞[L] on map load, invalidate on topology change
- [x] T21: Write test: cubical nerve of simple 2-zone map produces correct CAT(0) complex
- [x] T22: Write test: geodesic on CAT(0) is unique and matches BFS shortest path for grid maps
- [x] T23: Write benchmark: cubical nerve construction time vs map size

### Phase 4: GOAT Gate & Arena Proof
- [x] T24: Implement feature flag `cubical_topology` that enables all three fusions
- [x] T25: Write GOAT gate test: IntervalPruner + LatticeOpernad vs baseline in Sudoku arena
- [x] T26: Write GOAT gate test: CubicalNerve navigation vs LeoPotentialGrid::gradient() in Bomber arena (placeholder, blocked on Plan 251)
- [x] T27: Write benchmark: pruner composition overhead (operadic vs ad-hoc AND) for 2,4,8 pruners
- [x] T28: Document results: promote to default if quality ≥ baseline + structural guarantees, demote if overhead > 20%

### Phase 5: CPU/SIMD/GPU Auto-Route
- [x] T29: Implement adaptive backend: interval closure is O(vocab_size) → SIMD for large vocab
- [x] T30: Implement adaptive backend: cubical nerve is O(|L|) → CPU for small zone counts, SIMD for large
- [x] T31: Add threshold-based routing with configurable cutoffs

## Architecture

```
src/interval_pruner/
├── mod.rs           — Module root, feature gate
├── interval.rs      — IntervalPruner trait, interval closure algorithm
└── simd.rs          — SIMD-accelerated interval detection

src/lattice_operad/
├── mod.rs           — Module root, feature gate
├── expr.rs          — PrunerExpr enum, canonicalize(), eval()
├── compose.rs       — Operadic composition of pruner expressions
└── word_problem.rs  — Distributive lattice word problem solver

src/cubical_nerve/    (Phase 3, blocked on Plan 251)
├── mod.rs           — Module root, feature gate
├── poset.rs         — DistributiveMeetSemilattice trait
├── nerve.rs         — cubical_nerve() construction
├── cat0.rs          — CAT(0) geodesic computation
└── cache.rs         — Pre-compute on map load, invalidate on topology change
```

## Constraints
- Zero allocation in hot loop (pre-compute nerve, reuse scratch buffers)
- Files < 2048 lines each
- IntervalPruner must be backward-compatible with existing ConstraintPruner impls
- LatticeOpernad canonicalize() must terminate (distributive lattice word problem is decidable)
- CubicalNerve must only recompute on topology change, not per-frame
- All fusions behind feature flags: `interval_pruner`, `lattice_operad`, `cubical_nerve`
- Master gate: `cubical_topology` enables all three

## GOAT Gate
- Feature flag: `cubical_topology`
- A/B test: Phase 1+2 in Sudoku arena (IntervalPruner + LatticeOpernad vs baseline)
- A/B test: Phase 3 in Bomber arena (CubicalNerve vs naive gradient navigation)
- Promote to default if: quality ≥ baseline + at least one structural guarantee
- Demote if: overhead > 20% with no quality improvement

## Validation
- [x] Interval closure test passes (no Swiss-cheese valid regions after closure)
- [x] PrunerExpr canonicalization matches manual simplification
- [x] Cubical nerve of known poset matches expected complex
- [x] CAT(0) geodesic is unique and correct
- [x] GOAT gate configured, can run with and without each feature
- [x] All benchmarks show acceptable overhead
