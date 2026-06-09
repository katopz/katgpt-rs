# Plan 231: Deep Manifold GOAT Features — Union Bound + Pathway Tracker + Federation Composer

> **Source:** Research 205 — Deep Manifold Neural Network Mathematics Modelless Distillation
> **Date:** 2026-06
> **Status:** ✅ Complete — All 3 features GOAT-proven, promoted to default-ON
> **Feature Gates:** `union_bound_confidence`, `pathway_tracker`, `federation_composer`
> **Related:** Research 51 (COMPLETE, Plan 085), Research 205 (modelless distillations)

---

## Tasks

- [x] T1: Add `union_bound_confidence` feature gate to `Cargo.toml`
- [x] T2: Implement `BranchConfidence` trait + `UnionBoundScorer` in `src/speculative/branch_confidence.rs`
- [x] T3: GOAT test — prove Boole's inequality correctness + linear degradation + low overhead
- [x] T4: Add `pathway_tracker` feature gate to `Cargo.toml`
- [x] T5: Implement `PathwayTracker` in `src/speculative/pathway_tracker.rs`
- [x] T6: GOAT test — prove pathway detection reduces thinking budget without quality loss
- [x] T7: Add `federation_composer` feature gate to `Cargo.toml`
- [x] T8: Implement `FederationComposer<C,S>` in `src/pruners/federation_composer.rs`
- [x] T9: GOAT test — prove federation composer early termination saves compute
- [x] T10: Add all three features to `full` feature set
- [x] T11: Update README.md — Deep Manifold GOAT section
- [x] T12: Run full benchmark suite — no regressions
- [x] T13: GOAT gate proof — all gates passing per feature

---

## GOAT Results

### Union Bound Confidence — GOAT 6/6 PASS

| Gate | Criterion | Result |
|------|-----------|--------|
| G1 | Boole's inequality correctness | union ≤ mult for 6/6 test cases, trivial cases match |
| G2 | Linear vs exponential degradation | Exact formulas verified, mult(n=16) < 0.2 |
| G3 | HybridScorer routing | ≤4 → mult, >4 → union, boundary correct |
| G4 | Per-step overhead | 76 ns (8-elem), linear scaling |
| G5 | Edge cases | Empty, zeros, single, perfect, clamp — all correct |
| G6 | Feature gate isolation | All types accessible, trait objects work |

**Note on +36% claim:** Corrected — by Boole's inequality, union confidence ≤ multiplicative. The value is architectural correctness (models additive error per §2.4.2) + predictable linear degradation.

### Pathway Tracker — GOAT 7/7 PASS

| Gate | Criterion | Result |
|------|-----------|--------|
| G1 | Convergence detection accuracy | **100%** (20/20 converged, 20/20 divergent) |
| G2 | Thinking budget savings | **85%** (3/20 avg steps, 0 false early exits) |
| G3 | Stability monotonicity | 0 violations over 9 steps |
| G4 | Per-step overhead | update: 123 ns, stability: 2.7 μs |
| G5 | Ring buffer correctness | stability unchanged after wrap |
| G6 | Feature gate isolation | PathwayTracker accessible |
| G7 | Minimum step enforcement | <3 steps → never converged |

### Federation Composer — GOAT 7/7 PASS

| Gate | Criterion | Result |
|------|-----------|--------|
| G1 | Early termination rate | **70%** (700/1000 queries) |
| G2 | Compute savings | **35%** (1300/2000 checks saved) |
| G3 | Pipeline correctness | 100→50→25, verified |
| G4 | Per-query overhead | 2.9 μs/call (100 candidates) |
| G5 | Residual calculation | Correct math, no div-by-zero |
| G6 | Feature isolation | All types accessible |
| G7 | Edge cases | Empty, all-pruned, all-pass — all correct |

---

## Promotion Decision

All three features promoted to **default-ON** in `Cargo.toml`.

| Feature | GOAT | Key Gain | Decision |
|---------|------|----------|----------|
| `union_bound_confidence` | 6/6 | Linear degradation, correct Boole model | ✅ default-ON |
| `pathway_tracker` | 7/7 | 85% thinking budget savings, 100% accuracy | ✅ default-ON |
| `federation_composer` | 7/7 | 70% early termination, 35% compute savings | ✅ default-ON |

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/speculative/branch_confidence.rs` | 199 | UnionBoundScorer + HybridScorer + 9 unit tests |
| `src/speculative/pathway_tracker.rs` | 173 | PathwayTracker ring buffer + 8 unit tests |
| `src/pruners/federation_composer.rs` | 178 | FederationComposer + ResidualCheck + 5 unit tests |
| `tests/bench_231_union_bound_goat.rs` | 249 | GOAT proof 6/6 |
| `tests/bench_231_pathway_tracker_goat.rs` | 306 | GOAT proof 7/7 |
| `tests/bench_231_federation_composer_goat.rs` | ~310 | GOAT proof 7/7 |
| `.benchmarks/231_union_bound_goat.md` | 82 | GOAT documentation |
| `.benchmarks/231_pathway_tracker_goat.md` | 82 | GOAT documentation |
| `.benchmarks/231_federation_composer_goat.md` | 79 | GOAT documentation |

---

## TL;DR

Three GOAT-rated modelless features from Deep Manifold Part 2, all **promoted to default-ON**:
1. **Union Bound Confidence** — additive branch scoring (§2.4.2), mathematically correct Boole model
2. **PathwayTracker** — intrinsic pathway stability detection (§4.2), 85% thinking budget savings
3. **FederationComposer** — explicit Model→Agent→Tool with residual check (§7.5), 70% early termination
