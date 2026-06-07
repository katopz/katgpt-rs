# Plan 214: CoExplain Bidirectional Alignment — GOAT Proof

**Date:** 2026-06
**Status:** GOAT Verified
**Feature Flags:** `ted_lite`, `coexplain_pruner`, `coexplain_riir`

## GOAT Gate Matrix

| Gate | Criterion | Status |
|------|-----------|--------|
| G1 | Divergence metric correctness | ✅ PASS |
| G2 | Self-refining improves accuracy | ✅ PASS |
| G3 | Snapshot integrity | ✅ PASS |
| G4 | Translation rule extraction | ✅ PASS |
| G5 | Zero perf hurt when disabled | ✅ PASS |
| G6 | Feature isolation | ✅ PASS |

## Test Summary

| Phase | Module | Unit Tests | GOAT Tests |
|-------|--------|------------|------------|
| P1 | `ted_lite.rs` | 7 | G1 |
| P2 | `self_refining.rs` | 10 | G2 |
| P3 | `editable_constraint.rs` | 11 | G3 |
| P4+5 | `riir_feedback.rs` | 8 | G4, G5 |
| Integration | `coexplain_goat.rs` | — | 6 (G1-G6) |
| **Total** | | **36** | **6** |

## Performance Characteristics

| Operation | Complexity | Routing |
|-----------|-----------|---------|
| PrunerDivergence::compute | O(k) per pruner | CPU |
| clamp_adjustment | O(1) | CPU |
| PrunerAccuracy::record | O(1) | CPU |
| compute_threshold_adjustment | O(1) | CPU |
| PrunerSnapshot::new | O(k) blake3 | CPU |
| PrunerSnapshot::verify | O(k) blake3 | CPU |
| extract_translation_rules | O(n) + dedup | CPU |
| RuleBandit::record | O(1) amortized | CPU |
| RuleBandit::success_rate | O(1) | CPU |
| RuleBandit::best_rule | O(k) | CPU |
| classify_workload | O(1) match | CPU |
| WASM compilation (future) | CPU-bound | AsyncWorker |

## Feature Gate Hierarchy

```
ted_lite = []                           # P1: Divergence metric
coexplain_pruner = ["ted_lite", "bandit"]  # P2+P3: Self-refining + editable
coexplain_riir = ["coexplain_pruner"]   # P4+P5: RIIR feedback loop
```

- All features opt-in (not in `default`)
- `coexplain_riir` in `full` feature set
- Zero code compiled when disabled

## Promotion: OPT-IN

Depends on Curator API integration for full P4 Curator rule ingestion.
All infrastructure is in place and tested. Bandit refinement and translation
rule extraction are production-ready.

## Files Changed

- `src/pruners/riir_feedback.rs` — P4+P5 implementation (8 unit tests)
- `tests/coexplain_goat.rs` — 6 GOAT integration tests
- `examples/coexplain_demo.rs` — Full pipeline demo
- `src/pruners/mod.rs` — Module registration + re-exports
- `Cargo.toml` — Feature gate + test/example entries
