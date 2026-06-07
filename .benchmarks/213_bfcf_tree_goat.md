# Plan 213: BFCF Tree — GOAT Proof

**Date:** 2026-06-08
**Feature Gate:** `bfcf_tree` — OPT-IN, GOAT-gated

## GOAT Gate Matrix

| Gate | Criterion | Status |
|------|-----------|--------|
| G1 | Region pruning correctness | ✅ PASS |
| G2 | PWC closure maintained after 100 updates | ✅ PASS |
| G3 | Percept routing ≥ 95% accuracy | ✅ PASS |
| G4 | Preimage improvement ≥ 10% | ✅ PASS |
| G5 | Zero perf hurt when disabled | ✅ PASS |
| G6 | Feature isolation / sigmoid bounded | ✅ PASS |

## Test Coverage

| Test | Gate | File |
|------|------|------|
| `goat_region_pruning_correctness` | G1 | `tests/bfcf_tree_goat.rs` |
| `goat_pwc_closure_after_n_updates` | G2 | `tests/bfcf_tree_goat.rs` |
| `goat_percept_routing_accuracy` | G3 | `tests/bfcf_tree_goat.rs` |
| `goat_preimage_improves_acceptance` | G4 | `tests/bfcf_tree_goat.rs` |
| `goat_feature_isolation_empty_inputs` | G5 | `tests/bfcf_tree_goat.rs` |
| `goat_complexity_sigmoid_bounded` | G6 | `tests/bfcf_tree_goat.rs` |

## Percept Router Tests (Phase 4)

| Test | File |
|------|------|
| `test_complexity_low_for_simple_partition` | `src/pruners/percept_router.rs` |
| `test_complexity_high_for_complex_partition` | `src/pruners/percept_router.rs` |
| `test_route_fast_for_simple` | `src/pruners/percept_router.rs` |
| `test_route_deep_for_complex` | `src/pruners/percept_router.rs` |
| `test_route_standard_for_medium` | `src/pruners/percept_router.rs` |
| `test_complexity_bounded_unit_interval` | `src/pruners/percept_router.rs` |
| `test_entropy_of_uniform_labels` | `src/pruners/percept_router.rs` |

## Expected Gains

| Metric | Before (token-by-token) | After (BFCF Tree) |
|--------|------------------------|-------------------|
| Evaluations per step | O(vocab_size ≈ 128K) | O(regions ≈ 50) |
| Routing accuracy | Fixed threshold | ≥ 95% measurable |

## Decision: OPT-IN

Feature stays behind `bfcf_tree` flag. Needs real inference benchmark before promotion to default.
