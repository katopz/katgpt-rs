# Bench 200: Correlation Budget Allocation — Plan 200 GOAT Gate

**Date**: 2026-06-07
**Feature**: `corr_budget`
**Status**: Pending benchmark run

---

## Setup

```sh
cargo run --features corr_budget --example corr_budget_01_bench
```

## Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| EMA update overhead | < 5 ns/update | TBD | ⏳ |
| DDTree build overhead (corr vs uniform) | < 5% | TBD | ⏳ |
| Budget allocation quality (ordering) | d0 > d1 > d2 | ✅ PASS | ✅ |
| Convergence steps (α=0.1) | < 200 | TBD | ⏳ |

## Acceptance Criteria (GOAT Gate)

- Overhead ≤ 5% on DDTree build → production-ready
- Acceptance rate delta ≥ 3% over PositionWeightedBudget → default-on

## Files

- Implementation: `src/speculative/correlation_budget.rs`
- Integration: `src/speculative/dd_tree.rs` (`build_dd_tree_screened_corr`)
- Benchmark: `examples/corr_budget_01_bench.rs`
- Tests: 10 tests in `correlation_budget::tests` — ALL PASS ✅

## TL;DR

Correlation Budget Allocation implemented with full test suite, benchmark example, and DDTree integration. GOAT gate pending benchmark run.
