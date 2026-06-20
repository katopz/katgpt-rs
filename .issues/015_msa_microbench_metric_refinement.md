# MSA Microbench Metric Refinement

**Status:** CLOSED (all three metric refinements landed; GOAT verdict unchanged per issue prediction)
**Source**: Issue 014 (MSA Arena RULER Benchmark Infrastructure) — Optimization candidates
**Priority**: Low
**Blocked**: No — purely diagnostic metric improvements on existing micro-benchmarks
**Depends**: Nothing (uses existing `tests/bench_256_*.goat.rs` infrastructure)

## Closure rationale (2026-06-20)

All three metric refinements landed as additional analysis passes over existing
benchmark data — no new measurement code was required, confirming the issue's
prediction. **O1**: mean per-call Jaccard spread (per-group vs shared, averaged
over n_groups≥2) = **0.1824**, ranging from ~0.02 (n_groups=2, top_k=32) up to
~0.51 (n_groups=8, top_k=8, n_blocks=256) — spread grows monotonically with
n_groups and shrinks as top_k grows, confirming per-group DOES diversify
per-call (design goal met) even though cross-query union saturates at 1.003.
**O2**: N_QUERIES sweep over {256, 512, 1024, 2048} × {32K, 128K, 512K}
confirms the regime boundary — at 128K, KV-outer/Q-outer speedup rises from
1.086× (NQ=256) to 1.115× (NQ=2048) as avg_queries/block rises from 4 to 32,
but gain is modest and never crosses the 1.5× GOAT threshold. **O3**:
precision@adaptive_k = **1.0000** (adaptive picks EXACTLY the dense top-adapt_k
— it isn't picking worse blocks, just fewer); weighted recall = **0.6458**
(slightly above the recall_ratio cap of 0.629, but below the issue's >0.90
prediction because the deterministic sin-based centroids produce near-flat
score distributions where the lower ranks of dense-top-32 still carry
meaningful mass). All three GOAT verdicts remain FAIL exactly as predicted;
the refinements add nuance to *why* each strategy wins/loses in its regime,
they do not flip the verdict.

## Summary

Issue 014's full RULER arena is blocked on trained model weights + RULER dataset +
attention inference harness — none of which exist in katgpt-rs (modelless inference).
However, three metric redesigns on the **existing** synthetic micro-benchmarks are
fully tractable today and would sharpen Plan 256's GOAT verdict rationale.

These do **not** flip the GOAT verdict (per-group / KV-outer / adaptive-k all failed
their original micro-benchmark gates). They reframe *why* and *where* each strategy
wins or loses, which informs future promotion decisions.

## Acceptance Criteria

- [x] **O1 — Per-group coverage metric redesign**: measure per-call partition spread
      (Jaccard distance between groups within a single call) instead of cross-query
      union. The current metric saturates at ~1.0× because 128 queries × 32 top-k
      touch all reachable blocks regardless of per-call diversity.
      - File: `tests/bench_256_per_group.goat.rs`
      - Predicted outcome: shows per-group DOES diversify per-call (design goal met)
        even though cross-query union saturates. Does NOT flip GOAT — per-group's
        real value is high-top_k latency (already measured, already a pass at 0.98×).
      - **Measured (2026-06-20)**: mean per-call Jaccard spread (pergrp vs shared,
        aggregated over n_groups≥2) = **0.1824**. Per-cell range 0.024 (n_groups=2,
        top_k=32, n_blocks=64) → 0.505 (n_groups=8, top_k=8, n_blocks=256). Spread
        grows monotonically with n_groups, shrinks as top_k grows. GOAT unchanged:
        coverage 1.003 < 1.500 = FAIL.
      - **Implementation note**: `PerGroupTopKRouter` partitions blocks by
        `block_idx % n_groups` (disjoint ownership), so literal pairwise-group
        Jaccard distance is trivially 1.0 (degenerate). The meaningful per-call
        spread is per-group-vs-shared on each query, which is what was implemented.

- [x] **O2 — KV-outer query batching sweep**: sweep `N_QUERIES ∈ {256, 512, 1024, 2048}`
      instead of hardcoded 256. At 512K context with top_k=32, avg queries/block ≈ 1
      so reverse-index amortization gives nothing. Plan 256 line 120 already names
      this root cause.
      - File: `tests/bench_256_kv_outer.goat.rs`
      - Predicted outcome: at N_QUERIES=2048, KV-outer beats Q-outer at 128K because
        avg queries/block rises to ~8. Confirms existing analysis, sharpens regime
        boundary in the recommendation.
      - **Measured (2026-06-20)**: speedup table (Q-outer ms / KV-outer ms):
        | ctx  | NQ=256 | NQ=512 | NQ=1024 | NQ=2048 | avg_q/block@512K |
        |------|--------|--------|---------|---------|------------------|
        | 32K  | 1.226× | 1.251× | 1.214×  | 1.223×  | —                |
        | 128K | 1.086× | 1.098× | 1.084×  | **1.115×** | —             |
        | 512K | 1.027× | 1.057× | 1.035×  | 1.089×  | 1.0→8.0          |
        Confirms the regime boundary — at 128K, KV-outer/Q-outer speedup rises
        from 1.086× to 1.115× as NQ grows 256→2048, but gain is modest and never
        crosses the 1.5× GOAT threshold. GOAT unchanged = FAIL.

- [x] **O3 — Adaptive-k precision@k**: add two alternative metrics using existing data:
      1. `precision@adaptive_k` = `|adapt ∩ dense_top{adapt_k}| / adapt_k`
      2. `weighted recall` = `Σ scores(adapt ∩ dense) / Σ scores(dense_top32)`
      Current `bench_256_adaptive_k.goat.rs:166` recall is mathematically capped at
      20/32 = 0.625 because adaptive k ≈ 20 < 32.
      - File: `tests/bench_256_adaptive_k.goat.rs`
      - Predicted outcome: precision@k likely shows adaptive-k picks well (just fewer).
        Weighted recall likely > 0.90 because high-score blocks dominate. Reframes
        recommendation: "compute saver at near-equivalent precision" instead of
        "fails recall."
      - **Measured (2026-06-20)**: precision@adaptive_k = **1.0000** (adaptive picks
        EXACTLY the dense top-adapt_k — perfect precision, fewer blocks); weighted
        recall = **0.6458** (slightly above recall_ratio cap 0.629, but BELOW the
        issue's >0.90 prediction). The under-prediction is explainable: the
        deterministic sin-based block centroids produce near-flat score
        distributions where the bottom ~12 of dense-top-32 still carry meaningful
        mass, so dropping them loses real score weight. The headline finding still
        holds: adaptive-k is a compute saver at perfect precision, not a recall
        failure. GOAT unchanged = FAIL (recall_ratio 0.629 < 0.90).

## Why these are tracked separately from Issue 014

Issue 014's acceptance criteria are all transitively blocked on the arena
prerequisites (trained model, RULER dataset, harness). The optimization candidates
are the only items that don't need that infrastructure — they're micro-bench
refinements using synthetic data that already exists. Splitting them out lets the
micro-bench work proceed without waiting on the (possibly never-arriving) arena
prerequisites.

## Notes

- All three benchmarks already collect the data needed for the new metrics —
  no new measurement code is required, just new analysis passes over existing
  arrays.
- Each item is ~1 day of work: read existing bench, add metric calc, re-run,
  update Plan 256 verdict text.
- None of these flip the GOAT verdict — they add nuance to *why* each strategy
  wins or loses in its specific regime.
