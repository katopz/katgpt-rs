# Bench 677 — Issue 690: `ActionSpaceLog` f64 accumulator fix

**Date:** 2026-08-26 · **Repo:** katgpt-rs · **Issue:** `690`

## Defect

`ActionSpaceLog::avg_action_space()` under-reported the mean once the f32
running total passed ~2²⁴ (one ULP > 1 → every `+= n` rounds, systematically
downward). `PlayerAgg::sum` had the identical defect for
`avg_action_space_for()`. `peak_action_space()` (usize) was unaffected.

Measured regime (the issue's numbers, reproduced by the in-test old-shape
A/B): 2,260,000 records × 50 actions = total 1.13e8 → 48.593884 vs true 50
(**−2.81%**). That scale is one arm-seed of the Plan 348 go arena, not a
stress figure. The defect surfaced as a phantom "0.5% pruned on an
UNCONSTRAINED control arm" in the constraint-DSL sweep (f32 "before" vs f64
"after" accumulators).

## Fix (direction 1 from the issue — preferred)

`total_sum: f32 → f64`, `PlayerAgg::sum: f32 → f64`; `f32` return types
preserved by casting at the boundary (division stays in f64 — the count
itself can exceed 2²⁴ in f32). No API change, no allocation change. f64
holds exact integer sums to 2⁵³, unreachable at any arena scale.

## Gates

| Gate | Result |
|---|---|
| `avg_exact_past_2p24_issue690` | PASS — 2.26M records × 50 → `avg == 50.0` **exactly** (global + per-player) |
| `old_f32_shape_fails_issue690` | PASS — the pre-fix accumulation shape over the same stream drifts >1% (measured −2.8%) — the falsifiable A/B proving the fix load-bearing |
| `small_counts_still_exact` | PASS — katgpt-pruners' existing scales unchanged |
| `large_single_action_space_exact` | PASS — per-entry values past 2²⁴ (even values, exactly representable) → exact mean |
| Consumer regression | katgpt-pruners 126/126, katgpt-core default lib 1917/1917 |

## Consumer follow-up

riir-train `arena_constraints::BranchingProfile` (Plan 348 Item C) carries a
consumer-side f64 guard + `log_mean_before()` exposure "so the drift stays
visible" — that guard can be removed once this lands in its dep graph
(riir-train consumes katgpt-core transitively via riir-engine).

**Record:** commit on `develop` (see git log `fix(traits)` — Issue 690).
