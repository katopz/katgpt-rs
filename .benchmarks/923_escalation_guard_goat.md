# 923 — escalation_guard GOAT (Issue 923 / riir-refine Plan 202 R1)

**Status:** GOAT G1/G2/G3/G4 PASS (2026-10-07) — lands OPT-IN (`escalation_guard = []`, katgpt-core). NOT PROMOTED: no consumer-measured gain exists yet — the consumers land separately (riir-refine's Plan-202 R1 escalation-manifest half; the instinct/rethink migration onto the shared receipt tiers). Feature Flag Discipline: promotion needs a consumer-measured gain.

Owner: Issue 923 (closed 2026-10-07, file removed — record in [HISTORY.md](../HISTORY.md)) · Plan: [riir-refine Plan 202 R1](../../riir-refine/.plans/202_decision_stack_arsenal_alignment.md) (round-3 AGREE implementation baseline). Extracted from riir-rethink's ESC lane (`EscRateGuard` / `kill_switch_decode` / the `ESC:think|cheap|demoted` tiers) — one definition at the lowest shared dep.

## Box state (the G2 rule)

- macOS 26.6.2, Apple M3 Max (16 cores), AC plugged (battery 79% charging).
- CPU ~36% at session start (≥4 sibling agent sessions active: seal-game-editor viewer, riir-train HF reorg, riir-ai hero-AI, riir-chain CLI, riir-reflex audit); best-of-3 timing is the load discipline for this state.
- Release profile, the shared `/Users/katopz/git/katgpt-rs` checkout (no sibling edits in this crate — verified clean at start).

## G1 — correctness (in-module `#[cfg(test)]`, 17 tests)

| Arm | Pins |
|---|---|
| `rate_is_none_until_the_window_fills` | 199 observes → `rate()` None at every step (a partial window is no measurement); the 200th → Some |
| `rate_counts_escalations_over_the_window` | exactly 50/200 escalations → rate 0.25, no latch at permissive bounds |
| `window_rolls_old_decisions_out` | window-4 ring: [1,1,1,0] → 0.75; four falses roll the trues out → 0.0 |
| `determinism_same_sequence_same_state` | 500 mixed observes → identical (rate, latched) across two instances |
| `latch_fires_above_max_and_is_sticky` | 150/200 = 0.75 > 0.60 → `rate_above_max`; 1000 quiet observes never unlatch |
| `latch_fires_below_min_on_both_bounds_lanes` | 10/200 = 0.05 < 0.20 → `rate_below_min` |
| `cost_ceiling_lane_has_no_below_min_latch` | 0/200 at cost-ceiling-only → rate 0.0, NOT latched (`rate_below_min` cannot exist without a min — refine's shape); then 0.75 → `rate_above_max` still works |
| `in_bounds_rate_never_latches` | 0.30 inside [0.15, 0.60] → unlatched |
| `rate_bounds_grammar_truth_table` | `0 < min < max < 1`, finite, ordered — the shared predicate instinct's `EscalateSpec` grammar consumes |
| `both_bounds_constructor_enforces_the_grammar` / `cost_ceiling_constructor_enforces_the_grammar` / `zero_window_refused` | construction-time panics with the grammar in the message |
| `demote_only_decode_exact_literal_table` | ONLY `Some("0")` demotes; None/`"1"`/`"true"`/`""`/junk/`" 0"`/`"0.0"` all armed |
| `tpr_parse_kill_is_the_same_table_inverted` | polarity cross-pin against katgpt-core's existing private `tpr::parse_kill` table — neither drifts |
| `receipt_tiers_reproduce_the_esc_bytes` | `ESC:think` / `ESC:cheap(A1)` / `ESC:demoted(A0)` byte-identical to riir-rethink's tiers (migration = pure move) |
| `receipt_tiers_namespace_parameterized` | `REFINE:cheap(clippy_lints)` etc. — refine's arm-H receipt vocabulary |
| `observe_is_allocation_free` | 1000 observes = 0 allocs (TrackingAllocator, construction before the reset) |

Test-gate row: `katgpt-core:2158:escalation_guard` (2158 = 2141 default + 17).

## G2 — latency (bench_923, best-of-3, release)

| Path | Measured | Ceiling |
|---|---|---|
| `observe(escalated)` — rolling-window ring replace + counter update + latch evaluate, steady-state (pre-filled window) | **1.74 ns** | 20 |

(The timed loop also consumes `latched()` + `rate()` per iteration through
the sink — the 1.74 ns includes those reads, so it over-states the pure
observe cost.)

## G3 — no-regression

Default-feature lib count measured **2141 passed / 8 ignored** at this
landing base (develop, clean); the feature compiles to nothing at default
features — the default surface is byte-unchanged. Clippy
`--all-targets` at the feature: zero warnings; default clippy: zero new
warnings.

## G4 — alloc-free

`observe` = 0 allocs across 1000 calls (in-module debug-assertion suite
over the lib test binary's TrackingAllocator; construction allocates the
ring ONCE — `with_window` — and is excluded by the reset placement).

## The substrate-first record (why this is here, not a fourth copy)

The exact-literal kill-switch decode existed as THREE private spellings
(riir-rethink `kill_switch_decode` public; katgpt-core `tpr::parse_kill`
private+OnceLock'd; `ugc_schedule`'s `UGC_DEBUG` pattern-copy) and the
rate latch as ONE public sibling copy (`EscRateGuard`) — a refine-side
copy (Plan 202's original shape) would have been the FOURTH spelling of
the decode and the SECOND of the latch, breaking the one-definition law
the plan itself cites. `tpr`/`ugc_schedule` stay on their private copies
(both are default-on surfaces an opt-in feature cannot serve without
either breaking default builds or forcing premature promotion);
truth tables are cross-pinned by tests on both sides instead.
Mechanism-distinct from `rate_control` (Issue 873): that is a dual-EWLS
effect-size CONTROLLER nudging a scale factor on a noisy scalar; this is
a binary escalation-flag window latch with a terminal demotion state.
