# Plan 063: Percepta CHT Hull KV Cache Upgrade (Phase A)

Replace Graham Scan + Ternary Search with Dynamic Convex Hull Trick (CHT) / LineContainer, matching the reference implementation at `.raw/transformer-vm/attention/hull2d_cht.h`.

**Distillation strategy:** Percepta's `transformer-vm` is Apache-2.0. We distill to Rust under MIT per `.research/32_percepta_distillation_strategy.md`. This is Phase A (P0–P2: CHT + cumulative sum + parabolic encoding). Phase B (P3: ReGLU/stepglu) follows. Phase C (P4–P6: full compiler) is a pivot decision.

## Goal

Upgrade `KVCache2D` to handle arbitrary 2D points, support both upper and lower hull queries, add tie-breaking modes (LATEST/AVERAGE), and enable cumulative sum via uniform attention.

## Background

Our current `KVCache2D` (in `src/percepta.rs`) has fundamental limitations:
- Requires monotonically non-decreasing X (sequential execution traces only)
- Only maintains upper hull — `qy < 0` queries produce wrong results (documented in adversarial tests)
- Stores all N keys — O(N) memory, no sublinear compression
- No tie-breaking — cannot do cumulative sum (needs AVERAGE) or latest-write semantics
- Uses `usize` values — cannot store f64 pairs needed for proper attention output

The reference uses a **Dynamic Convex Hull Trick** (CHT) via `std::multiset<Line>` which:
- Handles arbitrary 2D points (no monotonic-X requirement)
- Maintains upper + lower hulls + edge metadata for all query directions
- Stores only hull vertices with aggregated `HullMeta` — sublinear memory
- Supports LATEST and AVERAGE tie-breaking
- O(log n) for both insert and query (no ternary search)

## Tasks

- [ ] **T1: Create `src/percepta/` module directory**
  - Move `src/percepta.rs` → `src/percepta/mod.rs` (re-export everything)
  - Create `src/percepta/cht.rs` for the new CHT implementation
  - Create `src/percepta/hull.rs` for the `HardAttentionHead` wrapper
  - Create `src/percepta/gates.rs` for ReGLU/stepglu primitives (placeholder)
  - Update `src/lib.rs` and any imports

- [ ] **T2: Implement `HullMeta` value aggregation**
  - `vsum: [f64; 2]` — running sum of value pairs
  - `vlast: [f64; 2]` — most recent value by sequence number
  - `count: usize` — number of merged points
  - `last_seq: i64` — highest sequence number
  - `add(val: [f64; 2], seq: i64)` — merge a new point
  - `merge(other: &HullMeta)` — combine two metas
  - `resolve(tb: TieBreak) -> [f64; 2]` — produce LATEST or AVERAGE result

- [ ] **T3: Implement `TieBreak` enum and `CHT` data structure**
  - `enum TieBreak { Average, Latest }`
  - `struct Line { m: f64, b: f64, p: OrderedFloat, meta: HullMeta }` — slope, intercept, breakpoint
  - `struct CHT { lines: BTreeSet<Line> }` — ordered by slope
  - `add_line(m, b, meta)` — insert maintaining max envelope, O(log h) amortized
  - `argmax(x) -> &Line` — binary search on breakpoint, O(log h)
  - `isect(x, y)` — compute intersection, detect dominated lines
  - Handle equal-slope cases (merge, dominate, or replace)

- [ ] **T4: Implement `HullHalf` wrapper**
  - `struct HullHalf { cht: CHT, is_upper: bool }`
  - `insert(kx, ky, val: [f64; 2], seq)` — maps to `cht.add_line(kx, ky, meta)` or negated for lower
  - `query(qx, qy, tb) -> [f64; 2]` — computes `m = qx/qy`, calls `cht.argmax(m)`, handles ties by checking neighbors

- [ ] **T5: Implement `HardAttentionHead` (replaces `KVCache2D`)**
  - `upper: HullHalf` — max envelope for `qy > 0`
  - `lower: HullHalf` — min envelope for `qy < 0`
  - `global: HullMeta` — all values (for `qx == 0 && qy == 0`)
  - `left_meta: HullMeta` — min kx values (for `qy == 0 && qx < 0`)
  - `right_meta: HullMeta` — max kx values (for `qy == 0 && qx > 0`)
  - `n: usize` — total points inserted
  - `insert(key: [f64; 2], val: [f64; 2], seq: i64)` — update all structures
  - `query(q: [f64; 2], tb: TieBreak) -> [f64; 2]` — dispatch to correct hull/edge
  - `clear()`, `len()`, `is_empty()`, `hull_size()`

- [ ] **T6: Implement parabolic key encoding helpers**
  - `encode_key(k: f64, offset: f64, tie_break: TieBreak, inv_log_pos: f64) -> [f64; 2]` — `k → (2k - 2·offset, -k² + 2k·offset - offset² + tie_break_term)`
  - `encode_query(q: f64, offset: f64) -> [f64; 2]` — `q → (q - offset, 1)`
  - `clear_key(key: [f64; 2], big: f64) -> [f64; 2]` — subtract `big` from ky

- [ ] **T7: Implement cumulative sum (`fetch_sum` equivalent)**
  - `insert_cumsum(value: f64, position: f64, seq: i64)` — uniform key (constant) + value
  - `query_cumsum(position: f64) -> f64` — average * position = exact cumulative sum
  - Uses AVERAGE tie-breaking and uniform keys

- [ ] **T8: Keep legacy `KVCache2D` as `KVCache2DLegacy`**
  - Rename current struct, keep all existing tests passing
  - This preserves our proven test suite as a correctness reference
  - Add feature flag `percepta_cht` to gate the new implementation

- [ ] **T9: Port all existing tests to new `HardAttentionHead`**
  - Verify parity: all 30+ existing tests pass with new CHT implementation
  - The adversarial V-shape tests should now PASS (lower hull handles `qy < 0`)
  - Add new tests for:
    - LATEST vs AVERAGE tie-breaking
    - Arbitrary (non-monotonic-X) point distributions
    - Cumulative sum correctness (Fibonacci, counter, DFA)
    - Parabolic key encoding round-trip
    - HullMeta merge correctness
    - Edge cases: `qy == 0`, `qx == 0`, empty cache, single point
  - Stress test: 100K+ points with random queries

- [ ] **T10: Integration with existing `StreamingSolver` and `Sudoku9x9`**
  - Update `StreamingSolver` to use `HardAttentionHead` internally
  - Verify 9×9 Arto Inkala still solves correctly
  - Benchmark: compare Graham Scan vs CHT throughput on execution traces

## Design Decisions

1. **Use `BTreeSet` not `Vec`**: The CHT requires ordered insertion and deletion by slope. Rust's `BTreeSet` is equivalent to C++ `std::multiset`. We need a wrapper to handle duplicate slopes (use a secondary key like insertion order).

2. **`OrderedFloat` for `p` (breakpoint)**: Breakpoints are `f64` but must be comparable. Use `ordered_float::OrderedFloat` or implement our own wrapper.

3. **`f64` values, not `usize`**: The reference stores `[f64; 2]` value pairs for attention output. Our `usize` values were sufficient for tests but not for real attention integration.

4. **Keep module split clean**: `cht.rs` (data structure), `hull.rs` (attention head), `gates.rs` (future ReGLU/stepglu), `mod.rs` (re-exports).

5. **Feature-gate the new code**: `percepta_cht` feature flag. Legacy `KVCache2D` stays as default until new code is fully validated.

## Dependencies

- `ordered_float` crate (or manual `Ord` wrapper for `f64`)
- No other new dependencies

## Constraints

- Keep `src/percepta.rs` < 2048 lines (use module split)
- All existing tests must continue to pass
- No performance regression on execution-trace workloads (monotonic X)
- Must fix the adversarial V-shape failure (qy < 0 queries)

## Success Criteria

- [ ] All existing tests pass with both legacy and CHT implementations
- [ ] Adversarial V-shape test PASSES with CHT (was failing with legacy)
- [ ] Arbitrary 2D point distributions work correctly
- [ ] LATEST and AVERAGE tie-breaking verified
- [ ] Cumulative sum works via uniform attention
- [ ] Parabolic key encoding API available
- [ ] 100K point stress test passes
- [ ] No performance regression on monotonic-X traces

## References

- `.raw/transformer-vm/attention/hull2d_cht.h` — CHT data structure (323 lines, Apache-2.0 © Percepta)
- `.raw/transformer-vm/attention/hull_cache.py` — Python wrapper (44 lines)
- `.raw/transformer-vm/graph/core.py` — `fetch()`, `fetch_sum()`, parabolic encoding
- `.research/31_percepta_deep_dive.md` — Full gap analysis
- `.research/32_percepta_distillation_strategy.md` — Phased distillation verdict (Phase A/B/C)
- `.research/03_Commercial_Open_Source_Strategy_Verdict.md` — Engine/Fuel split strategy