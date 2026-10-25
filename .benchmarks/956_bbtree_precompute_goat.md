# Bench 956 — Issue 964 GOAT: solver static-invariant precompute (bbtree-class ordered-structure audit)

**Status:** GOAT G1–G4 PASS — landed ungated (zero-risk refactor class: byte-identical outputs, no feature flag; the flag-promotion rule does not apply — nothing was gated)
**Date:** 2026-10-25
**Issue:** katgpt-rs Issue 964 (bbtree distill, Research 322) — `crates/katgpt-assign`
**Claim lease:** `katgpt-rs-m3-964a`, epoch=1792932000, expires 2026-10-27T12:40:00Z

## What changed

1. **(a) Sorted+deduped GroupCount member lists precomputed at `specs::compile()`** — `Compiled.group_members` (spec-index aligned). The solver's `refresh_heat` + `heat_of_container` consumed a `members.clone() + sort_unstable + dedup` PER recompute before; both now read the compile-time list. Compile itself gained nothing (the same sort now feeds the at-pair family + the goal-term build — one sort per spec, was two).
2. **(b) `heat_order` baseline precomputed at `Solver::new`** — containers in tie-permutation order (the static `0..b` + seeded `tie_perm`); `refresh_heat` drops its per-recompute `clear()+extend(0..b)` and keeps the comparison sort keyed by `(Reverse(heat), tie)`. Tie values are unique, so the output is the exact same permutation as sorting `0..b` from scratch. **The issue's conditional bucket/radix pass over the heat key is measured out** — the sort never dominated (the member clone did; the G2 deltas below are the whole effect); not built, per the issue's own "only if the sort shows up".
3. **Two bonus G4 findings, same audit class, caught by the new gate tests (both fixed):**
   - `leaf_deltas` under-sizing: capacity `2·dims+2`, but a GroupCount member's at-leaf slice is `b` wide → the scratch grew mid-run (cap 4→8→16 = 384 B measured once per solve). Now sized `3·max_at + dims·(b+2) + 3` (worst case across single/swap/triple builders).
   - Per-container membership Vecs grew past their construction high-water during the solve (one container took a 33rd object → cap 32→64, 256 B). Now preallocated to the `member_snapshot` precedent's own law ("the widest container can hold every object"): `Vec::with_capacity(n)` each. Memory note: b·n·4 worst case (8 MB at the 20000×100 bench; the shard-rebalancer consumer shapes are far below that).

## Gates

- **G1 (byte-identical outputs):** the full crate suite 57/57 (incl. `determinism.rs`); AND the bench quality columns (folded/violation/spread/moved/max_util/**evals**) identical on every row before vs after — identical evals = identical trajectory, the strongest byte-identity form. AND the two new G4 tests' solves reach the same `evals=263575 / NoImprovingMove` as the pre-fix run.
- **G2 (perf, --release):** targeted GroupCount class **−17.0% / −22.2%** (balance_20c_b12 / balance_5c_b20 local); every other row within noise (±1% on the 2.4–24 s rows, sub-ms rows noisy). Single run per side on a loaded shared box — wall deltas are directional; the evals-identity is exact.
- **G3 (no regression):** same bench matrix, no row regressed beyond noise; suite counts 55 → 57 (two added gates, zero moved).
- **G4 (alloc-free hot path):** two new gate tests in `src/alloc.rs` — `group_count_solve_hot_loop_is_alloc_free`, `heat_of_container_is_alloc_free` — assert `(0 allocations, 0 bytes)` for a GroupCount-heavy solve and the explain-heat loop. **Canary proven on the pre-fix code** (base worktree `d80e040a4` + the test files only): the solve gate reads **50,010 allocations / 22,403,968 bytes** and the heat gate **12 allocations / 9,600 bytes** — the gates fire loud, not green-zero.

## Table (before = base `d80e040a4`, after = fix; `cargo bench -p katgpt-assign --bench bench_924_assignment_g2_baselines`, release)

| fixture | method | before ms | after ms | Δ | evals (b=a) |
|---|---|---|---|---|---|
| balance_5c_b20 | initial | 0.15 | 0.05 | -66.7% | 0 |
| balance_5c_b20 | ffd | 0.09 | 0.08 | -11.1% | 0 |
| balance_5c_b20 | local | 34.59 | 26.90 | **-22.2%** | 125410 |
| balance_20c_b12 | initial | 0.02 | 0.02 | +0.0% | 0 |
| balance_20c_b12 | ffd | 0.05 | 0.04 | -20.0% | 0 |
| balance_20c_b12 | local | 12.67 | 10.51 | **-17.0%** | 31170 |
| rand_200x10 | initial | 0.01 | 0.05 | +400.0% | 0 |
| rand_200x10 | ffd | 0.02 | 0.04 | +100.0% | 0 |
| rand_200x10 | local | 2407.73 | 2403.90 | -0.2% | 10000384 |
| rand_1000x20 | initial | 0.03 | 0.03 | +0.0% | 0 |
| rand_1000x20 | ffd | 0.07 | 0.06 | -14.3% | 0 |
| rand_1000x20 | local | 2739.43 | 2722.52 | -0.6% | 10000384 |
| rand_5000x50 | initial | 0.06 | 0.08 | +33.3% | 0 |
| rand_5000x50 | ffd | 0.37 | 0.41 | +10.8% | 0 |
| rand_5000x50 | local | 4164.88 | 4204.96 | +1.0% | 10000134 |
| rand_20000x100 | initial | 0.19 | 0.18 | -5.3% | 0 |
| rand_20000x100 | ffd | 1.93 | 1.92 | -0.5% | 0 |
| rand_20000x100 | local | 7197.81 | 6973.89 | -3.1% | 10000384 |
| rand_20000x200_4d | initial | 0.30 | 0.31 | +3.3% | 0 |
| rand_20000x200_4d | ffd | 5.76 | 5.53 | -4.0% | 0 |
| rand_20000x200_4d | local | 23869.19 | 23887.51 | +0.1% | 10000384 |
| repair_10c_L1000_e50 | initial | 0.02 | 0.01 | -50.0% | 0 |
| repair_10c_L1000_e50 | ffd | 0.02 | 0.01 | -50.0% | 0 |
| repair_10c_L1000_e50 | local | 0.04 | 0.04 | +0.0% | 141 |
| shard_256x8 | initial | 0.01 | 0.01 | +0.0% | 0 |
| shard_256x8 | ffd | 0.02 | 0.02 | +0.0% | 0 |
| shard_256x8 | local | 3234.76 | 3210.87 | -0.7% | 8284447 |
| shard_1024x16 | initial | 0.04 | 0.04 | +0.0% | 0 |
| shard_1024x16 | ffd | 0.08 | 0.07 | -12.5% | 0 |
| shard_1024x16 | local | 8042.21 | 8094.02 | +0.6% | 10000384 |
| balance_stall (seed 4) | local / +3cyc / +kl | 34.63 / 32.14 / 32.52 | 34.05 / 34.28 / 34.03 | -1.7 / +6.7 / +4.6% | 250161 / 251560 / 253868 |
| balance_stall (seed 10) | local / +3cyc / +kl | 38.20 / 36.19 / 36.43 | 36.80 / 35.87 / 36.22 | -3.7 / -0.9 / -0.6% | 287735 / 288211 / 291438 |
| balance_stall (seed 27) | local / +3cyc / +kl | 26.82 / 28.45 / 27.83 | 28.02 / 27.79 / 28.33 | +4.5 / -2.3 / +1.8% | 212819 / 213144 / 216523 |
| balance_stall (seed 34) | local / +3cyc / +kl | 3.84 / 4.33 / 4.72 | 4.08 / 4.48 / 4.91 | +6.3 / +3.5 / +4.0% | 25250 / 25799 / 28958 |
| phase2_specs_500x20 | local | 107.74 | 108.45 | +0.7% | 200564 |
| phase2_specs_500x20 | local+all | 120.46 | 119.74 | -0.6% | 215468 |

Reading notes: the two big `local` wins ARE the targeted class — `perfect_balance` builds GroupCount specs, so its whole hot path ran the per-recompute clone+sort+dedup before. The rand/shard `local` rows have no GroupCount spec: their heat path never touched the changed code (±1% = box noise), which is exactly the expected neutrality. Sub-ms and stall rows are load noise (single-run).

## Box state (the latency-claim rule)

```
PROVENANCE: power=AC Power load=4.04 swap=5.56M canary=skipped powermode=2(high)
```
M3 Max 16 cores, 64 GiB (92% free), macOS 27.0.1, load avg 4.04/4.09/4.12 (shared box — sibling agent cargo builds active), AC power, charged. BEFORE and AFTER ran sequentially in the same window (interleaving was NOT done — a swap-order replication would strengthen the wall claims; the evals-identity carries the correctness half regardless).

## References

- Issue 964 (closed, file removed per noise-reduction; record in `HISTORY.md`)
- bbtree: <https://bbtree.xiangpeng.systems/post/> · `XiangpengHao/bbtree@8adae97a0ed97e5c98fa2790312eb0aec9719846`
- Bench 924 (the reused G2 harness): `.benchmarks/924_assignment_g2_baselines.md`
