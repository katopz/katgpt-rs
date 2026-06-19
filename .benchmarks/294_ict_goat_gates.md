# Plan 294 — GOAT Gates G4–G6: Perf, Alloc, Isolation

**Date:** 2026-06-19
**Status:** ✅ All three PASS — G4 mean 1.96µs (target ≤ 50µs), G5 0 allocs/call, G6 feature isolation confirmed.

## Summary

| Gate | Test | Target | Result | Verdict |
|------|------|--------|--------|---------|
| G4 | `benches/bench_294_ict_perf.rs` | ≤ 50µs per `observe_and_detect_into` call (K=8, action_dim=32) | mean **1.96µs**, p50 2.00µs, p99 2.00µs | ✅ PASS (25× headroom) |
| G5 | `tests/bench_294_ict_g5.rs` | 0 allocs/call after warmup | **0 allocs** across 1000 measured calls | ✅ PASS |
| G6 | `tests/bench_294_ict_g6.rs` | feature isolation via cargo + nm | (a), (b), (c) all pass | ✅ PASS |

## G4 — Hot-path cost

```
=== Plan 294 Phase 5 G4 — BranchingDetector hot-path cost ===
K=8, action_dim=32, warmup=1000, timed=10000
Target: ≤ 50µs per observe_and_detect_into call.

     mean µs       p50 µs       p99 µs       max µs      verdict
       1.962        2.000        2.000        5.000         PASS
```

**25× headroom over target.** At 20Hz × thousands of NPCs this is well
within the cognitive-budget envelope. The hot path consists of:

1. **Population-mean accumulation** (chunked-4): K × action_dim = 256 f32
   adds. Autovectorizes to NEON/AVX2.
2. **`collision_purity(P̄)`** via `simd_dot_f32`: 32 f32 FMAs.
3. **K × `js_divergence`** (chunked-4 m-buffer + scalar log accumulator):
   K × action_dim = 256 f32 ops plus K × action_dim log calls.
4. **Top-k% threshold** via `scratch_sorted` (pre-allocated field) + stable
   sort of K elements + `branching_point_mask_into` (K comparisons).
5. **EMA updates**: 2 FMAs.

The chunked-4 inner loops (per AGENTS.md "write chunked 4-wide loops so
LLVM autovectorizes") are what brings this under 2µs.

## G5 — Zero-alloc hot path

```
=== G5 — Zero-alloc hot path ===
K=8, action_dim=32, warmup=100, measured=1000
Allocations during measured window: count = 0, bytes = 0
Tolerance: 0 allocs/call.
G5 PASS: 0.000 allocs/call (mean), 0 total across 1000 calls.
```

Verified via `katgpt_rs::alloc::TrackingAllocator` (debug-only, per-thread
counters — see `src/alloc.rs`). All scratch buffers are pre-allocated in
`BranchingDetector::new`:

- `scratch_p_avg: Vec<f32>` — length `action_dim`
- `scratch_m: Vec<f32>` — length `action_dim` (reused K times per call)
- `scratch_u: Vec<f32>` — length `k_trajectories`
- `scratch_mask: Vec<bool>` — length `k_trajectories`
- `scratch_sorted: Vec<f32>` — length `k_trajectories` (added during
  implementation when the first G5 run flagged a per-call sort allocation)

The caller passes a pre-allocated `BranchingReport` (also reused — see the
`observe_into_reuses_report_allocation` unit test in `detector.rs`).

## G6 — Feature isolation

```
=== G6 (a) — cargo build --no-default-features ===
G6 (a) PASS: default-OFF build succeeds.

=== G6 (b) — cargo build --no-default-features --features ict_branching ===
G6 (b) PASS: feature-only build succeeds.

=== G6 (c) — no ict_branching symbols leak into default-features build ===
G6 (c) PASS: no ict_branching symbols in default-OFF build.
```

The `ict_branching` feature is correctly isolated:

- The default build (which has `ict_branching` OFF) compiles cleanly with
  no reference to the ict module.
- The feature-only build (`--no-default-features --features ict_branching`)
  compiles cleanly — `ict_branching` has no accidental coupling to other
  features.
- `nm target/release/libkatgpt_core-*.rlib` in the default-OFF build
  contains zero `ict_branching`-named symbols.

The gate shells out to cargo + nm for an end-to-end check — the same
commands a downstream consumer would run.

## Run

```text
# G4 (perf)
cargo bench --bench bench_294_ict_perf --features ict_branching

# G5 (zero-alloc)
cargo test --features ict_branching --test bench_294_ict_g5 -- --nocapture

# G6 (isolation)
cargo test --features ict_branching --test bench_294_ict_g6 -- --nocapture
```

## References

- Plan 294 §Phase 5 T5.1–T5.4
- Research 270 §1.4 (plasma budget rationale)
- `crates/katgpt-core/src/ict/detector.rs` (the hot path)
- `benches/fpcg_probe_forecast_bench.rs` (the bench convention this follows)
