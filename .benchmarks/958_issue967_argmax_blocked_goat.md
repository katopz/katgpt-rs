# Bench 958 — blocked two-level NEON argmax GOAT gate (Issue 967)

**Status:** RECORD — G1 + G2 PASS → `argmax_blocked` promoted to katgpt-types DEFAULT (2026-10-27)
**Date:** 2026-10-27
**Issue:** [967](../.issues/967_blocked_neon_argmax.md) (found by Issue 966 G2)
**Vehicle:** `cargo test -p katgpt-types --release --test bench_967_argmax_blocked_ab -- --nocapture --test-threads=1`

## What changed

`simd_argmax_f32` on aarch64 was the lane-wise single pass: one dependent
`vcgt → vmax → vbsl` chain per 4 floats, so it ran at the chain's LATENCY. The
Issue 966 G2 run found the top-16 pool extraction (a 256-wide `simd_max_f32`
block-skip) beating it at a 151 936 vocab. The blocked kernel splits the work.
Pass 1 reduces each 256-float block with 8 INDEPENDENT `vmaxq_f32` accumulators,
so it is throughput-bound, and keeps the first block whose max is strictly
greater than every earlier one. Pass 2 runs `simd_first_ge_f32` (16-wide
`vcgeq` probes) inside that one block. Slices with n < 512 keep the lane-wise
kernel. A NaN in any block max sends the whole slice to the lane-wise kernel,
which is bit-identical by construction.

## Pre-registered bars (in the test header before the first run)

- **G1:** same index, and a value `==` to it, against the lane-wise kernel over
  uniform, 4-level ties, peaked (with a later duplicate max), signed zeros,
  all-equal, ±inf, all -inf, and NaN inputs (bit-identical on NaN).
  Lengths 0…151 936 straddle the 256 block and the 512 floor; 6 reps each.
- **G2:** interleaved median ratio blocked / lane-wise ≤ 0.80 at n ∈
  {32 000, 151 936} for the `iid` shape, and ≤ 1.10 (no regression) for every
  shape `iid`/`early`/`mid`/`late` at n ∈ {512, 4096, 32 000, 151 936}.

## Results (final code, `--release`)

| n | iid | early | mid | late | lane ns (iid) | blocked ns (iid) |
|---|---|---|---|---|---|---|
| 512 | 0.348 | 0.288 | 0.329 | 0.282 | 74.5 | 23.0 |
| 4096 | 0.187 | 0.323 | 0.285 | 0.316 | 502.4 | 120.0 |
| 32 000 | **0.273** | 0.267 | 0.265 | 0.267 | 3958.4 | 1105.4 |
| 151 936 | **0.318** | 0.325 | 0.327 | 0.314 | 18399.9 | 5949.4 |

G1 **PASS**. G2 **PASS**: the blocked kernel is 2.9–5.3× faster at every max
position. One G2 red happened on the way: the first version rescanned the
winning block with a scalar loop, and `512/late` read **1.197**. The fix was a
NEON `>=` probe (`simd_first_ge_f32`), not a higher floor. A mutation probe
confirmed the gate can fail. G4: no heap use (stack accumulators only).

**Box state:** M5 Ultra (amo), macOS 27.0.1. `PROVENANCE: power=AC Power
load=5.80 swap=5731.75M canary=skipped powermode=0(auto)` (preflight PASSED,
read right after the run). The verdict is the interleaved ratio; treat the
absolute ns as indicative only. A sibling riir-train test was using ~15 cores
during the earlier iterations (load 9–14). Ratios matched across those runs
within the min/max bands.

## Platform scope

- **aarch64:** the claim. **x86_64:** unchanged (the two-pass path already won
  Issue 817); clippy `-D warnings` is clean on `x86_64-unknown-linux-musl`. The
  blocked kernel was not ported to x86_64 and is not executed there.
  **wasm32:** unchanged; clippy is clean with and without `simd128`.
- Regression suites: katgpt-types lib 169/0 (default + no-default), katgpt-core
  lib 2167/0, katgpt-rs lib 212/0, katgpt-speculative lib 310/0 (default) and
  343/0 (`grid_reselect`).

## Knock-on: Issue 966 G2 re-opened

966's G2 baseline IS `simd_argmax_f32`, so that bar got 3.5× harder. The
re-run on the old pool code read **3.59× / 1.58×** (bar ≤ 1.50). Three new
shared primitives went into the pool extraction: `simd_block_max_256_sub64`,
`simd_block_max_256_sub64_into` and `simd_first_ge_f32`. On top of them it
ranks the block maxima into an exact floor τ (the K-th largest block max
lower-bounds the K-th largest element), then rescans only the 64-wide
sub-blocks that clear `max(τ, tail)`. Result: **1.580 @32k (FAIL) · 1.126
@152k (PASS)**. Reselect is now 26.0 µs per 16-step block @32k (was 57.7) and
98.9 µs @152k (was 157.5). The bit-identity battery now includes floor-binding
shapes (32 077/k16, 8192/k4, and past `FLOOR_MAX_BLOCKS`); the mutation
`m < τ → m <= τ` is caught. Measured dead ends, kept in the code comments:
a 32-wide probe was slower than 16, chunked ranking was noise, fusing the
ranking into the max loop was noise, and a per-32-block running floor was
about 25% slower (the looser floor admits more candidates than L1 residency
saves).
