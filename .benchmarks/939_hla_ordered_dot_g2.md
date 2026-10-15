# Bench 939 — HLA ordered-fold G2: the committed-state dot swap at decode-loop scale (Issue 940 HL-1/2/3)

**Date:** 2026-10-16
**Verdict:** **G2 PASS — REPORTED, not barred** (the bench_844 convention). The ordered fold costs one dot per state-update step; the whole-step A/B reads **-1.1% to +2.4%** across arms — within run-to-run scatter of the predicted +0.8% (raw-dot delta 36.7 ns on a ~4.8 µs step). The decode readout hot path (`hla_denom` / `hla_readout_normalized` / `ahla_denom`, HL-4 LOCAL) is untouched and stays platform SIMD.

## What landed

Fix unit ④ of the Issue 940 Part C queue: the state-feeding dots in
`katgpt-hla/src/kernel.rs` — `hla_state_update` :123 (HL-1), `hla_per_head_update`
:194 (HL-2), `ahla_step` :474 (HL-3), **plus the audit-missed fourth site**
`ahla_per_head_step` :637 (same committed `q_head.n` as `ahla_step`; found by
grepping every `simd_dot_f32` in the crate and classifying by consumer) — moved
from `simd_dot_f32` (platform lane order) to `dot_f32_ordered` (sequential fold).
They feed `q_head.h` / `q_head.n` / `mQ`-adjacent cross-terms that mutate
`hla_moments` — the Cold-tier Pod, BLAKE3-committed on freeze. Per the Issue 940
law, a reduction whose result is committed pins its reduction order.

The readout/denominator dots (:321/:322, :389/:390, :494 → logits → int token
ids) stay `simd_dot_f32` — LOCAL by the audit's classification (tokens, not
bits, cross the wire), and they are the per-token decode-loop hot path.

## G2 numbers (best-of-5 minimum ns/op, `cargo bench -p katgpt-hla --bench bench_939_hla_ordered_dot_g2`)

| arm | BEFORE (platform) | AFTER (ordered) | delta |
|---|---|---|---|
| raw dot hd=64 | simd 2.8 / ordered 15.6 | simd 3.1 / ordered 16.3 | ordered/simd ≈ 5.2–5.6× |
| raw dot hd=128 | simd 6.0 / ordered 42.7 | simd 6.2 / ordered 44.5 | **+36.7 ns/call ≈ the whole fix cost** |
| raw dot hd=256 | simd 12.0 / ordered 106.8 | simd 12.5 / ordered 111.4 | ≈ 8.9× |
| `hla_state_update` hd=128 | 4845.7 | 4794.1 | **-1.1%** (noise) |
| `ahla_step` hd=128 | 3620.1 | 3708.5 | **+2.4%** |
| layer = 32 heads × `hla_state_update` (per token per layer) | 158 897.7 | 161 700.5 | **+1.8%** |

**Box state:** macOS 26.6.2, Apple M3 Max (16 cores), AC power, battery 100%,
~46.5/64 GiB RAM free at session start; concurrent sibling-agent load present
throughout (both runs same session, same load class) — the quoted percentages
carry that load class; the load-robust best-of-5 minimum is the statistic.

The honest reading: the per-step cost of the fix is exactly one +36.7 ns dot
(hd=128), i.e. +0.76% of a `hla_state_update` step; the measured arm deltas
(-1.1% .. +2.4%) straddle it inside the box's run-to-run scatter. No arm shows
a step-function cost. G2 discharged: cross-node bit-stability of committed HLA
moments is bought for ≤ ~2% on the update path and 0% on the readout path.

## Correctness pins (the anti-regression witnesses)

- `k_mq_folds_in_index_order_not_platform_order` — the certified len-16
  cancellation craft (`[1e8,1,1,1,-1e8,1,1,1,…]` vs all-ones) lifted through
  `hla_state_update`: two steps accumulate `mQ = craft`, then `k = ones`
  exposes the dot in `h`; asserts `h[0] == 3.0` (sequential) and that the
  platform kernel reads 6.0 on the same input (non-vacuity).
- `q_mk_folds_in_index_order_not_platform_order` — the AHLA twin through
  `ahla_step` (the dot feeds `q_head.n`); the probe k rides 2^-25 (below half
  an ulp at 1.0, far below the ulp at 1e8) so `mK` stays bit-identical while
  `n[i] = 3·2^-25` pins the ordered fold bit-exactly.

Validation: katgpt-hla 18/18 lib (16 existing + 2 new), clippy `-D warnings`
(default + `--all-features`), katgpt-forward 131/131 (`forward_hla`/`forward_ahla`
finite/divergence suites), katgpt-core 2157/2157 lib default (the `katgpt_core::hla`
re-export posture).

## One-time committed-bits change

`hla_moments` `h` and `n` arrays (and any BLAKE3 freeze commitment over them,
e.g. riir-engine's Cold-tier snapshots) move at this upgrade boundary on all
platforms. Freeze/thaw versioning absorbs it; consumers re-baseline once.
SK / CQV / G / mQ are untouched (they accumulate by elementwise `+=` /
outer-product — no reduction). The sibling half (riir-engine HlaCacheProxy →
NeuronShard freeze) is riir-ai Issue 1074's confirmation lane.
