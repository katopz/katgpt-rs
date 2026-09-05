# Issue 726 — `gauge_rebalance` is 3.7x its Plan 279 target; the rank-wise accumulate is scalar

**Status:** OPEN — filed 2026-09-05 by Issue 723 T7. Not blocking: the gate
(`bench_270_gauge_invariant_goat::t08`) is re-pinned to the measured 30 µs with
provenance and is GREEN, and the 5 µs figure is reported beside it as the paper
target it always was. This issue is the optimisation, not the tolerance.

## The measurement

`gauge_rebalance(256x16, 256x16, alpha=1.0)`, best-of-200 with a warm core,
M3 Max, release, 2026-09-05: **17.67 – 21.67 µs over twelve runs** at box load
6–14. Plan 279's target is **5 µs**.

Three things had to be fixed before that number could be read at all, and all
three are recorded in the gate's own doc comment
(`tests/bench_270_gauge_invariant_goat.rs`, `t08`):

1. The timed closure re-generated both 256x16 operands *inside* the clock.
2. `warmup = 3` did not ramp the P-core — the same gate read **47 µs run
   alone** and **18.6 µs run inside its own suite**, so the verdict depended on
   the test filter.
3. The 5 µs bar had never been executed in release at all (Issue 723's
   compile-vs-EXECUTE axis).

## The candidate

`power_iterate_sigma_max` (`crates/katgpt-spectral/src/gauge_invariant.rs:108`)
runs 5 steps per factor, 2 factors. Its inner operator is:

```rust
for i in 0..outer {
    let row = &mat[i * rank..(i + 1) * rank];
    let u_i = simd_dot_f32(row, vin, rank);   // SIMD
    for k in 0..rank {
        vout[k] += row[k] * u_i;              // SCALAR — the candidate
    }
}
```

The dot is SIMD; the rank-wise accumulate is a scalar loop. At `outer = 256`,
`rank = 16`, 5 steps, 2 factors that is **40,960 scalar FMAs** on the critical
path. `katgpt_core::simd::simd_fused_scale_acc(dst, src, scale, len)` is
exactly `dst[i] += scale * src[i]` and is already a workspace dependency of
this crate's neighbours.

## The catch — read before assuming this is free

`simd_fused_scale_acc`'s NEON backend uses `vfmaq_f32`, a **single-rounding**
FMA, while `vout[k] += row[k] * u_i` is mul-then-add — two roundings. The swap
is therefore **not bit-identical**, and the current code's own comment claims
"original behavior preserved byte-for-byte". `t01_gauge_rebalance_preserves_abt_exactly`
and the `compose` gauge-invariance tests must be re-read against their
tolerances before this lands, not after.

## Tasks

- [ ] T1 — Price the accumulate: perf-record or an A/B with the inner loop
      stubbed, to confirm it is the dominant term rather than assuming it.
- [ ] T2 — Swap to `simd_fused_scale_acc`; measure `t08` best-of-200 three
      times, interleaved against the current implementation.
- [ ] T3 — Re-read every exactness assertion in
      `bench_270_gauge_invariant_goat` under the new rounding. If any tightens
      to a real failure, the swap is refused, not the assertion loosened.
- [ ] T4 — If T2 shows a gain, re-pin `t08` down with the new provenance and
      note whether 5 µs is now reachable or should be retired as an aspiration.

## Gates

| Gate | Criterion |
|---|---|
| G1 | `t08` best-of-200 improves by >= 20% over the 17.67–21.67 µs band, three runs |
| G2 | Every exactness assertion in `bench_270_gauge_invariant_goat` still passes at its committed tolerance |
| G3 | No new allocation in `power_iterate_sigma_max` (it takes its scratch by `&mut`) |

## References

- `.issues/723` T7 — the wall-clock-gate treatment that produced this number
- `tests/bench_270_gauge_invariant_goat.rs` `t08` — the gate and its provenance
- `crates/katgpt-spectral/src/gauge_invariant.rs:108` — `power_iterate_sigma_max`
