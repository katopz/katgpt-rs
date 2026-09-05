# Issue 726 — `gauge_rebalance` is 3.7x its Plan 279 target; the rank-wise accumulate is scalar

**Status:** RESOLVED — 2026-09-05 (same day). T1–T4 all executed; the swap
LANDED (G1/G2/G3 pass, no assertion loosened); `t08` re-pinned 30 → 15 µs with
provenance. Not blocking: the gate was already green at 30 µs; now green at
15 µs with the measured 9.00 µs.

## The measurement

`gauge_rebalance(256x16, 256x16, alpha=1.0)`, best-of-200 with a warm core,
M3 Max, release, 2026-09-05: **17.67 – 21.67 µs over twelve runs** at box load
6–14. Plan 279's target is **5 µs**.

*(This session's own pre-swap baseline at load 4.8: 18.96 / 19.21 / 19.21 µs —
inside the filed band.)*

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

- [x] T1 — Price the accumulate: interleaved A/B, scalar vs accumulate-stubbed
      (per-row dot + `vout[0]` lane kept, 15/16 of the accumulate removed,
      identical control flow), `tests/common/ab_timing.rs`
      `ab_median_ratio`, 9 rounds × 200 iters, 3 runs. Scalar arm 19.9 / 19.4 /
      20.1 µs (matches t08 — sanity), stub arm 6.5 / 5.4 / 5.4 µs; median
      ratios 0.2794 / 0.2787 / 0.2646 → **the accumulate is ~77–78% of the
      whole call. Dominant term CONFIRMED.**
- [x] T2 — Swapped to `simd_fused_scale_acc` (already exported at
      `katgpt_core::simd` — a re-export of `katgpt_types::simd`, research
      module; `gauge_invariant = ["katgpt-core/newton_schulz"]` already gives
      katgpt-spectral the katgpt-core dep, no Cargo.toml change). Interleaved
      median-of-ratios ×3 (9 rounds × 200 iters, box load 3.5–4.9, all 27
      rounds survived): **54.1% / 52.7% / 50.9% improvement** (medians 0.4589 /
      0.4727 / 0.4907, per-round ranges within ±5% of each median). Absolute
      t08 best-of-200: before 18.96 / 19.21 / 19.21 µs → after **9.00 / 9.00 /
      9.00 µs** (−53%). G1 (≥20% × 3 runs) PASS with ~2.5× margin.
- [x] T3 — Exactness re-read under the new rounding. Bit-compare
      (`to_bits`, scalar vs simd on identical inputs): **NOT bit-identical —
      the scalar loop was NOT FMA-contracted by LLVM** (t01-shape: 64/64 and
      48/48 elements differ; t08-shape: A 3136/4096, B 0/4096 — the σ-ratio's
      1-ULP change rounds `inv_c` to the same f32 there; max |Δ| = 1.19e-7 ≈
      1 ULP). Every exactness assertion still passes at its committed
      tolerance — full suite 20/20 (17 committed + 3 temp measurement tests
      since removed), t01's A·B^T diff now 1.2e-7 vs 1e-5 (~80× headroom);
      compose invariance vs 1e-3; neighbours re-run: bench_279 G6 PASS
      (1.19e-7 vs 1e-3), katgpt-sparse lib 39/39 (t07's home crate incl. the
      1e-4 compose-equivalence test). No tolerance touched — swap LANDED.
- [x] T4 — `t08` re-pinned **30 → 15 µs** (measured 9.00 ×3; ~1.7× the best
      sample, matching the old pin's ~1.4×-worst-sample headroom posture) with
      the full provenance block appended in the gate's doc comment (pricing,
      interleave numbers, bit-compare verdict, neighbour re-runs). 5 µs paper
      target: now a ~1.8× gap; the T1 stub arm shows the remaining floor is
      the σ-dots + the two full-matrix in-place scales + the final ‖M·v‖ pass
      (~5.4–6.5 µs at this shape), so 5 µs is NOT reachable by the accumulate
      swap alone — kept as the paper aspiration, noted in the gate doc.

## Gates

| Gate | Criterion | Verdict |
|---|---|---|
| G1 | `t08` best-of-200 improves by >= 20% over the 17.67–21.67 µs band, three runs | **PASS** — 54.1 / 52.7 / 50.9% interleaved; absolute 19.21 → 9.00 µs (−53%) |
| G2 | Every exactness assertion in `bench_270_gauge_invariant_goat` still passes at its committed tolerance | **PASS** — 17/17 committed tests green (t01 diff 1.2e-7 vs 1e-5); no tolerance changed |
| G3 | No new allocation in `power_iterate_sigma_max` (it takes its scratch by `&mut`) | **PASS** — `simd_fused_scale_acc` takes slices, zero alloc; scratch contract unchanged |

## References

- `.issues/723` T7 — the wall-clock-gate treatment that produced this number
- `tests/bench_270_gauge_invariant_goat.rs` `t08` — the gate and its provenance
- `crates/katgpt-spectral/src/gauge_invariant.rs:108` — `power_iterate_sigma_max`

## Resolution record (2026-09-05)

- Worktree change, uncommitted (session directive): `gauge_invariant.rs`
  accumulate swap + comment, `bench_270` t08 re-pin + provenance, this file.
  No other file touched (`tests/bench_sp_kv.rs` dirt in the shared checkout is
  the sibling Issue-727 session's WIP, not this issue's).
- Instrumentation: all measurement through `tests/common/ab_timing.rs`
  (`ab_median_ratio` interleaved median-of-ratios for the A/Bs, `best_of_us`
  min-of-200 for the absolute gate), release profile, isolated
  `CARGO_TARGET_DIR=/tmp/t726`, box load 3.5–4.9 (another benchmark session
  live — hence the interleaving mandate, honored).
- The A/B arms ran on the fixed-point state (no per-call operand restore):
  `gauge_rebalance`'s scale converges to c = 1 after the first call, so after
  warmup every measured call does identical work in both arms — the t08 gate
  itself keeps its restore-per-iteration discipline unchanged.
