# Issue 967: blocked two-level NEON argmax for `simd_argmax_f32`

**Status:** DONE 2026-10-27 — GOAT PASS → `argmax_blocked` DEFAULT in katgpt-types ([Bench 958](../.benchmarks/958_issue967_argmax_blocked_goat.md)).

**Origin:** Issue 966 G2 side finding: the top-16 pool head cost less than `simd_argmax_f32` at a 151 936 vocab.

## Tasks

- [x] Blocked kernel (8 independent max accumulators per 256-float block + one-block `simd_first_ge_f32` rescan; NaN slices fall back to the lane-wise kernel), behind `argmax_blocked`.
- [x] G1 bit-identity battery vs the lane-wise kernel (`bench_967_argmax_blocked_ab::g1_blocked_matches_lanewise`).
- [x] G2 interleaved A/B: 0.19–0.35 at every shape and n ≥ 512.
- [x] Promote to default; regression suites green; clippy clean on aarch64, x86_64-musl, wasm32 ± simd128.
- [x] Knock-on: Issue 966 G2 re-run against the faster baseline. Pool extraction moved to an exact block-floor path. 32k still above the bar; recorded in 966.
- [-] x86_64 port: deferred. Issue 817 measured the two-pass idiom as the x86_64 winner. Re-try a blocked AVX2 variant only on the x86_64 matrix box (the 4090), since amo cannot execute it.
