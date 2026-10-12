# Issue 935 — the Q2_0_g128 ternary matvec is the whole Bonsai-2-27B CPU decode (and compute-bound)

**Status:** OPEN — T1–T4 DONE (Bonsai-2-27B CPU decode 1.96 → 7.86 tok/s, riir-infer Bench 877, q8 PROMOTED in riir-infer; AVX2 q8 port landed, bit-identical); only T4b (native x86_64 wall on the 4090) remains.

## Finding

`sample` over the real `Ternary-Bonsai-2-27B-PQ2_0.gguf` CPU decode (riir-infer
`twt_perf_tg_pp`, M3 Max): of the busy samples, **~97% sit in
`katgpt_types::simd::ternary_group::neon_row_range_hoisted`**; everything else
(GDN step, conv, rotation, gates) is noise. The rest of the wall is rayon idle
wait (`__psynch_cvwait`).

The kernel expands every weight bit into an f32 sign and FMAs it — ~18 NEON ops
per 8 weights. At 2.125 bits/weight the memory floor is ~7 GB/token ≈ 20 ms on
the M3 Max; decode measured ~520 ms/token → compute-bound by ~25×.

## Tasks

- [x] T1 — rows-per-task capped at 32 (`PAR_MATVEC_CHUNK_ROWS`, shared by the
  bit-plane and trit tiers). One chunk per worker left the 4 E-cores finishing
  last; capped tasks let work-stealing rebalance. Bit-identical (pure row
  partition; `chunked_matches_serial_bit_identical_for_any_chunk`). Kernel
  A/B (`examples/ternary_group_par_chunk_ab.rs`): 0.67–0.72× wall on all four
  Bonsai-2 decode shapes.
- [x] T1b — real model: tg 1.96/2.03 → 2.46/2.47 tok/s, pp +32%, 65/65 greedy tokens identical (`0c5ed0fe7`).
- [x] T2 — `ternary_group_q8_act`: activations quantized once per call to int8
  per 128-group, `tbl`+`tst` bit expansion, `sdot` (~7 ops / 16 weights).
  NEON bit-identical to its scalar integer reference; ragged cols fall back to
  the f32 kernel. Opt-in (LOSSY: activation rounding — llama.cpp's CPU Q2_0
  regime). Consumer: riir-infer `ternary_q8_act` → `ternary_matvec_parallel`.
- [x] T3 — real-model GOAT gate (riir-infer `tests/ternary_q8_act_goat.rs`):
  256 + 512 positions — top-1 agree 256/256 + 510/512, confident 165/165 +
  348/348, hit-rate identical, NLL Δ −0.003 / −0.019, greedy identical, wall
  0.319 / 0.326×. PROMOTED in riir-infer (`deltanet_ternary_inference` implies
  `ternary_q8_act`); end-to-end tg 7.86 tok/s, 65/65 greedy tokens identical
  to f32 (riir-infer Bench 877). katgpt-types keeps the feature opt-in — the
  consumer owns the lossy choice.
- [x] T4 — x86_64 q8 port: `avx2_q8_row_range` (runtime-probed via
  `is_avx2_fma_available`, never a compile-time `target_feature`). Per 32
  weights: broadcast u32 per plane, `pshufb` + `and`/`cmpeq` → byte masks,
  `sub` → {-1,0,+1}, `psignb` onto xq, `pmaddubsw` vs ones → i16; the four
  i16 partials of a group (|sum| ≤ 1016) widen once via `pmaddwd`. Integer
  inner sum ⇒ **bit-identical** to `ternary_group_q8_rows_scalar` — the same
  reference the NEON arm matches, so the aarch64 real-model GOAT gate (T3)
  transfers to x86_64 exactly. Verified on the M3 under Rosetta 2 with
  `ROSETTA_ADVERTISE_AVX=1` (without it Rosetta hides AVX/AVX2/FMA in CPUID
  and the test prints `SKIP`): 21/21 ternary_group tests, new
  `saturated_extremes_match_exact_integer_sum` (every |xq| = 127, every
  weight ±1), mutation canary (`sub(pm, nm)`) reds 3/5. Clippy clean on both
  triples. Rosetta-translated kernel A/B (indicative ONLY — translated AVX2,
  not a native wall): q8 = 0.467–0.482× the shipped f32 AVX2 kernel on all
  four Bonsai-2 shapes (~0.6× vs chunk-32 f32).
- [ ] T4b — native x86_64 wall on the 4090 (`ternary_group_par_chunk_ab
  --features ternary_group_q8_act` + riir-infer `twt_perf_tg_pp`). Deferred
  only while the Plan 620 session holds that box (sync via `gitsync`
  schtask first). No promotion decision rides on it: the consumer feature is
  already default in riir-infer and the result is bit-identical to the
  gated aarch64 arm.
