# Issue 935 — the Q2_0_g128 ternary matvec is the whole Bonsai-2-27B CPU decode (and compute-bound)

**Status:** OPEN — T1 (capped rows-per-task) DONE, bit-identical; T2 (int8-activation SDOT kernel) landed opt-in, real-model gate pending.

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
- [ ] T1b — real-model tg tok/s before/after T1.
- [x] T2 — `ternary_group_q8_act`: activations quantized once per call to int8
  per 128-group, `tbl`+`tst` bit expansion, `sdot` (~7 ops / 16 weights).
  NEON bit-identical to its scalar integer reference; ragged cols fall back to
  the f32 kernel. Opt-in (LOSSY: activation rounding — llama.cpp's CPU Q2_0
  regime). Consumer: riir-infer `ternary_q8_act` → `ternary_matvec_parallel`.
- [ ] T3 — real-model GOAT gate (riir-infer `tests/ternary_q8_act_goat.rs`):
  top-1 agreement, confident-conditional agreement, NLL Δ, hit-rate Δ,
  interleaved wall ratio. Promote the riir-infer feature to default if it holds.
