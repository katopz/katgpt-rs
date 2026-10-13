# Issue 937 — Q2 row-parallel matvec reaches ~72% of the CPU streaming roof

**Status:** OPEN 2026-10-13 — priced from existing measurements only; no new run yet. Sequenced AFTER Issue 936, because the fork-join overhead it removes is inside this gap.

## Why

On Bonsai-2-27B CPU decode, the 2-bit projections are ~90% of the token
(riir-infer Bench 879: FFN 54.2, `dn_input_proj` 15.8, `dn_out_proj` 8.6,
`attn_qkv` 4.2, `attn_out_proj` 3.4, LM head 3.3 of 100.9 ms/token). Each
token streams about 7 GB of weights. From the numbers already recorded:

| where | effective weight bandwidth | source |
|---|---|---|
| measured DRAM streaming roof, M3 Max CPU | ~125 GB/s | Bench 879 §Kernel |
| `simd_ternary_q2_matvec_q8_parallel`, 17408×5120, isolated, 16 threads | ~90 GB/s (23.7 MB in 263 µs) | Bench 879 §Kernel |
| same kernel, single thread | ~17.7 GB/s (1341 µs) | Bench 879 §Kernel |
| whole decode, in-model | ~70 GB/s (7 GB in ~100 ms) | Bench 879 §Section breakdown |

12 P-cores × 17.7 GB/s single-thread is well past the roof, so the kernel
should be bandwidth-bound at ~125 GB/s, yet it stops at ~90 GB/s isolated.
At the roof the projections would cost ~56 ms/token instead of ~90.

The in-model loss (70 vs 90) is mostly the parked-worker wake-up, which
Issue 936 targets. The isolated loss (90 vs 125) is not explained yet.

## Candidate causes (none measured)

- Fork-join and tail cost per call: 32-row tasks over 544 chunks, with the
  E-core chunks finishing last. A spin pool (Issue 936) changes this term,
  so re-measure after it lands before blaming the kernel.
- Prefetch depth: the row loop may not keep enough loads in flight per core
  to cover DRAM latency. Test with explicit `prfm` a few blocks ahead,
  bit-identical by construction.
- Activation working set: `xq`/`xperm` are re-read for every row. They
  should stay L1/L2-resident at these widths; check this first.

## Tasks

- [ ] T1: after Issue 936 lands, re-measure the isolated kernel with both
      executors and record GB/s per shape. Is the gap still there?
- [ ] T2: if a gap remains above ~10%, sweep the thread count (8/12/16) and
      the P-only placement to split it into scheduling vs per-core
      throughput.
- [ ] T3: prefetch-distance probe in `q2_row_range`, behind a feature,
      gated on bit-identity plus a paired ratio ≤ 0.95 on the real model.
      Re-run 3×, record box state (AC, load, power mode).
