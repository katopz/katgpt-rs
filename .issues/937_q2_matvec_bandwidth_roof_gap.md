# Issue 937 — Q2 row-parallel matvec reaches ~72% of the CPU streaming roof

**Status:** OPEN 2026-10-13 — software prefetch REFUTED on both arches (T3 M3, T5 4090) and the `q2_prefetch` probe RETIRED (code removed; the kernel is byte-identical to before this issue). Lead: on the M3 the isolated kernel streamed more with fewer threads (10 → 103.5 GB/s vs 16 → 80–89, loaded box) — load-confounded, unproven at idle. The 4090 is the opposite (monotone up to 24 threads). Next: T4 on an idle M3 with `examples/q2_bandwidth_sweep.rs` + the in-model worker count.

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

- [-] T1: after Issue 936 lands, re-measure the isolated kernel with both
      executors and record GB/s per shape. Is the gap still there?
      **Rayon half done** (below): yes, 80–91 GB/s at 16 threads on a
      working set past the SLC. The spin-pool half folds into T4.
- [x] T2: thread-count split (rayon, isolated, `q2_prefetch_sweep` with
      `Q2_PF_SWEEP=0`). Measured, see §Measurements. Peak at 10 threads.
- [x] T3: prefetch-distance probe — `q2_prefetch` feature (NEON
      `prfm pldl1strm` / AVX2 `prefetchnta`), bit-identical at every
      distance (29038059f). **NEGATIVE on the M3**: best paired ratio
      0.983–0.984 once, then 1.020/1.024 on repeats at 4096 B; ≥ 16 KB a
      loss (1.03–1.27). The hardware prefetcher already covers the stream.
      With T5 also negative, the feature was RETIRED (removed, not left
      opt-in): code in 29038059f if anyone re-opens it.
- [ ] T4: in-model worker count on the decode spin pool (now on in
      riir-infer `deltanet_ternary_inference`, 12 participants on the M3):
      `KATGPT_SPIN_POOL_WORKERS` 7/9/11 on Bonsai-2-27B decode, idle M3,
      paired ratio vs 11, 3×, box state recorded. The isolated rayon peak at
      10 threads predicts 9 workers + caller; the in-model shape (gaps,
      smaller matrices) may not. Promote a new macOS default only on a
      real-model win.
- [x] T5: 4090 AVX2 prefetch sweep (i7-13700K, 24 threads, 29038059f,
      two repeats): **NEGATIVE** — 256 B 0.990/0.993 (noise), 1 KB
      1.025–1.052, 4 KB 1.074–1.077, 8 KB 1.26–1.29, 16 KB 1.44–1.46.
      `prefetchnta` bypasses the cache levels the hardware prefetcher
      would have filled.

## Measurements

Box for all rows: M3 Max (12P+4E), AC power, other sessions running (load
average 3.3–4.3, falling from ~8). `examples/q2_prefetch_sweep.rs`: six
distinct 17408×5120 Q2_0 matrices (142 MB, past the SLC) per pass, `--release`,
31 interleaved rounds, median.

T2 — threads vs streamed bandwidth, prefetch off, two repeats:

| rayon threads | rep 1 | rep 2 |
|---|---|---|
| 4 | 64.9 GB/s | 62.1 GB/s |
| 6 | 87.1 | 83.5 |
| 8 | 96.4 | 96.2 |
| **10** | **103.4** | **103.5** |
| 12 | 94.8 | 95.3 |
| 16 (default) | 80.5 | 89.4 |

The 16-thread loss is partly contention with the other sessions (≈3.5
runnable threads), but 12 (= the P-core count) also lost to 10 in both
repeats, so it is not only that. Candidate mechanism, untested: fewer
concurrent DRAM streams thrash fewer pages / fabric queues. Re-measure on an
idle box before quoting the peak.

T3 — prefetch distance vs off, 16 threads (ratios, lower is better):
256 B 1.024 · 512 B 1.034 · 1024 B 1.016 · 2048 B 0.993 · 4096 B 0.979 /
0.984 / 1.020 / 1.024 · 8192 B 0.983 / 1.015 / 1.025 · 16 KB 1.029 · 32 KB
1.095 · 64 KB 1.272.

4090 (i7-13700K, 8P+8E/24 threads, DDR5, Windows, prefetch off,
sequential processes) — threads vs streamed bandwidth: 4 → 35.1 · 6 → 47.4 ·
8 → 53.2 · 12 → 59.2 · 16 → 61.9 · **24 → 67.7 GB/s**. Monotone: the
M3's fewer-threads effect does NOT transfer to x86, so any thread cap from
T4 must stay macOS-only.

Load confound, measured: `examples/q2_bandwidth_sweep.rs` (one rayon pool
per arm, interleaved in one process) at load average 12–13 gave ratios vs
16 threads of 8 → 0.668, 10 → 0.675, 12 → 0.802, at ~14–22 GB/s. Under
contention fewer threads win big, so the loaded T2 table above over-states
the idle effect by an unknown amount. T4 runs this instrument on an idle
M3 first.
