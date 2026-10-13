# Issue 936 — decode-scoped spin-barrier pool for the row-parallel matvecs

**Status:** OPEN 2026-10-13 — design + prototype. The prize is measured, not the mechanism. Consumer: riir-infer Issue 045 T4.

## Why

On Bonsai-2-27B CPU decode (M3 Max, 16 rayon workers, ~100 ms/token), the
rayon workers park in the short serial gaps between forks (norms, conv1d,
gates, SwiGLU). Waking them costs ~100 µs per fork. riir-infer Issue 045
priced it: a timed no-op `rayon::broadcast` before every parallel section
costs 14.2 ms/token, and the sections then drop by ~16 ms. **About 14–16% of
each token is wake-up.**

Two in-rayon fixes were measured and refuted (riir-infer Issue 045):
- Keeping the state warm via `rayon::join` while the caller runs the serial
  gap: +2.1–2.6%.
- A per-token keepalive (each worker spins in `rayon::yield_now`): 5.7×
  slower. Spin backoff did not help, so deque contention is not the cause.
  Root cause unknown.

## Shape (llama.cpp's threadpool, adapted)

- N persistent workers, pinned to the decode for one token (or one
  generation), spinning on an epoch counter with no OS park while a token is
  in flight. Park only after an idle timeout between tokens.
- Each parallel section becomes "publish (fn, args, n_tasks); bump epoch;
  every thread including the caller takes task indices from an atomic
  counter; barrier". There is no work-stealing deque, and the static
  partition keeps it bit-identical (the matvec rows are independent).
- The kernels (`simd_ternary_q2_matvec_q8_parallel[_n]`, bit-plane q8, f32
  group matvec) take a `&dyn RowParallel`-style executor. rayon stays the
  default implementation, so every non-decode caller is unchanged.

## Tasks

- [ ] T1: substrate check (done 2026-10-13: no spin/barrier pool in
      katgpt-rs, riir-infer or riir-ai; there are 30+ `par_chunks_mut` sites in
      `katgpt-types/src/simd`). Define the executor trait in katgpt-types and
      make rayon its default impl, with no behaviour change.
- [ ] T2: spin-barrier impl behind a feature `decode_spin_pool` (opt-in).
      Microbench: fork latency after a 30 µs serial gap, rayon vs spin.
- [ ] T3: riir-infer wiring + the in-process paired real-model GOAT (one load,
      switch flipped per position; bit identity + ratio ≤ 0.95). Re-run 3×.
      Promote to default if it passes. Record power/thermal (spinning burns
      cores between sections; laptop AC vs battery is part of the claim).
