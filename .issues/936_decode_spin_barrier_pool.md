# Issue 936 — decode-scoped spin-barrier pool for the row-parallel matvecs

**Status:** T1–T3 DONE 2026-10-13. Promoted downstream into riir-infer `deltanet_ternary_inference` (`27a4da7`): Bonsai-2-27B CPU decode ~100 → ~72 ms/token, bit-identical. The katgpt-types feature stays opt-in (a consumer installs the pool per thread). OPEN for T4/T5 (measurements only).

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

- [x] T1: substrate check (no spin/barrier pool in katgpt-rs, riir-infer or
      riir-ai). Shipped as a thread-local install (`with_spin_pool` /
      `spin_pool_run` / `spin_row_chunks`) instead of an executor trait: the
      kernels take a pool arm when one is installed, so no kernel signature
      changed (`bb3b24f2e`, `65574c5d9`).
- [x] T2: `SpinPool` behind `decode_spin_pool`. Tasks are claimed by CAS on one
      `epoch|n|idx` word and retired per task. Workers spin only inside an open
      scope, park after a 100 µs idle window, and park when no scope is open
      (`9c4f7f9dd`). Sizing: P-cores − 1 on macOS; 0 (= rayon) elsewhere.
      `examples/spin_pool_gap_ab` (FFN gate+up after a 0–100 µs gap), 11
      workers vs rayon on a worker: 0.60–0.84. 15 workers swung 0.84–1.29;
      7 were bandwidth-short.
- [x] T3: riir-infer `on_decode_pool` + pool arms for the GDN recurrence and
      attention heads + `tests/decode_spin_pool_goat.rs` (riir-infer `27a4da7`,
      Bench 879 §Decode spin pool). Bit-identical in every run. 100 µs window:
      0.679–0.734 plain, 0.794–0.822 beside a 6-thread burner. Spin-forever was
      1.22–1.35× SLOWER under contention; the idle window fixed it. Promoted.
      Power/thermal: AC only, not measured on battery.
- [x] T3b: macOS worker count, measured in-model (katgpt-rs-21, Issue 937 T4,
      closed `c47a1dffd`). `decode_spin_pool_goat`, 4 rounds with the order
      rotated, load 8.8–12.2, 0 mismatches in all 16 runs. Pool ms/token: 11 →
      69.5–71.8, 10 → 72.1–73.1, 9 → 73.4–75.5, 7 → 82.9–85.2. 11 won every
      round. The isolated kernel's 10-thread bandwidth peak does not carry into
      the model. Default stays P-cores − 1 = 11. Paired ratio at 11: 0.633–0.682.
- [ ] T4: off macOS. 4090 i7-13700K microbench, before the idle window, by two
      sessions: spin/rayon at the decode posture 0.97–0.99 (7 workers), 1.71–1.75
      (11), 2.55 (15), 3.86–3.93 (23); mechanism not established. The default
      there is 0 workers. Re-measure WITH the idle window and the real-model gate
      before considering a non-zero default.
- [ ] T5: GPU-hooked decode (the pool is installed there too; workers park
      after 100 µs idle while the GPU works). Measure one run, Metal lane.
