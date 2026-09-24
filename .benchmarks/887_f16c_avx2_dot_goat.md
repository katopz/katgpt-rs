# Bench 887 — AVX2+F16C kernel for `simd_dot_f16_f32` (the x86 f16 GEMV lane; found by the Issue 883 P0 calibration run)

**Status:** COMPLETE — measured **30.3× micro** (d=2304 dot: 1903 → 63 ns) · **8× end-to-end** (gemma-2-2b f16 full forward, the 883 P0 calibration workload: 1 → 8 tok/s on the i7-13700K). NOT a default-feature change (the kernel is a runtime-dispatched backend inside an always-on fn — the F16C-less x86 fallback keeps the scalar backend bit-identically; aarch64/NEON untouched).

**Box:** 4090 workstation (i7-13700K, Windows 11, AC power, idle desktop; the calibration pass running concurrently during the e2e figure — both arms equally exposed). Release profile.

## What was measured

`simd_dot_f16_f32` (`katgpt-types/src/simd/dot.rs`) had exactly ONE non-aarch64 backend: `scalar_dot_f16_f32` (4-chain `to_f32().mul_add()`). Every x86 box running the f16 weight lanes (riir-infer gemma-2 f16 decode/calibration) paid scalar speed for a lane the M3 runs in NEON. Found by the Issue 883 P0 calibration run: **1 tok/s** on gemma-2-2b-it f16, which made the dashboard's first slice infeasible.

- **The kernel** (`avx2_dot_f16_f32`): the `avx2_dot_f32` structure (4 independent accumulators, 32 elements/iter, 8-wide leftover, scalar tail) with the f16→f32 widening done in-register by `vcvtph2ps` (8 lanes/instruction) — no per-element `to_f32()` on the critical path. `#[target_feature(enable = "avx2,fma,f16c")]`, gated by a **runtime CPUID probe** (`is_f16c_available`, CPUID.1:ECX bit 29 ∧ AVX — F16C is Haswell-baseline but not triple-implied; the `is_avx2_fma_available` shape).
- **Micro A/B** (throwaway harness, d=2304 — gemma-2's n_embd, best-of-3 × 2000 iters): scalar 1903 ns/dot vs f16c-live 63 ns/dot = **30.3×**.
- **End-to-end** (vk_calibration, gemma-2-2b-it f16, seq 512): 1 tok/s → **8 tok/s** — the gap vs 30× is the GEMV being memory-bandwidth-bound (5.2 GiB weights/token read at ~40-50 GB/s effective), exactly as expected: the kernel removes the COMPUTE stall, the wall clock then rides the DRAM.
- **Correctness**: the existing `simd_dot_f16_f32` arms (len 0/4/8/13 vs the scalar reference, `simd/tests.rs`) pass **through the live kernel on this box** (the probe armed it), plus the micro harness's own relative-error check (|a−b|/|b| < 1e-3 at d=2304).

## Numerics disclosure

The kernel's accumulation order differs from the scalar fallback (8-lane parallel + horizontal reduce vs 4-chain mul_add) — the same class of per-arch difference `avx2_dot_f32` vs `scalar_dot_f32` already carries. Single-rounding per FMA is preserved. No bit-parity claim across backends; F16C-less x86 keeps the scalar backend bit-identically (probe false ⇒ old path, unchanged).

## Posture

Always-on fn, runtime-dispatched backend — the shipped_target_feature law's runtime-probe form (`simd_level()` precedent; a compile-time `target_feature = "f16c"` cfg would compile the arm to nothing on every ordinary build). No feature flag changes. Consumers: every f16 weight lane on x86_64 (riir-infer gemma-2 f16 forward + the 883 P0 calibration bin; riir-ai engine lanes that path through).
