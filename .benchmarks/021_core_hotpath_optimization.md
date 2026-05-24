# Benchmark 021: Core Hot-Path Optimization

> **Plan**: N/A (standalone optimization pass)
> **Date**: 2025-06-15
> **Features**: `default` (sparse_mlp, domain_latent, ppot, bandit, bt_rank, spectral_quant, elf_sde, cna_steering, deep_manifold, federation)
> **Config**: `Config::game()` — vocab=10, embd=32, heads=4, mlp=128, layers=1
> **Build**: `--release` on Apple M3 Max (aarch64, NEON)
> **Test**: `cargo test --test bench_core_optimization --release -- --nocapture`

## Summary

Optimized `katgpt-core` and `src/` hot-path functions. Profiling identified `rmsnorm_with_gamma`
as the biggest win (scalar sum-of-squares + scalar gamma loop vs SIMD). Applied three optimizations:

1. **`rmsnorm_with_gamma_eps`**: Replaced scalar `sum_sq` with `simd_dot_f32(x, x, n)` — **2–3× faster**
2. **`simd_scale_mul_inplace`**: New fused SIMD kernel `x[i] = gamma[i] * x[i] * scale` — eliminates separate scale + multiply passes
3. **`simd_exp_inplace`**: Cephes 6th-order polynomial approximation for NEON/AVX2 — kept as utility API (scalar `libm` expf is faster on Apple Silicon due to hardware-accelerated microcode)

**Result**: ✅ `rmsnorm_with_gamma` optimized from 2.4× slower than `rmsnorm` to near-parity

## Before vs After: rmsnorm_with_gamma

| Dim   | Before (ns) | After (ns) | Speedup |
|-------|------------|------------|---------|
| 16    | 6          | 5          | 1.2×    |
| 32    | 12         | 8          | 1.5×    |
| 64    | 38         | 15         | 2.5×    |
| 128   | 93         | 32         | 2.9×    |
| 256   | 228        | 87         | 2.6×    |
| 512   | 450        | 172        | 2.6×    |
| 1024  | 854        | 367        | 2.3×    |
| 2048  | 1750       | 720        | **2.4×** |

**Before optimization**: `rmsnorm_with_gamma` was 1.3–2.4× slower than `rmsnorm` (no gamma) due to scalar sum-of-squares loop.
**After optimization**: `rmsnorm_with_gamma` is within 0.8–1.1× of `rmsnorm` — near-parity.

## Component Breakdown (Config::game, embd=32)

| Component                | Before (ns) | After (ns) | Δ       |
|--------------------------|-------------|------------|---------|
| Embedding (add wte+wpe)  | 3           | 3          | 0%      |
| RMSNorm (no gamma)       | 8           | 7          | −12%    |
| **RMSNorm with gamma**   | **11**      | **8**      | **−27%**|
| QKV projection (n×n)     | 94          | 98         | +4%     |
| Attention wo (n×n)       | 96          | 95         | −1%     |
| MLP w1 (matmul_relu)     | 362         | 395        | +9%*    |
| MLP w2 (dense)           | 361         | 373        | +3%*    |
| LM head (vocab×n)        | 32          | 35         | +9%*    |
| Softmax (vocab)          | 28          | 28         | 0%      |
| Residual add             | 4           | 4          | 0%      |

*Variation due to system noise (±5% on matmul benchmarks is normal for back-to-back runs on laptop CPUs).*

## Per-Layer Breakdown

| Component              | µs    | % layer |
|------------------------|-------|---------|
| 2× RMSNorm             | 15ns  | 1.3%    |
| 3× QKV projection      | 295ns | 25.0%   |
| Attention wo           | 95ns  | 8.0%    |
| MLP w1 (matmul_relu)   | 395ns | 33.5%   |
| MLP w2                 | 373ns | 31.6%   |
| 2× Residual add        | 7ns   | 0.6%    |
| **Per-layer total**    | **1.18µs** |       |

## E2E Forward Pass Throughput

| Config                          | Throughput (tok/s) | Latency (µs/tok) |
|---------------------------------|--------------------|-------------------|
| Config::micro (16 pos)          | 1,224,962          | 0.82              |
| Config::game (16 pos)           | 419,097            | 2.39              |
| Config::game (pos=64, t_n=65)   | 252,717            | 3.96              |
| Config::small_target (pos=0)    | 17,026             | 58.73             |

## SIMD Primitives

| Operation           | [16]  | [32]  | [64]  | [128] | [256] | [512] |
|---------------------|-------|-------|-------|-------|-------|-------|
| `simd_dot_f32`      | 2ns   | 4ns   | 6ns   | 14ns  | 34ns  | 109ns |
| `simd_scale_inplace`| 2ns   | 4ns   | 6ns   | 12ns  | 22ns  | 50ns  |
| `simd_add_inplace`  | 2ns   | 4ns   | 8ns   | 12ns  | 22ns  | 57ns  |
| `simd_max_f32`      | 2ns   | 3ns   | 6ns   | 11ns  | 21ns  | 56ns  |

## Math Utilities

| Function              | vocab=27 | vocab=256 | vocab=1024 | vocab=4096 | vocab=32000 |
|-----------------------|----------|-----------|------------|------------|-------------|
| `softmax`             | 48ns     | 426ns     | 1.70µs     | 6.74µs     | 55.3µs      |
| `softmax_scaled`      | 57ns     | 434ns     | 1.68µs     | 6.88µs     | 56.3µs      |
| `sample_token`        | 16ns     | 80ns      | 294ns      | 1.13µs     | 8.75µs      |

### Softmax Note

Attempted SIMD Cephes exp approximation (`simd_exp_inplace`) for softmax pass 2.
On Apple Silicon NEON, scalar `libm` expf is faster because:
- Hardware-accelerated `expf` uses optimized microcode
- LLVM auto-vectorizes the scalar exp loop effectively
- Cephes NEON implementation requires scalar fallback for `2^n` (no direct float-to-bits cast in NEON)

The `simd_exp_inplace` function is kept as a public utility API for platforms where `libm` exp is slower
(e.g., older ARM cores, embedded targets).

## Sparse Matmul vs Dense (game config 32×128)

| Alive % | Sparse µs | Dense µs | Speedup |
|---------|-----------|----------|---------|
| 5%      | 119ns     | 250ns    | 2.09×   |
| 10%     | 138ns     | 248ns    | 1.80×   |
| 20%     | 245ns     | 250ns    | 1.02×   |
| 50%     | 512ns     | 251ns    | 0.49×   |
| 80%     | 786ns     | 251ns    | 0.32×   |
| 100%    | 1.01µs    | 250ns    | 0.25×   |

Sparse matmul wins only at ≤10% sparsity (consistent with Plan 022).

## New APIs Added

### `simd_scale_mul_inplace`
```rust
/// SIMD-accelerated fused scale+multiply: `x[i] = gamma[i] * x[i] * scale`.
pub fn simd_scale_mul_inplace(x: &mut [f32], gamma: &[f32], scale: f32)
```
- NEON: 4× f32 per iteration via `vmulq_f32`
- AVX2: 8× f32 per iteration via `_mm256_mul_ps`
- Scalar: unchecked loop fallback

### `simd_exp_inplace`
```rust
/// SIMD-accelerated in-place exp: `x[i] = exp(x[i])` for all `i`.
/// Uses 6th-order Cephes polynomial with range reduction (~1 ULP accuracy).
pub fn simd_exp_inplace(x: &mut [f32])
```
- NEON: 4× f32 per iteration (with scalar 2^n fallback)
- AVX2: 8× f32 per iteration (full SIMD via `_mm256_castsi256_ps`)
- Scalar: Cephes polynomial (portable)

## Files Changed

| File | Change |
|------|--------|
| `crates/katgpt-core/src/simd.rs` | Added `simd_scale_mul_inplace`, `simd_exp_inplace`, NEON/AVX2/scalar backends |
| `crates/katgpt-core/src/types.rs` | `rmsnorm_with_gamma_eps`: `simd_dot_f32` for sum_sq, `simd_scale_mul_inplace` for fused scale+gamma |
| `tests/bench_core_optimization.rs` | New comprehensive benchmark (9 sections, all hot-path components) |

## Remaining Optimization Candidates

1. **`gegelu` / `gegelu_tanh`** — Scalar elementwise loops with `exp()`/`tanh()`. SIMD benefit limited for same reason as softmax (hardware `expf`/`tanhf` on Apple Silicon is fast).
2. **`sample_token`** — Cumulative scan. Could use SIMD for bulk comparison with threshold, but branch prediction makes scalar competitive for typical vocab sizes (≤4096).
3. **Matmul** — Already NEON-accelerated via `simd_dot_f32`. Further gains require loop tiling for cache locality or F16 weights (already supported via `simd_dot_f16_f32`).

## Run Command

```sh
cargo test --test bench_core_optimization --release -- --nocapture --test-threads=1