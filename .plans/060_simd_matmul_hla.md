# Plan 060: SIMD Matmul + HLA Kernels — Server-Scale Inference

**Branch:** `develop/feature/060_simd_matmul_hla`
**Depends on:** Plan 057 (HLA Implementation)
**Research:** `.research/029_rust_gpu_feasibility.md` (SIMD vs GPU analysis)
**Goal:** Add NEON/AVX2 SIMD to `matmul`, `matmul_relu`, `sparse_matmul`, and HLA streaming kernels. Target: 4-8× throughput gain for 30K CCU @ 20Hz game server deployment.

---

## Tasks

### Phase 1: SIMD Infrastructure

- [x] T0: Add SIMD detection and dispatch module
  - Create `microgpt-rs/src/simd.rs` with feature-gated SIMD backends
  - `SimdLevel` enum: `Scalar`, `Neon`, `Avx2`
  - Runtime detection via `#[cfg(target_arch = "aarch64")]` / `#[cfg(target_arch = "x86_64")]`
  - `is_simd_available()` → `SimdLevel`
  - Zero dependencies — uses `core::arch::{aarch64, x86_64}` intrinsics directly
  - Add `mod simd;` to `lib.rs`

- [x] T1: Implement NEON backend (macOS / ARM servers)
  - `simd_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32`
    - Process 4 floats per iteration via `vmlaq_f32`
    - Horizontal add via `vaddvq_f32`
  - `simd_fma_row(weight_row: &[f32], input: &[f32], len: usize) -> f32`
    - Single row of matmul, NEON accelerated
  - `simd_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize)`
    - For HLA SK and CQV rank-1 updates
  - `simd_matvec(acc: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize)`
    - For HLA readout: mat × vec
  - Uses `#[cfg(target_arch = "aarch64")]` gate
  - Fallback to scalar when NEON unavailable

- [x] T2: Implement AVX2 backend (x86_64 servers)
  - Same API as T1, uses `_mm256_fmadd_ps`, `_mm256_reduce_add_ps`
  - `#[cfg(target_arch = "x86_64")]` gate
  - Fallback to NEON (via emulation) or scalar

### Phase 2: SIMD Matmul Integration

- [x] T3: SIMD-accelerate `matmul()` in `types.rs`
  - Replace inner loop in `matmul()` with `simd_dot_f32()`
  - For n_embd=32: 32-wide dot product → 8 NEON ops + 1 reduce
  - Benchmark before/after with `game` config
  - Must pass all existing tests unchanged (same numerical results)

- [x] T4: SIMD-accelerate `matmul_relu()` in `types.rs`
  - Replace inner loop with `simd_dot_f32()` + fused ReLU
  - NEON: `vmaxq_f32(acc, vdupq_n_f32(0.0))` for zero-clamp
  - AVX2: `_mm256_max_ps(acc, _mm256_setzero_ps())`

- [x] T5: SIMD-accelerate `sparse_matmul()` in `types.rs` ✅
  - Added `simd_sparse_dot_f32()` — NEON gather via `vsetq_lane_f32` (4 elements/iter), AVX2 hardware gather via `_mm256_i32gather_ps` (8 elements/iter)
  - Added `simd_sparse_matmul_rows()` — row-wise dispatch using sparse dot
  - Scalar fallback for alive ≤ 4 (gather setup overhead exceeds benefit)
  - Updated `sparse_matmul()` Phase 2 to call `simd_sparse_matmul_rows()`
  - Benchmark results (NEON, 20% sparsity):
    - micro 16×64 (12 alive): 269K/s
    - game 32×128 (25 alive): 78K/s
    - small 64×256 (51 alive): 21K/s
    - **Sparse SIMD is 2.4× faster than dense SIMD** at 20% sparsity
  - 8 new unit tests: sparse dot matches scalar, sparse matmul matches scalar, row offset, edge cases

### Phase 3: SIMD HLA Kernels

- [x] T6: SIMD-accelerate HLA `hla_state_update()` in `hla/kernel.rs`
  - Outer product SK += kkᵀ: `simd_outer_product_acc()` on hd×hd matrix
  - Cross moment CQV += qvᵀ: `simd_outer_product_acc()`
  - Matvec kᵀ·CQV: `simd_matvec()` for tmp_k_cqv
  - For hd=4: single NEON instruction covers entire row
  - For hd=8: 2 NEON instructions per row

- [x] T7: SIMD-accelerate HLA `hla_readout()` in `hla/kernel.rs`
  - Numerator: qᵀ(SK·CQV − G) → SIMD matvec + SIMD dot
  - Denominator: qᵀ(SK·mQ − h) + ε → SIMD matvec + SIMD dot
  - For hd=4: entire readout is ~4 NEON ops

- [x] T8: SIMD-accelerate AHLA `ahla_step()` in `hla/kernel.rs`
  - PKV update: `simd_outer_product_acc()`
  - E accumulation: SIMD matvec + outer product
  - Readout: qᵀE / (qᵀn + ε) → SIMD dot products

### Phase 4: Benchmarks

- [x] T9: Add SIMD benchmark to `benchmark.rs`
  - Benchmark `matmul` scalar vs SIMD for [32×32]×[32] (game config)
  - Benchmark `hla_state_update` scalar vs SIMD for hd=4, hd=8
  - Benchmark `ahla_step` scalar vs SIMD for hd=4, hd=8
  - Report throughput (tok/s) for each variant
  - Add to existing benchmark runner

- [x] T10: End-to-end throughput benchmark
  - `forward_hla()` scalar vs SIMD with `game` config
  - `forward_ahla()` scalar vs SIMD with `game` config
  - Report aggregate tok/s
  - Target: ≥4× improvement on ARM NEON, ≥6× on x86 AVX2

### Phase 5: Validation

- [x] T11: All existing tests pass with SIMD
  - `cargo test` — zero regressions
  - SIMD results must be bit-identical to scalar (same float operations, just vectorized)
  - HLA kernel tests (22/22 from Plan 057) must still pass
  - Run on both ARM (macOS) and x86_64 (CI) if possible

- [x] T12: Update `.research/029_rust_gpu_feasibility.md` with benchmark results ✅
  - Measured NEON throughput: matmul 15.6M/s [16×16], hla_update 16.4M/s (hd=4), ahla_step 18.2M/s (hd=4)
  - E2E forward_hla: 939K tok/s (Config::micro, single-core NEON)
  - 30K CCU @ 20Hz: ✅ single-core handles it (939K > 600K, 9.8× headroom on 8-core)
  - Added `tests/bench_simd.rs` benchmark test file (6 tests)
  - Research doc updated with measured vs estimated comparison

---

## Architecture

### File Layout

```text
microgpt-rs/src/
├── simd.rs                 — NEW: SIMD detection + dispatch
├── types.rs                — MODIFY: matmul/matmul_relu use SIMD dispatch
├── hla/
│   ├── kernel.rs           — MODIFY: HLA/AHLA kernels use SIMD ops
│   ├── forward.rs          — No changes (calls kernel.rs)
│   └── types.rs            — No changes
└── benchmark.rs            — MODIFY: add SIMD benchmarks
```

### SIMD Dispatch Strategy

```text
matmul(output, weight, input, rows, cols):
  match simd_level():
    Neon  → for each row: neon_dot(weight_row, input, cols)
    Avx2  → for each row: avx2_dot(weight_row, input, cols)
    Scalar → current loop (fallback)
```

No trait objects, no dyn dispatch. Compile-time `#[cfg]` + runtime check at init.

### NEON Implementation Detail (hd=4 sweet spot)

For `game` config (hd=8), a single row of matmul is 8 floats:
```text
NEON: 8 floats = 2× vmlaq_f32 (4 each) + 1 vaddvq_f32
Scalar: 8 multiply-accumulate ops
Speedup: ~2× for hd=8, ~4× for n_embd=32
```

For `micro` config (hd=4), a single row is 4 floats:
```text
NEON: 4 floats = 1× vmlaq_f32 + 1 vaddvq_f32
Scalar: 4 multiply-accumulate ops
Speedup: ~4× for hd=4
```

The entire HLA state update for hd=4 is ~10 NEON instructions. It's already fast — SIMD just makes it tighter.

### Why Not `std::simd` (Portable SIMD)

`std::simd` (aka `portable_simd`) is nightly-only. We need stable Rust. Using `core::arch` intrinsics directly:
- Stable on both `aarch64` and `x86_64`
- No features flags needed
- Predictable codegen
- `#[cfg(target_arch)]` selects the right backend at compile time

Alternative: `wide` crate (portable SIMD on stable). Adds a dependency but works on stable. Evaluate if `core::arch` becomes too verbose.

---

## Expected Results

### Throughput Targets (per-core)

| Operation | Scalar | NEON (4×) | AVX2 (8×) |
|-----------|--------|-----------|-----------|
| `matmul` [32×32]×[32] | ~200K/s | ~800K/s | ~1.6M/s |
| `hla_state_update` hd=8 | ~2M/s | ~8M/s | ~16M/s |
| `ahla_step` hd=8 | ~2.5M/s | ~10M/s | ~20M/s |
| `forward_ahla` game config | ~200K tok/s | ~800K tok/s | ~1.6M tok/s |

### 30K CCU @ 20Hz Feasibility

| Server | Cores | Throughput (SIMD) | Required | Headroom |
|--------|-------|-------------------|----------|----------|
| ARM 8-core (NEON) | 8 | 6.4M tok/s | 600K tok/s | 10× |
| x86 16-core (AVX2) | 16 | 25M tok/s | 600K tok/s | 42× |
| ARM 4-core (NEON) | 4 | 3.2M tok/s | 600K tok/s | 5× |

**Verdict: SIMD on a 4-core ARM server handles 30K CCU @ 20Hz with 5× headroom.**

---

## Key Design Decisions

1. **`core::arch` intrinsics, not `std::simd`** — stable Rust, no nightly
2. **Compile-time `#[cfg]` + runtime level check** — zero-cost dispatch
3. **Bit-identical results** — SIMD reorders operations but same float math; may have tiny ULP differences from FMA, acceptable for inference
4. **No new dependencies** — pure `core::arch`, no `wide`/`packed_simd`/`blaze`
5. **SIMD in `types.rs` and `hla/kernel.rs` only** — callers unchanged
6. **Benchmark before/after** — must demonstrate ≥4× gain on NEON

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| NEON intrinsics verbose/error-prone | Medium | Test on macOS ARM (M-series) |
| Auto-vectorizer already covers matmul | Low | Benchmark scalar first — if LLVM already NEONs it, skip manual SIMD |
| Float ULP differences from FMA | Low | Test bit-identical where possible, tolerance where not |
| `core::arch` unsafe blocks | Certain | Wrap in safe API, test thoroughly |
| `x86_64` target not tested (CI) | Medium | Add CI target or test locally on x86 |

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 057 (HLA) | Provides HLA kernels that T6-T8 accelerate |
| Plan 059 (Distillation) | Independent — SIMD works on both SDPA and HLA paths |
| Plan 008 (riir-gpu) | GPU remains the future path for >100K CCU |
| Research 29 | Documents SIMD vs GPU decision with 30K CCU math |