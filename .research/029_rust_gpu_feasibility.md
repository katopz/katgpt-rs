# Research 29: rust-gpu Feasibility — WGSL → Rust Compute Shader Migration

**Date:** 2025-07
**Status:** Verdict Complete
**Scope:** Can `riir-gpu` replace 27 WGSL kernels with rust-gpu compute shaders?

## Context

We maintain **27 WGSL compute shaders** in `riir-gpu/src/kernels/` and **parallel CPU implementations** in `microgpt-rs/src/types.rs`. The duplication is a maintenance burden. rust-gpu (Rust→SPIR-V compiler) promises to eliminate it by compiling the same Rust code to both GPU and CPU.

The question: **Is rust-gpu production-ready for our use case today?**

## What rust-gpu Does

rust-gpu is a `rustc` compiler backend (`rustc_codegen_spirv`) that compiles Rust to SPIR-V. The resulting SPIR-V loads into wgpu via naga, the same pipeline our WGSL shaders use:

```text
Current:  WGSL source → wgpu/naga → Metal/Vulkan/DX12
Proposed: Rust source → rust-gpu → SPIR-V → wgpu/naga → Metal/Vulkan/DX12
```

The project provides:
- `spirv-std` — GPU intrinsics, `#[spirv(compute(threads(N)))]` attribute, storage buffers, workgroup memory
- `spirv-builder` — Build pipeline integration via `build.rs`
- Example runners — wgpu, ash (Vulkan), CPU (rayon), WASM

## Proof of Concept: matmul.wgsl → Rust

### Current WGSL Implementation

Our tiled matmul in `riir-gpu/src/kernels/matmul.wgsl`:

```wgsl
var<workgroup> tile_a: array<f32, 256>;  // 16x16 tile of A
var<workgroup> tile_b: array<f32, 256>;  // 16x16 tile of B

@group(0) @binding(0) var<storage, read>        a_data: array<f32>;
@group(0) @binding(1) var<storage, read>        b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write>  c_data: array<f32>;
@group(0) @binding(3) var<uniform>              params: MatmulParams;

struct MatmulParams {
    m: u32, n: u32, p: u32,
}

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    let local_row = lid.x;
    let local_col = lid.y;
    let valid_output = row < params.m && col < params.p;
    var sum: f32 = 0.0;
    let num_tiles = (params.n + 15u) / 16u;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 16u + local_col;
        if (row < params.m && a_col < params.n) {
            tile_a[local_row * 16u + local_col] = a_data[row * params.n + a_col];
        } else {
            tile_a[local_row * 16u + local_col] = 0.0;
        }
        let b_row = t * 16u + local_row;
        if (b_row < params.n && col < params.p) {
            tile_b[local_row * 16u + local_col] = b_data[b_row * params.p + col];
        } else {
            tile_b[local_row * 16u + local_col] = 0.0;
        }
        workgroupBarrier();
        if (valid_output) {
            for (var k = 0u; k < 16u; k = k + 1u) {
                sum = sum + tile_a[local_row * 16u + k] * tile_b[k * 16u + local_col];
            }
        }
        workgroupBarrier();
    }
    if (valid_output) {
        c_data[row * params.p + col] = sum;
    }
}
```

### Hypothetical rust-gpu Port

Based on the `reduce` example pattern (which uses workgroup shared memory + barriers):

```rust
// gpu-shaders/src/matmul.rs
#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::UVec3;
#[cfg(target_arch = "spirv")]
use spirv_std::memory::Scope;
use spirv_std::spirv;

#[repr(C)]
pub struct MatmulParams {
    pub m: u32,
    pub n: u32,
    pub p: u32,
}

#[spirv(compute(threads(16, 16, 1)))]
pub fn matmul_tiled(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a_data: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b_data: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c_data: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] params: &MatmulParams,
    #[spirv(workgroup)] tile_a: &mut [f32; 256],
    #[spirv(workgroup)] tile_b: &mut [f32; 256],
) {
    let row = gid.x;
    let col = gid.y;
    let local_row = lid.x;
    let local_col = lid.y;
    let valid_output = row < params.m && col < params.p;

    let mut sum: f32 = 0.0;
    let num_tiles = (params.n + 15) / 16;

    for t in 0..num_tiles {
        let a_col = t * 16 + local_col;
        let a_idx = local_row * 16 + local_col;
        tile_a[a_idx] = if row < params.m && a_col < params.n {
            a_data[(row * params.n + a_col) as usize]
        } else {
            0.0
        };

        let b_row = t * 16 + local_row;
        tile_b[a_idx] = if b_row < params.n && col < params.p {
            b_data[(b_row * params.p + col) as usize]
        } else {
            0.0
        };

        spirv_std::arch::workgroup_memory_barrier_with_group_sync();

        if valid_output {
            for k in 0..16u32 {
                sum += tile_a[(local_row * 16 + k) as usize]
                     * tile_b[(k * 16 + local_col) as usize];
            }
        }

        spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    }

    if valid_output {
        c_data[(row * params.p + col) as usize] = sum;
    }
}

// CPU fallback — runs on native target with rayon
#[cfg(not(target_arch = "spirv"))]
pub fn matmul_cpu(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += unsafe { *weight.get_unchecked(row_off + c) }
                 * unsafe { *input.get_unchecked(c) };
        }
        unsafe { *output.get_unchecked_mut(r) = sum; }
    }
}
```

### Port Complexity Assessment

| Feature in WGSL | rust-gpu Equivalent | Status |
|---|---|---|
| `@compute @workgroup_size(16,16,1)` | `#[spirv(compute(threads(16, 16, 1)))]` | ✅ Direct mapping |
| `@builtin(global_invocation_id)` | `#[spirv(global_invocation_id)]` | ✅ Direct mapping |
| `@builtin(local_invocation_id)` | `#[spirv(local_invocation_id)]` | ✅ Direct mapping |
| `var<storage, read>` | `&[f32]` + `#[spirv(storage_buffer)]` | ✅ Direct mapping |
| `var<storage, read_write>` | `&mut [f32]` + `#[spirv(storage_buffer)]` | ✅ Direct mapping |
| `var<uniform>` | `&Struct` + `#[spirv(uniform)]` | ✅ Direct mapping |
| `var<workgroup>` shared memory | `&mut [f32; N]` + `#[spirv(workgroup)]` | ✅ Works (see reduce example) |
| `workgroupBarrier()` | `spirv_std::arch::workgroup_memory_barrier_with_group_sync()` | ✅ Direct mapping |
| Bounds checking on `[]` indexing | Must use `as usize` carefully, no panics in GPU | ⚠️ Guard with `if` before access |
| `vec3<u32>` | `spirv_std::glam::UVec3` | ✅ Via glam re-export |

**Verdict on port complexity:** The matmul port is a ~1:1 translation. Most WGSL constructs have direct `spirv-std` equivalents. The reduce example proves workgroup shared memory + barriers work.

## Build Pipeline Analysis

### Current WGSL Pipeline (Zero Friction)

```text
include_str!("matmul.wgsl") → wgpu::ShaderModuleDescriptor → pipeline
```

Build time: **instant** (string include at compile time).

### rust-gpu Build Pipeline

```text
1. rust-toolchain.toml pins nightly-2026-04-11
2. Shader crate compiled with -Zcodegen-backend=rustc_codegen_spirv
3. spirv-builder runs cargo rustc with SPIR-V target
4. SPIR-V .spv file written to target/
5. Host crate loads .spv via std::fs::read → wgpu::ShaderSource::SpirV
```

From the `spirv-builder` source, the build invokes:

```text
cargo rustc --lib \
    --message-format=json-render-diagnostics \
    -Zbuild-std=core \
    -Zbuild-std-features=compiler-builtins-mem \
    --target spirv-unknown-vulkan1.2 \
    --crate-type dylib
```

This requires:
- **Nightly Rust** (pinned to specific version, currently `nightly-2026-04-11`)
- **`-Zbuild-std`** — rebuilds core/std for SPIR-V target
- **Custom toolchain** — `rustc_codegen_spirv` dylib must be in `LD_LIBRARY_PATH`
- **Build isolation** — shader crate must be a separate crate with `#![no_std]` on SPIR-V target

### Build Time Impact

| Step | Time Estimate | Notes |
|---|---|---|
| First build (shader crate) | 30-60s | Builds `core` for SPIR-V target via `-Zbuild-std` |
| Incremental shader build | 5-15s | Only shader crate recompiled |
| Host crate build | Unchanged | Loads pre-built `.spv` file |
| CI cold build | +60-120s | SPIR-V backend + build-std cache miss |

For comparison, our current WGSL pipeline adds **zero** build time.

## Kernel-by-Kernel Port Feasibility

| Kernel | Lines | Shared Memory | Barriers | Subgroups | Port Difficulty |
|--------|-------|---------------|----------|-----------|----------------|
| `matmul.wgsl` | 67 | ✅ 2× 256 f32 | ✅ | ❌ | Easy |
| `matmul_transpose.wgsl` | ~70 | ✅ | ✅ | ❌ | Easy |
| `elementwise.wgsl` | ~80 | ❌ | ❌ | ❌ | Trivial |
| `scale.wgsl` | ~20 | ❌ | ❌ | ❌ | Trivial |
| `softmax.wgsl` | ~50 | ❌ | ✅ | ❌ | Medium (row-wise reduce) |
| `layernorm.wgsl` | ~40 | ❌ | ❌ | ❌ | Trivial |
| `embedding.wgsl` | ~30 | ❌ | ❌ | ❌ | Trivial |
| `attention_score.wgsl` | ~80 | ❌ | ✅ | ❌ | Medium |
| `attention_qkv.wgsl` | ~60 | ❌ | ❌ | ❌ | Easy |
| `lora.wgsl` (a+b) | ~80 | ❌ | ❌ | ❌ | Trivial |
| `loss_per_sample.wgsl` | ~40 | ❌ | ❌ | ❌ | Trivial |
| `optimizer.wgsl` | ~50 | ❌ | ❌ | ❌ | Trivial |
| `attention_backward.wgsl` | ~120 | ❌ | ✅ | ❌ | Medium |
| `flashprefill_*.wgsl` | ~300 | ✅ | ✅ | ❌ | Hard (4 kernels) |
| `dpo_*.wgsl` | ~100 | ❌ | ❌ | ❌ | Easy |
| **Total** | **~27 files** | | | | |

**20/27 kernels are trivial or easy.** The 4 flashprefill kernels are the hardest due to multi-stage sparse attention with online softmax.

## Critical Blockers

### Blocker 1: Nightly Rust Requirement

rust-gpu requires a **pinned nightly** toolchain (`nightly-2026-04-11` at time of research). Our workspace uses **stable Rust**. Mixing nightly (shader crate) and stable (host crate) requires:

```text
riir-ai/
├── crates/
│   ├── gpu-shaders/          # NEW: nightly-only shader crate
│   │   ├── rust-toolchain.toml  # channel = "nightly-2026-04-11"
│   │   └── src/
│   │       └── matmul.rs
│   ├── riir-gpu/             # EXISTING: stable host crate
│   │   └── build.rs          # invokes spirv-builder on gpu-shaders/
│   └── ...
```

This **bifurcates the toolchain** — shader developers must use nightly, host developers use stable. CI must install both.

### Blocker 2: `spirv-std` API Gaps

From the `spirv-std` source and examples:

| Feature | Status | Impact |
|---|---|---|
| Storage buffer read/write | ✅ Works | Core functionality |
| Workgroup shared memory | ✅ Works (reduce example) | Needed for tiled matmul |
| Barrier / sync | ✅ Works | Needed for all shared-mem kernels |
| `#[spirv(uniform)]` structs | ✅ Works | Needed for param passing |
| Subgroup operations | ⚠️ Requires raw `asm!` | Needed if we add subgroup optimize |
| Dynamic indexing into storage buffers | ⚠️ Needs `RuntimeArray` | May need workarounds |
| `debug_printf` | ⚠️ Requires Vulkan validation layers | Debug only, not portable |

### Blocker 3: No CPU Fallback in Practice

The rust-gpu examples have a **CPU runner**, but it works by **calling the same functions directly** on CPU — not by compiling the shader to CPU. From the CPU runner source:

```rust
// CPU runner calls shader functions directly via rayon
let color = shader_module::fs(&push_constants, frag_coord, 100);
```

This means:
- The shader functions must be written to work on **both** `target_arch = "spirv"` and native
- All GPU-specific APIs (`spirv_std::arch::*`) must be `#[cfg(target_arch = "spirv")]` gated
- **We still maintain two code paths** — the CPU version and the GPU version are just in the same file, conditionally compiled

This is **not** the "write once, run anywhere" dream. It's "write once, maintain two `#[cfg]` branches."

### Blocker 4: Project Maturity

From the README:

> "This project is still heavily in development and is at an early stage."
> "We make no guarantees about backwards compatibility."
> "Currently only support building from source with the latest main branch."

Key signals:
- **v0.10.0-alpha.1** — still pre-release
- Embark handed off maintenance to community in 2024
- Pinning to specific nightly: `nightly-2026-04-11`
- 188 open issues
- SPIR-V output still needs naga to translate for wgpu (extra compilation step)

### Blocker 5: WASM Compatibility Uncertain

Our Plan 008 targets WASM (WebGPU). rust-gpu's WASM story:

- The examples have a `run-wasm` runner, but it uses `wgpu`'s SPIR-V passthrough
- **WebGPU does not accept SPIR-V natively** — it only accepts WGSL
- wgpu on web converts SPIR-V → WGSL via naga, but this adds latency at shader module creation
- The `spirv-builder` uses `-Zbuild-std` which may not work with `wasm32-unknown-unknown`
- No tested WASM compute shader example exists in the repo (only graphics shaders)

## Code Sharing Analysis

The core promise of rust-gpu is **eliminating CPU/GPU code duplication**. Let's check how much we actually duplicate:

### Current Duplication

| Operation | CPU (`types.rs`) | GPU (`.wgsl`) | Lines duplicated |
|---|---|---|---|
| `matmul` | ~12 lines | 67 lines | Different algo (tiled vs naive) |
| `rmsnorm` | ~15 lines | ~40 lines | Similar logic |
| `softmax` | ~20 lines | ~50 lines | Different algo (online vs two-pass) |
| `embedding` | ~5 lines | ~30 lines | Simple lookup |
| `matmul_relu` | ~15 lines | N/A (fused in matmul) | — |

**Key insight:** The CPU and GPU implementations are **not identical** — GPU uses tiled matmul, shared memory barriers, online softmax. CPU uses simple loops. rust-gpu does **not** eliminate this difference — you still write different code paths for GPU parallelism vs CPU sequential.

The "shared code" would be limited to:
- Param structs (`MatmulParams`, `SoftmaxParams`) — already shared via `bytemuck`
- Pure math functions (element-wise ops) — trivially duplicated, not worth a new build pipeline

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Nightly breaks rust-gpu | High | Build fails, blocked | Pin nightly, delay updates |
| spirv-std API gap blocks kernel | Medium | Kernel can't be ported | Keep WGSL fallback |
| CI time doubles | Certain | Slower iteration | Cache SPIR-V build |
| WASM target broken | High | Plan 008 blocked | Stay with WGSL for WASM |
| Community abandons rust-gpu | Medium | No fixes, no updates | Lock to last working commit |
| Debug shader issues | High | Can't use wgpu error messages | Use debugPrintf with Vulkan SDK |

## Decision Matrix

| Factor | WGSL (Current) | rust-gpu (Proposed) | Weight |
|---|---|---|---|
| Build complexity | ⭐⭐⭐⭐⭐ instant | ⭐⭐ nightly + build-std + spirv-builder | **High** |
| Maintenance burden | ⭐⭐⭐ two codebases | ⭐⭐⭐ still two #[cfg] branches | Medium |
| Code sharing | ⭐⭐ duplicated | ⭐⭐⭐ same crate, different arch | Low |
| Debugging experience | ⭐⭐⭐⭐ wgpu errors | ⭐⭐ SPIR-V opaque, needs Vulkan SDK | **High** |
| WASM support | ⭐⭐⭐⭐⭐ native WGSL | ⭐⭐ SPIR-V→naga→WGSL, untested | **High** |
| Project stability | ⭐⭐⭐⭐⭐ wgpu stable | ⭐⭐ alpha, pinned nightly | **High** |
| CI integration | ⭐⭐⭐⭐⭐ cargo test | ⭐⭐ nightly + spirv-builder + cache | **High** |
| Developer experience | ⭐⭐⭐⭐⭐ any toolchain | ⭐⭐ must use pinned nightly | **High** |

## Verdict: **DEFER — Do Not Adopt rust-gpu Now**

### Recommendation

**Stay with WGSL.** The port is technically feasible (most kernels translate 1:1), but the practical costs outweigh benefits:

### Why Not Now

1. **Build complexity is the killer.** Requiring a pinned nightly toolchain + `-Zbuild-std` + `spirv-builder` for 27 shaders that currently load as string includes is an unacceptable tradeoff for our small team.

2. **No real code sharing gain.** GPU kernels use tiled algorithms with shared memory and barriers. CPU uses simple loops. These are fundamentally different — rust-gpu doesn't unify them, it just puts them in the same file behind `#[cfg(target_arch)]`.

3. **WASM target is unproven.** Our Plan 008 design requires WASM+WebGPU. rust-gpu's WASM compute story is untested. WGSL is the native WebGPU shading language — zero friction.

4. **Project is pre-release alpha.** Pinning production code to `v0.10.0-alpha.1` with a specific nightly is a maintenance time bomb.

### When to Revisit

Re-evaluate if **all** of these conditions are met:

- [ ] rust-gpu reaches **v1.0 stable** with backward compatibility guarantees
- [ ] **Stable Rust** support (no nightly requirement)
- [ ] **WASM compute shader** example tested and documented
- [ ] Our shader count grows beyond 50+ (duplication pain exceeds build complexity)
- [ ] We need **shared CPU/GPU algorithm** code (e.g., for testing GPU against CPU reference)

### What to Do Instead

1. **Keep WGSL for GPU shaders** — it works, it's stable, it's native to wgpu/WebGPU
2. **Keep CPU implementations in `types.rs`** — they serve a different purpose (inference, not training)
3. **Share param structs via `bytemuck`** — already doing this
4. **Test GPU↔CPU parity** — run same inputs through both, compare outputs (already done in `riir-gpu` tests)
5. **Monitor rust-gpu** — check quarterly for stability milestones

### Cost of Deferral

If rust-gpu reaches v1.0 stable on stable Rust, migration cost is:
- ~2-3 days to port 27 kernels (most are trivial)
- ~1 day to set up `spirv-builder` build pipeline
- ~1 day to test all GPU↔CPU parity

Total: **~1 week** when conditions are right. Not a big regret.

## Post-Script: SIMD vs GPU for Inference (2025-07 Update)

> Re-evaluated inference compute path for 30K concurrent game AI users at 20Hz tick rate.

### Scenario: 30K CCU × 20Hz MMORPG

The deployment target is server-side AI for a real-time game:
- **30K concurrent players** each needing AI decisions
- **20 ticks/second** (50ms tick budget)
- **600K forward passes/second** aggregate throughput required
- Config: `game()` — vocab=10, n_embd=32, head_dim=8, mlp_hidden=128

### Throughput Math

#### Estimated (pre-implementation)

| Config | Single-core scalar | Single-core SIMD (4×) | 8-core SIMD | GPU Batched |
|--------|-------------------|----------------------|-------------|-------------|
| `micro` (hd=4, n=16) | 863K/s | 3.4M/s | 27M/s | ~50M/s |
| `game` (hd=8, n=32) | ~200K/s | ~800K/s | ~6.4M/s | ~50M/s |

#### Measured (Plan 060 T12, NEON, Apple M-series ARM, release build)

**Kernel-level throughput:**

| Operation | Throughput | µs/op |
|-----------|-----------|-------|
| matmul [16×16] | 15.6M/s | 0.06µs |
| matmul [32×32] | 5.1M/s | 0.20µs |
| matmul [64×64] | 2.1M/s | 0.48µs |
| matmul_relu [32×32] | 4.4M/s | 0.23µs |
| matmul_relu [128×32] | 1.8M/s | 0.55µs |
| hla_update hd=4 | 16.4M/s | 0.06µs |
| hla_update hd=8 | 9.9M/s | 0.10µs |
| ahla_step hd=4 | 18.2M/s | 0.05µs |
| ahla_step hd=8 | 10.2M/s | 0.10µs |

**End-to-end forward throughput (Config::micro, 8 positions):**

| Variant | tok/s | µs/tok |
|---------|-------|--------|
| forward (SDPA) | 1.1M/s | 0.93µs |
| forward_hla | 939K/s | 1.06µs |
| forward_ahla | 1.2M/s | 0.84µs |

**30K CCU @ 20Hz feasibility (NEON, single-core):**

| Metric | Value |
|--------|-------|
| Required | 600K tok/s (30K × 20Hz) |
| Single-core HLA | 939K tok/s |
| Cores needed | 1 |
| 8-core headroom | 9.8× |
| Full-node headroom (16c) | 19.6× |

**Note**: E2E throughput is lower than kernel-level estimates because forward pass includes embedding, layer norm, residual connections, MLP, and LM head — not just matmul. The kernel-level 15M/s matmul doesn't translate to 15M tok/s E2E.

Required: 600K/s for 30K × 20Hz.

- **SIMD single-core (NEON)**: ✅ sufficient (939K/s > 600K/s, 1.6× margin)
- **SIMD multi-core (8c)**: comfortable headroom (7.5M/s, 12.5× margin)
- **SIMD multi-core (16c)**: large headroom (15M/s, 25× margin)
- **GPU batched**: massive overkill for this config but lowest latency per tick (~0.5ms for 30K batch)

### Decision: SIMD First, GPU Later — ✅ VALIDATED (Plan 060)

**Phase 1: SIMD (NEON/AVX2) — ✅ DONE — handles 30K CCU on 1 server core**

Apple NEON (4× f32) and x86 AVX2 (8× f32) SIMD implemented in Plan 060:

```text
Changes delivered (Plan 060):
  src/simd.rs                 — NEW: SimdLevel enum, NEON/AVX2 dispatch
  types.rs: matmul()          → SIMD dot product (NEON vmlaq_f32 / AVX2 _mm256_mul_ps)
  types.rs: matmul_relu()     → SIMD dot + fused ReLU zero-clamp
  hla/kernel.rs: hla_state_update → simd_outer_product_acc for SK, CQV, G
  hla/kernel.rs: hla_readout     → simd_dot_f32 for numerator/denominator
  hla/kernel.rs: ahla_step       → simd_outer_product_acc for PKV, E
  benchmark.rs: bench_simd       → NEW: kernel + E2E throughput benchmarks
```

No new dependencies, no build pipeline changes, no nightly toolchain.
Zero test regressions (516/516 pass). Bit-identical results to scalar.

**Phase 2: GPU Batched Inference — for scale beyond 100K CCU or larger configs**

When configs grow (n_embd > 128) or CCU exceeds 100K, GPU batching becomes necessary:
- All 30K inferences tick synchronously → perfect GPU batch
- Existing `riir-gpu` WGSL matmul kernels can be reused
- Need: batch packing layer + HLA state update workgroups

### Why SIMD Wins for Our Configs

| Factor | SIMD | GPU |
|--------|------|-----|
| Build complexity | None (core::arch intrinsics) | WGSL kernels, wgpu runtime |
| Latency for hd=4-8 | ~1µs per forward pass (measured) | ~500µs dispatch overhead alone |
| Batch requirement | None (per-stream) | Need 30K batch to amortize |
| Memory | L1/L2 cache resident | GPU VRAM + upload/download |
| Deployment | Any server | Needs GPU-capable server |
| Cross-platform | ARM NEON + x86 AVX2 | Metal/Vulkan/DX12/WebGPU |
| Measured E2E | 939K tok/s single-core (NEON) | Not yet benchmarked |

For head_dim=4-8, the HLA state operations (outer products, matvec on 4×4 or 8×8 matrices) are **too small for GPU** — dispatch overhead exceeds compute time. SIMD processes them in a handful of instructions (16.4M/s for hla_update hd=4, 18.2M/s for ahla_step hd=4).

**Plan 059 Update**: HLA distillation experiment (SDPA→HLA via LoRA) shows KL divergence does NOT converge — Path C decision: HLA is inference-only, cannot be trained via SDPA distillation. HLA remains useful for streaming attention but DeltaMemoryState handles facts/retrieval.

### Constraints for Future GPU Path

When GPU batching becomes necessary:
1. **Minimum batch size**: ~1K streams (below this, GPU dispatch overhead dominates)
2. **Minimum head_dim**: ~32 (below this, HLA state ops aren't worth GPU offloading)
3. **Architecture**: all streams tick synchronously → batch all → single GPU dispatch
4. **HLA on GPU**: needs WGSL kernels for `hla_state_update`, `hla_readout`, `ahla_step` — these don't exist yet
5. **Shared matmuls** (QKV, MLP, LM head): can reuse existing WGSL matmul kernels in batched mode

## Appendix A: rust-gpu Compute Example (Reference)

The canonical compute shader from `examples/shaders/compute-shader/src/lib.rs`:

```rust
#![cfg_attr(target_arch = "spirv", no_std)]
#![deny(warnings)]

use glam::UVec3;
use spirv_std::{glam, spirv};

pub fn collatz(mut n: u32) -> Option<u32> {
    let mut i = 0;
    if n == 0 { return None; }
    while n != 1 {
        n = if n.is_multiple_of(2) { n / 2 }
            else { 3 * n + 1 };
        i += 1;
    }
    Some(i)
}

#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] prime_indices: &mut [u32],
) {
    let index = id.x as usize;
    prime_indices[index] = collatz(prime_indices[index]).unwrap_or(u32::MAX);
}
```

Key observations:
- Simple element-wise compute, no shared memory, no barriers
- `collatz()` is pure Rust that works on both CPU and SPIR-V targets
- Storage buffer bound via `#[spirv(storage_buffer, descriptor_set = 0, binding = 0)]`
- Thread count via `#[spirv(compute(threads(64)))]`

## Appendix B: reduce Example (Shared Memory + Barriers)

From `examples/shaders/reduce/src/lib.rs` — proves workgroup shared memory works:

```rust
#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::UVec3;
#[cfg(target_arch = "spirv")]
use spirv_std::memory::Scope;
use spirv_std::spirv;

#[spirv(compute(threads(256)))]
pub fn main(
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(local_invocation_id)] local_invocation_id: UVec3,
    #[spirv(subgroup_local_invocation_id)] subgroup_local_invocation_id: u32,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(num_subgroups)] num_subgroups: u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 256],
) {
    // ... subgroup reduction + shared memory + barrier pattern ...
    shared[subgroup_id as usize] = sum;
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    // ... final reduction ...
}
```

This proves the building blocks for tiled matmul (shared memory, barriers, subgroup ops) are available.