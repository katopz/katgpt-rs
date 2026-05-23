# Plan 103: CODA-Inspired Fused SIMD Kernels

> **Parent**: Research 67 (CODA GEMM-Epilogue Programming)
> **Depends**: Plan 060 (SIMD Matmul HLA) ✅, Plan 069 (SIMD Scale Zero-Alloc Audit) ✅, Plan 102 (TileRT Pipeline) ✅
> **Scope**: Algebraic reparameterization of matmul→residual→rmsnorm→activation into fused SIMD kernels
> **Feature Gate**: `coda_fusion` in microgpt-rs (opt-in, proven via GOAT)
> **Cross-project**: Guides riir-ai Plan 106 (CubeCL) epilogue patterns

## Tasks

### D1: Core Fused SIMD Kernels — Feature Gate `coda_fusion`

- [x] **T1**: Add `coda_fusion` feature to `microgpt-rs/Cargo.toml` (no default)
  ```toml
  [features]
  coda_fusion = []
  ```

- [x] **T2**: Create `crates/microgpt-core/src/coda.rs` — fused SIMD kernel implementations
  ```rust
  //! CODA-inspired fused SIMD kernels (Research 67).
  //!
  //! Algebraic reparameterization: fuse matmul+residual+rmsnorm+activation
  //! into single-pass SIMD loops, eliminating intermediate buffer writes.
  //!
  //! Key identity (CODA §3.2.1):
  //!   RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'
  //!
  //! This lets us delay the row-wise RMSNorm scale past the next GEMM.
  ```

- [x] **T3**: Implement `simd_matmul_residual_partial_rms()` — the core fused kernel
  ```
  For each output row i:
    acc = dot(weight_row[i], input)     // SIMD dot product
    acc += residual[i]                  // fused residual add
    partial_rms[i/block] += acc^2 / block_size  // partial RMS accumulation
    output[i] = acc * gamma[i]          // norm weight scaling
  ```
  Returns: `(D: Vec<f32>, partial_sums: Vec<f32>, O: Vec<f32>)`
  - D = matmul output + residual (unscaled)
  - partial_sums = per-block mean-square values
  - O = D * gamma (norm-weight scaled)

- [x] **T4**: Implement `compute_rstd()` — lightweight reduction over partial sums
  ```rust
  pub fn compute_rstd(partial_sums: &[f32], eps: f32) -> Vec<f32> {
      // One rstd per row: 1/sqrt(sum(partial_sums) + eps)
  }
  ```
  This is the "auxiliary reduction" from CODA — tiny compared to full RMSNorm kernel.

- [x] **T5**: Implement `simd_matmul_rmsnorm_swiglu()` — fused MLP gate+up
  ```
  For each output row i:
    acc = dot(weight_row[i], input) * rstd[row_of_i]   // matmul + delayed RMS scale
    if i is gate row: acc = silu(acc)                   // SwiGLU gate activation
    output[i/2] = gate * up                             // dimension-reducing
  ```
  Returns: output of dimension N/2 (SwiGLU halves the paired features)

- [x] **T6**: Implement `simd_matmul_residual()` — fused down-proj + residual
  ```
  For each output row i:
    acc = dot(weight_row[i], input)    // SIMD dot product
    acc += residual[i]                 // fused residual add
    output[i] = acc
  ```

- [x] **T7**: Implement `simd_matmul_rmsnorm_rope()` — fused QKV + RoPE (stretch)
  ```
  For each output row i (paired features 2i, 2i+1):
    acc = dot(weight_row[i], input) * rstd[row_of_i]   // matmul + RMS scale
    // RoPE rotation on adjacent pairs
    cos_val = cos_table[pos * head_dim + i//2]
    sin_val = sin_table[pos * head_dim + i//2]
    q_even = acc; q_odd = acc_next
    output[2i]   = q_even * cos_val - q_odd * sin_val
    output[2i+1] = q_even * sin_val + q_odd * cos_val
  ```

### D2: Wire Fused Kernels into Forward Pass

- [x] **T8**: Add `forward_coda()` in `src/transformer.rs` behind `#[cfg(feature = "coda_fusion")]`
  - Replaces the standard `forward_base()` layer loop with CODA-fused version
  - Layer structure changes from:
    ```
    rmsnorm → save_residual → rmsnorm → matmul QKV → attention → matmul out_proj → residual_add
    save_residual → rmsnorm → matmul gate_up → relu → matmul down → residual_add
    ```
    To:
    ```
    rmsnorm → save_residual → rmsnorm → matmul QKV → attention
    matmul_residual_partial_rms(out_proj, attn_out, xr, gamma) → compute_rstd()
    matmul_rmsnorm_swiglu(gate_up, O, rstd) → matmul_residual(down, hidden, xr2)
    ```
  - Falls back to `forward_base()` when feature is disabled (zero-cost abstraction)

- [x] **T9**: Handle Gemma2-specific activations
  - Gemma2 uses `gegelu_tanh` not `silu` for the gate activation
  - The `simd_matmul_rmsnorm_swiglu` kernel needs a generic activation parameter
  - Consider: `fn simd_matmul_rmsnorm_activation<A: Activation>(...)` where Activation is a trait

- [x] **T10**: Handle LoRA application in fused path
  - LoRA is applied after each matmul: `output += B @ A @ input`
  - In the fused path, LoRA should be applied to the pre-residual matmul output
  - i.e., `acc = dot(W_row, input) + lora_dot(B_row, A @ input)`
  - This keeps LoRA mathematically identical

### D3: GOAT Proof — Benchmark

- [x] **T11**: Create `tests/bench_103_coda_fusion_goat.rs`
  - Benchmark: `forward_base()` vs `forward_coda()` on micro config (n_embd=64, 4 layers)
  - Measure: wall time per token (ns), L1 cache misses (via `perf stat` on Linux, `sample` on macOS)
  - Assert: `forward_coda()` >= 5% faster than `forward_base()` for BS=1 decode
  - Assert: output tokens are bit-identical (or within f32 epsilon for the partial RMS path)
  - Assert: zero overhead when `coda_fusion` feature is disabled

- [x] **T12**: Create `microgpt-rs/.benchmarks/030_coda_fusion_simd.md`
  - Report: baseline vs fused, per-layer breakdown, buffer write count comparison
  - Include: numerical accuracy comparison (cosine similarity of outputs)

- [x] **T13**: Cross-validation with riir-ai GPU path
  - Document the CODA epilogue patterns that should guide Plan 106 (CubeCL) tasks T2.3-T2.6
  - Specifically: which visitor primitives map to which CubeCL kernel types

## GOAT Proof Criteria

| # | Criterion | Pass Threshold | Stretch Goal |
|:--|:---|:---|:---|
| G1 | Fused kernel correctness | Bit-identical or ε < 1e-5 | Bit-identical |
| G2 | Decode speedup (micro) | ≥ 5% | ≥ 10% |
| G3 | Buffer write reduction | ≥ 20% fewer writes per layer | ≥ 30% |
| G4 | Feature isolation | Compiles with/without `coda_fusion` | Zero overhead when disabled |
| G5 | Numerical stability | Cosine sim ≥ 0.9999 vs baseline | Bit-identical |

## Architecture

### Current (Separate Kernels)
```
forward_base() per layer:
  rmsnorm(x)        → 2 SIMD passes (sum_sq + scale)
  copy → xr         → 1 memcpy
  rmsnorm(x)        → 2 SIMD passes
  matmul(q, wq, x)  → n SIMD dot products
  matmul(k, wk, x)  → n_kv SIMD dot products
  matmul(v, wv, x)  → n_kv SIMD dot products
  attention(...)     → O(n * t) work
  matmul(x, wo, ao)  → n SIMD dot products
  add(x, xr)        → 1 SIMD pass
  copy → xr2        → 1 memcpy
  rmsnorm(x)        → 2 SIMD passes
  matmul_relu(h, w1) → mlp_hidden SIMD dot products
  matmul(x, w2, h)   → n SIMD dot products
  add(x, xr2)       → 1 SIMD pass

  Total per layer: ~6n + n_kv + mlp_hidden SIMD dot products
                   + ~8 buffer passes (rmsnorm×3, add×2, copy×2, matmul_output×~5)
```

### CODA Fused (This Plan)
```
forward_coda() per layer:
  rmsnorm(x)        → 2 SIMD passes (can't fuse pre-first-GEMM norm)
  copy → xr         → 1 memcpy
  rmsnorm(x)        → 2 SIMD passes
  matmul(q, wq, x)  → n SIMD dot products (QKV separate for attention)
  matmul(k, wk, x)  → n_kv SIMD dot products
  matmul(v, wv, x)  → n_kv SIMD dot products
  attention(...)     → O(n * t) work
  # FUSED: matmul + residual + partial_rms + gamma_scale in one pass
  matmul_residual_partial_rms(x, wo, ao, xr, gamma) → n fused SIMD dots
  compute_rstd(partial_sums)  → tiny reduction
  # FUSED: matmul + rms_scale + SwiGLU in one pass
  matmul_rmsnorm_swiglu(hidden, w_gate_up, O, rstd) → mlp_hidden/2 fused SIMD dots
  # FUSED: matmul + residual in one pass
  matmul_residual(x, w_down, hidden, xr2) → n fused SIMD dots

  Total per layer: ~6n + n_kv + mlp_hidden SIMD dot products (same compute)
                   + ~3 buffer passes (rmsnorm×2, copy×1 — eliminated 5 passes!)
```

### Buffer Write Savings

| Buffer Write | Baseline | CODA Fused | Eliminated? |
|:---|:---|:---|:---|
| rmsnorm output (pre-QKV) | 2 passes | 2 passes | No (can't fuse before first GEMM) |
| xr copy | 1 memcpy | 1 memcpy | No |
| matmul out_proj → ctx.x | 1 write | 0 (fused) | ✅ |
| residual add ctx.x += xr | 1 read-modify-write | 0 (fused) | ✅ |
| xr2 copy | 1 memcpy | 1 memcpy | No |
| rmsnorm (pre-MLP) | 2 passes | 0 (fused into matmul via delayed r) | ✅ |
| matmul gate_up → hidden | 1 write | 0 (fused) | ✅ |
| relu activation | 1 pass | 0 (fused into SwiGLU) | ✅ |
| matmul down → ctx.x | 1 write | 0 (fused) | ✅ |
| residual add ctx.x += xr2 | 1 read-modify-write | 0 (fused) | ✅ |
| **Total** | **~10 passes** | **~4 passes** | **60% reduction** |

## Files Changed

| File | Change |
|------|--------|
| `microgpt-rs/Cargo.toml` | Add feature gate: `coda_fusion` |
| `crates/microgpt-core/src/coda.rs` | New: fused SIMD kernels (T3-T7) |
| `crates/microgpt-core/src/lib.rs` | Add `pub mod coda;` behind feature gate |
| `microgpt-rs/src/transformer.rs` | Add `forward_coda()` behind feature gate |
| `tests/bench_103_coda_fusion_goat.rs` | New: GOAT benchmark |
| `.benchmarks/030_coda_fusion_simd.md` | New: results |

## Risks

| Risk | Probability | Impact | Mitigation |
|------|:---:|:---:|:---|
| Partial RMS numerical drift | Low | Medium | GOAT test with cosine similarity ≥ 0.9999 |
| No measurable speedup on micro config | Medium | Low | Small vectors fit L1 anyway; feature gate means zero cost |
| LoRA integration breaks fusion | Low | Medium | LoRA applied to pre-residual output; mathematically identical |
| Increased code complexity | Medium | Low | Feature gate isolates; standard path untouched |

## Estimated Impact

| Metric | Before | After (Estimate) | Reason |
|:---|:---:|:---:|:---|
| Buffer passes per layer | ~10 | ~4 | Fused kernels eliminate intermediate writes |
| Function calls per layer | ~12 | ~8 | Fewer separate kernel calls |
| Expected speedup (micro) | baseline | +5-10% | Function call overhead reduction at BS=1 |
| Expected speedup (larger) | baseline | +10-20% | L1 pressure reduction for larger n_embd |

## References

- CODA paper: `.raw/coda-kernels/README.md` (local copy)
- CODA code: `.raw/coda-kernels/kernels/gens/` (CuTeDSL implementations)
- Research 67: `.research/067_CODA_GEMM_Epilogue_Programming.md`
- Plan 106: `riir-ai/.plans/106_cubecl_gpu_kernel_rewrite.md` (GPU epilogue patterns)
- Plan 102: `.plans/102_tilert_execution_pipeline.md` (execution pipeline optimization)
- Plan 060: `.plans/060_simd_matmul_hla.md` (SIMD matmul baseline)