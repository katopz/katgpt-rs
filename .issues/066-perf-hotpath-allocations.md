# PERF-001: Hot-path allocation elimination and SIMD optimization

## Summary

Performance audit identified critical allocation-in-hot-loop and SIMD inefficiency issues
across the inference hot path. Every generated token triggers unnecessary heap allocations
and redundant computation.

## Scope

### P0 — Inference hot path (every token)

1. **`is_avx2_fma_available()` runs CPUID per dot product** (`katgpt-core/src/simd.rs` ~L88)
   - Two serializing CPUID instructions thousands of times per matmul
   - Fix: Cache result in `OnceLock<AtomicBool>`

2. **AVX2 dot uses `mul_ps` + `add_ps` instead of FMA** (`katgpt-core/src/simd.rs` ~L159)
   - 2 instructions per 8 elements when `_mm256_fmadd_ps` does it in 1
   - Fix: Replace with fused multiply-add

3. **`lora_apply` step-2 (B @ hidden) is scalar loop** (`katgpt-core/src/types.rs` ~L1816)
   - ~4-8× slower than SIMD dot for every LoRA-enabled forward pass
   - Fix: Use `simd_dot_f32` per row

4. **O(n) linear scan in `ScalarCodebook::quantize`** (`octopus/codebook.rs` L310, `turboquant/codebook.rs` L199)
   - ~256 comparisons when binary search needs ~8
   - Fix: Use `partition_point()` (binary search on monotonic boundaries)

5. **`(hd as f32).sqrt()` recomputed per-layer** (`transformer.rs` 9+ call sites)
   - `head_dim` is constant; sqrt + div recomputed every layer of every token
   - Fix: Pre-compute `attn_scale` in `ForwardContext::new()`

6. **GDN2: 5+ heap allocations per token decode** (`gdn2/forward.rs` L76-83, `gdn2/kernel.rs` L86)
   - `out_buf`, `temp_buf`, `erase_b`, `decay_alpha`, `write_w_channel`, `delta` — all fixed-size
   - Fix: Pre-allocate into cache/context struct, `clear()` + reuse

## Acceptance Criteria

- [ ] All P0 fixes implemented with no behavioral changes
- [ ] Existing tests pass (`cargo test`)
- [ ] No new dependencies added
