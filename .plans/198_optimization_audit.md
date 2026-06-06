# Plan 198: Optimization Audit — SIMD Vectorization & Zero-Alloc

> **Status**: Active
> **Depends On**: `.contexts/optimization.md`
> **Scope**: `crates/katgpt-core/src/`, `src/transformer.rs`, `src/types.rs`

## Objective

Apply optimization.md patterns to hot-path code: SIMD vectorization of scalar loops,
struct field reordering for cache locality, and zero-alloc audit of remaining allocators.

## Tasks

- [x] T1: SIMD-vectorize `WallPrefixState::compute_gate_from_key` (scalar exp/ln loop)
- [x] T2: Reorder `ForwardContext` fields to eliminate padding — already well-ordered, no change needed
- [x] T3: Optimize `kv_group_lut` from `[usize; 128]` to `[u8; 128]` (saves 896 bytes, better cache locality)
- [x] T4: Fuse `attention_head` score max-scan into SIMD pass (`simd_scale_inplace` + `simd_max_f32`)
- [x] T5: Audit remaining Vec allocations — `sample_token` callers in production already use `_into` variant
- [x] T6: Run benchmarks & verify all tests pass (2040 passed, 1 pre-existing flaky alloc tracker test)
- [x] T7: Commit with `feat:` prefix (commit 03d70223)
