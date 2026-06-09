# Plan 236: Core Optimization Audit 2 — katgpt-core

## Summary
Second pass applying optimization.md guidelines to `crates/katgpt-core/src/`. Focus on branch elimination in ternary ops, SIMD fusion in activation inner loops, allocation-free verification, and scalar→SIMD conversion in init paths.

## Tasks
- [x] P0: `simd.rs` — `simd_ternary_dot_f32`: Replace branching sign extraction with branchless `(pos - neg) as f32` + hoisted scale + unchecked access
- [x] P0: `types.rs` — `SenseModule::project`: Replace branching ternary dot with branchless + hoisted loop bound
- [x] P1: `types.rs` — `gegelu`/`swiglu` inner reciprocal loops: Replace scalar `buf[j] = 1.0 / buf[j]` with `simd_reciprocal_inplace` on chunk
- [x] P1: `types.rs` — `SenseModule::verify`: Avoid full struct clone (~232B), compare commitment bytes directly
- [x] P2: `types.rs` — `rmsnorm`: Eliminate f64 intermediate (`1e-5` → `1e-5f32`) to avoid f64 round-trip
- [x] P2: `shard_embedding.rs` — `JlProjectionMatrix::generate`: Use `simd_dot_f32` + `simd_sum_sq` for Gram-Schmidt, multiply by `inv_norm` instead of divide
- [x] P3: `types.rs` — `sample_token`/`sample_token_into`: Replace `binary_search_by` closure with `partition_point` (branch-predictor friendly)

## GOAT Gate
- All changes preserve existing public API signatures ✓
- Branchless ternary dot must match existing test vectors exactly ✓
- `verify()` must remain byte-compatible with `commit()` output ✓
- All 157 tests pass ✓

## Validation
```
cargo check -p katgpt-core    # ✓ compiles
cargo test -p katgpt-core     # ✓ 157 passed, 0 failed
```
