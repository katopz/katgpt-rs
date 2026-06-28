# Issue 006: x86_64 SIMD Backends Missing `#[target_feature]` (Blocks Release Verify)

> **Type:** Bug fix (correctness — compile error on x86_64)
> **Status:** Resolved
> **Owner:** develop
> **Created:** 2026-06-27
> **Cross-repo:** lands in katgpt-rs only (`crates/katgpt-core/src/simd/`).
> **Origin:** release-plz `publish_no_verify = true` workaround (commit `2735bf84`),
> put in place because `cargo publish` failed verification on the x86_64
> GitHub Actions runner with 38 errors of the form
> `cannot find function _mm256_setzero_ps` / `is_avx2_fma_available` /
> `horizontal_sum_256`.
> **References:** [release-plz.toml](../release-plz.toml) · katgpt-core@0.2.0 publish

---

## TL;DR

`katgpt-core`'s AVX2 SIMD backends have **never compiled on x86_64**. Two
classes of bugs, both invisible on Apple Silicon (arm64) because the entire
`#[cfg(target_arch = "x86_64")]` surface is compiled out there:

1. **Missing `#[target_feature(enable = "avx2,fma")]`** on every
   `unsafe fn avx2_*`. Without it, the AVX2/FMA intrinsics
   (`_mm256_fmadd_ps`, `_mm256_setzero_ps`, `_mm256_i32gather_ps`, etc.)
   are not callable — Rust reports them as "cannot find function".
2. **Missing `use super::{is_avx2_fma_available, horizontal_sum_256, ...}`**
   in the submodule files. The bare-name calls resolve on arm64 only because
   the call sites live inside `#[cfg(target_arch = "x86_64")]` blocks that
   are excluded from arm64 compilation. On x86_64 they are real unresolved
   names. (`research.rs` already has `use super::*;` so it's unaffected.)

The release-plz workaround `publish_no_verify = true` bypassed `cargo`'s
build-verification step during `cargo publish`, allowing katgpt-core@0.2.0 to
ship. Once this issue is resolved, that workaround MUST be removed so future
releases are actually verified to compile on the target architecture.

---

## Root cause

`std::arch::x86_64::_mm256_*` intrinsics carry `#[target_feature(enable =
"avx")]` (or `"avx2"`, `"fma"`). They are only callable from a context that
has the matching feature enabled. The `avx2_*` kernels in this crate declare
the function as plain `unsafe fn` with only `#[cfg(target_arch = "x86_64")]`
and `#[inline]` — neither attribute enables the target feature, so the
intrinsic calls fail to resolve.

The runtime detection (`is_avx2_fma_available()` via `cpuid`) is correct; the
compile-time gating is what's missing.

---

## Files & functions affected

| File | AVX2 fns needing `#[target_feature]` | Missing imports |
|---|---|---|
| `simd/horizontal.rs` | `horizontal_sum_256`, `horizontal_max_256` (use `_mm256_*`) | — (defines them) |
| `simd/activations.rs` | `avx2_reciprocal_inplace`, `avx2_exp_inplace`, `avx2_sigmoid_tanh_clamp`, `avx2_sigmoid_inplace`, `avx2_exp_sum_inplace` | `is_avx2_fma_available`, `horizontal_sum_256` |
| `simd/dot.rs` | `avx2_dot_f32`, `avx2_outer_product_acc`, `avx2_outer_product_acc_scaled` | `is_avx2_fma_available`, `horizontal_sum_256` |
| `simd/elementwise.rs` | `avx2_scale_inplace`, `avx2_add_inplace`, `avx2_add_scalar_inplace`, `avx2_fused_sub_scale_inplace`, `avx2_sum_f32`, `avx2_add_into`, `avx2_max_f32`, `avx2_fused_decay_write`, `avx2_scale_mul_inplace` | `is_avx2_fma_available`, `horizontal_sum_256`, `horizontal_max_256` |
| `simd/research.rs` | `avx2_sum_sq`, `avx2_sum_sq_quartic`, `avx2_sum_abs_f32`, `avx2_dist_sq`, `avx2_fused_sub_acc`, `avx2_fused_scale_acc` | already has `use super::*;` |
| `simd/sparse.rs` | `avx2_sparse_dot_f32` | `is_avx2_fma_available`, `horizontal_sum_256` |
| `simd/ternary.rs` | `avx2_ternary_matvec` | `horizontal_sum_256` (dispatches via `simd_level()`, not `is_avx2_fma_available`) |

Not affected:
- `simd/argmax.rs` — no AVX2 path (uses scalar two-pass on x86_64 by design).
- `simd/maxsim.rs` — delegates to `simd_dot_f32`, no direct intrinsics.
- `simd/horizontal.rs::horizontal_sum_128` — SSE-only (baseline on x86_64).

---

## Fix

1. Add `#[target_feature(enable = "avx2,fma")]` to every `unsafe fn avx2_*`
   and to `horizontal_sum_256` / `horizontal_max_256` (which use `_mm256_*`
   AVX intrinsics). Attribute placement: after `#[cfg(target_arch = "x86_64")]`,
   before `#[inline]`.
2. Add `#[cfg(target_arch = "x86_64")] use super::{...};` imports to the
   submodule files so the bare-name calls resolve on x86_64 without producing
   unused-import warnings on arm64/wasm32.
3. Remove `publish_no_verify = true` from `release-plz.toml`.
4. Verify with `cargo check -p katgpt-core --target x86_64-apple-darwin`
   (cross-target available locally on Apple Silicon).

`#[target_feature(enable = "avx2,fma")]` is chosen over the narrower
`"avx,fma"` to match the runtime gate `is_avx2_fma_available()` exactly —
the function only returns `true` when AVX2 **and** FMA **and** AVX are all
present, so the compile-time gate should require the same set.

---

## Acceptance

- [x] `cargo check -p katgpt-core --target x86_64-apple-darwin` is clean.
- [x] `cargo check -p katgpt-core --target x86_64-apple-darwin --all-features` is clean.
- [x] `cargo test -p katgpt-core --lib` still passes on arm64 (847 tests, no regression).
- [x] `publish_no_verify = true` removed from `release-plz.toml`.
- [x] Commit on `develop` with `fix:` prefix.

## Outcome

Resolved. The x86_64 AVX2 backends now compile cleanly under both default and
`--all-features`. Three classes of pre-existing bug were fixed (all invisible
on arm64 because the entire `#[cfg(target_arch = "x86_64")]` surface is
compiled out there):

1. Added `#[target_feature(enable = "avx2,fma")]` to every `unsafe fn avx2_*`
   and the two `horizontal_*_256` reducers in `horizontal.rs`, plus the f64
   AVX2 kernels in `peira.rs` (`avx2_outer_product_ema_f64`,
   `avx2_outer_product_f64`, `avx2_dot_f64`, `horizontal_sum_256d`).
2. Added `#[cfg(target_arch = "x86_64")] use super::{...}` imports so the
   bare-name calls to `is_avx2_fma_available` and the horizontal reducers
   resolve on x86_64 (they were only resolvable on arm64 because the call
   sites are cfg'd out there).
3. Fixed two missing-intrinsic-import bugs surfaced only on x86_64:
   `_mm256_setzero_ps` in `activations.rs::avx2_exp_sum_inplace` and
   `_mm256_setzero_pd` in `peira.rs::avx2_dot_f64`.
4. Fixed a malformed `#[target_feature(enable = "avx2", enable = "fma")]`
   in `elementwise.rs::avx2_fused_sub_scale_inplace` (duplicate `enable` key;
   the correct form is `enable = "avx2,fma"`).

Bonus: removed redundant `unsafe` blocks and unused imports that became
visible once the `target_feature` attribute made the matching intrinsics
safe to call within the function body.

The `release-plz.toml` workaround `publish_no_verify = true` has been removed
— future `cargo publish` will actually verify the build on the target arch.

**Pre-existing failure NOT caused by this fix** (left untouched per the
"don't fix unrelated bugs" rule): `curator::tests::test_verification_weight_thresholds`
fails under `--all-features` on arm64 both before and after this change
(verified via `git stash` on `develop@2735bf84`). Unrelated to SIMD.
