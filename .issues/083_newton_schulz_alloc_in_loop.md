# Issue 083: newton_schulz allocates inside iteration loop

## Severity: Medium (under newton_schulz feature — training/offline path)

## Location
- `src/newton_schulz.rs` — `newton_schulz5_square()` (L105-151), `matmul_ax()` (L48-60), `frobenius_norm()` (L63-66)

## Problem

`newton_schulz5_square` allocates 4 temporary matrices **inside the setup** plus `matmul_ax` allocates a transpose buffer **every call** (called 5× per invocation):

```rust
// L114: allocates m*n elements once (ok)
let mut x = vec![0.0f32; m * n];

// L122-124: allocates 3 more m*m + m*n buffers (ok, one-time)
let mut a_mat = vec![0.0f32; m * m];
let mut b_mat = vec![0.0f32; m * m];
let mut bx = vec![0.0f32; m * n];

// L131: allocates INSIDE the iteration loop! (called 5 times)
let mut at = vec![0.0f32; m * m];  // in the for _ in 0..ITERS loop
transpose(&a_mat, m, m, &mut at);
```

Additionally, `matmul_ax()` allocates a `vec![0.0f32; m * n]` transpose buffer every call (called once per iteration = 5 allocations).

`frobenius_norm()` uses `m.iter().map(|v| v * v).sum()` — a scalar reduction that could use `simd_sum_sq`.

## Fix

1. **Pre-allocate `at` buffer outside the loop**: move `let mut at = vec![0.0f32; m * m]` before `for _ in 0..ITERS`, just `at.clear(); at.resize(m * m, 0.0)` or reuse with fill
2. **Pre-allocate transpose buffer in `matmul_ax`**: pass it as `&mut [f32]` parameter
3. **Use `simd_sum_sq` for Frobenius norm**: replace scalar `.map(|v| v * v).sum()` with `crate::simd::simd_sum_sq(m, m.len())`
4. Consider a `NewtonSchulzContext` struct that owns all scratch buffers and can be reused across calls

## Expected Impact
- Eliminates 5 allocations per call (transpose in `matmul_ax`) + 5 more (transpose `at` in loop) = 10 allocations removed
- Frobenius norm gets ~4-8× SIMD speedup for large matrices

## Optimization Reference
- optimization.md → "Don't: Allocate inside hot loops"
- optimization.md → "SIMD / Auto-vectorization" — use existing SIMD kernels
