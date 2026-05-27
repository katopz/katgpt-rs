# Issue 087: forward_prefill delta_routing uses scalar delta accumulation instead of simd_fused_sub_acc

## Severity: Medium (under delta_routing + prefill path)

## Location
- `src/transformer.rs` — `forward_prefill()` L2858-2863

## Problem

In `forward_prefill`, the delta routing accumulation uses a scalar loop instead of the SIMD-accelerated `simd_fused_sub_acc` that `forward_base` uses:

```rust
// L2858-2863 in forward_prefill (SCALAR):
for d in 0..n {
    let delta = ctx.x[d] - ctx.xr[d];
    if block_idx < ctx.block_deltas.len() {
        ctx.block_deltas[block_idx][d] += delta;
    }
}
```

vs `forward_base` (SIMD-accelerated):

```rust
// L1821-1826 in forward_base (SIMD):
if block_idx < ctx.block_deltas.len() {
    crate::simd::simd_fused_sub_acc(
        &mut ctx.block_deltas[block_idx][..n],
        &ctx.x[..n],
        &ctx.xr[..n],
        n,
    );
}
```

## Fix

Replace the scalar loop in `forward_prefill` with the same `simd_fused_sub_acc` call:

```rust
if block_idx < ctx.block_deltas.len() {
    crate::simd::simd_fused_sub_acc(
        &mut ctx.block_deltas[block_idx][..n],
        &ctx.x[..n],
        &ctx.xr[..n],
        n,
    );
}
```

## Expected Impact
- ~3-4× speedup on the delta accumulation for n_embd ≥ 64
- Consistency with `forward_base` — same operation, same SIMD dispatch

## Optimization Reference
- optimization.md → "SIMD / Auto-vectorization" — use existing SIMD kernels
