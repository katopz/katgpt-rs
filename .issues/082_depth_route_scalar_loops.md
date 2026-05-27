# Issue 082: depth_route uses scalar loops instead of SIMD kernels

## Severity: Medium-High (hot-path under delta_routing feature)

## Location
- `src/transformer.rs` — `depth_route()` (L1448-1504) and `depth_route_with_indices()` (L1509-1567)

## Problem

Both `depth_route` and `depth_route_with_indices` compute RMSNorm and dot products using **scalar for-loops** instead of the SIMD-accelerated kernels already available in `katgpt-core::simd`:

```rust
// CURRENT: scalar loops in depth_route_with_indices (L1530-1542)
let mut sum_sq = 0.0f32;
for &val in src.iter() {
    sum_sq += val * val;
}
let rms = (sum_sq / n_embd as f32 + eps).sqrt();
let inv_rms = 1.0 / rms;

let mut logit = 0.0f32;
for ((&s, &nw), &qw) in src.iter().zip(norm_weight.iter()).zip(query_weight.iter()) {
    let normalized = s * inv_rms * nw;
    logit += qw * normalized;
}
```

This is called at every block boundary for every layer (every 4th layer × n_embd iterations). The same pattern appears in `depth_route_weights()` (L1575-1623) which also allocates a `Vec<f32>` per call.

## Fix

1. Replace scalar `sum_sq` loop with `simd_sum_sq(src, n_embd)`
2. Replace scalar dot product with `simd_dot_f32()` on pre-scaled buffer
3. Replace the weighted-sum accumulation (step 3) with `simd_fused_scale_acc()`
4. In `depth_route_weights`, pre-allocate the logits buffer once and pass it in (or reuse an existing buffer from `ForwardContext`)

## Expected Impact
- ~3-4× speedup on the RMSNorm+dot portion for n_embd ≥ 64 (NEON/AVX2 process 4/8 elements per cycle)
- Eliminates one allocation in `depth_route_weights`

## Optimization Reference
- optimization.md → "SIMD / Auto-vectorization" (use SIMD kernels)
- optimization.md → "Allocation" (pre-build cached data, reuse buffers)
