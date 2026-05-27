# Issue 088: gegelu / silu / swiglu use scalar exp — no SIMD acceleration

## Severity: Medium (called in MLP layer every decode step)

## Location
- `crates/katgpt-core/src/types.rs` — `gegelu()` L1374-1381, `silu()` L1399-1403, `swiglu()` L1408-1414, `gegelu_tanh()` L1387-1395

## Problem

The activation functions use scalar `.exp()` calls in elementwise loops:

```rust
// gegelu (L1377):
let sigmoid = 1.0 / (1.0 + (-1.702 * g).exp());

// silu (L1400-1402):
*v = *v / (1.0 + (-*v).exp());

// swiglu (L1411):
let silu_val = g / (1.0 + (-g).exp());
```

These are called once per MLP layer per token decode. For `mlp_hidden` dimensions of 512+, the scalar exp is a bottleneck — each `exp()` call involves multiple floating-point operations (range reduction, polynomial approximation, reconstruction).

The codebase already has `simd_exp_inplace` for softmax — it could be adapted for these fused activation patterns.

## Fix

### Option A: Fused SIMD activation kernels
Create specialized SIMD kernels that fuse the sigmoid/exp computation with the elementwise multiply:

```rust
// simd_silu: x[i] = x[i] * sigmoid(x[i]) using SIMD exp
pub fn simd_silu_inplace(x: &mut [f32]) {
    // NEON/AVX2: broadcast 1.0, compute exp(-x), add 1.0, divide, multiply
}

// simd_swiglu: hidden[i] = silu(gate[i]) * up[i]
pub fn simd_swiglu(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    // Fused: hidden[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
}
```

### Option B: Polynomial sigmoid approximation
Replace `1.0 / (1.0 + (-x).exp())` with a faster sigmoid approximation (e.g., the one from `gegelu` could use the identity `sigmoid(x) ≈ 0.5 + 0.5 * tanh(0.5 * x)` which avoids exp entirely).

## Expected Impact
- For mlp_hidden=512: ~3-4× speedup on activation computation
- These are called every layer of every decode step — savings compound with depth

## Optimization Reference
- optimization.md → "SIMD / Auto-vectorization" — use SIMD kernels, keep inner loops branch-free
- optimization.md → "Keep inner loops branch-free (use SIMD)"
