# Plan 158: MoA Inference Support — Token-Adaptive Activation Mixing

**Date:** 2026-05-27
**Status:** Done
**Research:** R126 (MoA Mixture of Activations)
**Feature Gate:** `moa_inference` (opt-in, default-OFF until GOAT proved)
**After GOAT proof:** If no perf hurt → default-ON

---

## Task

- [x] T1: Extend `GateActivation` enum with MoA dictionary activations
- [x] T2: Implement `MoaConfig` weight struct (gating params `u_k`, `v_ℓ`)
- [x] T3: Implement `moa_swiglu()` — token-adaptive bi-MoA forward pass
- [x] T4: SIMD-optimized `simd_matmul_rmsnorm_moa_swiglu()` fused kernel
- [x] T5: GOAT proof: correctness + overhead ≤ 1.15× + zero-regression fallback
- [x] T6: Wire into feature gate `moa_inference`, ensure OFF = zero impact

---

## T1: Extend GateActivation

Current enum has: Relu, Silu, GegeluTanh, Gegelu.

Add MoA dictionary entries:
```rust
pub enum MoaActivation {
    Id,         // x
    Relu,       // max(0, x)
    Relu2,      // max(0, x)²
    LeakyRelu,  // max(x, ηx), η=0.01
    Gelu,       // xΦ(x)
    Silu,       // x·sigmoid(x)
    Tanh,       // tanh(x)
}
```

Separate from `GateActivation` since MoA activations are only used in the mixing context.

---

## T2: MoaConfig Weight Struct

```rust
/// MoA gating parameters for a single FFN layer.
/// Present only when model was trained with MoA.
pub struct MoaConfig {
    /// Gate branch mixing params: [num_activations × d_model]
    pub gate_gating: Vec<f32>,  // u_k for each σ_k
    /// Up branch mixing params: [num_activations × d_model]
    pub up_gating: Vec<f32>,    // v_ℓ for each σ_ℓ
    /// Number of activations in dictionary (typically 7)
    pub num_activations: usize,
    /// Input dimension d_model
    pub d_model: usize,
}
```

If `MoaConfig` is `None` in weight struct → fall back to standard SwiGLU. Zero regression.

---

## T3: moa_swiglu Forward Pass

```rust
pub fn moa_swiglu(
    hidden: &mut [f32],        // output [d_model]
    gate_proj: &[f32],         // W₁x, y = [d_ffn]
    up_proj: &[f32],           // W₂x, z = [d_ffn]
    input: &[f32],             // x, for gating = [d_model]
    moa: &MoaConfig,
) {
    let d = moa.d_model;
    let n = gate_proj.len();    // d_ffn
    let k = moa.num_activations;

    // Compute gating weights: π_k = sigmoid(u_k^T x)
    // gate_weights: [k], up_weights: [k]
    let gate_weights = compute_moa_gates(input, &moa.gate_gating, k, d);
    let up_weights = compute_moa_gates(input, &moa.up_gating, k, d);

    // Mixed activations: Σ_k ρ_k σ_k(y) ⊙ Σ_ℓ π_ℓ σ_ℓ(z)
    // Per-element to avoid extra allocation
    for i in 0..n {
        let y = gate_proj[i];
        let z = up_proj[i];
        let mut mixed_gate = 0.0f32;
        let mut mixed_up = 0.0f32;
        for j in 0..k {
            mixed_gate += gate_weights[j] * moa_activate(j, y);
            mixed_up += up_weights[j] * moa_activate(j, z);
        }
        hidden[i] = mixed_gate * mixed_up;
    }
}
```

---

## T4: SIMD Fused Kernel

Extend `simd_matmul_rmsnorm_swiglu()` pattern:

```
matmul → MoA mixing (7 activations × sigmoid gate) → RMS scale → output
```

The MoA mixing is O(d_ffn × |K|) = O(4d × 7) = O(28d) — negligible vs O(d²) matmul.

---

## T5: GOAT Proof

1. **Correctness**: Reference implementation in Python → compare output elementwise, ε < 1e-5
2. **Overhead**: Benchmark MoA vs fixed SwiGLU on same model weights → wall-clock ≤ 1.15×
3. **Fallback**: When MoA config absent → same perf as current SwiGLU path (binary comparison of output)

---

## T6: Feature Gate

```toml
[features]
moa_inference = []  # opt-in until GOAT proved
```

If GOAT passes + no perf hurt → move to default. All code behind `#[cfg(feature = "moa_inference")]`.
When feature OFF: MoAConfig is `()` zero-sized, all MoA code eliminated by compiler.
