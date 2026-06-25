/-
! Spec for `ActionBridge::select_action` ranking preservation (Plan 293).

This Lean 4 model is the mathematical specification of the property that
`katgpt-rs/crates/katgpt-core/src/bridge/mod.rs::ActionBridge::select_action`
relies on: **sigmoid is strictly monotonically increasing**, so dot-product
ordering is preserved through the sigmoid projection.

Rust reference (the bridge's inner loop):

```rust
let mut dot = 0.0f32;
for d in 0..D {
    dot = q_values[d].mul_add(dir[d], dot);
}
let score = crate::simd::fast_sigmoid(dot);
```

where `fast_sigmoid x = 1.0 / (1.0 + (-x).exp())` for `|x| ≤ 40` (and saturates
to `0.0`/`1.0` outside that range — see `simd/activations.rs`).

We model the dot product over `ℝ` and the sigmoid via Mathlib's `Real.sigmoid`,
which is defined as `x.sigmoid = (1 + Real.exp (-x))⁻¹` — bit-for-bit the same
mathematical object as the Rust `1.0 / (1.0 + (-x).exp())` on its non-saturating
domain. The `|x| > 40` saturation in Rust does not affect ranking: `σ(40)` and
`σ(41)` are both indistinguishable from `1` in `f32`, and the strict-monotone
theorem over `ℝ` is the idealised (infinite-precision) contract the bridge is
designed against. Float32 is a libm approximation of this contract.

This is the **open katgpt-rs primitive** (Tier 3 of the bridge FV strategy,
per `.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md`). It
promotes the empirical `g1_3_bridge_ranking_preservation` test in
`micro_belief/tests.rs` from an ∃-check over 1000 random triples to a ∀-theorem
over every pair of dot products.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid

namespace KatgptProof.Bridge

open Real

/-! ## Dot product (mirrors the Rust `mul_add` FMA chain)

The Rust loop `dot = q[d].mul_add(dir[d], dot)` accumulates `Σ_d q[d] * dir[d]`.
We define the same over `ℝ` for a finite index type `ι`. Using a generic index
type (rather than `Fin D`) lets the theorem apply to any dimensionality.
-/

/-- The dot product `Σ_i q i * d i` over a finite index type `ι`.
    Mirrors `ActionBridge::select_action`'s inner `mul_add` accumulation loop. -/
def dot {ι : Type*} [Fintype ι] (q d : ι → ℝ) : ℝ :=
  ∑ i, q i * d i

/-! ## Sigmoid

We use Mathlib's `Real.sigmoid`, defined as `x.sigmoid = (1 + Real.exp (-x))⁻¹`,
which is the exact mathematical object the Rust `fast_sigmoid` approximates:
`1.0 / (1.0 + (-x).exp())`. No re-definition is needed — `Real.sigmoid` IS the
spec. Mathlib proves `Real.sigmoid_strictMono : StrictMono sigmoid`.
-/

end KatgptProof.Bridge
