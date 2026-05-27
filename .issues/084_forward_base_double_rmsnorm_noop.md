# Issue 084: forward_base double rmsnorm on same buffer — no-op second norm

## Severity: Low-Medium (correctness concern + wasted compute)

## Location
- `src/transformer.rs` — `forward_base()` L1672-1674

## Problem

`forward_base` calls `rmsnorm` twice on the same buffer `ctx.x` without any intervening write:

```rust
// L1672-1674
rmsnorm(&mut ctx.x);                    // 1st: normalizes x in-place
ctx.xr[..n].copy_from_slice(&ctx.x[..n]); // saves normalized x
rmsnorm(&mut ctx.x);                    // 2nd: normalizes already-normalized x → NO-OP
```

After the first `rmsnorm`, `ctx.x` has mean-square ≈ 1.0 (by definition). The second `rmsnorm` computes `inv_rms ≈ 1.0` and multiplies each element by ~1.0 — it's a no-op that still costs a full SIMD `sum_sq` pass + `scale_inplace` pass over `n_embd` elements.

The same pattern appears in `forward_single_layer` (L1166-1168) and `forward_prefill` Phase A (L2606-2608).

**Note**: If this double-norm is intentional (e.g., matching a specific model architecture like Gemma 2 which uses pre-norm and post-norm), the second norm should at least be made conditional on the architecture config. If it's unintentional, it's pure waste.

## Fix

1. If the double-norm is a bug: remove the second `rmsnorm` call and just use the first normalized result
2. If it matches a specific architecture: gate the second norm behind `config.model_arch == ModelArchitecture::Gemma2` or `config.post_norm`
3. Same fix for `forward_single_layer` and `forward_prefill`

## Expected Impact
- Saves 2 SIMD passes (sum_sq + scale) over `n_embd` per layer per token
- For 4-layer model: saves 8 SIMD passes per token decode

## Optimization Reference
- optimization.md → "Don't: Recompute unchanged values"
