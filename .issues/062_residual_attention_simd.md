# Issue 062: Residual Add + Attention SIMD Audit (Issue 058 Follow-up)

## Status: OPEN

## Summary

Deep audit of `src/` against `.agent/optimization.md` after Issues 057/058. Found 6 concrete optimization opportunities across 3 categories: SIMD residual/embedding adds, SIMD raven attention, and SIMD softmax max-reduction.

## Evidence (Baseline: bench/071_results.csv)

```
Component                          | Current        | Issue
-----------------------------------|----------------|------------------------------------------------------------
residual add (4× per layer)        | scalar loop    | for i in 0..n { x[i] += xr[i] } — no SIMD
embedding add (1× per forward)     | scalar loop    | for i in 0..n { x[i] = wte[off+i] + wpe[off+i] } — no SIMD
raven_readout_into Q·K dot         | manual loop    | for d in 0..kv_dim { dot += q[d]*k[d] } — simd_dot_f32 unused
raven_update decay/write           | scalar loop    | for d in 0..kv_dim { keys[d] = decay*keys[d] + write*new[d] }
sample_residual_distribution_into  | scalar loop    | for val in scratch { *val *= inv_sum } — simd_scale_inplace unused
softmax max-finding                | scalar loop    | for i in 1..len { if v > max { max = v } } — no SIMD
```

## Tasks

- [ ] T1: Add `simd_add_inplace(dst: &mut [f32], src: &[f32], len: usize)` to `src/simd.rs`
- [ ] T2: Add `simd_add_into(dst: &mut [f32], a: &[f32], b: &[f32], len: usize)` to `src/simd.rs`
- [ ] T3: Wire `simd_add_into` into all embedding add loops (`forward_base`, `forward_paged`, `forward_raven`, `forward_turboquant`, `forward_prefill`)
- [ ] T4: Wire `simd_add_inplace` into all residual add loops (attn residual + MLP residual × 4 forward variants)
- [ ] T5: SIMD-ize `raven_readout_into` Q·K dot — replace manual loop with `simd_dot_f32`
- [ ] T6: SIMD-ize `raven_update` decay/write — use `simd_scale_inplace` + `simd_add_inplace` or FMA pattern
- [ ] T7: Wire `simd_scale_inplace` into `sample_residual_distribution_into` normalize
- [ ] T8: Add `simd_max_f32(x: &[f32], len: usize) -> f32` for softmax max-finding
- [ ] T9: Wire `simd_max_f32` into `softmax` and `softmax_scaled` pass 1
- [ ] T10: Add benchmark `bench_residual_simd_062` for component-level timing

## Architecture

### T1: `simd_add_inplace` (Category A — in-place accumulate)

```text
// Before (transformer.rs, ~8 call sites):
for i in 0..n {
    unsafe { *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i); }
}

// After:
simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n], n);
```

NEON: `vaddq_f32` — 4× f32 per op. AVX2: `_mm256_add_ps` — 8× f32 per op.

### T2: `simd_add_into` (Category A — zip add)

```text
// Before (transformer.rs, ~5 call sites):
for i in 0..n {
    unsafe {
        *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
            + *weights.wpe.get_unchecked(pos_off_emb + i);
    }
}

// After:
unsafe {
    simd_add_into(&mut ctx.x[..n],
        &weights.wte[tok_off..tok_off+n],
        &weights.wpe[pos_off_emb..pos_off_emb+n], n);
}
```

Same SIMD primitives as T1, but writes `a[i] + b[i]` instead of `dst[i] += src[i]`.

### T5: `raven_readout_into` SIMD (Category B)

```text
// Before (transformer.rs ~L2054):
for s in 0..num_slots {
    let k_off = s * kv_dim;
    let mut dot = 0.0f32;
    for d in 0..kv_dim {
        unsafe { dot += *query.get_unchecked(d) * *keys.get_unchecked(k_off + d); }
    }
    ...
}

// After:
for s in 0..num_slots {
    let k_off = s * kv_dim;
    let dot = unsafe {
        let q_slice = std::slice::from_raw_parts(query.as_ptr(), kv_dim);
        let k_slice = std::slice::from_raw_parts(keys.as_ptr().add(k_off), kv_dim);
        crate::simd::simd_dot_f32(q_slice, k_slice, kv_dim)
    };
    ...
}
```

Same pattern as Issue 057 T2 (`attention_head` Q·K). Raven readout is called per-head per-layer.

### T6: `raven_update` SIMD (Category B)

```text
// Before (transformer.rs ~L2019):
for d in 0..kv_dim {
    keys[offset + d] = decay * keys[offset + d] + write * new_key[d];
    values[offset + d] = decay * values[offset + d] + write * new_value[d];
}

// After (fused scale + FMA):
// 1. scale keys[offset..offset+kv_dim] by decay → simd_scale_inplace
// 2. scale new_key by write → temp (or fused)
// 3. add → simd_add_inplace
// OR: single fused loop with SIMD FMA
```

Two options:
- **Option A**: `simd_scale_inplace` on slot slice + `simd_scale_inplace` on temp new + `simd_add_inplace`. 3 SIMD passes but vectorized.
- **Option B**: New `simd_fused_decay_write` that does `dst = decay * dst + write * src` in one pass. Fewer memory round-trips.

Recommend Option B for this pattern (2 loads + 1 FMA + 1 store per element vs 2 loads + 1 mul + 1 store + 2 loads + 1 mul + 1 add + 1 store).

### T7: `sample_residual_distribution_into` normalize (Category C)

```text
// Before (src/speculative/sampling.rs ~L39):
for val in &mut scratch[..len] {
    *val *= inv_sum;
}

// After:
crate::simd::simd_scale_inplace(&mut scratch[..len], inv_sum);
```

Trivial — `simd_scale_inplace` already exists from Issue 058 T1.

### T8-T9: `simd_max_f32` for softmax (Category C)

```text
// Before (src/types.rs softmax ~L615):
let mut max = x[0];
for i in 1..len {
    let v = unsafe { *x.get_unchecked(i) };
    if v > max { max = v; }
}

// After:
let max_val = crate::simd::simd_max_f32(x, x.len());
```

NEON: `vmaxq_f32` reduction. AVX2: `_mm256_max_ps` reduction.
Must handle scalar tail elements + final horizontal max reduction.
Pattern follows existing `simd_scale_inplace` dispatch structure.

## Priority Order

1. **T1-T4** (SIMD add) — highest impact, ~13 call sites, every forward pass
2. **T5** (raven_readout SIMD dot) — same proven pattern as Issue 057 T2
3. **T7** (sample_residual scale) — trivial, one line change
4. **T6** (raven_update FMA) — moderate impact, needs new fused kernel
5. **T8-T9** (softmax max SIMD) — marginal, compiler may already auto-vectorize
6. **T10** (benchmark) — validation after all changes

## Expected Improvement

| Component | Before | After | Method |
|-----------|--------|-------|--------|
| residual add (n_embd=64) | scalar 64× add | NEON 16 / AVX2 8 ops | `simd_add_inplace` |
| embedding add (n_embd=64) | scalar 64× add | NEON 16 / AVX2 8 ops | `simd_add_into` |
| raven_readout Q·K (kv_dim=64) | manual dot loop | SIMD dot | `simd_dot_f32` |
| raven_update (kv_dim=64, 16 slots) | scalar FMA loop | SIMD fused | new kernel |
| residual normalize (vocab=256) | scalar 256× mul | SIMD scale | `simd_scale_inplace` |
| softmax max (block_size=128) | scalar cmp | SIMD max | `simd_max_f32` |

## Call Site Inventory

### Residual Add (`simd_add_inplace`) — 4× per forward variant × n_layer
- `forward_base` L639-643 (attn residual), L691-695 (MLP residual)
- `forward_paged` L1529-1533 (attn residual), L1575-1579 (MLP residual)
- `forward_raven` L2230-2234 (attn residual), L2276-2280 (MLP residual)
- `forward_turboquant` L2408-2412 (attn residual), L2454-2458 (MLP residual)
- `forward_prefill` L1228-1232 (attn residual), L1280-1284 (MLP residual)

### Embedding Add (`simd_add_into`) — 1× per forward variant
- `forward_base` L552-558
- `forward_paged` L1475-1481
- `forward_raven` L2138-2144
- `forward_turboquant` L2323-2329
- `forward_prefill` L1090-1098, L1112-1118

## Files to Modify

| File | Changes |
|------|---------|
| `src/simd.rs` | Add `simd_add_inplace`, `simd_add_into`, `simd_max_f32`, optionally `simd_fused_decay_write` |
| `src/transformer.rs` | Wire SIMD add into embedding/residual loops, wire `simd_dot_f32` into `raven_readout_into`, wire FMA into `raven_update` |
| `src/types.rs` | Wire `simd_max_f32` into `softmax` / `softmax_scaled` pass 1 |
| `src/speculative/sampling.rs` | Wire `simd_scale_inplace` into `sample_residual_distribution_into` |
| `tests/bench_residual_simd_062.rs` | Component benchmark (new file) |

## Related

- `.agent/optimization.md` — hot-path patterns reference
- Issue 057 — SIMD dot + zero-alloc (CLOSED, +25-32% forward improvement)
- Issue 058 — SIMD scale + extract paths (CLOSED)
- Plan 069 — SIMD scale audit (COMPLETED)