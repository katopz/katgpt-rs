# Hot-Path Performance Audit: katgpt-core + src/

Audited against `optimization.md` guidelines. All findings reference concrete file paths and line numbers.

## Priority Legend

| Priority | Meaning |
|---|---|
| 🔴 HIGH | Measurable perf win, violates explicit optimization.md "Do/Don't" rule |
| 🟡 MEDIUM | Likely win but depends on config/workload size |
| 🟢 LOW | Minor, style-level, or defensive improvement |

---

## 🔴 ISSUE 1: Heap allocation inside `forward_looped` hot loop ✅ FIXED

**File:** `src/transformer.rs` L765

```rust
let mut prev_h = vec![0.0f32; n];
```

**Rule violated:** "Don't: Allocate inside hot loops" — `forward_looped` is called per-token and this `vec!` allocates a new heap buffer every invocation.

**Fix:** Add a `prev_h: Vec<f32>` field to `ForwardContext`, pre-allocate once in `ForwardContext::new`:

```rust
// In ForwardContext:
pub(crate) prev_h: Vec<f32>, // [n_embd]

// In ForwardContext::new:
prev_h: vec![0.0; config.n_embd],
```

Then replace `let mut prev_h = vec![...]` with `ctx.prev_h[..n].fill(0.0)` (or copy_from_slice).

**Impact:** Eliminates one heap alloc + dealloc per token during looped inference. For `n_embd=2304` this is a 9 KB allocation per call.

---

## 🔴 ISSUE 2: `select_topk_indices` allocates on every call ✅ FIXED

**File:** `src/transformer.rs` L1252-1268

```rust
pub fn select_topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // ...
    indexed[..k].iter().map(|(i, _)| *i).collect()  // another alloc
}
```

**Rule violated:** "Don't: Allocate inside hot loops" — allocates two `Vec`s per call.

**Fix:** Use the existing `select_topk_indices_into_buf` (L1301-1324) which reuses pre-allocated buffers. The `select_topk_indices` public API should either:
1. Be deprecated in favor of `_into_buf` variants, or
2. Accept a `&mut Vec<(usize, f32)>` scratch buffer parameter.

Additionally, the `.collect()` at L1268 allocates a second Vec — the `_into_buf` variant already avoids this.

---

## 🔴 ISSUE 3: `cluster_scores_buf` allocated at vocab_size but only `num_clusters` elements used ✅ FIXED

**File:** `src/transformer.rs` L330, L391

```rust
cluster_scores_buf: Vec<f32>, // [vocab_size] upper-bound for cluster scores
// ...
cluster_scores_buf: vec![0.0; config.vocab_size],
```

But in `clustered_lm_head` (L1354):

```rust
let cluster_scores = &mut cluster_scores_buf[..num_clusters];
```

**Rule violated:** "Do: Pre-allocate output arrays upfront" — but also don't over-allocate by orders of magnitude.

**Fix:** `cluster_scores_buf` should be sized to `num_clusters` (= `vocab_size / cluster_size`) not `vocab_size`. For a 256K vocab with `cluster_size=1024`, this wastes 255K × 4B = ~1 MB. Change the allocation in `ForwardContext::new`:

```rust
cluster_scores_buf: vec![0.0; config.vocab_size.div_ceil(config.mtp_cluster_size.max(1))],
```

Similarly, `topk_indexed_buf` at L331/L392 is sized to `vocab_size` but only needs `num_clusters` entries (it's used for cluster top-K, not token top-K).

---

## 🔴 ISSUE 4: `softmax` in `attention_head` uses separate passes for subtract-max, exp, and normalize ✅ FIXED

**File:** `src/transformer.rs` L536-549

```rust
for s in scores_slice.iter_mut() { *s -= max_score; }
crate::simd::simd_exp_inplace(scores_slice);
let sum: f32 = scores_slice.iter().copied().sum();
let inv_sum = 1.0 / sum;
for t in 0..t_n { *scores_buf.get_unchecked_mut(t) *= inv_sum; }
```

This is 4 passes over `scores_buf`:
1. Subtract max (scalar loop)
2. SIMD exp
3. Sum (scalar `.iter().copied().sum()`)
4. Scale by inv_sum (scalar loop)

**Rule violated:** "Do: SIMD / Auto-vectorization" — Passes 1, 3, and 4 are scalar loops that should use SIMD operations.

**Fix:** Fuse subtract-max into `simd_exp_inplace` (sub+exp is a common fused operation), or at minimum replace:
- Pass 1 (`*s -= max_score`): use `simd_add_inplace` with `-max_score`
- Pass 3 (`.sum()`): use a SIMD horizontal sum
- Pass 4 (`*= inv_sum`): use `simd_scale_inplace`

```rust
// Fused approach:
crate::simd::simd_scale_inplace(scores_slice, 1.0); // noop, but pattern...
// Better: create simd_sub_scale_inplace(x, val, scale) that does x[i] = (x[i] - val) * scale
```

The existing `simd_exp_inplace` already takes a mutable slice — a `simd_sub_exp_inplace(slice, max)` would save one full pass.

---

## 🟡 ISSUE 5: `attention_head` value accumulation loop is scalar ✅ FIXED

**File:** `src/transformer.rs` L558-568

```rust
for t in 0..t_n {
    let s = unsafe { *scores_buf.get_unchecked(t) };
    let v_row = unsafe { ... };
    for d in 0..hd {
        unsafe { *attn_out.get_unchecked_mut(q_head_offset + d) += s * v_row[d]; }
    }
}
```

**Observation:** The inner `d` loop broadcasts scalar `s` across `hd` elements. This is effectively `s * v_row + attn_out` — a fused multiply-add over `hd` elements. For `hd >= 64` (common), a SIMD `vfmaq` / `_mm256_fmadd` would process 4-8 elements per iteration.

**Fix:** Use `simd_scale_inplace`-like pattern but with accumulation:

```rust
// After computing all softmax weights, fuse the value accumulation:
for t in 0..t_n {
    let s = scores_buf[t];
    // SIMD: attn_out[d..d+4] += s * v_row[d..d+4] (vfmaq_f32)
    crate::simd::simd_fused_scale_acc(&mut attn_out[q_head_offset..], &v_row, s, hd);
}
```

This would require adding a `simd_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize)` kernel — trivially derived from existing `simd_fused_decay_write`.

**Impact:** For `hd=64`, `t_n=256`, this is 16K multiply-accumulates. SIMD would reduce to ~4K instructions.

---

## 🟡 ISSUE 6: `rmsnorm` computes `sum_sq` + `sqrt` + `scale` as separate passes

**File:** `crates/katgpt-core/src/types.rs` L1357-1368

```rust
let sum_sq = crate::simd::simd_sum_sq(x, x.len());
let inv_rms = 1.0 / (sum_sq / x.len() as f32 + 1e-5).sqrt();
crate::simd::simd_scale_inplace(x, inv_rms);
```

**Observation:** This is already well-structured (2 passes: sum-of-squares, scale). The sum-of-squares pass uses SIMD. No explicit violation, but for `x.len()` < 16 (small head dims), the SIMD dispatch overhead may exceed the benefit.

**No immediate fix needed** — this is noted for awareness. If profiling shows `rmsnorm` as a bottleneck, the two passes could be fused into a single SIMD pass that computes `x[i] /= sqrt(mean_sq + eps)` without intermediate.

---

## 🟡 ISSUE 7: `forward_base` does `ctx.attn_out[..n].fill(0.0)` before attention heads

**File:** `src/transformer.rs` L1739

```rust
ctx.attn_out[..n].fill(0.0);
```

Then `attention_head` (L552-555) also zeros its output slice per head:

```rust
for d in 0..hd {
    unsafe { *attn_out.get_unchecked_mut(q_head_offset + d) = 0.0; }
}
```

**Observation:** Double zeroing — the `fill(0.0)` in `forward_base` zeros the entire `attn_out`, then each `attention_head` call zeros its own head's slice again. The per-head zeroing inside `attention_head` is necessary (it's called independently), but the `fill(0.0)` before the head loop is redundant when all heads are computed.

**Fix:** Remove the `ctx.attn_out[..n].fill(0.0)` at L1739 (and L1191, L2033) since `attention_head` already zeros its output range. If there are configs where not all heads are computed, keep it — but for standard MHA/GQA, all `n_head` heads cover the full `n_embd` range.

**Impact:** Saves one `memset(n_embd * 4)` per layer per token. For `n_embd=2304`, that's 9 KB memset × `n_layer` × tokens.

---

## 🟡 ISSUE 8: `kv_group_lut` is a `Vec<usize>` but could be `[usize; MAX_HEADS]` ✅ FIXED

**File:** `src/transformer.rs` L334, L394-396

```rust
kv_group_lut: Vec<usize>, // [n_head]
// ...
kv_group_lut: (0..config.n_head).map(|h| h * config.n_kv_head / config.n_head).collect(),
```

**Rule violated:** "Do: Use fixed-size arrays `[T; N]` when domain is bounded."

**Observation:** `n_head` is bounded by config (typically 4-32). The heap allocation for a small lookup table is marginal, but it's accessed in the inner attention loop. A stack-allocated array avoids the indirection.

**Fix:** Use a const-sized array (e.g., `[usize; 64]`) since head counts are always small:

```rust
kv_group_lut: [usize; 64], // fixed-size, n_head <= 64
kv_group_lut_count: usize, // actual count
```

Or compute `kv_group` inline: `h * config.n_kv_head / config.n_head` — this is a single multiply+divide that the compiler can optimize (especially if `n_kv_head` divides `n_head` evenly).

---

## 🟡 ISSUE 9: `select_topk_indices_into` still allocates a return `Vec<usize>` ✅ FIXED

**File:** `src/transformer.rs` L1275-1297

```rust
pub fn select_topk_indices_into(
    scores: &[f32],
    k: usize,
    indexed_buf: &mut Vec<(usize, f32)>,
) -> Vec<usize> {  // <-- still allocates return Vec
    // ...
    indexed_buf[..k].iter().map(|(i, _)| *i).collect()  // alloc
}
```

**Rule violated:** "Don't: Allocate inside hot loops" — the `_into` variant still allocates via `.collect()`.

**Fix:** Callers should use `select_topk_indices_into_buf` instead, which writes into a pre-allocated `output_buf`. Or modify `select_topk_indices_into` to accept an `output: &mut Vec<usize>` parameter.

---

## 🟢 ISSUE 10: `sample_token` linear scan

**File:** `crates/katgpt-core/src/types.rs` L1556-1566

```rust
pub fn sample_token(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum { return i; }
    }
    probs.len() - 1
}
```

**Observation:** Linear scan over the full vocabulary. For large vocab (256K), this averages 128K comparisons per token.

**Potential fix:** Build a prefix-sum array once, then binary search (O(log V)). However, this only helps if `sample_token` is called with the full vocab (it usually is). The trade-off is an extra `vocab_size` memory and the prefix-sum build cost. Only worth it if sampling is on the hot path — in speculative decoding, the verify step typically uses argmax, not sampling.

---

## 🟢 ISSUE 11: `LayerWeights` uses 6 separate `Vec<f32>` instead of slice references into contiguous allocation

**File:** `src/transformer.rs` L24-31

```rust
pub struct LayerWeights {
    pub attn_wq: Vec<f32>,
    pub attn_wk: Vec<f32>,
    pub attn_wv: Vec<f32>,
    pub attn_wo: Vec<f32>,
    pub mlp_w1: Vec<f32>,
    pub mlp_w2: Vec<f32>,
}
```

**Observation:** The `ContiguousWeights` struct in `src/weights.rs` already implements the contiguous pattern but isn't used in the forward pass — `forward_base` accesses `layer_weights.attn_wq` etc. which are separate heap allocations.

**Fix:** Long-term, `forward_base` and `attention_head` should accept `&ContiguousWeights` or `&[f32]` slices from the contiguous buffer. This improves L2 cache locality since sequential weight reads (wq → wk → wv → wo → w1 → w2) hit adjacent memory regions.

This is a larger refactor — the `weights.rs` infrastructure is already in place.

---

## 🟢 ISSUE 12: `simd_dot_f32` dispatch overhead on x86_64

**File:** `crates/katgpt-core/src/simd.rs` L84-101

```rust
pub fn simd_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() { ... } // runtime check every call
    }
}
```

**Observation:** `is_avx2_fma_available()` checks a cached `AtomicBool` on every call. On aarch64 this is compile-time resolved. On x86_64, this is a function call + atomic load per `simd_dot_f32` invocation — which happens thousands of times per forward pass.

**Potential fix:** The function is already `#[inline]`, so LLVM may hoist the check. But if profiling shows the branch as hot, consider:
1. Using `#[target_feature(enable = "avx2,fma")]` with runtime detection at startup, then calling the AVX2 version directly.
2. Or checking once at the top of `forward_base` and calling a monomorphized inner function.

The current approach is fine for most cases — this is a micro-optimization.

---

## Summary

| # | Priority | Issue | File | Rule |
|---|---|---|---|---|
| 1 | 🔴 | Heap alloc per token in `forward_looped` | transformer.rs:765 | Don't: allocate in hot loops |
| 2 | 🔴 | `select_topk_indices` allocates 2 Vecs per call | transformer.rs:1252 | Don't: allocate in hot loops |
| 3 | 🔴 | `cluster_scores_buf` over-allocated by 100-1000× | transformer.rs:330 | Do: right-size pre-allocations |
| 4 | 🔴 | Scalar loops in `attention_head` softmax | transformer.rs:536-549 | Do: SIMD for bulk operations |
| 5 | 🟡 | Scalar value accumulation in `attention_head` | transformer.rs:558-568 | Do: SIMD for bulk operations |
| 6 | 🟡 | `rmsnorm` is already good — noting for awareness | types.rs:1357 | — |
| 7 | 🟡 | Double zeroing of `attn_out` | transformer.rs:1739,552 | Redundant work |
| 8 | 🟡 | `kv_group_lut` Vec vs fixed array | transformer.rs:334 | Do: fixed-size arrays |
| 9 | 🟡 | `_into` variant still allocates return Vec | transformer.rs:1275 | Don't: allocate in hot loops |
| 10 | 🟢 | `sample_token` O(V) linear scan | types.rs:1556 | Linear scan in hot path |
| 11 | 🟢 | `LayerWeights` not using contiguous allocation | transformer.rs:24 | Cache locality |
| 12 | 🟢 | Runtime SIMD dispatch on x86_64 | simd.rs:84 | Micro-opt |

**Recommended action order:** Issues 1 → 4 → 7 → 3 → 2 → 5 → 9 → 8 → 10 → 11 → 12
