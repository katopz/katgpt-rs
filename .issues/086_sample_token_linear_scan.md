# Issue 086: sample_token uses O(V) linear scan — cache-unfriendly for large vocab

## Severity: Medium (called every decode step)

## Location
- `crates/katgpt-core/src/types.rs` — `sample_token()` L1556-1566

## Problem

`sample_token` performs a linear scan through the full probability distribution to find the cumulative bin containing the random sample:

```rust
pub fn sample_token(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
```

For large vocabulary (e.g., 256K tokens), this scans on average V/2 elements per sample. This is a data-dependent branch (unpredictable) that causes pipeline stalls.

## Fix

### Option A: Prefix-sum + binary search (O(V) build, O(log V) query)
Build a cumulative sum array once after softmax, then binary search:

```rust
pub fn sample_token(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    // Binary search for the cumulative probability bin
    let mut cumsum = 0.0;
    let mut lo = 0usize;
    let mut hi = probs.len();
    // Two-pass: build partial prefix sum chunks for cache locality
    // ... or simply:
    match probs.binary_search_by(|&p| {
        // This won't work directly — need prefix sum array
        todo!()
    }) { ... }
}
```

### Option B: Top-K + renormalize (better for inference)
For inference, after softmax, only the top-K tokens have meaningful probability. Use top-K filtering (already have `select_topk_indices`) then sample from the filtered set.

### Option C: Alias method (O(1) sampling)
Build a Vose alias table once after softmax: O(V) build, O(1) per sample. Best when sampling multiple tokens from the same distribution (speculative decoding).

## Expected Impact
- For V=256K: average scan length goes from ~128K to O(log K) or O(1)
- Significant for models with large vocabulary (Gemma 2, LLaMA 3)
- Less impactful for small vocab (V=27 in current micro config)

## Optimization Reference
- optimization.md → "Don't: Linear scan for hot-path queries"
- optimization.md → "Data Structures" — pre-compute lookup tables for O(1) reads
