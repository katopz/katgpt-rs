# Plan 185: Q-K=V Projection Sharing — Inference KV Cache Halving

**Date:** 2026-06-05
**Research:** 165 (Q-K=V Projection Sharing)
**Status:** Planning
**Scope:** katgpt-rs (modelless, inference-time only)

---

## Tasks

- [x] **T1: Add `AttentionProjection` enum to transformer config**
  - `Full` (default: Q, K, V separate)
  - `SharedKV` (Q-K=V: K=V weight tying)
  - Add to `ModelConfig` / `TransformerConfig`
  - Zero breaking change — default is `Full`

- [x] **T2: Implement K=V weight merging (post-hoc, no retraining)**
  - Load pretrained Q, K, V weights normally
  - If `SharedKV`: merge K,V via `W_kv = (W_k + W_v) / 2`
  - Store merged `W_kv` in place of `W_v` (or new field)
  - Skip V projection at inference: reuse K output as V
  - Add `merge_kv_weights(&mut LayerWeights)` helper

- [x] **T3: Update KV cache backends to skip V storage**
  - `MultiLayerKVCache`: when `SharedKV`, allocate half the cache
  - `PagedKVCache`: page stores K only when `SharedKV`
  - `RavenKVCache`: slot stores K only → 2× slots
  - `TurboQuantKVCache`: quantize K only → 2× effective density
  - `SpectralQuantKVCache`: same
  - All other backends: same pattern
  - Add `cache_layout: CacheLayout` enum (`KV` | `K`)

- [x] **T4: Update attention forward pass**
  - In `forward_base()` / `forward_turboquant()`: skip V projection, use K as V
  - `let v = if shared_kv { &k } else { project(x, wv) };`
  - Ensure GQA still works (K=V within each KV group)
  - Ensure wall attention works (K=V with diagonal gate)

- [x] **T5: GOAT proof tests** — `tests/kv_share_goat.rs` (7/7 passing)
  - Dense, Paged, Raven, TurboQuant, SpectralQuant, KVarN
  - Measure: peak memory, decode throughput, per-token latency
  - Expected: 50% cache reduction, +4-5% throughput per paper
  - Document results in benchmark table

- [x] **T6: Test/example showing thinking vs non-thinking**
  - Example: `qkv_sharing_demo` — load model, compare Full vs SharedKV
  - Show: memory usage, tokens/sec, perplexity delta
  - Demonstrate K=V × GQA compound savings

- [x] **T7: Feature gate** — `kv_share = []` in Cargo.toml
  - Feature: `kv_share` (default on)
  - Compile-time: `#[cfg(feature = "kv_share")]` on new code paths
  - Runtime: `config.attention_projection = SharedKV` to opt out
  - Verify no perf regression when feature is off

- [x] **T8: Update DDTree speculative verification**
  - When `SharedKV`: verification reads K only, V = K
  - Reduce verification FLOPs by ~30%
  - Measure acceptance rate change

---

## Architecture

```rust
/// Attention projection configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionProjection {
    /// Standard Q, K, V (3 projections, full KV cache)
    #[default]
    Full,
    /// Q-K=V: K and V share projection (2 projections, K-only cache)
    /// 50% KV cache reduction, ~3% perplexity cost
    SharedKV,
}

/// KV cache layout (derived from AttentionProjection)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLayout {
    /// Store both K and V (standard)
    KV,
    /// Store K only, V = K at read (SharedKV)
    K,
}
```

## Expected Results

| Backend | Memory Before | Memory After | Throughput Change |
|---------|--------------|-------------|------------------|
| Dense KV | 2 × n_layers × seq × head_dim | 1 × ... | +5% |
| TurboQuant | 2 × quant_tensor | 1 × quant_tensor | +5% |
| Raven RSM | N slots × 2 × head_dim | N slots × head_dim (or 2N slots) | +5% |
| SpectralQuant | 2 × eigen_tensor | 1 × eigen_tensor | +5% |
| + GQA-4 | 75% reduction | **87.5% reduction** | +5% |

## Dependencies

- None — this is self-contained within katgpt-rs
- Works with all existing features (GQA, wall attention, speculative decode, etc.)

## Constraints

- **Modelless**: No LLM training. Post-hoc weight merging only.
- **Default ON**: Feature gate defaults to enabled.
- **No perf hurt**: Must benchmark before merge.
- **SOLID**: Enum-based, open/closed, zero breaking changes.
