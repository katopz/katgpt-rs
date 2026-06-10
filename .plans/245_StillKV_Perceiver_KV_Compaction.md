# Plan 245: StillKV — Perceiver-Based KV Cache Compaction (Modelless)

**Date:** 2026-06-10
**Research:** 213 (Still Perceiver KV Cache Compaction)
**Feature:** `still_kv`
**Depends On:** `dash_attn`, `vortex_flow`, `mux_latent` (optional), `kvarn`
**GOAT Status:** GATE — Must prove quality gain over MUX-Latent at same compression

---

## Overview

Implement Still-inspired KV cache compaction using heuristic Perceiver cross-attention synthesis. Replaces learned latent queries with inference-time heuristic query banks (TF-IDF clusters, attention-sink patterns, BFCF region centroids). No training required.

## Architecture

```
QuantizedKVCache trait
    └── compact_into(budget: usize, strategy: CompactionStrategy) -> Result<CompactKVCache>

CompactKVCache
    ├── keys: Vec<[f16; d]>       // t compact keys (re-rotated)
    ├── values: Vec<[f16; d]>     // t compact values  
    ├── position_offset: usize    // T - t for RoPE continuation
    └── metadata: CompactionMeta  // strategy, ratio, quality score

CompactionStrategy enum
    ├── ClusterCentroids          // k-means on [K;V] concatenation
    ├── AttentionWeighted         // DashAttention scores as importance weights
    ├── SpectralProjection        // Eigenbasis projection (top-t eigenvectors)
    ├── MuxSuperposition          // MUX-Latent vocabulary superposition
    └── BfcfRegionBlend           // BFCF region centroids as queries

StillPerceiver
    ├── cross_attention(query: &[f32], kv: &[f32]) -> Vec<f32>
    ├── self_attention(latents: &[f32]) -> Vec<f32>
    ├── project_keys(latents: &[f32]) -> Vec<[f16; d]>
    ├── project_values(latents: &[f32]) -> Vec<[f16; d]>
    └── blocks: u8               // Number of refinement blocks (default: 2)

PositionFreeCompactor
    ├── un_rotate_keys(keys: &[f32], positions: &[usize]) -> Vec<f32>
    ├── re_rotate_keys(keys: &[f32], new_positions: &[usize]) -> Vec<f32>
    └── compute_position_offset(original_len: usize, compact_len: usize) -> usize

IterativeChunkCompactor
    ├── chunk_size: usize         // c * t tokens per chunk
    ├── lookahead_buffer: bool    // 1-chunk raw KV buffer
    └── compact_stream(chunks: impl Iterator<Item=KVChunk>) -> impl Iterator<Item=CompactKVChunk>
```

## Tasks

### Phase 1: Core Infrastructure
- [x] T1: Create `src/still_kv/mod.rs` with feature gate `still_kv`
- [x] T2: Implement `PositionFreeCompactor` — RoPE un-rotate/re-rotate using existing RoPE infra
- [x] T3: Implement `CompactKVCache` struct with `position_offset` field
- [x] T4: Extend `QuantizedKVCache` trait with `compact_into()` method (default impl returns error)
- [x] T5: Implement `CompactionStrategy` enum with strategy-specific query generation

### Phase 2: Heuristic Query Banks
- [x] T6: Implement `ClusterCentroids` — mini-batch k-means on [K;V] concat (max 10 iterations)
- [x] T7: Implement `AttentionWeighted` — use DashAttention scores to weight KV averaging
- [x] T8: Implement `SpectralProjection` — project to top-t eigenvectors from SpectralQuant eigenbasis
- [x] T9: Implement `BfcfRegionBlend` — BFCF region centroids as cross-attention queries
- [x] T10: Implement `MuxSuperposition` — MUX-Latent encoder produces t superposed queries (behind `mux_latent` feature)

### Phase 3: StillPerceiver Cross-Attention
- [x] T11: Implement `StillPerceiver` with cross-attention from queries to KV cache
- [x] T12: Implement 2-block self-attention refinement (RMSNorm + residual)
- [x] T13: Implement output projection heads (identity init for near pass-through at t=T)
- [x] T14: Wire `StillPerceiver` into `compact_into()` pipeline

### Phase 4: Iterative Chunked Compaction
- [x] T15: Implement `IterativeChunkCompactor` with fixed compression ratio c
- [x] T16: Implement 1-chunk lookahead buffer (raw KV between compressed chunks)
- [x] T17: Implement position offset accounting for multi-chunk compaction
- [x] T18: Integrate with `SegmentCheckpoint` for growing memory pattern

### Phase 5: Tests & Benchmarks
- [x] T19: Unit test — position-free compaction round-trip (un-rotate → compact → re-rotate ≈ original)
- [x] T20: Unit test — compact_into produces correct budget size
- [x] T21: Unit test — iterative compaction produces linear growth at rate 1/c
- [ ] T22: Benchmark — StillKV vs MUX-Latent at 8x, 16x, 32x compression on synthetic KV data
- [ ] T23: Benchmark — StillKV synthesis vs selection (H2O-style) quality comparison
- [ ] T24: GOAT gate — measure compact-cache quality (MSE vs original) at each compression ratio

### Phase 6: StillCoT (Conditional on StillKV GOAT pass)
- [ ] T25: Extend `ChainFolder` with `compact_trace()` using StillKV
- [ ] T26: Benchmark StillCoT vs ThoughtFold (selection) on CoT reduction + quality
- [ ] T27: GOAT gate — StillCoT must match or exceed ThoughtFold's 78% reduction with better quality

---

## Performance Considerations

- Cross-attention: O(t × T × d) — only run when t << T (e.g., 128 latents for 8k+ tokens)
- Self-attention: O(t² × d) — negligible when t is small (128-512)
- Mini-batch k-means: O(10 × T × t × d) — bounded by iteration cap
- RoPE un-rotate/re-rotate: O(T × d) — same as prefill
- SIMD: Use existing NEON/AVX2 kernels from katgpt-core for attention ops
- Memory: Peak = full KV + compact KV + compactor activations (release full KV after compact)

## File Structure

```
src/still_kv/
├── mod.rs                    # Feature gate, public API
├── compact_cache.rs          # CompactKVCache struct
├── position_free.rs          # PositionFreeCompactor (RoPE handling)
├── perceiver.rs              # StillPerceiver (cross-attn + self-attn)
├── query_bank.rs             # Heuristic query generation strategies
├── iterative.rs              # IterativeChunkCompactor
└── tests/
    ├── mod.rs                # Test module
    └── benches.rs            # Benchmarks (behind `bench` feature)
```

---

## GOAT Gate Criteria

The feature stays gated until ALL pass:
1. Compact-cache MSE ≤ selection-based compaction at same budget
2. TTFT with StillKV ≤ 1.5× TTFT without compaction
3. Quality (perplexity proxy) at 16x compression ≥ 90% of full-context quality
4. No allocation in hot path (cross-attention uses pre-allocated scratch buffers)
5. Iterative compaction stable through 32k context (no collapse below no-context floor)
