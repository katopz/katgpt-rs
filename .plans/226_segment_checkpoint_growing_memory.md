# Plan 226: SegmentCheckpoint — Inference-Time Growing Memory via Cached KV Segments

**Date:** 2026-06
**Status:** Active
**Research:** `.research/199_Memory_Caching_Growing_RNN_Memory.md`
**Depends On:** KVarN (Plan 159), TriggerGate (Plan 176)
**Feature Gate:** `segment_checkpoint` (ON BY DEFAULT)

---

## Overview

Implement Memory Caching (arXiv 2602.24281) as an inference-time enhancement that caches compressed KV state checkpoints at segment boundaries. GRM-style gating provides context-dependent retrieval. SSC variant for sparse top-k selection. Zero training required — pure modelless inference enhancement.

**Why:** MC-enhanced Titans beats Transformers on retrieval (SWDE 16K: 50.1 vs 44.0). Post-training proven. Zero-cost alignment with KVarN tile boundaries.

---

## Architecture

```
Sequence: [tok_0 ... tok_127 | tok_128 ... tok_255 | ... ]
           ──── Segment 1 ──── ──── Segment 2 ────
                    ↓                    ↓
              M(1) = compress(    M(2) = compress(
                KV[0..127])          KV[128..255])
                    ↓                    ↓
              Cached in            Cached in
              SegmentStore          SegmentStore
             
At query time:
  γ(i) = sigmoid(<u_t, summary(S(i))>)   // GRM gating
  y_t = γ_online · M_online(q_t) + Σ γ(i) · M_cached(i)(q_t)
```

Segment boundaries align with KVarN tile_size (128) for zero-copy checkpoints.

---

## Implementation

### Phase 1: Core SegmentCheckpoint (GRM Variant)

#### Task 1: SegmentStore Struct
- [x] Create `katgpt-rs/src/segment_checkpoint/mod.rs`
- [x] Define `SegmentCheckpoint` struct:
  ```rust
  pub struct SegmentCheckpoint {
      pub segment_id: u32,
      pub key_compressed: Vec<u8>,    // KVarN-quantized keys
      pub val_compressed: Vec<u8>,    // KVarN-quantized values
      pub summary: Vec<f32>,          // MeanPool of segment keys (for γ computation)
      pub pos_start: usize,
      pub pos_end: usize,
  }
  ```
- [x] Define `SegmentStore` struct with papaya HashMap for lock-free reads:
  ```rust
  pub struct SegmentStore {
      segments: papaya::HashMap<u32, SegmentCheckpoint>,
      max_segments: usize,           // bounded memory
      segment_size: usize,           // default 128 (= KVarN tile_size)
      gamma_proj: Vec<f32>,          // W_u projection for γ computation
  }
  ```
- [x] Implement `insert`, `get`, `top_k` methods on SegmentStore
- [x] Add segment eviction: LFU when `segments.len() > max_segments`

#### Task 2: GRM Gating
- [x] Implement `compute_gates` function:
  ```rust
  pub fn compute_gates(
      query: &[f32],         // u_t = x_t · W_u
      summaries: &[&[f32]],  // MeanPool(S(i)) for each segment
  ) -> Vec<f32> {
      // sigmoid(dot(query, summary)) — NOT softmax
      summaries.iter()
          .map(|s| sigmoid(dot(query, s)))
          .collect()
  }
  ```
- [x] Use sigmoid per project convention (not softmax from paper)
- [x] Pre-compute segment summaries at checkpoint time (O(1) per query)

#### Task 3: Checkpoint Emission
- [x] Hook into speculative decoding pipeline after `speculative_step` accepts tokens
- [x] When `accepted_len >= segment_size`, emit checkpoint:
  - Snapshot KVarN tile-aligned KV state
  - Compute MeanPool of segment keys → summary vector
  - Insert into SegmentStore
- [x] Add `SegmentStore` to `InferenceConfig` (or new `SegmentCheckpointConfig`)
- [x] Ensure zero-copy: checkpoint references KVarN tile boundaries, no re-quantization

#### Task 4: Retrieval Integration
- [x] In attention forward pass, add segment retrieval:
  - Compute γ gates for all cached segments
  - If `segment_checkpoint` feature enabled, add weighted contribution of cached segments to attention output
  - For linear memory (KVarN): aggregate quantized contributions before dequantization
- [x] Batch γ computation for all segments in one SIMD pass
- [x] Skip retrieval if only 1 segment (no benefit)

### Phase 2: SSC Sparse Selective Variant

#### Task 5: Top-k Segment Selection
- [x] Implement `top_k_segments` as pure gate-based function in `ssc.rs`:
  ```rust
  pub fn top_k_segments(gates: &[(u32, f32)], k: usize) -> Vec<(u32, f32)>
  ```
- [x] Use `select_nth_unstable_by` for O(N) partial partition instead of O(N log N) full sort
- [x] Default k = min(8, num_segments) — paper shows diminishing returns beyond k=8
- [x] Only load k segment checkpoints from store, skip remaining

#### Task 6: SSC-Enhanced Speculative Drafting
- [x] Implement `SscDrafter` with `update_context` and `enhance_draft` methods
- [x] Feed top-k cached segment summaries as additional context to drafter
- [x] Drafter produces sigmoid-biased logits informed by long-range context from cached segments
- [x] Feature-gated behind `ssc_spec_draft`

### Phase 3: Memory-Soup DDTree Branch Merging

#### Task 7: DDTree Branch State Averaging
- [ ] At DDTree leaf evaluation, compute γ-weighted average of cached branch KV states
- [ ] Apply averaged state as additional context for constraint pruning
- [ ] Feature-gate behind `memory_soup_dtree` (opt-in, experimental)

### Phase 4: TriggerGate Integration

#### Task 8: Tier-Aware Checkpoint Policy
- [ ] In `TriggerGate`, add `should_checkpoint(&self) -> CheckpointPolicy`:
  - `CpuOnly` → lazy checkpointing (only on segment boundary)
  - `CpuGpu` → normal checkpointing (every segment boundary)
  - `CpuGpuAne` → eager checkpointing (every segment boundary + pre-compute summaries)
- [ ] `CheckpointPolicy` enum: `Lazy`, `Normal`, `Eager`

### Phase 5: Tests & Benchmarks

#### Task 9: Unit Tests
- [ ] Test SegmentStore insert/get/eviction
- [ ] Test GRM gate computation (sigmoid, not softmax)
- [ ] Test top-k segment selection (correct ranking)
- [ ] Test checkpoint emission at segment boundaries
- [ ] Test retrieval with 0, 1, and N segments
- [ ] Test zero-copy alignment with KVarN tile boundaries

#### Task 10: Before/After Benchmarks
- [ ] Benchmark: NIAH-style retrieval with and without segment checkpointing
  - Expected: +10-20% accuracy at 4K+ context with GRM
- [ ] Benchmark: speculative draft acceptance rate with and without SSC
  - Expected: +5-10% acceptance rate
- [ ] Benchmark: throughput with varying segment_size (64, 128, 256, 512)
  - Expected: SSC at k=8 adds <5% overhead
- [ ] Benchmark: memory usage with varying max_segments
  - Expected: Linear growth O(max_segments × tile_size × 2)
- [ ] Profile: Gate computation cost (should be <1% of total inference)

### Phase 6: CPU/GPU Auto-Route

#### Task 11: Auto-Route Integration
- [ ] When `inference_router` detects high load (QPS > threshold), switch to SSC with lower k
- [ ] When load is low, use GRM with full segment retrieval for best accuracy
- [ ] Dynamic segment_size adjustment: shorter segments (64) at low load, longer (256) at high load

---

## Feature Gate Configuration

```toml
[features]
segment_checkpoint = []          # ON BY DEFAULT — core GRM segment caching
ssc_spec_draft = ["segment_checkpoint"]  # Opt-in — SSC sparse speculative drafting
memory_soup_dtree = ["segment_checkpoint"]  # Opt-in — experimental DDTree branch merging
```

Default: `segment_checkpoint` enabled. SSC and Memory Soup opt-in.

---

## Performance Targets

| Metric | Without MC | With GRM | With SSC |
|--------|-----------|----------|----------|
| NIAH 4K accuracy | Baseline | +10-15% | +8-12% |
| NIAH 16K accuracy | Baseline | +15-25% | +12-20% |
| Spec acceptance rate | Baseline | +3-5% | +5-10% |
| Throughput overhead | 0% | +3-8% | +1-3% |
| Memory overhead | 0% | +50-100% | +20-40% |

---

## File Structure

```
src/segment_checkpoint/
├── mod.rs           # SegmentStore, SegmentCheckpoint, public API
├── gating.rs        # GRM gate computation (sigmoid-based)
├── ssc.rs           # Top-k sparse selective caching
└── bench.rs         # Before/after benchmarks (#[test] with --nocapture)
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Memory overhead grows with segment count | LFU eviction + SSC top-k + KVarN compression |
| Gate computation adds latency | SIMD batch computation, pre-computed summaries |
| Zero-copy alignment breaks with non-default tile_size | Assert segment_size % tile_size == 0 |
| Thread contention on SegmentStore | papaya lock-free HashMap |
| Feature flag bloat affecting unrelated code | Isolated module, only active when feature enabled |

---

## TL;DR

Implement Memory Caching as `segment_checkpoint` feature: cache KVarN tile-aligned KV state at segment boundaries, retrieve with GRM sigmoid gating. Zero training needed. ON BY DEFAULT. Expected +10-25% retrieval accuracy at 4K+ context with <8% throughput overhead.
