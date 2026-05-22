# Plan 102: TileRT-Inspired Execution Pipeline Optimization

> **Parent**: Research 66 (TileRT Persistent Tile Pipeline Inference)
> **Depends**: Plan 096 (Spec Cost Model) ✅, Plan 055 (MTP Drafter) ✅, Plan 060 (SIMD Matmul HLA) ✅
> **Scope**: Execution stability metrics, contiguous weight allocation, stage-specialized decode path
> **Feature Gates**: `stability_metrics` (D1), `decode_specialize` (D3)

## Tasks

### D1: Execution Stability Metrics — Feature Gate `stability_metrics`

- [x] T1: Define `StabilitySnapshot` struct in `src/speculative/types.rs`
  ```rust
  #[cfg(feature = "stability_metrics")]
  pub struct StabilitySnapshot {
      pub step_latencies_ns: Vec<u64>,      // per-step wall time
      pub p50_ns: u64,
      pub p99_ns: u64,
      pub mean_ns: u64,
      pub cv: f64,                           // coefficient of variation (std/mean)
      pub stability_score: f64,             // 1.0 - (P99 / P50), 1.0 = perfect
      pub total_steps: usize,
  }
  ```
- [x] T2: Add `#[cfg(feature = "stability_metrics")]` field to `DraftResult`:
  ```rust
  #[cfg(feature = "stability_metrics")]
  pub stability: StabilitySnapshot,
  ```
- [x] T3: Instrument `speculative_step_rollback()` in `src/speculative/step.rs` with `std::time::Instant` probes
  - Record wall time for: draft phase, snapshot phase, verify phase, accept/reject phase
  - Zero overhead when feature is disabled (compile-time elimination)
- [x] T4: Add `stability_metrics` feature to `Cargo.toml` (no default)
- [x] T5: Implement `StabilitySnapshot::compute()` — calculate P50, P99, mean, CV, stability score from raw latency vector
- [x] T6: GOAT benchmark — `goat_stability_metrics` in `tests/bench_102_tilert_pipeline_goat.rs`
  - 1000 decode steps, measure stability across different KV cache sizes (16, 64, 256, 1024 positions)
  - Assert: `stability_score > 0.7` (P99 < 3.3× P50) for all sizes
  - Assert: `cv < 0.5` for micro config
  - Verify: zero overhead when feature disabled (bench with/without)

### D2: Contiguous Weight Allocation (Internal Refactor, No Feature Gate) — **In Progress**

- [x] T7: Create `ContiguousWeights` struct in `src/weights.rs` ✅
  ```rust
  pub struct ContiguousWeights {
      buffer: Vec<f32>,                              // single allocation
      layers: Vec<WeightSlice>,                      // offset+len per layer
  }
  
  struct WeightSlice {
      wq_offset: usize, wq_len: usize,
      wk_offset: usize, wk_len: usize,
      wv_offset: usize, wv_len: usize,
      wo_offset: usize, wo_len: usize,
      w1_offset: usize, w1_len: usize,
      w2_offset: usize, w2_len: usize,
      w3_offset: usize, w3_len: usize,
      rms_norm_offset: usize, rms_norm_len: usize,
      rms_norm_final_offset: usize, rms_norm_final_len: usize,
  }
  ```
- [x] T8: Implement `ContiguousWeights::from_weights(weights: &TransformerWeights) -> Self`
  - Calculate total size, single `Vec::with_capacity`, copy all weights with 64-byte alignment padding
  - Each weight matrix accessed via `&buffer[slice.offset..slice.offset + slice.len]`
- [x] T9: Add `ContiguousWeights` accessor methods (`layer_wq`, `layer_wk`, `wte()`, `wpe()`, `lm_head()`, etc.) ✅
  - Returns zero-copy views into the contiguous buffer
  - Same API shape as current per-layer `Vec<f32>` access
- [x] T10: Wire into `forward()` — use `ContiguousWeights` when available, fallback to existing per-Vec path
- [x] T11: Benchmark: compare L2 cache miss rate (via `perf stat`) on 16-layer micro config decode
  - Expected: <5% improvement on micro (weights fit in L2 anyway)
  - Target: >10% improvement on larger configs (>8 layers)
- [x] T12: If benchmark shows no gain for micro config, keep behind a `const USE_CONTIGUOUS: bool` flag
  - Do NOT add a feature gate — this is an internal optimization, not a user-visible change

### D3: Stage-Specialized Decode Path — Feature Gate `decode_specialize`

- [x] T13: Define `DecodeStage` enum in `src/transformer.rs`
  ```rust
  #[derive(Clone, Copy, PartialEq, Eq)]
  pub enum DecodeStage {
      Prefill,  // batch-friendly, attention-heavy, needs full KV write
      Draft,    // small batch, can skip screening, matmul-heavy
      Verify,   // single batch, needs exact attention, KV read-heavy
      Sample,   // SIMD-only, no attention needed
  }
  ```
- [x] T14: Create `forward_decode_stage()` function — specialized `forward()` for `DecodeStage::Draft` and `DecodeStage::Verify`
  - `Draft`: skip `ScreeningPruner`, skip KV cache write for positions > draft_length, use approximate attention
  - `Verify`: exact attention, full KV write, enable screening
  - `Sample`: only head projection + softmax, skip all intermediate layers
- [x] T15: Wire into speculative step: `speculative_step_rollback()` calls `forward_decode_stage(DecodeStage::Draft)` for drafting, `forward_decode_stage(DecodeStage::Verify)` for verification
- [x] T16: Add `decode_specialize` feature to `Cargo.toml`
- [x] T17: GOAT benchmark — measure speculative step wall time with/without `decode_specialize`
  - Assert: draft phase ≥10% faster (skips screening + reduced KV writes)
  - Assert: verify phase unchanged (same logic, different code path)
  - Assert: acceptance rate unchanged (quality-neutral optimization)

## Objective

Distill the three highest-value insights from TileRT's execution pipeline into our CPU SIMD inference stack:

1. **D1 (Execution Stability)**: TileRT's production insight — "the hardest problems are increasingly systemic." Our GOAT proofs measure mean speed, not P99 stability. Add per-step latency instrumentation behind `stability_metrics` feature gate. This is the foundation for diagnosing performance regressions and validating optimization claims.

2. **D2 (Contiguous Weights)**: TileRT packs all model parameters into a single contiguous allocation with 1024-byte alignment. On CPU, this means better L2 cache spatial locality — sequential weight reads hit the same cache lines. For our micro config the gain may be marginal, but the pattern scales to larger models.

3. **D3 (Stage Specialization)**: TileRT's heterogeneous workers specialize GPU devices for different tasks. Our CPU analog: specialize the decode code path for draft vs verify. Draft can skip screening and reduce KV writes. Verify needs exact attention. Currently both use the same `forward()` — wasting cycles on unnecessary work during drafting.

**Honest scope**: We are NOT implementing persistent kernels (no CUDA on CPU), NOT doing warp/block specialization (no GPU), NOT doing AOT graph capture (Rust compiler handles monomorphization). We're stealing the *principles*, not the implementation.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                    Plan 102 Execution Pipeline                       │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │
│  │ D1: Stability│    │ D2: Contiguous│   │ D3: Stage Specialize    │ │
│  │  Metrics     │    │   Weights     │   │                          │ │
│  │              │    │              │   │  DecodeStage::Prefill    │ │
│  │ StepTimer    │    │ ┌──────────┐ │   │  DecodeStage::Draft     │ │
│  │ ↓ P50/P99    │    │ │ buffer   │ │   │  DecodeStage::Verify    │ │
│  │ ↓ CV/Stab    │    │ │ [f32]    │ │   │  DecodeStage::Sample    │ │
│  │ ↓ per-phase  │    │ │          │ │   │         ↓               │ │
│  │              │    │ │ wq₀ wk₀ │ │   │  forward_decode_stage() │ │
│  │ Feature:     │    │ │ wv₀ wo₀ │ │   │  → skip screening       │ │
│  │ stability_   │    │ │ ...      │ │   │  → reduced KV writes    │ │
│  │ metrics      │    │ │ wqₙ wkₙ │ │   │  → exact attention      │ │
│  └─────────────┘    │ └──────────┘ │   │                          │ │
│                      │              │   │  Feature:               │ │
│                      │ No feature   │   │  decode_specialize      │ │
│                      │ gate needed  │   └──────────────────────────┘ │
│                      └──────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

## Why These Three (And Not Others)

| TileRT Insight | Action | Reason |
|---|---|---|
| Persistent execution | ❌ Skip | No CUDA on CPU. Rust compiler handles inlining/monomorphization. |
| Warp/block specialization | ❌ Skip | GPU-only concept. |
| Tile-level pipeline overlap | ⚠️ Partial | D3 captures the *spirit* (specialize for stage), not the mechanism (tile overlap). |
| Heterogeneous workers | ✅ D3 | CPU analog: different decode stages get different hot paths. |
| Execution stability | ✅ D1 | Production-critical. We have zero tail latency metrics today. |
| Contiguous allocation | ✅ D2 | Low-effort refactor, CPU-relevant, scales to larger models. |
| CUDA graph capture | ❌ Skip | GPU-only. Rust generics are our "AOT." |
| FP8 quantization | ❌ Skip | We do f32/f16 only. riir-gpu handles quantization separately. |
| MLA attention | ❌ Skip | Our attention is standard MHA/GQA, not Multi-head Latent Attention. |
| Fused ops | ⚠️ Partial | Our SIMD kernels are already "fused" by CPU standards (same thread, same cache). D3's stage specialization is the CPU fusion boundary. |

## What This Does NOT Prove

- This does NOT prove TileRT-level speedups. TileRT achieves 500-600 tok/s on 8×B200. We target 10-30% improvement on our micro config.
- This does NOT change the model-based/modelless duality. These are *execution* optimizations, not *reasoning* changes.
- This does NOT validate persistent kernel architecture on CPU. That's a GPU-specific pattern.

## Integration Points

| Component | D1 | D2 | D3 |
|---|---|---|---|
| `src/speculative/step.rs` | ✅ Instrument timing | — | ✅ Stage dispatch |
| `src/speculative/types.rs` | ✅ StabilitySnapshot | — | — |
| `src/transformer.rs` | — | ✅ ContiguousWeights | ✅ DecodeStage |
| `src/types.rs` | — | — | — |
| `src/lib.rs` | — | — | ✅ Feature gate export |
| `Cargo.toml` | ✅ Feature gate | — | ✅ Feature gate |
| `tests/bench_102_*.rs` | ✅ GOAT proof | ✅ Cache bench | ✅ Stage bench |

## Priority Assessment

| Task | Value | Effort | Risk | Priority |
|---|---|---|---|---|
| D1 (Stability Metrics) | HIGH — foundational for all future perf work | LOW — ~100 lines | LOW — additive only | **P0** |
| D2 (Contiguous Weights) | MEDIUM — scales with model size | LOW — ~150 lines | LOW — fallback exists | **P1** |
| D3 (Stage Specialize) | MEDIUM — 10-30% speculative speedup | MEDIUM — ~200 lines | MEDIUM — changes hot path | **P2** |

## References

- Research 66: `.research/66_TileRT_Persistent_Tile_Pipeline_Inference.md`
- TileRT blog: https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html
- TileRT code: `.raw/TileRT/python/` (verified dual-model MTP pipeline, contiguous allocation, algorithm enums)
- Related research: 55 (Tri-Mode), 59 (MoE+SD Co-Design), 34 (D2F)
- Related plans: 089 (Tri-Mode), 096 (Spec Cost Model), 055 (MTP Drafter)