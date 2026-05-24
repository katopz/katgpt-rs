# Plan 126: RTPurbo — Retrieval Head Sparse Decode via Low-Dimensional Indexing

**Branch:** `develop/feature/126_rt_turbo_retrieval_head`
**Depends on:** Plan 106 (DashAttention), Plan 044 (PFlash), Plan 070 (SP-KV)
**Research:** 86 (RTPurbo — Retrieval Head Sparse Attention)
**Feature Gate:** `rt_turbo` (opt-in, requires GOAT proof before default-on promotion)
**Goal:** Add head-wise retrieval/local classification + dynamic top-p token selection for decode-phase sparse attention. Complements DashAttention's α-entmax block routing with per-head specialization from RTPurbo.

---

## Tasks

### Phase 1: Offline Head Calibration
- [x] **T1**: Create `src/rt_turbo/calibration.rs` — `HeadCalibration` struct with `retrieval_scores: Vec<f32>` (one per query head), `retrieval_set: Vec<usize>`, `threshold: f32`
- [x] **T2**: Implement `calibrate_heads()` — takes model weights + calibration sequence (needle at beginning and end), computes per-head retrieval score `R_h = mean(attn from post-needle to pre-needle)`, partitions into H_ret (top 15%) and H_loc
- [x] **T3**: Implement `HeadCalibration::save()` / `HeadCalibration::load()` — serialize to JSON/TOML for offline reuse
- [x] **T4**: Unit tests: synthetic attention patterns → correct head classification, single-head edge case, all-retrieval / all-local edge cases
- [x] **T5**: Add `RtTurboConfig` to `katgpt-core/src/types.rs` — `retrieval_head_ratio: f32` (0.15), `low_dim: usize` (16), `top_p: f32` (0.9), `sliding_window: usize` (8192), `sink_tokens: usize` (4), `block_size: usize` (64)
- [x] **T6**: Register `#[cfg(feature = "rt_turbo")]` gate in `Cargo.toml` features, add `pub mod rt_turbo` in `lib.rs`, require `dash_attn` feature

### Phase 2: Low-Dimensional Pre-RoPE Projection
- [x] **T7**: Create `src/rt_turbo/projection.rs` — `RetrievalProjection` struct with `w_q: Vec<f32>` and `w_k: Vec<f32>` per retrieval head (shape `[head_dim, low_dim]`)
- [x] **T8**: Implement `project_score()` — compute low-dim relevance: `s(m,n) = (W_Q · q_pre)ᵀ · (W_K · k_pre)` for pre-RoPE query/key vectors
- [x] **T9**: Implement `batch_project_scores()` — vectorized scoring over full KV cache for a single retrieval head using SIMD. Returns `Vec<f32>` of scores `[seq_len]`
- [x] **T10**: Unit tests: zero-initialized projection → uniform scores, identity projection → matches full-dim dot product on first 16 dims, dimensionality check — 31 projection tests passing

### Phase 3: Dynamic Top-P Token Selection
- [ ] **T11**: Create `src/rt_turbo/top_p.rs` — sort-free top-p selector for CPU
- [ ] **T12**: Implement `select_top_p()` — given scores `[seq_len]` and threshold p (0.9): sort descending, compute cumulative softmax mass, return indices where cumsum < p + last index that pushes over threshold
- [ ] **T13**: Implement `select_top_p_blockwise()` — block-level variant that scores blocks via low-dim projection, selects blocks where cumulative block mass ≥ p, then fine-grained within selected blocks
- [ ] **T14**: Unit tests: concentrated scores (single peak) → 1-2 tokens selected, uniform scores → many tokens selected, edge case p=1.0 → all tokens, edge case p=0.0 → top-1 only, exact mass preservation check

### Phase 4: Head-Wise Sparse Decode Integration
- [ ] **T15**: Create `src/rt_turbo/forward.rs` — `forward_rt_turbo_decode()` integrating head-wise routing
- [ ] **T16**: Implement local-head path — sliding window (8192) + sink tokens (4) only, skip full KV scan, use existing `forward_sdpa` on window slice
- [ ] **T17**: Implement retrieval-head path — low-dim projection → top-p token selection → full-dim SDPA on selected token indices only
- [ ] **T18**: Implement `forward_rt_turbo_prefill()` — local heads: window + sinks; retrieval heads: dense (delegate to standard prefill). This matches RTPurbo design
- [ ] **T19**: Add `RtTurboCache` struct — per-layer storage for: calibration result reference, projection weights, selected token indices (reusable across decode steps until KV shift)
- [ ] **T20**: Integration tests: prefill → decode round-trip, micro config, verify output shape matches standard decode

### Phase 5: GOAT Proof (6/6 Required for Default-On)
- [ ] **T21**: Proof 1 — Calibration stability: single-sequence calibration vs 10-sequence calibration produces identical partition (±0 heads). Test with 3 random seeds
- [ ] **T22**: Proof 2 — Top-p vs top-k: synthetic long-context test showing top-p achieves >90% attention mass recall with fewer tokens than top-k=4096
- [ ] **T23**: Proof 3 — Low-dim recall: 16-dim projection achieves >85% overlap with top-256 full-dim token indices across 100 random query/key pairs
- [ ] **T24**: Proof 4 — Decode throughput: head-gated decode is faster than uniform decode at seq_len ≥ 8192. Benchmark `rt_turbo` vs `dash_attn` vs baseline
- [ ] **T25**: Proof 5 — Accuracy preservation: micro benchmark perplexity within 1% of dense baseline on validation set
- [ ] **T26**: Proof 6 — Compatibility: test with `spectral_quant`, `hybrid_oct_pq`, `gdn2_attention` feature combinations. No panics, no NaN

### Phase 6: Benchmarks & Documentation
- [ ] **T27**: Create `benchmarks/012_rt_turbo_goat.md` — all 6 GOAT proof results with commands to reproduce
- [ ] **T28**: Add `rt_turbo_01_calibration` example — demonstrate offline calibration on synthetic model
- [ ] **T29**: Add `rt_turbo_02_decode_bench` example — throughput comparison: baseline vs dash_attn vs rt_turbo
- [ ] **T30**: Update `README.md` — add RTPurbo section under DashAttention, document feature gate, link to benchmark results
- [ ] **T31**: Update `.docs/` if applicable

---

## Architecture

```
src/rt_turbo/
├── mod.rs              # Module index, re-exports, feature gate
├── calibration.rs      # HeadCalibration: offline needle-based scoring
├── projection.rs       # RetrievalProjection: 16-dim W_Q/W_K per head
├── top_p.rs            # Dynamic top-p token/block selection
├── forward.rs          # Head-wise sparse decode/prefill integration
├── types.rs            # RtTurboCache, per-layer state
└── tests.rs            # Integration tests
```

## Feature Gate

```toml
# Cargo.toml
rt_turbo = ["dash_attn"]  # Requires DashAttention as base
```

```rust
// lib.rs
#[cfg(feature = "rt_turbo")]
pub mod rt_turbo;
```

## Key Design Decisions

1. **Offline calibration only** — no online head reclassification. Calibration is a one-time cost per model, serialized to disk.
2. **Pre-RoPE projection** — project BEFORE RoPE injection, matching RTPurbo's insight that high-frequency RoPE components are noise for long-range retrieval.
3. **Top-p at token level, entmax at block level** — DashAttention's α-entmax routes blocks; RTPurbo's top-p selects tokens within selected blocks. Complementary.
4. **Local heads skip projection entirely** — 85% of heads use window + sinks only, zero low-dim overhead.
5. **CPU sort-based top-p** — GPU histogram kernel not ported. Use existing `entmax_1p5` sort + cumsum pattern for consistency.

## Compatibility Matrix

| Feature | Compatible | Notes |
|---------|-----------|-------|
| `dash_attn` | ✅ Required | Base sparse attention |
| `spectral_quant` | ✅ | KV cache compression orthogonal |
| `hybrid_oct_pq` | ✅ | Block-diagonal rotation orthogonal |
| `gdn2_attention` | ✅ | Different layers/heads can use different mechanisms |
| `lt2_looped` | ⚠️ Test | Looped inference may conflict with head specialization |
| `mls_aggregate` | ✅ | Multi-layer sum independent of per-head routing |
| `tiled_attention` | ✅ | Tile-level attention can incorporate head gating |

## Expected Outcome

If GOAT proofs pass (6/6):
- Promote `rt_turbo` to default-on in `[features]` default list
- Add to production stack in README tech table
- Expected decode speedup: 1.5–2× at 32K+ context for models with clear head specialization

If GOAT proofs fail:
- Keep as opt-in `rt_turbo` feature gate
- Document negative result in `.benchmarks/`
- Record which proofs failed and why

## References

- Research 86 (RTPurbo — Retrieval Head Sparse Attention)
- Research 68 (DashAttention — α-entmax routing)
- RTPurbo paper: https://arxiv.org/pdf/2605.16928
- DuoAttention: https://openreview.net/forum?id=cFu7ze7xUm
- RazorAttention: https://openreview.net/forum?id=tkiZQlL04w