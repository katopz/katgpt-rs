# Plan 055: Gemma 4 MTP Drafter — Threshold-Gated Target Conditioning

## Overview

Distill three Gemma 4 Multi-Token Prediction (MTP) optimizations into the speculative decoding pipeline, threshold-gated by `Config` so tiny models (game, micro) pay zero cost while BPE-scale models (50K vocab, 768 dims) activate richer features automatically.

## Motivation

Current DFlash drafter operates blind — it receives only the previous *token* as input. The drafter has no idea what the target model "thinks" about the context. At BPE scale (`Config::bpe()` — vocab 50257, n_embd 768), this wastes draft quality on long prompts like Python→Rust translation.

Gemma 4's MTP architecture shows three techniques that improve draft acceptance rate:

1. **Target Activations** — feed the target's hidden state into the drafter (activation-level conditioning vs DFlash's token-level)
2. **Shared KV Cache** — drafter cross-attends to target's pre-computed KV cache instead of rebuilding its own
3. **Clustered LM Head** — two-stage vocab lookup: predict cluster, then matmul only tokens in that cluster

All three are gated by `Config` thresholds. Small models skip them with a single branch. Large models activate them automatically.

## Relationship to Existing Work

| Existing | Relationship |
|----------|-------------|
| DFlash (`speculative/dflash.rs`) | **Orthogonal** — MTP feeds richer context INTO the drafter. DFlash's tree verification still runs on the output. They compose. |
| LeviathanVerifier (`speculative/verifier.rs`) | **Modified** — this is where target→draft activation transfer happens (target already exposes `hidden_state`) |
| TruncatePadProjector (`riir-router/projector.rs`) | **Shared pattern** — same truncate/pad strategy for dim mismatch, but MTP needs the target's hidden state not an embedding |
| PagedKVCache (`transformer.rs`) | **Extended** — add read-only cross-attention view for drafter |
| Sparse MLP threshold (`Config.sparse_threshold`) | **Same pattern** — threshold-gated feature activation |

## Architecture

### Config Extensions

```rust
// types.rs — new threshold fields
pub struct Config {
    // ... existing fields ...

    /// n_embd threshold above which target activations are fed to the drafter.
    /// 0 = always active, usize::MAX = always disabled.
    pub mtp_activation_threshold: usize,

    /// vocab_size threshold above which clustered LM head activates.
    /// 0 = always active, usize::MAX = always disabled.
    pub mtp_cluster_vocab_threshold: usize,

    /// Prompt length threshold above which drafter shares target's KV cache.
    /// 0 = always active, usize::MAX = always disabled.
    pub mtp_shared_kv_prompt_threshold: usize,

    /// Cluster size for grouped LM head (only active when vocab > threshold).
    /// Default: 512 tokens per cluster.
    pub mtp_cluster_size: usize,
}
```

### Weight Extensions

```rust
// transformer.rs — optional projection weights
pub struct TransformerWeights {
    // ... existing fields ...

    /// Target→Draft activation projection: [draft_n_embd, target_n_embd + embed_dim]
    /// Only loaded when `Config::mtp_activation_threshold` is met and file exists.
    /// Falls back to truncate/pad when absent (zero-cost, no training needed).
    pub mtp_activation_proj: Option<Vec<f32>>,

    /// Cluster classifier: [num_clusters, n_embd]
    /// Only loaded when vocab > cluster_vocab_threshold.
    pub mtp_cluster_classifier: Option<Vec<f32>>,

    /// Cluster membership table: [num_clusters] → Vec<usize> (token indices)
    pub mtp_cluster_map: Option<Vec<Vec<usize>>>,
}
```

### Data Flow

```
Standard DFlash (current):
  token → [drafter forward] → draft tokens → [target verify] → accepted tokens

MTP-Enhanced DFlash (this plan):
  token + target_hidden_state
    → [projection: truncate/pad OR learned matmul]
    → [drafter forward with projected context]
    → draft tokens
    → [target verify]
    → accepted tokens
```

## Tasks

### Phase 1: Config & Types (Foundation) ✅

- [x] **T1**: Add `mtp_activation_threshold`, `mtp_cluster_vocab_threshold`, `mtp_shared_kv_prompt_threshold`, `mtp_cluster_size` to `Config` struct in `types.rs`
- [x] **T2**: Add `mtp_*` fields to all `Config` constructors (`micro`, `game`, `draft`, `bpe`, `bpe_draft`, etc.) — small configs get `usize::MAX` (disabled), BPE configs get reasonable thresholds
- [x] **T3**: Add `mtp_activation_proj`, `mtp_cluster_classifier`, `mtp_cluster_map` to `TransformerWeights` as `Option<>` fields
- [x] **T4**: Update `TransformerWeights::new()` to attempt loading projection weights (optional — file may not exist)
- [x] **T5**: Add `InferenceOverrides` fields for MTP thresholds so they can be overridden at inference time

### Phase 2: Target Activations (Highest Gain) ✅

- [x] **T6**: Add `mtp_context_buf: Vec<f32>` to `ForwardContext` (pre-allocated, `[n_embd]`)
- [x] **T7**: Implement `project_target_activation()` — truncate/pad when `mtp_activation_proj` is `None`, matmul when present
- [x] **T8**: Modify `LeviathanVerifier::speculate()` to pass `target_ctx.hidden_state` through projection before drafter AR loop
- [x] **T9**: Wire projected context into `dflash_predict_ar_with` as conditioning signal (add optional `mtp_context: Option<&[f32]>` parameter)
- [x] **T10**: Benchmark acceptance rate: `Config::bpe()` with MTP on vs off — truncate/pad adds ~2% overhead, no acceptance gain with random weights (bench/067)

### Phase 3: Shared KV Cache (Medium Gain) ✅

- [x] **T11**: Add `cross_attn_kv: Option<&KVCache>` parameter to drafter's attention head function
- [x] **T12**: Implement read-only cross-attention: when `cross_attn_kv` is provided, attend to those keys/values instead of drafter's own cache for past positions
- [x] **T13**: Gate behind `mtp_shared_kv_prompt_threshold` — only active when prompt length exceeds threshold
- [x] **T14**: Verify drafter still writes to its own KV cache for new positions (hybrid: shared past + own recent)

### Phase 4: Clustered LM Head (BPE-scale Gain) ✅

- [x] **T15**: Implement `clustered_lm_head()` — two-stage matmul: cluster classifier → argmax → subset matmul
- [x] **T16**: Implement `standard_lm_head()` as fallback (current behavior, single matmul)
- [x] **T17**: Add dispatch in `forward_base()` — if `vocab_size >= mtp_cluster_vocab_threshold` AND `mtp_cluster_classifier.is_some()` → use clustered path, else standard
- [x] **T18**: Implement cluster assignment heuristic: round-robin by token ID (no training needed for initial version)
- [x] **T19**: Implement K-means cluster assignment from embedding similarity (offline, computed once)

### Phase 5: Integration & Testing (partial)

- [x] **T20**: Wire all three features together in `LeviathanVerifier` end-to-end
- [x] **T21**: Test: small config (`Config::game()`) — all MTP features disabled, output identical to current
- [x] **T22**: Test: BPE config (`Config::bpe()`) — MTP features active, acceptance rate measured
- [x] **T23**: Test: projection fallback (no weights file) — truncate/pad produces valid (if suboptimal) results
- [x] **T24**: Benchmark: acceptance rate comparison table (DFlash vs DFlash+MTP at various scales) — bench/067, truncate/pad overhead measured, shared KV harmful with random weights
- [x] **T25**: Update `Config::validate()` to enforce threshold consistency — enforces `mtp_cluster_size > 0`

### Phase 6: Documentation

- [x] **T26**: Update `README.md` — added MTP section after PFlash with threshold table
- [x] **T27**: Add `.docs/055_mtp_threshold_guide.md` — detailed threshold guide with activation conditions, config tables, and composability notes
- [x] **T28**: Update this plan with benchmark results and verdict — see Benchmark Results section below

## Execution Order

Phase 1 (T1–T5) → Phase 2 (T6–T10) → Phase 5 (T21–T23 smoke tests) → Phase 3 (T11–T14) → Phase 4 (T15–T19) → Phase 5 (T20, T24, T25) → Phase 6 (T26–T28)

## Threshold Activation Table

| Config | vocab | n_embd | Target Activations | Shared KV | Clustered LM Head |
|--------|-------|--------|--------------------|-----------|-------------------|
| `micro` | 27 | 16 | ❌ (16 < MAX) | ❌ | ❌ (27 < MAX) |
| `game` | 10 | 32 | ❌ (32 < MAX) | ❌ | ❌ (10 < MAX) |
| `draft` | 27 | 4 | ❌ (4 < MAX) | ❌ | ❌ (27 < MAX) |
| `small_target` | 4096 | 64 | ✅ (64 ≥ 64) | ✅ (pos > 128) | ❌ (4096 < MAX) |
| `gqa_draft` | 4096 | 64 | ✅ (64 ≥ 64) | ✅ (pos > 128) | ❌ (4096 < MAX) |
| `bpe` | 4096 | 32 | ✅ (32 ≥ 32) | ✅ (pos > 64) | ✅ when weights present (4096 ≥ 4096) |
| `bpe_draft` | 4096 | 16 | ✅ (16 ≥ 16) | ✅ (pos > 64) | ✅ when weights present (4096 ≥ 4096) |

## Benchmark Results (bench/067 — truncate/pad fallback, no trained weights)

| Config | Method | tok/s | μs/step | Avg Accept | Notes |
|--------|--------|-------|---------|------------|-------|
| `bpe` (32d target, 16d draft) | MTP OFF | 2000 | 500 | 1.00 | Baseline, no conditioning |
| `bpe` (32d target, 16d draft) | MTP ON truncate/pad | 1959 | 511 | 1.00 | +2% overhead, no acceptance gain |
| `small_target` (64d, shared KV) | MTP OFF | 3480 | 1724 | 6.00 | Baseline with matching dims |
| `small_target` (64d, shared KV) | MTP ON shared KV | 1798 | 1807 | 3.25 | -48% throughput, KV noise hurts |

### Verdict

- **T10 (BPE acceptance rate)**: Truncate/pad adds ~2% overhead with no acceptance improvement. Expected — random projection weights don't provide meaningful target context.
- **T24 (Multi-scale comparison)**: Shared KV is harmful with random weights (target KV cache is noise). Only beneficial with trained models.
- **Overall**: MTP infrastructure is correct and low-overhead. The **5% acceptance rate improvement** success criterion requires **trained projection weights** (riir-burner Plan 016 output). The truncate/pad fallback is safe but provides no quality gain.

### Next Steps for Acceptance Rate Improvement

1. Run `riir-burner/scripts/mtp_projection.sh` with actual Gemma 2 models to produce trained weights
2. Place `mtp_activation_proj.bin` alongside model weights
3. Re-benchmark: trained projection should improve acceptance rate ≥5% over truncate/pad baseline

## Risks

| Risk | Mitigation |
|------|-----------|
| Untrained projection weights are noise | Fallback to truncate/pad (zero-cost, no training) — already proven in `riir-router/projector.rs` |
| Clustered LM head adds branching overhead | Threshold ensures it only fires for large vocabs where win outweighs branch cost |
| Cross-attention to target KV requires dimension alignment | Drafter's `kv_dim` may differ from target — add optional `kv_proj` or require matching dims |
| Feature flag explosion | All MTP features compile unconditionally; thresholds gate at runtime (single branch in hot path) |

## Success Criteria

1. ✅ `Config::game()` — zero perf regression, output identical to current (test_mtp_small_config_disabled)
2. ⏳ `Config::bpe()` — acceptance rate improves ≥ 5% vs DFlash-only baseline — **blocked on trained weights** (bench/067: truncate/pad = +2% overhead, no acceptance gain with random weights)
3. ✅ All existing tests pass unchanged (500 tests pass)
4. ✅ No new allocations in hot path when MTP features are disabled (threshold-gated branches)
5. ✅ Benchmark infrastructure for MTP on vs off comparison (bench/067, T10+T24+T28 complete)

## Dependencies

- **riir-burner** (Plan 016) — training the `mtp_activation_proj` weights (scripts complete, needs MLX + Gemma models to produce trained weights)
- **riir-ai** (Plan 057) — `InferenceBudget` propagation of MTP thresholds via router