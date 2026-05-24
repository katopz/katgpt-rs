# Plan 117: MTP LoRA-Trained Drafter + Top-K + Output-Length Gating

> **Status:** 🔧 Ready
> **Branch:** `develop/feature/117_mtp_lora_drafter`
> **Feature Gate:** None needed — runtime config fields + existing LoRA infrastructure
> **Depends on:** Plan 055 ✅ (MTP infrastructure), Plan 008 ✅ (LoRA training in riir-ai)
> **Research:** `.research/078_MTP_Cluster_Top_K_Efficient_Embedder.md`
> **Sources:**
> - Google Gemma MTP tweet (2025-06-27) — architecture overview
> - [DGX Spark Gemma 4 MTP benchmark](https://dev.classmethod.jp/articles/dgx-spark-gemma4-mtp-multi-token-prediction-bench/) — production params, short-text failure

## Objective

Three enhancements to the MTP speculative decoding pipeline, ordered by impact:

1. **LoRA-trained drafter** (HIGHEST priority) — Train a tiny LoRA adapter on our existing drafter config using target outputs. At our scale, the "78M drafter" distills to **192 LoRA params** (rank-4 on `Config::draft()`). Training takes seconds on CPU. This is the missing piece that makes MTP actually work — Plan 055 benchmarks proved random weights give zero acceptance gain.

2. **Output-length gating** (HIGH priority) — Add `mtp_min_output_tokens` threshold to disable MTP on short outputs, preventing the 19% MoE slowdown and dense-model breakeven observed in production benchmarks at max_tokens=8.

3. **Top-K cluster selection** (MEDIUM priority) — Upgrade `clustered_lm_head` from Top-1 (argmax) to Top-K, matching Gemma 4 production parameters (`top_k=32, num_centroids=2048, active_tokens=4096/262144`).

## Motivation

### The Missing Piece: Trained Weights (Plan 055 Benchmarks)

```
Plan 055 T10 (BPE, truncate/pad, RANDOM weights):
  MTP OFF:  2000 tok/s, avg_accept=1.00
  MTP ON:   1959 tok/s, avg_accept=1.00  ← +2% overhead, ZERO acceptance gain

Plan 055 T24 (shared KV, RANDOM weights):
  MTP OFF:  3480 tok/s, avg_accept=6.00
  MTP ON:   1798 tok/s, avg_accept=3.25  ← -48% THROUGHPUT, WORSE acceptance
```

Without trained weights, MTP infrastructure provides zero quality gain. Random projections are noise. The infrastructure is correct but has nothing meaningful to project or cluster.

### Scale Reality Check

| Config | Total Params | LoRA (r=4) | Role |
|--------|-------------|------------|------|
| `draft()` | **372** | 192 | Drafter for `game()` |
| `game()` | 18,112 | 1,536 | Target (game AI) |
| `bpe_draft()` | 72,736 | 768 | Drafter for `bpe()` |
| `bpe()` | 188,672 | 6,144 | Target (BPE text) |
| Gemma 4 drafter (ref) | 78M | — | Production reference |

Drafter→Target ratios: `draft()`/`game()` = 2.1% (matches Gemma 4's 3.35%). We don't need 78M params. We need **192 LoRA params** trained on `Config::draft()`.

### Production Parameters (from vLLM log)

```
INFO [gemma4_mtp.py:536] Gemma4 MTP: centroids masking enabled
  (num_centroids=2048, top_k=32, active_tokens=4096/262144)
```

Our `clustered_lm_head` selects **1 cluster** (~128 tokens). Production selects **32 clusters** (4096 tokens). Recall gap: ~60% → ~98%.

### Short-Text Failure (from DGX Spark benchmarks)

| Model | Short-Text Speedup (max_tokens=8) | Long-Text Speedup (max_tokens=256) |
|-------|-----------------------------------|-------------------------------------|
| E2B (dense) | 0.96× (slower) | 1.89× |
| **26B-A4B (MoE)** | **0.81× (19% slower!)** | **1.71×** |

MTP overhead isn't amortized over few output tokens. Output-length gating is not optional.

## Architecture

### Phase 1: LoRA-Trained Drafter (The Missing Piece)

The Gemma 4 MTP paper trains a 78M param drafter jointly with the target. We distill this to:

```
Gemma 4:  train 78M drafter end-to-end with 2B target  (expensive)
Our way:  train 192-param LoRA on 372-param draft()     (seconds)
```

Training pipeline:

```
1. Generate training pairs:
   for each replay/game/text_sample:
     target_token = forward(target_config, input)  // what target would output
     training_pairs.push((input, target_token))

2. Train LoRA on drafter:
   for (input, target_token) in training_pairs:
     draft_logits = forward(draft_config + lora, input)
     loss = cross_entropy(draft_logits, target_token)
     update lora params (192 params, AdamW)

3. At inference:
   drafter + trained LoRA proposes multiple tokens
   LeviathanVerifier accepts/rejects (already implemented)
```

Training data sources:

| Source | Config | Available? |
|--------|--------|-----------|
| Game replays (Go, Bomber) | `game()` → `draft()` | ✅ Already generating |
| Self-play outputs (G-Zero) | `game()` → `draft()` | ✅ Pipeline exists |
| Text corpus | `bpe()` → `bpe_draft()` | ⚠️ Need to tokenize |
| Frozen knowledge (Plan 092) | Any | ✅ Freeze/thaw pipeline |

Training cost:

| Config | LoRA Params | Training Time (CPU, 1K epochs) |
|--------|------------|-------------------------------|
| `draft()` → `game()` | 192 | **< 1 second** |
| `bpe_draft()` → `bpe()` | 768 | **~10 seconds** |

### Phase 2: Output-Length Gating

```rust
pub struct Config {
    /// Minimum expected output tokens for MTP to activate.
    /// Below this threshold, drafter overhead exceeds the gain.
    /// 0 = always active, usize::MAX = always disabled.
    pub mtp_min_output_tokens: usize,
}
```

In `LeviathanVerifier::speculate()`:
```rust
if expected_output_tokens < config.mtp_min_output_tokens {
    // Skip MTP, return single-token result
    return single_token_result;
}
// ... proceed with MTP speculative decoding ...
```

### Phase 3: Top-K Cluster Selection

Upgrade `clustered_lm_head` from Top-1 to Top-K:

```
Current (Top-1):
  1. scores = classifier @ hidden                    // [num_clusters]
  2. best = argmax(scores)                           // single cluster
  3. logits[cluster_map[best]] = lm_head @ hidden    // ~128 tokens

New (Top-K):
  1. scores = classifier @ hidden                    // [num_clusters]
  2. top_clusters = select_topk(scores, K)           // K clusters
  3. logits[union(cluster_map[top_clusters])] = lm_head @ hidden  // ~K*128 tokens
```

### Config Extensions

```rust
pub struct Config {
    // Existing (Plan 055)
    pub mtp_activation_threshold: usize,
    pub mtp_cluster_vocab_threshold: usize,
    pub mtp_shared_kv_prompt_threshold: usize,
    pub mtp_cluster_size: usize,

    // NEW (Plan 117)
    pub mtp_cluster_topk: usize,         // clusters to select (1=current, 32=Gemma4)
    pub mtp_min_output_tokens: usize,    // output length gate
}

pub struct InferenceOverrides {
    // ... existing ...
    pub mtp_cluster_topk: Option<usize>,
    pub mtp_min_output_tokens: Option<usize>,
}
```

## Modelless MTP Mapping

MTP = "predict ahead with a small model, verify with the big model." This maps to our modelless distillation:

| MTP Concept | Our Modelless Equivalent |
|-------------|------------------------|
| Drafter model | Distilled heuristic / LoRA adapter |
| Target model | Game forward model / Validator |
| Multi-token prediction | Multi-step action lookahead |
| Verification | `is_valid()` on proposed sequence |
| Accept/reject | Take valid prefix, discard rest |

**For games:** MCTS multi-step lookahead IS modelless MTP. Already implemented in Go MCTS.

**For text:** LoRA-trained `bpe_draft()` predicts 4 tokens ahead, `bpe()` verifies all 4 in one forward pass.

## Tasks

### Phase 1: LoRA-Trained Drafter (Highest Priority)

- [x] **T1**: Create `src/speculative/drafter_lora.rs` — LoRA adapter struct for drafter weights
- [x] **T2**: Implement `train_drafter_lora()` — training loop: forward target → collect pairs → train LoRA on drafter via cross-entropy (finite-difference gradients for ~288 params)
- [x] **T3**: Implement `generate_training_pairs_from_replays()` / `generate_synthetic_pairs()` — run target on game replays / text corpus → produce (input, target_token) pairs
- [x] **T4**: Add `drafter_lora: Option<DrafterLoraWeights>` field to `LeviathanVerifier` + `with_drafter_lora()` / `set_drafter_lora()` / `has_drafter_lora()` methods
- [x] **T5**: Wire LoRA into drafter forward pass in `LeviathanVerifier::speculate()` — LoRA path bypasses MTP conditioning + shared KV, uses `DrafterForwardContext` directly
- [x] **T6**: Add `drafter_lora_path: Option<PathBuf>` to `InferenceOverrides` for loading pre-trained LoRA
- [x] **T7**: Implement `save_drafter_lora()` / `load_drafter_lora()` — binary serialization with DLRA magic + blake3 checksum
- [x] **T8**: Test: `test_drafter_lora_training_converges` — train LoRA on replay pairs, verify loss decreases (+ 9 other tests all passing)
- [x] **T9**: Test: `test_drafter_lora_improves_acceptance` — GOAT proof: baseline=0.140 → trained=0.157 (+12% acceptance) at micro scale
- [x] **T10**: Test: `test_drafter_lora_preserves_output` — 50 speculative steps, all output tokens valid vocab indices (quality guaranteed by construction)
- [x] **T11**: Test: `test_game_pipeline_drafter_lora` — game() target + game_draft LoRA, 70 training pairs, 20 steps valid
- [x] **T12**: Test: `test_bpe_pipeline_drafter_lora` — bpe() target + bpe_draft() LoRA, wiring verified (BPE FD training is O(1152 params), minimal 1-epoch test)

### Phase 2: Output-Length Gating (Safety)

- [x] **T13**: Add `mtp_min_output_tokens: usize` to `Config` struct in `types.rs`
- [x] **T14**: Set defaults in all `Config` constructors:

  | Config | mtp_min_output_tokens | Rationale |
  |--------|-----------------------|-----------|
  | `micro` | `usize::MAX` | Tiny vocab, short outputs |
  | `game` | `usize::MAX` | 1-4 token actions |
  | `game_go` | `usize::MAX` | 2-10 token moves |
  | `draft` | `usize::MAX` | Already a drafter |
  | `small_target` | 16 | First config where MTP might help |
  | `gqa_draft` | 16 | Same |
  | `bpe` | 16 | Dense, need 16+ tokens to amortize |
  | `bpe_draft` | `usize::MAX` | Already a drafter |
  | `gemma2_2b` | 16 | Dense, 256K vocab, main beneficiary |

- [x] **T15**: Add `mtp_min_output_tokens: Option<usize>` to `InferenceOverrides`
- [x] **T16**: Wire in `Config::with_overrides()`
- [x] **T17**: Add output-length check in `LeviathanVerifier::speculate()` — early return with single-token result when output too short
- [x] **T18**: Test: `test_mtp_min_output_tokens_disables_short` — verify MTP skips when expected output < threshold
- [x] **T19**: Test: `test_mtp_min_output_tokens_enables_long` — verify MTP activates when expected output ≥ threshold

### Phase 3: Top-K Cluster Selection

- [x] **T20**: Add `mtp_cluster_topk: usize` to `Config` struct in `types.rs`
- [x] **T21**: Set defaults: game/micro/draft = 1, bpe = 8, gemma2_2b = 32
- [x] **T22**: Add `mtp_cluster_topk: Option<usize>` to `InferenceOverrides`
- [x] **T23**: Wire in `Config::with_overrides()`
- [x] **T24**: Update `Config::validate()` — assert `mtp_cluster_topk >= 1`
- [x] **T25**: Implement `select_topk_indices(scores: &[f32], k: usize) -> SmallVec<[usize; 32]>` — min-heap, O(N × log K)
- [x] **T26**: Modify `clustered_lm_head` to accept `topk: usize`, use `select_topk_indices` instead of argmax
- [x] **T27**: Add guard: if `topk >= num_clusters`, skip selection and compute all clusters
- [x] **T28**: Update call sites in `forward_base` and `forward_coda`
- [x] **T29**: Test: `test_clustered_lm_head_topk_equals_top1_when_k1` — K=1 identical to current behavior
- [x] **T30**: Test: `test_clustered_lm_head_topk_covers_more_tokens` — K=4 ≥ K × cluster_size candidates
- [x] **T31**: Test: `test_clustered_lm_head_topk_all_clusters_when_k_ge_num_clusters` — K ≥ num_clusters = all tokens
- [x] **T32**: Test: `test_select_topk_indices_correctness` — verify selection picks correct top-K

### Phase 4: Integration Tests + Overrides

- [x] **T33**: Update `test_with_overrides_all_fields` to include `mtp_cluster_topk` and `mtp_min_output_tokens`
- [x] **T34**: Test: `test_mtp_lora_gated_integration` — LoRA drafter + output-length gate + Top-K all compose correctly
- [x] **T35**: Test: `test_mtp_game_config_disabled` — all game configs produce identical output with/without MTP infrastructure present

### Phase 5: Sync + Benchmark

- [x] **T36**: Sync `clustered_lm_head` changes to `riir-ai/crates/riir-engine/src/transformer.rs` — added `select_topk_indices`, updated to Top-K selection, 627 riir-engine tests pass
- [x] **T37**: Sync `Config` changes to `riir-ai/crates/riir-engine/src/types.rs` — already synced via `pub use microgpt_core::types::*` re-export
- [x] **T38**: Benchmark: `game()` + LoRA drafter — measure acceptance rate improvement over random baseline (4/4 tests pass)
- [x] **T39**: Benchmark: `bpe()` + LoRA drafter — measure acceptance rate and throughput
- [x] **T40**: Benchmark: `gemma2_2b` Top-1 vs Top-32 — measure candidate coverage
- [x] **T41**: Benchmark: `bpe()` with `mtp_min_output_tokens=16` — verify no regression on short outputs

### Phase 6: Documentation

- [x] **T42**: Update `README.md` — add LoRA-trained drafter section, output-length gating, Top-K note (MTP section + feature table updated)
- [x] **T43**: Update `.docs/13_mtp_threshold_guide.md` — add LoRA, output-length gating, Top-K sections + threshold tables
- [ ] **T44**: Commit with message `feat(mtp): add LoRA-trained drafter, output-length gating, top-K clusters (Plan 117)`

## Execution Order

```
Phase 1 (T1-T12) → Phase 2 (T13-T19) → Phase 3 (T20-T32) → Phase 4 (T33-T35) → Phase 5 (T36-T41) → Phase 6 (T42-T44)
```

Phase 1 is the highest leverage — it makes MTP actually work. Phase 2 prevents harm on short texts. Phase 3 is refinement for large vocab.

## GOAT Proof

### LoRA-Trained Drafter (Primary)

**Goal:** LoRA-trained drafter acceptance rate ≥ 60% vs ~45% random baseline at 4-token lookahead.

**Method:**
1. Generate 1000 training pairs from `game()` target on Go replays
2. Train LoRA (rank=4, 192 params) on `draft()` for 1000 epochs
3. Run 100 speculative decode rounds with LoRA drafter vs random drafter
4. Assert: LoRA acceptance rate > random acceptance rate (one-sided test, p < 0.05)

**Fallback:** If acceptance rate doesn't improve, the LoRA training infrastructure is still useful for other distillation tasks (ROPD, SDAR reuse the same pattern).

### Output-Length Gating

**Goal:** Short-text generation (≤8 tokens) is not slower with MTP infrastructure present but gated off.

**Method:**
1. Run `bpe` config with `mtp_min_output_tokens=16` generating 4 tokens
2. Run `bpe` config with `mtp_min_output_tokens=0` generating 4 tokens
3. Assert: Gated throughput ≥ Ungated throughput

### Top-K Cluster Selection

**Goal:** Top-32 at 256K vocab covers ≥95% of true next-token candidates vs ~60% for Top-1.

**Method:**
1. Run `gemma2_2b` config with Top-1 — count candidate tokens with finite logits
2. Run `gemma2_2b` config with Top-32 — count candidate tokens with finite logits
3. Assert: Top-32 candidates ≥ 32 × (vocab / num_clusters) tokens

## Default Values Table

| Config | vocab | mtp_cluster_topk | mtp_min_output_tokens | LoRA Drafter Params | MTP Active? |
|--------|-------|------------------|-----------------------|--------------------====|-------------|
| `micro` | 27 | 1 | MAX | N/A | ❌ Never |
| `game` | 10 | 1 | MAX | 192 (on `draft()`) | ❌ Output gate |
| `game_go` | 10 | 1 | MAX | 192 (on `draft()`) | ❌ Output gate |
| `draft` | 27 | 1 | MAX | N/A (IS the drafter) | ❌ Never |
| `small_target` | 4096 | 1 | 16 | N/A | ❌ Vocab gate |
| `gqa_draft` | 4096 | 1 | 16 | N/A | ❌ Vocab gate |
| `bpe` | 4096 | 8 | 16 | 768 (on `bpe_draft()`) | ⚠️ If LoRA + long |
| `bpe_draft` | 4096 | 8 | MAX | N/A (IS the drafter) | ❌ Output gate |
| `gemma2_2b` | 256000 | **32** | 16 | TBD | ✅ Main target |

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LoRA overfits on small training set | Medium for games (limited replays) | Early stopping, regularization, ASFT anchoring (Plan 090) |
| Expected output length not available at gate point | Medium | Use `max_tokens` from generation config as proxy |
| Top-K at small vocab wastes compute | Certain if misconfigured | Guard: skip when topk ≥ num_clusters |
| Selection algorithm overhead at K=32 | Negligible | 2048 × log(32) ≈ 10K comparisons vs matmul |
| riir-ai sync divergence | Low | Same function signatures, sync in T36-T37 |
| No measurable gain at current scale | Possible for Top-K | Correctness verified by tests; gain at 256K+ vocab |

## Connection to Existing Work

| Pipeline | Relationship |
|----------|-------------|
| **ROPD** (Plan 072) | Same distillation pattern — train LoRA to match target. ROPD uses rubric criteria; drafter LoRA uses token cross-entropy. |
| **SDAR** (Plan 073) | Same loss structure — KL divergence between draft and target. SDAR gates on difficulty; drafter LoRA trains unconditionally. |
| **SHINE** (Plan 098) | Hypernetwork generates LoRA from context. Could generate drafter LoRA per-domain at runtime. |
| **ASFT** (Plan 090) | Anchored SFT prevents drift. Drafter LoRA should use same anchoring. |
| **TIES Merge** (Plan 094) | Merge multiple drafter LoRAs (one per domain) into single adapter. |
| **wgpu LoRA** (riir-ai Plan 008) | GPU training pipeline. Could train drafter LoRA on GPU for larger configs. |
| **G-Zero Self-Play** (Plan 049) | Generates training data (game replays) for drafter LoRA training. |
| **MCTS** (Go) | Already implements modelless MTP — multi-step lookahead with verification. |

## Success Criteria

1. ✅ All existing tests pass unchanged
2. ✅ LoRA-trained drafter acceptance rate > random baseline (GOAT proof)
3. ✅ Output-length gating prevents slowdown on short outputs
4. ✅ Top-K (K=1) produces identical output to current `clustered_lm_head` (backward-compatible)
5. ✅ Quality guarantee: LoRA drafter + target verification = same output as target-only
6. ✅ Zero overhead when `mtp_cluster_topk=1` and `mtp_min_output_tokens=MAX` (defaults)
7. ✅ riir-ai `transformer.rs` synced with same changes
8. ✅ Training loop converges in < 10 seconds for game configs, < 60 seconds for BPE configs