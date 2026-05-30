# MTP Threshold Guide (Plan 055 + Plan 117)

## When Each Feature Activates

### Target Activations (`mtp_activation_threshold`)

**Purpose**: Feed the target model's final hidden state into the drafter, giving it richer context than just the previous token.

**Activation condition**: `target_config.n_embd >= mtp_activation_threshold`

**Threshold values**:
| Config | n_embd | Threshold | Active? |
|--------|--------|-----------|---------|
| micro | 16 | MAX | ❌ |
| game | 32 | MAX | ❌ |
| game_go | 32 | MAX | ❌ |
| draft | 4 | MAX | ❌ |
| small_target | 64 | 64 | ✅ |
| gqa_draft | 64 | 64 | ✅ |
| bpe | 32 | 32 | ✅ |
| bpe_draft | 16 | 16 | ✅ |
| gemma2_2b | 2304 | 0 | ✅ (always active) |

**Fallback**: When `mtp_activation_proj` weights are not loaded (always the case currently), uses truncate/pad — copies `min(draft_n_embd, target_n_embd)` elements from target's hidden state.

**Expected gain**: Higher draft acceptance rate on complex prompts (Python→Rust translation, etc.)

### Shared KV Cache (`mtp_shared_kv_prompt_threshold`)

**Purpose**: Preload the drafter's KV cache with the target's pre-computed keys/values for past positions.

**Activation condition**: `pos > mtp_shared_kv_prompt_threshold` AND `target_kv_dim == draft_kv_dim`

**Threshold values**:
| Config | Threshold | Active? |
|--------|-----------|---------|
| micro | MAX | ❌ |
| game | MAX | ❌ |
| game_go | MAX | ❌ |
| draft | MAX | ❌ |
| small_target | 128 | ✅ (pos > 128) |
| gqa_draft | 128 | ✅ (pos > 128) |
| bpe | 64 | ✅ (pos > 64) |
| bpe_draft | 64 | ✅ (pos > 64) |
| gemma2_2b | 8192 | ✅ (pos > 8192) |

**Constraint**: Only works when target and draft have matching `kv_dim` (n_kv_head × head_dim). When dimensions differ (e.g., bpe kv_dim=32 vs bpe_draft kv_dim=16), silently skips preload.

### Clustered LM Head (`mtp_cluster_vocab_threshold`)

**Purpose**: Two-stage vocab lookup: predict cluster → compute exact logits only for tokens in that cluster.

**Activation condition**: `vocab_size >= mtp_cluster_vocab_threshold` AND `mtp_cluster_classifier` AND `mtp_cluster_map` weights are loaded.

**Threshold values**:
| Config | vocab | Threshold | Active? |
|--------|-------|-----------|---------|
| micro | 27 | MAX | ❌ |
| game | 10 | MAX | ❌ |
| game_go | 85 | MAX | ❌ |
| draft | 27 | MAX | ❌ |
| small_target | 4096 | MAX | ❌ |
| gqa_draft | 4096 | MAX | ❌ |
| bpe | 4096 | 4096 | ✅ (when weights present) |
| bpe_draft | 4096 | 4096 | ✅ (when weights present) |
| gemma2_2b | 256000 | 256000 | ✅ (when weights present) |

**Current status**: Cluster weights are never loaded (always `None`), so the standard full-vocab LM head is always used. To activate, load trained cluster weights into `TransformerWeights::mtp_cluster_classifier` and `mtp_cluster_map`.

**Cluster assignment**: Round-robin by token ID (baseline). K-means from embedding similarity planned for riir-burner (Plan 056).

### LoRA-Trained Drafter (Plan 117 Phase 1)

**Purpose**: Train a tiny LoRA adapter on the drafter using target outputs. At our scale, the "78M drafter" distills to **288 LoRA params** (rank-4 on `Config::draft()`).

**How it works**: `DrafterLoraWeights` stores 6 rank-4 LoRA adapters (Q, K, V, O, MLP1, MLP2) per drafter layer. Standard LoRA init: A is random (Kaiming-like), B is zeros, so ΔW = B@A ≈ 0 preserves the base model at initialization. Training uses finite-difference gradients on cross-entropy loss against target token predictions.

**GOAT result**: +12% acceptance rate over random baseline at micro scale (0.157 vs 0.140).

**Threshold values**:
| Config | LoRA Params | Active? |
|--------|------------|---------|
| draft() | N/A (IS the drafter) | ❌ |
| game() → draft() | 288 | ✅ (when loaded) |
| bpe() → bpe_draft() | 1152 | ✅ (when loaded) |

**Serialization**: Binary format with `DLRA` magic + blake3 checksum via `save_drafter_lora()` / `load_drafter_lora()`.

### Output-Length Gating (`mtp_min_output_tokens`) (Plan 117 Phase 2)

**Purpose**: Disable MTP on short outputs to prevent the 19% MoE slowdown observed in production benchmarks at `max_tokens=8`.

**Activation condition**: `remaining_tokens >= mtp_min_output_tokens` → MTP active. Otherwise, single-token path.

**Threshold values**:
| Config | `mtp_min_output_tokens` | Rationale |
|--------|------------------------|-----------|
| micro | `MAX` | Tiny vocab, short outputs |
| game | `MAX` | 1-4 token actions |
| game_go | `MAX` | 2-10 token moves |
| draft | `MAX` | Already a drafter |
| small_target | 16 | First config where MTP might help |
| gqa_draft | 16 | Same |
| bpe | 16 | Dense, need 16+ tokens to amortize |
| bpe_draft | `MAX` | Already a drafter |
| gemma2_2b | 16 | Dense, 256K vocab, main beneficiary |

### Top-K Cluster Selection (`mtp_cluster_topk`) (Plan 117 Phase 3)

**Purpose**: Upgrade `clustered_lm_head` from Top-1 (argmax, ~60% recall) to Top-K (32 clusters → ~98% recall), matching Gemma 4 production parameters.

**Activation condition**: `mtp_cluster_topk > 1` AND clustered LM head is active (vocab threshold + weights present).

**Threshold values**:
| Config | `mtp_cluster_topk` | Rationale |
|--------|--------------------|----------|
| micro | 1 | No clustering |
| game | 1 | No clustering |
| game_go | 1 | No clustering |
| draft | 1 | No clustering |
| small_target | 1 | No clustering |
| gqa_draft | 1 | No clustering |
| bpe | 8 | Medium vocab |
| bpe_draft | 1 | Draft model, no clustering |
| gemma2_2b | 1 | Cluster weights not yet trained |

**Guard**: When `topk >= num_clusters`, all clusters are selected (no pruning, same as full vocab).

## Overriding at Inference Time

All MTP thresholds can be overridden via `InferenceOverrides`:

```rust
let overrides = InferenceOverrides {
    mtp_activation_threshold: Some(64),
    mtp_shared_kv_prompt_threshold: Some(128),
    mtp_cluster_topk: Some(8),               // Plan 117: Top-K cluster selection
    mtp_min_output_tokens: Some(16),         // Plan 117: output-length gating
    drafter_lora_path: Some("model.dlra".into()), // Plan 117: LoRA drafter weights
    ..Default::default()
};
let config = Config::bpe().with_overrides(&overrides);
```

## Validation

`Config::validate()` enforces:
- `mtp_cluster_size > 0` — cluster size must be positive (only checked for `ModelArchitecture::Generic`; Gemma 2 and Llama skip this check)
- `mtp_cluster_topk >= 1` — top-K must be at least 1 (Plan 117; applies to all architectures)

Other thresholds use `usize::MAX` as "disabled" sentinel, which is valid and needs no special enforcement.

## Architecture Diagram

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

## Composability with Existing Features

| Feature | Relationship |
|---------|-------------|
| DFlash (`speculative/dflash.rs`) | **Orthogonal** — MTP feeds richer context INTO the drafter. DFlash's tree verification still runs on the output. |
| LeviathanVerifier (`speculative/verifier.rs`) | **Modified** — target→draft activation transfer happens here (target already exposes `hidden_state`) |
| PagedKVCache (`transformer.rs`) | **Extended** — read-only cross-attention view for drafter |
| Sparse MLP threshold (`Config.sparse_threshold`) | **Same pattern** — threshold-gated feature activation |
| TurboQuant | **Independent** — compresses precision, MTP improves draft quality |
| PFlash | **Independent** — compresses sequence, MTP improves draft quality |

## References

- [Gemma 4 architecture](https://blog.google/technology/ai/gemma-technical-report/) — Multi-Token Prediction design
- [DGX Spark Gemma 4 MTP benchmark](https://dev.classmethod.jp/articles/dgx-spark-gemma4-mtp-multi-token-prediction-bench/) — production params, short-text failure
- Plan 055 — `katgpt-rs/.plans/055_gemma_mtp_drafter.md`
- Plan 056 — riir-burner cluster weight training
- Plan 117 — `katgpt-rs/.plans/117_mtp_cluster_topk_efficient_embedder.md`
- 🧪 `tests/bench_117_mtp_lora_topk_goat.rs` — LoRA acceptance, Top-K coverage, output-length gating (4/4 pass)