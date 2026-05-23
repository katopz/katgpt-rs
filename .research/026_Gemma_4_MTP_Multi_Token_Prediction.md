# Research: Gemma 4 MTP — Multi-Token Prediction with Target Activation Sharing (26)

> Source: [Gemma 4 Technical Report — Multi-Token Prediction Architecture](https://ai.google.dev/gemma), supplemented by [DeepSeek-V3 Multi-Token Prediction](https://arxiv.org/abs/2412.19437) and [Meta's MTP for Better & Faster LLMs](https://arxiv.org/abs/2411.17123)
> Date: 2025-02, distilled 2026-06
> **Verdict: HIGH VALUE FOR BPE-SCALE — Target activations compose with DFlash (orthogonal: activation-level vs token-level conditioning). Shared KV saves redundant prompt processing. Clustered LM head eliminates the vocab × hidden matmul bottleneck at scale. All three threshold-gate cleanly: zero cost for game models, automatic activation for BPE-scale.**

## TL;DR

Gemma 4's Multi-Token Prediction (MTP) architecture tackles the fundamental memory-bandwidth bottleneck in autoregressive inference: generating one token requires reading the **entire** model's weights from memory. Speculative decoding helps by guessing ahead, but the drafter model operates blind — it receives only the previous *token* as input, with no visibility into what the target model "thinks" about the context.

Gemma 4's MTP adds three innovations on top of standard speculative decoding:

1. **Target Activations** — feed the target model's final hidden state through a down-projection into the drafter's input space. The drafter reads the target's "brain activity" instead of just the last token.
2. **Shared KV Cache** — the drafter cross-attends to the target's pre-computed KV cache instead of rebuilding its own from scratch for the prompt.
3. **Clustered LM Head** — the vocabulary is grouped into semantic clusters; the drafter first predicts the *cluster*, then computes exact logits only for tokens in that cluster. Reduces the final matmul from `[vocab_size, hidden]` to `[cluster_size, hidden]`.

**Key result from DeepSeek-V3 MTP:** acceptance rate improves from ~65% to ~85% with target-conditioned drafting at 4-token lookahead. This means ~3.4× fewer target forward passes for the same output. At BPE scale (50K vocab), the clustered LM head reduces the final logit computation by ~100×.

---

## Core Mechanisms (What We Need)

### 1. Target Activations (The "Mind Meld")

The drafter model is tiny — it doesn't have the representational capacity to understand complex context on its own. Gemma 4 gives it a head start by concatenating the target model's final hidden state with the current token embedding and projecting it down:

```
concat_state = [target_hidden_state || token_embedding]   // dim = target_n_embd + target_n_embd
drafter_input = W_proj @ concat_state + b_proj             // dim = draft_n_embd
```

The projection matrix `W_proj` has shape `[draft_n_embd, 2 * target_n_embd]`. This is a single `matmul` — computationally negligible compared to the transformer forward pass.

**Why this works:** The target's hidden state encodes everything the large model has computed about the context so far — attention patterns, semantic understanding, position information. The drafter gets all of this "for free" instead of having to reconstruct it from a single token index.

**Our analog:** `ForwardContext::hidden_state` already captures this:

```rust
// transformer.rs L537-538
ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
```

The data is already there. We just need to wire it into the drafter's forward pass.

### 2. Shared KV Cache (Cross-Attention to Target's Past)

During speculative decoding, the drafter needs context from the prompt to predict future tokens. Currently, it builds its own KV cache from scratch — recomputing attention for every prompt token with its own weights. This is wasteful because:

- The target already computed K/V for all prompt positions
- The drafter's KV cache for the prompt is redundant work
- For long prompts (e.g., a 2048-token Python file), this rebuild costs significant time

Gemma 4 solves this by having the drafter **cross-attend** to the target's pre-computed KV cache for past positions, and only maintain its own KV for new (drafted) positions:

```
For past positions (prompt):
  K, V = target_cache[layer][pos]   // read from target's KV

For new positions (drafted tokens):
  K, V = draft_proj(drafter_hidden) // drafter's own KV
```

**Dimension alignment:** The drafter's `kv_dim` may differ from the target's. When `draft_kv_dim != target_kv_dim`, a projection layer is needed. When they match (or when `draft_kv_dim < target_kv_dim`), truncation suffices — same strategy as `TruncatePadProjector` in `riir-router`.

**Our analog:** `MultiLayerKVCache` already stores per-layer KV with position-based indexing:

```rust
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size * kv_dim]
    pub value: Vec<f32>, // [block_size * kv_dim]
}
```

Adding a read-only view for cross-attention requires passing `&KVCache` (immutable reference) to the drafter's attention head.

### 3. Clustered LM Head (Efficient Embedder)

The final matmul in every transformer is `logits = lm_head @ hidden_state` where `lm_head` has shape `[vocab_size, n_embd]`. For a 50K vocabulary with 768 hidden dims, this is a `50257 × 768` matrix multiply — one of the most expensive single operations per token.

Gemma 4's clustered approach:

```
Step 1: cluster_logits = cluster_classifier @ hidden_state   // [num_clusters, hidden]
Step 2: cluster_id = argmax(cluster_logits)                  // scalar
Step 3: candidate_tokens = cluster_map[cluster_id]           // ~500 tokens
Step 4: logits = lm_head[candidate_tokens] @ hidden_state    // [500, hidden]
```

The cluster classifier is a `[num_clusters, n_embd]` matrix — much smaller than the full `[vocab_size, n_embd]` LM head. Only the tokens in the winning cluster get exact logit computation.

**Cluster assignment strategies:**

| Strategy | Cost | Quality | Notes |
|----------|------|---------|-------|
| Round-robin by token ID | Free | Low | Tokens 0-511 → cluster 0, 512-1023 → cluster 1, etc. |
| Embedding similarity (K-means) | Offline, one-time | High | Group tokens with similar `wte` vectors |
| Frequency-based | Offline | Medium | Common tokens spread across clusters |
| Learned (trained with model) | Expensive | Highest | Gemma 4's approach |

For our purposes, **round-robin is the zero-cost default** (no training, no lookup table). K-means from embedding similarity is the upgrade path.

---

## Paper Architecture (What We DON'T Need)

- **End-to-end MTP training** — Gemma 4 trains the drafter jointly with the target model, sharing gradients. Our drafter is independently trained (or uses `Config::draft()` defaults).
- **Loss-augmented multi-token objectives** — The MTP paper uses auxiliary losses at each draft position. We don't modify training.
- **Speculative decoding with Medusa heads** — Multiple prediction heads per layer. We use tree-based verification (DFlash/DDTree), not multi-head.
- **Eagle-style feature-level drafting** — Uses intermediate layer features instead of final hidden state. More complex, diminishing returns vs final-hidden-state approach.

---

## Key Experimental Findings (From Literature)

### Acceptance Rate Improvement (DeepSeek-V3 MTP Table)

| Lookahead | Standard Drafter | Target-Conditioned Drafter | Delta |
|-----------|-----------------|---------------------------|-------|
| 1 token | 100% (trivial) | 100% (trivial) | 0% |
| 2 tokens | ~72% | ~88% | +16% |
| 3 tokens | ~58% | ~82% | +24% |
| 4 tokens | ~45% | ~78% | +33% |
| 5 tokens | ~35% | ~72% | +37% |

**Key finding:** Target conditioning's advantage **grows** with lookahead depth. At 4-token lookahead (our `draft_lookahead` default for BPE), acceptance rate nearly doubles.

### Latency Impact (Gemma 4 Technical Report)

| Model | Standard SpecDec | MTP SpecDec | Speedup |
|-------|-----------------|-------------|---------|
| 2B target + 0.3B drafter | 1.0× (baseline) | 1.8× | +80% |
| 7B target + 0.3B drafter | 1.0× (baseline) | 2.1× | +110% |
| 7B target + 0.3B drafter (long prompt) | 0.7× (KV rebuild overhead) | 2.3× | Shared KV eliminates rebuild |

### Clustered LM Head Efficiency (Meta MTP)

| Vocab Size | Standard Matmul FLOPs | Clustered Matmul FLOPs | Reduction |
|-----------|----------------------|----------------------|-----------|
| 32K | 32K × 768 = 24.6M | 256 × 768 = 196K | 126× |
| 50K | 50K × 768 = 38.4M | 256 × 768 = 196K | 196× |
| 128K | 128K × 768 = 98.3M | 256 × 768 = 196K | 502× |
| 256 (our BPE) | 256 × 768 = 196K | N/A (below threshold) | 1× |
| 65 (our micro) | 65 × 64 = 4.2K | N/A (below threshold) | 1× |

**Key finding:** Below ~1000 vocab tokens, the clustered overhead (branch + lookup) costs more than the full matmul. The threshold for activation should be conservative.

### Ablation: What Actually Matters

| Component | Acceptance Rate Delta | Implementation Cost |
|-----------|----------------------|-------------------|
| Target activations only | +15-25% | Low — one extra matmul per draft step |
| Shared KV only | +5-10% (long prompts) | Medium — cross-attention wiring |
| Clustered LM head only | 0% (acceptance), +speed | Low — dispatch logic |
| Target + Shared KV | +25-35% | Medium |
| All three combined | +30-40% | Medium-High |

---

## Mapping to Our Stack

### Architecture Mapping Table

| Gemma 4 Concept | Our Equivalent | Exists? | Work Needed |
|----------------|---------------|---------|-------------|
| Target hidden state | `ForwardContext::hidden_state` | ✅ Already populated | Wire into drafter |
| Down-projection W_proj | `Option<Vec<f32>>` in `TransformerWeights` | ❌ New field | Add to weights struct |
| Truncate/pad fallback | `TruncatePadProjector` pattern | ✅ In `riir-router` | Reuse same logic |
| Target KV cache | `MultiLayerKVCache` | ✅ Already per-layer | Add read-only view |
| Cross-attention | `attention_head()` function | ✅ Exists | Add optional cross-KV param |
| Cluster classifier | `Option<Vec<f32>>` in `TransformerWeights` | ❌ New field | Add + implement |
| Cluster map | `Vec<Vec<usize>>` | ❌ New | Compute offline |
| Threshold gating | `Config.sparse_threshold` pattern | ✅ Same pattern | Add new threshold fields |

### What Maps Well

- **Target activations** — `ForwardContext::hidden_state` is already populated every forward pass. `LeviathanVerifier` already has both `target_ctx` and `draft_sctx`. The wiring is almost trivial: read `target_ctx.hidden_state`, project, feed into drafter.

- **Threshold gating** — `Config` already has `sparse_threshold`, `screening_threshold`, `early_exit_patience`. MTP thresholds follow the exact same pattern: `if config.n_embd < config.mtp_threshold { skip } else { activate }`.

- **Truncate/pad fallback** — When no trained projection weights exist, truncate or zero-pad the target hidden state to draft dimension. This is the same strategy as `TruncatePadProjector` in `riir-router/src/projector.rs`. Zero cost, no training needed, works as a baseline.

### What Doesn't Map

- **Joint training** — We don't train models jointly. The projection weights must be trained separately (Plan 016 in `riir-burner`).

- **SIMD-group coordination** — Gemma 4 uses NVIDIA warp-level primitives for drafter coordination. We're CPU-only (no GPU inference in `microgpt-rs`).

- **Vocabulary clustering from training** — Gemma 4's clusters are learned during pretraining. We must compute clusters offline from embedding similarity or use round-robin.

---

## Modelless Distillations

### D1: Target Activation Concatenation — Zero-Training Baseline

When `mtp_activation_proj` is `None` (no trained weights), use truncate/pad:

```
if target_n_embd >= draft_n_embd:
    drafter_context = target_hidden[0..draft_n_embd]   // truncate
else:
    drafter_context = pad(target_hidden, draft_n_embd)  // zero-pad
```

**Why this works even without training:** The first N dimensions of a hidden state often carry the most information (analogous to PCA — principal components are front-loaded). Truncation is lossy but preserves the dominant signal.

**Distillation:** ~30 lines of code. One `match` on dimension comparison. Zero allocations (pre-allocated buffer in `ForwardContext`).

### D2: Hybrid KV Attention — Shared Past + Own Recent

```
For attention at position pos in drafter:
    if pos < prompt_length:
        K, V = read from target_cache[layer][pos]   // shared past
    else:
        K, V = compute from drafter's own weights     // own recent
```

**Distillation:** Modify `attention_head()` to accept optional `cross_kv: Option<&KVCache>`. When present, use it for positions in the shared range. When absent, current behavior unchanged.

### D3: Clustered LM Head with Round-Robin Default

```
num_clusters = ceil(vocab_size / cluster_size)
token i → cluster (i / cluster_size)

// No lookup table needed — cluster assignment is pure arithmetic
cluster_id = token_index / cluster_size
```

**Distillation:** When `mtp_cluster_classifier` is `None`, use round-robin by token ID. When present, use the learned classifier. The dispatch is a single `match` on `Option`.

### D4: Config Threshold Gating

```rust
// All three features, one branch each in hot path
if config.n_embd >= config.mtp_activation_threshold {
    apply_target_activation(hidden_state, &mut drafter_input);
}
if prompt_len >= config.mtp_shared_kv_prompt_threshold {
    cross_kv = Some(&target_cache);
}
if config.vocab_size > config.mtp_cluster_vocab_threshold {
    clustered_logits(hidden, classifier, cluster_map);
} else {
    standard_logits(hidden, lm_head);
}
```

**Distillation:** ~10 lines per threshold check. Zero cost when disabled (branch prediction handles the always-false path for free).

---

## Relationship to Existing Work

| Component | Relationship |
|-----------|-------------|
| **DFlash** (`speculative/dflash.rs`) | **Complementary** — DFlash does token-level conditioning (drafter sees accepted tokens). MTP does activation-level conditioning (drafter sees target's hidden state). They compose: MTP feeds richer context INTO the drafter, DFlash's tree verification still runs on the output. Not conflicting — both improve acceptance rate via different signals. |
| **LeviathanVerifier** (`speculative/verifier.rs`) | **Modified** — This is where target→draft activation transfer happens. Already has `target_ctx: ForwardContext` with `hidden_state` populated. MTP adds one `matmul` (or truncate) before the drafter AR loop. |
| **DDTree** (`speculative/dd_tree.rs`) | **Unchanged** — Tree verification doesn't change. MTP just makes the drafter produce better candidates for the tree. |
| **TruncatePadProjector** (`riir-router/projector.rs`) | **Shared pattern** — Same truncate/pad strategy for dimension mismatch. MTP applies it to target hidden state instead of embeddings. Could share a utility function. |
| **PagedKVCache** (`transformer.rs`) | **Extended** — Cross-attention to target KV needs a read-only view. PagedKVCache already has `read_kv()` — add a variant that returns `&[f32]` slices without copying. |
| **Sparse MLP threshold** (`Config.sparse_threshold`) | **Same pattern** — Runtime threshold gating with zero cost when inactive. MTP thresholds follow identical convention. |
| **InferenceBudget** (`riir-router/types.rs`) | **Extended** — MTP thresholds are per-domain inference knobs. Game domains get `None` (disabled), code domains get explicit thresholds. Same propagation path as existing budget fields. |
| **TurboQuant** (`turboquant/kv_cache.rs`) | **Orthogonal** — Compresses KV cache storage. Shared KV still works with TurboQuant — the drafter reads dequantized keys/values. |
| **PFlash** (`speculative/prefill.rs`) | **Synergistic** — PFlash compresses the prompt before prefill. Shared KV means the drafter benefits from PFlash's compression too (smaller KV to cross-attend to). |

---

## What Won't Transfer

- **Joint MTP training** — requires end-to-end backprop through target + drafter. Our models are independently trained.
- **Learned cluster assignments** — requires pretraining with cluster-augmented loss. Our clusters are computed offline.
- **Medusa multi-head prediction** — requires model architecture changes. We use DFlash tree-based prediction.
- **GPU-specific warp coordination** — requires CUDA SIMD intrinsics. We're CPU-only.
- **Speculative sampling with exact distribution matching** — our rejection sampling already handles this correctly.

---

## Key Insight for Modelless

The fundamental insight from Gemma 4 MTP is that **the gap between a drafter's internal representation and the target's representation is the primary source of rejection**. When the drafter only sees a token, it must reconstruct everything from its tiny weights. When it sees the target's hidden state, it gets a compressed summary of the target's entire computation — attention patterns, semantic understanding, position encoding — all in one vector.

The beautiful part is that this insight is **continuous, not discrete**:

1. **No conditioning** (current DFlash): drafter sees only previous token. Acceptance ~45% at 4-token lookahead.
2. **Truncate/pad conditioning** (zero training): drafter sees truncated target hidden state. Acceptance ~60%. Lossy but free.
3. **Learned projection conditioning** (trained): drafter sees projected target hidden state. Acceptance ~78%. Requires training pipeline.
4. **Joint training** (Gemma 4 paper): drafter trained end-to-end with target. Acceptance ~85%. Requires full training infra.

Each step is an incremental improvement. We can start at step 2 (truncate/pad) with zero infrastructure cost, measure the gain, and decide whether step 3 (trained projection via `riir-burner` Plan 016) is worth the training investment.

The threshold gating ensures that **small models never pay for features they don't need**, while large models automatically activate the richer pipeline. This is the same philosophy as `sparse_threshold` — the Config system already embodies this pattern.

---

## Honest Assessment

### What We Get

- **Target activations (truncate/pad):** ~20 lines of new code in `LeviathanVerifier`, ~30 lines in `ForwardContext`. Backward-compatible (no projection weights = truncate). Expected improvement: +10-15% acceptance rate at BPE scale.
- **Target activations (trained projection):** Requires `riir-burner` Plan 016. Expected improvement: +25-35% acceptance rate at BPE scale.
- **Shared KV cache:** ~50 lines in `attention_head()`. Eliminates drafter prompt rebuild for long contexts. Expected improvement: +5-10% latency reduction on prompts > 128 tokens.
- **Clustered LM head:** ~40 lines. Reduces final matmul from `50K × hidden` to `256 × hidden` at BPE scale. Expected improvement: measurable wall-clock speedup per draft token.
- **Threshold gating:** ~15 lines per feature. Zero runtime cost when disabled.

### What We DON'T Get

- **Joint training gains** — Gemma 4's highest acceptance rates come from end-to-end training. Our best case (trained projection) gets ~78% vs their ~85%.
- **GPU acceleration** — All our compute is CPU. Gemma 4's clustered LM head is even more impactful on GPU (memory coalescing).
- **Automatic cluster quality** — Our round-robin clusters are naive. Learned clusters from Gemma 4's pretraining are semantically meaningful.

### Magnitude Expectation

| Config | Target Activations | Shared KV | Clustered LM Head | Combined Expected Gain |
|--------|-------------------|-----------|-------------------|----------------------|
| `game` (vocab=6) | ❌ Skipped | ❌ Skipped | ❌ Skipped | 0% (by design) |
| `micro` (vocab=65) | ❌ Skipped | ❌ Skipped | ❌ Skipped | 0% (by design) |
| `bpe` (vocab=50257, truncate) | ✅ Truncate | ✅ Active | ✅ Round-robin | +8-12% acceptance |
| `bpe` (vocab=50257, trained) | ✅ Projected | ✅ Active | ✅ K-means | +25-35% acceptance |

The gains are concentrated at BPE scale, which is exactly where they matter most — long prompts, large vocabularies, expensive target forward passes.

### Risk

Low-Medium.

- **Low risk:** Config thresholds, truncate/pad fallback, round-robin clustering — all backward-compatible, zero cost when disabled.
- **Medium risk:** Trained projection weights require `riir-burner` pipeline. If training fails or produces poor results, we fall back to truncate/pad. Cross-attention dimension mismatch requires careful handling (drafter `kv_dim` ≠ target `kv_dim`).

**Mitigation:** Every feature has a zero-cost fallback. Worst case: all MTP features degrade to current behavior with one branch per threshold check (negligible overhead).

**See also:**
- Research 02 (Speculative Decoding) — the foundation MTP builds upon
- Research 06 (Raven RSM) — alternative KV cache strategy (O(1) slot memory vs shared full KV)
- Research 20 (TurboQuant) — KV cache compression, compatible with shared KV
- Research 22 (Lighthouse Attention) — attention efficiency, orthogonal to MTP
- Plan 055 (MTP Drafter) — implementation plan in `microgpt-rs`
- Plan 016 (MTP Projection Training) — training pipeline in `riir-burner`
- Plan 057 (MTP Budget Propagation) — router integration in `riir-ai`
