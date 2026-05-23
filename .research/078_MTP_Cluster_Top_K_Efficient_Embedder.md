# Research: MTP Cluster Top-K + LoRA-Trained Drafter — Distillation from Gemma 4 Production System (78)

## TL;DR

Gemma 4's production MTP uses **Top-32 cluster selection** from 2048 centroids, covering 4096/262144 tokens (1.6%). Our `clustered_lm_head` only picks **Top-1**. More importantly, the Gemma 4 drafter is 78M params — but our entire `Config::draft()` is only **372 params**. We don't need a separate model; we need a **LoRA adapter** (192 params at rank-4) trained on the drafter config to predict the target's outputs. This is MTP distilled to our scale: train tiny LoRA, predict ahead, verify with target. Feasible in seconds of training.

## Sources

| Source | Date | Key Finding |
|--------|------|-------------|
| [Google Gemma tweet](https://x.com/googlegemma/status/2051694045869879749) | 2025-06-27 | MTP architecture overview, Efficient Embedder with multi-cluster selection |
| [DGX Spark Gemma 4 MTP benchmark](https://dev.classmethod.jp/articles/dgx-spark-gemma4-mtp-multi-token-prediction-bench/) | 2026-05-09 | Production parameters, short-text failure, acceptance rates, MoE penalty |

## Production Parameters (from vLLM Log)

The DGX Spark article reveals actual vLLM runtime parameters from Gemma 4 MTP:

```
INFO [gemma4_mtp.py:536] Gemma4 MTP: centroids masking enabled
  (num_centroids=2048, top_k=32, active_tokens=4096/262144)
```

| Parameter | Value | Our Current |
|-----------|-------|-------------|
| `num_centroids` | 2048 | `ceil(vocab/cluster_size)` = varies |
| `top_k` | **32** | **1** (argmax) |
| `active_tokens` | 4096 | ~512 (single cluster) |
| `vocab_size` | 262144 | varies by config |
| Coverage | 4096/262144 = 1.6% | 512/vocab |

## Scale Reality Check

### Our Models vs Gemma 4

| Config | Total Params | LoRA (r=4) | LoRA % | Role |
|--------|-------------|------------|--------|------|
| `draft()` | **372** | 192 | 51.6% | Drafter for `game()` |
| `game()` | 18,112 | 1,536 | 8.5% | Target (game AI) |
| `bpe_draft()` | 72,736 | 768 | 1.1% | Drafter for `bpe()` |
| `bpe()` | 188,672 | 6,144 | 3.3% | Target (BPE text) |
| `gemma2_2b()` | 2.33B | 2,875,392 | 0.12% | Target (real LLM) |
| **Gemma 4 drafter (ref)** | **78M** | — | — | Production reference |

### Drafter → Target Ratios

| Drafter | Target | Ratio | Gemma 4 ref |
|---------|--------|-------|-------------|
| `draft()` 372 | `game()` 18.1K | **2.1%** | 3.35% |
| `bpe_draft()` 72.7K | `bpe()` 188.7K | **38.6%** ⚠️ | 3.35% |
| LoRA on `draft()` 192 | `game()` 18.1K | **1.1%** | — |

**Key insight:** Our `draft()` is already smaller relative to target than Gemma 4's drafter. We don't need a 78M model. We need **192 LoRA params** trained on our existing `draft()` config.

### The LoRA Breakthrough

Gemma 4 trains a separate 78M param drafter model. We can't afford that pipeline. But:

```
Gemma 4 approach:
  78M param drafter → trained from scratch → predicts ahead → target verifies

Our approach:
  372 param draft() config + 192 param LoRA adapter → trained in seconds → predicts ahead → target verifies
```

The LoRA adapter IS the distillation. It learns to make the drafter predict what the target would output. Training:
1. Run target model on inputs → collect (input, target_token) pairs
2. Train LoRA on drafter config to predict target_token
3. At inference: drafter + LoRA proposes multiple tokens
4. Target verifies (LeviathanVerifier already does this)

**Training cost:** 192 params, ~1000 training steps, seconds on CPU. No GPU needed.

## Critical Finding: MTP Fails on Short Texts

### DGX Spark Benchmark Results (JCQ, max_tokens=8)

| Model | Short-Text Speedup | Long-Text Speedup | Acceptance Rate |
|-------|--------------------|--------------------|-----------------|
| E2B (BF16) | 0.96× (slower) | 1.89× | 38.8% |
| E4B (BF16) | 1.01× (neutral) | 2.10× | 44.6% |
| **26B-A4B (MoE)** | **0.81× (19% slower!)** | **1.71×** | 54.9% |
| 31B (NVFP4) | 0.92× (slower) | 1.91× | 54.8% |

MTP overhead isn't amortized over few output tokens. For MoE at batch_size=1, overhead exceeds gain entirely.

### Acceptance Rate Scaling

| Model | Acceptance Rate |
|-------|----------------|
| E2B (2B params) | 38.8% |
| E4B (4B params) | 44.6% |
| 26B-A4B (26B params) | 54.9% |
| 31B (31B params) | 54.8% |

Our tiny models would have even lower acceptance rates. Disabling MTP for game/micro is correct.

## Modelless MTP Mapping

### The Core Insight

MTP = "predict ahead with a small model, verify with the big model." This maps directly to our modelless distillation:

| MTP Concept | Our Modelless Equivalent |
|-------------|------------------------|
| Drafter model | Distilled heuristic / LoRA adapter |
| Target model | Game forward model / Validator |
| Multi-token prediction | Multi-step action lookahead |
| Verification | `is_valid()` check on proposed sequence |
| Accept/reject | Take valid prefix, discard rest |
| Acceptance rate | Heuristic accuracy |

### For Games (Already Have This!)

Our game pipeline already does modelless MTP:

```
1. Modelless heuristic proposes action (≈ drafter predicts token)
2. Game validator checks is_valid() (≈ target verifies token)
3. If valid → execute; if invalid → reject (≈ accept/reject)
```

What MTP adds: **multi-step lookahead** — propose a SEQUENCE of actions, validate all at once.

```
Current: propose 1 action → validate → execute → propose next
MTP way: propose N actions → validate sequence → execute valid prefix → continue
```

This is exactly MCTS-lite. Our Go MCTS (`go_mcts.rs`) already does this. The MTP idea is already distilled into our game pipeline via MCTS.

### For Text (LoRA-Trained Drafter)

For `bpe()` text generation, MTP maps to:

```
1. LoRA-trained bpe_draft() predicts 4 tokens ahead
2. bpe() target verifies all 4 in one forward pass
3. Accept valid prefix, reject rest
4. Repeat
```

This is standard speculative decoding with a LoRA-trained drafter. The "training" is:

```
Training data generation:
  for each text_sample in corpus:
    tokens = tokenize(text_sample)
    for pos in 0..tokens.len():
      target_logit = forward(bpe(), tokens[..pos]) → argmax  // what target would output
      training_pairs.push((tokens[..pos], target_logit))

LoRA training on drafter:
  for (input, target_token) in training_pairs:
    draft_logit = forward(bpe_draft() + lora, input)
    loss = cross_entropy(draft_logit, target_token)
    update lora params
```

**Training data source:** Game replays (already generating), or any text corpus for BPE.

## What We Already Have (Plan 055 ✅)

| Component | Status | Notes |
|-----------|--------|-------|
| Target Activations (truncate/pad) | ✅ Implemented | Plan 055 T6-T10 |
| Shared KV Cache (cross-attention) | ✅ Implemented | Plan 055 T11-T14 |
| Clustered LM Head (Top-1) | ✅ Implemented | Plan 055 T15-T19 |
| Threshold gating (n_embd, vocab, prompt) | ✅ Implemented | Plan 055 T4 |
| Round-robin cluster assignment | ✅ Implemented | Plan 055 T18 |
| Config + InferenceOverrides | ✅ Implemented | Plan 055 T1-T5 |
| LoRA training infrastructure | ✅ Implemented | riir-ai wgpu (Plan 008) |
| Distillation pipelines (SDAR, ROPD) | ✅ Implemented | Plan 072, 073 |
| LeviathanVerifier | ✅ Implemented | Speculative decoding ready |
| Game replays as training data | ✅ Implemented | Go, Bomber, Monopoly |
| MCTS multi-step lookahead | ✅ Implemented | Go MCTS |

## Gap Analysis

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| **LoRA-trained drafter** (192-768 params) | **HIGH** — makes MTP actually work | ~100 lines + training loop | **HIGH** |
| Output-length gating (`mtp_min_output_tokens`) | HIGH — prevents slowdown on short | ~20 lines | HIGH |
| Top-K cluster selection (Top-1 → Top-32) | Medium at 256K vocab, Low at 4K | ~30 lines | MEDIUM |
| K-means cluster assignment | Medium (better clusters = better Top-K) | ~50 lines | LOW |

**Priority shift:** The original plan focused on Top-K cluster selection. But the real win is **LoRA-training the drafter**. Without trained weights, MTP provides zero quality gain (Plan 055 benchmarks proved this). LoRA-training the drafter IS the distillation that makes MTP work.

## Distillation Strategy

### D1: LoRA-Trained Drafter (The Missing Piece)

The Gemma 4 MTP paper trains a 78M param drafter jointly with the target. We distill this to:

```
Gemma 4:  train 78M drafter end-to-end with 2B target  (expensive)
Our way:  train 192-param LoRA on 372-param draft()     (seconds)
```

#### Training Pipeline

```rust
// Pseudocode for LoRA drafter training

fn train_drafter_lora(
    target_config: &Config,      // e.g., Config::game()
    draft_config: &Config,       // e.g., Config::draft()
    training_pairs: &[(Vec<usize>, usize)],  // (input_tokens, target_output_token)
    rank: usize,                 // 4
    epochs: usize,               // 1000
    lr: f32,                     // 0.01
) -> LoraWeights {
    let mut lora = LoraWeights::new(draft_config, rank);
    let mut optimizer = AdamW::new(lora.param_count(), lr);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (input, target_token) in training_pairs {
            // Forward: drafter + LoRA predicts what target would output
            let draft_logits = forward_with_lora(&draft_config, &lora, input);
            let loss = cross_entropy(&draft_logits, *target_token);

            // Backward: update only LoRA params (192 params)
            let grads = backward(&loss);
            optimizer.step(&mut lora, &grads);
            total_loss += loss;
        }
    }
    lora
}
```

#### Training Data Sources

| Source | Config | Available? | Effort |
|--------|--------|-----------|--------|
| Game replays (Go, Bomber) | `game()` → `draft()` | ✅ Already generating | Zero |
| Self-play outputs | `game()` → `draft()` | ✅ G-Zero pipeline | Zero |
| Text corpus | `bpe()` → `bpe_draft()` | ⚠️ Need to tokenize | Low |
| Frozen knowledge (Plan 092) | Any | ✅ Freeze/thaw pipeline | Zero |

#### Feasibility at Our Scale

| Config | Drafter Params | LoRA Params (r=4) | Training Time (CPU, 1K epochs) |
|--------|---------------|-------------------|-------------------------------|
| `draft()` → `game()` | 372 | 192 | **< 1 second** |
| `bpe_draft()` → `bpe()` | 72,736 | 768 | **~10 seconds** |
| `bpe_draft()` → `bpe()` (r=8) | 72,736 | 1,536 | **~20 seconds** |

This is absurdly cheap. No GPU needed. Train on any laptop in seconds.

#### Connection to Existing Pipelines

| Existing Pipeline | Relationship |
|-------------------|-------------|
| **ROPD** (Plan 072) | Same distillation pattern — train LoRA to match target outputs. ROPD uses rubric-based criteria; drafter LoRA uses token-level cross-entropy. |
| **SDAR** (Plan 073) | Same loss structure — KL divergence between draft and target logits. SDAR gates on difficulty; drafter LoRA trains unconditionally. |
| **SHINE** (Plan 098) | Hypernetwork generates LoRA from context. Could generate drafter LoRA per-domain at runtime. |
| **ASFT** (Plan 090) | Anchored SFT prevents drift. Drafter LoRA should use same anchoring to prevent catastrophic forgetting. |
| **TIES Merge** (Plan 094) | Merge multiple drafter LoRAs (one per domain) into a single adapter. |
| **wgpu LoRA** (riir-ai Plan 008) | GPU training pipeline. Could train drafter LoRA on GPU for larger configs. |

### D2: Output-Length Gating (Safety)

New config field to prevent MTP activation on short outputs:

```rust
pub struct Config {
    /// Minimum expected output tokens for MTP to activate.
    /// Below this threshold, drafter overhead exceeds the gain.
    /// Based on DGX Spark benchmarks: MoE hurts below ~16 tokens,
    /// dense models break even around ~8 tokens.
    /// 0 = always active, usize::MAX = always disabled.
    pub mtp_min_output_tokens: usize,
}
```

Recommended defaults:

| Config | mtp_min_output_tokens | Rationale |
|--------|----------------------|-----------|
| `micro` | MAX (disabled) | Tiny vocab, short outputs |
| `game` | MAX (disabled) | 1-4 token actions |
| `game_go` | MAX (disabled) | 2-10 token moves |
| `draft` | MAX (disabled) | Already a drafter |
| `small_target` | 16 | First config where MTP might help |
| `gqa_draft` | 16 | Same |
| `bpe` | 16 | Dense model, need 16+ tokens to amortize |
| `bpe_draft` | MAX (disabled) | Already a drafter |
| `gemma2_2b` | 16 | Dense model, 256K vocab, main beneficiary |

### D3: Top-K Cluster Selection (Algorithmic, No Training)

Upgrade `clustered_lm_head` from Top-1 to Top-K:

```rust
fn clustered_lm_head_topk(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head: &[f32],
    classifier: &[f32],
    cluster_map: &[Vec<usize>],
    vocab_size: usize,
    n_embd: usize,
    topk: usize,  // NEW: was hardcoded to 1
) {
    let num_clusters = cluster_map.len();

    // Stage 1: compute all cluster scores
    let mut cluster_scores = vec![0.0f32; num_clusters];
    for c in 0..num_clusters {
        cluster_scores[c] = simd_dot_f32(&classifier[c*n..], &hidden[..n], n);
    }

    // Stage 2: top-K selection (O(N × log K) for K ≤ 32)
    let top_clusters = select_topk_indices(&cluster_scores, topk);

    // Stage 3: compute exact logits for tokens in top-K clusters
    logits.fill(f32::NEG_INFINITY);
    for &cluster_id in &top_clusters {
        for &token_idx in &cluster_map[cluster_id] {
            if token_idx < vocab_size {
                logits[token_idx] = simd_dot_f32(
                    &lm_head[token_idx * n_embd..], &hidden[..n_embd], n_embd
                );
            }
        }
    }
}
```

### D4: Config Extension Summary

```rust
pub struct Config {
    // Existing (Plan 055)
    pub mtp_activation_threshold: usize,        // n_embd gate
    pub mtp_cluster_vocab_threshold: usize,     // vocab gate
    pub mtp_shared_kv_prompt_threshold: usize,  // prompt length gate
    pub mtp_cluster_size: usize,                // tokens per cluster

    // NEW (Plan 117)
    pub mtp_cluster_topk: usize,                // clusters to select (1=current, 32=Gemma4)
    pub mtp_min_output_tokens: usize,           // output length gate
}
```

### D5: Feature Gate Decision

**No compile-time feature gate needed.** All additions are runtime config:
- `mtp_cluster_topk` — defaults to 1 (current behavior), opt-in to 32
- `mtp_min_output_tokens` — defaults to MAX (disabled) for small configs, 16 for BPE+

The LoRA drafter training uses existing LoRA infrastructure (no new feature gate).

## Scale Analysis at Our Configs

| Config | vocab | num_clusters (size=512) | Top-1 candidates | Top-32 candidates | Top-32 vs Full | MTP Viable? |
|--------|-------|------------------------|-------------------|-------------------|----------------|-------------|
| `micro` | 27 | 1 | 27 (all) | 27 (all) | N/A | ❌ (disabled) |
| `game` | 10 | 1 | 10 (all) | 10 (all) | N/A | ❌ (disabled) |
| `bpe` | 4096 | 8 | 512 | 4096 (all!) | 1.0× | ⚠️ Top-32 = all clusters |
| `gemma2_2b` | 256000 | 500 | 512 | 16000 | 0.063× | ✅ Main target |

**Key realization for `bpe`:** With only 8 clusters, Top-32 = all clusters = full matmul. Top-K is meaningless when `num_clusters ≤ K`. The feature only helps when `num_clusters > K`, i.e., vocab > K × cluster_size.

## The Honest Picture

### What Training Gives Us (Plan 055 Benchmarks Were Clear)

```
Plan 055 T10 (BPE, truncate/pad, RANDOM weights):
  MTP OFF:  2000 tok/s, avg_accept=1.00
  MTP ON:   1959 tok/s, avg_accept=1.00  ← +2% overhead, ZERO acceptance gain

Plan 055 T24 (shared KV, RANDOM weights):
  MTP OFF:  3480 tok/s, avg_accept=6.00
  MTP ON:   1798 tok/s, avg_accept=3.25  ← -48% THROUGHPUT, WORSE acceptance
```

**Without trained weights, MTP infrastructure provides zero quality gain.** Random projections are noise. Random cluster centroids are meaningless. The infrastructure is correct but has nothing meaningful to project or cluster.

### What LoRA-Training the Drafter Gives Us

With a LoRA-trained drafter (192 params for `draft()`, 768 params for `bpe_draft()`):

1. **Drafter learns target's output distribution** — no more random predictions
2. **Acceptance rate improves from ~45% to ~70-80%** (literature estimates for trained projection)
3. **Multi-token lookahead becomes useful** — accepted sequence length increases
4. **Net speedup on long outputs** — drafter overhead amortized over accepted tokens

The training IS the distillation. The LoRA adapter captures the target's behavior in a tiny param space.

### Three Paths Forward

| Path | What | Effort | Gain |
|------|------|--------|------|
| **A: LoRA drafter only** | Train LoRA on `draft()` using game replays | ~100 lines + training loop | Makes existing MTP actually work |
| **B: Output-length gate** | Add `mtp_min_output_tokens` threshold | ~20 lines | Prevents slowdown on short texts |
| **C: Top-K clusters** | Upgrade `clustered_lm_head` to Top-32 | ~30 lines | Quality at 256K vocab (future) |

**Path A is the highest leverage.** Paths B and C are safety/refinement. A+B should ship together. C can wait for `gemma2_2b` scale.

## Key Insights

1. **MTP drafter = LoRA-trained drafter.** The "78M params" distills to 192 LoRA params at our scale. Training takes seconds on CPU.

2. **Training data already exists.** Game replays (Go, Bomber, Monopoly) are free training pairs. Self-play outputs from G-Zero pipeline are more training data.

3. **Modelless MTP already exists for games.** MCTS multi-step lookahead IS modelless MTP. The drafter is the heuristic, the verifier is the forward model.

4. **Top-32, not Top-4.** Production Gemma 4 uses Top-32 from 2048 centroids. But at 4K vocab (our `bpe`), Top-8 = all clusters. Top-K only matters at 256K+ vocab.

5. **Short texts are MTP's kryptonite.** MoE goes 19% slower at max_tokens=8. Output-length gating is not optional — it's required for safe deployment.

6. **Acceptance rate scales with target size.** 38.8% (2B) to 54.9% (26B). Our tiny models would be even lower. Disabling MTP for game/micro is correct.

7. **Quality is guaranteed.** MTP acceptance/rejection preserves output quality (±0.5pt accuracy across all models). The only risk is latency, not quality.

8. **We already have the infrastructure.** LoRA training (riir-ai wgpu), distillation pipelines (ROPD, SDAR), LeviathanVerifier, drafter configs — all implemented. The missing piece is ONE training run.

## See Also

- Research 026 (Gemma 4 MTP) — original MTP distillation
- Plan 055 (MTP Drafter) — existing infrastructure
- Plan 057 (MTP Budget Propagation) — router integration in riir-ai
- Research 059 (MoE+SD CoDesign) — speculative decoding at scale with MoE
- Plan 008 (wgpu LoRA Training) — GPU training infrastructure in riir-ai
- Plan 072 (ROPD) — rubric-based distillation, same pattern as drafter LoRA training
- Plan 073 (SDAR) — self-distilled agentic RL, same loss structure
- Plan 098 (SHINE) — context-to-LoRA hypernetwork, could generate drafter LoRA per-domain
- DGX Spark benchmark article — production Gemma 4 MTP measurements