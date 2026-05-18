# microgpt-rs: Speculative Decoding Engine

## Pipeline Overview

```
Token + Position
       │
       ▼
  DFlash Predict ──→ Marginal Distributions [depth][vocab]
       │
       ▼
  DDTree Build ──→ Candidate Token Tree (best-first search)
       │                    │
       │              ConstraintPruner (optional)
       │              filters invalid branches
       ▼                    ▼
  SpeculativeVerifier
  ├─ SimulatedVerifier: DDTree path + acceptance rate
  └─ LeviathanVerifier: real p/q rejection + residual + bonus token
       │
       ▼
  Accepted Tokens (1 to draft_lookahead+1)
```

---

## DDTree (`speculative/dd_tree.rs`)

### TreeNode

```rust
pub struct TreeNode {
    pub score: f32,         // log-probability
    pub depth: usize,       // position in lookahead
    pub token_idx: usize,   // token ID
    pub parent_path: u128,  // 16 bits per depth, LSB-first
}
```

**Path encoding:** 16-bit slots per depth level → max token ID 65535, max depth 8. The `parent_path` field encodes the entire ancestry in a single `u128`, enabling zero-alloc path reconstruction.

- `extract_parent_tokens(path, n) → Vec<usize>` — decodes the path into a vector of token IDs.
- `extract_parent_tokens_into(path, n, buf)` — zero-alloc variant that writes into a pre-allocated buffer.

### Tree Building

| Function | Description |
|----------|-------------|
| `build_dd_tree(marginals, config) → Vec<TreeNode>` | Standard best-first tree construction from marginal distributions. |
| `build_dd_tree_pruned(marginals, config, pruner, chain_seed) → Vec<TreeNode>` | Best-first with constraint pruning and optional chain-seed backbone. |
| `build_dd_tree_screened(marginals, config, screener, chain_seed) → Vec<TreeNode>` | ScreeningPruner-graded relevance: `blended = parent_score + ln(P) + ln(R)`. Hard-trim at R ≤ 0. |
| `build_dd_tree_balanced(marginals, config, screener) → Vec<TreeNode>` | Balanced tree: ensures equal expansion across depth levels for broader exploration. |
| `extract_best_path(tree) → Vec<usize>` | Extract the highest-scoring token path from the built tree. |
| `extract_best_path_into(tree, buf)` | Zero-alloc variant — writes into pre-allocated buffer. |
| `build_inference_result(tree, config) → InferenceResult` | Build a result struct from the best tree path with reward and domain info. |
| `merge_retrieved_branches(tree, marginals, config, retrieved, rest_weight)` | Inject REST-retrieved token sequences into the tree with blended scores. |

**Chain-seed mode:**
1. **Phase A** — builds a greedy argmax backbone (chain) by always selecting the highest-probability token at each depth.
2. **Phase B** — expands branch nodes from every chain node using the remaining budget, allowing exploration of alternative continuations.

**Budget:** `config.tree_budget` caps the total number of nodes in the tree, ensuring bounded memory and compute.

### TreeBuilder (zero-alloc)

```rust
pub struct TreeBuilder {
    heap: BinaryHeap<TreeNode>,
    tree: Vec<TreeNode>,
    chain_nodes: Vec<TreeNode>,
    chain_parent_tokens: Vec<usize>,
}
```

- Pre-allocated once, cleared via `clear()` (reuses existing capacity across calls).
- `build(&mut self, marginals: &[&[f32]], config, pruner, chain_seed) -> &[TreeNode]` — main entry point, returns a slice into the internal buffer.
- `build_and_merge(...)` — performs tree build + REST merge in a single call.
- `build_screened(&mut self, marginals, config, screener, chain_seed) -> &[TreeNode]` — screening pruner with graded relevance scoring.
- `build_balanced(&mut self, marginals, config, screener) -> &[TreeNode]` — balanced expansion across depth levels.

---

## DFlash (`speculative/dflash.rs`)

Draft-Flash produces marginal distributions over future tokens. It runs the draft model in a single forward pass per depth level, avoiding autoregressive serialization.

### Functions

| Function | Description |
|----------|-------------|
| `dflash_predict(weights, config, token, pos) → Vec<Vec<f32>>` | Independent marginals per depth (same token/pos fed each step). |
| `dflash_predict_ar(weights, config, token, pos, rng)` | Autoregressive: sample a token → feed it back → repeat for each depth. |
| `dflash_predict_parallel(weights, config, token, pos)` | Rayon parallel marginals across depths (skips parallelism if `n_embd ≤ threshold`). |
| `dflash_predict_conditioned(weights, config, token, pos, hidden_state)` | Target-conditioned: seeds the draft KV cache with the target model's hidden state for better alignment. |

### Zero-alloc Variants

All `_with` variants accept a `&mut SpeculativeContext` and write results into pre-allocated buffers:

| Function | Returns |
|----------|---------|
| `dflash_predict_with(sctx, weights, config, token, pos) -> usize` | Count of depth levels produced; marginals stored in `sctx`. |
| `dflash_predict_ar_with(sctx, weights, config, token, pos, rng) -> usize` | Same contract, autoregressive mode. |
| `dflash_predict_conditioned_with(sctx, weights, config, token, pos, hidden_state) -> usize` | Same contract, target-conditioned mode. |

---

## SpeculativeVerifier (`speculative/verifier.rs`)

### Trait

```rust
pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

### SimulatedVerifier

A lightweight verifier that estimates acceptance without running the target model. Useful for benchmarking draft quality independently of target throughput.

**Pipeline:**
1. DFlash produces marginals.
2. DDTree builds the candidate tree.
3. Extract the best path from the tree.
4. Apply a simulated acceptance rate (default 0.75) to cap accepted token count.
5. **Bonus token:** on full acceptance (all draft tokens accepted), sample +1 token from the last marginal — this is "free" since the marginal was already computed.

**Configuration:**
- `acceptance_rate: f32` — probability of accepting each draft token (default 0.75).

### LeviathanVerifier

Implements the full Algorithm 1 from Leviathan et al. 2022 ("Fast Inference from Transformers via Speculative Decoding").

**Internal state:**
- Pre-allocated target `ForwardContext` + `MultiLayerKVCache`
- Pre-allocated draft `SpeculativeContext`

**Pipeline:**
1. DFlash draft → save draft distributions q(x) at each depth.
2. Target model forward pass on the full draft token sequence → get target distributions p(x).
3. For each draft token, accept with probability min(1, p/q).
4. On rejection: sample from the residual distribution max(0, p − q), normalized.
5. On full acceptance of all draft tokens: sample a bonus token from p(x) at position γ (the last draft position).

**Key properties:**
- Guarantees exact target distribution (no approximation drift).
- Real p/q rejection sampling + residual distribution + bonus token.
- Currently slow at 4× model ratio — requires training/distillation for viable throughput in production.

---

## Sampling (`speculative/sampling.rs`)

Low-level sampling primitives used by verifiers and tree builders.

| Function | Description |
|----------|-------------|
| `sample_from_distribution(probs, rng) -> usize` | Categorical sampling. Zero-alloc, operates in-place on the provided slice. |
| `sample_residual_distribution(p, q, rng) -> usize` | Sample from max(0, p − q) normalized. Used by Leviathan verifier on rejection. |
| `sample_residual_distribution_into(p, q, scratch, rng) -> usize` | Zero-alloc variant that writes the residual into a pre-allocated scratch buffer. |

---

## Step Functions (`speculative/step.rs`)

High-level entry points that compose the full speculative decoding pipeline.

| Function | Description |
|----------|-------------|
| `speculative_step(draft_weights, draft_config, token, pos, rng)` | Default: uses `SimulatedVerifier` internally. |
| `speculative_step_verifier(..., verifier)` | Plugs in a custom `SpeculativeVerifier` implementation. |
| `speculative_step_rest(..., rest_client)` | REST-augmented speculative decoding (behind `"rest"` feature). **Note:** REST client lives in `riir-ai/riir-rest`; this repo has only the bridge test + `merge_retrieved_branches()` stub (Plan 009). |
| `speculative_step_rollback(..., verifier)` | KV snapshot/rollback per branch — enables speculative branching without corrupting the main KV cache. |
| `speculative_step_rollback_with(..., sctx, verifier)` | Zero-alloc variant: accepts pre-allocated `SpeculativeContext`. |
| `speculative_step_conditioned(...)` | Target-conditioned draft seeding + Leviathan verification for highest-fidelity output. |
| `speculative_step_conditioned_with(..., sctx)` | Zero-alloc variant: accepts pre-allocated `SpeculativeContext`. |
| `speculative_step_rollback_paged(..., branch_cache)` | Paged KV cache rollback — uses `DDTreeBranchCache` with copy-on-write fork for branch isolation. Avoids full snapshot copies. |

---

## D2F: Discrete Diffusion Forcing (`speculative/d2f.rs`, behind `"dllm"` feature)

A third decode strategy alongside autoregressive and speculative. D2F decodes entire blocks of tokens in parallel via iterative denoising, using block-causal attention: **bidirectional within each block** (intra-block positions attend to each other), **causal across blocks** (inter-block attention is strictly left-to-right). This preserves standard KV cache semantics — previously decoded blocks accumulate KV entries that subsequent blocks reuse without recomputation.

### How It Works

```
Step 0:  [prompt] [MASK] [MASK] [MASK] [MASK]    ← initialize block as masks
Step 1:  [prompt] [tok_0] [MASK] [tok_2] [MASK]  ← forward_block_causal → sample → confidence remask
Step 2:  [prompt] [tok_0] [tok_1] [tok_2] [MASK]  ← repeat until all unmasked or max steps
Step 3:  [prompt] [tok_0] [tok_1] [tok_2] [tok_3] ← FullyActivated → commit KV, start next block
```

Each denoising step:
1. `forward_block_causal_with()` — zero-alloc forward pass writing into `D2fContext` flat buffers
2. Sample from logits at masked positions (temperature-scaled)
3. Confidence remasking: keep tokens with probability ≥ `confidence_threshold`, re-mask others
4. Early exit when all positions are unmasked

### When to Use D2F

| Condition | Recommended Strategy |
|-----------|---------------------|
| Generating 1–3 tokens | Autoregressive (no block benefit) |
| Have draft model, need fast AR | Speculative (DFlash + DDTree) |
| Generating blocks of 8+ tokens | D2F (parallel denoising) |
| Need constraint-guided output | D2F (pruner integrates at each denoising step) |

### API

| Function | Description |
|----------|-------------|
| `d2f_decode_block(weights, config, decode_config, pruner, rng)` | Decode a single block. Returns `D2fBlockResult` with tokens, steps used, confidence history. |
| `d2f_decode_block_with_prompt(weights, config, decode_config, prompt, pruner, rng)` | Decode with prompt context. Prompt positions are never masked. |
| `d2f_decode_block_with(dctx, weights, config, decode_config, pruner, rng)` | Zero-alloc variant — reuses pre-allocated `D2fContext`. Preferred for hot loops. |
| `d2f_decode_block_with_target(weights, config, decode_config, target, pruner, rng)` | With ground truth for accuracy measurement (testing/benchmarking). |

### D2fPipeline — Multi-Block Decode

```rust
let pipeline = D2fPipeline::with_prompt(&config, decode_config, total_len, &prompt);
let result = pipeline.decode_all(&weights, &NoPruner, &mut rng);
// result.tokens — all decoded tokens
// result.block_results — per-block breakdown
// result.n_fully_activated — blocks that fully denoised
```

Pipeline decodes blocks sequentially. Each block uses block-causal attention, so previously decoded blocks provide causal context. After each block completes, `D2fContext::commit(len)` preserves KV entries — subsequent blocks skip recomputation for committed positions.

### D2fDecodeConfig

| Field | Default | Purpose |
|-------|---------|---------|
| `denoise_steps` | 8 | Max denoising iterations per block |
| `confidence_threshold` | 0.7 | Keep token if probability ≥ this, re-mask otherwise |
| `activation_threshold` | 0.5 | Block confidence to count as "activated" |
| `addition_threshold` | 0.3 | Block confidence to allow starting successor block |
| `block_size` | 8 | Tokens per block (should match training) |
| `max_pipeline_depth` | 4 | Max simultaneous in-flight blocks |
| `temperature` | 1.0 | Sampling temperature during denoising |

Presets: `D2fDecodeConfig::quality()` (16 steps, 0.9 confidence), `D2fDecodeConfig::speed()` (4 steps, 0.5 confidence).

### DecodeStrategy — Automatic Selection

```rust
pub enum DecodeStrategy {
    Autoregressive,         // 1 token/step
    Speculative,            // DFlash + DDTree + verifier
    DiscreteDiffusion,      // D2F block-parallel (behind "dllm" feature)
}
```

`DecodeStrategy::recommend(block_size, n_tokens, has_draft_model)` auto-selects:
1. `dllm` feature enabled **and** `n_tokens >= block_size` → `DiscreteDiffusion`
2. `has_draft_model` → `Speculative`
3. Otherwise → `Autoregressive`

Set via `InferenceOverrides::decode_strategy` for explicit override.

📖 See [`.research/34_D2F_Discrete_Diffusion_Forcing.md`](../.research/34_D2F_Discrete_Diffusion_Forcing.md) for experimental results and research context.

---

## REST Bridge (behind `"rest"` feature)

Augments speculative decoding with historically successful token sequences retrieved from an external vector store.

> **Note:** The full REST client (`RestClient`, `RetrievalResult`) lives in `riir-ai/riir-rest`. This repo contains only `merge_retrieved_branches()` (DDTree merge logic) and a bridge test. The `"rest"` feature flag enables the test; the merge function is always compiled.

### REST Architecture

```
Target Model Forward Pass
    │
    ├─ ctx.x (hidden state) ──► copy to embedding buffer ──► POST /search/vector
    │                                                          │
    │                                                          ▼
    │                                                    anyrag returns
    │                                                    historical continuations
    │                                                    (token sequences from
    │                                                     past successful compilations)
    │                                                          │
    ▼                                                          ▼
DDTree Build ◄──────── merge retrieved sequences as candidate branches
    │
    ▼
ConstraintPruner validates
    │
    ▼
Target Verification
```

### RestClient

```rust
pub struct RestClient {
    base_url: String,
    client: reqwest::Client,
}
```

- `retrieve(embedding: &[f32], top_k: usize) -> RetrievalResult` — queries an anyrag `/search/vector` endpoint with the current hidden state embedding. Returns historical token continuations ranked by cosine similarity.

### RetrievalResult

```rust
pub struct RetrievalResult {
    pub token_sequences: Vec<Vec<usize>>,
    pub scores: Vec<f32>,
}
```

### DDTree Merge

`merge_retrieved_branches(tree, marginals, config, retrieved, rest_weight)` blends REST results into the draft tree:

- **Score blending:** `(1 − w) × log(draft_prob) + w × log(retrieval_score)` where `w = rest_weight`.
- Merged branches coexist with DDTree candidates; both respect `tree_budget`.
- Retrieval branches that duplicate draft branches are merged (scores summed in log-space).

---

## SpeculativeContext (`speculative/types.rs`)

Pre-allocated buffer struct for zero-alloc speculative decoding across all pipeline stages.

```rust
pub struct SpeculativeContext {
    pub ctx: ForwardContext,
    pub cache: MultiLayerKVCache,
    pub marginals_flat: Vec<f32>,       // [draft_lookahead × vocab_size]
    pub probs_buf: Vec<f32>,            // [vocab_size] temp for softmax
    pub sampled_tokens: Vec<usize>,     // [draft_lookahead]
    pub accepted_buf: Vec<usize>,       // [draft_lookahead + 1]
    pub path_buf: Vec<usize>,           // [draft_lookahead + 1]
    pub residual_buf: Vec<f32>,         // [vocab_size]
    pub p_distributions_flat: Vec<f32>, // [(draft_lookahead + 1) × vocab_size] for Leviathan
    pub parent_tokens_buf: Vec<usize>,  // [draft_lookahead + 1] for pruner
}
```

- `new(config)` — allocates all buffers based on `draft_lookahead` and `vocab_size`.
- `reset()` — clears lengths to zero; capacity is reused.
- Used by all `_with` variants and both verifier implementations.
- `DDTreeBranchCache` (separate struct) — paged KV cache for `speculative_step_rollback_paged`, avoids full snapshot copies via copy-on-write fork.

---

## ConstraintPruner (`speculative/types.rs`)

Trait for filtering invalid token branches during tree construction.

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}
```

### Implementations

| Pruner | Feature | Description |
|--------|---------|-------------|
| `NoPruner` | (always available) | Always returns `true`. No-op pass-through. |
| `NoScreeningPruner` | (always available) | Returns `relevance = 1.0` for all inputs. No-op screening. |
| `BinaryScreeningPruner<P>` | (always available) | Adapts a `ConstraintPruner` to `ScreeningPruner` — R ∈ {0.0, 1.0}. |
| `SudokuPruner` | `"sudoku"` | Row/column/box validation with path-aware cross-depth checks. Ensures generated token sequences satisfy Sudoku constraints. |
| `SynPruner` | `"validator"` | Bracket balance + syntax parse validation. Ensures generated code/structured output remains syntactically valid. |
| `FlowPruner<P>` | `"bandit"` | GFlowNet-inspired stop-probability regularization: `relevance = inner × (1 + λ × (1 - stop_prob))`. Wraps inner `ScreeningPruner`. |

Pruners are called during tree building for every candidate node before it enters the priority heap. Invalid branches are discarded immediately, saving budget for valid explorations.

## Event & Diagnostic Types (`speculative/types.rs`)

```rust
pub enum RejectionReason {
    LowProbability,
    ConstraintViolation,
    LowRelevance,
    DivergedFromTarget,
}

pub enum DraftEvent {
    Drafting { depth: usize, token_idx: usize },
    Pruned { depth: usize, token_idx: usize, reason: RejectionReason },
    Verified { depth: usize, token_idx: usize },
    BranchRejected { path: Vec<usize>, reason: RejectionReason },
    StepComplete { accepted: usize, total: usize },
}

pub enum PrefillMode {
    Off,
    Auto,
    Always,
}

pub struct FlashPrefillConfig {
    pub block_size: usize,
    pub attention_sink: usize,
    pub window: usize,
    pub last_n_full: usize,
    pub alpha: f32,
}

pub struct BlockScores {
    pub scores: Vec<f32>,
    pub block_size: usize,
}
```

- `RejectionReason` — why a branch was pruned (diagnostics + PPoT rescue targeting).
- `DraftEvent` — trace events for debugging pipeline behavior.
- `PrefillMode` — controls when block-sparse prefill activates.
- `FlashPrefillConfig` — PFlash block selection rules: sink, window, last_n_full, alpha threshold.
- `BlockScores` — per-block importance scores from `BlockAttentionScorer`.

---

## Speculative Prefill (`speculative/prefill.rs`)

PFlash-inspired prompt compression for reducing Time-To-First-Token (TTFT) on long prompts.

### PrefillScorer Trait

```rust
pub trait PrefillScorer: Send + Sync {
    fn score(&self, tokens: &[usize], weights, config) -> Vec<f32>;
}
```

| Scorer | Description |
|--------|-------------|
| `AttentionScorer` | Uses Q·K dot-product attention magnitudes to estimate token importance. Tokens with high attention scores are retained. |
| `BlockAttentionScorer` | Block-level attention scoring for PFlash — scores entire blocks instead of individual tokens. |
| `RandomScorer` | Random importance scores (baseline/ablation). |
| `UniformScorer` | Uniform scores (baseline/ablation). |

### Compression

| Function | Description |
|----------|-------------|
| `compress_prompt(tokens, scores, keep_ratio) -> Vec<usize>` | Retains first/last tokens + top-scoring spans proportional to `keep_ratio`. |
| `compress_prompt_blocks(scores, cfg, prefix, suffix) -> Vec<usize>` | Block-level compression — selects important blocks via `FlashPrefillConfig` rules. |
| `block_select(block_scores, cfg) -> Vec<usize>` | Selects important blocks: sink + window + last_n_full + alpha threshold. |
| `block_select_grid(grid, m, n, h, cfg) -> Vec<usize>` | 2D grid block selection for multi-head/multi-layer scoring. |
| `should_compress(prompt_len, cfg) -> bool` | Quick check: skip compression overhead for short prompts. |

### Full Prefill Pipeline

| Function | Description |
|----------|-------------|
| `speculative_prefill(draft_weights, draft_config, target_weights, target_config, prompt, scorer, keep_ratio)` | Token-level scoring + compression + target prefill. ~10× TTFT reduction. |
| `speculative_prefill_block(...)` | Block-level PFlash prefill — uses `FlashPrefillConfig` for structured block selection. |
| `speculative_prefill_adaptive(...)` | Adaptive prefill — automatically selects token vs block compression based on prompt length. |

Pipeline steps:
1. Score prompt tokens using the draft model (fast).
2. Compress the prompt based on scores and `keep_ratio` / `FlashPrefillConfig`.
3. Run target prefill on the compressed prompt (slow, but ~10× shorter).
4. Result: ~10× TTFT reduction with minimal quality loss for long prompts.

---

## Feature Flags Summary

| Flag | Enables |
|------|---------|
| `sparse_mlp` (default) | TwELL-inspired sparse MLP matmul |
| `domain_latent` (default) | Free Transformer mid-layer domain conditioning |
| `ppot` (default) | PPoT logit-parameterized CPU resampling + adaptive rescue |
| `bandit` (default) | Multi-armed bandit + FlowPruner + AbsorbCompress + HotSwapPruner |
| `"sudoku"` | `SudokuPruner` — constrained decoding for Sudoku |
| `"validator"` | `SynPruner` — syntax-aware constrained decoding |
| `"rest"` | REST bridge test + `merge_retrieved_branches` (client in `riir-ai/riir-rest`) |
| `"feedback"` | E2E feedback loop — sends `InferenceResult` to REST endpoint |
| `"dllm"` | D2F Discrete Diffusion Forcing — `D2fContext`, `D2fPipeline`, block-parallel decode (Plan 066) |

---

## Key Design Principles

1. **Zero-allocation hot path:** All `_with` variants, `TreeBuilder`, and `D2fContext` reuse pre-allocated buffers. No `Vec::push` in the inner loop — only `clear()` + index writes. D2F uses flat 2D buffers (`[pos * dim..(pos+1) * dim]`) instead of `Vec<Vec<f32>>`.
2. **Separable pipeline stages:** DFlash → DDTree → Verifier are independent. D2F is an alternative pipeline: `forward_block_causal` → confidence remasking → constraint pruning. Each can be swapped, benchmarked, or disabled independently.
3. **Config-driven behavior:** `SpeculativeConfig` controls lookahead depth, tree budget, acceptance rates, and parallelism thresholds. `D2fDecodeConfig` controls denoising steps, confidence thresholds, and block size. `DecodeStrategy` auto-selects the optimal pipeline. No runtime branching on magic numbers.
4. **Feature-gated complexity:** REST bridge, constraint pruners, and D2F are behind feature flags. The default build stays lean.