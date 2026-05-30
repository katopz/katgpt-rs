# katgpt-rs: Speculative Decoding Engine

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
       │              ScreeningPruner (optional, graded relevance)
       │              FlowPruner (optional, GFlowNet bonus)
       │              PeiraPruner (optional, alignment-modulated)
       │              AlphaTarget (optional, lattice deduction)
       ▼                    ▼
  SpeculativeVerifier
  ├─ SimulatedVerifier: DDTree path + acceptance rate
  ├─ LeviathanVerifier: real p/q rejection + residual + bonus token (+ optional LoRA drafter)
  ├─ D2fDrafterVerifier: D2F block draft + AR verify (behind "tri_mode")
  └─ ParallelProbeVerifier: multi-branch consensus + answer extraction (behind "parallel_probe")
       │
       ▼
  Accepted Tokens (1 to draft_lookahead+1)
```

---

## Module Structure (`speculative/`)

| Module | Feature Gate | Description |
|--------|-------------|-------------|
| `dd_tree` | (always) | DDTree construction, path extraction, SDE noise |
| `dflash` | (always) | Draft-Flash marginal distribution generation |
| `types` | (always) | Core types: `SpeculativeContext`, `TreeNode`, pruners, events |
| `sampling` | (always) | Categorical + residual distribution sampling |
| `step` | (always) | High-level step entry points |
| `verifier` | (always) | `SpeculativeVerifier` trait + `SimulatedVerifier` + `LeviathanVerifier` |
| `prefill` | (always) | PFlash prompt compression for TTFT reduction |
| `drafter_lora` | (always) | LoRA-trained drafter weights for LeviathanVerifier |
| `ppot` | `"ppot"` | Probabilistic Programs of Thought — CPU resampling + adaptive rescue |
| `flow_pruner` | `"bandit"` | GFlowNet-inspired stop-probability regularization |
| `peira_pruner` | `"peira_distill"` | PEIRA alignment-modulated ScreeningPruner |
| `d2f` | `"dllm"` | Discrete Diffusion Forcing — block-parallel decode |
| `d2f_verifier` | `"tri_mode"` | D2F drafter + AR verifier (self-speculation) |
| `diffusion_sampler` | `"tri_mode"` | Learned accept/reject for D2F denoising steps |
| `alpha` | `"lattice_deduction"` | LDT α-operator for progressive multi-solution supervision |
| `answer_extract` | `"parallel_probe"` | Answer extraction from token streams (regex, think tags, discrete actions) |
| `parallel_probe` | `"parallel_probe"` | Multi-branch 2D controller with consensus + pruning |

---

## DDTree (`speculative/dd_tree.rs`)

### TreeNode

```rust
pub struct TreeNode {
    pub parent_path: u128,  // 16 bits per depth, LSB-first
    pub depth: usize,       // position in lookahead
    pub token_idx: usize,   // token ID
    pub score: f32,         // log-probability
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
| `build_dd_tree_sde(marginals, config, pruner, sde_config, rng) → Vec<TreeNode>` | SDE noise injection during tree expansion (ELF Plan 079). |
| `build_dd_tree_balanced_sde(marginals, config, screener, sde_config, rng) → Vec<TreeNode>` | Balanced + SDE noise combined. |
| `extract_best_path(tree) → Vec<usize>` | Extract the highest-scoring token path from the built tree. |
| `extract_best_path_into(tree, buf)` | Zero-alloc variant — writes into pre-allocated buffer. |
| `extract_candidate_sequences(tree) → Vec<Vec<usize>>` | Extract all candidate sequences from the tree. |
| `extract_all_sequences(tree) → Vec<Vec<usize>>` | Extract all sequences (leaves to root). |
| `find_valid_sequence(tree, pruner, parent_tokens) → Option<Vec<usize>>` | Find first valid sequence respecting a constraint pruner. |
| `par_find_valid_sequence(tree, pruner, parent_tokens) → Option<Vec<usize>>` | Parallel valid sequence search. |
| `par_find_shortest_sequence(tree, pruner, parent_tokens) → Option<Vec<usize>>` | Parallel shortest valid sequence search. |
| `build_inference_result(tree, config) → InferenceResult` | Build a result struct from the best tree path with reward and domain info. |
| `merge_retrieved_branches(tree, marginals, config, retrieved, rest_weight)` | Inject REST-retrieved token sequences into the tree with blended scores. |
| `inject_sde_noise(marginals, config, sde_config, rng)` | SDE noise injection into marginals (ELF Plan 079). |
| `best_of_k_rollouts(marginals, config, pruner, sde_config, rng, width_config) → Vec<TreeNode>` | Multiple stochastic rollouts with width scaling (ELF Plan 079). |

**Chain-seed mode:**
1. **Phase A** — builds a greedy argmax backbone (chain) by always selecting the highest-probability token at each depth.
2. **Phase B** — expands branch nodes from every chain node using the remaining budget, allowing exploration of alternative continuations.

**Budget:** `config.tree_budget` caps the total number of nodes in the tree, ensuring bounded memory and compute.

### SDE Noise Injection (ELF Plan 079)

| Type | Description |
|------|-------------|
| `SdeConfig` | Config for noise: `gamma` (noise scale), `confidence_floor`, `preserve_top1`. |
| `WidthScaleConfig` | K-rollouts + `WidthSelectionMode` (`BestQ`, `MostFrequent`, `Top1Converged`). |
| `ResidualTracker` | Tracks residual convergence (behind `"eqr_convergence"` feature). |
| `entropy_truncate_horizon(marginals) → usize` | Compute truncated horizon from entropy profile (behind `"sr2am_configurator"` feature). |

### TreeBuilder (zero-alloc)

```rust
pub struct TreeBuilder {
    heap: BinaryHeap<TreeNode>,
    tree: Vec<TreeNode>,
    chain_nodes: Vec<TreeNode>,
    chain_parent_tokens: Vec<usize>,
    parent_tokens_buf: Vec<usize>,
}
```

- Pre-allocated once, cleared via `clear()` (reuses existing capacity across calls).
- `build(&mut self, marginals: &[&[f32]], config, pruner, chain_seed) -> &[TreeNode]` — main entry point, returns a slice into the internal buffer.
- `build_and_merge(...)` — performs tree build + REST merge in a single call.
- `build_screened(&mut self, marginals, config, screener, chain_seed) -> &[TreeNode]` — screening pruner with graded relevance scoring.
- `build_and_merge_screened(...)` — screened build + REST merge.
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
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize>;
}
```

### SimulatedVerifier

A lightweight verifier that estimates acceptance without running the target model. Useful for benchmarking draft quality independently of target throughput.

**Internal state:**
- Pre-allocated `SpeculativeContext` + `TreeBuilder`

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
- Pre-allocated draft `SpeculativeContext` + `TreeBuilder`
- Optional `DrafterLoraWeights` + `DrafterForwardContext` (Plan 117: MTP LoRA Drafter)

**Pipeline:**
1. DFlash draft → save draft distributions q(x) at each depth.
2. Target model forward pass on the full draft token sequence → get target distributions p(x).
3. For each draft token, accept with probability min(1, p/q).
4. On rejection: sample from the residual distribution max(0, p − q), normalized.
5. On full acceptance of all draft tokens: sample a bonus token from p(x) at position γ (the last draft position).

**Key properties:**
- Guarantees exact target distribution (no approximation drift).
- Real p/q rejection sampling + residual distribution + bonus token.
- Supports LoRA-trained drafter via `with_drafter_lora()` / `set_drafter_lora()` for improved draft quality.

**LoRA Drafter methods:**
- `with_drafter_lora(lora, draft_config) → Self` — builder-style LoRA attachment.
- `set_drafter_lora(&mut self, lora, draft_config)` — set LoRA on existing verifier.
- `has_drafter_lora(&self) → bool` — check if LoRA is attached.

### D2fDrafterVerifier (`speculative/d2f_verifier.rs`, behind `"tri_mode"` feature)

Self-speculation mode — uses the same model for both drafting and verification (Plan 089: Tri-Mode Inference). D2F block decode acts as the drafter (parallel, bidirectional within block), standard AR forward pass acts as the verifier.

**Internal state:**
- Pre-allocated target `ForwardContext` + `MultiLayerKVCache`
- Pre-allocated `D2fContext` for block decode
- `probs_buf` for target probability distribution

**Pipeline:**
1. **Phase 0** — Score the initial token through the target AR model → p_dist[0].
2. **Phase 1** — D2F block decode in parallel → draft tokens (up to `draft_width`).
3. **Phase 2** — Score each draft token through the target AR model → p_dist[i+1].
4. **Phase 3** — Argmax prefix matching: accept draft[i] if it matches argmax(p_dist[i+1]); on first mismatch, take target's preferred token.
5. **Phase 4** — Bonus token: if all draft tokens accepted, sample +1 from p_dist at last position.

**Key difference from LeviathanVerifier:**
- Draft: `d2f_decode_block()` (parallel, bidirectional within block)
- Verify: `forward()` with causal attention (same as Leviathan)
- KV caches are separate (block-causal for draft, causal for verify)

**Configuration (`SelfSpecConfig`):**

| Field | Default | Purpose |
|-------|---------|---------|
| `draft_width` | 8 | Number of tokens per D2F draft block |
| `d2f_config` | `D2fDecodeConfig::default()` | D2F decode parameters |
| `sampler` | `None` | Optional `DiffusionSampler` for adaptive confidence (Plan 116) |

### ParallelProbeVerifier (`speculative/parallel_probe.rs`, behind `"parallel_probe"` feature)

Multi-branch speculative decoding with answer extraction and consensus-based early stopping. Runs multiple branches in parallel, extracts answers, and stops when branches agree.

**Internal state:**
- Inner `SpeculativeVerifier` (e.g., `LeviathanVerifier`)
- `ParallelProbeController` for consensus tracking and pruning
- `AnswerExtractor` for extracting answers from token streams
- Per-branch text buffers and token counters

**Key types:**
- `ParallelProbeConfig` — probe interval, stability patience, prune patience, warmup steps, min active branches, prune vote ratio.
- `ProbeDecision` — `Continue`, `Stop { answer }`, `Prune { branch_ids }`, `StopAndPrune { answer, branch_ids }`.
- `BranchProbeState` — per-branch state: branch_id, last answer, disagree streak, is_pruned, is_finished.
- `ProbingMatrix<A>` — dense answer matrix for majority voting across branches and probe steps.

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
| `speculative_step_rollback(..., verifier)` | KV snapshot/rollback per branch — enables speculative branching without corrupting the main KV cache. |
| `speculative_step_rollback_with(..., sctx, verifier)` | Zero-alloc variant: accepts pre-allocated `SpeculativeContext`. |
| `speculative_step_conditioned(...)` | Target-conditioned draft seeding + Leviathan verification for highest-fidelity output. |
| `speculative_step_conditioned_with(..., sctx)` | Zero-alloc variant: accepts pre-allocated `SpeculativeContext`. |
| `speculative_step_rollback_paged(..., branch_cache)` | Paged KV cache rollback — uses `DDTreeBranchCache` with copy-on-write fork for branch isolation. |
| `speculative_step_with_configurator(...)` | SR2AM configurator integration (behind `"sr2am_configurator"` feature). |

**Deprecated (still re-exported):** `speculative_step_conditioned`, `speculative_step_conditioned_with`, `speculative_step_rollback`, `speculative_step_rollback_with`.

---

## D2F: Discrete Diffusion Forcing (`speculative/d2f.rs`, behind `"dllm"` feature)

A third decode strategy alongside autoregressive and speculative. D2F decodes entire blocks of tokens in parallel via iterative denoising, using block-causal attention: **bidirectional within each block** (intra-block positions attend to each other), **causal across blocks** (inter-block attention is strictly left-to-right). This preserves standard KV cache semantics — previously decoded blocks accumulate KV entries that subsequent blocks reuse without recomputation.

### How It Works

```
Step 0:  [prompt]  ←  ←  ←     ← initialize block as masks
Step 1:  [prompt] [tok_0]  ← [tok_2]  ← forward_block_causal → sample → confidence remask
Step 2:  [prompt] [tok_0] [tok_1] [tok_2]  ← repeat until all unmasked or max steps
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
| `d2f_decode_block_with_prompt_with(dctx, weights, config, decode_config, prompt, pruner, rng)` | Zero-alloc + prompt context. |
| `d2f_decode_block_with_target(weights, config, decode_config, target, pruner, rng)` | With ground truth for accuracy measurement (testing/benchmarking). |
| `d2f_decode_block_with_target_with(dctx, weights, config, decode_config, target, pruner, rng)` | Zero-alloc + target. |
| `d2f_decode_block_with_sampler(dctx, weights, config, decode_config, pruner, rng, sampler)` | Adaptive confidence via `DiffusionSampler` (behind `"tri_mode"`). |
| `d2f_decode_block_with_prompt_with_sampler(dctx, weights, config, decode_config, prompt, pruner, rng, sampler)` | Adaptive confidence + prompt (behind `"tri_mode"`). |

### D2fPipeline — Multi-Block Decode

```rust
let pipeline = D2fPipeline::with_prompt(&config, decode_config, total_len, &prompt);
let result = pipeline.decode_all(&weights, &NoPruner, &mut rng);
// result.tokens — all decoded tokens
// result.block_results — per-block breakdown
// result.n_fully_activated — blocks that fully denoised
// result.n_semi_activated — blocks partially denoised
```

Pipeline decodes blocks sequentially. Each block uses block-causal attention, so previously decoded blocks provide causal context. After each block completes, `D2fContext::commit(len)` preserves KV entries — subsequent blocks skip recomputation for committed positions.

Supports optional `SoftDecodeConfig` via `with_soft_config()` for hybrid embedding D2F (Plan 109, `"dmax_spd"` feature).

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
| `schedule` | `ScheduleKind::Uniform` | Noise schedule type (`Uniform` or `LogitNormal { mean, std }`) |
| `multistep` | `false` | DPM-Solver++(2M) multistep logit extrapolation |

Presets: `D2fDecodeConfig::quality()` (16 steps, 0.9 confidence), `D2fDecodeConfig::speed()` (4 steps, 0.5 confidence), `D2fDecodeConfig::multistep_quality()`.

### D2fBlockState

```rust
pub enum D2fBlockState {
    SemiActivated { step: usize, confidence: f32 },
    FullyActivated,
}
```

### D2fBlockResult

| Field | Description |
|-------|-------------|
| `tokens` | Decoded token IDs |
| `steps_used` | Number of denoising steps consumed |
| `confidence_history` | Per-step confidence values |
| `accuracy` | Accuracy vs target (when target provided) |
| `state` | `D2fBlockState` at completion |

### Soft Decode (DMax Soft Parallel Decode, behind `"dmax_spd"` feature)

| Type | Description |
|------|-------------|
| `SoftDecodeConfig` | Config for hybrid embedding decode: `use_hybrid_embeddings`, `decode_threshold`, `accept_threshold`, `contiguous_prefix`, `consistency_check`. |
| `HybridEmbedding` | Hybrid token embedding with confidence: `{ confidence: f32, token_id: usize }`. |
| `BlockConvergence` | Enum: `NotConverged`, `ConfidenceConverged`, `ConsistencyConverged`. |
| `check_block_convergence(...)` | Check if a block has converged. |
| `contiguous_prefix_promote(...)` | Promote contiguous prefix of accepted tokens. |
| `d2f_decode_block_soft(...)` | Soft decode with hybrid embeddings. |

### ScheduleKind

```rust
pub enum ScheduleKind {
    Uniform,
    LogitNormal { mean: f32, std: f32 },
}
```

### DecodeStrategy — Automatic Selection

```rust
pub enum DecodeStrategy {
    Autoregressive,                 // 1 token/step (default)
    Speculative,                    // DFlash + DDTree + verifier
    DiscreteDiffusion,              // D2F block-parallel (behind "dllm")
    DiscreteDiffusionSoft,          // DMax Soft Parallel Decode (behind "dmax_spd")
    SelfSpeculation,                // D2F drafts + AR verifies (behind "tri_mode")
}
```

`DecodeStrategy::recommend(block_size, n_tokens, has_draft_model)` auto-selects:
1. `tri_mode` enabled **and** `has_draft_model` **and** `n_tokens >= block_size` → `SelfSpeculation`
2. `dmax_spd` enabled **and** `n_tokens >= block_size` → `DiscreteDiffusionSoft`
3. `dllm` enabled **and** `n_tokens >= block_size` → `DiscreteDiffusion`
4. `has_draft_model` → `Speculative`
5. Otherwise → `Autoregressive`

Set via `InferenceOverrides::decode_strategy` for explicit override.

📖 See [`.research/034_D2F_Discrete_Diffusion_Forcing.md`](../.research/034_D2F_Discrete_Diffusion_Forcing.md) for experimental results and research context.

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
    pub steps_populated: usize,         // number of steps populated in last operation
    pub sde_config: SdeConfig,          // SDE noise injection config (ELF Plan 079)
}
```

- `new(config)` — allocates all buffers based on `draft_lookahead` and `vocab_size`.
- `reset()` — clears lengths to zero; capacity is reused.
- `marginal_slice(step) -> &[f32]` — view into a single step's marginal.
- `marginal_slice_mut(step) -> &mut [f32]` — mutable view.
- `marginals_view() -> Vec<&[f32]>` — slice-per-step view of all marginals.
- `marginals_into(buf) -> Vec<Vec<f32>>` — copy marginals into owned vectors.
- `p_dist_slice(step) -> &[f32]` — view into p-distributions (Leviathan).
- `p_dist_slice_mut(step) -> &mut [f32]` — mutable p-distribution view.
- Used by all `_with` variants and both verifier implementations.
- `DDTreeBranchCache` (separate struct) — paged KV cache for `speculative_step_rollback_paged`, avoids full snapshot copies via copy-on-write fork.

### DDTreeBranchCache

```rust
pub struct DDTreeBranchCache {
    paged: PagedKVCache,
    branch_count: usize,
    max_branches: usize,
}
```

Methods: `new(config, max_branches)`, `fork_branch(...)`, `forward_branch(...)`, `rollback_branch(...)`, `discard_branch(...)`, `reset()`.

### SelfSpecConfig (`speculative/types.rs`, behind `"tri_mode"` feature)

Configuration for D2F self-speculation mode (used by `D2fDrafterVerifier`).

```rust
pub struct SelfSpecConfig {
    pub draft_width: usize,        // tokens per D2F draft block (default: 8)
    pub d2f_config: D2fDecodeConfig,
    pub sampler: Option<DiffusionSampler>,  // adaptive confidence (Plan 116)
}
```

Types re-export base primitives from `crate::types` (`Config`, `Rng`, `softmax_scaled`, etc.) and `katgpt_core::traits` (`ConstraintPruner`, `ScreeningPruner`, etc.) — no direct `katgpt-core` dependency in the speculative module beyond trait re-exports.

### SdeConfig

```rust
pub struct SdeConfig {
    pub gamma: f32,              // noise scale (default: 0.1)
    pub confidence_floor: f32,   // minimum probability to keep (default: 0.01)
    pub preserve_top1: bool,     // keep top-1 token unchanged (default: true)
}
```

---

## ConstraintPruner (`speculative/types.rs`)

Trait for filtering invalid token branches during tree construction.

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}
```

Re-exported from `katgpt_core::traits` (Plan 107 Phase 0).

### Implementations

| Pruner | Feature | Description |
|--------|---------|-------------|
| `NoPruner` | (always available) | Always returns `true`. No-op pass-through. |
| `NoScreeningPruner` | (always available) | Returns `relevance = 1.0` for all inputs. No-op screening. |
| `BinaryScreeningPruner<P>` | (always available) | Adapts a `ConstraintPruner` to `ScreeningPruner` — R ∈ {0.0, 1.0}. |
| `SudokuPruner` | `"sudoku"` | Row/column/box validation with path-aware cross-depth checks. |
| `FlowPruner<P>` | `"bandit"` | GFlowNet-inspired stop-probability regularization. Wraps inner `ScreeningPruner`. |
| `PeiraPruner<P>` | `"peira_distill"` | PEIRA alignment-modulated relevance. Wraps inner `ScreeningPruner`. |

Pruners are called during tree building for every candidate node before it enters the priority heap. Invalid branches are discarded immediately, saving budget for valid explorations.

### ScreeningPruner Trait

```rust
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}
```

Re-exported from `katgpt_core::traits`.

### EarlyStopGate

```rust
pub struct EarlyStopGate<P: ScreeningPruner> {
    pub inner: P,
    pub confidence_threshold: f32,
    pub enabled: bool,
}
```

Depth-aware early stopping gate (PTRM Plan 083). Wraps any `ScreeningPruner` and prunes branches where relevance falls below `confidence_threshold` at depth > 0. At depth 0, always passthrough. Behind `"elf_sde"` feature.

### FlowPruner (`speculative/flow_pruner.rs`, behind `"bandit"` feature)

GFlowNet-inspired stop-probability regularization. Wraps any `ScreeningPruner` and adds a multiplicative flow bonus:

```text
relevance_flow(depth, token, path) = inner_relevance × (1 + λ × (1 - stop_prob(depth)))
```

Key methods:
- `new(inner, lambda, stop_probs)` — create with explicit stop probabilities.
- `with_inner(inner)` — create with default lambda (0.3) and empty stop probs.
- `set_stop_probs_from_marginals(marginals, eos_token_idx)` — extract EOS probabilities.
- `set_stop_probs_from_entropy(marginals)` — use entropy as proxy.
- `flow_bonus(depth) -> f32` — get the flow bonus at a given depth.

### PeiraPruner (`speculative/peira_pruner.rs`, behind `"peira_distill"` feature)

PEIRA alignment-modulated ScreeningPruner. Wraps any `ScreeningPruner` and modulates its relevance signal using PEIRA's spectral alignment score:

```text
relevance_peira(d, t, path) = inner_relevance × alignment^α
```

Key methods:
- `new(inner)` — starts with `alignment = 0.0` (fully attenuated).
- `with_alpha(alpha)` — set modulation exponent (default: 0.5).
- `set_alignment(alignment)` — update cached alignment score.
- `modulation_factor() -> f32` — compute `alignment^alpha`.

---

## Event & Diagnostic Types (`speculative/types.rs`)

```rust
pub enum RejectionReason {
    LowProbability,
    ConstraintViolation,
    LowRelevance { score: f32 },
    DivergedFromTarget,
}

pub enum DraftEvent {
    Drafting { pos: usize, candidates: usize },
    Pruned { pos: usize, kept: usize, rejected: usize },
    Verified { pos: usize, accepted: usize, bonus: bool },
    BranchRejected { pos: usize, reason: RejectionReason },
    StepComplete { tokens_accepted: usize, latency_us: u64 },
}

pub enum PrefillMode {
    Off,
    Auto,
    Always,
}

pub enum ScoreReduction {
    SoftmaxSum,                          // default
    MaxSim,                              // behind "maxsim" feature
}

pub struct FlashPrefillConfig {
    pub block_size: usize,
    pub attention_sink: usize,
    pub window: usize,
    pub last_n_full: usize,
    pub tail_window: usize,
    pub alpha: f32,
    pub score_reduction: ScoreReduction,
}

pub struct BlockScores {
    pub num_blocks: usize,
    pub block_size: usize,
    pub scores: Vec<f32>,
    pub selected: Vec<usize>,
}
```

- `RejectionReason` — why a branch was pruned (diagnostics + PPoT rescue targeting). `LowRelevance` carries the relevance score.
- `DraftEvent` — trace events for debugging pipeline behavior. `Drafting` reports position + candidate count. `Pruned` reports kept/rejected counts. `Verified` reports accepted count + bonus flag. `StepComplete` reports accepted tokens + wall-clock latency.
- `PrefillMode` — controls when block-sparse prefill activates.
- `ScoreReduction` — `SoftmaxSum` (default) or `MaxSim` (behind `"maxsim"` feature) for block pair scoring.
- `FlashPrefillConfig` — PFlash block selection rules: sink, window, last_n_full, tail_window, alpha threshold, score reduction mode.
- `BlockScores` — per-block importance scores from `BlockAttentionScorer`, includes `num_blocks` and `selected` indices.

Presets for `FlashPrefillConfig`: `metal()`, `long_context()`, `short_context()`.

### Diagnostic Snapshots (feature-gated)

| Type | Feature | Fields |
|------|---------|--------|
| `DraftResult` | (always) | `marginals`, `sampled_tokens` |
| `DraftResult.routing_overlap` | `"domain_latent"` | `Option<RoutingOverlapSnapshot>` |
| `DraftResult.cost_snapshot` | `"spec_cost_model"` | `Option<SpecCostSnapshot>` |
| `DraftResult.stability` | `"stability_metrics"` | `Option<StabilitySnapshot>` |
| `StabilitySnapshot` | `"stability_metrics"` | `phase_latencies_ns`, `p50_ns`, `p99_ns`, `mean_ns`, `cv`, `stability_score`, `total_steps` |
| `RoutingOverlapSnapshot` | `"domain_latent"` | `step_overlap`, `unique_slots`, `top_k`, `n_tokens` |
| `SpecCostSnapshot` | `"spec_cost_model"` | `f_sparse`, `f_fixed`, `unique_ratio`, `predicted_ratio`, `actual_ratio`, `k` |

---

## Speculative Prefill (`speculative/prefill.rs`)

PFlash-inspired prompt compression for reducing Time-To-First-Token (TTFT) on long prompts.

### PrefillScorer Trait

```rust
pub trait PrefillScorer: Send + Sync {
    fn score(&self, tokens: &[usize], weights, config) -> Vec<f32>;
    fn score_into(&self, tokens: &[usize], weights, config, buf: &mut Vec<f32>);
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
| `block_select_entmax(block_scores, cfg) -> Vec<usize>` | Entmax-based block selection with adaptive support. |
| `block_select_grid(grid, m, n, h, cfg) -> Vec<usize>` | 2D grid block selection for multi-head/multi-layer scoring. |
| `block_score_maxsim(q_blocks, d_blocks) -> Vec<f32>` | MaxSim block scoring (behind `"maxsim"` feature). |
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

## Drafter LoRA (`speculative/drafter_lora.rs`)

LoRA-trained drafter weights for improved speculative decoding quality (Plan 117: MTP LoRA Drafter).

### Key Types

```rust
pub struct DrafterLoraWeights {
    pub q_lora: LoraAdapter,
    pub k_lora: LoraAdapter,
    pub v_lora: LoraAdapter,
    pub o_lora: LoraAdapter,
    pub mlp1_lora: LoraAdapter,
    pub mlp2_lora: LoraAdapter,
}

pub struct TrainingPair {
    pub input_tokens: Vec<usize>,
    pub target_token: usize,
}

pub struct DrafterForwardContext { /* internal buffers */ }
```

### Key Functions

| Function | Description |
|----------|-------------|
| `train_drafter_lora(weights, config, pairs, epochs, lr) -> DrafterLoraWeights` | Train LoRA adapters from input-target pairs. |
| `generate_training_pairs_from_replays(tokens, window, rng) -> Vec<TrainingPair>` | Generate training pairs from replay buffers. |
| `generate_synthetic_pairs(weights, config, prompt, rng) -> Vec<TrainingPair>` | Generate synthetic training pairs from model outputs. |
| `save_drafter_lora(path, weights, config) -> Result<()>` | Serialize LoRA weights to file. |
| `load_drafter_lora(path, config) -> Result<DrafterLoraWeights>` | Deserialize LoRA weights from file. |

---

## Alpha: Lattice Deduction (`speculative/alpha.rs`, behind `"lattice_deduction"` feature)

LDT α-operator for progressive multi-solution supervision (Plan 088).

### Key Functions

| Function | Description |
|----------|-------------|
| `is_consistent(current: &[Option<usize>], solution: &[usize]) -> bool` | Check if a solution is consistent with current commitments. |
| `alpha_intersect(current, solutions) -> Vec<HashSet<usize>>` | Intersect current state with union of consistent solutions. |

### AlphaTarget

```rust
pub struct AlphaTarget {
    // current: Vec<Option<usize>>,  // Some = committed, None = open
    // solutions: Vec<Vec<usize>>,    // K pre-computed valid solutions
}
```

Methods: `new(len, solutions)`, `commit(pos, val)`, `uncommit(pos)`, `reset()`, `target()`, `is_allowed(pos, token)`, `remaining_solutions()`, `current()`, `len()`, `is_empty()`.

### LDT Pruning Types

| Type | Description |
|------|-------------|
| `LdtPruneConfig` | Config: `theta_elim` (entropy elimination threshold), `enabled`. |
| `ConflictDetector` trait | `is_conflicted(marginals, current, pos) -> bool`. |
| `EntropyConflictDetector` | Implementation: entropy-based conflict detection with `max_prune_rate`, `entropy_floor`. |
| `LDT_THETA_ELIM` | Default theta elimination constant. |

---

## Answer Extraction (`speculative/answer_extract.rs`, behind `"parallel_probe"` feature)

Extracts answers from token streams for parallel probe consensus.

| Type | Description |
|------|-------------|
| `AnswerExtractor` trait | `extract_answer(text: &str) -> Option<A>`. |
| `RegexAnswerExtractor` | Extracts answers via `\boxed{}`, `answer is`, or numeric patterns. |
| `ThinkTokenExtractor` | Extracts from `</think closing tags. |
| `DiscreteActionExtractor` | Extracts integer action IDs (e.g., `Action 3`, `= 2`). |

---

## DiffusionSampler (`speculative/diffusion_sampler.rs`, behind `"tri_mode"` feature)

Learned accept/reject decisions for D2F denoising steps (Plan 116).

### Key Types

```rust
pub struct SamplerFeatures {
    pub top1_prob: f32,
    pub margin: f32,
    pub top3_mass: f32,
    pub entropy: f32,
    pub step_norm: f32,
    pub pos_norm: f32,
}

pub enum SamplerVariant {
    Logistic,
    Mlp { hidden_dim: usize },
    Transformer { d_model: usize, n_layers: usize },
}

pub enum DiffusionSampler {
    Logistic(LogisticSampler),
    Mlp(MlpSampler),
    Transformer(TransformerSampler),
}

pub struct SamplerTrajectory {
    pub features: SamplerFeatures,
    pub correct: bool,
}

pub struct SamplerDecision {
    pub p_correct: f32,
    pub accept: bool,
}
```

### Key Functions

| Function | Description |
|----------|-------------|
| `DiffusionSampler::auto(config) -> Self` | Auto-select variant based on model size. |
| `DiffusionSampler::predict(&self, features) -> f32` | Predict correctness probability. |
| `DiffusionSampler::decide(&self, features) -> SamplerDecision` | Make accept/reject decision. |
| `DiffusionSampler::train(&mut self, trajectories)` | Train on collected trajectories. |
| `collect_trajectories(weights, config, decode_config, pruner, rng, n, cap) -> Vec<SamplerTrajectory>` | Collect training data from D2F decode runs. |
| `train_logistic_on_patterns(weights, config, decode_config, pruner, rng, n) -> DiffusionSampler` | Convenience: collect + train logistic sampler. |

---

## PPoT: Probabilistic Programs of Thought (`speculative/ppot/`, behind `"ppot"` feature)

Logit-parameterized CPU resampling for rescuing rejected speculative paths. No additional GPU forward passes needed.

### Architecture

```
DFlash → DDTree → Verify
                ↓ (all rejected)
          ┌─────────────────────────────────┐
          │     PPoT Rescue (CPU only)       │
          │                                 │
          │  1. Read marginals              │
          │  2. Calculate per-position H(i) │
          │  3. Identify high-H positions   │
          │  4. For m samples:              │
          │     a. Resample positions       │
          │     b. Screen via Pruner        │
          │     c. If valid → return path   │
          │  5. All invalid → greedy fallback│
          └─────────────────────────────────┘
```

### Sub-modules

| Module | Description |
|--------|-------------|
| `entropy` | Entropy-based position identification + adaptive + rule-based |
| `resample` | Resampling + rescue (baseline + adaptive + multi-strategy) |
| `rank` | Self-consistency ranking + weighted ranking + variant selection |
| `knowledge` | Rejection memory + session knowledge (TRT-inspired) |
| `types` | `PpotConfig`, `TokenRule` |

### Key Re-exports

| Item | Description |
|------|-------------|
| `ppot_rescue(marginals, pruner, rng, config)` | Baseline rescue: random resampling at high-entropy positions. |
| `ppot_rescue_adaptive(marginals, pruner, rng, config, knowledge)` | Adaptive rescue: rejection memory + strategy cycling. |
| `ppot_resample(marginals, positions, rng)` | Resample specific positions. |
| `ppot_resample_different_value(marginals, position, current, rng)` | Resample to a different value. |
| `ppot_resample_with_support(marginals, positions, support, rng)` | Resample with restricted support set. |
| `ppot_resample_multi_strategy(marginals, positions, rng, strategies)` | Multi-strategy resampling. |
| `identify_high_entropy_positions(marginals, threshold)` | Find positions above entropy threshold. |
| `identify_positions_adaptive(marginals, config, knowledge)` | Adaptive position selection using session knowledge. |
| `identify_positions_by_rule(marginals, rule)` | Rule-based position selection. |
| `token_entropy(marginals, position)` | Compute Shannon entropy at a position. |
| `rank_by_consistency(sequences)` | Rank candidate sequences by self-consistency. |
| `select_best_variant(sequences, scores)` | Select the best variant from candidates. |

---

## Feature Flags Summary

| Flag | Enables |
|------|---------|
| `sparse_mlp` (default) | TwELL-inspired sparse MLP matmul |
| `domain_latent` (default) | Free Transformer mid-layer domain conditioning |
| `ppot` (default) | PPoT logit-parameterized CPU resampling + adaptive rescue |
| `bandit` (default) | Multi-armed bandit + FlowPruner + AbsorbCompress + HotSwapPruner |
| `peira_distill` | PeiraPruner — PEIRA alignment-modulated screening |
| `lattice_deduction` | AlphaTarget + ConflictDetector + LDT pruning (Plan 088) |
| `parallel_probe` | ParallelProbeVerifier + AnswerExtractor (Plan 133) |
| `dllm` | D2F Discrete Diffusion Forcing — `D2fContext`, `D2fPipeline`, block-parallel decode (Plan 066) |
| `tri_mode` | Tri-Mode inference — depends on `"dllm"`. `D2fDrafterVerifier` + `SelfSpecConfig` + `DiffusionSampler` (Plan 089, 116) |
| `dmax_spd` | DMax Soft Parallel Decode — hybrid embedding D2F (Plan 109) |
| `elf_sde` | SDE noise injection + `EarlyStopGate` + `WidthScaleConfig` (ELF Plan 079) |
| `eqr_convergence` | `ResidualTracker` for convergence tracking |
| `sr2am_configurator` | `entropy_truncate_horizon()` + `speculative_step_with_configurator()` |
| `stability_metrics` | `StabilitySnapshot` compute + from_phases |
| `spec_cost_model` | `SpecCostSnapshot` for cost analysis |
| `tes_loop` | SimpleTES re-exports: `TesConfig`, `TesNode`, `TrajectoryCredit` |
| `decode_specialize` | `DecodeStage` re-export from transformer module |
| `maxsim` | `ScoreReduction::MaxSim` + `block_score_maxsim()` |
| `"sudoku"` | `SudokuPruner` — constrained decoding for Sudoku |
| `"rest"` | REST bridge test + `merge_retrieved_branches` (client in `riir-ai/riir-rest`) |
| `"feedback"` | E2E feedback loop — sends `InferenceResult` to REST endpoint |

---

## Key Design Principles

1. **Zero-allocation hot path:** All `_with` variants, `TreeBuilder`, and `D2fContext` reuse pre-allocated buffers. No `Vec::push` in the inner loop — only `clear()` + index writes. D2F uses flat 2D buffers (`[pos * dim..(pos+1) * dim]`) instead of `Vec<Vec<f32>>`.
2. **Separable pipeline stages:** DFlash → DDTree → Verifier are independent. D2F is an alternative pipeline: `forward_block_causal` → confidence remasking → constraint pruning. Each can be swapped, benchmarked, or disabled independently.
3. **Config-driven behavior:** `SpeculativeConfig` controls lookahead depth, tree budget, acceptance rates, and parallelism thresholds. `D2fDecodeConfig` controls denoising steps, confidence thresholds, block size, and multistep mode. `DecodeStrategy` auto-selects the optimal pipeline. No runtime branching on magic numbers.
4. **Feature-gated complexity:** REST bridge, constraint pruners, D2F, PPoT, and advanced pruners are behind feature flags. The default build stays lean.
5. **Composable pruners:** `FlowPruner<P>` and `PeiraPruner<P>` wrap any `ScreeningPruner` — zero-modification composition. `EarlyStopGate<P>` adds depth-aware gating. All can be layered: `PeiraPruner<FlowPruner<BanditPruner>>`.
