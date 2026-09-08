# katgpt-rs: Advanced Techniques (Lucebox-Hub Distillation)

## Source
Techniques distilled from [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — open LLM inference optimized per-chip. We take the algorithmic ideas (chain-seed DDTree, importance scoring, rollback) and implement them on CPU without CUDA.

## Plan Dependency Map

```
Plan 009 (REST)              Plan 010 (Multi-Layer)        Plan 011 (GQA + Paged)
     │                              │                            │
     │ hidden_state                 │ MultiLayerKVCache          │ n_kv_head, kv_dim()
     │ RestClient                   │ LayerWeights               │ PagedKVCache
     │ merge_retrieved_branches()   │ Config::small_target()     │ Config::gqa_draft()
     │                              │ Config.n_layer             │
     ▼                              ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Lucebox Techniques (This Doc)                       │
│                                                                         │
│  Chain-Seed DDTree ────── uses build_dd_tree_pruned() +                │
│                           build_dd_tree_screened/balanced/sde           │
│                           coexists with merge_retrieved_branches        │
│                                                                         │
│  Budget Sweep ─────────── uses Config.tree_budget,                     │
│                           sweeps micro/draft/small_target/gqa           │
│                                                                         │
│  KV Snapshot/Rollback ─── uses MultiLayerKVCache.layers,               │
│                           kv_dim(), forward() per layer                 │
│                           PagedKVCache.fork()/rollback() (Plan 014)     │
│                                                                         │
│  Speculative Prefill ──── uses hidden_state for scoring,                │
│                           draft model + MultiLayerKVCache,              │
│                           bridge to speculative_step_rest()             │
│                                                                         │
│  Target-Conditioned ───── uses hidden_state + MultiLayerKVCache         │
│                                                                         │
│  TurboQuant KV Cache ─── compresses f32→2-4bit per coordinate,          │
│                           random rotation + Lloyd-Max codebook           │
│                           composable with PFlash (precision × seq)      │
│                                                                         │
│  SpectralQuant KV Cache ─ data-driven eigenbasis + water-fill bits,    │
│                            two-regime (semantic/tail) quantization      │
│                            composable with PFlash + MaxSim               │
│                                                                         │
│  PFlash Block-Sparse ─── block-level importance scoring (sink+window+   │
│                           last_n_full+tail_window+alpha),               │
│                           entmax sparse routing (dash_attn),            │
│                           MaxSim late-interaction scoring (maxsim),     │
│                           block_select_grid for per-(q,k,head) scoring  │
│                           ported from lucebox-hub C++/CUDA              │
│                                                                         │
│  Next-Gen Extensions (Plans 164/166/167)                                │
│  Budget Adaptation ───── PFlash compression_ratio → DDTree budget       │
│  FlashAR Anchor-Fill ─── two-round AR+D2F (anchor then denoise)         │
│  FlashAR Consensus ───── dual-path H/V + ternary thermal routing         │
│  PhraseBoost ─────────── context trie phrase boosting for DDTree        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Technique 1: Chain-Seed DDTree

### Problem
Pure best-first DDTree gives acceptance length (AL) ~4 on quantized targets. The tree lacks a high-confidence "spine" to branch from.

### Solution
Two-phase tree construction:
1. **Phase A (Chain)**: Greedy argmax over marginals for each depth → build backbone of highest-probability tokens
2. **Phase B (Branch)**: Best-first expansion from ALL chain nodes (not just root) → branch alternatives

```rust
// dd_tree.rs — chain_seed parameter
build_dd_tree_pruned(marginals, config, pruner, chain_seed: bool)
build_dd_tree_screened(marginals, config, screener, chain_seed: bool)
build_dd_tree_balanced(marginals, config, screener, chain_seed, stop_probs, backward_weight, lambda_flow)
build_dd_tree_sde(marginals, config, screener, chain_seed, sde_config, rng)
build_dd_tree_balanced_sde(marginals, config, screener, chain_seed, stop_probs, backward_weight, lambda_flow, sde_config, rng)
```

- Chain nodes consume budget: if chain length = L, remaining = tree_budget - L
- Chain broken by constraint → fall through to standard best-first
- Coexists with `merge_retrieved_branches()` (REST merge adds branches AFTER chain-seed build)
- **Screened** variant blends LLM log-probs with relevance `R ∈ [0.0, 1.0]` via `ln(R)` penalty
- **Balanced** variant adds GFlowNet backward-weighted scoring + flow bonus
- **SDE** variants inject Gaussian noise (γ > 0) before tree building for diversity

### SDE Noise Injection (ELF Plan 079)
```rust
// dd_tree.rs
pub struct SdeConfig {
    pub gamma: f32,            // noise scale (default: 0.0 = disabled)
    pub confidence_floor: f32, // skip noise on confident tokens
    pub preserve_top1: bool,   // keep best token noise-free
}
```

### Width Scaling (PTRM Plan 083)
```rust
pub enum WidthSelectionMode { BestQ, MostFrequent, Top1Converged }
pub struct WidthScaleConfig { pub k_rollouts: usize, pub selection: WidthSelectionMode }
pub fn best_of_k_rollouts(...)  // build k trees, select best path
```

### Results
| Config | DDTree (no chain) | DDTree (chain-seed) |
|--------|:-:|:-:|
| micro | 364,458 trees/s | 385,957 trees/s |
| Draft sweep AL | baseline | marginal improvement at draft scale |

Lucebox found AL recovered from ~4 to ~9 at 27B scale. Benefit grows with model size.

## Technique 2: DDTree Budget Sweep

### Problem
Tree budget was hardcoded (16 or 32). Optimal budget depends on model size and target ratio.

### Solution
Sweep budgets empirically: `[4, 8, 12, 16, 20, 22, 24, 32, 48, 64]`
- Per budget: measure tree build time, node count, simulated acceptance length
- Lucebox found budget=22 sweet spot for RTX 3090 + 27B Q4_K_M

### Results (draft config, 75% simulated acceptance)
| Budget | Throughput | AL |
|--------|-----------|-----|
| 4 | fastest | low |
| **8** | **585K trees/s (optimal)** | good |
| 16 | baseline | good |
| 32+ | diminishing returns | marginal |

Optimal: budget=8 for draft config (throughput tradeoff). Budget scaling is model-dependent.

## Technique 3: KV-Cache Snapshot & Rollback

### Problem
DDTree branch verification writes to shared KV cache. On reject, stale data corrupts subsequent branches.

### Solution
```rust
// transformer.rs
pub struct KVSnapshot {
    pub pos: usize,
    pub layers: Vec<KVLayerSnapshot>,
}

pub struct KVLayerSnapshot {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

impl MultiLayerKVCache {
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        // Copies only filled slots [0..pos * kv_dim) per layer
    }
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        // Writes snapshot data back (no zeroing — each position written before read)
    }
}
```

- Cheap: copies only `[0..pos * kv_dim)` per layer, not entire `[block_size * kv_dim]`
- Micro config: ~2 KB per snapshot
- small_target (4 layers, kv_dim=64): ~128 KB per snapshot

### Integration
`speculative_step_rollback()` in `step.rs`:
1. Snapshot KV cache before verifying each DDTree branch
2. Run forward passes for branch tokens
3. On reject at position k: restore snapshot, try next branch
4. Extracts top-3 candidate paths (sorted by score), verifies each with rollback

### PagedKVCache fork-based rollback (Plan 014+)
```rust
// transformer.rs
pub struct PagedKVCache { /* paged KV storage with reference counting */ }

impl PagedKVCache {
    pub fn fork(&self) -> Self;            // shares prefix pages (copy-on-write)
    pub fn rollback(&mut self, pos: usize); // frees exclusive pages, restores shared
    pub fn write_kv(&mut self, layer, pos, key, value, config);
    pub fn read_kv(&self, layer, pos, config) -> (&[f32], &[f32]);
}
```

- `fork()` shares prefix pages (copy-on-write via `page_ref_counts`)
- `rollback()` frees exclusive pages, restores shared prefix
- Used by `speculative_step_rollback_paged()` in `step.rs`

### Results
| Method | Throughput | Notes |
|--------|-----------|-------|
| Leviathan (no rollback) | 108,827 tok/s | Corrupts cache on reject |
| **Leviathan (w/ rollback)** | **161,324 tok/s** | **+49% per accepted token** |

## Technique 4: Speculative Prefill (PFlash-Inspired)

### Problem
Long prompts require expensive target model prefill over every token. 128K tokens → slow TTFT.

### Solution
Use draft model's attention scores to identify important tokens, compress prompt before target prefill.

```rust
// src/speculative/prefill.rs
pub trait PrefillScorer: Send + Sync {
    fn score(&self, draft_weights, draft_config, prompt_tokens) -> Vec<f32>;
    fn score_into(&self, draft_weights, draft_config, prompt_tokens, scores: &mut [f32]);
}
pub struct AttentionScorer;       // Q·K attention importance (PFlash-inspired)
pub struct RandomScorer { pub seed: u64 };  // Baseline
pub struct UniformScorer;         // Baseline: equal importance for all tokens
pub struct BlockAttentionScorer { pub config: FlashPrefillConfig };  // Block-level aggregation
```

### Pipeline
1. `score_token_importance()` — run draft model forward per token, extract Q·K attention scores
2. `compress_prompt(importance_scores, keep_ratio, prefix_len, suffix_len)` — always keep first/last N, select top middle spans
3. `speculative_prefill(scorer, draft_weights, draft_config, prompt_tokens, keep_ratio, prefix_len, suffix_len)` — target model forward on compressed prompt → filled KV cache

### Block-Sparse Pipeline
4. `speculative_prefill_block(scorer, draft_weights, draft_config, prompt_tokens, cfg, prefix_len, suffix_len)` — PFlash compression with block scoring
5. `speculative_prefill_adaptive(scorer, ..., mode: PrefillMode, threshold, cfg, ...)` — adaptive threshold selection
6. `should_compress(mode: PrefillMode, prompt_len, threshold)` — whether to apply compression

### PrefillMode
```rust
pub enum PrefillMode {
    Off,     // Never compress
    Auto,    // Compress when prompt length >= threshold
    Always,  // Always compress (even short prompts)
}
```

### Results
| Method | Throughput | Effective Tokens | Notes |
|--------|-----------|:---:|-------|
| Prefill (no compress) | 2,691K tok/s | 64 | Full prompt |
| **Prefill (compressed)** | **1,714K tok/s** | **7** | ~10.9% keep ratio |

Compression trades throughput for compute savings: 128K → 2.6K tokens would give ~10.4× TTFT reduction.

### Bridge to REST
After prefill compression, `speculative_step_rest()` continues decode with REST retrieval.

## Technique 5: Target-Conditioned Draft

### Problem
DFlash produces independent marginals (same token/pos each step). Every position conditions on the same input, not on real target features.

### Solution
Seed draft model's KV cache with target hidden state:
```rust
// dflash.rs
pub fn dflash_predict_conditioned(
    draft_weights, config, token, pos,
    target_hidden_state: &[f32],
    rng: &mut Rng,
) -> DraftResult
```
- Projects target `hidden_state` to draft `kv_dim`
- Seeds draft KV cache with projected hidden state
- Draft model conditions on real target features, not its own noisy predictions
- Returns `DraftResult` with marginals + sampled_tokens (+ optional routing_overlap, cost_snapshot, stability via feature gates)

### Integration
`speculative_step_conditioned()` — target forward → hidden state → conditioned draft → DDTree → simulated acceptance

### Results
| Method | Throughput | Accept Len |
|--------|-----------|:---:|
| Spec (unconditioned) | 842,657 tok/s | 5.00 |
| **Spec (conditioned)** | **972,163 tok/s** | **6.74** |

+15% acceptance length improvement from target conditioning.

## Technique 6: TurboQuant KV Cache Compression (Plan 043)

### Problem
KV cache is the memory bottleneck for long-context inference. `MultiLayerKVCache` stores f32 keys+values growing linearly with sequence length. 32K context × 128 head_dim × 32 layers = 1 GB.

### Solution
Compress each KV coordinate from 32-bit f32 to 2-4 bits using TurboQuant (Zandieh et al., 2025):
1. **Normalize** → unit vector
2. **Random rotation** (QR-based orthogonal Π) → coordinates become Beta-distributed
3. **Lloyd-Max codebook** → optimal scalar quantizer per coordinate
4. **Bit-pack** → 2/3/4 bits per coordinate stored as u8 array

```rust
// crates/katgpt-quant/src/turboquant/kv_cache.rs
pub struct TurboQuantKVCache { /* bit-packed indices + norms + rotation matrices */ }

impl TurboQuantKVCache {
    pub fn new(config: &Config) -> Self;
    pub fn new_asymmetric(key_bits: u8, val_bits: u8, config: &Config) -> Self;
    pub fn with_config(cfg: &TurboQuantKVCacheConfig) -> Self;
    pub fn store_key(&mut self, layer, pos, key: &[f32]);    // quantize + pack
    pub fn store_value(&mut self, layer, pos, value: &[f32]);
    pub fn dequantize_key(&self, layer, pos) -> Vec<f32>;     // unpack + rotate back
    pub fn dequantize_value(&self, layer, pos) -> Vec<f32>;
    pub fn dequantize_key_into(&self, layer, pos, out: &mut [f32]);
    pub fn dequantize_value_into(&self, layer, pos, out: &mut [f32]);
    pub fn bytes_per_token(&self) -> usize;                    // packed size
    pub fn compression_ratio(&self) -> f64;                    // flat / packed
    pub fn pos(&self) -> usize;
    pub fn set_pos(&mut self, pos: usize);
    pub fn kv_dim(&self) -> usize;
    pub fn reset(&mut self);
}
```

### Configuration
```rust
// crates/katgpt-quant/src/turboquant/types.rs
pub struct TurboQuantKVCacheConfig {
    pub n_layers: usize,
    pub kv_dim: usize,
    pub max_seq_len: usize,
    pub seed: u64,
    pub key_bits: u8,   // default: 3
    pub val_bits: u8,   // default: 3
}
```

### Key Properties
- **Data-oblivious**: No calibration data needed, works on any distribution
- **Online**: Per-token quantization, no preprocessing
- **Unbiased**: E[estimated ⟨Q,K⟩] = true ⟨Q,K⟩ (Algorithm 2 guarantee)
- **Composable**: Orthogonal to Raven (sequence compression), SpectralQuant, and PFlash (token reduction)

### Results
| Bits | Compression | Key cos_sim | Attention corr | Output cos_sim |
|:----:|:-----------:|:-----------:|:--------------:|:--------------:|
| 2 | 8.0× | 0.9242 | 0.9450 | 0.9699 |
| **3** | **5.3×** | **0.9825** | **0.9907** | **0.9989** |
| 4 | 5.3× | 0.9958 | 0.9978 | 0.9975 |

At 32K context (hypothetical hd=128): **1073.7 MB → 151.0 MB (7.1× compression)**.

### Modules
- `crates/katgpt-quant/src/turboquant/codebook.rs` — Lloyd-Max codebook computation
- `crates/katgpt-quant/src/turboquant/rotation.rs` — QR-based orthogonal rotation + QJL projection
- `crates/katgpt-quant/src/turboquant/kv_cache.rs` — Bit-packed compressed KV cache (implements `QuantizedKVCache` trait from `src/types.rs`)
- `crates/katgpt-quant/src/turboquant/forward.rs` — Dequantization + attention forward path
- `crates/katgpt-quant/src/turboquant/types.rs` — `TurboQuantCodebook`, `TurboQuantLayer`, `TurboQuantKVCacheConfig`

## Technique 7: SpectralQuant KV Cache Compression (Plan 078)

### Problem
TurboQuant uses data-oblivious (random) rotation — optimal on average but suboptimal for any specific model. Real KV cache distributions have strong spectral structure: a few eigenvalues dominate.

### Solution
Data-driven eigenbasis quantization:
1. **Offline calibration**: Collect KV samples → covariance eigendecomposition → eigenbasis V
2. **Two-regime allocation**: Semantic (top d_eff eigenvalues) get more bits; tail gets fewer
3. **Water-fill bit allocation**: Per-dim bits ∝ eigenvalue (Lagrange multiplier optimization)
4. **Lloyd-Max codebook**: Per-dim non-uniform scalar quantizer for each regime
5. **QJL projection**: Subspace residual estimation for quantization error bounds

```rust
// crates/katgpt-spectral/src/spectral_kv_cache.rs
pub struct SpectralQuantKVCache {
    pub layers: Vec<SpectralQuantLayer>,
    // variable-bit packed key/val indices, norms, scratch buffers
}

// crates/katgpt-spectral/src/types.rs
pub struct SpectralQuantLayer {
    pub calibration: SpectralQuantCalibration,
    pub qjl_signs: Vec<f32>,
    pub tail_codebook: LloydMaxCodebook,
    pub semantic_codebook: Option<LloydMaxCodebook>,      // v1 uniform
    pub per_dim_semantic_codebooks: Option<Vec<LloydMaxCodebook>>, // v2 water-fill
    pub d_eff: usize,
    pub b_high: u8,  // semantic regime bits
    pub b_low: u8,   // tail regime bits
}
```

### Calibration
```rust
pub fn calibrate_eigenbasis(kv_samples, config) -> CalibrationResult;
pub fn waterfill_bits(eigenvalues, total_budget, min_bits, max_bits) -> WaterfillAllocation;
pub fn generate_selective_qjl_signs(d_eff, qjl_dim, seed) -> Vec<f32>;
```

### Modules
- `crates/katgpt-spectral/src/spectral.rs` — Eigenbasis calibration, water-fill, bit allocation
- `crates/katgpt-spectral/src/nonuniform_quant.rs` — Non-uniform scalar quantizer (`NonUniformQuantizer`, `CompressedVector`)
- `crates/katgpt-spectral/src/spectral_rotation.rs` — Eigenbasis rotation (`SpectralRotation`), random rotation (`RandomRotation`, gated by `turboquant` feature)
- `crates/katgpt-spectral/src/spectral_kv_cache.rs` — SpectralQuant KV cache (implements `QuantizedKVCache`)
- `crates/katgpt-spectral/src/forward.rs` — Dequantization + attention, parallel dequantize, MaxSim scoring (gated by `maxsim` feature)
- `crates/katgpt-spectral/src/types.rs` — `LloydMaxCodebook`, `SpectralQuantCalibration`, `WaterfillAllocation`, `SpectralQuantKVCacheConfig`

## Technique 8: PFlash Block-Sparse Speculative Prefill (Plan 044)

### Problem
Long-context prefill is O(S²). Vanilla llama.cpp on RTX 3090 takes ~257s to prefill 131K tokens. User waits 4+ minutes before first token.

### Solution
Score per-block importance using draft model's tail attention, then select important blocks with structured rules:

```rust
// src/speculative/prefill.rs
pub fn block_select(block_scores: &[f32], cfg: &FlashPrefillConfig) -> Vec<usize>;
pub fn block_select_grid(score: &[f32], num_q_blocks, num_k_blocks, num_heads, cfg: &FlashPrefillConfig) -> (Vec<i32>, Vec<i32>);
pub fn block_select_entmax(block_scores: &[f32], cfg: &FlashPrefillConfig) -> Vec<usize>;  // gated by dash_attn
pub fn block_score_maxsim(q_block: &[f32], k_block: &[f32], block_len_q, block_len_k, dim) -> f32;  // gated by maxsim
pub fn compress_prompt_blocks(importance_scores: &[f32], cfg: &FlashPrefillConfig, prefix_len: usize, suffix_len: usize) -> Vec<usize>;
```

### Block Selection Rules
1. **Sink rule**: First `attention_sink` blocks always kept (system prompt)
2. **Window rule**: Blocks within `window` of query position always kept (local context)
3. **last_n_full**: When query is in last N blocks, keep all (short prompt safety)
4. **Alpha rule**: Keep blocks with `score >= max_score × alpha` (importance threshold)
5. **Entmax rule** (`dash_attn` feature): α-entmax (α=1.5) sparse routing — variable support size adapts to query difficulty
6. **MaxSim scoring** (`maxsim` feature): Late-interaction `Σ_i max_j dot(Q[i], K[j])` replaces mean dot-product for better needle detection

### Pipeline
```
prompt tokens
    │
    ▼
block_select / block_select_entmax (sink + window + last_n + alpha/entmax)
    │
    ▼
block_select_grid (per-(q_block, k_block, head) selection)  [optional]
    │
    ▼
block_score_maxsim (MaxSim late-interaction scoring)  [maxsim feature]
    │
    ▼
compress_prompt_blocks (prefix + suffix + selected blocks)
    │
    ▼
target model prefill on compressed tokens
```

### Config
```rust
// crates/katgpt-core/src/speculative/types.rs
pub struct FlashPrefillConfig {
    pub block_size: usize,          // tokens per block (default: 32)
    pub attention_sink: usize,      // initial blocks to keep (default: 1)
    pub window: usize,              // adjacent blocks to keep (default: 2)
    pub last_n_full: usize,         // final blocks getting full attention (default: 1)
    pub tail_window: usize,         // tail blocks for importance scoring (default: 4)
    pub alpha: f32,                 // importance threshold (default: 0.15)
    pub score_reduction: ScoreReduction,  // SoftmaxSum or MaxSim
}

pub enum ScoreReduction {
    SoftmaxSum,  // standard attention (default)
    MaxSim,      // late-interaction (gated by maxsim feature)
}
```

### Config Presets
```rust
FlashPrefillConfig::default()        // block_size=32, sink=1, window=2, last_n=1, tail_window=4, alpha=0.15
FlashPrefillConfig::metal()          // block_size=64, optimized for Apple Silicon
FlashPrefillConfig::long_context()   // block_size=64, alpha=0.85, tail_window=8, aggressive compression for 64K+ ctx
FlashPrefillConfig::short_context()  // block_size=32, alpha=0.12, tail_window=2, conservative for <4K ctx
```

### BlockScores
```rust
pub struct BlockScores {
    pub num_blocks: usize,
    pub block_size: usize,
    pub scores: Vec<f32>,
    pub selected: Vec<usize>,
}
```

### Results
| Context | Alpha | Before | After | Reduction | NIAH |
|:-------:|:-----:|:------:|:-----:|:---------:|:----:|
| 512 | 0.15 | 512 | 192 | 2.7× | ✅ |
| 1024 | 0.15 | 1024 | 192 | 5.3× | ✅ |
| 2048 | 0.15 | 2048 | 192 | 10.7× | ✅ |
| 4096 | 0.15 | 4096 | 192 | 21.3× | ✅ |

NIAH retrieval: **20/20 = 100%** across all context sizes and alpha values.

C++ reference (RTX 3090, BSA): 128K → 2.6K (50× reduction), TTFT 257s → 24.8s (**10.4×** speedup).

### Composable with TurboQuant
| Config | Sequence | Memory | Combined |
|--------|----------|--------|----------|
| TQ 3-bit + PF α=0.15 | 9.4% | 18.8% | **14.9% (6.7× reduction)** |

Both reductions multiply: PFlash reduces tokens, TurboQuant reduces bits per token.

> **PASS-Redirects (synthesis):** Jiang et al. [arXiv:2407.02490 "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention"] — per-head static patterns (A-shape/Vertical-Slash/Block-Sparse) computed once at calibration = the same offline-calibration shape as RT-Turbo's per-head retrieval scoring; covered by PFlash + rt_turbo (this doc + Bench 035), league-adjacent dense cells exclude approximate prefill by pin. Lai et al. [arXiv:2502.20766 "FlexPrefill: A Context-Aware Sparse Attention Framework for Long-Context LLM Prefilling"] — per-prompt context-aware selection with lower selection-scan overhead; the runtime-selection endpoint's long-P regression is already measured (negative_results #36: KV-outer 2.02× @32K → 0.83× @512K); no product pull below the opt-in bar.

## Additional Types

### DraftResult
```rust
// crates/katgpt-core/src/speculative/types.rs
pub struct DraftResult {
    pub marginals: Vec<Vec<f32>>,
    pub sampled_tokens: Vec<usize>,
    pub routing_overlap: Option<RoutingOverlapSnapshot>,   // gated by domain_latent
    pub cost_snapshot: Option<SpecCostSnapshot>,            // gated by spec_cost_model
    pub stability: Option<StabilitySnapshot>,               // gated by stability_metrics
}
```

### SpeculativeContext
```rust
pub struct SpeculativeContext {
    pub ctx: ForwardContext,
    pub cache: MultiLayerKVCache,
    pub marginals_flat: Vec<f32>,
    pub probs_buf: Vec<f32>,
    pub sampled_tokens: Vec<usize>,
    pub accepted_buf: Vec<usize>,
    pub path_buf: Vec<usize>,
    pub residual_buf: Vec<usize>,
    pub p_distributions_flat: Vec<f32>,
    pub steps_populated: usize,
    pub sde_config: SdeConfig,
}
```

### DDTreeBranchCache
```rust
pub struct DDTreeBranchCache { /* paged KV cache with branch fork/rollback */ }
impl DDTreeBranchCache {
    pub fn new(config: &Config) -> Self;
    pub fn fork_branch(&mut self, config: &Config) -> Option<usize>;
    pub fn forward_branch(&mut self, branch: usize, ...);
    pub fn rollback_branch(&mut self, branch: usize);
    pub fn discard_branch(&mut self, branch: usize);
    pub fn reset(&mut self);
}
```

### DecodeStrategy
```rust
pub enum DecodeStrategy {
    Autoregressive,                                          // default
    Speculative,                                             // draft model
    DiscreteDiffusion,                                       // gated by dllm
    DiscreteDiffusionSoft,                                   // gated by dmax_spd
    SelfSpeculation,                                         // gated by tri_mode
}
```

### LDT Pruning
```rust
pub const LDT_THETA_ELIM: f32 = ...;
pub struct LdtPruneConfig { pub theta_elim: f32, pub enabled: bool }
```

### Self-Speculation (Tri-Mode)
```rust
pub struct SelfSpecConfig { pub draft_width: usize, pub d2f_config: D2FConfig, pub sampler: ... }
```

### Conflict Detection
```rust
pub trait ConflictDetector { fn is_conflicted(&self, logits: &[f32], ...original: &[f32]) -> bool; }
pub struct EntropyConflictDetector { pub max_prune_rate: f32, pub entropy_floor: f32 }
```

### Trajectory Credit (PTRM)
```rust
pub struct TrajectoryCredit { pub num_trajectories, best_score, worst_score, best_trajectory_idx, worst_trajectory_idx }
impl TrajectoryCredit { pub fn from_trajectory_scores(...), pub fn node_weight(...), pub fn all_weights(...), pub fn assign_credit(...) }
```

### TES Config (PTRM)
```rust
pub struct TesConfig { pub global_width, refinement_depth, local_sample_size, bandit_strategy }
pub struct TesNode { pub solution, score, metadata, parent_idx, visit_count, propagated_value }
```

## Architecture Decisions

1. **Chain-seed is additive** — `build_dd_tree()` works as before (chain_seed=false)
2. **Prefill is a new module** — `src/speculative/prefill.rs`, no feature flag needed
3. **KV snapshot copies only filled slots** — cheap at our scale, uses `kv_dim()` for GQA
4. **Target conditioning via KV seed** — simplest option, no weight changes
5. **Flat cache + PagedKVCache** — `PagedKVCache` with fork/rollback now implemented (Plan 014+)
6. **No new model weights** — reuses draft model attention + target hidden_state
7. **TurboQuant is a separate module** — not extension of existing KV cache, lives in `src/turboquant/`
8. **SpectralQuant is a separate module** — data-driven alternative to TurboQuant, lives in `src/spectralquant/`
9. **PFlash uses FlashPrefillConfig** — config-driven, no feature flag for core path
10. **Feature gates for research extensions** — `dash_attn` (entmax), `maxsim` (MaxSim scoring), `elf_sde` (noise injection), `dllm` / `dmax_spd` / `tri_mode` (decode strategies), `spectral_quant` (SpectralQuant), `domain_latent` / `spec_cost_model` / `stability_metrics` (telemetry), `budget_adaptation` (Plan 167), `flashar_anchor` (Plan 166), `phrase_boost` (Plan 164), `plasma_path` (Plan 148 ternary weights)

## Key References
- [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time
- [DFlash: Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2602.06036) — Wang et al., 2026
- [DDTree: Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989) — Ringel & Romano, 2026
- [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631) — Liu et al., ICLR 2026
- [FlashPrefill](https://arxiv.org/abs/2603.06199) — Fan et al., 2026
- [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/pdf/2504.19874) — Zandieh et al., 2025

---

## Next-Generation Lucebox Extensions

Techniques that extend the core Lucebox ideas into new decode strategies, adaptive
budgets, and domain-specific boosting. These are newer (Plans 164–167) and gated
behind feature flags until validated by GOAT benchmarks.

### Technique 9: Budget Adaptation (Plan 167)

**Problem.** Fixed `tree_budget` over-spends on simple prompts and under-spends on
complex ones. The PFlash pipeline already computes a *compression ratio* — the
fraction of prompt blocks that pass the importance filter — but discards it after
block selection.

**Solution.** Feed the compression ratio back into the DDTree budget as a complexity
signal. High ratio (many blocks selected) → complex prompt → more tree budget.
Low ratio → simple → less budget. Budget is clamped to `[base/2, base*2]`.

```rust
pub fn adaptive_tree_budget(
    base_budget: usize,
    compression_ratio: f32,
    mode: BudgetAdaptation,
) -> usize {
    match mode {
        BudgetAdaptation::Off => base_budget,
        BudgetAdaptation::Compression => {
            let r = compression_ratio.clamp(0.0, 1.0);
            let scale = 0.5 + 1.5 * r;
            let adapted = (base_budget as f32 * scale) as usize;
            adapted.max(base_budget / 2).min(base_budget * 2)
        }
        BudgetAdaptation::Entropy => base_budget, // TODO: future
    }
}
```

```rust
pub enum BudgetAdaptation {
    #[default]
    Off,
    Compression,  // Scale by PFlash compression ratio
    Entropy,      // Placeholder for first-marginal entropy
}
```

**Key property:** Zero additional cost — reuses a value already computed during
PFlash block selection. Feature-gated behind `budget_adaptation`.

**Source:** `crates/katgpt-speculative/src/budget.rs`, `crates/katgpt-speculative/src/budget_compat.rs`

---

### Technique 10: FlashAR Anchor-Then-Fill (Plan 166)

**Problem.** Pure D2F denoising on large blocks requires many iterations because
every position starts masked. The denoiser must simultaneously discover both local
and global structure.

**Solution.** Split decoding into two rounds:

1. **Round 1 (Anchor):** AR predicts every S-th position (stride). A few AR
   forward passes produce high-quality anchor tokens.
2. **Round 2 (Fill):** D2F denoises the remaining positions with anchor tokens
   pre-filled (unmasked). Reduced search space → fewer denoising steps.

```rust
pub struct AnchorConfig {
    pub stride: usize,  // S=1 → pure AR, S=block_size → pure D2F
}

pub struct AnchorFillResult {
    pub tokens: Vec<usize>,
    pub n_anchors: usize,
    pub fill_steps_used: usize,
    pub baseline_steps_used: usize,
    pub step_reduction: usize,
}
```

**Key property:** Anchor positions start unmasked in Round 2, directly reducing
the denoising search space. The `step_reduction` field quantifies the gain over
baseline D2F. Feature-gated behind `flashar_anchor` (requires `dllm`).

**Source:** `src/speculative/flashar_anchor.rs`

---

### Technique 11: FlashAR Consensus (Plan 166)

**Problem.** Tri-mode's prefix-match acceptance is coarse — either both paths agree
on the entire prefix or they don't. No per-position granularity, no confidence signal.

**Solution.** Dual-path draft with per-position ternary consensus and thermal routing:

- **Path H:** AR/MTP draft → tokens + confidence
- **Path V:** D2F block draft → tokens + confidence

Ternary consensus per position:

```
 +1 → H wins (conf_H > conf_V)
  0 → AGREE (both same token) → PLASMA PATH (skip verify)
 -1 → V wins (conf_V >= conf_H)
```

Thermal routing assigns a verification level based on consensus + confidence:

```rust
pub enum ThermalPath {
    Plasma,  // Both agree, high conf → accept immediately (zero verify)
    Hot,     // One wins, high conf → accept winner
    Warm,    // One wins, mid conf → AR spot-check
    Cold,    // Both low conf → fallback prefix-match
}
```

**Key property:** `Plasma` positions skip verification entirely — a direct win when
both paths converge. This uses Plasma Path ternary weights (Plan 148, `TernaryWeights`)
for the underlying SIMD matvec. Feature-gated behind `plasma_path`.

**Source:** `src/speculative/flashar_consensus.rs`, `crates/katgpt-dec/src/simd.rs`

---

### Technique 12: PhraseBoost (Plan 164)

**Problem.** DDTree screening pruners are generic — they score tokens by
positional/entropy heuristics but have no domain vocabulary knowledge. For
structured domains (code, medical, legal), known phrases should be boosted.

**Solution.** Wrap any `ScreeningPruner` with a `PhraseBoostPruner` that layers
context-trie phrase boosting on top of the base pruner's relevance scores.

```rust
pub struct PhraseBoostPruner<P: ScreeningPruner> {
    inner: P,                                    // base pruner (preserved)
    trie: PhraseTrie,                            // pre-built phrase trie
    boost_score: f32,                            // raw boost value
    active_states: RwLock<HashMap<u128, Vec<usize>>>,  // per-path trie states
}
```

Boost is additive and normalized: `boost_score / (1.0 + boost_score)` ensures
the result stays bounded. Default `boost_score = 5.0` → normalized ≈ 0.833.

The trie tracks active states per DDTree path: after each token, only phrases
that continue from the current context remain active. Zero training cost —
phrases are provided at call site.

**Key property:** Modeled after parakeet.cpp's phrase boosting, adapted to our
DDTree pipeline. Feature-gated behind `phrase_boost` (default-on, Plan 164
GOAT 7/7).

**Source:** `crates/katgpt-pruners/src/phrase_boost.rs`, `crates/katgpt-pruners/src/phrase_trie.rs`
