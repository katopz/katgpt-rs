# katgpt-rs: Advanced Techniques (Lucebox-Hub Distillation)

## Source
Techniques distilled from [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — open LLM inference optimized per-chip. We take the algorithmic ideas (chain-seed DDTree, importance scoring, rollback) and implement them on CPU without CUDA.

## Plan Dependency Map

```
Plan 009 (REST)              Plan 010 (Multi-Layer)        Plan 011 (GQA + Paged)
     │                              │                            │
     │ hidden_state                 │ MultiLayerKVCache          │ n_kv_head, kv_dim()
     │ RestClient                   │ LayerWeights               │ PagedKVCache (pending)
     │ merge_retrieved_branches()   │ Config::small_target()     │ Config::gqa_draft()
     │                              │ Config.n_layer             │
     ▼                              ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Lucebox Techniques (This Doc)                       │
│                                                                         │
│  Chain-Seed DDTree ────── uses build_dd_tree_pruned() +                │
│                           coexists with merge_retrieved_branches        │
│                                                                         │
│  Budget Sweep ─────────── uses Config.tree_budget,                     │
│                           sweeps micro/draft/small_target/gqa           │
│                                                                         │
│  KV Snapshot/Rollback ─── uses MultiLayerKVCache.layers,               │
│                           kv_dim(), forward() per layer                 │
│                           future: PagedKVCache.fork()                   │
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
│  PFlash Block-Sparse ─── block-level importance scoring (sink+window+   │
│                           last_n_full+alpha), compress_prompt_blocks()  │
│                           ported from lucebox-hub C++/CUDA              │
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
```

- Chain nodes consume budget: if chain length = L, remaining = tree_budget - L
- Chain broken by constraint → fall through to standard best-first
- Coexists with `merge_retrieved_branches()` (REST merge adds branches AFTER chain-seed build)

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
    pub layers: Vec<KVLayerSnapshot>,
    pub pos: usize,
}

impl MultiLayerKVCache {
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        // Copies only filled slots [0..pos * kv_dim] per layer
    }
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        // Writes snapshot back + zeros stale positions
    }
}
```

- Cheap: copies only `[0..pos * kv_dim]` per layer, not entire `[block_size * kv_dim]`
- Micro config: ~2 KB per snapshot
- small_target (4 layers, kv_dim=64): ~128 KB per snapshot

### Integration
`speculative_step_rollback()` in `step.rs`:
1. Snapshot KV cache before verifying each DDTree branch
2. Run forward passes for branch tokens
3. On reject at position k: restore snapshot, try next branch
4. Extracts top-3 candidate paths, verifies each with rollback

### Results
| Method | Throughput | Notes |
|--------|-----------|-------|
| Leviathan (no rollback) | 108,827 tok/s | Corrupts cache on reject |
| **Leviathan (w/ rollback)** | **161,324 tok/s** | **+49% per accepted token** |

### Future: PagedKVCache fork-based rollback
- `PagedKVCache.fork()` shares prefix pages (copy-on-write)
- Only new pages allocated after fork point
- Deferred to Plan 014 — currently uses flat snapshot/restore

## Technique 4: Speculative Prefill (PFlash-Inspired)

### Problem
Long prompts require expensive target model prefill over every token. 128K tokens → slow TTFT.

### Solution
Use draft model's attention scores to identify important tokens, compress prompt before target prefill.

```rust
// speculative/prefill.rs
pub trait PrefillScorer: Send + Sync {
    fn score(&self, draft_weights, draft_config, prompt_tokens) -> Vec<f32>;
}
pub struct AttentionScorer;  // Q·K attention importance (PFlash-inspired)
pub struct RandomScorer;     // Baseline
pub struct UniformScorer;    // Baseline: keep every Nth token
```

### Pipeline
1. `score_token_importance()` — run draft model forward per token, extract Q·K attention scores
2. `compress_prompt(tokens, scores, keep_ratio)` — always keep first/last N, select top middle spans
3. `speculative_prefill()` — target model forward on compressed prompt → filled KV cache

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
    weights, config, token, pos, hidden_state: &[f32]
) -> Vec<Vec<f32>>
```
- Projects target `hidden_state` to draft `kv_dim`
- Seeds draft KV cache with projected hidden state
- Draft model conditions on real target features, not its own noisy predictions

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
// turboquant/kv_cache.rs
pub struct TurboQuantKVCache { /* bit-packed indices + norms + rotation matrices */ }

impl TurboQuantKVCache {
    pub fn store_key(&mut self, layer, pos, key: &[f32]);    // quantize + pack
    pub fn dequantize_key(&self, layer, pos) -> Vec<f32>;     // unpack + rotate back
    pub fn bytes_per_token(&self) -> usize;                    // packed size
    pub fn compression_ratio(&self) -> f64;                    // flat / packed
}
```

### Key Properties
- **Data-oblivious**: No calibration data needed, works on any distribution
- **Online**: Per-token quantization, no preprocessing
- **Unbiased**: E[estimated ⟨Q,K⟩] = true ⟨Q,K⟩ (Algorithm 2 guarantee)
- **Composable**: Orthogonal to Raven (sequence compression) and PFlash (token reduction)

### Results
| Bits | Compression | Key cos_sim | Attention corr | Output cos_sim |
|:----:|:-----------:|:-----------:|:--------------:|:--------------:|
| 2 | 8.0× | 0.9242 | 0.9450 | 0.9699 |
| **3** | **5.3×** | **0.9825** | **0.9907** | **0.9989** |
| 4 | 5.3× | 0.9958 | 0.9978 | 0.9975 |

At 32K context (hypothetical hd=128): **1073.7 MB → 151.0 MB (7.1× compression)**.

### Modules
- `turboquant/codebook.rs` — Lloyd-Max codebook computation
- `turboquant/rotation.rs` — QR-based orthogonal rotation + QJL projection
- `turboquant/kv_cache.rs` — Bit-packed compressed KV cache (implements `QuantizedKVCache` trait from `src/types.rs`)
- `turboquant/forward.rs` — Dequantization + attention forward path
- `spectralquant/spectral_kv_cache.rs` — SpectralQuant KV cache (also implements `QuantizedKVCache` trait)

## Technique 7: PFlash Block-Sparse Speculative Prefill (Plan 044)

### Problem
Long-context prefill is O(S²). Vanilla llama.cpp on RTX 3090 takes ~257s to prefill 131K tokens. User waits 4+ minutes before first token.

### Solution
Score per-block importance using draft model's tail attention, then select important blocks with structured rules:

```rust
// speculative/prefill.rs
pub fn block_select(block_scores: &[f32], cfg: &FlashPrefillConfig) -> Vec<usize>;
pub fn block_select_grid(grid: &[f32], m: usize, n: usize, h: usize, cfg: &FlashPrefillConfig) -> Vec<usize>;
pub fn compress_prompt_blocks(scores: &[f32], cfg: &FlashPrefillConfig, prefix: usize, suffix: usize) -> Vec<usize>;
```

### Block Selection Rules
1. **Sink rule**: First `attention_sink` blocks always kept (system prompt)
2. **Window rule**: Blocks within `window` of query position always kept (local context)
3. **last_n_full**: When query is in last N blocks, keep all (short prompt safety)
4. **Alpha rule**: Keep blocks with `score >= max_score × alpha` (importance threshold)

### Pipeline
```
prompt tokens
    │
    ▼
block_select (sink + window + last_n + alpha)
    │
    ▼
compress_prompt_blocks (prefix + suffix + selected blocks)
    │
    ▼
target model prefill on compressed tokens
```

### Config Presets
```rust
FlashPrefillConfig::default()        // block_size=32, sink=1, window=2, last_n=1, alpha=0.15
FlashPrefillConfig::metal()          // block_size=64, optimized for Apple Silicon
FlashPrefillConfig::long_context()   // aggressive compression for 64K+ ctx
FlashPrefillConfig::short_context()  // conservative for <4K ctx
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

## Architecture Decisions

1. **Chain-seed is additive** — `build_dd_tree()` works as before (chain_seed=false)
2. **Prefill is a new module** — `speculative/prefill.rs`, no feature flag needed
3. **KV snapshot copies only filled slots** — cheap at our scale, uses `kv_dim()` for GQA
4. **Target conditioning via KV seed** — simplest option, no weight changes
5. **Flat cache first** — PagedKVCache integration deferred to Plan 014
6. **No new model weights** — reuses draft model attention + target hidden_state
7. **TurboQuant is a separate module** — not extension of existing KV cache, lives in `src/turboquant/`
8. **PFlash uses FlashPrefillConfig** — config-driven, no feature flag, CPU path with GPU kernel reserved for future

## Key References
- [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time
- [DFlash: Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2602.06036) — Wang et al., 2026
- [DDTree: Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989) — Ringel & Romano, 2026
- [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631) — Liu et al., ICLR 2026
- [FlashPrefill](https://arxiv.org/abs/2603.06199) — Fan et al., 2026
- [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/pdf/2504.19874) — Zandieh et al., 2025