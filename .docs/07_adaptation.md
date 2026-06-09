# katgpt-rs: Model Adaptation Techniques

Fifteen production techniques that adapt the transformer to different tasks and domains **without modifying base weights**. All are feature-gated, zero-copy, and backward-compatible.

| # | Technique | Plan | Feature Flag | What It Does |
|---|-----------|------|-------------|--------------|
| 1 | Bidirectional Prefill | 025 | — | Prompt tokens attend to ALL others during prefill |
| 2 | Modality LoRA Switching | 025 | — | reader→writer LoRA swap at prefill→decode boundary |
| 3 | Sparse MLP (TwELL) | 022 | `sparse_mlp` | Skip dead ReLU neurons, O(alive) FLOPs |
| 4 | Domain Latent Injection | 038 | `domain_latent` | Mid-layer K/V conditioning per domain |
| 5 | HLA Streaming Attention | 057/060 | `hla_attention` | O(1) constant-state attention, SIMD-accelerated |
| 6 | SpectralQuant KV Compression | 077 | `spectral_quant` | Calibrated eigenbasis + water-fill KV compression |
| 7 | ELF SDE Noise Injection | 079 | `elf_sde` | Logit-normal noise for diverse speculative tree expansion |
| 8 | CNA Steering | 087 | `cna_steering` | Contrastive neuron circuit discovery + modulation |
| 9 | Deep Manifold + Federation | 085 | `deep_manifold` / `federation` | Fixed-point residual scoring + federated KL coupling |
| 10 | SimpleTES Loop (RPUCG) | 086 | `tes_loop` | Credit-guided trajectory search with RPUCG bandit |
| 11 | GEPA-D Reflective Config Evolution | 164 | `gepa_reflective` | MeMo trajectory reflection → Pareto bandit config evolution |
| 12 | PhraseBoost Context Trie | 164 | `phrase_boost` | Zero-cost phrase boosting for DDTree screening |
| 13 | Hydra-Aware Adaptive Layer Budget | 165 | `hydra_budget` | Emergent self-repair layer skipping via logit lens |
| 14 | FlashAR Consensus Tri-Mode | 166 | `flashar_consensus` | Dual-path ternary thermal routing for consensus decode |
| 15 | Budget Adaptation | 167 | `budget_adaptation` | Compression-adaptive decode budget scaling |

## Adaptation Pipeline

```
Prompt tokens
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    BIDIRECTIONAL PREFILL                     │
│  Phase A: K/V projections for all positions → cache         │
│  Phase B: Each position attends to K/V[0..prompt_len]       │
│           (no causal mask — code is non-linear)             │
│           reader_lora active                                 │
│           domain_latent injected at layer L/2               │
│           hydra_budget: pre-computed layer importance        │
└─────────────────────┬───────────────────────────────────────┘
                      │ KV cache populated
                      │ first generated token
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      CAUSAL DECODE                           │
│  Standard autoregressive: attend to K/V[0..pos+1]           │
│  writer_lora active (reference swap, zero data movement)    │
│  sparse_mlp: skip dead neurons in w2 @ hidden               │
│  domain_latent still conditioned from prefill               │
│  phrase_boost: trie-boosted phrase screening                │
│  budget_adaptation: PFlash ratio scales DDTree budget       │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   SPECULATIVE TREE (DDTree)                  │
│  elf_sde: SDE noise for diverse expansion                   │
│  tes_loop: RPUCG credit-guided search                       │
│  hydra_budget: adaptive layer skipping (modelless/modeled)  │
│  flashar_consensus: dual-path ternary thermal routing       │
│    Path H (AR/MTP) + Path V (D2F block)                     │
│    PLASMA → HOT → WARM → COLD thermal states               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  GEPA-D CONFIG EVOLUTION                     │
│  MeMo trajectory reflection → config mutation               │
│  Pareto-frontier bandit selection on (quality, cost)        │
│  Evolves: rubric weights, template hints, bandit params     │
│  No gradient updates, no LoRA                               │
└─────────────────────────────────────────────────────────────┘
```

## Technique 1: Bidirectional Prefill

### Problem
Causal attention during prefill means each prompt token only sees preceding tokens. For code, this is wrong — a function body references a struct defined 3,000 tokens earlier. The model needs the whole file at once.

### Solution
Two-phase per-layer processing:

```
Layer L:
  Phase A: for p in 0..prompt_len { K[p], V[p] → cache }     // fill KV
  Phase B: for p in 0..prompt_len {                            // attend to ALL
    Q[p] → attend(Q[p], K[0..prompt_len], V[0..prompt_len])
    → output projection → MLP → hidden state
  }
```

The existing `attention_head` already accepts `t_n: usize` (number of KV positions). Prefill passes `prompt_len`; decode passes `pos + 1`. No API change.

### Implementation

```rust
// transformer.rs — PrefillContext (Plan 025)
pub struct PrefillContext {
    hidden: Vec<f32>,       // [max_prompt_len × n_embd] — multi-layer hidden states
    queries: Vec<f32>,      // [max_prompt_len × n_embd] — fused Phase A Q projections
    residuals: Vec<f32>,    // [max_prompt_len × n_embd] — pre-attention residuals (xr)
    lora_buf: Vec<f32>,     // [lora_rank] — pre-allocated LoRA intermediate
    max_prompt_len: usize,
}

pub fn forward_prefill(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    lora: Option<&LoraAdapter>,
    domain_latent: Option<&DomainLatent>,  // cfg(feature = "domain_latent")
) -> &mut [f32]
```

### Buffer Strategy (Zero Alloc)

| Buffer | Size | Allocation | Reuse |
|--------|------|------------|-------|
| `ForwardContext::x, q, k, v, attn_out, hidden, scores, logits` | Existing | `ForwardContext::new()` (once) | Per-position |
| `PrefillContext::hidden` | `block_size × n_embd` | `PrefillContext::new()` (once) | Between layers |
| `PrefillContext::queries` | `block_size × n_embd` | `PrefillContext::new()` (once) | Fused Phase A → B reuse |
| `PrefillContext::residuals` | `block_size × n_embd` | `PrefillContext::new()` (once) | Fused Phase A → B reuse |
| `PrefillContext::lora_buf` | `[lora_rank]` | `PrefillContext::new()` (once) | Per LoRA application |
| `MultiLayerKVCache` | Existing | Already pre-allocated | K/V storage |

**Single-layer optimization**: `PrefillContext::hidden` unused. Embeddings computed on-the-fly from `wte`/`wpe`. Zero extra memory.

### Performance

| Metric | Value |
|--------|-------|
| Prefill overhead vs causal | ~2× (two passes per layer) |
| Decode throughput impact | Zero (untouched code path) |
| Memory overhead (single-layer) | Zero extra beyond `lora_buf` |
| Memory overhead (multi-layer) | `3 × block_size × n_embd × 4` bytes (hidden + queries + residuals) |

Prefill runs once per request. For 5K prompt → 500 generated tokens, prefill is 1 call, decode is 500. The 2× prefill overhead amortizes to near-zero.

## Technique 2: Modality LoRA Switching

### Problem
Different phases of a task need different behavior. During prefill, the model reads Python; during decode, it writes Rust. One LoRA can't optimize for both.

### Solution
Load two LoRA adapters per domain — `reader_lora` (active during prefill) and `writer_lora` (active during decode). The switch is a reference swap at the prefill→decode boundary.

```rust
// crates/katgpt-core/src/types.rs — LoRA pair (Plan 025)
pub struct LoraPair {
    /// Active during bidirectional prefill (e.g., Python Reader).
    pub reader: Option<LoraAdapter>,
    /// Active during causal decode (e.g., Rust Writer).
    pub writer: Option<LoraAdapter>,
}
```

### LoRA Application — In-Place Delta

```rust
// crates/katgpt-core/src/types.rs
pub struct LoraAdapter {
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub a: Vec<f32>,     // [rank × in_dim]
    pub b: Vec<f32>,     // [out_dim × rank]
    pub alpha: f32,
}
```

Loading methods:
- `LoraAdapter::load(path)` — loads a single-adapter binary file (Plan 008 format: `[LORA 4B][VERSION 4B][RANK 4B][ALPHA 4B][A rows×cols f32][B rows×cols f32]`)
- `LoraAdapter::load_from_bin(path)` — loads a multi-adapter binary file, returns `Vec<LoraAdapter>` (one per target projection). Alpha defaults to `rank * 2`.

```rust
/// Apply LoRA delta in-place: output += (α/r) × B @ (A @ input)
/// `lora_buf` is pre-allocated [rank] intermediate, zero alloc in hot path.
pub fn lora_apply(output: &mut [f32], lora: &LoraAdapter, input: &[f32], lora_buf: &mut [f32])
```

Applied after each Q/K/V/O/MLP projection when a LoRA is active. The delta is fused into the `matmul` output — no separate accumulation buffer.

### Switch Point

```rust
// transformer.rs — generate_with_prefill (Plan 025)

// 1. Bidirectional prefill with reader LoRA
let logits = forward_prefill(ctx, prefill, weights, cache, prompt_tokens, config,
    lora_pair.reader.as_ref(), domain_latent);

// 2. Switch to writer LoRA for decode
// ... reference swap, zero data movement ...

// 3. Causal decode with writer LoRA
let logits = forward_base(ctx, weights, cache, token, pos, config,
    lora_pair.writer.as_ref(), domain_latent);
```

### Performance

| Metric | Value |
|--------|-------|
| LoRA switch cost | Zero (reference swap) |
| LoRA apply overhead | 2 × rank × dim FLOPs per projection |
| Decode throughput impact | Negligible (small rank, fused into matmul) |

## Technique 3: Sparse MLP (TwELL-Inspired)

### Problem
ReLU zeros out ~50% of MLP neurons by definition. With L1 regularization during training, sparsity reaches 90-99%. Dense matmul wastes FLOPs on dead neurons.

### Solution
CPU index-packing sparse matmul for the MLP's second weight matrix (`w2 @ hidden`). Skip dead neurons to reduce FLOPs.

```rust
// crates/katgpt-core/src/types.rs — sparse_matmul (Plan 022)
/// Pack alive neurons (input[c] > 0.0) and multiply only those.
/// Returns alive count for diagnostics.
pub fn sparse_matmul(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    active_indices: &mut [usize],   // pre-allocated [mlp_hidden]
    active_values: &mut [f32],      // pre-allocated [mlp_hidden]
) -> usize
```

### Runtime Auto-Detection

Even with `sparse_mlp` feature enabled, the actual sparsity is checked at runtime:

```rust
// transformer.rs — forward_base MLP section
#[cfg(feature = "sparse_mlp")]
{
    let alive = types::sparse_matmul(
        &mut ctx.x, &layer.w2, &ctx.hidden, n, mlp_hidden,
        &mut ctx.active_indices, &mut ctx.active_values,
    );
    let alive_ratio = alive as f32 / mlp_hidden as f32;
    // Fallback to dense if not sparse enough
    if alive_ratio > (1.0 - config.sparse_threshold) {
        matmul(&mut ctx.x, &layer.w2, &ctx.hidden, n, mlp_hidden);
    }
}
```

- `sparse_threshold = 0.8` (default): use sparse when >80% of neurons are dead
- `0.0`: always use sparse
- `1.0`: never use sparse (always dense)

### Config

```rust
// crates/katgpt-core/src/types.rs
pub struct Config {
    pub sparse_threshold: f32,  // default: 0.8
    // ...
}
```

Feature flags:

| Flag | Description |
|------|-------------|
| `sparse_mlp` | TwELL-inspired sparse MLP matmul |
| `game_domain` | implies `domain_latent` |
| `full` | includes `sparse_mlp`, `ppot`, `domain_latent` |

### Pre-Allocated Buffers

```rust
// transformer.rs — ForwardContext (Plan 022)
#[cfg(feature = "sparse_mlp")]
active_indices: Vec<usize>,   // [mlp_hidden] — allocated once
#[cfg(feature = "sparse_mlp")]
active_values: Vec<f32>,     // [mlp_hidden] — allocated once
```

No `Vec::push` in hot loop. Buffers allocated in `ForwardContext::new()`, reused every forward pass.

### Design Decisions

1. **CPU-Only**: GPU stays dense. Unstructured sparsity causes warp divergence. Structured N:M sparsity (2:4, 4:8) is a separate plan.
2. **Feature-Gated**: Small models (mlp_hidden=64) won't benefit — packing overhead > savings. Users benchmark before enabling.
3. **w2 Only**: `w1 @ x` has dense input (no ReLU yet). `w2 @ hidden` has ReLU'd input → sparse.

### When It Helps

| Config | mlp_hidden | Benefit |
|--------|-----------|---------|
| micro | 64 | ❌ Packing overhead > savings |
| bpe | 128 | ❌ Marginal |
| small_target | 256 | ⚠️ Moderate (needs >80% sparsity) |
| large (real LLM) | 16384 | ✅ Significant at >50% sparsity |

## Technique 4: Domain Latent Injection

### Problem
LoRA adapts weights per domain, but has no mechanism for injecting an explicit domain signal. The model "knows" the domain implicitly through weight deltas, not through a direct conditioning vector.

### Solution
Distill the Free Transformer's mid-layer latent injection into a LoRA-compatible mechanism. Inject a learned domain embedding at layer `L/2` via K/V modulation.

```rust
// crates/katgpt-core/src/types.rs — DomainLatent (Plan 038)
pub struct DomainLatent {
    pub embedding: Vec<f32>,  // [kv_dim]
}

impl DomainLatent {
    pub fn load(path: &Path) -> Result<Self>;    // binary format with BLAKE3 checksum
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn zeros(kv_dim: usize) -> Self;
    pub fn from_vec(embedding: Vec<f32>) -> Self;
}
```

Binary format: `[MAGIC: "DLAT" 4B][VERSION: 1B][KV_DIM: 4B LE][EMBEDDING: kv_dim × f32 LE][BLAKE3: 32B]`

### Forward Pass Modification

At `layer_idx == n_layer / 2`, after K/V projections + LoRA, before cache write:

```rust
// transformer.rs — forward_base (Plan 038)
#[cfg(feature = "domain_latent")]
if layer_idx == config.n_layer / 2
    && let Some(dl) = domain_latent
{
    for i in 0..kvd {
        ctx.k[i] += dl.embedding[i];
        ctx.v[i] += dl.embedding[i];
    }
}
```

Cost: 2 × kv_dim additions at one layer. Zero allocations, zero RNG calls.

### Data Flow

```
Prompt tokens
     │
     ▼
┌─────────────┐
│ Layers 0..  │  Standard causal Transformer (no changes)
│   L/2 - 1   │
└─────┬───────┘
      │ X_{L/2}  [n_embd]
      │
      ├──► K/V projections ──► cache_k, cache_v
      │
      │    domain_embedding [kv_dim]  ◄── DomainLatent.embedding
      │         │
      │         ▼
      │    cache_k += domain_embedding
      │    cache_v += domain_embedding
      │
      ▼
┌─────────────┐
│ Layers L/2  │  Standard Transformer (conditioned on domain)
│   .. L-1    │
└─────┬───────┘
      │
      ▼
   Logits
```

### Why This Design

| Aspect | Free Transformer (Paper) | Our Domain Latent |
|--------|-------------------------|-------------------|
| Z source | VAE encoder (unsupervised) | Domain label (supervised) |
| Z dimension | 65536 (one-hot, H=16 bits) | kv_dim (continuous) |
| Training | From scratch + VAE loss | LoRA fine-tune + embedding |
| Inference | Uniform random Z sampling | Deterministic per domain |
| Requires new base model | Yes | No |

### Works with Bidirectional Prefill

Domain latent is injected in both `forward_base` (decode) and `forward_prefill` (prefill):

```rust
// transformer.rs — forward_prefill (Plan 038)
#[cfg(feature = "domain_latent")]
if layer_idx == config.n_layer / 2
    && let Some(dl) = domain_latent
{
    for i in 0..kvd {
        ctx.k[i] += dl.embedding[i];
        ctx.v[i] += dl.embedding[i];
    }
}
```

Both reader_lora and domain_latent condition the prefill phase. The second half of the model processes domain-informed K/V representations.

### GPU Training Support

`riir-gpu` provides training infrastructure:

```rust
// riir-gpu/src/domain_latent.rs
pub struct GpuDomainLatent {
    // GPU buffers for trainable domain latent (params, grads, m, v)
}

pub fn export_domain_latent(gpu_latent: &GpuDomainLatent, kv_dim: usize) -> DomainLatent;
// Downloads from GPU, saves as .dlat binary
```

`train_bomber.rs` trains LoRA + domain latent together, exporting both.

### Performance

| Metric | Value |
|--------|-------|
| Inference overhead | 2 × kv_dim additions at one layer (< 0.01% FLOPs) |
| Memory overhead | kv_dim × 4 bytes per domain (negligible) |
| Training overhead | One additional embedding vector (negligible vs LoRA) |

## Technique 5: HLA Streaming Attention

### Problem
Standard SDPA attention stores KV cache for all past tokens — O(T) memory per stream. For 30K concurrent game AI streams at 20Hz, this grows unbounded. We need constant-state attention that doesn't degrade with sequence length.

### Solution
Higher-Order Linear Attention (HLA) replaces softmax attention with streaming outer-product updates. State is fixed-size (hd×hd matrix) regardless of sequence length.

Two variants implemented:
- **HLA** (symmetric): maintains SK, CQV, G matrices — O(d²) state per head
- **AHLA** (asymmetric): maintains PKV, E matrices — O(d·dv) state per head

```rust
// hla/kernel.rs — O(1) state update (Plan 057, SIMD-accelerated Plan 060)
pub fn hla_state_update(sk: &mut [f32], q_head: &mut HlaQHeadState, q: &[f32], k: &[f32], v: &[f32], hd: usize, gamma: f32, tmp_k_cqv: &mut [f32], tmp_q_g: &mut [f32])
pub fn hla_readout(q: &[f32], sk: &[f32], q_head: &HlaQHeadState, hd: usize, out: &mut [f32], tmp_u: &mut [f32])
pub fn ahla_step(pkv: &mut [f32], mk: &mut [f32], q_head: &mut AhlaQHeadState, q: &[f32], k: &[f32], v: &[f32], hd: usize, gamma: f32, out: &mut [f32], tmp_r: &mut [f32])
```

### SIMD Acceleration (Plan 060)

All HLA kernels dispatch through `crates/katgpt-core/src/simd.rs` (re-exported via `src/simd.rs`) — runtime NEON/AVX2 detection:

| Operation | NEON Throughput (hd=4) |
|-----------|----------------------|
| hla_update | 16.4M ops/s |
| ahla_step | 18.2M ops/s |
| forward_hla (E2E) | 939K tok/s |
| forward_ahla (E2E) | 1.2M tok/s |

Single ARM core handles 30K CCU @ 20Hz with 9.8× headroom.

### Forward Variants

```rust
// hla/forward.rs — drop-in replacements for forward()
pub fn forward_hla(ctx: &mut ForwardContext, weights: &TransformerWeights, cache: &mut MultiLayerHlaCache, token: usize, pos: usize, config: &Config) -> &mut [f32]
pub fn forward_ahla(ctx: &mut ForwardContext, weights: &TransformerWeights, cache: &mut MultiLayerAhlaCache, token: usize, pos: usize, config: &Config) -> &mut [f32]
```

Same weights, same API — swap `MultiLayerKVCache` for `MultiLayerHlaCache` / `MultiLayerAhlaCache`.

### Plan 059: Inference-Only (Path C Decision)

SDPA→HLA distillation experiment shows KL divergence does NOT converge:
- SDPA→AHLA: KL diverges 4.62→7.43 over 500 steps (lr=1e-4)
- SDPA→HLA: KL oscillates 8.54→8.42, cosine similarity drops
- Root cause: LoRA on QKV adjusts *inputs*, not the *attention mechanism itself*

**HLA is inference-only** — streaming attention without SDPA's quadratic cost. It cannot be trained to approximate SDPA outputs. Use DeltaMemoryState for facts/retrieval.

### Performance

| Metric | Value |
|--------|-------|
| Memory per stream | hd×hd × 4B per head (16 floats for hd=4) |
| vs KV cache | O(1) vs O(T) — no unbounded growth |
| Throughput | 939K tok/s single-core (NEON) |
| 30K CCU @ 20Hz | ✅ 1 core sufficient (9.8× headroom on 8-core) |

## Technique 6: SpectralQuant Calibrated KV Compression

### Problem
TurboQuant uses random rotation + uniform codebook — 5.3× compression but cosine 0.9692. The random rotation doesn't exploit the spectral structure of real KV activations, leaving compression quality on the table.

### Solution
SpectralQuant (Plan 077) replaces random rotation with eigenbasis rotation calibrated from activation statistics, then allocates bits per dimension via water-fill optimization.

```rust
// spectralquant/spectral.rs — eigenbasis calibration
pub fn calibrate_eigenbasis(samples: &[Vec<f32>], head_dim: usize) -> CalibrationResult
pub fn waterfill_bits(eigenvalues: &[f64], total_bits: usize, min_bits: u8, max_bits: Option<u8>) -> Vec<u8>

// spectralquant/nonuniform_quant.rs — Lloyd-Max scalar quantizer
pub struct NonUniformQuantizer { /* eigenvalues, avg_bits, head_dim, per-dim codebooks */ }
pub struct CompressedVector { /* semantic_indices, tail_indices, d_eff, bits metadata */ }
```

### Water-Fill Bit Allocation
Each dimension gets bits proportional to its variance share. High-variance dimensions get more bits (preserving information), low-variance get fewer (saving space).

### Forward Integration

```rust
// transformer.rs — quantized KV forward, generic over QuantizedKVCache trait (crates/katgpt-core/src/traits.rs)
pub fn forward_quantized<C: types::QuantizedKVCache>(ctx, weights, cache: &mut C, ...)
// AttentionMode::SpectralQuant dispatches to spectralquant::forward
```

### Performance (Bench 013)

| Metric | SpectralQuant | TurboQuant |
|--------|:-:|:-:|
| Compression | **9.1×** | 5.3× |
| Cosine similarity | **0.9917** | 0.9692 |
| MaxSim error | **18.90%** | 40.54% |
| Feature flag | `spectral_quant` (default) | `turboquant` (legacy) |

📖 See [`.docs/08_lucebox_techniques.md`](.docs/08_lucebox_techniques.md) for TurboQuant→SpectralQuant migration.

## Technique 7: ELF SDE Noise Injection

### Problem
DDTree builds homogeneous candidate trees — similar prefixes explored, wasting budget on near-duplicates. At γ=1.0, only 14 unique prefixes from 100 rollouts.

### Solution
ELF SDE (Plan 079) injects logit-normal noise during tree expansion, biasing exploration toward t=0 (early tokens) where diversity matters most.

```rust
// speculative/types.rs — SDE configuration (default-on)
pub struct SdeConfig {
    pub gamma: f32,           // noise re-injection scale
    pub confidence_floor: f32, // minimum logit magnitude for noise application
    pub preserve_top1: bool,  // keep highest-prob token clean
}

// speculative/dd_tree.rs — noise-augmented tree building
pub fn build_dd_tree_sde(...)    // SDE-augmented expansion
pub fn build_dd_tree_balanced_sde(...) // balanced + SDE
```

### Logit-Normal Schedule
Noise concentrates near t=0 via logit-normal distribution: 2.2× concentration at early positions. Later positions receive diminishing noise, preserving coherent continuations.

### Width Scaling (PTRM, Plan 083)

```rust
// speculative/dd_tree.rs — width scaling via best-of-K
pub struct WidthScaleConfig {
    pub k_rollouts: usize,                // K parallel rollouts
    pub selection: WidthSelectionMode,     // how to pick winner
}
pub enum WidthSelectionMode { BestQ, MostFrequent, Top1Converged }
pub fn best_of_k_rollouts(marginals, config, screener, sde_config, width_config, base_seed) -> Vec<usize>

// crates/katgpt-core/src/types.rs — Config convenience fields
pub struct Config {
    pub width_rollouts: usize,            // default: 1 (disabled)
    pub early_stop_threshold: f32,        // default: 0.0 (disabled)
}
```

PTRM proves width >> depth for small models. K=64 rollouts + `EarlyStopGate` depth-aware pruning.

### Performance (Bench 012)

| Metric | ELF SDE | Baseline |
|--------|:-:|:-:|
| Unique prefixes (γ=1.0) | **145** | 14 |
| Diversity gain | **10-22×** | — |
| Overhead | 3.2µs (<3%) | — |

## Technique 8: CNA Steering (Contrastive Neuron Attribution)

### Problem
Residual-stream steering (CAA) uses linear probes that ignore layer structure, achieving <0.60 cosine quality. No attribution of *which* MLP neurons matter for a given behavior.

### Solution
CNA Steering (Plan 087) discovers sparse MLP circuits via contrastive attribution, then modulates only the discovered neurons at runtime.

```rust
// pruners/cna.rs — circuit discovery + modulation
pub struct CnaNeuron { pub layer: usize, pub index: usize, pub delta: f32 }
pub struct CnaCircuit {
    pub neurons: Vec<CnaNeuron>,
    pub universal_excluded: Vec<(usize, usize)>,
    pub n_positive: usize,
    pub n_negative: usize,
    pub total_mlp_activations: usize,
}
pub struct CnaDiscoveryConfig { pub top_pct: f32, pub universal_threshold: f32, pub late_layer_fraction: f32 }
pub struct CnaModulator { pub circuit: CnaCircuit, pub multiplier: f32 }

pub fn cna_discover(positive: &[(usize, &[f32])], negative: &[(usize, &[f32])], n_layers: usize, mlp_hidden: usize, config: &CnaDiscoveryConfig) -> CnaCircuit
pub fn cna_modulate(hidden: &mut [f32], layer_idx: usize, modulator: &CnaModulator)
pub fn detect_universal_neurons(diverse_activations: &[Vec<(usize, Vec<f32>)>], n_layers: usize, mlp_hidden: usize, threshold: f32) -> Vec<(usize, usize)>

pub struct CnaScreeningPruner { /* composable with BanditPruner */ }
```

### Discovery → Modulation Pipeline
1. **Discovery**: Contrast positive vs negative activations → top-K attributions (~10µs/pair)
2. **Modulation**: Scale only discovered neurons — O(K) sparse forward hook (163ns for K=50)
3. **Universal Filtering**: `detect_universal_neurons` removes neurons that fire in ≥80% of diverse prompts
4. **Pruner Integration**: `CnaScreeningPruner` composes with existing `BanditPruner`

### Performance (Bench 015, GOAT proved)

| Metric | Value |
|--------|-------|
| Discovery speed | ~10µs/pair |
| Modulation speed | 163ns (K=50) |
| Cosine quality | **1.0** at all strengths |
| Late-layer concentration | 100% |
| Quality vs CAA | **1.0** vs <0.60 |

## Technique 9: Deep Manifold + Federation

### Problem
BanditPruner Q-values implicitly encode residual distance, but without explicit fixed-point structure. For multi-expert setups, experts train independently with no coupling.

### Solution
Deep Manifold (Plan 085) makes residual distance explicit via L2/KL scoring. Federation adds symmetric KL boundary alignment between experts — no data exchange needed.

```rust
// pruners/manifold_residual.rs — fixed-point residual scoring
pub trait ManifoldResidual: Send + Sync {
    fn residual(&self, candidate: &[f32], base: &[f32]) -> f32;
    fn is_converged(&self, residual: f32, tolerance: f32) -> bool;
    fn per_position_residual(&self, candidate: &[f32], base: &[f32]) -> Vec<f32>;
}
pub struct L2ResidualScorer { pub tolerance: f32 }
pub struct KlResidualScorer { pub tolerance: f32 }
pub struct ResidualRelevanceScorer<R: ManifoldResidual> { pub residual_scorer: R, pub residual_weight: f32 }

// pruners/boundary_alignment.rs — federated KL coupling
pub trait BoundaryAlignment: Send + Sync {
    fn kl_divergence(&self, local: &[f32], ensemble: &[f32]) -> f32;
    fn coupling_weight(&self, domain: &str, neighbors: &[&str]) -> f32;
    fn boundary_penalty(&self, local: &[f32], ensemble: &[f32], lambda: f32) -> f32;
}
pub struct KlBoundaryAligner { pub epsilon: f32 }
```

### Per-Position Hotspot Analysis
ResidualRelevanceScorer produces per-position scores: high residual = prediction far from fixed-point manifold. BanditPruner can use this to prioritize high-uncertainty positions.

### Federation: No Privacy Concern
Experts never exchange raw data — only KL boundary statistics. `KlBoundaryAligner` computes symmetric KL divergence between expert boundaries and adjusts toward consensus.

### Performance (GOAT 6/6)

| Metric | Value |
|--------|-------|
| GOAT tests | 6/6 passed |
| L2 residual | O(n) SIMD-able |
| KL coupling | No data exchange |
| Default-on | ✅ |

## Technique 10: SimpleTES Loop (RPUCG)

### Problem
Greedy DDTree expansion wastes budget on low-value branches. No trajectory-level credit assignment — each node scored independently.

### Solution
SimpleTES (Plan 086) implements RPUCG (Rollout Policy Using Credit-Guided search): full C×L×K budget loop with trajectory credit assignment bridging to G-Zero Phase 2.

```rust
// pruners/tes_loop.rs — SimpleTES loop
pub trait TesLoop: Send + Sync {
    fn config(&self) -> &TesConfig;
    fn budget(&self) -> usize;
    fn select_inspirations(&self, history: &[TesNode], count: usize) -> Vec<usize>;
    fn update_propagated_values(&self, history: &mut [TesNode], gamma: f32);
    fn rpucg_score(&self, node: &TesNode, total_visits: usize, lambda: f32) -> f32;
    fn select_rpucg(&self, history: &[TesNode], count: usize, lambda: f32) -> Vec<usize>;
}
pub struct SimpleTesLoop<E: BanditEnv> {
    config: TesConfig,       // (C, L, K, Φ) parameters
    env: E,                  // bandit environment
    history: Vec<TesNode>,   // all evaluated nodes
    best_score: f32,
    best_idx: usize,
}

// speculative/types.rs — trajectory credit
pub struct TrajectoryCredit {
    pub num_trajectories: usize,
    pub best_score: f32,
    pub worst_score: f32,
    pub best_trajectory_idx: usize,
    pub worst_trajectory_idx: usize,
}
pub fn TrajectoryCredit::from_trajectory_scores(scores: &[(usize, f32)]) -> Self
```

### Budget Scaling: Wide > Narrow
Wide budget (24×5×8) achieves 0.9988 quality vs narrow (2×8×30) at 0.8266 (spread=0.17). RPUCG beats greedy: 42.8% vs 10.6% wins.

### Bridge to G-Zero Phase 2
`TrajectoryCredit` provides the credit signal needed for GRPO Proposer + length-normalized DPO Generator in G-Zero's model-based self-play.

### Performance (Bench 016+017, GOAT 8/8)

| Metric | Value |
|--------|-------|
| GOAT tests | 8/8 passed |
| RPUCG vs greedy | **42.8%** vs 10.6% wins |
| Wide vs narrow | 0.9988 vs 0.8266 |
| Feature flag | `tes_loop` (requires `bandit`) |

## Technique 11: GEPA-D Reflective Config Evolution

### Problem
System-level configuration (rubric weights, template hints, bandit parameters) is hand-tuned and static. No feedback loop from inference trajectories back to config.

### Solution
GEPA-D (Plan 164, Research 146) evolves system-level configuration from MeMo trajectory reflection using Pareto-frontier bandit selection. Mutates rubric weights, template hints, and bandit params — no gradient updates, no LoRA.

```rust
// gepa/reflective.rs
pub struct GepaReflectiveConfig {
    pub bandit: BanditConfig,
    pub memo_reflections: bool,
    pub pareto_frontier_size: usize,
}

pub fn gepa_evolve_config(
    trajectories: &[MeMoTrajectory],
    config: &mut SystemConfig,
    bandit: &mut ParetoBandit,
) -> Vec<ConfigMutation> { ... }
```

### Design Decisions
Operates entirely in config-space, not weight-space. Pareto-frontier bandit selects on (quality, cost) multi-objective. Composes with all other techniques since it only mutates configuration, not model internals.

### Performance

| Metric | Value |
|--------|-------|
| Feature flag | `gepa_reflective` (default-on, requires `bandit`, `memo_reflections`) |
| Gradient updates | None |
| LoRA required | No |

## Technique 12: PhraseBoost Context Trie

### Problem
DDTree screening prunes candidates without domain-specific phrase awareness. Important phrase matches may be discarded early.

### Solution
PhraseBoost (Plan 164, Research 147) adds a context trie phrase boosting layer for DDTree. Zero training cost — phrases are provided at call site. `PhraseBoostPruner` wraps any `ScreeningPruner` and adds normalized boost for phrase matches.

```rust
// pruners/phrase_boost.rs
pub struct PhraseBoostPruner<P: ScreeningPruner> {
    inner: P,
    phrase_trie: ContextTrie,
    boost_weight: f32,
}

impl<P: ScreeningPruner> ScreeningPruner for PhraseBoostPruner<P> {
    fn prune(&self, candidates: &[Candidate], context: &Context) -> Vec<Candidate> {
        // Apply inner pruner, then boost scores for phrase matches
    }
}
```

### Zero Training Cost
Phrases are injected at call site via `ContextTrie` — no pre-training, no fine-tuning. Wraps any existing `ScreeningPruner` as a decorator.

### Performance

| Metric | Value |
|--------|-------|
| Feature flag | `phrase_boost` (default-on) |
| Training cost | Zero |
| Wraps | Any `ScreeningPruner` |

## Technique 13: Hydra-Aware Adaptive Layer Budget

### Problem
All transformer layers execute unconditionally. Some layers contribute disproportionately more to output quality; skipping low-importance layers saves compute without quality loss.

### Solution
Hydra Budget (Plan 165, Research 148) implements emergent self-repair layer skipping inspired by the Hydra Effect (arXiv:2307.15771). Two modes: modelless (pre-computed layer importance profiles) and model-based (per-layer logit lens scoring).

```rust
// hydra/budget.rs
pub struct HydraBudgetConfig {
    pub mode: HydraMode,
    pub skip_threshold: f32,
}

pub enum HydraMode {
    Modelless { importance_profile: Vec<f32> },
    ModelBased { logit_lens: LogitLensConfig },
}

pub fn hydra_layer_decision(
    layer_idx: usize,
    hidden: &Tensor,
    config: &HydraBudgetConfig,
) -> bool { ... }
```

### Modelless vs Model-Based
Modelless uses pre-computed per-layer importance profiles (no runtime overhead). Model-based uses per-layer logit lens scoring for dynamic skip decisions (higher quality, marginal cost).

### Performance

| Metric | Value |
|--------|-------|
| Feature flag | `hydra_budget` (default-on) |
| Modes | Modelless (pre-computed), Model-based (logit lens) |
| Inspiration | Hydra Effect (arXiv:2307.15771) |

## Technique 14: FlashAR Consensus Tri-Mode

### Problem
Consensus decoding offers a single verification path. Complex speculative trees benefit from dual-path routing with thermal state awareness.

### Solution
FlashAR Consensus (Plan 166, Research 149) implements dual-path ternary thermal routing for consensus tri-mode decoding. Path H (AR/MTP) handles horizontal autoregressive + multi-token prediction. Path V (D2F block) handles vertical drafter-to-final verification. PLASMA/HOT/WARM/COLD thermal routing selects verification aggressiveness.

```rust
// flashar/consensus.rs
pub struct FlashArConfig {
    pub tri_mode: bool,
    pub plasma_path: PathBuf,
}

pub enum ThermalState {
    Plasma,  // Full verification, max quality
    Hot,     // Aggressive verification
    Warm,    // Moderate verification
    Cold,    // Minimal verification, max speed
}

pub fn flashar_route(
    token: &Token,
    thermal: &ThermalState,
    config: &FlashArConfig,
) -> ConsensusPath { ... }
```

### Thermal Routing
PLASMA → HOT → WARM → COLD transitions are driven by consensus confidence. High confidence → COLD (fast path). Low confidence → PLASMA/HOT (thorough verification).

### Performance

| Metric | Value |
|--------|-------|
| Feature flag | `flashar_consensus` (default-on, requires `tri_mode`, `plasma_path`) |
| Paths | Path H (AR/MTP), Path V (D2F block) |
| Thermal states | PLASMA / HOT / WARM / COLD |

## Technique 15: Budget Adaptation

### Problem
Fixed decode budget wastes compute on simple prompts (over-searching) and under-serves complex prompts (under-searching).

### Solution
Budget Adaptation (Plan 167, Research R050) implements compression-adaptive decode budget: PFlash ratio scales DDTree budget within [0.5×, 2.0×]. Simple prompts get less search (0.5×), complex prompts get more (2.0×).

```rust
// budget/adaptation.rs
pub struct BudgetAdaptationConfig {
    pub min_scale: f32,  // 0.5
    pub max_scale: f32,  // 2.0
}

pub fn adapt_decode_budget(
    base_budget: usize,
    pflash_ratio: f32,
    config: &BudgetAdaptationConfig,
) -> usize {
    let scale = config.min_scale
        + (config.max_scale - config.min_scale) * pflash_ratio;
    (base_budget as f32 * scale) as usize
}
```

### PFlash Ratio Signal
PFlash ratio measures compression complexity of the prompt. High ratio → complex prompt → more DDTree budget. Low ratio → simple prompt → less budget. The [0.5×, 2.0×] range prevents pathological under/over-allocation.

### Performance

| Metric | Value |
|--------|-------|
| Feature flag | `budget_adaptation` (default-on) |
| Budget range | [0.5×, 2.0×] base |
| Signal | PFlash compression ratio |

## Interaction Matrix

All sixteen techniques compose without conflicts:

| Technique | Affects Prefill | Affects Decode | Feature Flag |
|-----------|:-:|:-:|-------------|
| Bidirectional Prefill | ✅ full attention | — | — |
| LoRA Switching | ✅ reader_lora | ✅ writer_lora | — |
| Sparse MLP | ✅ (if enabled) | ✅ (if enabled) | `sparse_mlp` |
| Domain Latent | ✅ K/V at L/2 | ✅ K/V at L/2 | `domain_latent` |
| HLA Streaming | — | ✅ replaces KV cache | `hla_attention` |
| OCTOPUS | — | ✅ KV compression | `octopus` (default, primary) |
| SpectralQuant | — | ✅ KV compression | `spectral_quant` (default, secondary) |
| ELF SDE | — | ✅ tree diversity | `elf_sde` (default) |
| CNA Steering | — | ✅ neuron modulation | `cna_steering` (default) |
| Deep Manifold | — | ✅ residual scoring | `deep_manifold` (default) |
| Federation | — | ✅ expert alignment | `federation` (default) |
| SimpleTES | — | ✅ credit-guided search | `tes_loop` |
| GEPA-D Config Evolution | — | ✅ config mutation | `gepa_reflective` (default, requires `bandit`, `memo_reflections`) |
| PhraseBoost | — | ✅ phrase screening | `phrase_boost` (default) |
| Hydra Budget | ✅ layer importance | ✅ layer skipping | `hydra_budget` (default) |
| FlashAR Consensus | — | ✅ thermal routing | `flashar_consensus` (default, requires `tri_mode`, `plasma_path`) |
| Budget Adaptation | — | ✅ budget scaling | `budget_adaptation` (default) |
| ManifoldPruner Soft Validity | — | ✅ soft sigmoid pruning | `manifold_pruner` (opt-in, Plan 234) |

All are additive and backward-compatible. Standard `forward()` with no features works exactly as before.

## Key References

- [ZAYA1-VL-8B Technical Report](https://arxiv.org/abs/2504.02268) — Bidirectional prefix attention, token-specific LoRAs (Plan 025)
- [Sakana TwELL](https://sakana.ai/twell/) — Tile-wise ELLPACK sparse format (Plan 022 inspiration, GPU-specific; we use CPU index-packing)
- [The Free Transformer](https://arxiv.org/abs/2503.23153) — Mid-layer latent injection via K/V modulation (Plan 038)
- [Higher-Order Linear Attention](https://arxiv.org/abs/2504.13764) — O(1) streaming attention via outer-product state (Plan 057)
- [TurboQuant](https://arxiv.org/abs/2504.19874) — KV cache compression via learned codebooks (Plan 043, legacy baseline)
- [OCTOPUS](https://arxiv.org/abs/2605.21226) — Octahedral triplet KV compression, data-oblivious, -22% to -49% MSE vs SQ (Plan 099, primary default)
- [SpectralQuant](https://arxiv.org/pdf/2504.19874) — Calibrated eigenbasis + water-fill bit allocation (Plan 077, secondary)
- [Embedded Language Flows](https://arxiv.org/abs/2406.09970) — SDE noise injection for path diversity (Plan 079)
- [PTRM](https://arxiv.org/abs/2605.19943) — Width >> depth for small models (Plan 083)
- [Contrastive Neuron Attribution](https://arxiv.org/pdf/2605.12290) — Sparse circuit discovery + modulation (Plan 087)
- [Deep Manifold Part 2](https://arxiv.org/pdf/2512.06563) — Fixed-point boundary conditions, federation (Plan 085)
- [SimpleTES](https://arxiv.org/abs/2604.19341) — Credit-guided trajectory search (Plan 086)
- [G-Zero](https://arxiv.org/pdf/2605.09959) — Self-play distillation, Hint-δ, GRPO+DPO (Plan 049)
- [GEPA-D](https://arxiv.org/abs/2406.14289) — Reflective config evolution via MeMo trajectory reflection (Plan 164)
- [Hydra Effect](https://arxiv.org/abs/2307.15771) — Emergent self-repair in layer skipping (Plan 165)
- [FlashAR](https://arxiv.org/abs/2605.15730) — Dual-path ternary thermal consensus routing (Plan 166)