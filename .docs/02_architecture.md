# katgpt-rs: Core Architecture

## Overview
The transformer is a from-scratch GPT-2 style implementation. No frameworks — weights are `Vec<f32>`, ops are hand-written matmul/softmax/rmsnorm. Supports multi-layer, grouped-query attention (GQA), and zero-allocation inference.

## Config (`crates/katgpt-core/src/types.rs`, re-exported via `src/types.rs`)
```rust
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,     // max sequence length
    pub n_embd: usize,         // embedding dimension
    pub n_head: usize,         // number of attention Q heads
    pub head_dim: usize,       // dimension per head (n_embd / n_head)
    pub mlp_hidden: usize,     // MLP intermediate size
    pub n_layer: usize,        // number of transformer layers
    pub n_kv_head: usize,      // number of K/V heads (≤ n_head for GQA)
    pub bos_token: usize,
    pub temperature: f32,
    pub draft_lookahead: usize,
    pub tree_budget: usize,
    pub parallel_threshold: usize,  // skip rayon if n_embd ≤ this
    pub lora_rank: usize,           // LoRA adapter rank (Plan 008)
    pub lora_alpha: f32,            // LoRA scaling factor
    pub lora_dropout: f32,          // LoRA dropout probability
    pub lora_targets: Vec<String>,  // which projections to apply LoRA
    pub screening_threshold: f32,   // hard-trim cutoff for ScreeningPruner (Plan 021)
    pub sparse_threshold: f32,      // use sparse_mlp when alive ratio ≤ this (Plan 022)
    pub early_exit_patience: usize, // AutoTTS early exit patience (Plan 026)
    pub early_exit_gap: f32,        // AutoTTS early exit confidence gap
    // MTP Drafter thresholds (Plan 055: Gemma 4 MTP)
    pub mtp_activation_threshold: usize,    // enable MTP when n_embd >= this
    pub mtp_cluster_vocab_threshold: usize, // enable cluster LM head when vocab_size >= this
    pub mtp_shared_kv_prompt_threshold: usize, // enable shared KV for prompt when pos >= this
    pub mtp_cluster_size: usize,            // cluster size for round-robin vocab mapping
    pub mtp_min_output_tokens: usize,       // skip MTP when remaining tokens < threshold (Plan 117 T15)
    pub mtp_cluster_topk: usize,            // compute logits for top-K clusters (Plan 117 T22)
    // HLA Attention (Plan 057: Higher-order Linear Attention)
    pub hla_mode: HlaMode,                  // Standard, Hla, Ahla
    pub hla_normalize: bool,                // normalize HLA output
    pub hla_decay: f32,                     // decay factor for HLA state
    // D2F Discrete Diffusion Forcing (Plan 066)
    pub mask_token: usize,                  // mask token ID for dLLM
    pub attention_mode: AttentionMode,      // Causal, Bidirectional, BlockCausal, SpKv
    // SP-KV self-pruned KV attention (Plan 070)
    pub sp_kv_window: usize,               // sliding window size for SP-KV
    pub sp_kv_threshold: f32,              // gate threshold for SP-KV utility predictor
    pub sp_kv_predictor_hidden: usize,     // hidden dim for utility predictor MLP
    pub sp_kv_predictor_lr_mult: f32,      // learning rate multiplier for predictor
    // Gemma 2 architecture (Plan 087)
    pub model_arch: ModelArchitecture,      // Generic, Gemma2
    pub rms_norm_eps: f64,                  // epsilon for RMSNorm (1e-5 default, 1e-6 for Gemma2)
    pub rms_norm_offset: bool,              // add offset in RMSNorm (Gemma2: true)
    pub tied_embeddings: bool,              // share wte and lm_head (Gemma2: true)
    pub use_rope: bool,                     // rotary position embeddings (Gemma2: true)
    pub rope_theta: f32,                    // RoPE base frequency
    pub post_norm: bool,                    // post-attention norm (Gemma2: true)
    pub attn_logit_softcapping: f32,        // cap attention logits (Gemma2: 50.0)
    pub final_logit_softcapping: f32,       // cap final logits (Gemma2: 30.0)
    pub weight_dtype: WeightDtype,          // F32, F16, BF16
    // PTRM width scaling (Plan 083)
    pub width_rollouts: usize,              // number of parallel rollouts
    pub early_stop_threshold: f32,          // stop early when reward exceeds this
    // EqR Convergence Selection (Plan 119)
    pub convergence_selector: ConvergenceSelector, // rollout selection strategy
    // D2F block size for discrete diffusion forcing
    pub d2f_block_size: usize,              // block size for D2F diffusion
    // MLS Multi-Layer Sum aggregation (Plan 104: Research 68)
    pub mls_layers: usize,                  // number of last layers to aggregate (0 = disabled)
    // LT2 Looped Inference Pipeline (Plan 108, Research 73)
    pub loop_mode: LoopMode,                // None or WeightShared { loop_count }
    pub hybrid_pattern: HybridPattern,      // Uniform, Interleave, Bookend
    pub loop_min: usize,                    // elastic override floor (Issue 035, Research 273 — ELT Any-Time)
    pub loop_max: usize,                    // elastic override ceiling (0 = derive from loop_mode)
    pub gated_attn: bool,                   // whether to use SDPA output gate
    // Parallax Attention (Plan 135)
    pub parallax_gate_scale: f32,            // covariance correction gate scale (0.0=disabled)
    pub parallax_zero_init: bool,            // whether W_R starts zeroed
    // Emotion Vector (Plan 162, Research 144)
    pub emotion_desperation_threshold: f32,  // desperation threshold for session flagging
    // Hydra Adaptive Layer Budget (Research 148, Plan 165)
    #[cfg(feature = "hydra_budget")]
    pub hydra_profiles: Vec<HydraLayerProfile>, // per-layer importance profiles (empty = disabled)
    // DeltaNet Inference (Plan 182)
    #[cfg(feature = "deltanet_inference")]
    pub layer_types: Vec<DeltaNetLayerType>,     // per-layer type: Attention vs DeltaNet
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_conv_kernel_size: usize,        // depthwise conv kernel size (typically 4)
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_state_dim: usize,               // recurrence state dim per head
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_linear_head_dim: usize,         // linear attention key/value head dim
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_linear_n_heads: usize,          // number of linear attention key heads
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_linear_n_value_heads: usize,    // number of linear attention value heads
    // RiM Reasoning Buffer Slots (Plan 172, Research 192)
    #[cfg(feature = "rim_slots")]
    pub rim_block_count: usize,                 // number of reasoning buffer blocks (K in RiM paper), 0 = disabled
    #[cfg(feature = "rim_slots")]
    pub rim_tokens_per_block: usize,            // tokens per buffer block (M in RiM paper), default 2
    #[cfg(feature = "rim_slots")]
    pub rim_buffer_token: usize,                // token ID used for buffer positions (default: bos_token)
    // Wall Attention (Plan 173)
    #[cfg(feature = "wall_attention")]
    pub wall_config: Option<WallConfig>,        // None = use RoPE/fallback
}
```
- All configs constructed via factory methods: `Config::micro()`, `Config::micro_lora()`, `Config::draft()`, `Config::game()`, `Config::game_go()`, `Config::gemma2_2b()`, `Config::micro_dllm()`, `Config::bpe()`, `Config::bpe_draft()`, `Config::small_target()`, `Config::gqa_draft()`, `Config::qwen_deltanet()` (requires `deltanet_inference` feature)
- Validation: `n_head % n_kv_head == 0`, `n_embd == n_head * head_dim`
- `kv_dim()` helper returns `n_kv_head * head_dim`

### Key Enums (`crates/katgpt-core/src/types.rs`)

```rust
#[repr(u8)]
pub enum ConvergenceSelector {
    BestQ,          // Highest cumulative relevance (default)
    MajorityVote,   // Most common path across rollouts (mode@K)
    Top1Converged,  // Smallest residual ∥p_{d+1} − p_d∥ (EqR proxy)
    BtRank,         // Pairwise Bradley-Terry ranking (requires `bt_rank` feature)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DashAttnConfig {
    pub chunk_size: usize,          // tile size for chunked attention
    pub alpha: f32,                 // mixing coefficient
    pub scaling_factor: f32,        // attention scale override
    pub sigma: f32,                 // smoothing parameter
    pub estimate_diagonal: bool,    // whether to estimate diagonal terms
}

#[repr(u8)]
pub enum DeltaRoutingMode {
    Off,           // No delta routing (standard layer-by-layer)
    DeltaBlock,    // Route accumulated block deltas
    DeltaAttnRes,  // Route attention residual deltas
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeltaRoutingConfig {
    pub mode: DeltaRoutingMode,     // routing mode
    pub block_size: usize,          // layers per block (default 4)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopMode {
    #[default]
    None,                                    // standard single-pass
    WeightShared { loop_count: usize },      // T-pass weight-shared loop
    TrainingFree,                            // ODE-refined sub-stepping, no extra params (Plan 136)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HybridPattern {
    #[default]
    Uniform,                                  // all layers use same attention
    Interleave { full_ratio: usize },         // every Nth layer is full SDPA
    Bookend,                                  // first+last layers are full SDPA
}

#[derive(Clone, Debug)]
pub struct SdpaOutputGate {
    pub w_gate: Vec<f32>,    // [n_embd] sigmoid gate weights (zero-init)
}
// forward(&self, attn_out, n_embd) — applies sigmoid gate after SDPA

#[derive(Clone, Debug)]
pub struct ResidualGate {
    pub gates: Vec<f32>,     // [loop_count] per-loop learned gate ρ_τ (zero-init)
}
// new(loop_count, n_embd) — creates zero-init gates

// Feature-gated: `sr2am_configurator`
pub enum PlanningDecision {
    PlanNew,                              // reset tree, full budget (high uncertainty)
    PlanExtend,                           // keep tree, extend depth (moderate uncertainty)
    PlanSkip,                             // skip tree search, direct sample (low uncertainty)
    SpecHop { k: usize },                 // k speculative threads (Plan 131)
    #[cfg(feature = "sia_feedback")]
    HarnessUpdate,                        // AbsorbCompress promote + HotSwapPruner reload (Plan 163 T5)
    #[cfg(feature = "sia_feedback")]
    WeightUpdate,                         // trigger riir-gpu training step (Plan 163 T6)
}

// Feature-gated: `sr2am_configurator`
pub struct ConfiguratorContext {
    pub domain: usize,           // domain index from bandit infrastructure
    pub entropy_bin: usize,      // coarse entropy bin: floor(entropy * 10.0), 0..9
    pub desperation_bin: usize,  // coarse desperation bin (Plan 162 T11): 0..9
}
```


```rust
#[repr(u8)]
pub enum HlaMode {
    Standard,  // SDPA with KV cache (default)
    Hla,       // Symmetric second-order linear attention — O(1) per-token memory
    Ahla,      // Asymmetric second-order linear attention — lower state cost
}

#[repr(u8)]
pub enum AttentionMode {
    Causal,       // Standard autoregressive (default)
    Bidirectional, // Attend to ALL positions — dLLM masked prediction
    BlockCausal,  // Bidirectional within block, causal across blocks — D2F student
    SpKv,         // Self-pruned key-value attention with learned utility (Plan 070)
    SpKvQuant,    // SP-KV + Quantized KV fusion (Plan 070 Phase 3, Task T12)
    DashAttn,     // Chunked linear attention (Research 077, DashAttnConfig)
}

#[repr(u8)]
pub enum ModelArchitecture {
    Generic,  // Default GPT-2 style
    Gemma2,   // Gemma 2 architecture (Plan 087)
    Llama,    // Llama architecture
    #[cfg(feature = "deltanet_inference")]
    QwenDeltaNet, // Hybrid DeltaNet/Attention (Plan 182)
}

#[repr(u8)]
pub enum WeightDtype {
    F32,   // Full precision (default)
    F16,   // Half precision
    BF16,  // Bfloat16
}

// --- Attention projection variant (SharedKV cache reduction) ---
pub enum AttentionProjection {
    Full,      // Standard Q, K, V (3 projections, full KV cache)
    SharedKV,  // Q-K=V: K and V share projection (2 projections, K-only cache). 50% KV cache reduction.
}

pub enum CacheLayout {
    KV,  // Store both K and V (standard)
    K,   // Store K only, V = K at read (SharedKV)
}

pub enum RetrievalHeadRole {
    Local,      // Local head — sliding window + sink tokens only, no full KV scan
    Retrieval,  // Retrieval head — low-dim projection + dynamic top-p token selection
}

// --- Training-Free Loop sub-step types (Plan 136) ---
pub enum SubStepStrategy {
    DampedEuler,                              // x ← x + (1/K)·(y − x)
    KStageRK { beta: f32 },                   // x ← β·y + (1−β)·x
}

pub enum IterationMode {
    Block,   // Apply the full window [a, b] as one block per sub-step
    Layer,   // Apply each layer in the window individually per sub-step
}

pub enum CacheStrategy {
    Last,   // Use the final loop iteration's hidden state for KV cache
    First,  // Use the pre-loop hidden state for KV cache (first iteration)
}

// --- Data Gate types (Plan 141, feature `data_gate`) ---
pub enum TaskType {
    CodeIO,       // Python code output prediction
    DslExpr,      // DSL expression evaluation
    GameAction,   // Game action (Bomber, Go, FFT, Monopoly)
    OpenEnded,    // Open-ended generation
}

pub enum GateDecision {
    Admit,           // Task passes the gate — admitted to training pool
    Reject(String),  // Task rejected with reason
}
```

### Wall Attention (`crates/katgpt-core/src/types.rs`, Plan 173)

Feature-gated behind `wall_attention`.

```rust
pub struct WallConfig {
    pub gate_bias: f32,            // gate bias initialization. Default 6.0 = open gate (vanilla attention)
    pub gate_max: f32,             // maximum gate log-sigmoid clamp. Default 0.87
    pub use_key_projected: bool,   // derive gate from K projection (preferred: zero KV cache overhead)
}
```

### RiM Reasoning Buffer Slots (`crates/katgpt-core/src/types.rs`, Plan 172, Research 192)

Feature-gated behind `rim_slots`. Fields on `Config`:
- `rim_block_count: usize` — number of reasoning buffer blocks (K in RiM paper), 0 = disabled
- `rim_tokens_per_block: usize` — tokens per buffer block (M), default 2
- `rim_buffer_token: usize` — token ID used for buffer positions (default: bos_token)

### Hydra Types (`crates/katgpt-core/src/types.rs`, Research 148, Plan 165)

Feature-gated behind `hydra_budget`.

```rust
/// Per-layer Hydra profile entry (modelless mode).
pub struct HydraLayerProfile {
    pub mean_de: f32,           // mean absolute direct effect on top-token logit
    pub backup_frequency: f32,  // fraction of prompts where this layer is a Hydra backup
    pub is_erasure: bool,       // whether this layer acts as erasure (mean DE < 0 for MLP)
}

/// Hydra budget configuration.
pub struct HydraBudgetConfig {
    pub skip_threshold: f32,        // skip layers with |DE| below this (default 0.01)
    pub cumulative_threshold: f32,  // early-terminate when cumulative DE reaches this fraction (default 0.95)
    pub modelless: bool,            // use pre-computed profiles (true) vs logit lens scoring (false)
    pub skip_erasure_draft: bool,   // skip erasure MLPs during draft stage
}
```

### Training-Free Loop (`crates/katgpt-core/src/types.rs`, Plan 136)

Used when `Config.loop_mode = TrainingFree`.

```rust
pub struct TrainingFreeLoopConfig {
    pub window_start: usize,        // start of the loop window (inclusive layer index)
    pub window_end: usize,          // end of the loop window (inclusive layer index)
    pub loop_count: usize,          // number of loop iterations (K in the paper)
    pub strategy: SubStepStrategy,  // sub-step integration strategy
    pub iteration_mode: IterationMode, // block vs layer-wise
    pub cache_strategy: CacheStrategy, // KV cache write strategy
}
```

### Problem Mutator (`crates/katgpt-core/src/traits.rs`, feature `problem_mutator`)

FrontierSmith closed→open problem synthesis: mutate game configs into harder variants.

```rust
pub struct GameConfig {
    pub grid_size: u32,          // board size (e.g., 9 for 9x9)
    pub opponent_count: u32,     // number of opponents/NPCs
    pub max_steps: u32,          // maximum steps per episode
    pub survival_weight: f32,    // weight for survival objective
    pub kill_weight: f32,        // weight for kill/objective
}

pub enum MutationKind {
    GoalReweight,       // shift objective weights
    ConstrainOutputs,   // reduce action space or add constraints
    GeneralizeInputs,   // vary input parameters
}

pub struct MutantConfig {
    pub difficulty_delta: f32,       // estimated difficulty increase over seed
    pub mutation_kind: MutationKind, // which mutation strategy was applied
    pub description: String,        // human-readable description
}
```

### Data Gate (`crates/katgpt-core/src/types.rs`, feature `data_gate`)

Task-level admission gate for self-play training pool.

```rust
pub struct ProposerTask {
    pub id: usize,                      // task identifier for diagnostics
    pub query: String,                  // the problem/query text
    pub program: Option<String>,        // optional code or DSL expression to execute
    pub program_input: Option<String>,  // optional input for the program
    pub task_type: TaskType,            // task type discriminator
}
```

### InferenceOverrides (`crates/katgpt-core/src/types.rs`)

Runtime override fields that can be applied per-inference call without modifying the base `Config`:

```rust
pub struct InferenceOverrides {
    pub tree_budget: Option<usize>,
    pub temperature: Option<f32>,
    pub draft_lookahead: Option<usize>,
    pub parallel_threshold: Option<usize>,
    pub screening_threshold: Option<f32>,
    pub sparse_threshold: Option<f32>,
    pub early_exit_patience: Option<usize>,
    pub early_exit_gap: Option<f32>,
    // MTP Drafter overrides (Plan 055)
    pub mtp_activation_threshold: Option<usize>,
    pub mtp_cluster_vocab_threshold: Option<usize>,
    pub mtp_shared_kv_prompt_threshold: Option<usize>,
    pub mtp_cluster_size: Option<usize>,
    pub mtp_min_output_tokens: Option<usize>,  // skip MTP when remaining tokens < threshold (Plan 117 T15)
    pub mtp_cluster_topk: Option<usize>,       // compute logits for top-K clusters (Plan 117 T22)
    // SP-KV inference-time threshold knob (Plan 070)
    pub sp_kv_threshold: Option<f32>,
    // PTRM width scaling (Plan 083)
    pub width_rollouts: Option<usize>,
    pub early_stop_threshold: Option<f32>,
    // EqR Convergence Selection (Plan 119)
    pub convergence_selector: Option<ConvergenceSelector>,
    // MLS Multi-Layer Sum override (Plan 104)
    pub mls_layers: Option<usize>,
    // Drafter LoRA path (Plan 117: MTP LoRA Drafter)
    pub drafter_lora_path: Option<std::path::PathBuf>,
    // SR²AM horizon truncation override (Plan 112 T11)
    pub max_plan_horizon: Option<usize>,
    // Hydra Adaptive Layer Budget (Research 148, Plan 165)
    #[cfg(feature = "hydra_budget")]
    pub hydra_skip_threshold: Option<f32>,       // override Hydra skip threshold
    #[cfg(feature = "hydra_budget")]
    pub hydra_skip_erasure_draft: Option<bool>,  // override erasure-skip-draft flag
}
```

Overrides are merged onto a base `Config` at inference time, allowing per-request parameter tuning without cloning or mutating the shared config.

### InferenceResult (`crates/katgpt-core/src/types.rs`)

Output of a single inference pass with reward signal for feedback loop:

```rust
pub struct InferenceResult {
    pub domain: String,
    pub reward: f32,
    pub tree_budget_used: usize,
    pub budget_level: u8,
    pub prompt_hash: u64,
    pub output: String,
    pub timestamp: i64,
    pub screened: bool,
    // Feature-gated: `sr2am_configurator` (Plan 112)
    pub planning_decision: Option<PlanningDecision>,  // SR²AM planning decision
    pub plan_horizon_used: usize,                     // actual horizon after entropy truncation
}
```

### QuantizedKVCache (`src/types.rs`)

Shared interface for quantized KV caches, katgpt-rs–specific (not in katgpt-core):

```rust
pub trait QuantizedKVCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]);
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]);
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    fn reset(&mut self);
    fn pos(&self) -> usize;
    fn set_pos(&mut self, pos: usize);
}
```

Enables `forward_quantized` to work with any compression backend (TurboQuant, SpectralQuant, or future methods).

## TransformerWeights (`transformer.rs`)
```rust
pub struct TransformerWeights {
    pub wte: Vec<f32>,              // [vocab_size, n_embd] — token embedding
    pub wpe: Vec<f32>,              // [block_size, n_embd] — position embedding
    pub lm_head: Vec<f32>,          // [vocab_size, n_embd] — output projection
    pub layers: Vec<LayerWeights>,  // per-layer weights (n_layer entries)
    pub mtp_activation_proj: Option<Vec<f32>>,  // MTP target activation projection (Plan 055)
    pub mtp_cluster_classifier: Option<Vec<f32>>, // MTP cluster classifier (Plan 055)
    pub mtp_cluster_map: Option<Vec<usize>>,     // MTP vocab cluster mapping (Plan 055)
}

pub struct LayerWeights {
    pub attn_wq: Vec<f32>,   // [n_embd, n_embd]
    pub attn_wk: Vec<f32>,   // [n_embd, kv_dim]
    pub attn_wv: Vec<f32>,   // [n_embd, kv_dim]
    pub attn_wo: Vec<f32>,   // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,    // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,    // [n_embd, mlp_hidden]
}
```
- Weight init: Kaiming-style `rng.normal() * sqrt(2 / (n_embd * n_layer))`
- Embedding init: `sqrt(2 / n_embd)`
- `TransformerWeights::new(config, rng)` creates all layers

## ForwardContext (`transformer.rs`)
Pre-allocated scratch buffers for zero-allocation forward passes:
```rust
pub struct ForwardContext {
    x: Vec<f32>,              // [n_embd] — hidden state (mutated in-place)
    q: Vec<f32>,              // [n_embd]
    k: Vec<f32>,              // [kv_dim]
    v: Vec<f32>,              // [kv_dim]
    attn_out: Vec<f32>,       // [n_embd]
    hidden: Vec<f32>,         // [mlp_hidden]
    xr: Vec<f32>,             // [n_embd] — residual buffer 1
    xr2: Vec<f32>,            // [n_embd] — residual buffer 2
    scores: Vec<f32>,         // [block_size] — attention scores
    logits: Vec<f32>,         // [vocab_size]
    pub hidden_state: Vec<f32>, // [n_embd] — snapshot before lm_head (for REST/Validator)
    // Feature-gated buffers (allocated once, zero runtime cost when unused):
    lora_buf: Vec<f32>,       // [rank] — LoRA intermediate (always allocated)
    // #[cfg(feature = "sparse_mlp")]
    active_indices: Vec<usize>, // [mlp_hidden] — alive neuron indices (Plan 022)
    // #[cfg(feature = "sparse_mlp")]
    active_values: Vec<f32>,    // [mlp_hidden] — alive neuron values (Plan 022)
    // MTP Drafter buffers (Plan 055)
    mtp_context_buf: Vec<f32>,    // MTP projection intermediate buffer
    // TurboQuant buffers
    tq_dequant_pos: Vec<f32>,     // dequantized KV for current position
    // Paged KV cache: pre-allocated flat buffers for attention computation
    paged_flat_key: Vec<f32>,     // [block_size * kv_dim]
    paged_flat_value: Vec<f32>,   // [block_size * kv_dim]
    // Raven: pre-allocated query buffer for per-head slot attention
    raven_query_buf: Vec<f32>,    // [kv_dim]
    // Quantized KV cache incremental dequant tracking
    dequant_pos: Vec<usize>,      // [n_layer] last dequantized position per layer
    // Delta routing (Plan 097, feature: `delta_routing`)
    block_deltas: Vec<Vec<f32>>,  // [n_blocks][n_embd] accumulated deltas per block
    delta_routing_logits: Vec<f32>, // [max_sources] routing logits temp buffer
    // CODA fused kernels (Plan 103, feature: `coda_fusion`)
    coda_partial_sums: Vec<f32>,  // [1] single-block RMS sum of squares
    // MLS Multi-Layer Sum (Plan 104, feature: `mls_aggregate`)
    mls_buf: Vec<f32>,            // [n_embd] accumulator for last K layer residuals
    mls_count: usize,             // how many layers accumulated
    // Tiled attention (Plan 115, feature: `tiled_attention`)
    tiled_q: Vec<f32>,            // [block_size × n_embd] repacked queries per head
    tiled_k: Vec<f32>,            // [block_size × kv_dim] repacked keys per kv group
    tiled_v: Vec<f32>,            // [block_size × kv_dim] repacked values per kv group
    tiled_out: Vec<f32>,          // [block_size × n_embd] tiled output before transpose
}
```
- Created once, reused across calls via `ctx.reset()`
- `hidden_state` is copied from `x` before lm_head projection — "free embedding" for vector search
- `lora_buf` avoids per-projection LoRA allocation; fused into `lora_apply()` in-place
- Sparse MLP buffers pack alive ReLU neurons for `sparse_matmul()` — only used when `alive_ratio ≤ sparse_threshold`

## MultiLayerKVCache (`transformer.rs`)
```rust
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
}
pub struct KVCache {
    pub key: Vec<f32>,    // [block_size, kv_dim]
    pub value: Vec<f32>,  // [block_size, kv_dim]
}
```
- One KVCache per layer
- `kv_dim = n_kv_head * head_dim` (may be < n_embd with GQA)
- `reset()` clears all layers
- `snapshot(pos, config)` → `KVSnapshot` (copies only filled slots `[0..pos*kv_dim]`)
- `restore(snapshot, config)` — rollback to earlier state

## Forward Pass (`transformer.rs`)

`forward()` is the **public API** — it delegates to internal `forward_base()` with feature-appropriate parameters:

```rust
// Public API — handles domain_latent feature gating internally
pub fn forward(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &mut [f32]  // logits

// Internal — called by forward(), forward_prefill(), and generate_with_prefill()
// Accepts optional LoRA adapter and domain latent
fn forward_base(
    ctx, weights, cache, token, pos, config,
    lora: Option<&LoraAdapter>,        // cfg: always available
    domain_latent: Option<&DomainLatent>,  // cfg(feature = "domain_latent")
) -> &mut [f32]
```

Pipeline (inside `forward_base`):
1. **Embedding**: `x = wte[token] + wpe[pos]`
2. **Layer loop** (n_layer iterations):
   a. RMSNorm → QKV projection (GQA: K/V use kv_group)
   b. Store K/V in per-layer cache at position `pos`
   c. Multi-head attention (fused: score → softmax → weighted value)
   d. Output projection + residual add
   e. RMSNorm → MLP (matmul_relu + matmul) + residual add
   f. *(domain_latent)* At layer `n_layer / 2`: inject `DomainLatent` into K/V
3. **Snapshot**: `hidden_state = x` (before lm_head)
4. **LM Head**: `logits = lm_head @ x`

### GQA (Grouped-Query Attention)
When `n_kv_head < n_head`, K/V heads are shared:
- `kv_group = q_head * n_kv_head / n_head`
- K/V projection outputs `kv_dim` instead of `n_embd`
- 4× KV cache reduction for `n_head=8, n_kv_head=2`

## Math Kernels (`crates/katgpt-core/src/types.rs`)
All hot-path kernels are `#[inline(always)]` with `unsafe get_unchecked`:
- `matmul(out, w, x, rows, cols)` — out = W @ x — SIMD-accelerated via `simd_dot_f32` (Plan 060)
- `matmul_relu(out, w, x, rows, cols)` — fused matmul + ReLU — SIMD-accelerated with fused ReLU zero-clamp (Plan 060)
- `sparse_matmul(out, w, x, rows, cols, active_indices, active_values)` — skip dead ReLU neurons (Plan 022)
- `softmax(x)` — in-place, one-pass exp+sum, uses `inv_sum` multiply
- `softmax_scaled(x, scale)` — scaled softmax for attention (divides by sqrt(head_dim) before exp)
- `rmsnorm(x)` — in-place, two-pass with `inv_rms` multiply
- `attention_head(...)` — fused: score → softmax → weighted value (avoids separate softmax write)
- `sample_token(logits, rng)` — categorical sampling
- `lora_apply(output, lora, input, lora_buf)` — in-place LoRA delta: `output += (α/r) × B @ (A @ input)`
- `gegelu(hidden, gate, up)` — GeGLU activation for Gemma 2 MLP: `GELU(gate) * up`
- `gegelu_tanh(hidden, gate, up)` — GeGLU with tanh approximation
- `rmsnorm_with_gamma(x, gamma)` — RMSNorm with learnable gain parameter
- `rmsnorm_with_gamma_eps(x, gamma, eps)` — RMSNorm with gain and custom epsilon

## SIMD Kernels (`crates/katgpt-core/src/simd.rs`, Plan 060)

Runtime SIMD detection and dispatch for hot-path operations:
- `SimdLevel` enum: `Scalar`, `Neon` (ARM), `Avx2` (x86_64)
- `simd_level()` — runtime detection of available SIMD level
- `simd_dot_f32(a, b, len)` — NEON `vfmaq_f32` / AVX2 `_mm256_mul_ps` dot product
- `simd_outer_product_acc(acc, a, b, m, n)` — rank-1 update for HLA SK, CQV, PKV
- `simd_matmul_rows(out, w, x, rows, cols)` — row-major matmul via SIMD dot
- `simd_matmul_relu_rows(out, w, x, rows, cols)` — SIMD matmul + fused ReLU clamp
- `simd_fused_decay_write(dst, decay, src, write)` — fused decay+write for HLA state update
- `maxsim_score(queries, documents, lq, ld, dim)` — MaxSim late-interaction scoring
- `maxsim_score_packed(queries, query_offsets, documents, doc_offsets, pair_q_ids, pair_d_ids, dim)` — batched MaxSim for packed representations
- `simd_add_into(dst, a, b)` — SIMD-accelerated element-wise vector add
- No dependencies — pure `core::arch::{aarch64, x86_64}` intrinsics
- Zero-cost dispatch: compile-time `#[cfg(target_arch)]` + runtime level check

## Additional Forward Variants (`transformer.rs`)

| Function | Description |
|----------|-------------|
| `forward_prefill(ctx, prefill, weights, cache, tokens, config, lora, domain_latent)` | Bidirectional prefill — all prompt tokens attend to all others (Plan 025) |
| `forward_paged(ctx, weights, paged_cache, token, pos, config, seq_idx)` | Paged KV cache forward — copy-on-write branch isolation |
| `forward_raven(ctx, weights, raven_cache, token, pos, config)` | Raven RSM forward — slot-based O(1) routing attention |
| `forward_turboquant(ctx, weights, tq_cache, token, pos, config)` | TurboQuant forward — bit-packed KV cache with dequantize-on-read |
| `forward_hla(ctx, weights, hla_cache, token, pos, config)` | Symmetric second-order HLA — O(d²) constant-state attention, SIMD-accelerated (Plan 057/060, `hla_attention`) |
| `forward_ahla(ctx, weights, ahla_cache, token, pos, config)` | Asymmetric AHLA — O(d·dv) constant-state attention, SIMD-accelerated (Plan 057/060, `hla_attention`) |
| `forward_with_domain_latent(ctx, weights, cache, token, pos, config, dl)` | Convenience wrapper — `forward_base` with domain latent only (no LoRA) |
| `forward_sp_kv(ctx, weights, sp_kv_cache, token, pos, config, predictors, bias)` | SP-KV self-pruned KV forward — utility-gated attention with learned predictor MLP (Plan 070, `sp_kv`) |
| `forward_looped(ctx, weights, cache, ahla_cache, token, pos, config, residual_gate, sdpa_gate)` | LT2 looped forward — weight-shared T-pass loop with hybrid SDPA+AHLA dispatch (Plan 108, `lt2_looped`) |
| `forward_coda(ctx, weights, cache, token, pos, config, lora, domain_latent)` | CODA-fused forward — single-pass SIMD kernels eliminate intermediate buffer writes (Plan 103, `coda_fusion`) |
| `forward_decode_stage(ctx, weights, cache, token, pos, config, stage)` | DecodeStage dispatch — routes to draft/target/coda based on stage enum |
| `depth_route(residual, sources, query_weight, norm_weight, logits_buf, n_embd)` | Delta routing — softmax-weighted blend of accumulated block deltas (Plan 097) |
| `depth_route_weights(sources, query_weight, norm_weight, n_embd)` | Returns routing weights without mutation (for analysis/logging) |

> **Plan 059 Note**: HLA is inference-only — SDPA→HLA distillation via LoRA shows KL divergence does NOT converge. HLA provides streaming O(1) attention for inference but cannot be trained to approximate SDPA outputs. Use DeltaMemoryState for facts/retrieval.

## LT2 Looped Forward Pass (`transformer.rs`, Plan 108)

Weight-shared T-pass loop: same layer weights applied T times, yielding effective depth T×n_layer with no extra parameters. Hybrid dispatch mixes SDPA (full attention) and AHLA (O(1) constant-state) layers per loop iteration.

```
Input: x = wte[token] + wpe[pos]
For τ = 1..T:
  Save prev_h = x
  For ℓ = 1..n_layer:
    is_full = match hybrid_pattern {
      Uniform    => true,
      Interleave{full_ratio:5} => (ℓ % 5) == 4,
      Bookend    => ℓ == 0 || ℓ == n_layer-1,
    }
    h' = h + Mixer_ℓ(h, is_full)    // AHLA or SDPA
    h  = h' + FFN_ℓ(h')             // shared FFN
    if gated_attn && is_full: h = SdpaOutputGate(h)  // sigmoid gate, zero-init
  h = h̃ + ρ_τ ⊙ prev_h             // per-loop residual gate (zero-init)
Output: lm_head(h)
```

**Key types** (`crates/katgpt-core/src/types.rs`):

| Type | Description |
|------|-------------|
| `LoopMode` | `None` (standard) or `WeightShared { loop_count: T }` |
| `HybridPattern` | `Uniform`, `Interleave { full_ratio }`, `Bookend` |
| `ResidualGate` | Per-loop learned gate ρ_τ — zero-init → first iteration is identity |
| `SdpaOutputGate` | Sigmoid gate after SDPA before Wo — zero-init → sigmoid(0) = 0.5 neutral |

**Memory scaling**: AHLA layers use O(d·dv) constant state (no growth with L or T). SDPA layers use O(L·d) KV cache (no growth with T). Hybrid 1:4 achieves ~95% throughput of pure SDPA T=4 with 80% constant-memory layers.

**Any-Time LT2 Dispatch (Issue 035, Research 273 — ELT arXiv:2604.09168)**: the same artifact serves requests at any compute budget by exiting the loop early or over-iterating, without retraining. `forward_looped()` takes a final `elastic_loop_override: Option<usize>` parameter:

- `None` → byte-identical to pre-Issue-035 behavior (uses `loop_mode.loop_count`).
- `Some(L)` → runs L loops, clamped to `[max(loop_min,1), 2×max(loop_max, base)]` per `Config::effective_loop_count()`.
  - `Config::loop_min` (default 0 → treated as 1) = floor; refusal to exit below this preserves representational capacity (ELT §1.4: `1N × 32L` collapsed to FID 10.30).
  - `Config::loop_max` (default 0 → derive from `loop_mode`) = trained max; `2×` is the over-iteration cap (ELT §1.5: modest over-looping regularized).
- Override refused when `loop_mode` is `None` or `TrainingFree` (no weight-shared loop to exit from).

The parameter is unconditional (no feature gate) — it is a caller input, not a feature. Zero-overhead when `None`. Use cases: crowd NPCs run L_min, hero NPCs run L_max, crisis moments over-iterate to 2×L_max — all from one BLAKE3-committed snapshot.

**Feature gate**: `lt2_looped = ["hla_attention"]` (default-on). GOAT: 11/11 proofs pass.

### Gain/Cost Loop Halting (`transformer.rs` + `katgpt-core::gain_cost_halt`, Plan 304)

**Plan 304 / Research 282 / arXiv:2606.18023 (LoopCoder-v2).** An opt-in primitive that decides — per loop, per dispatch, at runtime — whether to continue iterating or halt, based on the **gain/cost scissors** criterion: *halt when marginal refinement gain < marginal drift cost × τ*. This is the dynamic counterpart to Issue 035's static `elastic_loop_override`: instead of the caller passing `Some(L)`, the halter decides L each loop from the hidden-state trajectory.

**Kernel** (`katgpt-core::gain_cost_halt`): `GainCostLoopHalter` — a 12-byte state struct + `halt_decision(loop_idx, gain, cost, cos_theta) -> HaltDecision`. Returns one of:
- `Continue` — gain ≥ cost × τ and no oscillation detected.
- `Halt { reason }` — `GainBelowCost` (scissors crossover) or `Oscillation` (cos θ < 0 for `patience` consecutive loops).
- `RefusedFloor` — `loop_idx < l_min`; refuse to halt to protect representational capacity (ELT §1.4: sub-floor loops collapse).

**Forward-path wiring** (`forward_looped`, Plan 304 T2.1–T2.3): the function takes a final `#[cfg(feature = "gain_cost_halt")] halter: Option<&mut GainCostLoopHalter>` parameter. When `Some(halter)`, after each loop `tau > 0`:
1. **gain** = `step_size(ctx.x, ctx.prev_h)` = `||h^(tau) - h^(tau-1)||₂`.
2. **cost** = fixed tax (`cost_floor`), cached as `0.01 × first_loop_step_size` on `tau == 1` — mirrors LoopCoder-v2's flat Ω(r).
3. **cos_theta** = `angular_change(curr_step, prev_step)` between successive update directions.
4. Call `halter.halt_decision(tau + 1, gain, cost, cos_theta)`. On `Halt` → `break`.

**Composition rules:**
- **With `elastic_loop_override` (Issue 035, T2.2):** static override wins. If the caller passes `Some(L)`, the halter is IGNORED entirely (`halter_active = elastic_loop_override.is_none()`). The caller asked for a specific loop count — respect it.
- **With `recursion_gate` (Plan 283 T2.2):** both can fire in the same loop. The recursion gate checks dead-compute via logit improvement; the halter checks via gain/cost scissors + oscillation. Either can `break` the loop independently. They compose because they measure different failure modes.

**Open questions (resolved in Phase 2):**
1. **Cost signal default** → fixed tax (`0.01 × first_step`), overridable. riir-ai can substitute coherence-decay/staleness by not using this code path.
2. **Gain signal** → **DEVIATION:** the plan called for effective-rank delta, but the per-loop hidden state in `forward_looped` is a single vector (`ctx.x[..n]`, S=1), for which `hidden_erank` returns 0.0 (degenerate). Phase 2 uses `step_size` as the gain signal — monotone in refinement, cheaper than erank, and the kernel ships `step_size` for exactly this use.
3. **Determinism** → gain/cost are pure functions of deterministic hidden state, so L is deterministic. Tested: `refused_floor_never_halts_when_l_min_above_loop_count` confirms the no-op path.
4. **TF-Loop interaction** → Phase 2.5 (DEFERRED). Only wired into `forward_looped` (WeightShared path), not `forward_tf_looped`.

**Latent vs Raw:** gain/cost signals are local latent (per-loop hidden-state deltas). The halt count L is a deterministic raw scalar, safe to sync/replay.

**Feature gate:** `gain_cost_halt = ["katgpt-core/gain_cost_halt"]` (opt-in, OFF by default). The `halter = None` path is byte-identical to pre-Plan-304 via `#[cfg]` + `Option::is_none()`. GOAT gate G1–G5 (Research 149 §5) must pass before default promotion; G2 (≥75% crowd-NPC compute savings) and G3 (no important-NPC regression) are the headline gates — **deferred: requires synthetic loop suite harness**.

## MTP Projection (`transformer.rs`, Plan 055)

Multi-Token Prediction projection weights for draft model acceleration:
- `MtpProjection` — Projection weights for target-activation-based MTP drafting
- `project_target_activation()` — Projects hidden state to draft token logits
- `cluster_map_round_robin()` — Round-robin vocab cluster assignment
- `cluster_map_from_embeddings()` — Embedding-similarity-based cluster assignment
- Threshold-gated: features activate only when config thresholds are met (see `13_mtp_threshold_guide.md`)

## Generate (`transformer.rs`)
```rust
pub fn generate(ctx, cache, weights, config, rng, token, n_tokens) -> Vec<usize>
pub fn generate_into(ctx, cache, weights, config, rng, tokens, n_tokens)  // zero-alloc variant
pub fn generate_batch(ctx, cache, weights, config, rng, token, n_tokens, n_samples) -> Vec<Vec<usize>>
pub fn generate_with_prefill(
    ctx, prefill, cache, weights, config, rng,
    prompt_tokens, n_tokens,
    // Optional per-call overrides:
    lora_pair: Option<&LoraPair>,          // reader→writer LoRA switching
    domain_latent: Option<&DomainLatent>,  // mid-layer domain conditioning
) -> Vec<usize>
```
- Autoregressive: sample → feed back → repeat
- `generate_into` reuses pre-allocated buffers (zero-alloc hot path)
- `generate_batch` uses Rayon `par_iter` with per-worker contexts
- `generate_with_prefill` runs bidirectional prefill (reader LoRA) then switches to causal decode (writer LoRA), with optional domain latent injection
- `tokens_to_string(tokens, config)` — converts token IDs back to string via `id_to_vocab` lookup

## PagedKVCache (implemented, DDTree integration pending)
```rust
pub struct PagedKVCache {
    pages: Vec<Vec<f32>>,                    // pool of fixed-size pages
    layer_page_tables: Vec<Vec<Vec<usize>>>, // per-layer, per-sequence page indices
    free_pages: Vec<usize>,                  // reuse pool
    kv_dim: usize,
}
```
- Fixed `PAGE_SIZE = 16` tokens per page
- `fork(seq_idx, fork_at_pos)` — copy-on-write branch (shares prefix pages)
- Designed for DDTree branch exploration (each branch = one sequence)
- Deferred integration: currently DDTree uses flat `snapshot()/restore()` instead

## KVSnapshot
```rust
pub struct KVSnapshot {
    pub layers: Vec<KVLayerSnapshot>,
    pub pos: usize,
}
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,    // [pos, kv_dim]
    pub value: Vec<f32>,  // [pos, kv_dim]
}
```
- Cheap: copies only filled slots `[0..pos*kv_dim]` per layer
- Used in speculative rollback: snapshot before verify, restore on reject

## ScreeningPruner: Absolute Relevance (Plan 021)

Distilled from ["Screening Is Enough"](https://arxiv.org/abs/2604.01178) — upgrades binary pruning to **graded relevance**:

```rust
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}
```

Score formula: `blended = parent_score + ln(P_llm) + ln(R)`

| Relevance R | ln(R) | Effect |
|---|---|---|
| 1.0 | 0.0 | No penalty — perfect match |
| 0.5 | -0.69 | Soft penalty — mediocre match |
| 0.0 | -∞ | **Hard trim** — branch killed |

`ConstraintPruner` adapts via `BinaryScreeningPruner(pruner)` (R ∈ {0.0, 1.0}). `WasmPruner` implements `ScreeningPruner` natively — loads optional WASM `relevance` export (Q16.16 fixed-point), falls back to binary `is_valid` if missing.

`config.screening_threshold` (default `0.0`) controls hard-trim cutoff. Set `> 0.0` to aggressively trim low-relevance branches.

## Freeze/Thaw (`src/pruners/freeze.rs`, Plan 092)

Shared freeze/thaw disk I/O for `repr(C)` bandit knowledge structs. Zero-dependency binary persistence — raw `std::fs::write`/`read` on `repr(C)` data with magic bytes + version validation on load. No serde/bincode needed.

```rust
pub fn save_frozen<T>(path: &Path, data: &T) -> Result<(), String>
pub fn load_frozen<T>(path: &Path) -> Result<T, String>
```

### Key Fix: Per-Move Reward (Issue 065)

Initial freeze/thaw showed **negative** knowledge transfer (-3pp win rate). Root cause: binary game-end reward + low α=0.3 blended with per-move signal, causing all Q-values to converge to ~0.25 when losing 86% of games (no differentiation).

**Fix:** `HL_PER_MOVE_ALPHA = 1.0` (pure per-move reward, no game-end blending) + `HL_DELTA_AMPLIFICATION = 10.0` (amplifies raw heuristic delta ±0.01–0.06 → ±0.1–0.6).

Results (GoHL vs Validator, 100 rounds × 3 phases):
| Metric | Frozen | Baseline | Δ |
|--------|--------|----------|---|
| Win Rate | 25% | 14% | **+11pp ✅** |
| Avg Score | -13.3 | -16.8 | **+3.5 ✅** |

Q-values after learning:
- Corner: 0.80, Side: 0.64, Center: 0.74, Capture: 0.75, Defense: 0.40, Extend: 0.48, Influence: 0.59, Pass: 0.00
- 2× spread (Corner vs Defense) vs old flat ~0.25

Learning vs Random also verified: Q-values differentiate properly (spread > 0.1) with α=1.0, unlike old binary reward that collapsed all to ~0.85.

Run: `cargo run --example go_08_self_play_freeze --features go`

## CGSP — Curiosity-Guided Self-Play (`crates/katgpt-core/src/cgsp/`, Plan 274)

Modelless, inference-time distillation of the SGS triad (Bailey et al., arXiv:2604.20209). Three frozen role-fillers + one bandit; **no gradient updates**, only priority-table updates on direction vectors. The public-facing entry point is `CgspLoop`, generic over five pluggable trait-fillers wired in at compile time (zero-cost, no dynamic dispatch on the hot path).

```rust
pub struct CgspLoop<C, G, S, B, Col = EntropyCollapse, Df = NoOpDifficultyFilter, Qg = NoOpBatchGate>
where
    C: CuriosityConjecturer,  // samples k candidate directions per cycle
    G: QualityGuide,          // scores each candidate [0, 1]
    S: Solver,                // attempts a candidate, returns solve-rate
    B: HintDeltaBandit,        // absorbs r_synth, exposes priorities
    Col: CollapseSignal,       // entropy-collapse detector + recovery injector
    Df: DifficultyFilter,      // breakeven-complexity admission gate
    Qg: BatchQualityGate;      // degenerate-batch gate (Plan 111 data_gate)
```

**Per-cycle pipeline** (Plan 274 §2.3, see `crates/katgpt-core/src/cgsp/loop_.rs::cycle()`):

1. Conjecturer samples k candidates → `scratch.candidates`.
2. Guide scores each → `scratch.guide_scores`.
3. Difficulty filter admits/rejects → `scratch.admitted`.
4. Solver attempts admitted candidates → `scratch.solve_rates`.
5. Compute `r_synth[i] = (1 − solve_rates[i]) · guide_scores[i]`.
6. BatchQualityGate checks for degeneracy (skip update if degenerate).
7. Bandit absorbs `r_synth` per admitted candidate (Hint-δ absorb-compress).
8. CollapseSignal checks entropy; if < τ_low, inject exploration mass.

**Latent/raw boundary:** Only `f32`, `bool`, and `u32` cross the trait boundary (`CycleResult`). `Direction` and `Target` never escape — they are pure latent state. The freeze/thaw bridge is `CuriosityPrioritySnapshot` (BLAKE3-committed, fixed-size binary encoding; snapshot_id is Uuid v7 for monotonic ordering).

**GOAT gate status** (`.benchmarks/274_cgsp_goat.md`, 9 tests at `tests/bench_274_cgsp_goat.rs`):

| Gate | Measurement | Status |
|------|-------------|--------|
| G1 transfer-to-target | CGSP 0/64, baseline 0/64 | ⚠ INFORMATIONAL — CGSP is curiosity-driven, not target-seeking (root-cause: `(1 − solve_rate)` factor rewards intermediate-difficulty arms) |
| G2 collapse recovery | 1 cycle with aware; 200+ without | ✅ PASS |
| G3 feature isolation | `cargo check` clean both ways | ✅ PASS |
| G4 per-cycle overhead | 831.3 ns/cycle (release, isolated `--test-threads=1`, Apple Silicon arm64) | ✅ PASS |
| P2 1000 NPCs/tick | 808 µs/tick (0.81 µs/NPC, Rayon 8 chunks, isolated) | ✅ PASS |
| P3 allocations | 13.00 allocs/cycle (bounded, not zero) | ✅ PASS (bounded) — optimization tracked in `.issues/021_cgsp_cycle_allocation_reduction.md` |
| G6 latent/raw boundary | only f32+bool+u32 in CycleResult | ✅ PASS |

**Promotion decision:** KEEP OPT-IN. CGSP is architecturally sound and plasma-tier fast, but its value proposition is collapse recovery + degenerate-batch gating — *not* target-seeking. Promote to default only after riir-ai Plan 299 validates on real game domains.

**Consumer pattern:**

```rust
use katgpt_rs::cgsp::{
    CgspConfig, CgspLoop, ColinearityBatchGate, EntropyCollapse,
    BreakevenDifficultyFilter, HlaProjectionGuide, PoolConjecturer, ScratchBuffers, Target,
};

let mut lp = CgspLoop::new(conjecturer, guide, solver, bandit, CgspConfig::default())
    .with_collapse(EntropyCollapse::new(0.30))
    .with_difficulty_filter(BreakevenDifficultyFilter::default())
    .with_batch_gate(ColinearityBatchGate::default());
let mut scratch = ScratchBuffers::new(k, pool_size);
let result = lp.cycle(&target, &mut scratch);
```

See `examples/cgsp_minimal.rs` and `examples/cgsp_collapse_recovery.rs` for full runnable demos. Implementation lives in `crates/katgpt-core/src/cgsp/` so `riir-engine` (Plan 299) can consume it without depending on the root application crate; `src/cgsp.rs` is a thin re-export shim preserving the `katgpt_rs::cgsp::*` import path.

## SpeculativeVerifier (Strategy Pattern)

Based on [Algorithm 1 from Leviathan et al. 2022](https://arxiv.org/pdf/2211.17192) — the verification strategy is swappable via trait:

```rust
pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&mut self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

| Verifier | Availability | What it does |
|----------|--------------|--------------|
| `SimulatedVerifier` | always compiled | DFlash/AR draft → DDTree → simulated acceptance cap → bonus token from last marginal |
| `LeviathanVerifier` | always compiled | AR draft → target model p/q scoring → rejection sampling → residual distribution → bonus from target p(x). Proves Algorithm 1 works end-to-end. |
| `D2fDrafterVerifier` | `tri_mode` feature | D2F diffusion drafts in parallel (bidirectional within block) → AR verifies with causal attention (Plan 089: Tri-Mode "self-speculation") |

`SimulatedVerifier` is fast (no target model). `LeviathanVerifier` is the full Algorithm 1 — mathematically proven distribution-preserving, but needs large model asymmetry to be faster than pure AR.

## PPoT: Logit-Parameterized CPU Resampling (Plan 026 + 027)

After DFlash produces marginals and DDTree rejects all paths, PPoT identifies high-entropy positions in the saved marginals and resamples variant token sequences using **only CPU** — no additional GPU forward passes. Resampled paths are screened through `ScreeningPruner` for verification. This activates only on failure (zero overhead on success path).

Plan 027 extends baseline with TRT-inspired adaptive rescue: rejection memory (ring buffer of "don't" insights), per-sample strategy cycling across `TokenRule` variants, and self-consistency ranking for multi-valid variant selection. Knowledge accumulates within a generation session, biasing future resampling toward historically successful positions and rules.

```rust
pub enum TokenRule {
    Digit,      // prefer digit tokens
    Compare,    // prefer comparison operators
    Arithmetic, // prefer arithmetic operators
    Augment,    // prefer augmented assignment
    All,        // no preference
}
```

## Prompt Router: Batch-Level Domain Routing (Plan 023)

Inspired by [EMO: Pretraining Mixture of Experts for Emergent Modularity](https://arxiv.org/abs/2406.08732) — document-level routing constraints force experts to learn high-level semantic domains instead of syntax.

1. **Classify once** — `KeywordRouter` scores the prompt against domain keywords (V1, ~80% accuracy; embedding-based V2 via anyrag is planned)
2. **Select expert** — `ExpertRegistry` returns a `Box<dyn ScreeningPruner>` + optional LoRA path for the matched domain
3. **Lock for generation** — the selected `ScreeningPruner` is passed to `build_dd_tree_screened()`, preventing domain drift

```rust
let router = KeywordRouter::new(config.domain.clone());
let registry = ExpertRegistry::from_config(&config, pruner_dir);

let decision = router.route("solve this sudoku puzzle");
let expert = registry.get_expert(&decision.domain);
// expert.pruner is locked for the entire DDTree generation
```

Domains are defined in `domains.toml` — platform manages expert bundles via Web UI or MCP agent.

## Embedding Router: KV Cache Priming (Plan 024)

Extends keyword routing with **semantic embedding retrieval** from anyrag. When a user edits a known file, the system retrieves the most relevant document embedding, projects it to the draft model's hidden dimension, and injects it as KV cache priming context via `dflash_predict_conditioned_with`.

**Three-tier fallback** (graceful degradation when anyrag is unavailable):

```
1. Embedding search (POST /search/embedding)  ~200ms
   ↓ on failure
2. Domain classify (POST /classify/domain)     ~100ms
   ↓ on failure
3. KeywordRouter (local, no network)            <1ms
```

```rust
let router = EmbeddingRouter::new(
    embedding_config, domains, Box::new(TruncatePadProjector),
);

// Sync: delegates to KeywordRouter (no network)
let decision = router.route("fn validate_token(");

// Async: tries anyrag embedding search, falls back to keyword
let decision = router.route_async("fn validate_token(").await;

if let Some(embedding) = &decision.embedding {
    let projected = router.project_embedding(embedding, draft_config.n_embd);
    speculative_step_embedding_conditioned(&weights, &config, token, pos, &projected, &mut rng);
}
```

**Separation from target model conditioning:** `speculative_step_conditioned_with` uses the target model's hidden state (syntactic alignment). `speculative_step_embedding_conditioned` uses a retrieved embedding (semantic alignment). These are complementary signals.

## Bidirectional Prefill + Modality LoRA Switching (Plan 025)

Distilled from [ZAYA1-VL-8B Technical Report](https://arxiv.org/abs/2504.02268) — two production techniques adapted for the Python→Rust translation pipeline:

### 1. Bidirectional Prefill

During prefill, prompt tokens (Python code + anyRAG docs) attend to ALL other prompt tokens — no causal mask. Code is non-linear; a function body references a struct 3,000 tokens earlier. Generation tokens still use causal attention. Zero overhead on the decode hot path — prefill runs once per request.

### 2. Modality LoRA Switching

Load two LoRA adapters per domain — a `reader_lora` (active during prefill) and a `writer_lora` (active during decode). The switch is a reference swap at the prefill→decode boundary. Zero data movement.

```
  tokens[0..prompt_len]                    tokens[prompt_len..]
        │                                         │
   ┌────┴────┐                              ┌─────┴─────┐
   │ PREFILL │  bidirectional attention     │  DECODE   │  causal attention
   │         │  reader_lora active          │           │  writer_lora active
   └────┬────┘                              └─────┴─────┘
        │ KV cache populated                      │ generates tokens
        └──────────── shared KV cache ────────────┘
```

### LoraPair & PrefillContext

```rust
pub struct LoraPair {
    pub reader: Option<LoraAdapter>,  // active during bidirectional prefill
    pub writer: Option<LoraAdapter>,  // active during causal decode
}

pub struct PrefillContext {
    pub hidden: Vec<f32>,  // [prompt_len × n_embd] — pre-allocated once
}
```

Two-phase per layer (zero-copy):

| Phase | What | Buffers |
|-------|------|---------|
| A: KV Fill | Compute K/V for all positions → store in cache | Reuses `ForwardContext` per-position |
| B: Bidirectional Attend | Q attends to K/V[0..prompt_len] via `attention_head(t_n=prompt_len)` | `attention_head` unchanged — caller controls range |

```rust
let mut prefill = PrefillContext::new(&config);

// Bidirectional prefill with reader LoRA + optional domain latent
let logits = forward_prefill(&mut ctx, &mut prefill, &weights, &mut cache,
    &prompt_tokens, &config, lora_pair.reader.as_ref(), domain_latent);

// Causal decode — forward() delegates to forward_base(writer LoRA + domain latent)
let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
// Note: for explicit LoRA control during decode, use generate_with_prefill()
// which handles the reader→writer swap internally.
```

## CODA Fusion Kernels (`crates/katgpt-core/src/coda.rs`, Plan 103)

CODA-inspired fused SIMD kernels that algebraically reparameterize matmul+residual+rmsnorm+activation into single-pass SIMD loops, eliminating intermediate buffer writes.

**Key identity (CODA §3.2.1):**
```
RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'
```

This delays the row-wise RMSNorm scale past the next GEMM.

```rust
#[repr(u8)]
pub enum GateActivation {
    Relu,        // max(0, x) — standard 2-layer MLP
    Silu,        // x * sigmoid(x) — LLaMA SwiGLU
    GegeluTanh,  // tanh-approx GELU — Gemma 2 GeGLU
    Gegelu,      // sigmoid-approx GELU — standard GeGLU
}
```

| Kernel | Description |
|--------|-------------|
| `simd_matmul_residual(out_d, out_o, partial_sums, w, x, residual, gamma, bias, rows, cols)` | Fused matmul + residual add + delayed RMSNorm (Plan 103 T3) |
| `compute_rstd(partial_sums, n, eps)` | Compute reciprocal standard deviation from partial sums |
| `simd_matmul_rmsnorm_swiglu(out, x, norm, w_gate, w_up, w_down, rstd, hidden_buf, n)` | Fused RMSNorm + SwiGLU MLP (SiLU activation) |
| `simd_matmul_rmsnorm_activation(out, rstd, hidden, activation, n)` | Apply delayed activation with rstd scaling |
| `simd_matmul_rmsnorm_rope(out, q_buf, k_buf, x, wq, wk, wv, rstd, pos, head_dim, n_heads, theta)` | Fused QKV projection + RoPE with delayed RMSNorm |

**Feature gate:** `coda_fusion`

**Buffer write savings per layer:** ~8 passes (baseline) → ~0 passes (CODA fused).

### MoA — Mixture of Activations (Plan 158)

Token-adaptive activation mixing gated behind `moa_inference`. Instead of a single fixed activation, computes a weighted mixture over a dictionary of 7 activations per element, with gating weights determined per-token via sigmoid dot-product.

```rust
// 7-activation MoA dictionary
pub enum MoaActivation {
    Id,        // σ(x) = x
    Relu,      // max(0, x)
    Relu2,     // max(0, x)²
    LeakyRelu, // max(x, ηx), η = 0.01
    Gelu,      // xΦ(x) (sigmoid approx)
    Silu,      // x · sigmoid(x)
    Tanh,      // tanh(x)
}

pub struct MoaConfig {
    pub d_model: usize,
    pub gate_gating: Vec<f32>,  // [MOA_DICT_SIZE × d_model]
    pub up_gating: Vec<f32>,   // [MOA_DICT_SIZE × d_model]
}
```

**Key design choice:** Uses sigmoid gating (NOT softmax) — paper (arXiv 2605.26647) Table 2 shows sigmoid > softmax > tanh.

| Function | Description |
|----------|-------------|
| `compute_moa_gates(input, gating, d_model)` | Compute π_k = sigmoid(u_k^T x) for k ∈ [0..7) |
| `moa_swiglu(hidden, gate_proj, up_proj, input, moa)` | Token-adaptive bi-MoA SwiGLU: Σ_k ρ_k σ_k(y) ⊙ Σ_ℓ π_ℓ σ_ℓ(z) |
| `simd_matmul_moa(...)` | Fused kernel: matmul + delayed RMSNorm + MoA mixing |

**Feature gate:** `moa_inference` (opt-in)

## Tiled Attention (`crates/katgpt-core/src/attention.rs`, Plan 115)

CPU SIMD tiled flash attention using online-softmax algorithm, adapted from ThunderKittens (Research 077). Processes Q in SIMD-width row tiles, K/V in column tiles — avoids materializing full N×N score matrix.

```
Tile sizes: BR=8 (query rows), BC=128 (key/value columns)
Threshold: tiled path activates when N > 128 (score matrix > L1 cache)
```

| Function | Description |
|----------|-------------|
| `tiled_attention_forward(q, k, v, output, seq_len, head_dim, scale)` | Single-head tiled attention with online-softmax |
| `tiled_attention_batched(q, k, v, output, batch, heads, seq_len, head_dim)` | Multi-head batched via rayon `par_chunks_mut` |

**Online-softmax algorithm (per query tile):**
1. Initialize: `o_tile = 0, max_tile = -inf, norm_tile = 0`
2. For each K/V tile: score → update running max → correction factor → exp → accumulate
3. Final normalize: `o_tile / norm_tile`

**Feature gate:** `tiled_attention`

## Newton-Schulz Orthogonalization (`src/newton_schulz.rs`, Plan 152, Research 114)

5-iteration cubic fixed-point iteration that projects any matrix to its nearest orthogonal factor. Generic building block for Muon-family optimizers.

**Newton-Schulz iteration:**
```
X = G / ||G||_F
for 5 iters: A = X @ X^T; X = a*X + (b*A + c*A@A) @ X
```

Constants from the AMUSE paper (converges for σ ∈ [0, 1]):
- `a = 3.4445`, `b = -4.7750`, `c = 2.0315`
- 5 iterations (fixed)

| Function | Description |
|----------|-------------|
| `transpose(src, rows, cols, dst)` | Row-major transpose, 4-row unrolled for auto-vectorization |
| `matmul_xtx(x, m, n, a)` | Symmetric X·Xᵀ via SIMD dot products (upper triangle + mirror) |
| `newton_schulz(g, rows, cols, out)` | Full orthogonalization: normalize → 5 cubic iterations |

**Feature gate:** `newton_schulz` (default-on)

**GOAT:** 25/25 (Bench 050)

## River-Valley Diagnostics (`src/river_valley.rs`, Plan 152, Research 114)

Modelless training diagnostics that reveal why optimization is (or isn't) converging. Pure scalar arithmetic, no external dependencies.

| Metric | Description |
|--------|-------------|
| `subspace_ratios(gradient, dominant_eigvecs)` | Dominant vs bulk gradient alignment: `r_dom = ||U_k^T g|| / ||g||`, `r_bulk = sqrt(1 - r_dom²)` |
| `effective_rank(w, rows, cols)` | Entropy-based rank measure from singular value distribution |
| `update_cosine_similarity(w_old, w_new)` | Trajectory smoothness via cosine similarity of flattened weight updates |

**Feature gate:** `river_valley` (default-on)

**GOAT:** 25/25 (Bench 050)

## Energy-Gated Attention (`src/ega_attn.rs`, Plan 139)

Spectral salience gating for attention. Gates value aggregation by the spectral energy of key token embeddings — each key position's attention weight is scaled by a learned sigmoid gate derived from dot-product energy of the input embedding with a learned projection vector.

**Algorithm (Algorithm 1 from paper):**
```
Q, K, V ← XW_Q, XW_K, XW_V
S ← QKᵀ/√d + causal_mask;  A ← softmax(S)
e ← X · w_proj                    // [seq_len] energy scores
ẽ ← (e - μ) / (σ + ε)             // z-normalize
g ← σ(α · (ẽ - τ))                // sigmoid gate [seq_len]
Âᵢⱼ ← Aᵢⱼ · gⱼ                   // gate each key position
Âᵢⱼ ← Âᵢⱼ / Σₖ(Âᵢₖ + ε)          // renormalize (sum-to-one)
Y ← Â · V                         // value aggregation
```

**Per-head parameter overhead:** `d + 2` (`w_proj`: d, `alpha`: 1, `tau`: 1). Paper converges to α ≈ 2.2, τ ≈ 0.35.

| Function | Description |
|----------|-------------|
| `sigmoid(x)` | Standard sigmoid σ(x) = 1/(1+exp(-x)) |
| `z_normalize(scores)` | In-place z-normalization with SIMD sum-of-squares |
| `ega_forward(q, k, v, x, w_proj, alpha, tau, ...)` | Full EGA attention forward pass |

**Feature gate:** `ega_attn` (opt-in)

## ShardKV (`src/shard_kv/`, Plan 147, Research 109)

Asymmetric K/V cache compression inspired by the Shard paper. K and V have different structural properties requiring different compression methods.

**Compression paths:**

| Path | Prefill | Decode |
|------|---------|--------|
| K | undo RoPE → PCA rotation → water-fill bit allocation → Lloyd-Max quantize | Hadamard rotation → 8-bit Lloyd-Max streaming (guaranteed lossless) |
| V | Hadamard rotation → K-means VQ (groups of 4, 256 codebook) → 2 bits/elem | Hadamard rotation → 8-bit Lloyd-Max streaming (guaranteed lossless) |

Sink + window: attention sinks and recency window stored losslessly.

Reuses `spectralquant`'s `SpectralRotation`, `LloydMaxQuantizer`, `BitAllocator`, and `waterfill_bits` for the K path.

| Module | Description |
|--------|-------------|
| `kv_cache` | `ShardKVCache` implementation |
| `rope` | `undo_rope` / `reapply_rope` with `RopeFreqs` |
| `types` | `ShardConfig`, `ShardCalibration`, `ShardLayer`, `VqCodebook` |

**Feature gate:** `shard_kv` (opt-in, requires `spectral_quant`, `turboquant`)

## Sleep Consolidation (`src/sleep/`, Plan 154, Research 116)

Offline recursive memory consolidation at KV eviction boundary. When the KV cache fills, performs N offline recurrent passes to consolidate context into GDN2 fast weights, then evicts. Preserves single-pass wake-time latency for real-time constraints (20Hz frame sampling).

```
Existing LT2 Pipeline:
  Input → [SDPA → GDN2 → SDPA → GDN2 → ...]×T (wake-time loops) → Output

With Sleep:
  Input → Context fills → [SDPA → GDN2 → ...]×N (sleep-time consolidation) → Evict KV → Continue
         ↑ Single-pass at wake time (T=1)                    ↑ N-pass at eviction boundary
```

| Type | Description |
|------|-------------|
| `SleepConfig` | Configuration: consolidation passes, eviction threshold, etc. |
| `EvictionStrategy` | `HardEvict` / `SlidingWindow` eviction policy |
| `consolidation_pass(...)` | Single recurrent consolidation pass via GDN2 fast weights |
| `sleep(ctx, weights, kv_cache, gdn2_cache, config, ...)` | Full sleep cycle: N-pass consolidation + eviction |

| Module | Description |
|--------|-------------|
| `consolidation` | Core consolidation loop and `sleep` entry point |
| `eviction` | Eviction strategy implementations |
| `types` | `SleepConfig`, `EvictionStrategy` |

**Feature gate:** `sleep_consolidation` (default-on, requires `lt2_looped`, `gdn2_attention`)

## Spectral Hierarchy (`crates/katgpt-core/src/spectral_hierarchy.rs`, Plan 156, Research 121)

Validates that hierarchical splitting geometry in co-occurrence Gram matrices emerges under the decay assumptions (Theorems 1–2 from Research 121). Three diagnostics:

| Function | Description |
|----------|-------------|
| `eigenspace_alignment(gram, reference, n, k)` | Top-k eigenspace alignment g(k) = (1/k) Σ |⟨vᵢᴬ, vᵢᴮ⟩|. Values > 0.9 indicate strong alignment |
| `haar_wavelet_basis(depth)` | Constructs Haar wavelet basis (scaling + wavelet modes) for a depth-D binary tree |
| `cauchy_interlacing_check(full_eigenvalues, sub_eigenvalues)` | Validates Cauchy interlacing inequality for nested split blocks |

**Feature gate:** `spectral_hierarchy` (default-on)

## Roofline Cost (`crates/katgpt-core/src/roofline.rs`, Research R130, Plan 159)

GPU operator runtime prediction, ported from FlashLib's `info/roofline.py`. Predicts operator runtime in ~5µs CPU-only estimation, replacing ~100ms GemvAutotune benchmarking.

```rust
pub enum ComputeBound {
    Compute,  // FLOP throughput limited
    Memory,   // Bandwidth limited
    Launch,   // Too small; launch overhead dominates
}

pub struct RooflineCost {
    pub runtime_ms: f64,
    pub flops: u64,
    pub bytes_moved: u64,
    pub bound: ComputeBound,
}
```

Operator types: `Gemv`, `Gemm`, `Elementwise`, `Reduction`. Calibrated via `HardwarePeaks` throughput parameters.

**Feature gate:** `roofline_cost` (default-on)

## Dual-Gram PCA (`crates/katgpt-core/src/simd.rs`, Research R130, Plan 159)

Dual-Gram PCA routing for short-sequence calibration. When `seq_len < 4 * head_dim`, computes the Gram matrix G = X·Xᵀ (seq_len × seq_len) instead of the covariance C = Xᵀ·X (d_h × d_h), yielding correct eigenvectors without O(d²) work.

| Function | Description |
|----------|-------------|
| `simd_gram_f32(x, seq_len, d_h, gram_out)` | SIMD-accelerated Gram matrix computation G = X·Xᵀ |
| `calibrate_eigenbasis_dual_gram(samples, head_dim)` | Full dual-Gram calibration pipeline (in `spectralquant::spectral`) |

Reference: FlashLib `primitives/pca/triton/pca.py` L73–116 (Research R130).

**Feature gate:** `dual_gram_pca` (default-on)

## Consolidated Traits (`crates/katgpt-core/src/traits.rs`, Plan 107 Phase 0)

Shared traits for game AI and speculative decoding, consolidated from katgpt-rs and riir-engine to eliminate duplication. Both crates depend on `katgpt-core`, so moving traits here requires zero new dependency edges.

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
    fn batch_is_valid(&self, depth, candidates, parent_tokens, results); // default: per-item
}

pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

pub trait GameState: Clone {
    type Action: Clone;
    fn available_actions(&self, player_id: u8) -> Vec<Self::Action>;
    fn advance(&self, action: &Self::Action, player_id: u8) -> Self;
    fn is_terminal(&self) -> bool;
    fn reward(&self, player_id: u8) -> f32;
    fn tick(&self) -> u32;
}

pub trait StateHeuristic<S: GameState> {
    fn evaluate(&self, state: &S, player_id: u8) -> f32;
}

pub trait RolloutPolicy<S: GameState> {
    fn select(&mut self, state: &S, actions: &[S::Action], player_id: u8, rng: &mut Rng) -> usize;
}
```

| Struct | Trait | Description |
|--------|-------|-------------|
| `NoPruner` | `ConstraintPruner` | Allows all tokens (baseline) |
| `BinaryScreeningPruner<P>` | `ScreeningPruner` | Adapter: `ConstraintPruner` → binary `{0.0, 1.0}` relevance |
| `NoScreeningPruner` | `ScreeningPruner` | Returns 1.0 for everything (no penalty) |
| `RandomRolloutPolicy` | `RolloutPolicy` | Uniform random action selection |
| `ActionSpaceLog` | — | Per-tick branching factor metrics for analysis |

### LEO All-Goals Traits (Plan 155)

Goal-conditioned RL traits for agents that learn all goals simultaneously (LEO — Learning Everything Omnisciently). Feature-gated:
- `leo_all_goals` — `LeoHead`, `AllGoalsUpdate`, `sigmoid_bounded_q`
- `dual_leo` — additionally `DualLeoMixer`, `AutocurriculumSampler`

```rust
// Feature gate: leo_all_goals
pub trait LeoHead {
    fn all_goals_q(&self, state: &[f32]) -> Vec<f32>;  // [goals × actions] flattened
    fn goal_count(&self) -> usize;
    fn action_count(&self) -> usize;
    fn q_for_goal(&self, all_q: &[f32], goal: usize) -> &[f32]; // slice into row
}

pub trait AllGoalsUpdate {
    fn td_target(&self, rewards: &[f32], next_q: &[Vec<f32>], gamma: f32) -> Vec<f32>;
    fn loss(predicted: &[Vec<f32>], target: &[f32]) -> f32;  // MSE over goals
}

// Feature gate: dual_leo
pub trait DualLeoMixer {
    fn mix(&self, q_leo: &[f32], q_uvfa: &[f32], alpha: f32) -> Vec<f32>;  // α·Q_LEO + (1-α)·Q_UVFA
    fn combine_into(&self, out, q_leo, q_uvfa, alpha);  // ActingMode dispatch
    fn acting_mode(&self) -> ActingMode;   // Lc | LeoOnly | UvfaOnly | Max | Min
    fn alpha_schedule(&self) -> AlphaSchedule;  // Fixed(a) | LinearAnneal { start, end }
    fn bc_config(&self) -> Option<BcConfig>;    // BC regularization for Dual LEO PPO
}

pub trait AutocurriculumSampler {
    fn sample_goal(&self, rng: &mut Rng) -> usize;
    fn observe_goal(&mut self, goal: usize);
    fn update_goals_seen(&self, obs_batch, all_goals, current_mask) -> Vec<bool>;
    fn goals_completed_this_episode(&self) -> usize;
}
```

**Architecture note:** Implementors should use BatchRenorm (r_max=3, d_max=5, warmup=1000) rather than standard BatchNorm, for stability with highly off-policy replay data.

Re-exported from both `katgpt-core` and `katgpt-rs`.

LoRA application is fused in-place after each projection: `output += (α/r) × B @ (A @ input)`. Zero intermediate buffers — the delta accumulates directly into the output.

## Parallax Attention (`crates/katgpt-core/src/parallax_attn.rs`, Plan 135)

Streaming covariance-correction layer on top of tiled online-softmax flash attention. Reduces the regression gap between local-linear kernel attention and full SDPA from O(N²) computation to O(N) outer products via column-sum factorization.

**Formula:**
```
o_PLX = o_SA − gate_scale · Σ_KV · ρ
```
- `o_SA` = attention output under chosen activation (Softmax or Sigmoid)
- `Σ_KV = Σ_j c_j · v_j ⊗ k_j^T` — KV cross-covariance from column sums (O(N) outer products)
- `ρ = W_R · x` — learned probe from input residual via projection

**Column-Sum Factorization:** Computes `c_j = Σ_i p(i,j)` (column marginals) in one pass over the Q×K score matrix, then reconstructs Σ_KV as N outer products — avoiding the full N×N weight matrix.

```rust
pub enum ParallaxActivation {
    Softmax,  // Gaussian-like with attention sinks (backward compat)
    Sigmoid,  // Default — sink-free, better numerical stability
}

pub struct ParallaxConfig {
    pub gate_scale: f32,           // correction scaling (anneal to 0.0 to disable)
    pub zero_init: bool,           // W_R starts zeroed → plain attention fallback
    pub activation: ParallaxActivation,
}

pub struct ParallaxScratch {
    // Pre-allocated scratch buffers for zero-alloc hot paths
    // rho, col_sums, scores, sigma_kv, pv_buf, correction
}
```

| Function | Description |
|----------|-------------|
| `compute_rho(r_proj, x, out)` | ρ = W_R · x — matrix-vector product |
| `parallax_correction(sigma_kv, rho, out)` | correction = Σ_KV · ρ |
| `tiled_attention_parallax_forward(q, k, v, output, seq_len, head_dim, scale, r, x, config, scratch)` | Full forward: tiled flash attention + Parallax covariance correction |
| `ParallaxScratch::new(seq_len, head_dim)` | Pre-allocate scratch buffers |
| `ParallaxScratch::ensure_capacity(seq_len, head_dim)` | Resize when dimensions change |

**Zero-Init Fallback:** When `gate_scale = 0.0` or `W_R` is zero, skips Σ_KV computation entirely and falls back to plain tiled attention — performance equals base `tiled_attention_forward`.

**Sigmoid vs Softmax:** Sigmoid normalization (`σ(q·k·s) / Σ σ(q·k·s)`) avoids attention sinks common in softmax, improving COR (Covariance Over Representation) capacity. Softmax variant is provided for backward compatibility.

**Feature gate:** `parallax_attn` (requires `tiled_attention`, `newton_schulz`). **opt-in** — requires Muon-trained W_R weights.

## Emotion Vector Inference (`src/pruners/emotion_vector.rs`, Plan 162, Research 144)

Zero-cost read of emotion directions from mid-layer residual-stream activations during speculative decoding. Based on Anthropic Transformer Circuits research showing linear emotion representations causally drive behavior (desperation steering → 14× reward-hacking increase at +0.1 offset).

**Core Idea:** Pre-compute direction vectors for valence, arousal, desperation, and calm during training/calibration. At decode time, each read is a single O(d) dot product per step — zero additional forward passes, no feature gate required (enabled by default if T7 GOAT proof shows <0.1% overhead).

```rust
pub struct EmotionDirections {
    // Pre-computed direction vectors [d_model] for each emotion axis
    // valence, arousal, desperation, calm
}

pub struct EmotionReading {
    pub valence: f32,       // positive/negative sentiment projection
    pub arousal: f32,       // high/low activation projection
    pub desperation: f32,   // reward-hacking early-warning signal
    pub calm: f32,          // inverse of desperation; inhibits risk-taking
}
```

| Method | Description |
|--------|-------------|
| `EmotionDirections::zeros(d_model)` | Create zero-initialized directions (placeholder) |
| `EmotionDirections::new(valence, arousal, desperation, calm)` | Constructor with dimension validation |
| `EmotionDirections::project(activation, direction)` → f32 | O(d) dot product, zero-alloc, `#[inline(always)]` |
| `EmotionDirections::read_emotions(activations)` → `EmotionReading` | Project activations onto all four directions |

**Integration with ReviewMetrics:** Five new atomic fields on `ReviewMetrics`: `emotion_valence_sum`, `emotion_arousal_sum`, `desperation_score_sum`, `calm_score_sum`, `emotion_count`. Methods:
- `record_emotion(&EmotionReading)` — accumulate emotion projection values
- `is_desperate_session(threshold)` — returns `true` when mean desperation exceeds threshold
- `emotion_profile_summary()` — formatted string for logging

**Desperation Monitor:** When a session's `desperation_score` exceeds a configurable threshold, it signals potential reward-hacking behavior — allowing SR²AM configurator or the bandit to adjust planning strategy before the DDTree commits to a high-risk path.

**From Research 144:** Anthropic found 171 emotion concepts in LLM activation space organized by valence (PC1: 26% variance) and arousal (PC2: 15% variance) axes. Causal steering of `desperation` in blackmail scenario: +0.05 → 22% → 72% rate (+50pp), +0.1 → 5% → 70% rate (14× increase). `calm` direction is protective: +0.05 → 0% blackmail.

**Plan 162 Phase Status:**
- Phase 1 ✅ — Infrastructure complete (EmotionDirections, ReviewMetrics integration)
- Phase 2 ⏳ — GOAT proof: T7 overhead (<0.1%), T8 desperation↔entropy correlation
- Phase 3 📋 — Integrate into SR²AM ConfiguratorContext as feature input

## FlashAR Anchor-Then-Fill (`src/speculative/flashar_anchor.rs`, Plan 166 T11)

Two-round strided decoding inspired by FlashAR's diagonal-step parallel pattern. Feature-gated behind `flashar_anchor` (requires `dllm`).

**Round 1 (Anchor):** AR predicts every S-th position (stride S). Few AR forward passes (`block_size / stride`) produce high-quality anchor tokens.

**Round 2 (Fill):** D2F decodes remaining positions with anchors pre-filled. Anchor positions start unmasked, reducing the denoising search space → fewer iterations for convergence.

```rust
/// Configuration for strided anchor-then-fill decoding.
pub struct AnchorConfig {
    pub stride: usize,  // predict every S-th position via AR. S=1 → pure AR, S=block_size → pure D2F
}

/// Result of the two-round anchor-then-fill decode.
pub struct AnchorFillResult { /* anchor tokens, fill tokens, iteration count */ }
```

| Method | Description |
|--------|-------------|
| `AnchorConfig::with_stride(S)` | Create config with stride (min 1) |
| `anchor_then_fill_decode(...)` | Full two-round decode: AR anchors → D2F fill |

**Stride Guidance:** S=2–4 for balanced anchor density. Higher S → more parallelism but lower anchor quality.

## FlashAR Consensus (`src/speculative/flashar_consensus.rs`, Plan 166)

Dual-path consensus with ternary thermal routing. Replaces tri_mode's prefix-match acceptance. Feature-gated behind `flashar_consensus` (requires `dllm`).

**Architecture:**
- Path H: AR/MTP draft → per-position tokens + confidence
- Path V: D2F block draft → per-position tokens + confidence
- Ternary consensus per position: +1 (H wins), 0 (AGREE → skip verify), -1 (V wins)

```rust
/// Thermal path assigned per position.
pub enum ThermalPath {
    Plasma,  // both agree, high confidence — accept immediately, zero verification
    Hot,     // one path wins, high confidence — accept winner
    Warm,    // one path wins, moderate confidence — AR spot-check
    Cold,    // both low confidence — fallback prefix-match verification
}

/// Configuration for thermal path router.
pub struct ConsensusConfig {
    pub plasma_threshold: f32,     // confidence threshold for PLASMA (default 0.7)
    pub hot_threshold: f32,       // confidence threshold for HOT (default 0.5)
    pub warm_threshold: f32,      // confidence threshold for WARM (default 0.3)
    pub use_ternary_gate: bool,   // use simd_ternary_matvec fusion gate (requires plasma_path)
}
```

| Method | Description |
|--------|-------------|
| `consensus_decode_block(...)` | Full dual-path consensus decode |
| `route_thermal_path(ternary, conf_h, conf_v, config)` → `ThermalPath` | Classify each position into thermal path |
| `ternary_consensus(token_h, conf_h, token_v, conf_v)` → (i8, token) | Compute ternary signal + winner |

## Budget Adaptation (`src/speculative/budget.rs`, Plan 167)

Compression-adaptive decode budget — uses PFlash scoring ratio (a free byproduct of prefill) to dynamically scale DDTree budget per-prompt. Feature-gated behind `budget_adaptation`.

```rust
/// Controls how the DDTree tree budget adapts per-prompt.
pub enum BudgetAdaptation {
    Off,          // fixed budget (default)
    Compression,  // scale by compression ratio r ∈ (0,1]: high r → complex → more budget
    Entropy,      // scale by first-marginal entropy (placeholder)
}
```

| Method | Description |
|--------|-------------|
| `adaptive_tree_budget(base, ratio, mode)` → usize | Derive per-prompt budget. Clamped to [base/2, base*2] |
| `compression_ratio(total_blocks, selected_blocks)` → f32 | Fraction of blocks that passed importance filter |

**Scaling curve (Compression mode):**
```text
r=0.0 → scale=0.5  (budget halved, simple prompt)
r=0.5 → scale=1.25 (budget slightly above base)
r=1.0 → scale=2.0  (budget doubled, complex prompt)
```

## ILC Distillation (`src/distill/ilc.rs`, Plan 164)

Iterative Latent Clustering — synonym-aware DDTree pruning. Distilled from arXiv:2605.27734 (Korchinski, Favero, Wyart). Feature-gated behind `ilc_distill`.

**Architecture:**
- Offline: episode data → `IlcClusterer` (k-means on cousin context vectors) → `SynonymMap`
- Online: `SynonymMap::lookup(state)` → ClusterId (O(1), no allocation) → DDTree skips synonym branches

```rust
/// Offline k-means clusterer for cousin context vectors.
pub struct IlcClusterer { /* config: IlcConfig */ }

/// O(1) cluster lookup at inference time.
pub struct SynonymMap {
    // centers[level][cluster_id] → centroid, assignment[level][hash(state)] → ClusterId
}

/// ScreeningPruner wrapper that boosts diversity across clusters.
pub struct SynonymAwarePruner<P: ScreeningPruner> { inner: P, synonym_map: SynonymMap, /* ... */ }
```

| Method | Description |
|--------|-------------|
| `IlcClusterer::new(config)` | Create offline clusterer |
| `IlcClusterer::fit(data)` → `SynonymMap` | Run level-by-level k-means (Algorithm 1) |
| `SynonymMap::lookup(state, level)` → `ClusterId` | O(1) inference-time lookup |
| `build_dd_tree_screened_synonyms(...)` | DDTree variant that skips synonym branches |

## Data Probe (`src/data_probe/`, Plan 141)

Controlled information-theoretic validation via a Markov chain with known ground-truth transition probabilities. Feature-gated behind `data_probe`.

```text
markov → nll → typical_set → claim
                  ↓
            dirichlet_energy
                  ↓
              geometry
```

| Module | Description |
|--------|-------------|
| `markov` | Dirichlet-sampled Markov chain generator with entropy rate targeting |
| `nll` | NLL computation against known chain |
| `typical_set` | Three-way regime classification (Conservative/Typical/Uncertain) |
| `dirichlet_energy` | Dirichlet Energy structural alignment diagnostic |
| `claim` | Claim card infrastructure for formal C1–C4 validation |
| `geometry` | Representation geometry diagnostics (Plan 151, Research 113) |

**Key types:** `MarkovChain`, `Regime` (Conservative/Typical/Uncertain), `ClaimCard`, `GeometryReport`, `ValidityVerdict`.

## Depth-Invariance Diagnostic (`crates/katgpt-core/src/depth_invariance.rs`, Plan 306)

Root-cause counterpart to four existing symptom-only detectors
(`BeliefRankPruner`, `GainCostLoopHalter`, `latent_functor/reestimation.rs`,
`micro_belief/coherence_bench.rs`). Modelless math over flattened `&[f32]`
state chains from any recursive latent-state kernel. Detects
`DepthSpecificRefinement` (monotonically growing magnitude — the paper's
primary signal), `Collapsed` (effective rank trending to 1), `DepthInvariant`
(flat magnitude + stable cos step), or `Insufficient` (`k+1 < min_samples`).

```text
recursive kernel: h_{t+1} = f(h_t, x_t)
        ↓ (capture k+1 chain)
 classify_chain → { magnitude_slope, mean_cos_step, effective_rank_slope, kind }
```

| Module | Description |
|--------|-------------|
| `depth_invariance.rs` | The primitive: `classify_chain`, `classify_chain_batched`, `Scratch`, `DepthInvarianceConfig`, `DepthInvarianceKind` (`#[repr(u8)]`), `MagnitudeRegularization` enum, `apply_magnitude_regularization`. Zero-alloc hot path (caller-owned `Scratch`). |
| `speculative/belief_drafter.rs::BeliefDrafter::audit_depth_invariance` | Plan 306 Phase 3 G2 audit hook (behind `depth_invariance` × `belief_drafter`). Also `capture_chain` for the G2c inference-time RmsNorm demonstration. |
| `micro_belief/attractor.rs::AttractorKernel::audit_depth_invariance` | Plan 306 Phase 4 G3a negative control (behind `depth_invariance` × `micro_belief`). Attractor clamps to `(-1,1)` → classifies `DepthInvariant`. |
| `micro_belief/leaky.rs::LeakyIntegrator::audit_depth_invariance` | Plan 306 Phase 4 audit hook. The shipped `LeakyIntegrator` also clamps to `[-1,1]`, so it too classifies `DepthInvariant`; the G3b positive control strips the clamp inline in the test (no kernel-level support needed). |

**GOAT verdict (Plan 306):** G1 (8 correctness tests) + G2 (paper finding
reproduced on random-init `BeliefDrafter`: `DepthSpecificRefinement`,
locked-drift cos > 0.999) + G3 (negative control on `AttractorKernel` +
positive control on inline unclamped leaky) + G4 (latency, aspirational —
classify_chain is O(k·d) so the ≤5% target only holds at d ≥ 1024 k=4) all
pass. **Fix is split**: for kernels we own (HLA, functor, micro_belief,
engram, Raven) `MagnitudeRegularization` is the modelless upstream fix; for
the frozen-pretrained `BeliefDrafter` MLP the fix requires retraining
(paper §4.4 Table 4: -56% acceptance for inference-time pin on pre-norm) →
`riir-train`.

**Disambiguation (T8.3):** this is a *different paper + mechanism* than
Sink-Aware Attention (`sink_aware_attn`, Plan 287, Research 258,
arXiv:2606.08105). The two are frequently confused:

| Aspect | `depth_invariance` (Plan 306) | `sink_aware_attn` (Plan 287) |
|---|---|---|
| Paper | arXiv:2605.09992 (Eldenk et al., *Attention Drift*) | arXiv:2606.08105 (Fesser et al., *A Unifying View of Attention Sinks*) |
| Mechanism | Drafter-side magnitude accumulation in the recursive residual | Target-side attention-sink classification (NOP vs Broadcast) |
| Diagnostic | `magnitude_slope` on the hidden-state chain | `value_norm_ratio` + `stable_rank_of_update` per attention head |
| Fix | Post-norm on the recursive residual (retrain for BeliefDrafter) | Dual-policy attention (sigmoid gate for NOP heads, regular for Broadcast) |

Feature gate: `depth_invariance` (opt-in). Promotion to default is a
deliberate parent decision pending — G1–G3 pass, G4 is aspirational.

## SkillOpt (`src/skill_opt/`, Plan 144)

Text-space skill optimization: deterministic edit → apply → gate → buffer → optimizer pipeline. Feature-gated behind `skill_opt`.

```text
edit → apply → gate → buffer → optimizer
       ↑                      |
       └── protected section   └── JSONL persistence
```

| Module | Description |
|--------|-------------|
| `edit` | `EditOp`, `EditSource`, `SkillEdit` — edit operations |
| `apply` | `apply_edits`, `ApplyResult` — deterministic text patching with budget + protected sections |
| `gate` | `ValidationGate`, `RejectedEdit` — accept/reject by score delta |
| `schedule` | `EditBudgetSchedule` — constant/linear/cosine/autonomous schedules |
| `buffer` | `RejectedEditBuffer` — FIFO ring buffer for negative examples |
| `optimizer` | `SkillOptimizer` trait, `Benchmark` trait, `ScoredTrajectory` |

## Proof Certificates (`src/proof_cert/`, Plan 145)

Hierarchical GOAT proof certificates with dependency chains, topological verification, and blake3 checksum integrity. Feature-gated behind `proof_cert`.

```rust
pub struct ProofCertificate { /* property, evidence, result, checksum */ }
pub struct ProofEvidence { /* supporting data for a proof claim */ }
pub enum ProofProperty { /* the property being verified */ }
pub enum ProofResult { /* Pass, Fail, Inconclusive */ }
```

| Module | Description |
|--------|-------------|
| `certificate` | `ProofCertificate`, `ProofEvidence`, `ProofProperty`, `ProofResult` |
| `chain` | `verify_proof_chain()` — topological verification of dependency chain |
| `serde_impls` | `load_certificates`, `save_certificates`, `verify_checksum` (blake3) |
| `macros` | Certificate generation macros |
| `wasm_certificates` | `generate_wasm_validator_certificates` — WASM certificate generation |

## CachePrune (`src/cache_prune/`, Plan 140)

SAT + rolling hash + sensitivity masking for KV cache analysis. All modelless — no training, no model changes. Feature-gated behind `cache_prune`. Reference: arXiv:2605.23640.

| Module | Description |
|--------|-------------|
| `sat` | `SummedAreaTable` — O(1) rectangular attention queries |
| `rolling_hash` | `RollingHash`, `CachedSegment`, `KvSegmentPool` — O(n) variable-length segment matching |
| `sensitivity` | `SensitivityDetector` trait, `StrictDetector`, `OpenDetector` — selective KV sharing |

## Hydra Budget (`src/pruners/hydra_budget.rs`, Research 148, Plan 165)

Hydra-Aware Adaptive Layer Budget — emergent self-repair layer skipping. Distills the Hydra Effect (arXiv:2307.15771, McGrath et al.) into adaptive layer skipping. Feature-gated behind `hydra_budget`.

Two modes: **modelless** (pre-computed profiles, zero overhead) and **model-based** (per-layer logit lens scoring, one matmul per layer).

```rust
/// Pre-computed set of layers to skip.
pub struct HydraSkipPlan {
    pub skip_layers: Vec<bool>,    // bitmask: skip_layers[l] = true ⇒ skip layer l
    pub cumulative_de: Vec<f32>,   // cumulative DE thresholds for early termination
    pub total_de: f32,             // total DE across all layers
}

/// Result of adaptive budget computation.
pub struct HydraBudgetResult {
    pub skipped: Vec<usize>,             // layers to skip
    pub early_exit_layer: Option<usize>, // early termination point
    pub savings_fraction: f32,           // estimated compute savings
}
```

| Method | Description |
|--------|-------------|
| `hydra_layer_skip(profiles, config)` → `HydraSkipPlan` | Compute skip plan from profiles |
| `hydra_budget_result(plan, n_layers)` → `HydraBudgetResult` | Derive budget result |

**Skip Rules:** Never skips layers with `backup_frequency > 0.1` (Hydra backups) or non-erasure layers with significant `mean_de`. Erasure MLPs skipped during draft if `skip_erasure_draft` is set.

## GEPA-D Reflective (`src/pruners/gepa_reflective.rs`)

Pareto bandit config evolution via reflective distillation. Evolves system-level config (rubric weights, template hints, bandit params) from MeMo trajectory reflection using Pareto-frontier bandit selection. Feature-gated behind `gepa_reflective` (requires `bandit`, `memo_reflections`).

**No gradient updates. No LoRA.** Config variants = bandit arms, reflection quality = reward.

```rust
/// A point in configuration space — one bandit arm.
pub struct ConfigVariant {
    pub rubric_preset: u8,             // index into RUBRIC_PRESETS (4 presets)
    pub epsilon_index: u8,             // quantised exploration rate (0.05..0.40)
    pub template_hint: u8,             // template hint index
    pub absorb_threshold_index: u8,    // quantised absorb threshold (0.1..0.7)
}
```

Total arms = 4 × 4 × 4 × 4 = 256. Uses UCB1 selection. Rubric presets: balanced, relevance-heavy, novelty-heavy, uniform.

## PhraseBoost (`src/pruners/phrase_boost.rs` + `phrase_trie.rs`, Plan 164)

Context trie phrase boosting for DDTree. Zero training cost — phrases provided at call site. Feature-gated behind `phrase_boost`.

```rust
/// Compact token-level trie — O(1) child lookup via Vec<Option<usize>>.
pub struct PhraseTrie { /* nodes: Vec<PhraseTrieNode>, vocab_size */ }

/// Wraps ScreeningPruner and adds phrase-based token boosting.
pub struct PhraseBoostPruner<P: ScreeningPruner> {
    inner: P,
    trie: PhraseTrie,
    boost_score: f32,    // normalized via boost_score / (1 + boost_score)
}
```

| Method | Description |
|--------|-------------|
| `PhraseTrie::new(vocab_size)` | Create empty trie |
| `PhraseTrie::insert(&mut self, token_ids)` | Add phrase |
| `PhraseTrie::advance(&self, states, token)` → `Vec<usize>` | Advance active states |
| `PhraseBoostPruner::new(inner, trie, boost_score)` | Create wrapper |

Boost is additive — inner pruner's relevance is preserved. Uses `RwLock<HashMap>` for interior mutability on the `&self` `relevance()` interface.

## Speculative/Types Additions (`src/speculative/types.rs`)

New types alongside the existing `DecodeStrategy`, `DraftEvent`, `RejectionReason`, etc.:

```rust
/// Controls DDTree budget adaptation per-prompt.
pub enum BudgetAdaptation {
    Off,          // fixed budget (default)
    Compression,  // scale by compression ratio
    Entropy,      // scale by entropy (placeholder)
}

/// Score reduction mode for block scoring.
pub enum ScoreReduction {
    SoftmaxSum,  // standard attention (default)
    MaxSim,      // late-interaction scoring (requires maxsim feature)
}

/// Configuration for PFlash block-sparse prefill scoring.
pub struct FlashPrefillConfig {
    pub block_size: usize,
    pub attention_sink: usize,
    pub window: usize,
    pub last_n_full: usize,
    pub tail_window: usize,
    pub alpha: f32,
    pub score_reduction: ScoreReduction,
    pub budget_adaptation: BudgetAdaptation,
}

/// Prefill compression mode.
pub enum PrefillMode {
    Off,     // never compress (default)
    Auto,    // compress when prompt length >= threshold
    Always,  // always compress
}

/// Block importance scores from PFlash scoring.
pub struct BlockScores {
    pub num_blocks: usize,
    pub block_size: usize,
    pub scores: Vec<f32>,
    pub selected: Vec<usize>,
}
```

`FlashPrefillConfig` provides named constructors: `default()`, `metal()`, `long_context()`, `short_context()`.

---

## Sense Composition (Plan 221)

KG Latent Octree NPC sense modules — compresses game domain KG triples into fixed-type ternary bit-plane sense modules. NPCs compose modules at spawn time and query at ~45ns/tick via bitwise dot-product.

### Key Types

```rust
/// Composable NPC sense module with ternary bit-plane projection
pub struct SenseModule {
    octree_bits: u64,        // Octree occupancy mask (8 octants)
    pos_bits: TernaryDir,    // Positive direction bit-plane
    neg_bits: TernaryDir,    // Negative direction bit-plane
    row_scale: Vec<f32>,     // Per-dimension scale
    confidence: f32,         // Module quality [0,1]
    kind: SenseKind,         // Sense type classification
    #[cfg(feature = "bake_precision")]
    precision: Option<PrecisionEntry>, // BAKE precision tracking
}

/// NPC Brain — composes sense modules and projects HLA state
pub struct NpcBrain {
    modules: Vec<SenseModule>,
    hla_cache: HlaCacheProxy,
    gm_overrides: Vec<SenseOverride>,
    autonomous: bool,
}

/// Sense kinds — classification of what a module senses
pub enum SenseKind {
    CommonSense, FighterSense, GameTheorySense,
    SpatialSense, SocialSense, SkillSense, Reserved,
}
```

### Architecture

- **SenseModule::project()** — Ternary bitwise dot → sigmoid → 8-dim HLA projection at ~45ns/tick
- **SenseOctreeBuilder** — Converts `KgEmbedding` → bit-plane octree occupancy + ternary direction vectors
- **SenseHotSwap** — Lock-free `AtomicPtr` swap with `AtomicBool` module lock
- **SenseTrialLog** — Bandit feedback for module quality, `decay_direction()` EMA adjusts confidence
- **SenseBatch** — Parallel batch projection for multiple NPCs (rayon when N>64)
- **SNSE Serialization** — Binary format with BLAKE3 verification for persistent sense state
- **GM Override** — `SenseOverride` pins specific senses or disables autonomous mode for scripted NPCs

Feature gate: `sense_composition` (opt-in, requires `plasma_path`, `domain_latent`).

---

## Shard Embedding (Plan 230)

Johnson-Lindenstrauss random orthogonal projection for O(1) cosine similarity shard lookup.

```rust
/// 64→8 random orthogonal projection matrix
pub struct JlProjectionMatrix {
    matrix: [[f32; STYLE_DIM]; EMBED_DIM], // 64×8
    hash: [u8; 32], // BLAKE3 commitment
}

/// Compressed shard embedding
pub struct ShardEmbedding([f32; 8]);
```

- **Construction**: Gram-Schmidt orthogonal rows from RNG seed → BLAKE3 commitment
- **Projection**: SIMD dot-product `style_weights[64] → embedding[8]`
- **Lookup**: `cosine_similarity()` and `dist_sq()` on [f32; 8]
- No feature gate — always compiled in `katgpt-core`

---

## SLoD Spectral Level-of-Detail Pruner (Plan 235)

Modelless KG resolution control via spectral heat diffusion on hyperbolic kNN graph Laplacians. GOAT G1–G6 all pass.

### Key Types

```rust
pub struct SlodConfig {
    pub knn_k: usize,           // kNN neighbors (default: 8)
    pub n_scales: usize,        // Number of scale tiers (default: 3)
    pub participation_threshold: f32,
    pub entropy_threshold: f32,
}

pub struct SlodOperator {
    config: SlodConfig,
    scale_boundaries: Vec<ScaleBoundary>,
}

pub struct SlodPruner { operator: SlodOperator }
impl ConstraintPruner for SlodPruner { ... }
```

### Architecture

1. **Poincaré ball geometry**: `poincare_distance()`, `log_map()`, `exp_map()`, `frechet_mean()`
2. **kNN Laplacian**: Build graph from KG embeddings → Laplacian L = D - A
3. **Jacobi eigendecomposition**: Extract eigenvalues/eigenvectors
4. **Multi-signal boundary scan**: Participation ratio + diffusion entropy + spectral concentration → MAD peak picker
5. **Tier routing**: `SlodPruner::is_valid()` routes to appropriate resolution tier O(1)

Feature gate: `slod` (default-ON, depends on `spectral_hierarchy`).

---

## Schema Centroid (Plan 237)

Per-class embedding centroids for informed KG entity initialization. GOAT 7/7.

```rust
pub struct CentroidStats {
    pub mean: [f32; 8],
    pub std_dev: [f32; 8],
}

pub struct SchemaCentroidCache {
    centroids: papaya::HashMap<u64, CentroidStats>,
}
```

- **Construction**: Per-class `mean` + `std_dev` computed once from KG snapshots
- **Initialization**: `schema_init_entity()` → average class centroids + `γ·σ_c ⊙ noise` perturbation
- **Fallback**: Random `[-0.5, 0.5]` init if class not found
- **Cross-feature bridge**: When `bake_precision` enabled → `schema_init_with_precision()` uses informed prior

Feature gate: `schema_centroid` (default-ON, requires `dep:papaya`).

---

## BAKE Precision-Gated Bayesian Embedding (Plan 236)

Per-dimension precision tracking for KG embeddings. GOAT 10/10 but **demoted to opt-in** (drift 4.7% vs 30% target).

```rust
pub struct PrecisionEntry {
    pub lambda: [f32; 8],  // Per-dimension precision
    pub mu: [f32; 8],      // Per-dimension mean
}

pub struct BakePrecisionStore {
    entries: papaya::HashMap<u64, PrecisionEntry>,
}

pub struct BakeSession { /* lifecycle: begin → observe × N → end */ }
```

- **Bayesian update**: `λ_new = λ_old + λ_obs`, `μ_new = (λ_old ⊙ μ_old + λ_obs ⊙ obs) / λ_new`
- **Regularization**: `β · √(λ ⊙ (μ_current - μ_old)²)` penalty
- **O(8) arithmetic**: Zero-alloc, SIMD-friendly
- **Session lifecycle**: `begin()` → `observe()` × N → `end()` writes back to store

Feature gate: `bake_precision` (opt-in, requires `dep:papaya`, `sense_composition`).

---

## RatPlus Recurrence Bridge (Plan 225)

RAT+ recurrence bridge via GDN2 state for modelless dilated inference.

Feature gate: `rat_plus_bridge` (opt-in).

---

## MicroRecurrentBeliefState (`crates/katgpt-core/src/micro_belief/`, Plan 276)

Per-entity implicit state-tracking kernel — small frozen recurrent kernels implementing
`s_t = f(s_{t-1}, x_t)` over a fixed-size latent belief vector, applied once per
(entity, tick). The belief vector is latent/local (never synced); a bridge projects it to
bounded raw scalars that cross the sync boundary. Full reference: `.docs/26_micro_belief.md`.

```rust
pub trait MicroRecurrentBeliefState: Send + Sync {
    fn dim(&self) -> usize;
    fn step(&self, state: &mut [f32], input: &[f32]);                 // zero-alloc
    fn project_to_scalars(&self, state: &[f32], directions: &[f32],   // latent→raw bridge
                          dim: usize, out: &mut [f32]);
    fn family(&self) -> RecurrenceFamily;                              // routing
}
```

| Module | Family | Status |
|---|---|---|
| `micro_belief/attractor.rs` | A — `s_t = 2·σ(W_s·s + W_x·x + b) − 1` | Opt-in experiment (G1.4 + G2.1 FAIL) |
| `micro_belief/latent_thought.rs` | B — K iters of Family A per tick | Opt-in experiment (G1.6: K=1 bit-identical to A) |
| `micro_belief/leaky.rs` | C — monotone additive, `±max_delta` clamp | **Promotable** — byte-identical to `evolve_hla` |
| `micro_belief/snapshot.rs` | freeze/thaw — BLAKE3-committed weights | Opt-in (per-NPC personality divergence) |
| `micro_belief/bridge.rs` | `project_to_scalars` — sigmoid(dot) | Shared bridge (all families delegate) |
| `leaky_core.rs` (ungated) | shared `leaky_step` primitive | Single source of truth for Family C math |

**GOAT verdict:** trait unification + `LeakyIntegrator` are the promotable outputs.
Attractor demoted to Gain (G1.4 latency ~273ns; G2.1 coherence 569× more flip-flops than
leaky). The `evolve_hla` refactor (Phase 2) made `evolve_hla` delegate to the ungated
`leaky_core::leaky_step` — zero behavior change, no `micro_belief` feature coupling.

Feature gate: `micro_belief` (opt-in).