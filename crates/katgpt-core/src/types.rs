// Shared configuration, RNG, and math utilities.
// Superset of types from both katgpt-rs and riir-engine projects.

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Adaptive depth tier mapping to layer count (Plan 284).
/// Reuses ThermalPath naming convention from FlashAR Consensus (Plan 166).
///
/// | Tier     | Layers     | When                              |
/// |----------|------------|-----------------------------------|
/// | Plasma   | 1          | High entropy, easy positions      |
/// | Hot      | 2          | Medium entropy, standard tactics  |
/// | Warm     | all        | Low entropy, complex positions    |
/// | Cold     | all+verify | Critical, full verification       |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DepthTier {
    /// Easy positions: empty board, forced moves. 1 layer.
    Plasma = 0,
    /// Moderate: standard tactics. 2 layers.
    Hot = 1,
    /// Complex: all layers + spot-check verification.
    Warm = 2,
    /// Critical: all layers + full verification.
    Cold = 3,
}

impl DepthTier {
    /// Returns the maximum number of transformer layers to execute for this tier.
    pub fn max_layers(&self, total_layers: usize) -> usize {
        match self {
            DepthTier::Plasma => 1.min(total_layers),
            DepthTier::Hot => 2.min(total_layers),
            DepthTier::Warm => total_layers,
            DepthTier::Cold => total_layers,
        }
    }
}

/// Attention mode for HLA (Higher-order Linear Attention).
///
/// - `Standard`: SDPA with KV cache (default, backward-compatible).
/// - `Hla`: Symmetric second-order linear attention — O(1) per-token memory.
/// - `Ahla`: Asymmetric second-order linear attention — lower state cost than symmetric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HlaMode {
    #[default]
    Standard,
    /// Symmetric second-order: SK, CQV, mQ accumulators.
    Hla,
    /// Asymmetric second-order: PKV, mK accumulators.
    Ahla,
}

/// Attention mode for forward passes.
///
/// - `Causal`: Standard autoregressive — only attend to positions ≤ current (default).
/// - `Bidirectional`: Attend to ALL positions — used for dLLM masked prediction (Plan 066).
/// - `BlockCausal`: Bidirectional within current block, causal across blocks — D2F student.
/// - `SpKv`: SP-KV self-pruned key-value attention (Plan 070).
/// - `SpKvQuant`: SP-KV + Quantized KV fusion (Plan 070 Phase 3, Task T12).
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionMode {
    #[default]
    Causal,
    /// Full bidirectional: all positions see all positions (teacher mode).
    Bidirectional,
    /// Block-causal: bidirectional within block, causal across blocks (student mode).
    BlockCausal,
    /// SP-KV self-pruned key-value attention (Plan 070).
    /// Learns which KV pairs to retain via utility prediction.
    /// Gate bias = log(u) during training, 0|-inf during inference.
    SpKv,
    /// SP-KV + Quantized KV fusion (Plan 070 Phase 3, Task T12).
    /// Selective write (SP-KV utility gating) + lossy quantize (any QuantizedKVCache backend).
    /// Two-stage compression: only useful KV pairs kept, those compressed to 2-4 bits/coord.
    SpKvQuant,
    /// DashAttention: adaptive sparse hierarchical attention via α-entmax routing (Plan 106).
    /// Replaces fixed-budget top-k block selection with adaptive support selection.
    /// Learned chunk summaries via head_cls vectors.
    DashAttn,
}

/// Model architecture selector for forward pass dispatch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelArchitecture {
    #[default]
    Generic,
    Gemma2,
    Llama,
    /// Hybrid DeltaNet/Attention model (e.g., Qwen 3.5, Kimi Linear).
    /// Uses per-layer config to determine DeltaNet vs standard attention.
    /// Plan 182: Luce Megakernel Distill — DeltaNet GPU Inference.
    #[cfg(feature = "deltanet_inference")]
    QwenDeltaNet,
}

/// Attention projection configuration.
/// Controls whether K and V projections share weights (Q-K=V tying).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum AttentionProjection {
    /// Standard Q, K, V (3 projections, full KV cache)
    #[default]
    Full,
    /// Q-K=V: K and V share projection (2 projections, K-only cache).
    /// 50% KV cache reduction, ~3% perplexity cost.
    /// Post-hoc weight merging: W_kv = (W_k + W_v) / 2.
    SharedKV,
}

/// KV cache layout (derived from AttentionProjection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CacheLayout {
    /// Store both K and V (standard)
    KV,
    /// Store K only, V = K at read (SharedKV)
    K,
}

/// Weight storage dtype (affects loading and dequantization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum WeightDtype {
    #[default]
    F32,
    F16,
    BF16,
}

// ---------------------------------------------------------------------------
// Delta Routing (Plan 097, Research 061)
// ---------------------------------------------------------------------------

/// Delta routing mode — cross-layer information flow via delta vectors.
/// Research 061: Delta Attention Residuals (Plan 097).
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeltaRoutingMode {
    /// No delta routing (default).
    #[default]
    Off,
    /// Delta Block: accumulate deltas within blocks of `block_size` layers.
    /// B+1 sources per routing decision. ~20% throughput overhead.
    DeltaBlock,
    /// Delta Attention Residuals: per-sublayer delta routing.
    /// 2L sources. 69% throughput reduction at L=36. Use only for research.
    DeltaAttnRes,
}

/// Configuration for delta routing (Plan 097, Research 061).
///
/// Fields ordered by descending alignment to minimize padding:
/// usize (8B) → repr(u8) enum (1B) — 16 bytes total, no wasted padding.
#[derive(Clone, Copy, Debug)]
pub struct DeltaRoutingConfig {
    /// Block size for DeltaBlock mode (number of layers per block).
    /// Default: 4. Paper recommends B=4.
    pub block_size: usize,
    /// Routing mode.
    pub mode: DeltaRoutingMode,
}

impl Default for DeltaRoutingConfig {
    fn default() -> Self {
        Self {
            block_size: 4,
            mode: DeltaRoutingMode::Off,
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaNet Inference (Plan 182: Luce Megakernel Distill)
// ---------------------------------------------------------------------------

/// Per-layer type for hybrid DeltaNet/Attention models.
/// Each layer is either a standard attention layer or a DeltaNet recurrent layer.
#[cfg(feature = "deltanet_inference")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum DeltaNetLayerType {
    /// Standard multi-head attention with KV cache.
    #[default]
    Attention,
    /// DeltaNet linear recurrent layer (fast recurrent update, no KV cache needed).
    DeltaNet,
}

impl DeltaRoutingConfig {
    pub fn delta_block(block_size: usize) -> Self {
        Self {
            mode: DeltaRoutingMode::DeltaBlock,
            block_size,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.mode != DeltaRoutingMode::Off
    }
}

// ---------------------------------------------------------------------------
// DashAttention Config (Plan 106, Research 68)
// ---------------------------------------------------------------------------

/// Configuration for DashAttention adaptive sparse hierarchical attention.
/// Controls α-entmax routing, chunk summarization, and routing bias.
///
/// Fields ordered by descending alignment to minimize padding:
/// usize (8B) → f32 (4B) → bool (1B) — 24 bytes total, no wasted padding.
#[derive(Clone, Copy, Debug)]
pub struct DashAttnConfig {
    /// Chunk size for block-level attention (default: 64).
    pub chunk_size: usize,
    /// α parameter for entmax. Only α=1.5 supported (quadratic, closed-form).
    pub alpha: f32,
    /// Scaling factor γ applied to chunk logits before entmax (default: 1.0).
    pub scaling_factor: f32,
    /// Prior strength σ for routing bias (default: 1e6, weak prior).
    pub sigma: f32,
    /// Whether to estimate diagonal attention contribution (default: true).
    /// Tail-packed after f32 group to avoid bool-between-f32 padding.
    pub estimate_diagonal: bool,
}

impl Default for DashAttnConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            alpha: 1.5,
            scaling_factor: 1.0,
            sigma: 1e6,
            estimate_diagonal: true,
        }
    }
}

// ---------------------------------------------------------------------------
// RTPurbo Retrieval Head Sparse Decode (Plan 126, Research 86)
// ---------------------------------------------------------------------------

/// Head role classification for RTPurbo sparse decode.
///
/// Only ~15% of attention heads ("retrieval heads") need full long-context access.
/// The remaining ~85% ("local heads") attend only to local context + attention sinks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum RetrievalHeadRole {
    /// Local head — sliding window + sink tokens only, no full KV scan.
    #[default]
    Local,
    /// Retrieval head — low-dim projection + dynamic top-p token selection.
    Retrieval,
}

/// Configuration for RTPurbo retrieval head sparse decode.
///
/// Feature gate: `rt_turbo` (opt-in, requires `dash_attn`).
/// Adds head-wise retrieval/local classification + dynamic top-p token selection
/// for decode-phase sparse attention. Complements DashAttention's α-entmax block
/// routing with per-head specialization.
///
/// Must pass 6/6 GOAT proofs before default-on promotion.
///
/// Fields ordered by descending alignment to minimize padding:
/// usize (8B) → f32 (4B) — no padding between groups.
#[derive(Clone, Copy, Debug)]
pub struct RtTurboConfig {
    /// Low-dimensional projection size for pre-RoPE scoring (default: 16).
    /// Paper ablation: dim=16 is the sweet spot for low-frequency retrieval.
    pub low_dim: usize,
    /// Sliding window size for local heads (default: 8192).
    pub sliding_window: usize,
    /// Number of attention sink tokens always retained for local heads (default: 4).
    pub sink_tokens: usize,
    /// Block size for block-level top-p variant (default: 64).
    /// Should match `DashAttnConfig::chunk_size` for consistent routing.
    pub block_size: usize,
    /// Fraction of heads classified as retrieval heads (default: 0.15).
    /// Paper ablation: 15% is optimal balance of accuracy vs sparsity.
    pub retrieval_head_ratio: f32,
    /// Cumulative attention mass threshold for dynamic top-p selection (default: 0.9).
    /// Paper ablation: top-p=0.9 preserves >93% attention mass at 97% sparsity.
    pub top_p: f32,
}

impl Default for RtTurboConfig {
    fn default() -> Self {
        Self {
            low_dim: 16,
            sliding_window: 8192,
            sink_tokens: 4,
            block_size: 64,
            retrieval_head_ratio: 0.15,
            top_p: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// LT2 Looped Inference (Plan 108, Research 73)
// ---------------------------------------------------------------------------

/// Looped transformer mode — weight-shared layer repetition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum LoopMode {
    /// Standard single-pass (no looping).
    #[default]
    None,
    /// Weight-shared looping: same layers applied T times.
    /// Effective depth = n_layer × loop_count.
    WeightShared { loop_count: usize },
    /// Training-free loop: ODE-refined sub-stepping over a window of layers.
    /// No extra parameters — pure inference-time retrofit (Plan 136).
    TrainingFree,
}

/// Hybrid attention pattern for looped inference.
/// Controls which layers use full SDPA vs linear attention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HybridPattern {
    /// All layers use the same attention mode.
    #[default]
    Uniform,
    /// Depth-level interleave: every Nth layer uses full SDPA.
    /// e.g., Interleave { full_ratio: 5 } = every 5th layer is full.
    /// Paper optimal: 1:4 ratio (full_ratio=5).
    Interleave { full_ratio: usize },
    /// Bookend: first and last layers are full, middle is linear.
    Bookend,
}

/// Head-specific sigmoid gate after SDPA, before Wo.
/// Zero-initialized → starts at sigmoid(0) = 0.5 (neutral multiplicative identity).
#[derive(Clone)]
pub struct SdpaOutputGate {
    /// Gate weights: [n_heads * head_dim, dim].
    /// Zero-init so gate starts at sigmoid(0) = 0.5.
    pub w_gate: Vec<f32>,
}

impl SdpaOutputGate {
    /// Allocate zeroed gate weights.
    pub fn new(n_heads: usize, head_dim: usize, dim: usize) -> Self {
        Self {
            w_gate: vec![0.0; n_heads * head_dim * dim],
        }
    }

    /// Apply sigmoid-gated projection to attention output.
    ///
    /// Computes: gate[i] = sigmoid(W_gate[i] · attn_out), then attn_out[i] *= gate[i].
    /// Zero-init weights produce sigmoid(0) = 0.5 for all (neutral half-pass).
    /// Paper reference: +0.3–0.5 avg points on zero-shot benchmarks.
    pub fn forward(&self, attn_out: &mut [f32], dim: usize, temp: &mut [f32]) {
        let n = attn_out.len();
        debug_assert!(temp.len() >= n, "temp buffer too small");
        debug_assert!(self.w_gate.len() >= n * dim, "gate weights too small");

        // Step 1: Compute gate signal = sigmoid(W_gate @ attn_out)
        // Batch matvec then batch sigmoid avoids per-element loop overhead
        crate::simd::simd_matvec(temp, &self.w_gate, attn_out, n, dim);

        // SIMD sigmoid: temp = -temp, exp, then 1/(1+exp)
        crate::simd::simd_scale_inplace(&mut temp[..n], -1.0);
        crate::simd::simd_exp_inplace(&mut temp[..n]);
        crate::simd::simd_add_scalar_inplace(&mut temp[..n], 1.0);
        // temp now = 1 + exp(-x), invert: temp = 1/temp = sigmoid
        crate::simd::simd_reciprocal_inplace(&mut temp[..n]);

        // Step 2: Apply gate elementwise via SIMD scale-mul (fused)
        // attn_out[i] *= temp[i] is element-wise multiply
        // Use simd_scale_mul_inplace with scale=1.0: attn[i] = temp[i] * attn[i] * 1.0
        crate::simd::simd_scale_mul_inplace(attn_out, &temp[..n], 1.0);
    }
}

/// Per-loop residual scaling gate.
/// h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
/// Zero-init so first iteration is h̃^(1) (no residual from "previous").
#[derive(Clone)]
pub struct ResidualGate {
    /// Per-loop gates: [loop_count, dim].
    /// Each ρ_τ is element-wise, zero-init.
    pub gates: Vec<f32>,
}

impl ResidualGate {
    /// Allocate zeroed residual gates.
    pub fn new(loop_count: usize, dim: usize) -> Self {
        Self {
            gates: vec![0.0; loop_count * dim],
        }
    }
}

// ---------------------------------------------------------------------------
// SR²AM Configurator Bandit (Plan 112, Research 076)
// ---------------------------------------------------------------------------

/// SR²AM Configurator decision — learned per-turn planning regulation.
///
/// The configurator selects one of these arms per inference turn based on
/// context (domain + entropy bin). UCB1 balances exploration vs exploitation.
///
/// - `PlanNew`: reset tree, full budget allocation (high uncertainty, new sub-problem)
/// - `PlanExtend`: keep tree, extend depth by one level (moderate uncertainty, continuing)
/// - `PlanSkip`: skip tree search, direct token sampling (low uncertainty, confident)
#[cfg(feature = "sr2am_configurator")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum PlanningDecision {
    /// Reset tree, full budget allocation (high uncertainty, new sub-problem).
    PlanNew,
    /// Keep tree, extend depth by one level (moderate uncertainty, continuing).
    PlanExtend,
    /// Skip tree search, direct token sampling (low uncertainty, confident).
    PlanSkip,
    /// Activate SpecHop continuous speculation with k speculative threads (Plan 131).
    /// Selected when speculator latency α is low and tool ratio β is moderate.
    SpecHop { k: usize },
    /// Harness update: AbsorbCompress promote + HotSwapPruner reload (Plan 163 T5).
    /// Selected when harness has plateaued and a compressed arm set may improve.
    #[cfg(feature = "sia_feedback")]
    HarnessUpdate,
    /// Weight update: trigger riir-gpu training step on accumulated TrialLog (Plan 163 T6).
    /// Selected when stall detection fires — reward plateau suggests weights need updating.
    #[cfg(feature = "sia_feedback")]
    WeightUpdate,
}

/// Context key for configurator bandit — coarse entropy binning.
///
/// Entropy is discretized into 10 bins via `floor(entropy * 10.0)` clamped to 0..9.
/// Combined with domain index, this provides context-aware arm selection.
#[cfg(feature = "sr2am_configurator")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfiguratorContext {
    /// Domain index from bandit infrastructure.
    pub domain: usize,
    /// Coarse entropy bin: `floor(entropy * 10.0)`, clamped to 0..9.
    /// u8 — values are 0..9, packed after usize to avoid padding.
    pub entropy_bin: u8,
    /// Coarse desperation bin: `floor(desperation * 10.0)`, clamped to 0..9.
    /// Plan 162 T11: emotion vector desperation score as additional context.
    /// 0 = not desperate, 9 = highly desperate.
    pub desperation_bin: u8,
    /// Coarse epiplexity bin: `floor(epiplexity * 10.0)`, clamped to 0..9.
    /// Plan 130 T4: structural information content (S_T) as additional context.
    /// 0 = no structure detectable, 9 = highly structured.
    pub epiplexity_bin: u8,
}

#[cfg(feature = "sr2am_configurator")]
impl ConfiguratorContext {
    /// Create context without desperation information (legacy compatibility).
    ///
    /// Sets `desperation_bin` to 0 (not desperate). Use `with_desperation()`
    /// when emotion vector data is available.
    pub fn new(domain: usize, entropy_bin: usize) -> Self {
        Self {
            domain,
            entropy_bin: (entropy_bin.min(9)) as u8,
            desperation_bin: 0,
            epiplexity_bin: 0,
        }
    }

    /// Set the desperation bin from a raw desperation score.
    ///
    /// `floor(desperation * 10.0)`, clamped to 0..9.
    pub fn with_desperation(mut self, desperation: f32) -> Self {
        self.desperation_bin = ((desperation * 10.0).floor() as u8).min(9);
        self
    }

    /// Set the epiplexity bin from a raw epiplexity score (S_T).
    ///
    /// `floor(epiplexity * 10.0)`, clamped to 0..9.
    /// S_T measures structural information content — higher values indicate
    /// more structure that a bounded observer can extract from the data.
    pub fn with_epiplexity(mut self, epiplexity: f32) -> Self {
        self.epiplexity_bin = ((epiplexity * 10.0).floor() as u8).min(9);
        self
    }

    /// Create context from entropy and epiplexity signals.
    ///
    /// Convenience constructor that bins both entropy (H_T proxy) and
    /// epiplexity (S_T structural information) in one call.
    /// `desperation_bin` defaults to 0.
    pub fn from_entropy_epiplexity(domain: usize, entropy: f32, epiplexity: f32) -> Self {
        let entropy_bin = ((entropy * 10.0).floor() as u8).min(9);
        let epiplexity_bin = ((epiplexity * 10.0).floor() as u8).min(9);
        Self {
            domain,
            entropy_bin,
            desperation_bin: 0,
            epiplexity_bin,
        }
    }

    /// Discretize epiplexity (S_T) into a coarse bin index.
    ///
    /// `floor(epiplexity * 10.0)`, clamped to 0..9.
    pub fn epiplexity_bin(epiplexity: f32) -> u8 {
        ((epiplexity * 10.0).floor() as u8).min(9)
    }
}

// ---------------------------------------------------------------------------
// EqR Convergence Selection (Plan 119)
// ---------------------------------------------------------------------------

/// Selection strategy for width-scaled rollouts (EqR convergence-based selection).
///
/// Maps to [`WidthSelectionMode`](crate::speculative::dd_tree::WidthSelectionMode) at runtime.
/// This enum lives in `katgpt-core` so Config can reference it without depending on
/// the speculative decode module.
///
/// - `BestQ`: Highest cumulative relevance (PTRM default, no behavior change)
/// - `MajorityVote`: Most common path across rollouts (mode@K)
/// - `Top1Converged`: Smallest final residual ∥p_{d+1} − p_d∥ (EqR proxy)
/// - `BtRank`: Pairwise Bradley-Terry ranking (requires `bt_rank` feature)
///
/// **Precondition:** `Top1Converged` is only reliable after landscape shaping
/// (RI + NI training). See Research 079 (EqR) for theoretical justification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConvergenceSelector {
    /// Select rollout with highest cumulative relevance score (PTRM Q-head analog).
    #[default]
    BestQ,
    /// Select the most frequent path across all rollouts (mode@K, majority vote).
    MajorityVote,
    /// Select rollout with smallest marginal-change residual ∥p_{d+1} − p_d∥ (EqR proxy).
    Top1Converged,
    /// Pairwise Bradley-Terry ranking across rollouts (requires `bt_rank` feature).
    BtRank,
}

// ---------------------------------------------------------------------------
// Wall Attention — Diagonal Forget Gates Replacing RoPE (Plan 173)
// ---------------------------------------------------------------------------

/// Wall Attention configuration (Plan 173, Research: Wall Attention paper).
///
/// Wall replaces RoPE with diagonal forget gates applied as factorized Q/K rescaling:
/// `q̃_i = exp(P_i) ⊙ q_i`, `k̃_j = exp(-P_j) ⊙ k_j`.
/// This means attention kernels are UNCHANGED — they receive pre-rescaled Q and K.
///
/// Only applicable to Wall-trained models (requires W_g gate projection weights).
#[derive(Clone, Debug)]
#[cfg(feature = "wall_attention")]
pub struct WallConfig {
    /// Gate bias initialization value. Default 6.0 = open gate (vanilla attention behavior).
    /// Lower values → more active forgetting (gate_bias=0 → retention ≈ 0.62).
    pub gate_bias: f32,
    /// Maximum gate log-sigmoid clamp value. Default 0.87 (matches paper).
    /// Gates are clamped to (-gate_max, 0] after log-sigmoid.
    pub gate_max: f32,
    /// Use key-projected gate variant (derive gate from K projection).
    /// Preferred for zero KV cache overhead — gate is computed from key, not hidden state.
    pub use_key_projected: bool,
}

#[cfg(feature = "wall_attention")]
impl Default for WallConfig {
    fn default() -> Self {
        Self {
            gate_bias: 6.0,
            gate_max: 0.87,
            use_key_projected: true,
        }
    }
}

#[cfg(feature = "wall_attention")]
impl WallConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Collapse-Aware Adaptive Thinking (Plan 212)
// ---------------------------------------------------------------------------

/// Per-instance adaptive budget for collapse-aware thinking.
///
/// Controls when mid-reasoning early exit triggers and how efficiency rewards
/// are shaped. Feature-gated behind `collapse_aware_thinking`.
#[derive(Clone, Debug)]
pub struct ThinkingBudget {
    /// Maximum thinking tokens before forced termination.
    pub max_tokens: u32,
    /// Hesitation count threshold τ — collapse triggers when exceeded.
    pub collapse_threshold: u32,
    /// Efficiency–accuracy trade-off for reward shaping.
    /// Higher γ penalizes longer traces more aggressively.
    /// Range: [0.0, 1.0].
    pub efficiency_gamma: f32,
}

#[cfg(feature = "collapse_aware_thinking")]
impl Default for ThinkingBudget {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            collapse_threshold: 3,
            efficiency_gamma: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Transformer model configuration — superset of both katgpt-rs and riir-engine.
///
/// Fields are ordered by descending alignment to minimize padding:
/// usize/u64 → f64 → enums (usize-discriminant) → f32 → Vec → u16 → u8/bool.
#[derive(Clone)]
pub struct Config {
    // --- usize / pointer-sized fields (8-byte aligned) ---
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
    pub n_layer: usize,
    pub n_kv_head: usize,
    pub bos_token: usize,
    pub draft_lookahead: usize,
    pub tree_budget: usize,
    pub parallel_threshold: usize,
    pub lora_rank: usize,
    pub early_exit_patience: usize,
    pub mtp_activation_threshold: usize,
    pub mtp_cluster_vocab_threshold: usize,
    pub mtp_shared_kv_prompt_threshold: usize,
    pub mtp_cluster_size: usize,
    /// Minimum expected output tokens for MTP speculative decoding.
    /// If remaining tokens < this threshold, MTP is skipped (single-token path).
    /// Prevents MoE overhead on short texts (Plan 117 Phase 2).
    pub mtp_min_output_tokens: usize,
    /// Top-K cluster selection for clustered LM head (Plan 117 T20).
    /// When K > 1, compute logits for tokens in top-K clusters instead of just top-1.
    /// Default 1 = backward compatible (single cluster = current behavior).
    pub mtp_cluster_topk: usize,
    pub mask_token: usize,
    pub sp_kv_window: usize,
    pub sp_kv_predictor_hidden: usize,
    pub width_rollouts: usize,
    pub d2f_block_size: usize,
    /// Number of last layers to sum before LM head. 0 = disabled (standard).
    /// (Plan 104: Research 68)
    pub mls_layers: usize,

    // --- f64 (8-byte aligned) ---
    pub rms_norm_eps: f64,

    // --- f32 (4-byte aligned) ---
    pub sp_kv_predictor_lr_mult: f32,
    pub temperature: f32,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    // Screening Pruner (Plan 021)
    pub screening_threshold: f32,
    // Sparse MLP (Plan 022)
    pub sparse_threshold: f32,
    // Early exit (Plan 026: AutoTTS)
    pub early_exit_gap: f32,
    pub hla_decay: f32,
    pub rope_theta: f32,
    pub attn_logit_softcapping: f32,
    pub final_logit_softcapping: f32,
    pub sp_kv_threshold: f32,
    pub early_stop_threshold: f32,
    // Parallax Attention (Plan 135: Parameterized Local Linear Attention)
    /// Parallax covariance correction gate scale. 0.0 = disabled (pure softmax),
    /// 1.0 = full correction. Only meaningful when `parallax_attn` feature is enabled
    /// and R projection weights are loaded.
    pub parallax_gate_scale: f32,
    /// Desperation score threshold for emotion-aware session flagging (Plan 162 T12).
    /// When the mean desperation projection exceeds this value, `is_desperate_session()` returns true.
    /// Default: 0.5 (moderate desperation). Range: [0.0, 1.0].
    pub emotion_desperation_threshold: f32,

    // --- Vec (pointer-sized, 8-byte aligned) ---
    pub lora_targets: Vec<String>,

    // --- #[repr(u8)] enums (1-byte) + bool fields (1-byte), tail-packed ---
    // HLA Attention (Plan 057: Higher-order Linear Attention)
    pub hla_mode: HlaMode,
    // Gemma 2 architecture fields (Plan 087)
    pub model_arch: ModelArchitecture,
    // D2F Discrete Diffusion Forcing (Plan 066)
    pub attention_mode: AttentionMode,
    // EqR Convergence Selection (Plan 119)
    pub convergence_selector: ConvergenceSelector,
    // LT2 Looped Inference Pipeline (Plan 108, Research 73)
    pub loop_mode: LoopMode,
    pub hybrid_pattern: HybridPattern,
    pub weight_dtype: WeightDtype,
    pub hla_normalize: bool,
    pub rms_norm_offset: bool,
    pub tied_embeddings: bool,
    pub use_rope: bool,
    pub post_norm: bool,
    pub gated_attn: bool,
    /// Whether W_R starts zeroed (true = recover exact softmax at init).
    pub parallax_zero_init: bool,

    // --- Hydra Adaptive Layer Budget (Research 148, Plan 165) ---
    /// Per-layer Hydra importance profiles. Empty = disabled.
    /// Populated from calibration data via `calibrate_profiles()`.
    #[cfg(feature = "hydra_budget")]
    pub hydra_profiles: Vec<super::HydraLayerProfile>,

    // --- DeltaNet Inference (Plan 182: Luce Megakernel Distill) ---
    /// Per-layer type map: DeltaNet vs standard Attention.
    /// Length = n_layer. Empty = all layers are Attention (backward compatible).
    /// Only used when model_arch = QwenDeltaNet.
    #[cfg(feature = "deltanet_inference")]
    pub layer_types: Vec<DeltaNetLayerType>,
    /// Depthwise conv kernel size for DeltaNet layers (typically 4).
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_conv_kernel_size: usize,
    /// Recurrence state dimension per head (key_dim * value_dim, typically 128*128 = 16384).
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_state_dim: usize,
    /// Linear attention key/value head dimension (128 for Qwen 3.5).
    /// Separate from `head_dim` which refers to full attention heads.
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_linear_head_dim: usize,
    /// Number of linear attention key heads (16 for Qwen 3.5).
    /// Separate from `n_head` which refers to full attention heads.
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_linear_n_heads: usize,
    /// Number of linear attention value heads (16 for Qwen 3.5).
    /// Usually equals `deltanet_linear_n_heads`.
    #[cfg(feature = "deltanet_inference")]
    pub deltanet_linear_n_value_heads: usize,

    // --- RiM Reasoning Buffer Slots (Plan 172, Research 192) ---
    /// Number of reasoning buffer blocks (K in RiM paper). 0 = disabled.
    #[cfg(feature = "rim_slots")]
    pub rim_block_count: usize,
    /// Tokens per buffer block (M in RiM paper). Default 2.
    #[cfg(feature = "rim_slots")]
    pub rim_tokens_per_block: usize,
    /// Token ID used for buffer positions (default: bos_token, reused as buffer).
    #[cfg(feature = "rim_slots")]
    pub rim_buffer_token: usize,

    // --- Wall Attention (Plan 173) ---
    /// Wall attention config. None = use RoPE/fallback.
    #[cfg(feature = "wall_attention")]
    pub wall_config: Option<WallConfig>,

    // --- Collapse-Aware Adaptive Thinking (Plan 212) ---
    /// Per-instance adaptive budget for collapse-aware thinking.
    #[cfg(feature = "collapse_aware_thinking")]
    pub collapse_budget: ThinkingBudget,

    // --- NextLat Belief-State Speculative Drafter (Plan 217) ---
    /// Path to `nextlat.bin` MLP weights. None = random init.
    #[cfg(feature = "belief_drafter")]
    pub belief_drafter_path: Option<String>,
    /// Entropy threshold for belief drafter variable-length stopping.
    /// Lower = more conservative drafts. Higher = more aggressive.
    /// Default: 2.0. Only used when `belief_drafter` feature is enabled.
    #[cfg(feature = "belief_drafter")]
    pub belief_drafter_entropy_threshold: f32,
}

impl Config {
    /// Micro GPT config matching [talos-vs-macbook](https://github.com/AlexCheema/talos-vs-macbook) reference:
    /// vocab=27, block=16, n_layer=1, n_head=4, n_embd=16, head_dim=4,
    /// RMSNorm (no learnable gain), ReLU MLP (4x), no biases, untied lm_head.
    pub fn micro() -> Self {
        Self {
            vocab_size: 27,
            block_size: 16,
            n_embd: 16,
            n_head: 4,
            head_dim: 4,
            mlp_hidden: 64,
            n_layer: 1,
            n_kv_head: 4,
            bos_token: 26,
            temperature: 0.5,
            draft_lookahead: 8,
            tree_budget: 16,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: usize::MAX,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: usize::MAX,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: usize::MAX,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 8,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// Micro config with LoRA defaults (Plan 008).
    pub fn micro_lora() -> Self {
        let mut c = Self::micro();
        c.lora_rank = 4;
        c.lora_alpha = 8.0;
        c.lora_dropout = 0.0;
        c.lora_targets = vec![
            "q".into(),
            "k".into(),
            "v".into(),
            "o".into(),
            "mlp1".into(),
            "mlp2".into(),
        ];
        c
    }

    /// Micro config for Discrete Diffusion Language Model training (Plan 068: D2F).
    /// Bidirectional attention by default, mask_token = vocab_size - 1.
    pub fn micro_dllm() -> Self {
        Self {
            attention_mode: AttentionMode::Bidirectional,
            mask_token: 26,
            d2f_block_size: 8,
            ..Self::micro()
        }
    }

    /// Game config for Bomberman LoRA training (Plan 041).
    /// Tiny Transformer for board state → action prediction.
    /// 10-token vocab: 4 board cells (0-3) + 6 actions (4-9).
    /// 170-token sequences: 169 board cells + 1 action.
    /// ~18K params total, ~1.5K LoRA params (rank=4).
    pub fn game() -> Self {
        Self {
            vocab_size: 10,
            block_size: 170,
            n_embd: 32,
            n_head: 4,
            head_dim: 8,
            mlp_hidden: 128,
            n_layer: 1,
            n_kv_head: 4,
            bos_token: 0,
            temperature: 1.0,
            draft_lookahead: 0,
            tree_budget: 0,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: vec![
                "q".into(),
                "k".into(),
                "v".into(),
                "o".into(),
                "mlp1".into(),
                "mlp2".into(),
            ],
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: usize::MAX,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: usize::MAX,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: usize::MAX,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 8,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// Game config for Go 9×9 LoRA training (Plan 078).
    /// Tiny Transformer for board state → move prediction.
    /// 85-token vocab: 3 board cells (Empty=0, Black=1, White=2) + 81 positions (3..83) + 1 pass (84).
    /// 82-token sequences: 81 board cells + 1 action.
    /// ~16K params total, ~1.3K LoRA params (rank=4).
    pub fn game_go() -> Self {
        Self {
            vocab_size: 85,
            block_size: 82,
            n_embd: 32,
            n_head: 4,
            head_dim: 8,
            mlp_hidden: 128,
            n_layer: 1,
            n_kv_head: 4,
            bos_token: 0,
            temperature: 1.0,
            draft_lookahead: 0,
            tree_budget: 0,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: vec![
                "q".into(),
                "k".into(),
                "v".into(),
                "o".into(),
                "mlp1".into(),
                "mlp2".into(),
            ],
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: usize::MAX,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: usize::MAX,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: usize::MAX,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 8,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// Lightweight draft model for speculative decoding (~4× smaller than target).
    /// Same vocab/block to share embeddings, but embd=4, heads=2, mlp=16.
    pub fn draft() -> Self {
        Self {
            vocab_size: 27,
            block_size: 16,
            n_embd: 4,
            n_head: 2,
            head_dim: 2,
            mlp_hidden: 16,
            n_layer: 1,
            n_kv_head: 2,
            bos_token: 26,
            temperature: 0.5,
            draft_lookahead: 8,
            tree_budget: 16,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: usize::MAX,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: usize::MAX,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: usize::MAX,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 8,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// Small target model for multi-layer testing.
    /// vocab=4096, block=256, n_layer=4, n_head=4, n_embd=64, head_dim=16,
    /// MLP hidden=256.
    pub fn small_target() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 64,
            n_head: 4,
            head_dim: 16,
            mlp_hidden: 256,
            n_layer: 4,
            n_kv_head: 4,
            bos_token: 0,
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 64,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: 128,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: 16,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 16,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// GQA draft config: 8 Q heads, 2 KV heads (4:1 ratio, 4× KV cache reduction).
    pub fn gqa_draft() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 64,
            n_head: 8,
            head_dim: 8,
            mlp_hidden: 256,
            n_layer: 4,
            n_kv_head: 2,
            bos_token: 0,
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 64,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: 128,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: 16,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 16,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// BPE tokenizer config for Rust source code.
    /// vocab=4096, block=256, n_layer=1, n_head=4, n_embd=32, head_dim=8,
    /// MLP hidden=128.
    pub fn bpe() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 32,
            n_head: 4,
            head_dim: 8,
            mlp_hidden: 128,
            n_layer: 1,
            n_kv_head: 4,
            bos_token: 1,
            temperature: 0.8,
            draft_lookahead: 8,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 32,
            mtp_cluster_vocab_threshold: 4096,
            mtp_shared_kv_prompt_threshold: 64,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: 16,
            mtp_cluster_topk: 8,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 16,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// BPE draft model (smaller for speculative decoding).
    /// Same vocab/block as bpe(), but embd=16, heads=2, mlp=64.
    pub fn bpe_draft() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 16,
            n_head: 2,
            head_dim: 8,
            mlp_hidden: 64,
            n_layer: 1,
            n_kv_head: 2,
            bos_token: 1,
            temperature: 0.8,
            draft_lookahead: 8,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 16,
            mtp_cluster_vocab_threshold: 4096,
            mtp_shared_kv_prompt_threshold: 64,
            mtp_cluster_size: 512,
            mtp_min_output_tokens: usize::MAX,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Generic,
            rms_norm_eps: 1e-5,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: false,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::F32,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 16,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// Gemma 2 2B config for real model inference (Plan 087).
    /// hidden_size=2304, intermediate_size=9216, vocab=256000, layers=26,
    /// heads=8, kv_heads=4, head_dim=256, max_seq=8192.
    /// Uses GeGLU MLP, RoPE, RMSNorm offset, tied embeddings, post-norm.
    pub fn gemma2_2b() -> Self {
        Self {
            vocab_size: 256000,
            block_size: 8192,
            n_embd: 2304,
            n_head: 8,
            head_dim: 256,
            mlp_hidden: 9216,
            n_layer: 26,
            n_kv_head: 4,
            bos_token: 2, // Gemma 2 BOS token
            temperature: 0.8,
            draft_lookahead: 0,
            tree_budget: 0,
            parallel_threshold: 8192,
            lora_rank: 0,
            lora_alpha: 1.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.0,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 0,
            mtp_cluster_vocab_threshold: 256000,
            mtp_shared_kv_prompt_threshold: 8192,
            mtp_cluster_size: 1024,
            mtp_min_output_tokens: 16,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::Gemma2,
            rms_norm_eps: 1e-6,
            rms_norm_offset: true,
            tied_embeddings: true,
            use_rope: true,
            rope_theta: 10000.0,
            post_norm: true,
            attn_logit_softcapping: 50.0,
            final_logit_softcapping: 30.0,
            weight_dtype: WeightDtype::BF16,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 16,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            layer_types: Vec::new(),
            #[cfg(feature = "deltanet_inference")]
            deltanet_conv_kernel_size: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_state_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_head_dim: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_heads: 0,
            #[cfg(feature = "deltanet_inference")]
            deltanet_linear_n_value_heads: 0,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// Config for Qwen 3.5-0.8B hybrid DeltaNet/Attention model (Plan 182).
    ///
    /// Typical layout: early layers use DeltaNet (linear recurrence, no KV cache),
    /// later layers use standard attention. The `layer_types` vec specifies per-layer.
    /// If `layer_types` is empty, all layers default to Attention (backward compatible).
    #[cfg(feature = "deltanet_inference")]
    pub fn qwen_deltanet(n_layer: usize, layer_types: Vec<DeltaNetLayerType>) -> Self {
        let n_head = 16;
        let head_dim = 128;
        let n_embd = n_head * head_dim; // 2048
        let mlp_hidden = n_embd * 4; // 8192 (SwiGLU: gate+up = 2× mlp_hidden)
        let n_kv_head = n_head; // MHA (no GQA for 0.8B)

        Self {
            vocab_size: 151936,
            block_size: 32768,
            n_embd,
            n_head,
            head_dim,
            mlp_hidden,
            n_layer,
            n_kv_head,
            bos_token: 151643, // Qwen BOS
            temperature: 0.8,
            draft_lookahead: 0,
            tree_budget: 0,
            parallel_threshold: 8192,
            lora_rank: 0,
            lora_alpha: 1.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
            sparse_threshold: 0.0,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 0,
            mtp_cluster_vocab_threshold: 151936,
            mtp_shared_kv_prompt_threshold: 32768,
            mtp_cluster_size: 1024,
            mtp_min_output_tokens: 16,
            mtp_cluster_topk: 1,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
            model_arch: ModelArchitecture::QwenDeltaNet,
            rms_norm_eps: 1e-6,
            rms_norm_offset: false,
            tied_embeddings: false,
            use_rope: true,
            rope_theta: 10000.0,
            post_norm: false,
            attn_logit_softcapping: 0.0,
            final_logit_softcapping: 0.0,
            weight_dtype: WeightDtype::BF16,
            mask_token: 0,
            attention_mode: AttentionMode::Causal,
            sp_kv_window: 128,
            sp_kv_threshold: 0.5,
            sp_kv_predictor_hidden: 0,
            sp_kv_predictor_lr_mult: 5.0,
            width_rollouts: 1,
            early_stop_threshold: 0.0,
            convergence_selector: ConvergenceSelector::default(),
            d2f_block_size: 16,
            mls_layers: 0,
            loop_mode: LoopMode::None,
            hybrid_pattern: HybridPattern::Uniform,
            gated_attn: false,
            parallax_gate_scale: 0.0,
            emotion_desperation_threshold: 0.5,
            parallax_zero_init: true,
            #[cfg(feature = "hydra_budget")]
            hydra_profiles: Vec::new(),
            layer_types,
            deltanet_conv_kernel_size: 4,
            deltanet_state_dim: head_dim * head_dim, // 128 × 128 = 16384 per head
            deltanet_linear_head_dim: head_dim,
            deltanet_linear_n_heads: n_head,
            deltanet_linear_n_value_heads: n_kv_head,
            #[cfg(feature = "rim_slots")]
            rim_block_count: 0,
            #[cfg(feature = "rim_slots")]
            rim_tokens_per_block: 2,
            #[cfg(feature = "rim_slots")]
            rim_buffer_token: 0,
            #[cfg(feature = "wall_attention")]
            wall_config: None,
            #[cfg(feature = "collapse_aware_thinking")]
            collapse_budget: ThinkingBudget::default(),
            #[cfg(feature = "belief_drafter")]
            belief_drafter_path: None,
            #[cfg(feature = "belief_drafter")]
            belief_drafter_entropy_threshold: 2.0,
        }
    }

    /// Validate config consistency. Returns Err with message on invalid config.
    pub fn validate(&self) -> Result<(), String> {
        if !self.n_head.is_multiple_of(self.n_kv_head) {
            return Err(format!(
                "n_head ({}) must be divisible by n_kv_head ({})",
                self.n_head, self.n_kv_head
            ));
        }
        // Gemma 2 intentionally has q_dim != n_embd (e.g., 8*256=2048 != 2304)
        // LLaMA with GQA may also have q_dim != n_embd
        // QwenDeltaNet also has q_dim == n_embd but is excluded for forward compat
        let arch_exempt = match self.model_arch {
            ModelArchitecture::Gemma2 | ModelArchitecture::Llama => true,
            _ => {
                #[cfg(feature = "deltanet_inference")]
                if self.model_arch == ModelArchitecture::QwenDeltaNet {
                    // layer_types length must match n_layer when non-empty
                    if !self.layer_types.is_empty() && self.layer_types.len() != self.n_layer {
                        return Err(format!(
                            "layer_types length ({}) must match n_layer ({})",
                            self.layer_types.len(),
                            self.n_layer
                        ));
                    }
                    // deltanet_state_dim must be head_dim^2
                    let expected = self.head_dim * self.head_dim;
                    if self.deltanet_state_dim != expected {
                        return Err(format!(
                            "deltanet_state_dim ({}) must equal head_dim^2 ({})",
                            self.deltanet_state_dim, expected
                        ));
                    }
                    true
                } else {
                    false
                }
                #[cfg(not(feature = "deltanet_inference"))]
                false
            }
        };
        if !arch_exempt && self.n_head * self.head_dim != self.n_embd {
            return Err(format!(
                "n_head ({}) * head_dim ({}) must equal n_embd ({})",
                self.n_head, self.head_dim, self.n_embd
            ));
        }
        if self.n_kv_head * self.head_dim > self.n_embd {
            return Err(format!(
                "n_kv_head ({}) * head_dim ({}) must not exceed n_embd ({})",
                self.n_kv_head, self.head_dim, self.n_embd
            ));
        }
        // MTP thresholds must be consistent (only for Generic arch; Gemma 2 and Llama don't use MTP)
        if self.model_arch == ModelArchitecture::Generic && self.mtp_cluster_size == 0 {
            return Err("mtp_cluster_size must be > 0".into());
        }
        if self.mtp_cluster_topk == 0 {
            return Err("mtp_cluster_topk must be >= 1".into());
        }
        Ok(())
    }

    /// Apply per-domain inference overrides, returning a new Config.
    ///
    /// Total number of buffer tokens when RiM slots are active: K × M.
    /// Returns 0 when disabled (rim_block_count == 0).
    #[cfg(feature = "rim_slots")]
    #[inline]
    pub fn rim_total_buffer_tokens(&self) -> usize {
        if self.rim_block_count == 0 {
            0
        } else {
            self.rim_block_count * self.rim_tokens_per_block
        }
    }

    /// Whether RiM buffer slots are active.
    #[cfg(feature = "rim_slots")]
    #[inline]
    pub fn rim_enabled(&self) -> bool {
        self.rim_block_count > 0
    }

    /// Whether Wall Attention is active (Plan 173).
    /// True when feature is enabled AND config has wall_config set.
    #[cfg(feature = "wall_attention")]
    pub fn wall_enabled(&self) -> bool {
        self.wall_config.is_some()
    }

    /// `None` fields are left unchanged; `Some` fields replace the current value.
    /// Used by the router to inject domain-specific budgets from TOML config.
    pub fn with_overrides(mut self, overrides: &InferenceOverrides) -> Self {
        if let Some(v) = overrides.tree_budget {
            self.tree_budget = v;
        }
        if let Some(v) = overrides.draft_lookahead {
            self.draft_lookahead = v;
        }
        if let Some(v) = overrides.parallel_threshold {
            self.parallel_threshold = v;
        }
        if let Some(v) = overrides.screening_threshold {
            self.screening_threshold = v;
        }
        if let Some(v) = overrides.temperature {
            self.temperature = v;
        }
        if let Some(v) = overrides.sparse_threshold {
            self.sparse_threshold = v;
        }
        if let Some(v) = overrides.early_exit_patience {
            self.early_exit_patience = v;
        }
        if let Some(v) = overrides.early_exit_gap {
            self.early_exit_gap = v;
        }
        if let Some(v) = overrides.mtp_activation_threshold {
            self.mtp_activation_threshold = v;
        }
        if let Some(v) = overrides.mtp_cluster_vocab_threshold {
            self.mtp_cluster_vocab_threshold = v;
        }
        if let Some(v) = overrides.mtp_shared_kv_prompt_threshold {
            self.mtp_shared_kv_prompt_threshold = v;
        }
        if let Some(v) = overrides.mtp_cluster_size {
            self.mtp_cluster_size = v;
        }
        if let Some(v) = overrides.mtp_min_output_tokens {
            self.mtp_min_output_tokens = v;
        }
        if let Some(v) = overrides.mtp_cluster_topk {
            self.mtp_cluster_topk = v;
        }
        if let Some(v) = overrides.sp_kv_threshold {
            self.sp_kv_threshold = v;
        }
        if let Some(v) = overrides.width_rollouts {
            self.width_rollouts = v;
        }
        if let Some(v) = overrides.early_stop_threshold {
            self.early_stop_threshold = v;
        }
        if let Some(v) = overrides.convergence_selector {
            self.convergence_selector = v;
        }
        if let Some(v) = overrides.mls_layers {
            self.mls_layers = v;
        }
        // SR²AM horizon truncation override (Plan 112 T11)
        if let Some(v) = overrides.max_plan_horizon {
            self.draft_lookahead = self.draft_lookahead.min(v);
        }
        // Hydra Adaptive Layer Budget overrides (Research 148, Plan 165)
        // Applied via HydraBudgetConfig at call site, not directly on Config.
        // The overrides fields exist on InferenceOverrides for downstream consumption.
        self
    }
}

// ---------------------------------------------------------------------------
// InferenceOverrides
// ---------------------------------------------------------------------------

/// Override DTO for applying per-domain inference budget to a [`Config`].
///
/// All fields are `Option` — `None` means "keep Config's current value".
/// This is a plain struct (no serde) to keep `katgpt-core` dependency-free
/// from router/TOML types. Conversion from the router's `InferenceBudget`
/// happens at the router boundary.
///
/// Note: `decode_strategy` is NOT included here because it depends on
/// project-specific types. Each project handles it at the call site.
///
/// See Plan 026 (AutoTTS Dynamic Inference Budget).
#[derive(Debug, Clone, Default)]
// Fields ordered by descending alignment to minimize padding:
// Option<usize>/Option<PathBuf> (16/32 bytes) → Option<f32> (8 bytes) →
// Option<#[repr(u8)] enum> (2 bytes).
pub struct InferenceOverrides {
    // --- Option<usize> (16 bytes each, 8-byte aligned) ---
    pub tree_budget: Option<usize>,
    pub draft_lookahead: Option<usize>,
    pub parallel_threshold: Option<usize>,
    pub early_exit_patience: Option<usize>,
    // MTP Drafter overrides (Plan 055: Gemma 4 MTP)
    pub mtp_activation_threshold: Option<usize>,
    pub mtp_cluster_vocab_threshold: Option<usize>,
    pub mtp_shared_kv_prompt_threshold: Option<usize>,
    pub mtp_cluster_size: Option<usize>,
    /// Minimum expected output tokens for MTP (Plan 117 T15).
    /// When overridden, skips MTP when remaining tokens < threshold.
    pub mtp_min_output_tokens: Option<usize>,
    /// Top-K cluster selection for clustered LM head (Plan 117 T22).
    /// When K > 1, compute logits for tokens in top-K clusters instead of just top-1.
    pub mtp_cluster_topk: Option<usize>,
    // PTRM width scaling (Plan 083)
    pub width_rollouts: Option<usize>,
    // MLS Multi-Layer Sum override (Plan 104)
    pub mls_layers: Option<usize>,
    // SR²AM horizon truncation override (Plan 112 T11)
    pub max_plan_horizon: Option<usize>,

    // --- Option<PathBuf> (32 bytes, 8-byte aligned) ---
    // Drafter LoRA path (Plan 117: MTP LoRA Drafter)
    pub drafter_lora_path: Option<std::path::PathBuf>,

    // --- Option<f32> (8 bytes each, 4-byte aligned) ---
    pub screening_threshold: Option<f32>,
    pub temperature: Option<f32>,
    pub sparse_threshold: Option<f32>,
    pub early_exit_gap: Option<f32>,
    // SP-KV inference-time threshold knob (Plan 070)
    pub sp_kv_threshold: Option<f32>,
    pub early_stop_threshold: Option<f32>,

    // --- Option<#[repr(u8) enum> (2 bytes each, 1-byte aligned) ---
    // EqR Convergence Selection (Plan 119)
    pub convergence_selector: Option<ConvergenceSelector>,

    // --- Hydra Adaptive Layer Budget (Research 148, Plan 165) ---
    /// Override Hydra skip threshold.
    #[cfg(feature = "hydra_budget")]
    pub hydra_skip_threshold: Option<f32>,
    /// Override Hydra erasure-skip-draft flag.
    #[cfg(feature = "hydra_budget")]
    pub hydra_skip_erasure_draft: Option<bool>,

    // --- Adaptive Depth Tier (Plan 284) ---
    /// Override depth tier for layer count capping at inference time.
    /// When set, caps the layer loop to tier.max_layers().
    /// None = use all layers (backward compatible).
    pub depth_tier: Option<DepthTier>,
}

impl Default for Config {
    fn default() -> Self {
        Self::micro()
    }
}

// ---------------------------------------------------------------------------
// KV dimension helper
// ---------------------------------------------------------------------------

/// KV dimension: total float count per token in KV cache.
#[inline(always)]
pub fn kv_dim(config: &Config) -> usize {
    config.n_kv_head * config.head_dim
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

/// XorShift64 PRNG — deterministic per seed.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Construct from a 64-bit seed.
    ///
    /// Applies one round of SplitMix64 mixing to decorrelate low-entropy seeds
    /// (Issue 296). Without this, small seeds like `0`, `1`, `2`, `42` keep the
    /// XorShift64 state small enough that the first `next()` has zero upper 24
    /// bits, so `uniform()` returns exactly `0.0` — which sits on the left
    /// boundary of any inverse-CDF sampler and deterministically selects the
    /// first token with nonzero mass. SplitMix64 is the standard remedy for
    /// XorShift "small seed" weakness: it diffuses any input (including 0)
    /// into a well-distributed 64-bit state in O(1).
    ///
    /// Cost: ~3 mul/shift per `Rng::new`. Negligible — `new()` is called once
    /// per inference/training session, not in hot loops.
    pub fn new(seed: u64) -> Self {
        // SplitMix64 finalizer (Steele/Lea/Leiserson — see Java SplittableRandom).
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let state = z ^ (z >> 31);
        // XorShift64 with state == 0 is an absorbing state (stuck at 0 forever).
        // SplitMix64 never outputs 0 for any u64 input in practice, but guard
        // defensively: the cost is one branch on construction, not on hot paths.
        Self {
            state: if state == 0 { 1 } else { state },
        }
    }

    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform [0, 1) — 24 bits of entropy (full f32 mantissa precision).
    ///
    /// Note: returns a value in the half-open interval [0, 1). The exact bound
    /// `0.0` is reachable in principle (24 zero bits) but, after the SplitMix64
    /// seed mixing in `Rng::new`, is no longer the deterministic first draw for
    /// small seeds. Samplers that require a strictly positive variate should
    /// still redraw on `r == 0.0` as a defensive belt — see Issue 296.
    #[inline(always)]
    pub fn uniform(&mut self) -> f32 {
        // Bit-manipulation trick: take upper 24 bits, OR into the 23-bit mantissa
        // position with exponent set to 0x3f80 (1.0), then subtract 1.0.
        // This gives exactly 24 bits of uniform random bits mapped to [0, 1).
        let bits = ((self.next() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }

    /// Standard normal via Box-Muller transform.
    #[inline(always)]
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests_rng {
    use super::*;

    /// Issue 296 regression: `Rng::new(s).uniform()` must not be exactly `0.0`
    /// on the first draw for any small seed. Before the SplitMix64 fix in
    /// `Rng::new`, seeds `{0,1,2,3,42}` all produced `uniform() == 0.0` first.
    #[test]
    fn first_uniform_is_nonzero_for_small_seeds() {
        for &seed in &[0u64, 1, 2, 3, 4, 5, 42, 100, 1337, 0xFFFF_FFFF] {
            let first = Rng::new(seed).uniform();
            assert_ne!(
                first, 0.0,
                "seed {seed}: first uniform() must not be 0.0 (Issue 296)"
            );
            assert!(first > 0.0 && first < 1.0, "seed {seed}: uniform() out of range: {first}");
        }
    }

    /// `Rng::new(0)` must still produce a usable (non-zero, evolving) state —
    /// XorShift64 with state 0 is stuck at 0 forever.
    #[test]
    fn new_zero_seed_is_not_stuck() {
        let mut rng = Rng::new(0);
        let a = rng.next();
        let b = rng.next();
        assert_ne!(a, 0, "Rng::new(0).next() must not be 0");
        assert_ne!(a, b, "Rng::new(0) must evolve");
    }

    /// Same seed → same sequence (determinism preserved after SplitMix64).
    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..16 {
            assert_eq!(a.uniform(), b.uniform());
        }
    }

    /// Different small seeds → different first draws (the original defect:
    /// seeds 1 and 2 produced identical sampler output because both yielded 0.0).
    #[test]
    fn different_small_seeds_diverge_immediately() {
        let r1 = Rng::new(1).uniform();
        let r2 = Rng::new(2).uniform();
        let r3 = Rng::new(3).uniform();
        assert_ne!(r1, r2, "seeds 1 and 2 must differ on first draw");
        assert_ne!(r1, r3, "seeds 1 and 3 must differ on first draw");
        assert_ne!(r2, r3, "seeds 2 and 3 must differ on first draw");
    }

    /// Lightweight χ² goodness-of-fit on the first 65_536 uniforms: bin into
    /// 256 equal-width buckets and check the statistic is within reason.
    /// This is not a full DIEHARD suite, but catches catastrophic bias.
    #[test]
    fn uniform_is_well_distributed() {
        const BUCKETS: usize = 256;
        const N: usize = 65_536;
        let mut rng = Rng::new(42);
        let mut counts = [0u32; BUCKETS];
        for _ in 0..N {
            let u = rng.uniform();
            let idx = ((u * BUCKETS as f32) as usize).min(BUCKETS - 1);
            counts[idx] += 1;
        }
        let expected = N as f64 / BUCKETS as f64;
        let chi_sq: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d
            })
            .sum::<f64>()
            / expected;
        // 255 degrees of freedom: 99% critical value ≈ 330.09; 1% ≈ 188.33.
        // Use a generous upper bound of 400 to avoid flakes from V8/CI variance.
        assert!(
            chi_sq < 400.0,
            "χ²={chi_sq:.1} exceeds 400 (255 DoF, 99.99%ile ≈ 336) — distribution is biased"
        );
    }
}

// ---------------------------------------------------------------------------
// Math utilities — SIMD-accelerated
// ---------------------------------------------------------------------------

/// In-place softmax. Handles empty slices gracefully.
/// Three-pass: find max → shift+exp+sum (fused) → normalize.
#[inline(always)]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find max for numerical stability (SIMD-accelerated)
    let max_val = crate::simd::simd_max_f32(x);

    // Pass 2: subtract max (SIMD-accelerated)
    crate::simd::simd_add_scalar_inplace(x, -max_val);

    // Pass 3: SIMD exp + sum (fused — saves one full buffer traversal vs separate exp+sum)
    let sum: f32 = crate::simd::simd_exp_sum_inplace(x);
    let inv_sum = 1.0 / sum;
    crate::simd::simd_scale_inplace(x, inv_sum);
}

/// In-place softmax with temperature scaling: `softmax(x / temperature)`.
///
/// Fuses the temperature division into the exp computation, saving one full pass
/// vs separate `for p /= temp; softmax(x)`.
///
/// `inv_temp` should be `1.0 / temperature` — compute once, pass to every call.
#[inline(always)]
pub fn softmax_scaled(x: &mut [f32], inv_temp: f32) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find max for numerical stability (SIMD-accelerated)
    let max_val = crate::simd::simd_max_f32(x);

    // Pass 2: shift and apply temperature in one fused SIMD pass
    crate::simd::simd_fused_sub_scale_inplace(x, max_val, inv_temp);

    // Pass 3: SIMD exp + sum (fused — saves one full buffer traversal vs separate exp+sum)
    let sum: f32 = crate::simd::simd_exp_sum_inplace(x);
    let inv_sum = 1.0 / sum;
    crate::simd::simd_scale_inplace(x, inv_sum);
}

/// In-place RMSNorm (no learnable gain).
/// Two-pass: compute sum-of-squares, then scale.
#[inline(always)]
pub fn rmsnorm(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: sum of squares (SIMD-accelerated)
    let sum_sq = crate::simd::simd_sum_sq(x, x.len());

    // Pass 2: scale — stay f32 throughout to avoid f64 round-trip
    let inv_rms = 1.0 / (sum_sq / x.len() as f32 + 1e-5f32).sqrt();
    crate::simd::simd_scale_inplace(x, inv_rms);
}

/// GeGLU activation: hidden = gelu(gate) * up (elementwise).
/// Uses approximate GELU: gelu(x) ≈ x * sigmoid(1.702 * x).
/// `gate` and `up` are [mlp_hidden], output goes to `hidden`.
///
/// SIMD-accelerated: exp() computed via `simd_exp_inplace` on stack buffers.
#[inline(always)]
pub fn gegelu(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = -1.702 * gate[j] via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &gate[i..i + CHUNK], -1.702);
        // buf[j] = exp(-1.702 * gate[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = gate[j] * sigmoid(1.702 * gate[j]) * up[j]
        // SIMD: buf = 1 + buf, then buf = 1/buf (sigmoid), then fused gate*sigmoid*up
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        // Vectorized reciprocal: buf = sigmoid = 1/(1+exp(-1.702*gate))
        crate::simd::simd_reciprocal_inplace(&mut buf);
        // Fused: hidden = gate * up, then scale-multiply by sigmoid
        for j in 0..CHUNK {
            hidden[i + j] = gate[i + j] * up[i + j];
        }
        crate::simd::simd_scale_mul_inplace(&mut hidden[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for i in i..hidden.len() {
        let g = gate[i];
        let sigmoid = 1.0 / (1.0 + (-1.702 * g).exp());
        hidden[i] = g * sigmoid * up[i];
    }
}

/// GeGLU with tanh GELU approximation (Gemma 2 activation).
/// tanh GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// hidden[i] = gelu_tanh(gate[i]) * up[i]
///
/// SIMD-accelerated: exp() for tanh approximation computed via `simd_exp_inplace`.
#[inline(always)]
pub fn gegelu_tanh(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    const CHUNK: usize = 64;
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // (2.0f32 / π).sqrt()
    const SCALE_2: f32 = 1.595_769_2; // 2.0 * SQRT_2_OVER_PI
    let mut buf = [0.0f32; CHUNK];
    let mut buf2 = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = 0.044715 * g via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &gate[i..i + CHUNK], 0.044715);
        crate::simd::simd_scale_mul_inplace(&mut buf, &gate[i..i + CHUNK], 1.0); // buf = 0.044715 * g²
        // Finish cubic via SIMD: buf = 1 + 0.044715*g², then buf = scale_2 * g * (1 + 0.044715*g²)
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        crate::simd::simd_scale_mul_inplace(&mut buf, &gate[i..i + CHUNK], SCALE_2);
        // buf[j] = exp(2*inner[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = g * exp(2x) / (exp(2x) + 1) * up[j]
        // Compute denominator (exp + 1) via SIMD, then SIMD tanh + fused mul
        buf2[..CHUNK].copy_from_slice(&buf);
        crate::simd::simd_add_scalar_inplace(&mut buf2, 1.0); // buf2 = exp + 1
        // hidden = gate * up, then hidden *= tanh(inner)
        for j in 0..CHUNK {
            // Branch-free tanh: exp(2x) / (exp(2x) + 1) via division
            buf[j] /= buf2[j];
            hidden[i + j] = gate[i + j] * up[i + j];
        }
        crate::simd::simd_scale_mul_inplace(&mut hidden[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for i in i..hidden.len() {
        let g = gate[i];
        let inner = SQRT_2_OVER_PI * (g + 0.044715 * g * g * g);
        let gelu_val = 0.5 * g * (1.0 + inner.tanh());
        hidden[i] = gelu_val * up[i];
    }
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
/// Used in LLaMA, Mistral, and other LLaMA-family models for SwiGLU MLP.
///
/// SIMD-accelerated: exp() computed via `simd_exp_inplace` on stack buffers.
#[inline(always)]
pub fn silu(x: &mut [f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= x.len() {
        // buf[j] = -x[j] via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &x[i..i + CHUNK], -1.0);
        // buf[j] = exp(-x[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // x[j] = x[j] / (1 + exp(-x[j]))
        // SIMD: buf = 1 + exp(-x), then buf = 1/buf, then x *= buf elementwise
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        crate::simd::simd_reciprocal_inplace(&mut buf);
        crate::simd::simd_scale_mul_inplace(&mut x[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for v in x[i..].iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// SwiGLU activation: SiLU(gate) * up.
/// Used in LLaMA-family models (gate_proj and up_proj are separate weights).
/// Result stored in `hidden`: hidden[i] = silu(gate[i]) * up[i]
///
/// SIMD-accelerated: exp() computed via `simd_exp_inplace` on stack buffers.
#[inline(always)]
pub fn swiglu(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = -gate[j] via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &gate[i..i + CHUNK], -1.0);
        // buf[j] = exp(-gate[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = gate[j] / (1 + exp(-gate[j])) * up[j]
        // SIMD: buf = 1 + exp(-gate), then vectorized reciprocal + gate*up
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        // Vectorized reciprocal: buf = sigmoid = 1/(1+exp(-gate))
        crate::simd::simd_reciprocal_inplace(&mut buf);
        // Fused: hidden = gate * up, then scale-multiply by sigmoid
        for j in 0..CHUNK {
            hidden[i + j] = gate[i + j] * up[i + j];
        }
        crate::simd::simd_scale_mul_inplace(&mut hidden[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for i in i..hidden.len() {
        let g = gate[i];
        hidden[i] = g / (1.0 + (-g).exp()) * up[i];
    }
}

/// RMSNorm with learnable gamma (gain) vector.
/// Gemma 2 stores gamma as (gamma-1), so +1 is added during load.
/// `x` is normalized in-place then scaled by `gamma[i]`:
///   x[i] = gamma[i] * x[i] / sqrt(mean_sq + eps)
#[inline(always)]
pub fn rmsnorm_with_gamma(x: &mut [f32], gamma: &[f32]) {
    rmsnorm_with_gamma_eps(x, gamma, 1e-5)
}

/// RMSNorm with learnable gamma and configurable epsilon.
#[inline(always)]
pub fn rmsnorm_with_gamma_eps(x: &mut [f32], gamma: &[f32], eps: f64) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let sum_sq = crate::simd::simd_sum_sq(x, n);
    // Cast eps to f32 once — the f64 param is kept for API compat
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps as f32).sqrt();
    crate::simd::simd_scale_mul_inplace(x, gamma, inv_rms);
}

/// Matrix-vector multiply: output = weight @ input.
/// Weight layout: [rows, cols] row-major.
#[inline(always)]
pub fn matmul(output: &mut [f32], weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    crate::simd::simd_matmul_rows(output, weight, input, rows, cols);
}

/// Row-parallel matrix-vector multiply for large weight matrices (Plan 096).
///
/// Splits output rows across rayon threads. Use for large matmuls where
/// row count >> core count (e.g., `down_proj` 2304×9216, `lm_head` 256K×2304).
/// Falls back to sequential [`matmul`] for small matrices (rows < 512).
#[inline(always)]
pub fn matmul_parallel(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    crate::simd::simd_matmul_rows_parallel(output, weight, input, rows, cols);
}

/// Fused matrix-vector multiply + ReLU: output = max(0, weight @ input).
/// Saves one full buffer scan vs separate matmul + ReLU.
/// Used for MLP hidden layer where activation immediately follows projection.
#[inline(always)]
pub fn matmul_relu(output: &mut [f32], weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    crate::simd::simd_matmul_relu_rows(output, weight, input, rows, cols);
}

/// Matrix-vector multiply with f16 weights: output = f16_weight @ f32_input.
/// Weight layout: [rows, cols] row-major, stored as `half::f16`.
///
/// Converts f16 weights to f32 on-the-fly during dot product.
/// Halves memory bandwidth for weight reads vs f32 storage.
#[inline(always)]
pub fn matmul_f16(
    output: &mut [f32],
    weight: &[half::f16],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    crate::simd::simd_matmul_f16_f32_rows(output, weight, input, rows, cols);
}

/// Row-parallel f16×f32 matrix-vector multiply for large weight matrices (Plan 096).
///
/// Splits output rows across rayon threads. Use for large f16 matmuls where
/// row count >> core count (e.g., `down_proj` 2304×9216, `lm_head` 256K×2304).
/// Falls back to sequential [`matmul_f16`] for small matrices (rows < 512).
#[inline(always)]
pub fn matmul_f16_parallel(
    output: &mut [f32],
    weight: &[half::f16],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    crate::simd::simd_matmul_f16_f32_rows_parallel(output, weight, input, rows, cols);
}

/// Sparse matrix-vector multiply for ReLU-activated inputs (TwELL-inspired).
///
/// Only processes columns where `input[c] > 0.0`, skipping dead neurons entirely.
/// Exploits the natural sparsity of ReLU activations in MLP layers where 95-99%
/// of hidden neurons are exactly zero after training with L1 regularization.
///
/// Distilled from "Sparser, Faster, Lighter Transformer Language Models"
/// (arXiv:2603.23198) by Sakana AI & NVIDIA.
///
/// Two-phase execution:
/// 1. Dynamic Packing: scan input, store non-zero indices & values into pre-allocated buffers
/// 2. Sparse Multiply: only iterate weights at alive column indices
///
/// Returns the number of alive (non-zero) neurons for diagnostics/threshold checks.
/// Buffers `active_indices` and `active_values` must be pre-allocated to at least `cols` capacity.
#[cfg(feature = "sparse_mlp")]
#[inline(always)]
pub fn sparse_matmul(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    active_indices: &mut [usize],
    active_values: &mut [f32],
) -> usize {
    // Phase 1: Pack alive neurons (software TwELL formulation)
    // Branch-predicted: with 95-99% sparsity, the branch is predicted correctly
    // most of the time (~1 cycle), and we avoid the wasted store to active_values
    // for dead neurons that the branch-free version always performed.
    let mut alive = 0;
    for c in 0..cols {
        let val = unsafe { *input.get_unchecked(c) };
        if val > 0.0 {
            unsafe {
                *active_indices.get_unchecked_mut(alive) = c;
                *active_values.get_unchecked_mut(alive) = val;
            }
            alive += 1;
        }
    }

    // Phase 2: Sparse multiply — SIMD-accelerated (Plan 060 T5)
    // NEON gathers 4 elements/iter, AVX2 gathers 8 elements/iter via hardware gather.
    // Scalar fallback for alive ≤ 4 (gather overhead exceeds benefit).
    crate::simd::simd_sparse_matmul_rows(
        output,
        weight,
        active_indices,
        active_values,
        rows,
        cols,
        alive,
    );

    alive
}

/// Sample a token index from a probability distribution.
///
/// Builds a prefix-sum (CDF) then uses binary search for O(log V) lookup
/// instead of the O(V/2) average of a linear scan.
///
/// **Allocates a CDF buffer on every call.** For the hot decode loop, prefer
/// [`sample_token_into`] which reuses a pre-allocated buffer.
pub fn sample_token(probs: &[f32], rng: &mut Rng) -> usize {
    // Redraw on exactly 0.0: `rng.uniform()` can return 0.0 (notably the first draw
    // for low-entropy seeds), which deterministically maps to the first nonzero-mass
    // token via the left boundary of the inverse-CDF map. `partition_point(c <= r)`
    // below already implements the strict comparison, so this guard only fixes the
    // degenerate-zero draw without changing the comparison semantics.
    let mut r = rng.uniform();
    while r == 0.0 {
        r = rng.uniform();
    }
    let n = probs.len();
    if n == 0 {
        return 0;
    }

    // Build cumulative sum array — pre-allocated, direct write avoids per-push bounds check
    let mut cdf = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        sum += p;
        // SAFETY: cdf has length n, i < n by enumeration
        unsafe {
            *cdf.get_unchecked_mut(i) = sum;
        }
    }

    // partition_point: first index where cdf[i] > r — monotonically increasing
    let idx = cdf[..n].partition_point(|&c| c <= r);
    idx.min(n - 1)
}

/// Zero-alloc variant of [`sample_token`] that reuses a pre-allocated CDF buffer.
///
/// Pass a `cdf` buffer (e.g. `ForwardContext::cdf`) to avoid a ~vocab_size allocation
/// on every token decode. The buffer is cleared and refilled each call.
pub fn sample_token_into(probs: &[f32], rng: &mut Rng, cdf: &mut Vec<f32>) -> usize {
    // See `sample_token`: redraw on exactly 0.0 to avoid the degenerate left-boundary draw.
    let mut r = rng.uniform();
    while r == 0.0 {
        r = rng.uniform();
    }
    let n = probs.len();
    if n == 0 {
        return 0;
    }
    cdf.resize(n, 0.0);
    let mut sum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        sum += p;
        unsafe {
            *cdf.get_unchecked_mut(i) = sum;
        }
    }
    // partition_point: first index where cdf[i] > r — monotonically increasing
    // so this is equivalent to binary_search_by with Less/Greater but avoids
    // closure overhead and is branch-predictor friendly.
    let idx = cdf[..n].partition_point(|&c| c <= r);
    idx.min(n - 1)
}

// ---------------------------------------------------------------------------
// LoRA Adapter — CPU inference path (Plan 025)
// ---------------------------------------------------------------------------

/// CPU-side LoRA adapter for inference.
/// Loads from the same binary format as `GpuLoraAdapter` (Plan 008):
/// `[LORA(4) | version(4) | blake3(32) | payload...]`
/// where payload = `[n_adapters(4) | rank(4) | alpha(4) | adapter_data...]`
/// and adapter_data = `[in_dim(4) | out_dim(4) | a_f32s | b_f32s]`
///
/// Use [`LoraAdapter::load`] to read ALL adapters from a multi-adapter file
/// (correct for L2+ models), or [`LoraAdapter::load_first`] when only the
/// first adapter is needed (e.g., single-forward-pass heuristic players).
///
/// Zero-copy: loaded once per domain, reference-passed during inference.
///
/// Fields ordered by descending alignment to minimize padding:
/// usize/Vec (8-byte) → f32 (4-byte).
pub struct LoraAdapter {
    /// LoRA rank.
    pub rank: usize,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Down-projection: [rank × in_dim]
    pub a: Vec<f32>,
    /// Up-projection: [out_dim × rank]
    pub b: Vec<f32>,
    /// Scaling factor (alpha / rank).
    pub alpha: f32,
}

impl LoraAdapter {
    /// Load ALL adapters from a Plan 008 binary LoRA file.
    ///
    /// Multi-adapter files (e.g., L2+ with 6 adapters/layer × n_layer) return every
    /// adapter in declaration order. Single-adapter files return a 1-element Vec.
    ///
    /// Issue 299: previously this returned only the first adapter, silently
    /// discarding layers 1+ and invalidating L2+ arena benchmarks.
    pub fn load(path: &std::path::Path) -> Result<Vec<Self>, String> {
        const LORA_MAGIC: &[u8; 4] = b"LORA";
        const LORA_VERSION: u32 = 1;

        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read lora file: {e}"))?;

        if file_data.len() < 44 {
            return Err("File too small for lora header".into());
        }

        if &file_data[0..4] != LORA_MAGIC {
            return Err("Invalid lora magic bytes".into());
        }

        let version = u32::from_le_bytes(
            file_data[4..8]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("Version parse: {e}"))?,
        );
        if version != LORA_VERSION {
            return Err(format!("Unsupported lora version: {version}"));
        }

        let stored_checksum = &file_data[8..40];
        let payload = &file_data[40..];

        let computed = blake3::hash(payload);
        if computed.as_bytes() != stored_checksum {
            return Err("LoRA file checksum mismatch".into());
        }

        let mut offset = 0usize;
        let n_adapters = read_u32_le(payload, &mut offset)? as usize;
        let rank = read_u32_le(payload, &mut offset)? as usize;
        let alpha = read_f32_le(payload, &mut offset)?;

        if n_adapters == 0 {
            return Err("No adapters in lora file".into());
        }

        let mut adapters = Vec::with_capacity(n_adapters);
        for i in 0..n_adapters {
            let in_dim = read_u32_le(payload, &mut offset)? as usize;
            let out_dim = read_u32_le(payload, &mut offset)? as usize;

            let a_count = rank * in_dim;
            let b_count = out_dim * rank;
            let a_bytes = a_count * std::mem::size_of::<f32>();
            let b_bytes = b_count * std::mem::size_of::<f32>();

            if offset + a_bytes + b_bytes > payload.len() {
                return Err(format!("Truncated adapter {i} data"));
            }

            let a: Vec<f32> = {
                #[cfg(target_endian = "little")]
                {
                    let mut v = Vec::with_capacity(a_count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload[offset..].as_ptr(),
                            v.as_mut_ptr() as *mut u8,
                            a_bytes,
                        );
                        v.set_len(a_count);
                    }
                    v
                }
                #[cfg(not(target_endian = "little"))]
                {
                    payload[offset..offset + a_bytes]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                        .collect()
                }
            };
            offset += a_bytes;

            let b: Vec<f32> = {
                #[cfg(target_endian = "little")]
                {
                    let mut v = Vec::with_capacity(b_count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload[offset..].as_ptr(),
                            v.as_mut_ptr() as *mut u8,
                            b_bytes,
                        );
                        v.set_len(b_count);
                    }
                    v
                }
                #[cfg(not(target_endian = "little"))]
                {
                    payload[offset..offset + b_bytes]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                        .collect()
                }
            };
            offset += b_bytes;

            adapters.push(Self {
                rank,
                in_dim,
                out_dim,
                alpha,
                a,
                b,
            });
        }

        if offset != payload.len() {
            return Err(format!(
                "LoRA payload trailing data: consumed {offset}, payload {}",
                payload.len()
            ));
        }

        Ok(adapters)
    }

    /// Load only the first adapter from a Plan 008 binary LoRA file.
    ///
    /// Convenience for consumers that store a single `LoraAdapter` and only run
    /// one forward pass (e.g., `LoraPlayer`, `FullHLPlayer`). Multi-adapter
    /// files (L2+) have layers 1+ silently dropped — this is explicit about
    /// that limitation so callers cannot accidentally regress on Issue 299.
    ///
    /// For correct multi-adapter evaluation, use [`load`](Self::load) and apply
    /// each adapter to its target projection during the forward pass.
    pub fn load_first(path: &std::path::Path) -> Result<Self, String> {
        let adapters = Self::load(path)?;
        adapters
            .into_iter()
            .next()
            .ok_or_else(|| "LoRA file declared zero-length adapter list".into())
    }

    /// Load LoRA adapters from a compact binary format.
    ///
    /// Format:
    /// ```text
    /// [MAGIC: "LORA" 4B]
    /// [VERSION: 1B]
    /// [RANK: 2B LE]
    /// [N_LAYERS: 2B LE]
    /// [N_TARGETS: 2B LE]
    /// [TARGET_IDS: N_TARGETS × 2B LE]  (0=q_proj, 1=k_proj, 2=v_proj, 3=o_proj,
    ///                                    4=gate_proj, 5=up_proj, 6=down_proj)
    /// [LAYER_DATA: for each (layer, target):
    ///   [A_ROWS: 2B][A_COLS: 2B][A_DATA: A_ROWS×A_COLS × 4B f32]
    ///   [B_ROWS: 2B][B_COLS: 2B][B_DATA: B_ROWS×B_COLS × 4B f32]
    /// ]
    /// [BLAKE3_HASH: 32B]  — covers everything before it
    /// ```
    ///
    /// Alpha defaults to `rank * 2`.
    pub fn load_from_bin(path: &std::path::Path) -> Result<Vec<Self>, String> {
        const LORA_MAGIC: &[u8; 4] = b"LORA";
        const LORA_VERSION: u8 = 1;

        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read lora bin file: {e}"))?;

        // Minimum: magic(4) + version(1) + rank(2) + n_layers(2) + n_targets(2) + hash(32) = 43
        if file_data.len() < 43 {
            return Err("File too small for lora bin header".into());
        }

        // Validate blake3 checksum — last 32 bytes cover everything before them
        let data_len = file_data.len() - 32;
        let stored_checksum = &file_data[data_len..];
        let computed = blake3::hash(&file_data[..data_len]);
        if computed.as_bytes() != stored_checksum {
            return Err("LoRA bin file checksum mismatch".into());
        }

        let mut offset = 0usize;

        // Magic
        if &file_data[offset..offset + 4] != LORA_MAGIC {
            return Err("Invalid lora bin magic bytes".into());
        }
        offset += 4;

        // Version
        let version = file_data[offset];
        if version != LORA_VERSION {
            return Err(format!("Unsupported lora bin version: {version}"));
        }
        offset += 1;

        // Rank
        let rank = read_u16_le(&file_data, &mut offset)? as usize;

        // N_LAYERS
        let n_layers = read_u16_le(&file_data, &mut offset)? as usize;

        // N_TARGETS
        let n_targets = read_u16_le(&file_data, &mut offset)? as usize;

        if n_layers == 0 || n_targets == 0 {
            return Err("No layers or targets in lora bin file".into());
        }

        // TARGET_IDS
        let mut target_ids = Vec::with_capacity(n_targets);
        for _ in 0..n_targets {
            let tid = read_u16_le(&file_data, &mut offset)?;
            match tid {
                0..=6 => target_ids.push(tid),
                _ => return Err(format!("Invalid target ID: {tid}")),
            }
        }

        // LAYER_DATA
        let alpha = (rank * 2) as f32;
        let mut adapters = Vec::with_capacity(n_layers * n_targets);

        for _layer in 0..n_layers {
            for &_target_id in &target_ids {
                // A matrix: [rank × in_dim]
                let a_rows = read_u16_le(&file_data, &mut offset)? as usize;
                let a_cols = read_u16_le(&file_data, &mut offset)? as usize;
                let a_count = a_rows * a_cols;
                let a_bytes = a_count * std::mem::size_of::<f32>();

                if offset + a_bytes > data_len {
                    return Err("Truncated A matrix data".into());
                }

                let a: Vec<f32> = {
                    #[cfg(target_endian = "little")]
                    {
                        let mut v = Vec::with_capacity(a_count);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                file_data[offset..].as_ptr(),
                                v.as_mut_ptr() as *mut u8,
                                a_bytes,
                            );
                            v.set_len(a_count);
                        }
                        v
                    }
                    #[cfg(not(target_endian = "little"))]
                    {
                        file_data[offset..offset + a_bytes]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                            .collect()
                    }
                };
                offset += a_bytes;

                // B matrix: [out_dim × rank]
                let b_rows = read_u16_le(&file_data, &mut offset)? as usize;
                let b_cols = read_u16_le(&file_data, &mut offset)? as usize;
                let b_count = b_rows * b_cols;
                let b_bytes = b_count * std::mem::size_of::<f32>();

                if offset + b_bytes > data_len {
                    return Err("Truncated B matrix data".into());
                }

                let b: Vec<f32> = {
                    #[cfg(target_endian = "little")]
                    {
                        let mut v = Vec::with_capacity(b_count);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                file_data[offset..].as_ptr(),
                                v.as_mut_ptr() as *mut u8,
                                b_bytes,
                            );
                            v.set_len(b_count);
                        }
                        v
                    }
                    #[cfg(not(target_endian = "little"))]
                    {
                        file_data[offset..offset + b_bytes]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                            .collect()
                    }
                };
                offset += b_bytes;

                let in_dim = a_cols;
                let out_dim = b_rows;

                adapters.push(Self {
                    rank,
                    in_dim,
                    out_dim,
                    alpha,
                    a,
                    b,
                });
            }
        }

        if offset != data_len {
            return Err(format!(
                "Unexpected trailing data: read {offset}, expected {data_len}"
            ));
        }

        if adapters.is_empty() {
            return Err("No adapters loaded from lora bin file".into());
        }

        Ok(adapters)
    }
}

/// Apply LoRA delta in-place: `output += (alpha/rank) × B @ (A @ input)`
///
/// `lora_buf` is a pre-allocated `[rank]` intermediate buffer — zero alloc in hot path.
/// The B×hidden multiplication and scaling are fused directly into the output accumulation,
/// avoiding a separate delta buffer.
#[inline(always)]
pub fn lora_apply(output: &mut [f32], lora: &LoraAdapter, input: &[f32], lora_buf: &mut [f32]) {
    let scale = lora.alpha / lora.rank as f32;

    // 1. hidden = A @ input  (rank × in_dim) @ [in_dim] → [rank]
    matmul(lora_buf, &lora.a, input, lora.rank, lora.in_dim);

    // 2. output += scale × (B @ hidden) — SIMD-accelerated per-row dot product
    for r in 0..lora.out_dim {
        let row_off = r * lora.rank;
        let sum =
            crate::simd::simd_dot_f32(&lora.b[row_off..row_off + lora.rank], lora_buf, lora.rank);
        unsafe {
            *output.get_unchecked_mut(r) += scale * sum;
        }
    }
}

/// A loaded LoRA pair for modality-specific inference (Plan 025).
/// Reader is active during bidirectional prefill, writer during causal decode.
/// Switching is a reference swap — zero data movement.
pub struct LoraPair {
    /// LoRA active during bidirectional prefill (e.g., Python Reader).
    pub reader: Option<LoraAdapter>,
    /// LoRA active during causal decode (e.g., Rust Writer).
    pub writer: Option<LoraAdapter>,
}

impl LoraPair {
    /// Empty pair — no LoRA applied.
    pub fn none() -> Self {
        Self {
            reader: None,
            writer: None,
        }
    }
}

// ---------------------------------------------------------------------------
// GPart Isometric Partition Adapter — replaces LoRA BA with Pθ_d (Research 227, Plan 257)
// ---------------------------------------------------------------------------

/// Binary format magic for GPart adapter files.
#[cfg(feature = "gpart_adapter")]
pub const GPART_MAGIC: &[u8; 5] = b"GPART";
/// Current GPart binary format version.
#[cfg(feature = "gpart_adapter")]
pub const GPART_VERSION: u32 = 1;

/// GPart isometric partition adapter — replaces LoRA's bilinear BA with single isometric Pθ_d.
///
/// `W = W₀ + Pθ_d` where `P^T P = I_d` (isometric, no bilinear distortion).
/// Storage: d + 1 values (seed + theta). Single-pass O(N) broadcast apply.
///
/// Reference: Research 227 (arXiv 2605.14841)
#[cfg(feature = "gpart_adapter")]
#[derive(Clone, Debug)]
pub struct GpartAdapter {
    /// Partition dimension — the only hyperparameter.
    pub d: usize,
    /// Seed for deterministic pseudorandom partition generation.
    pub seed: u64,
    /// Trainable vector θ_d ∈ R^d (produced by training side, riir-ai).
    pub theta: Vec<f32>,
}

#[cfg(feature = "gpart_adapter")]
impl GpartAdapter {
    /// Apply the isometric partition adapter to base weights in-place.
    /// `W = W₀ + Pθ_d` where P is seed-generated.
    /// Single-pass O(N) broadcast. SIMD-friendly (chunked 4/8).
    pub fn apply(&self, base_weights: &mut [f32]) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }

        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);

        // Single-pass broadcast: w[i] += theta[g(i)] / sqrt(n_g)
        for i in 0..n {
            let g = assignments[i];
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            base_weights[i] += scale * self.theta[g];
        }
    }

    /// Apply with pre-allocated scratch buffers — zero alloc in hot path.
    pub fn apply_with_scratch(
        &self,
        base_weights: &mut [f32],
        assignments: &mut [usize],
        group_sizes: &mut [usize],
    ) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }

        self.generate_assignments_into(n, assignments);
        self.compute_group_sizes_into(n, assignments, group_sizes);

        for i in 0..n {
            let g = assignments[i];
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            base_weights[i] += scale * self.theta[g];
        }
    }

    /// Apply with pre-allocated scratch buffers AND a per-group boolean mask.
    ///
    /// Groups where `group_mask[g] == false` contribute zero to `base_weights`.
    /// This is the inference-time pruning path: low-magnitude groups are skipped,
    /// trading output fidelity for reduced effective FLOPs on the downstream matmul.
    ///
    /// `group_mask.len()` must be `>= self.d`. When `group_mask` is all-true this
    /// is identical to [`apply_with_scratch`].
    ///
    /// Gates: `gpart_pruning` feature (Issue 008). Static top-k magnitude mask;
    /// BanditPruner-driven dynamic mask is deferred to a separate plan.
    #[cfg(feature = "gpart_pruning")]
    pub fn apply_with_scratch_masked(
        &self,
        base_weights: &mut [f32],
        assignments: &mut [usize],
        group_sizes: &mut [usize],
        group_mask: &[bool],
    ) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }
        debug_assert!(
            group_mask.len() >= self.d,
            "group_mask len {} < d {}",
            group_mask.len(),
            self.d,
        );

        self.generate_assignments_into(n, assignments);
        self.compute_group_sizes_into(n, assignments, group_sizes);

        // Branch-free inner loop: precompute per-group scale×θ, zeroed when masked.
        // LLVM vectorises the all-true path to match `apply_with_scratch`.
        let mut group_delta: Vec<f32> = Vec::with_capacity(self.d);
        for g in 0..self.d {
            let active = group_mask[g];
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            // `active as f32` is 0.0 or 1.0 — multiply zeros out masked groups
            // without a per-element branch.
            group_delta.push(scale * self.theta[g] * (active as u8 as f32));
        }

        for i in 0..n {
            base_weights[i] += group_delta[assignments[i]];
        }
    }

    /// SIMD-optimised apply: pre-computes per-group delta once (O(d) divisions
    /// instead of O(N)), then applies in 8-wide chunks.
    ///
    /// The "SIMD" benefit here is from eliminating per-element division — the
    /// inner loop is branch-free (pure array lookups), which LLVM auto-vectorises
    /// well. For contiguous same-group runs, `simd_add_scalar_inplace` is used.
    pub fn apply_simd(&self, base_weights: &mut [f32]) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }

        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);

        // Pre-compute per-group delta: θ[g] / √n_g — one division per group
        let group_delta: Vec<f32> = (0..self.d)
            .map(|g| self.theta[g] / (group_sizes[g] as f32).sqrt())
            .collect();

        // Process in 8-wide chunks for auto-vectorisation
        let chunks = n / 8;
        for c in 0..chunks {
            let base = c * 8;
            for j in 0..8 {
                let i = base + j;
                base_weights[i] += group_delta[assignments[i]];
            }
        }
        // Scalar tail
        for i in (chunks * 8)..n {
            base_weights[i] += group_delta[assignments[i]];
        }
    }

    /// Build a top-k magnitude mask: keep the `k` groups with largest `|θ[g]|`.
    ///
    /// Returns a `Vec<bool>` of length `d` where `true` = active. When `k >= d`, all
    /// groups are active (no pruning). This is the static selection policy for
    /// [`apply_with_scratch_masked`] — cheap to compute, deterministic, requires no
    /// reward signal. Dynamic bandit-based masking is deferred to a separate plan.
    ///
    /// Gates: `gpart_pruning` feature (Issue 008).
    #[cfg(feature = "gpart_pruning")]
    pub fn topk_mask(&self, k: usize) -> Vec<bool> {
        if k >= self.d {
            return vec![true; self.d];
        }
        if k == 0 {
            return vec![false; self.d];
        }

        // (|θ[g]|, g) pairs — partial sort by magnitude to find the top-k threshold.
        let mut magnitudes: Vec<(f32, usize)> =
            (0..self.d).map(|g| (self.theta[g].abs(), g)).collect();
        // `select_nth_unstable_by` partitions around the k-th largest in O(d) avg.
        // Descending comparator: after it returns, indices [0..k] hold the k
        // largest magnitudes (unordered).
        magnitudes.select_nth_unstable_by(k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut mask = vec![false; self.d];
        for &(_, g) in &magnitudes[..k] {
            mask[g] = true;
        }
        mask
    }

    /// Generate group assignments from seed.
    /// Uses fastrand for deterministic cross-platform pseudorandom permutation.
    fn generate_assignments(&self, n: usize) -> Vec<usize> {
        let mut rng = fastrand::Rng::with_seed(self.seed);
        let mut assignments = Vec::with_capacity(n);
        for _ in 0..n {
            assignments.push(rng.usize(..self.d));
        }
        assignments
    }

    /// Generate assignments into pre-allocated scratch buffer.
    fn generate_assignments_into(&self, n: usize, out: &mut [usize]) {
        let mut rng = fastrand::Rng::with_seed(self.seed);
        for i in 0..n.min(out.len()) {
            out[i] = rng.usize(..self.d);
        }
    }

    /// Compute group sizes from assignments.
    fn compute_group_sizes(&self, n: usize, assignments: &[usize]) -> Vec<usize> {
        let mut sizes = vec![0usize; self.d];
        for &g in &assignments[..n] {
            sizes[g] += 1;
        }
        sizes
    }

    /// Compute group sizes into pre-allocated scratch buffer.
    fn compute_group_sizes_into(&self, n: usize, assignments: &[usize], out: &mut [usize]) {
        for g in out.iter_mut().take(self.d) {
            *g = 0;
        }
        for &g in &assignments[..n] {
            out[g] += 1;
        }
    }

    /// BLAKE3 commitment over (seed || theta).
    pub fn commitment(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed.to_le_bytes());
        let theta_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.theta.as_ptr() as *const u8,
                self.theta.len() * std::mem::size_of::<f32>(),
            )
        };
        hasher.update(theta_bytes);
        *hasher.finalize().as_bytes()
    }

    /// Verify commitment matches.
    pub fn verify(&self, expected: &[u8; 32]) -> bool {
        self.commitment() == *expected
    }

    /// Save adapter to binary format:
    /// `[GPART(5) | version(4) | d(4) | seed(8) | blake3(32) | theta(d×4)]`
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let d_bytes = self.d as u32;
        let theta_byte_len = self.theta.len() * std::mem::size_of::<f32>();
        // magic(5) + version(4) + d(4) + seed(8) + commitment(32) + theta
        let total = 5 + 4 + 4 + 8 + 32 + theta_byte_len;
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(GPART_MAGIC);
        buf.extend_from_slice(&GPART_VERSION.to_le_bytes());
        buf.extend_from_slice(&d_bytes.to_le_bytes());
        buf.extend_from_slice(&self.seed.to_le_bytes());

        // Commitment placeholder — will overwrite after computing
        let commit_offset = buf.len();
        buf.extend_from_slice(&[0u8; 32]);

        // Theta data
        let theta_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.theta.as_ptr() as *const u8, theta_byte_len) };
        buf.extend_from_slice(theta_bytes);

        // Compute commitment over (seed || theta) and write into placeholder
        let commitment = self.commitment();
        buf[commit_offset..commit_offset + 32].copy_from_slice(&commitment);

        std::fs::write(path, &buf).map_err(|e| format!("Failed to write gpart file: {e}"))
    }

    /// Load adapter from binary format.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read gpart file: {e}"))?;

        // Minimum: magic(5) + version(4) + d(4) + seed(8) + commitment(32) = 53
        if file_data.len() < 53 {
            return Err("File too small for gpart header".into());
        }

        if &file_data[0..5] != GPART_MAGIC {
            return Err("Invalid gpart magic bytes".into());
        }

        let version = u32::from_le_bytes(
            file_data[5..9]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("Version parse: {e}"))?,
        );
        if version != GPART_VERSION {
            return Err(format!("Unsupported gpart version: {version}"));
        }

        let d = u32::from_le_bytes(
            file_data[9..13]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("d parse: {e}"))?,
        ) as usize;

        let seed = u64::from_le_bytes(
            file_data[13..21]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("seed parse: {e}"))?,
        );

        let stored_commitment = &file_data[21..53];

        let theta_bytes_start = 53;
        let theta_bytes_len = d * std::mem::size_of::<f32>();
        if theta_bytes_start + theta_bytes_len > file_data.len() {
            return Err("Truncated theta data".into());
        }

        // Load theta
        let theta: Vec<f32> = {
            #[cfg(target_endian = "little")]
            {
                let src = &file_data[theta_bytes_start..theta_bytes_start + theta_bytes_len];
                let count = d;
                let mut v = Vec::with_capacity(count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        v.as_mut_ptr() as *mut u8,
                        theta_bytes_len,
                    );
                    v.set_len(count);
                }
                v
            }
            #[cfg(not(target_endian = "little"))]
            {
                file_data[theta_bytes_start..theta_bytes_start + theta_bytes_len]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect()
            }
        };

        // Verify commitment
        let adapter = Self { d, seed, theta };
        let computed = adapter.commitment();
        if computed != stored_commitment {
            return Err("GPart file commitment mismatch".into());
        }

        Ok(adapter)
    }

    /// Storage size in bytes (seed + theta).
    pub fn storage_bytes(&self) -> usize {
        8 + self.theta.len() * std::mem::size_of::<f32>()
    }

    /// Verify isometry: ||Pθ||² = ||θ||².
    /// For the partition matrix, each group g contributes n_g × (θ_g/√n_g)² = θ_g².
    /// Summing: Σ_g θ_g² = ||θ||². QED.
    pub fn check_isometry(&self, n: usize) -> bool {
        if n == 0 || self.d == 0 {
            return true;
        }

        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);

        // Compute ||Pθ||²
        let mut projected_norm_sq = 0.0f32;
        for i in 0..n {
            let g = assignments[i];
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            let delta = scale * self.theta[g];
            projected_norm_sq += delta * delta;
        }

        // Compute ||θ||²
        let theta_norm_sq: f32 = self.theta.iter().map(|&v| v * v).sum();

        (projected_norm_sq - theta_norm_sq).abs() < 1e-3
    }
}

/// Pre-computed per-element deltas for zero-allocation apply.
///
/// Call [`GpartAdapter::prepare`] once for a given weight dimension `n`,
/// then [`GpartPrepared::apply`] in the hot loop — no RNG, no allocation.
#[cfg(feature = "gpart_adapter")]
#[derive(Clone, Debug)]
pub struct GpartPrepared {
    /// Per-element delta: `theta[group(i)] / sqrt(n_group)` for each weight index.
    deltas: Vec<f32>,
}

#[cfg(feature = "gpart_adapter")]
impl GpartAdapter {
    /// Pre-compute per-element deltas for a weight vector of length `n`.
    ///
    /// This is the **fast path** — call once after loading the adapter,
    /// then use [`GpartPrepared::apply`] in the hot inference loop.
    /// Cost: O(n) RNG + O(n) broadcast setup (amortised once per model load).
    pub fn prepare(&self, n: usize) -> GpartPrepared {
        if n == 0 || self.d == 0 {
            return GpartPrepared { deltas: Vec::new() };
        }
        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);
        // Match apply() arithmetic exactly: scale = 1/sqrt(n_g), delta = scale * theta[g]
        let group_delta: Vec<f32> = (0..self.d)
            .map(|g| {
                let scale = 1.0 / (group_sizes[g] as f32).sqrt();
                scale * self.theta[g]
            })
            .collect();
        let deltas = assignments.iter().map(|&g| group_delta[g]).collect();
        GpartPrepared { deltas }
    }
}

#[cfg(feature = "gpart_adapter")]
impl GpartPrepared {
    /// Apply pre-computed deltas in-place: `w[i] += delta[i]`.
    ///
    /// Pure O(N) broadcast, zero allocation, branch-free inner loop.
    /// LLVM auto-vectorises this to SIMD.
    pub fn apply(&self, base_weights: &mut [f32]) {
        let len = base_weights.len().min(self.deltas.len());
        for i in 0..len {
            base_weights[i] += self.deltas[i];
        }
    }
}

/// A loaded GPart pair for modality-specific inference, mirroring LoraPair.
/// Reader is active during bidirectional prefill, writer during causal decode.
#[cfg(feature = "gpart_adapter")]
#[derive(Clone, Debug)]
pub struct GpartPair {
    /// GPart active during bidirectional prefill.
    pub reader: Option<GpartAdapter>,
    /// GPart active during causal decode.
    pub writer: Option<GpartAdapter>,
}

#[cfg(feature = "gpart_adapter")]
impl GpartPair {
    /// Empty pair — no GPart applied.
    pub fn none() -> Self {
        Self {
            reader: None,
            writer: None,
        }
    }

    /// Apply the reader (prefill) adapter to base weights, if present.
    pub fn apply_prefill(&self, base_weights: &mut [f32]) {
        if let Some(ref adapter) = self.reader {
            adapter.apply(base_weights);
        }
    }

    /// Apply the writer (decode) adapter to base weights, if present.
    pub fn apply_decode(&self, base_weights: &mut [f32]) {
        if let Some(ref adapter) = self.writer {
            adapter.apply(base_weights);
        }
    }
}

/// Conversion from LoRA to GPart (lossy, requires pre-computed θ_d).
///
/// **Note:** θ_d = P⁺ΔW must be computed by the riir-ai training pipeline.
/// This conversion is a placeholder until that pipeline provides θ_d.
#[cfg(feature = "gpart_adapter")]
impl TryFrom<&LoraAdapter> for GpartAdapter {
    type Error = &'static str;

    fn try_from(_lora: &LoraAdapter) -> Result<Self, Self::Error> {
        Err("GpartAdapter requires pre-computed θ_d from riir-ai training pipeline (P⁺ΔW)")
    }
}

// NeuronShard Pod compatibility: seed(8) + d(max=90)×4(360) = 368 bytes max.
#[cfg(feature = "gpart_adapter")]
const _: () = assert!(8 + 90 * 4 <= 368);

// ---------------------------------------------------------------------------
// DomainLatent — feature-gated (Plan 038)
// ---------------------------------------------------------------------------

/// Domain latent embedding for mid-layer conditioning (Plan 038).
///
/// Injected at layer `n_layer / 2` by adding to K and V projections before cache write.
/// Inspired by the Free Transformer's mid-layer latent injection, adapted for
/// supervised domain conditioning via LoRA fine-tuning.
///
/// Shape: `[kv_dim]` — one embedding per domain, matching K/V dimension for GQA.
///
/// # Binary format
///
/// ```text
/// [MAGIC: "DLAT" 4B][VERSION: 1B][KV_DIM: 4B LE][EMBEDDING: kv_dim × f32 LE][BLAKE3: 32B]
/// ```
///
/// BLAKE3 checksum covers everything before it (magic through embedding).
#[cfg(feature = "domain_latent")]
#[derive(Debug)]
pub struct DomainLatent {
    /// Domain embedding vector, shape `[kv_dim]`.
    pub embedding: Vec<f32>,
}

#[cfg(feature = "domain_latent")]
impl DomainLatent {
    const MAGIC: &[u8; 4] = b"DLAT";
    const VERSION: u8 = 1;

    /// Load domain latent from binary file.
    ///
    /// Format: `[MAGIC 4B][VERSION 1B][KV_DIM 4B LE][EMBEDDING kv_dim×f32 LE][BLAKE3 32B]`
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data =
            std::fs::read(path).map_err(|e| format!("Failed to read domain_latent file: {e}"))?;

        // Minimum: magic(4) + version(1) + kv_dim(4) + hash(32) = 41
        if data.len() < 41 {
            return Err("File too small for domain_latent header".into());
        }

        // Validate BLAKE3 checksum — last 32 bytes cover everything before them
        let payload_end = data.len() - 32;
        let stored_checksum = &data[payload_end..];
        let computed = blake3::hash(&data[..payload_end]);
        if computed.as_bytes() != stored_checksum {
            return Err("Domain latent file checksum mismatch".into());
        }

        let mut offset = 0usize;

        // Magic
        if &data[offset..offset + 4] != Self::MAGIC {
            return Err("Invalid domain_latent magic bytes".into());
        }
        offset += 4;

        // Version
        let version = data[offset];
        if version != Self::VERSION {
            return Err(format!("Unsupported domain_latent version: {version}"));
        }
        offset += 1;

        // KV_DIM
        let kv_dim = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("kv_dim parse: {e}"))?,
        ) as usize;
        offset += 4;

        // Embedding data — bulk copy on LE targets, element-by-element otherwise
        let embed_bytes = kv_dim * std::mem::size_of::<f32>();
        if offset + embed_bytes > payload_end {
            return Err(format!(
                "Truncated embedding data: expected {embed_bytes} bytes at offset {offset}, payload ends at {payload_end}"
            ));
        }

        let embedding: Vec<f32> = {
            #[cfg(target_endian = "little")]
            {
                let mut v = Vec::with_capacity(kv_dim);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data[offset..].as_ptr(),
                        v.as_mut_ptr() as *mut u8,
                        embed_bytes,
                    );
                    v.set_len(kv_dim);
                }
                v
            }
            #[cfg(not(target_endian = "little"))]
            {
                data[offset..offset + embed_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect()
            }
        };

        if embedding.len() != kv_dim {
            return Err(format!(
                "Embedding length mismatch: got {}, expected {kv_dim}",
                embedding.len()
            ));
        }

        Ok(Self { embedding })
    }

    /// Save domain latent to binary file (for tests and training export).
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let kv_dim = self.embedding.len();
        let embed_bytes = kv_dim * std::mem::size_of::<f32>();
        let payload_len = 4 + 1 + 4 + embed_bytes;
        let mut buf = Vec::with_capacity(payload_len + 32);

        buf.extend_from_slice(Self::MAGIC);
        buf.push(Self::VERSION);
        buf.extend_from_slice(&(kv_dim as u32).to_le_bytes());
        // Bulk write embedding data — avoids per-element extend_from_slice overhead.
        // SAFETY: f32 is plain-old-data with no padding; to_ne_bytes gives [u8; 4] per f32.
        // On LE targets (all Apple Silicon, all modern x86), to_ne_bytes == to_le_bytes.
        #[cfg(target_endian = "little")]
        {
            let bytes = unsafe {
                std::slice::from_raw_parts(self.embedding.as_ptr() as *const u8, embed_bytes)
            };
            buf.extend_from_slice(bytes);
        }
        #[cfg(not(target_endian = "little"))]
        {
            for &val in &self.embedding {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        let hash = blake3::hash(&buf);
        buf.extend_from_slice(hash.as_bytes());

        std::fs::write(path, &buf)
            .map_err(|e| format!("Failed to write domain_latent file: {e}"))?;

        Ok(())
    }

    /// Create a zero-initialized domain latent of the given kv_dim.
    pub fn zeros(kv_dim: usize) -> Self {
        Self {
            embedding: vec![0.0; kv_dim],
        }
    }

    /// Create a domain latent from a raw embedding vector.
    pub fn from_vec(embedding: Vec<f32>) -> Self {
        Self { embedding }
    }
}

// ---------------------------------------------------------------------------
// Binary helper functions
// ---------------------------------------------------------------------------

fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32, String> {
    if *offset + 4 > data.len() {
        return Err("Unexpected end of data reading u32".into());
    }
    let val = u32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("u32 parse: {e}"))?,
    );
    *offset += 4;
    Ok(val)
}

fn read_f32_le(data: &[u8], offset: &mut usize) -> Result<f32, String> {
    if *offset + 4 > data.len() {
        return Err("Unexpected end of data reading f32".into());
    }
    let val = f32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("f32 parse: {e}"))?,
    );
    *offset += 4;
    Ok(val)
}

fn read_u16_le(data: &[u8], offset: &mut usize) -> Result<u16, String> {
    if *offset + 2 > data.len() {
        return Err("Unexpected end of data reading u16".into());
    }
    let val = u16::from_le_bytes(
        data[*offset..*offset + 2]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("u16 parse: {e}"))?,
    );
    *offset += 2;
    Ok(val)
}

// ---------------------------------------------------------------------------
// InferenceResult
// ---------------------------------------------------------------------------

/// Output of a single inference pass, with reward signal for feedback loop.
///
/// Fields ordered by descending alignment to minimize padding:
/// u64/i64/usize/String (8-byte) → f32 (4-byte) → Option<#[repr(u8)]> (2-byte) → u8/bool (1-byte).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    // --- 8-byte aligned ---
    /// Input prompt hash (for dedup, not stored).
    pub prompt_hash: u64,
    /// Timestamp (Uuid v7 prefix).
    pub timestamp: i64,
    /// Number of nodes explored in DDTree.
    pub tree_budget_used: usize,
    /// Actual planning horizon used this turn (after entropy truncation, Plan 112 T13).
    #[cfg(feature = "sr2am_configurator")]
    pub plan_horizon_used: usize,
    /// Domain that handled this inference.
    pub domain: String,
    /// Generated output text.
    pub output: String,

    // --- 4-byte aligned ---
    /// Best-path reward (max relevance score from WasmPruner).
    pub reward: f32,

    // --- 2-byte aligned (Option<#[repr(u8)] enum>) ---
    /// SR²AM configurator planning decision for this turn (Plan 112).
    #[cfg(feature = "sr2am_configurator")]
    pub planning_decision: Option<PlanningDecision>,

    // --- 1-byte fields (tail-packed) ---
    /// Was this result screened out (reward below threshold)?
    pub screened: bool,
    /// Inference budget level (0=cheap, 1=moderate, 2=expensive).
    pub budget_level: u8,
}

// ---------------------------------------------------------------------------
// Data Gate — Self-Play Stability (Plan 111, Research 075)
// ---------------------------------------------------------------------------

/// Discriminator for different self-play task types.
#[cfg(feature = "data_gate")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskType {
    /// Python code output prediction
    CodeIO,
    /// DSL expression evaluation
    DslExpr,
    /// Game action (Bomber, Go, FFT, Monopoly)
    GameAction,
    /// Open-ended generation
    OpenEnded,
}

/// A task proposed by the self-play proposer, before solver evaluation.
#[cfg(feature = "data_gate")]
#[derive(Debug, Clone)]
pub struct ProposerTask {
    /// Task identifier for diagnostics.
    pub id: usize,
    /// The problem/query text.
    pub query: String,
    /// Optional code or DSL expression to execute.
    pub program: Option<String>,
    /// Optional input for the program.
    pub program_input: Option<String>,
    /// Task type discriminator.
    pub task_type: TaskType,
}

/// Gate admission decision.
///
/// Decides whether a proposer-generated task should enter the training pool
/// BEFORE the solver attempts it.
#[cfg(feature = "data_gate")]
#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Task passes the gate — admitted to training pool.
    Admit,
    /// Task rejected with reason.
    Reject(String),
}

/// Task-level admission gate for self-play training pool.
///
/// Decides whether a proposer-generated task should enter the training pool
/// BEFORE the solver attempts it. This is the binding constraint for self-play
/// stability (Survive or Collapse, Pu et al. 2026).
///
/// Key finding: a strict gate is sufficient for stability under every reward
/// variant; no reward variant is sufficient once the gate is removed.
#[cfg(feature = "data_gate")]
pub trait DataGate {
    /// Admit or reject a proposed task.
    ///
    /// Returns `GateDecision::Admit` if the task passes the gate,
    /// `GateDecision::Reject(reason)` if not.
    fn admit(&self, task: &ProposerTask) -> GateDecision;

    /// Current leak rate ε (fraction of failed tasks admitted).
    /// ε=0 means strict gate (optimal). ε=1 means gate off (collapse).
    fn leak_rate(&self) -> f32;
}

// ---------------------------------------------------------------------------
// Training-Free Loop Types (Plan 136)
// ---------------------------------------------------------------------------

/// Sub-step integration strategy for the training-free loop.
///
/// Controls how intermediate loop outputs are combined with the running state.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SubStepStrategy {
    /// Damped Euler: x ← x + (1/K)·(y − x)
    #[default]
    DampedEuler,
    /// K-stage Runge-Kutta: x ← β·y + (1−β)·x
    KStageRK {
        /// Blend factor β ∈ [0, 1]. 0.5 is neutral (equal weight).
        beta: f32,
    },
}

/// Iteration mode for the training-free loop window.
///
/// Controls whether the window is applied as a single block or iterated
/// layer-by-layer within each sub-step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IterationMode {
    /// Apply the full window [a, b] as one block per sub-step.
    #[default]
    Block,
    /// Apply each layer in the window individually per sub-step.
    Layer,
}

/// KV cache write strategy for the training-free loop.
///
/// Controls which loop iteration writes the canonical KV entries.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Use the final loop iteration's hidden state for KV cache.
    #[default]
    Last,
    /// Use the pre-loop hidden state for KV cache (first iteration).
    First,
}

/// Configuration for training-free loop wrapper (Plan 136).
///
/// Pure inference-time retrofit: re-applies a contiguous mid-stack block of
/// layers with ODE-motivated damped sub-stepping. No training needed.
#[derive(Clone, Debug)]
pub struct TrainingFreeLoopConfig {
    /// Start of the loop window (inclusive layer index).
    pub window_start: usize,
    /// End of the loop window (inclusive layer index).
    pub window_end: usize,
    /// Number of loop iterations (K in the paper).
    pub loop_count: usize,
    /// Sub-step integration strategy.
    pub strategy: SubStepStrategy,
    /// Iteration mode (block vs layer-wise).
    pub iteration_mode: IterationMode,
    /// KV cache write strategy.
    pub cache_strategy: CacheStrategy,
}

impl Default for TrainingFreeLoopConfig {
    fn default() -> Self {
        Self {
            window_start: 0,
            window_end: 0,
            loop_count: 2,
            strategy: SubStepStrategy::KStageRK { beta: 0.5 },
            iteration_mode: IterationMode::Block,
            cache_strategy: CacheStrategy::First,
        }
    }
}

impl TrainingFreeLoopConfig {
    /// Creates a config with sensible defaults for a given `Config`.
    ///
    /// Window heuristic: center at 48% depth, ±1 layer.
    /// For small models (≤4 layers), defaults to (0, n_layer−1).
    /// Uses the paper-recommended K-stage RK strategy with β=0.5.
    pub fn from_config(config: &Config) -> Self {
        let n = config.n_layer;
        let (window_start, window_end) = if n <= 4 {
            (0, n.saturating_sub(1))
        } else {
            let center = (n as f32 * 0.48) as usize;
            (center.saturating_sub(1), (center + 2).min(n - 1))
        };
        Self {
            window_start,
            window_end,
            loop_count: 2,
            strategy: SubStepStrategy::KStageRK { beta: 0.5 },
            iteration_mode: IterationMode::Block,
            cache_strategy: CacheStrategy::First,
        }
    }
}

/// Bit-plane packed ternary weights: each element is {-1, 0, +1}.
///
/// 64 weights per block stored as two u64 bitmasks:
/// - pos_bits[block] bit k set → weight[row][k] = +1
/// - neg_bits[block] bit k set → weight[row][k] = -1
/// - both zero → weight = 0 (implicit skip, no storage needed)
///
/// `row_scale[r]` rescales the accumulated sum back toward original float magnitudes.
/// Memory: ~1.58 bits/weight (log₂3), plus one f32 per row for scale.
#[cfg(feature = "plasma_path")]
#[derive(Clone, Debug)]
pub struct TernaryWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,     // (cols + 63) / 64
    pub pos_bits: Vec<u64>,  // [rows * blocks64]
    pub neg_bits: Vec<u64>,  // [rows * blocks64]
    pub row_scale: Vec<f32>, // [rows]
}

#[cfg(feature = "plasma_path")]
impl TernaryWeights {
    /// Create zeroed ternary weights.
    pub fn new(rows: usize, cols: usize) -> Self {
        let blocks64 = cols.div_ceil(64);
        Self {
            rows,
            cols,
            blocks64,
            pos_bits: vec![0u64; rows * blocks64],
            neg_bits: vec![0u64; rows * blocks64],
            row_scale: vec![1.0f32; rows],
        }
    }

    /// Set a single ternary value at (row, col). Panics if out of bounds or value not in {-1, 0, +1}.
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        assert!(
            (-1..=1).contains(&value),
            "ternary value must be -1, 0, or +1"
        );
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        match value {
            1 => {
                self.pos_bits[idx] |= mask;
                self.neg_bits[idx] &= !mask;
            }
            -1 => {
                self.pos_bits[idx] &= !mask;
                self.neg_bits[idx] |= mask;
            }
            0 => {
                self.pos_bits[idx] &= !mask;
                self.neg_bits[idx] &= !mask;
            }
            _ => unreachable!(),
        }
    }

    /// Get the ternary value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        let pos = (self.pos_bits[idx] & mask) != 0;
        let neg = (self.neg_bits[idx] & mask) != 0;
        pos as i8 - neg as i8
    }

    /// Quantize f32 weights to ternary with row-wise error compensation.
    ///
    /// For each row:
    ///   scale = mean(|row|)
    ///   threshold = 0.5 * scale
    ///   for each weight: adjusted = value + carry
    ///     if adjusted > threshold → +1
    ///     if adjusted < -threshold → -1
    ///     else → 0
    ///     carry = adjusted - (q * scale)
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(
            weights.len(),
            rows * cols,
            "weights slice must be rows*cols"
        );
        let mut tw = Self::new(rows, cols);

        for r in 0..rows {
            let row_start = r * cols;
            let row = &weights[row_start..row_start + cols];

            // Compute scale = mean(|row|)
            let abs_sum = crate::simd::simd_sum_abs_f32(row);
            let scale = abs_sum / cols as f32;
            tw.row_scale[r] = if scale > 0.0 { scale } else { 1.0 };

            let threshold = 0.5 * tw.row_scale[r];
            let mut carry = 0.0f32;

            // Inline bit manipulation to avoid per-element bounds checks in set()
            let row_base = r * tw.blocks64;
            for (c, &val) in row.iter().enumerate() {
                let adjusted = val + carry;
                let q = if adjusted > threshold {
                    1i8
                } else if adjusted < -threshold {
                    -1i8
                } else {
                    0i8
                };
                let block = c >> 6;
                let bit = c & 63;
                let mask = 1u64 << bit;
                let idx = row_base + block;
                // Branch-free: clear both bits, then set the one that matches q
                tw.pos_bits[idx] &= !mask;
                tw.neg_bits[idx] &= !mask;
                // q is 1 or -1 or 0; only set the relevant bit
                tw.pos_bits[idx] |= (q == 1) as u64 * mask;
                tw.neg_bits[idx] |= (q == -1) as u64 * mask;
                carry = adjusted - (q as f32 * tw.row_scale[r]);
            }
        }

        tw
    }

    /// Compute a checksum over all values (sum of row_scale[r] * sum of signs in row r).
    /// Used for cross-implementation verification.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        for r in 0..self.rows {
            // Accumulate as integer to avoid per-element f32 conversion overhead.
            let mut row_sum: i32 = 0;
            let row_base = r * self.blocks64;
            for b in 0..self.blocks64 {
                let idx = row_base + b;
                row_sum += self.pos_bits[idx].count_ones() as i32;
                row_sum -= self.neg_bits[idx].count_ones() as i32;
            }
            total += self.row_scale[r] * row_sum as f32;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Hydra Adaptive Layer Budget (Research 148, Plan 165)
// ---------------------------------------------------------------------------

/// Per-layer Hydra profile entry (modelless mode).
/// Pre-computed from calibration data, stored in config.
#[cfg(feature = "hydra_budget")]
#[derive(Clone, Debug)]
pub struct HydraLayerProfile {
    /// Mean absolute direct effect on top-token logit.
    pub mean_de: f32,
    /// Fraction of prompts where this layer is a Hydra backup.
    pub backup_frequency: f32,
    /// Whether this layer acts as erasure (mean DE < 0 for MLP).
    pub is_erasure: bool,
}

/// Hydra budget configuration.
#[cfg(feature = "hydra_budget")]
#[derive(Clone, Debug)]
pub struct HydraBudgetConfig {
    /// Skip layers with |DE| below this threshold.
    pub skip_threshold: f32,
    /// Early-terminate when cumulative DE reaches this fraction of total.
    pub cumulative_threshold: f32,
    /// Use modelless mode (lookup) vs model-based (logit lens).
    pub modelless: bool,
    /// Skip erasure MLPs during draft stage.
    pub skip_erasure_draft: bool,
}

#[cfg(feature = "hydra_budget")]
impl Default for HydraBudgetConfig {
    fn default() -> Self {
        Self {
            skip_threshold: 0.01,
            cumulative_threshold: 0.95,
            modelless: true,
            skip_erasure_draft: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests_types {
    use super::*;

    #[test]
    fn test_with_overrides_none_unchanged() {
        let config = Config::draft();
        let overrides = InferenceOverrides::default();
        let original_tb = config.tree_budget;
        let original_temp = config.temperature;
        let original_dl = config.draft_lookahead;
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, original_tb);
        assert_eq!(result.temperature, original_temp);
        assert_eq!(result.draft_lookahead, original_dl);
    }

    #[test]
    fn test_with_overrides_some_applied() {
        let config = Config::draft();
        let overrides = InferenceOverrides {
            tree_budget: Some(99),
            temperature: Some(0.123),
            ..Default::default()
        };
        let original_dl = config.draft_lookahead;
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, 99);
        assert!((result.temperature - 0.123).abs() < 1e-6);
        // Non-overridden fields stay the same
        assert_eq!(result.draft_lookahead, original_dl);
    }

    #[test]
    fn test_with_overrides_all_fields() {
        let config = Config::draft();
        let overrides = InferenceOverrides {
            tree_budget: Some(1),
            draft_lookahead: Some(2),
            parallel_threshold: Some(3),
            screening_threshold: Some(0.1),
            temperature: Some(0.2),
            sparse_threshold: Some(0.3),
            early_exit_patience: Some(4),
            early_exit_gap: Some(0.4),
            mtp_activation_threshold: Some(5),
            mtp_cluster_vocab_threshold: Some(6),
            mtp_shared_kv_prompt_threshold: Some(7),
            mtp_cluster_size: Some(8),
            mtp_min_output_tokens: Some(10),
            mtp_cluster_topk: Some(2),
            sp_kv_threshold: Some(0.5),
            width_rollouts: Some(9),
            early_stop_threshold: Some(0.6),
            convergence_selector: Some(ConvergenceSelector::Top1Converged),
            mls_layers: Some(3),
            // drafter_lora_path is consumed by the caller, not applied to Config
            drafter_lora_path: None,
            max_plan_horizon: Some(5),
            #[cfg(feature = "hydra_budget")]
            hydra_skip_threshold: None,
            #[cfg(feature = "hydra_budget")]
            hydra_skip_erasure_draft: None,
            depth_tier: None,
        };
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, 1);
        assert_eq!(result.draft_lookahead, 2);
        assert_eq!(result.parallel_threshold, 3);
        assert!((result.screening_threshold - 0.1).abs() < 1e-6);
        assert!((result.temperature - 0.2).abs() < 1e-6);
        assert!((result.sparse_threshold - 0.3).abs() < 1e-6);
        assert_eq!(result.early_exit_patience, 4);
        assert!((result.early_exit_gap - 0.4).abs() < 1e-6);
        assert_eq!(result.mtp_activation_threshold, 5);
        assert_eq!(result.mtp_cluster_vocab_threshold, 6);
        assert_eq!(result.mtp_shared_kv_prompt_threshold, 7);
        assert_eq!(result.mtp_cluster_size, 8);
        assert_eq!(result.mtp_min_output_tokens, 10);
        assert_eq!(result.mtp_cluster_topk, 2);
        assert!((result.sp_kv_threshold - 0.5).abs() < 1e-6);
        assert_eq!(result.width_rollouts, 9);
        assert!((result.early_stop_threshold - 0.6).abs() < 1e-6);
        assert_eq!(
            result.convergence_selector,
            ConvergenceSelector::Top1Converged
        );
        assert_eq!(result.mls_layers, 3);
        // max_plan_horizon caps draft_lookahead when lower (Plan 112 T11)
        assert_eq!(result.draft_lookahead, 2); // 2 (from override) < 5 (horizon cap), stays 2
    }

    #[test]
    fn test_early_exit_defaults_disabled() {
        let config = Config::micro();
        assert_eq!(config.early_exit_patience, 0);
        assert!((config.early_exit_gap).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_save_load_roundtrip() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_domain_latent.bin");
        let original = DomainLatent::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        original.save(&tmp).unwrap();
        let loaded = DomainLatent::load(&tmp).unwrap();
        assert_eq!(original.embedding, loaded.embedding);
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_zeros() {
        let dl = DomainLatent::zeros(8);
        assert_eq!(dl.embedding.len(), 8);
        assert!(dl.embedding.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_invalid_magic() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_bad_magic.bin");
        let mut buf = b"XXXX".to_vec();
        buf.push(1); // version
        buf.extend_from_slice(&4u32.to_le_bytes()); // kv_dim
        buf.extend_from_slice(
            &[
                0.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
            ]
            .concat(),
        );
        let hash = blake3::hash(&buf);
        buf.extend_from_slice(hash.as_bytes());
        std::fs::write(&tmp, &buf).unwrap();
        assert!(DomainLatent::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_checksum_mismatch() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_bad_checksum.bin");
        let mut buf = b"DLAT".to_vec();
        buf.push(1); // version
        buf.extend_from_slice(&4u32.to_le_bytes()); // kv_dim
        buf.extend_from_slice(&[0.0f32.to_le_bytes(); 4].concat());
        buf.extend_from_slice(&[0u8; 32]); // wrong checksum
        std::fs::write(&tmp, &buf).unwrap();
        assert!(DomainLatent::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_file_too_small() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_too_small.bin");
        std::fs::write(&tmp, b"DLAT").unwrap();
        assert!(DomainLatent::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_config_game() {
        let config = Config::game();
        assert_eq!(config.vocab_size, 10);
        assert_eq!(config.block_size, 170);
        assert_eq!(config.n_embd, 32);
        assert_eq!(config.n_head, 4);
        assert_eq!(config.head_dim, 8);
        assert_eq!(config.mlp_hidden, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_isometry() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert!(adapter.check_isometry(256));
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_apply_correctness() {
        let adapter = GpartAdapter {
            d: 2,
            seed: 123,
            theta: vec![1.0, -1.0],
        };
        let mut weights = vec![0.0f32; 8];
        adapter.apply(&mut weights);
        // Every weight should have changed (each element assigned to group 0 or 1)
        assert!(weights.iter().all(|&w| w != 0.0));
        // Verify deterministic: apply again with same adapter should produce same result
        let mut weights2 = vec![0.0f32; 8];
        adapter.apply(&mut weights2);
        assert_eq!(weights, weights2);
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_commitment_roundtrip() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 99,
            theta: vec![0.5, -0.5, 1.0, -1.0],
        };
        let commit = adapter.commitment();
        assert!(adapter.verify(&commit));
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_tamper_detection() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 99,
            theta: vec![0.5, -0.5, 1.0, -1.0],
        };
        let commit = adapter.commitment();
        // Tampered commitment
        let mut tampered = commit;
        tampered[0] ^= 0xFF;
        assert!(!adapter.verify(&tampered));
        // Different theta produces different commitment
        let tampered_adapter = GpartAdapter {
            d: 4,
            seed: 99,
            theta: vec![0.5, -0.5, 1.0, 0.0], // last element changed
        };
        assert_ne!(adapter.commitment(), tampered_adapter.commitment());
        // Different seed produces different commitment
        let seed_adapter = GpartAdapter {
            d: 4,
            seed: 100,
            theta: vec![0.5, -0.5, 1.0, -1.0],
        };
        assert_ne!(adapter.commitment(), seed_adapter.commitment());
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_save_load_roundtrip() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_gpart.bin");
        let adapter = GpartAdapter {
            d: 8,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        adapter.save(&tmp).unwrap();
        let loaded = GpartAdapter::load(&tmp).unwrap();
        assert_eq!(loaded.d, adapter.d);
        assert_eq!(loaded.seed, adapter.seed);
        assert_eq!(loaded.theta, adapter.theta);
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_determinism_same_seed() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut w1 = vec![0.0f32; 64];
        let mut w2 = vec![0.0f32; 64];
        adapter.apply(&mut w1);
        adapter.apply(&mut w2);
        assert_eq!(w1, w2);
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_masked_all_true_matches_unmasked() {
        // When group_mask is all-true, masked apply must equal unmasked apply.
        let adapter = GpartAdapter {
            d: 4,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let n = 256;
        let mut w_ref = vec![0.0f32; n];
        let mut assignments = vec![0usize; n];
        let mut group_sizes = vec![0usize; adapter.d];
        adapter.apply_with_scratch(&mut w_ref, &mut assignments, &mut group_sizes);

        let mut w_masked = vec![0.0f32; n];
        let mask = vec![true; adapter.d];
        adapter.apply_with_scratch_masked(
            &mut w_masked,
            &mut assignments,
            &mut group_sizes,
            &mask,
        );

        for (i, (a, b)) in w_ref.iter().zip(w_masked.iter()).enumerate() {
            assert_eq!(a, b, "mismatch at index {i}");
        }
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_masked_zeroes_masked_groups() {
        // If group g is masked off, NO base_weights element assigned to g
        // should receive any delta from theta[g]. Concretely: the masked
        // delta for elements of masked groups must be exactly zero.
        let adapter = GpartAdapter {
            d: 4,
            seed: 7,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let n = 512;
        let mut w_zero = vec![0.0f32; n]; // baseline: all groups masked off
        let mut assignments = vec![0usize; n];
        let mut group_sizes = vec![0usize; adapter.d];
        let all_false = vec![false; adapter.d];
        adapter.apply_with_scratch_masked(
            &mut w_zero,
            &mut assignments,
            &mut group_sizes,
            &all_false,
        );
        // With every group masked, weights should remain exactly zero.
        assert!(w_zero.iter().all(|&w| w == 0.0));
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_masked_single_group_preserves_others() {
        // Masking group 0 only: every element in groups 1..d should still receive
        // its full delta, while elements in group 0 should stay at baseline.
        let adapter = GpartAdapter {
            d: 4,
            seed: 7,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let n = 512;

        // Full (unmasked) reference.
        let mut w_full = vec![0.0f32; n];
        let mut assignments_full = vec![0usize; n];
        let mut group_sizes = vec![0usize; adapter.d];
        adapter.apply_with_scratch(&mut w_full, &mut assignments_full, &mut group_sizes);

        // Masked: drop group 0.
        let mut w_masked = vec![0.0f32; n];
        let mut assignments = vec![0usize; n];
        let mut mask = vec![true; adapter.d];
        mask[0] = false;
        adapter.apply_with_scratch_masked(
            &mut w_masked,
            &mut assignments,
            &mut group_sizes,
            &mask,
        );

        // assignments[] is regenerated deterministically from seed, so it matches
        // between the two calls — compare per-element using the masked call's
        // assignment vector.
        for i in 0..n {
            let g = assignments[i];
            if g == 0 {
                // Group 0 masked off → masked output should equal baseline (0.0).
                assert_eq!(w_masked[i], 0.0, "group-0 element {i} should be zero");
            } else {
                // Other groups untouched → should equal full reference.
                assert_eq!(
                    w_masked[i], w_full[i],
                    "group-{g} element {i} should match unmasked"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_topk_mask_edge_cases() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 0,
            theta: vec![0.1, 0.5, 0.3, 0.9],
        };
        // k >= d → all active
        assert!(adapter.topk_mask(adapter.d).iter().all(|&b| b));
        assert!(adapter.topk_mask(adapter.d + 10).iter().all(|&b| b));
        // k == 0 → all inactive
        assert!(adapter.topk_mask(0).iter().all(|&b| !b));
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_topk_mask_selects_largest_magnitudes() {
        // θ = [0.1, 0.5, -0.3, 0.9] → |θ| = [0.1, 0.5, 0.3, 0.9]
        // top-2 by magnitude should keep groups {3 (0.9), 1 (0.5)}.
        let adapter = GpartAdapter {
            d: 4,
            seed: 0,
            theta: vec![0.1, 0.5, -0.3, 0.9],
        };
        let mask = adapter.topk_mask(2);
        assert_eq!(mask.len(), 4);
        assert!(mask[1] && mask[3], "groups 1 and 3 must be in top-2");
        assert!(!mask[0] && !mask[2], "groups 0 and 2 must be pruned");
        // Exactly 2 active.
        assert_eq!(mask.iter().filter(|&&b| b).count(), 2);
    }

    // ── Issue 299: multi-adapter LoRA loading ──────────────────────

    /// Build a Plan 008 LoRA binary file with `n` adapters for testing.
    /// Each adapter's A/B matrices are filled with a distinct sentinel pattern
    /// so we can verify the loader returns the right data per adapter.
    fn build_test_lora_file(path: &std::path::Path, n_adapters: usize, rank: usize) {
        let in_dim = 4usize;
        let out_dim = 4usize;
        let alpha = 8.0f32;

        // Payload: [n_adapters(4)][rank(4)][alpha(4)] + per-adapter [in_dim(4)][out_dim(4)][A][B]
        let header = 4 + 4 + 4;
        let per_adapter = 4 + 4 + (rank * in_dim + out_dim * rank) * std::mem::size_of::<f32>();
        let mut payload = Vec::with_capacity(header + n_adapters * per_adapter);

        payload.extend_from_slice(&(n_adapters as u32).to_le_bytes());
        payload.extend_from_slice(&(rank as u32).to_le_bytes());
        payload.extend_from_slice(&alpha.to_le_bytes());

        for i in 0..n_adapters {
            payload.extend_from_slice(&(in_dim as u32).to_le_bytes());
            payload.extend_from_slice(&(out_dim as u32).to_le_bytes());
            // A: [rank × in_dim], sentinel = (i + 1) * 0.1
            let a_sentinel = (i as f32 + 1.0) * 0.1;
            for _ in 0..(rank * in_dim) {
                payload.extend_from_slice(&a_sentinel.to_le_bytes());
            }
            // B: [out_dim × rank], sentinel = (i + 1) * -0.2
            let b_sentinel = (i as f32 + 1.0) * -0.2;
            for _ in 0..(out_dim * rank) {
                payload.extend_from_slice(&b_sentinel.to_le_bytes());
            }
        }

        let checksum = blake3::hash(&payload);
        let mut file_data = Vec::with_capacity(4 + 4 + 32 + payload.len());
        file_data.extend_from_slice(b"LORA");
        file_data.extend_from_slice(&1u32.to_le_bytes());
        file_data.extend_from_slice(checksum.as_bytes());
        file_data.extend_from_slice(&payload);
        std::fs::write(path, &file_data).unwrap();
    }

    #[test]
    fn test_lora_load_single_adapter_returns_one_element_vec() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_single.bin");
        build_test_lora_file(&tmp, 1, 4);
        let adapters = LoraAdapter::load(&tmp).expect("single-adapter load should succeed");
        assert_eq!(adapters.len(), 1, "single-adapter file must yield 1-element Vec");
        assert_eq!(adapters[0].rank, 4);
        assert_eq!(adapters[0].in_dim, 4);
        assert_eq!(adapters[0].out_dim, 4);
        assert!((adapters[0].alpha - 8.0).abs() < 1e-6);
        drop(std::fs::remove_file(&tmp));
    }

    /// Issue 299 regression: a 12-adapter file (L2 model, 6 adapters/layer × 2 layers)
    /// must load ALL 12 adapters. Previously `load()` returned only the first.
    #[test]
    fn test_lora_load_multi_adapter_returns_all() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_multi.bin");
        build_test_lora_file(&tmp, 12, 4);
        let adapters = LoraAdapter::load(&tmp).expect("12-adapter load should succeed");
        assert_eq!(
            adapters.len(),
            12,
            "12-adapter file must yield 12-element Vec (Issue 299 regression)"
        );

        // Verify each adapter has its own sentinel data — proves layers 1..11 aren't dropped
        for (i, a) in adapters.iter().enumerate() {
            let expected_a = (i as f32 + 1.0) * 0.1;
            let expected_b = (i as f32 + 1.0) * -0.2;
            assert_eq!(a.rank, 4, "adapter {i} rank");
            assert_eq!(a.in_dim, 4, "adapter {i} in_dim");
            assert_eq!(a.out_dim, 4, "adapter {i} out_dim");
            assert_eq!(a.a.len(), 4 * 4, "adapter {i} A matrix size");
            assert_eq!(a.b.len(), 4 * 4, "adapter {i} B matrix size");
            assert!(
                a.a.iter().all(|&v| (v - expected_a).abs() < 1e-6),
                "adapter {i} A data mismatch"
            );
            assert!(
                a.b.iter().all(|&v| (v - expected_b).abs() < 1e-6),
                "adapter {i} B data mismatch"
            );
        }
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_first_returns_only_first_adapter() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_first.bin");
        build_test_lora_file(&tmp, 6, 4);
        let first = LoraAdapter::load_first(&tmp).expect("load_first should succeed");
        assert_eq!(first.rank, 4);
        // Adapter 0 sentinel is 1.0 * 0.1 = 0.1
        assert!(first.a.iter().all(|&v| (v - 0.1).abs() < 1e-6));
        assert!(first.b.iter().all(|&v| (v - (-0.2)).abs() < 1e-6));
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_rejects_truncated_multi_adapter() {
        // Truncate a 12-adapter file so only 6 adapters' bytes fit — loader must error
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_truncated.bin");
        build_test_lora_file(&tmp, 12, 4);
        let full = std::fs::read(&tmp).unwrap();
        // Cut the payload in half (keep header + checksum + ~half the adapters)
        let cut = full.len() / 2 + 20; // keep enough to pass header check
        std::fs::write(&tmp, &full[..cut]).unwrap();
        // Re-generate checksum won't match — but if it did, the truncation check fires.
        // Either way, load must fail (not silently return wrong number of adapters).
        let result = LoraAdapter::load(&tmp);
        assert!(result.is_err(), "truncated multi-adapter file must error, not silently drop adapters");
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_rejects_bad_magic() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_bad_magic.bin");
        std::fs::write(&tmp, b"NOPE\0\0\0\0").unwrap();
        assert!(LoraAdapter::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_rejects_zero_adapters() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_zero.bin");
        // Build a valid-checksum file that declares 0 adapters
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_le_bytes()); // n_adapters = 0
        payload.extend_from_slice(&4u32.to_le_bytes()); // rank
        payload.extend_from_slice(&8.0f32.to_le_bytes()); // alpha
        let checksum = blake3::hash(&payload);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LORA");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(checksum.as_bytes());
        buf.extend_from_slice(&payload);
        std::fs::write(&tmp, &buf).unwrap();
        assert!(LoraAdapter::load(&tmp).is_err(), "zero-adapter declaration must be rejected");
        drop(std::fs::remove_file(&tmp));
    }
}

// ---------------------------------------------------------------------------
// Shard Embedding — JL Random Orthogonal Projection (Plan 230)
// ---------------------------------------------------------------------------

/// Low-dimensional projection of NeuronShard style_weights for fast similarity search.
/// Produced by Johnson-Lindenstrauss random orthogonal projection.
/// 8 × f32 = 32 bytes — fits in cache line, suitable for SIMD cosine similarity.
///
/// Plan 230: Shard Embedding Projection — modelless linear weight-to-vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShardEmbedding(pub [f32; 8]);

impl ShardEmbedding {
    pub const ZERO: Self = Self([0.0; 8]);
    pub const DIM: usize = 8;

    /// Cosine similarity between two embeddings.
    /// Uses SIMD dot-product and SIMD sum-of-squares for the normalization.
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = crate::simd::simd_dot_f32(&self.0, &other.0, Self::DIM);
        let sq_a = crate::simd::simd_sum_sq(&self.0, Self::DIM);
        let sq_b = crate::simd::simd_sum_sq(&other.0, Self::DIM);
        let denom = sq_a * sq_b;
        if denom < 1e-16 {
            return 0.0;
        }
        let inv_norm = 1.0 / denom.sqrt();
        dot * inv_norm
    }

    /// Euclidean distance squared between two embeddings.
    /// Uses SIMD-accelerated fused-subtract-accumulate.
    #[inline]
    pub fn dist_sq(&self, other: &Self) -> f32 {
        crate::simd::simd_dist_sq(&self.0, &other.0, 8)
    }
}

impl Default for ShardEmbedding {
    fn default() -> Self {
        Self::ZERO
    }
}

// Hash for use as HashMap key (bit-level, NOT semantic hash)
impl std::hash::Hash for ShardEmbedding {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Single write of all 32 bytes — fewer virtual calls than four write_u64.
        state.write(unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, 32) });
    }
}

impl Eq for ShardEmbedding {}

// ---------------------------------------------------------------------------
// Sense Composition — KG Latent Octree (Plan 221)
// ---------------------------------------------------------------------------

/// Kind of sense module for NPC brain composition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SenseKind {
    CommonSense = 0,
    FighterSense = 1,
    GameTheorySense = 2,
    #[default]
    SpatialSense = 3,
    SocialSense = 4,
    SkillSense = 5,
    /// LinOSS Modal Threat Prediction (Plan 241). Requires `spectral_threat` feature.
    #[cfg(feature = "spectral_threat")]
    SpectralThreat = 6,
    Reserved = 7,
}

/// Ternary direction vector: +1/0/-1 encoded as two bitmasks + row scale.
/// 20 bytes each (u64, u64, f32).
/// `#[repr(C)]` required — embedded in `SenseModule` which hashes raw bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TernaryDir {
    /// Bitmask for positive (+1) entries.
    pub pos_bits: u64,
    /// Bitmask for negative (-1) entries.
    pub neg_bits: u64,
    /// Scale factor for this direction row.
    pub row_scale: f32,
}

impl TernaryDir {
    pub const SIZE: usize = 20;

    pub fn zero() -> Self {
        Self {
            pos_bits: 0,
            neg_bits: 0,
            row_scale: 0.0,
        }
    }

    /// Zero padding bytes for deterministic hashing.
    /// TernaryDir is 20 logical bytes but 24 with alignment padding.
    /// This zeroes the 4 trailing padding bytes in one write.
    #[inline(always)]
    pub fn zero_padding(&mut self) {
        // Write 4 zero bytes at once via u32 store instead of 4 individual byte writes
        unsafe {
            let ptr = self as *mut Self as *mut u8;
            (ptr.add(20) as *mut u32).write(0);
        }
    }
}

/// Fixed-size sense module: KG latent octree + ternary direction vectors.
/// ~232 bytes. BLAKE3 committed.
///
/// Field ordering: u64-aligned first, then f32, then u8 tail.
/// `commitment` must remain LAST for the hash-until-end-of-struct pattern.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct SenseModule {
    /// Octree occupancy bit-planes (up to depth 3 → 128 nodes in 2×u64).
    pub octree_bits: [u64; 4],
    /// Ternary direction vectors for projection.
    pub directions: [TernaryDir; 8],
    /// Module confidence [0, 1].
    pub confidence: f32,
    pub kind: SenseKind,
    pub version: u8,
    pub octree_depth: u8,
    pub n_directions: u8,
    pub _reserved: u8,
    /// BLAKE3 commitment over all preceding fields.
    pub commitment: [u8; 32],
}

/// Fast rational sigmoid approximation — avoids `exp()` (~15ns) for ~1ns cost.
/// Max absolute error: ~0.003 vs `1/(1+exp(-x))`.
#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    let x = x.clamp(-12.0, 12.0);
    0.5 + x / (2.0 + (4.0 + x * x).sqrt())
}

impl SenseModule {
    /// Project HLA state onto this module's ternary directions → sigmoid scalar.
    ///
    /// KG weight bridge: output is scaled by module confidence so that
    /// high-confidence KG triples produce stronger sense activations and
    /// low-confidence triples are attenuated. Confidence 1.0 = unchanged.
    ///
    /// Optimized: branch-free bit extraction via shift+AND, flat loop
    /// (LLVM auto-vectorizes better than chunked), fast rational sigmoid.
    #[inline(always)]
    pub fn project(&self, hla_state: &[f32; 8]) -> f32 {
        let n = self.n_directions as usize;
        let mut dot = 0.0f32;

        // Unrolled-friendly flat loop: extract ternary sign per-dim, FMA into dot.
        // bool-as-u32 then cast to f32 is zero-extend (no int-to-float conversion).
        for (i, hla_val) in hla_state.iter().enumerate().take(n) {
            let dir = &self.directions[i];
            let pos = ((dir.pos_bits >> i) & 1) as u32 as f32;
            let neg = ((dir.neg_bits >> i) & 1) as u32 as f32;
            // sign ∈ {-1, 0, +1} — FMA: dot += sign * hla * scale
            dot += (pos - neg) * hla_val * dir.row_scale;
        }

        // Fast sigmoid * confidence
        self.confidence * fast_sigmoid(dot)
    }

    /// Query octree occupancy at given level and index.
    /// Returns None if indices out of bounds.
    pub fn query_octree(&self, level: u8, index: u8) -> Option<bool> {
        let nodes_at_level = 1usize << (level * 2); // quadtree-like
        if index as usize >= nodes_at_level || level > self.octree_depth {
            return None;
        }
        // flat_idx = 0 for all levels in this simplified indexing;
        // the fold computes 0 * 4^level = 0, so flat_idx = index.
        let flat_idx = index as usize;
        let word = flat_idx / 64;
        let bit = flat_idx % 64;
        if word >= self.octree_bits.len() {
            return None;
        }
        Some(self.octree_bits[word] & (1 << bit) != 0)
    }

    /// Compute and store BLAKE3 commitment.
    /// Zeros TernaryDir padding bytes first for deterministic hashing.
    pub fn commit(&mut self) {
        // Zero commitment before hashing
        self.commitment = [0u8; 32];
        // Zero padding in direction vectors for deterministic hash
        for dir in &mut self.directions {
            dir.zero_padding();
        }
        let size_before_commit = std::mem::offset_of!(SenseModule, commitment);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, size_before_commit)
        };
        self.commitment = *blake3::hash(bytes).as_bytes();
    }

    /// Verify BLAKE3 commitment.
    /// Uses a stack buffer to avoid cloning the entire 232-byte struct.
    /// Zeros TernaryDir padding bytes before comparing to match commit() behavior.
    pub fn verify(&self) -> bool {
        let size_before_commit = std::mem::offset_of!(SenseModule, commitment);
        let mut buf = [0u8; std::mem::size_of::<SenseModule>()];
        // Copy raw bytes to stack buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                self as *const Self as *const u8,
                buf.as_mut_ptr(),
                size_before_commit,
            );
        }
        // Zero commitment region and trailing padding in buffer
        buf[size_before_commit..].fill(0);
        // Zero TernaryDir padding in buffer — each TernaryDir is 24 bytes, padding at bytes 20..24
        let dirs_offset = std::mem::offset_of!(SenseModule, directions);
        let dir_size = std::mem::size_of::<TernaryDir>();
        for i in 0..8 {
            let dir_start = dirs_offset + i * dir_size;
            // Zero 4 padding bytes at offset 20 within each TernaryDir
            buf[dir_start + 20..dir_start + dir_size].fill(0);
        }
        let bytes = &buf[..size_before_commit];
        let expected = blake3::hash(bytes);
        self.commitment == *expected.as_bytes()
    }
}

impl Default for SenseModule {
    fn default() -> Self {
        Self {
            octree_bits: [0; 4],
            directions: [TernaryDir::zero(); 8],
            confidence: 0.0,
            kind: SenseKind::default(),
            version: 1,
            octree_depth: 3,
            n_directions: 0,
            _reserved: 0,
            commitment: [0u8; 32],
        }
    }
}

// ---------------------------------------------------------------------------
// DilationConfig — RAT+ Recurrence Bridge sparse attention (Plan 225)
// ---------------------------------------------------------------------------

/// Dilation configuration for RAT+ bridge sparse attention.
/// Controls stride D for KV cache access during decode.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DilationConfig {
    D1 = 1, // Dense (no dilation)
    D2 = 2,
    D4 = 4,
    D8 = 8,
    D16 = 16,
    D32 = 32,
    D64 = 64,
}

impl DilationConfig {
    /// Returns the dilation stride as usize.
    #[inline]
    pub fn stride(&self) -> usize {
        *self as usize
    }

    /// Select dilation from queries-per-second heuristic.
    /// Low QPS → dense, High QPS → aggressive dilation.
    pub fn from_qps(qps: f32) -> Self {
        match qps {
            ..1.0 => Self::D1,
            1.0..5.0 => Self::D4,
            5.0..20.0 => Self::D16,
            _ => Self::D64,
        }
    }
}
