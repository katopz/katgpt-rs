//! Small configuration enums and feature-config structs.

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
///
/// Kept compiled even when `delta_routing` is off so config round-trips
/// serialize identically across feature sets. Reachable via `Config` defaults
/// once the routing backend lands.
#[allow(dead_code)]
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
#[allow(dead_code)]
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

// DeltaRoutingConfig::delta_block / is_enabled are intended for the
// delta_routing backend (Plan 097) which is still being wired up. Silence
// dead-code until callers land.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
