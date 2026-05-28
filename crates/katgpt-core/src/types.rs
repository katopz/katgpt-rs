// Shared configuration, RNG, and math utilities.
// Superset of types from both katgpt-rs and riir-engine projects.

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Attention mode for HLA (Higher-order Linear Attention).
///
/// - `Standard`: SDPA with KV cache (default, backward-compatible).
/// - `Hla`: Symmetric second-order linear attention — O(1) per-token memory.
/// - `Ahla`: Asymmetric second-order linear attention — lower state cost than symmetric.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug)]
pub struct DeltaRoutingConfig {
    /// Routing mode.
    pub mode: DeltaRoutingMode,
    /// Block size for DeltaBlock mode (number of layers per block).
    /// Default: 4. Paper recommends B=4.
    pub block_size: usize,
}

impl Default for DeltaRoutingConfig {
    fn default() -> Self {
        Self {
            mode: DeltaRoutingMode::Off,
            block_size: 4,
        }
    }
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
#[derive(Clone, Copy, Debug)]
pub struct RtTurboConfig {
    /// Fraction of heads classified as retrieval heads (default: 0.15).
    /// Paper ablation: 15% is optimal balance of accuracy vs sparsity.
    pub retrieval_head_ratio: f32,
    /// Low-dimensional projection size for pre-RoPE scoring (default: 16).
    /// Paper ablation: dim=16 is the sweet spot for low-frequency retrieval.
    pub low_dim: usize,
    /// Cumulative attention mass threshold for dynamic top-p selection (default: 0.9).
    /// Paper ablation: top-p=0.9 preserves >93% attention mass at 97% sparsity.
    pub top_p: f32,
    /// Sliding window size for local heads (default: 8192).
    pub sliding_window: usize,
    /// Number of attention sink tokens always retained for local heads (default: 4).
    pub sink_tokens: usize,
    /// Block size for block-level top-p variant (default: 64).
    /// Should match `DashAttnConfig::chunk_size` for consistent routing.
    pub block_size: usize,
}

impl Default for RtTurboConfig {
    fn default() -> Self {
        Self {
            retrieval_head_ratio: 0.15,
            low_dim: 16,
            top_p: 0.9,
            sliding_window: 8192,
            sink_tokens: 4,
            block_size: 64,
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
        for (i, gate_val) in temp.iter_mut().enumerate().take(n) {
            let w_off = i * dim;
            let dot: f32 = self.w_gate[w_off..w_off + dim]
                .iter()
                .zip(attn_out.iter())
                .map(|(w, a)| w * a)
                .sum();
            *gate_val = 1.0 / (1.0 + (-dot).exp()); // sigmoid
        }

        // Step 2: Apply gate elementwise
        for (attn, gate) in attn_out.iter_mut().zip(temp.iter()) {
            *attn *= gate;
        }
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
    pub entropy_bin: usize,
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

    // --- Enum types with usize discriminant (8-byte aligned) ---
    // HLA Attention (Plan 057: Higher-order Linear Attention)
    pub hla_mode: HlaMode,
    // Gemma 2 architecture fields (Plan 087)
    pub model_arch: ModelArchitecture,
    // D2F Discrete Diffusion Forcing (Plan 066)
    pub attention_mode: AttentionMode,
    // SP-KV self-pruned KV attention (Plan 070)
    pub sp_kv_predictor_lr_mult: f32,
    // EqR Convergence Selection (Plan 119)
    pub convergence_selector: ConvergenceSelector,
    // LT2 Looped Inference Pipeline (Plan 108, Research 73)
    pub loop_mode: LoopMode,
    pub hybrid_pattern: HybridPattern,

    // --- f32 (4-byte aligned) ---
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

    // --- WeightDtype (could be u8 or enum, place after f32s) ---
    pub weight_dtype: WeightDtype,

    // --- Vec (pointer-sized, 8-byte aligned) ---
    pub lora_targets: Vec<String>,

    // --- bool / u8 fields (1-byte aligned, tail-packed) ---
    pub hla_normalize: bool,
    pub rms_norm_offset: bool,
    pub tied_embeddings: bool,
    pub use_rope: bool,
    pub post_norm: bool,
    pub gated_attn: bool,
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
        if self.model_arch != ModelArchitecture::Gemma2
            && self.model_arch != ModelArchitecture::Llama
            && self.n_head * self.head_dim != self.n_embd
        {
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
    /// `None` fields are left unchanged; `Some` fields replace the current value.
    /// Used by the router to inject domain-specific budgets from TOML config.
    pub fn with_overrides(&self, overrides: &InferenceOverrides) -> Self {
        let mut c = self.clone();
        if let Some(v) = overrides.tree_budget {
            c.tree_budget = v;
        }
        if let Some(v) = overrides.draft_lookahead {
            c.draft_lookahead = v;
        }
        if let Some(v) = overrides.parallel_threshold {
            c.parallel_threshold = v;
        }
        if let Some(v) = overrides.screening_threshold {
            c.screening_threshold = v;
        }
        if let Some(v) = overrides.temperature {
            c.temperature = v;
        }
        if let Some(v) = overrides.sparse_threshold {
            c.sparse_threshold = v;
        }
        if let Some(v) = overrides.early_exit_patience {
            c.early_exit_patience = v;
        }
        if let Some(v) = overrides.early_exit_gap {
            c.early_exit_gap = v;
        }
        if let Some(v) = overrides.mtp_activation_threshold {
            c.mtp_activation_threshold = v;
        }
        if let Some(v) = overrides.mtp_cluster_vocab_threshold {
            c.mtp_cluster_vocab_threshold = v;
        }
        if let Some(v) = overrides.mtp_shared_kv_prompt_threshold {
            c.mtp_shared_kv_prompt_threshold = v;
        }
        if let Some(v) = overrides.mtp_cluster_size {
            c.mtp_cluster_size = v;
        }
        if let Some(v) = overrides.mtp_min_output_tokens {
            c.mtp_min_output_tokens = v;
        }
        if let Some(v) = overrides.mtp_cluster_topk {
            c.mtp_cluster_topk = v;
        }
        if let Some(v) = overrides.sp_kv_threshold {
            c.sp_kv_threshold = v;
        }
        if let Some(v) = overrides.width_rollouts {
            c.width_rollouts = v;
        }
        if let Some(v) = overrides.early_stop_threshold {
            c.early_stop_threshold = v;
        }
        if let Some(v) = overrides.convergence_selector {
            c.convergence_selector = v;
        }
        if let Some(v) = overrides.mls_layers {
            c.mls_layers = v;
        }
        // SR²AM horizon truncation override (Plan 112 T11)
        if let Some(v) = overrides.max_plan_horizon {
            c.draft_lookahead = c.draft_lookahead.min(v);
        }
        c
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
pub struct InferenceOverrides {
    pub tree_budget: Option<usize>,
    pub draft_lookahead: Option<usize>,
    pub parallel_threshold: Option<usize>,
    pub screening_threshold: Option<f32>,
    pub temperature: Option<f32>,
    pub sparse_threshold: Option<f32>,
    pub early_exit_patience: Option<usize>,
    pub early_exit_gap: Option<f32>,
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
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
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

    /// Uniform [0, 1).
    #[inline(always)]
    pub fn uniform(&mut self) -> f32 {
        (self.next() >> 11) as f32 * (1.0 / 9007199254740992.0)
    }

    /// Standard normal via Box-Muller transform.
    #[inline]
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Math utilities — SIMD-accelerated
// ---------------------------------------------------------------------------

/// In-place softmax. Handles empty slices gracefully.
/// Three-pass: find max → exp+sum → normalize.
#[inline(always)]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find max for numerical stability (SIMD-accelerated)
    let max_val = crate::simd::simd_max_f32(x);

    // Pass 2: subtract max (SIMD-accelerated)
    crate::simd::simd_add_scalar_inplace(x, -max_val);

    // Pass 3: SIMD exp
    crate::simd::simd_exp_inplace(x);

    // Pass 4: sum + normalize (SIMD-accelerated sum)
    let sum: f32 = crate::simd::simd_sum_f32(x);
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

    // Pass 3: SIMD exp
    crate::simd::simd_exp_inplace(x);

    // Pass 4: sum + normalize (SIMD-accelerated sum)
    let sum: f32 = crate::simd::simd_sum_f32(x);
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

    // Pass 2: scale
    let inv_rms = 1.0 / (sum_sq / x.len() as f32 + 1e-5).sqrt();
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
        // buf[j] = -1.702 * gate[j] via copy + SIMD scale
        buf[..CHUNK].copy_from_slice(&gate[i..i + CHUNK]);
        crate::simd::simd_scale_inplace(&mut buf, -1.702);
        // buf[j] = exp(-1.702 * gate[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = gate[j] * sigmoid(1.702 * gate[j]) * up[j]
        for j in 0..CHUNK {
            let sigmoid = 1.0 / (1.0 + buf[j]);
            hidden[i + j] = gate[i + j] * sigmoid * up[i + j];
        }
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
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt(); // ≈0.7979
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = 2 * inner[j] for tanh approximation
        for j in 0..CHUNK {
            let g = gate[i + j];
            buf[j] = 2.0 * sqrt_2_over_pi * (g + 0.044715 * g * g * g);
        }
        // buf[j] = exp(2*inner[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = 0.5 * gate[j] * (1 + tanh(inner[j])) * up[j]
        // where tanh(inner) = (exp(2*inner) - 1) / (exp(2*inner) + 1)
        for j in 0..CHUNK {
            let exp_2inner = buf[j];
            let tanh_val = (exp_2inner - 1.0) / (exp_2inner + 1.0);
            let g = gate[i + j];
            hidden[i + j] = 0.5 * g * (1.0 + tanh_val) * up[i + j];
        }
        i += CHUNK;
    }
    // Scalar remainder
    for i in i..hidden.len() {
        let g = gate[i];
        let inner = sqrt_2_over_pi * (g + 0.044715 * g * g * g);
        let gelu_val = 0.5 * g * (1.0 + inner.tanh());
        hidden[i] = gelu_val * up[i];
    }
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
/// Used in LLaMA, Mistral, and other LLaMA-family models for SwiGLU MLP.
///
/// SIMD-accelerated: exp() computed via `simd_exp_inplace` on stack buffers.
#[inline]
pub fn silu(x: &mut [f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= x.len() {
        // buf[j] = -x[j] via copy + SIMD scale
        buf[..CHUNK].copy_from_slice(&x[i..i + CHUNK]);
        crate::simd::simd_scale_inplace(&mut buf, -1.0);
        // buf[j] = exp(-x[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // x[j] = x[j] / (1 + exp(-x[j]))
        for j in 0..CHUNK {
            x[i + j] /= 1.0 + buf[j];
        }
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
#[inline]
pub fn swiglu(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = -gate[j] via copy + SIMD scale
        buf[..CHUNK].copy_from_slice(&gate[i..i + CHUNK]);
        crate::simd::simd_scale_inplace(&mut buf, -1.0);
        // buf[j] = exp(-gate[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = gate[j] / (1 + exp(-gate[j])) * up[j]
        for j in 0..CHUNK {
            hidden[i + j] = gate[i + j] / (1.0 + buf[j]) * up[i + j];
        }
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
    let mut alive = 0;
    for c in 0..cols {
        if unsafe { *input.get_unchecked(c) } > 0.0 {
            unsafe {
                *active_indices.get_unchecked_mut(alive) = c;
                *active_values.get_unchecked_mut(alive) = *input.get_unchecked(c);
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
/// **Note:** This allocates a CDF Vec internally. For decode loops, prefer
/// [`sample_token_into()`] to avoid per-token heap allocation.
pub fn sample_token(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let n = probs.len();
    if n == 0 {
        return 0;
    }

    // Build cumulative sum array
    let mut cdf = Vec::with_capacity(n);
    let mut sum = 0.0f32;
    for &p in probs {
        sum += p;
        cdf.push(sum);
    }

    // Binary search: find the first index where cdf[i] > r
    match cdf.binary_search_by(|&c| {
        if c > r {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }) {
        Ok(i) | Err(i) => i.min(n - 1),
    }
}

/// Zero-alloc variant of [`sample_token`] that reuses a pre-allocated CDF buffer.
///
/// Pass a `cdf` buffer (e.g. `ForwardContext::cdf`) to avoid a ~vocab_size allocation
/// on every token decode. The buffer is cleared and refilled each call.
pub fn sample_token_into(probs: &[f32], rng: &mut Rng, cdf: &mut Vec<f32>) -> usize {
    let r = rng.uniform();
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
    match cdf[..n].binary_search_by(|&c| {
        if c > r {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }) {
        Ok(i) | Err(i) => i.min(n - 1),
    }
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
/// Zero-copy: loaded once per domain, reference-passed during inference.
pub struct LoraAdapter {
    /// Down-projection: [rank × in_dim]
    pub a: Vec<f32>,
    /// Up-projection: [out_dim × rank]
    pub b: Vec<f32>,
    /// LoRA rank.
    pub rank: usize,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Scaling factor (alpha / rank).
    pub alpha: f32,
}

impl LoraAdapter {
    /// Load a single-adapter LoRA file from the Plan 008 binary format.
    /// For multi-adapter files (multiple targets like Q, K, V), loads the first adapter.
    /// Returns the adapter with its rank, alpha, and weight matrices.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
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
        let n_adapters = read_u32_le(payload, &mut offset)?;
        let rank = read_u32_le(payload, &mut offset)? as usize;
        let alpha = read_f32_le(payload, &mut offset)?;

        if n_adapters == 0 {
            return Err("No adapters in lora file".into());
        }

        // Load first adapter
        let in_dim = read_u32_le(payload, &mut offset)? as usize;
        let out_dim = read_u32_le(payload, &mut offset)? as usize;

        let a_count = rank * in_dim;
        let b_count = out_dim * rank;
        let a_bytes = a_count * std::mem::size_of::<f32>();
        let b_bytes = b_count * std::mem::size_of::<f32>();

        if offset + a_bytes + b_bytes > payload.len() {
            return Err("Truncated adapter data".into());
        }

        let a: Vec<f32> = payload[offset..offset + a_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
            .collect();
        offset += a_bytes;

        let b: Vec<f32> = payload[offset..offset + b_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
            .collect();

        Ok(Self {
            a,
            b,
            rank,
            alpha,
            in_dim,
            out_dim,
        })
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

                let a: Vec<f32> = file_data[offset..offset + a_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect();
                offset += a_bytes;

                // B matrix: [out_dim × rank]
                let b_rows = read_u16_le(&file_data, &mut offset)? as usize;
                let b_cols = read_u16_le(&file_data, &mut offset)? as usize;
                let b_count = b_rows * b_cols;
                let b_bytes = b_count * std::mem::size_of::<f32>();

                if offset + b_bytes > data_len {
                    return Err("Truncated B matrix data".into());
                }

                let b: Vec<f32> = file_data[offset..offset + b_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect();
                offset += b_bytes;

                let in_dim = a_cols;
                let out_dim = b_rows;

                adapters.push(Self {
                    a,
                    b,
                    rank,
                    alpha,
                    in_dim,
                    out_dim,
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

        // Embedding data
        let embed_bytes = kv_dim * std::mem::size_of::<f32>();
        if offset + embed_bytes > payload_end {
            return Err(format!(
                "Truncated embedding data: expected {embed_bytes} bytes at offset {offset}, payload ends at {payload_end}"
            ));
        }

        let embedding: Vec<f32> = data[offset..offset + embed_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
            .collect();

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
        for &val in &self.embedding {
            buf.extend_from_slice(&val.to_le_bytes());
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    /// Domain that handled this inference.
    pub domain: String,
    /// Best-path reward (max relevance score from WasmPruner).
    pub reward: f32,
    /// Number of nodes explored in DDTree.
    pub tree_budget_used: usize,
    /// Inference budget level (0=cheap, 1=moderate, 2=expensive).
    pub budget_level: u8,
    /// Input prompt hash (for dedup, not stored).
    pub prompt_hash: u64,
    /// Generated output text.
    pub output: String,
    /// Timestamp (Uuid v7 prefix).
    pub timestamp: i64,
    /// Was this result screened out (reward below threshold)?
    pub screened: bool,
    /// SR²AM configurator planning decision for this turn (Plan 112).
    #[cfg(feature = "sr2am_configurator")]
    pub planning_decision: Option<PlanningDecision>,
    /// Actual planning horizon used this turn (after entropy truncation, Plan 112 T13).
    #[cfg(feature = "sr2am_configurator")]
    pub plan_horizon_used: usize,
}

// ---------------------------------------------------------------------------
// Data Gate — Self-Play Stability (Plan 111, Research 075)
// ---------------------------------------------------------------------------

/// Discriminator for different self-play task types.
#[cfg(feature = "data_gate")]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            let abs_sum: f32 = row.iter().map(|v| v.abs()).sum();
            let scale = abs_sum / cols as f32;
            tw.row_scale[r] = if scale > 0.0 { scale } else { 1.0 };

            let threshold = 0.5 * tw.row_scale[r];
            let mut carry = 0.0f32;

            for (c, &val) in row.iter().enumerate() {
                let adjusted = val + carry;
                let q = if adjusted > threshold {
                    1i8
                } else if adjusted < -threshold {
                    -1i8
                } else {
                    0i8
                };
                tw.set(r, c, q);
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
            let mut row_sum = 0.0f32;
            let row_base = r * self.blocks64;
            for b in 0..self.blocks64 {
                let idx = row_base + b;
                let pos = self.pos_bits[idx];
                let neg = self.neg_bits[idx];
                // Count bits in this block (up to 64 elements)
                let pos_count = pos.count_ones() as f32;
                let neg_count = neg.count_ones() as f32;
                row_sum += pos_count - neg_count;
            }
            total += self.row_scale[r] * row_sum;
        }
        total
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
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, config.tree_budget);
        assert_eq!(result.temperature, config.temperature);
        assert_eq!(result.draft_lookahead, config.draft_lookahead);
    }

    #[test]
    fn test_with_overrides_some_applied() {
        let config = Config::draft();
        let overrides = InferenceOverrides {
            tree_budget: Some(99),
            temperature: Some(0.123),
            ..Default::default()
        };
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, 99);
        assert!((result.temperature - 0.123).abs() < 1e-6);
        // Non-overridden fields stay the same
        assert_eq!(result.draft_lookahead, config.draft_lookahead);
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
        let _ = std::fs::remove_file(&tmp);
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
        let _ = std::fs::remove_file(&tmp);
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
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_file_too_small() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_too_small.bin");
        std::fs::write(&tmp, b"DLAT").unwrap();
        assert!(DomainLatent::load(&tmp).is_err());
        let _ = std::fs::remove_file(&tmp);
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
}
