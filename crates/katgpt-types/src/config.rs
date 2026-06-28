//! Transformer Config + InferenceOverrides + kv_dim.

use super::*;

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
    // Any-Time LT2 Dispatch (Issue 035, Research 273 — ELT arXiv:2604.09168).
    // `loop_min` = floor for elastic override (refuse exit below this).
    // `loop_max` = trained max loop count; 0 = sentinel meaning "derive from
    //   loop_mode" (i.e. use WeightShared.loop_count).
    // Hard ceiling for elastic over-iteration is `2 × loop_max` (ELT §1.5:
    // modest over-looping beyond L_max is regularized by training; cap at 2×
    // to prevent runaway). Both default to 0 = "derive from loop_mode".
    pub loop_min: usize,
    pub loop_max: usize,
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
    /// Compute the effective loop count for `forward_looped`, applying an
    /// optional elastic override clamped to `[loop_min, 2×loop_max]`.
    ///
    /// (Issue 035, Research 273 — ELT arXiv:2604.09168 Any-Time inference.)
    ///
    /// - `elastic_override = None` → use `loop_mode`'s natural loop count
    ///   (byte-identical to pre-Issue-035 behavior).
    /// - `elastic_override = Some(L)` with `LoopMode::WeightShared` → clamp L
    ///   to `[max(loop_min, 1), 2 × max(loop_max, base)]`.
    ///   - Below `loop_min` (default 1): clamped up. ELT §1.4 establishes a
    ///     minimum depth for representational capacity (`1N × 32L` collapsed
    ///     to FID 10.30 vs 2.83 for `16N × 2L`).
    ///   - Above `2 × loop_max`: clamped down. ELT §1.5 shows modest
    ///     over-looping beyond L_max is regularized (UCF-101 peak FVD at L=6
    ///     with L_max=4), but quality eventually deteriorates — 2× is the cap.
    /// - `elastic_override = Some(_)` with `LoopMode::None` or `TrainingFree` →
    ///   refused (returns base); there is no weight-shared loop to elastically
    ///   exit from.
    ///
    /// `loop_max == 0` is a sentinel meaning "derive from `loop_mode`" (use
    /// `WeightShared.loop_count`). This keeps the 12 existing Config
    /// constructors unchanged in semantics — they default to `loop_max: 0`
    /// which resolves to the natural loop count.
    #[inline]
    pub fn effective_loop_count(&self, elastic_override: Option<usize>) -> usize {
        let base = match self.loop_mode {
            LoopMode::WeightShared { loop_count } => loop_count,
            LoopMode::None | LoopMode::TrainingFree => 1,
        };
        let requested = match elastic_override {
            None => return base,
            Some(o) => o,
        };
        // Refuse elastic override when there's no weight-shared loop to exit.
        if !matches!(self.loop_mode, LoopMode::WeightShared { .. }) {
            return base;
        }
        let lo = self.loop_min.max(1);
        let max_base = if self.loop_max == 0 { base } else { self.loop_max };
        let hi = max_base.max(base).max(lo);
        let hard_cap = 2 * hi;
        requested.clamp(lo, hard_cap)
    }

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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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

    /// Game config for FFT Tactics Arena LoRA training (Plan 296 T7.3).
    /// Tiny Transformer for battle state → action prediction.
    ///
    /// # Token Layout
    ///
    /// - State vocab (values 0..9): team(0-1), class(0-5), hp_bucket(0-7),
    ///   mp_bucket(0-3), pos_x(0-7), pos_y(0-7), alive(0-1).
    /// - Action tokens: 10..19 (9 FFT ActionTypes).
    ///
    /// Sequence (58 tokens):
    ///   `[tick, u0_team, u0_class, u0_hp, u0_mp, u0_x, u0_y, u0_alive,
    ///     u1_..., ..., u7_..., action_token]`
    ///
    /// Per-unit = 7 tokens × 8 units = 56, +1 tick +1 action = 58 tokens.
    /// ~18K params total, ~1.5K LoRA params (rank=4). Comparable to Bomber/Go.
    pub fn game_fft() -> Self {
        Self {
            vocab_size: 19,
            block_size: 58,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
            loop_min: 0,
            loop_max: 0,
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
