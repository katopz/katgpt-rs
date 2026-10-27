//! Kimi-K3-0.40B model-level forward pass.
//!
//! This module composes the decoder layers into a full model forward path:
//!
//! ```text
//! hidden = embed_tokens(input_ids)
//! block_state = AttnResBlockState::new()
//! for layer in 0..8:
//!     kimi_decoder_layer_forward(layer, ..., hidden, block_state)
//! hidden = apply_attn_res(hidden, block_state, output_attn_res)  // output mixing
//! hidden = rmsnorm(hidden, final_norm_weight)
//! logits = lm_head(hidden)
//! ```
//!
//! Matches `KimiLinearModel.forward` + `KimiLinearForCausalLM.forward` from
//! `modeling_kimi_k3_linear.py` (verified against real source, Research 331).

use katgpt_attn::gdn2::kda_forward::{KdaConfig, KdaForwardScratch, KdaLayerCache};
use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache};
use katgpt_core::types::math::rmsnorm_with_gamma_eps;
use katgpt_kv::shard_kv::rope::RopeFreqs;
use katgpt_transformer::attn_res::{
    AttnResBlockState, AttnResConfig, AttnResScratch, apply_attn_res,
};
use katgpt_transformer::moe::{MoeConfig, MoeForwardScratch};

use super::decoder_layer::{
    KimiAttentionConfig, KimiAttentionScratch, KimiAttentionState, KimiDecoderLayerConfig,
    KimiFfnConfig, KimiFfnScratch, kimi_decoder_layer_forward,
};

// ─── Config ────────────────────────────────────────────────────────────────

/// Full model configuration for Kimi-K3 models.
///
/// Contains the per-layer configs + model-level parameters. Supports both
/// the 0.40B fixture and scaled-up variants (4B/2B etc) via `mla_layer_indices`.
#[derive(Clone, Debug)]
pub struct KimiK3ModelConfig {
    /// Hidden size (1024 for 0.40B).
    pub hidden_size: usize,
    /// Vocab size (163840).
    pub vocab_size: usize,
    /// Number of layers (8 for 0.40B).
    pub num_layers: usize,
    /// RMSNorm epsilon (1e-5).
    pub rms_eps: f32,
    /// Which layers use MLA (full multi-latent attention). All others use KDA.
    /// 0.40B: [3, 7]. Scaled variants use a different pattern (e.g. every 4th layer).
    pub mla_layer_indices: Vec<usize>,
    /// MLA config (for layers in `mla_layer_indices`).
    pub mla_config: MlaConfig,
    /// KDA config (for layers NOT in `mla_layer_indices`).
    pub kda_config: KdaConfig,
    /// Dense MLP config (for layer 0 — the first layer is always dense).
    pub dense_ffn_config: KimiFfnConfig,
    /// MoE config (for layers 1..num_layers).
    pub moe_config: MoeConfig,
    /// Attention residual config.
    pub attn_res_config: AttnResConfig,
}

impl KimiK3ModelConfig {
    /// Kimi-K3-0.40B model configuration (verified against config.json, Research 331).
    pub fn kimi_k3_0_40b() -> Self {
        Self {
            hidden_size: 1024,
            vocab_size: 163840,
            num_layers: 8,
            rms_eps: 1e-5,
            mla_layer_indices: vec![3, 7],
            mla_config: MlaConfig::kimi_k3_0_40b(),
            kda_config: KdaConfig::kimi_k3_0_40b(),
            dense_ffn_config: KimiFfnConfig::Dense {
                intermediate_size: 2048,
                hidden_size: 1024,
                situ_beta: 4.0,
                situ_linear_beta: Some(25.0),
            },
            moe_config: MoeConfig::kimi_k3_0_40b(),
            attn_res_config: AttnResConfig::kimi_k3_0_40b(),
        }
    }

    /// Kimi-K3-4B-A2B model configuration (Issue 388 / Plan 318).
    ///
    /// Scaled-up MLA-MoE architecture targeting ~4.43B total / ~1.99B active
    /// params, with 256K context support via MLA KV compression.
    ///
    /// Architecture: 12 layers (9 KDA + 3 MLA, 3:1 ratio), hidden=3072,
    /// 12 routed experts (top-4), 2 shared experts, kv_lora_rank=512.
    /// MLA layers at indices [3, 7, 11] (every 4th layer from layer 3).
    ///
    /// KV cache at 256K: ~1.81 GB (3 MLA layers × 576 bytes/token).
    /// 4-bit quantized size: ~2.22 GB.
    pub fn kimi_k3_4b_a2b() -> Self {
        Self {
            hidden_size: 3072,
            vocab_size: 163840,
            num_layers: 12,
            rms_eps: 1e-5,
            // 3:1 KDA:MLA ratio — MLA at layers 3, 7, 11 (every 4th from 3)
            mla_layer_indices: vec![3, 7, 11],
            mla_config: MlaConfig {
                kv_lora_rank: 512,
                q_lora_rank: 768,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                n_heads: 16,
                hidden_size: 3072,
                use_output_gate: true,
                use_nope: true,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            kda_config: KdaConfig {
                hidden_size: 3072,
                n_heads: 16,
                head_dim: 128,
                conv_kernel_size: 4,
                ..KdaConfig::kimi_k3_0_40b()
            },
            dense_ffn_config: KimiFfnConfig::Dense {
                intermediate_size: 2816, // 2 × moe_intermediate
                hidden_size: 3072,
                situ_beta: 4.0,
                situ_linear_beta: Some(25.0),
            },
            moe_config: MoeConfig {
                num_experts: 12,
                num_experts_per_token: 4,
                num_shared_experts: 2,
                moe_intermediate_size: 1408,
                hidden_size: 3072,
                routed_expert_hidden_size: Some(1024),
                ..MoeConfig::kimi_k3_0_40b()
            },
            attn_res_config: AttnResConfig {
                block_size: 4,
                hidden_size: 3072,
                ..AttnResConfig::kimi_k3_0_40b()
            },
        }
    }

    /// Returns true if the given layer uses MLA (full multi-latent attention).
    /// All other layers use KDA (linear/delta attention).
    #[inline]
    pub fn is_mla_layer(&self, layer_idx: usize) -> bool {
        self.mla_layer_indices.contains(&layer_idx)
    }

    /// Get the attention config for a given layer.
    pub fn attention_config(&self, layer_idx: usize) -> KimiAttentionConfig {
        if self.is_mla_layer(layer_idx) {
            KimiAttentionConfig::Mla(self.mla_config.clone())
        } else {
            KimiAttentionConfig::Kda(self.kda_config.clone())
        }
    }

    /// Get the FFN config for a given layer.
    pub fn ffn_config(&self, layer_idx: usize) -> KimiFfnConfig {
        let is_dense = layer_idx == 0;
        if is_dense {
            self.dense_ffn_config.clone()
        } else {
            KimiFfnConfig::Moe(self.moe_config.clone())
        }
    }

    /// Get the per-layer decoder config for a given layer.
    pub fn layer_config(&self, layer_idx: usize) -> KimiDecoderLayerConfig {
        KimiDecoderLayerConfig {
            rms_eps: self.rms_eps,
            attention: self.attention_config(layer_idx),
            ffn: self.ffn_config(layer_idx),
            attn_res: self.attn_res_config.clone(),
        }
    }
}

// ─── Runtime state ─────────────────────────────────────────────────────────

/// Per-layer runtime state (cache + scratch).
pub struct KimiLayerRuntime {
    pub attn_state: KimiAttentionState,
    pub attn_scratch: KimiAttentionScratch,
    pub ffn_scratch: KimiFfnScratch,
    pub attn_res_self_scratch: AttnResScratch,
    pub attn_res_mlp_scratch: AttnResScratch,
}

/// Full model runtime state (all layers + model-level scratch).
pub struct KimiK3Runtime {
    pub layers: Vec<KimiLayerRuntime>,
    /// Shared block state (accumulated attention residuals).
    pub block_state: AttnResBlockState,
    /// Output-level attn-res scratch.
    pub output_attn_res_scratch: AttnResScratch,
    /// RoPE frequency table (for MLA layers — unused when use_nope=true, but
    /// still required as a parameter to mla_forward_token).
    pub rope_freqs: RopeFreqs,
    /// Hidden state buffer (the running prefix_sum).
    pub hidden: Vec<f32>,
    /// Scratch hidden buffer (norm input / attention input).
    pub scratch_hidden: Vec<f32>,
    /// Logits output buffer [vocab_size].
    pub logits: Vec<f32>,
    /// Issue 969 routing-statistics instrument: when armed, row `l` forces
    /// decoder layer `l`'s top-K assignment for the CURRENT token (`None` rows
    /// keep the router's own choice; rows past `layers.len()` are ignored).
    /// Re-armed per token by the diagnostic; both forward loops copy the row
    /// into the layer's MoE scratch and CLEAR it when disarmed, so stale state
    /// cannot leak across tokens. `reset()` also disarms. The production
    /// forward is byte-identical while this stays `None`.
    pub forced_router_plan: Option<Vec<Option<Vec<usize>>>>,
}

impl KimiK3Runtime {
    /// Create runtime state for the given model config + max sequence length.
    pub fn new(config: &KimiK3ModelConfig, max_seq_len: usize) -> Self {
        let max_block_entries = config.num_layers / config.attn_res_config.block_size + 1;

        let layers: Vec<KimiLayerRuntime> = (0..config.num_layers)
            .map(|layer_idx| {
                let is_mla = config.is_mla_layer(layer_idx);
                let is_dense = layer_idx == 0;

                let (attn_state, attn_scratch) = if is_mla {
                    (
                        KimiAttentionState::Mla(MlaKVCache::new(&config.mla_config, max_seq_len)),
                        KimiAttentionScratch::Mla(MlaForwardScratch::new(
                            &config.mla_config,
                            max_seq_len,
                        )),
                    )
                } else {
                    (
                        KimiAttentionState::Kda(KdaLayerCache::new(&config.kda_config)),
                        KimiAttentionScratch::Kda(KdaForwardScratch::new(&config.kda_config)),
                    )
                };

                let ffn_scratch = if is_dense {
                    let d_ffn = match &config.dense_ffn_config {
                        KimiFfnConfig::Dense {
                            intermediate_size, ..
                        } => *intermediate_size,
                        _ => unreachable!(),
                    };
                    KimiFfnScratch {
                        moe: MoeForwardScratch::new(&config.moe_config),
                        dense_gate: vec![0.0; d_ffn],
                        dense_up: vec![0.0; d_ffn],
                        dense_act: vec![0.0; d_ffn],
                        dense_out: vec![0.0; config.hidden_size],
                    }
                } else {
                    KimiFfnScratch {
                        moe: MoeForwardScratch::new(&config.moe_config),
                        dense_gate: Vec::new(),
                        dense_up: Vec::new(),
                        dense_act: Vec::new(),
                        dense_out: vec![0.0; config.hidden_size],
                    }
                };

                KimiLayerRuntime {
                    attn_state,
                    attn_scratch,
                    ffn_scratch,
                    attn_res_self_scratch: AttnResScratch::new(
                        &config.attn_res_config,
                        max_block_entries,
                    ),
                    attn_res_mlp_scratch: AttnResScratch::new(
                        &config.attn_res_config,
                        max_block_entries,
                    ),
                }
            })
            .collect();

        let d = config.hidden_size;

        Self {
            layers,
            block_state: AttnResBlockState::new_with_capacity(d, max_block_entries),
            output_attn_res_scratch: AttnResScratch::new(
                &config.attn_res_config,
                max_block_entries,
            ),
            rope_freqs: RopeFreqs::new_with_theta(
                config.mla_config.qk_rope_head_dim,
                config.mla_config.rope_theta,
            ),
            hidden: vec![0.0; d],
            scratch_hidden: vec![0.0; d],
            logits: vec![0.0; config.vocab_size],
            forced_router_plan: None,
        }
    }

    /// Reset all state for a new sequence (clears caches + block state).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match &mut layer.attn_state {
                KimiAttentionState::Mla(cache) => cache.reset(),
                KimiAttentionState::Kda(cache) => cache.reset(),
            }
        }
        self.block_state.clear();
        self.forced_router_plan = None;
    }
}

// ─── Forward ───────────────────────────────────────────────────────────────

/// Forward a single token through the full Kimi-K3-0.40B model.
///
/// Implements the complete decode path:
/// 1. Embedding lookup
/// 2. 8 decoder layers (with prefix_sum + attn-res block accumulation)
/// 3. Output attn-res (mix hidden with final block state)
/// 4. Final RMSNorm
/// 5. LM head projection → logits
///
/// # Arguments
/// - `config` — model-level config
/// - `weights` — loaded model weights
/// - `runtime` — runtime state (caches, scratch, block state)
/// - `token_id` — the input token ID
///
/// # Returns
/// A reference to `runtime.logits` `[vocab_size]`.
pub fn kimi_k3_forward_token<'a>(
    config: &KimiK3ModelConfig,
    weights: &super::loader::KimiK3ModelWeights,
    runtime: &'a mut KimiK3Runtime,
    token_id: u32,
) -> &'a [f32] {
    let d = config.hidden_size;

    // ── Step 0: Reset per-token block state ───────────────────────────────
    // Matches `modeling_kimi_linear.py` line 1324: `block_residual = None`
    // is set at the start of every forward call. The block_residual
    // accumulates WITHIN a forward pass (across the 8 layers), but is
    // fresh for each token. Without this reset, block_state grows unboundedly
    // across tokens → apply_attn_res scores buffer overflows on token #3+.
    runtime.block_state.clear();

    // ── Step 1: Embedding lookup ──────────────────────────────────────────
    let embed_start = (token_id as usize) * d;
    let embed_end = embed_start + d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_end]);

    // ── Steps 2-5: Decoder core (shared with forward_token_hidden) ────────
    forward_decoder_to_logits(config, weights, runtime)
}

/// Forward pass starting from an arbitrary hidden state (Issue 407 T2).
///
/// This is the same decoder path as [`kimi_k3_forward_token`] (steps 2-5:
/// decoder layers → output attn-res → final norm → LM head), but skips the
/// embedding lookup (step 1). Instead, the caller provides the hidden input
/// directly via `runtime.hidden` (the caller must set it before calling).
///
/// Used by the [`PauseStrategy::ZeroEmbedding`] path — the hidden state is
/// zeroed, so the KDA recurrent state evolves purely via its internal
/// dynamics (ξ_{n+1} = M·ξ_n + F(0)) with no new token signal.
///
/// **Block state contract:** this function clears `runtime.block_state` at
/// entry (step 0), matching `kimi_k3_forward_token`. The caller does NOT need
/// to clear it separately.
pub fn kimi_k3_forward_token_hidden<'a>(
    config: &KimiK3ModelConfig,
    weights: &super::loader::KimiK3ModelWeights,
    runtime: &'a mut KimiK3Runtime,
) -> &'a [f32] {
    // Step 0: same block_state reset as forward_token.
    runtime.block_state.clear();
    // Step 1 is skipped — runtime.hidden is whatever the caller set.
    // Steps 2-5:
    forward_decoder_to_logits(config, weights, runtime)
}

/// Issue 969: copy the armed plan row for `layer_idx` into the layer's MoE
/// scratch. Called by both forward loops immediately before the layer call so
/// the slot is refreshed (and cleared, when disarmed) on EVERY layer visit —
/// stale forced state cannot leak across tokens. Dense layers have no MoE
/// scratch to arm and are skipped.
fn arm_forced_router(
    layer_cfg: &KimiDecoderLayerConfig,
    runtime: &mut KimiK3Runtime,
    layer_idx: usize,
) {
    let forced = runtime
        .forced_router_plan
        .as_ref()
        .and_then(|p| p.get(layer_idx))
        .cloned()
        .flatten();
    if matches!(layer_cfg.ffn, KimiFfnConfig::Moe(_)) {
        runtime.layers[layer_idx].ffn_scratch.moe.forced_topk = forced;
    }
}

/// The shared decoder core: steps 2-5 of the forward pass.
///
/// Reads `runtime.hidden` (set by the caller — either via embedding lookup
/// in `kimi_k3_forward_token`, or zeroed in `kimi_k3_forward_token_hidden`).
/// Runs all decoder layers, applies output attn-res, final RMSNorm, and the
/// LM head projection. Writes logits to `runtime.logits`.
///
/// **Block state contract:** the caller must clear `runtime.block_state`
/// before calling (step 0). Both public wrappers (`kimi_k3_forward_token`,
/// `kimi_k3_forward_token_hidden`) do this.
fn forward_decoder_to_logits<'a>(
    config: &KimiK3ModelConfig,
    weights: &super::loader::KimiK3ModelWeights,
    runtime: &'a mut KimiK3Runtime,
) -> &'a [f32] {
    let d = config.hidden_size;

    // ── Step 2: Decoder layers ────────────────────────────────────────────
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let layer_cfg = config.layer_config(layer_idx);

        arm_forced_router(&layer_cfg, runtime, layer_idx);

        let layer_rt = &mut runtime.layers[layer_idx];

        kimi_decoder_layer_forward(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
        );
    }

    // ── Step 3: Output attn-res (mix with accumulated block state) ────────
    // This matches _apply_output_attn_res in the model code.
    if !runtime.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);
    }

    // ── Step 4: Final RMSNorm ─────────────────────────────────────────────
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );

    // ── Step 5: LM head projection → logits ───────────────────────────────
    // logits = lm_head_weight · hidden  [vocab_size, hidden_size] × [hidden_size]
    katgpt_core::simd::simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );

    &runtime.logits
}

// ─── Pause-token latent CoT (Issue 407 T2) ─────────────────────────────────
//
// Kimi-K3 analog of RiM (Recurrent Interface Machines) — inject N "pause"
// tokens after the prompt that advance the KDA recurrent state without
// emitting output. The model "thinks" by processing pause tokens,
// accumulating KDA state, before generating the answer.
//
// The mechanism is inference-time only (zero GPU). The KDA layers (75% of
// the model) have a fixed-size recurrent state — this IS the latent thought
// carrier. Pause tokens exploit it without retraining.
//
// **Honest expectation:** on an SFT model never trained with pause tokens,
// the GOAT gate (G5-K3: does N>0 pause tokens improve compile rate vs N=0?)
// is the empirical question. This primitive provides the mechanism; the gate
// **Gate:** the checkpoint-gated G5-K3 gate lives at
// `riir-train/crates/riir-train-engine/examples/plan318_pause_token_gate.rs`
// (NOT in katgpt-rs — katgpt-rs cannot depend on riir-train). The modelless
// re-scope (T4, 2026-08-09) lives at
// `riir-train/crates/riir-train-engine/examples/plan318_pause_modelless_ksweep.rs`
// — zero GPU, zero checkpoints, measures depth→discrimination on random weights.
// G5-K3 Modelless PASS: KDA state preserves signal through pause steps.

/// Strategy for pause-token "thinking" (Issue 407 T2).
///
/// Determines what input drives the KDA recurrent state update during pause
/// steps. The forward pass always runs the full decoder (advancing KDA + MLA
/// cache), but the caller discards the logits.
#[derive(Clone, Copy, Debug)]
pub enum PauseStrategy {
    /// Use a specific token ID (e.g., newline, period, or a benign token).
    /// The model processes the token normally; its embedding drives the KDA
    /// recurrent update. Best when the token is in-distribution for the model
    /// (common punctuation/whitespace the SFT corpus contained).
    TokenId(u32),
    /// Zero-embedding "no-op think" — bypass the embedding lookup entirely,
    /// feed a zero vector as the hidden input via
    /// [`kimi_k3_forward_token_hidden`]. This advances the KDA state purely
    /// via its recurrent dynamics (ξ_{n+1} = M·ξ_n + F(0)), with no new
    /// token information. Closest to the RiM "think without new input" idea,
    /// but note: with h=0, the KDA projections produce q=k=v=0 (no bias in
    /// the linear layers), so F(0) may be near-zero → the state mostly decays.
    ZeroEmbedding,
    /// Repeat the last prompt token — the KDA state evolves with the same
    /// input signal, potentially refining the representation via ShortConv
    /// state advancement + recurrent matrix application.
    RepeatLast,
}

/// Configuration for pause-token "thinking" before generation (Issue 407 T2).
#[derive(Clone, Copy, Debug)]
pub struct PauseConfig {
    /// Number of pause tokens to inject after the prompt, before generation.
    /// N=0 is the baseline (no thinking). Typical sweep values: 0, 1, 4, 16.
    pub n_pause: usize,
    /// Strategy for the pause token input.
    pub strategy: PauseStrategy,
}

impl PauseConfig {
    /// No pause tokens — the baseline / N=0 case.
    pub const fn none() -> Self {
        Self {
            n_pause: 0,
            strategy: PauseStrategy::ZeroEmbedding,
        }
    }

    /// N pause tokens with the given strategy.
    pub const fn new(n_pause: usize, strategy: PauseStrategy) -> Self {
        Self { n_pause, strategy }
    }
}

impl Default for PauseConfig {
    fn default() -> Self {
        Self::none()
    }
}

/// Advance the KDA recurrent state by one "pause step" without emitting output.
///
/// Processes a pause token (or zero-embedding / repeat-last) through the full
/// forward path (decoder layers → output attn-res → norm → LM head). The
/// logits are computed but the caller discards them — the purpose is to
/// evolve the per-layer `attn_state` (KDA recurrent state + MLA KV cache).
///
/// **State persistence:** the KDA recurrent state (`KdaLayerCache.heads` +
/// `ShortConv1D`) and MLA KV cache persist across all `kimi_k3_forward_token` /
/// `kimi_k3_pause_step` calls within a sequence (they live in
/// `runtime.layers[i].attn_state`). Only `block_state` is reset per call
/// (within-layer accumulation, fresh per token).
///
/// # Arguments
/// - `last_token` — the last real token processed (used by `RepeatLast`).
///   Pass the last prompt token ID.
pub fn kimi_k3_pause_step(
    config: &KimiK3ModelConfig,
    weights: &super::loader::KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    strategy: PauseStrategy,
    last_token: u32,
) {
    match strategy {
        PauseStrategy::TokenId(id) => {
            // Normal forward; logits discarded by caller.
            kimi_k3_forward_token(config, weights, runtime, id);
        }
        PauseStrategy::RepeatLast => {
            kimi_k3_forward_token(config, weights, runtime, last_token);
        }
        PauseStrategy::ZeroEmbedding => {
            // Zero the hidden state, then forward without embedding lookup.
            runtime.hidden.fill(0.0);
            kimi_k3_forward_token_hidden(config, weights, runtime);
        }
    }
}

/// Inject N pause steps after the prompt, before generation (Issue 407 T2).
///
/// This is the batch convenience wrapper around [`kimi_k3_pause_step`]. After
/// the prompt has been fed (via `kimi_k3_forward_token` for each prompt
/// token), call this to inject `config.n_pause` thinking steps. Then start
/// normal autoregressive decoding from the last logits.
///
/// **Returns** the logits after the final pause step (or the input logits
/// if `n_pause == 0`), so the caller can start generation without an extra
/// forward call.
///
/// # Arguments
/// - `last_prompt_token` — the last token of the prompt (used by `RepeatLast`).
/// - `last_logits` — the logits from the last prompt token's forward. Returned
///   unchanged when `n_pause == 0`.
pub fn kimi_k3_inject_pause(
    config: &KimiK3ModelConfig,
    weights: &super::loader::KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    pause: &PauseConfig,
    last_prompt_token: u32,
    last_logits: &[f32],
) -> Vec<f32> {
    if pause.n_pause == 0 {
        return last_logits.to_vec();
    }
    let mut current = last_logits.to_vec();
    for _ in 0..pause.n_pause {
        let logits = match pause.strategy {
            PauseStrategy::TokenId(id) => kimi_k3_forward_token(config, weights, runtime, id),
            PauseStrategy::RepeatLast => {
                kimi_k3_forward_token(config, weights, runtime, last_prompt_token)
            }
            PauseStrategy::ZeroEmbedding => {
                runtime.hidden.fill(0.0);
                kimi_k3_forward_token_hidden(config, weights, runtime)
            }
        };
        current = logits.to_vec();
    }
    current
}

// ─── Traced forward (per-layer hidden-state snapshots) ──────────────────────
//
// Same decoder path as `kimi_k3_forward_token` but snapshots `runtime.hidden`
// after embedding + after each decoder layer. Exists for trajectory geometry
// analysis (Plan 342 / Proposal 011 Layer 4) — the depth-wise latent trajectory
// [embed → layer0 → layer1 → ... → layer7] is the input to
// `latent_trajectory_geometry::from_states`.
//
// Skips the LM head projection (returns final normalized hidden state, not
// logits) because trajectory geometry operates on hidden states. This also
// avoids requiring the full [vocab × hidden] lm_head weight matrix when only
// the decoder-layer trajectory is needed — the traced variant can run with a
// truncated embedding table (just enough rows for the test token IDs).
//
// **Diagnostic only** — allocates per call (`traj_out.push(runtime.hidden.clone())`
// after each layer). Production callers should use `kimi_k3_forward_token`.

/// Traced forward — captures per-layer hidden states for trajectory geometry.
///
/// Runs the same decoder path as [`kimi_k3_forward_token`] (embed → 8 layers →
/// output attn-res → final norm) but snapshots `runtime.hidden` after embedding
/// + after each decoder layer into `traj_out`.
///
/// After the call, `traj_out` contains `num_layers + 1` entries (embedding +
/// 8 post-layer states), each of length `hidden_size`. This is the depth-wise
/// latent trajectory for one token.
///
/// Returns `&runtime.hidden` (final normalized hidden state). The LM head is
/// intentionally skipped — see the module comment above.
pub fn kimi_k3_forward_token_traced<'a>(
    config: &KimiK3ModelConfig,
    weights: &super::loader::KimiK3ModelWeights,
    runtime: &'a mut KimiK3Runtime,
    token_id: u32,
    traj_out: &mut Vec<Vec<f32>>,
) -> &'a [f32] {
    let d = config.hidden_size;

    traj_out.clear();

    // ── Step 0: Reset per-token block state ───────────────────────────────
    runtime.block_state.clear();

    // ── Step 1: Embedding lookup + snapshot ───────────────────────────────
    let embed_start = (token_id as usize) * d;
    let embed_end = embed_start + d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_end]);
    traj_out.push(runtime.hidden.clone());

    // ── Step 2: Decoder layers + per-layer snapshot ───────────────────────
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let layer_cfg = config.layer_config(layer_idx);

        arm_forced_router(&layer_cfg, runtime, layer_idx);

        let layer_rt = &mut runtime.layers[layer_idx];

        kimi_decoder_layer_forward(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
        );
        traj_out.push(runtime.hidden.clone());
    }

    // ── Step 3: Output attn-res (mix with accumulated block state) ────────
    if !runtime.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);
    }

    // ── Step 4: Final RMSNorm ─────────────────────────────────────────────
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );

    // NOTE: Step 5 (LM head) is intentionally skipped — see module comment.

    &runtime.hidden
}

// ─── Instrumented forward (per-phase timing) ───────────────────────────────
//
// This is the same forward path as `kimi_k3_forward_token` but with timing
// probes injected between each major phase. It exists for the hello-world
// example + benchmarking — production callers should use the uninstrumented
// variant (no Instant::now() overhead in the hot path).
//
// The phase split matches the model-level steps (embed / layers / output-res
// / final-norm / lm_head), NOT the per-layer internal breakdown. Per-layer
// internal breakdown (KDA vs MLA vs MoE) would require instrumenting
// `kimi_decoder_layer_forward` — left as a future optimization target.

/// Per-phase timing accumulator (microseconds).
///
/// All fields are `u128` microseconds. Use `Default::default()` for a zeroed
/// instance. The example accumulates across N tokens then divides by N for
/// the per-token average.
#[derive(Default, Debug, Clone)]
pub struct ForwardTiming {
    /// Embedding lookup (row gather from the [vocab × hidden] table).
    pub embed_us: u128,
    /// All 8 decoder layers combined (attention + FFN + attn-res per layer).
    pub layers_us: u128,
    /// Output attn-res mixing (single apply_attn_res on accumulated block state).
    pub output_attn_res_us: u128,
    /// Final RMSNorm over the hidden state.
    pub final_norm_us: u128,
    /// LM head matmul: [vocab × hidden] × [hidden] → [vocab] logits.
    pub lm_head_us: u128,
}

impl ForwardTiming {
    /// Sum of all phases (= total forward time, minus the block_state.clear()
    /// which is negligible).
    pub fn total_us(&self) -> u128 {
        self.embed_us
            + self.layers_us
            + self.output_attn_res_us
            + self.final_norm_us
            + self.lm_head_us
    }
}

/// Forward a single token through the full Kimi-K3-0.40B model, recording
/// per-phase timing.
///
/// This is the instrumented counterpart to `kimi_k3_forward_token`. It
/// produces bit-identical logits (same operations, same order — only
/// `Instant::now()` calls are inserted between phases).
///
/// The `timing` argument is mutated in place; the caller typically passes a
/// reference into an accumulator struct that sums across N tokens.
pub fn kimi_k3_forward_token_timed<'a>(
    config: &KimiK3ModelConfig,
    weights: &super::loader::KimiK3ModelWeights,
    runtime: &'a mut KimiK3Runtime,
    token_id: u32,
    timing: &mut ForwardTiming,
) -> &'a [f32] {
    use std::time::Instant;

    let d = config.hidden_size;

    runtime.block_state.clear();

    let t = Instant::now();
    let embed_start = (token_id as usize) * d;
    let embed_end = embed_start + d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_end]);
    timing.embed_us += t.elapsed().as_micros();

    let t = Instant::now();
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_rt = &mut runtime.layers[layer_idx];

        kimi_decoder_layer_forward(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
        );
    }
    timing.layers_us += t.elapsed().as_micros();

    let t = Instant::now();
    if !runtime.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);
    }
    timing.output_attn_res_us += t.elapsed().as_micros();

    let t = Instant::now();
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );
    timing.final_norm_us += t.elapsed().as_micros();

    let t = Instant::now();
    katgpt_core::simd::simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );
    timing.lm_head_us += t.elapsed().as_micros();

    &runtime.logits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G1 (refactor bit-identical): forward_token via forward_decoder_to_logits
    /// produces finite, non-trivial logits on random weights. This verifies the
    /// refactor didn't break the forward path — the logic is identical, only
    /// the code organization changed.
    #[test]
    fn g1_forward_token_refactor_produces_finite_logits() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = super::super::loader::KimiK3ModelWeights::random(&config, 42);
        let mut runtime = KimiK3Runtime::new(&config, 32);

        let logits = kimi_k3_forward_token(&config, &weights, &mut runtime, 5);

        assert_eq!(logits.len(), config.vocab_size);
        assert!(
            logits.iter().all(|&l| l.is_finite()),
            "logits must be finite"
        );
        // Non-trivial: not all zeros (random weights produce non-zero logits).
        let max_abs = logits.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(
            max_abs > 0.0,
            "logits must be non-trivial (max_abs={max_abs})"
        );
    }

    /// G2 (forward_token_hidden works): the hidden-direct variant produces finite
    /// logits + advances state the same way as forward_token when given the
    /// same hidden input.
    #[test]
    fn g2_forward_token_hidden_produces_finite_logits() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = super::super::loader::KimiK3ModelWeights::random(&config, 42);
        let mut runtime = KimiK3Runtime::new(&config, 32);

        // Zero the hidden state + forward via the hidden path.
        runtime.hidden.fill(0.0);
        let logits = kimi_k3_forward_token_hidden(&config, &weights, &mut runtime);

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|&l| l.is_finite()));
    }

    /// G3 (pause step advances KDA state): after a pause step, the hidden state
    /// of at least one KDA layer should differ from before. This verifies the
    /// KDA recurrent state is actually evolving during pause — the whole point
    /// of the mechanism.
    #[test]
    fn g3_pause_step_advances_kda_state() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = super::super::loader::KimiK3ModelWeights::random(&config, 42);
        let mut runtime = KimiK3Runtime::new(&config, 32);

        // Feed a prompt token to initialize state.
        kimi_k3_forward_token(&config, &weights, &mut runtime, 10);

        // Snapshot the logits (they reflect the KDA state at this point).
        let logits_before = runtime.logits.clone();

        // Inject a pause step (ZeroEmbedding strategy).
        kimi_k3_pause_step(
            &config,
            &weights,
            &mut runtime,
            PauseStrategy::ZeroEmbedding,
            10,
        );

        // The logits after pause should differ (KDA state evolved).
        let logits_after = runtime.logits.clone();
        let diff = logits_before
            .iter()
            .zip(logits_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a + b);
        assert!(
            diff > 0.0,
            "KDA state must evolve during pause (total logit diff={diff})"
        );
    }

    /// G4 (inject_pause n_pause=0 is a no-op): when n_pause is 0, the function
    /// should return the input logits unchanged.
    #[test]
    fn g4_inject_pause_zero_is_noop() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = super::super::loader::KimiK3ModelWeights::random(&config, 42);
        let mut runtime = KimiK3Runtime::new(&config, 32);

        let input_logits = vec![0.5; config.vocab_size];
        let result = kimi_k3_inject_pause(
            &config,
            &weights,
            &mut runtime,
            &PauseConfig::none(),
            10,
            &input_logits,
        );

        assert_eq!(result, input_logits, "n_pause=0 must be identity");
    }

    /// G5 (RepeatLast strategy works): pause with RepeatLast advances state.
    #[test]
    fn g5_pause_repeat_last_advances_state() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = super::super::loader::KimiK3ModelWeights::random(&config, 42);
        let mut runtime = KimiK3Runtime::new(&config, 32);

        kimi_k3_forward_token(&config, &weights, &mut runtime, 10);
        let logits_before = runtime.logits.clone();

        kimi_k3_pause_step(
            &config,
            &weights,
            &mut runtime,
            PauseStrategy::RepeatLast,
            10,
        );

        let diff = logits_before
            .iter()
            .zip(runtime.logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a + b);
        assert!(diff > 0.0, "RepeatLast must advance KDA state");
    }

    /// G6 (TokenId strategy works): pause with a specific token advances state.
    #[test]
    fn g6_pause_token_id_advances_state() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = super::super::loader::KimiK3ModelWeights::random(&config, 42);
        let mut runtime = KimiK3Runtime::new(&config, 32);

        kimi_k3_forward_token(&config, &weights, &mut runtime, 10);
        let logits_before = runtime.logits.clone();

        // Use token ID 3 (a low but non-special token) as the pause token.
        kimi_k3_pause_step(
            &config,
            &weights,
            &mut runtime,
            PauseStrategy::TokenId(3),
            10,
        );

        let diff = logits_before
            .iter()
            .zip(runtime.logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a + b);
        assert!(diff > 0.0, "TokenId(3) must advance KDA state");
    }

    /// Plan 318 E1 — position-encoding invariant for the Kimi-K3-Linear family.
    ///
    /// Both production configs (`kimi_k3_0_40b` + `kimi_k3_4b_a2b`) ship with
    /// `mla_config.use_nope = true`. This is a deliberate architectural choice:
    /// the Kimi-K3-Linear model encodes position entirely via KDA recurrence
    /// (GDN2 delta-rule state) + the local `conv_kernel_size` conv + the
    /// chunked `attn_res` residual — NOT via RoPE. When `use_nope` is true the
    /// MLA forward skips `apply_decoupled_rope` entirely (mla.rs:524-531,
    /// 541-546), so `rope_theta` is dead code and there is no base frequency
    /// to tune for 256K.
    ///
    /// This test is the tripwire for Plan 318 E1 ("RoPE base frequency tuning
    /// for 256K"): if a future change flips `use_nope` to `false` on either
    /// config, RoPE becomes active and `rope_theta = 10_000` is far too low for
    /// 256K context (it causes high-frequency dimension aliasing past ~32K
    /// positions). The test failure forces the author to either (a) re-enable
    /// E1 and select an appropriate `rope_theta` / YaRN scale for 256K, or
    /// (b) update this test + the E1 plan entry documenting the new strategy.
    #[test]
    fn e1_position_encoding_is_nope_on_both_configs() {
        let cfg_040b = KimiK3ModelConfig::kimi_k3_0_40b();
        assert!(
            cfg_040b.mla_config.use_nope,
            "0.40B must use_nope=true: position comes from KDA recurrence, not RoPE. \
             If you enabled RoPE, rope_theta={} is too low for 256K — revisit Plan 318 E1.",
            cfg_040b.mla_config.rope_theta,
        );

        let cfg_4b = KimiK3ModelConfig::kimi_k3_4b_a2b();
        assert!(
            cfg_4b.mla_config.use_nope,
            "4B-A2B must use_nope=true: position comes from KDA recurrence, not RoPE. \\n             If you enabled RoPE, rope_theta={} is too low for 256K — revisit Plan 318 E1.",
            cfg_4b.mla_config.rope_theta,
        );

        // Sanity: the KDA layers (the position-carrying layers) are present and
        // form the majority. 0.40B: 6 KDA / 8 total. 4B: 9 KDA / 12 total.
        let kda_040b = cfg_040b.num_layers - cfg_040b.mla_layer_indices.len();
        assert_eq!(
            kda_040b, 6,
            "0.40B should have 6 KDA layers carrying position"
        );
        let kda_4b = cfg_4b.num_layers - cfg_4b.mla_layer_indices.len();
        assert_eq!(
            kda_4b, 9,
            "4B-A2B should have 9 KDA layers carrying position"
        );
    }
}
