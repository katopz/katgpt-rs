//! Kimi-K3 decoder layer composition — wires attention + FFN + attn-res.
//!
//! This is the Phase 5 model-composition layer (Proposal 032 T5.6). It lives
//! in the katgpt-rs root crate because it needs BOTH:
//! - MLA/KDA from `katgpt-attn` (which cannot depend on katgpt-transformer)
//! - MoE/attn-res from `katgpt-transformer`
//!
//! The root crate depends on both leaf crates, so it's the natural composition
//! point (mirrors `tf_loop.rs`, `forward_hla`, etc.).
//!
//! # Layer topology (VERIFIED against safetensors header, Research 331)
//!
//! | Layer | Attention | FFN     | Notes                          |
//! |-------|-----------|---------|--------------------------------|
//! | 0     | KDA       | Dense   | `first_k_dense_replace: 1`     |
//! | 1-2   | KDA       | MoE     |                                |
//! | 3     | MLA       | MoE     | Every 4th layer is full attn   |
//! | 4-6   | KDA       | MoE     |                                |
//! | 7     | MLA       | MoE     | Every 4th layer is full attn   |
//!
//! Config `full_attn_layers: [4, 8]` is 1-indexed → MLA at 0-indexed 3, 7.
//! All 8 layers use the attention residual block (`attn_res_block_size: 4`).
//!
//! # Forward path (single-token decode)
//!
//! Matches the actual `KimiDecoderLayer.forward` from
//! `modeling_kimi_k3_linear.py` (Research 330 §5):
//!
//! ```text
//! hidden = apply_attn_res(hidden, block_state, self_attn_res_weights)
//! residual = hidden
//! hidden = input_layernorm(hidden)
//! hidden = attention(hidden)      // MLA or KDA
//! hidden = residual + hidden
//! hidden = apply_attn_res(hidden, block_state, mlp_res_weights)
//! residual = hidden
//! hidden = post_attention_layernorm(hidden)
//! hidden = ffn(hidden)            // Dense MLP or MoE
//! hidden = residual + hidden
//! if layer_idx is block boundary:
//!     block_state.push(hidden.clone())
//! ```

use katgpt_attn::gdn2::kda_forward::{
    KdaConfig, KdaForwardScratch, KdaLayerCache, KdaWeights, kda_forward_token,
};
use katgpt_attn::mla::{
    MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token_dispatched,
};
use katgpt_core::types::math::rmsnorm_with_gamma_eps;
use katgpt_kv::shard_kv::rope::RopeFreqs;
use katgpt_transformer::attn_res::{
    AttnResBlockState, AttnResConfig, AttnResScratch, AttnResWeights, apply_attn_res,
};
use katgpt_transformer::moe::{
    MoeConfig, MoeForwardScratch, MoeWeights, SwiGluExpertWeights, moe_forward_token,
};

// ─── Attention type enum ───────────────────────────────────────────────────

/// Per-layer attention weights — either MLA (full) or KDA (linear).
///
/// Layer topology from Research 330 §8: layers 0 + 4 are MLA; layers 1,2,3,5,6,7
/// are KDA.
#[derive(Clone)]
pub enum KimiAttentionWeights {
    /// Full attention (Multi-head Latent Attention). Layers 0, 4.
    Mla(MlaWeights),
    /// Linear attention (Kimi Delta Attention). Layers 1, 2, 3, 5, 6, 7.
    Kda(KdaWeights),
}

/// Per-layer attention config — mirrors [`KimiAttentionWeights`].
#[derive(Clone, Debug)]
pub enum KimiAttentionConfig {
    Mla(MlaConfig),
    Kda(KdaConfig),
}

// ─── FFN type enum ─────────────────────────────────────────────────────────

/// Per-layer FFN weights — either Dense MLP (layer 0) or MoE (layers 1-7).
///
/// Layer 0 uses a dense SiTU MLP with `intermediate_size = 2048` (the only
/// dense layer, per `first_k_dense_replace: 1`). All other layers use the
/// latent MoE with SiTU experts.
///
/// The dense MLP reuses `SwiGluExpertWeights` (gate + up + down) — the weight
/// layout is structurally identical; only the activation function differs
/// (SiTU instead of SwiGLU, both computed in the forward path).
#[derive(Clone)]
pub enum KimiFfnWeights {
    /// Dense SiTU MLP (layer 0 only). `intermediate_size = 2048`, `d_in = hidden_size`.
    Dense(SwiGluExpertWeights),
    /// Latent MoE with SiTU experts (layers 1-7).
    Moe(MoeWeights),
}

/// Per-layer FFN config — mirrors [`KimiFfnWeights`].
#[derive(Clone, Debug)]
pub enum KimiFfnConfig {
    /// Dense MLP config: `(intermediate_size, hidden_size, situ_beta, situ_linear_beta)`.
    Dense {
        intermediate_size: usize,
        hidden_size: usize,
        situ_beta: f32,
        situ_linear_beta: Option<f32>,
    },
    /// MoE config.
    Moe(MoeConfig),
}

// ─── Decoder layer weights ─────────────────────────────────────────────────

/// Full decoder layer weights for one Kimi-K3 layer.
///
/// Combines: input/post-attn RMSNorm + attention (MLA or KDA) + FFN (Dense or
/// MoE) + two attn-res weight sets (self-attn + MLP).
#[derive(Clone)]
pub struct KimiDecoderLayerWeights {
    /// RMSNorm gamma before attention (`input_layernorm`). Shape `[hidden_size]`.
    pub input_layernorm_weight: Vec<f32>,
    /// RMSNorm gamma before FFN (`post_attention_layernorm`). Shape `[hidden_size]`.
    pub post_attention_layernorm_weight: Vec<f32>,
    /// Attention weights (MLA or KDA).
    pub attention: KimiAttentionWeights,
    /// FFN weights (Dense MLP or MoE).
    pub ffn: KimiFfnWeights,
    /// Attn-res weights for the self-attention block.
    pub self_attn_res: AttnResWeights,
    /// Attn-res weights for the MLP block.
    pub mlp_attn_res: AttnResWeights,
}

/// Per-layer config — the non-weight parameters needed for forward.
#[derive(Clone, Debug)]
pub struct KimiDecoderLayerConfig {
    /// RMSNorm epsilon (1e-5 for Kimi-K3).
    pub rms_eps: f32,
    /// Attention config (MLA or KDA).
    pub attention: KimiAttentionConfig,
    /// FFN config (Dense or MoE).
    pub ffn: KimiFfnConfig,
    /// Attn-res config.
    pub attn_res: AttnResConfig,
}

// ─── Per-layer runtime state (caches + scratch) ────────────────────────────

/// Per-layer attention cache — either MLA KV cache or KDA recurrent state.
pub enum KimiAttentionState {
    Mla(MlaKVCache),
    Kda(KdaLayerCache),
}

/// Per-layer attention scratch — either MLA or KDA forward scratch.
#[allow(clippy::large_enum_variant)]
pub enum KimiAttentionScratch {
    Mla(MlaForwardScratch),
    Kda(KdaForwardScratch),
}

/// Per-layer FFN scratch — MoE scratch (Dense MLP uses a small inline scratch).
pub struct KimiFfnScratch {
    /// MoE forward scratch (used when FFN is MoE).
    pub moe: MoeForwardScratch,
    /// Dense MLP intermediate buffer `[intermediate_size]` (gate proj output).
    pub dense_gate: Vec<f32>,
    /// Dense MLP intermediate buffer `[intermediate_size]` (up proj output).
    pub dense_up: Vec<f32>,
    /// Dense MLP activation buffer `[intermediate_size]` (SiTU output).
    pub dense_act: Vec<f32>,
    /// Dense MLP output buffer `[hidden_size]`.
    pub dense_out: Vec<f32>,
}

// ─── Layer forward ─────────────────────────────────────────────────────────

/// Forward a single token through one Kimi-K3 decoder layer.
///
/// Implements the actual `_forward_attn_residual` path from
/// `modeling_kimi_k3_linear.py` (verified against the real source, Research 331).
///
/// The key insight is the **prefix_sum** pattern: within an attention-residual
/// block, a running `prefix_sum` accumulates residual contributions across
/// layers. At block boundaries (`layer_idx % block_size == 0`), the accumulated
/// `prefix_sum` is pushed to `block_state` and reset. Before each sub-layer
/// (attention + FFN), `prefix_sum` is mixed with past block residuals via
/// `apply_attn_res` to produce the input `hidden`.
///
/// ```text
/// prefix_sum = hidden  (input to this layer)
///
/// if block_state has entries:
///     hidden = apply_attn_res(prefix_sum, block_state, self_attn_res)
///
/// if layer_idx % block_size == 0:
///     block_state.push(prefix_sum)   (push BEFORE attention — the raw input)
///     prefix_sum = None              (reset for new block)
///
/// hidden = input_layernorm(hidden)
/// hidden = attention(hidden)
/// prefix_sum = (prefix_sum ?? 0) + hidden   (accumulate attention output)
///
/// hidden = apply_attn_res(prefix_sum, block_state, mlp_res)
/// hidden = post_attention_layernorm(hidden)
/// hidden = ffn(hidden)
/// prefix_sum += hidden               (accumulate FFN output)
/// ```
///
/// # Arguments
/// - `layer_idx` — 0-based layer index (for block-boundary check)
/// - `config` — per-layer config
/// - `weights` — per-layer weights
/// - `attn_state` — attention cache (MLA KV or KDA recurrent)
/// - `attn_scratch` — attention forward scratch
/// - `ffn_scratch` — FFN forward scratch
/// - `attn_res_self_scratch` — attn-res scratch for self-attn block
/// - `attn_res_mlp_scratch` — attn-res scratch for MLP block
/// - `block_state` — accumulated block residuals (shared across layers)
/// - `rope_freqs` — RoPE frequency table (MLA only; KDA doesn't use it)
/// - `prefix_sum` — the running residual sum `[hidden_size]` (input = output)
/// - `scratch_hidden` — scratch buffer `[hidden_size]` for norm/attention input
///
/// On return, `prefix_sum` holds the updated running sum (becomes the input
/// to the next layer). `block_state` may have a new entry if this was a boundary.
#[allow(clippy::too_many_arguments)]
pub fn kimi_decoder_layer_forward(
    layer_idx: usize,
    config: &KimiDecoderLayerConfig,
    weights: &KimiDecoderLayerWeights,
    attn_state: &mut KimiAttentionState,
    attn_scratch: &mut KimiAttentionScratch,
    ffn_scratch: &mut KimiFfnScratch,
    attn_res_self_scratch: &mut AttnResScratch,
    attn_res_mlp_scratch: &mut AttnResScratch,
    block_state: &mut AttnResBlockState,
    rope_freqs: Option<&mut RopeFreqs>,
    prefix_sum: &mut [f32],
    scratch_hidden: &mut [f32],
) {
    let d = config.attn_res.d();
    debug_assert_eq!(prefix_sum.len(), d);
    debug_assert_eq!(scratch_hidden.len(), d);

    let eps = config.rms_eps;
    let block_size = config.attn_res.block_size;
    let is_boundary = layer_idx.is_multiple_of(block_size);

    // ── Step 1: apply_attn_res (self-attention) — mix prefix_sum with blocks ─
    // If block_state has entries, mix. Otherwise hidden = prefix_sum (unchanged).
    // scratch_hidden receives the mixed hidden (the input to attention).
    if !block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res,
            &weights.self_attn_res,
            block_state,
            attn_res_self_scratch,
            prefix_sum,
        );
        scratch_hidden.copy_from_slice(mixed);
    } else {
        scratch_hidden.copy_from_slice(prefix_sum);
    }

    // ── Step 2: Block boundary — push prefix_sum BEFORE attention ──────────
    // At block boundaries, push the raw input (prefix_sum) to block_state,
    // then reset prefix_sum to zero (it will be rebuilt from attention + FFN).
    if is_boundary {
        block_state.push(prefix_sum);
        // Reset prefix_sum — new block starts fresh.
        // It will be set to attn_out below (step 4: prefix_sum was None → = hidden).
        // `fill` lowers to a memset instead of an element-at-a-time store loop.
        prefix_sum.fill(0.0);
    }

    // ── Step 3: input_layernorm → attention ───────────────────────────────
    // Normalize the mixed hidden, then run attention.
    rmsnorm_with_gamma_eps(scratch_hidden, &weights.input_layernorm_weight, eps as f64);

    // attention forward (MLA or KDA)
    let attn_out: &[f32] = match (&config.attention, &weights.attention) {
        (KimiAttentionConfig::Mla(mla_cfg), KimiAttentionWeights::Mla(mla_w)) => {
            let KimiAttentionState::Mla(cache) = attn_state else {
                panic!("MLA attention config but non-MLA cache state");
            };
            let KimiAttentionScratch::Mla(scratch) = attn_scratch else {
                panic!("MLA attention config but non-MLA scratch");
            };
            let Some(rf) = rope_freqs else {
                panic!("MLA attention requires rope_freqs");
            };
            // Issue 926: routes to the absorbed path under `mla_absorbed`
            // (kill-switch `KATGPT_MLA_RECONSTRUCT=1`); reconstruct otherwise.
            mla_forward_token_dispatched(mla_cfg, mla_w, cache, scratch, rf, scratch_hidden)
        }
        (KimiAttentionConfig::Kda(kda_cfg), KimiAttentionWeights::Kda(kda_w)) => {
            let KimiAttentionState::Kda(cache) = attn_state else {
                panic!("KDA attention config but non-KDA cache state");
            };
            let KimiAttentionScratch::Kda(scratch) = attn_scratch else {
                panic!("KDA attention config but non-KDA scratch");
            };
            kda_forward_token(kda_cfg, kda_w, cache, scratch, scratch_hidden)
        }
        _ => panic!("attention config/weights mismatch"),
    };

    // ── Step 4: Accumulate attention output into prefix_sum ───────────────
    // prefix_sum += attn_out (if boundary: prefix_sum was 0 → prefix_sum = attn_out)
    // Elementwise SIMD add — no reassociation (each lane is a single independent
    // `a + b`), so bit-identical to the scalar loop. The `[..d]` slices keep the
    // old panic-on-short-operand behaviour.
    katgpt_core::simd::simd_add_inplace(&mut prefix_sum[..d], &attn_out[..d]);

    // ── Step 5: apply_attn_res (MLP) — mix prefix_sum with blocks ─────────
    let mixed = apply_attn_res(
        &config.attn_res,
        &weights.mlp_attn_res,
        block_state,
        attn_res_mlp_scratch,
        prefix_sum,
    );
    scratch_hidden.copy_from_slice(mixed);

    // ── Step 6: post_attention_layernorm → FFN ────────────────────────────
    rmsnorm_with_gamma_eps(
        scratch_hidden,
        &weights.post_attention_layernorm_weight,
        eps as f64,
    );

    // FFN forward (Dense or MoE)
    let ffn_out: &[f32] = match (&config.ffn, &weights.ffn) {
        (
            KimiFfnConfig::Dense {
                situ_beta,
                situ_linear_beta,
                ..
            },
            KimiFfnWeights::Dense(expert),
        ) => dense_situ_ffn_forward(
            expert,
            scratch_hidden,
            ffn_scratch,
            *situ_beta,
            *situ_linear_beta,
        ),
        (KimiFfnConfig::Moe(moe_cfg), KimiFfnWeights::Moe(moe_w)) => {
            moe_forward_token(
                moe_w,
                moe_cfg,
                scratch_hidden,
                &mut ffn_scratch.dense_out,
                &mut ffn_scratch.moe,
            );
            &ffn_scratch.dense_out[..d]
        }
        _ => panic!("FFN config/weights mismatch"),
    };

    // ── Step 7: Accumulate FFN output into prefix_sum ─────────────────────
    // Elementwise SIMD add — see Step 4 for the bit-identity argument.
    katgpt_core::simd::simd_add_inplace(&mut prefix_sum[..d], &ffn_out[..d]);
}

/// Dense SiTU FFN forward (layer 0 only).
///
/// Computes: `down_proj(SiTU(gate_proj(h), up_proj(h)))`
///
/// The weight layout matches `SwiGluExpertWeights` but the activation is SiTU
/// (not SwiGLU). The forward path is:
/// 1. `gate = gate_proj · h`  `[d_ffn]`
/// 2. `up = up_proj · h`  `[d_ffn]`
/// 3. `act = SiTU(gate, up)`  (in-place on gate)
/// 4. `out = down_proj · act`  `[d_in]`
fn dense_situ_ffn_forward<'s>(
    expert: &SwiGluExpertWeights,
    hidden: &[f32],
    scratch: &'s mut KimiFfnScratch,
    beta: f32,
    linear_beta: Option<f32>,
) -> &'s [f32] {
    use katgpt_core::simd::simd_matmul_rows;
    use katgpt_core::types::math::situ;

    let d_in = scratch.dense_out.len();
    let d_ffn = expert.gate_proj.len() / d_in;

    // gate = gate_proj · h  [d_ffn]
    simd_matmul_rows(
        &mut scratch.dense_gate,
        &expert.gate_proj,
        hidden,
        d_ffn,
        d_in,
    );
    // up = up_proj · h  [d_ffn]
    simd_matmul_rows(&mut scratch.dense_up, &expert.up_proj, hidden, d_ffn, d_in);

    // SiTU activation: act = SiTU(gate, up, beta, linear_beta)
    // situ(hidden/output, gate, up, beta, linear_beta)
    situ(
        &mut scratch.dense_act,
        &scratch.dense_gate,
        &scratch.dense_up,
        beta,
        linear_beta,
    );

    // out = down_proj · act  [d_in]
    simd_matmul_rows(
        &mut scratch.dense_out,
        &expert.down_proj,
        &scratch.dense_act,
        d_in,
        d_ffn,
    );

    &scratch.dense_out[..d_in]
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_layer(
        layer_idx: usize,
        d: usize,
    ) -> (
        KimiDecoderLayerConfig,
        KimiDecoderLayerWeights,
        KimiAttentionState,
        KimiAttentionScratch,
        KimiFfnScratch,
        AttnResScratch,
        AttnResScratch,
    ) {
        let is_mla = layer_idx == 3 || layer_idx == 7;
        let is_dense = layer_idx == 0;

        let (attn_cfg, attn_w, attn_state, attn_scratch) = if is_mla {
            let cfg = MlaConfig::kimi_k3_0_40b();
            let w = MlaWeights::random(&cfg, layer_idx as u64 * 1000 + 1);
            let cache = MlaKVCache::new(&cfg, 64);
            let scratch = MlaForwardScratch::new(&cfg, 64);
            (
                KimiAttentionConfig::Mla(cfg),
                KimiAttentionWeights::Mla(w),
                KimiAttentionState::Mla(cache),
                KimiAttentionScratch::Mla(scratch),
            )
        } else {
            let cfg = KdaConfig::kimi_k3_0_40b();
            let w = KdaWeights::random(&cfg, layer_idx as u64 * 1000 + 1);
            let cache = KdaLayerCache::new(&cfg);
            let scratch = KdaForwardScratch::new(&cfg);
            (
                KimiAttentionConfig::Kda(cfg),
                KimiAttentionWeights::Kda(w),
                KimiAttentionState::Kda(cache),
                KimiAttentionScratch::Kda(scratch),
            )
        };

        let d_ffn_dense = 2048;
        let (ffn_cfg, ffn_w, ffn_scratch) = if is_dense {
            let expert = SwiGluExpertWeights {
                gate_proj: (0..d_ffn_dense * d)
                    .map(|i| (i as f32 % 10.0 - 5.0) * 0.01)
                    .collect(),
                up_proj: (0..d_ffn_dense * d)
                    .map(|i| (i as f32 % 10.0 - 5.0) * 0.01)
                    .collect(),
                down_proj: (0..d * d_ffn_dense)
                    .map(|i| (i as f32 % 10.0 - 5.0) * 0.01)
                    .collect(),
            };
            (
                KimiFfnConfig::Dense {
                    intermediate_size: d_ffn_dense,
                    hidden_size: d,
                    situ_beta: 4.0,
                    situ_linear_beta: Some(25.0),
                },
                KimiFfnWeights::Dense(expert),
                KimiFfnScratch {
                    moe: MoeForwardScratch::new(&MoeConfig::kimi_k3_0_40b()),
                    dense_gate: vec![0.0; d_ffn_dense],
                    dense_up: vec![0.0; d_ffn_dense],
                    dense_act: vec![0.0; d_ffn_dense],
                    dense_out: vec![0.0; d],
                },
            )
        } else {
            let cfg = MoeConfig::kimi_k3_0_40b();
            let w = MoeWeights::random(&cfg, layer_idx as u64 * 1000 + 2);
            (
                KimiFfnConfig::Moe(cfg.clone()),
                KimiFfnWeights::Moe(w),
                KimiFfnScratch {
                    moe: MoeForwardScratch::new(&cfg),
                    dense_gate: Vec::new(),
                    dense_up: Vec::new(),
                    dense_act: Vec::new(),
                    dense_out: vec![0.0; d],
                },
            )
        };

        let attn_res_cfg = AttnResConfig::kimi_k3_0_40b();
        let self_attn_res = AttnResWeights::random(d, layer_idx as u64 * 1000 + 3);
        let mlp_attn_res = AttnResWeights::random(d, layer_idx as u64 * 1000 + 4);

        let max_block_entries = 3;
        let self_scratch = AttnResScratch::new(&attn_res_cfg, max_block_entries);
        let mlp_scratch = AttnResScratch::new(&attn_res_cfg, max_block_entries);

        let layer_cfg = KimiDecoderLayerConfig {
            rms_eps: 1e-5,
            attention: attn_cfg,
            ffn: ffn_cfg,
            attn_res: attn_res_cfg,
        };
        let layer_w = KimiDecoderLayerWeights {
            input_layernorm_weight: vec![1.0; d],
            post_attention_layernorm_weight: vec![1.0; d],
            attention: attn_w,
            ffn: ffn_w,
            self_attn_res,
            mlp_attn_res,
        };

        (
            layer_cfg,
            layer_w,
            attn_state,
            attn_scratch,
            ffn_scratch,
            self_scratch,
            mlp_scratch,
        )
    }

    #[test]
    fn smoke_layer_0_kda_dense() {
        // Layer 0: KDA + Dense MLP (the first dense layer)
        let d = 1024;
        let (cfg, w, mut attn_state, mut attn_scratch, mut ffn_scratch, mut self_res, mut mlp_res) =
            make_test_layer(0, d);

        let mut prefix_sum: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.1).collect();
        let mut scratch_hidden = vec![0.0; d];
        let mut block_state = AttnResBlockState::new(d);

        kimi_decoder_layer_forward(
            0,
            &cfg,
            &w,
            &mut attn_state,
            &mut attn_scratch,
            &mut ffn_scratch,
            &mut self_res,
            &mut mlp_res,
            &mut block_state,
            None,
            &mut prefix_sum,
            &mut scratch_hidden,
        );

        for &v in &prefix_sum {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn smoke_layer_3_mla_moe() {
        // Layer 3: MLA + MoE (MLA at layers 3,7 per verified topology)
        let d = 1024;
        let (cfg, w, mut attn_state, mut attn_scratch, mut ffn_scratch, mut self_res, mut mlp_res) =
            make_test_layer(3, d);

        let mut prefix_sum: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.1).collect();
        let mut scratch_hidden = vec![0.0; d];
        let mut block_state = AttnResBlockState::new(d);

        // MLA requires rope_freqs (d_r = qk_rope_head_dim = 32, theta = 10_000)
        let mla_cfg = match &cfg.attention {
            KimiAttentionConfig::Mla(c) => c.clone(),
            _ => unreachable!(),
        };
        let mut rope = RopeFreqs::new_with_theta(mla_cfg.d_r(), mla_cfg.rope_theta);

        kimi_decoder_layer_forward(
            3,
            &cfg,
            &w,
            &mut attn_state,
            &mut attn_scratch,
            &mut ffn_scratch,
            &mut self_res,
            &mut mlp_res,
            &mut block_state,
            Some(&mut rope),
            &mut prefix_sum,
            &mut scratch_hidden,
        );

        for &v in &prefix_sum {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn block_boundary_pushes_on_layer_0() {
        // Block boundary: layer_idx % block_size == 0.
        // Layer 0 is a block boundary (0 % 4 == 0) → push happens.
        let d = 1024;
        let (cfg, w, mut attn_state, mut attn_scratch, mut ffn_scratch, mut self_res, mut mlp_res) =
            make_test_layer(0, d);

        let mut prefix_sum: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.1).collect();
        let mut scratch_hidden = vec![0.0; d];
        let mut block_state = AttnResBlockState::new(d);

        assert_eq!(block_state.len(), 0);

        kimi_decoder_layer_forward(
            0,
            &cfg,
            &w,
            &mut attn_state,
            &mut attn_scratch,
            &mut ffn_scratch,
            &mut self_res,
            &mut mlp_res,
            &mut block_state,
            None,
            &mut prefix_sum,
            &mut scratch_hidden,
        );

        assert_eq!(
            block_state.len(),
            1,
            "block boundary layer 0 should push 1 entry"
        );
    }

    #[test]
    fn non_boundary_does_not_push() {
        // Layer 1 is NOT a block boundary (1 % 4 != 0).
        let d = 1024;
        let (cfg, w, mut attn_state, mut attn_scratch, mut ffn_scratch, mut self_res, mut mlp_res) =
            make_test_layer(1, d);

        let mut prefix_sum: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.1).collect();
        let mut scratch_hidden = vec![0.0; d];
        let mut block_state = AttnResBlockState::new(d);

        kimi_decoder_layer_forward(
            1,
            &cfg,
            &w,
            &mut attn_state,
            &mut attn_scratch,
            &mut ffn_scratch,
            &mut self_res,
            &mut mlp_res,
            &mut block_state,
            None,
            &mut prefix_sum,
            &mut scratch_hidden,
        );

        assert_eq!(block_state.len(), 0, "non-boundary layer should not push");
    }

    #[test]
    fn layer_3_mla_requires_rope_freqs() {
        // MLA at layer 3: passing None for rope_freqs panics.
        let d = 1024;
        let (cfg, w, mut attn_state, mut attn_scratch, mut ffn_scratch, mut self_res, mut mlp_res) =
            make_test_layer(3, d);

        let mut prefix_sum: Vec<f32> = vec![0.0; d];
        let mut scratch_hidden = vec![0.0; d];
        let mut block_state = AttnResBlockState::new(d);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            kimi_decoder_layer_forward(
                3,
                &cfg,
                &w,
                &mut attn_state,
                &mut attn_scratch,
                &mut ffn_scratch,
                &mut self_res,
                &mut mlp_res,
                &mut block_state,
                None,
                &mut prefix_sum,
                &mut scratch_hidden,
            );
        }));

        // The panic happens inside mla_forward_token when rope_freqs is None
        // (we pass None to the forward, which panics on MLA path).
        // Actually, looking at the code: we only call mla_forward_token when
        // rope_freqs is Some. So passing None to an MLA layer panics with
        // "MLA attention requires rope_freqs".
        assert!(
            result.is_err(),
            "MLA layer with None rope_freqs should panic"
        );
    }
}
