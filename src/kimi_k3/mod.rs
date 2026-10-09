//! Kimi-K3 native support — decoder layer composition + model loader.
//!
//! This module is the Phase 5+ composition layer for Kimi-K3-0.40B
//! (Proposal 032). It lives in the katgpt-rs root crate because the decoder
//! layer needs BOTH:
//! - MLA/KDA from `katgpt-attn`
//! - MoE/attn-res from `katgpt-transformer`
//!
//! The root crate depends on both leaf crates, making it the natural
//! composition point.
//!
//! # Feature gates
//!
//! - `kimi_k3` — decoder layer composition (this module). Pulls in all four
//!   substrate primitives: `mla_attention`, `kda_linear`, `transformer_moe`,
//!   `transformer_attn_res`.
//! - `kimi_k3_loader` — adds the safetensors loader + tiktoken tokenizer
//!   (heavy deps: `safetensors`, `base64`).
//!
//! Both are opt-in (default-off). Promotion to default requires the Phase 6
//! real-model GOAT gate (logits match PyTorch reference).

pub mod decoder_layer;

/// Full-model analytic backward (Plan 318 Phase C C6).
///
/// Composes the three per-primitive backward modules (MLA + MoE + KDA from
/// C4/C5) with the model-level composition backward (attn-res + RMSNorm +
/// dense SiTU FFN + LM head + embedding) into a full-model gradient pass.
/// CPU reference for the GPU training loop (C10).
#[cfg(feature = "kimi_k3_backward")]
pub mod backward;

/// Gradient checkpointing backward (Plan 318 Phase C C7).
///
/// Recompute-activations variant of the full-model backward — cuts activation
/// memory from ~24 GB to ~5 GB at the cost of one extra forward pass during
/// backward. Same gradient output as `backward::kimi_k3_backward_sequence`.
#[cfg(feature = "kimi_k3_backward")]
pub mod checkpoint;

#[cfg(feature = "kimi_k3_loader")]
pub mod model;

/// Delay-architecture forward + analytic backward (riir-train Issue 482 /
/// Plan 452): `DelayArchConfig { dense_delay, expert_delay }` — the pre-dense
/// routing anchor + delayed expert-output injection over the split-stream MoE
/// substrate (`moe_forward_token_split`). G0 law: δ0/δe0 is BIT-IDENTICAL to
/// the standard forward; the backward matches within the 1-ULP split
/// reassociation band. Training-time CPU reference, opt-in.
#[cfg(feature = "delay_arch")]
pub mod delay;

pub use decoder_layer::{
    KimiAttentionConfig, KimiAttentionScratch, KimiAttentionState, KimiAttentionWeights,
    KimiDecoderLayerConfig, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnScratch, KimiFfnWeights,
    kimi_decoder_layer_forward,
};

#[cfg(feature = "kimi_k3_loader")]
pub use model::{
    ForwardTiming, KimiK3ModelConfig, KimiK3Runtime, PauseConfig, PauseStrategy,
    kimi_k3_forward_token, kimi_k3_forward_token_hidden, kimi_k3_forward_token_timed,
    kimi_k3_forward_token_traced, kimi_k3_inject_pause, kimi_k3_pause_step,
};

/// Stale-residual speculative layer-execution simulator (Issue 691 /
/// Research 508, arXiv:2608.23841 §6.3): runtime snapshot/restore, true-run
/// per-layer capture, replay-from-layer-ℓ+1 on stale/corrected residuals,
/// KL/top-1 outcome metrics, and the Approach-B closed-form δ-predictors
/// (router-logit + x_in-linear, fit via the katgpt-attn-match OLS substrate).
// Issue 691: the module's code consumes katgpt_core::stale_residual, which
// only exists under the root `stale_residual` feature (it forwards
// katgpt-core/stale_residual). Gating on `kimi_k3_loader` alone made the lib
// un-compilable under every loader-implying feature set (kimi_k3_backward
// included) — found by Issue 694's PoC harness check at clean HEAD 271a8b10.
#[cfg(all(feature = "kimi_k3_loader", feature = "stale_residual"))]
pub mod stale_residual;

// Safetensors loader + tiktoken tokenizer (gated by `kimi_k3_loader`).
#[cfg(feature = "kimi_k3_loader")]
pub mod loader;
#[cfg(feature = "kimi_k3_loader")]
pub mod tiktoken;
