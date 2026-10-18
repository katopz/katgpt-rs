//! Decode-order instrumentation for discrete/diffusion LM lanes (Plan 602).
//!
//! Houses the modelless decode-order instruments distilled from dQwen3.5:
//! Hybrid-Attention Diffusion Language Models (arXiv:2609.20751, Research
//! 575) — statistics over a decode trajectory π that measure how
//!      autoregressive-like a diffusion decode was, without touching weights.
//!
//! π is the per-position unmask step: `π[i]` = the decode iteration at which
//! position `i` committed (see [`UNMASKED_NEVER`] for the never-committed
//! sentinel). The decode emitters live in katgpt-forward
//! (`d2f_decode_block_with_unmask_steps`, `SetDiffusionResult::unmask_steps`)
//! behind the same `decode_order_metrics` feature; this module is the pure
//! math half (zero alloc, no deps).
//!
//! OPT-IN pending the Phase-3 G3 GOAT (predictor-chosen (w, block size)
//! matches or beats fixed settings at matched NFE) — promotion to default is
//! NOT claimed here.
//!
//! Plan 602 (2026-09-20): decode-order AR-ness instruments for the DLM lane
//! (ALR/AGR over per-position unmask steps). katgpt-forward's π-logging
//! forwards this same feature.

pub mod arness;

pub use arness::{UNMASKED_NEVER, global_ar_ness, global_ar_ness_windowed, local_ar_ness};
