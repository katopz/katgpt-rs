//! RTPurbo — Retrieval Head Sparse Decode via Low-Dimensional Indexing.
//!
//! Feature gate: `rt_turbo` (opt-in, requires `dash_attn`, Plan 126, Research 86).
//!
//! Adds head-wise retrieval/local classification + dynamic top-p token selection
//! for decode-phase sparse attention. Complements DashAttention's α-entmax block
//! routing with per-head specialization from the RTPurbo paper (arXiv 2605.16928).
//!
//! # Key Insight
//!
//! Only ~15% of attention heads ("retrieval heads") need full long-context access.
//! The remaining ~85% ("local heads") attend only to local context + attention sinks.
//! With a 16-dim pre-RoPE projection + dynamic top-p selection, you can sparsify
//! in ~600 training steps with near-lossless accuracy.
//!
//! # Architecture
//!
//! | Component | Purpose |
//! |-----------|---------|
//! | [`HeadCalibration`] | Offline needle-based per-head retrieval scoring |
//! | [`HeadClassification`] | Per-head role (retrieval vs local) + score |
//! | [`RetrievalProjection`] | Low-dim pre-RoPE W_Q/W_K per retrieval head |
//! | [`calibrate_from_scores`] | Partition heads into retrieval/local sets |
//! | [`compute_retrieval_score`] | Single-head retrieval score from attention matrix |
//!
//! # Calibration
//!
//! Heads are classified at load-time via attention entropy on calibration data.
//! One forward pass with identical "needle" spans at beginning and end of a document
//! is sufficient — head behavior is input-agnostic.
//!
//! # Feature Gate
//!
//! `rt_turbo` is opt-in and requires `dash_attn` as base. Must pass 6/6 GOAT proofs
//! before default-on promotion.

#[cfg(test)]
mod tests;

pub mod calibration;
pub mod forward;
pub mod projection;
pub mod top_p;

pub use calibration::{
    CalibrationConfigSnapshot, HeadCalibration, HeadClassification, calibrate_from_scores,
    compute_all_retrieval_scores, compute_retrieval_score,
};
pub use forward::{
    RtTurboCache, RtTurboDecodeResult, RtTurboPrefillResult, forward_rt_turbo_decode,
    forward_rt_turbo_prefill,
};
pub use projection::RetrievalProjection;
pub use top_p::{select_top_p, select_top_p_blockwise};
