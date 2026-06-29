//! SP-KV: Self-Pruned Key-Value Attention.
//!
//! Based on "Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility"
//! (arXiv:2605.14037, Meta FAIR). Learns which KV pairs to retain by predicting future utility.
//!
//! ## Architecture
//!
//! ```text
//! h ∈ R^{d_model}
//!   → UtilityPredictor (2-layer MLP: SiLU → sigmoid)
//!   → u ∈ (0,1) per KV head
//!   → gate_bias = log(u) (soft) or 0|-∞ (hard)
//!   → attention scores += gate_bias
//!   → conditional KV cache write if u ≥ τ
//! ```
//!
//! ## Pipeline Composability
//!
//! | Stage    | Mechanism  | Role                          |
//! |----------|------------|-------------------------------|
//! | Prefill  | PFlash     | Block-sparse token selection  |
//! | Decode   | **SP-KV**  | Selective KV write            |
//! | Storage  | TurboQuant | Quantize retained KV entries  |
//!
//! ## Feature Flag
//!
//! Enabled via `sp_kv` feature in `Cargo.toml`.
//!
//! ## Glue code split (Issue 015 Phase 3)
//!
//! The full pipeline functions `forward_sp_kv` / `forward_sp_kv_quant` live
//! in the ROOT crate at `src/sp_kv_forward.rs` because they take
//! `crate::transformer::ForwardContext` (a root-crate type with ~15 fields
//! that cannot be made generic without an unergonomic trait). katgpt-kv owns
//! the types and utility predictor; the root crate owns the transformer glue.

pub mod types;
pub mod utility_predictor;

pub use types::{
    GateBiasBuffer, SpKvCache, SpKvConfig, SpKvGateMode, SpKvLayerCache, SpKvPredictors,
    SpKvQuantCache, SpKvQuantLayerMeta, UtilityPredictorWeights,
};
pub use utility_predictor::{
    UtilityAggregation, aggregate_utilities, hard_gate_bias, predict, predict_into,
    predict_single_head, soft_gate_bias, tahg_gate_bias,
};
