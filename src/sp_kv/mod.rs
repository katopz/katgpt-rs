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

pub mod forward;
pub mod types;
pub mod utility_predictor;

pub use forward::{
    GateBias, NoBias, SpKvForwardContext, attention_head_core, attention_head_gated,
};
pub use types::{
    GateBiasBuffer, SpKvCache, SpKvConfig, SpKvGateMode, SpKvLayerCache, SpKvPredictors,
    UtilityPredictorWeights,
};
pub use utility_predictor::{
    UtilityAggregation, aggregate_utilities, hard_gate_bias, predict, predict_single_head,
    soft_gate_bias, tahg_gate_bias,
};
