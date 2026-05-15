//! Higher-order Linear Attention (HLA) — O(1) inference cache.
//!
//! Implements second-order HLA (symmetric + asymmetric AHLA) as an alternative
//! to standard KV-cache attention. Achieves O(1) per-token memory independent
//! of sequence length, replacing the growing KV cache with fixed-size prefix
//! sufficient statistics that capture higher-order query-key interactions.
//!
//! # Variants
//!
//! | Variant | State per head | Per-token cost | Best for |
//! |---------|---------------|---------------|----------|
//! | **Symmetric HLA** | O(d² + d·dv) | O(d² + d·dv) | Small hd, quality-critical |
//! | **AHLA** (asymmetric) | O(d·dv) | O(d·dv) | Large hd, perf-critical |
//!
//! # Usage
//!
//! ```ignore
//! use microgpt::hla::{forward_hla, MultiLayerHlaCache};
//!
//! let config = Config::micro();
//! let weights = TransformerWeights::random(&config);
//! let mut ctx = ForwardContext::new(&config);
//! let mut cache = MultiLayerHlaCache::new(&config);
//!
//! // Streaming inference — no context window limit
//! let logits = forward_hla(&mut ctx, &weights, &mut cache, token, pos, &config);
//! ```
//!
//! # Key Insight
//!
//! The second-order attention matrix QKᵀQKᵀᵀ = Q(KᵀK)Qᵀ depends only on
//! KᵀK (a d×d matrix), not the full N×N attention matrix. HLA maintains
//! running summaries (prefix sufficient statistics) of these moments.
//!
//! **Note:** HLA computes a different function than softmax attention.
//! Models must be trained with HLA from scratch for quality.
//!
//! Reference: Zhang, Qin, Wang, Gu (2026). "Higher-order Linear Attention."
//! See `.research/28_Higher_order_Linear_Attention.md` for full derivation.

pub mod forward;
pub mod kernel;
pub mod types;

pub use forward::{forward_ahla, forward_hla, generate_ahla_into, generate_hla_into};
pub use kernel::{
    ahla_denom, ahla_layer_step, ahla_step, hla_denom, hla_layer_readout, hla_layer_update,
    hla_readout, hla_state_update,
};
pub use types::{
    AhlaLayerState, AhlaQHeadState, HlaLayerState, HlaQHeadState, HlaVariant, MultiLayerAhlaCache,
    MultiLayerHlaCache,
};
