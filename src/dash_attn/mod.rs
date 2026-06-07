//! DashAttention — Adaptive Sparse Hierarchical Attention via α-entmax routing.
//!
//! Feature gate: `dash_attn` (Plan 106, Research 68).
//! Replaces PFlash's fixed-budget top-k block selection with adaptive support
//! selection via α-entmax (α=1.5). Includes learned chunk summaries via head_cls.
//!
//! # Architecture
//!
//! | Component | Purpose |
//! |-----------|---------|
//! | [`entmax_1p5`] | α=1.5 entmax with closed-form threshold |
//! | [`ChunkSummaryQuery`] | Per-KV-head learned class token for chunk summarization |
//! | [`ChunkSummaryCache`] | Cached chunk summaries across layers |
//! | [`score_blocks_entmax`] | Adaptive sparse chunk routing |
//! | [`forward_dash_attn_prefill`] | Prefill with chunk summarization + entmax routing |
//! | [`forward_dash_attn_decode`] | Decode reusing cached summaries |

#[cfg(test)]
mod tests;

pub mod chunk_summary;
pub mod entmax;
pub mod forward;
pub mod routing;

#[cfg(all(feature = "dash_attn", feature = "cache_prune"))]
pub mod sat_analysis;

pub use chunk_summary::{ChunkSummaryCache, ChunkSummaryQuery};
pub use entmax::{entmax_1p5, entmax_gqa_aggregate, entmax_support};
pub use forward::{forward_dash_attn_decode, forward_dash_attn_prefill};

#[cfg(feature = "vortex_flow")]
pub use forward::forward_dash_attn_decode_vortex;
pub use routing::{compute_routing_bias, score_blocks_entmax};

#[cfg(all(feature = "dash_attn", feature = "cache_prune"))]
pub use sat_analysis::{HeadSparsityInfo, head_sparsity_profile};

// VortexFlow composable sparse routing (Plan 196, default-OFF)
#[cfg(feature = "vortex_flow")]
pub mod block_topk;
#[cfg(feature = "vortex_flow")]
pub mod channel_aware;
#[cfg(feature = "vortex_flow")]
pub mod entmax_router;
#[cfg(feature = "vortex_flow")]
pub mod meta_router;
#[cfg(feature = "vortex_flow")]
pub mod value_energy;
#[cfg(feature = "vortex_flow")]
pub mod vortex_flow;

#[cfg(feature = "vortex_flow")]
pub use block_topk::{BlockTopKCache, BlockTopKRouter};
#[cfg(feature = "vortex_flow")]
pub use channel_aware::{
    ChannelAwareCache, ChannelAwareRouter, RoutingChannelDiscovery, RoutingChannelMask,
    simd_dot_f32,
};
#[cfg(feature = "vortex_flow")]
pub use entmax_router::{EntmaxCache, EntmaxRouter};
#[cfg(feature = "vortex_flow")]
pub use meta_router::{DynPolicy, DynRoutingCache, MetaRouter, compute_reward};
#[cfg(feature = "vortex_flow")]
pub use value_energy::{ValueEnergyCache, ValueEnergyRouter};
#[cfg(feature = "vortex_flow")]
pub use vortex_flow::{
    RoutingDecision, VortexFlow, VortexFlowConfig, VortexFlowExt, VortexRouter, VortexRouterCache,
    VortexScratch, build_vortex_router,
};
