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

pub use chunk_summary::{ChunkSummaryCache, ChunkSummaryQuery};
pub use entmax::{entmax_1p5, entmax_gqa_aggregate, entmax_support};
pub use forward::{forward_dash_attn_decode, forward_dash_attn_prefill};
pub use routing::{compute_routing_bias, score_blocks_entmax};
