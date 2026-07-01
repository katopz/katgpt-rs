//! DashAttention — Adaptive Sparse Hierarchical Attention via α-entmax routing.
//!
//! This module owns the DashAttention *clean core* primitives: `entmax`,
//! `routing`, and `chunk_summary`. These are the self-contained kernels with
//! no cross-domain dependencies.
//!
//! The VortexFlow cluster (`vortex_flow`, `block_topk`, `channel_aware`,
//! `entmax_router`, `kv_outer_prefill`, `msa_distill`, `value_energy`,
//! `adaptive_k`, `meta_router`, `sat_analysis`) stays in the root crate because
//! `meta_router` depends on `pruners::bandit` + `speculative::types` (root-only
//! modules), and `vortex_flow` depends on `meta_router` — creating a dependency
//! chain that can't resolve in katgpt-attn without a circular dep.
//!
//! The composition layer (`forward_dash_attn_prefill` / `forward_dash_attn_decode`,
//! which take `ForwardContext`) also stays in root.
//!
//! Feature gate: `dash_attn` (Plan 106, Research 68).

pub mod chunk_summary;
pub mod entmax;
pub mod routing;

pub use chunk_summary::{ChunkSummaryCache, ChunkSummaryQuery};
pub use entmax::{entmax_1p5, entmax_gqa_aggregate, entmax_support};
pub use routing::{compute_routing_bias, score_blocks_entmax};
