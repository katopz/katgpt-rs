//! DashAttention — Adaptive Sparse Hierarchical Attention via α-entmax routing.
//!
//! This module owns the DashAttention primitives. The clean core (`entmax`,
//! `routing`, `chunk_summary`) has zero cross-domain deps. The VortexFlow cluster
//! (`vortex_flow`, `block_topk`, `channel_aware`, `entmax_router`,
//! `kv_outer_prefill`, `msa_distill`, `value_energy`, `adaptive_k`,
//! `meta_router`, `sat_analysis`) moved here in Phase 12 (2026-07-04) — the
//! original blocker (root-only `pruners::bandit` + `speculative::types`)
//! dissolved once those landed in `katgpt-pruners` and `katgpt-core::traits`.
//! `meta_router` now imports from `katgpt_pruners::bandit` +
//! `katgpt_core::traits::ScreeningPruner`; `sat_analysis` imports from
//! `katgpt_kv::cache_prune::SummedAreaTable`.
//!
//! The composition layer (`forward_dash_attn_prefill` / `forward_dash_attn_decode`,
//! which take `ForwardContext`) also lives here. The token-level
//! `forward_dash_attn_decode_vortex` variant stays in root `src/dash_attn/tests.rs`
//! (needs root transformer glue).
//!
//! Feature gate: `dash_attn` (Plan 106, Research 68).

pub mod chunk_summary;
// ASEntmax length-adaptive damping schedule (Issue 747 P0, Research 549 —
// arXiv:2506.16640 Eq 10): the entmax-side mirror of katgpt-core's SSMax
// socket. Opt-in until the Issue 747 GOAT gate.
#[cfg(feature = "asentmax_schedule")]
pub mod asentmax;
pub mod entmax;
// Issue 747 P2/P3 (Research 549 — arXiv:2506.16640 Prop E.2 + Lemma 3.1):
// theorem-backed ALiBi×entmax KV-eviction window + Lemma-1 incremental
// decode entmax. Same family flag as the schedule (P1 precedent).
#[cfg(feature = "asentmax_schedule")]
pub mod entmax_incremental;
#[cfg(feature = "asentmax_schedule")]
pub mod eviction_window;
pub mod routing;
// Composition layer (Issue 007 Phase F.4a, 2026-07-02):
// forward_dash_attn_prefill / forward_dash_attn_decode moved here from root
// `src/dash_attn/forward.rs`. NOTE: forward_dash_attn_decode_vortex was
// STRIPPED (vortex_flow cluster was root-only at the time) — see forward.rs
// comment. Phase 12 (2026-07-04) moved the vortex_flow primitives here, but
// the stripped decode variant was never re-added (no consumer needed it).
pub mod forward;

// ── Phase 12 absorption (Proposal 003, 2026-07-04): VortexFlow cluster moved
// here from root `src/dash_attn/`. Zero-dep primitives + katgpt-core simd
// consumers + two cross-crate deps (meta_router→katgpt-pruners+katgpt-core::traits,
// sat_analysis→katgpt-kv). All deps resolve cleanly; the original "stays root"
// blocker dissolved when pruners/speculative/cache_prune landed in their leaves.
pub mod adaptive_k;
pub mod block_topk;
pub mod channel_aware;
pub mod entmax_router;
// FlashMemory-style periodic sparse attention for MLA (Issue 584 Phase 1).
// Gated by `flashmemory_sparse` (implies `mla_attention` + `dash_attn`).
#[cfg(feature = "flashmemory_sparse")]
pub mod flashmemory_sparse;
pub mod kv_outer_prefill;
pub mod meta_router;
pub mod msa_distill;
pub mod sat_analysis;
pub mod value_energy;
pub mod vortex_flow;

pub use chunk_summary::{ChunkSummaryCache, ChunkSummaryQuery, summarize_chunk_with_entropy};
#[cfg(feature = "asentmax_schedule")]
pub use asentmax::{AsentmaxSchedule, RollingSigmaEstimator, apply_asentmax_inplace};
pub use entmax::{entmax_1p5, entmax_gqa_aggregate, entmax_support};
// Issue 747 P3: Lemma-1 incremental decode entmax.
#[cfg(feature = "asentmax_schedule")]
pub use entmax_incremental::IncrementalEntmax1p5;
// Issue 747 P2: ALiBi×entmax eviction window (Prop E.2).
#[cfg(feature = "asentmax_schedule")]
pub use eviction_window::{alibi_entmax_window_1p5, evicted_kv_fraction, kv_within_window};
pub use forward::{forward_dash_attn_decode, forward_dash_attn_prefill};
pub use routing::{compute_routing_bias, score_blocks_entmax, score_blocks_entmax_with_entropy};
#[cfg(feature = "asentmax_schedule")]
pub use routing::score_blocks_entmax_with_schedule_into;
