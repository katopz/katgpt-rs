#![allow(unexpected_cfgs)]

// Issue 125 (2026-07-10): removed 136 `pub use katgpt_*` back-compat re-export
// shims. Call sites now import from the owning leaf crate directly:
//   `katgpt_rs::cce`       → `katgpt_core::cce`
//   `katgpt_rs::clr`       → `katgpt_claim::clr`
//   `katgpt_rs::compaction`→ `katgpt_core::compaction`
//   ... (see .issues/125_remove_root_reexport_shims.md + .benchmarks/125_reexport_map.md)
//
// The remaining `pub mod X { pub use katgpt_Y::X::*; }` blocks below are
// module-scoped re-exports that are entangled with genuine root code or provide
// the flat-import surface root modules need. They are a separate concern.

// ── Device backends (macOS-only re-export blocks) ──────────────────────────
#[cfg(all(target_os = "macos", feature = "ane"))]
pub mod ane_backend {
    pub use katgpt_backend::{AneBackend, AneError};
}
#[cfg(all(target_os = "macos", feature = "gpu_inference"))]
pub mod gpu_backend {
    pub use katgpt_backend::GpuBackend;
}

// ── Adaptive CoT compaction glue (genuine root module) ─────────────────────
#[cfg(feature = "adaptive_cot_compaction")]
pub mod attn_match_adaptive_cot;

// ── Kimi-K3 native support (Proposal 032) ──────────────────────────────────
// Decoder layer composition + (optional) safetensors loader + tiktoken tokenizer.
// Gated by `kimi_k3` (composition) / `kimi_k3_loader` (loader + tokenizer).
#[cfg(feature = "kimi_k3")]
pub mod kimi_k3;

// ── Benchmark harness ──────────────────────────────────────────────────────
pub mod benchmark;

// ── DashAttention module-scoped re-export ──────────────────────────────────
#[cfg(feature = "dash_attn")]
pub mod dash_attn {
    pub use katgpt_attn::dash_attn::chunk_summary::{
        ChunkSummaryCache, ChunkSummaryQuery, summarize_chunk_with_entropy,
    };
    pub use katgpt_attn::dash_attn::entmax::{entmax_1p5, entmax_gqa_aggregate, entmax_support};
    pub use katgpt_attn::dash_attn::forward::{
        forward_dash_attn_decode, forward_dash_attn_prefill,
    };
    pub use katgpt_attn::dash_attn::routing::{
        compute_routing_bias, score_blocks_entmax, score_blocks_entmax_with_entropy,
    };
    pub use katgpt_attn::dash_attn::{
        adaptive_k, block_topk, channel_aware, chunk_summary, entmax, entmax_router, forward,
        kv_outer_prefill, meta_router, msa_distill, routing, sat_analysis, value_energy,
        vortex_flow,
    };

    #[cfg(feature = "msa_adaptive_k")]
    pub use katgpt_attn::dash_attn::adaptive_k::{AdaptiveKConfig, AdaptiveKRouter};
    #[cfg(feature = "msa_per_group")]
    pub use katgpt_attn::dash_attn::block_topk::PerGroupTopKRouter;
    #[cfg(feature = "vortex_flow")]
    pub use katgpt_attn::dash_attn::block_topk::{BlockTopKCache, BlockTopKRouter};
    #[cfg(feature = "vortex_flow")]
    pub use katgpt_attn::dash_attn::channel_aware::{
        ChannelAwareCache, ChannelAwareRouter, RoutingChannelDiscovery, RoutingChannelMask,
        simd_dot_f32,
    };
    #[cfg(feature = "vortex_flow")]
    pub use katgpt_attn::dash_attn::entmax_router::{EntmaxCache, EntmaxRouter};
    #[cfg(feature = "msa_kv_outer")]
    pub use katgpt_attn::dash_attn::kv_outer_prefill::{KvOuterIndex, KvOuterPrefill};
    #[cfg(feature = "vortex_flow")]
    pub use katgpt_attn::dash_attn::meta_router::{
        DynPolicy, DynRoutingCache, MetaRouter, compute_reward,
    };
    #[cfg(feature = "msa_sparse")]
    pub use katgpt_attn::dash_attn::msa_distill::{
        MaxPoolBlockScorer, MaxStdDevBlockScorer, MsaBlockCache,
    };
    #[cfg(all(feature = "dash_attn", feature = "cache_prune"))]
    pub use katgpt_attn::dash_attn::sat_analysis::{HeadSparsityInfo, head_sparsity_profile};
    #[cfg(feature = "vortex_flow")]
    pub use katgpt_attn::dash_attn::value_energy::{ValueEnergyCache, ValueEnergyRouter};
    #[cfg(feature = "vortex_flow")]
    pub use katgpt_attn::dash_attn::vortex_flow::{
        RoutingDecision, VortexFlow, VortexFlowConfig, VortexFlowExt, VortexRouter,
        VortexRouterCache, VortexScratch, build_vortex_router,
    };
}

// ── Data Probe module-scoped re-export ─────────────────────────────────────
#[cfg(feature = "data_probe")]
pub mod data_probe {
    pub mod markov {
        pub use katgpt_core::data_probe::markov::*;
    }
    pub mod nll {
        pub use katgpt_core::data_probe::nll::*;
    }
    pub mod typical_set {
        pub use katgpt_core::data_probe::typical_set::*;
    }
    pub mod dirichlet_energy {
        pub use katgpt_core::data_probe::dirichlet_energy::*;
    }
    pub mod claim {
        pub use katgpt_core::data_probe::claim::*;
    }
    #[cfg(feature = "sink_aware_attn")]
    pub mod geometry {
        pub use katgpt_core::data_probe::geometry::*;
    }
    #[cfg(feature = "knn_entropy")]
    pub mod entropy {
        pub use katgpt_core::data_probe::entropy::*;
    }
    #[cfg(feature = "sink_aware_attn")]
    pub mod sink_classify {
        pub use katgpt_core::data_probe::{
            CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic,
            SinkKind, StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
            apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
            classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat,
            stable_rank_update_into, stable_rank_update_into_flat,
        };
    }
    #[cfg(feature = "sink_aware_attn")]
    pub use katgpt_core::data_probe::{
        CachedSinkClassification, GeometryReport, LayerSinkSummary, SinkAwarePolicy,
        SinkClassifierConfig, SinkDiagnostic, SinkKind, StableRankScratch, apply_dual_policy_gate,
        apply_dual_policy_gate_cached, apply_dual_policy_gate_cached_flat,
        apply_dual_policy_gate_flat, avg_cosine_similarity, classify_all_sinks,
        classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, effective_rank,
        representation_geometry_report, stable_rank_update_into, stable_rank_update_into_flat,
        summarize_layer_sinks,
    };
    pub use katgpt_core::data_probe::{
        ClaimCard, Intervention, MarkovChain, Regime, RegimeDistribution, ValidityVerdict,
        average_nll, classify_regime, consecutive_adjacency, functor_adjacency,
        generate_markov_chain, kv_cache_dirichlet_energy, nll_profile, regime_distribution,
        sample_sequence,
    };
}

// ── Distillation umbrella module-scoped re-export ──────────────────────────
#[cfg(any(
    feature = "peira_distill",
    feature = "ilc_distill",
    feature = "trd_refined_draft"
))]
pub mod distill {
    #[cfg(feature = "peira_distill")]
    pub use katgpt_spectral::peira;
    #[cfg(feature = "ilc_distill")]
    pub use katgpt_speculative::distill::ilc;
    #[cfg(feature = "trd_refined_draft")]
    pub use katgpt_speculative::distill::trd;
}

// ── D2F / Discrete Diffusion (genuine root module) ─────────────────────────
#[cfg(feature = "dllm")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub mod dllm;

// ── GDN2 module-scoped re-export ───────────────────────────────────────────
#[cfg(feature = "gdn2_attention")]
pub mod gdn2 {
    pub use katgpt_attn::gdn2::forward::{forward_gdn2, generate_gdn2_into};
    #[cfg(feature = "gdn_tree_verify")]
    pub use katgpt_attn::gdn2::tree_verify_bridge;
    pub use katgpt_attn::gdn2::{
        Gdn2GateConfig, Gdn2HeadState, Gdn2LayerState, MultiLayerGdn2Cache, gdn2_recurrent_step,
        gdn2_state_readout, gdn2_state_update, l2_normalize, sigmoid,
    };
    pub use katgpt_attn::gdn2::{kernel, types};
    // Plan 436 T4.8: re-export `forward_tree_gdn2` so riir-gpu can build the
    // GPU-corrector speculative step without a direct katgpt-attn dep (cycle
    // safety: the GPU orchestration lives in riir-gpu, which reaches the tree
    // forward via this re-export under the `gdn_tree_verify` feature).
    #[cfg(feature = "gdn_tree_verify")]
    pub use katgpt_attn::gdn2::tree_forward::forward_tree_gdn2;
}

// ── HLA module-scoped re-export ────────────────────────────────────────────
#[cfg(feature = "hla_attention")]
pub mod hla {
    pub use katgpt_core::hla::{
        AhlaLayerState, AhlaQHeadState, HlaLayerState, HlaQHeadState, HlaVariant,
        MultiLayerAhlaCache, MultiLayerHlaCache, MultiLayerParallaxAhlaCache,
        ParallaxAhlaLayerState, ParallaxAhlaQHeadState,
    };
    pub use katgpt_core::hla::{
        ahla_denom, ahla_layer_step, ahla_step, hla_denom, hla_layer_readout, hla_layer_update,
        hla_readout, hla_readout_normalized, hla_state_update,
    };
    pub use katgpt_core::hla::{kernel, types};
    pub use katgpt_forward::{forward_ahla, forward_hla, generate_ahla_into, generate_hla_into};
}

// ── CompressionDrafter re-export (Plan 285, gated) ───────────────────────
#[cfg(feature = "compression_drafter")]
pub use katgpt_core::compression_drafter;

// ── Inference Router (genuine root module) ─────────────────────────────────
pub mod inference_router;

// ── Plot (genuine root module, gated) ──────────────────────────────────────
#[cfg(feature = "plot")]
pub mod plot;

// ── Pruners (genuine root module — bomber + katgpt-pruners re-export) ──────
pub mod pruners;

// ── DenseMesh module-scoped re-export ──────────────────────────────────────
#[cfg(feature = "dense_mesh")]
pub mod dense_mesh {
    pub use katgpt_forward::TransformerNode;
    pub use katgpt_transformer::dense_mesh::*;
}

// ── Sleep consolidation (genuine root module) ──────────────────────────────
#[cfg(feature = "sleep_consolidation")]
pub mod sleep;

// ── SP-KV module-scoped re-export (combines leaf + root forward glue) ──────
#[cfg(feature = "sp_kv")]
pub mod sp_kv {
    pub use katgpt_kv::sp_kv::*;
    pub mod forward {
        pub use crate::sp_kv_forward_mod::*;
    }
    pub use crate::sp_kv_forward_mod::{
        GateBias, NoBias, SpKvForwardContext, attention_head_core, attention_head_gated,
        forward_sp_kv, forward_sp_kv_quant,
    };
}
#[cfg(feature = "sp_kv")]
mod sp_kv_forward_mod;

// ── SpectralQuant module-scoped re-export ──────────────────────────────────
#[cfg(feature = "spectral_quant")]
pub mod spectralquant {
    pub use katgpt_spectral::*;
}

// ── Speculative decoding (genuine root module) ─────────────────────────────
pub mod speculative;

// ── Transformer (genuine root module — re-exports + helpers) ───────────────
pub mod transformer;

// ── Types (genuine root module — re-exports + root-specific types) ─────────
pub mod types;

// ── Allocator registration (unit tests only) ───────────────────────────────
// Issue 721 T3: this used to be a bare `#[cfg(debug_assertions)]` — a
// LIBRARY registering the process allocator for every binary that links it.
// That conflicted with any consumer's own `#[global_allocator]` (riir-ai
// Issues 830/855, riir-train Issue 721 T4) and silently served counters to
// consumers whose gates then depended on this crate staying linked. The
// registration now lives only in this crate's own unit-test build — the
// same `cfg(all(test, debug_assertions))` house pattern katgpt-core uses —
// and every integration test / example that asserts on the alloc counters
// installs its own copy via `tests/common/alloc_tracking.rs`.
#[cfg(all(test, debug_assertions))]
#[global_allocator]
static GLOBAL_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;

// ── TF Loop (genuine root module) ──────────────────────────────────────────
#[cfg(feature = "tf_loop")]
pub mod tf_loop;

// ── Root-resident integration test (Issue 121) ─────────────────────────────
// dash_attn_tests consumes root transformer glue (ForwardContext,
// MultiLayerKVCache, TransformerWeights) that can't move to a leaf crate.
#[cfg(test)]
mod dash_attn_tests;
