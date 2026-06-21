#![allow(unexpected_cfgs)]
#[cfg(all(target_os = "macos", feature = "ane"))]
pub mod ane_backend;
#[cfg(feature = "attn_match")]
pub mod attn_match;
#[cfg(feature = "async_qdq_overlap")]
pub mod async_qdq;
pub mod benchmark;
#[cfg(feature = "band_conditioner")]
pub mod band_conditioner;
#[cfg(feature = "bckvss")]
pub mod bckvss;
#[cfg(feature = "breakeven_routing")]
pub mod breakeven;
#[cfg(feature = "cache_prune")]
pub mod cache_prune;
#[cfg(feature = "channel_simd_align")]
pub mod channel_simd;
#[cfg(feature = "cgsp")]
pub mod cgsp;
#[cfg(feature = "clr")]
pub mod clr;
// CLR — Claim-Level Reliability runtime (Plan 284, Research 255).
// Opt-in behind the `clr` feature until G1-G5 GOAT gate passes. Re-exports the
// public surface so consumers can `use katgpt::clr_vote` etc. without nesting.
#[cfg(feature = "clr")]
pub use clr::{
    allocate_budget, brevity_tiebreak, Claim, ClaimExtractor, ClaimVerifier, ClrConfig, ClrScratch,
    Cluster, DirectionVectorSource, FnClaimExtractor, learning_potential, mgpo_sampling_weight,
    ReliabilityScore, should_write_memory, SigmoidProjectionVerifier, Trajectory, Verdict,
    VoteResult, clr_vote, clr_vote_minimal,
};
pub mod cumprodsum;
#[cfg(feature = "ssd_block")]
pub mod ssd_block;
#[cfg(feature = "dash_attn")]
pub mod dash_attn;
#[cfg(feature = "data_probe")]
pub mod data_probe;
#[cfg(all(target_os = "macos", feature = "ane_npc"))]
pub mod npc_ane_backend;
#[cfg(feature = "sense_composition")]
pub mod npc_brain_router;
// Shared diagonal gate abstraction (GDN2 + Wall).
// Available when either gdn2_attention or wall_attention is enabled.
#[cfg(feature = "cubical_nerve")]
pub mod cubical_nerve;
#[cfg(feature = "collider_consistency")]
pub mod collider_pruner;
// CompressionDrafter — corpus-as-model quest grammar drafter (Plan 285, Research 256).
// Re-exports katgpt-core's compression_drafter module for downstream consumers (riir-games).
// Opt-in behind the `compression_drafter` feature until GOAT gate passes.
#[cfg(feature = "compression_drafter")]
pub use katgpt_core::compression_drafter;
#[cfg(feature = "cs_kv_probe")]
pub mod cs_kv_probe;
#[cfg(any(feature = "gdn2_attention", feature = "wall_attention"))]
pub mod diagonal_gate;
#[cfg(any(
    feature = "peira_distill",
    feature = "ilc_distill",
    feature = "trd_refined_draft"
))]
pub mod distill;
#[cfg(feature = "dllm")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub mod dllm;
#[cfg(feature = "critical_interval_gate")]
pub mod dllm_solver;
#[cfg(feature = "ega_attn")]
pub mod ega_attn;
#[cfg(feature = "feedback")]
pub mod feedback;
#[cfg(feature = "chain_fold")]
pub mod fold;
// CCE — Coarse Correlated Equilibria moderator primitives (Plan 295 + Plan 300, Research 274, arxiv 2606.20062).
// Generic, game-agnostic LP-CCE formulation + external-regret functional +
// heterogeneous (subjective-CCE) extension + primal-dual iterator.
// DEFAULT-ON after GOAT gates all PASS (G1+G2+G3+G4): G1 homogeneous
// equivalence regression, G2 regret transfer on synthetic heterogeneous
// CWMs (er_heterogeneous(ρ⋆) ≤ 1e-3), G3 primal-dual convergence at log-log
// slope -1.0 (beats paper's -0.5 O(N⁻¹ᐟ²) bound — Plan 300 T4.3b), G4 16-player
// latency = 33.97ms < 50ms target.
#[cfg(feature = "cce_moderator")]
pub mod cce;
#[cfg(feature = "freq_bandit")]
pub mod freq_bandit;
#[cfg(feature = "gdn2_attention")]
pub mod gdn2;
#[cfg(all(target_os = "macos", feature = "gpu_inference"))]
pub mod gpu_backend;
#[cfg(feature = "hla_attention")]
pub mod hla;
#[cfg(feature = "hybrid_oct_pq")]
pub mod hybrid_oct_pq;
pub mod inference_backend;
pub mod inference_router;
#[cfg(feature = "interval_pruner")]
pub mod interval_pruner;
#[cfg(feature = "iso_quant")]
pub mod iso_quant;
#[cfg(feature = "kv_share")]
pub mod kv_share;
pub mod kvarn;
#[cfg(feature = "lattice_operad")]
pub mod lattice_operad;
#[cfg(feature = "gauge_invariant")]
pub mod gauge_invariant;
pub mod spectral_retract;
#[cfg(feature = "manifold_power_iter_router")]
pub mod manifold_power_iter_router;
#[cfg(feature = "kog_cpu_fusion")]
pub mod mbu;
#[cfg(feature = "newton_schulz")]
pub mod newton_schulz;
#[cfg(feature = "off_principal_retrieval")]
pub mod off_principal;
#[cfg(feature = "octopus")]
pub mod octopus;
#[cfg(feature = "osc_kv")]
pub mod osc_kv;
pub mod percepta;
#[cfg(feature = "modality_pruned_load")]
pub mod pipeline_pruner;
#[cfg(feature = "planar_quant")]
pub mod planar_quant;
pub mod plot;
#[cfg(feature = "precision_aware_draft")]
pub mod precision_aware_draft;
#[cfg(feature = "progressive_mcgs")]
#[doc(alias = "mcts")]
#[doc(alias = "mcgs")]
#[doc(alias = "graph_search")]
#[doc(alias = "monte_carlo")]
pub mod progressive_mcgs;
#[cfg(feature = "proof_cert")]
pub mod proof_cert;
pub mod pruners;
// DenseMesh — latent node network for modelless inference (Plan 266, Research 234).
#[cfg(feature = "dense_mesh")]
pub mod dense_mesh;
#[cfg(feature = "rat_plus_bridge")]
pub mod rat_bridge;
#[cfg(feature = "maxsim")]
pub mod rerank;
#[cfg(feature = "river_valley")]
pub mod river_valley;
#[cfg(feature = "rt_turbo")]
pub mod rt_turbo;
#[cfg(feature = "ruliology")]
pub mod ruliology;
#[cfg(feature = "segment_checkpoint")]
pub mod segment_checkpoint;
#[cfg(feature = "shard_kv")]
pub mod shard_kv;
pub mod simd;
#[cfg(feature = "chiaroscuro")]
pub mod chiaroscuro;
// Functional Attention composition layer — Plan 286 Phase 5 (T5.1–T5.3). Each
// submodule is independently feature-gated; the module root compiles when any
// of the three composition features is on.
#[cfg(any(
    feature = "funcattn_spectral_pre_rotate",
    feature = "funcattn_chiar_blend",
    feature = "funcattn_freeze_thaw"
))]
pub mod funcattn_compose;
#[cfg(feature = "specialist_projection")]
pub mod specialist_projection;
#[cfg(feature = "sparse_task_vector")]
pub mod sparse_task_vector;
#[cfg(feature = "sparse_task_vector")]
pub mod sparse_compose;
#[cfg(feature = "skill_opt")]
pub mod skill_opt;
#[cfg(feature = "sleep_consolidation")]
pub mod sleep;
#[cfg(feature = "sp_kv")]
pub mod sp_kv;
pub mod spec_reconciliation;
#[cfg(feature = "spechop")]
pub mod spechop;
#[cfg(feature = "spectral_budget")]
pub mod spectral_budget;
#[cfg(feature = "spectral_rank")]
pub mod spectral_concentration;
#[cfg(feature = "spectral_quant")]
pub mod spectralquant;
pub mod speculative;
// SwiR Switch-Thinking — Explicit↔Latent mode controller (Plan 275, Research 241).
#[cfg(feature = "swir_switch_thinking")]
pub mod swir;
#[cfg(feature = "static_cal_tables")]
pub mod static_cal;
#[cfg(feature = "stiff_anomaly")]
pub mod stiff_anomaly;
#[cfg(feature = "still_kv")]
pub mod still_kv;
#[cfg(feature = "targeted_precision")]
pub mod targeted_precision;
// thinking_cot — adaptive CoT framework (Plan 194). The feature is a
// meta-feature that pulls in the bandit/prune/probe machinery required by
// speculative::thinking_controller; the module itself owns the shared
// ThinkingStrategy trait (Plan 275 Phase 2).
#[cfg(feature = "thinking_cot")]
pub mod thinking_cot;
pub mod tokenizer;
pub mod transformer;
pub mod trigger_gate;
#[cfg(feature = "turboquant")]
pub mod turboquant;
pub mod types;
#[cfg(feature = "unit_distance")]
pub mod unit_distance;
pub mod weights;

// Plan 265 Phase 4: Adaptive CoT stopping criterion (depends on band_conditioner).
#[cfg(feature = "adaptive_cot_identifiability")]
pub mod adaptive_cot_stopper;

#[cfg(debug_assertions)]
pub mod alloc;

/// Debug-only global allocator that tracks allocation count and bytes.
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;

#[cfg(feature = "mux_demux")]
pub mod mux_demux;

#[cfg(feature = "mux_latent_context")]
pub mod mux_latent;

// Memory Soup LoRA Artifact Importer (Plan 253 T19 G5).
// Standalone MSP0 binary format parser — uses only std + blake3, no riir-gpu dep.
// Proves katgpt-rs can consume riir-gpu's exported Memory Soup artifacts.
#[cfg(feature = "memory_soup_lora")]
pub mod memory_soup_lora;

#[cfg(feature = "llmexec_guard")]
pub mod llmexec_guard;

#[cfg(feature = "validator")]
pub mod validator;

#[cfg(feature = "breakeven_routing")]
pub use breakeven::{BreakevenBandit, BreakevenStats, BreakevenTierPair, BreakevenTracker};

#[cfg(feature = "tf_loop")]
pub mod tf_loop;

// Closure-Expansion Instrument — runtime wiring (Plan 290 Phase 4 T4.2/T4.3).
// `closure_wire` decorates any ScreeningPruner (BanditPruner / AbsorbCompressLayer)
// with PTG recording; `closure_mining` runs motif mining + admission at
// sleep-cycle boundaries. Both are gated on `closure_instrument`; the
// AbsorbCompress auto-tracing impl in `closure_wire` additionally needs `bandit`.
#[cfg(feature = "closure_instrument")]
pub mod closure_wire;

#[cfg(feature = "closure_instrument")]
pub mod closure_mining;
