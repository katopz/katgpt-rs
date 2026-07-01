#![allow(unexpected_cfgs)]
#[cfg(all(target_os = "macos", feature = "ane"))]
pub mod ane_backend;
// Issue 359: `attn_match` extracted to the katgpt-attn-match leaf. The root
// re-exports the leaf as `attn_match` so all historical `katgpt_rs::attn_match::*`
// paths continue to resolve (Issue 014/015 re-export contract). The
// `adaptive_cot` glue stays in root (composes root-only `freq_bandit`).
#[cfg(feature = "attn_match")]
pub use katgpt_attn_match as attn_match;
/// Adaptive CoT compaction glue — composes the leaf's online compactor with the
/// root-only `freq_bandit` bandit threshold tuner. Stays in root per Issue 359
/// (freq_bandit depends on root-only `trigger_gate`).
#[cfg(feature = "adaptive_cot_compaction")]
pub mod attn_match_adaptive_cot;
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

// Claim Rubric Runtime — L1/L2/L3 evidence ladder validator (Plan 307,
// Research 287, arxiv 2606.07612). Generic meta-discipline that grades
// probe/steering claims by evidence level: L1 (Behavioral) / L2
// (Functional) / L3 (Causal-mechanistic). Vocabulary must match evidence
// — "causally controls" requires L3 evidence; "reads" is L1-safe. Opt-in
// until Phase 2 round-trip tests pass on R287 §4 scores.
#[cfg(feature = "claim_rubric")]
pub mod claim_rubric;
#[cfg(feature = "claim_rubric")]
pub use claim_rubric::{
    ChecklistSection, ClaimValidator, EvidenceItem, EvidenceItemId, EvidenceLevel, Grade,
    VocabularyViolation,
};
pub mod cumprodsum;
// CUCG — Closed-Unit Compaction Gate (Plan 333, Research 300, arxiv 2606.23525).
// Generic rubric-gated trajectory compaction primitive. DEFAULT-ON since
// Phase 6 (2026-06-25): 7/7 GOAT gates PASS. Re-exports the public surface
// for ergonomic use.
#[cfg(feature = "closed_unit_compaction")]
pub mod compaction;
#[cfg(feature = "closed_unit_compaction")]
pub use compaction::{
    Backstop, ClosedUnitCompactionGate, ClosedUnitCompactionGateBuilder, CombineOp,
    CompactionAuditRecord, CompactionDecision, DecisionKind, FireRule, FireRuleEval, PredicateAudit,
    PredicateReason, PredicateResult, Rubric, RubricScratch, RubricVerdict,
};
#[cfg(feature = "ssd_block")]
pub mod ssd_block;
#[cfg(feature = "dash_attn")]
pub mod dash_attn;
#[cfg(feature = "data_probe")]
pub mod data_probe;
// Issue 007 Phase C: `npc_ane_backend` and `npc_brain_router` moved to
// riir-engine (NPC runtime IP). They depended on `katgpt_core::sense::backend`
// which moved, and are themselves gameplay-runtime IP per the 5-repo strategy.
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
// `feedback` module exiled to `katgpt-deprecated` (Phase 3a, Proposal 003).
// Re-export preserved for back-compat: `katgpt_rs::feedback::*` still resolves.
#[cfg(feature = "feedback")]
pub use katgpt_deprecated::feedback;
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
pub use katgpt_core::newton_schulz;  // Extracted to katgpt-core per Issue 355 Phase 1a; re-export preserves historical `katgpt_rs::newton_schulz::*` paths.
#[cfg(feature = "off_principal_retrieval")]
pub mod off_principal;
#[cfg(feature = "hla_eigenbasis_recovery")]
pub mod hla_eigenbasis;  // Issue 001: per-NPC eigenbasis recovery from windowed HLA activations
#[cfg(feature = "octopus")]
pub mod octopus;
#[cfg(feature = "modality_pruned_load")]
pub mod pipeline_pruner;
#[cfg(feature = "planar_quant")]
pub mod planar_quant;
#[cfg(feature = "plot")]
pub mod plot;  // Issue 355 Phase 2a: gated behind `plot` feature (plotters is now optional). DEFAULT-ON.
// Orthogonal Procrustes — cross-frame embedding alignment via polar
// decomposition (Newton-Schulz on B^T A). Issue 001 (katgpt-rs). GOAT
// candidate — gated behind `orthogonal_procrustes` until benchmark gates
// G1–G6 (Issue 001) pass. Promotes to default-on if GOAT, demoted if not.
#[cfg(feature = "orthogonal_procrustes")]
pub mod procrustes;
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
#[cfg(feature = "kv_share")]
pub use katgpt_kv::kv_share;
#[cfg(feature = "osc_kv")]
pub use katgpt_kv::osc_kv;
#[cfg(feature = "cs_kv_probe")]
pub use katgpt_kv::cs_kv_probe;
#[cfg(feature = "shard_kv")]
pub use katgpt_kv::shard_kv;
#[cfg(feature = "still_kv")]
pub use katgpt_kv::still_kv;
#[cfg(feature = "kvarn")]
pub use katgpt_kv::kvarn;
#[cfg(feature = "targeted_precision")]
pub use katgpt_kv::targeted_precision;
#[cfg(feature = "sp_kv")]
pub mod sp_kv {
    //! SP-KV re-export bridge (Issue 015 Phase 5).
    //!
    //! Combines `katgpt_kv::sp_kv` (types + utility predictor) with the local
    //! `sp_kv_forward_mod` (transformer glue that depends on ForwardContext).
    pub use katgpt_kv::sp_kv::*;
    pub mod forward {
        pub use crate::sp_kv_forward_mod::*;
    }
    // Re-export the forward-module building blocks at the sp_kv top level so
    // historical `katgpt_rs::sp_kv::{GateBias, attention_head_core, ...}` paths
    // still resolve (back-compat with tests/examples).
    pub use crate::sp_kv_forward_mod::{
        GateBias, NoBias, SpKvForwardContext, attention_head_core, attention_head_gated,
        forward_sp_kv, forward_sp_kv_quant,
    };
}
/// SP-KV transformer glue — full pipeline functions that take ForwardContext.
/// Kept private; surfaced via the `sp_kv::forward` re-export above.
#[cfg(feature = "sp_kv")]
mod sp_kv_forward_mod;
pub mod spec_reconciliation;
#[cfg(feature = "spechop")]
pub mod spechop;
#[cfg(feature = "spectral_budget")]
pub mod spectral_budget;
#[cfg(feature = "spectral_rank")]
pub mod spectral_concentration;
#[cfg(feature = "spectral_quant")]
pub mod spectralquant {
    //! Spectralquant re-export shim (Issue 015 Phase 5).
    //!
    //! The substrate physically lives in `crates/katgpt-spectral/`. This
    //! module re-exports it so all historical `katgpt_rs::spectralquant::*`
    //! paths continue to resolve for `funcattn_compose`, `chiaroscuro`,
    //! `benchmark/infrastructure`, and all tests/examples.
    pub use katgpt_spectral::*;
}
pub mod speculative;
// SwiR Switch-Thinking — Explicit↔Latent mode controller (Plan 275, Research 241).
#[cfg(feature = "swir_switch_thinking")]
pub mod swir;
#[cfg(feature = "static_cal_tables")]
pub mod static_cal;
#[cfg(feature = "stiff_anomaly")]
pub mod stiff_anomaly;
// thinking_cot — adaptive CoT framework (Plan 194). The feature is a
// meta-feature that pulls in the bandit/prune/probe machinery required by
// speculative::thinking_controller; the module itself owns the shared
// ThinkingStrategy trait (Plan 275 Phase 2).
#[cfg(feature = "thinking_cot")]
pub mod thinking_cot;
pub use katgpt_tokenizer as tokenizer;  // re-export (Issue 014): preserves `katgpt_rs::tokenizer::*` paths for tests/examples/validator
pub mod transformer;
pub mod trigger_gate;
#[cfg(feature = "turboquant")]
pub mod turboquant;
pub mod types;
#[cfg(feature = "unit_distance")]
pub use katgpt_deprecated::unit_distance;

// Plan 008 Step 2: weight-packing substrate now lives in `katgpt-transformer`.
// Historical `crate::weights::ContiguousWeights` / `load_ternary_bits` callers
// resolve through this re-export unchanged.
pub use katgpt_transformer::{ContiguousWeights, load_ternary_bits};

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

// Salience Tri-Gate Primitive — open 3-way per-tick emit gate (Speak / Silent /
// Delegate) distilled from JoyAI-VL-Interaction (Plan 303, Research 281,
// arxiv 2606.14777). Two stacked sigmoids (never softmax); silence is a
// first-class variant, not a threshold-suppression default; zero-allocation
// hot path; deterministic for replay/sync. Game-side NPC wiring lives in
// riir-ai Plan 330 — this crate stays math-only, MIT, no game IP.
// Opt-in until G1 (determinism + monotonicity) + G2 (two-sigmoid ablation
// parity) + <50ns decide() latency gates pass.
#[cfg(feature = "salience_tri_gate")]
pub mod salience;
#[cfg(feature = "salience_tri_gate")]
pub use salience::{
    DelegateToken, FoldbackTarget, SalienceDecision, SalienceTriGate, SilenceToken,
};

// Algorithmic-Probability Sampler + Coincidence Gate — two open primitives
// distilled from Dingle & Hutter 2026 (Plan 305, Research 284, Entropy
// 28(2):226). `CompressionPriorSampler<K>` replaces uniform candidate sampling
// in MCTS / bandits / DDTree / speculative drafters with a simplicity-biased
// prior (sigmoid per candidate, never softmax as the public API). Pluggable
// K̃: RLE ratio, Shannon entropy, L1 norm (LZ4 / BLAKE3 stubs gated behind
// sub-features). `CoincidenceGate` probes a found optimum against other simple
// objectives for theorem-backed cross-task transfer. riir-ai Plan 331 wires
// this to HLA / functor / shard vectors (private).
// Opt-in until G1 (sampler safety) + G2 (exponential speedup) gates pass.
#[cfg(feature = "complexity_prior_sampler")]
pub mod screening;
#[cfg(feature = "complexity_prior_sampler")]
pub use screening::{
    CoincidenceGate, CompressionPriorSampler, ComplexityProxy, EntropyComplexity, L1Complexity,
    LatentCompressionPriorSampler, RleComplexity, quantize_latent,
};

// Alien Sampler Primitive — Coherence × Availability Frontier Ranking
// (Plan 311, Research 293, arxiv 2603.01092, Artiles et al. "The Alien Space
// of Science" May 2026). Generic, modelless within-pool ranking: z-scored
// linear fusion `(1−β)·zC + β·zU` of a coherence score and an unavailability
// score. `MedianTopMAvailability` implements the paper's load-bearing
// median-of-top-m cosine rule. Open math only — NPC population banks, CGSP
// binding, and zone emission feeds live in riir-ai Plan 312+.
// Opt-in until Phase 3 GOAT gate (G1 motif collapse ≤50% of OPUS baseline,
// G2 quality ≥90% of coherence-only, G3 perf ≤5× baseline, G4 no Vec<f32>
// escapes rank()) passes.
#[cfg(feature = "alien_sampler")]
pub use katgpt_deprecated::alien_sampler;
#[cfg(feature = "alien_sampler")]
pub use katgpt_deprecated::alien_sampler::{
    AlienConfig, AlienSampler, AlienSamplerError, AvailabilityScorer, CoherenceScorer,
    MedianTopMAvailability, ScoredCandidate,
};

// Vessel — Extract-Once Secure Wire Format Primitive (Plan 315 / Research 297).
//
// MOVED to riir-neuron-db/src/vessel/ (Plan 006, 2026-06-29). The primitive
// is now PRIVATE — publishing the security-enforcement wire format (magic
// bytes, BLAKE3 protocol, projector signatures, fuel-gating budgets, exact
// 52-byte header layout) handed attackers a free threat-model blueprint with
// zero public adoption value. No external dev builds on "secure WASM vessel
// format"; the only consumers were the private repos (riir-neuron-db
// NeuronVesselSidecar, riir-chain delivery, riir-ai runtime). See
// riir-neuron-db/.plans/006_vessel_primitive_migration.md.
//
// The `secure_vessel` Cargo feature, the `vessel_extract_bench`, and the
// `vessel_minimal` / `vessel_project` examples were removed in the same
// migration. Historical docs remain in katgpt-rs/.docs, .benchmarks, .plans,
// .research as the public record of what existed.
