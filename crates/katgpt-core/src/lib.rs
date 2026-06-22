//! katgpt-core: Shared types and SIMD kernels for katgpt-rs and riir-engine.
//!
//! This crate contains the common core shared between the two projects:
//! - **types**: Config, Rng, math utilities, LoRA, DomainLatent
//! - **simd**: NEON/AVX2 accelerated linear algebra kernels
//! - **traits**: Shared traits for game AI and speculative decoding
//!
//! No feature flags on types — both projects get the full superset.

#[cfg(feature = "tiled_attention")]
pub mod attention;
#[cfg(feature = "coda_fusion")]
pub mod coda;
#[cfg(feature = "dec_operators")]
pub mod dec;
pub mod leaky_core;
#[cfg(feature = "parallax_attn")]
pub mod parallax_attn;
pub mod shard_embedding;
pub mod simd;
pub mod traits;
pub mod types;

// CGSP — Curiosity-Guided Self-Play modelless triad (Plan 274, Research 240).
// Self-contained: Direction/Target/Candidate, CgspLoop, PoolConjecturer,
// HlaProjectionGuide, BreakevenDifficultyFilter, ColinearityBatchGate,
// EntropyCollapse, CuriosityPrioritySnapshot (BLAKE3-committed).
// Consumed by riir-engine Plan 299 (NPC curiosity runtime).
#[cfg(feature = "cgsp")]
pub mod cgsp;
#[cfg(feature = "cgsp")]
pub use cgsp::{
    BatchQualityGate, BreakevenDifficultyFilter, Candidate, CgspConfig, CgspLoop,
    ColinearityBatchGate, CollapseSignal, ComplexityWeights, CuriosityConjecturer,
    CuriosityPrioritySnapshot, CycleResult, CycleStats, DEFAULT_HLA_DIM, DEFAULT_K,
    DEFAULT_POOL_SIZE, DifficultyFilter, Direction, EntropyCollapse, HintDeltaBandit,
    HlaProjectionGuide, NoOpBatchGate, NoOpDifficultyFilter, PoolConjecturer, Priority,
    QualityGuide, ScratchBuffers, SolveRate, Solver, Target, entropy_nats, sigmoid,
    structural_complexity,
};

// CGSP dual-pool extension — DecentMem distillation (Plan 282, Research 249).
#[cfg(feature = "cgsp_dual_pool")]
pub use cgsp::{DualPoolBandit, DualPoolConfig, PoolId, ReachableDualPoolRouter};

// ActionBridge — generic latent→raw action bridge (Plan 262).
#[cfg(feature = "action_bridge")]
pub mod bridge;
#[cfg(feature = "action_bridge")]
pub use bridge::ActionBridge;

// Re-export consolidated traits (Plan 107 Phase 0)
pub use traits::{
    ActionSpaceLog, BestBuddyAligner, BinaryScreeningPruner, ConstraintPruner, DominoPruner,
    FeatureClass, GameState, NoPruner, NoScreeningPruner, RandomRolloutPolicy, RolloutPolicy,
    ScreeningPruner, StateHeuristic, best_buddies, pearson_correlation,
};
pub use traits::{GenerativeConstraintPruner, SpeculativeGenerator};

// RecursionLogits — opt-in trait for generators that expose pre/post recursion
// logits so AdvantageMarginGate can wrap them (Plan 283 T2.3, arxiv:2511.16886).
// Opt-in: not in default feature list. Non-recursing generators do not implement it.
#[cfg(feature = "recursion_logits")]
pub use traits::RecursionLogits;

// Q-Guided Flow (Plan 268) — test-time Q-gradient guidance primitive.
#[cfg(feature = "qgf_oracle")]
pub use traits::{NoGuidanceOracle, QGradientOracle};
#[cfg(feature = "qgf")]
pub mod qgf;

// MicroRecurrentBeliefState — per-entity recurrent state kernel (Plan 276, Research 242).
// Trait + Family A (attractor) + Family C (leaky) + BLAKE3 snapshot + sigmoid bridge.
// Opt-in until G1.1–G1.5 GOAT gate passes.
#[cfg(feature = "micro_belief")]
pub mod micro_belief;
#[cfg(feature = "micro_belief")]
pub use micro_belief::{
    AttractorKernel, KernelConfig, LeakyIntegrator, MicroRecurrentBeliefState,
    MicroRecurrentKernelSnapshot, RecurrenceFamily, SNAPSHOT_VERSION, project_to_scalars,
};

// BoMSampler — K-hypothesis single-pass belief sampling (Plan 281, Research 248).
// Opt-in extension of MicroRecurrentBeliefState; gated on bom_sampling which implies micro_belief.
#[cfg(feature = "bom_sampling")]
pub use micro_belief::{BoMSampler, NoiseQueryConfig, SeedStrategy, dot_product_scorer};

// BoM G2 arena harness — Plan 281 T2.3.
// Engine-side traits + synthetic reference env. riir-ai implements the traits
// over a real bomber/go sim to produce the empirical G2 gate.
#[cfg(feature = "bom_sampling")]
pub use micro_belief::{
    ArenaAction, ArenaEnvironment, BeliefPlanner, BoMMeanPlanner, BoMMinimaxPlanner,
    ComparisonResult, DeterministicPlanner, EnvHint, PlannerOutcome, SyntheticThreatArena,
    bom_mean_attractor, bom_minimax_attractor, bom_minimax_leaky, run_arena_comparison,
};

// FaithfulnessProbe — causal intervention diagnostic for injected memory (Plan 278, Research 244).
// Moved from katgpt root to katgpt-core so riir-engine (Plan 308) can consume via katgpt-core.
// Two features:
// - `triggered_injection` (default-ON after GOAT G3): sigmoid-thresholded inject/skip hot-path gate.
// - `faithfulness_probe` (opt-in, audit cadence): full intervention suite + perturbation + attribution.
// The module is compiled when EITHER feature is on; submodules are individually gated in `mod.rs`.
#[cfg(any(feature = "faithfulness_probe", feature = "triggered_injection"))]
pub mod faithfulness;

// ReviewMetrics — inference-time path-consistency / reward-hacking counter (Plan 054, Research 244).
// Moved from katgpt root to katgpt-core so riir-engine (Plan 308 Phase 3) can consume via
// katgpt-core/review_metrics. Whole `pruners` module is feature-gated, matching the
// faithfulness/cgsp precedent.
#[cfg(feature = "review_metrics")]
pub mod pruners;

// Temporal Derivative Kernel — dual fast/slow EMA surprise signal (Plan 277, Research 243).
// Turns any streaming latent vector into a signed "surprise" signal — the implicit
// prediction-error channel for credit assignment, computed locally with no backprop.
// Opt-in until ≥2 fusion gates (G2–G5) pass.
#[cfg(feature = "temporal_deriv")]
pub mod temporal_deriv;
#[cfg(feature = "temporal_deriv")]
pub use temporal_deriv::{TemporalDerivativeKernel, sigmoid_surprise_gate};

#[cfg(feature = "dual_leo")]
pub use traits::{
    ActingMode, AlphaSchedule, AutocurriculumSampler, BcConfig, BcTarget, DualLeoMixer,
};
#[cfg(feature = "leo_all_goals")]
pub use traits::{AllGoalsUpdate, LeoHead, sigmoid_bounded_q};

// Re-export key types at crate root for convenience
pub use shard_embedding::{EMBED_DIM, JlProjectionMatrix, STYLE_DIM as JL_STYLE_DIM};
pub use types::{
    AttentionMode, AttentionProjection, CacheLayout, Config, ConvergenceSelector, DashAttnConfig,
    DilationConfig, HlaMode, HybridPattern, InferenceOverrides, InferenceResult, LoopMode,
    LoraAdapter, LoraPair, ModelArchitecture, ResidualGate, RetrievalHeadRole, Rng, RtTurboConfig,
    SdpaOutputGate, ShardEmbedding, WeightDtype, kv_dim, lora_apply, matmul, matmul_f16,
    matmul_f16_parallel, matmul_parallel, matmul_relu, rmsnorm, sample_token, sample_token_into,
    softmax, softmax_scaled,
};

#[cfg(feature = "domain_latent")]
pub use types::DomainLatent;

#[cfg(feature = "sr2am_configurator")]
pub use types::{ConfiguratorContext, PlanningDecision};

#[cfg(feature = "data_gate")]
pub use types::{DataGate, GateDecision, ProposerTask, TaskType};

#[cfg(feature = "sparse_mlp")]
pub use types::sparse_matmul;

#[cfg(feature = "coda_fusion")]
pub use coda::{
    GateActivation, MoaConfig, compute_rstd, simd_matmul_residual,
    simd_matmul_residual_partial_rms, simd_matmul_rmsnorm_activation, simd_matmul_rmsnorm_rope,
    simd_matmul_rmsnorm_swiglu,
};

#[cfg(all(feature = "coda_fusion", feature = "moa_inference"))]
pub use coda::{MoaActivation, moa_swiglu, simd_matmul_rmsnorm_moa_swiglu};

#[cfg(feature = "tiled_attention")]
pub use attention::{
    tiled_attention_batched, tiled_attention_forward, tiled_attention_forward_with_scores,
};

#[cfg(feature = "parallax_attn")]
pub use parallax_attn::{
    ParallaxActivation, ParallaxConfig, ParallaxScratch, compute_rho, parallax_correction,
    tiled_attention_parallax_forward, tiled_attention_parallax_forward_retaining,
};

// Sink-aware composition (Plan 289). Requires both parallax_attn (for the
// forward) and sink_aware_attn (for the classifier + flat gate). The
// `tiled_attention_parallax_forward_sink_aware` entry point short-circuits to
// vanilla parallax when policy = Uniform, so this is a zero-cost abstraction
// for callers who construct the scratch but never enable DualPolicy.
#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]
pub use parallax_attn::{SinkAwareParallaxScratch, tiled_attention_parallax_forward_sink_aware};

pub use simd::SimdLevel;

#[cfg(feature = "hydra_budget")]
pub use types::{HydraBudgetConfig, HydraLayerProfile};

#[cfg(feature = "collapse_aware_thinking")]
pub use types::ThinkingBudget;

#[cfg(feature = "questbench")]
pub mod questbench;
#[cfg(feature = "questbench")]
pub use questbench::{
    CspDomain, MemoryTier, QuestBenchDecision, SyntheticCsp, UnderspecConfig, find_sufficient_set,
    generate_synthetic_csps, tier_from_score, underspecification_score,
};

#[cfg(feature = "tf_loop")]
pub use types::{CacheStrategy, IterationMode, SubStepStrategy, TrainingFreeLoopConfig};

#[cfg(feature = "plasma_path")]
pub use simd::{simd_ternary_matmul_batch, simd_ternary_matvec, ternary_matvec_scalar};
#[cfg(feature = "plasma_path")]
pub use types::TernaryWeights;

#[cfg(feature = "peira_distill")]
pub mod peira;
#[cfg(feature = "peira_distill")]
pub use peira::{PeiraConfig, PeiraCovariance, peira_aux_loss};

#[cfg(feature = "dirichlet_energy")]
pub mod dirichlet;
#[cfg(feature = "dirichlet_energy")]
pub use dirichlet::{
    consecutive_adjacency, dirichlet_energy, functor_adjacency, kv_cache_dirichlet_energy,
};

#[cfg(feature = "spectral_hierarchy")]
pub mod spectral_hierarchy;
#[cfg(feature = "spectral_hierarchy")]
pub use spectral_hierarchy::{cauchy_interlacing_check, eigenspace_alignment, haar_wavelet_basis};

#[cfg(feature = "sigmoid_margin")]
pub use simd::{compute_retrieval_margin, dim_sufficiency_bound, sigmoid_margin_loss};

#[cfg(feature = "dual_gram_pca")]
pub use simd::simd_gram_f32;

#[cfg(feature = "roofline_cost")]
pub mod roofline;
#[cfg(feature = "roofline_cost")]
pub use roofline::{
    ComputeBound, Dtype, HardwarePeaks, OpType, RooflineCost, gemm_cost, gemv_cost, gram_cost,
    roofline_estimate,
};

#[cfg(feature = "and_or_dtree")]
pub mod and_or;
#[cfg(feature = "and_or_dtree")]
pub use and_or::AndOrNode;

#[cfg(feature = "partial_scoring")]
pub use traits::{GameTrace, PartialScorer};

#[cfg(feature = "problem_mutator")]
pub use traits::{GameConfig, MutantConfig, MutationKind, ProblemMutator};

#[cfg(feature = "modal_spec")]
pub mod linoss;
#[cfg(feature = "mux_pruner")]
pub mod mux;

#[cfg(feature = "sense_composition")]
pub mod sense;

#[cfg(feature = "slod")]
pub mod slod;
#[cfg(feature = "slod")]
pub use slod::{
    ScaleBoundary, SlodConfig, SlodOperator, SlodPruner, exp_map, frechet_mean,
    heat_kernel_weights, log_map, poincare_distance,
};

// Spectral Irrep Pruner - spectral flatness-based speculative decoding pruning (Plan 246).
// Prunes tokens when logit spectrum shows competing modes (high spectral flatness).
// GOAT PASS: +3.6% overhead, default-ON.
#[cfg(feature = "spectral_pruner")]
pub mod irrep_pruner;

// Subspace phase-gate primitives — participation ratio, numerical rank, N≥d
// phase-transition gate (Wang et al. Thm 4, arXiv:2409.02426), and runtime
// Jacobian SVD via forward differences (Plan 301, Research 279). Pure numeric,
// no game/shard/chain semantics. Consumers (riir-neuron-db Plan 002, future
// riir-ai HLA self-discovery plan) apply these to their own maps.
// Opt-in until G1 GOAT gate passes.
#[cfg(feature = "subspace_phase_gate")]
pub mod subspace_phase_gate;
#[cfg(feature = "spectral_pruner")]
pub use irrep_pruner::{
    IrrepPruner, IrrepPrunerConfig, irrep_pruner_from_config, spectral_flatness,
};
#[cfg(feature = "subspace_phase_gate")]
pub use subspace_phase_gate::{
    IntrinsicDimMethod, JacobianSvdScratch, SvdResult, estimate_intrinsic_dim, jacobian_svd_at,
    numerical_rank, participation_ratio, phase_transition_gate,
};

#[cfg(feature = "flow_field_nav")]
pub mod flow;
#[cfg(feature = "flow_field_nav")]
pub use flow::{
    FlowField, FlowFieldCache, FlowFieldConfig, LeoPotentialGrid, blend_steering, fft_smooth,
    fft_smooth_into, flow_steering, inflate_obstacles, should_use_flow_field,
};

// Merkle octree — hierarchical BLAKE3 commitment for KG latent octree nodes (Plan 221-M).
#[cfg(feature = "merkle_octree")]
pub mod merkle;
#[cfg(feature = "merkle_octree")]
pub use merkle::{
    HASH_SIZE, MERKLE_OCTREE_DEPTH, MERKLE_OCTREE_INTERNAL, MERKLE_OCTREE_LEAVES,
    MERKLE_OCTREE_NODES, MerkleOctree, MerkleProof,
};

// Curator verification layer for Merkle octree (Plan 253).
#[cfg(feature = "merkle_octree")]
pub mod curator;
#[cfg(feature = "merkle_octree")]
pub use curator::{
    CuratorArm, CuratorBandit, CuratorVerdict, FrozenTarget, MerkleEnvelope, MerkleFrozenStore,
    verification_weight,
};

// RTDC — Resolution-Tiered Deterministic Commitment (Plan 302, Research 280).
// Wraps `MerkleOctree` with 3 per-depth roots aligned to SLoD σ-boundaries,
// enabling trust-minimized semantic zoom: a light client verifies its
// fog-of-war view is a faithful sub-summation of the chain-committed full KG,
// with O(log n) proof at the abstraction level it operates at.
//
// Phase 1 ships the open primitive (types + trait + depth-2 sound proofs).
// Cross-depth soundness (`subtree_inclusion`) is Phase 3 (Candidate C —
// probabilistic sampling) behind `rtdc_subtree_inclusion` — see
// `riir-chain/issues/002_rtdc_subtree_inclusion_research.md`.
// LatCal-backed `DeterministicLeafEncode` impl lives in riir-chain (Plan 003).
#[cfg(feature = "rtdc")]
pub mod rtdc;
#[cfg(feature = "rtdc")]
pub use rtdc::{
    DepthSelector, DepthTieredMerkleOctree, DepthTieredRoots, DeterministicLeafEncode, RtdcError,
    RtdcProof,
};
#[cfg(feature = "rtdc_subtree_inclusion")]
pub use rtdc::{RTDC_SUBTREE_DEFAULT_K, SubtreeProof, min_k_for_95pct_confidence};

// GPart isometric partition adapter — replaces LoRA's bilinear BA with single isometric Pθ_d (Plan 257).
#[cfg(feature = "gpart_adapter")]
pub use types::{GPART_MAGIC, GPART_VERSION, GpartAdapter, GpartPair, GpartPrepared};

#[cfg(feature = "dendritic_gate")]
pub mod dendritic_gate;
#[cfg(feature = "dendritic_gate")]
pub use dendritic_gate::{DendriticGate, dendritic_sigmoid};
#[cfg(feature = "dendritic_gate")]
pub use simd::{coincidence_score, entropy_f32};

// CompressionDrafter — Hot-tier modelless LZ4 corpus-as-model drafter (Plan 285,
// Research 256, nathan.rs/gzip-lm). The compressor IS the model: score candidate
// continuations by compressed length against a frozen corpus. Corpus is appendable
// for online learning and is itself the wired format (bytes + BLAKE3).
// Opt-in until G1–G3 GOAT gate passes.
#[cfg(feature = "compression_drafter")]
pub mod compression_drafter;
#[cfg(feature = "compression_drafter")]
pub use compression_drafter::{CompressionDrafter, Lz4FlexDrafter};

// Functional Attention — closed-form Tikhonov spectral transport operator
// (Plan 286, Research 257, arxiv 2605.31559, Xiao et al. ICML 2026). DUAL FORM
// matching the reference implementation (`.raw/FUNCATTN/PDE-StandardBenchmark/model/
// Functional_attention.py`): convex-combo regularization `(1-α)·K̃ᵀK̃ + α·I_d`,
// column-normalized slice tokens, per-slice-token to_q/to_k/to_v linear
// projections. Sigmoid-basis default per AGENTS.md (partition-of-unity holds
// for any row-normalized non-negative kernel). Gain-tier open primitive:
// paper itself defers NLP validation (§6); promote only after G1–G5 GOAT
// gate passes.
#[cfg(feature = "funcattn")]
pub mod funcattn;
#[cfg(feature = "funcattn")]
pub use funcattn::{
    FuncAttnBasis, FuncAttnConfig, FuncAttnError, FuncAttnScratch, compute_basis_into,
    funcattn_forward, pre_rotate_basis_weights_into, solve_convex_combo_dual,
};

// ChunkedContentStore — Lore-distilled chunked content-addressed Merkle store (Plan 272, Research 262).
// Open primitive: chunks → BLAKE3 → dedup via papaya → binary Merkle root. No game/chain IP.
// Consumed by riir-ai Plan 319 (Executable Asset Vessel + Quorum Gitflow).
//
// NOTE: the binary-Merkle `MerkleProof` here is renamed on re-export to
// `BinaryMerkleProof` to avoid colliding with `merkle_octree::MerkleProof`
// when both features are active simultaneously (caught by `cargo check
// --all-features`). Internal callers still reach the type via
// `crate::content_store::MerkleProof`.
#[cfg(feature = "chunked_content_store")]
pub mod content_store;
#[cfg(feature = "chunked_content_store")]
pub use content_store::{
    BlobId, ChunkFetcher, ChunkRange, ChunkedContentStore, ChunkerConfig, ChunkingStrategy,
    FastCdcChunker, FixedSizeChunker, InMemoryChunkedStore, MerkleProof as BinaryMerkleProof,
    StoreStats, build_binary_merkle_proof, build_binary_merkle_root, verify_binary_merkle_proof,
};

// Closure-Expansion Instrument (CEI) — PTG recorder + motif miner + PRI/CDG/TaR metrics
// (Plan 290, Research 264, arxiv 2606.15386, Momennejad & Raileanu). Open measurement
// layer: turns open-ended inference into observable metrics. Opt-in until G1–G4 GOAT
// gate passes; G5 demotes to opt-in diagnostic if metrics don't correlate with quality.
#[cfg(feature = "closure_instrument")]
pub mod closure;
#[cfg(feature = "closure_instrument")]
pub use closure::{
    OperatorKind, PrimitiveKind, PrimitiveTransitionGraph, PtgEdge, PtgNode,
    admit::{GateResult, MotifAdmitter, RejectionReason},
    bridge::{
        DEFAULT_MOTIF_DIRS, MotifDirections, motif_embedding_to_tar_score, ptg_to_motif_embedding,
    },
    commitment, deserialize_postcard,
    metrics::{CdgScore, PriScores, compute_cdg, compute_pri, compute_tar_score, motif_multiset},
    motif::{
        FixedU32Set, MAX_MOTIF_EDGES, MAX_MOTIF_NODES, Motif, MotifMiner, RING_BUFFER_K,
        enumerate_subgraph_hashes,
    },
    serialize_postcard,
    trace::{DEFAULT_TRACE_CAPACITY, NodeId, PtgRecorder},
};

// Sink-Aware Attention — NOP/Broadcast classifier + dual-policy sigmoid gate
// (Plan 287, Research 258, arxiv 2606.08105, Fesser et al.). Per-head
// classifier (value-norm-ratio + stable-rank-of-update) decides whether a
// sink is Adaptive NOP (gate it via sigmoid) or Broadcast (preserve it).
// Staged integration: the policy enum + standalone apply_dual_policy_gate
// ship here; direct wiring into parallax_attn / funcattn forward paths is
// deferred until synthetic G2 + latency G3 gates pass on a real model
// (validation fallback per Plan 287 §Validation).
#[cfg(feature = "sink_aware_attn")]
pub mod data_probe;
#[cfg(feature = "sink_aware_attn")]
pub use data_probe::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};

// ICT Distributional Branching-Point Detector — open generic math (Plan 294,
// Research 270, arxiv 2606.19771). Collision purity β(π) = Σ π² (proven
// unconditionally monotone, ICT §A.2.5 — H₁ is wrong below π > e⁻¹ ≈ 0.37),
// Rényi H₂, Jensen-Shannon divergence to group mean, BranchingDetector
// (top-k% selector over K candidate trajectories + per-step β EMA), and the
// Bebop H₁→H₂ acceptance-forecast upgrade. No game semantics, no chain;
// runtime fusion (CLR gating, HLA updates, KG emission) is riir-ai Plan 324.
// Opt-in until G3 (Spearman ρ(H₁, JS-uniqueness) < 0.5) AND G8 (riir-ai
// Plan 324 runtime validation) pass.
#[cfg(feature = "ict_branching")]
pub mod ict;
#[cfg(feature = "ict_branching")]
pub use ict::{
    AcceptanceForecastH2, BranchingDetector, BranchingReport, branching_point_mask,
    branching_point_mask_into, collision_purity, collision_purity_into, is_critical_branching,
    js_divergence, js_divergence_batch, renyi_h2, shannon_h1,
};

// ── Induced Code World Model (Plan 296, Research 275, arxiv 2510.04542) ───────
//
// Open half of the CWM Super-GOAT: a marker trait over `GameState` for forward
// models that are verifiable, BLAKE3-committable, and hot-swappable. The
// LLM-induction pipeline that *produces* an `InducedCwmKernel` impl is private
// (riir-ai Plan 326). The runtime never sees the LLM — only the frozen kernel.
#[cfg(feature = "induced_cwm")]
pub mod induced_cwm;
#[cfg(feature = "induced_cwm")]
pub use induced_cwm::{
    BeliefInferenceFn, CwmCommitment, InducedCwmKernel, TransitionTestFailure, TransitionUnitTest,
    make_transition_tests_from_trajectory, verify_transition,
};

// Phase 2 (Plan 296 T2.1–T2.5): Information-Set MCTS over an induced CWM +
// belief fn. Self-contained search tree (does NOT reuse root-crate
// `mcts_search` — that lives in katgpt-rs/src, katgpt-core cannot depend on the
// root). Gated by `induced_cwm_ismcts` (which auto-enables
// `induced_cwm`).
#[cfg(feature = "induced_cwm_ismcts")]
pub use induced_cwm::{InformationSet, NodeStats, ismcts_search_with_inference};

// ── Personality-Weighted Layer Composition (Plan 297, Research 276) ───────
//
// Open MIT-licensed primitive for the Entity Cognition Stack Super-GOAT.
// A `PersonalityWeightedComposition<N, D>` kernel composes `N` latent
// direction vectors via per-layer sigmoid-gated weights, then drifts those
// weights via a reward-surprise Hebbian update. Zero-allocation, sigmoid-gated
// (NOT softmax — per AGENTS.md), belief-gated, BLAKE3-snapshot-integrated.
// Entity-agnostic (NPC, player, predator, prey, robot, recommender user).
//
// Consumed by riir-ai Plan 327 (runtime wiring) — the game-specific 7-layer
// mapping, archetype table, taming transition stay private in riir-ai.
// Opt-in until G4 (<1µs/entity) + G5 (zero alloc) GOAT gate passes.
#[cfg(feature = "personality_composition")]
pub mod personality_composition;
#[cfg(feature = "personality_composition")]
pub use personality_composition::{
    ArchetypeLabel, EntityCognitionComposition, LayerDirectionSource, PersonalityConfig,
    PersonalitySnapshot, PersonalityWeightedComposition, sigmoid as personality_sigmoid,
    sigmoid_into as personality_sigmoid_into,
};

// ── Engram — Hash-Addressed Pattern Memory (Plan 299, Research 278) ───────
//
// Open MIT-licensed primitive: the first conditional-MEMORY axis in the
// katgpt stack (complementary to Raven's conditional-COMPUTATION axis).
// N-gram-suffix → multi-head hash → O(1) slot lookup → sigmoid gate (RMSNorm
// dot σ) → residual-fuse into hidden state. Frozen table, atomic swaps for
// updates, BLAKE3 commitment as sync-boundary audit artifact.
//
// CRITICAL: sigmoid, not softmax — per AGENTS.md. No `softmax` symbol here.
//
// Open half of the Engram Super-GOAT: private selling-point guide lives in
// riir-ai Guide 147; chain commitment bridge is riir-chain R001 (TODO).
// Opt-in until G1–G7 GOAT gate passes.
#[cfg(feature = "engram")]
pub mod engram;
#[cfg(feature = "engram")]
pub use engram::{
    CacheResult, CacheTier, ColdFetcher, EngramConfig, EngramHash, EngramHotSwap, EngramTable,
    EngramTableBuilder, EngramTableId, HashHead, IDENTITY_KERNEL, InMemoryEngramTable, K_MAX,
    SigmoidFusionConfig, SurjectiveMap, SurjectiveMapLoadError, TokenId, TokenizerSpec,
    ZipfianCacheHierarchy, ZipfianStats, ZipfianStatsSnapshot, build_merkle_root,
    build_surjective_map, compress_token, conv_causal_into, fuse_into_hidden_state,
    multi_head_hash, rmsnorm_into, sigmoid_fuse_into, sigmoid_fuse_multi_branch_into,
    try_compress_token,
};
