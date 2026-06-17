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
#[cfg(feature = "parallax_attn")]
pub mod parallax_attn;
pub mod leaky_core;
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
    CuriosityPrioritySnapshot, CycleResult, CycleStats, Direction, DifficultyFilter,
    EntropyCollapse, HlaProjectionGuide, HintDeltaBandit, NoOpBatchGate, NoOpDifficultyFilter,
    PoolConjecturer, Priority, QualityGuide, ScratchBuffers, Solver, SolveRate, Target,
    DEFAULT_HLA_DIM, DEFAULT_K, DEFAULT_POOL_SIZE, entropy_nats, sigmoid, structural_complexity,
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
    GameState, NoPruner, NoScreeningPruner, RandomRolloutPolicy, RolloutPolicy, ScreeningPruner,
    StateHeuristic, best_buddies, pearson_correlation,
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
    tiled_attention_parallax_forward,
};

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
#[cfg(feature = "spectral_pruner")]
pub use irrep_pruner::{
    IrrepPruner, IrrepPrunerConfig, irrep_pruner_from_config, spectral_flatness,
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
    CuratorArm, CuratorBandit, CuratorVerdict, CuratorVerifier, FrozenTarget, MerkleEnvelope,
    MerkleFrozenStore, verification_weight,
};

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
    funcattn_forward, solve_convex_combo_dual,
};
