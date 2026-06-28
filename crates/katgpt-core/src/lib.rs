//! katgpt-core: Shared types and SIMD kernels for katgpt-rs and riir-engine.
//!
//! This crate contains the common core shared between the two projects:
//! - **types**: Config, Rng, math utilities, LoRA, DomainLatent
//! - **simd**: NEON/AVX2 accelerated linear algebra kernels
//! - **hla**: Higher-order Linear Attention substrate (cache types + kernels)
//! - **mcts**: Generic Monte Carlo Tree Search over any `GameState`
//! - **delta_mem**: δ-mem associative memory substrate (state, hasher, multi-domain)
//! - **traits**: Shared traits for game AI and speculative decoding
//! - **speculative**: Speculative-decoding substrate types + sampling primitives
//!   (TreeNode, DraftResult, configs, LDT conflict detector, TES credit
//!   assignment, CDF/residual samplers)
//!
//! No feature flags on types/simd/hla/mcts/delta_mem/speculative — both projects
//! get the full substrate. Composition layers (root-only types like
//! `BanditRolloutPolicy`, `MemorySteeredPruner<P>`) stay in the consuming crate.

#[cfg(feature = "tiled_attention")]
pub mod attention;

// best_belief — ε-quantile Beta lower bound for conservative selection
// (Plan 336, Research 320, RQGM arXiv:2606.26294 Prop. 4). Complements
// `sample_beta` (Thompson sampling for EXPLORATION) with a conservative
// EXPLOITATION / SELECTION counterpart. Opt-in until the G1+G2+G4 GOAT gate
// passes.
#[cfg(feature = "best_belief")]
pub mod best_belief;
#[cfg(feature = "best_belief")]
pub use best_belief::{best_belief_score, best_belief_scores, select_best_belief};
#[cfg(feature = "coda_fusion")]
pub mod coda;
#[cfg(feature = "dec_operators")]
pub use katgpt_dec as dec;
pub mod delta_mem;
pub mod hla;
// Shared leaky-integrator primitive. Spun out to the `katgpt-types` leaf
// (Issue 007 Phase E Tier 1 #3) so both katgpt-core (`sense::reconstruction`)
// and `katgpt-micro-belief` (`LeakyIntegrator::step`) can consume it without
// a cycle. Re-exported here as `katgpt_core::leaky_core` for backwards
// compatibility.
pub use katgpt_types::leaky_core;
/// Generic Monte Carlo Tree Search over any [`crate::traits::GameState`].
///
/// Always-on substrate. Composition that needs root-only types
/// (`BanditRolloutPolicy` depends on `crate::pruners::bandit::BanditStats`)
/// stays in the consuming crate.
pub mod mcts;
#[cfg(feature = "parallax_attn")]
pub mod parallax_attn;
pub mod shard_embedding;
// SIMD-accelerated linear algebra kernels (NEON / AVX2 / WASM-SIMD128 /
// scalar fallback). Spun out to the `katgpt-types` crate (Issue 007 Phase E
// Tier 1 #2) and re-exported here as `katgpt_core::simd` for backwards
// compatibility. All `crate::simd::*` paths resolve unchanged.
pub use katgpt_types::simd;
pub mod speculative;
pub mod traits;
// Shared configuration, RNG, math utilities, LoRA, domain embeddings, and
// inference types. Spun out to the `katgpt-types` crate (Issue 007 Phase E
// Tier 1 #2) and re-exported here as `katgpt_core::types` for backwards
// compatibility. All `crate::types::*` paths resolve unchanged.
pub use katgpt_types as types;

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
// Spun out to the `katgpt-micro-belief` crate (Issue 007 Phase E Tier 1 #3) and
// re-exported here as `katgpt_core::micro_belief` for backwards compatibility.
#[cfg(feature = "micro_belief")]
pub use katgpt_micro_belief as micro_belief;
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

// Pruners module (Plan 054 review_metrics + Plan 320 indicator_probe_bank, etc.).
// Parent module is always compiled; individual submodules gate their own features.
// (Previously the whole `pruners` module was gated behind `review_metrics`; that
// coupling was broken in Plan 320 so indicator_probe_bank can gate independently.)
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

// Viable Manifold Graph — discrete safe-manifold navigation primitive.
// Distillation of arXiv:2206.00106 (González-Duque et al., *Mario Plays on a
// Manifold*, 2022). Generic over any smooth map `f: R^n → R^m` (closure) and
// a viability predicate `V(z)`. Computes the pullback volume field
// `log det(J_f^T J_f)` (via Plan 301's `jacobian_svd_at`), filters a latent
// sample to a discrete safe-manifold subgraph, and runs A* / random-walk
// navigation that stays inside the viable set by construction. Game / shard /
// chain wiring lives in riir-ai (R154). Opt-in until G1–G6 GOAT gates pass.
#[cfg(feature = "viable_manifold_graph")]
pub mod viable_manifold_graph;

// AC-GPT Arbitrary-Conditional Prefix — modelless mask builder + sequence
// augmenter that turns any causal Transformer forward into a single-pass
// arbitrary-conditional forward p(xe | xc) via position-aware copies of xc at
// the front and a [xc-bidirectional | causal-everywhere-else] attention mask
// (Lu et al., Mila, arXiv:2606.14943, Plan 313, Research 295). Phase 1 ships
// types + bit math only — no attention kernel dep, no SVD. Opt-in until G1–G4
// GOAT gates pass.
#[cfg(feature = "ac_prefix")]
pub mod ac_prefix;
#[cfg(feature = "spectral_pruner")]
pub use irrep_pruner::{
    IrrepPruner, IrrepPrunerConfig, irrep_pruner_from_config, spectral_flatness,
};
#[cfg(feature = "subspace_phase_gate")]
pub use subspace_phase_gate::{
    IntrinsicDimMethod, JacobianSvdScratch, SvdResult, SvdResultScratch, SvdScratch,
    estimate_intrinsic_dim, jacobian_svd_at, numerical_rank, participation_ratio,
    phase_transition_gate, thin_svd, thin_svd_into,
};

#[cfg(feature = "viable_manifold_graph")]
pub use viable_manifold_graph::{
    ClosurePredicate, GraphBuildConfig, SafeManifoldGraph, ViabilityPredicate, VolumeFieldConfig,
    build_safe_manifold_graph, manifold_curiosity_walk, manifold_geodesic, manifold_random_walk,
    pullback_volume,
};

#[cfg(feature = "ac_prefix")]
pub use ac_prefix::{AcPrefix, AcPrefixMask};

#[cfg(feature = "flow_field_nav")]
pub mod flow;
#[cfg(feature = "flow_field_nav")]
pub use flow::{
    FlowField, FlowFieldCache, FlowFieldConfig, LeoPotentialGrid, blend_steering, fft_smooth,
    fft_smooth_into, flow_steering, inflate_obstacles, should_use_flow_field,
};

// Spectral primitives — Fourier-basis algebra on discrete samples.
// Distilled from the FNO practical-perspective survey (Research 307).
// Each operator ships behind its own feature flag and is independently GOAT-gated.
// - `continuation` (feature `fourier_continuation`, Plan 323): Fourier
//   continuation for non-periodic latent fields — closed-form polynomial
//   periodic extension so the FFT does not produce Gibbs ringing at the
//   boundaries. The one modelless FNO primitive the codebase genuinely
//   lacked (Research 307 §3 candidate plan #1). Opt-in until G1–G4 pass.
// - `differentiation` (feature `spectral_differentiation`, Plan 325):
//   standalone FFT-based spectral differentiation on periodic uniform 1D
//   grids — multiply FFT coefficients by `(iω)^m`, IFFT back. The
//   specialized 1D-periodic case where DEC `exterior_derivative` is
//   overkill. Opt-in until G1–G4 pass.
#[cfg(any(feature = "fourier_continuation", feature = "spectral_differentiation"))]
pub mod spectral;
#[cfg(feature = "fourier_continuation")]
pub use spectral::continuation::{
    FcConfig, FcScratch, FourierContinuationError, MAX_POLY_ORDER, fourier_continue,
    fourier_continue_into,
};
#[cfg(feature = "spectral_differentiation")]
pub use spectral::differentiation::{
    MAX_ORDER, SpecDiffConfig, SpecDiffError, SpecDiffScratch, spectral_differentiate,
    spectral_differentiate_into,
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

// BabelCodec — Readability-relaxed semantic codec (Plan 331, Research 312,
// arXiv:2606.19857 BabelTele). Successor text codec to CompressionDrafter:
// where CompressionDrafter failed G2 twice on the Seal corpus (byte-level LZ4
// matching on short quest-grammar strings), BabelCodec operates on semantic
// STRUCTURE (BT-P8 fixed symbolic mapping rules) — purpose-built for KG-triple
// / entity-attribute / config / quest-grammar surfaces. Ships three pieces:
// (1) generic `BabelCodec` trait, (2) `FixedRuleTextCodec` (deterministic BT-P8
// text codec, the modelless subset of BabelTele), (3) `SigmoidLatentCodec<D>`
// (generic-trait facade over existing DensityBudget infrastructure, latent-level
// analog — value is API uniformity, NOT new capability), plus BLAKE3 commitment
// for the future LatCal chain bridge (.issues/002). Sigmoid, not softmax.
// Opt-in until the G1–G5 GOAT gate passes — the same G2 (≥ 2× on real corpus)
// gate that killed CompressionDrafter twice.
#[cfg(feature = "babel_codec")]
pub mod babel_codec;
#[cfg(feature = "babel_codec")]
pub use babel_codec::{
    BabelCodec, BabelCommitment, BabelPair, CompressedLatent, FixedRuleTextCodec, SigmoidLatentCodec,
};

// Analytic Lattice — k×k transport operator chain composer + ASOC trait shapes
// + direction-vector SIMD decoder + spectral audit (Plan 330, Research 311).
// katgpt-core half: pure math primitives + generic trait shapes (NO GpuFuture
// import — leaf-clean). The ComposerTick: GpuFuture impl + Join3 combinator
// ship in riir-engine under the `analytic_lattice_runtime` feature (Phase 1b).
// Opt-in until G1–G6 GOAT gate passes.
#[cfg(feature = "analytic_lattice")]
pub mod analytic_lattice;
#[cfg(feature = "analytic_lattice")]
pub use analytic_lattice::{
    ChainError, ComposerCtx, LatticeVector, PlasmaDraft, RederiveOp, TransportOperator,
    apply_operator_into, audit::AuditReport, audit::spectral_audit, batch_compose_chain,
    batch_compose_chain_into, compose_chain, compose_chain_into, decoder::direction_vector_decode,
    decoder::direction_vector_decode_into,
};

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
// Plan 332 — principled multi-scale basis constructors (DCT-log, Haar-packet).
// gated by the dedicated `funcattn_structured_basis` feature (implies funcattn).
#[cfg(feature = "funcattn_structured_basis")]
pub use funcattn::{make_dct_log_basis, make_haar_packet_basis};

// Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN (Plan 310,
// Research 291, arxiv 2605.31559). Generalizes FUNCATTN to d_src ≠ d_dst,
// enabling train-on-small-deploy-on-large latent transfer without retraining.
// Open primitive: frozen BLAKE3-committed bases + zero-alloc transport.
// Opt-in until G1–G4 GOAT gate passes.
#[cfg(feature = "cross_resolution_transport")]
pub mod cross_resolution;
#[cfg(feature = "cross_resolution_transport")]
pub use cross_resolution::{
    CrossResScratch, CrossResolutionBases, CrossResolutionError, project_to_spectral_into,
    reconstruct_from_spectral_into, transport_cross_domain_cross_resolution_into,
    transport_cross_resolution, transport_cross_resolution_into,
};

// Latent Field Steering — top-down direction-vector injection into mutable
// latent state (Plan 309, Research 290, CAA + functional emotions). The missing
// fourth quadrant: CNA mutates neurons, EmotionDirections is read-only, FPCG
// refuses mutation — this injects directly into the latent state on the hot
// path. Zero-alloc SIMD SAXPY + sigmoid-falloff localized support.
// Opt-in until G1–G5 GOAT gate passes (G2 make-or-break: rank preservation ≥0.95).
#[cfg(feature = "latent_field_steering")]
pub mod latent_steering;
#[cfg(feature = "latent_field_steering")]
pub use latent_steering::{
    FieldSupport, HLA_AROUSAL, HLA_CALM, HLA_DESPERATION, HLA_DIM, HLA_FEAR, HLA_VALENCE,
    LatentField, LatentSteeringError, LatentSteeringVector, apply_field_to_crowd,
    apply_latent_steering, apply_latent_steering_weighted, kernel_weight,
};

// Phase-Modulated Subspace Rotation Gate — norm-preserving latent coupling
// `cos α ⊙ a + sin α ⊙ b` with phase from a sigmoid projection onto a frozen
// direction vector (Plan 322, Research 305, arxiv 2605.12700 UFO). The
// genuinely-new operation class: every other latent op in the crate is
// additive / convex-combo / dot-projection / wedge-detection / linear-transport
// / spatial-sum — none has the `sin²α+cos²α=1` Pythagorean norm-preservation
// invariant. §3.5 modelless Path 2 unblock: the trained `γ_θ` is replaced with
// `α = sigmoid(⟨state, direction⟩ · λ) · π/2` (closed-form). Opt-in until the
// G1–G4 GOAT gate passes (G1 norm-preservation <1e-4 is the kill switch).
#[cfg(feature = "phase_rotation_coupling")]
pub mod phase_rotation;
#[cfg(feature = "phase_rotation_coupling")]
pub use phase_rotation::{
    PhaseRotationError, PhaseRotationGate, PhaseRotationScratch, compute_phase_from_projection,
    compute_phase_per_channel_into, phase_rotation_gate_into,
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

// ── Bisimulation Operator Inference (Plan 324, Research 308, arxiv 2602.19260) ─
//
// Open primitive: quotient an observed transition graph into bisimulation-
// equivalent state classes (signature-based partition refinement, O((S+E)
// log² S log d)) and infer an abstract PDDL-like operator schema. The
// lighter-weight PDDL-side counterpart to Induced CWM (Plan 296): where CWM
// induces executable *code* via an LLM, this induces an *operator schema* via
// a deterministic graph algorithm. Closes Research 264 §2.2 gaps #1 (PTG) +
// #2 (motif mining). Opt-in by design — downstream pipelines (riir-ai NPC
// runtime, riir-chain LatCal consumer) opt in by enabling the feature.
#[cfg(feature = "bisimulation_operator_inference")]
pub mod bisimulation;
#[cfg(feature = "bisimulation_operator_inference")]
pub use bisimulation::{
    BisimulationQuotient, OperatorDef, OperatorLabel, OperatorSchema, Plan, QuotientEdge,
    StateClassId, StateId, Transition, TransitionGraph, TransitionGraphBuilder, partition_refine,
    plan as bisimulation_plan,
};

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

// ── Committed Field Blend (Plan 321, Research 302) ───────────────────────
//
// Open MIT-licensed primitive: the sampling-invariant half of the FAME
// Super-GOAT. A `CommittedFieldBlend<N, D>` computes blend weights pi ONCE
// from a trajectory summary via sigmoid projection, then FREEZES them for
// the entity's lifetime. The blended field f_pi(z) = Σ_k sigmoid(pi_k/tau) ·
// f_k(z) governs dynamics; because both pi and the fields are frozen, the
// trajectory is sampling-invariant (FAME Proposition 3 / Young-integral).
// Zero-alloc apply + BLAKE3-committed. Reuses personality_composition's
// sigmoid + simd::simd_fused_scale_acc (DRY).
// Opt-in until G1–G5 GOAT gate passes; G2 (sampling invariance) is the
// make-or-break gate. Private selling-point guide at riir-ai/.research/158.
#[cfg(feature = "committed_field_blend")]
pub mod committed_field_blend;
#[cfg(feature = "committed_field_blend")]
pub use committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend, TriArchetypeBlend};

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

// Gain/Cost Loop Halting Primitive — open substrate-agnostic kernel for per-loop
// halting decisions (Plan 304, Research 282, arXiv:2606.18023, LoopCoder-v2).
//
// halt when marginal refinement gain < marginal drift cost × τ; oscillation
// early-halt via cos θ < 0; L_min floor protects representational capacity.
// Composes with the shipped elastic-loop override (Issue 035) — Phase 2 will wire
// this into `forward_looped()` (separate scope). Phase 1 ships the kernel only.
//
// Latent vs Raw: gain/cost signals are local latent (per-loop hidden-state
// deltas); the halt count L is a deterministic raw scalar safe to sync/replay.
//
// Opt-in until G1–G5 GOAT gate (Research 149 §5) passes.
#[cfg(feature = "gain_cost_halt")]
pub mod gain_cost_halt;
#[cfg(feature = "gain_cost_halt")]
pub use gain_cost_halt::{
    GainCostLoopHalter, HaltDecision, HaltReason, angular_change, hidden_erank, step_size,
};

// Depth-Invariance Diagnostic + Magnitude-Regularized Residual — the
// root-cause counterpart to four symptom-only detectors (BeliefRankPruner,
// GainCostLoopHalter, latent_functor/reestimation.rs,
// micro_belief/coherence_bench.rs). Modelless math, no game semantics.
// Classifies recursive latent-state chains as DepthInvariant /
// DepthSpecificRefinement / Collapsed / Insufficient. The MagnitudeReg
// wrapper is the modelless fix for kernels we own (HLA, functor,
// micro_belief, engram, Raven); for frozen MLPs (BeliefDrafter) the fix
// requires retraining → riir-train.
// Plan 306 Phase 1+5; Research 286; arXiv:2605.09992 Eldenk et al.
// Opt-in until G1 (8 correctness tests) passes.
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance;
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance::{
    DepthInvarianceConfig, DepthInvarianceDiagnostic, DepthInvarianceKind, MagnitudeRegularization,
    Scratch, apply_magnitude_regularization, classify_chain, classify_chain_batched,
};

// Shared linear-algebra kernels. Originally extracted for `karc`'s ridge-style
// solvers (Plan 308); the f32 Cholesky/ridge path lives here as a standalone
// extraction of the PEIRA `(N + λI)⁻¹` pattern — see the module note for why
// PEIRA's f64 path is left untouched. Plan 319 (Clifford geometric product)
// and Plan 326 (Tucker/HOSVD tensor factorization) ship peers under `linalg::`
// — each must also gate this `pub mod` so the crate compiles when only that
// feature is on.
#[cfg(any(
    feature = "karc_forecaster",
    feature = "geometric_product",
    feature = "tucker_factorization"
))]
pub mod linalg;

// KARC — Kolmogorov-Arnold Reservoir Computing delay-basis-ridge forecaster
// (Plan 308, Research 288, arXiv:2606.19984). Modelless, inference-time
// trajectory forecaster: delay-embedding × sealed KarcBasis (Fourier/Chebyshev/
// BSpline) × closed-form ridge readout, with a zero-alloc forecast matvec.
// Opt-in until G1–G4 GOAT gate passes (no root-feature alias in Phase 1).
#[cfg(feature = "karc_forecaster")]
pub mod karc;
#[cfg(feature = "karc_forecaster")]
pub use karc::{
    BSplineBasis, ChebyshevBasis, DelayRing, FitError, FourierBasis, KarcBasis, KarcForecaster,
    KarcScratch, LowRankFitScratch, chunked_gram_into, feature_expand, feature_expand_higher_order,
    forecast_low_rank_apply, higher_order_feature_count, low_rank_fit,
};

// ARG Protocol Primitives — open half of the ARG × Latent Substrate Super-GOAT
// fusion (Plan 327 Phases 1-3, Research 309, Guide 160 private). Five generic
// protocol primitives distilled from the ARG Standard
// (https://protocol.airistech.ai/arg-core.html, Iris Technologies 2026):
// `PolicyEnvelope` (Step 1 hard gate), `TaxonomyValidator` (Step 3 deterministic
// label-set validator), `LifecycleState` + `RedirectTable` (Step E ontology
// lifecycle continuity), `TypedOfflineCandidate` + `CandidateIntent` (Step C
// typed offline candidate), `OfflineCandidateScorer` (Step C scoring with the
// G5 silence-bias penalty), `InfoRegistry` (Step 9 + Step C two-phase dedup
// with grey-zone review). Private runtime composition with HLA / Entity
// Cognition Stack / VMG / Sub-Goal Compaction lives in riir-ai Plan 337.
// No game/chain/shard semantics. DEFAULT-ON (Plan 327 Phase 4, 2026-06-25):
#[cfg(feature = "arg_protocol")]
pub mod arg;
#[cfg(feature = "arg_protocol")]
pub use arg::{
    AccessScope, CandidateIntent, CandidateKind, CompareFn, CompareResult,
    DEFAULT_AUTO_COMMIT_THRESHOLD, Evidence, EvidenceId, GainComponents, InfoKey, InfoOutcomeStatus,
    InfoRegistry, InfoType, InfoUnit, LabelId, LabelSet, LabelSignature, LifecycleState,
    MatchResult, MatchScratch, OfflineCandidateScorer, PayloadHash, PayloadHashCompare,
    PolicyConstraints, PolicyDecision, PolicyEnvelope, PolicyState, Provenance, RedirectTable,
    ResponseMode, ScoredCandidate, ShouldProceed, TaxonomyKind, TaxonomyNode, TaxonomyValidator,
    TypedOfflineCandidate, ValidationError, ValidationResult, ValidationScratch,
};

// Non-Interference Memory Branches — Super-GOAT fusion (Plan 329, Research 310,
// arXiv:2606.20638 Goel et al. Oxford Jun 2026). Five generic open primitives:
// BranchBank (bounded persistent CognitiveBranch bank with spawn/merge/prune
// lifecycle), BranchRouter (dot-product snap + Jaccard fallback), VerifierGate
// (reward + curiosity + centroid-quarantine write gate, composes with CLR),
// NonInterferenceProjection (orthogonal latent subspace per branch),
// BudgetCompiler (priority-cascade context compiler under fixed budget). Fuses
// BAKE × CLR × MCGS × Engram × ARG × closure-instrument × Salience into a new
// capability class: per-NPC continual adaptation without catastrophic
// forgetting. Composes with arg_protocol LifecycleState when both features on.
// Opt-in until G1–G5 GOAT gate passes (Phase 3).
#[cfg(feature = "non_interference_branches")]
pub mod branching;
#[cfg(feature = "non_interference_branches")]
pub use branching::{
    AssignError, AssignResult, BranchBank, BranchId, BranchLifecycle, BranchRouter, BranchStats,
    BudgetCompiler, CognitiveBranch, CompiledContext, CompiledItem, DEFAULT_ASSIGN_MAX_INTERFERENCE,
    DEFAULT_BUDGET_BYTES, DEFAULT_MAX_BRANCHES, DEFAULT_ORTHOGONAL_EPSILON, DEFAULT_PROJECTION_DIM,
    DEFAULT_QUARANTINE_CENTROID_THRESH, DEFAULT_TAU_CURIOSITY, DEFAULT_TAU_JACCARD,
    DEFAULT_TAU_SNAP, DEFAULT_TAU_SPAWN, DEFAULT_TAU_WRITE, EpisodicEntry, FailureEntry,
    NonInterferenceProjection, PriorityTier, ProceduralRule, RetrievedMaterials, RouteMode,
    RouteResult, VerifierGate, WriteDecision, max_orthogonal_branches,
};

// Sleep-Time Query Anticipator — open primitive for offline query anticipation
// (Plan 334, Research 318, arXiv:2504.13171 Lin et al. Letta/Berkeley).
// Implements the open math half: SleepTimeAnticipator orchestrates per-direction
// sleep-time compute → emits reusable AnticipatedQuerySet (the c' artifact,
// BLAKE3-committed) → wake-time consume() does cheap dot-product + sigmoid-
// gated lookup, falling through to fresh compute on low-predictability queries.
// PredictabilityScorer trait + DotPredictabilityScorer default
// (p = sigmoid(α·dot(c,dir)+β)); AmortizationCostModel operationalizes the
// paper's §5.3 cost model. Game-specific direction-vector catalogs, NPC tiering,
// HLA wiring, and chain commitment live in riir-ai Plan 341 (private).
// Phase 1 ships traits + types + IdentityFunctorOp (synthetic-test default);
// Phase 2 ships synthetic gates G1/G2/G5/G6/G7. G2/G3/G4 quality gates require
// a real predictability-labeled corpus → deferred to riir-ai Plan 341.
// Opt-in until G1–G5 GOAT gate passes; promotion to default-on requires
// Plan 341 G1–G5 to clear on a real game corpus.
#[cfg(feature = "sleep_time_anticipation")]
pub mod sleep_time;
#[cfg(feature = "sleep_time_anticipation")]
pub use sleep_time::{
    AmortizationCostModel, AnticipatedQueryDir, AnticipatedQuerySet, AnticipatedSlot,
    DEFAULT_LATENCY_PREMIUM, DotPredictabilityScorer, IdentityFunctorOp, SLEEP_TIME_DEFAULT_K,
    PredictabilityScorer, SleepTimeAnticipator, SleepTimeComputeOp, SleepTimeScratch, commit_direction,
    consume, consume_gate, consume_with_match_mode, consume_gate_with_match_mode, ConsumeMatchMode,
};

// PairedLossGap — generic modelless paired token-level loss gap diagnostic
// (Plan 335, Research 319, arXiv:2606.20936 Li & Merrill AI2). Pure
// measurement tool: given two log-prob traces over the same prefixes, compute
// per-token Δ_i = ℓ_A − ℓ_B, stratify by token class, report filtered
// aggregates (ALL / TOP-K∩NO-COPY / COPY-N-ONLY) that amplify small
// architecture gaps aggregate loss hides. ClassSizeBound exposes Proposition 1
// (DKL ≤ log|V_τ|) — the volume-of-support bound justifying raw-vs-latent
// sync. Generic math, no game/chain/shard semantics — legitimately public.
// NOT an inference mechanism (measurement tool only) → not Super-GOAT.
// Opt-in until G1–G4 GOAT gate passes.
#[cfg(feature = "paired_loss_diagnostic")]
pub mod paired_loss;
#[cfg(feature = "paired_loss_diagnostic")]
pub use paired_loss::{
    ClassGapReport, ClassGapRow, ClassSizeBound, CopyNGramTagger, FilterKind, FilterScratch,
    PairedLossGap, TokenClass, TokenTagger,
};
