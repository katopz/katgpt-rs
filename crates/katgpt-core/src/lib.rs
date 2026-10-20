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

/// Standard logistic sigmoid: `σ(x) = 1 / (1 + e^{-x})`.
///
/// Numerically stable sigmoid σ(x) = 1/(1+e^{-x}), output in (0, 1).
///
/// Delegates to [`simd::fast_sigmoid`] (Cephes 6th-order polynomial exp,
/// ~1 ULP accurate, ~1.7× faster than libm `exp` on aarch64). The two-branch
/// form that previously lived here was a workaround for libm `exp`'s overflow
/// behavior; `fast_sigmoid` handles this internally via ±40 early-exits.
///
/// Always available — no feature gate — because it's a pure math utility
/// consumed across many domains (band conditioning, CGSP, faithfulness gates,
/// personality composition, etc.). Hoisted here from `band_conditioner::sigmoid`
/// (Proposal 003 Phase 0.1) so the upcoming `katgpt-band` extraction doesn't
/// drag a math utility into the band crate. Per the project rule: sigmoid,
/// never softmax.
///
/// ⚠ This returns the APPROXIMATION. For committed-value paths that must not
/// inherit it (consensus scoring, reward gates, replay-pinned values), use
/// [`exact_sigmoid`] — the libm two-branch reference form.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    simd::fast_sigmoid(x)
}

/// Exact logistic sigmoid: the two-branch numerically stable form over libm
/// `exp` — no Cephes polynomial, no ±40 saturation early-exit. The
/// **bit-stable reference variant** of [`sigmoid`] for committed-value
/// paths (consensus scoring, reward gates, replay-pinned values); the two
/// differ on ~9% of `[-40, 40]` inputs and on every input past ±40, where
/// [`sigmoid`] clamps to exactly `0.0`/`1.0` while the true value is a
/// representable tiny/one-minus-tiny.
///
/// Always available — no feature gate (the [`crate::float_order`]
/// ungated-math precedent; pure math, additive API, no existing call site
/// rerouted). Per the project rule: sigmoid, never softmax.
#[inline]
pub fn exact_sigmoid(x: f32) -> f32 {
    simd::exact_sigmoid(x)
}

/// [`exact_sigmoid`] in f64 — the form callers compute in when they narrow
/// to f32 only at the end (riir-chain's congestion/forensic paths; Issue
/// 156). Narrowing first and using the f32 variant would be a numerics
/// change, not a delegation.
#[inline]
pub fn exact_sigmoid_f64(x: f64) -> f64 {
    simd::exact_sigmoid_f64(x)
}

/// Noisy-OR span aggregation `1 − Π(1−kᵢ)`, direct product form.
///
/// Generalized from the riir-games-civ salience gate's literal
/// `1 − (1−c)(1−boost)` (Research 491 distilled item 5 / Issue 672 T4 — the
/// civ site delegates here). Inputs are clamped to `[0, 1]` (each k is a
/// probability-style weight). Boundary identities: all k = 0 ⇒ exactly 0;
/// any k = 1 ⇒ exactly 1; monotone non-decreasing in every k.
///
/// **Bit-compatibility note:** for the two-term case the accumulation is
/// exactly `(1.0−k₀)·(1.0−k₁)` in source order — bit-identical to the civ
/// inline formula it replaces (pinned by the sterling module's test).
///
/// Always available — no feature gate — same rationale as [`sigmoid`]: a
/// pure math utility consumed across domains, with a DEFAULT-ON consumer
/// (civ salience gate) that must not gain a feature dependency for a
/// behavior-preserving refactor.
#[inline]
pub fn noisy_or(ks: &[f32]) -> f32 {
    let mut acc = 1.0f32;
    for &k in ks {
        let one_minus = 1.0 - k.clamp(0.0, 1.0);
        acc *= one_minus;
    }
    1.0 - acc
}

/// Log1p-stable noisy-OR for spans of MANY small weights:
/// `1 − exp(Σ ln(1−kᵢ)) = −expm1(Σ log1p(−kᵢ))`.
///
/// The direct product form underflows to `1 − 0 = 1` prematurely when
/// enough small factors accumulate (catastrophic for long spans of tiny
/// probabilities); the log1p/expm1 form keeps resolution. Inputs clamped
/// to `[0, 1]`: k = 1 yields `log1p(−1) = −∞` ⇒ the stable form returns
/// exactly 1.0 — the correct saturated limit.
#[inline]
pub fn noisy_or_stable(ks: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &k in ks {
        let kc = k.clamp(0.0, 1.0);
        sum += (-kc).ln_1p(); // ln(1 − k), stable for small k
    }
    -sum.exp_m1()
}

#[cfg(feature = "tiled_attention")]
pub mod attention;

#[cfg(feature = "newton_schulz")]
pub mod newton_schulz;

pub mod float_order;

pub mod stats;

#[cfg(any(feature = "probe_guidance", feature = "loop_guidance"))]
pub mod contrast_combine;

#[cfg(feature = "path_kernel_sim")]
pub mod path_kernel_sim;

#[cfg(feature = "rating")]
pub mod rating;

#[cfg(feature = "laprop")]
pub mod laprop;

pub mod prover_selection;

#[cfg(any(feature = "linking_fold_fold", feature = "linking_fold_detector"))]
pub mod linking_fold;

#[cfg(feature = "best_belief")]
pub mod best_belief;
#[cfg(feature = "best_belief")]
pub use best_belief::{best_belief_score, best_belief_scores, select_best_belief};

#[cfg(feature = "hint_regret")]
pub mod hint_regret;

#[cfg(feature = "state_probe")]
pub mod state_probe;

#[cfg(feature = "risk_control_exit")]
pub mod risk_control_exit;

#[cfg(feature = "distributional_steering")]
pub mod distributional_steering;

#[cfg(feature = "twist_smc")]
pub mod twist_cache;

#[cfg(feature = "entropic_tilt")]
pub mod entropic_tilt;
#[cfg(feature = "entropic_tilt")]
pub use entropic_tilt::{
    KL_BUDGET_LN2, solve_beta, tilt_advantages_into, tilt_advantages_loo_into, tilted_weights,
};

#[cfg(feature = "tether")]
pub mod tether;
#[cfg(feature = "tether")]
pub use tether::{
    DEFAULT_EMA_DECAY, DEFAULT_RHO, DEFAULT_WINDOW, DEGENERATE_EPS, EvAccumulator, TetherBlend,
    TetherStats, control_variate_improves, fit_rho, horizon_decay, sse,
};

#[cfg(feature = "ignition_schedule")]
pub mod ignition;
#[cfg(feature = "ignition_schedule")]
pub use ignition::{IgnitionSchedule, commit_time_star, ignition_time, order_by_ignition_into};

pub mod conformal;
#[cfg(feature = "conformal_predictive_intervals")]
#[cfg(feature = "limit_fixture")]
pub mod limit_fixture;

#[cfg(feature = "incidence_algebra")]
pub mod incidence;

#[cfg(feature = "evidence_tripwire")]
pub mod evidence_tripwire;

#[cfg(feature = "state_option_scoring")]
pub mod state_option_scoring;

#[cfg(feature = "event_state_windows")]
pub mod event_state_window;

#[cfg(feature = "escalation_probe_gate")]
pub mod escalation_probe_gate;

#[cfg(feature = "escalation_probe_gate")]
pub mod state_probe_eval;

#[cfg(feature = "template_decode")]
pub mod template_decode;
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::metrics::{
    crps, crps_interval, empirical_coverage, mean_crps_interval, mean_winkler, winkler_score,
};
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::{
    ConformalIntervalCalibrator, DecayUnit, PointForecaster, PredictiveInterval, ResidualMode,
    ResidualRingBuffer, RingBuffer, SeasonalNaiveForecaster, SeasonalPoolForecaster,
    seasonal_naive_floor,
};

#[cfg(all(
    feature = "conformal_predictive_intervals",
    feature = "karc_forecaster"
))]
pub use conformal::KarcChannelForecaster;

#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::{
    FloorAdapter, FloorComparisonReport, OverallVerdict, PredictiveOutput, TrajectoryCorpus,
    UqMetrics, UqPrimitiveUnderTest, empirical_quantile_interval, run_floor_comparison,
};
#[cfg(feature = "coda_fusion")]
pub mod coda;
#[cfg(feature = "dec_operators")]
pub use katgpt_dec as dec;
// assign — spec-driven constrained assignment solver (Plan 620, 2026-10-07; OPT-IN pending the GOAT
// gate + consumer wiring). Spun out to the `katgpt-assign` crate; re-exported for backwards compatibility.
#[cfg(feature = "assignment")]
pub use katgpt_assign as assign;
#[cfg(feature = "dec_operators")]
pub mod dec_freeze;
#[cfg(feature = "dec_operators")]
pub use dec_freeze::CochainFreezeEnvelope;
pub mod delta_mem;
// hla — Higher-order Linear Attention substrate (cache types + streaming kernels), spun out to the
// `katgpt-hla` crate (Issue 007 Phase E Tier 2 #4); re-exported as `katgpt_core::hla` for backwards compatibility.
pub use katgpt_hla as hla;
// leaky_core — shared leaky-integrator primitive, spun out to the `katgpt-types` leaf (Issue 007 Phase E
// Tier 1 #3); re-exported here as `katgpt_core::leaky_core` for backwards compatibility.
pub use katgpt_types::leaky_core;

pub mod mcts;

#[cfg(feature = "mcts_state_action_cache")]
pub mod mcts_state_action_cache;

pub mod freeze;

#[cfg(feature = "kpt_archive")]
pub mod kpt_archive;

pub mod proof_cache;

#[cfg(feature = "parallax_attn")]
pub mod parallax_attn;
pub mod thinking_mode;

#[cfg(feature = "tropical_algebra")]
pub mod algebra;
pub mod shard_embedding;

#[cfg(feature = "ssmax_temperature")]
pub mod ssmax;

#[cfg(feature = "scale_invariant_attn")]
pub mod scale_invariant;

#[cfg(feature = "logit_regime")]
pub mod logit_regime;

#[cfg(feature = "simd_lut_dequant")]
pub mod simd_lut_dequant;

#[cfg(feature = "smooth_min_similarity")]
pub mod similarity;
#[cfg(feature = "smooth_min_similarity")]
pub use similarity::{edit_penalty, smooth_min_similarity};

#[cfg(feature = "recos")]
pub use similarity::{recos_sim, recos_sim_ranking, recos_sim_slice, recos_sim_slice_into};

#[cfg(feature = "elasticity_gated_update")]
pub mod elasticity_gated_update;
#[cfg(feature = "elasticity_gated_update")]
pub use elasticity_gated_update::{
    ElasticityConfig, compute_error, effective_neighborhood_size, elasticity_gated_update_into,
    neighborhood_weight,
};

#[cfg(all(feature = "dec_operators", feature = "elasticity_gated_update"))]
pub use elasticity_gated_update::{
    error_weighted_graph_laplacian, error_weighted_graph_laplacian_into,
};

pub mod set_diffusion_schedule;
pub use set_diffusion_schedule::{
    PositionOffsetSchedule, ar_order, block_causal_gen_steps, mdlm_gen_steps, order_to_gen_steps,
    probability_order, uniform_order, uniform_order_with,
};

pub mod ugc_schedule;
pub use ugc_schedule::{
    UGC_MASK, UgcBlockPlan, UgcDenoiser, UgcIntervalEstimate, UgcProfile, UgcScratch,
    bernoulli_unmask_with_grid, certified_block_plan, certified_iteration_count, dp_partition,
    equal_sqrt_mass_grid, estimate_interval, estimate_profile, inv_log_reveal_odds,
    log_reveal_odds, reveal_grid_from_plan, reveal_odds,
};

#[cfg(feature = "minmax_partition")]
pub mod partition;

#[cfg(feature = "switch_cost")]
pub mod switch_cost;
#[cfg(feature = "switch_cost")]
pub use switch_cost::{
    DEFAULT_ALPHA, FactorizedSwitchCost, NEUTRAL_ACC, SwitchCostSnapshot, SwitchCostTable, cdf_rank,
};

#[cfg(feature = "freedom_selection")]
pub mod extension_count;
#[cfg(feature = "freedom_selection")]
pub use extension_count::{
    ExtensionOccupancy, FIRST_ACTIVATION_GAIN, LossGate, freedom_gain, log_freedom,
};

#[cfg(feature = "effective_degree")]
pub mod effective_degree;
#[cfg(feature = "effective_degree")]
pub use effective_degree::{
    EdConfig, EdError, EdResult, EdScratch, MAX_ED_DEGREE, MAX_ED_TERMS, ed_from_coeff_norms,
    ed_over_pairs, effective_degree_along_path, effective_degree_along_path_multi,
    randomized_cosine_nodes,
};
// simd — SIMD-accelerated linear algebra kernels (NEON / AVX2 / WASM-SIMD128 / scalar fallback), spun out
// to the `katgpt-types` crate (Issue 007 Phase E Tier 1 #2); re-exported as `katgpt_core::simd` for back-compat.
pub use katgpt_types::simd;
pub mod speculative;
pub mod traits;

pub mod lsh_seed;

pub mod prompt_backend;
pub use prompt_backend::InferenceBackend;
// types — shared configuration, RNG, math utilities, LoRA, domain embeddings, and inference types; spun
// out to the `katgpt-types` crate (Issue 007 Phase E Tier 1 #2); re-exported as `katgpt_core::types` for back-compat.
pub use katgpt_types as types;

#[cfg(feature = "cgsp")]
pub mod cgsp;
#[cfg(feature = "cgsp")]
pub use cgsp::{
    BatchQualityGate,
    BeliefGridProjectionGuide,
    BreakevenDifficultyFilter,
    Candidate,
    CgspConfig,
    CgspLoop,
    ColinearityBatchGate,
    CollapseSignal,
    ComplexityWeights,
    CuriosityConjecturer,
    CuriosityPrioritySnapshot,
    CycleResult,
    CycleStats,
    DEFAULT_BELIEF_DIRECTION_DIM,
    DEFAULT_K,
    DEFAULT_POOL_SIZE,
    DifficultyFilter,
    Direction,
    EntropyCollapse,
    HintDeltaBandit,
    NoOpBatchGate,
    NoOpDifficultyFilter,
    PoolConjecturer,
    Priority,
    QualityGuide,
    ScratchBuffers,
    SolveRate,
    Solver,
    Target,
    entropy_nats,
    structural_complexity,
    // Note: `sigmoid` is no longer re-exported here — it's now an always-on
    // top-level `katgpt_core::sigmoid` (Proposal 003 Phase 0.1). The module-local
    // `katgpt_core::cgsp::sigmoid` (in cgsp/types.rs) remains for `cgsp::*` paths.
};

#[cfg(feature = "cgsp_dual_pool")]
pub use cgsp::{DualPoolBandit, DualPoolConfig, PoolId, ReachableDualPoolRouter};

#[cfg(all(feature = "cgsp", feature = "gain_cost_halt"))]
pub use cgsp::{KnpcDecision, KnpcSelector};

#[cfg(feature = "action_bridge")]
pub mod bridge;
#[cfg(feature = "action_bridge")]
pub use bridge::ActionBridge;

pub use traits::{
    ActionSpaceLog, BestBuddyAligner, BinaryScreeningPruner, ConstraintPruner, DominoPruner,
    FeatureClass, GameState, NoPruner, NoScreeningPruner, RandomRolloutPolicy, RolloutPolicy,
    ScreeningPruner, StateHeuristic, best_buddies, pearson_correlation,
};
pub use traits::{GenerativeConstraintPruner, SpeculativeGenerator};

#[cfg(feature = "recursion_logits")]
pub use traits::RecursionLogits;

#[cfg(feature = "qgf_oracle")]
pub use traits::{NoGuidanceOracle, QGradientOracle};
#[cfg(feature = "qgf")]
pub mod qgf;

// micro_belief — per-entity recurrent state kernel (Plan 276, Research 242), spun out to the
// `katgpt-micro-belief` crate (Issue 007 Phase E Tier 1 #3); re-exported here for backwards compatibility.
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
pub use micro_belief::{BoMSampler, NoiseQueryConfig, QmcMethod, SeedStrategy, dot_product_scorer};

#[cfg(all(feature = "qmc_sampling", feature = "bom_sampling"))]
pub use speculative::fill_noise_queries_gaussian_qmc_by_method;

// BoM G2 arena harness — Plan 281 T2.3: engine-side traits + synthetic reference env; riir-ai
// implements the traits over a real bomber/go sim to produce the empirical G2 gate.
#[cfg(feature = "bom_sampling")]
pub use micro_belief::{
    ArenaAction, ArenaEnvironment, BeliefPlanner, BoMMeanPlanner, BoMMinimaxPlanner,
    ComparisonResult, DeterministicPlanner, EnvHint, PlannerOutcome, SyntheticThreatArena,
    bom_mean_attractor, bom_minimax_attractor, bom_minimax_leaky, run_arena_comparison,
};

#[cfg(any(feature = "faithfulness_probe", feature = "triggered_injection"))]
pub mod faithfulness;

pub mod pruners;

#[cfg(feature = "temporal_deriv")]
pub mod temporal_deriv;
#[cfg(feature = "temporal_deriv")]
pub use temporal_deriv::{TemporalDerivativeKernel, sigmoid_surprise_gate};

#[cfg(feature = "hippocampal_cache")]
pub mod hippocampal_cache;
#[cfg(feature = "hippocampal_cache")]
pub use hippocampal_cache::{HippocampalCache, SortedSlotCache};

#[cfg(feature = "hippocampal_cache")]
pub mod hippocampal_cache_dyn;
#[cfg(feature = "hippocampal_cache")]
pub use hippocampal_cache_dyn::HippocampalCacheDyn;

pub mod tiered_kv;

#[cfg(feature = "hga")]
pub mod hga;
#[cfg(feature = "hga")]
pub use hga::{GroupSummaryCache, MixedRopeSummarizer};

#[cfg(feature = "renoise_ce")]
pub mod renoise_ce;
#[cfg(feature = "renoise_ce")]
pub use renoise_ce::{
    Proposer, RenoiseCeConfig, RenoiseCeProbe, RenoiseCeScore, best_of_n_stability,
    renoise_ce_score, verify_and_restart,
};

#[cfg(feature = "freedom_selection")]
pub use renoise_ce::best_of_n_freedom;

#[cfg(all(feature = "renoise_ce", feature = "horizon_weights"))]
pub use renoise_ce::{RenoiseCeHorizon, renoise_ce_score_horizon};

#[cfg(feature = "renoise_ce_surprise")]
pub use renoise_ce::renoise_ce_surprise;

#[cfg(feature = "horizon_weights")]
pub mod horizon_weights;
#[cfg(feature = "horizon_weights")]
pub use horizon_weights::{
    HORIZON_WEIGHT_GRID, HorizonWeightTable, TimeAnnealRange, pfd_horizon_weight_at,
    pfd_horizon_weights, remaining_horizon_t_sample, remaining_horizon_weight,
    remaining_horizon_weights, terminal_truncation_ceiling, truncated_w_mass_fraction,
};

#[cfg(feature = "dual_leo")]
pub use traits::{
    ActingMode, AlphaSchedule, AutocurriculumSampler, BcConfig, BcTarget, DualLeoMixer,
};
#[cfg(feature = "leo_all_goals")]
pub use traits::{AllGoalsUpdate, LeoHead, sigmoid_bounded_q};

#[allow(deprecated)]
pub use shard_embedding::{EMBED_DIM, JlProjectionMatrix, STYLE_DIM as JL_STYLE_DIM};
#[cfg(feature = "loop_stability_fix")]
pub use types::LoopStabilityMode;
#[allow(deprecated)]
pub use types::ShardEmbedding;
#[allow(deprecated)]
pub use types::sample_token;
pub use types::{
    AttentionMode, AttentionProjection, CacheLayout, CalibrationMode, Config, ConvergenceSelector,
    CopyLateShape, DashAttnConfig, DilationConfig, HlaMode, HybridPattern, InferenceOverrides,
    InferenceResult, LoopMode, LoraAdapter, LoraPair, ModelArchitecture, ResidualGate,
    RetrievalHeadRole, Rng, RtTurboConfig, SdpaOutputGate, WeightDtype, kv_dim, lora_apply, matmul,
    matmul_f16, matmul_f16_parallel, matmul_parallel, matmul_relu, rmsnorm, sample_token_into,
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
    simd_matmul_rmsnorm_swiglu, simd_matmul_rmsnorm_swiglu_split,
};

#[cfg(all(feature = "coda_fusion", feature = "moa_inference"))]
pub use coda::{MoaActivation, moa_swiglu, simd_matmul_rmsnorm_moa_swiglu};

#[cfg(feature = "tiled_attention")]
pub use attention::{
    tiled_attention_batched, tiled_attention_forward, tiled_attention_forward_with_scores,
};

#[cfg(all(feature = "tiled_attention", feature = "scale_invariant_attn"))]
pub use attention::tiled_attention_forward_si;

#[cfg(feature = "parallax_attn")]
pub use parallax_attn::{
    ParallaxActivation, ParallaxConfig, ParallaxScratch, compute_rho, parallax_correction,
    tiled_attention_parallax_forward, tiled_attention_parallax_forward_retaining,
};

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
pub use simd::{
    DEFAULT_L3_BYTES, l3_cache_bytes, plasma_prefers_ternary, plasma_prefers_ternary_with_l3,
    simd_matvec_plasma_dispatch, simd_matvec_plasma_dispatch_with_l3, simd_ternary_matmul_batch,
    simd_ternary_matvec, ternary_matvec_scalar,
};
#[cfg(feature = "plasma_path")]
pub use types::TernaryWeights;

#[cfg(feature = "binary_plasma")]
pub use simd::{binary_matvec_scalar, simd_binary_matmul_batch, simd_binary_matvec};
#[cfg(feature = "binary_plasma")]
pub use types::{BinaryWeights, GROUP_SIZE as BINARY_GROUP_SIZE};

// Issue 578: the Q2_0_g128 container — ternary {-1,0,+1} bit-planes with the per-128 f16 group scale;
// neither tier could hold it (plasma_path: zero state but per-row scale; binary_plasma: group scale, no zero state).
#[cfg(feature = "ternary_q2_q8")]
pub use simd::{
    Q2Block, TernaryQ2Weights, q2_fast_path_available, simd_ternary_q2_matvec_q8_parallel,
    simd_ternary_q2_matvec_q8_parallel_n, simd_ternary_q2_matvec_q8_parallel_pair,
};
#[cfg(feature = "decode_spin_pool")]
pub use simd::{SpinPool, SyncMutPtr, spin_pool_run, spin_row_chunks, with_spin_pool};
#[cfg(feature = "ternary_group_q8_act")]
pub use simd::{quantize_activations_q8, simd_ternary_group_matvec_q8_parallel};
#[cfg(feature = "ternary_group_scale")]
pub use simd::{
    simd_ternary_group_matmul_batch, simd_ternary_group_matvec, simd_ternary_group_matvec_parallel,
    ternary_group_matvec_scalar,
};
#[cfg(feature = "ternary_group_scale")]
pub use types::{TernaryFfnHook, TernaryGroupWeights, TernaryInputProjHook, TernaryMatvecHook};

// Issue 582: the base-3 footprint tier — same alphabet and group scale as the Q2_0_g128
// container, 5 trits per byte instead of two bit-planes (1.75 vs 2.125 bits/weight, -17.6%).
#[cfg(feature = "ternary_trit_pack")]
pub use simd::{
    simd_ternary_trit_matvec, simd_ternary_trit_matvec_parallel, ternary_trit_matvec_scalar,
};
#[cfg(feature = "ternary_trit_pack")]
pub use types::{TRITS_PER_BYTE, TernaryTritWeights};

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
pub use simd::{
    ArgmaxAudit, argmaxable_witness, audit_argmaxable, compute_retrieval_margin,
    dim_capacity_ceiling, dim_capacity_floor, dim_capacity_required, dim_sufficiency_bound,
    ln_binomial, matrix_rank, sigmoid_margin_loss,
};

#[cfg(feature = "dual_gram_pca")]
pub use simd::simd_gram_f32;

#[cfg(feature = "roofline_cost")]
pub mod roofline;
#[cfg(feature = "roofline_cost")]
pub use roofline::{
    ComputeBound, Dtype, HardwarePeaks, OpType, RooflineCost, gemm_cost, gemv_cost, gram_cost,
    roofline_estimate,
};

#[cfg(feature = "ane_roofline")]
pub mod ane_roofline;
#[cfg(feature = "ane_roofline")]
pub use ane_roofline::{
    AneBound, AneCost, AneFamily, AneOpShape, AnePeaks, Device, ane_conv3x3_cost, ane_estimate,
    ane_gemm_cost, ane_gemv_cost,
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

// Sense substrate was spun out to the `katgpt-sense` crate (Issue 007 Phase E
// Tier 2 #7, Plan 338). `spectral_threat` stayed local (depends on `linoss`);
// it lives at `crate::sense_threat` and is re-exported through the
// `sense::spectral_threat` shim path below to preserve external consumers'
// `katgpt_core::sense::spectral_threat::*` paths bit-for-bit.
#[cfg(feature = "sense_composition")]
pub mod sense {
    pub use katgpt_sense::*;
    #[cfg(feature = "spectral_threat")]
    pub mod spectral_threat {
        pub use crate::sense_threat::*;
    }
}
#[cfg(feature = "spectral_threat")]
pub mod sense_threat;

#[cfg(feature = "slod")]
pub mod slod;
#[cfg(feature = "slod")]
pub use slod::{
    ScaleBoundary, SlodConfig, SlodOperator, SlodPruner, exp_map, frechet_mean,
    heat_kernel_weights, log_map, poincare_distance,
};

#[cfg(feature = "spectral_pruner")]
pub mod irrep_pruner;

#[cfg(feature = "frozen_evidence")]
pub mod frozen_evidence;
#[cfg(feature = "frozen_evidence")]
pub use frozen_evidence::{
    DEFAULT_DRIFT_CHAOS, DEFAULT_DRIFT_STUCK, FrozenEvidence, RecurrenceHealth, classify_drift,
    cosine, deliberate, gate_map_into, transition_into,
};

#[cfg(feature = "dual_wave")]
pub mod dual;

#[cfg(feature = "subspace_phase_gate")]
pub mod subspace_phase_gate;

#[cfg(feature = "subspace_intervention")]
pub mod subspace_intervention;

#[cfg(feature = "refinement_marginal")]
pub mod refinement_marginal;

#[cfg(feature = "group_invariance_probe")]
pub mod group_invariance_probe;

#[cfg(feature = "latent_trajectory_geometry")]
pub mod latent_trajectory_geometry;

#[cfg(feature = "swe_trajectory_freeze")]
pub mod swe_trajectory_freeze;

#[cfg(feature = "latent_confounder_audit")]
pub mod latent_confounder_audit;

#[cfg(feature = "leakage_probe")]
pub mod leakage_probe;

#[cfg(feature = "interpolation_geometry")]
pub mod interpolation_geometry;

#[cfg(feature = "viable_manifold_graph")]
pub mod viable_manifold_graph;

#[cfg(feature = "certified_frontier")]
pub mod certified_frontier;

#[cfg(feature = "set_admission")]
pub mod set_admission;

#[cfg(feature = "set_admission")]
pub mod set_admission_freeze;

#[cfg(feature = "usage_rate_eviction")]
pub mod kv_eviction;

#[cfg(feature = "canvas_schema")]
pub mod canvas;

#[cfg(feature = "zone_affective_manifold")]
pub mod zone_manifold;

#[cfg(feature = "zone_density_routing")]
pub mod zone_density;

#[cfg(feature = "ac_prefix")]
pub mod ac_prefix;

#[cfg(feature = "causal_head_importance")]
pub mod causal_head_importance;

#[cfg(feature = "causal_identification")]
pub mod causal_id;
#[cfg(feature = "spectral_pruner")]
pub use irrep_pruner::{
    IrrepPruner, IrrepPrunerConfig, irrep_pruner_from_config, spectral_flatness,
};
#[cfg(feature = "subspace_phase_gate")]
pub use subspace_phase_gate::{
    IntrinsicDimMethod, JacobianSvdScratch, SvdResult, SvdResultScratch, SvdScratch,
    estimate_intrinsic_dim, jacobian_svd_at, jacobian_svd_at_into, numerical_rank,
    participation_ratio, phase_transition_gate, thin_svd, thin_svd_into,
};

#[cfg(feature = "group_invariance_probe")]
pub use group_invariance_probe::{
    GroupAction, Matrix, SubgroupClass, SubgroupReport, classify_subgroup, classify_subgroup_with,
    commutant_basis, commutant_binary_association, commutant_of_matrices, commutant_shift,
    discover_subgroup, discover_subgroup_into, invariance_score, score_concentration,
    score_variance,
};

#[cfg(feature = "causal_head_importance")]
pub use causal_head_importance::{
    ScaleNormalizedFusion, SpanLogitDiffReadout, direct_effect_importance,
    fuse_across_capabilities, indirect_effect_importance, partition_by_causal_score,
    per_capability_score,
};

#[cfg(feature = "adaptive_causal_calibration")]
pub use causal_head_importance::{adaptive_partition, suspect_indices};

#[cfg(feature = "cross_stage_relocation")]
pub mod cross_stage_relocation;
#[cfg(feature = "cross_stage_relocation")]
pub use cross_stage_relocation::{
    ClusterClass, PermeationMap, RelocateOp, RelocatePair, RelocatingForward, permeation_scan_into,
    permeation_scan_square_into,
};

#[cfg(feature = "latent_trajectory_geometry")]
pub use latent_trajectory_geometry::{
    BifurcationResult, LatentTrajectoryGeometry, bifurcation_ratio, fast_acos, from_states,
    from_states_into,
};

#[cfg(feature = "swe_trajectory_freeze")]
pub use swe_trajectory_freeze::{
    FrozenAttempt, FrozenValueAttempt, GeometrySummaryEncoder, SWTF_MAGIC, SWTF_VERSION,
    StateMagnitudeEncoder, SweTrajectoryFreezer, TrajectoryFreezeEnvelope, derive_directions,
    derive_directions_and_centroid,
};

#[cfg(feature = "latent_confounder_audit")]
pub use latent_confounder_audit::{AuditScratch, LatentConfounderAudit, audit_confounders};

#[cfg(feature = "interpolation_geometry")]
pub use interpolation_geometry::{
    EuclideanLatentSpace, FixtureRng, GaussianMixtureSpace, ImauveScore, InterventionReport,
    LatentSpace, imauve_score, intervention_battery,
};

#[cfg(feature = "viable_manifold_graph")]
pub use viable_manifold_graph::{
    ClosurePredicate, GraphBuildConfig, SafeManifoldGraph, ViabilityPredicate, VolumeFieldConfig,
    build_safe_manifold_graph, manifold_curiosity_walk, manifold_geodesic, manifold_random_walk,
    pullback_volume,
};

#[cfg(feature = "certified_frontier")]
pub use certified_frontier::{
    CertifiedFrontier, DilationFeasibility, DualPosteriorBuffer, FrontierCell, FrontierConfig,
    LinearPosterior, PosteriorBuffer, SIGMOID_LIPSCHITZ, SPHERE_EXCLUSION_MAX_CENTERS,
    SphereExclusion, advance_horizon, beta_mean_variance, beta_union_bound, confidence_schedule,
    laurent_massart_radius, linear_information_gain, prefer_dual, should_advance,
    sphere_exclusion_coverage, spherical_cap_bound, vendi_diversity,
};

#[cfg(all(feature = "certified_frontier", feature = "viable_manifold_graph"))]
pub use certified_frontier::{CertifiedManifoldGraph, certified_manifold_graph};

#[cfg(feature = "ac_prefix")]
pub use ac_prefix::{AcPrefix, AcPrefixMask};

#[cfg(feature = "ac_prefix")]
pub use ac_prefix::{
    keyed_gumbel_max_sample, keyed_gumbel_max_sample_masked, keyed_gumbel_max_sample_truncated,
    keyed_gumbel_noise, keyed_posterior_gumbel_noise, truncation_keep_mask,
};

#[cfg(feature = "flow_field_nav")]
pub mod flow;
#[cfg(feature = "flow_field_nav")]
pub use flow::{
    FlowField, FlowFieldCache, FlowFieldConfig, LeoPotentialGrid, blend_steering, fft_smooth,
    fft_smooth_into, flow_steering, inflate_obstacles, should_use_flow_field,
};

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

#[cfg(feature = "merkle_octree")]
pub mod merkle;
#[cfg(feature = "merkle_octree")]
pub use merkle::{
    HASH_SIZE, MERKLE_OCTREE_DEPTH, MERKLE_OCTREE_INTERNAL, MERKLE_OCTREE_LEAVES,
    MERKLE_OCTREE_NODES, MerkleOctree, MerkleProof,
};

#[cfg(feature = "merkle_octree")]
pub mod curator;
#[cfg(feature = "merkle_octree")]
pub use curator::{
    CuratorArm, CuratorBandit, CuratorVerdict, FrozenTarget, MerkleEnvelope, MerkleFrozenStore,
    verification_weight,
};

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

#[cfg(feature = "compression_drafter")]
pub mod compression_drafter;
#[cfg(feature = "compression_drafter")]
pub use compression_drafter::{CompressionDrafter, Lz4FlexDrafter};

#[cfg(feature = "babel_codec")]
pub mod babel_codec;
#[cfg(feature = "babel_codec")]
pub use babel_codec::{
    BabelCodec, BabelCommitment, BabelPair, CompressedLatent, FixedRuleTextCodec,
    SigmoidLatentCodec,
};

#[cfg(feature = "analytic_lattice")]
pub mod analytic_lattice;

#[cfg(feature = "lattice_memory")]
pub mod lattice_memory;
#[cfg(feature = "analytic_lattice")]
pub use analytic_lattice::{
    ChainError, ComposerCtx, LatticeVector, PlasmaDraft, RederiveOp, TransportOperator,
    apply_operator_into, audit::AuditReport, audit::spectral_audit, batch_compose_chain,
    batch_compose_chain_into, compose_chain, compose_chain_into, decoder::direction_vector_decode,
    decoder::direction_vector_decode_into, decoder::direction_vector_decode_slice,
};

#[cfg(feature = "funcattn")]
pub mod funcattn;
#[cfg(feature = "funcattn")]
pub use funcattn::{
    FuncAttnBasis, FuncAttnConfig, FuncAttnError, FuncAttnScratch, compute_basis_into,
    funcattn_forward, pre_rotate_basis_weights_into, solve_convex_combo_dual,
};

#[cfg(feature = "funcattn_structured_basis")]
pub use funcattn::{make_dct_log_basis, make_haar_packet_basis};

#[cfg(feature = "functional_substitution_gate")]
pub mod functional_substitution;
#[cfg(feature = "functional_substitution_gate")]
pub use functional_substitution::{HeadSubstitutionGate, iou, worst_case_behavior_delta};

#[cfg(feature = "cross_resolution_transport")]
pub mod cross_resolution;
#[cfg(feature = "cross_resolution_transport")]
pub use cross_resolution::{
    CrossResScratch, CrossResolutionBases, CrossResolutionError, project_to_spectral_into,
    reconstruct_from_spectral_into, transport_cross_domain_cross_resolution_into,
    transport_cross_resolution, transport_cross_resolution_into,
};

#[cfg(feature = "latent_field_steering")]
pub mod latent_steering;
#[cfg(feature = "latent_field_steering")]
pub use latent_steering::{
    BELIEF_AROUSAL, BELIEF_CALM, BELIEF_DESPERATION, BELIEF_DIM, BELIEF_FEAR, BELIEF_VALENCE,
    FieldSupport, LatentField, LatentSteeringError, LatentSteeringVector, apply_field_to_crowd,
    apply_latent_steering, apply_latent_steering_weighted, kernel_weight,
};

#[cfg(feature = "subspace_steering")]
pub mod subspace_steering;
#[cfg(feature = "subspace_steering")]
pub use subspace_steering::{
    SubspaceSteeringError, SubspaceSteeringField, apply_subspace_steering, block_energy,
    compute_block_commitment, walk_manifold,
};

#[cfg(feature = "region_subspace_steering")]
pub mod region_subspace;
#[cfg(feature = "region_subspace_steering")]
pub use region_subspace::{
    RegionDecomposition, RegionSubspaceError, RegionSubspaceField, compute_field_commitment,
    reconstruct,
};

#[cfg(feature = "phase_rotation_coupling")]
pub mod phase_rotation;
#[cfg(feature = "phase_rotation_coupling")]
pub use phase_rotation::{
    PhaseRotationError, PhaseRotationGate, PhaseRotationScratch, compute_phase_from_projection,
    compute_phase_per_channel_into, phase_rotation_gate_into,
};

#[cfg(feature = "grapem_rodrigues")]
pub mod grapem;
#[cfg(feature = "grapem_rodrigues")]
pub use grapem::{GrapemError, Rank2Plane, grapem_apply_into};

#[cfg(feature = "position_group_action")]
pub mod position_group_action;
#[cfg(feature = "position_group_action")]
pub use position_group_action::{
    AlibiAction, FoxAction, GrapeMAction, NopeAction, PositionGroupAction, RopeAction, WallAction,
};

#[cfg(feature = "rotary_value_embedding")]
pub mod rotary_value_embedding;

#[cfg(feature = "grape_ap_vector")]
pub mod grape_ap;
#[cfg(feature = "grape_ap_vector")]
pub use grape_ap::{GrapeApError, GrapeApGate, RotationSchedule, log_sigmoid};

#[cfg(feature = "grape_joint_lift")]
pub mod grape_joint_lift;
#[cfg(feature = "grape_joint_lift")]
pub use grape_joint_lift::{GrapeJointLift, JointLiftError, softplus};

#[cfg(feature = "spherical_steering")]
pub mod spherical_steering;
#[cfg(feature = "spherical_steering")]
pub use spherical_steering::{
    SlerpError, SlerpScratch, slerp_steering_into, spherical_steering_into, vmf_confidence_gate,
};

#[cfg(feature = "sphere_sampling")]
pub mod sphere_sampling;
#[cfg(feature = "sphere_sampling")]
pub use sphere_sampling::{
    COT_FLOOR, EXP_MAP_FLOOR, SphereError, TRANSPORT_FLOOR, jacobian_logdet_cot_correction,
    parallel_transport_householder_into, sphere_exp_map_into,
};

#[cfg(feature = "mag_mining")]
pub mod mag;
#[cfg(feature = "mag_mining")]
pub use mag::{
    DataSet, MagDirection, MagError, MagOperator, RankEntry, TransferMetric, apply_operator,
    calibrate_alpha, mine_contrast_direction, mine_direction, rank_candidates,
    reconstruction_error, transfer_score,
};

#[cfg(feature = "chunked_content_store")]
pub mod content_store;
#[cfg(feature = "chunked_content_store")]
pub use content_store::{
    BlobId, ChunkFetcher, ChunkRange, ChunkedContentStore, ChunkerConfig, ChunkingStrategy,
    FastCdcChunker, FixedSizeChunker, InMemoryChunkedStore, MerkleProof as BinaryMerkleProof,
    StoreStats, build_binary_merkle_proof, build_binary_merkle_root, verify_binary_merkle_proof,
};

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
    mining::{SleepCycleClosureReport, fold_cdg_at_sleep_cycle, mine_motifs_at_sleep_cycle},
    motif::{
        FixedU32Set, MAX_MOTIF_EDGES, MAX_MOTIF_NODES, Motif, MotifMiner, RING_BUFFER_K,
        enumerate_subgraph_hashes,
    },
    serialize_postcard,
    trace::{DEFAULT_TRACE_CAPACITY, NodeId, PtgRecorder},
};

#[cfg(feature = "ptg_functor_edges")]
pub use closure::{FunctorEdgeParams, FunctorPtg, apply_functor_edge_into, functor_edge_gate};

pub mod data_probe;
#[cfg(feature = "sink_aware_attn")]
pub use data_probe::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};

#[cfg(feature = "mi_est")]
pub mod mi;

#[cfg(feature = "ict_branching")]
pub mod ict;
#[cfg(feature = "ict_branching")]
pub use ict::{
    AcceptanceForecastH2, BranchingDetector, BranchingReport, branching_point_mask,
    branching_point_mask_into, collision_purity, collision_purity_into, is_critical_branching,
    js_divergence, js_divergence_batch, renyi_h2, shannon_h1,
};

#[cfg(feature = "induced_cwm")]
pub mod induced_cwm;
#[cfg(feature = "induced_cwm")]
pub use induced_cwm::{
    BeliefInferenceFn, CwmCommitment, InducedCwmKernel, TransitionTestFailure, TransitionUnitTest,
    make_transition_tests_from_trajectory, verify_transition,
};

#[cfg(feature = "induced_cwm_ismcts")]
pub use induced_cwm::{InformationSet, NodeStats, ismcts_search_with_inference};

#[cfg(feature = "bisimulation_operator_inference")]
pub mod bisimulation;
#[cfg(feature = "bisimulation_operator_inference")]
pub use bisimulation::{
    BisimulationQuotient, OperatorDef, OperatorLabel, OperatorSchema, Plan, QuotientEdge,
    StateClassId, StateId, Transition, TransitionGraph, TransitionGraphBuilder, partition_refine,
    plan as bisimulation_plan,
};

#[cfg(feature = "operator_consistency")]
pub use bisimulation::consistency::{
    ApplicationOutcome, ConsistencyGateConfig, ConsistencyRegime, ConsistencyReport,
    PromotionVerdict, promotion_verdict, rule_consistency,
};

#[cfg(feature = "occupancy_ratio")]
pub mod occupancy;

// personality_composition — Personality-Weighted Layer Composition (Plan 297, Research 276; DEFAULT-ON per Plan 297 Phase 4);
// substrate in the katgpt-personality crate (Issue 007 Phase E Tier 2 #5); re-exported here for backwards compatibility.
#[cfg(feature = "personality_composition")]
pub use katgpt_personality as personality_composition;
#[cfg(feature = "personality_composition")]
pub use personality_composition::{
    ArchetypeLabel, EntityCognitionComposition, LayerDirectionSource, PersonalityConfig,
    PersonalitySnapshot, PersonalityWeightedComposition, sigmoid as personality_sigmoid,
    sigmoid_into as personality_sigmoid_into,
};

#[cfg(feature = "committed_field_blend")]
pub mod committed_field_blend;
#[cfg(feature = "committed_field_blend")]
pub use committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend, TriArchetypeBlend};

#[cfg(feature = "jsd_topk")]
pub mod jsd_topk;

#[cfg(feature = "meld")]
pub mod meld;
#[cfg(feature = "meld")]
pub use meld::{MeldCompose, MeldLaw};

#[cfg(feature = "variable_rank_domain_expert")]
pub mod variable_rank_domain_expert;

#[cfg(feature = "engram")]
pub mod engram;
#[cfg(feature = "engram")]
pub use engram::{
    CacheResult, CacheTier, ColdFetcher, EngramConfig, EngramHash, EngramHotSwap, EngramTable,
    EngramTableBuilder, EngramTableId, HashHead, IDENTITY_KERNEL, InMemoryEngramTable, K_MAX,
    SigmoidFusionConfig, StagingEngramTable, StagingError, SurjectiveMap, SurjectiveMapLoadError,
    TokenId, TokenizerSpec, ZipfianCacheHierarchy, ZipfianStats, ZipfianStatsSnapshot,
    build_merkle_root, build_surjective_map, compress_token, conv_causal_dyn_into,
    conv_causal_into, fuse_into_hidden_state, multi_head_hash, rmsnorm_into, sigmoid_fuse_into,
    sigmoid_fuse_multi_branch_into, try_compress_token,
};

#[cfg(feature = "engram_privilege")]
pub use engram::{
    CreditAssignment, PrivilegeConfig, PrivilegeLedger, PrivilegeTrace,
    fuse_into_hidden_state_privileged, sigmoid_fuse_scaled_into,
};

#[cfg(feature = "product_key_memory")]
pub mod product_key_memory;
#[cfg(feature = "product_key_memory")]
pub use product_key_memory::{
    D_K_FLOOR, PkEntry, PkQuery, PkmScratch, ProductKeyMemory, SQRT_N_FLOOR, ScoreFn, score_dot,
    score_idw,
};

// Shared tabular-kernel helpers (one-hot fast path + SIMD row dot) —
// hoisted verbatim from mop::solve (Plan 573) so hmm_control (Plan 590)
// consumes the same bit-identity-hardened primitives without implying the
// mop feature. Compiled only when a consumer feature is on (a bare
// --no-default-features build must not see dead_code).
#[cfg(any(feature = "mop_path_entropy", feature = "hmm_homeostasis"))]
pub(crate) mod tabular_kernel;

#[cfg(feature = "mop_path_entropy")]
pub mod mop;
#[cfg(feature = "mop_path_entropy")]
pub use mop::{MopConfig, MopConfigError, MopScratch, MopSolution, MopSolver};

#[cfg(feature = "hmm_homeostasis")]
pub mod hmm_control;
#[cfg(feature = "hmm_homeostasis")]
pub use hmm_control::{HmmControlSolver, HmmInputError, HmmSolution};

#[cfg(feature = "gw_alignment")]
pub mod gw_alignment;
#[cfg(feature = "gw_alignment")]
pub use gw_alignment::{
    GW_ITERS, GW_MAX, GW_SCORE_BETA, GW_TAIL_PASSES, GwError, GwScratch, gw_coupling,
    score_from_loss,
};

#[cfg(feature = "product_key_memory_freeze")]
pub use product_key_memory::FrozenProductKeyMemory;

#[cfg(feature = "product_key_memory_episodic")]
pub use product_key_memory::PkmEpisodicStore;

#[cfg(feature = "gain_cost_halt")]
pub mod gain_cost_halt;
#[cfg(feature = "gain_cost_halt")]
pub use gain_cost_halt::{
    GainCostLoopHalter, HaltDecision, HaltReason, angular_change, hidden_erank, step_size,
};

#[cfg(feature = "qrr_convergence")]
pub mod qrr_convergence;
#[cfg(feature = "qrr_convergence")]
pub use qrr_convergence::{
    AlphaReplayVerdict, EnergySplit, ModeEstimate, ModeFitter, ModeKind, OscillationVerdict,
    R1Certificate, SylvesterBlocks, alpha_replay_check, energy_split, replay_probe_state,
    settling_steps, solve_sylvester, tolerance_for_budget,
};

#[cfg(feature = "saddle_escape")]
pub mod saddle_escape;

#[cfg(feature = "saddle_escape")]
pub use saddle_escape::{
    FlipDetector, HaltOutcome, SaddleEscapeGate, TrapConfig, TrapObservables, apply_kick,
};

#[cfg(feature = "guided_width_rollouts")]
pub mod guided_width;

#[cfg(feature = "marginal_rewind")]
pub mod marginal_rewind;
#[cfg(feature = "marginal_rewind")]
pub use marginal_rewind::{RewindPlan, gamma_dial_plan, rewind_into, rewind_plan};

#[cfg(feature = "metabolic_gate")]
pub mod metabolic_gate;
#[cfg(feature = "metabolic_gate")]
pub use metabolic_gate::{
    MetabolicGate, defector_starves, execution_share, metabolic_drag_threshold, steal_lossy,
};

#[cfg(feature = "structural_cot_halt")]
pub mod structural_cot_halt;
#[cfg(feature = "structural_cot_halt")]
pub use structural_cot_halt::{
    BacktrackRevisitHalt, ClassifiedPattern, HaltPolicy, HaltVote, Pattern, SelfLoopHalt,
    StructuralHaltDecision, StructuralHaltReason, StructuralTraceMonitor, StructuralTransition,
    compose_votes, normalized_answer_hash,
};

#[cfg(feature = "cadence_gate")]
pub mod convergence_cadence;
#[cfg(feature = "cadence_gate")]
pub use convergence_cadence::{CadenceConfig, CadenceVerdict, ConvergenceCadence};

#[cfg(feature = "kl_depth_probe")]
pub mod loop_depth_probe;

#[cfg(feature = "loop_alignment_probe")]
pub mod loop_alignment_probe;

#[cfg(feature = "regime_probe")]
pub mod regime_probe;
#[cfg(feature = "regime_probe")]
pub use regime_probe::{
    BasinReport, BasinScratch, EntropyGapReport, FrozenRenovator, basin_probe, basin_probe_into,
    basin_radius_bound, basin_radius_from_kappa, conditional_entropies_into,
    conditional_entropy_nats, entropy_gap, entropy_gap_into, gamma_capacity, kappa_max,
    kappa_max_bisection, mean_conditional_entropy, phi_cdf, phi_pdf,
};

#[cfg(feature = "set_attention")]
pub mod set_attention;
#[cfg(feature = "set_attention")]
pub use set_attention::{
    SetAttentionConfig, SetAttentionError, identity, identity_into, identity_projection,
    identity_projection_into, set_sigmoid_attention_into, set_sigmoid_attention_wide_into,
};
#[cfg(feature = "clr_weighted_set_attention")]
pub use set_attention::{clr_reliability_scores, clr_weighted_set_attention_into};

#[cfg(feature = "set_refine")]
pub mod set_refine;
#[cfg(feature = "set_refine")]
pub use set_refine::{
    SetRefineConfig, SetRefineError, SetRefineScratch, gate_weight, refine_state_into,
};

// depth_invariance — chain-depth classifier + MagnitudeReg (Plan 306, Research 286, arXiv:2605.09992; DEFAULT-ON
// per Plan 306 T7.4). Spun out to the `katgpt-types` crate; re-exported here for backwards compatibility.
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance;
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance::{
    DepthInvarianceConfig, DepthInvarianceDiagnostic, DepthInvarianceKind, MagnitudeRegularization,
    Scratch, apply_magnitude_regularization, classify_chain, classify_chain_batched,
};

#[cfg(any(
    feature = "karc_forecaster",
    feature = "geometric_product",
    feature = "tucker_factorization",
    feature = "svd_cca",
    feature = "twist_smc",
    feature = "mi_est",
    // Both joined the rule late and broke it silently: each consumes
    // `linalg::ridge_solve` and neither gated this `pub mod`, so
    // `--no-default-features --features <either>` failed E0433. Found by the
    // Issue 701 R1b default-on isolation sweep (2026-09-01) — the FIRST run of
    // that scope, and it found them in 4.1 minutes. Nothing else catches this
    // class: `--all-features` compiles the union, where some other consumer
    // always brings `linalg` in.
    feature = "velocity_field_ensemble",
    feature = "hebbian_kernel_memory",
    // Issue 707: `tpr::als` consumes `ridge_solve_direct_f32` / `cholesky_f32`
    // / `chol_solve_f32` for its four closed-form ALS blocks and the cached
    // projection factor — same rule, joined at birth rather than late.
    feature = "tpr",
    // Issue 736: `leakage_probe::transport` consumes `symmetric_eig` (f64
    // eigenvalues+vectors) for PCA whitening and the orthogonal-Procrustes
    // polar factor — same rule, joined at birth rather than late.
    feature = "leakage_probe",
    // Issue 763: `lif_graph::fit_readout` consumes `ridge_solve_direct_f64`
    // (the KARC-precedent closed-form readout) — same rule, joined at birth.
    feature = "lif_graph",
    // Issue 767: `mb_value`'s G1 floor test consumes `ridge_solve_direct_f64`
    // (the ridge-batch-on-the-same-code baseline) — joined at birth per the
    // same rule (the crate must compile when only this feature is on).
    feature = "mb_value",
    // Issue 839: `linalg::kron_tile` IS a linalg submodule, so it gates this
    // `pub mod` too — joined at birth per the same rule.
    feature = "kron_tile",
    // Plan 607 T3: `state_option_scoring::head` consumes
    // `ridge_solve_direct_f64` for the corpus-fitted linear head (the KARC
    // Plan-308 fit math) — joined at birth per the same rule.
    feature = "state_option_scoring"
))]
pub mod linalg;

#[cfg(feature = "karc_forecaster")]
pub mod karc;
#[cfg(feature = "karc_forecaster")]
pub use karc::{
    BSplineBasis, ChebyshevBasis, DelayRing, FitError, FourierBasis, KarcBasis, KarcForecaster,
    KarcScratch, LowRankFitScratch, chunked_gram_into, feature_expand, feature_expand_higher_order,
    forecast_low_rank_apply, higher_order_feature_count, low_rank_fit, low_rank_fit_warm_start,
};

#[cfg(feature = "karc_regime_gate")]
pub use karc::regime_gate::{KarcRegime, KarcRegimeGate, RegimeVerdict};

#[cfg(feature = "karc_batched_matvec")]
pub use karc::batched::{KarcBatchForecaster, karc_batched_matvec_into};

#[cfg(feature = "karc_lod_tier")]
pub use karc::lod_tier::{KarcLodTier, is_identity_projection, project_wout_lod_into};

#[cfg(feature = "karc_forecaster")]
pub mod karc_dp;
#[cfg(feature = "karc_forecaster")]
pub use karc_dp::{KarcDpNoiseConfig, apply_dp_noise_to_wout};

#[cfg(feature = "lif_graph")]
pub mod lif_graph;

#[cfg(feature = "mb_value")]
pub mod mb_value;

#[cfg(feature = "hope_capacity")]
pub mod hope;

#[cfg(feature = "hebbian_kernel_memory")]
pub mod hebbian_kernel_memory;

#[cfg(feature = "spectral_pencil")]
pub mod spectral_pencil;

#[cfg(feature = "orthogonal_factorization")]
pub mod orthogonal_factorization;

#[cfg(feature = "cp_hopfield")]
pub mod cp_hopfield;

#[cfg(feature = "transformer_inversion")]
pub mod inversion;
#[cfg(feature = "grad_policy")]
pub use inversion::InversionGradient;
#[cfg(feature = "transformer_inversion")]
pub use inversion::{
    AcceptanceRegion, InversionConfig, InversionError, InversionForward, InversionPolicy,
    InversionResult, ObservedStates, RandomPolicy, accept_observation, accept_observation_into,
    invert_sequence, invert_sequence_into,
};

#[cfg(feature = "similarity_inference")]
pub mod similarity_inference;
#[cfg(feature = "similarity_inference")]
pub use similarity_inference::{
    JointActionHistory, PayoffMatrix, SimilarityError, SimilarityPosterior, canonical_pd,
    embedded_best_response, embedded_best_response_into,
};

#[cfg(feature = "arg_protocol")]
pub mod arg;
#[cfg(feature = "arg_protocol")]
pub use arg::{
    AccessScope, CandidateIntent, CandidateKind, CompareFn, CompareResult,
    DEFAULT_AUTO_COMMIT_THRESHOLD, Evidence, EvidenceId, GainComponents, InfoKey,
    InfoOutcomeStatus, InfoRegistry, InfoType, InfoUnit, LabelId, LabelSet, LabelSignature,
    LifecycleState, MatchResult, MatchScratch, OfflineCandidateScorer, PayloadHash,
    PayloadHashCompare, PolicyConstraints, PolicyDecision, PolicyEnvelope, PolicyState, Provenance,
    RedirectTable, ResponseMode, ScoredCandidate, ShouldProceed, TaxonomyKind, TaxonomyNode,
    TaxonomyValidator, TypedOfflineCandidate, ValidationError, ValidationResult, ValidationScratch,
};

#[cfg(feature = "non_interference_branches")]
pub mod branching;
#[cfg(feature = "non_interference_branches")]
pub use branching::{
    AssignError, AssignResult, BranchBank, BranchId, BranchLifecycle, BranchRouter, BranchStats,
    BudgetCompiler, CognitiveBranch, CompiledContext, CompiledItem,
    DEFAULT_ASSIGN_MAX_INTERFERENCE, DEFAULT_BUDGET_BYTES, DEFAULT_MAX_BRANCHES,
    DEFAULT_ORTHOGONAL_EPSILON, DEFAULT_PROJECTION_DIM, DEFAULT_QUARANTINE_CENTROID_THRESH,
    DEFAULT_TAU_CURIOSITY, DEFAULT_TAU_JACCARD, DEFAULT_TAU_SNAP, DEFAULT_TAU_WRITE, EpisodicEntry,
    FailureEntry, NonInterferenceProjection, PriorityTier, ProceduralRule, RetrievedMaterials,
    RouteMode, RouteResult, VerifierGate, WriteDecision, max_orthogonal_branches,
};

#[cfg(feature = "local_branch_routing")]
pub mod branch_routing;
#[cfg(feature = "local_branch_routing")]
pub use branch_routing::{
    ColliderRouterAdapter, DotProductRouter, PostCandidateRouter, PreservationScorer,
};

// sleep_time — Sleep-Time Query Anticipator (Plan 334, Research 318, arXiv:2504.13171; opt-in until the Plan
// 341 GOAT clears on a real game corpus). Substrate in the katgpt-sleep crate (Issue 007 Phase E Tier 2 #6); re-exported for back-compat.
#[cfg(feature = "sleep_time_anticipation")]
pub use katgpt_sleep as sleep_time;
#[cfg(feature = "sleep_time_anticipation")]
pub use sleep_time::{
    AmortizationCostModel, AnticipatedQueryDir, AnticipatedQuerySet, AnticipatedSlot,
    ConsumeMatchMode, DEFAULT_LATENCY_PREMIUM, DotPredictabilityScorer, IdentityFunctorOp,
    PredictabilityScorer, SLEEP_TIME_DEFAULT_K, SleepTimeAnticipator, SleepTimeComputeOp,
    SleepTimeScratch, commit_direction, consume, consume_gate, consume_gate_with_match_mode,
    consume_with_match_mode,
};

#[cfg(feature = "paired_loss_diagnostic")]
pub mod paired_loss;
#[cfg(feature = "paired_loss_diagnostic")]
pub use paired_loss::{
    ClassGapReport, ClassGapRow, ClassSizeBound, CopyNGramTagger, FilterKind, FilterScratch,
    PairedLossGap, TokenClass, TokenTagger,
};

#[cfg(feature = "temp_loss_fingerprint")]
pub mod diversity;
#[cfg(feature = "temp_loss_fingerprint")]
pub use diversity::temp::{
    LossKernel, blake3_noise_fill, extrapolated_snapshot_schedule, lipschitz_gradient_bound,
    pairwise_bound, perturbed_loss_vector, select_diverse_subset, select_diverse_subset_in_place,
};

#[cfg(all(feature = "temp_loss_fingerprint", feature = "qmc_sampling"))]
pub use diversity::temp::extrapolated_snapshot_schedule_qmc;

#[cfg(feature = "manifold_bandit")]
pub mod manifold_bandit;

#[cfg(feature = "mean_field_regime")]
pub mod mean_field;
#[cfg(feature = "mean_field_regime")]
pub use mean_field::{
    DEFAULT_CLASSIFIER, HopfParams, MeanFieldOverlap, Regime, RegimeClassifier, hopf_boundary,
    static_boundary,
};

#[cfg(feature = "factorized_action")]
pub mod factorized_action;
#[cfg(feature = "factorized_action")]
pub use factorized_action::{
    AggregatorType, EffectCodebook, FactorizedActionLatent, FilmProjectionBank, MAX_K, MAX_PATCHES,
    TransitionFactors, aggregate_action_latent_into, factor_token_into, finalize_factors,
    fit_codebook_kmeans_into, motion_input_velocity_into, patchify_1d, relevance_score,
};

#[cfg(feature = "direction_bank_audit")]
pub mod ooo_audit;
#[cfg(feature = "direction_bank_audit")]
pub use ooo_audit::{
    AuditScratch as OooAuditScratch, BankAudit, CurationResult, OooAuditConfig, audit_bank_into,
    cross_ooo, exemplar_rbf_sim_into, greedy_curate, ooo_score, select_meis_into,
};

#[cfg(feature = "velocity_field_ensemble")]
pub mod velocity_field_ensemble;
#[cfg(feature = "velocity_field_ensemble")]
pub use velocity_field_ensemble::{
    ClosureField, EnsembleFitScratch, Schedule, VelocityField, VelocityFieldEnsemble,
    accumulate_pair_into, stochastic_interpolant_step_into,
};

#[cfg(feature = "velocity_field_disagreement")]
pub mod velocity_field_disagreement;
#[cfg(feature = "velocity_field_disagreement")]
pub use velocity_field_disagreement::{VfdScore, VfdScratch, VfdVarianceSignal, vfd_score_into};

pub mod welford;
pub use welford::WelfordVariance;

#[cfg(feature = "perturbation_ensemble")]
pub mod perturbation_ensemble;

// ── Phase 10 absorption (Proposal 003, 2026-07-04): modules moved from katgpt-rs/src/.
// Always-on (no feature gate):
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
pub mod alloc;
pub mod cumprodsum; // Cumprodsum primitive (Plan 263) — always-on
pub mod trigger_gate; // Compute-tier trigger gate — always-on
// ── Phase 12 absorption (Proposal 003, 2026-07-04): more modules moved from katgpt-rs/src/.
// Feature-gated (mirror root feature names):
#[cfg(feature = "critical_interval_gate")]
pub mod dllm_solver; // Discrete Critical Interval Solver Switching (Plan 222)

#[cfg(feature = "decode_order_metrics")]
pub mod dllm;

#[cfg(feature = "entropy_bounded_commit")]
pub mod entropy_bounded_commit;

#[cfg(feature = "decode_order_metrics")]
pub mod anchor_score;

#[cfg(feature = "decode_order_metrics")]
pub use ugc_schedule::CertifiedSpineView;

#[cfg(feature = "decode_order_metrics")]
pub use set_diffusion_schedule::{
    ORDER_STATS_TO_W_TABLE, confidence_threshold_eligible, inverse_lambda_slot_counts,
    predict_w_from_order_stats, predict_w_residual,
};
#[cfg(feature = "modality_pruned_load")]
pub mod pipeline_pruner; // Pipeline Pruner — modality-aware inference pipeline selection (Plan 227 Phase 3)
// ── Phase 12 T4.3: folder moves from katgpt-rs/src/.
#[cfg(feature = "breakeven_routing")]
pub mod breakeven;
#[cfg(feature = "closed_unit_compaction")]
pub mod compaction; // Closed-Unit Compaction Gate — CUCG (Plan 333)
#[cfg(feature = "cubical_nerve")]
pub mod cubical_nerve; // CubicalNerve CAT(0) cubical complexes (Plan 252 Phase 3)
#[cfg(feature = "mux_latent_context")]
pub mod mux_latent; // MUX-Latent Context Compression (Research 211, Plan 238) // Breakeven complexity cost-aware routing (Plan 250)
// Feature-gated (mirror root feature names):
#[cfg(feature = "cce_moderator")]
pub mod cce;
#[cfg(feature = "llmexec_guard")]
pub mod llmexec_guard;
#[cfg(feature = "memory_soup_lora")]
pub mod memory_soup_lora;
#[cfg(feature = "mux_demux")]
pub mod mux_demux;
#[cfg(feature = "salience_tri_gate")]
pub mod salience;
#[cfg(feature = "salience_tri_gate")]
pub use salience::{
    DelegateToken, FoldbackTarget, SalienceDecision, SalienceTriGate, SilenceToken,
};
#[cfg(feature = "channel_simd_align")]
pub mod channel_simd;

#[cfg(feature = "bf16_simd")]
pub mod bf16_convert;
#[cfg(feature = "skill_opt")]
pub mod skill_opt;
#[cfg(feature = "ssd_block")]
pub mod ssd_block;

#[cfg(feature = "gdn_tree_verify")]
pub mod gdn_tree_verify;

#[cfg(feature = "tilr_invariant_subspace")]
pub mod tilr;
#[cfg(feature = "tilr_invariant_subspace")]
pub use tilr::{
    TilrError, TilrScratch, check_orthonormal, tilr_refine, tilr_refine_apply, tilr_refine_into,
};

#[cfg(all(feature = "tilr_invariant_subspace", feature = "subspace_phase_gate"))]
pub use tilr::discover_invariant_subspace;

#[cfg(feature = "basis_split")]
pub mod basis_split;
#[cfg(feature = "basis_split")]
pub use basis_split::{
    BasisSplitError, PairDifferenceMoment, SplitReport, UnidentifiedReason, Verdict, Whitening,
    build_zca_whitener, check_rank_gate, held_out_leakage, project_invariant_into, split,
};

#[cfg(all(feature = "manifold_erasure", feature = "subspace_phase_gate"))]
pub mod manifold_erasure;

#[cfg(all(feature = "quant_error_lora", feature = "subspace_phase_gate"))]
pub mod quant_error_lora;
#[cfg(all(feature = "manifold_erasure", feature = "subspace_phase_gate"))]
pub use manifold_erasure::{
    ManceConfig, ManceError, ManceScratch, ManceStepInfo, ManceTangentCache,
    covmatch_second_moment_into, leace_first_moment_into, mance_plus_plus_step_into,
    mance_plus_step_into, manifold_erasure_loop_cached_into, manifold_erasure_loop_into,
    manifold_erasure_step, manifold_erasure_step_cached_into, manifold_erasure_step_into,
};

#[cfg(feature = "multi_agent_path")]
pub mod multi_agent_path;

#[cfg(all(feature = "poincare_navigator", feature = "subspace_phase_gate"))]
pub mod poincare;
#[cfg(all(feature = "poincare_navigator", feature = "subspace_phase_gate"))]
pub use poincare::{
    FitConfig, LATENT_DIM_MAX, PHI_HIDDEN_DEFAULT, PHI_OUT_DEFAULT, PoincareAdapter,
    PoincareFitError, RIDGE_ALPHA_DEFAULT, TARGET_DIM_MAX, accumulate_pinv_into, eval_phi_into,
    fit_poincare_adapter, poincare_multi_step_into, poincare_navigate_into,
};

#[cfg(feature = "phase_separation")]
pub mod phase_separation;
#[cfg(feature = "phase_separation")]
pub use phase_separation::{
    circular_distance, from_latent_projection, from_speeds_and_tick, phase_separation,
    phase_separation_all, phase_separation_sorted,
};

#[cfg(feature = "signed_coupling_dynamics")]
pub mod signed_coupling;
#[cfg(feature = "signed_coupling_dynamics")]
pub use signed_coupling::{
    Couplings, InformedCouplings, PAPER_BETA_MINUS_RANGE, PAPER_BETA_PLUS_RANGE,
    PAPER_BETA_ZERO_RANGE, PAPER_TRUTH_GAP_RANGE, SignedGraph, SignedGraphError,
    SusceptibilityAccumulator, crowd_conviction, net_opinion, sample_states_into,
    signed_coupling_update_informed_into, signed_coupling_update_into,
};

#[cfg(feature = "qsg_gossip")]
pub mod qsg_gossip;
#[cfg(feature = "qsg_gossip")]
pub use qsg_gossip::{
    MAX_K as QSG_MAX_K,
    // Aliased at the flat re-export only: factorized_action already exports
    // `MAX_K` at the crate root, and the two gated modules collide under
    // --all-features (E0252 at HEAD 98823631 — Plan 589 landing bug,
    // unblocked by Plan 590's all-features gate leg). The module path
    // `qsg_gossip::MAX_K` is unchanged.
    MessageMode,
    QsgConfig,
    QsgError,
    SIMPLEX_EPS,
    UniformStream,
    qsg_blend_listener_into,
    qsg_disagreement_v,
    qsg_draw_categorical,
    qsg_gossip_run_into,
    qsg_gossip_step_into,
    qsg_init_uniform_into,
    qsg_mean_into,
    qsg_polarization_u,
    qsg_sample_message_into,
    qsg_uncertainty,
    uniform_ordered_pair,
};

#[cfg(feature = "rrq_quant")]
pub mod rrq_quant;
#[cfg(feature = "rrq_quant")]
pub use rrq_quant::{
    BITS_PER_STAGE, CODES_PER_BYTE, DEFAULT_DIRECT_RTN_BITS, DEFAULT_GROUP_SIZE, DEFAULT_N_STAGES,
    KS_FLAG_THRESHOLD, LEVELS_PER_STAGE, PMR_THRESHOLD_2_2, QuantStrategy, RrqStage, RrqWeights,
    peak_to_mean_ratio, select_quant_strategy,
};

#[cfg(feature = "selection_propagation")]
pub mod selection_propagation;
#[cfg(feature = "selection_propagation")]
pub use selection_propagation::{
    PropagationBlend, PropagationConfig, PropagationOutcome, SelectionPropagationScratch,
    propagate_selection_to_fixpoint_into,
};

#[cfg(feature = "sterling_primitives")]
pub mod sterling;

#[cfg(feature = "recirculation")]
pub mod recirculation;

#[cfg(feature = "contrastive_scope")]
pub mod contrastive_scope;

#[cfg(feature = "successor_density_critic")]
pub mod successor_density_critic;

#[cfg(feature = "bounded_target")]
pub mod bounded_target;
#[cfg(feature = "bounded_target")]
pub mod realization_gap;

#[cfg(feature = "anchored_reach")]
pub mod anchored_reach;
#[cfg(feature = "anti_common_mode")]
pub mod anti_common_mode;

#[cfg(feature = "numeric_stability")]
pub mod numeric_stability;

#[cfg(feature = "kinematic_rollout")]
pub mod kinematics;

#[cfg(feature = "stale_residual")]
pub mod stale_residual;

#[cfg(feature = "cond_audit")]
pub mod cond_audit;

#[cfg(feature = "tpr")]
pub mod tpr;

// special_fn — shared Lanczos ln_gamma substrate (Plan 597 T1.1): the one
// ln_gamma in the crate, consumed by best_belief + bmr. pub(crate), gated
// to its consumer set (2026-09-13): under --no-default-features both
// consumers compile away and the ungated module tripped dead_code — it now
// compiles to nothing with them (the cfg-gated-target discipline, not an
// #[allow(dead_code)] blanket).
#[cfg(any(feature = "best_belief", feature = "bmr"))]
pub(crate) mod special_fn;

#[cfg(feature = "slice_tca")]
pub mod slice_tca;

#[cfg(feature = "bmr")]
pub mod bmr;

#[cfg(feature = "slt")]
pub mod slt;

#[cfg(feature = "graph_stable_pool")]
pub mod graph_stable_pool;

#[cfg(feature = "pool_admission")]
pub mod pool_admission;

#[cfg(feature = "rate_control")]
pub mod rate_control;

#[cfg(feature = "escalation_guard")]
pub mod escalation_guard;

#[cfg(feature = "lthash")]
pub mod lthash;

// Test-only `#[global_allocator]` so `alloc::tests::*` pass when running
// `cargo test -p katgpt-core --lib`. Downstream consumers (katgpt-rs root,
// riir-engine, etc.) install their OWN `#[global_allocator]`; this static is
// `cfg(test)` so it does not exist when katgpt-core is consumed as a library
// dep — no double-declare conflict. Mirrors the root crate's
// `static GLOBAL_ALLOC: TrackingAllocator` (src/lib.rs:356).
#[cfg(all(test, any(debug_assertions, feature = "alloc_tracking")))]
#[global_allocator]
static TEST_GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;

#[cfg(feature = "sigmoid_calibration")]
pub mod sigmoid_calibration;

#[cfg(feature = "distance_abstain")]
pub mod distance_abstain;

#[cfg(feature = "gmm_support")]
pub mod gmm_support;

#[cfg(feature = "dirichlet_dist")]
pub mod dirichlet_dist;

#[cfg(feature = "grouped_evidence_weighting")]
pub mod grouped_evidence;

#[cfg(feature = "fitted_anchor_tables")]
pub mod fitted_anchor_table;

#[cfg(feature = "act_channel_moments")]
pub mod act_channel_moments;

#[cfg(feature = "differential_anchor")]
pub mod differential_anchor;

#[cfg(feature = "attention_snr")]
pub mod attention_snr;

#[cfg(feature = "row_logit_floor")]
pub mod row_logit_floor;

#[cfg(feature = "canonical_context")]
pub mod canonical_context;

#[cfg(feature = "fitted_value_tables")]
pub mod fitted_value_table;

#[cfg(feature = "habituation_filter")]
pub mod habituation_filter;

#[cfg(feature = "ladder_gate")]
pub mod ladder_gate;

#[cfg(feature = "chance_puct")]
pub mod chance_puct;

#[cfg(feature = "two_fidelity_bai")]
pub mod two_fidelity_bai;

#[cfg(feature = "exact_mass_admit")]
pub mod exact_mass_admit;

#[cfg(feature = "exact_mass_admit")]
pub mod log_frontier;

#[cfg(feature = "calibration_staleness")]
pub mod calibration_staleness;

#[cfg(feature = "legal_token_set")]
pub mod legal_token_set;

#[cfg(feature = "decision_wire")]
pub mod decision_wire;

#[cfg(feature = "extraction_wire")]
pub mod extraction_wire;

#[cfg(feature = "runetrace")]
pub mod runetrace;

#[cfg(feature = "kv_sink_window")]
pub mod kv_sink_window;
