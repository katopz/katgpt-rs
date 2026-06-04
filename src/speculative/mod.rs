pub mod budget_compat;
pub mod dd_tree;
pub mod dflash;
pub mod drafter_lora;
pub mod prefill;
pub mod residency_audit;
pub mod sampling;
pub mod step;
pub mod types;
pub mod verifier;

#[cfg(feature = "trust_region_spec")]
pub mod trust_region;

#[cfg(feature = "budget_adaptation")]
pub mod budget;

#[cfg(feature = "ppot")]
pub mod ppot;

#[cfg(feature = "bandit")]
pub mod flow_pruner;

#[cfg(feature = "peira_distill")]
pub mod peira_pruner;

#[cfg(feature = "dllm")]
pub mod d2f;

#[cfg(feature = "tri_mode")]
pub mod d2f_verifier;

#[cfg(feature = "tri_mode")]
pub mod diffusion_sampler;

#[cfg(feature = "lattice_deduction")]
pub mod alpha;

#[cfg(feature = "parallel_probe")]
pub mod answer_extract;

#[cfg(feature = "parallel_probe")]
pub mod parallel_probe;

// Re-exports — preserves existing import paths like `speculative::build_dd_tree`
pub use dd_tree::{
    TreeBuilder, build_dd_tree, build_dd_tree_balanced, build_dd_tree_balanced_sde,
    build_dd_tree_pruned, build_dd_tree_screened, build_dd_tree_sde, build_inference_result,
    extract_all_sequences, extract_best_path, extract_best_path_into, extract_candidate_sequences,
    extract_parent_tokens, find_valid_sequence, inject_sde_noise, merge_retrieved_branches,
    par_find_shortest_sequence, par_find_valid_sequence,
};

#[cfg(feature = "thinking_prune")]
pub use dd_tree::build_dd_tree_screened_with_schedule;

#[cfg(feature = "gdsd_distill")]
pub use dd_tree::build_dd_tree_gdsd;

#[cfg(feature = "eqr_convergence")]
pub use dd_tree::ResidualTracker;
#[cfg(feature = "sr2am_configurator")]
pub use dd_tree::entropy_truncate_horizon;
#[cfg(feature = "recfm")]
pub use dd_tree::{
    CrossScaleConfig, branch_velocity_at, build_dd_tree_screened_recfm, cross_scale_consistent,
};
#[cfg(feature = "elf_sde")]
pub use dd_tree::{WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts};
pub use dflash::{
    dflash_predict, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_conditioned,
    dflash_predict_conditioned_with, dflash_predict_parallel, dflash_predict_with,
};
pub use prefill::{
    AttentionScorer, BlockAttentionScorer, PrefillScorer, RandomScorer, UniformScorer,
    block_select, block_select_grid, compress_prompt, compress_prompt_blocks, should_compress,
    speculative_prefill, speculative_prefill_adaptive, speculative_prefill_block,
};
pub use sampling::{
    sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into,
};
pub use step::{speculative_step, speculative_step_verifier};
pub use types::{
    BinaryScreeningPruner, BlockScores, BudgetAdaptation, ConstraintPruner, DDTreeBranchCache,
    DecodeStrategy, DraftEvent, DraftResult, FlashPrefillConfig, NoPruner, NoScreeningPruner,
    PrefillMode, RejectionReason, ScreeningPruner, SdeConfig, SpeculativeContext, TreeNode,
};

#[cfg(feature = "elf_sde")]
pub use types::EarlyStopGate;

#[cfg(feature = "tri_mode")]
pub use types::SelfSpecConfig;

// ── MoE+SD Co-Design (Plan 096, Research 59) ──────────────────
#[cfg(feature = "domain_latent")]
pub use types::RoutingOverlapSnapshot;
#[cfg(feature = "spec_cost_model")]
pub use types::SpecCostSnapshot;
#[cfg(feature = "stability_metrics")]
pub use types::StabilitySnapshot;

// ── Stage-Specialized Decode Paths (Plan 102) ──────────────────
#[cfg(feature = "decode_specialize")]
pub use crate::transformer::DecodeStage;

// ── LDT Lattice Deduction Transformer re-exports (Plan 088, Plan 170) ──
#[cfg(feature = "lattice_deduction")]
pub use alpha::{
    AlphaScreeningPruner, AlphaTarget, ConflictClauseDB, alpha_intersect, is_consistent,
};
#[cfg(feature = "lattice_deduction")]
pub use types::{ConflictDetector, EntropyConflictDetector, LDT_THETA_ELIM, LdtPruneConfig};

// ── SimpleTES re-exports (Plan 086, feature: tes_loop) ────────
#[cfg(feature = "tes_loop")]
pub use types::{TesConfig, TesNode, TrajectoryCredit};
pub use verifier::{SimulatedVerifier, SpeculativeVerifier};

pub use verifier::LeviathanVerifier;

// ── Drafter LoRA re-exports (Plan 117: MTP LoRA Drafter) ──────
pub use drafter_lora::{
    DrafterForwardContext, DrafterLoraWeights, TrainingPair, generate_synthetic_pairs,
    generate_training_pairs_from_replays, load_drafter_lora, save_drafter_lora, train_drafter_lora,
};

#[allow(deprecated)]
pub use step::{
    speculative_step_conditioned, speculative_step_conditioned_with, speculative_step_rollback,
    speculative_step_rollback_with,
};

#[cfg(feature = "sr2am_configurator")]
pub use step::speculative_step_with_configurator;

#[cfg(feature = "bandit")]
pub use flow_pruner::FlowPruner;

#[cfg(feature = "maxsim")]
pub use prefill::block_score_maxsim;

// ── D2F Re-exports (Plan 066 Phase 2) ─────────────────────────
#[cfg(feature = "dllm")]
pub use crate::dllm::D2fContext;
#[cfg(feature = "dllm")]
pub use d2f::{
    D2fBlockResult, D2fBlockState, D2fDecodeConfig, D2fPipeline, D2fPipelineResult, ScheduleKind,
    d2f_decode_block, d2f_decode_block_with, d2f_decode_block_with_prompt,
    d2f_decode_block_with_prompt_with, d2f_decode_block_with_target,
    d2f_decode_block_with_target_with,
};

// ── DMax Soft Parallel Decode Re-exports (Plan 109, feature: dmax_spd) ──
#[cfg(feature = "dmax_spd")]
pub use d2f::{
    BlockConvergence, HybridEmbedding, SoftDecodeConfig, check_block_convergence,
    contiguous_prefix_promote, d2f_decode_block_soft,
};

// ── D2F Drafter Verifier Re-exports (Plan 089, Tri-Mode) ───
#[cfg(feature = "tri_mode")]
pub use d2f_verifier::D2fDrafterVerifier;

// ── DiffusionSampler Re-exports (Plan 116, Tri-Mode) ──────────
#[cfg(feature = "tri_mode")]
pub use d2f::{d2f_decode_block_with_prompt_with_sampler, d2f_decode_block_with_sampler};
#[cfg(feature = "tri_mode")]
pub use diffusion_sampler::{
    DiffusionSampler, SamplerDecision, SamplerFeatures, SamplerTrajectory, SamplerVariant,
    collect_trajectories, train_logistic_on_patterns,
};

#[cfg(feature = "sudoku")]
pub use crate::pruners::SudokuPruner;

pub use budget_compat::{effective_tree_budget, scaled_draft_lookahead};

// ── Budget Adaptation Re-exports (Plan 167, feature: budget_adaptation) ──
#[cfg(feature = "budget_adaptation")]
pub use budget::{adaptive_tree_budget, compression_ratio, entropy_signal, shannon_entropy};
#[cfg(feature = "budget_adaptation")]
pub use prefill::block_compression_ratio;

// ── PPoT Re-exports (Plan 026 + 027) ──────────────────────────
#[cfg(feature = "ppot")]
pub use ppot::{
    ErrorKind, PpotConfig, RejectionInsight, SessionKnowledge, TokenRule,
    identify_high_entropy_positions, identify_high_entropy_positions_into,
    identify_positions_adaptive, identify_positions_adaptive_into, identify_positions_by_rule,
    identify_positions_by_rule_into, ppot_resample, ppot_resample_different_value,
    ppot_resample_multi_strategy, ppot_resample_with_support, ppot_rescue, ppot_rescue_adaptive,
    rank_by_consistency, rank_by_consistency_weighted, select_best_variant, token_entropy,
};

// ── FlashAR Strided Anchor-Then-Fill (Plan 166 T11, feature: flashar_anchor) ──
#[cfg(feature = "flashar_anchor")]
pub mod flashar_anchor;

#[cfg(feature = "flashar_anchor")]
pub use flashar_anchor::{AnchorConfig, AnchorFillResult, anchor_then_fill};

// ── FlashAR Consensus Tri-Mode re-exports (Plan 166, feature: flashar_consensus) ──
#[cfg(feature = "flashar_consensus")]
pub mod flashar_consensus;

#[cfg(feature = "flashar_consensus")]
pub use flashar_consensus::{
    ConsensusConfig, ConsensusResult, DualPathResult, FlashARConsensusVerifier, MAX_DRAFT_WIDTH,
    ThermalPath, compute_ternary_consensus, dual_path_draft, route_thermal_paths,
};

// ── Parallel-Probe 2D Controller re-exports (Plan 133, feature: parallel_probe) ──
#[cfg(feature = "parallel_probe")]
pub use parallel_probe::{
    BranchProbeState, ParallelProbeConfig, ParallelProbeController, ParallelProbeVerifier,
    ProbeDecision, ProbingMatrix,
};

#[cfg(feature = "parallel_probe")]
pub use answer_extract::{
    AnswerExtractor, DiscreteActionExtractor, RegexAnswerExtractor, ThinkTokenExtractor,
};

// ── DFlare Modelless Inference re-exports (Plan 174, feature: dflare_fusion) ──
#[cfg(feature = "dflare_fusion")]
pub use dflash::dflash_predict_ar_with_fusion;
#[cfg(feature = "dflare_fusion")]
pub use dflash::marginal_fusion_blend;
#[cfg(feature = "dflare_fusion")]
pub use types::MarginalFusionConfig;

// ── DFlare KV Routing re-exports (Plan 174, feature: dflare_kv_routing) ──
#[cfg(feature = "dflare_kv_routing")]
pub use dflash::dflash_predict_conditioned_with_routing;
#[cfg(feature = "dflare_kv_routing")]
pub use types::KvRoutingConfig;

// ── DFlare Progressive Budget re-exports (Plan 174, feature: dflare_progressive_budget) ──
#[cfg(feature = "dflare_progressive_budget")]
pub use dd_tree::build_dd_tree_screened_progressive;
#[cfg(feature = "dflare_progressive_budget")]
pub use types::PositionWeightedBudget;

// ── Adaptive CoT Thinking Controller (Plan 194, feature: thinking_cot) ──
#[cfg(feature = "thinking_cot")]
pub mod thinking_controller;

#[cfg(feature = "thinking_cot")]
pub use thinking_controller::{
    Rng, ThinkingBanditFrozen, ThinkingConfig, ThinkingController, ThinkingMode, ThinkingSelector,
};

#[cfg(feature = "vocab_coreset")]
pub mod vocab_coreset;

#[cfg(feature = "vocab_coreset")]
pub use vocab_coreset::{should_use_delta_sparse, vocab_coreset};

// ── Trust-Region Adaptive Speculation (Plan 182, Research 162) ──
#[cfg(feature = "trust_region_spec")]
pub use trust_region::{
    TrustArm, TrustRegionConfig, TrustRegionState, TrustTracker, adaptive_window, blend_sample,
    find_blend_beta,
};
