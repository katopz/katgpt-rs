//! Domain-specific constraint pruners for the DDTree search engine.

pub mod dungeon_pathfinder;
pub mod dungeon_pruner;
pub mod map_generator;
pub mod pathfinder;
pub mod tactical_pruner;

pub use dungeon_pathfinder::{
    DungeonAction, MultiFloorBlocked, MultiFloorTarget, enumerate_multifloor_targets,
    find_path_multifloor, find_path_on_floor,
};
pub use dungeon_pruner::{DungeonMap, DungeonPruner, DungeonState, FloorGrid, StairConnection};
pub use map_generator::{GeneratedDungeon, GeneratedMap, MapGenerator};
pub use pathfinder::{Target, enumerate_targets, find_distance, find_path, reachable_positions};
pub use tactical_pruner::{GameState, TacticalPruner};

#[cfg(feature = "bomber")]
pub mod bomber;

pub mod game_state; // Always compiled — GameState trait has no bevy_ecs dependency (G1 fix, Plan 065)

#[cfg(feature = "subterranean")]
pub mod subterranean;

pub mod freeze;

pub mod emotion_vector;

#[cfg(feature = "thinking_prune")]
pub mod frozen_base_guard;

#[cfg(feature = "sudoku")]
pub mod sudoku_pruner;

#[cfg(feature = "sudoku")]
pub use sudoku_pruner::SudokuPruner;

#[cfg(feature = "bandit")]
pub mod absorb_compress;

#[cfg(feature = "bandit")]
pub mod bandit;

#[cfg(feature = "bandit")]
pub mod hot_swap;

#[cfg(feature = "bandit")]
pub mod regression;

#[cfg(feature = "bandit")]
pub mod review_metrics;

#[cfg(feature = "bandit")]
pub mod trial_log;

#[cfg(feature = "safe_bandit")]
pub mod safe_phased;

#[cfg(feature = "sr2am_configurator")]
pub mod configurator_bandit;

#[cfg(feature = "sia_feedback")]
pub mod feedback_bandit;

#[cfg(feature = "dreamer")]
pub mod dreamer;

#[cfg(feature = "dreamer")]
pub use dreamer::{
    ConsolidationResult, CounterfactualEstimator, DecayPolicy, DreamerConfig, DreamerConsolidator,
    DreamerFrozenBank, DreamerPipeline, DreamerScheduler, MemoryDecay, ReplacementSet,
    WorkingRegion, load_frozen_dreamer, save_frozen_dreamer,
};

#[cfg(feature = "bandit")]
pub use absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};

#[cfg(feature = "bandit")]
pub use bandit::{
    BanditEnv, BanditEvent, BanditPruner, BanditResult, BanditSession, BanditStats, BanditStrategy,
    BernoulliEnv, GaussianEnv, SharedBanditStats,
};

#[cfg(feature = "bandit")]
pub use review_metrics::{
    EmotionProfileSummary, EntropyAnomalySummary, ReviewMetrics, ReviewStrategy, ReviewSummary,
};

#[cfg(feature = "bandit")]
pub use hot_swap::HotSwapPruner;

#[cfg(feature = "bandit")]
pub use regression::{GoldenTrace, RegressionResult, RegressionSuite, ReplayReward};

#[cfg(feature = "bandit")]
pub use trial_log::{SharedTrialLog, TrialLog, TrialRecord, TrialSummary};

#[cfg(feature = "bandit_top_p")]
pub use bandit::select_arms_top_p;

#[cfg(feature = "safe_bandit")]
pub use safe_phased::SafePhasedState;

#[cfg(feature = "sr2am_configurator")]
pub use configurator_bandit::{ConfiguratorBandit, ExplorationOutcome, PrunerSchedule};

#[cfg(feature = "sia_feedback")]
pub use feedback_bandit::{
    FeedbackBandit, FeedbackBanditConfig, RlAlgorithmHint, TrajectorySummary, WeightUpdateRequest,
};

#[cfg(feature = "g_zero")]
pub mod g_zero;

#[cfg(feature = "g_zero")]
pub use g_zero::{
    BomberTemplate, BomberTemplateProposer, DeltaBanditPruner, DeltaGatedAbsorbCompress,
    DeltaGatedConfig, GeneratedPair, HintDelta, LogProbResult, QueryTemplate, TemplateProposer,
};

#[cfg(feature = "ropd_rubric")]
pub mod ropd_rubric;

#[cfg(feature = "ropd_rubric")]
pub use ropd_rubric::{
    PatternRule, PatternScorer, RubricBanditConfig, RubricBanditPruner, RubricCriterion,
    RubricGatedAbsorbCompress, RubricGatedConfig, RubricScorer, RubricTemplate, RubricVector,
    ScoreResult, score_with_references, score_with_references_id,
};

#[cfg(feature = "memo_reflections")]
pub mod reflection;

#[cfg(feature = "memo_reflections")]
pub use reflection::{
    GameStateSnapshot, ReflectionDomain, ReflectionQA, ReflectionResult, ReflectionStep,
    consolidate_facts, extract_facts, surface_entities, synthesize_cross_game,
    synthesize_reflections, verify_self_containment,
};

#[cfg(feature = "sdar_gate")]
pub mod sdar;
#[cfg(feature = "sdar_gate")]
pub mod sdar_gate;

#[cfg(feature = "sdar_gate")]
pub use sdar::{
    GateStats, PromotionStats, SdarAbsorbConfig, SdarBanditConfig, SdarBanditPruner,
    SdarGatedAbsorbCompress,
};
#[cfg(feature = "sdar_gate")]
pub use sdar_gate::{
    SDAR_BETA, SDAR_BETA_MAX, SDAR_BETA_MIN, sdar_benefit_gate, sdar_gate, sdar_gate_default,
    sdar_gated_reward, sdar_modulate, sdar_modulate_default, sdar_should_promote,
};

// ── SDPG Bandit — Modelless Self-Distilled Policy Gradient (Research 160, Plan 180) ──

#[cfg(feature = "sdpg_bandit")]
pub mod sdpg;

#[cfg(feature = "sdpg_bandit")]
pub use sdpg::{
    AdvantageMode, BetaSchedule, KlAnchor, SdpgBanditPruner, centered_log_ratio,
    raw_delta_advantage, sigmoid_advantage, softmax_scaled,
};

#[cfg(feature = "vpd_em_distill")]
pub mod vpd_em;

#[cfg(feature = "vpd_em_distill")]
pub use vpd_em::{BcoOptimizer, BcoSample, VpdConfig, VpdEmCycle};

#[cfg(feature = "rmsd_distill")]
pub mod rmsd_relevance;

#[cfg(feature = "rmsd_distill")]
pub use rmsd_relevance::{
    LogprobMagnitudeFilter, MagnitudeJudge, RmsdConfig, RmsdMetrics, RmsdRelevanceFilter,
    TeacherContinuation, TopKlApproximator, rmsd_loss,
};

#[cfg(feature = "cna_steering")]
pub mod cna;

#[cfg(feature = "cna_steering")]
pub use cna::{
    CnaCircuit, CnaDiscoveryConfig, CnaModulator, CnaNeuron, CnaScreeningPruner,
    ContrastivePairProvider, cna_discover, cna_modulate, detect_universal_neurons,
};

#[cfg(feature = "deep_manifold")]
pub mod manifold_residual;

#[cfg(feature = "deep_manifold")]
pub use manifold_residual::{
    KlResidualScorer, L2ResidualScorer, ManifoldResidual, ResidualRelevanceScorer,
};

#[cfg(feature = "federation")]
pub mod boundary_alignment;

#[cfg(feature = "federation")]
pub use boundary_alignment::{BoundaryAlignment, KlBoundaryAligner};

#[cfg(feature = "replaid_schedules")]
pub mod variance_minimizer;

#[cfg(feature = "replaid_schedules")]
pub use variance_minimizer::{VarianceMinimizer, VarianceMinimizerConfig};

#[cfg(feature = "bt_rank")]
pub mod bt_rank;

#[cfg(feature = "bt_rank")]
pub use bt_rank::{
    BtComparison, BtConfig, BtOutcome, BtScores, bt_fit, bt_fit_from_fn, sigmoid as bt_sigmoid,
};

#[cfg(feature = "stepcode")]
pub mod stepcode;

#[cfg(feature = "stepcode")]
pub use stepcode::{PathStep, ShapedPath, path_consistency, shape_path};

#[cfg(feature = "delta_mem")]
pub mod delta_mem;

#[cfg(feature = "delta_mem")]
pub use delta_mem::{
    AggregationStrategy, ContextFeatures, CorrectionMode, DeltaMemoryConfig, DeltaMemorySnapshot,
    DeltaMemoryState, FeatureHasher, MemorySteeredPruner, MultiDomainMemory,
    MultiDomainMemoryPruner, OutcomeFeatures, WriteGranularity,
};

#[cfg(feature = "tes_loop")]
pub mod tes_loop;

#[cfg(feature = "tes_loop")]
pub use tes_loop::{SimpleTesLoop, TesLoop};

#[cfg(all(feature = "g_zero", feature = "fft"))]
pub use g_zero::{FFTTemplate, FFTTemplateProposer};

#[cfg(all(feature = "fft", feature = "g_zero"))]
pub use fft::GZeroFFTPlayer;

#[cfg(feature = "bomber")]
pub use bomber::{
    ArenaGrid, BomberAction, BomberPlayer, GridPos, ScoreBoard, TickCounter, run_tick,
    spawn_players,
};

pub use game_state::{ActionSpaceLog, StateHeuristic, mcts_search};

pub use freeze::{load_frozen, save_frozen};

#[cfg(feature = "thinking_prune")]
pub use frozen_base_guard::FrozenBaseGuard;

#[cfg(feature = "gdsd_distill")]
pub mod gdsd;

#[cfg(feature = "gdsd_distill")]
pub use gdsd::{
    GdsdConfig, GdsdPruner, clamped_advantage, identity_advantage, sigmoid_advantage,
    tanh_advantage, token_logit_centralization,
};

#[cfg(all(feature = "game_state", feature = "bomber"))]
pub use game_state::{BombSnapshot, BomberHeuristic, BomberState, PlayerSnapshot};

#[cfg(feature = "monopoly")]
pub mod monopoly;

#[cfg(feature = "monopoly")]
pub use monopoly::{
    Board, BoardSquare, CardDeck, CardEffect, DecisionContext, GameConfig, GamePhase, JailDecision,
    JailReason, MonopolyPlayer, Owned, Player, Property, PropertyGroup, ReleaseMethod, Statistics,
    Strategy, TaxKind, TradeOffer, TradeResponse, TurnPhase, build_board, shuffle_decks,
    square_kind, square_name,
};

#[cfg(feature = "go")]
pub mod go;

#[cfg(feature = "go")]
pub use go::{
    AutoGoClient, AutoGoError, AutoGoGameState, AutoGoProxyPlayer, DEFAULT_KOMI, GameOutcome,
    GameResult, GoAction, GoCell, GoGZeroPlayer, GoGreedyPlayer, GoHLPlayer, GoHeuristic,
    GoMctsPlayer, GoMoveCategory, GoPlayer, GoPlayerType, GoRandomPlayer, GoReplay, GoState,
    GoTemplate, GoTournamentConfig, GoTournamentResult, GoValidatorPlayer, MoveRecord,
    ParsedResult, ReplayError, TournamentDef, parse_go_result, print_batch_table, run_tournament,
    run_tournament_batch,
};

#[cfg(feature = "event_log")]
pub mod event_log;

#[cfg(feature = "event_log")]
pub use event_log::{Actor, DiffEvent, EvalCache, Event, EventId, EventLog, EventType, ForkDiff};

#[cfg(any(feature = "bomber", feature = "fft", feature = "tes_loop"))]
pub mod arena;

#[cfg(feature = "randopt_weight")]
pub mod randopt;

#[cfg(feature = "randopt_weight")]
pub use randopt::{
    AccuracyScorer, RandOptConfig, RandOptEnsemble, RandOptResult, RandOptScorer, RandOptSession,
    RandOptWeightSampler,
};

#[cfg(all(any(feature = "bomber", feature = "fft"), not(feature = "go")))]
pub use arena::GameResult;
#[cfg(all(any(feature = "bomber", feature = "fft"), feature = "go"))]
pub use arena::GameResult as ArenaGameResult;
#[cfg(any(feature = "bomber", feature = "fft"))]
pub use arena::{ArenaKind, EloCalculator, Leaderboard, Matchup, MatchupResult, Ranking};

#[cfg(feature = "tes_loop")]
pub use arena::TrajectoryPruner;

#[cfg(feature = "fft")]
pub mod fft;

#[cfg(feature = "subterranean")]
pub use subterranean::{
    BomberNode, BomberProcedure, BridgeError, ComplexityTier, DecisionPoint, GoNode, GoProcedure,
    NodeStateMapping, PathEnumerator, PathSampler, ProcedureCostModel, ProcedureEdge,
    ProcedureGameState, ProcedureGraph, ProcedureNode, Sample, SampleFilter,
    SubterraneanTrainingMode, TrainingBudget, Trajectory, TrajectoryBanditSummary,
    TrajectoryValidator, extract_all_decision_points, extract_decision_points,
    graph_to_global_session, graph_trajectories_to_sessions, graph_trajectories_to_sessions_seeded,
};

#[cfg(feature = "fft")]
pub use fft::{
    Action, ActionType, ActiveEffect, BattleState, Class, FftPlayer, GameEvent, GreedyFFTPlayer,
    HLFFTPlayer, Pos, Stats, StatusEffect, Team, Unit, ValidatorFFTPlayer, resolve_action,
};

#[cfg(feature = "opus_selection")]
pub mod opus;

#[cfg(feature = "opus_selection")]
pub use opus::{
    CountSketch, OpusBanditPruner, OpusConfig, OpusRedundantEnv, boltzmann_probabilities,
    boltzmann_sample, boltzmann_sample_batch, exact_inner_product, squared_norm,
};

#[cfg(feature = "proof_sketch_evolution")]
pub mod proof;

#[cfg(feature = "proof_sketch_evolution")]
pub use proof::{
    DEFAULT_ELO, DEFAULT_EPSILON, DEFAULT_EXPLORATION_C, DTreeCacheSnapshot, DTreeGoalCache,
    DiversityHint, DiversityStrategy, ELO_SCALE, EvictionReport, Goal, GoalHash, GoalResult,
    MAX_LESSONS, MAX_PENDING_GOALS, ParallelismGuard, PlackettLuceConfig, PlackettLuceRater,
    PopulationConfig, ProofGoalCache, ProofGoalSnapshot, ProofState, SketchEntry, SketchId,
    SketchPopulation, SketchSampler, SketchSamplerConfig, SketchSelectionStrategy,
    encode_constraint_key, select_strategy, should_use_population,
};

#[cfg(feature = "epiplexity_scoring")]
pub mod epiplexity;

#[cfg(feature = "epiplexity_scoring")]
pub use epiplexity::{
    EpiplexityEstimator, EpiplexityScreeningPruner, EpiplexityWeight, FactorizationOrder,
    FactorizationScorer, LossCurveTracker, PerPositionLossTracker, TimeBoundedEntropy,
};

#[cfg(feature = "committee_boost")]
pub mod committee_boost;

#[cfg(feature = "committee_boost")]
pub use committee_boost::{
    BlindSpotEstimate, BudgetError, CommitteeBudget, ConvergenceFit, CoverageAction,
    CoverageDiagnostic, DebiasedComparator, FailureMode, OracleGapRecovery, committee_budget,
    coverage_diagnostic, debiased_compare, estimate_blind_spot_floor, fit_convergence,
};

#[cfg(feature = "state_source")]
pub mod state_source;

#[cfg(feature = "state_source")]
pub use state_source::{
    ContinuationScore, ContinuationScorer, PUCBSelector, RetentionMetric, StateVisitationTracker,
    adaptive_c,
};

// ── GEPA-D Reflective Config Evolution (Research 146, Plan 164) ──

#[cfg(feature = "gepa_reflective")]
pub mod gepa_reflective;

#[cfg(feature = "gepa_reflective")]
pub use gepa_reflective::{
    ConfigVariant, ParetoConfigFrontier, ReflectionScore, ReflectiveBanditPruner,
};

// ── PhraseBoost Context Trie (Research 147, Plan 164) ──

pub mod curvature_alloc;

pub use curvature_alloc::{
    CurvatureInfluenceScorer, CurvatureWeightedBudget, EosProxyScorer, verification_depth,
};

#[cfg(feature = "nds_proxy")]
pub use curvature_alloc::NdsAwareScorer;

// ── PhraseBoost Context Trie (Research 147, Plan 164) ──

#[cfg(feature = "phrase_boost")]
pub mod phrase_boost;

#[cfg(feature = "phrase_boost")]
pub mod phrase_trie;

#[cfg(feature = "phrase_boost")]
pub use phrase_boost::{DEFAULT_BOOST_SCORE, PhraseBoostPruner};

#[cfg(feature = "phrase_boost")]
pub use phrase_trie::PhraseTrie;

// ── Hydra Adaptive Layer Budget (Research 148, Plan 165) ──

#[cfg(feature = "hydra_budget")]
pub mod hydra_budget;

#[cfg(feature = "hydra_budget")]
pub use hydra_budget::{
    HydraBudgetResult, HydraSkipPlan, LogitLensScore, adaptive_depth_gate, calibrate_from_prompts,
    calibrate_profiles, detect_erasure_layers, hydra_adaptive_budget, hydra_layer_skip,
    logit_lens_score, should_skip_layer,
};

#[cfg(all(feature = "hydra_budget", feature = "decode_specialize"))]
pub use hydra_budget::should_skip_layer_stage;

#[cfg(feature = "mux_pruner")]
pub mod mux_span;

#[cfg(feature = "mux_pruner")]
pub use mux_span::MuxSpanPruner;

#[cfg(feature = "mux_ddtree")]
pub mod mux_ddtree;

#[cfg(feature = "mux_ddtree")]
pub use mux_ddtree::{MuxDdTree, MuxNode};

#[cfg(feature = "mux_bandit_width")]
pub mod mux_bandit;

#[cfg(feature = "mux_bandit_width")]
pub use mux_bandit::MuxBanditWidth;

#[cfg(feature = "mux_freeze_thaw")]
pub mod mux_freeze_thaw;

#[cfg(feature = "mux_freeze_thaw")]
pub use mux_freeze_thaw::{DifficultyTier, MuxPatternStore, MuxTarget};

#[cfg(feature = "mux_bfs")]
pub mod mux_bfs;

#[cfg(feature = "mux_bfs")]
pub use mux_bfs::MuxBfs;

// ── Direction-Adaptive Credit — Entropy-Bifurcated Pruning (Plan 184) ──

#[cfg(feature = "directional_credit")]
pub mod entropy_bifurcated;

#[cfg(feature = "directional_credit")]
pub use entropy_bifurcated::{EntropyBifurcatedPruner, EntropyState, SelfDrivenTokenTracker};

#[cfg(feature = "mech_attribution")]
pub mod mech_attribution;

#[cfg(feature = "mech_attribution")]
pub use mech_attribution::{
    ActivationInfluenceProxy, CatalystPattern, CatalystTemplate, InfluenceConfig,
    MechInfluenceScore, batch_influence_rank, catalyst_score, detect_catalyst_pattern,
    extract_template, generate_synthetic,
};

// ── WealthPruner — Economic Bandit Arms via Hayek Market Selection (Plan 187) ──

#[cfg(feature = "wealth_pruner")]
pub mod wealth_bandit;

#[cfg(feature = "wealth_pruner")]
pub use wealth_bandit::{ChainCreditAssigner, WealthArm, WealthBanditPruner, WealthPrunerConfig};

// ── Inference-Time Skill Evolution (Plan 192, Research 172) ──

#[cfg(feature = "skill_lifecycle")]
pub mod skill_memory;

#[cfg(feature = "skill_lifecycle")]
pub mod skill_test;

#[cfg(feature = "skill_lifecycle")]
pub mod skill_catalog;

#[cfg(feature = "skill_lifecycle")]
pub use skill_memory::{MemoryEntry, PrunerMemory, compute_hash};

#[cfg(feature = "skill_lifecycle")]
pub use skill_test::{
    BomberTestGate, PrunerTestGate, SimpleTestGate, TestCase, TestResult, TestStatus,
};

#[cfg(feature = "skill_lifecycle")]
pub use skill_catalog::{
    FileSkillLoader, LazySkillLoader, SkillCatalog, SkillDescriptor, write_arm_data,
};

// ── NDS Curvature Proxy — Modelless Inference-Time Budget Control (Plan 186) ──

#[cfg(feature = "nds_proxy")]
pub mod nds_proxy;

#[cfg(feature = "nds_proxy")]
pub use nds_proxy::{
    LayerDepth, NdsBudgetModifier, SpectralFlatnessBudget, layer_nds_depth,
    nds_proxy as compute_nds_proxy, nds_scaled_budget, spectral_balance_bonus,
    spectral_balance_score,
};

// ── Open-Ended Problem Evolution Arena (Plan 191, Research 171) ──

#[cfg(feature = "partial_scoring")]
pub mod partial_scorer;

#[cfg(feature = "partial_scoring")]
pub use partial_scorer::{BomberPartialScorer, WinLossScorer};

#[cfg(feature = "problem_mutator")]
pub mod problem_mutator;

#[cfg(feature = "problem_mutator")]
pub use problem_mutator::BomberConfigMutator;

#[cfg(feature = "idea_divergence")]
pub mod idea_divergence;

#[cfg(feature = "idea_divergence")]
pub use idea_divergence::IdeaDivergence;

#[cfg(feature = "rosetta_pruner")]
pub mod rosetta;

#[cfg(feature = "rosetta_pruner")]
pub use rosetta::{ConstraintConcept, RosettaPruner};

// ── RV-Gated Compute Routing — AcceptanceVarianceTracker (Plan 202) ──

#[cfg(feature = "rv_gated_routing")]
pub mod acceptance_variance;

#[cfg(feature = "rv_gated_routing")]
pub use acceptance_variance::AcceptanceVarianceTracker;

// ── Episode-Guided Constraint Synthesis — EpisodePruner (Plan 206) ──

#[cfg(feature = "egcs")]
pub mod episode_pruner;

#[cfg(feature = "egcs")]
pub use episode_pruner::{
    ConstraintSynthesizer, Episode, EpisodeLookup, EpisodeMetadata, EpisodePruner,
    MemoryEpisodeLookup, StructuralDiffSynthesizer, SynthesizedConstraint,
};

#[cfg(feature = "egcs")]
pub mod vr_loop;

#[cfg(feature = "egcs")]
pub use vr_loop::{VrGenerator, VrLoop, VrLoopResult, VrRoundFeedback, VrVerifier};

// ── Lodestar — Completion-Distance Pruning (Plan 207, Research 183) ──

#[cfg(feature = "lodestar")]
pub mod lodestar;

#[cfg(feature = "lodestar")]
pub use lodestar::{LodestarAutomaton, LodestarConfig, LodestarPruner, UNREACHABLE};

#[cfg(feature = "lodestar")]
mod lodestar_cot;

#[cfg(feature = "lodestar")]
pub use lodestar_cot::{AdaptiveCoTBudget, AdaptiveCoTConfig};

// ── Self-Distilling Pruner Bandit — Episode-Guided Arm Selection (Plan 208) ──

#[cfg(feature = "self_distilling_bandit")]
pub mod self_distilling_bandit;

#[cfg(feature = "self_distilling_bandit")]
pub use self_distilling_bandit::{
    ConvergenceMetrics, EpisodeRewardComputer, SelfDistillingBandit, SelfDistillingConfig,
    compute_match_ratio,
};

// ── FOL Logical Rule Inference (Plan 209) ────────────────────────────────

#[cfg(feature = "fol_constraints")]
pub mod fol_pruner;

#[cfg(feature = "fol_constraints")]
pub use fol_pruner::{FolConstraint, FolPruner, extract_fol_constraints};

#[cfg(feature = "rule_extraction")]
pub mod rule_extractor;

#[cfg(feature = "rule_extraction")]
pub use rule_extractor::{ExtractedRule, RuleExtractor, TreeNode, deduplicate_rules};

#[cfg(feature = "decision_trace")]
pub mod decision_trace;

#[cfg(feature = "decision_trace")]
pub use decision_trace::{DecisionTrace, DecisionTraceBuilder};

#[cfg(feature = "decision_explain")]
pub mod decision_explainer;

#[cfg(feature = "decision_explain")]
pub use decision_explainer::{
    CandidateRecord, DecisionExplainer, DecisionExplanation, PerturbationExplainer,
    PrunerAttribution, RejectedAlternative, SensitivityCache, TokenChoice, TraceNode, trace_hash,
};

#[cfg(feature = "reward_mem")]
pub mod reward_mem_pruner;

#[cfg(feature = "reward_mem")]
pub use reward_mem_pruner::{CompileOutcome, PatternHasher, RewardMemPruner};

// ── INSIGHT Symbolic Distillation & Explanation (Plan 210) ──────────
//
// A modelless explore→distill→explain pipeline:
//   F1: Symbolic expression fitting from DDTree traces
//   F2: Concept grounding for human-readable explanations
//   F3: Perturbation-based decision explanation with sensitivity analysis
//   F4: Reward-gated pruner calibration with absorption
//
// Feature gate: `insight_explain` (convenience parent)
// Individual features: `symbolic_distill`, `concept_grounding`, `decision_explain`, `reward_calibrator`
// All independently gateable, zero-cost when disabled.

// ── Symbolic Expression Distillation (Plan 210 F1) ───────────────────────

#[cfg(feature = "symbolic_distill")]
pub mod symbolic_expression;

#[cfg(feature = "symbolic_distill")]
pub mod expression_pruner;

#[cfg(feature = "symbolic_distill")]
pub use expression_pruner::{DefaultFeatureExtractor, ExpressionPruner, FeatureExtractor};

#[cfg(feature = "symbolic_distill")]
pub use symbolic_expression::{
    BasisFn, SymbolicExpression, SymbolicExpressionFitter, Term, TraceDataset, TraceRecord,
    TraceRecorder,
};

// ── Reward-Gated Pruner Calibration (Plan 210 F4) ─────────────────────

#[cfg(feature = "reward_calibrator")]
pub mod reward_calibrator;

#[cfg(feature = "reward_calibrator")]
pub use reward_calibrator::{
    CalibrationStep, CalibratorConfig, ParameterKey, ParameterStats, RewardGatedCalibrator,
};

// ── Concept Grounding — Template-Based Pruner Explanation (Plan 210 F2) ──

#[cfg(feature = "concept_grounding")]
pub mod concept_grounding;

#[cfg(feature = "concept_grounding")]
pub use concept_grounding::{
    ConceptGrounding, ConceptMapping, GroundingSource, PolicyExplanation, PrunerState,
    TemplateGrounding,
};

// ── Collapse-Aware Adaptive Thinking (Plan 212) ─────────────────────
//
// Three-layer adaptive thinking: Pre-Decide (SelectivityRouter) →
// Mid-Think CollapseDetector (new) → Post-Verify T2M OptionStripper (new).
// Monitors token stream during reasoning and triggers early exit when
// reasoning collapse is detected (hesitation patterns, repetitive tokens).
//
// Feature gate: `collapse_aware_thinking` (convenience parent)

#[cfg(feature = "collapse_aware_thinking")]
pub mod collapse_detector;

#[cfg(feature = "collapse_aware_thinking")]
pub use collapse_detector::{
    CollapseAction, CollapseDetectorFrozen, S2FCollapseDetector, check_collapse_action,
    efficiency_reward,
};

// ── Collapse-Aware Adaptive Thinking — T2M Option Stripper (Plan 212 T5/T6) ──

#[cfg(feature = "collapse_aware_thinking")]
pub mod option_stripper;

#[cfg(feature = "collapse_aware_thinking")]
pub use option_stripper::OptionStripper;

// ── Three-Mode Neuro-Symbolic Bandit Router (Plan 211) ─────────────
//
// Dynamic mode selection between L4R, R4L, and LR neuro-symbolic modes
// via 6-arm UCB1 bandit with sigmoid-gated mixing weights.
//
// Feature gate: `three_mode_router`

#[cfg(feature = "three_mode_router")]
pub mod three_mode_bandit;

#[cfg(feature = "three_mode_router")]
pub use three_mode_bandit::{
    BanditArm, ModeFeatures, NeuroSymbolicMode, RollingWindow, ThreeModeBandit,
    compute_mode_features, grounding_quality,
};

// ── Safe Exploration Budget (Plan 211 F3) ────────────────────

#[cfg(feature = "safe_exploration_budget")]
pub mod exploration_budget;

#[cfg(feature = "safe_exploration_budget")]
pub use exploration_budget::{
    ExplorationBudget, ExplorationBudgetConfig, VerificationResult, VerificationTier, check_budget,
};

// ── Auto Constraint Synthesis (Plan 211 F2) ────────────────────

#[cfg(feature = "auto_constraint_synthesis")]
pub mod constraint_miner;

#[cfg(feature = "auto_constraint_synthesis")]
pub use constraint_miner::{
    ConstraintMiner, Pattern, SequenceConstraint, extract_frequent_sequences, mine_and_insert,
};

// ── BFCF Tree — Perceptual Region Folding (Plan 213) ──────────
//
// Replaces O(vocab_size) token-by-token screening with O(regions)
// region-level pruning. ScreeningPruner threshold crossings partition
// logit space into convex BFCP regions.
//
// Feature gate: `bfcf_tree`

#[cfg(feature = "bfcf_tree")]
pub mod bfcf_types;

#[cfg(feature = "bfcf_tree")]
pub use bfcf_types::{BFCP, BorelRegion, HalfSpace, PWCValueFunction, RegionLabel};

#[cfg(feature = "bfcf_tree")]
pub mod bfcp_pruner;

#[cfg(feature = "bfcf_tree")]
pub use bfcp_pruner::BFCPPruner;

#[cfg(feature = "bfcf_tree")]
pub mod bfcp_preimage;

#[cfg(feature = "bfcf_tree")]
pub use bfcp_preimage::{acceptance_rate, compute_preimage, maybe_rate, refine_partition};

#[cfg(feature = "bfcf_tree")]
pub mod pwc_bandit;

#[cfg(feature = "bfcf_tree")]
pub use pwc_bandit::RegionBandit;

#[cfg(feature = "bfcf_tree")]
pub mod percept_router;

#[cfg(feature = "bfcf_tree")]
pub use percept_router::{ComputePath, PerceptRouter, PerceptRouterConfig, SigmoidPerceptRouter};

#[cfg(feature = "bfcf_tree")]
pub use bfcf_types::BfcpPartition;

// ── BFCF × LFU × Sharding — Region Cache (Plan 218 Phase 1) ──────────────
//
// LFU cache for BFCP partitions with BLAKE3 commitment, sigmoid-gated admission,
// and Hot/Warm/Cold frequency tiers. papaya lock-free HashMap for concurrent access.
//
// Feature gate: `bfcf_lfu_shard`

#[cfg(feature = "bfcf_lfu_shard")]
pub mod bfcp_region_cache;

#[cfg(feature = "bfcf_lfu_shard")]
pub use bfcp_region_cache::{BfcpRegionCache, FreqTier, RegionCaching, blake3_logit_hash};

// ── BFCF × LFU × Sharding — Region Shard Map (Plan 218 Phase 2) ────────────
//
// Frequency-aware (RegionLabel × FreqTier) → shard index mapping.
// Hot pinned to shard 0, Cold to last shard, Warm round-robin.
// Sequential fallback when regions < 30.

#[cfg(feature = "bfcf_lfu_shard")]
pub mod region_shard_map;

#[cfg(feature = "bfcf_lfu_shard")]
pub use region_shard_map::{RegionShardMap, RegionSharding};

// ── BFCF × LFU × Sharding — Region Batch (Plan 218 Phase 3) ─────────────
//
// Batch processing for accept/reject/maybe regions:
// - accept → gather token indices (deterministic, SIMD-friendly loop)
// - reject → sum token counts (O(regions), zero allocation)
// - maybe  → classify tokens via ScreeningPruner, split into sub-regions

#[cfg(feature = "bfcf_lfu_shard")]
pub mod region_batch;

#[cfg(feature = "bfcf_lfu_shard")]
pub use region_batch::{RegionBatcher, RegionBatching};

// ── SubstrateGate — Inference-Time Capability Substrate Routing (Plan 216) ──
//
// Pre-computed per-capability MLP channel masks intersected with ReLU
// activation masks for dual sparsity. DDTree branches route through
// different capability substrates.
//
// Feature gate: `substrate_gate`

#[cfg(feature = "substrate_gate")]
pub mod substrate_types;

#[cfg(feature = "substrate_gate")]
pub use substrate_types::{NoSubstrateRouter, SubstrateConfig, SubstrateMask, SubstrateRouter};

#[cfg(feature = "substrate_gate")]
pub mod substrate_execution;

#[cfg(feature = "substrate_gate")]
pub mod substrate_ddtree;

#[cfg(feature = "substrate_gate")]
pub use substrate_ddtree::{SubstrateBranch, substrate_branch_score};

#[cfg(feature = "substrate_gate")]
pub mod substrate_pruner;

#[cfg(feature = "substrate_gate")]
pub use substrate_pruner::SubstrateScreeningPruner;

#[cfg(feature = "substrate_gate")]
pub mod substrate_loader;

#[cfg(feature = "substrate_gate")]
pub use substrate_loader::{load_substrate_mask, save_substrate_mask};

// ── Belief-State Rank Pruner (Plan 217 Phase 3) ──────────────
//
// Uses hidden state effective rank (participation ratio) as screening signal.
// Low rank → confident → accept drafts. High rank → uncertain → reject.
//
// Feature gate: `belief_drafter`

#[cfg(feature = "belief_drafter")]
pub mod belief_rank_pruner;

#[cfg(feature = "belief_drafter")]
pub use belief_rank_pruner::BeliefRankPruner;

// ── CoExplain Bidirectional Alignment (Plan 214) ──────────────
//
// Read/Write/Enhance cycle for self-refining pruners.
// TED-Lite divergence metric + bandit-driven threshold/topology adjustment +
// editable ConstraintPruner with bidirectional editing.
//
// Feature gates: `ted_lite` (P1), `coexplain_pruner` (P2+3)

#[cfg(feature = "ted_lite")]
pub mod ted_lite;

#[cfg(feature = "ted_lite")]
pub use ted_lite::PrunerDivergence;

#[cfg(feature = "coexplain_pruner")]
pub mod self_refining;

#[cfg(feature = "coexplain_pruner")]
pub use self_refining::{
    PrunerAccuracy, TopologyAction, adjust_topology, compute_threshold_adjustment,
};

#[cfg(feature = "coexplain_pruner")]
pub mod editable_constraint;

#[cfg(feature = "coexplain_pruner")]
pub use editable_constraint::{
    DivergenceError, EditableConstraintPruner, PrunerSnapshot, RuleEdit, parse_rules,
};

#[cfg(feature = "coexplain_riir")]
pub mod riir_feedback;

#[cfg(feature = "coexplain_riir")]
pub use riir_feedback::{
    CuratorIngestion, CuratorRule, RuleBandit, TranslationRule, WorkloadRoute, classify_workload,
    extract_translation_rules,
};

#[cfg(feature = "regime_transition")]
pub mod regime_transition;

#[cfg(feature = "regime_transition")]
pub use regime_transition::{
    AdversarialBreaker, CollapseClassifier, CollapseType, DDTreeStats, FailurePattern, FailureRule,
    GateResult, ProvenanceChain, ProvenanceStep, PrunerType, RegimeCollapseClassifier,
    RegimeTransitionGate, RegimeTransitionScheduler, TransitionDeferred, TransportResult,
};

#[cfg(feature = "regime_transition")]
pub mod four_regime_router;

#[cfg(feature = "regime_transition")]
pub use four_regime_router::{FourRegimeRouter, Heaviness, Regime, RegimeArm, RegimeFeatures};
