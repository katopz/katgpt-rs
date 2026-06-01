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

#[cfg(feature = "safe_bandit")]
pub use safe_phased::SafePhasedState;

#[cfg(feature = "sr2am_configurator")]
pub use configurator_bandit::{ConfiguratorBandit, ExplorationOutcome};

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

#[cfg(feature = "mech_attribution")]
pub mod mech_attribution;

#[cfg(feature = "mech_attribution")]
pub use mech_attribution::{
    ActivationInfluenceProxy, CatalystPattern, CatalystTemplate, InfluenceConfig,
    MechInfluenceScore, batch_influence_rank, catalyst_score, detect_catalyst_pattern,
    extract_template, generate_synthetic,
};
