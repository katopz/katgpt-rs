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

pub mod freeze;

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
pub use review_metrics::{EntropyAnomalySummary, ReviewMetrics, ReviewStrategy, ReviewSummary};

#[cfg(feature = "bandit")]
pub use hot_swap::HotSwapPruner;

#[cfg(feature = "bandit")]
pub use regression::{GoldenTrace, RegressionResult, RegressionSuite, ReplayReward};

#[cfg(feature = "bandit")]
pub use trial_log::{SharedTrialLog, TrialLog, TrialRecord, TrialSummary};

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

#[cfg(any(feature = "bomber", feature = "fft", feature = "tes_loop"))]
pub mod arena;

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

#[cfg(feature = "fft")]
pub use fft::{
    Action, ActionType, ActiveEffect, BattleState, Class, FftPlayer, GameEvent, GreedyFFTPlayer,
    HLFFTPlayer, Pos, Stats, StatusEffect, Team, Unit, ValidatorFFTPlayer, resolve_action,
};
