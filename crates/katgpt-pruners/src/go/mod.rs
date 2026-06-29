//! Go game integration — AutoGo API bridge, GameState, and tournament infrastructure.
//!
//! Plan 065: AutoGo Distillation
//!
//! ## Modules
//!
//! - [`types`] — `GoAction`, `GoCell` enums
//! - [`state`] — `GoState` board with full Go logic + `GameState` trait impl + `GoHeuristic`
//! - [`autogo_client`] — REST API client for AutoGo's `play.py` server
//! - [`replay`] — Game recording and deterministic playback
//! - [`players`] — `GoPlayer` trait + 6 AI player implementations (Plan 065 Phase 2)
//! - [`tournament`] — Head-to-head tournament runner against AutoGo agents (Plan 065 Phase 3)
//! - [`g_zero_player`] — G-Zero self-play with HintDelta and absorb-compress (Plan 065 Phase 4)
//! - [`autoresearch`] — AutoResearch loop for automated hyperparameter search (Plan 065 Phase 5)

pub mod analytics;
pub mod autogo_client;
pub mod autoresearch;
pub mod g_zero_player;
pub mod players;
pub mod replay;
pub mod replay_writer;
pub mod state;
pub mod tournament;
pub mod types;
pub mod utils;

#[cfg(feature = "event_log")]
pub mod event_log_player;

#[cfg(all(feature = "sdpg_bandit", feature = "go"))]
pub mod sdpg_player;

// ── Re-exports ─────────────────────────────────────────────────

// Types
pub use types::{GoAction, GoCell, GoFrozenBandit, GoFrozenTemplates};

// State
pub use state::{DEFAULT_KOMI, GoHeuristic, GoState};

// Replay
pub use replay::{GoReplay, MoveRecord, ReplayError};

// API Client
pub use autogo_client::{AutoGoClient, AutoGoError, AutoGoGameState};

// Players
pub use players::{
    GoGZeroPlayer, GoGreedyPlayer, GoHLPlayer, GoMctsPlayer, GoMoveCategory, GoPlayer,
    GoRandomPlayer, GoTemplate, GoValidatorPlayer,
};

// Analytics
pub use analytics::{
    GoGameAnalytics, RawGoAction, RawGoSample, compute_analytics, samples_to_replay,
    split_samples_into_games,
};

// Replay Writer (Plan 271 T2.1)
pub use replay_writer::{GameSampleCollector, GoActionType, GoReplayWriter, JsonlGoSample};

// G-Zero Self-Play
pub use g_zero_player::{
    GoDeltaGatedAbsorbCompress, GoDeltaGatedConfig, GoGZeroSelfPlayConfig, GoGZeroSelfPlayResults,
    GoSelfPlayResult, GoTemplateProposer, MoveDelta, compute_go_delta, compute_go_delta_into,
    run_gzero_selfplay,
};

// Tournament
pub use tournament::{
    AutoGoProxyPlayer, GameOutcome, GameResult, GoPlayerType, GoTournamentConfig,
    GoTournamentResult, ParsedResult, TournamentDef, parse_go_result, print_batch_table,
    run_tournament, run_tournament_batch,
};

// AutoResearch
pub use autoresearch::{
    ArmStatus, AutoResearchConfig, AutoResearchResult, BaselinePlayer, GoResearchConfig,
    ResearchArm, TrialLog, run_autoresearch,
};

// Event Log (Plan 124)
#[cfg(feature = "event_log")]
pub use event_log_player::{GoEventLog, GoForkDiff};

// SDPG Player (Plan 194)
#[cfg(all(feature = "sdpg_bandit", feature = "go"))]
pub use sdpg_player::GoSdpgPlayer;
