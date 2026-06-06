//! Bomberman HL Arena — 4-Player Heuristic Learning Proof (Plan 033)
//!
//! bevy_ecs standalone + ratatui TUI arena where 4 AI players compete,
//! each using progressively more HL technology.

pub mod arena;
pub mod gate_player;
pub mod players;
pub mod replay;
pub mod replay_backward;
pub mod systems;

#[cfg(feature = "bomber-agent")]
pub mod validator_agent;

pub mod arena_runner;
#[cfg(feature = "g_zero")]
pub mod g_zero_player;
#[cfg(feature = "rmsd_distill")]
pub mod rmsd_player;
#[cfg(feature = "ropd_rubric")]
pub mod rubric_player;
#[cfg(feature = "sdar_gate")]
pub mod sdar_player;
#[cfg(feature = "sdpg_bandit")]
pub mod sdpg_player;
#[cfg(feature = "sr2am_configurator")]
pub mod sr2am_player;
#[cfg(feature = "g_zero")]
pub mod tft_player;
#[cfg(feature = "vpd_em_distill")]
pub mod vpd_player;
#[cfg(feature = "bomber-wasm")]
pub mod wasm_pruner;
#[cfg(feature = "bomber-wasm")]
pub mod wasm_state;

#[cfg(feature = "event_log")]
pub mod event_log_player;

#[cfg(feature = "skill_lifecycle")]
pub mod skill_lifecycle_player;

pub use arena::ArenaGrid;
pub use gate_player::GatePlayer;
pub use players::{BomberPlayer, GreedyPlayer, HLPlayer, RandomPlayer, ValidatorPlayer};
pub use replay_backward::{BackwardSample, BackwardWalkResult, ReplayBackwardWalker};

#[cfg(feature = "bomber-agent")]
pub use validator_agent::{
    AgentLoop, AgentLoopResult, ArenaEvaluation, FailureTrace, RulePlayer, TemplateProposer,
    ValidatorCandidate, ValidatorRule, evaluate_validator, propose_from_trace,
};

#[cfg(feature = "bandit")]
pub use crate::pruners::SharedBanditStats;

pub use arena_runner::{BomberArenaConfig, BomberRoundResult, run_bomber_game, run_bomber_matchup};
#[cfg(feature = "g_zero")]
pub use g_zero_player::GZeroPlayer;
#[cfg(feature = "bomber-wasm")]
pub use players::{LoraPlayer, LoraWasmPlayer, NNPlayer, create_players_with_wasm, is_safe_action};
#[cfg(feature = "rmsd_distill")]
pub use rmsd_player::RmsdPlayer;
#[cfg(feature = "ropd_rubric")]
pub use rubric_player::RubricPlayer;
#[cfg(feature = "sdar_gate")]
pub use sdar_player::SdarPlayer;
#[cfg(feature = "sdpg_bandit")]
pub use sdpg_player::SdpgPlayer;
#[cfg(feature = "sr2am_configurator")]
pub use sr2am_player::Sr2amPlayer;
pub use systems::*;
#[cfg(feature = "g_zero")]
pub use tft_player::TftPlayer;
#[cfg(feature = "vpd_em_distill")]
pub use vpd_player::VpdPlayer;
#[cfg(feature = "bomber-wasm")]
pub use wasm_state::{ZeroCopyStateBuffer, serialize_grid_only, serialize_into_buffer};

#[cfg(feature = "event_log")]
pub use event_log_player::{BomberEventLog, BomberForkDiff, BomberPos};

#[cfg(feature = "skill_lifecycle")]
pub use skill_lifecycle_player::{LifecycleStats, SkillLifecyclePlayer};

use std::fmt;

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────

pub const ARENA_W: usize = 13;
pub const ARENA_H: usize = 13;
pub const BOMB_FUSE_TICKS: u32 = 4;
pub const DEFAULT_BLAST_RANGE: u32 = 2;
pub const DEFAULT_MAX_BOMBS: u8 = 1;
pub const DEFAULT_SPEED: u8 = 1;
pub const TICK_LIMIT: u32 = 500;
pub const DESTRUCTIBLE_FILL: f32 = 0.40;
pub const SPAWN_POSITIONS: [(i32, i32); 4] = [(1, 1), (11, 1), (1, 11), (11, 11)];

// ── Action ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BomberAction {
    Up,
    Down,
    Left,
    Right,
    Bomb,
    Wait,
    /// Detonate all remote bombs owned by this player.
    Detonate,
}

impl BomberAction {
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
            Self::Left => 2,
            Self::Right => 3,
            Self::Bomb => 4,
            Self::Wait => 5,
            Self::Detonate => 6,
        }
    }

    pub fn all() -> [BomberAction; 7] {
        [
            Self::Up,
            Self::Down,
            Self::Left,
            Self::Right,
            Self::Bomb,
            Self::Wait,
            Self::Detonate,
        ]
    }
}

impl fmt::Display for BomberAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Up => "↑",
            Self::Down => "↓",
            Self::Left => "←",
            Self::Right => "→",
            Self::Bomb => "💣",
            Self::Wait => "⏸",
            Self::Detonate => "💥",
        };
        write!(f, "{s}")
    }
}

impl From<usize> for BomberAction {
    fn from(val: usize) -> Self {
        match val {
            0 => Self::Up,
            1 => Self::Down,
            2 => Self::Left,
            3 => Self::Right,
            4 => Self::Bomb,
            5 => Self::Wait,
            6 => Self::Detonate,
            _ => Self::Wait,
        }
    }
}

// ── Power-Up ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerUpKind {
    BombUp,
    FireUp,
    SpeedUp,
}

impl fmt::Display for PowerUpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::BombUp => "🌠",
            Self::FireUp => "🎇",
            Self::SpeedUp => "👟",
        };
        write!(f, "{s}")
    }
}

// ── Grid Cell ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Floor,
    FixedWall,
    DestructibleWall,
    PowerUpHidden(PowerUpKind),
}

// ── Components ─────────────────────────────────────────────────

#[derive(Component)]
pub struct Player {
    pub id: u8,
}

/// Bomb type determining blast behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BombType {
    /// Default: fuse-based, stops at walls.
    #[default]
    Timed,
    /// Blast continues through destructible walls.
    Piercing,
    /// Detonates on player action.
    Remote,
    /// Invisible until stepped on, 1-range instant blast.
    Landmine,
}

impl BombType {
    /// Encode bomb type as u8 for WASM/replay serialization.
    ///
    /// Encoding: 0=Timed, 1=Piercing, 2=Remote, 3=Landmine.
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Timed => 0,
            Self::Piercing => 1,
            Self::Remote => 2,
            Self::Landmine => 3,
        }
    }

    /// Decode bomb type from u8, defaulting to `Timed` for unknown values.
    ///
    /// Provides backward compat for old replays/WASM state without bomb_type.
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Timed,
            1 => Self::Piercing,
            2 => Self::Remote,
            3 => Self::Landmine,
            _ => Self::Timed,
        }
    }
}

/// Bomb component with type information.
#[derive(Clone, Copy, Component, Debug, Default)]
pub struct Bomb {
    pub bomb_type: BombType,
}

impl Bomb {
    /// Create a new bomb with default type (Timed).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a bomb with a specific type.
    pub fn with_type(bomb_type: BombType) -> Self {
        Self { bomb_type }
    }
}

#[derive(Component)]
pub struct PowerUp {
    pub kind: PowerUpKind,
}

#[derive(Component)]
pub struct Blast;

#[derive(Component)]
pub struct DestructibleWall;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct GridPos {
    pub x: i32,
    pub y: i32,
}

#[derive(Component)]
pub struct BombFuse {
    pub owner: Entity,
    pub ticks_remaining: u32,
}

#[derive(Component)]
pub struct BombRange {
    pub cells: u32,
}

#[derive(Component)]
pub struct BombCount {
    pub max: u8,
    pub active: u8,
}

#[derive(Component)]
pub struct Speed {
    pub cells_per_tick: u8,
}

#[derive(Component, Default)]
pub struct Alive;

// ── Resources ──────────────────────────────────────────────────

#[derive(Resource)]
pub struct GameRng {
    pub seed: u64,
}

#[derive(Resource, Default)]
pub struct TickCounter {
    pub tick: u32,
}

impl TickCounter {
    pub fn inc(&mut self) -> u32 {
        self.tick += 1;
        self.tick
    }
}

#[derive(Resource, Default)]
pub struct ScoreBoard {
    pub scores: [i32; 4],
}

impl ScoreBoard {
    pub fn add(&mut self, player: u8, points: i32) {
        if let 0..=3 = player {
            self.scores[player as usize] += points
        }
    }
}

#[derive(Resource)]
pub struct PlayerEntities {
    pub entities: [Entity; 4],
}

// ── Events ─────────────────────────────────────────────────────

#[derive(Event, Clone, Debug)]
pub enum GameEvent {
    PlayerMoved {
        player: u8,
        from: (i32, i32),
        to: (i32, i32),
    },
    BombPlaced {
        player: u8,
        pos: (i32, i32),
    },
    BombExploded {
        pos: (i32, i32),
        range: u32,
    },
    PlayerKilled {
        victim: u8,
        killer: Option<u8>,
    },
    PowerUpCollected {
        player: u8,
        kind: PowerUpKind,
        pos: (i32, i32),
    },
    PowerUpRevealed {
        pos: (i32, i32),
        kind: PowerUpKind,
    },
    WallDestroyed {
        pos: (i32, i32),
    },
    RoundEnd {
        survivors: Vec<u8>,
    },
}

// ── Frozen Knowledge (Plan 092) ────────────────────────────────

/// Frozen bomber bandit state for disk persistence.
///
/// Captures only learned knowledge (Q-values, visits, compressed flags).
/// Transient game state (bombs, positions, opponents) is NOT persisted.
///
/// Uses `u8` instead of `bool` for compressed flags to avoid `repr(C)` padding issues.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BomberFrozenBandit {
    /// Magic bytes: b"BDTB" (Bomber DaTa Bandit)
    pub magic: [u8; 4],
    /// Format version: 1
    pub version: u32,
    /// Q-value estimates per action (7 actions: Up/Down/Left/Right/Bomb/Wait/Detonate)
    pub q_values: [f32; 7],
    /// Visit counts per action
    pub visits: [u32; 7],
    /// Total bandit pulls across all actions
    pub total_pulls: u32,
    /// Compressed flags per action (0=active, 1=compressed)
    pub compressed: [u8; 7],
    /// Reserved for future use
    pub reserved: [u8; 16],
}

impl BomberFrozenBandit {
    /// Magic bytes for bomber bandit format.
    pub const MAGIC: [u8; 4] = *b"BDTB";
    /// Current format version.
    pub const VERSION: u32 = 1;

    /// Create a new empty frozen bandit with magic and version set.
    pub fn new_empty() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            q_values: [0.0; 7],
            visits: [0; 7],
            total_pulls: 0,
            compressed: [0; 7],
            reserved: [0; 16],
        }
    }

    /// Validate magic bytes and version.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "Invalid magic: expected {:?}, got {:?}",
                Self::MAGIC,
                self.magic
            ));
        }
        if self.version != Self::VERSION {
            return Err(format!(
                "Unsupported version: expected {}, got {}",
                Self::VERSION,
                self.version
            ));
        }
        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_roundtrip() {
        for (i, expected) in BomberAction::all().into_iter().enumerate() {
            let action = BomberAction::from(i);
            assert_eq!(action, expected);
            assert_eq!(action.as_usize(), i);
        }
    }

    #[test]
    fn action_from_overflow_clamps_to_wait() {
        assert_eq!(BomberAction::from(99), BomberAction::Wait);
    }

    #[test]
    fn scoreboard_add_valid() {
        let mut sb = ScoreBoard::default();
        sb.add(0, 10);
        sb.add(2, -5);
        sb.add(3, 7);
        assert_eq!(sb.scores, [10, 0, -5, 7]);
    }

    #[test]
    fn scoreboard_add_ignores_invalid_player() {
        let mut sb = ScoreBoard::default();
        sb.add(4, 100);
        assert_eq!(sb.scores, [0, 0, 0, 0]);
    }

    #[test]
    fn tick_counter_inc() {
        let mut tc = TickCounter::default();
        assert_eq!(tc.inc(), 1);
        assert_eq!(tc.inc(), 2);
        assert_eq!(tc.tick, 2);
    }

    #[test]
    fn powerup_display() {
        assert_eq!(format!("{}", PowerUpKind::BombUp), "🌠");
        assert_eq!(format!("{}", PowerUpKind::FireUp), "🎇");
        assert_eq!(format!("{}", PowerUpKind::SpeedUp), "👟");
    }

    #[test]
    fn action_display() {
        assert_eq!(format!("{}", BomberAction::Up), "↑");
        assert_eq!(format!("{}", BomberAction::Bomb), "💣");
        assert_eq!(format!("{}", BomberAction::Wait), "⏸");
    }
}
