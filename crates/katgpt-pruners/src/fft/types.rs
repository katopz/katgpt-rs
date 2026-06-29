//! Core types for FFT Tactics Arena.
//!
//! Enhanced from fft_01_arena.rs with ATB timing, status effects,
//! and expanded class/action set.

use std::fmt;

// ── Constants ──────────────────────────────────────────────────

pub const GRID_W: i32 = 8;
pub const GRID_H: i32 = 8;
pub const TURN_LIMIT: u32 = 200;
pub const POTION_HP: i32 = 30;
pub const BASE_HIT_RATE: f32 = 0.90;
pub const MAGIC_HIT_RATE: f32 = 0.95;
pub const BLACK_MAGIC_MP: i32 = 15;
pub const WHITE_MAGIC_MP: i32 = 10;
pub const CURE_POISON_MP: i32 = 5;
pub const ESUNA_MP: i32 = 15;
pub const DISPEL_MP: i32 = 10;
pub const DEFEND_MP_RECOVERY: i32 = 5;
pub const CT_THRESHOLD: f32 = 100.0;
pub const BASE_CT_FILL: f32 = 10.0;
pub const POISON_CHANCE: f32 = 0.30;

// ── Class ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    Knight,
    Archer,
    BlackMage,
    WhiteMage,
    Monk,
    TimeMage,
}

impl Class {
    pub fn stats(self) -> Stats {
        match self {
            Self::Knight => Stats {
                max_hp: 120,
                max_mp: 20,
                speed: 3,
                atk: 14,
                def: 12,
                mag: 4,
                range: 1,
                move_range: 3,
                ct_speed: 3.0,
            },
            Self::Archer => Stats {
                max_hp: 80,
                max_mp: 30,
                speed: 5,
                atk: 10,
                def: 6,
                mag: 6,
                range: 4,
                move_range: 3,
                ct_speed: 5.0,
            },
            Self::BlackMage => Stats {
                max_hp: 70,
                max_mp: 60,
                speed: 4,
                atk: 4,
                def: 4,
                mag: 16,
                range: 3,
                move_range: 2,
                ct_speed: 4.0,
            },
            Self::WhiteMage => Stats {
                max_hp: 80,
                max_mp: 70,
                speed: 4,
                atk: 4,
                def: 6,
                mag: 14,
                range: 3,
                move_range: 2,
                ct_speed: 4.0,
            },
            Self::Monk => Stats {
                max_hp: 110,
                max_mp: 30,
                speed: 4,
                atk: 16,
                def: 8,
                mag: 6,
                range: 1,
                move_range: 3,
                ct_speed: 4.5,
            },
            Self::TimeMage => Stats {
                max_hp: 75,
                max_mp: 80,
                speed: 4,
                atk: 4,
                def: 4,
                mag: 12,
                range: 3,
                move_range: 2,
                ct_speed: 4.0,
            },
        }
    }

    pub fn emoji(self) -> &'static str {
        match self {
            Self::Knight => "⚔️",
            Self::Archer => "🏹",
            Self::BlackMage => "🔮",
            Self::WhiteMage => "✨",
            Self::Monk => "👊",
            Self::TimeMage => "⏳",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Knight => "Knight",
            Self::Archer => "Archer",
            Self::BlackMage => "BMage",
            Self::WhiteMage => "WMage",
            Self::Monk => "Monk",
            Self::TimeMage => "TMage",
        }
    }

    pub fn all() -> [Self; 6] {
        [
            Self::Knight,
            Self::Archer,
            Self::BlackMage,
            Self::WhiteMage,
            Self::Monk,
            Self::TimeMage,
        ]
    }
}

// ── Team ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Team {
    Party,
    Enemy,
}

impl fmt::Display for Team {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Party => write!(f, "Party"),
            Self::Enemy => write!(f, "Enemy"),
        }
    }
}

// ── ActionType ─────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionType {
    Attack,
    Defend,
    BlackMagic,
    WhiteMagic,
    Potion,
    Wait,
    CurePoison,
    Esuna,
    Dispel,
}

impl ActionType {
    pub const fn all() -> [Self; 9] {
        [
            Self::Attack,
            Self::Defend,
            Self::BlackMagic,
            Self::WhiteMagic,
            Self::Potion,
            Self::Wait,
            Self::CurePoison,
            Self::Esuna,
            Self::Dispel,
        ]
    }

    pub const fn count() -> usize {
        9
    }

    pub const fn as_usize(self) -> usize {
        match self {
            Self::Attack => 0,
            Self::Defend => 1,
            Self::BlackMagic => 2,
            Self::WhiteMagic => 3,
            Self::Potion => 4,
            Self::Wait => 5,
            Self::CurePoison => 6,
            Self::Esuna => 7,
            Self::Dispel => 8,
        }
    }

    pub fn mp_cost(self) -> i32 {
        match self {
            Self::BlackMagic => BLACK_MAGIC_MP,
            Self::WhiteMagic => WHITE_MAGIC_MP,
            Self::CurePoison => CURE_POISON_MP,
            Self::Esuna => ESUNA_MP,
            Self::Dispel => DISPEL_MP,
            _ => 0,
        }
    }

    pub fn is_magic(self) -> bool {
        matches!(
            self,
            Self::BlackMagic | Self::WhiteMagic | Self::CurePoison | Self::Esuna | Self::Dispel
        )
    }
}

impl From<usize> for ActionType {
    fn from(v: usize) -> Self {
        match v {
            0 => Self::Attack,
            1 => Self::Defend,
            2 => Self::BlackMagic,
            3 => Self::WhiteMagic,
            4 => Self::Potion,
            5 => Self::Wait,
            6 => Self::CurePoison,
            7 => Self::Esuna,
            _ => Self::Dispel,
        }
    }
}

impl fmt::Display for ActionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Attack => "Attack",
            Self::Defend => "Defend",
            Self::BlackMagic => "Fire",
            Self::WhiteMagic => "Heal",
            Self::Potion => "Potion",
            Self::Wait => "Wait",
            Self::CurePoison => "CurePsn",
            Self::Esuna => "Esuna",
            Self::Dispel => "Dispel",
        };
        write!(f, "{s}")
    }
}

// ── Stats ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub max_hp: i32,
    pub max_mp: i32,
    pub speed: i32,
    pub atk: i32,
    pub def: i32,
    pub mag: i32,
    pub range: i32,
    pub move_range: i32,
    pub ct_speed: f32,
}

// ── Position ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

impl Pos {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn manhattan(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }

    pub fn in_bounds(self) -> bool {
        self.x >= 0 && self.x < GRID_W && self.y >= 0 && self.y < GRID_H
    }
}

// ── Unit ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Unit {
    pub id: u8,
    pub class: Class,
    pub team: Team,
    pub hp: i32,
    pub mp: i32,
    pub stats: Stats,
    pub pos: Pos,
    pub alive: bool,
    pub defending: bool,
    pub has_potion: bool,
    pub ct_gauge: f32,
}

impl Unit {
    pub fn new(id: u8, class: Class, team: Team, pos: Pos) -> Self {
        let stats = class.stats();
        Self {
            id,
            class,
            team,
            hp: stats.max_hp,
            mp: stats.max_mp / 2,
            stats,
            pos,
            alive: true,
            defending: false,
            has_potion: true,
            ct_gauge: 0.0,
        }
    }

    pub fn hp_pct(&self) -> f32 {
        self.hp as f32 / self.stats.max_hp as f32
    }

    pub fn can_afford(&self, action: ActionType) -> bool {
        match action {
            ActionType::Potion => self.has_potion,
            a if a.mp_cost() > 0 => self.mp >= a.mp_cost(),
            _ => true,
        }
    }

    pub fn spend(&mut self, action: ActionType) {
        match action {
            ActionType::Potion => self.has_potion = false,
            a => {
                let cost = a.mp_cost();
                if cost > 0 {
                    self.mp -= cost;
                }
            }
        }
    }
}

// ── Action ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Action {
    pub action_type: ActionType,
    pub target_id: Option<u8>,
    pub move_to: Option<Pos>,
}

// ── Game Event ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum GameEvent {
    DamageDealt {
        attacker: u8,
        target: u8,
        damage: i32,
    },
    Healed {
        healer: u8,
        target: u8,
        amount: i32,
    },
    Missed {
        attacker: u8,
        target: u8,
    },
    UnitDied {
        unit: u8,
        killer: u8,
    },
    EffectApplied {
        target: u8,
        effect: String,
        duration: u8,
    },
    EffectExpired {
        target: u8,
        effect: String,
    },
    EffectTicked {
        target: u8,
        effect: String,
        damage: i32,
    },
    DebuffCured {
        healer: u8,
        target: u8,
        effect: String,
    },
    BuffDispelled {
        caster: u8,
        target: u8,
        effect: String,
    },
}

// ── TFT Types (Plan 055) ──────────────────────────────────────

#[cfg(feature = "g_zero")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TftMode {
    Nice,
    Retaliatory { target: u8, ticks_left: u8 },
}

#[cfg(feature = "g_zero")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvokeLevel {
    None,
    Personal(u8),  // attacker id
    Team(u8),      // attacker id
    Escalated(u8), // attacker id (killer)
}

#[cfg(feature = "g_zero")]
#[derive(Clone, Debug)]
pub struct PartyTftState {
    pub provoked_by: Option<u8>,
    pub provoke_level: ProvokeLevel,
    pub forgive_timer: u8,
    pub escalation_count: u8,
    pub generous_chance: f32,
}

#[cfg(feature = "g_zero")]
impl Default for PartyTftState {
    fn default() -> Self {
        Self {
            provoked_by: None,
            provoke_level: ProvokeLevel::None,
            forgive_timer: 0,
            escalation_count: 0,
            generous_chance: 0.10,
        }
    }
}

#[cfg(feature = "g_zero")]
impl PartyTftState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_generous_chance(mut self, chance: f32) -> Self {
        self.generous_chance = chance;
        self
    }

    pub fn reset(&mut self) {
        self.provoked_by = None;
        self.provoke_level = ProvokeLevel::None;
        self.forgive_timer = 0;
        self.escalation_count = 0;
    }
}

#[cfg(feature = "g_zero")]
#[derive(Clone, Debug)]
pub struct UnitTftState {
    pub mode: TftMode,
    pub last_attacker: Option<u8>,
    pub class: Class,
}

#[cfg(feature = "g_zero")]
impl UnitTftState {
    pub fn new(class: Class) -> Self {
        Self {
            mode: TftMode::Nice,
            last_attacker: None,
            class,
        }
    }

    pub fn reset(&mut self) {
        self.mode = TftMode::Nice;
        self.last_attacker = None;
    }
}
