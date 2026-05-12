//! FFT Tactics Arena — 4v4 Turn-Based Battle (Plan 047)
//!
//! Final Fantasy Tactics-inspired headless battle arena.
//! 8 units (4v4) with classes, AI strategies, HP/MP, and speed-based turns.
//!
//! Party: ⚔️Knight-Random  🏹Archer-Greedy  🔮BMage-Validator  ✨WMage-HL
//! Enemy: ⚔️Knight-HL     🏹Archer-Validator 🔮BMage-Greedy   ✨WMage-Random
//!
//! Run: `cargo run --example fft_01_arena`

use std::any::Any;
use std::cmp::Ordering;
use std::fmt;

use fastrand::Rng;

// ── Constants ──────────────────────────────────────────────────

const GRID_W: i32 = 8;
const GRID_H: i32 = 8;
const ROUNDS: usize = 100;
const TURN_LIMIT: u32 = 120;
const POTION_HP: i32 = 30;
const BASE_HIT_RATE: f32 = 0.90;
const MAGIC_HIT_RATE: f32 = 0.95;
const BLACK_MAGIC_MP: i32 = 15;
const WHITE_MAGIC_MP: i32 = 10;
const DEFEND_MP_RECOVERY: i32 = 5;

// ── Class ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Class {
    Knight,
    Archer,
    BlackMage,
    WhiteMage,
}

impl Class {
    fn stats(self) -> Stats {
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
            },
        }
    }

    fn emoji(self) -> &'static str {
        match self {
            Self::Knight => "⚔️",
            Self::Archer => "🏹",
            Self::BlackMage => "🔮",
            Self::WhiteMage => "✨",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Knight => "Knight",
            Self::Archer => "Archer",
            Self::BlackMage => "BMage",
            Self::WhiteMage => "WMage",
        }
    }
}

// ── Team ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Team {
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
enum ActionType {
    Attack,
    Defend,
    BlackMagic,
    WhiteMagic,
    Potion,
    Wait,
}

impl ActionType {
    const fn all() -> [Self; 6] {
        [
            Self::Attack,
            Self::Defend,
            Self::BlackMagic,
            Self::WhiteMagic,
            Self::Potion,
            Self::Wait,
        ]
    }

    const fn as_usize(self) -> usize {
        match self {
            Self::Attack => 0,
            Self::Defend => 1,
            Self::BlackMagic => 2,
            Self::WhiteMagic => 3,
            Self::Potion => 4,
            Self::Wait => 5,
        }
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
            _ => Self::Wait,
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
        };
        write!(f, "{s}")
    }
}

// ── Stats ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Stats {
    max_hp: i32,
    max_mp: i32,
    speed: i32,
    atk: i32,
    def: i32,
    mag: i32,
    range: i32,
    move_range: i32,
}

// ── Position ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pos {
    x: i32,
    y: i32,
}

impl Pos {
    const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn manhattan(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }

    fn in_bounds(self) -> bool {
        self.x >= 0 && self.x < GRID_W && self.y >= 0 && self.y < GRID_H
    }
}

// ── Unit ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Unit {
    id: u8,
    _class: Class,
    team: Team,
    hp: i32,
    mp: i32,
    stats: Stats,
    pos: Pos,
    alive: bool,
    defending: bool,
    has_potion: bool,
}

impl Unit {
    fn new(id: u8, class: Class, team: Team, pos: Pos) -> Self {
        let stats = class.stats();
        Self {
            id,
            _class: class,
            team,
            hp: stats.max_hp,
            mp: stats.max_mp,
            stats,
            pos,
            alive: true,
            defending: false,
            has_potion: true,
        }
    }

    fn hp_pct(&self) -> f32 {
        self.hp as f32 / self.stats.max_hp as f32
    }

    fn can_afford(&self, action: ActionType) -> bool {
        match action {
            ActionType::BlackMagic => self.mp >= BLACK_MAGIC_MP,
            ActionType::WhiteMagic => self.mp >= WHITE_MAGIC_MP,
            ActionType::Potion => self.has_potion,
            _ => true,
        }
    }

    fn spend(&mut self, action: ActionType) {
        match action {
            ActionType::BlackMagic => self.mp -= BLACK_MAGIC_MP,
            ActionType::WhiteMagic => self.mp -= WHITE_MAGIC_MP,
            ActionType::Potion => self.has_potion = false,
            _ => {}
        }
    }
}

// ── Action ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Action {
    action_type: ActionType,
    target_id: Option<u8>,
    move_to: Option<Pos>,
}

// ── Game Event ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum GameEvent {
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
}

// ── Battle State ───────────────────────────────────────────────

struct BattleState {
    units: Vec<Unit>,
    events: Vec<GameEvent>,
    tick: u32,
}

impl BattleState {
    fn new() -> Self {
        let party_pos = [
            Pos::new(1, 1),
            Pos::new(1, 6),
            Pos::new(0, 3),
            Pos::new(0, 5),
        ];
        let enemy_pos = [
            Pos::new(6, 1),
            Pos::new(6, 6),
            Pos::new(7, 3),
            Pos::new(7, 5),
        ];
        let party_classes = [
            Class::Knight,
            Class::Archer,
            Class::BlackMage,
            Class::WhiteMage,
        ];
        let enemy_classes = [
            Class::Knight,
            Class::Archer,
            Class::BlackMage,
            Class::WhiteMage,
        ];

        let mut units = Vec::with_capacity(8);
        for (i, (&class, &pos)) in party_classes.iter().zip(&party_pos).enumerate() {
            units.push(Unit::new(i as u8, class, Team::Party, pos));
        }
        for (i, (&class, &pos)) in enemy_classes.iter().zip(&enemy_pos).enumerate() {
            units.push(Unit::new((i + 4) as u8, class, Team::Enemy, pos));
        }

        Self {
            units,
            events: Vec::new(),
            tick: 0,
        }
    }

    fn unit_at(&self, pos: Pos) -> Option<u8> {
        self.units
            .iter()
            .find(|u| u.alive && u.pos == pos)
            .map(|u| u.id)
    }

    fn reachable_positions(&self, unit_id: u8) -> Vec<Pos> {
        let unit = &self.units[unit_id as usize];
        if !unit.alive {
            return Vec::new();
        }

        let mut result = Vec::new();
        let range = unit.stats.move_range;
        for dx in -range..=range {
            for dy in -range..=range {
                if dx.abs() + dy.abs() > range || dx == 0 && dy == 0 {
                    continue;
                }
                let pos = Pos::new(unit.pos.x + dx, unit.pos.y + dy);
                if pos.in_bounds() && self.unit_at(pos).is_none() {
                    result.push(pos);
                }
            }
        }
        result
    }

    fn targets_in_range(&self, pos: Pos, range: i32, target_team: Team) -> Vec<u8> {
        self.units
            .iter()
            .filter(|u| u.alive && u.team == target_team && u.pos.manhattan(pos) <= range)
            .map(|u| u.id)
            .collect()
    }

    fn turn_order(&self) -> Vec<u8> {
        let mut ids: Vec<u8> = self
            .units
            .iter()
            .filter(|u| u.alive)
            .map(|u| u.id)
            .collect();
        ids.sort_by(|a, b| {
            let ua = &self.units[*a as usize];
            let ub = &self.units[*b as usize];
            ub.stats.speed.cmp(&ua.stats.speed).then_with(|| a.cmp(b))
        });
        ids
    }

    fn check_winner(&self) -> Option<Team> {
        let party_alive = self.units.iter().any(|u| u.alive && u.team == Team::Party);
        let enemy_alive = self.units.iter().any(|u| u.alive && u.team == Team::Enemy);
        match (party_alive, enemy_alive) {
            (true, false) => Some(Team::Party),
            (false, true) => Some(Team::Enemy),
            (false, false) => Some(Team::Party),
            _ => None,
        }
    }

    fn enemy_team(team: Team) -> Team {
        match team {
            Team::Party => Team::Enemy,
            Team::Enemy => Team::Party,
        }
    }

    fn team_hp(&self, team: Team) -> i32 {
        self.units
            .iter()
            .filter(|u| u.alive && u.team == team)
            .map(|u| u.hp)
            .sum()
    }
}

// ── Action Resolution ──────────────────────────────────────────

fn resolve_action(state: &mut BattleState, unit_id: u8, action: &Action, rng: &mut Rng) {
    // Move first
    if let Some(to) = action.move_to {
        state.units[unit_id as usize].pos = to;
    }

    let stats = state.units[unit_id as usize].stats;
    let pos = state.units[unit_id as usize].pos;
    match action.action_type {
        ActionType::Attack => {
            let Some(&target_id) = action.target_id.as_ref() else {
                return;
            };
            let target_pos = state.units[target_id as usize].pos;
            if target_pos.manhattan(pos) > stats.range {
                return;
            }

            let atk = stats.atk;
            let def = state.units[target_id as usize].stats.def;
            let defending = state.units[target_id as usize].defending;
            let raw = (atk as f32 * 1.5 - def as f32 * 0.3).max(1.0) as i32;
            let damage = if defending {
                (raw as f32 * 0.5) as i32
            } else {
                raw
            };

            if rng.f32() < BASE_HIT_RATE {
                state.units[target_id as usize].hp -= damage;
                state.events.push(GameEvent::DamageDealt {
                    attacker: unit_id,
                    target: target_id,
                    damage,
                });
                check_death(state, target_id, unit_id);
            } else {
                state.events.push(GameEvent::Missed {
                    attacker: unit_id,
                    target: target_id,
                });
            }
        }
        ActionType::BlackMagic => {
            let Some(&target_id) = action.target_id.as_ref() else {
                return;
            };
            let target_pos = state.units[target_id as usize].pos;
            if target_pos.manhattan(pos) > stats.range {
                return;
            }

            state.units[unit_id as usize].spend(ActionType::BlackMagic);
            let mag = stats.mag;
            let def = state.units[target_id as usize].stats.def;
            let defending = state.units[target_id as usize].defending;
            let raw = (mag as f32 * 1.8 - def as f32 * 0.2).max(1.0) as i32;
            let damage = if defending {
                (raw as f32 * 0.5) as i32
            } else {
                raw
            };

            if rng.f32() < MAGIC_HIT_RATE {
                state.units[target_id as usize].hp -= damage;
                state.events.push(GameEvent::DamageDealt {
                    attacker: unit_id,
                    target: target_id,
                    damage,
                });
                check_death(state, target_id, unit_id);
            } else {
                state.events.push(GameEvent::Missed {
                    attacker: unit_id,
                    target: target_id,
                });
            }
        }
        ActionType::WhiteMagic => {
            let Some(&target_id) = action.target_id.as_ref() else {
                return;
            };
            let target_pos = state.units[target_id as usize].pos;
            if target_pos.manhattan(pos) > stats.range {
                return;
            }

            state.units[unit_id as usize].spend(ActionType::WhiteMagic);
            let heal = (stats.mag as f32 * 2.0) as i32;
            let target = &mut state.units[target_id as usize];
            let actual = heal.min(target.stats.max_hp - target.hp);
            target.hp += actual;
            state.events.push(GameEvent::Healed {
                healer: unit_id,
                target: target_id,
                amount: actual,
            });
        }
        ActionType::Defend => {
            state.units[unit_id as usize].defending = true;
            let u = &mut state.units[unit_id as usize];
            u.mp = (u.mp + DEFEND_MP_RECOVERY).min(u.stats.max_mp);
        }
        ActionType::Potion => {
            state.units[unit_id as usize].spend(ActionType::Potion);
            let target_id = action.target_id.unwrap_or(unit_id);
            let u = &mut state.units[target_id as usize];
            let actual = POTION_HP.min(u.stats.max_hp - u.hp);
            u.hp += actual;
        }
        ActionType::Wait => {}
    }
}

fn check_death(state: &mut BattleState, target_id: u8, killer_id: u8) {
    if state.units[target_id as usize].hp <= 0 {
        state.units[target_id as usize].hp = 0;
        state.units[target_id as usize].alive = false;
        state.events.push(GameEvent::UnitDied {
            unit: target_id,
            killer: killer_id,
        });
    }
}

// ── AI Trait ───────────────────────────────────────────────────

trait FftPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action;
    fn name(&self) -> &'static str;
    fn reset(&mut self) {}
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ── Helpers ────────────────────────────────────────────────────

fn weakest_target(state: &BattleState, targets: &[u8]) -> Option<u8> {
    targets
        .iter()
        .min_by_key(|&&id| state.units[id as usize].hp)
        .copied()
}

fn lowest_hp_ally(state: &BattleState, allies: &[u8]) -> Option<u8> {
    allies
        .iter()
        .min_by_key(|&&id| state.units[id as usize].hp)
        .copied()
}

fn nearest_enemy_pos(state: &BattleState, pos: Pos, team: Team) -> Option<Pos> {
    state
        .units
        .iter()
        .filter(|u| u.alive && u.team != team)
        .min_by_key(|u| pos.manhattan(u.pos))
        .map(|u| u.pos)
}

fn move_toward(reachable: &[Pos], target: Pos) -> Option<Pos> {
    reachable
        .iter()
        .min_by_key(|p| p.manhattan(target))
        .copied()
}

fn move_away(reachable: &[Pos], threat: Pos) -> Option<Pos> {
    reachable
        .iter()
        .max_by_key(|p| p.manhattan(threat))
        .copied()
}

// ── Random Player ──────────────────────────────────────────────

struct RandomPlayer;

impl FftPlayer for RandomPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let reachable = state.reachable_positions(unit_id);
        let move_to = reachable.get(rng.usize(..reachable.len().max(1))).copied();

        let enemy_team = BattleState::enemy_team(unit.team);
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);

        let mut options = vec![ActionType::Wait, ActionType::Defend];
        if !enemies.is_empty() {
            options.push(ActionType::Attack);
        }
        if !enemies.is_empty() && unit.can_afford(ActionType::BlackMagic) {
            options.push(ActionType::BlackMagic);
        }
        if !allies.is_empty() && unit.can_afford(ActionType::WhiteMagic) {
            options.push(ActionType::WhiteMagic);
        }
        if unit.can_afford(ActionType::Potion) {
            options.push(ActionType::Potion);
        }

        let action_type = options[rng.usize(..options.len())];
        let target_id = match action_type {
            ActionType::Attack | ActionType::BlackMagic => {
                enemies.get(rng.usize(..enemies.len().max(1))).copied()
            }
            ActionType::WhiteMagic => allies.get(rng.usize(..allies.len().max(1))).copied(),
            ActionType::Potion => Some(unit_id),
            _ => None,
        };

        Action {
            action_type,
            target_id,
            move_to,
        }
    }

    fn name(&self) -> &'static str {
        "Random"
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Greedy Player ──────────────────────────────────────────────

struct GreedyPlayer;

impl FftPlayer for GreedyPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, _rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let hp_pct = unit.hp_pct();
        let reachable = state.reachable_positions(unit_id);
        let enemy_team = BattleState::enemy_team(unit.team);

        let move_to = nearest_enemy_pos(state, unit.pos, unit.team)
            .and_then(|ep| move_toward(&reachable, ep));

        // Critical HP: potion
        if hp_pct < 0.3 && unit.can_afford(ActionType::Potion) {
            return Action {
                action_type: ActionType::Potion,
                target_id: Some(unit_id),
                move_to,
            };
        }

        // Attack weakest enemy
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        if let Some(target) = weakest_target(state, &enemies) {
            let dist = unit.pos.manhattan(state.units[target as usize].pos);
            if dist > 1 && unit.can_afford(ActionType::BlackMagic) {
                return Action {
                    action_type: ActionType::BlackMagic,
                    target_id: Some(target),
                    move_to,
                };
            }
            return Action {
                action_type: ActionType::Attack,
                target_id: Some(target),
                move_to,
            };
        }

        // Heal wounded ally
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
        if unit.can_afford(ActionType::WhiteMagic)
            && let Some(ally) = lowest_hp_ally(state, &allies)
            && state.units[ally as usize].hp_pct() < 0.7
        {
            return Action {
                action_type: ActionType::WhiteMagic,
                target_id: Some(ally),
                move_to,
            };
        }

        Action {
            action_type: ActionType::Defend,
            target_id: None,
            move_to,
        }
    }

    fn name(&self) -> &'static str {
        "Greedy"
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Validator Player ───────────────────────────────────────────

struct ValidatorPlayer;

impl FftPlayer for ValidatorPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, _rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let hp_pct = unit.hp_pct();
        let reachable = state.reachable_positions(unit_id);
        let enemy_team = BattleState::enemy_team(unit.team);

        // Critical HP: potion
        if hp_pct < 0.25 && unit.can_afford(ActionType::Potion) {
            return Action {
                action_type: ActionType::Potion,
                target_id: Some(unit_id),
                move_to: None,
            };
        }

        // Heal critical ally first
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
        if unit.can_afford(ActionType::WhiteMagic) {
            for &ally in &allies {
                if state.units[ally as usize].hp_pct() < 0.4 {
                    return Action {
                        action_type: ActionType::WhiteMagic,
                        target_id: Some(ally),
                        move_to: None,
                    };
                }
            }
        }

        // Attack if safe
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        if !enemies.is_empty() && (enemies.len() <= 2 || hp_pct > 0.5) {
            let target = weakest_target(state, &enemies);
            if unit.can_afford(ActionType::BlackMagic) {
                return Action {
                    action_type: ActionType::BlackMagic,
                    target_id: target,
                    move_to: None,
                };
            }
            return Action {
                action_type: ActionType::Attack,
                target_id: target,
                move_to: None,
            };
        }

        // Retreat if low HP
        let move_to = if hp_pct < 0.5 {
            nearest_enemy_pos(state, unit.pos, unit.team).and_then(|ep| move_away(&reachable, ep))
        } else {
            None
        };

        Action {
            action_type: ActionType::Defend,
            target_id: None,
            move_to,
        }
    }

    fn name(&self) -> &'static str {
        "Validator"
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── HL Player (Bandit Q-Learning) ─────────────────────────────

struct HLPlayer {
    q_values: [f32; 6],
    visits: [u32; 6],
    total_pulls: u32,
    epsilon: f32,
    last_action: Option<ActionType>,
}

impl HLPlayer {
    fn new() -> Self {
        Self {
            q_values: [0.0; 6],
            visits: [0; 6],
            total_pulls: 0,
            epsilon: 0.15,
            last_action: None,
        }
    }

    fn update_outcome(&mut self, survived: bool, kills: u32, damage_dealt: i32, healing_done: i32) {
        let reward = if survived { 1.0 } else { -2.0 }
            + kills as f32 * 0.5
            + damage_dealt as f32 * 0.01
            + healing_done as f32 * 0.005;

        if let Some(action) = self.last_action {
            let idx = action.as_usize();
            let alpha = 1.0 / (1.0 + self.visits[idx] as f32).sqrt();
            self.q_values[idx] += alpha * (reward - self.q_values[idx]);
        }

        self.epsilon = (self.epsilon * 0.995).max(0.05);
        self.last_action = None;
    }

    fn best_available(&self, available: &[ActionType]) -> ActionType {
        available
            .iter()
            .max_by(|a, b| {
                let qa = self.q_values[a.as_usize()];
                let qb = self.q_values[b.as_usize()];
                qa.partial_cmp(&qb).unwrap_or(Ordering::Equal)
            })
            .copied()
            .unwrap_or(ActionType::Wait)
    }
}

impl FftPlayer for HLPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let hp_pct = unit.hp_pct();
        let reachable = state.reachable_positions(unit_id);
        let enemy_team = BattleState::enemy_team(unit.team);

        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);

        let mut available = vec![ActionType::Wait, ActionType::Defend];
        if !enemies.is_empty() {
            available.push(ActionType::Attack);
        }
        if !enemies.is_empty() && unit.can_afford(ActionType::BlackMagic) {
            available.push(ActionType::BlackMagic);
        }
        if !allies.is_empty() && unit.can_afford(ActionType::WhiteMagic) {
            available.push(ActionType::WhiteMagic);
        }
        if unit.can_afford(ActionType::Potion) && hp_pct < 0.5 {
            available.push(ActionType::Potion);
        }

        let action_type = if rng.f32() < self.epsilon {
            available[rng.usize(..available.len())]
        } else {
            self.best_available(&available)
        };

        self.last_action = Some(action_type);
        self.visits[action_type.as_usize()] += 1;
        self.total_pulls += 1;

        let target_id = match action_type {
            ActionType::Attack | ActionType::BlackMagic => weakest_target(state, &enemies),
            ActionType::WhiteMagic => lowest_hp_ally(state, &allies),
            ActionType::Potion => Some(unit_id),
            _ => None,
        };

        let move_to = if let Some(tid) = target_id {
            let target_pos = state.units[tid as usize].pos;
            let range = match action_type {
                ActionType::Attack => unit.stats.range,
                ActionType::BlackMagic => unit.stats.range,
                _ => unit.stats.range + 2,
            };
            if unit.pos.manhattan(target_pos) <= range {
                None
            } else {
                move_toward(&reachable, target_pos)
            }
        } else {
            nearest_enemy_pos(state, unit.pos, unit.team).and_then(|ep| move_toward(&reachable, ep))
        };

        Action {
            action_type,
            target_id,
            move_to,
        }
    }

    fn name(&self) -> &'static str {
        "HL"
    }

    fn reset(&mut self) {
        self.last_action = None;
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Battle Runner ──────────────────────────────────────────────

struct BattleResult {
    winner: Option<Team>,
    kills: Vec<(u8, u8)>,
    ticks: u32,
    events: Vec<GameEvent>,
}

fn run_battle(players: &mut [Box<dyn FftPlayer>], rng: &mut Rng) -> BattleResult {
    let mut state = BattleState::new();

    for tick in 0..TURN_LIMIT {
        state.tick = tick;
        let turn_order = state.turn_order();

        for unit_id in turn_order {
            if !state.units[unit_id as usize].alive {
                continue;
            }

            state.units[unit_id as usize].defending = false;

            if let Some(winner) = state.check_winner() {
                let kills = extract_kills(&state.events);
                return BattleResult {
                    winner: Some(winner),
                    kills,
                    ticks: tick,
                    events: state.events,
                };
            }

            let action = players[unit_id as usize].select_action(unit_id, &state, rng);
            resolve_action(&mut state, unit_id, &action, rng);
        }
    }

    // Timeout: compare HP
    let winner = match state.team_hp(Team::Party).cmp(&state.team_hp(Team::Enemy)) {
        Ordering::Greater => Some(Team::Party),
        Ordering::Less => Some(Team::Enemy),
        Ordering::Equal => None,
    };

    let kills = extract_kills(&state.events);
    BattleResult {
        winner,
        kills,
        ticks: TURN_LIMIT,
        events: state.events,
    }
}

fn extract_kills(events: &[GameEvent]) -> Vec<(u8, u8)> {
    events
        .iter()
        .filter_map(|e| match e {
            GameEvent::UnitDied { unit, killer } => Some((*killer, *unit)),
            _ => None,
        })
        .collect()
}

// ── Unit Descriptor (for display) ──────────────────────────────

struct UnitDesc {
    class: Class,
    team: Team,
    strategy: &'static str,
}

const UNIT_DESCS: [UnitDesc; 8] = [
    UnitDesc {
        class: Class::Knight,
        team: Team::Party,
        strategy: "Random",
    },
    UnitDesc {
        class: Class::Archer,
        team: Team::Party,
        strategy: "Greedy",
    },
    UnitDesc {
        class: Class::BlackMage,
        team: Team::Party,
        strategy: "Validator",
    },
    UnitDesc {
        class: Class::WhiteMage,
        team: Team::Party,
        strategy: "HL",
    },
    UnitDesc {
        class: Class::Knight,
        team: Team::Enemy,
        strategy: "HL",
    },
    UnitDesc {
        class: Class::Archer,
        team: Team::Enemy,
        strategy: "Validator",
    },
    UnitDesc {
        class: Class::BlackMage,
        team: Team::Enemy,
        strategy: "Greedy",
    },
    UnitDesc {
        class: Class::WhiteMage,
        team: Team::Enemy,
        strategy: "Random",
    },
];

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let mut rng = Rng::with_seed(42);

    println!("╔═══ FFT Tactics Arena ═════════════════════════════════════════╗");
    println!("║  Party: ⚔️Knight-Rnd  🏹Archer-Grd  🔮BMage-Val  ✨WMage-HL  ║");
    println!("║  Enemy: ⚔️Knight-HL   🏹Archer-Val  🔮BMage-Grd  ✨WMage-Rnd  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Persistent players (HL learns across rounds)
    let mut players: Vec<Box<dyn FftPlayer>> = vec![
        Box::new(RandomPlayer),    // 0: Party Knight
        Box::new(GreedyPlayer),    // 1: Party Archer
        Box::new(ValidatorPlayer), // 2: Party BlackMage
        Box::new(HLPlayer::new()), // 3: Party WhiteMage
        Box::new(HLPlayer::new()), // 4: Enemy Knight
        Box::new(ValidatorPlayer), // 5: Enemy Archer
        Box::new(GreedyPlayer),    // 6: Enemy BlackMage
        Box::new(RandomPlayer),    // 7: Enemy WhiteMage
    ];

    let mut party_wins = 0u32;
    let mut enemy_wins = 0u32;
    let mut draws = 0u32;
    let mut total_ticks = 0u32;
    let mut unit_kills = [0u32; 8];
    let mut unit_deaths = [0u32; 8];

    for round in 0..ROUNDS {
        for p in players.iter_mut() {
            p.reset();
        }

        let result = run_battle(&mut players, &mut rng);
        total_ticks += result.ticks;

        match result.winner {
            Some(Team::Party) => party_wins += 1,
            Some(Team::Enemy) => enemy_wins += 1,
            None => draws += 1,
        }

        for (killer, victim) in &result.kills {
            unit_kills[*killer as usize] += 1;
            unit_deaths[*victim as usize] += 1;
        }

        // Update HL players (unit 3: Party WMage, unit 4: Enemy Knight)
        for hl_id in [3u8, 4u8] {
            let survived = !result.kills.iter().any(|(_, v)| *v == hl_id);
            let kills = result.kills.iter().filter(|(k, _)| *k == hl_id).count() as u32;
            let damage: i32 = result
                .events
                .iter()
                .filter_map(|e| match e {
                    GameEvent::DamageDealt {
                        attacker, damage, ..
                    } if *attacker == hl_id => Some(*damage),
                    _ => None,
                })
                .sum();
            let healing: i32 = result
                .events
                .iter()
                .filter_map(|e| match e {
                    GameEvent::Healed { healer, amount, .. } if *healer == hl_id => Some(*amount),
                    _ => None,
                })
                .sum();

            if let Some(hl) = players[hl_id as usize]
                .as_any_mut()
                .downcast_mut::<HLPlayer>()
            {
                hl.update_outcome(survived, kills, damage, healing);
            }
        }

        // Print round
        let winner_str = match result.winner {
            Some(Team::Party) => "Party",
            Some(Team::Enemy) => "Enemy",
            None => "Draw",
        };
        let kill_str = result
            .kills
            .iter()
            .map(|(k, v)| format!("{k}→{v}"))
            .collect::<Vec<_>>()
            .join(" ");

        println!(
            "Round {:>3}: Winner={:<6} Ticks={:>3} Kills=[{kill_str}]",
            round + 1,
            winner_str,
            result.ticks,
        );
    }

    // ── Final Standings ────────────────────────────────────────

    println!();
    println!("═══ Final Standings ({ROUNDS} rounds) ═══");
    println!("  Party: Wins={party_wins}  Losses={enemy_wins}  Draws={draws}");
    println!("  Enemy: Wins={enemy_wins}  Losses={party_wins}  Draws={draws}");
    println!("  Avg Ticks: {:.1}", total_ticks as f64 / ROUNDS as f64);

    // ── Unit Stats ─────────────────────────────────────────────

    println!();
    println!("═══ Unit Stats ═══");
    for (i, desc) in UNIT_DESCS.iter().enumerate() {
        let strat = players[i].name();
        println!(
            "  {} {:<8}-{:<10} Kills={:>3} Deaths={:>3}",
            desc.team,
            desc.class.label(),
            strat,
            unit_kills[i],
            unit_deaths[i],
        );
    }

    // MVP
    let mvp = unit_kills
        .iter()
        .enumerate()
        .max_by_key(|(_, k)| *k)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let desc = &UNIT_DESCS[mvp];
    let mvp_strat = players[mvp].name();
    println!();
    println!(
        "  MVP: {} {}-{} ({} kills, {} deaths)",
        desc.team,
        desc.class.emoji(),
        mvp_strat,
        unit_kills[mvp],
        unit_deaths[mvp],
    );

    // HL Q-values report
    println!();
    println!("═══ HL Q-Values ═══");
    for hl_id in [3u8, 4u8] {
        if let Some(hl) = players[hl_id as usize]
            .as_any_mut()
            .downcast_mut::<HLPlayer>()
        {
            let desc = &UNIT_DESCS[hl_id as usize];
            print!("  {} {}-{}: ", desc.team, desc.class.label(), desc.strategy);
            for action in ActionType::all() {
                let idx = action.as_usize();
                let q = hl.q_values[idx];
                let v = hl.visits[idx];
                print!("{action}={q:+.2}({v}) ");
            }
            println!("epsilon={:.3}", hl.epsilon);
        }
    }

    println!();
}
