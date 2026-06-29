//! AI player trait and implementations for FFT Tactics Arena.
//!
//! Three player strategies:
//! - **GreedyFFTPlayer** — attacks weakest, heals low HP, uses potions in crisis
//! - **ValidatorFFTPlayer** — safety-first, cures debuffs, heals critical allies, retreats
//! - **HLFFTPlayer** — bandit Q-learning over 9 action types, adapts across rounds

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;

use super::battle::BattleState;
use super::status::{self, ActiveEffect, StatusEffect};
use super::types::*;

// ── Trait ───────────────────────────────────────────────────────

pub trait FftPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action;
    fn name(&self) -> &'static str;
    fn reset(&mut self) {}
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Called after each game with per-unit outcome stats.
    /// Learning players override this to update Q-values.
    /// `unit_id` is the global unit index (0–3 party, 4–7 enemy).
    fn on_game_end(
        &mut self,
        _unit_id: u8,
        _survived: bool,
        _kills: u32,
        _damage: i32,
        _healing: i32,
    ) {
    }
}

// ── Helpers ────────────────────────────────────────────────────

pub fn weakest_target(state: &BattleState, targets: &[u8]) -> Option<u8> {
    targets
        .iter()
        .min_by_key(|&&id| state.units[id as usize].hp)
        .copied()
}

pub fn lowest_hp_ally(state: &BattleState, allies: &[u8]) -> Option<u8> {
    targets_min_by(state, allies, |u| u.hp)
}

pub fn most_debuffed_ally(
    _state: &BattleState,
    effects: &[ActiveEffect],
    allies: &[u8],
) -> Option<u8> {
    allies
        .iter()
        .max_by_key(|&&id| {
            effects
                .iter()
                .filter(|e| e.source == id && e.effect.is_debuff())
                .count()
        })
        .copied()
}

fn targets_min_by(state: &BattleState, targets: &[u8], f: fn(&Unit) -> i32) -> Option<u8> {
    targets
        .iter()
        .min_by_key(|&&id| f(&state.units[id as usize]))
        .copied()
}

pub fn nearest_enemy_pos(state: &BattleState, pos: Pos, team: Team) -> Option<Pos> {
    state
        .units
        .iter()
        .filter(|u| u.alive && u.team != team)
        .min_by_key(|u| pos.manhattan(u.pos))
        .map(|u| u.pos)
}

pub fn move_toward(reachable: &[Pos], target: Pos) -> Option<Pos> {
    reachable
        .iter()
        .min_by_key(|p| p.manhattan(target))
        .copied()
}

pub fn move_away(reachable: &[Pos], threat: Pos) -> Option<Pos> {
    reachable
        .iter()
        .max_by_key(|p| p.manhattan(threat))
        .copied()
}

// ── Greedy Player ──────────────────────────────────────────────

/// Aggressive player: attacks weakest enemy, heals low HP, uses potion in crisis.
/// Added awareness of CurePoison for self when poisoned and low HP.
pub struct GreedyFFTPlayer;

impl FftPlayer for GreedyFFTPlayer {
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

        // Cure own poison if low HP
        if hp_pct < 0.5 && unit.can_afford(ActionType::CurePoison) {
            let poisoned = state
                .effects
                .iter()
                .any(|e| e.source == unit_id && e.effect == StatusEffect::Poison);
            if poisoned {
                return Action {
                    action_type: ActionType::CurePoison,
                    target_id: Some(unit_id),
                    move_to,
                };
            }
        }

        // Attack weakest enemy
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        if let Some(target) = weakest_target(state, &enemies) {
            let target_hp = state.units[target as usize].hp_pct();
            if target_hp < 0.3 && unit.can_afford(ActionType::BlackMagic) {
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

/// Safety-first player: cures debuffs, heals critical allies, attacks only when safe.
/// Retreating when low HP. Uses Esuna and CurePoison for allies.
pub struct ValidatorFFTPlayer;

impl FftPlayer for ValidatorFFTPlayer {
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

        // Cure debuffs first (priority: cure poison before heal)
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
        if unit.can_afford(ActionType::CurePoison) {
            for &ally in &allies {
                let ally_hp = state.units[ally as usize].hp_pct();
                let poisoned = state
                    .effects
                    .iter()
                    .any(|e| e.source == ally && e.effect == StatusEffect::Poison);
                if poisoned && ally_hp < 0.5 {
                    return Action {
                        action_type: ActionType::CurePoison,
                        target_id: Some(ally),
                        move_to: None,
                    };
                }
            }
        }
        if unit.can_afford(ActionType::Esuna)
            && let Some(ally) = most_debuffed_ally(state, &state.effects, &allies)
        {
            return Action {
                action_type: ActionType::Esuna,
                target_id: Some(ally),
                move_to: None,
            };
        }

        // Heal critical ally
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

/// Bandit Q-learning player over 9 action types.
/// Explores with epsilon-greedy, updates Q-values from battle outcomes.
/// Aware of CurePoison, Esuna, and Dispel based on battlefield state.
pub struct HLFFTPlayer {
    pub q_values: [f32; 9],
    pub visits: [u32; 9],
    pub total_pulls: u32,
    pub epsilon: f32,
    pub last_action: Option<ActionType>,
}

impl HLFFTPlayer {
    pub fn new() -> Self {
        Self {
            q_values: [0.0; 9],
            visits: [0; 9],
            total_pulls: 0,
            epsilon: 0.15,
            last_action: None,
        }
    }

    pub fn update_outcome(
        &mut self,
        survived: bool,
        kills: u32,
        damage_dealt: i32,
        healing_done: i32,
    ) {
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

impl FftPlayer for HLFFTPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let hp_pct = unit.hp_pct();
        let reachable = state.reachable_positions(unit_id);
        let enemy_team = BattleState::enemy_team(unit.team);

        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);

        // Build available actions based on battlefield state
        let mut available = vec![ActionType::Wait, ActionType::Defend];
        if !enemies.is_empty() {
            available.push(ActionType::Attack);
        }
        if !enemies.is_empty()
            && unit.can_afford(ActionType::BlackMagic)
            && status::can_cast(unit, &state.effects)
        {
            available.push(ActionType::BlackMagic);
        }
        if !allies.is_empty()
            && unit.can_afford(ActionType::WhiteMagic)
            && status::can_cast(unit, &state.effects)
        {
            available.push(ActionType::WhiteMagic);
        }
        if unit.can_afford(ActionType::Potion) && hp_pct < 0.5 {
            available.push(ActionType::Potion);
        }
        if unit.can_afford(ActionType::CurePoison) && status::can_cast(unit, &state.effects) {
            let any_poisoned = allies.iter().any(|&a| {
                state
                    .effects
                    .iter()
                    .any(|e| e.source == a && e.effect == StatusEffect::Poison)
            });
            if any_poisoned {
                available.push(ActionType::CurePoison);
            }
        }
        if unit.can_afford(ActionType::Esuna) && status::can_cast(unit, &state.effects) {
            let any_debuffed = allies.iter().any(|&a| {
                state
                    .effects
                    .iter()
                    .any(|e| e.source == a && e.effect.esuna_curable())
            });
            if any_debuffed {
                available.push(ActionType::Esuna);
            }
        }
        if !enemies.is_empty()
            && unit.can_afford(ActionType::Dispel)
            && status::can_cast(unit, &state.effects)
        {
            let any_buffed = enemies.iter().any(|&e| {
                state
                    .effects
                    .iter()
                    .any(|ef| ef.source == e && ef.effect.is_buff())
            });
            if any_buffed {
                available.push(ActionType::Dispel);
            }
        }

        // Epsilon-greedy selection
        let action_type = if rng.f32() < self.epsilon {
            available[rng.usize(..available.len())]
        } else {
            self.best_available(&available)
        };

        self.last_action = Some(action_type);
        self.visits[action_type.as_usize()] += 1;
        self.total_pulls += 1;

        // Select target based on action type
        let target_id = match action_type {
            ActionType::Attack | ActionType::BlackMagic => weakest_target(state, &enemies),
            ActionType::WhiteMagic => lowest_hp_ally(state, &allies),
            ActionType::CurePoison => allies
                .iter()
                .find(|&&a| {
                    state
                        .effects
                        .iter()
                        .any(|e| e.source == a && e.effect == StatusEffect::Poison)
                })
                .copied(),
            ActionType::Esuna => most_debuffed_ally(state, &state.effects, &allies),
            ActionType::Dispel => enemies
                .iter()
                .find(|&&e| {
                    state
                        .effects
                        .iter()
                        .any(|ef| ef.source == e && ef.effect.is_buff())
                })
                .copied(),
            ActionType::Potion => Some(unit_id),
            _ => None,
        };

        // Movement: move toward target if out of range, else toward nearest enemy
        let move_to = if let Some(tid) = target_id {
            let target_pos = state.units[tid as usize].pos;
            if unit.pos.manhattan(target_pos) <= unit.stats.range {
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

    fn on_game_end(&mut self, _unit_id: u8, survived: bool, kills: u32, damage: i32, healing: i32) {
        self.update_outcome(survived, kills, damage, healing);
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Default for HLFFTPlayer {
    fn default() -> Self {
        Self::new()
    }
}
