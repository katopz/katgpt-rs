//! Battle state and ATB resolution for FFT Tactics Arena.
//!
//! Active Time Battle (ATB) model: units act independently when CT gauge fills.
//! Battle loop: advance_ct → tick_effects → ready_units → collect actions → resolve → reset_ct

use super::status::{self, ActiveEffect, StatusEffect};
use super::types::*;

// ── Battle State ───────────────────────────────────────────────

/// `Clone` added in Plan 298 Phase 4 (FFT EdgeLoRA follow-up) so the FFT
/// self-play adapter can snapshot state for encode-before/encode-after episode
/// emission. All fields are `Clone` (`Vec<Unit>`, `Vec<GameEvent>`,
/// `Vec<ActiveEffect>`, `u32`), so this is semantically free.
#[derive(Clone)]
pub struct BattleState {
    pub units: Vec<Unit>,
    pub events: Vec<GameEvent>,
    pub effects: Vec<ActiveEffect>,
    pub tick: u32,
}

impl BattleState {
    /// Create default 4v4 battle with standard classes.
    pub fn new() -> Self {
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
            effects: Vec::new(),
            tick: 0,
        }
    }

    /// Create with configurable party composition.
    pub fn new_with_config(party: &[Class], enemy: &[Class]) -> Self {
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
        let mut units = Vec::with_capacity(party.len() + enemy.len());
        for (i, (&class, &pos)) in party.iter().zip(&party_pos).enumerate() {
            units.push(Unit::new(i as u8, class, Team::Party, pos));
        }
        for (i, (&class, &pos)) in enemy.iter().zip(&enemy_pos).enumerate() {
            units.push(Unit::new((i + party.len()) as u8, class, Team::Enemy, pos));
        }
        Self {
            units,
            events: Vec::new(),
            effects: Vec::new(),
            tick: 0,
        }
    }

    /// Create random 4v4 composition.
    pub fn new_random_8(rng: &mut fastrand::Rng) -> Self {
        let classes = Class::all();
        let party: Vec<Class> = (0..4)
            .map(|_| classes[rng.usize(..classes.len())])
            .collect();
        let enemy: Vec<Class> = (0..4)
            .map(|_| classes[rng.usize(..classes.len())])
            .collect();
        Self::new_with_config(&party, &enemy)
    }

    /// Create random NvM composition.
    pub fn new_random_n(rng: &mut fastrand::Rng, party_size: usize, enemy_size: usize) -> Self {
        let classes = Class::all();
        let party: Vec<Class> = (0..party_size)
            .map(|_| classes[rng.usize(..classes.len())])
            .collect();
        let enemy: Vec<Class> = (0..enemy_size)
            .map(|_| classes[rng.usize(..classes.len())])
            .collect();
        Self::new_with_config(&party, &enemy)
    }

    pub fn unit_at(&self, pos: Pos) -> Option<u8> {
        self.units
            .iter()
            .find(|u| u.alive && u.pos == pos)
            .map(|u| u.id)
    }

    pub fn reachable_positions(&self, unit_id: u8) -> Vec<Pos> {
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

    pub fn targets_in_range(&self, pos: Pos, range: i32, target_team: Team) -> Vec<u8> {
        self.units
            .iter()
            .filter(|u| u.alive && u.team == target_team && u.pos.manhattan(pos) <= range)
            .map(|u| u.id)
            .collect()
    }

    pub fn check_winner(&self) -> Option<Team> {
        let party_alive = self.units.iter().any(|u| u.alive && u.team == Team::Party);
        let enemy_alive = self.units.iter().any(|u| u.alive && u.team == Team::Enemy);
        match (party_alive, enemy_alive) {
            (true, false) => Some(Team::Party),
            (false, true) => Some(Team::Enemy),
            (false, false) => Some(Team::Party),
            _ => None,
        }
    }

    pub fn enemy_team(team: Team) -> Team {
        match team {
            Team::Party => Team::Enemy,
            Team::Enemy => Team::Party,
        }
    }

    pub fn team_hp(&self, team: Team) -> i32 {
        self.units
            .iter()
            .filter(|u| u.alive && u.team == team)
            .map(|u| u.hp)
            .sum()
    }

    // ── ATB Methods ────────────────────────────────────────────

    /// Advance CT gauges for all alive units.
    pub fn advance_ct(&mut self) {
        for unit in &mut self.units {
            if !unit.alive {
                continue;
            }
            let rate = status::ct_fill_rate(unit, &self.effects);
            unit.ct_gauge += rate;
        }
    }

    /// Get all alive units with CT gauge >= threshold (ready to act).
    pub fn ready_units(&self) -> Vec<u8> {
        self.units
            .iter()
            .filter(|u| u.alive && u.ct_gauge >= CT_THRESHOLD)
            .map(|u| u.id)
            .collect()
    }

    /// Reset CT gauge for units that acted.
    pub fn reset_ct(&mut self, unit_ids: &[u8]) {
        for &id in unit_ids {
            if (id as usize) < self.units.len() {
                self.units[id as usize].ct_gauge = 0.0;
            }
        }
    }

    /// Apply all ticking effects (poison, regen, duration countdown).
    pub fn tick_effects(&mut self) {
        let events = status::apply_tick_effects(&mut self.units, &mut self.effects);
        self.events.extend(events);
    }
}

impl Default for BattleState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Action Resolution ──────────────────────────────────────────

pub fn resolve_action(
    state: &mut BattleState,
    unit_id: u8,
    action: &Action,
    rng: &mut fastrand::Rng,
) {
    // Move first
    if let Some(to) = action.move_to {
        state.units[unit_id as usize].pos = to;
    }

    let pos = state.units[unit_id as usize].pos;
    let stats = state.units[unit_id as usize].stats;

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
            let def = status::effective_phys_def(&state.units[target_id as usize], &state.effects);
            let defending = state.units[target_id as usize].defending;
            let raw = (atk as f32 * 1.5 - def as f32 * 0.3).max(1.0) as i32;
            let damage = if defending {
                (raw as f32 * 0.5) as i32
            } else {
                raw
            };

            let hit_rate =
                status::effective_hit_rate(&state.units[unit_id as usize], &state.effects);
            if rng.f32() < hit_rate {
                wake_on_damage(state, target_id);
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
            if !status::can_cast(&state.units[unit_id as usize], &state.effects) {
                return;
            }

            state.units[unit_id as usize].spend(ActionType::BlackMagic);
            let mag = stats.mag;
            let def = status::effective_mag_def(&state.units[target_id as usize], &state.effects);
            let defending = state.units[target_id as usize].defending;
            let raw = (mag as f32 * 1.8 - def as f32 * 0.2).max(1.0) as i32;
            let damage = if defending {
                (raw as f32 * 0.5) as i32
            } else {
                raw
            };

            if rng.f32() < MAGIC_HIT_RATE {
                wake_on_damage(state, target_id);
                state.units[target_id as usize].hp -= damage;
                state.events.push(GameEvent::DamageDealt {
                    attacker: unit_id,
                    target: target_id,
                    damage,
                });

                // Poison chance on hit
                if rng.f32() < POISON_CHANCE {
                    let potency = (mag / 3).clamp(5, 15) as u8;
                    let duration = rng.u8(3..=5);
                    state.effects.push(ActiveEffect::new(
                        StatusEffect::Poison,
                        duration,
                        potency,
                        target_id,
                    ));
                    state.events.push(GameEvent::EffectApplied {
                        target: target_id,
                        effect: "Poison".to_string(),
                        duration,
                    });
                }

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
            if !status::can_cast(&state.units[unit_id as usize], &state.effects) {
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
        ActionType::CurePoison => {
            let Some(&target_id) = action.target_id.as_ref() else {
                return;
            };
            if !status::can_cast(&state.units[unit_id as usize], &state.effects) {
                return;
            }

            state.units[unit_id as usize].spend(ActionType::CurePoison);
            let before = state.effects.len();
            state
                .effects
                .retain(|e| !(e.source == target_id && e.effect == StatusEffect::Poison));
            if state.effects.len() < before {
                state.events.push(GameEvent::DebuffCured {
                    healer: unit_id,
                    target: target_id,
                    effect: "Poison".to_string(),
                });
            }
        }
        ActionType::Esuna => {
            let Some(&target_id) = action.target_id.as_ref() else {
                return;
            };
            if !status::can_cast(&state.units[unit_id as usize], &state.effects) {
                return;
            }

            state.units[unit_id as usize].spend(ActionType::Esuna);
            if let Some(idx) = state
                .effects
                .iter()
                .position(|e| e.source == target_id && e.effect.esuna_curable())
            {
                let effect_name = state.effects[idx].effect.name().to_string();
                state.effects.remove(idx);
                state.events.push(GameEvent::DebuffCured {
                    healer: unit_id,
                    target: target_id,
                    effect: effect_name,
                });
            }
        }
        ActionType::Dispel => {
            let Some(&target_id) = action.target_id.as_ref() else {
                return;
            };
            if !status::can_cast(&state.units[unit_id as usize], &state.effects) {
                return;
            }

            state.units[unit_id as usize].spend(ActionType::Dispel);
            if let Some(idx) = state
                .effects
                .iter()
                .position(|e| e.source == target_id && e.effect.dispellable())
            {
                let effect_name = state.effects[idx].effect.name().to_string();
                state.effects.remove(idx);
                state.events.push(GameEvent::BuffDispelled {
                    caster: unit_id,
                    target: target_id,
                    effect: effect_name,
                });
            }
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
            state.events.push(GameEvent::Healed {
                healer: unit_id,
                target: target_id,
                amount: actual,
            });
        }
        ActionType::Wait => {}
    }
}

/// Check if a unit died and emit event.
fn check_death(state: &mut BattleState, target_id: u8, killer_id: u8) {
    if state.units[target_id as usize].hp <= 0 {
        state.units[target_id as usize].hp = 0;
        state.units[target_id as usize].alive = false;
        // Remove all effects on dead unit
        state.effects.retain(|e| e.source != target_id);
        state.events.push(GameEvent::UnitDied {
            unit: target_id,
            killer: killer_id,
        });
    }
}

/// Wake sleeping units when they take damage.
fn wake_on_damage(state: &mut BattleState, target_id: u8) {
    let before = state.effects.len();
    state
        .effects
        .retain(|e| !(e.source == target_id && e.effect == StatusEffect::Sleep));
    if state.effects.len() < before {
        state.events.push(GameEvent::EffectExpired {
            target: target_id,
            effect: "Sleep".to_string(),
        });
    }
}

// ── TFT Provocation Detection (Plan 055) ──────────────────────

/// TFT forgiveness check — Generous TFT randomly forgives provocation.
#[cfg(feature = "g_zero")]
pub fn should_forgive(provoke_level: &ProvokeLevel, rng: &mut fastrand::Rng) -> bool {
    let chance = match provoke_level {
        ProvokeLevel::None => 1.0,          // Nothing to forgive
        ProvokeLevel::Personal(_) => 0.10,  // 10% forgive
        ProvokeLevel::Team(_) => 0.10,      // 10% forgive
        ProvokeLevel::Escalated(_) => 0.05, // 5% forgive (kills are harder to forgive)
    };
    rng.f32() < chance
}
