//! Status effects for FFT Tactics Arena.
//!
//! 9 status effects: Poison, Regen, Protect, Shell, Haste, Slow, Silence, Blind, Sleep.
//! Each has tick behavior, duration, potency, and dispel rules.

use super::types::{BASE_CT_FILL, BASE_HIT_RATE, GameEvent, Unit};

// ── Status Effect ───────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StatusEffect {
    Poison,
    Regen,
    Protect,
    Shell,
    Haste,
    Slow,
    Silence,
    Blind,
    Sleep,
}

impl StatusEffect {
    pub fn name(self) -> &'static str {
        match self {
            Self::Poison => "Poison",
            Self::Regen => "Regen",
            Self::Protect => "Protect",
            Self::Shell => "Shell",
            Self::Haste => "Haste",
            Self::Slow => "Slow",
            Self::Silence => "Silence",
            Self::Blind => "Blind",
            Self::Sleep => "Sleep",
        }
    }

    pub fn emoji(self) -> &'static str {
        match self {
            Self::Poison => "🟢",
            Self::Regen => "💚",
            Self::Protect => "🛡️",
            Self::Shell => "🔵",
            Self::Haste => "⚡",
            Self::Slow => "🐌",
            Self::Silence => "🔇",
            Self::Blind => "😵",
            Self::Sleep => "💤",
        }
    }

    pub fn is_debuff(self) -> bool {
        matches!(
            self,
            Self::Poison | Self::Slow | Self::Silence | Self::Blind | Self::Sleep
        )
    }

    pub fn is_buff(self) -> bool {
        matches!(
            self,
            Self::Regen | Self::Protect | Self::Shell | Self::Haste
        )
    }

    pub fn is_tickable(self) -> bool {
        matches!(self, Self::Poison | Self::Regen)
    }

    /// Can be cured by Esuna.
    pub fn esuna_curable(self) -> bool {
        matches!(self, Self::Slow | Self::Silence | Self::Blind | Self::Sleep)
    }

    /// Can be removed by Dispel.
    pub fn dispellable(self) -> bool {
        matches!(
            self,
            Self::Protect | Self::Shell | Self::Haste | Self::Regen
        )
    }
}

// ── Active Effect ───────────────────────────────────────────────

/// An active status effect on a unit.
/// `source` = the unit ID carrying this effect (i.e., the target of the buff/debuff).
#[derive(Clone, Debug)]
pub struct ActiveEffect {
    pub effect: StatusEffect,
    pub remaining_ticks: u8,
    pub potency: u8,
    /// The unit ID this effect is applied to (the carrier).
    pub source: u8,
}

impl ActiveEffect {
    pub fn new(effect: StatusEffect, duration: u8, potency: u8, source: u8) -> Self {
        Self {
            effect,
            remaining_ticks: duration,
            potency,
            source,
        }
    }
}

// ── Tick Resolution ─────────────────────────────────────────────

/// Apply all ticking effects for all alive units.
/// Returns events generated from ticks, expirations, and deaths.
pub fn apply_tick_effects(units: &mut [Unit], effects: &mut Vec<ActiveEffect>) -> Vec<GameEvent> {
    let mut events = Vec::new();

    for active in effects.iter_mut() {
        if active.remaining_ticks == 0 {
            continue;
        }

        let target_id = active.source as usize;
        if target_id >= units.len() || !units[target_id].alive {
            active.remaining_ticks = 0;
            continue;
        }

        active.remaining_ticks -= 1;

        match active.effect {
            StatusEffect::Poison => {
                let dmg = active.potency as i32;
                units[target_id].hp -= dmg;
                events.push(GameEvent::EffectTicked {
                    target: active.source,
                    effect: "Poison".to_string(),
                    damage: dmg,
                });
                if units[target_id].hp <= 0 {
                    units[target_id].hp = 0;
                    units[target_id].alive = false;
                    events.push(GameEvent::UnitDied {
                        unit: active.source,
                        killer: active.source, // Poison killed them
                    });
                }
            }
            StatusEffect::Regen => {
                let heal = active.potency as i32;
                let unit = &mut units[target_id];
                let actual = heal.min(unit.stats.max_hp - unit.hp);
                unit.hp += actual;
                events.push(GameEvent::EffectTicked {
                    target: active.source,
                    effect: "Regen".to_string(),
                    damage: -actual, // Negative damage = healing
                });
            }
            StatusEffect::Sleep => {
                // Sleep doesn't tick damage, just counts down.
                // Wake on damage is handled in resolve_action.
            }
            _ => {} // Protect, Shell, Haste, Slow, Silence, Blind: passive, no tick
        }

        if active.remaining_ticks == 0 {
            events.push(GameEvent::EffectExpired {
                target: active.source,
                effect: active.effect.name().to_string(),
            });
        }
    }

    effects.retain(|e| e.remaining_ticks > 0);

    events
}

// ── Stat Modifier Helpers ───────────────────────────────────────

/// Whether a unit can cast magic (not Silenced or Asleep).
pub fn can_cast(unit: &Unit, effects: &[ActiveEffect]) -> bool {
    !effects.iter().any(|e| {
        e.remaining_ticks > 0
            && e.source == unit.id
            && matches!(e.effect, StatusEffect::Silence | StatusEffect::Sleep)
    })
}

/// Whether a unit can act at all (not Asleep).
pub fn can_act(unit: &Unit, effects: &[ActiveEffect]) -> bool {
    !effects
        .iter()
        .any(|e| e.remaining_ticks > 0 && e.source == unit.id && e.effect == StatusEffect::Sleep)
}

/// CT fill rate modified by Haste/Slow.
pub fn ct_fill_rate(unit: &Unit, effects: &[ActiveEffect]) -> f32 {
    let mut rate = unit.stats.ct_speed * BASE_CT_FILL;
    for e in effects {
        if e.remaining_ticks > 0 && e.source == unit.id {
            match e.effect {
                StatusEffect::Haste => rate *= 1.5,
                StatusEffect::Slow => rate *= 0.5,
                _ => {}
            }
        }
    }
    rate
}

/// Effective physical DEF with Protect buff (+50%).
pub fn effective_phys_def(unit: &Unit, effects: &[ActiveEffect]) -> i32 {
    let has_protect = effects
        .iter()
        .any(|e| e.remaining_ticks > 0 && e.source == unit.id && e.effect == StatusEffect::Protect);
    let base = unit.stats.def;
    if has_protect {
        (base as f32 * 1.5) as i32
    } else {
        base
    }
}

/// Effective magic DEF with Shell buff (+50%).
pub fn effective_mag_def(unit: &Unit, effects: &[ActiveEffect]) -> i32 {
    let has_shell = effects
        .iter()
        .any(|e| e.remaining_ticks > 0 && e.source == unit.id && e.effect == StatusEffect::Shell);
    let base = unit.stats.def;
    if has_shell {
        (base as f32 * 1.5) as i32
    } else {
        base
    }
}

/// Hit rate modified by Blind debuff (halved).
pub fn effective_hit_rate(unit: &Unit, effects: &[ActiveEffect]) -> f32 {
    let has_blind = effects
        .iter()
        .any(|e| e.remaining_ticks > 0 && e.source == unit.id && e.effect == StatusEffect::Blind);
    if has_blind {
        BASE_HIT_RATE * 0.5
    } else {
        BASE_HIT_RATE
    }
}
