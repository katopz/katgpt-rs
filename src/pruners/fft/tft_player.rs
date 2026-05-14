//! TFT Party AI Player — Tit-for-Tat provocation with role-based action selection.
//!
//! Implements a Generous TFT strategy where:
//! - Units start in Nice mode (role-based cooperative behavior)
//! - Provocation from enemy attacks triggers Retaliatory mode
//! - Retaliation targets the provoker for FORGIVE_DURATION ticks
//! - Generous TFT randomly forgives provocations based on generous_chance
//! - Safety filter overrides for critical healing/potion
//! - ε-greedy exploration for learning diversity
//!
//! # Feature Gate
//!
//! Entire module gated by `#[cfg(feature = "g_zero")]` in `mod.rs`.

use std::any::Any;

use fastrand::Rng;

use super::*;

// ── Constants ──────────────────────────────────────────────────

const FORGIVE_DURATION: u8 = 5;
const TFT_EPSILON: f32 = 0.05;

// ── TftFFTPlayer ──────────────────────────────────────────────

/// Tit-for-Tat FFT player with role-based action selection.
///
/// Starts in Nice mode with class-specific cooperative behavior.
/// Detects provocation from battle events and switches to Retaliatory
/// mode targeting the provoker. Uses Generous TFT to randomly forgive.
pub struct TftFFTPlayer {
    /// Shared team TFT state (cloned per player for ownership).
    party_state: PartyTftState,
    /// Individual TFT state (mode, last attacker).
    unit_state: UnitTftState,
    /// Player identifier.
    #[allow(dead_code)]
    id: u8,
    /// ε-greedy exploration rate.
    epsilon: f32,
}

impl TftFFTPlayer {
    /// Create a new TFT player with given ID and class.
    pub fn new(id: u8, class: Class) -> Self {
        Self {
            party_state: PartyTftState::new(),
            unit_state: UnitTftState::new(class),
            id,
            epsilon: TFT_EPSILON,
        }
    }

    /// Get current TFT mode.
    pub fn mode(&self) -> &TftMode {
        &self.unit_state.mode
    }

    // ── Provocation Detection ──────────────────────────────────

    /// Detect provocation from recent battle events.
    ///
    /// Scans events for:
    /// - `DamageDealt` where target is ally and attacker is enemy
    /// - `UnitDied` where dead unit is ally
    /// - `EffectApplied` with "Poison" on ally
    ///
    /// Applies Generous TFT: randomly forgives based on `generous_chance`.
    /// Returns the highest provocation level found.
    fn detect_provocation(
        &self,
        unit_id: u8,
        state: &BattleState,
        rng: &mut Rng,
    ) -> Option<ProvokeLevel> {
        let unit = &state.units[unit_id as usize];
        let my_team = unit.team;
        let mut best: Option<ProvokeLevel> = None;

        for event in &state.events {
            let candidate = match event {
                GameEvent::DamageDealt {
                    attacker, target, ..
                } => {
                    let attacker_unit = &state.units[*attacker as usize];
                    let target_unit = &state.units[*target as usize];
                    if target_unit.team == my_team && attacker_unit.team != my_team {
                        if *target == unit_id {
                            Some(ProvokeLevel::Personal(*attacker))
                        } else {
                            Some(ProvokeLevel::Team(*attacker))
                        }
                    } else {
                        None
                    }
                }
                GameEvent::UnitDied { unit: dead, killer } => {
                    let dead_unit = &state.units[*dead as usize];
                    if dead_unit.team == my_team {
                        Some(ProvokeLevel::Escalated(*killer))
                    } else {
                        None
                    }
                }
                GameEvent::EffectApplied { target, effect, .. } => {
                    let target_unit = &state.units[*target as usize];
                    if target_unit.team == my_team && effect == "Poison" {
                        Some(ProvokeLevel::Team(0))
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(level) = candidate {
                // Generous TFT: randomly forgive this provocation
                if rng.f32() < self.party_state.generous_chance {
                    continue;
                }
                // Keep highest provocation level
                let best_priority =
                    Self::provoke_level_priority(best.as_ref().unwrap_or(&ProvokeLevel::None));
                if Self::provoke_level_priority(&level) > best_priority {
                    best = Some(level);
                }
            }
        }

        best
    }

    /// Extract target ID from provocation level.
    fn provoke_target(level: &ProvokeLevel) -> u8 {
        match level {
            ProvokeLevel::None => 0,
            ProvokeLevel::Personal(t) | ProvokeLevel::Team(t) | ProvokeLevel::Escalated(t) => *t,
        }
    }

    /// Priority ordering for provocation levels (higher = more severe).
    fn provoke_level_priority(level: &ProvokeLevel) -> u8 {
        match level {
            ProvokeLevel::None => 0,
            ProvokeLevel::Team(_) => 1,
            ProvokeLevel::Personal(_) => 2,
            ProvokeLevel::Escalated(_) => 3,
        }
    }

    // ── Role Actions ───────────────────────────────────────────

    /// Select action in Nice mode (cooperative, class-based).
    fn role_nice_action(&self, unit_id: u8, state: &BattleState, _rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let reachable = state.reachable_positions(unit_id);
        let enemy_team = BattleState::enemy_team(unit.team);

        match unit.class {
            Class::Knight => {
                // Defend, move toward nearest ally (protect)
                let ally_pos = state
                    .units
                    .iter()
                    .filter(|u| u.alive && u.team == unit.team && u.id != unit_id)
                    .min_by_key(|u| unit.pos.manhattan(u.pos))
                    .map(|u| u.pos);
                let move_to = ally_pos.and_then(|ap| move_toward(&reachable, ap));
                Action {
                    action_type: ActionType::Defend,
                    target_id: None,
                    move_to,
                }
            }
            Class::Archer => {
                // Attack weakest enemy in range, else move toward nearest enemy
                let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
                match weakest_target(state, &enemies) {
                    Some(target) => Action {
                        action_type: ActionType::Attack,
                        target_id: Some(target),
                        move_to: None,
                    },
                    None => {
                        let move_to = nearest_enemy_pos(state, unit.pos, unit.team)
                            .and_then(|ep| move_toward(&reachable, ep));
                        Action {
                            action_type: ActionType::Defend,
                            target_id: None,
                            move_to,
                        }
                    }
                }
            }
            Class::BlackMage => {
                // Attack weakest enemy if MP > 50%, else Defend
                let mp_pct = unit.mp as f32 / unit.stats.max_mp as f32;
                if mp_pct > 0.5 {
                    let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
                    if let Some(target) = weakest_target(state, &enemies) {
                        if unit.can_afford(ActionType::BlackMagic) && can_cast(unit, &state.effects)
                        {
                            return Action {
                                action_type: ActionType::BlackMagic,
                                target_id: Some(target),
                                move_to: None,
                            };
                        }
                        return Action {
                            action_type: ActionType::Attack,
                            target_id: Some(target),
                            move_to: None,
                        };
                    }
                }
                Action {
                    action_type: ActionType::Defend,
                    target_id: None,
                    move_to: None,
                }
            }
            Class::WhiteMage => {
                // Heal lowest HP ally if HP < 70%, CureDebuff if poisoned, else Defend
                let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);

                if unit.can_afford(ActionType::WhiteMagic)
                    && can_cast(unit, &state.effects)
                    && let Some(ally) = lowest_hp_ally(state, &allies)
                    && state.units[ally as usize].hp_pct() < 0.7
                {
                    return Action {
                        action_type: ActionType::WhiteMagic,
                        target_id: Some(ally),
                        move_to: None,
                    };
                }

                if unit.can_afford(ActionType::CurePoison) && can_cast(unit, &state.effects) {
                    for &ally in &allies {
                        let poisoned = state
                            .effects
                            .iter()
                            .any(|e| e.source == ally && e.effect == StatusEffect::Poison);
                        if poisoned {
                            return Action {
                                action_type: ActionType::CurePoison,
                                target_id: Some(ally),
                                move_to: None,
                            };
                        }
                    }
                }

                Action {
                    action_type: ActionType::Defend,
                    target_id: None,
                    move_to: None,
                }
            }
            Class::Monk => {
                // Attack nearest enemy in range, else move toward nearest enemy
                let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
                let nearest = enemies
                    .iter()
                    .min_by_key(|&&id| unit.pos.manhattan(state.units[id as usize].pos))
                    .copied();
                match nearest {
                    Some(target) => Action {
                        action_type: ActionType::Attack,
                        target_id: Some(target),
                        move_to: None,
                    },
                    None => {
                        let move_to = nearest_enemy_pos(state, unit.pos, unit.team)
                            .and_then(|ep| move_toward(&reachable, ep));
                        Action {
                            action_type: ActionType::Defend,
                            target_id: None,
                            move_to,
                        }
                    }
                }
            }
            Class::TimeMage => {
                // Defend (no Haste/Slow in current action set)
                Action {
                    action_type: ActionType::Defend,
                    target_id: None,
                    move_to: None,
                }
            }
        }
    }

    /// Select action in Retaliatory mode (aggressive, targeting provoker).
    fn role_retaliate_action(
        &self,
        unit_id: u8,
        target: u8,
        state: &BattleState,
        _rng: &mut Rng,
    ) -> Action {
        let unit = &state.units[unit_id as usize];
        let reachable = state.reachable_positions(unit_id);
        let target_pos = state.units[target as usize].pos;
        let in_range = unit.pos.manhattan(target_pos) <= unit.stats.range;

        match unit.class {
            Class::Knight => {
                // Move toward target, Attack if in range
                let move_to = move_toward_if_out_of_range(&reachable, target_pos, in_range);
                Action {
                    action_type: ActionType::Attack,
                    target_id: Some(target),
                    move_to,
                }
            }
            Class::Archer => {
                // Move toward target range, Attack target
                let move_to = move_toward_if_out_of_range(&reachable, target_pos, in_range);
                Action {
                    action_type: ActionType::Attack,
                    target_id: Some(target),
                    move_to,
                }
            }
            Class::BlackMage => {
                // BlackMagic on target if in range and MP available, else move toward
                if in_range
                    && unit.can_afford(ActionType::BlackMagic)
                    && can_cast(unit, &state.effects)
                {
                    Action {
                        action_type: ActionType::BlackMagic,
                        target_id: Some(target),
                        move_to: None,
                    }
                } else {
                    let move_to = move_toward(&reachable, target_pos);
                    Action {
                        action_type: ActionType::Defend,
                        target_id: None,
                        move_to,
                    }
                }
            }
            Class::WhiteMage => {
                // Heal wounded ally FIRST (if any ally HP < 40%), THEN help focus target
                let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
                if unit.can_afford(ActionType::WhiteMagic) && can_cast(unit, &state.effects) {
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
                // Help focus target with Attack
                let move_to = move_toward_if_out_of_range(&reachable, target_pos, in_range);
                Action {
                    action_type: ActionType::Attack,
                    target_id: Some(target),
                    move_to,
                }
            }
            Class::Monk => {
                // Move toward target, Attack if adjacent
                let move_to = move_toward_if_out_of_range(&reachable, target_pos, in_range);
                Action {
                    action_type: ActionType::Attack,
                    target_id: Some(target),
                    move_to,
                }
            }
            Class::TimeMage => {
                // Attack target if in range, else Defend
                if in_range {
                    Action {
                        action_type: ActionType::Attack,
                        target_id: Some(target),
                        move_to: None,
                    }
                } else {
                    Action {
                        action_type: ActionType::Defend,
                        target_id: None,
                        move_to: None,
                    }
                }
            }
        }
    }

    // ── Safety Filter ──────────────────────────────────────────

    /// Apply safety filter: override action for critical healing/potion.
    fn safety_filter(&self, action: Action, unit_id: u8, state: &BattleState) -> Action {
        let unit = &state.units[unit_id as usize];

        // Critical HP: use potion
        if unit.hp_pct() < 0.25 && unit.can_afford(ActionType::Potion) {
            return Action {
                action_type: ActionType::Potion,
                target_id: Some(unit_id),
                move_to: action.move_to,
            };
        }

        // Poisoned and low HP: cure self
        if unit.hp_pct() < 0.4
            && unit.can_afford(ActionType::CurePoison)
            && can_cast(unit, &state.effects)
        {
            let poisoned = state
                .effects
                .iter()
                .any(|e| e.source == unit_id && e.effect == StatusEffect::Poison);
            if poisoned {
                return Action {
                    action_type: ActionType::CurePoison,
                    target_id: Some(unit_id),
                    move_to: action.move_to,
                };
            }
        }

        // Critical ally: heal if possible
        if unit.can_afford(ActionType::WhiteMagic) && can_cast(unit, &state.effects) {
            let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
            for &ally in &allies {
                if state.units[ally as usize].hp_pct() < 0.3 {
                    return Action {
                        action_type: ActionType::WhiteMagic,
                        target_id: Some(ally),
                        move_to: action.move_to,
                    };
                }
            }
        }

        action
    }

    // ── ε-greedy Random Action ─────────────────────────────────

    /// Pick a random valid action for exploration.
    fn random_valid_action(&self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let enemy_team = BattleState::enemy_team(unit.team);
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);

        let mut available = vec![ActionType::Wait, ActionType::Defend];

        if !enemies.is_empty() {
            available.push(ActionType::Attack);
        }
        if !enemies.is_empty()
            && unit.can_afford(ActionType::BlackMagic)
            && can_cast(unit, &state.effects)
        {
            available.push(ActionType::BlackMagic);
        }
        if !allies.is_empty()
            && unit.can_afford(ActionType::WhiteMagic)
            && can_cast(unit, &state.effects)
        {
            available.push(ActionType::WhiteMagic);
        }
        if unit.can_afford(ActionType::Potion) {
            available.push(ActionType::Potion);
        }
        if unit.can_afford(ActionType::CurePoison) && can_cast(unit, &state.effects) {
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

        let action_type = available[rng.usize(..available.len())];

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
            ActionType::Potion => Some(unit_id),
            _ => None,
        };

        let reachable = state.reachable_positions(unit_id);
        let move_to = if let Some(tid) = target_id {
            let tp = state.units[tid as usize].pos;
            if unit.pos.manhattan(tp) <= unit.stats.range {
                None
            } else {
                move_toward(&reachable, tp)
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
}

// ── Helper ─────────────────────────────────────────────────────

/// Returns move_toward position if not in range, else None.
fn move_toward_if_out_of_range(reachable: &[Pos], target: Pos, in_range: bool) -> Option<Pos> {
    if in_range {
        None
    } else {
        move_toward(reachable, target)
    }
}

// ── FftPlayer Trait Implementation ────────────────────────────

impl FftPlayer for TftFFTPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        // 1. Detect provocation from events
        let provocation = self.detect_provocation(unit_id, state, rng);

        // 2. Update mode based on provocation
        if let Some(prov) = &provocation {
            let target = Self::provoke_target(prov);
            self.unit_state.mode = TftMode::Retaliatory {
                target,
                ticks_left: FORGIVE_DURATION,
            };
            self.unit_state.last_attacker = Some(target);
        }

        // 3-4. Handle timer decrement, expiry, and provoker death
        self.unit_state.mode = match self.unit_state.mode {
            TftMode::Nice => TftMode::Nice,
            TftMode::Retaliatory { target, ticks_left } => {
                if !state.units[target as usize].alive {
                    TftMode::Nice
                } else {
                    let new_ticks = ticks_left.saturating_sub(1);
                    if new_ticks == 0 {
                        TftMode::Nice
                    } else {
                        TftMode::Retaliatory {
                            target,
                            ticks_left: new_ticks,
                        }
                    }
                }
            }
        };

        // 5. Get action based on mode
        let action = match self.unit_state.mode {
            TftMode::Nice => self.role_nice_action(unit_id, state, rng),
            TftMode::Retaliatory { target, .. } => {
                self.role_retaliate_action(unit_id, target, state, rng)
            }
        };

        // 6. Safety filter
        let action = self.safety_filter(action, unit_id, state);

        // 7. ε-greedy exploration
        if rng.f32() < self.epsilon {
            self.random_valid_action(unit_id, state, rng)
        } else {
            action
        }
    }

    fn name(&self) -> &'static str {
        "TFT"
    }

    fn reset(&mut self) {
        self.party_state.reset();
        self.unit_state.reset();
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────

    fn make_player(id: u8, class: Class) -> TftFFTPlayer {
        let mut p = TftFFTPlayer::new(id, class);
        p.party_state.generous_chance = 0.0; // Never forgive in tests by default
        p.epsilon = 0.0; // No exploration in tests by default
        p
    }

    fn make_battle() -> BattleState {
        BattleState::new_with_config(
            &[Class::Knight, Class::Archer, Class::WhiteMage, Class::Monk],
            &[
                Class::Knight,
                Class::Archer,
                Class::BlackMage,
                Class::TimeMage,
            ],
        )
    }

    // ── Test 1: Initial State ──────────────────────────────────

    #[test]
    fn test_new_player_initial_state() {
        let player = TftFFTPlayer::new(0, Class::Knight);
        assert_eq!(player.id, 0);
        assert!(matches!(player.unit_state.mode, TftMode::Nice));
        assert!((player.epsilon - TFT_EPSILON).abs() < f32::EPSILON);
        assert!((player.party_state.generous_chance - 0.10).abs() < f32::EPSILON);
        assert_eq!(player.unit_state.class, Class::Knight);
        assert_eq!(player.name(), "TFT");
    }

    // ── Test 2: Nice Mode Knight Defends ───────────────────────

    #[test]
    fn test_nice_mode_knight_defends() {
        let mut player = make_player(0, Class::Knight);
        let state = make_battle();
        let mut rng = Rng::with_seed(42);

        let action = player.select_action(0, &state, &mut rng);
        assert_eq!(action.action_type, ActionType::Defend);
    }

    // ── Test 3: Nice Mode Archer Attacks In Range ──────────────

    #[test]
    fn test_nice_mode_archer_attacks_in_range() {
        let mut player = make_player(0, Class::Archer);
        let mut state = BattleState::new_with_config(&[Class::Archer], &[Class::Knight]);
        // Move enemy into archer range (4): (1,1) → (3,1) = distance 2
        state.units[1].pos = Pos::new(3, 1);

        let mut rng = Rng::with_seed(42);
        let action = player.select_action(0, &state, &mut rng);

        assert_eq!(action.action_type, ActionType::Attack);
        assert_eq!(action.target_id, Some(1));
    }

    // ── Test 4: Provocation Detection DamageDealt ──────────────

    #[test]
    fn test_provocation_detection_damage_dealt() {
        let player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.events.push(GameEvent::DamageDealt {
            attacker: 4,
            target: 0,
            damage: 20,
        });

        let mut rng = Rng::with_seed(42);
        let prov = player.detect_provocation(0, &state, &mut rng);

        assert!(matches!(prov, Some(ProvokeLevel::Personal(4))));
    }

    // ── Test 5: Escalation from UnitDied ───────────────────────

    #[test]
    fn test_provocation_escalation_unit_died() {
        let player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state
            .events
            .push(GameEvent::UnitDied { unit: 1, killer: 5 });

        let mut rng = Rng::with_seed(42);
        let prov = player.detect_provocation(0, &state, &mut rng);

        assert!(matches!(prov, Some(ProvokeLevel::Escalated(5))));
    }

    // ── Test 6: Mode Transition Nice → Retaliatory ─────────────

    #[test]
    fn test_mode_transition_nice_to_retaliatory() {
        let mut player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.events.push(GameEvent::DamageDealt {
            attacker: 4,
            target: 0,
            damage: 20,
        });

        let mut rng = Rng::with_seed(42);
        let _action = player.select_action(0, &state, &mut rng);

        match player.unit_state.mode {
            TftMode::Retaliatory { target, ticks_left } => {
                assert_eq!(target, 4);
                assert_eq!(ticks_left, FORGIVE_DURATION - 1);
            }
            ref mode => panic!("Expected Retaliatory, got {mode:?}"),
        }
    }

    // ── Test 7: Mode Transition Retaliatory → Nice (Timer) ─────

    #[test]
    fn test_mode_transition_retaliatory_to_nice_timer() {
        let mut player = make_player(0, Class::Knight);
        player.unit_state.mode = TftMode::Retaliatory {
            target: 4,
            ticks_left: 1,
        };

        let state = make_battle();
        let mut rng = Rng::with_seed(42);
        let _action = player.select_action(0, &state, &mut rng);

        assert!(matches!(player.unit_state.mode, TftMode::Nice));
    }

    // ── Test 8: Provoker Death → Nice ──────────────────────────

    #[test]
    fn test_provoker_death_returns_to_nice() {
        let mut player = make_player(0, Class::Knight);
        player.unit_state.mode = TftMode::Retaliatory {
            target: 4,
            ticks_left: 3,
        };

        let mut state = make_battle();
        state.units[4].alive = false;

        let mut rng = Rng::with_seed(42);
        let _action = player.select_action(0, &state, &mut rng);

        assert!(matches!(player.unit_state.mode, TftMode::Nice));
    }

    // ── Test 9: Generous Forgiveness ───────────────────────────

    #[test]
    fn test_generous_forgiveness() {
        let mut player = TftFFTPlayer::new(0, Class::Knight);
        player.party_state.generous_chance = 1.0; // Always forgive
        player.epsilon = 0.0;

        let mut state = make_battle();
        state.events.push(GameEvent::DamageDealt {
            attacker: 4,
            target: 0,
            damage: 20,
        });

        let mut rng = Rng::with_seed(42);
        let _action = player.select_action(0, &state, &mut rng);

        assert!(matches!(player.unit_state.mode, TftMode::Nice));
    }

    // ── Test 10: Safety Filter Potion ──────────────────────────

    #[test]
    fn test_safety_filter_potion() {
        let mut player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.units[0].hp = 10; // ~8% HP, below 25%

        let mut rng = Rng::with_seed(42);
        let action = player.select_action(0, &state, &mut rng);

        assert_eq!(action.action_type, ActionType::Potion);
        assert_eq!(action.target_id, Some(0));
    }

    // ── Test 11: Safety Filter Cure Poison ─────────────────────

    #[test]
    fn test_safety_filter_cure_poison() {
        let mut player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.units[0].hp = 30; // 25% HP, below 40% threshold
        state.units[0].mp = 20;
        state
            .effects
            .push(ActiveEffect::new(StatusEffect::Poison, 5, 5, 0));

        let mut rng = Rng::with_seed(42);
        let action = player.select_action(0, &state, &mut rng);

        assert_eq!(action.action_type, ActionType::CurePoison);
        assert_eq!(action.target_id, Some(0));
    }

    // ── Test 12: White Mage Heal Priority Retaliatory ──────────

    #[test]
    fn test_white_mage_heal_priority_retaliatory() {
        let mut player = make_player(2, Class::WhiteMage);
        player.unit_state.mode = TftMode::Retaliatory {
            target: 4,
            ticks_left: 5,
        };

        let mut state = make_battle();
        state.units[0].hp = 10; // Knight ally ~8% HP (critical)
        state.units[2].mp = 70;

        let mut rng = Rng::with_seed(42);
        let action = player.select_action(2, &state, &mut rng);

        assert_eq!(action.action_type, ActionType::WhiteMagic);
        assert_eq!(action.target_id, Some(0));
    }

    // ── Test 13: Nice Mode TimeMage Defends ────────────────────

    #[test]
    fn test_nice_mode_time_mage_defends() {
        let mut player = make_player(0, Class::TimeMage);
        let mut state = make_battle();
        state.units[0].class = Class::TimeMage;
        state.units[0].stats = Class::TimeMage.stats();

        let mut rng = Rng::with_seed(42);
        let action = player.select_action(0, &state, &mut rng);

        assert_eq!(action.action_type, ActionType::Defend);
    }

    // ── Test 14: Team Provocation ──────────────────────────────

    #[test]
    fn test_team_provocation() {
        let player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.events.push(GameEvent::DamageDealt {
            attacker: 5,
            target: 1, // ally, not us
            damage: 15,
        });

        let mut rng = Rng::with_seed(42);
        let prov = player.detect_provocation(0, &state, &mut rng);

        assert!(matches!(prov, Some(ProvokeLevel::Team(5))));
    }

    // ── Test 15: Reset Clears Mode ─────────────────────────────

    #[test]
    fn test_reset_clears_mode() {
        let mut player = make_player(0, Class::Knight);
        player.unit_state.mode = TftMode::Retaliatory {
            target: 4,
            ticks_left: 3,
        };
        player.unit_state.last_attacker = Some(4);

        player.reset();

        assert!(matches!(player.unit_state.mode, TftMode::Nice));
        assert!(player.unit_state.last_attacker.is_none());
        assert!(player.party_state.provoked_by.is_none());
    }

    // ── Test 16: Poison Effect Provocation ─────────────────────

    #[test]
    fn test_poison_effect_provocation() {
        let player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.events.push(GameEvent::EffectApplied {
            target: 1,
            effect: "Poison".to_string(),
            duration: 3,
        });

        let mut rng = Rng::with_seed(42);
        let prov = player.detect_provocation(0, &state, &mut rng);

        assert!(matches!(prov, Some(ProvokeLevel::Team(0))));
    }

    // ── Test 17: No Provocation from Own Team ──────────────────

    #[test]
    fn test_no_provocation_from_own_team() {
        let player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.events.push(GameEvent::DamageDealt {
            attacker: 1, // ally
            target: 2,   // ally
            damage: 10,
        });

        let mut rng = Rng::with_seed(42);
        let prov = player.detect_provocation(0, &state, &mut rng);

        assert!(prov.is_none());
    }

    // ── Test 18: Higher Provocation Wins ───────────────────────

    #[test]
    fn test_higher_provocation_wins() {
        let player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.events.push(GameEvent::DamageDealt {
            attacker: 4,
            target: 1, // ally → Team(4)
            damage: 10,
        });
        state.events.push(GameEvent::DamageDealt {
            attacker: 5,
            target: 0, // us → Personal(5)
            damage: 15,
        });

        let mut rng = Rng::with_seed(42);
        let prov = player.detect_provocation(0, &state, &mut rng);

        // Personal (priority 2) > Team (priority 1)
        assert!(matches!(prov, Some(ProvokeLevel::Personal(5))));
    }

    // ── Test 19: Escalated Provocation Highest Priority ────────

    #[test]
    fn test_escalated_highest_priority() {
        let player = make_player(0, Class::Knight);
        let mut state = make_battle();
        state.events.push(GameEvent::DamageDealt {
            attacker: 4,
            target: 0, // us → Personal(4)
            damage: 10,
        });
        state.events.push(GameEvent::UnitDied {
            unit: 1, // ally → Escalated(5)
            killer: 5,
        });

        let mut rng = Rng::with_seed(42);
        let prov = player.detect_provocation(0, &state, &mut rng);

        // Escalated (priority 3) > Personal (priority 2)
        assert!(matches!(prov, Some(ProvokeLevel::Escalated(5))));
    }

    // ── Test 20: Nice Mode Monk Attacks Nearest ────────────────

    #[test]
    fn test_nice_mode_monk_attacks_nearest() {
        let mut player = make_player(3, Class::Monk);
        let mut state = make_battle();
        // Monk at (0,5), move enemy close enough for range 1
        state.units[7].pos = Pos::new(0, 4); // distance 1

        let mut rng = Rng::with_seed(42);
        let action = player.select_action(3, &state, &mut rng);

        assert_eq!(action.action_type, ActionType::Attack);
        assert_eq!(action.target_id, Some(7));
    }
}
