//! Rubric FFT Player — ROPD rubric-vector-aware FFT Tactics arena player.
//!
//! Replaces GZeroFFTPlayer's scalar δ bandit/absorb with rubric-vector-aware
//! components from the `ropd_rubric` module. This is the multi-axis domain
//! where rubrics should help most (Plan 071 hypothesis).
//!
//! # Architecture
//!
//! ```text
//! RubricFFTPlayer
//!   ├── FFTTemplateProposer       (UCB1 over 10 templates — same as GZeroFFT)
//!   ├── RubricBanditPruner        (rubric-vector reward for arm selection)
//!   ├── RubricGatedAbsorbCompress (rubric-gated absorb-compress)
//!   ├── q_values: [f32; 9]        (per-action Q-learning)
//!   └── round_actions             (episode tracking)
//! ```
//!
//! # Flow
//!
//! 1. Compute base heuristic scores for all 9 action types
//! 2. Select strategy template via UCB1
//! 3. Apply template score overrides → hinted scores
//! 4. Compute rubric vector from battle state + round stats
//! 5. Feed rubric to bandit/absorb, δ to proposer
//! 6. Blend hinted (80%) + Q-values (20%), safety filter, ε-greedy
//! 7. Record (action, δ) for episode outcome update
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "ropd_rubric")]`.

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;

use super::battle::BattleState;
use super::players::{
    FftPlayer, lowest_hp_ally, most_debuffed_ally, move_toward, nearest_enemy_pos, weakest_target,
};
use super::status;
use super::types::*;

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
use crate::pruners::bandit::{BanditPruner, BanditStrategy};
use crate::pruners::g_zero::fft_templates::{self, FFTTemplate, FFTTemplateProposer};
use crate::pruners::ropd_rubric::{
    RubricBanditPruner, RubricGatedAbsorbCompress, RubricGatedConfig, RubricTemplate, RubricVector,
};
use crate::speculative::types::NoScreeningPruner;

// ── Constants ──────────────────────────────────────────────────

const HEURISTIC_WEIGHT: f32 = 0.8;
const BANDIT_WEIGHT: f32 = 0.2;
const EPSILON: f32 = 0.05;
const NUM_ACTIONS: usize = 9;
const NUM_TEMPLATES: usize = 10;
const FFT_CRITERIA: usize = 3;

// ── RoundStats ─────────────────────────────────────────────────

/// Per-round metrics tracking for rubric scoring.
#[derive(Clone, Debug, Default)]
struct RoundStats {
    attacks_made: u32,
    heals_made: u32,
    debuffs_cured: u32,
    damage_dealt: i32,
}

// ── RubricFFTPlayer ────────────────────────────────────────────

/// ROPD rubric-vector-aware FFT player.
///
/// Replaces scalar Hint-δ with structured multi-criteria rubric vectors.
/// Uses [`RubricBanditPruner`] and [`RubricGatedAbsorbCompress`] for
/// vectorized reward signals across 3 criteria: TaskFulfillment, Completeness,
/// ConstraintSatisfaction.
pub struct RubricFFTPlayer {
    /// UCB1 proposer for strategy templates.
    template_proposer: FFTTemplateProposer,
    /// Rubric-vector-driven bandit pruner.
    rubric_bandit: RubricBanditPruner<NoScreeningPruner>,
    /// Rubric-gated absorb-compress for knowledge promotion.
    rubric_absorb: RubricGatedAbsorbCompress<NoScreeningPruner>,
    /// Perfect reference rubric for gap computation.
    reference_rubric: RubricVector,
    /// Per-round action metrics for rubric scoring.
    round_stats: RoundStats,
    /// Actions taken this round with their δ values.
    round_actions: Vec<(ActionType, f32)>,
    /// Per-action Q-values (updated from episode outcomes).
    q_values: [f32; NUM_ACTIONS],
    /// Per-action visit counts.
    visits: [u32; NUM_ACTIONS],
    /// Last selected template for introspection.
    last_template: Option<FFTTemplate>,
    /// Player identifier.
    id: u8,
}

impl RubricFFTPlayer {
    /// Create a new rubric FFT player with given ID.
    pub fn new(id: u8) -> Self {
        let inner_bandit =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);
        let rubric_bandit = RubricBanditPruner::new(inner_bandit, NUM_TEMPLATES, FFT_CRITERIA);

        let compress_config = CompressConfig::default();
        let absorb_layer =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, compress_config);
        let rubric_absorb = RubricGatedAbsorbCompress::new(
            absorb_layer,
            NUM_TEMPLATES,
            RubricGatedConfig::default(),
        );

        let template = RubricTemplate::fft_tactics();
        let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();
        let reference_rubric = RubricVector::perfect(weights, 0);

        Self {
            template_proposer: FFTTemplateProposer::new(),
            rubric_bandit,
            rubric_absorb,
            reference_rubric,
            round_stats: RoundStats::default(),
            round_actions: Vec::new(),
            q_values: [0.0; NUM_ACTIONS],
            visits: [0; NUM_ACTIONS],
            last_template: None,
            id,
        }
    }

    /// Update Q-values from episode outcome.
    ///
    /// Computes reward from survival, kills, damage, healing.
    /// Also computes outcome rubric and feeds to bandit/absorb.
    pub fn update_outcome(&mut self, survived: bool, kills: u32, damage: i32, healing: i32) {
        let reward = if survived { 1.0 } else { -2.0 }
            + kills as f32 * 0.5
            + damage as f32 * 0.01
            + healing as f32 * 0.005;

        for (action, delta) in &self.round_actions {
            let idx = action.as_usize();
            let alpha = 1.0 / (1.0 + self.visits[idx] as f32).sqrt();
            self.q_values[idx] += alpha * (reward + delta - self.q_values[idx]);
            self.visits[idx] += 1;
        }

        // Compute outcome rubric and feed to bandit/absorb
        let outcome_rubric = Self::compute_outcome_rubric(survived, kills, damage, healing);
        self.rubric_bandit
            .observe_rubric(0, &outcome_rubric, &self.reference_rubric);
        self.rubric_absorb.observe_rubric(
            0,
            &outcome_rubric,
            std::slice::from_ref(&self.reference_rubric),
        );

        self.round_actions.clear();
        self.round_stats = RoundStats::default();
    }

    /// Run absorb-compress cycle to promote stable knowledge.
    pub fn compress_cycle(&mut self) {
        self.rubric_absorb.compress();
    }

    /// Player ID.
    pub fn id(&self) -> u8 {
        self.id
    }

    /// Compute FFT rubric from battle state and round stats.
    fn compute_fft_rubric(
        unit: &Unit,
        _state: &BattleState,
        round_stats: &RoundStats,
    ) -> RubricVector {
        let template = RubricTemplate::fft_tactics();
        let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();

        // TaskFulfillment — role-dependent scoring
        let task_fulfillment = match unit.class {
            Class::Knight | Class::Archer | Class::Monk => {
                (round_stats.attacks_made as f32 / 3.0).min(1.0)
            }
            Class::WhiteMage => (round_stats.heals_made as f32 / 2.0).min(1.0),
            Class::BlackMage => (round_stats.damage_dealt as f32 / 50.0).min(1.0),
            Class::TimeMage => 0.5 + 0.5 * (round_stats.debuffs_cured as f32 / 1.0).min(1.0),
        };

        // Completeness — team contribution
        let completeness =
            ((round_stats.heals_made + round_stats.debuffs_cured) as f32 / 3.0).min(1.0);

        // ConstraintSatisfaction — survival
        let constraint_satisfaction = if unit.alive { unit.hp_pct() } else { 0.0 };

        let scores = vec![task_fulfillment, completeness, constraint_satisfaction];
        RubricVector::new(scores, weights, 0)
    }

    /// Compute outcome rubric from episode results.
    fn compute_outcome_rubric(
        survived: bool,
        kills: u32,
        _damage: i32,
        healing: i32,
    ) -> RubricVector {
        let template = RubricTemplate::fft_tactics();
        let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();

        let task_fulfillment = (kills as f32 / 2.0).min(1.0);
        let completeness = (healing as f32 / 30.0).clamp(0.0, 1.0);
        let constraint_satisfaction = if survived { 1.0 } else { 0.0 };

        let scores = vec![task_fulfillment, completeness, constraint_satisfaction];
        RubricVector::new(scores, weights, 0)
    }

    /// Update round stats based on selected action type.
    fn update_round_stats(&mut self, action_type: ActionType) {
        match action_type {
            ActionType::Attack | ActionType::BlackMagic => {
                self.round_stats.attacks_made += 1;
            }
            ActionType::WhiteMagic => {
                self.round_stats.heals_made += 1;
            }
            ActionType::CurePoison | ActionType::Esuna => {
                self.round_stats.debuffs_cured += 1;
            }
            _ => {}
        }
    }

    /// Compute base heuristic score for an action.
    fn heuristic_score(action: ActionType, unit_id: u8, state: &BattleState) -> f32 {
        let unit = &state.units[unit_id as usize];
        let hp_pct = unit.hp_pct();
        let enemy_team = BattleState::enemy_team(unit.team);
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
        let can_cast = status::can_cast(unit, &state.effects);

        match action {
            ActionType::Attack if !enemies.is_empty() => 2.0,
            ActionType::Defend => 1.0,
            ActionType::BlackMagic
                if !enemies.is_empty() && can_cast && unit.can_afford(action) =>
            {
                2.5
            }
            ActionType::WhiteMagic if !allies.is_empty() && can_cast && unit.can_afford(action) => {
                let wounded = allies
                    .iter()
                    .any(|&a| state.units[a as usize].hp_pct() < 0.7);
                if wounded { 3.0 } else { 0.5 }
            }
            ActionType::Potion if hp_pct < 0.5 && unit.can_afford(action) => 3.0,
            ActionType::CurePoison if can_cast && unit.can_afford(action) => {
                let poisoned = allies.iter().any(|&a| {
                    state
                        .effects
                        .iter()
                        .any(|e| e.source == a && e.effect == status::StatusEffect::Poison)
                });
                if poisoned { 2.5 } else { 0.0 }
            }
            ActionType::Esuna if can_cast && unit.can_afford(action) => {
                let debuffed = allies.iter().any(|&a| {
                    state
                        .effects
                        .iter()
                        .any(|e| e.source == a && e.effect.esuna_curable())
                });
                if debuffed { 2.0 } else { 0.0 }
            }
            ActionType::Dispel if can_cast && unit.can_afford(action) => {
                let buffed = enemies.iter().any(|&e| {
                    state
                        .effects
                        .iter()
                        .any(|ef| ef.source == e && ef.effect.is_buff())
                });
                if buffed { 2.0 } else { 0.0 }
            }
            ActionType::Wait => 0.0,
            _ => f32::NEG_INFINITY,
        }
    }

    /// Compute query scores for all 9 action types.
    fn compute_query_scores(unit_id: u8, state: &BattleState) -> [f32; NUM_ACTIONS] {
        let mut scores = [f32::NEG_INFINITY; NUM_ACTIONS];
        for (i, action) in ActionType::all().into_iter().enumerate() {
            scores[i] = Self::heuristic_score(action, unit_id, state);
        }
        scores
    }

    /// Apply template overrides to get hinted scores.
    fn compute_hinted_scores(
        template: FFTTemplate,
        query_scores: &[f32; NUM_ACTIONS],
        state: &BattleState,
        unit_id: u8,
    ) -> [f32; NUM_ACTIONS] {
        let mut hinted = *query_scores;
        for (i, action) in ActionType::all().into_iter().enumerate() {
            if hinted[i] > f32::NEG_INFINITY {
                let override_val =
                    fft_templates::hint_score_override(template, action, state, unit_id);
                hinted[i] += override_val;
            }
        }
        hinted
    }

    /// Safety filter: check if action is actually executable.
    fn is_action_available(action: ActionType, unit_id: u8, state: &BattleState) -> bool {
        let unit = &state.units[unit_id as usize];
        let enemy_team = BattleState::enemy_team(unit.team);
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
        let can_cast = status::can_cast(unit, &state.effects);

        match action {
            ActionType::Attack => !enemies.is_empty(),
            ActionType::Defend => true,
            ActionType::BlackMagic => !enemies.is_empty() && can_cast && unit.can_afford(action),
            ActionType::WhiteMagic => !allies.is_empty() && can_cast && unit.can_afford(action),
            ActionType::Potion => unit.can_afford(action) && unit.hp_pct() < 0.5,
            ActionType::Wait => true,
            ActionType::CurePoison => {
                can_cast
                    && unit.can_afford(action)
                    && allies.iter().any(|&a| {
                        state
                            .effects
                            .iter()
                            .any(|e| e.source == a && e.effect == status::StatusEffect::Poison)
                    })
            }
            ActionType::Esuna => {
                can_cast
                    && unit.can_afford(action)
                    && allies.iter().any(|&a| {
                        state
                            .effects
                            .iter()
                            .any(|e| e.source == a && e.effect.esuna_curable())
                    })
            }
            ActionType::Dispel => {
                can_cast
                    && unit.can_afford(action)
                    && enemies.iter().any(|&e| {
                        state
                            .effects
                            .iter()
                            .any(|ef| ef.source == e && ef.effect.is_buff())
                    })
            }
        }
    }

    /// Select best action from blended scores with safety filter.
    fn select_best_action(
        &self,
        hinted_scores: &[f32; NUM_ACTIONS],
        unit_id: u8,
        state: &BattleState,
        rng: &mut Rng,
    ) -> ActionType {
        // ε-greedy exploration
        if rng.f32() < EPSILON {
            let available: Vec<ActionType> = ActionType::all()
                .into_iter()
                .filter(|&a| Self::is_action_available(a, unit_id, state))
                .collect();
            if !available.is_empty() {
                return available[rng.usize(..available.len())];
            }
        }

        // Blend hinted scores (80%) with Q-values (20%)
        let mut blended = [f32::NEG_INFINITY; NUM_ACTIONS];
        for (i, action) in ActionType::all().into_iter().enumerate() {
            if !Self::is_action_available(action, unit_id, state) {
                continue;
            }
            let hinted = if hinted_scores[i] > f32::NEG_INFINITY {
                hinted_scores[i]
            } else {
                0.0
            };
            let q_val = self.q_values[i];
            blended[i] = HEURISTIC_WEIGHT * hinted + BANDIT_WEIGHT * q_val;
        }

        // Pick best blended action
        (0..NUM_ACTIONS)
            .filter(|&i| blended[i] > f32::NEG_INFINITY)
            .max_by(|a, b| {
                blended[*a]
                    .partial_cmp(&blended[*b])
                    .unwrap_or(Ordering::Equal)
            })
            .map(ActionType::from)
            .unwrap_or(ActionType::Wait)
    }

    /// Resolve target for the selected action type.
    fn resolve_target(action_type: ActionType, unit_id: u8, state: &BattleState) -> Option<u8> {
        let unit = &state.units[unit_id as usize];
        let enemy_team = BattleState::enemy_team(unit.team);
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);

        match action_type {
            ActionType::Attack | ActionType::BlackMagic => weakest_target(state, &enemies),
            ActionType::WhiteMagic => lowest_hp_ally(state, &allies),
            ActionType::CurePoison => allies
                .iter()
                .find(|&&a| {
                    state
                        .effects
                        .iter()
                        .any(|e| e.source == a && e.effect == status::StatusEffect::Poison)
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
        }
    }
}

// ── FftPlayer Trait ────────────────────────────────────────────

impl FftPlayer for RubricFFTPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let reachable = state.reachable_positions(unit_id);

        // 1. Compute base heuristic scores
        let query_scores = Self::compute_query_scores(unit_id, state);

        // 2. Select strategy template via UCB1
        let (template, template_id) = self.template_proposer.select();
        self.last_template = Some(template);

        // 3. Compute hinted scores with template overrides
        let hinted_scores = Self::compute_hinted_scores(template, &query_scores, state, unit_id);

        // 4. Compute scalar δ for proposer + Q-learning compatibility
        let delta = fft_templates::compute_game_delta(&query_scores, &hinted_scores);

        // 5. Compute current rubric vector from battle state
        let student_rubric = Self::compute_fft_rubric(unit, state, &self.round_stats);

        // 6. Feed δ to proposer (same as GZeroFFT)
        self.template_proposer.observe_delta(template_id, delta);

        // 7. Feed rubric observation to bandit and absorb
        self.rubric_bandit
            .observe_rubric(template_id, &student_rubric, &self.reference_rubric);
        self.rubric_absorb.observe_rubric(
            template_id,
            &student_rubric,
            std::slice::from_ref(&self.reference_rubric),
        );

        // 8. Select action via blended scores + safety filter + ε-greedy
        let action_type = self.select_best_action(&hinted_scores, unit_id, state, rng);

        // 9. Record action with δ and update round stats
        self.round_actions.push((action_type, delta));
        self.update_round_stats(action_type);

        // 10. Resolve target
        let target_id = Self::resolve_target(action_type, unit_id, state);

        // 11. Movement: move toward target if out of range
        let move_to = if let Some(tid) = target_id {
            let target_pos = state.units[tid as usize].pos;
            if unit.pos.manhattan(target_pos) > unit.stats.range {
                move_toward(&reachable, target_pos)
            } else {
                None
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
        "RubricFFT"
    }

    fn reset(&mut self) {
        self.round_actions.clear();
        self.round_stats = RoundStats::default();
        self.last_template = None;
    }

    fn on_game_end(&mut self, _unit_id: u8, survived: bool, kills: u32, damage: i32, healing: i32) {
        self.update_outcome(survived, kills, damage, healing);
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Default for RubricFFTPlayer {
    fn default() -> Self {
        Self::new(0)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit(id: u8, class: Class, team: Team) -> Unit {
        Unit::new(id, class, team, Pos::new(1, 1))
    }

    #[test]
    fn test_compute_fft_rubric_knight_attacking() {
        let unit = make_unit(0, Class::Knight, Team::Party);
        let state = BattleState::new();
        let round_stats = RoundStats {
            attacks_made: 3,
            heals_made: 0,
            debuffs_cured: 0,
            damage_dealt: 0,
        };

        let rubric = RubricFFTPlayer::compute_fft_rubric(&unit, &state, &round_stats);
        // Knight with 3 attacks → TaskFulfillment = min(3/3, 1.0) = 1.0
        assert!(
            (rubric.score(0) - 1.0).abs() < 1e-6,
            "TaskFulfillment should be 1.0, got {}",
            rubric.score(0)
        );
    }

    #[test]
    fn test_compute_fft_rubric_healer() {
        let unit = make_unit(0, Class::WhiteMage, Team::Party);
        let state = BattleState::new();
        let round_stats = RoundStats {
            attacks_made: 0,
            heals_made: 2,
            debuffs_cured: 1,
            damage_dealt: 0,
        };

        let rubric = RubricFFTPlayer::compute_fft_rubric(&unit, &state, &round_stats);
        // WhiteMage with 2 heals → TaskFulfillment = min(2/2, 1.0) = 1.0
        assert!(
            (rubric.score(0) - 1.0).abs() < 1e-6,
            "TaskFulfillment should be 1.0"
        );
        // Completeness = min((2+1)/3, 1.0) = 1.0
        assert!(
            (rubric.score(1) - 1.0).abs() < 1e-6,
            "Completeness should be 1.0"
        );
    }

    #[test]
    fn test_compute_fft_rubric_dead() {
        let mut unit = make_unit(0, Class::Knight, Team::Party);
        unit.alive = false;
        unit.hp = 0;
        let state = BattleState::new();
        let round_stats = RoundStats {
            attacks_made: 3,
            heals_made: 0,
            debuffs_cured: 0,
            damage_dealt: 0,
        };

        let rubric = RubricFFTPlayer::compute_fft_rubric(&unit, &state, &round_stats);
        // Dead unit → ConstraintSatisfaction = 0.0
        assert!(
            rubric.score(2).abs() < 1e-6,
            "ConstraintSatisfaction should be 0.0 when dead"
        );
    }

    #[test]
    fn test_new_player_initial_state() {
        let player = RubricFFTPlayer::new(3);
        assert_eq!(player.id(), 3);
        assert_eq!(player.last_template, None);
        assert!(player.round_actions.is_empty());
        assert_eq!(player.q_values, [0.0; 9]);
        assert_eq!(player.visits, [0; 9]);
        assert_eq!(player.reference_rubric.len(), FFT_CRITERIA);
    }

    #[test]
    fn test_select_action_returns_valid() {
        let mut player = RubricFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        let action = player.select_action(0, &state, &mut rng);
        assert!(matches!(
            action.action_type,
            ActionType::Attack
                | ActionType::Defend
                | ActionType::BlackMagic
                | ActionType::WhiteMagic
                | ActionType::Wait
        ));
    }

    #[test]
    fn test_update_outcome_updates_q_values() {
        let mut player = RubricFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        let action = player.select_action(0, &state, &mut rng);
        let action_type = action.action_type;

        player.update_outcome(true, 2, 50, 30);

        let idx = action_type.as_usize();
        assert!(player.visits[idx] > 0);
        assert!(player.q_values[idx] != 0.0);
    }

    #[test]
    fn test_reset_clears_round() {
        let mut player = RubricFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        player.select_action(0, &state, &mut rng);
        assert!(!player.round_actions.is_empty());

        player.reset();
        assert!(player.round_actions.is_empty());
        assert!(player.last_template.is_none());
        assert_eq!(player.round_stats.attacks_made, 0);
    }
}
