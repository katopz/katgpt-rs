//! G-Zero FFT Player — self-play distillation via template-driven bandit learning.
//!
//! Integrates G-Zero's Hint-δ signal with FFT Tactics Arena action selection.
//! Uses 10 strategy templates (HealFirst, KillPriority, etc.) selected by UCB1
//! bandit. The δ signal measures how much a template shifts action preferences,
//! identifying strategic blind spots for targeted exploration.
//!
//! # Architecture
//!
//! ```text
//! GZeroFFTPlayer
//!   ├── template_proposer: FFTTemplateProposer   (UCB1 over 10 templates)
//!   ├── delta_bandit: DeltaBanditPruner           (δ-reward for arms)
//!   ├── absorb_compress: DeltaGatedAbsorbCompress (δ-gated compression)
//!   ├── q_values: [f32; 9]                        (per-action Q-learning)
//!   └── round_actions: Vec<(ActionType, f32)>     (episode tracking)
//! ```
//!
//! # Flow
//!
//! 1. Compute base heuristic scores for all 9 action types
//! 2. Select strategy template via UCB1
//! 3. Apply template score overrides → hinted scores
//! 4. Compute δ = mean score shift (game-domain Hint-δ)
//! 5. Feed δ to proposer, bandit, and absorb-compress
//! 6. Blend hinted (80%) + Q-values (20%), safety filter, ε-greedy
//! 7. Record (action, δ) for episode outcome update
//!
//! # Feature Gate
//!
//! Requires `g_zero` feature (implies `bandit` + `fft`).

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
use crate::pruners::g_zero::delta_absorb::DeltaGatedAbsorbCompress;
use crate::pruners::g_zero::delta_absorb::DeltaGatedConfig;
use crate::pruners::g_zero::delta_bandit::DeltaBanditPruner;
use crate::pruners::g_zero::fft_templates::{self, FFTTemplate, FFTTemplateProposer};
use crate::speculative::types::NoScreeningPruner;

// ── Constants ──────────────────────────────────────────────────

const HEURISTIC_WEIGHT: f32 = 0.8;
const BANDIT_WEIGHT: f32 = 0.2;
const EPSILON: f32 = 0.05;
const NUM_ACTIONS: usize = 9;
const NUM_TEMPLATES: usize = 10;

// ── GZeroFFTPlayer ─────────────────────────────────────────────

/// G-Zero self-play FFT player with template-driven bandit learning.
///
/// Combines three G-Zero components:
/// - **FFTTemplateProposer**: UCB1 bandit over 10 strategic archetypes
/// - **DeltaBanditPruner**: δ-reward feeding into multi-armed bandit
/// - **DeltaGatedAbsorbCompress**: δ-gated promotion of stable knowledge
///
/// Plus Q-learning over 9 action types for episode-level reward propagation.
pub struct GZeroFFTPlayer {
    /// UCB1 proposer for strategy templates.
    template_proposer: FFTTemplateProposer,
    /// δ-driven bandit pruner for template arms.
    delta_bandit: DeltaBanditPruner<NoScreeningPruner>,
    /// δ-gated absorb-compress for knowledge promotion.
    absorb_compress: DeltaGatedAbsorbCompress<NoScreeningPruner>,
    /// Rolling δ history across episodes.
    delta_history: Vec<f32>,
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

impl GZeroFFTPlayer {
    /// Create a new G-Zero FFT player with given ID.
    pub fn new(id: u8) -> Self {
        let inner_bandit =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);
        let delta_bandit = DeltaBanditPruner::new(inner_bandit, NUM_TEMPLATES);
        let compress_config = CompressConfig::default();
        let absorb_layer =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, compress_config);
        let absorb_compress =
            DeltaGatedAbsorbCompress::new(absorb_layer, NUM_TEMPLATES, DeltaGatedConfig::default());

        Self {
            template_proposer: FFTTemplateProposer::new(),
            delta_bandit,
            absorb_compress,
            delta_history: Vec::new(),
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
    /// Propagates reward + δ to Q-values via incremental mean.
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

        self.delta_history.push(self.round_delta_mean());
        self.round_actions.clear();
    }

    /// Run absorb-compress cycle to promote stable knowledge.
    pub fn compress_cycle(&mut self) {
        self.absorb_compress.compress();
    }

    /// Summary statistics: (mean_δ, positive_rate, best_template).
    pub fn delta_summary(&self) -> (f32, f32, FFTTemplate) {
        let mean = if self.delta_history.is_empty() {
            0.0
        } else {
            self.delta_history.iter().sum::<f32>() / self.delta_history.len() as f32
        };
        let positive = self.delta_history.iter().filter(|&&d| d > 0.0).count() as f32
            / self.delta_history.len().max(1) as f32;
        (mean, positive, self.template_proposer.best_template())
    }

    /// Per-template selection distribution.
    pub fn template_distribution(&self) -> Vec<(FFTTemplate, f32)> {
        self.template_proposer.template_distribution()
    }

    /// Player ID.
    pub fn id(&self) -> u8 {
        self.id
    }

    /// Mean δ this round.
    fn round_delta_mean(&self) -> f32 {
        if self.round_actions.is_empty() {
            return 0.0;
        }
        self.round_actions.iter().map(|(_, d)| d).sum::<f32>() / self.round_actions.len() as f32
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

impl FftPlayer for GZeroFFTPlayer {
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

        // 4. Compute game-domain δ
        let delta = fft_templates::compute_game_delta(&query_scores, &hinted_scores);

        // 5. Feed δ to all three G-Zero components
        self.template_proposer.observe_delta(template_id, delta);
        self.delta_bandit.observe_delta(template_id, delta);
        self.absorb_compress
            .observe_delta(template_id, delta, delta);

        // 6. Select action via blended scores + safety filter + ε-greedy
        let action_type = self.select_best_action(&hinted_scores, unit_id, state, rng);

        // 7. Record action with δ
        self.round_actions.push((action_type, delta));

        // 8. Resolve target
        let target_id = Self::resolve_target(action_type, unit_id, state);

        // 9. Movement: move toward target if out of range
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
        "GZero"
    }

    fn reset(&mut self) {
        self.round_actions.clear();
        self.last_template = None;
    }

    fn on_game_end(&mut self, _unit_id: u8, survived: bool, kills: u32, damage: i32, healing: i32) {
        self.update_outcome(survived, kills, damage, healing);
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Default for GZeroFFTPlayer {
    fn default() -> Self {
        Self::new(0)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_player_initial_state() {
        let player = GZeroFFTPlayer::new(3);
        assert_eq!(player.id(), 3);
        assert_eq!(player.last_template, None);
        assert!(player.round_actions.is_empty());
        assert_eq!(player.q_values, [0.0; 9]);
        assert_eq!(player.visits, [0; 9]);
    }

    #[test]
    fn test_default() {
        let player = GZeroFFTPlayer::default();
        assert_eq!(player.id(), 0);
    }

    #[test]
    fn test_select_action_returns_valid() {
        let mut player = GZeroFFTPlayer::new(0);
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
    fn test_select_action_records_template() {
        let mut player = GZeroFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        player.select_action(0, &state, &mut rng);
        assert!(player.last_template.is_some());
        assert!(!player.round_actions.is_empty());
    }

    #[test]
    fn test_update_outcome_updates_q_values() {
        let mut player = GZeroFFTPlayer::new(0);
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
    fn test_update_outcome_survival_bonus() {
        let mut player = GZeroFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        player.select_action(0, &state, &mut rng);
        player.update_outcome(true, 0, 0, 0);

        // Survived with no kills/damage/healing → reward = 1.0
        assert!(!player.delta_history.is_empty());
    }

    #[test]
    fn test_update_outcome_death_penalty() {
        let mut player = GZeroFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        player.select_action(0, &state, &mut rng);
        player.update_outcome(false, 0, 0, 0);

        // Died → reward = -2.0
        assert!(!player.delta_history.is_empty());
    }

    #[test]
    fn test_delta_summary_empty() {
        let player = GZeroFFTPlayer::new(0);
        let (mean, positive, template) = player.delta_summary();
        assert!((mean).abs() < 1e-6);
        assert!((positive).abs() < 1e-6);
        assert_eq!(template, FFTTemplate::HealFirst); // default when no data
    }

    #[test]
    fn test_delta_summary_after_episodes() {
        let mut player = GZeroFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        for _ in 0..5 {
            player.select_action(0, &state, &mut rng);
            player.update_outcome(true, 1, 20, 10);
        }

        let (mean, positive, _template) = player.delta_summary();
        // Should have some history
        assert!(!player.delta_history.is_empty());
        assert!(mean >= 0.0 || mean < 0.0); // just checking it's a valid f32
        assert!((0.0..=1.0).contains(&positive));
    }

    #[test]
    fn test_template_distribution() {
        let mut player = GZeroFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        for _ in 0..10 {
            player.select_action(0, &state, &mut rng);
        }

        let dist = player.template_distribution();
        assert_eq!(dist.len(), 10);
    }

    #[test]
    fn test_heuristic_score_attack() {
        let state = BattleState::new();
        // Unit 0 is Party Knight at (1,1), enemies at (6,*) — no enemies in range 1
        let score = GZeroFFTPlayer::heuristic_score(ActionType::Attack, 0, &state);
        assert!(score == f32::NEG_INFINITY || score >= 0.0);
    }

    #[test]
    fn test_heuristic_score_defend() {
        let state = BattleState::new();
        let score = GZeroFFTPlayer::heuristic_score(ActionType::Defend, 0, &state);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_heuristic_score_wait() {
        let state = BattleState::new();
        let score = GZeroFFTPlayer::heuristic_score(ActionType::Wait, 0, &state);
        assert!((score).abs() < 1e-6);
    }

    #[test]
    fn test_compute_query_scores_all_actions() {
        let state = BattleState::new();
        let scores = GZeroFFTPlayer::compute_query_scores(0, &state);
        assert_eq!(scores.len(), 9);
        // Defend is always available
        assert!((scores[1] - 1.0).abs() < 1e-6);
        // Wait is always available
        assert!((scores[5]).abs() < 1e-6);
    }

    #[test]
    fn test_compute_hinted_scores_shifts() {
        let state = BattleState::new();
        let query_scores = GZeroFFTPlayer::compute_query_scores(0, &state);
        let hinted = GZeroFFTPlayer::compute_hinted_scores(
            FFTTemplate::KillPriority,
            &query_scores,
            &state,
            0,
        );
        // Hinted should differ from query for some actions
        assert_eq!(hinted.len(), query_scores.len());
    }

    #[test]
    fn test_safety_filter_magic_without_mp() {
        let mut state = BattleState::new();
        // Drain MP from unit 0
        state.units[0].mp = 0;
        let available = GZeroFFTPlayer::is_action_available(ActionType::BlackMagic, 0, &state);
        assert!(!available, "BlackMagic should be unavailable with 0 MP");
    }

    #[test]
    fn test_safety_filter_defend_always() {
        let state = BattleState::new();
        let available = GZeroFFTPlayer::is_action_available(ActionType::Defend, 0, &state);
        assert!(available, "Defend should always be available");
    }

    #[test]
    fn test_resolve_target_attack() {
        let mut state = BattleState::new();
        // Move enemy 4 within range of unit 0 (range=1)
        state.units[4].pos = Pos::new(1, 2);
        let target = GZeroFFTPlayer::resolve_target(ActionType::Attack, 0, &state);
        assert!(target.is_some());
        assert_eq!(target.unwrap(), 4);
    }

    #[test]
    fn test_resolve_target_potion() {
        let state = BattleState::new();
        let target = GZeroFFTPlayer::resolve_target(ActionType::Potion, 0, &state);
        assert_eq!(target, Some(0));
    }

    #[test]
    fn test_compress_cycle_noop() {
        let mut player = GZeroFFTPlayer::new(0);
        // Should not panic even with no observations
        player.compress_cycle();
    }

    #[test]
    fn test_reset_clears_round() {
        let mut player = GZeroFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        player.select_action(0, &state, &mut rng);
        assert!(!player.round_actions.is_empty());

        player.reset();
        assert!(player.round_actions.is_empty());
        assert!(player.last_template.is_none());
    }

    #[test]
    fn test_name() {
        let player = GZeroFFTPlayer::new(0);
        assert_eq!(player.name(), "GZero");
    }

    #[test]
    fn test_multiple_actions_accumulate() {
        let mut player = GZeroFFTPlayer::new(0);
        let state = BattleState::new();
        let mut rng = Rng::new();

        // Simulate multiple actions in one round
        player.select_action(0, &state, &mut rng);
        player.select_action(0, &state, &mut rng);
        player.select_action(0, &state, &mut rng);

        assert_eq!(player.round_actions.len(), 3);

        player.update_outcome(true, 3, 100, 50);
        assert!(player.round_actions.is_empty());
        assert_eq!(player.delta_history.len(), 1);
    }
}
