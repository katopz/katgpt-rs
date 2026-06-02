//! SR²AM FFT player — extends GZeroFFTPlayer with ConfiguratorBandit for learned per-turn planning regulation.
//!
//! Wraps GZero's template-based learning but adds a `ConfiguratorBandit` that decides
//! per-tick whether to do full template search (`PlanNew`), reuse last template
//! (`PlanExtend`), skip template entirely (`PlanSkip`), or let the speculator handle
//! prediction (`SpecHop`).
//!
//! # Architecture
//!
//! ```text
//! FftSr2amPlayer
//!   ├── FFTTemplateProposer         (UCB1 template selection)
//!   ├── DeltaBanditPruner           (δ as dense reward for arm selection)
//!   ├── DeltaGatedAbsorbCompress    (δ-gated absorb-compress)
//!   ├── Cross-round Q-values        (action-level bandit memory)
//!   └── ConfiguratorBandit          (SR²AM: learned planning regulation)
//!       ├── Shannon entropy context  (uncertainty measure)
//!       └── PlanNew/PlanExtend/PlanSkip/SpecHop arms
//! ```
//!
//! # Flow (per tick)
//!
//! 1. Compute base heuristic scores (query_scores)
//! 2. Compute Shannon entropy → bin context for ConfiguratorBandit
//! 3. Query ConfiguratorBandit → PlanningDecision
//! 4. Execute planning strategy (PlanNew / PlanExtend / PlanSkip / SpecHop)
//! 5. Compute δ, feed to bandit components
//! 6. Compute reward signal, update ConfiguratorBandit
//! 7. Blend hinted scores with Q-values, safety filter, ε-greedy
//! 8. Track decision for stats
//!
//! # Feature Gate
//!
//! Requires `sr2am_configurator` feature (implies `g_zero` + `bandit`).
//! When `sia_feedback` is also enabled, uses `FeedbackBandit` (6 arms) instead
//! of `ConfiguratorBandit` (4 arms).

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;
use katgpt_core::{ConfiguratorContext, PlanningDecision};

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
use crate::pruners::bandit::{BanditPruner, BanditStrategy};
use crate::pruners::configurator_bandit::ConfiguratorBandit;
use crate::pruners::g_zero::fft_templates::{self, FFTTemplate, FFTTemplateProposer};
use crate::pruners::g_zero::{DeltaBanditPruner, DeltaGatedAbsorbCompress, DeltaGatedConfig};
use crate::speculative::types::NoScreeningPruner;

use super::battle::BattleState;
use super::players::{
    FftPlayer, lowest_hp_ally, most_debuffed_ally, move_toward, nearest_enemy_pos, weakest_target,
};
use super::status;
use super::types::*;

// ── Constants ──────────────────────────────────────────────────

const HEURISTIC_WEIGHT: f32 = 0.8;
const BANDIT_WEIGHT: f32 = 0.2;
const EPSILON: f32 = 0.05;
const NUM_ACTIONS: usize = 9;
const NUM_TEMPLATES: usize = 10;
const FFT_DOMAIN: usize = 1; // FFT domain index (bomber uses 0)

// ── Helper Functions ───────────────────────────────────────────

/// Compute Shannon entropy on softmax-normalized scores (only valid actions).
fn shannon_entropy(scores: &[f32; NUM_ACTIONS]) -> f32 {
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores
        .iter()
        .map(|&s| {
            if s <= f32::NEG_INFINITY {
                0.0
            } else {
                (s - max_val).exp()
            }
        })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let entropy: f32 = exps
        .iter()
        .zip(scores.iter())
        .filter(|(e, _)| **e > 0.0)
        .map(|(e, _)| {
            let p = *e / sum;
            -p * p.ln()
        })
        .sum();
    entropy
}

/// Compute game-domain Hint-δ: mean delta over valid actions.
fn compute_game_delta(
    query_scores: &[f32; NUM_ACTIONS],
    hinted_scores: &[f32; NUM_ACTIONS],
) -> f32 {
    fft_templates::compute_game_delta(query_scores, hinted_scores)
}

/// Compute planning cost for reward signal.
fn planning_cost(decision: PlanningDecision) -> f32 {
    match decision {
        PlanningDecision::PlanNew => 1.0,
        PlanningDecision::PlanExtend => 0.3,
        PlanningDecision::PlanSkip => 0.0,
        PlanningDecision::SpecHop { k } => 0.1 * (k.min(8) as f32),
        #[cfg(feature = "sia_feedback")]
        PlanningDecision::HarnessUpdate => 0.5,
        #[cfg(feature = "sia_feedback")]
        PlanningDecision::WeightUpdate => 2.0, // Training is expensive
    }
}

// ── FftSr2amPlayer ─────────────────────────────────────────────

/// SR²AM FFT player — extends GZeroFFTPlayer with learned per-turn planning regulation.
///
/// Uses [`ConfiguratorBandit`] to decide per-tick whether to:
/// - `PlanNew`: Full template search (normal GZero behavior)
/// - `PlanExtend`: Reuse last template, re-evaluate with current state
/// - `PlanSkip`: Skip template entirely, use pure heuristic + Q-values only
/// - `SpecHop`: Skip template (speculator handles prediction independently)
pub struct FftSr2amPlayer {
    // G-Zero components (same as GZeroFFTPlayer)
    template_proposer: FFTTemplateProposer,
    delta_bandit: DeltaBanditPruner<NoScreeningPruner>,
    absorb_compress: DeltaGatedAbsorbCompress<NoScreeningPruner>,
    delta_history: Vec<f32>,
    round_actions: Vec<(ActionType, f32)>,
    round_template_ids: Vec<usize>,
    q_values: [f32; NUM_ACTIONS],
    visits: [u32; NUM_ACTIONS],
    last_template: Option<FFTTemplate>,
    last_template_id: Option<usize>,
    id: u8,
    // SR²AM additions
    #[cfg(not(feature = "sia_feedback"))]
    configurator: ConfiguratorBandit,
    #[cfg(feature = "sia_feedback")]
    configurator: crate::pruners::feedback_bandit::FeedbackBandit,
    decision_history: Vec<PlanningDecision>,
    plan_skip_count: usize,
    plan_new_count: usize,
    plan_extend_count: usize,
    plan_spechop_count: usize,
    #[cfg(feature = "sia_feedback")]
    plan_harness_count: usize,
    #[cfg(feature = "sia_feedback")]
    plan_weight_count: usize,
}

impl FftSr2amPlayer {
    /// Create a new FftSr2amPlayer with the given player ID.
    pub fn new(id: u8) -> Self {
        let bandit_inner =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);
        let delta_bandit = DeltaBanditPruner::new(bandit_inner, NUM_TEMPLATES);

        let absorb_inner =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());
        let absorb_compress =
            DeltaGatedAbsorbCompress::new(absorb_inner, NUM_TEMPLATES, DeltaGatedConfig::default());

        Self {
            template_proposer: FFTTemplateProposer::new(),
            delta_bandit,
            absorb_compress,
            delta_history: Vec::new(),
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: [0.0; NUM_ACTIONS],
            visits: [0; NUM_ACTIONS],
            last_template: None,
            last_template_id: None,
            id,
            #[cfg(not(feature = "sia_feedback"))]
            configurator: ConfiguratorBandit::new(),
            #[cfg(feature = "sia_feedback")]
            configurator: crate::pruners::feedback_bandit::FeedbackBandit::new(),
            decision_history: Vec::new(),
            plan_skip_count: 0,
            plan_new_count: 0,
            plan_extend_count: 0,
            plan_spechop_count: 0,
            #[cfg(feature = "sia_feedback")]
            plan_harness_count: 0,
            #[cfg(feature = "sia_feedback")]
            plan_weight_count: 0,
        }
    }

    /// Mean δ across all actions this round.
    fn round_delta_mean(&self) -> f32 {
        if self.round_actions.is_empty() {
            return 0.0;
        }
        self.round_actions.iter().map(|(_, d)| d).sum::<f32>() / self.round_actions.len() as f32
    }

    /// Update Q-values from episode outcome + feed outcome reward to template bandit.
    pub fn update_outcome(&mut self, survived: bool, kills: u32, damage: i32, healing: i32) {
        if self.round_actions.is_empty() {
            return;
        }

        let reward = if survived { 1.0 } else { -2.0 }
            + kills as f32 * 0.5
            + damage as f32 * 0.01
            + healing as f32 * 0.005;

        // Update per-action Q-values with blended reward
        for (action, delta) in &self.round_actions {
            let idx = action.as_usize();
            let alpha = 1.0f32 / (1.0f32 + self.visits[idx] as f32).sqrt();
            self.q_values[idx] += alpha * (reward + delta - self.q_values[idx]);
            self.visits[idx] += 1;
        }

        self.delta_history.push(self.round_delta_mean());
        self.round_actions.clear();
        self.round_template_ids.clear();
    }

    /// Run absorb-compress cycle.
    pub fn compress_cycle(&mut self) {
        self.absorb_compress.compress();
    }

    /// Get delta summary: (mean_δ, positive_ratio, best_template).
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

    /// Normalized pull distribution across templates.
    pub fn template_distribution(&self) -> Vec<(FFTTemplate, f32)> {
        self.template_proposer.template_distribution()
    }

    /// Player ID.
    pub fn id(&self) -> u8 {
        self.id
    }

    /// Get planning decision distribution: (plan_new, plan_extend, plan_skip, spechop) counts.
    pub fn decision_stats(&self) -> (usize, usize, usize, usize) {
        (
            self.plan_new_count,
            self.plan_extend_count,
            self.plan_skip_count,
            self.plan_spechop_count,
        )
    }

    /// Get FeedbackBandit decision counts: (harness, weight).
    #[cfg(feature = "sia_feedback")]
    pub fn feedback_decision_stats(&self) -> (usize, usize) {
        (self.plan_harness_count, self.plan_weight_count)
    }

    /// Compute query_scores — weak heuristic baseline (same as GZero).
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
                .find(|&e| {
                    state
                        .effects
                        .iter()
                        .any(|ef| ef.source == *e && ef.effect.is_buff())
                })
                .copied(),
            ActionType::Potion => Some(unit_id),
            _ => None,
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
}

// ── FftPlayer Trait ─────────────────────────────────────────────

impl FftPlayer for FftSr2amPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let reachable = state.reachable_positions(unit_id);

        // 1. Compute base heuristic scores
        let query_scores = Self::compute_query_scores(unit_id, state);

        // 2. Compute Shannon entropy → bin context for ConfiguratorBandit
        let entropy = shannon_entropy(&query_scores);
        let entropy_bin = ConfiguratorBandit::entropy_bin(entropy);
        let context = ConfiguratorContext::new(FFT_DOMAIN, entropy_bin);

        // 3. Query ConfiguratorBandit → PlanningDecision
        let decision = self.configurator.select(context);

        // 4. Execute planning strategy based on decision
        let (hinted_scores, template_id) = match decision {
            PlanningDecision::PlanNew => {
                // Full template search via UCB1 (normal GZero behavior)
                let (template, tid) = self.template_proposer.select();
                self.last_template = Some(template);
                self.last_template_id = Some(tid);
                self.round_template_ids.push(tid);
                let hinted = Self::compute_hinted_scores(template, &query_scores, state, unit_id);
                (hinted, Some(tid))
            }
            PlanningDecision::PlanExtend => {
                // Reuse last_template, recompute hint with current state
                match self.last_template {
                    Some(template) => {
                        let tid = self.last_template_id.unwrap_or(0);
                        self.round_template_ids.push(tid);
                        let hinted =
                            Self::compute_hinted_scores(template, &query_scores, state, unit_id);
                        (hinted, Some(tid))
                    }
                    None => {
                        // No previous template — fall back to PlanNew
                        let (template, tid) = self.template_proposer.select();
                        self.last_template = Some(template);
                        self.last_template_id = Some(tid);
                        self.round_template_ids.push(tid);
                        let hinted =
                            Self::compute_hinted_scores(template, &query_scores, state, unit_id);
                        (hinted, Some(tid))
                    }
                }
            }
            PlanningDecision::PlanSkip => {
                // Skip template entirely — use only heuristic query_scores + Q-values
                (query_scores, None)
            }
            PlanningDecision::SpecHop { .. } => {
                // SpecHop operates at hop level — skip template search here.
                // The speculator handles prediction independently; use query_scores as base.
                (query_scores, None)
            }
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::HarnessUpdate => {
                // HarnessUpdate: use current best template (like PlanExtend)
                match self.last_template {
                    Some(template) => {
                        let tid = self.last_template_id.unwrap_or(0);
                        self.round_template_ids.push(tid);
                        let hinted =
                            Self::compute_hinted_scores(template, &query_scores, state, unit_id);
                        (hinted, Some(tid))
                    }
                    None => (query_scores, None),
                }
            }
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::WeightUpdate => {
                // WeightUpdate: defer to Q-values (no template search)
                (query_scores, None)
            }
        };

        // 5. Compute δ (game-domain Hint-δ)
        let delta_value = compute_game_delta(&query_scores, &hinted_scores);

        // Feed δ to components (only if we actually used a template)
        if let Some(tid) = template_id {
            self.template_proposer.observe_delta(tid, delta_value);
            self.delta_bandit.observe_delta(tid, delta_value);
            self.absorb_compress
                .observe_delta(tid, delta_value, delta_value);
        }

        // 6. Compute reward signal and update configurator bandit
        let quality_gain = delta_value;
        let token_cost = planning_cost(decision);
        let reward = ConfiguratorBandit::reward_signal(quality_gain, token_cost, 0.1);
        self.configurator.update(context, decision, reward);

        // Track decision for stats
        match decision {
            PlanningDecision::PlanNew => self.plan_new_count += 1,
            PlanningDecision::PlanExtend => self.plan_extend_count += 1,
            PlanningDecision::PlanSkip => self.plan_skip_count += 1,
            PlanningDecision::SpecHop { .. } => self.plan_spechop_count += 1,
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::HarnessUpdate => self.plan_harness_count += 1,
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::WeightUpdate => self.plan_weight_count += 1,
        }
        self.decision_history.push(decision);

        // 7. Select action via blended scores + safety filter + ε-greedy
        let action_type = self.select_best_action(&hinted_scores, unit_id, state, rng);

        // 8. Record action with δ
        self.round_actions.push((action_type, delta_value));

        // 9. Resolve target
        let target_id = Self::resolve_target(action_type, unit_id, state);

        // 10. Movement: move toward target if out of range
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
        "SR2AM"
    }

    fn reset(&mut self) {
        self.round_actions.clear();
        self.round_template_ids.clear();
        self.last_template = None;
        self.last_template_id = None;
    }

    fn on_game_end(&mut self, _unit_id: u8, survived: bool, kills: u32, damage: i32, healing: i32) {
        self.update_outcome(survived, kills, damage, healing);
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Default for FftSr2amPlayer {
    fn default() -> Self {
        Self::new(0)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> BattleState {
        BattleState::new()
    }

    #[test]
    fn test_new_player_initial_state() {
        let player = FftSr2amPlayer::new(0);
        assert_eq!(player.id, 0);
        assert!(player.round_actions.is_empty());
        assert!(player.round_template_ids.is_empty());
        assert!(player.delta_history.is_empty());
        assert_eq!(player.q_values, [0.0; NUM_ACTIONS]);
        assert_eq!(player.visits, [0; NUM_ACTIONS]);
        assert!(player.last_template.is_none());
        assert!(player.last_template_id.is_none());
        // SR²AM additions start at zero
        let (new, extend, skip, spechop) = player.decision_stats();
        assert_eq!(new, 0);
        assert_eq!(extend, 0);
        assert_eq!(skip, 0);
        assert_eq!(spechop, 0);
    }

    #[test]
    fn test_select_action_returns_valid() {
        let mut player = FftSr2amPlayer::new(0);
        let state = make_state();
        let mut rng = Rng::with_seed(42);

        let action = player.select_action(0, &state, &mut rng);
        assert!(ActionType::all().contains(&action.action_type));
    }

    #[test]
    fn test_update_outcome_updates_q_values() {
        let mut player = FftSr2amPlayer::new(0);
        let state = make_state();
        let mut rng = Rng::with_seed(42);

        player.select_action(0, &state, &mut rng);
        assert!(!player.round_actions.is_empty());

        let q_before = player.q_values;
        player.update_outcome(true, 0, 10, 5);
        assert!(player.round_actions.is_empty());
        assert!(!player.delta_history.is_empty());

        // At least one Q-value should have changed
        let changed = player
            .q_values
            .iter()
            .zip(q_before.iter())
            .any(|(a, b)| (a - b).abs() > f32::EPSILON);
        assert!(changed, "Q-values should change after outcome update");
    }

    #[test]
    fn test_delta_summary() {
        let player = FftSr2amPlayer::new(0);

        // Empty → (0, 0, best_template)
        let (mean, pos_rate, _template) = player.delta_summary();
        assert_eq!(mean, 0.0);
        assert_eq!(pos_rate, 0.0);
    }

    #[test]
    fn test_decision_stats() {
        let mut player = FftSr2amPlayer::new(0);
        let state = make_state();
        let mut rng = Rng::with_seed(42);

        // Run enough ticks to exercise all decision paths
        for _ in 0..30 {
            player.select_action(0, &state, &mut rng);
        }

        let (new, extend, skip, spechop) = player.decision_stats();
        #[allow(unused_mut)]
        let mut total = new + extend + skip + spechop;
        // With sia_feedback, FeedbackBandit has 6 arms
        #[cfg(feature = "sia_feedback")]
        {
            let (harness, weight) = player.feedback_decision_stats();
            total += harness + weight;
        }
        assert_eq!(total, 30, "all 30 ticks should have a decision recorded");
        // UCB1 explores all arms, so we expect at least some of each type
        assert!(new > 0, "PlanNew should be selected at least once");
    }

    #[test]
    fn test_name() {
        let player = FftSr2amPlayer::new(0);
        assert_eq!(player.name(), "SR2AM");
    }

    #[test]
    fn test_default() {
        let player = FftSr2amPlayer::default();
        assert_eq!(player.id, 0);
        assert_eq!(player.name(), "SR2AM");
    }

    #[test]
    fn test_reset_clears_round() {
        let mut player = FftSr2amPlayer::new(0);
        let state = make_state();
        let mut rng = Rng::with_seed(42);

        player.select_action(0, &state, &mut rng);
        assert!(!player.round_actions.is_empty());

        player.reset();

        assert!(player.round_actions.is_empty());
        assert!(player.round_template_ids.is_empty());
        assert!(player.last_template.is_none());
        assert!(player.last_template_id.is_none());
        // Q-values persist
        assert_eq!(player.q_values, [0.0; NUM_ACTIONS]);
        // Decision stats persist across reset
        let (new, _extend, _skip, _spechop) = player.decision_stats();
        assert!(new > 0, "decision stats should persist across reset");
    }
}
