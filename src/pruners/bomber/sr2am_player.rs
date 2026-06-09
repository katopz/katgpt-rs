//! SR²AM bomber player — extends GZero with ConfiguratorBandit for learned per-turn planning regulation.
//!
//! Wraps GZero's template-based learning but adds a `ConfiguratorBandit` that decides
//! per-tick whether to do full template search (`PlanNew`), reuse last template
//! (`PlanExtend`), or skip template entirely (`PlanSkip`).
//!
//! # Architecture
//!
//! ```text
//! Sr2amPlayer
//!   ├── BomberTemplateProposer      (UCB1 template selection)
//!   ├── DeltaBanditPruner           (δ as dense reward for arm selection)
//!   ├── DeltaGatedAbsorbCompress    (δ-gated absorb-compress)
//!   ├── Cross-round Q-values        (action-level bandit memory)
//!   └── ConfiguratorBandit          (SR²AM: learned planning regulation)
//!       ├── Shannon entropy context  (uncertainty measure)
//!       └── PlanNew/PlanExtend/PlanSkip arms
//! ```
//!
//! # Flow (per tick)
//!
//! 1. Update game state from events
//! 2. Compute heuristic baseline (query_scores)
//! 3. Compute Shannon entropy → bin context for ConfiguratorBandit
//! 4. Query ConfiguratorBandit → PlanningDecision
//! 5. Execute planning strategy (PlanNew / PlanExtend / PlanSkip)
//! 6. Compute δ, feed to bandit components
//! 7. Compute reward signal, update ConfiguratorBandit
//! 8. Blend scores with Q-values, safety filter, ε-greedy
//! 9. Track decision for stats

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;
use katgpt_core::{ConfiguratorContext, PlanningDecision};

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
use crate::pruners::bandit::{BanditPruner, BanditStrategy};
use crate::pruners::configurator_bandit::ConfiguratorBandit;
use crate::pruners::g_zero::{
    BomberTemplate, BomberTemplateProposer, DeltaBanditPruner, DeltaGatedAbsorbCompress,
    DeltaGatedConfig, hint_score_override,
};
use crate::speculative::types::NoScreeningPruner;

use super::players::BomberPlayer;
use super::players::{in_blast_zone, score_action, should_place_bomb};
use super::{
    ARENA_H, ARENA_W, ArenaGrid, BOMB_FUSE_TICKS, BomberAction, BomberFrozenBandit, Cell,
    DEFAULT_BLAST_RANGE, GameEvent, GridPos,
};

// ── Constants ──────────────────────────────────────────────────

const ACTION_COUNT: usize = 7;
const NUM_TEMPLATES: usize = 8;
const SR2AM_DOMAIN: usize = 0; // Bomber domain index for configurator

const ALL_ACTIONS: [BomberAction; ACTION_COUNT] = [
    BomberAction::Up,
    BomberAction::Down,
    BomberAction::Left,
    BomberAction::Right,
    BomberAction::Bomb,
    BomberAction::Wait,
    BomberAction::Detonate,
];

/// Tracked bomb: (position, blast_range, fuse_ticks_remaining).
type KnownBomb = ((i32, i32), u32, u32);

/// Tracked opponent: (player_id, current_pos, prev_pos).
type KnownOpponent = (u8, (i32, i32), Option<(i32, i32)>);

// ── Helper Functions ───────────────────────────────────────────

/// Compute target position after applying a move action.
fn move_target(action: BomberAction, pos: GridPos) -> GridPos {
    match action {
        BomberAction::Up => GridPos {
            x: pos.x,
            y: pos.y - 1,
        },
        BomberAction::Down => GridPos {
            x: pos.x,
            y: pos.y + 1,
        },
        BomberAction::Left => GridPos {
            x: pos.x - 1,
            y: pos.y,
        },
        BomberAction::Right => GridPos {
            x: pos.x + 1,
            y: pos.y,
        },
        BomberAction::Bomb | BomberAction::Wait | BomberAction::Detonate => pos,
    }
}

/// Update known bomb list from events.
fn update_bombs(bombs: &mut Vec<KnownBomb>, events: &[GameEvent]) {
    for bomb in bombs.iter_mut() {
        bomb.2 = bomb.2.saturating_sub(1);
    }
    for event in events {
        match event {
            GameEvent::BombPlaced { pos, .. } => {
                if !bombs.iter().any(|(p, _, _)| *p == *pos) {
                    bombs.push((*pos, DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
                }
            }
            GameEvent::BombExploded { pos, .. } => {
                bombs.retain(|(p, _, _)| *p != *pos);
            }
            _ => {}
        }
    }
}

/// Update known power-up list from events.
fn update_powerups(powerups: &mut Vec<(i32, i32)>, events: &[GameEvent]) {
    for event in events {
        match event {
            GameEvent::PowerUpRevealed { pos, .. } => {
                if !powerups.contains(pos) {
                    powerups.push(*pos);
                }
            }
            GameEvent::PowerUpCollected { pos, .. } => {
                powerups.retain(|p| p != pos);
            }
            _ => {}
        }
    }
}

/// Track opponent positions from events.
fn update_opponents(opponents: &mut Vec<KnownOpponent>, events: &[GameEvent], my_id: u8) {
    for event in events {
        match event {
            GameEvent::PlayerMoved { player, to, .. } => {
                if *player == my_id {
                    continue;
                }
                if let Some(entry) = opponents.iter_mut().find(|(p, _, _)| *p == *player) {
                    entry.2 = Some(entry.1);
                    entry.1 = *to;
                } else {
                    opponents.push((*player, *to, None));
                }
            }
            GameEvent::BombPlaced { player, pos } => {
                if *player == my_id {
                    continue;
                }
                if let Some(entry) = opponents.iter_mut().find(|(p, _, _)| *p == *player) {
                    entry.2 = Some(entry.1);
                    entry.1 = *pos;
                } else {
                    opponents.push((*player, *pos, None));
                }
            }
            GameEvent::PlayerKilled { victim, .. } => {
                opponents.retain(|(p, _, _)| *p != *victim);
            }
            _ => {}
        }
    }
}

/// Compute game-domain Hint-δ: delta at argmax action (not mean).
fn compute_game_delta(
    query_scores: &[f32; ACTION_COUNT],
    hinted_scores: &[f32; ACTION_COUNT],
) -> f32 {
    let best_idx = hinted_scores
        .iter()
        .enumerate()
        .filter(|(_, s)| **s > f32::NEG_INFINITY)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i);

    match best_idx {
        Some(idx) => hinted_scores[idx] - query_scores[idx],
        None => 0.0,
    }
}

/// Compute Shannon entropy on sigmoid-normalized scores (only valid actions).
///
/// Uses per-element sigmoid instead of softmax per project rule.
/// Sigmoid scores are then normalized by their sum to form a probability
/// distribution for entropy computation.
fn shannon_entropy(scores: &[f32; ACTION_COUNT]) -> f32 {
    let sigs: Vec<f32> = scores
        .iter()
        .map(|&s| {
            if s <= f32::NEG_INFINITY {
                0.0
            } else {
                1.0 / (1.0 + (-s).exp())
            }
        })
        .collect();
    let sum: f32 = sigs.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let entropy: f32 = sigs
        .iter()
        .filter(|&&s| s > 0.0)
        .map(|&s| {
            let p = s / sum;
            -p * p.ln()
        })
        .sum();
    entropy
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

// ── Sr2amPlayer ────────────────────────────────────────────────

/// SR²AM bomber player — extends GZero with learned per-turn planning regulation.
///
/// Uses [`ConfiguratorBandit`] to decide per-tick whether to:
/// - `PlanNew`: Full template search (normal GZero behavior)
/// - `PlanExtend`: Reuse last template, re-evaluate with current state
/// - `PlanSkip`: Skip template entirely, use pure heuristic + Q-values only
pub struct Sr2amPlayer {
    _id: u8,
    // Game state tracking (same as GZero)
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<KnownOpponent>,
    last_dir: Option<BomberAction>,
    // G-Zero components (same as GZero)
    template_proposer: BomberTemplateProposer,
    delta_bandit: DeltaBanditPruner<NoScreeningPruner>,
    absorb_compress: DeltaGatedAbsorbCompress<NoScreeningPruner>,
    delta_history: Vec<f32>,
    round_actions: Vec<(BomberAction, f32)>,
    round_template_ids: Vec<usize>,
    // Cross-round Q-values (same as GZero)
    q_values: [f32; ACTION_COUNT],
    visits: [u32; ACTION_COUNT],
    // SR²AM additions
    #[cfg(not(feature = "sia_feedback"))]
    configurator: ConfiguratorBandit,
    #[cfg(feature = "sia_feedback")]
    configurator: crate::pruners::feedback_bandit::FeedbackBandit,
    last_template: Option<BomberTemplate>,
    last_template_id: Option<usize>,
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

impl Sr2amPlayer {
    /// Create a new Sr2amPlayer with the given player ID.
    pub fn new(id: u8) -> Self {
        let bandit_inner =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);
        let delta_bandit = DeltaBanditPruner::new(bandit_inner, NUM_TEMPLATES);

        let absorb_inner =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());
        let absorb_compress =
            DeltaGatedAbsorbCompress::new(absorb_inner, NUM_TEMPLATES, DeltaGatedConfig::default());

        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_dir: None,
            template_proposer: BomberTemplateProposer::new(),
            delta_bandit,
            absorb_compress,
            delta_history: Vec::new(),
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            #[cfg(not(feature = "sia_feedback"))]
            configurator: ConfiguratorBandit::new(),
            #[cfg(feature = "sia_feedback")]
            configurator: crate::pruners::feedback_bandit::FeedbackBandit::new(),
            last_template: None,
            last_template_id: None,
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

    /// Update Q-values from round outcome + feed outcome reward to template bandit.
    pub fn update_outcome(&mut self, survived: bool, killed: bool, powerups: u32) {
        if self.round_actions.is_empty() {
            return;
        }

        let outcome_reward = if survived { 1.0 } else { -1.0 }
            + if killed { 2.0 } else { 0.0 }
            + powerups as f32 * 0.5;

        // Distribute outcome reward across ALL templates used this round.
        let template_reward = if survived { 1.0 } else { -0.5 }
            + if killed { 1.0 } else { 0.0 }
            + powerups as f32 * 0.3;
        let share = if self.round_template_ids.is_empty() {
            0.0
        } else {
            template_reward / self.round_template_ids.len() as f32
        };
        for &tid in &self.round_template_ids {
            self.template_proposer.observe_outcome(tid, share);
        }

        // Update per-action Q-values with blended reward
        for (action, delta) in &self.round_actions {
            let idx = action.as_usize();
            let alpha = 1.0 / (1.0 + self.visits[idx] as f32).sqrt();
            self.q_values[idx] += alpha * (outcome_reward + delta - self.q_values[idx]);
            self.visits[idx] += 1;
        }

        self.delta_history.push(self.round_delta_mean());
        self.round_actions.clear();
        self.round_template_ids.clear();
    }

    /// Run absorb-compress cycle. Returns newly compressed arm indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        self.absorb_compress.compress()
    }

    /// Get delta summary: (mean_δ, positive_ratio, best_template).
    pub fn delta_summary(&self) -> (f32, f32, BomberTemplate) {
        let len = self.delta_history.len().max(1);
        let mean = self.delta_history.iter().sum::<f32>() / len as f32;
        let positive = self.delta_history.iter().filter(|&&d| d > 0.0).count() as f32 / len as f32;
        (mean, positive, self.template_proposer.best_template())
    }

    /// Normalized pull distribution across templates.
    pub fn template_distribution(&self) -> Vec<(BomberTemplate, f32)> {
        self.template_proposer.template_distribution()
    }

    /// Freeze cross-round bandit knowledge into a `repr(C)` struct.
    pub fn freeze(&self) -> BomberFrozenBandit {
        BomberFrozenBandit {
            magic: BomberFrozenBandit::MAGIC,
            version: BomberFrozenBandit::VERSION,
            q_values: self.q_values,
            visits: self.visits,
            total_pulls: self.visits.iter().sum(),
            compressed: [0; 7],
            reserved: [0; 16],
        }
    }

    /// Thaw an Sr2amPlayer from frozen bandit knowledge.
    pub fn thaw(frozen: &BomberFrozenBandit, id: u8) -> Result<Self, String> {
        frozen.validate()?;
        let bandit_inner =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);
        let delta_bandit = DeltaBanditPruner::new(bandit_inner, NUM_TEMPLATES);
        let absorb_inner =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());
        let absorb_compress =
            DeltaGatedAbsorbCompress::new(absorb_inner, NUM_TEMPLATES, DeltaGatedConfig::default());
        Ok(Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_dir: None,
            template_proposer: BomberTemplateProposer::new(),
            delta_bandit,
            absorb_compress,
            delta_history: Vec::new(),
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: frozen.q_values,
            visits: frozen.visits,
            #[cfg(not(feature = "sia_feedback"))]
            configurator: ConfiguratorBandit::new(),
            #[cfg(feature = "sia_feedback")]
            configurator: crate::pruners::feedback_bandit::FeedbackBandit::new(),
            last_template: None,
            last_template_id: None,
            decision_history: Vec::new(),
            plan_skip_count: 0,
            plan_new_count: 0,
            plan_extend_count: 0,
            plan_spechop_count: 0,
            #[cfg(feature = "sia_feedback")]
            plan_harness_count: 0,
            #[cfg(feature = "sia_feedback")]
            plan_weight_count: 0,
        })
    }

    /// Get planning decision distribution: (plan_new, plan_extend, plan_skip, spechop, harness, weight) counts.
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
    fn compute_query_scores(
        &self,
        grid: &ArenaGrid,
        pos: GridPos,
        bomb_positions: &[(i32, i32)],
    ) -> [f32; ACTION_COUNT] {
        let mut query_scores = [0.0f32; ACTION_COUNT];
        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            query_scores[i] = match action {
                BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right => {
                    let target = move_target(*action, pos);
                    if !grid.is_walkable(target.x, target.y) {
                        f32::NEG_INFINITY
                    } else {
                        let mut s = 1.0;
                        // Mild powerup attraction
                        if let Some(pu) = self
                            .known_powerups
                            .iter()
                            .min_by_key(|p| (target.x - p.0).abs() + (target.y - p.1).abs())
                        {
                            let dist = (target.x - pu.0).abs() + (target.y - pu.1).abs();
                            s += 0.5 / (dist as f32 + 1.0);
                        }
                        // Penalize being near bombs
                        let min_bomb_dist = bomb_positions
                            .iter()
                            .map(|b| (target.x - b.0).abs() + (target.y - b.1).abs())
                            .min()
                            .unwrap_or(999);
                        if min_bomb_dist <= 2 {
                            s -= 2.0;
                        }
                        // Mild center bias
                        let center = ARENA_W as i32 / 2;
                        let dist_after = (target.x - center).abs() + (target.y - center).abs();
                        s += 0.1 * (center as f32 - dist_after as f32) / center as f32;
                        s
                    }
                }
                BomberAction::Bomb => {
                    let wall_adj = [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)]
                        .iter()
                        .filter(|(dx, dy)| {
                            matches!(
                                grid.get(pos.x + dx, pos.y + dy),
                                Cell::DestructibleWall | Cell::PowerUpHidden(_)
                            )
                        })
                        .count();
                    if wall_adj > 0 { 1.0 } else { 0.0 }
                }
                BomberAction::Wait | BomberAction::Detonate => 0.0,
            };
        }
        query_scores
    }

    /// Apply template hint overrides to scores.
    fn apply_template_hints(
        template: BomberTemplate,
        query_scores: &[f32; ACTION_COUNT],
        pos: GridPos,
        bomb_positions: &[(i32, i32)],
        opponent_positions: &[(i32, i32)],
    ) -> [f32; ACTION_COUNT] {
        let mut hinted_scores = *query_scores;
        for i in 0..ACTION_COUNT {
            if query_scores[i] > f32::NEG_INFINITY {
                let hint = hint_score_override(
                    template,
                    i,
                    (pos.x, pos.y),
                    bomb_positions,
                    opponent_positions,
                    &[],
                    ARENA_W as i32,
                    ARENA_H as i32,
                );
                hinted_scores[i] += hint;
            }
        }
        hinted_scores
    }
}

// ── BomberPlayer Trait ─────────────────────────────────────────

impl BomberPlayer for Sr2amPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        // 1. Update game state from events
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);
        update_opponents(&mut self.known_opponents, events, self._id);

        let bomb_positions: Vec<(i32, i32)> = self.known_bombs.iter().map(|(p, _, _)| *p).collect();
        let opponent_positions: Vec<(i32, i32)> =
            self.known_opponents.iter().map(|(_, op, _)| *op).collect();

        // 2. Compute query_scores (WEAK heuristic — same as GZero)
        let query_scores = self.compute_query_scores(grid, pos, &bomb_positions);

        // 3. Compute Shannon entropy → bin context for ConfiguratorBandit
        let entropy = shannon_entropy(&query_scores);
        let entropy_bin = ConfiguratorBandit::entropy_bin(entropy);
        let context = ConfiguratorContext::new(SR2AM_DOMAIN, entropy_bin);

        // 4. Query ConfiguratorBandit → PlanningDecision
        let decision = self.configurator.select(context);

        // 5. Execute planning strategy based on decision
        let (hinted_scores, template_id) = match decision {
            PlanningDecision::PlanNew => {
                // Full template search via UCB1 (normal GZero behavior)
                let (template, tid) = self.template_proposer.select();
                self.last_template = Some(template);
                self.last_template_id = Some(tid);
                self.round_template_ids.push(tid);
                let hinted = Self::apply_template_hints(
                    template,
                    &query_scores,
                    pos,
                    &bomb_positions,
                    &opponent_positions,
                );
                (hinted, Some(tid))
            }
            PlanningDecision::PlanExtend => {
                // Reuse last_template, recompute hint with current state
                match self.last_template {
                    Some(template) => {
                        let tid = self.last_template_id.unwrap_or(0);
                        self.round_template_ids.push(tid);
                        let hinted = Self::apply_template_hints(
                            template,
                            &query_scores,
                            pos,
                            &bomb_positions,
                            &opponent_positions,
                        );
                        (hinted, Some(tid))
                    }
                    None => {
                        // No previous template — fall back to PlanNew
                        let (template, tid) = self.template_proposer.select();
                        self.last_template = Some(template);
                        self.last_template_id = Some(tid);
                        self.round_template_ids.push(tid);
                        let hinted = Self::apply_template_hints(
                            template,
                            &query_scores,
                            pos,
                            &bomb_positions,
                            &opponent_positions,
                        );
                        (hinted, Some(tid))
                    }
                }
            }
            PlanningDecision::PlanSkip => {
                // Skip template entirely — use only heuristic query_scores + Q-values
                // hinted_scores = query_scores (no template modification)
                (query_scores, None)
            }
            PlanningDecision::SpecHop { .. } => {
                // SpecHop operates at hop level (Plan 131) — skip template search here.
                // The speculator handles prediction independently; use query_scores as base.
                (query_scores, None)
            }
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::HarnessUpdate => {
                // HarnessUpdate: use current best template (like PlanExtend)
                // AbsorbCompress promote + HotSwapPruner reload handled by caller.
                match self.last_template {
                    Some(template) => {
                        let tid = self.last_template_id.unwrap_or(0);
                        self.round_template_ids.push(tid);
                        let hinted = Self::apply_template_hints(
                            template,
                            &query_scores,
                            pos,
                            &bomb_positions,
                            &opponent_positions,
                        );
                        (hinted, Some(tid))
                    }
                    None => (query_scores, None),
                }
            }
            #[cfg(feature = "sia_feedback")]
            PlanningDecision::WeightUpdate => {
                // WeightUpdate: defer to Q-values (no template search)
                // Training trigger handled by caller via WeightUpdateRequest event.
                (query_scores, None)
            }
        };

        // 6. Compute δ (game-domain Hint-δ)
        let delta_value = compute_game_delta(&query_scores, &hinted_scores);

        // Feed δ to components (only if we actually used a template)
        if let Some(tid) = template_id {
            self.template_proposer.observe_delta(tid, delta_value);
            self.delta_bandit.observe_delta(tid, delta_value);
            self.absorb_compress
                .observe_delta(tid, delta_value, delta_value.max(0.0));
        }

        // 7. Compute reward signal and update configurator bandit
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

        // 8. Blend hinted_scores with q_values (80% heuristic + 20% bandit)
        let mut final_scores = [0.0f32; ACTION_COUNT];
        for i in 0..ACTION_COUNT {
            if hinted_scores[i] <= f32::NEG_INFINITY {
                final_scores[i] = f32::NEG_INFINITY;
            } else {
                let bandit_q = if self.visits[i] > 0 {
                    self.q_values[i]
                } else {
                    0.0
                };
                final_scores[i] = hinted_scores[i] * 0.8 + bandit_q * 0.2;
            }
        }

        // 9. Safety filter — wall-aware blast zones with escape guidance
        let currently_in_blast = in_blast_zone(pos, grid, &self.known_bombs);

        for i in 0..ACTION_COUNT {
            let action = ALL_ACTIONS[i];
            match action {
                BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right => {
                    let target = move_target(action, pos);
                    if !grid.is_walkable(target.x, target.y) {
                        final_scores[i] = f32::NEG_INFINITY;
                    } else if currently_in_blast {
                        // In danger: use score_action's escape-distance BFS to guide out
                        final_scores[i] = score_action(
                            &action,
                            grid,
                            pos,
                            &self.known_bombs,
                            &self.known_powerups,
                            self.last_dir,
                        );
                    } else if in_blast_zone(target, grid, &self.known_bombs) {
                        // Safe: block moves INTO blast zone
                        final_scores[i] = f32::NEG_INFINITY;
                    }
                }
                BomberAction::Bomb => {
                    if !should_place_bomb(grid, pos, &self.known_bombs) {
                        final_scores[i] = f32::NEG_INFINITY;
                    }
                }
                BomberAction::Wait => {
                    if in_blast_zone(pos, grid, &self.known_bombs) {
                        final_scores[i] = f32::NEG_INFINITY;
                    }
                }
                BomberAction::Detonate => {
                    // Detonate is safe (player doesn't move), no-op for now
                }
            }
        }

        // 10. ε-greedy exploration (15% — diverse template discovery, only safe moves)
        let best_action = if rng.f32() < 0.15 {
            let safe: Vec<usize> = (0..ACTION_COUNT)
                .filter(|&i| {
                    if final_scores[i] <= f32::NEG_INFINITY {
                        return false;
                    }
                    let action = ALL_ACTIONS[i];
                    matches!(
                        action,
                        BomberAction::Up
                            | BomberAction::Down
                            | BomberAction::Left
                            | BomberAction::Right
                    )
                })
                .collect();
            if safe.is_empty() {
                BomberAction::Wait
            } else {
                ALL_ACTIONS[safe[rng.usize(0..safe.len())]]
            }
        } else {
            final_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .map(|(i, _)| ALL_ACTIONS[i])
                .unwrap_or(BomberAction::Wait)
        };

        // Track bomb placement (prevents walking back into own bomb)
        if best_action == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }

        // Record action + delta for outcome update
        self.round_actions.push((best_action, delta_value));
        if matches!(
            best_action,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best_action);
        }

        best_action
    }

    fn name(&self) -> &str {
        "SR²AM"
    }

    fn emoji(&self) -> &str {
        "🎯"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.known_opponents.clear();
        self.round_actions.clear();
        self.round_template_ids.clear();
        self.last_dir = None;
        // NOTE: Q-values, visits, template stats, configurator persist across rounds (bandit memory)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_grid() -> ArenaGrid {
        ArenaGrid::generate(42)
    }

    #[test]
    fn test_sr2am_player_creates() {
        let player = Sr2amPlayer::new(0);
        assert_eq!(player._id, 0);
        assert!(player.known_bombs.is_empty());
        assert!(player.known_powerups.is_empty());
        assert!(player.known_opponents.is_empty());
        assert!(player.last_dir.is_none());
        assert!(player.round_actions.is_empty());
        assert_eq!(player.q_values, [0.0; ACTION_COUNT]);
        assert_eq!(player.visits, [0; ACTION_COUNT]);
        // SR²AM additions start at zero
        let (new, extend, skip, spechop) = player.decision_stats();
        assert_eq!(new, 0);
        assert_eq!(extend, 0);
        assert_eq!(skip, 0);
        assert_eq!(spechop, 0);
    }

    #[test]
    fn test_sr2am_player_selects_action() {
        let mut player = Sr2amPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        let action = player.select_action(&grid, pos, &[], &mut rng);
        assert!(matches!(
            action,
            BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right
                | BomberAction::Bomb
                | BomberAction::Wait
                | BomberAction::Detonate
        ));
    }

    #[test]
    fn test_sr2am_player_decision_stats() {
        let mut player = Sr2amPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        // Run enough ticks to exercise all decision paths
        for _ in 0..30 {
            player.select_action(&grid, pos, &[], &mut rng);
        }

        let (new, extend, skip, spechop) = player.decision_stats();
        let total = new + extend + skip + spechop;
        assert_eq!(total, 30, "all 30 ticks should have a decision recorded");
        // UCB1 explores all arms, so we expect at least some of each type
        assert!(new > 0, "PlanNew should be selected at least once");
    }

    #[test]
    fn test_sr2am_entropy_uniform_high() {
        // Uniform scores → high entropy (all directions equally viable)
        let scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let entropy = shannon_entropy(&scores);
        // For 7 equal-probability outcomes: ln(7) ≈ 1.946
        assert!(
            entropy > 1.5,
            "uniform scores should have high entropy, got {entropy}"
        );
    }

    #[test]
    fn test_sr2am_entropy_single_dominant_lower_than_uniform() {
        // With sigmoid normalization, a single dominant score produces lower
        // entropy than uniform scores, but not near-zero (sigmoid doesn't
        // flatten differences as aggressively as softmax).
        let uniform = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let dominant = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let entropy_uniform = shannon_entropy(&uniform);
        let entropy_dominant = shannon_entropy(&dominant);
        assert!(
            entropy_dominant < entropy_uniform,
            "dominant score entropy ({entropy_dominant}) should be < uniform ({entropy_uniform})"
        );
    }

    #[test]
    fn test_sr2am_entropy_with_neg_inf() {
        // Some actions are -inf (unwalkable) — entropy only on valid actions
        let scores = [
            1.0,
            f32::NEG_INFINITY,
            1.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
        ];
        let entropy = shannon_entropy(&scores);
        // 5 valid actions with moderate spread
        assert!(
            entropy > 0.5,
            "should have moderate entropy from valid actions, got {entropy}"
        );
    }

    #[test]
    fn test_sr2am_entropy_all_neg_inf() {
        // All -inf → entropy is 0
        let scores = [
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let entropy = shannon_entropy(&scores);
        assert!(
            (entropy - 0.0).abs() < 0.001,
            "all -inf should have zero entropy, got {entropy}"
        );
    }

    #[test]
    fn test_sr2am_player_trait_methods() {
        let player = Sr2amPlayer::new(1);
        assert_eq!(player.name(), "SR²AM");
        assert_eq!(player.emoji(), "🎯");
    }

    #[test]
    fn test_sr2am_player_reset() {
        let mut player = Sr2amPlayer::new(0);
        player.known_bombs.push(((5, 5), 2, 4));
        player.known_powerups.push((3, 3));
        player.round_actions.push((BomberAction::Up, 0.5));
        player.last_dir = Some(BomberAction::Up);

        player.reset();

        assert!(player.known_bombs.is_empty());
        assert!(player.known_powerups.is_empty());
        assert!(player.known_opponents.is_empty());
        assert!(player.round_actions.is_empty());
        assert!(player.last_dir.is_none());
        // Q-values persist
        assert_eq!(player.q_values, [0.0; ACTION_COUNT]);
        // Decision stats persist across reset
        let (new, _extend, _skip, _spechop) = player.decision_stats();
        assert_eq!(new, 0);
    }

    #[test]
    fn test_sr2am_player_update_outcome() {
        let mut player = Sr2amPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        player.select_action(&grid, pos, &[], &mut rng);
        assert!(!player.round_actions.is_empty());

        player.update_outcome(true, false, 1);
        assert!(player.round_actions.is_empty());
        assert!(!player.delta_history.is_empty());
    }

    #[test]
    fn test_sr2am_player_update_outcome_empty() {
        let mut player = Sr2amPlayer::new(0);
        player.update_outcome(true, false, 0);
        assert!(player.delta_history.is_empty());
    }

    #[test]
    fn test_planning_cost() {
        assert_eq!(planning_cost(PlanningDecision::PlanNew), 1.0);
        assert_eq!(planning_cost(PlanningDecision::PlanExtend), 0.3);
        assert_eq!(planning_cost(PlanningDecision::PlanSkip), 0.0);
        assert!((planning_cost(PlanningDecision::SpecHop { k: 4 }) - 0.4).abs() < f32::EPSILON);
        assert!((planning_cost(PlanningDecision::SpecHop { k: 8 }) - 0.8).abs() < f32::EPSILON);
        #[cfg(feature = "sia_feedback")]
        {
            assert!((planning_cost(PlanningDecision::HarnessUpdate) - 0.5).abs() < f32::EPSILON);
            assert!((planning_cost(PlanningDecision::WeightUpdate) - 2.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_compute_game_delta() {
        let query = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let hinted = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let delta = compute_game_delta(&query, &hinted);
        assert!((delta - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_update_bombs() {
        let mut bombs: Vec<KnownBomb> = vec![((5, 5), 2, 2)];
        let events = vec![GameEvent::BombPlaced {
            player: 0,
            pos: (3, 3),
        }];
        update_bombs(&mut bombs, &events);
        assert_eq!(bombs.len(), 2);
        assert_eq!(bombs[0].2, 1);
    }

    #[test]
    fn test_update_powerups() {
        let mut powerups: Vec<(i32, i32)> = Vec::new();
        let events = vec![GameEvent::PowerUpRevealed {
            pos: (3, 3),
            kind: super::super::PowerUpKind::BombUp,
        }];
        update_powerups(&mut powerups, &events);
        assert_eq!(powerups.len(), 1);
        assert_eq!(powerups[0], (3, 3));
    }

    #[test]
    fn test_update_opponents() {
        let mut opponents: Vec<KnownOpponent> = Vec::new();
        let events = vec![GameEvent::PlayerMoved {
            player: 1,
            from: (3, 3),
            to: (4, 3),
        }];
        update_opponents(&mut opponents, &events, 0);
        assert_eq!(opponents.len(), 1);
        assert_eq!(opponents[0].0, 1);
        assert_eq!(opponents[0].1, (4, 3));
    }

    #[test]
    fn test_update_opponents_ignores_self() {
        let mut opponents: Vec<KnownOpponent> = Vec::new();
        let events = vec![GameEvent::PlayerMoved {
            player: 0,
            from: (1, 1),
            to: (2, 1),
        }];
        update_opponents(&mut opponents, &events, 0);
        assert!(opponents.is_empty());
    }
}
