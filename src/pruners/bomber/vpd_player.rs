//! VPD EM-style bomber player — co-evolutionary teacher-student distillation.
//!
//! Extends [`SdarPlayer`] with an explicit EM cycle:
//! - **M-step** (every round): KL-gated distillation of teacher → student
//! - **E-step** (every F=5 rounds): BCO unpaired preference refinement of teacher
//!
//! # Architecture
//!
//! ```text
//! VpdPlayer
//!   ├── BomberTemplateProposer     (UCB1 template selection — same as SDAR)
//!   ├── VpdEmCycle                 (E-step BCO + M-step KL-gated absorb)
//!   │   ├── BcoOptimizer           (unpaired preference teacher refinement)
//!   │   └── SdarGatedAbsorbCompress (KL-gated distillation for student)
//!   └── Cross-round Q-values       (action-level bandit memory)
//! ```
//!
//! # Key Difference from SdarPlayer
//!
//! SdarPlayer treats the teacher signal as passive (sigmoid gate only).
//! VpdPlayer actively trains the teacher via BCO every F rounds, then distills
//! back to the student. This breaks the SDAR plateau identified in VPD paper.
//!
//! # Plan 120
//!
//! Tests hypothesis: VPD EM co-evolution ≥ passive SDAR gating in bomber arena.

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
use crate::pruners::g_zero::{BomberTemplate, BomberTemplateProposer, hint_score_override};
use crate::pruners::sdar::{SdarAbsorbConfig, SdarGatedAbsorbCompress};
use crate::pruners::vpd_em::{VpdConfig, VpdEmCycle};
use crate::speculative::types::NoScreeningPruner;

use super::players::BomberPlayer;
use super::players::{in_blast_zone, score_action, should_place_bomb};
use super::{
    ARENA_H, ARENA_W, ArenaGrid, BOMB_FUSE_TICKS, BomberAction, Cell, DEFAULT_BLAST_RANGE,
    GameEvent, GridPos,
};

// ── Constants ──────────────────────────────────────────────────

const ACTION_COUNT: usize = 7;
const NUM_TEMPLATES: usize = 8;

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

// ── Helpers (same as SdarPlayer) ──────────────────────────────

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

// ── VPD Scalar Reward ─────────────────────────────────────────

/// Compute scalar reward from game state (same weights as SdarPlayer).
///
/// | Component | Weight |
/// |-----------|--------|
/// | Survival  | 0.50   |
/// | Safety    | 0.35   |
/// | Completeness | 0.15 |
#[cfg(test)]
fn compute_vpd_reward(alive: bool, danger: f32, powerups_collected: u32) -> f32 {
    let survival = if alive { 1.0 } else { 0.0 };
    let safety = 1.0 - danger.clamp(0.0, 1.0);
    let completeness = (powerups_collected as f32 / 3.0).min(1.0);
    survival * 0.5 + safety * 0.35 + completeness * 0.15
}

/// Compute danger level: fraction of adjacent cells (including current) in blast zone.
#[cfg(test)]
fn danger_level(pos: GridPos, grid: &ArenaGrid, bombs: &[KnownBomb]) -> f32 {
    let directions: [(i32, i32); 5] = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)];
    let total = directions.len() as f32;
    let in_blast = directions
        .iter()
        .filter(|(dx, dy)| {
            in_blast_zone(
                GridPos {
                    x: pos.x + dx,
                    y: pos.y + dy,
                },
                grid,
                bombs,
            )
        })
        .count() as f32;
    in_blast / total
}

// ── VpdPlayer ──────────────────────────────────────────────────

/// Bomber arena player using VPD EM-style co-evolutionary distillation.
///
/// Extends [`SdarPlayer`] architecture with an explicit EM cycle:
/// - M-step every round: KL-gated distillation via [`SdarGatedAbsorbCompress`]
/// - E-step every F=5 rounds: BCO teacher refinement via [`VpdEmCycle`]
pub struct VpdPlayer {
    _id: u8,
    // Game state tracking
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<KnownOpponent>,
    last_dir: Option<BomberAction>,
    alive: bool,
    powerups_collected: u32,
    // G-Zero components (template proposer shared with SDAR)
    template_proposer: BomberTemplateProposer,
    // VPD EM cycle (replaces sdar_bandit + sdar_absorb with unified E/M loop)
    em_cycle: VpdEmCycle<NoScreeningPruner>,
    // Absorb-compress layer (owned by player, passed to em_cycle.m_step by ref)
    absorb: SdarGatedAbsorbCompress<NoScreeningPruner>,
    // Cross-round memory
    round_actions: Vec<(BomberAction, f32)>,
    round_template_ids: Vec<usize>,
    q_values: [f32; ACTION_COUNT],
    visits: [u32; ACTION_COUNT],
    round_count: usize,
}

impl VpdPlayer {
    /// Create a new VpdPlayer with the given player ID.
    pub fn new(id: u8) -> Self {
        let absorb_inner =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());
        let absorb =
            SdarGatedAbsorbCompress::new(absorb_inner, NUM_TEMPLATES, SdarAbsorbConfig::default());

        let em_cycle = VpdEmCycle::new(VpdConfig::default(), NUM_TEMPLATES);

        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_dir: None,
            alive: true,
            powerups_collected: 0,
            template_proposer: BomberTemplateProposer::new(),
            em_cycle,
            absorb,
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            round_count: 0,
        }
    }

    /// Create a new VpdPlayer with custom VPD config.
    pub fn with_config(id: u8, config: VpdConfig) -> Self {
        let absorb_inner =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());
        let absorb =
            SdarGatedAbsorbCompress::new(absorb_inner, NUM_TEMPLATES, SdarAbsorbConfig::default());

        let em_cycle = VpdEmCycle::new(config, NUM_TEMPLATES);

        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_dir: None,
            alive: true,
            powerups_collected: 0,
            template_proposer: BomberTemplateProposer::new(),
            em_cycle,
            absorb,
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            round_count: 0,
        }
    }

    /// Update Q-values from round outcome + feed outcome to EM cycle.
    ///
    /// Computes scalar outcome reward and feeds to the EM M-step for each
    /// template used. Runs E-step if due (every F rounds).
    pub fn update_outcome(&mut self, survived: bool, killed: bool, powerups: u32) {
        if self.round_actions.is_empty() {
            return;
        }

        self.round_count += 1;

        // Outcome scalar reward (same formula as SdarPlayer)
        let outcome_reward = if survived { 1.0 } else { -1.0 }
            + if killed { 2.0 } else { 0.0 }
            + powerups as f32 * 0.5;

        // Feed outcome to EM cycle for each template used
        for &tid in &self.round_template_ids {
            // M-step: KL-gated distillation
            let should_e = self.em_cycle.m_step(tid, outcome_reward, &mut self.absorb);

            // Collect sample for next E-step
            let outcome_binary = if survived { 1.0 } else { 0.0 };
            self.em_cycle
                .collect_sample(tid, outcome_binary, outcome_reward);

            // E-step: BCO teacher refinement (every F M-steps)
            if should_e {
                let _loss = self.em_cycle.e_step();
                log::debug!(
                    "VPD E-step at round {}, loss={:.4}, shift={:.4}",
                    self.round_count,
                    _loss,
                    self.em_cycle.reward_shift()
                );
            }
        }

        // Template proposer outcome reward (same as SdarPlayer)
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

        // Update per-action Q-values with blended reward (same as SdarPlayer)
        for (action, delta) in &self.round_actions {
            let idx = action.as_usize();
            let alpha = 1.0 / (1.0 + self.visits[idx] as f32).sqrt();
            self.q_values[idx] += alpha * (outcome_reward + delta - self.q_values[idx]);
            self.visits[idx] += 1;
        }

        self.round_actions.clear();
        self.round_template_ids.clear();
    }

    /// Run absorb-compress cycle. Returns newly compressed arm indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        self.absorb.compress()
    }

    /// Get VPD summary: (m_step_count, e_step_count_est, reward_shift).
    pub fn vpd_summary(&self) -> (usize, usize, f32, BomberTemplate) {
        let m_steps = self.em_cycle.m_step_count();
        let e_steps = if self.em_cycle.config().e_step_frequency > 0 {
            m_steps / self.em_cycle.config().e_step_frequency
        } else {
            0
        };
        (
            m_steps,
            e_steps,
            self.em_cycle.reward_shift(),
            self.template_proposer.best_template(),
        )
    }

    /// Normalized pull distribution across templates.
    pub fn template_distribution(&self) -> Vec<(BomberTemplate, f32)> {
        self.template_proposer.template_distribution()
    }

    /// Select template guided by EM cycle Q-values when available.
    ///
    /// When the EM cycle has completed at least one E-step (m_step_count >= e_step_frequency),
    /// we blend UCB1 exploration with the EM-learned `student_q` to select better templates.
    /// Otherwise, fall back to pure UCB1 (same as SDAR).
    ///
    /// This is the key VPD advantage: the EM cycle actively learns which templates
    /// produce better outcomes, then biases selection toward them.
    fn select_template_guided(&mut self, rng: &mut Rng) -> (BomberTemplate, usize) {
        let n_templates = NUM_TEMPLATES;
        let m_steps = self.em_cycle.m_step_count();
        let min_steps_for_em = self.em_cycle.config().e_step_frequency;

        // Not enough data for EM guidance — fall back to UCB1
        if m_steps < min_steps_for_em {
            return self.template_proposer.select();
        }

        // EM cycle has learned: blend UCB1 score with EM student_q
        let student_q = self.em_cycle.student_q();
        let q_max = student_q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let q_min = student_q.iter().copied().fold(f32::INFINITY, f32::min);
        let q_range = q_max - q_min;

        // 10% chance to explore purely via UCB1 (avoid premature convergence)
        if rng.f32() < 0.10 {
            return self.template_proposer.select();
        }

        // Score each template: blend UCB1 with normalized EM Q-value
        let total_pulls = self.template_proposer.total_pulls();
        let mut best_id = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (tid, sq) in student_q.iter().enumerate().take(n_templates) {
            let ucb1 = self.template_proposer.ucb1_score(tid, total_pulls);
            let q_norm = if q_range.abs() > 1e-6 {
                (sq - q_min) / q_range
            } else {
                0.5
            };
            // 40% UCB1 exploration + 60% EM learned quality
            let blended = ucb1 * 0.4 + q_norm * 0.6;

            if blended > best_score {
                best_score = blended;
                best_id = tid;
            }
        }

        self.template_proposer.record_pull(best_id);
        (BomberTemplate::all()[best_id], best_id)
    }

    /// Compute hint weight based on EM cycle's learned Q-value for a template.
    ///
    /// Templates with high `student_q` get stronger hints (more confident guidance).
    /// Templates with low `student_q` get weaker hints (less confident, more exploration).
    fn em_hint_weight(&self, template_id: usize) -> f32 {
        let student_q = self.em_cycle.student_q();
        let m_steps = self.em_cycle.m_step_count();
        let min_steps = self.em_cycle.config().e_step_frequency;

        // No EM learning yet — neutral weight
        if m_steps < min_steps {
            return 1.0;
        }

        let q = student_q.get(template_id).copied().unwrap_or(0.0);
        let q_max = student_q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let q_min = student_q.iter().copied().fold(f32::INFINITY, f32::min);
        let q_range = q_max - q_min;

        // Normalized Q in [0, 1], then map to [0.5, 1.5] hint weight
        let q_norm = if q_range.abs() > 1e-6 {
            (q - q_min) / q_range
        } else {
            0.5
        };

        0.5 + q_norm
    }
}

// ── BomberPlayer Implementation ────────────────────────────────

impl BomberPlayer for VpdPlayer {
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

        // Track alive state and powerup collection
        for event in events {
            match event {
                GameEvent::PlayerKilled { victim, .. } => {
                    if *victim == self._id {
                        self.alive = false;
                    }
                }
                GameEvent::PowerUpCollected { player, .. } => {
                    if *player == self._id {
                        self.powerups_collected += 1;
                    }
                }
                _ => {}
            }
        }

        let bomb_positions: Vec<(i32, i32)> = self.known_bombs.iter().map(|(p, _, _)| *p).collect();
        let opponent_positions: Vec<(i32, i32)> =
            self.known_opponents.iter().map(|(_, op, _)| *op).collect();

        // 2. Compute query_scores (WEAK heuristic — same as SDAR)
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
                        if let Some(pu) = self
                            .known_powerups
                            .iter()
                            .min_by_key(|p| (target.x - p.0).abs() + (target.y - p.1).abs())
                        {
                            let dist = (target.x - pu.0).abs() + (target.y - pu.1).abs();
                            s += 0.5 / (dist as f32 + 1.0);
                        }
                        let min_bomb_dist = bomb_positions
                            .iter()
                            .map(|b| (target.x - b.0).abs() + (target.y - b.1).abs())
                            .min()
                            .unwrap_or(999);
                        if min_bomb_dist <= 2 {
                            s -= 2.0;
                        }
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

        // 3. Select template — EM-guided when cycle has learned, UCB1 fallback
        let (template, template_id) = self.select_template_guided(rng);
        self.round_template_ids.push(template_id);

        // 4. Compute hinted_scores = query_scores + EM-weighted hint
        let em_weight = self.em_hint_weight(template_id);
        let mut hinted_scores = query_scores;
        for i in 0..ACTION_COUNT {
            if query_scores[i] > f32::NEG_INFINITY {
                let hint = hint_score_override(
                    template,
                    i,
                    (pos.x, pos.y),
                    &bomb_positions,
                    &opponent_positions,
                    &self.known_powerups,
                    ARENA_W as i32,
                    ARENA_H as i32,
                );
                // EM-weighted: high-Q templates get stronger hints
                hinted_scores[i] += hint * em_weight;
            }
        }

        // 5. Compute scalar δ for Q-value blend (compatibility)
        let delta_value = compute_game_delta(&query_scores, &hinted_scores);

        // 6. Feed template delta to template proposer for UCB1 exploration
        self.template_proposer
            .observe_delta(template_id, delta_value);

        // 7. Blend hinted_scores with Q-values
        // When EM has learned, shift weight from heuristic toward bandit Q
        let em_learned = self.em_cycle.m_step_count() > 0;
        let heuristic_w = if em_learned { 0.7 } else { 0.8 };
        let bandit_w = if em_learned { 0.3 } else { 0.2 };

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
                final_scores[i] = hinted_scores[i] * heuristic_w + bandit_q * bandit_w;
            }
        }

        // 8. Safety filter — wall-aware blast zones with escape guidance
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
                        final_scores[i] = score_action(
                            &action,
                            grid,
                            pos,
                            &self.known_bombs,
                            &self.known_powerups,
                            self.last_dir,
                        );
                    } else if in_blast_zone(target, grid, &self.known_bombs) {
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
                BomberAction::Detonate => {}
            }
        }

        // 9. ε-greedy exploration (15%)
        let best_action = if rng.f32() < 0.15 {
            let safe: Vec<usize> = (0..ACTION_COUNT)
                .filter(|&i| {
                    if final_scores[i] <= f32::NEG_INFINITY {
                        return false;
                    }
                    matches!(
                        ALL_ACTIONS[i],
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

        // Track bomb placement
        if best_action == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }

        // 10. Record (action, δ) for outcome update
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
        "VPD"
    }

    fn emoji(&self) -> &str {
        "🧬"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.known_opponents.clear();
        self.round_actions.clear();
        self.round_template_ids.clear();
        self.last_dir = None;
        self.alive = true;
        self.powerups_collected = 0;
        // NOTE: Q-values, visits, template stats, EM cycle memory persist across rounds
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
    fn test_compute_vpd_reward_alive_safe() {
        let reward = compute_vpd_reward(true, 0.0, 0);
        assert!(
            (reward - 0.85).abs() < 1e-5,
            "alive + safe = 0.5 + 0.35 + 0.0 = 0.85, got {reward}"
        );
    }

    #[test]
    fn test_compute_vpd_reward_dead() {
        let reward = compute_vpd_reward(false, 0.0, 0);
        assert!(
            (reward - 0.35).abs() < 1e-5,
            "dead + safe = 0.0 + 0.35 + 0.0 = 0.35, got {reward}"
        );
    }

    #[test]
    fn test_compute_vpd_reward_in_danger() {
        let reward = compute_vpd_reward(true, 0.8, 0);
        let expected = 0.5 + (1.0 - 0.8) * 0.35;
        assert!(
            (reward - expected).abs() < 1e-5,
            "alive + danger=0.8 = {expected}, got {reward}"
        );
    }

    #[test]
    fn test_compute_vpd_reward_all_zero() {
        let reward = compute_vpd_reward(false, 1.0, 0);
        let expected = 0.0 + 0.0 + 0.0;
        assert!(
            (reward - expected).abs() < 1e-5,
            "dead + max danger = {expected}, got {reward}"
        );
    }

    #[test]
    fn test_new_player_initial_state() {
        let player = VpdPlayer::new(0);
        assert_eq!(player._id, 0);
        assert!(player.known_bombs.is_empty());
        assert!(player.known_powerups.is_empty());
        assert!(player.known_opponents.is_empty());
        assert!(player.alive);
        assert_eq!(player.powerups_collected, 0);
        assert_eq!(player.round_count, 0);
        assert!(player.round_actions.is_empty());
        assert!(player.round_template_ids.is_empty());
        assert!(player.q_values.iter().all(|&q| q == 0.0));
        assert!(player.visits.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_select_action_returns_valid() {
        let mut player = VpdPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        let action = player.select_action(&grid, pos, &[], &mut rng);
        assert!(
            ALL_ACTIONS.contains(&action),
            "action {action:?} should be valid"
        );
    }

    #[test]
    fn test_update_outcome_updates_q_values() {
        let mut player = VpdPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        // Select an action to populate round_actions
        let _action = player.select_action(&grid, pos, &[], &mut rng);

        player.update_outcome(true, false, 1);

        // At least one Q-value should have changed from outcome
        let any_nonzero = player.q_values.iter().any(|&q| q != 0.0);
        assert!(
            any_nonzero,
            "At least one Q-value should update after outcome"
        );
        assert_eq!(player.round_count, 1);
    }

    #[test]
    fn test_reset_clears_round() {
        let mut player = VpdPlayer::new(0);
        player.known_bombs.push(((1, 1), 2, 3));
        player.known_powerups.push((2, 2));
        player.round_actions.push((BomberAction::Up, 0.5));
        player.round_template_ids.push(0);
        player.alive = false;
        player.powerups_collected = 2;

        player.reset();

        assert!(player.known_bombs.is_empty());
        assert!(player.known_powerups.is_empty());
        assert!(player.round_actions.is_empty());
        assert!(player.round_template_ids.is_empty());
        assert!(player.alive);
        assert_eq!(player.powerups_collected, 0);
        // Q-values persist
        assert!(player.q_values.iter().all(|&q| q == 0.0));
    }

    #[test]
    fn test_name_and_emoji() {
        let player = VpdPlayer::new(0);
        assert_eq!(player.name(), "VPD");
        assert_eq!(player.emoji(), "🧬");
    }

    #[test]
    fn test_vpd_summary() {
        let player = VpdPlayer::new(0);
        let (m_steps, _e_steps, shift, _template) = player.vpd_summary();
        assert_eq!(m_steps, 0);
        assert!(
            (shift - 0.0).abs() < 1e-5,
            "initial reward shift = 0, got {shift}"
        );
    }
}
