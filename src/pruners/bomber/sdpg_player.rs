//! SDPG Bandit bomber player — oracle-informed self-distilled policy gradient.
//!
//! Same architecture as [`SdarPlayer`] but uses [`SdpgBanditPruner`] instead of
//! [`SdarBanditPruner`]. SDPG's centered log-ratio advantage provides dense
//! per-arm credit assignment informed by oracle teacher Q-values.
//!
//! # Architecture
//!
//! ```text
//! SdpgPlayer
//!   ├── BomberTemplateProposer     (UCB1 template selection — same as GZero/Rubric/SDAR)
//!   ├── SdpgBanditPruner           (oracle-informed CLR advantage bandit)
//!   ├── AbsorbCompressLayer        (plain absorb-compress, no SDAR gating)
//!   └── Cross-round Q-values       (action-level bandit memory)
//! ```
//!
//! # Key Difference from SdarPlayer
//!
//! - SdarPlayer uses `SdarBanditPruner` (sigmoid-gated reward)
//! - SdpgPlayer uses `SdpgBanditPruner` (oracle-informed centered log-ratio advantage)
//! - Arena outcome gates positive-advantage signal (only on wins)
//! - Teacher Q-values start uniform (no oracle data in tournament mode)
//!
//! # Plan 180

#![cfg(feature = "sdpg_bandit")]

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
use crate::pruners::bandit::{BanditPruner, BanditStrategy};
use crate::pruners::g_zero::{BomberTemplate, BomberTemplateProposer, hint_score_override};
use crate::pruners::sdpg::AdvantageMode;
use crate::pruners::sdpg::SdpgBanditPruner;
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

// ── Helpers (same as GZeroPlayer/RubricPlayer/SdarPlayer) ─────

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

// ── SDPG Scalar Reward ─────────────────────────────────────────

/// Compute scalar reward from game state.
///
/// Same weights as `RubricTemplate::bomber()` and `compute_sdar_reward`:
///
/// | Component | Weight |
/// |-----------|--------|
/// | Survival  | 0.50   |
/// | Safety    | 0.35   |
/// | Completeness | 0.15 |
fn compute_sdpg_reward(alive: bool, danger: f32, powerups_collected: u32) -> f32 {
    let survival = if alive { 1.0 } else { 0.0 };
    let safety = 1.0 - danger.clamp(0.0, 1.0);
    let completeness = (powerups_collected as f32 / 3.0).min(1.0);
    survival * 0.5 + safety * 0.35 + completeness * 0.15
}

/// Compute danger level: fraction of adjacent cells (including current) in blast zone.
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

// ── SdpgPlayer ─────────────────────────────────────────────────

/// Bomber arena player using SDPG oracle-informed policy gradient.
///
/// Replaces SdarPlayer's sigmoid-gated components with SDPG's centered
/// log-ratio advantage. Uses [`SdpgBanditPruner`] for oracle-informed reward
/// updates and plain [`AbsorbCompressLayer`] (no SDAR gating needed).
///
/// Arena outcome (win/loss) gates positive-advantage signal — only receives
/// teacher signal on wins.
pub struct SdpgPlayer {
    _id: u8,
    // Game state tracking
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<KnownOpponent>,
    last_dir: Option<BomberAction>,
    alive: bool,
    powerups_collected: u32,
    // G-Zero components (template proposer shared with GZero/Rubric/SDAR)
    template_proposer: BomberTemplateProposer,
    // SDPG components (replace SDAR bandit and SDAR absorb)
    sdpg_bandit: SdpgBanditPruner<NoScreeningPruner>,
    absorb: AbsorbCompressLayer<NoScreeningPruner>,
    // Cross-round memory
    round_actions: Vec<(BomberAction, f32)>,
    round_template_ids: Vec<usize>,
    q_values: [f32; ACTION_COUNT],
    visits: [u32; ACTION_COUNT],
    // Arena outcome tracking for SDPG positive-advantage gating
    last_arena_outcome: Option<f32>,
}

impl SdpgPlayer {
    /// Create a new SdpgPlayer with the given player ID.
    pub fn new(id: u8) -> Self {
        let bandit_inner =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);
        // Teacher Q-values: uniform (no oracle data in tournament mode)
        let teacher_q = vec![1.0; NUM_TEMPLATES];
        let sdpg_bandit = SdpgBanditPruner::with_defaults(bandit_inner, teacher_q);

        let absorb =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());

        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_dir: None,
            alive: true,
            powerups_collected: 0,
            template_proposer: BomberTemplateProposer::new(),
            sdpg_bandit,
            absorb,
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            last_arena_outcome: None,
        }
    }

    /// Create SdpgPlayer with oracle teacher Q-values from replay data.
    pub fn with_replay(id: u8, replay_path: &std::path::Path) -> std::io::Result<Self> {
        let bandit_inner =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);

        use crate::pruners::sdpg::{BetaSchedule, KlAnchor};
        let sdpg_bandit = SdpgBanditPruner::from_replay(
            bandit_inner,
            replay_path,
            BetaSchedule::default_schedule(),
            KlAnchor::default_urkl(),
            1.0,
            AdvantageMode::Sigmoid,
        )?;

        let absorb =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());

        Ok(Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_dir: None,
            alive: true,
            powerups_collected: 0,
            template_proposer: BomberTemplateProposer::new(),
            sdpg_bandit,
            absorb,
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            last_arena_outcome: None,
        })
    }

    /// Create SdpgPlayer with pre-built oracle teacher Q-values.
    ///
    /// Use this when teacher Q-values are known from prior burn-in or replay analysis.
    pub fn with_teacher_q(id: u8, teacher_q: Vec<f32>) -> Self {
        assert_eq!(
            teacher_q.len(),
            NUM_TEMPLATES,
            "teacher_q length must match NUM_TEMPLATES ({NUM_TEMPLATES})"
        );
        let bandit_inner =
            BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_TEMPLATES);
        let sdpg_bandit = SdpgBanditPruner::with_defaults(bandit_inner, teacher_q);

        let absorb =
            AbsorbCompressLayer::new(NoScreeningPruner, NUM_TEMPLATES, CompressConfig::default());

        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_dir: None,
            alive: true,
            powerups_collected: 0,
            template_proposer: BomberTemplateProposer::new(),
            sdpg_bandit,
            absorb,
            round_actions: Vec::new(),
            round_template_ids: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            last_arena_outcome: None,
        }
    }

    /// Get a reference to the inner SDPG bandit pruner.
    ///
    /// Use this to extract learned Q-values after burn-in.
    pub fn sdpg_bandit(&self) -> &SdpgBanditPruner<NoScreeningPruner> {
        &self.sdpg_bandit
    }

    /// Update Q-values from round outcome + feed outcome reward to SDPG bandit.
    ///
    /// Computes scalar outcome reward from survival/powerup stats and feeds to
    /// SDPG bandit with arena outcome for positive-advantage gating.
    /// Also distributes template rewards for UCB1.
    pub fn update_outcome(&mut self, survived: bool, killed: bool, powerups: u32) {
        if self.round_actions.is_empty() {
            return;
        }

        // Outcome scalar reward (same formula as GZero/Rubric/SDAR)
        let outcome_reward = if survived { 1.0 } else { -1.0 }
            + if killed { 2.0 } else { 0.0 }
            + powerups as f32 * 0.5;

        // Arena outcome for SDPG positive-advantage gating
        self.last_arena_outcome = Some(outcome_reward);

        // Feed outcome reward to SDPG components for each template used
        for &tid in &self.round_template_ids {
            self.sdpg_bandit
                .update(tid, outcome_reward, Some(outcome_reward));
            self.absorb.absorb(tid, outcome_reward);
        }

        // Template proposer outcome reward (same formula as GZero/Rubric/SDAR)
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

        self.round_actions.clear();
        self.round_template_ids.clear();
    }

    /// Run absorb-compress cycle. Returns newly compressed arm indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        self.absorb.compress()
    }

    /// Get SDPG summary: (mean_delta, beta, best_template).
    pub fn sdpg_summary(&self) -> (f32, f32, BomberTemplate) {
        let deltas = self
            .round_actions
            .iter()
            .map(|(_, d)| *d)
            .collect::<Vec<_>>();
        let mean_delta = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f32>() / deltas.len() as f32
        };
        (
            mean_delta,
            self.sdpg_bandit.beta(),
            self.template_proposer.best_template(),
        )
    }

    /// Normalized pull distribution across templates.
    pub fn template_distribution(&self) -> Vec<(BomberTemplate, f32)> {
        self.template_proposer.template_distribution()
    }
}

// ── BomberPlayer Implementation ────────────────────────────────

impl BomberPlayer for SdpgPlayer {
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

        // 2. Compute query_scores (WEAK heuristic — same as GZero/Rubric/SDAR)
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

        // 3. Select template via UCB1 — track for outcome attribution
        let (template, template_id) = self.template_proposer.select();
        self.round_template_ids.push(template_id);

        // 4. Compute hinted_scores = query_scores + hint_score_override
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
                hinted_scores[i] += hint;
            }
        }

        // 5. Compute scalar δ for Q-value blend (compatibility)
        let delta_value = compute_game_delta(&query_scores, &hinted_scores);

        // 6. Compute scalar reward from current game state
        let danger = danger_level(pos, grid, &self.known_bombs);
        let reward = compute_sdpg_reward(self.alive, danger, self.powerups_collected);

        // 7. Feed scalar reward to SDPG components (with arena outcome for positive-advantage gating)
        self.sdpg_bandit
            .update(template_id, reward, self.last_arena_outcome);
        self.absorb.absorb(template_id, reward);

        // Also feed scalar δ to template proposer for UCB1 exploration
        self.template_proposer
            .observe_delta(template_id, delta_value);

        // 8. Blend hinted_scores with Q-values (80% heuristic + 20% bandit)
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

        // 10. ε-greedy exploration (15%)
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

        // 11. Record (action, δ) for outcome update
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
        "SDPG"
    }

    fn emoji(&self) -> &str {
        "🎓"
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
        self.last_arena_outcome = None;
        // NOTE: Q-values, visits, template stats, SDPG memory persist across rounds
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
    fn test_compute_sdpg_reward_alive_safe() {
        let reward = compute_sdpg_reward(true, 0.0, 0);
        let expected = 1.0 * 0.5 + 1.0 * 0.35 + 0.0 * 0.15;
        assert!(
            (reward - expected).abs() < 1e-6,
            "Alive + safe + no powerups: expected {expected}, got {reward}"
        );
    }

    #[test]
    fn test_compute_sdpg_reward_dead() {
        let reward = compute_sdpg_reward(false, 0.5, 2);
        let expected = 0.0 * 0.5 + 0.5 * 0.35 + (2.0_f32 / 3.0).min(1.0) * 0.15;
        assert!(
            (reward - expected).abs() < 1e-6,
            "Dead + danger 0.5 + 2 powerups: expected {expected}, got {reward}"
        );
    }

    #[test]
    fn test_compute_sdpg_reward_in_danger() {
        let reward = compute_sdpg_reward(true, 0.6, 5);
        let expected = 1.0 * 0.5 + 0.4 * 0.35 + 1.0 * 0.15;
        assert!(
            (reward - expected).abs() < 1e-6,
            "Alive + danger 0.6 + 5 powerups: expected {expected}, got {reward}"
        );
    }

    #[test]
    fn test_compute_sdpg_reward_all_zero() {
        let reward = compute_sdpg_reward(false, 1.0, 0);
        let expected = 0.0 * 0.5 + 0.0 * 0.35 + 0.0 * 0.15;
        assert!(
            (reward - expected).abs() < 1e-6,
            "Dead + max danger + no powerups: expected {expected}, got {reward}"
        );
    }

    #[test]
    fn test_new_player_initial_state() {
        let player = SdpgPlayer::new(0);
        assert_eq!(player._id, 0);
        assert!(player.known_bombs.is_empty());
        assert!(player.known_powerups.is_empty());
        assert!(player.known_opponents.is_empty());
        assert!(player.alive);
        assert_eq!(player.powerups_collected, 0);
        assert!(player.round_actions.is_empty());
        assert!(player.round_template_ids.is_empty());
        assert!(player.q_values.iter().all(|&q| q == 0.0));
        assert!(player.visits.iter().all(|&v| v == 0));
        assert!(player.last_arena_outcome.is_none());
    }

    #[test]
    fn test_select_action_returns_valid() {
        let mut player = SdpgPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        let action = player.select_action(&grid, pos, &[], &mut rng);

        // Action should be a valid variant
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

        // Should have recorded the action
        assert_eq!(player.round_actions.len(), 1);
        assert_eq!(player.round_actions[0].0, action);
    }

    #[test]
    fn test_update_outcome_updates_q_values() {
        let mut player = SdpgPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        // Select an action to populate round_actions
        let action = player.select_action(&grid, pos, &[], &mut rng);

        // Verify Q-values are still zero before outcome
        let idx = action.as_usize();
        assert!((player.q_values[idx]).abs() < 1e-6);

        // Update outcome
        player.update_outcome(true, false, 1);

        // Q-values should now be non-zero for the selected action
        assert!(
            player.q_values[idx].abs() > 0.0,
            "Q-value for {action:?} should be updated"
        );
        assert_eq!(player.visits[idx], 1);

        // Arena outcome should be recorded
        assert!(player.last_arena_outcome.is_some());

        // Round state should be cleared
        assert!(player.round_actions.is_empty());
        assert!(player.round_template_ids.is_empty());
    }

    #[test]
    fn test_reset_clears_round() {
        let mut player = SdpgPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        // Play some actions and update outcome to establish Q-values
        player.select_action(&grid, pos, &[], &mut rng);
        player.select_action(&grid, pos, &[], &mut rng);
        player.update_outcome(true, false, 1);

        // Q-values should now be non-zero
        assert!(
            player.q_values.iter().any(|&q| q != 0.0),
            "Q-values should be updated after outcome"
        );
        assert!(
            player.visits.iter().any(|&v| v > 0),
            "Visits should be incremented after outcome"
        );

        // Simulate some state
        player.alive = false;
        player.powerups_collected = 3;
        player.known_bombs.push(((3, 3), 2, 4));
        player.last_arena_outcome = Some(2.5);

        // Reset
        player.reset();

        // Round state cleared
        assert!(player.known_bombs.is_empty());
        assert!(player.known_powerups.is_empty());
        assert!(player.known_opponents.is_empty());
        assert!(player.round_actions.is_empty());
        assert!(player.round_template_ids.is_empty());
        assert!(player.alive);
        assert_eq!(player.powerups_collected, 0);
        assert!(player.last_dir.is_none());
        assert!(player.last_arena_outcome.is_none());

        // Bandit/SDPG memory persists across rounds
        assert!(
            player.q_values.iter().any(|&q| q != 0.0) || player.visits.iter().any(|&v| v > 0),
            "Q-values/visits should persist across resets"
        );
    }

    #[test]
    fn test_name_and_emoji() {
        let player = SdpgPlayer::new(0);
        assert_eq!(player.name(), "SDPG");
        assert_eq!(player.emoji(), "🎓");
    }
}
