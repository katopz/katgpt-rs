//! G-Zero self-play bomber player — template-driven strategic adaptation.
//!
//! Uses G-Zero's Hint-δ signal (intrinsic reward from template-based hints)
//! to adaptively select strategic templates via UCB1 bandit.
//!
//! # Architecture
//!
//! ```text
//! GZeroPlayer
//!   ├── BomberTemplateProposer  (UCB1 template selection)
//!   ├── DeltaBanditPruner       (δ as dense reward for arm selection)
//!   ├── DeltaGatedAbsorbCompress (δ-gated absorb-compress)
//!   └── Cross-round Q-values    (action-level bandit memory)
//! ```
//!
//! # Flow (per tick)
//!
//! 1. Update game state from events
//! 2. Compute heuristic baseline (query_scores)
//! 3. Select template via UCB1 → compute hinted_scores
//! 4. Compute δ = mean(hinted - query) → feed to all components
//! 5. Blend hinted_scores with Q-values (80/20)
//! 6. Safety filter + ε-greedy exploration
//! 7. Record (action, δ) for outcome update

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
use crate::pruners::bandit::{BanditPruner, BanditStrategy};
use crate::pruners::g_zero::{
    BomberTemplate, BomberTemplateProposer, DeltaBanditPruner, DeltaGatedAbsorbCompress,
    DeltaGatedConfig, hint_score_override,
};
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
///
/// Mean δ is always positive because hints add to 2-4 actions and subtract from 0-2,
/// giving the bandit zero discriminative signal. Argmax δ measures how much the hint
/// improved the **chosen** action — templates that shift the best action get rewarded.
fn compute_game_delta(
    query_scores: &[f32; ACTION_COUNT],
    hinted_scores: &[f32; ACTION_COUNT],
) -> f32 {
    // Find argmax of hinted_scores among valid actions
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

// ── GZeroPlayer ────────────────────────────────────────────────

/// G-Zero self-play bomber player with template-driven strategic adaptation.
///
/// Uses [`BomberTemplateProposer`] (UCB1) to select strategic archetypes,
/// then computes Hint-δ (intrinsic reward) from the score shift.
/// δ feeds back to all bandit components for adaptive learning.
pub struct GZeroPlayer {
    _id: u8,
    // Game state tracking
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<KnownOpponent>,
    last_dir: Option<BomberAction>,
    // G-Zero components
    template_proposer: BomberTemplateProposer,
    delta_bandit: DeltaBanditPruner<NoScreeningPruner>,
    absorb_compress: DeltaGatedAbsorbCompress<NoScreeningPruner>,
    delta_history: Vec<f32>,
    round_actions: Vec<(BomberAction, f32)>,
    /// All template IDs selected this round — for outcome-based bandit reward attribution.
    round_template_ids: Vec<usize>,
    // Cross-round Q-values
    q_values: [f32; ACTION_COUNT],
    visits: [u32; ACTION_COUNT],
}

impl GZeroPlayer {
    /// Create a new GZeroPlayer with the given player ID.
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
        }
    }

    /// Mean δ across all actions this round.
    fn round_delta_mean(&self) -> f32 {
        if self.round_actions.is_empty() {
            return 0.0;
        }
        self.round_actions.iter().map(|(_, d)| d).sum::<f32>() / self.round_actions.len() as f32
    }

    /// Update Q-values from round outcome + feed outcome reward to template bandit (F4).
    ///
    /// Per-tick δ alone cannot differentiate templates (issue 055: always positive).
    /// The round outcome (survived/killed/powerups) provides the real discriminative signal
    /// for the template proposer — surviving with a template = positive reward, dying = negative.
    pub fn update_outcome(&mut self, survived: bool, killed: bool, powerups: u32) {
        if self.round_actions.is_empty() {
            return;
        }

        let outcome_reward = if survived { 1.0 } else { -1.0 }
            + if killed { 2.0 } else { 0.0 }
            + powerups as f32 * 0.5;

        // Distribute outcome reward across ALL templates used this round (not just last one).
        // Each template gets equal share of the reward — survived = positive, died = negative.
        // Uses observe_outcome (not observe_delta) so UCB1 score uses outcome-based reward.
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
}

// ── BomberPlayer Trait ─────────────────────────────────────────

impl BomberPlayer for GZeroPlayer {
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

        // 2. Compute query_scores (WEAK heuristic — templates have room to add value)
        //
        // Uses simple walkability + distance scoring (no BFS escape, no wall-aware blast).
        // The strong score_action is reserved for the safety filter (step 10).
        // This is F3 from issue 055: weak baseline → hints can actually shift the decision.
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
                        // Mild powerup attraction (weaker than score_action's)
                        if let Some(pu) = self
                            .known_powerups
                            .iter()
                            .min_by_key(|p| (target.x - p.0).abs() + (target.y - p.1).abs())
                        {
                            let dist = (target.x - pu.0).abs() + (target.y - pu.1).abs();
                            s += 0.5 / (dist as f32 + 1.0);
                        }
                        // Penalize being near bombs (simple distance, not wall-aware)
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

        // 5. Compute δ (game-domain Hint-δ)
        let delta_value = compute_game_delta(&query_scores, &hinted_scores);

        // 6-8. Feed δ to all components
        self.template_proposer
            .observe_delta(template_id, delta_value);
        self.delta_bandit.observe_delta(template_id, delta_value);
        self.absorb_compress
            .observe_delta(template_id, delta_value, delta_value.max(0.0));

        // 9. Blend hinted_scores with q_values (80% heuristic + 20% bandit)
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

        // 10. Safety filter — wall-aware blast zones with escape guidance
        //
        // Key insight: when already in a blast zone, blocking ALL blast-zone cells
        // leaves no valid moves → Wait → death. Instead, when in danger, override
        // move scores with score_action's BFS escape-distance guidance.
        // When safe, block moves INTO blast zones to prevent walking into danger.
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

        // 11. ε-greedy exploration (15% — diverse template discovery, only safe moves)
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
        "GZero"
    }

    fn emoji(&self) -> &str {
        "🤖"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.known_opponents.clear();
        self.round_actions.clear();
        self.round_template_ids.clear();
        self.last_dir = None;
        // NOTE: Q-values, visits, template stats persist across rounds (bandit memory)
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
    fn test_move_target() {
        let pos = GridPos { x: 5, y: 5 };
        assert_eq!(move_target(BomberAction::Up, pos), GridPos { x: 5, y: 4 });
        assert_eq!(move_target(BomberAction::Down, pos), GridPos { x: 5, y: 6 });
        assert_eq!(move_target(BomberAction::Left, pos), GridPos { x: 4, y: 5 });
        assert_eq!(
            move_target(BomberAction::Right, pos),
            GridPos { x: 6, y: 5 }
        );
        assert_eq!(move_target(BomberAction::Bomb, pos), pos);
        assert_eq!(move_target(BomberAction::Wait, pos), pos);
    }

    #[test]
    fn test_in_blast_zone_wall_aware() {
        let grid = empty_grid();
        let bombs: Vec<KnownBomb> = vec![((5, 5), 2, 4)];
        // On the bomb itself — always in blast zone
        assert!(in_blast_zone(GridPos { x: 5, y: 5 }, &grid, &bombs));
        // Adjacent cell (1 step) — in range 2, walkable near spawn area
        assert!(in_blast_zone(GridPos { x: 5, y: 4 }, &grid, &bombs));
        // Out of range — diagonal offset > 2 in both axes
        assert!(!in_blast_zone(GridPos { x: 1, y: 1 }, &grid, &bombs));
    }

    #[test]
    fn test_compute_game_delta() {
        let query = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let hinted = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let delta = compute_game_delta(&query, &hinted);
        assert!((delta - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_game_delta_with_neg_inf() {
        let query = [1.0, f32::NEG_INFINITY, 3.0, 4.0, 5.0, 6.0, 7.0];
        let hinted = [1.5, f32::NEG_INFINITY, 3.5, 4.5, 5.5, 6.5, 7.5];
        let delta = compute_game_delta(&query, &hinted);
        // Only 4 valid pairs (excluding index 1)
        assert!((delta - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gzero_player_new() {
        let player = GZeroPlayer::new(0);
        assert_eq!(player._id, 0);
        assert!(player.known_bombs.is_empty());
        assert!(player.known_powerups.is_empty());
        assert!(player.known_opponents.is_empty());
        assert!(player.last_dir.is_none());
        assert!(player.round_actions.is_empty());
        assert_eq!(player.q_values, [0.0; ACTION_COUNT]);
        assert_eq!(player.visits, [0; ACTION_COUNT]);
    }

    #[test]
    fn test_gzero_player_select_action() {
        let mut player = GZeroPlayer::new(0);
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
    fn test_gzero_player_trait_methods() {
        let player = GZeroPlayer::new(1);
        assert_eq!(player.name(), "GZero");
        assert_eq!(player.emoji(), "🤖");
    }

    #[test]
    fn test_gzero_player_reset() {
        let mut player = GZeroPlayer::new(0);
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
    }

    #[test]
    fn test_gzero_player_update_outcome() {
        let mut player = GZeroPlayer::new(0);
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
    fn test_gzero_player_update_outcome_empty() {
        let mut player = GZeroPlayer::new(0);
        player.update_outcome(true, false, 0);
        assert!(player.delta_history.is_empty());
    }

    #[test]
    fn test_gzero_player_delta_summary() {
        let player = GZeroPlayer::new(0);
        let (mean, positive, template) = player.delta_summary();
        assert!((mean - 0.0).abs() < 0.01);
        assert!((positive - 0.0).abs() < 0.01);
        // When all deltas are 0.0, max_by picks the last tied template (WaitTrap)
        assert!(BomberTemplate::all().contains(&template));
    }

    #[test]
    fn test_gzero_player_compress_cycle() {
        let mut player = GZeroPlayer::new(0);
        let compressed = player.compress_cycle();
        // No observations yet, should be empty
        assert!(compressed.is_empty());
    }

    #[test]
    fn test_score_action_unwalkable() {
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        // Up from (1,1) is (1,0) which is a border wall
        let score = score_action(&BomberAction::Up, &grid, pos, &[], &[], None);
        assert_eq!(score, f32::NEG_INFINITY);
    }

    #[test]
    fn test_score_action_bomb() {
        let grid = empty_grid();
        let pos = GridPos { x: 3, y: 3 };
        let score = score_action(&BomberAction::Bomb, &grid, pos, &[], &[], None);
        // should_place_bomb checks walls + escape, may be NEG_INF if no escape
        assert!(score == f32::NEG_INFINITY || score >= 0.0);
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
        // Original bomb fuse decremented
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
