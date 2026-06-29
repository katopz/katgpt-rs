//! Bomberman-specific strategy templates for G-Zero self-play.
//!
//! Each template represents a strategic archetype that modifies action scores.
//! The bandit-weighted proposer selects templates via UCB1, and Hint-δ feeds
//! back to adjust template selection toward strategies that reveal blind spots.

use std::cmp::Ordering;

// ── BomberTemplate ───────────────────────────────────────────────

/// 8 strategy archetypes for Bomberman G-Zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BomberTemplate {
    /// Maximize distance from bombs.
    FleeBlast,
    /// Move toward closest opponent.
    ChaseNearest,
    /// Place bomb near destructible walls.
    BombWall,
    /// Prefer corner positions with escape routes.
    CampCorner,
    /// Prioritize powerup collection.
    PowerUpHunt,
    /// Place bombs to trap opponents.
    CutoffOpponent,
    /// Prefer center grid positions.
    CenterControl,
    /// Wait to bait opponents into blast zones.
    WaitTrap,
}

impl BomberTemplate {
    /// All 8 template variants.
    pub const fn all() -> [Self; 8] {
        [
            Self::FleeBlast,
            Self::ChaseNearest,
            Self::BombWall,
            Self::CampCorner,
            Self::PowerUpHunt,
            Self::CutoffOpponent,
            Self::CenterControl,
            Self::WaitTrap,
        ]
    }

    /// Template name for display.
    pub fn name(self) -> &'static str {
        match self {
            Self::FleeBlast => "FleeBlast",
            Self::ChaseNearest => "ChaseNearest",
            Self::BombWall => "BombWall",
            Self::CampCorner => "CampCorner",
            Self::PowerUpHunt => "PowerUpHunt",
            Self::CutoffOpponent => "CutoffOpponent",
            Self::CenterControl => "CenterControl",
            Self::WaitTrap => "WaitTrap",
        }
    }

    /// Total number of templates.
    pub const fn count() -> usize {
        8
    }
}

// ── TemplateStats ────────────────────────────────────────────────

/// Per-template UCB1 statistics.
struct TemplateStats {
    total_delta: f32,
    delta_count: usize,
    pulls: u32,
    /// Outcome-based reward (survived/killed/powerups) — primary bandit signal.
    total_outcome: f32,
    outcome_count: usize,
}

impl TemplateStats {
    fn new() -> Self {
        Self {
            total_delta: 0.0,
            delta_count: 0,
            pulls: 0,
            total_outcome: 0.0,
            outcome_count: 0,
        }
    }

    fn mean_delta(&self) -> f32 {
        if self.delta_count == 0 {
            return 0.0;
        }
        self.total_delta / self.delta_count as f32
    }

    /// UCB1 using outcome-based reward (not δ). Outcome reward is the primary
    /// discriminative signal — survived templates get +reward, dead templates get -reward.
    fn ucb1_score(&self, total_pulls: u32) -> f32 {
        if self.pulls == 0 {
            return f32::MAX; // Prioritize unvisited
        }
        let reward = if self.outcome_count > 0 {
            self.total_outcome / self.outcome_count as f32
        } else {
            self.mean_delta() // Fallback to δ if no outcome yet
        };
        let exploration = (2.0 * (total_pulls as f32).ln() / self.pulls as f32).sqrt();
        reward + exploration
    }
}

// ── BomberTemplateProposer ──────────────────────────────────────

/// Bandit-weighted template proposer for Bomberman strategies.
///
/// Uses UCB1 to balance exploration of new strategies vs exploitation
/// of known-effective ones. Hint-δ observations feed back to adjust
/// which templates get selected more often.
pub struct BomberTemplateProposer {
    stats: [TemplateStats; 8],
    total_pulls: u32,
}

impl BomberTemplateProposer {
    /// Create a new proposer with uniform prior.
    pub fn new() -> Self {
        Self {
            stats: std::array::from_fn(|_| TemplateStats::new()),
            total_pulls: 0,
        }
    }

    /// Select a template via UCB1, returns (template, template_id).
    pub fn select(&mut self) -> (BomberTemplate, usize) {
        let best_id = (0..8)
            .max_by(|a, b| {
                self.stats[*a]
                    .ucb1_score(self.total_pulls)
                    .partial_cmp(&self.stats[*b].ucb1_score(self.total_pulls))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);

        self.stats[best_id].pulls += 1;
        self.total_pulls += 1;
        (BomberTemplate::all()[best_id], best_id)
    }

    /// Feed Hint-δ back to the bandit for a template (per-tick signal, secondary).
    pub fn observe_delta(&mut self, template_id: usize, delta_value: f32) {
        if template_id >= 8 {
            return;
        }
        self.stats[template_id].total_delta += delta_value;
        self.stats[template_id].delta_count += 1;
    }

    /// Feed round outcome reward to a specific template (primary signal, F4).
    ///
    /// This is the real discriminative signal: survived = positive, died = negative.
    /// Per-tick δ alone is always positive (issue 055) and cannot differentiate templates.
    /// Caller distributes reward across all templates used in the round.
    pub fn observe_outcome(&mut self, template_id: usize, reward: f32) {
        if template_id >= 8 {
            return;
        }
        self.stats[template_id].total_outcome += reward;
        self.stats[template_id].outcome_count += 1;
    }

    /// Mean δ for a specific template.
    pub fn mean_delta(&self, template_id: usize) -> f32 {
        self.stats
            .get(template_id)
            .map(TemplateStats::mean_delta)
            .unwrap_or(0.0)
    }

    /// Template with highest mean outcome reward (most successful).
    pub fn best_template(&self) -> BomberTemplate {
        (0..8)
            .max_by(|a, b| {
                let a_reward = if self.stats[*a].outcome_count > 0 {
                    self.stats[*a].total_outcome / self.stats[*a].outcome_count as f32
                } else {
                    self.stats[*a].mean_delta()
                };
                let b_reward = if self.stats[*b].outcome_count > 0 {
                    self.stats[*b].total_outcome / self.stats[*b].outcome_count as f32
                } else {
                    self.stats[*b].mean_delta()
                };
                a_reward.partial_cmp(&b_reward).unwrap_or(Ordering::Equal)
            })
            .map(|i| BomberTemplate::all()[i])
            .unwrap_or(BomberTemplate::FleeBlast)
    }

    /// Total number of template selections made.
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Normalized pull distribution across templates.
    pub fn template_distribution(&self) -> Vec<(BomberTemplate, f32)> {
        let total = self.total_pulls.max(1) as f32;
        BomberTemplate::all()
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, self.stats[i].pulls as f32 / total))
            .collect()
    }

    /// Get UCB1 score for a specific template (exposes private TemplateStats method).
    pub fn ucb1_score(&self, template_id: usize, total_pulls: u32) -> f32 {
        self.stats
            .get(template_id)
            .map(|s| s.ucb1_score(total_pulls))
            .unwrap_or(f32::MAX)
    }

    /// Record a template pull without going through `select()`.
    ///
    /// Used when template selection is overridden by EM-guided logic.
    pub fn record_pull(&mut self, template_id: usize) {
        if template_id >= 8 {
            return;
        }
        self.stats[template_id].pulls += 1;
        self.total_pulls += 1;
    }
}

impl Default for BomberTemplateProposer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Hint Score Override ─────────────────────────────────────────

/// Compute score modifier for an action given a strategy template.
///
/// This is the game-domain equivalent of a "hint" — it biases action
/// selection toward the template's strategic archetype.
///
/// # Arguments
/// * `template` - The selected strategy template
/// * `action_idx` - Action index (0=Up,1=Down,2=Left,3=Right,4=Bomb,5=Wait)
/// * `pos` - Current player position (x, y)
/// * `bomb_positions` - Known bomb positions
/// * `opponents` - Known opponent positions
/// * `grid_w` - Grid width
/// * `grid_h` - Grid height
#[allow(clippy::too_many_arguments)]
pub fn hint_score_override(
    template: BomberTemplate,
    action_idx: usize,
    pos: (i32, i32),
    bomb_positions: &[(i32, i32)],
    opponents: &[(i32, i32)],
    powerups: &[(i32, i32)],
    grid_w: i32,
    grid_h: i32,
) -> f32 {
    let is_move = action_idx < 4;
    let is_bomb = action_idx == 4;
    let is_wait = action_idx == 5;

    let move_delta: (i32, i32) = match action_idx {
        0 => (0, -1), // Up
        1 => (0, 1),  // Down
        2 => (-1, 0), // Left
        3 => (1, 0),  // Right
        _ => (0, 0),
    };
    let target = (pos.0 + move_delta.0, pos.1 + move_delta.1);

    match template {
        BomberTemplate::FleeBlast => {
            hint_flee_blast(is_move, is_bomb, is_wait, pos, target, bomb_positions)
        }
        BomberTemplate::ChaseNearest => {
            hint_chase_nearest(is_move, is_bomb, pos, target, opponents)
        }
        BomberTemplate::BombWall => hint_bomb_wall(is_move, is_bomb, is_wait),
        BomberTemplate::CampCorner => {
            hint_camp_corner(is_move, is_wait, pos, target, grid_w, grid_h)
        }
        BomberTemplate::PowerUpHunt => {
            hint_powerup_hunt(is_move, is_bomb, is_wait, pos, target, powerups)
        }
        BomberTemplate::CutoffOpponent => hint_cutoff_opponent(is_bomb, is_move, pos, opponents),
        BomberTemplate::CenterControl => hint_center_control(is_move, pos, target, grid_w, grid_h),
        BomberTemplate::WaitTrap => hint_wait_trap(is_wait, is_bomb, is_move, pos, opponents),
    }
}

fn hint_flee_blast(
    is_move: bool,
    is_bomb: bool,
    is_wait: bool,
    pos: (i32, i32),
    target: (i32, i32),
    bomb_positions: &[(i32, i32)],
) -> f32 {
    if is_move {
        let current_min_bomb = bomb_positions
            .iter()
            .map(|b| (pos.0 - b.0).abs() + (pos.1 - b.1).abs())
            .min()
            .unwrap_or(999);
        let target_min_bomb = bomb_positions
            .iter()
            .map(|b| (target.0 - b.0).abs() + (target.1 - b.1).abs())
            .min()
            .unwrap_or(999);
        if target_min_bomb > current_min_bomb {
            3.0
        } else {
            -1.0
        }
    } else if is_bomb || is_wait {
        -2.0
    } else {
        0.0
    }
}

fn hint_chase_nearest(
    is_move: bool,
    is_bomb: bool,
    pos: (i32, i32),
    target: (i32, i32),
    opponents: &[(i32, i32)],
) -> f32 {
    if is_move {
        let nearest = opponents
            .iter()
            .min_by_key(|o| (pos.0 - o.0).abs() + (pos.1 - o.1).abs());
        if let Some(op) = nearest {
            let current_dist = (pos.0 - op.0).abs() + (pos.1 - op.1).abs();
            let target_dist = (target.0 - op.0).abs() + (target.1 - op.1).abs();
            if target_dist < current_dist {
                2.0
            } else {
                -0.5
            }
        } else {
            0.0
        }
    } else if is_bomb {
        1.0
    } else {
        0.0
    }
}

fn hint_bomb_wall(is_move: bool, is_bomb: bool, is_wait: bool) -> f32 {
    if is_bomb {
        3.0
    } else if is_move {
        0.5
    } else if is_wait {
        -1.0
    } else {
        0.0
    }
}

fn hint_camp_corner(
    is_move: bool,
    is_wait: bool,
    pos: (i32, i32),
    target: (i32, i32),
    grid_w: i32,
    grid_h: i32,
) -> f32 {
    if is_move {
        let corners = [
            (1, 1),
            (1, grid_h - 2),
            (grid_w - 2, 1),
            (grid_w - 2, grid_h - 2),
        ];
        let current_corner_dist = corners
            .iter()
            .map(|c| (pos.0 - c.0).abs() + (pos.1 - c.1).abs())
            .min()
            .unwrap_or(0);
        let target_corner_dist = corners
            .iter()
            .map(|c| (target.0 - c.0).abs() + (target.1 - c.1).abs())
            .min()
            .unwrap_or(0);
        if target_corner_dist < current_corner_dist {
            2.0
        } else {
            -0.5
        }
    } else if is_wait {
        1.0
    } else {
        0.0
    }
}

/// PowerUpHunt: +2.5 toward nearest powerup, -1.0 away, 0.0 if none known.
fn hint_powerup_hunt(
    is_move: bool,
    is_bomb: bool,
    is_wait: bool,
    pos: (i32, i32),
    target: (i32, i32),
    powerups: &[(i32, i32)],
) -> f32 {
    if powerups.is_empty() {
        // No powerups known — don't bias actions
        return if is_wait { -0.5 } else { 0.0 };
    }
    if is_move {
        let current_min = powerups
            .iter()
            .map(|p| (pos.0 - p.0).abs() + (pos.1 - p.1).abs())
            .min()
            .unwrap_or(i32::MAX);
        let target_min = powerups
            .iter()
            .map(|p| (target.0 - p.0).abs() + (target.1 - p.1).abs())
            .min()
            .unwrap_or(i32::MAX);
        if target_min < current_min {
            2.5 // Moving toward nearest powerup
        } else if target_min == current_min && target_min <= 2 {
            1.0 // Lateral but close — mild positive
        } else {
            -1.0 // Moving away from powerups
        }
    } else if is_bomb {
        -0.5 // Don't bomb when hunting powerups (blast may destroy them)
    } else if is_wait {
        -1.0 // Don't wait when powerups to collect
    } else {
        0.0
    }
}

fn hint_cutoff_opponent(
    is_bomb: bool,
    is_move: bool,
    pos: (i32, i32),
    opponents: &[(i32, i32)],
) -> f32 {
    if is_bomb {
        let near_opponent = opponents
            .iter()
            .any(|o| (pos.0 - o.0).abs() + (pos.1 - o.1).abs() <= 3);
        if near_opponent { 3.0 } else { 0.0 }
    } else if is_move {
        1.0
    } else {
        0.0
    }
}

fn hint_center_control(
    is_move: bool,
    pos: (i32, i32),
    target: (i32, i32),
    grid_w: i32,
    grid_h: i32,
) -> f32 {
    if !is_move {
        return 0.0;
    }
    let center = (grid_w / 2, grid_h / 2);
    let current_dist = (pos.0 - center.0).abs() + (pos.1 - center.1).abs();
    let target_dist = (target.0 - center.0).abs() + (target.1 - center.1).abs();
    if target_dist < current_dist {
        2.0
    } else {
        -0.5
    }
}

fn hint_wait_trap(
    is_wait: bool,
    is_bomb: bool,
    is_move: bool,
    pos: (i32, i32),
    opponents: &[(i32, i32)],
) -> f32 {
    if is_wait {
        3.0
    } else if is_bomb {
        let near_opponent = opponents
            .iter()
            .any(|o| (pos.0 - o.0).abs() + (pos.1 - o.1).abs() <= 2);
        if near_opponent { 1.5 } else { 0.0 }
    } else if is_move {
        -1.0
    } else {
        0.0
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_templates_count() {
        assert_eq!(BomberTemplate::all().len(), 8);
    }

    #[test]
    fn test_select_returns_valid_template() {
        let mut proposer = BomberTemplateProposer::new();
        let (template, id) = proposer.select();
        assert!(id < 8);
        assert_eq!(template, BomberTemplate::all()[id]);
    }

    #[test]
    fn test_observe_delta_updates_stats() {
        let mut proposer = BomberTemplateProposer::new();
        proposer.select();
        proposer.observe_delta(0, 0.5);
        proposer.observe_delta(0, 0.3);
        assert!((proposer.mean_delta(0) - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_observe_delta_out_of_bounds_noop() {
        let mut proposer = BomberTemplateProposer::new();
        proposer.observe_delta(99, 0.5);
        // No panic
    }

    #[test]
    fn test_best_template() {
        let mut proposer = BomberTemplateProposer::new();
        proposer.observe_delta(0, 0.1);
        proposer.observe_delta(3, 0.9);
        assert_eq!(proposer.best_template(), BomberTemplate::CampCorner);
    }

    #[test]
    fn test_template_distribution_sums_to_one() {
        let mut proposer = BomberTemplateProposer::new();
        for _ in 0..100 {
            proposer.select();
        }
        let dist = proposer.template_distribution();
        let sum: f32 = dist.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ucb1_prioritizes_unvisited() {
        let mut proposer = BomberTemplateProposer::new();
        for _ in 0..20 {
            proposer.select();
        }
        let dist = proposer.template_distribution();
        let non_zero = dist.iter().filter(|(_, p)| *p > 0.0).count();
        assert!(non_zero > 1, "UCB1 should explore multiple templates");
    }

    #[test]
    fn test_hint_score_override_flee_blast() {
        let bombs = [(5, 5)];
        let score = hint_score_override(
            BomberTemplate::FleeBlast,
            0, // Up
            (4, 4),
            &bombs,
            &[],
            &[],
            13,
            13,
        );
        assert!(
            score > 0.0,
            "FleeBlast should reward moving away from bombs"
        );
    }

    #[test]
    fn test_hint_score_override_chase_nearest() {
        let opponents = [(8, 8)];
        let score = hint_score_override(
            BomberTemplate::ChaseNearest,
            3, // Right
            (4, 4),
            &[],
            &opponents,
            &[],
            13,
            13,
        );
        assert!(
            score > 0.0,
            "ChaseNearest should reward moving toward opponents"
        );
    }

    #[test]
    fn test_hint_score_override_bomb_wall() {
        let score = hint_score_override(
            BomberTemplate::BombWall,
            4, // Bomb
            (5, 5),
            &[],
            &[],
            &[],
            13,
            13,
        );
        assert!(score > 0.0, "BombWall should reward placing bombs");
    }

    #[test]
    fn test_default() {
        let proposer = BomberTemplateProposer::default();
        assert_eq!(proposer.total_pulls(), 0);
    }

    #[test]
    fn test_template_names() {
        for t in BomberTemplate::all() {
            assert!(!t.name().is_empty(), "{:?} should have a name", t);
        }
    }

    #[test]
    fn test_count_matches_all() {
        assert_eq!(BomberTemplate::count(), BomberTemplate::all().len());
    }

    #[test]
    fn test_hint_score_override_camp_corner() {
        // Moving toward corner (1,1) from (3,3) going left
        let score = hint_score_override(
            BomberTemplate::CampCorner,
            2, // Left
            (3, 3),
            &[],
            &[],
            &[],
            13,
            13,
        );
        assert!(
            score > 0.0,
            "CampCorner should reward moving toward corners"
        );
    }

    #[test]
    fn test_hint_score_override_center_control() {
        // Moving toward center (6,6) from (3,3) going right
        let score = hint_score_override(
            BomberTemplate::CenterControl,
            3, // Right
            (3, 3),
            &[],
            &[],
            &[],
            13,
            13,
        );
        assert!(
            score > 0.0,
            "CenterControl should reward moving toward center"
        );
    }

    #[test]
    fn test_hint_score_override_wait_trap() {
        let score = hint_score_override(
            BomberTemplate::WaitTrap,
            5, // Wait
            (5, 5),
            &[],
            &[],
            &[],
            13,
            13,
        );
        assert!(score > 0.0, "WaitTrap should reward waiting");
    }

    #[test]
    fn test_hint_score_override_cutoff_opponent_nearby() {
        let opponents = [(6, 6)];
        let score = hint_score_override(
            BomberTemplate::CutoffOpponent,
            4, // Bomb
            (5, 5),
            &[],
            &opponents,
            &[],
            13,
            13,
        );
        assert!(
            score > 0.0,
            "CutoffOpponent should reward bomb near opponent"
        );
    }

    #[test]
    fn test_hint_score_override_cutoff_opponent_far() {
        let opponents = [(12, 12)];
        let score = hint_score_override(
            BomberTemplate::CutoffOpponent,
            4, // Bomb
            (5, 5),
            &[],
            &opponents,
            &[],
            13,
            13,
        );
        assert_eq!(
            score, 0.0,
            "CutoffOpponent should not reward bomb with no nearby opponent"
        );
    }
}
