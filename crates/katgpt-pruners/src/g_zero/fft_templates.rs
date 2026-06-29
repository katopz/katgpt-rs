//! FFT Tactics-specific strategy templates for G-Zero self-play.
//!
//! Each template represents a priority archetype (heal-first, cure-first, kill-first, etc.)
//! that modifies action scores. The bandit-weighted proposer selects templates via UCB1.

use std::cmp::Ordering;

use crate::fft::battle::BattleState;
use crate::fft::types::*;

// ── FFTTemplate ────────────────────────────────────────────────

/// FFT Tactics strategy archetypes for G-Zero self-play.
///
/// Each template biases action selection toward a different strategic priority.
/// The bandit-weighted proposer learns which templates are most effective
/// by observing the δ signal from template-modified vs unmodified scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FFTTemplate {
    HealFirst,
    CureDebuffFirst,
    KillPriority,
    BuffFirst,
    ProtectSquishy,
    FocusFire,
    BurstDamage,
    EconomyPlay,
    DispelEnemy,
    Kite,
}

impl FFTTemplate {
    pub const fn all() -> [Self; 10] {
        [
            Self::HealFirst,
            Self::CureDebuffFirst,
            Self::KillPriority,
            Self::BuffFirst,
            Self::ProtectSquishy,
            Self::FocusFire,
            Self::BurstDamage,
            Self::EconomyPlay,
            Self::DispelEnemy,
            Self::Kite,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::HealFirst => "HealFirst",
            Self::CureDebuffFirst => "CureDebuff",
            Self::KillPriority => "KillPrio",
            Self::BuffFirst => "BuffFirst",
            Self::ProtectSquishy => "ProtectSq",
            Self::FocusFire => "FocusFire",
            Self::BurstDamage => "BurstDmg",
            Self::EconomyPlay => "Economy",
            Self::DispelEnemy => "Dispel",
            Self::Kite => "Kite",
        }
    }

    pub const fn count() -> usize {
        10
    }
}

// ── TemplateStats (same UCB1 pattern as bomber_templates.rs) ──

struct TemplateStats {
    total_delta: f32,
    delta_count: usize,
    pulls: u32,
}

impl TemplateStats {
    fn new() -> Self {
        Self {
            total_delta: 0.0,
            delta_count: 0,
            pulls: 0,
        }
    }

    fn mean_delta(&self) -> f32 {
        if self.delta_count == 0 {
            return 0.0;
        }
        self.total_delta / self.delta_count as f32
    }

    fn ucb1_score(&self, total_pulls: u32) -> f32 {
        if self.pulls == 0 {
            return f32::MAX;
        }
        let exploration = (2.0 * (total_pulls as f32).ln() / self.pulls as f32).sqrt();
        self.mean_delta() + exploration
    }
}

// ── FFTTemplateProposer ────────────────────────────────────────

/// UCB1 bandit proposer over FFT strategy templates.
///
/// Selects templates to apply during action scoring, then observes
/// the δ signal (how much the template shifted scores) as reward.
/// Over time, converges to the most informative templates.
pub struct FFTTemplateProposer {
    stats: [TemplateStats; 10],
    total_pulls: u32,
}

impl FFTTemplateProposer {
    pub fn new() -> Self {
        Self {
            stats: std::array::from_fn(|_| TemplateStats::new()),
            total_pulls: 0,
        }
    }

    /// Select a template via UCB1. Returns (template, template_id).
    pub fn select(&mut self) -> (FFTTemplate, usize) {
        let best_id = (0..10)
            .max_by(|a, b| {
                self.stats[*a]
                    .ucb1_score(self.total_pulls)
                    .partial_cmp(&self.stats[*b].ucb1_score(self.total_pulls))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);

        self.stats[best_id].pulls += 1;
        self.total_pulls += 1;
        (FFTTemplate::all()[best_id], best_id)
    }

    /// Observe δ reward for a template.
    pub fn observe_delta(&mut self, template_id: usize, delta_value: f32) {
        if template_id >= 10 {
            return;
        }
        self.stats[template_id].total_delta += delta_value;
        self.stats[template_id].delta_count += 1;
    }

    /// Mean δ for a specific template.
    pub fn mean_delta(&self, template_id: usize) -> f32 {
        self.stats
            .get(template_id)
            .map(|s| s.mean_delta())
            .unwrap_or(0.0)
    }

    /// Template with highest mean δ.
    ///
    /// Returns `HealFirst` when no observations exist.
    pub fn best_template(&self) -> FFTTemplate {
        // Return default when no δ observations exist
        if !self.stats.iter().any(|s| s.delta_count > 0) {
            return FFTTemplate::HealFirst;
        }
        (0..10)
            .max_by(|a, b| {
                self.stats[*a]
                    .mean_delta()
                    .partial_cmp(&self.stats[*b].mean_delta())
                    .unwrap_or(Ordering::Equal)
            })
            .map(|i| FFTTemplate::all()[i])
            .unwrap_or(FFTTemplate::HealFirst)
    }

    /// Total number of template selections.
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Per-template selection distribution (sums to 1.0).
    pub fn template_distribution(&self) -> Vec<(FFTTemplate, f32)> {
        let total = self.total_pulls.max(1) as f32;
        FFTTemplate::all()
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, self.stats[i].pulls as f32 / total))
            .collect()
    }
}

impl Default for FFTTemplateProposer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Hint Score Override ─────────────────────────────────────────

/// Compute score modifier for an action given a strategy template.
///
/// Returns a bonus/penalty to add to the base heuristic score,
/// biasing action selection toward the template's strategic archetype.
pub fn hint_score_override(
    template: FFTTemplate,
    action: ActionType,
    state: &BattleState,
    unit_id: u8,
) -> f32 {
    let unit = &state.units[unit_id as usize];
    let effects = &state.effects;
    let hp_pct = unit.hp_pct();
    let enemy_team = BattleState::enemy_team(unit.team);
    let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
    let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);

    match template {
        FFTTemplate::HealFirst => {
            let any_wounded = allies
                .iter()
                .any(|&a| state.units[a as usize].hp_pct() < 0.5);
            match action {
                ActionType::WhiteMagic if any_wounded => 3.0,
                ActionType::Potion if hp_pct < 0.5 => 3.0,
                ActionType::Attack | ActionType::BlackMagic if any_wounded => -1.0,
                _ => 0.0,
            }
        }
        FFTTemplate::CureDebuffFirst => {
            let any_debuffed = allies.iter().any(|&a| {
                effects
                    .iter()
                    .any(|e| e.source == a && e.effect.is_debuff())
            });
            match action {
                ActionType::CurePoison if any_debuffed => 4.0,
                ActionType::Esuna if any_debuffed => 4.0,
                ActionType::WhiteMagic if any_debuffed => -1.0,
                _ => 0.0,
            }
        }
        FFTTemplate::KillPriority => {
            let weak_enemy = enemies
                .iter()
                .any(|&e| state.units[e as usize].hp_pct() < 0.3);
            match action {
                ActionType::Attack if weak_enemy => 3.0,
                ActionType::BlackMagic if weak_enemy => 3.0,
                _ => 0.0,
            }
        }
        FFTTemplate::BuffFirst => {
            let has_buff = effects
                .iter()
                .any(|e| e.source == unit_id && e.effect.is_buff());
            match action {
                ActionType::Defend if !has_buff => 2.0,
                _ => 0.0,
            }
        }
        FFTTemplate::ProtectSquishy => {
            let squishy_nearby = allies
                .iter()
                .any(|&a| state.units[a as usize].stats.def <= 6);
            match action {
                ActionType::Defend if squishy_nearby => 3.0,
                ActionType::WhiteMagic if squishy_nearby => 1.0,
                _ => 0.0,
            }
        }
        FFTTemplate::FocusFire => match action {
            ActionType::Attack if !enemies.is_empty() => 2.0,
            ActionType::BlackMagic if !enemies.is_empty() => 1.5,
            _ => 0.0,
        },
        FFTTemplate::BurstDamage => match action {
            ActionType::BlackMagic if unit.mp > unit.stats.max_mp / 2 => 3.0,
            ActionType::WhiteMagic => -2.0,
            _ => 0.0,
        },
        FFTTemplate::EconomyPlay => match action {
            ActionType::Defend if unit.mp < unit.stats.max_mp * 3 / 10 => 2.0,
            ActionType::BlackMagic => -1.0,
            _ => 0.0,
        },
        FFTTemplate::DispelEnemy => {
            let enemy_buffed = enemies.iter().any(|&e| {
                effects
                    .iter()
                    .any(|ef| ef.source == e && ef.effect.is_buff())
            });
            match action {
                ActionType::Dispel if enemy_buffed => 3.0,
                _ => 0.0,
            }
        }
        FFTTemplate::Kite => {
            let melee_near = enemies.iter().any(|&e| {
                state.units[e as usize].pos.manhattan(unit.pos) <= 1
                    && state.units[e as usize].stats.range <= 1
            });
            match action {
                ActionType::Wait if melee_near => 2.0,
                ActionType::Defend if melee_near => 1.5,
                ActionType::Attack if melee_near => -1.0,
                _ => 0.0,
            }
        }
    }
}

// ── Game-domain Hint-δ computation ─────────────────────────────

/// Compute game-domain delta: mean score shift from template override.
///
/// Compares the query (base heuristic) scores against the hinted
/// (template-modified) scores. Positive delta means the template
/// shifted action preferences — a potential blind spot indicator.
pub fn compute_game_delta(query_scores: &[f32; 9], hinted_scores: &[f32; 9]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..9 {
        if query_scores[i] > f32::NEG_INFINITY && hinted_scores[i] > f32::NEG_INFINITY {
            sum += hinted_scores[i] - query_scores[i];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_templates_count() {
        assert_eq!(FFTTemplate::all().len(), 10);
    }

    #[test]
    fn test_select_returns_valid() {
        let mut proposer = FFTTemplateProposer::new();
        let (template, id) = proposer.select();
        assert!(id < 10);
        assert_eq!(template, FFTTemplate::all()[id]);
    }

    #[test]
    fn test_observe_delta() {
        let mut proposer = FFTTemplateProposer::new();
        proposer.select();
        proposer.observe_delta(0, 0.5);
        proposer.observe_delta(0, 0.3);
        assert!((proposer.mean_delta(0) - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_best_template() {
        let mut proposer = FFTTemplateProposer::new();
        proposer.observe_delta(0, 0.1);
        proposer.observe_delta(5, 0.9);
        assert_eq!(proposer.best_template(), FFTTemplate::FocusFire);
    }

    #[test]
    fn test_distribution_sums_to_one() {
        let mut proposer = FFTTemplateProposer::new();
        for _ in 0..100 {
            proposer.select();
        }
        let dist = proposer.template_distribution();
        let sum: f32 = dist.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ucb1_explores() {
        let mut proposer = FFTTemplateProposer::new();
        for _ in 0..30 {
            proposer.select();
        }
        let dist = proposer.template_distribution();
        let non_zero = dist.iter().filter(|(_, p)| *p > 0.0).count();
        assert!(non_zero > 1);
    }

    #[test]
    fn test_default() {
        let proposer = FFTTemplateProposer::default();
        assert_eq!(proposer.total_pulls(), 0);
    }

    #[test]
    fn test_compute_game_delta() {
        let query = [2.0, 1.0, 2.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hinted = [1.0, 1.0, 5.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delta = compute_game_delta(&query, &hinted);
        // Mean shift: (−1 + 0 + 3 + 0 + 0 + 0 + 0 + 0 + 0) / 9
        assert!((delta - 2.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_game_delta_with_neg_inf() {
        let query = [2.0, f32::NEG_INFINITY, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hinted = [3.0, f32::NEG_INFINITY, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delta = compute_game_delta(&query, &hinted);
        // 8 valid slots (index 1 skipped): (1 + 1 + 0×6) / 8 = 0.25
        assert!((delta - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_observe_delta_out_of_bounds() {
        let mut proposer = FFTTemplateProposer::new();
        proposer.observe_delta(99, 1.0);
        assert!((proposer.mean_delta(99)).abs() < 1e-6);
    }

    #[test]
    fn test_template_names() {
        let names: Vec<&str> = FFTTemplate::all().map(|t| t.name()).to_vec();
        assert_eq!(names[0], "HealFirst");
        assert_eq!(names[9], "Kite");
    }

    #[test]
    fn test_count_matches_all() {
        assert_eq!(FFTTemplate::count(), FFTTemplate::all().len());
    }

    #[test]
    fn test_heal_first_override() {
        let state = BattleState::new();
        // Unit 0 is Party Knight at (1,1) with full HP
        let bonus = hint_score_override(FFTTemplate::HealFirst, ActionType::Attack, &state, 0);
        // No wounded allies → no bonus
        assert!((bonus).abs() < 1e-6);
    }

    #[test]
    fn test_kill_priority_override() {
        let mut state = BattleState::new();
        // Move enemy unit 4 within range of unit 0 (Knight range=1)
        state.units[4].pos = Pos::new(1, 2);
        // Make enemy unit 4 low HP (10/120 ≈ 8%)
        state.units[4].hp = 10;
        let bonus = hint_score_override(FFTTemplate::KillPriority, ActionType::Attack, &state, 0);
        // Enemy in range and < 30% HP → +3.0
        assert!((bonus - 3.0).abs() < 1e-6);
    }
}
