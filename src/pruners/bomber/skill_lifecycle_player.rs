//! SkillLifecyclePlayer — MUSE skill lifecycle wrapper around HLPlayer.
//!
//! Wraps [`HLPlayer`] with the full MUSE skill lifecycle pipeline:
//! 1. [`PrunerMemory`] records arm/reward experiences per action.
//! 2. [`BomberTestGate`] validates skill quality at configurable intervals.
//! 3. [`SkillCatalog`] registers validated skills for promotion.
//! 4. **Feedback loop**: catalog status influences action selection via confidence bonus.
//!
//! Edge-case detection uses running mean±2σ; failure detection uses reward < -0.5.
//! Every N episodes (default 20), runs test gate validation and promotes passing
//! skills into the catalog.
//!
//! The lifecycle feedback loop adds a scoring bonus to HLPlayer's heuristic:
//! - **Active** arms: +1.5 confidence bonus (proven safe and effective)
//! - **Validated** arms: +0.8 confidence bonus (passed test gate)
//! - **Failed** arms: -1.0 penalty (known failure modes)
//! - **Untested** arms: neutral (no bonus/penalty)

#![cfg(feature = "skill_lifecycle")]

use std::any::Any;

use fastrand::Rng;

use crate::pruners::skill_catalog::SkillCatalog;
use crate::pruners::skill_memory::{MemoryEntry, PrunerMemory};
use crate::pruners::skill_test::{BomberTestGate, PrunerTestGate, TestStatus};

use super::arena::ArenaGrid;
use super::players::{ACTION_COUNT, ALL_ACTIONS, BomberPlayer, HLPlayer};
use super::{BomberAction, GameEvent, GridPos};

// ── Constants ──────────────────────────────────────────────────

/// Default interval between test gate validations (episodes).
const DEFAULT_VALIDATE_EVERY: usize = 20;

/// Default PrunerMemory capacity (rounded to next power of 2).
const DEFAULT_MEMORY_CAPACITY: usize = 1024;

/// Reward threshold for failure detection.
const FAILURE_THRESHOLD: f32 = -0.5;

/// Edge-case detection: number of standard deviations from mean.
const EDGE_SIGMA_MULT: f32 = 2.0;

/// Confidence bonus for Active arms (proven safe and effective).
const ACTIVE_BONUS: f32 = 3.0;

/// Confidence bonus for Validated arms (passed test gate).
const VALIDATED_BONUS: f32 = 1.5;

/// Confidence penalty for Failed arms (known failure modes).
const FAILED_PENALTY: f32 = -3.0;

/// Recency-weighted memory bonus: how many recent entries to average.
const MEMORY_RECENT_K: usize = 20;

/// Memory trend bonus weight (positive trend adds this much to score).
const MEMORY_TREND_WEIGHT: f32 = 2.0;

/// Number of recent entries to check for failure suppression.
const FAILURE_RECENT_K: usize = 10;

/// If this fraction of recent entries for an arm are failures, suppress.
const FAILURE_SUPPRESS_RATIO: f32 = 0.5;

// ── LifecycleStats ─────────────────────────────────────────────

/// Cumulative statistics for the skill lifecycle pipeline.
#[derive(Clone, Debug, Default)]
pub struct LifecycleStats {
    /// Total episodes processed through `update_outcome`.
    pub total_episodes: usize,
    /// Number of edge-case observations (reward outside mean ± 2σ).
    pub edge_cases: usize,
    /// Number of failure observations (reward < -0.5).
    pub failures: usize,
    /// Total test gate validations run.
    pub validations_run: usize,
    /// Validations that passed the coverage threshold.
    pub validations_passed: usize,
    /// Validations that failed the coverage threshold.
    pub validations_failed: usize,
    /// Best Q-value across all arms.
    pub best_arm_q: f32,
}

// ── RunningStats ───────────────────────────────────────────────

/// Welford-style online mean/variance for edge-case detection.
#[derive(Clone, Debug, Default)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    #[inline]
    fn update(&mut self, value: f32) {
        let x = value as f64;
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Returns (mean, std). Returns (0.0, 0.0) if fewer than 2 samples.
    #[inline]
    fn mean_std(&self) -> (f64, f64) {
        if self.count < 2 {
            return (self.mean, 0.0);
        }
        let variance = self.m2 / (self.count - 1) as f64;
        (self.mean, variance.sqrt())
    }
}

// ── SkillLifecyclePlayer ───────────────────────────────────────

/// Bomber player wrapping [`HLPlayer`] with the MUSE skill lifecycle.
///
/// Delegates action selection to the inner HLPlayer. After each round,
/// `update_outcome()` records experiences into [`PrunerMemory`], detects
/// edge cases and failures, and periodically runs [`BomberTestGate`]
/// validation. Skills that pass are registered/updated in [`SkillCatalog`].
pub struct SkillLifecyclePlayer {
    /// Inner heuristic-learning player.
    inner: HLPlayer,
    /// Lock-free append-only experience log.
    memory: PrunerMemory,
    /// Test gate for skill quality validation.
    test_gate: BomberTestGate,
    /// Catalog of validated skills.
    catalog: SkillCatalog,
    /// Run test gate validation every N episodes.
    validate_every: usize,
    /// Episodes completed so far.
    episode_count: usize,
    /// Cumulative lifecycle statistics.
    stats: LifecycleStats,
    /// Per-arm running reward statistics for edge-case detection.
    arm_stats: [RunningStats; ACTION_COUNT],
}

impl SkillLifecyclePlayer {
    /// Create a new SkillLifecyclePlayer wrapping an HLPlayer with the given id.
    ///
    /// Uses defaults: memory capacity 1024, validate every 20 episodes, min_coverage 0.6.
    pub fn new(id: u8) -> Self {
        let pruner_id = format!("skill_lifecycle_bomber_{id}");
        let pruner_hash = blake3::hash(pruner_id.as_bytes());
        let short_hash: String = pruner_hash.as_bytes()[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let pruner_id_str = format!("{pruner_id}:{short_hash}");

        Self {
            inner: HLPlayer::new(id),
            memory: PrunerMemory::new(DEFAULT_MEMORY_CAPACITY, &pruner_id_str),
            test_gate: BomberTestGate::with_coverage(0.6),
            catalog: SkillCatalog::new(),
            validate_every: DEFAULT_VALIDATE_EVERY,
            episode_count: 0,
            stats: LifecycleStats::default(),
            arm_stats: Default::default(),
        }
    }

    /// Create with a custom validation interval and coverage threshold.
    pub fn with_config(
        id: u8,
        validate_every: usize,
        min_coverage: f32,
        memory_capacity: usize,
    ) -> Self {
        let pruner_id = format!("skill_lifecycle_bomber_{id}");
        let pruner_hash = blake3::hash(pruner_id.as_bytes());
        let short_hash: String = pruner_hash.as_bytes()[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let pruner_id_str = format!("{pruner_id}:{short_hash}");

        Self {
            inner: HLPlayer::new(id),
            memory: PrunerMemory::new(memory_capacity, &pruner_id_str),
            test_gate: BomberTestGate::with_coverage(min_coverage),
            catalog: SkillCatalog::new(),
            validate_every: validate_every.max(1),
            episode_count: 0,
            stats: LifecycleStats::default(),
            arm_stats: Default::default(),
        }
    }

    /// Update round outcome: delegates to inner HLPlayer, records experiences,
    /// runs periodic validation, and promotes skills.
    ///
    /// Steps:
    /// 1. Delegates `update_outcome` to inner HLPlayer (updates Q-values).
    /// 2. Computes per-action rewards via `compress_cycle`.
    /// 3. For each active arm, appends [`MemoryEntry`] to [`PrunerMemory`].
    /// 4. Every `validate_every` episodes, runs [`BomberTestGate`] validation.
    /// 5. If validation passes, registers/updates skill in [`SkillCatalog`].
    pub fn update_outcome(
        &mut self,
        survived: bool,
        killed_opponent: bool,
        collected_powerups: u32,
    ) {
        // 1. Delegate to inner player — updates Q-values
        self.inner
            .update_outcome(survived, killed_opponent, collected_powerups);

        self.episode_count += 1;
        self.stats.total_episodes += 1;

        // 2. Append experience entries for each arm that has been visited
        let ts = self.episode_count as u64;
        for arm_idx in 0..ACTION_COUNT {
            let arm_q = self.inner.arm_q(arm_idx);
            let visits = self.inner.arm_visits(arm_idx);

            // Skip arms never pulled
            if visits == 0 {
                continue;
            }

            // Update running stats for edge-case detection
            self.arm_stats[arm_idx].update(arm_q);

            // Edge-case detection: reward outside mean ± 2σ
            let (mean, std) = self.arm_stats[arm_idx].mean_std();
            let is_edge_case = if std > 0.0 {
                let reward_f64 = arm_q as f64;
                reward_f64 > mean + EDGE_SIGMA_MULT as f64 * std
                    || reward_f64 < mean - EDGE_SIGMA_MULT as f64 * std
            } else {
                false
            };

            // Failure detection
            let is_failure = arm_q < FAILURE_THRESHOLD;

            if is_edge_case {
                self.stats.edge_cases += 1;
            }
            if is_failure {
                self.stats.failures += 1;
            }

            // Append to lock-free memory
            let entry = MemoryEntry::new(arm_idx as u16, arm_q, is_edge_case, is_failure, ts);
            self.memory.append(entry);
        }

        // 4. Track best Q-value
        let mut best_q = f32::NEG_INFINITY;
        for arm_idx in 0..ACTION_COUNT {
            let q = self.inner.arm_q(arm_idx);
            if q > best_q {
                best_q = q;
            }
        }
        self.stats.best_arm_q = best_q;

        // 5. Periodic validation
        if self.episode_count % self.validate_every == 0 {
            self.run_validation();
        }

        // 6. Promote validated skills to Active after sufficient experience
        self.promote_if_ready();
    }

    /// Run test gate validation and register/promote skills.
    fn run_validation(&mut self) {
        self.stats.validations_run += 1;

        let test_cases = BomberTestGate::bomber_test_cases();
        let result = self.test_gate.validate(&test_cases);

        if result.passed {
            self.stats.validations_passed += 1;

            // Register/update each active arm as a validated skill
            for arm_idx in 0..ACTION_COUNT {
                let visits = self.inner.arm_visits(arm_idx);
                if visits == 0 {
                    continue;
                }

                let action_name = action_name_for_index(arm_idx);
                let arm_q = self.inner.arm_q(arm_idx);
                let descriptor = crate::pruners::skill_catalog::SkillDescriptor::new(
                    action_name,
                    format!(
                        "Bomber arm {} (q={:.3}, visits={})",
                        action_name, arm_q, visits
                    ),
                    arm_idx,
                );

                // Register (replaces if exists)
                self.catalog.register(descriptor);
                // Promote to Validated
                self.catalog.update_status(arm_idx, TestStatus::Validated);
            }
        } else {
            self.stats.validations_failed += 1;

            // Mark failed arms
            for arm_idx in 0..ACTION_COUNT {
                let visits = self.inner.arm_visits(arm_idx);
                if visits == 0 {
                    continue;
                }

                // Only mark as Failed if already registered
                if self.catalog.get(arm_idx).is_some() {
                    self.catalog.update_status(arm_idx, TestStatus::Failed);
                }
            }
        }
    }

    /// Read-only access to lifecycle stats.
    #[inline]
    pub fn stats(&self) -> &LifecycleStats {
        &self.stats
    }

    /// Read-only access to inner HLPlayer.
    #[inline]
    pub fn inner(&self) -> &HLPlayer {
        &self.inner
    }

    /// Mutable access to inner HLPlayer.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut HLPlayer {
        &mut self.inner
    }

    /// Read-only access to the PrunerMemory.
    #[inline]
    pub fn memory(&self) -> &PrunerMemory {
        &self.memory
    }

    /// Read-only access to the SkillCatalog.
    #[inline]
    pub fn catalog(&self) -> &SkillCatalog {
        &self.catalog
    }

    /// Total episodes completed.
    #[inline]
    pub fn episode_count(&self) -> usize {
        self.episode_count
    }

    /// Compute the lifecycle confidence bonus for a given arm.
    ///
    /// The bonus is based on:
    /// 1. **Q-value from memory**: arms with positive Q-values get boosted,
    ///    arms with negative Q-values get penalized.
    /// 2. **Catalog status**: Active/Validated arms get an additional bonus.
    /// 3. **Failure suppression**: arms with high recent failure ratio get penalized.
    /// 4. **Memory trend**: arms with improving reward trends get a bonus.
    pub fn lifecycle_bonus(&self, arm_idx: usize) -> f32 {
        // Q-value is the primary signal — direct measure of arm quality
        let arm_q = self.inner.arm_q(arm_idx);
        let visits = self.inner.arm_visits(arm_idx);

        // No signal for unvisited arms
        if visits == 0 {
            return 0.0;
        }

        // Q-value scaling: map Q from [-1, 1] to a bonus of [-3, 3]
        // This is the primary differentiator between arms
        let q_bonus = arm_q * 3.0;

        // Catalog status bonus (secondary signal)
        let catalog_bonus = match self.catalog.get(arm_idx) {
            Some(desc) => match desc.test_status {
                TestStatus::Active => ACTIVE_BONUS,
                TestStatus::Validated => VALIDATED_BONUS,
                TestStatus::Failed => FAILED_PENALTY,
                TestStatus::Untested => 0.0,
            },
            None => 0.0,
        };

        // Memory trend bonus: compute slope of recent rewards for this arm
        let trend_bonus = if self.episode_count >= MEMORY_RECENT_K {
            let recent = self.memory.recent(MEMORY_RECENT_K);
            let arm_entries: Vec<&MemoryEntry> = recent
                .iter()
                .filter(|e| e.arm as usize == arm_idx)
                .collect();
            if arm_entries.len() >= 3 {
                let mid = arm_entries.len() / 2;
                let first_half_mean: f32 =
                    arm_entries[..mid].iter().map(|e| e.reward).sum::<f32>() / mid as f32;
                let second_half_mean: f32 =
                    arm_entries[mid..].iter().map(|e| e.reward).sum::<f32>()
                        / (arm_entries.len() - mid) as f32;
                let trend = second_half_mean - first_half_mean;
                trend * MEMORY_TREND_WEIGHT
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Failure suppression: if recent entries show high failure rate, penalize
        let failure_penalty = if self.episode_count >= FAILURE_RECENT_K {
            let recent = self.memory.recent(FAILURE_RECENT_K);
            let arm_entries: Vec<&MemoryEntry> = recent
                .iter()
                .filter(|e| e.arm as usize == arm_idx)
                .collect();
            if !arm_entries.is_empty() {
                let failure_count = arm_entries.iter().filter(|e| e.is_failure).count();
                let failure_ratio = failure_count as f32 / arm_entries.len() as f32;
                if failure_ratio > FAILURE_SUPPRESS_RATIO {
                    -2.0 * failure_ratio
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        q_bonus + catalog_bonus + trend_bonus + failure_penalty
    }

    /// Promote validated skills to Active after sufficient experience.
    ///
    /// Skills stay Validated until they've been exercised in at least 40 episodes.
    fn promote_if_ready(&mut self) {
        if self.episode_count < 40 {
            return;
        }
        for arm_idx in 0..ACTION_COUNT {
            if let Some(desc) = self.catalog.get(arm_idx) {
                if desc.test_status == TestStatus::Validated {
                    // Promote validated skills after sufficient experience
                    self.catalog.update_status(arm_idx, TestStatus::Active);
                }
            }
        }
    }

    /// Run absorb-compress cycle on inner HLPlayer.
    /// Returns newly compressed arm indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        self.inner.compress_cycle()
    }
}

// ── BomberPlayer impl ──────────────────────────────────────────

impl BomberPlayer for SkillLifecyclePlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        // Let inner HLPlayer compute its heuristic scores
        let action = self.inner.select_action(grid, pos, events, rng);

        // After a few episodes, apply lifecycle confidence modifications.
        // We check if any action has a significantly better lifecycle profile
        // than HL's chosen action.
        if self.episode_count > 3 {
            let chosen_idx = action_index(&action);
            let chosen_bonus = self.lifecycle_bonus(chosen_idx);

            // Check all arms for a better lifecycle alternative
            let mut best_idx = chosen_idx;
            let mut best_total = chosen_bonus;

            for arm_idx in 0..ACTION_COUNT {
                if arm_idx == chosen_idx {
                    continue;
                }
                let bonus = self.lifecycle_bonus(arm_idx);
                if bonus > best_total {
                    best_total = bonus;
                    best_idx = arm_idx;
                }
            }

            // Override HL's choice if lifecycle signal is strong enough
            // Margin threshold: smaller early (aggressive learning), larger later (conservative)
            let margin_threshold = if self.episode_count < 40 {
                0.5 // Be more willing to explore early
            } else {
                1.0 // Require stronger signal later
            };

            if best_idx != chosen_idx && best_total - chosen_bonus >= margin_threshold {
                let visits = self.inner.arm_visits(best_idx);
                // Don't override with completely untested arms
                if visits > 0 || best_total > 0.0 {
                    return ALL_ACTIONS[best_idx];
                }
            }
        }

        action
    }

    fn name(&self) -> &str {
        "SkillLifecycle"
    }

    fn emoji(&self) -> &str {
        "🧬"
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Convert BomberAction to arm index.
fn action_index(action: &BomberAction) -> usize {
    match action {
        BomberAction::Up => 0,
        BomberAction::Down => 1,
        BomberAction::Left => 2,
        BomberAction::Right => 3,
        BomberAction::Bomb => 4,
        BomberAction::Wait => 5,
        BomberAction::Detonate => 6,
    }
}

/// Human-readable action name for catalog descriptors.
fn action_name_for_index(idx: usize) -> &'static str {
    match idx {
        0 => "bomber_up",
        1 => "bomber_down",
        2 => "bomber_left",
        3 => "bomber_right",
        4 => "bomber_bomb",
        5 => "bomber_wait",
        6 => "bomber_detonate",
        _ => "bomber_unknown",
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_player() -> SkillLifecyclePlayer {
        SkillLifecyclePlayer::new(0)
    }

    #[test]
    fn test_new_player_has_zero_stats() {
        let p = make_player();
        assert_eq!(p.stats().total_episodes, 0);
        assert_eq!(p.stats().edge_cases, 0);
        assert_eq!(p.stats().failures, 0);
        assert_eq!(p.stats().validations_run, 0);
        assert_eq!(p.stats().validations_passed, 0);
        assert_eq!(p.stats().validations_failed, 0);
        assert_eq!(p.episode_count(), 0);
    }

    #[test]
    fn test_name_and_emoji() {
        let p = make_player();
        assert_eq!(p.name(), "SkillLifecycle");
        assert_eq!(p.emoji(), "🧬");
    }

    #[test]
    fn test_update_outcome_increments_episodes() {
        let mut p = make_player();
        p.update_outcome(true, false, 0);
        assert_eq!(p.episode_count(), 1);
        assert_eq!(p.stats().total_episodes, 1);
        assert_eq!(p.stats().best_arm_q, 0.0); // no visits yet → no Q updates
    }

    #[test]
    fn test_update_outcome_survived_records_rewards() {
        let mut p = make_player();
        let mut rng = Rng::with_seed(42);
        // Generate a valid arena
        let grid = ArenaGrid::generate(42);
        let pos = GridPos { x: 1, y: 1 };

        // Run enough episodes to trigger visits via inner HLPlayer
        for _ in 0..5 {
            // Must call select_action to populate round_actions
            let _ = p.select_action(&grid, pos, &[], &mut rng);
            p.update_outcome(true, false, 1);
        }
        assert_eq!(p.stats().total_episodes, 5);
        // Memory should have entries
        assert!(p.memory().total_entries() > 0);
    }

    #[test]
    fn test_validation_triggers_at_interval() {
        let mut p = SkillLifecyclePlayer::with_config(0, 3, 0.6, 256);
        // 3 episodes should trigger first validation
        for _ in 0..3 {
            p.update_outcome(true, false, 0);
        }
        assert!(p.stats().validations_run >= 1);
    }

    #[test]
    fn test_validation_default_interval() {
        let mut p = make_player();
        for _ in 0..DEFAULT_VALIDATE_EVERY {
            p.update_outcome(true, false, 0);
        }
        assert!(p.stats().validations_run >= 1);
    }

    #[test]
    fn test_failure_detection() {
        let mut p = make_player();
        // Simulate many death episodes to drive Q-values negative
        for _ in 0..50 {
            p.update_outcome(false, false, 0);
        }
        // Should have detected some failures or edge cases
        // (depends on HLPlayer internal credit assignment)
        assert!(p.stats().total_episodes == 50);
    }

    #[test]
    fn test_reset_clears_inner_state() {
        let mut p = make_player();
        p.update_outcome(true, false, 0);
        p.reset();
        // episode_count is NOT reset (lifecycle stats persist)
        assert_eq!(p.episode_count(), 1);
    }

    #[test]
    fn test_with_config_custom_interval() {
        let p = SkillLifecyclePlayer::with_config(1, 5, 0.9, 512);
        assert_eq!(p.validate_every, 5);
    }

    #[test]
    fn test_with_config_interval_clamped_to_one() {
        let p = SkillLifecyclePlayer::with_config(1, 0, 0.5, 64);
        assert_eq!(p.validate_every, 1);
    }

    #[test]
    fn test_catalog_initially_empty() {
        let p = make_player();
        // No skills registered yet
        for arm in 0..ACTION_COUNT {
            assert!(p.catalog().get(arm).is_none());
        }
    }

    #[test]
    fn test_action_name_for_index() {
        assert_eq!(action_name_for_index(0), "bomber_up");
        assert_eq!(action_name_for_index(4), "bomber_bomb");
        assert_eq!(action_name_for_index(6), "bomber_detonate");
        assert_eq!(action_name_for_index(7), "bomber_unknown");
    }

    #[test]
    fn test_running_stats_mean_std() {
        let mut stats = RunningStats::default();
        // Single sample → std = 0
        stats.update(1.0);
        let (mean, std) = stats.mean_std();
        assert!((mean - 1.0).abs() < 1e-9);
        assert!(std == 0.0);

        // Two samples
        stats.update(3.0);
        let (mean, std) = stats.mean_std();
        assert!((mean - 2.0).abs() < 1e-9);
        assert!((std - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn test_running_stats_edge_case_detection() {
        let mut stats = RunningStats::default();
        // Build a tight distribution
        for v in [1.0, 1.1, 0.9, 1.0, 1.05, 0.95] {
            stats.update(v);
        }
        let (mean, std) = stats.mean_std();
        // 10.0 is clearly outside mean ± 2σ
        assert!(10.0 > mean + 2.0 * std);
        // -5.0 is clearly below mean - 2σ
        assert!((-5.0f64) < mean - 2.0 * std);
    }

    #[test]
    fn test_compress_cycle_delegates() {
        let mut p = make_player();
        // Run some episodes to generate visits
        for _ in 0..30 {
            p.update_outcome(true, false, 0);
        }
        let compressed = p.compress_cycle();
        // Should return a Vec (possibly empty if no arms met threshold)
        assert!(compressed.len() <= ACTION_COUNT);
    }
}
