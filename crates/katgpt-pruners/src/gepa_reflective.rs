//! GEPA-D Reflective Config Evolution — Modelless Distillation
//!
//! Distills GEPA's reflective prompt evolution into our modelless stack:
//! evolve system-level configuration (rubric weights, template hints, bandit params)
//! from MeMo trajectory reflection using Pareto-frontier bandit selection.
//!
//! No gradient updates. No LoRA. No model-based path.
//! Config variants = bandit arms, reflection quality = reward.
//!
//! Feature-gated behind `gepa_reflective = ["bandit", "memo_reflections"]`.

use super::bandit::BanditPruner;
use super::reflection::ReflectionResult;
use super::review_metrics::ReviewMetrics;
use katgpt_speculative::ScreeningPruner;
use katgpt_types::Rng;

// ── Constants ──────────────────────────────────────────────────

/// Maximum number of Pareto-optimal configs retained.
const MAX_CONFIGS: usize = 24;

/// Number of rubric-weight presets.
const NUM_RUBRIC_PRESETS: usize = 4;

/// Number of discrete bandit ε values explored.
const NUM_EPSILON_VALUES: usize = 4;

/// Number of template hint indices.
const NUM_TEMPLATE_HINTS: usize = 4;

/// Number of absorb threshold levels.
const NUM_ABSORB_THRESHOLDS: usize = 4;

/// Total arms = rubric × epsilon × template × absorb = 256.
const NUM_ARMS: usize =
    NUM_RUBRIC_PRESETS * NUM_EPSILON_VALUES * NUM_TEMPLATE_HINTS * NUM_ABSORB_THRESHOLDS;

/// UCB1 exploration constant.
const UCB1_C: f32 = 2.0;

// ── ConfigVariant ──────────────────────────────────────────────

/// A point in configuration space — one bandit arm.
///
/// Each variant specifies a complete set of knobs for the screening
/// pipeline: rubric weights, exploration rate, template selection,
/// and absorb threshold. These are the dimensions over which GEPA-D
/// searches for optimal performance using reflection quality as reward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfigVariant {
    /// Rubric-weight preset index (0..NUM_RUBRIC_PRESETS).
    pub rubric_preset: u8,
    /// Bandit ε (exploration rate) quantised to 0.05 / 0.10 / 0.20 / 0.40.
    pub epsilon_index: u8,
    /// Template hint index for g_zero::TemplateProposer.
    pub template_hint: u8,
    /// Absorb threshold quantised to 0.1 / 0.3 / 0.5 / 0.7.
    pub absorb_threshold_index: u8,
}

/// Predefined rubric-weight presets.
///
/// Each preset is a 4-element weight vector `[relevance, coherence,
/// novelty, safety]` normalised to sum to 1.0.
pub const RUBRIC_PRESETS: [[f32; 4]; NUM_RUBRIC_PRESETS] = [
    [0.40, 0.30, 0.20, 0.10], // balanced
    [0.55, 0.20, 0.15, 0.10], // relevance-heavy
    [0.20, 0.20, 0.40, 0.20], // novelty-heavy
    [0.25, 0.25, 0.25, 0.25], // uniform
];

/// Epsilon values corresponding to `epsilon_index`.
pub const EPSILON_VALUES: [f32; NUM_EPSILON_VALUES] = [0.05, 0.10, 0.20, 0.40];

/// Absorb thresholds corresponding to `absorb_threshold_index`.
pub const ABSORB_THRESHOLDS: [f32; NUM_ABSORB_THRESHOLDS] = [0.1, 0.3, 0.5, 0.7];

impl ConfigVariant {
    /// Create a config variant from flat arm index.
    #[inline]
    pub fn from_arm(arm: usize) -> Self {
        let a = arm % NUM_ABSORB_THRESHOLDS;
        let rest = arm / NUM_ABSORB_THRESHOLDS;
        let t = rest % NUM_TEMPLATE_HINTS;
        let rest = rest / NUM_TEMPLATE_HINTS;
        let e = rest % NUM_EPSILON_VALUES;
        let r = rest / NUM_EPSILON_VALUES;
        Self {
            rubric_preset: r as u8,
            epsilon_index: e as u8,
            template_hint: t as u8,
            absorb_threshold_index: a as u8,
        }
    }

    /// Convert back to flat arm index.
    #[inline]
    pub fn to_arm(&self) -> usize {
        let r = self.rubric_preset as usize;
        let e = self.epsilon_index as usize;
        let t = self.template_hint as usize;
        let a = self.absorb_threshold_index as usize;
        ((r * NUM_EPSILON_VALUES + e) * NUM_TEMPLATE_HINTS + t) * NUM_ABSORB_THRESHOLDS + a
    }

    /// Resolved rubric weights for this variant.
    #[inline]
    pub fn rubric_weights(&self) -> [f32; 4] {
        RUBRIC_PRESETS[self.rubric_preset as usize % NUM_RUBRIC_PRESETS]
    }

    /// Resolved epsilon for this variant.
    #[inline]
    pub fn epsilon(&self) -> f32 {
        EPSILON_VALUES[self.epsilon_index as usize % NUM_EPSILON_VALUES]
    }

    /// Resolved absorb threshold for this variant.
    #[inline]
    pub fn absorb_threshold(&self) -> f32 {
        ABSORB_THRESHOLDS[self.absorb_threshold_index as usize % NUM_ABSORB_THRESHOLDS]
    }

    /// Default (balanced) config variant — arm 0.
    pub fn default_variant() -> Self {
        Self {
            rubric_preset: 0,
            epsilon_index: 0,
            template_hint: 0,
            absorb_threshold_index: 0,
        }
    }
}

impl Default for ConfigVariant {
    fn default() -> Self {
        Self::default_variant()
    }
}

// ── ReflectionScore ────────────────────────────────────────────

/// Scalar score computed from a [`ReflectionResult`], used as bandit reward.
///
/// The score combines verification rate and pair count into a single
/// signal that rewards high-quality reflections. If a [`ReviewMetrics`]
/// reference is available, `benefit_ratio` provides an alternative
/// quality signal.
#[derive(Clone, Copy, Debug)]
pub struct ReflectionScore(f32);

impl ReflectionScore {
    /// Compute score from a [`ReflectionResult`].
    ///
    /// Formula: `verification_rate * 0.7 + pair_signal * 0.3`
    /// where `pair_signal = min(pairs.len() / 50.0, 1.0)`.
    /// This rewards both verified correctness and reflection breadth.
    ///
    /// O(1) — only reads `verification_rate` and `pairs.len()`.
    pub fn from_reflection(result: &ReflectionResult) -> Self {
        let vr = result.verification_rate as f32;
        let pair_signal = (result.pairs.len() as f32 / 50.0).min(1.0);
        Self((vr * 0.7 + pair_signal * 0.3).clamp(0.0, 1.0))
    }

    /// Compute score from [`ReviewMetrics`] benefit ratio when no
    /// reflection is available. Falls back to a capped benefit ratio.
    ///
    /// Uses `min(benefit_ratio / 10.0, 1.0)` so infinite ratios
    /// map to 1.0.
    pub fn from_metrics(metrics: &ReviewMetrics) -> Self {
        let br = metrics.benefit_ratio() as f32;
        Self((br / 10.0).min(1.0))
    }

    /// Compute score preferring reflection, falling back to metrics.
    pub fn from_reflection_or_metrics(
        result: Option<&ReflectionResult>,
        metrics: Option<&ReviewMetrics>,
    ) -> Self {
        if let Some(r) = result {
            Self::from_reflection(r)
        } else if let Some(m) = metrics {
            Self::from_metrics(m)
        } else {
            Self(0.0)
        }
    }

    /// Raw scalar value in [0.0, 1.0].
    #[inline]
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Default for ReflectionScore {
    fn default() -> Self {
        Self(0.0)
    }
}

// ── ParetoConfigFrontier ───────────────────────────────────────

/// A single entry on the Pareto frontier.
#[derive(Clone, Copy, Debug)]
struct FrontierEntry {
    config: ConfigVariant,
    reward: f32,
    /// Lower is better — represents "cost" (inverse exploration efficiency).
    cost: f32,
    occupied: bool,
}

/// Fixed-size Pareto frontier of configuration variants.
///
/// Maintains up to [`MAX_CONFIGS`] non-dominated `(ConfigVariant, reward, cost)`
/// triples. A new variant dominates an existing one if it has equal-or-better
/// reward AND strictly lower cost, or strictly better reward AND equal-or-lower
/// cost. Variants that are dominated by any existing entry are rejected.
///
/// Uses fixed-size arrays (no heap allocation in hot path).
#[derive(Clone, Debug)]
pub struct ParetoConfigFrontier {
    entries: [FrontierEntry; MAX_CONFIGS],
    len: usize,
    /// Free-list stack of unoccupied slot indices. Avoids O(MAX_CONFIGS) linear scan.
    free_slots: Vec<usize>,
}

impl Default for ParetoConfigFrontier {
    fn default() -> Self {
        Self::new()
    }
}

impl ParetoConfigFrontier {
    /// Create an empty frontier.
    pub fn new() -> Self {
        let mut free_slots = Vec::with_capacity(MAX_CONFIGS);
        for i in (0..MAX_CONFIGS).rev() {
            free_slots.push(i);
        }
        Self {
            entries: [FrontierEntry {
                config: ConfigVariant::default(),
                reward: 0.0,
                cost: 0.0,
                occupied: false,
            }; MAX_CONFIGS],
            len: 0,
            free_slots,
        }
    }

    /// Number of entries currently on the frontier.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the frontier is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a variant. Returns `true` if it was added (non-dominated).
    ///
    /// After insertion, removes any entries dominated by the new one.
    /// If the frontier is full and the new entry is dominated, returns `false`.
    pub fn insert(&mut self, config: ConfigVariant, reward: f32, cost: f32) -> bool {
        // Check if dominated by any existing entry.
        for i in 0..MAX_CONFIGS {
            let e = &self.entries[i];
            if !e.occupied {
                continue;
            }
            // Dominated: existing has >= reward AND <= cost with at least one strict.
            let dominated =
                e.reward >= reward && e.cost <= cost && (e.reward > reward || e.cost < cost);
            if dominated {
                return false;
            }
        }

        // Remove entries dominated by the new one.
        for i in 0..MAX_CONFIGS {
            let e = &self.entries[i];
            if !e.occupied {
                continue;
            }
            // New dominates existing: reward >= existing AND cost <= existing,
            // with at least one strict.
            let new_dominates =
                reward >= e.reward && cost <= e.cost && (reward > e.reward || cost < e.cost);
            if new_dominates {
                self.entries[i].occupied = false;
                self.len -= 1;
                self.free_slots.push(i);
            }
        }

        // Find an empty slot via free-list (O(1)) instead of linear scan.
        let slot = self.free_slots.pop();
        match slot {
            Some(i) => {
                self.entries[i] = FrontierEntry {
                    config,
                    reward,
                    cost,
                    occupied: true,
                };
                self.len += 1;
                true
            }
            None => false, // Frontier full and no dominated entries to evict.
        }
    }

    /// Return the config with the highest reward on the frontier.
    ///
    /// Returns `None` if the frontier is empty.
    pub fn best(&self) -> Option<ConfigVariant> {
        let mut best: Option<&FrontierEntry> = None;
        for e in &self.entries {
            if !e.occupied {
                continue;
            }
            if best.is_none_or(|b| e.reward > b.reward) {
                best = Some(e);
            }
        }
        best.map(|e| e.config)
    }

    /// Return the lowest-cost config on the frontier.
    ///
    /// Useful for selecting the most efficient variant when reward
    /// differences are marginal.
    pub fn cheapest(&self) -> Option<ConfigVariant> {
        let mut best: Option<&FrontierEntry> = None;
        for e in &self.entries {
            if !e.occupied {
                continue;
            }
            if best.is_none_or(|b| e.cost < b.cost) {
                best = Some(e);
            }
        }
        best.map(|e| e.config)
    }

    /// Iterate over occupied entries as `(config, reward, cost)`.
    pub fn iter(&self) -> impl Iterator<Item = (ConfigVariant, f32, f32)> + '_ {
        self.entries
            .iter()
            .filter(|e| e.occupied)
            .map(|e| (e.config, e.reward, e.cost))
    }
}

// ── ReflectiveBanditPruner ─────────────────────────────────────

/// GEPA-D Reflective Config Evolution pruner.
///
/// Wraps a [`BanditPruner`] and treats each [`ConfigVariant`] as a bandit
/// arm. Reflection quality (via [`ReflectionScore`]) is the reward signal.
/// The internal [`ParetoConfigFrontier`] tracks non-dominated configs
/// so that `best_config()` always returns a Pareto-optimal configuration.
///
/// # Usage
///
/// ```rust,ignore
/// let inner = BanditPruner::new(base_pruner, BanditStrategy::Ucb1, NUM_ARMS);
/// let mut gepa = ReflectiveBanditPruner::new(inner);
///
/// // After reflection completes:
/// gepa.observe_reflection(arm, &reflection_result);
///
/// // Get best config for next episode:
/// let config = gepa.best_config();
/// ```
pub struct ReflectiveBanditPruner<P: ScreeningPruner> {
    /// Underlying bandit pruner that tracks Q-values per config arm.
    bandit: BanditPruner<P>,
    /// Pareto frontier of non-dominated configs.
    frontier: ParetoConfigFrontier,
    /// Q-values mirrored in fixed array for frontier cost computation.
    q_values: [f32; NUM_ARMS],
    /// Visit counts mirrored in fixed array.
    visits: [u32; NUM_ARMS],
    /// Total pulls across all arms.
    total_pulls: u32,
}

impl<P: ScreeningPruner> ReflectiveBanditPruner<P> {
    /// Create a new reflective bandit pruner wrapping the given bandit.
    ///
    /// The inner bandit should already be configured with [`BanditStrategy::Ucb1`]
    /// and `NUM_ARMS` arms for best results, but any strategy works.
    pub fn new(bandit: BanditPruner<P>) -> Self {
        Self {
            bandit,
            frontier: ParetoConfigFrontier::new(),
            q_values: [0.0; NUM_ARMS],
            visits: [0; NUM_ARMS],
            total_pulls: 0,
        }
    }

    /// Observe a reflection result for the given arm and feed it as reward.
    ///
    /// Computes [`ReflectionScore`], updates the bandit Q-value, and
    /// inserts the config variant into the Pareto frontier if it is
    /// non-dominated.
    pub fn observe_reflection(&mut self, arm: usize, result: &ReflectionResult) {
        let score = ReflectionScore::from_reflection(result);
        self.observe_reward(arm, score.value());
    }

    /// Observe a raw reward for the given arm and update tracking.
    ///
    /// Updates Q-value (incremental mean), mirrors into fixed arrays,
    /// and attempts Pareto frontier insertion.
    pub fn observe_reward(&mut self, arm: usize, reward: f32) {
        if arm >= NUM_ARMS {
            return;
        }
        self.bandit.update(arm, reward);

        // Mirror into fixed arrays.
        self.visits[arm] += 1;
        self.total_pulls += 1;
        let n = self.visits[arm] as f32;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;

        // Cost = 1.0 - epsilon (lower exploration cost is better).
        let config = ConfigVariant::from_arm(arm);
        let cost = 1.0 - config.epsilon();
        self.frontier.insert(config, self.q_values[arm], cost);
    }

    /// Return the best config from the Pareto frontier (highest reward).
    ///
    /// Falls back to UCB1-selected config if the frontier is empty.
    pub fn best_config(&self) -> ConfigVariant {
        self.frontier
            .best()
            .unwrap_or_else(|| ConfigVariant::from_arm(self.bandit.best_arm()))
    }

    /// Select the next config to try using UCB1.
    ///
    /// Unvisited arms get maximum priority. Ties broken by lower arm index.
    pub fn next_config(&self) -> ConfigVariant {
        let mut best_arm = 0;
        let mut best_score = self.ucb1_score(0);
        for arm in 1..NUM_ARMS {
            let score = self.ucb1_score(arm);
            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }
        ConfigVariant::from_arm(best_arm)
    }

    /// Select next config with random tie-breaking for diversity.
    pub fn next_config_seeded(&self, rng: &mut Rng) -> ConfigVariant {
        let mut best_arm = 0;
        let mut best_score = self.ucb1_score(0);
        for arm in 1..NUM_ARMS {
            let score = self.ucb1_score(arm);
            if score > best_score || (score == best_score && rng.uniform() > 0.5) {
                best_score = score;
                best_arm = arm;
            }
        }
        ConfigVariant::from_arm(best_arm)
    }

    /// UCB1 score for a given arm.
    ///
    /// Returns `f32::MAX` for unvisited arms.
    #[inline]
    fn ucb1_score(&self, arm: usize) -> f32 {
        let n = self.visits[arm];
        if n == 0 || self.total_pulls == 0 {
            return f32::MAX;
        }
        let q = self.q_values[arm];
        q + (UCB1_C * (self.total_pulls as f32).ln() / n as f32).sqrt()
    }

    /// Access the underlying bandit pruner.
    pub fn bandit(&self) -> &BanditPruner<P> {
        &self.bandit
    }

    /// Access the underlying bandit pruner (mutable).
    pub fn bandit_mut(&mut self) -> &mut BanditPruner<P> {
        &mut self.bandit
    }

    /// Access the Pareto frontier.
    pub fn frontier(&self) -> &ParetoConfigFrontier {
        &self.frontier
    }

    /// Q-value for a given arm.
    #[inline]
    pub fn q_value(&self, arm: usize) -> f32 {
        self.q_values.get(arm).copied().unwrap_or(0.0)
    }

    /// Visit count for a given arm.
    #[inline]
    pub fn visits(&self, arm: usize) -> u32 {
        self.visits.get(arm).copied().unwrap_or(0)
    }

    /// Total pulls across all arms.
    #[inline]
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }
}

impl<P: ScreeningPruner> ScreeningPruner for ReflectiveBanditPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Delegate to the inner bandit pruner's relevance computation.
        self.bandit.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandit::BanditStrategy;

    /// Helper: build an empty reflection result with given verification_rate
    /// and pair count.
    fn make_result(verification_rate: f64, num_pairs: usize) -> ReflectionResult {
        use super::super::reflection::{ReflectionDomain, ReflectionQA, ReflectionStep};
        let pairs: Vec<ReflectionQA> = (0..num_pairs)
            .map(|_| ReflectionQA {
                question: String::new(),
                answer: String::new(),
                step: ReflectionStep::DirectExtraction,
                domain: ReflectionDomain::Bomber,
                consolidation_count: 0,
                verified: verification_rate > 0.5,
            })
            .collect();
        ReflectionResult {
            pairs,
            step_counts: [0; 6],
            verification_rate,
        }
    }

    // ── ReflectionScore tests ────────────────────────────────

    #[test]
    fn test_known_reflection_expected_score() {
        // verification_rate = 0.8, 25 pairs → pair_signal = 25/50 = 0.5
        // score = 0.8 * 0.7 + 0.5 * 0.3 = 0.56 + 0.15 = 0.71
        let result = make_result(0.8, 25);
        let score = ReflectionScore::from_reflection(&result);
        let expected = 0.8 * 0.7 + 0.5 * 0.3;
        assert!(
            (score.value() - expected).abs() < 1e-6,
            "score = {}, expected = {}",
            score.value(),
            expected,
        );
    }

    #[test]
    fn test_perfect_reflection_scores_one() {
        // verification_rate = 1.0, 100 pairs → pair_signal = 1.0
        // score = 1.0 * 0.7 + 1.0 * 0.3 = 1.0
        let result = make_result(1.0, 100);
        let score = ReflectionScore::from_reflection(&result);
        assert!((score.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_reflection_scores_zero() {
        let result = make_result(0.0, 0);
        let score = ReflectionScore::from_reflection(&result);
        assert!((score.value() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_from_metrics() {
        let metrics = ReviewMetrics::new();
        // No observations: benefit_ratio = 0.0, score = 0.0
        let score = ReflectionScore::from_metrics(&metrics);
        assert!((score.value() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_from_reflection_or_metrics_prefers_reflection() {
        let result = make_result(0.9, 50);
        let score = ReflectionScore::from_reflection_or_metrics(Some(&result), None);
        assert!(score.value() > 0.0);
    }

    // ── ConfigVariant roundtrip tests ────────────────────────

    #[test]
    fn test_config_variant_arm_roundtrip() {
        for arm in 0..NUM_ARMS {
            let variant = ConfigVariant::from_arm(arm);
            assert_eq!(variant.to_arm(), arm, "roundtrip failed for arm {}", arm);
        }
    }

    #[test]
    fn test_default_variant_is_arm_zero() {
        let v = ConfigVariant::default();
        assert_eq!(v.to_arm(), 0);
    }

    #[test]
    fn test_rubric_weights_bounds() {
        let v = ConfigVariant::default();
        let w = v.rubric_weights();
        let sum: f32 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "weights should sum to 1.0, got {}",
            sum
        );
    }

    // ── ParetoConfigFrontier tests ───────────────────────────

    #[test]
    fn test_empty_frontier_best_returns_none() {
        let f = ParetoConfigFrontier::new();
        assert!(f.best().is_none());
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn test_dominated_variant_not_in_frontier() {
        let mut f = ParetoConfigFrontier::new();
        let c1 = ConfigVariant::default();
        let c2 = ConfigVariant {
            rubric_preset: 1,
            ..ConfigVariant::default()
        };

        // Insert high-reward, low-cost variant.
        assert!(f.insert(c1, 0.9, 0.1));
        // Insert lower-reward, higher-cost variant → dominated.
        assert!(!f.insert(c2, 0.5, 0.5));
        assert_eq!(f.len(), 1);
        assert_eq!(f.best(), Some(c1));
    }

    #[test]
    fn test_non_dominated_variant_expands_frontier() {
        let mut f = ParetoConfigFrontier::new();
        let c1 = ConfigVariant {
            rubric_preset: 0,
            ..ConfigVariant::default()
        };
        let c2 = ConfigVariant {
            rubric_preset: 1,
            ..ConfigVariant::default()
        };

        // c1: high reward, high cost
        assert!(f.insert(c1, 0.9, 0.8));
        // c2: medium reward, low cost — not dominated
        assert!(f.insert(c2, 0.6, 0.2));
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_best_returns_highest_reward() {
        let mut f = ParetoConfigFrontier::new();
        let c1 = ConfigVariant {
            rubric_preset: 0,
            ..ConfigVariant::default()
        };
        let c2 = ConfigVariant {
            rubric_preset: 1,
            ..ConfigVariant::default()
        };
        let c3 = ConfigVariant {
            rubric_preset: 2,
            ..ConfigVariant::default()
        };

        f.insert(c1, 0.5, 0.1);
        f.insert(c2, 0.9, 0.5);
        f.insert(c3, 0.7, 0.2);

        assert_eq!(f.best(), Some(c2));
    }

    #[test]
    fn test_cheapest_returns_lowest_cost() {
        let mut f = ParetoConfigFrontier::new();
        let c1 = ConfigVariant {
            rubric_preset: 0,
            ..ConfigVariant::default()
        };
        let c2 = ConfigVariant {
            rubric_preset: 1,
            ..ConfigVariant::default()
        };

        f.insert(c1, 0.9, 0.8);
        f.insert(c2, 0.6, 0.2);

        assert_eq!(f.cheapest(), Some(c2));
    }

    #[test]
    fn test_insert_removes_dominated_entries() {
        let mut f = ParetoConfigFrontier::new();
        let c1 = ConfigVariant {
            rubric_preset: 0,
            ..ConfigVariant::default()
        };
        let c2 = ConfigVariant {
            rubric_preset: 1,
            ..ConfigVariant::default()
        };
        let c3 = ConfigVariant {
            rubric_preset: 0,
            ..ConfigVariant::default()
        };

        // c1: medium reward, medium cost
        f.insert(c1, 0.6, 0.5);
        // c2: low reward, low cost — non-dominated
        f.insert(c2, 0.4, 0.2);
        assert_eq!(f.len(), 2);

        // c3 dominates c1: same reward, lower cost
        f.insert(c3, 0.6, 0.3);
        // c1 should be evicted
        assert_eq!(f.len(), 2);
        // Best should be c3 (reward 0.6) over c2 (reward 0.4)
        assert_eq!(f.best().unwrap().rubric_preset, 0);
    }

    // ── ReflectiveBanditPruner tests ─────────────────────────

    /// Minimal pruner that always returns 1.0 relevance.
    #[derive(Clone, Debug)]
    struct UnitPruner;

    impl ScreeningPruner for UnitPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    fn make_gepa() -> ReflectiveBanditPruner<UnitPruner> {
        let bp = BanditPruner::new(UnitPruner, BanditStrategy::Ucb1, NUM_ARMS);
        ReflectiveBanditPruner::new(bp)
    }

    #[test]
    fn test_good_reflection_arm_preferred() {
        let mut gepa = make_gepa();

        // Arm 0: good reflection
        let good_result = make_result(0.95, 40);
        gepa.observe_reflection(0, &good_result);

        // Arm 1: bad reflection
        let bad_result = make_result(0.1, 2);
        gepa.observe_reflection(1, &bad_result);

        // Both arms visited once, so next_config picks by Q-value.
        // Arm 0 should have higher Q.
        let q0 = gepa.q_value(0);
        let q1 = gepa.q_value(1);
        assert!(
            q0 > q1,
            "arm 0 (q={}) should be preferred over arm 1 (q={})",
            q0,
            q1,
        );

        // best_config should return arm 0's config.
        assert_eq!(gepa.best_config().to_arm(), 0);
    }

    #[test]
    fn test_frontier_populated_from_observations() {
        let mut gepa = make_gepa();

        let result = make_result(0.8, 30);
        gepa.observe_reflection(0, &result);

        assert!(!gepa.frontier().is_empty());
        assert_eq!(gepa.frontier().len(), 1);
    }

    #[test]
    fn test_next_config_explores_unvisited_arms() {
        let gepa = make_gepa();
        // No observations yet: all arms unvisited, UCB1 = MAX.
        // next_config should return arm 0 (first with max score).
        let config = gepa.next_config();
        assert_eq!(config.to_arm(), 0);
    }

    #[test]
    fn test_multiple_observations_converge() {
        let mut gepa = make_gepa();

        // Feed arm 0 good rewards 20 times.
        let good_result = make_result(0.9, 40);
        for _ in 0..20 {
            gepa.observe_reflection(0, &good_result);
        }

        // Feed arm 1 mediocre rewards 20 times.
        let mediocre_result = make_result(0.3, 5);
        for _ in 0..20 {
            gepa.observe_reflection(1, &mediocre_result);
        }

        // best_config should be arm 0.
        assert_eq!(gepa.best_config().to_arm(), 0);
        assert!(gepa.q_value(0) > gepa.q_value(1));
    }

    #[test]
    fn test_pareto_frontier_tracks_diverse_configs() {
        let mut gepa = make_gepa();

        // Arm 0: preset=0, eps_idx=0, template=0, absorb=0 → epsilon=0.05, cost=0.95
        let r0 = make_result(0.9, 40);
        gepa.observe_reflection(0, &r0);

        // Arm with different epsilon: arm 16 = preset=0, eps_idx=1, template=0, absorb=0
        // epsilon=0.10, cost=0.90 — different cost/reward tradeoff
        let r16 = make_result(0.7, 30);
        gepa.observe_reflection(16, &r16);

        // Both should be on frontier (non-dominated: arm 0 has higher reward but higher cost).
        assert!(gepa.frontier().len() >= 2);
    }

    #[test]
    fn test_total_pulls_tracking() {
        let mut gepa = make_gepa();
        assert_eq!(gepa.total_pulls(), 0);

        gepa.observe_reward(0, 0.5);
        assert_eq!(gepa.total_pulls(), 1);

        gepa.observe_reward(5, 0.8);
        assert_eq!(gepa.total_pulls(), 2);
    }

    #[test]
    fn test_observe_reward_out_of_bounds_ignored() {
        let mut gepa = make_gepa();
        gepa.observe_reward(NUM_ARMS, 1.0);
        assert_eq!(gepa.total_pulls(), 0);
    }

    #[test]
    fn test_relevance_delegates_to_inner_bandit() {
        let gepa = make_gepa();
        // UnitPruner returns 1.0, bandit wraps it.
        let rel = gepa.relevance(0, 0, &[]);
        // Cold start with no data → bandit returns domain relevance.
        assert_eq!(rel, 1.0);
    }
}
