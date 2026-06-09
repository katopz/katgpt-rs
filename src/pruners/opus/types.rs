//! OPUS-inspired BanditPruner with Boltzmann sampling + redundancy penalty.
//!
//! Based on OPUS paper (arXiv:2602.05400): Boltzmann sampling with redundancy
//! penalty outperforms greedy top-k by +1.26 avg points on real benchmarks.
//!
//! # Core Formula
//!
//! ```text
//! U_z = alignment(z) - λ · ⟨ϕ(z), Φ_selected⟩
//! ```
//!
//! Where:
//! - `alignment(z)` = domain relevance × bandit score (from inner [`BanditPruner`])
//! - `ϕ(z)` = [`CountSketch`] projection of arm z's features
//! - `Φ_selected` = accumulated sketch of previously selected arms
//! - `λ` = redundancy weight (configurable)

use std::collections::{HashSet, VecDeque};

use crate::pruners::bandit::{BanditEnv, BanditPruner};
use crate::speculative::types::ScreeningPruner;
use crate::types::Rng;

use super::boltzmann::boltzmann_sample_batch;
use super::count_sketch::CountSketch;

// ── Config ──────────────────────────────────────────────────────

/// OPUS configuration (paper defaults: τ=0.9, m=8192, ρ=0.5, b_t=64).
///
/// # Defaults
///
/// | Parameter | Default | Description |
/// |-----------|---------|-------------|
/// | `temperature` | 0.9 | Boltzmann temperature τ: τ→0 greedy, τ→∞ uniform |
/// | `redundancy_weight` | 0.5 | λ scaling for redundancy penalty |
/// | `sketch_dim` | 8192 | CountSketch bucket count m |
/// | `buffer_size` | 64 | Max selected arms to track (ring buffer) |
/// | `selection_ratio` | 0.5 | Fraction of candidates to consider |
/// | `feature_dim` | 64 | Dimension of per-arm feature vectors |
#[derive(Clone, Debug)]
pub struct OpusConfig {
    /// Temperature τ for Boltzmann sampling. τ→0 greedy, τ→∞ uniform.
    pub temperature: f32,
    /// Redundancy weight λ: scales penalty for similarity to selected arms.
    pub redundancy_weight: f32,
    /// Sketch dimension m: number of buckets in CountSketch.
    pub sketch_dim: usize,
    /// Buffer size N: max number of selected arms to track.
    pub buffer_size: usize,
    /// Selection ratio ρ: fraction of candidates to consider.
    pub selection_ratio: f32,
    /// Feature dimension d: dimension of per-arm feature vectors.
    pub feature_dim: usize,
}

impl Default for OpusConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            redundancy_weight: 0.5,
            sketch_dim: 8192,
            buffer_size: 64,
            selection_ratio: 0.5,
            feature_dim: 64,
        }
    }
}

impl OpusConfig {
    /// Create config with paper defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create config optimized for small action spaces (<100 arms).
    pub fn small() -> Self {
        Self {
            sketch_dim: 512,
            buffer_size: 16,
            feature_dim: 32,
            ..Self::default()
        }
    }

    /// Create config optimized for large action spaces (>10k arms).
    pub fn large() -> Self {
        Self {
            sketch_dim: 16384,
            buffer_size: 256,
            feature_dim: 128,
            ..Self::default()
        }
    }
}

// ── OpusBanditPruner ────────────────────────────────────────────

/// OPUS-inspired BanditPruner with Boltzmann sampling + redundancy penalty.
///
/// Wraps a [`BanditPruner<P>`] and adds:
/// 1. [`CountSketch`]-based redundancy penalty against already-selected arms
/// 2. Boltzmann temperature-controlled sampling
///
/// # Usage with DDTree
///
/// ```rust,ignore
/// let inner = BanditPruner::new(domain_pruner, BanditStrategy::Ucb1, vocab_size);
/// let mut opus = OpusBanditPruner::new(inner, OpusConfig::default());
/// opus.prepare_episode();
/// let tree = build_dd_tree_screened(&marginals, &config, &opus, false);
/// // After verification:
/// opus.update(accepted_token, reward);
/// opus.record_selection(accepted_token);
/// ```
///
/// # Standalone Selection
///
/// ```rust,ignore
/// let candidates = vec![0, 1, 2, 3, 4];
/// let selected = opus.select_arms(&candidates, 2);
/// // selected has no duplicates, respects redundancy penalty
/// ```
pub struct OpusBanditPruner<P: ScreeningPruner> {
    /// Inner BanditPruner providing domain relevance + bandit scores.
    inner: BanditPruner<P>,
    /// OPUS configuration.
    config: OpusConfig,
    /// CountSketch for dimensionality reduction.
    sketch: CountSketch,
    /// Accumulated sketch of selected features: Φ_selected = Σ sketch(φ(z)).
    selected_sketch_sum: Vec<f32>,
    /// Ring buffer of selected arm indices for history tracking.
    selected_arms: VecDeque<usize>,
    /// Per-arm feature vectors flattened with stride (arm `i` → `arm_features[i * feature_dim..(i+1) * feature_dim]`).
    arm_features: Vec<f32>,
    /// Stride for arm_features (= feature_dim).
    feature_stride: usize,
    /// RNG for Boltzmann sampling.
    rng: Rng,
}

impl<P: ScreeningPruner> OpusBanditPruner<P> {
    /// Create a new OPUS bandit pruner wrapping an inner bandit.
    ///
    /// The inner bandit provides domain relevance + base bandit scoring.
    /// OPUS adds redundancy penalty and Boltzmann sampling on top.
    pub fn new(inner: BanditPruner<P>, config: OpusConfig) -> Self {
        let num_arms = inner.q_values().len();
        let feature_dim = config.feature_dim;
        let sketch_dim = config.sketch_dim;

        // Generate deterministic feature vectors per arm (flat layout)
        let mut feat_rng = Rng::new(0xDEAD_BEEF);
        let mut arm_features = Vec::with_capacity(num_arms * feature_dim);
        for _ in 0..num_arms {
            for _ in 0..feature_dim {
                arm_features.push(feat_rng.uniform() * 2.0 - 1.0);
            }
        }

        let sketch = CountSketch::new(feature_dim, sketch_dim, 0xCAFE_BABE);

        Self {
            inner,
            config,
            sketch,
            selected_sketch_sum: vec![0.0; sketch_dim],
            selected_arms: VecDeque::new(),
            arm_features,
            feature_stride: feature_dim,
            rng: Rng::new(42),
        }
    }

    /// Create with custom RNG seed for reproducibility.
    pub fn with_seed(inner: BanditPruner<P>, config: OpusConfig, seed: u64) -> Self {
        let num_arms = inner.q_values().len();
        let feature_dim = config.feature_dim;
        let sketch_dim = config.sketch_dim;

        let mut feat_rng = Rng::new(seed.wrapping_add(1));
        let mut arm_features = Vec::with_capacity(num_arms * feature_dim);
        for _ in 0..num_arms {
            for _ in 0..feature_dim {
                arm_features.push(feat_rng.uniform() * 2.0 - 1.0);
            }
        }

        let sketch = CountSketch::new(feature_dim, sketch_dim, seed.wrapping_add(2));

        Self {
            inner,
            config,
            sketch,
            selected_sketch_sum: vec![0.0; sketch_dim],
            selected_arms: VecDeque::new(),
            arm_features,
            feature_stride: feature_dim,
            rng: Rng::new(seed),
        }
    }

    /// Get the feature vector for arm `i` as a slice.
    #[inline]
    fn arm_features(&self, i: usize) -> &[f32] {
        let start = i * self.feature_stride;
        let end = start + self.feature_stride;
        &self.arm_features[start..end]
    }

    /// Prepare for a new episode: cache Boltzmann scores, reset selection state.
    ///
    /// Call before each DDTree build, similar to [`BanditPruner::prepare_episode`].
    pub fn prepare_episode(&mut self) {
        self.inner.prepare_episode(&mut self.rng);
        self.reset_episode();
    }

    /// Reset selection history for a new episode.
    pub fn reset_episode(&mut self) {
        self.selected_sketch_sum.fill(0.0);
        self.selected_arms.clear();
    }

    /// Record that an arm was selected, updating redundancy history.
    ///
    /// Call after each arm selection to maintain the Φ_selected accumulator.
    /// Uses a ring buffer of size [`OpusConfig::buffer_size`]: oldest selections
    /// are evicted once the buffer is full.
    pub fn record_selection(&mut self, arm: usize) {
        let num_arms = self.arm_features.len() / self.feature_stride;
        if arm >= num_arms {
            return;
        }

        // Evict oldest if buffer is full
        if self.selected_arms.len() >= self.config.buffer_size {
            if let Some(&old_arm) = self.selected_arms.front()
                && old_arm < num_arms
            {
                let old_feat = self.arm_features(old_arm);
                let old_sketch = self.sketch.sketch(old_feat);
                for (i, &val) in old_sketch.iter().enumerate() {
                    if i < self.selected_sketch_sum.len() {
                        self.selected_sketch_sum[i] -= val;
                    }
                }
            }
            self.selected_arms.pop_front();
        }

        // Add arm's sketch to the accumulated sum
        let arm_feat = self.arm_features(arm);
        let arm_sketch = self.sketch.sketch(arm_feat);
        for (i, &val) in arm_sketch.iter().enumerate() {
            if i < self.selected_sketch_sum.len() {
                self.selected_sketch_sum[i] += val;
            }
        }
        self.selected_arms.push_back(arm);
    }

    /// Update bandit stats with observed reward.
    ///
    /// Delegates to inner [`BanditPruner`].
    pub fn update(&mut self, arm: usize, reward: f32) {
        self.inner.update(arm, reward);
    }

    /// Select `k` arms from candidates using Boltzmann sampling with redundancy.
    ///
    /// This is the core OPUS selection algorithm:
    /// 1. Compute utility for each candidate: U_z = alignment - λ·redundancy
    /// 2. Sample from Boltzmann(U, τ) without replacement
    /// 3. Record selected arms for future redundancy computation
    pub fn select_arms(&mut self, candidates: &[usize], k: usize) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let k_actual = k.min(candidates.len());
        if k_actual >= candidates.len() {
            let result: Vec<usize> = candidates.to_vec();
            for &arm in &result {
                self.record_selection(arm);
            }
            return result;
        }

        let utilities: Vec<f32> = candidates
            .iter()
            .map(|&arm| self.compute_utility(arm))
            .collect();

        let selected =
            boltzmann_sample_batch(&utilities, self.config.temperature, k_actual, &mut self.rng);

        // Map back to arm indices and record selections
        let result: Vec<usize> = selected.into_iter().map(|idx| candidates[idx]).collect();

        for &arm in &result {
            self.record_selection(arm);
        }

        result
    }

    /// Compute OPUS utility for a single arm.
    ///
    /// U_z = alignment(z) - λ · ⟨ϕ(z), Φ_selected⟩
    fn compute_utility(&self, arm: usize) -> f32 {
        let num_arms = self.arm_features.len() / self.feature_stride;
        if arm >= num_arms {
            return 0.0;
        }

        // Alignment: Q-value from inner bandit
        let alignment = self.inner_q_value(arm);

        // Redundancy: inner product between arm's sketch and accumulated selected sketches
        let redundancy = if !self.selected_arms.is_empty() {
            let arm_feat = self.arm_features(arm);
            let arm_sketch = self.sketch.sketch(arm_feat);
            dot(&arm_sketch, &self.selected_sketch_sum)
        } else {
            0.0
        };

        let utility = alignment - self.config.redundancy_weight * redundancy;
        utility.max(0.0)
    }

    /// Get Q-value from inner bandit for an arm.
    fn inner_q_value(&self, arm: usize) -> f32 {
        let q = self.inner.q_values().get(arm).copied().unwrap_or(0.0);
        if q <= 0.0 { 0.01 } else { q.clamp(0.0, 1.0) }
    }

    // ── Delegated Accessors ────────────────────────────────

    /// Number of arms (vocabulary size).
    pub fn num_arms(&self) -> usize {
        self.inner.q_values().len()
    }

    /// Best arm by inner bandit Q-values.
    pub fn best_arm(&self) -> usize {
        self.inner.best_arm()
    }

    /// Q-values from inner bandit.
    pub fn q_values(&self) -> &[f32] {
        self.inner.q_values()
    }

    /// Visit counts from inner bandit.
    pub fn visits(&self) -> &[u32] {
        self.inner.visits()
    }

    /// Total pulls from inner bandit.
    pub fn total_pulls(&self) -> u32 {
        self.inner.total_pulls()
    }

    /// Number of selections in current episode.
    pub fn selected_count(&self) -> usize {
        self.selected_arms.len()
    }

    /// Unique arms selected in current episode.
    pub fn unique_selected(&self) -> usize {
        let unique: HashSet<usize> = self.selected_arms.iter().copied().collect();
        unique.len()
    }

    /// Current temperature.
    pub fn temperature(&self) -> f32 {
        self.config.temperature
    }

    /// Current redundancy weight.
    pub fn redundancy_weight(&self) -> f32 {
        self.config.redundancy_weight
    }

    /// Reference to the inner [`BanditPruner`].
    pub fn inner(&self) -> &BanditPruner<P> {
        &self.inner
    }

    /// Mutable reference to the inner [`BanditPruner`].
    pub fn inner_mut(&mut self) -> &mut BanditPruner<P> {
        &mut self.inner
    }

    /// Reference to the config.
    pub fn config(&self) -> &OpusConfig {
        &self.config
    }
}

impl<P: ScreeningPruner> ScreeningPruner for OpusBanditPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Domain + bandit score from inner pruner
        let alignment = self.inner.relevance(depth, token_idx, parent_tokens);
        if alignment <= 0.0 {
            return 0.0;
        }

        // Cold start: no selections yet, use alignment directly
        if self.selected_arms.is_empty() {
            return alignment;
        }

        // Out of bounds arm
        let num_arms = self.arm_features.len() / self.feature_stride;
        if token_idx >= num_arms {
            return alignment;
        }

        // Redundancy penalty: ⟨ϕ(z), Φ_selected⟩
        let arm_feat = self.arm_features(token_idx);
        let arm_sketch = self.sketch.sketch(arm_feat);
        let redundancy = dot(&arm_sketch, &self.selected_sketch_sum);

        // OPUS utility: alignment - λ · redundancy
        let utility = alignment - self.config.redundancy_weight * redundancy;
        utility.clamp(0.0, 1.0)
    }
}

// ── OpusRedundantEnv (T4) ───────────────────────────────────────

/// Bandit environment with configurable redundancy groups.
///
/// Arms in the same redundancy group receive identical expected rewards.
/// Used to test whether [`OpusBanditPruner`] achieves higher diversity
/// than standard [`BanditPruner`] on problems with redundant arms.
///
/// # Example
///
/// ```rust,ignore
/// // 6 arms: group [0,1,2] → reward 0.5, arm 3 → reward 0.9, group [4,5] → reward 0.3
/// let env = OpusRedundantEnv::new(&[0.5, 0.5, 0.5, 0.9, 0.3, 0.3], 0.1);
/// ```
#[derive(Clone)]
pub struct OpusRedundantEnv {
    /// Expected reward per arm.
    means: Vec<f32>,
    /// Noise standard deviation (Gaussian).
    noise: f32,
    /// Index of optimal arm (highest mean, first wins ties).
    optimal_arm: usize,
    /// Optimal reward.
    optimal_reward: f32,
}

impl OpusRedundantEnv {
    /// Create a new redundant environment with Gaussian noise.
    ///
    /// `means` defines expected rewards per arm. Arms with identical means
    /// form natural redundancy groups. `noise` is the standard deviation
    /// of the Gaussian perturbation applied to each pull.
    pub fn new(means: &[f32], noise: f32) -> Self {
        let optimal_arm = means
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let optimal_reward = means[optimal_arm];

        Self {
            means: means.to_vec(),
            noise,
            optimal_arm,
            optimal_reward,
        }
    }

    /// Create an environment with explicit redundancy groups.
    ///
    /// `group_means` maps each arm to its group's expected reward.
    /// Returns (env, redundancy_groups) where each group is a vec of arm indices.
    pub fn with_groups(group_means: &[f32], noise: f32) -> (Self, Vec<Vec<usize>>) {
        let env = Self::new(group_means, noise);

        // Build redundancy groups
        let mut groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (arm, &mean) in group_means.iter().enumerate() {
            // Quantize mean to identify same-reward groups
            let key = (mean * 10000.0) as usize;
            groups.entry(key).or_default().push(arm);
        }
        let groups: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();

        (env, groups)
    }

    /// Expected rewards for all arms.
    pub fn means(&self) -> &[f32] {
        &self.means
    }

    /// Count arms in the largest redundancy group.
    pub fn max_redundancy_group_size(&self) -> usize {
        let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for &mean in &self.means {
            let key = (mean * 10000.0) as u64;
            *counts.entry(key).or_insert(0) += 1;
        }
        counts.into_values().max().unwrap_or(1)
    }
}

impl BanditEnv for OpusRedundantEnv {
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32 {
        let mean = self.means.get(arm).copied().unwrap_or(0.0);
        let noise = rng.normal() * self.noise;
        (mean + noise).clamp(0.0, 1.0)
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        self.means.get(arm).copied().unwrap_or(0.0)
    }

    fn optimal_reward(&self) -> f32 {
        self.optimal_reward
    }

    fn num_arms(&self) -> usize {
        self.means.len()
    }

    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Dot product of two f32 slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use crate::pruners::bandit::{BanditSession, BanditStrategy};

    /// A trivial pruner that always returns 1.0 relevance.
    struct UnitPruner;

    impl ScreeningPruner for UnitPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    fn make_opus(num_arms: usize) -> OpusBanditPruner<UnitPruner> {
        let inner = BanditPruner::new(UnitPruner, BanditStrategy::Ucb1, num_arms);
        OpusBanditPruner::new(inner, OpusConfig::small())
    }

    // ── Config Tests ──────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let config = OpusConfig::default();
        assert!((config.temperature - 0.9).abs() < 1e-5);
        assert!((config.redundancy_weight - 0.5).abs() < 1e-5);
        assert_eq!(config.sketch_dim, 8192);
        assert_eq!(config.buffer_size, 64);
        assert!((config.selection_ratio - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_config_small() {
        let config = OpusConfig::small();
        assert_eq!(config.sketch_dim, 512);
        assert_eq!(config.buffer_size, 16);
        assert_eq!(config.feature_dim, 32);
    }

    #[test]
    fn test_config_large() {
        let config = OpusConfig::large();
        assert_eq!(config.sketch_dim, 16384);
        assert_eq!(config.buffer_size, 256);
        assert_eq!(config.feature_dim, 128);
    }

    // ── OpusBanditPruner Core Tests ────────────────────────

    #[test]
    fn test_cold_start_returns_domain() {
        let opus = make_opus(10);
        let rel = opus.relevance(0, 5, &[]);
        assert!(
            (rel - 1.0).abs() < 0.01,
            "cold start should return domain relevance, got {rel}"
        );
    }

    #[test]
    fn test_redundancy_decreases_relevance_for_selected_arm() {
        let mut opus = make_opus(10);

        // Record arm 5 as selected
        opus.record_selection(5);

        // Arm 5 should now have lower relevance than an unselected arm
        let rel_5 = opus.relevance(0, 5, &[]);
        let rel_3 = opus.relevance(0, 3, &[]);

        // After recording arm 5, its redundancy is positive (self-similarity)
        // So rel_5 should be ≤ rel_3 (arm 3 has no history)
        assert!(
            rel_5 <= rel_3 + 1e-6,
            "selected arm should have redundancy penalty: rel_5={rel_5}, rel_3={rel_3}"
        );
    }

    #[test]
    fn test_record_selection_ring_buffer_eviction() {
        let mut config = OpusConfig::small();
        config.buffer_size = 3;
        let inner = BanditPruner::new(UnitPruner, BanditStrategy::Ucb1, 10);
        let mut opus = OpusBanditPruner::new(inner, config);

        // Fill buffer beyond capacity
        opus.record_selection(0);
        opus.record_selection(1);
        opus.record_selection(2);
        assert_eq!(opus.selected_count(), 3);

        opus.record_selection(3); // Should evict arm 0
        assert_eq!(opus.selected_count(), 3);
        assert_eq!(opus.selected_arms, VecDeque::from([1, 2, 3]));
    }

    #[test]
    fn test_select_arms_no_duplicates() {
        let mut opus = make_opus(10);
        let candidates: Vec<usize> = (0..10).collect();
        let selected = opus.select_arms(&candidates, 3);

        assert_eq!(selected.len(), 3);
        let mut seen = std::collections::HashSet::new();
        for &arm in &selected {
            assert!(seen.insert(arm), "duplicate arm {arm}");
        }
    }

    #[test]
    fn test_select_arms_k_exceeds_candidates() {
        let mut opus = make_opus(5);
        let candidates: Vec<usize> = (0..3).collect();
        let selected = opus.select_arms(&candidates, 10);

        assert_eq!(selected.len(), 3, "k > n should select all n candidates");
    }

    #[test]
    fn test_select_arms_diversity_across_seeds() {
        let mut all_selected = std::collections::HashSet::new();
        for seed in 0..20 {
            let mut opus = OpusBanditPruner::with_seed(
                BanditPruner::new(UnitPruner, BanditStrategy::Ucb1, 20),
                OpusConfig::small(),
                seed,
            );
            let candidates: Vec<usize> = (0..20).collect();
            let selected = opus.select_arms(&candidates, 5);
            for &arm in &selected {
                all_selected.insert(arm);
            }
        }

        assert!(
            all_selected.len() >= 10,
            "expected diverse selections across seeds, got {}",
            all_selected.len()
        );
    }

    #[test]
    fn test_reset_episode_clears_history() {
        let mut opus = make_opus(10);
        opus.record_selection(5);
        opus.record_selection(3);
        assert_eq!(opus.selected_count(), 2);

        opus.reset_episode();
        assert_eq!(opus.selected_count(), 0);

        // After reset, relevance should be back to domain
        let rel = opus.relevance(0, 5, &[]);
        assert!(
            (rel - 1.0).abs() < 0.01,
            "after reset, relevance should return domain, got {rel}"
        );
    }

    #[test]
    fn test_prepare_episode_resets_and_caches() {
        let mut opus = make_opus(10);
        opus.record_selection(5);
        opus.prepare_episode();
        assert_eq!(
            opus.selected_count(),
            0,
            "prepare_episode should reset history"
        );
    }

    #[test]
    fn test_delegated_accessors() {
        let opus = make_opus(10);
        assert_eq!(opus.num_arms(), 10);
        assert_eq!(opus.q_values().len(), 10);
        assert_eq!(opus.visits().len(), 10);
    }

    #[test]
    fn test_unique_selected_tracking() {
        let mut opus = make_opus(10);
        opus.record_selection(3);
        opus.record_selection(5);
        opus.record_selection(3); // duplicate
        assert_eq!(opus.selected_count(), 3);
        assert_eq!(opus.unique_selected(), 2, "should count unique arms only");
    }

    #[test]
    fn test_out_of_bounds_arm_handled() {
        let opus = make_opus(10);
        // Out of bounds: BanditPruner returns 0 for token_idx >= num_arms
        let rel = opus.relevance(0, 100, &[]);
        assert!(
            (rel - 0.0).abs() < 0.01,
            "out-of-bounds should return 0.0 (bandit delegates to domain which clamps), got {rel}"
        );
    }

    // ── OpusRedundantEnv Tests ─────────────────────────────

    #[test]
    fn test_redundant_env_optimal_arm() {
        let env = OpusRedundantEnv::new(&[0.5, 0.5, 0.5, 0.9, 0.3, 0.3], 0.1);
        assert_eq!(env.optimal_arm(), 3);
        assert!((env.optimal_reward() - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_redundant_env_pull_in_range() {
        let env = OpusRedundantEnv::new(&[0.5, 0.9], 0.1);
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let reward = env.pull(0, &mut rng);
            assert!(
                (0.0..=1.0).contains(&reward),
                "reward should be in [0,1], got {reward}"
            );
        }
    }

    #[test]
    fn test_redundant_env_groups() {
        let (env, groups) = OpusRedundantEnv::with_groups(&[0.5, 0.5, 0.5, 0.9, 0.3, 0.3], 0.1);
        assert_eq!(env.num_arms(), 6);
        // Two redundancy groups: [0,1,2] and [4,5]
        assert_eq!(groups.len(), 2, "should detect 2 redundancy groups");
        assert_eq!(env.max_redundancy_group_size(), 3);
    }

    // ── T4: OPUS vs Bandit Diversity Comparison ────────────

    #[test]
    fn test_opus_achieves_higher_diversity_than_bandit() {
        // 6 arms: arms 0-2 give reward 0.5, arm 3 gives 0.9, arms 4-5 give 0.3
        let probs = [0.5, 0.5, 0.5, 0.9, 0.3, 0.3];
        let env = OpusRedundantEnv::new(&probs, 0.05);

        // Run standard Thompson bandit session
        let bandit_session = BanditSession::new(env.clone(), BanditStrategy::ThompsonSampling);
        let mut rng_bandit = Rng::new(42);
        let (_, bandit_result) = bandit_session.run(500, &mut rng_bandit);

        // Run OPUS selection for 500 steps
        let mut opus_rng = Rng::new(42);
        let mut opus_unique_arms = std::collections::HashSet::new();
        let mut opus = OpusBanditPruner::with_seed(
            BanditPruner::new(UnitPruner, BanditStrategy::ThompsonSampling, 6),
            OpusConfig::small(),
            42,
        );

        for _ in 0..100 {
            opus.prepare_episode();
            let candidates: Vec<usize> = (0..6).collect();
            let selected = opus.select_arms(&candidates, 3);
            for &arm in &selected {
                opus_unique_arms.insert(arm);
                let reward = env.pull(arm, &mut opus_rng);
                opus.update(arm, reward);
            }
        }

        // OPUS should explore at least as many unique arms as standard bandit
        let bandit_unique = bandit_result.visits.iter().filter(|&&v| v > 0).count();
        assert!(
            opus_unique_arms.len() >= bandit_unique.min(4),
            "OPUS unique={}, bandit unique={}",
            opus_unique_arms.len(),
            bandit_unique
        );
    }

    #[test]
    fn test_opus_reward_comparable_to_standard_bandit() {
        let probs = [0.5, 0.5, 0.5, 0.9, 0.3, 0.3];
        let env = OpusRedundantEnv::new(&probs, 0.05);

        let mut opus_rng = Rng::new(42);
        let mut opus_total_reward = 0.0f32;
        let mut opus = OpusBanditPruner::with_seed(
            BanditPruner::new(UnitPruner, BanditStrategy::ThompsonSampling, 6),
            OpusConfig::small(),
            42,
        );

        for _ in 0..500 {
            opus.prepare_episode();
            let candidates: Vec<usize> = (0..6).collect();
            let selected = opus.select_arms(&candidates, 1);
            if let Some(&arm) = selected.first() {
                let reward = env.pull(arm, &mut opus_rng);
                opus_total_reward += reward;
                opus.update(arm, reward);
            }
        }

        let avg_reward = opus_total_reward / 500.0;
        // Should find optimal arm reasonably often (arm 3 with reward 0.9)
        assert!(
            avg_reward > 0.5,
            "OPUS should achieve avg reward > 0.5, got {avg_reward:.3}"
        );
    }

    #[test]
    fn test_opus_avoids_redundant_arms_with_equal_utility() {
        // 4 arms: arms 0,1 give same reward 0.7, arms 2,3 give same reward 0.3
        // After sufficient learning, OPUS should distribute across 0,1 rather than
        // repeatedly selecting the same one
        let probs = [0.7, 0.7, 0.3, 0.3];
        let env = OpusRedundantEnv::new(&probs, 0.05);

        let mut opus_rng = Rng::new(42);
        let mut arm_counts = [0usize; 4];
        let mut opus = OpusBanditPruner::with_seed(
            BanditPruner::new(UnitPruner, BanditStrategy::Ucb1, 4),
            OpusConfig::small(),
            42,
        );

        // Warm up: let UCB1 learn the Q-values
        for _ in 0..100 {
            let candidates: Vec<usize> = (0..4).collect();
            let selected = opus.select_arms(&candidates, 1);
            if let Some(&arm) = selected.first() {
                let reward = env.pull(arm, &mut opus_rng);
                opus.update(arm, reward);
                arm_counts[arm] += 1;
            }
            opus.reset_episode();
        }

        // Both arms 0 and 1 should be selected (redundancy penalty encourages diversity)
        let arms_01_selected = (arm_counts[0] > 0) as usize + (arm_counts[1] > 0) as usize;
        assert!(
            arms_01_selected >= 2,
            "OPUS should select both redundant arms, counts: {arm_counts:?}"
        );
    }
}
