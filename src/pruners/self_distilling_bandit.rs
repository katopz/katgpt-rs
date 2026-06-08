//! Self-Distilling Pruner Bandit — Episode-Guided Arm Selection (Plan 208).
//!
//! Wraps any `BanditPruner<P>` with episode-guided reward signal.
//! After generation, compares output to episode reference, computes match reward,
//! blends with acceptance reward, and updates the inner bandit.
//!
//! # Architecture
//!
//! ```text
//! DDTree → SelfDistillingBandit → inner BanditPruner.relevance()
//!                      │
//!                EpisodeLookup → reference?
//!                      │ yes
//!                EpisodeRewardComputer → match_ratio
//!                      │
//!                combined_reward → inner.update()
//! ```
//!
//! Zero cost on miss path (no episode → delegate directly to inner).
//! Feature-gated behind `self_distilling_bandit`.

use super::bandit::{BanditPruner, BanditStats};
use super::episode_pruner::EpisodeLookup;
use crate::speculative::types::ScreeningPruner;
use crate::types::Rng;

// ── Config ──────────────────────────────────────────────────────

/// Configuration for the self-distilling bandit.
#[derive(Clone, Debug)]
pub struct SelfDistillingConfig {
    /// Blend factor: `(1 - alpha) * episode_reward + alpha * acceptance_reward`.
    /// Higher alpha → more weight on acceptance signal.
    /// Default: 0.3 (70% episode, 30% acceptance when episode exists).
    pub alpha: f32,
    /// Steepness of sigmoid reward curve: `sigmoid(k * (match_ratio - 0.5))`.
    /// Higher k → sharper reward transition around 50% match.
    /// Default: 4.0.
    pub k: f32,
    /// Minimum samples per domain before using domain-specific Q-values.
    /// Below this threshold, falls back to global Q-values.
    /// Default: 10.
    pub min_domain_samples: usize,
    /// Maximum number of domain buckets.
    /// Default: 64.
    pub max_domains: usize,
    /// Rolling window size for convergence tracking.
    /// Default: 100.
    pub convergence_window: usize,
}

impl Default for SelfDistillingConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            k: 4.0,
            min_domain_samples: 10,
            max_domains: 64,
            convergence_window: 100,
        }
    }
}

// ── Reward Computation ──────────────────────────────────────────

/// Computes episode-guided reward from match quality.
///
/// Reward formula: `sigmoid(k * (match_ratio - 0.5))` for episode component,
/// blended with acceptance reward via alpha.
pub struct EpisodeRewardComputer {
    k: f32,
    alpha: f32,
}

impl EpisodeRewardComputer {
    /// Create a new reward computer with the given config.
    pub fn new(config: &SelfDistillingConfig) -> Self {
        Self {
            k: config.k,
            alpha: config.alpha,
        }
    }

    /// Compute combined reward from episode match and acceptance signal.
    ///
    /// - `match_ratio`: fraction of tokens matching reference (0.0..1.0)
    /// - `acceptance_reward`: the usual acceptance signal (0.0 or 1.0)
    ///
    /// Returns blended reward in (0.0, 1.0).
    #[inline]
    pub fn compute_reward(&self, match_ratio: f32, acceptance_reward: f32) -> f32 {
        let episode_reward = sigmoid(self.k * (match_ratio - 0.5));
        (1.0 - self.alpha) * episode_reward + self.alpha * acceptance_reward
    }

    /// Compute pure episode reward (no acceptance blend).
    #[inline]
    pub fn episode_reward_only(&self, match_ratio: f32) -> f32 {
        sigmoid(self.k * (match_ratio - 0.5))
    }
}

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
/// Monotonic, bounded (0, 1). Used instead of softmax per project rules.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute match ratio between generated and reference token sequences.
///
/// Returns fraction of matching positions. Sequences of different lengths
/// are compared up to the shorter length, with mismatches for the tail.
#[inline]
pub fn compute_match_ratio(generated: &[usize], reference: &[usize]) -> f32 {
    if generated.is_empty() && reference.is_empty() {
        return 1.0;
    }
    if generated.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let min_len = generated.len().min(reference.len());
    let max_len = generated.len().max(reference.len());
    let matching = generated[..min_len]
        .iter()
        .zip(reference[..min_len].iter())
        .filter(|(a, b)| a == b)
        .count();
    matching as f32 / max_len as f32
}

// ── Domain Q-Table ──────────────────────────────────────────────

/// Per-domain Q-value tracking.
///
/// Routes arm selections to domain-specific Q-values when sufficient
/// samples exist. Falls back to global Q-values for cold domains.
struct DomainQTable {
    /// Per-domain arm stats. Key = domain hash % max_domains.
    domains: Vec<Option<BanditStats>>,
    /// Global stats (fallback for cold domains).
    global: BanditStats,
    /// Number of arms per domain.
    num_arms: usize,
    /// Minimum samples before using domain-specific values.
    min_domain_samples: usize,
}

impl DomainQTable {
    fn new(num_arms: usize, max_domains: usize, min_domain_samples: usize) -> Self {
        let mut domains = Vec::with_capacity(max_domains);
        for _ in 0..max_domains {
            domains.push(None);
        }
        Self {
            domains,
            global: BanditStats::new(num_arms),
            num_arms,
            min_domain_samples,
        }
    }

    /// Get the effective Q-values for a domain.
    /// Returns domain-specific values if warm enough, else global.
    fn q_values_for_domain(&self, domain_hash: u64) -> &[f32] {
        let slot = (domain_hash as usize) % self.domains.len();
        match &self.domains[slot] {
            Some(stats) if u64::from(stats.total_pulls()) >= self.min_domain_samples as u64 => {
                stats.q_values()
            }
            _ => self.global.q_values(),
        }
    }

    /// Update reward for an arm in a specific domain.
    fn update(&mut self, domain_hash: u64, arm: usize, reward: f32) {
        // Always update global
        self.global.update(arm, reward);

        // Update domain-specific if slot is claimed by same domain
        let slot = (domain_hash as usize) % self.domains.len();
        match &mut self.domains[slot] {
            Some(stats) => {
                stats.update(arm, reward);
            }
            None => {
                let mut stats = BanditStats::new(self.num_arms);
                stats.update(arm, reward);
                self.domains[slot] = Some(stats);
            }
        }
    }

    /// Get global Q-values.
    #[allow(dead_code)]
    fn global_q_values(&self) -> &[f32] {
        self.global.q_values()
    }

    /// Number of domains with sufficient samples.
    fn warm_domain_count(&self) -> usize {
        self.domains
            .iter()
            .filter(|d| {
                d.as_ref()
                    .map(|s| u64::from(s.total_pulls()) >= self.min_domain_samples as u64)
                    .unwrap_or(false)
            })
            .count()
    }
}

// ── Convergence Metrics ─────────────────────────────────────────

/// Rolling convergence metrics for monitoring bandit learning.
#[derive(Clone, Debug, Default)]
pub struct ConvergenceMetrics {
    /// Rolling average of combined rewards.
    pub avg_reward: f32,
    /// Fraction of updates that found an episode (hit rate).
    pub episode_hit_rate: f32,
    /// Entropy of arm selection distribution (lower = more confident).
    pub arm_entropy: f32,
    /// Number of total updates.
    pub total_updates: usize,
    /// Number of warm domains.
    pub warm_domains: usize,
}

// ── SelfDistillingBandit ────────────────────────────────────────

/// Self-distilling bandit that learns pruner configurations from episode outcomes.
///
/// Wraps an inner `BanditPruner<P>` and adds episode-guided reward signal.
/// After generation, compares output to reference, computes match reward,
/// and updates the inner bandit with a blended reward.
///
/// # Type Parameters
///
/// - `P`: Inner `ScreeningPruner` wrapped by the bandit.
/// - `L`: Episode lookup backend implementing `EpisodeLookup`.
pub struct SelfDistillingBandit<P: ScreeningPruner, L: EpisodeLookup> {
    /// Inner bandit pruner that handles arm selection.
    inner: BanditPruner<P>,
    /// Episode lookup backend.
    lookup: L,
    /// Reward computation parameters.
    reward_computer: EpisodeRewardComputer,
    /// Domain-keyed Q-table.
    domain_q: DomainQTable,
    /// Rolling reward history for convergence tracking.
    reward_history: Vec<f32>,
    /// Episode hit counter.
    episode_hits: usize,
    /// Total update counter.
    total_updates: usize,
    /// Config.
    config: SelfDistillingConfig,
}

impl<P: ScreeningPruner, L: EpisodeLookup> SelfDistillingBandit<P, L> {
    /// Create a new self-distilling bandit.
    ///
    /// - `inner`: The `BanditPruner` to wrap.
    /// - `lookup`: Episode lookup backend.
    /// - `config`: Configuration (use `SelfDistillingConfig::default()` for defaults).
    pub fn new(inner: BanditPruner<P>, lookup: L, config: SelfDistillingConfig) -> Self {
        let num_arms = inner.q_values().len();
        Self {
            inner,
            lookup,
            reward_computer: EpisodeRewardComputer::new(&config),
            domain_q: DomainQTable::new(num_arms, config.max_domains, config.min_domain_samples),
            reward_history: Vec::with_capacity(config.convergence_window),
            episode_hits: 0,
            total_updates: 0,
            config,
        }
    }

    /// Episode-guided reward update.
    ///
    /// After generation, call this with the prompt hash and generated token sequence.
    /// Looks up the episode, computes match reward, blends with acceptance reward,
    /// and updates both the inner bandit and domain Q-table.
    ///
    /// - `prompt_hash`: Hash identifying the prompt.
    /// - `arm`: The arm (token index) that was selected.
    /// - `generated`: The generated token sequence.
    /// - `acceptance_reward`: Binary acceptance signal (1.0 = accepted, 0.0 = rejected).
    /// - `domain_hash`: Hash identifying the domain/problem type (0 = global).
    pub fn episode_update(
        &mut self,
        prompt_hash: u64,
        arm: usize,
        generated: &[usize],
        acceptance_reward: f32,
        domain_hash: u64,
    ) {
        let combined_reward = match self.lookup.lookup(prompt_hash) {
            Some(episode) => {
                self.episode_hits += 1;
                let match_ratio = compute_match_ratio(generated, &episode.reference_tokens);
                self.reward_computer
                    .compute_reward(match_ratio, acceptance_reward)
            }
            None => {
                // No episode → pure acceptance reward (zero regression)
                acceptance_reward
            }
        };

        // Update inner bandit
        self.inner.update(arm, combined_reward);

        // Update domain Q-table
        self.domain_q.update(domain_hash, arm, combined_reward);

        // Track convergence
        self.total_updates += 1;
        self.reward_history.push(combined_reward);
        if self.reward_history.len() > self.config.convergence_window {
            self.reward_history.remove(0);
        }
    }

    /// Batch episode update for multiple arms at once.
    ///
    /// Convenience method for updating multiple arms from the same generation.
    pub fn episode_update_batch(
        &mut self,
        prompt_hash: u64,
        arms: &[usize],
        generated: &[usize],
        acceptance_reward: f32,
        domain_hash: u64,
    ) {
        for &arm in arms {
            self.episode_update(prompt_hash, arm, generated, acceptance_reward, domain_hash);
        }
    }

    /// Prepare for a new episode. Delegates to inner bandit.
    pub fn prepare_episode(&mut self, rng: &mut Rng) {
        self.inner.prepare_episode(rng);
    }

    /// Get the best arm (highest Q-value) for a specific domain.
    ///
    /// Returns domain-specific best arm if warm, else global best.
    pub fn best_arm_for_domain(&self, domain_hash: u64) -> usize {
        let q_values = self.domain_q.q_values_for_domain(domain_hash);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get convergence metrics for monitoring.
    pub fn convergence_metrics(&self) -> ConvergenceMetrics {
        let avg_reward = if self.reward_history.is_empty() {
            0.0
        } else {
            self.reward_history.iter().sum::<f32>() / self.reward_history.len() as f32
        };

        let episode_hit_rate = if self.total_updates == 0 {
            0.0
        } else {
            self.episode_hits as f32 / self.total_updates as f32
        };

        // Compute arm entropy from global Q-values
        let q = self.inner.q_values();
        let total_q: f32 = q.iter().sum::<f32>().max(1e-10);
        let arm_entropy = q
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / total_q;
                -p * p.ln()
            })
            .sum();

        ConvergenceMetrics {
            avg_reward,
            episode_hit_rate,
            arm_entropy,
            total_updates: self.total_updates,
            warm_domains: self.domain_q.warm_domain_count(),
        }
    }

    /// Access the inner bandit pruner.
    pub fn inner(&self) -> &BanditPruner<P> {
        &self.inner
    }

    /// Access the inner bandit pruner mutably.
    pub fn inner_mut(&mut self) -> &mut BanditPruner<P> {
        &mut self.inner
    }

    /// Get global Q-values from the domain Q-table.
    pub fn domain_q_values(&self, domain_hash: u64) -> &[f32] {
        self.domain_q.q_values_for_domain(domain_hash)
    }
}

// ── ScreeningPruner Implementation ──────────────────────────────

impl<P: ScreeningPruner, L: EpisodeLookup> ScreeningPruner for SelfDistillingBandit<P, L> {
    /// Delegate relevance to inner bandit. Zero overhead on the screening path.
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bandit::BanditStrategy;
    use crate::pruners::episode_pruner::Episode;
    use crate::speculative::types::NoScreeningPruner;

    /// Simple in-memory episode lookup for tests.
    struct TestEpisodeLookup {
        episodes: std::collections::HashMap<u64, Episode>,
    }

    impl TestEpisodeLookup {
        fn new() -> Self {
            Self {
                episodes: std::collections::HashMap::new(),
            }
        }

        fn add_episode(&mut self, prompt_hash: u64, tokens: Vec<usize>) {
            self.episodes.insert(
                prompt_hash,
                Episode {
                    prompt_hash,
                    reference_tokens: tokens,
                    metadata: Default::default(),
                },
            );
        }
    }

    impl EpisodeLookup for TestEpisodeLookup {
        fn lookup(&self, prompt_hash: u64) -> Option<Episode> {
            self.episodes.get(&prompt_hash).cloned()
        }
    }

    // ── T2: ScreeningPruner delegation ──

    #[test]
    fn test_screening_pruner_delegation() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 10);
        let lookup = TestEpisodeLookup::new();
        let sd = SelfDistillingBandit::new(inner, lookup, SelfDistillingConfig::default());

        // Should delegate to inner bandit (all relevance = 1.0 for fresh UCB1)
        let rel = sd.relevance(0, 0, &[]);
        assert!(rel > 0.0, "relevance should be positive for fresh bandit");
    }

    // ── Reward computation ──

    #[test]
    fn test_reward_computer_perfect_match() {
        let config = SelfDistillingConfig::default();
        let computer = EpisodeRewardComputer::new(&config);
        let reward = computer.compute_reward(1.0, 1.0);
        // sigmoid(4.0 * 0.5) = sigmoid(2.0) ≈ 0.88, blended with alpha=0.3
        // (1 - 0.3) * 0.88 + 0.3 * 1.0 ≈ 0.616 + 0.3 ≈ 0.916
        assert!(
            reward > 0.85,
            "perfect match + accepted should give high reward: got {reward}"
        );
    }

    #[test]
    fn test_reward_computer_no_match() {
        let config = SelfDistillingConfig::default();
        let computer = EpisodeRewardComputer::new(&config);
        let reward = computer.compute_reward(0.0, 0.0);
        // sigmoid(4.0 * -0.5) = sigmoid(-2.0) ≈ 0.12, blended
        // (1 - 0.3) * 0.12 + 0.3 * 0.0 ≈ 0.084
        assert!(
            reward < 0.15,
            "no match + rejected should give low reward: got {reward}"
        );
    }

    #[test]
    fn test_reward_computer_no_episode_pure_acceptance() {
        let config = SelfDistillingConfig::default();
        let computer = EpisodeRewardComputer::new(&config);
        // When alpha=0.3, no episode → should still use acceptance
        // This is tested via the full flow, not the computer directly
        let reward_accepted = computer.compute_reward(0.5, 1.0);
        let reward_rejected = computer.compute_reward(0.5, 0.0);
        assert!(
            reward_accepted > reward_rejected,
            "accepted should give higher reward"
        );
    }

    // ── Match ratio ──

    #[test]
    fn test_match_ratio_identical() {
        let ratio = compute_match_ratio(&[1, 2, 3, 4], &[1, 2, 3, 4]);
        assert!((ratio - 1.0).abs() < 1e-6, "identical sequences: {ratio}");
    }

    #[test]
    fn test_match_ratio_partial() {
        let ratio = compute_match_ratio(&[1, 2, 3, 4], &[1, 2, 5, 6]);
        assert!((ratio - 0.5).abs() < 1e-6, "50% match: {ratio}");
    }

    #[test]
    fn test_match_ratio_empty() {
        let ratio = compute_match_ratio(&[], &[]);
        assert!((ratio - 1.0).abs() < 1e-6, "both empty: {ratio}");

        let ratio = compute_match_ratio(&[1], &[]);
        assert!((ratio - 0.0).abs() < 1e-6, "one empty: {ratio}");
    }

    // ── T3: Episode-guided update ──

    #[test]
    fn test_episode_update_with_reference() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 10);
        let mut lookup = TestEpisodeLookup::new();
        lookup.add_episode(42, vec![0, 1, 2, 3]);

        let mut sd = SelfDistillingBandit::new(inner, lookup, SelfDistillingConfig::default());

        // Perfect match → high reward
        sd.episode_update(42, 0, &[0, 1, 2, 3], 1.0, 0);
        let metrics = sd.convergence_metrics();
        assert_eq!(metrics.total_updates, 1);
        assert!((metrics.episode_hit_rate - 1.0).abs() < 1e-6);
        assert!(
            metrics.avg_reward > 0.8,
            "high reward on perfect match: {}",
            metrics.avg_reward
        );
    }

    #[test]
    fn test_episode_update_without_reference() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 10);
        let lookup = TestEpisodeLookup::new();
        let mut sd = SelfDistillingBandit::new(inner, lookup, SelfDistillingConfig::default());

        // No episode → pure acceptance reward
        sd.episode_update(99, 0, &[0, 1, 2], 1.0, 0);
        let metrics = sd.convergence_metrics();
        assert!((metrics.episode_hit_rate - 0.0).abs() < 1e-6);
        assert!(
            (metrics.avg_reward - 1.0).abs() < 1e-6,
            "pure acceptance: {}",
            metrics.avg_reward
        );
    }

    // ── T4/T5: Domain-keyed selection ──

    #[test]
    fn test_domain_keyed_different_best_arms() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        let mut lookup = TestEpisodeLookup::new();

        // Domain A: reference prefers arm 2
        lookup.add_episode(1, vec![2, 2, 2]);
        // Domain B: reference prefers arm 4
        lookup.add_episode(2, vec![4, 4, 4]);

        let mut sd = SelfDistillingBandit::new(
            inner,
            lookup,
            SelfDistillingConfig {
                min_domain_samples: 3,
                ..Default::default()
            },
        );

        // Warm up domain A (hash 100 → slot 100 % 64 = 36)
        for _ in 0..5 {
            sd.episode_update(1, 2, &[2, 2, 2], 1.0, 100);
        }

        // Warm up domain B (hash 200 → slot 200 % 64 = 8)
        for _ in 0..5 {
            sd.episode_update(2, 4, &[4, 4, 4], 1.0, 200);
        }

        // Different domains should have different best arms
        let best_a = sd.best_arm_for_domain(100);
        let best_b = sd.best_arm_for_domain(200);
        assert_ne!(
            best_a, best_b,
            "different domains should prefer different arms: A={best_a}, B={best_b}"
        );
    }

    #[test]
    fn test_cold_domain_falls_back_to_global() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        let lookup = TestEpisodeLookup::new();
        let sd = SelfDistillingBandit::new(
            inner,
            lookup,
            SelfDistillingConfig {
                min_domain_samples: 100, // Very high threshold → always cold
                ..Default::default()
            },
        );

        // Cold domain should return global Q-values
        let domain_q = sd.domain_q_values(999);
        let global_q = sd.domain_q_values(0);
        assert_eq!(
            domain_q.len(),
            global_q.len(),
            "cold domain should match global dimension"
        );
    }

    // ── T6/T7: Convergence ──

    #[test]
    fn test_convergence_improves_over_time() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        let mut lookup = TestEpisodeLookup::new();
        lookup.add_episode(42, vec![1, 2, 3]);

        let mut sd = SelfDistillingBandit::new(
            inner,
            lookup,
            SelfDistillingConfig {
                convergence_window: 50,
                ..Default::default()
            },
        );

        let mut early_reward = 0.0f32;

        for i in 0..100 {
            // Gradually improve match quality
            let generated = if i < 50 {
                vec![0, 0, 0] // Poor match early
            } else {
                vec![1, 2, 3] // Perfect match late
            };
            sd.episode_update(42, 1, &generated, 1.0, 0);

            if i == 49 {
                early_reward = sd.convergence_metrics().avg_reward;
            }
        }
        let late_reward = sd.convergence_metrics().avg_reward;

        assert!(
            late_reward > early_reward,
            "reward should improve over time: early={early_reward}, late={late_reward}"
        );
    }

    #[test]
    fn test_convergence_metrics_cold_start() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 10);
        let lookup = TestEpisodeLookup::new();
        let sd = SelfDistillingBandit::new(inner, lookup, SelfDistillingConfig::default());

        let metrics = sd.convergence_metrics();
        assert_eq!(metrics.total_updates, 0);
        assert!((metrics.avg_reward - 0.0).abs() < 1e-6);
        assert!((metrics.episode_hit_rate - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_bounded() {
        for x in [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
            let s = sigmoid(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s} not in (0, 1)");
        }
        // Extreme values should clamp to 0 or 1 (numerically exact)
        assert!(sigmoid(-100.0) >= 0.0, "sigmoid(-100) should be >= 0");
        assert!(sigmoid(100.0) <= 1.0, "sigmoid(100) should be <= 1");
    }

    #[test]
    fn test_sigmoid_monotonic() {
        let prev = sigmoid(-1.0);
        let curr = sigmoid(0.0);
        let next = sigmoid(1.0);
        assert!(
            prev < curr && curr < next,
            "sigmoid should be monotonically increasing"
        );
    }
}
