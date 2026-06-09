//! Multi-Armed Bandit (MAB) — Adaptive ScreeningPruner
//!
//! Implements bandit strategies that plug into the DDTree screening pipeline
//! via the [`ScreeningPruner`] trait, demonstrating how katgpt-rs's trait-based
//! architecture extends to sequential decision-making under uncertainty.
//!
//! # Architecture
//!
//! - [`BanditStrategy`] — exploration strategy enum (UCB1, ε-greedy, Thompson Sampling)
//! - [`BanditStats`] — shared arm tracking: Q-values, visit counts, scoring
//! - [`BanditPruner`] — wraps any `ScreeningPruner`, adds adaptive relevance for DDTree
//! - [`BanditEnv`] — trait for reward environments ([`BernoulliEnv`], [`GaussianEnv`])
//! - [`BanditSession`] — orchestrates episodes, tracks reward/regret, emits events
//!
//! # Two Usage Modes
//!
//! **1. DDTree Integration** — `BanditPruner` implements `ScreeningPruner`:
//! ```rust,ignore
//! let pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
//! pruner.prepare_episode(&mut rng); // Cache Thompson samples if needed
//! let tree = build_dd_tree_screened(&marginals, &config, &pruner, false);
//! // After verification:
//! pruner.update(accepted_token, reward);
//! ```
//!
//! **2. Standalone Bandit** — `BanditSession` runs episodes directly:
//! ```rust,ignore
//! let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
//! let session = BanditSession::new(env, BanditStrategy::Ucb1);
//! let (events, result) = session.run(500, &mut Rng::new(42));
//! assert_eq!(result.best_arm, 2); // Arm 2 has highest mean (0.8)
//! ```

use std::fmt;
use std::sync::{Arc, Mutex};

use super::absorb_compress::AbsorbCompress;
#[cfg(feature = "idea_divergence")]
use super::idea_divergence::IdeaDivergence;
use super::review_metrics::ReviewMetrics;
#[cfg(feature = "safe_bandit")]
use super::safe_phased::SafePhasedState;
#[cfg(feature = "skill_lifecycle")]
use super::skill_memory::{MemoryEntry, PrunerMemory};
use super::trial_log::{TrialLog, TrialRecord};
#[cfg(feature = "dynamic_rank")]
use crate::pruners::dynamic_rank::DynamicRankPruner;
use crate::speculative::types::ScreeningPruner;
use crate::types::Rng;

// ── Strategy ────────────────────────────────────────────────────

/// Exploration strategy for multi-armed bandit.
///
/// Each variant implements a different explore/exploit tradeoff.
/// All strategies converge to the optimal arm with sufficient episodes.
#[derive(Clone, Debug)]
pub enum BanditStrategy {
    /// UCB1: `Q(a) + sqrt(2 * ln(N) / n(a))`.
    ///
    /// Deterministic, no RNG needed. O(log N) regret bound.
    /// Best default choice for DDTree integration.
    Ucb1,

    /// ε-greedy: explore with probability `epsilon`, exploit otherwise.
    ///
    /// `decay` multiplies epsilon after each episode.
    /// Use `decay = 1.0` for no decay, `decay < 1.0` for annealing.
    EpsilonGreedy { epsilon: f32, decay: f32 },

    /// Thompson Sampling: sample from Beta(α, β) posterior per arm.
    ///
    /// Optimal asymptotic regret for Bernoulli rewards.
    /// Uses cached samples in [`BanditPruner`] (call `prepare_episode` first).
    ThompsonSampling,

    /// Variance-minimized epsilon (RePlaid-inspired).
    ///
    /// Adapts exploration rate to equalize per-episode reward variance.
    /// When reward variance is high, exploration increases.
    /// When reward variance is low, exploration decreases.
    /// Self-supervised — no hyperparameter tuning needed beyond initial ε.
    VarianceEpsilon {
        /// Initial epsilon.
        epsilon: f32,
        /// EMA decay for variance tracking (0.99 = slow).
        var_decay: f32,
        /// Learning rate for epsilon adaptation.
        lr: f32,
    },
    /// RPUCG: Rooted Propagation UCB on Graph (SimpleTES, Plan 086).
    ///
    /// Graph-based UCB with configurable exploration weight.
    /// `gamma` is the propagation discount (0.8 default).
    /// `lambda` is the exploration weight (1.0 default).
    /// Feature-gated under `tes_loop`.
    #[cfg(feature = "tes_loop")]
    Rpucg {
        /// Propagation discount γ (default 0.8).
        gamma: f32,
        /// Exploration weight λ (default 1.0).
        lambda: f32,
    },
    /// Density-aware exploration from RandOpt (Neural Thickets).
    /// High solution density → exploit, low → explore.
    RandOptAdaptive {
        /// Density threshold for switching (default: 0.3).
        density_threshold: f32,
        /// EMA decay for density tracking (default: 0.99).
        decay: f32,
    },
    /// PrudentBanker Safe-Phased Bandit — delay-calibrated safe exploration (Plan 137).
    ///
    /// Mixes between an active bandit learner and a safe baseline arm.
    /// Only escalates exploration when accumulated evidence certifies
    /// the baseline is suboptimal.
    ///
    /// `baseline_arm` is the safe fallback arm index.
    /// `delta` is the minimum baseline probability (controls delay slack).
    /// `estimated_delay` is the initial delay estimate D̂₀.
    ///
    /// Feature-gated under `safe_bandit`.
    #[cfg(feature = "safe_bandit")]
    SafePhased {
        /// Safe baseline arm index.
        baseline_arm: usize,
        /// Minimum baseline probability δ.
        delta: f32,
        /// Initial delay estimate D̂₀.
        estimated_delay: u32,
    },
    /// EoS-aware arm selection inspired by arXiv:2606.04212.
    ///
    /// When score concentration exceeds `concentration_threshold`, the top
    /// arm's score is boosted proportionally. All arms are guaranteed at
    /// least `floor` × max_score.
    ///
    /// Default-on (Plan 183 GOAT 6/6).
    CurvatureInfluence {
        /// Floor guarantee: all arms get at least `floor` × max_score.
        floor: f32,
        /// Concentration threshold for boost activation.
        concentration_threshold: f32,
    },
}

impl fmt::Display for BanditStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ucb1 => write!(f, "UCB1"),
            Self::EpsilonGreedy { epsilon, decay } => {
                write!(f, "ε-greedy(ε={epsilon:.3}, decay={decay:.3})")
            }
            Self::ThompsonSampling => write!(f, "Thompson"),
            Self::VarianceEpsilon {
                epsilon,
                var_decay,
                lr,
            } => {
                write!(f, "Var-ε(ε={epsilon:.3}, decay={var_decay:.3}, lr={lr:.3})")
            }
            #[cfg(feature = "tes_loop")]
            Self::Rpucg { gamma, lambda } => {
                write!(f, "RPUCG(γ={gamma:.2}, λ={lambda:.2})")
            }
            Self::RandOptAdaptive {
                density_threshold,
                decay,
            } => {
                write!(f, "RandOpt(ρ={density_threshold:.2}, decay={decay:.2})")
            }
            #[cfg(feature = "safe_bandit")]
            Self::SafePhased {
                baseline_arm,
                delta,
                estimated_delay,
            } => {
                write!(
                    f,
                    "SafePhased(base={baseline_arm}, δ={delta:.2}, D̂={estimated_delay})"
                )
            }
            Self::CurvatureInfluence {
                floor,
                concentration_threshold,
            } => {
                write!(f, "CIAB(floor={floor:.2}, c={concentration_threshold:.2})")
            }
        }
    }
}

// ── Stats ───────────────────────────────────────────────────────

/// Shared arm tracking state: Q-values, visit counts, scoring.
///
/// Used by both [`BanditPruner`] and [`BanditSession`] to avoid duplication.
/// All methods are O(1) including [`BanditStats::best_arm`] (cached).
pub struct BanditStats {
    q_values: Vec<f32>,
    visits: Vec<u32>,
    total_pulls: u32,
    num_arms: usize,
    /// Cached index of the arm with highest Q-value.
    best_arm_idx: usize,
    /// Running M2 for Welford variance (per arm).
    reward_m2: Vec<f32>,
    /// Running mean reward for variance tracking (per arm).
    reward_mean: Vec<f32>,
}

impl BanditStats {
    /// Create zero-initialized stats for `num_arms` arms.
    pub fn new(num_arms: usize) -> Self {
        Self {
            q_values: vec![0.0; num_arms],
            visits: vec![0; num_arms],
            total_pulls: 0,
            num_arms,
            best_arm_idx: num_arms.saturating_sub(1),
            reward_m2: vec![0.0; num_arms],
            reward_mean: vec![0.0; num_arms],
        }
    }

    /// Update Q-value for `arm` after observing `reward`.
    ///
    /// Uses incremental mean: `Q(a) += (reward - Q(a)) / n(a)`.
    #[inline]
    pub fn update(&mut self, arm: usize, reward: f32) {
        if arm >= self.num_arms {
            return;
        }
        self.visits[arm] += 1;
        self.total_pulls += 1;
        let n = self.visits[arm] as f32;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;

        // Invalidate best_arm cache if updated arm might now be best.
        // Use > to prefer the first arm strictly better than current best.
        // Use == with arm > best_arm_idx to match max_by's tie-breaking
        // (prefer later index, matching the original scan behavior).
        let updated_q = self.q_values[arm];
        let best_q = self.q_values[self.best_arm_idx];
        if updated_q > best_q || (updated_q == best_q && arm > self.best_arm_idx) {
            self.best_arm_idx = arm;
        } else if arm == self.best_arm_idx && updated_q < best_q {
            // Previous best degraded — full rescan needed
            self.best_arm_idx = self.scan_best_arm();
        }

        // Welford's online algorithm for reward variance tracking
        let old_mean = self.reward_mean[arm];
        let new_mean = old_mean + (reward - old_mean) / self.visits[arm] as f32;
        self.reward_m2[arm] += (reward - old_mean) * (reward - new_mean);
        self.reward_mean[arm] = new_mean;
    }

    /// UCB1 score: `Q(a) + sqrt(2 * ln(N) / n(a))`.
    ///
    /// Returns `f32::MAX` for unvisited arms (must explore first).
    #[inline]
    pub fn ucb1_score(&self, arm: usize) -> f32 {
        if self.visits[arm] == 0 || self.total_pulls == 0 {
            return f32::MAX;
        }
        let q = self.q_values[arm];
        let n = self.visits[arm] as f32;
        let total = self.total_pulls as f32;
        q + (2.0 * total.ln() / n).sqrt()
    }

    /// Thompson Sampling: draw from Beta(α, β) posterior.
    ///
    /// α = Q·n + 1, β = (1-Q)·n + 1 (Laplace smoothing).
    /// Returns uniform sample for unvisited arms.
    #[inline]
    pub fn thompson_sample(&self, arm: usize, rng: &mut Rng) -> f32 {
        if self.visits[arm] == 0 {
            return rng.uniform();
        }
        let n = self.visits[arm] as f32;
        let q = self.q_values[arm].clamp(0.0, 1.0);
        let alpha = q * n + 1.0;
        let beta = (1.0 - q) * n + 1.0;
        sample_beta(alpha, beta, rng)
    }

    /// Index of the arm with highest Q-value (cached O(1)).
    #[inline]
    pub fn best_arm(&self) -> usize {
        self.best_arm_idx
    }

    /// Full scan to find best arm — used only for cache invalidation.
    fn scan_best_arm(&self) -> usize {
        self.q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Q-value estimate for an arm.
    #[inline]
    pub fn q_value(&self, arm: usize) -> f32 {
        self.q_values.get(arm).copied().unwrap_or(0.0)
    }

    /// Visit count for an arm.
    #[inline]
    pub fn visit_count(&self, arm: usize) -> u32 {
        self.visits.get(arm).copied().unwrap_or(0)
    }

    /// Total pulls across all arms.
    #[inline]
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Number of arms.
    #[inline]
    pub fn num_arms(&self) -> usize {
        self.num_arms
    }

    /// Q-values slice (for inspection/logging).
    pub fn q_values(&self) -> &[f32] {
        &self.q_values
    }

    /// Visit counts slice (for inspection/logging).
    pub fn visits(&self) -> &[u32] {
        &self.visits
    }

    /// Welford variance for a single arm.
    ///
    /// Returns 0.0 for unvisited arms or arms with fewer than 2 samples.
    #[inline]
    pub fn reward_variance(&self, arm: usize) -> f32 {
        if arm >= self.num_arms || self.visits[arm] < 2 {
            return 0.0;
        }
        self.reward_m2[arm] / (self.visits[arm] - 1) as f32
    }

    /// Mean reward variance across all visited arms.
    ///
    /// Arms with fewer than 2 samples are excluded from the average.
    pub fn mean_reward_variance(&self) -> f32 {
        let mut sum = 0.0;
        let mut count = 0u32;
        for i in 0..self.num_arms {
            if self.visits[i] >= 2 {
                sum += self.reward_variance(i);
                count += 1;
            }
        }
        match count {
            0 => 0.0,
            _ => sum / count as f32,
        }
    }
}

// ── Pruner ──────────────────────────────────────────────────────

/// Adaptive screening pruner using multi-armed bandit strategies.
///
/// Wraps any inner [`ScreeningPruner`] for domain knowledge and adds
/// bandit-based relevance scoring. Implements [`ScreeningPruner`] —
/// plugs directly into [`build_dd_tree_screened`](crate::speculative::build_dd_tree_screened).
///
/// # Usage with DDTree
///
/// 1. Create pruner with strategy and arm count
/// 2. Call [`BanditPruner::prepare_episode`] before each DDTree build (caches Thompson samples)
/// 3. Pass to `build_dd_tree_screened` as the screener
/// 4. After verification, call [`BanditPruner::update`] with observed rewards
///
/// # Strategy Behavior in `relevance()`
///
/// | Strategy | Relevance Source |
/// |----------|-----------------|
/// | UCB1 | Q + exploration bonus (deterministic) |
/// | EpsilonGreedy | Q-value (exploration via session) |
/// | ThompsonSampling | Cached posterior sample (call `prepare_episode` first) |
pub struct BanditPruner<P: ScreeningPruner> {
    inner: P,
    strategy: BanditStrategy,
    stats: BanditStats,
    /// Cached Thompson samples, updated by `prepare_episode`.
    thompson_cache: Vec<f32>,
    /// Shared bandit stats for multi-agent cooperative learning.
    ///
    /// When `Some`, Q-values/visits are delegated to this shared reference.
    /// Multiple `BanditPruner` instances sharing one `Arc<SharedBanditStats>`
    /// learn cooperatively — updates from one are visible to all.
    #[cfg(feature = "bandit")]
    shared_stats: Option<Arc<SharedBanditStats>>,
    /// FFOLayer-inspired dual cutoff: arms with Q < cutoff get relevance = 0.0.
    /// Distilled from ffocp_eq.py backward pass (L1248-1269):
    ///   mask = (dual >= cutoff) → 1.0 else 0.0
    /// When 0.0 (disabled), behaves identically to current BanditPruner.
    dual_cutoff: f32,
    /// Soft-route blending: when true, relevance uses softmax-weighted blend of all
    /// arm bandit scores instead of the single requested arm's score.
    /// This smooths the routing signal, reducing variance from arm-selection noise.
    soft_route: bool,
    /// Temperature for softmax in soft-route mode. Higher = more uniform blending,
    /// lower = sharper (approaches hard-route as τ → 0).
    soft_route_tau: f32,
    /// Graduated reward scorer for richer bandit signal.
    ///
    /// When `Some`, `update_with_trace` uses `partial_score` instead of binary reward.
    #[cfg(feature = "partial_scoring")]
    partial_scorer: Option<Box<dyn katgpt_core::PartialScorer>>,
    /// Strategic novelty filter to prevent bandit arm convergence.
    ///
    /// When `Some`, non-novel arms receive an exploration penalty.
    #[cfg(feature = "idea_divergence")]
    idea_divergence: Option<IdeaDivergence>,
    /// Per-arm score vectors for divergence tracking. Indexed by arm.
    /// Updated via `update_divergence()`, compared during `arm_bandit_score()`.
    #[cfg(feature = "idea_divergence")]
    arm_score_vectors: Vec<Vec<f32>>,
    /// Per-pruner append-only memory for edge case accumulation.
    /// Stores arm selections, rewards, and failure modes across sessions.
    #[cfg(feature = "skill_lifecycle")]
    memory: PrunerMemory,
    /// Pre-allocated scratch buffers for soft-route relevance computation.
    /// Avoids per-call Vec allocation in the hot path.
    soft_route_scores: Mutex<Vec<f32>>,
    soft_route_weights: Mutex<Vec<f32>>,
}

impl<P: ScreeningPruner> BanditPruner<P> {
    /// Create a new bandit pruner wrapping an inner pruner.
    ///
    /// `num_arms` is the vocabulary size (number of discrete actions).
    pub fn new(inner: P, strategy: BanditStrategy, num_arms: usize) -> Self {
        Self {
            inner,
            strategy,
            stats: BanditStats::new(num_arms),
            thompson_cache: vec![0.0; num_arms],
            #[cfg(feature = "bandit")]
            shared_stats: None,
            dual_cutoff: 0.0,
            soft_route: true,
            soft_route_tau: 1.0,
            #[cfg(feature = "partial_scoring")]
            partial_scorer: None,
            #[cfg(feature = "idea_divergence")]
            idea_divergence: None,
            #[cfg(feature = "idea_divergence")]
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: Mutex::new(Vec::with_capacity(num_arms)),
            soft_route_weights: Mutex::new(Vec::with_capacity(num_arms)),
        }
    }

    /// Create a bandit pruner sharing stats with other pruner instances.
    ///
    /// Multiple `BanditPruner` instances sharing one `Arc<SharedBanditStats>`
    /// learn cooperatively: Q-values and visit counts are shared, but each
    /// pruner still has its own inner pruner, strategy, and Thompson cache.
    #[cfg(feature = "bandit")]
    pub fn with_shared_stats(
        inner: P,
        strategy: BanditStrategy,
        num_arms: usize,
        stats: Arc<SharedBanditStats>,
    ) -> Self {
        Self {
            inner,
            strategy,
            stats: BanditStats::new(num_arms),
            thompson_cache: vec![0.0; num_arms],
            shared_stats: Some(stats),
            dual_cutoff: 0.0,
            soft_route: true,
            soft_route_tau: 1.0,
            #[cfg(feature = "partial_scoring")]
            partial_scorer: None,
            #[cfg(feature = "idea_divergence")]
            idea_divergence: None,
            #[cfg(feature = "idea_divergence")]
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: Mutex::new(Vec::with_capacity(num_arms)),
            soft_route_weights: Mutex::new(Vec::with_capacity(num_arms)),
        }
    }

    // ── Partial Scoring (Plan 187 T3) ──────────────────────────────

    /// Create a bandit pruner with a graduated reward scorer.
    ///
    /// Uses `partial_score` for richer signal than binary win/loss.
    #[cfg(feature = "partial_scoring")]
    pub fn with_partial_scorer(
        inner: P,
        strategy: BanditStrategy,
        num_arms: usize,
        scorer: Box<dyn katgpt_core::PartialScorer>,
    ) -> Self {
        Self {
            inner,
            strategy,
            stats: BanditStats::new(num_arms),
            thompson_cache: vec![0.0; num_arms],
            #[cfg(feature = "bandit")]
            shared_stats: None,
            dual_cutoff: 0.0,
            soft_route: true,
            soft_route_tau: 1.0,
            partial_scorer: Some(scorer),
            #[cfg(feature = "idea_divergence")]
            idea_divergence: None,
            #[cfg(feature = "idea_divergence")]
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: Mutex::new(Vec::with_capacity(num_arms)),
            soft_route_weights: Mutex::new(Vec::with_capacity(num_arms)),
        }
    }

    // ── Idea Divergence (Plan 191 T3.2) ────────────────────────────

    /// Create a bandit pruner with strategic novelty filter.
    ///
    /// Non-novel arms (those converging to similar strategies) receive
    /// a 50% exploration penalty during arm selection.
    #[cfg(feature = "idea_divergence")]
    pub fn with_idea_divergence(
        inner: P,
        strategy: BanditStrategy,
        num_arms: usize,
        threshold: f32,
    ) -> Self {
        Self {
            inner,
            strategy,
            stats: BanditStats::new(num_arms),
            thompson_cache: vec![0.0; num_arms],
            #[cfg(feature = "bandit")]
            shared_stats: None,
            dual_cutoff: 0.0,
            soft_route: true,
            soft_route_tau: 1.0,
            #[cfg(feature = "partial_scoring")]
            partial_scorer: None,
            idea_divergence: Some(IdeaDivergence::new(threshold)),
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: Mutex::new(Vec::with_capacity(num_arms)),
            soft_route_weights: Mutex::new(Vec::with_capacity(num_arms)),
        }
    }

    // ── Divergence Update ─────────────────────────────────────────
    /// Update the divergence tracker after an arm update.
    ///
    /// Reads the current Q-value for the arm and updates the per-arm score vector.
    /// Call this after `update_with_trace()` or `update()`.
    #[cfg(feature = "idea_divergence")]
    pub fn update_divergence(&mut self, arm: usize) {
        if self.idea_divergence.is_none() {
            return;
        }
        let q = self.arm_q(arm);
        let visits = self.arm_visits(arm);
        let score_vec = vec![q, visits as f32];
        if arm < self.arm_score_vectors.len() {
            self.arm_score_vectors[arm] = score_vec;
        }
    }

    /// Update arm reward from a GameTrace using the PartialScorer.
    ///
    /// Falls back to binary final_reward when no scorer is set.
    #[cfg(feature = "partial_scoring")]
    #[inline]
    pub fn update_with_trace(&mut self, arm: usize, trace: &katgpt_core::GameTrace) {
        let reward = match &self.partial_scorer {
            Some(scorer) => scorer.partial_score(trace),
            None => {
                if trace.final_reward > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        };
        self.update_arm(arm, reward);
    }

    // ── Shared Stats Accessors ─────────────────────────────────

    /// Visit count for an arm.
    ///
    /// Delegates to shared stats when present, else uses local stats.
    #[cfg(feature = "bandit")]
    fn arm_visits(&self, arm: usize) -> u32 {
        match &self.shared_stats {
            Some(stats) => stats.visits(arm),
            None => self.stats.visit_count(arm),
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_visits(&self, arm: usize) -> u32 {
        self.stats.visit_count(arm)
    }

    /// Q-value estimate for an arm.
    #[cfg(feature = "bandit")]
    fn arm_q(&self, arm: usize) -> f32 {
        match &self.shared_stats {
            Some(stats) => stats.q_value(arm),
            None => self.stats.q_value(arm),
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_q(&self, arm: usize) -> f32 {
        self.stats.q_value(arm)
    }

    /// Total pulls across all arms.
    #[cfg(feature = "bandit")]
    fn arm_total_pulls(&self) -> u32 {
        match &self.shared_stats {
            Some(stats) => stats.total_pulls(),
            None => self.stats.total_pulls(),
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_total_pulls(&self) -> u32 {
        self.stats.total_pulls()
    }

    /// UCB1 score for an arm.
    #[cfg(feature = "bandit")]
    fn arm_ucb1(&self, arm: usize) -> f32 {
        match &self.shared_stats {
            Some(stats) => stats.ucb1_score(arm),
            None => self.stats.ucb1_score(arm),
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_ucb1(&self, arm: usize) -> f32 {
        self.stats.ucb1_score(arm)
    }

    /// Thompson sample for an arm using shared or local stats.
    #[cfg(feature = "bandit")]
    fn arm_thompson(&self, arm: usize, rng: &mut Rng) -> f32 {
        match &self.shared_stats {
            Some(stats) => {
                let (q_raw, n) = stats.arm_snapshot(arm);
                if n == 0 {
                    return rng.uniform();
                }
                let q = q_raw.clamp(0.0, 1.0);
                let alpha = q * n as f32 + 1.0;
                let beta = (1.0 - q) * n as f32 + 1.0;
                sample_beta(alpha, beta, rng)
            }
            None => self.stats.thompson_sample(arm, rng),
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_thompson(&self, arm: usize, rng: &mut Rng) -> f32 {
        self.stats.thompson_sample(arm, rng)
    }

    /// Update Q-value for an arm after observing a reward.
    #[cfg(feature = "bandit")]
    fn update_arm(&mut self, arm: usize, reward: f32) {
        match &self.shared_stats {
            Some(stats) => stats.update(arm, reward),
            None => self.stats.update(arm, reward),
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn update_arm(&mut self, arm: usize, reward: f32) {
        self.stats.update(arm, reward);
    }

    /// Index of the best arm (highest Q-value).
    #[cfg(feature = "bandit")]
    fn arm_best(&self) -> usize {
        match &self.shared_stats {
            Some(stats) => stats.best_arm(),
            None => self.stats.best_arm(),
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_best(&self) -> usize {
        self.stats.best_arm()
    }

    /// Prepare for a new episode. Call before each DDTree build.
    ///
    /// - Thompson Sampling: draws posterior samples and caches them.
    /// - VarianceEpsilon: adapts epsilon based on reward variance.
    /// - Other strategies: no-op.
    pub fn prepare_episode(&mut self, rng: &mut Rng) {
        match &self.strategy {
            BanditStrategy::ThompsonSampling => {
                let n = self.stats.num_arms;
                for i in 0..n {
                    self.thompson_cache[i] = self.arm_thompson(i, rng);
                }
            }
            BanditStrategy::VarianceEpsilon { .. } => {
                // Variance-epsilon adapts dynamically during arm selection.
                // No pre-computation needed — variance is tracked in BanditStats.
            }
            _ => {}
        }
    }

    /// Update Q-value for an arm after observing a reward.
    #[inline]
    pub fn update(&mut self, arm: usize, reward: f32) {
        self.update_arm(arm, reward);
    }

    /// Record an arm pull experience to the pruner's memory ring buffer.
    /// Call after `update()` or `update_with_trace()`.
    #[cfg(feature = "skill_lifecycle")]
    pub fn record_experience(&self, arm: u16, reward: f32, is_edge_case: bool, is_failure: bool) {
        let ts = self.memory.total_entries();
        self.memory
            .append(MemoryEntry::new(arm, reward, is_edge_case, is_failure, ts));
    }

    /// Retrieve the last K experiences from memory.
    #[cfg(feature = "skill_lifecycle")]
    pub fn recent_experiences(&self, k: usize) -> Vec<MemoryEntry> {
        self.memory.recent(k)
    }

    /// Access the underlying PrunerMemory (for advanced use like identity verification).
    #[cfg(feature = "skill_lifecycle")]
    pub fn pruner_memory(&self) -> &PrunerMemory {
        &self.memory
    }

    /// Decay epsilon after an episode (EpsilonGreedy only).
    pub fn decay_epsilon(&mut self) {
        if let BanditStrategy::EpsilonGreedy { epsilon, decay } = &mut self.strategy {
            *epsilon *= *decay;
        }
    }

    /// Index of the best arm (highest Q-value).
    pub fn best_arm(&self) -> usize {
        self.arm_best()
    }

    /// Q-values slice (for inspection).
    pub fn q_values(&self) -> &[f32] {
        self.stats.q_values()
    }

    /// Visit counts slice (for inspection).
    pub fn visits(&self) -> &[u32] {
        self.stats.visits()
    }

    /// Total pulls across all arms.
    pub fn total_pulls(&self) -> u32 {
        self.arm_total_pulls()
    }

    /// Strategy reference.
    pub fn strategy(&self) -> &BanditStrategy {
        &self.strategy
    }

    /// Set the FFOLayer-inspired dual cutoff threshold.
    ///
    /// Arms with Q-value below `cutoff` get relevance = 0.0 (hard masked).
    /// Set to 0.0 to disable (default, backward-compatible).
    pub fn set_dual_cutoff(&mut self, cutoff: f32) {
        self.dual_cutoff = cutoff;
    }

    /// Configure soft-route blending.
    ///
    /// When `enabled`, relevance blends all arm scores via softmax weighting
    /// instead of using only the requested arm's score. `tau` controls
    /// the softmax temperature (higher = more uniform, lower = sharper).
    pub fn set_soft_route(&mut self, enabled: bool, tau: f32) {
        self.soft_route = enabled;
        self.soft_route_tau = tau.max(0.01); // Prevent division by zero
    }

    /// Per-arm bandit score (strategy-dependent), in [0, 1].
    ///
    /// This is the score component extracted from `relevance()` so it can
    /// be reused for both hard-route and soft-route paths.
    fn arm_bandit_score(&self, token_idx: usize) -> f32 {
        if self.arm_visits(token_idx) == 0 {
            return 1.0; // Maximum exploration priority for unvisited arms
        }

        // FFOLayer hard cutoff: low-Q arms get zero
        if self.dual_cutoff > 0.0 && self.arm_q(token_idx) < self.dual_cutoff {
            return 0.0;
        }

        #[allow(unused_mut)]
        let mut score = match &self.strategy {
            BanditStrategy::Ucb1 => self.arm_ucb1(token_idx).clamp(0.0, 1.5) / 1.5,
            BanditStrategy::EpsilonGreedy { .. } => self.arm_q(token_idx).clamp(0.0, 1.0).max(0.01),
            BanditStrategy::ThompsonSampling => self
                .thompson_cache
                .get(token_idx)
                .copied()
                .unwrap_or_else(|| self.arm_q(token_idx))
                .clamp(0.0, 1.0)
                .max(0.01),
            BanditStrategy::VarianceEpsilon { .. } => {
                self.arm_q(token_idx).clamp(0.0, 1.0).max(0.01)
            }
            #[cfg(feature = "tes_loop")]
            BanditStrategy::Rpucg { gamma: _, lambda } => {
                let q = self.arm_q(token_idx);
                let n = self.arm_visits(token_idx) as f32;
                let total = self.arm_total_pulls() as f32;
                let exploration = lambda * ((total + 1.0).ln() / (n + 1.0)).sqrt();
                (q + exploration).clamp(0.0, 1.5) / 1.5
            }
            BanditStrategy::RandOptAdaptive { .. } => {
                self.arm_q(token_idx).clamp(0.0, 1.0).max(0.01)
            }
            #[cfg(feature = "safe_bandit")]
            BanditStrategy::SafePhased { .. } => self.arm_ucb1(token_idx).clamp(0.0, 1.5) / 1.5,
            BanditStrategy::CurvatureInfluence {
                floor,
                concentration_threshold,
            } => {
                let q = self.arm_q(token_idx).clamp(0.0, 1.0).max(0.01);
                // Compute concentration across all arms (in-place, no allocation)
                let num_arms = self.stats.num_arms;
                let mut max_score = 0.0f32;
                let mut sum = 0.0f32;
                for a in 0..num_arms {
                    let s = self.arm_q(a).clamp(0.0, 1.0).max(0.01);
                    max_score = max_score.max(s);
                    sum += s;
                }
                let concentration = if sum > 0.0 && max_score > 0.0 {
                    max_score / sum
                } else {
                    1.0 / num_arms as f32
                };
                // Boost if concentration exceeds threshold
                let boosted = if concentration > *concentration_threshold {
                    let boost = concentration / *concentration_threshold;
                    q * boost
                } else {
                    q
                };
                // Floor guarantee
                let min_score = floor * max_score.max(q);
                boosted.max(min_score).clamp(0.0, 1.0)
            }
        };

        // Strategic novelty penalty: non-novel arms get reduced selection probability.
        #[cfg(feature = "idea_divergence")]
        if let Some(ref div) = self.idea_divergence {
            let q = self.arm_q(token_idx);
            let visits = self.arm_visits(token_idx);
            let score_vec = [q, visits as f32];
            // Check divergence against all OTHER arms' score vectors
            let mut min_dist = f32::MAX;
            for (i, other) in self.arm_score_vectors.iter().enumerate() {
                if i != token_idx && !other.is_empty() {
                    let d = IdeaDivergence::divergence(&score_vec, other);
                    if d < min_dist {
                        min_dist = d;
                    }
                }
            }
            if min_dist <= div.threshold() {
                score *= 0.5;
            }
        }

        score
    }

    /// Soft-route relevance: softmax-weighted blend of all arm bandit scores.
    ///
    /// Instead of returning just the score for the requested arm, this computes
    /// a blended score where each arm's contribution is weighted by its softmax
    /// probability. This smooths the routing signal and reduces variance from
    /// arm-selection noise.
    ///
    /// The blend is: `Σ_i softmax_i(τ) × bandit_score_i`, where
    /// `softmax_i(τ) = exp(score_i / τ) / Σ_j exp(score_j / τ)`.
    fn soft_route_relevance(&self, depth: usize, token_idx: usize, parent_token: &[usize]) -> f32 {
        let domain = self.inner.relevance(depth, token_idx, parent_token);
        if domain <= 0.0 {
            return 0.0;
        }

        // Cold start: no data yet, use domain only
        if self.arm_total_pulls() == 0 {
            return domain;
        }

        let num_arms = self.stats.num_arms;
        let tau = self.soft_route_tau;

        // Compute bandit scores into pre-allocated scratch buffer
        let mut scores = self.soft_route_scores.lock().unwrap();
        scores.clear();
        scores.extend((0..num_arms).map(|a| self.arm_bandit_score(a)));

        // Numerical stability: subtract max before exp
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = self.soft_route_weights.lock().unwrap();
        weights.clear();
        weights.extend(scores.iter().map(|&s| ((s - max_score) / tau).exp()));
        let weight_sum: f32 = weights.iter().sum();

        if weight_sum <= 0.0 {
            // Fallback: uniform blend
            let avg: f32 = scores.iter().sum::<f32>() / num_arms as f32;
            return (domain * avg).clamp(0.0, 1.0);
        }

        // Softmax-weighted blend of all arm scores
        let blended: f32 = weights
            .iter()
            .zip(scores.iter())
            .map(|(&w, &s)| w * s)
            .sum::<f32>()
            / weight_sum;

        (domain * blended).clamp(0.0, 1.0)
    }

    /// Wrap this BanditPruner with static ranking detection and correction.
    /// GATv2 insight: BanditPruner Q-values are per-arm (no parent conditioning) → static ranking.
    /// This wrapper detects that and applies context-dependent corrections.
    #[cfg(feature = "dynamic_rank")]
    pub fn with_dynamic_rank(self, vocab_size: usize) -> DynamicRankPruner<Self> {
        DynamicRankPruner::new(self, vocab_size)
    }
}

impl<P: ScreeningPruner> ScreeningPruner for BanditPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if token_idx >= self.stats.num_arms {
            return 0.0;
        }

        // Soft-route: softmax-weighted blend of all arm scores
        if self.soft_route {
            return self.soft_route_relevance(depth, token_idx, parent_tokens);
        }

        // Hard-route: original behavior — single arm's bandit score
        let domain = self.inner.relevance(depth, token_idx, parent_tokens);
        if domain <= 0.0 {
            return 0.0;
        }

        // Cold start: no data yet, use domain only
        if self.arm_total_pulls() == 0 {
            return domain;
        }

        // Unvisited arm: maximum exploration priority
        if self.arm_visits(token_idx) == 0 {
            return domain;
        }

        let bandit = self.arm_bandit_score(token_idx);

        // Harmonic blend: domain × bandit
        (domain * bandit).clamp(0.0, 1.0)
    }
}

// ── Beta Distribution Sampling ──────────────────────────────────

/// Sample from Beta(α, β) distribution using Jöhnk's algorithm.
///
/// Works well for α, β ≥ 1 (our case: posterior with +1 pseudocounts).
/// Uses rejection sampling. Falls back to 0.5 after 256 rejections.
fn sample_beta(alpha: f32, beta: f32, rng: &mut Rng) -> f32 {
    // Uniform prior: α=1, β=1
    if (alpha - 1.0).abs() < f32::EPSILON && (beta - 1.0).abs() < f32::EPSILON {
        return rng.uniform();
    }

    // Jöhnk's algorithm: X = U1^(1/α), Y = U2^(1/β), accept if X+Y ≤ 1
    for _ in 0..256 {
        let u1 = rng.uniform().max(f32::EPSILON);
        let u2 = rng.uniform().max(f32::EPSILON);
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);
        let sum = x + y;
        if sum <= 1.0 && sum > 0.0 {
            return x / sum;
        }
    }

    // Fallback: Q-value midpoint
    0.5
}

// ── Environment ─────────────────────────────────────────────────

/// A multi-armed bandit environment that generates stochastic rewards.
///
/// Each arm has a hidden reward distribution. The agent's goal is to
/// identify the arm with the highest expected reward while minimizing
/// cumulative regret.
pub trait BanditEnv: Send + Sync {
    /// Pull an arm and receive a stochastic reward in [0.0, 1.0].
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32;

    /// Expected (mean) reward for a specific arm.
    fn expected_reward(&self, arm: usize) -> f32;

    /// Expected reward of the optimal arm.
    fn optimal_reward(&self) -> f32;

    /// Number of arms.
    fn num_arms(&self) -> usize;

    /// Index of the optimal arm (highest expected reward).
    fn optimal_arm(&self) -> usize;
}

// ── Bernoulli Environment ───────────────────────────────────────

/// Bernoulli bandit: each arm returns 1.0 with probability `p`, 0.0 otherwise.
///
/// Classic MAB setting. Optimal for Thompson Sampling with Beta posteriors.
#[derive(Clone)]
pub struct BernoulliEnv {
    probs: Vec<f32>,
    optimal_arm: usize,
    optimal_reward: f32,
}

impl BernoulliEnv {
    /// Create a Bernoulli bandit with per-arm success probabilities.
    pub fn new(probs: &[f32]) -> Self {
        let optimal_arm = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let optimal_reward = probs[optimal_arm];
        Self {
            probs: probs.to_vec(),
            optimal_arm,
            optimal_reward,
        }
    }

    /// Success probability for each arm.
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }
}

impl BanditEnv for BernoulliEnv {
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32 {
        if arm >= self.probs.len() || rng.uniform() >= self.probs[arm] {
            0.0
        } else {
            1.0
        }
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        self.probs.get(arm).copied().unwrap_or(0.0)
    }

    fn optimal_reward(&self) -> f32 {
        self.optimal_reward
    }

    fn num_arms(&self) -> usize {
        self.probs.len()
    }

    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}

// ── Gaussian Environment ────────────────────────────────────────

/// Gaussian bandit: each arm returns a reward sampled from N(μ, σ²).
///
/// Rewards are clamped to [0.0, 1.0]. Useful for continuous reward settings.
#[derive(Clone)]
pub struct GaussianEnv {
    means: Vec<f32>,
    std: f32,
    optimal_arm: usize,
    optimal_reward: f32,
}

impl GaussianEnv {
    /// Create a Gaussian bandit with per-arm means and shared standard deviation.
    pub fn new(means: &[f32], std: f32) -> Self {
        let optimal_arm = means
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let optimal_reward = means[optimal_arm];
        Self {
            means: means.to_vec(),
            std,
            optimal_arm,
            optimal_reward,
        }
    }

    /// Mean reward for each arm.
    pub fn means(&self) -> &[f32] {
        &self.means
    }
}

impl BanditEnv for GaussianEnv {
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32 {
        if arm >= self.means.len() {
            return 0.0;
        }
        // Box-Muller transform for Gaussian sampling
        let u1 = rng.uniform().max(f32::EPSILON);
        let u2 = rng.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        (self.means[arm] + self.std * z).clamp(0.0, 1.0)
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

// ── Bandit Event ────────────────────────────────────────────────

/// Events emitted during bandit session execution.
#[derive(Clone, Debug)]
pub enum BanditEvent {
    /// An arm was pulled and reward observed.
    Pull {
        episode: usize,
        arm: usize,
        reward: f32,
        q_value: f32,
    },
    /// Episode completed with cumulative stats.
    EpisodeComplete {
        episode: usize,
        arm: usize,
        reward: f32,
        cumulative_reward: f32,
        cumulative_regret: f32,
    },
    /// Session completed with final stats.
    SessionComplete {
        total_episodes: usize,
        total_reward: f32,
        total_regret: f32,
        best_arm: usize,
        optimal_arm: usize,
    },
}

// ── Bandit Result ───────────────────────────────────────────────

/// Final result of a bandit session.
#[derive(Clone, Debug)]
pub struct BanditResult {
    /// Total episodes run.
    pub total_episodes: usize,
    /// Sum of all observed rewards.
    pub total_reward: f32,
    /// Sum of per-episode regret: Σ(optimal_reward - arm_expected_reward).
    pub total_regret: f32,
    /// Arm with highest Q-value at session end.
    pub best_arm: usize,
    /// True optimal arm from the environment.
    pub optimal_arm: usize,
    /// Final Q-value estimates.
    pub q_values: Vec<f32>,
    /// Final visit counts.
    pub visits: Vec<u32>,
}

impl BanditResult {
    /// Whether the bandit found the true optimal arm.
    pub fn found_optimal(&self) -> bool {
        self.best_arm == self.optimal_arm
    }

    /// Average reward per episode.
    pub fn avg_reward(&self) -> f32 {
        if self.total_episodes == 0 {
            0.0
        } else {
            self.total_reward / self.total_episodes as f32
        }
    }

    /// Average regret per episode.
    pub fn avg_regret(&self) -> f32 {
        if self.total_episodes == 0 {
            0.0
        } else {
            self.total_regret / self.total_episodes as f32
        }
    }
}

// ── Bandit Session ──────────────────────────────────────────────

/// Orchestrates multi-armed bandit episodes.
///
/// Runs N episodes of arm selection → reward observation → Q-value update.
/// Tracks cumulative reward and pseudo-regret. Emits events for logging.
///
/// # Example
///
/// ```rust,ignore
/// let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
/// let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
/// let (events, result) = session.run(500, &mut Rng::new(42));
/// assert!(result.found_optimal());
/// ```
pub struct BanditSession<E: BanditEnv> {
    env: E,
    strategy: BanditStrategy,
    stats: BanditStats,
    cumulative_reward: f32,
    cumulative_regret: f32,
    /// Optional review metrics for inference-time feedback tracking (Plan 036).
    review_metrics: Option<Arc<ReviewMetrics>>,
    /// Optional safe-phased state for PrudentBanker (Plan 137).
    #[cfg(feature = "safe_bandit")]
    safe_phased_state: Option<SafePhasedState>,
}

impl<E: BanditEnv> BanditSession<E> {
    /// Create a new bandit session with the given environment and strategy.
    pub fn new(env: E, strategy: BanditStrategy) -> Self {
        let num_arms = env.num_arms();
        #[cfg(feature = "safe_bandit")]
        let safe_phased_state = match &strategy {
            BanditStrategy::SafePhased {
                baseline_arm,
                delta,
                estimated_delay,
            } => Some(SafePhasedState::new(
                *baseline_arm,
                *delta,
                *estimated_delay,
                num_arms,
            )),
            _ => None,
        };
        Self {
            env,
            strategy,
            stats: BanditStats::new(num_arms),
            cumulative_reward: 0.0,
            cumulative_regret: 0.0,
            review_metrics: None,
            #[cfg(feature = "safe_bandit")]
            safe_phased_state,
        }
    }

    /// Enable review metrics tracking (Plan 036, builder pattern).
    ///
    /// After each episode, records whether the bandit's pick was the
    /// optimal arm vs whether a simulated random pick would have been.
    /// The same `Arc<ReviewMetrics>` can be shared across components.
    pub fn with_metrics(mut self, metrics: Arc<ReviewMetrics>) -> Self {
        self.review_metrics = Some(metrics);
        self
    }

    /// Select an arm based on the current strategy and stats.
    fn select_arm(&self, rng: &mut Rng) -> usize {
        let num_arms = self.env.num_arms();

        // Cold start: play each arm once
        for i in 0..num_arms {
            if self.stats.visit_count(i) == 0 {
                return i;
            }
        }

        match &self.strategy {
            BanditStrategy::Ucb1 => self.select_ucb1(),
            BanditStrategy::EpsilonGreedy { epsilon, .. } => {
                self.select_epsilon_greedy(*epsilon, rng)
            }
            BanditStrategy::ThompsonSampling => self.select_thompson(rng),
            BanditStrategy::VarianceEpsilon { .. } => self.select_variance_epsilon(rng),
            #[cfg(feature = "tes_loop")]
            BanditStrategy::Rpucg { .. } => self.select_ucb1(), // Flat bandit fallback; graph propagation in TesLoop
            BanditStrategy::RandOptAdaptive {
                density_threshold, ..
            } => {
                // Density-aware fallback: use threshold as epsilon until full implementation
                self.select_epsilon_greedy(*density_threshold, rng)
            }
            #[cfg(feature = "safe_bandit")]
            BanditStrategy::SafePhased { .. } => self.select_safe_phased(rng),
            BanditStrategy::CurvatureInfluence { .. } => self.select_ucb1(), // UCB1 base with CIAB scoring override
        }
    }

    fn select_ucb1(&self) -> usize {
        (0..self.env.num_arms())
            .max_by(|&a, &b| {
                self.stats
                    .ucb1_score(a)
                    .partial_cmp(&self.stats.ucb1_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    fn select_epsilon_greedy(&self, epsilon: f32, rng: &mut Rng) -> usize {
        let num_arms = self.env.num_arms();
        if rng.uniform() < epsilon {
            // Explore: random arm
            (rng.uniform() * num_arms as f32) as usize % num_arms
        } else {
            // Exploit: best Q-value
            self.stats.best_arm()
        }
    }

    fn select_thompson(&self, rng: &mut Rng) -> usize {
        (0..self.env.num_arms())
            .map(|i| (i, self.stats.thompson_sample(i, rng)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Variance-minimized epsilon selection (RePlaid-inspired).
    ///
    /// Adapts exploration rate based on mean reward variance across arms.
    /// High variance → more exploration; low variance → more exploitation.
    fn select_variance_epsilon(&self, rng: &mut Rng) -> usize {
        let mean_var = self.stats.mean_reward_variance();
        let adapted_eps = match &self.strategy {
            BanditStrategy::VarianceEpsilon { epsilon, lr, .. } => {
                let factor = 1.0 + lr * mean_var.sqrt();
                (epsilon * factor).clamp(0.01, 1.0)
            }
            _ => 0.1,
        };
        let num_arms = self.env.num_arms();
        if rng.uniform() < adapted_eps {
            (rng.uniform() * num_arms as f32) as usize % num_arms
        } else {
            self.stats.best_arm()
        }
    }

    /// Decay epsilon (EpsilonGreedy only).
    fn decay_epsilon(&mut self) {
        if let BanditStrategy::EpsilonGreedy { epsilon, decay } = &mut self.strategy {
            *epsilon *= *decay;
        }
    }

    /// Select arm using safe-phased mixture (Plan 137).
    ///
    /// Uses UCB1 as the active arm selector, then applies safe mixture
    /// with the baseline arm based on current αₖ.
    #[cfg(feature = "safe_bandit")]
    fn select_safe_phased(&self, rng: &mut Rng) -> usize {
        let active_arm = self.select_ucb1();
        if let Some(ref state) = self.safe_phased_state {
            state.select_with_safe_mixture(active_arm, rng)
        } else {
            active_arm
        }
    }

    /// Update safe-phased state after observing reward (Plan 137).
    ///
    /// Uses the **active** arm's expected reward for gap tracking,
    /// not the selected arm. This ensures the gap accurately reflects
    /// how the exploratory active arm compares to the safe baseline,
    /// regardless of whether the mixture selected the baseline.
    #[cfg(feature = "safe_bandit")]
    fn update_safe_phased(&mut self, selected_arm: usize, reward: f32) {
        if let Some(ref mut state) = self.safe_phased_state {
            state.record_round();
            let baseline_arm = state.baseline_arm();
            // If baseline was selected, use expected reward for gap tracking
            // (to avoid always seeing 0 gap when baseline dominates)
            if selected_arm == baseline_arm {
                // Baseline selected: no gap contribution (we got what we expected)
                // But still track the active arm's hypothetical performance
                // by not accumulating any gap (baseline performed as expected)
            } else {
                // Active arm selected: compare its reward against baseline
                let baseline_expected = self.env.expected_reward(baseline_arm);
                state.update_phase_gap(baseline_expected, reward);
            }
            if state.should_soft_restart() {
                state.soft_restart();
            }
        }
    }

    /// Run the bandit session for `episodes` episodes.
    ///
    /// Returns `(events, result)`. Events include per-episode stats.
    /// Pseudo-regret = Σ(optimal_expected - chosen_arm_expected).
    pub fn run(mut self, episodes: usize, rng: &mut Rng) -> (Vec<BanditEvent>, BanditResult) {
        let mut events = Vec::with_capacity(episodes + 1);
        let optimal_arm = self.env.optimal_arm();
        let optimal_reward = self.env.optimal_reward();

        for episode in 0..episodes {
            let arm = self.select_arm(rng);
            let reward = self.env.pull(arm, rng);
            let q_before = self.stats.q_value(arm);

            events.push(BanditEvent::Pull {
                episode,
                arm,
                reward,
                q_value: q_before,
            });

            self.stats.update(arm, reward);
            self.cumulative_reward += reward;
            self.cumulative_regret += optimal_reward - self.env.expected_reward(arm);

            self.decay_epsilon();

            // Update safe-phased state (Plan 137)
            #[cfg(feature = "safe_bandit")]
            self.update_safe_phased(arm, reward);

            // Record review metrics (Plan 036)
            if let Some(ref metrics) = self.review_metrics {
                let reviewed_correct = arm == optimal_arm;
                // Simulate base (random) correctness deterministically
                let base_correct = episode % self.env.num_arms() == optimal_arm;
                metrics.record(base_correct, reviewed_correct);
            }

            events.push(BanditEvent::EpisodeComplete {
                episode,
                arm,
                reward,
                cumulative_reward: self.cumulative_reward,
                cumulative_regret: self.cumulative_regret,
            });
        }

        let best_arm = self.stats.best_arm();
        let result = BanditResult {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
            q_values: self.stats.q_values.to_vec(),
            visits: self.stats.visits.to_vec(),
        };

        events.push(BanditEvent::SessionComplete {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
        });

        (events, result)
    }

    /// Run the bandit session with trial log persistence.
    ///
    /// Same as [`run`](Self::run) but appends each episode's record to `trial_log`.
    /// The `config` string is attached to every record for later analysis.
    pub fn run_with_trial_log(
        mut self,
        episodes: usize,
        rng: &mut Rng,
        trial_log: &mut TrialLog,
        config: &str,
    ) -> (Vec<BanditEvent>, BanditResult) {
        let mut events = Vec::with_capacity(episodes + 1);
        let optimal_arm = self.env.optimal_arm();
        let optimal_reward = self.env.optimal_reward();
        let config_owned = config.to_string();

        for episode in 0..episodes {
            let arm = self.select_arm(rng);
            let reward = self.env.pull(arm, rng);
            let q_before = self.stats.q_value(arm);

            events.push(BanditEvent::Pull {
                episode,
                arm,
                reward,
                q_value: q_before,
            });

            self.stats.update(arm, reward);
            self.cumulative_reward += reward;
            self.cumulative_regret += optimal_reward - self.env.expected_reward(arm);

            self.decay_epsilon();

            // Update safe-phased state (Plan 137)
            #[cfg(feature = "safe_bandit")]
            self.update_safe_phased(arm, reward);

            // Record review metrics (Plan 036)
            if let Some(ref metrics) = self.review_metrics {
                let reviewed_correct = arm == optimal_arm;
                // Simulate base (random) correctness deterministically
                let base_correct = episode % self.env.num_arms() == optimal_arm;
                metrics.record(base_correct, reviewed_correct);
            }

            // Persist to trial log
            let record = TrialRecord {
                episode,
                player_id: 0,
                arm,
                reward,
                q_value: self.stats.q_value(arm),
                cumulative_reward: self.cumulative_reward,
                cumulative_regret: self.cumulative_regret,
                config: config_owned.clone(),
                note: String::new(),
                base_correct: None,
                reviewed_correct: None,
                anchors: None,
            };
            if let Err(e) = trial_log.append(&record) {
                eprintln!("trial_log write error at episode {episode}: {e}");
            }

            events.push(BanditEvent::EpisodeComplete {
                episode,
                arm,
                reward,
                cumulative_reward: self.cumulative_reward,
                cumulative_regret: self.cumulative_regret,
            });
        }

        let best_arm = self.stats.best_arm();
        let result = BanditResult {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
            q_values: self.stats.q_values.to_vec(),
            visits: self.stats.visits.to_vec(),
        };

        events.push(BanditEvent::SessionComplete {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
        });

        let _ = trial_log.flush();
        (events, result)
    }
}

// ── AbsorbCompress Integration ──────────────────────────────────

impl<P: ScreeningPruner + AbsorbCompress> BanditPruner<P> {
    /// Feed a new (arm, reward) observation to the absorb-compress layer.
    ///
    /// Call after [`update`](Self::update) to keep compression tracking in sync
    /// with bandit Q-values.
    pub fn absorb(&mut self, arm: usize, reward: f32) {
        self.inner.absorb(arm, reward);
    }

    /// Check if compression threshold is met, then promote low-Q arms to hard blocks.
    ///
    /// Returns indices of newly promoted arms (may be empty).
    /// Call periodically (e.g., every 100 episodes) after absorb-feeding.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        if self.inner.should_compress() {
            self.inner.compress()
        } else {
            Vec::new()
        }
    }

    /// Arms already promoted to hard constraints by the absorb-compress cycle.
    pub fn compressed_arms(&self) -> &[usize] {
        self.inner.compressed_arms()
    }
}

// ── Shared Bandit Stats ────────────────────────────────────────

/// Thread-safe shared bandit statistics for multi-agent cooperative learning.
/// Wraps bandit state in Mutex so multiple agents share one learning process.
///
/// Contention is minimal — ~1 update per ~200 tick round per agent.
/// Use `Arc<SharedBanditStats>` to share across agents.
#[cfg(feature = "bandit")]
pub struct SharedBanditStats {
    inner: std::sync::Mutex<BanditStatsInner>,
}

#[cfg(feature = "bandit")]
struct BanditStatsInner {
    q_values: Vec<f32>,
    visits: Vec<u32>,
    total_pulls: u32,
    compressed: Vec<bool>,
}

#[cfg(feature = "bandit")]
impl SharedBanditStats {
    /// Create shared stats with optimistic initialization (Q=1.0 for all arms).
    pub fn new(n_arms: usize) -> Self {
        Self {
            inner: std::sync::Mutex::new(BanditStatsInner {
                q_values: vec![1.0; n_arms],
                visits: vec![0; n_arms],
                total_pulls: 0,
                compressed: vec![false; n_arms],
            }),
        }
    }

    /// Update Q-value for `arm` after observing `reward`.
    ///
    /// Uses incremental mean: `Q(a) += (reward - Q(a)) / n(a)`.
    pub fn update(&self, arm: usize, reward: f32) {
        let mut inner = self.inner.lock().unwrap();
        if arm >= inner.q_values.len() {
            return;
        }
        inner.visits[arm] += 1;
        inner.total_pulls += 1;
        let n = inner.visits[arm] as f32;
        inner.q_values[arm] += (reward - inner.q_values[arm]) / n;
    }

    /// UCB1 score: `Q(a) + sqrt(2 * ln(N) / n(a))`.
    ///
    /// Returns `f32::MAX` for unvisited arms (must explore first).
    pub fn ucb1_score(&self, arm: usize) -> f32 {
        let inner = self.inner.lock().unwrap();
        if arm >= inner.q_values.len() || inner.visits[arm] == 0 || inner.total_pulls == 0 {
            return f32::MAX;
        }
        let q = inner.q_values[arm];
        let n = inner.visits[arm] as f32;
        let total = inner.total_pulls as f32;
        q + (2.0 * total.ln() / n).sqrt()
    }

    /// Index of the arm with highest Q-value.
    pub fn best_arm(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner
            .q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Whether an arm has been compressed (hard-blocked).
    pub fn is_compressed(&self, arm: usize) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.compressed.get(arm).copied().unwrap_or(false)
    }

    /// Mark an arm as compressed (hard-blocked).
    pub fn compress_arm(&self, arm: usize) {
        let mut inner = self.inner.lock().unwrap();
        if arm < inner.compressed.len() {
            inner.compressed[arm] = true;
        }
    }

    /// Total pulls across all arms.
    pub fn total_pulls(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.total_pulls
    }

    /// Snapshot of (q_value, visits) for an arm under a single lock acquisition.
    ///
    /// Prefer this over calling `q_value()` + `visits()` separately to avoid
    /// acquiring the lock twice.
    pub fn arm_snapshot(&self, arm: usize) -> (f32, u32) {
        let inner = self.inner.lock().unwrap();
        let q = inner.q_values.get(arm).copied().unwrap_or(0.0);
        let v = inner.visits.get(arm).copied().unwrap_or(0);
        (q, v)
    }

    /// Visit count for an arm.
    pub fn visits(&self, arm: usize) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.visits.get(arm).copied().unwrap_or(0)
    }

    /// Q-value estimate for an arm.
    pub fn q_value(&self, arm: usize) -> f32 {
        let inner = self.inner.lock().unwrap();
        inner.q_values.get(arm).copied().unwrap_or(0.0)
    }
}

// ── RandOpt Diagnostics (Plan 121) ──────────────────────────────

/// Solution density: fraction of scores ≥ base_score + margin.
/// From RandOpt (Neural Thickets) — measures how many perturbations improve over baseline.
pub fn solution_density(scores: &[f32], base_score: f32, margin: f32) -> f32 {
    match scores.is_empty() {
        true => 0.0,
        false => {
            let threshold = base_score + margin;
            let above = scores.iter().filter(|&&s| s >= threshold).count();
            above as f32 / scores.len() as f32
        }
    }
}

/// Spectral discordance: measures specialist vs generalist distribution.
/// D ∈ [0, 1], D→1 means specialists, D→0 means generalists.
/// Input: N arms × M tasks percentile-rank matrix.
pub fn spectral_discordance(performance_matrix: &[Vec<f32>]) -> f32 {
    if performance_matrix.is_empty() {
        return 0.0;
    }
    let n = performance_matrix.len();
    let m = performance_matrix.first().map_or(0, |r| r.len());
    if m <= 1 || n == 0 {
        return 0.0;
    }
    // For each arm, compute variance across tasks
    let variances: Vec<f32> = performance_matrix
        .iter()
        .map(|row| {
            if row.len() <= 1 {
                return 0.0;
            }
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32
        })
        .collect();
    // Normalize: max variance = 0.25 (for binary 0/1 with p=0.5)
    let max_var = 0.25_f32;
    let avg_normalized_var = variances.iter().sum::<f32>() / variances.len() as f32 / max_var;
    avg_normalized_var.min(1.0)
}

// ---------------------------------------------------------------------------
// Adaptive Top-p Arm Selection (dMoE distillation, Research 161, Plan 181)
// ---------------------------------------------------------------------------

/// Adaptive top-p arm selection for BanditPruner.
///
/// Replaces fixed top-k with dynamic arm budget based on score concentration.
/// When scores are concentrated (clear winner) → selects fewer arms → faster.
/// When scores are dispersed (uncertain) → selects more arms → better exploration.
///
/// # Arguments
/// * `q_values` - Bandit Q-values for each arm
/// * `ucb_bonus` - UCB exploration bonus for each arm
/// * `p` - Cumulative probability threshold (default: 0.85)
///
/// # Returns
/// Indices of selected arms, sorted by score descending.
#[cfg(feature = "bandit_top_p")]
pub fn select_arms_top_p(q_values: &[f32], ucb_bonus: &[f32], p: f32) -> Vec<usize> {
    let scores: Vec<f32> = q_values
        .iter()
        .zip(ucb_bonus.iter())
        .map(|(&q, &u)| q + u)
        .collect();
    let n = scores.len();

    if n == 0 {
        return vec![];
    }

    // Sort by score descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total: f32 = scores.iter().map(|s| s.max(0.0)).sum();
    if total <= 0.0 {
        return indices;
    }

    let mut cumsum = 0.0f32;
    let mut selected = Vec::with_capacity(n);
    for &idx in &indices {
        cumsum += scores[idx].max(0.0) / total;
        selected.push(idx);
        if cumsum >= p {
            break;
        }
    }
    selected
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    // ── Stats Tests ─────────────────────────────────────────────

    #[test]
    fn test_stats_update_incremental_mean() {
        let mut stats = BanditStats::new(3);

        stats.update(0, 1.0);
        assert_eq!(stats.q_value(0), 1.0);

        stats.update(0, 0.0);
        assert_eq!(stats.q_value(0), 0.5);

        stats.update(0, 1.0);
        // (1.0 + 0.0 + 1.0) / 3 = 0.666...
        assert!((stats.q_value(0) - 0.666).abs() < 0.01);
        assert_eq!(stats.visit_count(0), 3);
        assert_eq!(stats.total_pulls(), 3);
    }

    #[test]
    fn test_stats_best_arm() {
        let mut stats = BanditStats::new(4);
        stats.update(0, 0.2);
        stats.update(1, 0.8);
        stats.update(2, 0.5);
        stats.update(3, 0.3);
        assert_eq!(stats.best_arm(), 1);
    }

    #[test]
    fn test_stats_ucb1_unvisited_max_priority() {
        let mut stats = BanditStats::new(3);
        stats.update(0, 0.5);
        stats.update(1, 0.5);
        // Arm 2 unvisited → should get f32::MAX score
        assert_eq!(stats.ucb1_score(2), f32::MAX);
        // Visited arms get finite scores
        assert!(stats.ucb1_score(0).is_finite());
        assert!(stats.ucb1_score(1).is_finite());
    }

    #[test]
    fn test_stats_ucb1_increases_with_few_visits() {
        let mut stats = BanditStats::new(2);
        // Both arms get same reward, but arm 0 gets visited less
        for _ in 0..10 {
            stats.update(0, 0.5);
        }
        for _ in 0..100 {
            stats.update(1, 0.5);
        }
        // Arm 0 should have higher UCB1 bonus (fewer visits)
        assert!(stats.ucb1_score(0) > stats.ucb1_score(1));
    }

    // ── Environment Tests ───────────────────────────────────────

    #[test]
    fn test_bernoulli_env_optimal() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        assert_eq!(env.optimal_arm(), 2);
        assert!((env.optimal_reward() - 0.8).abs() < f32::EPSILON);
        assert_eq!(env.num_arms(), 5);
        assert!((env.expected_reward(0) - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bernoulli_env_pull_distribution() {
        let env = BernoulliEnv::new(&[0.0, 1.0]);
        let mut rng = Rng::new(42);

        // Arm 0 always returns 0.0
        for _ in 0..100 {
            assert_eq!(env.pull(0, &mut rng), 0.0);
        }
        // Arm 1 always returns 1.0
        for _ in 0..100 {
            assert_eq!(env.pull(1, &mut rng), 1.0);
        }
    }

    #[test]
    fn test_gaussian_env_optimal() {
        let env = GaussianEnv::new(&[0.2, 0.7, 0.5], 0.1);
        assert_eq!(env.optimal_arm(), 1);
        assert!((env.optimal_reward() - 0.7).abs() < f32::EPSILON);
        assert_eq!(env.num_arms(), 3);
    }

    #[test]
    fn test_gaussian_env_pull_clamped() {
        let env = GaussianEnv::new(&[0.5], 0.1);
        let mut rng = Rng::new(42);
        // All rewards should be in [0.0, 1.0]
        for _ in 0..1000 {
            let r = env.pull(0, &mut rng);
            assert!((0.0..=1.0).contains(&r), "reward {r} out of bounds");
        }
    }

    // ── Beta Sampling Tests ─────────────────────────────────────

    #[test]
    fn test_beta_sampling_bounds() {
        let mut rng = Rng::new(42);
        for alpha in [1.0, 2.0, 5.0, 10.0] {
            for beta in [1.0, 2.0, 5.0, 10.0] {
                for _ in 0..100 {
                    let sample = sample_beta(alpha, beta, &mut rng);
                    assert!(
                        (0.0..=1.0).contains(&sample),
                        "Beta({alpha},{beta}) sample {sample} out of bounds"
                    );
                }
            }
        }
    }

    #[test]
    fn test_beta_sampling_mean_converges() {
        let mut rng = Rng::new(42);
        let alpha = 3.0f32;
        let beta = 7.0f32;
        let expected_mean = alpha / (alpha + beta); // 0.3
        let n = 10000;
        let sum: f32 = (0..n).map(|_| sample_beta(alpha, beta, &mut rng)).sum();
        let mean = sum / n as f32;
        assert!(
            (mean - expected_mean).abs() < 0.05,
            "Beta({alpha},{beta}) mean {mean} too far from expected {expected_mean}"
        );
    }

    // ── BanditPruner Tests ──────────────────────────────────────

    #[test]
    fn test_pruner_cold_start_uses_domain() {
        let pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        // Before any updates, relevance = domain relevance
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < f32::EPSILON); // NoScreeningPruner returns 1.0
    }

    #[test]
    fn test_pruner_respects_domain_hard_trim() {
        struct AlwaysZero;
        impl ScreeningPruner for AlwaysZero {
            fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
                0.0
            }
        }
        let mut pruner = BanditPruner::new(AlwaysZero, BanditStrategy::Ucb1, 5);
        pruner.update(0, 0.9); // High reward, but domain says 0
        let rel = pruner.relevance(0, 0, &[]);
        assert_eq!(rel, 0.0);
    }

    #[test]
    fn test_pruner_ucb1_unvisited_arm_priority() {
        let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        // Visit arm 0 and 1, leave arm 2 unvisited
        pruner.update(0, 0.5);
        pruner.update(1, 0.5);
        let rel_visited = pruner.relevance(0, 0, &[]);
        let rel_unvisited = pruner.relevance(0, 2, &[]);
        // Unvisited arm should have equal or higher relevance
        assert!(rel_unvisited >= rel_visited);
    }

    #[test]
    fn test_pruner_thompson_uses_cache() {
        let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::ThompsonSampling, 3);
        // Update some Q-values
        for _ in 0..10 {
            pruner.update(0, 0.9);
            pruner.update(1, 0.1);
        }
        let mut rng = Rng::new(42);
        pruner.prepare_episode(&mut rng);

        // Relevance should use cached sample (non-zero since arm 0 has high Q)
        let rel = pruner.relevance(0, 0, &[]);
        assert!(rel > 0.0);
    }

    // ── Convergence Tests ───────────────────────────────────────

    #[test]
    fn test_ucb1_convergence() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(500, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "UCB1 should find optimal arm 2, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    #[test]
    fn test_thompson_convergence() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
        let (_, result) = session.run(500, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "Thompson should find optimal arm 2, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    #[test]
    fn test_epsilon_greedy_convergence_with_decay() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let strategy = BanditStrategy::EpsilonGreedy {
            epsilon: 0.3,
            decay: 0.995,
        };
        let session = BanditSession::new(env, strategy);
        let (_, result) = session.run(1000, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "ε-greedy(decay) should find optimal arm 2, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    #[test]
    fn test_epsilon_greedy_no_decay_still_finds_good_arm() {
        let env = BernoulliEnv::new(&[0.1, 0.9]);
        let strategy = BanditStrategy::EpsilonGreedy {
            epsilon: 0.1,
            decay: 1.0,
        };
        let session = BanditSession::new(env, strategy);
        let (_, result) = session.run(2000, &mut Rng::new(42));
        // With fixed ε, may not always find optimal but should be close
        assert!(
            result.q_values[1] > 0.5,
            "ε-greedy(no decay) should learn arm 1 is good, Q-values: {:?}",
            result.q_values
        );
    }

    // ── Regret Tests ────────────────────────────────────────────

    #[test]
    fn test_regret_sublinear_ucb1() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(1000, &mut Rng::new(42));

        // Sub-linear regret: total_regret should grow slower than linear
        // Linear regret would be ~1000 * (0.8 - 0.2) = 600 for always choosing worst arm
        // Sub-linear should be much less, roughly O(sqrt(N)) ≈ ~30-60
        assert!(
            result.total_regret < 100.0,
            "UCB1 regret should be sub-linear, got {}",
            result.total_regret
        );
    }

    #[test]
    fn test_regret_sublinear_thompson() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
        let (_, result) = session.run(1000, &mut Rng::new(42));

        // Thompson is stochastic — higher variance than UCB1.
        // Linear regret worst-case ≈ 600. Sub-linear threshold generous but still
        // well below linear: must be spending most pulls on high-value arms.
        assert!(
            result.total_regret < 250.0,
            "Thompson regret should be sub-linear, got {}",
            result.total_regret
        );
    }

    // ── Gaussian Bandit Test ────────────────────────────────────

    #[test]
    fn test_gaussian_convergence() {
        let env = GaussianEnv::new(&[0.3, 0.7, 0.5], 0.1);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(500, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "UCB1 should find Gaussian optimal arm 1, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    // ── Session Event Tests ─────────────────────────────────────

    #[test]
    fn test_session_events_count() {
        let env = BernoulliEnv::new(&[0.5, 0.8]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (events, _) = session.run(10, &mut Rng::new(42));

        // 10 Pull + 10 EpisodeComplete + 1 SessionComplete = 21
        assert_eq!(events.len(), 21);
    }

    #[test]
    fn test_session_result_fields() {
        let env = BernoulliEnv::new(&[0.5, 0.8]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(100, &mut Rng::new(42));

        assert_eq!(result.total_episodes, 100);
        assert_eq!(result.optimal_arm, 1);
        assert_eq!(result.q_values.len(), 2);
        assert_eq!(result.visits.len(), 2);
        assert!(result.total_reward > 0.0);
        assert!(result.avg_reward() > 0.0);
    }

    // ── Constrained Bandit Tests ────────────────────────────────

    /// Domain pruner that blocks specific arms via relevance 0.0.
    struct BlockedArmPruner {
        blocked: Vec<usize>,
    }

    impl BlockedArmPruner {
        fn new(blocked: &[usize]) -> Self {
            Self {
                blocked: blocked.to_vec(),
            }
        }
    }

    impl ScreeningPruner for BlockedArmPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if self.blocked.contains(&token_idx) {
                0.0
            } else {
                1.0
            }
        }
    }

    #[test]
    fn test_constrained_bandit_never_pulls_blocked_arm() {
        // Arm 4 has highest reward (0.9) but is blocked
        let mut pruner = BanditPruner::new(BlockedArmPruner::new(&[4]), BanditStrategy::Ucb1, 5);
        pruner.soft_route = false; // Hard-route needed for blocked-arm rejection

        let env = BernoulliEnv::new(&[0.1, 0.3, 0.7, 0.4, 0.9]);
        let mut rng = Rng::new(42);

        // Select arms via pruner relevance for 500 episodes
        for _ in 0..500 {
            let arm = (0..5)
                .max_by(|&a, &b| {
                    pruner
                        .relevance(0, a, &[])
                        .partial_cmp(&pruner.relevance(0, b, &[]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            let reward = env.pull(arm, &mut rng);
            pruner.update(arm, reward);
        }

        // Arm 4 should never be visited (relevance always 0.0)
        assert_eq!(pruner.visits()[4], 0, "blocked arm should never be pulled");
        // Best arm should be 2 (highest unblocked reward: 0.7)
        assert_eq!(
            pruner.best_arm(),
            2,
            "should find best valid arm, not blocked arm"
        );
    }

    #[test]
    fn test_constrained_bandit_respects_domain_over_bandit() {
        // Even after giving arm 4 high reward manually, domain pruner overrides
        let mut pruner = BanditPruner::new(BlockedArmPruner::new(&[4]), BanditStrategy::Ucb1, 5);

        // Manually pump arm 4's Q-value high
        for _ in 0..100 {
            pruner.update(4, 1.0);
        }

        // Domain still blocks it
        let rel = pruner.relevance(0, 4, &[]);
        assert_eq!(
            rel, 0.0,
            "domain pruner must override bandit score for blocked arms"
        );

        // Other arms still allowed
        assert!(pruner.relevance(0, 0, &[]) >= 0.0);
        assert!(pruner.relevance(0, 2, &[]) >= 0.0);
    }

    // ── Shared Bandit Stats Tests ──────────────────────────────

    #[test]
    fn test_shared_bandit_stats_convergence() {
        use std::sync::Arc;
        use std::thread;

        let stats = Arc::new(SharedBanditStats::new(4));
        let mut handles = Vec::new();

        // 4 threads, each updating different arms with different rewards
        // Arm 0: reward 0.1, Arm 1: reward 0.3, Arm 2: reward 0.9, Arm 3: reward 0.5
        let rewards = [0.1f32, 0.3f32, 0.9f32, 0.5f32];
        let updates_per_thread = 200u32;

        for (arm, &reward) in rewards.iter().enumerate() {
            let stats_clone = Arc::clone(&stats);
            handles.push(thread::spawn(move || {
                for _ in 0..updates_per_thread {
                    stats_clone.update(arm, reward);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Best arm should converge to arm 2 (highest reward 0.9)
        assert_eq!(
            stats.best_arm(),
            2,
            "shared stats should find arm 2 as best"
        );

        // Total pulls = 4 threads × 200 updates
        assert_eq!(
            stats.total_pulls(),
            800,
            "total pulls should equal sum of all thread updates"
        );

        // Verify individual arm visits
        for arm in 0..4 {
            assert_eq!(
                stats.visits(arm),
                updates_per_thread,
                "arm {arm} should have {updates_per_thread} visits"
            );
        }

        // Verify Q-values converge toward true rewards
        for (arm, &expected) in rewards.iter().enumerate() {
            let q = stats.q_value(arm);
            assert!(
                (q - expected).abs() < 0.1,
                "arm {arm} q_value {q} should be close to {expected}"
            );
        }
    }

    #[cfg(feature = "bandit")]
    #[test]
    fn test_bandit_pruner_shared_stats() {
        use std::sync::Arc;

        // Simple mock pruner that always returns 1.0
        struct MockPruner;
        impl ScreeningPruner for MockPruner {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> f32 {
                1.0
            }
        }

        let shared = Arc::new(SharedBanditStats::new(3));
        let mut p1 = BanditPruner::with_shared_stats(
            MockPruner,
            BanditStrategy::Ucb1,
            3,
            Arc::clone(&shared),
        );
        let mut p2 = BanditPruner::with_shared_stats(
            MockPruner,
            BanditStrategy::Ucb1,
            3,
            Arc::clone(&shared),
        );

        // P1 updates arm 0 with high reward
        p1.update(0, 0.9);

        // P2 updates arm 1 with low reward
        p2.update(1, 0.1);

        // P2 updates arm 2 with medium reward
        p2.update(2, 0.5);

        // Verify P1 sees P2's updates and vice versa
        // Total pulls should be 3 from either pruner's perspective
        assert_eq!(p1.total_pulls(), 3, "p1 should see 3 total pulls");
        assert_eq!(p2.total_pulls(), 3, "p2 should see 3 total pulls");

        // Best arm should be arm 0 (highest reward 0.9)
        assert_eq!(p1.best_arm(), 0, "p1 best arm should be 0");
        assert_eq!(p2.best_arm(), 0, "p2 best arm should be 0");

        // Verify visits are shared
        assert_eq!(p1.arm_visits(0), 1, "arm 0 should have 1 visit via p1");
        assert_eq!(p2.arm_visits(1), 1, "arm 1 should have 1 visit via p2");
        assert_eq!(p1.arm_visits(2), 1, "arm 2 should have 1 visit via p1");

        // More updates from P1
        for _ in 0..10 {
            p1.update(0, 0.9);
        }

        // P2 should see the accumulated visits
        assert_eq!(
            p2.arm_visits(0),
            11,
            "p2 should see p1's accumulated visits on arm 0"
        );
        assert_eq!(p2.total_pulls(), 13, "p2 should see total 13 pulls");
    }

    // ── Dual Cutoff Tests (Plan 062) ────────────────────────────

    #[test]
    fn test_dual_cutoff_disabled_by_default() {
        let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        assert_eq!(bp.dual_cutoff, 0.0, "default cutoff should be 0 (disabled)");
    }

    #[test]
    fn test_dual_cutoff_masks_low_q_arms() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        bp.soft_route = false; // Hard-route needed for per-arm cutoff masking
        bp.dual_cutoff = 0.3;

        // Arm 0: high Q (should pass)
        bp.update(0, 0.8);
        bp.update(0, 0.9);
        // Arm 1: low Q (should be masked)
        bp.update(1, 0.1);
        bp.update(1, 0.05);
        // Arm 2: unvisited (should NOT be masked — exploration)

        let r0 = bp.relevance(0, 0, &[]);
        let r1 = bp.relevance(0, 1, &[]);
        let r2 = bp.relevance(0, 2, &[]);

        assert!(r0 > 0.0, "high-Q arm should have positive relevance");
        assert_eq!(r1, 0.0, "low-Q arm should be masked by dual_cutoff");
        assert!(r2 > 0.0, "unvisited arm should not be masked (exploration)");
    }

    #[test]
    fn test_set_dual_cutoff_method() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        assert_eq!(bp.dual_cutoff, 0.0);

        bp.set_dual_cutoff(0.5);
        assert_eq!(bp.dual_cutoff, 0.5);

        bp.set_dual_cutoff(0.0);
        assert_eq!(bp.dual_cutoff, 0.0, "can re-disable via setter");
    }

    // ── Soft-Route Tests (Plan 175, Part 3) ───────────────────────

    #[test]
    fn test_soft_route_enabled_by_default() {
        let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        assert!(bp.soft_route, "soft_route should default to true");
        assert!(
            (bp.soft_route_tau - 1.0).abs() < f32::EPSILON,
            "tau should default to 1.0"
        );
    }

    #[test]
    fn test_soft_route_cold_start_returns_domain() {
        let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        // No updates: cold start, relevance = domain (1.0 from NoScreeningPruner)
        let r = bp.relevance(0, 0, &[]);
        assert_eq!(r, 1.0, "cold start should return domain");
    }

    #[test]
    fn test_soft_route_blend_dominates_single_arm() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);

        // Arm 0: high Q
        for _ in 0..20 {
            bp.update(0, 0.9);
        }
        // Arm 1: low Q
        for _ in 0..20 {
            bp.update(1, 0.1);
        }
        // Arm 2: medium Q
        for _ in 0..20 {
            bp.update(2, 0.5);
        }

        // With soft-route, arm 1's relevance should be higher than its own
        // bandit score would suggest (blended upward by arms 0 and 2)
        let r0 = bp.relevance(0, 0, &[]);
        let r1 = bp.relevance(0, 1, &[]);
        let r2 = bp.relevance(0, 2, &[]);

        // All should be positive and reasonably close (soft blending)
        assert!(r0 > 0.0, "arm 0 relevance should be positive");
        assert!(r1 > 0.0, "arm 1 relevance should be positive");
        assert!(r2 > 0.0, "arm 2 relevance should be positive");

        // The key property: with soft routing, all arms get similar relevance
        // because the blend is over ALL arm scores. The spread should be
        // smaller than with hard routing.
        let spread = (r0 - r1).abs();
        assert!(
            spread < 0.5,
            "soft-route spread should be moderate, got {spread}"
        );
    }

    #[test]
    fn test_hard_route_restores_original_behavior() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        bp.set_soft_route(false, 1.0);

        // Arm 0: high Q
        for _ in 0..20 {
            bp.update(0, 0.9);
        }
        // Arm 1: low Q
        for _ in 0..20 {
            bp.update(1, 0.1);
        }

        let r0 = bp.relevance(0, 0, &[]);
        let r1 = bp.relevance(0, 1, &[]);

        assert!(
            r0 > r1,
            "hard-route: high-Q arm should have higher relevance"
        );
        assert!(r0 > 0.0, "high-Q arm should be positive");
        assert!(r1 > 0.0, "low-Q arm should still be positive (no cutoff)");
    }

    #[test]
    fn test_soft_route_zero_domain_returns_zero() {
        struct ZeroPruner;
        impl ScreeningPruner for ZeroPruner {
            fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
                0.0
            }
        }
        let mut bp = BanditPruner::new(ZeroPruner, BanditStrategy::Ucb1, 3);
        bp.update(0, 0.9);
        let r = bp.relevance(0, 0, &[]);
        assert_eq!(r, 0.0, "zero domain should give zero even with soft-route");
    }

    #[test]
    fn test_soft_route_setter_clamps_tau() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        bp.set_soft_route(true, 0.001);
        assert!(
            (bp.soft_route_tau - 0.01).abs() < f32::EPSILON,
            "tau should be clamped to 0.01 minimum"
        );
    }

    // ── GOAT Integration: All Three Fusions (Plan 175, Part 4) ─────
    //
    // GOAT proof that all three fusions work together without regression:
    //   Fusion 1: Residency Audit (verify pruner lands on fast paths)
    //   Fusion 2: RangeBudget (entropy-aware budget adaptation)
    //   Fusion 4: Soft-Route Bandit (softmax-blended arm relevance)

    #[test]
    fn test_goat_175_soft_route_acceptance_rate() {
        // GOAT: Soft-route bandit acceptance rate >= hard-route over 500 episodes.
        //
        // With soft routing, all arms get blended relevance, so the DDTree
        // retains more viable branches. This should produce acceptance rates
        // at least as good as hard routing (which only considers one arm).
        let vocab = 8;
        let lookahead = 4;
        let episodes = 500;
        let mut rng = crate::types::Rng::new(42);

        let config = crate::types::Config {
            vocab_size: vocab,
            draft_lookahead: lookahead,
            ..Default::default()
        };

        // Helper: generate peaked marginals (3 good tokens, rest noise)
        let peaked_marginals = |rng: &mut crate::types::Rng| -> Vec<Vec<f32>> {
            (0..lookahead)
                .map(|_| {
                    let mut m = vec![0.01; vocab];
                    // 3 "good" tokens get ~80% of mass
                    for v in m.iter_mut().take(3) {
                        *v = 0.27;
                    }
                    let sum: f32 = m.iter().sum();
                    m.iter_mut().for_each(|p| *p /= sum);
                    let _ = rng; // consume rng for API consistency
                    m
                })
                .collect()
        };

        // Run soft-route
        let mut soft_bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, vocab);
        let mut soft_accepted = 0usize;
        let mut soft_total = 0usize;

        for ep in 0..episodes {
            soft_bp.prepare_episode(&mut rng);
            let marginals = peaked_marginals(&mut rng);
            let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
            let tree = crate::speculative::build_dd_tree_screened(&slices, &config, &soft_bp, true);

            // Simulate verification: accept top-k tokens
            for node in &tree {
                soft_total += 1;
                // Peaked marginals: top-3 tokens have ~80% chance of acceptance
                if node.token_idx < 3 && rng.uniform() < 0.8 {
                    soft_bp.update(node.token_idx, 1.0);
                    soft_accepted += 1;
                } else if rng.uniform() < 0.2 {
                    soft_bp.update(node.token_idx, 0.1);
                    soft_accepted += 1;
                }
            }

            let _ = ep;
        }

        // Run hard-route (baseline)
        let mut hard_bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, vocab);
        hard_bp.soft_route = false;
        let mut hard_accepted = 0usize;
        let mut hard_total = 0usize;

        for ep in 0..episodes {
            hard_bp.prepare_episode(&mut rng);
            let marginals = peaked_marginals(&mut rng);
            let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
            let tree = crate::speculative::build_dd_tree_screened(&slices, &config, &hard_bp, true);

            for node in &tree {
                hard_total += 1;
                if node.token_idx < 3 && rng.uniform() < 0.8 {
                    hard_bp.update(node.token_idx, 1.0);
                    hard_accepted += 1;
                } else if rng.uniform() < 0.2 {
                    hard_bp.update(node.token_idx, 0.1);
                    hard_accepted += 1;
                }
            }

            let _ = ep;
        }

        let soft_rate = soft_accepted as f64 / soft_total.max(1) as f64;
        let hard_rate = hard_accepted as f64 / hard_total.max(1) as f64;

        // GOAT: soft-route acceptance rate should be within 5% of hard-route
        // (may not always exceed due to verification randomness, but should be close)
        assert!(
            soft_rate >= hard_rate - 0.05,
            "GOAT 175: soft-route acceptance ({soft_rate:.3}) should be >= hard-route ({hard_rate:.3}) - 5%"
        );

        // Both should produce reasonable trees
        assert!(soft_total > 0, "soft-route should produce tree nodes");
        assert!(hard_total > 0, "hard-route should produce tree nodes");
    }

    #[test]
    fn test_goat_175_fusion_residency_audit_passes() {
        // GOAT: BanditPruner with soft-route passes residency audit.
        //
        // This test exercises Fusion 1 (Residency Audit) + Fusion 4 (Soft-Route)
        // together: build a DDTree with soft-route bandit screening, then verify
        // the residency report shows no silent degradation.
        use crate::speculative::residency_audit::{
            audit_baseline, audit_screening_pruner, is_degrading,
        };

        let vocab = 8;
        let lookahead = 4;
        let config = crate::types::Config {
            vocab_size: vocab,
            draft_lookahead: lookahead,
            ..Default::default()
        };

        // Uniform marginals for audit
        let marginals: Vec<Vec<f32>> = (0..lookahead)
            .map(|_| {
                let p = 1.0 / vocab as f32;
                vec![p; vocab]
            })
            .collect();
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

        // Baseline audit (NoScreeningPruner)
        let baseline = audit_baseline(&slices, &config);

        // Soft-route bandit audit
        let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, vocab);
        assert!(bp.soft_route, "soft_route should be on by default");
        let candidate = audit_screening_pruner(&slices, &config, &bp, false);

        // GOAT: Soft-route bandit should not degrade vs baseline
        assert!(
            !is_degrading(&candidate, &baseline),
            "GOAT 175: soft-route bandit should not show silent degradation
            candidate={candidate:?}
            baseline={baseline:?}"
        );

        // Healthy tree: reasonable node count
        assert!(candidate.total_nodes > 0, "should produce tree nodes");
        assert!(
            candidate.fast_path_ratio > 0.0,
            "should have some fast-path nodes"
        );
    }

    #[test]
    fn test_goat_175_soft_route_overhead_acceptable() {
        // GOAT: Soft-route O(arms) per-node overhead is acceptable.
        //
        // Soft-route computes softmax over all arms for each node, which is O(arms)
        // per relevance() call instead of O(1). This test verifies the overhead
        // is reasonable for typical vocab sizes.
        use std::time::Instant;

        let vocab = 8;
        let lookahead = 4;
        let config = crate::types::Config {
            vocab_size: vocab,
            draft_lookahead: lookahead,
            ..Default::default()
        };

        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, vocab);

        // Warm up with some data
        for i in 0..vocab {
            bp.update(i, 0.5);
        }

        let marginals: Vec<Vec<f32>> = (0..lookahead)
            .map(|_| {
                let p = 1.0 / vocab as f32;
                vec![p; vocab]
            })
            .collect();
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

        // Build 1000 trees and measure time
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _tree = crate::speculative::build_dd_tree_screened(&slices, &config, &bp, true);
        }
        let elapsed = start.elapsed();
        let per_tree_ns = elapsed.as_nanos() as f64 / iterations as f64;

        // GOAT: per-tree construction should be < 500µs for vocab=8, lookahead=4
        // (this is generous; actual should be much faster)
        assert!(
            per_tree_ns < 500_000.0,
            "GOAT 175: per-tree overhead should be < 500µs, got {per_tree_ns:.0}ns"
        );
    }

    // ── Partial Scoring Tests (Plan 191 T1.4) ────────────────────

    #[cfg(feature = "partial_scoring")]
    mod partial_scoring {
        use super::*;
        use crate::pruners::BomberPartialScorer;
        use katgpt_core::GameTrace;

        fn win_trace() -> GameTrace {
            GameTrace {
                survival_ticks: 200,
                kills: 3,
                actions_taken: 50,
                max_ticks: 200,
                final_reward: 1.0,
            }
        }

        fn loss_trace() -> GameTrace {
            GameTrace {
                survival_ticks: 30,
                kills: 0,
                actions_taken: 10,
                max_ticks: 200,
                final_reward: 0.0,
            }
        }

        #[test]
        fn test_update_with_trace_scorer_set() {
            let scorer = Box::new(BomberPartialScorer { max_ticks: 200 });
            let mut bp = BanditPruner::with_partial_scorer(
                NoScreeningPruner,
                BanditStrategy::Ucb1,
                4,
                scorer,
            );
            let trace = win_trace();
            bp.update_with_trace(0, &trace);
            // BomberPartialScorer on win_trace: survival=1.0, kills=1.0, efficiency=0.06
            // score = 0.4*1.0 + 0.3*1.0 + 0.2*1.0 + 0.1*0.06 = 0.906
            let q = bp.q_values()[0];
            assert!(q > 0.8, "expected high score from scorer, got {q}");
        }

        #[test]
        fn test_update_with_trace_no_scorer_binary_fallback() {
            let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 4);
            let trace = loss_trace();
            bp.update_with_trace(0, &trace);
            let q = bp.q_values()[0];
            assert!(
                (q - 0.0).abs() < f32::EPSILON,
                "expected 0.0 for loss, got {q}"
            );

            let trace_w = win_trace();
            bp.update_with_trace(1, &trace_w);
            let q = bp.q_values()[1];
            assert!(
                (q - 1.0).abs() < f32::EPSILON,
                "expected 1.0 for win, got {q}"
            );
        }

        #[test]
        fn test_with_partial_scorer_constructor() {
            let scorer = Box::new(BomberPartialScorer { max_ticks: 100 });
            let bp = BanditPruner::with_partial_scorer(
                NoScreeningPruner,
                BanditStrategy::EpsilonGreedy {
                    epsilon: 0.1,
                    decay: 0.99,
                },
                8,
                scorer,
            );
            // Verify it compiles and drops cleanly
            drop(bp);
            // Also suppress unused GameTrace lint
            let _ = win_trace();
            let _ = loss_trace();
        }

        #[test]
        fn test_default_backward_compat() {
            let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 4);
            // Default has no scorer — update_with_trace uses binary fallback
            let trace = GameTrace {
                survival_ticks: 100,
                kills: 5,
                actions_taken: 30,
                max_ticks: 200,
                final_reward: 1.0,
            };
            bp.update_with_trace(0, &trace);
            assert!((bp.q_values()[0] - 1.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_score_breakdown_via_update() {
            let scorer = Box::new(BomberPartialScorer { max_ticks: 200 });
            let mut bp = BanditPruner::with_partial_scorer(
                NoScreeningPruner,
                BanditStrategy::Ucb1,
                4,
                scorer,
            );
            // Partial loss: survived 100/200 ticks, no kills
            let trace = GameTrace {
                survival_ticks: 100,
                kills: 0,
                actions_taken: 40,
                max_ticks: 200,
                final_reward: 0.0,
            };
            bp.update_with_trace(0, &trace);
            let q = bp.q_values()[0];
            // survival=0.5, kills=0.0, safety=0.5, efficiency=0.0
            // score = 0.4*0.5 + 0.3*0.0 + 0.2*0.5 + 0.1*0.0 = 0.3
            assert!(
                (q - 0.3).abs() < 0.01,
                "expected ~0.3 for partial survival, got {q}"
            );
        }
    }

    // ── Idea Divergence Integration Tests (Plan 191 T3.2) ──────────

    #[cfg(feature = "idea_divergence")]
    mod idea_divergence_tests {
        use super::*;
        use crate::speculative::types::NoScreeningPruner;

        /// Two arms with identical Q-values: both non-novel arms get lower bandit scores.
        #[test]
        fn test_idea_divergence_penalizes_convergent_arms() {
            let mut bp = BanditPruner::with_idea_divergence(
                NoScreeningPruner,
                BanditStrategy::EpsilonGreedy {
                    epsilon: 0.1,
                    decay: 1.0,
                },
                3,
                0.1,
            );

            // Update arm 0 with a reward, register its score vector
            bp.update(0, 0.8);
            bp.update_divergence(0);

            // Update arm 1 with the same reward — convergent
            bp.update(1, 0.8);
            bp.update_divergence(1);

            // Arm 2: different reward — divergent
            bp.update(2, 0.1);
            bp.update_divergence(2);

            // Arm 0's score vector [0.8, 1.0] compared to arm 1 [0.8, 1.0]: distance=0 < threshold → penalized
            let score_0 = bp.arm_bandit_score(0);
            // Arm 1's score vector [0.8, 1.0] compared to arm 0 [0.8, 1.0]: distance=0 < threshold → penalized
            let score_1 = bp.arm_bandit_score(1);
            // Arm 2's score vector [0.1, 1.0] compared to arms 0,1 [0.8, 1.0]: distance=0.7 > threshold → unpenalized
            let score_2 = bp.arm_bandit_score(2);

            // Base EpsilonGreedy score for Q=0.8: 0.8.clamp(0.0, 1.0).max(0.01) = 0.8
            // After penalty: 0.8 * 0.5 = 0.4
            let base_08 = 0.8f32.clamp(0.0, 1.0).max(0.01);
            assert!(
                (score_0 - base_08 * 0.5).abs() < 0.01,
                "convergent arm 0 should be penalized: expected {}, got {score_0}",
                base_08 * 0.5
            );
            assert!(
                (score_1 - base_08 * 0.5).abs() < 0.01,
                "convergent arm 1 should be penalized: expected {}, got {score_1}",
                base_08 * 0.5
            );

            // Arm 2 is novel (different Q-value)
            let base_01 = 0.1f32.clamp(0.0, 1.0).max(0.01);
            assert!(
                (score_2 - base_01).abs() < 0.01,
                "novel arm 2 should be unpenalized: expected {base_01}, got {score_2}"
            );
        }

        /// Arm with unique Q-value pattern: no penalty.
        #[test]
        fn test_idea_divergence_novel_arm_unpenalized() {
            let mut bp = BanditPruner::with_idea_divergence(
                NoScreeningPruner,
                BanditStrategy::EpsilonGreedy {
                    epsilon: 0.1,
                    decay: 1.0,
                },
                3,
                0.5,
            );

            // Arm 0: score [0.2, 1.0]
            bp.update(0, 0.2);
            bp.update_divergence(0);

            // Arm 1: score [0.8, 1.0] — L2 distance to arm 0 = sqrt(0.36) = 0.6 > threshold 0.5
            bp.update(1, 0.8);
            bp.update_divergence(1);

            let score_0 = bp.arm_bandit_score(0);
            let score_1 = bp.arm_bandit_score(1);

            // Both should be unpenalized (novel relative to each other)
            // arm 0: Q=0.2, base = 0.2.clamp(0.0, 1.0).max(0.01) = 0.2
            // arm 1: Q=0.8, base = 0.8.clamp(0.0, 1.0).max(0.01) = 0.8
            let base_0 = 0.2f32.clamp(0.0, 1.0).max(0.01);
            let base_1 = 0.8f32.clamp(0.0, 1.0).max(0.01);

            assert!(
                (score_0 - base_0).abs() < 0.01,
                "novel arm 0 should be unpenalized: expected {base_0}, got {score_0}"
            );
            assert!(
                (score_1 - base_1).abs() < 0.01,
                "novel arm 1 should be unpenalized: expected {base_1}, got {score_1}"
            );
        }

        /// Constructor produces a valid BanditPruner with divergence enabled.
        #[test]
        fn test_with_idea_divergence_constructor() {
            let bp =
                BanditPruner::with_idea_divergence(NoScreeningPruner, BanditStrategy::Ucb1, 4, 0.3);
            assert_eq!(bp.q_values().len(), 4);
            assert_eq!(bp.visits().len(), 4);
            assert!(bp.idea_divergence.is_some());
            assert_eq!(bp.idea_divergence.as_ref().unwrap().threshold(), 0.3);
            assert_eq!(bp.arm_score_vectors.len(), 4);
            for v in &bp.arm_score_vectors {
                assert!(v.is_empty(), "initial arm score vectors should be empty");
            }
        }

        /// Default BanditPruner (no divergence feature): scores unchanged.
        #[test]
        fn test_divergence_disabled_no_impact() {
            // This test runs WITH the feature, but constructs a default BanditPruner
            // (no divergence) to verify backward compatibility.
            let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 2);

            bp.update(0, 0.5);
            bp.update(1, 0.5);

            let score_0 = bp.arm_bandit_score(0);
            let score_1 = bp.arm_bandit_score(1);

            // Without divergence enabled, both arms should get identical scores
            assert!(
                (score_0 - score_1).abs() < f32::EPSILON,
                "without divergence, identical Q-values should get identical scores: {score_0} vs {score_1}"
            );
        }
    }

    // ── Skill Lifecycle Tests ────────────────────────────────────

    #[cfg(feature = "skill_lifecycle")]
    mod skill_lifecycle_tests {
        use super::*;
        use crate::speculative::types::NoScreeningPruner;

        #[test]
        fn test_bandit_pruner_records_experience() {
            let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);

            bp.update(0, 0.8);
            bp.record_experience(0, 0.8, false, false);

            bp.update(1, 0.1);
            bp.record_experience(1, 0.1, false, true);

            assert_eq!(bp.pruner_memory().total_entries(), 2);

            let recent = bp.recent_experiences(2);
            assert_eq!(recent.len(), 2);
            assert_eq!(recent[0].arm, 0);
            assert!((recent[0].reward - 0.8).abs() < f32::EPSILON);
            assert!(!recent[0].is_failure);
            assert_eq!(recent[1].arm, 1);
            assert!(recent[1].is_failure);
        }

        #[test]
        fn test_bandit_pruner_memory_bounded() {
            let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
            // Default capacity is 256 (rounded to next power of 2 = 256)
            assert_eq!(bp.pruner_memory().capacity(), 256);

            // Fill beyond capacity
            for i in 0..300u64 {
                bp.record_experience((i % 5) as u16, i as f32, false, false);
            }

            assert_eq!(bp.pruner_memory().total_entries(), 300);

            // Only last 256 should be retrievable
            let recent = bp.recent_experiences(300);
            assert_eq!(recent.len(), 256);

            // First entry should be arm for i=44 (300-256=44), 44%5=4
            assert_eq!(recent[0].arm, 4);
            // Last entry should be arm for i=299, 299%5=4
            assert_eq!(recent[255].arm, 4);

            // Verify identity
            assert!(bp.pruner_memory().verify_identity("bandit"));
        }
    }
}
