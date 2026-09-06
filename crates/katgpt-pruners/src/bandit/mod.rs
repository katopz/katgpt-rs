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
#[cfg(feature = "skill_lifecycle")]
use super::skill_memory::{MemoryEntry, PrunerMemory};
#[cfg(feature = "dynamic_rank")]
use crate::dynamic_rank::DynamicRankPruner;
use katgpt_speculative::ScreeningPruner;
use katgpt_types::Rng;

// ── Sibling modules (Issue 177 functional split) ─────────────────
//
// Each sibling owns one cohesive section; mod.rs re-exports the public surface
// so callers see no change vs the pre-split single-file layout.

mod environment;
mod randopt;
mod session;
#[cfg(feature = "bandit")]
mod shared_stats;

pub use environment::*;
pub use randopt::*;
pub use session::*;
#[cfg(feature = "bandit")]
pub use shared_stats::*;

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

/// Welford online variance tracking — lazily allocated only when
/// `BanditStrategy::VarianceEpsilon` is used.
///
/// Extracted from `BanditStats` to eliminate 48 bytes of struct bloat
/// (2 × `Vec<f32>`) from the hot-path `update()` call for strategies
/// that never read variance data (Ucb1, EpsilonGreedy, ThompsonSampling).
/// See Issue 036 cache-line analysis.
struct WelfordStats {
    /// Running M2 for Welford variance (per arm).
    reward_m2: Vec<f32>,
    /// Running mean reward for variance tracking (per arm).
    reward_mean: Vec<f32>,
}

/// Shared arm tracking state: Q-values, visit counts, scoring.
///
/// Used by both [`BanditPruner`] and [`BanditSession`] to avoid duplication.
/// All methods are O(1) including [`BanditStats::best_arm`] (cached).
///
/// Welford variance tracking (`reward_m2`, `reward_mean`) is lazily
/// allocated behind `Option<Box<WelfordStats>>` — `None` by default,
/// only allocated via [`enable_variance_tracking`](Self::enable_variance_tracking)
/// when `VarianceEpsilon` strategy is used. This keeps `BanditStats` at
/// 80 bytes (fits in ~2 cache lines) instead of 120 bytes (3 cache lines)
/// for the common case.
pub struct BanditStats {
    q_values: Vec<f32>,
    visits: Vec<u32>,
    total_pulls: u32,
    num_arms: usize,
    /// Cached index of the arm with highest Q-value.
    best_arm_idx: usize,
    /// Lazily-allocated Welford variance tracking.
    ///
    /// `None` for Ucb1/EpsilonGreedy/ThompsonSampling (the common case) —
    /// `update()` skips the Welford arithmetic entirely (0 heap accesses).
    /// `Some` only when `VarianceEpsilon` strategy is used.
    welford: Option<Box<WelfordStats>>,
}

impl BanditStats {
    /// Create zero-initialized stats for `num_arms` arms.
    ///
    /// Welford variance tracking is **not** allocated by default.
    /// Call [`enable_variance_tracking`](Self::enable_variance_tracking)
    /// if the strategy needs reward variance data (e.g. `VarianceEpsilon`).
    pub fn new(num_arms: usize) -> Self {
        Self {
            q_values: vec![0.0; num_arms],
            visits: vec![0; num_arms],
            total_pulls: 0,
            num_arms,
            best_arm_idx: num_arms.saturating_sub(1),
            welford: None,
        }
    }

    /// Enable Welford variance tracking.
    ///
    /// Allocates `reward_m2` and `reward_mean` Vecs. After this call,
    /// `update()` will maintain running variance statistics, and
    /// `reward_variance()` / `mean_reward_variance()` will return
    /// meaningful values.
    ///
    /// Only needed for `BanditStrategy::VarianceEpsilon`. Other strategies
    /// (Ucb1, EpsilonGreedy, ThompsonSampling) never read variance data
    /// and should leave this disabled (the default) for zero overhead.
    pub fn enable_variance_tracking(&mut self) {
        if self.welford.is_none() {
            self.welford = Some(Box::new(WelfordStats {
                reward_m2: vec![0.0; self.num_arms],
                reward_mean: vec![0.0; self.num_arms],
            }));
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

        // Welford's online algorithm for reward variance tracking.
        // Only executed when variance tracking is enabled (VarianceEpsilon
        // strategy). For Ucb1/EpsilonGreedy/ThompsonSampling, `welford` is
        // `None` and this entire block is skipped — 0 heap accesses, 0
        // extra arithmetic (Issue 036 cache-line optimization).
        if let Some(w) = self.welford.as_mut() {
            // `n` already holds visits[arm] as f32 from the Q-value update above;
            // reuse it instead of re-indexing and re-casting.
            let old_mean = w.reward_mean[arm];
            let new_mean = old_mean + (reward - old_mean) / n;
            w.reward_m2[arm] += (reward - old_mean) * (reward - new_mean);
            w.reward_mean[arm] = new_mean;
        }
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
            // float_order: a NaN q-value must never win best-arm.
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i)
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
    /// Returns 0.0 for unvisited arms, arms with fewer than 2 samples,
    /// or when variance tracking is not enabled (non-`VarianceEpsilon` strategies).
    #[inline]
    pub fn reward_variance(&self, arm: usize) -> f32 {
        if arm >= self.num_arms || self.visits[arm] < 2 {
            return 0.0;
        }
        match &self.welford {
            Some(w) => w.reward_m2[arm] / (self.visits[arm] - 1) as f32,
            None => 0.0,
        }
    }

    /// Mean reward variance across all visited arms.
    ///
    /// Arms with fewer than 2 samples are excluded from the average.
    /// Returns 0.0 when variance tracking is not enabled.
    pub fn mean_reward_variance(&self) -> f32 {
        let Some(w) = &self.welford else {
            return 0.0;
        };
        let mut sum = 0.0f32;
        let mut count = 0u32;
        // Inline reward_variance computation: we already know i < num_arms
        // and visits[i] >= 2 here, so skip the per-arm bounds check.
        for i in 0..self.num_arms {
            let n = self.visits[i];
            if n >= 2 {
                sum += w.reward_m2[i] / (n - 1) as f32;
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
/// plugs directly into [`build_dd_tree_screened`](katgpt_speculative::dd_tree::build_dd_tree_screened).
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
    /// Lazily allocated — `None` unless `set_soft_route(true, …)` is called,
    /// so the default `soft_route = false` path pays zero Mutex + zero heap
    /// cost. Previously these were always-allocated `Mutex<Vec<f32>>`, which
    /// added ~128 bytes of Mutex state (pthread_mutex_t × 2) + 2 heap Vecs
    /// to every `BanditPruner` even when soft-route was never enabled.
    /// Regression audit 2026-07-03: the always-on Mutex fields contributed to
    /// struct bloat that cost Bandit update() ~30% vs the May-29 peak.
    soft_route_scores: Option<Mutex<Vec<f32>>>,
    soft_route_weights: Option<Mutex<Vec<f32>>>,
}

/// Create `BanditStats` with variance tracking enabled if the strategy needs it.
/// Only `VarianceEpsilon` reads reward variance; other strategies skip it
/// for zero overhead (Issue 036 cache-line optimization).
pub(super) fn make_stats(num_arms: usize, strategy: &BanditStrategy) -> BanditStats {
    let mut stats = BanditStats::new(num_arms);
    if matches!(strategy, BanditStrategy::VarianceEpsilon { .. }) {
        stats.enable_variance_tracking();
    }
    stats
}

impl<P: ScreeningPruner> BanditPruner<P> {
    /// Create a new bandit pruner wrapping an inner pruner.
    ///
    /// `num_arms` is the vocabulary size (number of discrete actions).
    pub fn new(inner: P, strategy: BanditStrategy, num_arms: usize) -> Self {
        let stats = make_stats(num_arms, &strategy);
        Self {
            inner,
            strategy,
            stats,
            thompson_cache: vec![0.0; num_arms],
            #[cfg(feature = "bandit")]
            shared_stats: None,
            dual_cutoff: 0.0,
            soft_route: false,
            soft_route_tau: 1.0,
            #[cfg(feature = "partial_scoring")]
            partial_scorer: None,
            #[cfg(feature = "idea_divergence")]
            idea_divergence: None,
            #[cfg(feature = "idea_divergence")]
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: None,
            soft_route_weights: None,
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
        let bandit_stats = make_stats(num_arms, &strategy);
        Self {
            inner,
            strategy,
            stats: bandit_stats,
            thompson_cache: vec![0.0; num_arms],
            shared_stats: Some(stats),
            dual_cutoff: 0.0,
            soft_route: false,
            soft_route_tau: 1.0,
            #[cfg(feature = "partial_scoring")]
            partial_scorer: None,
            #[cfg(feature = "idea_divergence")]
            idea_divergence: None,
            #[cfg(feature = "idea_divergence")]
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: None,
            soft_route_weights: None,
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
        let stats = make_stats(num_arms, &strategy);
        Self {
            inner,
            strategy,
            stats,
            thompson_cache: vec![0.0; num_arms],
            #[cfg(feature = "bandit")]
            shared_stats: None,
            dual_cutoff: 0.0,
            soft_route: false,
            soft_route_tau: 1.0,
            partial_scorer: Some(scorer),
            #[cfg(feature = "idea_divergence")]
            idea_divergence: None,
            #[cfg(feature = "idea_divergence")]
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: None,
            soft_route_weights: None,
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
        let stats = make_stats(num_arms, &strategy);
        Self {
            inner,
            strategy,
            stats,
            thompson_cache: vec![0.0; num_arms],
            #[cfg(feature = "bandit")]
            shared_stats: None,
            dual_cutoff: 0.0,
            soft_route: false,
            soft_route_tau: 1.0,
            #[cfg(feature = "partial_scoring")]
            partial_scorer: None,
            idea_divergence: Some(IdeaDivergence::new(threshold)),
            arm_score_vectors: vec![vec![]; num_arms],
            #[cfg(feature = "skill_lifecycle")]
            memory: PrunerMemory::new(256, "bandit"),
            soft_route_scores: None,
            soft_route_weights: None,
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
        let reward = if let Some(scorer) = &self.partial_scorer {
            scorer.partial_score(trace)
        } else if trace.final_reward > 0.0 {
            1.0
        } else {
            0.0
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
        // Lazily allocate scratch buffers on first enable. Stays `None`
        // for the default `soft_route = false` path — zero Mutex + zero heap.
        if enabled {
            if self.soft_route_scores.is_none() {
                self.soft_route_scores = Some(Mutex::new(Vec::with_capacity(self.stats.num_arms)));
            }
            if self.soft_route_weights.is_none() {
                self.soft_route_weights = Some(Mutex::new(Vec::with_capacity(self.stats.num_arms)));
            }
        }
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

    /// Compute bandit scores for all arms under CurvatureInfluence in a single O(N) pass.
    ///
    /// `arm_bandit_score`'s CI branch rescans all arms per call to derive the
    /// global concentration statistic, making an N-arm soft-route relevance
    /// call O(N²). This helper factors out the concentration pass and applies
    /// the boost + floor guarantee in a second O(N) sweep, preserving the
    /// exact per-arm semantics of `arm_bandit_score` (unvisited → 1.0,
    /// dual_cutoff → 0.0, then CI boost + floor clamp).
    #[inline]
    fn fill_ci_scores(&self, scores: &mut Vec<f32>, floor: f32, concentration_threshold: f32) {
        let num_arms = self.stats.num_arms;
        // Pass 1: concentration is computed over clamp(q, 0, 1).max(0.01) for
        // all arms (matches arm_bandit_score's inner scan; unvisited arms have
        // q=0 → contribute 0.01 here, but are overridden to 1.0 in pass 2).
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
        let boost = if concentration > concentration_threshold {
            concentration / concentration_threshold
        } else {
            1.0
        };
        // Pass 2: per-arm final score, honoring unvisited/dual_cutoff early
        // returns exactly like arm_bandit_score.
        scores.clear();
        // Ensure capacity once to avoid reallocation during push.
        if scores.capacity() < num_arms {
            scores.reserve(num_arms - scores.capacity());
        }
        for a in 0..num_arms {
            let s = if self.arm_visits(a) == 0 {
                1.0
            } else if self.dual_cutoff > 0.0 && self.arm_q(a) < self.dual_cutoff {
                0.0
            } else {
                let q = self.arm_q(a).clamp(0.0, 1.0).max(0.01);
                let boosted = q * boost;
                let min_score = floor * max_score.max(q);
                boosted.max(min_score).clamp(0.0, 1.0)
            };
            scores.push(s);
        }
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

        // Compute bandit scores into pre-allocated scratch buffer.
        //
        // CurvatureInfluence fast path: the strategy's concentration stat is a
        // global property of all arms, but `arm_bandit_score`'s CI branch
        // recomputes it per call. Calling that N times would be O(N²);
        // `fill_ci_scores` computes concentration once and applies it in a
        // single O(N) pass. Only eligible when idea_divergence is off (that
        // feature's per-arm scan does not factor into the concentration
        // invariant and is handled by the generic fallback below).
        //
        // Scratch buffers are lazily allocated in `set_soft_route(true, …)`;
        // if somehow `soft_route` is true but buffers are `None` (e.g. struct
        // built via `with_dynamic_rank` before enabling), fall back to a local
        // Vec — correctness over zero-alloc in that rare path.
        let mut local_scores;
        let mut scores_guard;
        let scores: &mut Vec<f32> = if let Some(m) = self.soft_route_scores.as_ref() {
            scores_guard = m.lock().unwrap();
            scores_guard.clear();
            &mut scores_guard
        } else {
            local_scores = Vec::with_capacity(num_arms);
            &mut local_scores
        };
        let use_ci_fast_path = {
            #[cfg(feature = "idea_divergence")]
            {
                false
            }
            #[cfg(not(feature = "idea_divergence"))]
            {
                matches!(self.strategy, BanditStrategy::CurvatureInfluence { .. })
            }
        };
        if use_ci_fast_path {
            if let BanditStrategy::CurvatureInfluence {
                floor,
                concentration_threshold,
            } = &self.strategy
            {
                self.fill_ci_scores(scores, *floor, *concentration_threshold);
            }
        } else {
            scores.extend((0..num_arms).map(|a| self.arm_bandit_score(a)));
        }

        // Numerical stability: subtract max before exp
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut local_weights;
        let mut weights_guard;
        let weights: &mut Vec<f32> = if let Some(m) = self.soft_route_weights.as_ref() {
            weights_guard = m.lock().unwrap();
            weights_guard.clear();
            &mut weights_guard
        } else {
            local_weights = Vec::with_capacity(num_arms);
            &mut local_weights
        };
        let inv_tau = 1.0 / tau;
        weights.extend(scores.iter().map(|&s| {
            use katgpt_core::simd::fast_exp;
            fast_exp((s - max_score) * inv_tau)
        }));
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

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../bandit_tests.rs"]
mod tests;
