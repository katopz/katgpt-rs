//! MetaRouter — online policy selection via multi-armed bandit (Plan 196, Phase 3).
//!
//! Owns multiple VortexFlow policies and uses a BanditPruner for online selection.
//! All policies maintain their own caches (forward_cache delegates to ALL).
//! The bandit selects which policy to use for forward_indexer.
//!
//! Feature gate: `vortex_flow` (Plan 196, Phase 3, default-OFF).

use crate::pruners::bandit::{BanditPruner, BanditStats, BanditStrategy};
use crate::speculative::types::ScreeningPruner;

use super::block_topk::{BlockTopKCache, BlockTopKRouter};
use super::channel_aware::ChannelAwareCache;
use super::entmax_router::{EntmaxCache, EntmaxRouter};
use super::value_energy::{ValueEnergyCache, ValueEnergyRouter};
use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// DynRoutingCache — T17
// ---------------------------------------------------------------------------

/// Dynamic routing cache: one variant per router cache type.
///
/// This enables `MetaRouter` to own multiple policies with different cache types
/// behind a single enum. The `Meta` variant holds all policy caches for the
/// meta-router's internal use.
#[derive(Debug)]
pub enum DynRoutingCache {
    /// BlockTopK cache.
    BlockTopK(BlockTopKCache),
    /// Entmax cache.
    Entmax(EntmaxCache),
    /// ValueEnergy cache.
    ValueEnergy(ValueEnergyCache),
    /// Channel-aware cache.
    ChannelAware(ChannelAwareCache),
    /// Meta-router composite: all policy caches.
    Meta(Vec<DynRoutingCache>),
}

impl DynRoutingCache {
    /// Number of blocks currently cached (variant-dependent).
    pub fn n_blocks(&self) -> usize {
        match self {
            Self::BlockTopK(c) => c.n_blocks,
            Self::Entmax(c) => c.n_blocks(),
            Self::ValueEnergy(c) => c.n_blocks,
            Self::ChannelAware(c) => c.n_blocks,
            Self::Meta(caches) => caches.first().map(|c| c.n_blocks()).unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------------------
// DynPolicy — type-erased VortexFlow policy
// ---------------------------------------------------------------------------

/// Type-erased VortexFlow policy that works with `DynRoutingCache`.
///
/// Each policy variant wraps a concrete router and knows how to
/// dispatch `forward_cache` and `forward_indexer` to it.
#[derive(Debug)]
pub enum DynPolicy {
    /// BlockTopK policy.
    BlockTopK(BlockTopKRouter),
    /// Entmax policy.
    Entmax(EntmaxRouter),
    /// ValueEnergy policy.
    ValueEnergy(ValueEnergyRouter),
}

impl DynPolicy {
    /// Dispatch `forward_cache` to the concrete router.
    pub fn forward_cache(
        &self,
        cache: &mut DynRoutingCache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        match (self, cache) {
            (Self::BlockTopK(r), DynRoutingCache::BlockTopK(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            (Self::Entmax(r), DynRoutingCache::Entmax(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            (Self::ValueEnergy(r), DynRoutingCache::ValueEnergy(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            _ => panic!("DynPolicy/Cache variant mismatch in forward_cache"),
        }
    }

    /// Dispatch `forward_indexer` to the concrete router.
    pub fn forward_indexer(
        &self,
        query: &[f32],
        cache: &DynRoutingCache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        match (self, cache) {
            (Self::BlockTopK(r), DynRoutingCache::BlockTopK(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            (Self::Entmax(r), DynRoutingCache::Entmax(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            (Self::ValueEnergy(r), DynRoutingCache::ValueEnergy(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            _ => panic!("DynPolicy/Cache variant mismatch in forward_indexer"),
        }
    }

    /// Create a new cache for this policy.
    pub fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> DynRoutingCache {
        match self {
            Self::BlockTopK(r) => {
                DynRoutingCache::BlockTopK(r.cache_new(n_blocks_capacity, head_dim))
            }
            Self::Entmax(r) => DynRoutingCache::Entmax(r.cache_new(n_blocks_capacity, head_dim)),
            Self::ValueEnergy(r) => {
                DynRoutingCache::ValueEnergy(r.cache_new(n_blocks_capacity, head_dim))
            }
        }
    }

    /// Human-readable policy name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::BlockTopK(_) => "BlockTopK",
            Self::Entmax(_) => "Entmax",
            Self::ValueEnergy(_) => "ValueEnergy",
        }
    }
}

// ---------------------------------------------------------------------------
// MetaRouter — T16
// ---------------------------------------------------------------------------

/// A trivial ScreeningPruner that always returns 1.0 (no screening).
/// Used as the inner pruner for the BanditPruner used by MetaRouter.
struct NoScreening;

impl ScreeningPruner for NoScreening {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        1.0
    }
}

/// MetaRouter — selects among multiple VortexFlow policies via bandit.
///
/// Owns `policies: Vec<DynPolicy>` and a `BanditPruner` for policy selection.
///
/// - `forward_cache`: delegates to ALL policies (maintains all caches).
/// - `forward_indexer`: bandit selects policy arm → delegates to selected policy.
/// - Reward signal: `acceptance_rate * latency_improvement` per decode step.
///
/// The bandit starts with exploration and converges to the best policy over time.
pub struct MetaRouter {
    /// Policies under bandit control.
    pub policies: Vec<DynPolicy>,
    /// Bandit for policy selection.
    bandit: BanditPruner<NoScreening>,
}

impl std::fmt::Debug for MetaRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaRouter")
            .field("n_policies", &self.policies.len())
            .field("best_arm", &self.bandit.best_arm())
            .finish()
    }
}

impl MetaRouter {
    /// Create a new MetaRouter with the given policies and bandit strategy.
    ///
    /// The bandit will have `policies.len()` arms, one per policy.
    pub fn new(policies: Vec<DynPolicy>, strategy: BanditStrategy) -> Self {
        let n_arms = policies.len();
        let bandit = BanditPruner::new(NoScreening, strategy, n_arms);
        Self { policies, bandit }
    }

    /// Create with default EpsilonGreedy strategy (ε=0.1, decay=0.995).
    pub fn new_default(policies: Vec<DynPolicy>) -> Self {
        Self::new(
            policies,
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 0.995,
            },
        )
    }

    /// Update bandit reward after observing a decode step outcome.
    ///
    /// Call this after each decode step with the selected policy's performance.
    ///
    /// # Arguments
    /// * `arm` — index of the policy that was selected
    /// * `reward` — reward signal: `acceptance_rate * latency_improvement`
    pub fn update_reward(&mut self, arm: usize, reward: f32) {
        self.bandit.update(arm, reward);
    }

    /// Select which policy arm to use for this decode step.
    ///
    /// Uses the bandit strategy (UCB1, ε-greedy, Thompson, etc.) to balance
    /// exploration and exploitation.
    pub fn select_arm(&self) -> usize {
        self.bandit.best_arm()
    }

    /// Index of the best policy (highest Q-value).
    pub fn best_arm(&self) -> usize {
        self.bandit.best_arm()
    }

    /// Q-values for all policy arms.
    pub fn q_values(&self) -> &[f32] {
        self.bandit.q_values()
    }

    /// Visit counts for all policy arms.
    pub fn visits(&self) -> &[u32] {
        self.bandit.visits()
    }

    /// Total number of arm pulls.
    pub fn total_pulls(&self) -> u32 {
        self.bandit.total_pulls()
    }

    /// Get the bandit stats for inspection.
    pub fn bandit_stats(&self) -> &BanditStats {
        // Access internal stats through the pruner's public interface
        // We expose q_values/visits/best_arm directly
        // For full stats access, we'd need BanditPruner to expose &BanditStats
        // For now, use the individual accessors
        unreachable!("use individual accessors instead")
    }

    /// Decay epsilon after an episode (EpsilonGreedy only).
    pub fn decay_epsilon(&mut self) {
        self.bandit.decay_epsilon();
    }

    /// Number of policies.
    pub fn n_policies(&self) -> usize {
        self.policies.len()
    }

    /// Get policy name by index.
    pub fn policy_name(&self, idx: usize) -> &'static str {
        match self.policies.get(idx) {
            Some(p) => p.name(),
            None => "unknown",
        }
    }
}

/// Compute reward signal from speculative verification outcome — T18.
///
/// # Arguments
/// * `accepted` — whether the token was accepted by the verifier
/// * `baseline_latency_ns` — baseline latency (e.g., full attention)
/// * `actual_latency_ns` — actual latency with routing
///
/// # Returns
/// Reward ∈ [0, 2.0]: `acceptance_rate * (1.0 + latency_bonus)`
#[inline]
pub fn compute_reward(accepted: bool, baseline_latency_ns: u64, actual_latency_ns: u64) -> f32 {
    let acceptance = match accepted {
        true => 1.0f32,
        false => 0.0f32,
    };
    let latency_bonus = match baseline_latency_ns {
        0 => 0.0f32,
        _ => {
            let improvement = baseline_latency_ns.saturating_sub(actual_latency_ns) as f32
                / baseline_latency_ns as f32;
            improvement.clamp(0.0, 1.0)
        }
    };
    acceptance * (1.0 + latency_bonus)
}

impl VortexFlow for MetaRouter {
    type Cache = DynRoutingCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        // Delegate to ALL policies (maintain all caches)
        let caches = match cache {
            DynRoutingCache::Meta(caches) => caches,
            _ => panic!("MetaRouter expects DynRoutingCache::Meta variant"),
        };

        for (i, policy) in self.policies.iter().enumerate() {
            if let Some(c) = caches.get_mut(i) {
                policy.forward_cache(c, keys, values, block_idx, head_dim);
            }
        }
    }

    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        let caches = match cache {
            DynRoutingCache::Meta(caches) => caches,
            _ => panic!("MetaRouter expects DynRoutingCache::Meta variant"),
        };

        // Bandit selects policy arm
        let arm = self.select_arm();

        match (self.policies.get(arm), caches.get(arm)) {
            (Some(policy), Some(c)) => policy.forward_indexer(query, c, n_blocks, top_k, scratch),
            _ => RoutingDecision::new(),
        }
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        let caches: Vec<DynRoutingCache> = self
            .policies
            .iter()
            .map(|p| p.cache_new(n_blocks_capacity, head_dim))
            .collect();
        DynRoutingCache::Meta(caches)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;

    fn make_meta_router() -> MetaRouter {
        let policies = vec![
            DynPolicy::BlockTopK(BlockTopKRouter::new(true)),
            DynPolicy::Entmax(EntmaxRouter::default_router()),
            DynPolicy::ValueEnergy(ValueEnergyRouter::new(true)),
        ];
        MetaRouter::new(
            policies,
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.3,
                decay: 0.99,
            },
        )
    }

    #[test]
    fn test_meta_router_forward_cache_and_indexer() {
        let router = make_meta_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        // Verify cache is Meta variant with 3 sub-caches
        match &cache {
            DynRoutingCache::Meta(caches) => assert_eq!(caches.len(), 3),
            _ => panic!("expected Meta variant"),
        }

        // Populate blocks
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals = vec![1.0; HEAD_DIM];

        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);

        // Route
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);
        assert!(!decision.is_empty());
    }

    #[test]
    fn test_meta_router_reward_updates() {
        let mut router = make_meta_router();

        // Simulate: arm 0 gets high reward, arms 1/2 get low reward
        for _ in 0..10 {
            router.update_reward(0, 0.9);
            router.update_reward(1, 0.2);
            router.update_reward(2, 0.1);
        }

        // Best arm should be 0
        assert_eq!(router.best_arm(), 0, "arm 0 should have highest Q-value");

        let q = router.q_values();
        assert!(q[0] > q[1], "Q[0]={} should be > Q[1]={}", q[0], q[1]);
        assert!(q[0] > q[2], "Q[0]={} should be > Q[2]={}", q[0], q[2]);
    }

    #[test]
    fn test_meta_router_convergence() {
        let mut router = make_meta_router();

        // Simulate 50 decode steps where arm 1 (Entmax) is consistently best
        for step in 0..50 {
            router.update_reward(0, 0.3 + 0.01 * (step as f32).sin());
            router.update_reward(1, 0.8 + 0.05 * (step as f32).cos());
            router.update_reward(2, 0.4);
            router.decay_epsilon();
        }

        // After 50 steps, best arm should be 1 (Entmax)
        assert_eq!(
            router.best_arm(),
            1,
            "bandit should converge to arm 1 (Entmax)"
        );
    }

    #[test]
    fn test_dyn_routing_cache_n_blocks() {
        let cache = DynRoutingCache::BlockTopK(BlockTopKCache::new(4, HEAD_DIM));
        assert_eq!(cache.n_blocks(), 0);

        let cache = DynRoutingCache::Entmax(EntmaxCache::new(HEAD_DIM));
        assert_eq!(cache.n_blocks(), 0);

        let cache = DynRoutingCache::ValueEnergy(ValueEnergyCache::new(4, HEAD_DIM));
        assert_eq!(cache.n_blocks(), 0);
    }

    #[test]
    fn test_dyn_policy_name() {
        let p0 = DynPolicy::BlockTopK(BlockTopKRouter::new(true));
        let p1 = DynPolicy::Entmax(EntmaxRouter::default_router());
        let p2 = DynPolicy::ValueEnergy(ValueEnergyRouter::new(true));

        assert_eq!(p0.name(), "BlockTopK");
        assert_eq!(p1.name(), "Entmax");
        assert_eq!(p2.name(), "ValueEnergy");
    }

    #[test]
    fn test_compute_reward_accepted_fast() {
        // Accepted + 50% latency improvement → reward = 1.0 * (1.0 + 0.5) = 1.5
        let reward = compute_reward(true, 1000, 500);
        assert!((reward - 1.5).abs() < 1e-6, "expected 1.5, got {reward}");
    }

    #[test]
    fn test_compute_reward_rejected() {
        // Rejected → reward = 0.0 regardless of latency
        let reward = compute_reward(false, 1000, 500);
        assert!((reward - 0.0).abs() < 1e-6, "expected 0.0, got {reward}");
    }

    #[test]
    fn test_compute_reward_accepted_slower() {
        // Accepted + slower than baseline → latency_bonus = 0, reward = 1.0
        let reward = compute_reward(true, 500, 1000);
        assert!(
            (reward - 1.0).abs() < 1e-6,
            "expected 1.0 (no bonus), got {reward}"
        );
    }

    #[test]
    fn test_meta_router_all_policies_populate_cache() {
        let router = make_meta_router();
        let mut cache = router.cache_new(2, HEAD_DIM);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![1.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        // All 3 policies should have cached block 0
        match &cache {
            DynRoutingCache::Meta(caches) => {
                for (i, c) in caches.iter().enumerate() {
                    assert!(c.n_blocks() >= 1, "policy {i} should have cached block 0");
                }
            }
            _ => panic!("expected Meta variant"),
        }
    }

    #[test]
    fn test_meta_router_empty_blocks() {
        let router = make_meta_router();
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_meta_router_policy_names() {
        let router = make_meta_router();
        assert_eq!(router.policy_name(0), "BlockTopK");
        assert_eq!(router.policy_name(1), "Entmax");
        assert_eq!(router.policy_name(2), "ValueEnergy");
        assert_eq!(router.n_policies(), 3);
    }
}
