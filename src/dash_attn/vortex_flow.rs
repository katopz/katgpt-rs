//! VortexFlow — composable sparse routing trait for KV block selection.
//!
//! Each router implements two phases:
//! 1. `forward_cache` — query-independent cache update when KV blocks are appended
//! 2. `forward_indexer` — query-dependent block selection during decode
//!
//! Feature gate: `vortex_flow` (Plan 196, Phase 1, default-OFF).

use std::fmt::Debug;

use super::block_topk::{BlockTopKCache, BlockTopKRouter};
use super::channel_aware::{ChannelAwareCache, ChannelAwareRouter};
use super::entmax_router::{EntmaxCache, EntmaxRouter};
use super::meta_router::{DynPolicy, DynRoutingCache, MetaRouter};
use super::value_energy::{ValueEnergyCache, ValueEnergyRouter};

// ---------------------------------------------------------------------------
// VortexFlow trait
// ---------------------------------------------------------------------------

/// Composable sparse routing trait for KV block selection.
///
/// Implementors provide:
/// - A cache type for query-independent block summaries
/// - Cache update logic when new KV blocks arrive
/// - Query-dependent top-k block selection
pub trait VortexFlow: Send + Sync {
    /// Router-specific cache type.
    type Cache: Send + Debug;

    /// Update cache when a new KV block is appended (query-independent).
    /// Called during prefill or when KV cache grows.
    ///
    /// # Arguments
    /// * `cache` — mutable router cache to update
    /// * `keys` — flat `[block_size * head_dim]` key vectors for this block
    /// * `values` — flat `[block_size * head_dim]` value vectors for this block
    /// * `block_idx` — index of the block being cached
    /// * `head_dim` — dimension per head
    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    );

    /// Select top-k blocks for the given query (query-dependent).
    /// Called during each decode step.
    ///
    /// # Arguments
    /// * `query` — query vector `[head_dim]`
    /// * `cache` — immutable router cache
    /// * `n_blocks` — total number of blocks currently cached
    /// * `top_k` — maximum number of blocks to select
    /// * `scratch` — reusable scratch buffer for intermediate computations
    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision;

    /// Create a new cache instance pre-allocated for `n_blocks_capacity` blocks.
    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache;
}

// ---------------------------------------------------------------------------
// VortexFlowConfig
// ---------------------------------------------------------------------------

/// Router selection for VortexFlow decode path.
///
/// `DashAttn` (default) preserves existing behavior.
/// Other variants select a VortexFlow router implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum VortexFlowConfig {
    /// Use existing DashAttention routing (default).
    #[default]
    DashAttn,
    /// Use BlockTopK router (centroid + dot-product).
    BlockTopK,
    /// Use EntmaxRouter (wraps existing entmax scoring).
    Entmax,
    /// Use ValueEnergyRouter (centroid · ‖v‖ gating).
    ValueEnergy,
    /// Use ChannelAwareRouter (SIMD-optimized channel-aware routing).
    ChannelAware,
    /// Use MetaRouter (bandit-based policy selection).
    Meta,
}

// ---------------------------------------------------------------------------
// VortexFlowExt — extension for DashAttnConfig (katgpt-rs only)
// ---------------------------------------------------------------------------

/// Extension to `DashAttnConfig` for VortexFlow router selection.
///
/// Since `DashAttnConfig` lives in `katgpt-core` (immutable from katgpt-rs),
/// this wrapper carries the VortexFlow-specific configuration alongside
/// the standard DashAttention config.
#[derive(Debug, Clone, Default)]
pub struct VortexFlowExt {
    /// Which router to use during decode.
    pub config: VortexFlowConfig,
}

impl VortexFlowExt {
    /// Create extension with specific router config.
    pub fn new(config: VortexFlowConfig) -> Self {
        Self { config }
    }

    /// Whether VortexFlow routing should replace DashAttention routing.
    #[inline]
    pub fn is_vortex(&self) -> bool {
        !matches!(self.config, VortexFlowConfig::DashAttn)
    }
}

// ---------------------------------------------------------------------------
// VortexRouter — enum-based dispatch over all router types
// ---------------------------------------------------------------------------

/// Enum wrapper providing a single type for any VortexFlow router.
///
/// Avoids `Box<dyn VortexFlow<Cache = ?>>` — the Cache associated type
/// differs per router, so dynamic dispatch requires either enum dispatch
/// or a separate `DynRoutingCache` (Phase 3). This enum is the Phase 1 solution.
#[derive(Debug)]
pub enum VortexRouter {
    /// BlockTopK router (centroid + dot-product top-k).
    BlockTopK(BlockTopKRouter),
    /// Entmax router (wraps existing DashAttention entmax).
    Entmax(EntmaxRouter),
    /// ValueEnergy router (centroid · ‖v‖ gating).
    ValueEnergy(ValueEnergyRouter),
    /// Channel-aware router (SIMD-optimized routing over critical channels).
    ChannelAware(ChannelAwareRouter),
    /// Meta-router (bandit-based policy selection over multiple routers).
    Meta(MetaRouter),
}

/// Cache storage for [`VortexRouter`] — mirrors the enum variants.
#[derive(Debug)]
pub enum VortexRouterCache {
    /// BlockTopK cache.
    BlockTopK(BlockTopKCache),
    /// Entmax cache.
    Entmax(EntmaxCache),
    /// ValueEnergy cache.
    ValueEnergy(ValueEnergyCache),
    /// Channel-aware cache.
    ChannelAware(ChannelAwareCache),
    /// Meta-router cache (dynamic routing cache).
    Meta(DynRoutingCache),
}

impl VortexRouterCache {
    /// Number of blocks currently cached (variant-dependent).
    pub fn n_blocks(&self) -> usize {
        match self {
            Self::BlockTopK(c) => c.n_blocks,
            Self::Entmax(c) => c.summaries.len(),
            Self::ValueEnergy(c) => c.n_blocks,
            Self::ChannelAware(c) => c.n_blocks,
            Self::Meta(c) => c.n_blocks(),
        }
    }
}

impl VortexRouter {
    /// Build a router from config.
    pub fn from_config(config: VortexFlowConfig) -> Self {
        match config {
            VortexFlowConfig::BlockTopK => Self::BlockTopK(BlockTopKRouter::new(true)),
            VortexFlowConfig::Entmax => Self::Entmax(EntmaxRouter::default_router()),
            VortexFlowConfig::ValueEnergy => Self::ValueEnergy(ValueEnergyRouter::new(true)),
            VortexFlowConfig::ChannelAware => Self::ChannelAware(ChannelAwareRouter::new(true)),
            VortexFlowConfig::Meta => Self::Meta(MetaRouter::new_default(vec![
                DynPolicy::BlockTopK(BlockTopKRouter::new(true)),
                DynPolicy::Entmax(EntmaxRouter::default_router()),
                DynPolicy::ValueEnergy(ValueEnergyRouter::new(true)),
            ])),
            VortexFlowConfig::DashAttn => {
                unreachable!("DashAttn does not produce a VortexRouter; check is_vortex() first")
            }
        }
    }
}

impl VortexFlow for VortexRouter {
    type Cache = VortexRouterCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        match (self, cache) {
            (Self::BlockTopK(r), VortexRouterCache::BlockTopK(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            (Self::Entmax(r), VortexRouterCache::Entmax(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            (Self::ValueEnergy(r), VortexRouterCache::ValueEnergy(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            (Self::ChannelAware(r), VortexRouterCache::ChannelAware(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            (Self::Meta(r), VortexRouterCache::Meta(c)) => {
                r.forward_cache(c, keys, values, block_idx, head_dim)
            }
            _ => panic!("VortexRouter/Cache variant mismatch"),
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
        match (self, cache) {
            (Self::BlockTopK(r), VortexRouterCache::BlockTopK(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            (Self::Entmax(r), VortexRouterCache::Entmax(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            (Self::ValueEnergy(r), VortexRouterCache::ValueEnergy(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            (Self::ChannelAware(r), VortexRouterCache::ChannelAware(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            (Self::Meta(r), VortexRouterCache::Meta(c)) => {
                r.forward_indexer(query, c, n_blocks, top_k, scratch)
            }
            _ => panic!("VortexRouter/Cache variant mismatch"),
        }
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        match self {
            Self::BlockTopK(r) => {
                VortexRouterCache::BlockTopK(r.cache_new(n_blocks_capacity, head_dim))
            }
            Self::Entmax(r) => VortexRouterCache::Entmax(r.cache_new(n_blocks_capacity, head_dim)),
            Self::ValueEnergy(r) => {
                VortexRouterCache::ValueEnergy(r.cache_new(n_blocks_capacity, head_dim))
            }
            Self::ChannelAware(r) => {
                VortexRouterCache::ChannelAware(r.cache_new(n_blocks_capacity, head_dim))
            }
            Self::Meta(r) => VortexRouterCache::Meta(r.cache_new(n_blocks_capacity, head_dim)),
        }
    }
}

/// Build a `(VortexRouter, VortexRouterCache)` pair from config.
///
/// Convenience function for callers that need both the router and its cache.
pub fn build_vortex_router(
    config: VortexFlowConfig,
    n_blocks_capacity: usize,
    head_dim: usize,
) -> Option<(VortexRouter, VortexRouterCache)> {
    match config {
        VortexFlowConfig::DashAttn => None,
        _ => {
            let router = VortexRouter::from_config(config);
            let cache = router.cache_new(n_blocks_capacity, head_dim);
            Some((router, cache))
        }
    }
}

// ---------------------------------------------------------------------------
// RoutingDecision
// ---------------------------------------------------------------------------

/// Result of routing: which blocks to attend to and their weights.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected block indices (sorted by relevance, descending).
    pub blocks: Vec<usize>,
    /// Routing weights for selected blocks.
    pub weights: Vec<f32>,
}

impl RoutingDecision {
    /// Create an empty routing decision.
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity for `top_k` entries.
    pub fn with_capacity(top_k: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(top_k),
            weights: Vec::with_capacity(top_k),
        }
    }

    /// Clear for reuse across decode steps without deallocating.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.weights.clear();
    }

    /// Number of selected blocks.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether no blocks were selected.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl Default for RoutingDecision {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VortexScratch
// ---------------------------------------------------------------------------

/// Reusable scratch buffer for routing computations.
#[derive(Debug)]
pub struct VortexScratch {
    /// Block scores buffer `[max_blocks]`.
    pub scores: Vec<f32>,
    /// Top-k index buffer.
    pub indices: Vec<usize>,
}

impl VortexScratch {
    /// Create scratch buffers sized for `max_blocks` blocks.
    pub fn new(max_blocks: usize) -> Self {
        Self {
            scores: vec![0.0; max_blocks],
            indices: Vec::with_capacity(max_blocks),
        }
    }

    /// Ensure buffers can hold at least `n` blocks, growing if needed.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.scores.len() < n {
            self.scores.resize(n, 0.0);
        }
        if self.indices.capacity() < n {
            // Reserve enough for n total elements
            let additional = n.saturating_sub(self.indices.len());
            self.indices.reserve(additional);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_decision_new() {
        let rd = RoutingDecision::new();
        assert!(rd.is_empty());
        assert_eq!(rd.len(), 0);
    }

    #[test]
    fn test_routing_decision_with_capacity() {
        let rd = RoutingDecision::with_capacity(8);
        assert!(rd.is_empty());
        assert!(rd.blocks.capacity() >= 8);
        assert!(rd.weights.capacity() >= 8);
    }

    #[test]
    fn test_routing_decision_clear_reuse() {
        let mut rd = RoutingDecision::with_capacity(4);
        rd.blocks.push(0);
        rd.blocks.push(1);
        rd.weights.push(0.7);
        rd.weights.push(0.3);
        assert_eq!(rd.len(), 2);

        let block_cap = rd.blocks.capacity();
        let weight_cap = rd.weights.capacity();

        rd.clear();
        assert!(rd.is_empty());
        // Capacity preserved after clear
        assert_eq!(rd.blocks.capacity(), block_cap);
        assert_eq!(rd.weights.capacity(), weight_cap);
    }

    #[test]
    fn test_routing_decision_default() {
        let rd = RoutingDecision::default();
        assert!(rd.is_empty());
    }

    #[test]
    fn test_vortex_scratch_new() {
        let scratch = VortexScratch::new(16);
        assert_eq!(scratch.scores.len(), 16);
        assert!(scratch.scores.iter().all(|&s| s == 0.0));
        assert!(scratch.indices.is_empty());
        assert!(scratch.indices.capacity() >= 16);
    }

    #[test]
    fn test_vortex_scratch_ensure_capacity_grow() {
        let mut scratch = VortexScratch::new(4);
        scratch.ensure_capacity(16);
        assert!(scratch.scores.len() >= 16);
        // After ensure_capacity, pushing n elements should not reallocate
        for i in 0..16 {
            scratch.indices.push(i);
        }
        assert_eq!(scratch.indices.len(), 16);
    }

    #[test]
    fn test_vortex_scratch_ensure_capacity_noop_when_sufficient() {
        let mut scratch = VortexScratch::new(32);
        let scores_ptr = scratch.scores.as_ptr();
        scratch.ensure_capacity(16);
        // Should not reallocate
        assert_eq!(scratch.scores.as_ptr(), scores_ptr);
    }

    // -----------------------------------------------------------------------
    // VortexFlowConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vortex_flow_config_default_is_dash_attn() {
        assert_eq!(VortexFlowConfig::default(), VortexFlowConfig::DashAttn);
    }

    // -----------------------------------------------------------------------
    // VortexFlowExt tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vortex_flow_ext_default_not_vortex() {
        let ext = VortexFlowExt::default();
        assert!(!ext.is_vortex());
    }

    #[test]
    fn test_vortex_flow_ext_block_topk_is_vortex() {
        let ext = VortexFlowExt::new(VortexFlowConfig::BlockTopK);
        assert!(ext.is_vortex());
    }

    #[test]
    fn test_vortex_flow_ext_entmax_is_vortex() {
        let ext = VortexFlowExt::new(VortexFlowConfig::Entmax);
        assert!(ext.is_vortex());
    }

    #[test]
    fn test_vortex_flow_ext_value_energy_is_vortex() {
        let ext = VortexFlowExt::new(VortexFlowConfig::ValueEnergy);
        assert!(ext.is_vortex());
    }

    #[test]
    fn test_vortex_flow_ext_channel_aware_is_vortex() {
        let ext = VortexFlowExt::new(VortexFlowConfig::ChannelAware);
        assert!(ext.is_vortex());
    }

    #[test]
    fn test_vortex_flow_ext_meta_is_vortex() {
        let ext = VortexFlowExt::new(VortexFlowConfig::Meta);
        assert!(ext.is_vortex());
    }

    // -----------------------------------------------------------------------
    // VortexRouter enum dispatch tests
    // -----------------------------------------------------------------------

    const HD: usize = 4;

    #[test]
    fn test_vortex_router_block_topk_dispatch() {
        let router = VortexRouter::from_config(VortexFlowConfig::BlockTopK);
        let mut cache = router.cache_new(2, HD);
        let mut scratch = VortexScratch::new(2);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HD];
        router.forward_cache(&mut cache, &keys, &vals, 0, HD);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks, vec![0]);
    }

    #[test]
    fn test_vortex_router_channel_aware_dispatch() {
        let router = VortexRouter::from_config(VortexFlowConfig::ChannelAware);
        let mut cache = router.cache_new(2, HD);
        let mut scratch = VortexScratch::new(2);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HD];
        router.forward_cache(&mut cache, &keys, &vals, 0, HD);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks, vec![0]);
    }

    #[test]
    fn test_vortex_router_meta_dispatch() {
        let router = VortexRouter::from_config(VortexFlowConfig::Meta);
        let mut cache = router.cache_new(2, HD);
        let mut scratch = VortexScratch::new(2);

        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![1.0, 1.0, 1.0, 1.0];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HD);

        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![1.0, 1.0, 1.0, 1.0];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HD);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);
        assert!(!decision.is_empty());
    }

    #[test]
    fn test_vortex_router_entmax_dispatch() {
        let router = VortexRouter::from_config(VortexFlowConfig::Entmax);
        let mut cache = router.cache_new(2, HD);
        let mut scratch = VortexScratch::new(2);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HD];
        router.forward_cache(&mut cache, &keys, &vals, 0, HD);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks, vec![0]);
    }

    #[test]
    fn test_vortex_router_value_energy_dispatch() {
        let router = VortexRouter::from_config(VortexFlowConfig::ValueEnergy);
        let mut cache = router.cache_new(2, HD);
        let mut scratch = VortexScratch::new(2);

        // Block 0: aligned centroid + non-zero energy
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![1.0, 1.0, 1.0, 1.0];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HD);

        // Block 1: orthogonal centroid
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![1.0, 1.0, 1.0, 1.0];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HD);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);
        assert_eq!(decision.blocks[0], 0);
    }

    #[test]
    fn test_build_vortex_router_returns_none_for_dash_attn() {
        let result = build_vortex_router(VortexFlowConfig::DashAttn, 4, HD);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_vortex_router_returns_some_for_block_topk() {
        let (router, _cache) = build_vortex_router(VortexFlowConfig::BlockTopK, 4, HD)
            .expect("BlockTopK should build");
        match router {
            VortexRouter::BlockTopK(_) => {}
            _ => panic!("expected BlockTopK variant"),
        }
    }

    #[test]
    fn test_build_vortex_router_returns_some_for_entmax() {
        let (router, _cache) =
            build_vortex_router(VortexFlowConfig::Entmax, 4, HD).expect("Entmax should build");
        match router {
            VortexRouter::Entmax(_) => {}
            _ => panic!("expected Entmax variant"),
        }
    }

    #[test]
    fn test_build_vortex_router_returns_some_for_value_energy() {
        let (router, _cache) = build_vortex_router(VortexFlowConfig::ValueEnergy, 4, HD)
            .expect("ValueEnergy should build");
        match router {
            VortexRouter::ValueEnergy(_) => {}
            _ => panic!("expected ValueEnergy variant"),
        }
    }

    #[test]
    fn test_build_vortex_router_returns_some_for_channel_aware() {
        let (router, _cache) = build_vortex_router(VortexFlowConfig::ChannelAware, 4, HD)
            .expect("ChannelAware should build");
        match router {
            VortexRouter::ChannelAware(_) => {}
            _ => panic!("expected ChannelAware variant"),
        }
    }

    #[test]
    fn test_build_vortex_router_returns_some_for_meta() {
        let (router, _cache) =
            build_vortex_router(VortexFlowConfig::Meta, 4, HD).expect("Meta should build");
        match router {
            VortexRouter::Meta(_) => {}
            _ => panic!("expected Meta variant"),
        }
    }
}
