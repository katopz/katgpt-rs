//! Hop speculator trait and implementations for SpecHop pipeline.
//!
//! Speculators predict what a tool/hop will return before the actual response
//! arrives. The pipeline speculates ahead on k threads and verifies when
//! the target returns.
//!
//! - `CacheSpeculator`: simple HashMap lookup (modelless path)
//! - `BanditSpeculator`: uses ScreeningPruner relevance to gate predictions
//!   (modelless-to-model-based bridge)

use std::collections::HashMap;

use crate::spechop::types::SpecError;

// ── RecFM Cross-Hop Consistency (Plan 168 T3) ───────────────────────

/// Configuration for RecFM cross-hop velocity consistency (Research 150).
///
/// Tracks observation velocity across consecutive SpecHop hops.
/// Converging observations (decreasing velocity) get higher confidence;
/// diverging observations get penalized.
///
/// When `enable` is `false`, all RecFM checks are no-ops (zero cost).
#[cfg(feature = "recfm")]
#[derive(Debug, Clone)]
pub struct CrossHopConfig {
    /// Enable cross-hop velocity consistency checks.
    pub enable: bool,
    /// Velocity threshold — observations diverging faster than this are penalized.
    /// Default: 0.5 (moderate divergence tolerance).
    pub velocity_threshold: f32,
    /// Minimum hops before consistency checks are applied.
    /// Default: 2 (need at least 2 observations to compute velocity).
    pub min_hops_for_consistency: usize,
}

#[cfg(feature = "recfm")]
impl Default for CrossHopConfig {
    fn default() -> Self {
        Self {
            enable: false,
            velocity_threshold: 0.5,
            min_hops_for_consistency: 2,
        }
    }
}

/// Compute observation velocity between two consecutive observations.
///
/// Returns the normalized distance: `1.0 - overlap_ratio`, where
/// overlap_ratio is the fraction of shared tokens (O(1) hot-path via
/// simple character-level comparison).
///
/// - velocity = 0.0 → identical observations (converged)
/// - velocity = 1.0 → completely different observations (diverged)
#[cfg(feature = "recfm")]
#[inline]
pub fn observation_velocity(obs_k: &str, obs_k1: &str) -> f32 {
    if obs_k.is_empty() && obs_k1.is_empty() {
        return 0.0;
    }
    if obs_k.is_empty() || obs_k1.is_empty() {
        return 1.0;
    }
    // Simple character overlap ratio: count matching characters in shortest common prefix
    let min_len = obs_k.len().min(obs_k1.len());
    let max_len = obs_k.len().max(obs_k1.len());
    let matching = obs_k
        .chars()
        .zip(obs_k1.chars())
        .take_while(|(a, b)| a == b)
        .count();
    1.0 - (matching as f32 / max_len as f32)
}

/// Trait for predicting tool-call observations ahead of actual responses.
///
/// Paper Section 3: speculator S predicts observation o_spec for action a
/// while the target tool processes the request. If `p = P(S correct)` is
/// high enough, speculation reduces wall-clock latency.
pub trait HopSpeculator: Send + Sync {
    /// Speculate what observation the given action will produce.
    ///
    /// Returns `Ok(observation)` if a prediction is available,
    /// `Err(SpecError)` if no prediction can be made.
    fn speculate(&self, action: &str) -> Result<String, SpecError>;

    /// Record an observed (action, observation) pair for future speculation.
    ///
    /// Called after the target tool returns — feeds the speculator's
    /// internal cache/bandit for next time.
    fn observe(&mut self, action: &str, observation: &str);
}

// ── CacheSpeculator ───────────────────────────────────────────

/// Simple HashMap-based speculator. Looks up past observations by action.
///
/// Modelless path: no bandit, no relevance scoring. Just caches
/// (action → observation) pairs and returns them on hit.
///
/// Cache hit rate = effective speculator accuracy `p`. A 25% cache hit
/// rate gives p̂ ≥ 0.25, which is the GOAT Proof 4 threshold.
#[derive(Clone, Debug)]
pub struct CacheSpeculator {
    cache: HashMap<String, String>,
}

impl CacheSpeculator {
    /// Create an empty cache speculator.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Create a speculator pre-populated with known (action, observation) pairs.
    pub fn with_entries(
        entries: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            cache: entries
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    /// Number of cached action→observation pairs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for CacheSpeculator {
    fn default() -> Self {
        Self::new()
    }
}

impl HopSpeculator for CacheSpeculator {
    fn speculate(&self, action: &str) -> Result<String, SpecError> {
        self.cache
            .get(action)
            .cloned()
            .ok_or_else(|| SpecError::CacheMiss {
                action: action.to_string(),
            })
    }

    fn observe(&mut self, action: &str, observation: &str) {
        self.cache
            .insert(action.to_string(), observation.to_string());
    }
}

// ── BanditSpeculator ──────────────────────────────────────────

/// Bandit-gated speculator using `ScreeningPruner` relevance scores.
///
/// Modelless-to-model-based bridge: uses bandit Q-values (modelless signal)
/// to decide whether to predict an observation (model-based behavior).
///
/// Only speculates when:
/// 1. A cached observation exists for the action
/// 2. The bandit relevance score for that action exceeds `confidence_threshold`
///
/// This filters out low-confidence predictions, improving speculator
/// accuracy `p` at the cost of fewer speculation attempts.
#[cfg(feature = "bandit")]
pub struct BanditSpeculator<P: crate::speculative::types::ScreeningPruner> {
    /// Inner cache of action → (observation, depth, token_idx).
    cache: HashMap<String, CachedAction>,
    /// Bandit pruner for relevance scoring.
    pruner: P,
    /// Minimum relevance score to speculate. Default: 0.5.
    pub confidence_threshold: f32,
}

/// Cached action metadata for bandit relevance lookup.
#[cfg(feature = "bandit")]
#[derive(Clone, Debug)]
struct CachedAction {
    /// The predicted observation text.
    observation: String,
    /// Depth to use for relevance query.
    depth: usize,
    /// Token index to use for relevance query.
    token_idx: usize,
}

#[cfg(feature = "bandit")]
impl<P: crate::speculative::types::ScreeningPruner> BanditSpeculator<P> {
    /// Create a new bandit speculator with the given pruner and threshold.
    pub fn new(pruner: P, confidence_threshold: f32) -> Self {
        Self {
            cache: HashMap::new(),
            pruner,
            confidence_threshold: confidence_threshold.clamp(0.0, 1.0),
        }
    }

    /// Pre-populate with a known (action, observation, depth, token_idx) entry.
    pub fn with_entry(
        &mut self,
        action: impl Into<String>,
        observation: impl Into<String>,
        depth: usize,
        token_idx: usize,
    ) {
        self.cache.insert(
            action.into(),
            CachedAction {
                observation: observation.into(),
                depth,
                token_idx,
            },
        );
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(feature = "bandit")]
impl<P: crate::speculative::types::ScreeningPruner> HopSpeculator for BanditSpeculator<P> {
    fn speculate(&self, action: &str) -> Result<String, SpecError> {
        let cached = self.cache.get(action).ok_or_else(|| SpecError::CacheMiss {
            action: action.to_string(),
        })?;

        // Query bandit relevance for this action's (depth, token_idx) context
        let relevance = self.pruner.relevance(cached.depth, cached.token_idx, &[]);

        if relevance >= self.confidence_threshold {
            Ok(cached.observation.clone())
        } else {
            Err(SpecError::LowConfidence {
                action: action.to_string(),
                score: (relevance * 1000.0) as u32,
            })
        }
    }

    fn observe(&mut self, action: &str, observation: &str) {
        // Store with depth 0 as default — caller can update via with_entry
        self.cache.insert(
            action.to_string(),
            CachedAction {
                observation: observation.to_string(),
                depth: 0,
                token_idx: 0,
            },
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CacheSpeculator tests (T14) ───────────────────────────

    #[test]
    fn test_cache_speculator_hit() {
        let mut spec = CacheSpeculator::new();
        spec.observe("search_rust", "Rust is a systems programming language");

        let result = spec.speculate("search_rust");
        assert_eq!(result.unwrap(), "Rust is a systems programming language");
    }

    #[test]
    fn test_cache_speculator_miss() {
        let spec = CacheSpeculator::new();

        let result = spec.speculate("search_unknown");
        assert!(result.is_err());
        match result.unwrap_err() {
            SpecError::CacheMiss { action } => assert_eq!(action, "search_unknown"),
            other => panic!("expected CacheMiss, got {other:?}"),
        }
    }

    #[test]
    fn test_cache_speculator_overwrite() {
        let mut spec = CacheSpeculator::new();
        spec.observe("search", "old result");
        spec.observe("search", "new result");

        assert_eq!(spec.speculate("search").unwrap(), "new result");
    }

    #[test]
    fn test_cache_speculator_with_entries() {
        let spec = CacheSpeculator::with_entries([("a", "result_a"), ("b", "result_b")]);

        assert_eq!(spec.len(), 2);
        assert_eq!(spec.speculate("a").unwrap(), "result_a");
        assert_eq!(spec.speculate("b").unwrap(), "result_b");
        assert!(spec.speculate("c").is_err());
    }

    #[test]
    fn test_cache_speculator_clear() {
        let mut spec = CacheSpeculator::with_entries([("a", "result")]);
        assert!(!spec.is_empty());
        spec.clear();
        assert!(spec.is_empty());
        assert!(spec.speculate("a").is_err());
    }

    #[test]
    fn test_cache_speculator_default() {
        let spec = CacheSpeculator::default();
        assert!(spec.is_empty());
    }

    // ── BanditSpeculator tests (T14) ──────────────────────────

    #[cfg(feature = "bandit")]
    mod bandit_tests {
        use super::*;

        /// A ScreeningPruner that always returns a fixed relevance.
        #[derive(Clone)]
        struct FixedRelevance {
            relevance: f32,
        }

        impl crate::speculative::types::ScreeningPruner for FixedRelevance {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
                self.relevance
            }
        }

        #[test]
        fn test_bandit_speculator_high_relevance_speculates() {
            let pruner = FixedRelevance { relevance: 0.8 };
            let mut spec = BanditSpeculator::new(pruner, 0.5);
            spec.with_entry("search", "cached result", 0, 0);

            let result = spec.speculate("search");
            assert_eq!(result.unwrap(), "cached result");
        }

        #[test]
        fn test_bandit_speculator_low_relevance_rejects() {
            let pruner = FixedRelevance { relevance: 0.2 };
            let mut spec = BanditSpeculator::new(pruner, 0.5);
            spec.with_entry("search", "cached result", 0, 0);

            let result = spec.speculate("search");
            assert!(result.is_err());
            match result.unwrap_err() {
                SpecError::LowConfidence { action, .. } => {
                    assert_eq!(action, "search");
                }
                other => panic!("expected LowConfidence, got {other:?}"),
            }
        }

        #[test]
        fn test_bandit_speculator_cache_miss() {
            let pruner = FixedRelevance { relevance: 0.9 };
            let spec: BanditSpeculator<FixedRelevance> = BanditSpeculator::new(pruner, 0.5);

            let result = spec.speculate("unknown");
            assert!(result.is_err());
            match result.unwrap_err() {
                SpecError::CacheMiss { action } => assert_eq!(action, "unknown"),
                other => panic!("expected CacheMiss, got {other:?}"),
            }
        }

        #[test]
        fn test_bandit_speculator_observe_then_speculate() {
            let pruner = FixedRelevance { relevance: 0.7 };
            let mut spec = BanditSpeculator::new(pruner, 0.5);

            // observe stores with depth=0, token_idx=0
            spec.observe("action", "observed result");

            let result = spec.speculate("action");
            assert_eq!(result.unwrap(), "observed result");
        }

        #[test]
        fn test_bandit_speculator_threshold_clamped() {
            let pruner = FixedRelevance { relevance: 0.6 };
            let _spec = BanditSpeculator::new(pruner, 1.5); // clamped to 1.0
            // relevance 0.6 < threshold 1.0 → reject

            let mut spec2 = BanditSpeculator::new(FixedRelevance { relevance: 0.6 }, 1.5);
            spec2.observe("a", "result");
            assert!(spec2.speculate("a").is_err());
        }

        #[test]
        fn test_bandit_speculator_exact_threshold() {
            let pruner = FixedRelevance { relevance: 0.5 };
            let mut spec = BanditSpeculator::new(pruner, 0.5);
            spec.observe("a", "result");

            // relevance == threshold → should pass (>=)
            assert_eq!(spec.speculate("a").unwrap(), "result");
        }
    }
}
