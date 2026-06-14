//! BAKE Precision-Gated Bayesian Embedding Update (Plan 236).
//!
//! Per-dimension precision tracking for KgEmbedding.
//! High precision → anchor (resist change). Low precision → explore (absorb eagerly).
//! O(8) arithmetic per update, zero-alloc, SIMD-friendly.

/// Uninformative prior precision for new entities.
pub const UNINFORMATIVE_PRECISION: f32 = 0.1;

/// Default observation precision.
pub const DEFAULT_OBS_PRECISION: f32 = 1.0;

/// BAKE eq 2: Bayesian precision update.
/// λ_new = λ_old + λ_obs  (precision grows monotonically)
#[inline]
pub fn bake_update_precision(lambda_old: &[f32; 8], lambda_obs: f32) -> [f32; 8] {
    let obs = lambda_obs;
    [
        lambda_old[0] + obs,
        lambda_old[1] + obs,
        lambda_old[2] + obs,
        lambda_old[3] + obs,
        lambda_old[4] + obs,
        lambda_old[5] + obs,
        lambda_old[6] + obs,
        lambda_old[7] + obs,
    ]
}

/// BAKE eq 3: Precision-weighted mean update.
/// μ_new = (λ_old ⊙ μ_old + λ_obs ⊙ obs) / λ_new
/// SIMD-friendly: operates on [f32; 8] which auto-vectorizes.
#[inline]
pub fn bake_update_mean(
    mu_old: &[f32; 8],
    lambda_old: &[f32; 8],
    observation: &[f32; 8],
    lambda_obs: f32,
) -> [f32; 8] {
    let mut mu_new = [0.0f32; 8];
    for d in 0..8 {
        let lambda_new_d = lambda_old[d] + lambda_obs;
        mu_new[d] = (lambda_old[d] * mu_old[d] + lambda_obs * observation[d]) / lambda_new_d;
    }
    mu_new
}

/// Combined BAKE update: returns (new_mean, new_precision).
#[inline]
pub fn bake_update(
    mu_old: &[f32; 8],
    lambda_old: &[f32; 8],
    observation: &[f32; 8],
    lambda_obs: f32,
) -> ([f32; 8], [f32; 8]) {
    let mut mu_new = [0.0f32; 8];
    let mut lambda_new = [0.0f32; 8];
    for d in 0..8 {
        lambda_new[d] = lambda_old[d] + lambda_obs;
        mu_new[d] = (lambda_old[d] * mu_old[d] + lambda_obs * observation[d]) / lambda_new[d];
    }
    (mu_new, lambda_new)
}

/// BAKE eq 4: Precision-weighted regularization penalty.
/// β · √(λ ⊙ (μ_current - μ_old)²)
/// Returns penalty — high when current deviates from high-precision prior.
#[inline]
pub fn bake_regularize(
    mu_old: &[f32; 8],
    lambda: &[f32; 8],
    mu_current: &[f32; 8],
    beta: f32,
) -> f32 {
    // Algebraic identity: √(λ·diff²) = |diff| · √λ.
    // Hoisting √λ out of the (λ·diff²)·sqrt dependency chain lets the FPU pipeline
    // the sqrt in parallel with the diff subtract.
    let mut penalty = 0.0f32;
    for d in 0..8 {
        let sqrt_lambda = lambda[d].sqrt();
        let diff = mu_current[d] - mu_old[d];
        penalty += diff.abs() * sqrt_lambda;
    }
    penalty * beta
}

/// Compute effective confidence from precision vector.
/// confidence = sigmoid(mean(precision) - 1.0)
/// Higher average precision → higher confidence.
#[inline]
pub fn precision_to_confidence(lambda: &[f32; 8]) -> f32 {
    let mean_lambda = crate::simd::simd_sum_f32(lambda) * 0.125;
    1.0 / (1.0 + (-(mean_lambda - 1.0)).exp()) // sigmoid
}

/// Exploration priority for a dimension (0..7).
/// Lower precision → higher priority for exploration.
/// Returns value in [0, 1] where 1 = highest priority.
#[inline]
pub fn exploration_priority(lambda: &[f32; 8], dimension: usize) -> f32 {
    debug_assert!(dimension < 8, "dimension must be 0..7");
    exploration_priority_with_max(lambda, dimension, max_lambda(lambda))
}

/// Pre-compute max lambda for batch calls to `exploration_priority_with_max`.
#[inline]
pub fn max_lambda(lambda: &[f32; 8]) -> f32 {
    crate::simd::simd_max_f32(lambda)
}

/// Exploration priority with pre-computed max_lambda — avoids O(8) scan per call.
#[inline]
pub fn exploration_priority_with_max(lambda: &[f32; 8], dimension: usize, max_lam: f32) -> f32 {
    debug_assert!(dimension < 8, "dimension must be 0..7");
    if max_lam < 1e-6 {
        return 1.0;
    }
    1.0 - lambda[dimension] / max_lam
}

/// Informed prior precision from schema class density.
/// Dense classes (many entities) → higher precision (confident centroid).
/// λ_init = class_count / (1 + class_count) ∈ [0, 1).
#[inline]
pub fn informed_prior_precision(class_count: usize) -> [f32; 8] {
    let p = (class_count as f32) / (1.0 + class_count as f32);
    [p; 8]
}

// ---------------------------------------------------------------------------
// Persistent precision storage + session boundary update (Plan 236 Phase 3)
// ---------------------------------------------------------------------------

#[cfg(feature = "bake_precision")]
mod bake_store {
    use super::{DEFAULT_OBS_PRECISION, UNINFORMATIVE_PRECISION, bake_update};

    /// Per-entity precision state. Copy + Clone (two [f32; 8]).
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PrecisionEntry {
        pub mean: [f32; 8],
        pub precision: [f32; 8],
    }

    impl PrecisionEntry {
        /// Uninformative prior entry: zero mean, low precision.
        #[inline]
        pub fn uninformative() -> Self {
            Self {
                mean: [0.0; 8],
                precision: [UNINFORMATIVE_PRECISION; 8],
            }
        }
    }

    /// Lock-free entity-keyed precision map.
    ///
    /// Precision is tracked externally alongside the BFCF × LFU shard pipeline.
    /// Uses papaya `HashMap` for concurrent reads without blocking.
    pub struct BakePrecisionStore {
        entries: papaya::HashMap<u64, PrecisionEntry>,
    }

    impl Default for BakePrecisionStore {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BakePrecisionStore {
        /// Create an empty precision store.
        pub fn new() -> Self {
            Self {
                entries: papaya::HashMap::new(),
            }
        }

        /// Read precision for entity. Returns `None` if untracked.
        pub fn get(&self, entity_hash: u64) -> Option<PrecisionEntry> {
            self.entries.pin().get(&entity_hash).copied()
        }

        /// Apply BAKE update and store result.
        ///
        /// If entity is not tracked, creates an entry with uninformative prior first,
        /// then applies the update. Uses a single papaya pin to avoid double epoch
        /// overhead.
        pub fn update(&self, entity_hash: u64, observation: &[f32; 8], lambda_obs: f32) {
            let guard = self.entries.pin();
            let entry = guard
                .get(&entity_hash)
                .copied()
                .unwrap_or_else(PrecisionEntry::uninformative);

            let (new_mean, new_precision) =
                bake_update(&entry.mean, &entry.precision, observation, lambda_obs);

            guard.insert(
                entity_hash,
                PrecisionEntry {
                    mean: new_mean,
                    precision: new_precision,
                },
            );
        }

        /// Return current mean, or `[0.0; 8]` if untracked.
        pub fn snapshot_mean(&self, entity_hash: u64) -> [f32; 8] {
            self.entries
                .pin()
                .get(&entity_hash)
                .map(|e| e.mean)
                .unwrap_or([0.0; 8])
        }

        /// Evict entity (for LFU cache eviction). Returns the evicted entry.
        pub fn remove(&self, entity_hash: u64) -> Option<PrecisionEntry> {
            self.entries.pin().remove(&entity_hash).copied()
        }

        /// Number of tracked entities.
        pub fn len(&self) -> usize {
            self.entries.pin().len()
        }

        /// Whether the store is empty.
        pub fn is_empty(&self) -> bool {
            self.entries.pin().is_empty()
        }
    }

    // -----------------------------------------------------------------------
    // BakeSession — session boundary Bayesian update
    // -----------------------------------------------------------------------

    /// Session-level precision evolution for a single entity.
    ///
    /// Snapshots precision at `begin`, accumulates observations, then applies
    /// a batch Bayesian update at `end`. This avoids per-observation write
    /// contention on the shared store.
    #[derive(Debug)]
    pub struct BakeSession {
        entity_hash: u64,
        start_mean: [f32; 8],
        start_precision: [f32; 8],
        observation_count: u32,
        accumulated_obs_sum: [f32; 8],
    }

    impl BakeSession {
        /// Snapshot current state from the store as session start.
        ///
        /// For entities not in the store, uses uninformative prior.
        pub fn begin(entity_hash: u64, store: &BakePrecisionStore) -> Self {
            let entry = store
                .get(entity_hash)
                .unwrap_or_else(PrecisionEntry::uninformative);
            Self {
                entity_hash,
                start_mean: entry.mean,
                start_precision: entry.precision,
                observation_count: 0,
                accumulated_obs_sum: [0.0; 8],
            }
        }

        /// Accumulate an observation (running sum, zero-alloc).
        pub fn observe(&mut self, observation: &[f32; 8]) {
            // Direct indexing for fixed-size array — LLVM unrolls fully.
            for (d, obs_d) in observation.iter().enumerate() {
                self.accumulated_obs_sum[d] += obs_d;
            }
            self.observation_count += 1;
        }

        /// Apply batch Bayesian update and store result.
        ///
        /// Computes mean observation from accumulated sum / count, applies
        /// `bake_update` with `effective_lambda = DEFAULT_OBS_PRECISION * count`,
        /// stores result back, and returns the new entry.
        ///
        /// If no observations were made, returns the unchanged start entry
        /// without writing to the store.
        pub fn end(self, store: &BakePrecisionStore) -> PrecisionEntry {
            if self.observation_count == 0 {
                // No observations — write back start state if not already tracked.
                let result = PrecisionEntry {
                    mean: self.start_mean,
                    precision: self.start_precision,
                };
                store.entries.pin().insert(self.entity_hash, result);
                return result;
            }

            let count = self.observation_count as f32;
            let mut mean_obs = [0.0f32; 8];
            for (d, mean_d) in mean_obs.iter_mut().enumerate() {
                *mean_d = self.accumulated_obs_sum[d] / count;
            }

            let effective_lambda = DEFAULT_OBS_PRECISION * count;
            let (new_mean, new_precision) = bake_update(
                &self.start_mean,
                &self.start_precision,
                &mean_obs,
                effective_lambda,
            );

            let result = PrecisionEntry {
                mean: new_mean,
                precision: new_precision,
            };
            store.entries.pin().insert(self.entity_hash, result);
            result
        }

        /// Whether the session has accumulated observations.
        pub fn is_active(&self) -> bool {
            self.observation_count > 0
        }
    }
}

#[cfg(feature = "bake_precision")]
pub use bake_store::{BakePrecisionStore, BakeSession, PrecisionEntry};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_monotonicity() {
        let mut lambda = [0.1f32; 8];
        for _ in 0..10 {
            let old = lambda;
            lambda = bake_update_precision(&old, 1.0);
            for d in 0..8 {
                assert!(
                    lambda[d] >= old[d],
                    "precision should be monotonically non-decreasing"
                );
            }
        }
    }

    #[test]
    fn test_uninformative_prior_absorbs() {
        let mu_old = [0.0f32; 8];
        let lambda_old = [0.01f32; 8];
        let obs = [1.0f32; 8];
        let (mu_new, _) = bake_update(&mu_old, &lambda_old, &obs, 10.0);
        (0..8).for_each(|d| {
            assert!(
                (mu_new[d] - 1.0).abs() < 0.01,
                "should absorb observation when precision is low"
            );
        });
    }

    #[test]
    fn test_high_precision_resists() {
        let mu_old = [0.0f32; 8];
        let lambda_old = [100.0f32; 8];
        let obs = [1.0f32; 8];
        let (mu_new, _) = bake_update(&mu_old, &lambda_old, &obs, 1.0);
        (0..8).for_each(|d| {
            assert!(
                mu_new[d].abs() < 0.02,
                "should resist change when precision is high, got {}",
                mu_new[d]
            );
        });
    }

    #[test]
    fn test_regularize_zero_when_aligned() {
        let mu = [0.5f32; 8];
        let lambda = [1.0f32; 8];
        let penalty = bake_regularize(&mu, &lambda, &mu, 1.0);
        assert!(penalty.abs() < 1e-6, "penalty should be zero when aligned");
    }

    #[test]
    fn test_regularize_high_when_deviant() {
        let mu_old = [0.0f32; 8];
        let lambda = [10.0f32; 8];
        let mu_current = [1.0f32; 8];
        let penalty = bake_regularize(&mu_old, &lambda, &mu_current, 1.0);
        assert!(
            penalty > 3.0,
            "penalty should be high when deviating from high-precision prior, got {}",
            penalty
        );
    }

    #[test]
    fn test_confidence_increases_with_precision() {
        let low = precision_to_confidence(&[0.1f32; 8]);
        let high = precision_to_confidence(&[10.0f32; 8]);
        assert!(high > low, "higher precision should give higher confidence");
    }

    #[test]
    fn test_exploration_priority_inversely_related() {
        let lambda = [1.0, 5.0, 10.0, 0.5, 2.0, 8.0, 3.0, 0.1];
        let p7 = exploration_priority(&lambda, 7);
        let p2 = exploration_priority(&lambda, 2);
        assert!(
            p7 > p2,
            "low precision dim should have higher exploration priority"
        );
    }

    #[test]
    fn test_informed_prior_dense_class_higher() {
        let sparse = informed_prior_precision(1);
        let dense = informed_prior_precision(100);
        assert!(
            dense[0] > sparse[0],
            "dense classes should have higher initial precision"
        );
    }

    // -----------------------------------------------------------------------
    // BakePrecisionStore + BakeSession tests (Plan 236 Phase 3)
    // -----------------------------------------------------------------------

    #[cfg(feature = "bake_precision")]
    mod bake_precision_tests {
        use super::*;

        #[test]
        fn test_precision_store_insert_and_get() {
            let store = BakePrecisionStore::new();
            let entity = 42u64;
            let obs = [1.0f32; 8];

            store.update(entity, &obs, 1.0);

            let entry = store.get(entity).expect("entity should be tracked");
            // After one update from uninformative prior (precision 0.1) with obs precision 1.0:
            // new_precision = 0.1 + 1.0 = 1.1
            for d in 0..8 {
                assert!(
                    (entry.precision[d] - 1.1).abs() < 1e-6,
                    "expected precision 1.1, got {}",
                    entry.precision[d]
                );
                // mean = (0.1*0 + 1.0*1.0) / 1.1 ≈ 0.909
                let expected_mean = 1.0_f32 / 1.1_f32;
                assert!(
                    (entry.mean[d] - expected_mean).abs() < 1e-6,
                    "expected mean {}, got {}",
                    expected_mean,
                    entry.mean[d]
                );
            }
        }

        #[test]
        fn test_precision_store_missing_returns_none() {
            let store = BakePrecisionStore::new();
            assert!(
                store.get(999).is_none(),
                "untracked entity should return None"
            );
        }

        #[test]
        fn test_precision_store_eviction() {
            let store = BakePrecisionStore::new();
            let entity = 123u64;
            store.update(entity, &[0.5f32; 8], 1.0);

            assert!(store.get(entity).is_some(), "entity should be tracked");

            let evicted = store.remove(entity);
            assert!(evicted.is_some(), "should evict existing entry");
            assert!(store.get(entity).is_none(), "evicted entity should be gone");
        }

        #[test]
        fn test_precision_store_monotonic_via_update() {
            let store = BakePrecisionStore::new();
            let entity = 7u64;
            let obs = [0.5f32; 8];
            let mut prev_precision = [UNINFORMATIVE_PRECISION; 8];

            for _ in 0..100 {
                store.update(entity, &obs, 1.0);
                let entry = store.get(entity).expect("entity should be tracked");
                for d in 0..8 {
                    assert!(
                        entry.precision[d] >= prev_precision[d],
                        "precision should be monotonically non-decreasing via store update"
                    );
                }
                prev_precision = entry.precision;
            }
        }

        #[test]
        fn test_session_lifecycle() {
            let store = BakePrecisionStore::new();
            let entity = 42u64;

            // Initialize entity with some prior state.
            store.update(entity, &[0.0f32; 8], 0.5);
            let before = store.get(entity).expect("entity should exist");

            let mut session = BakeSession::begin(entity, &store);
            assert!(
                !session.is_active(),
                "new session should have no observations"
            );

            // Observe 10 times with obs = [1.0; 8]
            for _ in 0..10 {
                session.observe(&[1.0f32; 8]);
            }
            assert!(session.is_active(), "session should have observations");

            let result = session.end(&store);

            // Precision should have grown.
            for d in 0..8 {
                assert!(
                    result.precision[d] > before.precision[d],
                    "precision should grow after session, got {} vs {}",
                    result.precision[d],
                    before.precision[d]
                );
            }

            // Mean should have moved toward [1.0; 8].
            for d in 0..8 {
                assert!(
                    result.mean[d] > before.mean[d],
                    "mean should move toward observations"
                );
            }

            // Store should reflect the session result.
            let stored = store.get(entity).expect("entity should be in store");
            assert_eq!(stored, result, "store should match session result");
        }

        #[test]
        fn test_session_new_entity() {
            let store = BakePrecisionStore::new();
            let entity = 999u64; // Not in store

            assert!(store.get(entity).is_none());

            let mut session = BakeSession::begin(entity, &store);
            for _ in 0..5 {
                session.observe(&[0.5f32; 8]);
            }

            let result = session.end(&store);

            // Precision should be uninformative + 5 * DEFAULT_OBS_PRECISION.
            let expected_precision = UNINFORMATIVE_PRECISION + 5.0 * DEFAULT_OBS_PRECISION;
            for d in 0..8 {
                assert!(
                    (result.precision[d] - expected_precision).abs() < 1e-6,
                    "expected precision {}, got {}",
                    expected_precision,
                    result.precision[d]
                );
            }

            // Entity should now be tracked.
            assert!(
                store.get(entity).is_some(),
                "entity should be tracked after session"
            );
        }

        #[test]
        fn test_session_empty_noop() {
            let store = BakePrecisionStore::new();
            let entity = 42u64;

            // Set up known state.
            store.update(entity, &[0.5f32; 8], 1.0);
            let before = store.get(entity).expect("entity should exist");

            // Begin + end with no observations.
            let session = BakeSession::begin(entity, &store);
            assert!(!session.is_active());

            let result = session.end(&store);

            // Mean and precision should be unchanged.
            for d in 0..8 {
                assert!(
                    (result.mean[d] - before.mean[d]).abs() < 1e-6,
                    "empty session should not change mean, diff at dim {}",
                    d
                );
                assert!(
                    (result.precision[d] - before.precision[d]).abs() < 1e-6,
                    "empty session should not change precision, diff at dim {}",
                    d
                );
            }
        }
    }
}
