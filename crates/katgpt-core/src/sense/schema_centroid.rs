//! Schema-Centroid Informed KG Embedding Initialization (Plan 237).
//!
//! Pre-computes per-class embedding centroids from KG snapshots,
//! then initializes new entity embeddings near their class centroid
//! with controlled perturbation — avoiding cold-start randomness.

#![allow(clippy::needless_range_loop)]

use super::octree::KgEmbedding;

/// Pre-computed centroid statistics for a KG schema class.
///
/// Computed once per KG snapshot update, O(d·|E_c|) per class.
/// Stored in `SchemaCentroidCache` keyed by blake3 class hash.
#[derive(Clone, Copy, Debug)]
pub struct CentroidStats {
    /// Mean embedding of all entities in this class: v_c = (1/|E_c|) Σ e_i
    pub mean: [f32; 8],
    /// Per-dimension standard deviation: σ_c[d] = sqrt(Var(e[d]))
    pub std_dev: [f32; 8],
    /// Number of entities used to compute this centroid
    pub count: usize,
}

/// Compute centroid statistics from a slice of KgEmbedding values belonging to the same class.
///
/// Returns `None` if `embeddings` is empty (degenerate class).
/// Uses Welford's online algorithm for single-pass mean + variance.
/// O(d·|E_c|) — pure arithmetic, zero allocation beyond the return value.
#[inline]
pub fn compute_centroid(embeddings: &[KgEmbedding]) -> Option<CentroidStats> {
    if embeddings.is_empty() {
        return None;
    }

    let mut mean = [0.0f32; 8];
    let mut m2 = [0.0f32; 8];

    for (k, emb) in embeddings.iter().enumerate() {
        let inv_k_plus_1 = 1.0 / (k as f32 + 1.0);
        for d in 0..8 {
            let delta = emb.embedding[d] - mean[d];
            mean[d] += delta * inv_k_plus_1;
            let delta2 = emb.embedding[d] - mean[d];
            m2[d] += delta * delta2;
        }
    }

    let n = embeddings.len() as f32;
    let mut std_dev = [0.0f32; 8];
    for d in 0..8 {
        std_dev[d] = (m2[d] / n).sqrt();
    }

    Some(CentroidStats {
        mean,
        std_dev,
        count: embeddings.len(),
    })
}

/// Lock-free cache of schema class centroids, keyed by class hash.
///
/// Pre-computed once per KG snapshot update. O(1) lookup at entity init time.
/// Uses papaya lock-free HashMap for concurrent reads without blocking.
pub struct SchemaCentroidCache {
    centroids: papaya::HashMap<u64, CentroidStats>,
}

impl SchemaCentroidCache {
    /// Create an empty centroid cache.
    pub fn new() -> Self {
        Self {
            centroids: papaya::HashMap::new(),
        }
    }

    /// Look up pre-computed centroid for a class hash.
    ///
    /// Returns a cloned `CentroidStats` (72 bytes, cheap to clone).
    pub fn get(&self, class_hash: u64) -> Option<CentroidStats> {
        self.centroids.pin().get(&class_hash).cloned()
    }

    /// Insert or update centroid stats for a class hash.
    pub fn insert(&self, class_hash: u64, stats: CentroidStats) {
        self.centroids.pin().insert(class_hash, stats);
    }

    /// Number of cached classes.
    pub fn len(&self) -> usize {
        self.centroids.pin().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.centroids.pin().is_empty()
    }

    /// Clear all cached centroids.
    pub fn clear(&self) {
        self.centroids.pin().clear();
    }

    /// Compute centroid from embeddings and insert into cache.
    ///
    /// Returns `true` if computation succeeded (non-empty embeddings),
    /// `false` if `embeddings` was empty and nothing was inserted.
    pub fn compute_and_insert(&self, class_hash: u64, embeddings: &[KgEmbedding]) -> bool {
        match compute_centroid(embeddings) {
            Some(stats) => {
                self.insert(class_hash, stats);
                true
            }
            None => false,
        }
    }
}

impl Default for SchemaCentroidCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize a new entity's embedding from schema class centroids.
///
/// For each class the entity belongs to, look up centroid + std_dev.
/// Average centroids with perturbation: `μ = (1/|C|) Σ_c (v_c + γ·σ_c ⊙ r_c)`
/// Where `r_c` is random noise ∈ [-1, 1] per dimension per class (prevents identical init).
///
/// # Arguments
/// * `classes` — blake3 hashes of schema classes this entity belongs to
/// * `cache` — pre-computed centroid cache
/// * `gamma` — perturbation strength (0.0 = deterministic centroid, 1.0 = one σ noise)
/// * `rng` — random number generator for noise
///
/// # Returns
/// Initialized embedding `[f32; 8]`. Falls back to random init if no class found in cache.
pub fn schema_init_entity(
    classes: &[u64],
    cache: &SchemaCentroidCache,
    gamma: f32,
    rng: &mut fastrand::Rng,
) -> [f32; 8] {
    // Collect (mean, std_dev) for classes found in cache — stack-allocated, max 8 classes.
    // Single pin() guard reused for all lookups — avoids 8× epoch pin/unpin.
    let guard = cache.centroids.pin();
    let mut found: [Option<([f32; 8], [f32; 8])>; 8] = [None; 8];
    let mut found_count = 0usize;
    for &class_hash in classes {
        if found_count >= 8 {
            break;
        }
        if let Some(stats) = guard.get(&class_hash) {
            found[found_count] = Some((stats.mean, stats.std_dev));
            found_count += 1;
        }
    }
    drop(guard);

    // Fallback: random init in [-0.5, 0.5]
    if found_count == 0 {
        return random_init(rng);
    }

    let n_found = found_count as f32;
    let inv_n = 1.0 / n_found;
    let mut result = [0.0f32; 8];

    for i in 0..found_count {
        let (mean, std_dev) = found[i].expect("found[i] invariant: index < found_count");
        for d in 0..8 {
            let noise = rng.f32() * 2.0 - 1.0; // ∈ [-1, 1]
            result[d] += (mean[d] + gamma * std_dev[d] * noise) * inv_n;
        }
    }

    result
}

/// Initialize a new entity's embedding AND precision from schema class centroids.
///
/// When both `schema_centroid` and `bake_precision` features are enabled:
/// - `mean` = schema_init_entity(...) (centroid-based initialization)
/// - `precision` = informed_prior_precision(class_count) (dense classes → higher precision)
///
/// This upgrades BAKE's "uninformative prior" to an "informed prior" —
/// the entity starts closer to optimal, so BAKE converges faster.
///
/// Uses a single papaya pin to collect all class stats, avoiding double lookups.
#[cfg(feature = "bake_precision")]
pub fn schema_init_with_precision(
    classes: &[u64],
    cache: &SchemaCentroidCache,
    gamma: f32,
    rng: &mut fastrand::Rng,
) -> ([f32; 8], [f32; 8]) {
    use crate::sense::bake::informed_prior_precision;

    // Single pin — collect all class stats in one pass
    let guard = cache.centroids.pin();
    let mut found_stats: [Option<CentroidStats>; 8] = [None; 8];
    let mut found_count = 0usize;
    for &class_hash in classes.iter().take(8) {
        if let Some(stats) = guard.get(&class_hash).cloned() {
            found_stats[found_count] = Some(stats);
            found_count += 1;
        }
    }
    drop(guard);

    // Fallback: random init + uninformative prior
    if found_count == 0 {
        let embedding = random_init(rng);
        let precision = [crate::sense::bake::UNINFORMATIVE_PRECISION; 8];
        return (embedding, precision);
    }

    let n_found = found_count as f32;
    let inv_n = 1.0 / n_found;
    let mut embedding = [0.0f32; 8];
    let mut total_count = 0usize;

    for i in 0..found_count {
        let stats = found_stats[i].expect("found_stats[i] invariant: index < found_count");
        total_count += stats.count;
        for d in 0..8 {
            let noise = rng.f32() * 2.0 - 1.0;
            embedding[d] += (stats.mean[d] + gamma * stats.std_dev[d] * noise) * inv_n;
        }
    }

    let avg_count = total_count / found_count;
    let precision = informed_prior_precision(avg_count);

    (embedding, precision)
}

/// Random initialization fallback — uniform in [-0.5, 0.5] per dimension.
fn random_init(rng: &mut fastrand::Rng) -> [f32; 8] {
    let mut emb = [0.0f32; 8];
    for d in 0..8 {
        emb[d] = rng.f32() - 0.5;
    }
    emb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: [f32; 8]) -> KgEmbedding {
        KgEmbedding {
            entity_hash: 0,
            relation_hash: 0,
            embedding: values,
            sign: true,
            confidence: 1.0,
        }
    }

    #[test]
    fn test_compute_centroid_single() {
        let vals = [0.5, -0.3, 0.1, 0.8, -0.2, 0.4, 0.0, -0.6];
        let embs = [make_embedding(vals)];
        let stats = compute_centroid(&embs).expect("should return Some");

        assert_eq!(stats.count, 1);
        for d in 0..8 {
            assert!(
                (stats.mean[d] - vals[d]).abs() < 1e-6,
                "mean mismatch at dim {d}"
            );
            assert!(
                stats.std_dev[d].abs() < 1e-6,
                "std_dev should be 0 at dim {d}"
            );
        }
    }

    #[test]
    fn test_compute_centroid_multiple() {
        let embs = [
            make_embedding([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            make_embedding([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            make_embedding([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ];
        let stats = compute_centroid(&embs).expect("should return Some");

        assert_eq!(stats.count, 3);
        let expected_mean = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        for d in 0..8 {
            assert!(
                (stats.mean[d] - expected_mean[d]).abs() < 1e-5,
                "mean mismatch at dim {d}: got {} expected {}",
                stats.mean[d],
                expected_mean[d]
            );
        }

        // Population std_dev: sqrt(((1-3)^2 + (3-3)^2 + (5-3)^2) / 3) = sqrt(8/3) ≈ 1.6330
        let expected_std = (8.0f32 / 3.0).sqrt();
        for d in 0..8 {
            assert!(
                (stats.std_dev[d] - expected_std).abs() < 1e-4,
                "std_dev mismatch at dim {d}: got {} expected {}",
                stats.std_dev[d],
                expected_std
            );
        }
    }

    #[test]
    fn test_compute_centroid_empty() {
        assert!(compute_centroid(&[]).is_none());
    }

    #[test]
    fn test_cache_insert_get() {
        let cache = SchemaCentroidCache::new();
        let stats = CentroidStats {
            mean: [1.0; 8],
            std_dev: [0.5; 8],
            count: 10,
        };
        let hash = 42u64;

        cache.insert(hash, stats);
        let retrieved = cache.get(hash).expect("should find inserted");

        assert_eq!(retrieved.count, 10);
        for d in 0..8 {
            assert!((retrieved.mean[d] - 1.0).abs() < 1e-6);
            assert!((retrieved.std_dev[d] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cache_miss() {
        let cache = SchemaCentroidCache::new();
        assert!(cache.get(999).is_none());
    }

    #[test]
    fn test_schema_init_single_class() {
        let cache = SchemaCentroidCache::new();
        let class_hash = 100u64;
        let embs = [make_embedding([0.5; 8])];
        cache.compute_and_insert(class_hash, &embs);

        let mut rng = fastrand::Rng::with_seed(42);
        let result = schema_init_entity(&[class_hash], &cache, 0.0, &mut rng);

        // gamma=0 → result should be exactly the centroid mean
        for d in 0..8 {
            assert!(
                (result[d] - 0.5).abs() < 1e-6,
                "dim {d}: got {} expected 0.5",
                result[d]
            );
        }
    }

    #[test]
    fn test_schema_init_multi_class() {
        let cache = SchemaCentroidCache::new();

        let c1 = 200u64;
        cache.compute_and_insert(c1, &[make_embedding([1.0; 8])]);

        let c2 = 300u64;
        cache.compute_and_insert(c2, &[make_embedding([3.0; 8])]);

        let mut rng = fastrand::Rng::with_seed(99);
        let result = schema_init_entity(&[c1, c2], &cache, 0.0, &mut rng);

        // gamma=0 → average of [1.0;8] and [3.0;8] = [2.0;8]
        for d in 0..8 {
            assert!(
                (result[d] - 2.0).abs() < 1e-5,
                "dim {d}: got {} expected 2.0",
                result[d]
            );
        }
    }

    #[test]
    fn test_schema_init_fallback_unknown_class() {
        let cache = SchemaCentroidCache::new();
        let mut rng = fastrand::Rng::with_seed(7);
        let result = schema_init_entity(&[404u64], &cache, 0.5, &mut rng);

        // Should be random init — not all zeros
        let all_zero = result.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "fallback should produce non-zero random embedding"
        );

        // Should be in [-0.5, 0.5]
        for d in 0..8 {
            assert!(
                result[d] >= -0.5 && result[d] <= 0.5,
                "dim {d}: {} outside [-0.5, 0.5]",
                result[d]
            );
        }
    }

    #[test]
    fn test_schema_init_fallback_empty_classes() {
        let cache = SchemaCentroidCache::new();
        let mut rng = fastrand::Rng::with_seed(13);
        let result = schema_init_entity(&[], &cache, 0.5, &mut rng);

        let all_zero = result.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "empty classes should produce non-zero random embedding"
        );
    }

    #[test]
    fn test_schema_init_gamma_zero_deterministic() {
        let cache = SchemaCentroidCache::new();
        let class_hash = 500u64;
        cache.compute_and_insert(class_hash, &[make_embedding([0.7; 8])]);

        let seed = 12345u64;
        let r1 = schema_init_entity(
            &[class_hash],
            &cache,
            0.0,
            &mut fastrand::Rng::with_seed(seed),
        );
        let r2 = schema_init_entity(
            &[class_hash],
            &cache,
            0.0,
            &mut fastrand::Rng::with_seed(seed + 1),
        );

        // gamma=0 ignores rng, so both should be identical regardless of seed
        for d in 0..8 {
            assert!(
                (r1[d] - r2[d]).abs() < 1e-6,
                "dim {d}: {} != {} (should be deterministic with gamma=0)",
                r1[d],
                r2[d]
            );
        }
    }

    #[test]
    fn test_schema_init_gamma_nonzero_diverse() {
        let cache = SchemaCentroidCache::new();
        let class_hash = 600u64;
        cache.compute_and_insert(
            class_hash,
            &[
                make_embedding([1.0; 8]),
                make_embedding([2.0; 8]),
                make_embedding([3.0; 8]),
            ],
        );

        let r1 = schema_init_entity(&[class_hash], &cache, 1.0, &mut fastrand::Rng::with_seed(1));
        let r2 = schema_init_entity(&[class_hash], &cache, 1.0, &mut fastrand::Rng::with_seed(2));

        // gamma=1.0 with different seeds → embeddings should differ
        let identical = (0..8).all(|d| (r1[d] - r2[d]).abs() < 1e-6);
        assert!(
            !identical,
            "different seeds with gamma>0 should produce different embeddings"
        );
    }

    #[test]
    fn test_schema_init_perturbation_bounded() {
        let cache = SchemaCentroidCache::new();
        let class_hash = 700u64;
        let mean = [0.5, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6, 0.3];
        let embs: Vec<KgEmbedding> = (0..20)
            .map(|i| {
                let mut v = mean;
                for d in 0..8 {
                    v[d] += (i as f32 * 0.1) - 1.0; // spread around mean
                }
                make_embedding(v)
            })
            .collect();
        cache.compute_and_insert(class_hash, &embs);

        // gamma=1.0 → perturbation = ±1σ per dimension
        // Result = mean ± 1σ (from single class, after division by 1)
        // Run many times, check all results are within mean ± k*std_dev
        let k = 3.0; // 3σ bound should hold for uniform noise in [-1,1]
        let stats = cache.get(class_hash).unwrap();

        for seed in 0..50u64 {
            let result = schema_init_entity(
                &[class_hash],
                &cache,
                1.0,
                &mut fastrand::Rng::with_seed(seed),
            );
            for d in 0..8 {
                let lo = stats.mean[d] - k * stats.std_dev[d];
                let hi = stats.mean[d] + k * stats.std_dev[d];
                assert!(
                    result[d] >= lo && result[d] <= hi,
                    "seed {seed} dim {d}: {} outside [{lo}, {hi}]",
                    result[d]
                );
            }
        }
    }

    #[cfg(feature = "bake_precision")]
    #[test]
    fn test_schema_init_with_precision_informed() {
        let cache = SchemaCentroidCache::new();
        let class_hash = 42u64;
        // Create 10 embeddings for this class
        let embs: Vec<KgEmbedding> = (0..10)
            .map(|i| make_embedding([0.5 + i as f32 * 0.01; 8]))
            .collect();
        cache.compute_and_insert(class_hash, &embs);

        let mut rng = fastrand::Rng::with_seed(42);
        let (embedding, precision) =
            schema_init_with_precision(&[class_hash], &cache, 0.3, &mut rng);

        // Embedding should be near centroid
        let stats = cache.get(class_hash).unwrap();
        for d in 0..8 {
            assert!(
                (embedding[d] - stats.mean[d]).abs() < 1.0,
                "embedding should be near centroid"
            );
        }

        // Precision should be informed (dense class → higher precision)
        let expected_p = 10.0f32 / (1.0 + 10.0); // count=10, p = 10/11 ≈ 0.909
        for d in 0..8 {
            assert!(
                (precision[d] - expected_p).abs() < 1e-4,
                "precision should be informed by class count at dim {d}: got {} expected {}",
                precision[d],
                expected_p
            );
        }
    }

    #[cfg(feature = "bake_precision")]
    #[test]
    fn test_schema_init_with_precision_fallback() {
        let cache = SchemaCentroidCache::new();
        let mut rng = fastrand::Rng::with_seed(99);

        // Unknown class → should use uninformative prior
        let (embedding, precision) = schema_init_with_precision(&[404u64], &cache, 0.3, &mut rng);

        // Embedding should be random fallback (not all zeros)
        assert!(!embedding.iter().all(|&v| v == 0.0));

        // Precision should be uninformative
        for d in 0..8 {
            assert!(
                (precision[d] - 0.1).abs() < 1e-6,
                "fallback precision should be uninformative (0.1)"
            );
        }
    }

    #[cfg(feature = "bake_precision")]
    #[test]
    fn test_schema_init_with_precision_multi_class() {
        let cache = SchemaCentroidCache::new();

        // Class 1: 5 entities
        let c1 = 100u64;
        let embs1: Vec<KgEmbedding> = (0..5)
            .map(|i| make_embedding([1.0 + i as f32 * 0.1; 8]))
            .collect();
        cache.compute_and_insert(c1, &embs1);

        // Class 2: 20 entities
        let c2 = 200u64;
        let embs2: Vec<KgEmbedding> = (0..20)
            .map(|i| make_embedding([3.0 + i as f32 * 0.05; 8]))
            .collect();
        cache.compute_and_insert(c2, &embs2);

        let mut rng = fastrand::Rng::with_seed(42);
        let (_embedding, precision) = schema_init_with_precision(&[c1, c2], &cache, 0.0, &mut rng);

        // avg_count = (5 + 20) / 2 = 12, p = 12/13 ≈ 0.923
        let expected_p = 12.0f32 / (1.0 + 12.0);
        for d in 0..8 {
            assert!(
                (precision[d] - expected_p).abs() < 1e-3,
                "multi-class precision at dim {d}: got {} expected {}",
                precision[d],
                expected_p
            );
        }
    }
}
