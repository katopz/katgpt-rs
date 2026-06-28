//! Hashes context features into a compact r-dimensional vector.
//!
//! Replaces the paper's learned projections (Wmq, Wmk, Wmv).
//! Verified from `delta_impl.py` L805-814 (_normalize_memory_projection):
//!   key = L2_norm(tanh(W_mk · x))   (unit sphere, prevents explosion)
//!   val = W_mv · x                  (no normalization on values)
//!
//! We replace learned W with random LSH projection + same normalization.
//!
//! # Storage layout (Plan 008 Phase 2.6 reconciliation, 2026-06-28)
//!
//! The projection matrix is **generated column-major** then **transposed to
//! row-major** at construction. This preserves the historical bit-pattern of
//! `logical(i, j) = rng_call(j * rank + i)` (so any persisted snapshot seeded
//! under previous versions still hashes consistently) while giving the hot
//! path contiguous rows for SIMD dot-product acceleration. The transpose
//! trick is the same one used by `FourierFeatureHasher::init_projection`
//! (`riir-engine/src/fourier/opponent_hash.rs`).

/// Feature hasher using random projection.
///
/// Same normalization as paper: tanh → L2 normalize for keys/queries,
/// raw projection for values.
pub struct FeatureHasher {
    /// Memory rank.
    rank: usize,
    /// Input feature dimension (cached to avoid recomputation in hot paths).
    feature_dim: usize,
    /// Random projection matrix [rank × feature_dim], **row-major**
    /// (`projection[i * feature_dim + j]` = logical element (i, j)).
    /// Initialized column-major from `fastrand` for bit-stability, then
    /// transposed in-place to row-major for SIMD-friendly access.
    projection: Vec<f32>,
    /// Seed for deterministic hashing.
    seed: u64,
}

impl FeatureHasher {
    /// Create a new feature hasher with random projection.
    ///
    /// Uses Kaiming-like initialization scaled by sqrt(2/rank).
    ///
    /// The RNG sequence is consumed column-major (matching the historical
    /// katgpt-core bit-pattern) and then transposed to row-major storage
    /// so the hot path can use contiguous SIMD dot products.
    pub fn new(rank: usize, feature_dim: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let scale = (2.0 / rank as f32).sqrt();

        // Generate column-major: cm[j * rank + i] = rng_call(j * rank + i) = logical(i, j).
        // This matches the historical katgpt-core bit-pattern exactly.
        let cm: Vec<f32> = (0..(rank * feature_dim))
            .map(|_| rng.f32() * 2.0 * scale - scale)
            .collect();

        // Transpose to row-major: rm[i * feature_dim + j] = cm[j * rank + i] = logical(i, j).
        // Row-major storage lets `project_into` use contiguous SIMD dots.
        let mut projection = vec![0.0f32; rank * feature_dim];
        for i in 0..rank {
            for j in 0..feature_dim {
                projection[i * feature_dim + j] = cm[j * rank + i];
            }
        }

        Self {
            rank,
            feature_dim,
            projection,
            seed,
        }
    }

    /// Hash to L2-normalized key/query vector.
    /// `L2_norm(tanh(projection · features))` — same as paper Eq 4.
    pub fn hash_key(&self, features: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.rank];
        self.hash_key_into(features, &mut result);
        result
    }

    /// Hash key into pre-allocated buffer. Zero-alloc for hot path.
    /// `L2_norm(tanh(projection · features))` — same as paper Eq 4.
    ///
    /// Produces output bit-identical to [`Self::hash_key`] when called with
    /// the same `features` and an equally-sized `out` buffer.
    pub fn hash_key_into(&self, features: &[f32], out: &mut [f32]) {
        self.project_into(features, out);
        // tanh in-place
        for val in out.iter_mut() {
            *val = val.tanh();
        }
        // L2 normalize in-place (prevents state explosion — verified from source).
        // SIMD-accelerated sum-of-squares for the norm denominator.
        let norm: f32 = crate::simd::simd_sum_sq(out, out.len()).sqrt().max(1e-8);
        for val in out.iter_mut() {
            *val /= norm;
        }
    }

    /// Hash to raw value vector (no normalization, same as paper).
    /// `projection · features`
    pub fn hash_value(&self, features: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.rank];
        self.hash_value_into(features, &mut result);
        result
    }

    /// Hash value into pre-allocated buffer. Zero-alloc for hot path.
    /// `projection · features` — no normalization, same as paper.
    ///
    /// Produces output bit-identical to [`Self::hash_value`].
    pub fn hash_value_into(&self, features: &[f32], out: &mut [f32]) {
        self.project_into(features, out);
    }

    /// Project into pre-allocated buffer. Zero-alloc for hot path.
    /// Matrix-vector multiply: projection · features
    ///
    /// Uses SIMD dot product for each row when feature_dim is large enough.
    /// Row-major storage means each row is contiguous — ideal for SIMD.
    #[inline]
    fn project_into(&self, features: &[f32], out: &mut [f32]) {
        assert_eq!(out.len(), self.rank, "output dimension must match rank");
        out.fill(0.0);
        let fd = self.feature_dim;
        for (i, slot) in out.iter_mut().enumerate() {
            let row_off = i * fd;
            *slot =
                crate::simd::simd_dot_f32(&self.projection[row_off..row_off + fd], features, fd);
        }
    }

    /// Get the rank dimension.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the input feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Get the seed used for initialization.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Clone for FeatureHasher {
    fn clone(&self) -> Self {
        Self {
            rank: self.rank,
            feature_dim: self.feature_dim,
            projection: self.projection.clone(),
            seed: self.seed,
        }
    }
}

/// Extract features from DDTree context for memory hashing.
#[derive(Clone, Debug)]
pub struct ContextFeatures {
    /// Domain hash (from PromptRouter domain string)
    pub domain: u64,
    /// Current depth in DDTree (normalized to [0, 1])
    pub depth_normalized: f32,
    /// Token entropy at current position (from marginals)
    pub token_entropy: f32,
    /// Parent path length (normalized)
    pub path_length_normalized: f32,
    /// Screening relevance score at parent
    pub parent_relevance: f32,
}

impl ContextFeatures {
    /// Feature vector dimension (always 8).
    pub const FEATURE_DIM: usize = 8;

    /// Convert to feature vector for hashing.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            (self.domain & 0xFF) as f32 / 255.0, // Low byte of domain hash
            ((self.domain >> 8) & 0xFF) as f32 / 255.0,
            ((self.domain >> 16) & 0xFF) as f32 / 255.0,
            ((self.domain >> 24) & 0xFF) as f32 / 255.0,
            self.depth_normalized,
            self.token_entropy,
            self.path_length_normalized,
            self.parent_relevance,
        ]
    }

    /// Convert to feature vector into pre-allocated buffer. Zero-alloc for hot path.
    /// Buffer is cleared and resized if needed.
    pub fn to_vec_into(&self, buf: &mut Vec<f32>) {
        buf.clear();
        buf.extend_from_slice(&[
            (self.domain & 0xFF) as f32 / 255.0,
            ((self.domain >> 8) & 0xFF) as f32 / 255.0,
            ((self.domain >> 16) & 0xFF) as f32 / 255.0,
            ((self.domain >> 24) & 0xFF) as f32 / 255.0,
            self.depth_normalized,
            self.token_entropy,
            self.path_length_normalized,
            self.parent_relevance,
        ]);
    }

    /// Convert to fixed-size feature array. Zero-alloc, no Vec needed.
    /// Preferred for hot paths where the caller has stack space.
    pub fn to_array(&self) -> [f32; 8] {
        [
            (self.domain & 0xFF) as f32 / 255.0,
            ((self.domain >> 8) & 0xFF) as f32 / 255.0,
            ((self.domain >> 16) & 0xFF) as f32 / 255.0,
            ((self.domain >> 24) & 0xFF) as f32 / 255.0,
            self.depth_normalized,
            self.token_entropy,
            self.path_length_normalized,
            self.parent_relevance,
        ]
    }

    /// Extract from DDTree context during build.
    pub fn from_tree_context(depth: usize, _token_idx: usize, parent_tokens: &[usize]) -> Self {
        let max_depth = 32.0f32; // Typical DDTree max depth
        let max_path = 256.0f32; // Typical max path length

        Self {
            domain: 0, // Set by caller based on PromptRouter
            depth_normalized: (depth as f32 / max_depth).min(1.0),
            token_entropy: 0.0, // Would be filled from model marginals
            path_length_normalized: (parent_tokens.len() as f32 / max_path).min(1.0),
            parent_relevance: 1.0, // Default: no parent info
        }
    }
}

/// Extract features from generation outcome for memory values.
#[derive(Clone, Debug)]
pub struct OutcomeFeatures {
    /// Hint-δ value (from DeltaBanditPruner)
    pub delta: f32,
    /// Solution quality (path length / budget ratio)
    pub quality: f32,
    /// Whether DDTree found a valid solution (0.0 or 1.0)
    pub success: f32,
}

impl OutcomeFeatures {
    /// Feature vector dimension (always 3).
    pub const FEATURE_DIM: usize = 3;

    /// Convert to feature vector for memory value.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.delta, self.quality, self.success]
    }

    /// Convert to feature vector into pre-allocated buffer. Zero-alloc for hot path.
    pub fn to_vec_into(&self, buf: &mut Vec<f32>) {
        buf.clear();
        buf.extend_from_slice(&[self.delta, self.quality, self.success]);
    }

    /// Convert to fixed-size feature array. Zero-alloc, no Vec needed.
    pub fn to_array(&self) -> [f32; 3] {
        [self.delta, self.quality, self.success]
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_key_is_normalized() {
        let hasher = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let key = hasher.hash_key(&features);

        // L2 norm should be ~1.0
        let norm: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Key should be L2-normalized, got norm={norm}"
        );
    }

    #[test]
    fn test_hash_value_is_raw() {
        let hasher = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let val = hasher.hash_value(&features);

        // Values should NOT be normalized (can have any magnitude)
        assert_eq!(val.len(), 8);
        // Values should generally be non-zero for non-zero input
        assert!(val.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_deterministic_same_seed() {
        let h1 = FeatureHasher::new(8, 5, 42);
        let h2 = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let k1 = h1.hash_key(&features);
        let k2 = h2.hash_key(&features);

        for (a, b) in k1.iter().zip(k2.iter()) {
            assert!((a - b).abs() < 1e-6, "Same seed should produce same hash");
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let h1 = FeatureHasher::new(8, 5, 42);
        let h2 = FeatureHasher::new(8, 5, 99);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let k1 = h1.hash_key(&features);
        let k2 = h2.hash_key(&features);

        // Very unlikely all dimensions are identical
        let identical = k1.iter().zip(k2.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !identical,
            "Different seeds should produce different hashes"
        );
    }

    #[test]
    fn test_hash_key_into_matches_hash_key() {
        let hasher = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_alloc = hasher.hash_key(&features);
        let mut key_prealloc = vec![0.0; 8];
        hasher.hash_key_into(&features, &mut key_prealloc);

        for (a, b) in key_alloc.iter().zip(key_prealloc.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "_into should match allocating version"
            );
        }
    }

    #[test]
    fn test_hash_value_into_matches_hash_value() {
        let hasher = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let val_alloc = hasher.hash_value(&features);
        let mut val_prealloc = vec![0.0; 8];
        hasher.hash_value_into(&features, &mut val_prealloc);

        for (a, b) in val_alloc.iter().zip(val_prealloc.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "_into should match allocating version"
            );
        }
    }

    #[test]
    #[should_panic(expected = "output dimension must match rank")]
    fn test_project_into_wrong_size_panics() {
        let hasher = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut wrong = vec![0.0; 4];
        hasher.hash_key_into(&features, &mut wrong);
    }

    #[test]
    fn test_context_features_to_vec() {
        let ctx = ContextFeatures {
            domain: 0x01020304,
            depth_normalized: 0.5,
            token_entropy: 0.3,
            path_length_normalized: 0.2,
            parent_relevance: 0.8,
        };
        let vec = ctx.to_vec();
        assert_eq!(vec.len(), 8);
        assert!((vec[0] - 4.0 / 255.0).abs() < 1e-6); // low byte
        assert!((vec[4] - 0.5).abs() < 1e-6); // depth_normalized
    }

    #[test]
    fn test_context_features_to_vec_into_matches_to_vec() {
        let ctx = ContextFeatures {
            domain: 0x01020304,
            depth_normalized: 0.5,
            token_entropy: 0.3,
            path_length_normalized: 0.2,
            parent_relevance: 0.8,
        };
        let alloc = ctx.to_vec();
        let mut buf = Vec::new();
        ctx.to_vec_into(&mut buf);
        assert_eq!(alloc.len(), buf.len());
        for (a, b) in alloc.iter().zip(buf.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "_into should match allocating version"
            );
        }
    }

    #[test]
    fn test_context_features_to_array_matches_to_vec() {
        let ctx = ContextFeatures {
            domain: 0x01020304,
            depth_normalized: 0.5,
            token_entropy: 0.3,
            path_length_normalized: 0.2,
            parent_relevance: 0.8,
        };
        let vec = ctx.to_vec();
        let arr = ctx.to_array();
        assert_eq!(vec.len(), arr.len());
        for (a, b) in vec.iter().zip(arr.iter()) {
            assert!((a - b).abs() < 1e-6, "to_array should match to_vec");
        }
    }

    #[test]
    fn test_context_features_from_tree() {
        let ctx = ContextFeatures::from_tree_context(16, 3, &[1, 2, 3]);
        assert!((ctx.depth_normalized - 0.5).abs() < 0.1);
        assert!((ctx.path_length_normalized - 3.0 / 256.0).abs() < 0.01);
    }

    #[test]
    fn test_outcome_features_to_vec() {
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        let vec = outcome.to_vec();
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_outcome_features_to_vec_into_matches_to_vec() {
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        let alloc = outcome.to_vec();
        let mut buf = Vec::new();
        outcome.to_vec_into(&mut buf);
        assert_eq!(alloc.len(), buf.len());
        for (a, b) in alloc.iter().zip(buf.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "_into should match allocating version"
            );
        }
    }

    #[test]
    fn test_outcome_features_to_array_matches_to_vec() {
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        let vec = outcome.to_vec();
        let arr = outcome.to_array();
        assert_eq!(vec.len(), arr.len());
        for (a, b) in vec.iter().zip(arr.iter()) {
            assert!((a - b).abs() < 1e-6, "to_array should match to_vec");
        }
    }

    #[test]
    fn test_key_tanh_bounds() {
        let hasher = FeatureHasher::new(8, 5, 42);
        // Very large features
        let features = vec![100.0, -100.0, 100.0, -100.0, 100.0];
        let key = hasher.hash_key(&features);

        // After tanh, all values should be in [-1, 1]
        for &k in &key {
            assert!(k.abs() <= 1.0 + 1e-6, "tanh output should be in [-1, 1]");
        }
    }

    /// Bit-stability guard: the column-major-then-transpose construction MUST
    /// produce the same logical projection as the historical katgpt-core
    /// column-major access pattern. If this test ever fails, the construction
    /// order changed and persisted snapshots seeded under older versions would
    /// hash inconsistently.
    #[test]
    fn test_projection_logical_values_match_column_major_rng_sequence() {
        let rank = 8;
        let feature_dim = 5;
        let seed = 42;
        let hasher = FeatureHasher::new(rank, feature_dim, seed);

        // Re-derive the expected logical(i,j) from the raw RNG sequence
        // (column-major fill: cm[j*rank+i] = rng_call(j*rank+i)).
        let mut rng = fastrand::Rng::with_seed(seed);
        let scale = (2.0 / rank as f32).sqrt();
        let cm: Vec<f32> = (0..(rank * feature_dim))
            .map(|_| rng.f32() * 2.0 * scale - scale)
            .collect();

        for i in 0..rank {
            for j in 0..feature_dim {
                let expected = cm[j * rank + i];
                let actual = hasher.projection[i * feature_dim + j];
                assert!(
                    (expected - actual).abs() == 0.0,
                    "projection[({}, {})] bit-drift: expected {} got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
    }

    /// Sanity: feature_dim accessor returns the configured value.
    #[test]
    fn test_feature_dim_accessor() {
        let hasher = FeatureHasher::new(8, 5, 42);
        assert_eq!(hasher.feature_dim(), 5);
        assert_eq!(hasher.rank(), 8);
    }
}
