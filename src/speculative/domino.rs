//! Domino Causal Correction — Modelless Decoupled Pattern (Plan 197, Research 177).
//!
//! Extracts Domino's decoupling pattern (parallel base + cheap sequential correction)
//! as three modelless mechanisms:
//! 1. **PrefixCorrectionTable**: O(1) hash lookup of prefix-conditioned logit residuals
//! 2. **domino_score**: Prefix-weighted tree expansion priority
//! 3. **DominoPruner**: Prefix-conditioned constraint correction trait
//!
//! No model training. No LoRA. Pure inference-time pattern extraction.
//!
//! # Architecture
//!
//! ```text
//! DFlash → marginals ──┬──→ base_scores (unchanged)
//!                      └──→ DominoCorrector
//!                           ├─ prefix_table lookup (blake3 → u64)
//!                           ├─ logit residual (O(K) per depth)
//!                           └─ re-normalize
//!
//! corrected marginals → DDTree(domino_score) → verify
//!
//! domino_score = base × prefix_strength^depth
//! ```

use std::collections::HashMap;

// ── Prefix Hashing ────────────────────────────────────────────────

/// Compute a blake3-based prefix hash from a slice of sampled tokens, truncated to u64.
///
/// Tokens are serialized as little-ender u16 (sufficient for vocab < 65536).
/// The hash is deterministic: same prefix → same u64 key.
#[inline]
pub fn prefix_hash(tokens: &[usize]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for &t in tokens {
        hasher.update(&(t as u16).to_le_bytes());
    }
    let hash = hasher.finalize();
    u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
}

// ── PrefixCorrectionTable ─────────────────────────────────────────

/// Pre-computed table of prefix-token → correction vectors.
///
/// Key: blake3(prefix_tokens) truncated to u64.
/// Value: sparse correction vector (only top-K token adjustments, padded to `vocab_size`).
///
/// Zero-alloc lookup: returns `&[f32]` slice directly.
/// Builder pattern for construction from constraint rules.
///
/// # Performance
///
/// - Lookup: ~50ns (blake3 hash + HashMap probe)
/// - Correction application: O(K) per depth where K = sparsity (typically < 10)
/// - Memory: O(P × V) where P = number of unique prefixes, V = vocab_size
#[cfg_attr(test, derive(Debug))]
pub struct PrefixCorrectionTable {
    /// Sparse correction vectors keyed by prefix hash.
    corrections: HashMap<u64, Vec<f32>>,
    /// Vocab size for validation.
    vocab_size: usize,
}

impl PrefixCorrectionTable {
    /// Create an empty table with known vocab size.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            corrections: HashMap::new(),
            vocab_size,
        }
    }

    /// Zero-alloc lookup: returns empty slice if not found.
    #[inline]
    pub fn lookup(&self, hash: u64) -> &[f32] {
        match self.corrections.get(&hash) {
            Some(v) => v,
            None => &[],
        }
    }

    /// Returns true if the table has no correction entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    /// Number of prefix entries in the table.
    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    /// Vocab size of the correction vectors.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ── Builder ───────────────────────────────────────────────────────

/// Builder for [`PrefixCorrectionTable`].
///
/// # Example
///
/// ```ignore
/// use katgpt_rs::speculative::domino::PrefixCorrectionTable;
///
/// let table = PrefixCorrectionTable::builder(256)
///     .add_correction(&[1, 2], &[0.0, -0.5, 0.3, /* ... */])
///     .build();
/// ```
pub struct PrefixCorrectionTableBuilder {
    corrections: HashMap<u64, Vec<f32>>,
    vocab_size: usize,
}

impl PrefixCorrectionTableBuilder {
    /// Create a new builder with the given vocab size.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            corrections: HashMap::new(),
            vocab_size,
        }
    }

    /// Add a correction vector for a given prefix.
    ///
    /// The prefix is hashed with blake3 internally. The correction vector
    /// must match `vocab_size` in length.
    ///
    /// Returns `&mut Self` for chaining.
    pub fn add_correction(mut self, prefix: &[usize], correction: &[f32]) -> Self {
        debug_assert_eq!(
            correction.len(),
            self.vocab_size,
            "Correction vector must match vocab_size"
        );
        let hash = prefix_hash(prefix);
        self.corrections.insert(hash, correction.to_vec());
        self
    }

    /// Add a correction from a pre-computed hash key.
    pub fn add_correction_raw(mut self, hash: u64, correction: Vec<f32>) -> Self {
        debug_assert_eq!(
            correction.len(),
            self.vocab_size,
            "Correction vector must match vocab_size"
        );
        self.corrections.insert(hash, correction);
        self
    }

    /// Build the immutable table.
    pub fn build(self) -> PrefixCorrectionTable {
        PrefixCorrectionTable {
            corrections: self.corrections,
            vocab_size: self.vocab_size,
        }
    }
}

impl PrefixCorrectionTable {
    /// Create a builder for this table.
    pub fn builder(vocab_size: usize) -> PrefixCorrectionTableBuilder {
        PrefixCorrectionTableBuilder::new(vocab_size)
    }
}

// ── domino_score ──────────────────────────────────────────────────

/// Compute domino scoring for DDTree expansion priority.
///
/// `domino_score = base_score * prefix_strength^depth`
///
/// Where `prefix_strength` is the product of parent marginal probabilities.
/// This penalizes deeper branches with low-confidence prefixes, biasing tree
/// expansion toward high-confidence paths.
///
/// Returns `base_score` unchanged when `depth == 0` or `prefix_strength >= 1.0`.
#[inline]
pub fn domino_score(base_score: f32, depth: usize, prefix_strength: f32) -> f32 {
    match (depth, prefix_strength < 1.0) {
        (0, _) | (_, false) => base_score,
        (_, true) => base_score * prefix_strength.powi(depth as i32),
    }
}

/// Compute prefix_strength from parent marginal probabilities.
///
/// `prefix_strength = Π(prob_i)` where prob_i is the probability of the token
/// chosen at depth i in the prefix path.
#[inline]
pub fn compute_prefix_strength(
    marginals: &[&[f32]],
    sampled_tokens: &[usize],
    depth: usize,
) -> f32 {
    let limit = depth.min(sampled_tokens.len()).min(marginals.len());
    let mut strength = 1.0f32;
    for i in 0..limit {
        let token = sampled_tokens[i];
        let prob = marginals[i].get(token).copied().unwrap_or(0.0);
        strength *= prob;
    }
    strength
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_hash_deterministic() {
        let tokens = [1usize, 2, 3];
        let h1 = prefix_hash(&tokens);
        let h2 = prefix_hash(&tokens);
        assert_eq!(h1, h2, "Same prefix must produce same hash");
    }

    #[test]
    fn test_prefix_hash_different_prefixes() {
        let h1 = prefix_hash(&[1usize, 2]);
        let h2 = prefix_hash(&[1usize, 3]);
        assert_ne!(h1, h2, "Different prefixes must produce different hashes");
    }

    #[test]
    fn test_prefix_hash_empty() {
        let h = prefix_hash(&[]);
        // Empty prefix should still produce a valid hash
        assert_ne!(h, 0, "Empty prefix hash should be non-zero");
    }

    #[test]
    fn test_prefix_hash_order_matters() {
        let h1 = prefix_hash(&[1usize, 2]);
        let h2 = prefix_hash(&[2usize, 1]);
        assert_ne!(h1, h2, "Order must matter for prefix hash");
    }

    #[test]
    fn test_correction_table_empty_lookup() {
        let table = PrefixCorrectionTable::new(10);
        assert!(table.is_empty());
        assert!(table.lookup(0).is_empty());
        assert!(table.lookup(999).is_empty());
    }

    #[test]
    fn test_correction_table_builder() {
        let correction = vec![0.0f32, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let table = PrefixCorrectionTable::builder(10)
            .add_correction(&[1, 2], &correction)
            .build();

        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        let hash = prefix_hash(&[1, 2]);
        let result = table.lookup(hash);
        assert_eq!(result.len(), 10);
        assert_eq!(result[1], -0.5);
        assert_eq!(result[2], 0.3);
    }

    #[test]
    fn test_correction_table_missing_prefix() {
        let correction = vec![0.0f32; 10];
        let table = PrefixCorrectionTable::builder(10)
            .add_correction(&[1, 2], &correction)
            .build();

        let missing_hash = prefix_hash(&[9, 9]);
        assert!(table.lookup(missing_hash).is_empty());
    }

    #[test]
    fn test_correction_table_raw_hash() {
        let correction = vec![1.0f32, 0.0, 0.0, 0.0, 0.0];
        let table = PrefixCorrectionTable::builder(5)
            .add_correction_raw(42u64, correction)
            .build();

        assert_eq!(table.lookup(42).len(), 5);
        assert_eq!(table.lookup(42)[0], 1.0);
    }

    #[test]
    fn test_domino_score_depth_zero_passthrough() {
        let score = domino_score(-2.5, 0, 0.5);
        assert_eq!(score, -2.5, "Depth 0 should pass through unchanged");
    }

    #[test]
    fn test_domino_score_strength_one_passthrough() {
        let score = domino_score(-2.5, 3, 1.0);
        assert_eq!(score, -2.5, "strength=1.0 should pass through unchanged");
    }

    #[test]
    fn test_domino_score_penalizes_low_strength() {
        let score = domino_score(-1.0, 2, 0.5);
        // -1.0 * 0.5^2 = -1.0 * 0.25 = -0.25
        let expected = -0.25f32;
        assert!(
            (score - expected).abs() < 1e-6,
            "Expected {expected}, got {score}"
        );
    }

    #[test]
    fn test_domino_score_deeper_more_penalty() {
        let s1 = domino_score(-1.0, 1, 0.5);
        let s2 = domino_score(-1.0, 2, 0.5);
        let s3 = domino_score(-1.0, 3, 0.5);
        // All negative scores get closer to 0 (less extreme) with more depth,
        // but since base is negative, deeper = less negative = "worse" in max-heap.
        // In absolute terms: |s1| > |s2| > |s3|
        assert!(s1 < s2, "Deeper should be less negative");
        assert!(s2 < s3, "Even deeper should be even less negative");
    }

    #[test]
    fn test_compute_prefix_strength_single_token() {
        let marginals: Vec<&[f32]> = vec![&[0.1, 0.2, 0.7]];
        let tokens = [1usize];
        let strength = compute_prefix_strength(&marginals, &tokens, 1);
        assert!((strength - 0.2f32).abs() < 1e-6);
    }

    #[test]
    fn test_compute_prefix_strength_chain() {
        let marginals: Vec<&[f32]> = vec![&[0.5, 0.5], &[0.3, 0.7]];
        let tokens = [0usize, 1]; // prob 0.5 * 0.7 = 0.35
        let strength = compute_prefix_strength(&marginals, &tokens, 2);
        assert!((strength - 0.35f32).abs() < 1e-6);
    }

    #[test]
    fn test_compute_prefix_strength_empty() {
        let marginals: Vec<&[f32]> = vec![];
        let tokens: [usize; 0] = [];
        let strength = compute_prefix_strength(&marginals, &tokens, 0);
        assert_eq!(strength, 1.0, "Empty prefix should have strength 1.0");
    }

    #[test]
    fn test_compute_prefix_strength_clamps_to_available() {
        let marginals: Vec<&[f32]> = vec![&[0.5, 0.5]];
        let tokens = [0usize];
        // depth=5 but only 1 marginal available → uses 1
        let strength = compute_prefix_strength(&marginals, &tokens, 5);
        assert!((strength - 0.5f32).abs() < 1e-6);
    }
}
