//! PhraseBoost — context trie phrase boosting for DDTree.
//!
//! Wraps any [`ScreeningPruner`] and adds domain-specific token biasing via a
//! Context Trie. Zero training cost — phrases are provided at call site.
//! Modeled after parakeet.cpp's phrase boosting, adapted to our DDTree pipeline.
//!
//! Feature-gated behind `phrase_boost` (default-OFF until GOAT proves gain).

use std::collections::HashMap;
use std::sync::RwLock;

use crate::speculative::types::ScreeningPruner;

use super::phrase_trie::PhraseTrie;

// ── Constants ──────────────────────────────────────────────────

/// Default raw boost score (5.0). After normalization: 5.0 / 6.0 ≈ 0.833.
pub const DEFAULT_BOOST_SCORE: f32 = 5.0;

// ── PhraseBoostPruner ──────────────────────────────────────────

/// Wraps any [`ScreeningPruner`] and adds phrase-based token boosting via a context trie.
///
/// For each position in the DDTree, the pruner:
/// 1. Looks up the active trie states from the parent path.
/// 2. Checks if the candidate token is a child of any active state.
/// 3. If so, adds a normalized boost: `boost_score / (1.0 + boost_score)`.
///
/// This is additive — the inner pruner's relevance is preserved, and the boost is layered on top.
///
/// # Interior Mutability
///
/// [`ScreeningPruner::relevance()`] takes `&self`, but we need to track active trie states
/// per path (mutating `HashMap`). We use `RefCell` for safe interior mutability.
pub struct PhraseBoostPruner<P: ScreeningPruner> {
    /// Inner domain-specific pruner.
    inner: P,
    /// Context trie holding all phrase sequences.
    trie: PhraseTrie,
    /// Raw boost score before normalization.
    boost_score: f32,
    /// Per-path active trie states. Keyed by hash of the parent token sequence.
    active_states: RwLock<HashMap<u128, Vec<usize>>>,
}

impl<P: ScreeningPruner> PhraseBoostPruner<P> {
    /// Create a new PhraseBoostPruner.
    ///
    /// - `inner`: the base pruner to delegate to.
    /// - `trie`: pre-built phrase trie.
    /// - `boost_score`: raw boost value; will be normalized via `boost_score / (1 + boost_score)`.
    pub fn new(inner: P, trie: PhraseTrie, boost_score: f32) -> Self {
        Self {
            inner,
            trie,
            boost_score,
            active_states: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default boost score (raw 5.0, normalizes to 5/6 ≈ 0.833).
    pub fn with_default_boost(inner: P, trie: PhraseTrie) -> Self {
        Self::new(inner, trie, DEFAULT_BOOST_SCORE)
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Normalized boost value: `boost_score / (1.0 + boost_score)`.
    ///
    /// Maps any positive `boost_score` to [0, 1), ensuring the result stays bounded.
    /// Example: boost_score = 5.0 → normalized = 5/6 ≈ 0.833.
    pub fn normalized_boost(&self) -> f32 {
        self.boost_score / (1.0 + self.boost_score)
    }

    /// Compute a fast hash of the parent token sequence for state tracking.
    ///
    /// Uses FNV-1a-like folding for speed — not cryptographic, just a cache key.
    fn hash_path(parent_tokens: &[usize]) -> u128 {
        let mut hash: u128 = 0x6c62272e07bb0142u128; // FNV offset basis (128-bit)
        for &tok in parent_tokens {
            hash ^= tok as u128;
            hash = hash.wrapping_mul(0x1000000000000000000013bu128); // FNV prime (128-bit)
        }
        hash
    }

    /// Get or compute the active trie states for a given parent path.
    fn get_active(&self, parent_tokens: &[usize]) -> Vec<usize> {
        let key = Self::hash_path(parent_tokens);
        let mut states = self.active_states.write().unwrap();
        states
            .entry(key)
            .or_insert_with(|| {
                // Walk the trie from root, advancing through each parent token.
                let mut active = vec![0]; // start at root
                for &tok in parent_tokens {
                    active = self.trie.advance(&active, tok);
                }
                active
            })
            .clone()
    }

    /// Check if `token_idx` is boosted given the current parent path.
    fn is_boosted(&self, parent_tokens: &[usize], token_idx: usize) -> bool {
        let active = self.get_active(parent_tokens);
        let boosted = self.trie.get_boosted_tokens(&active);
        boosted.contains(&token_idx)
    }
}

// ── ScreeningPruner impl ───────────────────────────────────────

impl<P: ScreeningPruner> ScreeningPruner for PhraseBoostPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let base = self.inner.relevance(depth, token_idx, parent_tokens);
        let boosted = self.is_boosted(parent_tokens, token_idx);
        let boost = if boosted {
            self.normalized_boost()
        } else {
            0.0
        };
        base + boost
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: pruner that always returns 0.0.
    /// Lets us isolate boost contribution in assertions.
    struct ZeroPruner;

    impl ScreeningPruner for ZeroPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            0.0
        }
    }

    fn make_test_trie() -> PhraseTrie {
        // "hello" = [5], "hello world" = [5, 10], "hello there" = [5, 20]
        let mut trie = PhraseTrie::new(128);
        trie.insert(&[5, 10]); // "hello world"
        trie.insert(&[5, 20]); // "hello there"
        trie.insert(&[30]); // single-token phrase
        trie
    }

    #[test]
    fn test_boost_known_phrase_tokens() {
        let trie = make_test_trie();
        let pruner = PhraseBoostPruner::new(ZeroPruner, trie, 5.0);

        // At root (empty path), tokens 5 and 30 should be boosted
        let rel_5 = pruner.relevance(0, 5, &[]);
        assert!(
            rel_5 > 0.0,
            "token 5 (start of 'hello') should get boost, got {rel_5}"
        );

        let rel_30 = pruner.relevance(0, 30, &[]);
        assert!(
            rel_30 > 0.0,
            "token 30 (single-token phrase) should get boost, got {rel_30}"
        );

        // Token 99 is not in any phrase — no boost
        let rel_99 = pruner.relevance(0, 99, &[]);
        assert_eq!(rel_99, 0.0, "token 99 should not be boosted");
    }

    #[test]
    fn test_boost_normalization_bounded() {
        let trie = make_test_trie();
        let pruner = PhraseBoostPruner::new(ZeroPruner, trie, 5.0);

        // Normalized boost = 5.0 / 6.0 ≈ 0.833
        let normalized = pruner.normalized_boost();
        assert!(
            (normalized - 0.8333).abs() < 0.01,
            "expected ~0.833, got {normalized}"
        );
        assert!(normalized < 1.0, "normalized boost should be < 1.0");

        // Total relevance = 0.0 (base) + 0.833 (boost) < 2.0
        let rel = pruner.relevance(0, 5, &[]);
        assert!(rel < 2.0, "total relevance should stay bounded");
    }

    #[test]
    fn test_multi_token_context_tracking() {
        let trie = make_test_trie();
        let pruner = PhraseBoostPruner::new(ZeroPruner, trie, 5.0);

        // After seeing token 5, tokens 10 and 20 should be boosted
        let rel_10_after_hello = pruner.relevance(1, 10, &[5]);
        assert!(
            rel_10_after_hello > 0.0,
            "token 10 ('world') should be boosted after token 5"
        );

        let rel_20_after_hello = pruner.relevance(1, 20, &[5]);
        assert!(
            rel_20_after_hello > 0.0,
            "token 20 ('there') should be boosted after token 5"
        );

        // Token 30 IS boosted even after token 5, because root is always active
        // and the standalone phrase [30] starts from root.
        let rel_30_after_hello = pruner.relevance(1, 30, &[5]);
        assert!(
            rel_30_after_hello > 0.0,
            "token 30 (standalone phrase) is boostable from root, which is always active"
        );
    }

    #[test]
    fn test_default_boost_score() {
        let trie = make_test_trie();
        let pruner = PhraseBoostPruner::with_default_boost(ZeroPruner, trie);
        let expected: f32 = 5.0 / 6.0;
        assert!(
            (pruner.normalized_boost() - expected).abs() < 0.01,
            "default boost should be 5/6, got {}",
            pruner.normalized_boost()
        );
    }

    #[test]
    fn test_no_boost_for_unrelated_context() {
        let trie = make_test_trie();
        let pruner = PhraseBoostPruner::new(ZeroPruner, trie, 5.0);

        // After tokens [99, 98, 97] (unrelated), no phrase continues
        let rel = pruner.relevance(3, 10, &[99, 98, 97]);
        assert_eq!(rel, 0.0, "unrelated context should get no boost");
    }

    #[test]
    fn test_inner_pruner_preserved() {
        // Use a pruner that returns 0.5 for everything
        struct HalfPruner;
        impl ScreeningPruner for HalfPruner {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
                0.5
            }
        }

        let trie = make_test_trie();
        let pruner = PhraseBoostPruner::new(HalfPruner, trie, 5.0);

        // Boosted token: 0.5 (inner) + normalized_boost
        let boosted_rel = pruner.relevance(0, 5, &[]);
        assert!(
            boosted_rel > 0.5,
            "boosted token should exceed inner baseline"
        );

        // Non-boosted token: pure inner (0.5)
        let plain_rel = pruner.relevance(0, 99, &[]);
        assert_eq!(plain_rel, 0.5, "non-boosted token should be pure inner");
    }
}
