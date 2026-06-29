//! Reward-Weighted Branch Memorization — Plan 209, Phase 3 (T3).
//!
//! When compilation succeeds, boost branches that led to success.
//! Uses blake3 for deterministic pattern hashing of (prompt_type, path).
//! Wraps any inner `ConstraintPruner` — boost is exposed via `get_boost()`
//! for external use by `ScreeningPruner` or sampling logic.
//!
//! # Architecture
//!
//! ```text
//! DDTree → RewardMemPruner → inner.is_valid()  (unchanged)
//!                    ↓
//!              reward_path(path, outcome)         (post-compilation)
//!                    ↓
//!              PatternHasher → blake3 → HashMap<[u8;32], score>
//!                    ↓
//!              get_boost(depth, token, parents)    (pre-inference)
//! ```
//!
//! Zero cost on miss path (no pattern entry → returns 0.0).
//! Feature-gated behind `reward_mem`.

use std::collections::HashMap;

use blake3::Hasher;

use katgpt_speculative::ConstraintPruner;

// ── CompileOutcome ─────────────────────────────────────────────────

/// Result of a compilation attempt against a drafted path.
///
/// Used to drive reward signals: success boosts the path pattern,
/// errors penalize it.
#[derive(Clone, Debug)]
#[repr(u8)]
pub enum CompileOutcome {
    /// Compilation succeeded — positive reward signal.
    Success,
    /// Compilation failed — negative reward signal with error message.
    Error(String),
}

// ── PatternHasher ──────────────────────────────────────────────────

/// Deterministic, zero-alloc blake3 hasher for branch patterns.
///
/// Hashes `(prompt_type, path_pattern)` into a fixed 32-byte key
/// for lookup in the reward table. Stack-only: no heap allocation.
pub struct PatternHasher;

impl PatternHasher {
    /// blake3 hash of `(prompt_type, path_pattern)` — stack-only, zero-alloc.
    pub fn hash(prompt_type: &str, path: &[usize]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(prompt_type.as_bytes());
        for &idx in path {
            hasher.update(&idx.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

// ── RewardMemPruner ────────────────────────────────────────────────

/// Reward-weighted branch memorization pruner.
///
/// Wraps any `ConstraintPruner` and maintains a score table keyed by
/// blake3-hashed path patterns. After compilation, call `reward_path()`
/// to record success/failure. During inference, call `get_boost()` to
/// retrieve the accumulated score for a partial path prefix.
///
/// The inner pruner is never modified — boost scores are consumed
/// externally (e.g., by `ScreeningPruner` or sampling).
#[cfg(feature = "reward_mem")]
pub struct RewardMemPruner<P: ConstraintPruner> {
    inner: P,
    /// Rewarded patterns: blake3 hash → accumulated score.
    rewarded_patterns: HashMap<[u8; 32], f32>,
    /// Current prompt type for pattern matching.
    current_prompt_type: String,
}

#[cfg(feature = "reward_mem")]
impl<P: ConstraintPruner> RewardMemPruner<P> {
    /// Create a new `RewardMemPruner` wrapping the given inner pruner.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            rewarded_patterns: HashMap::new(),
            current_prompt_type: String::new(),
        }
    }

    /// Set the current prompt type before inference.
    ///
    /// This is used as a salt in the pattern hash so that patterns from
    /// different prompt types don't collide.
    pub fn set_prompt_type(&mut self, prompt_type: &str) {
        self.current_prompt_type = prompt_type.to_owned();
    }

    /// Record a compilation outcome for a full path.
    ///
    /// - `CompileOutcome::Success` → reward = 1.0
    /// - `CompileOutcome::Error(_)` → reward = -0.5
    ///
    /// Updates the pattern score using exponential moving average:
    /// `new_score = old_score + lr * (reward - old_score)` with lr = 0.1
    pub fn reward_path(&mut self, path: &[usize], outcome: &CompileOutcome) {
        let reward = match outcome {
            CompileOutcome::Success => 1.0,
            CompileOutcome::Error(_) => -0.5,
        };

        let key = PatternHasher::hash(&self.current_prompt_type, path);
        let lr = 0.1;
        let old_score = self.rewarded_patterns.get(&key).copied().unwrap_or(0.0);
        let new_score = old_score + lr * (reward - old_score);
        self.rewarded_patterns.insert(key, new_score);
    }

    /// Look up the boost score for a partial path prefix at the given depth.
    ///
    /// Constructs the key from `(current_prompt_type, parents + [token_idx])`
    /// and returns the accumulated reward score. Returns `0.0` if no
    /// pattern entry exists (miss path — zero overhead).
    pub fn get_boost(&self, depth: usize, token_idx: usize, parents: &[usize]) -> f32 {
        // Build the full path: parents[..depth] + token_idx
        let mut full_path = Vec::with_capacity(depth + 1);
        let parent_slice = if depth <= parents.len() {
            &parents[..depth]
        } else {
            parents
        };
        full_path.extend_from_slice(parent_slice);
        full_path.push(token_idx);

        let key = PatternHasher::hash(&self.current_prompt_type, &full_path);
        self.rewarded_patterns.get(&key).copied().unwrap_or(0.0)
    }

    /// Reference to the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable reference to the inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Number of rewarded patterns stored.
    pub fn pattern_count(&self) -> usize {
        self.rewarded_patterns.len()
    }

    /// Clear all rewarded patterns and prompt type.
    pub fn reset(&mut self) {
        self.rewarded_patterns.clear();
        self.current_prompt_type.clear();
    }
}

// ── ConstraintPruner delegation ────────────────────────────────────

#[cfg(feature = "reward_mem")]
impl<P: ConstraintPruner> ConstraintPruner for RewardMemPruner<P> {
    #[inline]
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        self.inner.is_valid(depth, token_idx, parent_tokens)
    }

    #[inline]
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        self.inner
            .batch_is_valid(depth, candidates, parent_tokens, results);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::collapsible_if, clippy::collapsible_match, clippy::len_zero)]
mod tests {
    use super::*;

    /// Trivial inner pruner that accepts everything.
    struct AcceptAll;

    impl ConstraintPruner for AcceptAll {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            true
        }

        fn batch_is_valid(
            &self,
            _depth: usize,
            _candidates: &[usize],
            _parent_tokens: &[usize],
            results: &mut [bool],
        ) {
            results.fill(true);
        }
    }

    #[test]
    fn test_success_propagates_positive_reward() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("rust_fn");

        let path = vec![10, 20, 30];
        pruner.reward_path(&path, &CompileOutcome::Success);

        // Score should be positive: 0.0 + 0.1 * (1.0 - 0.0) = 0.1
        let boost = pruner.get_boost(3, 30, &[10, 20]);
        assert!(
            boost > 0.0,
            "expected positive boost after success, got {boost}"
        );
        let expected = 0.1_f32;
        assert!(
            (boost - expected).abs() < 1e-6,
            "expected ~{expected}, got {boost}",
        );
    }

    #[test]
    fn test_error_propagates_negative_reward() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("rust_fn");

        let path = vec![5, 15];
        pruner.reward_path(&path, &CompileOutcome::Error("missing semicolon".into()));

        // Score should be negative: 0.0 + 0.1 * (-0.5 - 0.0) = -0.05
        let boost = pruner.get_boost(2, 15, &[5]);
        assert!(
            boost < 0.0,
            "expected negative boost after error, got {boost}"
        );
        let expected = -0.05_f32;
        assert!(
            (boost - expected).abs() < 1e-6,
            "expected ~{expected}, got {boost}",
        );
    }

    #[test]
    fn test_pattern_lookup_retrieves_rewarded_branches() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("python_class");

        // Reward the same path multiple times — EMA should converge upward
        let path = vec![1, 2, 3];
        for _ in 0..10 {
            pruner.reward_path(&path, &CompileOutcome::Success);
        }

        let boost = pruner.get_boost(3, 3, &[1, 2]);
        // After 10 successes with lr=0.1, score should approach 1.0
        assert!(
            boost > 0.6,
            "expected boost to converge toward 1.0 after 10 successes, got {boost}",
        );
    }

    #[test]
    fn test_miss_path_returns_zero_boost() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("rust_fn");

        // Reward one path
        pruner.reward_path(&[10, 20], &CompileOutcome::Success);

        // Lookup a completely different path — should be 0.0
        let boost = pruner.get_boost(2, 99, &[42]);
        assert_eq!(boost, 0.0, "miss path should return 0.0, got {boost}");
    }

    #[test]
    fn test_no_prompt_type_set_miss_path_returns_zero() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        // Don't set prompt type — reward under empty string

        let path = vec![1, 2];
        pruner.reward_path(&path, &CompileOutcome::Success);

        // Different prompt type should not see the reward
        pruner.set_prompt_type("different");
        let boost = pruner.get_boost(2, 2, &[1]);
        assert_eq!(boost, 0.0, "different prompt type should miss, got {boost}");
    }

    #[test]
    fn test_blake3_hash_deterministic() {
        let h1 = PatternHasher::hash("rust_fn", &[1, 2, 3]);
        let h2 = PatternHasher::hash("rust_fn", &[1, 2, 3]);
        assert_eq!(h1, h2, "same input must produce same hash");
    }

    #[test]
    fn test_blake3_hash_differs_for_different_paths() {
        let h1 = PatternHasher::hash("rust_fn", &[1, 2, 3]);
        let h2 = PatternHasher::hash("rust_fn", &[3, 2, 1]);
        assert_ne!(h1, h2, "different paths must produce different hashes");
    }

    #[test]
    fn test_blake3_hash_differs_for_different_prompt_types() {
        let h1 = PatternHasher::hash("rust_fn", &[1, 2, 3]);
        let h2 = PatternHasher::hash("python_class", &[1, 2, 3]);
        assert_ne!(
            h1, h2,
            "different prompt types must produce different hashes"
        );
    }

    #[test]
    fn test_constraint_pruner_delegates_to_inner() {
        let pruner = RewardMemPruner::new(AcceptAll);
        assert!(pruner.is_valid(0, 42, &[]));
        assert!(pruner.is_valid(3, 10, &[1, 2, 3]));
    }

    #[test]
    fn test_batch_is_valid_delegates() {
        let pruner = RewardMemPruner::new(AcceptAll);
        let mut results = [false; 4];
        pruner.batch_is_valid(1, &[10, 20, 30, 40], &[5], &mut results);
        assert_eq!(results, [true, true, true, true]);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("test");
        pruner.reward_path(&[1], &CompileOutcome::Success);
        assert_eq!(pruner.pattern_count(), 1);

        pruner.reset();
        assert_eq!(pruner.pattern_count(), 0);
        let boost = pruner.get_boost(1, 1, &[]);
        assert_eq!(boost, 0.0, "reset should clear rewards");
    }

    // ── Integration: Before/After Accuracy (Plan 209, T3.6) ──────────

    #[test]
    fn test_reward_mem_improves_accuracy_over_cycles() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("rust_fn");

        // "Good" paths that should compile, "bad" paths that shouldn't.
        let good_paths: &[&[usize]] = &[&[0, 1, 2], &[0, 3, 4], &[0, 5, 6]];
        let bad_paths: &[&[usize]] = &[&[1, 9, 9], &[2, 8, 7], &[3, 0, 0]];

        // Phase 1: Before — all boosts are zero (no history)
        let before_good_avg: f32 = good_paths
            .iter()
            .map(|p| pruner.get_boost(p.len(), p[p.len() - 1], &p[..p.len() - 1]))
            .sum::<f32>()
            / good_paths.len() as f32;
        let before_bad_avg: f32 = bad_paths
            .iter()
            .map(|p| pruner.get_boost(p.len(), p[p.len() - 1], &p[..p.len() - 1]))
            .sum::<f32>()
            / bad_paths.len() as f32;

        assert_eq!(before_good_avg, 0.0, "no history → zero boost");
        assert_eq!(before_bad_avg, 0.0, "no history → zero boost");

        // Phase 2: Warm-up — feed compilation outcomes
        for _ in 0..20 {
            for &path in good_paths {
                pruner.reward_path(path, &CompileOutcome::Success);
            }
            for &path in bad_paths {
                pruner.reward_path(path, &CompileOutcome::Error("type mismatch".into()));
            }
        }

        // Phase 3: After — good paths have positive boost, bad paths negative
        let after_good_avg: f32 = good_paths
            .iter()
            .map(|p| pruner.get_boost(p.len(), p[p.len() - 1], &p[..p.len() - 1]))
            .sum::<f32>()
            / good_paths.len() as f32;
        let after_bad_avg: f32 = bad_paths
            .iter()
            .map(|p| pruner.get_boost(p.len(), p[p.len() - 1], &p[..p.len() - 1]))
            .sum::<f32>()
            / bad_paths.len() as f32;

        assert!(
            after_good_avg > before_good_avg,
            "rewarded paths should improve: before={before_good_avg} after={after_good_avg}"
        );
        assert!(
            after_bad_avg < before_bad_avg,
            "penalized paths should degrade: before={before_bad_avg} after={after_bad_avg}"
        );
        assert!(
            after_good_avg - after_bad_avg > 0.5,
            "good/bad separation should be significant: good={after_good_avg} bad={after_bad_avg}"
        );
    }

    // ── GOAT Proof: Reward Propagation ≥10% Gain (Plan 209, T5.4) ──────

    #[test]
    fn goat_reward_accuracy_gain() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("test_prompt");

        let good_path: &[usize] = &[0, 3, 5];
        let bad_path: &[usize] = &[0, 4, 8];

        // Phase 1: Baseline — no reward history → both zero
        let baseline_good = pruner.get_boost(3, 5, &[0, 3]);
        let baseline_bad = pruner.get_boost(3, 8, &[0, 4]);
        assert_eq!(baseline_good, 0.0);
        assert_eq!(baseline_bad, 0.0);
        let baseline_separation = baseline_good - baseline_bad;

        // Phase 2: Warm-up — 50 compilations each
        for _ in 0..50 {
            pruner.reward_path(good_path, &CompileOutcome::Success);
            pruner.reward_path(bad_path, &CompileOutcome::Error("fail".into()));
        }

        // Phase 3: After — good boosted, bad penalized
        let after_good = pruner.get_boost(3, 5, &[0, 3]);
        let after_bad = pruner.get_boost(3, 8, &[0, 4]);
        let after_separation = after_good - after_bad;

        // Gain: separation improvement from reward memory
        let gain = after_separation - baseline_separation;

        assert!(
            gain >= 0.1,
            "reward propagation gain {:.3} < 0.1 (good={:.3}, bad={:.3})",
            gain,
            after_good,
            after_bad
        );
        assert!(
            after_good > 0.9,
            "good path should converge toward 1.0, got {after_good}"
        );
        assert!(
            after_bad < -0.4,
            "bad path should converge toward -0.5, got {after_bad}"
        );
    }

    #[test]
    fn test_ema_convergence() {
        let mut pruner = RewardMemPruner::new(AcceptAll);
        pruner.set_prompt_type("test");

        let path = [5];
        // 50 successes → should converge very close to 1.0
        for _ in 0..50 {
            pruner.reward_path(&path, &CompileOutcome::Success);
        }
        let boost = pruner.get_boost(1, 5, &[]);
        assert!(
            boost > 0.99,
            "50 successes should converge to ~1.0, got {boost}",
        );

        // Now 50 errors → should converge toward -0.5
        for _ in 0..50 {
            pruner.reward_path(&path, &CompileOutcome::Error("err".into()));
        }
        let boost = pruner.get_boost(1, 5, &[]);
        assert!(
            boost < -0.4,
            "50 errors after 50 successes should converge to ~-0.5, got {boost}",
        );
    }
}
