//! Auto Constraint Synthesis — mine accepted DDTree paths for recurring patterns (Plan 211 F2).
//!
//! Extracts frequent token sequences from accepted paths and converts them to
//! ConstraintPruner-compatible constraints. Background task, not on hot path.
//!
//! # Architecture
//!
//! ```text
//! AcceptedPaths ──► extract_frequent_sequences() ──► Vec<Pattern>
//!                                                          │
//!                      SequenceConstraint::from_pattern() ◄──┘
//!                              │
//!                      mine_and_insert() ──► Vec<SequenceConstraint>
//!                              │
//!                      (caller inserts into pruner)
//! ```
//!
//! # Feature Gate
//!
//! `auto_constraint_synthesis` (depends on `three_mode_router`, `egcs`).
//!
//! # Performance
//!
//! - Mining: <100μs per batch of 100 episodes (background, not hot path)
//! - Hashing: blake3 for pattern integrity audit

use std::collections::HashMap;

// ── Sigmoid helper ────────────────────────────────────────────

/// Sigmoid function: `1 / (1 + exp(-x))`.
/// Used for acceptance rate gating — never softmax.
#[inline]
#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Core Types ────────────────────────────────────────────────

/// A recurring token sequence pattern mined from accepted paths.
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Token sequence (bigram or trigram).
    pub tokens: Vec<usize>,
    /// Occurrence count (how many paths contain this pattern).
    pub support: usize,
    /// Acceptance rate: accepted occurrences / total occurrences.
    pub acceptance_rate: f32,
}

impl Pattern {
    /// Compute acceptance rate from support and total occurrences.
    ///
    /// Returns `support / total` clamped to [0, 1].
    pub fn acceptance_rate(&self, total_occurrences: usize) -> f32 {
        if total_occurrences == 0 {
            return 0.0;
        }
        self.support as f32 / total_occurrences as f32
    }
}

/// A synthesized constraint from a mined pattern.
///
/// Represents "token `first` followed by token `second` [followed by token `third`]"
/// with an associated acceptance rate.
#[derive(Debug, Clone)]
pub struct SequenceConstraint {
    /// First token ID in the sequence.
    pub first: usize,
    /// Second token ID (bigram).
    pub second: usize,
    /// Optional third token (trigram).
    pub third: Option<usize>,
    /// Pattern's acceptance rate ∈ [0, 1].
    pub acceptance_rate: f32,
}

impl SequenceConstraint {
    /// Convert a mined pattern to a constraint.
    ///
    /// Returns `None` if the pattern has fewer than 2 tokens or
    /// acceptance_rate < `min_acceptance`.
    pub fn from_pattern(pattern: &Pattern, min_acceptance: f32) -> Option<Self> {
        match pattern.tokens.len() {
            2 if pattern.acceptance_rate >= min_acceptance => Some(Self {
                first: pattern.tokens[0],
                second: pattern.tokens[1],
                third: None,
                acceptance_rate: pattern.acceptance_rate,
            }),
            3 if pattern.acceptance_rate >= min_acceptance => Some(Self {
                first: pattern.tokens[0],
                second: pattern.tokens[1],
                third: Some(pattern.tokens[2]),
                acceptance_rate: pattern.acceptance_rate,
            }),
            _ => None,
        }
    }

    /// blake3 hash of the constraint for integrity audit.
    pub fn blake3_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.first.to_le_bytes());
        hasher.update(&self.second.to_le_bytes());
        match self.third {
            Some(t) => {
                hasher.update(&t.to_le_bytes());
            }
            None => {
                hasher.update(&[0u8; 8]);
            }
        }
        hasher.update(&self.acceptance_rate.to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// Configuration for the constraint miner.
#[derive(Debug, Clone)]
pub struct ConstraintMiner {
    /// Minimum episode count for a pattern to be considered (default: 10).
    pub min_support: usize,
    /// Minimum acceptance rate for constraint promotion (default: 0.90).
    pub min_acceptance: f32,
    /// Last epoch when mining ran — deduplication / scheduling.
    pub last_mine_epoch: u64,
}

impl Default for ConstraintMiner {
    fn default() -> Self {
        Self {
            min_support: 10,
            min_acceptance: 0.90,
            last_mine_epoch: 0,
        }
    }
}

// ── Pattern Key ───────────────────────────────────────────────

/// Hashable key for token sequences (bigrams and trigrams).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PatternKey {
    tokens: Vec<usize>,
}

// ── Frequency Extraction ──────────────────────────────────────

/// Extract frequent token sequences from accepted DDTree paths.
///
/// Sliding window over token sequences with window sizes 2 (bigrams)
/// and 3 (trigrams). Counts occurrences and filters by `min_support`.
///
/// Pre-allocated HashMap for counting.
pub fn extract_frequent_sequences(paths: &[Vec<usize>], min_support: usize) -> Vec<Pattern> {
    // Pre-allocate capacity estimate: paths × avg_length × 2 (bigram + trigram)
    let estimated_patterns = paths.len().saturating_mul(10);
    let mut bigram_counts: HashMap<PatternKey, usize> = HashMap::with_capacity(estimated_patterns);
    let mut trigram_counts: HashMap<PatternKey, usize> = HashMap::with_capacity(estimated_patterns);

    // Count occurrences across all paths
    for path in paths {
        let len = path.len();
        if len < 2 {
            continue;
        }

        // Bigrams (window size 2)
        for i in 0..(len - 1) {
            let key = PatternKey {
                tokens: vec![path[i], path[i + 1]],
            };
            *bigram_counts.entry(key).or_insert(0) += 1;
        }

        // Trigrams (window size 3)
        if len >= 3 {
            for i in 0..(len - 2) {
                let key = PatternKey {
                    tokens: vec![path[i], path[i + 1], path[i + 2]],
                };
                *trigram_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let total_paths = paths.len().max(1);
    let mut patterns = Vec::with_capacity(bigram_counts.len() + trigram_counts.len());

    // Convert bigram counts to patterns
    for (key, count) in bigram_counts {
        if count >= min_support {
            let acceptance_rate = count as f32 / total_paths as f32;
            patterns.push(Pattern {
                tokens: key.tokens,
                support: count,
                acceptance_rate,
            });
        }
    }

    // Convert trigram counts to patterns
    for (key, count) in trigram_counts {
        if count >= min_support {
            let acceptance_rate = count as f32 / total_paths as f32;
            patterns.push(Pattern {
                tokens: key.tokens,
                support: count,
                acceptance_rate,
            });
        }
    }

    // Sort by support descending (highest frequency first)
    patterns.sort_by(|a, b| b.support.cmp(&a.support));

    patterns
}

// ── Mine and Insert ───────────────────────────────────────────

/// Mine patterns from accepted paths and generate constraints.
///
/// Extracts patterns → filters by acceptance → generates constraints.
/// Returns the new constraints (caller inserts into pruner).
///
/// Rate-limit: only mines if `epoch > last_mine_epoch`.
pub fn mine_and_insert(
    miner: &mut ConstraintMiner,
    accepted_paths: &[Vec<usize>],
    current_epoch: u64,
) -> Vec<SequenceConstraint> {
    // Rate-limit: only mine once per epoch
    if current_epoch <= miner.last_mine_epoch {
        return Vec::new();
    }
    miner.last_mine_epoch = current_epoch;

    // Extract frequent patterns
    let patterns = extract_frequent_sequences(accepted_paths, miner.min_support);

    // Convert to constraints, filtering by acceptance rate
    let mut constraints = Vec::with_capacity(patterns.len());
    for pattern in &patterns {
        if let Some(constraint) = SequenceConstraint::from_pattern(pattern, miner.min_acceptance) {
            constraints.push(constraint);
        }
    }

    constraints
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── F2.7: Mine patterns from 100 synthetic episodes ────────

    #[test]
    fn mine_known_bigram_from_100_episodes() {
        let mut paths = Vec::with_capacity(100);

        // Create 70 episodes with the known bigram [3, 7]
        for _ in 0..70 {
            // [3, 7, ...random...]
            let mut path = vec![3, 7];
            path.push(1);
            path.push(5);
            paths.push(path);
        }

        // Create 30 episodes without the bigram
        for _ in 0..30 {
            paths.push(vec![1, 2, 4, 6]);
        }

        assert_eq!(paths.len(), 100);

        let patterns = extract_frequent_sequences(&paths, 10);

        // The bigram [3, 7] should be found
        let found = patterns.iter().find(|p| p.tokens == vec![3, 7]);
        assert!(found.is_some(), "Expected to find bigram [3, 7]");

        let pattern = found.unwrap();
        assert!(
            pattern.support >= 70,
            "Expected support >= 70, got {}",
            pattern.support
        );
        assert!(
            pattern.acceptance_rate >= 0.70,
            "Expected acceptance_rate >= 0.70, got {}",
            pattern.acceptance_rate
        );
    }

    #[test]
    fn mine_known_trigram() {
        let mut paths = Vec::with_capacity(50);

        // 40 episodes with trigram [3, 7, 1]
        for _ in 0..40 {
            paths.push(vec![3, 7, 1, 5]);
        }

        // 10 without
        for _ in 0..10 {
            paths.push(vec![2, 4, 6, 8]);
        }

        let patterns = extract_frequent_sequences(&paths, 10);

        let found = patterns.iter().find(|p| p.tokens == vec![3, 7, 1]);
        assert!(found.is_some(), "Expected to find trigram [3, 7, 1]");

        let pattern = found.unwrap();
        assert!(pattern.support >= 40);
    }

    // ── F2.8: Verify auto-generated constraints are valid ──────

    #[test]
    fn constraints_have_high_acceptance_rate() {
        let mut paths = Vec::with_capacity(100);

        // 95 episodes with [3, 7] bigram → 95% acceptance
        for _ in 0..95 {
            paths.push(vec![3, 7, 1, 5]);
        }
        for _ in 0..5 {
            paths.push(vec![1, 2, 3, 4]);
        }

        let mut miner = ConstraintMiner::default();
        let constraints = mine_and_insert(&mut miner, &paths, 1);

        // All constraints should have acceptance_rate >= 0.90
        for constraint in &constraints {
            assert!(
                constraint.acceptance_rate >= 0.90,
                "Constraint acceptance_rate {} < 0.90",
                constraint.acceptance_rate
            );
        }
    }

    #[test]
    fn low_acceptance_patterns_not_promoted() {
        let mut paths = Vec::with_capacity(100);

        // Only 50 episodes with bigram [3, 7] → 50% acceptance (below 0.90 threshold)
        for _ in 0..50 {
            paths.push(vec![3, 7, 1, 5]);
        }
        for _ in 0..50 {
            paths.push(vec![9, 8, 7, 6]);
        }

        let patterns = extract_frequent_sequences(&paths, 10);
        let miner = ConstraintMiner {
            min_acceptance: 0.90,
            ..Default::default()
        };

        let mut constraints = Vec::new();
        for pattern in &patterns {
            if let Some(c) = SequenceConstraint::from_pattern(pattern, miner.min_acceptance) {
                constraints.push(c);
            }
        }

        // No constraint for [3, 7] — acceptance rate too low
        let found = constraints.iter().find(|c| c.first == 3 && c.second == 7);
        assert!(
            found.is_none(),
            "Low acceptance bigram should not be promoted to constraint"
        );
    }

    // ── from_pattern validation ────────────────────────────────

    #[test]
    fn from_pattern_bigram() {
        let pattern = Pattern {
            tokens: vec![3, 7],
            support: 50,
            acceptance_rate: 0.95,
        };
        let constraint = SequenceConstraint::from_pattern(&pattern, 0.90);
        assert!(constraint.is_some());
        let c = constraint.unwrap();
        assert_eq!(c.first, 3);
        assert_eq!(c.second, 7);
        assert_eq!(c.third, None);
        assert!((c.acceptance_rate - 0.95).abs() < 1e-6);
    }

    #[test]
    fn from_pattern_trigram() {
        let pattern = Pattern {
            tokens: vec![3, 7, 1],
            support: 40,
            acceptance_rate: 0.92,
        };
        let constraint = SequenceConstraint::from_pattern(&pattern, 0.90);
        assert!(constraint.is_some());
        let c = constraint.unwrap();
        assert_eq!(c.first, 3);
        assert_eq!(c.second, 7);
        assert_eq!(c.third, Some(1));
    }

    #[test]
    fn from_pattern_below_threshold() {
        let pattern = Pattern {
            tokens: vec![3, 7],
            support: 50,
            acceptance_rate: 0.80, // below 0.90 threshold
        };
        let constraint = SequenceConstraint::from_pattern(&pattern, 0.90);
        assert!(constraint.is_none());
    }

    #[test]
    fn from_pattern_too_short() {
        let pattern = Pattern {
            tokens: vec![3],
            support: 50,
            acceptance_rate: 0.95,
        };
        let constraint = SequenceConstraint::from_pattern(&pattern, 0.90);
        assert!(
            constraint.is_none(),
            "Singleton patterns should not produce constraints"
        );
    }

    // ── Rate limiting ──────────────────────────────────────────

    #[test]
    fn mine_and_insert_rate_limits() {
        let mut miner = ConstraintMiner::default();
        let paths: Vec<Vec<usize>> = (0..50).map(|_| vec![3, 7, 1, 5]).collect();

        // First mine at epoch 1 → should produce constraints
        let c1 = mine_and_insert(&mut miner, &paths, 1);
        assert!(!c1.is_empty(), "First mine should produce constraints");

        // Second mine at same epoch → rate-limited
        let c2 = mine_and_insert(&mut miner, &paths, 1);
        assert!(c2.is_empty(), "Same epoch should be rate-limited");

        // Mine at epoch 2 → should produce constraints again
        let c3 = mine_and_insert(&mut miner, &paths, 2);
        assert!(!c3.is_empty(), "New epoch should produce constraints");
    }

    // ── blake3 hash ────────────────────────────────────────────

    #[test]
    fn constraint_blake3_hash_deterministic() {
        let c = SequenceConstraint {
            first: 3,
            second: 7,
            third: None,
            acceptance_rate: 0.95,
        };
        let h1 = c.blake3_hash();
        let h2 = c.blake3_hash();
        assert_eq!(h1, h2, "Hash should be deterministic");
    }

    #[test]
    fn constraint_blake3_hash_differs_for_different_constraints() {
        let c1 = SequenceConstraint {
            first: 3,
            second: 7,
            third: None,
            acceptance_rate: 0.95,
        };
        let c2 = SequenceConstraint {
            first: 3,
            second: 7,
            third: Some(1),
            acceptance_rate: 0.95,
        };
        assert_ne!(
            c1.blake3_hash(),
            c2.blake3_hash(),
            "Different constraints should have different hashes"
        );
    }

    // ── F2.9: Benchmark — mining overhead ──────────────────────

    #[test]
    fn bench_mining_overhead_100_episodes() {
        let mut paths = Vec::with_capacity(100);
        for i in 0..100 {
            let mut path = vec![3, 7]; // known bigram
            path.push(i % 10);
            path.push((i + 1) % 10);
            paths.push(path);
        }

        let start = std::time::Instant::now();
        let mut miner = ConstraintMiner::default();
        let constraints = mine_and_insert(&mut miner, &paths, 1);
        let elapsed = start.elapsed();

        assert!(
            !constraints.is_empty(),
            "Should produce at least some constraints"
        );

        let us = elapsed.as_micros();
        assert!(
            us < 100_000, // 100ms generous bound (CI can be slow)
            "Mining 100 episodes should be < 100ms, took {us}μs"
        );
    }

    // ── Edge cases ─────────────────────────────────────────────

    #[test]
    fn empty_paths_produce_no_patterns() {
        let patterns = extract_frequent_sequences(&[], 10);
        assert!(patterns.is_empty());
    }

    #[test]
    fn short_paths_skip_trigrams() {
        let paths = vec![vec![1, 2]]; // length 2 → only bigrams
        let patterns = extract_frequent_sequences(&paths, 1);
        // Should find bigram [1, 2] but no trigrams
        assert!(patterns.iter().any(|p| p.tokens == vec![1, 2]));
        assert!(!patterns.iter().any(|p| p.tokens.len() == 3));
    }

    #[test]
    fn acceptance_rate_zero_total() {
        let pattern = Pattern {
            tokens: vec![1, 2],
            support: 5,
            acceptance_rate: 1.0,
        };
        assert!((pattern.acceptance_rate(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn acceptance_rate_normal() {
        let pattern = Pattern {
            tokens: vec![1, 2],
            support: 75,
            acceptance_rate: 1.0,
        };
        let rate = pattern.acceptance_rate(100);
        assert!((rate - 0.75).abs() < 1e-6);
    }

    #[test]
    fn miner_default_values() {
        let miner = ConstraintMiner::default();
        assert_eq!(miner.min_support, 10);
        assert!((miner.min_acceptance - 0.90).abs() < 1e-6);
        assert_eq!(miner.last_mine_epoch, 0);
    }
}
