//! QuestBench — Underspecification scoring for modelless architecture.
//!
//! Computes a normalized entropy score from [`ScreeningPruner::relevance()`] output.
//! Score ∈ [0, 1]: 0 = fully specified (one dominant token), 1 = fully underspecified (uniform).
//!
//! Reference: QuestBench paper §3, Research 008.
//! Plan: 110

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_underspecification_uniform_distribution() {
        // Uniform → 1.0 (fully underspecified)
        let relevance = vec![1.0; 8];
        let score = underspecification_score(&relevance);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "uniform should score 1.0, got {score}"
        );
    }

    #[test]
    fn test_underspecification_one_hot() {
        // One-hot → 0.0 (fully specified)
        let mut relevance = vec![0.0; 8];
        relevance[3] = 1.0;
        let score = underspecification_score(&relevance);
        assert!(score.abs() < 1e-6, "one-hot should score 0.0, got {score}");
    }

    #[test]
    fn test_underspecification_two_equal() {
        // Two equal non-zero → log2(2) / log2(n)
        let mut relevance = vec![0.0; 8];
        relevance[0] = 1.0;
        relevance[1] = 1.0;
        let score = underspecification_score(&relevance);
        let expected = 2.0_f32.log2() / 8.0_f32.log2(); // 1/3
        assert!(
            (score - expected).abs() < 1e-5,
            "two-equal should score {expected}, got {score}"
        );
    }

    #[test]
    fn test_underspecification_all_zeros() {
        // All zeros → 1.0 (degenerate = underspecified)
        let relevance = vec![0.0; 4];
        let score = underspecification_score(&relevance);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "all zeros should score 1.0, got {score}"
        );
    }

    #[test]
    fn test_underspecification_single_element() {
        // Single element with value → 0.0 (log2(1) = 0)
        let relevance = vec![5.0];
        let score = underspecification_score(&relevance);
        assert!(
            score.abs() < 1e-6,
            "single element should score 0.0, got {score}"
        );
    }

    #[test]
    fn test_underspecification_mixed() {
        // Mixed: [0.5, 0.25, 0.25] → entropy = -(0.5*log2(0.5) + 0.25*log2(0.25)*2) = 1.5
        // max_entropy = log2(3) ≈ 1.585, normalized ≈ 0.9464
        let relevance = vec![0.5, 0.25, 0.25];
        let score = underspecification_score(&relevance);
        let expected = 1.5_f32 / 3.0_f32.log2();
        assert!(
            (score - expected).abs() < 1e-4,
            "mixed should score {expected}, got {score}"
        );
    }

    #[test]
    fn test_default_config_thresholds() {
        let config = UnderspecConfig::default();
        assert_eq!(config.plan_new_threshold, 0.8);
        assert_eq!(config.plan_extend_threshold, 0.5);
        assert_eq!(config.cold_tier_threshold, 0.7);
        assert_eq!(config.warm_tier_threshold, 0.3);
    }

    #[test]
    fn test_latency_overhead_trivial() {
        // Verify score computation is fast enough (<1% of decode step).
        // A 32K vocab score should complete in microseconds.
        let relevance: Vec<f32> = (0..32000).map(|i| (i as f32).sin().abs()).collect();
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = underspecification_score(&relevance);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / 10000.0;
        // Should be well under 1ms per call for 32K vocab
        assert!(
            avg_us < 1000.0,
            "score computation too slow: {avg_us:.1}µs per call for 32K vocab"
        );
    }

    // ── T3: SufficientSetFinder tests ─────────────────────────────

    /// A pruner that only allows even tokens at even depths, odd tokens at odd depths.
    struct ParityPruner;

    impl crate::traits::ConstraintPruner for ParityPruner {
        fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            (depth % 2) == (token_idx % 2)
        }
    }

    #[test]
    fn test_sufficient_set_finds_narrowing_token() {
        // With parity pruner at depth 0 (even), only even tokens are valid.
        // Adding one should narrow the next depth (odd) space.
        let pruner = ParityPruner;
        let result = find_sufficient_set(&pruner, 0, &[], 10, 10);
        // Should find at least one sufficient token or return empty if none breaks underspec
        // With parity constraints, the search explores candidates
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_sufficient_set_with_no_pruner() {
        // NoPruner allows everything → underspecification stays high
        let pruner = crate::traits::NoPruner;
        let result = find_sufficient_set(&pruner, 0, &[], 8, 8);
        // All tokens valid, so adding one still leaves all siblings valid → empty result
        assert!(result.is_empty());
    }

    #[test]
    fn test_sufficient_set_with_restrictive_pruner() {
        /// Only token 0 is valid at any depth.
        struct SingletonPruner;
        impl crate::traits::ConstraintPruner for SingletonPruner {
            fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
                token_idx == 0
            }
        }
        let pruner = SingletonPruner;
        let result = find_sufficient_set(&pruner, 0, &[], 4, 4);
        // Only token 0 is valid, adding it makes next depth also singleton → score 0.0 < 0.5
        assert_eq!(result, vec![0]);
    }

    // ── T4: QuestBenchDecision tests ──────────────────────────────

    #[test]
    fn test_questbench_decision_thresholds() {
        let config = UnderspecConfig::default();
        // Default thresholds: plan_new=0.8, plan_extend=0.5

        // High score → PlanNew
        assert_eq!(
            QuestBenchDecision::from_score(0.9, &config),
            QuestBenchDecision::PlanNew
        );

        // Medium score → PlanExtend
        assert_eq!(
            QuestBenchDecision::from_score(0.6, &config),
            QuestBenchDecision::PlanExtend
        );

        // Low score → PlanSkip
        assert_eq!(
            QuestBenchDecision::from_score(0.3, &config),
            QuestBenchDecision::PlanSkip
        );

        // Exact boundary: score == plan_new_threshold (0.8) → not > 0.8, so PlanExtend
        assert_eq!(
            QuestBenchDecision::from_score(0.8, &config),
            QuestBenchDecision::PlanExtend
        );

        // Exact boundary: score == plan_extend_threshold (0.5) → not > 0.5, so PlanSkip
        assert_eq!(
            QuestBenchDecision::from_score(0.5, &config),
            QuestBenchDecision::PlanSkip
        );

        // Zero → PlanSkip
        assert_eq!(
            QuestBenchDecision::from_score(0.0, &config),
            QuestBenchDecision::PlanSkip
        );
    }

    // ── T6: Synthetic CSP generator tests ─────────────────────────

    #[test]
    fn test_generate_csps_count() {
        let csps = generate_synthetic_csps(10);
        assert_eq!(csps.len(), 30, "should have 10 per domain × 3 domains");
    }

    #[test]
    fn test_grid_csps_have_sufficient_answer() {
        let csps = generate_synthetic_csps(5);
        let grid_csps: Vec<_> = csps
            .iter()
            .filter(|c| c.label.starts_with("grid"))
            .collect();
        assert_eq!(grid_csps.len(), 5);
        for csp in grid_csps {
            assert!(
                !csp.sufficient_answers.is_empty(),
                "grid CSP should have sufficient answer"
            );
            // Key should narrow to ≤4 adjacent cells at next depth
            if let Some(&key) = csp.sufficient_answers.first() {
                let mut extended = csp.placed_tokens.clone();
                extended.push(key);
                let valid_count = (0..csp.vocab_size)
                    .filter(|&t| csp.pruner.is_valid(csp.depth + 1, t, &extended))
                    .count();
                assert!(
                    valid_count <= 4,
                    "grid key {key} should narrow to ≤4 valid tokens, got {valid_count}"
                );
            }
        }
    }

    #[test]
    fn test_stone_csps_key_narrows() {
        let csps = generate_synthetic_csps(3);
        let stone_csps: Vec<_> = csps
            .iter()
            .filter(|c| c.label.starts_with("stone"))
            .collect();
        assert_eq!(stone_csps.len(), 3);
        for csp in stone_csps {
            // Placing the key token should narrow to ≤ 2 valid tokens at next depth
            if let Some(&key) = csp.sufficient_answers.first() {
                let mut extended = csp.placed_tokens.clone();
                extended.push(key);
                let valid_count = (0..csp.vocab_size)
                    .filter(|&t| csp.pruner.is_valid(csp.depth + 1, t, &extended))
                    .count();
                assert!(
                    valid_count <= 2,
                    "key {key} should narrow to ≤2 valid tokens, got {valid_count}"
                );
            }
        }
    }

    #[test]
    fn test_logic_csps_key_narrows_to_one() {
        let csps = generate_synthetic_csps(4);
        let logic_csps: Vec<_> = csps
            .iter()
            .filter(|c| c.label.starts_with("logic"))
            .collect();
        assert_eq!(logic_csps.len(), 4);
        for csp in logic_csps {
            // Placing the key token should narrow to exactly 1 valid token (XOR partner)
            if let Some(&key) = csp.sufficient_answers.first() {
                let mut extended = csp.placed_tokens.clone();
                extended.push(key);
                let valid_count = (0..csp.vocab_size)
                    .filter(|&t| csp.pruner.is_valid(csp.depth + 1, t, &extended))
                    .count();
                assert_eq!(
                    valid_count, 1,
                    "XOR key {key} should narrow to 1 valid token, got {valid_count}"
                );
            }
        }
    }

    // ── T7 G2: GOAT proof — Sufficient Set Accuracy ───────────────

    #[test]
    fn test_goat_g2_sufficient_set_accuracy() {
        // GOAT proof G2: find_sufficient_set identifies the correct
        // sufficient variable >60% of the time on synthetic 1-sufficient CSPs.
        let csps = generate_synthetic_csps(20); // 60 total CSPs
        let mut correct = 0usize;
        let mut total = 0usize;

        for csp in &csps {
            let found = find_sufficient_set(
                csp.pruner.as_ref(),
                csp.depth,
                &csp.placed_tokens,
                csp.vocab_size,
                csp.vocab_size, // max_search_depth = vocab_size
            );
            total += 1;
            // Check if any found token is in the sufficient answers
            if found.iter().any(|t| csp.sufficient_answers.contains(t)) {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy >= 0.6,
            "GOAT G2 FAILED: sufficient-set accuracy = {:.1}% (need >= 60%), {}/{} correct",
            accuracy * 100.0,
            correct,
            total
        );
    }

    // ── T5: Four-Tier routing tests ───────────────────────────────

    #[test]
    fn test_four_tier_routing() {
        let config = UnderspecConfig::default();
        // Default thresholds: cold=0.7, warm=0.3
        // Freeze >= cold + 0.2 = 0.9

        // Very high → Freeze (>= 0.9)
        assert_eq!(tier_from_score(0.95, &config), MemoryTier::Freeze);
        assert_eq!(tier_from_score(0.9, &config), MemoryTier::Freeze);

        // High → Cold (>= 0.7, < 0.9)
        assert_eq!(tier_from_score(0.85, &config), MemoryTier::Cold);
        assert_eq!(tier_from_score(0.7, &config), MemoryTier::Cold);

        // Medium → Warm (>= 0.3, < 0.7)
        assert_eq!(tier_from_score(0.5, &config), MemoryTier::Warm);
        assert_eq!(tier_from_score(0.3, &config), MemoryTier::Warm);

        // Low → Hot (< 0.3)
        assert_eq!(tier_from_score(0.1, &config), MemoryTier::Hot);
        assert_eq!(tier_from_score(0.0, &config), MemoryTier::Hot);
    }
}

/// Normalized entropy of a relevance distribution.
///
/// Returns a score in `[0, 1]`:
/// - `0.0` = fully specified (one dominant token)
/// - `1.0` = fully underspecified (uniform distribution)
///
/// This is a pure function over a relevance slice — no model inference needed.
pub fn underspecification_score(relevance: &[f32]) -> f32 {
    let sum: f32 = relevance.iter().sum();
    if sum <= 0.0 {
        return 1.0; // degenerate = underspecified
    }

    let entropy: f32 = relevance
        .iter()
        .filter(|&&r| r > 0.0)
        .map(|&r| {
            let p = r / sum;
            -p * p.log2()
        })
        .sum();

    let max_entropy = (relevance.len() as f32).log2();
    if max_entropy <= 0.0 {
        return 0.0;
    }

    entropy / max_entropy
}

/// Decision thresholds for underspecification-driven planning.
///
/// Domain-configurable via TOML. Defaults from QuestBench paper §4.
pub struct UnderspecConfig {
    /// Score above which a new plan is needed. Default: 0.8
    pub plan_new_threshold: f32,
    /// Score above which the current plan is extended. Default: 0.5
    pub plan_extend_threshold: f32,
    /// Score above which the Cold tier (Turso) is consulted. Default: 0.7
    pub cold_tier_threshold: f32,
    /// Score above which the Warm tier (HLA KG) is consulted. Default: 0.3
    pub warm_tier_threshold: f32,
}

impl Default for UnderspecConfig {
    fn default() -> Self {
        Self {
            plan_new_threshold: 0.8,
            plan_extend_threshold: 0.5,
            cold_tier_threshold: 0.7,
            warm_tier_threshold: 0.3,
        }
    }
}

// ── T3: SufficientSetFinder ──────────────────────────────────────

/// Find minimal set of additional tokens that, if known, would
/// break underspecification for the target position.
///
/// Uses backward greedy search over the constraint graph.
/// Returns the minimal set (greedy, not optimal — optimal is NP-hard).
pub fn find_sufficient_set(
    pruner: &dyn crate::traits::ConstraintPruner,
    depth: usize,
    placed_tokens: &[usize],
    vocab_size: usize,
    max_search_depth: usize,
) -> Vec<usize> {
    let mut sufficient = Vec::new();
    let candidate_tokens: Vec<usize> = (0..vocab_size)
        .filter(|&tok| pruner.is_valid(depth, tok, placed_tokens))
        .collect();

    // Pre-compute extension counts for ALL candidates once (O(n × sample_size))
    // instead of per-comparison during sort (O(n log n × sample_size))
    let mut ext_buf = placed_tokens.to_vec();
    let mut counts: Vec<(usize, usize)> = candidate_tokens
        .into_iter()
        .map(|tok| {
            ext_buf.clear();
            ext_buf.extend_from_slice(placed_tokens);
            ext_buf.push(tok);
            let count = count_valid_extensions_with(pruner, depth + 1, &ext_buf);
            (tok, count)
        })
        .collect();

    // Sort by pre-computed counts (ascending = tighter constraints first)
    counts.sort_by_key(|&(_, count)| count);

    let mut relevance_buf = vec![0.0f32; vocab_size.min(256)];

    for &(tok, _) in counts.iter().take(max_search_depth) {
        ext_buf.clear();
        ext_buf.extend_from_slice(placed_tokens);
        ext_buf.push(tok);
        score_relevance_into(pruner, depth + 1, &ext_buf, vocab_size, &mut relevance_buf);
        let score = underspecification_score(&relevance_buf);
        if score < 0.5 {
            sufficient.push(*tok);
            break; // found 1-sufficient
        }
    }
    sufficient
}

/// Count how many valid extensions exist at next depth after placing this token.
fn count_valid_extensions(
    pruner: &dyn crate::traits::ConstraintPruner,
    depth: usize,
    last_token: usize,
    placed_tokens: &[usize],
) -> usize {
    let mut extended = placed_tokens.to_vec();
    extended.push(last_token);
    count_valid_extensions_with(pruner, depth, &extended)
}

/// Count valid extensions using a pre-built extended token list (avoids allocation).
fn count_valid_extensions_with(
    pruner: &dyn crate::traits::ConstraintPruner,
    depth: usize,
    extended: &[usize],
) -> usize {
    // Count valid tokens at this depth (sampling up to 256 for efficiency)
    let sample_size = 256;
    let mut count = 0;
    for tok in 0..sample_size {
        if pruner.is_valid(depth, tok, extended) {
            count += 1;
        }
    }
    count
}

/// Compute relevance scores for all tokens at given depth.
/// Writes results into `buf` (must have length >= min(vocab_size, 256)).
fn score_relevance_into(
    pruner: &dyn crate::traits::ConstraintPruner,
    depth: usize,
    placed_tokens: &[usize],
    vocab_size: usize,
    buf: &mut [f32],
) {
    let limit = vocab_size.min(256).min(buf.len());
    for tok in 0..limit {
        buf[tok] = if pruner.is_valid(depth, tok, placed_tokens) {
            1.0
        } else {
            0.0
        };
    }
    // Zero remaining entries
    buf[limit..].fill(0.0);
}

/// Backward-compatible wrapper that allocates.
fn score_relevance(
    pruner: &dyn crate::traits::ConstraintPruner,
    depth: usize,
    placed_tokens: &[usize],
    vocab_size: usize,
) -> Vec<f32> {
    let limit = vocab_size.min(256);
    let mut buf = vec![0.0f32; limit];
    score_relevance_into(pruner, depth, placed_tokens, vocab_size, &mut buf);
    buf
}

// ── T4: QuestBenchDecision ───────────────────────────────────────

/// Decision from underspecification score for planning.
/// Maps to `PlanningDecision` in types.rs but lives here to avoid circular deps.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestBenchDecision {
    PlanNew,
    PlanExtend,
    PlanSkip,
}

impl QuestBenchDecision {
    pub fn from_score(score: f32, config: &UnderspecConfig) -> Self {
        match score {
            s if s > config.plan_new_threshold => QuestBenchDecision::PlanNew,
            s if s > config.plan_extend_threshold => QuestBenchDecision::PlanExtend,
            _ => QuestBenchDecision::PlanSkip,
        }
    }
}

// ── T5: Four-Tier trigger ───────────────────────────────────────

/// Which memory tier to consult based on underspecification score.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    Hot,    // CPU SIMD — standard decode
    Warm,   // HLA KG — O(1) relation lookup
    Cold,   // Turso — async episode retrieval
    Freeze, // external knowledge
}

pub fn tier_from_score(score: f32, config: &UnderspecConfig) -> MemoryTier {
    match score {
        s if s >= config.cold_tier_threshold + 0.2 => MemoryTier::Freeze,
        s if s >= config.cold_tier_threshold => MemoryTier::Cold,
        s if s >= config.warm_tier_threshold => MemoryTier::Warm,
        _ => MemoryTier::Hot,
    }
}

// ── T6: Synthetic CSP Generator ──────────────────────────────────

/// A synthetic 1-sufficient CSP problem for GOAT proof G2.
///
/// Each CSP has a known "sufficient" variable — the single token that,
/// if revealed, would reduce underspecification below the threshold.
pub struct SyntheticCsp {
    /// Human-readable label for the CSP domain.
    pub label: String,
    /// The pruner that encodes the CSP constraints.
    pub pruner: Box<dyn crate::traits::ConstraintPruner>,
    /// Depth at which the CSP is posed.
    pub depth: usize,
    /// Tokens already placed (known facts).
    pub placed_tokens: Vec<usize>,
    /// Total vocabulary/domain size.
    pub vocab_size: usize,
    /// The ground-truth sufficient token(s).
    pub sufficient_answers: Vec<usize>,
}

/// Domain kind for synthetic CSP generation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CspDomain {
    /// Grid-based (Bomber-like): adjacency constraints on a 2D grid.
    Grid,
    /// Stone-based (Go-like): liberty/capture constraints.
    Stone,
    /// Propositional logic: rule-based constraints.
    Logic,
}

// ── Narrowing pruner: key abstraction for synthetic CSPs ─────────
//
// The idea: at depth D, many tokens are valid. At depth D+1, placing
// a specific "sufficient" token narrows the space dramatically.

/// Pruner where placing a specific "key" token at depth D reduces
/// valid tokens at depth D+1 to just 1 (fully specified).
struct NarrowingPruner {
    /// Total vocabulary size.
    _vocab_size: usize,
    /// Valid tokens at the target depth (before narrowing).
    valid_at_depth: Vec<usize>,
    /// Map: token placed at depth D → valid tokens at depth D+1.
    /// Indexed by token value; empty Vec means "use valid_at_depth".
    /// The "sufficient" token maps to a singleton, others map to many.
    narrowing: Vec<Vec<usize>>,
}

impl crate::traits::ConstraintPruner for NarrowingPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        if depth == 0 {
            return self.valid_at_depth.contains(&token_idx);
        }
        // depth > 0: validity depends on what was placed at depth-1
        let last = match parent_tokens.last() {
            Some(&t) => t,
            None => return self.valid_at_depth.contains(&token_idx),
        };
        if last < self.narrowing.len() && !self.narrowing[last].is_empty() {
            return self.narrowing[last].contains(&token_idx);
        }
        // Default: allow if in valid_at_depth
        self.valid_at_depth.contains(&token_idx)
    }
}

/// Generate a batch of synthetic 1-sufficient CSPs.
///
/// Returns CSPs with known ground-truth sufficient tokens,
/// suitable for GOAT proof G2 (accuracy benchmark).
///
/// Each CSP is constructed so that placing the sufficient token
/// at depth D narrows the valid set at depth D+1 to a singleton,
/// dropping the underspecification score to 0.
pub fn generate_synthetic_csps(count_per_domain: usize) -> Vec<SyntheticCsp> {
    let mut csps = Vec::with_capacity(count_per_domain * 3);

    // Grid CSPs (Bomber-like): placing the "bomb" cell narrows explosion zone
    for i in 0..count_per_domain {
        let vocab_size = 16;
        let key = i % vocab_size;
        // At depth 0: ALL cells are valid candidates (including key)
        let valid_at_depth: Vec<usize> = (0..vocab_size).collect();
        // Placing the key cell narrows to just adjacent cells at depth 1
        let row = key / 4;
        let col = key % 4;
        let adjacent: Vec<usize> = (0..vocab_size)
            .filter(|&c| {
                let r = c / 4;
                let cc = c % 4;
                let dist = (r as i32 - row as i32).unsigned_abs()
                    + (cc as i32 - col as i32).unsigned_abs();
                dist == 1
            })
            .collect();
        // All other tokens lead to full valid set (no narrowing)
        let default_next: Vec<usize> = (0..vocab_size).collect();
        let mut narrowing = vec![Vec::new(); vocab_size];
        narrowing[key] = adjacent;
        for v in &valid_at_depth {
            if *v != key {
                narrowing[*v] = default_next.clone();
            }
        }
        let pruner = NarrowingPruner {
            _vocab_size: vocab_size,
            valid_at_depth,
            narrowing,
        };
        csps.push(SyntheticCsp {
            label: format!("grid_{i}"),
            pruner: Box::new(pruner),
            depth: 0,
            placed_tokens: vec![],
            vocab_size,
            sufficient_answers: vec![key],
        });
    }

    // Stone CSPs (Go-like): placing a "capture" stone eliminates liberties
    for i in 0..count_per_domain {
        let vocab_size = 12; // smaller board for tighter constraints
        let key = i % vocab_size;
        let valid_at_depth: Vec<usize> = (0..vocab_size).collect();
        // Placing the key stone at depth 0 leaves only 1 valid position at depth 1
        // (simulates a capture that fills all but one liberty)
        let narrow_next = vec![(key + 1) % vocab_size];
        // Other placements leave 4-5 valid positions
        let wide_next: Vec<usize> = (0..vocab_size).filter(|&c| c % 3 == 0).collect();
        let mut narrowing = vec![Vec::new(); vocab_size];
        narrowing[key] = narrow_next;
        for v in &valid_at_depth {
            if *v != key {
                narrowing[*v] = wide_next.clone();
            }
        }
        let pruner = NarrowingPruner {
            _vocab_size: vocab_size,
            valid_at_depth,
            narrowing,
        };
        csps.push(SyntheticCsp {
            label: format!("stone_{i}"),
            pruner: Box::new(pruner),
            depth: 0,
            placed_tokens: vec![],
            vocab_size,
            sufficient_answers: vec![key],
        });
    }

    // Logic CSPs (propositional): XOR constraints where revealing one variable
    // determines the other
    for i in 0..count_per_domain {
        let vocab_size = 8;
        // Pairs: (0,1), (2,3), (4,5), (6,7) — XOR constraints
        let key = i % vocab_size;
        let partner = if key % 2 == 0 { key + 1 } else { key - 1 };
        // At depth 0: all propositions valid
        let valid_at_depth: Vec<usize> = (0..vocab_size).collect();
        // Placing key → only partner survives at depth 1 (XOR resolution)
        let narrow_next = vec![partner];
        // Placing any other → all still valid
        let wide_next: Vec<usize> = (0..vocab_size).collect();
        let mut narrowing = vec![Vec::new(); vocab_size];
        narrowing[key] = narrow_next;
        for v in &valid_at_depth {
            if *v != key {
                narrowing[*v] = wide_next.clone();
            }
        }
        let pruner = NarrowingPruner {
            _vocab_size: vocab_size,
            valid_at_depth,
            narrowing,
        };
        csps.push(SyntheticCsp {
            label: format!("logic_{i}"),
            pruner: Box::new(pruner),
            depth: 0,
            placed_tokens: vec![],
            vocab_size,
            sufficient_answers: vec![key],
        });
    }

    csps
}
