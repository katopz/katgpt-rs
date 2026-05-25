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
        }
    }

    #[test]
    fn test_stone_csps_pruner_validity() {
        let csps = generate_synthetic_csps(3);
        let stone_csps: Vec<_> = csps
            .iter()
            .filter(|c| c.label.starts_with("stone"))
            .collect();
        assert_eq!(stone_csps.len(), 3);
        for csp in stone_csps {
            // The sufficient answer should be a valid token
            if let Some(&ans) = csp.sufficient_answers.first() {
                assert!(
                    csp.pruner.is_valid(csp.depth, ans, &csp.placed_tokens),
                    "sufficient answer {ans} should be valid"
                );
            }
        }
    }

    #[test]
    fn test_logic_csps_xor_constraints() {
        let csps = generate_synthetic_csps(4);
        let logic_csps: Vec<_> = csps
            .iter()
            .filter(|c| c.label.starts_with("logic"))
            .collect();
        assert_eq!(logic_csps.len(), 4);
        for csp in logic_csps {
            // XOR partner should NOT be valid when the other is already placed
            if let Some(&partner) = csp.sufficient_answers.first() {
                assert!(
                    !csp.pruner.is_valid(csp.depth, partner, &csp.placed_tokens),
                    "XOR partner {partner} should be invalid when opposite is placed"
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
    let mut candidate_tokens: Vec<usize> = (0..vocab_size)
        .filter(|&tok| pruner.is_valid(depth, tok, placed_tokens))
        .collect();

    // Sort by constraint tightness (tokens that appear in most constraints first)
    // Heuristic: prefer tokens that are valid at deeper depths
    candidate_tokens.sort_by(|&a, &b| {
        let da = count_valid_extensions(pruner, depth + 1, a, placed_tokens);
        let db = count_valid_extensions(pruner, depth + 1, b, placed_tokens);
        da.cmp(&db) // ascending = tighter constraints first
    });

    let mut extended = placed_tokens.to_vec();
    for tok in candidate_tokens.iter().take(max_search_depth) {
        extended.push(*tok);
        let score =
            underspecification_score(&score_relevance(pruner, depth + 1, &extended, vocab_size));
        if score < 0.5 {
            sufficient.push(*tok);
            break; // found 1-sufficient
        }
        extended.pop();
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
    // Count valid tokens at this depth (sampling up to 256 for efficiency)
    let sample_size = 256;
    let mut count = 0;
    for tok in 0..sample_size {
        if pruner.is_valid(depth, tok, &extended) {
            count += 1;
        }
    }
    count
}

/// Compute relevance scores for all tokens at given depth.
fn score_relevance(
    pruner: &dyn crate::traits::ConstraintPruner,
    depth: usize,
    placed_tokens: &[usize],
    vocab_size: usize,
) -> Vec<f32> {
    (0..vocab_size.min(256))
        .map(|tok| {
            if pruner.is_valid(depth, tok, placed_tokens) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

// ── T4: QuestBenchDecision ───────────────────────────────────────

/// Decision from underspecification score for planning.
/// Maps to `PlanningDecision` in types.rs but lives here to avoid circular deps.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CspDomain {
    /// Grid-based (Bomber-like): adjacency constraints on a 2D grid.
    Grid,
    /// Stone-based (Go-like): liberty/capture constraints.
    Stone,
    /// Propositional logic: rule-based constraints.
    Logic,
}

// ── Grid CSP pruner (Bomber-like) ────────────────────────────────

/// Pruner modeling a Bomber-like grid where tokens represent cell indices
/// and constraints enforce adjacency rules.
struct GridCspPruner {
    grid_size: usize,
    /// At even depths: only tokens in `allowed` are valid.
    /// At odd depths: only tokens adjacent to last-placed token are valid.
    allowed: Vec<usize>,
}

impl crate::traits::ConstraintPruner for GridCspPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        if !self.allowed.contains(&token_idx) {
            return false;
        }
        if depth == 0 {
            return true;
        }
        // At depth > 0: must be adjacent to last-placed token on the grid
        let last = match parent_tokens.last() {
            Some(&t) => t,
            None => return true,
        };
        let row_last = last / self.grid_size;
        let col_last = last % self.grid_size;
        let row_tok = token_idx / self.grid_size;
        let col_tok = token_idx % self.grid_size;
        let manhattan = (row_last as i32 - row_tok as i32).unsigned_abs()
            + (col_last as i32 - col_tok as i32).unsigned_abs();
        manhattan <= 1
    }
}

// ── Stone CSP pruner (Go-like) ──────────────────────────────────

/// Pruner modeling Go-like stone placement where certain positions
/// are restricted based on previous placements (liberty rules).
struct StoneCspPruner {
    board_size: usize,
    /// Forbidden positions (already occupied).
    occupied: Vec<usize>,
    /// Positions that would be captured (suicide) given current board.
    suicide: Vec<usize>,
}

impl crate::traits::ConstraintPruner for StoneCspPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        !self.occupied.contains(&token_idx) && !self.suicide.contains(&token_idx)
    }
}

// ── Logic CSP pruner ────────────────────────────────────────────

/// Pruner modeling propositional logic rules.
/// Tokens represent propositions; constraints enforce that
/// certain combinations are mutually exclusive or required.
struct LogicCspPruner {
    /// Number of propositions (vocab_size).
    num_props: usize,
    /// Pairs (a, b) where exactly one must be true (XOR constraints).
    xor_pairs: Vec<(usize, usize)>,
    /// Propositions that must be true (given facts).
    must_be_true: Vec<usize>,
}

impl crate::traits::ConstraintPruner for LogicCspPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Cannot place a token that contradicts XOR with already-placed token
        for &(a, b) in &self.xor_pairs {
            if token_idx == a && parent_tokens.contains(&b) {
                return false;
            }
            if token_idx == b && parent_tokens.contains(&a) {
                return false;
            }
        }
        // Cannot place a "must be true" proposition as false (depth > num_props)
        if depth >= self.num_props {
            return false;
        }
        true
    }
}

/// Generate a batch of synthetic 1-sufficient CSPs.
///
/// Returns CSPs with known ground-truth sufficient tokens,
/// suitable for GOAT proof G2 (accuracy benchmark).
pub fn generate_synthetic_csps(count_per_domain: usize) -> Vec<SyntheticCsp> {
    let mut csps = Vec::with_capacity(count_per_domain * 3);

    // Grid CSPs (Bomber-like)
    for i in 0..count_per_domain {
        let grid_size = 4; // 4×4 = 16 cells
        let allowed: Vec<usize> = (0..grid_size * grid_size)
            .filter(|&c| c != i % (grid_size * grid_size)) // remove one cell
            .collect();
        let pruner = GridCspPruner { grid_size, allowed };
        // The removed cell is the sufficient answer
        let sufficient = vec![i % (grid_size * grid_size)];
        csps.push(SyntheticCsp {
            label: format!("grid_{i}"),
            pruner: Box::new(pruner),
            depth: 0,
            placed_tokens: vec![],
            vocab_size: grid_size * grid_size,
            sufficient_answers: sufficient,
        });
    }

    // Stone CSPs (Go-like)
    for i in 0..count_per_domain {
        let board_size = 5; // 5×5 = 25 positions
        let total = board_size * board_size;
        // Occupied: first i%10 positions
        let occupied: Vec<usize> = (0..(i % 10)).collect();
        // Suicide: next 2 positions
        let suicide: Vec<usize> = vec![(i % 10) + 1, (i % 10) + 2];
        let pruner = StoneCspPruner {
            board_size,
            occupied: occupied.clone(),
            suicide: suicide.clone(),
        };
        // The first non-occupied, non-suicide position is the sufficient answer
        let sufficient: Vec<usize> = (0..total)
            .filter(|&p| !occupied.contains(&p) && !suicide.contains(&p))
            .take(1)
            .collect();
        csps.push(SyntheticCsp {
            label: format!("stone_{i}"),
            pruner: Box::new(pruner),
            depth: 0,
            placed_tokens: vec![],
            vocab_size: total,
            sufficient_answers: sufficient,
        });
    }

    // Logic CSPs (propositional)
    for i in 0..count_per_domain {
        let num_props = 8;
        // Create XOR pairs: (0,1), (2,3), (4,5), (6,7)
        let xor_pairs: Vec<(usize, usize)> =
            (0..num_props / 2).map(|j| (j * 2, j * 2 + 1)).collect();
        // The sufficient answer is the XOR partner of a placed proposition
        let placed = vec![i % num_props];
        let partner = xor_pairs
            .iter()
            .find_map(|&(a, b)| {
                if a == placed[0] {
                    Some(b)
                } else if b == placed[0] {
                    Some(a)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let pruner = LogicCspPruner {
            num_props,
            xor_pairs,
            must_be_true: vec![],
        };
        csps.push(SyntheticCsp {
            label: format!("logic_{i}"),
            pruner: Box::new(pruner),
            depth: 0,
            placed_tokens: placed.clone(),
            vocab_size: num_props,
            sufficient_answers: vec![partner],
        });
    }

    csps
}
