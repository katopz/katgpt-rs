//! DDTree Path → Logical Rule Extraction (Plan 209, Phase 2 / T2).
//!
//! After DDTree AND-OR exploration, extract the TOP-K highest-scoring paths as
//! FOL-like logical rules. Each rule represents a pattern:
//! "if these tokens were chosen at these depths → this token is likely at this depth."
//!
//! # Architecture
//!
//! ```text
//! DDTree (TreeNode forest)
//!     │
//!     ├── DFS walk → collect complete paths with cumulative scores
//!     │
//!     ├── Filter: score >= min_score
//!     │
//!     ├── TOP-K selection by score
//!     │
//!     ├── Deduplication: Hamming distance ≤ threshold → merge
//!     │
//!     └── Vec<ExtractedRule> for downstream consumption
//! ```
//!
//! # Feature Gate
//!
//! Entire module is behind `#[cfg(feature = "rule_extraction")]`.
//! Zero-cost when disabled — no codegen, no vtable, no allocation.
//!
//! # References
//!
//! - Plan 209: FOL Logical Rule Inference — Modelless DDTree→FOL Pipeline
//! - Research 184: FOL-LNN Inference-Time Logical Rules (arXiv 2110.10963)

// Future wiring: RuleExtractor will implement ConstraintPruner integration
// when T3 (RewardMemPruner) and T4 (DecisionTrace) are wired.
#[allow(unused_imports)]
use crate::speculative::types::ConstraintPruner;

// ── TreeNode ────────────────────────────────────────────────────────

/// Lightweight DDTree node for rule extraction.
///
/// Represents a single token choice at a depth in the tree.
/// This is a simplified view of the full DDTree — only the fields needed
/// for path extraction and rule synthesis.
#[derive(Clone, Debug)]
pub struct TreeNode {
    /// Depth in the tree (0 = root level).
    pub depth: usize,
    /// Token index at this node.
    pub token_idx: usize,
    /// Score (weight/marginal) of this branch.
    pub score: f32,
    /// Children nodes (next depth choices).
    pub children: Vec<TreeNode>,
}

// ── ExtractedRule ───────────────────────────────────────────────────

/// A logical rule extracted from a DDTree path.
///
/// FOL interpretation:
/// ```text
/// conditions₀ ∧ conditions₁ ∧ … ∧ conditionsₙ₋₁ → action
/// ```
///
/// Each condition is `(depth, token_idx)` — "token X was chosen at depth D".
/// The action is the last `(depth, token_idx)` — the predicted consequent.
#[derive(Clone, Debug)]
pub struct ExtractedRule {
    /// Conditions: (depth, token_idx) pairs forming the antecedent.
    pub conditions: Vec<(usize, usize)>,
    /// Action: (depth, token_idx) — the consequent.
    pub action: (usize, usize),
    /// Score of this rule (cumulative product of DDTree path weights).
    pub score: f32,
    /// Number of times similar paths were merged into this rule.
    pub support: u32,
}

impl ExtractedRule {
    /// Create a new extracted rule.
    pub fn new(
        conditions: Vec<(usize, usize)>,
        action: (usize, usize),
        score: f32,
        support: u32,
    ) -> Self {
        Self {
            conditions,
            action,
            score,
            support,
        }
    }
}

// ── RuleExtractor ───────────────────────────────────────────────────

/// Configuration for rule extraction from DDTree paths.
///
/// Controls TOP-K selection and minimum quality threshold.
#[derive(Clone, Debug)]
pub struct RuleExtractor {
    /// Maximum rules to return.
    pub top_k: usize,
    /// Minimum score threshold for rule extraction.
    pub min_score: f32,
}

impl RuleExtractor {
    /// Create a new `RuleExtractor` with the given TOP-K and minimum score.
    pub fn new(top_k: usize, min_score: f32) -> Self {
        Self { top_k, min_score }
    }

    /// Extract logical rules from a DDTree forest.
    ///
    /// Walks each tree depth-first, collects complete root→leaf paths with
    /// cumulative scores (product of branch weights), filters by `min_score`,
    /// and returns the TOP-K rules sorted by descending score.
    ///
    /// For each path:
    /// - `conditions` = all nodes except the last
    /// - `action` = the last (deepest) node
    /// - `score` = cumulative product of node scores along the path
    /// - `support` = 1 (newly extracted, not yet deduplicated)
    pub fn extract(&self, trees: &[TreeNode]) -> Vec<ExtractedRule> {
        // Early return: empty forest → no rules.
        match trees.is_empty() {
            true => return Vec::new(),
            false => {}
        }

        let mut paths: Vec<PathAccumulator> = Vec::new();
        let mut stack: Vec<WalkState> = Vec::new();

        // Seed the DFS stack with root-level trees.
        for tree in trees {
            match tree.children.is_empty() {
                // Leaf at root level: single-node path.
                true => {
                    let path = PathAccumulator {
                        nodes: vec![(tree.depth, tree.token_idx)],
                        score: tree.score,
                    };
                    paths.push(path);
                }
                // Has children: push for DFS traversal.
                false => {
                    stack.push(WalkState {
                        path: vec![(tree.depth, tree.token_idx)],
                        cumulative_score: tree.score,
                        children: &tree.children,
                    });
                }
            }
        }

        // DFS traversal with iterative stack (no recursion → no stack overflow).
        while let Some(state) = stack.pop() {
            for child in state.children {
                let child_score = state.cumulative_score * child.score;
                let mut child_path = state.path.clone();
                child_path.push((child.depth, child.token_idx));

                match child.children.is_empty() {
                    // Leaf node: complete path found.
                    true => {
                        paths.push(PathAccumulator {
                            nodes: child_path,
                            score: child_score,
                        });
                    }
                    // Internal node: continue DFS.
                    false => {
                        stack.push(WalkState {
                            path: child_path,
                            cumulative_score: child_score,
                            children: &child.children,
                        });
                    }
                }
            }
        }

        // Filter by min_score threshold.
        let mut rules: Vec<ExtractedRule> = paths
            .into_iter()
            .filter_map(|path| {
                // A rule needs at least 2 nodes: conditions + action.
                match path.nodes.len() < 2 {
                    true => None,
                    false => match path.score < self.min_score {
                        true => None,
                        false => {
                            let action = path.nodes[path.nodes.len() - 1];
                            let conditions = path.nodes[..path.nodes.len() - 1].to_vec();
                            Some(ExtractedRule {
                                conditions,
                                action,
                                score: path.score,
                                support: 1,
                            })
                        }
                    },
                }
            })
            .collect();

        // Sort descending by score, keep TOP-K.
        rules.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rules.truncate(self.top_k);

        rules
    }
}

// ── Deduplication ───────────────────────────────────────────────────

/// Deduplicate rules by merging similar paths.
///
/// Two rules are "similar" when the Hamming distance between their conditions
/// is ≤ `hamming_threshold`. Hamming distance is the number of positions where
/// `(depth, token_idx)` pairs differ when aligned.
///
/// On merge:
/// - Keep the higher-scored rule.
/// - Increment `support` by the merged rule's support.
/// - Discard the lower-scored duplicate.
///
/// # Complexity
///
/// O(n²) in the number of rules — acceptable because TOP-K bounds n.
pub fn deduplicate_rules(rules: &mut Vec<ExtractedRule>, hamming_threshold: usize) {
    match rules.len() {
        0 | 1 => return,
        _ => {}
    }

    // Sort by score descending so higher-scored rules come first.
    rules.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut merged = Vec::with_capacity(rules.len());
    let mut consumed = vec![false; rules.len()];

    for i in 0..rules.len() {
        match consumed[i] {
            true => continue,
            false => {}
        }

        let mut rule = rules[i].clone();

        for j in (i + 1)..rules.len() {
            match consumed[j] {
                true => continue,
                false => {}
            }

            let dist = hamming_distance(&rule.conditions, &rules[j].conditions);
            match dist <= hamming_threshold {
                true => {
                    rule.support += rules[j].support;
                    consumed[j] = true;
                }
                false => {}
            }
        }

        merged.push(rule);
    }

    *rules = merged;
}

/// Compute Hamming distance between two condition lists.
///
/// Conditions are `(depth, token_idx)` pairs. Hamming distance counts positions
/// where pairs differ. If lengths differ, each extra position counts as a mismatch.
fn hamming_distance(a: &[(usize, usize)], b: &[(usize, usize)]) -> usize {
    let min_len = a.len().min(b.len());
    let len_diff = a.len().abs_diff(b.len());

    let mut dist = 0usize;
    for k in 0..min_len {
        match a[k] == b[k] {
            true => {}
            false => dist += 1,
        }
    }

    dist + len_diff
}

// ── Internal helpers ────────────────────────────────────────────────

/// Accumulator for a single root→leaf path during DFS.
struct PathAccumulator {
    nodes: Vec<(usize, usize)>,
    score: f32,
}

/// Stack entry for iterative DFS traversal.
struct WalkState<'a> {
    path: Vec<(usize, usize)>,
    cumulative_score: f32,
    children: &'a [TreeNode],
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a chain of `TreeNode`s from (depth, token_idx, score) triples.
    ///
    /// Returns a single root node with a linear chain of children.
    fn build_chain(pairs: &[(usize, usize, f32)]) -> TreeNode {
        assert!(!pairs.is_empty(), "chain must have at least one node");
        let mut iter = pairs.iter().rev();
        let last = iter.next().unwrap();
        let mut node = TreeNode {
            depth: last.0,
            token_idx: last.1,
            score: last.2,
            children: Vec::new(),
        };
        for &(depth, token_idx, score) in iter {
            node = TreeNode {
                depth,
                token_idx,
                score,
                children: vec![node],
            };
        }
        node
    }

    // ── Test: empty tree → no rules ────────────────────────────────

    #[test]
    fn test_empty_tree_no_rules() {
        let extractor = RuleExtractor::new(10, 0.0);
        let rules = extractor.extract(&[]);
        assert!(rules.is_empty(), "empty tree should yield no rules");
    }

    // ── Test: single path → single rule ────────────────────────────

    #[test]
    fn test_single_path_single_rule() {
        // Chain: depth=0 tok=1 (0.9) → depth=1 tok=5 (0.8) → depth=2 tok=3 (0.7)
        let chain = build_chain(&[(0, 1, 0.9), (1, 5, 0.8), (2, 3, 0.7)]);

        let extractor = RuleExtractor::new(10, 0.0);
        let rules = extractor.extract(&[chain]);

        assert_eq!(rules.len(), 1, "single path should yield one rule");

        let rule = &rules[0];
        // conditions = first two nodes, action = last node
        assert_eq!(rule.conditions, vec![(0, 1), (1, 5)]);
        assert_eq!(rule.action, (2, 3));
        // score = 0.9 * 0.8 * 0.7 = 0.504
        let expected_score = 0.9f32 * 0.8 * 0.7;
        assert!(
            (rule.score - expected_score).abs() < 1e-5,
            "expected score ~{expected_score}, got {}",
            rule.score
        );
        assert_eq!(rule.support, 1);
    }

    // ── Test: TOP-K extraction ─────────────────────────────────────

    #[test]
    fn test_top_k_extraction() {
        // Build two independent chains with different scores.
        let chain_a = build_chain(&[
            (0, 10, 0.9), // high score
            (1, 20, 0.9),
            (2, 30, 0.9),
        ]);
        let chain_b = build_chain(&[
            (0, 11, 0.5), // lower score
            (1, 21, 0.5),
            (2, 31, 0.5),
        ]);
        let chain_c = build_chain(&[
            (0, 12, 0.3), // lowest score
            (1, 22, 0.3),
            (2, 32, 0.3),
        ]);

        let extractor = RuleExtractor::new(2, 0.0);
        let rules = extractor.extract(&[chain_a, chain_b, chain_c]);

        assert_eq!(rules.len(), 2, "top_k=2 should return 2 rules");
        // First rule should be chain_a (highest score).
        assert_eq!(rules[0].action, (2, 30));
        assert!(
            rules[0].score > rules[1].score,
            "rules should be sorted by descending score"
        );
    }

    // ── Test: min_score threshold filters low-quality rules ────────

    #[test]
    fn test_min_score_threshold() {
        let chain_high = build_chain(&[(0, 1, 0.9), (1, 2, 0.9)]);
        let chain_low = build_chain(&[(0, 3, 0.1), (1, 4, 0.1)]);

        // min_score = 0.5 → chain_low (0.1*0.1 = 0.01) should be filtered.
        let extractor = RuleExtractor::new(10, 0.5);
        let rules = extractor.extract(&[chain_high, chain_low]);

        assert_eq!(rules.len(), 1, "low-score rule should be filtered");
        assert_eq!(rules[0].action, (1, 2));
    }

    // ── Test: deduplication merges similar paths ───────────────────

    #[test]
    fn test_deduplication_merges_similar() {
        let mut rules = vec![
            ExtractedRule {
                conditions: vec![(0, 1), (1, 2), (2, 3)],
                action: (3, 4),
                score: 0.9,
                support: 1,
            },
            // Identical conditions → Hamming distance 0 → should merge.
            ExtractedRule {
                conditions: vec![(0, 1), (1, 2), (2, 3)],
                action: (3, 4),
                score: 0.85,
                support: 1,
            },
            // One position different → Hamming distance 1 → merge with threshold 1.
            ExtractedRule {
                conditions: vec![(0, 1), (1, 99), (2, 3)],
                action: (3, 4),
                score: 0.8,
                support: 1,
            },
            // Two positions different → Hamming distance 2 → NOT merged with threshold 1.
            ExtractedRule {
                conditions: vec![(0, 88), (1, 99), (2, 3)],
                action: (3, 4),
                score: 0.7,
                support: 1,
            },
        ];

        deduplicate_rules(&mut rules, 1);

        assert_eq!(rules.len(), 2, "should have 2 rules after dedup");
        // Highest-scored kept with merged support.
        assert_eq!(
            rules[0].support, 3,
            "top rule should have support=3 (1+1+1)"
        );
        assert!(
            (rules[0].score - 0.9).abs() < 1e-5,
            "should keep highest score"
        );
        // The dissimilar rule survives alone.
        assert_eq!(rules[1].support, 1);
    }

    // ── Test: branching tree produces multiple paths ───────────────

    #[test]
    fn test_branching_tree() {
        // Root with two children, each with two leaves = 4 paths.
        let root = TreeNode {
            depth: 0,
            token_idx: 1,
            score: 1.0,
            children: vec![
                TreeNode {
                    depth: 1,
                    token_idx: 10,
                    score: 0.6,
                    children: vec![
                        TreeNode {
                            depth: 2,
                            token_idx: 100,
                            score: 0.9,
                            children: Vec::new(),
                        },
                        TreeNode {
                            depth: 2,
                            token_idx: 101,
                            score: 0.3,
                            children: Vec::new(),
                        },
                    ],
                },
                TreeNode {
                    depth: 1,
                    token_idx: 20,
                    score: 0.4,
                    children: vec![
                        TreeNode {
                            depth: 2,
                            token_idx: 200,
                            score: 0.8,
                            children: Vec::new(),
                        },
                        TreeNode {
                            depth: 2,
                            token_idx: 201,
                            score: 0.2,
                            children: Vec::new(),
                        },
                    ],
                },
            ],
        };

        let extractor = RuleExtractor::new(10, 0.0);
        let rules = extractor.extract(&[root]);

        assert_eq!(rules.len(), 4, "branching tree with 4 leaves → 4 rules");

        // Verify all rules are sorted by descending score.
        for window in rules.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "rules should be sorted descending: {} >= {}",
                window[0].score,
                window[1].score
            );
        }

        // Best path: root(1.0) → child(0.6) → leaf(0.9) = 0.54
        assert!(
            (rules[0].score - (1.0f32 * 0.6 * 0.9)).abs() < 1e-5,
            "best path score should be 0.54, got {}",
            rules[0].score
        );
        assert_eq!(rules[0].conditions, vec![(0, 1), (1, 10)]);
        assert_eq!(rules[0].action, (2, 100));
    }

    // ── Test: leaf-only tree (single node) produces no rules ───────

    #[test]
    fn test_single_node_no_rule() {
        let single = TreeNode {
            depth: 0,
            token_idx: 42,
            score: 0.99,
            children: Vec::new(),
        };

        let extractor = RuleExtractor::new(10, 0.0);
        let rules = extractor.extract(&[single]);

        assert!(
            rules.is_empty(),
            "single-node path (no conditions + action split) should yield no rules"
        );
    }

    // ── Test: hamming_distance correctness ─────────────────────────

    #[test]
    fn test_hamming_distance() {
        let a: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3)];
        let b: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3)];
        assert_eq!(hamming_distance(&a, &b), 0, "identical → distance 0");

        let c: Vec<(usize, usize)> = vec![(0, 1), (1, 99), (2, 3)];
        assert_eq!(
            hamming_distance(&a, &c),
            1,
            "one position differs → distance 1"
        );

        let d: Vec<(usize, usize)> = vec![(0, 88), (1, 99), (2, 3)];
        assert_eq!(
            hamming_distance(&a, &d),
            2,
            "two positions differ → distance 2"
        );

        // Different lengths
        let e: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
        assert_eq!(
            hamming_distance(&a, &e),
            1,
            "length diff 1 + 0 mismatches = 1"
        );

        let f: Vec<(usize, usize)> = vec![(0, 88)];
        assert_eq!(
            hamming_distance(&a, &f),
            3,
            "length diff 2 + 1 mismatch = 3"
        );
    }

    // ── Test: deduplicate empty and single-element ─────────────────

    #[test]
    fn test_deduplication_edge_cases() {
        let mut empty: Vec<ExtractedRule> = Vec::new();
        deduplicate_rules(&mut empty, 0);
        assert!(empty.is_empty());

        let mut single = vec![ExtractedRule {
            conditions: vec![(0, 1)],
            action: (1, 2),
            score: 0.5,
            support: 1,
        }];
        deduplicate_rules(&mut single, 0);
        assert_eq!(single.len(), 1, "single rule should survive");
        assert_eq!(single[0].support, 1);
    }
}
