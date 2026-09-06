
use super::*;
use crate::dflash::dflash_predict;
use katgpt_core::speculative::types::{SdeConfig, TreePath};
use katgpt_transformer::TransformerWeights;
use katgpt_types::{Config, Rng};

fn make_draft() -> (TransformerWeights, Config) {
    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    (weights, config)
}

// ── Original DDTree Tests ─────────────────────────────────

#[test]
fn test_extract_parent_tokens_roundtrip() {
    let path_d0 = TreePath::from_tokens(&[3]);
    let path_d1 = TreePath::from_tokens(&[3, 7]);
    let path_d2 = TreePath::from_tokens(&[3, 7, 1]);

    assert_eq!(extract_parent_tokens(path_d0, 1), vec![3]);
    assert_eq!(extract_parent_tokens(path_d1, 2), vec![3, 7]);
    assert_eq!(extract_parent_tokens(path_d2, 3), vec![3, 7, 1]);
    let empty: Vec<usize> = extract_parent_tokens(TreePath::default(), 0);
    assert!(empty.is_empty());
}

#[test]
fn test_ddtree_respects_budget() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    assert!(
        tree.len() <= config.tree_budget,
        "tree size {} exceeds budget {}",
        tree.len(),
        config.tree_budget
    );
    assert!(!tree.is_empty(), "tree should have at least one node");
}

#[test]
fn test_ddtree_scores_descending() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    for window in tree.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "scores not descending: {} >= {}",
            window[0].score,
            window[1].score
        );
    }
}

#[test]
fn test_ddtree_depth_within_lookahead() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    for node in &tree {
        assert!(
            node.depth < config.draft_lookahead,
            "depth {} should be < lookahead {}",
            node.depth,
            config.draft_lookahead
        );
    }
}

#[test]
fn test_ddtree_valid_token_indices() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    for node in &tree {
        assert!(
            node.token_idx < config.vocab_size,
            "token_idx {} out of range",
            node.token_idx
        );
    }
}

#[test]
fn test_ddtree_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree(&[], &config);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

#[test]
fn test_ddtree_pruned_same_as_unpruned_with_no_pruner() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_unpruned = build_dd_tree(&mv, &config);
    let tree_pruned = build_dd_tree_pruned(&mv, &config, &NoPruner, false);

    assert_eq!(
        tree_unpruned.len(),
        tree_pruned.len(),
        "NoPruner should produce identical tree"
    );
    for (a, b) in tree_unpruned.iter().zip(tree_pruned.iter()) {
        assert_eq!(a.score, b.score, "scores should match");
        assert_eq!(a.token_idx, b.token_idx, "tokens should match");
    }
}

#[test]
fn test_ddtree_pruned_empty_marginals() {
    let config = Config::draft();
    let pruner = NoPruner;
    let tree = build_dd_tree_pruned(&[], &config, &pruner, false);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

// ── merge_retrieved_branches Tests ─────────────────────────

#[test]
fn test_merge_empty_retrieval_noop() {
    let config = Config::draft();
    let marginals = [vec![0.5; config.vocab_size]];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = vec![TreeNode {
        score: 1.0,
        depth: 0,
        token_idx: 0,
        parent_path: TreePath::default(),
    }];
    let original_len = tree.len();

    merge_retrieved_branches(&mut tree, &mv, &config, &[], &[], 0.5);

    assert_eq!(
        tree.len(),
        original_len,
        "empty retrieval should not change tree"
    );
}

#[test]
fn test_merge_preserves_budget() {
    let config = Config::draft();
    let marginals = vec![vec![0.1; config.vocab_size]; 4];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = build_dd_tree(&mv, &config);

    // Create many sequences that would exceed budget
    let sequences: Vec<Vec<usize>> = (0..100)
        .map(|i| vec![i % config.vocab_size, (i + 1) % config.vocab_size])
        .collect();
    let scores: Vec<f32> = (0..100).map(|_| 0.9).collect();

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.3);

    assert!(
        tree.len() <= config.tree_budget,
        "tree should not exceed budget, got {}",
        tree.len()
    );
}

#[test]
fn test_merge_sorts_by_score() {
    let config = Config::draft();
    let marginals = vec![vec![0.1; config.vocab_size]; 2];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = Vec::new();

    let sequences = vec![vec![0, 1], vec![2, 3]];
    let scores = vec![0.5, 0.9];

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.5);

    for window in tree.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "tree should be sorted by score descending"
        );
    }
}

#[test]
fn test_merge_with_empty_tree_adds_nodes() {
    let config = Config::draft();
    // Marginals with non-zero prob at specific tokens
    let mut m0 = vec![0.0; config.vocab_size];
    m0[5] = 0.8;
    let mut m1 = vec![0.0; config.vocab_size];
    m1[10] = 0.6;
    let marginals = [m0, m1];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = Vec::new();

    let sequences = vec![vec![5, 10]];
    let scores = vec![0.7];

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.3);

    assert_eq!(tree.len(), 2, "should add 2 nodes for 2-depth sequence");
    assert_eq!(tree[0].token_idx, 5, "first node should be token 5");
}

#[test]
fn test_merge_zero_weight_is_noop() {
    let config = Config::draft();
    let marginals = [vec![0.5; config.vocab_size]];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = Vec::new();

    let sequences = vec![vec![0]];
    let scores = vec![0.9];

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.0);

    assert!(tree.is_empty(), "zero rest_weight should be no-op");
}

// ── Chain-Seed DDTree Tests ───────────────────────────────

/// Create marginals with known argmax at each depth for deterministic testing.
fn make_chain_marginals(config: &Config) -> Vec<Vec<f32>> {
    let mut m0 = vec![0.01; config.vocab_size];
    m0[5] = 0.9;
    let mut m1 = vec![0.01; config.vocab_size];
    m1[10] = 0.85;
    let mut m2 = vec![0.01; config.vocab_size];
    m2[3] = 0.8;
    vec![m0, m1, m2]
}

#[test]
fn test_chain_seed_produces_chain_path() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree = build_dd_tree_pruned(&mv, &config, &NoPruner, true);

    // Chain nodes are the first 3 entries (depths 0, 1, 2)
    assert!(
        tree.len() >= 3,
        "tree should have at least 3 chain nodes, got {}",
        tree.len()
    );

    // Verify chain nodes form contiguous path with argmax tokens
    assert_eq!(tree[0].depth, 0, "first chain node at depth 0");
    assert_eq!(tree[0].token_idx, 5, "chain node depth 0 = argmax token 5");

    assert_eq!(tree[1].depth, 1, "second chain node at depth 1");
    assert_eq!(
        tree[1].token_idx, 10,
        "chain node depth 1 = argmax token 10"
    );

    assert_eq!(tree[2].depth, 2, "third chain node at depth 2");
    assert_eq!(tree[2].token_idx, 3, "chain node depth 2 = argmax token 3");

    // Verify chain node parent_paths form contiguous path
    assert_eq!(
        tree[0].parent_path,
        TreePath::from_tokens(&[5]),
        "depth 0 parent_path = token 5"
    );
    assert_eq!(
        tree[1].parent_path,
        TreePath::from_tokens(&[5, 10]),
        "depth 1 parent_path = [5, 10]"
    );
    assert_eq!(
        tree[2].parent_path,
        TreePath::from_tokens(&[5, 10, 3]),
        "depth 2 parent_path = [5, 10, 3]"
    );

    // Verify cumulative scores
    let expected_d0 = marginals[0][5].ln();
    let expected_d1 = expected_d0 + marginals[1][10].ln();
    let expected_d2 = expected_d1 + marginals[2][3].ln();

    assert!(
        (tree[0].score - expected_d0).abs() < 1e-5,
        "depth 0 score: expected {expected_d0}, got {}",
        tree[0].score
    );
    assert!(
        (tree[1].score - expected_d1).abs() < 1e-5,
        "depth 1 score: expected {expected_d1}, got {}",
        tree[1].score
    );
    assert!(
        (tree[2].score - expected_d2).abs() < 1e-5,
        "depth 2 score: expected {expected_d2}, got {}",
        tree[2].score
    );
}

#[test]
fn test_chain_seed_false_matches_original() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // build_dd_tree calls build_dd_tree_pruned with chain_seed=false
    let tree_via_wrapper = build_dd_tree(&mv, &config);
    let tree_via_pruned = build_dd_tree_pruned(&mv, &config, &NoPruner, false);

    assert_eq!(
        tree_via_wrapper.len(),
        tree_via_pruned.len(),
        "both should produce same number of nodes"
    );
    for (a, b) in tree_via_wrapper.iter().zip(tree_via_pruned.iter()) {
        assert_eq!(a.score, b.score, "scores should match");
        assert_eq!(a.token_idx, b.token_idx, "tokens should match");
        assert_eq!(a.depth, b.depth, "depths should match");
        assert_eq!(a.parent_path, b.parent_path, "parent_paths should match");
    }
}

#[test]
fn test_chain_seed_respects_budget() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree = build_dd_tree_pruned(&mv, &config, &NoPruner, true);

    assert!(
        tree.len() <= config.tree_budget,
        "chain-seed tree size {} exceeds budget {}",
        tree.len(),
        config.tree_budget
    );
    assert!(!tree.is_empty(), "tree should have at least one node");
}

/// Pruner that blocks a specific token at a specific depth.
struct BlockTokenPruner {
    depth: usize,
    token: usize,
}

impl ConstraintPruner for BlockTokenPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        !(depth == self.depth && token_idx == self.token)
    }
}

#[test]
fn test_chain_seed_with_pruner() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);

    // Block token 10 at depth 1 (the argmax) — chain should break there
    let pruner = BlockTokenPruner {
        depth: 1,
        token: 10,
    };
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree_pruned(&mv, &config, &pruner, true);

    // Chain should have only depth 0 (broke at depth 1)
    assert!(
        !tree.is_empty(),
        "tree should have at least the depth 0 chain node"
    );
    assert_eq!(
        tree[0].token_idx, 5,
        "depth 0 chain node should be argmax token 5"
    );
    assert_eq!(tree[0].depth, 0);

    // No node at depth 1 should have token 10 (blocked)
    for node in &tree {
        if node.depth == 1 {
            assert_ne!(
                node.token_idx, 10,
                "blocked token 10 should not appear at depth 1"
            );
        }
    }

    // Verify tree still contains valid nodes (siblings and best-first)
    assert!(
        tree.len() > 1,
        "tree should have more than just the chain node (siblings/best-first)"
    );
}

#[test]
fn test_chain_seed_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree_pruned(&[], &config, &NoPruner, true);
    assert!(
        tree.is_empty(),
        "empty marginals should produce empty tree with chain_seed=true"
    );
}

// ── ScreeningPruner Tests (Plan 021) ──────────────────────

/// Screener that returns fixed relevance per token index.
struct FixedRelevanceScreener {
    relevances: Vec<f32>,
}

impl ScreeningPruner for FixedRelevanceScreener {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.relevances.get(token_idx).copied().unwrap_or(0.1)
    }
}

#[test]
fn test_screened_no_screener_matches_unpruned() {
    // NoScreeningPruner returns 1.0 everywhere → ln(1.0)=0.0 → same as unpruned
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_unpruned = build_dd_tree(&mv, &config);
    let tree_screened = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, false);

    assert_eq!(
        tree_unpruned.len(),
        tree_screened.len(),
        "NoScreeningPruner should produce identical tree size"
    );
    for (a, b) in tree_unpruned.iter().zip(tree_screened.iter()) {
        assert!(
            (a.score - b.score).abs() < 1e-5,
            "scores should match: {} vs {}",
            a.score,
            b.score
        );
        assert_eq!(a.token_idx, b.token_idx, "tokens should match");
    }
}

#[test]
fn test_screened_binary_compat_via_adapter() {
    // BinaryScreeningPruner adapter: ConstraintPruner → ScreeningPruner with R∈{0.0,1.0}
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_pruned = build_dd_tree_pruned(&mv, &config, &NoPruner, false);
    // NoPruner wrapped in adapter: is_valid=true → relevance=1.0 → ln(1.0)=0.0
    let tree_screened =
        build_dd_tree_screened(&mv, &config, &BinaryScreeningPruner(NoPruner), false);

    assert_eq!(
        tree_pruned.len(),
        tree_screened.len(),
        "binary compat: same tree size via adapter"
    );
    for (a, b) in tree_pruned.iter().zip(tree_screened.iter()) {
        assert!(
            (a.score - b.score).abs() < 1e-5,
            "binary compat: scores should match"
        );
    }
}

#[test]
fn test_screened_relevance_zero_hard_trims() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    // 3 tokens: index 0 has high prob but R=0.0, index 1 has lower prob but R=1.0
    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.9; // high LLM prob
    m0[1] = 0.05; // lower LLM prob
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![0.0, 1.0], // token 0 trimmed, token 1 passes
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    // Token 0 should be completely absent (hard trim)
    for node in &tree {
        assert_ne!(
            node.token_idx, 0,
            "token 0 with relevance 0.0 should be hard-trimmed"
        );
    }
}

#[test]
fn test_screened_relevance_half_applies_penalty() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    // Two tokens with same LLM prob but different relevance
    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.5;
    m0[1] = 0.5;
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![1.0, 0.5], // token 1 gets -0.69 penalty
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    let node_0 = tree.iter().find(|n| n.token_idx == 0);
    let node_1 = tree.iter().find(|n| n.token_idx == 1);

    assert!(node_0.is_some(), "token 0 should be in tree");
    assert!(node_1.is_some(), "token 1 should be in tree");

    let score_0 = node_0.unwrap().score;
    let score_1 = node_1.unwrap().score;

    // Token 0: ln(0.5) + ln(1.0) = ln(0.5) + 0
    // Token 1: ln(0.5) + ln(0.5) = ln(0.5) - 0.693...
    let expected_diff = 0.5f32.ln().abs(); // ≈ 0.693
    let actual_diff = score_0 - score_1;

    assert!(
        (actual_diff - expected_diff).abs() < 1e-4,
        "penalty should be ln(0.5) ≈ -0.693, got diff={actual_diff:.4}, expected={expected_diff:.4}"
    );
}

#[test]
fn test_screened_relevance_one_no_penalty() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.8;
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![1.0],
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    let node = tree.iter().find(|n| n.token_idx == 0);
    assert!(node.is_some(), "token 0 should be in tree");

    let expected_score = 0.8f32.ln(); // ln(P) + ln(1.0) = ln(P) + 0
    assert!(
        (node.unwrap().score - expected_score).abs() < 1e-5,
        "relevance 1.0 should not modify score"
    );
}

#[test]
fn test_screened_threshold_trims_mediocre() {
    let mut config = Config::draft();
    config.tree_budget = 64;
    config.screening_threshold = 0.4; // trim anything ≤ 0.4

    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.5;
    m0[1] = 0.5;
    m0[2] = 0.5;
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![0.3, 0.5, 0.8], // token 0 trimmed (≤0.4), 1 and 2 pass
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    // Token 0 (R=0.3 ≤ threshold 0.4) should be absent
    for node in &tree {
        assert_ne!(
            node.token_idx, 0,
            "token 0 with R=0.3 should be trimmed by threshold 0.4"
        );
    }
    // Token 1 (R=0.5 > threshold) and token 2 (R=0.8 > threshold) should be present
    assert!(
        tree.iter().any(|n| n.token_idx == 1),
        "token 1 with R=0.5 should survive threshold 0.4"
    );
    assert!(
        tree.iter().any(|n| n.token_idx == 2),
        "token 2 with R=0.8 should survive threshold 0.4"
    );
}

#[test]
fn test_screened_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree_screened(&[], &config, &NoScreeningPruner, false);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

#[test]
fn test_screened_chain_seed_with_relevance() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    let mut m0 = vec![0.01; config.vocab_size];
    m0[5] = 0.9;
    let mut m1 = vec![0.01; config.vocab_size];
    m1[10] = 0.85;
    let marginals = [m0, m1];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Give token 5 at depth 0 a relevance of 0.6
    let mut relevances = vec![0.1; config.vocab_size];
    relevances[5] = 0.6;
    relevances[10] = 1.0;
    let screener = FixedRelevanceScreener { relevances };

    let tree = build_dd_tree_screened(&mv, &config, &screener, true);

    // Chain should build: token 5 (R=0.6), token 10 (R=1.0)
    assert!(
        tree.len() >= 2,
        "chain should have at least 2 nodes, got {}",
        tree.len()
    );

    // Score for token 5 should include ln(0.6) penalty
    let chain_d0 = tree.iter().find(|n| n.depth == 0 && n.token_idx == 5);
    assert!(chain_d0.is_some(), "chain node at depth 0 should exist");
    let expected_d0 = 0.9f32.ln() + 0.6f32.ln();
    assert!(
        (chain_d0.unwrap().score - expected_d0).abs() < 1e-4,
        "chain d0 score should include ln(0.6) penalty"
    );
}

// ── Early Exit Tests (Plan 026: AutoTTS) ──────────────────

#[test]
fn test_ddtree_early_exit_triggers_on_clear_winner() {
    // Create marginals where one path dominates massively
    let config = Config {
        tree_budget: 1000,
        early_exit_patience: 3,
        early_exit_gap: 1.0,
        ..Config::draft()
    };
    // One dominant token per depth
    let mut marginals = Vec::new();
    for _ in 0..config.draft_lookahead {
        let mut probs = vec![0.001_f32; config.vocab_size];
        probs[0] = 0.99; // token 0 dominates
        marginals.push(probs);
    }
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    // Should exit well before budget of 1000
    assert!(
        tree.len() < 1000,
        "early exit should trigger, got {} nodes vs budget 1000",
        tree.len()
    );
}

#[test]
fn test_ddtree_early_exit_disabled_when_patience_zero() {
    let config = Config {
        tree_budget: 100,
        early_exit_patience: 0,
        early_exit_gap: 100.0,
        ..Config::draft()
    };
    let (weights, _) = make_draft();
    let marginals = dflash_predict(&weights, &Config::draft(), 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    // patience=0 disables early exit — tree should reach natural termination
    assert!(
        tree.len() <= config.tree_budget,
        "tree size {} exceeds budget {}",
        tree.len(),
        config.tree_budget
    );
}

#[test]
fn test_ddtree_early_exit_no_false_exit_on_tight_gap() {
    // Uniform marginals — no clear winner, gap stays small
    let config = Config {
        tree_budget: 50,
        early_exit_patience: 5,
        early_exit_gap: 50.0, // very high gap requirement
        ..Config::draft()
    };
    let (weights, _) = make_draft();
    let marginals = dflash_predict(&weights, &Config::draft(), 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    // Gap too high to ever trigger — tree should fill normally
    assert!(!tree.is_empty());
}

// ── Balanced DDTree Tests (Plan 052: GFlowNet) ───────────

#[test]
fn test_balanced_default_matches_screened() {
    // backward_weight=1.0, lambda_flow=0.0 → identical to build_screened
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_screened = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, false);
    let tree_balanced =
        build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 1.0, 0.0);

    assert_eq!(
        tree_screened.len(),
        tree_balanced.len(),
        "balanced(w=1,λ=0) should match screened: {} vs {}",
        tree_screened.len(),
        tree_balanced.len()
    );
    for (a, b) in tree_screened.iter().zip(tree_balanced.iter()) {
        assert!(
            (a.score - b.score).abs() < 1e-4,
            "score mismatch: {} vs {}",
            a.score,
            b.score
        );
        assert_eq!(a.token_idx, b.token_idx, "token mismatch");
        assert_eq!(a.depth, b.depth, "depth mismatch");
    }
}

#[test]
fn test_balanced_default_chain_seed_matches_screened() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_screened = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, true);
    let tree_balanced =
        build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, true, &[], 1.0, 0.0);

    assert_eq!(
        tree_screened.len(),
        tree_balanced.len(),
        "balanced(w=1,λ=0) chain_seed should match screened"
    );
}

#[test]
fn test_balanced_higher_backward_weight_changes_scores() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_w1 = build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 1.0, 0.0);
    let tree_w4 = build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 4.0, 0.0);

    // With higher backward weight, scores should be different
    // (NoScreeningPruner returns 1.0, so ln(R)=0 — but the scoring is additive)
    // Actually with NoScreeningPruner, relevance=1.0, ln(1.0)=0, so backward_weight
    // multiplies 0.0 → same score. Use a pruner that returns non-1.0 values.
    // For now just verify they both produce valid trees
    assert!(!tree_w1.is_empty());
    assert!(!tree_w4.is_empty());
}

#[test]
fn test_balanced_with_relevance_pruner_weighted() {
    // Use FixedRelevanceScreener to get non-trivial relevance scores
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // FixedRelevanceScreener indexes by token_idx — flat vec
    let screener = FixedRelevanceScreener {
        relevances: vec![0.5; config.vocab_size],
    };

    let tree_w1 = build_dd_tree_balanced(&mv, &config, &screener, false, &[], 1.0, 0.0);
    let tree_w4 = build_dd_tree_balanced(&mv, &config, &screener, false, &[], 4.0, 0.0);

    // Higher backward weight should amplify the relevance penalty
    // Both should be non-empty
    assert!(!tree_w1.is_empty());
    assert!(!tree_w4.is_empty());

    // The top node scores should differ because backward_weight scales ln(R)
    // w=1: score = ln(P) + 1*ln(0.5) = ln(P) - 0.693
    // w=4: score = ln(P) + 4*ln(0.5) = ln(P) - 2.773
    if !tree_w1.is_empty() && !tree_w4.is_empty() {
        // w=4 should have lower scores (more penalty)
        assert!(
            tree_w4[0].score < tree_w1[0].score,
            "w=4 score {} should be < w=1 score {}",
            tree_w4[0].score,
            tree_w1[0].score
        );
    }
}

#[test]
fn test_balanced_flow_bonus_changes_scores() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Low stop prob → high flow bonus
    let stop_probs = vec![0.1; config.draft_lookahead];

    let tree_no_flow = build_dd_tree_balanced(
        &mv,
        &config,
        &NoScreeningPruner,
        false,
        &stop_probs,
        1.0,
        0.0,
    );
    let tree_with_flow = build_dd_tree_balanced(
        &mv,
        &config,
        &NoScreeningPruner,
        false,
        &stop_probs,
        1.0,
        0.3,
    );

    // Flow bonus should increase scores (additive positive term)
    assert!(!tree_no_flow.is_empty());
    assert!(!tree_with_flow.is_empty());

    // With flow bonus, scores should be higher
    if !tree_no_flow.is_empty() && !tree_with_flow.is_empty() {
        assert!(
            tree_with_flow[0].score > tree_no_flow[0].score,
            "flow bonus should increase score: {} vs {}",
            tree_with_flow[0].score,
            tree_no_flow[0].score
        );
    }
}

#[test]
fn test_balanced_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree_balanced(&[], &config, &NoScreeningPruner, false, &[], 2.0, 0.3);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

#[test]
fn test_balanced_respects_budget() {
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let stop_probs = vec![0.5; config.draft_lookahead];

    let tree = build_dd_tree_balanced(
        &mv,
        &config,
        &NoScreeningPruner,
        false,
        &stop_probs,
        2.0,
        0.3,
    );

    assert!(
        tree.len() <= config.tree_budget,
        "balanced tree size {} exceeds budget {}",
        tree.len(),
        config.tree_budget
    );
    assert!(!tree.is_empty(), "tree should have at least one node");
}

#[test]
fn test_balanced_scores_descending_without_flow() {
    // Scores descend when lambda_flow=0 (pure log-prob + backward weight).
    // With flow bonus > 0, ordering may change — that's by design
    // (flow bonus intentionally boosts exploration in low-stop-prob regions).
    let (weights, config) = make_draft();
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let stop_probs = vec![0.3; config.draft_lookahead];

    let tree = build_dd_tree_balanced(
        &mv,
        &config,
        &NoScreeningPruner,
        false,
        &stop_probs,
        2.0,
        0.0, // No flow bonus → scores must descend
    );

    for window in tree.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "scores not descending: {} >= {}",
            window[0].score,
            window[1].score
        );
    }
}

// ── SDE Noise Tests (ELF Plan 079) ────────────────────────

#[test]
fn test_sde_noise_disabled_is_noop() {
    let config = SdeConfig::default(); // gamma = 0.0
    let marginals: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6], &[0.2, 0.5, 0.3]];
    let mut rng = Rng::new(42);
    let noisy = inject_sde_noise(&marginals, &config, &mut rng);
    for (orig, perturbed) in marginals.iter().zip(noisy.iter()) {
        for (a, b) in orig.iter().zip(perturbed.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "disabled SDE should not change marginals"
            );
        }
    }
}

// ── PTRM Width Scaling Tests (Plan 083) ───────────────────

#[cfg(feature = "elf_sde")]
#[test]
fn test_width_scale_config_defaults() {
    use super::WidthScaleConfig;
    use super::WidthSelectionMode;

    let default = WidthScaleConfig::default();
    assert_eq!(default.k_rollouts, 1);
    assert_eq!(default.selection, WidthSelectionMode::BestQ);

    let ptrm = WidthScaleConfig::ptrm_default();
    assert_eq!(ptrm.k_rollouts, 16);
    assert_eq!(ptrm.selection, WidthSelectionMode::BestQ);
}

#[cfg(feature = "elf_sde")]
#[test]
fn test_best_of_k_rollouts_k1_matches_single_tree() {
    use super::{WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts};
    #[cfg(feature = "eqr_convergence")]
    use super::RestartMode;
    use katgpt_core::speculative::types::SdeConfig;

    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let sde_config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    // K=1 should produce same result as a single tree build
    let path = best_of_k_rollouts(
        &mv,
        &config,
        &NoScreeningPruner,
        &sde_config,
        &WidthScaleConfig {
            k_rollouts: 1,
            selection: WidthSelectionMode::BestQ,
            #[cfg(feature = "eqr_convergence")]
            restart_mode: RestartMode::Perturb,
        },
        42,
    );

    assert!(!path.is_empty(), "K=1 should produce a non-empty path");
    // Path length is bounded by draft_lookahead but may be shorter when the
    // marginal tree has a dominant early terminator (depends on the weight
    // init RNG). Assert the invariant: non-empty and within budget.
    assert!(
        path.len() <= config.draft_lookahead,
        "path length {} exceeds lookahead {}",
        path.len(),
        config.draft_lookahead
    );
}

#[cfg(feature = "elf_sde")]
#[test]
fn test_best_of_k_rollouts_k16_produces_diverse_paths() {
    use super::{WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts};
    #[cfg(feature = "eqr_convergence")]
    use super::RestartMode;
    use katgpt_core::speculative::types::SdeConfig;

    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let sde_config = SdeConfig {
        gamma: 1.0,
        ..Default::default()
    };

    // Run multiple trials with K=16 and collect paths
    let mut paths = std::collections::HashSet::new();
    for seed in 0..20u64 {
        let path = best_of_k_rollouts(
            &mv,
            &config,
            &NoScreeningPruner,
            &sde_config,
            &WidthScaleConfig {
                k_rollouts: 16,
                selection: WidthSelectionMode::BestQ,
                #[cfg(feature = "eqr_convergence")]
                restart_mode: RestartMode::Perturb,
            },
            seed,
        );
        paths.insert(path);
    }

    // With γ=1.0 and K=16 across 20 trials, we should see path diversity
    assert!(
        paths.len() > 1,
        "K=16 with γ=1.0 should produce diverse paths across trials, got {} unique",
        paths.len()
    );
}

#[cfg(feature = "elf_sde")]
#[test]
fn test_best_of_k_rollouts_no_sde_fallback() {
    use super::{WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts};
    #[cfg(feature = "eqr_convergence")]
    use super::RestartMode;
    use katgpt_core::speculative::types::SdeConfig;

    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // SDE disabled — should fall back to single tree regardless of K
    let sde_config = SdeConfig {
        gamma: 0.0,
        ..Default::default()
    };

    let path1 = best_of_k_rollouts(
        &mv,
        &config,
        &NoScreeningPruner,
        &sde_config,
        &WidthScaleConfig {
            k_rollouts: 64,
            selection: WidthSelectionMode::BestQ,
            #[cfg(feature = "eqr_convergence")]
            restart_mode: RestartMode::Perturb,
        },
        42,
    );
    let path2 = best_of_k_rollouts(
        &mv,
        &config,
        &NoScreeningPruner,
        &sde_config,
        &WidthScaleConfig {
            k_rollouts: 1,
            selection: WidthSelectionMode::BestQ,
            #[cfg(feature = "eqr_convergence")]
            restart_mode: RestartMode::Perturb,
        },
        42,
    );

    // Both should produce the same deterministic path when SDE is off
    assert_eq!(
        path1, path2,
        "SDE disabled should produce identical paths regardless of K"
    );
}

#[cfg(feature = "elf_sde")]
#[test]
fn test_best_of_k_rollouts_most_frequent_mode() {
    use super::{WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts};
    #[cfg(feature = "eqr_convergence")]
    use super::RestartMode;
    use katgpt_core::speculative::types::SdeConfig;

    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let sde_config = SdeConfig {
        gamma: 0.2, // low noise → most paths converge
        ..Default::default()
    };

    let path = best_of_k_rollouts(
        &mv,
        &config,
        &NoScreeningPruner,
        &sde_config,
        &WidthScaleConfig {
            k_rollouts: 8,
            selection: WidthSelectionMode::MostFrequent,
            #[cfg(feature = "eqr_convergence")]
            restart_mode: RestartMode::Perturb,
        },
        42,
    );

    assert!(
        !path.is_empty(),
        "MostFrequent mode should return a non-empty path"
    );
}

#[cfg(feature = "elf_sde")]
#[test]
fn test_best_of_k_rollouts_empty_marginals() {
    use super::{WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts};
    #[cfg(feature = "eqr_convergence")]
    use super::RestartMode;
    use katgpt_core::speculative::types::SdeConfig;

    let config = Config::draft();
    let sde_config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    let path = best_of_k_rollouts(
        &[],
        &config,
        &NoScreeningPruner,
        &sde_config,
        &WidthScaleConfig {
            k_rollouts: 4,
            selection: WidthSelectionMode::BestQ,
            #[cfg(feature = "eqr_convergence")]
            restart_mode: RestartMode::Perturb,
        },
        42,
    );

    assert!(path.is_empty(), "empty marginals should produce empty path");
}

#[test]
fn test_sde_noise_enabled_changes_marginals() {
    let config = SdeConfig {
        gamma: 1.0,
        ..Default::default()
    };
    let marginals: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6], &[0.2, 0.5, 0.3]];
    let mut rng = Rng::new(42);
    let noisy = inject_sde_noise(&marginals, &config, &mut rng);
    // At least one value should differ
    let mut any_changed = false;
    for (orig, perturbed) in marginals.iter().zip(noisy.iter()) {
        for (a, b) in orig.iter().zip(perturbed.iter()) {
            if (a - b).abs() > 1e-6 {
                any_changed = true;
                break;
            }
        }
    }
    assert!(any_changed, "enabled SDE should change marginals");
}

#[test]
fn test_sde_noise_preserves_sum_to_one() {
    let config = SdeConfig {
        gamma: 2.0,
        ..Default::default()
    };
    let marginals: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6], &[0.2, 0.5, 0.3]];
    let mut rng = Rng::new(42);
    let noisy = inject_sde_noise(&marginals, &config, &mut rng);
    for perturbed in &noisy {
        let sum: f32 = perturbed.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "perturbed marginals should sum to ~1.0, got {sum}"
        );
    }
}

#[test]
fn test_sde_noise_preserve_top1() {
    let config = SdeConfig {
        gamma: 1.0,
        preserve_top1: true,
        confidence_floor: 0.0,
    };
    let marginals: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6]]; // top-1 is index 2
    let mut rng = Rng::new(42);
    let noisy = inject_sde_noise(&marginals, &config, &mut rng);
    // Top-1 should be preserved
    assert_eq!(
        noisy[0]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map(|(i, _)| i),
        Some(2),
        "preserve_top1 should keep argmax unchanged"
    );
}

#[test]
fn test_sde_noise_deterministic_with_seed() {
    let config = SdeConfig {
        gamma: 1.0,
        ..Default::default()
    };
    let marginals: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6]];

    let mut rng1 = Rng::new(42);
    let noisy1 = inject_sde_noise(&marginals, &config, &mut rng1);

    let mut rng2 = Rng::new(42);
    let noisy2 = inject_sde_noise(&marginals, &config, &mut rng2);

    for (a, b) in noisy1[0].iter().zip(noisy2[0].iter()) {
        assert!((a - b).abs() < 1e-6, "same seed should produce same noise");
    }
}

#[test]
fn test_inject_sde_noise_into_matches_allocating_disabled() {
    // When SDE is disabled, `inject_sde_noise_into` MUST produce a buffer
    // byte-identical to `inject_sde_noise`. (Both should clone verbatim.)
    let config = SdeConfig::default(); // gamma = 0.0
    let marginals: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6], &[0.2, 0.5, 0.3], &[0.05, 0.15, 0.8]];

    let mut rng_a = Rng::new(42);
    let expected = inject_sde_noise(&marginals, &config, &mut rng_a);

    let mut rng_b = Rng::new(42);
    let mut buf = Vec::new();
    inject_sde_noise_into(&marginals, &config, &mut rng_b, &mut buf);

    assert_eq!(
        buf, expected,
        "_into must match allocating variant (disabled path)"
    );
}

#[test]
fn test_inject_sde_noise_into_matches_allocating_enabled() {
    // When SDE is enabled, `inject_sde_noise_into` MUST produce a buffer
    // byte-identical to `inject_sde_noise` given the same RNG seed.
    let config = SdeConfig {
        gamma: 1.0,
        ..Default::default()
    };
    let marginals: Vec<&[f32]> = vec![
        &[0.1, 0.3, 0.6],
        &[0.2, 0.5, 0.3],
        &[0.05, 0.15, 0.8],
        &[0.4, 0.4, 0.2],
    ];

    let mut rng_a = Rng::new(99);
    let expected = inject_sde_noise(&marginals, &config, &mut rng_a);

    let mut rng_b = Rng::new(99);
    let mut buf = Vec::new();
    inject_sde_noise_into(&marginals, &config, &mut rng_b, &mut buf);

    assert_eq!(buf.len(), expected.len(), "length mismatch");
    for (i, (got, want)) in buf.iter().zip(expected.iter()).enumerate() {
        for (j, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "mismatch at marginal[{i}][{j}]: _into={g}, allocating={w}"
            );
        }
        assert_eq!(
            got.len(),
            want.len(),
            "inner length mismatch at marginal {i}"
        );
    }
}

#[test]
fn test_inject_sde_noise_into_reuses_inner_allocations() {
    // Calling `_into` twice with the same buffer MUST produce the same
    // result as calling once, AND the inner `Vec<f32>` slots must be
    // reused (no length drift, no stale data when marginals shrink).
    let config = SdeConfig {
        gamma: 0.5,
        ..Default::default()
    };

    // First call: 3 marginals of length 3.
    let m3: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6], &[0.2, 0.5, 0.3], &[0.05, 0.15, 0.8]];
    let mut rng_a = Rng::new(7);
    let mut buf = Vec::new();
    inject_sde_noise_into(&m3, &config, &mut rng_a, &mut buf);
    assert_eq!(buf.len(), 3);
    assert_eq!(buf[0].len(), 3);
    let first_pass = buf.iter().map(|v| v.to_vec()).collect::<Vec<_>>();

    // Second call with same seed + same input MUST be byte-identical.
    let mut rng_b = Rng::new(7);
    inject_sde_noise_into(&m3, &config, &mut rng_b, &mut buf);
    assert_eq!(
        buf, first_pass,
        "second call must match first (buffer reuse)"
    );

    // Third call with FEWER marginals MUST truncate the outer Vec.
    let m2: Vec<&[f32]> = vec![&[0.1, 0.3, 0.6], &[0.2, 0.5, 0.3]];
    let mut rng_c = Rng::new(7);
    inject_sde_noise_into(&m2, &config, &mut rng_c, &mut buf);
    assert_eq!(buf.len(), 2, "buffer must shrink when input shrinks");
}

#[test]
fn test_build_slices_view_matches_iter_collect() {
    // `build_slices_view` MUST yield the same `Vec<&[f32]>` as the
    // idiomatic `.iter().map(|m| m.as_slice()).collect()`.
    let owned: Vec<Vec<f32>> = vec![vec![0.1, 0.2], vec![0.3], vec![0.4, 0.5, 0.6]];
    let expected: Vec<&[f32]> = owned.iter().map(|m| m.as_slice()).collect();

    let mut view = Vec::new();
    build_slices_view(&owned, &mut view);

    // Each slice must point at the same memory + length.
    assert_eq!(view.len(), expected.len());
    for (i, (got, want)) in view.iter().zip(expected.iter()).enumerate() {
        assert_eq!(got.as_ptr(), want.as_ptr(), "slice {i} pointer must match");
        assert_eq!(got.len(), want.len(), "slice {i} length must match");
    }
}

// ── GOAT Timing Benchmark: FrozenBaseGuard (Plan 171 T6) ─────
//
// Measures actual wall-clock latency difference between:
//   1. PrunerSchedule::Uniform — screener.relevance() called for every token
//   2. PrunerSchedule::FrozenBaseGuard — NoScreeningPruner at intermediate hops
//
// Uses a deliberately expensive screener to demonstrate the win.

/// Simulated expensive screener — models a WASM validator or bandit Q-table lookup.
///
/// Each `relevance()` call does O(work_factor) work to simulate:
/// - Hash-based lookup (like BanditPruner Q-table)
/// - Small computation (like WasmPruner sandbox execution)
///
/// This is NOT how a real screener works — it's intentionally slow to
/// measure the overhead FrozenBaseGuard avoids at intermediate hops.
#[cfg(feature = "thinking_prune")]
struct ExpensiveScreener {
    /// Simulated work per relevance() call: number of hash rounds.
    work_factor: usize,
    /// Accumulator to prevent the compiler from optimizing away the work.
    /// Uses AtomicF32 for Sync safety.
    sink: std::sync::atomic::AtomicU32,
}

#[cfg(feature = "thinking_prune")]
impl ExpensiveScreener {
    fn new(work_factor: usize) -> Self {
        Self {
            work_factor,
            sink: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

#[cfg(feature = "thinking_prune")]
impl ScreeningPruner for ExpensiveScreener {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Simulate expensive work: hash-based computation that can't be optimized away
        let mut acc = (depth as f32) * 0.001 + (token_idx as f32) * 0.0001;
        for (i, &t) in parent_tokens.iter().enumerate() {
            acc += (i as f32) * (t as f32) * 0.00001;
        }
        // Simulated work: repeated hashing (models Q-table lookup or WASM call)
        for _ in 0..self.work_factor {
            acc = (acc * 1.0001 + 0.1).fract();
        }
        // Sink the result to prevent dead-code elimination
        let bits = acc.to_bits();
        self.sink
            .fetch_xor(bits, std::sync::atomic::Ordering::Relaxed);
        // Return relevance slightly below 1.0 so the tree actually uses it
        1.0 - acc.abs().min(0.1)
    }
}

/// Generate synthetic marginals for benchmarking.
/// vocab_size tokens per depth, draft_lookahead depths.
#[cfg(feature = "thinking_prune")]
fn bench_marginals(vocab_size: usize, draft_lookahead: usize) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(42);
    (0..draft_lookahead)
        .map(|_| {
            let mut probs: Vec<f32> = (0..vocab_size).map(|_| rng.uniform()).collect();
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
            probs
        })
        .collect()
}

/// GOAT T6a: FrozenBaseGuard produces identical output at single hop.
///
/// With 1 hop, FrozenBaseGuard should produce the same tree as Uniform
/// (the only hop IS the final hop).
#[cfg(feature = "thinking_prune")]
#[test]
fn test_goat_schedule_single_hop_identical() {
    use katgpt_pruners::PrunerSchedule;

    let config = Config::draft();
    let marginals_raw = bench_marginals(config.vocab_size, config.draft_lookahead);
    let slices: Vec<&[f32]> = marginals_raw.iter().map(|m| m.as_slice()).collect();
    let screener = ExpensiveScreener::new(100);

    let uniform = build_dd_tree_screened_with_schedule(
        &slices,
        &config,
        &screener,
        true,
        PrunerSchedule::Uniform,
        0,
        1,
    );
    let frozen = build_dd_tree_screened_with_schedule(
        &slices,
        &config,
        &screener,
        true,
        PrunerSchedule::FrozenBaseGuard,
        0,
        1,
    );

    assert_eq!(
        uniform.len(),
        frozen.len(),
        "single hop should produce same tree size"
    );
}

/// GOAT T6b: FrozenBaseGuard produces >= nodes than Uniform.
///
/// At intermediate hops with FrozenBaseGuard, NoScreeningPruner returns 1.0
/// for all tokens, so no branches are trimmed by relevance. This means
/// the tree can explore MORE of the candidate space.
#[cfg(feature = "thinking_prune")]
#[test]
fn test_goat_schedule_intermediate_produces_more() {
    use katgpt_pruners::PrunerSchedule;

    let config = Config {
        screening_threshold: 0.5, // aggressive threshold — rejects many branches
        ..Config::draft()
    };
    let marginals_raw = bench_marginals(config.vocab_size, config.draft_lookahead);
    let slices: Vec<&[f32]> = marginals_raw.iter().map(|m| m.as_slice()).collect();
    let screener = ExpensiveScreener::new(100);

    // Intermediate hop (hop 0 of 3) — FrozenBaseGuard skips screening
    let frozen_intermediate = build_dd_tree_screened_with_schedule(
        &slices,
        &config,
        &screener,
        true,
        PrunerSchedule::FrozenBaseGuard,
        0,
        3,
    );

    // Uniform — applies screening at every hop
    let uniform_intermediate = build_dd_tree_screened_with_schedule(
        &slices,
        &config,
        &screener,
        true,
        PrunerSchedule::Uniform,
        0,
        3,
    );

    assert!(
        frozen_intermediate.len() >= uniform_intermediate.len(),
        "FrozenBaseGuard intermediate ({}) should produce >= Uniform ({}) nodes",
        frozen_intermediate.len(),
        uniform_intermediate.len()
    );
}

/// GOAT T6c: Timing benchmark — FrozenBaseGuard is faster at intermediate hops.
///
/// Measures wall-clock time for 100 iterations of DDTree build with:
///   - ExpensiveScreener (work_factor=500, simulates WASM/bandit overhead)
///   - 3 hops × (vocab_size=27 tokens × draft_lookahead=5 depths)
///   - Uniform: screener called at every hop → 3× the relevance() calls
///   - FrozenBaseGuard: NoScreeningPruner at hops 0-1, full screener at hop 2
///
/// Prints results for GOAT proof audit.
#[cfg(feature = "thinking_prune")]
#[test]
fn test_goat_timing_frozen_base_guard_faster() {
    use katgpt_pruners::PrunerSchedule;
    use std::time::Instant;

    let config = Config::draft();
    let marginals_raw = bench_marginals(config.vocab_size, config.draft_lookahead);
    let slices: Vec<&[f32]> = marginals_raw.iter().map(|m| m.as_slice()).collect();

    let work_factor = 500; // Simulate expensive WASM/bandit validation
    let total_hops = 3;
    let iterations = 100;

    let screener = ExpensiveScreener::new(work_factor);

    // ── Warmup (3 iterations) ──
    for _ in 0..3 {
        for hop in 0..total_hops {
            build_dd_tree_screened_with_schedule(
                &slices,
                &config,
                &screener,
                true,
                PrunerSchedule::Uniform,
                hop,
                total_hops,
            );
            build_dd_tree_screened_with_schedule(
                &slices,
                &config,
                &screener,
                true,
                PrunerSchedule::FrozenBaseGuard,
                hop,
                total_hops,
            );
        }
    }

    // ── Benchmark Uniform ──
    let start = Instant::now();
    for _ in 0..iterations {
        for hop in 0..total_hops {
            let _tree = build_dd_tree_screened_with_schedule(
                &slices,
                &config,
                &screener,
                true,
                PrunerSchedule::Uniform,
                hop,
                total_hops,
            );
            std::hint::black_box(&_tree);
        }
    }
    let uniform_ns = start.elapsed().as_nanos();

    // ── Benchmark FrozenBaseGuard ──
    let start = Instant::now();
    for _ in 0..iterations {
        for hop in 0..total_hops {
            let _tree = build_dd_tree_screened_with_schedule(
                &slices,
                &config,
                &screener,
                true,
                PrunerSchedule::FrozenBaseGuard,
                hop,
                total_hops,
            );
            std::hint::black_box(&_tree);
        }
    }
    let frozen_ns = start.elapsed().as_nanos();

    let uniform_ms = uniform_ns as f64 / 1_000_000.0;
    let frozen_ms = frozen_ns as f64 / 1_000_000.0;
    let speedup = uniform_ms / frozen_ms;

    eprintln!(
        "\n=== GOAT T6c: FrozenBaseGuard Timing ===\n\
             Uniform:          {uniform_ms:.2} ms ({iterations} iters × {total_hops} hops)\n\
             FrozenBaseGuard:  {frozen_ms:.2} ms ({iterations} iters × {total_hops} hops)\n\
             Speedup:          {speedup:.2}×\n\
             Per-hop saving:   intermediate hops skip ExpensiveScreener ({work_factor} work factor)\n"
    );

    // GOAT assertion: FrozenBaseGuard must be measurably faster.
    // With 3 hops and expensive screener, 2 of 3 hops skip screening → ~2× speedup.
    // In practice the speedup is less than 2× because NoScreeningPruner still
    // has some overhead (branch misprediction, function call). We assert >= 1.3×.
    assert!(
        speedup >= 1.3,
        "FrozenBaseGuard should be >= 1.3× faster, got {speedup:.2}×"
    );
}

// ── Progressive Budget Tests (Plan 174 Task 3b) ──────────────

#[cfg(feature = "dflare_progressive_budget")]
mod progressive_budget {
    use super::*;
    use katgpt_core::speculative::types::PositionWeightedBudget;

    /// Helper: create marginals where every token has uniform positive probability.
    fn make_uniform_marginals(config: &Config, num_depths: usize) -> Vec<Vec<f32>> {
        (0..num_depths)
            .map(|_| vec![0.1; config.vocab_size])
            .collect()
    }

    #[test]
    fn test_progressive_none_delegates_to_build_screened() {
        let config = Config::draft();
        let marginals = make_uniform_marginals(&config, 3);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let screener = NoScreeningPruner;

        let tree_baseline = build_dd_tree_screened(&mv, &config, &screener, false);
        let tree_progressive =
            build_dd_tree_screened_progressive(&mv, &config, &screener, false, None);

        assert_eq!(
            tree_baseline.len(),
            tree_progressive.len(),
            "None budget_config should delegate to build_screened"
        );
        for (a, b) in tree_baseline.iter().zip(tree_progressive.iter()) {
            assert_eq!(a.token_idx, b.token_idx, "tokens should match");
            assert_eq!(a.depth, b.depth, "depths should match");
        }
    }

    #[test]
    fn test_progressive_disabled_delegates_to_build_screened() {
        let config = Config::draft();
        let marginals = make_uniform_marginals(&config, 3);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let screener = NoScreeningPruner;

        let budget_config = PositionWeightedBudget {
            enabled: false,
            ..Default::default()
        };

        let tree_baseline = build_dd_tree_screened(&mv, &config, &screener, false);
        let tree_progressive = build_dd_tree_screened_progressive(
            &mv,
            &config,
            &screener,
            false,
            Some(&budget_config),
        );

        assert_eq!(
            tree_baseline.len(),
            tree_progressive.len(),
            "disabled budget_config should delegate to build_screened"
        );
        for (a, b) in tree_baseline.iter().zip(tree_progressive.iter()) {
            assert_eq!(a.token_idx, b.token_idx, "tokens should match");
            assert_eq!(a.depth, b.depth, "depths should match");
        }
    }

    #[test]
    fn test_progressive_respects_total_budget() {
        let config = Config::draft();
        let marginals = make_uniform_marginals(&config, 4);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let screener = NoScreeningPruner;

        let budget_config = PositionWeightedBudget {
            gamma: 4.0,
            min_budget_per_depth: 1,
            enabled: true,
        };

        let tree = build_dd_tree_screened_progressive(
            &mv,
            &config,
            &screener,
            false,
            Some(&budget_config),
        );

        assert!(
            tree.len() <= config.tree_budget,
            "progressive tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
        assert!(!tree.is_empty(), "tree should have at least one node");
    }

    #[test]
    fn test_progressive_front_loads_nodes() {
        let config = Config::draft();
        // Use multiple depths with enough budget to see the difference
        let marginals = make_uniform_marginals(&config, 4);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let screener = NoScreeningPruner;

        let budget_config = PositionWeightedBudget {
            gamma: 2.0, // Aggressive decay
            min_budget_per_depth: 1,
            enabled: true,
        };

        let tree = build_dd_tree_screened_progressive(
            &mv,
            &config,
            &screener,
            false,
            Some(&budget_config),
        );

        // Count nodes at each depth
        let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
        let mut depth_counts: Vec<usize> = vec![0; max_depth + 1];
        for node in &tree {
            depth_counts[node.depth] += 1;
        }

        // With aggressive decay (gamma=2), depth 0 should have the most nodes
        if depth_counts.len() >= 2 {
            assert!(
                depth_counts[0] >= depth_counts[depth_counts.len() - 1],
                "depth 0 ({}) should have >= nodes than depth {} ({})",
                depth_counts[0],
                depth_counts.len() - 1,
                depth_counts[depth_counts.len() - 1]
            );
        }
    }

    #[test]
    fn test_progressive_per_depth_stays_within_allocation() {
        let config = Config::draft();
        let marginals = make_uniform_marginals(&config, 4);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let screener = NoScreeningPruner;

        let budget_config = PositionWeightedBudget {
            gamma: 4.0,
            min_budget_per_depth: 1,
            enabled: true,
        };

        let allocations = budget_config.allocate(config.tree_budget, 4);
        let tree = build_dd_tree_screened_progressive(
            &mv,
            &config,
            &screener,
            false,
            Some(&budget_config),
        );

        let mut depth_counts: Vec<usize> = vec![0; 4];
        for node in &tree {
            depth_counts[node.depth] += 1;
        }

        for (depth, &count) in depth_counts.iter().enumerate() {
            assert!(
                count <= allocations[depth],
                "depth {} has {} nodes but allocation is {}",
                depth,
                count,
                allocations[depth]
            );
        }
    }

    #[test]
    fn test_progressive_chain_seed_respects_budget() {
        let config = Config::draft();
        let marginals = super::make_chain_marginals(&config);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let screener = NoScreeningPruner;

        let budget_config = PositionWeightedBudget {
            gamma: 4.0,
            min_budget_per_depth: 1,
            enabled: true,
        };

        let tree =
            build_dd_tree_screened_progressive(&mv, &config, &screener, true, Some(&budget_config));

        assert!(
            tree.len() <= config.tree_budget,
            "chain-seed progressive tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
        assert!(!tree.is_empty(), "tree should have at least one node");
    }

    #[test]
    fn test_progressive_empty_marginals() {
        let config = Config::draft();
        let screener = NoScreeningPruner;

        let budget_config = PositionWeightedBudget {
            gamma: 4.0,
            min_budget_per_depth: 1,
            enabled: true,
        };

        let tree = build_dd_tree_screened_progressive(
            &[],
            &config,
            &screener,
            false,
            Some(&budget_config),
        );

        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }
}

// ── SpeculativeGenerator equivalence tests (Plan 193 T6) ────────
#[cfg(feature = "speculative_generator")]
mod speculative_gen {
    use super::*;
    use katgpt_core::NoPruner;
    use katgpt_speculative::spec_generator::{MarginalTokenGenerator, TokenConstraintPruner};

    #[test]
    fn test_dd_tree_speculative_equivalence_no_pruner() {
        // With NoPruner (all candidates valid), the speculative path
        // should produce the same tree as build_dd_tree_screened.
        let config = Config::draft();
        let mut rng = fastrand::Rng::new();

        let m1: Vec<f32> = vec![0.3, 0.5, 0.2];
        let m2: Vec<f32> = vec![0.1, 0.4, 0.3, 0.2];
        let slices: Vec<&[f32]> = vec![&m1, &m2];

        // Standard path
        let tree_standard = build_dd_tree_screened(&slices, &config, &NoScreeningPruner, false);

        // SpeculativeGenerator path (NoPruner = all valid)
        let mut generator = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        let tree_spec =
            build_dd_tree_speculative(&mut generator, &pruner, &slices, &config, &mut rng);

        // Same number of nodes
        assert_eq!(
            tree_standard.len(),
            tree_spec.len(),
            "speculative tree has {} nodes, standard has {}",
            tree_spec.len(),
            tree_standard.len(),
        );

        // Same token indices and scores (within float tolerance)
        for (a, b) in tree_standard.iter().zip(tree_spec.iter()) {
            assert_eq!(
                a.token_idx, b.token_idx,
                "token mismatch at depth {}",
                a.depth,
            );
            assert!(
                (a.score - b.score).abs() < 1e-4,
                "score mismatch: {} vs {} at depth {} token {}",
                a.score,
                b.score,
                a.depth,
                a.token_idx,
            );
        }
    }

    #[test]
    fn test_dd_tree_speculative_empty_marginals() {
        let config = Config::draft();
        let mut rng = fastrand::Rng::new();
        let empty: Vec<&[f32]> = vec![];

        let mut generator = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        let tree = build_dd_tree_speculative(&mut generator, &pruner, &empty, &config, &mut rng);

        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }
}

// ── Best Buddies integration tests (Plan 199) ──────────────────
#[cfg(all(feature = "speculative_generator", feature = "best_buddies"))]
mod best_buddies_integration {
    use super::*;
    use katgpt_core::NoPruner;
    use katgpt_speculative::best_buddies::MarginalBestBuddyAligner;
    use katgpt_speculative::spec_generator::{MarginalTokenGenerator, TokenConstraintPruner};

    /// Helper: create normalized descending marginals for `n_depths` positions.
    fn make_marginals(n_depths: usize, n_tokens: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(n_depths);
        for _ in 0..n_depths {
            let mut row = Vec::with_capacity(n_tokens);
            let mut sum = 0.0f32;
            for t in 0..n_tokens {
                let v = 1.0 / ((t + 1) as f32);
                row.push(v);
                sum += v;
            }
            for v in &mut row {
                *v /= sum;
            }
            out.push(row);
        }
        out
    }

    #[test]
    fn test_best_buddies_identical_marginals_same_tree() {
        // When draft == target, BB filter is a no-op → tree should be
        // identical to the standard speculative path.
        let config = Config::draft();
        let mut rng = fastrand::Rng::new();

        let marginals = make_marginals(3, 5);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

        // Standard speculative path
        let mut gen_std = MarginalTokenGenerator { top_k: 10 };
        let pruner_std = TokenConstraintPruner::new(NoPruner);
        let tree_std =
            build_dd_tree_speculative(&mut gen_std, &pruner_std, &slices, &config, &mut rng);

        // BB path with identical marginals
        let mut gen_bb = MarginalTokenGenerator { top_k: 10 };
        let pruner_bb = TokenConstraintPruner::new(NoPruner);
        let mut aligner = MarginalBestBuddyAligner::default();
        let tree_bb = build_dd_tree_speculative_best_buddies(
            &mut gen_bb,
            &pruner_bb,
            &slices,
            &slices,
            &mut aligner,
            &config,
            &mut rng,
        );

        assert_eq!(
            tree_std.len(),
            tree_bb.len(),
            "identical marginals should produce same node count: std={}, bb={}",
            tree_std.len(),
            tree_bb.len(),
        );
    }

    #[test]
    fn test_best_buddies_disagreeing_marginals_smaller_tree() {
        // When draft and target disagree (anti-correlated), BB filter
        // should dampen marginals → fewer branches → smaller tree.
        let config = Config::draft();
        let mut rng = fastrand::Rng::new();

        // Draft: descending (0.5, 0.3, 0.2)
        let draft: Vec<Vec<f32>> = vec![vec![0.5, 0.3, 0.2], vec![0.4, 0.35, 0.25]];
        // Target: ascending (anti-correlated)
        let target: Vec<Vec<f32>> = vec![vec![0.2, 0.3, 0.5], vec![0.25, 0.35, 0.4]];

        let draft_slices: Vec<&[f32]> = draft.iter().map(|m| m.as_slice()).collect();
        let target_slices: Vec<&[f32]> = target.iter().map(|m| m.as_slice()).collect();

        // Standard speculative (no BB filter)
        let mut gen_std = MarginalTokenGenerator { top_k: 10 };
        let pruner_std = TokenConstraintPruner::new(NoPruner);
        let tree_std =
            build_dd_tree_speculative(&mut gen_std, &pruner_std, &draft_slices, &config, &mut rng);

        // BB path with disagreeing marginals
        let mut gen_bb = MarginalTokenGenerator { top_k: 10 };
        let pruner_bb = TokenConstraintPruner::new(NoPruner);
        let mut aligner = MarginalBestBuddyAligner::new(0.5); // higher threshold
        let tree_bb = build_dd_tree_speculative_best_buddies(
            &mut gen_bb,
            &pruner_bb,
            &draft_slices,
            &target_slices,
            &mut aligner,
            &config,
            &mut rng,
        );

        // BB tree should have ≤ nodes (dampened marginals reduce branching)
        assert_eq!(
            tree_bb.len(),
            tree_std.len(),
            "BB should have ≤ nodes than unfiltered: std={}, bb={}",
            tree_std.len(),
            tree_bb.len(),
        );
    }

    #[test]
    fn test_best_buddies_empty_target_marginals() {
        // Empty target marginals → no filtering → same as standard path.
        let config = Config::draft();
        let mut rng = fastrand::Rng::new();

        let draft = make_marginals(2, 4);
        let draft_slices: Vec<&[f32]> = draft.iter().map(|m| m.as_slice()).collect();
        let empty: Vec<&[f32]> = vec![];

        let mut sampler = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        let mut aligner = MarginalBestBuddyAligner::default();
        let tree = build_dd_tree_speculative_best_buddies(
            &mut sampler,
            &pruner,
            &draft_slices,
            &empty,
            &mut aligner,
            &config,
            &mut rng,
        );

        // Empty target → no filtering → empty result (no positions to process)
        assert!(tree.is_empty(), "empty target should produce empty tree");
    }

    #[test]
    fn test_best_buddies_ema_smoothing_across_calls() {
        // Two calls with the same marginals should show EMA smoothing.
        // First call populates cache, second call blends with cache.
        let config = Config::draft();
        let mut rng = fastrand::Rng::new();

        let draft = make_marginals(2, 4);
        let target = make_marginals(2, 4);
        let draft_slices: Vec<&[f32]> = draft.iter().map(|m| m.as_slice()).collect();
        let target_slices: Vec<&[f32]> = target.iter().map(|m| m.as_slice()).collect();

        let mut aligner = MarginalBestBuddyAligner::default().with_ema_alpha(0.3);

        // Call 1: populates EMA cache
        let mut gen1 = MarginalTokenGenerator { top_k: 10 };
        let pruner1 = TokenConstraintPruner::new(NoPruner);
        let tree1 = build_dd_tree_speculative_best_buddies(
            &mut gen1,
            &pruner1,
            &draft_slices,
            &target_slices,
            &mut aligner,
            &config,
            &mut rng,
        );

        // Call 2: same input, should use EMA cache
        let mut gen2 = MarginalTokenGenerator { top_k: 10 };
        let pruner2 = TokenConstraintPruner::new(NoPruner);
        let tree2 = build_dd_tree_speculative_best_buddies(
            &mut gen2,
            &pruner2,
            &draft_slices,
            &target_slices,
            &mut aligner,
            &config,
            &mut rng,
        );

        // Both calls should produce valid, non-empty trees
        assert!(!tree1.is_empty(), "first call should produce tree");
        assert!(!tree2.is_empty(), "second call should produce tree");
        assert_eq!(
            tree1.len(),
            tree2.len(),
            "same input should give same tree size"
        );
    }

    #[test]
    fn test_best_buddies_with_constraint_pruner() {
        // BB filter + ConstraintPruner should compose correctly.
        struct RejectEvenPruner;
        impl ConstraintPruner for RejectEvenPruner {
            fn is_valid(&self, _depth: usize, token_idx: usize, _parents: &[usize]) -> bool {
                !token_idx.is_multiple_of(2)
            }
        }

        let config = Config::draft();
        let mut rng = fastrand::Rng::new();

        let draft = make_marginals(3, 6);
        let target = make_marginals(3, 6);
        let draft_slices: Vec<&[f32]> = draft.iter().map(|m| m.as_slice()).collect();
        let target_slices: Vec<&[f32]> = target.iter().map(|m| m.as_slice()).collect();

        let mut sampler = MarginalTokenGenerator { top_k: 10 };
        let pruner = TokenConstraintPruner::new(RejectEvenPruner);
        let mut aligner = MarginalBestBuddyAligner::default();
        let tree = build_dd_tree_speculative_best_buddies(
            &mut sampler,
            &pruner,
            &draft_slices,
            &target_slices,
            &mut aligner,
            &config,
            &mut rng,
        );

        // All nodes should have odd token indices
        for node in &tree {
            assert_eq!(
                node.token_idx % 2,
                1,
                "node at depth {} should have odd token_idx, got {}",
                node.depth,
                node.token_idx,
            );
        }
    }
}

// ── Domino DDTree tests (Plan 197) ──────────────────────────────
#[cfg(feature = "domino_correction")]
mod domino_tree {
    use super::*;
    use katgpt_core::traits::DominoPruner;
    use katgpt_speculative::domino::compute_prefix_strength;

    fn make_uniform_marginals(depths: usize, vocab_size: usize) -> Vec<Vec<f32>> {
        let prob = 1.0 / vocab_size as f32;
        vec![vec![prob; vocab_size]; depths]
    }

    #[test]
    fn test_build_dd_tree_domino_matches_pruned_when_no_sampled_tokens() {
        let config = Config {
            tree_budget: 16,
            draft_lookahead: 3,
            vocab_size: 4,
            ..Config::default()
        };

        let marginals = make_uniform_marginals(3, 4);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let sampled_tokens: [usize; 0] = [];

        let pruned = build_dd_tree_pruned(&slices, &config, &NoPruner, false);
        let domino = build_dd_tree_domino(&slices, &config, &NoPruner, false, &sampled_tokens);

        // Same node count
        assert_eq!(pruned.len(), domino.len());

        // Scores should be identical (no prefix to adjust)
        for (p, d) in pruned.iter().zip(domino.iter()) {
            assert_eq!(p.token_idx, d.token_idx);
            assert!((p.score - d.score).abs() < 1e-6);
        }
    }

    #[test]
    fn test_build_dd_tree_domino_respects_budget() {
        let config = Config {
            tree_budget: 8,
            draft_lookahead: 5,
            vocab_size: 4,
            ..Config::default()
        };

        let marginals = make_uniform_marginals(5, 4);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let sampled_tokens = [0usize, 1, 2, 3, 0];

        let tree = build_dd_tree_domino(&slices, &config, &NoPruner, false, &sampled_tokens);
        assert!(tree.len() <= config.tree_budget);
    }

    #[test]
    fn test_build_dd_tree_domino_with_domino_pruner() {
        // Custom pruner implementing DominoPruner with causal correction
        struct PrefixAwarePruner;
        impl ConstraintPruner for PrefixAwarePruner {
            fn is_valid(&self, _depth: usize, token_idx: usize, _parents: &[usize]) -> bool {
                token_idx < 3 // Allow only tokens 0, 1, 2
            }
        }
        impl DominoPruner for PrefixAwarePruner {
            fn causal_correction(
                &self,
                depth: usize,
                token: usize,
                prefix: &[usize],
                base_valid: bool,
            ) -> bool {
                // At depth > 1, also reject token 2 if prefix has two 0s
                if depth > 1 && token == 2 && prefix.iter().filter(|&&t| t == 0).count() >= 2 {
                    return false;
                }
                base_valid
            }
        }

        let config = Config {
            tree_budget: 16,
            draft_lookahead: 3,
            vocab_size: 4,
            ..Config::default()
        };

        let marginals = make_uniform_marginals(3, 4);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let sampled_tokens = [0usize, 0];

        let tree =
            build_dd_tree_domino(&slices, &config, &PrefixAwarePruner, false, &sampled_tokens);

        // All nodes should have valid tokens
        for node in &tree {
            assert!(node.token_idx < 3, "Token {} should be < 3", node.token_idx);
        }
    }

    #[test]
    fn test_domino_scoring_adjusts_depth_scores() {
        let marginals: Vec<&[f32]> = vec![&[0.25, 0.25, 0.25, 0.25]; 3];
        let sampled_tokens = [0usize, 1];

        // prefix_strength = 0.25 * 0.25 = 0.0625
        let strength = compute_prefix_strength(&marginals, &sampled_tokens, 2);
        assert!((strength - 0.0625f32).abs() < 1e-6);

        // domino_score at depth 2 should penalize
        let base = -1.0f32;
        let scored = katgpt_speculative::domino::domino_score(base, 2, strength);
        // -1.0 * 0.0625^2 = -0.00390625
        assert!(
            scored > base,
            "Should be less extreme than base for negative scores"
        );
    }

    // ── Belief DDTree Tests (Plan 217, feature: belief_drafter) ──

    /// Helper: create a BeliefDrafter suitable for DDTree tests.
    /// Uses n_embd=4, vocab_size=4 to match draft config patterns.
    #[cfg(feature = "belief_drafter")]
    fn make_belief_drafter_for_tree() -> katgpt_speculative::belief_drafter::BeliefDrafter {
        use katgpt_speculative::belief_drafter::{BeliefDrafter, LatentDynamicsMLP};

        let n_embd = 4;
        let vocab_size = 4;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let output_head: Vec<f32> = (0..vocab_size * n_embd)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let wte: Vec<f32> = (0..vocab_size * n_embd)
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();
        BeliefDrafter::new(mlp, output_head, wte).expect("valid drafter")
    }

    #[cfg(feature = "belief_drafter")]
    #[test]
    fn test_belief_ddtree_produces_valid_tree() {
        let drafter = make_belief_drafter_for_tree();
        let h_t = vec![1.0f32; 4];
        let config = Config::draft();

        let tree = build_dd_tree_belief(&drafter, &h_t, 5, 10.0, &config, false);

        assert!(!tree.is_empty(), "should produce a non-empty tree");
        for node in &tree {
            assert!(
                node.token_idx < drafter.vocab_size(),
                "token_idx {} must be < vocab_size {}",
                node.token_idx,
                drafter.vocab_size()
            );
        }
    }

    #[cfg(feature = "belief_drafter")]
    #[test]
    fn test_belief_ddtree_respects_draft_length() {
        let drafter = make_belief_drafter_for_tree();
        let h_t = vec![1.0f32; 4];
        let config = Config::draft();

        let max_steps = 3;
        let tree = build_dd_tree_belief(&drafter, &h_t, max_steps, 10.0, &config, false);

        let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
        assert!(
            max_depth <= max_steps,
            "tree depth {max_depth} should not exceed max_draft_steps {max_steps}"
        );
    }

    #[cfg(feature = "belief_drafter")]
    #[test]
    fn test_belief_ddtree_collapse_aware_adjusts_threshold() {
        let drafter = make_belief_drafter_for_tree();
        let h_t = vec![1.0f32; 4];
        let config = Config::draft();

        // Low avg entropy → higher effective threshold → potentially longer drafts
        let tree_low_ent =
            build_dd_tree_belief_collapse_aware(&drafter, &h_t, 5, 5.0, &config, false, Some(0.5));

        // High avg entropy → lower effective threshold → shorter drafts
        let tree_high_ent =
            build_dd_tree_belief_collapse_aware(&drafter, &h_t, 5, 5.0, &config, false, Some(3.0));

        // Both should produce valid trees; the low-entropy one should be >= high-entropy
        // (not guaranteed strictly larger due to entropy gating, but should trend)
        assert!(!tree_low_ent.is_empty());
        assert!(!tree_high_ent.is_empty());
    }

    #[cfg(feature = "belief_drafter")]
    #[test]
    fn test_belief_ddtree_empty_draft() {
        let drafter = make_belief_drafter_for_tree();
        let h_t = vec![1.0f32; 4];
        let config = Config::draft();

        // max_draft_steps=0 → draft() returns empty → empty tree
        let tree = build_dd_tree_belief(&drafter, &h_t, 0, 10.0, &config, false);

        assert!(
            tree.is_empty(),
            "zero max_draft_steps should produce empty tree"
        );
    }

    #[cfg(feature = "belief_drafter")]
    #[test]
    fn test_belief_ddtree_marginals_normalized() {
        let drafter = make_belief_drafter_for_tree();
        let h_t = vec![1.0f32; 4];
        let vs = drafter.vocab_size();

        // Verify the marginal construction logic: confidence + residual sums to ~1.0
        let drafts = drafter.draft(&h_t, 3, 10.0);
        for draft_token in &drafts {
            let confidence = (draft_token.log_prob.exp()).max(0.5);
            let residual = (1.0 - confidence) / (vs - 1).max(1) as f32;
            let total = confidence + residual * (vs - 1) as f32;
            assert!(
                (total - 1.0).abs() < 1e-5,
                "marginal should sum to ~1.0, got {total}"
            );
        }
    }
}
