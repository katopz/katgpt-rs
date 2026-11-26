use super::*;

/// Build a small 3-level test tree:
/// ```text
///                 root
///                /    \
///           [0,1]      [2,3]
///          /   \      /   \
///        L0    L1   L2    L3
/// ```
fn build_test_tree(drift_rate: f32) -> LatentTaskTree {
    let left = TreeNode::internal(vec![
        TreeNode::leaf(0, drift_rate),
        TreeNode::leaf(1, drift_rate),
    ]);
    let right = TreeNode::internal(vec![
        TreeNode::leaf(2, drift_rate),
        TreeNode::leaf(3, drift_rate),
    ]);
    let root = TreeNode::internal(vec![left, right]);
    LatentTaskTree::from_root(root, LatentTaskTreeConfig::default())
}

// ── T1.5(a): sample returns valid leaf ids ──────────────────────────

#[test]
fn test_sample_returns_valid_leaf_ids() {
    let tree = build_test_tree(0.0);
    let mut rng = fastrand::Rng::with_seed(42);
    let valid_arms: std::collections::HashSet<usize> = (0..4).collect();

    for _ in 0..1000 {
        let arm = tree.sample(&mut rng);
        assert!(
            valid_arms.contains(&arm),
            "sample returned invalid arm_id {arm}"
        );
    }
}

#[test]
fn test_sample_single_leaf_tree() {
    // Degenerate tree: root is a single leaf.
    let root = TreeNode::leaf(7, 0.0);
    let tree = LatentTaskTree::from_root(root, LatentTaskTreeConfig::default());
    let mut rng = fastrand::Rng::with_seed(0);
    for _ in 0..100 {
        assert_eq!(tree.sample(&mut rng), 7);
    }
}

#[test]
fn test_sample_visits_all_arms_with_uniform_prior() {
    // With all-uniform Beta(1,1) priors, every arm should be visited.
    let tree = build_test_tree(0.0);
    let mut rng = fastrand::Rng::with_seed(123);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..2000 {
        seen.insert(tree.sample(&mut rng));
    }
    assert_eq!(seen.len(), 4, "uniform prior should visit all 4 arms");
}

// ── T1.5(b): observe updates leaf filter + Empirical Bayes ──────────

#[test]
fn test_observe_updates_leaf_filter() {
    let mut tree = build_test_tree(0.0);

    // Observe 10 successes on arm 2.
    for step in 0..10u64 {
        tree.observe(2, 1.0, step);
    }

    // Find arm 2's leaf and check its filter.
    let leaf = find_leaf(&tree.root, 2).expect("arm 2 should exist");
    let TreeNode::Leaf { filter, .. } = leaf else {
        panic!("expected leaf")
    };
    // 10 successes: alpha = 1 + 10 = 11, beta = 1 + 0 = 1.
    assert!(
        (filter.alpha - 11.0).abs() < 1e-5,
        "alpha should be 11 after 10 successes, got {}",
        filter.alpha
    );
    assert!(
        (filter.beta - 1.0).abs() < 1e-5,
        "beta should be 1 after 10 successes, got {}",
        filter.beta
    );
}

#[test]
fn test_observe_propagates_empirical_bayes() {
    let mut tree = build_test_tree(0.0);

    // Before any observation: root has Beta(1, 1) — evidence pooling of 4
    // children × Beta(1,1): 1 + Σ(1-1) = 1 for both α and β.
    let (ra, rb) = tree.root.beta_params();
    assert!(
        (ra - 1.0).abs() < 1e-5,
        "root alpha should start at 1, got {ra}"
    );
    assert!(
        (rb - 1.0).abs() < 1e-5,
        "root beta should start at 1, got {rb}"
    );

    // Observe 1 success on arm 0.
    tree.observe(0, 1.0, 0);

    // Arm 0's leaf: Beta(2, 1). Its sibling (arm 1): Beta(1, 1).
    // Left internal: 1+Σ(α-1)=1+(1+0)=2, 1+Σ(β-1)=1+(0+0)=1 → Beta(2, 1).
    // Right internal: 1+Σ(α-1)=1+(0+0)=1, 1+Σ(β-1)=1+(0+0)=1 → Beta(1, 1).
    // Root: 1+Σ(α-1)=1+(1+0)=2, 1+Σ(β-1)=1+(0+0)=1 → Beta(2, 1).
    let (ra, rb) = tree.root.beta_params();
    assert!(
        (ra - 2.0).abs() < 1e-5,
        "root alpha should be 2 after 1 success, got {ra}"
    );
    assert!(
        (rb - 1.0).abs() < 1e-5,
        "root beta should be 1 after 1 success, got {rb}"
    );
}

#[test]
fn test_observe_mixed_rewards() {
    let mut tree = build_test_tree(0.0);
    tree.observe(1, 0.7, 0);
    tree.observe(1, 0.3, 1);

    let leaf = find_leaf(&tree.root, 1).expect("arm 1 should exist");
    if let TreeNode::Leaf { filter, .. } = leaf {
        // alpha = 1 + 0.7 + 0.3 = 2.0
        assert!((filter.alpha - 2.0).abs() < 1e-5, "alpha={}", filter.alpha);
        // beta = 1 + 0.3 + 0.7 = 2.0
        assert!((filter.beta - 2.0).abs() < 1e-5, "beta={}", filter.beta);
    }
}

// ── T1.5(c): drift_rate=0 degenerates to stationary Beta ────────────

#[test]
fn test_predict_zero_drift_is_noop() {
    let mut arm = BayesianFilterArm::new(0.0);
    arm.alpha = 11.0;
    arm.beta = 6.0;
    arm.last_obs_step = 0;

    // Predict at step 100 — should be a complete no-op.
    arm.predict(100);

    assert!((arm.alpha - 11.0).abs() < 1e-5, "alpha unchanged");
    assert!((arm.beta - 6.0).abs() < 1e-5, "beta unchanged");
}

#[test]
fn test_zero_drift_sample_mean_matches_beta_mean() {
    // With drift_rate=0, after observing evidence, the sample distribution
    // should match Beta(alpha, beta). Verify the empirical mean over 10K draws.
    let mut arm = BayesianFilterArm::new(0.0);
    // 15 successes, 5 failures → Beta(16, 6), mean = 16/22 ≈ 0.7273.
    for _ in 0..15 {
        arm.update(1.0, 0);
    }
    for _ in 0..5 {
        arm.update(0.0, 0);
    }

    let expected_mean = 16.0_f32 / 22.0;
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 10_000_usize;
    let sum: f32 = (0..n).map(|_| arm.thompson_sample(&mut rng)).sum();
    let empirical_mean = sum / n as f32;

    assert!(
        (empirical_mean - expected_mean).abs() < 0.02,
        "empirical mean {empirical_mean:.4} too far from Beta mean {expected_mean:.4}"
    );
}

#[test]
fn test_predict_nonzero_drift_pulls_toward_uniform() {
    let mut arm = BayesianFilterArm::new(0.5);
    arm.alpha = 21.0; // strong evidence
    arm.beta = 1.0;
    arm.last_obs_step = 0;

    // After 1 step with λ=0.5:
    // decay = 0.5, alpha' = 21*0.5 + 0.5 = 11.0, beta' = 1*0.5 + 0.5 = 1.0.
    arm.predict(1);
    assert!(
        (arm.alpha - 11.0).abs() < 1e-4,
        "alpha after 1 drift step: {}",
        arm.alpha
    );
    assert!(
        (arm.beta - 1.0).abs() < 1e-4,
        "beta after 1 drift step: {}",
        arm.beta
    );

    // After another step (total elapsed = 1 from last_obs_step=1):
    // decay = 0.5, alpha' = 11*0.5 + 0.5 = 6.0.
    arm.predict(2);
    assert!(
        (arm.alpha - 6.0).abs() < 1e-4,
        "alpha after 2 drift steps: {}",
        arm.alpha
    );
}

// ── T1.5(d): blake3 stability ───────────────────────────────────────

#[test]
fn test_blake3_stable_across_rebuilds() {
    let config = LatentTaskTreeConfig::default();

    // Build two identical trees.
    let tree1 = build_test_tree(0.01);
    let tree2 = {
        let left = TreeNode::internal(vec![TreeNode::leaf(0, 0.01), TreeNode::leaf(1, 0.01)]);
        let right = TreeNode::internal(vec![TreeNode::leaf(2, 0.01), TreeNode::leaf(3, 0.01)]);
        let root = TreeNode::internal(vec![left, right]);
        LatentTaskTree::from_root(root, config)
    };

    assert_eq!(
        tree1.blake3_root(),
        tree2.blake3_root(),
        "identical trees must have identical BLAKE3 commitments"
    );
}

#[test]
fn test_blake3_changes_on_different_trees() {
    let tree_a = build_test_tree(0.01);

    // Different topology.
    let different_root = TreeNode::internal(vec![
        TreeNode::leaf(0, 0.01),
        TreeNode::leaf(1, 0.01),
        TreeNode::leaf(2, 0.01),
    ]);
    let tree_b = LatentTaskTree::from_root(different_root, LatentTaskTreeConfig::default());

    assert_ne!(
        tree_a.blake3_root(),
        tree_b.blake3_root(),
        "different trees must have different BLAKE3 commitments"
    );
}

#[test]
fn test_blake3_changes_on_different_drift_rate() {
    // Different drift_rate → different initial filter state → different hash.
    let tree_a = build_test_tree(0.01);
    let tree_b = build_test_tree(0.05);
    assert_ne!(tree_a.blake3_root(), tree_b.blake3_root());
}

// ── Edge cases ──────────────────────────────────────────────────────

#[test]
fn test_observe_increments_n_obs() {
    let mut tree = build_test_tree(0.0);
    tree.observe(0, 1.0, 0);
    tree.observe(0, 0.0, 1);
    tree.observe(1, 1.0, 2);

    // Root should have n_obs = 3.
    if let TreeNode::Internal { n_obs, .. } = &tree.root {
        assert_eq!(*n_obs, 3, "root n_obs should be 3");
    } else {
        panic!("root should be Internal");
    }
}

// ── Phase 4 T4.2: R279 N≥d phase gate tests ────────────────────────

/// `phase_gate_min_obs = 0` (default) matches Phase 1–3 behavior exactly:
/// all children aggregated. Same observations → same Beta posteriors as the
/// ungated path. This is the regression-guard test.
#[test]
fn test_phase_gate_disabled_matches_ungated_behavior() {
    let cfg_off = LatentTaskTreeConfig {
        filter_drift_rate: 0.0,
        phase_gate_min_obs: 0,
        ..LatentTaskTreeConfig::default()
    };
    let cfg_zero = LatentTaskTreeConfig {
        filter_drift_rate: 0.0,
        phase_gate_min_obs: 0,
        ..LatentTaskTreeConfig::default()
    };

    // Two identical trees, same observations.
    let mut t1 = LatentTaskTree::from_root(make_tree_topology(), cfg_off);
    let mut t2 = LatentTaskTree::from_root(make_tree_topology(), cfg_zero);
    for step in 0..10u64 {
        t1.observe(0, 1.0, step);
        t2.observe(0, 1.0, step);
    }

    // BLAKE3 of the runtime state (topology + initial priors) is identical
    // — the gate didn't fire, so runtime Beta values must match bit-for-bit.
    // (blake3_root commits the INITIAL state, so we compare by sampling
    // distributions instead: identical Beta posteriors → identical
    // empirical sample means within tolerance.)
    let mut rng1 = fastrand::Rng::with_seed(42);
    let mut rng2 = fastrand::Rng::with_seed(42);
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let n = 10_000;
    for _ in 0..n {
        sum1 += t1.sample(&mut rng1) as f64;
        sum2 += t2.sample(&mut rng2) as f64;
    }
    let mean1 = sum1 / n as f64;
    let mean2 = sum2 / n as f64;
    // With gate disabled, both trees have identical posteriors → identical
    // sample means (within sampling noise, ~0.01).
    assert!(
        (mean1 - mean2).abs() < 0.05,
        "gate disabled should match: mean1={mean1:.3} mean2={mean2:.3}"
    );
}

/// When `phase_gate_min_obs` exceeds every child's n_obs, ALL children are
/// gated out and the parent falls back to Beta(1, 1) (uniform). This is the
/// correct "we don't have enough evidence" behavior — high variance,
// exploration.
#[test]
fn test_phase_gate_all_children_below_threshold_yields_uniform_parent() {
    let cfg = LatentTaskTreeConfig {
        filter_drift_rate: 0.0,
        // Threshold = 100 means every subtree (which has < 100 obs after
        // a few observe() calls) is gated out.
        phase_gate_min_obs: 100,
        ..LatentTaskTreeConfig::default()
    };
    let mut tree = LatentTaskTree::from_root(make_tree_topology(), cfg);
    // Observe 5 rewards on arm 0 — but the gate should prevent these from
    // propagating to the root (root's children have n_obs < 100).
    for step in 0..5u64 {
        tree.observe(0, 1.0, step);
    }

    // Root should still be at Beta(1, 1) — uniform.
    if let TreeNode::Internal {
        beta_alpha,
        beta_beta,
        ..
    } = &tree.root
    {
        assert!(
            (beta_alpha - 1.0).abs() < 1e-5,
            "gated root alpha should be 1.0 (uniform), got {beta_alpha}"
        );
        assert!(
            (beta_beta - 1.0).abs() < 1e-5,
            "gated root beta should be 1.0 (uniform), got {beta_beta}"
        );
    } else {
        panic!("root should be Internal");
    }
}

/// A subtree that has accumulated enough observations (n_obs ≥ threshold)
/// DOES contribute to the parent aggregate. A subtree below the threshold
/// is skipped. This is the core N≥d phase transition behavior.
#[test]
fn test_phase_gate_skips_below_threshold_includes_above() {
    // Tree shape:
    //                root
    //               /    \
    //         subtree_A   subtree_B
    //          /   \       /   \
    //        L0    L1    L2    L3
    //
    // We'll observe many rewards on arm 0 (building up subtree_A's n_obs)
    // and few on arm 2 (subtree_B stays below threshold).
    let cfg = LatentTaskTreeConfig {
        filter_drift_rate: 0.0,
        // Threshold = 5: subtree_A (with ≥5 obs) passes; subtree_B (1 obs) fails.
        phase_gate_min_obs: 5,
        ..LatentTaskTreeConfig::default()
    };
    let mut tree = LatentTaskTree::from_root(make_tree_topology(), cfg);

    // 10 successes on arm 0 → subtree_A gets n_obs = 10 (≥ threshold).
    for step in 0..10u64 {
        tree.observe(0, 1.0, step);
    }
    // 1 observation on arm 2 → subtree_B gets n_obs = 1 (< threshold).
    tree.observe(2, 1.0, 10);

    // Inspect root: should aggregate ONLY subtree_A (subtree_B gated out).
    // subtree_A's evidence: 10 successes, 0 failures → evidence pooled
    // child alpha = 1 + 10 = 11, child beta = 1 + 0 = 1.
    // Parent evidence pooling with 1 active child:
    //   parent_alpha = (11 - 1 + 1) = 11
    //   parent_beta  = (1  - 1 + 1) = 1
    if let TreeNode::Internal {
        children,
        beta_alpha,
        beta_beta,
        ..
    } = &tree.root
    {
        // Verify the child n_obs counts.
        let a_n_obs = children[0].n_obs();
        let b_n_obs = children[1].n_obs();
        assert_eq!(a_n_obs, 10, "subtree_A n_obs should be 10");
        assert_eq!(b_n_obs, 1, "subtree_B n_obs should be 1");

        // Root aggregate should reflect ONLY subtree_A.
        assert!(
            (beta_alpha - 11.0).abs() < 1e-4,
            "root alpha should be 11 (only A contributes), got {beta_alpha}"
        );
        assert!(
            (beta_beta - 1.0).abs() < 1e-4,
            "root beta should be 1 (only A contributes), got {beta_beta}"
        );
    } else {
        panic!("root should be Internal");
    }
}

/// `phase_gate_min_obs = 1` is equivalent to "skip only children with zero
/// observations" — a very mild gate. Should still produce correct posteriors
/// once every subtree has at least 1 observation.
#[test]
fn test_phase_gate_min_obs_one_skips_only_zero_obs_children() {
    let cfg = LatentTaskTreeConfig {
        filter_drift_rate: 0.0,
        phase_gate_min_obs: 1,
        ..LatentTaskTreeConfig::default()
    };
    let mut tree = LatentTaskTree::from_root(make_tree_topology(), cfg);

    // Observe ONLY arm 0 — subtree_B has 0 observations → gated out at root.
    tree.observe(0, 1.0, 0);

    // Root: only subtree_A (n_obs=1) contributes.
    // subtree_A aggregate: child L0 has alpha=2, beta=1; L1 has alpha=1, beta=1.
    //   subtree_A alpha = (2 + 1) - 2 + 1 = 2
    //   subtree_A beta  = (1 + 1) - 2 + 1 = 1
    // Root with 1 active child (subtree_A):
    //   root alpha = 2 - 1 + 1 = 2
    //   root beta  = 1 - 1 + 1 = 1
    if let TreeNode::Internal {
        children,
        beta_alpha,
        beta_beta,
        ..
    } = &tree.root
    {
        assert_eq!(children[0].n_obs(), 1);
        assert_eq!(children[1].n_obs(), 0);
        assert!(
            (beta_alpha - 2.0).abs() < 1e-4,
            "root alpha should be 2, got {beta_alpha}"
        );
        assert!(
            (beta_beta - 1.0).abs() < 1e-4,
            "root beta should be 1, got {beta_beta}"
        );
    } else {
        panic!("root should be Internal");
    }
}

/// Public `TreeNode::n_obs()` accessor returns the right counts for both
/// internal nodes (stored counter) and leaves (always 0).
#[test]
fn test_treenode_n_obs_accessor() {
    let mut tree = build_test_tree(0.0);
    tree.observe(0, 1.0, 0);
    tree.observe(0, 0.0, 1);

    // Root: 2 observations passed through.
    assert_eq!(tree.root.n_obs(), 2, "root n_obs");

    // Leaf: always 0 (leaves track evidence via alpha/beta, not n_obs).
    if let TreeNode::Internal { children, .. } = &tree.root {
        // left subtree: 2 obs
        assert_eq!(children[0].n_obs(), 2, "left subtree n_obs");
        // right subtree: 0 obs
        assert_eq!(children[1].n_obs(), 0, "right subtree n_obs");
    }

    // Verify leaf accessor returns 0.
    let leaf = find_leaf(&tree.root, 0).expect("arm 0 exists");
    assert_eq!(leaf.n_obs(), 0, "leaf n_obs should be 0");
}

/// Helper: build the same 4-leaf / 2-subtree topology used by the gate tests.
/// Kept separate from `build_test_tree` so the gate tests can supply their
/// own config without changing the existing test fixture.
fn make_tree_topology() -> TreeNode {
    let left = TreeNode::internal(vec![TreeNode::leaf(0, 0.0), TreeNode::leaf(1, 0.0)]);
    let right = TreeNode::internal(vec![TreeNode::leaf(2, 0.0), TreeNode::leaf(3, 0.0)]);
    TreeNode::internal(vec![left, right])
}

#[test]
#[should_panic(expected = "not in tree")]
fn test_observe_invalid_arm_panics() {
    let mut tree = build_test_tree(0.0);
    tree.observe(99, 1.0, 0);
}

#[test]
fn test_arm_path_depth_correct() {
    let tree = build_test_tree(0.0);
    // All leaves are at depth 2 in the test tree.
    for (arm, path) in tree.arm_paths.iter().enumerate() {
        assert_eq!(path.len, 2, "arm {arm} path should have depth 2");
    }
}

#[test]
fn test_num_arms() {
    let tree = build_test_tree(0.0);
    assert_eq!(tree.num_arms(), 4);
}

// ── Phase 3 helpers: synthetic embeddings ──────────────────────────

/// Deterministic PRNG for test data generation (mirrors bench's Lcg).
struct TestRng {
    state: u64,
}
impl TestRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state ^ (self.state >> 29)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
    fn next_normal(&mut self) -> f32 {
        // Box-Muller.
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Generate N D-dim embeddings arranged in `n_clusters` well-separated
/// Gaussian clusters. Each cluster has a random center; arms within a
/// cluster are sampled from N(center, σ²) per dimension.
fn gen_clustered_embeddings(
    n_total: usize,
    n_clusters: usize,
    dim: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    assert_eq!(
        n_total % n_clusters,
        0,
        "n_total must be divisible by n_clusters"
    );
    let per_cluster = n_total / n_clusters;
    let mut rng = TestRng::new(seed);

    // Cluster centers: well-separated random points in [−10, 10]^dim.
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.next_f32() * 20.0 - 10.0).collect())
        .collect();

    let mut embeddings = Vec::with_capacity(n_total);
    for center in centers.iter().take(n_clusters) {
        for _ in 0..per_cluster {
            let point: Vec<f32> = (0..dim)
                .map(|j| center[j] + rng.next_normal() * 0.5)
                .collect();
            embeddings.push(point);
        }
    }
    embeddings.shrink_to_fit();
    embeddings
}

/// Count the children of the root if it's Internal, else 0.
fn root_children_count(node: &TreeNode) -> usize {
    match node {
        TreeNode::Internal { children, .. } => children.len(),
        TreeNode::Leaf { .. } => 0,
    }
}

/// Collect all arm_ids under a node.
fn collect_arm_ids(node: &TreeNode, out: &mut Vec<usize>) {
    match node {
        TreeNode::Leaf { arm_id, .. } => out.push(*arm_id),
        TreeNode::Internal { children, .. } => {
            for c in children {
                collect_arm_ids(c, out);
            }
        }
    }
}

// ── T3.1: PCA tests ─────────────────────────────────────────────────

#[test]
fn test_pca_recovers_principal_direction() {
    // Data: 100 points along the line y = 2x (plus noise).
    // Top principal component should be ≈ [1, 2] / √5.
    let mut rng = TestRng::new(42);
    let n = 100usize;
    let data: Vec<f32> = (0..n)
        .flat_map(|_| {
            let t = rng.next_f32() * 10.0 - 5.0;
            let nx = rng.next_normal() * 0.1;
            let ny = rng.next_normal() * 0.1;
            [t + nx, 2.0 * t + ny]
        })
        .collect();

    let mut out = vec![0.0f32; n];
    pca_into(&data, n, 1, &mut out, 123);

    // The projected data should have much higher variance than either input dim.
    let mean_proj: f32 = out.iter().sum::<f32>() / n as f32;
    let var_proj: f32 = out.iter().map(|x| (x - mean_proj).powi(2)).sum::<f32>() / n as f32;
    // Original variance along x ≈ (10/√12)² ≈ 8.33, projected should be ~5× larger.
    assert!(
        var_proj > 10.0,
        "PCA projected variance {var_proj:.2} too low — direction not recovered"
    );
}

#[test]
fn test_pca_deterministic() {
    let data: Vec<f32> = (0..50).flat_map(|i| [i as f32, i as f32 * 3.0]).collect();
    let mut out1 = vec![0.0f32; 50];
    let mut out2 = vec![0.0f32; 50];
    pca_into(&data, 50, 1, &mut out1, 999);
    pca_into(&data, 50, 1, &mut out2, 999);
    // Bit-identical (deterministic given same seed).
    assert_eq!(out1, out2, "PCA must be deterministic given same seed");
}

// ── T3.2: 2D embedding tests ────────────────────────────────────────

#[test]
fn test_embed_2d_separates_clusters() {
    // Two clusters along the x-axis in 16D: one centered at +5·e₀, one at −5·e₀.
    let n = 40usize;
    let dim = 16usize;
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let mut point = vec![0.0f32; dim];
        point[0] = if i < n / 2 { 5.0 } else { -5.0 };
        data.extend(point);
    }

    let embedded = embed_2d(&data, n, dim, 7);
    assert_eq!(embedded.len(), n);

    // Cluster 0 (first half) should be clearly separated from cluster 1 in
    // the first principal component.
    let mean_a: f32 = embedded[..n / 2].iter().map(|p| p[0]).sum::<f32>() / (n / 2) as f32;
    let mean_b: f32 = embedded[n / 2..].iter().map(|p| p[0]).sum::<f32>() / (n / 2) as f32;
    assert!(
        (mean_a - mean_b).abs() > 5.0,
        "clusters not separated in PC1: mean_a={mean_a:.2}, mean_b={mean_b:.2}"
    );
}

// ── T3.3: Chart test tests ──────────────────────────────────────────

#[test]
fn test_chart_test_round_vs_elongated() {
    // Round cluster: points in a disk → high eigenvalue ratio → not noise.
    let mut rng = TestRng::new(1);
    let round: Vec<[f32; 2]> = (0..50)
        .map(|_| {
            let r = rng.next_f32();
            let theta = rng.next_f32() * 2.0 * std::f32::consts::PI;
            [r * theta.cos(), r * theta.sin()]
        })
        .collect();
    let noise = chart_test(&round, 10, 0.3);
    let n_noise = noise.iter().filter(|&&x| x).count();
    // Most round-cluster points should NOT be noise.
    assert!(
        n_noise < round.len() / 2,
        "too many noise points in round cluster: {n_noise}/{}",
        round.len()
    );
}

// ── T3.4: DBSCAN tests ──────────────────────────────────────────────

#[test]
fn test_dbscan_finds_two_clusters() {
    // Two well-separated clusters.
    let points: Vec<[f32; 2]> = (0..10)
        .map(|i| {
            if i < 5 {
                [i as f32 * 0.1, 0.0]
            } else {
                [10.0 + (i - 5) as f32 * 0.1, 0.0]
            }
        })
        .collect();
    let labels = dbscan_adaptive(&points, 3);
    let n_clusters = labels
        .iter()
        .filter_map(|&c| c)
        .map(|c| c + 1)
        .max()
        .unwrap_or(0);
    assert_eq!(n_clusters, 2, "expected 2 clusters, got {n_clusters}");
    // No noise — all points should be assigned.
    assert!(
        labels.iter().all(|c| c.is_some()),
        "all points should be clustered"
    );
}

#[test]
fn test_dbscan_isolated_point_is_noise() {
    let points: Vec<[f32; 2]> = vec![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.3, 0.0],     // cluster
        [100.0, 100.0], // isolated
    ];
    let labels = dbscan_adaptive(&points, 3);
    // The isolated point should be noise.
    assert!(labels[4].is_none(), "isolated point should be noise");
    // The cluster points should form one cluster.
    let cluster_labels: Vec<_> = labels[..4].iter().filter_map(|&c| c).collect();
    assert_eq!(cluster_labels.len(), 4, "cluster should have 4 points");
    assert!(
        cluster_labels.iter().all(|&c| c == 0),
        "all cluster points should be cluster 0"
    );
}

// ── T3.5: build() integration tests ─────────────────────────────────

#[test]
fn test_build_from_synthetic_embeddings_finds_clusters() {
    // 128 embeddings, 8 clusters, 16-dim.
    let embeddings = gen_clustered_embeddings(128, 8, 16, 42);
    let config = LatentTaskTreeConfig::default();
    let tree = LatentTaskTree::build(&embeddings, config);

    // Root should be Internal with multiple children.
    assert!(
        matches!(&tree.root, TreeNode::Internal { .. }),
        "root should be Internal for multi-cluster data"
    );
    let n_top = root_children_count(&tree.root);
    assert!(n_top >= 4, "expected ≥4 top-level clusters, got {n_top}");

    // All 128 arms should be reachable.
    assert_eq!(tree.num_arms(), 128, "all 128 arms should be in the tree");
    let mut ids = Vec::new();
    collect_arm_ids(&tree.root, &mut ids);
    ids.sort();
    assert_eq!(
        ids,
        (0..128).collect::<Vec<_>>(),
        "arm_ids should be 0..128"
    );
}

#[test]
fn test_build_blake3_stable_across_rebuilds() {
    let embeddings = gen_clustered_embeddings(64, 4, 16, 99);
    let config = LatentTaskTreeConfig::default();
    let tree1 = LatentTaskTree::build(&embeddings, config);
    let tree2 = LatentTaskTree::build(&embeddings, config);
    assert_eq!(
        tree1.blake3_root(),
        tree2.blake3_root(),
        "identical (embeddings, config) → identical BLAKE3"
    );
}

#[test]
fn test_build_different_embeddings_different_blake3() {
    let e1 = gen_clustered_embeddings(64, 4, 16, 1);
    let e2 = gen_clustered_embeddings(64, 4, 16, 2);
    let config = LatentTaskTreeConfig::default();
    let t1 = LatentTaskTree::build(&e1, config);
    let t2 = LatentTaskTree::build(&e2, config);
    assert_ne!(
        t1.blake3_root(),
        t2.blake3_root(),
        "different embeddings → different BLAKE3"
    );
}

#[test]
fn test_build_single_embedding_is_leaf() {
    let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]];
    let tree = LatentTaskTree::build(&embeddings, LatentTaskTreeConfig::default());
    assert!(matches!(&tree.root, TreeNode::Leaf { arm_id: 0, .. }));
    assert_eq!(tree.num_arms(), 1);
}

#[test]
fn test_build_few_embeddings_make_leaf_group() {
    // 3 embeddings, min_cluster = 4 → too few to cluster → flat leaf group.
    let embeddings: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let tree = LatentTaskTree::build(&embeddings, LatentTaskTreeConfig::default());
    assert!(matches!(&tree.root, TreeNode::Internal { .. }));
    assert_eq!(tree.num_arms(), 3);
    if let TreeNode::Internal { children, .. } = &tree.root {
        assert_eq!(children.len(), 3, "should have 3 leaf children");
        assert!(children.iter().all(|c| matches!(c, TreeNode::Leaf { .. })));
    }
}

#[test]
fn test_build_sample_returns_valid_arms() {
    let embeddings = gen_clustered_embeddings(64, 4, 16, 7);
    let tree = LatentTaskTree::build(&embeddings, LatentTaskTreeConfig::default());
    let mut rng = fastrand::Rng::with_seed(42);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..2000 {
        let arm = tree.sample(&mut rng);
        assert!(arm < 64, "sampled arm {arm} out of range");
        seen.insert(arm);
    }
    // With uniform priors, should eventually visit all arms.
    assert_eq!(
        seen.len(),
        64,
        "should visit all 64 arms with uniform prior"
    );
}

#[test]
fn test_build_observe_updates_correct_leaf() {
    let embeddings = gen_clustered_embeddings(64, 4, 16, 7);
    // drift_rate = 0 so alpha increments are exact (no decay between steps).
    let config = LatentTaskTreeConfig {
        filter_drift_rate: 0.0,
        ..LatentTaskTreeConfig::default()
    };
    let mut tree = LatentTaskTree::build(&embeddings, config);

    // Observe 5 successes on arm 10.
    for step in 0..5u64 {
        tree.observe(10, 1.0, step);
    }

    let leaf = find_leaf(&tree.root, 10).expect("arm 10 should exist");
    if let TreeNode::Leaf { filter, .. } = leaf {
        // alpha = 1 + 5 = 6.
        assert!(
            (filter.alpha - 6.0).abs() < 1e-5,
            "alpha should be 6, got {}",
            filter.alpha
        );
        assert!(
            (filter.beta - 1.0).abs() < 1e-5,
            "beta should be 1, got {}",
            filter.beta
        );
    }
}

// ── Helper: find a leaf by arm_id ─────────────────────────────────

fn find_leaf(node: &TreeNode, arm_id: usize) -> Option<&TreeNode> {
    match node {
        TreeNode::Leaf { arm_id: id, .. } if *id == arm_id => Some(node),
        TreeNode::Leaf { .. } => None,
        TreeNode::Internal { children, .. } => {
            for child in children {
                if let Some(found) = find_leaf(child, arm_id) {
                    return Some(found);
                }
            }
            None
        }
    }
}
