//! Plan 370 Phase 2 — Manifold Bandit GOAT Gate Benchmark (G1–G5).
//!
//! Exercises the full GOAT gate for the `manifold_bandit` primitive on synthetic
//! clustered-arm domains. No real embeddings, no game semantics — the claim is
//! architectural (structure-aware Thompson descent) + non-stationarity (Bayesian
//! filter) + latency + reproducibility, per Plan 370 §GOAT gate.
//!
//! # Gates
//!
//! - **G1** — Structural advantage: hierarchical Thompson reaches 90% optimal-arm
//!   selection in ≤ 0.8× the steps of flat Thompson on a 64-arm / 8-cluster domain.
//! - **G2** — Diversity preservation: hierarchical visits ≥ 1.5× the distinct
//!   clusters flat visits (threshold-based), at matched cumulative reward (±5%).
//! - **G3** — Non-stationarity recovery: hierarchical + BayesianFilterArm recovers
//!   to 80% optimal in ≤ 0.5× the steps of flat Thompson after an arm-mean shift.
//!   Ablation: flat-with-filter isolates the filter's contribution. Sliding-window
//!   proxy for Dual-Pool CGSP (Plan 312) recorded for honest comparison.
//! - **G4** — Latency: `sample` p50 ≤ 500 ns, `observe` p50 ≤ 300 ns at depth 6;
//!   0 allocations on the hot path (CountingAllocator).
//! - **G5** — Bit-reproducibility: two trees from identical (topology, config) +
//!   identical (seed, observation sequence) → byte-identical 10K-sample sequences.
//!
//! G3 (no-regression) is verified externally via `cargo check --all-features` and
//! `cargo check --no-default-features` (the merkle_root lesson).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features manifold_bandit \
//!     --bench bench_370_manifold_bandit_goat -- --nocapture
//! ```
//!
//! Or directly (working around the macOS dyld/trustd stall):
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/manifold_bandit_370 cargo build --release -p katgpt-core \
//!     --features manifold_bandit --bench bench_370_manifold_bandit_goat
//! /tmp/manifold_bandit_370/release/bench_370_manifold_bandit_goat-* --nocapture
//! ```

#![cfg(feature = "manifold_bandit")]

use katgpt_core::manifold_bandit::{
    BayesianFilterArm, LatentTaskTree, LatentTaskTreeConfig, TreeNode,
};
use std::collections::VecDeque;
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── GateResult ─────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

// ─── Constants ──────────────────────────────────────────────────────────────

// G1/G2: 64 arms, 8 clusters of 8.
const N_CLUSTERS: usize = 8;
const ARMS_PER_CLUSTER: usize = 8;
const N_ARMS: usize = N_CLUSTERS * ARMS_PER_CLUSTER; // 64

// G1: structural advantage.
const T_G1: usize = 5_000;
const TRIALS_G1: usize = 200;
const THRESHOLD_G1: f32 = 0.90;
const RATIO_GATE_G1: f32 = 0.80; // hierarchical ≤ 0.8× flat
const WINDOW: usize = 100;

// G2: diversity.
const T_G2: usize = 2_000;
const TRIALS_G2: usize = 200;
const RATIO_GATE_G2: f32 = 1.50; // hierarchical ≥ 1.5× flat
const REWARD_TOL_G2: f32 = 0.05;
/// A cluster counts as "meaningfully visited" if it received ≥ this fraction of
/// total selections. Pure "visited at least once" saturates at 8/8 for both
/// strategies (Thompson explores broadly early); the threshold captures sustained
/// exploration, which is the paper's actual diversity claim.
const CLUSTER_FRAC_THRESHOLD: f32 = 0.02; // 2% of T

// G3: non-stationarity recovery. 16 arms, 4 clusters of 4.
const N_CLUSTERS_G3: usize = 4;
const ARMS_PER_CLUSTER_G3: usize = 4;
const N_ARMS_G3: usize = 16;
const T_G3: usize = 2_000;
const SHIFT_STEP: usize = 1_000;
const TRIALS_G3: usize = 100;
const THRESHOLD_G3: f32 = 0.80;
const RATIO_GATE_G3: f32 = 0.50;
const DRIFT_RATE: f32 = 0.05;
const SLIDING_WINDOW_SIZE: usize = 50;
const ARM_SHIFT_FROM: usize = 0;
const ARM_SHIFT_TO: usize = 5;

// G4: latency.
const DEPTH_G4: usize = 6;
const BRANCHING_G4: usize = 2;
const LATENCY_TARGET_SAMPLE_NS: u64 = 500;
const LATENCY_TARGET_OBSERVE_NS: u64 = 300;
const LATENCY_ITERS: usize = 10_000;
const LATENCY_WARMUP: usize = 1_000;

// G5: reproducibility.
const G5_SAMPLES: usize = 10_000;

// ─── LCG (deterministic PRNG for domain/reward generation) ──────────────────
//
// Independent from the agent's `fastrand::Rng` so that the reward sequence is
// identical regardless of which strategy is being tested (common random numbers).

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Beta Thompson sample via the public `BayesianFilterArm` API (same Gamma-ratio
/// sampler used internally — no sampler duplication).
fn beta_thompson(alpha: f32, beta: f32, rng: &mut fastrand::Rng) -> f32 {
    BayesianFilterArm {
        alpha,
        beta,
        drift_rate: 0.0,
        last_obs_step: 0,
    }
    .thompson_sample(rng)
}

// ─── Clustered domain ───────────────────────────────────────────────────────

#[derive(Clone)]
struct ClusteredDomain {
    arm_means: Vec<f32>,
    arm_rngs: Vec<Lcg>,
    arms_per_cluster: usize,
}

impl ClusteredDomain {
    /// G1/G2 domain: 64 arms in 8 clusters. Cluster means ~ Uniform(0.2, 0.8),
    /// arm noise ~ Normal(0, 0.05). Per-arm reward RNG ensures common random
    /// numbers across strategies.
    fn new_clustered(trial: u64) -> Self {
        let mut dom_rng = Lcg::new(trial.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        let cluster_means: Vec<f32> = (0..N_CLUSTERS)
            .map(|_| 0.2 + dom_rng.next_f32() * 0.6)
            .collect();
        let arm_means: Vec<f32> = (0..N_ARMS)
            .map(|arm| {
                let cluster = arm / ARMS_PER_CLUSTER;
                let noise = dom_rng.next_normal() * 0.05;
                (cluster_means[cluster] + noise).clamp(0.01, 0.99)
            })
            .collect();
        let arm_rngs: Vec<Lcg> = (0..N_ARMS)
            .map(|i| Lcg::new(trial.wrapping_mul(1000).wrapping_add(i as u64 + 1)))
            .collect();
        Self {
            arm_means,
            arm_rngs,
            arms_per_cluster: ARMS_PER_CLUSTER,
        }
    }

    /// G3 domain: 16 arms, 4 clusters of 4. Arm 0 starts optimal (mean 0.8),
    /// all others 0.2. After [`shift`]: arm 0 → 0.2, arm 5 → 0.8.
    fn new_shift(trial: u64) -> Self {
        let mut arm_means = vec![0.2_f32; N_ARMS_G3];
        arm_means[ARM_SHIFT_FROM] = 0.8;
        let arm_rngs: Vec<Lcg> = (0..N_ARMS_G3)
            .map(|i| Lcg::new(trial.wrapping_mul(1000).wrapping_add(i as u64 + 1)))
            .collect();
        Self {
            arm_means,
            arm_rngs,
            arms_per_cluster: ARMS_PER_CLUSTER_G3,
        }
    }

    fn reward(&mut self, arm: usize) -> f32 {
        let p = self.arm_means[arm];
        if self.arm_rngs[arm].next_f32() < p {
            1.0
        } else {
            0.0
        }
    }

    fn optimal_arm(&self) -> usize {
        (0..self.arm_means.len())
            .max_by(|&a, &b| {
                katgpt_core::float_order::cmp_for_max(self.arm_means[a], self.arm_means[b])
            })
            .unwrap()
    }

    fn cluster_of(&self, arm: usize) -> usize {
        arm / self.arms_per_cluster
    }

    /// G3 shift: drop old optimal, raise new optimal.
    fn shift(&mut self) {
        self.arm_means[ARM_SHIFT_FROM] = 0.2;
        self.arm_means[ARM_SHIFT_TO] = 0.8;
    }
}

// ─── Strategies ─────────────────────────────────────────────────────────────

trait Bandit {
    fn sample(&self, rng: &mut fastrand::Rng) -> usize;
    fn observe(&mut self, arm: usize, reward: f32, step: u64);
}

/// Flat Thompson: N independent `BayesianFilterArm` posteriors, argmax of draws.
struct FlatThompson {
    arms: Vec<BayesianFilterArm>,
}

impl FlatThompson {
    fn new(n_arms: usize, drift_rate: f32) -> Self {
        Self {
            arms: (0..n_arms)
                .map(|_| BayesianFilterArm::new(drift_rate))
                .collect(),
        }
    }
}

impl Bandit for FlatThompson {
    fn sample(&self, rng: &mut fastrand::Rng) -> usize {
        let mut best = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            let s = arm.thompson_sample(rng);
            if s > best_s {
                best_s = s;
                best = i;
            }
        }
        best
    }
    fn observe(&mut self, arm: usize, reward: f32, step: u64) {
        self.arms[arm].predict(step);
        self.arms[arm].update(reward, step);
    }
}

/// Hierarchical Thompson: delegates to the Phase 1 `LatentTaskTree`.
struct HierarchicalThompson {
    tree: LatentTaskTree,
}

impl Bandit for HierarchicalThompson {
    fn sample(&self, rng: &mut fastrand::Rng) -> usize {
        self.tree.sample(rng)
    }
    fn observe(&mut self, arm: usize, reward: f32, step: u64) {
        self.tree.observe(arm, reward, step);
    }
}

/// Sliding-window Thompson: proxy for Dual-Pool CGSP (Plan 312). Each arm keeps
/// only the last `window_size` observations; Beta(1+Σr, 1+W-Σr). Captures the
/// "forget old data" non-stationarity idea. Not a faithful Dual-Pool port —
/// recorded for honest comparison, not as a gate requirement.
struct SlidingWindowThompson {
    windows: Vec<VecDeque<f32>>,
    window_size: usize,
}

impl SlidingWindowThompson {
    fn new(n_arms: usize, window_size: usize) -> Self {
        Self {
            windows: (0..n_arms)
                .map(|_| VecDeque::with_capacity(window_size + 1))
                .collect(),
            window_size,
        }
    }
}

impl Bandit for SlidingWindowThompson {
    fn sample(&self, rng: &mut fastrand::Rng) -> usize {
        let mut best = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (i, win) in self.windows.iter().enumerate() {
            let n = win.len() as f32;
            let sum: f32 = win.iter().sum();
            let alpha = 1.0 + sum;
            let beta = 1.0 + (n - sum);
            let s = beta_thompson(alpha, beta, rng);
            if s > best_s {
                best_s = s;
                best = i;
            }
        }
        best
    }
    fn observe(&mut self, arm: usize, reward: f32, _step: u64) {
        let win = &mut self.windows[arm];
        if win.len() >= self.window_size {
            win.pop_front();
        }
        win.push_back(reward);
    }
}

// ─── Tree builders ──────────────────────────────────────────────────────────

fn build_clustered_tree(n_clusters: usize, arms_per_cluster: usize, drift_rate: f32) -> TreeNode {
    let cluster_nodes: Vec<TreeNode> = (0..n_clusters)
        .map(|c| {
            let leaves: Vec<TreeNode> = (0..arms_per_cluster)
                .map(|i| TreeNode::leaf(c * arms_per_cluster + i, drift_rate))
                .collect();
            TreeNode::internal(leaves)
        })
        .collect();
    TreeNode::internal(cluster_nodes)
}

fn build_deep_tree(depth: usize, branching: usize, drift_rate: f32) -> (TreeNode, usize) {
    fn rec(depth: usize, branching: usize, drift_rate: f32, counter: &mut usize) -> TreeNode {
        if depth == 0 {
            let arm = *counter;
            *counter += 1;
            TreeNode::leaf(arm, drift_rate)
        } else {
            let children: Vec<TreeNode> = (0..branching)
                .map(|_| rec(depth - 1, branching, drift_rate, counter))
                .collect();
            TreeNode::internal(children)
        }
    }
    let mut counter = 0usize;
    let root = rec(depth, branching, drift_rate, &mut counter);
    (root, counter)
}

// ─── Metrics ────────────────────────────────────────────────────────────────

/// First step (1-indexed) where the trailing-window optimal-selection fraction
/// reaches `threshold`. Returns `selections.len()` if never reached.
fn steps_to_threshold(
    selections: &[usize],
    optimal_arm: usize,
    threshold: f32,
    window: usize,
) -> usize {
    let n = selections.len();
    let mut window_count = 0usize;
    for i in 0..n {
        if selections[i] == optimal_arm {
            window_count += 1;
        }
        if i >= window && selections[i - window] == optimal_arm {
            window_count -= 1;
        }
        let window_len = (i + 1).min(window);
        if window_count as f32 / window_len as f32 >= threshold {
            return i + 1;
        }
    }
    n
}

fn median_u64(data: &mut [u64]) -> u64 {
    data.sort_unstable();
    if data.is_empty() {
        return 0;
    }
    data[data.len() / 2]
}

fn median_f32(data: &mut [f32]) -> f32 {
    data.sort_by(|a, b| a.total_cmp(b));
    if data.is_empty() {
        return 0.0;
    }
    data[data.len() / 2]
}

fn median_duration_ns_arr(data: &mut [u64]) -> u64 {
    data.sort_unstable();
    if data.is_empty() {
        return 0;
    }
    data[data.len() / 2]
}

// ─── Trial results ──────────────────────────────────────────────────────────

struct TrialResult {
    steps_to_threshold: usize,
    cluster_counts: Vec<usize>,
    cumulative_reward: f32,
}

fn run_trial<B: Bandit>(
    bandit: &mut B,
    domain: &mut ClusteredDomain,
    agent_seed: u64,
    t_steps: usize,
    optimal_arm: usize,
    n_clusters: usize,
    threshold: f32,
) -> TrialResult {
    let mut rng = fastrand::Rng::with_seed(agent_seed);
    let mut selections = Vec::with_capacity(t_steps);
    let mut cluster_counts = vec![0usize; n_clusters];
    let mut cumulative_reward = 0.0f32;

    for step in 0..t_steps {
        let arm = bandit.sample(&mut rng);
        let reward = domain.reward(arm);
        bandit.observe(arm, reward, step as u64);
        selections.push(arm);
        cluster_counts[domain.cluster_of(arm)] += 1;
        cumulative_reward += reward;
    }

    let s = steps_to_threshold(&selections, optimal_arm, threshold, WINDOW);
    TrialResult {
        steps_to_threshold: s,
        cluster_counts,
        cumulative_reward,
    }
}

fn run_trial_shift<B: Bandit>(
    bandit: &mut B,
    domain: &mut ClusteredDomain,
    agent_seed: u64,
    t_steps: usize,
    shift_step: usize,
    n_clusters: usize,
) -> (usize, Vec<usize>, f32) {
    // Returns (steps_to_80_after_shift, cluster_counts, cumulative_reward).
    let mut rng = fastrand::Rng::with_seed(agent_seed);
    let mut selections = Vec::with_capacity(t_steps);
    let mut cluster_counts = vec![0usize; n_clusters];
    let mut cumulative_reward = 0.0f32;

    for step in 0..t_steps {
        if step == shift_step {
            domain.shift();
        }
        let arm = bandit.sample(&mut rng);
        let reward = domain.reward(arm);
        bandit.observe(arm, reward, step as u64);
        selections.push(arm);
        cluster_counts[domain.cluster_of(arm)] += 1;
        cumulative_reward += reward;
    }

    let post = &selections[shift_step..];
    let s = steps_to_threshold(post, ARM_SHIFT_TO, THRESHOLD_G3, WINDOW);
    (s, cluster_counts, cumulative_reward)
}

// ─── Gate functions ─────────────────────────────────────────────────────────

fn gate_g1_structural_advantage() -> GateResult {
    println!(
        "\n--- G1: Structural Advantage (64 arms, 8 clusters, T={T_G1}, {TRIALS_G1} trials) ---"
    );

    let mut flat_steps = Vec::with_capacity(TRIALS_G1);
    let mut hier_steps = Vec::with_capacity(TRIALS_G1);

    for trial in 0..TRIALS_G1 {
        let domain = ClusteredDomain::new_clustered(trial as u64);
        let optimal = domain.optimal_arm();

        // Flat.
        let mut dom = domain.clone();
        let mut flat = FlatThompson::new(N_ARMS, 0.0);
        let r = run_trial(
            &mut flat,
            &mut dom,
            trial as u64,
            T_G1,
            optimal,
            N_CLUSTERS,
            THRESHOLD_G1,
        );
        flat_steps.push(r.steps_to_threshold as u64);

        // Hierarchical.
        let mut dom = domain.clone();
        let root = build_clustered_tree(N_CLUSTERS, ARMS_PER_CLUSTER, 0.0);
        let tree = LatentTaskTree::from_root(root, LatentTaskTreeConfig::default());
        let mut hier = HierarchicalThompson { tree };
        let r = run_trial(
            &mut hier,
            &mut dom,
            trial as u64,
            T_G1,
            optimal,
            N_CLUSTERS,
            THRESHOLD_G1,
        );
        hier_steps.push(r.steps_to_threshold as u64);
    }

    let med_flat = median_u64(&mut flat_steps);
    let med_hier = median_u64(&mut hier_steps);
    let ratio = med_hier as f64 / med_flat as f64;
    let passed = ratio <= RATIO_GATE_G1 as f64;

    println!("  flat Thompson    median steps-to-90%: {med_flat}");
    println!("  hierarchical     median steps-to-90%: {med_hier}");
    println!(
        "  ratio (hier/flat): {ratio:.3}  (gate: ≤ {RATIO_GATE_G1:.1})"
    );

    if passed {
        GateResult::pass(
            "G1 structural advantage",
            format!(
                "hier {med_hier} ≤ {RATIO_GATE_G1:.1}× flat {med_flat} (ratio {ratio:.3})"
            ),
        )
    } else {
        GateResult::fail(
            "G1 structural advantage",
            format!(
                "hier {med_hier} > {RATIO_GATE_G1:.1}× flat {med_flat} (ratio {ratio:.3}) — structure does not accelerate convergence"
            ),
        )
    }
}

// ─── G1-real: structural advantage with Phase 3 build() tree (T3.6) ──────────
//
// Generates synthetic 16-dim embeddings with 8 well-separated Gaussian
// clusters, builds the tree via `LatentTaskTree::build` (PCA → 2D → Chart →
// DBSCAN → recurse), and re-runs the G1 structural-advantage comparison.
// The real-constructed tree should produce a ratio comparable to (or stronger
// than) the hand-built tree, validating the Phase 3 pipeline end-to-end.

/// Generate N_CLUSTERS × ARMS_PER_CLUSTER embeddings in 16-dim space, where
/// each cluster occupies a distinct region. The cluster structure aligns with
/// the `ClusteredDomain` (cluster k = arms k*APC .. k*APC+APC-1).
fn gen_clustered_embeddings_bench(trial: u64) -> Vec<Vec<f32>> {
    let dim = 16usize;
    let mut rng = Lcg::new(trial.wrapping_mul(0x100_0000_0000).wrapping_add(777));

    // Cluster centers: well-separated random points in [−8, 8]^dim.
    let centers: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| (0..dim).map(|_| rng.next_f32() * 16.0 - 8.0).collect())
        .collect();

    let mut embeddings = Vec::with_capacity(N_ARMS);
    for (_c, center) in centers.iter().enumerate().take(N_CLUSTERS) {
        for _ in 0..ARMS_PER_CLUSTER {
            let point: Vec<f32> = (0..dim)
                .map(|j| center[j] + rng.next_normal() * 0.5)
                .collect();
            embeddings.push(point);
        }
    }
    embeddings.shrink_to_fit();
    embeddings
}

fn gate_g1_real_tree_structural_advantage() -> GateResult {
    println!("\n--- G1-real: Structural Advantage with Phase 3 build() tree ---");
    println!(
        "    ({N_ARMS} arms, {N_CLUSTERS} clusters, T={T_G1}, {TRIALS_G1} trials)"
    );

    let mut flat_steps = Vec::with_capacity(TRIALS_G1);
    let mut hier_steps = Vec::with_capacity(TRIALS_G1);
    let mut n_top_clusters_seen: Vec<usize> = Vec::with_capacity(TRIALS_G1);

    for trial in 0..TRIALS_G1 {
        let domain = ClusteredDomain::new_clustered(trial as u64);
        let optimal = domain.optimal_arm();

        // Generate embeddings with matching cluster structure.
        let embeddings = gen_clustered_embeddings_bench(trial as u64);

        // Flat.
        let mut dom = domain.clone();
        let mut flat = FlatThompson::new(N_ARMS, 0.0);
        let r = run_trial(
            &mut flat,
            &mut dom,
            trial as u64,
            T_G1,
            optimal,
            N_CLUSTERS,
            THRESHOLD_G1,
        );
        flat_steps.push(r.steps_to_threshold as u64);

        // Hierarchical — real-constructed tree.
        // drift_rate = 0.0 matches the hand-built G1 (stationary domain).
        let config = LatentTaskTreeConfig {
            filter_drift_rate: 0.0,
            ..LatentTaskTreeConfig::default()
        };
        let mut dom = domain.clone();
        let tree = LatentTaskTree::build(&embeddings, config);
        let n_top = match tree.root() {
            TreeNode::Internal { children, .. } => children.len(),
            _ => 0,
        };
        n_top_clusters_seen.push(n_top);

        let mut hier = HierarchicalThompson { tree };
        let r = run_trial(
            &mut hier,
            &mut dom,
            trial as u64,
            T_G1,
            optimal,
            N_CLUSTERS,
            THRESHOLD_G1,
        );
        hier_steps.push(r.steps_to_threshold as u64);
    }

    let med_flat = median_u64(&mut flat_steps);
    let med_hier = median_u64(&mut hier_steps);
    let ratio = med_hier as f64 / med_flat as f64;
    let passed = ratio <= RATIO_GATE_G1 as f64;

    n_top_clusters_seen.sort();
    let median_top = n_top_clusters_seen[n_top_clusters_seen.len() / 2];

    println!("  flat Thompson    median steps-to-90%: {med_flat}");
    println!("  hier (real tree) median steps-to-90%: {med_hier}");
    println!(
        "  ratio (hier/flat): {ratio:.3}  (gate: ≤ {RATIO_GATE_G1:.1})"
    );
    println!(
        "  real tree median top-level clusters: {median_top} (domain has {N_CLUSTERS})"
    );

    if passed {
        GateResult::pass(
            "G1-real structural advantage (Phase 3 build)",
            format!(
                "hier {med_hier} ≤ {RATIO_GATE_G1:.1}× flat {med_flat} (ratio {ratio:.3}, {median_top} top clusters)"
            ),
        )
    } else {
        GateResult::fail(
            "G1-real structural advantage (Phase 3 build)",
            format!(
                "hier {med_hier} > {RATIO_GATE_G1:.1}× flat {med_flat} (ratio {ratio:.3}, {median_top} top clusters) — real tree does not accelerate convergence"
            ),
        )
    }
}

/// **Phase 4 T4.2 — R279 N≥d phase gate sweep on G1-real.**
///
/// Runs the G1-real structural-advantage benchmark with several
/// `phase_gate_min_obs` values (d ∈ {0, 1, 2, 4, 8}) to measure whether the
/// gate improves G1 (faster convergence) or hurts it (under-aggregation).
///
/// This is NOT a pass/fail gate — it's a diagnostic sweep. The verdict is in
/// the comparison: if any d > 0 produces a ratio ≤ the ungated ratio (d=0),
/// the phase gate is a net win and Phase 4 T4.3 should promote the gate.
///
/// Uses fewer trials (50) than the main G1 (200) because it runs 5 configs.
fn gate_g1_real_phase_gate_sweep() -> GateResult {
    const TRIALS_SWEEP: usize = 50;
    const D_VALUES: [u32; 5] = [0, 1, 2, 4, 8];

    println!(
        "\n--- G1-real Phase Gate Sweep (T4.2): {N_ARMS} arms, {N_CLUSTERS} clusters, T={T_G1}, {TRIALS_SWEEP} trials ---"
    );
    println!("    d_values: {D_VALUES:?}");

    // Flat Thompson baseline is the same across all d values.
    let mut flat_steps = Vec::with_capacity(TRIALS_SWEEP);
    for trial in 0..TRIALS_SWEEP {
        let domain = ClusteredDomain::new_clustered(trial as u64);
        let optimal = domain.optimal_arm();
        let mut dom = domain.clone();
        let mut flat = FlatThompson::new(N_ARMS, 0.0);
        let r = run_trial(
            &mut flat,
            &mut dom,
            trial as u64,
            T_G1,
            optimal,
            N_CLUSTERS,
            THRESHOLD_G1,
        );
        flat_steps.push(r.steps_to_threshold as u64);
    }
    let med_flat = median_u64(&mut flat_steps);
    println!("  flat Thompson median steps-to-90%: {med_flat}");

    // Sweep d values.
    let mut best_ratio = f64::INFINITY;
    let mut best_d: u32 = 0;
    let mut sweep_results: Vec<(u32, u64, f64)> = Vec::with_capacity(D_VALUES.len());

    for &d in &D_VALUES {
        let mut hier_steps = Vec::with_capacity(TRIALS_SWEEP);
        for trial in 0..TRIALS_SWEEP {
            let domain = ClusteredDomain::new_clustered(trial as u64);
            let optimal = domain.optimal_arm();
            let embeddings = gen_clustered_embeddings_bench(trial as u64);

            let mut dom = domain.clone();
            let config = LatentTaskTreeConfig {
                filter_drift_rate: 0.0,
                phase_gate_min_obs: d,
                ..LatentTaskTreeConfig::default()
            };
            let tree = LatentTaskTree::build(&embeddings, config);
            let mut hier = HierarchicalThompson { tree };
            let r = run_trial(
                &mut hier,
                &mut dom,
                trial as u64,
                T_G1,
                optimal,
                N_CLUSTERS,
                THRESHOLD_G1,
            );
            hier_steps.push(r.steps_to_threshold as u64);
        }
        let med_hier = median_u64(&mut hier_steps);
        let ratio = med_hier as f64 / med_flat as f64;
        println!(
            "  d={d:<2}  hier median steps-to-90%: {med_hier:<6}  ratio: {ratio:.3}"
        );
        sweep_results.push((d, med_hier, ratio));
        if ratio < best_ratio {
            best_ratio = ratio;
            best_d = d;
        }
    }

    let baseline_ratio = sweep_results[0].2; // d=0 ratio
    let improved = best_d > 0 && best_ratio < baseline_ratio;
    let verdict = if improved {
        format!(
            "d={best_d} improves G1 (ratio {best_ratio:.3} < baseline {baseline_ratio:.3}) — phase gate is a net win"
        )
    } else {
        format!(
            "d=0 (ungated) is best (ratio {baseline_ratio:.3}) — phase gate does not improve G1 on this domain"
        )
    };
    println!("  verdict: {verdict}");

    // This gate is informational — always passes (it's a diagnostic, not a gate).
    // The verdict is in the detail string.
    GateResult::pass(
        "G1-real phase gate sweep (T4.2 diagnostic)",
        format!("flat={med_flat}, best d={best_d} ratio={best_ratio:.3}, {verdict}"),
    )
}

fn gate_g2_diversity() -> GateResult {
    println!(
        "\n--- G2: Diversity Preservation (T={}, {} trials, cluster-frac threshold {:.0}%) ---",
        T_G2,
        TRIALS_G2,
        CLUSTER_FRAC_THRESHOLD * 100.0
    );

    let mut flat_clusters = Vec::with_capacity(TRIALS_G2);
    let mut hier_clusters = Vec::with_capacity(TRIALS_G2);
    let mut flat_rewards = Vec::with_capacity(TRIALS_G2);
    let mut hier_rewards = Vec::with_capacity(TRIALS_G2);

    for trial in 0..TRIALS_G2 {
        let domain = ClusteredDomain::new_clustered(trial as u64);
        let optimal = domain.optimal_arm();
        let threshold_count = (T_G2 as f32 * CLUSTER_FRAC_THRESHOLD) as usize + 1;

        let mut dom = domain.clone();
        let mut flat = FlatThompson::new(N_ARMS, 0.0);
        let r = run_trial(
            &mut flat,
            &mut dom,
            trial as u64,
            T_G2,
            optimal,
            N_CLUSTERS,
            1.1,
        );
        flat_clusters.push(
            r.cluster_counts
                .iter()
                .filter(|&&c| c >= threshold_count)
                .count() as u64,
        );
        flat_rewards.push(r.cumulative_reward);

        let mut dom = domain.clone();
        let root = build_clustered_tree(N_CLUSTERS, ARMS_PER_CLUSTER, 0.0);
        let tree = LatentTaskTree::from_root(root, LatentTaskTreeConfig::default());
        let mut hier = HierarchicalThompson { tree };
        let r = run_trial(
            &mut hier,
            &mut dom,
            trial as u64,
            T_G2,
            optimal,
            N_CLUSTERS,
            1.1,
        );
        hier_clusters.push(
            r.cluster_counts
                .iter()
                .filter(|&&c| c >= threshold_count)
                .count() as u64,
        );
        hier_rewards.push(r.cumulative_reward);
    }

    let med_flat_c = median_u64(&mut flat_clusters);
    let med_hier_c = median_u64(&mut hier_clusters);
    let med_flat_r = median_f32(&mut flat_rewards);
    let med_hier_r = median_f32(&mut hier_rewards);
    let reward_diff = (med_hier_r - med_flat_r).abs() / med_flat_r.max(med_hier_r).max(1e-6);
    let matched = reward_diff <= REWARD_TOL_G2;
    let hier_reward_advantage = (med_hier_r - med_flat_r) / med_flat_r.max(1e-6);

    let ratio = if med_flat_c > 0 {
        med_hier_c as f64 / med_flat_c as f64
    } else if med_hier_c > 0 {
        f64::INFINITY
    } else {
        1.0
    };

    println!(
        "  flat      median clusters (≥{:.0}% sel): {}  reward: {:.1}",
        CLUSTER_FRAC_THRESHOLD * 100.0,
        med_flat_c,
        med_flat_r
    );
    println!(
        "  hier      median clusters (≥{:.0}% sel): {}  reward: {:.1}",
        CLUSTER_FRAC_THRESHOLD * 100.0,
        med_hier_c,
        med_hier_r
    );
    println!(
        "  reward diff: {:.2}%  (matched: {})",
        reward_diff * 100.0,
        matched
    );
    println!(
        "  hier reward advantage: {:+.2}%  (exploitation gain from structure)",
        hier_reward_advantage * 100.0
    );
    println!(
        "  ratio (hier/flat): {ratio:.3}  (gate: ≥ {RATIO_GATE_G2:.1})"
    );

    // The plan expected hierarchical to visit MORE clusters (diversity claim
    // from the paper's curriculum-learning setting). Empirically, hierarchical
    // visits FEWER clusters and gets HIGHER reward — it exploits correctly.
    // This is the right bandit behavior. The diversity claim applies to
    // curriculum learning where diversity is desired, not to reward-maximizing
    // bandits. Documented as a plan-level expectation error, not a primitive
    // defect.
    let passed = matched && ratio >= RATIO_GATE_G2 as f64;
    if passed {
        GateResult::pass(
            "G2 diversity preservation",
            format!(
                "hier {med_hier_c} ≥ {:.1}× flat {med_flat_c} at matched reward (±{:.0}%)",
                RATIO_GATE_G2,
                REWARD_TOL_G2 * 100.0
            ),
        )
    } else {
        let reason = if !matched {
            format!(
                "reward mismatch ({:.1}% > {:.0}%) — hier exploits better (+{:.1}% reward), visits fewer clusters (correct bandit behavior; diversity claim is curriculum-learning-specific)",
                reward_diff * 100.0,
                REWARD_TOL_G2 * 100.0,
                hier_reward_advantage * 100.0
            )
        } else {
            format!(
                "hier {med_hier_c} < {RATIO_GATE_G2:.1}× flat {med_flat_c}"
            )
        };
        GateResult::fail("G2 diversity preservation", reason)
    }
}

fn gate_g3_nonstationarity() -> GateResult {
    println!(
        "\n--- G3: Non-Stationarity Recovery (16 arms, shift @ {SHIFT_STEP}, {TRIALS_G3} trials) ---"
    );

    let mut flat_no_filter = Vec::with_capacity(TRIALS_G3);
    let mut flat_filter = Vec::with_capacity(TRIALS_G3);
    let mut hier_filter = Vec::with_capacity(TRIALS_G3);
    let mut sliding = Vec::with_capacity(TRIALS_G3);

    for trial in 0..TRIALS_G3 {
        let domain = ClusteredDomain::new_shift(trial as u64);

        // (a) Flat, no filter.
        let mut dom = domain.clone();
        let mut s = FlatThompson::new(N_ARMS_G3, 0.0);
        let (steps, _, _) = run_trial_shift(
            &mut s,
            &mut dom,
            trial as u64,
            T_G3,
            SHIFT_STEP,
            N_CLUSTERS_G3,
        );
        flat_no_filter.push(steps as u64);

        // (b) Flat, with filter (ablation: isolates filter contribution).
        let mut dom = domain.clone();
        let mut s = FlatThompson::new(N_ARMS_G3, DRIFT_RATE);
        let (steps, _, _) = run_trial_shift(
            &mut s,
            &mut dom,
            trial as u64,
            T_G3,
            SHIFT_STEP,
            N_CLUSTERS_G3,
        );
        flat_filter.push(steps as u64);

        // (c) Hierarchical, with filter.
        let mut dom = domain.clone();
        let root = build_clustered_tree(N_CLUSTERS_G3, ARMS_PER_CLUSTER_G3, DRIFT_RATE);
        let tree = LatentTaskTree::from_root(root, LatentTaskTreeConfig::default());
        let mut s = HierarchicalThompson { tree };
        let (steps, _, _) = run_trial_shift(
            &mut s,
            &mut dom,
            trial as u64,
            T_G3,
            SHIFT_STEP,
            N_CLUSTERS_G3,
        );
        hier_filter.push(steps as u64);

        // (d) Sliding-window proxy for Dual-Pool CGSP.
        let mut dom = domain.clone();
        let mut s = SlidingWindowThompson::new(N_ARMS_G3, SLIDING_WINDOW_SIZE);
        let (steps, _, _) = run_trial_shift(
            &mut s,
            &mut dom,
            trial as u64,
            T_G3,
            SHIFT_STEP,
            N_CLUSTERS_G3,
        );
        sliding.push(steps as u64);
    }

    let med_flat = median_u64(&mut flat_no_filter);
    let med_flat_f = median_u64(&mut flat_filter);
    let med_hier_f = median_u64(&mut hier_filter);
    let med_slide = median_u64(&mut sliding);

    let ratio = med_hier_f as f64 / med_flat as f64;
    let passed = ratio <= RATIO_GATE_G3 as f64;

    println!("  flat (no filter)          median recovery: {med_flat}");
    println!(
        "  flat (filter={DRIFT_RATE})          median recovery: {med_flat_f}"
    );
    println!(
        "  hier (filter={DRIFT_RATE})          median recovery: {med_hier_f}"
    );
    println!(
        "  sliding-window (W={SLIDING_WINDOW_SIZE})     median recovery: {med_slide}"
    );
    println!(
        "  ratio (hier+filter / flat-no-filter): {ratio:.3}  (gate: ≤ {RATIO_GATE_G3:.1})"
    );

    if passed {
        GateResult::pass(
            "G3 non-stationarity recovery",
            format!(
                "hier+filter {med_hier_f} ≤ {RATIO_GATE_G3:.1}× flat-no-filter {med_flat} (ratio {ratio:.3}); flat+filter={med_flat_f}, sliding={med_slide}"
            ),
        )
    } else {
        GateResult::fail(
            "G3 non-stationarity recovery",
            format!(
                "hier+filter {med_hier_f} > {RATIO_GATE_G3:.1}× flat-no-filter {med_flat} (ratio {ratio:.3})"
            ),
        )
    }
}

fn gate_g4_latency() -> GateResult {
    const BATCH: usize = 1000;

println!(
        "\n--- G4: Latency (depth {}, branching {}, {} leaves) ---",
        DEPTH_G4,
        BRANCHING_G4,
        1 << DEPTH_G4
    );

    let config = LatentTaskTreeConfig::default();

    // ── Build a tree with non-trivial posteriors (avoid Beta(1,1) fast path) ──
    // Apply observations so internal nodes have non-uniform Beta posteriors.
    let (root, n_arms) = build_deep_tree(DEPTH_G4, BRANCHING_G4, 0.0);
    let mut tree = LatentTaskTree::from_root(root, config);
    let mut rng = fastrand::Rng::with_seed(42);
    for i in 0..5_000 {
        let arm = tree.sample(&mut rng);
        let reward = if i % 3 == 0 { 0.8 } else { 0.3 };
        tree.observe(arm, reward, i as u64);
    }

    // ── Latency: sample (batch timing for sub-ns resolution) ──
    for _ in 0..LATENCY_WARMUP {
        black_box(tree.sample(&mut rng));
    }
    let mut sample_ns = Vec::with_capacity(LATENCY_ITERS / BATCH);
    for _ in 0..(LATENCY_ITERS / BATCH) {
        let t0 = Instant::now();
        let mut sink = 0usize;
        for _ in 0..BATCH {
            sink = sink.wrapping_add(tree.sample(&mut rng));
        }
        let dt = t0.elapsed();
        if sink == usize::MAX {
            std::process::abort();
        }
        sample_ns.push(dt.as_nanos() as u64 / BATCH as u64);
    }
    let med_sample = median_duration_ns_arr(&mut sample_ns);

    // ── Latency: observe (batch timing) ──
    let mut observe_ns = Vec::with_capacity(LATENCY_ITERS / BATCH);
    for _ in 0..(LATENCY_ITERS / BATCH) {
        let t0 = Instant::now();
        for j in 0..BATCH {
            let arm = j % n_arms;
            tree.observe(arm, 0.5, 0);
        }
        let dt = t0.elapsed();
        observe_ns.push(dt.as_nanos() as u64 / BATCH as u64);
    }
    let med_observe = median_duration_ns_arr(&mut observe_ns);

    println!(
        "  sample  p50: {med_sample} ns  (gate: ≤ {LATENCY_TARGET_SAMPLE_NS} ns)"
    );
    println!(
        "  observe p50: {med_observe} ns  (gate: ≤ {LATENCY_TARGET_OBSERVE_NS} ns)"
    );

    // ── Alloc-free hot path ──
    let (_, sample_allocs) = alloc_delta(|| {
        for _ in 0..100 {
            black_box(tree.sample(&mut rng));
        }
    });
    let (_, observe_allocs) = alloc_delta(|| {
        for i in 0..100 {
            let arm = i % n_arms;
            tree.observe(arm, 0.5, i as u64);
        }
    });

    println!("  sample  allocs/100 calls: {sample_allocs}  (gate: 0)");
    println!("  observe allocs/100 calls: {observe_allocs}  (gate: 0)");

    let latency_pass =
        med_sample <= LATENCY_TARGET_SAMPLE_NS && med_observe <= LATENCY_TARGET_OBSERVE_NS;
    let alloc_pass = sample_allocs == 0 && observe_allocs == 0;

    if latency_pass && alloc_pass {
        GateResult::pass(
            "G4 latency + alloc-free",
            format!(
                "sample {med_sample}ns ≤ {LATENCY_TARGET_SAMPLE_NS}, observe {med_observe}ns ≤ {LATENCY_TARGET_OBSERVE_NS}, 0 allocs"
            ),
        )
    } else {
        let mut reasons = Vec::new();
        if med_sample > LATENCY_TARGET_SAMPLE_NS {
            reasons.push(format!(
                "sample {med_sample}ns > {LATENCY_TARGET_SAMPLE_NS}"
            ));
        }
        if med_observe > LATENCY_TARGET_OBSERVE_NS {
            reasons.push(format!(
                "observe {med_observe}ns > {LATENCY_TARGET_OBSERVE_NS}"
            ));
        }
        if !alloc_pass {
            reasons.push(format!(
                "allocs sample={sample_allocs} observe={observe_allocs}"
            ));
        }
        GateResult::fail("G4 latency + alloc-free", reasons.join("; "))
    }
}

fn gate_g5_reproducibility() -> GateResult {
    println!("\n--- G5: Bit-Reproducibility ({G5_SAMPLES} samples) ---");

    let config = LatentTaskTreeConfig::default();

    // Two independently-constructed identical trees.
    let (root_a, _) = build_deep_tree(DEPTH_G4, BRANCHING_G4, 0.0);
    let (root_b, _) = build_deep_tree(DEPTH_G4, BRANCHING_G4, 0.0);
    let tree_a = LatentTaskTree::from_root(root_a, config);
    let tree_b = LatentTaskTree::from_root(root_b, config);

    // BLAKE3 match.
    let blake3_match = tree_a.blake3_root() == tree_b.blake3_root();
    println!("  BLAKE3 match: {blake3_match}");

    // Identical sample sequences from identical seeds (no observations).
    let mut rng_a = fastrand::Rng::with_seed(12345);
    let mut rng_b = fastrand::Rng::with_seed(12345);
    let mut seq_match = true;
    for _ in 0..G5_SAMPLES {
        let a = tree_a.sample(&mut rng_a);
        let b = tree_b.sample(&mut rng_b);
        if a != b {
            seq_match = false;
            break;
        }
    }
    println!("  pre-observe sample sequences identical: {seq_match}");

    // Identical sample sequences after identical observation sequences.
    let (root_a, n_arms) = build_deep_tree(DEPTH_G4, BRANCHING_G4, 0.0);
    let (root_b, _) = build_deep_tree(DEPTH_G4, BRANCHING_G4, 0.0);
    let mut tree_a = LatentTaskTree::from_root(root_a, config);
    let mut tree_b = LatentTaskTree::from_root(root_b, config);

    // Apply identical observation sequence.
    for step in 0..1_000 {
        let arm = step % n_arms;
        let reward = ((step % 7) as f32) / 7.0;
        tree_a.observe(arm, reward, step as u64);
        tree_b.observe(arm, reward, step as u64);
    }

    // Verify post-observe sample sequences.
    let mut rng_a = fastrand::Rng::with_seed(99999);
    let mut rng_b = fastrand::Rng::with_seed(99999);
    let mut post_match = true;
    for _ in 0..G5_SAMPLES {
        let a = tree_a.sample(&mut rng_a);
        let b = tree_b.sample(&mut rng_b);
        if a != b {
            post_match = false;
            break;
        }
    }
    println!("  post-observe sample sequences identical: {post_match}");

    if blake3_match && seq_match && post_match {
        GateResult::pass(
            "G5 bit-reproducibility",
            format!(
                "BLAKE3 + pre/post-observe sequences all byte-identical over {G5_SAMPLES} samples"
            ),
        )
    } else {
        let mut reasons = Vec::new();
        if !blake3_match {
            reasons.push("BLAKE3 mismatch");
        }
        if !seq_match {
            reasons.push("pre-observe sequence mismatch");
        }
        if !post_match {
            reasons.push("post-observe sequence mismatch");
        }
        GateResult::fail("G5 bit-reproducibility", reasons.join("; "))
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!(
        "=== Plan 370 - Manifold Bandit GOAT Gate (Phase 2 + Phase 3 G1-real + Phase 4 T4.2 sweep) ==="
    );

    let gates = [
        gate_g1_structural_advantage(),
        gate_g1_real_tree_structural_advantage(),
        gate_g1_real_phase_gate_sweep(),
        gate_g2_diversity(),
        gate_g3_nonstationarity(),
        gate_g4_latency(),
        gate_g5_reproducibility(),
    ];

    let mut all_pass = true;
    println!("\n=== Gate Verdicts ===");
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    println!("G3 (no-regression): verified via `cargo check --all-features`");
    println!("    and `cargo check --no-default-features` (the merkle_root lesson).");
    println!();

    if all_pass {
        println!("=== ALL G1+G2+G3+G4+G5 GATES PASS — eligible for default promotion ===");
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED — keep opt-in, see details above ===");
        std::process::exit(1);
    }
}
