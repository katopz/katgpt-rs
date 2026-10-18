//! manifold_bandit — Latent Task Tree + Hierarchical Thompson Sampler.
//!
//! Distilled from McKenzie, Hansen, Wang, *Manifold Bandits: Bayesian Curriculum
//! Learning over the Latent Geometry of Large Language Models* (arXiv:2606.19750,
//! UCSD 2026). The paper's headline (BMC training curriculum for GSPO/GRPO RL on
//! Qwen3-8B) is training-only → `riir-train`. This module ships the **modelless
//! inference-time routing primitive**: a frozen, BLAKE3-committable hierarchical
//! clustering of an arm space + top-down Beta posterior descent + per-arm
//! non-stationary Bayesian filtering.
//!
//! # Three composable parts
//!
//! 1. **[`LatentTaskTree`]** — frozen hierarchical clustering of arms by latent
//!    similarity. Phase 1 accepts a pre-computed topology ([`TreeNode`]); the real
//!    PCA → UMAP → Chart Test → HDBSCAN construction pipeline ships in Phase 3.
//! 2. **Top-down Thompson descent** — [`LatentTaskTree::sample`] descends root →
//!    leaf by Thompson-sampling each child's Beta and picking the max. O(depth)
//!    Beta draws. Structure-aware: reward on one arm updates siblings + ancestors
//!    via bottom-up Empirical Bayes.
//! 3. **[`BayesianFilterArm`]** — per-arm non-stationary belief via a predict-
//!    update filter (drift toward uniform between observations). Handles the
//!    documented Plan 030 gap ("Non-stationary environments — Out of Scope").
//!    Complementary to Dual-Pool CGSP (Plan 312) — they handle different axes of
//!    non-stationarity.
//!
//! # Design notes
//!
//! - **Beta(α, β) everywhere, sigmoid never softmax** (AGENTS.md §2). The Beta
//!   posterior is conjugate to Bernoulli reward; the Thompson sample is a single
//!   Beta draw — no normalization, no softmax.
//! - **Empirical Bayes = evidence pooling.** Parent (α, β) = (1 + Σ(α_c − 1),
//!   1 + Σ(β_c − 1)) over children — pools observed successes/failures without
//!   pseudo-count dilution. Starts at Beta(1, 1) when all children are uniform
//!   (high variance → explores); concentrates correctly with data (sharper
//!   than SUM which dilutes signal with per-child pseudo-counts). See Plan 370
//!   Phase 2 GOAT gate.
//! - **Frozen tree, mutable sampler state.** Topology is immutable at inference
//!   time; only Beta posteriors drift. The tree is BLAKE3-committable at build
//!   time (topology + initial priors); runtime mutations are tracked separately.
//! - **Zero allocations after construction.** [`LatentTaskTree::sample`] and
//!   [`LatentTaskTree::observe`] are allocation-free. The arm→path lookup uses a
//!   stack-allocated `ArmPath` (Copy) to avoid the borrow-checker conflict
//!   between reading the path and mutating the tree.
//!
//! # DRY note
//!
//! The Beta sampler (`sample_beta`) is a private helper using the Gamma-ratio
//! method (Marsaglia-Tsang gamma + Box-Muller normal). The same algorithm is used
//! in `katgpt-rs/src/dense_mesh/edge_bandit.rs` (with a rough normal
//! approximation). Other copies (`katgpt-pruners/src/bandit.rs`,
//! `katgpt-rs/src/fold/fold_bandit.rs`, `katgpt-rs/src/speculative/
//! thinking_controller.rs`, `katgpt-rs/tests/partial_scoring_goat.rs`) use
//! Jöhnk's algorithm, which is correct but has catastrophically low acceptance
//! rates for large α, β (the posteriors after many observations). This module
//! uses the Gamma-ratio method for correctness across the full parameter range.
//! Consolidating all copies into a shared `katgpt-core::rng_util` is a worthwhile
//! follow-up refactor (tracked separately).
//!
//! See: `katgpt-rs/.research/370_manifold_bandits_latent_task_tree_hierarchical_thompson.md`
//! See: `katgpt-rs/.plans/370_manifold_bandit_latent_task_tree.md`
//!
//! Manifold Bandits — Latent Task Tree + Hierarchical Thompson Sampler +
//! BayesianFilterArm (Plan 370, Research 370, arXiv:2606.19750 McKenzie et al.
//! UCSD 2026). Modelless inference-time routing primitive: frozen, BLAKE3-
//! committable hierarchical clustering of an arm space + top-down Beta posterior
//! descent + per-arm non-stationary Bayesian filtering. Closes the contextual +
//! non-stationary bandit gap (Plans 030/032/025). The BMC training curriculum
//! routes to riir-train; this ships the modelless inference-time routing
//! primitive. DEFAULT-ON (Plan 370 Phase 2, 2026-07-03): G1+G3+G4+G5 PASS; G2 FAIL is plan-level expectation (curriculum-learning-specific).

// ──────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────

/// Maximum supported tree depth. Bounds the stack-allocated path buffer so that
/// [`LatentTaskTree::observe`] can copy the path without allocating.
///
/// 16 is generous — the Phase 3 construction pipeline caps at `max_depth = 6`,
/// and deeper trees blow the ≤500 ns latency budget (G4) anyway.
const MAX_DEPTH: usize = 16;

// ──────────────────────────────────────────────────────────────────────────
// ArmPath — zero-alloc path lookup for observe
// ──────────────────────────────────────────────────────────────────────────

/// Pre-computed root→leaf path as a stack-allocated Copy struct.
///
/// Indexed by arm_id at construction time. [`LatentTaskTree::observe`] copies the
/// path out of the lookup table (no borrow held), then descends the tree with
/// `&mut` access — this is the zero-alloc solution to the borrow-checker conflict
/// between reading `&self.arm_paths[arm_id]` and mutating `&mut self.root`.
#[derive(Clone, Copy, Debug)]
struct ArmPath {
    /// Child indices from root toward the leaf.
    indices: [usize; MAX_DEPTH],
    /// Number of valid entries in `indices` (= tree depth to this leaf).
    len: usize,
}

impl ArmPath {
    fn empty() -> Self {
        Self {
            indices: [0; MAX_DEPTH],
            len: 0,
        }
    }

    fn as_slice(&self) -> &[usize] {
        &self.indices[..self.len]
    }
}

// ──────────────────────────────────────────────────────────────────────────
// TreeNode
// ──────────────────────────────────────────────────────────────────────────

/// A node in the Latent Task Tree.
///
/// Internal nodes carry the aggregate Empirical Bayes Beta(α, β) posterior for
/// their subtree; leaves carry the per-arm non-stationary [`BayesianFilterArm`].
#[derive(Clone, Debug)]
pub enum TreeNode {
    /// A branching node. `beta_alpha` / `beta_beta` are the Empirical Bayes
    /// aggregate of all children's Beta parameters (mean of α_c, mean of β_c),
    /// recomputed bottom-up after each observation.
    Internal {
        /// Child subtrees (ordered by cluster id from the construction pipeline).
        children: Vec<TreeNode>,
        /// Beta(α) for "how good is this subtree" — the aggregate of children.
        beta_alpha: f32,
        /// Beta(β) companion.
        beta_beta: f32,
        /// Total observations that have passed through this node.
        n_obs: u32,
    },
    /// A terminal node — one arm in the original arm space.
    Leaf {
        /// The arm id (index into the original embedding list).
        arm_id: usize,
        /// Non-stationary belief filter for this arm.
        filter: BayesianFilterArm,
    },
}

impl TreeNode {
    /// Returns the (alpha, beta) Beta parameters for this node.
    ///
    /// - Internal: the aggregate Empirical Bayes posterior.
    /// - Leaf: the per-arm [`BayesianFilterArm`] posterior.
    fn beta_params(&self) -> (f32, f32) {
        match self {
            Self::Internal {
                beta_alpha,
                beta_beta,
                ..
            } => (*beta_alpha, *beta_beta),
            Self::Leaf { filter, .. } => (filter.alpha, filter.beta),
        }
    }

    /// Total observations that have passed through this node (Phase 4 T4.2).
    ///
    /// - Internal: the stored `n_obs` counter.
    /// - Leaf: 0 (leaves don't aggregate; their filter tracks per-arm evidence
    ///   via `alpha`/`beta` instead).
    ///
    /// Used by the R279 N≥d phase gate: a child subtree with `n_obs <
    /// phase_gate_min_obs` is skipped during Empirical Bayes aggregation.
    pub fn n_obs(&self) -> u32 {
        match self {
            Self::Internal { n_obs, .. } => *n_obs,
            Self::Leaf { .. } => 0,
        }
    }

    /// Convenience: construct a leaf with a fresh uniform-prior filter.
    pub fn leaf(arm_id: usize, drift_rate: f32) -> Self {
        Self::Leaf {
            arm_id,
            filter: BayesianFilterArm::new(drift_rate),
        }
    }

    /// Convenience: construct an internal node whose Empirical Bayes aggregate
    /// pools the **evidence** (observed successes/failures) from children:
    /// `parent_α = 1 + Σ(child_α - 1)`, `parent_β = 1 + Σ(child_β - 1)`.
    ///
    /// This is the standard Beta-Bernoulli evidence pooling: each child's
    /// pseudocount (the +1 prior) is subtracted before summing, then a single
    /// +1 is added back. The result starts at Beta(1, 1) when all children are
    /// uniform (high variance → explores), and concentrates faster than either
    /// SUM (Beta(N, N) — over-concentrated initially, signal diluted by
    /// pseudo-counts) or MEAN (never concentrates — too diffuse). See Plan 370
    /// Phase 2 GOAT gate.
    pub fn internal(children: Vec<Self>) -> Self {
        let n = children.len() as f32;
        let (a, b) = children.iter().fold((0.0_f32, 0.0_f32), |(sa, sb), c| {
            let (ca, cb) = c.beta_params();
            (sa + ca, sb + cb)
        });
        // Evidence pooling: subtract per-child pseudocount, add back one.
        Self::Internal {
            children,
            beta_alpha: (a - n + 1.0).max(1.0),
            beta_beta: (b - n + 1.0).max(1.0),
            n_obs: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// BayesianFilterArm
// ──────────────────────────────────────────────────────────────────────────

/// Per-arm non-stationary belief via a predict-update Bayesian Filter.
///
/// The "predict" step decays the Beta posterior toward uniform Beta(1, 1)
/// proportional to elapsed steps since the last observation — modeling belief
/// drift in a non-stationary environment. The "update" step applies the standard
/// Beta-Bernoulli conjugate update on the observed reward.
///
/// With `drift_rate = 0.0`, `predict` is a no-op and the arm degenerates to a
/// stationary Beta posterior (identical to flat Thompson from Plan 030).
///
/// Composes with Dual-Pool CGSP (Plan 312): Dual-Pool handles pool-level strategy
/// switches; this handles per-arm belief drift. They operate on different axes
/// of non-stationarity.
#[derive(Clone, Copy, Debug)]
pub struct BayesianFilterArm {
    /// Beta(α) — successes + pseudocount. Starts at 1.0 (uniform prior).
    pub alpha: f32,
    /// Beta(β) — failures + pseudocount. Starts at 1.0 (uniform prior).
    pub beta: f32,
    /// Drift rate λ ∈ [0, 1). 0 = stationary (no drift). Higher = faster forgetting.
    pub drift_rate: f32,
    /// The step at which the filter was last touched (predict or update).
    pub last_obs_step: u64,
}

impl BayesianFilterArm {
    /// Create a fresh arm with uniform Beta(1, 1) prior and the given drift rate.
    pub fn new(drift_rate: f32) -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            drift_rate: drift_rate.clamp(0.0, 0.999),
            last_obs_step: 0,
        }
    }

    /// Predict step: decay belief toward uniform Beta(1, 1).
    ///
    /// Applies `elapsed` steps of geometric decay:
    /// ```text
    /// decay = (1 - λ)^elapsed
    /// α' = α · decay + (1 - decay)     ← pulls toward 1
    /// β' = β · decay + (1 - decay)
    /// ```
    /// With `drift_rate = 0` or `elapsed = 0`, this is a no-op.
    pub fn predict(&mut self, current_step: u64) {
        let elapsed = current_step.saturating_sub(self.last_obs_step);
        if elapsed == 0 || self.drift_rate <= 0.0 {
            return;
        }
        let decay = (1.0 - self.drift_rate).powi(elapsed as i32);
        self.alpha = self.alpha * decay + (1.0 - decay);
        self.beta = self.beta * decay + (1.0 - decay);
        self.last_obs_step = current_step;
    }

    /// Update step: Beta-Bernoulli conjugate update.
    ///
    /// `reward` is clamped to `[0, 1]`: `α += r`, `β += (1 - r)`.
    /// Updates `last_obs_step` so the next `predict` measures drift from here.
    pub fn update(&mut self, reward: f32, current_step: u64) {
        let r = reward.clamp(0.0, 1.0);
        self.alpha += r;
        self.beta += 1.0 - r;
        self.last_obs_step = current_step;
    }

    /// Thompson sample: draw from Beta(α, β) via Jöhnk's algorithm.
    pub fn thompson_sample(&self, rng: &mut fastrand::Rng) -> f32 {
        sample_beta(self.alpha, self.beta, rng)
    }

    /// Posterior mean `α / (α + β)` — useful for diagnostics, not the hot path.
    pub fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }
}

impl Default for BayesianFilterArm {
    fn default() -> Self {
        Self::new(0.01)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// LatentTaskTreeConfig
// ──────────────────────────────────────────────────────────────────────────

/// Configuration for [`LatentTaskTree`] construction.
///
/// Fields `pca_dim` through `hdbscan_min_cluster` configure the Phase 3
/// construction pipeline (PCA → UMAP → Chart Test → HDBSCAN). In Phase 1 they are
/// stored but unused — the tree topology is supplied directly via
/// [`LatentTaskTree::from_root`].
#[derive(Clone, Copy, Debug)]
pub struct LatentTaskTreeConfig {
    /// Reduce embeddings to this dimensionality before UMAP (Phase 3). Default: 16.
    pub pca_dim: usize,
    /// Fixed seed for UMAP determinism (Phase 3). Default: 42.
    pub umap_seed: u64,
    /// Manifold locality threshold for the Chart Test (Phase 3). Default: 0.85.
    pub chart_test_threshold: f32,
    /// Minimum cluster size for HDBSCAN (Phase 3). Default: 4.
    pub hdbscan_min_cluster: usize,
    /// Tree depth cap. Default: 6 (keeps sample/observe within the G4 latency
    /// budget; deeper trees blow ≤500 ns per sample).
    pub max_depth: usize,
    /// Default drift rate λ for new [`BayesianFilterArm`] instances. Default: 0.01.
    pub filter_drift_rate: f32,
    /// R279 N≥d phase gate for Empirical Bayes aggregation (Phase 4 T4.2).
    ///
    /// When > 0, a child subtree whose `n_obs < phase_gate_min_obs` is **skipped**
    /// during bottom-up aggregation — its posterior is below the R279 phase
    /// transition (N<d), so it's noise rather than evidence. Only children with
    /// enough observations to support a stable posterior contribute to the
    /// parent's aggregate. When 0, the gate is disabled (Phase 1–3 behavior:
    /// all children aggregated). Default: 0 (disabled).
    ///
    /// Composes with [`crate::subspace_phase_gate::phase_transition_gate`]:
    /// `phase_transition_gate(child.n_obs(), phase_gate_min_obs as usize)` is
    /// exactly the per-child inclusion predicate. The gate is modelless
    /// (deterministic threshold on integer counts) and zero-alloc.
    pub phase_gate_min_obs: u32,
}

impl Default for LatentTaskTreeConfig {
    fn default() -> Self {
        Self {
            pca_dim: 16,
            umap_seed: 42,
            chart_test_threshold: 0.85,
            hdbscan_min_cluster: 4,
            max_depth: 6,
            filter_drift_rate: 0.01,
            phase_gate_min_obs: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// LatentTaskTree
// ──────────────────────────────────────────────────────────────────────────

/// The frozen, BLAKE3-committable Latent Task Tree + its sampler state.
///
/// The tree topology (the [`TreeNode`] structure) is immutable at inference time
/// — only the Beta posteriors drift. The [`blake3_root`](Self::blake3_root)
/// commits the **initial** topology + priors (computed once at construction);
/// runtime mutations are the mutable sampler state, tracked separately for
/// freeze/thaw snapshots.
pub struct LatentTaskTree {
    root: TreeNode,
    config: LatentTaskTreeConfig,
    blake3: [u8; 32],
    /// arm_id → root-to-leaf path. Built once at construction; zero-alloc reads.
    arm_paths: Vec<ArmPath>,
}

impl LatentTaskTree {
    /// Construct from a pre-built tree topology (Phase 1 entry point).
    ///
    /// Stamps uniform Beta(1, 1) priors (via [`BayesianFilterArm::new`] on each
    /// leaf), builds the arm→path lookup, and computes the BLAKE3 commitment of
    /// the initial tree state.
    ///
    /// The full `build(embeddings, config)` pipeline (PCA → UMAP → Chart Test →
    /// HDBSCAN) ships in Phase 3.
    pub fn from_root(root: TreeNode, config: LatentTaskTreeConfig) -> Self {
        // Collect arm paths for O(1) observe lookup.
        let mut arm_paths: Vec<ArmPath> = Vec::new();
        let mut current_path: Vec<usize> = Vec::with_capacity(MAX_DEPTH);
        Self::collect_arm_paths(&root, &mut current_path, &mut arm_paths);

        // Compute BLAKE3 of the initial tree (topology + initial Beta priors).
        let mut hasher = blake3::Hasher::new();
        Self::compute_blake3(&root, &mut hasher);
        let blake3 = *hasher.finalize().as_bytes();

        Self {
            root,
            config,
            blake3,
            arm_paths,
        }
    }

    /// Thompson-sample a leaf (arm_id) by descending the tree.
    ///
    /// At each internal node, draws a Beta sample from each child's posterior and
    /// descends into the child with the highest sample. O(branching × depth) Beta
    /// draws — typically 4–6 levels × 4–8 children.
    ///
    /// Zero allocations.
    pub fn sample(&self, rng: &mut fastrand::Rng) -> usize {
        let mut node = &self.root;
        loop {
            match node {
                TreeNode::Leaf { arm_id, .. } => return *arm_id,
                TreeNode::Internal { children, .. } => {
                    if children.is_empty() {
                        // Degenerate internal node with no children — should not
                        // happen in a well-formed tree, but avoid an infinite loop.
                        panic!("manifold_bandit: Internal node with no children");
                    }
                    let mut best_idx = 0usize;
                    let mut best_sample = f32::NEG_INFINITY;
                    for (i, child) in children.iter().enumerate() {
                        let (a, b) = child.beta_params();
                        let s = sample_beta(a, b, rng);
                        if s > best_sample {
                            best_sample = s;
                            best_idx = i;
                        }
                    }
                    node = &children[best_idx];
                }
            }
        }
    }

    /// Observe a reward on an arm.
    ///
    /// Descends to the leaf for `arm_id`, runs the predict-update Bayesian filter,
    /// then propagates bottom-up via Empirical Bayes (parent α, β = mean of
    /// children's α, β). O(branching × depth) — the Empirical Bayes recompute
    /// iterates all children at each level.
    ///
    /// Zero allocations (the path is copied from the pre-computed lookup as a
    /// stack-local `ArmPath`).
    ///
    /// # Panics
    /// Panics if `arm_id` is not present in the tree.
    pub fn observe(&mut self, arm_id: usize, reward: f32, current_step: u64) {
        assert!(
            arm_id < self.arm_paths.len(),
            "manifold_bandit: arm_id {arm_id} not in tree ({} arms)",
            self.arm_paths.len()
        );
        // Copy the path (no borrow held on self after this line).
        let path = self.arm_paths[arm_id];
        Self::observe_recursive(
            &mut self.root,
            path.as_slice(),
            reward,
            current_step,
            self.config.phase_gate_min_obs,
        );
    }

    /// BLAKE3 commitment of the frozen tree (topology + initial Beta priors at
    /// construction time).
    ///
    /// This is the **initial** commitment — it does NOT reflect runtime Beta
    /// mutations. Use it for freeze/thaw integrity envelopes: snapshot the current
    /// Beta state separately, and verify the frozen topology matches on thaw.
    #[inline]
    pub fn blake3_root(&self) -> [u8; 32] {
        self.blake3
    }

    /// Number of arms (leaves) in the tree.
    pub fn num_arms(&self) -> usize {
        self.arm_paths.len()
    }

    /// Read-only access to the config.
    pub fn config(&self) -> &LatentTaskTreeConfig {
        &self.config
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Recursively collect root→leaf paths for every arm_id.
    fn collect_arm_paths(node: &TreeNode, current_path: &mut Vec<usize>, paths: &mut Vec<ArmPath>) {
        match node {
            TreeNode::Leaf { arm_id, .. } => {
                while paths.len() <= *arm_id {
                    paths.push(ArmPath::empty());
                }
                let mut buf = [0usize; MAX_DEPTH];
                for (i, &idx) in current_path.iter().enumerate() {
                    buf[i] = idx;
                }
                paths[*arm_id] = ArmPath {
                    indices: buf,
                    len: current_path.len(),
                };
            }
            TreeNode::Internal { children, .. } => {
                for (i, child) in children.iter().enumerate() {
                    current_path.push(i);
                    Self::collect_arm_paths(child, current_path, paths);
                    current_path.pop();
                }
            }
        }
    }

    /// Recursively descend following `path`, update the leaf filter, then
    /// recompute each ancestor's Empirical Bayes aggregate on the way back up.
    ///
    /// The R279 N≥d phase gate (`phase_gate_min_obs > 0`) skips children whose
    /// subtree `n_obs` is below the threshold during aggregation — their
    /// posterior is below the phase transition and contributes noise rather
    /// than evidence. When `phase_gate_min_obs == 0`, all children aggregate
    /// (Phase 1–3 behavior).
    fn observe_recursive(
        node: &mut TreeNode,
        path: &[usize],
        reward: f32,
        step: u64,
        phase_gate_min_obs: u32,
    ) {
        match node {
            TreeNode::Leaf { filter, .. } => {
                filter.predict(step);
                filter.update(reward, step);
            }
            TreeNode::Internal {
                children,
                beta_alpha,
                beta_beta,
                n_obs,
            } => {
                let child_idx = path[0];
                Self::observe_recursive(
                    &mut children[child_idx],
                    &path[1..],
                    reward,
                    step,
                    phase_gate_min_obs,
                );
                // Recompute this node's aggregate via EVIDENCE pooling: subtract
                // each active child's pseudocount before summing, add back one.
                // This gives the parent the total observed evidence (successes /
                // failures) without dilution from unobserved children.
                //
                // R279 phase gate: when `phase_gate_min_obs > 0`, an INTERNAL child
                // subtree with `n_obs < phase_gate_min_obs` is skipped — its
                // posterior is below the N≥d phase transition (noise, not
                // evidence). LEAF children are always included: a leaf IS the
                // atomic observation (intrinsic dim d=1, satisfied trivially by
                // any evidence), so gating it would prevent leaf evidence from
                // ever reaching the root. When all internal children are gated
                // out, the parent falls back to evidence from leaf children only
                // (or Beta(1, 1) if there are no leaves).
                let mut sum_a = 0.0_f32;
                let mut sum_b = 0.0_f32;
                let mut n_active = 0u32;
                for c in children.iter() {
                    let gated = phase_gate_min_obs > 0
                        && matches!(c, TreeNode::Internal { .. })
                        && c.n_obs() < phase_gate_min_obs;
                    if gated {
                        continue;
                    }
                    let (ca, cb) = c.beta_params();
                    sum_a += ca;
                    sum_b += cb;
                    n_active += 1;
                }
                let n_active_f = n_active as f32;
                *beta_alpha = (sum_a - n_active_f + 1.0).max(1.0);
                *beta_beta = (sum_b - n_active_f + 1.0).max(1.0);
                *n_obs += 1;
            }
        }
    }

    /// Recursively hash the tree topology + initial Beta priors into `hasher`.
    ///
    /// Canonical pre-order traversal: Internal nodes tagged `0x01`, leaves tagged
    /// `0x02`. All numeric fields in little-endian bytes for cross-platform
    /// determinism.
    fn compute_blake3(node: &TreeNode, hasher: &mut blake3::Hasher) {
        match node {
            TreeNode::Leaf { arm_id, filter } => {
                hasher.update(&[0x02]);
                hasher.update(&arm_id.to_le_bytes());
                hasher.update(&filter.alpha.to_le_bytes());
                hasher.update(&filter.beta.to_le_bytes());
                hasher.update(&filter.drift_rate.to_le_bytes());
            }
            TreeNode::Internal {
                children,
                beta_alpha,
                beta_beta,
                n_obs,
            } => {
                hasher.update(&[0x01]);
                hasher.update(&beta_alpha.to_le_bytes());
                hasher.update(&beta_beta.to_le_bytes());
                hasher.update(&n_obs.to_le_bytes());
                hasher.update(&(children.len() as u32).to_le_bytes());
                for child in children.iter() {
                    Self::compute_blake3(child, hasher);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Beta sampling — Gamma-ratio method (Marsaglia-Tsang + Box-Muller)
// ─────────────────────────────────────────────────────────────────────

/// Sample from Beta(α, β) via the Gamma-ratio identity:
/// `Beta(α,β) = Gamma(α,1) / (Gamma(α,1) + Gamma(β,1))`.
///
/// Correct for all α, β > 0. Our regime is α, β ≥ 1 (the +1 pseudocount
/// guarantees this), so we use Marsaglia-Tsang for shape ≥ 1.
///
/// **Why not Jöhnk's:** the other copies in this codebase use Jöhnk's algorithm,
/// which has acceptance rate Γ(α+1)Γ(β+1)/Γ(α+β+1) — this drops to ~0.001% for
/// the posteriors that arise after 10+ observations (e.g. Beta(16, 6)). The
/// Gamma-ratio method has >90% acceptance per iteration regardless of α, β.
///
/// See the module-level DRY note — this is a private copy because `katgpt-core`
/// cannot import from `katgpt-pruners` (wrong dependency direction).
fn sample_beta(alpha: f32, beta: f32, rng: &mut fastrand::Rng) -> f32 {
    // Uniform prior Beta(1, 1): just return a uniform draw.
    if (alpha - 1.0).abs() < f32::EPSILON && (beta - 1.0).abs() < f32::EPSILON {
        return rng.f32();
    }
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    x / (x + y)
}

/// Sample from Gamma(shape, scale=1) via the Marsaglia-Tsang squeeze method.
///
/// For `shape ≥ 1` the acceptance rate is >90% per iteration (typically 1–2
/// iterations). For `shape < 1` (should not arise with our +1 pseudocount, but
/// handled for correctness), the boost trick `gamma(s) = gamma(s+1) · U^(1/s)`
/// is applied.
fn sample_gamma(shape: f32, rng: &mut fastrand::Rng) -> f32 {
    if shape < 1.0 {
        let g = sample_gamma(shape + 1.0, rng);
        let u = rng.f32().max(f32::EPSILON);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.f32().max(f32::EPSILON);
        // Squeeze (fast acceptance).
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v3;
        }
        // Full acceptance check.
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

/// Standard normal variate via Box-Muller transform.
///
/// Returns `N(0, 1)` using two uniform draws. Uses `cos` (discards the `sin`
/// component — a future micro-optimization could cache the second variate).
#[inline]
fn standard_normal(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(f32::EPSILON);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    r * (2.0 * std::f32::consts::PI * u2).cos()
}

// ──────────────────────────────────────────────────────────────────────────
// Phase 3 — Real Tree Construction Pipeline
// (PCA → 2D embed → Chart Test → DBSCAN → recursive subdivision)
// ──────────────────────────────────────────────────────────────────────────
//
// # Modelless contract
//
// The entire construction pipeline is deterministic and gradient-free:
// - PCA via power iteration with Hotelling deflation (numerical linear algebra).
// - 2D embedding via PCA-to-2D (deterministic linear projection — see T3.2 note).
// - Chart test via local PCA residual (deterministic).
// - DBSCAN with adaptive ε from median kNN distance (deterministic density
//   clustering, no k parameter).
//
// # Design decisions (T3.2, T3.4 — plan explicitly deferred these to Phase 3)
//
// **UMAP substitute (T3.2):** PCA-to-2D as the deterministic, modelless,
// zero-dep substitute. The sampler only needs "deterministic + preserves local
// neighborhoods" — PCA preserves global structure and separates well-separated
// clusters. Spectral embedding (Laplacian eigenmaps) is a Phase 3.5 upgrade if
// T3.6 shows insufficient structural advantage. The `umap_seed` config field
// seeds the power-iteration initial vector (avoids pathological starts); it is
// NOT consumed by a stochastic UMAP optimizer.
//
// **HDBSCAN substitute (T3.4):** adaptive-ε DBSCAN where ε = median kNN
// distance. This adapts to data density without the full HDBSCAN
// mutual-reachability MST hierarchy. The recursive tree construction (cluster
// → recurse on each cluster) recovers the hierarchical structure that HDBSCAN
// would produce natively.

/// Maximum power-iteration steps before giving up.
const PCA_MAX_ITERS: usize = 200;
/// Convergence threshold for power iteration (1 − cosine similarity).
const PCA_CONVERGENCE: f32 = 1e-6;
/// Default k for kNN in chart test and DBSCAN's adaptive-ε estimation.
const DEFAULT_KNN_K: usize = 15;

/// Deterministic PRNG for construction (SplitMix64).
///
/// Same algorithm as `factorized_action::codebook::SplitMix64` — kept private
/// here to avoid cross-module coupling. Used only for power-iteration
/// initialization (a non-uniform start vector avoids pathological slow
/// convergence on degenerate covariance structures).
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}

/// Squared Euclidean distance between two equal-length slices.
///
/// 4-way unrolled accumulation for auto-vectorization.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

/// Normalize a vector in-place. If the norm is ~0, leaves it unchanged.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-30 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

// ── T3.1: PCA via power iteration with Hotelling deflation ──────────────

/// Reduce N×D row-major `data` to N×`n_components` via PCA.
///
/// Centers the data, computes the D×D covariance matrix, and extracts the top
/// `n_components` eigenvectors via power iteration with Hotelling deflation.
/// Writes the projected data (N×`n_components`, row-major) into `out`.
///
/// Deterministic given (data, n, n_components, seed). No gradient descent.
///
/// # Panics
///
/// Panics (debug) if `data.len() != n * d`, `out.len() != n * n_components`,
/// or `n_components > min(d, n−1)`.
fn pca_into(data: &[f32], n: usize, n_components: usize, out: &mut [f32], seed: u64) {
    assert!(!data.is_empty() && n > 0, "pca_into: empty input");
    assert_eq!(
        data.len() % n,
        0,
        "pca_into: data.len() {} not divisible by n {}",
        data.len(),
        n
    );
    let d = data.len() / n;
    let max_components = d.min(n.saturating_sub(1)).max(1);
    assert!(
        n_components <= max_components,
        "pca_into: n_components {n_components} > min(d={d}, n−1)={max_components}"
    );
    assert_eq!(out.len(), n * n_components, "pca_into: out length mismatch");

    if n_components == 0 {
        return;
    }

    // 1. Center the data (subtract per-dimension mean).
    let mut centered = vec![0.0f32; n * d];
    let mut mean = vec![0.0f32; d];
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    let inv_n = 1.0f32 / n as f32;
    for m in mean.iter_mut().take(d) {
        *m *= inv_n;
    }
    for i in 0..n {
        let src = &data[i * d..(i + 1) * d];
        let dst = &mut centered[i * d..(i + 1) * d];
        for j in 0..d {
            dst[j] = src[j] - mean[j];
        }
    }

    // 2. Covariance C = X^T X / (n−1) (D×D, symmetric).
    let denom = (n as f32 - 1.0).max(1.0);
    let mut cov = vec![0.0f32; d * d];
    for i in 0..d {
        for j in i..d {
            let mut s = 0.0f32;
            for k in 0..n {
                s += centered[k * d + i] * centered[k * d + j];
            }
            cov[i * d + j] = s / denom;
            cov[j * d + i] = cov[i * d + j]; // Symmetric.
        }
    }

    // 3. Power iteration with Hotelling deflation.
    // Pre-allocate two work vectors outside both loops — the inner
    // iteration loop used to allocate a fresh `Vec<f32>` of size `d` per
    // iteration (PCA_MAX_ITERS × n_components allocations).
    let mut work_cov = cov;
    let mut eigenvectors = vec![0.0f32; n_components * d];
    let mut rng = SplitMix64::new(seed);
    let mut v: Vec<f32> = Vec::with_capacity(d);
    let mut new_v: Vec<f32> = vec![0.0f32; d];

    for c in 0..n_components {
        // Deterministic non-uniform initial vector.
        v.clear();
        v.extend((0..d).map(|_| rng.next_f32() * 2.0 - 1.0));
        normalize(&mut v);

        for _ in 0..PCA_MAX_ITERS {
            // v ← C v  (SIMD-accelerated per-row dot product)
            for i in 0..d {
                let row = &work_cov[i * d..(i + 1) * d];
                new_v[i] = crate::simd::simd_dot_f32(row, &v[..d], d);
            }
            normalize(&mut new_v);

            // Convergence: cosine similarity → 1.
            let dot: f32 = crate::simd::simd_dot_f32(&v[..d], &new_v[..d], d);
            let converged = (dot.abs() - 1.0).abs() < PCA_CONVERGENCE;
            std::mem::swap(&mut v, &mut new_v);
            if converged {
                break;
            }
        }

        // Eigenvalue λ = v^T C v.
        // Reuse new_v as scratch for C v (already normalized, but we're
        // about to overwrite it).
        for i in 0..d {
            let row = &work_cov[i * d..(i + 1) * d];
            new_v[i] = crate::simd::simd_dot_f32(row, &v[..d], d);
        }
        let lambda: f32 = crate::simd::simd_dot_f32(&v[..d], &new_v[..d], d);

        // Deflate: C ← C − λ v v^T.
        for i in 0..d {
            let vi = v[i];
            let row = &mut work_cov[i * d..(i + 1) * d];
            for j in 0..d {
                row[j] -= lambda * vi * v[j];
            }
        }

        eigenvectors[c * d..(c + 1) * d].copy_from_slice(&v[..d]);
    }

    // 4. Project: out = centered × eigenvectors^T  (SIMD-accelerated).
    for i in 0..n {
        let centered_row = &centered[i * d..(i + 1) * d];
        for c in 0..n_components {
            let eigen_row = &eigenvectors[c * d..(c + 1) * d];
            out[i * n_components + c] = crate::simd::simd_dot_f32(centered_row, eigen_row, d);
        }
    }
}

// ── T3.2: 2D embedding (PCA-to-2D, UMAP substitute) ─────────────────────

/// Embed N D-dim points into 2D via PCA-to-2D.
///
/// This is the deterministic, modelless substitute for UMAP (see module-level
/// design note). For D > 2 the data is projected directly to its top-2
/// principal components. Returns N 2D points.
fn embed_2d(data: &[f32], n: usize, dim: usize, seed: u64) -> Vec<[f32; 2]> {
    assert_eq!(data.len(), n * dim);
    let target = 2.min(dim).min(n.saturating_sub(1)).max(1);
    let mut out = vec![0.0f32; n * 2];
    pca_into(data, n, target, &mut out, seed);
    // If only 1 component, second dim stays 0.
    (0..n).map(|i| [out[i * 2], out[i * 2 + 1]]).collect()
}

// ── T3.3: Chart Test (manifold locality, diagnostic) ────────────────────

/// Chart test: for each point, check local linear structure via kNN PCA.
///
/// For each point, finds its k nearest neighbors, computes the eigenvalue
/// ratio λ₂/λ₁ of the local 2D neighborhood covariance. A high ratio means
/// the neighborhood is "round" (inside a cluster); a low ratio means it's
/// "elongated" (spanning clusters / between-cluster noise).
///
/// Returns a noise mask: `true` = likely noise (elongated neighborhood).
///
/// **Note:** in the current pipeline this is computed as a diagnostic — DBSCAN
/// has its own noise detection. The chart test can be enabled as a pre-filter
/// in a Phase 3.5 upgrade if tighter noise rejection is needed.
fn chart_test(points: &[[f32; 2]], k: usize, threshold: f32) -> Vec<bool> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }
    let kk = k.min(n.saturating_sub(1)).max(1);

    let mut noise = vec![false; n];
    for i in 0..n {
        // Find kk nearest neighbors of point i.
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, sq_dist(&points[i], &points[j])))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Local PCA on the k+1 points (point i + its kk neighbors).
        let mut local: Vec<[f32; 2]> = Vec::with_capacity(kk + 1);
        local.push(points[i]);
        for (j, _) in dists.iter().take(kk) {
            local.push(points[*j]);
        }

        let flat: Vec<f32> = local.iter().flat_map(|p| p.iter().copied()).collect();
        let m = flat.len() / 2;
        let nc = 2.min(m.saturating_sub(1));
        if nc < 2 {
            continue; // Can't compute ratio with < 2 components.
        }
        let mut eig = [0.0f32, 0.0f32];
        // Reuse pca_into for eigenvalues via the covariance diagonal entries.
        // Actually, compute the 2×2 covariance eigenvalues directly.
        let (cx, cy, cxy) = cov_2d(&flat, m);
        let (l1, l2) = eig_2x2_sym(cx, cy, cxy);
        eig[0] = l1;
        eig[1] = l2;

        // Ratio λ₂/λ₁ (l1 ≥ l2). High ratio → round → on-manifold.
        let ratio = if eig[0] > 1e-30 { eig[1] / eig[0] } else { 0.0 };
        // Noise if ratio < threshold (elongated neighborhood).
        noise[i] = ratio < threshold;
    }
    noise
}

/// 2D covariance of m points (flat row-major x,y pairs). Returns (cxx, cyy, cxy).
fn cov_2d(flat: &[f32], m: usize) -> (f32, f32, f32) {
    let mut mx = 0.0f32;
    let mut my = 0.0f32;
    for i in 0..m {
        mx += flat[i * 2];
        my += flat[i * 2 + 1];
    }
    mx /= m as f32;
    my /= m as f32;
    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for i in 0..m {
        let dx = flat[i * 2] - mx;
        let dy = flat[i * 2 + 1] - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    let denom = (m as f32 - 1.0).max(1.0);
    (cxx / denom, cyy / denom, cxy / denom)
}

/// Eigenvalues of a 2×2 symmetric matrix [[cxx, cxy], [cxy, cyy]], descending.
fn eig_2x2_sym(cxx: f32, cyy: f32, cxy: f32) -> (f32, f32) {
    let tr = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = ((tr * tr / 4.0) - det).max(0.0);
    let sq = disc.sqrt();
    (tr / 2.0 + sq, tr / 2.0 - sq)
}

// ── T3.4: Adaptive-ε DBSCAN (HDBSCAN substitute) ────────────────────────

/// Cluster N 2D points via DBSCAN with ε = median kNN distance.
///
/// ε adapts to the data density: the median of each point's k-th nearest
/// neighbor distance. This avoids the classic DBSCAN ε-tuning problem.
///
/// Returns a vector of `Option<cluster_id>` — `None` for noise points.
///
/// Deterministic (stable sort, no randomness).
fn dbscan_adaptive(points: &[[f32; 2]], min_pts: usize) -> Vec<Option<usize>> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }
    if n < min_pts {
        // All noise — too few points.
        return vec![None; n];
    }

    // Adaptive ε: median of k-th nearest neighbor distances (k = min_pts).
    let k = min_pts.min(n.saturating_sub(1));
    let mut knn_dists: Vec<f32> = (0..n)
        .map(|i| {
            let mut dists: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| sq_dist(&points[i], &points[j]).sqrt())
                .collect();
            dists.sort_by(|a, b| a.total_cmp(b));
            dists[k.saturating_sub(1)]
        })
        .collect();
    knn_dists.sort_by(|a, b| a.total_cmp(b));
    let eps = knn_dists[knn_dists.len() / 2];

    if eps <= 0.0 {
        // All points co-located — one big cluster.
        return vec![Some(0); n];
    }

    // Pre-compute ε-neighborhoods.
    let neighborhoods: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && sq_dist(&points[i], &points[j]) <= eps * eps)
                .collect()
        })
        .collect();

    let mut labels = vec![None; n];
    let mut cluster_id = 0usize;

    for i in 0..n {
        if labels[i].is_some() {
            continue;
        }
        if neighborhoods[i].len() + 1 < min_pts {
            // Not a core point — leave as noise (may be reclaimed as border).
            continue;
        }

        // Start new cluster, BFS-expand.
        labels[i] = Some(cluster_id);
        let mut queue: Vec<usize> = neighborhoods[i].clone();
        let mut head = 0usize;
        while head < queue.len() {
            let j = queue[head];
            head += 1;
            if labels[j].is_none() {
                labels[j] = Some(cluster_id);
                // If j is also a core point, expand its neighborhood.
                if neighborhoods[j].len() + 1 >= min_pts {
                    queue.extend_from_slice(&neighborhoods[j]);
                }
            }
        }
        cluster_id += 1;
    }

    labels
}

// ── T3.5: Recursive tree construction ────────────────────────────────────

/// Effective PCA target dimension before 2D embedding.
///
/// Capped at `min(pca_dim, dim, n−1)` — can't extract more principal
/// components than the rank of the centered data.
fn effective_pca_dim(pca_dim: usize, dim: usize, n: usize) -> usize {
    pca_dim.min(dim).min(n.saturating_sub(1)).max(1)
}

/// Recursively build a [`TreeNode`] subtree from a subset of embeddings.
///
/// `arm_indices` are indices into the original `embeddings` array (the arm_ids).
/// At each level: PCA → 2D embed → chart test (diagnostic) → DBSCAN → recurse
/// on each cluster. Base case: cluster too small or max depth → leaf group.
fn build_recursive(
    arm_indices: &[usize],
    embeddings: &[Vec<f32>],
    config: &LatentTaskTreeConfig,
    depth: usize,
) -> TreeNode {
    let n = arm_indices.len();
    let drift = config.filter_drift_rate;

    // Base case: too few points or max depth → make leaves.
    if n <= config.hdbscan_min_cluster || depth >= config.max_depth {
        return make_leaf_or_single(arm_indices, drift);
    }

    let dim = embeddings[0].len();

    // Flatten the subset into row-major f32.
    let subset: Vec<f32> = arm_indices
        .iter()
        .flat_map(|&i| embeddings[i].iter().copied())
        .collect();

    // PCA pre-reduction (dim → pca_dim) if beneficial.
    let pca_target = effective_pca_dim(config.pca_dim, dim, n);
    let pca_data: Vec<f32> = if dim > pca_target {
        let mut reduced = vec![0.0f32; n * pca_target];
        pca_into(
            &subset,
            n,
            pca_target,
            &mut reduced,
            config.umap_seed.wrapping_add(depth as u64),
        );
        reduced
    } else {
        subset
    };
    let pca_dim_actual = if dim > pca_target { pca_target } else { dim };

    // 2D embedding (PCA-to-2D).
    let embedded = embed_2d(
        &pca_data,
        n,
        pca_dim_actual,
        config.umap_seed.wrapping_add(depth as u64).wrapping_add(1),
    );

    // Chart test (diagnostic — computed but not used for filtering in this phase).
    let _noise_mask = chart_test(&embedded, DEFAULT_KNN_K, config.chart_test_threshold);

    // DBSCAN clustering.
    let clusters = dbscan_adaptive(&embedded, config.hdbscan_min_cluster);

    // Count clusters.
    let n_clusters = clusters
        .iter()
        .filter_map(|&c| c)
        .map(|c| c + 1)
        .max()
        .unwrap_or(0);

    if n_clusters <= 1 {
        // No meaningful subdivision → flat leaf group.
        return make_leaf_or_single(arm_indices, drift);
    }

    // Assign points to clusters (noise → nearest cluster by distance).
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (local_idx, &global_idx) in arm_indices.iter().enumerate() {
        let cluster = clusters[local_idx]
            .unwrap_or_else(|| nearest_cluster_idx(local_idx, &embedded, &clusters));
        cluster_members[cluster].push(global_idx);
    }

    // Recurse on each non-empty cluster.
    let children: Vec<TreeNode> = cluster_members
        .iter()
        .filter(|m| !m.is_empty())
        .map(|members| build_recursive(members, embeddings, config, depth + 1))
        .collect();

    if children.len() <= 1 {
        // DBSCAN found multiple cluster IDs but after filtering only 1 survived.
        return make_leaf_or_single(arm_indices, drift);
    }

    TreeNode::internal(children)
}

/// Make a leaf group, or a single leaf if only one arm.
fn make_leaf_or_single(arm_indices: &[usize], drift_rate: f32) -> TreeNode {
    if arm_indices.len() == 1 {
        TreeNode::leaf(arm_indices[0], drift_rate)
    } else {
        TreeNode::internal(
            arm_indices
                .iter()
                .map(|&i| TreeNode::leaf(i, drift_rate))
                .collect(),
        )
    }
}

/// Find the nearest cluster for a noise point (local_idx).
///
/// Returns the cluster_id of the nearest non-noise point.
fn nearest_cluster_idx(local_idx: usize, points: &[[f32; 2]], clusters: &[Option<usize>]) -> usize {
    let mut best_dist = f32::INFINITY;
    let mut best_cluster = 0usize;
    for (j, &c) in clusters.iter().enumerate() {
        if let Some(cid) = c {
            let d = sq_dist(&points[local_idx], &points[j]);
            if d < best_dist {
                best_dist = d;
                best_cluster = cid;
            }
        }
    }
    best_cluster
}

impl LatentTaskTree {
    /// Build the tree offline from a list of embeddings (Phase 3 constructor).
    ///
    /// Pipeline: PCA → 2D embed → Chart Test → DBSCAN → recurse on each cluster.
    /// All modelless, deterministic given (embeddings, config). BLAKE3-committable
    /// via [`Self::blake3_root`] (delegates to [`Self::from_root`]).
    ///
    /// # Panics
    ///
    /// Panics if `embeddings` is empty, or if embeddings have inconsistent
    /// dimensionality.
    pub fn build(embeddings: &[Vec<f32>], config: LatentTaskTreeConfig) -> Self {
        assert!(
            !embeddings.is_empty(),
            "manifold_bandit::build: embeddings must not be empty"
        );
        let dim = embeddings[0].len();
        assert!(
            dim > 0,
            "manifold_bandit::build: embeddings must have non-zero dimension"
        );
        for (i, e) in embeddings.iter().enumerate() {
            assert_eq!(
                e.len(),
                dim,
                "manifold_bandit::build: embedding {i} has dim {} expected {dim}",
                e.len()
            );
        }

        let arm_indices: Vec<usize> = (0..embeddings.len()).collect();
        let root = build_recursive(&arm_indices, embeddings, &config, 0);
        Self::from_root(root, config)
    }

    /// Read-only access to the root node (for diagnostics / testing).
    pub fn root(&self) -> &TreeNode {
        &self.root
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
