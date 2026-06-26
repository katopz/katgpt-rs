//! Hop-level DDTree for SpecHop integration (Plan 131, Phase 6, T25–T27).
//!
//! Extends the DDTree concept from token-level to hop-level speculation.
//! Each tree node represents an (action, predicted observation) pair with
//! speculator confidence as the branch score.
//!
//! ## Design
//!
//! | Aspect              | Token-level DDTree              | Hop-level DDTree                    |
//! |---------------------|---------------------------------|--------------------------------------|
//! | Node payload        | `token_idx: usize`              | `action + observation: String`       |
//! | Score source        | `ln(P_llm)` marginals           | `ln(confidence)` from speculator     |
//! | Parent tracking     | `parent_path: u128` bitfield    | `parent_idx: Option<usize>`          |
//! | Verification        | Exact logit match               | `ObservationVerifier` (fuzzy match)  |
//!
//! ## Verification (T27)
//!
//! Unlike token-level verification (exact logit match), hop-level uses
//! [`ObservationVerifier`](super::verifier::ObservationVerifier) which supports
//! fuzzy matching (Jaccard similarity, refusal detection, numeric consistency).

use std::collections::BinaryHeap;

use super::speculator::HopSpeculator;
use super::types::SpecHopConfig;
use super::verifier::ObservationVerifier;

// ── Hop Verify State ──────────────────────────────────────────

/// Verification state of a hop tree node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HopVerifyState {
    /// Not yet verified (speculative).
    Pending,
    /// Observation verified against target → commit.
    Committed,
    /// Observation mismatched → rollback (branch pruned).
    RolledBack,
}

// ── Hop Tree Node ─────────────────────────────────────────────

/// A node in the hop-level DDTree (T25).
///
/// Each node represents a single speculative hop: (action, predicted observation).
/// The tree is built using best-first search on speculator confidence scores.
///
/// Unlike token-level `TreeNode` which stores `token_idx` and `parent_path`,
/// hop nodes store the full action and observation strings for verification.
#[derive(Clone, Debug)]
pub struct HopTreeNode {
    /// Cumulative log-confidence score (higher = more likely correct path).
    /// Score = Σ ln(confidence) across all hops from root to this node.
    pub score: f64,
    /// Hop depth in the trajectory (0 = first hop).
    pub depth: usize,
    /// The action that triggered this hop (e.g., "search:rust language").
    pub action: String,
    /// The predicted observation from the speculator.
    pub observation: String,
    /// Index of parent node in the tree Vec, or `None` for root-level nodes.
    pub parent_idx: Option<usize>,
    /// Verification status of this node.
    pub verified: HopVerifyState,
}

impl PartialEq for HopTreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
            && self.depth == other.depth
            && self.action == other.action
            && self.observation == other.observation
    }
}

impl Eq for HopTreeNode {}

impl PartialOrd for HopTreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HopTreeNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is max-heap → higher score = higher priority
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.depth.cmp(&self.depth)) // deeper nodes break ties
    }
}

// ── Hop Marginal ──────────────────────────────────────────────

/// A single candidate observation with confidence score.
#[derive(Clone, Debug)]
pub struct HopCandidate {
    /// Predicted observation string.
    pub observation: String,
    /// Speculator confidence that this observation matches the target.
    /// Must be in (0, 1]. Used as `ln(confidence)` for score accumulation.
    pub confidence: f64,
}

impl HopCandidate {
    /// Create a new hop candidate.
    pub fn new(observation: impl Into<String>, confidence: f64) -> Self {
        Self {
            observation: observation.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Log-confidence score. Returns `-inf` if confidence is 0.
    #[inline]
    pub fn log_confidence(&self) -> f64 {
        if self.confidence <= 0.0 {
            return f64::NEG_INFINITY;
        }
        self.confidence.ln()
    }
}

/// Speculator confidence distribution for a single hop depth.
///
/// Each entry is (observation, confidence) where confidence ∈ (0, 1].
/// Analogous to token marginals in the token-level DDTree, but operating
/// on observation strings instead of token indices.
#[derive(Clone, Debug)]
pub struct HopMarginal {
    /// The action for this hop depth.
    pub action: String,
    /// Candidate observations with speculator confidence scores.
    /// Should be sorted by confidence descending for efficient expansion.
    pub candidates: Vec<HopCandidate>,
}

// ── Hop Tree Config ───────────────────────────────────────────

/// Configuration for hop DDTree building.
#[derive(Clone, Debug)]
pub struct HopTreeConfig {
    /// Maximum number of tree nodes (budget). Default: 64.
    pub tree_budget: usize,
    /// Minimum confidence threshold for expanding a branch. Default: 0.01.
    pub confidence_floor: f64,
    /// Whether to use greedy chain seeding (argmax at each depth first).
    pub chain_seed: bool,
}

impl Default for HopTreeConfig {
    fn default() -> Self {
        Self {
            tree_budget: 64,
            confidence_floor: 0.01,
            chain_seed: true,
        }
    }
}

// ── Build Hop DDTree (T26) ────────────────────────────────────

/// Build a hop-level DDTree from speculator confidence distributions.
///
/// Uses best-first search (same algorithm as token-level DDTree) but operates
/// on (action, observation) pairs instead of token indices.
///
/// # Algorithm
///
/// 1. Seed heap with all candidates at depth 0
/// 2. Pop highest-score node from heap
/// 3. Expand: for each candidate at depth+1, push child with accumulated score
/// 4. Repeat until budget exhausted or all paths explored
///
/// # Arguments
///
/// * `marginals` — Per-depth speculator confidence distributions
/// * `config` — Hop tree configuration (budget, thresholds)
///
/// # Returns
///
/// Tree nodes in expansion order (not sorted; use `extract_best_hop_path` for ranking).
pub fn build_hop_dd_tree(marginals: &[HopMarginal], config: &HopTreeConfig) -> Vec<HopTreeNode> {
    if marginals.is_empty() {
        return Vec::new();
    }

    let mut heap: BinaryHeap<HopTreeNode> = BinaryHeap::with_capacity(config.tree_budget);
    let mut tree: Vec<HopTreeNode> = Vec::with_capacity(config.tree_budget);

    // ── Chain seed: build greedy backbone first ────────────────
    if config.chain_seed {
        let mut cumulative_score: f64 = 0.0;
        let mut parent_idx: Option<usize> = None;

        for (depth, marginal) in marginals.iter().enumerate() {
            if tree.len() >= config.tree_budget {
                break;
            }

            // Pick highest-confidence candidate (argmax)
            let best = marginal
                .candidates
                .iter()
                .filter(|c| c.confidence >= config.confidence_floor)
                .max_by(|a, b| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            let Some(candidate) = best else {
                break;
            };

            cumulative_score += candidate.log_confidence();
            let node_idx = tree.len();
            tree.push(HopTreeNode {
                score: cumulative_score,
                depth,
                action: marginal.action.clone(),
                observation: candidate.observation.clone(),
                parent_idx,
                verified: HopVerifyState::Pending,
            });

            // Seed heap with siblings of the chain node (non-best candidates)
            for candidate in &marginal.candidates {
                if candidate.confidence < config.confidence_floor {
                    continue;
                }
                // Skip the chain node itself
                if candidate.observation == tree[node_idx].observation {
                    continue;
                }
                heap.push(HopTreeNode {
                    score: cumulative_score - tree[node_idx].score + candidate.log_confidence(),
                    depth,
                    action: marginal.action.clone(),
                    observation: candidate.observation.clone(),
                    parent_idx,
                    verified: HopVerifyState::Pending,
                });
            }

            parent_idx = Some(node_idx);
        }

        // Seed heap with children of the last chain node
        if let Some(last_idx) = parent_idx {
            let last_depth = tree[last_idx].depth;
            let next_depth = last_depth + 1;
            if next_depth < marginals.len() {
                let next_marginal = &marginals[next_depth];
                for candidate in &next_marginal.candidates {
                    if candidate.confidence < config.confidence_floor {
                        continue;
                    }
                    let child_score = tree[last_idx].score + candidate.log_confidence();
                    heap.push(HopTreeNode {
                        score: child_score,
                        depth: next_depth,
                        action: next_marginal.action.clone(),
                        observation: candidate.observation.clone(),
                        parent_idx: Some(last_idx),
                        verified: HopVerifyState::Pending,
                    });
                }
            }
        }
    } else {
        // ── Standard: seed depth-0 candidates ──────────────────
        if let Some(marginal) = marginals.first() {
            for candidate in &marginal.candidates {
                if candidate.confidence < config.confidence_floor {
                    continue;
                }
                heap.push(HopTreeNode {
                    score: candidate.log_confidence(),
                    depth: 0,
                    action: marginal.action.clone(),
                    observation: candidate.observation.clone(),
                    parent_idx: None,
                    verified: HopVerifyState::Pending,
                });
            }
        }
    }

    // ── Best-first expansion ───────────────────────────────────
    while let Some(node) = heap.pop() {
        if tree.len() >= config.tree_budget {
            break;
        }

        let next_depth = node.depth + 1;
        let node_idx = tree.len();
        // Only `node.score` is read after the push, so capture it and move the
        // node into the tree instead of deep-cloning its two String fields.
        let node_score = node.score;
        tree.push(node);

        // Expand to next depth if available
        if next_depth >= marginals.len() {
            continue;
        }

        let next_marginal = &marginals[next_depth];
        for candidate in &next_marginal.candidates {
            if candidate.confidence < config.confidence_floor {
                continue;
            }
            if tree.len() >= config.tree_budget {
                break;
            }

            let child_score = node_score + candidate.log_confidence();
            heap.push(HopTreeNode {
                score: child_score,
                depth: next_depth,
                action: next_marginal.action.clone(),
                observation: candidate.observation.clone(),
                parent_idx: Some(node_idx),
                verified: HopVerifyState::Pending,
            });
        }
    }

    tree
}

/// Build a hop-level DDTree with `PrunerSchedule`-aware screening (Plan 171).
///
/// Identical to [`build_hop_dd_tree`] but applies the `FrozenBaseGuard` pattern:
/// intermediate hops relax the confidence floor to 0.0 (accept all candidates),
/// while the final hop uses the full `confidence_floor` from config.
///
/// When `schedule` is [`PrunerSchedule::Uniform`], delegates to [`build_hop_dd_tree`]
/// unchanged. When `PrunerSchedule::FrozenBaseGuard`, intermediate hops skip
/// the confidence-based pruning, reserving it for the final hop only.
///
/// # Arguments
///
/// * `marginals` — Per-depth speculator confidence distributions
/// * `config` — Hop tree configuration (budget, thresholds)
/// * `schedule` — Pruner schedule (Uniform or FrozenBaseGuard)
///
/// # Returns
///
/// Tree nodes in expansion order.
#[cfg(feature = "thinking_prune")]
pub fn build_hop_dd_tree_with_schedule(
    marginals: &[HopMarginal],
    config: &HopTreeConfig,
    schedule: crate::pruners::PrunerSchedule,
) -> Vec<HopTreeNode> {
    if marginals.is_empty() {
        return Vec::new();
    }

    // For Uniform schedule, delegate to standard build
    if schedule.is_uniform() {
        return build_hop_dd_tree(marginals, config);
    }

    // FrozenBaseGuard: compute effective floor per depth
    let total_hops = marginals.len();
    let effective_floors: Vec<f64> = (0..total_hops)
        .map(|depth| {
            if schedule.should_screen_full(depth, total_hops) {
                config.confidence_floor
            } else {
                // Intermediate hop: accept all candidates (no floor)
                0.0
            }
        })
        .collect();

    let mut heap: BinaryHeap<HopTreeNode> = BinaryHeap::with_capacity(config.tree_budget);
    let mut tree: Vec<HopTreeNode> = Vec::with_capacity(config.tree_budget);

    // ── Chain seed with relaxed intermediate floors ──────────
    if config.chain_seed {
        let mut cumulative_score: f64 = 0.0;
        let mut parent_idx: Option<usize> = None;

        for (depth, marginal) in marginals.iter().enumerate() {
            if tree.len() >= config.tree_budget {
                break;
            }

            let floor = effective_floors[depth];
            let best = marginal
                .candidates
                .iter()
                .filter(|c| c.confidence >= floor)
                .max_by(|a, b| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            let Some(candidate) = best else {
                break;
            };

            cumulative_score += candidate.log_confidence();
            let node_idx = tree.len();
            tree.push(HopTreeNode {
                score: cumulative_score,
                depth,
                action: marginal.action.clone(),
                observation: candidate.observation.clone(),
                parent_idx,
                verified: HopVerifyState::Pending,
            });

            for candidate in &marginal.candidates {
                if candidate.confidence < floor {
                    continue;
                }
                if candidate.observation == tree[node_idx].observation {
                    continue;
                }
                heap.push(HopTreeNode {
                    score: cumulative_score - tree[node_idx].score + candidate.log_confidence(),
                    depth,
                    action: marginal.action.clone(),
                    observation: candidate.observation.clone(),
                    parent_idx,
                    verified: HopVerifyState::Pending,
                });
            }

            parent_idx = Some(node_idx);
        }

        if let Some(last_idx) = parent_idx {
            let last_depth = tree[last_idx].depth;
            let next_depth = last_depth + 1;
            if next_depth < marginals.len() {
                let next_marginal = &marginals[next_depth];
                let floor = effective_floors[next_depth];
                for candidate in &next_marginal.candidates {
                    if candidate.confidence < floor {
                        continue;
                    }
                    let child_score = tree[last_idx].score + candidate.log_confidence();
                    heap.push(HopTreeNode {
                        score: child_score,
                        depth: next_depth,
                        action: next_marginal.action.clone(),
                        observation: candidate.observation.clone(),
                        parent_idx: Some(last_idx),
                        verified: HopVerifyState::Pending,
                    });
                }
            }
        }
    } else {
        // ── Standard: seed depth-0 candidates ──────────────
        if let Some(marginal) = marginals.first() {
            let floor = effective_floors[0];
            for candidate in &marginal.candidates {
                if candidate.confidence < floor {
                    continue;
                }
                heap.push(HopTreeNode {
                    score: candidate.log_confidence(),
                    depth: 0,
                    action: marginal.action.clone(),
                    observation: candidate.observation.clone(),
                    parent_idx: None,
                    verified: HopVerifyState::Pending,
                });
            }
        }
    }

    // ── Best-first expansion ───────────────────────────────
    while let Some(node) = heap.pop() {
        if tree.len() >= config.tree_budget {
            break;
        }

        let next_depth = node.depth + 1;
        let node_idx = tree.len();
        // Only `node.score` is read after the push, so capture it and move the
        // node into the tree instead of deep-cloning its two String fields.
        let node_score = node.score;
        tree.push(node);

        if next_depth >= marginals.len() {
            continue;
        }

        let next_marginal = &marginals[next_depth];
        let floor = effective_floors[next_depth];
        for candidate in &next_marginal.candidates {
            if candidate.confidence < floor {
                continue;
            }
            if tree.len() >= config.tree_budget {
                break;
            }

            let child_score = node_score + candidate.log_confidence();
            heap.push(HopTreeNode {
                score: child_score,
                depth: next_depth,
                action: next_marginal.action.clone(),
                observation: candidate.observation.clone(),
                parent_idx: Some(node_idx),
                verified: HopVerifyState::Pending,
            });
        }
    }

    tree
}

/// Extract the best-scored path from a hop DDTree.
///
/// Returns the sequence of (action, observation) pairs along the highest-scored
/// path from root to the deepest-achieving node. Uses log-confidence scores which
/// accumulate (more negative = deeper), so we pick the best score at the maximum
/// reachable depth.
pub fn extract_best_hop_path(tree: &[HopTreeNode]) -> Vec<(String, String)> {
    if tree.is_empty() {
        return Vec::new();
    }

    // Single-pass: track max_depth and best (highest-score) node at that depth together.
    // Replaces 3 tree traversals (max → filter → max_by) with one.
    let best = tree.iter().reduce(|acc, n| {
        use std::cmp::Ordering;
        match n.depth.cmp(&acc.depth) {
            Ordering::Greater => n,
            Ordering::Less => acc,
            Ordering::Equal => {
                if n.score > acc.score {
                    n
                } else {
                    acc
                }
            }
        }
    }).expect("non-empty tree has at least one node");

    reconstruct_path(tree, best)
}

/// Extract the deepest path reaching the maximum depth.
///
/// Prefers paths that cover the full trajectory (all hop depths).
/// Among max-depth nodes, selects the one with the highest score.
pub fn extract_deepest_hop_path(tree: &[HopTreeNode]) -> Vec<(String, String)> {
    if tree.is_empty() {
        return Vec::new();
    }

    // Single-pass: track max_depth and best (highest-score) node at that depth together.
    // Replaces 3 tree traversals (max → filter → max_by) with one.
    let best_at_max = tree.iter().reduce(|acc, n| {
        use std::cmp::Ordering;
        match n.depth.cmp(&acc.depth) {
            Ordering::Greater => n,
            Ordering::Less => acc,
            Ordering::Equal => {
                if n.score > acc.score {
                    n
                } else {
                    acc
                }
            }
        }
    }).expect("non-empty tree has at least one node");

    reconstruct_path(tree, best_at_max)
}

/// Reconstruct path from root to the given leaf node.
fn reconstruct_path(tree: &[HopTreeNode], leaf: &HopTreeNode) -> Vec<(String, String)> {
    let mut stack = Vec::with_capacity(leaf.depth + 1);
    let mut current: Option<&HopTreeNode> = Some(leaf);

    // Walk from leaf to root, collecting (action, observation) pairs
    while let Some(node) = current {
        stack.push((node.action.clone(), node.observation.clone()));
        current = node.parent_idx.and_then(|idx| tree.get(idx));
    }

    // Reverse to get root-to-leaf order
    stack.reverse();
    stack
}

// ── Verified Hop Path ─────────────────────────────────────────

/// Result of verifying a hop DDTree against actual observations.
#[derive(Clone, Debug, Default)]
pub struct VerifiedHopPath {
    /// The verified path of (action, observation) pairs.
    pub path: Vec<(String, String)>,
    /// Number of speculative nodes that matched (committed).
    pub commits: usize,
    /// Number of speculative nodes that mismatched (rolled back).
    pub rollbacks: usize,
    /// Number of hops where speculator had no prediction (direct commit).
    pub direct_commits: usize,
    /// Total hops in the trajectory.
    pub total_hops: usize,
}

impl VerifiedHopPath {
    /// Speculator accuracy: `commits / (commits + rollbacks)`.
    ///
    /// Returns 0.0 when no speculations were attempted.
    pub fn accuracy(&self) -> f64 {
        let total = self.commits + self.rollbacks;
        if total == 0 {
            return 0.0;
        }
        self.commits as f64 / total as f64
    }

    /// Total committed hops (commits + rollbacks + direct_commits).
    pub fn total_committed(&self) -> usize {
        self.commits + self.rollbacks + self.direct_commits
    }
}

// ── Verify Hop Tree (T27) ─────────────────────────────────────

/// Verify a hop DDTree against actual observations.
///
/// For each hop depth:
/// 1. Find speculative nodes at that depth with matching action
/// 2. Verify the predicted observation against the actual
/// 3. If match → commit (add to verified path)
/// 4. If mismatch → rollback (prune branch, commit real observation)
///
/// # Arguments
///
/// * `tree` — The hop DDTree (built by [`build_hop_dd_tree`])
/// * `actual` — Actual `(action, observation)` pairs from the target tools
/// * `verifier` — Observation equivalence checker
///
/// # Returns
///
/// Verified path with commit/rollback statistics.
pub fn verify_hop_tree(
    tree: &[HopTreeNode],
    actual: &[(String, String)],
    verifier: &dyn ObservationVerifier,
) -> VerifiedHopPath {
    let mut result = VerifiedHopPath {
        total_hops: actual.len(),
        ..Default::default()
    };

    // Pre-bucket tree nodes by depth in a single pass → O(T) build.
    // `actual` depths run 0..actual.len(); any tree node deeper than that
    // can never match and is ignored. This replaces the previous O(D × T)
    // full-tree rescan per depth (one `.filter` over the whole tree per hop).
    //
    // Buckets preserve expansion order, which is best-first by score, so the
    // "try candidates in score order" semantics below are unchanged.
    let n_depths = actual.len();
    let mut buckets: Vec<Vec<usize>> = (0..n_depths).map(|_| Vec::new()).collect();
    for (idx, node) in tree.iter().enumerate() {
        if node.depth < n_depths {
            buckets[node.depth].push(idx);
        }
    }

    for (depth, (actual_action, actual_obs)) in actual.iter().enumerate() {
        // Scan only this depth's bucket (already filtered by depth) and match
        // by action — no full-tree rescan, no per-depth Vec allocation.
        // Track `had_candidate` during the same pass so the direct-commit vs
        // rollback classification needs no second scan of the bucket.
        let mut matched = false;
        let mut had_candidate = false;
        for &node_idx in &buckets[depth] {
            let node = &tree[node_idx];
            if node.action != *actual_action {
                continue;
            }
            had_candidate = true;
            if verifier.verify(actual_obs, &node.observation) {
                result.commits += 1;
                result
                    .path
                    .push((actual_action.clone(), actual_obs.clone()));
                matched = true;
                break;
            }
        }

        if !matched {
            // "no speculative node for this action" → direct commit;
            // "candidates existed but all mismatched" → rollback.
            if had_candidate {
                result.rollbacks += 1;
            } else {
                result.direct_commits += 1;
            }
            result
                .path
                .push((actual_action.clone(), actual_obs.clone()));
        }
    }

    result
}

// ── Convenience: Build & Verify in One Step ───────────────────

/// Build a hop DDTree from a speculator and trajectory, then verify.
///
/// Convenience wrapper that:
/// 1. Queries the speculator for each action in the trajectory
/// 2. Builds marginals from speculator responses
/// 3. Builds the hop DDTree
/// 4. Verifies against actual observations
///
/// This is the primary integration point for the SpecHop pipeline with DDTree.
pub fn build_and_verify_hop_tree<S: HopSpeculator>(
    speculator: &S,
    config: &SpecHopConfig,
    tree_config: &HopTreeConfig,
    trajectory: &[(String, String)],
    verifier: &dyn ObservationVerifier,
) -> VerifiedHopPath {
    // Build marginals from speculator queries
    let marginals: Vec<HopMarginal> = trajectory
        .iter()
        .map(|(action, _)| {
            let mut candidates = Vec::new();
            if let Ok(obs) = speculator.speculate(action) {
                candidates.push(HopCandidate::new(obs, config.p));
            }
            HopMarginal {
                action: action.clone(),
                candidates,
            }
        })
        .collect();

    let tree = build_hop_dd_tree(&marginals, tree_config);
    verify_hop_tree(&tree, trajectory, verifier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spechop::speculator::CacheSpeculator;
    use crate::spechop::verifier::RuleBasedVerifier;

    // ── Helpers ────────────────────────────────────────────────

    fn default_tree_config() -> HopTreeConfig {
        HopTreeConfig::default()
    }

    fn single_marginal(action: &str, obs: &str, confidence: f64) -> HopMarginal {
        HopMarginal {
            action: action.to_string(),
            candidates: vec![HopCandidate::new(obs, confidence)],
        }
    }

    fn multi_marginal(action: &str, candidates: Vec<(&str, f64)>) -> HopMarginal {
        HopMarginal {
            action: action.to_string(),
            candidates: candidates
                .into_iter()
                .map(|(obs, conf)| HopCandidate::new(obs, conf))
                .collect(),
        }
    }

    // ── HopCandidate ───────────────────────────────────────────

    #[test]
    fn test_hop_candidate_log_confidence() {
        let c = HopCandidate::new("obs", 0.5);
        assert!((c.log_confidence() - 0.5_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_hop_candidate_zero_confidence_is_neg_inf() {
        let zero = HopCandidate::new("obs", 0.0);
        assert!(zero.log_confidence().is_infinite());
        assert!(zero.log_confidence().is_sign_negative());
    }

    #[test]
    fn test_hop_candidate_confidence_clamped() {
        let c = HopCandidate::new("obs", 1.5);
        assert!((c.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hop_candidate_negative_clamped() {
        let c = HopCandidate::new("obs", -0.5);
        assert!(c.confidence <= 0.0);
    }

    // ── build_hop_dd_tree ──────────────────────────────────────

    #[test]
    fn test_build_empty_marginals() {
        let tree = build_hop_dd_tree(&[], &default_tree_config());
        assert!(tree.is_empty());
    }

    #[test]
    fn test_build_single_hop() {
        let marginals = vec![single_marginal("search:rust", "Rust is fast", 0.9)];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].depth, 0);
        assert_eq!(tree[0].action, "search:rust");
        assert_eq!(tree[0].observation, "Rust is fast");
        assert_eq!(tree[0].parent_idx, None);
    }

    #[test]
    fn test_build_two_hops_accumulates_score() {
        let marginals = vec![
            single_marginal("search:rust", "Rust is fast", 0.9),
            single_marginal("search:go", "Go is simple", 0.7),
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        let depth_0: Vec<_> = tree.iter().filter(|n| n.depth == 0).collect();
        let depth_1: Vec<_> = tree.iter().filter(|n| n.depth == 1).collect();

        assert_eq!(depth_0.len(), 1);
        assert_eq!(depth_1.len(), 1);

        // Depth 1 score = depth 0 score + ln(0.7)
        let expected = 0.9_f64.ln() + 0.7_f64.ln();
        assert!((depth_1[0].score - expected).abs() < 1e-10);
        assert_eq!(depth_1[0].parent_idx, Some(0)); // parent is index 0
    }

    #[test]
    fn test_build_multiple_candidates_produces_branches() {
        let marginals = vec![
            multi_marginal("search", vec![("obs_a", 0.9), ("obs_b", 0.5)]),
            multi_marginal("compute", vec![("result_1", 0.8), ("result_2", 0.3)]),
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        // Chain seed picks best at depth 0 + best at depth 1 = 2 chain nodes
        // Sibling of chain depth 0 (obs_b) + children of that sibling = more nodes
        // Should have >= 3 nodes
        assert!(tree.len() >= 3);
    }

    #[test]
    fn test_build_respects_budget() {
        let config = HopTreeConfig {
            tree_budget: 3,
            ..Default::default()
        };
        let marginals = vec![
            multi_marginal("a", vec![("x", 0.9), ("y", 0.8), ("z", 0.7)]),
            multi_marginal("b", vec![("p", 0.9), ("q", 0.8)]),
        ];
        let tree = build_hop_dd_tree(&marginals, &config);
        assert!(tree.len() <= 3);
    }

    #[test]
    fn test_build_skips_low_confidence_candidates() {
        let config = HopTreeConfig {
            confidence_floor: 0.5,
            ..Default::default()
        };
        let marginals = vec![multi_marginal("search", vec![("good", 0.9), ("bad", 0.1)])];
        let tree = build_hop_dd_tree(&marginals, &config);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].observation, "good");
    }

    #[test]
    fn test_build_without_chain_seed() {
        let config = HopTreeConfig {
            chain_seed: false,
            ..Default::default()
        };
        let marginals = vec![
            multi_marginal("a", vec![("x", 0.9), ("y", 0.5)]),
            single_marginal("b", "z", 0.8),
        ];
        let tree = build_hop_dd_tree(&marginals, &config);

        // Without chain seed: 2 roots + children of each = up to 4 nodes
        let roots: Vec<_> = tree.iter().filter(|n| n.depth == 0).collect();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_build_chain_seed_picks_best_at_each_depth() {
        let marginals = vec![
            multi_marginal("a", vec![("best_a", 0.95), ("ok_a", 0.5)]),
            multi_marginal("b", vec![("best_b", 0.9), ("ok_b", 0.4)]),
        ];
        let config = default_tree_config();
        let tree = build_hop_dd_tree(&marginals, &config);

        // First node should be the chain root (best at depth 0)
        assert_eq!(tree[0].observation, "best_a");

        // Find depth 1 node with parent_idx = 0 (chain child)
        let chain_child = tree
            .iter()
            .find(|n| n.depth == 1 && n.parent_idx == Some(0));
        assert!(chain_child.is_some());
        assert_eq!(chain_child.unwrap().observation, "best_b");
    }

    #[test]
    fn test_build_empty_candidates_at_depth() {
        let marginals = vec![
            single_marginal("a", "obs_a", 0.9),
            HopMarginal {
                action: "b".to_string(),
                candidates: vec![], // no candidates → chain stops here
            },
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        // Only depth 0 node should exist
        assert!(tree.iter().all(|n| n.depth == 0));
    }

    // ── extract_best_hop_path ──────────────────────────────────

    #[test]
    fn test_extract_best_path_empty() {
        assert!(extract_best_hop_path(&[]).is_empty());
    }

    #[test]
    fn test_extract_best_path_single_node() {
        let marginals = vec![single_marginal("a", "obs_a", 0.9)];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());
        let path = extract_best_hop_path(&tree);

        assert_eq!(path.len(), 1);
        assert_eq!(path[0], ("a".to_string(), "obs_a".to_string()));
    }

    #[test]
    fn test_extract_best_path_multi_depth() {
        let marginals = vec![
            single_marginal("a", "obs_a", 0.9),
            single_marginal("b", "obs_b", 0.8),
            single_marginal("c", "obs_c", 0.7),
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());
        let path = extract_best_hop_path(&tree);

        assert_eq!(path.len(), 3);
        assert_eq!(path[0].0, "a");
        assert_eq!(path[1].0, "b");
        assert_eq!(path[2].0, "c");
    }

    #[test]
    fn test_extract_best_path_picks_highest_score() {
        let marginals = vec![multi_marginal("a", vec![("low", 0.3), ("high", 0.9)])];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());
        let path = extract_best_hop_path(&tree);

        assert_eq!(path[0].1, "high");
    }

    // ── extract_deepest_hop_path ───────────────────────────────

    #[test]
    fn test_extract_deepest_path_empty() {
        assert!(extract_deepest_hop_path(&[]).is_empty());
    }

    #[test]
    fn test_extract_deepest_path_prefers_full_depth() {
        let marginals = vec![
            single_marginal("a", "obs_a", 0.9),
            single_marginal("b", "obs_b", 0.8),
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());
        let path = extract_deepest_hop_path(&tree);

        assert_eq!(path.len(), 2);
    }

    // ── verify_hop_tree ────────────────────────────────────────

    #[test]
    fn test_verify_empty_tree_all_direct_commits() {
        let actual = vec![
            ("a".to_string(), "obs_a".to_string()),
            ("b".to_string(), "obs_b".to_string()),
        ];
        let verifier = RuleBasedVerifier::default();
        let result = verify_hop_tree(&[], &actual, &verifier);

        assert_eq!(result.direct_commits, 2);
        assert_eq!(result.commits, 0);
        assert_eq!(result.rollbacks, 0);
        assert_eq!(result.path.len(), 2);
    }

    #[test]
    fn test_verify_exact_match_commits() {
        let marginals = vec![
            single_marginal("a", "obs_a", 0.9),
            single_marginal("b", "obs_b", 0.8),
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        let actual = vec![
            ("a".to_string(), "obs_a".to_string()),
            ("b".to_string(), "obs_b".to_string()),
        ];
        let verifier = RuleBasedVerifier::default();
        let result = verify_hop_tree(&tree, &actual, &verifier);

        assert_eq!(result.commits, 2);
        assert_eq!(result.rollbacks, 0);
        assert!((result.accuracy() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verify_mismatch_rollback() {
        let marginals = vec![single_marginal("a", "wrong prediction", 0.9)];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        let actual = vec![("a".to_string(), "actual result".to_string())];
        let verifier = RuleBasedVerifier::default();
        let result = verify_hop_tree(&tree, &actual, &verifier);

        assert_eq!(result.rollbacks, 1);
        assert_eq!(result.commits, 0);
        assert_eq!(result.path[0].1, "actual result");
    }

    #[test]
    fn test_verify_paraphrased_match_commits() {
        // Use a pair with high Jaccard overlap (≥ 0.55)
        let marginals = vec![single_marginal(
            "search",
            "Paris is the capital of France in Europe",
            0.9,
        )];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        let actual = vec![(
            "search".to_string(),
            "Paris is the capital city of France located in Europe".to_string(),
        )];
        let verifier = RuleBasedVerifier::default();
        let result = verify_hop_tree(&tree, &actual, &verifier);

        assert_eq!(result.commits, 1);
    }

    #[test]
    fn test_verify_mixed_commits_and_rollbacks() {
        let marginals = vec![
            single_marginal("a", "match", 0.9),
            single_marginal("b", "mismatch", 0.8),
            single_marginal("c", "also match", 0.7),
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        let actual = vec![
            ("a".to_string(), "match".to_string()),
            ("b".to_string(), "different".to_string()),
            ("c".to_string(), "also match".to_string()),
        ];
        let verifier = RuleBasedVerifier::default();
        let result = verify_hop_tree(&tree, &actual, &verifier);

        assert_eq!(result.commits, 2);
        assert_eq!(result.rollbacks, 1);
        assert!((result.accuracy() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_verify_partial_tree_direct_commits_remaining() {
        // Tree only covers depth 0, actual has depth 0 and 1
        let marginals = vec![single_marginal("a", "obs_a", 0.9)];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        let actual = vec![
            ("a".to_string(), "obs_a".to_string()),
            ("b".to_string(), "obs_b".to_string()),
        ];
        let verifier = RuleBasedVerifier::default();
        let result = verify_hop_tree(&tree, &actual, &verifier);

        assert_eq!(result.commits, 1);
        assert_eq!(result.direct_commits, 1);
    }

    // ── VerifiedHopPath metrics ────────────────────────────────

    #[test]
    fn test_verified_hop_path_accuracy_no_speculation() {
        let result = VerifiedHopPath::default();
        assert!((result.accuracy() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verified_hop_path_total_committed() {
        let result = VerifiedHopPath {
            commits: 3,
            rollbacks: 1,
            direct_commits: 2,
            total_hops: 6,
            ..Default::default()
        };
        assert_eq!(result.total_committed(), 6);
    }

    // ── T28: Integration — perfect speculator same path as sequential ──

    #[test]
    fn test_t28_perfect_speculator_same_path_as_sequential() {
        // When speculator is perfect (p=1.0, all predictions correct),
        // the hop DDTree best path should match sequential execution exactly.
        let marginals = vec![
            single_marginal("search:rust", "Rust is a systems programming language", 1.0),
            single_marginal("search:python", "Python is an interpreted language", 1.0),
            single_marginal("search:go", "Go is a compiled language", 1.0),
            single_marginal("compute:sum", "The sum is 42", 1.0),
        ];
        let tree = build_hop_dd_tree(&marginals, &default_tree_config());

        let actual = vec![
            (
                "search:rust".to_string(),
                "Rust is a systems programming language".to_string(),
            ),
            (
                "search:python".to_string(),
                "Python is an interpreted language".to_string(),
            ),
            (
                "search:go".to_string(),
                "Go is a compiled language".to_string(),
            ),
            ("compute:sum".to_string(), "The sum is 42".to_string()),
        ];

        let verifier = RuleBasedVerifier::default();
        let result = verify_hop_tree(&tree, &actual, &verifier);

        // Perfect speculator → all commits, no rollbacks, no direct commits
        assert_eq!(result.commits, 4);
        assert_eq!(result.rollbacks, 0);
        assert_eq!(result.direct_commits, 0);
        assert_eq!(result.total_hops, 4);

        // Path matches sequential execution exactly
        assert_eq!(result.path.len(), 4);
        for (i, (expected_action, expected_obs)) in actual.iter().enumerate() {
            assert_eq!(
                result.path[i].0, *expected_action,
                "action mismatch at hop {i}"
            );
            assert_eq!(
                result.path[i].1, *expected_obs,
                "observation mismatch at hop {i}"
            );
        }

        // Best path from tree also matches
        let best_path = extract_best_hop_path(&tree);
        assert_eq!(best_path.len(), 4);
        for (i, (action, obs)) in best_path.iter().enumerate() {
            assert_eq!(*action, actual[i].0, "tree path action mismatch at hop {i}");
            assert_eq!(*obs, actual[i].1, "tree path obs mismatch at hop {i}");
        }
    }

    // ── build_and_verify_hop_tree convenience ──────────────────

    #[test]
    fn test_build_and_verify_with_cache_speculator() {
        let speculator = CacheSpeculator::with_entries(vec![
            ("search:rust", "Rust is fast"),
            ("search:go", "Go is simple"),
            // "compute" not in cache → direct commit
        ]);

        let config = SpecHopConfig {
            p: 0.9,
            ..Default::default()
        };
        let tree_config = default_tree_config();
        let verifier = RuleBasedVerifier::default();

        let trajectory = vec![
            ("search:rust".to_string(), "Rust is fast".to_string()),
            ("search:go".to_string(), "Go is simple".to_string()),
            ("compute:sum".to_string(), "42".to_string()),
        ];

        let result =
            build_and_verify_hop_tree(&speculator, &config, &tree_config, &trajectory, &verifier);

        assert_eq!(result.total_hops, 3);
        assert_eq!(result.commits, 2); // search:rust and search:go
        assert_eq!(result.direct_commits, 1); // compute:sum (not in cache)
        assert_eq!(result.path.len(), 3);
    }

    #[test]
    fn test_build_and_verify_empty_cache_all_direct() {
        let speculator = CacheSpeculator::with_entries(vec![("", "")]);
        let config = SpecHopConfig::default();
        let tree_config = default_tree_config();
        let verifier = RuleBasedVerifier::default();

        let trajectory = vec![
            ("a".to_string(), "obs_a".to_string()),
            ("b".to_string(), "obs_b".to_string()),
        ];

        let result =
            build_and_verify_hop_tree(&speculator, &config, &tree_config, &trajectory, &verifier);

        assert_eq!(result.direct_commits, 2);
        assert_eq!(result.commits, 0);
    }

    // ── GOAT Benchmark: FrozenBaseGuard (Plan 171 T6) ────────────

    /// GOAT proof: FrozenBaseGuard schedule produces same or more nodes
    /// than Uniform when confidence_floor is non-zero.
    ///
    /// Rationale: intermediate hops with floor=0.0 accept more candidates,
    /// expanding the tree. Final hop still applies the floor for quality.
    #[cfg(feature = "thinking_prune")]
    #[test]
    fn test_goat_frozen_base_guard_produces_more_nodes() {
        use crate::pruners::PrunerSchedule;

        let config = HopTreeConfig {
            tree_budget: 128,
            confidence_floor: 0.3,
            chain_seed: true,
        };

        let marginals = vec![
            multi_marginal("a", vec![("obs_1", 0.9), ("obs_2", 0.1)]),
            multi_marginal("b", vec![("obs_3", 0.8), ("obs_4", 0.2)]),
            multi_marginal("c", vec![("obs_5", 0.7), ("obs_6", 0.15)]),
        ];

        let uniform_tree =
            build_hop_dd_tree_with_schedule(&marginals, &config, PrunerSchedule::Uniform);
        let frozen_tree =
            build_hop_dd_tree_with_schedule(&marginals, &config, PrunerSchedule::FrozenBaseGuard);

        // FrozenBaseGuard should produce at least as many nodes
        // (intermediate hops accept low-confidence candidates that Uniform rejects)
        assert!(
            frozen_tree.len() >= uniform_tree.len(),
            "FrozenBaseGuard ({}) should produce >= Uniform ({}) nodes",
            frozen_tree.len(),
            uniform_tree.len()
        );
    }

    /// GOAT proof: FrozenBaseGuard with zero floor is equivalent to standard build.
    ///
    /// When all floors are already 0.0, both schedules produce identical trees.
    #[cfg(feature = "thinking_prune")]
    #[test]
    fn test_goat_frozen_base_guard_zero_floor_identical() {
        use crate::pruners::PrunerSchedule;

        let config = HopTreeConfig {
            tree_budget: 64,
            confidence_floor: 0.0, // zero floor
            chain_seed: true,
        };

        let marginals = vec![
            multi_marginal("a", vec![("obs_1", 0.9), ("obs_2", 0.5)]),
            multi_marginal("b", vec![("obs_3", 0.8), ("obs_4", 0.3)]),
        ];

        let standard = build_hop_dd_tree(&marginals, &config);
        let uniform = build_hop_dd_tree_with_schedule(&marginals, &config, PrunerSchedule::Uniform);
        let frozen =
            build_hop_dd_tree_with_schedule(&marginals, &config, PrunerSchedule::FrozenBaseGuard);

        // All three should produce identical trees when floor is 0.0
        assert_eq!(standard.len(), uniform.len());
        assert_eq!(standard.len(), frozen.len());
    }

    /// GOAT proof: FrozenBaseGuard single hop behaves as final step.
    ///
    /// With a single marginal, there's only one hop, so it should apply
    /// the full confidence_floor regardless of schedule.
    #[cfg(feature = "thinking_prune")]
    #[test]
    fn test_goat_frozen_base_single_hop_is_final() {
        use crate::pruners::PrunerSchedule;

        let config = HopTreeConfig {
            tree_budget: 64,
            confidence_floor: 0.5,
            chain_seed: false,
        };

        let marginals = vec![multi_marginal(
            "a",
            vec![("obs_high", 0.9), ("obs_low", 0.1)],
        )];

        let frozen =
            build_hop_dd_tree_with_schedule(&marginals, &config, PrunerSchedule::FrozenBaseGuard);

        // Should only accept the high-confidence candidate (floor=0.5 applied at final step)
        assert_eq!(frozen.len(), 1);
        assert_eq!(frozen[0].observation, "obs_high");
    }
}
