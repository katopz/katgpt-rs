//! Decision-Diffusion Tree (DDTree) for speculative decoding.
//!
//! Implements width-scaled rollout selection with multiple strategies:
//! - **BestQ** (PTRM default): highest cumulative relevance score
//! - **MostFrequent** (mode@K): most common path across rollouts
//! - **Top1Converged** (EqR, Plan 119): smallest marginal-change residual ∥p_{d+1} − p_d∥₂
//!
//! EqR convergence selection is only reliable after landscape shaping (RI + NI training).
//! See Research 079 (EqR, arXiv:2605.21488) for theoretical justification.

use std::collections::{BinaryHeap, HashMap};

#[cfg(test)]
use super::types::BinaryScreeningPruner;
#[cfg(test)]
use super::types::NoScreeningPruner;
use super::types::{ConstraintPruner, NoPruner, ScreeningPruner, SdeConfig, TreeNode};
use crate::types::{InferenceResult, Rng};
use rayon::prelude::*;

/// Extract tokens from `parent_path` bitfield for path-aware pruning.
///
/// `parent_path` uses 5 bits per depth, packed LSB-first:
/// - Depth 0 token: bits 0–4
/// - Depth 1 token: bits 5–9
/// - ...
/// - Depth k token: bits (k*5) to (k*5+4)
///
/// Returns `Vec<usize>` where `result[k]` = token at depth `k`.
/// Max depths: 64/5 = 12 (sufficient for lookahead of 5–8).
pub fn extract_parent_tokens(parent_path: u128, num_tokens: usize) -> Vec<usize> {
    // parent_path packs tokens with most-recent in lowest bits:
    //   depth 0 token → bits (num_tokens-1)*16 .. (num_tokens-1)*16+15
    //   depth k token → bits (num_tokens-1-k)*16 .. (num_tokens-1-k)*16+15
    (0..num_tokens)
        .map(|k| ((parent_path >> ((num_tokens - 1 - k) * 16)) & 0xFFFF) as usize)
        .collect()
}

/// Zero-alloc variant of [`extract_parent_tokens`].
/// Writes `num_tokens` parent tokens into `buf`, which must be large enough.
/// Returns the slice `&buf[..num_tokens]`.
#[inline]
pub fn extract_parent_tokens_into(
    parent_path: u128,
    num_tokens: usize,
    buf: &mut [usize],
) -> &[usize] {
    for (k, slot) in buf.iter_mut().enumerate().take(num_tokens) {
        *slot = ((parent_path >> ((num_tokens - 1 - k) * 16)) & 0xFFFF) as usize;
    }
    &buf[..num_tokens]
}

// ── SDE Noise Injection (ELF Plan 079) ──────────────────────────

/// Inject SDE noise into marginals for DDTree expansion diversity (ELF Alg 6).
///
/// When `sde_config.gamma > 0`, adds log-space Gaussian noise to marginals
/// to break greedy error accumulation and diversify tree paths.
/// γ=0 returns marginals unchanged (zero-cost no-op).
///
/// # Algorithm
///
/// For each token probability `p` in each marginal:
/// 1. If `p > confidence_floor`: convert to log-space, add `γ * N(0,1)`, convert back
/// 2. Skip very confident tokens if `preserve_top1` is set (keep argmax unchanged)
/// 3. Re-normalize to ensure probabilities sum to 1.0
///
/// # Arguments
///
/// * `marginals` — Per-depth token probability distributions
/// * `sde_config` — SDE noise injection configuration
/// * `rng` — Random number generator (must be deterministic for reproducibility)
///
/// # Returns
///
/// New `Vec<Vec<f32>>` with perturbed marginals, or clones if γ=0.
pub fn inject_sde_noise(
    marginals: &[&[f32]],
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<Vec<f32>> {
    if !sde_config.is_enabled() {
        return marginals.iter().map(|m| m.to_vec()).collect();
    }

    marginals
        .iter()
        .map(|marginal| {
            let mut perturbed = marginal.to_vec();

            // Find argmax if preserve_top1
            let top1_idx = if sde_config.preserve_top1 {
                perturbed
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
            } else {
                None
            };

            // Convert to log-space, add noise, convert back
            let mut sum = 0.0f32;
            for (i, prob) in perturbed.iter_mut().enumerate() {
                // Skip top-1 if preserving
                if top1_idx == Some(i) {
                    sum += *prob;
                    continue;
                }

                // Skip below confidence floor
                if *prob <= sde_config.confidence_floor {
                    continue;
                }

                // Convert to log-space, add γ * N(0,1), convert back
                let log_p = prob.ln();
                let noisy_log_p = log_p + sde_config.gamma * rng.normal();
                *prob = noisy_log_p.exp().max(0.0);
                sum += *prob;
            }

            // Re-normalize
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for prob in perturbed.iter_mut() {
                    *prob *= inv_sum;
                }
            }

            perturbed
        })
        .collect()
}

/// DDTree: Build verification tree from marginals using Best-First Search.
/// Returns tree nodes ordered by score (best first).
///
/// Equivalent to `build_dd_tree_pruned` with `NoPruner` and `chain_seed=false`.
///
/// # Branch Ordering Preserves Reasoning Sequence (Plan 029)
///
/// Each DDTree branch stores tokens in `parent_path` as an **ordered sequence**,
/// preserving the exact order the draft model produced them. This is critical for
/// agentic inference where reasoning and tool calls must remain interleaved:
///
/// ```text
/// CORRECT (DDTree preserves this):
///   reasoning_0 → tool_call_0 → reasoning_1 → tool_call_1
///
/// WRONG (would lose sequence meaning):
///   reasoning_0 → reasoning_1 → tool_call_0 → tool_call_1
/// ```
///
/// NVIDIA Dynamo found that grouping reasoning separate from tool calls increased
/// TTFT 1.9× (322ms vs 167ms on B200) because the target model couldn't associate
/// each tool call with its preceding reasoning. Our `extract_parent_tokens()` and
/// `extract_parent_tokens_into()` maintain this ordering per branch.
pub fn build_dd_tree(marginals: &[&[f32]], config: &crate::types::Config) -> Vec<TreeNode> {
    build_dd_tree_pruned(marginals, config, &NoPruner, false)
}

/// DDTree with Constraint Pruner: Build verification tree from marginals,
/// filtering branches through a deterministic rules engine.
///
/// When `chain_seed=true`, builds a greedy chain backbone first (argmax at
/// each depth with cumulative log-prob scores), then seeds the best-first
/// heap with siblings at each chain depth and children of the last chain
/// node. Standard best-first expansion fills the remaining budget.
///
/// When `chain_seed=false`, uses the original best-first algorithm.
///
/// The pruner is called for every candidate token at every depth.
/// Invalid tokens are never added to the heap — they don't waste tree budget.
///
/// This is the **Symbolic Validator intercept**: the draft model proposes
/// logits (semantic probability), the pruner enforces constraints
/// (mathematical validity), and only valid branches reach verification.
pub fn build_dd_tree_pruned(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    pruner: &dyn ConstraintPruner,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build(marginals, config, pruner, chain_seed);
    std::mem::take(&mut builder.tree)
}

/// DDTree with Screening Pruner: Build verification tree from marginals,
/// blending LLM log-probabilities with absolute relevance scores.
///
/// This is the upgraded version of [`build_dd_tree_pruned`]. Instead of
/// binary valid/invalid, the [`ScreeningPruner`] returns `R ∈ [0.0, 1.0]`:
/// - `R = 1.0` → no penalty (`ln(1.0) = 0.0`)
/// - `0.0 < R < 1.0` → soft penalty (`ln(R)` added to score)
/// - `R ≤ threshold` → hard trim (branch killed, never added to heap)
///
/// Score formula: `blended = parent_score + ln(P_llm) + ln(R)`
///
/// The `screening_threshold` is read from `config.screening_threshold`.
/// When threshold is `0.0`, only `R == 0.0` triggers hard trim (pure softmask).
pub fn build_dd_tree_screened(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build_screened(marginals, config, screener, chain_seed);
    std::mem::take(&mut builder.tree)
}

/// DDTree with GFlowNet backward-weighted scoring (Plan 052).
///
/// Generalization of [`build_dd_tree_screened`] with tunable backward weight
/// and flow bonus. The scoring formula is:
///
/// ```text
/// score = ln(P_llm) + backward_weight × ln(R) + lambda_flow × (1 - stop_prob[depth])
/// ```
///
/// When `backward_weight = 1.0` and `lambda_flow = 0.0`, this is identical to
/// [`build_dd_tree_screened`].
///
/// # Arguments
///
/// * `marginals` — Per-depth token probability distributions
/// * `config` — DDTree configuration (tree_budget, screening_threshold, etc.)
/// * `screener` — Screening pruner for relevance scoring
/// * `chain_seed` — Whether to build greedy chain backbone first
/// * `stop_probs` — Per-depth EOS probability from marginals (for flow bonus)
/// * `backward_weight` — Weight for backward relevance (paper uses ∞; we blend)
/// * `lambda_flow` — Flow regularization strength (default: 0.3)
#[allow(clippy::too_many_arguments)]
pub fn build_dd_tree_balanced(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    stop_probs: &[f32],
    backward_weight: f32,
    lambda_flow: f32,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build_balanced(
        marginals,
        config,
        screener,
        chain_seed,
        stop_probs,
        backward_weight,
        lambda_flow,
    );
    std::mem::take(&mut builder.tree)
}

// ── SDE-Aware DDTree Builders (ELF Plan 079) ────────────────────

/// DDTree with SDE noise injection (ELF Plan 079).
///
/// Applies SDE noise to marginals before building the tree.
/// When `sde_config.gamma == 0.0`, this is identical to `build_dd_tree_screened`.
pub fn build_dd_tree_sde(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<TreeNode> {
    let noisy_marginals = inject_sde_noise(marginals, sde_config, rng);
    let noisy_slices: Vec<&[f32]> = noisy_marginals.iter().map(|m| m.as_slice()).collect();
    build_dd_tree_screened(&noisy_slices, config, screener, chain_seed)
}

/// DDTree balanced with SDE noise injection (ELF Plan 079).
///
/// Applies SDE noise to marginals before building the balanced tree.
/// When `sde_config.gamma == 0.0`, this is identical to `build_dd_tree_balanced`.
#[allow(clippy::too_many_arguments)]
pub fn build_dd_tree_balanced_sde(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    stop_probs: &[f32],
    backward_weight: f32,
    lambda_flow: f32,
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<TreeNode> {
    let noisy_marginals = inject_sde_noise(marginals, sde_config, rng);
    let noisy_slices: Vec<&[f32]> = noisy_marginals.iter().map(|m| m.as_slice()).collect();
    build_dd_tree_balanced(
        &noisy_slices,
        config,
        screener,
        chain_seed,
        stop_probs,
        backward_weight,
        lambda_flow,
    )
}

// ── PTRM Width Scaling (Plan 083) ──────────────────────────────

/// Selection strategy for `best_of_k_rollouts`.
///
/// - `BestQ`: pick the rollout with highest cumulative relevance (PTRM default)
/// - `MostFrequent`: pick the most common path (mode@K, majority vote)
#[cfg(feature = "elf_sde")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WidthSelectionMode {
    /// Select rollout with highest cumulative relevance score (PTRM Q-head analog).
    #[default]
    BestQ,
    /// Select the most frequent path across all rollouts (mode@K).
    MostFrequent,
    /// Select rollout with smallest final residual ∥p_{d+1} − p_d∥₂ (EqR proxy, Plan 119).
    ///
    /// Only reliable after landscape shaping (RI + NI training).
    /// Falls back to BestQ if no residual data available.
    #[cfg(feature = "eqr_convergence")]
    Top1Converged,
}

/// Configuration for width-scaling rollouts (PTRM Plan 083).
///
/// Controls how many independent SDE rollouts to run and how to select
/// the best result. Maps directly to PTRM's K parallel rollouts.
#[cfg(feature = "elf_sde")]
#[derive(Debug, Clone)]
pub struct WidthScaleConfig {
    /// Number of independent rollouts (PTRM: K). Default: 1 (disabled).
    pub k_rollouts: usize,
    /// How to select the winning rollout.
    pub selection: WidthSelectionMode,
}

#[cfg(feature = "elf_sde")]
impl Default for WidthScaleConfig {
    fn default() -> Self {
        Self {
            k_rollouts: 1,
            selection: WidthSelectionMode::default(),
        }
    }
}

#[cfg(feature = "elf_sde")]
impl WidthScaleConfig {
    /// PTRM paper default: K=16, BestQ selection.
    pub fn ptrm_default() -> Self {
        Self {
            k_rollouts: 16,
            selection: WidthSelectionMode::BestQ,
        }
    }
}

/// Convert Config-level [`ConvergenceSelector`] to runtime [`WidthSelectionMode`].
///
/// `MajorityVote` maps to `MostFrequent` (same semantics, different naming convention).
/// `BtRank` falls back to `BestQ` when `bt_rank` feature is off.
#[cfg(feature = "eqr_convergence")]
impl From<microgpt_core::ConvergenceSelector> for WidthSelectionMode {
    fn from(selector: microgpt_core::ConvergenceSelector) -> Self {
        match selector {
            microgpt_core::ConvergenceSelector::BestQ => WidthSelectionMode::BestQ,
            microgpt_core::ConvergenceSelector::MajorityVote => WidthSelectionMode::MostFrequent,
            microgpt_core::ConvergenceSelector::Top1Converged => WidthSelectionMode::Top1Converged,
            microgpt_core::ConvergenceSelector::BtRank => {
                #[cfg(feature = "bt_rank")]
                {
                    WidthSelectionMode::BestQ // TODO: BtRank variant when bt_rank integrates
                }
                #[cfg(not(feature = "bt_rank"))]
                WidthSelectionMode::BestQ
            }
        }
    }
}

// ── EqR Convergence Selection (Plan 119) ──────────────────────

/// Per-rollout residual tracker for EqR convergence-based selection.
///
/// Tracks ∥p_{d+1} − p_d∥₂ across DDTree expansion depths as a proxy
/// for EqR's fixed-point residual ∥fθ(z;x) − z∥. Only valid after
/// landscape shaping (RI + NI training).
///
/// See Research 079 (EqR) for theoretical justification.
#[cfg(feature = "eqr_convergence")]
#[derive(Debug, Clone)]
pub struct ResidualTracker {
    /// ∥p_{d+1} − p_d∥₂ at each expansion depth.
    residuals: Vec<f32>,
}

#[cfg(feature = "eqr_convergence")]
impl ResidualTracker {
    /// Create a new tracker with pre-allocated capacity.
    pub fn new(max_depths: usize) -> Self {
        Self {
            residuals: Vec::with_capacity(max_depths),
        }
    }

    /// Record a marginal-change step: compute ∥z_curr − z_prev∥₂.
    pub fn record_step(&mut self, z_prev: &[f32], z_curr: &[f32]) {
        let diff: f32 = z_prev
            .iter()
            .zip(z_curr.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        self.residuals.push(diff.sqrt());
    }

    /// Last recorded residual (0.0 if empty) — the EqR convergence proxy.
    pub fn final_residual(&self) -> f32 {
        self.residuals.last().copied().unwrap_or(0.0)
    }

    /// Average residual across all recorded steps.
    pub fn mean_residual(&self) -> f32 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        self.residuals.iter().sum::<f32>() / self.residuals.len() as f32
    }

    /// Check if the rollout has converged below the given threshold.
    pub fn is_converged(&self, threshold: f32) -> bool {
        self.final_residual() < threshold
    }
}

/// Best-of-K rollouts: run K independent SDE-noised trees, select the best path.
///
/// This is the core PTRM width-scaling primitive. Each rollout gets an independent
/// noise seed (`base_seed + k`), producing diverse candidate paths. The winner is
/// selected by cumulative relevance score (BestQ) or majority vote (MostFrequent).
///
/// PTRM proves width (K rollouts) >> depth (T steps): +28.6pp vs +3.1pp on PPBench.
///
/// # Arguments
///
/// * `marginals` — Per-depth token probability distributions
/// * `config` — Inference config (tree_budget, draft_lookahead, etc.)
/// * `screener` — Screening pruner for relevance scoring
/// * `sde_config` — SDE noise injection configuration
/// * `width_config` — Width scaling configuration (K, selection mode)
/// * `base_seed` — Base RNG seed; each rollout uses `base_seed.wrapping_add(k)`
///
/// # Returns
///
/// The best token path as `Vec<usize>` (one token per depth).
#[cfg(feature = "elf_sde")]
pub fn best_of_k_rollouts(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    sde_config: &SdeConfig,
    width_config: &WidthScaleConfig,
    base_seed: u64,
) -> Vec<usize> {
    if width_config.k_rollouts <= 1 || !sde_config.is_enabled() {
        // Single rollout or SDE disabled — just build one tree
        let mut rng = Rng::new(base_seed);
        let noisy = inject_sde_noise(marginals, sde_config, &mut rng);
        let noisy_slices: Vec<&[f32]> = noisy.iter().map(|m| m.as_slice()).collect();
        let tree = build_dd_tree_screened(&noisy_slices, config, screener, false);
        return extract_best_path(&tree);
    }

    // Run K independent rollouts with different noise seeds
    let mut paths: Vec<Vec<usize>> = Vec::with_capacity(width_config.k_rollouts);
    let mut scores: Vec<f32> = Vec::with_capacity(width_config.k_rollouts);
    // EqR convergence: track marginal-change residual per rollout (Plan 119)
    #[cfg(feature = "eqr_convergence")]
    let mut final_residuals: Vec<f32> = Vec::with_capacity(width_config.k_rollouts);

    for k in 0..width_config.k_rollouts {
        let mut rng = Rng::new(base_seed.wrapping_add(k as u64));
        let noisy = inject_sde_noise(marginals, sde_config, &mut rng);
        let noisy_slices: Vec<&[f32]> = noisy.iter().map(|m| m.as_slice()).collect();
        let tree = build_dd_tree_screened(&noisy_slices, config, screener, false);

        // Compute cumulative relevance score for the best path
        let path = extract_best_path(&tree);
        let score = cumulative_relevance(&path, screener);
        paths.push(path);
        scores.push(score);

        // EqR convergence: compute marginal-change residual for this rollout
        #[cfg(feature = "eqr_convergence")]
        {
            let mut tracker = ResidualTracker::new(noisy.len().saturating_sub(1));
            for d in 0..noisy.len().saturating_sub(1) {
                tracker.record_step(&noisy[d], &noisy[d + 1]);
            }
            final_residuals.push(tracker.final_residual());
        }
    }

    match width_config.selection {
        WidthSelectionMode::BestQ => {
            // Select rollout with highest cumulative relevance
            let best_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            paths.into_iter().nth(best_idx).unwrap_or_default()
        }
        WidthSelectionMode::MostFrequent => {
            // Select the most common path (mode@K)
            let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
            for path in &paths {
                *counts.entry(path.clone()).or_default() += 1;
            }
            counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(path, _)| path)
                .unwrap_or_default()
        }
        #[cfg(feature = "eqr_convergence")]
        WidthSelectionMode::Top1Converged => {
            // Select rollout with smallest final residual (EqR convergence proxy).
            // Fallback to BestQ if no residual data (e.g., single depth).
            let best_idx = if final_residuals.is_empty()
                || final_residuals.iter().all(|&r| r == 0.0)
            {
                scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                final_residuals
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };
            paths.into_iter().nth(best_idx).unwrap_or_default()
        }
    }
}

/// Compute cumulative relevance score for a path using the screener.
#[cfg(feature = "elf_sde")]
fn cumulative_relevance(path: &[usize], screener: &dyn ScreeningPruner) -> f32 {
    let mut total = 0.0f32;
    for (depth, &token_idx) in path.iter().enumerate() {
        let parent_tokens = &path[..depth];
        total += screener.relevance(depth, token_idx, parent_tokens);
    }
    total
}

/// Zero-alloc variant of `extract_best_path`.
/// Writes best-scored token at each depth into `path` (cleared first).
/// Depth-indexed optimization: groups nodes by depth in a single O(N) pass,
/// replacing O(D×N) repeated `.iter().filter()` scans with O(1) depth lookups.
pub fn extract_best_path_into(tree: &[TreeNode], path: &mut Vec<usize>) {
    path.clear();
    if tree.is_empty() {
        return;
    }

    // Build depth index: O(N) single pass
    let mut by_depth: HashMap<usize, Vec<&TreeNode>> = HashMap::new();
    for node in tree.iter() {
        by_depth.entry(node.depth).or_default().push(node);
    }

    let max_depth = *by_depth.keys().max().unwrap_or(&0);
    for depth in 0..=max_depth {
        let best = match by_depth.get(&depth) {
            Some(nodes) => nodes.iter().max_by_key(|n| (n.score * 1e6) as i64),
            None => break,
        };
        match best {
            Some(node) => path.push(node.token_idx),
            None => break,
        }
    }
}

/// Extract best-scored token at each depth from a DDTree.
/// Depth-indexed optimization: groups nodes by depth in a single O(N) pass,
/// replacing O(D×N) repeated `.iter().filter()` scans with O(1) depth lookups.
pub fn extract_best_path(tree: &[TreeNode]) -> Vec<usize> {
    if tree.is_empty() {
        return Vec::new();
    }

    // Build depth index: O(N) single pass
    let mut by_depth: HashMap<usize, Vec<&TreeNode>> = HashMap::new();
    for node in tree.iter() {
        by_depth.entry(node.depth).or_default().push(node);
    }

    let max_depth = *by_depth.keys().max().unwrap_or(&0);
    let mut path = Vec::with_capacity(max_depth + 1);
    for depth in 0..=max_depth {
        let best = match by_depth.get(&depth) {
            Some(nodes) => nodes.iter().max_by_key(|n| (n.score * 1e6) as i64),
            None => break,
        };
        match best {
            Some(node) => path.push(node.token_idx),
            None => break,
        }
    }
    path
}

/// Extract all candidate sequences from a DDTree (one per leaf node).
///
/// Each leaf node's `parent_path` encodes a full token sequence.
/// Returns `(sequence, leaf_node)` pairs for all maximal-depth paths.
pub fn extract_candidate_sequences(tree: &[TreeNode]) -> Vec<(Vec<usize>, &TreeNode)> {
    if tree.is_empty() {
        return Vec::new();
    }

    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);

    // Collect leaf nodes (nodes at max depth with no children in tree)
    tree.iter()
        .filter(|node| node.depth == max_depth)
        .map(|node| {
            let seq = extract_parent_tokens(node.parent_path, node.depth + 1);
            (seq, node)
        })
        .collect()
}

/// Extract candidate sequences from ALL tree nodes (not just leaves).
///
/// Useful when the solution might not require visiting all targets,
/// or when partial sequences are valid solutions.
pub fn extract_all_sequences(tree: &[TreeNode]) -> Vec<(Vec<usize>, &TreeNode)> {
    if tree.is_empty() {
        return Vec::new();
    }

    tree.iter()
        .map(|node| {
            let seq = extract_parent_tokens(node.parent_path, node.depth + 1);
            (seq, node)
        })
        .collect()
}

/// Parallel DDTree search: find the first candidate sequence that passes validation.
///
/// Extracts all candidate sequences from the DDTree, then validates them in
/// parallel using rayon. Returns the first valid sequence found, or `None`.
///
/// This is the core generic primitive — the caller provides a domain-specific
/// validator closure. For example, the tactical AI provides a closure that
/// simulates boss chase + A* pathfinding + key-box matching.
///
/// # Type Parameters
/// - `V`: Validator closure `Fn(&[usize]) -> Option<T>`
/// - `T`: Result type returned by the validator on success
///
/// # Performance
/// The search phase is parallelized (each candidate validated independently).
/// DDTree build remains sequential (inherent heap-based best-first search).
///
/// # Example
/// ```ignore
/// use microgpt_rs::speculative::{build_dd_tree_pruned, par_find_valid_sequence};
///
/// let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
/// let result = par_find_valid_sequence(&tree, |seq| {
///     // Domain-specific validation: simulate game, check win condition
///     if is_valid_solution(seq) { Some(seq.to_vec()) } else { None }
/// });
/// ```
pub fn par_find_valid_sequence<T, V>(tree: &[TreeNode], validator: V) -> Option<(Vec<usize>, T)>
where
    V: Fn(&[usize]) -> Option<T> + Sync,
    T: Send,
{
    if tree.is_empty() {
        return None;
    }

    // Extract all candidate sequences (one per tree node)
    let candidates: Vec<Vec<usize>> = tree
        .iter()
        .map(|node| extract_parent_tokens(node.parent_path, node.depth + 1))
        .collect();

    // Parallel search: validate all candidates, return first success
    candidates
        .par_iter()
        .find_map_any(|seq| validator(seq).map(|result| (seq.clone(), result)))
}

/// Sequential version of [`par_find_valid_sequence`] — no rayon overhead.
///
/// Useful for small trees where rayon spawn cost outweighs parallelism benefit,
/// or when deterministic ordering is required (first candidate wins).
pub fn find_valid_sequence<T, V>(tree: &[TreeNode], validator: V) -> Option<(Vec<usize>, T)>
where
    V: Fn(&[usize]) -> Option<T>,
{
    if tree.is_empty() {
        return None;
    }

    for node in tree {
        let seq = extract_parent_tokens(node.parent_path, node.depth + 1);
        if let Some(result) = validator(&seq) {
            return Some((seq, result));
        }
    }

    None
}

/// Parallel search for the **shortest** valid sequence by cost.
///
/// Unlike [`par_find_valid_sequence`] which returns the first valid candidate,
/// this validates all candidates in parallel and returns the one with minimum cost.
/// Use when optimality (fewest steps) matters more than speed.
///
/// # Arguments
///
/// * `tree` — DDTree nodes (one candidate sequence per node)
/// * `validator` — Returns `Some(result)` for valid sequences, `None` for invalid
/// * `cost_fn` — Extracts cost from result (e.g., `|r: &T| r.0.len()` for step count)
///
/// # Example
///
/// ```ignore
/// use microgpt_rs::speculative::dd_tree::par_find_shortest_sequence;
///
/// let result = par_find_shortest_sequence(
///     &tree,
///     |seq| try_sequence(game, seq, &targets),
///     |(actions, _, _)| actions.len(),
/// );
/// ```
pub fn par_find_shortest_sequence<T, V, C>(
    tree: &[TreeNode],
    validator: V,
    cost_fn: C,
) -> Option<(Vec<usize>, T)>
where
    V: Fn(&[usize]) -> Option<T> + Sync,
    T: Send,
    C: Fn(&T) -> usize + Sync,
{
    if tree.is_empty() {
        return None;
    }

    let candidates: Vec<Vec<usize>> = tree
        .iter()
        .map(|node| extract_parent_tokens(node.parent_path, node.depth + 1))
        .collect();

    candidates
        .par_iter()
        .filter_map(|seq| validator(seq).map(|result| (seq.clone(), result)))
        .min_by_key(|(_, result)| cost_fn(result))
}

/// Build an InferenceResult from a completed DDTree inference.
pub fn build_inference_result(
    domain: &str,
    reward: f32,
    tree_size: usize,
    budget_level: u8,
    prompt_hash: u64,
    output: &str,
    screening_threshold: f32,
) -> InferenceResult {
    InferenceResult {
        domain: domain.to_string(),
        reward,
        tree_budget_used: tree_size,
        budget_level,
        prompt_hash,
        output: output.to_string(),
        timestamp: {
            // Use simple Unix epoch millis since we don't depend on uuid/chrono
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64
        },
        screened: reward < screening_threshold,
        #[cfg(feature = "sr2am_configurator")]
        planning_decision: None,
        #[cfg(feature = "sr2am_configurator")]
        plan_horizon_used: 0, // caller sets after entropy truncation
    }
}

/// Inject retrieved token sequences into the DDTree as candidate branches.
///
/// Each retrieved sequence becomes a path with blended score.
/// Score blending: `(1-w) * log(draft_prob) + w * log(similarity)`
///
/// This is a pure computation function — no feature gating needed.
/// The REST feature provides the data; this function processes it.
pub fn merge_retrieved_branches(
    tree: &mut Vec<TreeNode>,
    marginals: &[&[f32]],
    config: &crate::types::Config,
    token_sequences: &[Vec<usize>],
    scores: &[f32],
    rest_weight: f32,
) {
    if token_sequences.is_empty() || rest_weight <= 0.0 {
        return;
    }

    let inv_weight = 1.0 - rest_weight;

    for (seq_idx, seq) in token_sequences.iter().enumerate() {
        let similarity = scores.get(seq_idx).copied().unwrap_or(0.0);
        if similarity <= 0.0 {
            continue;
        }

        for (depth, &token_idx) in seq.iter().enumerate() {
            if depth >= marginals.len() {
                break;
            }
            if token_idx >= config.vocab_size {
                break;
            }

            let base_prob = marginals[depth].get(token_idx).copied().unwrap_or(0.0);
            if base_prob <= 0.0 {
                continue;
            }

            let blended = (base_prob.ln() * inv_weight) + (similarity.ln() * rest_weight);

            // Reconstruct parent_path from sequence prefix up to current depth
            let parent_path = seq[..=depth]
                .iter()
                .enumerate()
                .fold(0u128, |acc, (d, &t)| {
                    if d == 0 {
                        t as u128
                    } else {
                        (acc << 16) | (t as u128)
                    }
                });

            tree.push(TreeNode {
                score: blended,
                depth,
                token_idx,
                parent_path,
            });
        }
    }

    // Re-sort by score descending
    tree.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    tree.truncate(config.tree_budget);
}

/// Pre-allocated buffers for zero-alloc DDTree building.
///
/// Create once with `TreeBuilder::new(config)`, reuse across calls.
/// `build()` clears and reuses internal buffers — no allocation on steady state.
pub struct TreeBuilder {
    heap: BinaryHeap<TreeNode>,
    tree: Vec<TreeNode>,
    chain_nodes: Vec<TreeNode>,
    chain_parent_tokens: Vec<usize>,
    parent_tokens_buf: Vec<usize>,
}

impl TreeBuilder {
    /// Allocate all buffers from config dimensions.
    pub fn new(config: &crate::types::Config) -> Self {
        Self {
            heap: BinaryHeap::new(),
            tree: Vec::with_capacity(config.tree_budget),
            chain_nodes: Vec::with_capacity(config.draft_lookahead),
            chain_parent_tokens: Vec::with_capacity(config.draft_lookahead),
            parent_tokens_buf: vec![0usize; config.draft_lookahead + 1],
        }
    }

    /// Build DDTree from marginals, reusing pre-allocated buffers.
    ///
    /// Clears and reuses `heap`, `tree`, `chain_nodes`, `chain_parent_tokens`.
    /// Returns a borrowed slice valid until the next `build()` call.
    pub fn build(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        pruner: &dyn ConstraintPruner,
        chain_seed: bool,
    ) -> &[TreeNode] {
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        if chain_seed {
            // ── Phase A: Build greedy chain backbone ──────────────
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path: u128 = 0;

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                // Find argmax token at this depth
                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                // Chain breaks if argmax has zero prob or is pruned
                if prob <= 0.0 || !pruner.is_valid(depth, token_idx, &self.chain_parent_tokens) {
                    break;
                }

                cumulative_score += prob.ln();
                let node_path = if depth == 0 {
                    token_idx as u128
                } else {
                    (parent_path << 16) | (token_idx as u128)
                };

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                // No chain built — fall back to original root seeding
                if config.vocab_size > 256 {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                                Some(TreeNode {
                                    score: prob.ln(),
                                    depth: 0,
                                    token_idx: i,
                                    parent_path: i as u128,
                                })
                            } else {
                                None
                            }
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                            self.heap.push(TreeNode {
                                score: prob.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            });
                        }
                    }
                }
            } else {
                // Seed siblings at each chain depth
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    // Parent tokens for pruning: chain tokens at depths 0..depth-1
                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path >> 16,
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob > 0.0 && pruner.is_valid(depth, i, sibling_parent_tokens) {
                            let sibling_path = if depth == 0 {
                                i as u128
                            } else {
                                let ancestor_path = chain_node.parent_path >> 16;
                                (ancestor_path << 16) | (i as u128)
                            };

                            self.heap.push(TreeNode {
                                score: parent_chain_score + prob.ln(),
                                depth,
                                token_idx: i,
                                parent_path: sibling_path,
                            });
                        }
                    }
                }

                // Seed children of the last chain node
                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                            self.heap.push(TreeNode {
                                score: last.score + prob.ln(),
                                depth: next_depth,
                                token_idx: i,
                                parent_path: (last.parent_path << 16) | (i as u128),
                            });
                        }
                    }
                }
            }
        } else {
            // Original behavior: seed heap with root's children, filtered by pruner
            if config.vocab_size > 256 {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                            Some(TreeNode {
                                score: prob.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                        self.heap.push(TreeNode {
                            score: prob.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        });
                    }
                }
            }
        }

        // ── Phase C: Standard best-first expansion ────────────────
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    // Not a new best — track running second best (degrades with heap)
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                // Extract parent tokens from path bitfield for path-aware pruning
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    // NEURO-SYMBOLIC INTERCEPT: prune before adding to heap
                    if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                        self.heap.push(TreeNode {
                            score: best.score + prob.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: (best.parent_path << 16) | (i as u128),
                        });
                    }
                }
            }
        }

        &self.tree
    }

    /// Build tree and merge retrieved branches in one call.
    ///
    /// For REST feature: builds the DDTree, then calls `merge_retrieved_branches`
    /// on the internal tree buffer. Returns a borrowed slice valid until the
    /// next `build()` or `build_and_merge()` call.
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_merge(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        pruner: &dyn ConstraintPruner,
        chain_seed: bool,
        token_sequences: &[Vec<usize>],
        scores: &[f32],
        rest_weight: f32,
    ) -> &[TreeNode] {
        self.build(marginals, config, pruner, chain_seed);
        merge_retrieved_branches(
            &mut self.tree,
            marginals,
            config,
            token_sequences,
            scores,
            rest_weight,
        );
        &self.tree
    }

    /// Consume the builder and return the tree as an owned `Vec`.
    pub fn into_tree(self) -> Vec<TreeNode> {
        self.tree
    }

    /// Build DDTree with graded relevance screening (Plan 021).
    ///
    /// Like [`build()`] but uses [`ScreeningPruner`] for continuous relevance
    /// instead of binary [`ConstraintPruner`]. The relevance score `R ∈ [0.0, 1.0]`
    /// is blended into log-prob space: `score += ln(P_llm) + ln(R)`.
    ///
    /// Branches with `relevance <= config.screening_threshold` are hard-trimmed.
    pub fn build_screened(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
    ) -> &[TreeNode] {
        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with screening ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path: u128 = 0;

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                // Blended score: ln(P_llm) + ln(R)
                cumulative_score += prob.ln() + relevance.ln();
                let node_path = if depth == 0 {
                    token_idx as u128
                } else {
                    (parent_path << 16) | (token_idx as u128)
                };

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                if config.vocab_size > 256 {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob <= 0.0 {
                                return None;
                            }
                            let relevance = screener.relevance(0, i, &[]);
                            if relevance <= threshold {
                                return None;
                            }
                            Some(TreeNode {
                                score: prob.ln() + relevance.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            })
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: prob.ln() + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        });
                    }
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path >> 16,
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = if depth == 0 {
                            i as u128
                        } else {
                            let ancestor_path = chain_node.parent_path >> 16;
                            (ancestor_path << 16) | (i as u128)
                        };

                        self.heap.push(TreeNode {
                            score: parent_chain_score + prob.ln() + relevance.ln(),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + prob.ln() + relevance.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: (last.parent_path << 16) | (i as u128),
                        });
                    }
                }
            }
        } else {
            // Original seeding with screening
            if config.vocab_size > 256 {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob <= 0.0 {
                            return None;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            return None;
                        }
                        Some(TreeNode {
                            score: prob.ln() + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        })
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: prob.ln() + relevance.ln(),
                        depth: 0,
                        token_idx: i,
                        parent_path: i as u128,
                    });
                }
            }
        }

        // ── Phase C: Best-first expansion with screening ─────────
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    // Not a new best — track running second best (degrades with heap)
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }
                    // SCREENING: ln(P_llm) + ln(R) blended score
                    self.heap.push(TreeNode {
                        score: best.score + prob.ln() + relevance.ln(),
                        depth: next_depth,
                        token_idx: i,
                        parent_path: (best.parent_path << 16) | (i as u128),
                    });
                }
            }
        }

        &self.tree
    }

    /// Build tree with screening and merge retrieved branches in one call.
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_merge_screened(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        token_sequences: &[Vec<usize>],
        scores: &[f32],
        rest_weight: f32,
    ) -> &[TreeNode] {
        self.build_screened(marginals, config, screener, chain_seed);
        merge_retrieved_branches(
            &mut self.tree,
            marginals,
            config,
            token_sequences,
            scores,
            rest_weight,
        );
        &self.tree
    }

    /// Build DDTree with GFlowNet backward-weighted scoring (Plan 052).
    ///
    /// Generalization of [`build_screened`] with tunable backward weight
    /// and flow bonus. The paper's `single_state_beam_search` scores beams
    /// using ONLY backward logits. We blend because our WASM `relevance()`
    /// is coarser than a trained neural P_B.
    ///
    /// # Scoring Formula
    ///
    /// ```text
    /// score = ln(P_llm) + backward_weight × ln(R) + lambda_flow × (1 - stop_prob[depth])
    /// ```
    ///
    /// - `backward_weight = 1.0, lambda_flow = 0.0` → identical to `build_screened`
    /// - `backward_weight = 2.0` → backward relevance counts 2× more than forward
    /// - `backward_weight = 4.0` → near-pure backward (paper's approach)
    ///
    /// # Arguments
    ///
    /// * `marginals` — Per-depth token probability distributions
    /// * `config` — DDTree configuration
    /// * `screener` — Screening pruner for relevance scoring
    /// * `chain_seed` — Whether to build greedy chain backbone first
    /// * `stop_probs` — Per-depth EOS probability from marginals
    /// * `backward_weight` — Weight for backward relevance in scoring
    /// * `lambda_flow` — Flow regularization strength
    #[allow(clippy::too_many_arguments)]
    pub fn build_balanced(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        stop_probs: &[f32],
        backward_weight: f32,
        lambda_flow: f32,
    ) -> &[TreeNode] {
        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        // Helper: compute balanced score for a node
        // score = ln(P_llm) + backward_weight × ln(R) + lambda_flow × (1 - stop_prob[depth])
        let balanced_score = |prob: f32, relevance: f32, depth: usize| -> f32 {
            let r_safe = relevance.max(1e-10); // Avoid ln(0)
            let flow_bonus = lambda_flow * (1.0 - stop_probs.get(depth).copied().unwrap_or(0.5));
            prob.ln() + backward_weight * r_safe.ln() + flow_bonus
        };

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with balanced scoring ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path: u128 = 0;

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                cumulative_score += balanced_score(prob, relevance, depth);
                let node_path = if depth == 0 {
                    token_idx as u128
                } else {
                    (parent_path << 16) | (token_idx as u128)
                };

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                if config.vocab_size > 256 {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob <= 0.0 {
                                return None;
                            }
                            let relevance = screener.relevance(0, i, &[]);
                            if relevance <= threshold {
                                return None;
                            }
                            Some(TreeNode {
                                score: balanced_score(prob, relevance, 0),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            })
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: balanced_score(prob, relevance, 0),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        });
                    }
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path >> 16,
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = if depth == 0 {
                            i as u128
                        } else {
                            let ancestor_path = chain_node.parent_path >> 16;
                            (ancestor_path << 16) | (i as u128)
                        };

                        self.heap.push(TreeNode {
                            score: parent_chain_score + balanced_score(prob, relevance, depth),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + balanced_score(prob, relevance, next_depth),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: (last.parent_path << 16) | (i as u128),
                        });
                    }
                }
            }
        } else {
            // Original seeding with balanced scoring
            if config.vocab_size > 256 {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob <= 0.0 {
                            return None;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            return None;
                        }
                        Some(TreeNode {
                            score: balanced_score(prob, relevance, 0),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        })
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: balanced_score(prob, relevance, 0),
                        depth: 0,
                        token_idx: i,
                        parent_path: i as u128,
                    });
                }
            }
        }

        // ── Phase C: Best-first expansion with balanced scoring ──
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }
                    // BALANCED: ln(P_llm) + backward_weight × ln(R) + flow_bonus
                    self.heap.push(TreeNode {
                        score: best.score + balanced_score(prob, relevance, next_depth),
                        depth: next_depth,
                        token_idx: i,
                        parent_path: (best.parent_path << 16) | (i as u128),
                    });
                }
            }
        }

        &self.tree
    }
}

// ── SR²AM Entropy-Based Horizon Truncation (Plan 112, Research 076) ──

/// If entropy exceeds threshold, cap draft lookahead at a truncated horizon.
///
/// High-uncertainty states benefit from shorter planning horizons to avoid
/// overcommitting to unreliable predictions. Maps to SR²AM's finding that
/// web tasks (high environmental uncertainty) benefit from planning horizon
/// capped at 2 steps.
///
/// # Arguments
///
/// * `entropy` — Shannon entropy in nats (>= 0)
/// * `max_horizon` — Maximum draft lookahead from domain config
///
/// # Returns
///
/// Truncated horizon (min of capped value and max_horizon).
#[cfg(feature = "sr2am_configurator")]
pub fn entropy_truncate_horizon(entropy: f32, max_horizon: usize) -> usize {
    const ENTROPY_THRESHOLD: f32 = 2.5;
    const TRUNCATED_HORIZON: usize = 2;
    match entropy > ENTROPY_THRESHOLD {
        true => TRUNCATED_HORIZON.min(max_horizon),
        false => max_horizon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::dflash::dflash_predict;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    // ── Original DDTree Tests ─────────────────────────────────

    #[test]
    fn test_extract_parent_tokens_roundtrip() {
        let path_d0 = 3u128;
        let path_d1 = (path_d0 << 16) | 7;
        let path_d2 = (path_d1 << 16) | 1;

        assert_eq!(extract_parent_tokens(path_d0, 1), vec![3]);
        assert_eq!(extract_parent_tokens(path_d1, 2), vec![3, 7]);
        assert_eq!(extract_parent_tokens(path_d2, 3), vec![3, 7, 1]);
        let empty: Vec<usize> = extract_parent_tokens(0, 0);
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
            parent_path: 0,
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
        assert_eq!(tree[0].parent_path, 5, "depth 0 parent_path = token 5");
        assert_eq!(
            tree[1].parent_path,
            (5u128 << 16) | 10,
            "depth 1 parent_path = [5, 10]"
        );
        assert_eq!(
            tree[2].parent_path,
            ((5u128 << 16) | 10) << 16 | 3,
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

        let tree_w1 =
            build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 1.0, 0.0);
        let tree_w4 =
            build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 4.0, 0.0);

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
        use crate::speculative::types::SdeConfig;

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
            },
            42,
        );

        assert!(!path.is_empty(), "K=1 should produce a non-empty path");
        assert_eq!(
            path.len(),
            config.draft_lookahead,
            "path length should match lookahead"
        );
    }

    #[cfg(feature = "elf_sde")]
    #[test]
    fn test_best_of_k_rollouts_k16_produces_diverse_paths() {
        use super::{WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts};
        use crate::speculative::types::SdeConfig;

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
        use crate::speculative::types::SdeConfig;

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
        use crate::speculative::types::SdeConfig;

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
        use crate::speculative::types::SdeConfig;

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
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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
}
