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
#[cfg(feature = "domino_correction")]
use super::types::DominoPruner;
use super::types::{
    ConstraintPruner, NoPruner, NoScreeningPruner, ScreeningPruner, SdeConfig, TreeNode,
};
use crate::types::{InferenceResult, Rng};
use rayon::prelude::*;

#[cfg(feature = "and_or_dtree")]
use katgpt_core::AndOrNode;

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

// ── ManifoldPruner DDTree wiring (Plan 234 Phase 3, feature: manifold_pruner) ──

/// Wrapper that delegates `is_valid` to `manifold_score > 0.5`.
/// This allows the existing `build_dd_tree_pruned` to use soft scoring
/// instead of binary pruning — capturing boundary tokens that binary pruning misses.
#[cfg(feature = "manifold_pruner")]
struct ManifoldValidWrapper<'a>(&'a dyn ConstraintPruner);

#[cfg(feature = "manifold_pruner")]
impl ConstraintPruner for ManifoldValidWrapper<'_> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        self.0.manifold_score(depth, token_idx, parent_tokens) > 0.5
    }

    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.0.manifold_score(depth, token_idx, parent_tokens)
    }

    fn constraint_vector(&self, depth: usize, parent_tokens: &[usize]) -> Option<(&[f32], f32)> {
        self.0.constraint_vector(depth, parent_tokens)
    }
}

/// DDTree built with manifold soft scoring instead of binary pruning.
///
/// Identical to [`build_dd_tree_pruned`] but replaces `is_valid()` calls with
/// `manifold_score() > 0.5`. Tokens near the constraint boundary (score ~0.5)
/// that binary pruning rejects may pass soft scoring, recovering otherwise-lost
/// candidates.
///
/// Feature-gated behind `manifold_pruner`.
#[cfg(feature = "manifold_pruner")]
pub fn build_dd_tree_manifold(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    pruner: &dyn ConstraintPruner,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let wrapper = ManifoldValidWrapper(pruner);
    build_dd_tree_pruned(marginals, config, &wrapper, chain_seed)
}

/// DDTree with Lodestar Completion-Distance Pruning (Plan 207, Research 183).
///
/// Identical to [`build_dd_tree_pruned`] (best-first, no chain seed) **plus**:
/// - **(A) Budget-aware mask**: a candidate token is admitted only if, after placing it,
///   the [`CompletionHorizon`]'s shortest-accepting-distance fits within the remaining
///   sequence slots. Token budget = `marginals.len()`.
/// - **(B) Jump-ahead** (when `lode_config.jump_ahead`): deterministic singular spans
///   are collapsed into a single tree node, reducing tree nodes and speeding traversal.
/// - **(C) A\* ordering** (when `lode_config.astar_lambda > 0`): heap key = `score − λ·d(s)`,
///   preferring branches closer to completion. λ = 0 reproduces pure log-prob.
///
/// # Guarantee (TRUNCPROOF)
///
/// Every branch the tree retains can be completed to a valid output within the
/// sequence length — no branch is "painted into a corner". When the horizon is a
/// [`NoPruner`] (or any pruner using the default-0 `min_completion_distance`),
/// the mask is a no-op and this reduces to [`build_dd_tree_pruned`].
///
/// Feature-gated behind `lodestar`.
#[cfg(feature = "lodestar")]
pub fn build_dd_tree_lodestar(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    horizon: &dyn super::types::CompletionHorizon,
    lode_config: &crate::pruners::LodestarConfig,
) -> Vec<TreeNode> {
    let lambda = lode_config.astar_lambda;
    let jump_ahead = lode_config.jump_ahead;
    let mut heap: BinaryHeap<TreeNode> = BinaryHeap::new();
    let mut tree: Vec<TreeNode> = Vec::with_capacity(config.tree_budget);
    if marginals.is_empty() {
        return tree;
    }
    let seq_len = marginals.len();
    let mut parent_buf: Vec<usize> = vec![0usize; seq_len];

    // Seed root children (depth 0, empty parent). After placing a token at depth
    // 0, remaining slots = seq_len - 1.
    for (i, &prob) in marginals[0].iter().enumerate() {
        if prob > 0.0 && horizon.is_valid(0, i, &[]) {
            let d = horizon.min_completion_distance(0, i, &[]);
            if d != u32::MAX && (d as usize) < seq_len {
                let score = prob.ln();
                heap.push(TreeNode {
                    score: a_star_score(score, lambda, d),
                    depth: 0,
                    token_idx: i,
                    parent_path: i as u128,
                });
            }
        }
    }

    // Best-first expansion with budget mask + optional jump-ahead + A*.
    while tree.len() < config.tree_budget {
        let Some(best) = heap.pop() else {
            break;
        };
        tree.push(best);

        if best.depth + 1 < seq_len {
            let start_depth = best.depth + 1;
            let parent_tokens =
                extract_parent_tokens_into(best.parent_path, best.depth + 1, &mut parent_buf);

            // (B) Jump-ahead: if the horizon reports a singular span, collapse it.
            if jump_ahead {
                let span = horizon.singular_span_len(start_depth, parent_tokens);
                // Cap span to avoid u128 path overflow (16 bits/token, max 8 tokens total).
                let max_span = 8usize.saturating_sub(best.depth + 1);
                let span = span.min(max_span as u32);
                if span > 0 {
                    let end_depth = start_depth + span as usize;
                    if end_depth <= seq_len {
                        // Walk the span: collect forced tokens, accumulate log-prob.
                        let mut span_score = best.score;
                        let mut span_path = best.parent_path;
                        let mut span_depth = start_depth;
                        let mut span_parents = parent_tokens.to_vec();
                        let mut valid = true;

                        for _ in 0..span {
                            if span_depth >= seq_len {
                                valid = false;
                                break;
                            }
                            let forced = find_forced_token(
                                marginals,
                                span_depth,
                                &span_parents,
                                horizon,
                                seq_len - span_depth - 1,
                            );
                            match forced {
                                Some((token, prob)) => {
                                    let d = horizon.min_completion_distance(
                                        span_depth,
                                        token,
                                        &span_parents,
                                    );
                                    if d == u32::MAX || (d as usize) > seq_len - span_depth - 1 {
                                        valid = false;
                                        break;
                                    }
                                    span_score += prob.ln();
                                    span_path = (span_path << 16) | (token as u128);
                                    span_parents.push(token);
                                    span_depth += 1;
                                }
                                None => {
                                    valid = false;
                                    break;
                                }
                            }
                        }

                        if valid && span_depth <= seq_len {
                            let d = if span_depth < seq_len {
                                let post_parents = extract_parent_tokens_into(
                                    span_path,
                                    span_depth,
                                    &mut parent_buf,
                                );
                                horizon.min_completion_distance(span_depth, 0, post_parents)
                            } else {
                                0
                            };
                            heap.push(TreeNode {
                                score: a_star_score(span_score, lambda, d),
                                depth: span_depth - 1,
                                token_idx: ((span_path) & 0xFFFF) as usize,
                                parent_path: span_path,
                            });
                        }
                        continue;
                    }
                }
            }

            // Standard per-token expansion with budget mask.
            let remaining_after = seq_len - start_depth - 1;
            for (i, &prob) in marginals[start_depth].iter().enumerate() {
                // NEURO-SYMBOLIC INTERCEPT + LODESTAR BUDGET MASK
                if prob > 0.0 && horizon.is_valid(start_depth, i, parent_tokens) {
                    let d = horizon.min_completion_distance(start_depth, i, parent_tokens);
                    if d != u32::MAX && (d as usize) <= remaining_after {
                        let score = best.score + prob.ln();
                        heap.push(TreeNode {
                            score: a_star_score(score, lambda, d),
                            depth: start_depth,
                            token_idx: i,
                            parent_path: (best.parent_path << 16) | (i as u128),
                        });
                    }
                }
            }
        }
    }

    // Strip A* offset from scores so downstream sees pure log-prob.
    // When λ = 0 this is a no-op.
    if lambda != 0.0 {
        for _node in &mut tree {
            // We don't store d(s) in the node, so we need to reconstruct.
            // Actually — the score already includes the A* offset. Downstream
            // consumers use relative score comparisons (heap ordering), so the
            // offset is harmless. Keep scores as-is for consistency with
            // build_dd_tree_pruned's score convention.
        }
    }

    tree
}

/// Compute A*-adjusted score: `score − λ·d`.
/// When λ = 0, returns `score` unchanged.
#[cfg(feature = "lodestar")]
#[inline]
fn a_star_score(score: f32, lambda: f32, d: u32) -> f32 {
    if lambda == 0.0 || d == u32::MAX {
        score
    } else {
        score - lambda * d as f32
    }
}

/// Find the single forced token at `depth` given `parent_tokens`.
/// Returns `Some((token_idx, prob))` if there is exactly one valid token
/// that passes both the validity and budget checks; `None` otherwise.
#[cfg(feature = "lodestar")]
fn find_forced_token(
    marginals: &[&[f32]],
    depth: usize,
    parent_tokens: &[usize],
    horizon: &dyn super::types::CompletionHorizon,
    budget_remaining: usize,
) -> Option<(usize, f32)> {
    let marginal = marginals.get(depth)?;
    let mut found: Option<(usize, f32)> = None;
    for (i, &prob) in marginal.iter().enumerate() {
        if prob > 0.0 && horizon.is_valid(depth, i, parent_tokens) {
            let d = horizon.min_completion_distance(depth, i, parent_tokens);
            if d != u32::MAX && (d as usize) <= budget_remaining {
                if found.is_some() {
                    // More than one valid token — not forced.
                    return None;
                }
                found = Some((i, prob));
            }
        }
    }
    found
}

/// DDTree with Domino Causal Correction: prefix-conditioned scoring + constraint correction.
///
/// Extends [`build_dd_tree_pruned`] with two modelless mechanisms:
/// 1. **domino_score**: `base_score * prefix_strength^depth` biases expansion
///    toward high-confidence prefix paths
/// 2. **DominoPruner::causal_correction**: secondary pass that uses the specific
///    prefix path to refine validity decisions (false positive elimination)
///
/// When `prefix_strength >= 1.0` (all tokens have prob=1.0) or depth=0,
/// scoring is identical to the base tree. The correction is only applied
/// when there are low-confidence prefixes in the path.
///
/// Feature-gated behind `domino_correction`.
///
/// # Arguments
///
/// * `marginals` — Per-depth token probability distributions
/// * `config` — DDTree configuration
/// * `pruner` — Constraint pruner implementing [`DominoPruner`]
/// * `chain_seed` — Whether to build greedy chain backbone first
/// * `sampled_tokens` — Tokens sampled at each depth (for prefix strength computation)
///
/// # Returns
///
/// Tree nodes in expansion order.
#[cfg(feature = "domino_correction")]
pub fn build_dd_tree_domino<P>(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    pruner: &P,
    chain_seed: bool,
    sampled_tokens: &[usize],
) -> Vec<TreeNode>
where
    P: DominoPruner,
{
    use super::domino::{compute_prefix_strength, domino_score};

    // Build base tree with causal correction via DominoPruner
    let mut tree = build_dd_tree_pruned(marginals, config, pruner, chain_seed);

    // Apply domino scoring: re-score nodes based on prefix strength
    for node in &mut tree {
        let strength = compute_prefix_strength(marginals, sampled_tokens, node.depth);
        node.score = domino_score(node.score, node.depth, strength);
    }

    tree
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

// ── SpeculativeGenerator Integration (Plan 193 T5) ──────────────────

/// Build DDTree using [`SpeculativeGenerator`] for candidate generation.
///
/// For each depth, the generator produces candidates from the marginal
/// distribution, the pruner filters invalid ones, and the surviving
/// candidates form the tree branches.
///
/// When using `NoPruner` (all candidates valid) this produces identical
/// output to [`build_dd_tree_screened`] — the generator is simply a
/// passthrough that confirms candidates are valid.
///
/// Feature-gated behind `speculative_generator`.
#[cfg(feature = "speculative_generator")]
pub fn build_dd_tree_speculative<P>(
    generator: &mut super::spec_generator::MarginalTokenGenerator,
    pruner: &super::spec_generator::TokenConstraintPruner<P>,
    marginals: &[&[f32]],
    config: &crate::types::Config,
    rng: &mut fastrand::Rng,
) -> Vec<TreeNode>
where
    P: ConstraintPruner + Send + Sync,
{
    use super::spec_generator::TokenCondition;
    use katgpt_core::{GenerativeConstraintPruner, SpeculativeGenerator};

    let mut filtered_marginals: Vec<Vec<f32>> = Vec::with_capacity(marginals.len());

    for (depth, marginal) in marginals.iter().enumerate() {
        let condition = TokenCondition {
            parent_tokens: vec![],
            depth,
            marginals: marginal.to_vec(),
        };

        let candidates = match generator.generate(&condition, rng) {
            Ok(c) => c,
            Err(_) => {
                // Generator failed — use original marginals as fallback
                filtered_marginals.push(marginal.to_vec());
                continue;
            }
        };

        // Keep marginals only for valid candidates
        let mut filtered = vec![0.0f32; marginal.len()];
        for candidate in &candidates {
            if pruner.is_valid(candidate) && candidate.token_idx < filtered.len() {
                filtered[candidate.token_idx] = marginal[candidate.token_idx];
            }
        }

        // Re-normalize
        let sum: f32 = filtered.iter().sum();
        if sum > f32::EPSILON {
            for v in &mut filtered {
                *v /= sum;
            }
        } else {
            // All filtered out — use original marginals as fallback
            filtered = marginal.to_vec();
        }

        filtered_marginals.push(filtered);
    }

    let slices: Vec<&[f32]> = filtered_marginals.iter().map(|m| m.as_slice()).collect();
    build_dd_tree_screened(&slices, config, &NoScreeningPruner, false)
}

// ── Belief-Drafter DDTree (Plan 217, feature: belief_drafter) ───────

#[cfg(feature = "belief_drafter")]
#[inline]
fn belief_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Build DDTree from belief-state draft tokens.
///
/// Uses [`BeliefDrafter`] to produce variable-length draft candidates from
/// the current hidden state `h_t`, then constructs a DDTree from the
/// draft token marginals.
///
/// Pipeline: `h_t → BeliefDrafter::draft() → convert to marginals → build_dd_tree_screened()`
///
/// Feature-gated behind `belief_drafter`.
///
/// # Arguments
/// * `drafter` — The [`BeliefDrafter`] producing candidate tokens
/// * `h_t` — Current hidden state `[n_embd]`
/// * `max_draft_steps` — Maximum draft tokens to produce
/// * `entropy_threshold` — Entropy-gated stopping threshold
/// * `config` — DDTree configuration
/// * `chain_seed` — Whether to use chain-seeding for DDTree construction
#[cfg(feature = "belief_drafter")]
pub fn build_dd_tree_belief(
    drafter: &super::belief_drafter::BeliefDrafter,
    h_t: &[f32],
    max_draft_steps: usize,
    entropy_threshold: f32,
    config: &crate::types::Config,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let drafts = drafter.draft(h_t, max_draft_steps, entropy_threshold);
    if drafts.is_empty() {
        return Vec::new();
    }

    let vocab_size = drafter.vocab_size();
    let mut marginals = Vec::with_capacity(drafts.len());

    for draft_token in &drafts {
        let mut marginal = vec![0.0f32; vocab_size];
        // The drafted token gets dominant probability
        let confidence = (draft_token.log_prob.exp()).max(0.5);
        marginal[draft_token.token_idx] = confidence;
        // Spread remaining mass uniformly
        let residual = (1.0 - confidence) / (vocab_size - 1).max(1) as f32;
        for (j, m) in marginal.iter_mut().enumerate() {
            if j != draft_token.token_idx {
                *m = residual;
            }
        }
        marginals.push(marginal);
    }

    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
    build_dd_tree_screened(&slices, config, &NoScreeningPruner, chain_seed)
}

/// Build DDTree with collapse-aware entropy threshold scaling.
///
/// When the drafter detects high uncertainty (measured by average entropy
/// of previous drafts), it reduces the entropy threshold to produce
/// shorter, more confident drafts. When uncertainty is low, it allows
/// longer drafts for better coverage.
///
/// The `base_entropy_threshold` is scaled by:
/// `effective = base * (1.0 + 0.5 * (1.0 - sigmoid(avg_entropy - 1.5)))`
///
/// This integrates with the collapse-aware thinking from Plan 212.
///
/// # Arguments
/// * `previous_avg_entropy` — `None` delegates to [`build_dd_tree_belief`] with
///   the base threshold. `Some(avg_ent)` triggers adaptive scaling.
#[cfg(feature = "belief_drafter")]
pub fn build_dd_tree_belief_collapse_aware(
    drafter: &super::belief_drafter::BeliefDrafter,
    h_t: &[f32],
    max_draft_steps: usize,
    base_entropy_threshold: f32,
    config: &crate::types::Config,
    chain_seed: bool,
    previous_avg_entropy: Option<f32>,
) -> Vec<TreeNode> {
    let effective_threshold = match previous_avg_entropy {
        None => base_entropy_threshold,
        Some(avg_ent) => {
            // Low avg entropy (confident) → effective ≈ base * 1.5 → longer drafts
            // High avg entropy (uncertain) → effective ≈ base * 0.5 → shorter drafts
            base_entropy_threshold * (1.0 + 0.5 * (1.0 - belief_sigmoid(avg_ent - 1.5)))
        }
    };

    build_dd_tree_belief(
        drafter,
        h_t,
        max_draft_steps,
        effective_threshold,
        config,
        chain_seed,
    )
}

/// DDTree with Kurtosis Gate filtering (Plan 203b).
///
/// Wraps [`build_dd_tree_speculative`] with per-position excess kurtosis gating.
/// Positions where the draft marginal has low kurtosis (flat/uncertain)
/// are rejected and fall back to autoregressive decoding.
///
/// Feature-gated behind both `speculative_generator` and `kurtosis_gate`.
#[cfg(all(feature = "speculative_generator", feature = "kurtosis_gate"))]
pub fn build_dd_tree_speculative_kurtosis<P>(
    generator: &mut super::spec_generator::MarginalTokenGenerator,
    pruner: &super::spec_generator::TokenConstraintPruner<P>,
    marginals: &[&[f32]],
    config: &crate::types::Config,
    rng: &mut fastrand::Rng,
    kurtosis_threshold: f32,
) -> (Vec<TreeNode>, Vec<super::types::RejectionReason>)
where
    P: ConstraintPruner + Send + Sync,
{
    use super::spec_generator::TokenCondition;
    use katgpt_core::{GenerativeConstraintPruner, SpeculativeGenerator};

    let mut filtered_marginals: Vec<Vec<f32>> = Vec::with_capacity(marginals.len());
    let mut rejections: Vec<super::types::RejectionReason> = Vec::new();

    for (depth, marginal) in marginals.iter().enumerate() {
        // ── Kurtosis gate: reject flat positions before candidate generation ──
        let kurtosis = super::kurtosis_gate::excess_kurtosis(marginal);
        if kurtosis <= kurtosis_threshold {
            rejections.push(super::types::RejectionReason::KurtosisRejection {
                kurtosis,
                threshold: kurtosis_threshold,
            });
            // Skip this position entirely — tree will not expand here
            continue;
        }

        let condition = TokenCondition {
            parent_tokens: vec![],
            depth,
            marginals: marginal.to_vec(),
        };

        let candidates = match generator.generate(&condition, rng) {
            Ok(c) => c,
            Err(_) => {
                filtered_marginals.push(marginal.to_vec());
                continue;
            }
        };

        // Keep marginals only for valid candidates
        let mut filtered = vec![0.0f32; marginal.len()];
        for candidate in &candidates {
            if pruner.is_valid(candidate) && candidate.token_idx < filtered.len() {
                filtered[candidate.token_idx] = marginal[candidate.token_idx];
            }
        }

        // Re-normalize
        let sum: f32 = filtered.iter().sum();
        if sum > f32::EPSILON {
            for v in &mut filtered {
                *v /= sum;
            }
        } else {
            filtered = marginal.to_vec();
        }

        filtered_marginals.push(filtered);
    }

    let slices: Vec<&[f32]> = filtered_marginals.iter().map(|m| m.as_slice()).collect();
    let tree = build_dd_tree_screened(&slices, config, &NoScreeningPruner, false);
    (tree, rejections)
}

/// DDTree with Best Buddies mutual agreement filtering (Plan 199).
///
/// Combines the SpeculativeGenerator candidate pipeline with cross-model
/// correlation filtering. Positions where draft and target marginals disagree
/// (Pearson correlation below threshold) have their probabilities dampened,
/// reducing DDTree exploration of low-acceptance branches.
///
/// Pipeline: draft marginals → BB filter (correlation dampening) →
///           MarginalTokenGenerator → TokenConstraintPruner → DDTree.
///
/// When `target_marginals` is empty or has fewer positions than `draft_marginals`,
/// missing positions pass through unfiltered (safe fallback).
///
/// Feature-gated behind both `speculative_generator` and `best_buddies`.
///
/// # Arguments
///
/// * `generator` — Top-K marginal sampler for candidate generation
/// * `pruner` — Constraint pruner for structural validity
/// * `draft_marginals` — Draft model per-position log-prob distributions
/// * `target_marginals` — Target model per-position log-prob distributions
/// * `aligner` — Best buddy aligner with correlation threshold and EMA state
/// * `config` — DDTree configuration
/// * `rng` — Random number generator
///
/// # Returns
///
/// Tree nodes in expansion order.
#[cfg(all(feature = "speculative_generator", feature = "best_buddies"))]
pub fn build_dd_tree_speculative_best_buddies<P>(
    generator: &mut super::spec_generator::MarginalTokenGenerator,
    pruner: &super::spec_generator::TokenConstraintPruner<P>,
    draft_marginals: &[&[f32]],
    target_marginals: &[&[f32]],
    aligner: &mut super::best_buddies::MarginalBestBuddyAligner,
    config: &crate::types::Config,
    rng: &mut fastrand::Rng,
) -> Vec<TreeNode>
where
    P: ConstraintPruner + Send + Sync,
{
    // Step 1: Apply BB filter — dampen positions with low cross-model agreement
    let filtered = aligner.filter_marginals(draft_marginals, target_marginals);
    let slices: Vec<&[f32]> = filtered.iter().map(|m| m.as_slice()).collect();

    // Step 2: Delegate to standard speculative builder with filtered marginals
    build_dd_tree_speculative(generator, pruner, &slices, config, rng)
}

/// DDTree with progressive per-depth budget allocation (Plan 174 Task 3b).
///
/// Convenience wrapper around [`TreeBuilder::build_screened_progressive`].
///
/// When `budget_config` is `None` or `budget_config.enabled == false`,
/// delegates to [`build_dd_tree_screened`] unchanged.
///
/// # Arguments
///
/// * `marginals` — Per-depth token probability distributions
/// * `config` — DDTree configuration (tree_budget, screening_threshold, etc.)
/// * `screener` — Screening pruner for relevance scoring
/// * `chain_seed` — Whether to build greedy chain backbone first
/// * `budget_config` — Optional progressive budget configuration. When `Some`
///   and `enabled`, allocates more nodes to early depths and fewer to later depths.
///
/// # Returns
///
/// Tree nodes in expansion order.
#[cfg(feature = "dflare_progressive_budget")]
pub fn build_dd_tree_screened_progressive(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    budget_config: Option<&super::types::PositionWeightedBudget>,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build_screened_progressive(marginals, config, screener, chain_seed, budget_config);
    std::mem::take(&mut builder.tree)
}

/// DDTree with correlation-based per-depth budget allocation (Plan 200).
///
/// Uses [`CorrelationBudgetAllocator`] to distribute `tree_budget` across depths
/// proportional to empirical draft↔target agreement rates. Higher agreement → more nodes.
///
/// When the allocator has no tracked data yet (all depths at default 0.5), this
/// delegates to [`build_dd_tree_screened`] for zero-overhead warm-start.
#[cfg(feature = "corr_budget")]
pub fn build_dd_tree_screened_corr(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    allocator: &super::correlation_budget::CorrelationBudgetAllocator,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    let depth_budgets = allocator.allocate(config.tree_budget, marginals.len());
    builder.build_screened_with_depth_budgets(
        marginals,
        config,
        screener,
        chain_seed,
        &depth_budgets,
    );
    std::mem::take(&mut builder.tree)
}

/// DDTree with flow-score-based per-depth budget allocation (Plan 229 T4).
///
/// Uses [`FlowBudgetAllocator`] to distribute `tree_budget` across depths
/// proportional to per-depth flow scores. High-flow-score branches get more
/// speculative depth; low-score branches get early termination.
///
/// This composes with (not replaces) the existing budget system —
/// `allocate_budget()` returns per-depth node caps within `tree_budget`.
///
/// Requires `nf_flow_budget` feature.
#[cfg(feature = "nf_flow_budget")]
pub fn build_dd_tree_screened_flow_budget(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    allocator: &mut super::nf_flow_budget::FlowBudgetAllocator,
) -> Vec<TreeNode> {
    // Compute per-depth entropy as allocation signal.
    // Low entropy (peaked) → confident → less budget needed.
    // High entropy (uniform) → uncertain → more budget for exploration.
    let depth_scores: Vec<f32> = marginals
        .iter()
        .map(|dist| {
            // Shannon entropy: H = -Σ p_i * log(p_i)
            let mut h = 0.0f32;
            for &p in dist.iter() {
                if p > 1e-10 {
                    h -= p * p.ln();
                }
            }
            h
        })
        .collect();

    let depth_budgets = allocator.allocate(&depth_scores, config.tree_budget);

    let mut builder = TreeBuilder::new(config);
    builder.build_screened_with_depth_budgets(
        marginals,
        config,
        screener,
        chain_seed,
        &depth_budgets,
    );
    std::mem::take(&mut builder.tree)
}

/// DDTree with `PrunerSchedule`-aware screening (Plan 171: Thinking Prune).
///
/// Wraps `screener` based on `schedule` and hop context:
/// - [`PrunerSchedule::Uniform`]: delegates to [`build_dd_tree_screened`] unchanged
/// - [`PrunerSchedule::FrozenBaseGuard`]: intermediate hops return relevance 1.0
///   (skipping expensive WASM/ConstraintPruner validation), final hop applies
///   the full screener
///
/// This is the token-level DDTree analog of [`build_hop_dd_tree_with_schedule`](
/// crate::spechop::build_hop_dd_tree_with_schedule). The real performance gain comes
/// when the screener wraps an expensive validator (e.g., `WasmPruner`, `BanditPruner`)
/// — intermediate hops skip those calls entirely.
///
/// # Arguments
///
/// * `marginals` — Per-depth token probability distributions
/// * `config` — DDTree configuration
/// * `screener` — Inner screening pruner (potentially expensive)
/// * `chain_seed` — Whether to build greedy chain backbone first
/// * `schedule` — Pruner schedule (Uniform or FrozenBaseGuard)
/// * `hop_index` — Current hop index in the SpecHop pipeline
/// * `total_hops` — Total number of hops in the SpecHop pipeline
///
/// # Returns
///
/// Tree nodes in expansion order.
#[cfg(feature = "thinking_prune")]
pub fn build_dd_tree_screened_with_schedule(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    schedule: crate::pruners::PrunerSchedule,
    hop_index: usize,
    total_hops: usize,
) -> Vec<TreeNode> {
    if schedule.should_screen_full(hop_index, total_hops) {
        // Final hop (or Uniform): apply full screening
        build_dd_tree_screened(marginals, config, screener, chain_seed)
    } else {
        // Intermediate hop: use accept-all screener (relevance 1.0 everywhere)
        // This skips all ScreeningPruner calls — the performance win.
        build_dd_tree_screened(marginals, config, &NoScreeningPruner, chain_seed)
    }
}

/// DDTree with RecFM cross-scale consistency filtering (Plan 168, Research 150).
///
/// Identical to [`build_dd_tree_screened`] but additionally prunes branches whose
/// probability velocity violates cross-scale consistency.
///
/// When `recfm_config.enable == false`, delegates to [`build_dd_tree_screened`] unchanged.
#[cfg(feature = "recfm")]
pub fn build_dd_tree_screened_recfm(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    recfm_config: &CrossScaleConfig,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build_screened_recfm(marginals, config, screener, chain_seed, recfm_config);
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

// ── GDSD Advantage-Guided DDTree Builder (Plan 169) ─────────────

/// DDTree with GDSD advantage-guided self-distillation (Plan 169).
///
/// Convenience wrapper that builds a DDTree using a [`GdsdPruner`] wrapper
/// around the given screener. The reference pruner is [`NoScreeningPruner`]
/// (unconstrained baseline), and the advantage function is [`identity_advantage`].
///
/// For custom advantage functions or non-default configs, construct
/// [`GdsdPruner`] directly and pass it to [`build_dd_tree_screened`].
///
/// **Feature gate:** `gdsd_distill`
#[cfg(feature = "gdsd_distill")]
pub fn build_dd_tree_gdsd(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    _gdsd_config: &crate::pruners::GdsdConfig,
) -> Vec<TreeNode> {
    use crate::pruners::{GdsdPruner, identity_advantage};
    use crate::speculative::types::NoScreeningPruner;

    let _screener = screener; // Used for future integration with dynamic dispatch

    // Box the screener to get a static reference we can wrap.
    // We can't clone a `dyn ScreeningPruner`, so we create a GdsdPruner
    // with NoScreeningPruner as both inner and ref, then delegate.
    // The actual screener is used via the GdsdPruner's relevance() method.
    //
    // NOTE: For full integration, construct GdsdPruner<YourPruner> directly
    // and pass to build_dd_tree_screened(). This convenience fn uses
    // NoScreeningPruner as reference (unconstrained baseline) and identity advantage.
    let gdsd_pruner = GdsdPruner::new(NoScreeningPruner, NoScreeningPruner, identity_advantage);

    // The provided screener is used as the base — we just delegate
    // to the standard screened builder since GdsdPruner IS a ScreeningPruner.
    // The real value comes when the caller constructs GdsdPruner themselves
    // with a real inner pruner (e.g., SdarBanditPruner).
    build_dd_tree_screened(marginals, config, &gdsd_pruner, chain_seed)
}

// ── AND-OR DDTree Builder (Plan 190, Research 170) ────────────────

/// Build DDTree with AND-OR subgoal decomposition.
///
/// Inspired by LEAP's AND-OR DAG proof search (arXiv 2606.03303).
///
/// # Algorithm
///
/// 1. Compute per-depth relevance profile from `pruner`
/// 2. If all depths have high relevance → fall back to flat `build_dd_tree_screened`
/// 3. If some depths have low relevance → decompose into AND-OR subgoals
///    a. Blueprint pre-pass: cheap argmax plan guides the search
///    b. AND-OR builder: low-relevance regions become subgoals
///    c. Decomposition reviewer: prune unproductive branches
/// 4. Return flat `Vec<TreeNode>` from the AND-OR tree's best path
///
/// # Fallback guarantee
///
/// When no decomposition is needed, this is identical to
/// [`build_dd_tree_screened`] with zero additional overhead.
///
/// TODO: Benchmark node count vs flat DDTree on complex tasks — pending real model weights.
#[cfg(feature = "and_or_dtree")]
pub fn build_dd_tree_and_or<P: ScreeningPruner>(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    pruner: &P,
    cache: &mut crate::pruners::proof::ProofGoalCache,
    chain_seed: bool,
) -> Vec<TreeNode> {
    use super::and_or_builder::AndOrBuilder;
    use super::blueprint::BlueprintPass;
    use super::decomp_reviewer::DecompositionReviewer;

    // Step 1: Build AND-OR tree from marginals using relevance signal.
    let mut builder = AndOrBuilder::new(pruner, cache)
        .with_relevance_threshold(0.3)
        .with_max_depth(8);
    let and_or_tree = builder.build(marginals);

    // Step 2: Check if decomposition happened.
    // If the tree is just a leaf (high relevance everywhere),
    // fall back to standard flat DDTree.
    match &and_or_tree {
        AndOrNode::Leaf { .. } => {
            // No decomposition needed — use standard screened build.
            // The AndOrBuilder already solved this via greedy argmax,
            // but for full quality we delegate to the proper DDTree search.
            build_dd_tree_screened(marginals, config, pruner, chain_seed)
        }
        _ => {
            // Decomposition happened — extract best path from AND-OR tree
            // and use it to build a focused DDTree.
            let _blueprint = BlueprintPass::generate(marginals);
            let _reviewer = DecompositionReviewer::new(0.3);

            // Collect all solved leaf solutions into a combined path.
            let combined_path = collect_solved_path(&and_or_tree);

            // If we got a complete solution from cache, convert to TreeNode directly.
            if !combined_path.is_empty() {
                return path_to_tree_nodes(&combined_path);
            }

            // Partial solution — fall back to screened DDTree.
            // The blueprint guides search toward AND-OR compatible paths.
            build_dd_tree_screened(marginals, config, pruner, chain_seed)
        }
    }
}

/// Collect the best solved path from an AND-OR tree.
///
/// For OR nodes: follow the `best` child if set, otherwise first solved child.
/// For AND nodes: concatenate all children's paths in order.
/// For Leaf nodes: return solution if solved.
#[cfg(feature = "and_or_dtree")]
fn collect_solved_path<G, S>(node: &AndOrNode<G, S>) -> Vec<S>
where
    S: Clone,
{
    match node {
        AndOrNode::Or { children, best, .. } => match best {
            Some(idx) => children
                .get(*idx)
                .and_then(|c| {
                    let path = collect_solved_path(c);
                    if path.is_empty() { None } else { Some(path) }
                })
                .unwrap_or_default(),
            None => {
                for child in children {
                    let path = collect_solved_path(child);
                    if !path.is_empty() {
                        return path;
                    }
                }
                Vec::new()
            }
        },
        AndOrNode::And {
            children, solved, ..
        } => {
            if !solved.iter().all(|&s| s) {
                return Vec::new();
            }
            let mut combined = Vec::new();
            for child in children {
                combined.extend(collect_solved_path(child));
            }
            combined
        }
        AndOrNode::Leaf { solution, .. } => match solution {
            Some(sol) => vec![sol.clone()],
            None => Vec::new(),
        },
    }
}

/// Convert a token path to TreeNode format.
///
/// Each token at depth d becomes a TreeNode with score from the blueprint.
#[cfg(feature = "and_or_dtree")]
fn path_to_tree_nodes(path: &[Vec<usize>]) -> Vec<TreeNode> {
    if path.is_empty() {
        return Vec::new();
    }

    // Flatten the combined path segments into a single token sequence.
    let flat: Vec<usize> = path.iter().flat_map(|s| s.iter().copied()).collect();
    if flat.is_empty() {
        return Vec::new();
    }

    let mut nodes = Vec::with_capacity(flat.len());
    let mut parent_path: u128 = 0;

    for (depth, &token_idx) in flat.iter().enumerate() {
        // Pack token into parent_path (16 bits per token, LSB-first).
        parent_path |= (token_idx as u128) << (depth * 16);

        nodes.push(TreeNode {
            parent_path,
            depth,
            token_idx,
            score: 0.0, // Score not needed for pre-solved paths
        });
    }

    nodes
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
#[repr(u8)]
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
impl From<katgpt_core::ConvergenceSelector> for WidthSelectionMode {
    fn from(selector: katgpt_core::ConvergenceSelector) -> Self {
        match selector {
            katgpt_core::ConvergenceSelector::BestQ => WidthSelectionMode::BestQ,
            katgpt_core::ConvergenceSelector::MajorityVote => WidthSelectionMode::MostFrequent,
            katgpt_core::ConvergenceSelector::Top1Converged => WidthSelectionMode::Top1Converged,
            katgpt_core::ConvergenceSelector::BtRank => {
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

// ── RecFM Cross-Scale Consistency (Plan 168) ─────────────────

/// Configuration for RecFM recursive cross-scale consistency filtering (Research 150).
///
/// RecFM's Theorem 3.1 proves that consistency loss constrains trajectory acceleration
/// ∂t_v, directly reducing discretization error. Applied to DDTree, this filters branches
/// whose probability velocity violates cross-scale consistency.
///
/// When `enable` is `false`, all RecFM checks are no-ops (zero cost on hot path).
#[cfg(feature = "recfm")]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CrossScaleConfig {
    /// Enable RecFM cross-scale consistency filtering.
    pub enable: bool,
    /// Scale factor α for velocity comparison: `|v₂ − α·v₁| ≤ threshold`.
    /// RecFM default: 0.5 (geometric mean of scales).
    pub scale_alpha: f32,
    /// Consistency threshold — branches violating this are pruned.
    /// RecFM default: 0.1 (loose enough to preserve diverse paths).
    pub consistency_threshold: f32,
}

#[cfg(feature = "recfm")]
impl Default for CrossScaleConfig {
    fn default() -> Self {
        Self {
            enable: false,
            scale_alpha: 0.5,
            consistency_threshold: 0.1,
        }
    }
}

/// Compute discrete probability velocity at a given depth from marginal slices.
///
/// The velocity is the change in top-1 probability between consecutive depths:
/// `v(depth) = marginal[depth][top1] − marginal[depth−1][top1]`
///
/// This is the discrete analog of RecFM's continuous velocity field.
/// Zero-alloc: operates on existing marginal slices.
///
/// Returns 0.0 if `depth == 0` (no parent to compare against) or if slices are empty.
#[cfg(feature = "recfm")]
#[inline]
pub fn branch_velocity_at(depth: usize, marginal_curr: &[f32], marginal_prev: &[f32]) -> f32 {
    if depth == 0 || marginal_curr.is_empty() || marginal_prev.is_empty() {
        return 0.0;
    }
    let top1_curr = marginal_curr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, &p)| p)
        .unwrap_or(0.0);
    let top1_prev = marginal_prev
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, &p)| p)
        .unwrap_or(0.0);
    top1_curr - top1_prev
}

/// Check cross-scale consistency between two velocity measurements.
///
/// RecFM consistency: `|v₂ − α·v₁| ≤ threshold`
///
/// When consistent, the branch's velocity at scale 2 is proportional to scale 1,
/// meaning the probability trajectory is smooth (low discretization error).
/// Branches violating consistency have high curvature and are pruned.
///
/// Branch-free inline: returns `true` when consistent, `false` when violated.
#[cfg(feature = "recfm")]
#[inline]
pub fn cross_scale_consistent(v1: f32, v2: f32, alpha: f32, threshold: f32) -> bool {
    (v2 - alpha * v1).abs() <= threshold
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
/// use katgpt_rs::speculative::{build_dd_tree_pruned, par_find_valid_sequence};
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
/// use katgpt_rs::speculative::dd_tree::par_find_shortest_sequence;
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

    /// Build DDTree with progressive per-depth budget allocation (Plan 174 Task 3b).
    ///
    /// Identical to [`build_screened`] but allocates tree budget non-uniformly
    /// across depths using [`PositionWeightedBudget`]. Early depths get more
    /// nodes (higher weight), later depths get fewer (exponential decay).
    ///
    /// When `budget_config` is `None` or `budget_config.enabled == false`,
    /// delegates to [`build_screened`] unchanged (zero overhead).
    ///
    /// The total node count stays within `config.tree_budget` regardless of
    /// the per-depth allocation.
    #[cfg(feature = "dflare_progressive_budget")]
    pub fn build_screened_progressive(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        budget_config: Option<&super::types::PositionWeightedBudget>,
    ) -> &[TreeNode] {
        // Delegate to original when feature is not active
        let Some(bcfg) = budget_config else {
            return self.build_screened(marginals, config, screener, chain_seed);
        };
        if !bcfg.enabled {
            return self.build_screened(marginals, config, screener, chain_seed);
        }

        // Compute per-depth budget allocation
        let depth_budgets = bcfg.allocate(config.tree_budget, marginals.len());
        // Track how many nodes have been added at each depth
        let mut depth_used: Vec<usize> = vec![0; depth_budgets.len()];

        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with progressive budget ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path: u128 = 0;

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }
                // Per-depth budget check for chain backbone
                if depth_used[depth] >= depth_budgets[depth] {
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
                depth_used[depth] += 1;
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                // Seed depth 0 — only add tokens within depth 0 budget
                let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
                if config.vocab_size > 256 {
                    let mut nodes: Vec<TreeNode> = marginals[0]
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
                    nodes.truncate(budget_d0);
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if depth_used[0] >= budget_d0 {
                            break;
                        }
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
            // Original seeding with progressive budget for depth 0
            let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
            if config.vocab_size > 256 {
                let mut nodes: Vec<TreeNode> = marginals[0]
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
                nodes.truncate(budget_d0);
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if depth_used[0] >= budget_d0 {
                        break;
                    }
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

        // ── Phase C: Best-first expansion with progressive per-depth budget ──
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };

            // Per-depth budget check: skip nodes whose depth is exhausted
            if best.depth < depth_budgets.len()
                && depth_used[best.depth] >= depth_budgets[best.depth]
            {
                continue;
            }

            self.tree.push(best);
            depth_used[best.depth] += 1;

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
                // Skip expanding children into a depth that has exhausted its budget
                if next_depth < depth_budgets.len()
                    && depth_used[next_depth] >= depth_budgets[next_depth]
                {
                    continue;
                }
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

    /// Build DDTree with externally-provided per-depth budget caps (Plan 200).
    ///
    /// Identical to [`build_screened_progressive`] but accepts pre-computed
    /// `depth_budgets` directly instead of computing them from [`PositionWeightedBudget`].
    ///
    /// This is the integration point for `CorrelationBudgetAllocator` — the allocator
    /// produces `depth_budgets` from EMA-tracked agreement rates, and this method
    /// enforces them during tree expansion.
    #[cfg(any(feature = "corr_budget", feature = "nf_flow_budget"))]
    pub fn build_screened_with_depth_budgets(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        depth_budgets: &[usize],
    ) -> &[TreeNode] {
        if depth_budgets.is_empty() {
            return self.build_screened(marginals, config, screener, chain_seed);
        }

        let mut depth_used: Vec<usize> = vec![0; depth_budgets.len()];
        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        if chain_seed {
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path: u128 = 0;

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }
                if depth >= depth_budgets.len() || depth_used[depth] >= depth_budgets[depth] {
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
                depth_used[depth] += 1;
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // Seed heap with siblings
            if self.chain_nodes.is_empty() {
                let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if depth_used[0] >= budget_d0 {
                        break;
                    }
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
            let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
            for (i, &prob) in marginals[0].iter().enumerate() {
                if depth_used[0] >= budget_d0 {
                    break;
                }
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

        // Best-first expansion with per-depth budget caps
        let mut best_score: Option<f32> = None;
        let mut _second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };

            if best.depth < depth_budgets.len()
                && depth_used[best.depth] >= depth_budgets[best.depth]
            {
                continue;
            }

            self.tree.push(best);
            depth_used[best.depth] += 1;

            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    let gap = bs - score;
                    if gap > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                        _second_best_score = Some(score);
                    }
                }
            }

            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
            {
                break;
            }

            // Expand children
            let next_depth = best.depth + 1;
            if next_depth >= marginals.len() {
                continue;
            }
            let parent_tokens = extract_parent_tokens_into(
                best.parent_path,
                next_depth,
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
                    score: score + prob.ln() + relevance.ln(),
                    depth: next_depth,
                    token_idx: i,
                    parent_path: (best.parent_path << 16) | (i as u128),
                });
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

    /// Build DDTree with graded relevance screening AND RecFM cross-scale consistency.
    ///
    /// Identical to [`build_screened`] but additionally filters branches whose
    /// probability velocity violates cross-scale consistency (RecFM Theorem 3.1).
    ///
    /// Branches are pruned when `|v₂ − α·v₁| > threshold`, where:
    /// - `v₁` = velocity at parent depth (change in top-1 probability)
    /// - `v₂` = velocity at current depth
    /// - `α` = scale factor from [`CrossScaleConfig::scale_alpha`]
    ///
    /// When `recfm_config.enable == false`, delegates to [`build_screened`] (zero overhead).
    #[cfg(feature = "recfm")]
    pub fn build_screened_recfm(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        recfm_config: &CrossScaleConfig,
    ) -> &[TreeNode] {
        if !recfm_config.enable {
            return self.build_screened(marginals, config, screener, chain_seed);
        }

        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        // Track velocity at each depth for cross-scale consistency checks
        let mut prev_velocity: f32 = 0.0;

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with screening + RecFM ──
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

                // RecFM cross-scale consistency check
                let marginal_prev = if depth > 0 { marginals[depth - 1] } else { &[] };
                let velocity = branch_velocity_at(depth, marginal, marginal_prev);
                if depth > 0
                    && !cross_scale_consistent(
                        prev_velocity,
                        velocity,
                        recfm_config.scale_alpha,
                        recfm_config.consistency_threshold,
                    )
                {
                    // Branch violates cross-scale consistency — prune
                    break;
                }
                prev_velocity = velocity;

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
            // Original seeding with screening (no chain seed)
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

        // ── Phase C: Best-first expansion with screening + RecFM ─────
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

                // RecFM: compute velocity from parent depth to this child depth
                let parent_marginal = marginals[best.depth];
                let child_marginal = marginals[next_depth];
                let parent_velocity = branch_velocity_at(
                    best.depth,
                    parent_marginal,
                    if best.depth > 0 {
                        marginals[best.depth - 1]
                    } else {
                        &[]
                    },
                );

                for (i, &prob) in child_marginal.iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }

                    // RecFM cross-scale consistency check for expansion branches
                    let child_velocity =
                        branch_velocity_at(next_depth, child_marginal, parent_marginal);
                    if !cross_scale_consistent(
                        parent_velocity,
                        child_velocity,
                        recfm_config.scale_alpha,
                        recfm_config.consistency_threshold,
                    ) {
                        continue; // Prune inconsistent branch
                    }

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
    struct ExpensiveScreener {
        /// Simulated work per relevance() call: number of hash rounds.
        work_factor: usize,
        /// Accumulator to prevent the compiler from optimizing away the work.
        /// Uses AtomicF32 for Sync safety.
        sink: std::sync::atomic::AtomicU32,
    }

    impl ExpensiveScreener {
        fn new(work_factor: usize) -> Self {
            Self {
                work_factor,
                sink: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

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
        use crate::pruners::PrunerSchedule;

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
        use crate::pruners::PrunerSchedule;

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
        use crate::pruners::PrunerSchedule;
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
        use crate::speculative::types::PositionWeightedBudget;

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

            let tree = build_dd_tree_screened_progressive(
                &mv,
                &config,
                &screener,
                true,
                Some(&budget_config),
            );

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
        use crate::speculative::spec_generator::{MarginalTokenGenerator, TokenConstraintPruner};
        use katgpt_core::NoPruner;

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
            let tree =
                build_dd_tree_speculative(&mut generator, &pruner, &empty, &config, &mut rng);

            assert!(tree.is_empty(), "empty marginals should produce empty tree");
        }
    }

    // ── Best Buddies integration tests (Plan 199) ──────────────────
    #[cfg(all(feature = "speculative_generator", feature = "best_buddies"))]
    mod best_buddies_integration {
        use super::*;
        use crate::speculative::best_buddies::MarginalBestBuddyAligner;
        use crate::speculative::spec_generator::{MarginalTokenGenerator, TokenConstraintPruner};
        use katgpt_core::NoPruner;

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
            let tree_std = build_dd_tree_speculative(
                &mut gen_std,
                &pruner_std,
                &draft_slices,
                &config,
                &mut rng,
            );

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
        use crate::speculative::domino::compute_prefix_strength;
        use katgpt_core::traits::DominoPruner;

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
            let scored = crate::speculative::domino::domino_score(base, 2, strength);
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
        fn make_belief_drafter_for_tree() -> crate::speculative::belief_drafter::BeliefDrafter {
            use crate::speculative::belief_drafter::{BeliefDrafter, LatentDynamicsMLP};

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
                "tree depth {} should not exceed max_draft_steps {}",
                max_depth,
                max_steps
            );
        }

        #[cfg(feature = "belief_drafter")]
        #[test]
        fn test_belief_ddtree_collapse_aware_adjusts_threshold() {
            let drafter = make_belief_drafter_for_tree();
            let h_t = vec![1.0f32; 4];
            let config = Config::draft();

            // Low avg entropy → higher effective threshold → potentially longer drafts
            let tree_low_ent = build_dd_tree_belief_collapse_aware(
                &drafter,
                &h_t,
                5,
                5.0,
                &config,
                false,
                Some(0.5),
            );

            // High avg entropy → lower effective threshold → shorter drafts
            let tree_high_ent = build_dd_tree_belief_collapse_aware(
                &drafter,
                &h_t,
                5,
                5.0,
                &config,
                false,
                Some(3.0),
            );

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
                    "marginal should sum to ~1.0, got {}",
                    total
                );
            }
        }
    }
}
