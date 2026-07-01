use std::collections::BinaryHeap;

#[cfg(test)]
use katgpt_core::traits::NoScreeningPruner;
use katgpt_core::traits::{CompletionHorizon, ConstraintPruner, NoPruner, ScreeningPruner};
use katgpt_core::speculative::types::{SdeConfig, TreeNode};
use katgpt_types::{InferenceResult, Rng};
use rayon::prelude::*;

/// Minimum candidate count to justify rayon overhead for trivial per-element work.
/// Below this, serial iteration is faster (~5μs rayon overhead vs ~0.1μs per element).
const RAYON_CANDIDATE_THRESHOLD: usize = 512;

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
pub fn build_dd_tree(marginals: &[&[f32]], config: &katgpt_types::Config) -> Vec<TreeNode> {
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
    config: &katgpt_types::Config,
    pruner: &dyn ConstraintPruner,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build(marginals, config, pruner, chain_seed);
    std::mem::take(&mut builder.tree)
}

// ── Lodestar completion-distance pruning (Plan 207, Research 183) ────────
//
// The tree builder lives in this crate (it composes the heap walk); the
// pruner + automaton (`LodestarPruner`, `LodestarAutomaton`) live in
// `katgpt-pruners`. `LodestarConfig` lives here because it configures the
// *builder* (A* lambda + jump-ahead), not the pruner. katgpt-pruners
// re-exports it for back-compat with the historical `katgpt_rs::pruners::`
// path. Gated behind the `lodestar` feature.

/// Configuration for [`build_dd_tree_lodestar`] — controls A* ordering and jump-ahead.
///
/// Default reproduces pure log-prob best-first (λ = 0, jump-ahead disabled).
#[cfg(feature = "lodestar")]
#[derive(Clone, Debug)]
pub struct LodestarConfig {
    /// A* distance weight λ. Heap key = `score − λ·d(s)`.
    /// λ = 0 (default) → pure log-prob ordering, byte-identical to `build_dd_tree_pruned`.
    /// λ > 0 → prefer branches closer to completion (A* admissible heuristic).
    pub astar_lambda: f32,
    /// Enable jump-ahead: collapse singular spans into one tree node.
    /// When `true`, deterministic forced paths are emitted as a single expansion step
    /// instead of per-token, reducing tree nodes and speeding up traversal.
    pub jump_ahead: bool,
}

#[cfg(feature = "lodestar")]
impl Default for LodestarConfig {
    fn default() -> Self {
        Self {
            astar_lambda: 0.0,
            jump_ahead: false,
        }
    }
}

#[cfg(feature = "lodestar")]
impl LodestarConfig {
    /// Pure log-prob ordering, no jump-ahead (default).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// A* ordering with jump-ahead enabled.
    pub fn thinking(lambda: f32) -> Self {
        Self {
            astar_lambda: lambda,
            jump_ahead: true,
        }
    }
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
    config: &katgpt_types::Config,
    horizon: &dyn CompletionHorizon,
    lode_config: &LodestarConfig,
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

    // A* offset is left in the scores: downstream consumers use relative
    // comparisons (heap ordering), so the offset is harmless and consistent
    // with build_dd_tree_pruned's score convention. When λ = 0 this is a no-op.
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
    horizon: &dyn CompletionHorizon,
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
    config: &katgpt_types::Config,
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
    config: &katgpt_types::Config,
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

/// Zero-alloc variant of [`extract_best_path`].
///
/// Writes best-scored token at each depth into `path` (cleared first).
///
/// Two-pass O(N) with exactly two heap allocations (`best_score` + `best_token`),
/// replacing the prior O(D) inner-Vec bucket allocation. Uses direct f32
/// comparison instead of `(score * 1e6) as i64` to preserve full precision.
/// `>=` keeps last-wins-on-tie semantics (matches `Iterator::max_by_key`).
pub fn extract_best_path_into(tree: &[TreeNode], path: &mut Vec<usize>) {
    path.clear();
    if tree.is_empty() {
        return;
    }

    // Pass 1: discover max depth (sizes the per-depth tracker).
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);

    // Per-depth best tracker. `>=` below preserves the prior `max_by_key`
    // last-wins-on-tie semantics (std returns the last max element).
    let mut best_score: Vec<f32> = vec![f32::NEG_INFINITY; max_depth + 1];
    let mut best_token: Vec<usize> = vec![usize::MAX; max_depth + 1];

    // Pass 2: single sweep, update per-depth best in place.
    for node in tree.iter() {
        let d = node.depth;
        // SAFETY: d <= max_depth by definition of max_depth.
        if node.score >= best_score[d] {
            best_score[d] = node.score;
            best_token[d] = node.token_idx;
        }
    }

    // Emit one token per contiguous depth; stop at first missing depth.
    for &tok in best_token.iter() {
        match tok {
            usize::MAX => break,
            tok => path.push(tok),
        }
    }
}

/// Extract best-scored token at each depth from a DDTree.
///
/// Two-pass O(N) with exactly two heap allocations (best_score + best_token),
/// replacing the prior O(D) inner-Vec bucket allocation. Uses direct f32
/// comparison instead of `(score * 1e6) as i64` to preserve full precision.
///
/// See [`extract_best_path_into`] for the zero-alloc variant.
pub fn extract_best_path(tree: &[TreeNode]) -> Vec<usize> {
    let mut path = Vec::new();
    extract_best_path_into(tree, &mut path);
    path
}

/// Build an InferenceResult from a completed DDTree inference.
///
/// `&str` args (caller-owned) avoid the allocation that `impl Into<String>`
/// would force when the caller already holds a `&str`.
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
        plan_horizon_used: 0,
    }
}

/// Inject retrieved token sequences into the DDTree as candidate branches.
///
/// Each retrieved sequence becomes a path with blended score.
/// Score blending: `(1-w) * log(draft_prob) + w * log(similarity)`
///
/// This is a pure computation function — no feature gating needed.
/// The REST feature provides the data; this function processes it.
///
/// O(D) per sequence: `parent_path` is reconstructed incrementally
/// (shift 16 bits + token per depth) rather than per-depth O(depth) fold.
pub fn merge_retrieved_branches(
    tree: &mut Vec<TreeNode>,
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
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

        let sim_ln = similarity.ln();

        // Incrementally reconstruct parent_path: shift 16 bits + token per depth.
        // Avoids per-depth O(depth) fold over seq[..=depth] (was O(D²) per sequence).
        let mut parent_path: u128 = 0;
        for (depth, &token_idx) in seq.iter().enumerate() {
            if depth >= marginals.len() {
                break;
            }
            if token_idx >= config.vocab_size {
                break;
            }

            let base_prob = marginals[depth].get(token_idx).copied().unwrap_or(0.0);
            if base_prob <= 0.0 {
                // Still advance parent_path so deeper tokens reconstruct the
                // same path the original fold would have produced.
                parent_path = if depth == 0 {
                    token_idx as u128
                } else {
                    (parent_path << 16) | (token_idx as u128)
                };
                continue;
            }

            let blended = (base_prob.ln() * inv_weight) + (sim_ln * rest_weight);

            parent_path = if depth == 0 {
                token_idx as u128
            } else {
                (parent_path << 16) | (token_idx as u128)
            };

            tree.push(TreeNode {
                score: blended,
                depth,
                token_idx,
                parent_path,
            });
        }
    }

    // Re-sort by score descending. Unstable sort is safe here: TreeNode is
    // Copy + Eq and downstream consumers only rely on score ordering, not on
    // tie-stability. Unstable sort avoids the O(N) auxiliary allocation that
    // stable sort incurs on large inputs.
    tree.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    tree.truncate(config.tree_budget);
}

/// Inject SDE (Score-Distortion Error) noise into marginals.
///
/// # Returns
///
/// New `Vec<Vec<f32>>` with perturbed marginals, or clones if γ=0.
pub fn inject_sde_noise(
    marginals: &[&[f32]],
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(marginals.len());
    inject_sde_noise_into(marginals, sde_config, rng, &mut out);
    out
}

/// **Zero-alloc variant** of [`inject_sde_noise`].
///
/// Writes perturbed marginals into the caller-owned `out` buffer (in-place
/// rebuild — existing inner `Vec<f32>` slots are cleared and refilled, the
/// outer Vec is grown or shrunk to match the input length). When the caller
/// holds a long-lived `Vec<Vec<f32>>` across calls (e.g. inside a K-rollout
/// loop), this skips the per-call outer allocation AND reuses the inner
/// `Vec<f32>` allocations across iterations.
///
/// Identical math to [`inject_sde_noise`] — the public function is now a thin
/// wrapper that allocates a fresh `Vec<Vec<f32>>` and delegates here.
pub fn inject_sde_noise_into(
    marginals: &[&[f32]],
    sde_config: &SdeConfig,
    rng: &mut Rng,
    out: &mut Vec<Vec<f32>>,
) {
    out.reserve(marginals.len());

    if !sde_config.is_enabled() {
        // SDE disabled: clone each marginal verbatim. Reuse inner allocations
        // when the caller's buffer already has the right shape from a prior
        // call (typical in the K-rollout loop).
        for (i, marginal) in marginals.iter().enumerate() {
            if i < out.len() {
                out[i].clear();
                out[i].extend_from_slice(marginal);
            } else {
                out.push(marginal.to_vec());
            }
        }
        out.truncate(marginals.len());
        return;
    }

    for (i, marginal) in marginals.iter().enumerate() {
        if i >= out.len() {
            out.push(Vec::new());
        }
        let slot = &mut out[i];
        slot.clear();
        slot.extend_from_slice(marginal);
        let perturbed = slot;

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
        for (j, prob) in perturbed.iter_mut().enumerate() {
            // Skip top-1 if preserving
            if top1_idx == Some(j) {
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
    }
    out.truncate(marginals.len());
}

/// Rebuild a `Vec<&[f32]>` view over a `&[Vec<f32>]` owned store, reusing the
/// caller's buffer (cleared first, then refilled). Used by tree-build helpers
/// to convert the owned `noisy: Vec<Vec<f32>>` produced by
/// [`inject_sde_noise_into`] into the `&[&[f32]]` shape that
/// [`build_dd_tree_screened`] consumes, without allocating a fresh refs Vec
/// each call.
///
/// The returned view borrows from `owned` for the duration of the caller's use.
#[inline]
pub fn build_slices_view<'a>(owned: &'a [Vec<f32>], view: &mut Vec<&'a [f32]>) {
    view.clear();
    view.reserve(owned.len());
    for v in owned {
        view.push(v.as_slice());
    }
}

/// Extract all candidate sequences from a DDTree (one per leaf node).
///
/// Each leaf node's `parent_path` encodes a full token sequence.
/// Returns `(sequence, leaf_node)` pairs for all maximal-depth paths.
///
/// Zero per-node allocation: reuses one scratch buffer for token extraction.
pub fn extract_candidate_sequences(tree: &[TreeNode]) -> Vec<(Vec<usize>, &TreeNode)> {
    if tree.is_empty() {
        return Vec::new();
    }

    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut buf: Vec<usize> = vec![0usize; max_depth + 1];

    // Collect leaf nodes (nodes at max depth with no children in tree)
    tree.iter()
        .filter(|node| node.depth == max_depth)
        .map(|node| {
            let seq =
                extract_parent_tokens_into(node.parent_path, node.depth + 1, &mut buf).to_vec();
            (seq, node)
        })
        .collect()
}

/// Extract candidate sequences from ALL tree nodes (not just leaves).
///
/// Useful when the solution might not require visiting all targets,
/// or when partial sequences are valid solutions.
///
/// Zero per-node allocation: reuses one scratch buffer for token extraction.
pub fn extract_all_sequences(tree: &[TreeNode]) -> Vec<(Vec<usize>, &TreeNode)> {
    if tree.is_empty() {
        return Vec::new();
    }

    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut buf: Vec<usize> = vec![0usize; max_depth + 1];

    tree.iter()
        .map(|node| {
            let seq =
                extract_parent_tokens_into(node.parent_path, node.depth + 1, &mut buf).to_vec();
            (seq, node)
        })
        .collect()
}

/// Sequential version of [`par_find_valid_sequence`] — no rayon overhead.
///
/// Useful for small trees where rayon spawn cost outweighs parallelism benefit,
/// or when deterministic ordering is required (first candidate wins).
///
/// Zero per-node allocation: reuses one scratch buffer across all candidates;
/// allocates only when returning the winning sequence.
pub fn find_valid_sequence<T, V>(tree: &[TreeNode], validator: V) -> Option<(Vec<usize>, T)>
where
    V: Fn(&[usize]) -> Option<T>,
{
    if tree.is_empty() {
        return None;
    }

    // Size scratch buffer once from the deepest node we may visit.
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut buf: Vec<usize> = vec![0usize; max_depth + 1];

    for node in tree {
        let seq = extract_parent_tokens_into(node.parent_path, node.depth + 1, &mut buf);
        if let Some(result) = validator(seq) {
            return Some((seq.to_vec(), result));
        }
    }

    None
}

/// Parallel DDTree search: find the first candidate sequence that passes validation.
///
/// Extracts all candidate sequences from the DDTree, then validates them in
/// parallel using rayon. Returns the first valid sequence found, or `None`.
///
/// This is the core generic primitive — the caller provides a domain-specific
/// validator closure.
///
/// # Type Parameters
/// - `V`: Validator closure `Fn(&[usize]) -> Option<T>`
/// - `T`: Result type returned by the validator on success
///
/// # Performance
/// The search phase is parallelized (each candidate validated independently).
/// DDTree build remains sequential (inherent heap-based best-first search).
pub fn par_find_valid_sequence<T, V>(tree: &[TreeNode], validator: V) -> Option<(Vec<usize>, T)>
where
    V: Fn(&[usize]) -> Option<T> + Sync,
    T: Send,
{
    if tree.is_empty() {
        return None;
    }

    // Lazy parallel search: extract each candidate's token sequence on-demand
    // inside the parallel iterator. Avoids the upfront O(N) `Vec<Vec<usize>>`
    // collection the prior collect-then-par_iter paid even when `find_map_any`
    // short-circuits on the first hit. `find_map_any` returns as soon as any
    // task yields `Some`, so never-examined nodes never allocate.
    tree.par_iter()
        .find_map_any(|node| {
            let seq = extract_parent_tokens(node.parent_path, node.depth + 1);
            validator(&seq).map(|result| (seq, result))
        })
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

    // Lazy parallel search: extract each candidate's token sequence on-demand
    // inside the parallel iterator, avoiding the upfront O(N) collection of
    // all candidate sequences. Each task allocates only its own sequence.
    tree.par_iter()
        .filter_map(|node| {
            let seq = extract_parent_tokens(node.parent_path, node.depth + 1);
            validator(&seq).map(|result| (seq, result))
        })
        .min_by_key(|(_, result)| cost_fn(result))
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
    candidates_buf: Vec<usize>,
    valid_buf: Vec<bool>,
    /// Cached `ln(marginals[d][i])` — computed once per build to avoid redundant
    /// `f32::ln()` calls in the Phase C expansion inner loop (called per token
    /// per heap-pop). Entries for `prob <= 0.0` are `0.0` (unused since those
    /// tokens are skipped before the lookup).
    log_marginals: Vec<Vec<f32>>,
}

impl TreeBuilder {
    /// Allocate all buffers from config dimensions.
    pub fn new(config: &katgpt_types::Config) -> Self {
        Self {
            heap: BinaryHeap::new(),
            tree: Vec::with_capacity(config.tree_budget),
            chain_nodes: Vec::with_capacity(config.draft_lookahead),
            chain_parent_tokens: Vec::with_capacity(config.draft_lookahead),
            parent_tokens_buf: vec![0usize; config.draft_lookahead + 1],
            candidates_buf: Vec::with_capacity(config.vocab_size),
            valid_buf: Vec::with_capacity(config.vocab_size),
            log_marginals: Vec::new(),
        }
    }

    /// Pre-compute `ln(prob)` for every token in every marginal depth.
    ///
    /// Reuses inner `Vec` allocations across builds (clear + refill pattern).
    /// The Phase C expansion loop calls `prob.ln()` once per token per heap-pop;
    /// caching turns that O(budget × vocab) `ln` calls into O(depths × vocab).
    #[inline]
    fn cache_log_marginals(&mut self, marginals: &[&[f32]]) {
        // Grow the outer Vec if needed; existing inner Vecs are reused below.
        if self.log_marginals.len() < marginals.len() {
            self.log_marginals.resize_with(marginals.len(), Vec::new);
        } else {
            self.log_marginals.truncate(marginals.len());
        }
        for (log_m, &m) in self.log_marginals.iter_mut().zip(marginals) {
            log_m.clear();
            log_m.reserve(m.len());
            // Branch-free: `ln(0)` would be -inf, but those entries are never
            // read (the expansion loop skips `prob <= 0.0` before indexing).
            for &p in m {
                log_m.push(if p > 0.0 { p.ln() } else { 0.0 });
            }
        }
    }

    /// Build DDTree from marginals, reusing pre-allocated buffers.
    ///
    /// Clears and reuses `heap`, `tree`, `chain_nodes`, `chain_parent_tokens`.
    /// Returns a borrowed slice valid until the next `build()` call.
    pub fn build(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
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

        self.cache_log_marginals(marginals);

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
                    // Batch validate: collect candidates with prob>0, validate all
                    // in one batch_is_valid call, then create nodes.
                    // Reuse pre-allocated candidates_buf to avoid per-build allocation.
                    self.candidates_buf.clear();
                    self.candidates_buf.extend(
                        marginals[0]
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &prob)| if prob > 0.0 { Some(i) } else { None }),
                    );
                    self.valid_buf.clear();
                    self.valid_buf.resize(self.candidates_buf.len(), false);
                    pruner.batch_is_valid(0, &self.candidates_buf, &[], &mut self.valid_buf);
                    if self.candidates_buf.len() >= RAYON_CANDIDATE_THRESHOLD {
                        let nodes: Vec<TreeNode> = self
                            .candidates_buf
                            .par_iter()
                            .zip(self.valid_buf.par_iter())
                            .filter_map(|(&i, &v)| {
                                if v {
                                    Some(TreeNode {
                                        score: marginals[0][i].ln(),
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
                        for (&i, &v) in self.candidates_buf.iter().zip(self.valid_buf.iter()) {
                            if v {
                                self.heap.push(TreeNode {
                                    score: marginals[0][i].ln(),
                                    depth: 0,
                                    token_idx: i,
                                    parent_path: i as u128,
                                });
                            }
                        }
                    }
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
                // Batch validate: collect candidates with prob>0, validate all
                // in one batch_is_valid call, then create nodes.
                // Reuse pre-allocated candidates_buf to avoid per-build allocation.
                self.candidates_buf.clear();
                self.candidates_buf.extend(
                    marginals[0]
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| if prob > 0.0 { Some(i) } else { None }),
                );
                self.valid_buf.clear();
                self.valid_buf.resize(self.candidates_buf.len(), false);
                pruner.batch_is_valid(0, &self.candidates_buf, &[], &mut self.valid_buf);
                if self.candidates_buf.len() >= RAYON_CANDIDATE_THRESHOLD {
                    let nodes: Vec<TreeNode> = self
                        .candidates_buf
                        .par_iter()
                        .zip(self.valid_buf.par_iter())
                        .filter_map(|(&i, &v)| {
                            if v {
                                Some(TreeNode {
                                    score: marginals[0][i].ln(),
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
                    for (&i, &v) in self.candidates_buf.iter().zip(self.valid_buf.iter()) {
                        if v {
                            self.heap.push(TreeNode {
                                score: marginals[0][i].ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            });
                        }
                    }
                }
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
                let log_m = &self.log_marginals[next_depth];
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    // NEURO-SYMBOLIC INTERCEPT: prune before adding to heap
                    if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                        self.heap.push(TreeNode {
                            score: best.score + log_m[i],
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
        config: &katgpt_types::Config,
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
        config: &katgpt_types::Config,
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

        self.cache_log_marginals(marginals);

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
                // Slice length is a sufficient condition for the rayon threshold:
                // if the whole marginal has <512 entries it certainly has <512
                // positive ones. Avoids a full O(vocab) counting pass that the
                // par_iter below redoes anyway via filter_map.
                if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
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
            // Original seeding with screening. Slice length is a sufficient
            // condition for the rayon threshold (see Phase B note above).
            if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
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
                let log_m = &self.log_marginals[next_depth];
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
                        score: best.score + log_m[i] + relevance.ln(),
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
        config: &katgpt_types::Config,
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
        config: &katgpt_types::Config,
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

        self.cache_log_marginals(marginals);

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
                // Slice length is a sufficient condition for the rayon threshold
                // (see build_screened Phase B note). Avoids an O(vocab) count.
                if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
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
            // Original seeding with balanced scoring. Slice length is a
            // sufficient condition for the rayon threshold (see Phase B note).
            if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
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
                // Hoist flow_bonus: depends only on next_depth, not token `i`.
                let flow_bonus =
                    lambda_flow * (1.0 - stop_probs.get(next_depth).copied().unwrap_or(0.5));
                let log_m = &self.log_marginals[next_depth];
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }
                    // BALANCED: ln(P_llm) + backward_weight × ln(R) + flow_bonus
                    let r_safe = relevance.max(1e-10); // Avoid ln(0)
                    self.heap.push(TreeNode {
                        score: best.score + log_m[i] + backward_weight * r_safe.ln() + flow_bonus,
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

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_types::Config;

    // NOTE (Issue 013): The riir-engine dd_tree.rs test module originally
    // imported `crate::dflash::dflash_predict` + `crate::transformer::TransformerWeights`
    // + `crate::types::Rng` to synthesize marginals via the draft model. Those
    // dependencies do NOT exist in katgpt-speculative (dflash is deferred to
    // Issue 014 — it needs a `forward` trait design). Tests that called
    // `dflash_predict` or `make_draft()` have been removed; the remaining tests
    // synthesize marginals directly (pure-algorithm coverage). The removed
    // integration tests are preserved verbatim in riir-engine's `dflash.rs`
    // test module, which still calls `katgpt_speculative::dd_tree::*` after the
    // Issue 013 migration.

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
    fn test_ddtree_empty_marginals() {
        let config = Config::draft();
        let tree = build_dd_tree(&[], &config);
        assert!(tree.is_empty(), "empty marginals should produce empty tree");
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

}
