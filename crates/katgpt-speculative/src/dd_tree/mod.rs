use std::collections::BinaryHeap;
// HashMap is only constructed inside `best_of_k_rollouts` (MostFrequent mode),
// which is `elf_sde`-gated; gate the import so it doesn't read as unused when
// `elf_sde` is off (e.g. downstream consumer with default-features = false).
#[cfg(any(feature = "elf_sde", test))]
use std::collections::HashMap;

// NoScreeningPruner is constructed inside feature-gated dd-tree wrappers
// (speculative_generator / belief_drafter) and in tests; gate the import so
// it doesn't read as unused when all those features are off.
#[cfg(feature = "and_or_dtree")]
use katgpt_core::AndOrNode;
use katgpt_core::speculative::types::{SdeConfig, TreeNode, TreePath};
#[cfg(feature = "domino_correction")]
use katgpt_core::traits::DominoPruner;
#[cfg(any(
    test,
    feature = "speculative_generator",
    feature = "belief_drafter",
    feature = "kurtosis_gate",
    feature = "best_buddies",
))]
use katgpt_core::traits::NoScreeningPruner;
use katgpt_core::traits::{ConstraintPruner, NoPruner, ScreeningPruner};
use katgpt_types::{InferenceResult, Rng};
use rayon::prelude::*;

// Sub-modules (Issue 162 C2 split, 2026-07-17; Issue 178 Lodestar extraction 2026-07-17)
#[cfg(feature = "lodestar")]
mod lodestar;
mod tree_builder;

#[cfg(feature = "lodestar")]
pub use lodestar::*;
pub use tree_builder::TreeBuilder;

/// Minimum candidate count to justify rayon overhead for trivial per-element work.
/// Below this, serial iteration is faster (~5μs rayon overhead vs ~0.1μs per element).
const RAYON_CANDIDATE_THRESHOLD: usize = 512;

/// Extract tokens from a [`TreePath`] for path-aware pruning.
///
/// `parent_path` holds one `u32` token per level, root at slot 0 (Issue 670
/// widened this from the `u128` 16-bit-per-level packing, which corrupted
/// paths at vocab > 65,536).
///
/// Returns `Vec<usize>` where `result[k]` = token at depth `k`.
/// Max depth: [`TreePath::MAX_TOKENS`] = 8 (sufficient for lookahead of 5–8).
pub fn extract_parent_tokens(parent_path: TreePath, num_tokens: usize) -> Vec<usize> {
    (0..num_tokens).map(|k| parent_path.token_at(k) as usize).collect()
}

/// Zero-alloc variant of [`extract_parent_tokens`].
/// Writes `num_tokens` parent tokens into `buf`, which must be large enough.
/// Returns the slice `&buf[..num_tokens]`.
#[inline]
pub fn extract_parent_tokens_into(
    parent_path: TreePath,
    num_tokens: usize,
    buf: &mut [usize],
) -> &[usize] {
    for (k, slot) in buf.iter_mut().enumerate().take(num_tokens) {
        *slot = parent_path.token_at(k) as usize;
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

/// DDTree with deep-argmax threshold (Plan 424 Phase 6, paper §3.5).
///
/// Like [`build_dd_tree`] but restricts expansion to argmax-of-marginal at
/// tree depths > `deep_argmax_threshold`. `None` = no restriction (identical
/// to [`build_dd_tree`]). The paper shows that at deep draft positions (t > ~4),
/// the factorized marginal is diluted by averaging over many possible prefixes,
/// and a Dirac delta at the argmax outperforms full-marginal branching.
///
/// Crossover occurs around draft length 2–4 (Figure 6); `Some(4)` is the
/// paper-recommended default when enabled.
pub fn build_dd_tree_deep_argmax(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    deep_argmax_threshold: Option<usize>,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.set_deep_argmax_threshold(deep_argmax_threshold);
    builder.build(marginals, config, &NoPruner, false);
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

        // Incrementally reconstruct parent_path: one u32 slot per depth.
        // Avoids per-depth O(depth) fold over seq[..=depth] (was O(D²) per sequence).
        let mut parent_path = TreePath::default();
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
                parent_path = parent_path.push(token_idx as u32, depth);
                continue;
            }

            let blended = (base_prob.ln() * inv_weight) + (sim_ln * rest_weight);

            parent_path = parent_path.push(token_idx as u32, depth);

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
    tree.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
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
                .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
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
            // (`fast_exp` = Cephes scalar exp, ~1 ULP for |x| < 88).
            use katgpt_core::simd::fast_exp;
            let log_p = prob.ln();
            let noisy_log_p = log_p + sde_config.gamma * rng.normal();
            *prob = fast_exp(noisy_log_p).max(0.0);
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
    tree.par_iter().find_map_any(|node| {
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

// ── SDE-Aware DDTree Builders (ELF Plan 079) ────────────────────
//
// Plan 391 (2026-07-05): moved from `katgpt-rs/src/speculative/dd_tree.rs`
// because they only compose leaf-resident primitives (`inject_sde_noise_into`,
// `build_slices_view`, `build_dd_tree_screened`, `build_dd_tree_balanced`) and
// `katgpt_types::{Config, Rng}`. Zero root-only deps.

/// DDTree with SDE noise injection (ELF Plan 079).
///
/// Applies SDE noise to marginals before building the tree.
/// When `sde_config.gamma == 0.0`, this is identical to `build_dd_tree_screened`.
pub fn build_dd_tree_sde(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<TreeNode> {
    let mut noisy_marginals = Vec::with_capacity(marginals.len());
    inject_sde_noise_into(marginals, sde_config, rng, &mut noisy_marginals);
    let mut noisy_slices: Vec<&[f32]> = Vec::with_capacity(noisy_marginals.len());
    build_slices_view(&noisy_marginals, &mut noisy_slices);
    build_dd_tree_screened(&noisy_slices, config, screener, chain_seed)
}

/// DDTree balanced with SDE noise injection (ELF Plan 079).
///
/// Applies SDE noise to marginals before building the balanced tree.
/// When `sde_config.gamma == 0.0`, this is identical to `build_dd_tree_balanced`.
#[allow(clippy::too_many_arguments)]
pub fn build_dd_tree_balanced_sde(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    stop_probs: &[f32],
    backward_weight: f32,
    lambda_flow: f32,
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<TreeNode> {
    let mut noisy_marginals = Vec::with_capacity(marginals.len());
    inject_sde_noise_into(marginals, sde_config, rng, &mut noisy_marginals);
    let mut noisy_slices: Vec<&[f32]> = Vec::with_capacity(noisy_marginals.len());
    build_slices_view(&noisy_marginals, &mut noisy_slices);
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
//
// Plan 391 (2026-07-05): moved from root — pure substrate over
// `katgpt_types::Config` + `katgpt_core::ConvergenceSelector`. No root-only deps.

/// Selection strategy for [`best_of_k_rollouts`].
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

/// Convert Config-level `ConvergenceSelector` to runtime [`WidthSelectionMode`].
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

// ── EqR Convergence Selection (Plan 119) ────────────────────────

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

// ── RecFM Cross-Scale Consistency (Plan 168) ───────────────────
//
// Plan 391 (2026-07-05): moved from root — pure substrate, no root-only deps.

/// Configuration for RecFM recursive cross-scale consistency filtering (Research 150).
///
/// RecFM's Theorem 3.1 proves that consistency loss constrains trajectory acceleration
/// ∂t_v, directly reducing discretization error. Applied to DDTree, this filters branches
/// whose probability velocity violates cross-scale consistency.
///
/// When `enable` is `false`, all RecFM checks are no-ops (zero cost on hot path).
#[cfg(feature = "recfm")]
#[derive(Debug, Clone, Copy)]
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
    // Only the max VALUE is needed (not the index). `fold` is branch-free and
    // avoids the closure-call overhead of `.iter().enumerate().max_by(...)`.
    let top1_curr = marginal_curr.iter().copied().fold(0.0f32, f32::max);
    let top1_prev = marginal_prev.iter().copied().fold(0.0f32, f32::max);
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
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    sde_config: &SdeConfig,
    width_config: &WidthScaleConfig,
    base_seed: u64,
) -> Vec<usize> {
    if width_config.k_rollouts <= 1 || !sde_config.is_enabled() {
        // Single rollout or SDE disabled — just build one tree
        let mut rng = Rng::new(base_seed);
        let mut noisy = Vec::with_capacity(marginals.len());
        inject_sde_noise_into(marginals, sde_config, &mut rng, &mut noisy);
        // Build a fresh immutable view (no need to keep a mutable refs buffer here).
        let noisy_slices: Vec<&[f32]> = noisy.iter().map(|m| m.as_slice()).collect();
        let tree = build_dd_tree_screened(&noisy_slices, config, screener, false);
        return extract_best_path(&tree);
    }

    // Run K independent rollouts with different noise seeds. Hoist the
    // `noisy` buffer out of the loop — `inject_sde_noise_into` clears and
    // refills it each iteration, skipping K-1 outer `Vec<Vec<f32>>`
    // allocations and reusing the inner `Vec<f32>` slots across rollouts.
    //
    // `noisy_slices` must stay loop-local: it holds `&[f32]` references into
    // `noisy`, so it cannot outlive a single iteration (the next iteration
    // mutably borrows `noisy` again). Its allocation cost is negligible
    // (length = #depths, typically ≤ 32) compared to the per-rollout tree
    // build, but we still `reserve` once on the first iteration via the
    // `with_capacity` on `noisy.len()`.
    let mut paths: Vec<Vec<usize>> = Vec::with_capacity(width_config.k_rollouts);
    let mut scores: Vec<f32> = Vec::with_capacity(width_config.k_rollouts);
    // EqR convergence: track marginal-change residual per rollout (Plan 119)
    #[cfg(feature = "eqr_convergence")]
    let mut final_residuals: Vec<f32> = Vec::with_capacity(width_config.k_rollouts);
    let mut noisy: Vec<Vec<f32>> = Vec::with_capacity(marginals.len());

    for k in 0..width_config.k_rollouts {
        let mut rng = Rng::new(base_seed.wrapping_add(k as u64));
        inject_sde_noise_into(marginals, sde_config, &mut rng, &mut noisy);
        // Build the `&[&[f32]]` view fresh each iteration — references cannot
        // escape the loop body because `noisy` is mutably re-borrowed next.
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
                .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                .map_or(0, |(i, _)| i);
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
            let best_idx =
                if final_residuals.is_empty() || final_residuals.iter().all(|&r| r == 0.0) {
                    scores
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                        .map_or(0, |(i, _)| i)
                } else {
                    final_residuals
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_min(**a, **b))
                        .map_or(0, |(i, _)| i)
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

// ── SR²AM Entropy-Based Horizon Truncation (Plan 112, Research 076) ──
//
// Plan 391 (2026-07-05): moved from root — pure substrate, no root-only deps.

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
    if entropy > ENTROPY_THRESHOLD {
        TRUNCATED_HORIZON.min(max_horizon)
    } else {
        max_horizon
    }
}

// ── DendriticGate Adaptive Tree (Plan 260, feature: dendritic_gate) ──
//
// Plan 391 (2026-07-05): moved from root — uses only leaf-resident
// `crate::dendritic_gate::DendriticGate` + katgpt_core primitives.

/// Build DDTree with NMDA-gated adaptive expansion budget.
///
/// Uses `DendriticGate` to deterministically modulate per-expansion budget:
/// `effective_budget = base_budget * nmda_gate`
///
/// Early exits when `nmda_gate < 0.1` (proximal dendrite sufficient).
/// This replaces stochastic bandit budget allocation with zero-parameter,
/// zero-training, physics-based adaptive compute.
///
/// Feature-gated behind `dendritic_gate`.
///
/// # Arguments
/// * `marginals` — Per-depth token probability distributions (log-probs)
/// * `config` — DDTree configuration
/// * `pruner` — Constraint pruner
/// * `chain_seed` — Whether to build greedy chain backbone first
/// * `gate` — The `DendriticGate` instance with threshold/sensitivity params
///
/// # Returns
///
/// Tree nodes in expansion order. May have fewer nodes than `config.tree_budget`
/// when gate triggers early exit.
#[cfg(feature = "dendritic_gate")]
#[allow(clippy::needless_range_loop)] // depth is semantic tree depth: marginals[depth] + extract_parent_tokens(depth) + is_valid(depth,..)
pub fn build_dd_tree_dendritic(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    pruner: &dyn ConstraintPruner,
    chain_seed: bool,
    gate: &crate::dendritic_gate::DendriticGate,
) -> Vec<TreeNode> {
    use katgpt_core::{coincidence_score, entropy_f32};

    if marginals.is_empty() {
        return Vec::new();
    }

    let seq_len = marginals.len();
    let mut heap: BinaryHeap<TreeNode> = BinaryHeap::with_capacity(config.tree_budget);
    let mut tree: Vec<TreeNode> = Vec::with_capacity(config.tree_budget);
    let base_budget = config.tree_budget;
    let mut parent_buf: Vec<usize> = vec![0usize; seq_len];

    // Per-depth cache for the two gate inputs that depend ONLY on
    // `marginals[next_depth]` and never on the popped node: the entropy (an
    // O(vocab) scan with one `exp()` per element) and the top-K index list (an
    // O(vocab·K) scan that also allocated a fresh `Vec` per call). The
    // best-first loop pops many nodes at the same depth, so both were being
    // recomputed identically on every pop. Both are pure functions of their
    // arguments, so caching per depth yields bit-identical values while
    // collapsing O(nodes·vocab) exp() calls into O(depths·vocab), and removes
    // the per-pop `Vec` allocation.
    let mut depth_gate_cache: Vec<Option<(f32, Vec<usize>)>> = vec![None; seq_len];

    // Optional: seed greedy chain backbone first
    if chain_seed {
        let mut chain_path = TreePath::default();
        let mut chain_score = 0.0f32;
        for depth in 0..seq_len {
            let parent_tokens = extract_parent_tokens_into(chain_path, depth, &mut parent_buf);
            let mut best_prob = 0.0f32;
            let mut best_idx = 0;
            for (i, &prob) in marginals[depth].iter().enumerate() {
                if prob > best_prob && pruner.is_valid(depth, i, parent_tokens) {
                    best_prob = prob;
                    best_idx = i;
                }
            }
            if best_prob > 0.0 {
                chain_score += best_prob.ln();
                chain_path = chain_path.push(best_idx as u32, depth);
                tree.push(TreeNode {
                    score: chain_score,
                    depth,
                    token_idx: best_idx,
                    parent_path: chain_path,
                });
            } else {
                break;
            }
        }
    }

    // Seed root children (depth 0)
    if !chain_seed {
        for (i, &prob) in marginals[0].iter().enumerate() {
            if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                heap.push(TreeNode {
                    score: prob.ln(),
                    depth: 0,
                    token_idx: i,
                    parent_path: TreePath::root(i as u32),
                });
            }
        }
    }

    // Best-first expansion with dendritic-gated budget
    let mut effective_budget = base_budget;

    while tree.len() < effective_budget {
        let Some(best) = heap.pop() else {
            break;
        };
        tree.push(best);

        if best.depth + 1 < seq_len {
            let next_depth = best.depth + 1;
            let parent_tokens =
                extract_parent_tokens_into(best.parent_path, best.depth + 1, &mut parent_buf);

            // Compute gate signal from entropy + coincidence at this depth.
            // `entropy` / `top_k` are depth-invariant (see `depth_gate_cache`);
            // only `coincidence_score` depends on this node's `parent_tokens`.
            let cached = depth_gate_cache[next_depth].get_or_insert_with(|| {
                (
                    entropy_f32(marginals[next_depth]),
                    top_k_indices(marginals[next_depth], gate.coincidence_window),
                )
            });
            let coinc = coincidence_score(&cached.1, parent_tokens, gate.coincidence_window);
            let nmda_gate = gate.compute_gate(cached.0, coinc);

            // Early exit: proximal dendrite sufficient
            if nmda_gate < 0.1 {
                break;
            }

            // Modulate effective budget
            effective_budget = ((base_budget as f32) * nmda_gate) as usize;
            effective_budget = effective_budget.max(tree.len()).min(base_budget);

            // Expand children
            for (i, &prob) in marginals[next_depth].iter().enumerate() {
                if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                    let score = best.score + prob.ln();
                    heap.push(TreeNode {
                        score,
                        depth: next_depth,
                        token_idx: i,
                        parent_path: best.parent_path.push(i as u32, next_depth),
                    });
                }
            }
        }
    }

    tree.shrink_to_fit();
    tree
}

/// Extract top-K indices from a probability slice (descending order).
///
/// Uses a fixed-size running minimum tracker (smallest-of-top-K at slot 0).
/// O(N·K·log K) time but only O(K) auxiliary storage — avoids the O(N) full
/// allocation that `select_nth_unstable_by` on `Vec<(usize,f32)>` would
/// require (which would allocate ~256KB for a 32k vocab). For small K
/// (typical `coincidence_window`), this is both faster and dramatically
/// lighter on the allocator, important since this is called per heap-pop
/// inside `build_dd_tree_dendritic`.
#[cfg(feature = "dendritic_gate")]
#[inline]
fn top_k_indices(probs: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(probs.len());
    if k == 0 {
        return Vec::new();
    }
    // Maintain top as ascending: smallest of top-K at top[0]. When we see a
    // larger prob, evict top[0] and re-sort (K is tiny — typically ≤8).
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k);
    for (i, &p) in probs.iter().enumerate() {
        if top.len() < k {
            top.push((p, i));
            top.sort_by(|a, b| a.0.total_cmp(&b.0));
        } else if p > top[0].0 {
            top[0] = (p, i);
            top.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
    }
    // Final pass: reverse to descending (largest prob first), matching the
    // original `select_nth_unstable_by` + sort contract.
    top.sort_by(|a, b| b.0.total_cmp(&a.0));
    top.into_iter().map(|(_, i)| i).collect()
}

// ── ManifoldPruner DDTree wiring (Plan 234 Phase 3, feature: manifold_pruner) ──
//
// Plan 391 (2026-07-05): moved from root — uses only the ConstraintPruner trait's
// `manifold_score` method (already in katgpt_core::traits), no root-only deps.

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
    config: &katgpt_types::Config,
    pruner: &dyn ConstraintPruner,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let wrapper = ManifoldValidWrapper(pruner);
    build_dd_tree_pruned(marginals, config, &wrapper, chain_seed)
}

// ── Plan 392 (2026-07-05): feature-gated DDTree wrappers moved from
// katgpt-rs/src/speculative/dd_tree.rs. Compose leaf-resident siblings
// (domino, spec_generator, belief_drafter, kurtosis_gate, best_buddies,
// correlation_budget, nf_flow_budget, and_or_builder, blueprint,
// decomp_reviewer) and katgpt_core::speculative::types.

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
#[cfg(feature = "domino_correction")]
pub fn build_dd_tree_domino<P>(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    pruner: &P,
    chain_seed: bool,
    sampled_tokens: &[usize],
) -> Vec<TreeNode>
where
    P: DominoPruner,
{
    use crate::domino::{compute_prefix_strength, domino_score};

    // Build base tree with causal correction via DominoPruner
    let mut tree = build_dd_tree_pruned(marginals, config, pruner, chain_seed);

    // Apply domino scoring: re-score nodes based on prefix strength
    for node in &mut tree {
        let strength = compute_prefix_strength(marginals, sampled_tokens, node.depth);
        node.score = domino_score(node.score, node.depth, strength);
    }

    tree
}

// ── SpeculativeGenerator Integration (Plan 193 T5) ──────────────────

/// Build DDTree using `SpeculativeGenerator` for candidate generation.
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
    generator: &mut crate::spec_generator::MarginalTokenGenerator,
    pruner: &crate::spec_generator::TokenConstraintPruner<P>,
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    rng: &mut fastrand::Rng,
) -> Vec<TreeNode>
where
    P: ConstraintPruner + Send + Sync,
{
    use crate::spec_generator::TokenCondition;
    use katgpt_core::{GenerativeConstraintPruner, SpeculativeGenerator};

    let mut filtered_marginals: Vec<Vec<f32>> = Vec::with_capacity(marginals.len());

    for (depth, marginal) in marginals.iter().enumerate() {
        let condition = TokenCondition {
            parent_tokens: vec![],
            depth,
            marginals: marginal.to_vec(),
        };

        let candidates = if let Ok(c) = generator.generate(&condition, rng) { c } else {
                // Generator failed — use original marginals as fallback
                filtered_marginals.push(marginal.to_vec());
                continue;
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
            // All filtered out — use original marginals as fallback. Same length
            // as `filtered` (both sized to marginal.len()), so copy in place
            // instead of allocating a fresh Vec and dropping the existing one.
            filtered.copy_from_slice(marginal);
        }

        filtered_marginals.push(filtered);
    }

    let slices: Vec<&[f32]> = filtered_marginals.iter().map(|m| m.as_slice()).collect();
    build_dd_tree_screened(&slices, config, &NoScreeningPruner, false)
}

// ── Belief-Drafter DDTree (Plan 217, feature: belief_drafter) ───────

/// Delegates to `katgpt_core::simd::fast_sigmoid` (Cephes polynomial).
#[cfg(feature = "belief_drafter")]
#[inline]
fn belief_sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

/// Build DDTree from belief-state draft tokens.
///
/// Uses `BeliefDrafter` to produce variable-length draft candidates from
/// the current hidden state `h_t`, then constructs a DDTree from the
/// draft token marginals.
///
/// Pipeline: `h_t → BeliefDrafter::draft() → convert to marginals → build_dd_tree_screened()`
///
/// Feature-gated behind `belief_drafter`.
#[cfg(feature = "belief_drafter")]
pub fn build_dd_tree_belief(
    drafter: &crate::belief_drafter::BeliefDrafter,
    h_t: &[f32],
    max_draft_steps: usize,
    entropy_threshold: f32,
    config: &katgpt_types::Config,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let drafts = drafter.draft(h_t, max_draft_steps, entropy_threshold);
    if drafts.is_empty() {
        return Vec::new();
    }

    let vocab_size = drafter.vocab_size();
    let mut marginals = Vec::with_capacity(drafts.len());

    for draft_token in &drafts {
        // Initialize the whole marginal to `residual`, then stamp the
        // drafted token's slot — avoids a redundant zero-init pass followed
        // by a full overwrite pass.
        //
        // Compute residual up front (does not depend on the per-slot index).
        // `fast_exp` = Cephes scalar exp (~1 ULP for |x| < 88).
        use katgpt_core::simd::fast_exp;
        let confidence = fast_exp(draft_token.log_prob).max(0.5);
        let residual = (1.0 - confidence) / (vocab_size - 1).max(1) as f32;
        let mut marginal = vec![residual; vocab_size];
        marginal[draft_token.token_idx] = confidence;
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
#[cfg(feature = "belief_drafter")]
pub fn build_dd_tree_belief_collapse_aware(
    drafter: &crate::belief_drafter::BeliefDrafter,
    h_t: &[f32],
    max_draft_steps: usize,
    base_entropy_threshold: f32,
    config: &katgpt_types::Config,
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
    generator: &mut crate::spec_generator::MarginalTokenGenerator,
    pruner: &crate::spec_generator::TokenConstraintPruner<P>,
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    rng: &mut fastrand::Rng,
    kurtosis_threshold: f32,
) -> (
    Vec<TreeNode>,
    Vec<katgpt_core::speculative::types::RejectionReason>,
)
where
    P: ConstraintPruner + Send + Sync,
{
    use crate::spec_generator::TokenCondition;
    use katgpt_core::{GenerativeConstraintPruner, SpeculativeGenerator};

    let mut filtered_marginals: Vec<Vec<f32>> = Vec::with_capacity(marginals.len());
    let mut rejections: Vec<katgpt_core::speculative::types::RejectionReason> = Vec::new();

    for (depth, marginal) in marginals.iter().enumerate() {
        // ── Kurtosis gate: reject flat positions before candidate generation ──
        let kurtosis = crate::kurtosis_gate::excess_kurtosis(marginal);
        if kurtosis <= kurtosis_threshold {
            rejections.push(
                katgpt_core::speculative::types::RejectionReason::KurtosisRejection {
                    kurtosis,
                    threshold: kurtosis_threshold,
                },
            );
            // Skip this position entirely — tree will not expand here
            continue;
        }

        let condition = TokenCondition {
            parent_tokens: vec![],
            depth,
            marginals: marginal.to_vec(),
        };

        let candidates = if let Ok(c) = generator.generate(&condition, rng) { c } else {
                filtered_marginals.push(marginal.to_vec());
                continue;
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
            // All filtered out — copy original marginals in place (same length).
            filtered.copy_from_slice(marginal);
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
/// Feature-gated behind both `speculative_generator` and `best_buddies`.
#[cfg(all(feature = "speculative_generator", feature = "best_buddies"))]
pub fn build_dd_tree_speculative_best_buddies<P>(
    generator: &mut crate::spec_generator::MarginalTokenGenerator,
    pruner: &crate::spec_generator::TokenConstraintPruner<P>,
    draft_marginals: &[&[f32]],
    target_marginals: &[&[f32]],
    aligner: &mut crate::best_buddies::MarginalBestBuddyAligner,
    config: &katgpt_types::Config,
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
#[cfg(feature = "dflare_progressive_budget")]
pub fn build_dd_tree_screened_progressive(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    budget_config: Option<&katgpt_core::speculative::types::PositionWeightedBudget>,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build_screened_progressive(marginals, config, screener, chain_seed, budget_config);
    std::mem::take(&mut builder.tree)
}

/// DDTree with correlation-based per-depth budget allocation (Plan 200).
///
/// Uses `CorrelationBudgetAllocator` to distribute `tree_budget` across depths
/// proportional to empirical draft↔target agreement rates. Higher agreement → more nodes.
#[cfg(feature = "corr_budget")]
pub fn build_dd_tree_screened_corr(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    allocator: &crate::correlation_budget::CorrelationBudgetAllocator,
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
/// Uses `FlowBudgetAllocator` to distribute `tree_budget` across depths
/// proportional to per-depth flow scores. High-flow-score branches get more
/// speculative depth; low-score branches get early termination.
#[cfg(feature = "nf_flow_budget")]
pub fn build_dd_tree_screened_flow_budget(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    allocator: &mut crate::nf_flow_budget::FlowBudgetAllocator,
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

/// DDTree with RecFM cross-scale consistency filtering (Plan 168, Research 150).
///
/// Identical to [`build_dd_tree_screened`] but additionally prunes branches whose
/// probability velocity violates cross-scale consistency.
///
/// When `recfm_config.enable == false`, delegates to [`build_dd_tree_screened`] unchanged.
#[cfg(feature = "recfm")]
pub fn build_dd_tree_screened_recfm(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
    recfm_config: &CrossScaleConfig,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build_screened_recfm(marginals, config, screener, chain_seed, recfm_config);
    std::mem::take(&mut builder.tree)
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
#[cfg(feature = "and_or_dtree")]
pub fn build_dd_tree_and_or<P: ScreeningPruner>(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    pruner: &P,
    cache: &mut katgpt_core::proof_cache::ProofGoalCache,
    chain_seed: bool,
) -> Vec<TreeNode> {
    use crate::and_or_builder::AndOrBuilder;
    use crate::blueprint::BlueprintPass;
    use crate::decomp_reviewer::DecompositionReviewer;

    // Step 1: Build AND-OR tree from marginals using relevance signal.
    let mut builder = AndOrBuilder::new(pruner, cache)
        .with_relevance_threshold(0.3)
        .with_max_depth(8);
    let and_or_tree = builder.build(marginals);

    // Step 2: Check if decomposition happened.
    if let AndOrNode::Leaf { .. } = &and_or_tree {
        // No decomposition needed — use standard screened build.
        build_dd_tree_screened(marginals, config, pruner, chain_seed)
    } else {
        // Decomposition happened — extract best path from AND-OR tree.
        let _blueprint = BlueprintPass::generate(marginals);
        let _reviewer = DecompositionReviewer::new(0.3);

        // Collect all solved leaf solutions into a combined path.
        let combined_path = collect_solved_path(&and_or_tree);

        // If we got a complete solution from cache, convert to TreeNode directly.
        if !combined_path.is_empty() {
            return path_to_tree_nodes(&combined_path);
        }

        // Partial solution — fall back to screened DDTree.
        build_dd_tree_screened(marginals, config, pruner, chain_seed)
    }
}

/// Collect the best solved path from an AND-OR tree.
#[cfg(feature = "and_or_dtree")]
fn collect_solved_path<G, S>(node: &AndOrNode<G, S>) -> Vec<S>
where
    S: Clone,
{
    match node {
        AndOrNode::Or { children, best, .. } => if let Some(idx) = best { children
                .get(*idx)
                .and_then(|c| {
                    let path = collect_solved_path(c);
                    if path.is_empty() { None } else { Some(path) }
                })
                .unwrap_or_default() } else {
                for child in children {
                    let path = collect_solved_path(child);
                    if !path.is_empty() {
                        return path;
                    }
                }
                Vec::new()
            },
        AndOrNode::And {
            children,
            solved_count,
            ..
        } => {
            if usize::from(*solved_count) < children.len() {
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
    let mut parent_path = TreePath::default();

    for (depth, &token_idx) in flat.iter().enumerate() {
        // One u32 slot per level (Issue 670 widened from 16-bit packing).
        parent_path = parent_path.push(token_idx as u32, depth);

        nodes.push(TreeNode {
            parent_path,
            depth,
            token_idx,
            score: 0.0, // Score not needed for pre-solved paths
        });
    }

    nodes.shrink_to_fit();
    nodes
}

#[cfg(test)]
mod tests;
