//! Issue 917 T3 — EB-Sampler-class entropy-bounded lane wiring.
//!
//! Wires the landed T1 primitive ([`katgpt_core::entropy_bounded_commit`],
//! feature `entropy_bounded_commit`) into the two drafter lanes the issue
//! names, as ADDITIVE variants — the legacy builders and `dflash_*` cores
//! are untouched and bit-identical when the feature is off:
//!
//! 1. **DDTree expansion** — [`build_dd_tree_eb`] mirrors the best-first
//!    `build()` but replaces the fixed per-expansion child count with the EB
//!    prefix: at every expansion (the root seed and each popped node) the
//!    valid children of the next depth are the candidates, and
//!    [`entropy_bounded_commit`] decides HOW MANY enter the heap. Best-first
//!    order, the score definition, and the confidence-gap early exit are the
//!    incumbent's.
//! 2. **DFlash block commit** — [`dflash_block_commit_eb_with`] takes the
//!    drafted block's per-depth marginals (`marginals_flat`, the exact
//!    layout `dflash_predict_with` produces), derives per-position stats via
//!    [`position_stats_into`], and commits the largest depth prefix with
//!    `Σ H − max H ≤ γ`, emitting argmax tokens for committed depths only.
//!    [`dflash_predict_eb_with`] composes draft + commit in one zero-alloc
//!    call.
//!
//! # The per-child marginal entropy (DDTree lane)
//!
//! A tree expansion draws its candidates from ONE categorical row (the next
//! depth's marginal), so a child token has no distribution of its own. The
//! per-child entropy the EB residual consumes is the child's **surprisal**
//! `H(t) = −ln p(t)` — the token's share of the row's cross-entropy and the
//! standard per-candidate error proxy. The EB residual
//! `Σ_{t∈U} H(t) − max_{t∈U} H(t) ≤ γ` then bounds the joint surprisal the
//! committed children carry beyond their head: many children commit together
//! exactly when several tokens are comparably plausible, and the width
//! collapses toward the argmax when the row is peaked — the adaptive width
//! the fixed per-expansion count (all valid children, or the deep-argmax
//! singleton) never had. The no-stall floor is T1's, preserved per expansion:
//! a singleton prefix has residual 0, so any expansion with at least one
//! finite-surprisal valid child commits ≥ 1 child.
//!
//! [`ErrorProxy`] reorders the candidates the same way T1 does — the
//! residual always runs on the surprisals, the proxy only picks the order:
//! `Entropy` = surprisal, `Confidence` = `−p(t)`, `Margin` = `−(p(t) − p₂)`
//! against the row's second-largest probability (the argmax carries the
//! row's true margin, every other child ranks by descending probability).
//!
//! # Fixed-width-k comparator
//!
//! A fixed per-expansion width of `k` is the SAME policy with the budget
//! disabled: `EbCommitConfig { gamma: f32::INFINITY, max_commit: k, .. }` —
//! an infinite γ accepts every candidate the cap allows, in the identical
//! proxy order (the T1 oracle bench uses exactly this form). The lane bench
//! `bench_917_eb_lane_wiring` A/Bs the two at matched caps.
//!
//! # Zero allocation
//!
//! All scratch lives in caller-owned structs ([`EbBuildScratch`],
//! [`EbCommitScratch`]) grown once and reused — the steady state allocates
//! nothing (G4, `entropy_bounded_lane_alloc_check`).

use crate::TreeNode;
use crate::TreePath;
use crate::dd_tree::extract_parent_tokens_into;
use crate::dflash::{DflashCache, DflashCtx, dflash_predict_with};
// Re-exported for lane consumers/tests — the policy config travels with the
// lanes that consume it (re-export, never duplicate — the DRY rule).
pub use katgpt_core::entropy_bounded_commit::{EbCommitConfig, ErrorProxy, PositionStats};
use katgpt_core::entropy_bounded_commit::{
    entropy_bounded_commit, entropy_bounded_commit_stats, position_stats, position_stats_into,
};
use katgpt_core::float_order::cmp_for_max;
use katgpt_core::traits::ConstraintPruner;
use std::collections::BinaryHeap;

/// Caller-owned scratch for [`build_dd_tree_eb_into`] — pre-allocated buffers
/// plus the per-build counters the bench reads. Create once, reuse across
/// builds; `build_dd_tree_eb_into` clears and refills.
#[derive(Default)]
pub struct EbBuildScratch {
    candidates: Vec<u32>,
    entropy: Vec<f32>,
    key: Vec<f32>,
    log_marginals: Vec<Vec<f32>>,
    heap: BinaryHeap<TreeNode>,
    tree: Vec<TreeNode>,
    parent_tokens: Vec<usize>,
    /// Expansions run this build (the root seed counts as one).
    pub expansions: usize,
    /// Children committed across all expansions this build.
    pub children_committed: usize,
    /// Children committed by the LAST expansion (root seed first).
    pub last_expansion_children: usize,
    /// Histogram of per-expansion commit counts — `commit_histogram[k]` is
    /// how many expansions committed exactly `k` children. Cleared per build;
    /// grows only when a larger count first appears (steady-state stable).
    pub commit_histogram: Vec<u64>,
}

impl EbBuildScratch {
    fn reset(&mut self) {
        self.heap.clear();
        self.tree.clear();
        self.expansions = 0;
        self.children_committed = 0;
        self.last_expansion_children = 0;
        self.commit_histogram.fill(0);
    }

    /// `ln(prob)` per token per depth, cached once per build (the TreeBuilder
    /// pattern): the best-first loop pops many nodes at the same depth and
    /// the expansion needs `ln` for both the surprisal keys and the child
    /// scores. Entries for `prob <= 0.0` are `0.0` (never read — those
    /// tokens are skipped before the lookup).
    fn cache_log_marginals(&mut self, marginals: &[&[f32]]) {
        if self.log_marginals.len() < marginals.len() {
            self.log_marginals.resize_with(marginals.len(), Vec::new);
        } else {
            self.log_marginals.truncate(marginals.len());
        }
        for (log_m, &m) in self.log_marginals.iter_mut().zip(marginals) {
            log_m.clear();
            log_m.reserve(m.len());
            for &p in m {
                log_m.push(if p > 0.0 { p.ln() } else { 0.0 });
            }
        }
    }

    /// One EB-gated expansion: gather the valid finite-surprisal children of
    /// `parent` from `row`, let [`entropy_bounded_commit`] pick the prefix,
    /// push exactly those children into the heap. Records the counters.
    /// `log_m[depth]` is read from the per-build cache (disjoint field).
    fn expand_eb_children(
        &mut self,
        row: &[f32],
        depth: usize,
        parent_score: f32,
        parent_path: TreePath,
        pruner: &dyn ConstraintPruner,
        eb: &EbCommitConfig,
    ) {
        let log_m: &[f32] = &self.log_marginals[depth];
        // Per-token stat arrays, indexed by token id — grow once per vocab.
        if self.entropy.len() < row.len() {
            self.entropy.resize(row.len(), 0.0);
            self.key.resize(row.len(), 0.0);
        }
        // Margin proxy needs the row's top-2 up front (the argmax carries
        // the row margin; every other child keys against `p2`).
        let (argmax, p1, p2) = match eb.proxy {
            ErrorProxy::Margin => {
                let mut best = 0usize;
                let (mut top1, mut top2) = (0.0f32, 0.0f32);
                for (i, &p) in row.iter().enumerate() {
                    if p > top1 {
                        top2 = top1;
                        top1 = p;
                        best = i;
                    } else if p > top2 {
                        top2 = p;
                    }
                }
                (best, top1, top2)
            }
            _ => (0, 0.0, 0.0),
        };

        // The parent-token extraction buffer needs one slot per depth.
        if self.parent_tokens.len() < depth {
            self.parent_tokens.resize(depth, 0);
        }
        let n_tokens =
            extract_parent_tokens_into(parent_path, depth, &mut self.parent_tokens).len();
        self.candidates.clear();
        for (i, &prob) in row.iter().enumerate() {
            // `prob > 0.0` also rejects NaN; `is_finite` rejects +inf, whose
            // surprisal would terminate the EB prefix at the head and stall
            // the expansion — a malformed marginal is skipped, not committed.
            if !(prob > 0.0 && prob.is_finite()) {
                continue;
            }
            if !pruner.is_valid(depth, i, &self.parent_tokens[..n_tokens]) {
                continue;
            }
            let surprisal = -log_m[i];
            self.entropy[i] = surprisal;
            self.key[i] = match eb.proxy {
                ErrorProxy::Entropy => surprisal,
                ErrorProxy::Confidence => -prob,
                ErrorProxy::Margin => {
                    if i == argmax {
                        p2 - p1
                    } else {
                        p2 - prob
                    }
                }
            };
            self.candidates.push(i as u32);
        }

        // The T1 policy: sort candidates under the strict (NaN-last, key,
        // index) order and return the largest Σ-surprisal-minus-max prefix
        // within the cap. A singleton's residual is 0, so any expansion with
        // a candidate commits ≥ 1 child (no-stall, per expansion).
        let k = entropy_bounded_commit(
            &mut self.candidates,
            &self.entropy,
            &self.key,
            eb.gamma,
            eb.max_commit,
        );

        for &t in &self.candidates[..k] {
            self.heap.push(TreeNode {
                score: parent_score + log_m[t as usize],
                depth,
                token_idx: t as usize,
                parent_path: parent_path.push(t, depth),
            });
        }
        self.expansions += 1;
        self.children_committed += k;
        self.last_expansion_children = k;
        if self.commit_histogram.len() <= k {
            self.commit_histogram.resize(k + 1, 0);
        }
        self.commit_histogram[k] += 1;
    }
}

/// Best-first DDTree with EB-bounded expansion width (Issue 917 T3).
///
/// Allocating convenience over [`build_dd_tree_eb_into`] — builds a fresh
/// [`EbBuildScratch`] per call. See the module docs for the semantics; the
/// incumbent [`crate::dd_tree::build_dd_tree_pruned`] is untouched.
pub fn build_dd_tree_eb(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    pruner: &dyn ConstraintPruner,
    eb: &EbCommitConfig,
) -> Vec<TreeNode> {
    let mut scratch = EbBuildScratch::default();
    build_dd_tree_eb_into(&mut scratch, marginals, config, pruner, eb);
    std::mem::take(&mut scratch.tree)
}

/// Zero-alloc [`build_dd_tree_eb`] on caller scratch — the steady state
/// reuses every buffer; the returned slice is valid until the next call on
/// the same scratch.
///
/// Chain seeding is deliberately not carried over: the EB variant treats the
/// root seed as an ordinary EB-gated expansion, which is the policy under
/// test. The confidence-gap early exit (Plan 026) is mirrored exactly.
pub fn build_dd_tree_eb_into<'a>(
    scratch: &'a mut EbBuildScratch,
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    pruner: &dyn ConstraintPruner,
    eb: &EbCommitConfig,
) -> &'a [TreeNode] {
    scratch.reset();
    if marginals.is_empty() {
        return &scratch.tree;
    }
    scratch.cache_log_marginals(marginals);

    // Root expansion — EB-gated like every other expansion (depth-0 row,
    // virtual root, no parent tokens).
    scratch.expand_eb_children(marginals[0], 0, 0.0, TreePath::default(), pruner, eb);

    // ── Best-first expansion (the incumbent's loop, EB-gated) ──
    let mut best_score: Option<f32> = None;
    let mut second_best_score: Option<f32> = None;
    let mut consecutive_dominant: usize = 0;
    while scratch.tree.len() < config.tree_budget {
        let Some(best) = scratch.heap.pop() else {
            break;
        };
        scratch.tree.push(best);

        // Confidence-gap early exit (Plan 026: AutoTTS) — verbatim.
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
            && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0) > config.early_exit_gap
        {
            break;
        }

        if best.depth + 1 < marginals.len() {
            let next_depth = best.depth + 1;
            scratch.expand_eb_children(
                marginals[next_depth],
                next_depth,
                best.score,
                best.parent_path,
                pruner,
                eb,
            );
        }
    }
    &scratch.tree
}

/// Caller-owned scratch for the DFlash EB block commit (zero-alloc steady
/// state). Grown once via the commit call, reused across blocks.
#[derive(Default)]
pub struct EbCommitScratch {
    stats: Vec<PositionStats>,
    entropy: Vec<f32>,
    key: Vec<f32>,
    candidates: Vec<u32>,
}

impl EbCommitScratch {
    fn grow(&mut self, steps: usize) {
        if self.stats.len() < steps {
            self.stats.resize(steps, PositionStats::default());
            self.entropy.resize(steps, 0.0);
            self.key.resize(steps, 0.0);
            self.candidates.resize(steps, 0);
        }
    }

    /// The committed depth ids of the last [`dflash_block_commit_eb_with`]
    /// call — the first `k` candidate slots, in ascending depth order after
    /// the emit. Diagnostics/tests: which depths the policy committed.
    pub fn committed_ids(&self, k: usize) -> &[u32] {
        &self.candidates[..k.min(self.candidates.len())]
    }
}

/// DFlash EB block commit — the committed prefix of an already-drafted block.
///
/// `marginals_flat` is the `dflash_predict_with` layout: step `i`'s marginal
/// occupies `[i*vocab .. (i+1)*vocab]`. Per-depth stats come from
/// [`position_stats_into`]; the committed set is the largest
/// `(NaN-last, proxy key, index)`-ordered prefix with `Σ H − max H ≤ γ`
/// within `cfg.max_commit`. On return `tokens_out[..k]` holds the argmax
/// token of every committed depth in ASCENDING DEPTH order (the block stays
/// well-formed for sequential verification — the EB decision picks WHICH
/// depths, never a different order), and `k` is returned. A NaN logits row
/// is a non-distribution: never committed.
///
/// Zero alloc after the first call sizes the scratch.
pub fn dflash_block_commit_eb_with(
    marginals_flat: &[f32],
    steps: usize,
    vocab: usize,
    scratch: &mut EbCommitScratch,
    tokens_out: &mut [usize],
    cfg: &EbCommitConfig,
) -> usize {
    if steps == 0 || vocab == 0 {
        return 0;
    }
    debug_assert!(marginals_flat.len() >= steps * vocab, "flat block shape");
    debug_assert!(tokens_out.len() >= steps, "one token slot per depth");
    scratch.grow(steps);

    let stats = &mut scratch.stats[..steps];
    position_stats_into(&marginals_flat[..steps * vocab], steps, vocab, stats);
    for (i, slot) in scratch.candidates[..steps].iter_mut().enumerate() {
        *slot = i as u32;
    }
    let k = entropy_bounded_commit_stats(
        &mut scratch.candidates[..steps],
        stats,
        &mut scratch.entropy,
        &mut scratch.key,
        cfg,
    );

    // Emit in ascending depth order: the committed prefix of the key order
    // is an arbitrary subset of depths, and the block is verified in depth
    // order. (In-place sort of the k committed ids — no allocation.)
    scratch.candidates[..k].sort_unstable();
    for (slot, &d) in tokens_out.iter_mut().zip(&scratch.candidates[..k]) {
        let row = &marginals_flat[d as usize * vocab..(d as usize + 1) * vocab];
        *slot = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i);
    }
    k
}

/// Allocating convenience over [`dflash_block_commit_eb_with`] — returns the
/// argmax tokens of the committed depths in ascending depth order.
pub fn dflash_block_commit_eb(
    marginals_flat: &[f32],
    steps: usize,
    vocab: usize,
    cfg: &EbCommitConfig,
) -> Vec<usize> {
    let mut scratch = EbCommitScratch::default();
    let mut tokens = vec![0usize; steps];
    let k =
        dflash_block_commit_eb_with(marginals_flat, steps, vocab, &mut scratch, &mut tokens, cfg);
    tokens.truncate(k);
    tokens
}

/// Draft + EB block commit in one call: [`dflash_predict_with`] fills the
/// block, then [`dflash_block_commit_eb_with`] picks the committed prefix.
///
/// Returns `(steps drafted, positions committed)`; `tokens_out[..committed]`
/// holds the argmax tokens in ascending depth order. Zero alloc in steady
/// state (all buffers caller-owned). The marginals path is the incumbent's —
/// no sampling, no RNG; the EB decision is deterministic given the block.
#[allow(clippy::too_many_arguments)]
pub fn dflash_predict_eb_with<Ctx, Cache, Weights, F>(
    ctx: &mut Ctx,
    cache: &mut Cache,
    weights: &Weights,
    forward_fn: F,
    probs_buf: &mut [f32],
    marginals_flat: &mut [f32],
    eb_scratch: &mut EbCommitScratch,
    tokens_out: &mut [usize],
    eb_cfg: &EbCommitConfig,
    draft_config: &katgpt_types::Config,
    token: usize,
    pos: usize,
) -> (usize, usize)
where
    Ctx: DflashCtx<Weights>,
    Cache: DflashCache,
    F: Fn(&mut Ctx, &Weights, &mut Cache, usize, usize, &katgpt_types::Config),
{
    let steps = dflash_predict_with(
        ctx,
        cache,
        weights,
        forward_fn,
        probs_buf,
        marginals_flat,
        draft_config,
        token,
        pos,
    );
    let committed = dflash_block_commit_eb_with(
        marginals_flat,
        steps,
        draft_config.vocab_size,
        eb_scratch,
        tokens_out,
        eb_cfg,
    );
    (steps, committed)
}

/// One [`position_stats`] read, re-exported for lane consumers that want the
/// per-depth scalars the commit decision saw (diagnostics, tests).
#[inline]
pub fn depth_position_stats(logits: &[f32]) -> PositionStats {
    position_stats(logits)
}
