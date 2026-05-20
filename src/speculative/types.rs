use crate::transformer::{
    ForwardContext, MultiLayerKVCache, PagedKVCache, TransformerWeights, forward_paged,
};
use crate::types::Config;
use std::cmp::Ordering;

// ── Constraint Pruner: Neuro-Symbolic Intercept ──────────────────

/// Trait for pruning drafted tokens against deterministic constraints.
///
/// The Deterministic Validator concept: before the target model verifies drafted
/// branches, a rules engine prunes invalid ones. This prevents the DDTree
/// from wasting budget on branches that can never be accepted.
///
/// Without pruner: DDTree explores ALL high-probability tokens.
/// With pruner:    DDTree explores only VALID high-probability tokens.
pub trait ConstraintPruner: Send + Sync {
    /// Check if `token_idx` at the given `depth` is valid, given the
    /// tokens placed at earlier depths in this path.
    ///
    /// `parent_token[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    ///
    /// Returns `false` to prune (reject) this branch.
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}

/// No-op pruner: allows all tokens (original DDTree behavior).
pub struct NoPruner;

impl ConstraintPruner for NoPruner {
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        true
    }
}

// ── Screening Pruner: Absolute Relevance (Plan 021) ─────────────

/// Graded relevance pruner replacing binary valid/invalid with continuous score.
///
/// Distilled from "Screening Is Enough" (arXiv:2604.01178).
/// Returns `R ∈ [0.0, 1.0]` which is blended into log-prob space:
/// - `1.0` = perfect match, no penalty (`ln(1.0) = 0.0`)
/// - `0.5` = mediocre match, soft penalty (`ln(0.5) ≈ -0.69`)
/// - `0.0` = hard rejection / trim (`ln(0.0) = -∞`)
///
/// This subsumes [`ConstraintPruner`] as the special case `R ∈ {0.0, 1.0}`.
/// A blanket impl provides automatic upgrade from any `ConstraintPruner`.
///
/// # Ownership Boundary with ConstraintPruner (Plan 029, Task 7)
///
/// Single parser ownership: `ConstraintPruner` and `ScreeningPruner` make
/// **independent** decisions and must not compete for the same judgment:
///
/// - **`ConstraintPruner`** = hard structural validity (syntax, brackets, keywords).
///   Returns `bool`. Owns the decision: "is this token *syntactically* legal here?"
///
/// - **`ScreeningPruner`** = graded semantic relevance (domain fit, topic match).
///   Returns `f32` in `[0.0, 1.0]`. Owns the decision: "is this token *semantically*
///   relevant to the current domain?"
///
/// - **`BinaryScreeningPruner` adapter** = bridge only, zero additional logic.
///   Converts `ConstraintPruner::is_valid()` → `{0.0, 1.0}` relevance.
///
/// Both may prune the same token for different reasons — that's fine.
/// Both must NOT claim ownership of the same decision type — that's a bug.
///
/// NVIDIA Dynamo's lesson: competing parser layers caused silent malformation.
/// Explicit ownership boundaries prevent this class of error.
pub trait ScreeningPruner: Send + Sync {
    /// Returns the absolute relevance of taking this token given the path.
    ///
    /// `parent_tokens[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

/// Adapter: wraps any [`ConstraintPruner`] as a [`ScreeningPruner`] with binary relevance.
/// - `is_valid() == true` → relevance 1.0 (no penalty)
/// - `is_valid() == false` → relevance 0.0 (hard trim)
///
/// Use this to pass a `ConstraintPruner` where a `ScreeningPruner` is expected.
/// We use an explicit adapter instead of a blanket impl to avoid conflicts
/// with types that implement `ConstraintPruner` but need a custom `ScreeningPruner`.
pub struct BinaryScreeningPruner<P>(pub P);

impl<P: ConstraintPruner + Send + Sync> ScreeningPruner for BinaryScreeningPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.0.is_valid(depth, token_idx, parent_tokens) {
            1.0
        } else {
            0.0
        }
    }
}

/// No-op screener: returns 1.0 for everything (no penalty, no trimming).
pub struct NoScreeningPruner;

impl ScreeningPruner for NoScreeningPruner {
    #[inline]
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        1.0
    }
}

// ── DDTree Node ────────────────────────────────────────────────

/// DDTree node for Best-First Search.
#[derive(Copy, Clone, PartialEq)]
pub struct TreeNode {
    pub score: f32,
    pub depth: usize,
    pub token_idx: usize,
    pub parent_path: u128,
}

impl Eq for TreeNode {}

impl PartialOrd for TreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TreeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Draft Result ────────────────────────────────────────────────

/// Result of autoregressive drafting: marginals + sampled tokens.
pub struct DraftResult {
    pub marginals: Vec<Vec<f32>>,
    pub sampled_tokens: Vec<usize>,
}

// ── Pre-allocated Speculative Context ──────────────────────────

/// Pre-allocated buffers for zero-alloc speculative decoding.
///
/// Create once with `SpeculativeContext::new(config)`, reuse across calls.
/// Call `reset()` between decode steps to clear transient state.
///
/// All hot-path operations borrow from this struct instead of allocating:
/// - `dflash_predict_with` reuses `ctx`, `cache`, `marginals_flat`, `probs_buf`
/// - `build_dd_tree` reuses `TreeBuilder` heap/tree buffers
/// - `sample_residual_distribution_into` reuses `residual_buf`
/// - Leviathan rejection sampling reuses `p_distributions_flat`
pub struct SpeculativeContext {
    /// Pre-allocated forward pass buffers (embedding, attention, MLP, logits).
    pub ctx: ForwardContext,
    /// Pre-allocated KV cache for draft model.
    pub cache: MultiLayerKVCache,
    /// Flat marginals buffer: `[draft_lookahead * vocab_size]`.
    /// Each step's marginal occupies `[step * vocab_size..(step+1) * vocab_size]`.
    pub marginals_flat: Vec<f32>,
    /// Temp probs buffer: `[vocab_size]` for logits→softmax in-place.
    pub probs_buf: Vec<f32>,
    /// Pre-allocated sampled tokens: `[draft_lookahead]`.
    pub sampled_tokens: Vec<usize>,
    /// Pre-allocated accepted tokens: `[draft_lookahead + 1]`.
    pub accepted_buf: Vec<usize>,
    /// Pre-allocated path buffer: `[draft_lookahead + 1]`.
    pub path_buf: Vec<usize>,
    /// Residual distribution scratch: `[vocab_size]` for `sample_residual_distribution_into`.
    pub residual_buf: Vec<f32>,
    /// Flat p-distributions buffer for Leviathan: `[(draft_lookahead + 1) * vocab_size]`.
    pub p_distributions_flat: Vec<f32>,
    /// Number of steps populated in last operation (for slicing).
    pub steps_populated: usize,
    /// SDE noise injection config for DDTree expansion (ELF Plan 079).
    pub sde_config: SdeConfig,
}

impl SpeculativeContext {
    /// Allocate all buffers from config dimensions.
    pub fn new(config: &Config) -> Self {
        let vocab_size = config.vocab_size;
        let draft_lookahead = config.draft_lookahead;

        Self {
            ctx: ForwardContext::new(config),
            cache: MultiLayerKVCache::new(config),
            marginals_flat: vec![0.0f32; draft_lookahead * vocab_size],
            probs_buf: vec![0.0f32; vocab_size],
            sampled_tokens: vec![0usize; draft_lookahead],
            accepted_buf: vec![0usize; draft_lookahead + 1],
            path_buf: vec![0usize; draft_lookahead + 1],
            residual_buf: vec![0.0f32; vocab_size],
            p_distributions_flat: vec![0.0f32; (draft_lookahead + 1) * vocab_size],
            steps_populated: 0,
            sde_config: SdeConfig::default(),
        }
    }

    /// Reset transient state between decode steps.
    /// Clears lengths to zero; buffers retain capacity for reuse.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.steps_populated = 0;
    }

    /// Get marginal slice for a specific step.
    /// Returns empty slice if step is out of range.
    pub fn marginal_slice(&self, step: usize, vocab_size: usize) -> &[f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.marginals_flat.len() && step < self.steps_populated {
            &self.marginals_flat[start..end]
        } else {
            &[]
        }
    }

    /// Get mutable marginal slice for a specific step.
    pub fn marginal_slice_mut(&mut self, step: usize, vocab_size: usize) -> &mut [f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.marginals_flat.len() {
            &mut self.marginals_flat[start..end]
        } else {
            &mut []
        }
    }

    /// Get populated marginals as slice-of-slices (borrowed view).
    /// Returns a Vec of borrowed slices for compatibility with existing APIs.
    /// Prefer [`marginals_into`] for hot paths (zero-alloc).
    pub fn marginals_view(&self, vocab_size: usize) -> Vec<&[f32]> {
        (0..self.steps_populated)
            .map(|step| self.marginal_slice(step, vocab_size))
            .collect()
    }

    /// Zero-alloc marginals view: writes borrowed slices into caller-provided buffer.
    ///
    /// Returns the populated portion of `buf` as `&[&[f32]]`.
    /// `buf` must be at least `steps_populated` long (bounded by `draft_lookahead`, typically ≤64).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut buf: [&[f32]; 64] = [&[]; 64];
    /// let view = sctx.marginals_into(&mut buf, vocab_size);
    /// tree_builder.build(view, config, &NoPruner, false);
    /// ```
    pub fn marginals_into<'s, 'a>(
        &'s self,
        buf: &'a mut [&'s [f32]],
        vocab_size: usize,
    ) -> &'a [&'s [f32]] {
        let count = self.steps_populated.min(buf.len());
        for (i, slot) in buf.iter_mut().enumerate().take(count) {
            *slot = self.marginal_slice(i, vocab_size);
        }
        &buf[..count]
    }

    /// Get populated sampled tokens.
    pub fn sampled_tokens(&self) -> &[usize] {
        &self.sampled_tokens[..self.steps_populated]
    }

    /// Get p-distribution slice for Leviathan step.
    pub fn p_dist_slice(&self, step: usize, vocab_size: usize) -> &[f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.p_distributions_flat.len() {
            &self.p_distributions_flat[start..end]
        } else {
            &[]
        }
    }

    /// Get mutable p-distribution slice for Leviathan step.
    pub fn p_dist_slice_mut(&mut self, step: usize, vocab_size: usize) -> &mut [f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.p_distributions_flat.len() {
            &mut self.p_distributions_flat[start..end]
        } else {
            &mut []
        }
    }
}

// ── DDTree Branch Cache ────────────────────────────────────────

/// Paged KV cache wrapper for DDTree branch exploration.
/// Forks share prefix pages, only new pages allocate after fork point.
///
/// This enables best-first search in DDTree where multiple token branches
/// are explored from a shared prefix — copy-on-write avoids duplicating
/// the entire KV cache for each branch.
pub struct DDTreeBranchCache {
    paged: PagedKVCache,
    branch_count: usize,
    max_branches: usize,
}

impl DDTreeBranchCache {
    /// Create a new branch cache.
    /// `max_branches` determines how many concurrent sequences the paged cache supports.
    pub fn new(config: &Config, max_branches: usize) -> Self {
        Self {
            paged: PagedKVCache::new(config, max_branches),
            branch_count: 1, // sequence 0 = trunk
            max_branches,
        }
    }

    /// Fork from an existing sequence at the given position.
    /// Returns the new sequence index.
    /// Shared prefix pages are NOT copied — copy-on-write semantics.
    pub fn fork_branch(&mut self, from: usize, at_pos: usize) -> usize {
        if self.branch_count >= self.max_branches {
            return from; // budget exhausted, reuse source
        }
        let new_seq = self.paged.fork(from, at_pos);
        self.branch_count += 1;
        new_seq
    }

    /// Forward pass on a specific branch sequence.
    /// Returns logits slice via `forward_paged`.
    pub fn forward_branch<'a>(
        &mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        seq_idx: usize,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        forward_paged(ctx, weights, &mut self.paged, seq_idx, token, pos, config)
    }

    /// Rollback a branch to a given position, freeing exclusive pages.
    ///
    /// Keeps pages covering positions `[0..at_pos)` and truncates the rest.
    /// Pages exclusively owned by this branch (not shared with other branches)
    /// are returned to the free list. Shared prefix pages are preserved.
    pub fn rollback_branch(&mut self, seq_idx: usize, at_pos: usize) {
        self.paged.rollback(seq_idx, at_pos);
    }

    /// Fully discard a branch, freeing all its exclusive pages.
    ///
    /// Rolls back to position 0 and decrements `branch_count` if the branch
    /// is not the trunk (seq 0). The trunk cannot be discarded — use `reset()`
    /// to clear everything.
    pub fn discard_branch(&mut self, seq_idx: usize) {
        self.paged.rollback(seq_idx, 0);
        if seq_idx > 0 && self.branch_count > 1 {
            self.branch_count -= 1;
        }
    }

    /// Reset all branches, freeing pages back to pool.
    pub fn reset(&mut self) {
        self.paged.reset();
        self.branch_count = 1;
    }
}

// ── Draft Event Streaming (Plan 029, Dynamo Lesson 2) ────────────

/// Reason a drafted branch was rejected during verification.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectionReason {
    /// Token probability below acceptance threshold.
    LowProbability,
    /// Constraint pruner rejected this branch.
    ConstraintViolation,
    /// Screening relevance score too low.
    LowRelevance { score: f32 },
    /// Branch diverged from target model's preference.
    DivergedFromTarget,
}

/// Streaming event emitted during speculative decoding steps.
///
/// Generalizes `SolveEvent` (Sudoku-specific) into a domain-agnostic event system
/// for real-time monitoring, REST streaming, and TUI display.
///
/// Inspired by NVIDIA Dynamo's `tool_call_dispatch` side channel —
/// events fire as soon as structurally complete, not when the entire step finishes.
#[derive(Debug, Clone, PartialEq)]
pub enum DraftEvent {
    /// Draft model is proposing candidates at this position.
    Drafting {
        /// Position in the token sequence.
        pos: usize,
        /// Number of candidate branches being explored.
        candidates: usize,
    },
    /// Pruning phase completed — some branches removed.
    Pruned {
        /// Position where pruning occurred.
        pos: usize,
        /// Branches that survived pruning.
        kept: usize,
        /// Branches removed by pruner.
        rejected: usize,
    },
    /// Target model verified accepted tokens.
    Verified {
        /// Position of the accepted span.
        pos: usize,
        /// Number of tokens accepted in this verification.
        accepted: usize,
        /// Whether a bonus token was produced (accepted all + 1).
        bonus: bool,
    },
    /// A specific branch was rejected with a reason.
    BranchRejected {
        /// Position where rejection occurred.
        pos: usize,
        /// Why the branch was rejected.
        reason: RejectionReason,
    },
    /// A complete speculative step finished.
    StepComplete {
        /// Total tokens accepted in this step.
        tokens_accepted: usize,
        /// Wall-clock time for this step in microseconds.
        latency_us: u64,
    },
}

// ── Decode Strategy (Plan 066 Phase 3.1) ──────────────────────

/// Decode strategy for token generation.
///
/// Controls which decoding algorithm the generation loop uses:
/// - **Autoregressive**: one token per forward pass (default, safest)
/// - **Speculative**: draft-then-verify with a draft model (DFlash/DDTree)
/// - **DiscreteDiffusion**: block-parallel denoising (D2F, feature-gated)
///
/// Use [`DecodeStrategy::recommend`] to auto-select based on task characteristics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeStrategy {
    /// Standard autoregressive: one token per step.
    #[default]
    Autoregressive,
    /// Speculative decoding with draft model (DFlash/DDTree).
    Speculative,
    /// Block-parallel discrete diffusion forcing (D2F).
    #[cfg(feature = "dllm")]
    DiscreteDiffusion,
}

impl DecodeStrategy {
    /// Recommend a strategy based on task characteristics.
    ///
    /// Heuristic:
    /// - If `dllm` feature is enabled **and** we have enough tokens to fill a block → D2F
    /// - Else if a draft model is available → Speculative
    /// - Otherwise → Autoregressive
    #[allow(unused_variables)]
    pub fn recommend(block_size: usize, n_tokens: usize, has_draft_model: bool) -> Self {
        #[cfg(feature = "dllm")]
        if n_tokens >= block_size {
            return Self::DiscreteDiffusion;
        }
        if has_draft_model {
            Self::Speculative
        } else {
            Self::Autoregressive
        }
    }
}

// ── SDE Noise Injection (ELF Plan 079) ─────────────────────────

/// SDE noise injection config for DDTree expansion (ELF Alg 6 adaptation).
///
/// ELF shows that injecting small noise during continuous sampling breaks
/// greedy error accumulation and improves quality in few-step regimes.
/// We adapt this to DDTree: at each expansion depth, add Gaussian noise
/// to logits before top-k selection.
///
/// γ=0 is identical to current behavior (safe default).
/// γ>0 increases exploration diversity at potential cost to greedy optimality.
#[derive(Debug, Clone)]
pub struct SdeConfig {
    /// Noise re-injection scale (ELF default: 1.0, our default: 0.0 = disabled).
    pub gamma: f32,
    /// Whether to apply noise only to non-top-1 candidates (preserve best, diversify rest).
    pub preserve_top1: bool,
    /// Minimum logit magnitude for noise application (skip very confident tokens).
    pub confidence_floor: f32,
}

impl Default for SdeConfig {
    fn default() -> Self {
        Self {
            gamma: 0.0, // disabled by default — must prove benefit first
            preserve_top1: false,
            confidence_floor: 0.0,
        }
    }
}

impl SdeConfig {
    /// ELF paper default: γ=1.0.
    pub fn elf_default() -> Self {
        Self {
            gamma: 1.0,
            preserve_top1: false,
            confidence_floor: 0.0,
        }
    }

    /// Check if SDE noise is enabled (γ > 0).
    pub fn is_enabled(&self) -> bool {
        self.gamma > 0.0
    }
}

// ── PFlash Block-Sparse Prefill (Plan 044) ─────────────────────

/// Whether to apply block-sparse prefill compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillMode {
    /// Never compress — use full prompt.
    #[default]
    Off,
    /// Auto: compress when prompt length >= threshold.
    Auto,
    /// Always compress (even short prompts).
    Always,
}

// ── Score Reduction Mode (Research 45, Plan 080) ──────────────

/// Reduction mode for block/pair scoring and compressed attention.
///
/// Controls how dot-product scores are reduced in attention and block scoring.
/// `SoftmaxSum` is standard attention (softmax-weighted value accumulation).
/// `MaxSim` is late-interaction scoring: max per query token, then sum.
///
/// Distilled from erikkaum/maxsim (Research 45). The MaxSim kernel achieves
/// 3-4× speedup over naive by streaming with running max — same principle
/// applies to our PFlash block scoring and TurboQuant/SpectralQuant fused
/// dequantize+scoring paths.
///
/// # Feature flag
/// `maxsim` — Plan 080
///
/// # GOAT proof (Plan 080 T9-T11)
/// MaxSim mode must match uncompressed `maxsim_score` within 1e-3.
/// Latency overhead vs SoftmaxSum mode must be ≤5%.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoreReduction {
    /// Standard attention: softmax-weighted sum (existing behavior).
    #[default]
    SoftmaxSum,
    /// MaxSim: max per query token, then sum over query tokens.
    /// `score = Σ_i max_j dot(q_i, d_j)` — ColBERT/PyLate late-interaction.
    #[cfg(feature = "maxsim")]
    MaxSim,
}

/// Configuration for PFlash block-sparse prefill scoring.
///
/// Controls how prompts are compressed before target model prefill.
/// Inspired by FlashPrefill (arXiv:2506.07317) and PFlash speculative prefill.
#[derive(Debug, Clone)]
pub struct FlashPrefillConfig {
    /// Tokens per block for scoring granularity.
    pub block_size: usize,
    /// Number of initial blocks to always keep (attention sink).
    pub attention_sink: usize,
    /// Number of adjacent blocks to always keep (local window).
    pub window: usize,
    /// Number of final query blocks that get full attention.
    pub last_n_full: usize,
    /// Importance threshold: keep blocks with score >= max_score * alpha.
    pub alpha: f32,
    /// Number of tail blocks to use for importance scoring.
    pub tail_window: usize,
    /// Score reduction mode for block pair scoring.
    /// When `maxsim` feature is disabled, always behaves as SoftmaxSum.
    pub score_reduction: ScoreReduction,
}

impl Default for FlashPrefillConfig {
    fn default() -> Self {
        Self {
            block_size: 32,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.15,
            tail_window: 4,
            score_reduction: ScoreReduction::default(),
        }
    }
}

impl FlashPrefillConfig {
    /// Config for GPU path (Metal). Larger blocks for GPU parallelism.
    pub fn metal() -> Self {
        Self {
            block_size: 64,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.15,
            tail_window: 4,
            score_reduction: ScoreReduction::default(),
        }
    }

    /// Config tuned for long-context compression (keep_ratio <= 0.05).
    pub fn long_context() -> Self {
        Self {
            block_size: 64,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.85,
            tail_window: 8,
            score_reduction: ScoreReduction::default(),
        }
    }

    /// Config for short/medium prompts (keep_ratio 0.1-0.3).
    pub fn short_context() -> Self {
        Self {
            block_size: 32,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.12,
            tail_window: 2,
            score_reduction: ScoreReduction::default(),
        }
    }
}

/// Block importance scores from PFlash scoring.
#[derive(Debug, Clone)]
pub struct BlockScores {
    /// Number of blocks scored.
    pub num_blocks: usize,
    /// Block size used for scoring.
    pub block_size: usize,
    /// Per-block importance scores (num_blocks entries).
    pub scores: Vec<f32>,
    /// Selected block indices after applying rules.
    pub selected: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::Rng;

    #[test]
    fn test_branch_cache_fork_branch() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        // Run trunk forward at pos 0..3
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        for pos in 0..4 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Fork at pos 2 — should return a new sequence index (not source)
        let branch = cache.fork_branch(0, 2);
        assert_ne!(
            branch, 0,
            "first fork should return different seq_idx than source"
        );

        // Fork again — should return another new unique sequence index
        let branch2 = cache.fork_branch(0, 2);
        assert_ne!(
            branch2, 0,
            "second fork should return different seq_idx than source"
        );
        assert_ne!(branch2, branch, "each fork should return unique seq_idx");
    }

    #[test]
    fn test_branch_cache_fork_branch_budget_exhausted() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 2); // max 2 branches

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk at pos 0
        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // First fork succeeds — returns a new seq_idx (not source)
        let b1 = cache.fork_branch(0, 0);
        assert_ne!(b1, 0, "fork should return new seq_idx");

        // Budget exhausted (branch_count == max_branches), returns source
        let b2 = cache.fork_branch(0, 0);
        assert_eq!(b2, 0, "budget exhausted should return source seq_idx");
    }

    #[test]
    fn test_branch_cache_forward_branch_logits() {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut cache = DDTreeBranchCache::new(&config, 4);
        let mut ctx = ForwardContext::new(&config);

        let logits = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);

        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} not finite: {l}");
        }
    }

    #[test]
    fn test_branch_cache_fork_shared_prefix_forward() {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut cache = DDTreeBranchCache::new(&config, 4);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2
        for pos in 0..3 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Fork at pos 2 — returns a new seq_idx (not source)
        let branch = cache.fork_branch(0, 2);
        assert_ne!(branch, 0, "fork should return new seq_idx");

        // Continue branch from pos 2 with different token
        let logits = cache.forward_branch(&mut ctx, &weights, branch, 5, 2, &config);
        assert_eq!(logits.len(), config.vocab_size);

        // All logits should be finite
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "branch logit {i} not finite: {l}");
        }
    }

    #[test]
    fn test_branch_cache_reset() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);
        let _ = cache.fork_branch(0, 0);

        assert_eq!(cache.branch_count, 2);

        cache.reset();

        assert_eq!(
            cache.branch_count, 1,
            "reset should restore branch_count to 1"
        );
    }

    #[test]
    fn test_branch_cache_rollback_branch_allows_forward_after() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2, 3
        for pos in 0..4 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Rollback trunk to pos 2
        cache.rollback_branch(0, 2);

        // Forward should still work after rollback — pages for pos 0..2 are intact
        let logits = cache.forward_branch(&mut ctx, &weights, 0, 5, 2, &config);
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "logit {i} after rollback should be finite: {l}"
            );
        }
    }

    #[test]
    fn test_branch_cache_discard_branch_decrements_count() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk at pos 0
        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // Fork a branch
        let branch = cache.fork_branch(0, 0);
        assert_ne!(branch, 0, "fork should return new seq_idx");
        assert_eq!(cache.branch_count, 2);

        // Forward on branch
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 5, 1, &config);

        // Discard branch
        cache.discard_branch(branch);
        assert_eq!(
            cache.branch_count, 1,
            "discard should decrement branch_count"
        );
    }

    #[test]
    fn test_branch_cache_discard_trunk_does_not_decrement() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // Discard trunk (seq 0) should not decrement branch_count
        cache.discard_branch(0);
        assert_eq!(
            cache.branch_count, 1,
            "discarding trunk should not decrement branch_count"
        );
    }

    #[test]
    fn test_branch_cache_rollback_shared_pages_preserved() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2
        for pos in 0..3 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Capture trunk state at pos 2 before branching
        let _trunk_logits = cache
            .forward_branch(&mut ctx, &weights, 0, 0, 3, &config)
            .to_vec();

        // Fork at pos 1
        let branch = cache.fork_branch(0, 1);
        assert_ne!(branch, 0, "fork should return new seq_idx");

        // Forward on branch to allocate exclusive pages
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 5, 1, &config);
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 7, 2, &config);

        // Rollback branch to pos 1 — shared prefix pages should be preserved
        cache.rollback_branch(branch, 1);

        // Trunk should still work: shared pages (pos 0) were not freed
        let trunk_logits_after = cache
            .forward_branch(&mut ctx, &weights, 0, 1, 3, &config)
            .to_vec();
        for (i, &l) in trunk_logits_after.iter().enumerate() {
            assert!(
                l.is_finite(),
                "trunk logit {i} should be finite after branch rollback: {l}"
            );
        }

        // Forking again from trunk should succeed — shared pages are intact
        let branch2 = cache.fork_branch(0, 1);
        assert_ne!(branch2, 0, "fork after rollback should return new seq_idx");
        let logits2 = cache.forward_branch(&mut ctx, &weights, branch2, 3, 1, &config);
        assert_eq!(logits2.len(), config.vocab_size);
        for (i, &l) in logits2.iter().enumerate() {
            assert!(l.is_finite(), "branch2 logit {i} should be finite: {l}");
        }
    }

    // ── DraftEvent Tests (Plan 029) ─────────────────────────────────

    #[test]
    fn test_draft_event_drafting() {
        let event = DraftEvent::Drafting {
            pos: 0,
            candidates: 5,
        };
        assert!(matches!(
            event,
            DraftEvent::Drafting {
                pos: 0,
                candidates: 5
            }
        ));
    }

    #[test]
    fn test_draft_event_pruned() {
        let event = DraftEvent::Pruned {
            pos: 3,
            kept: 4,
            rejected: 2,
        };
        if let DraftEvent::Pruned { kept, rejected, .. } = event {
            assert_eq!(kept, 4);
            assert_eq!(rejected, 2);
        }
    }

    #[test]
    fn test_draft_event_verified_with_bonus() {
        let event = DraftEvent::Verified {
            pos: 1,
            accepted: 3,
            bonus: true,
        };
        assert!(matches!(event, DraftEvent::Verified { bonus: true, .. }));
    }

    #[test]
    fn test_draft_event_branch_rejected() {
        let event = DraftEvent::BranchRejected {
            pos: 2,
            reason: RejectionReason::LowRelevance { score: 0.15 },
        };
        if let DraftEvent::BranchRejected {
            reason: RejectionReason::LowRelevance { score },
            ..
        } = event
        {
            assert!((score - 0.15).abs() < 1e-6);
        }
    }

    #[test]
    fn test_draft_event_step_complete() {
        let event = DraftEvent::StepComplete {
            tokens_accepted: 5,
            latency_us: 120,
        };
        if let DraftEvent::StepComplete {
            tokens_accepted,
            latency_us,
        } = event
        {
            assert_eq!(tokens_accepted, 5);
            assert_eq!(latency_us, 120);
        }
    }

    #[test]
    fn test_rejection_reason_variants() {
        let reasons = [
            RejectionReason::LowProbability,
            RejectionReason::ConstraintViolation,
            RejectionReason::LowRelevance { score: 0.0 },
            RejectionReason::DivergedFromTarget,
        ];
        assert_eq!(reasons.len(), 4);
    }

    // ── DecodeStrategy Tests (Plan 066 Phase 3.1) ──────────────

    #[test]
    fn test_decode_strategy_default_is_autoregressive() {
        assert_eq!(DecodeStrategy::default(), DecodeStrategy::Autoregressive);
    }

    #[test]
    fn test_decode_strategy_recommend_no_draft_model() {
        // Without draft model and without enough tokens → AR
        let strategy = DecodeStrategy::recommend(8, 4, false);
        assert_eq!(strategy, DecodeStrategy::Autoregressive);
    }

    #[test]
    fn test_decode_strategy_recommend_with_draft_model() {
        // With draft model → Speculative
        let strategy = DecodeStrategy::recommend(8, 4, true);
        assert_eq!(strategy, DecodeStrategy::Speculative);
    }

    #[test]
    #[cfg(feature = "dllm")]
    fn test_decode_strategy_recommend_discrete_diffusion_when_enough_tokens() {
        // With dllm feature and enough tokens → DiscreteDiffusion
        let strategy = DecodeStrategy::recommend(4, 8, false);
        assert_eq!(strategy, DecodeStrategy::DiscreteDiffusion);
    }

    #[test]
    #[cfg(feature = "dllm")]
    fn test_decode_strategy_recommend_discrete_diffusion_over_speculative() {
        // D2F takes priority over speculative when enough tokens
        let strategy = DecodeStrategy::recommend(4, 8, true);
        assert_eq!(strategy, DecodeStrategy::DiscreteDiffusion);
    }

    #[test]
    #[cfg(feature = "dllm")]
    fn test_decode_strategy_recommend_falls_through_when_tokens_less_than_block() {
        // Not enough tokens for a block → falls through to speculative/AR
        let strategy_ar = DecodeStrategy::recommend(16, 8, false);
        assert_eq!(strategy_ar, DecodeStrategy::Autoregressive);

        let strategy_spec = DecodeStrategy::recommend(16, 8, true);
        assert_eq!(strategy_spec, DecodeStrategy::Speculative);
    }

    #[test]
    fn test_decode_strategy_variants_are_copy() {
        let a = DecodeStrategy::Autoregressive;
        let b = a; // Copy, not move
        let _c = a; // Still valid after copy
        assert_eq!(a, b);
    }
}
