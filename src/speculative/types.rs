use crate::transformer::{
    ForwardContext, MultiLayerKVCache, PagedKVCache, TransformerWeights, forward_paged,
};
use crate::types::Config;
use std::cmp::Ordering;

#[cfg(feature = "tes_loop")]
use crate::pruners::bandit::BanditStrategy;

// ── Re-exported from katgpt-core (Plan 107 Phase 0) ─────────────
// ConstraintPruner, ScreeningPruner, and companion structs consolidated
// into katgpt-core/src/traits.rs to eliminate duplication with riir-engine.

pub use katgpt_core::traits::{
    BinaryScreeningPruner, CompletionHorizon, ConstraintPruner, DominoPruner, NoPruner,
    NoScreeningPruner, ScreeningPruner,
};

/// Depth-aware early stopping gate (PTRM Plan 083).
///
/// Wraps any [`ScreeningPruner`] and adds depth-aware pruning: at depth > 0,
/// if the inner pruner's relevance falls below `confidence_threshold`, the branch
/// is pruned (relevance 0.0). At depth 0, always passthrough — we need at least
/// one candidate to start.
///
/// Maps to PTRM's Q-head early stopping: prune trajectories whose cumulative
/// quality decays past a threshold at deeper recursion levels.
///
/// Set `enabled = false` or `confidence_threshold = 0.0` to disable (passthrough).
#[cfg(feature = "elf_sde")]
#[derive(Debug, Clone)]
pub struct EarlyStopGate<P> {
    /// Inner screener to delegate relevance queries to.
    pub inner: P,
    /// Minimum relevance to continue at depth > 0. Default: 0.0 (disabled).
    pub confidence_threshold: f32,
    /// Runtime toggle. Default: true.
    pub enabled: bool,
}

#[cfg(feature = "elf_sde")]
impl<P: ScreeningPruner> ScreeningPruner for EarlyStopGate<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let inner_rel = self.inner.relevance(depth, token_idx, parent_tokens);
        if !self.enabled || self.confidence_threshold <= 0.0 || depth == 0 {
            return inner_rel;
        }
        if inner_rel < self.confidence_threshold {
            0.0
        } else {
            inner_rel
        }
    }
}

#[cfg(feature = "elf_sde")]
impl<P: Default + ScreeningPruner> Default for EarlyStopGate<P> {
    fn default() -> Self {
        Self {
            inner: P::default(),
            confidence_threshold: 0.0,
            enabled: true,
        }
    }
}

// ── DDTree Node ────────────────────────────────────────────────

/// DDTree node for Best-First Search.
///
/// Field order: largest alignment first (u128, usize) → f32 last.
/// Eliminates 4 bytes of padding between `score` and `depth` on 64-bit targets.
#[derive(Copy, Clone, PartialEq)]
pub struct TreeNode {
    pub parent_path: u128,
    pub depth: usize,
    pub token_idx: usize,
    pub score: f32,
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

// ── Execution Stability Metrics (Plan 102) ─────────────────────

/// Per-step execution stability metrics (Plan 102: TileRT pipeline).
/// Zero overhead when `stability_metrics` feature is disabled.
#[cfg(feature = "stability_metrics")]
#[derive(Debug, Clone)]
pub struct StabilitySnapshot {
    /// Per-phase wall time in nanoseconds: [draft, snapshot, verify, accept_reject]
    pub phase_latencies_ns: [u64; 4],
    /// P50 latency across all steps (if accumulated)
    pub p50_ns: u64,
    /// P99 latency across all steps
    pub p99_ns: u64,
    /// Mean latency
    pub mean_ns: u64,
    /// Coefficient of variation (std/mean)
    pub cv: f64,
    /// Stability score: 1.0 - (P99 / P50), 1.0 = perfect
    pub stability_score: f64,
    /// Total steps recorded
    pub total_steps: usize,
}

#[cfg(feature = "stability_metrics")]
impl StabilitySnapshot {
    /// Compute stability statistics from a sorted vector of step latencies.
    /// Input MUST be sorted ascending.
    pub fn compute(sorted_latencies_ns: &[u64]) -> Self {
        let n = sorted_latencies_ns.len();
        if n == 0 {
            return Self {
                phase_latencies_ns: [0; 4],
                p50_ns: 0,
                p99_ns: 0,
                mean_ns: 0,
                cv: 0.0,
                stability_score: 1.0,
                total_steps: 0,
            };
        }

        let sum: u64 = sorted_latencies_ns.iter().sum();
        let mean = sum as f64 / n as f64;

        let p50_idx = n / 2;
        let p99_idx = ((n as f64) * 0.99).floor() as usize;
        let p99_idx = p99_idx.min(n - 1);
        let p50 = sorted_latencies_ns[p50_idx];
        let p99 = sorted_latencies_ns[p99_idx];

        // Variance and CV — uses `mul_add` so LLVM emits a single fused multiply-add
        // per element instead of multiply + add (one rounding, one fewer op).
        let variance = if n > 1 {
            let sum_sq = sorted_latencies_ns
                .iter()
                .map(|&v| {
                    let diff = v as f64 - mean;
                    diff.mul_add(diff, 0.0)
                })
                .sum::<f64>();
            sum_sq / (n as f64)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        let stability_score = if p50 > 0 {
            1.0 - (p99 as f64 / p50 as f64).min(1.0)
        } else {
            1.0
        };

        Self {
            phase_latencies_ns: [0; 4],
            p50_ns: p50,
            p99_ns: p99,
            mean_ns: mean as u64,
            cv,
            stability_score,
            total_steps: n,
        }
    }

    /// Create from individual phase timings (single step).
    pub fn from_phases(draft_ns: u64, snapshot_ns: u64, verify_ns: u64, accept_ns: u64) -> Self {
        let total = draft_ns + snapshot_ns + verify_ns + accept_ns;
        Self {
            phase_latencies_ns: [draft_ns, snapshot_ns, verify_ns, accept_ns],
            p50_ns: total,
            p99_ns: total,
            mean_ns: total,
            cv: 0.0,
            stability_score: 1.0,
            total_steps: 1,
        }
    }
}

// ── Draft Result ────────────────────────────────────────────────

/// Result of autoregressive drafting: marginals + sampled tokens.
pub struct DraftResult {
    pub marginals: Vec<Vec<f32>>,
    pub sampled_tokens: Vec<usize>,
    /// Raven slot routing overlap diagnostic (Plan 096)
    #[cfg(feature = "domain_latent")]
    pub routing_overlap: Option<RoutingOverlapSnapshot>,
    /// Amdahl cost model snapshot (Plan 096)
    #[cfg(feature = "spec_cost_model")]
    pub cost_snapshot: Option<SpecCostSnapshot>,
    /// Execution stability metrics (Plan 102)
    #[cfg(feature = "stability_metrics")]
    pub stability: Option<StabilitySnapshot>,
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
#[repr(u8)]
pub enum RejectionReason {
    /// Token probability below acceptance threshold.
    LowProbability,
    /// Constraint pruner rejected this branch.
    ConstraintViolation,
    /// Screening relevance score too low.
    LowRelevance { score: f32 },
    /// Branch diverged from target model's preference.
    DivergedFromTarget,
    /// Kurtosis gate rejected — draft distribution too flat for speculation (Plan 203b).
    #[cfg(feature = "kurtosis_gate")]
    KurtosisRejection {
        /// Excess kurtosis of draft marginal at this position.
        kurtosis: f32,
        /// Threshold that was not met.
        threshold: f32,
    },
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
#[repr(u8)]
pub enum DecodeStrategy {
    /// Standard autoregressive: one token per step.
    #[default]
    Autoregressive,
    /// Speculative decoding with draft model (DFlash/DDTree).
    Speculative,
    /// Block-parallel discrete diffusion forcing (D2F).
    #[cfg(feature = "dllm")]
    DiscreteDiffusion,
    /// DMax Soft Parallel Decode — hybrid embedding D2F (Plan 109 T6).
    /// Prefer over DiscreteDiffusion when dmax_spd feature is enabled.
    #[cfg(feature = "dmax_spd")]
    DiscreteDiffusionSoft,
    /// D2F drafts → AR verifies (self-speculation / tri-mode).
    #[cfg(feature = "tri_mode")]
    SelfSpeculation,
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
        #[cfg(feature = "tri_mode")]
        if has_draft_model && n_tokens >= block_size {
            return Self::SelfSpeculation;
        }
        // Prefer soft D2F when dmax_spd is enabled (Plan 109 T6).
        // dmax_spd implies dllm, so this check must come before the binary D2F fallback.
        #[cfg(feature = "dmax_spd")]
        if n_tokens >= block_size {
            return Self::DiscreteDiffusionSoft;
        }
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
    /// Minimum logit magnitude for noise application (skip very confident tokens).
    pub confidence_floor: f32,
    /// Whether to apply noise only to non-top-1 candidates (preserve best, diversify rest).
    pub preserve_top1: bool,
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

// ── DFlare Marginal Fusion (Plan 174 T1, feature: dflare_fusion) ──

/// Configuration for marginal fusion: blending marginals from multiple conditioning sources.
///
/// Each conditioning source extracts hidden states from a different set of target layers,
/// producing independent marginals. These are blended with alpha weights to produce
/// the final fused marginals: `fused[k][v] = Σ_i alpha_i * marginals_i[k][v]`.
///
/// This is modelless — no training required. Alpha weights must sum to 1.0.
#[cfg(feature = "dflare_fusion")]
#[derive(Debug, Clone)]
pub struct MarginalFusionConfig {
    /// Per-conditioning-source blend weights. Must sum to 1.0.
    pub alpha_weights: Vec<f32>,
    /// Which target layers to extract per conditioning source.
    /// `condition_layer_ids[i]` = layer indices for source i.
    pub condition_layer_ids: Vec<Vec<usize>>,
    /// Whether marginal fusion is enabled.
    pub enabled: bool,
}

#[cfg(feature = "dflare_fusion")]
impl MarginalFusionConfig {
    /// Create a disabled config (fusion inactive).
    pub fn disabled() -> Self {
        Self {
            alpha_weights: vec![],
            condition_layer_ids: vec![],
            enabled: false,
        }
    }

    /// Create a two-source config with equal weights (0.5/0.5).
    ///
    /// Source 0: early target layers (first third).
    /// Source 1: late target layers (last third).
    pub fn balanced(num_target_layers: usize) -> Self {
        let n = num_target_layers;
        // Need at least 2 layers to split into two non-empty sources.
        if n < 2 {
            return Self::disabled();
        }
        let mid = n / 2;
        Self {
            alpha_weights: vec![0.5, 0.5],
            condition_layer_ids: vec![(0..mid).collect(), (mid..n).collect()],
            enabled: true,
        }
    }

    /// Validate that alpha weights sum to ~1.0 and layer IDs are non-empty.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.alpha_weights.len() != self.condition_layer_ids.len() {
            return Err(format!(
                "alpha_weights len ({}) != condition_layer_ids len ({})",
                self.alpha_weights.len(),
                self.condition_layer_ids.len()
            ));
        }
        let sum: f32 = self.alpha_weights.iter().sum();
        if (sum - 1.0).abs() > 0.01 {
            return Err(format!("alpha weights sum to {sum:.4}, expected 1.0"));
        }
        for (i, ids) in self.condition_layer_ids.iter().enumerate() {
            if ids.is_empty() {
                return Err(format!("condition_layer_ids[{i}] is empty"));
            }
        }
        Ok(())
    }
}

// ── DFlare KV Routing (Plan 174 T2, feature: dflare_kv_routing) ──

/// Configuration for pruner-confidence KV routing.
///
/// Routes between target-conditioned and unconditioned KV based on pruner
/// confidence at each step:
/// - High confidence (> high_threshold): use conditioned KV
/// - Low confidence (< low_threshold): use unconditioned KV
/// - Medium: blend proportional to confidence
#[cfg(feature = "dflare_kv_routing")]
#[derive(Debug, Clone)]
pub struct KvRoutingConfig {
    /// Above this pruner relevance, use fully conditioned KV.
    pub high_confidence_threshold: f32,
    /// Below this pruner relevance, use fully unconditioned KV.
    pub low_confidence_threshold: f32,
    /// Whether KV routing is enabled.
    pub enabled: bool,
}

#[cfg(feature = "dflare_kv_routing")]
impl Default for KvRoutingConfig {
    fn default() -> Self {
        Self {
            high_confidence_threshold: 0.8,
            low_confidence_threshold: 0.3,
            enabled: false,
        }
    }
}

#[cfg(feature = "dflare_kv_routing")]
impl KvRoutingConfig {
    /// Compute blend factor for conditioned vs unconditioned KV.
    ///
    /// Returns 0.0 = fully unconditioned, 1.0 = fully conditioned.
    pub fn blend_factor(&self, pruner_relevance: f32) -> f32 {
        if !self.enabled {
            return 1.0; // default: conditioned
        }
        if pruner_relevance >= self.high_confidence_threshold {
            1.0
        } else if pruner_relevance <= self.low_confidence_threshold {
            0.0
        } else {
            // Linear interpolation in medium range
            let range = self.high_confidence_threshold - self.low_confidence_threshold;
            (pruner_relevance - self.low_confidence_threshold) / range
        }
    }
}

// ── DFlare Position-Weighted Budget (Plan 174 T3, feature: dflare_progressive_budget) ──

/// Configuration for position-weighted DDTree budget allocation.
///
/// Biases DDTree expansion toward early positions using exponential decay:
/// `weight(depth) = exp(-depth / gamma)`. More nodes at early depths,
/// fewer at later depths. Total budget stays the same.
#[cfg(feature = "dflare_progressive_budget")]
#[derive(Debug, Clone)]
pub struct PositionWeightedBudget {
    /// Exponential decay rate. Higher = more front-loaded.
    /// Typical values: 2, 4, 8.
    pub gamma: f32,
    /// Minimum budget per depth level (floor).
    pub min_budget_per_depth: usize,
    /// Whether progressive budget is enabled.
    pub enabled: bool,
}

#[cfg(feature = "dflare_progressive_budget")]
impl Default for PositionWeightedBudget {
    fn default() -> Self {
        Self {
            gamma: 4.0,
            min_budget_per_depth: 1,
            enabled: false,
        }
    }
}

#[cfg(feature = "dflare_progressive_budget")]
impl PositionWeightedBudget {
    /// Compute position weight for a given depth.
    ///
    /// `w(d) = exp(-d / gamma)`, clamped to [0, 1].
    pub fn weight(&self, depth: usize) -> f32 {
        (-(depth as f32) / self.gamma).exp()
    }

    /// Allocate total budget across depths proportional to position weights.
    ///
    /// Returns `Vec<usize>` of length `max_depth` where each element is the
    /// number of tree nodes allocated to that depth. Sum equals `total_budget`.
    /// Each depth gets at least `min_budget_per_depth`.
    pub fn allocate(&self, total_budget: usize, max_depth: usize) -> Vec<usize> {
        if max_depth == 0 {
            return vec![];
        }

        let weights: Vec<f32> = (0..max_depth).map(|d| self.weight(d)).collect();
        let weight_sum: f32 = weights.iter().sum();

        // First pass: allocate proportional, floor to min_budget_per_depth
        let mut allocation: Vec<usize> = weights
            .iter()
            .map(|&w| (w / weight_sum * total_budget as f32).floor() as usize)
            .collect();

        // Enforce minimum
        for a in &mut allocation {
            *a = (*a).max(self.min_budget_per_depth);
        }

        // Adjust to match total_budget exactly
        let current_total: usize = allocation.iter().sum();
        if current_total < total_budget {
            let mut remaining = total_budget - current_total;
            // Distribute remainder to earliest depths (highest weight)
            for i in 0..max_depth {
                if remaining == 0 {
                    break;
                }
                allocation[i] += 1;
                remaining -= 1;
            }
        } else if current_total > total_budget {
            // Trim from latest depths (lowest weight) if we over-allocated
            let mut excess = current_total - total_budget;
            for i in (0..max_depth).rev() {
                if excess == 0 {
                    break;
                }
                let trim = excess.min(allocation[i].saturating_sub(self.min_budget_per_depth));
                allocation[i] -= trim;
                excess -= trim;
            }
        }

        allocation
    }
}

// ── PFlash Block-Sparse Prefill (Plan 044) ─────────────────────

/// Whether to apply block-sparse prefill compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PrefillMode {
    /// Never compress — use full prompt.
    #[default]
    Off,
    /// Auto: compress when prompt length >= threshold.
    Auto,
    /// Always compress (even short prompts).
    Always,
}

/// Controls how the DDTree tree budget adapts per-prompt based on complexity signals.
///
/// When enabled, the compression ratio from PFlash attention scoring (a free byproduct
/// of prefill) scales the tree budget: simple prompts → less search, complex → more.
/// Budget is clamped to [base/2, base*2] regardless of signal.
///
/// # Feature flag
/// `budget_adaptation` — Plan 167
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BudgetAdaptation {
    /// Fixed budget — current behavior, no adaptation.
    #[default]
    Off,
    /// Scale by compression ratio from attention scores.
    /// r ∈ (0,1]: fraction of blocks that pass alpha threshold.
    /// High r → complex → more budget. Low r → simple → less budget.
    Compression,
    /// Scale by first-marginal entropy (placeholder for future).
    Entropy,
    /// Scale by ECHO prediction consistency entropy (Plan 247 T5).
    /// Low inter-branch entropy → confident → contract budget.
    /// High inter-branch entropy → uncertain → expand budget.
    /// Signal is the consistency gate entropy, scaled by its own threshold.
    #[cfg(feature = "echo_env_predictor")]
    EchoConsistency,
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
#[repr(u8)]
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
    /// Number of tail blocks to use for importance scoring.
    pub tail_window: usize,
    /// Importance threshold: keep blocks with score >= max_score * alpha.
    pub alpha: f32,
    /// Score reduction mode for block pair scoring.
    /// When `maxsim` feature is disabled, always behaves as SoftmaxSum.
    pub score_reduction: ScoreReduction,
    /// Budget adaptation mode for per-prompt tree budget scaling.
    pub budget_adaptation: BudgetAdaptation,
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
            budget_adaptation: BudgetAdaptation::default(),
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
            budget_adaptation: BudgetAdaptation::default(),
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
            budget_adaptation: BudgetAdaptation::default(),
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
            budget_adaptation: BudgetAdaptation::default(),
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

// ── LDT Lattice Deduction Transformer (Plan 088) ─────────────
// Feature gate: `lattice_deduction`
//
// Distilled from "Lattice Deduction Transformers" (arXiv:2605.08605).
// LDT (800K params, 15min training on B200) achieves 100% on Sudoku-Extreme
// where frontier LLMs score 0%. Key insight: operate on an interpretable
// lattice so deduction is structurally sound.
//
// All three enhancements are modelless (zero training):
// T1: Asymmetric pruning threshold (θ_elim ≈ 0.111)
// T2: Entropy-based conflict detection for early backtracking
// T3: α-operator for multi-solution supervision (in alpha.rs)

/// LDT-derived asymmetric elimination threshold.
///
/// From w+/w− = 8 in BCE loss: penalize false elimination 8× harder than
/// false retention. The natural threshold: `θ_elim = 1/(1 + w+/w−) = 1/9 ≈ 0.111`.
///
/// Only eliminate candidates when confidence is very high.
#[cfg(feature = "lattice_deduction")]
pub const LDT_THETA_ELIM: f32 = 1.0 / (1.0 + 8.0); // ≈ 0.111

/// Configuration for LDT-style asymmetric pruning (T1).
///
/// When enabled, DDTree expansion uses `theta_elim` instead of the default
/// screening threshold, making the pruner conservative: only eliminate
/// candidates when very confident.
#[cfg(feature = "lattice_deduction")]
#[derive(Debug, Clone)]
pub struct LdtPruneConfig {
    /// Elimination threshold (default: LDT_THETA_ELIM ≈ 0.111).
    pub theta_elim: f32,
    /// Whether to use asymmetric threshold (default: true).
    pub enabled: bool,
}

#[cfg(feature = "lattice_deduction")]
impl Default for LdtPruneConfig {
    fn default() -> Self {
        Self {
            theta_elim: LDT_THETA_ELIM,
            enabled: true,
        }
    }
}

/// LDT-inspired conflict detection for early backtracking (T2).
///
/// LDT uses a separate CLS sigmoid for conflict detection.
/// Our modelless translation: detect conflict via entropy/marginal analysis.
///
/// Returns true when the current search state is likely unsatisfiable,
/// triggering early backtracking instead of continued exploration.
#[cfg(feature = "lattice_deduction")]
pub trait ConflictDetector: Send + Sync {
    /// Check if the current state shows conflict signals.
    ///
    /// - `marginals` — per-depth token probability distributions
    /// - `pruned_count` — how many candidates were eliminated this step
    /// - `total_candidates` — total candidates before pruning
    fn is_conflicted(
        &self,
        marginals: &[&[f32]],
        pruned_count: usize,
        total_candidates: usize,
    ) -> bool;

    /// Check conflict with depth-aware escalation (Plan 170 F4).
    ///
    /// Default implementation delegates to [`is_conflicted`].
    /// Implementors that support depth-adaptive thresholds should
    /// override this method to tighten detection at deeper search depths.
    ///
    /// - `depth` — current search depth (0 = root, higher = more committed)
    fn is_conflicted_at_depth(
        &self,
        marginals: &[&[f32]],
        pruned_count: usize,
        total_candidates: usize,
        depth: usize,
    ) -> bool {
        let _ = depth;
        self.is_conflicted(marginals, pruned_count, total_candidates)
    }
}

/// Entropy-based conflict detector (T2 + Plan 170 F4).
///
/// Flags conflict when:
/// 1. Pruning rate exceeds threshold (too aggressive = likely wrong path)
/// 2. Any position has near-zero entropy (overconfident = probably hallucinating)
///
/// LDT's conflict threshold θ_cls = 0.6 → analogous to 60% max prune rate.
///
/// F4 extension: threshold tightens with depth via `depth_escalation`.
/// At depth 0, uses full `max_prune_rate`. At deeper depths, the effective
/// threshold decreases: `effective = max_prune_rate - depth * depth_escalation`.
/// This mirrors LDT's θ_eval_CLS > θ_train_CLS insight: conflict signals
/// become more trustworthy as the state commits.
#[cfg(feature = "lattice_deduction")]
#[derive(Debug, Clone)]
pub struct EntropyConflictDetector {
    /// Maximum fraction of candidates that can be pruned in one step.
    /// LDT's conflict threshold θ_cls = 0.6 analog.
    pub max_prune_rate: f32,
    /// Minimum entropy per position (below = conflict signal).
    pub entropy_floor: f32,
    /// Rate at which max_prune_rate decreases per search depth (Plan 170 F4).
    /// Default: 0.02 (at depth 10, threshold drops by 0.2).
    /// Effective threshold = max(max_prune_rate - depth * depth_escalation, 0.1).
    pub depth_escalation: f32,
}

#[cfg(feature = "lattice_deduction")]
impl Default for EntropyConflictDetector {
    fn default() -> Self {
        Self {
            max_prune_rate: 0.6,    // LDT θ_cls = 0.6 analog
            entropy_floor: 0.01,    // Near-deterministic = suspicious
            depth_escalation: 0.02, // F4: threshold tightens by 0.02 per depth
        }
    }
}

#[cfg(feature = "lattice_deduction")]
impl ConflictDetector for EntropyConflictDetector {
    fn is_conflicted(
        &self,
        marginals: &[&[f32]],
        pruned_count: usize,
        total_candidates: usize,
    ) -> bool {
        self.is_conflicted_at_depth(marginals, pruned_count, total_candidates, 0)
    }

    fn is_conflicted_at_depth(
        &self,
        marginals: &[&[f32]],
        pruned_count: usize,
        total_candidates: usize,
        depth: usize,
    ) -> bool {
        // Hard conflict: no candidates remain
        if total_candidates == 0 {
            return true;
        }

        // F4: Depth-escalating threshold — tighter at deeper search
        let effective_max = (self.max_prune_rate - depth as f32 * self.depth_escalation)
            .max(0.1)
            .min(self.max_prune_rate);

        // Check pruning rate: too aggressive = likely wrong path
        let prune_rate = pruned_count as f32 / total_candidates as f32;
        if prune_rate > effective_max {
            return true;
        }

        // Check entropy per position: overconfident = probably hallucinating
        for marginal in marginals {
            let entropy = compute_entropy(marginal);
            if entropy < self.entropy_floor && marginal.len() > 1 {
                return true;
            }
        }

        false
    }
}
/// H(p) = -Σ p_i * ln(p_i)
#[cfg(feature = "lattice_deduction")]
fn compute_entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| (-p).mul_add(p.ln(), 0.0))
        .sum()
}

// ── Self-Speculation Config (Plan 089, Tri-Mode) ────────────

/// Config for D2F-drafter self-speculation mode.
/// Wraps D2F decode config + draft width for speculative step.
#[cfg(feature = "tri_mode")]
#[derive(Debug, Clone)]
pub struct SelfSpecConfig {
    /// Number of tokens per D2F draft block (default: 8).
    pub draft_width: usize,
    /// D2F decode parameters.
    pub d2f_config: crate::speculative::d2f::D2fDecodeConfig,
    /// Optional trained sampler for adaptive confidence (Plan 116 T3).
    /// When `Some`, uses per-position features to decide accept/reject instead
    /// of the fixed `tau_conf` threshold in the D2F denoising loop.
    pub sampler: Option<crate::speculative::diffusion_sampler::DiffusionSampler>,
}

#[cfg(feature = "tri_mode")]
impl Default for SelfSpecConfig {
    fn default() -> Self {
        Self {
            draft_width: 8,
            d2f_config: crate::speculative::d2f::D2fDecodeConfig::default(),
            sampler: None,
        }
    }
}

// ── Routing Overlap Diagnostic (Plan 096, Research 59) ───────

/// Diagnostic: Raven slot routing overlap across K+1 tokens.
/// Analogous to Cohere's "expert overlap" metric.
/// Only collected when `domain_latent` feature is active.
#[cfg(feature = "domain_latent")]
#[derive(Clone, Debug, Default)]
pub struct RoutingOverlapSnapshot {
    /// Per-step overlap ratio: shared slots / top_k
    pub step_overlap: Vec<f64>,
    /// Total unique slots across all K+1 tokens
    pub unique_slots: usize,
    /// top_k (slots selected per token)
    pub top_k: usize,
    /// Number of tokens in verification batch
    pub n_tokens: usize,
    /// Raw routing vector from RavenKVCache (num_slots entries).
    /// Non-zero entries indicate active slots — can be fed to anyrag `routed_search()`.
    pub routing_vector: Vec<f32>,
}

// ── Amdahl Cost Model (Plan 096, Research 59) ────────────────

/// Amdahl decomposition of speculative verification cost.
/// T(K+1)/T(1) = f_sparse * unique_ratio + (1-f_sparse)
#[cfg(feature = "spec_cost_model")]
#[derive(Clone, Debug)]
pub struct SpecCostSnapshot {
    /// Fraction of forward pass in sparse MLP operations
    pub f_sparse: f64,
    /// Fraction in fixed costs (attention, norms, sampling, kernel overhead)
    pub f_fixed: f64,
    /// Ratio of unique active neurons across K+1 tokens vs single token
    pub unique_ratio: f64,
    /// Amdahl prediction: f_sparse * unique_ratio + f_fixed
    pub predicted_ratio: f64,
    /// Wall-clock measurement: T(K+1) / T(1) in nanoseconds
    pub actual_ratio: f64,
    /// Draft length K used
    pub k: usize,
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
        let mut reasons = vec![
            RejectionReason::LowProbability,
            RejectionReason::ConstraintViolation,
            RejectionReason::LowRelevance { score: 0.0 },
            RejectionReason::DivergedFromTarget,
        ];
        #[cfg(feature = "kurtosis_gate")]
        reasons.push(RejectionReason::KurtosisRejection {
            kurtosis: -1.0,
            threshold: 0.0,
        });
        #[cfg(not(feature = "kurtosis_gate"))]
        assert_eq!(reasons.len(), 4);
        #[cfg(feature = "kurtosis_gate")]
        assert_eq!(reasons.len(), 5);
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
    #[cfg(feature = "dmax_spd")]
    fn test_decode_strategy_recommend_discrete_diffusion_soft_when_enough_tokens() {
        // With dmax_spd feature and enough tokens → DiscreteDiffusionSoft (Plan 109 T6)
        let strategy = DecodeStrategy::recommend(4, 8, false);
        assert_eq!(strategy, DecodeStrategy::DiscreteDiffusionSoft);
    }

    #[test]
    #[cfg(all(feature = "dmax_spd", not(feature = "tri_mode")))]
    fn test_decode_strategy_recommend_discrete_diffusion_soft_over_speculative() {
        // Soft D2F takes priority over speculative when enough tokens (Plan 109 T6)
        let strategy = DecodeStrategy::recommend(4, 8, true);
        assert_eq!(strategy, DecodeStrategy::DiscreteDiffusionSoft);
    }

    #[test]
    #[cfg(feature = "tri_mode")]
    fn test_decode_strategy_recommend_self_speculation_over_discrete_diffusion() {
        // With tri_mode + draft model + enough tokens → SelfSpeculation wins
        let strategy = DecodeStrategy::recommend(4, 8, true);
        assert_eq!(strategy, DecodeStrategy::SelfSpeculation);
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

        #[cfg(feature = "tri_mode")]
        {
            let s = DecodeStrategy::SelfSpeculation;
            let s2 = s;
            let _s3 = s;
            assert_eq!(s, s2);
        }
    }

    // ── EarlyStopGate Tests (Plan 083) ────────────────────────

    #[cfg(feature = "elf_sde")]
    #[test]
    fn test_early_stop_gate_passthrough_at_depth_zero() {
        // A screener that always returns 0.3 (below threshold)
        struct LowRelevance;
        impl ScreeningPruner for LowRelevance {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
                0.3
            }
        }

        let gate = EarlyStopGate {
            inner: LowRelevance,
            confidence_threshold: 0.5,
            enabled: true,
        };

        // Depth 0: always passthrough regardless of threshold
        let rel = gate.relevance(0, 0, &[]);
        assert!(
            (rel - 0.3).abs() < 1e-6,
            "depth 0 should passthrough inner relevance, got {rel}"
        );
    }

    #[cfg(feature = "elf_sde")]
    #[test]
    fn test_early_stop_gate_prunes_below_threshold() {
        struct VariableRelevance;
        impl ScreeningPruner for VariableRelevance {
            fn relevance(&self, depth: usize, _token_idx: usize, _parent_token: &[usize]) -> f32 {
                // Returns depth as relevance: 0.0, 0.1, 0.2, 0.3, ...
                depth as f32 * 0.1
            }
        }

        let gate = EarlyStopGate {
            inner: VariableRelevance,
            confidence_threshold: 0.25,
            enabled: true,
        };

        // Depth 0: passthrough (relevance 0.0)
        assert!((gate.relevance(0, 0, &[]) - 0.0).abs() < 1e-6);
        // Depth 1: 0.1 < 0.25 → pruned to 0.0
        assert!((gate.relevance(1, 0, &[]) - 0.0).abs() < 1e-6);
        // Depth 2: 0.2 < 0.25 → pruned to 0.0
        assert!((gate.relevance(2, 0, &[]) - 0.0).abs() < 1e-6);
        // Depth 3: 0.3 >= 0.25 → passthrough
        assert!((gate.relevance(3, 0, &[]) - 0.3).abs() < 1e-6);
        // Depth 5: 0.5 >= 0.25 → passthrough
        assert!((gate.relevance(5, 0, &[]) - 0.5).abs() < 1e-6);
    }

    #[cfg(feature = "elf_sde")]
    #[test]
    fn test_early_stop_gate_disabled_passthrough() {
        struct LowRelevance;
        impl ScreeningPruner for LowRelevance {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> f32 {
                0.1
            }
        }

        let gate = EarlyStopGate {
            inner: LowRelevance,
            confidence_threshold: 0.5,
            enabled: false,
        };

        // Even at depth > 0, disabled gate should passthrough
        assert!((gate.relevance(5, 0, &[]) - 0.1).abs() < 1e-6);
    }

    #[cfg(feature = "elf_sde")]
    #[test]
    fn test_early_stop_gate_zero_threshold_passthrough() {
        struct LowRelevance;
        impl ScreeningPruner for LowRelevance {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> f32 {
                0.01
            }
        }

        let gate = EarlyStopGate {
            inner: LowRelevance,
            confidence_threshold: 0.0,
            enabled: true,
        };

        // threshold=0.0 means disabled → passthrough
        assert!((gate.relevance(5, 0, &[]) - 0.01).abs() < 1e-6);
    }

    #[cfg(feature = "elf_sde")]
    #[test]
    fn test_early_stop_gate_default_values() {
        let gate = EarlyStopGate {
            inner: NoScreeningPruner,
            confidence_threshold: 0.0,
            enabled: true,
        };
        assert!(gate.enabled);
        assert!((gate.confidence_threshold - 0.0).abs() < 1e-6);
    }

    #[cfg(feature = "elf_sde")]
    #[test]
    fn test_early_stop_gate_wraps_no_screener() {
        let gate = EarlyStopGate {
            inner: NoScreeningPruner,
            confidence_threshold: 0.5,
            enabled: true,
        };

        // NoScreeningPruner always returns 1.0, which is >= any threshold
        assert!((gate.relevance(0, 0, &[]) - 1.0).abs() < 1e-6);
        assert!((gate.relevance(5, 0, &[]) - 1.0).abs() < 1e-6);
    }
    // ── DFlare Marginal Fusion tests (Plan 174 T1e) ──

    #[cfg(feature = "dflare_fusion")]
    mod dflare_fusion {
        use super::*;

        #[test]
        fn test_marginal_fusion_balanced_config() {
            let config = MarginalFusionConfig::balanced(28);
            assert_eq!(config.alpha_weights, vec![0.5, 0.5]);
            assert_eq!(config.condition_layer_ids.len(), 2);
            assert!(config.validate().is_ok());
        }

        #[test]
        fn test_marginal_fusion_weights_sum_to_one() {
            let config = MarginalFusionConfig::balanced(28);
            let sum: f32 = config.alpha_weights.iter().sum();
            assert!((sum - 1.0).abs() < 0.001, "alpha weights sum to {sum}");
        }

        #[test]
        fn test_marginal_fusion_rejects_bad_weights() {
            let config = MarginalFusionConfig {
                alpha_weights: vec![0.3, 0.3],
                condition_layer_ids: vec![vec![1], vec![2]],
                enabled: true,
            };
            assert!(config.validate().is_err());
        }

        #[test]
        fn test_marginal_fusion_disabled_always_valid() {
            let config = MarginalFusionConfig {
                alpha_weights: vec![],
                condition_layer_ids: vec![],
                enabled: false,
            };
            assert!(config.validate().is_ok());
        }
    }

    #[cfg(feature = "dflare_kv_routing")]
    mod dflare_kv_routing {
        use super::*;

        fn enabled_config() -> KvRoutingConfig {
            KvRoutingConfig {
                high_confidence_threshold: 0.8,
                low_confidence_threshold: 0.3,
                enabled: true,
            }
        }

        #[test]
        fn test_kv_routing_high_confidence() {
            let config = enabled_config();
            let blend = config.blend_factor(0.9);
            assert!(
                (blend - 1.0).abs() < 0.001,
                "high confidence should blend=1.0, got {blend}"
            );
        }

        #[test]
        fn test_kv_routing_low_confidence() {
            let config = enabled_config();
            let blend = config.blend_factor(0.1);
            assert!(
                (blend - 0.0).abs() < 0.001,
                "low confidence should blend=0.0, got {blend}"
            );
        }

        #[test]
        fn test_kv_routing_medium_confidence() {
            let config = enabled_config();
            // 0.3 threshold, 0.8 threshold, range=0.5
            // pruner_relevance=0.5 → (0.5-0.3)/0.5 = 0.4
            let blend = config.blend_factor(0.5);
            assert!(
                (blend - 0.4).abs() < 0.001,
                "medium should blend=0.4, got {blend}"
            );
        }

        #[test]
        fn test_kv_routing_disabled_returns_one() {
            let config = KvRoutingConfig {
                enabled: false,
                ..Default::default()
            };
            let blend = config.blend_factor(0.0);
            assert!(
                (blend - 1.0).abs() < 0.001,
                "disabled should always return 1.0"
            );
        }
    }

    #[cfg(feature = "dflare_progressive_budget")]
    mod dflare_progressive_budget {
        use super::*;

        #[test]
        fn test_position_weight_exponential_decay() {
            let config = PositionWeightedBudget::default();
            let w0 = config.weight(0);
            let w1 = config.weight(1);
            let w2 = config.weight(2);
            assert!(
                (w0 - 1.0).abs() < 0.001,
                "depth 0 weight should be 1.0, got {w0}"
            );
            assert!(w1 < w0, "weight should decrease with depth");
            assert!(w2 < w1, "weight should decrease with depth");
        }

        #[test]
        fn test_allocate_sums_to_budget() {
            let config = PositionWeightedBudget {
                gamma: 4.0,
                min_budget_per_depth: 1,
                enabled: true,
            };
            let alloc = config.allocate(100, 8);
            let total: usize = alloc.iter().sum();
            assert_eq!(total, 100, "allocation should sum to budget");
        }

        #[test]
        fn test_allocate_front_loaded() {
            let config = PositionWeightedBudget {
                gamma: 2.0,
                min_budget_per_depth: 1,
                enabled: true,
            };
            let alloc = config.allocate(100, 8);
            // Early depths should get more nodes than later depths
            assert!(
                alloc[0] > alloc[7],
                "depth 0 ({}) should exceed depth 7 ({})",
                alloc[0],
                alloc[7]
            );
        }

        #[test]
        fn test_allocate_respects_minimum() {
            let config = PositionWeightedBudget {
                gamma: 4.0,
                min_budget_per_depth: 3,
                enabled: true,
            };
            let alloc = config.allocate(100, 8);
            for (i, &a) in alloc.iter().enumerate() {
                assert!(a >= 3, "depth {i} allocation {a} below minimum 3");
            }
        }

        #[test]
        fn test_allocate_empty_depth() {
            let config = PositionWeightedBudget::default();
            let alloc = config.allocate(100, 0);
            assert!(alloc.is_empty());
        }
    }
}

// ── SimpleTES Evaluation-Driven Scaling (Plan 086) ────────────

/// SimpleTES configuration (C, L, K) hyperparameters.
///
/// C = global_width: parallel trajectories (default 32)
/// L = refinement_depth: iterations per trajectory (default 100)
/// K = local_sample_size: candidates per step (default 16)
///
/// Budget = C × L × K total evaluations per solve.
#[cfg(feature = "tes_loop")]
#[derive(Clone, Debug)]
pub struct TesConfig {
    /// C: parallel trajectories.
    pub global_width: usize,
    /// L: iterations per trajectory.
    pub refinement_depth: usize,
    /// K: candidates per step.
    pub local_sample_size: usize,
    /// Bandit strategy for proposal selection (Φ).
    pub bandit_strategy: BanditStrategy,
}

#[cfg(feature = "tes_loop")]
impl Default for TesConfig {
    fn default() -> Self {
        Self {
            global_width: 32,
            refinement_depth: 100,
            local_sample_size: 16,
            bandit_strategy: BanditStrategy::Rpucg {
                gamma: 0.8,
                lambda: 1.0,
            },
        }
    }
}

#[cfg(feature = "tes_loop")]
impl TesConfig {
    /// Total evaluation budget: C × L × K.
    pub fn budget(&self) -> usize {
        self.global_width * self.refinement_depth * self.local_sample_size
    }
}

/// Node in the TES evaluation graph.
///
/// Each node represents a candidate solution with:
/// - Direct evaluation score `score`
/// - Graph-propagated value `propagated_value` (max of own score and children's values)
/// - Visit count for UCB exploration
#[cfg(feature = "tes_loop")]
#[derive(Clone, Debug)]
pub struct TesNode {
    /// The candidate tokens.
    pub solution: Vec<usize>,
    /// Evaluator score r.
    pub score: f32,
    /// Feedback text.
    pub metadata: String,
    /// Parent index for graph propagation.
    pub parent_idx: Option<usize>,
    /// Visit count for RPUCG exploration.
    pub visit_count: usize,
    /// Propagated value: U_i = max(r_i, γ · max_child_U).
    pub propagated_value: f32,
}

#[cfg(feature = "tes_loop")]
impl TesNode {
    /// Create a new node with the given solution and parent reference.
    pub fn new(solution: Vec<usize>, parent_idx: Option<usize>) -> Self {
        Self {
            solution,
            score: 0.0,
            metadata: String::new(),
            parent_idx,
            visit_count: 0,
            propagated_value: 0.0,
        }
    }
}

/// Trajectory-level credit assignment for G-Zero Phase 2 bridge.
///
/// SimpleTES assigns credit by **max trajectory score** to ALL nodes in that
/// trajectory (not per-step reward). This is coarser but more robust to sparse
/// rewards and aligns with the evaluation-driven scaling paradigm.
///
/// # Credit Assignment Rule
///
/// - `weight = 1` for all nodes in the best trajectory
/// - `weight = 0` for all nodes in the worst trajectory
/// - Linear interpolation for intermediate trajectories
///
/// This bridges trajectory-level evaluation (SimpleTES) to per-step credit
/// signals needed for DPO/GRPO training (G-Zero Phase 2).
#[cfg(feature = "tes_loop")]
#[derive(Clone, Debug)]
pub struct TrajectoryCredit {
    /// Number of trajectories (C in SimpleTES notation).
    pub num_trajectories: usize,
    /// Max score observed across all trajectories.
    pub best_score: f32,
    /// Min score observed across all trajectories.
    pub worst_score: f32,
    /// Index of the best trajectory.
    pub best_trajectory_idx: usize,
    /// Index of the worst trajectory.
    pub worst_trajectory_idx: usize,
}

#[cfg(feature = "tes_loop")]
impl TrajectoryCredit {
    /// Compute credit weights from trajectory scores.
    ///
    /// Takes a slice of (trajectory_index, max_score) pairs and returns
    /// normalized credit weights for each trajectory.
    ///
    /// Returns `Vec<f32>` of weights in the same order as input.
    pub fn from_trajectory_scores(scores: &[(usize, f32)]) -> Self {
        if scores.is_empty() {
            return Self {
                num_trajectories: 0,
                best_score: 0.0,
                worst_score: 0.0,
                best_trajectory_idx: 0,
                worst_trajectory_idx: 0,
            };
        }

        // Single-pass: track both best and worst simultaneously (was 2 iterator
        // reductions). Initialize from first element so NaN/edge cases preserve
        // the original `max_by`/`min_by` semantics (first wins on ties).
        let &(_, first_score) = &scores[0];
        let mut best_score = first_score;
        let mut worst_score = first_score;
        let mut best_trajectory_idx = scores[0].0;
        let mut worst_trajectory_idx = scores[0].0;

        for &(idx, score) in &scores[1..] {
            // max_by: replace only on strictly greater (first-max wins on ties).
            if score > best_score {
                best_score = score;
                best_trajectory_idx = idx;
            }
            // min_by: replace only on strictly less (first-min wins on ties).
            if score < worst_score {
                worst_score = score;
                worst_trajectory_idx = idx;
            }
        }

        Self {
            num_trajectories: scores.len(),
            best_score,
            worst_score,
            best_trajectory_idx,
            worst_trajectory_idx,
        }
    }

    /// Compute per-node weight for a given trajectory score.
    ///
    /// SimpleTES rule:
    /// - `w = 1.0` if score == best_score
    /// - `w = 0.0` if score == worst_score
    /// - Linear interpolation otherwise
    pub fn node_weight(&self, score: f32) -> f32 {
        let range = self.best_score - self.worst_score;
        if range.abs() < f32::EPSILON {
            // All trajectories have the same score
            return 1.0;
        }
        ((score - self.worst_score) / range).clamp(0.0, 1.0)
    }

    /// Compute per-node weights for all trajectories.
    ///
    /// Returns `Vec<(trajectory_idx, weight)>` sorted by weight descending.
    pub fn all_weights(&self, scores: &[(usize, f32)]) -> Vec<(usize, f32)> {
        let mut weighted: Vec<(usize, f32)> = scores
            .iter()
            .map(|(idx, score)| (*idx, self.node_weight(*score)))
            .collect();
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        weighted
    }

    /// Assign credit to nodes based on their trajectory membership.
    ///
    /// Takes nodes grouped by trajectory and assigns max-trajectory-score
    /// credit to all nodes in each trajectory. This is the SimpleTES
    /// credit assignment used for G-Zero Phase 2 training signal.
    pub fn assign_credit(nodes: &mut [TesNode], trajectory_ids: &[usize]) -> Self {
        // Group nodes by trajectory and find max score per trajectory
        let mut traj_scores: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();

        for (node_idx, &traj_id) in trajectory_ids.iter().enumerate() {
            let entry = traj_scores.entry(traj_id).or_insert(f32::MIN);
            *entry = entry.max(nodes[node_idx].score);
        }

        // Build scores view without consuming traj_scores — the HashMap stays
        // available for O(1) per-node lookup in the second loop below.
        let scores: Vec<(usize, f32)> = traj_scores
            .iter()
            .map(|(&id, &s)| (id, s))
            .collect();
        let credit = Self::from_trajectory_scores(&scores);

        // Assign propagated credit to each node based on its trajectory's max score.
        // O(1) HashMap lookup per node (was O(N) linear scan via `scores.find`).
        for (node_idx, &traj_id) in trajectory_ids.iter().enumerate() {
            let traj_max = traj_scores.get(&traj_id).copied().unwrap_or(0.0);
            // Weight is the trajectory's normalized credit
            let weight = credit.node_weight(traj_max);
            // Store credit as metadata (don't overwrite propagated_value which is RPUCG)
            nodes[node_idx].metadata = format!("{weight:.4}");
        }

        credit
    }
}
