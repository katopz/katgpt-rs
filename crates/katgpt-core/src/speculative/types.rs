//! Speculative-decoding substrate types (Plan 008 Step 5).
//!
//! Origin: moved verbatim from `katgpt-rs/src/speculative/types.rs` (2026-06-28).
//! The pure-substrate half lives here (data types + configs + algorithms that
//! depend only on `crate::types::Config` + `crate::traits::*` + std). The
//! composition half (`SpeculativeContext` — moved to `katgpt-forward` Plan 393
//! because it composes `ForwardContext` which also lives there;
//! `DDTreeBranchCache` — which needs `katgpt-transformer::{ForwardContext,
//! PagedKVCache, forward_paged}`) and the root-only composition half
//! (`TesConfig` — which needs `BanditStrategy`; `SelfSpecConfig` — which needs
//! `D2fDecodeConfig`) stay in their consumer crates as thin shims that
//! re-export from here.
//!
//! The companion traits (`ConstraintPruner`, `ScreeningPruner`, `DominoPruner`,
//! `NoPruner`, `NoScreeningPruner`, `BinaryScreeningPruner`) live in
//! `crate::traits` (Plan 107 Phase 0); this module only adds the dependent
//! substrate types.
//!
//! No feature gate on the module itself — always-on (like `simd`, `types`,
//! `traits`, `hla`). Individual types are gated by their respective feature
//! flags (forwarded from the consumer crate via `katgpt-core/<feature>`).

#[allow(unused_imports)]
use crate::traits::ScreeningPruner;
#[allow(unused_imports)]
use crate::types::Config;
use std::cmp::Ordering;

// ── EarlyStopGate (Plan 083) ──────────────────────────────────

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

/// Root→leaf token path for [`TreeNode`], one `u32` slot per level.
///
/// Replaces the `u128` 16-bit-per-level packing (Issue 670), which corrupted
/// the ancestor bits for any token id ≥ 65,536 and could never recover them
/// (modern vocabs — Bonsai 248,320 — need 18-bit ids). Slot `k` holds the
/// `k`-th token from the root; a node at `depth` `d` occupies slots `0..=d`
/// and slots beyond the leaf are zero. Same 32-byte footprint as the old
/// `u128`, and `Copy`/`Eq`/`Ord`/`Hash` all derive.
///
/// A bare path does not encode its own length (`[t]` and `[t, 0]` share a
/// representation), so every in-tree key pairs the path with the node's
/// `depth` — that pair is unique per node. Lexicographic slot order matches
/// the old packed-`u128` comparison order for same-length paths.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Debug, Hash)]
pub struct TreePath([u32; TreePath::MAX_TOKENS]);

impl TreePath {
    /// Maximum path length in tokens (one per level; lookahead ≤ 8 — the
    /// same budget the `u128` packing allowed).
    pub const MAX_TOKENS: usize = 8;

    /// Path holding a single root token (a depth-0 node's own path).
    #[inline]
    pub const fn root(token: u32) -> Self {
        let mut p = [0u32; Self::MAX_TOKENS];
        p[0] = token;
        Self(p)
    }

    /// Append `token` as the leaf at level `depth` (the child's depth).
    ///
    /// Mirrors the old `(parent_path << 16) | token_idx`. Slots deeper than
    /// `depth` are cleared so paths stay canonical across `parent` reuse.
    /// Indexing past [`MAX_TOKENS`] panics — loud, where the old `u128`
    /// shift silently corrupted the path.
    #[inline]
    pub const fn push(mut self, token: u32, depth: usize) -> Self {
        self.0[depth] = token;
        let mut i = depth + 1;
        while i < Self::MAX_TOKENS {
            self.0[i] = 0;
            i += 1;
        }
        self
    }

    /// Ancestor path: drop the leaf token (the old `parent_path >> 16`).
    ///
    /// `depth` is the leaf node's depth — the node whose own token is being
    /// removed. Without a stored length the caller must supply it (every
    /// call site has the node in hand).
    #[inline]
    pub const fn parent(mut self, depth: usize) -> Self {
        self.0[depth] = 0;
        self
    }

    /// The `k`-th token from the root (0-based). Mirrors the old
    /// `>> ((num_tokens - 1 - k) * 16) & 0xFFFF` extraction.
    #[inline]
    pub const fn token_at(self, k: usize) -> u32 {
        self.0[k]
    }

    /// Path from a token slice (≤ [`MAX_TOKENS`]); constructor for tests and
    /// examples.
    #[inline]
    pub fn from_tokens(tokens: &[u32]) -> Self {
        assert!(
            tokens.len() <= Self::MAX_TOKENS,
            "TreePath holds at most {} tokens, got {}",
            Self::MAX_TOKENS,
            tokens.len()
        );
        let mut p = Self::default();
        for (k, &t) in tokens.iter().enumerate() {
            p = p.push(t, k);
        }
        p
    }
}

/// DDTree node for Best-First Search.
///
/// Field order: largest alignment first (`usize` ×2, 32-byte `TreePath`) →
/// f32 last.
#[derive(Copy, Clone, PartialEq)]
pub struct TreeNode {
    pub parent_path: TreePath,
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
        self.score.total_cmp(&other.score)
    }
}

// ── Stability Snapshot (Plan 102) ──────────────────────────────

/// Per-step execution stability metrics (Plan 102: TileRT pipeline).
/// Zero overhead when `stability_metrics` feature is disabled.
#[cfg(feature = "stability_metrics")]
#[derive(Debug, Clone, Copy)]
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
    /// Tail-to-median stability: `p50 / p99` clamped to `[0, 1]`.
    /// **1.0 = perfect** (no tail at all), → 0 as the tail grows.
    ///
    /// Was `1.0 - (p99/p50).min(1.0)`, which is **identically 0.0** for every
    /// non-empty input — `compute` documents its input as sorted ascending, so
    /// `p99 >= p50` by construction and the `.min(1.0)` always binds. Three
    /// workloads with CV 0.09–0.12 all printed `stability=0.0000` while the
    /// bench around them claimed production-ready observability
    /// (`.issues/722`). Any 0.0000 in a historical `.benchmarks/` doc is that
    /// constant, not a measurement, and is not comparable with a value
    /// produced after this change.
    pub stability_score: f64,
    /// Total steps recorded
    pub total_steps: usize,
    /// Samples at or above the reported `p99_ns` rank (`n - p99_idx`).
    ///
    /// The quantity that says whether `p99_ns` is a tail or a single
    /// observation. It is **1 for every n <= 100** — with a hundred samples or
    /// fewer there is no rank below the maximum holding 99% of the mass, so
    /// the nearest-rank p99 legitimately IS the max and `stability_score`
    /// is decided by the single worst step. That is the statistic being
    /// unavailable, not a defect; reporting it without saying so was the
    /// defect. 2 at n=101, 3 at n=200, 11 at n=1000.
    pub p99_tail_support: usize,
}

/// 0-indexed **nearest-rank** p99 for a sample of `n`: `ceil(0.99 * n) - 1`.
///
/// The shape this replaced was `floor(n as f64 * 0.99)`, which returns `n - 1`
/// — the **maximum** — for every `n <= 100`, because `floor(n * p) == n - 1`
/// whenever `n * (1 - p) <= 1`. `.min(n - 1)` prevented a panic, not a wrong
/// statistic, and `stability_score` (a p99/p50 ratio either way — see
/// `.issues/722` for the separate defect in its direction) was therefore
/// decided by the single worst step in every run of at most 100 steps. Direction: the naive
/// index is one rank too HIGH, so a `p99 <= budget` assert was stricter than
/// advertised (a false red, not a false green) — but the number is also
/// printed, and a misleading printed number is the input to a promote call.
///
/// Integer arithmetic on purpose. `(100.0_f64 * 0.99).ceil()` is not reliably
/// 99 — 0.99 has no exact binary representation, so a float `ceil` can land
/// back on the max and silently reinstate the defect at exactly the boundary
/// this fixes. `(n * 99).div_ceil(100)` is exact for every n. Widened to `u64`
/// so the multiply cannot overflow a 32-bit `usize` (wasip2 is a live target
/// here).
#[cfg(feature = "stability_metrics")]
#[inline]
fn nearest_rank_p99(n: usize) -> usize {
    debug_assert!(n > 0, "caller must handle the empty sample");
    let rank = (n as u64 * 99).div_ceil(100) as usize;
    rank.saturating_sub(1).min(n - 1)
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
                p99_tail_support: 0,
            };
        }

        let sum: u64 = sorted_latencies_ns.iter().sum();
        let mean = sum as f64 / n as f64;

        // p50 stays at the upper-middle sample: `n / 2` can only land on
        // `n - 1` for n <= 2, so it is not the max-landing shape, and moving
        // the median convention would churn every published p50 for no
        // correctness gain. p99 is a different matter — see nearest_rank_p99.
        let p50_idx = n / 2;
        let p99_idx = nearest_rank_p99(n);
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

        // `p50 / p99`, NOT `1.0 - p99 / p50`. The input is sorted ascending
        // (see this fn's contract), so the p99 rank is never below the p50
        // rank and `p99 >= p50` is an invariant of the caller's own data —
        // which makes `1.0 - (p99/p50).min(1.0)` the constant 0.0 for every
        // non-empty sample, and 0.0 for a SINGLE perfectly repeatable
        // observation too. Do not "simplify" it back: the non-vacuity test
        // `stability_score_orders_distributions_and_is_not_constant` pins the
        // old expression's constancy by name (.issues/722).
        let stability_score = match p99 {
            0 => 1.0, // an all-zero sample is perfectly stable at this resolution
            _ => (p50 as f64 / p99 as f64).clamp(0.0, 1.0),
        };

        Self {
            phase_latencies_ns: [0; 4],
            p50_ns: p50,
            p99_ns: p99,
            mean_ns: mean as u64,
            cv,
            stability_score,
            total_steps: n,
            p99_tail_support: n - p99_idx,
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
            p99_tail_support: 1,
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

impl DraftResult {
    /// Construct a `DraftResult` from marginals and sampled tokens.
    ///
    /// All feature-gated diagnostic fields (`routing_overlap`, `cost_snapshot`,
    /// `stability`) default to `None`. This constructor is the recommended way
    /// for **consumer crates** (e.g. katgpt-rs root) to build a `DraftResult`
    /// because it encapsulates the feature gates inside katgpt-core, where
    /// `#[cfg(feature = "...")]` actually matches the struct definition.
    ///
    /// Consumer crates that gate on their own (e.g. root's `domain_latent`)
    /// can diverge from katgpt-core's feature state due to transitive feature
    /// activation (katgpt-core's `octree_ctc → sense_composition → domain_latent`
    /// is ON by default even when root's `domain_latent` is OFF). Using this
    /// constructor avoids that mismatch (Issue 016).
    pub fn new(marginals: Vec<Vec<f32>>, sampled_tokens: Vec<usize>) -> Self {
        Self {
            marginals,
            sampled_tokens,
            #[cfg(feature = "domain_latent")]
            routing_overlap: None,
            #[cfg(feature = "spec_cost_model")]
            cost_snapshot: None,
            #[cfg(feature = "stability_metrics")]
            stability: None,
        }
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
    /// Set Diffusion — sliding-window set decode with per-step KV commit
    /// (Research 376 Phase 4 T4.1/T4.2). Generalizes DiscreteDiffusion to
    /// arbitrary position-set orderings via the position-offset schedule.
    /// Requires a model trained with the set-causal architecture.
    #[cfg(feature = "set_diffusion")]
    SetDiffusion,
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
        // Set Diffusion is opt-in and requires a set-causal-trained model —
        // never auto-recommended (caller must explicitly opt in). Falls through
        // to the D2F / speculative / AR ladder below.
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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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
            for a in allocation.iter_mut().take(max_depth) {
                if remaining == 0 {
                    break;
                }
                *a += 1;
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
#[derive(Debug, Clone, Copy)]
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
    /// Default implementation delegates to [`is_conflicted`](Self::is_conflicted).
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
#[derive(Debug, Clone, Copy)]
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

// ── Heal Conflict Detector (Issue 133 — post-heal validation) ───────

/// Post-heal conflict detector — validates that a healed state is semantically
/// coherent.
///
/// This is the heal-path analog of LDT's [`ConflictDetector`] (Plan 088).
/// Where `ConflictDetector` checks token candidate sets for satisfiability
/// (signature: `marginals`, `pruned_count`, `total_candidates`), this trait
/// checks **healed runtime state** for semantic consistency.
///
/// The signature is intentionally different from `ConflictDetector`: healed
/// state is a flat `&[f32]` (style_weights for shards, emotion axes for HLA),
/// not token distributions. Forcing the token-specific `ConflictDetector`
/// signature onto heal validation would abuse its parameters (Interface
/// Segregation — the "I" in SOLID).
///
/// # Implementations
///
/// - `ShardConflictDetector` (riir-neuron-db) — checks healed `style_weights`
///   for degenerate blends (NaN, L2 norm range, variance, archetype distance).
/// - `HlaConflictDetector` (riir-games) — checks healed emotion axes for
///   cross-axis impossibilities (anger+calm both > 0.7, etc.).
///
/// # Contract
///
/// All implementations MUST be:
/// - **Zero-allocation** on the check path (`is_heal_conflicted` takes `&[f32]`).
/// - **Modelless** (pure threshold checks, no training/backprop).
/// - **Stack-only** (no heap allocation in the hot path).
///
/// # Feature Gate
///
/// `heal_validation` — DEFAULT-ON (Issue 133 GOAT gate G1–G6 ALL PASS, 2026-07-12).
/// Trait is passive — zero behavior change unless consumers implement it.
/// Both consumer impls pass GOAT: `ShardConflictDetector` (riir-neuron-db, 30ns),
/// `HlaConflictDetector` (riir-games, 2ns), both < 50ns target.
#[cfg(feature = "heal_validation")]
pub trait HealConflictDetector: Send + Sync {
    /// Check if the healed state is semantically conflicted.
    ///
    /// `healed_state` — the healed values as a flat slice:
    /// - For shards: `style_weights` (`STYLE_DIM` values, typically 64).
    /// - For HLA: emotion axes in `EmotionAxis::ALL` order (6 values:
    ///   valence, arousal, desperation, calm, fear, anger).
    ///
    /// Returns `true` when the healed state violates semantic constraints
    /// (degenerate vector, impossible emotion combination, etc.).
    fn is_heal_conflicted(&self, healed_state: &[f32]) -> bool;
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
#[derive(Clone, Copy, Debug)]
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

// ── SimpleTES (Plan 086) substrate types ──────────────────────
// NB: `TesConfig` stays in the consumer crate — it has a root-only dep on
// `BanditStrategy` (from `crate::pruners::bandit`). Only the pure-data
// `TesNode` + the pure-algorithm `TrajectoryCredit` move here.

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
#[derive(Clone, Copy, Debug)]
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
        weighted.sort_by(|a, b| b.1.total_cmp(&a.1));
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
        let scores: Vec<(usize, f32)> = traj_scores.iter().map(|(&id, &s)| (id, s)).collect();
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

// Note: `Config` is part of the documented substrate surface for consumers but
// is not currently named directly inside this file (the substrate types here
// are pure data / algorithm / trait-implementations that do not require a
// Config parameter). The import above is kept `#[allow(unused_imports)]` so
// downstream `use katgpt_core::speculative::types::Config` resolves.

#[cfg(test)]
mod tests {
    use super::*;

    // ── StabilitySnapshot percentile rank ─────────────────────────

    /// The boundary this fix exists for: `floor(n * 0.99)` is `n - 1` — the
    /// MAX — for every `n <= 100`, and n=100 is where the two forms first
    /// disagree. Pinned as a table rather than a spot check because the
    /// interesting behaviour is entirely at the small-n end, where a
    /// latency sample of a few dozen steps actually lives.
    #[cfg(feature = "stability_metrics")]
    #[test]
    fn nearest_rank_p99_is_not_the_max_past_the_boundary() {
        // (n, expected idx, expected tail support = n - idx)
        let table = [
            (1usize, 0usize, 1usize),
            (2, 1, 1),
            (10, 9, 1),
            (50, 49, 1),
            (99, 98, 1),
            // n <= 100: the nearest-rank p99 legitimately IS the max. There
            // is no rank below it holding 99% of the mass. Support 1 is what
            // says so, which is the whole point of publishing the field.
            (100, 98, 2),
            (101, 99, 2),
            (200, 197, 3),
            (1000, 989, 11),
        ];
        for (n, want_idx, want_support) in table {
            let got = nearest_rank_p99(n);
            assert_eq!(got, want_idx, "n={n}: rank index");
            assert_eq!(n - got, want_support, "n={n}: tail support");
            assert!(got < n, "n={n}: index {got} out of range");
        }
        // The defect, stated directly: the old form returns the max at n=100.
        assert_eq!(((100_f64) * 0.99).floor() as usize, 99, "the old shape");
        assert_eq!(nearest_rank_p99(100), 98, "the new one");
    }

    /// T2 of `.issues/722`. NOT "the score is 0.7 here" — the constant this
    /// replaced would have satisfied a spot check at 0.0 just as well. The
    /// assertion that has content is that the metric **orders** distributions,
    /// plus a direct pin on the old expression's constancy so a revert is
    /// caught by name rather than by someone re-noticing three equal zeros.
    #[cfg(feature = "stability_metrics")]
    #[test]
    fn stability_score_orders_distributions_and_is_not_constant() {
        // Three sorted samples of the same length, same median, growing tail.
        let mk = |tail: u64| {
            let mut v: Vec<u64> = vec![1000; 200];
            for (i, x) in v.iter_mut().enumerate().skip(190) {
                *x = 1000 + tail * (i as u64 - 189);
            }
            v
        };
        let tight = StabilitySnapshot::compute(&mk(1));
        let moderate = StabilitySnapshot::compute(&mk(100));
        let wide = StabilitySnapshot::compute(&mk(10_000));

        assert!(
            tight.stability_score > moderate.stability_score,
            "tight {} must beat moderate {}",
            tight.stability_score,
            moderate.stability_score,
        );
        assert!(
            moderate.stability_score > wide.stability_score,
            "moderate {} must beat wide {}",
            moderate.stability_score,
            wide.stability_score,
        );
        // ...and it must actually reach the useful end of the range, not just
        // be monotone somewhere near zero.
        assert!(tight.stability_score > 0.9, "tight = {}", tight.stability_score);
        assert!(wide.stability_score < 0.1, "wide = {}", wide.stability_score);

        // A single perfectly repeatable observation is perfectly stable. The
        // old form scored it 0.0 — maximally UNstable — which is the defect at
        // its clearest.
        assert_eq!(StabilitySnapshot::compute(&[1000]).stability_score, 1.0);
        // Empty keeps its documented 1.0 (the one arm the constant never hit,
        // and the only value any test in the workspace had ever asserted).
        assert_eq!(StabilitySnapshot::compute(&[]).stability_score, 1.0);

        // The old expression, pinned as constant. p99 >= p50 is guaranteed by
        // `compute`'s sorted-input contract, so this holds for every sample
        // the API accepts — it is not a property of these fixtures.
        for s in [&tight, &moderate, &wide] {
            let old = 1.0 - (s.p99_ns as f64 / s.p50_ns as f64).min(1.0);
            assert_eq!(old, 0.0, "the replaced expression is the constant 0.0");
            assert!(s.p99_ns >= s.p50_ns, "sorted input invariant");
        }
    }

    /// End-to-end through `compute`, on the shape the GOAT test uses: 100
    /// distinct ascending samples. `p99_ns` must be the second-largest, not
    /// the largest, and the snapshot must SAY its support is 2.
    #[cfg(feature = "stability_metrics")]
    #[test]
    fn compute_reports_p99_with_its_tail_support() {
        let known: Vec<u64> = (100..200).collect(); // n = 100, max = 199
        let snap = StabilitySnapshot::compute(&known);
        assert_eq!(snap.total_steps, 100);
        assert_eq!(snap.p99_ns, 198, "p99 of 100 samples is the 99th, not the max");
        assert_eq!(snap.p99_tail_support, 2);
        assert!(snap.p99_ns < *known.last().unwrap(), "p99 must not be the max here");

        // Empty and single-sample arms keep their support honest.
        assert_eq!(StabilitySnapshot::compute(&[]).p99_tail_support, 0);
        let one = StabilitySnapshot::compute(&[1000]);
        assert_eq!((one.p99_ns, one.p99_tail_support), (1000, 1));
    }

    // ── TreePath (Issue 670) ──────────────────────────────────────

    /// G1: the slot representation must reproduce the old u128 16-bit
    /// packing's extracted tokens exactly at vocab ≤ 65,536 (lossless regime
    /// of the old encoding).
    #[test]
    fn tree_path_matches_legacy_u128_packing_at_small_vocab() {
        let tokens: [u32; 8] = [5, 65_535, 0, 1, 2, 3, 4, 65_534];
        let path = TreePath::from_tokens(&tokens);
        // Legacy packing: root in the high bits, leaf in the low bits.
        let mut legacy: u128 = 0;
        for &t in tokens.iter() {
            legacy = (legacy << 16) | (t as u128);
        }
        for k in 0..8 {
            let legacy_tok = ((legacy >> ((8 - 1 - k) * 16)) & 0xFFFF) as u32;
            assert_eq!(path.token_at(k), legacy_tok);
        }
    }

    /// G2 (Issue 670): an 8-level path round-trips every id at Bonsai vocab
    /// (248,320 — 18-bit ids), including the max id 248,319. Under the old
    /// 16-bit packing these ids corrupted the ancestor bits irrecoverably.
    #[test]
    fn tree_path_round_trips_bonsai_vocab_ids() {
        const VOCAB: u32 = 248_320;
        let ids = [
            0,
            1,
            65_535,
            65_536, // first id the old packing corrupted
            65_537,
            131_071,
            131_072,
            200_000,
            VOCAB - 1, // 248,319
        ];
        // Fill all 8 levels with a rotating selection of the hard ids.
        let mut tokens = [0u32; TreePath::MAX_TOKENS];
        for (k, slot) in tokens.iter_mut().enumerate() {
            *slot = ids[k % ids.len()];
        }
        let path = TreePath::from_tokens(&tokens);
        for (k, &t) in tokens.iter().enumerate() {
            assert_eq!(path.token_at(k), t, "level {k} lost");
        }
        // push/parent compose to the same identity at any vocab.
        let leaf = TreePath::MAX_TOKENS - 1;
        let ancestor = path.parent(leaf).push(tokens[leaf], leaf);
        assert_eq!(ancestor, path);
    }

    /// `parent` drops the leaf token and nothing else — the ancestor lookup
    /// key (`gdn_tree_verify`, `extract_ddtree_paths`) depends on it.
    #[test]
    fn tree_path_parent_drops_only_the_leaf() {
        let path = TreePath::from_tokens(&[7, 9, 11]);
        let ancestor = path.parent(2);
        assert_eq!(ancestor, TreePath::from_tokens(&[7, 9]));
        assert_eq!(ancestor.token_at(2), 0, "cleared slot beyond the new leaf");
    }

    /// Lexicographic slot order ≡ the old packed-u128 order for same-length
    /// paths (`gdn_tree_verify` sorts by it for deterministic topology).
    #[test]
    fn tree_path_order_matches_legacy_packing_order() {
        let a = TreePath::from_tokens(&[1, 2, 3]);
        let b = TreePath::from_tokens(&[1, 2, 4]);
        let c = TreePath::from_tokens(&[1, 3, 0]);
        assert!(a < b);
        assert!(b < c);
        // Root-only vs extended: `[1]` < `[1, x]` for x > 0 (matches old
        // `1 < (1 << 16) | x`).
        assert!(TreePath::from_tokens(&[1]) < TreePath::from_tokens(&[1, 1]));
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
        #[cfg(not(feature = "kurtosis_gate"))]
        {
            let reasons = [
                RejectionReason::LowProbability,
                RejectionReason::ConstraintViolation,
                RejectionReason::LowRelevance { score: 0.0 },
                RejectionReason::DivergedFromTarget,
            ];
            assert_eq!(reasons.len(), 4);
        }
        #[cfg(feature = "kurtosis_gate")]
        {
            let reasons = [
                RejectionReason::LowProbability,
                RejectionReason::ConstraintViolation,
                RejectionReason::LowRelevance { score: 0.0 },
                RejectionReason::DivergedFromTarget,
                RejectionReason::KurtosisRejection {
                    kurtosis: -1.0,
                    threshold: 0.0,
                },
            ];
            assert_eq!(reasons.len(), 5);
        }
    }

    #[test]
    fn test_decode_strategy_default_is_autoregressive() {
        let s = DecodeStrategy::default();
        assert_eq!(s, DecodeStrategy::Autoregressive);
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

        // Research 376 Phase 4 T4.2: SetDiffusion variant is Copy like the rest.
        #[cfg(feature = "set_diffusion")]
        {
            let s = DecodeStrategy::SetDiffusion;
            let s2 = s;
            let _s3 = s;
            assert_eq!(s, s2);
        }
    }

    // Research 376 Phase 4 T4.2 — SetDiffusion is never auto-recommended.
    // It requires a set-causal-trained model, so the caller MUST explicitly
    // opt in by constructing DecodeStrategy::SetDiffusion directly.
    // recommend() always falls through to the D2F/speculative/AR ladder.
    #[test]
    #[cfg(feature = "set_diffusion")]
    fn test_decode_strategy_set_diffusion_never_recommended() {
        // Even with enough tokens and a draft model, recommend() must NOT
        // return SetDiffusion — it requires explicit opt-in.
        let s = DecodeStrategy::recommend(4, 8, true);
        assert_ne!(s, DecodeStrategy::SetDiffusion);

        let s = DecodeStrategy::recommend(4, 8, false);
        assert_ne!(s, DecodeStrategy::SetDiffusion);

        let s = DecodeStrategy::recommend(16, 4, false);
        assert_ne!(s, DecodeStrategy::SetDiffusion);
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
        use crate::traits::NoScreeningPruner;
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
        use crate::traits::NoScreeningPruner;
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
            let cfg = MarginalFusionConfig::balanced(6);
            assert!(cfg.enabled);
            assert_eq!(cfg.alpha_weights.len(), 2);
            assert_eq!(cfg.condition_layer_ids.len(), 2);
            assert_eq!(cfg.condition_layer_ids[0], vec![0, 1, 2]);
            assert_eq!(cfg.condition_layer_ids[1], vec![3, 4, 5]);
        }

        #[test]
        fn test_marginal_fusion_weights_sum_to_one() {
            let cfg = MarginalFusionConfig::balanced(6);
            let sum: f32 = cfg.alpha_weights.iter().sum();
            assert!((sum - 1.0).abs() < 0.001);
        }

        #[test]
        fn test_marginal_fusion_rejects_bad_weights() {
            let cfg = MarginalFusionConfig {
                alpha_weights: vec![0.5, 0.7], // sums to 1.2
                condition_layer_ids: vec![vec![0], vec![1]],
                enabled: true,
            };
            assert!(cfg.validate().is_err());
        }

        #[test]
        fn test_marginal_fusion_disabled_always_valid() {
            let cfg = MarginalFusionConfig::disabled();
            assert!(cfg.validate().is_ok());
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
            let cfg = enabled_config();
            assert_eq!(cfg.blend_factor(0.9), 1.0);
        }

        #[test]
        fn test_kv_routing_low_confidence() {
            let cfg = enabled_config();
            assert_eq!(cfg.blend_factor(0.2), 0.0);
        }

        #[test]
        fn test_kv_routing_medium_confidence() {
            let cfg = enabled_config();
            // At 0.55 (midpoint of [0.3, 0.8]), expect 0.5 blend.
            assert!((cfg.blend_factor(0.55) - 0.5).abs() < 0.01);
        }

        #[test]
        fn test_kv_routing_disabled_returns_one() {
            let cfg = KvRoutingConfig::default(); // enabled = false
            assert_eq!(cfg.blend_factor(0.0), 1.0);
        }
    }

    #[cfg(feature = "dflare_progressive_budget")]
    mod dflare_progressive_budget {
        use super::*;

        #[test]
        fn test_position_weight_exponential_decay() {
            let cfg = PositionWeightedBudget::default();
            let w0 = cfg.weight(0);
            let w1 = cfg.weight(1);
            let w2 = cfg.weight(2);
            // exp(-d/gamma) should decay.
            assert!(w0 > w1);
            assert!(w1 > w2);
            // gamma = 4.0 default.
            assert!((w0 - 1.0).abs() < 1e-6);
            assert!((w1 - (-(1.0_f32) / 4.0).exp()).abs() < 1e-6);
        }

        #[test]
        fn test_allocate_sums_to_budget() {
            let cfg = PositionWeightedBudget::default();
            let budget = 16;
            let allocation = cfg.allocate(budget, 4);
            let sum: usize = allocation.iter().sum();
            assert_eq!(sum, budget);
            assert_eq!(allocation.len(), 4);
        }

        #[test]
        fn test_allocate_front_loaded() {
            let cfg = PositionWeightedBudget::default();
            let allocation = cfg.allocate(16, 4);
            // Front depths should get more than later depths.
            assert!(allocation[0] >= allocation[3]);
        }

        #[test]
        fn test_allocate_respects_minimum() {
            let cfg = PositionWeightedBudget {
                gamma: 4.0,
                min_budget_per_depth: 2,
                enabled: true,
            };
            let allocation = cfg.allocate(8, 4);
            // Each depth should get at least min_budget_per_depth.
            for &a in &allocation {
                assert!(a >= 2);
            }
        }

        #[test]
        fn test_allocate_empty_depth() {
            let cfg = PositionWeightedBudget::default();
            let allocation = cfg.allocate(16, 0);
            assert!(allocation.is_empty());
        }
    }
}
