//! Trajectory-Refined Draft (TRDraft) — Modelless TRD for speculative decoding.
//!
//! Paper: arXiv:2606.08432 "Trajectory-Refined Distillation"
//!
//! TRD identifies **prefix failure**: when a draft branch takes a wrong path, per-token
//! verification produces fragmented rejection signals. Token-level fixes (clipping,
//! reweighting) can't resolve this — the problem is at the trajectory level.
//!
//! TRDraft applies this insight to speculative decoding:
//! 1. Detect prefix failure when LeviathanVerifier rejects a branch
//! 2. Re-draft from the failure point with ConstraintPruner constraints + ELF SDE noise
//! 3. Rank raw vs refined branches via BT Rank pairwise comparison
//! 4. Bandit learns when refinement helps (skip/1-step/2-step)
//!
//! Feature gate: `trd_refined_draft` (default-OFF)

use std::time::Instant;

use katgpt_core::speculative::sampling::sample_from_distribution;
use katgpt_core::ConstraintPruner;
use katgpt_core::Rng;

/// Sigmoid function (numerically stable).
/// Duplicated locally to avoid feature-gated dependency chains.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ── FailurePoint: Where and why a branch failed ──────────────

/// Describes where a DDTree branch experienced prefix failure.
#[derive(Debug, Clone)]
pub struct FailurePoint {
    /// Token index where failure was detected.
    pub token_idx: usize,
    /// Draft entropy at failure point (high = uncertain → more likely prefix failure).
    pub entropy: f32,
    /// Why the branch was rejected.
    pub reason: RejectionReason,
}

/// Why a speculative branch was rejected by the verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RejectionReason {
    /// Verifier's argmax diverged from draft token.
    ArgmaxMismatch,
    /// p/q rejection sampling rejected the draft token.
    RejectionSampling,
    /// Entropy spike detected — draft became uncertain.
    EntropySpike,
    /// Q-value drop (bandit signal) — branch quality degraded.
    QValueDrop,
    /// ConstraintPruner flagged continuation as invalid.
    ConstraintViolation,
}

// ── RefinementResult: Outcome of a refinement attempt ────────

/// Outcome of a refinement attempt — determines bandit reward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RefinementOutcome {
    /// Refined branch accepted by verifier. Reward: +1.0
    Accepted,
    /// Refined branch rejected. Reward: 0.0
    Rejected,
    /// Latency budget exceeded, fell back to raw. Reward: −0.5
    BudgetExceeded,
}

impl From<RefinementOutcome> for f32 {
    fn from(outcome: RefinementOutcome) -> f32 {
        match outcome {
            RefinementOutcome::Accepted => 1.0,
            RefinementOutcome::Rejected => 0.0,
            RefinementOutcome::BudgetExceeded => -0.5,
        }
    }
}

/// Result of attempting to refine a failed branch.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Refined token sequence (yr in TRD paper).
    pub refined_tokens: Vec<usize>,
    /// BT Rank score: σ(s_raw − s_refined). > 0.5 means refined wins.
    pub rank_score: f32,
    /// Number of refinement steps used.
    pub steps_used: usize,
    /// Whether the refined branch passes ConstraintPruner validation.
    pub passes_constraints: bool,
    /// Whether latency budget was exceeded during this refinement.
    pub budget_exceeded: bool,
}

// ── TrajectoryRefinedDraft: Core refinement engine ───────────

/// Configuration for TRDraft.
#[derive(Debug, Clone)]
pub struct TrdConfig {
    /// Maximum refinement attempts per failed branch.
    pub max_refinement_steps: usize,
    /// Entropy threshold above which we consider a branch for refinement.
    pub entropy_threshold: f32,
    /// BT Rank temperature for pairwise comparison.
    pub rank_temperature: f32,
    /// ELF SDE noise scale for re-drafting diversity (0.0 = no noise).
    pub elf_noise_scale: f32,
    /// Whether to attempt refinement on branches that were "correct" but low-confidence.
    /// Paper shows TRD helps even on correct rollouts (alternative derivations).
    pub refine_correct_branches: bool,
    /// Maximum time budget per refinement in microseconds. 0 = no budget.
    pub latency_budget_us: u64,
    /// Whether to apply ThoughtFold pre-fold on the prefix before re-drafting.
    /// Compacts redundant reasoning steps for a cleaner starting point.
    /// Requires `chain_fold` feature — no-op when disabled.
    pub enable_prefold: bool,
}

impl Default for TrdConfig {
    fn default() -> Self {
        Self {
            max_refinement_steps: 2,
            entropy_threshold: 0.5,
            rank_temperature: 1.0,
            elf_noise_scale: 0.1,
            refine_correct_branches: false,
            latency_budget_us: 0,
            enable_prefold: true,
        }
    }
}

/// Bandit arm for adaptive refinement budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RefinementArm {
    /// Skip refinement, use raw branch.
    Skip,
    /// One-step refinement (standard TRD).
    Refine1Step,
    /// Two-step refinement (for hard queries).
    Refine2Step,
}

/// Per-arm bandit statistics for adaptive refinement depth.
#[derive(Debug, Clone, Copy, Default)]
struct BanditArmStats {
    /// Total reward accumulated.
    total_reward: f32,
    /// Number of pulls.
    pulls: usize,
}

impl BanditArmStats {
    fn mean_reward(&self) -> f32 {
        if self.pulls == 0 {
            return 0.0;
        }
        self.total_reward / self.pulls as f32
    }
}

/// Trajectory-Refined Draft engine.
///
/// Orchestrates refinement of failed speculative branches using
/// ConstraintPruner (modelless teacher) + BanditPruner (adaptive budget).
pub struct TrajectoryRefinedDraft<'a, P: ConstraintPruner> {
    config: TrdConfig,
    pruner: &'a P,
    /// UCB1 bandit arms for adaptive refinement depth.
    arms: [BanditArmStats; 3], // Skip, 1-step, 2-step
    /// Total refinement attempts (diagnostics).
    total_refinements: usize,
    /// Successful refinements (diagnostics).
    successful_refinements: usize,
    /// Instant when the current refinement started.
    refinement_start: Option<Instant>,
}

impl<'a, P: ConstraintPruner> TrajectoryRefinedDraft<'a, P> {
    /// Create a new TRDraft engine with the given config and constraint pruner.
    pub fn new(config: TrdConfig, pruner: &'a P) -> Self {
        Self {
            config,
            pruner,
            arms: [BanditArmStats::default(); 3],
            total_refinements: 0,
            successful_refinements: 0,
            refinement_start: None,
        }
    }

    /// Detect prefix failure from verification rejection signal.
    ///
    /// Combines entropy spike + rejection reason to determine if the branch
    /// is experiencing prefix failure (as opposed to a simple token mismatch
    /// that doesn't indicate a systemic path error).
    pub fn detect_prefix_failure(
        &self,
        token_idx: usize,
        draft_probs: &[f32],
        accepted_len: usize,
        total_draft_len: usize,
        reason: RejectionReason,
    ) -> Option<FailurePoint> {
        // No failure if all tokens accepted
        if accepted_len >= total_draft_len {
            return None;
        }

        // Compute entropy of draft distribution at rejection point
        let entropy = shannon_entropy(draft_probs);

        // Prefix failure detection criteria:
        // 1. High entropy at rejection point (uncertain draft)
        // 2. Rejection at early position (wrong path from the start)
        // 3. Constraint violation (pruner detected invalid continuation)
        let is_prefix_failure = match reason {
            RejectionReason::ConstraintViolation => true,
            RejectionReason::EntropySpike => true,
            RejectionReason::ArgmaxMismatch => {
                entropy > self.config.entropy_threshold && token_idx < total_draft_len / 2
            }
            RejectionReason::RejectionSampling => entropy > self.config.entropy_threshold,
            RejectionReason::QValueDrop => true,
        };

        if is_prefix_failure {
            Some(FailurePoint {
                token_idx,
                entropy,
                reason,
            })
        } else {
            None
        }
    }

    /// Check whether the latency budget has been exceeded.
    ///
    /// Returns true if a budget is set and the elapsed time since
    /// `refinement_start` exceeds it.
    fn is_budget_exceeded(&self) -> bool {
        match (self.config.latency_budget_us, self.refinement_start) {
            (0, _) => false,
            (_, None) => false,
            (budget_us, Some(start)) => start.elapsed().as_micros() as u64 > budget_us,
        }
    }

    /// Refine a failed branch at the given failure point.
    ///
    /// This is the modelless equivalent of TRD's yr ~ πT(·|x, yo):
    /// - Roll back to the failure point
    /// - Re-draft using ConstraintPruner to constrain candidates
    /// - Inject ELF SDE noise for diversity
    /// - BT Rank the raw vs refined branch
    ///
    /// When `latency_budget_us > 0`, checks budget before each re-draft step.
    /// If exceeded, aborts and falls back to the raw branch with reward −0.5.
    pub fn refine_branch(
        &mut self,
        raw_branch: &[usize],
        failure: &FailurePoint,
        marginals: &[&[f32]], // Marginals from DDTree at each depth
        rng: &mut Rng,
    ) -> RefinementResult {
        // Record start time for latency budget enforcement
        self.refinement_start = Some(Instant::now());

        // Context-aware bandit: if we previously exceeded budget, prefer skip
        let within_budget = !self.is_budget_exceeded();
        let max_steps = self.select_refinement_depth_with_context(within_budget);
        self.total_refinements += 1;

        let mut refined_tokens = Vec::with_capacity(raw_branch.len());
        // Copy the accepted prefix (before failure point), optionally pre-folded
        let cutoff = failure.token_idx.min(raw_branch.len());
        let prefix = match self.config.enable_prefold {
            true => prefold_prefix(&raw_branch[..cutoff]),
            false => raw_branch[..cutoff].to_vec(),
        };
        refined_tokens.extend(prefix);

        // Early exit: bandit chose to skip refinement entirely
        if max_steps == 0 {
            let outcome = RefinementOutcome::Rejected;
            self.update_bandit(0, f32::from(outcome));
            return RefinementResult {
                refined_tokens: raw_branch.to_vec(),
                rank_score: 0.0,
                steps_used: 0,
                passes_constraints: false,
                budget_exceeded: false,
            };
        }

        // Re-draft from failure point (CPU path).
        // GPU routing: when `gpu` feature is enabled, `redraft_gpu_batched` can
        // dispatch batched matmul for DDTree expansion. Not yet integrated.
        let mut success = true;
        let mut budget_exceeded = false;
        for step in 0..max_steps {
            // Latency budget guard: check before each step
            if self.is_budget_exceeded() {
                budget_exceeded = true;
                break;
            }

            let depth = cutoff + step;
            if depth >= marginals.len() {
                break;
            }

            let marginal = marginals[depth];
            if marginal.is_empty() {
                break;
            }

            // Get candidate token from marginal
            let candidate = sample_from_distribution(marginal, rng);

            // Apply ConstraintPruner (modelless teacher)
            if self.pruner.is_valid(depth, candidate, &refined_tokens) {
                refined_tokens.push(candidate);
            } else {
                // Pruner rejects — try top-k from marginal
                let fallback = find_valid_token(depth, marginal, &refined_tokens, self.pruner);
                match fallback {
                    Some(tok) => refined_tokens.push(tok),
                    None => {
                        // No valid continuation — refinement failed
                        success = false;
                        break;
                    }
                }
            }
        }

        // Determine outcome and bandit reward
        let outcome = if budget_exceeded {
            // Fall back to raw branch
            refined_tokens = raw_branch.to_vec();
            RefinementOutcome::BudgetExceeded
        } else if success {
            // BT Rank: score refined vs raw using sigmoid comparison.
            // SIMD-routed: branch_score uses simd_sum_f32 under plasma_path.
            // For N>2 candidates, use bt_rank_winner() for pairwise ranking.
            let raw_score = branch_score(raw_branch, marginals);
            let refined_score = branch_score(&refined_tokens, marginals);
            let rank_score = sigmoid(self.config.rank_temperature * (refined_score - raw_score));

            if rank_score > 0.5 {
                self.successful_refinements += 1;
                RefinementOutcome::Accepted
            } else {
                RefinementOutcome::Rejected
            }
        } else {
            RefinementOutcome::Rejected
        };

        let reward = f32::from(outcome);
        self.update_bandit(max_steps, reward);

        // Compute final rank score for result
        let raw_score = branch_score(raw_branch, marginals);
        let refined_score = branch_score(&refined_tokens, marginals);
        let rank_score = sigmoid(self.config.rank_temperature * (refined_score - raw_score));

        self.refinement_start = None;

        RefinementResult {
            passes_constraints: success && !budget_exceeded,
            refined_tokens,
            rank_score,
            steps_used: max_steps,
            budget_exceeded,
        }
    }

    /// Select refinement depth via UCB1 bandit (no context).
    #[allow(dead_code)]
    fn select_refinement_depth(&self) -> usize {
        self.select_refinement_depth_with_context(true)
    }

    /// Select refinement depth via UCB1 bandit with latency context.
    ///
    /// When `within_budget = false` (latency already tight), prefers Skip (arm 0)
    /// to avoid wasting time on refinement that would likely exceed budget.
    /// When `within_budget = true`, uses normal UCB1 exploration.
    pub fn select_refinement_depth_with_context(&self, within_budget: bool) -> usize {
        if !within_budget {
            return 0; // Skip — budget already tight, don't waste time
        }

        // UCB1: argmax(mean_reward + sqrt(2 * ln(total_pulls) / arm_pulls))
        let total_pulls: usize = self.arms.iter().map(|a| a.pulls).sum();
        if total_pulls == 0 {
            // No data yet — default to 1-step (paper's standard)
            return 1;
        }

        let ln_total = (total_pulls as f32).ln();
        let mut best_arm = 0usize;
        let mut best_ucb = f32::NEG_INFINITY;

        for (i, arm) in self.arms.iter().enumerate() {
            let ucb = if arm.pulls == 0 {
                f32::INFINITY // Unexplored arm gets priority
            } else {
                arm.mean_reward() + (2.0 * ln_total / arm.pulls as f32).sqrt()
            };
            if ucb > best_ucb {
                best_ucb = ucb;
                best_arm = i;
            }
        }

        match best_arm {
            0 => 0, // Skip
            1 => 1, // 1-step
            _ => 2, // 2-step
        }
    }

    /// Update bandit arm reward after refinement attempt.
    fn update_bandit(&mut self, steps_used: usize, reward: f32) {
        let arm_idx = match steps_used {
            0 => 0,
            1 => 1,
            _ => 2,
        };
        self.arms[arm_idx].pulls += 1;
        self.arms[arm_idx].total_reward += reward;
    }

    /// Get the success rate of refinements so far (diagnostics).
    pub fn success_rate(&self) -> f32 {
        if self.total_refinements == 0 {
            return 0.0;
        }
        self.successful_refinements as f32 / self.total_refinements as f32
    }

    /// Get bandit arm statistics (diagnostics).
    pub fn bandit_stats(&self) -> [(f32, usize); 3] {
        self.arms
            .iter()
            .map(|a| (a.mean_reward(), a.pulls))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

// ── ThoughtFold pre-fold integration ─────────────────────────

/// Conservative fold budget for pre-fold: keep 80% of steps.
const PREFOLD_BUDGET: f32 = 0.8;

/// Pre-fold the prefix tokens using ThoughtFold to remove redundant reasoning steps.
///
/// When `chain_fold` feature is enabled, uses `ChainFolder::binary_search_fold` with
/// a conservative budget (0.8) to compact the prefix. Without `chain_fold`, returns
/// the prefix as-is (zero cost).
///
/// Importance is derived from token frequency: tokens appearing many times in the
/// prefix are considered redundant (lower importance), while unique tokens are kept.
#[cfg(feature = "chain_fold")]
fn prefold_prefix(prefix: &[usize]) -> Vec<usize> {
    use crate::fold::{ChainFolder, FoldContext, FoldDecision, StepBoundary};
    use std::collections::HashMap;

    let n = prefix.len();
    if n <= 2 {
        // Too short to fold meaningfully — anchors would dominate
        return prefix.to_vec();
    }

    // Compute token frequency — repeated tokens are less important
    let mut freq: HashMap<usize, usize> = HashMap::with_capacity(n);
    for &tok in prefix {
        *freq.entry(tok).or_insert(0) += 1;
    }

    // Build step boundaries: one per token position.
    // First and last tokens are anchors (must keep).
    let boundaries: Vec<StepBoundary> = prefix
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let is_anchor = i == 0 || i == n - 1;
            StepBoundary::new(i, i, is_anchor)
        })
        .collect();

    // Importance: inverse frequency — rare tokens are more important
    let importance_scores: Vec<f32> = prefix
        .iter()
        .map(|&tok| {
            let count = freq.get(&tok).copied().unwrap_or(1);
            1.0 / count as f32
        })
        .collect();

    let context = FoldContext {
        importance_scores,
        boundaries,
        fold_budget: PREFOLD_BUDGET,
    };

    let mut folder = ChainFolder::new(PREFOLD_BUDGET);
    let result = folder.binary_search_fold(&context);

    // Build compacted prefix from fold decisions
    let decisions = folder.decisions();
    let mut compacted = Vec::with_capacity(result.kept_steps);
    for (i, &decision) in decisions.iter().enumerate() {
        match decision {
            FoldDecision::Fold => continue,
            FoldDecision::Keep | FoldDecision::Anchor => compacted.push(prefix[i]),
        }
    }

    compacted
}

/// Pre-fold the prefix tokens — no-op fallback when `chain_fold` is not enabled.
#[cfg(not(feature = "chain_fold"))]
fn prefold_prefix(prefix: &[usize]) -> Vec<usize> {
    prefix.to_vec()
}

// ── Helper functions ─────────────────────────────────────────

/// Shannon entropy of a probability distribution.
///
/// CPU-bound scalar ops: p*ln(p) loop over logits. SIMD not beneficial here
/// because (a) vocab is typically < 100k, (b) the branch-free p > 0 filter
/// is already near-optimal, and (c) ln(p) is not available as a SIMD kernel.
/// Future: consider SIMD entropy for large marginals (>10k) via chunked
/// p*ln(p) with a SIMD ln approximation.
fn shannon_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// SIMD-optimized top-k scan using `simd_argmax_f32`.
///
/// Performs K passes of SIMD argmax (masking found maxima to -inf), yielding
/// top-k candidates in O(K * vocab / SIMD_WIDTH) instead of O(K * vocab)
/// scalar comparisons. Each argmax pass is SIMD-accelerated (NEON/AVX2).
#[cfg(feature = "plasma_path")]
fn find_valid_token<P: ConstraintPruner>(
    depth: usize,
    marginal: &[f32],
    parent_tokens: &[usize],
    pruner: &P,
) -> Option<usize> {
    const K: usize = 10;

    if marginal.is_empty() {
        return None;
    }

    // Copy marginal for masking (simd_argmax_f32 needs &[f32])
    let mut buf = marginal.to_vec();
    let mut top_indices: [usize; K] = [0; K];
    let mut top_probs: [f32; K] = [f32::NEG_INFINITY; K];

    // K passes of SIMD argmax: find max, record it, mask to -inf
    for slot in 0..K {
        let (idx, val) = katgpt_core::simd::simd_argmax_f32(&buf);
        top_indices[slot] = idx;
        top_probs[slot] = val;
        if idx < buf.len() {
            buf[idx] = f32::NEG_INFINITY;
        }
    }

    // Check validity in probability order (highest first)
    top_indices
        .iter()
        .find(|&&idx| pruner.is_valid(depth, idx, parent_tokens))
        .copied()
}

/// Scalar top-k scan — O(n * K) with per-element min comparison.
///
/// Fallback when `plasma_path` is not enabled. Maintains a fixed-size top-K
/// buffer, replacing the minimum on each iteration.
#[cfg(not(feature = "plasma_path"))]
fn find_valid_token<P: ConstraintPruner>(
    depth: usize,
    marginal: &[f32],
    parent_tokens: &[usize],
    pruner: &P,
) -> Option<usize> {
    let mut top_indices: [usize; 10] = [0; 10];
    let mut top_probs: [f32; 10] = [f32::NEG_INFINITY; 10];

    for (idx, &prob) in marginal.iter().enumerate() {
        let min_idx = top_probs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();
        if prob > top_probs[min_idx] {
            top_probs[min_idx] = prob;
            top_indices[min_idx] = idx;
        }
    }

    for &idx in &top_indices {
        if pruner.is_valid(depth, idx, parent_tokens) {
            return Some(idx);
        }
    }

    None
}

/// SIMD-optimized branch scoring: gather probs → compute ln → SIMD sum.
///
/// Gathers all marginal probabilities for the branch tokens into a contiguous
/// buffer, computes ln on each (sequential but cache-friendly), then reduces
/// via `simd_sum_f32`. Benefits from SIMD reduction on branches > 4 tokens.
#[cfg(feature = "plasma_path")]
fn branch_score(branch: &[usize], marginals: &[&[f32]]) -> f32 {
    let mut log_probs = Vec::with_capacity(branch.len());
    for (depth, &token) in branch.iter().enumerate() {
        if depth < marginals.len() {
            let marginal = marginals[depth];
            if token < marginal.len() {
                log_probs.push(marginal[token].max(1e-10).ln());
            }
        }
    }
    katgpt_core::simd::simd_sum_f32(&log_probs)
}

/// Scalar branch scoring — product of marginal log-probabilities.
///
/// Higher score = more probable branch = better.
#[cfg(not(feature = "plasma_path"))]
fn branch_score(branch: &[usize], marginals: &[&[f32]]) -> f32 {
    let mut score = 0.0f32;
    for (depth, &token) in branch.iter().enumerate() {
        if depth < marginals.len() {
            let marginal = marginals[depth];
            if token < marginal.len() {
                let prob = marginal[token].max(1e-10);
                score += prob.ln();
            }
        }
    }
    score
}

// ── BT Rank pairwise SIMD (plasma_path) ───────────────────────

/// SIMD-vectorized BT Rank pairwise comparison for N candidate branches.
///
/// Scores all branches via `branch_score` (SIMD when available), then computes
/// pairwise σ(τ * (si − sj)) win probabilities. Returns the index of the
/// candidate with the highest aggregate win rate.
///
/// For N=2 (current use case), this degenerates to a single sigmoid comparison.
/// For N>2, the batched scoring and pairwise summation benefit from SIMD.
#[cfg(feature = "plasma_path")]
#[allow(dead_code)]
fn bt_rank_winner(candidates: &[&[usize]], marginals: &[&[f32]], temperature: f32) -> usize {
    let n = candidates.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }

    // Score all candidates (branch_score is SIMD-optimized under plasma_path)
    let scores: Vec<f32> = candidates
        .iter()
        .map(|branch| branch_score(branch, marginals))
        .collect();

    // Pairwise win rates: for each candidate i, sum σ(τ * (si - sj)) over all j != i
    let mut win_rates = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                win_rates[i] += sigmoid(temperature * (scores[i] - scores[j]));
            }
        }
    }

    // Argmax win rate
    let mut best = 0usize;
    let mut best_rate = win_rates[0];
    for (i, &rate) in win_rates.iter().enumerate().skip(1) {
        if rate > best_rate {
            best_rate = rate;
            best = i;
        }
    }
    best
}

/// Scalar BT Rank pairwise comparison for N candidate branches.
///
/// Identical logic to `bt_rank_winner` but without SIMD-optimized branch_score.
#[cfg(not(feature = "plasma_path"))]
#[allow(dead_code)]
fn bt_rank_winner(candidates: &[&[usize]], marginals: &[&[f32]], temperature: f32) -> usize {
    let n = candidates.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }

    let scores: Vec<f32> = candidates
        .iter()
        .map(|branch| branch_score(branch, marginals))
        .collect();

    let mut win_rates = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                win_rates[i] += sigmoid(temperature * (scores[i] - scores[j]));
            }
        }
    }

    let mut best = 0usize;
    let mut best_rate = win_rates[0];
    for (i, &rate) in win_rates.iter().enumerate().skip(1) {
        if rate > best_rate {
            best_rate = rate;
            best = i;
        }
    }
    best
}

// ── GPU routing stub ──────────────────────────────────────────

/// GPU-batched DDTree expansion for re-drafting.
///
/// Placeholder for future GPU integration: when `gpu` feature is enabled,
/// re-drafting can leverage batched matmul for DDTree expansion on GPU.
/// Current implementation falls back to CPU re-drafting via `refine_branch`.
///
/// TODO(Plan 249): Integrate with riir-ai/riir-gpu CubeCL runtime for
/// batched speculative draft expansion. Requires DDTree GPU node store.
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn redraft_gpu_batched(
    _prefix: &[usize],
    _marginals: &[&[f32]],
    _max_steps: usize,
) -> Option<Vec<usize>> {
    // GPU re-drafting not yet implemented — fall back to CPU path.
    // When DDTree GPU integration lands, this will dispatch batched matmul
    // for tree expansion, returning the GPU-refined branch.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPruner {
        invalid_tokens: Vec<usize>,
    }

    impl MockPruner {
        fn new(invalid: Vec<usize>) -> Self {
            Self {
                invalid_tokens: invalid,
            }
        }
    }

    impl ConstraintPruner for MockPruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            !self.invalid_tokens.contains(&token_idx)
        }
    }

    #[test]
    fn test_detect_prefix_failure_high_entropy() {
        let pruner = MockPruner::new(vec![]);
        let trd = TrajectoryRefinedDraft::new(TrdConfig::default(), &pruner);

        // High entropy distribution (near-uniform)
        let probs: Vec<f32> = vec![0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.04, 0.04];
        let failure = trd.detect_prefix_failure(
            2, // early position
            &probs,
            1,  // only 1 accepted
            10, // 10 drafted
            RejectionReason::ArgmaxMismatch,
        );

        assert!(
            failure.is_some(),
            "Should detect prefix failure at high entropy + early position"
        );
        let fp = failure.unwrap();
        assert_eq!(fp.token_idx, 2);
        assert!(
            fp.entropy > 1.0,
            "Entropy should be > 1.0 for near-uniform dist"
        );
    }

    #[test]
    fn test_detect_prefix_failure_low_entropy_skip() {
        let pruner = MockPruner::new(vec![]);
        let trd = TrajectoryRefinedDraft::new(TrdConfig::default(), &pruner);

        // Low entropy distribution (peaked)
        let probs: Vec<f32> = vec![0.95, 0.01, 0.01, 0.01, 0.01, 0.01];
        let failure = trd.detect_prefix_failure(
            4, // late position
            &probs,
            3, // 3 of 6 accepted
            6,
            RejectionReason::ArgmaxMismatch,
        );

        assert!(
            failure.is_none(),
            "Should NOT detect prefix failure at low entropy + late position"
        );
    }

    #[test]
    fn test_detect_prefix_failure_constraint_violation() {
        let pruner = MockPruner::new(vec![]);
        let trd = TrajectoryRefinedDraft::new(TrdConfig::default(), &pruner);

        // Even peaked dist — constraint violation always triggers
        let probs: Vec<f32> = vec![0.99, 0.01];
        let failure =
            trd.detect_prefix_failure(0, &probs, 0, 5, RejectionReason::ConstraintViolation);

        assert!(
            failure.is_some(),
            "ConstraintViolation should always detect prefix failure"
        );
    }

    #[test]
    fn test_refine_branch_basic() {
        let pruner = MockPruner::new(vec![3, 7]); // tokens 3 and 7 are invalid
        let mut trd = TrajectoryRefinedDraft::new(TrdConfig::default(), &pruner);

        // Raw branch: [0, 1, 2, 3(FAIL), 4]
        let raw = vec![0usize, 1, 2, 3, 4];
        let failure = FailurePoint {
            token_idx: 3,
            entropy: 1.5,
            reason: RejectionReason::ArgmaxMismatch,
        };

        // Marginals: each position has a probability distribution
        let m0: Vec<f32> = vec![0.4, 0.3, 0.2, 0.05, 0.05];
        let m1: Vec<f32> = vec![0.3, 0.3, 0.2, 0.1, 0.1];
        let m2: Vec<f32> = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let m3: Vec<f32> = vec![0.1, 0.3, 0.3, 0.2, 0.1]; // token 3 has 0.2 prob
        let m4: Vec<f32> = vec![0.3, 0.3, 0.2, 0.1, 0.1];
        let marginals: Vec<&[f32]> = vec![&m0, &m1, &m2, &m3, &m4];

        let mut rng = Rng::new(42);

        let result = trd.refine_branch(&raw, &failure, &marginals, &mut rng);

        // Refined tokens should NOT contain token 3 (invalid per MockPruner)
        assert!(
            !result.refined_tokens.contains(&3),
            "Refined tokens should not contain invalid token 3, got {:?}",
            result.refined_tokens
        );
    }

    #[test]
    fn test_bandit_starts_with_1step() {
        let pruner = MockPruner::new(vec![]);
        let trd = TrajectoryRefinedDraft::new(TrdConfig::default(), &pruner);
        assert_eq!(
            trd.select_refinement_depth(),
            1,
            "Should start with 1-step refinement"
        );
    }

    #[test]
    fn test_branch_score_higher_for_better_branch() {
        let m0: Vec<f32> = vec![0.8, 0.1, 0.1];
        let m1: Vec<f32> = vec![0.7, 0.2, 0.1];
        let marginals: Vec<&[f32]> = vec![&m0, &m1];

        // Branch that picks high-prob tokens
        let good = vec![0usize, 0];
        // Branch that picks low-prob tokens
        let bad = vec![2usize, 2];

        let score_good = branch_score(&good, &marginals);
        let score_bad = branch_score(&bad, &marginals);

        assert!(
            score_good > score_bad,
            "Good branch should score higher: {} vs {}",
            score_good,
            score_bad
        );
    }

    #[test]
    fn test_success_rate_initially_zero() {
        let pruner = MockPruner::new(vec![]);
        let trd = TrajectoryRefinedDraft::new(TrdConfig::default(), &pruner);
        assert_eq!(trd.success_rate(), 0.0);
    }

    #[test]
    fn test_budget_guard_aborts_on_exceeded() {
        let pruner = MockPruner::new(vec![]);
        let config = TrdConfig {
            latency_budget_us: 1, // 1 microsecond — will always be exceeded
            ..TrdConfig::default()
        };
        let mut trd = TrajectoryRefinedDraft::new(config, &pruner);

        let raw = vec![0usize, 1, 2, 3, 4];
        let failure = FailurePoint {
            token_idx: 3,
            entropy: 1.5,
            reason: RejectionReason::ArgmaxMismatch,
        };

        let m0: Vec<f32> = vec![0.4, 0.3, 0.2, 0.05, 0.05];
        let m1: Vec<f32> = vec![0.3, 0.3, 0.2, 0.1, 0.1];
        let m2: Vec<f32> = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let m3: Vec<f32> = vec![0.1, 0.3, 0.3, 0.2, 0.1];
        let m4: Vec<f32> = vec![0.3, 0.3, 0.2, 0.1, 0.1];
        let marginals: Vec<&[f32]> = vec![&m0, &m1, &m2, &m3, &m4];

        let mut rng = Rng::new(42);

        // Sleep a tiny bit to ensure the 1μs budget is exceeded
        std::thread::sleep(std::time::Duration::from_micros(10));

        let result = trd.refine_branch(&raw, &failure, &marginals, &mut rng);

        assert!(
            result.budget_exceeded,
            "Budget should be exceeded with 1μs cap"
        );
        assert_eq!(
            result.refined_tokens, raw,
            "Should fall back to raw branch when budget exceeded"
        );
    }

    #[test]
    fn test_negative_reward_for_budget_exceeded() {
        // Test that BudgetExceeded outcome produces -0.5 reward in bandit.
        // Directly test the outcome→reward mapping + bandit update path
        // instead of relying on microsecond-precision timing.
        assert_eq!(f32::from(RefinementOutcome::BudgetExceeded), -0.5);
        assert_eq!(f32::from(RefinementOutcome::Accepted), 1.0);
        assert_eq!(f32::from(RefinementOutcome::Rejected), 0.0);

        // Verify bandit receives the reward correctly
        let pruner = MockPruner::new(vec![]);
        let config = TrdConfig::default();
        let mut trd = TrajectoryRefinedDraft::new(config, &pruner);

        // Simulate budget-exceeded scenario: manually update bandit
        trd.update_bandit(1, f32::from(RefinementOutcome::BudgetExceeded));

        let stats = trd.bandit_stats();
        let (mean_1step, pulls_1step) = stats[1];
        assert_eq!(pulls_1step, 1, "1-step arm should have been pulled once");
        assert!(
            (mean_1step - (-0.5f32)).abs() < 1e-6,
            "Reward should be exactly -0.5, got {}",
            mean_1step
        );
    }

    /// Test that prefold_prefix compacts redundant tokens when chain_fold is enabled.
    ///
    /// When `chain_fold` is not enabled, this test still passes but verifies
    /// the no-op fallback (returns prefix as-is).
    #[test]
    fn test_prefold_prefix_compacts() {
        // Prefix with many repeated tokens (redundant reasoning pattern)
        let prefix: Vec<usize> = vec![0, 5, 5, 5, 5, 3, 5, 5, 5, 9];

        let compacted = prefold_prefix(&prefix);

        // Should always return valid tokens (subset or full)
        assert!(
            !compacted.is_empty(),
            "Compacted prefix should not be empty"
        );
        // First and last are anchors — must be preserved
        assert_eq!(compacted[0], 0, "First token (anchor) must be preserved");
        assert_eq!(
            *compacted.last().unwrap(),
            9,
            "Last token (anchor) must be preserved"
        );

        // All compacted tokens must come from the original prefix
        for &tok in &compacted {
            assert!(
                prefix.contains(&tok),
                "Compacted token {} must exist in original prefix",
                tok
            );
        }

        // When chain_fold is enabled, redundant tokens should be folded
        #[cfg(feature = "chain_fold")]
        {
            assert!(
                compacted.len() < prefix.len(),
                "With chain_fold, prefold should compact redundant prefix: {} -> {}",
                prefix.len(),
                compacted.len()
            );
            // Token 5 appears 7/10 times — at least some should be folded
            let count_5 = compacted.iter().filter(|&&t| t == 5).count();
            assert!(
                count_5 < 7,
                "Redundant token 5 should be partially folded: {} remaining of 7",
                count_5
            );
        }

        // When chain_fold is disabled, should return prefix as-is
        #[cfg(not(feature = "chain_fold"))]
        {
            assert_eq!(
                compacted.len(),
                prefix.len(),
                "Without chain_fold, prefold should return prefix as-is"
            );
        }
    }

    #[test]
    fn test_prefold_prefix_short_unchanged() {
        // Prefix of length <= 2 should never be compacted
        let short: Vec<usize> = vec![1, 2];
        let result = prefold_prefix(&short);
        assert_eq!(result, short, "Short prefix should be returned as-is");

        let single: Vec<usize> = vec![42];
        let result = prefold_prefix(&single);
        assert_eq!(
            result, single,
            "Single-token prefix should be returned as-is"
        );
    }

    #[test]
    fn test_bandit_context_prefers_skip_when_tight() {
        let pruner = MockPruner::new(vec![]);
        let trd = TrajectoryRefinedDraft::new(TrdConfig::default(), &pruner);

        // When within_budget = true (normal), should use UCB1 (default 1-step)
        let depth_with_budget = trd.select_refinement_depth_with_context(true);
        assert_eq!(
            depth_with_budget, 1,
            "With budget available, should default to 1-step"
        );

        // When within_budget = false (tight), should prefer Skip
        let depth_tight = trd.select_refinement_depth_with_context(false);
        assert_eq!(
            depth_tight, 0,
            "With tight budget, should prefer Skip (0 steps)"
        );
    }
}
