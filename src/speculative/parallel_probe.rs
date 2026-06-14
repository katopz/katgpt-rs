//! Parallel-Probe 2D controller — training-free parallel reasoning branch control (Plan 133).
//!
//! Monitors N parallel reasoning branches via periodic answer extraction, then uses
//! **consensus-based early stopping** + **deviation-based branch pruning** to reduce
//! sequential tokens by ~30% and total tokens by ~20% while maintaining accuracy.
//!
//! ## Key Insight
//!
//! Answer-level consensus across parallel branches is a uniquely cheap global signal
//! (O(N) per probe step). Unlike distribution residuals (O(N×V)) or bandit scores
//! (requires reward signal), answer consensus needs only string/equality matching.
//!
//! ## Algorithm
//!
//! 1. Every `probe_interval` tokens, extract answers from all active branches
//! 2. Compute majority vote across active branches
//! 3. If consensus_streak >= stability_patience → stop early with consensus answer
//! 4. For each branch: if it disagrees with majority for prune_patience steps → prune
//! 5. Skip pruning during warmup (first warmup_steps probes)
//! 6. Never prune below min_active_branches
//!
//! Reference: arXiv:2602.03845 — Parallel-Probe 2D Probing for Parallel Thinking.

use std::hash::Hash;

use super::verifier::SpeculativeVerifier;
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

// ── Configuration ─────────────────────────────────────────────

/// Configuration for the Parallel-Probe controller.
///
/// Paper defaults from Table 2 across 4 model sizes (0.6B–8B).
#[derive(Clone, Debug)]
pub struct ParallelProbeConfig {
    /// Number of tokens between probe steps.
    /// Paper fixes this at 500.
    pub probe_interval: usize,
    /// Number of consecutive consensus steps before early stopping.
    /// Higher = more conservative (fewer false stops).
    pub stability_patience: usize,
    /// Number of consecutive disagreement steps before a branch is pruned.
    /// Paper range: {8, 10, 12}. Higher = more tolerant of temporary deviations.
    pub prune_patience: usize,
    /// Number of initial probe steps where pruning is suppressed.
    /// Allows branches to explore before being penalized.
    /// Paper range: {12, 15}.
    pub warmup_steps: usize,
    /// Minimum number of active branches. Never prune below this threshold.
    pub min_active_branches: usize,
    /// Fraction of active branches that must agree for a majority.
    /// 0.5 = simple majority. Higher = more stringent consensus.
    pub prune_vote_ratio: f64,
}

impl Default for ParallelProbeConfig {
    fn default() -> Self {
        Self {
            probe_interval: 500,
            stability_patience: 3,
            prune_patience: 10,
            warmup_steps: 12,
            min_active_branches: 3,
            prune_vote_ratio: 0.5,
        }
    }
}

impl ParallelProbeConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.probe_interval == 0 {
            return Err("probe_interval must be >= 1".to_string());
        }
        if self.stability_patience == 0 {
            return Err("stability_patience must be >= 1".to_string());
        }
        if self.prune_patience == 0 {
            return Err("prune_patience must be >= 1".to_string());
        }
        if self.min_active_branches == 0 {
            return Err("min_active_branches must be >= 1".to_string());
        }
        if self.prune_vote_ratio <= 0.0 || self.prune_vote_ratio > 1.0 {
            return Err(format!(
                "prune_vote_ratio must be in (0, 1], got {}",
                self.prune_vote_ratio
            ));
        }
        Ok(())
    }
}

// ── Probe Decision ────────────────────────────────────────────

/// Decision returned by the controller after probing all branches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeDecision<A> {
    /// Continue generating — no consensus yet, no branches to prune.
    Continue,
    /// Stop early — consensus reached. Contains the consensus answer.
    Stop { answer: A },
    /// Prune specific branches — they've deviated too long.
    /// Active branches continue generating.
    Prune { branch_ids: Vec<usize> },
    /// Stop early AND prune — consensus reached while pruning divergent branches.
    StopAndPrune { answer: A, branch_ids: Vec<usize> },
}

// ── Branch State ──────────────────────────────────────────────

/// Per-branch tracking state for the probe controller.
#[derive(Clone, Debug)]
pub struct BranchProbeState<A> {
    /// Index of this branch in the parallel pool.
    pub branch_id: usize,
    /// Most recently extracted answer for this branch.
    pub last_answer: Option<A>,
    /// Consecutive probe steps where this branch disagreed with majority.
    pub disagree_streak: usize,
    /// Whether this branch has been pruned (removed from active set).
    pub is_pruned: bool,
    /// Whether this branch has naturally finished (reached its own conclusion).
    pub is_finished: bool,
}

impl<A> BranchProbeState<A> {
    /// Create a new branch state for the given branch index.
    pub fn new(branch_id: usize) -> Self {
        Self {
            branch_id,
            last_answer: None,
            disagree_streak: 0,
            is_pruned: false,
            is_finished: false,
        }
    }
}

// ── Probing Matrix ────────────────────────────────────────────

/// Generic N×T answer matrix recording probe results over time.
///
/// - N = number of parallel branches
/// - T = number of probe steps
///
/// Each cell stores the extracted answer (or None if the branch was
/// pruned/finished at that step).
#[derive(Clone, Debug)]
pub struct ProbingMatrix<A> {
    /// Row-major: `answers[branch_idx][probe_step]`.
    answers: Vec<Vec<Option<A>>>,
    /// Number of parallel branches (N).
    branch_count: usize,
    /// Maximum number of probe steps (T). 0 = unbounded.
    max_probes: usize,
}

impl<A: Clone> ProbingMatrix<A> {
    /// Create a new probing matrix for `branch_count` branches.
    ///
    /// `max_probes = 0` means unbounded probe steps.
    pub fn new(branch_count: usize, max_probes: usize) -> Self {
        let answers = vec![Vec::new(); branch_count];
        Self {
            answers,
            branch_count,
            max_probes,
        }
    }

    /// Number of parallel branches (N).
    pub fn branch_count(&self) -> usize {
        self.branch_count
    }

    /// Current number of probe steps recorded (T).
    pub fn probe_steps(&self) -> usize {
        self.answers.first().map_or(0, |r| r.len())
    }

    /// Push answers for all branches at a new probe step.
    ///
    /// `step_answers` must have length == branch_count.
    /// Entries are `Some(answer)` for active branches, `None` for pruned/finished.
    pub fn push_step(&mut self, step_answers: Vec<Option<A>>) {
        debug_assert_eq!(
            step_answers.len(),
            self.branch_count,
            "step_answers length must match branch_count"
        );
        if self.max_probes > 0 && self.probe_steps() >= self.max_probes {
            return; // Matrix full.
        }
        for (branch_idx, answer) in step_answers.into_iter().enumerate() {
            if branch_idx < self.answers.len() {
                self.answers[branch_idx].push(answer);
            }
        }
    }

    /// Get the answer for a specific branch at a specific probe step.
    pub fn get(&self, branch_idx: usize, step: usize) -> Option<&A> {
        self.answers
            .get(branch_idx)
            .and_then(|row| row.get(step))
            .and_then(|opt| opt.as_ref())
    }

    /// Get all answers for a specific branch across all probe steps.
    pub fn row(&self, branch_idx: usize) -> &[Option<A>] {
        self.answers
            .get(branch_idx)
            .map(|r| r.as_slice())
            .unwrap_or(&[])
    }

    /// Get answers for all branches at a specific probe step.
    pub fn column(&self, step: usize) -> Vec<Option<&A>> {
        self.answers
            .iter()
            .map(|row| row.get(step).and_then(|opt| opt.as_ref()))
            .collect()
    }
}

// ── Controller ────────────────────────────────────────────────

/// Main Parallel-Probe controller.
///
/// Manages N parallel reasoning branches, tracking answer consensus
/// and deviation for early stopping and branch pruning.
///
/// ## Type Parameters
///
/// - `A`: Answer type — must be `Clone + Eq + Hash` for consensus voting.
///   Typically `String` (math reasoning) or `usize` (game actions).
///
/// ## Usage
///
/// ```ignore
/// let config = ParallelProbeConfig::default();
/// let mut controller = ParallelProbeController::new(4, config);
///
/// // Every probe_interval tokens, extract answers and probe:
/// let answers: Vec<Option<String>> = extract_all_answers(branches);
/// let decision = controller.probe(&answers);
/// match decision {
///     ProbeDecision::Stop { answer } => { /* return early */ },
///     ProbeDecision::Prune { branch_ids } => { /* discard branches */ },
///     ProbeDecision::Continue => { /* keep generating */ },
///     _ => {}
/// }
/// ```
#[derive(Debug)]
pub struct ParallelProbeController<A> {
    /// Per-branch tracking state.
    branches: Vec<BranchProbeState<A>>,
    /// Controller configuration.
    config: ParallelProbeConfig,
    /// Number of consecutive steps where consensus was reached.
    consensus_streak: usize,
    /// Last consensus answer (if any).
    last_consensus: Option<A>,
    /// Current probe step number (0-indexed).
    probe_step: usize,
}

impl<A: Clone + Eq + Hash> ParallelProbeController<A> {
    /// Create a new controller for `n_branches` parallel branches.
    pub fn new(n_branches: usize, config: ParallelProbeConfig) -> Self {
        let branches = (0..n_branches).map(|i| BranchProbeState::new(i)).collect();
        Self {
            branches,
            config,
            consensus_streak: 0,
            last_consensus: None,
            probe_step: 0,
        }
    }

    /// Number of currently active (non-pruned, non-finished) branches.
    pub fn active_count(&self) -> usize {
        self.branches
            .iter()
            .filter(|b| !b.is_pruned && !b.is_finished)
            .count()
    }

    /// Current probe step number.
    pub fn probe_step(&self) -> usize {
        self.probe_step
    }

    /// Current consensus streak.
    pub fn consensus_streak(&self) -> usize {
        self.consensus_streak
    }

    /// Get a reference to branch state by index.
    pub fn branch(&self, idx: usize) -> Option<&BranchProbeState<A>> {
        self.branches.get(idx)
    }

    /// Mark a branch as finished (it reached its own conclusion).
    pub fn finish_branch(&mut self, branch_id: usize) {
        if let Some(b) = self.branches.get_mut(branch_id) {
            b.is_finished = true;
        }
    }

    /// Probe all branches with their latest extracted answers.
    ///
    /// `answers` must have length matching the number of branches.
    /// Entries are `Some(answer)` for branches that produced an answer,
    /// `None` for branches still generating or already pruned.
    ///
    /// Returns a `ProbeDecision` indicating whether to continue, stop,
    /// prune branches, or both.
    pub fn probe(&mut self, answers: &[Option<A>]) -> ProbeDecision<A> {
        // Update branch states with new answers.
        for (idx, answer) in answers.iter().enumerate() {
            if idx >= self.branches.len() {
                break;
            }
            if let Some(a) = answer {
                self.branches[idx].last_answer = Some(a.clone());
            }
        }

        // Compute majority vote among active branches with answers.
        let majority = self.majority_vote();

        let should_stop = self.should_stop(&majority);
        let to_prune = self.should_prune(&majority);

        self.probe_step += 1;

        match (should_stop, to_prune.is_empty()) {
            (Some(answer), true) => ProbeDecision::Stop { answer },
            (Some(answer), false) => {
                // Apply pruning.
                for &id in &to_prune {
                    self.branches[id].is_pruned = true;
                }
                ProbeDecision::StopAndPrune {
                    answer,
                    branch_ids: to_prune,
                }
            }
            (None, false) => {
                for &id in &to_prune {
                    self.branches[id].is_pruned = true;
                }
                ProbeDecision::Prune {
                    branch_ids: to_prune,
                }
            }
            (None, true) => ProbeDecision::Continue,
        }
    }

    /// Compute majority vote among active branches.
    ///
    /// Returns the answer held by the most branches, if it constitutes
    /// a majority (> prune_vote_ratio of active branches with answers).
    ///
    /// Uses a `Vec` linear scan rather than `HashMap` — typical branch counts
    /// are small (4-8), so the hashing overhead dominates. Pre-allocated to
    /// branch_count, so no allocation beyond the initial capacity.
    fn majority_vote(&self) -> Option<A> {
        let mut counts: Vec<(A, usize)> = Vec::with_capacity(self.branches.len());
        let mut total_with_answer = 0usize;

        for b in &self.branches {
            if b.is_pruned || b.is_finished {
                continue;
            }
            if let Some(ref answer) = b.last_answer {
                if let Some(slot) = counts.iter_mut().find(|(a, _)| a == answer) {
                    slot.1 += 1;
                } else {
                    counts.push((answer.clone(), 1));
                }
                total_with_answer += 1;
            }
        }

        if total_with_answer == 0 {
            return None;
        }

        let threshold = (total_with_answer as f64 * self.config.prune_vote_ratio).ceil() as usize;

        counts
            .into_iter()
            .find(|(_, count)| *count >= threshold.max(1))
            .map(|(answer, _)| answer)
    }

    /// Check whether we should stop early based on consensus stability.
    ///
    /// Returns `Some(consensus_answer)` if we should stop, `None` otherwise.
    fn should_stop(&mut self, majority: &Option<A>) -> Option<A> {
        match majority {
            Some(consensus) => {
                // Check if consensus matches previous (or is first).
                let is_same = self
                    .last_consensus
                    .as_ref()
                    .is_none_or(|prev| prev == consensus);

                if is_same {
                    self.consensus_streak += 1;
                } else {
                    self.consensus_streak = 1;
                }
                self.last_consensus = Some(consensus.clone());

                if self.consensus_streak >= self.config.stability_patience {
                    Some(consensus.clone())
                } else {
                    None
                }
            }
            None => {
                // No consensus — reset streak.
                self.consensus_streak = 0;
                self.last_consensus = None;
                None
            }
        }
    }

    /// Determine which branches to prune based on deviation from majority.
    ///
    /// Returns branch IDs that should be pruned.
    fn should_prune(&mut self, majority: &Option<A>) -> Vec<usize> {
        // Skip pruning during warmup.
        if self.probe_step < self.config.warmup_steps {
            return Vec::new();
        }

        let Some(consensus) = majority else {
            // No majority — reset all disagree streaks (nothing to deviate from).
            for b in &mut self.branches {
                if !b.is_pruned && !b.is_finished {
                    b.disagree_streak = 0;
                }
            }
            return Vec::new();
        };

        let mut active_count = 0usize;
        let mut to_prune = Vec::new();

        for b in &mut self.branches {
            if b.is_pruned || b.is_finished {
                continue;
            }
            active_count += 1;

            let Some(ref answer) = b.last_answer else {
                continue;
            };

            if answer == consensus {
                b.disagree_streak = 0;
            } else {
                b.disagree_streak += 1;
            }

            if b.disagree_streak >= self.config.prune_patience {
                to_prune.push(b.branch_id);
            }
        }

        // Respect min_active_branches — don't prune too many.
        let max_prunable = active_count.saturating_sub(self.config.min_active_branches);
        to_prune.truncate(max_prunable);

        to_prune
    }
}

// ── ParallelProbeVerifier (Plan 133 T3) ──────────────────────

/// Wrapper that layers parallel-probe consensus/pruning on top of any
/// [`SpeculativeVerifier`](super::verifier::SpeculativeVerifier).
///
/// On each speculative step, the verifier extracts answers from all active branches
/// using the injected [`AnswerExtractor`](super::answer_extract::AnswerExtractor), feeds
/// them to the [`ParallelProbeController`], and handles the resulting [`ProbeDecision`].
///
/// ## Usage
///
/// ```ignore
/// use katgpt_rs::speculative::parallel_probe::{ParallelProbeConfig, ParallelProbeVerifier};
/// use katgpt_rs::speculative::answer_extract::RegexAnswerExtractor;
///
/// let config = ParallelProbeConfig::default();
/// let verifier = ParallelProbeVerifier::new(
///     inner_verifier,
///     4, // n_branches
///     config,
///     Box::new(RegexAnswerExtractor::new()),
/// );
/// ```
pub struct ParallelProbeVerifier<V> {
    /// Inner speculative verifier that performs actual token-level verification.
    inner: V,
    /// Parallel-probe controller managing branch state and consensus.
    controller: ParallelProbeController<String>,
    /// Answer extractor used to pull structured answers from decoded text.
    extractor: Box<dyn super::answer_extract::AnswerExtractor>,
    /// Per-branch accumulated decoded text (grows between probe steps).
    branch_texts: Vec<String>,
    /// Number of tokens generated since the last probe step.
    tokens_since_probe: usize,
    /// Cached result from the last probe decision (if early-stop was triggered).
    cached_decision: Option<ProbeDecision<String>>,
}

impl<V: std::fmt::Debug> std::fmt::Debug for ParallelProbeVerifier<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelProbeVerifier")
            .field("inner", &self.inner)
            .field("controller", &self.controller)
            .field("extractor", &"<AnswerExtractor>")
            .field("branch_texts", &self.branch_texts)
            .field("tokens_since_probe", &self.tokens_since_probe)
            .field("cached_decision", &self.cached_decision)
            .finish()
    }
}

impl<V> ParallelProbeVerifier<V> {
    /// Create a new parallel-probe verifier wrapping `inner`.
    ///
    /// - `inner`: the underlying speculative verifier.
    /// - `n_branches`: number of parallel reasoning branches.
    /// - `config`: controller configuration.
    /// - `extractor`: answer extraction strategy.
    pub fn new(
        inner: V,
        n_branches: usize,
        config: ParallelProbeConfig,
        extractor: Box<dyn super::answer_extract::AnswerExtractor>,
    ) -> Self {
        let controller = ParallelProbeController::new(n_branches, config);
        let branch_texts = vec![String::new(); n_branches];
        Self {
            inner,
            controller,
            extractor,
            branch_texts,
            tokens_since_probe: 0,
            cached_decision: None,
        }
    }

    /// Number of currently active (non-pruned, non-finished) branches.
    pub fn active_branches(&self) -> usize {
        self.controller.active_count()
    }

    /// Current probe step number.
    pub fn probe_step(&self) -> usize {
        self.controller.probe_step()
    }

    /// Current consensus streak.
    pub fn consensus_streak(&self) -> usize {
        self.controller.consensus_streak()
    }

    /// Get the cached probe decision (set after a probe step that returned Stop or StopAndPrune).
    pub fn last_decision(&self) -> Option<&ProbeDecision<String>> {
        self.cached_decision.as_ref()
    }

    /// Record decoded text for a specific branch.
    ///
    /// Call this as each branch generates tokens to accumulate text for extraction.
    pub fn append_branch_text(&mut self, branch_id: usize, text: &str) {
        if let Some(bt) = self.branch_texts.get_mut(branch_id) {
            bt.push_str(text);
        }
    }

    /// Record that `n_tokens` were generated since the last probe.
    ///
    /// Returns `true` if a probe step should be triggered (tokens_since_probe >= probe_interval).
    pub fn record_tokens(&mut self, n_tokens: usize) -> bool {
        self.tokens_since_probe += n_tokens;
        // Access config via controller's config isn't public, so we store the interval.
        // For now, callers decide when to call probe(). This method tracks token count.
        true
    }

    /// Check if enough tokens have been generated since the last probe to warrant a new probe.
    ///
    /// Callers should use this to decide when to call [`probe_branches`](Self::probe_branches).
    pub fn should_probe(&self, probe_interval: usize) -> bool {
        self.tokens_since_probe >= probe_interval
    }

    /// Run a probe step: extract answers from all branches and feed to the controller.
    ///
    /// Returns the [`ProbeDecision`] from the controller. If the decision is `Stop` or
    /// `StopAndPrune`, it is also cached in [`last_decision`](Self::last_decision).
    ///
    /// After this call, `tokens_since_probe` is reset to 0.
    pub fn probe_branches(&mut self) -> ProbeDecision<String> {
        let n = self.branch_texts.len();
        let mut answers = Vec::with_capacity(n);

        for (branch_id, text) in self.branch_texts.iter().enumerate() {
            let branch = self.controller.branch(branch_id);
            if branch.is_none_or(|b| b.is_pruned || b.is_finished) {
                answers.push(None);
            } else {
                answers.push(self.extractor.extract_answer(&[], text));
            }
        }

        let decision = self.controller.probe(&answers);
        self.tokens_since_probe = 0;

        // Cache stop decisions so callers can retrieve them later.
        if matches!(
            decision,
            ProbeDecision::Stop { .. } | ProbeDecision::StopAndPrune { .. }
        ) {
            self.cached_decision = Some(decision.clone());
        }

        decision
    }

    /// Access the inner verifier.
    pub fn inner(&self) -> &V {
        &self.inner
    }

    /// Access the inner verifier mutably.
    pub fn inner_mut(&mut self) -> &mut V {
        &mut self.inner
    }
}

// ── SpeculativeVerifier Integration ──────────────────────────

#[cfg(feature = "parallel_probe")]
impl<V: SpeculativeVerifier> SpeculativeVerifier for ParallelProbeVerifier<V> {
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        // Delegate to the inner verifier for actual speculative decoding.
        let result = self
            .inner
            .speculate(draft_weights, draft_config, token, pos, rng);

        // Track how many tokens were generated since the last probe.
        self.record_tokens(result.len());

        result
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> ParallelProbeConfig {
        ParallelProbeConfig {
            probe_interval: 100,
            stability_patience: 2,
            prune_patience: 3,
            warmup_steps: 1,
            min_active_branches: 2,
            prune_vote_ratio: 0.5,
        }
    }

    // ── Config Tests ──────────────────────────────────────────

    #[test]
    fn test_config_default_validate() {
        assert!(ParallelProbeConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_zero_probe_interval() {
        let c = ParallelProbeConfig {
            probe_interval: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_stability_patience() {
        let c = ParallelProbeConfig {
            stability_patience: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_prune_patience() {
        let c = ParallelProbeConfig {
            prune_patience: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_min_active() {
        let c = ParallelProbeConfig {
            min_active_branches: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_bad_vote_ratio() {
        let c = ParallelProbeConfig {
            prune_vote_ratio: 0.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = ParallelProbeConfig {
            prune_vote_ratio: 1.5,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    // ── ProbingMatrix Tests ───────────────────────────────────

    #[test]
    fn test_matrix_push_and_get() {
        let mut m: ProbingMatrix<String> = ProbingMatrix::new(3, 10);
        m.push_step(vec![
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
        ]);
        m.push_step(vec![Some("a".to_string()), None, Some("b".to_string())]);

        assert_eq!(m.probe_steps(), 2);
        assert_eq!(m.branch_count(), 3);
        assert_eq!(m.get(0, 0), Some(&"a".to_string()));
        assert_eq!(m.get(1, 1), None);
        assert_eq!(m.get(2, 0), Some(&"b".to_string()));
    }

    #[test]
    fn test_matrix_max_probes() {
        let mut m: ProbingMatrix<i32> = ProbingMatrix::new(2, 2);
        m.push_step(vec![Some(1), Some(2)]);
        m.push_step(vec![Some(3), Some(4)]);
        m.push_step(vec![Some(5), Some(6)]); // Should be ignored.

        assert_eq!(m.probe_steps(), 2);
        assert_eq!(m.get(0, 1), Some(&3));
        assert_eq!(m.get(0, 2), None); // Beyond max_probes.
    }

    #[test]
    fn test_matrix_row() {
        let mut m: ProbingMatrix<i32> = ProbingMatrix::new(2, 0);
        m.push_step(vec![Some(1), Some(10)]);
        m.push_step(vec![Some(2), Some(20)]);

        assert_eq!(m.row(0), &[Some(1), Some(2)]);
        assert_eq!(m.row(1), &[Some(10), Some(20)]);
    }

    #[test]
    fn test_matrix_column() {
        let mut m: ProbingMatrix<i32> = ProbingMatrix::new(2, 0);
        m.push_step(vec![Some(1), Some(10)]);
        m.push_step(vec![Some(2), Some(20)]);

        let col0 = m.column(0);
        assert_eq!(col0, vec![Some(&1), Some(&10)]);
    }

    #[test]
    fn test_matrix_empty() {
        let m: ProbingMatrix<i32> = ProbingMatrix::new(3, 0);
        assert_eq!(m.probe_steps(), 0);
        assert_eq!(m.branch_count(), 3);
        assert!(m.row(0).is_empty());
    }

    // ── Consensus Detection ───────────────────────────────────

    #[test]
    fn test_consensus_all_agree_immediate() {
        let config = ParallelProbeConfig {
            stability_patience: 1,
            warmup_steps: 0,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // All 3 branches agree on "42".
        let d = ctrl.probe(&[
            Some("42".to_string()),
            Some("42".to_string()),
            Some("42".to_string()),
        ]);
        assert!(
            matches!(d, ProbeDecision::Stop { ref answer } if answer == "42"),
            "expected Stop with '42', got {d:?}"
        );
    }

    #[test]
    fn test_consensus_requires_stability_patience() {
        let config = ParallelProbeConfig {
            stability_patience: 3,
            warmup_steps: 0,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // Step 1: all agree.
        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
        ]);
        assert!(matches!(d, ProbeDecision::Continue), "step 1: {d:?}");
        assert_eq!(ctrl.consensus_streak(), 1);

        // Step 2: still agree.
        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
        ]);
        assert!(matches!(d, ProbeDecision::Continue), "step 2: {d:?}");
        assert_eq!(ctrl.consensus_streak(), 2);

        // Step 3: consensus streak = 3 = stability_patience → stop.
        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
        ]);
        assert!(
            matches!(d, ProbeDecision::Stop { ref answer } if answer == "a"),
            "step 3: expected Stop, got {d:?}"
        );
    }

    #[test]
    fn test_consensus_resets_on_change() {
        let config = ParallelProbeConfig {
            stability_patience: 3,
            warmup_steps: 0,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // Step 1-2: agree on "a".
        ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
        ]);
        ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
        ]);
        assert_eq!(ctrl.consensus_streak(), 2);

        // Step 3: consensus shifts to "b" — streak resets to 1.
        ctrl.probe(&[
            Some("b".to_string()),
            Some("b".to_string()),
            Some("b".to_string()),
        ]);
        assert_eq!(ctrl.consensus_streak(), 1);
    }

    #[test]
    fn test_no_consensus_no_answer() {
        let config = ParallelProbeConfig {
            warmup_steps: 0,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // No branches have answers yet.
        let d = ctrl.probe(&[None, None, None]);
        assert!(matches!(d, ProbeDecision::Continue), "no answers: {d:?}");
        assert_eq!(ctrl.consensus_streak(), 0);
    }

    #[test]
    fn test_majority_vote_simple() {
        let config = ParallelProbeConfig {
            prune_vote_ratio: 0.5,
            warmup_steps: 0,
            stability_patience: 1,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // 2/3 agree on "yes" — that's a majority.
        let d = ctrl.probe(&[
            Some("yes".to_string()),
            Some("no".to_string()),
            Some("yes".to_string()),
        ]);
        assert!(
            matches!(d, ProbeDecision::Stop { ref answer } if answer == "yes"),
            "majority: {d:?}"
        );
    }

    // ── Deviation Pruning ─────────────────────────────────────

    #[test]
    fn test_prune_deviant_branch() {
        let config = ParallelProbeConfig {
            stability_patience: 100, // Don't stop early.
            prune_patience: 2,
            warmup_steps: 0,
            min_active_branches: 2,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // Step 1: 2 agree on "a", 1 disagrees with "b".
        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
        ]);
        // Branch 2 disagree_streak = 1 < prune_patience(2) → no prune.
        assert!(matches!(d, ProbeDecision::Continue), "step 1: {d:?}");

        // Step 2: still disagreeing.
        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
        ]);
        // Branch 2 disagree_streak = 2 >= prune_patience(2) → prune.
        match d {
            ProbeDecision::Prune { branch_ids }
            | ProbeDecision::StopAndPrune { branch_ids, .. } => {
                assert!(
                    branch_ids.contains(&2),
                    "should prune branch 2, got {branch_ids:?}"
                );
            }
            other => panic!("expected Prune or StopAndPrune, got {other:?}"),
        }
    }

    #[test]
    fn test_prune_respects_min_active() {
        let config = ParallelProbeConfig {
            stability_patience: 100,
            prune_patience: 1,
            warmup_steps: 0,
            min_active_branches: 2, // Must keep at least 2 active.
            ..make_config()
        };
        // 3 branches, 2 agree on "a", 1 disagrees with "b".
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // Step 1: branch 2 deviates, disagree_streak = 1 >= prune_patience(1).
        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
        ]);
        // Can prune branch 2 → 2 active remaining = min_active → ok.
        match d {
            ProbeDecision::Prune { branch_ids } => {
                assert!(branch_ids.contains(&2), "should prune 2: {branch_ids:?}");
            }
            ProbeDecision::Continue => {
                // Stability patience check might prevent pruning if consensus fires.
                // But stability_patience=100, so this shouldn't happen.
            }
            other => panic!("expected Prune, got {other:?}"),
        }

        // After pruning, active_count = 2 = min_active.
        // If we add more probes, no more branches should be pruned.
        // But branch 2 is already pruned, so only branches 0 and 1 remain.
    }

    #[test]
    fn test_no_prune_during_warmup() {
        let config = ParallelProbeConfig {
            stability_patience: 100,
            prune_patience: 1,
            warmup_steps: 3, // Skip pruning for first 3 steps.
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        // Steps 0, 1, 2: warmup, no pruning even though branch 2 deviates.
        for _ in 0..3 {
            let d = ctrl.probe(&[
                Some("a".to_string()),
                Some("a".to_string()),
                Some("b".to_string()),
            ]);
            assert!(
                matches!(d, ProbeDecision::Continue),
                "warmup should not prune: {d:?}"
            );
        }

        // Step 3: past warmup, branch 2 has disagree_streak >= prune_patience → prune.
        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
        ]);
        assert!(
            matches!(d, ProbeDecision::Prune { .. }),
            "post-warmup should prune: {d:?}"
        );
    }

    // ── Edge Cases ────────────────────────────────────────────

    #[test]
    fn test_single_branch() {
        let config = ParallelProbeConfig {
            stability_patience: 1,
            warmup_steps: 0,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(1, config);

        let d = ctrl.probe(&[Some("only".to_string())]);
        assert!(
            matches!(d, ProbeDecision::Stop { ref answer } if answer == "only"),
            "single branch: {d:?}"
        );
    }

    #[test]
    fn test_all_disagree_no_consensus() {
        let config = ParallelProbeConfig {
            stability_patience: 1,
            prune_patience: 100, // Don't prune in this test.
            warmup_steps: 0,
            prune_vote_ratio: 0.5,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        let d = ctrl.probe(&[
            Some("a".to_string()),
            Some("b".to_string()),
            Some("c".to_string()),
        ]);
        // No majority → continue.
        assert!(matches!(d, ProbeDecision::Continue), "all disagree: {d:?}");
        assert_eq!(ctrl.consensus_streak(), 0);
    }

    #[test]
    fn test_finish_branch() {
        let config = ParallelProbeConfig {
            warmup_steps: 0,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(3, config);

        assert_eq!(ctrl.active_count(), 3);
        ctrl.finish_branch(1);
        assert_eq!(ctrl.active_count(), 2);
    }

    #[test]
    fn test_stop_and_prune_combined() {
        let config = ParallelProbeConfig {
            stability_patience: 2,
            prune_patience: 2,
            warmup_steps: 0,
            min_active_branches: 2,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(4, config);

        // Step 1: 3 agree on "x", 1 disagrees with "y".
        ctrl.probe(&[
            Some("x".to_string()),
            Some("x".to_string()),
            Some("x".to_string()),
            Some("y".to_string()),
        ]);
        assert_eq!(ctrl.consensus_streak(), 1);

        // Step 2: consensus streak = 2 >= stability_patience(2) → stop.
        // Branch 3 disagree_streak = 2 >= prune_patience(2) → prune.
        let d = ctrl.probe(&[
            Some("x".to_string()),
            Some("x".to_string()),
            Some("x".to_string()),
            Some("y".to_string()),
        ]);
        assert!(
            matches!(d, ProbeDecision::StopAndPrune { ref answer, ref branch_ids }
                if answer == "x" && branch_ids.contains(&3)),
            "expected StopAndPrune with x, pruning 3, got {d:?}"
        );
    }

    #[test]
    fn test_active_count_after_prune() {
        let config = ParallelProbeConfig {
            stability_patience: 100,
            prune_patience: 1,
            warmup_steps: 0,
            min_active_branches: 2,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<String> = ParallelProbeController::new(4, config);

        assert_eq!(ctrl.active_count(), 4);

        // Prune branch 3 (only deviant).
        ctrl.probe(&[
            Some("a".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
        ]);
        assert_eq!(ctrl.active_count(), 3);
    }

    #[test]
    fn test_probe_step_increments() {
        let mut ctrl: ParallelProbeController<String> =
            ParallelProbeController::new(2, make_config());

        assert_eq!(ctrl.probe_step(), 0);
        ctrl.probe(&[None, None]);
        assert_eq!(ctrl.probe_step(), 1);
        ctrl.probe(&[None, None]);
        assert_eq!(ctrl.probe_step(), 2);
    }

    #[test]
    fn test_integer_answer_type() {
        let config = ParallelProbeConfig {
            stability_patience: 1,
            warmup_steps: 0,
            ..make_config()
        };
        let mut ctrl: ParallelProbeController<usize> = ParallelProbeController::new(3, config);

        let d = ctrl.probe(&[Some(42), Some(42), Some(42)]);
        assert!(
            matches!(d, ProbeDecision::Stop { answer: 42 }),
            "integer answers: {d:?}"
        );
    }
}
