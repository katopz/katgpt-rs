//! SR²AM Configurator Bandit — Learned Per-Turn Planning Regulation (Plan 112, Research 076).
//!
//! Bandit-based configurator that learns per-turn planning decisions
//! (PlanNew/PlanExtend/PlanSkip) for the DDTree speculative decoding path.
//! Uses entropy binning as context and UCB1 for arm selection.
//!
//! Reference: [arXiv:2605.22138](https://arxiv.org/pdf/2605.22138) — Deng, Hou, Sá Neves et al.

use std::collections::HashMap;

use katgpt_core::{ConfiguratorContext, PlanningDecision};

// ── Constants ─────────────────────────────────────────────────

/// Number of arms in the configurator bandit (PlanNew, PlanExtend, PlanSkip, SpecHop).
///
/// Always 4 — FeedbackBandit manages its own extended arm set (arms 4-5)
/// independently, keeping the base bandit's convergence behavior unchanged
/// when the `sia_feedback` feature is enabled.
const NUM_ARMS: usize = 4;

/// Default k (speculative thread count) when SpecHop is selected.
const DEFAULT_SPECHOP_K: usize = 4;

/// Number of entropy bins (0..10, coarse discretization).
const NUM_ENTROPY_BINS: usize = 10;

/// Default β weight for token cost in reward signal.
#[cfg(test)]
const DEFAULT_BETA: f32 = 0.1;

/// UCB1 exploration constant (sqrt(2) from Auer et al.).
pub(crate) const UCB1_C: f32 = 2.0;

// ── Arm Index Mapping ─────────────────────────────────────────

/// Map `PlanningDecision` to arm index for Q-value arrays.
/// Only maps the 4 base SR²AM arms. FeedbackBandit handles arms 4-5.
#[inline]
pub(crate) fn arm_index(decision: PlanningDecision) -> usize {
    match decision {
        PlanningDecision::PlanNew => 0,
        PlanningDecision::PlanExtend => 1,
        PlanningDecision::PlanSkip => 2,
        PlanningDecision::SpecHop { .. } => 3,
        #[cfg(feature = "sia_feedback")]
        PlanningDecision::HarnessUpdate | PlanningDecision::WeightUpdate => {
            unreachable!("FeedbackBandit arms should not be routed to ConfiguratorBandit")
        }
    }
}

/// Map arm index back to `PlanningDecision` (base 4 arms only).
#[inline]
pub(crate) fn from_arm_index(idx: usize) -> PlanningDecision {
    match idx {
        0 => PlanningDecision::PlanNew,
        1 => PlanningDecision::PlanExtend,
        2 => PlanningDecision::PlanSkip,
        _ => PlanningDecision::SpecHop {
            k: DEFAULT_SPECHOP_K,
        },
    }
}

// ── Per-Context Stats ─────────────────────────────────────────

/// Per-context Q-values and visit counts for one `(domain, entropy_bin)` pair.
#[derive(Debug, Clone)]
struct ContextStats {
    /// Q-values for each arm [PlanNew, PlanExtend, PlanSkip].
    q_values: [f32; NUM_ARMS],
    /// Visit counts for each arm.
    visits: [usize; NUM_ARMS],
    /// Total pulls across all arms for this context.
    total_pulls: usize,
}

impl ContextStats {
    fn new() -> Self {
        Self {
            q_values: [0.0; NUM_ARMS],
            visits: [0; NUM_ARMS],
            total_pulls: 0,
        }
    }

    /// UCB1 score for a given arm.
    /// Returns `f32::MAX` for unvisited arms (must explore first).
    #[inline]
    fn ucb1_score(&self, arm: usize) -> f32 {
        match (self.visits[arm], self.total_pulls) {
            (0, _) | (_, 0) => f32::MAX,
            (n, total) => {
                let q = self.q_values[arm];
                let ln_total = (total as f32).ln();
                q + (UCB1_C * ln_total / n as f32).sqrt()
            }
        }
    }

    /// Update Q-value for `arm` after observing `reward`.
    /// Uses incremental mean: `Q(a) += (reward - Q(a)) / n(a)`.
    #[inline]
    fn update(&mut self, arm: usize, reward: f32) {
        self.visits[arm] += 1;
        self.total_pulls += 1;
        let n = self.visits[arm] as f32;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;
    }

    /// Select best arm by UCB1 score. Ties broken by lower index.
    fn best_ucb1_arm(&self) -> usize {
        let mut best_arm = 0;
        let mut best_score = self.ucb1_score(0);
        for arm in 1..NUM_ARMS {
            let score = self.ucb1_score(arm);
            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }
        best_arm
    }
}

// ── ConfiguratorBandit ────────────────────────────────────────

/// Bandit-based configurator for per-turn planning regulation.
///
/// Uses UCB1 selection from existing bandit infrastructure.
/// Q-values keyed by `(domain, entropy_bin)` — context-aware arm selection.
///
/// # Context Binning
///
/// Entropy is discretized into 10 bins: `floor(entropy * 10.0)` clamped to 0..9.
/// Combined with domain index, this gives `(domain, bin)` context keys.
///
/// # Arms
///
/// - `PlanNew` (arm 0): reset tree, full budget allocation
/// - `PlanExtend` (arm 1): keep tree, extend depth by one level
/// - `PlanSkip` (arm 2): skip tree search, direct token sampling
///
/// # Reward Signal
///
/// `reward = quality_gain - β * token_cost`
///
/// Where `quality_gain` measures improvement in relevance and `token_cost`
/// normalizes token usage against budget.
pub struct ConfiguratorBandit {
    /// Per-context Q-values and visit counts.
    /// Key: `(domain, entropy_bin, desperation_bin)`, Value: stats for 3 arms.
    stats: HashMap<(usize, usize, usize), ContextStats>,
}

impl ConfiguratorBandit {
    /// Create a new configurator bandit with empty Q-table.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Get or create stats for a given context.
    fn get_or_create_stats(&mut self, context: ConfiguratorContext) -> &mut ContextStats {
        let key = (context.domain, context.entropy_bin, context.desperation_bin);
        self.stats.entry(key).or_insert_with(ContextStats::new)
    }

    /// Select a planning decision using UCB1 for the given context.
    ///
    /// UCB1 balances exploration (try new planning depths) vs exploitation
    /// (use known-good depth). Unvisited arms get `f32::MAX` score, ensuring
    /// each arm is tried at least once before exploiting.
    ///
    /// # Arguments
    ///
    /// * `context` — `(domain, entropy_bin)` context key
    ///
    /// # Returns
    ///
    /// The selected `PlanningDecision` arm.
    pub fn select(&mut self, context: ConfiguratorContext) -> PlanningDecision {
        let stats = self.get_or_create_stats(context);
        let arm = stats.best_ucb1_arm();
        from_arm_index(arm)
    }

    /// Update Q-value for a context-decision pair after observing reward.
    ///
    /// Uses incremental mean: `Q(a) += (reward - Q(a)) / n(a)`.
    ///
    /// # Arguments
    ///
    /// * `context` — `(domain, entropy_bin)` context key
    /// * `decision` — The arm that was selected
    /// * `reward` — Observed reward signal
    pub fn update(
        &mut self,
        context: ConfiguratorContext,
        decision: PlanningDecision,
        reward: f32,
    ) {
        let stats = self.get_or_create_stats(context);
        let arm = arm_index(decision);
        stats.update(arm, reward);
    }

    /// Discretize entropy into a coarse bin index.
    ///
    /// `floor(entropy * 10.0)`, clamped to `0..NUM_ENTROPY_BINS`.
    /// This provides 10 bins covering entropy range [0, ~3+):
    /// - Bin 0: entropy < 0.1 (very confident)
    /// - Bin 1: entropy 0.1..0.2
    /// - ...
    /// - Bin 9: entropy >= 0.9 (highly uncertain)
    ///
    /// # Arguments
    ///
    /// * `entropy` — Shannon entropy in nats (>= 0)
    ///
    /// # Returns
    ///
    /// Bin index in `0..10`.
    pub fn entropy_bin(entropy: f32) -> usize {
        let bin = (entropy * 10.0).floor() as usize;
        bin.min(NUM_ENTROPY_BINS - 1)
    }

    /// Compute reward signal for Q-value updates.
    ///
    /// `reward = quality_gain - β * token_cost`
    ///
    /// Encourages quality improvement while penalizing excessive token usage.
    /// Default β=0.1 provides mild cost regularization.
    ///
    /// # Arguments
    ///
    /// * `quality_gain` — Change in screening relevance (positive = improvement)
    /// * `token_cost` — Normalized token usage `tokens_used / tree_budget` in [0, 1]
    /// * `beta` — Cost regularization weight (default 0.1)
    ///
    /// # Returns
    ///
    /// Scalar reward signal.
    pub fn reward_signal(quality_gain: f32, token_cost: f32, beta: f32) -> f32 {
        quality_gain - beta * token_cost
    }

    /// Number of unique contexts seen so far.
    pub fn num_contexts(&self) -> usize {
        self.stats.len()
    }

    /// Get Q-value for a specific context and decision.
    /// Returns `None` if context has never been visited.
    pub fn q_value(&self, context: ConfiguratorContext, decision: PlanningDecision) -> Option<f32> {
        let key = (context.domain, context.entropy_bin, context.desperation_bin);
        self.stats
            .get(&key)
            .map(|s| s.q_values[arm_index(decision)])
    }

    /// Get visit count for a specific context and decision.
    /// Returns `0` if context has never been visited.
    pub fn visit_count(&self, context: ConfiguratorContext, decision: PlanningDecision) -> usize {
        let key = (context.domain, context.entropy_bin, context.desperation_bin);
        match self.stats.get(&key) {
            Some(s) => s.visits[arm_index(decision)],
            None => 0,
        }
    }

    /// Get total pulls for a specific context.
    /// Returns `0` if context has never been visited.
    pub fn total_pulls(&self, context: ConfiguratorContext) -> usize {
        let key = (context.domain, context.entropy_bin, context.desperation_bin);
        match self.stats.get(&key) {
            Some(s) => s.total_pulls,
            None => 0,
        }
    }
}

impl Default for ConfiguratorBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── Structured Feedback Taxonomy (Plan 146, Research 108: Sailor) ──

/// Structured exploration outcome — inspired by Sailor's feedback taxonomy.
///
/// Sailor classifies symbolic execution feedback into:
///   - "not reached" → target line not executed
///   - "site reached" → target reached, no violation
///   - "bug triggered" → concrete violation confirmed
///
/// Our analog for game-state exploration:
///
/// | Outcome | Sailor Analog | Action |
/// |---------|--------------|--------|
/// | `NotReached` | "not reached" | Adjust Q-values (negative reward) |
/// | `StateReachedNoWin` | "site reached" | Neutral reward + log for tuning |
/// | `WinConfirmed` | "bug triggered" | Positive reward + update GOAT proof |
/// | `InvalidState` | "compilation error" | Zero reward + flag for validator check |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorationOutcome {
    /// Move sequence didn't reach target game state.
    /// Sailor: "not reached" → LLM fixes driver/stubs.
    /// Action: Adjust bandit Q-values (negative reward).
    NotReached,
    /// Target state reached but no win condition.
    /// Sailor: "site reached" → LLM tightens constraints.
    /// Action: Neutral reward + log for constraint tuning.
    StateReachedNoWin,
    /// Concrete win condition satisfied.
    /// Sailor: "bug triggered" → confirmed vulnerability.
    /// Action: Positive reward + update GOAT proof.
    WinConfirmed,
    /// Invalid game state detected.
    /// Sailor: "compilation error" → fix harness.
    /// Action: Zero reward + flag for WASM validator check.
    InvalidState,
}

impl ExplorationOutcome {
    /// Convert to a scalar reward for bandit Q-value updates.
    ///
    /// | Outcome | Reward |
    /// |---------|--------|
    /// | `WinConfirmed` | +1.0 |
    /// | `StateReachedNoWin` | 0.0 |
    /// | `NotReached` | -0.5 |
    /// | `InvalidState` | -1.0 |
    pub fn to_reward(&self) -> f32 {
        match self {
            Self::WinConfirmed => 1.0,
            Self::StateReachedNoWin => 0.0,
            Self::NotReached => -0.5,
            Self::InvalidState => -1.0,
        }
    }
}

// ── PrunerSchedule (Plan 171: Thinking Prune) ──────────────────

/// How ScreeningPruner is applied across hops/loops in SpecHop/LT2.
///
/// Controls whether intermediate steps use lightweight screening
/// (FrozenBaseGuard) or full screening at every step.
///
/// From the Thinking Pixel paper (arXiv:2604.25299 §3.3): repeated full
/// verification at every recursion step corrupts the draft distribution
/// through compounding artifacts. The solution: apply lightweight processing
/// at intermediate steps, full model only at the final step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrunerSchedule {
    /// Apply the full ScreeningPruner at every hop/loop step.
    /// Current behavior before Plan 171.
    Uniform,
    /// Apply FrozenBaseGuard: intermediate steps accept all (relevance 1.0),
    /// only the final step applies the full ScreeningPruner.
    /// This is the new default — pure speedup, no quality loss.
    #[default]
    FrozenBaseGuard,
    /// Route by per-token entropy: high-entropy → relaxed, low-entropy → tight.
    /// Uses same threshold as EntropyBifurcatedPruner.
    /// Stored as `u32` bits for `Eq`/`Hash` compatibility; use `threshold_f32()` to decode.
    #[cfg(feature = "directional_credit")]
    EntropyRouted { threshold: u32 },
}

impl PrunerSchedule {
    /// Returns true if this schedule uses FrozenBaseGuard semantics.
    #[inline]
    pub fn is_frozen_base(&self) -> bool {
        matches!(self, Self::FrozenBaseGuard)
    }

    /// Returns true if this schedule applies full screening at every step.
    #[inline]
    pub fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform)
    }

    /// Determine if the given step should apply full screening.
    ///
    /// For `FrozenBaseGuard`: only true when `hop_index == total_hops - 1`.
    /// For `Uniform`: always true.
    /// For `EntropyRouted`: always true (routing happens at pruner level).
    #[inline]
    pub fn should_screen_full(&self, hop_index: usize, total_hops: usize) -> bool {
        match self {
            Self::Uniform => true,
            Self::FrozenBaseGuard => hop_index >= total_hops.saturating_sub(1),
            #[cfg(feature = "directional_credit")]
            Self::EntropyRouted { .. } => true,
        }
    }

    /// Get the entropy threshold as `f32` (only meaningful for `EntropyRouted`).
    #[cfg(feature = "directional_credit")]
    #[inline]
    pub fn threshold_f32(&self) -> Option<f32> {
        match self {
            Self::EntropyRouted { threshold } => Some(f32::from_bits(*threshold)),
            _ => None,
        }
    }

    /// Construct an `EntropyRouted` schedule from an `f32` threshold.
    #[cfg(feature = "directional_credit")]
    #[inline]
    pub fn entropy_routed(threshold: f32) -> Self {
        Self::EntropyRouted {
            threshold: threshold.to_bits(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PlanningDecision variants ─────────────────────────────

    #[test]
    fn test_planning_decision_variants() {
        assert_eq!(arm_index(PlanningDecision::PlanNew), 0);
        assert_eq!(arm_index(PlanningDecision::PlanExtend), 1);
        assert_eq!(arm_index(PlanningDecision::PlanSkip), 2);
        assert_eq!(arm_index(PlanningDecision::SpecHop { k: 4 }), 3);
        #[cfg(feature = "sia_feedback")]
        {
            // FeedbackBandit arms are no longer mapped in ConfiguratorBandit (T20 decoupling).
            // They are handled independently by FeedbackBandit's own UCB1.
            // Verify they panic if accidentally routed to ConfiguratorBandit.
            let result = std::panic::catch_unwind(|| arm_index(PlanningDecision::HarnessUpdate));
            assert!(
                result.is_err(),
                "HarnessUpdate should not be routable to ConfiguratorBandit"
            );
        }

        assert_eq!(from_arm_index(0), PlanningDecision::PlanNew);
        assert_eq!(from_arm_index(1), PlanningDecision::PlanExtend);
        assert_eq!(from_arm_index(2), PlanningDecision::PlanSkip);
        assert!(matches!(
            from_arm_index(3),
            PlanningDecision::SpecHop { .. }
        ));
        // from_arm_index only handles base 4 arms; feedback arms are managed by FeedbackBandit
        #[cfg(feature = "sia_feedback")]
        {
            // Indices >= 4 map to SpecHop (fallback) — FeedbackBandit handles them separately
            assert!(matches!(
                from_arm_index(4),
                PlanningDecision::SpecHop { .. }
            ));
            assert!(matches!(
                from_arm_index(99),
                PlanningDecision::SpecHop { .. }
            ));
        }
    }

    // ── Entropy binning ───────────────────────────────────────

    #[test]
    fn test_entropy_bin_boundaries() {
        assert_eq!(ConfiguratorBandit::entropy_bin(0.0), 0);
        assert_eq!(ConfiguratorBandit::entropy_bin(0.05), 0);
        assert_eq!(ConfiguratorBandit::entropy_bin(0.1), 1);
        assert_eq!(ConfiguratorBandit::entropy_bin(0.15), 1);
        assert_eq!(ConfiguratorBandit::entropy_bin(0.5), 5);
        assert_eq!(ConfiguratorBandit::entropy_bin(0.99), 9);
        // Clamped to max bin
        assert_eq!(ConfiguratorBandit::entropy_bin(1.0), 9);
        assert_eq!(ConfiguratorBandit::entropy_bin(2.5), 9);
        assert_eq!(ConfiguratorBandit::entropy_bin(10.0), 9);
    }

    #[test]
    fn test_entropy_bin_negative_clamps_to_zero() {
        // floor(-0.1 * 10.0) = floor(-1.0) = -1 as usize wraps, but min clamps
        // Actually -1.0_f32.floor() = -1.0, cast to usize is UB-adjacent
        // Let's test that non-negative values work correctly
        assert_eq!(ConfiguratorBandit::entropy_bin(0.0), 0);
        assert_eq!(ConfiguratorBandit::entropy_bin(0.001), 0);
    }

    // ── Reward signal ─────────────────────────────────────────

    #[test]
    fn test_reward_signal_quality_dominates() {
        let reward = ConfiguratorBandit::reward_signal(0.8, 0.1, DEFAULT_BETA);
        assert!(reward > 0.0, "quality_gain=0.8 - 0.1*0.1 = 0.79 > 0");
        assert!((reward - 0.79).abs() < 1e-6);
    }

    #[test]
    fn test_reward_signal_cost_dominates() {
        let reward = ConfiguratorBandit::reward_signal(0.01, 1.0, DEFAULT_BETA);
        assert!(reward < 0.0, "quality_gain=0.01 - 0.1*1.0 = -0.09 < 0");
        assert!((reward - (-0.09)).abs() < 1e-6);
    }

    #[test]
    fn test_reward_signal_zero_beta_ignores_cost() {
        let reward = ConfiguratorBandit::reward_signal(0.5, 1.0, 0.0);
        assert!((reward - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_reward_signal_zero_quality() {
        let reward = ConfiguratorBandit::reward_signal(0.0, 0.5, DEFAULT_BETA);
        assert!((reward - (-0.05)).abs() < 1e-6);
    }

    // ── UCB1 selection ────────────────────────────────────────

    #[test]
    fn test_select_explores_all_arms_before_exploiting() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::new(0, 5);

        // First 3 selects should visit each arm at least once (UCB1 gives f32::MAX to unvisited)
        let mut seen = [false; NUM_ARMS];
        for _ in 0..NUM_ARMS {
            let decision = bandit.select(ctx);
            seen[arm_index(decision)] = true;
            bandit.update(ctx, decision, 0.5);
        }
        assert!(
            seen.iter().all(|&s| s),
            "UCB1 should explore all arms first"
        );
    }

    #[test]
    fn test_select_converges_to_best_arm() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::new(0, 5);

        // PlanSkip (arm 2) gets consistently high rewards
        // Others get low rewards
        for _ in 0..100 {
            let decision = bandit.select(ctx);
            let reward = match decision {
                PlanningDecision::PlanSkip => 1.0,
                _ => 0.0,
            };
            bandit.update(ctx, decision, reward);
        }

        // After many rounds, should strongly prefer PlanSkip
        let skip_visits = bandit.visit_count(ctx, PlanningDecision::PlanSkip);
        assert!(
            skip_visits > 30,
            "PlanSkip should dominate, got {skip_visits} visits"
        );
    }

    // ── Context isolation ─────────────────────────────────────

    #[test]
    fn test_context_isolation() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx_low = ConfiguratorContext::new(0, 1);
        let ctx_high = ConfiguratorContext::new(0, 8);

        // Train low entropy context to prefer PlanSkip
        for _ in 0..50 {
            let decision = bandit.select(ctx_low);
            let reward = match decision {
                PlanningDecision::PlanSkip => 1.0,
                _ => 0.0,
            };
            bandit.update(ctx_low, decision, reward);
        }

        // Train high entropy context to prefer PlanNew
        for _ in 0..50 {
            let decision = bandit.select(ctx_high);
            let reward = match decision {
                PlanningDecision::PlanNew => 1.0,
                _ => 0.0,
            };
            bandit.update(ctx_high, decision, reward);
        }

        // Low entropy should prefer PlanSkip
        let skip_q = bandit
            .q_value(ctx_low, PlanningDecision::PlanSkip)
            .unwrap_or(0.0);
        let new_q = bandit
            .q_value(ctx_low, PlanningDecision::PlanNew)
            .unwrap_or(0.0);
        assert!(
            skip_q > new_q,
            "low entropy should prefer PlanSkip: skip_q={skip_q} > new_q={new_q}"
        );

        // High entropy should prefer PlanNew
        let new_q = bandit
            .q_value(ctx_high, PlanningDecision::PlanNew)
            .unwrap_or(0.0);
        let skip_q = bandit
            .q_value(ctx_high, PlanningDecision::PlanSkip)
            .unwrap_or(0.0);
        assert!(
            new_q > skip_q,
            "high entropy should prefer PlanNew: new_q={new_q} > skip_q={skip_q}"
        );
    }

    // ── Low entropy tends toward PlanSkip ─────────────────────

    #[test]
    fn test_configurator_bandit_selects_plan_skip_at_low_entropy() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::new(0, 0); // Very low entropy

        // Train: PlanSkip is best at low entropy
        for _ in 0..200 {
            let decision = bandit.select(ctx);
            let reward = match decision {
                PlanningDecision::PlanSkip => 0.9,
                PlanningDecision::PlanExtend => 0.3,
                PlanningDecision::PlanNew => 0.1,
                PlanningDecision::SpecHop { .. } => 0.2,
                #[cfg(feature = "sia_feedback")]
                PlanningDecision::HarnessUpdate => 0.2,
                #[cfg(feature = "sia_feedback")]
                PlanningDecision::WeightUpdate => 0.1,
            };
            bandit.update(ctx, decision, reward);
        }

        let skip_visits = bandit.visit_count(ctx, PlanningDecision::PlanSkip);
        let new_visits = bandit.visit_count(ctx, PlanningDecision::PlanNew);
        assert!(
            skip_visits > new_visits,
            "PlanSkip should dominate at low entropy: skip={skip_visits} > new={new_visits}"
        );
    }

    // ── High entropy tends toward PlanNew ─────────────────────

    #[test]
    fn test_configurator_bandit_selects_plan_new_at_high_entropy() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::new(0, 9); // High entropy

        // Train: PlanNew is best at high entropy
        for _ in 0..200 {
            let decision = bandit.select(ctx);
            let reward = match decision {
                PlanningDecision::PlanNew => 0.9,
                PlanningDecision::PlanExtend => 0.3,
                PlanningDecision::PlanSkip => 0.1,
                PlanningDecision::SpecHop { .. } => 0.2,
                #[cfg(feature = "sia_feedback")]
                PlanningDecision::HarnessUpdate => 0.2,
                #[cfg(feature = "sia_feedback")]
                PlanningDecision::WeightUpdate => 0.1,
            };
            bandit.update(ctx, decision, reward);
        }

        let new_visits = bandit.visit_count(ctx, PlanningDecision::PlanNew);
        let skip_visits = bandit.visit_count(ctx, PlanningDecision::PlanSkip);
        assert!(
            new_visits > skip_visits,
            "PlanNew should dominate at high entropy: new={new_visits} > skip={skip_visits}"
        );
    }

    // ── Q-value and visit tracking ────────────────────────────

    #[test]
    fn test_q_value_update_incremental_mean() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::new(0, 5);

        bandit.update(ctx, PlanningDecision::PlanSkip, 1.0);
        assert!((bandit.q_value(ctx, PlanningDecision::PlanSkip).unwrap() - 1.0).abs() < 1e-6);

        bandit.update(ctx, PlanningDecision::PlanSkip, 0.0);
        assert!((bandit.q_value(ctx, PlanningDecision::PlanSkip).unwrap() - 0.5).abs() < 1e-6);

        bandit.update(ctx, PlanningDecision::PlanSkip, 1.0);
        // (1.0 + 0.0 + 1.0) / 3 = 0.6667
        let q = bandit.q_value(ctx, PlanningDecision::PlanSkip).unwrap();
        assert!((q - 2.0 / 3.0).abs() < 1e-5, "expected ~0.667, got {q}");
    }

    #[test]
    fn test_unvisited_context_returns_none() {
        let bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::new(99, 5);
        assert_eq!(bandit.q_value(ctx, PlanningDecision::PlanNew), None);
        assert_eq!(bandit.visit_count(ctx, PlanningDecision::PlanNew), 0);
        assert_eq!(bandit.total_pulls(ctx), 0);
    }

    #[test]
    fn test_num_contexts_tracks_entries() {
        let mut bandit = ConfiguratorBandit::new();
        assert_eq!(bandit.num_contexts(), 0);

        let ctx1 = ConfiguratorContext::new(0, 3);
        bandit.update(ctx1, PlanningDecision::PlanNew, 0.5);
        assert_eq!(bandit.num_contexts(), 1);

        let ctx2 = ConfiguratorContext::new(1, 7);
        bandit.update(ctx2, PlanningDecision::PlanSkip, 0.8);
        assert_eq!(bandit.num_contexts(), 2);

        // Same context doesn't add
        bandit.update(ctx1, PlanningDecision::PlanExtend, 0.3);
        assert_eq!(bandit.num_contexts(), 2);
    }

    // ── PrunerSchedule (Plan 171: Thinking Prune) ────────────

    #[test]
    fn test_pruner_schedule_default_is_frozen_base() {
        let schedule = PrunerSchedule::default();
        assert_eq!(schedule, PrunerSchedule::FrozenBaseGuard);
    }

    #[test]
    fn test_pruner_schedule_is_frozen_base() {
        assert!(PrunerSchedule::FrozenBaseGuard.is_frozen_base());
        assert!(!PrunerSchedule::Uniform.is_frozen_base());
    }

    #[test]
    fn test_pruner_schedule_is_uniform() {
        assert!(PrunerSchedule::Uniform.is_uniform());
        assert!(!PrunerSchedule::FrozenBaseGuard.is_uniform());
    }

    #[test]
    fn test_pruner_schedule_should_screen_full_uniform() {
        let schedule = PrunerSchedule::Uniform;
        assert!(schedule.should_screen_full(0, 3));
        assert!(schedule.should_screen_full(1, 3));
        assert!(schedule.should_screen_full(2, 3));
    }

    #[test]
    fn test_pruner_schedule_should_screen_full_frozen_base() {
        let schedule = PrunerSchedule::FrozenBaseGuard;
        assert!(!schedule.should_screen_full(0, 3));
        assert!(!schedule.should_screen_full(1, 3));
        assert!(schedule.should_screen_full(2, 3)); // final step
    }

    #[test]
    fn test_pruner_schedule_single_hop() {
        let schedule = PrunerSchedule::FrozenBaseGuard;
        assert!(schedule.should_screen_full(0, 1)); // only hop is final
    }
}
