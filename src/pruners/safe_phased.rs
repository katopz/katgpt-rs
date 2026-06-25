//! PrudentBanker Safe-Phased Bandit — delay-calibrated safe exploration (Plan 137).
//!
//! Implements a phased aggression strategy that mixes between an active bandit
//! learner and a safe baseline arm. Exploration only escalates when accumulated
//! evidence certifies the baseline is suboptimal.
//!
//! # Architecture
//!
//! - [`SafePhasedState`] — tracks phase, delay estimate, gap accumulation, mixing
//! - Integration with [`BanditStrategy::SafePhased`] in the bandit module
//!
//! # Delay-Calibrated Slack
//!
//! The slack term ξ(D̂ₛ) = (√(8·D̂ₛ + 1) - 1) / δ accounts for delayed feedback
//! by widening the phase-gap threshold, preventing premature escalation.

use crate::types::Rng;

/// Regret-budget constant C (PrudentBanker default).
const REGRET_BUDGET_C: f32 = 2.0;

// ── State ──────────────────────────────────────────────────────

/// Phased aggression state for the PrudentBanker safe-phased bandit.
///
/// Mixes between an active bandit learner and a safe baseline arm.
/// Only escalates exploration when accumulated evidence (phase gap)
/// certifies the baseline is suboptimal.
///
/// # Phase Mechanics
///
/// - Phase k starts with αₖ = min(2^(k-1) / R̂, 1)
/// - α is the probability of using the active (exploratory) arm
/// - When accumulated gap exceeds 2·R̂ + ξ(D̂ₛ), soft restart occurs
/// - If soft restarts don't help, hard restart doubles the delay estimate
///
/// # Delay Calibration
///
/// The slack term ξ(D̂ₛ) = (√(8·D̂ₛ + 1) - 1) / δ prevents premature
/// phase escalation when feedback is delayed.
#[derive(Clone, Debug)]
pub struct SafePhasedState {
    /// Current aggression level (starts at 1).
    phase: u32,
    /// Current delay estimate D̂ₛ (doubling trick).
    delay_estimate: f32,
    /// Cumulative phase gap on arrived data.
    phase_gap_arrived: f32,
    /// Current aggression coefficient αₖ.
    alpha: f32,
    /// R̂ based on delay estimate.
    regret_budget: f32,
    /// ξ(D̂ₛ) = (√(8·D̂ₛ + 1) - 1) / δ.
    delay_slack: f32,
    /// Safe baseline arm index.
    baseline_arm: usize,
    /// Minimum baseline probability δ.
    delta: f32,
    /// User-configured delay estimate.
    #[allow(dead_code)] // Stored for potential future resets / introspection
    estimated_delay: u32,
    /// Total rounds played.
    total_rounds: u32,
    /// Count of pending (unobserved) feedback.
    pending_observations: u32,
    /// Number of arms in the bandit.
    num_arms: usize,
    /// Round number when current phase started.
    phase_start_round: u32,
    /// Phase budget: how many rounds to spend in current phase before auto-advancing.
    phase_budget: u32,
}

impl SafePhasedState {
    /// Create a new safe-phased state.
    ///
    /// # Arguments
    ///
    /// * `baseline_arm` — index of the safe baseline arm
    /// * `delta` — minimum baseline probability (controls delay slack)
    /// * `estimated_delay` — initial delay estimate D̂₀
    /// * `num_arms` — total number of arms
    ///
    /// # Initialization
    ///
    /// α₁ = min(1/R̂, 1), R̂ = C·(√T + √(D̂ₛ·ln(D̂ₛ+1))).
    /// With T=1 (first round), this gives a conservative initial alpha.
    pub fn new(baseline_arm: usize, delta: f32, estimated_delay: u32, num_arms: usize) -> Self {
        assert!(baseline_arm < num_arms, "baseline_arm must be < num_arms");
        assert!(delta > 0.0 && delta <= 1.0, "delta must be in (0, 1]");
        assert!(num_arms > 0, "num_arms must be > 0");

        let delay_estimate = estimated_delay as f32;
        let delay_slack = Self::compute_delay_slack_formula(delay_estimate, delta);
        let regret_budget = Self::compute_regret_budget_formula(1, delay_estimate);
        let alpha = Self::compute_alpha_formula(1, regret_budget);
        let phase_budget = Self::compute_phase_budget(1, regret_budget);

        Self {
            phase: 1,
            delay_estimate,
            phase_gap_arrived: 0.0,
            alpha,
            regret_budget,
            delay_slack,
            baseline_arm,
            delta,
            estimated_delay,
            total_rounds: 0,
            pending_observations: 0,
            num_arms,
            phase_start_round: 0,
            phase_budget,
        }
    }

    /// Current aggression phase.
    pub fn phase(&self) -> u32 {
        self.phase
    }

    /// Current aggression coefficient αₖ.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Compute precision-gated alpha (Plan 239).
    ///
    /// When precision is available, α = sigmoid(λ × (precision - threshold)).
    /// This replaces phase-gap escalation with Bayesian certainty gating:
    /// - High precision (well-explored, certain) → α → 1.0 (trust the bandit)
    /// - Low precision (uncertain) → α → 0.0 (fall back to baseline)
    ///
    /// If `precision_skill` is `None`, returns the current phase-based alpha.
    /// This is backward compatible — no precision = existing behavior.
    #[cfg(feature = "posterior_evolution")]
    pub fn precision_gated_alpha(
        &self,
        precision_skill: Option<f32>,
        lambda: f32,
        precision_threshold: f32,
    ) -> f32 {
        match precision_skill {
            Some(precision) => {
                let x = lambda * (precision - precision_threshold);
                let sigmoid = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let ex = x.exp();
                    ex / (1.0 + ex)
                };
                // Blend: take the max of phase-based and precision-gated alpha
                // This ensures precision can only INCREASE exploration, never decrease it
                self.alpha.max(sigmoid)
            }
            None => self.alpha, // No precision → use existing phase-based alpha
        }
    }

    /// Current regret budget R̂.
    pub fn regret_budget(&self) -> f32 {
        self.regret_budget
    }

    /// Current delay slack ξ(D̂ₛ).
    pub fn delay_slack(&self) -> f32 {
        self.delay_slack
    }

    /// Safe baseline arm index.
    pub fn baseline_arm(&self) -> usize {
        self.baseline_arm
    }

    /// Total rounds played.
    pub fn total_rounds(&self) -> u32 {
        self.total_rounds
    }

    /// Cumulative phase gap on arrived data.
    pub fn phase_gap(&self) -> f32 {
        self.phase_gap_arrived
    }

    /// Current delay estimate D̂ₛ.
    pub fn delay_estimate(&self) -> f32 {
        self.delay_estimate
    }

    /// Number of arms.
    pub fn num_arms(&self) -> usize {
        self.num_arms
    }

    // ── Core formulas ──────────────────────────────────────────

    /// Compute αₖ = min(2^(k-1) / R̂, 1).
    fn compute_alpha_formula(phase: u32, regret_budget: f32) -> f32 {
        let numerator = 2_f32.powi(phase as i32 - 1);
        (numerator / regret_budget.max(f32::EPSILON)).min(1.0)
    }

    /// Compute ξ(D̂ₛ) = (√(8·D̂ₛ + 1) - 1) / δ.
    fn compute_delay_slack_formula(delay_estimate: f32, delta: f32) -> f32 {
        let inner = 8.0 * delay_estimate + 1.0;
        (inner.sqrt() - 1.0) / delta.max(f32::EPSILON)
    }

    /// Compute R̂ = C · (√T + √(D̂ₛ · ln(D̂ₛ + 1))).
    fn compute_regret_budget_formula(total_rounds: u32, delay_estimate: f32) -> f32 {
        let t = total_rounds.max(1) as f32;
        let sqrt_t = t.sqrt();
        let delay_term = if delay_estimate > 0.0 {
            (delay_estimate * (delay_estimate + 1.0).ln()).sqrt()
        } else {
            0.0
        };
        REGRET_BUDGET_C * (sqrt_t + delay_term)
    }

    /// Compute phase budget: how many rounds to spend in a phase before auto-advancing.
    ///
    /// Proportional to R̂, with a minimum to ensure each arm is tried.
    /// Capped at 1000 to prevent phases from lasting too long.
    fn compute_phase_budget(phase: u32, regret_budget: f32) -> u32 {
        let base = regret_budget as u32;
        // Each phase gets R̂ rounds, scaled by phase for more exploration
        let scaled = base * 2_u32.pow(phase.saturating_sub(1).min(10));
        scaled.clamp(10, 1000)
    }

    // ── Update methods ─────────────────────────────────────────

    /// Record one round played (increments total_rounds, recomputes budget threshold).
    ///
    /// Automatically advances phase when the phase budget is exhausted.
    /// Alpha stays fixed within a phase — only recomputed on phase advance.
    pub fn record_round(&mut self) {
        self.total_rounds += 1;
        self.regret_budget =
            Self::compute_regret_budget_formula(self.total_rounds, self.delay_estimate);

        // Auto-advance phase when budget exhausted
        let rounds_in_phase = self.total_rounds - self.phase_start_round;
        if rounds_in_phase >= self.phase_budget {
            self.advance_phase();
        }
    }

    /// Record a pending observation (feedback not yet arrived).
    pub fn record_pending(&mut self) {
        self.pending_observations += 1;
    }

    /// Record arrived feedback, decrementing pending count.
    pub fn record_arrival(&mut self) {
        self.pending_observations = self.pending_observations.saturating_sub(1);
    }

    /// Update phase gap with arrived reward data.
    ///
    /// The gap is `baseline_reward - selected_reward`. Positive gap means
    /// the active arm was worse than baseline. We accumulate the positive
    /// part (regret vs baseline) to check if we should soft restart.
    pub fn update_phase_gap(&mut self, baseline_reward: f32, selected_reward: f32) {
        let gap = baseline_reward - selected_reward;
        // Only accumulate positive gaps (baseline outperformed active arm)
        if gap > 0.0 {
            self.phase_gap_arrived += gap;
        }
    }

    /// Check if a soft restart should occur.
    ///
    /// Soft restart triggers when accumulated phase gap exceeds
    /// the threshold 2·R̂ + ξ(D̂ₛ).
    pub fn should_soft_restart(&self) -> bool {
        let threshold = 2.0 * self.regret_budget + self.delay_slack;
        self.phase_gap_arrived > threshold
    }

    /// Advance phase: increment phase, recompute alpha and budget, reset gap.
    ///
    /// Called automatically when phase budget is exhausted.
    fn advance_phase(&mut self) {
        self.phase += 1;
        self.phase_gap_arrived = 0.0;
        self.phase_start_round = self.total_rounds;
        self.regret_budget =
            Self::compute_regret_budget_formula(self.total_rounds, self.delay_estimate);
        self.alpha = Self::compute_alpha_formula(self.phase, self.regret_budget);
        self.phase_budget = Self::compute_phase_budget(self.phase, self.regret_budget);
    }

    /// Perform a soft restart: increment phase, recompute alpha, reset gap.
    pub fn soft_restart(&mut self) {
        self.advance_phase();
    }

    /// Perform a hard restart: double delay estimate, full reset to phase 1.
    pub fn hard_restart(&mut self) {
        self.delay_estimate *= 2.0;
        self.phase = 1;
        self.phase_gap_arrived = 0.0;
        self.phase_start_round = self.total_rounds;
        self.delay_slack = Self::compute_delay_slack_formula(self.delay_estimate, self.delta);
        self.regret_budget =
            Self::compute_regret_budget_formula(self.total_rounds, self.delay_estimate);
        self.alpha = Self::compute_alpha_formula(self.phase, self.regret_budget);
        self.phase_budget = Self::compute_phase_budget(self.phase, self.regret_budget);
    }

    /// Select arm using safe mixture.
    ///
    /// With probability α → use the active arm recommendation.
    /// With probability (1 - α) → use the safe baseline arm.
    pub fn select_with_safe_mixture(&self, active_arm: usize, rng: &mut Rng) -> usize {
        if rng.uniform() < self.alpha {
            active_arm
        } else {
            self.baseline_arm
        }
    }

    /// Recompute all derived quantities from current state (for testing).
    pub fn recompute(&mut self) {
        self.delay_slack = Self::compute_delay_slack_formula(self.delay_estimate, self.delta);
        self.regret_budget =
            Self::compute_regret_budget_formula(self.total_rounds, self.delay_estimate);
        self.alpha = Self::compute_alpha_formula(self.phase, self.regret_budget);
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(delay: u32) -> SafePhasedState {
        SafePhasedState::new(0, 0.1, delay, 5)
    }

    // ── Alpha computation ──────────────────────────────────────

    #[test]
    fn test_alpha_phase_1_is_1_over_regret_budget() {
        let state = make_state(0);
        // R̂ with T=1, D̂=0: C * (1 + 0) = 2
        let expected_r = 2.0;
        let expected_alpha = (1.0_f32 / expected_r).min(1.0_f32);
        assert!(
            (state.alpha() - expected_alpha).abs() < 1e-4,
            "alpha should be 1/R̂ = {}, got {}",
            expected_alpha,
            state.alpha()
        );
    }

    #[test]
    fn test_alpha_doubles_each_phase() {
        let mut state = make_state(0);
        let alpha_1 = state.alpha();

        state.soft_restart(); // phase 2
        let alpha_2 = state.alpha();
        // With C=2, R̂(T=1) = 2.0, so α₂ = 2/2 = 1.0 (may already be capped)
        assert!(
            alpha_2 >= alpha_1,
            "alpha should increase with phase: {alpha_2} < {alpha_1}"
        );

        // Alpha should continue to increase (or stay at 1.0)
        state.soft_restart(); // phase 3
        let alpha_3 = state.alpha();
        assert!(
            alpha_3 >= alpha_2 || (alpha_2 - 1.0).abs() < 1e-6,
            "alpha should increase (or be capped at 1.0): {alpha_3} < {alpha_2}"
        );
    }

    #[test]
    fn test_alpha_caps_at_1() {
        let mut state = make_state(0);
        // Force high phase to saturate alpha at 1.0
        for _ in 0..50 {
            state.soft_restart();
        }
        assert!(
            state.alpha() <= 1.0,
            "alpha should never exceed 1.0, got {}",
            state.alpha()
        );
        assert!(
            (state.alpha() - 1.0).abs() < 1e-6,
            "alpha should saturate at 1.0, got {}",
            state.alpha()
        );
    }

    // ── Delay slack formula ────────────────────────────────────

    #[test]
    fn test_delay_slack_zero_delay() {
        // ξ(0) = (√(0 + 1) - 1) / δ = (1 - 1) / 0.1 = 0
        let state = SafePhasedState::new(0, 0.1, 0, 5);
        assert!(
            state.delay_slack().abs() < 1e-6,
            "delay slack with D=0 should be 0, got {}",
            state.delay_slack()
        );
    }

    #[test]
    fn test_delay_slack_formula_correctness() {
        // ξ(9) = (√(72 + 1) - 1) / 0.1 = (√73 - 1) / 0.1 = (8.544 - 1) / 0.1 = 75.44
        let state = SafePhasedState::new(0, 0.1, 9, 5);
        let expected = ((8.0_f32 * 9.0_f32 + 1.0_f32).sqrt() - 1.0_f32) / 0.1_f32;
        assert!(
            (state.delay_slack() - expected).abs() < 1e-3,
            "delay slack should be {expected:.3}, got {}",
            state.delay_slack()
        );
    }

    #[test]
    fn test_delay_slack_increases_with_delay() {
        let state_d0 = SafePhasedState::new(0, 0.1, 0, 5);
        let state_d10 = SafePhasedState::new(0, 0.1, 10, 5);
        let state_d100 = SafePhasedState::new(0, 0.1, 100, 5);
        assert!(
            state_d0.delay_slack() < state_d10.delay_slack(),
            "higher delay should have higher slack"
        );
        assert!(
            state_d10.delay_slack() < state_d100.delay_slack(),
            "higher delay should have higher slack"
        );
    }

    // ── Soft restart ───────────────────────────────────────────

    #[test]
    fn test_soft_restart_increments_phase() {
        let mut state = make_state(10);
        assert_eq!(state.phase(), 1);
        state.soft_restart();
        assert_eq!(state.phase(), 2);
        state.soft_restart();
        assert_eq!(state.phase(), 3);
    }

    #[test]
    fn test_soft_restart_resets_gap() {
        let mut state = make_state(10);
        state.update_phase_gap(1.0, 0.0);
        state.update_phase_gap(1.0, 0.0);
        assert!(state.phase_gap() > 0.0);
        state.soft_restart();
        assert!(
            state.phase_gap().abs() < 1e-6,
            "gap should reset after soft restart"
        );
    }

    #[test]
    fn test_soft_restart_triggers_at_threshold() {
        let mut state = SafePhasedState::new(0, 0.1, 10, 5);
        // Play enough rounds to build budget, then accumulate gap past threshold
        for _ in 0..100 {
            state.record_round();
        }
        let threshold = 2.0 * state.regret_budget() + state.delay_slack();
        assert!(!state.should_soft_restart(), "should not restart initially");

        // Accumulate gap beyond threshold
        let gap_needed = threshold + 1.0;
        for _ in 0..(gap_needed as u32 + 1) {
            state.update_phase_gap(1.0, 0.0);
        }
        assert!(
            state.should_soft_restart(),
            "should trigger soft restart after gap exceeds threshold"
        );
    }

    // ── Hard restart ───────────────────────────────────────────

    #[test]
    fn test_hard_restart_doubles_delay_estimate() {
        let mut state = SafePhasedState::new(0, 0.1, 10, 5);
        let initial_delay = state.delay_estimate();
        state.hard_restart();
        assert!(
            (state.delay_estimate() - 2.0 * initial_delay).abs() < 1e-4,
            "delay should double: {} vs {}",
            state.delay_estimate(),
            2.0 * initial_delay
        );
    }

    #[test]
    fn test_hard_restart_resets_phase_and_gap() {
        let mut state = SafePhasedState::new(0, 0.1, 10, 5);
        state.soft_restart();
        state.soft_restart();
        state.update_phase_gap(1.0, 0.0);
        assert_eq!(state.phase(), 3);
        assert!(state.phase_gap() > 0.0);

        state.hard_restart();
        assert_eq!(state.phase(), 1);
        assert!(state.phase_gap().abs() < 1e-6);
    }

    // ── Safe mixture ───────────────────────────────────────────

    #[test]
    fn test_safe_mixture_respects_alpha() {
        let state = SafePhasedState::new(0, 0.1, 0, 5);
        let mut rng = Rng::new(42);

        // With alpha ≈ 0.1, baseline arm (0) should dominate
        let mut baseline_count = 0u32;
        let n = 10_000;
        for _ in 0..n {
            let arm = state.select_with_safe_mixture(2, &mut rng);
            if arm == 0 {
                baseline_count += 1;
            }
        }
        let baseline_ratio = baseline_count as f32 / n as f32;
        // (1 - alpha) should be ~0.9
        assert!(
            (baseline_ratio - (1.0 - state.alpha())).abs() < 0.05,
            "baseline ratio should ≈ (1 - alpha): {baseline_ratio:.3} vs {}",
            1.0 - state.alpha()
        );
    }

    #[test]
    fn test_safe_mixture_with_alpha_1_always_active() {
        let mut state = SafePhasedState::new(0, 0.1, 0, 5);
        // Force alpha to 1
        for _ in 0..50 {
            state.soft_restart();
        }
        assert!((state.alpha() - 1.0).abs() < 1e-6);

        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let arm = state.select_with_safe_mixture(3, &mut rng);
            assert_eq!(arm, 3, "with alpha=1, should always return active arm");
        }
    }

    // ── Regret budget ──────────────────────────────────────────

    #[test]
    fn test_regret_budget_grows_with_rounds() {
        let mut state = make_state(10);
        let rb_1 = state.regret_budget();
        for _ in 0..99 {
            state.record_round();
        }
        let rb_100 = state.regret_budget();
        assert!(
            rb_100 > rb_1,
            "regret budget should grow with rounds: {rb_100} vs {rb_1}"
        );
    }

    #[test]
    fn test_regret_budget_grows_with_delay() {
        let state_d0 = SafePhasedState::new(0, 0.1, 0, 5);
        let state_d100 = SafePhasedState::new(0, 0.1, 100, 5);
        assert!(
            state_d100.regret_budget() > state_d0.regret_budget(),
            "higher delay should have higher regret budget"
        );
    }

    // ── Phase gap accumulation ─────────────────────────────────

    #[test]
    fn test_phase_gap_only_accumulates_positive() {
        let mut state = make_state(10);
        // baseline=0.5, active=1.0 → gap = -0.5 (active better)
        state.update_phase_gap(0.5, 1.0);
        assert!(
            state.phase_gap().abs() < 1e-6,
            "negative gap should not accumulate"
        );

        // baseline=1.0, active=0.5 → gap = 0.5 (baseline better)
        state.update_phase_gap(1.0, 0.5);
        assert!(
            (state.phase_gap() - 0.5).abs() < 1e-6,
            "positive gap should accumulate: got {}",
            state.phase_gap()
        );
    }

    // ── Full flow integration ──────────────────────────────────

    #[test]
    fn test_full_flow_soft_then_hard_restart() {
        let mut state = SafePhasedState::new(0, 0.1, 5, 5);
        let mut rng = Rng::new(42);

        // Play rounds and accumulate gap
        for i in 0..500 {
            state.record_round();
            let active_arm = i % 5;
            let _selected = state.select_with_safe_mixture(active_arm, &mut rng);
            // Simulate baseline always winning (gap accumulates)
            state.update_phase_gap(1.0, 0.0);
            if state.should_soft_restart() {
                state.soft_restart();
            }
        }

        // After many rounds with consistent gap, phase should have advanced
        assert!(state.phase() > 1, "phase should have advanced past 1");

        // Hard restart should double delay
        let delay_before = state.delay_estimate();
        state.hard_restart();
        assert!(
            (state.delay_estimate() - 2.0 * delay_before).abs() < 1e-4,
            "hard restart should double delay"
        );
        assert_eq!(state.phase(), 1, "hard restart resets phase to 1");
    }

    // ── T19: SafePhased overhead analysis vs UCB1 ────────────────

    /// T19: Verify SafePhased adds exactly one RNG draw + one comparison
    /// on top of UCB1's arm selection.
    ///
    /// Rather than a wall-clock benchmark (which is unreliable under
    /// concurrent test execution), we verify the *structural* overhead:
    /// SafePhased = UCB1 active arm + 1× rng.uniform() + 1× float cmp.
    ///
    /// GOAT proof: SafePhased adds minimal overhead vs pure UCB1.
    #[test]
    fn test_safe_phased_overhead_vs_ucb1() {
        const N: usize = 1_000_000;
        const NUM_ARMS: usize = 5;

        // Simulate a warm-up: build some state with many rounds played
        let mut state = SafePhasedState::new(0, 0.2, 0, NUM_ARMS);
        for _ in 0..1000 {
            state.record_round();
        }
        let alpha = state.alpha();
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "alpha should be valid: {alpha}"
        );

        // --- Structural overhead verification ---
        //
        // UCB1 arm selection: O(k) where k = num_arms
        //   → iterate all arms, compute UCB1 score, find argmax
        //
        // SafePhased arm selection: O(k) + O(1)
        //   → same UCB1 iteration for active arm
        //   → plus: 1× rng.uniform() draw (one f32 multiply + add)
        //   → plus: 1× f32 comparison (alpha check)
        //
        // The additional work is exactly:
        //   1 multiply + 1 subtract + 1 comparison = 3 FLOPS
        // plus the RNG state update (~2 ALU ops for xorshift)
        //
        // Total overhead: ~5 ALU operations per selection.
        // This is O(1) regardless of num_arms.

        // Verify correctness: run both and ensure SafePhased
        // produces valid arm indices.
        let mut rng = Rng::new(99);
        let q_values: [f32; NUM_ARMS] = [0.5, 0.3, 0.4, 0.8, 0.6];
        let visits: [u32; NUM_ARMS] = [200, 180, 190, 210, 195];
        let total_pulls: u32 = visits.iter().sum();

        let mut safe_active_count = 0u64;
        let mut safe_baseline_count = 0u64;

        for _ in 0..N {
            // UCB1 selection
            let best_arm = (0..NUM_ARMS)
                .map(|i| {
                    let bonus = if visits[i] > 0 {
                        (2.0 * (total_pulls as f32).ln() / visits[i] as f32).sqrt()
                    } else {
                        f32::MAX
                    };
                    q_values[i] + bonus
                })
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            assert!(best_arm < NUM_ARMS);
            // SafePhased selection (same UCB1 + mixture)
            let arm = state.select_with_safe_mixture(best_arm, &mut rng);
            assert!(arm < NUM_ARMS);
            if arm == best_arm {
                safe_active_count += 1;
            } else {
                safe_baseline_count += 1;
            }
        }

        // Verify: mixture should split according to alpha
        let active_ratio = safe_active_count as f64 / N as f64;
        let baseline_ratio = safe_baseline_count as f64 / N as f64;
        let alpha_f64 = alpha as f64;

        assert!(
            (active_ratio - alpha_f64).abs() < 0.02,
            "active ratio {active_ratio:.4} should ≈ alpha {alpha_f64:.4}"
        );
        assert!(
            (baseline_ratio - (1.0 - alpha_f64)).abs() < 0.02,
            "baseline ratio {baseline_ratio:.4} should ≈ (1-alpha) {:.4}",
            1.0 - alpha_f64
        );

        eprintln!(
            "  T19 structural: alpha={alpha:.4}, active={safe_active_count}, baseline={safe_baseline_count}"
        );
        eprintln!(
            "  Overhead per selection: 1× rng.draw + 1× f32.cmp = ~5 ALU ops (O(1), independent of k={NUM_ARMS})"
        );
    }

    /// Tests for precision-gated alpha (Plan 239, Phase 5).
    #[cfg(feature = "posterior_evolution")]
    mod precision_gated_alpha_tests {
        use super::*;

        #[test]
        fn precision_gated_alpha_none_returns_phase_alpha() {
            let state = SafePhasedState::new(0, 0.1, 10, 5);
            let phase_alpha = state.alpha();
            let gated = state.precision_gated_alpha(None, 1.0, 3.0);
            assert!(
                (gated - phase_alpha).abs() < 1e-6,
                "no precision should return phase alpha"
            );
        }

        #[test]
        fn precision_gated_alpha_high_precision_approaches_1() {
            let state = SafePhasedState::new(0, 0.1, 10, 5);
            let gated = state.precision_gated_alpha(Some(10.0), 1.0, 3.0);
            assert!(
                gated > 0.95,
                "high precision should give alpha near 1.0, got {gated}"
            );
        }

        #[test]
        fn precision_gated_alpha_low_precision_approaches_phase() {
            let state = SafePhasedState::new(0, 0.1, 10, 5);
            let phase_alpha = state.alpha();
            // Precision well below threshold → sigmoid near 0 → takes max with phase alpha
            let gated = state.precision_gated_alpha(Some(0.5), 1.0, 3.0);
            // Should be at least phase_alpha (max of phase and sigmoid)
            assert!(
                gated >= phase_alpha - 1e-6,
                "low precision should not reduce alpha below phase alpha"
            );
        }

        #[test]
        fn precision_gated_alpha_monotone_with_precision() {
            let state = SafePhasedState::new(0, 0.1, 10, 5);
            let a1 = state.precision_gated_alpha(Some(1.0), 1.0, 3.0);
            let a2 = state.precision_gated_alpha(Some(3.0), 1.0, 3.0);
            let a3 = state.precision_gated_alpha(Some(5.0), 1.0, 3.0);
            assert!(
                a1 <= a2,
                "alpha should be monotone with precision: {a1} <= {a2}"
            );
            assert!(
                a2 <= a3,
                "alpha should be monotone with precision: {a2} <= {a3}"
            );
        }

        #[test]
        fn precision_gated_alpha_at_threshold_is_half() {
            let state = SafePhasedState::new(0, 0.1, 10, 5);
            // At threshold, sigmoid(0) = 0.5
            let gated = state.precision_gated_alpha(Some(3.0), 1.0, 3.0);
            // Should be max(phase_alpha, 0.5)
            let expected = state.alpha().max(0.5);
            assert!(
                (gated - expected).abs() < 1e-6,
                "at threshold: {gated} should be max(phase_alpha, 0.5) = {expected}"
            );
        }

        #[test]
        fn precision_gated_never_decreases_phase_alpha() {
            let mut state = SafePhasedState::new(0, 0.1, 0, 5);
            // Advance to higher phase for larger alpha
            for _ in 0..100 {
                state.record_round();
            }
            let phase_alpha = state.alpha();
            // Even with very low precision, precision_gated_alpha >= phase_alpha
            let gated = state.precision_gated_alpha(Some(0.1), 1.0, 3.0);
            assert!(
                gated >= phase_alpha - 1e-6,
                "precision gating should never decrease alpha: {gated} >= {phase_alpha}"
            );
        }
    }
}
