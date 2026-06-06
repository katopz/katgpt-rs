//! SDPG Bandit Go player — oracle-informed self-distilled policy gradient.
//!
//! Wraps [`GoHLPlayer`] with [`SdpgBanditPruner`] for dense per-category
//! advantage signal from oracle teacher Q-values.
//!
//! # Architecture
//!
//! ```text
//! GoSdpgPlayer
//!   ├── GoHLPlayer (heuristic 80% + bandit 20% — unchanged move selection)
//!   ├── SdpgBanditPruner (8 arms = 8 GoMoveCategory, oracle advantage)
//!   └── category_trace → SDPG update at game end
//! ```
//!
//! # Key Design Decisions
//!
//! - **Categories ARE the arms**: Unlike Bomber (template arms), Go's 8 `GoMoveCategory`
//!   values directly map to SDPG bandit arms. No template proposer layer needed.
//! - **Oracle from burn-in**: Run `GoHLPlayer` vs `GoGreedyPlayer` → extract learned
//!   category Q-values as teacher oracle → inject into SDPG.
//! - **Sigmoid advantage**: Per AGENTS.md, uses `sigmoid_advantage` by default.
//!   Per-arm `σ(teacher/τ) - σ(student/τ)`, no cross-arm normalization.
//!
//! # Plan 194

#![cfg(all(feature = "sdpg_bandit", feature = "go"))]

use std::any::Any;
use std::cmp::Ordering;

use fastrand::Rng;

use crate::pruners::bandit::{BanditPruner, BanditStrategy};
use crate::pruners::game_state::{GameState, StateHeuristic};
use crate::pruners::sdpg::{AdvantageMode, BetaSchedule, KlAnchor, SdpgBanditPruner};
use crate::speculative::types::NoScreeningPruner;

use super::players::{
    BANDIT_WEIGHT, GoHLPlayer, GoMoveCategory, GoPlayer, HEURISTIC_WEIGHT, categorize_move,
};
use super::state::{GoHeuristic, GoState};
use super::types::GoAction;

// ── Constants ──────────────────────────────────────────────────

/// Number of SDPG arms = GoMoveCategory count.
const NUM_ARMS: usize = 8;

/// ε-greedy exploration rate (matched to GoHLPlayer).
const SDPG_EPSILON: f32 = 0.15;

/// ε decay per game.
const SDPG_EPSILON_DECAY: f32 = 0.995;

/// Minimum ε floor.
const SDPG_EPSILON_FLOOR: f32 = 0.05;

/// Recency half-life for credit assignment (in moves).
const RECENCY_HALF_LIFE: f32 = 50.0;

/// Per-move reward weight for blending with game-end reward.
const PER_MOVE_ALPHA: f32 = 1.0;

/// Heuristic delta amplification.
const DELTA_AMPLIFICATION: f32 = 10.0;

// ── GoSdpgPlayer ───────────────────────────────────────────────

/// SDPG-enhanced Go player: oracle-informed self-distilled policy gradient.
///
/// Uses `GoHLPlayer`'s move categorization and heuristic scoring as base,
/// then layers SDPG's teacher-student advantage on top for denser credit
/// assignment. The 8 `GoMoveCategory` arms directly map to SDPG bandit arms.
///
/// After burn-in, the oracle teacher Q-values encode which categories
/// (Capture, CornerStar, Defend, etc.) actually win games, giving SDPG
/// a meaningful signal that Bomber's interchangeable templates lacked.
pub struct GoSdpgPlayer {
    /// Inner HL player for move categorization and heuristic scoring.
    inner: GoHLPlayer,
    /// SDPG bandit with oracle teacher Q (8 arms = 8 categories).
    sdpg_bandit: SdpgBanditPruner<NoScreeningPruner>,
    /// Exploration rate ε (decays per game).
    epsilon: f32,
    /// Trace of (category, per-move heuristic delta) for current game.
    category_trace: Vec<(GoMoveCategory, f32)>,
    /// Last arena outcome for SDPG positive-advantage gating.
    last_arena_outcome: Option<f32>,
}

impl GoSdpgPlayer {
    /// Create a new GoSdpgPlayer with uniform teacher Q (no oracle).
    pub fn new() -> Self {
        let inner = GoHLPlayer::new();
        let bandit_inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_ARMS);
        let teacher_q = vec![0.5; NUM_ARMS];
        let sdpg_bandit = SdpgBanditPruner::with_defaults(bandit_inner, teacher_q);

        Self {
            inner,
            sdpg_bandit,
            epsilon: SDPG_EPSILON,
            category_trace: Vec::new(),
            last_arena_outcome: None,
        }
    }

    /// Create GoSdpgPlayer with oracle teacher Q-values from burn-in.
    ///
    /// The teacher Q-values encode which move categories win games after
    /// a burn-in phase with `GoHLPlayer`. SDPG's sigmoid advantage will
    /// then steer the student toward the teacher's category preferences.
    pub fn with_teacher_q(teacher_q: Vec<f32>) -> Self {
        assert_eq!(
            teacher_q.len(),
            NUM_ARMS,
            "teacher_q length must match NUM_ARMS ({NUM_ARMS})"
        );

        let inner = GoHLPlayer::new();
        let bandit_inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_ARMS);
        let sdpg_bandit = SdpgBanditPruner::with_defaults(bandit_inner, teacher_q);

        Self {
            inner,
            sdpg_bandit,
            epsilon: SDPG_EPSILON,
            category_trace: Vec::new(),
            last_arena_outcome: None,
        }
    }

    /// Create GoSdpgPlayer with full SDPG config.
    pub fn with_config(
        teacher_q: Vec<f32>,
        schedule: BetaSchedule,
        anchor: KlAnchor,
        temperature: f32,
        mode: AdvantageMode,
    ) -> Self {
        assert_eq!(
            teacher_q.len(),
            NUM_ARMS,
            "teacher_q length must match NUM_ARMS ({NUM_ARMS})"
        );

        let inner = GoHLPlayer::new();
        let bandit_inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, NUM_ARMS);
        let sdpg_bandit =
            SdpgBanditPruner::new(bandit_inner, teacher_q, schedule, anchor, temperature, mode);

        Self {
            inner,
            sdpg_bandit,
            epsilon: SDPG_EPSILON,
            category_trace: Vec::new(),
            last_arena_outcome: None,
        }
    }

    /// Update bandit stats based on game outcome.
    ///
    /// Uses the same recency-weighted credit assignment as `GoHLPlayer`,
    /// but feeds rewards through SDPG's positive-advantage gating.
    pub fn update_outcome(&mut self, won: bool) {
        if self.category_trace.is_empty() {
            self.epsilon = (self.epsilon * SDPG_EPSILON_DECAY).max(SDPG_EPSILON_FLOOR);
            return;
        }

        let game_end_reward = if won { 1.0_f32 } else { 0.0 };
        let arena_outcome = if won { Some(1.0_f32) } else { Some(0.0) };
        self.last_arena_outcome = arena_outcome;

        let total = self.category_trace.len();

        // Aggregate recency-weighted rewards per category
        let mut cat_rewards = [0.0f32; NUM_ARMS];
        let mut cat_weights = [0.0f32; NUM_ARMS];
        let mut cat_last_index = [0usize; NUM_ARMS]; // track last occurrence for update order

        for (i, &(cat, per_move_reward)) in self.category_trace.iter().enumerate() {
            let recency = 0.5_f32.powf((total - 1 - i) as f32 / RECENCY_HALF_LIFE);
            let idx = cat as usize;
            let final_reward =
                PER_MOVE_ALPHA * per_move_reward + (1.0 - PER_MOVE_ALPHA) * game_end_reward;
            cat_rewards[idx] += final_reward * recency;
            cat_weights[idx] += recency;
            cat_last_index[idx] = i; // last occurrence wins
        }

        // Feed aggregated reward through SDPG bandit per category
        for idx in 0..NUM_ARMS {
            if cat_weights[idx] == 0.0 {
                continue;
            }
            let reward = cat_rewards[idx] / cat_weights[idx];
            self.sdpg_bandit.update(idx, reward, arena_outcome);
        }

        // Also update inner HL player (it has its own category trace)
        self.inner.update_outcome(won);

        self.category_trace.clear();
        self.epsilon = (self.epsilon * SDPG_EPSILON_DECAY).max(SDPG_EPSILON_FLOOR);
    }

    /// SDPG bandit Q-values (for inspection).
    pub fn sdpg_q_values(&self) -> &[f32] {
        self.sdpg_bandit.q_values()
    }

    /// Inner HL player Q-values (for inspection).
    pub fn hl_q_values(&self) -> &[f32] {
        self.inner.q_values()
    }

    /// Current β from schedule.
    pub fn beta(&self) -> f32 {
        self.sdpg_bandit.beta()
    }

    /// Current exploration rate ε.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get reference to inner SDPG bandit.
    pub fn sdpg_bandit(&self) -> &SdpgBanditPruner<NoScreeningPruner> {
        &self.sdpg_bandit
    }
}

impl Default for GoSdpgPlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl GoPlayer for GoSdpgPlayer {
    fn select_move(
        &mut self,
        state: &GoState,
        legal_moves: &[(usize, usize)],
        rng: &mut Rng,
    ) -> GoAction {
        if legal_moves.is_empty() {
            self.category_trace.push((GoMoveCategory::Pass, 0.5));
            return GoAction::Pass;
        }

        let player_id = state.to_play.player_id();
        let heuristic = GoHeuristic;
        let h_before = heuristic.evaluate(state, player_id);

        // Score and categorize each move, blending heuristic + HL bandit + SDPG bandit
        let scored: Vec<_> = legal_moves
            .iter()
            .map(|&(r, c)| {
                let cat = categorize_move(state, r, c);
                let new_state = state.advance(&GoAction::Place(r, c), player_id);
                let h_after = heuristic.evaluate(&new_state, player_id);
                let h_normalized = (h_after + 1.0) / 2.0; // [-1,1] → [0,1]

                // Three-way blend: heuristic + HL bandit + SDPG bandit
                let hl_q = self.inner.q_values()[cat as usize];
                let sdpg_q = self.sdpg_bandit.q_values()[cat as usize];

                // SDPG Q gets extra weight when β is active (teacher signal present)
                let beta = self.sdpg_bandit.beta();
                let sdpg_weight = BANDIT_WEIGHT * (1.0 + beta);
                let hl_weight = BANDIT_WEIGHT;
                let total = HEURISTIC_WEIGHT + hl_weight + sdpg_weight;

                let blended =
                    (HEURISTIC_WEIGHT * h_normalized + hl_weight * hl_q + sdpg_weight * sdpg_q)
                        / total;

                // Per-move reward: amplified heuristic delta normalized to [0, 1]
                let delta = h_after - h_before;
                let per_move_reward = (delta * DELTA_AMPLIFICATION + 1.0).clamp(0.0, 2.0) / 2.0;

                ((r, c), cat, blended, per_move_reward)
            })
            .collect();

        // ε-greedy selection
        let chosen = if rng.f32() < self.epsilon {
            scored[rng.usize(..scored.len())]
        } else {
            *scored
                .iter()
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
                .expect("scored is non-empty")
        };

        self.category_trace.push((chosen.1, chosen.3));
        GoAction::Place(chosen.0.0, chosen.0.1)
    }

    fn name(&self) -> &'static str {
        "SDPG"
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.category_trace.clear();
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_player_default_state() {
        let p = GoSdpgPlayer::new();
        assert_eq!(p.name(), "SDPG");
        assert!((p.epsilon - SDPG_EPSILON).abs() < 1e-6);
        assert!(p.category_trace.is_empty());
        assert!(p.last_arena_outcome.is_none());
        assert_eq!(p.sdpg_q_values().len(), NUM_ARMS);
    }

    #[test]
    fn with_teacher_q_asserts_length() {
        let result = std::panic::catch_unwind(|| GoSdpgPlayer::with_teacher_q(vec![1.0; 3]));
        assert!(result.is_err(), "should panic on wrong length");
    }

    #[test]
    fn with_teacher_q_correct_length() {
        let p = GoSdpgPlayer::with_teacher_q(vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]);
        assert_eq!(p.sdpg_q_values().len(), NUM_ARMS);
        // SDPG bandit starts with zero Q-values, not teacher Q
        // Teacher Q is used for advantage computation, not initial student Q
    }

    #[test]
    fn update_outcome_wins_increases_category_q() {
        let mut p = GoSdpgPlayer::with_teacher_q(vec![0.1, 0.1, 0.1, 10.0, 0.1, 0.1, 0.1, 0.1]);
        // Simulate a game with only Capture moves (arm 3)
        for _ in 0..10 {
            p.category_trace.push((GoMoveCategory::Capture, 0.8));
        }
        p.update_outcome(true);

        // After winning with Capture, SDPG should have boosted arm 3
        let q = p.sdpg_q_values();
        assert!(
            q[GoMoveCategory::Capture as usize] > 0.0,
            "Capture Q should be positive after win: {:?}",
            q
        );
    }

    #[test]
    fn update_outcome_empty_trace_no_panic() {
        let mut p = GoSdpgPlayer::new();
        p.update_outcome(false); // should not panic
        assert!((p.epsilon - SDPG_EPSILON * SDPG_EPSILON_DECAY).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_trace() {
        let mut p = GoSdpgPlayer::new();
        p.category_trace.push((GoMoveCategory::Capture, 0.5));
        p.reset();
        assert!(p.category_trace.is_empty());
    }

    #[test]
    fn select_move_returns_valid_action() {
        let mut p = GoSdpgPlayer::new();
        let state = GoState::new(9);
        let legal = state.legal_moves();
        let mut rng = Rng::new();

        let action = p.select_move(&state, &legal, &mut rng);
        match action {
            GoAction::Place(r, c) => {
                assert!(legal.contains(&(r, c)), "move ({r},{c}) should be legal");
            }
            GoAction::Pass => {
                panic!("should not pass on empty board with legal moves");
            }
        }
        assert_eq!(p.category_trace.len(), 1);
    }

    #[test]
    fn select_move_passes_when_no_legal() {
        let mut p = GoSdpgPlayer::new();
        let state = GoState::new(9);
        let mut rng = Rng::new();
        let empty: [(usize, usize); 0] = [];

        let action = p.select_move(&state, &empty, &mut rng);
        assert!(matches!(action, GoAction::Pass));
        assert_eq!(p.category_trace.len(), 1);
        assert_eq!(p.category_trace[0].0, GoMoveCategory::Pass);
    }

    #[test]
    fn default_impl_matches_new() {
        let p1 = GoSdpgPlayer::new();
        let p2 = GoSdpgPlayer::default();
        assert_eq!(p1.name(), p2.name());
        assert!((p1.epsilon() - p2.epsilon()).abs() < 1e-6);
    }

    #[test]
    fn epsilon_decays_after_outcome() {
        let mut p = GoSdpgPlayer::new();
        let initial_eps = p.epsilon();
        // Empty trace still decays epsilon
        p.update_outcome(true);
        assert!(p.epsilon() < initial_eps, "epsilon should decay");
    }

    #[test]
    fn beta_decreases_with_schedule() {
        let mut p = GoSdpgPlayer::with_teacher_q(vec![0.9; NUM_ARMS]);
        // Advance through warmup to reach peak beta
        for _ in 0..100 {
            p.category_trace.push((GoMoveCategory::Capture, 0.5));
            p.update_outcome(true);
        }
        let peak_beta = p.beta();
        assert!(
            peak_beta > 0.0,
            "beta should be positive at peak: {peak_beta}"
        );
        // Continue through decay to reach zero
        for _ in 0..1000 {
            p.category_trace.push((GoMoveCategory::Capture, 0.5));
            p.update_outcome(true);
        }
        let final_beta = p.beta();
        assert!(
            final_beta < peak_beta,
            "beta should decrease after peak: {peak_beta} -> {final_beta}"
        );
    }
}
