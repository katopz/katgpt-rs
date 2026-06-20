//! Unit tests for Plan 296 Phase 2 ISMCTS (T2.4).
//!
//! G2 gate fixture: a 2-card "Leduc-style" imperfect-information game where
//! the player's own card is observed but the opponent's card is hidden. The
//! belief fn samples opponent-card strengths from a parameterised posterior
//! `P(strong)`. ISMCTS over the induced CWM must pick a non-fold action
//! ≥ 70% of the time when `P(strong) = 0.7`.
//!
//! # Mock domain design
//!
//! - State: `{ my_card_strength, opp_card_strength, pot, is_terminal, winner }`.
//!   `opp_card_strength` is the hidden variable that the belief fn samples.
//! - Actions: `Fold | Call | Raise`.
//! - Forward model:
//!   - Fold → terminal, winner = opponent.
//!   - Call → if `pot < CAP`, continue (opponent's turn — modelled as a
//!     self-loop with a Call/Raise stochastic opponent in the rollout); if
//!     `pot >= CAP`, showdown (higher card wins).
//!   - Raise → `pot += 2`, continue.
//!
//! Rewards are 0/1 at terminal: 1 iff the player wins the pot. With a strong
//! own card, Raise → showdown wins against most opponent samples; Fold always
//! loses. So ISMCTS should prefer non-fold actions.
//!
//! # Why the mock IS an `InducedCwmKernel`
//!
//! `canonical_bytes()` returns a constant tag — there's only one "rule
//! schema" for this mock (Leduc-style 2-card). Phase 1's `verify_transition`
//! therefore passes trivially. The kernel-ness is structural: the state
//! carries the rules (action enum, forward model) by virtue of `impl
//! GameState`, exactly as Phase 1's `MockCounterState` does.

#![cfg(all(test, feature = "induced_cwm_ismcts"))]

use crate::induced_cwm::{BeliefInferenceFn, InducedCwmKernel};
use crate::induced_cwm::ismcts::{
    InformationSet, NodeStats, action_hash, ismcts_search_with_inference,
};
use crate::traits::GameState;

// ── Mock Leduc-style state + kernel ───────────────────────────────────────

/// Pot cap above which a Call triggers showdown. Keeps the mock game short
/// (≤ 4 plies) so rollouts terminate well within `ROLLOUT_DEPTH_CAP`.
const POT_CAP: u32 = 6;

/// A 2-card Leduc-style IIG state.
///
/// Derives `Clone, Debug, PartialEq, Eq` (Phase 1's `verify_transition`
/// requires these; ISMCTS itself requires `Hash + Eq + Clone` on `Action`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LeducState {
    /// Our own card strength (observed).
    pub my_card_strength: u8, // 0=weak (loses ties), 1=strong (wins ties)
    /// Opponent's card strength (HIDDEN from the player; sampled by belief fn).
    pub opp_card_strength: u8,
    /// Current pot in arbitrary units.
    pub pot: u32,
    /// Tick counter (drives `GameState::tick`).
    pub tick: u32,
    /// Terminal flag.
    pub is_terminal: bool,
    /// Winner player_id at terminal (0 = us, 1 = opp, 255 = unset).
    pub winner: u8,
}

impl LeducState {
    pub(crate) fn new(my_card_strength: u8, opp_card_strength: u8) -> Self {
        Self {
            my_card_strength,
            opp_card_strength,
            pot: 2, // both players ante 1
            tick: 0,
            is_terminal: false,
            winner: 255,
        }
    }

    /// Resolve a showdown: higher card wins; ties → opponent (house edge).
    fn showdown(my: u8, opp: u8) -> u8 {
        if my > opp {
            0
        } else {
            1
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LeducAction {
    Fold,
    Call,
    Raise,
}

impl GameState for LeducState {
    type Action = LeducAction;

    fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
        if self.is_terminal {
            return Vec::new();
        }
        vec![LeducAction::Fold, LeducAction::Call, LeducAction::Raise]
    }

    fn advance(&self, action: &Self::Action, _player_id: u8) -> Self {
        if self.is_terminal {
            return self.clone();
        }
        match action {
            LeducAction::Fold => LeducState {
                my_card_strength: self.my_card_strength,
                opp_card_strength: self.opp_card_strength,
                pot: self.pot,
                tick: self.tick + 1,
                is_terminal: true,
                winner: 1, // folding loses
            },
            LeducAction::Call => {
                // Model opponent's response stochastically via the (hidden)
                // opp_card_strength. If opp is strong, opp raises (pot += 2);
                // if weak, opp checks (pot += 1). Either way pot grows; once
                // pot ≥ CAP → showdown.
                let new_pot = self.pot + 1 + (self.opp_card_strength as u32);
                if new_pot >= POT_CAP {
                    let w = Self::showdown(self.my_card_strength, self.opp_card_strength);
                    LeducState {
                        my_card_strength: self.my_card_strength,
                        opp_card_strength: self.opp_card_strength,
                        pot: new_pot,
                        tick: self.tick + 1,
                        is_terminal: true,
                        winner: w,
                    }
                } else {
                    LeducState {
                        my_card_strength: self.my_card_strength,
                        opp_card_strength: self.opp_card_strength,
                        pot: new_pot,
                        tick: self.tick + 1,
                        is_terminal: false,
                        winner: 255,
                    }
                }
            }
            LeducAction::Raise => {
                let new_pot = self.pot + 2;
                if new_pot >= POT_CAP {
                    let w = Self::showdown(self.my_card_strength, self.opp_card_strength);
                    LeducState {
                        my_card_strength: self.my_card_strength,
                        opp_card_strength: self.opp_card_strength,
                        pot: new_pot,
                        tick: self.tick + 1,
                        is_terminal: true,
                        winner: w,
                    }
                } else {
                    LeducState {
                        my_card_strength: self.my_card_strength,
                        opp_card_strength: self.opp_card_strength,
                        pot: new_pot,
                        tick: self.tick + 1,
                        is_terminal: false,
                        winner: 255,
                    }
                }
            }
        }
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn reward(&self, player_id: u8) -> f32 {
        if !self.is_terminal {
            // Non-terminal partial reward: small positive bias toward higher
            // pot (encourages Raise/Call over Fold in deep rollouts that
            // don't reach terminal within the cap).
            return (self.pot as f32) / (POT_CAP as f32 * 2.0);
        }
        if self.winner == player_id {
            1.0
        } else {
            0.0
        }
    }

    fn tick(&self) -> u32 {
        self.tick
    }
}

impl InducedCwmKernel for LeducState {
    fn canonical_bytes(&self) -> Vec<u8> {
        // Constant tag: only one rule schema for this mock. Two distinct
        // LeducState values produce identical bytes (Phase 1 G4 contract).
        b"leduc_mock_v1".to_vec()
    }
}

// ── Mock belief fn ────────────────────────────────────────────────────────

/// Parameterised belief fn that samples opponent-card strength from a
/// `p_strong` posterior. With probability `p_strong` it emits a strong-opp
/// state; otherwise weak-opp. Our own card is fixed at "strong" (1) for
/// the G2 test — we want ISMCTS to recognise "I have a strong hand, don't
/// fold".
pub(crate) struct LeducBelief {
    /// Our observed card strength (carried into every sample).
    pub my_card_strength: u8,
    /// P(opp is strong | our observation).
    pub p_strong: f32,
}

impl BeliefInferenceFn<LeducState> for LeducBelief {
    type Sample = LeducState;

    fn sample(
        &self,
        _obs_history: &[LeducAction],
        _action_history: &[LeducAction],
        _player_id: u8,
        n: usize,
        seed: u64,
    ) -> Vec<Self::Sample> {
        // Deterministic given seed: use fastrand (already a katgpt-core dep)
        // to draw a proper Bernoulli(p_strong) per sample. Different seeds
        // spread draws across [0,1) so the posterior is correctly sampled
        // (the naive `(seed+i)%1000/1000` formula clusters seeds 0..N into
        // a tiny sub-range and breaks the G2 gate).
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let u = rng.f32();
            let opp = if u < self.p_strong { 1 } else { 0 };
            out.push(LeducState::new(self.my_card_strength, opp));
        }
        out
    }
}

// ── NodeStats / InformationSet unit tests ─────────────────────────────────

#[test]
fn node_stats_default_is_zero() {
    let s = NodeStats::default();
    assert_eq!(s.visits, 0);
    assert_eq!(s.total_value, 0.0);
    assert_eq!(s.mean_value(), 0.0);
}

#[test]
fn node_stats_record_accumulates() {
    let mut s = NodeStats::default();
    s.record(0.5);
    s.record(1.0);
    assert_eq!(s.visits, 2);
    assert_eq!(s.total_value, 1.5);
    assert_eq!(s.mean_value(), 0.75);
}

#[test]
fn node_stats_ucb1_unvisited_is_infinite() {
    let s = NodeStats::default();
    assert!(s.ucb1(10).is_infinite(), "unvisited edge → +∞");
}

#[test]
fn node_stats_ucb1_visited_is_finite_and_increases_with_parent_visits() {
    let mut s = NodeStats::default();
    s.record(0.5);
    let score_low_parent = s.ucb1(2);
    let score_high_parent = s.ucb1(1000);
    assert!(score_low_parent.is_finite());
    assert!(
        score_high_parent > score_low_parent,
        "more parent visits → more exploration pressure → higher UCB1"
    );
}

#[test]
fn information_set_record_and_lookup() {
    let mut set = InformationSet::with_capacity(4);
    let a = LeducAction::Raise;
    set.record(&a, 1.0);
    set.record(&a, 0.0);
    assert_eq!(set.visits_for(&a), 2);
    assert_eq!(set.mean_value_for(&a), 0.5);
    assert_eq!(set.total_visits, 2);
    // Unrecorded action → 0 visits.
    assert_eq!(set.visits_for(&LeducAction::Fold), 0);
}

#[test]
fn action_hash_is_stable() {
    let a = LeducAction::Raise;
    let b = LeducAction::Raise;
    assert_eq!(action_hash(&a), action_hash(&b));
    assert_ne!(action_hash(&LeducAction::Fold), action_hash(&LeducAction::Raise));
}

// ── G2 gate: ISMCTS picks non-fold action ≥ 70% when P(strong) = 0.7 ───────

#[test]
fn ismcts_picks_nonfold_at_least_70pct_when_strong_hand() {
    // We hold a strong card (1). Belief says opp is strong with p=0.7.
    // Even when opp is strong, our card ties with opp's (both =1); the
    // showdown rule gives ties to opp (house edge), so strong-opp = lose.
    // Against p=0.7 strong-opp:
    //   - Fold always loses (reward 0).
    //   - Raise/Call → showdown: wins with prob 0.3 (when opp is weak).
    // ISMCTS sees Fold's mean ≈ 0, Raise/Call's mean ≈ 0.3. It should pick
    // non-fold ≥ 70% of runs.
    let belief = LeducBelief { my_card_strength: 1, p_strong: 0.7 };

    let mut nonfold_count = 0usize;
    let total_runs = 10usize;
    for seed in 0..total_runs as u64 {
        let action = ismcts_search_with_inference::<LeducState, LeducBelief>(
            &[],
            &[],
            0,
            &belief,
            32, // budget = 32 belief samples per run
            seed,
        );
        if action != LeducAction::Fold {
            nonfold_count += 1;
        }
    }
    let pct = (nonfold_count as f32) / (total_runs as f32);
    assert!(
        pct >= 0.7,
        "G2 gate failed: ISMCTS picked non-fold only {pct:.0}% ({nonfold_count}/{total_runs}); expected ≥ 70%"
    );
}

#[test]
fn ismcts_deterministic_given_seed() {
    let belief = LeducBelief { my_card_strength: 1, p_strong: 0.5 };
    let a1 = ismcts_search_with_inference::<LeducState, LeducBelief>(
        &[], &[], 0, &belief, 16, 12345,
    );
    let a2 = ismcts_search_with_inference::<LeducState, LeducBelief>(
        &[], &[], 0, &belief, 16, 12345,
    );
    assert_eq!(a1, a2, "same seed + same belief → same action");
}

#[test]
fn ismcts_with_certain_strong_hand_never_folds() {
    // P(strong) = 0.0 → opp is always weak → we always win showdown.
    // ISMCTS must never fold.
    let belief = LeducBelief { my_card_strength: 1, p_strong: 0.0 };
    for seed in 0..5u64 {
        let action = ismcts_search_with_inference::<LeducState, LeducBelief>(
            &[], &[], 0, &belief, 16, seed,
        );
        assert_ne!(
            action,
            LeducAction::Fold,
            "with certain win, ISMCTS must not fold (seed={seed})"
        );
    }
}

#[test]
fn leduc_kernel_canonical_bytes_is_constant() {
    // Phase 1 G4 contract: two distinct states share canonical bytes
    // (rule-schema is the same — only one Leduc mock).
    let a = LeducState::new(1, 1);
    let b = LeducState::new(0, 0);
    assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    assert_eq!(a.commitment(), b.commitment());
    assert_eq!(a.canonical_bytes(), b"leduc_mock_v1");
}

// ── Module docs (footer) ──────────────────────────────────────────────────
//
// These tests implement the G2 (play strength) gate for Plan 296 Phase 2.
// The mock domain is intentionally minimal — a 2-card Leduc-style IIG — to
// keep the test fast and the assertion crisp. The simplified root-aggregation
// ISMCTS is sufficient for this gate; see `ismcts.rs` module docs for the
// rationale.
//
// Paper: arxiv 2510.04542 §4.3 + §B.
// Plan: katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md (Phase 2 T2.4).
