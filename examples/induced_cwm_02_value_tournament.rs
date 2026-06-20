//! Plan 296 Phase 3 T3.5: value-function tournament smoke example.
//!
//! Demonstrates the [`ValueFnTournament`] end-to-end:
//!   1. A mock induced CWM (`RaceState`) — a 2-player "race to N" game.
//!   2. Three candidate heuristics wrapped in a single enum so they can
//!      coexist in `Vec<RaceHeuristic>` (the tournament's `candidates` field
//!      requires a single type).
//!   3. A weak baseline closure (always Stall).
//!   4. The tournament runs round-robin (candidate-vs-baseline +
//!      candidate-vs-candidate head-to-head) and prints the ranking plus the
//!      head-to-head matrix.
//!
//! Run with:
//!   cargo run --example induced_cwm_02_value_tournament --features induced_cwm_tournament
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! Heuristic outputs are scalars (raw). The tournament is offline / search-
//! local; `PlayerStats` and `head_to_head` are scratch, not synced.
//!
//! Paper: arxiv 2510.04542 (Lehrach et al., DeepMind Oct 2025), §4.4 + §C.
//! Plan: katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md (Phase 3 T3.5).

use katgpt_core::induced_cwm::{InducedCwmKernel, ValueFnTournament};
use katgpt_core::traits::{GameState, StateHeuristic};

// ── Mock induced CWM — same race-to-N as the unit tests ────────────────────
//
// Players alternate. On their turn they pick Advance (counter += 1) or
// Stall (counter += 0). First to push the shared counter to GOAL wins.
// GOAL > 2 * MCTS_ROLLOUT_DEPTH_CAP forces the MCTS leaf evaluation to come
// from the heuristic, making the tournament actually discriminate between
// heuristics (otherwise pure MCTS solves the game and all heuristics tie).

#[derive(Clone, Debug, PartialEq, Eq)]
struct RaceState {
    counter: u32,
    turn: u8,
    tick: u32,
    is_terminal: bool,
    winner: u8,
}

impl RaceState {
    const GOAL: u32 = 25;

    fn new() -> Self {
        Self { counter: 0, turn: 0, tick: 0, is_terminal: false, winner: 255 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RaceAction {
    Advance,
    Stall,
}

impl GameState for RaceState {
    type Action = RaceAction;

    fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
        if self.is_terminal {
            return Vec::new();
        }
        vec![RaceAction::Advance, RaceAction::Stall]
    }

    fn advance(&self, action: &Self::Action, _player_id: u8) -> Self {
        if self.is_terminal {
            return self.clone();
        }
        let delta = match action {
            RaceAction::Advance => 1,
            RaceAction::Stall => 0,
        };
        let new_counter = self.counter + delta;
        let next_turn = 1 - self.turn;
        let (is_terminal, winner) = if new_counter >= Self::GOAL {
            (true, self.turn)
        } else {
            (false, 255)
        };
        RaceState {
            counter: new_counter,
            turn: next_turn,
            tick: self.tick + 1,
            is_terminal,
            winner,
        }
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn reward(&self, player_id: u8) -> f32 {
        if !self.is_terminal {
            return (self.counter as f32) / (Self::GOAL as f32 * 2.0);
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

impl InducedCwmKernel for RaceState {
    fn canonical_bytes(&self) -> Vec<u8> {
        b"race_mock_v1".to_vec()
    }
}

// ── Candidate heuristics ───────────────────────────────────────────────────
//
// All three wrapped in `RaceHeuristic` so they share one Rust type.

#[derive(Clone, Copy, Debug, Default)]
struct AdvanceHeuristic;
impl StateHeuristic<RaceState> for AdvanceHeuristic {
    fn evaluate(&self, s: &RaceState, _pid: u8) -> f32 {
        (s.counter as f32) / (RaceState::GOAL as f32)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StallHeuristic;
impl StateHeuristic<RaceState> for StallHeuristic {
    fn evaluate(&self, s: &RaceState, _pid: u8) -> f32 {
        1.0 - (s.counter as f32) / (RaceState::GOAL as f32)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ConstantHeuristic;
impl StateHeuristic<RaceState> for ConstantHeuristic {
    fn evaluate(&self, _s: &RaceState, _pid: u8) -> f32 {
        0.5
    }
}

#[derive(Clone, Copy, Debug)]
enum RaceHeuristic {
    Constant(ConstantHeuristic),
    Advance(AdvanceHeuristic),
    Stall(StallHeuristic),
}

impl StateHeuristic<RaceState> for RaceHeuristic {
    fn evaluate(&self, s: &RaceState, pid: u8) -> f32 {
        match self {
            RaceHeuristic::Constant(h) => h.evaluate(s, pid),
            RaceHeuristic::Advance(h) => h.evaluate(s, pid),
            RaceHeuristic::Stall(h) => h.evaluate(s, pid),
        }
    }
}

impl RaceHeuristic {
    fn name(&self) -> &'static str {
        match self {
            RaceHeuristic::Constant(_) => "Constant",
            RaceHeuristic::Advance(_) => "Advance",
            RaceHeuristic::Stall(_) => "Stall",
        }
    }
}

// ── Baseline: always Stall ─────────────────────────────────────────────────

fn stall_baseline(_state: &RaceState, _pid: u8) -> RaceAction {
    RaceAction::Stall
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 296 Phase 3: Value Function Tournament smoke example ===\n");

    let candidates = vec![
        RaceHeuristic::Constant(ConstantHeuristic),
        RaceHeuristic::Advance(AdvanceHeuristic),
        RaceHeuristic::Stall(StallHeuristic),
    ];
    let candidate_names: Vec<&'static str> = candidates.iter().map(|c| c.name()).collect();

    println!("Mock induced CWM: race-to-{} (turn-alternating, shared counter)", RaceState::GOAL);
    println!("Baseline       : always Stall");
    println!("Candidates     : {:?}", candidate_names);
    println!("MCTS budget    : 24 iterations/move");
    println!("Games per match: 4 per (candidate, role) vs baseline; 4 per head-to-head pair");
    println!();

    let tournament = ValueFnTournament::new(candidates, 4, 42, 24).with_ply_cap(80);
    let winner = tournament.run(&RaceState::new(), &stall_baseline);

    // ── Per-candidate stats vs baseline ───────────────────────────────
    println!("Per-candidate stats vs Stall baseline:");
    println!(
        "  {:<10} {:>4} {:>4} {:>4} {:>10} {:>12}",
        "candidate", "W", "L", "D", "win_rate", "avg_reward"
    );
    println!("  {}", "-".repeat(50));
    for (i, name) in candidate_names.iter().enumerate() {
        let s = winner.vs_baseline[i];
        println!(
            "  {:<10} {:>4} {:>4} {:>4} {:>9.1}% {:>12.3}",
            name,
            s.wins,
            s.losses,
            s.draws,
            s.win_rate() * 100.0,
            s.avg_reward(),
        );
    }
    println!();

    // ── Head-to-head matrix ───────────────────────────────────────────
    println!("Head-to-head win-rate matrix (rows = candidate, cols = opponent):");
    print!("            ");
    for name in &candidate_names {
        print!("{:>10} ", name);
    }
    println!();
    for (i, name) in candidate_names.iter().enumerate() {
        print!("  {:<10} ", name);
        for (j, _opp_name) in candidate_names.iter().enumerate() {
            if i == j {
                print!("       --- ");
            } else {
                print!("{:>9.1}% ", winner.head_to_head[i][j] * 100.0);
            }
        }
        println!();
    }
    println!();

    // ── Winner ────────────────────────────────────────────────────────
    println!(
        "Winner: {} (idx {}) — win_rate vs baseline = {:.1}%",
        candidate_names[winner.winner_idx],
        winner.winner_idx,
        winner.vs_baseline[winner.winner_idx].win_rate() * 100.0
    );

    // Sanity assertion: AdvanceHeuristic (idx 1) should win.
    assert_eq!(
        winner.winner_idx, 1,
        "AdvanceHeuristic should win; got {}",
        candidate_names[winner.winner_idx]
    );
    println!("\nSanity assertion OK: AdvanceHeuristic wins the tournament.");
    println!("See .plans/296_induced_cwm_kernel_primitive.md for the plan.");
}
