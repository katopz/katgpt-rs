//! Plan 296 Phase 2 T2.5: smoke example for the induced CWM + ISMCTS primitive.
//!
//! Demonstrates the end-to-end lifecycle:
//!   1. A mock induced CWM (`MockGridState`) — 3 hidden states, 4 actions.
//!   2. A mock belief fn that returns 1–3 samples from a small support.
//!   3. ISMCTS picks an action given an observation/action history.
//!   4. Prints the chosen action and root information-set statistics.
//!
//! Run with:
//!   cargo run --example induced_cwm_01_mock_iig --features induced_cwm_ismcts
//!
//! # Latent vs raw boundary
//!
//! The hidden states sampled by the belief fn are latent (local to the
//! search). The chosen `MockGridAction` is the raw, syncable output. Nothing
//! latent crosses the example boundary.
//!
//! Paper: arxiv 2510.04542 (Lehrach et al., DeepMind Oct 2025), §4.2 + §4.3.
//! Plan: katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md (Phase 2 T2.5).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use katgpt_core::induced_cwm::{
    BeliefInferenceFn, InducedCwmKernel, InformationSet, NodeStats,
    ismcts_search_with_inference,
};
use katgpt_core::traits::GameState;

// ── Mock induced CWM — a tiny grid-world IIG ──────────────────────────────
//
// The player is on a 1-D strip and must move. The hidden state is which
// "exit" cell is open (only one is open per game). The player observes
// nothing about the exit — the belief fn samples plausible exit positions.
// Moving toward the open exit is good; moving away is bad.

#[derive(Clone, Debug, PartialEq)]
struct MockGridState {
    /// Player position on a 0..=3 strip.
    player_pos: u8,
    /// Hidden: which cell (0..=2) has the open exit. 3 = "no exit" sentinel.
    /// Sampled by the belief fn — never observed by the player.
    open_exit: u8,
    /// Tick counter.
    tick: u32,
    /// Terminal flag.
    is_terminal: bool,
    /// Reward at terminal: 1.0 if player ended on the open exit.
    final_reward: f32,
}

impl MockGridState {
    fn new(player_pos: u8, open_exit: u8) -> Self {
        Self {
            player_pos,
            open_exit,
            tick: 0,
            is_terminal: false,
            final_reward: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MockGridAction {
    Left,
    Right,
    Wait,
    Exit, // Commit to current cell as the exit.
}

impl GameState for MockGridState {
    type Action = MockGridAction;

    fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
        if self.is_terminal {
            return Vec::new();
        }
        vec![MockGridAction::Left, MockGridAction::Right, MockGridAction::Wait, MockGridAction::Exit]
    }

    fn advance(&self, action: &Self::Action, _player_id: u8) -> Self {
        if self.is_terminal {
            return self.clone();
        }
        match action {
            MockGridAction::Left => {
                let new_pos = self.player_pos.saturating_sub(1);
                Self::step(self, new_pos)
            }
            MockGridAction::Right => {
                let new_pos = (self.player_pos + 1).min(3);
                Self::step(self, new_pos)
            }
            MockGridAction::Wait => Self::step(self, self.player_pos),
            MockGridAction::Exit => {
                let reward = if self.player_pos == self.open_exit { 1.0 } else { 0.0 };
                MockGridState {
                    player_pos: self.player_pos,
                    open_exit: self.open_exit,
                    tick: self.tick + 1,
                    is_terminal: true,
                    final_reward: reward,
                }
            }
        }
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn reward(&self, _player_id: u8) -> f32 {
        if self.is_terminal {
            self.final_reward
        } else {
            // Non-terminal partial reward: closer to the open exit is better.
            // Use 1 / (1 + distance) — bounded (0, 1].
            let dist = (self.player_pos as i32 - self.open_exit as i32).unsigned_abs() as f32;
            1.0 / (1.0 + dist)
        }
    }

    fn tick(&self) -> u32 {
        self.tick
    }
}

impl MockGridState {
    fn step(&self, new_pos: u8) -> Self {
        MockGridState {
            player_pos: new_pos,
            open_exit: self.open_exit,
            tick: self.tick + 1,
            is_terminal: false,
            final_reward: 0.0,
        }
    }
}

impl InducedCwmKernel for MockGridState {
    fn canonical_bytes(&self) -> Vec<u8> {
        // Constant tag — only one rule schema for this mock. Two distinct
        // MockGridState values produce identical bytes (Phase 1 G4 contract).
        b"mock_grid_iig_v1".to_vec()
    }
}

// ── Mock belief fn ────────────────────────────────────────────────────────

/// Belief fn that samples plausible hidden states (open_exit positions).
///
/// `support` defines the plausible exits; the fn returns up to `n` of them
/// (cycling through the support) so each call returns 1–3 distinct samples.
struct MockGridBelief {
    /// Plausible open_exit positions, in the order they'll be emitted.
    support: Vec<u8>,
    /// Player's currently observed position (carried into every sample).
    observed_player_pos: u8,
}

impl BeliefInferenceFn<MockGridState> for MockGridBelief {
    type Sample = MockGridState;

    fn sample(
        &self,
        _obs_history: &[MockGridAction],
        _action_history: &[MockGridAction],
        _player_id: u8,
        n: usize,
        seed: u64,
    ) -> Vec<Self::Sample> {
        // Deterministic given seed: rotate the support by `seed % len` so
        // different seeds explore different orderings.
        if self.support.is_empty() {
            return Vec::new();
        }
        let count = n.min(self.support.len());
        let offset = (seed as usize) % self.support.len();
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let exit = self.support[(offset + i) % self.support.len()];
            out.push(MockGridState::new(self.observed_player_pos, exit));
        }
        out
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Stable u64 hash of an action (mirrors `ismcts::action_hash` — duplicated
/// here so the example doesn't reach into a private helper signature that
/// might change).
fn action_hash<A: Hash>(a: &A) -> u64 {
    let mut h = DefaultHasher::new();
    a.hash(&mut h);
    h.finish()
}

/// Run one ISMCTS search manually (mirroring `ismcts_search_with_inference`)
/// so we can inspect the root info-set stats after the call. The real search
/// returns only the chosen action; this helper re-runs the aggregation loop
/// and exposes the table for printing.
fn run_ismcts_and_collect_stats(
    belief: &MockGridBelief,
    player_id: u8,
    budget: usize,
    rng_seed: u64,
) -> (MockGridAction, InformationSet, std::collections::HashMap<u64, MockGridAction>) {
    use fastrand::Rng;

    let mut rng = Rng::with_seed(rng_seed);
    let mut root_set = InformationSet::with_capacity(4);
    let mut action_scratch: Vec<MockGridAction> = Vec::with_capacity(4);
    let mut rollout_scratch: Vec<MockGridAction> = Vec::with_capacity(4);
    let mut hash_to_action: std::collections::HashMap<u64, MockGridAction> =
        std::collections::HashMap::with_capacity(4);

    for iteration in 0..budget {
        let belief_seed = rng_seed.wrapping_add(iteration as u64);
        let samples = belief.sample(&[], &[], player_id, 1, belief_seed);
        let Some(sample) = samples.into_iter().next() else { continue };

        sample.available_actions_into(player_id, &mut action_scratch);
        if action_scratch.is_empty() {
            continue;
        }

        for action in action_scratch.iter() {
            let key = action_hash(action);
            hash_to_action.entry(key).or_insert_with(|| action.clone());

            let child = sample.advance(action, player_id);
            let reward = if child.is_terminal() {
                child.reward(player_id)
            } else {
                // Inline short rollout (mirrors ismcts::random_rollout).
                let mut cur = child.clone();
                for _ in 0..10 {
                    if cur.is_terminal() {
                        break;
                    }
                    cur.available_actions_into(player_id, &mut rollout_scratch);
                    if rollout_scratch.is_empty() {
                        break;
                    }
                    let pick = rng.usize(0..rollout_scratch.len());
                    let a = rollout_scratch[pick].clone();
                    cur = cur.advance(&a, player_id);
                }
                cur.reward(player_id)
            };
            root_set.record(action, reward);
        }
    }

    // Pick best action by visits, then mean value, then hash (tertiary
    // tie-break makes the example reproducible regardless of HashMap
    // iteration order).
    let best_hash = root_set
        .edges
        .iter()
        .max_by(|(ka, va), (kb, vb)| match va.visits.cmp(&vb.visits) {
            std::cmp::Ordering::Equal => {
                match va
                    .mean_value()
                    .partial_cmp(&vb.mean_value())
                    .unwrap_or(std::cmp::Ordering::Equal)
                {
                    std::cmp::Ordering::Equal => ka.cmp(kb),
                    ord => ord,
                }
            }
            ord => ord,
        })
        .map(|(k, _)| *k)
        .expect("no edges recorded");
    let best_action = hash_to_action.get(&best_hash).cloned().expect("missing action");

    (best_action, root_set, hash_to_action)
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 296 Phase 2: Induced CWM + ISMCTS smoke example ===\n");

    // 1. Mock induced CWM: player starts at position 1; plausible exits at
    //    cells {0, 2}. (3 hidden states total when we include the player's
    //    own position as part of the state tuple.)
    let belief = MockGridBelief {
        support: vec![0, 2],
        observed_player_pos: 1,
    };

    println!("Mock induced CWM:");
    println!("  player_pos     = {}", belief.observed_player_pos);
    println!("  open_exit ∈    = {:?}  (hidden, sampled by belief)", belief.support);
    println!("  actions        = [Left, Right, Wait, Exit]");
    println!("  canonical tag  = \"mock_grid_iig_v1\"");
    println!();

    // 2. Sample a few hidden states from the belief fn (demonstrates the
    //    1–3 sample range the spec calls for).
    println!("Belief samples (n=3, seed=42):");
    for (i, s) in belief.sample(&[], &[], 0, 3, 42).iter().enumerate() {
        println!("  [{}] player_pos={} open_exit={} (HIDDEN)", i, s.player_pos, s.open_exit);
    }
    println!();

    // 3. Run ISMCTS to pick an action — via the real public API.
    let budget = 24usize;
    let seed = 42u64;
    let player_id = 0u8;
    let chosen = ismcts_search_with_inference::<MockGridState, MockGridBelief>(
        &[], &[], player_id, &belief, budget, seed,
    );

    // Also run the internal aggregation loop (via the helper) so we can
    // inspect and print the root information-set statistics. Both runs use
    // the same seed, so they explore the same determinizations.
    let (chosen_from_stats, root_set, hash_to_action) =
        run_ismcts_and_collect_stats(&belief, player_id, budget, seed);
    assert_eq!(
        chosen, chosen_from_stats,
        "public API and stats helper must agree given the same seed"
    );

    println!("ISMCTS search (budget={budget}, seed={seed}, player_id={player_id}):");
    println!("  chosen action  = {:?}", chosen);
    println!();

    // 4. Print the root information-set statistics.
    println!("Root information-set statistics:");
    println!("  {:<10} {:>8} {:>12} {:>10}", "action", "visits", "total_value", "mean");
    println!("  {}", "-".repeat(44));

    // Print in a stable action order so output is reproducible.
    let mut entries: Vec<(MockGridAction, NodeStats)> = root_set
        .edges
        .iter()
        .map(|(k, v)| (hash_to_action.get(k).cloned().unwrap(), *v))
        .collect();
    entries.sort_by_key(|(a, _)| format!("{:?}", a));

    for (action, stats) in &entries {
        println!(
            "  {:<10} {:>8} {:>12.4} {:>10.4}",
            format!("{:?}", action),
            stats.visits,
            stats.total_value,
            stats.mean_value()
        );
    }
    println!("  {}", "-".repeat(44));
    println!("  {:<10} {:>8}", "TOTAL", root_set.total_visits);
    println!();

    // Sanity assertions.
    assert!(root_set.total_visits > 0, "search must have visited at least one edge");
    assert!(entries.iter().any(|(a, _)| *a == chosen), "chosen must be in stats");
    // Exit and Wait should generally have lower mean than directional moves
    // when the player doesn't know where the exit is — but we don't assert
    // that strictly; the smoke is just to confirm the pipeline runs.
    println!("All sanity assertions OK. See .plans/296_induced_cwm_kernel_primitive.md for the plan.");
}
