//! Value Function Tournament over an induced CWM (Plan 296 Phase 3).
//!
//! Paper §4.4 + §C (arxiv 2510.04542): once an induced CWM is verifiable and
//! committable, the play-strength GOAT gate (G2') needs a way to *select*
//! among candidate value functions (heuristics). The paper uses an
//! arena-play tournament — round-robin matches with a fixed baseline — and
//! ranks candidates by win-rate-vs-baseline, tie-broken by head-to-head.
//!
//! This module ships the **generic, IP-free half** of that idea:
//!
//! - [`ValueFnTournament`] — round-robin arena-play selector over
//!   [`StateHeuristic`] candidates.
//! - [`PlayerStats`] — per-candidate win/loss/draw + avg reward.
//! - [`TournamentWinner`] — wrapper around the winning candidate + full
//!   ranking, returned from [`ValueFnTournament::run`].
//!
//! # Algorithm
//!
//! 1. Each candidate heuristic `H_i` is wrapped into an MCTS-based policy
//!    `policy_i(state, pid) -> Action` that uses `H_i.evaluate(...)` as the
//!    MCTS leaf evaluation.
//! 2. For every candidate `i`:
//!    - Play `games_per_match` games as player 0 vs `baseline`, with the
//!      candidate moving first; then `games_per_match` games as player 1,
//!      with the baseline moving first.
//!    - Record wins/losses/draws and cumulative reward into `PlayerStats[i]`.
//! 3. Round-robin: for every pair `(i, j)`, `i != j`, play
//!    `games_per_match` games head-to-head (each candidate moves first in
//!    half the games). Record into a separate head-to-head matrix.
//! 4. Winner = highest win-rate-vs-baseline; tie-break by head-to-head
//!    win-rate.
//!
//! The baseline is a plain closure `Fn(&S, u8) -> S::Action` so callers can
//! plug in random play, a fixed policy, or a heuristic-of-record. The
//! tournament measures "MCTS-with-heuristic-H vs baseline" — i.e. it
//! isolates the *heuristic*'s contribution to play strength, since every
//! candidate uses the same search budget and the same induced CWM.
//!
//! # Self-containment (mirrors Phase 2)
//!
//! This module lives in `katgpt-core`, so it CANNOT import
//! `katgpt-rs/src/pruners/game_state/mcts.rs` (the root crate depends on
//! `katgpt-core`, which would be a circular dep — same constraint Phase 2's
//! ISMCTS hit). A minimal MCTS is mirrored here. It is intentionally simpler
//! than the root `mcts_search`: no transposition table, no warm-start. The
//! tournament only needs *relative* play strength between heuristics, so
//! absolute search quality is irrelevant as long as it's held constant
//! across candidates.
//!
//! # Deviation from Plan 296 §T3.1
//!
//! The plan signature was:
//! ```text
//! pub fn run<K: InducedCwmKernel>(&self, kernel: &K, baseline: ...) -> TournamentWinner<V>
//! ```
//! Dropped the `kernel: &K` parameter — it's redundant with `initial_state:
//! &S` because under the codebase's `GameState` convention (Phase 1 T1.6
//! deviation), the state IS the kernel. `S: InducedCwmKernel` is the type
//! bound; the *value* of `initial_state` is what gets cloned to start each
//! match. Matches Phase 1's `verify_transition(test: &TransitionUnitTest<S>)`
//! and Phase 2's `ismcts_search_with_inference<S, B>(...)` shape.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - `StateHeuristic::evaluate(...)` outputs are scalar (raw) — they cross
//!   no sync boundary because the whole tournament is offline / search-local.
//! - The chosen `S::Action` per playout is raw and (in principle) syncable.
//! - `PlayerStats` and `TournamentWinner` are search-local scratch — they
//!   are not synced. They may be *logged* (audit cadence) for the GOAT gate
//!   proof file.
//!
//! # References
//!
//! - Plan: [`crate::induced_cwm`] §Phase 3
//! - Source paper: [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542) §4.4, §C
//! - Phase 1 kernel trait: [`crate::induced_cwm::InducedCwmKernel`]
//! - Phase 2 ISMCTS cousin: [`crate::induced_cwm::ismcts`]
//! - `StateHeuristic` trait: [`crate::traits::StateHeuristic`]

use fastrand::Rng;

use crate::induced_cwm::InducedCwmKernel;
use crate::traits::{GameState, StateHeuristic};

/// Cap on plies per match before the game is declared a draw.
///
/// Keeps the tournament bounded — even if a heuristic pair produces
/// non-terminating play patterns (e.g. both players Wait forever), the
/// match ends after this many plies and is scored as a draw. Generous
/// relative to the mock IIGs (≤ 4 plies); real domains can override via
/// [`ValueFnTournament::with_ply_cap`].
const DEFAULT_PLY_CAP: u32 = 64;

/// UCB1 exploration constant — `sqrt(2)`, matching `mcts.rs` and
/// `induced_cwm/ismcts.rs`. Kept as a `const` so search is byte-stable
/// across re-runs (required for tournament reproducibility).
const UCB1_C: f32 = 1.414;

/// MCTS rollout depth cap for the per-candidate policy. Matches
/// `ROLLOUT_DEPTH_CAP` in `induced_cwm/ismcts.rs`.
const MCTS_ROLLOUT_DEPTH_CAP: usize = 10;

// ── T3.2 — PlayerStats ────────────────────────────────────────────────────

/// Per-candidate statistics accumulated over a tournament.
///
/// Win = candidate's `reward(0 or 1) > opponent's reward` at terminal. Draw
/// = equal rewards at terminal OR match hit the ply cap without terminating.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerStats {
    /// Games won (candidate's terminal reward strictly greater than opponent's).
    pub wins: u32,
    /// Games lost.
    pub losses: u32,
    /// Games drawn (equal rewards, or ply-cap reached without terminal).
    pub draws: u32,
    /// Sum of the candidate's terminal rewards across all games (win=1,
    /// loss=0, draw=0.5 — same convention as chess ELO).
    pub score: f32,
}

impl PlayerStats {
    /// Number of games this candidate played.
    #[inline]
    pub fn games(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    /// Win-rate as a fraction in `[0.0, 1.0]`. Draws count as half a win
    /// (chess convention).
    #[inline]
    pub fn win_rate(&self) -> f32 {
        let n = self.games();
        if n == 0 {
            return 0.0;
        }
        (self.wins as f32 + 0.5 * self.draws as f32) / n as f32
    }

    /// Average terminal reward across all games played.
    #[inline]
    pub fn avg_reward(&self) -> f32 {
        let n = self.games();
        if n == 0 {
            return 0.0;
        }
        self.score / n as f32
    }

    /// Record one match outcome from the candidate's perspective.
    #[inline]
    fn record(&mut self, candidate_reward: f32, opponent_reward: f32) {
        // Chess-style scoring: win=1, draw=0.5, loss=0. Terminal reward is
        // already in [0,1] per the GameState contract, so we use the strict
        // inequality for win/loss and treat equality as a draw.
        if candidate_reward > opponent_reward {
            self.wins += 1;
            self.score += 1.0;
        } else if candidate_reward < opponent_reward {
            self.losses += 1;
        } else {
            self.draws += 1;
            self.score += 0.5;
        }
    }
}

impl std::fmt::Display for PlayerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:>3}W {:>3}L {:>3}D  win_rate={:>5.1}%  avg_reward={:>5.3}",
            self.wins,
            self.losses,
            self.draws,
            self.win_rate() * 100.0,
            self.avg_reward(),
        )
    }
}

// ── T3.1 — TournamentWinner ───────────────────────────────────────────────

/// Outcome of a tournament: the winning candidate (by win-rate-vs-baseline,
/// tie-broken by head-to-head) and the full per-candidate statistics.
#[derive(Clone, Debug)]
pub struct TournamentWinner<'a, S, V>
where
    S: GameState,
    V: StateHeuristic<S>,
{
    /// Index into `ValueFnTournament::candidates` of the winning candidate.
    pub winner_idx: usize,
    /// Borrow back to the winning heuristic (borrowed from the tournament).
    pub winner: &'a V,
    /// Per-candidate stats, indexed parallel to `candidates`. The
    /// `vs_baseline` slice is the primary ranking signal; `head_to_head` is
    /// the tie-break matrix (`head_to_head[i][j]` = candidate `i`'s win-rate
    /// vs candidate `j`).
    pub vs_baseline: Vec<PlayerStats>,
    /// `head_to_head[i][j]` = candidate `i`'s win-rate (in `[0,1]`) against
    /// candidate `j`. Diagonal is `0.0` (a candidate doesn't play itself).
    pub head_to_head: Vec<Vec<f32>>,
    /// Phantom — `V` is implied by `candidates`, but the borrow chain needs
    /// it spelled out so the lifetime is clear.
    _phantom: std::marker::PhantomData<S>,
}

impl<'a, S, V> TournamentWinner<'a, S, V>
where
    S: GameState,
    V: StateHeuristic<S>,
{
    fn new(
        winner_idx: usize,
        winner: &'a V,
        vs_baseline: Vec<PlayerStats>,
        head_to_head: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            winner_idx,
            winner,
            vs_baseline,
            head_to_head,
            _phantom: std::marker::PhantomData,
        }
    }
}

// ── T3.1 — ValueFnTournament ──────────────────────────────────────────────

/// Round-robin arena-play tournament that selects the best
/// [`StateHeuristic`] for a given induced CWM.
///
/// Construct with [`ValueFnTournament::new`], optionally tune with
/// [`ValueFnTournament::with_ply_cap`], then call [`Self::run`] to play it
/// out.
///
/// # Reproducibility
///
/// The same `(candidates, games_per_match, rng_seed, initial_state)`
/// quintuple MUST produce the same `TournamentWinner.winner_idx` and
/// (within reason) the same `PlayerStats`. This is required so the GOAT
/// gate (G2' — heuristic selection) is reproducible. Reproducibility is
/// achieved by seeding `fastrand::Rng` from `rng_seed` plus per-match
/// offsets derived from `(candidate_idx, role, match_idx)` — no global
/// mutable RNG.
///
/// # Type bounds
///
/// * `S: GameState + InducedCwmKernel` — the kernel/state type.
/// * `S::Action: Clone + Debug` — needed for the in-tree MCTS (action
///   selection, debugging on overflow).
/// * `V: StateHeuristic<S>` — candidate value function.
#[derive(Clone, Debug)]
pub struct ValueFnTournament<S, V>
where
    S: GameState,
    V: StateHeuristic<S>,
{
    /// Candidate heuristics. Indexed `0..n`.
    pub candidates: Vec<V>,
    /// Games each candidate plays vs `baseline` in each role (player 0 and
    /// player 1). Total baseline games per candidate = `2 * games_per_match`.
    pub games_per_match: usize,
    /// RNG seed. Same seed + same inputs → same winner.
    pub rng_seed: u64,
    /// Maximum plies per match before declaring a draw.
    pub ply_cap: u32,
    /// MCTS budget per move. Held constant across candidates so the
    /// tournament measures *heuristic quality*, not *search depth*.
    pub mcts_budget: usize,
    _phantom: std::marker::PhantomData<S>,
}

impl<S, V> ValueFnTournament<S, V>
where
    S: GameState + InducedCwmKernel,
    S::Action: std::fmt::Debug,
    V: StateHeuristic<S>,
{
    /// Construct a new tournament.
    ///
    /// # Arguments
    ///
    /// * `candidates` — the heuristics to rank. Must be non-empty (calling
    ///   [`run`](Self::run) on an empty tournament panics).
    /// * `games_per_match` — games per (candidate, role) vs baseline, and
    ///   per (candidate, candidate, first-mover) in head-to-head.
    /// * `rng_seed` — base RNG seed.
    /// * `mcts_budget` — MCTS iterations per move. Same for every candidate.
    pub fn new(candidates: Vec<V>, games_per_match: usize, rng_seed: u64, mcts_budget: usize) -> Self {
        Self {
            candidates,
            games_per_match,
            rng_seed,
            ply_cap: DEFAULT_PLY_CAP,
            mcts_budget,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Override the per-match ply cap. Useful for domains with longer
    /// horizons than the mock IIGs.
    pub fn with_ply_cap(mut self, ply_cap: u32) -> Self {
        self.ply_cap = ply_cap;
        self
    }

    /// Run the tournament.
    ///
    /// Returns a [`TournamentWinner`] borrowing the winning candidate from
    /// `self.candidates`. The borrow is tied to `&self`.
    ///
    /// # Panics
    ///
    /// Panics if `self.candidates` is empty.
    pub fn run<'a>(
        &'a self,
        initial_state: &S,
        baseline: &dyn Fn(&S, u8) -> S::Action,
    ) -> TournamentWinner<'a, S, V> {
        assert!(
            !self.candidates.is_empty(),
            "ValueFnTournament::run: candidates must be non-empty"
        );
        let n = self.candidates.len();

        // Per-candidate stats vs baseline (aggregates both roles).
        let mut vs_baseline: Vec<PlayerStats> = vec![PlayerStats::default(); n];

        // Head-to-head win-rate matrix [i][j] = i's win-rate vs j.
        // Diagonal left at 0.0 (a candidate doesn't play itself).
        let mut head_to_head: Vec<Vec<f32>> = vec![vec![0.0; n]; n];

        // ── Phase A: each candidate vs baseline, both roles ────────────
        for (i, candidate) in self.candidates.iter().enumerate() {
            for role in 0u8..2 {
                for g in 0..self.games_per_match {
                    // Deterministic per-match seed derived from
                    // (base, candidate_idx, role, game_idx). This is what
                    // makes the tournament reproducible.
                    let match_seed = self.seed_for_match(i as u64, role, g as u64);
                    let (cand_r, opp_r) = self.play_one_match(
                        initial_state,
                        candidate,
                        baseline,
                        role, // candidate plays this role
                        match_seed,
                    );
                    vs_baseline[i].record(cand_r, opp_r);
                }
            }
        }

        // ── Phase B: round-robin head-to-head ─────────────────────────
        for i in 0..n {
            for j in (i + 1)..n {
                let mut i_wins = 0u32;
                let mut j_wins = 0u32;
                let mut draws = 0u32;
                let total = self.games_per_match as u32;
                // Half the games i moves first, half j moves first — control
                // for first-mover advantage.
                for g in 0..self.games_per_match {
                    let first_mover_is_i = g < self.games_per_match / 2 + self.games_per_match % 2;
                    let (i_role, j_role) = if first_mover_is_i { (0u8, 1u8) } else { (1u8, 0u8) };
                    let match_seed = self.seed_for_head_to_head(i as u64, j as u64, g as u64);

                    let (r_i, r_j) = self.play_head_to_head(
                        initial_state,
                        &self.candidates[i],
                        &self.candidates[j],
                        i_role,
                        j_role,
                        match_seed,
                    );
                    if r_i > r_j {
                        i_wins += 1;
                    } else if r_j > r_i {
                        j_wins += 1;
                    } else {
                        draws += 1;
                    }
                }
                // Win-rate with draws as half wins (chess convention).
                let i_wr = (i_wins as f32 + 0.5 * draws as f32) / total.max(1) as f32;
                let j_wr = (j_wins as f32 + 0.5 * draws as f32) / total.max(1) as f32;
                head_to_head[i][j] = i_wr;
                head_to_head[j][i] = j_wr;
            }
        }

        // ── Pick winner: highest win-rate vs baseline ──────────────────
        // Tie-break: highest sum of head-to-head win-rates (sum over row of
        // `head_to_head[i]` minus the diagonal). Tertiary tie-break: lowest
        // index (deterministic).
        let mut best_idx = 0usize;
        let mut best_wr = vs_baseline[0].win_rate();
        let mut best_tb: f32 = head_to_head[0].iter().sum();
        for i in 1..n {
            let wr = vs_baseline[i].win_rate();
            let tb: f32 = head_to_head[i].iter().sum();
            if (wr > best_wr)
                || ((wr - best_wr).abs() < f32::EPSILON && tb > best_tb)
            {
                best_idx = i;
                best_wr = wr;
                best_tb = tb;
            }
        }

        TournamentWinner::new(best_idx, &self.candidates[best_idx], vs_baseline, head_to_head)
    }

    // ── Internals ──────────────────────────────────────────────────────

    /// Deterministic per-match seed for the (candidate, baseline) phase.
    ///
    /// Mixing: `base ^ (cand_idx << 16) ^ (role << 8) ^ game_idx`. Uses XOR
    /// mixing rather than wrapping_add so distinct (cand, role, game)
    /// triples cannot collide modulo 2^64 in any realistic tournament size.
    fn seed_for_match(&self, cand_idx: u64, role: u8, game_idx: u64) -> u64 {
        self.rng_seed ^ (cand_idx << 16) ^ ((role as u64) << 8) ^ game_idx
    }

    /// Deterministic per-match seed for the head-to-head phase. Includes
    /// both candidate indices so `(i,j)` and `(j,i)` get different seeds
    /// only by `g` — but `g` is what differs within a pair anyway.
    fn seed_for_head_to_head(&self, i: u64, j: u64, g: u64) -> u64 {
        self.rng_seed
            .wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(j.wrapping_mul(0xBB67_AE85_84CA_A73B))
            .wrapping_add(g)
    }

    /// Play one match between a candidate (using MCTS-with-heuristic) and
    /// the baseline closure.
    ///
    /// Returns `(candidate_reward, baseline_reward)` at terminal (or at the
    /// ply-cap, treated as terminal with whatever `reward()` returns).
    fn play_one_match(
        &self,
        initial_state: &S,
        candidate: &V,
        baseline: &dyn Fn(&S, u8) -> S::Action,
        candidate_role: u8,
        match_seed: u64,
    ) -> (f32, f32) {
        let mut state = initial_state.clone();
        let mut rng = Rng::with_seed(match_seed);
        let baseline_role = 1 - candidate_role;

        for _ply in 0..self.ply_cap {
            if state.is_terminal() {
                break;
            }
            // Alternate turns. To keep this generic over turn-order domains,
            // we use the convention that the player with the smaller role
            // moves first (matches the test mock's design).
            for role in [0u8, 1u8] {
                if state.is_terminal() {
                    break;
                }
                let actions = state.available_actions(role);
                if actions.is_empty() {
                    continue;
                }
                let action = if role == candidate_role {
                    // Candidate uses MCTS-with-heuristic.
                    self.mcts_select(&state, candidate, role, &mut rng)
                } else {
                    // Baseline closure.
                    baseline(&state, role)
                };
                state = state.advance(&action, role);
            }
        }

        let cand_r = state.reward(candidate_role);
        let base_r = state.reward(baseline_role);
        (cand_r, base_r)
    }

    /// Play one head-to-head match between two candidate heuristics.
    ///
    /// Returns `(reward_for_i, reward_for_j)`.
    fn play_head_to_head(
        &self,
        initial_state: &S,
        cand_i: &V,
        cand_j: &V,
        role_i: u8,
        role_j: u8,
        match_seed: u64,
    ) -> (f32, f32) {
        let mut state = initial_state.clone();
        let mut rng = Rng::with_seed(match_seed);

        for _ply in 0..self.ply_cap {
            if state.is_terminal() {
                break;
            }
            for role in [0u8, 1u8] {
                if state.is_terminal() {
                    break;
                }
                let actions = state.available_actions(role);
                if actions.is_empty() {
                    continue;
                }
                let action = if role == role_i {
                    self.mcts_select(&state, cand_i, role, &mut rng)
                } else {
                    self.mcts_select(&state, cand_j, role, &mut rng)
                };
                state = state.advance(&action, role);
            }
        }
        (state.reward(role_i), state.reward(role_j))
    }

    /// MCTS-backed action selection for a candidate heuristic. Mirrors
    /// `mcts.rs::mcts_search_impl` minus the transposition table.
    ///
    /// The heuristic is used as the leaf evaluation function — when the
    /// rollout hits a non-terminal state at the depth cap, we use
    /// `heuristic.evaluate(...)` instead of `state.reward(...)`. This is
    /// the only place the candidate's heuristic enters the search; the
    /// rest of the MCTS (selection, expansion, UCB1) is identical across
    /// candidates.
    fn mcts_select(&self, root: &S, heuristic: &V, player_id: u8, rng: &mut Rng) -> S::Action {
        // Fast path: only one legal action.
        let mut root_actions: Vec<S::Action> = Vec::with_capacity(8);
        root.available_actions_into(player_id, &mut root_actions);
        if root_actions.len() == 1 {
            return root_actions.into_iter().next().expect("len == 1");
        }

        // Root statistics keyed by action index (root_actions is stable for
        // the lifetime of this call).
        let n_root = root_actions.len();
        let mut root_visits = vec![0u32; n_root];
        let mut root_total = vec![0f32; n_root];
        // Running total of root visits — avoids the per-iter O(n_root) sum
        // `root_visits.iter().copied().sum::<u32>()` inside the MCTS loop.
        let mut total_root_visits: u32 = 0;

        // Scratch buffers reused across iterations (AGENTS.md hot-loop rules).
        let mut path: Vec<usize> = Vec::with_capacity(MCTS_ROLLOUT_DEPTH_CAP + 1);
        let mut rollout_actions: Vec<S::Action> = Vec::with_capacity(8);

        for _iter in 0..self.mcts_budget {
            path.clear();
            let mut current = root.clone();
            let mut current_player = player_id;

            // ── Selection + expansion: walk down using UCB1 until we hit
            // an unexplored action or a terminal. We re-collect legal
            // actions at every depth because different states have
            // different action sets.
            let mut depth_actions: Vec<S::Action> = Vec::with_capacity(8);
            current.available_actions_into(current_player, &mut depth_actions);
            let mut depth = 0usize;

            loop {
                if current.is_terminal() || depth_actions.is_empty() {
                    break;
                }
                if depth >= MCTS_ROLLOUT_DEPTH_CAP {
                    break;
                }

                // If we're at the root, use root_visits/root_total directly;
                // for deeper nodes, do a 1-ply expansion (pick the first
                // unvisited action, mirroring flat UCB1 — keeps the tree
                // shallow, which is fine for the relative-strength
                // measurement).
                let parent_visits = if depth == 0 {
                    total_root_visits
                } else {
                    // For deeper nodes, there is no per-node stat table —
                    // fall through to "always expand" semantics (one
                    // rollout per deep action).
                    0
                };

                // Pick action by UCB1 at root; uniform-random at depth > 0.
                let action_idx = if depth == 0 {
                    pick_ucb1(&root_visits, &root_total, parent_visits, rng)
                } else {
                    rng.usize(0..depth_actions.len())
                };
                path.push(action_idx);
                let action = &depth_actions[action_idx];
                current = current.advance(action, current_player);
                current_player = 1 - current_player;
                depth += 1;

                // Refresh legal actions for the new state.
                depth_actions.clear();
                current.available_actions_into(current_player, &mut depth_actions);
            }

            // ── Simulation: uniform-random rollout from `current`.
            let leaf_reward = if current.is_terminal() {
                current.reward(player_id)
            } else {
                self.random_rollout(&current, player_id, heuristic, rng, &mut rollout_actions)
            };

            // ── Backpropagation: only the root has stats in this flat
            // variant. We backprop the leaf reward to the root action only.
            if let Some(&root_action_idx) = path.first() {
                root_visits[root_action_idx] += 1;
                root_total[root_action_idx] += leaf_reward;
                total_root_visits += 1;
            }
        }

        // Pick the root action with the highest visit count (robust child),
        // tie-broken by mean value. Robust child is the standard choice for
        // tournament play — it's less sensitive to outlier leaf rewards
        // than max-mean.
        let mut best_idx = 0usize;
        let mut best_visits = root_visits[0];
        let mut best_mean = if root_visits[0] > 0 {
            root_total[0] / root_visits[0] as f32
        } else {
            f32::NEG_INFINITY
        };
        for i in 1..n_root {
            let v = root_visits[i];
            let m = if v > 0 { root_total[i] / v as f32 } else { f32::NEG_INFINITY };
            if v > best_visits || (v == best_visits && m > best_mean) {
                best_idx = i;
                best_visits = v;
                best_mean = m;
            }
        }

        // O(1) instead of O(best_idx) — the rest of root_actions is dropped
        // anyway, so swap_remove is equivalent to nth(best_idx) but skips the
        // iterator advance chain.
        root_actions.swap_remove(best_idx)
    }

    /// Uniform-random rollout with a heuristic leaf evaluation at the depth
    /// cap. Mirrors `induced_cwm/ismcts::random_rollout` plus the heuristic
    /// fallback.
    fn random_rollout(
        &self,
        start: &S,
        player_id: u8,
        heuristic: &V,
        rng: &mut Rng,
        scratch: &mut Vec<S::Action>,
    ) -> f32 {
        let mut current = start.clone();
        for _ in 0..MCTS_ROLLOUT_DEPTH_CAP {
            if current.is_terminal() {
                return current.reward(player_id);
            }
            current.available_actions_into(player_id, scratch);
            if scratch.is_empty() {
                break;
            }
            let pick = rng.usize(0..scratch.len());
            let action = scratch[pick].clone();
            current = current.advance(&action, player_id);
        }
        // Depth cap hit without terminal: use the heuristic's evaluation as
        // the leaf value. THIS is the only place candidates differ.
        heuristic.evaluate(&current, player_id)
    }
}

/// Pick an action index by UCB1. Unvisited actions get priority (return the
/// first unvisited one found, breaking ties by random pick among them).
#[inline]
fn pick_ucb1(visits: &[u32], total: &[f32], parent_visits: u32, rng: &mut Rng) -> usize {
    // Collect unvisited action indices — these get +∞ UCB1.
    let mut unvisited: Vec<usize> = Vec::with_capacity(visits.len());
    for (i, &v) in visits.iter().enumerate() {
        if v == 0 {
            unvisited.push(i);
        }
    }
    if !unvisited.is_empty() {
        // Pick uniformly at random among unvisited — matches the standard
        // "first-play" tie-break in UCB1 implementations.
        return unvisited[rng.usize(0..unvisited.len())];
    }

    // All actions visited at least once: pick max UCB1.
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for i in 0..visits.len() {
        let v = visits[i];
        debug_assert!(v > 0, "v > 0 in UCB1 branch");
        let exploit = total[i] / v as f32;
        let explore = UCB1_C * (parent_visits.max(1) as f32).ln().sqrt() / (v as f32).sqrt();
        let score = exploit + explore;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mock domain: 2-player "first to N" race ────────────────────────
    //
    // Players alternate. On their turn they pick Advance (counter += 1) or
    // Stall (counter += 0). First player to push the shared counter to
    // GOAL wins; if the ply cap is hit, it's a draw. The state has perfect
    // information, so this is NOT a Phase 2 IIG — it's a clean test bed for
    // Phase 3's value-function tournament: a "near-perfect" heuristic
    // (always Advance) should beat a "stall" heuristic (always Stall)
    // against any reasonable baseline.
    //
    // We make the candidate's contribution visible by having the baseline
    // always Stall — so a candidate that Advances wins, a candidate that
    // Stalls draws (hits the ply cap with counter < GOAL).

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct RaceState {
        counter: u32,
        /// Whose turn it is (0 or 1). Drives player alternating.
        turn: u8,
        tick: u32,
        is_terminal: bool,
        winner: u8, // 255 = none
    }

    impl RaceState {
        // GOAL is intentionally > 2 * MCTS_ROLLOUT_DEPTH_CAP (=20). This
        // guarantees that rollouts from the root cannot reach terminal —
        // the depth cap is hit first, forcing the leaf evaluation to come
        // from the heuristic. Without this, the MCTS would discover the
        // winning strategy (Advance) purely through terminal rewards,
        // making the heuristic irrelevant and the test non-discriminating
        // (all three candidates would win vs a Stall baseline).
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
                // Non-terminal partial reward: closer to goal is better.
                // NOTE: this branch is unreachable during MCTS rollouts when
                // GOAL > 2 * MCTS_ROLLOUT_DEPTH_CAP, because the depth cap
                // triggers the heuristic fallback first. Kept for
                // correctness of `play_one_match` (which checks
                // `is_terminal` directly, not via reward).
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

    // ── Mock heuristics ────────────────────────────────────────────────
    //
    // The tournament requires all candidates to share the same Rust type
    // (Vec<V>). To compare multiple distinct heuristic strategies, we wrap
    // them in a single enum that dispatches via the `StateHeuristic` impl.

    /// "Near-perfect" heuristic: evaluates states by counter value (closer
    /// to goal = better, regardless of player). The MCTS using this will
    /// prefer Advance over Stall because Advance leads to states with
    /// higher counter.
    #[derive(Clone, Copy, Debug, Default)]
    struct AdvanceHeuristic;

    impl StateHeuristic<RaceState> for AdvanceHeuristic {
        fn evaluate(&self, state: &RaceState, _player_id: u8) -> f32 {
            // Higher counter = better. Scale to [0, 1].
            (state.counter as f32) / (RaceState::GOAL as f32)
        }
    }

    /// "Stall" heuristic: prefers states with lower counter. Should lose to
    /// AdvanceHeuristic — Stall keeps the game from terminating, eventually
    /// hitting the ply cap → draw at best.
    #[derive(Clone, Copy, Debug, Default)]
    struct StallHeuristic;

    impl StateHeuristic<RaceState> for StallHeuristic {
        fn evaluate(&self, state: &RaceState, _player_id: u8) -> f32 {
            1.0 - (state.counter as f32) / (RaceState::GOAL as f32)
        }
    }

    /// "Constant" heuristic — totally uninformative. Used as a third
    /// candidate to verify the tournament ranks all candidates, not just
    /// the top two.
    #[derive(Clone, Copy, Debug, Default)]
    struct ConstantHeuristic;

    impl StateHeuristic<RaceState> for ConstantHeuristic {
        fn evaluate(&self, _state: &RaceState, _player_id: u8) -> f32 {
            0.5
        }
    }

    /// Enum wrapping the three mock heuristics so they can coexist in a
    /// `Vec<RaceHeuristic>` (the tournament requires all candidates to
    /// share a single Rust type).
    #[derive(Clone, Copy, Debug)]
    enum RaceHeuristic {
        Constant(ConstantHeuristic),
        Advance(AdvanceHeuristic),
        Stall(StallHeuristic),
    }

    impl StateHeuristic<RaceState> for RaceHeuristic {
        fn evaluate(&self, state: &RaceState, player_id: u8) -> f32 {
            match self {
                RaceHeuristic::Constant(h) => h.evaluate(state, player_id),
                RaceHeuristic::Advance(h) => h.evaluate(state, player_id),
                RaceHeuristic::Stall(h) => h.evaluate(state, player_id),
            }
        }
    }

    impl Default for RaceHeuristic {
        fn default() -> Self {
            RaceHeuristic::Constant(ConstantHeuristic)
        }
    }

    // ── Baseline closure (always Stall — a deliberately weak opponent) ──

    fn stall_baseline(_state: &RaceState, _player_id: u8) -> RaceAction {
        RaceAction::Stall
    }

    // ── Tests proper ───────────────────────────────────────────────────

    #[test]
    fn player_stats_default_is_zero() {
        let s = PlayerStats::default();
        assert_eq!(s.wins, 0);
        assert_eq!(s.losses, 0);
        assert_eq!(s.draws, 0);
        assert_eq!(s.score, 0.0);
        assert_eq!(s.games(), 0);
        assert_eq!(s.win_rate(), 0.0);
        assert_eq!(s.avg_reward(), 0.0);
    }

    #[test]
    fn player_stats_record_win_loss_draw() {
        let mut s = PlayerStats::default();
        s.record(1.0, 0.0); // win
        s.record(0.0, 1.0); // loss
        s.record(0.5, 0.5); // draw
        assert_eq!(s.wins, 1);
        assert_eq!(s.losses, 1);
        assert_eq!(s.draws, 1);
        assert_eq!(s.games(), 3);
        // score = 1.0 (win) + 0.0 (loss) + 0.5 (draw) = 1.5
        assert!((s.score - 1.5).abs() < 1e-6);
        // win_rate = (1 + 0.5) / 3 = 0.5
        assert!((s.win_rate() - 0.5).abs() < 1e-6);
        // avg_reward = 1.5 / 3 = 0.5
        assert!((s.avg_reward() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn player_stats_display_formats() {
        let mut s = PlayerStats::default();
        s.record(1.0, 0.0);
        let out = format!("{}", s);
        assert!(out.contains("win_rate="));
        assert!(out.contains("avg_reward="));
    }

    #[test]
    fn tournament_picks_advance_over_stall_and_constant() {
        // Three candidates: Advance (near-perfect), Constant (uninformative),
        // Stall (counter-productive). With GOAL > 2 * MCTS_ROLLOUT_DEPTH_CAP,
        // the MCTS rollout's leaf evaluation comes from the heuristic — so
        // the candidate's action choice is directly determined by the
        // heuristic's preference. AdvanceHeuristic prefers high counter →
        // picks Advance → candidate wins vs Stall baseline. StallHeuristic
        // prefers low counter → picks Stall → no progress → draw at ply cap.
        let tournament = ValueFnTournament::new(
            vec![
                RaceHeuristic::Constant(ConstantHeuristic),
                RaceHeuristic::Advance(AdvanceHeuristic),
                RaceHeuristic::Stall(StallHeuristic),
            ],
            /* games_per_match */ 4,
            /* rng_seed */ 42,
            /* mcts_budget */ 24,
        )
        .with_ply_cap(80);

        let winner = tournament.run(&RaceState::new(), &stall_baseline);

        // AdvanceHeuristic is index 1 in the candidate list.
        assert_eq!(
            winner.winner_idx, 1,
            "AdvanceHeuristic should win; got idx {} with stats {:?}",
            winner.winner_idx, winner.vs_baseline
        );

        // Sanity: AdvanceHeuristic's win-rate vs Stall baseline must be > 0.5.
        let advance_stats = winner.vs_baseline[1];
        assert!(
            advance_stats.win_rate() > 0.5,
            "AdvanceHeuristic win_rate vs Stall baseline must be > 0.5; got {}",
            advance_stats.win_rate()
        );

        // StallHeuristic (index 2) should have lower win-rate than
        // AdvanceHeuristic — its MCTS prefers Stall, so the candidate
        // doesn't make progress against the Stall baseline → mostly draws.
        let stall_stats = winner.vs_baseline[2];
        assert!(
            stall_stats.win_rate() < advance_stats.win_rate(),
            "StallHeuristic win_rate ({}) must be < AdvanceHeuristic win_rate ({})",
            stall_stats.win_rate(),
            advance_stats.win_rate()
        );
    }

    #[test]
    fn tournament_head_to_head_matrix_is_consistent() {
        let tournament = ValueFnTournament::new(
            vec![
                RaceHeuristic::Advance(AdvanceHeuristic),
                RaceHeuristic::Stall(StallHeuristic),
            ],
            4,
            123,
            24,
        )
        .with_ply_cap(80);
        let winner = tournament.run(&RaceState::new(), &stall_baseline);

        // head_to_head is a 2x2 matrix. Diagonal is 0. Off-diagonal entries
        // must sum to ≤ 1.0 (each pair plays a fixed number of games; with
        // draws counted as half wins, i + j = 1 only if no draws).
        let h2h = &winner.head_to_head;
        assert_eq!(h2h.len(), 2);
        assert_eq!(h2h[0].len(), 2);
        assert_eq!(h2h[0][0], 0.0);
        assert_eq!(h2h[1][1], 0.0);
        // Off-diagonals sum to ≤ 1.0 (1.0 if no draws, < 1.0 with draws).
        let off_diag_sum = h2h[0][1] + h2h[1][0];
        assert!(
            off_diag_sum <= 1.0 + 1e-6,
            "off-diagonal sum must be ≤ 1.0 (chess-scoring); got {}",
            off_diag_sum
        );
    }

    #[test]
    fn tournament_is_deterministic_given_seed() {
        // Same seed + same inputs → same winner.
        let baseline = stall_baseline;
        let initial = RaceState::new();

        let mk_candidates = || {
            vec![
                RaceHeuristic::Constant(ConstantHeuristic),
                RaceHeuristic::Advance(AdvanceHeuristic),
                RaceHeuristic::Stall(StallHeuristic),
            ]
        };

        let t1 = ValueFnTournament::new(mk_candidates(), 4, 99, 24).with_ply_cap(80);
        let w1 = t1.run(&initial, &baseline);

        let t2 = ValueFnTournament::new(mk_candidates(), 4, 99, 24).with_ply_cap(80);
        let w2 = t2.run(&initial, &baseline);

        assert_eq!(w1.winner_idx, w2.winner_idx, "same seed → same winner");
        for i in 0..w1.vs_baseline.len() {
            assert_eq!(
                w1.vs_baseline[i].wins, w2.vs_baseline[i].wins,
                "same seed → same per-candidate wins"
            );
            assert_eq!(
                w1.vs_baseline[i].losses, w2.vs_baseline[i].losses,
                "same seed → same per-candidate losses"
            );
            assert_eq!(
                w1.vs_baseline[i].draws, w2.vs_baseline[i].draws,
                "same seed → same per-candidate draws"
            );
        }
    }

    #[test]
    #[should_panic(expected = "candidates must be non-empty")]
    fn tournament_panics_on_empty_candidates() {
        let tournament: ValueFnTournament<RaceState, RaceHeuristic> =
            ValueFnTournament::new(Vec::new(), 4, 42, 32);
        let _ = tournament.run(&RaceState::new(), &stall_baseline);
    }

    #[test]
    fn seed_for_match_is_distinct_for_distinct_inputs() {
        let t = ValueFnTournament::new(
            vec![RaceHeuristic::Advance(AdvanceHeuristic)],
            4,
            42,
            32,
        );
        let s00 = t.seed_for_match(0, 0, 0);
        let s01 = t.seed_for_match(0, 0, 1);
        let s10 = t.seed_for_match(1, 0, 0);
        let s0r = t.seed_for_match(0, 1, 0);
        assert_ne!(s00, s01, "different game → different seed");
        assert_ne!(s00, s10, "different candidate → different seed");
        assert_ne!(s00, s0r, "different role → different seed");
    }

    #[test]
    fn pick_ucb1_prioritises_unvisited() {
        let mut rng = Rng::with_seed(7);
        // visits = [1, 0, 2, 0] → unvisited are indices 1 and 3.
        let visits = vec![1, 0, 2, 0];
        let total = vec![0.5, 0.0, 1.0, 0.0];
        // Run 50 picks; the result must always be in {1, 3}.
        for _ in 0..50 {
            let idx = pick_ucb1(&visits, &total, 10, &mut rng);
            assert!(idx == 1 || idx == 3, "unvisited idx must be picked; got {}", idx);
        }
    }

    #[test]
    fn pick_ucb1_returns_highest_score_when_all_visited() {
        let mut rng = Rng::with_seed(7);
        // Two actions, both visited. Action 0 has higher mean → wins.
        let visits = vec![10, 10];
        let total = vec![10.0, 1.0]; // means 1.0 vs 0.1
        let idx = pick_ucb1(&visits, &total, 20, &mut rng);
        assert_eq!(idx, 0, "higher-mean action must win when exploration term is equal");
    }
}
