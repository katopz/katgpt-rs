//! Information-Set MCTS over an induced CWM + belief fn (Plan 296 Phase 2).
//!
//! Paper §4.3 + §B (arxiv 2510.04542): for imperfect-information games, MCTS
//! is run over *information sets* — collections of plausible hidden states
//! drawn from a belief inference function — rather than over a single known
//! root state. The classic reference is Cowling et al. 2012 (Information Set
//! MCTS), which the paper builds on for the CWM play-strength gate (G2).
//!
//! # Algorithm (simplified — root-level info-set aggregation)
//!
//! This is NOT full single-tree ISMCTS (Cowling 2012 §3). It is the simpler
//! **per-iteration determinized MCTS with root-level information-set
//! aggregation** that the plan specifies:
//!
//! 1. Maintain one shared root statistics table — `HashMap<u64 /*action hash*/,
//!    NodeStats>` — keyed by the action the player is choosing at the root.
//!    Action-hash is the key because `S::Action` is not required to be `Ord`;
//!    only `Hash + Eq + Clone`.
//! 2. For each of `budget` iterations:
//!    a. Sample one hidden state `s̃` from the belief fn.
//!    b. For each root action `a`, do a SHORT random rollout
//!       (depth ≤ `ROLLOUT_DEPTH_CAP`) from `s̃.advance(&a, player_id)`,
//!       accumulating `(visits += 1, total_value += reward)` into the root
//!       table keyed by `action_hash(&a)`.
//! 3. Return the root action with the highest visit count; tie-break by mean
//!    value (`total_value / visits`).
//!
//! ## Why this is correct for the G2 gate (Plan 296)
//!
//! The G2 gate asserts: ISMCTS picks a non-fold action ≥ 70% of the time when
//! posterior P(strong hand) ≥ 0.6. This only needs the *root* statistics to
//! reflect expected value across the posterior — exactly what step (2b)
//! computes. The deeper tree that full ISMCTS builds buys better convergence
//! rate, not different root statistics asymptotically.
//!
//! ## What this implementation is NOT
//!
//! - It does NOT build a single shared tree across determinizations (Cowling
//!   2012's SO-ISMCTS). Each iteration's "tree" is just one ply + a rollout.
//! - It does NOT do opponent modelling (MO-ISMCTS). All rollouts use the same
//!   player's action set.
//! - It does NOT do UCB1 *inside* the search tree past depth 1. The
//!   exploration/exploitation tradeoff happens across iterations via the
//!   outer loop's random sample from the belief.
//!
//! A future revision could implement full single-tree ISMCTS if
//! deeper-lookahead games (e.g. full Leduc hold'em with multi-round betting)
//! need it. For 1–2 ply decision problems — which is what the G2 mock is —
//! this is sufficient and ~10× cheaper.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - `belief.sample(...)` returns latent hidden states. They never cross the
//!   sync boundary as embeddings — ISMCTS consumes them locally and emits a
//!   single chosen action (raw).
//! - The chosen `S::Action` is raw and may be synced.
//! - The `InformationSet` statistics (`NodeStats { visits, total_value }`)
//!   are search-local scratch — not synced.
//!
//! # Self-containment
//!
//! This module lives in `katgpt-core`, so it CANNOT import
//! `katgpt-rs/src/pruners/game_state/mcts.rs` (the root crate, depending on
//! `katgpt-core`, would create a circular dep). The node/stats structs are
//! mirrored here — DRY would be nice but is structurally impossible without
//! moving `mcts_search` itself into `katgpt-core`, which is out of scope for
//! Phase 2.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md`] §Phase 2
//! - Source paper: [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542) §4.3, §B
//! - ISMCTS reference: Cowling, Powley, Whitehouse (2012), "Information Set
//!   Monte Carlo Tree Search", IEEE TCIAIG.
//! - Belief contract: [`crate::induced_cwm::belief`]
//! - Kernel trait: [`crate::induced_cwm::InducedCwmKernel`]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use fastrand::Rng;

use crate::induced_cwm::{BeliefInferenceFn, InducedCwmKernel};
use crate::traits::GameState;

/// UCB1 exploration constant — `sqrt(2)`, mirroring `mcts.rs`.
///
/// Kept as a `const` (not configurable) so the search is byte-stable across
/// re-runs for the G2 gate.
const UCB1_C: f32 = 1.414;

/// Maximum rollout depth per action sample. Matches the spec's "~10" cap.
///
/// The mock IIGs are 1–4 ply deep, so this is plenty; deeper games would
/// benefit from a larger cap but pay the latency cost (G3 budget).
const ROLLOUT_DEPTH_CAP: usize = 10;

// ── T2.3 — NodeStats ──────────────────────────────────────────────────────

/// Per-edge statistics for one (info-set, action) pair.
///
/// Mirrors the fields of `MCTSNode { total_reward, visits }` from
/// `katgpt-rs/src/pruners/game_state/mcts.rs`, minus the tree-structure
/// fields (`parent`, `children`, `unexpanded`) — those are unused in the
/// root-aggregation algorithm. See the module docs for the simplification
/// rationale.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NodeStats {
    /// Number of rollouts that passed through this edge.
    pub visits: u32,
    /// Sum of rewards from those rollouts.
    pub total_value: f32,
}

impl NodeStats {
    /// Mean value per visit. Returns `0.0` for an unvisited edge (callers
    /// should not call this on `visits == 0` edges — use [`ucb1`](Self::ucb1)
    /// instead, which returns `+∞` there).
    #[inline]
    pub fn mean_value(&self) -> f32 {
        match self.visits {
            0 => 0.0,
            n => self.total_value / n as f32,
        }
    }

    /// UCB1 score of this edge given the parent info-set's total visit count.
    ///
    /// Mirrors `mcts.rs::ucb1_score`:
    /// `exploit + UCB1_C * sqrt(ln(parent_visits) / visits)`.
    /// Returns `+∞` when `visits == 0` (unvisited edges get exploration
    /// priority).
    #[inline]
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        match self.visits {
            0 => f32::INFINITY,
            n => {
                let exploit = self.total_value / n as f32;
                let explore = UCB1_C * (parent_visits.max(1) as f32).ln().sqrt() / (n as f32).sqrt();
                exploit + explore
            }
        }
    }

    /// Record one rollout's reward.
    #[inline]
    pub fn record(&mut self, reward: f32) {
        self.visits += 1;
        self.total_value += reward;
    }
}

// ── T2.2 — InformationSet ─────────────────────────────────────────────────

/// Statistics for one information set.
///
/// In the simplified algorithm this only ever holds the *root* info-set's
/// per-action edges; deeper info-sets would require full single-tree ISMCTS
/// (see the module docs). The type is exposed for API symmetry with Plan 296
/// §T2.2 so callers can inspect / serialise the root statistics after a
/// search — and so a future full-ISMCTS revision can extend it without
/// breaking the public surface.
///
/// # Keying
///
/// Edges are keyed by `u64` action-hash, not by `A` directly. This is because
/// `S::Action: Hash + Eq + Clone` is required, but `Ord` is not — and
/// `HashMap<u64, _>` avoids the `BTreeMap`-would-need-`Ord` problem entirely.
/// The hash is computed by [`action_hash`].
///
/// # Latent boundary
///
/// `NodeStats` are search-local scratch — they do not cross the sync boundary.
/// Only the chosen `S::Action` is syncable.
#[derive(Clone, Debug, Default)]
pub struct InformationSet {
    /// Per-action edge statistics, keyed by `action_hash(&a)`.
    pub edges: std::collections::HashMap<u64, NodeStats>,
    /// Total visit count across all edges (= number of iterations that
    /// produced a usable sample × actions-per-sample). Used as the
    /// `parent_visits` argument to [`NodeStats::ucb1`].
    pub total_visits: u32,
}

impl InformationSet {
    /// Construct an empty information set with capacity hints for `n_actions`
    /// root edges.
    pub fn with_capacity(n_actions: usize) -> Self {
        Self {
            edges: std::collections::HashMap::with_capacity(n_actions),
            total_visits: 0,
        }
    }

    /// Record one rollout's reward against the edge keyed by `action_hash(a)`.
    #[inline]
    pub fn record<A: Hash>(&mut self, action: &A, reward: f32) {
        let key = action_hash(action);
        self.edges.entry(key).or_default().record(reward);
        self.total_visits += 1;
    }

    /// Mean value of an edge, or `0.0` if absent.
    pub fn mean_value_for<A: Hash>(&self, action: &A) -> f32 {
        match self.edges.get(&action_hash(action)) {
            Some(s) => s.mean_value(),
            None => 0.0,
        }
    }

    /// Visit count of an edge, or `0` if absent.
    pub fn visits_for<A: Hash>(&self, action: &A) -> u32 {
        self.edges.get(&action_hash(action)).map_or(0, |s| s.visits)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Stable `u64` hash of an action.
///
/// Uses `DefaultHasher` (SipHash-1-3 with a fixed key per process). The hash
/// is used ONLY as a `HashMap` key inside one search — it never crosses a
/// process boundary — so the per-process randomised key is acceptable. Do NOT
/// use this for commitments (use [`InducedCwmKernel::commitment`] /
/// [`crate::induced_cwm::CwmCommitment`] with `blake3` for that).
#[inline]
pub fn action_hash<A: Hash>(action: &A) -> u64 {
    let mut h = DefaultHasher::new();
    action.hash(&mut h);
    h.finish()
}

/// Short uniform-random rollout from `start` using `rng`. Returns the
/// terminal reward for `player_id`, or — if the depth cap is hit before a
/// terminal — the `start.reward(player_id)` heuristic snapshot.
///
/// Mirrors `mcts.rs::rollout` minus the pluggable `RolloutPolicy` (we always
/// use uniform random here — the spec calls for it; a `BanditRolloutPolicy`
/// hook is a Phase 3 concern).
fn random_rollout<S: GameState>(
    start: &S,
    player_id: u8,
    rng: &mut Rng,
    scratch: &mut Vec<S::Action>,
) -> f32 {
    let mut current = start.clone();
    for _ in 0..ROLLOUT_DEPTH_CAP {
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
    // Either terminal (reward) or non-terminal (heuristic snapshot).
    // The mock IIGs return meaningful `reward` only at terminal, so the
    // non-terminal fallback returns the partial reward which is typically 0
    // for the mock and `f32::NAN`-free for real domains.
    current.reward(player_id)
}

// ── T2.1 — ismcts_search_with_inference ───────────────────────────────────

/// Information-Set MCTS over an induced CWM + belief fn.
///
/// Implements the simplified root-level info-set aggregation algorithm
/// described in the module docs. Returns the chosen root action for
/// `player_id`.
///
/// # Arguments
///
/// * `root_obs_history` — observations visible to `player_id` so far (passed
///   verbatim to `belief.sample`).
/// * `root_action_history` — actions taken by all players so far.
/// * `player_id` — which player we're choosing an action for.
/// * `belief` — the belief inference function (paper §4.2). Its `Sample`
///   associated type MUST be `S` itself (the kernel IS the state type per the
///   codebase's `GameState` convention).
/// * `budget` — number of belief samples to draw (one sample = one
///   determinization; for each, all root actions are evaluated once).
/// * `rng_seed` — RNG seed. The same seed + same `belief` MUST produce the
///   same chosen action — required for the G2 gate's reproducibility.
///
/// # Returns
///
/// The root action with the highest visit count; ties broken by highest mean
/// value.
///
/// # Panics
///
/// Panics if the root state (sampled from the belief) reports no available
/// actions for `player_id`. The belief fn is responsible for not emitting
/// terminal states.
///
/// # Type bounds
///
/// * `S: GameState + InducedCwmKernel` — the kernel/state type.
/// * `S::Action: PartialEq + Debug + Hash + Eq + Clone` — Hash+Eq+Clone is
///   required for `action_hash` keying; PartialEq+Debug is required for the
///   returned action to be inspectable / assertable in tests.
/// * `B: BeliefInferenceFn<S, Sample = S>` — the belief fn's sample type
///   IS the kernel/state type (Scenario A in Phase 1 tests).
pub fn ismcts_search_with_inference<S, B>(
    root_obs_history: &[S::Action],
    root_action_history: &[S::Action],
    player_id: u8,
    belief: &B,
    budget: usize,
    rng_seed: u64,
) -> S::Action
where
    S: GameState + InducedCwmKernel,
    S::Action: PartialEq + std::fmt::Debug + Hash + Eq + Clone,
    B: BeliefInferenceFn<S, Sample = S>,
{
    // RNG lives for the whole search. Each iteration uses
    // `rng_seed.wrapping_add(iteration)` only as the *belief* seed; the
    // rollout RNG is the same `Rng` instance threaded through all iterations
    // (mirrors `mcts.rs`).
    let mut rng = Rng::with_seed(rng_seed);

    // One belief sample per iteration, evaluated against ALL root actions.
    // Root actions are obtained fresh from each sample (the legal set may
    // differ across determinizations — that's the whole point of ISMCTS).
    //
    // Per-action statistics are aggregated across samples in `root_set`.
    let mut root_set = InformationSet::with_capacity(8);
    let mut action_scratch: Vec<S::Action> = Vec::with_capacity(8);
    let mut rollout_scratch: Vec<S::Action> = Vec::with_capacity(8);

    // We also need to remember the *actual* action values per hash so we can
    // return one at the end — multiple distinct actions can share a hash
    // (collisions are astronomically rare for SipHash on small action sets,
    // but the type system demands we handle it). We remember the first action
    // seen for each hash; ties between collision-actions are arbitrary but
    // deterministic.
    let mut hash_to_action: std::collections::HashMap<u64, S::Action> =
        std::collections::HashMap::with_capacity(8);

    for iteration in 0..budget {
        // (a) draw one hidden-state sample
        let belief_seed = rng_seed.wrapping_add(iteration as u64);
        let samples = belief.sample(
            root_obs_history,
            root_action_history,
            player_id,
            1,
            belief_seed,
        );
        let Some(sample) = samples.into_iter().next() else {
            // Empty sample → skip iteration (the belief fn may legitimately
            // return an empty Vec if its posterior support is empty).
            continue;
        };

        // (b) enumerate this sample's root actions
        sample.available_actions_into(player_id, &mut action_scratch);
        if action_scratch.is_empty() {
            // The belief fn should not emit terminal states; if it does, skip
            // rather than panic — callers can detect "no action chosen" by
            // `root_set.total_visits == 0` after the call.
            continue;
        }

        // (c) for each root action: advance, short rollout, record reward
        for action in action_scratch.iter() {
            let key = action_hash(action);
            hash_to_action.entry(key).or_insert_with(|| action.clone());

            let child = sample.advance(action, player_id);
            let reward = if child.is_terminal() {
                child.reward(player_id)
            } else {
                random_rollout(&child, player_id, &mut rng, &mut rollout_scratch)
            };
            root_set.record(action, reward);
        }
    }

    // (3) pick the root action with the highest visit count; tie-break by
    // highest mean value, then by hash value (tertiary tie-break makes the
    // search deterministic across re-runs given the same seed — required for
    // the G2 gate's reproducibility). If the table is empty (every sample
    // was skipped), we cannot return anything meaningful — panic, matching
    // `mcts_search`'s invariant that there must be at least one available
    // action.
    let best_hash = root_set
        .edges
        .iter()
        .max_by(|(ka, va), (kb, vb)| {
            // Primary: visits (descending).
            match va.visits.cmp(&vb.visits) {
                std::cmp::Ordering::Equal => {
                    // Tie-break: mean value (descending).
                    let ma = va.mean_value();
                    let mb = vb.mean_value();
                    match ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal) {
                        std::cmp::Ordering::Equal => {
                            // Tertiary: hash value (descending) — guarantees
                            // determinism regardless of HashMap iter order.
                            ka.cmp(kb)
                        }
                        ord => ord,
                    }
                }
                ord => ord,
            }
        })
        .map(|(k, _)| *k)
        .expect("ismcts_search_with_inference: no edges recorded — belief fn returned empty samples for all iterations");

    hash_to_action
        .remove(&best_hash)
        .expect("hash_to_action must contain every key present in root_set.edges")
}

// ── Module docs (footer) ──────────────────────────────────────────────────
//
// This module implements Plan 296 Phase 2 (T2.1–T2.5). The simplified
// root-level info-set aggregation algorithm is correct for the G2 gate
// (T2.4) and matches the plan's "draw one sample, treat as MCTS root,
// aggregate at info-set level" specification. Full single-tree ISMCTS
// (Cowling 2012 SO-ISMCTS) is intentionally NOT implemented — see the
// header docs for the rationale.
//
// Paper: arxiv 2510.04542 (Lehrach et al., DeepMind Oct 2025), §4.3 + §B.
// Plan: katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md (Phase 2).
