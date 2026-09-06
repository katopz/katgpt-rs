//! PIBT — Priority Inheritance with Backtracking (Plan 440 T1.4, Issue 143).
//!
//! Distilled from Okumura et al. 2022 (*PIBT: Scalable and Prioritization
//! Planning for Multi-Agent Pathfinding*). PIBT is a one-step collision-free
//! joint action generator: given the current joint configuration and per-agent
//! guidance, produce a next-position for every agent. Edge collisions (swaps)
//! are prevented by construction; vertex collisions are prevented on
//! uncongested maps but may occur on congested maps (see §"Collision profile"
//! below for the tradeoff analysis).
//!
//! The lexicographic cost for agent `i` considering move to `u`:
//!
//! ```text
//! ⟨ Ind[Φ[i][0] ≠ u], dist(u, g_i), hindrance(i→u), ε ⟩
//! ```
//!
//! I.e. PIBT first prefers moves consistent with the guidance, then
//! goal-direction, then low hindrance, then random tiebreak.
//!
//! # Implementation: greedy PIBT + swap technique + LaCAM escalation
//!
//! The inner loop is the **greedy PIBT**: agents are processed in priority
//! order, each taking the first collision-free candidate. Later agents see
//! earlier agents' committed positions and adapt. This is aggressive (agents
//! don't wait for undecided occupants) and has high throughput on open maps.
//!
//! Issue 144 adds the **swap technique** (Okumura 2023a, arXiv:2309.02425):
//! when two agents face a head-on corridor deadlock (agent i wants j's cell,
//! j wants i's cell), the lower-priority agent uses reverse scoring to back
//! up, letting the higher-priority agent pass. This directly targets the
//! ht_chantry maze-map deadlock where local priority shuffling cannot help
//! because someone must yield.
//!
//! Issue 143 adds **LaCAM escalation** as an outer loop: when the greedy PIBT
//! produces stuck agents (deadlocks on maze maps), the escalation retries with
//! shuffled priority orderings. This breaks symmetric deadlocks without
//! degrading the fast path on open maps.
//!
//! ## Why not recursive priority inheritance?
//!
//! Issue 140 and Issue 143 both tested recursive priority inheritance (where
//! a high-priority agent evicts undecided occupants from their cells). Both
//! found it **collapses throughput** (empty-48-48 dropped from 18.6 → 1.5,
//! -92%) because the eviction forces agents to move away from their goals,
//! creating cascading stalls. The greedy variant — which lets agents
//! compromise by taking their next-best cell — has dramatically higher
//! collective throughput in the lifelong MAPF setting. The recursive variant
//! is the right algorithm for one-shot MAPF (where finding ANY solution is
//! the goal), but wrong for lifelong MAPF (where sustained throughput is the
//! goal).
//!
//! # Collision profile
//!
//! - **Vertex collisions**: PREVENTED on uncongested maps. On congested maps,
//!   stuck agents (no collision-free move AND current cell committed by a
//!   higher-priority agent) are forced to wait in place, which can produce a
//!   vertex collision. This is a deliberate throughput tradeoff documented
//!   below (§"Why not recursive priority inheritance?") and measured in
//!   `riir-ai/.benchmarks/516_g6c_collision_freedom_delta.md` (62.5% of
//!   congested-map ticks have vertex collisions on the G6 scenario). Plan 453
//!   (bounded one-step LaCAM escalation, behind the `lacam_escalation` feature)
//!   resolves this — the constraint tree + recursive PIBT eliminates vertex
//!   collisions entirely (G-col = 0.0%); the original Issue 154 analysis is
//!   closed as fixed by Plan 453.
//! - **Edge collisions (swaps)**: PREVENTED by construction. The greedy pass
//!   explicitly checks `A→B` and `B→A` simultaneously and rejects swap moves.
//!
//! # Output
//!
//! Returns `Ok(JointAction)` — always succeeds. Stuck agents wait in place
//! (which may vertex-collide on congested maps; see above). The caller may
//! inspect the result for collisions and congestion, and escalate further if
//! needed. For lifelong MAPF, temporary stalls and rare vertex collisions
//! are tolerated as the lesser evil vs the throughput collapse caused by
//! all-wait or recursive priority inheritance.
//!
//! # Determinism
//!
//! The random tiebreak `ε` uses a deterministic seeded RNG, preserving replay.

use super::config::{AgentId, JointAction, JointConfig};
use super::flow::FlowField;
use super::hindrance::HindranceEstimator;
use super::local_guidance::Guidance;
use super::position::Position;
use std::cmp::Ordering;

/// Default number of LaCAM escalation retries when greedy PIBT produces stuck agents.
///
/// Each retry runs the greedy PIBT with a different priority ordering. The
/// result with the fewest stuck agents is returned. Bounded to maintain
/// real-time perf — the paper's LaCAM does a full configuration search, but
/// for lifelong MAPF the bounded retry captures most of the benefit at a
/// fraction of the cost.
///
/// Only used by the legacy shuffled-retry path (`legacy_shuffled_retry`,
/// Issue 143). When `lacam_escalation` is ON (Plan 453), the real LaCAM
/// constraint tree is used instead and this constant is unused.
#[cfg(not(feature = "lacam_escalation"))]
const DEFAULT_LACAM_RETRIES: usize = 2;

/// Minimum number of stuck agents before LaCAM escalation triggers.
///
/// On open maps, some agents may get stuck each tick due to random
/// vertex collisions (an agent's current cell is taken by another). The
/// escalation overhead (up to 4 retries × full PIBT, ~240ms at 800 agents)
/// isn't worth it for small numbers of stuck agents — they'll likely
/// resolve naturally next tick. The threshold ensures retries only fire on
/// genuinely congested maps (maze, dense warehouse) where stuck agents are
/// systemic and the retry is likely to break deadlocks.
#[cfg_attr(feature = "lacam_escalation", allow(dead_code))]
const MIN_STUCK_FOR_RETRY: usize = 20;

/// Type alias mirroring [`super::local_guidance::NeighborFn`] for the
/// neighbor-supplying callback in [`pibt_step`]. Includes `Send + Sync` to
/// match the orchestrator's stored closure type.
///
/// `pub` so benchmark harnesses (Plan 453 T3.3) can call
/// [`lacam_escalation_step`] and [`pibt_step`] directly with wall-aware
/// neighbors. The orchestrator's [`LifelongLaCam::with_neighbors`] is the
/// consumer-facing API; this type is the lower-level plumbing.
pub type NeighborFn<P> = dyn Fn(&P) -> Vec<P> + Send + Sync;

/// Candidate move for an agent, with its lexicographic cost components.
///
/// The cost tuple ordering (Issue 149) is:
///
/// ```text
/// ⟨ guidance_mismatch, flow_mismatch, goal_dist, hindrance, ε ⟩
/// ```
///
/// Where:
/// 1. `guidance_mismatch` (0/1) — prefer moves consistent with the guidance Φ.
/// 2. `flow_mismatch` (0/1) — prefer moves that align with the assigned corridor
///    direction (Guided-PIBT, Issue 149). Zero on open maps (no corridors).
/// 3. `goal_dist` (f32) — prefer moves closer to the goal.
/// 4. `hindrance` (f32) — prefer moves that block fewer siblings.
/// 5. `ε` (f32) — random tiebreak for determinism.
///
/// The `flow_mismatch` term is inserted between `guidance_mismatch` and
/// `goal_dist` (Issue 149). This is the "safe promotion": on open maps,
/// `flow_mismatch` is always 0 (no corridors), so the tuple degenerates to
/// the paper-faithful ordering. On maze maps, `flow_mismatch` creates
/// directional lanes that eliminate head-on corridor deadlocks.
#[derive(Clone)]
/// `pub(super)` so the LaCAM recursive PIBT (`lacam.rs`) reuses the same
/// cost tuple as greedy PIBT (Plan 453).
pub(super) struct Candidate<P: Position + Clone> {
    pub(super) next: P,
    /// Cost component 1: Ind[Φ[i][0] ≠ u] (0 if guidance-consistent, 1 else).
    pub(super) guidance_mismatch: u8,
    /// Cost component 2: flow_mismatch (0 if aligned with corridor direction, 1 if against).
    /// Zero on non-corridor maps — see [`super::flow`].
    pub(super) flow_mismatch: u8,
    /// Cost component 3: dist(u, g_i) heuristic.
    pub(super) goal_dist: f32,
    /// Cost component 4: hindrance(i→u).
    pub(super) hindrance: f32,
    /// Cost component 5: random tiebreak ε ∈ [0, 1).
    pub(super) epsilon: f32,
}

impl<P: Position + Clone> Candidate<P> {
    /// Lexicographic comparison:
    /// guidance_mismatch → flow_mismatch → goal_dist → hindrance → ε.
    pub(super) fn lexicographic_cmp(&self, other: &Self) -> Ordering {
        self.guidance_mismatch
            .cmp(&other.guidance_mismatch)
            .then_with(|| self.flow_mismatch.cmp(&other.flow_mismatch))
            .then_with(|| {
                self.goal_dist.total_cmp(&other.goal_dist)
            })
            .then_with(|| {
                self.hindrance.total_cmp(&other.hindrance)
            })
            .then_with(|| {
                self.epsilon.total_cmp(&other.epsilon)
            })
    }
}

/// One step of greedy PIBT + LaCAM escalation (Issue 143).
///
/// The greedy PIBT processes agents in priority order, each taking the first
/// collision-free candidate (vertex + edge). When the greedy pass produces
/// stuck agents (true deadlocks), the LaCAM escalation retries with shuffled
/// priority orderings to break symmetric deadlocks. Note: on congested maps,
/// stuck agents forced to wait in place may vertex-collide — see the module
/// docs §"Collision profile" for the full tradeoff analysis.
///
/// # Arguments
///
/// - `config`: current joint configuration `Q_t`.
/// - `guidance`: per-agent guidance paths `Φ` (only `Φ[i][0]` is used — the
///   preferred next position).
/// - `goals`: per-agent goals `g_i`.
/// - `priorities`: per-agent priority weights (higher = processed first).
///   If empty, agents are processed in index order.
/// - `hindrance`: the hindrance estimator (pluggable seam #4).
/// - `flow_field`: the flow field for Guided-PIBT direction assignment
///   (Issue 149). Pass `&NoFlow` for paper-faithful behavior (no corridor
///   direction enforcement).
/// - `neighbors_fn`: supplies passable neighbors (`None` = `Position::neighbors()`).
/// - `rng`: deterministic RNG for the `ε` tiebreak and LaCAM shuffle.
///
/// # Returns
///
/// `Ok(JointAction)` always — stuck agents wait in place. Returns
/// `Err(Deadlock)` never (kept for API compat with the orchestrator's
/// `unwrap_or_else` fallback).
pub fn pibt_step<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    priorities: &[f32],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&NeighborFn<P>>,
    rng: &mut fastrand::Rng,
) -> Result<JointAction<P>, Deadlock>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    // Delegate to pibt_step_with_budget with the default budget. When the
    // lacam_escalation feature is ON, the default budget reproduces Plan 453
    // one-step behavior; callers who want multi-step (Issue 546) call
    // pibt_step_with_budget directly with EscalationBudget::multistep_default().
    #[cfg(feature = "lacam_escalation")]
    {
        pibt_step_with_budget(
            config,
            guidance,
            goals,
            priorities,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
            super::lacam::EscalationBudget::default(),
        )
    }
    #[cfg(not(feature = "lacam_escalation"))]
    {
        pibt_step_with_budget(
            config,
            guidance,
            goals,
            priorities,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
        )
    }
}

/// PIBT step with an explicit LaCAM escalation budget (Issue 546 multi-step).
///
/// Same as [`pibt_step`] but accepts an [`EscalationBudget`] argument, allowing
/// callers to opt into stuck-agent targeting + deeper constraint-tree search.
/// Only available when the `lacam_escalation` feature is ON.
///
/// [`EscalationBudget`]: super::lacam::EscalationBudget
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "lacam_escalation")]
pub(in crate::multi_agent_path) fn pibt_step_with_budget<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    priorities: &[f32],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    budget: super::lacam::EscalationBudget,
) -> Result<JointAction<P>, Deadlock>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let order = compute_priority_order(n, priorities);

    let no_backers = vec![false; n];

    let (moves, stuck) = greedy_pibt_pass(
        config,
        guidance,
        goals,
        hindrance,
        flow_field,
        neighbors_fn,
        rng,
        &order,
        &no_backers,
    );

    if stuck.len() < super::lacam::MIN_STUCK_FOR_LACAM {
        return Ok(JointAction::new(moves));
    }

    Ok(super::lacam::lacam_escalation_step(
        config,
        guidance,
        goals,
        priorities,
        hindrance,
        flow_field,
        neighbors_fn,
        rng,
        budget,
    ))
}

/// PIBT step without lacam_escalation feature (legacy shuffled retry path).
#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "lacam_escalation"))]
pub(in crate::multi_agent_path) fn pibt_step_with_budget<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    priorities: &[f32],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&NeighborFn<P>>,
    rng: &mut fastrand::Rng,
) -> Result<JointAction<P>, Deadlock>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let order = compute_priority_order(n, priorities);

    let no_backers = vec![false; n];

    let (moves, stuck) = greedy_pibt_pass(
        config,
        guidance,
        goals,
        hindrance,
        flow_field,
        neighbors_fn,
        rng,
        &order,
        &no_backers,
    );

    if stuck.len() < MIN_STUCK_FOR_RETRY {
        return Ok(JointAction::new(moves));
    }

    legacy_shuffled_retry(
        config,
        guidance,
        goals,
        hindrance,
        flow_field,
        neighbors_fn,
        rng,
        &order,
        &no_backers,
        moves,
        stuck,
    )
}

/// Legacy LaCAM escalation: shuffled-priority retry (Issue 143).
///
/// This is NOT real LaCAM — it just shuffles priority orderings and picks
/// the result with fewest stuck agents. Kept as the GOAT-gate baseline
/// (the `lacam_escalation` OFF path). When `lacam_escalation` is ON,
/// `pibt_step` delegates to the real constraint-tree search instead.
#[cfg(not(feature = "lacam_escalation"))]
#[allow(clippy::too_many_arguments)]
fn legacy_shuffled_retry<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    order: &[usize],
    no_backers: &[bool],
    mut best_moves: Vec<P>,
    mut best_stuck: Vec<AgentId>,
) -> Result<JointAction<P>, Deadlock>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    for _attempt in 0..DEFAULT_LACAM_RETRIES {
        let shuffled = shuffle_order(&best_stuck, order, rng);
        let (moves, stuck) = greedy_pibt_pass(
            config,
            guidance,
            goals,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
            &shuffled,
            no_backers,
        );

        if stuck.is_empty() {
            return Ok(JointAction::new(moves));
        }
        if stuck.len() < best_stuck.len() {
            best_moves = moves;
            best_stuck = stuck;
        }
    }

    // Place stuck agents as wait-in-place (best effort).
    //
    // NOTE: this can create vertex collisions (the stuck agent's current
    // position may be committed by another agent). See the fast-path comment
    // in `pibt_step` for the full analysis. The all-wait alternative kills
    // throughput.
    for &agent in &best_stuck {
        let i = usize::from(agent);
        best_moves[i] = config.pos(agent).clone();
    }

    Ok(JointAction::new(best_moves))
}

/// One greedy PIBT pass: process agents in `order`, each taking the first
/// collision-free candidate.
///
/// Agents in `swap_backers` (Issue 144) use **reverse scoring**: guidance is
/// discarded, and `−dist(v, g_i)` sorts candidates so the agent backs up
/// (moves to the cell farthest from its goal), breaking head-on corridor
/// deadlocks. Backers are processed before non-backers so their committed
/// back-up move clears the path for the forward agent.
///
/// Returns `(moves, stuck_agents)`. The `moves` vector has one entry per agent.
/// Stuck agents have their move set to wait (current position) — they're also
/// listed in the returned `stuck` vec for the LaCAM escalation.
#[allow(clippy::too_many_arguments)]
pub(super) fn greedy_pibt_pass<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    order: &[usize],
    swap_backers: &[bool],
) -> (Vec<P>, Vec<AgentId>)
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let mut moves: Vec<Option<P>> = vec![None; n];

    // Issue 516 T1g: O(1) collision detection structures (replaces O(n) scan).
    //
    // The original `is_collision_free` scanned all n agents for each candidate,
    // making PIBT O(n²) per pass. At n=1000 with LaCAM retries (3 passes),
    // that's ~15M comparisons. These two maps reduce it to O(n) per pass.
    //
    // `current_to_agent`: position → agent index. Built once from config.
    // Used for edge-collision (swap) detection: when agent i at A wants to
    // move to B, we check if the agent at B (if any) is moving to A.
    //
    // `committed_dests`: the set of destinations already committed by
    // higher-priority agents. Used for vertex-collision detection: no two
    // agents can have the same destination.
    let mut current_to_agent: std::collections::HashMap<P, usize> =
        std::collections::HashMap::with_capacity(n);
    for (i, pos) in config.positions.iter().enumerate() {
        // First agent at a position wins (valid configs have distinct positions).
        current_to_agent.entry(pos.clone()).or_insert(i);
    }
    let mut committed_dests: std::collections::HashSet<P> =
        std::collections::HashSet::with_capacity(n);

    // Issue 516 T1g: pre-compute hindrance data structures for O(1) lookups.
    // For `BlockingCount`, this builds a `reach_count` map that eliminates the
    // O(n²) scan in `hindrance()`. No-op for estimators that don't override
    // `prepare`.
    hindrance.prepare(config);

    let mut stuck = Vec::new();

    // Swap-aware ordering: backers first (so their back-up commits before the
    // forward agent is processed), then the rest in priority order.
    let mut ordered: Vec<usize> = Vec::with_capacity(n);
    for &i in order {
        if swap_backers.get(i).copied().unwrap_or(false) {
            ordered.push(i);
        }
    }
    for &i in order {
        if !swap_backers.get(i).copied().unwrap_or(false) {
            ordered.push(i);
        }
    }

    for &i in &ordered {
        let agent = AgentId(i as u32);
        let current = config.pos(agent);
        let goal = &goals[i];
        let is_backer = swap_backers.get(i).copied().unwrap_or(false);

        // Generate candidates.
        let neighbors: Vec<P> = if let Some(f) = neighbors_fn {
            f(current)
        } else {
            current.neighbors()
        };

        // Build candidates with lexicographic cost.
        //
        // Backer agents (Issue 144) use reverse scoring per Okumura 2023a:
        // discard guidance (guidance_mismatch = 0) and negate goal_dist so the
        // agent prefers the cell farthest from its goal (backs up). Hindrance
        // is also dropped (the reverse tuple is ⟨0, 0, −dist, 0, ε⟩).
        //
        // Non-backer agents use the full cost tuple (Issue 149):
        //   ⟨guidance_mismatch, flow_mismatch, goal_dist, hindrance, ε⟩
        // where flow_mismatch is the Guided-PIBT corridor direction penalty.
        let preferred = guidance.get(i).and_then(|g| g.first());
        let mut candidates: Vec<Candidate<P>> = neighbors
            .iter()
            .map(|next| {
                if is_backer {
                    Candidate {
                        next: next.clone(),
                        guidance_mismatch: 0,
                        flow_mismatch: 0,
                        goal_dist: -next.dist_heuristic(goal),
                        hindrance: 0.0,
                        epsilon: rng.f32(),
                    }
                } else {
                    Candidate {
                        next: next.clone(),
                        guidance_mismatch: match &preferred {
                            Some(p) => (**p != *next) as u8,
                            None => 0,
                        },
                        flow_mismatch: flow_field.mismatch(current, next),
                        goal_dist: next.dist_heuristic(goal),
                        hindrance: hindrance.hindrance(agent, next, config),
                        epsilon: rng.f32(),
                    }
                }
            })
            .collect();

        // Sort by lexicographic cost (ascending = best first).
        candidates.sort_by(|a, b| a.lexicographic_cmp(b));

        // Select the first collision-free candidate.
        // Issue 516 T1g: O(1) collision check via committed_dests +
        // current_to_agent (replaces the O(n) is_collision_free scan).
        let mut placed = false;
        for cand in &candidates {
            let next = &cand.next;
            // Vertex collision: destination already taken.
            if committed_dests.contains(next) {
                continue;
            }
            // Edge collision (swap): the agent currently at `next` (if any)
            // is committed to moving to my current position.
            if let Some(&j) = current_to_agent.get(next)
                && j != i
                && let Some(their_next) = &moves[j]
                && their_next == current
            {
                continue; // swap collision
            }
            committed_dests.insert(next.clone());
            moves[i] = Some(next.clone());
            placed = true;
            break;
        }

        if !placed {
            // Fallback: wait in place (if not itself a collision).
            let can_wait = !committed_dests.contains(current);
            let can_wait = can_wait && {
                // Edge collision check for wait-in-place.
                if let Some(&j) = current_to_agent.get(current) {
                    j == i || !matches!(&moves[j], Some(their_next) if their_next == current)
                } else {
                    true
                }
            };
            if can_wait {
                committed_dests.insert(current.clone());
                moves[i] = Some(current.clone());
            } else {
                stuck.push(agent);
            }
        }
    }

    // Place stuck agents as wait (best effort).
    for agent in &stuck {
        let i = usize::from(*agent);
        moves[i] = Some(config.pos(*agent).clone());
    }

    let final_moves: Vec<P> = moves
        .into_iter()
        .enumerate()
        .map(|(i, m)| m.unwrap_or_else(|| config.pos(AgentId(i as u32)).clone()))
        .collect();

    (final_moves, stuck)
}

/// Compute the agent processing order from priorities (descending priority,
/// ties broken by agent id for determinism).
pub(super) fn compute_priority_order(n: usize, priorities: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    if !priorities.is_empty() && priorities.len() == n {
        order.sort_by(|&a, &b| {
            priorities[b]
                .total_cmp(&priorities[a])
                .then_with(|| a.cmp(&b))
        });
    }
    order
}

/// Shuffle the priority order for a LaCAM escalation retry.
///
/// Elevates stuck agents (moves them earlier in the order) and perturbs
/// the rest slightly to break symmetric deadlocks.
///
/// Only used by the legacy shuffled-retry path (Issue 143). When
/// `lacam_escalation` is ON (Plan 453), the real constraint tree is used.
#[cfg(not(feature = "lacam_escalation"))]
fn shuffle_order(stuck: &[AgentId], order: &[usize], rng: &mut fastrand::Rng) -> Vec<usize> {
    let stuck_set: Vec<bool> = {
        let mut s = vec![false; order.len()];
        for &a in stuck {
            let idx = usize::from(a);
            if idx < s.len() {
                s[idx] = true;
            }
        }
        s
    };

    // Move stuck agents to the front (elevated priority), keep relative order
    // among the rest, with a small random perturbation.
    let mut front: Vec<usize> = Vec::new();
    let mut back: Vec<usize> = Vec::new();
    for &i in order {
        if stuck_set[i] {
            front.push(i);
        } else {
            back.push(i);
        }
    }

    // Random perturbation: occasionally swap adjacent non-stuck agents.
    if back.len() > 1 {
        for k in 0..back.len() - 1 {
            if rng.f32() < 0.3 {
                back.swap(k, k + 1);
            }
        }
    }

    front.extend(back);
    front
}

/// Detect swap-pair deadlocks (Issue 144, Okumura 2023a arXiv:2309.02425).
///
/// A swap deadlock occurs when agent `i`'s guidance-preferred next cell is
/// agent `j`'s current position, AND agent `j`'s guidance-preferred next cell
/// is agent `i`'s current position (a head-on corridor exchange). The greedy
/// PIBT cannot resolve this: both agents are blocked by the other's current
/// cell, and no priority ordering makes either move.
///
/// The swap technique resolves it by marking the lower-priority agent as a
/// **backer**. The backer uses reverse scoring `⟨0, −dist(v, g_i), ε⟩` so it
/// backs up (moves to the cell farthest from its goal), clearing the path for
/// the higher-priority forward agent.
///
/// Returns a `Vec<bool>` indexed by agent index: `true` means the agent is a
/// swap backer and should use reverse scoring in [`greedy_pibt_pass`].
///
/// # Infrastructure-only (Issue 144 negative result)
///
/// This function is benchmarked but NOT wired into the default `pibt_step`
/// escalation path. The swap technique does not improve any GOAT-gate map:
/// ht_chantry uses 2-wide corridors (agents sidestep naturally), and warehouse
/// regresses (forced back-ups reduce sustained throughput). The infrastructure
/// is kept for consumers with 1-wide corridor maps who can call this + pass
/// the result to `greedy_pibt_pass` directly.
///
/// # Complexity
///
/// O(n²) — for each agent, scan all agents for the swap partner. For n=1000
/// this is ~1M position comparisons, which completes in microseconds.
#[allow(dead_code)]
fn detect_swap_backers<P: Position>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    order: &[usize],
) -> Vec<bool> {
    let n = config.n_agents();
    if n == 0 {
        return Vec::new();
    }

    // Rank map: agent index → position in `order` (lower = higher priority).
    let mut rank = vec![0usize; n];
    for (pos, &agent) in order.iter().enumerate() {
        if agent < n {
            rank[agent] = pos;
        }
    }

    let mut is_backer = vec![false; n];
    let mut paired = vec![false; n];

    for i in 0..n {
        if paired[i] {
            continue;
        }
        let current_i = config.pos(AgentId(i as u32));
        let Some(preferred_i) = guidance.get(i).and_then(|g| g.first()) else {
            continue;
        };

        // Scan for agent j at preferred_i whose preferred cell is current_i.
        for j in 0..n {
            if j == i || paired[j] {
                continue;
            }
            let pos_j = config.pos(AgentId(j as u32));
            if *pos_j != *preferred_i {
                continue;
            }
            let preferred_j = guidance.get(j).and_then(|g| g.first());
            if let Some(pj) = preferred_j
                && *pj == *current_i
            {
                // Swap pair (i, j). Mark the lower-priority (later in
                // order) agent as the backer so the higher-priority agent
                // advances.
                let backer = if rank[i] < rank[j] { j } else { i };
                is_backer[backer] = true;
                paired[i] = true;
                paired[j] = true;
                break;
            }
        }
    }

    is_backer
}

/// Error: no collision-free joint action could be found for some agent.
///
/// Kept for API compatibility with the orchestrator's `unwrap_or_else` fallback.
/// In practice, `pibt_step` always returns `Ok` — stuck agents wait in place.
#[derive(Debug)]
pub struct Deadlock {
    /// Agents that could not be placed (must wait).
    pub stuck_agents: Vec<AgentId>,
}
