//! Bounded one-step LaCAM escalation (Plan 453, Research 441).
//!
//! Replaces the fake "LaCAM escalation" (shuffled-priority retries in
//! `pibt.rs`) with the real LaCAM mechanism: a **constraint tree** +
//! **recursive PIBT with priority inheritance**. The critical insight from
//! reading the reference implementation (`Kei18/lacam/src/planner.cpp`):
//! LaCAM DOES use recursive PIBT (the piece Issues 140/143 tried and
//! reverted), but it works because the constraint tree bounds the recursion
//! and provides systematic backtracking.
//!
//! **Scope:** one-step LaCAM — find a collision-free joint action for the
//! current tick, bounded by a node/time budget, with greedy-PIBT fallback.
//! This is NOT multi-step LaCAM* (which degrades lifelong throughput per
//! Research 424 §1.5); it's the one-step collision-freedom mechanism.
//!
//! # Algorithm
//!
//! 1. Run greedy PIBT (fast path). If no stuck agents, return immediately.
//! 2. If stuck agents exist, explore the **constraint tree**: systematically
//!    try forcing different agents to different cells, re-running recursive
//!    PIBT for each constraint. The first collision-free config found wins.
//! 3. If the budget is exhausted before finding a collision-free config,
//!    fall back to the greedy PIBT result (current behavior — collisions on
//!    congested maps, but throughput preserved).
//!
//! # The constraint tree (LaCAM's low-level search)
//!
//! Each constraint is a chain `(agent_0 → cell_0, agent_1 → cell_1, ...)`.
//! The root constraint is empty. Each child extends the parent by one
//! `(agent, cell)` assignment. The agent at each depth is determined by the
//! priority order. When a constraint is popped, `get_new_config` applies the
//! forced assignments then runs recursive PIBT for the remaining agents.
//!
//! # Why this doesn't collapse throughput (the Issue 140/143 lesson)
//!
//! Issues 140/143 implemented recursive PIBT WITHOUT the constraint tree.
//! Without backtracking, a single priority-inheritance push can cascade
//! (A pushes B, B pushes C, ...) and stall the entire system. The constraint
//! tree bounds the cascade: when recursive PIBT fails, the constraint tree
//! tries a different root assignment. See Research 441 §5 for the full
//! prior-art comparison table.

use super::config::{AgentId, JointAction, JointConfig};
use super::flow::FlowField;
use super::hindrance::HindranceEstimator;
use super::local_guidance::Guidance;
use super::pibt::{compute_priority_order, greedy_pibt_pass};
use super::position::Position;
use std::collections::{HashMap, HashSet, VecDeque};

/// Default node budget for the constraint-tree search.
///
/// At roughly 1μs per node (constraint application + recursive PIBT), this is
/// ~1ms of overhead — acceptable for a congested-map tick. On open maps the
/// constraint tree is never entered (greedy PIBT fast path), so zero overhead.
pub const DEFAULT_MAX_NODES: usize = 1000;

/// Default wall-clock budget for the constraint-tree search (microseconds).
///
/// Checked periodically (every 64 nodes) to bound latency. When exhausted,
/// falls back to greedy PIBT.
pub const DEFAULT_TIME_BUDGET_US: u64 = 5000;

/// Default maximum constraint-tree depth (Issue 546 multi-step extension).
///
/// Caps how many agents the constraint tree can force-assign. With
/// `target_stuck_agents = true`, this is the maximum number of stuck agents
/// the tree will try to break free. The ht_chantry diagnostic (commit
/// `2a8c378d`) measured P95 max-cluster-size = 8, so depth 8 is the minimum
/// useful bound on the hardest map. Higher values cover more of the tail but
/// cost latency (combinatorial expansion).
pub const DEFAULT_MAX_DEPTH: usize = 8;

/// Minimum number of stuck agents before LaCAM escalation triggers.
///
/// Re-exports the `pibt.rs` constant semantics: on open maps, a few agents
/// may get stuck each tick due to random collisions — they resolve naturally
/// next tick. The escalation is only worth its overhead on genuinely congested
/// maps (systemic stuck agents).
///
/// Exposed as `pub(super)` so the `pibt_step_with_budget` wrapper in
/// `pibt.rs` can use the same threshold when deciding whether to enter the
/// LaCAM constraint-tree search.
pub(super) const MIN_STUCK_FOR_LACAM: usize = 1;

/// Budget for the constraint-tree search.
///
/// Bounds the LaCAM escalation to maintain real-time perf. When exhausted,
/// falls back to the greedy PIBT result.
#[derive(Clone, Copy, Debug)]
pub struct EscalationBudget {
    /// Maximum number of constraint-tree nodes to explore.
    pub max_nodes: usize,
    /// Wall-clock budget in microseconds. Checked every 64 nodes.
    pub time_budget_us: u64,
    /// Maximum constraint-tree depth (Issue 546 multi-step extension).
    ///
    /// Caps how many agents the tree can force-assign. Default 8 covers the
    /// P95 cluster size on ht_chantry. Ignored when `target_stuck_agents`
    /// is false (legacy behavior expands to depth `n`).
    pub max_depth: usize,
    /// Target stuck agents in the constraint tree (Issue 546 multi-step).
    ///
    /// When `true`, the constraint tree iterates over stuck agents (computed
    /// by the initial greedy PIBT pass) instead of all agents in priority
    /// order. This makes depth-K constraints target the K stuck agents
    /// directly, dramatically reducing the search space on maps where stuck
    /// agents are deep in the priority order (ht_chantry-style maze maps).
    ///
    /// Default `false` preserves Plan 453 behavior (paper-faithful BFS over
    /// priority order). Default-on for the multi-step extension via
    /// [`EscalationBudget::multistep_default`].
    pub target_stuck_agents: bool,
}

impl Default for EscalationBudget {
    fn default() -> Self {
        Self {
            max_nodes: DEFAULT_MAX_NODES,
            time_budget_us: DEFAULT_TIME_BUDGET_US,
            max_depth: DEFAULT_MAX_DEPTH,
            target_stuck_agents: false,
        }
    }
}

impl EscalationBudget {
    /// Multi-step LaCAM defaults (Issue 546 reopened plan).
    ///
    /// Stuck-agent targeting + depth 8 + larger node/time budget for
    /// maze-class maps. Use this on ht_chantry-class maps where the
    /// paper-faithful BFS-over-priority-order cannot reach stuck agents
    /// within the default budget.
    ///
    /// Latency budget: 100ms (vs 5ms default). At ~1µs/node that's 100K
    /// nodes — well within the Issue 546 acceptance criteria (≤ 500ms on
    /// the hard map).
    pub fn multistep_default() -> Self {
        Self {
            max_nodes: 100_000,
            time_budget_us: 100_000,
            max_depth: DEFAULT_MAX_DEPTH,
            target_stuck_agents: true,
        }
    }
}

/// One constraint in the LaCAM constraint tree.
///
/// A chain of `(agent_index, forced_cell)` pairs. The root constraint is
/// empty (depth 0). Each child extends the parent by one assignment.
#[derive(Clone)]
struct Constraint<P: Position + Clone> {
    /// Agent indices, in the order they were constrained.
    who: Vec<usize>,
    /// Forced cells, parallel to `who`.
    where_cells: Vec<P>,
}

impl<P: Position + Clone> Constraint<P> {
    fn empty() -> Self {
        Self {
            who: Vec::new(),
            where_cells: Vec::new(),
        }
    }

    fn depth(&self) -> usize {
        self.who.len()
    }

    /// Create a child constraint by appending one `(agent, cell)` assignment.
    fn child(&self, agent: usize, cell: P) -> Self {
        let mut c = Constraint {
            who: Vec::with_capacity(self.who.len() + 1),
            where_cells: Vec::with_capacity(self.where_cells.len() + 1),
        };
        c.who.extend_from_slice(&self.who);
        c.where_cells.extend_from_slice(&self.where_cells);
        c.who.push(agent);
        c.where_cells.push(cell);
        c
    }
}

/// FIFO queue of constraints (BFS-style exploration).
///
/// LaCAM uses FIFO to explore shallow constraints first (fewer forced
/// assignments), which are more likely to succeed (less constraining).
struct ConstraintQueue<P: Position + Clone> {
    queue: VecDeque<Constraint<P>>,
}

impl<P: Position + Clone> ConstraintQueue<P> {
    fn with_capacity(cap: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(cap),
        }
    }

    fn push(&mut self, c: Constraint<P>) {
        self.queue.push_back(c);
    }

    fn pop(&mut self) -> Option<Constraint<P>> {
        self.queue.pop_front()
    }
}

/// Error from `get_new_config`: the constraint was rejected (collision or
/// PIBT failure). The constraint tree tries the next constraint.
#[derive(Debug)]
struct ConstraintRejected;

/// Bounded one-step LaCAM escalation.
///
/// Replaces the shuffled-priority retry loop in `pibt_step` when the
/// `lacam_escalation` feature is enabled. Runs greedy PIBT first (fast path);
/// if stuck agents exist, explores the constraint tree to find a
/// collision-free config. Falls back to greedy PIBT if the budget is
/// exhausted.
///
/// Returns `Ok(JointAction)` always — same API contract as `pibt_step`.
///
/// Marked `pub` so benchmark harnesses (e.g. Plan 453 T3.3 latency sweep)
/// can call it directly with a custom [`EscalationBudget`]. The orchestrator
/// [`LifelongLaCam::tick`](super::LifelongLaCam::tick) calls this via
/// [`pibt_step`](super::pibt::pibt_step) with `EscalationBudget::default()`.
#[allow(clippy::too_many_arguments)]
pub fn lacam_escalation_step<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    priorities: &[f32],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&super::pibt::NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    budget: EscalationBudget,
) -> JointAction<P>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let order = compute_priority_order(n, priorities);

    // Empty backer set (same as pibt_step — swap technique is infrastructure-only).
    let no_backers = vec![false; n];

    // Phase A: greedy PIBT (fast path).
    let (greedy_moves, stuck) = greedy_pibt_pass(
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

    // Fast path: no stuck agents (or too few). Return greedy result.
    if stuck.len() < MIN_STUCK_FOR_LACAM {
        return JointAction::new(greedy_moves);
    }

    // Phase B: constraint-tree search.
    //
    // Issue 546 multi-step extension: when `target_stuck_agents` is set, the
    // constraint tree iterates over stuck agents (computed above by greedy
    // PIBT) instead of all agents in priority order. This dramatically
    // reduces the search space on maze maps where stuck agents are deep in
    // the priority order.
    //
    // Build the expansion order: either the full priority order (legacy,
    // paper-faithful) or just the stuck agents (Issue 546 multi-step).
    let stuck_indices: Vec<usize> = if budget.target_stuck_agents {
        stuck.iter().map(|a| a.0 as usize).collect()
    } else {
        Vec::new()
    };
    let expansion_order: Vec<usize> = if budget.target_stuck_agents {
        stuck_indices.clone()
    } else {
        order.clone()
    };
    let expansion_depth_cap: usize = if budget.target_stuck_agents {
        // Cap at min(max_depth, stuck.len()) — can't constrain more agents
        // than are stuck, and don't exceed the configured depth bound.
        budget.max_depth.min(expansion_order.len())
    } else {
        n
    };

    let mut queue = ConstraintQueue::<P>::with_capacity(budget.max_nodes);
    queue.push(Constraint::empty());

    let mut nodes_explored: usize = 0;
    let start = std::time::Instant::now();
    let time_budget = std::time::Duration::from_micros(budget.time_budget_us);

    // Pre-build the current_to_agent map (shared across all constraint attempts).
    let mut current_to_agent: HashMap<P, usize> = HashMap::with_capacity(n);
    for (i, pos) in config.positions.iter().enumerate() {
        current_to_agent.entry(pos.clone()).or_insert(i);
    }

    while let Some(constraint) = queue.pop() {
        nodes_explored += 1;

        // Budget check: node cap.
        if nodes_explored > budget.max_nodes {
            break;
        }
        // Budget check: time cap (every 64 nodes to reduce branch overhead).
        if (nodes_explored & 63) == 0 && start.elapsed() > time_budget {
            break;
        }

        // Expand: push children for the next agent in the expansion order.
        //
        // Legacy (Plan 453): expansion_order = priority order, depth cap = n.
        // Issue 546 multi-step: expansion_order = stuck agents, depth cap =
        // min(max_depth, stuck.len()). The constraint at depth K forces the
        // K-th agent in `expansion_order` to one of its neighbor cells.
        let depth = constraint.depth();
        if depth < expansion_depth_cap {
            let i = expansion_order[depth];
            let current = config.pos(AgentId(i as u32));
            let neighbors: Vec<P> = if let Some(f) = neighbors_fn {
                f(current)
            } else {
                current.neighbors()
            };
            // Shuffle for diversity (deterministic via seeded rng).
            let mut shuffled: Vec<P> = neighbors;
            // Fisher-Yates shuffle with seeded rng.
            for k in (1..shuffled.len()).rev() {
                let j = rng.usize(0..=k);
                shuffled.swap(k, j);
            }
            for cell in shuffled {
                queue.push(constraint.child(i, cell));
            }
        }

        // Try to build a collision-free config with this constraint.
        match get_new_config(
            config,
            &constraint,
            guidance,
            goals,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
            &order,
            &current_to_agent,
        ) {
            Ok(moves) => {
                // Verify collision-free (vertex + edge).
                if is_collision_free(&moves, config) {
                    return JointAction::new(moves);
                }
            }
            Err(ConstraintRejected) => continue,
        }
    }

    // Phase C: budget exhausted — fall back to greedy PIBT result.
    let _ = stuck_indices; // suppress unused warning when not targeting
    JointAction::new(greedy_moves)
}

/// Build a collision-free next configuration by applying the constraint's
/// forced assignments, then running recursive PIBT for the remaining agents.
///
/// Adapted from `Kei18/lacam/src/planner.cpp:get_new_config`. Returns
/// `Ok(moves)` if a collision-free config was built, `Err(ConstraintRejected)`
/// if any forced assignment collides or PIBT fails for an unconstrained agent.
#[allow(clippy::too_many_arguments)]
fn get_new_config<P, H>(
    config: &JointConfig<P>,
    constraint: &Constraint<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&super::pibt::NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    order: &[usize],
    current_to_agent: &HashMap<P, usize>,
) -> Result<Vec<P>, ConstraintRejected>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let mut moves: Vec<Option<P>> = vec![None; n];
    let mut occupied_next: HashSet<P> = HashSet::with_capacity(n);
    let mut constrained_agents: HashSet<usize> = HashSet::with_capacity(constraint.depth());

    // 1. Apply constraints: force specific agents to specific cells.
    for (k, agent_i) in constraint.who.iter().enumerate() {
        let cell = &constraint.where_cells[k];
        // Vertex collision check.
        if occupied_next.contains(cell) {
            return Err(ConstraintRejected);
        }
        // Swap collision check: the agent currently at `cell` (if any) is
        // committed to moving to agent_i's current position.
        if let Some(&j) = current_to_agent.get(cell)
            && j != *agent_i
            && let Some(their_next) = &moves[j]
            && their_next == config.pos(AgentId(*agent_i as u32))
        {
            return Err(ConstraintRejected); // swap
        }
        occupied_next.insert(cell.clone());
        moves[*agent_i] = Some(cell.clone());
        constrained_agents.insert(*agent_i);
    }

    // 2. Run recursive PIBT for unconstrained agents (in priority order).
    hindrance.prepare(config);
    for &i in order {
        if constrained_agents.contains(&i) {
            continue;
        }
        if moves[i].is_some() {
            continue;
        }
        let mut pibt_state = PibtState {
            moves: &mut moves,
            occupied_next: &mut occupied_next,
            config,
            guidance,
            goals,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
            current_to_agent,
        };
        if !pibt_state.func_pibt_recursive(i) {
            return Err(ConstraintRejected);
        }
    }

    // 3. Finalize: fill any remaining None with wait-in-place (shouldn't happen
    //    if PIBT succeeded for all unconstrained agents, but defensive).
    let final_moves: Vec<P> = moves
        .into_iter()
        .enumerate()
        .map(|(i, m)| m.unwrap_or_else(|| config.pos(AgentId(i as u32)).clone()))
        .collect();

    Ok(final_moves)
}

/// Mutable state bundle for recursive PIBT.
///
/// Groups the shared mutable state (`moves`, `occupied_next`) and read-only
/// context into a single struct so the recursion signature stays manageable.
/// Mirrors the implicit `this`-bundled state in `Kei18/lacam:funcPIBT`.
struct PibtState<'a, P, H>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    moves: &'a mut Vec<Option<P>>,
    occupied_next: &'a mut HashSet<P>,
    config: &'a JointConfig<P>,
    guidance: &'a Guidance<P>,
    goals: &'a [P],
    hindrance: &'a mut H,
    flow_field: &'a dyn FlowField<P>,
    neighbors_fn: Option<&'a super::pibt::NeighborFn<P>>,
    rng: &'a mut fastrand::Rng,
    current_to_agent: &'a HashMap<P, usize>,
}

impl<'a, P, H> PibtState<'a, P, H>
where
    P: Position,
    H: HindranceEstimator<P>,
{
    /// Recursive PIBT with priority inheritance (LaCAM's `funcPIBT`).
    ///
    /// Returns `true` if agent `i` was successfully placed, `false` if it
    /// could not find a collision-free cell (including via PI pushes). The
    /// caller (`get_new_config`) rejects the constraint on `false`.
    ///
    /// The recursion is bounded by the constraint tree: when this returns
    /// false, the constraint tree tries a different root assignment. This is
    /// why recursive PI is safe here but collapsed throughput in Issues
    /// 140/143 (which had no constraint tree).
    fn func_pibt_recursive(&mut self, i: usize) -> bool {
        let agent = AgentId(i as u32);
        let current = self.config.pos(agent).clone();

        // Generate candidates: neighbors + wait, sorted by lexicographic cost.
        let neighbors: Vec<P> = if let Some(f) = self.neighbors_fn {
            f(&current)
        } else {
            current.neighbors()
        };

        let goal = &self.goals[i];
        let preferred = self.guidance.get(i).and_then(|g| g.first());

        // Build candidates with the same lexicographic cost as greedy PIBT.
        let mut candidates: Vec<super::pibt::Candidate<P>> = neighbors
            .iter()
            .map(|next| super::pibt::Candidate {
                next: next.clone(),
                guidance_mismatch: match &preferred {
                    Some(p) => (**p != *next) as u8,
                    None => 0,
                },
                flow_mismatch: self.flow_field.mismatch(&current, next),
                goal_dist: next.dist_heuristic(goal),
                hindrance: self.hindrance.hindrance(agent, next, self.config),
                epsilon: self.rng.f32(),
            })
            .collect();
        candidates.sort_by(|a, b| a.lexicographic_cmp(b));

        for cand in &candidates {
            let next = &cand.next;
            // Vertex collision check.
            if self.occupied_next.contains(next) {
                continue;
            }
            // Edge collision (swap) check.
            if let Some(&j) = self.current_to_agent.get(next)
                && j != i
                && let Some(their_next) = &self.moves[j]
                && their_next == &current
            {
                continue; // swap collision
            }

            // Reserve the cell.
            self.occupied_next.insert(next.clone());
            self.moves[i] = Some(next.clone());

            // Check the current occupant of `next`.
            let occupant = self.current_to_agent.get(next).copied();
            // Empty cell or staying → success.
            if occupant.is_none() || next == &current {
                return true;
            }

            // Priority inheritance: push the occupant.
            let k = occupant.unwrap();
            if k != i && self.moves[k].is_none() {
                if self.func_pibt_recursive(k) {
                    return true; // occupant moved, we get the cell
                }
                // Occupant couldn't move — undo our reservation and try next.
                self.occupied_next.remove(next);
                self.moves[i] = None;
                continue;
            }

            return true;
        }

        // Failed to find a collision-free cell. Stay in place if possible.
        let can_wait = !self.occupied_next.contains(&current);
        if can_wait {
            self.occupied_next.insert(current.clone());
            self.moves[i] = Some(current);
            true
        } else {
            false
        }
    }
}

/// Check if a joint action is collision-free (no vertex + no edge collisions).
///
/// Used to verify the output of `get_new_config` before accepting it. The
/// constraint application + recursive PIBT should always produce a
/// collision-free config when they succeed, but this defensive check guards
/// against bugs in the constraint/PIBT logic.
fn is_collision_free<P: Position>(moves: &[P], config: &JointConfig<P>) -> bool {
    let n = moves.len();
    // Vertex: all next positions distinct.
    let mut seen: HashSet<&P> = HashSet::with_capacity(n);
    for p in moves {
        if !seen.insert(p) {
            return false;
        }
    }
    // Edge: no swaps.
    for i in 0..n {
        for j in (i + 1)..n {
            let i_cur = &config.positions[i];
            let j_cur = &config.positions[j];
            if &moves[i] == j_cur && &moves[j] == i_cur {
                return false; // swap
            }
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────
// Constraint-tree census (riir-ai Issue 1010 T1 — measurement-only).
//
// Counts, per constraint-tree depth: the constraint chains expanded, how
// many admit a collision-free completion (live), how many are dead, and
// how many DISTINCT joint configurations the live chains collapse to.
// This is the pre-build duplicate-work measurement for the merged-frontier
// arm: chains/consumers that re-derive the same configuration are exactly
// the merge opportunity (and the dead share is what a modelless judge
// could prune). Reuses the shipped tree mechanics verbatim (`Constraint`,
// `get_new_config`, `is_collision_free`) so the census cannot drift from
// the search it measures; the only behavioral difference is that it does
// NOT early-return on the first success (production stops there).
// ─────────────────────────────────────────────────────────────────────

/// Exploration bounds for [`lacam_constraint_tree_census`].
#[derive(Clone, Copy, Debug)]
pub struct CensusLimits {
    /// Maximum constraint-chain depth to explore. Chains AT this depth are
    /// expanded; their children are not pushed. Capped at the escalation's
    /// own depth semantics (`min(max_depth, stuck.len())` when
    /// `target_stuck_agents`, else the agent count).
    pub max_depth: usize,
    /// Maximum total chains to expand — the deterministic runaway bound.
    /// When hit, [`CensusReport::truncated`] is set (the tree was larger
    /// than measured).
    pub max_chains: usize,
}

impl Default for CensusLimits {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_chains: 50_000,
        }
    }
}

/// Per-depth census row: one level of the constraint tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CensusDepthRow {
    /// Constraint-tree depth (0 = the empty root constraint).
    pub depth: usize,
    /// Constraint chains expanded at this depth.
    pub chains: usize,
    /// Chains whose expansion produced a collision-free joint action.
    pub live: usize,
    /// Chains that admit no collision-free completion in the one-step tree:
    /// recursive PIBT rejected (some agent stuck — the ANY-agent-stuck
    /// quantifier) or the produced action failed the vertex/edge check.
    /// Connectivity ("goal unreachable") is deliberately NOT part of this
    /// dead test — it is a separate optional pruner.
    pub dead: usize,
    /// Distinct joint configurations (agent-indexed next-position vectors)
    /// among this depth's live chains — the merge-key space at this depth.
    pub distinct_configs: usize,
}

impl CensusDepthRow {
    /// Dead share at this depth in `[0, 1]` (0.0 when nothing was expanded).
    pub fn dead_share(&self) -> f64 {
        if self.chains == 0 {
            return 0.0;
        }
        self.dead as f64 / self.chains as f64
    }

    /// Path-compression ratio at this depth: live chains per distinct
    /// configuration (1.0 = every chain yields its own config; 0.0 when no
    /// chain was live — the ratio is undefined for an all-dead depth).
    pub fn compression(&self) -> f64 {
        if self.distinct_configs == 0 {
            return 0.0;
        }
        self.live as f64 / self.distinct_configs as f64
    }
}

/// Full census of one escalation state's constraint tree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CensusReport {
    /// One row per explored depth, `0..=max_depth`.
    pub rows: Vec<CensusDepthRow>,
    /// Total chains expanded across all depths.
    pub total_chains: usize,
    /// Total live chains (collision-free completion found).
    pub total_live: usize,
    /// Total dead chains.
    pub total_dead: usize,
    /// Distinct configurations across ALL depths of this state (the union
    /// key space a merged-frontier arm would store for this expansion).
    pub distinct_configs_global: usize,
    /// Stuck agents from the greedy PIBT pass — the escalation trigger.
    /// `0` means the tree was never entered (greedy fast path).
    pub stuck_agents: usize,
    /// Depth at which the FIRST live chain was found in BFS order — i.e.
    /// where the shipped early-return (`lacam_escalation_step`) stops.
    /// `None` when no chain was live within the census bounds.
    pub first_live_depth: Option<usize>,
    /// True when `CensusLimits::max_chains` was hit before the tree was
    /// fully explored.
    pub truncated: bool,
}

impl CensusReport {
    /// Whether the constraint tree was entered at all (`stuck >= 1`).
    pub fn tree_entered(&self) -> bool {
        self.stuck_agents >= MIN_STUCK_FOR_LACAM
    }

    /// Overall dead share in `[0, 1]` (0.0 when nothing was expanded).
    pub fn dead_share(&self) -> f64 {
        if self.total_chains == 0 {
            return 0.0;
        }
        self.total_dead as f64 / self.total_chains as f64
    }

    /// Overall path-compression ratio: live chains per distinct
    /// configuration (0.0 when nothing was live).
    pub fn compression(&self) -> f64 {
        if self.distinct_configs_global == 0 {
            return 0.0;
        }
        self.total_live as f64 / self.distinct_configs_global as f64
    }
}

/// Census the LaCAM constraint tree for one escalation state (Issue 1010
/// T1 — duplicate-work measurement, no search behavior change).
///
/// Mirrors [`lacam_escalation_step`] phase-for-phase — same greedy trigger,
/// same expansion order, same child generation, same `get_new_config` +
/// collision check — except it never early-returns: every chain within the
/// census bounds is expanded and counted. All randomness flows through the
/// caller-supplied seeded `rng`; no global state is touched.
#[allow(clippy::too_many_arguments)]
pub fn lacam_constraint_tree_census<P, H>(
    config: &JointConfig<P>,
    guidance: &Guidance<P>,
    goals: &[P],
    priorities: &[f32],
    hindrance: &mut H,
    flow_field: &dyn FlowField<P>,
    neighbors_fn: Option<&super::pibt::NeighborFn<P>>,
    rng: &mut fastrand::Rng,
    budget: EscalationBudget,
    limits: CensusLimits,
) -> CensusReport
where
    P: Position,
    H: HindranceEstimator<P>,
{
    let n = config.n_agents();
    let order = compute_priority_order(n, priorities);

    // Phase A: greedy PIBT — the same escalation trigger the search uses.
    let no_backers = vec![false; n];
    let (_greedy_moves, stuck) = greedy_pibt_pass(
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

    let mut report = CensusReport {
        stuck_agents: stuck.len(),
        ..CensusReport::default()
    };
    if !report.tree_entered() {
        return report; // greedy fast path — the tree does not exist this tick
    }

    // Mirror the escalation's expansion order + depth semantics exactly.
    let expansion_order: Vec<usize> = if budget.target_stuck_agents {
        stuck.iter().map(|a| a.0 as usize).collect()
    } else {
        order.clone()
    };
    let expansion_depth_cap = if budget.target_stuck_agents {
        budget.max_depth.min(expansion_order.len())
    } else {
        n
    };
    let census_depth_cap = limits.max_depth.min(expansion_depth_cap);

    let mut queue = ConstraintQueue::<P>::with_capacity(1024);
    queue.push(Constraint::empty());

    let mut current_to_agent: HashMap<P, usize> = HashMap::with_capacity(n);
    for (i, pos) in config.positions.iter().enumerate() {
        current_to_agent.entry(pos.clone()).or_insert(i);
    }

    let mut chains_per_depth = vec![0usize; census_depth_cap + 1];
    let mut live_per_depth = vec![0usize; census_depth_cap + 1];
    let mut dead_per_depth = vec![0usize; census_depth_cap + 1];
    let mut distinct_per_depth: Vec<HashSet<Vec<P>>> =
        (0..=census_depth_cap).map(|_| HashSet::new()).collect();
    let mut distinct_global: HashSet<Vec<P>> = HashSet::new();

    while let Some(constraint) = queue.pop() {
        if report.total_chains >= limits.max_chains {
            report.truncated = true;
            break;
        }
        let depth = constraint.depth();

        // Push children FIRST — mirrors the shipped loop (children are
        // queued regardless of this chain's expansion outcome).
        if depth < census_depth_cap {
            let i = expansion_order[depth];
            let current = config.pos(AgentId(i as u32));
            let neighbors: Vec<P> = if let Some(f) = neighbors_fn {
                f(current)
            } else {
                current.neighbors()
            };
            // Fisher-Yates shuffle with seeded rng (same as the escalation).
            let mut shuffled: Vec<P> = neighbors;
            for k in (1..shuffled.len()).rev() {
                let j = rng.usize(0..=k);
                shuffled.swap(k, j);
            }
            for cell in shuffled {
                queue.push(constraint.child(i, cell));
            }
        }

        // Expand this chain: forced assignments + recursive PIBT, then the
        // vertex/edge check. Live = collision-free completion exists.
        chains_per_depth[depth] += 1;
        report.total_chains += 1;
        match get_new_config(
            config,
            &constraint,
            guidance,
            goals,
            hindrance,
            flow_field,
            neighbors_fn,
            rng,
            &order,
            &current_to_agent,
        ) {
            Ok(moves) if is_collision_free(&moves, config) => {
                live_per_depth[depth] += 1;
                report.total_live += 1;
                if report.first_live_depth.is_none() {
                    report.first_live_depth = Some(depth);
                }
                distinct_per_depth[depth].insert(moves.clone());
                distinct_global.insert(moves);
            }
            _ => {
                dead_per_depth[depth] += 1;
                report.total_dead += 1;
            }
        }
    }

    report.distinct_configs_global = distinct_global.len();
    report.rows = (0..=census_depth_cap)
        .map(|d| CensusDepthRow {
            depth: d,
            chains: chains_per_depth[d],
            live: live_per_depth[d],
            dead: dead_per_depth[d],
            distinct_configs: distinct_per_depth[d].len(),
        })
        .collect();
    report
}

#[cfg(test)]
mod census_tests {
    use super::*;
    use crate::multi_agent_path::NeighborFn;
    use crate::multi_agent_path::local_guidance::LocalGuidanceSource;
    use crate::multi_agent_path::{
        BlockingCount, GridMap, GridPos, GuidanceConfig, NoFlow, SpaceTimeGuidance,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Aggregated per-depth counters across many census states (ticks).
    #[derive(Default)]
    struct DepthAgg {
        chains: usize,
        live: usize,
        dead: usize,
        /// Σ of per-state distinct counts (per-expansion-state distinctness,
        /// summed — NOT a cross-state dedup; states are unrelated).
        distinct: usize,
    }

    struct WorkloadAgg {
        rows: Vec<DepthAgg>,
        total_chains: usize,
        total_live: usize,
        total_dead: usize,
        distinct_sum: usize,
        /// Census states explored (ticks where the tree was entered).
        states: usize,
        tree_entered_ticks: usize,
        max_stuck: usize,
        first_live_hist: BTreeMap<usize, usize>,
        truncated_any: bool,
    }

    impl WorkloadAgg {
        fn new(max_depth: usize) -> Self {
            Self {
                rows: (0..=max_depth).map(|_| DepthAgg::default()).collect(),
                total_chains: 0,
                total_live: 0,
                total_dead: 0,
                distinct_sum: 0,
                states: 0,
                tree_entered_ticks: 0,
                max_stuck: 0,
                first_live_hist: BTreeMap::new(),
                truncated_any: false,
            }
        }

        fn push(&mut self, r: &CensusReport) {
            if !r.tree_entered() {
                return; // greedy fast path this tick — no tree, nothing to count
            }
            self.tree_entered_ticks += 1;
            self.max_stuck = self.max_stuck.max(r.stuck_agents);
            self.states += 1;
            self.total_chains += r.total_chains;
            self.total_live += r.total_live;
            self.total_dead += r.total_dead;
            self.distinct_sum += r.distinct_configs_global;
            self.truncated_any |= r.truncated;
            if let Some(d) = r.first_live_depth {
                *self.first_live_hist.entry(d).or_insert(0) += 1;
            }
            for row in &r.rows {
                let agg = &mut self.rows[row.depth];
                agg.chains += row.chains;
                agg.live += row.live;
                agg.dead += row.dead;
                agg.distinct += row.distinct_configs;
            }
        }

        fn print(&self, name: &str, grid: &str, agents: usize, ticks: usize) {
            println!(
                "workload={name} grid={grid} agents={agents} ticks_simulated={ticks} \
                 tree_entered={}/{} census_states={} max_stuck={} truncated={}",
                self.tree_entered_ticks,
                ticks,
                self.states,
                self.max_stuck,
                self.truncated_any,
            );
            if self.states == 0 {
                println!("  (greedy fast path on every tick — constraint tree never entered)");
                return;
            }
            // NOTE: per-depth compression is structurally 1.0 — two chains at
            // the same depth differ in a forced cell, so their configs differ
            // at that agent. The merge-relevant collapse is CROSS-DEPTH and is
            // therefore read off the TOTAL line only (live chains vs the
            // distinct-config union). Collapse is measured under the shipped
            // rng tiebreaks (a same-semantics chain whose PIBT epsilons flip a
            // near-tie counts as a distinct config — the honest, exploitable
            // number).
            println!("  depth | chains |   live |   dead | dead%   | distinct*");
            for (d, agg) in self.rows.iter().enumerate() {
                if agg.chains == 0 {
                    continue;
                }
                println!(
                    "  {d:>5} | {a:>6} | {l:>6} | {x:>6} | {:>6.1}% | {ds:>9}",
                    100.0 * agg.dead as f64 / agg.chains as f64,
                    a = agg.chains,
                    l = agg.live,
                    x = agg.dead,
                    ds = agg.distinct,
                );
            }
            let compr_total = if self.distinct_sum == 0 {
                0.0
            } else {
                self.total_live as f64 / self.distinct_sum as f64
            };
            let c_le2: usize = self.rows.iter().take(3).map(|a| a.chains).sum();
            let d_le2: usize = self.rows.iter().take(3).map(|a| a.dead).sum();
            let dead_le2 = if c_le2 == 0 {
                0.0
            } else {
                100.0 * d_le2 as f64 / c_le2 as f64
            };
            println!(
                "  TOTAL: chains={} live={} dead={} dead-share={:.1}% distinct*={} compression={:.2}x | dead-share(≤depth2)={:.1}% | first-live-depth hist {:?}",
                self.total_chains,
                self.total_live,
                self.total_dead,
                100.0 * self.total_dead as f64 / self.total_chains.max(1) as f64,
                self.distinct_sum,
                compr_total,
                dead_le2,
                self.first_live_hist,
            );
        }
    }

    /// Simulate `ticks` ticks with the production one-step search and census
    /// every tick's constraint tree (dedicated seeded census RNG per tick —
    /// the census never perturbs the simulated state).
    #[allow(clippy::too_many_arguments)]
    fn run_census_workload(
        map: &GridMap,
        starts: Vec<GridPos>,
        goals: Vec<GridPos>,
        guidance_cfg: GuidanceConfig,
        ticks: usize,
        sim_seed: u64,
        limits: CensusLimits,
    ) -> WorkloadAgg {
        let n = starts.len();
        let mut guidance = SpaceTimeGuidance::new(guidance_cfg).with_neighbors({
            let m = map.clone();
            move |p| m.passable_neighbors(p)
        });
        let mut hindrance = BlockingCount::new();
        let priorities = vec![1.0f32; n];
        let map_arc = Arc::new(map.clone());
        let neighbor_fn = move |p: &GridPos| map_arc.passable_neighbors(p);

        let mut agg = WorkloadAgg::new(limits.max_depth);
        let mut current = JointConfig::new(starts);
        let mut sim_rng = fastrand::Rng::with_seed(sim_seed);

        for tick in 0..ticks {
            let mut scratch: Guidance<GridPos> = Vec::new();
            guidance.compute_guidance(&current, &goals, &mut scratch);
            let nf: &NeighborFn<GridPos> = &neighbor_fn;

            // Census FIRST on this tick's state, with its own rng stream.
            let mut census_rng = fastrand::Rng::with_seed(101_000 + tick as u64);
            let report = lacam_constraint_tree_census(
                &current,
                &scratch,
                &goals,
                &priorities,
                &mut hindrance,
                &NoFlow,
                Some(nf),
                &mut census_rng,
                EscalationBudget::default(),
                limits,
            );
            agg.push(&report);

            // Then advance the simulated state with the production search.
            let action = lacam_escalation_step(
                &current,
                &scratch,
                &goals,
                &priorities,
                &mut hindrance,
                &NoFlow,
                Some(nf),
                &mut sim_rng,
                EscalationBudget::default(),
            );
            current = JointConfig::new(action.moves);
        }
        agg
    }

    fn census_verdict(agg: &WorkloadAgg) -> String {
        if agg.states == 0 {
            return "N/A (constraint tree never entered)".to_string();
        }
        let compr = if agg.distinct_sum == 0 {
            0.0
        } else {
            agg.total_live as f64 / agg.distinct_sum as f64
        };
        let c_le2: usize = agg.rows.iter().take(3).map(|a| a.chains).sum();
        let d_le2: usize = agg.rows.iter().take(3).map(|a| a.dead).sum();
        let dead_le2 = if c_le2 == 0 {
            0.0
        } else {
            d_le2 as f64 / c_le2 as f64
        };
        let compr_ok = compr >= 3.0;
        let dead_ok = dead_le2 >= 0.5;
        let legs = format!(
            "compr={compr:.2}x (>=3x:{compr_ok}) dead<=d2={:.1}% (>=50%:{dead_ok})",
            100.0 * dead_le2
        );
        let verdict = match (compr_ok, dead_ok) {
            (true, true) => "GO",
            (false, false) => "NEGATIVE",
            _ => "MIXED",
        };
        format!("{verdict} [{legs}]")
    }

    /// Issue 1010 T1 — constraint-tree duplicate-work census over three
    /// representative workloads. Prints the tables (read with `--nocapture`);
    /// asserts only structural sanity (the census is non-vacuous on the
    /// congested workload), never the GO/NEGATIVE gate itself.
    #[test]
    fn constraint_tree_census_tables() {
        println!("=== LaCAM constraint-tree census — riir-ai Issue 1010 T1 ===");
        println!("mode: production default budget (paper-faithful BFS over priority order)");

        // W1 — open map, 10 agents (shape of tests.rs `test_throughput_sanity`).
        let w1 = run_census_workload(
            &GridMap::empty(10, 10),
            (0..10).map(|i| GridPos::new(i, 0)).collect(),
            (0..10).map(|i| GridPos::new(i, 9)).collect(),
            GuidanceConfig {
                w_phi: 5,
                alpha: 2.0,
                rounds: 2,
                max_expansions: 0,
            },
            30,
            123,
            CensusLimits {
                max_depth: 5,
                max_chains: 20_000,
            },
        );
        w1.print("open10", "10x10", 10, 30);

        // W2 — congested bottleneck, 60 agents (shape of bench_453's G6c
        // scenario: 20x20, wall at x=10, 6-cell gap rows 7..=12).
        let mut map = GridMap::empty(20, 20);
        for y in 0..20 {
            if !(7..=12).contains(&y) {
                map.set_wall(10, y);
            }
        }
        let left: Vec<GridPos> = (0..20)
            .flat_map(|y| (0..10).map(move |x| GridPos::new(x, y)))
            .filter(|p| map.is_passable(p.x, p.y))
            .collect();
        let right: Vec<GridPos> = (0..20)
            .flat_map(|y| (11..20).map(move |x| GridPos::new(x, y)))
            .filter(|p| map.is_passable(p.x, p.y))
            .collect();
        let starts: Vec<GridPos> = left.iter().take(60).cloned().collect();
        let goals: Vec<GridPos> = (0..60).map(|i| right[i % right.len()]).collect();
        let w2 = run_census_workload(
            &map,
            starts,
            goals,
            GuidanceConfig {
                w_phi: 5,
                alpha: 1.0,
                rounds: 2,
                max_expansions: 0,
            },
            40,
            7,
            CensusLimits {
                max_depth: 5,
                max_chains: 20_000,
            },
        );
        w2.print("bottleneck60", "20x20-gap6", 60, 40);
        // Non-vacuousness floor: the congested workload MUST exercise the tree.
        assert!(
            w2.states > 0,
            "bottleneck60 census is vacuous — constraint tree never entered"
        );
        assert!(w2.total_chains > 0);

        // W3 — 1-wide corridor deadlock, 2 agents (shape of tests.rs
        // `test_deadlock_corridor_falls_back_to_wait`).
        let w3 = run_census_workload(
            &GridMap::empty(3, 1),
            vec![GridPos::new(0, 0), GridPos::new(2, 0)],
            vec![GridPos::new(2, 0), GridPos::new(0, 0)],
            GuidanceConfig::default(),
            10,
            7,
            CensusLimits {
                max_depth: 3,
                max_chains: 1_000,
            },
        );
        w3.print("corridor2", "3x1", 2, 10);

        println!("=== census gate read (compression ≥3x AND dead-share≤d2 ≥50% => GO) ===");
        for (name, agg) in [("open10", &w1), ("bottleneck60", &w2), ("corridor2", &w3)] {
            println!("  {name}: {}", census_verdict(agg));
        }

        // Determinism: identical seeds must reproduce identical aggregates.
        let mut map = GridMap::empty(20, 20);
        for y in 0..20 {
            if !(7..=12).contains(&y) {
                map.set_wall(10, y);
            }
        }
        let left: Vec<GridPos> = (0..20)
            .flat_map(|y| (0..10).map(move |x| GridPos::new(x, y)))
            .filter(|p| map.is_passable(p.x, p.y))
            .collect();
        let right: Vec<GridPos> = (0..20)
            .flat_map(|y| (11..20).map(move |x| GridPos::new(x, y)))
            .filter(|p| map.is_passable(p.x, p.y))
            .collect();
        let w2_rerun = run_census_workload(
            &map,
            left.iter().take(60).cloned().collect(),
            (0..60).map(|i| right[i % right.len()]).collect(),
            GuidanceConfig {
                w_phi: 5,
                alpha: 1.0,
                rounds: 2,
                max_expansions: 0,
            },
            40,
            7,
            CensusLimits {
                max_depth: 5,
                max_chains: 20_000,
            },
        );
        assert_eq!(w2.total_chains, w2_rerun.total_chains);
        assert_eq!(w2.total_live, w2_rerun.total_live);
        assert_eq!(w2.total_dead, w2_rerun.total_dead);
        assert_eq!(w2.distinct_sum, w2_rerun.distinct_sum);
        assert_eq!(w2.states, w2_rerun.states);
    }
}
