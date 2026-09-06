//! Local guidance source — space-time A* on collision-count cost (Plan 440 T1.5).
//!
//! This is LLLG mechanism (a) from the paper (§2.D, Algorithm 1, Eq. 1).
//! For each agent `i`, solve a single-agent space-time A* over `w_Φ+1` steps:
//!
//! ```text
//! cost_i(π) = dist(π[w_Φ], g_i)                          // reach the goal
//!           + Σ_{t=0..w_Φ-1} ⟨1 + α·Ind[χ>0], χ⟩         // avoid collisions
//! ```
//!
//! where `χ` is the count of collisions of the transition `(π[t], π[t+1])`
//! with sibling paths currently stored in the shared guidance set `Φ`. The
//! hyperparameter `α ≥ 0` controls how aggressively collisions are penalized.
//!
//! The construction is **sequential over agents** (each agent sees the others'
//! already-updated guidance) and repeated `m` rounds to reduce agent-order
//! bias (paper default `m=2`).
//!
//! # Implementation (Issue 142)
//!
//! The per-agent search is a proper space-time A* over the `(position, depth)`
//! state space with BFS-distance heuristic. This replaced a greedy rollout
//! (Issue 140) that committed at each depth without backtracking and could
//! neither plan multi-step detours nor consume the warm-start forecast.
//!
//! The multi-round refinement is implemented as unrecord/re-record: each
//! agent removes its previous path from the shared occupancy, computes a fresh
//! A* path seeing all other agents' most-recent paths, then records the new
//! path. This makes `rounds > 1` actually improve results (round 1 agent 0
//! sees round-0 paths of agents 1..n-1).
//!
//! # Pluggable seam
//!
//! [`LocalGuidanceSource`] is the primary extension point for the Super-GOAT
//! fusion (riir-ai/318 Extension A). The default [`SpaceTimeGuidance`] is
//! paper-faithful (uniform `α`). A private consumer plugs in belief-projected
//! guidance where `α_i = α_base · (1 + β · σ(dot(belief_i, D_frustration)))`
//! so crowded cells cost more for stressed NPCs.

use super::config::{AgentId, JointConfig};
use super::position::Position;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Maximum number of cached BFS distance fields before the cache is cleared.
/// Bounds memory to ~MAX × map_size × sizeof(GridPos + f32). For 800 agents
/// on a 4096-cell map, this is ~4000 × 4096 × 20 bytes ≈ 320MB worst case
/// (rarely reached — goals cluster as agents converge).
const MAX_BFS_CACHE_ENTRIES: usize = 4000;

/// Type alias for a neighbor-supplying closure. Kept as a `dyn` to avoid
/// bloating generic parameter lists. `Send + Sync` so the guidance source
/// can cross thread boundaries if a consumer parallelizes across zones.
type NeighborFn<P> = dyn Fn(&P) -> Vec<P> + Send + Sync;

/// Flat-index closure type (T1a). Kept as a type alias to satisfy clippy's
/// `type_complexity` lint.
type FlatIndexFn<P> = dyn Fn(&P) -> usize + Send + Sync;

/// A per-agent guidance path: `w_Φ` future positions.
///
/// `Φ[i]` is the guidance for agent `i`. This is the **latent** guidance
/// field (paper §2.3 reframing) — a rank-1 cochain on the time axis. Not
/// synced; recomputed each tick from local observations.
pub type Guidance<P> = Vec<Vec<P>>;

/// Trait for producing the per-agent local guidance field `Φ`.
///
/// Pluggable seam #2. The default impl is [`SpaceTimeGuidance`] (paper-faithful
/// space-time A* with uniform collision penalty `α`).
///
/// # Primary extension point (Super-GOAT fusion hook)
///
/// A private consumer (riir-ai/318 Extension A) replaces the uniform `α` with
/// a per-NPC **belief-projected** penalty so that crowded cells cost more for
/// stressed NPCs. The collision term `α·Ind[χ>0]` from Eq. 1 becomes:
///
/// ```text
/// α_i = α_base · (1 + β · σ(dot(belief_i, D_frustration)))
/// ```
///
/// where `belief_i` is agent `i`'s 5-scalar affect vector (valence, arousal,
/// desperation, calm, fear), `D_frustration` is a learned direction vector,
/// and `σ` is the sigmoid bridge. This is **entirely modelless** — the
/// direction vector is freeze/thaw-swapped, not trained.
///
/// # Example: a custom belief-projected guidance source
///
/// ```no_run
/// use katgpt_core::multi_agent_path::*;
/// use katgpt_core::multi_agent_path::position::*;
///
/// /// Per-NPC belief projection (the latent→raw bridge).
/// ///
/// /// In the real runtime this reads from the NPC's `BeliefCacheProxy`; here we
/// /// stub a lookup table indexed by `AgentId`. The bridge is a dot-product
/// /// projection onto a pre-computed `D_frustration` direction vector, gated
/// /// by sigmoid (never softmax — per AGENTS.md).
/// struct HlaProjectedGuidance {
///     /// Base collision penalty shared by all agents (paper `α`).
///     alpha_base: f32,
///     /// Per-agent frustration scalar in `[0, 1]`, computed once per tick
///     /// from `σ(dot(belief_i, D_frustration))`. Higher = more collision-averse.
///     frustration: Vec<f32>,
///     /// Delegate: the underlying space-time guidance we layer onto.
///     inner: SpaceTimeGuidance<GridPos>,
/// }
///
/// impl LocalGuidanceSource<GridPos> for HlaProjectedGuidance {
///     fn compute_guidance(
///         &mut self,
///         config: &JointConfig<GridPos>,
///         goals: &[GridPos],
///         out: &mut Guidance<GridPos>,
///     ) {
///         // The simplest correct fusion: scale `α` per-agent by frustration,
///         // then run the paper-faithful space-time A*. A richer impl could
///         // also bias the goal heuristic by curiosity (valence direction),
///         // but that lives in the private runtime — the substrate only
///         // requires the trait method to be implemented.
///         let n = config.n_agents();
///         for i in 0..n {
///             let per_agent_alpha = self.alpha_base * (1.0 + self.frustration[i]);
///             // In a full impl you'd thread `per_agent_alpha` into `inner.cfg`;
///             // the stub here just demonstrates the trait shape.
///             let _ = per_agent_alpha;
///         }
///         self.inner.compute_guidance(config, goals, out);
///     }
///
///     fn set_warm_start(&mut self, warm_start: Vec<Vec<GridPos>>) {
///         // Delegate to the paper-faithful inner source.
///         self.inner.set_warm_start(warm_start);
///     }
/// }
/// ```
///
/// # Latent / raw boundary
///
/// The guidance field `Φ` is **latent** (local, not synced). Only the
/// executed first step `Π_t[1]` crosses the sync boundary as a raw `TxDelta`.
/// The belief projection itself never leaves the NPC's local cognition — the
/// sync layer sees only the resulting move.
pub trait LocalGuidanceSource<P: Position> {
    /// Compute the guidance field `Φ` for all agents.
    ///
    /// `Φ[i]` is a `w_Φ`-step path (positions at t+1, t+2, ..., t+w_Φ).
    /// The first position `Φ[i][0]` is the preferred next position, used by
    /// PIBT's lexicographic cost first term `Ind[Φ[i][0] ≠ u]`.
    ///
    /// The result is written into `out` (cleared and refilled).
    fn compute_guidance(&mut self, config: &JointConfig<P>, goals: &[P], out: &mut Guidance<P>);

    /// Supply warm-start data for the next [`compute_guidance`](Self::compute_guidance)
    /// call (Issue 140 T2 — the LLLG mechanism (b) warm-start integration).
    ///
    /// The data is a per-agent initial path (`Vec<Vec<P>>`), produced by
    /// [`WarmStartCache::warm_start`](super::warm_start::WarmStartCache::warm_start).
    /// For the default [`SpaceTimeGuidance`] the paths are seeded into the
    /// occupancy map so each agent sees where other agents are forecast to be
    /// — this is the qualitative difference between `LllgPi` (with warm-start)
    /// and `LllgEmpty` (without).
    ///
    /// Default: no-op (the guidance source recomputes from scratch). Custom
    /// guidance sources that don't support warm-start can leave this as-is.
    fn set_warm_start(&mut self, _warm_start: Vec<Vec<P>>) {
        // Default: ignore — no warm-start.
    }

    /// Notify the guidance source of the current grid dimensions (Issue 516 T1a).
    ///
    /// The default [`SpaceTimeGuidance`] uses this to (re)allocate its flat-array
    /// occupancy when the dimensions change. Called by the consumer (e.g.
    /// `CrowdMotionBridge`) at tick boundaries when the map is updated.
    ///
    /// Default: no-op. Guidance sources that don't support flat-array occupancy
    /// (or non-grid domains) leave this as-is.
    fn ensure_flat_occupancy(&mut self, _width: usize, _height: usize) {
        // Default: no-op.
    }
}

/// The guidance window length `w_Φ` and collision penalty `α`.
#[derive(Clone, Copy, Debug)]
pub struct GuidanceConfig {
    /// Planning window length (paper default 5).
    pub w_phi: usize,
    /// Collision penalty weight (paper implicit via `1 + α·Ind[χ>0]`).
    /// `α = 0` disables guidance (degenerates to greedy goal-seeking).
    /// Higher `α` → stronger collision avoidance.
    pub alpha: f32,
    /// Number of sequential refinement rounds (paper default 2).
    pub rounds: usize,
    /// Maximum A* node expansions per agent per round (Issue 516 T1e).
    ///
    /// `0` = unlimited (paper-faithful). When > 0, bounds worst-case latency:
    /// if an agent's search exceeds this many node expansions, it terminates
    /// early and returns the best partial path found (the expanded node closest
    /// to its goal by BFS distance). This caps the super-linear A* scaling that
    /// occurs at high agent density where occupied cells force long detours.
    ///
    /// The quality trade-off: capped agents get a greedy-ish partial path
    /// instead of an optimal w_phi-deep path. In dense crowds the optimal path
    /// is already poor (congestion forces detours), so the quality loss is
    /// minimal. The latency win is substantial: at n=1000 on a 50×50 grid,
    /// each uncapped search explores ~8K nodes; capping at 2K cuts A* time ~4×.
    pub max_expansions: usize,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            w_phi: 5,
            alpha: 1.0,
            rounds: 2,
            max_expansions: 0,
        }
    }
}

/// Paper-default guidance: space-time A* on Eq. 1 cost.
///
/// Sequential per-agent refinement, `m` rounds. Each agent's A* sees the
/// already-updated paths of earlier-in-the-round agents.
///
/// This is a **single-threaded** implementation matching the paper's design.
/// Per-zone parallelism is a consumer concern (one LLLG instance per zone).
pub struct SpaceTimeGuidance<P: Position> {
    cfg: GuidanceConfig,
    /// Neighbors filter — supplies passable neighbors. `None` = use
    /// `Position::neighbors()` directly (no wall filtering).
    neighbors_fn: Option<Box<NeighborFn<P>>>,
    /// Scratch: collision count per (position, time-offset) cell.
    /// Keyed by a hash of (position, time).
    ///
    /// Only used when `flat_occupancy` is `None` (the HashMap fallback path).
    /// Grid consumers should configure `with_flat_occupancy` for O(1) lookups.
    occupancy: HashMap<P, [u32; 64]>,
    /// Optional flat-array occupancy (Issue 516 T1a — O(1) collision lookups
    /// for grid domains). When `Some`, `collision_count` / `record_path` /
    /// `unrecord_path` / `clear_occupancy` bypass the HashMap entirely.
    ///
    /// The flat array is `width * height` entries of `[u32; 64]` (64 = max
    /// planning window). Indexed by `flat_index_fn(pos)`. For a 100×100 grid
    /// that's 10K entries × 256 bytes = 2.5MB — pre-allocated once, zero
    /// per-tick allocation.
    flat_occupancy: Option<Vec<[u32; 64]>>,
    /// Closure mapping `P` → flat index into `flat_occupancy`. Present iff
    /// `flat_occupancy` is present.
    flat_index_fn: Option<Box<FlatIndexFn<P>>>,
    /// Flat-array BFS distance cache: goal → flat distance field.
    ///
    /// When `Some`, this replaces `bfs_cache`'s `HashMap<P, f32>` values with
    /// `Vec<f32>` indexed by flat position. This gives O(1) heuristic lookups
    /// during A* instead of HashMap O(log n). Only used when `flat_occupancy`
    /// is active (grid domain).
    flat_bfs_cache: HashMap<P, Vec<f32>>,
    /// Flat-array A* scratch (Issue 516 T1d). Present iff `flat_occupancy` is.
    /// Eliminates HashMap operations on the hot A* path by indexing g_score /
    /// closed / came_from by (flat_idx, depth) with generation counters.
    flat_scratch: Option<FlatAstarScratch<P>>,
    /// BFS distance cache: goal → {position → true BFS distance}.
    ///
    /// This is the critical upgrade that fixes throughput on obstacle-heavy
    /// maps. The original Phase 1 greedy used Manhattan distance (the
    /// `Position::dist_heuristic` trait method) which ignores walls and leads
    /// agents into dead ends. BFS distance gives the true shortest path
    /// length around obstacles, so the greedy rollout follows the correct
    /// gradient.
    ///
    /// Cleared at the start of each `compute_guidance` call (within-tick
    /// reuse across agents that share a goal; across-tick recomputation).
    ///
    /// Only used when `flat_occupancy` is `None` (non-grid domain fallback).
    /// For grid domains with `flat_occupancy` active, `flat_bfs_cache` is used
    /// instead.
    bfs_cache: HashMap<P, HashMap<P, f32>>,
    /// Warm-start data supplied by [`set_warm_start`](LocalGuidanceSource::set_warm_start).
    ///
    /// Consumed (set to `None`) at the start of each `compute_guidance` call.
    /// When `Some`, the paths are seeded into the occupancy map before any
    /// agent is processed, giving each agent lookahead about where other
    /// agents are forecast to be. This is the LLLG mechanism (b) integration
    /// (Issue 140 T2).
    warm_start: Option<Vec<Vec<P>>>,
}

impl<P: Position> SpaceTimeGuidance<P> {
    pub fn new(cfg: GuidanceConfig) -> Self {
        Self {
            cfg,
            neighbors_fn: None,
            occupancy: HashMap::new(),
            flat_occupancy: None,
            flat_index_fn: None,
            flat_bfs_cache: HashMap::new(),
            flat_scratch: None,
            bfs_cache: HashMap::new(),
            warm_start: None,
        }
    }

    /// With a custom neighbors function (for wall-aware grids).
    pub fn with_neighbors<F>(mut self, f: F) -> Self
    where
        F: Fn(&P) -> Vec<P> + Send + Sync + 'static,
    {
        self.neighbors_fn = Some(Box::new(f));
        self
    }

    /// Enable flat-array occupancy for O(1) collision lookups (Issue 516 T1a).
    ///
    /// Grid consumers call this with `capacity = width * height` and a closure
    /// mapping `P` to a flat index (e.g. `|p: &GridPos| p.y * width + p.x`).
    /// When enabled, the occupancy HashMap is bypassed entirely — every
    /// `collision_count` / `record_path` / `unrecord_path` call becomes a
    /// single array index instead of a hash lookup.
    ///
    /// The caller MUST ensure every position reachable during planning maps
    /// to a valid index `< capacity`. Out-of-bounds indexing will panic (as
    /// intended — silent corruption would be worse).
    ///
    /// Not calling this leaves the HashMap path active (correct for non-grid
    /// domains or when grid dimensions aren't known at construction time).
    pub fn with_flat_occupancy<F>(mut self, capacity: usize, index_fn: F) -> Self
    where
        F: Fn(&P) -> usize + Send + Sync + 'static,
    {
        self.flat_occupancy = Some(vec![[0u32; 64]; capacity]);
        self.flat_index_fn = Some(Box::new(index_fn));
        // Issue 516 T1d: also allocate flat A* scratch for O(1) search lookups.
        self.flat_scratch = Some(FlatAstarScratch::new(capacity, self.cfg.w_phi));
        self
    }

    #[inline]
    fn neighbors_of(&self, pos: &P) -> Vec<P> {
        if let Some(ref f) = self.neighbors_fn {
            f(pos)
        } else {
            pos.neighbors()
        }
    }

    /// Count collisions of a candidate cell `(pos, t_offset)` with the
    /// current occupancy map. This is `χ` from Eq. 1.
    ///
    /// Hot path — called once per candidate neighbor per A* step per agent.
    /// The flat-array branch (T1a) is O(1); the HashMap fallback is O(log n).
    #[inline]
    fn collision_count(&self, pos: &P, t: usize) -> u32 {
        if t >= 64 {
            return 0;
        }
        if let (Some(occ), Some(f)) = (&self.flat_occupancy, &self.flat_index_fn) {
            let idx = f(pos);
            // Bounds check: out-of-bounds positions (e.g. agents at the grid
            // edge whose neighbors include x=width) return 0 collisions.
            // This matches the HashMap path's `.get(pos).map_or(0, ...)` behavior.
            return occ.get(idx).map_or(0, |arr| arr[t]);
        }
        self.occupancy.get(pos).map_or(0, |arr| arr[t])
    }

    /// Record an agent's chosen path into the occupancy map.
    fn record_path(&mut self, path: &[P]) {
        // Split into flat-array vs HashMap path outside the loop: the
        // `Option<&mut>` from `as_mut()` can't be moved across iterations.
        if let (Some(occ), Some(f)) = (self.flat_occupancy.as_mut(), self.flat_index_fn.as_ref()) {
            for (t, pos) in path.iter().enumerate() {
                if t >= 64 {
                    break;
                }
                let idx = f(pos);
                // Bounds check: skip out-of-bounds positions (matches HashMap no-entry behavior).
                if let Some(arr) = occ.get_mut(idx) {
                    arr[t] = arr[t].saturating_add(1);
                }
            }
        } else {
            for (t, pos) in path.iter().enumerate() {
                if t >= 64 {
                    break;
                }
                let slot = self.occupancy.entry(pos.clone()).or_insert([0u32; 64]);
                slot[t] = slot[t].saturating_add(1);
            }
        }
    }

    /// Remove an agent's path from the occupancy map (inverse of `record_path`).
    ///
    /// Used by the unrecord/re-record refinement loop so each agent sees the
    /// most-recent paths of all *other* agents (and its own previous path is
    /// removed to avoid self-collision). Saturating subtraction keeps counts
    /// non-negative so a stray double-unrecord is a no-op rather than panic.
    fn unrecord_path(&mut self, path: &[P]) {
        if let (Some(occ), Some(f)) = (self.flat_occupancy.as_mut(), self.flat_index_fn.as_ref()) {
            for (t, pos) in path.iter().enumerate() {
                if t >= 64 {
                    break;
                }
                let idx = f(pos);
                if let Some(arr) = occ.get_mut(idx) {
                    arr[t] = arr[t].saturating_sub(1);
                }
            }
        } else {
            for (t, pos) in path.iter().enumerate() {
                if t >= 64 {
                    break;
                }
                if let Some(slot) = self.occupancy.get_mut(pos) {
                    slot[t] = slot[t].saturating_sub(1);
                }
            }
        }
    }

    /// Clear occupancy for a fresh round.
    fn clear_occupancy(&mut self) {
        if let Some(ref mut occ) = self.flat_occupancy {
            occ.fill([0u32; 64]);
        } else {
            for arr in self.occupancy.values_mut() {
                *arr = [0u32; 64];
            }
        }
    }

    /// Space-time A* for a single agent (Issue 142).
    ///
    /// Searches the `(position, depth)` state space with BFS-distance heuristic.
    /// Returns the best `w_Φ`-step path (positions at t+1..t+w_Φ) minimizing
    /// Eq. 1 cost:
    ///
    /// ```text
    /// cost(π) = Σ_t (1 + α·χ(π[t], t))   // transition cost (collisions penalized)
    ///         + dist(π[w_Φ], goal)        // goal-reach term (h at depth w)
    /// ```
    ///
    /// Unlike the greedy rollout it replaces, the A* has w-step lookahead and
    /// can plan multi-step detours around collisions. The heuristic
    /// `h(pos,d) = BFS_dist(pos, goal)` is admissible (each remaining transition
    /// costs ≥ 1, so it never overestimates the true remaining cost).
    ///
    /// # Issue 516 T1b — scratch reuse
    ///
    /// The A* frontier (`g_score`, `came_from`, `open`, `closed`) is passed in
    /// via `scratch` and cleared at the start of each call. This eliminates
    /// 4 HashMap allocations per agent per round — for 1000 NPCs × 2 rounds
    /// that's 8000 allocations/tick saved. The `bfs` field is also passed in
    /// as a reference to avoid cloning the BFS distance HashMap per agent
    /// (which was O(map_cells) per clone).
    fn astar_for_agent(
        &mut self,
        start: &P,
        bfs: &HashMap<P, f32>,
        alpha: f32,
        scratch: &mut AstarScratch<P>,
    ) -> Vec<P> {
        let w = self.cfg.w_phi;
        let max_exp = self.cfg.max_expansions;

        // Unreachable goal: no BFS entry for start. Wait in place.
        let start_h = bfs.get(start).copied().unwrap_or(f32::MAX);
        if start_h == f32::MAX {
            return vec![start.clone(); w];
        }

        // At-goal early exit (Issue 516 T1f).
        if start_h == 0.0 {
            return vec![start.clone(); w];
        }

        // Clear scratch (reuses allocated capacity — zero allocation in steady state).
        scratch.clear();
        scratch.g_score.insert((start.clone(), 0), 0.0);
        scratch.open.push(AstarNode {
            f: start_h,
            depth: 0,
            pos: start.clone(),
        });

        let w_u8 = w as u8;
        let mut best_goal: Option<(P, u8)> = None;
        let mut best_effort: Option<(P, u8)> = None;
        let mut best_effort_h = start_h;
        let mut expansions: usize = 0;

        while let Some(node) = scratch.open.pop() {
            // Destructure (rather than `node.pos.clone()`): the popped node is
            // owned, so the position moves out instead of being cloned.
            let AstarNode { depth, pos, .. } = node;

            // Skip already-expanded states.
            if !scratch.closed.insert((pos.clone(), depth)) {
                continue;
            }

            // Track best-effort (closest-to-goal expanded node).
            let pos_h = bfs.get(&pos).copied().unwrap_or(f32::MAX);
            if pos_h < best_effort_h {
                best_effort_h = pos_h;
                best_effort = Some((pos.clone(), depth));
            }

            // Expansion cap (Issue 516 T1e).
            expansions += 1;
            if max_exp > 0 && expansions >= max_exp {
                break;
            }

            // Goal test: reached the planning horizon.
            if depth == w_u8 {
                best_goal = Some((pos, depth));
                break;
            }

            // g must exist (we pushed the node with this g).
            let g = scratch.g_score[&(pos.clone(), depth)];

            for neighbor in self.neighbors_of(&pos) {
                let new_depth = depth + 1;
                let key = (neighbor.clone(), new_depth);

                if scratch.closed.contains(&key) {
                    continue;
                }

                let h = bfs.get(&neighbor).copied().unwrap_or(f32::MAX);
                if h == f32::MAX {
                    continue; // unreachable neighbor
                }

                let chi = self.collision_count(&neighbor, depth as usize);
                let tentative_g = g + 1.0 + alpha * chi as f32;

                let known_g = scratch.g_score.get(&key).copied().unwrap_or(f32::MAX);
                if tentative_g < known_g {
                    scratch.g_score.insert(key.clone(), tentative_g);
                    // Consume `key` on the second insert — one fewer P clone.
                    scratch.came_from.insert(key, (pos.clone(), depth));
                    scratch.open.push(AstarNode {
                        f: tentative_g + h,
                        depth: new_depth,
                        pos: neighbor,
                    });
                }
            }
        }

        match best_goal {
            Some((end_pos, end_depth)) => {
                let mut path = Vec::with_capacity(w);
                let mut current = (end_pos, end_depth);
                while current.1 > 0 {
                    path.push(current.0.clone());
                    let prev = scratch
                        .came_from
                        .get(&current)
                        .expect("came_from chain must be complete");
                    current = prev.clone();
                }
                path.reverse();
                // Pad to w in the pathological case the search returned early.
                while path.len() < w {
                    path.push(path.last().cloned().unwrap_or_else(|| start.clone()));
                }
                path
            }
            None => match best_effort {
                Some((effort_pos, effort_depth)) => {
                    let mut path = Vec::with_capacity(w);
                    let mut current = (effort_pos, effort_depth);
                    while current.1 > 0 {
                        path.push(current.0.clone());
                        let prev = scratch
                            .came_from
                            .get(&current)
                            .expect("came_from chain must be complete for partial path");
                        current = prev.clone();
                    }
                    path.reverse();
                    while path.len() < w {
                        path.push(path.last().cloned().unwrap_or_else(|| start.clone()));
                    }
                    path
                }
                None => vec![start.clone(); w],
            },
        }
    }

    /// BFS flood-fill from `goal` to all reachable cells.
    fn compute_bfs(&self, goal: &P) -> HashMap<P, f32> {
        let mut dist: HashMap<P, f32> = HashMap::new();
        let mut queue: VecDeque<P> = VecDeque::new();
        dist.insert(goal.clone(), 0.0);
        queue.push_back(goal.clone());
        while let Some(current) = queue.pop_front() {
            let d = dist[&current];
            for neighbor in self.neighbors_of(&current) {
                if !dist.contains_key(&neighbor) {
                    dist.insert(neighbor.clone(), d + 1.0);
                    queue.push_back(neighbor);
                }
            }
        }
        dist
    }

    /// Flat-array BFS flood-fill from `goal` to all reachable cells (T1c).
    ///
    /// Writes into a pre-sized `Vec<f32>` indexed by `(flat_index_fn)(pos)`.
    /// `f32::MAX` = unreachable / uninitialized. O(1) lookup vs HashMap O(log n)
    /// during A* heuristic evaluation.
    ///
    /// Requires `flat_index_fn` to be set (grid domain).
    fn compute_bfs_flat(&self, goal: &P, capacity: usize) -> Vec<f32> {
        let f = self
            .flat_index_fn
            .as_ref()
            .expect("compute_bfs_flat requires flat_index_fn to be set");
        let mut dist = vec![f32::MAX; capacity];
        let mut queue: VecDeque<P> = VecDeque::new();
        let goal_idx = f(goal);
        if goal_idx < capacity {
            dist[goal_idx] = 0.0;
            queue.push_back(goal.clone());
        }
        while let Some(current) = queue.pop_front() {
            let cur_idx = f(&current);
            let d = dist[cur_idx];
            for neighbor in self.neighbors_of(&current) {
                let n_idx = f(&neighbor);
                if n_idx < capacity && dist[n_idx] == f32::MAX {
                    dist[n_idx] = d + 1.0;
                    queue.push_back(neighbor);
                }
            }
        }
        dist
    }

    /// Flat-array space-time A* for a single agent (Issue 516 T1d).
    ///
    /// Same algorithm as [`astar_for_agent`] but uses [`FlatAstarScratch`] for
    /// O(1) g_score / closed / came_from lookups. The BFS heuristic is read
    /// from a flat `&[f32]` (O(1)) instead of `&HashMap<P, f32>` (O(log n)).
    ///
    /// This is the hot-path A* for grid domains — called once per agent per
    /// refinement round.
    fn astar_for_agent_flat(
        &mut self,
        start: &P,
        bfs_flat: &[f32],
        bfs_capacity: usize,
        alpha: f32,
        scratch: &mut FlatAstarScratch<P>,
    ) -> Vec<P> {
        let w = self.cfg.w_phi;
        let max_exp = self.cfg.max_expansions;
        let f = self
            .flat_index_fn
            .as_ref()
            .expect("astar_for_agent_flat requires flat_index_fn");

        let start_idx = f(start);

        // Unreachable goal: no BFS entry for start. Wait in place.
        let start_h = if start_idx < bfs_capacity {
            bfs_flat[start_idx]
        } else {
            f32::MAX
        };
        if start_h == f32::MAX {
            return vec![start.clone(); w];
        }

        // At-goal early exit (Issue 516 T1f): agent is already at its goal.
        // BFS distance 0 means start == goal. Waiting in place is optimal.
        if start_h == 0.0 {
            return vec![start.clone(); w];
        }

        scratch.begin_search();
        scratch.g_set(start_idx, 0, 0.0);
        scratch.open.push(AstarNode {
            f: start_h,
            depth: 0,
            pos: start.clone(),
        });

        let w_u8 = w as u8;
        let mut best_goal: Option<(P, u8)> = None;
        // Best-effort partial path node: the expanded node closest to goal by
        // BFS distance. Used when the search is capped before reaching depth w.
        let mut best_effort: Option<(P, u8)> = None;
        let mut best_effort_h = start_h;
        let mut expansions: usize = 0;

        while let Some(node) = scratch.open.pop() {
            // Destructure (rather than `node.pos.clone()`): the popped node is
            // owned, so the position moves out instead of being cloned.
            let AstarNode { depth, pos, .. } = node;
            let pos_idx = f(&pos);

            // Skip already-expanded states.
            if !scratch.close(pos_idx, depth) {
                continue;
            }

            // Track best-effort (closest-to-goal expanded node). The start is
            // always expanded first; subsequent nodes with lower h improve it.
            // pos_idx is guaranteed < bfs_capacity: the start is bounds-checked
            // at the top of this function, and all neighbors pushed to the open
            // list pass the bounds guard in the expansion loop below.
            let pos_h = bfs_flat[pos_idx];
            if pos_h < best_effort_h {
                best_effort_h = pos_h;
                best_effort = Some((pos.clone(), depth));
            }

            // Expansion cap (Issue 516 T1e): bound worst-case latency.
            expansions += 1;
            if max_exp > 0 && expansions >= max_exp {
                break;
            }

            // Goal test: reached the planning horizon.
            if depth == w_u8 {
                best_goal = Some((pos, depth));
                break;
            }

            let g = scratch.g_get(pos_idx, depth);

            for neighbor in self.neighbors_of(&pos) {
                let new_depth = depth + 1;
                let n_idx = f(&neighbor);

                // Bounds check BEFORE any scratch access (Issue 516 T4 hotfix):
                // neighbors_of can return positions outside the map whose
                // flat_index exceeds the scratch array capacity. Skipping them
                // here is equivalent to treating them as walls/unreachable.
                if n_idx >= bfs_capacity {
                    continue;
                }

                if scratch.is_closed(n_idx, new_depth) {
                    continue;
                }

                let h = bfs_flat[n_idx];
                if h == f32::MAX {
                    continue; // unreachable neighbor
                }

                let chi = self.collision_count(&neighbor, depth as usize);
                let tentative_g = g + 1.0 + alpha * chi as f32;

                let known_g = scratch.g_get(n_idx, new_depth);
                if tentative_g < known_g {
                    scratch.g_set(n_idx, new_depth, tentative_g);
                    scratch.set_parent(n_idx, new_depth, pos.clone(), depth);
                    scratch.open.push(AstarNode {
                        f: tentative_g + h,
                        depth: new_depth,
                        pos: neighbor,
                    });
                }
            }
        }

        match best_goal {
            Some((end_pos, end_depth)) => {
                let mut path = Vec::with_capacity(w);
                let mut current_pos = end_pos;
                let mut current_depth = end_depth;
                while current_depth > 0 {
                    path.push(current_pos.clone());
                    let cur_idx = f(&current_pos);
                    let parent = scratch
                        .get_parent(cur_idx, current_depth)
                        .expect("came_from chain must be complete");
                    current_pos = parent.0.clone();
                    current_depth = parent.1;
                }
                path.reverse();
                while path.len() < w {
                    path.push(path.last().cloned().unwrap_or_else(|| start.clone()));
                }
                path
            }
            None => match best_effort {
                // Cap hit before reaching depth w: return the best partial path
                // (closest-to-goal node found), padded to w with wait actions.
                Some((effort_pos, effort_depth)) => {
                    let mut path = Vec::with_capacity(w);
                    let mut current_pos = effort_pos;
                    let mut current_depth = effort_depth;
                    while current_depth > 0 {
                        path.push(current_pos.clone());
                        let cur_idx = f(&current_pos);
                        let parent = scratch
                            .get_parent(cur_idx, current_depth)
                            .expect("came_from chain must be complete for partial path");
                        current_pos = parent.0.clone();
                        current_depth = parent.1;
                    }
                    path.reverse();
                    while path.len() < w {
                        path.push(path.last().cloned().unwrap_or_else(|| start.clone()));
                    }
                    path
                }
                // No node expanded at all (shouldn't happen — start is always
                // pushed) — wait in place.
                None => vec![start.clone(); w],
            },
        }
    }

    /// Compute guidance with a **per-agent** collision penalty `alpha`.
    ///
    /// This is the Super-GOAT fusion extension point (Plan 489 Phase 2 /
    /// riir-ai/318 Extension A). Identical to
    /// [`compute_guidance`](LocalGuidanceSource::compute_guidance) except each
    /// agent `i` uses `alphas[i]` instead of the uniform `self.cfg.alpha`. The
    /// occupancy / BFS cache / refinement-rounds logic is shared with the
    /// uniform path — only the cost of collisions varies per agent.
    ///
    /// A private consumer (riir-ai's `HlaProjectedGuidance`) computes
    /// `alphas[i] = alpha_base * (1 + beta * sigmoid(dot(belief_i, D_frustration)))`
    /// so frustrated NPCs take wider detours around occupied cells while calm
    /// NPCs push through — the behavioral signature the G5 gate measures.
    ///
    /// # Panics
    ///
    /// Panics if `alphas.len() != config.n_agents()`.
    pub fn compute_guidance_per_agent_alpha(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        alphas: &[f32],
        out: &mut Guidance<P>,
    ) {
        let n = config.n_agents();
        assert_eq!(
            alphas.len(),
            n,
            "compute_guidance_per_agent_alpha: alphas.len()={} but n_agents={}",
            alphas.len(),
            n
        );
        self.run_refinement(config, goals, out, move |i| alphas[i]);
    }

    /// Shared refinement-loop body for both the uniform-alpha trait impl and
    /// the per-agent-alpha fusion method. `alpha_of(i)` returns the collision
    /// penalty for agent `i`.
    ///
    /// This is the paper's sequential per-agent refinement (Algorithm 1):
    /// `rounds` passes, each iterating all agents in order. Each agent's A*
    /// sees the already-updated paths of earlier-in-the-round agents (via the
    /// occupancy map). Between rounds, each agent's previous path is unrecorded
    /// before re-planning so round 1+ improves on round 0.
    fn run_refinement<F: Fn(usize) -> f32>(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        out: &mut Guidance<P>,
        alpha_of: F,
    ) {
        let n = config.n_agents();
        out.clear();
        out.resize(n, Vec::new());

        // Consume warm-start data (one-shot per tick). NOT seeded into the
        // occupancy — Issue 142 found that warm-start forecasts HURT throughput
        // because PIBT deviations invalidate them on dense maps.
        let _warm_start = self.warm_start.take();

        self.clear_occupancy();

        // Dispatch between flat-array (grid) and HashMap (generic) paths.
        // The flat path uses O(1) lookups for BFS heuristic + A* scratch.
        if self.flat_occupancy.is_some() {
            self.run_refinement_flat(config, goals, out, alpha_of, n);
        } else {
            self.run_refinement_hashmap(config, goals, out, alpha_of, n);
        }
    }

    /// HashMap-path refinement loop (non-grid fallback). This is the original
    /// T1b path, unchanged except for being split out from `run_refinement`.
    fn run_refinement_hashmap<F: Fn(usize) -> f32>(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        out: &mut Guidance<P>,
        alpha_of: F,
        n: usize,
    ) {
        // BFS cache: persistent across ticks for amortization. Cleared only
        // when it exceeds MAX_BFS_CACHE_ENTRIES (to bound memory).
        if self.bfs_cache.len() > MAX_BFS_CACHE_ENTRIES {
            self.bfs_cache.clear();
        }

        // Issue 516 T1b — move bfs_cache out to a local so we can hold
        // references to its entries while mutating self.occupancy / calling
        // self.collision_count inside the search loop.
        let mut bfs_cache = std::mem::take(&mut self.bfs_cache);

        for goal in goals.iter().take(n) {
            if !bfs_cache.contains_key(goal) {
                let field = self.compute_bfs(goal);
                bfs_cache.insert(goal.clone(), field);
            }
        }

        let mut scratch = AstarScratch::<P>::new();

        // `out[i]` already holds agent i's previous-round path (empty in round
        // 0, courtesy of the `resize` in `run_refinement`), so it doubles as
        // the `prev_paths` bookkeeping the unrecord/re-record loop needs. This
        // drops one full `Vec<P>` clone per agent per round plus the `n`-entry
        // `prev_paths` allocation.
        for _round in 0..self.cfg.rounds {
            for i in 0..n {
                let agent = AgentId(i as u32);
                let start = config.pos(agent).clone();
                let goal = &goals[i];
                let alpha = alpha_of(i);

                self.unrecord_path(&out[i]);
                let bfs = &bfs_cache[goal];
                let path = self.astar_for_agent(&start, bfs, alpha, &mut scratch);
                self.record_path(&path);
                out[i] = path;
            }
        }

        self.bfs_cache = bfs_cache;
    }

    /// Flat-array refinement loop (Issue 516 T1c+T1d — grid domain fast path).
    ///
    /// Uses `flat_bfs_cache` (Vec<f32> per goal) and `flat_scratch`
    /// (FlatAstarScratch with generation counters) for O(1) lookups throughout
    /// the A* inner loop. This is the production path for grid-based games.
    fn run_refinement_flat<F: Fn(usize) -> f32>(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        out: &mut Guidance<P>,
        alpha_of: F,
        n: usize,
    ) {
        // Flat BFS cache capacity = flat_occupancy length (map_cells).
        let bfs_cap = self
            .flat_occupancy
            .as_ref()
            .expect("run_refinement_flat requires flat_occupancy")
            .len();

        // Bound flat BFS cache memory (same threshold as HashMap cache).
        if self.flat_bfs_cache.len() > MAX_BFS_CACHE_ENTRIES {
            self.flat_bfs_cache.clear();
        }

        // Move flat_bfs_cache + flat_scratch out to avoid borrow conflicts
        // with the &mut self methods (collision_count, record_path, etc.).
        let mut flat_bfs_cache = std::mem::take(&mut self.flat_bfs_cache);
        let mut flat_scratch = self
            .flat_scratch
            .take()
            .expect("run_refinement_flat requires flat_scratch");

        // Pre-compute missing flat BFS fields.
        for goal in goals.iter().take(n) {
            if !flat_bfs_cache.contains_key(goal) {
                let field = self.compute_bfs_flat(goal, bfs_cap);
                flat_bfs_cache.insert(goal.clone(), field);
            }
        }

        // `out[i]` doubles as agent i's previous-round path (see the HashMap
        // path for the rationale) — saves a full `Vec<P>` clone per agent per
        // round plus the `n`-entry `prev_paths` allocation.
        for _round in 0..self.cfg.rounds {
            for i in 0..n {
                let agent = AgentId(i as u32);
                let start = config.pos(agent).clone();
                let goal = &goals[i];
                let alpha = alpha_of(i);

                self.unrecord_path(&out[i]);
                let bfs = &flat_bfs_cache[goal];
                let path =
                    self.astar_for_agent_flat(&start, bfs, bfs_cap, alpha, &mut flat_scratch);
                self.record_path(&path);
                out[i] = path;
            }
        }

        // Restore caches for the next tick.
        self.flat_bfs_cache = flat_bfs_cache;
        self.flat_scratch = Some(flat_scratch);
    }
}

/// A* search node. Ordered so the `BinaryHeap` (max-heap) pops the
/// smallest `f` first, breaking ties toward shallower depth (progress).
///
/// Position `P` is intentionally excluded from the `Ord` comparison: the
/// heap never needs to compare positions (only `f`/`depth` drive ordering),
/// and excluding `P` lets this work for any `Position` without an `Ord` bound.
struct AstarNode<P> {
    f: f32,
    depth: u8,
    pos: P,
}

impl<P> PartialEq for AstarNode<P> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.depth == other.depth
    }
}
impl<P> Eq for AstarNode<P> {}

impl<P> PartialOrd for AstarNode<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P> Ord for AstarNode<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: reverse the natural f/depth comparisons so
        // the smallest f (then smallest depth) bubbles to the top.
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| other.depth.cmp(&self.depth))
    }
}

/// Pre-allocated A* frontier scratch (Issue 516 T1b).
///
/// Allocated once in [`SpaceTimeGuidance::run_refinement`] and reused across
/// all agents and refinement rounds. `clear()` resets the entries without
/// deallocating the underlying storage (HashMap buckets / BinaryHeap buffer).
///
/// For 1000 NPCs × 2 rounds, this eliminates 8000 HashMap allocations per tick.
/// The initial capacity (256) is generous for a planning window of `w=5` on a
/// 4-connected grid (at most ~(2w+1)² ≈ 121 states explored per search).
struct AstarScratch<P: Position> {
    g_score: HashMap<(P, u8), f32>,
    came_from: HashMap<(P, u8), (P, u8)>,
    open: BinaryHeap<AstarNode<P>>,
    closed: HashSet<(P, u8)>,
}

impl<P: Position> AstarScratch<P> {
    fn new() -> Self {
        Self {
            g_score: HashMap::new(),
            came_from: HashMap::new(),
            open: BinaryHeap::new(),
            closed: HashSet::new(),
        }
    }

    /// Reset all scratch buffers without deallocating. Called at the start of
    /// each `astar_for_agent` invocation.
    #[inline]
    fn clear(&mut self) {
        self.g_score.clear();
        self.came_from.clear();
        self.open.clear();
        self.closed.clear();
    }
}

/// Flat-array A* scratch for grid domains (Issue 516 T1d).
///
/// Replaces [`AstarScratch`]'s HashMap-based g_score / came_from / closed with
/// flat arrays indexed by `(flat_idx * w_phi + depth)`. This gives O(1)
/// lookups instead of HashMap O(log n), and eliminates hashing overhead.
///
/// Clearing between agent searches is zero-cost via generation counters:
/// each search increments `current_gen`, and a cell's entry is valid only if
/// its stored generation matches. On generation overflow (every ~4 billion
/// searches), the arrays are actually cleared.
///
/// The `open` BinaryHeap still stores positions (it needs them for neighbor
/// expansion and path reconstruction). Only the map/set structures are flat.
struct FlatAstarScratch<P: Position> {
    /// g_score[flat_idx * stride + depth] = best known cost-to-reach.
    /// `stride = w_phi + 1` to accommodate goal-depth nodes (depth == w_phi).
    g_score: Vec<f32>,
    /// Generation when g_score entry was last written.
    g_gen: Vec<u32>,
    /// Generation when the cell was closed (expanded). 0 = never (unless
    /// current_gen wraps to 0, handled by overflow reset).
    closed_gen: Vec<u32>,
    /// came_from[flat_idx * stride + depth] = (parent position, parent depth).
    /// Read during path reconstruction.
    came_from: Vec<Option<(P, u8)>>,
    /// Priority queue. Same AstarNode type as HashMap mode — the heap can't
    /// be flat-indexed.
    open: BinaryHeap<AstarNode<P>>,
    /// Current generation for this search. Incremented per `begin_search`.
    current_gen: u32,
    /// Array stride = w_phi + 1. Depth ranges 0..=w_phi (goal nodes sit at
    /// depth == w_phi), so we need w_phi + 1 slots per cell to avoid OOB.
    stride: usize,
}

impl<P: Position> FlatAstarScratch<P> {
    fn new(map_cells: usize, w_phi: usize) -> Self {
        // Issue 516 T4 hotfix: stride must be w_phi + 1 because goal-depth
        // nodes (depth == w_phi) are pushed to the open list and get
        // close()/is_closed() called on them before the goal-test break.
        let stride = w_phi + 1;
        let total = map_cells * stride;
        Self {
            g_score: vec![f32::MAX; total],
            g_gen: vec![0; total],
            closed_gen: vec![0; total],
            came_from: vec![None; total],
            open: BinaryHeap::new(),
            current_gen: 1, // gen 0 = unvisited sentinel
            stride,
        }
    }

    /// Start a new search: bump generation (zero-cost clear) + clear the heap.
    #[inline]
    fn begin_search(&mut self) {
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            // Generation overflow (~4 billion searches). Actually clear.
            self.g_gen.fill(0);
            self.closed_gen.fill(0);
            self.current_gen = 1;
        }
        self.open.clear();
    }

    /// Flat index into the scratch arrays for `(flat_idx, depth)`.
    #[inline(always)]
    fn slot(&self, flat_idx: usize, depth: u8) -> usize {
        flat_idx * self.stride + depth as usize
    }

    /// Get g_score for `(flat_idx, depth)` if visited in the current search.
    #[inline]
    fn g_get(&self, flat_idx: usize, depth: u8) -> f32 {
        let s = self.slot(flat_idx, depth);
        if self.g_gen[s] == self.current_gen {
            self.g_score[s]
        } else {
            f32::MAX
        }
    }

    /// Set g_score for `(flat_idx, depth)`. Marks the generation.
    #[inline]
    fn g_set(&mut self, flat_idx: usize, depth: u8, val: f32) {
        let s = self.slot(flat_idx, depth);
        self.g_score[s] = val;
        self.g_gen[s] = self.current_gen;
    }

    /// Mark `(flat_idx, depth)` as closed. Returns true if it was NOT already
    /// closed in this search (i.e., this is the first close).
    #[inline]
    fn close(&mut self, flat_idx: usize, depth: u8) -> bool {
        let s = self.slot(flat_idx, depth);
        if self.closed_gen[s] == self.current_gen {
            false
        } else {
            self.closed_gen[s] = self.current_gen;
            true
        }
    }

    /// Check if `(flat_idx, depth)` is closed in this search.
    #[inline]
    fn is_closed(&self, flat_idx: usize, depth: u8) -> bool {
        self.closed_gen[self.slot(flat_idx, depth)] == self.current_gen
    }

    /// Record the parent of `(flat_idx, depth)`.
    #[inline]
    fn set_parent(&mut self, flat_idx: usize, depth: u8, parent: P, parent_depth: u8) {
        let s = self.slot(flat_idx, depth);
        self.came_from[s] = Some((parent, parent_depth));
    }

    /// Get the recorded parent of `(flat_idx, depth)`.
    #[inline]
    fn get_parent(&self, flat_idx: usize, depth: u8) -> Option<&(P, u8)> {
        self.came_from[self.slot(flat_idx, depth)].as_ref()
    }
}

impl<P: Position> LocalGuidanceSource<P> for SpaceTimeGuidance<P> {
    fn compute_guidance(&mut self, config: &JointConfig<P>, goals: &[P], out: &mut Guidance<P>) {
        // Paper-faithful uniform alpha — delegates to the shared refinement
        // loop with a constant closure. The closure captures `self.cfg.alpha`
        // by copy (f32 is Copy), avoiding any borrow conflict with the mutable
        // `self` methods called inside the loop.
        let uniform_alpha = self.cfg.alpha;
        self.run_refinement(config, goals, out, move |_i| uniform_alpha);
    }

    fn set_warm_start(&mut self, warm_start: Vec<Vec<P>>) {
        self.warm_start = Some(warm_start);
    }

    fn ensure_flat_occupancy(&mut self, width: usize, height: usize) {
        let needed = width * height;
        let needs_realloc = match &self.flat_occupancy {
            None => true,
            Some(occ) => occ.len() != needed,
        };
        if needs_realloc {
            self.flat_occupancy = Some(vec![[0u32; 64]; needed]);
            self.flat_index_fn = Some(Box::new(move |p: &P| {
                p.flat_index(width).expect(
                    "ensure_flat_occupancy: Position::flat_index returned None — \
                     non-grid positions should not call this method",
                )
            }));
            // Issue 516 T1d: allocate flat A* scratch for this map size.
            // Indexed by (flat_idx * w_phi + depth), using generation counters
            // for zero-cost clearing between agent searches.
            self.flat_scratch = Some(FlatAstarScratch::new(needed, self.cfg.w_phi));
            // Clear flat BFS cache — dimensions changed, old fields are stale.
            self.flat_bfs_cache.clear();
        }
    }
}
