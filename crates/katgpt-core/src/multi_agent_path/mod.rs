//! Lifelong LaCAM with Local Guidance (LLLG) — modelless multi-agent
//! pathfinding substrate (Plan 440, Research 424, arXiv:2605.16855).
//!
//! Distilled from Arita & Okumura (AIST, AAAI 2026). A purely heuristic,
//! training-free, receding-horizon multi-agent pathfinder that scales to
//! **10,000 agents at <1s/step** with higher throughput than RHCR in dense
//! settings.
//!
//! # The five pluggable seams (Super-GOAT hooks)
//!
//! This substrate is generic over five mechanisms so a private consumer
//! (riir-ai/318) can fuse it with HLA, Crowd MCGS, and the warm-path stack
//! without forking:
//!
//! | Seam | Trait/enum | Default (paper) | Pluggable alternative |
//! |---|---|---|---|
//! | Cost function | [`CostFn<P>`] | Uniform (1/move) | Heightfield slope, threat cochain, faction zone penalty |
//! | Guidance source | [`LocalGuidanceSource<P>`] | Space-time A* on collision count | HLA-projected guidance (per-NPC emotional congestion avoidance) |
//! | Warm-start scheme | [`WarmStartScheme`] | `LllgPi` (prev solution suffix) | Personality-weighted blend |
//! | Hindrance estimator | [`HindranceEstimator<P>`] | Raw blocking count | Affect-aware blocking (fearful NPCs count more) |
//! | Flow field (Issue 149 + 150) | [`FlowField<P>`] | [`NoFlow`] (no corridor enforcement) | [`GridFlowField`] (1-wide + 2-wide corridor direction assignment) |
//!
//! A consumer that uses all five defaults gets the paper's LLLG verbatim.
//!
//! # Modelless mandate
//!
//! Entirely heuristic — no training, no backprop, no gradient descent. The
//! only weight mutations are freeze/thaw (swapping frozen snapshots), which
//! is not used here at all. Promotion to default-on is allowed once G1–G4
//! pass (the substrate is modelless).
//!
//! # Latent vs raw boundary
//!
//! Per AGENTS.md sync-boundary rule:
//! - **Raw (synced):** joint configuration `Q_t`, executed joint action `Π_t[1]`.
//! - **Latent (local):** guidance field `Φ`, hindrance scalars, warm-start cache.
//! - **Bridge:** `Φ → Π_t[1]` — latent guidance selects the raw action.
//!
//! See Research 424 §2.5 for the full table.
//!
//! # Example
//!
//! ```no_run
//! use katgpt_core::multi_agent_path::*;
//! use katgpt_core::multi_agent_path::position::*;
//!
//! // 10×10 grid, 2 agents.
//! let map = GridMap::empty(10, 10);
//! let starts = vec![GridPos::new(0, 0), GridPos::new(9, 9)];
//! let config = JointConfig::new(starts);
//! let goals = vec![GridPos::new(9, 9), GridPos::new(0, 0)];
//!
//! let guidance_cfg = GuidanceConfig::default();
//! // `with_neighbors` requires `'static`, so leak the map into a reference
//! // (a consumer would normally store the map in a long-lived struct field).
//! let map_ref: &'static GridMap = Box::leak(Box::new(map));
//! let mut guidance = SpaceTimeGuidance::new(guidance_cfg)
//!     .with_neighbors(move |p| map_ref.passable_neighbors(p));
//! let mut hindrance = BlockingCount::new();
//! let mut warm_start = WarmStartCache::new(WarmStartScheme::default(), guidance_cfg.w_phi);
//! let mut rng = fastrand::Rng::with_seed(42);
//!
//! let mut lacam = LifelongLaCam::new(warm_start);
//! let action = lacam.tick(&config, &goals, &mut guidance, &mut hindrance, &mut rng);
//! ```

#![allow(clippy::too_many_arguments)]

pub mod config;
pub mod flow;
pub mod hindrance;
#[cfg(feature = "lacam_escalation")]
pub(crate) mod lacam;
pub mod local_guidance;
pub mod pibt;
pub mod position;
pub mod warm_start;

#[cfg(test)]
mod tests;

pub use config::{AgentId, GoalAssignment, JointAction, JointConfig, UniformGoals};
pub use flow::{CorridorAxis, FlowDirection, FlowField, GridFlowField, NoFlow};
pub use hindrance::{
    BlockingCount, CounterFlowHindrance, HindranceEstimator, WeightedBlockingCount,
};
pub use local_guidance::{Guidance, GuidanceConfig, LocalGuidanceSource, SpaceTimeGuidance};
use pibt::pibt_step_with_budget;
pub use pibt::{Deadlock, NeighborFn, pibt_step};
pub use position::{GridMap, GridPos, Position, soft_cost};
pub use warm_start::{WarmStartCache, WarmStartScheme};

#[cfg(feature = "lacam_escalation")]
pub use lacam::{EscalationBudget, lacam_escalation_step};

// ─────────────────────────────────────────────────────────────────────
// CostFn trait — pluggable seam #1 (Plan 440 T1.2)
// ─────────────────────────────────────────────────────────────────────

/// Transition cost function — pluggable seam #1.
///
/// Returns the raw cost of transitioning from `from` to `to` in one step.
/// The default [`UniformCost`] returns 1.0 for any move (paper default).
///
/// # Extension points (private consumer, riir-ai/318)
///
/// A consumer's impl may incorporate:
/// - **Heightfield slope** — `cost = 1 + sigmoid(slope · β)` so uphill moves
///   cost more (the raw→latent bridge via [`soft_cost`]).
/// - **Threat cochain** — read the DEC codifferential of the threat field at
///   `to` and add it to the base cost.
/// - **Faction zone penalty** — higher cost in enemy territory.
/// - **Economy toll** — path through a toll gate costs gold.
///
/// All of these are modelless (closed-form, no training).
///
/// # Examples
///
/// ## Heightfield slope cost
///
/// Uphill moves cost more; downhill moves cost less. Uses the canonical
/// [`soft_cost`] raw→latent bridge (sigmoid-gated).
///
/// ```no_run
/// use katgpt_core::multi_agent_path::*;
/// use katgpt_core::multi_agent_path::position::*;
///
/// /// Heightfield-aware transition cost. `slope_at(pos)` returns the raw
/// /// terrain gradient magnitude at `pos` (caller-supplied).
/// struct HeightfieldCost<F: Fn(&GridPos) -> f32> {
///     slope_at: F,
///     beta: f32,
/// }
///
/// impl<F: Fn(&GridPos) -> f32> CostFn<GridPos> for HeightfieldCost<F> {
///     fn cost(&self, from: &GridPos, to: &GridPos) -> f32 {
///         // Uphill (slope > 0) costs more; downhill (slope < 0) costs less.
///         // `soft_cost` returns `sigmoid(slope * beta)` ∈ (0, 1).
///         1.0 + soft_cost((self.slope_at)(to), self.beta)
///     }
/// }
/// ```
///
/// ## Faction zone penalty
///
/// Moves into enemy territory cost more — the penalty is a closed-form
/// scalar lookup, not a learned value.
///
/// ```no_run
/// use katgpt_core::multi_agent_path::*;
/// use katgpt_core::multi_agent_path::position::*;
/// use std::collections::HashMap;
///
/// /// Per-cell faction ownership penalty. `penalty[pos]` is 0.0 in friendly
/// /// territory, higher in enemy territory (caller populates from the
/// /// social-domain KG triples).
/// struct FactionZoneCost {
///     penalty: HashMap<GridPos, f32>,
/// }
///
/// impl CostFn<GridPos> for FactionZoneCost {
///     fn cost(&self, _from: &GridPos, to: &GridPos) -> f32 {
///         1.0 + self.penalty.get(to).copied().unwrap_or(0.0)
///     }
/// }
/// ```
///
/// ## Threat cochain cost (DEC fusion)
///
/// A consumer that ships the DEC substrate (Plan 219) reads the codifferential
/// `δ` of the threat cochain at `to` and adds it as a latent cost. The stub
/// below shows the shape; the actual `δ` computation lives in
/// `katgpt_core::dec::codifferential`.
///
/// ```no_run
/// use katgpt_core::multi_agent_path::*;
/// use katgpt_core::multi_agent_path::position::*;
///
/// /// Threat-field-aware cost. `threat_density(pos)` is the magnitude of the
/// /// DEC codifferential of the threat cochain at `pos` — a latent scalar.
/// struct ThreatCochainCost<F: Fn(&GridPos) -> f32> {
///     threat_density: F,
/// }
///
/// impl<F: Fn(&GridPos) -> f32> CostFn<GridPos> for ThreatCochainCost<F> {
///     fn cost(&self, _from: &GridPos, to: &GridPos) -> f32 {
///         // High threat density → high cost. Clamped at a sensible ceiling.
///         1.0 + (self.threat_density)(to).min(10.0)
///     }
/// }
/// ```
pub trait CostFn<P: Position> {
    /// Cost of moving from `from` to `to`. Must be ≥ 0.
    fn cost(&self, from: &P, to: &P) -> f32;
}

/// Paper-default cost: 1.0 per move (uniform).
pub struct UniformCost;

impl Default for UniformCost {
    fn default() -> Self {
        Self
    }
}

impl<P: Position> CostFn<P> for UniformCost {
    #[inline]
    fn cost(&self, _from: &P, _to: &P) -> f32 {
        1.0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Orchestrator (Plan 440 T1.2)
// ─────────────────────────────────────────────────────────────────────

/// Type alias for the neighbor-supplying closure stored in the orchestrator.
type NeighborClosure<P> = Box<dyn Fn(&P) -> Vec<P> + Send + Sync>;

/// Type alias for the flow field stored in the orchestrator (Issue 149).
type FlowFieldBox<P> = Box<dyn FlowField<P> + Send + Sync>;

/// The LLLG orchestrator: one tick of receding-horizon windowed planning.
///
/// Generic over the position type `P`. Holds the warm-start cache and the
/// guidance config. The guidance source, hindrance estimator, and RNG are
/// passed by `&mut` to [`tick`](Self::tick) so they can be reused across
/// ticks without cloning.
///
/// # Lifecycle
///
/// 1. Construct once per zone (or per crowd) with the desired config.
/// 2. Call [`tick`](Self::tick) each game tick with the current config + goals.
/// 3. The returned [`JointAction`] is the first step of the windowed plan
///    (edge-collision-free; vertex collisions may occur on congested maps —
///    see [`JointAction`] and `pibt.rs` module docs).
pub struct LifelongLaCam<P: Position> {
    warm_start: WarmStartCache<P>,
    /// Scratch: per-agent guidance field `Φ`.
    guidance_scratch: Guidance<P>,
    /// Scratch: priority weights (uniform by default).
    priorities: Vec<f32>,
    /// Wall-aware neighbor function. When `None`, PIBT uses `Position::neighbors()`
    /// directly (no wall/bounds checking). Consumers with walls or bounded maps
    /// MUST set this via [`with_neighbors`](Self::with_neighbors).
    neighbors_fn: Option<NeighborClosure<P>>,
    /// Flow field for Guided-PIBT direction assignment (Issue 149).
    /// When `None`, uses [`NoFlow`] (paper-faithful — no corridor enforcement).
    /// Set via [`with_flow_field`](Self::with_flow_field) for maze maps.
    flow_field: Option<FlowFieldBox<P>>,
    /// LaCAM escalation budget (Issue 546 multi-step extension).
    ///
    /// Only consulted when the `lacam_escalation` feature is ON. Defaults to
    /// [`EscalationBudget::default`] (Plan 453 one-step behavior). Set to
    /// [`EscalationBudget::multistep_default`] via
    /// [`with_escalation_budget`](Self::with_escalation_budget) for maze maps
    /// where stuck-agent targeting + higher depth is needed.
    #[cfg(feature = "lacam_escalation")]
    escalation_budget: EscalationBudget,
}

impl<P: Position> LifelongLaCam<P> {
    /// Construct with the given warm-start cache.
    ///
    /// The guidance config is owned by the [`LocalGuidanceSource`] you pass to
    /// [`tick`](Self::tick); the orchestrator does not hold a separate copy.
    /// Use [`with_neighbors`](Self::with_neighbors) to set a wall-aware neighbor
    /// function if your map has walls or bounds.
    pub fn new(warm_start: WarmStartCache<P>) -> Self {
        Self {
            warm_start,
            guidance_scratch: Vec::new(),
            priorities: Vec::new(),
            neighbors_fn: None,
            flow_field: None,
            #[cfg(feature = "lacam_escalation")]
            escalation_budget: EscalationBudget::default(),
        }
    }

    /// Set a wall-aware neighbor function.
    ///
    /// When set, PIBT uses this instead of `Position::neighbors()` to generate
    /// candidate moves, ensuring agents never move through walls or out of
    /// bounds. The guidance source should be configured independently with the
    /// same wall-aware function.
    pub fn with_neighbors<F>(mut self, f: F) -> Self
    where
        F: Fn(&P) -> Vec<P> + Send + Sync + 'static,
    {
        self.neighbors_fn = Some(Box::new(f));
        self
    }

    /// Set per-agent priorities (higher = processed first by PIBT).
    ///
    /// Empty = uniform (index order). Length must match `config.n_agents()`.
    pub fn set_priorities(&mut self, priorities: Vec<f32>) {
        self.priorities = priorities;
    }

    /// Set a flow field for Guided-PIBT direction assignment (Issue 149).
    ///
    /// When set, PIBT penalizes moves that go against the assigned corridor
    /// direction (the `flow_mismatch` cost term). On maze maps (ht_chantry),
    /// this creates one-way directional lanes that eliminate head-on corridor
    /// deadlocks. On open maps, the flow field should be empty (no corridors),
    /// so this has no effect.
    ///
    /// When not set, PIBT uses [`NoFlow`] (paper-faithful — no corridor
    /// direction enforcement).
    pub fn with_flow_field<F>(mut self, flow_field: F) -> Self
    where
        F: FlowField<P> + Send + Sync + 'static,
    {
        self.flow_field = Some(Box::new(flow_field));
        self
    }

    /// Set the LaCAM escalation budget (Issue 546 multi-step extension).
    ///
    /// Only consulted when the `lacam_escalation` feature is ON. The default
    /// budget ([`EscalationBudget::default`]) reproduces Plan 453 one-step
    /// behavior. Pass [`EscalationBudget::multistep_default`] to enable
    /// stuck-agent targeting + depth-8 search for maze-class maps.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use katgpt_core::multi_agent_path::*;
    /// # let warm = WarmStartCache::<GridPos>::new(WarmStartScheme::default(), 0);
    /// let lacam = LifelongLaCam::new(warm)
    ///     .with_escalation_budget(EscalationBudget::multistep_default());
    /// ```
    #[cfg(feature = "lacam_escalation")]
    pub fn with_escalation_budget(mut self, budget: EscalationBudget) -> Self {
        self.escalation_budget = budget;
        self
    }

    /// One tick of LLLG planning.
    ///
    /// The full pipeline per tick:
    /// 1. Compute the warm-start initialization from the previous tick's data.
    /// 2. Pass it to the guidance source via [`set_warm_start`](LocalGuidanceSource::set_warm_start).
    /// 3. Compute the guidance field `Φ` via the guidance source (pluggable).
    /// 4. Run PIBT to produce the joint action `Π_t[1]` (edge-collision-free;
    ///    vertex collisions may occur on congested maps).
    /// 5. Record the executed action + `Φ` into the warm-start cache for the
    ///    next tick.
    ///
    /// # Arguments
    ///
    /// - `config`: current joint configuration `Q_t` (raw, synced).
    /// - `goals`: per-agent goals `g_i` (raw).
    /// - `guidance`: the guidance source (pluggable seam #2).
    /// - `hindrance`: the hindrance estimator (pluggable seam #4).
    /// - `rng`: deterministic RNG for the PIBT `ε` tiebreak.
    ///
    /// # Returns
    ///
    /// The [`JointAction`] `Π_t[1]` — edge-collision-free; vertex collisions
    /// may occur on congested maps (see [`JointAction`] struct doc).
    pub fn tick<G, H>(
        &mut self,
        config: &JointConfig<P>,
        goals: &[P],
        guidance: &mut G,
        hindrance: &mut H,
        rng: &mut fastrand::Rng,
    ) -> JointAction<P>
    where
        G: LocalGuidanceSource<P>,
        H: HindranceEstimator<P>,
    {
        // 1. Produce warm-start initialization for this tick.
        let warm = self.warm_start.warm_start();

        // 2. Pass it to the guidance source. The source consumes it inside
        //    `compute_guidance` (default sources ignore it via the trait's
        //    no-op `set_warm_start`).
        if !warm.is_empty() {
            guidance.set_warm_start(warm);
        }

        // 3. Compute guidance field Φ.
        guidance.compute_guidance(config, goals, &mut self.guidance_scratch);

        // 4. Run PIBT with wall-aware neighbors (if set) + flow field (Issue 149).
        //
        // When the flow field is not set, use NoFlow (paper-faithful, zero-cost).
        let no_flow;
        let flow: &dyn FlowField<P> = if let Some(f) = &self.flow_field { f.as_ref() } else {
                no_flow = NoFlow;
                &no_flow
            };

        let action = pibt_step_with_budget(
            config,
            &self.guidance_scratch,
            goals,
            &self.priorities,
            hindrance,
            flow,
            self.neighbors_fn.as_deref(),
            rng,
            #[cfg(feature = "lacam_escalation")]
            self.escalation_budget,
        )
        .unwrap_or_else(|deadlock| {
            log::debug!(
                "LLLG deadlock: {} agents stuck, falling back to wait",
                deadlock.stuck_agents.len()
            );
            JointAction::from_wait(config)
        });

        // 5. Record the executed action + Φ for the next tick's warm-start.
        //    The "solution" prepends the executed PIBT action so the suffix
        //    extraction in `WarmStartCache::prev_solution_suffix` correctly
        //    skips the executed step (not the guidance's preferred first
        //    step, which may differ from the executed action — Issue 140 T2.6).
        let n = config.n_agents();
        let mut solution: Vec<Vec<P>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut path = Vec::with_capacity(self.guidance_scratch[i].len() + 1);
            path.push(action.moves[i].clone());
            path.extend(self.guidance_scratch[i].iter().cloned());
            solution.push(path);
        }
        self.warm_start
            .record(solution, self.guidance_scratch.clone());

        action
    }

    /// Access the warm-start cache (for scheme changes or inspection).
    pub fn warm_start_mut(&mut self) -> &mut WarmStartCache<P> {
        &mut self.warm_start
    }
}

impl<P: Position> Default for LifelongLaCam<P> {
    fn default() -> Self {
        let w_phi = GuidanceConfig::default().w_phi;
        Self::new(WarmStartCache::new(WarmStartScheme::default(), w_phi))
    }
}
