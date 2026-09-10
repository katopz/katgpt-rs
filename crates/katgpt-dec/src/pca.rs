//! PCA global-function layer — DEC aggregates wired into the CA decision
//! function (Plan 591, Research 544).
//!
//! Programmable Cellular Automata (arXiv:2609.06102) measured that ONE global
//! function (count / connectivity) collapses CA iteration counts 3–8× and
//! unlocks tasks purely-local rules cannot solve at all (Zelda 0% → 99%
//! playability). This module is that composition over the shipped DEC
//! substrate: the local rule stays [`stochastic_birth_death_step`] (Plan 454,
//! untouched), and one global aggregate is evaluated ONCE per tick from the
//! pre-sweep state, then handed — together with each cell's own channel slice
//! — to a per-cell [`PcaDecision`].
//!
//! # The paper's constraint, enforced by the API shape
//!
//! The decision NEVER reads raw state beyond its two arguments: the cell's
//! local channel slice (a kernel output) + the tick's [`GlobalScalars`]. There
//! is no parameter by which a decision could reach the whole field — the
//! "global knowledge" channel is exactly one scalar per tick, which is what
//! makes the global function programmable rather than emergent.
//!
//! # The globals ([`PcaGlobalFn`])
//!
//! Each evaluates against the state cochain + complex via the shipped DEC
//! operators:
//!
//! - [`PcaGlobalFn::Betti0`] — connected-component count of the ALIVE support
//!   (channel 0 > 0.5). This is deliberately NOT [`betti_numbers`] — that call
//!   is state-blind (it ranks the FULL complex's boundary matrices; a solid
//!   grid gives β₀ = 1 regardless of which cells are alive). The support-aware
//!   count has no closed form on the subcomplex without restricted Gaussian
//!   elimination, so it ships as an honest O(V + E) union-find scan (the same
//!   "document the scan" standard Phase 2 sets for `LargestComponentSize`).
//! - [`PcaGlobalFn::AliveCount`] — alive-cell count: the paper's COUNT global
//!   (the placement budget). The one arm that is O(1) incrementable on both
//!   births and deaths — the live counter [`step_pca_async`] tracks.
//! - [`PcaGlobalFn::BoundaryFluxMass`] — net morphogen flux across the domain
//!   boundary: flow = d(channel 1), region = ALL faces, so interior edges
//!   cancel by orientation and only boundary edges contribute (Stokes).
//!   `debug_assert!(dim ≤ 3)` — boundary-vs-volume evaluation is a win only
//!   for d ≤ 3 (the curse-of-dimensionality rule; at d ≥ 8 the boundary is
//!   larger than the interior).
//! - [`PcaGlobalFn::BeliefMassDivergence`] — ‖δ₁(flow)‖₁, the Fokker-Planck
//!   conservation-violation mass (same definition as
//!   [`belief_mass_divergence`], computed via [`codifferential_into`] so the
//!   per-tick path stays zero-alloc; equivalence pinned by test).
//! - [`PcaGlobalFn::Codifferential`] — ‖δ₁(flow)‖₂ (RMS-scale). Where the L1
//!   mass counts EVERY small divergence, the L2 norm emphasizes CONCENTRATED
//!   divergence — the clustering signal (Phase 2's crosswalk: which vertices
//!   carry negative δf).
//!
//! # Seed decision
//!
//! [`GlobalTargetGate`] is the construct pure-local CA cannot express: a
//! GLOBAL termination term. When the selected global has crossed `target`
//! (from above or below, per [`StopWhen`]), births stop — the per-cell pass
//! reverts dead→alive flips. Existing alive cells keep running their local
//! death/decay dynamics; only new placements are refused.
//!
//! # Rank/channel contract
//!
//! `field` is the Plan 454 layout: rank-0, dim ≥ 2, channel 0 = alive
//! (binarized 0/1 by the kernel each tick), channel 1 = morphogen. The flow
//! globals read channel 1; [`PcaGlobalFn::Betti0`] reads channel 0.
//!
//! # Zero-alloc
//!
//! All scratch is caller-owned [`PcaScratch`]; every per-tick path uses the
//! `_into` / `_scratched` operator variants. `DecCache` integration is
//! deliberately deferred: no Phase-0 global needs a Hodge decomposition (the
//! flux mass skips the error bound — a per-tick CG solve — by using the
//! mass-only path).
//!
//! # References
//!
//! - Plan 591 (this composition layer), Research 544 (the distillation).
//! - Programmable Cellular Automata (arXiv:2609.06102).
//! - [`stochastic_birth_death_step`] — the local kernel (Plan 454, arXiv:2103.08737).

use crate::birth_death::{BirthDeathParams, SplitMix64, stochastic_birth_death_step};
use crate::operators::{codifferential_into, exterior_derivative_into};
use crate::stokes_calculus::boundary_flux_mass_only_scratched;
use crate::types::{CellComplex, CochainField};

/// Which side of [`GlobalTargetGate::target`] means "stop".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopWhen {
    /// Stop once `globals.value >= target` (count-style globals: "stop
    /// placing when count ≥ k").
    Above,
    /// Stop once `globals.value <= target` (connectivity-style globals: a
    /// Betti0 task stops at one component).
    Below,
}

/// The tick's global scalars — the ONLY channel through which a
/// [`PcaDecision`] sees world state beyond its own cell.
///
/// Phase 0 carries the single selected [`PcaGlobalFn`] value. Phase 1's async
/// step may extend this with incremental counters; the type is the stable
/// decision-input contract, so new fields must stay optional additions
/// (decisions cannot REQUIRE a field that older steps never filled).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalScalars {
    /// The selected [`PcaGlobalFn`]'s value for this tick, evaluated on the
    /// PRE-sweep state.
    pub value: f32,
}

/// The selected global aggregate, computed once per iteration.
///
/// Cheap to `Copy` — switching the global per tick is legal; the GOAT bench
/// (Phase 3) holds it fixed per run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcaGlobalFn {
    /// Connected-component count of the alive support (channel 0 > 0.5).
    /// NOT [`betti_numbers`] — that is state-blind (module doc explains).
    Betti0,
    /// Alive-cell count (channel 0 > 0.5) — the paper's COUNT global (the
    /// Zelda placement budget). O(1) incrementable on births AND deaths, so
    /// this is the one arm [`step_pca_async`] tracks with a live counter
    /// (all other arms freeze their pre-tick value through the sweep).
    AliveCount,
    /// Net morphogen flux across the domain boundary. The flow is the
    /// ENDPOINT-SUM edge lift of the morphogen (see [`morphogen_edge_lift`] —
    /// the gradient lift would be identically zero here by `d∘d = 0`).
    BoundaryFluxMass,
    /// ‖δ₁(flow)‖₁ — total conservation-violation mass.
    BeliefMassDivergence,
    /// ‖δ₁(flow)‖₂ — concentrated-divergence (clustering) emphasis.
    Codifferential,
}

impl PcaGlobalFn {
    /// Evaluate this global against the PRE-tick state. Does not mutate
    /// `field`; all scratch goes through `scratch` (zero-alloc).
    pub fn evaluate(&self, cx: &CellComplex, field: &CochainField, scratch: &mut PcaScratch) -> f32 {
        match self {
            Self::Betti0 => betti0_of_support(cx, field, scratch),
            Self::AliveCount => {
                let n = cx.n_cells(0);
                let mut count = 0u32;
                for v in 0..n {
                    count += alive_at(field, v) as u32;
                }
                count as f32
            }
            Self::BoundaryFluxMass => {
                // Boundary-vs-volume is a win only for d ≤ 3 (Plan 591
                // constraint; the AGENTS.md manifold rule). Untrippable with
                // MAX_RANK = 3 today — a forward-compatibility guard.
                debug_assert!(
                    complex_dim(cx) <= 3,
                    "BoundaryFluxMass on a d>3 complex: boundary evaluation is a LOSS there"
                );
                morphogen_edge_lift(cx, field, scratch);
                if scratch.region_cells.is_empty() {
                    return 0.0;
                }
                boundary_flux_mass_only_scratched(
                    cx,
                    &scratch.region_cells,
                    &scratch.flow,
                    &mut scratch.in_region,
                )
            }
            Self::BeliefMassDivergence => {
                morphogen_flow(cx, field, scratch);
                codifferential_into(cx, &scratch.flow, &mut scratch.div);
                // ‖δf‖₁ — same definition as `belief_mass_divergence`, via the
                // zero-alloc `_into` operator (equivalence pinned by test).
                scratch.div.data.iter().copied().map(f32::abs).sum()
            }
            Self::Codifferential => {
                morphogen_flow(cx, field, scratch);
                codifferential_into(cx, &scratch.flow, &mut scratch.div);
                let sum_sq: f32 = scratch.div.data.iter().map(|&x| x * x).sum();
                sum_sq.sqrt()
            }
        }
    }
}

/// Topological dimension of the complex: the highest rank with cells.
fn complex_dim(cx: &CellComplex) -> u8 {
    (0..=crate::types::MAX_RANK)
        .rev()
        .find(|&r| cx.n_cells(r) > 0)
        .unwrap_or(0)
}

/// Alive test on the Plan 454 layout: channel 0 > 0.5 (the kernel binarizes
/// to exactly 0/1 each tick; a hand-built pre-tick field is compared at the
/// same threshold the kernel's gate uses).
#[inline]
fn alive_at(field: &CochainField, v: usize) -> bool {
    field.data[v * field.dim] > 0.5
}

/// Copy channel 1 (morphogen) into `scratch.morph` and take its gradient into
/// `scratch.flow` — the rank-1 dim-1 flow the divergence globals run on.
fn morphogen_flow(cx: &CellComplex, field: &CochainField, scratch: &mut PcaScratch) {
    debug_assert!(field.dim >= 2, "morphogen channel missing (need dim >= 2)");
    debug_assert_eq!(scratch.morph.data.len(), cx.n_cells(0));
    for (dst, src) in scratch
        .morph
        .data
        .iter_mut()
        .zip(field.data[1..].iter().step_by(field.dim))
    {
        *dst = *src;
    }
    exterior_derivative_into(cx, &scratch.morph, &mut scratch.flow);
}

/// Lift the vertex morphogen to edges by ENDPOINT SUM — `f[e] = m(tail) +
/// m(head)` — into `scratch.flow`.
///
/// Why not the gradient d(morph)? Because the flux of a gradient around ANY
/// closed boundary is identically zero (`d∘d = 0` — the DEC identity this
/// crate is built on), so `boundary_flux(d(morph))` would be a degenerate
/// global: zero for every field, every region, every tick. The endpoint-sum
/// lift is the canonical operator-compatible pushforward under identity Hodge
/// stars and is NOT a gradient in general (its curl survives), so the flux
/// arm measures something real: how much morphogen sits on the domain rim vs
/// the interior.
fn morphogen_edge_lift(cx: &CellComplex, field: &CochainField, scratch: &mut PcaScratch) {
    debug_assert!(field.dim >= 2, "morphogen channel missing (need dim >= 2)");
    debug_assert_eq!(scratch.morph.data.len(), cx.n_cells(0));
    debug_assert_eq!(scratch.flow.data.len(), cx.n_cells(1));
    // Copy channel 1 into scratch.morph (same lift-in as the gradient path).
    for (dst, src) in scratch
        .morph
        .data
        .iter_mut()
        .zip(field.data[1..].iter().step_by(field.dim))
    {
        *dst = *src;
    }
    // f[e] = m(tail) + m(head) — the B₁ entries give (vertex, edge) pairs;
    // an edge accumulates the morphogen of each of its endpoints.
    scratch.flow.data.fill(0.0);
    for &(v, e, _) in cx.boundary_entries(0) {
        if v < scratch.morph.data.len() {
            scratch.flow.data[e] += scratch.morph.data[v];
        }
    }
}

/// Connected-component count of the alive support — O(V + E) union-find over
/// the complex's B₁ incidence (module doc explains why there is no closed
/// form). `count` starts at the alive-cell count and drops once per
/// successful union.
fn betti0_of_support(cx: &CellComplex, field: &CochainField, scratch: &mut PcaScratch) -> f32 {
    let n = cx.n_cells(0);
    debug_assert_eq!(field.rank, 0, "Betti0 needs the rank-0 state cochain");
    debug_assert_eq!(scratch.parent.len(), n, "PcaScratch built for this cx?");
    debug_assert!(
        scratch.first_alive_of_edge.len() >= cx.n_cells(1),
        "PcaScratch built for this cx?"
    );

    // Init: alive cell = own root; dead = sentinel.
    let mut count: u32 = 0;
    for (v, slot) in scratch.parent.iter_mut().enumerate().take(n) {
        if alive_at(field, v) {
            *slot = v as u32;
            count += 1;
        } else {
            *slot = u32::MAX;
        }
    }
    scratch
        .first_alive_of_edge
        .iter_mut()
        .for_each(|x| *x = u32::MAX);

    // One representative per edge: the first alive endpoint anchors the edge;
    // every later alive endpoint unions with it. Two B₁ entries of the same
    // edge may be non-adjacent in the triplet list (hand-built complexes), so
    // the per-edge slot — not entry order — is what makes this correct.
    for &(v, e, _) in cx.boundary_entries(0) {
        if v >= n || !alive_at(field, v) {
            continue;
        }
        debug_assert!(e < scratch.first_alive_of_edge.len());
        let anchor = scratch.first_alive_of_edge[e];
        if anchor == u32::MAX {
            scratch.first_alive_of_edge[e] = v as u32;
            continue;
        }
        let a = find_root(&mut scratch.parent, anchor);
        let b = find_root(&mut scratch.parent, v as u32);
        if a != b {
            // Union by smaller root (indices are stable) — no rank array, no
            // allocation; path halving in `find_root` keeps this near-linear.
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            scratch.parent[hi as usize] = lo;
            count -= 1;
        }
    }
    count as f32
}

/// Union-find root with path halving. Dead cells are `u32::MAX` sentinels and
/// never enter here (callers only `find` live roots).
fn find_root(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let grandparent = parent[parent[x as usize] as usize];
        parent[x as usize] = grandparent;
        x = grandparent;
    }
    x
}

/// The per-cell decision function — the paper's constraint as a trait: the
/// decision consumes ONLY its own cell's channel slice (a kernel output) plus
/// the tick's [`GlobalScalars`]. There is no way to reach the rest of the
/// field through this signature.
pub trait PcaDecision {
    /// Decide for one cell. Returns the birth permission: `> 0` lets the
    /// cell's dead→alive flip stand, `<= 0` reverts it. Called once per
    /// NEWBORN cell per tick (cells that were already alive are the local
    /// kernel's business; the global layer never kills them).
    fn decide(&self, local_neighborhood: &[f32], globals: &GlobalScalars) -> f32;
}

/// The seed decision (Plan 591 Phase 0): the GLOBAL termination term.
///
/// "Stop placing when count ≥ k" — with [`StopWhen::Above`], or "stop once
/// connected" — with [`StopWhen::Below`] on a Betti0 target of 1. While the
/// selected global has not crossed `target`, births run unmodified (this gate
/// returns 1.0); once crossed, every newborn is reverted (0.0) — the construct
/// a purely-local rule cannot express, per the Programmable-CA paper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalTargetGate {
    /// The termination threshold on the selected global's value.
    pub target: f32,
    /// Which side of `target` means "stop".
    pub stop_when: StopWhen,
}

impl PcaDecision for GlobalTargetGate {
    fn decide(&self, _local_neighborhood: &[f32], globals: &GlobalScalars) -> f32 {
        let reached = match self.stop_when {
            StopWhen::Above => globals.value >= self.target,
            StopWhen::Below => globals.value <= self.target,
        };
        if reached { 0.0 } else { 1.0 }
    }
}

/// Caller-owned scratch for the PCA step — pre-allocate once per
/// (complex, field-dim), reuse across every tick.
///
/// Sized for a rank-0 dim-`field_dim` state cochain on `cx`. Not tied to one
/// [`PcaGlobalFn`] — every arm reuses the same buffers.
pub struct PcaScratch {
    /// Rank-0 dim-1 copy of the morphogen channel (the d input).
    morph: CochainField,
    /// Rank-1 dim-1 gradient d(morph) — the flow the divergence/flux globals
    /// run on.
    flow: CochainField,
    /// Rank-0 dim-1 divergence δ(flow).
    div: CochainField,
    /// All face indices — the region for the domain-boundary flux (k=1 flow,
    /// k+1=2 cells).
    region_cells: Vec<u32>,
    /// Region-membership marker for [`boundary_flux_mass_only_scratched`].
    in_region: Vec<bool>,
    /// Union-find parent slots (alive root / `u32::MAX` dead sentinel).
    parent: Vec<u32>,
    /// Per-edge first-alive-vertex slot for the Betti0 scan.
    first_alive_of_edge: Vec<u32>,
    /// Alive channel snapshot taken before the kernel tick (the birth gate
    /// compares pre/post to find newborns).
    alive_before: Vec<u8>,
    /// The kernel's Laplacian scratch (see [`stochastic_birth_death_step`]).
    lap: CochainField,
    /// The kernel's dropout mask (see [`stochastic_birth_death_step`]).
    dropout: Vec<u8>,
}

impl PcaScratch {
    /// Size every buffer for a rank-0 dim-`field_dim` state on `cx`.
    pub fn for_complex(cx: &CellComplex, field_dim: usize) -> Self {
        let nv = cx.n_cells(0);
        let ne = cx.n_cells(1);
        let nf = cx.n_cells(2);
        Self {
            morph: CochainField::zeros(0, nv, 1),
            flow: CochainField::zeros(1, ne, 1),
            div: CochainField::zeros(0, nv, 1),
            region_cells: (0..nf as u32).collect(),
            in_region: vec![false; nf],
            parent: vec![0; nv],
            first_alive_of_edge: vec![u32::MAX; ne],
            alive_before: vec![0; nv],
            lap: CochainField::zeros(0, nv, field_dim),
            dropout: vec![0; nv],
        }
    }
}

/// One tick of PCA-sync: compute the global ONCE (pre-sweep), run the
/// untouched Plan 454 local kernel, then apply the per-cell decision to
/// newborns.
///
/// # Sequence (the plan's "compute globals once per iteration, then sweep")
///
/// 1. `global_fn.evaluate` on the PRE-tick state → [`GlobalScalars`].
/// 2. Alive-channel snapshot.
/// 3. [`stochastic_birth_death_step`] — the local rule, byte-identical to
///    running it standalone with the same seed.
/// 4. Per-cell pass: every dead→alive flip is offered to `decision.decide`
///    (the cell's post-kernel channel slice + the tick's globals); a
///    `<= 0` permission reverts the flip to dead.
///
/// A reverted newborn keeps its post-kernel morphogen (step D already skipped
/// decay for it) — it may re-attempt birth next tick and be refused again.
/// That is the intended "stop PLACING" semantics: the birth channel is
/// closed, existing cells keep their local dynamics.
///
/// # Returns
///
/// The tick's global value (the pre-sweep evaluation) — the caller's
/// termination/metrics readout without re-computing.
///
/// # Determinism
///
/// Bit-identical across runs for the same (field, params, seed, global_fn,
/// decision): steps 1–3 are pure functions of state, step 4 iterates in
/// vertex-index order.
#[inline]
pub fn step_pca_sync(
    cx: &CellComplex,
    field: &mut CochainField,
    params: &BirthDeathParams,
    rng: &mut SplitMix64,
    global_fn: &PcaGlobalFn,
    decision: &dyn PcaDecision,
    scratch: &mut PcaScratch,
) -> f32 {
    // 1. Globals once per iteration — PRE-sweep state.
    let value = global_fn.evaluate(cx, field, scratch);
    let globals = GlobalScalars { value };

    // 2. Alive snapshot (the gate compares pre/post to find newborns).
    let n = cx.n_cells(0);
    let dim = field.dim;
    debug_assert_eq!(field.rank, 0, "step_pca_sync needs the rank-0 state");
    debug_assert!(dim >= 2, "state needs alive + morphogen channels");
    debug_assert_eq!(scratch.alive_before.len(), n, "PcaScratch built for this cx?");
    for (v, slot) in scratch.alive_before.iter_mut().enumerate().take(n) {
        *slot = alive_at(field, v) as u8;
    }

    // 3. Local kernel tick — Plan 454's tested code, UNTOUCHED.
    stochastic_birth_death_step(cx, field, params, rng, &mut scratch.lap, &mut scratch.dropout);

    // 4. Per-cell decision pass over newborns, vertex-index order.
    for v in 0..n {
        if scratch.alive_before[v] == 0 && alive_at(field, v) {
            let local = &field.data[v * dim..v * dim + dim];
            if decision.decide(local, &globals) <= 0.0 {
                field.data[v * dim] = 0.0;
            }
        }
    }
    value
}

/// One tick of PCA-ASYNC — the paper's asynchronous dynamic: the decision
/// sees a LIVE global, updated O(1) on each cell write during the sweep,
/// instead of the frozen pre-tick value [`step_pca_sync`] gates against.
///
/// # The failure this closes (paper §3, the stale-count counter example)
///
/// On a k-target placement task the sync step gates the WHOLE birth batch
/// against the pre-tick count: with count k−1 and 8 pending births, every
/// birth sees "under target" and is allowed → final count k+7 — OVERSHOOT.
/// The async step processes the same batch in a FIXED row-major traversal,
/// incrementing the counter on every allowed birth, so placements stop
/// EXACTLY at k. Deterministic by traversal order (Gauss–Seidel discipline:
/// the fixed order replaces the sync step's batch semantics, not a wall
/// clock).
///
/// # Incremental-global scope (honest)
///
/// Only [`PcaGlobalFn::AliveCount`] supports O(1) updates on both births and
/// deaths; for every other arm this step DEGENERATES to sync semantics (the
/// pre-tick value frozen through the sweep — identical results, documented
/// rather than hidden). `Betti0` births are union-find-incremental, but a
/// death can SPLIT a component and union-find cannot delete — wiring that
/// hybrid (birth-incremental + death-dirty resync) is deferred until a
/// consumer needs it.
///
/// # Sequence
///
/// 1. Full `global_fn.evaluate` on the PRE-tick state (the live counter's
///    seed value).
/// 2. Alive snapshot; kernel tick — identical to [`step_pca_sync`].
/// 3. Fixed row-major pass over ALL cells: a DEATH (alive→dead, e.g. via
///    crowding) decrements the counter; a BIRTH is offered to `decision`
///    with the CURRENT counter value — allowed births increment it, refused
///    births are reverted (and do not increment).
///
/// # Returns
///
/// The POST-pass global value — for `AliveCount` the live counter (the
/// caller's exact current count); for non-incremental arms the pre-tick
/// value (matching [`step_pca_sync`]).
///
/// # Determinism
///
/// Same (field, params, seed, global_fn, decision) → bit-identical final
/// grid: the traversal order is fixed row-major, the kernel is the same
/// seeded batch update, and the counter path is pure arithmetic (property
/// test at ≥100 seeds).
#[inline]
pub fn step_pca_async(
    cx: &CellComplex,
    field: &mut CochainField,
    params: &BirthDeathParams,
    rng: &mut SplitMix64,
    global_fn: &PcaGlobalFn,
    decision: &dyn PcaDecision,
    scratch: &mut PcaScratch,
) -> f32 {
    // 1. The live counter's seed — full pre-tick evaluation.
    let mut live = global_fn.evaluate(cx, field, scratch);

    // 2. Alive snapshot + the untouched local kernel.
    let n = cx.n_cells(0);
    let dim = field.dim;
    debug_assert_eq!(field.rank, 0, "step_pca_async needs the rank-0 state");
    debug_assert!(dim >= 2, "state needs alive + morphogen channels");
    debug_assert_eq!(scratch.alive_before.len(), n, "PcaScratch built for this cx?");
    for (v, slot) in scratch.alive_before.iter_mut().enumerate().take(n) {
        *slot = alive_at(field, v) as u8;
    }
    stochastic_birth_death_step(cx, field, params, rng, &mut scratch.lap, &mut scratch.dropout);

    // 3. Fixed row-major pass with the live counter. Only AliveCount is
    //    incrementable (enum doc); other arms keep the frozen value.
    let incrementable = matches!(global_fn, PcaGlobalFn::AliveCount);
    for v in 0..n {
        let was_alive = scratch.alive_before[v] != 0;
        let now_alive = alive_at(field, v);
        if was_alive && !now_alive {
            // Death (crowding): O(1) decrement — before the births AFTER v
            // in the order see it, which is the point of the live counter.
            if incrementable {
                live -= 1.0;
            }
        } else if !was_alive && now_alive {
            let local = &field.data[v * dim..v * dim + dim];
            if decision.decide(local, &GlobalScalars { value: live }) <= 0.0 {
                field.data[v * dim] = 0.0;
            } else if incrementable {
                live += 1.0;
            }
        }
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stokes_calculus::belief_mass_divergence;
    use crate::types::CellComplex;

    fn make_field(cx: &CellComplex, dim: usize) -> CochainField {
        CochainField::zeros(0, cx.n_cells(0), dim)
    }

    fn seed_alive_morph(field: &mut CochainField, verts: &[usize], morph: f32, dim: usize) {
        for &v in verts {
            field.data[v * dim] = 1.0;
            field.data[v * dim + 1] = morph;
        }
    }

    /// Naive BFS component count — the independent reference the union-find
    /// is pinned against (spec-match convention).
    fn betti0_reference(cx: &CellComplex, field: &CochainField) -> usize {
        let n = cx.n_cells(0);
        let alive: Vec<bool> = (0..n).map(|v| alive_at(field, v)).collect();
        let mut seen = vec![false; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(v, e, _) in cx.boundary_entries(0) {
            if v < n {
                adj[v].push(e);
            }
        }
        // Edge → its vertices (from the same incidence list).
        let mut edge_verts: Vec<Vec<usize>> = vec![Vec::new(); cx.n_cells(1)];
        for &(v, e, _) in cx.boundary_entries(0) {
            if v < n && e < edge_verts.len() {
                edge_verts[e].push(v);
            }
        }
        let mut count = 0;
        for start in 0..n {
            if !alive[start] || seen[start] {
                continue;
            }
            count += 1;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(v) = stack.pop() {
                for &e in &adj[v] {
                    for &w in &edge_verts[e] {
                        if alive[w] && !seen[w] {
                            seen[w] = true;
                            stack.push(w);
                        }
                    }
                }
            }
        }
        count
    }

    #[test]
    fn betti0_counts_alive_components() {
        let cx = CellComplex::grid_2d(5, 5);
        let mut field = make_field(&cx, 2);
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        assert_eq!(PcaGlobalFn::Betti0.evaluate(&cx, &field, &mut scratch), 0.0);

        // {0,1,2} (row 0) + {12,13} (row 2) — two disconnected runs.
        seed_alive_morph(&mut field, &[0, 1, 2, 12, 13], 1.0, 2);
        assert_eq!(PcaGlobalFn::Betti0.evaluate(&cx, &field, &mut scratch), 2.0);

        // v7 (row 1, col 2) bridges both runs → one component.
        field.data[7 * 2] = 1.0;
        assert_eq!(PcaGlobalFn::Betti0.evaluate(&cx, &field, &mut scratch), 1.0);
    }

    #[test]
    fn betti0_union_find_matches_bfs_reference() {
        let cx = CellComplex::grid_2d(8, 6);
        let mut rng = SplitMix64::new(0xDEAD_BEEF);
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        for _ in 0..25 {
            let mut field = make_field(&cx, 2);
            for v in 0..cx.n_cells(0) {
                let r = (rng.next_u32() % 100) as f32 / 100.0;
                if r < 0.4 {
                    field.data[v * 2] = 1.0;
                }
            }
            let got = PcaGlobalFn::Betti0.evaluate(&cx, &field, &mut scratch);
            let want = betti0_reference(&cx, &field);
            assert_eq!(got as usize, want, "union-find disagrees with BFS");
        }
    }

    #[test]
    fn flux_matches_the_volume_integral_stokes_identity() {
        // Discrete Stokes: boundary flux over a face region == Σ_region_faces
        // d₁(flow). Non-degenerate by construction — the endpoint-sum lift is
        // NOT a gradient (its curl survives), so both sides are nonzero.
        let cx = CellComplex::grid_2d(6, 5);
        let mut field = make_field(&cx, 2);
        let mut scratch = PcaScratch::for_complex(&cx, 2);

        // morphogen = the x coordinate.
        let w = 6;
        for v in 0..cx.n_cells(0) {
            field.data[v * 2 + 1] = (v % w) as f32;
        }

        let boundary = PcaGlobalFn::BoundaryFluxMass.evaluate(&cx, &field, &mut scratch);

        // Hand-computed expectation (B₂ signs: bottom +1, top −1, right +1,
        // left −1; lift f[e] = m(tail)+m(head) on the x-ramp → horizontal
        // edges 2x+1, vertical edges 2x):
        //   bottom row (y=0, sign +1):  Σ_x (2x+1) = 2·10 + 5 = 25
        //   top row    (y=4, sign −1): −25
        //   left col   (x=0, sign −1): −Σ 0 = 0
        //   right col  (x=5, sign +1): 4 edges × 10 = +40
        assert!((boundary - 40.0).abs() < 1e-4, "boundary flux {boundary} != 40");

        // Volume side: the lift, then d₁ edges→faces, summed over ALL faces.
        morphogen_edge_lift(&cx, &field, &mut scratch);
        let flow_copy = scratch.flow.clone();
        let curl = crate::operators::exterior_derivative(&cx, &flow_copy);
        let volume: f32 = curl.data.iter().sum();
        assert!(
            (boundary - volume).abs() < 1e-3,
            "boundary flux {boundary} != volume integral {volume}"
        );
    }

    #[test]
    fn gradient_lift_flux_would_be_degenerate_but_sum_lift_is_not() {
        // The design finding as a regression pin: d(morph) around the closed
        // domain is identically zero (d∘d = 0), so the flux arm MUST NOT use
        // the gradient lift. The endpoint-sum lift stays nonzero on the same
        // field (proven by `flux_matches_the_volume_integral_stokes_identity`).
        let cx = CellComplex::grid_2d(6, 5);
        let mut field = make_field(&cx, 2);
        let w = 6;
        for v in 0..cx.n_cells(0) {
            field.data[v * 2 + 1] = (v % w) as f32;
        }
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        morphogen_flow(&cx, &field, &mut scratch); // the gradient path
        let degenerate = boundary_flux_mass_only_scratched(
            &cx,
            &scratch.region_cells,
            &scratch.flow,
            &mut scratch.in_region,
        );
        assert_eq!(degenerate, 0.0, "flux of a gradient must vanish by d∘d = 0");
    }

    #[test]
    fn constant_morphogen_yields_zero_flow_globals() {
        let cx = CellComplex::grid_2d(4, 4);
        let mut field = make_field(&cx, 2);
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        seed_alive_morph(&mut field, &[0], 1.0, 2);
        // Overwrite every morphogen with the same constant → d = 0, δ = 0.
        for v in 0..cx.n_cells(0) {
            field.data[v * 2 + 1] = 2.5;
        }
        assert_eq!(PcaGlobalFn::BoundaryFluxMass.evaluate(&cx, &field, &mut scratch), 0.0);
        assert_eq!(PcaGlobalFn::BeliefMassDivergence.evaluate(&cx, &field, &mut scratch), 0.0);
        assert_eq!(PcaGlobalFn::Codifferential.evaluate(&cx, &field, &mut scratch), 0.0);
    }

    #[test]
    fn belief_mass_divergence_l1_matches_the_shipped_function() {
        // The L1 arm must be BIT-IDENTICAL to the shipped
        // `belief_mass_divergence` (same definition, zero-alloc path).
        let cx = CellComplex::grid_2d(5, 4);
        let mut rng = SplitMix64::new(42);
        let mut field = make_field(&cx, 3); // dim=3: extra channel must be ignored
        for v in 0..cx.n_cells(0) {
            field.data[v * 3 + 1] = (rng.next_u32() % 1000) as f32 / 500.0 - 1.0;
        }
        let mut scratch = PcaScratch::for_complex(&cx, 3);
        let got = PcaGlobalFn::BeliefMassDivergence.evaluate(&cx, &field, &mut scratch);

        let mut morph = CochainField::zeros(0, cx.n_cells(0), 1);
        for v in 0..cx.n_cells(0) {
            morph.data[v] = field.data[v * 3 + 1];
        }
        let flow = crate::operators::exterior_derivative(&cx, &morph);
        let want = belief_mass_divergence(&cx, &flow);
        assert_eq!(got, want, "L1 divergence drifted from the shipped definition");
    }

    #[test]
    fn codifferential_l2_is_bounded_by_the_l1_mass() {
        let cx = CellComplex::grid_2d(6, 6);
        let mut rng = SplitMix64::new(7);
        let mut field = make_field(&cx, 2);
        for v in 0..cx.n_cells(0) {
            field.data[v * 2 + 1] = (rng.next_u32() % 2000) as f32 / 1000.0 - 1.0;
        }
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        let l2 = PcaGlobalFn::Codifferential.evaluate(&cx, &field, &mut scratch);
        let l1 = PcaGlobalFn::BeliefMassDivergence.evaluate(&cx, &field, &mut scratch);
        assert!(l2 <= l1 + 1e-6, "L2 {l2} exceeded L1 {l1}");
        assert!(l2 > 0.0, "random field should carry nonzero divergence");
    }

    #[test]
    fn evaluate_does_not_mutate_the_field() {
        let cx = CellComplex::grid_2d(4, 4);
        let mut field = make_field(&cx, 2);
        seed_alive_morph(&mut field, &[0, 5, 10], 0.7, 2);
        let before = field.data.clone();
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        for g in [
            PcaGlobalFn::Betti0,
            PcaGlobalFn::BoundaryFluxMass,
            PcaGlobalFn::BeliefMassDivergence,
            PcaGlobalFn::Codifferential,
        ] {
            let _ = g.evaluate(&cx, &field, &mut scratch);
            assert_eq!(field.data, before, "{g:?} mutated the state");
        }
    }

    #[test]
    fn d_guard_holds_and_all_arms_run_on_2d_and_3d() {
        let cx2 = CellComplex::grid_2d(3, 3);
        let cx3 = CellComplex::grid_3d(2, 2, 2);
        assert_eq!(complex_dim(&cx2), 2);
        assert_eq!(complex_dim(&cx3), 3);
        for cx in [&cx2, &cx3] {
            let mut field = make_field(cx, 2);
            seed_alive_morph(&mut field, &[0], 1.0, 2);
            let mut scratch = PcaScratch::for_complex(cx, 2);
            for g in [
                PcaGlobalFn::Betti0,
                PcaGlobalFn::BoundaryFluxMass,
                PcaGlobalFn::BeliefMassDivergence,
                PcaGlobalFn::Codifferential,
            ] {
                // Debug builds exercise the d ≤ 3 boundary-flux assert here.
                let _ = g.evaluate(cx, &field, &mut scratch);
            }
        }
    }

    /// Records every `decide` call — the decision-input purity probe.
    struct RecordingDecision {
        calls: std::cell::RefCell<Vec<(Vec<f32>, f32)>>,
    }
    impl RecordingDecision {
        fn new() -> Self {
            Self { calls: std::cell::RefCell::new(Vec::new()) }
        }
    }
    impl PcaDecision for RecordingDecision {
        fn decide(&self, local_neighborhood: &[f32], globals: &GlobalScalars) -> f32 {
            self.calls
                .borrow_mut()
                .push((local_neighborhood.to_vec(), globals.value));
            1.0 // always allow — the probe measures, it does not gate
        }
    }

    #[test]
    fn decision_sees_only_its_slice_and_the_pretick_global() {
        let cx = CellComplex::grid_2d(4, 4);
        let mut field = make_field(&cx, 2);
        seed_alive_morph(&mut field, &[0], 1.0, 2);

        // The global is a PRE-tick evaluation — pin the expected value on a
        // clone before stepping.
        let pre = field.clone();
        let mut probe_scratch = PcaScratch::for_complex(&cx, 2);
        let expected_global = PcaGlobalFn::Betti0.evaluate(&cx, &pre, &mut probe_scratch);

        let params = BirthDeathParams { dropout_prob: 0.0, ..BirthDeathParams::paper_defaults() };
        let probe = RecordingDecision::new();
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        let returned = step_pca_sync(
            &cx,
            &mut field,
            &params,
            &mut SplitMix64::new(99),
            &PcaGlobalFn::Betti0,
            &probe,
            &mut scratch,
        );
        assert_eq!(returned, expected_global, "step must return the pre-tick global");

        let calls = probe.calls.borrow();
        assert!(!calls.is_empty(), "the seed tick must birth neighbors");
        for (local, value) in calls.iter() {
            assert_eq!(value, &expected_global, "decision saw a non-pre-tick global");
            assert_eq!(local.len(), 2, "decision slice must be the cell's channel row");
        }

        // The recorded slices must be exactly the newborns' post-kernel rows,
        // in vertex order — cross-checked against a standalone kernel run.
        let mut clone = pre.clone();
        stochastic_birth_death_step(
            &cx,
            &mut clone,
            &params,
            &mut SplitMix64::new(99),
            &mut PcaScratch::for_complex(&cx, 2).lap,
            &mut PcaScratch::for_complex(&cx, 2).dropout,
        );
        let newborns: Vec<Vec<f32>> = (0..cx.n_cells(0))
            .filter(|&v| clone.data[v * 2] > 0.5 && pre.data[v * 2] <= 0.5)
            .map(|v| clone.data[v * 2..v * 2 + 2].to_vec())
            .collect();
        let recorded: Vec<Vec<f32>> = calls.iter().map(|(l, _)| l.clone()).collect();
        assert_eq!(recorded, newborns, "decision input != kernel output rows");
    }

    #[test]
    fn target_gate_stops_births_but_not_local_dynamics() {
        let cx = CellComplex::grid_2d(4, 4);
        let params = BirthDeathParams { dropout_prob: 0.0, ..BirthDeathParams::paper_defaults() };

        let mk = || {
            let mut f = make_field(&cx, 2);
            seed_alive_morph(&mut f, &[0], 1.0, 2);
            f
        };

        // Always-allow: the seed's neighbors get born tick 1.
        struct Allow;
        impl PcaDecision for Allow {
            fn decide(&self, _: &[f32], _: &GlobalScalars) -> f32 {
                1.0
            }
        }
        let mut grown = mk();
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        step_pca_sync(
            &cx,
            &mut grown,
            &params,
            &mut SplitMix64::new(5),
            &PcaGlobalFn::Betti0,
            &Allow,
            &mut scratch,
        );
        let grown_alive = (0..cx.n_cells(0)).filter(|&v| grown.data[v * 2] > 0.5).count();
        assert!(grown_alive > 1, "kernel must birth neighbors for this test to bite");

        // Already-tripped gate (Betti0 ≤ 5 is true at the seed's 1 component):
        // NO new births — alive count stays exactly 1 (the seed is untouched —
        // the global layer never kills existing cells).
        struct StopAll;
        impl PcaDecision for StopAll {
            fn decide(&self, _: &[f32], _: &GlobalScalars) -> f32 {
                0.0
            }
        }
        let mut stopped = mk();
        step_pca_sync(
            &cx,
            &mut stopped,
            &params,
            &mut SplitMix64::new(5),
            &PcaGlobalFn::Betti0,
            &StopAll,
            &mut scratch,
        );
        let stopped_alive = (0..cx.n_cells(0)).filter(|&v| stopped.data[v * 2] > 0.5).count();
        assert_eq!(stopped_alive, 1, "a tripped gate must refuse every birth");
    }

    #[test]
    fn global_target_gate_direction_semantics() {
        let g = GlobalScalars { value: 3.0 };
        let above = GlobalTargetGate { target: 5.0, stop_when: StopWhen::Above };
        let below = GlobalTargetGate { target: 5.0, stop_when: StopWhen::Below };
        assert_eq!(above.decide(&[], &g), 1.0, "3 < 5: count task keeps placing");
        assert_eq!(below.decide(&[], &g), 0.0, "3 <= 5: connectivity task already done");
        let at = GlobalScalars { value: 5.0 };
        assert_eq!(above.decide(&[], &at), 0.0, "value == target counts as reached");
        assert_eq!(below.decide(&[], &at), 0.0);
    }

    #[test]
    fn step_pca_sync_is_bit_identical_for_a_fixed_seed() {
        let cx = CellComplex::grid_2d(8, 8);
        let params = BirthDeathParams::paper_defaults();
        for seed in 0..10u64 {
            let run = || {
                let mut field = make_field(&cx, 2);
                seed_alive_morph(&mut field, &[0, 40], 1.0, 2);
                let mut scratch = PcaScratch::for_complex(&cx, 2);
                let gate = GlobalTargetGate { target: 1.0, stop_when: StopWhen::Below };
                for _ in 0..5 {
                    step_pca_sync(
                        &cx,
                        &mut field,
                        &params,
                        &mut SplitMix64::new(seed),
                        &PcaGlobalFn::Betti0,
                        &gate,
                        &mut scratch,
                    );
                }
                field.data.clone()
            };
            assert_eq!(run(), run(), "seed {seed}: same inputs must be bit-identical");
        }
    }

    #[test]
    fn alive_count_global_counts_the_support() {
        let cx = CellComplex::grid_2d(5, 5);
        let mut field = make_field(&cx, 2);
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        assert_eq!(PcaGlobalFn::AliveCount.evaluate(&cx, &field, &mut scratch), 0.0);
        seed_alive_morph(&mut field, &[0, 1, 12], 1.0, 2);
        assert_eq!(PcaGlobalFn::AliveCount.evaluate(&cx, &field, &mut scratch), 3.0);
    }

    #[test]
    fn async_avoids_the_stale_count_over_placement() {
        // The paper §3 counter example: a k-target placement task. Sync gates
        // the WHOLE batch against the pre-tick count → overshoot; async's
        // live counter stops placements EXACTLY at k.
        let cx = CellComplex::grid_2d(5, 5);
        let params = BirthDeathParams { dropout_prob: 0.0, ..BirthDeathParams::paper_defaults() };
        let k = 5.0f32;

        // Two far-apart interior seeds (v7 row 1, v17 row 3 on the 5×5):
        // 2 alive pre-tick, 7 unique pending births across the batch.
        let mk = || {
            let mut f = make_field(&cx, 2);
            seed_alive_morph(&mut f, &[7, 17], 1.0, 2);
            f
        };
        let budget_gate = |target: f32| GlobalTargetGate {
            target,
            stop_when: StopWhen::Above,
        };

        // SYNC: every birth in the batch sees the frozen pre-tick count 2 < k
        // → all allowed → OVERSHOOT past k (seeds v7/v17 interior: 7 unique
        // pending births → 9; the exact number depends on neighborhood
        // geometry — the semantic point is sync > k).
        let mut synced = mk();
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        step_pca_sync(
            &cx,
            &mut synced,
            &params,
            &mut SplitMix64::new(11),
            &PcaGlobalFn::AliveCount,
            &budget_gate(k),
            &mut scratch,
        );
        let sync_alive = synced.data.iter().step_by(2).filter(|&&a| a > 0.5).count();
        assert!(sync_alive as f32 > k, "sync must overshoot k={k} (stale-count batch), got {sync_alive}");

        // ASYNC: the counter goes live — births stop EXACTLY at k.
        let mut asynced = mk();
        let returned = step_pca_async(
            &cx,
            &mut asynced,
            &params,
            &mut SplitMix64::new(11),
            &PcaGlobalFn::AliveCount,
            &budget_gate(k),
            &mut scratch,
        );
        let async_alive = asynced.data.iter().step_by(2).filter(|&&a| a > 0.5).count();
        assert_eq!(async_alive, k as usize, "async must land exactly on k={k}");
        assert_eq!(returned, k, "async returns the live post-pass counter");

        // Consistency: the live counter matches a fresh full evaluation.
        assert_eq!(
            PcaGlobalFn::AliveCount.evaluate(&cx, &asynced, &mut scratch),
            returned,
            "live counter drifted from the field"
        );
    }

    #[test]
    fn async_degenerates_to_sync_semantics_for_non_incremental_arms() {
        // Documented contract: non-incremental arms freeze the pre-tick value
        // — same results as sync (same seed).
        let cx = CellComplex::grid_2d(5, 5);
        let params = BirthDeathParams { dropout_prob: 0.0, ..BirthDeathParams::paper_defaults() };
        let gate = GlobalTargetGate { target: 1.0, stop_when: StopWhen::Below };
        let mk = || {
            let mut f = make_field(&cx, 2);
            seed_alive_morph(&mut f, &[0, 24], 1.0, 2);
            f
        };
        let mut a = mk();
        let mut s = mk();
        let mut scratch = PcaScratch::for_complex(&cx, 2);
        step_pca_async(
            &cx,
            &mut a,
            &params,
            &mut SplitMix64::new(9),
            &PcaGlobalFn::Betti0,
            &gate,
            &mut scratch,
        );
        step_pca_sync(
            &cx,
            &mut s,
            &params,
            &mut SplitMix64::new(9),
            &PcaGlobalFn::Betti0,
            &gate,
            &mut scratch,
        );
        assert_eq!(a.data, s.data, "async(Betti0) must equal sync(Betti0)");
    }

    #[test]
    fn step_pca_async_is_bit_identical_across_100_seeds() {
        // Phase 1 G1: same seed + fixed traversal → bit-identical final grid,
        // property-tested at ≥100 seeds.
        let cx = CellComplex::grid_2d(6, 6);
        let params = BirthDeathParams::paper_defaults();
        let gate = GlobalTargetGate { target: 4.0, stop_when: StopWhen::Above };
        for seed in 0..100u64 {
            let run = || {
                let mut field = make_field(&cx, 2);
                seed_alive_morph(&mut field, &[0, 21, 14], 1.0, 2);
                let mut scratch = PcaScratch::for_complex(&cx, 2);
                for _ in 0..3 {
                    step_pca_async(
                        &cx,
                        &mut field,
                        &params,
                        &mut SplitMix64::new(seed),
                        &PcaGlobalFn::AliveCount,
                        &gate,
                        &mut scratch,
                    );
                }
                field.data.clone()
            };
            assert_eq!(run(), run(), "seed {seed}: async must be bit-identical");
        }
    }

    #[test]
    fn connectivity_task_terminates_and_then_holds() {
        // The G2 seed task in miniature: two seeds must merge to b0 == 1 and
        // the global gate must then HOLD the line — the alive count freezes
        // after termination (no runaway filling, no die-off).
        let cx = CellComplex::grid_2d(6, 3);
        let params = BirthDeathParams { dropout_prob: 0.0, ..BirthDeathParams::paper_defaults() };
        let gate = GlobalTargetGate { target: 1.0, stop_when: StopWhen::Below };
        let mut field = make_field(&cx, 2);
        seed_alive_morph(&mut field, &[0, 17], 1.0, 2);
        let mut scratch = PcaScratch::for_complex(&cx, 2);

        // Phase A: grow until the pre-tick global reports one component.
        let mut ticks_to_merge = 0;
        while ticks_to_merge < 50 {
            let b0 = step_pca_sync(
                &cx,
                &mut field,
                &params,
                &mut SplitMix64::new(3),
                &PcaGlobalFn::Betti0,
                &gate,
                &mut scratch,
            );
            ticks_to_merge += 1;
            if b0 <= 1.0 {
                break;
            }
        }
        assert!(ticks_to_merge < 50, "seeds never merged");
        assert_eq!(
            PcaGlobalFn::Betti0.evaluate(&cx, &field, &mut scratch),
            1.0,
            "two seeds must end as one component"
        );

        // Phase B: the gate holds — births are refused, alive count frozen.
        let frozen: Vec<f32> = field.data.clone();
        for _ in 0..6 {
            step_pca_sync(
                &cx,
                &mut field,
                &params,
                &mut SplitMix64::new(3),
                &PcaGlobalFn::Betti0,
                &gate,
                &mut scratch,
            );
        }
        let alive_before = frozen.iter().step_by(2).filter(|&&a| a > 0.5).count();
        let alive_after = field.data.iter().step_by(2).filter(|&&a| a > 0.5).count();
        assert_eq!(alive_after, alive_before, "a tripped b0==1 gate must freeze the alive set");
        assert!(alive_before > 2, "the merge must have grown past the seeds");
    }
}
