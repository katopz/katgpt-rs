//! Strict-improvement local search over the folded-objective DAG.
//!
//! Move vocabulary (Phase 1): `Single` (move one object to another
//! container) + `Swap` (exchange two objects' containers). Acceptance is
//! first-improvement with the plan's window `obj_δ ≤ 0`: negative deltas
//! always accept; zero deltas only while a sideways budget lasts (default
//! 0 = strict improvement — the deterministic default posture).
//!
//! **Hot-container ordering** (the paper's measured decisive win, Fig. 6 —
//! kept per Plan 620): source containers are scanned hottest-first, where
//! heat ≈ the paper's node potentials approximated leaf-side: violation
//! mass `max(0, util − limit)` plus balance deviation `|util − mean|`
//! (integer mean; the per-dim total is invariant under reassignment).
//! Leaf-side heat is linear-cost, exact, and deterministic — the full
//! node-potential machinery is the documented Phase 2 refinement.
//!
//! Determinism discipline (G1): move generation iterates dense arrays in
//! fixed orders (containers via the seeded tie permutation, objects by
//! index); NO HashMap is iterated anywhere in the solve path; the RNG
//! touches only the tie permutation at construction. Same input + same
//! seed ⇒ byte-identical `Assignment`.
//!
//! The evaluate/apply split: `eval_*` computes a candidate root value via
//! delta evaluation without mutating committed state; `apply_*` commits
//! the completed evaluation (overlay → values) plus the assignment and
//! membership indexes.

use std::cmp::Reverse;
use std::time::Instant;

use crate::expr::DeltaScratch;
use crate::rng::SplitMix64;
use crate::specs::{self, Compiled};
use crate::types::{
    Assignment, ContainerId, InvalidProblem, Limits, Policy, Problem, Seed, Solution, Spec,
    StopReason,
};

pub mod explain;

/// A candidate move (diagnostic vocabulary; the scans return these).
/// `Triple` is the Phase 2 TRIPLE_LOOP rotation: `a`→container(b),
/// `b`→container(c), `c`→container(a) — a 3-cycle that escapes local minima
/// pairwise swaps cannot (upstream `TripleLoopMoveType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Single { object: u32, dst: ContainerId },
    Swap { a: u32, b: u32 },
    Triple { a: u32, b: u32, c: u32 },
}

pub struct Solver<'p> {
    problem: &'p Problem,
    compiled: Compiled,
    /// Cached node values (leaf slots hold current utilizations).
    values: Vec<i64>,
    assignment: Vec<ContainerId>,
    /// Container membership (dense; swap-remove maintained).
    members: Vec<Vec<u32>>,
    obj_pos: Vec<u32>,
    /// Scratch (allocated once; the hot loop allocates nothing — G4).
    scratch: DeltaScratch,
    leaf_deltas: Vec<(u32, i64)>,
    member_snapshot: Vec<u32>,
    heat: Vec<i64>,
    heat_order: Vec<ContainerId>,
    /// Seeded container permutation — the tie-break under equal heat.
    tie_perm: Vec<u32>,
    /// Per referenced dim: integer mean utilization (heat input; the
    /// per-dim total is assignment-invariant so this is computed once).
    ref_dim_mean: Vec<i64>,
    limits: Limits,
    sideways_remaining: u64,
    deadline: Option<Instant>,
    deadline_hit: bool,
    evals: u64,
    accepted: u64,
    sweeps: u64,
    stopped_by: StopReason,
    /// Immediate-reversal guard for sideways moves: the inverse of the
    /// last accepted δ=0 move. Accepting it again would ping-pong;
    /// cleared on every strict (δ<0) acceptance.
    blocked_inverse: Option<Move>,
    /// Set when a deterministic limit trips inside a scan (the run loop
    /// converts it into the stop reason).
    limit_hit: Option<StopReason>,
    /// Seeded RNG (the construction shuffle consumed a prefix; sampling
    /// draws continue the SAME stream — same seed ⇒ same draws, G1).
    rng: SplitMix64,
    /// Equivalence class per object (Plan 620 Phase 2, the paper's Alg 2
    /// object axis distilled to local search): objects with identical
    /// referenced-dim demands + initial container + affinity profile +
    /// GroupCount memberships are interchangeable — every move of one has
    /// the exact root delta of the same move of the other. Scans dedupe
    /// candidate objects by class (first-in-iteration-order representative).
    class_of: Vec<u32>,
    /// Per-class seen stamps for scan dedupe (generation-based, no
    /// clearing — the DeltaScratch pattern; G4-safe).
    class_stamp: Vec<u32>,
    class_gen: u32,
    /// Destination-draw stamps for sampled singles (same mechanism).
    dst_stamp: Vec<u32>,
    dst_gen: u32,
    /// Two extra member snapshots for the triple scan's concurrent loops
    /// (member_snapshot is the first).
    member_snapshot2: Vec<u32>,
    member_snapshot3: Vec<u32>,
    /// Remaining triple-rotation evaluations (armed from
    /// `Limits::triple_budget`).
    triple_remaining: u64,
}

impl<'p> Solver<'p> {
    /// Build a solver over a validated problem. Allocates all scratch
    /// here, never during `run` (G4).
    pub fn new(problem: &'p Problem, seed: Seed, limits: Limits) -> Result<Self, InvalidProblem> {
        problem.validate()?;
        let compiled = specs::compile(problem);
        let n = problem.num_objects;
        let b = problem.num_containers;

        // Integer mean per referenced dim (total demand / containers —
        // the total is invariant, so this never needs refreshing).
        let mut ref_dim_mean = Vec::with_capacity(compiled.referenced_dims.len());
        for &dim in &compiled.referenced_dims {
            let mut total: i128 = 0;
            for o in 0..n {
                total += problem.demands[o * problem.num_dims + dim] as i128;
            }
            ref_dim_mean.push((total / b as i128) as i64);
        }

        // Seeded tie permutation: identity shuffled once. Different seeds
        // explore equal-heat orders differently; the same seed always
        // produces the same permutation. The RNG field continues the SAME
        // stream for Phase 2 sampling (never re-seeded).
        let mut tie_perm: Vec<u32> = (0..b as u32).collect();
        let mut rng = SplitMix64::new(seed.0);
        rng.shuffle_u32(&mut tie_perm);

        let mut members: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut obj_pos = vec![0u32; n];
        for (o, &c) in problem.initial.iter().enumerate() {
            obj_pos[o] = members[c as usize].len() as u32;
            members[c as usize].push(o as u32);
        }

        // ── Equivalence classes (the paper's Alg 2 object axis, distilled
        // to local search): sort object ids by their move-delta signature;
        // equal-signature runs share a class. The signature is EXACTLY the
        // input a delta builder reads: demands over the referenced dims,
        // the initial container (moved-leaf deltas), the sorted affinity
        // profile (at-leaf deltas + their weights), and the sorted
        // GroupCount membership (which at-pairs exist). AvoidMoving pins
        // need no term — a pin changes no delta shape, only how the moved
        // leaf's change is INTERPRETED downstream (identical for members
        // of a class).
        let mut class_of = vec![0u32; n];
        {
            // Affinity profile per object (entries sorted by (container,
            // weight)); GroupCount membership per object (sorted spec idx).
            let mut aff_profile: Vec<Vec<(u32, i64)>> = vec![Vec::new(); n];
            let mut gc_member: Vec<Vec<u32>> = vec![Vec::new(); n];
            for (s, spec) in problem.specs.iter().enumerate() {
                match spec {
                    Spec::AssignmentAffinities { affinities } => {
                        for a in affinities {
                            aff_profile[a.object as usize].push((a.container, a.weight));
                        }
                    }
                    Spec::GroupCount { members, .. } => {
                        for &o in members {
                            gc_member[o as usize].push(s as u32);
                        }
                    }
                    _ => {}
                }
            }
            for p in aff_profile.iter_mut() {
                p.sort_unstable();
            }
            for g in gc_member.iter_mut() {
                g.sort_unstable();
                g.dedup();
            }
            let rd = &compiled.referenced_dims;
            let odims = problem.num_dims;
            let sig = |o: usize| -> Vec<i64> {
                let mut key = Vec::with_capacity(
                    rd.len() + 3 + aff_profile[o].len() * 2 + gc_member[o].len(),
                );
                for &d in rd {
                    key.push(problem.demands[o * odims + d]);
                }
                key.push(problem.initial[o] as i64);
                key.push(aff_profile[o].len() as i64);
                for &(c, w) in &aff_profile[o] {
                    key.push(c as i64);
                    key.push(w);
                }
                key.push(gc_member[o].len() as i64);
                for &g in &gc_member[o] {
                    key.push(g as i64);
                }
                key
            };
            let mut order: Vec<u32> = (0..n as u32).collect();
            order.sort_unstable_by_key(|&x| sig(x as usize));
            let mut class_id = 0u32;
            let mut prev: Option<Vec<i64>> = None;
            for &o in &order {
                let s = sig(o as usize);
                if let Some(p) = &prev
                    && *p != s
                {
                    class_id += 1;
                }
                class_of[o as usize] = class_id;
                prev = Some(s);
            }
        }

        let dag = &compiled.dag;
        let deadline = limits.time.map(|t| Instant::now() + t);
        let sideways_remaining = limits.sideways_budget;
        let triple_remaining = limits.triple_budget;
        // Pre-size scan scratch to worst case so the hot loop never grows
        // a buffer (G4): the widest container can hold every object.
        let scratch = dag.scratch();
        let leaf_deltas = Vec::with_capacity(2 * compiled.referenced_dims.len() + 2);
        let member_snapshot = Vec::with_capacity(n);
        let mut solver = Self {
            problem,
            scratch,
            values: vec![0; dag.num_nodes()],
            compiled,
            assignment: problem.initial.clone(),
            members,
            obj_pos,
            leaf_deltas,
            member_snapshot,
            heat: vec![0; b],
            heat_order: Vec::with_capacity(b),
            tie_perm,
            ref_dim_mean,
            limits,
            sideways_remaining,
            deadline,
            deadline_hit: false,
            evals: 0,
            accepted: 0,
            sweeps: 0,
            stopped_by: StopReason::NoImprovingMove,
            blocked_inverse: None,
            limit_hit: None,
            rng,
            class_stamp: vec![0; n],
            class_gen: 0,
            class_of,
            dst_stamp: vec![0; b],
            dst_gen: 0,
            member_snapshot2: Vec::with_capacity(n),
            member_snapshot3: Vec::with_capacity(n),
            triple_remaining,
        };
        // Seed the leaf values NOW (not only in `run`): the public eval /
        // explain seams are usable on a fresh solver without a stale-zero
        // footgun. `run` re-seeds (idempotent).
        solver.seed_leaf_values();
        Ok(solver)
    }

    // ── Diagnostic / test surface (also the Phase 2 `explain` seed) ──────

    /// Current assignment (borrowed).
    pub fn assignment(&self) -> &[ContainerId] {
        &self.assignment
    }

    /// Current folded objective.
    pub fn folded_objective(&self) -> i64 {
        self.values[self.compiled.dag.root as usize]
    }

    /// Current (violation_root, goal_sum) decomposition.
    pub fn objective_parts(&self) -> (i64, i64) {
        (
            self.values[self.compiled.violation_root as usize],
            self.values[self.compiled.goal_root as usize],
        )
    }

    /// Full recompute from the current assignment (delta-property oracle).
    pub fn recompute_all(&mut self) {
        self.seed_leaf_values();
    }

    /// Candidate root value for moving `object` to `dst` (no mutation).
    pub fn eval_single(&mut self, object: u32, dst: ContainerId) -> i64 {
        self.build_single_deltas(object, dst);
        self.compiled
            .dag
            .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        self.last_new_root()
    }

    /// Candidate root value for swapping `a` and `b` (no mutation).
    pub fn eval_swap(&mut self, a: u32, b: u32) -> i64 {
        self.build_swap_deltas(a, b);
        self.compiled
            .dag
            .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        self.last_new_root()
    }

    /// Candidate root value for the `Triple` rotation a→cb, b→cc, c→ca
    /// (no mutation).
    pub fn eval_triple(&mut self, a: u32, b: u32, c: u32) -> i64 {
        self.build_triple_deltas(a, b, c);
        self.compiled
            .dag
            .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        self.last_new_root()
    }

    /// Commit the last evaluated move (single or swap variant chosen by
    /// the caller — must match the last `eval_*` call).
    pub fn apply_single(&mut self, object: u32, dst: ContainerId) {
        self.compiled.dag.commit(&mut self.values, &self.scratch);
        let src = self.assignment[object as usize];
        debug_assert_ne!(src, dst);
        self.assignment[object as usize] = dst;
        // Membership: swap-remove from src, push to dst. `swap_remove`
        // returns the REMOVED element (== object); the element that was
        // LAST in src relocates into `pos` and needs its index fixed.
        let pos = self.obj_pos[object as usize] as usize;
        self.members[src as usize].swap_remove(pos);
        if let Some(&relocated) = self.members[src as usize].get(pos) {
            self.obj_pos[relocated as usize] = pos as u32;
        }
        self.obj_pos[object as usize] = self.members[dst as usize].len() as u32;
        self.members[dst as usize].push(object);
    }

    /// Commit a swap (must follow `eval_swap(a, b)`).
    pub fn apply_swap(&mut self, a: u32, b: u32) {
        self.compiled.dag.commit(&mut self.values, &self.scratch);
        let ca = self.assignment[a as usize];
        let cb = self.assignment[b as usize];
        self.assignment[a as usize] = cb;
        self.assignment[b as usize] = ca;
        // Membership: positions swap contents within their containers.
        let pa = self.obj_pos[a as usize] as usize;
        let pb = self.obj_pos[b as usize] as usize;
        self.members[ca as usize][pa] = b;
        self.members[cb as usize][pb] = a;
        self.obj_pos[a as usize] = pb as u32;
        self.obj_pos[b as usize] = pa as u32;
    }

    /// Commit the `Triple` rotation (must follow `eval_triple(a, b, c)`):
    /// one DAG commit, then three membership relocations (each a
    /// swap-remove + push — the relocated-element index fix mirrors
    /// `apply_single`).
    pub fn apply_triple(&mut self, a: u32, b: u32, c: u32) {
        self.compiled.dag.commit(&mut self.values, &self.scratch);
        let ca = self.assignment[a as usize];
        let cb = self.assignment[b as usize];
        let cc = self.assignment[c as usize];
        self.assignment[a as usize] = cb;
        self.assignment[b as usize] = cc;
        self.assignment[c as usize] = ca;
        for (o, from, to) in [(a, ca, cb), (b, cb, cc), (c, cc, ca)] {
            let pos = self.obj_pos[o as usize] as usize;
            self.members[from as usize].swap_remove(pos);
            if let Some(&relocated) = self.members[from as usize].get(pos) {
                self.obj_pos[relocated as usize] = pos as u32;
            }
            self.obj_pos[o as usize] = self.members[to as usize].len() as u32;
            self.members[to as usize].push(o);
        }
    }

    fn last_new_root(&self) -> i64 {
        self.scratch.last_new_root(self.compiled.dag.root)
    }

    // ── Delta construction ────────────────────────────────────────────────

    fn build_single_deltas(&mut self, object: u32, dst: ContainerId) {
        self.leaf_deltas.clear();
        let src = self.assignment[object as usize];
        debug_assert_ne!(src, dst, "single move to same container");
        let b = self.problem.num_containers;
        let odims = self.problem.num_dims;
        let o = object as usize;
        for (ref_idx, &dim) in self.compiled.referenced_dims.iter().enumerate() {
            let d = self.problem.demands[o * odims + dim];
            if d != 0 {
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * b + src as usize], -d));
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * b + dst as usize], d));
            }
        }
        if !self.compiled.moved_leaf.is_empty() {
            let init = self.problem.initial[o];
            let cur = (src != init) as i64;
            let new = (dst != init) as i64;
            if new != cur {
                self.leaf_deltas
                    .push((self.compiled.moved_leaf[o], new - cur));
            }
        }
        // At-leaves of the object: (o, b) flips only for b ∈ {src, dst}.
        let s = self.compiled.at_off[o] as usize;
        let e = self.compiled.at_off[o + 1] as usize;
        for i in s..e {
            let b_at = self.compiled.at_items[i];
            let delta = (dst == b_at) as i64 - (src == b_at) as i64;
            if delta != 0 {
                self.leaf_deltas.push((self.compiled.at_nodes[i], delta));
            }
        }
    }

    fn build_swap_deltas(&mut self, a: u32, b: u32) {
        self.leaf_deltas.clear();
        let ca = self.assignment[a as usize];
        let cb = self.assignment[b as usize];
        debug_assert_ne!(ca, cb, "swap within one container is a no-op");
        let nb = self.problem.num_containers;
        let odims = self.problem.num_dims;
        let (oa, ob) = (a as usize, b as usize);
        for (ref_idx, &dim) in self.compiled.referenced_dims.iter().enumerate() {
            let da = self.problem.demands[oa * odims + dim];
            let db = self.problem.demands[ob * odims + dim];
            if da != db {
                // ca loses da, gains db; cb loses db, gains da.
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * nb + ca as usize], db - da));
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * nb + cb as usize], da - db));
            }
        }
        if !self.compiled.moved_leaf.is_empty() {
            let (ia, ib) = (self.problem.initial[oa], self.problem.initial[ob]);
            let cur_a = (ca != ia) as i64;
            let new_a = (cb != ia) as i64;
            if new_a != cur_a {
                self.leaf_deltas
                    .push((self.compiled.moved_leaf[oa], new_a - cur_a));
            }
            let cur_b = (cb != ib) as i64;
            let new_b = (ca != ib) as i64;
            if new_b != cur_b {
                self.leaf_deltas
                    .push((self.compiled.moved_leaf[ob], new_b - cur_b));
            }
        }
        // At-leaves: a's flips for b' ∈ {ca, cb}, same for b.
        for (o, from, to) in [(a, ca, cb), (b, cb, ca)] {
            let oi = o as usize;
            let s = self.compiled.at_off[oi] as usize;
            let e = self.compiled.at_off[oi + 1] as usize;
            for i in s..e {
                let b_at = self.compiled.at_items[i];
                let delta = (to == b_at) as i64 - (from == b_at) as i64;
                if delta != 0 {
                    self.leaf_deltas.push((self.compiled.at_nodes[i], delta));
                }
            }
        }
    }

    /// Deltas for the TRIPLE_LOOP rotation: a→container(b), b→container(c),
    /// c→container(a) (three distinct containers, debug-asserted).
    fn build_triple_deltas(&mut self, a: u32, b: u32, c: u32) {
        self.leaf_deltas.clear();
        let ca = self.assignment[a as usize];
        let cb = self.assignment[b as usize];
        let cc = self.assignment[c as usize];
        debug_assert_ne!(ca, cb, "triple rotation needs 3 distinct containers");
        debug_assert_ne!(cb, cc, "triple rotation needs 3 distinct containers");
        debug_assert_ne!(ca, cc, "triple rotation needs 3 distinct containers");
        let nb = self.problem.num_containers;
        let odims = self.problem.num_dims;
        for (ref_idx, &dim) in self.compiled.referenced_dims.iter().enumerate() {
            let da = self.problem.demands[a as usize * odims + dim];
            let db = self.problem.demands[b as usize * odims + dim];
            let dc = self.problem.demands[c as usize * odims + dim];
            // NET per-container deltas: the rotation's three legs are
            // a: ca→cb, b: cb→cc, c: cc→ca, so each of the three DISTINCT
            // containers both loses and gains (ca: dc − da, cb: da − db,
            // cc: db − dc). One entry per leaf — pushing per-leg entries
            // would name the same leaf twice whenever consecutive legs
            // carry equal demands (the DAG rejects duplicate leaf deltas).
            for (container, net) in [(ca, dc - da), (cb, da - db), (cc, db - dc)] {
                if net != 0 {
                    self.leaf_deltas.push((
                        self.compiled.util_leaf[ref_idx * nb + container as usize],
                        net,
                    ));
                }
            }
        }
        if !self.compiled.moved_leaf.is_empty() {
            for (o, from, to) in [(a, ca, cb), (b, cb, cc), (c, cc, ca)] {
                let oi = o as usize;
                let init = self.problem.initial[oi];
                let cur = (from != init) as i64;
                let new = (to != init) as i64;
                if new != cur {
                    self.leaf_deltas
                        .push((self.compiled.moved_leaf[oi], new - cur));
                }
            }
        }
        for (o, from, to) in [(a, ca, cb), (b, cb, cc), (c, cc, ca)] {
            let oi = o as usize;
            let s = self.compiled.at_off[oi] as usize;
            let e = self.compiled.at_off[oi + 1] as usize;
            for i in s..e {
                let b_at = self.compiled.at_items[i];
                let delta = (to == b_at) as i64 - (from == b_at) as i64;
                if delta != 0 {
                    self.leaf_deltas.push((self.compiled.at_nodes[i], delta));
                }
            }
        }
    }

    // ── Leaf seeding ──────────────────────────────────────────────────────

    fn seed_leaf_values(&mut self) {
        let b = self.problem.num_containers;
        let odims = self.problem.num_dims;
        // Zero util leaves, then accumulate object demands.
        for i in 0..self.compiled.util_leaf.len() {
            self.values[self.compiled.util_leaf[i] as usize] = 0;
        }
        for o in 0..self.problem.num_objects {
            let c = self.assignment[o] as usize;
            for (ref_idx, &dim) in self.compiled.referenced_dims.iter().enumerate() {
                let d = self.problem.demands[o * odims + dim];
                if d != 0 {
                    let leaf = self.compiled.util_leaf[ref_idx * b + c] as usize;
                    self.values[leaf] += d;
                }
            }
        }
        if !self.compiled.moved_leaf.is_empty() {
            for o in 0..self.problem.num_objects {
                let moved = (self.assignment[o] != self.problem.initial[o]) as i64;
                self.values[self.compiled.moved_leaf[o] as usize] = moved;
            }
        }
        // At-leaves: 1 iff the object sits in the leaf's container.
        for o in 0..self.problem.num_objects {
            let cur = self.assignment[o];
            let s = self.compiled.at_off[o] as usize;
            let e = self.compiled.at_off[o + 1] as usize;
            for i in s..e {
                let c = self.compiled.at_items[i];
                self.values[self.compiled.at_nodes[i] as usize] = (cur == c) as i64;
            }
        }
        self.compiled.dag.eval_full(&mut self.values);
    }

    // ── Heat + ordering ───────────────────────────────────────────────────

    fn refresh_heat(&mut self) {
        let b = self.problem.num_containers;
        for h in self.heat.iter_mut() {
            *h = 0;
        }
        for spec in &self.problem.specs {
            match spec {
                Spec::Capacity { dim, limit, .. } => {
                    let ref_idx = self
                        .compiled
                        .referenced_dims
                        .binary_search(dim)
                        .expect("validated: referenced dim");
                    for c in 0..b {
                        let u = self.values[self.compiled.util_leaf[ref_idx * b + c] as usize];
                        if u > *limit {
                            self.heat[c] += u - *limit;
                        }
                    }
                }
                Spec::Balance { dim, .. } => {
                    let ref_idx = self
                        .compiled
                        .referenced_dims
                        .binary_search(dim)
                        .expect("validated: referenced dim");
                    let mean = self.ref_dim_mean[ref_idx];
                    for c in 0..b {
                        let u = self.values[self.compiled.util_leaf[ref_idx * b + c] as usize];
                        self.heat[c] += (u - mean).abs();
                    }
                }
                Spec::MinimizeMovement { .. } => {}
                Spec::GroupCount { members, .. } => {
                    // Count-deviation mass: |count(c) − mean_count| — the
                    // Balance arm over member counts (mean is
                    // assignment-invariant, deduped like the compile).
                    let mut group: Vec<u32> = members.clone();
                    group.sort_unstable();
                    group.dedup();
                    let mean = (group.len() / b.max(1)) as i64;
                    let mut count = vec![0i64; b];
                    for &o in &group {
                        count[self.assignment[o as usize] as usize] += 1;
                    }
                    for (c, &cnt) in count.iter().enumerate() {
                        self.heat[c] += (cnt - mean).abs();
                    }
                }
                Spec::AvoidMoving { objects, .. } => {
                    // A displaced pin is mass sitting in the WRONG place —
                    // drain its current container (the sum-repair term
                    // provides the pull back to the initial container).
                    for &o in objects {
                        let cur = self.assignment[o as usize];
                        if cur != self.problem.initial[o as usize] {
                            self.heat[cur as usize] += 1;
                        }
                    }
                }
                Spec::AssignmentAffinities { affinities } => {
                    // A mis-placed preference is drain mass at the object's
                    // current container, scaled by its weight.
                    for a in affinities {
                        let cur = self.assignment[a.object as usize];
                        if cur != a.container {
                            self.heat[cur as usize] += a.weight;
                        }
                    }
                }
            }
        }
        // Hot-first, seeded permutation breaking ties — the unique key
        // makes the unstable sort fully deterministic.
        self.heat_order.clear();
        self.heat_order.extend(0..b as u32);
        let tie = &self.tie_perm;
        let heat = &self.heat;
        self.heat_order
            .sort_unstable_by_key(|&c| (Reverse(heat[c as usize]), tie[c as usize]));
    }

    /// Read-only heat of ONE container (the `explain` surface: reports why
    /// a sideways candidate was gated without touching the heat buffers).
    pub fn heat_of_container(&self, c: ContainerId) -> i64 {
        let mut h = 0i64;
        let b = self.problem.num_containers;
        for spec in &self.problem.specs {
            match spec {
                Spec::Capacity { dim, limit, .. } => {
                    let ref_idx = self
                        .compiled
                        .referenced_dims
                        .binary_search(dim)
                        .expect("validated: referenced dim");
                    let u = self.values[self.compiled.util_leaf[ref_idx * b + c as usize] as usize];
                    if u > *limit {
                        h += u - *limit;
                    }
                }
                Spec::Balance { dim, .. } => {
                    let ref_idx = self
                        .compiled
                        .referenced_dims
                        .binary_search(dim)
                        .expect("validated: referenced dim");
                    let mean = self.ref_dim_mean[ref_idx];
                    let u = self.values[self.compiled.util_leaf[ref_idx * b + c as usize] as usize];
                    h += (u - mean).abs();
                }
                Spec::MinimizeMovement { .. } => {}
                Spec::GroupCount { members, .. } => {
                    let mut group: Vec<u32> = members.clone();
                    group.sort_unstable();
                    group.dedup();
                    let mean = (group.len() / b.max(1)) as i64;
                    let mut count = 0i64;
                    for &o in &group {
                        if self.assignment[o as usize] == c {
                            count += 1;
                        }
                    }
                    h += (count - mean).abs();
                }
                Spec::AvoidMoving { objects, .. } => {
                    for &o in objects {
                        if self.assignment[o as usize] == c
                            && self.assignment[o as usize] != self.problem.initial[o as usize]
                        {
                            h += 1;
                        }
                    }
                }
                Spec::AssignmentAffinities { affinities } => {
                    for a in affinities {
                        if self.assignment[a.object as usize] == c && c != a.container {
                            h += a.weight;
                        }
                    }
                }
            }
        }
        h
    }

    // ── Scans (first improvement) ─────────────────────────────────────────

    fn trip_limit(&mut self, reason: StopReason) {
        if self.limit_hit.is_none() {
            self.limit_hit = Some(reason);
        }
    }

    fn budget_check(&mut self) -> bool {
        // True = keep scanning.
        if self.evals >= self.limits.max_evaluations {
            self.trip_limit(StopReason::EvalLimit);
            return false;
        }
        if self.accepted >= self.limits.max_accepted_moves {
            self.trip_limit(StopReason::MoveLimit);
            return false;
        }
        if self.deadline_hit {
            self.trip_limit(StopReason::TimeLimit);
            return false;
        }
        true
    }

    fn time_tick(&mut self) {
        if !self.deadline_hit
            && let Some(dl) = self.deadline
            && Instant::now() >= dl
        {
            self.deadline_hit = true;
        }
    }

    /// Sideways (δ=0) acceptance predicate under the configured
    /// [`Policy`] (Plan 620 Phase 2 policy variants):
    /// - `Strict` — never;
    /// - `Default` (the Phase 1 measured posture) — while the sideways
    ///   budget lasts, the candidate is not the blocked reversal, and it
    ///   DRAINS an imbalanced container (source heat > 0) — sideways
    ///   shuffles between balanced containers only manufacture fresh
    ///   imbalance (measured: the walk otherwise burns its whole budget
    ///   drifting the overage around instead of closing it);
    /// - `Plateau` — the heat gate is dropped (walks between any
    ///   containers), the budget + reversal guard stay.
    fn sideways_allowed(&self, candidate: Move, src_heat: i64) -> bool {
        if self.limits.policy == Policy::Strict {
            return false;
        }
        if self.sideways_remaining == 0 || self.blocked_inverse == Some(candidate) {
            return false;
        }
        match self.limits.policy {
            Policy::Plateau => true,
            _ => src_heat > 0,
        }
    }

    /// Evaluate + accept-check ONE `Single` candidate (no mutation on
    /// rejection; consumes the sideways budget + arms the reversal guard on
    /// a δ=0 acceptance — the caller applies immediately). Shared by the
    /// exhaustive and sampled destination loops.
    fn try_single_candidate(
        &mut self,
        o: u32,
        src: ContainerId,
        src_heat: i64,
        dst: ContainerId,
    ) -> Option<Move> {
        if dst == src {
            return None;
        }
        if self.evals.is_multiple_of(1024) {
            self.time_tick();
            if self.deadline_hit {
                return None;
            }
            if !self.budget_check() {
                return None;
            }
        }
        self.evals += 1;
        self.build_single_deltas(o, dst);
        let new_root =
            self.compiled
                .dag
                .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        let delta = new_root - self.folded_objective();
        let candidate = Move::Single { object: o, dst };
        let strict = delta < 0;
        if strict || (delta == 0 && self.sideways_allowed(candidate, src_heat)) {
            if delta == 0 {
                self.sideways_remaining -= 1;
                // Block the exact reversal (o moving back to `src`) — the
                // ping-pong guard.
                self.blocked_inverse = Some(Move::Single {
                    object: o,
                    dst: src,
                });
            } else {
                self.blocked_inverse = None;
            }
            return Some(candidate);
        }
        None
    }

    /// Destination iteration for one candidate object: exhaustive
    /// `0..num_containers` by default, or `single_sample` seeded random
    /// draws when armed (SINGLE_RANDOM-style sampling; draws may repeat or
    /// hit the source — both skipped; deterministic by stream).
    fn for_each_single_dst(&mut self, o: u32, src: ContainerId, src_heat: i64) -> Option<Move> {
        let b = self.problem.num_containers;
        let sample = self.limits.single_sample;
        if sample == 0 || sample >= b {
            for dst in 0..b as u32 {
                if let Some(mv) = self.try_single_candidate(o, src, src_heat, dst) {
                    return Some(mv);
                }
                if self.limit_hit.is_some() || self.deadline_hit {
                    return None;
                }
            }
            return None;
        }
        // Sampled posture: a fresh generation per object keeps repeat-draw
        // dedupe local (the DeltaScratch stamp pattern; pre-allocated — G4).
        self.dst_gen = self.dst_gen.wrapping_add(1);
        if self.dst_gen == 0 {
            self.dst_stamp.iter_mut().for_each(|s| *s = 0);
            self.dst_gen = 1;
        }
        let dst_gen = self.dst_gen;
        for _ in 0..sample {
            let dst = self.rng.below(b) as u32;
            if dst == src || self.dst_stamp[dst as usize] == dst_gen {
                continue;
            }
            self.dst_stamp[dst as usize] = dst_gen;
            if let Some(mv) = self.try_single_candidate(o, src, src_heat, dst) {
                return Some(mv);
            }
            if self.limit_hit.is_some() || self.deadline_hit {
                return None;
            }
        }
        None
    }

    /// Scan for the first acceptable `Single` move (hot containers first;
    /// candidate objects deduped by equivalence class — the paper's Alg 2
    /// object axis: same class ⇒ same delta for every destination, so the
    /// first representative carries the whole class).
    fn scan_single(&mut self) -> Option<Move> {
        if !self.budget_check() {
            return None;
        }
        self.refresh_heat();
        for si in 0..self.heat_order.len() {
            self.time_tick();
            if self.deadline_hit {
                return None;
            }
            let src = self.heat_order[si];
            let src_heat = self.heat[src as usize];
            self.member_snapshot.clear();
            self.member_snapshot
                .extend_from_slice(&self.members[src as usize]);
            self.bump_class_gen();
            for mi in 0..self.member_snapshot.len() {
                let o = self.member_snapshot[mi];
                if !self.mark_class(o) {
                    continue; // an equivalent object was already scanned here
                }
                if let Some(mv) = self.for_each_single_dst(o, src, src_heat) {
                    return Some(mv);
                }
                if self.limit_hit.is_some() || self.deadline_hit {
                    return None;
                }
            }
        }
        None
    }

    /// Fresh generation for the class-dedupe stamps (the DeltaScratch
    /// wrap-clear pattern; no per-scan clearing — G4-safe).
    fn bump_class_gen(&mut self) {
        self.class_gen = self.class_gen.wrapping_add(1);
        if self.class_gen == 0 {
            self.class_stamp.iter_mut().for_each(|s| *s = 0);
            self.class_gen = 1;
        }
    }

    /// Mark object `o`'s equivalence class as seen in the current
    /// generation; false when an equivalent object was already marked
    /// (the caller skips it — same class ⇒ same delta).
    fn mark_class(&mut self, o: u32) -> bool {
        let class = self.class_of[o as usize] as usize;
        if self.class_stamp[class] == self.class_gen {
            return false;
        }
        self.class_stamp[class] = self.class_gen;
        true
    }

    /// Evaluate + accept-check ONE `Swap` candidate (mirror of
    /// `try_single_candidate`).
    fn try_swap_candidate(&mut self, a: u32, cand: u32, src_heat: i64) -> Option<Move> {
        if self.evals.is_multiple_of(1024) {
            self.time_tick();
            if self.deadline_hit {
                return None;
            }
            if !self.budget_check() {
                return None;
            }
        }
        self.evals += 1;
        self.build_swap_deltas(a, cand);
        let new_root =
            self.compiled
                .dag
                .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        let delta = new_root - self.folded_objective();
        let candidate = Move::Swap { a, b: cand };
        let strict = delta < 0;
        if strict || (delta == 0 && self.sideways_allowed(candidate, src_heat)) {
            if delta == 0 {
                self.sideways_remaining -= 1;
                // A swap is its own inverse.
                self.blocked_inverse = Some(candidate);
            } else {
                self.blocked_inverse = None;
            }
            return Some(candidate);
        }
        None
    }

    /// Scan for the first acceptable `Swap` move (hottest source
    /// containers only, capped by `max_swap_source_containers`; both the
    /// anchor and the candidate objects dedupe by equivalence class).
    fn scan_swap(&mut self) -> Option<Move> {
        if !self.budget_check() {
            return None;
        }
        self.refresh_heat();
        let sources = self
            .limits
            .max_swap_source_containers
            .min(self.problem.num_containers);
        let n = self.problem.num_objects as u32;
        for si in 0..sources {
            self.time_tick();
            if self.deadline_hit {
                return None;
            }
            let src = self.heat_order[si];
            let src_heat = self.heat[src as usize];
            self.member_snapshot.clear();
            self.member_snapshot
                .extend_from_slice(&self.members[src as usize]);
            self.bump_class_gen();
            for mi in 0..self.member_snapshot.len() {
                let a = self.member_snapshot[mi];
                if !self.mark_class(a) {
                    continue;
                }
                let ca = self.assignment[a as usize];
                // The candidate dedupe is PER-ANCHOR: fresh generation.
                self.bump_class_gen();
                for cand in 0..n {
                    if cand == a || self.assignment[cand as usize] == ca {
                        continue; // same container: no-op swap
                    }
                    if !self.mark_class(cand) {
                        continue;
                    }
                    if let Some(mv) = self.try_swap_candidate(a, cand, src_heat) {
                        return Some(mv);
                    }
                    if self.limit_hit.is_some() || self.deadline_hit {
                        return None;
                    }
                }
            }
        }
        None
    }

    // ── Phase 2 escape stages (triple rotation + KL chain) ──────────────

    /// TRIPLE_LOOP scan (Plan 620 Phase 2, upstream `TripleLoopMoveType`):
    /// evaluate 3-cycle rotations a→cb, b→cc, c→ca over the hot-container
    /// neighborhood and return the BEST strict improvement found within
    /// the remaining budget (`Limits::triple_budget`, total; evals also
    /// count against `max_evaluations`). Objects dedupe by equivalence
    /// class in all three loops. Strict-δ acceptance only: a δ=0 rotation
    /// churns three objects for nothing. Runs ONLY at a stall (the run
    /// loop's escape-stage placement — see the never-worse note there).
    fn scan_triple(&mut self) -> Option<Move> {
        if !self.budget_check() || self.triple_remaining == 0 {
            return None;
        }
        self.refresh_heat();
        let b = self.problem.num_containers;
        let sources = self.limits.max_swap_source_containers.min(b);
        let mut best: Option<(i64, Move)> = None;
        for hi in 0..sources {
            self.time_tick();
            if self.deadline_hit {
                break;
            }
            let hot = self.heat_order[hi];
            self.member_snapshot.clear();
            self.member_snapshot
                .extend_from_slice(&self.members[hot as usize]);
            self.bump_class_gen();
            for ai in 0..self.member_snapshot.len() {
                let a = self.member_snapshot[ai];
                if !self.mark_class(a) {
                    continue;
                }
                for ci in 0..sources {
                    let c1 = self.heat_order[ci];
                    if c1 == hot {
                        continue;
                    }
                    self.member_snapshot2.clear();
                    self.member_snapshot2
                        .extend_from_slice(&self.members[c1 as usize]);
                    self.bump_class_gen();
                    for bi in 0..self.member_snapshot2.len() {
                        let b_obj = self.member_snapshot2[bi];
                        if !self.mark_class(b_obj) {
                            continue;
                        }
                        for cj in 0..sources {
                            let c2 = self.heat_order[cj];
                            if c2 == hot || c2 == c1 {
                                continue;
                            }
                            self.member_snapshot3.clear();
                            self.member_snapshot3
                                .extend_from_slice(&self.members[c2 as usize]);
                            self.bump_class_gen();
                            for cci in 0..self.member_snapshot3.len() {
                                let c_obj = self.member_snapshot3[cci];
                                if !self.mark_class(c_obj) {
                                    continue;
                                }
                                if self.triple_remaining == 0
                                    || self.evals >= self.limits.max_evaluations
                                {
                                    self.trip_limit(StopReason::EvalLimit);
                                    return best.filter(|&(delta, _)| delta < 0).map(|(_, mv)| mv);
                                }
                                self.triple_remaining -= 1;
                                self.evals += 1;
                                self.build_triple_deltas(a, b_obj, c_obj);
                                let new_root = self.compiled.dag.eval_delta(
                                    &self.values,
                                    &self.leaf_deltas,
                                    &mut self.scratch,
                                );
                                let delta = new_root - self.folded_objective();
                                if delta < 0 && best.is_none_or(|(bd, _)| delta < bd) {
                                    best = Some((
                                        delta,
                                        Move::Triple {
                                            a,
                                            b: b_obj,
                                            c: c_obj,
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        best.filter(|&(delta, _)| delta < 0).map(|(_, mv)| mv)
    }

    /// KL_SEARCH escape stage (upstream `KLSearchMoveType`): per (hot,
    /// cold) container pair, a Kernighan–Lin chain — repeatedly pick the
    /// best-δ single move in either direction (worsening moves allowed,
    /// no object repeats), then commit the minimum-objective improving
    /// prefix and roll back the rest. Commits at most `max_kl_passes`
    /// chains per invocation; true when any committed. Runs only when the
    /// single/swap/triple scans find nothing (the escape hatch).
    fn kl_escape(&mut self) -> bool {
        if !self.budget_check() || self.limits.max_kl_passes == 0 {
            return false;
        }
        self.refresh_heat();
        let sources = self
            .limits
            .max_swap_source_containers
            .min(self.problem.num_containers);
        let mut committed = 0u32;
        for hi in 0..sources {
            for ci in 0..sources {
                if committed >= self.limits.max_kl_passes {
                    return true;
                }
                if !self.budget_check() {
                    return committed > 0;
                }
                let hot = self.heat_order[hi];
                let cold = self.heat_order[ci];
                if hot == cold {
                    continue;
                }
                if self.kl_pair(hot, cold) {
                    committed += 1;
                }
            }
        }
        committed > 0
    }

    /// One KL chain over the `(hot, cold)` pair; true when an improving
    /// prefix committed. Tentative chain steps apply to the LIVE state and
    /// roll back from the base snapshots (escape-stage allocations — the
    /// hot-loop G4 gate covers the default posture, where KL is off).
    fn kl_pair(&mut self, hot: ContainerId, cold: ContainerId) -> bool {
        let base_assignment = self.assignment.clone();
        let base_values = self.values.clone();
        let mut chain: Vec<(u32, ContainerId)> = Vec::new();
        let mut cum = 0i64;
        let mut best_prefix: Option<(usize, i64)> = None; // (len, cumulative)
        self.bump_class_gen();
        let tried_gen = self.class_gen;
        while chain.len() < self.limits.max_kl_chain {
            // Best next step: singles in either direction over the pair's
            // members, untried objects only, first-minimum wins ties.
            let mut best: Option<(i64, u32, ContainerId)> = None;
            for (from, to) in [(hot, cold), (cold, hot)] {
                self.member_snapshot.clear();
                self.member_snapshot
                    .extend_from_slice(&self.members[from as usize]);
                for mi in 0..self.member_snapshot.len() {
                    let o = self.member_snapshot[mi];
                    if self.class_stamp[o as usize] == tried_gen {
                        continue;
                    }
                    if !self.budget_check() {
                        break;
                    }
                    self.evals += 1;
                    self.build_single_deltas(o, to);
                    let new_root = self.compiled.dag.eval_delta(
                        &self.values,
                        &self.leaf_deltas,
                        &mut self.scratch,
                    );
                    let delta = new_root - self.folded_objective();
                    if best.is_none_or(|(bd, _, _)| delta < bd) {
                        best = Some((delta, o, from));
                    }
                }
                if !self.budget_check() {
                    break;
                }
            }
            if !self.budget_check() {
                break;
            }
            let Some((delta, o, from)) = best else {
                break;
            };
            // Tentatively apply the chosen step (the picked candidate's
            // evaluation was overwritten by later candidates — re-eval,
            // then commit).
            let to = if from == hot { cold } else { hot };
            self.class_stamp[o as usize] = tried_gen;
            self.evals += 1;
            self.build_single_deltas(o, to);
            self.compiled
                .dag
                .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
            self.apply_single(o, to);
            chain.push((o, from));
            cum += delta;
            if best_prefix.is_none_or(|(_, bc)| cum < bc) {
                best_prefix = Some((chain.len(), cum));
            }
        }
        let Some((k, best_cum)) = best_prefix else {
            self.kl_rollback(&base_assignment, &base_values);
            return false;
        };
        if best_cum >= 0 {
            self.kl_rollback(&base_assignment, &base_values);
            return false;
        }
        // Roll back the whole chain, then re-apply the improving prefix
        // (identical deltas — the base state is restored bit-for-bit).
        let prefix: Vec<(u32, ContainerId)> = chain[..k].to_vec();
        self.kl_rollback(&base_assignment, &base_values);
        for (o, from) in prefix {
            let to = if from == hot { cold } else { hot };
            self.evals += 1;
            self.build_single_deltas(o, to);
            self.compiled
                .dag
                .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
            self.apply_single(o, to);
            self.accepted += 1;
        }
        self.blocked_inverse = None;
        true
    }

    /// Restore the base state after a failed/uncommitted KL chain:
    /// assignment + values from the snapshots, membership rebuilt from the
    /// assignment (O(n + b)).
    fn kl_rollback(&mut self, assignment: &[ContainerId], values: &[i64]) {
        self.assignment.clear();
        self.assignment.extend_from_slice(assignment);
        for m in self.members.iter_mut() {
            m.clear();
        }
        for (o, &c) in self.assignment.iter().enumerate() {
            self.obj_pos[o] = self.members[c as usize].len() as u32;
            self.members[c as usize].push(o as u32);
        }
        self.values.clear();
        self.values.extend_from_slice(values);
    }

    // ── Run loop ──────────────────────────────────────────────────────

    /// Execute the search. Consumes the solver (the Solution owns the
    /// final assignment).
    pub fn run(mut self) -> Solution {
        self.seed_leaf_values();
        loop {
            if let Some(reason) = self.limit_hit {
                self.stopped_by = reason;
                break;
            }
            if self.sweeps >= self.limits.max_sweeps {
                self.stopped_by = StopReason::SweepLimit;
                break;
            }
            self.sweeps += 1;
            let mut found = false;
            while let Some(mv) = self.scan_single() {
                found = true;
                match mv {
                    Move::Single { object, dst } => self.apply_single(object, dst),
                    _ => unreachable!("scan_single yields singles"),
                }
                self.accepted += 1;
                if self.limit_hit.is_some() {
                    break;
                }
            }
            if self.limit_hit.is_some() {
                continue; // converted to a stop reason at loop head
            }
            while let Some(mv) = self.scan_swap() {
                found = true;
                match mv {
                    Move::Swap { a, b } => self.apply_swap(a, b),
                    _ => unreachable!("scan_swap yields swaps"),
                }
                self.accepted += 1;
                if self.limit_hit.is_some() {
                    break;
                }
            }
            if self.limit_hit.is_some() {
                continue;
            }
            // TRIPLE_LOOP escape (Phase 2): armed only by a non-zero
            // `triple_budget`, and only at a stall — the escape-stage
            // placement that keeps the never-worse property (the armed run
            // shares the plain run's trajectory until the first stall, and
            // every escape accept is δ ≤ 0 from there).
            if !found && self.limits.triple_budget > 0 {
                while let Some(mv) = self.scan_triple() {
                    found = true;
                    match mv {
                        Move::Triple { a, b, c } => {
                            // scan_triple scans PAST the best candidate
                            // (best-of-budget semantics), so the scratch
                            // overlay belongs to the last-evaluated
                            // candidate — re-evaluate the chosen one
                            // before the commit.
                            self.eval_triple(a, b, c);
                            self.apply_triple(a, b, c);
                        }
                        _ => unreachable!("scan_triple yields triples"),
                    }
                    self.accepted += 1;
                    if self.limit_hit.is_some() {
                        break;
                    }
                }
                if self.limit_hit.is_some() {
                    continue;
                }
            }
            // KL_SEARCH escape (Phase 2): the last stage before declaring
            // NoImprovingMove — big-neighborhood chains where pairwise
            // (and triple) moves stall.
            if !found && self.limits.max_kl_passes > 0 && self.kl_escape() {
                found = true;
            }
            if !found {
                self.stopped_by = StopReason::NoImprovingMove;
                break;
            }
        }
        let (violation_root, goal_sum) = self.objective_parts();
        Solution {
            folded_objective: self.folded_objective(),
            violation_root,
            goal_sum,
            assignment: Assignment {
                containers: self.assignment,
            },
            moves_evaluated: self.evals,
            moves_accepted: self.accepted,
            sweeps: self.sweeps,
            stopped_by: self.stopped_by,
        }
    }
}

/// One-call API.
pub fn solve(problem: &Problem, seed: Seed, limits: &Limits) -> Result<Solution, InvalidProblem> {
    Ok(Solver::new(problem, seed, limits.clone())?.run())
}

/// Folded-objective oracle: compile + evaluate an arbitrary assignment
/// (full recompute, no search). Returns `(folded, violation_root,
/// goal_sum)`. The G1/G2 tests' ground truth and the brute-force
/// comparator.
pub fn eval_assignment(
    problem: &Problem,
    assignment: &[ContainerId],
) -> Result<(i64, i64, i64), InvalidProblem> {
    problem.validate()?;
    if assignment.len() != problem.num_objects {
        return Err(InvalidProblem::InitialLen {
            expected: problem.num_objects,
            got: assignment.len(),
        });
    }
    let compiled = specs::compile(problem);
    let mut values = vec![0i64; compiled.dag.num_nodes()];
    let b = problem.num_containers;
    let odims = problem.num_dims;
    for (o, &container) in assignment.iter().enumerate() {
        let c = container as usize;
        for (ref_idx, &dim) in compiled.referenced_dims.iter().enumerate() {
            let d = problem.demands[o * odims + dim];
            if d != 0 {
                values[compiled.util_leaf[ref_idx * b + c] as usize] += d;
            }
        }
    }
    if !compiled.moved_leaf.is_empty() {
        for (o, &container) in assignment.iter().enumerate() {
            values[compiled.moved_leaf[o] as usize] = (container != problem.initial[o]) as i64;
        }
    }
    for (o, &container) in assignment.iter().enumerate() {
        let s = compiled.at_off[o] as usize;
        let e = compiled.at_off[o + 1] as usize;
        for i in s..e {
            values[compiled.at_nodes[i] as usize] = (container == compiled.at_items[i]) as i64;
        }
    }
    compiled.dag.eval_full(&mut values);
    Ok((
        values[compiled.dag.root as usize],
        values[compiled.violation_root as usize],
        values[compiled.goal_root as usize],
    ))
}
