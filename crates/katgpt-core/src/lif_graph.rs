//! `lif_graph` — signed-graph LIF reservoir: event-driven sparse propagation.
//!
//! The fly-connectome architecture class (Issue 763 / riir-ai Research 379):
//! a **fixed sparse sign-constrained graph** running current-based LIF (leaky
//! integrate-and-fire) dynamics, with a **closed-form ridge readout** (no
//! training loop, no gradient descent — consumes `linalg::ridge_solve`, the
//! KARC precedent).
//!
//! # Model (Shiu et al. Nature 2024, current-based LIF)
//!
//! `dv/dt = (v_0 − v + g)/t_mbr`, `dg/dt = −g/tau_syn`, integrated exactly
//! over one `dt` (the RuVector trick — decays precomputed once, 3 muls per
//! active neuron per tick):
//!
//! ```text
//! v ← v_0 + (v − v_0)·a_m + g·c_gs     a_m = exp(−dt/t_mbr)
//! g ← g·a_s                             a_s = exp(−dt/tau_syn)
//! c_gs = tau_syn/(tau_syn − t_mbr)·(a_s − a_m)
//! ```
//!
//! Threshold `v ≥ v_th` (outside refractory) → **spike**: `v ← v_rst`,
//! refractory counter set; during refractory `v` is clamped to `v_rst` while
//! `g` keeps decaying. Spikes propagate along signed CSR edges after a fixed
//! tick delay (timing-wheel ring). Canonical constants: [`LifParams::shiu`].
//!
//! **Weight units are PSP millivolts** (the Shiu convention `w = 0.275 mV ×
//! signed_count`): delivery converts to synaptic drive via
//! `g += w·(t_mbr/tau_syn)` — the scale under which one unit of weight
//! produces one mV of total postsynaptic depolarization. In the Shiu regime
//! a single excitatory synapse moves `v` by 0.275 mV against a 7 mV
//! threshold gap: firing is **coincidence detection** (~26 synchronous
//! synapses), exactly as in the whole-fly model.
//!
//! # Event-driven with EXACT dense parity
//!
//! A node is **quiescent** iff its next dense update is the identity:
//! `refrac == 0 ∧ g == 0.0 ∧ (v_0 + (v−v_0)·a_m + g·c_gs) == v` bitwise.
//! (`v == v_0` always qualifies, and `v_rst == v_0` in the Shiu constants,
//! so a fired node returns to quiescence when its refractory ends with `g`
//! decayed to zero.) The event-driven [`LifReservoir::step`] visits only the
//! active set; the reference [`LifReservoir::step_dense`] visits every node
//! through the same shared [`LifReservoir::update_node`] — so the two paths
//! produce **bit-identical trajectories** (pinned by `tests/lif_graph_g1.rs`).
//! This differs from RuVector's tolerance-based quiescence dropout, which is
//! approximate by design; here the skip criterion IS the fixed-point test.
//!
//! Synaptic tails are snapped to exact zero below `g_floor` (default 1e-5
//! mV·(t_mbr/tau_syn), ~0.0002% of a single-synapse drive) — applied
//! identically in both paths, so parity is preserved while the active set is
//! not pinned open by subnormal tails (which would otherwise decay for
//! ~4300 ticks before flushing to zero on their own).
//!
//! # Performance shape (Bench 760)
//!
//! Per tick: O(|active|) node updates + O(Σ out-degree of spikers) event
//! writes — never O(N²), never a full edge scan. The classic reservoir
//! baseline (dense `W·spike_vector` matvec) is O(N²)/tick regardless of
//! activity; the CSR full-scan arm ([`LifReservoir::step_dense`]) is O(N).
//!
//! # Substrate check (substrate-first skill, 2026-09-13 — Issue 763 T1)
//!
//! - LIF/membrane/refractory dynamics: **no substrate** anywhere in the
//!   workspace (every "spike" hit is metaphorical — see the skill's run log).
//! - Ridge readout: **consumes** [`crate::linalg::ridge_solve_direct_f64`]
//!   (the KARC precedent) — never re-implemented.
//! - CSR: `riir_engine::engram_runtime::EngramKgCsr` is downstream and
//!   KG-specific (wrong direction — this crate is upstream of it);
//!   `dirichlet.rs` uses unweighted edge pairs. [`SignedAdjacency`] is the
//!   new weighted signed-CSR home.
//!
//! # Data-agnostic
//!
//! Callers supply any signed sparse graph. NO connectome dataset ships here
//! (FlyWire/MaleCNS licensing; fixtures are synthetic). Any training loop is
//! out of scope (ridge is closed-form; gradient-free by construction).
//!
//! lif_graph — signed-graph LIF reservoir: event-driven sparse propagation
//! (Issue 763 / riir-ai Research 379, the fly-connectome class). Current-based
//! LIF (Shiu et al. Nature 2024 canonical constants) on a fixed signed CSR
//! adjacency, exact exponential integration (3 muls per active neuron per
//! tick), timing-wheel spike delays, and an EXACT-parity event-driven active
//! set (quiescence = bitwise fixed point of the leak map — the dense update
//! of a skipped node IS the identity). Closed-form ridge readout consumes
//! `linalg::ridge_solve_direct_f64` (the KARC precedent). Data-agnostic:
//! callers supply any signed sparse graph; no connectome datasets ship here.
//! Opt-in — POC landed with GOAT G1–G4 PASS (Bench 760); promotion needs a
//! consumer (the riir-ai per-archetype circuit shard path, Research 379 §7).

use crate::linalg::ridge_solve_direct_f64;

// ─── Fixture RNG (tests + benches only; the house xorshift64* pattern) ──────
//
// The `interpolation_geometry::FixtureRng` precedent: a pub deterministic RNG
// for fixture generation, scoped to this module, NOT used in the tick path —
// the reservoir dynamics are fully deterministic.

/// Deterministic xorshift64* RNG for fixture generation (tests + benches).
/// NOT cryptographically secure; NOT used in the tick path.
#[derive(Clone)]
pub struct FixtureRng(pub u64);

impl FixtureRng {
    /// New RNG with the given seed. Seed 0 is remapped to 1.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    /// Next raw 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform `[0, 1)` draw.
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        let bits = ((self.next_u64() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }

    /// Uniform integer in `[0, n)`.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

// ─── Parameters ─────────────────────────────────────────────────────────────

/// Current-based LIF parameters with precomputed exact-integration decays.
///
/// Defaults are the Shiu et al. (Nature 2024) whole-fly canonical constants;
/// all times are in milliseconds, all potentials in millivolts.
#[derive(Clone, Debug)]
pub struct LifParams {
    /// Rest / baseline potential `v_0` (mV). Equal to `v_rst` in the Shiu
    /// constants — the property that makes the reset state quiescent.
    pub v0: f32,
    /// Spike threshold (mV).
    pub v_th: f32,
    /// Reset potential (mV).
    pub v_rst: f32,
    /// Membrane time constant (ms).
    pub t_mbr: f32,
    /// Synaptic decay time constant (ms).
    pub tau_syn: f32,
    /// Integration step (ms). Must satisfy `dt ≤ min(t_mbr, tau_syn)/2`.
    pub dt: f32,
    /// Refractory period, in ticks (`ceil(2.2/dt)` for the Shiu constants).
    pub refrac_ticks: u32,
    /// Spike propagation delay, in ticks. Must be ≥ 1 (no same-tick cascade).
    pub delay_ticks: u32,
    /// Synaptic-drive floor: `|g|` below this is snapped to exact `0.0`
    /// (identically in both step paths — see the module doc). In g units
    /// (PSP mV × `t_mbr/tau_syn`).
    pub g_floor: f32,
    // Precomputed once, consumed every tick (the 3-mul hot loop).
    a_m: f32,
    a_s: f32,
    c_gs: f32,
}

impl LifParams {
    /// Shiu et al. canonical constants at `dt = 0.1` ms:
    /// v0 = v_rst = −52 mV, v_th = −45 mV, t_mbr = 20 ms, tau_syn = 5 ms,
    /// refrac 2.2 ms (22 ticks), delay 1.8 ms (18 ticks).
    pub fn shiu() -> Self {
        Self::with_dt(0.1)
    }

    /// Shiu constants at a different `dt`; refrac/delay ticks are re-derived.
    ///
    /// # Panics
    /// If `dt` is not in `(0, min(t_mbr, tau_syn)/2]`.
    pub fn with_dt(dt: f32) -> Self {
        let t_mbr = 20.0_f32;
        let tau_syn = 5.0_f32;
        assert!(
            dt > 0.0 && 2.0 * dt <= t_mbr.min(tau_syn),
            "dt must be in (0, {}]",
            t_mbr.min(tau_syn) / 2.0
        );
        let mut p = Self {
            v0: -52.0,
            v_th: -45.0,
            v_rst: -52.0,
            t_mbr,
            tau_syn,
            dt,
            refrac_ticks: (2.2 / dt).ceil().max(1.0) as u32,
            delay_ticks: (1.8 / dt).ceil().max(1.0) as u32,
            g_floor: 1e-5,
            a_m: 0.0,
            a_s: 0.0,
            c_gs: 0.0,
        };
        p.recompute();
        p
    }

    /// Re-derive the precomputed decays after mutating the time constants.
    ///
    /// # Panics
    /// If `tau_syn == t_mbr` (the exact-integration form requires them to
    /// differ).
    pub fn recompute(&mut self) {
        assert!(
            self.tau_syn != self.t_mbr,
            "exact integration requires tau_syn != t_mbr"
        );
        self.a_m = (-self.dt / self.t_mbr).exp();
        self.a_s = (-self.dt / self.tau_syn).exp();
        self.c_gs = (self.tau_syn / (self.tau_syn - self.t_mbr)) * (self.a_s - self.a_m);
    }

    /// The precomputed decay coefficients `(a_m, a_s, c_gs)`.
    pub fn decays(&self) -> (f32, f32, f32) {
        (self.a_m, self.a_s, self.c_gs)
    }

    /// PSP-mV → synaptic-drive scale (`t_mbr/tau_syn` = 4.0 for Shiu).
    #[inline]
    pub fn w_scale(&self) -> f32 {
        self.t_mbr / self.tau_syn
    }
}

impl Default for LifParams {
    fn default() -> Self {
        Self::shiu()
    }
}

// ─── Signed CSR adjacency ───────────────────────────────────────────────────

/// Signed sparse adjacency in CSR form (row = source node; the edge sign is
/// pre-folded into the f32 weight — the RuVector layout).
///
/// Weights are **PSP millivolts** (see the module doc). Committable by
/// consumers: hash `offsets ‖ targets ‖ weights.to_bits()` with BLAKE3 —
/// the layout is fixed for the graph's lifetime.
#[derive(Clone, Debug)]
pub struct SignedAdjacency {
    n: u32,
    /// Row offsets, length `n + 1` (`offsets[0] == 0`).
    offsets: Box<[u32]>,
    /// Edge targets, parallel to `weights`.
    targets: Box<[u32]>,
    /// Signed weights (positive = excitatory, negative = inhibitory).
    weights: Box<[f32]>,
}

impl SignedAdjacency {
    /// Build from `(source, target, weight)` edges. Duplicate edges stack
    /// (synapse counts fold into weights upstream: `w = 0.275 mV × count`).
    pub fn from_edges(n: u32, edges: &[(u32, u32, f32)]) -> Self {
        let mut offsets = vec![0u32; n as usize + 1];
        for &(s, _, _) in edges {
            assert!(s < n, "source {s} out of bounds (n={n})");
            offsets[s as usize + 1] += 1;
        }
        for i in 0..n as usize {
            offsets[i + 1] += offsets[i];
        }
        // Counting sort by source (stable in input order within a row).
        let mut cursor = offsets.clone();
        let mut targets = vec![u32::MAX; edges.len()];
        let mut weights = vec![0.0_f32; edges.len()];
        for &(s, t, w) in edges {
            assert!(t < n, "target {t} out of bounds (n={n})");
            let slot = cursor[s as usize] as usize;
            targets[slot] = t;
            weights[slot] = w;
            cursor[s as usize] += 1;
        }
        Self {
            n,
            offsets: offsets.into_boxed_slice(),
            targets: targets.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
        }
    }

    /// ER-matched control (the FlyDoom null ladder's bottom rung): `n_edges`
    /// uniform-random directed edges at the same N and E as the real graph,
    /// excitatory `w_exc` with probability `1 − p_inh`, inhibitory `−w_inh`
    /// otherwise — "any sparse signed reservoir would do".
    pub fn er_matched(
        n: u32,
        n_edges: usize,
        p_inh: f32,
        w_exc: f32,
        w_inh: f32,
        seed: u64,
    ) -> Self {
        let mut rng = FixtureRng::new(seed);
        let mut edges = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            let s = rng.below(n as u64) as u32;
            let t = rng.below(n as u64) as u32;
            let w = if rng.uniform() < p_inh { -w_inh } else { w_exc };
            edges.push((s, t, w));
        }
        Self::from_edges(n, &edges)
    }

    /// Maslov–Sneppen degree-preserving control: `swaps` double-edge swaps on
    /// a clone. Two edges `(u1→t1, u2→t2)` are rewired to `(u1→t2, u2→t1)`
    /// when the swap introduces no self-loop and no duplicate edge —
    /// preserving every in/out degree while destroying higher structure.
    /// Weights travel with their source (sign is a source-neurotransmitter
    /// property in the connectome model).
    pub fn maslov_sneppen(&self, swaps: usize, seed: u64) -> Self {
        let mut adj = self.clone();
        let mut rng = FixtureRng::new(seed);
        let n_edges = adj.targets.len();
        if n_edges < 2 {
            return adj;
        }
        let mut done = 0usize;
        while done < swaps {
            let e1 = rng.below(n_edges as u64) as usize;
            let e2 = rng.below(n_edges as u64) as usize;
            if e1 == e2 {
                continue;
            }
            let (u1, t1) = (adj.row_of(e1), adj.targets[e1]);
            let (u2, t2) = (adj.row_of(e2), adj.targets[e2]);
            if t1 == u2 || t2 == u1 {
                continue;
            }
            if adj.row_contains(u1, t2) || adj.row_contains(u2, t1) {
                continue;
            }
            adj.targets[e1] = t2;
            adj.targets[e2] = t1;
            done += 1;
        }
        adj
    }

    #[inline]
    fn row_of(&self, edge: usize) -> u32 {
        // offsets is non-decreasing; binary search for the row owning `edge`.
        let offsets = &self.offsets;
        let mut lo = 0usize;
        let mut hi = self.n as usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if offsets[mid + 1] as usize <= edge {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo as u32
    }

    #[inline]
    fn row_contains(&self, row: u32, target: u32) -> bool {
        let (a, b) = self.row_range(row);
        self.targets[a..b].contains(&target)
    }

    /// `[start, end)` flat-index range of `row`'s edges.
    #[inline]
    pub fn row_range(&self, row: u32) -> (usize, usize) {
        (
            self.offsets[row as usize] as usize,
            self.offsets[row as usize + 1] as usize,
        )
    }

    /// Number of nodes.
    #[inline]
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Number of edges.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.targets.len()
    }

    /// Row offsets (length `n + 1`).
    #[inline]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Edge targets (parallel to [`Self::weights`]).
    #[inline]
    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    /// Signed edge weights (parallel to [`Self::targets`]).
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

// ─── The reservoir ──────────────────────────────────────────────────────────

/// Event-driven signed-graph LIF reservoir over a fixed [`SignedAdjacency`].
///
/// All per-tick state is preallocated at construction; the tick loop is
/// allocation-free in steady state (ring buckets grow only on their first
/// peak — pinned by `tests/lif_graph_g4_alloc.rs`).
pub struct LifReservoir {
    adj: SignedAdjacency,
    params: LifParams,
    // Per-node state.
    v: Box<[f32]>,
    g: Box<[f32]>,
    refrac: Box<[u32]>,
    // Timing-wheel delay ring (fixed length = delay_ticks).
    ring: Box<[Vec<(u32, f32)>]>,
    ring_head: usize,
    // Active-set bookkeeping (event path). `active` holds the live set;
    // `active_next` is the scratch the next set is built into.
    active: Vec<u32>,
    active_next: Vec<u32>,
    is_active: Box<[bool]>,
    // This tick's spikers (ascending in the dense path, visit-order in the
    // event path — spikes are order-independent: effects deferred ≥ 1 tick).
    spikes: Vec<u32>,
}

impl LifReservoir {
    /// Construct with all nodes at rest (`v = v0`, `g = 0`, refractory 0 —
    /// the quiescent fixed point).
    pub fn new(adj: SignedAdjacency, params: LifParams) -> Self {
        let n = adj.n() as usize;
        let ring_len = params.delay_ticks.max(1) as usize;
        let v0 = params.v0;
        Self {
            adj,
            params,
            v: vec![v0; n].into_boxed_slice(),
            g: vec![0.0; n].into_boxed_slice(),
            refrac: vec![0; n].into_boxed_slice(),
            ring: (0..ring_len).map(|_| Vec::new()).collect(),
            ring_head: 0,
            active: Vec::with_capacity(n),
            active_next: Vec::with_capacity(n),
            is_active: vec![false; n].into_boxed_slice(),
            spikes: Vec::with_capacity(n),
        }
    }

    /// The signed CSR adjacency (fixed for the reservoir's lifetime).
    pub fn adjacency(&self) -> &SignedAdjacency {
        &self.adj
    }

    /// The parameters (with precomputed decays).
    pub fn params(&self) -> &LifParams {
        &self.params
    }

    /// Per-node state snapshot `(v, g, refrac)`.
    pub fn state(&self) -> (&[f32], &[f32], &[u32]) {
        (&self.v, &self.g, &self.refrac)
    }

    /// Number of nodes.
    #[inline]
    pub fn n(&self) -> u32 {
        self.adj.n()
    }

    /// Current active-set size (event path bookkeeping; diagnostic).
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Reset to the all-quiescent rest state (capacities retained).
    pub fn reset(&mut self) {
        self.v.fill(self.params.v0);
        self.g.fill(0.0);
        self.refrac.fill(0);
        for b in self.ring.iter_mut() {
            b.clear();
        }
        self.active.clear();
        self.active_next.clear();
        self.is_active.fill(false);
        self.spikes.clear();
        self.ring_head = 0;
    }

    /// Inject external current into `node` (PSP mV units, delivered
    /// immediately; marks the node active — the only way a quiescent node
    /// wakes). Identical in both step paths; call between `step`s.
    pub fn inject(&mut self, node: u32, weight_psp: f32) {
        assert!(node < self.n(), "node {node} out of bounds");
        self.g[node as usize] += weight_psp * self.params.w_scale();
        self.wake(node);
    }

    #[inline]
    fn wake(&mut self, node: u32) {
        let i = node as usize;
        if !self.is_active[i] {
            self.is_active[i] = true;
            self.active.push(node);
        }
    }

    /// One event-driven tick: deliver due events, update the active set,
    /// collect + schedule spikes. Returns this tick's spiking nodes
    /// (unsorted; compare as a set against the dense path).
    ///
    /// Bit-identical to [`Self::step_dense`] (the G1 pin).
    pub fn step(&mut self) -> &[u32] {
        self.deliver_due();
        let cur = core::mem::take(&mut self.active);
        let mut next = core::mem::take(&mut self.active_next);
        next.clear();
        let mut spikes = core::mem::take(&mut self.spikes);
        spikes.clear();
        let (a_m, a_s, c_gs) = self.params.decays();
        let g_floor = self.params.g_floor;
        for &u in cur.iter() {
            if self.update_node(u, a_m, a_s, c_gs, g_floor) {
                spikes.push(u);
            }
            if !self.is_quiescent(u, a_m, c_gs) {
                next.push(u);
            } else {
                self.is_active[u as usize] = false;
            }
        }
        // Canonical spike order (ascending): the dense path collects in
        // node order by construction; sorting here makes the EVENT path's
        // delay-ring bucket writes byte-identical to it — two spikers
        // hitting one target must accumulate g in the SAME order or the
        // f32 rounding diverges by ULPs (caught by the G1 parity gate).
        spikes.sort_unstable();
        self.active_next = cur; // old list becomes the next scratch
        self.active = next;
        self.spikes = spikes;
        self.schedule_spikes();
        &self.spikes
    }

    /// One dense reference tick: identical phases, but every node is visited.
    /// O(N) node updates + spiker edge writes — the CSR full-scan arm.
    pub fn step_dense(&mut self) -> &[u32] {
        self.deliver_due();
        let n = self.n();
        let mut spikes = core::mem::take(&mut self.spikes);
        spikes.clear();
        let (a_m, a_s, c_gs) = self.params.decays();
        let g_floor = self.params.g_floor;
        for u in 0..n {
            if self.update_node(u, a_m, a_s, c_gs, g_floor) {
                spikes.push(u);
            }
        }
        // Rebuild the active set so a dense→event mode switch stays correct.
        self.active.clear();
        self.is_active.fill(false);
        for u in 0..n {
            if !self.is_quiescent(u, a_m, c_gs) {
                self.is_active[u as usize] = true;
                self.active.push(u);
            }
        }
        self.spikes = spikes;
        self.schedule_spikes();
        &self.spikes
    }

    /// The exact skip criterion: the next dense update of `u` would be the
    /// identity (`refrac == 0 ∧ g == 0.0 ∧ leak(v) == v` bitwise). Evaluated
    /// with the same expression shape as [`Self::update_node`] so `±0.0`
    /// edge cases classify identically.
    #[inline]
    fn is_quiescent(&self, u: u32, a_m: f32, c_gs: f32) -> bool {
        let i = u as usize;
        let v = self.v[i];
        let g = self.g[i];
        self.refrac[i] == 0
            && g == 0.0
            && (self.params.v0 + (v - self.params.v0) * a_m + g * c_gs).to_bits() == v.to_bits()
    }

    /// Phase 1: drain the due bucket into `g` (both paths share this —
    /// delivery order is the ring's insertion order, so parity is
    /// structural).
    fn deliver_due(&mut self) {
        let mut bucket = core::mem::take(&mut self.ring[self.ring_head]);
        let w_scale = self.params.w_scale();
        for &(t, w_psp) in bucket.iter() {
            self.g[t as usize] += w_psp * w_scale;
            self.wake(t);
        }
        bucket.clear();
        self.ring[self.ring_head] = bucket;
        self.ring_head = (self.ring_head + 1) % self.ring.len();
    }

    /// Phase 2 (per-node): exact exponential integration + threshold.
    /// Shared by both paths — THE parity guarantee.
    #[inline]
    fn update_node(&mut self, u: u32, a_m: f32, a_s: f32, c_gs: f32, g_floor: f32) -> bool {
        let i = u as usize;
        let p = &self.params;
        if self.refrac[i] > 0 {
            self.refrac[i] -= 1;
            self.v[i] = p.v_rst;
            self.g[i] = decay_g(self.g[i], a_s, g_floor);
            return false;
        }
        let v_old = self.v[i];
        let g_old = self.g[i];
        let v_new = p.v0 + (v_old - p.v0) * a_m + g_old * c_gs;
        let g_new = decay_g(g_old, a_s, g_floor);
        if v_new >= p.v_th {
            self.v[i] = p.v_rst;
            self.g[i] = g_new;
            self.refrac[i] = p.refrac_ticks;
            true
        } else {
            self.v[i] = v_new;
            self.g[i] = g_new;
            false
        }
    }

    /// Phase 3: schedule this tick's spikers' outgoing edges into the delay
    /// ring. `ring_head` was already advanced past the just-drained bucket,
    /// so `+delay_ticks` lands on that same bucket — drained exactly
    /// `delay_ticks` ticks later.
    fn schedule_spikes(&mut self) {
        debug_assert_eq!(self.params.delay_ticks as usize, self.ring.len());
        let len = self.ring.len();
        let slot = (self.ring_head + len - 1) % len;
        let spikes = core::mem::take(&mut self.spikes);
        {
            let bucket = &mut self.ring[slot];
            let adj = &self.adj;
            for &s in spikes.iter() {
                let (a, b) = adj.row_range(s);
                for e in a..b {
                    bucket.push((adj.targets()[e], adj.weights()[e]));
                }
            }
        }
        self.spikes = spikes;
    }
}

#[inline]
fn decay_g(g: f32, a_s: f32, g_floor: f32) -> f32 {
    let g_next = g * a_s;
    if g_next.abs() < g_floor { 0.0 } else { g_next }
}

// ─── Ridge readout (closed-form; consumes the KARC-precedent solver) ────────

/// Fit a linear readout `ŷ = Wᵀ·x` from reservoir state history to targets,
/// by f64 ridge regression (`W = (XᵀX + λI)⁻¹·XᵀY`).
///
/// `states` is `T×n_features` row-major (typically the per-tick `v` slice or
/// `v ⊕ g`), `targets` is `T×n_out`. Returns `W` as `n_features × n_out`
/// row-major. Cold path (fit once); Gram/cov accumulate in f64 — the KARC
/// numerics discipline for small-λ regimes.
///
/// Feature-space form: feasible when `n_features ≤ ~2000`; beyond that use
/// the sample-space (Woodbury) form on the caller side (see
/// `linalg::ridge_solve_woodbury_f32`).
///
/// # Panics
/// If `lambda <= 0` (SPD precondition) or the slices are not row-aligned.
pub fn fit_readout(
    states: &[f32],
    targets: &[f32],
    n_features: usize,
    n_out: usize,
    lambda: f64,
) -> Vec<f32> {
    let t = states.len() / n_features;
    assert_eq!(states.len(), t * n_features, "states not row-aligned");
    assert_eq!(targets.len(), t * n_out, "targets not row-aligned");
    assert!(lambda > 0.0, "lambda > 0 is a hard precondition (SPD Gram)");
    let d = n_features;
    let mut gram_reg = vec![0.0_f64; d * d];
    let mut cov = vec![0.0_f64; d * n_out];
    for row in 0..t {
        let x = &states[row * d..(row + 1) * d];
        let y = &targets[row * n_out..(row + 1) * n_out];
        for i in 0..d {
            let xi = x[i] as f64;
            for j in i..d {
                gram_reg[i * d + j] += xi * x[j] as f64;
            }
            for (o, &yv) in y.iter().enumerate() {
                cov[i * n_out + o] += xi * yv as f64;
            }
        }
    }
    for i in 0..d {
        gram_reg[i * d + i] += lambda;
        for j in (i + 1)..d {
            gram_reg[j * d + i] = gram_reg[i * d + j]; // symmetrize
        }
    }
    let mut w_t = vec![0.0_f64; d * n_out];
    let mut l_scratch = vec![0.0_f64; d * d];
    let mut z_scratch = vec![0.0_f64; d * n_out];
    ridge_solve_direct_f64(
        &mut w_t,
        &mut l_scratch,
        &mut z_scratch,
        &gram_reg,
        &cov,
        d,
        n_out,
    );
    w_t.into_iter().map(|x| x as f32).collect()
}

// ─── Unit tests (integration G1/G4 live in tests/) ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-synapse PSP in the Shiu regime: w = 0.275 mV.
    const W: f32 = 0.275;

    #[test]
    fn shiu_params_decays() {
        let p = LifParams::shiu();
        let (a_m, a_s, c_gs) = p.decays();
        assert!((a_m - (-0.1_f32 / 20.0).exp()).abs() < 1e-6);
        assert!((a_s - (-0.1_f32 / 5.0).exp()).abs() < 1e-6);
        // Excitation depolarizes: c_gs > 0 (tau_syn < t_mbr).
        assert!(c_gs > 0.0);
        assert_eq!(p.refrac_ticks, 22);
        assert_eq!(p.delay_ticks, 18);
        assert_eq!(p.w_scale(), 4.0);
    }

    #[test]
    fn chain_first_fire_at_exact_delays() {
        // Instant-suprathreshold weights (v jumps over threshold on the
        // delivery tick itself) make the FIRST-fire ticks exactly 0 / 18 /
        // 36. The large drive then decays over several refractory periods —
        // burst re-fires are expected and asserted only as "≥ first fires".
        let adj = SignedAdjacency::from_edges(3, &[(0, 1, 400.0), (1, 2, 400.0)]);
        let mut r = LifReservoir::new(adj, LifParams::shiu());
        r.inject(0, 400.0);
        let mut fire_ticks = Vec::with_capacity(200);
        for t in 0..200 {
            for &s in r.step() {
                fire_ticks.push((s, t));
            }
        }
        let first = |node: u32| fire_ticks.iter().find(|(s, _)| *s == node).unwrap().1;
        assert_eq!(first(0), 0, "driven node fires on its drive tick");
        assert_eq!(first(1), 18, "one synapse delay (1.8 ms / 0.1 ms)");
        assert_eq!(first(2), 36, "two synapse delays");
        assert!(fire_ticks.len() >= 3);
        // Determinism of the full burst pattern (run twice, compare).
        let mut r2 = LifReservoir::new(
            SignedAdjacency::from_edges(3, &[(0, 1, 400.0), (1, 2, 400.0)]),
            LifParams::shiu(),
        );
        r2.inject(0, 400.0);
        let mut fire_ticks2 = Vec::with_capacity(200);
        for t in 0..200 {
            for &s in r2.step() {
                fire_ticks2.push((s, t));
            }
        }
        assert_eq!(fire_ticks, fire_ticks2);
    }

    #[test]
    fn never_hit_node_stays_bitwise_at_rest() {
        let adj = SignedAdjacency::from_edges(4, &[(0, 1, 400.0)]);
        let mut r = LifReservoir::new(adj, LifParams::shiu());
        r.inject(0, 400.0);
        for _ in 0..300 {
            r.step();
        }
        let (v, g, refrac) = r.state();
        assert_eq!(v[3].to_bits(), (-52.0_f32).to_bits());
        assert_eq!(g[3].to_bits(), 0.0_f32.to_bits());
        assert_eq!(refrac[3], 0);
    }

    #[test]
    fn maslov_sneppen_preserves_degrees() {
        let base = SignedAdjacency::er_matched(50, 300, 0.2, W, W, 7);
        let swapped = base.maslov_sneppen(500, 11);
        assert_eq!(base.edge_count(), swapped.edge_count());
        for row in 0..base.n() {
            assert_eq!(
                base.row_range(row).1 - base.row_range(row).0,
                swapped.row_range(row).1 - swapped.row_range(row).0,
                "out-degree of row {row} must be preserved"
            );
        }
        let mut in_base = vec![0u32; base.n() as usize];
        let mut in_swap = vec![0u32; base.n() as usize];
        for &t in base.targets() {
            in_base[t as usize] += 1;
        }
        for &t in swapped.targets() {
            in_swap[t as usize] += 1;
        }
        assert_eq!(in_base, in_swap, "in-degrees must be preserved");
    }

    #[test]
    fn fit_readout_recovers_linear_map() {
        // y = 2·x0 − x1 exactly: ridge must recover it to f32 noise.
        let states: Vec<f32> = (0..200)
            .flat_map(|i| {
                let a = (i as f32 * 0.37).sin();
                let b = (i as f32 * 0.11).cos();
                vec![a, b]
            })
            .collect();
        let targets: Vec<f32> = (0..200)
            .map(|i| 2.0 * (i as f32 * 0.37).sin() - (i as f32 * 0.11).cos())
            .collect();
        let w = fit_readout(&states, &targets, 2, 1, 1e-6);
        assert!((w[0] - 2.0).abs() < 1e-3, "w0={}", w[0]);
        assert!((w[1] + 1.0).abs() < 1e-3, "w1={}", w[1]);
    }
}
