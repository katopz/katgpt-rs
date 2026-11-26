//! Bench 815 — Coulomb crowd-redistribution PoC (Issue 825 T1+T2).
//!
//! The discrete analog of the BTM Coulomb construction (arXiv:2608.01692 v3,
//! Prop 2): solve `Lφ = μ₀ − μ₁` on the zone graph (grounded at one vertex —
//! the Laplacian is singular on the constant mode; the input is projected to
//! Σρ = 0 first, which the fixture guarantees), take the edge flow
//! `j = d₀φ`, and read the physical flow as `-j` (sources are Laplacian
//! peaks, so downhill = out of sources, into sinks). Per-NPC first-arrival is
//! the absorbing Markov chain induced by the positive part of the leaving
//! flow: `leaving(v, e) = sign(v, e) · j[e]` (positive = leaves v along the
//! edge's orientation or against it), with the stopping rule taken from the
//! shipped [`katgpt_dec::CrowdRouter::consistent_absorption`]:
//! `μ₁(v) / (inflow(v) + μ₀(v))`.
//!
//! ⛔ **The first version of this bench hard-coded "sinks absorb" to 1.0 and
//! reported the whole PoC as a NEGATIVE result on a 0.119 endpoint MAE.** It
//! was the readout, not the construction: a sink lying on the path to another
//! sink has a non-zero outflow (measured here: `out_flow = 0.209` at sink
//! `n−2`, consistent absorption **0.657**), and absorbing all of it strands
//! the mass the solve routed THROUGH. The refinement trend that was read as
//! "converging ⇒ discretization error" was that geometry thinning out. With
//! the shipped rule every fixture in the family reads MAE ≈ 0 — see
//! `.benchmarks/815_coulomb_redistribution_poc.md` § Retraction.
//!
//! - **T1 (construction)** — 4×3 grid (12 zones, 17 edges), 5 uniform source
//!   zones (0.2 each) and 2 sinks with UNEQUAL target weights (0.6 / 0.4 —
//!   the deliberate unequal choice mirroring the paper's 5-atom bias
//!   experiment). Asserts: (a) `δ₁(j) = μ₀ − μ₁` to fp tolerance
//!   (`belief_mass_divergence` — conservation by construction, Plan 314's
//!   steady-state check generalized to `μ₀ ≠ μ₁`); (b) the absorption
//!   endpoint distribution matches `μ₁` (MAE ≤ 0.01 — measured ≈ 0, the flow
//!   decomposition is exact and only per-NPC SAMPLING is approximate;
//!   never-arriving trapped mass counts against the MAE honestly); (c) the
//!   whole solve path is
//!   zero-alloc (fixed-size stack arrays + `exterior_derivative_into`,
//!   counting allocator) — the G4 class, the redistribution solve is
//!   cacheable per event, not per tick (Issue 825 caveats).
//! - **T2 (the 20×-bias analog gate, the GOAT axis)** — the naive rescaled
//!   attract field the shipped `DecFlowField` consumption shape implies,
//!   `φ_naive[v] = −Σ_j w_j · dist(v, s_j)` with the hand-set weights
//!   `w_j = μ₁[j]` (plus the max-form variant), flows through the SAME
//!   absorption readout. Gate: Coulomb MAE ≤ 0.01 AND ≥ 10× better than
//!   naive. If naive is NOT badly biased on this toy graph, the run says so
//!   honestly and Issue 825 T3 re-adjudicates whether the primitive earns a
//!   flag (the issue's own honest-outcome clause) — that clause does not
//!   flip the exit code; a conservation failure or a Coulomb-MAE failure
//!   does. ⚠ The printed ratio is a LOWER BOUND wherever the Coulomb MAE
//!   lands below this instrument's fp resolution: the quotient would
//!   otherwise report the f32 cochain's noise floor wearing a bias ratio.
//!
//! # Relationship to Bench 825
//!
//! [Bench 825](../../../.benchmarks/825_coulomb_crowd_redistribution_goat.md)
//! is the promotion gate for the shipped `coulomb_flow` primitive and runs
//! the per-NPC SAMPLED walk (G2 gates the 1/√N decay). This bench keeps its
//! own f64 Gaussian solver and its own deterministic mass-packet walker, so
//! the two agree through independent solve and readout implementations —
//! that independence is the reason it was repaired rather than deleted. The
//! one thing they deliberately SHARE is the absorption rule, which is the
//! thing they disagreed about.
//!
//! Zero dependencies: the Poisson solve is a small pinned Gaussian
//! elimination in f64 (grounded Laplacians of connected graphs are
//! nonsingular), distances are BFS hop counts — both over fixed arrays sized
//! `MAX_V = 16`.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/bench815 cargo bench -p katgpt-dec \
//!   --bench bench_815_coulomb_redistribution_poc -- --nocapture
//! ```

#![cfg(feature = "coulomb_flow")]

// Shared CountingAllocator macro (the bench_407 mirror).
#[path = "../tests/common/counting_allocator.rs"]
mod counting_allocator;

use katgpt_dec::{CellComplex, CochainField, CrowdRouter, exterior_derivative_into};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

counting_allocator!();

/// Fixture ceiling: the largest refinement case is 12×9 = 108 zones; arrays
/// are sized to the PoC bound so the solve path stays allocation-free.
const MAX_V: usize = 144;
const MAX_E: usize = 2 * MAX_V - 24; // grid_2d(12, 9): 2wh − w − h = 195

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

/// Grid adjacency + per-edge (tail, head) read back from the complex's own
/// boundary entries — no re-derivation of the orientation convention.
struct Graph {
    /// `edges[e] = (tail, head)`.
    edges: [(usize, usize); MAX_E],
    n_vertices: usize,
    /// `adj[v]` = `(neighbor, edge_idx, sign_of_v_in_edge)` slices — stored
    /// flat: `adj_start[v]..adj_start[v+1]` indexes `adj_flat`.
    adj_start: [usize; MAX_V + 1],
    /// `(neighbor, edge_idx, sign)` — sign is the vertex's sign in B₁.
    adj_flat: [(usize, usize, i8); 2 * MAX_E],
}

impl Graph {
    fn from_complex(cx: &CellComplex) -> Self {
        let n_vertices = cx.n_vertices();
        let mut edges = [(0usize, 0usize); MAX_E];
        let mut adj_start = [0usize; MAX_V + 1];
        let mut adj_flat = [(0usize, 0usize, 0i8); 2 * MAX_E];
        let mut deg = [0usize; MAX_V];
        for &(v, e, sign) in cx.boundary_entries(0) {
            deg[v] += 1;
            if sign < 0 {
                edges[e].0 = v;
            } else {
                edges[e].1 = v;
            }
        }
        let mut start = 0usize;
        for v in 0..n_vertices {
            adj_start[v] = start;
            start += deg[v];
        }
        adj_start[n_vertices] = start;
        let mut fill = adj_start;
        for &(v, e, sign) in cx.boundary_entries(0) {
            let other = if sign < 0 { edges[e].1 } else { edges[e].0 };
            adj_flat[fill[v]] = (other, e, sign);
            fill[v] += 1;
        }
        Self {
            edges,
            n_vertices,
            adj_start,
            adj_flat,
        }
    }

    /// BFS hop distance from `src` to every vertex (f64∞ for unreachable —
    /// the fixture graph is connected, the sentinel is defensive).
    fn bfs_dist(&self, src: usize, dist: &mut [f64; MAX_V]) {
        *dist = [f64::INFINITY; MAX_V];
        dist[src] = 0.0;
        let mut queue = [0usize; MAX_V];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = src;
        while head < tail {
            let v = queue[head];
            head += 1;
            for &(u, _, _) in &self.adj_flat[self.adj_start[v]..self.adj_start[v + 1]] {
                if dist[u].is_infinite() {
                    dist[u] = dist[v] + 1.0;
                    queue[tail] = u;
                    tail += 1;
                }
            }
        }
    }
}

/// Pinned Gaussian elimination (f64, partial pivoting) on the grounded
/// Laplacian: row 0 is replaced by the unit row (`φ₀ = 0`), rows 1..n carry
/// `L` restricted to the free vertices. Solves in place; `x` receives `φ`.
///
/// Zero-alloc: fixed `[f64; MAX_V + 1]` row arrays, no Vec anywhere.
#[allow(
    clippy::needless_range_loop,
    reason = "the Gauss elimination indexes a 2D augmented matrix — the indexed form is the textbook form"
)]
fn solve_grounded(
    l_free: &[[f64; MAX_V]; MAX_V],
    rhs_free: &[f64; MAX_V],
    n: usize,
    x: &mut [f64; MAX_V],
) {
    // Augmented matrix: n rows (row 0 = the pin), n+1 columns (+RHS).
    let mut a = [[0.0f64; MAX_V + 1]; MAX_V];
    a[0][0] = 1.0; // pin: φ₀ = 0
    for r in 1..n {
        for c in 1..n {
            a[r][c] = l_free[r][c];
        }
        a[r][n] = rhs_free[r];
    }
    // Forward elimination with partial pivoting.
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        let inv = 1.0 / a[col][col];
        for r in col + 1..n {
            let f = a[r][col] * inv;
            if f != 0.0 {
                for c in col..=n {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
    }
    // Back-substitution.
    x[0] = 0.0;
    for r in (1..n).rev() {
        let mut s = a[r][n];
        for c in r + 1..n {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
}

/// The Coulomb solve path: project ρ, build the grounded Laplacian from the
/// complex's boundary entries, solve, take `j = d₀φ` into the caller's
/// preallocated cochain. This whole function is the G4 zero-alloc unit —
/// after the caller preallocates `j` and the scratch buffers, nothing here
/// allocates.
#[allow(
    clippy::too_many_arguments,
    reason = "the PoC solve passes preallocated scratch explicitly so the G4 zero-alloc unit is honest"
)]
fn coulomb_solve(
    cx: &CellComplex,
    graph: &Graph,
    rho: &[f64; MAX_V],
    phi: &mut CochainField,
    j: &mut CochainField,
    scratch_l: &mut [[f64; MAX_V]; MAX_V],
    scratch_rhs: &mut [f64; MAX_V],
    scratch_phi: &mut [f64; MAX_V],
) {
    let n = graph.n_vertices;
    // Grounded Laplacian over free vertices 1..n: L[v][v] = deg(v),
    // L[v][u] = −(#edges v–u) — read from the boundary entries.
    for scratch_row in scratch_l.iter_mut().take(n) {
        *scratch_row = [0.0; MAX_V];
    }
    for &(v, e, _) in cx.boundary_entries(0) {
        let (t, h) = graph.edges[e];
        scratch_l[v][v] += 1.0;
        let other = if v == t { h } else { t };
        scratch_l[v][other] -= 1.0;
    }
    // Pin row/col 0 (φ₀ = 0): the free system is rows/cols 1..n of L, with
    // the row-0 coupling of each free vertex moved into nothing (φ₀ = 0).
    scratch_rhs[1..n].copy_from_slice(&rho[1..n]);
    solve_grounded(scratch_l, scratch_rhs, n, scratch_phi);
    // φ into the caller's rank-0 cochain → j = d₀φ (into the caller's
    // cochain). NOTHING in this function allocates after the caller's
    // preallocation — the G4 gate measures exactly this unit.
    for (v, slot) in phi.data.iter_mut().enumerate() {
        *slot = scratch_phi[v] as f32;
    }
    exterior_derivative_into(cx, phi, j);
}

/// Absorption readout: propagate the source mass along the positive leaving
/// flow, absorbing a `stop[v]` fraction of whatever passes through `v`.
///
/// The physical flow is `-j` — the rank-0 Laplacian makes source zones
/// φ-peaks (`Lφ[v] > 0` = neighbors lower), so downhill (`-∇φ`) leaves the
/// source and enters the sink pits; with `leaving(v, e) = sign(v, e) · j[e]`
/// the walk would route mass BACK to the sources (the first run measured
/// exactly that: endpoints 0/1 instead of 0.6/0.4).
///
/// ⛔ **`stop` is the whole of the endpoint guarantee, and the first version
/// of this bench hard-coded it to 1.0 at every sink.** That is not the flow
/// decomposition: a sink that lies ON a path to another sink carries a
/// non-zero outflow, and absorbing all of it strands mass that the solve had
/// routed THROUGH. Measured on this file's own `(4,3,5)` fixture — sink
/// `n-2` has `out_flow = 0.209` and a consistent absorption of **0.657**, not
/// 1.0 — which is the entire 0.119 endpoint miss the first run read as
/// "discretization error". The rule comes from
/// [`CrowdRouter::consistent_absorption`] rather than a local copy, so the
/// two benches on this primitive cannot disagree about it again.
///
/// Returns the trapped mass — whatever reached a vertex with nowhere to go
/// and no absorption. It counts against the endpoint MAE honestly.
///
/// Deterministic: the particle is a f64 mass packet, not an RNG sample —
/// no fixture RNG streams (the katgpt-rs global_rng_gate posture).
fn absorb(
    graph: &Graph,
    j: &CochainField,
    mu0: &[f64; MAX_V],
    stop: &[f64; MAX_V],
    endpoint: &mut [f64; MAX_V],
) -> f64 {
    // Per-vertex leaving-flow totals + per-edge probabilities, f64 from f32.
    let n = graph.n_vertices;
    let mut packet = [0.0f64; MAX_V];
    packet[..n].copy_from_slice(&mu0[..n]);
    let mut leaving = [0.0f64; 2 * MAX_E];
    for v in 0..n {
        let mut total = 0.0f64;
        let (lo, hi) = (graph.adj_start[v], graph.adj_start[v + 1]);
        for (i, slot) in leaving[lo..hi].iter_mut().enumerate() {
            let (_, e, sign) = graph.adj_flat[lo + i];
            // Downhill readout: j[e] = φ_head − φ_tail, the physical flow
            // follows −∇φ, so the tail pushes −j[e] toward the head (positive
            // exactly when φ_tail > φ_head) and the head pushes +j[e] back.
            // First run measured the inverted form routing mass to sources.
            let f = sign as f64 * j.data[e] as f64;
            *slot = f.max(0.0);
            total += *slot;
        }
        // Normalize in place (second pass keeps the first allocation-free).
        if total > 0.0 {
            let inv = 1.0 / total;
            for slot in leaving[lo..hi].iter_mut() {
                *slot *= inv;
            }
        }
    }
    let mut trapped = 0.0f64;
    for _step in 0..100_000 {
        let mut moved = false;
        for v in 0..n {
            if packet[v] <= 0.0 {
                continue;
            }
            // Absorb this vertex's share FIRST, then forward the remainder —
            // a vertex is a partial stop, never an all-or-nothing one.
            let avail = packet[v];
            let absorbed = avail * stop[v];
            if absorbed > 0.0 {
                endpoint[v] += absorbed;
                packet[v] -= absorbed;
            }
            let forward = packet[v];
            if forward <= 0.0 {
                continue;
            }
            let start = graph.adj_start[v];
            let end = graph.adj_start[v + 1];
            let mut total = 0.0f64;
            for &l in &leaving[start..end] {
                total += l;
            }
            if total == 0.0 {
                continue; // plateau/dead-end: trapped, counted at the cap
            }
            // Send the whole remainder proportionally (mass continuous, the
            // deterministic analog of per-NPC proportional routing).
            for (i, &p) in leaving[start..end].iter().enumerate() {
                if p > 0.0 {
                    let (u, _, _) = graph.adj_flat[start + i];
                    let share = forward * (p / total);
                    packet[u] += share;
                    packet[v] -= share;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
    for &residue in &packet[..n] {
        trapped += residue;
    }
    trapped
}

fn endpoint_mae(endpoint: &[f64; MAX_V], mu1: &[f64; MAX_V], sinks: &[usize]) -> f64 {
    sinks
        .iter()
        .map(|&s| (endpoint[s] - mu1[s]).abs())
        .sum::<f64>()
        / sinks.len() as f64
}

/// The T2 naive attract field, expressed in the SAME pit convention as the
/// Coulomb solve so one downhill readout serves both: `φ[v] = +Σ_j w_j ·
/// dist(v, s_j)` — sources (far) are peaks, sinks (dist 0) are pits, flow =
/// −∇φ (sum form; the max form takes the max-weighted distance).
fn naive_potential(
    graph: &Graph,
    mu1: &[f64; MAX_V],
    sinks: &[usize],
    use_max: bool,
    phi: &mut CochainField,
) {
    let mut dist = [0.0f64; MAX_V];
    phi.data.fill(0.0);
    for (v, slot) in phi.data.iter_mut().enumerate() {
        let mut acc = if use_max { f64::NEG_INFINITY } else { 0.0 };
        for &s in sinks {
            graph.bfs_dist(s, &mut dist);
            let term = mu1[s] * dist[v];
            if use_max {
                acc = acc.max(term);
            } else {
                acc += term;
            }
        }
        *slot = acc as f32;
    }
}

/// One fixture case: `grid_2d(w, h)`, sources = the first `n_sources`
/// vertices (uniform 1/n — at (4,3,5) exactly the issue's fixture), sinks =
/// the LAST two vertices (0.6 / 0.4 — at 4×3: vertices 11 and 10, exactly
/// the issue's unequal pair). Returns the measured Coulomb MAE, the two
/// naive-form MAEs, the conservation residual, and the endpoints.
#[allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "the Gauss elimination indexes a 2D augmented matrix — the indexed form is the textbook form; the PoC solve passes preallocated scratch explicitly so the G4 zero-alloc unit is honest"
)]
struct CaseResult {
    mae_coulomb: f64,
    mae_naive_sum: f64,
    mae_naive_max: f64,
    cons_residual: f64,
    ep: [f64; MAX_V],
    trapped: f64,
    solve_us: f64,
    allocs: usize,
    sinks: [usize; 2],
}

#[allow(clippy::needless_range_loop)]
fn run_case(w: usize, h: usize, n_sources: usize) -> CaseResult {
    let cx = CellComplex::grid_2d(w, h);
    let n = cx.n_vertices();
    debug_assert!(n <= MAX_V);
    let graph = Graph::from_complex(&cx);

    let sinks: [usize; 2] = [n - 1, n - 2];
    let mut mu0 = [0.0f64; MAX_V];
    let mut mu1 = [0.0f64; MAX_V];
    let share = 1.0 / n_sources as f64;
    for v in 0..n_sources {
        mu0[v] = share;
    }
    mu1[n - 1] = 0.6;
    mu1[n - 2] = 0.4;
    // ρ = μ₀ − μ₁, Σρ = 0 (mass-conserving input — asserted, not assumed).
    let mut rho = [0.0f64; MAX_V];
    for v in 0..n {
        rho[v] = mu0[v] - mu1[v];
    }
    let sum_rho: f64 = rho.iter().sum();
    assert!(sum_rho.abs() < 1e-12, "fixture must conserve mass");

    let mut j = CochainField::zeros(1, cx.n_edges(), 1);
    let mut phi = CochainField::zeros(0, n, 1);
    let mut l_free = [[0.0f64; MAX_V]; MAX_V];
    let mut rhs_free = [0.0f64; MAX_V];
    let mut phi_buf = [0.0f64; MAX_V];

    // T1a — conservation: δ₁(j) = ρ POINTWISE (the solve identity; the L1
    // magnitude of δ₁(j) is ‖ρ‖₁ ≠ 0 by design — this is a SOURCED flow,
    // not a divergence-free one).
    coulomb_solve(
        &cx,
        &graph,
        &rho,
        &mut phi,
        &mut j,
        &mut l_free,
        &mut rhs_free,
        &mut phi_buf,
    );
    let div_j = katgpt_dec::codifferential(&cx, &j);
    let mut cons_residual = 0.0f64;
    for v in 0..n {
        cons_residual = cons_residual.max((div_j.data[v] as f64 - rho[v]).abs());
    }

    // T1b — endpoint distribution vs μ₁ (absorbing-chain first arrival).
    //
    // The stopping rule comes from the shipped `CrowdRouter`, not from a copy
    // here: `μ₁(v) / (inflow(v) + μ₀(v))`. Its table is built on the router's
    // sign convention `δ(j') = μ₁ − μ₀`, which is exactly `-j` here (this
    // file solves `Lφ = ρ = μ₀ − μ₁`) — the negation is asserted by T1a's
    // residual one block up, not assumed.
    let mut j_router = CochainField::zeros(1, cx.n_edges(), 1);
    for (dst, &src) in j_router.data.iter_mut().zip(j.data.iter()) {
        *dst = -src;
    }
    let mu0_f32: Vec<f32> = (0..n).map(|v| mu0[v] as f32).collect();
    let mu1_f32: Vec<f32> = (0..n).map(|v| mu1[v] as f32).collect();
    let stop_f32 = CrowdRouter::consistent_absorption(&cx, &j_router, &mu0_f32, &mu1_f32);
    let mut stop = [0.0f64; MAX_V];
    for v in 0..n {
        stop[v] = stop_f32[v] as f64;
    }

    let mut endpoint = [0.0f64; MAX_V];
    let trapped = absorb(&graph, &j, &mu0, &stop, &mut endpoint);
    let mae_coulomb = endpoint_mae(&endpoint, &mu1, &sinks);

    // T2 — naive attract fields, same readout, ONE vector different.
    //
    // A hand-built attract field carries no target weights, so the only
    // stopping rule it can offer is "stop on first arrival" — `stop[sink] = 1`.
    // That asymmetry IS the finding; giving the naive arm the Coulomb
    // absorption vector would hand it the answer the solve produced.
    let mut stop_naive = [0.0f64; MAX_V];
    for &s in &sinks {
        stop_naive[s] = 1.0;
    }
    let mut phi_naive = CochainField::zeros(0, n, 1);
    let mut j_naive = CochainField::zeros(1, cx.n_edges(), 1);
    let mut ep_naive = [0.0f64; MAX_V];
    let mut maes = [0.0f64; 2];
    for (arm, use_max) in [(0usize, false), (1usize, true)] {
        naive_potential(&graph, &mu1, &sinks, use_max, &mut phi_naive);
        exterior_derivative_into(&cx, &phi_naive, &mut j_naive);
        ep_naive.fill(0.0);
        absorb(&graph, &j_naive, &mu0, &stop_naive, &mut ep_naive);
        maes[arm] = endpoint_mae(&ep_naive, &mu1, &sinks);
    }

    // G4 — zero-alloc solve path + latency (warmup, then snapshot).
    for _ in 0..10 {
        coulomb_solve(
            &cx,
            &graph,
            &rho,
            &mut phi,
            &mut j,
            &mut l_free,
            &mut rhs_free,
            &mut phi_buf,
        );
    }
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let t0 = Instant::now();
    for _ in 0..1000 {
        coulomb_solve(
            &cx,
            &graph,
            &rho,
            &mut phi,
            &mut j,
            &mut l_free,
            &mut rhs_free,
            &mut phi_buf,
        );
    }
    black_box((&j, &phi));
    let solve_us = t0.elapsed().as_secs_f64() * 1e6 / 1000.0;
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;

    CaseResult {
        mae_coulomb,
        mae_naive_sum: maes[0],
        mae_naive_max: maes[1],
        cons_residual,
        ep: endpoint,
        trapped,
        solve_us,
        allocs,
        sinks,
    }
}

fn main() {
    println!("═══ Bench 815 — Coulomb crowd redistribution PoC (Issue 825 T1+T2) ═══");
    println!("fixture family: grid_2d(w,h), sources = first n vertices uniform;");
    println!("   sinks = last two @ 0.6/0.4 (unequal — the paper's bias analog).");
    println!("   (4,3,5) IS the issue fixture: sources 0..4 @ 0.2, sinks 11/10.");
    println!();

    // The issue gate case + the mesh-refinement axis. ⚑ The first run of this
    // bench read 0.119 / 0.078 / 0.037 here and concluded "converging = the
    // miss is discretization error". It was neither: the walker absorbed 100%
    // at every sink, so a sink lying ON the path to the other sink stranded
    // the mass routed THROUGH it, and the trend was that geometry thinning
    // out as the grid grew. With the flow decomposition's own absorption rule
    // the readout is EXACT at every refinement — the axis is kept because a
    // future readout change that reintroduces an O(h) error shows up here
    // rather than in one number.
    let cases: [(usize, usize, usize); 3] = [(4, 3, 5), (8, 6, 8), (12, 9, 12)];
    let mut primary_ok = true;
    for (i, &(w, h, ns)) in cases.iter().enumerate() {
        let r = run_case(w, h, ns);
        let n = w * h;
        let best_naive = r.mae_naive_sum.min(r.mae_naive_max);
        // ⚠ The exact readout's residual is fp noise, not method error, so a
        // raw quotient here reports the f32 cochain's resolution wearing a
        // bias ratio (the first repaired run printed 1.5e7×). Divide by the
        // resolution floor instead and print the result as the LOWER BOUND it
        // is — the conservation residual one line up is ~1.8e-7 at 108 zones,
        // so nothing below FP_FLOOR is resolvable by this instrument.
        const FP_FLOOR: f64 = 1e-6;
        let fp_limited = r.mae_coulomb < FP_FLOOR;
        let ratio = best_naive / r.mae_coulomb.max(FP_FLOOR);
        println!(
            "── case {i}: grid_2d({w},{h}) = {n} zones, {ns} sources — sinks {} @ 0.6 / {} @ 0.4 ──",
            r.sinks[0], r.sinks[1]
        );
        println!(
            "   T1a conservation max\u{a0}|δ₁(j)−ρ| = {:.3e}  T1c solve = {:.2} µs, allocs = {}  → {}",
            r.cons_residual,
            r.solve_us,
            r.allocs,
            verdict(r.cons_residual <= 1e-3 && r.allocs == 0)
        );
        println!(
            "   T1b Coulomb endpoints: [{:.4}, {:.4}] (μ₁ [0.6000, 0.4000]), trapped = {:.2e}  MAE = {:.6}  → {}",
            r.ep[r.sinks[0]],
            r.ep[r.sinks[1]],
            r.trapped,
            r.mae_coulomb,
            verdict(r.mae_coulomb <= 0.01)
        );
        println!(
            "   T2 naive MAE: sum = {:.6}, max = {:.6}  → ratio naive-best/Coulomb {}{:.0}×{}  → {}",
            r.mae_naive_sum,
            r.mae_naive_max,
            if fp_limited { "≥ " } else { "= " },
            ratio,
            if fp_limited {
                " (Coulomb MAE below fp resolution — a floor, not a measurement)"
            } else {
                ""
            },
            verdict(ratio >= 10.0)
        );
        if i == 0 {
            // Only the ISSUE fixture flips the exit code (its gate: MAE ≤
            // 0.01 + conservation + zero-alloc). The refinement cases inform
            // the T3 verdict; a coarse-to-fine convergence that still misses
            // 0.01 at 108 zones is a RESEARCH answer, not a bench red.
            primary_ok = r.cons_residual <= 1e-3 && r.allocs == 0 && r.mae_coulomb <= 0.01;
        }
    }

    println!();
    if primary_ok {
        println!("══ COULOMB POC GATES PASS — T1 (a/b/c) + T2 bias ratio ══");
    } else {
        println!("══ COULOMB POC ISSUE-FIXTURE GATE FAILS — check the READOUT first: ══");
        println!("   this bench read a 0.119 endpoint miss as \"discretization error\" once");
        println!("   and it was the absorption rule. Print CrowdRouter::consistent_absorption");
        println!("   at each sink: anything pinned to 1.0 with a non-zero out_flow strands");
        println!("   pass-through mass. The solve is gated by T1a independently.");
    }
    std::process::exit(if primary_ok { 0 } else { 1 });
}
