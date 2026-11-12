//! Plan 634 T2.1 — the F1 ordering-law battery (paper Prop 5.6 shape) on
//! the planted clique-graph (Def 5.5: K cliques of M, K-cycle meta-graph ⇒
//! r = 2, c = 1 pinned+asserted in the shared fixture).
//!
//! Three arms:
//! (i) the SRW meta-graph gap is EXACTLY `γ_srw = (r/(M−1+r))·(1 − λ₂(P_Q))`
//!     (paper eqs C.14–C.15) — asserted to pinned ε in f64 against the
//!     EMPIRICAL gap of the fixture's own walk matrix (in-crate Jacobi);
//! (ii) the Alg-4 walk's meta-gap satisfies
//!     `γ_alg4 ≥ ((M−1+r)/r)·min(1−η_M, η_M)·γ_srw` (paper eqs C.6–C.10 +
//!     C.20, `η_M = (B_M−2)/(A_M+B_M−4)`, `A_M = M−2+(r+1)/(M−1)`,
//!     `B_M = c+(d−c)/r`, `d = M−1+r`) — asserted with the plan's 0.5
//!     safety factor on the provable bound (f32 arithmetic upstream);
//! (iii) the global-roulette arm (uniform jump over all nodes) REPORTED
//!     beside both — the CGSP-relevant baseline; it has no cluster-lingering
//!     to fix, so no bar.
//!
//! A failure of (i) means the FIXTURE or the closed form is wrong (fix the
//! math, never loosen ε); a failure of (ii) means the shipped Alg-4 walk
//! does not carry the paper's escape guarantee (the primitive is wrong,
//! not under-tuned).

#[path = "plan634/mod.rs"]
mod plan634_common;

use katgpt_core::diversity::diffusion::{LocalScratch, neighbor_diversity_pick};
use plan634_common::{eig_sym, perron_deflated_radius, planted_clique_graph};

/// η for the Alg-4 floor (the paper's pinned optimal `η_M`, computed from
/// the Def-5.5 constants — see `eta_m` below).
const TAU: usize = 2;
const SAFETY: f64 = 0.5;

/// `A_M = M − 2 + (r+1)/(M−1)` (paper C.6–C.10 constants).
fn a_m(m: usize, r: usize) -> f64 {
    (m - 2) as f64 + (r + 1) as f64 / (m - 1) as f64
}

/// `B_M = c + (d−c)/r` with `d = M−1+r`, `c = 1` (the fixture's pin).
fn b_m(r: usize, d: usize, c: usize) -> f64 {
    c as f64 + (d - c) as f64 / r as f64
}

/// `η_M = (B_M − 2)/(A_M + B_M − 4)` — the paper's optimal floor.
fn eta_m(m: usize, r: usize, d: usize, c: usize) -> f64 {
    let (a, b) = (a_m(m, r), b_m(r, d, c));
    (b - 2.0) / (a + b - 4.0)
}

/// The dense SRW matrix `P = D⁻¹W` of the fixture graph (f64 from the
/// CSR's f32 weights; the fixture's weights are exactly representable at
/// `1/M` for M ∈ {4, 5, 6, 8} — M = 6's 1/6 carries the f32 tail, which
/// the uniform-degree structure cancels: every row computes the SAME
/// degree, so P = W/d_computed stays exactly symmetric).
fn srw_matrix(graph: &katgpt_core::diversity::diffusion::SimGraph) -> Vec<f64> {
    let n = graph.n;
    let mut w = vec![0.0_f64; n * n];
    for x in 0..n {
        let (cols, ws) = graph.row(x);
        for (&y, &wt) in cols.iter().zip(ws.iter()) {
            w[x * n + y] = wt as f64;
        }
    }
    // Uniformity of the computed degree is the closed form's premise —
    // assert it, never assume it.
    let deg: Vec<f64> = (0..n)
        .map(|x| (0..n).map(|y| w[x * n + y]).sum::<f64>())
        .collect();
    let d0 = deg[0];
    for (x, &dx) in deg.iter().enumerate() {
        assert!(
            (dx - d0).abs() < 1e-9,
            "degree not uniform at node {x}: {dx} vs {d0} — the exact closed form's premise broke"
        );
    }
    let mut p = vec![0.0_f64; n * n];
    for i in 0..n * n {
        p[i] = w[i] / d0;
    }
    p
}

/// The meta-projected chain of a node-constant-on-cliques block matrix
/// (the within-clique symmetry makes the projection exact: every node of a
/// clique carries the same row by the graph's automorphisms — asserted).
/// The meta-projected chain of a node-constant-on-cliques block matrix.
/// The within-clique symmetry makes the projection exact: the CROSS-clique
/// entries of a row are automorphism-invariant, so the AGGREGATED row
/// (mass per target clique) is the same for every node of a clique —
/// asserted here per clique (per-ENTRY agreement is NOT claimed: the QP
/// legitimately favors some intra-clique mates over others).
fn meta_project(p: &[f64], n: usize, m: usize, k: usize) -> Vec<f64> {
    let aggregate = |node: usize| -> [f64; 64] {
        let mut agg = [0.0_f64; 64];
        let row = &p[node * n..(node + 1) * n];
        for (j, &v) in row.iter().enumerate() {
            agg[j / m] += v;
        }
        agg
    };
    let mut meta = vec![0.0_f64; k * k];
    for c in 0..k {
        let agg0 = aggregate(c * m);
        for i in 1..m {
            let aggr = aggregate(c * m + i);
            for j in 0..k {
                // f32 tolerance: the walk rows come out of the shipped
                // solver's f32 QP, so cross-node symmetry holds to the
                // f32 noise floor (~1e-7 relative), never f64 exact.
                assert!(
                    (agg0[j] - aggr[j]).abs() < 1e-5,
                    "Alg-4 aggregated row differs within clique {c} (node 0 vs {i}, target clique {j}: {} vs {}) — the meta projection is not exact",
                    agg0[j],
                    aggr[j]
                );
            }
        }
        for j in 0..k {
            meta[c * k + j] = agg0[j];
        }
    }
    meta
}

fn run_fixture(k: usize, m: usize, label: &str) {
    let (g, _cluster_of, r, d, meta_adj) = planted_clique_graph(k, m);
    let n = g.n;
    assert_eq!(n, k * m);
    let c = 1_usize;

    // ── Arm (i): SRW gap, closed form vs empirical ─────────────────────
    let p = srw_matrix(&g);
    let (p_eig, _) = eig_sym(&p, n);
    // Second-largest eigenvalue: the top is exactly 1 (row-stochastic);
    // guard the sort's top, then read [1].
    assert!(
        (p_eig[0] - 1.0).abs() < 1e-9,
        "Perron root must be 1, got {}",
        p_eig[0]
    );
    let gamma_emp = 1.0 - p_eig[1];

    // λ₂(P_Q) from the meta SRW (r-regular: P_Q = adj/r), plus its
    // analytic cycle value cos(2π/K) as an extra pin.
    let mut q = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            q[i * k + j] = if meta_adj[i][j] { 1.0 / r as f64 } else { 0.0 };
        }
    }
    let (q_eig, _) = eig_sym(&q, k);
    let lambda2_q = q_eig[1];
    let lambda2_cycle = (2.0 * std::f64::consts::PI / k as f64).cos();
    assert!(
        (lambda2_q - lambda2_cycle).abs() < 1e-9,
        "λ₂(P_Q) {lambda2_q} must equal the cycle's cos(2π/K) {lambda2_cycle}"
    );

    let gamma_srw_closed = (r as f64 / (m - 1 + r) as f64) * (1.0 - lambda2_q);
    let err_i = (gamma_emp - gamma_srw_closed).abs();
    assert!(
        err_i < 1e-6,
        "[{label}] arm (i) FAILED: empirical SRW gap {gamma_emp:.12} vs closed form \
         {gamma_srw_closed:.12} (err {err_i:.3e} >= 1e-6) — the fixture or the closed form is wrong"
    );

    // ── Arm (ii): the Alg-4 walk's meta-gap vs the Prop 5.6 bound ──────
    let eta = eta_m(m, r, d, c);
    assert!((0.0..1.0).contains(&eta), "η_M {eta} out of (0,1)");
    // Build the Alg-4 transition rows from the SHIPPED walk-time selector.
    let mut scratch = LocalScratch::new(n);
    let mut p_alg4 = vec![0.0_f64; n * n];
    for x in 0..n {
        let (cols, _) = g.row(x);
        let kk = cols.len();
        assert!(
            kk <= 64,
            "node {x} degree {kk} > 64 (the walk solver's bound)"
        );
        let mut mu = vec![0.0_f32; kk];
        neighbor_diversity_pick(&g, x, TAU, eta as f32, &mut scratch, &mut mu);
        let total: f32 = mu.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "row mass {total} at node {x}");
        for (&y, &mv) in cols.iter().zip(mu.iter()) {
            p_alg4[x * n + y] = mv as f64;
        }
    }
    let meta_alg4 = meta_project(&p_alg4, n, m, k);
    // The projected chain is row-stochastic by construction; assert.
    for i in 0..k {
        let row_sum: f64 = meta_alg4[i * k..(i + 1) * k].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "meta row {i} sums to {row_sum}"
        );
    }
    let radius_alg4 = perron_deflated_radius(&meta_alg4, k, 400);
    let gamma_alg4 = 1.0 - radius_alg4;

    let bound = ((m - 1 + r) as f64 / r as f64) * (1.0 - eta).min(eta) * gamma_srw_closed;
    let ratio = gamma_alg4 / gamma_srw_closed;
    assert!(
        gamma_alg4 >= SAFETY * bound,
        "[{label}] arm (ii) FAILED: Alg-4 meta gap {gamma_alg4:.8} < 0.5·bound {:.8} \
         (bound {bound:.8}, ratio γ_alg4/γ_srw = {ratio:.4}, η_M = {eta:.6}) — \
         the shipped walk does not carry the paper's escape guarantee",
        SAFETY * bound
    );

    // ── Arm (iii): global-roulette baseline, REPORTED ──────────────────
    // Uniform jump over all nodes: the induced meta chain is the complete
    // walk over K states (every clique equally likely), whose second
    // eigenvalue is 0 ⇒ γ = 1. Reported beside (i)/(ii), never a bar.
    let mut p_jump = vec![0.0_f64; n * n];
    for v in p_jump.iter_mut() {
        *v = 1.0 / n as f64;
    }
    let meta_jump = meta_project(&p_jump, n, m, k);
    let radius_jump = perron_deflated_radius(&meta_jump, k, 200);
    let gamma_jump = 1.0 - radius_jump;

    println!(
        "[{label}] K={k} M={m} r={r} d={d} c={c} τ={TAU} η_M={eta:.6}\n  \
         (i)  γ_srw: empirical {gamma_emp:.10} == closed {gamma_srw_closed:.10} (err {err_i:.2e}) ✓\n  \
         (ii) γ_alg4 = {gamma_alg4:.8} ≥ 0.5·bound {:.8} — ratio {ratio:.4}× γ_srw ✓\n  \
         (iii) γ_jump = {gamma_jump:.8} (informational baseline)",
        SAFETY * bound
    );
}

#[test]
fn t21_f1_ordering_law_planted_clique_graphs() {
    run_fixture(8, 6, "planted-K8-M6");
    run_fixture(6, 5, "planted-K6-M5");
}
