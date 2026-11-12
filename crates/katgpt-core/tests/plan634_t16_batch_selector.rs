//! Plan 634 T1.6 — the batch selector's focused fixtures (the standing F5
//! battery with its planted-clique-graph pinned bars is Phase 2's T2.5; this
//! file pins the SEAM): (1) the T1.6 identity "cosine on e_τ = truncated
//! normalized φ_τ" measured, not assumed; (2) the selector's contract smoke —
//! coverage across clusters, no duplicates, byte determinism, the η mass
//! floor refusing, and the D > n zero-padding posture.

use katgpt_core::diversity::diffusion::{DiffusionEmbedding, SimGraph, batch_select_diffusion};

/// Two 3-cliques {0,1,2} and {3,4,5} joined by the 2–3 bridge (weight 0.5).
/// Within-clique weight 1.0. n = 6.
fn two_clique_graph() -> SimGraph {
    SimGraph::build(
        6,
        &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
            (2, 3, 0.5),
        ],
    )
}

/// Truncated normalized φ_τ computed DIRECTLY from the embedding's fields
/// (f64) — the identity's independent authority, never the code under test.
/// (`stride` = n recovered from the row-major eigvec storage — `n` itself is
/// private.)
fn truncated_normalized_phi(
    emb: &DiffusionEmbedding,
    x: usize,
    y: usize,
    tau: usize,
    d: usize,
) -> f64 {
    let raw = |a: usize, b: usize| truncated_phi_raw(emb, a, b, tau, d);
    let acc = raw(x, y);
    let xx = raw(x, x);
    let yy = raw(y, y);
    // Cross-cluster truncated cells may legitimately read NEGATIVE (the
    // truncated basis carries no sign guarantee) — the cosine reads them
    // too; only a degenerate self-mass denominator refuses.
    if xx <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    acc / (xx * yy).sqrt()
}

/// Raw truncated kernel cell (no normalization): `Σ_{k<d} λ_k^{2τ}·ψ_k(x)·ψ_k(y)
/// / (√π(x)√π(y))` — the 2τ power is the kernel's (the coordinate inner
/// product form: ⟨λ^τψ_x/√π, λ^τψ_y/√π⟩). `stride` = the eigvec row length
/// = n (row-major ψ[k*n + x], recovered from the storage shape).
fn truncated_phi_raw(emb: &DiffusionEmbedding, x: usize, y: usize, tau: usize, d: usize) -> f64 {
    let stride = emb.eigvecs.len() / emb.eigvals.len();
    let k_max = emb.eigvals.len().min(d);
    let mut acc = 0.0_f64;
    for k in 0..k_max {
        let lam = emb.eigvals[k];
        acc += lam.powi((2 * tau) as i32)
            * emb.eigvecs[k * stride + x]
            * emb.eigvecs[k * stride + y]
            * emb.inv_sqrt_deg[x]
            * emb.inv_sqrt_deg[y];
    }
    acc
}

/// T1.6 identity, measured: cosine on `spectral_coordinates::<D>` equals the
/// truncated normalized φ_τ (direct f64 authority) to pinned ε — on BOTH
/// postures D < n (real truncation) and D > n (zero-padded, = the full
/// normalized kernel).
/// The identity arm at ONE compile-time D: cosine on `::<D>` coordinates vs
/// the authority truncated at the same D.
fn identity_arm<const D: usize>(
    emb: &DiffusionEmbedding,
    pairs: &[(usize, usize)],
    taus: &[usize],
) {
    for &tau in taus {
        for &(x, y) in pairs {
            let ex = emb.spectral_coordinates::<D>(x, tau);
            let ey = emb.spectral_coordinates::<D>(y, tau);
            let dot: f64 = ex
                .iter()
                .zip(ey.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let nx: f64 = ex
                .iter()
                .map(|a| (*a as f64) * (*a as f64))
                .sum::<f64>()
                .sqrt();
            let ny: f64 = ey
                .iter()
                .map(|b| (*b as f64) * (*b as f64))
                .sum::<f64>()
                .sqrt();
            let cosine = if nx > 0.0 && ny > 0.0 {
                dot / (nx * ny)
            } else {
                0.0
            };
            let expect = truncated_normalized_phi(emb, x, y, tau, D);
            let err = (cosine - expect).abs();
            assert!(
                err < 1e-4,
                "tau={tau} D={D} pair=({x},{y}): cosine {cosine} vs truncated normalized φ {expect} (err {err})"
            );
        }
    }
}

#[test]
fn t16_identity_cosine_on_e_tau_equals_truncated_normalized_phi() {
    let g = two_clique_graph();
    let emb = DiffusionEmbedding::build(&g);
    let pairs = [(0, 3), (0, 1), (2, 4), (1, 5), (3, 5), (0, 2)];
    let taus = [1usize, 2, 4];
    // D = 4: real truncation (4 < n = 6) — the truncated kernel regime.
    identity_arm::<4>(&emb, &pairs, &taus);
    // D = 8: zero-padded past n = 6 — the full-rank regime.
    identity_arm::<8>(&emb, &pairs, &taus);
}

/// The selector's contract smoke: cross-cluster coverage (one representative
/// per near-identical cluster under the 0.95 colinearity cap), no duplicate
/// slots, byte determinism, the η mass floor refusing everything above any
/// real mass, and sentinel discipline.
#[test]
fn t16_batch_selector_coverage_determinism_and_eta_floor() {
    let g = two_clique_graph();
    let pool: Vec<usize> = (0..6).collect();
    let tau = 2usize;

    // Determinism: two calls, byte-identical out.
    let mut out_a = [u16::MAX; 4];
    let mut out_b = [u16::MAX; 4];
    let k_a = batch_select_diffusion::<4>(&g, &pool, tau, 0.0, &mut out_a);
    let k_b = batch_select_diffusion::<4>(&g, &pool, tau, 0.0, &mut out_b);
    assert_eq!(k_a, k_b, "admitted count must be deterministic");
    assert_eq!(out_a, out_b, "byte-identical picks (Issue-940 tie law)");

    // Coverage: with two near-identical cliques the admitted set spans BOTH
    // (cross-cluster coverage is the operator's whole point; within a clique
    // the 0.95 colinearity cap admits one representative).
    assert!(
        k_a >= 2,
        "expected >= 2 admitted (one per cluster), got {k_a}"
    );
    let mut clusters = [false, false];
    let mut seen = std::collections::HashSet::new();
    for &slot in out_a.iter().take(k_a) {
        assert!(
            slot != u16::MAX,
            "admitted slots must not carry the sentinel"
        );
        assert!(seen.insert(slot), "duplicate pool slot admitted: {slot}");
        let node = pool[slot as usize];
        if node < 3 {
            clusters[0] = true;
        } else {
            clusters[1] = true;
        }
    }
    assert!(
        clusters[0] && clusters[1],
        "both clusters must be represented: {clusters:?}"
    );

    // Sentinel discipline: slots beyond the count stay u16::MAX.
    for &slot in out_a.iter().skip(k_a) {
        assert_eq!(
            slot,
            u16::MAX,
            "slots past the admitted count must be sentinel"
        );
    }

    // η mass floor: a floor above every node's kernel self-mass refuses the
    // whole pool (typed loud behavior: 0 admitted, all sentinel).
    let mut out_floor = [u16::MAX; 4];
    let k_floor = batch_select_diffusion::<4>(&g, &pool, tau, 1.0e6, &mut out_floor);
    assert_eq!(k_floor, 0, "an impossible floor must admit nothing");
    assert!(out_floor.iter().all(|&s| s == u16::MAX));
}

/// The zero-padding posture end-to-end: with BOTH dimensions at or above n
/// the coordinate views are identical (padding carries no information), so
/// the selector's picks match exactly. (Two TRUNCATING dimensions — e.g. 4
/// vs 8 on this 6-node graph — legitimately see different kernels; the
/// coverage smoke above already pins that regime.)
#[test]
fn t16_batch_selector_dimension_transparent() {
    let g = two_clique_graph();
    let pool: Vec<usize> = (0..6).collect();
    let mut out6 = [u16::MAX; 4];
    let mut out8 = [u16::MAX; 4];
    let k6 = batch_select_diffusion::<6>(&g, &pool, 2, 0.0, &mut out6);
    let k8 = batch_select_diffusion::<8>(&g, &pool, 2, 0.0, &mut out8);
    assert_eq!(
        k6, k8,
        "admitted count must not depend on the padded dimension"
    );
    assert_eq!(
        out6, out8,
        "picks must be identical across D when both D >= n"
    );
}
