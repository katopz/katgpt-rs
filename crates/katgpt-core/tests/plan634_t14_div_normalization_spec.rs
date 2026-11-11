#![cfg(feature = "diffusion_diversity")]
//! Plan 634 T1.4 spec — the `Div_P^τ` normalization law.
//!
//! The governing identity: **μ = π ⇒ Div = 1.0 exactly**, on any graph —
//! `Σ_x π(x)²/π(x) = Σ_x π(x) = 1`, and on a graph where diffusion
//! preserves π (π is P's stationary distribution by construction) the
//! diffused μ stays π. Every f32 value in the two fixtures below is
//! exactly representable, so the pins are `assert_eq!`, no tolerance band.
//!
//! Reproducer of the 2026-11-11 fix: the norm fold's 1/π used
//! `inv_stationary_unnorm · total_weight` = `Σdeg/(2·deg)` — half the
//! normalized weight (`π = deg/Σdeg` sums to 1, so `1/π = Σdeg/deg`).
//! Every emitted Div read exactly 2× the documented formula (the
//! two-node μ=π graph read 2.0). The exact-value consumers at the time
//! (reflex Bench 178, riir-refine Issue 222 T1) were rank-based /
//! report-only, so no recorded verdict changes — the fix lands before
//! any absolute-value consumer pins the inflated scale.

use katgpt_core::diversity::diffusion::SimGraph;

fn div_of(n: usize, edges: &[(usize, usize, f32)], mu: &[f32]) -> f32 {
    let g = SimGraph::build(n, edges);
    let mut q = vec![0.0_f32; n];
    let mut r = vec![0.0_f32; n];
    g.div_tau(mu, 1, &mut q, &mut r)
}

#[test]
fn mu_equal_pi_reads_exactly_one_on_a_two_node_graph() {
    // Single edge: deg = (w, w), π = (0.5, 0.5), μ = π.
    // Diffusion preserves the symmetric μ exactly; every value is
    // exactly representable in f32.
    let div = div_of(2, &[(0, 1, 0.7)], &[0.5, 0.5]);
    assert_eq!(div, 1.0, "μ=π must read Div=1.0 exactly, got {div}");
}

#[test]
fn mu_equal_pi_scale_is_edge_weight_invariant() {
    // The identity must not depend on the edge magnitude (a heavier
    // edge renormalizes π to the same uniform measure).
    let div = div_of(2, &[(0, 1, 3.5)], &[0.5, 0.5]);
    assert_eq!(
        div, 1.0,
        "edge-weight invariance broke the identity, got {div}"
    );
}

#[test]
fn mu_equal_pi_reads_exactly_one_on_disconnected_pairs() {
    // Two independent edges: still regular, π uniform, μ = π = (0.25 ×4).
    // Mass stays within each pair; q remains the uniform vector exactly.
    let div = div_of(4, &[(0, 1, 1.0), (2, 3, 1.0)], &[0.25; 4]);
    assert_eq!(
        div, 1.0,
        "disconnected regular graph broke the identity, got {div}"
    );
}

#[test]
fn hub_concentrated_mu_reads_below_one() {
    // Star K1,4 (center 0, all quarters — exactly representable).
    // μ = e_center: one τ step spreads the mass onto the four
    // low-degree leaves, q = (0, 0.25 ×4);
    // ‖q‖²_{ℓ2(1/π)} = 4·(0.0625/0.125) = 2 ⇒ Div = 0.5. Concentrated
    // intake reads BELOW the π-spread anchor of 1.
    let edges = [
        (0usize, 1usize, 1.0f32),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (0, 4, 1.0),
    ];
    let div = div_of(5, &edges, &[1.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(div, 0.5, "hub-concentrated μ must read 0.5, got {div}");
}

#[test]
fn leaves_concentrated_mu_reads_below_one() {
    // Same star, μ uniform over the leaves: one τ step pulls the mass
    // INTO the high-degree center, q = (1, 0 ×4);
    // ‖q‖²_{ℓ2(1/π)} = 1/0.5 = 2 ⇒ Div = 0.5.
    let edges = [
        (0usize, 1usize, 1.0f32),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (0, 4, 1.0),
    ];
    let div = div_of(5, &edges, &[0.0, 0.25, 0.25, 0.25, 0.25]);
    assert_eq!(div, 0.5, "leaf-uniform μ must read 0.5, got {div}");
}

#[test]
fn zero_mu_still_reads_the_honest_zero() {
    // The zero-μ sentinel (no silent epsilon division) is unchanged.
    let div = div_of(2, &[(0, 1, 1.0)], &[0.0, 0.0]);
    assert_eq!(div, 0.0, "zero μ must read 0.0, got {div}");
}
