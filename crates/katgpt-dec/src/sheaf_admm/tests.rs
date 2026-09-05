//! Tests for sheaf_admm (extracted from mod.rs by Issue 176).

use super::*;
use crate::operators::graph_laplacian;
use crate::types::{CellComplex, CochainField};

/// `SheafMaps::identity` lays out `[I_{d_e}; 0]` for both endpoints.
#[test]
fn identity_maps_construct_correctly() {
    // 3 vertices, 2 edges: 0-1, 1-2.
    let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
    let maps = SheafMaps::identity(&cx, 4, 2);
    assert_eq!(maps.d_e, 2);
    assert_eq!(maps.d_v, 4);
    assert_eq!(maps.n_edges, 2);
    assert!(maps.is_identity);
    // Expected 2×4 block: [[1,0,0,0],[0,1,0,0]] for both endpoints.
    let expected = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    for e in 0..2 {
        for endpoint in 0..2 {
            let m = maps.edge_map(e, endpoint);
            assert_eq!(
                m,
                expected.as_slice(),
                "identity map (e={e}, endpoint={endpoint}) wrong: {m:?}"
            );
        }
    }
}

/// `SheafMaps::selector` with `[0, 2]` picks dims 0 and 2.
#[test]
fn selector_maps_pick_correct_dims() {
    let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
    let maps = SheafMaps::selector(&cx, 4, &[0, 2]);
    assert_eq!(maps.d_e, 2);
    assert_eq!(maps.d_v, 4);
    assert!(!maps.is_identity, "selector [0,2] should NOT be identity");
    // Row 0 = e_0 = [1,0,0,0], Row 1 = e_2 = [0,0,1,0].
    let expected = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    for e in 0..2 {
        for endpoint in 0..2 {
            let m = maps.edge_map(e, endpoint);
            assert_eq!(
                m,
                expected.as_slice(),
                "selector map (e={e}, endpoint={endpoint}) wrong: {m:?}"
            );
        }
    }
}

/// Selector with `[0, 1, …]` collapses to identity and sets the flag.
#[test]
fn selector_collapses_to_identity_when_ordered() {
    let cx = CellComplex::from_edges(2, &[(0, 1)]);
    let maps = SheafMaps::selector(&cx, 4, &[0, 1]);
    assert!(maps.is_identity, "selector [0,1] should detect identity");
}

/// x-update (DiagonalQuadratic) matches the hand-derived closed form.
#[test]
fn x_update_diagonal_quadratic() {
    // 2 vertices, 1 edge. d_v = d_e = 2 (identity maps, but maps unused by x-update).
    let cx = CellComplex::from_edges(2, &[(0, 1)]);
    let maps = SheafMaps::identity(&cx, 2, 2);
    let mut primal_x = CochainField::zeros(0, 2, 2);
    let mut consensus_z = CochainField::from_vec(0, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let mut dual_u = CochainField::from_vec(0, 2, vec![0.1, 0.2, 0.3, 0.4]);
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0, 1.0, 2.0, 2.0],
        q: vec![0.5, -0.5, 1.0, -1.0],
    };
    let mut scratch = AdmmScratch::new(&cx, 2, 2);
    let rho = 2.0;

    // Snapshot pre-step z (the x-update reads these).
    let z_pre = consensus_z.data.clone();
    let u_pre = dual_u.data.clone();

    sheaf_admm_step(
        &cx,
        &maps,
        &mut primal_x,
        &mut consensus_z,
        &mut dual_u,
        &objective,
        rho,
        0.1,
        1,
        &mut scratch,
    );

    // Expected x_i = (ρ(z-u) - q) / (diag_q + ρ).
    let expected = [
        (rho * (z_pre[0] - u_pre[0]) - 0.5) / (1.0 + rho), // v0d0
        (rho * (z_pre[1] - u_pre[1]) - (-0.5)) / (1.0 + rho), // v0d1
        (rho * (z_pre[2] - u_pre[2]) - 1.0) / (2.0 + rho), // v1d0
        (rho * (z_pre[3] - u_pre[3]) - (-1.0)) / (2.0 + rho), // v1d1
    ];
    for (k, (&got, &exp)) in primal_x.data.iter().zip(&expected).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "x_update v{k}: got {got}, expected {exp}"
        );
    }
}

/// x-update (DiagonalQuadL1) soft-thresholds the quadratic solve.
#[test]
fn x_update_diagonal_quad_l1_soft_thresholds() {
    let cx = CellComplex::from_edges(2, &[(0, 1)]);
    let maps = SheafMaps::identity(&cx, 2, 2);
    let mut primal_x = CochainField::zeros(0, 2, 2);
    let mut consensus_z = CochainField::from_vec(0, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let mut dual_u = CochainField::from_vec(0, 2, vec![0.1, 0.2, 0.3, 0.4]);
    // lambda[0]=2.0 makes v0d0 threshold (2/3 ≈ 0.667) exceed its quad
    // solve (≈0.4333) → result 0. Tests the max(0, ...) zeroing path.
    let objective = LocalObjective::DiagonalQuadL1 {
        diag_q: vec![1.0, 1.0, 2.0, 2.0],
        q: vec![0.5, -0.5, 1.0, -1.0],
        lambda: vec![2.0, 0.1, 0.5, 3.0],
    };
    let mut scratch = AdmmScratch::new(&cx, 2, 2);
    let rho = 2.0;
    let z_pre = consensus_z.data.clone();
    let u_pre = dual_u.data.clone();

    sheaf_admm_step(
        &cx,
        &maps,
        &mut primal_x,
        &mut consensus_z,
        &mut dual_u,
        &objective,
        rho,
        0.1,
        1,
        &mut scratch,
    );

    let xq = |k: usize, dq: f32, lin: f32| (rho * (z_pre[k] - u_pre[k]) - lin) / (dq + rho);
    let xq0 = xq(0, 1.0, 0.5); // ≈ 0.4333
    let xq1 = xq(1, 1.0, -0.5); // ≈ 1.3667
    let xq2 = xq(2, 2.0, 1.0); // = 1.1
    let xq3 = xq(3, 2.0, -1.0); // = 2.05
    let expected = [
        soft_threshold(xq0, 2.0 / 3.0), // thresh 0.667 > 0.4333 → 0
        soft_threshold(xq1, 0.1 / 3.0), // ≈ 1.3333
        soft_threshold(xq2, 0.5 / 4.0), // ≈ 0.975
        soft_threshold(xq3, 3.0 / 4.0), // = 1.3
    ];
    // Sanity: xq0 should indeed be zeroed.
    assert!(
        (expected[0]).abs() < 1e-6,
        "v0d0 should be soft-zeroed, got {}",
        expected[0]
    );
    for (k, (&got, &exp)) in primal_x.data.iter().zip(&expected).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "x_update_l1 v{k}: got {got}, expected {exp}"
        );
    }
}

/// u-update invariant: `u^{k+1} − u^k == x^{k+1} − z^{k+1}` (G2 sanity).
#[test]
fn u_update_accumulates_disagreement() {
    let cx = CellComplex::grid_2d(3, 3);
    let maps = SheafMaps::identity(&cx, 2, 2);
    let total = cx.n_vertices() * 2;
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), 2);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), 2);
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), 2);
    // Deterministic non-trivial initial values.
    for k in 0..total {
        primal_x.data[k] = 0.1 * (k as f32);
        consensus_z.data[k] = 0.05 * (k as f32);
        dual_u.data[k] = 0.01 * (k as f32);
    }
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![0.0; total],
    };
    let mut scratch = AdmmScratch::new(&cx, 2, 2);

    let u_before = dual_u.data.clone();
    sheaf_admm_step(
        &cx,
        &maps,
        &mut primal_x,
        &mut consensus_z,
        &mut dual_u,
        &objective,
        1.0,
        0.1,
        3,
        &mut scratch,
    );

    // Post-step x and z are exactly what the u-update read; the invariant
    // is bit-exact because both sides compute the same expression.
    for (k, ((&u_post, &u_pre), (&x_val, &z_val))) in dual_u
        .data
        .iter()
        .zip(&u_before)
        .zip(primal_x.data.iter().zip(&consensus_z.data))
        .enumerate()
    {
        let du = u_post - u_pre;
        let dxz = x_val - z_val;
        assert!(
            (du - dxz).abs() < 1e-6,
            "u invariant k={k}: du={du}, x-z={dxz}"
        );
    }
}

/// DEC identity: for identity maps with `d_e == d_v`, the sheaf Laplacian
/// via explicit maps equals the graph Laplacian (Research 384 §1.3).
#[test]
fn sheaf_laplacian_identity_matches_graph_laplacian_first_de_dims() {
    let cx = CellComplex::grid_2d(4, 4);
    let d_v = 2;
    let maps = SheafMaps::identity(&cx, d_v, d_v);
    let mut z = CochainField::zeros(0, cx.n_vertices(), d_v);
    // Deterministic non-trivial z.
    for k in 0..z.data.len() {
        z.data[k] = 0.1 * ((k * 13 + 7) as f32).sin().abs();
    }
    let mut scratch = AdmmScratch::new(&cx, d_v, d_v);

    sheaf_laplacian_via_maps(&cx, &maps, &z.data, &mut scratch);
    let gl = graph_laplacian(&cx, &z);

    // f32 accumulation order differs between the two paths; use a loose-but-safe tol.
    for k in 0..z.data.len() {
        assert!(
            (scratch.sheaf_laplacian_z[k] - gl.data[k]).abs() < 1e-4,
            "sheaf_laplacian vs graph_laplacian k={k}: sheaf={}, graph={}",
            scratch.sheaf_laplacian_z[k],
            gl.data[k]
        );
    }
}

/// Smoke test: one ADMM step on a 4×4 grid runs without panic.
#[test]
fn one_admm_step_runs_without_panic() {
    let cx = CellComplex::grid_2d(4, 4);
    let d_v = 4;
    let d_e = 2;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
    for k in 0..total {
        primal_x.data[k] = 0.1 * (k as f32);
    }
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![0.0; total],
    };
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);
    sheaf_admm_step(
        &cx,
        &maps,
        &mut primal_x,
        &mut consensus_z,
        &mut dual_u,
        &objective,
        1.0,
        0.1,
        3,
        &mut scratch,
    );
    // If we get here, no panic. Sanity: primal is finite.
    for v in primal_x.data.iter() {
        assert!(v.is_finite(), "non-finite primal after step");
    }
}

/// Weak G1 preview: K ADMM steps with identity maps reduce the primal
/// max-edge-disagreement (consensus reached).
///
/// Parameters are tuned so the z-projection is near-exact each step (T=50
/// diffusion steps with eta=0.2 drives the non-constant residual to <0.2%
/// on a 4×4 grid), and the local/global balance is well-conditioned
/// (diag_q == rho). A 2-vertex hand-trace confirms geometric convergence
/// (disagreement halves each ADMM step) under these settings.
#[test]
fn identity_maps_reach_consensus() {
    let cx = CellComplex::grid_2d(4, 4);
    let d_v = 2;
    let d_e = 2;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;
    // Balanced local objective with per-vertex distinct preferred targets.
    // q = -target * diag_q ⇒ unconstrained minimizer of f_i is `target`.
    // diag_q == rho ⇒ x-update is a 50/50 blend of (z-u) and target.
    let diag_q_val = 1.0;
    let rho = 1.0;
    let mut target = vec![0.0f32; total];
    for i in 0..cx.n_vertices() {
        for d in 0..d_v {
            target[i * d_v + d] = (0.3 * (i as f32) + 0.7 * (d as f32)) * 0.5;
        }
    }
    let q: Vec<f32> = target.iter().map(|t| -t * diag_q_val).collect();
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![diag_q_val; total],
        q,
    };
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
    // Seed primal with the targets (measures the initial disagreement).
    primal_x.data.copy_from_slice(&target);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

    let d_initial = max_edge_disagreement(&cx, &primal_x);
    // eta = 0.2 keeps diffusion stable (grid Laplacian λ_max ≈ 6.8 on 4×4;
    // 0.2·6.8 ≈ 1.36 < 2). T = 50 drives the z-projection near-exact.
    for _ in 0..30 {
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            rho,
            0.2,
            50,
            &mut scratch,
        );
    }
    let d_final = max_edge_disagreement(&cx, &primal_x);

    eprintln!("identity_maps_reach_consensus: d_initial={d_initial:.5}, d_final={d_final:.5}");
    assert!(
        d_final < d_initial,
        "consensus not reached: d_final {d_final} >= d_initial {d_initial}"
    );
    // Stronger: meaningful reduction (geometric convergence ⇒ near-zero).
    assert!(
        d_final < 0.1 * d_initial,
        "consensus reduction too weak: d_final {d_final} >= 0.1*d_initial {}",
        0.1 * d_initial
    );
}

/// Max over edges and dims of `|x_tail[d] − x_head[d]|` — the identity-map
/// disagreement norm (‖F x‖_∞ for `d_e == d_v`).
fn max_edge_disagreement(cx: &CellComplex, x: &CochainField) -> f32 {
    let dim = x.dim;
    let mut max_d = 0.0f32;
    for pair in cx.boundary_entries(0).as_chunks::<2>().0 {
        let v_tail = pair[0].0;
        let v_head = pair[1].0;
        for d in 0..dim {
            let diff = (x.data[v_tail * dim + d] - x.data[v_head * dim + d]).abs();
            if diff > max_d {
                max_d = diff;
            }
        }
    }
    max_d
}

// ========================================================================
// Plan 407 Phase 3 — T3.2 (selector_per_edge + topk + fast-path)
// ========================================================================

/// `selector_per_edge` builds compact indices for per-edge dim subsets.
#[test]
fn selector_per_edge_construct_correctly() {
    // 3 vertices, 2 edges: 0-1, 1-2.
    let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
    let d_v = 4;
    // Edge 0 selects dims [0, 2], edge 1 selects dims [1, 3].
    let maps = SheafMaps::selector_per_edge(&cx, d_v, &[&[0, 2], &[1, 3]]);
    assert_eq!(maps.d_e, 2);
    assert_eq!(maps.d_v, 4);
    assert_eq!(maps.n_edges, 2);
    assert!(maps.is_selector);
    assert!(
        !maps.is_identity,
        "heterogeneous selectors should not be identity"
    );
    assert!(
        maps.maps.is_empty(),
        "selector maps should not materialize dense maps"
    );
    assert_eq!(maps.selector_indices.len(), 2 * 2 * 2); // n_edges * 2 * d_e
    // Edge 0, endpoint 0 (tail): indices [0, 2].
    assert_eq!(maps.selector_edge_indices(0, 0), &[0, 2]);
    // Edge 0, endpoint 1 (head): same [0, 2] (both endpoints).
    assert_eq!(maps.selector_edge_indices(0, 1), &[0, 2]);
    // Edge 1, endpoint 0: [1, 3].
    assert_eq!(maps.selector_edge_indices(1, 0), &[1, 3]);
}

/// `selector_per_edge` detects identity when all edges pick [0, 1, …, d_e-1].
#[test]
fn selector_per_edge_collapses_to_identity_when_uniform_ordered() {
    let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
    let maps = SheafMaps::selector_per_edge(&cx, 4, &[&[0, 1], &[0, 1]]);
    assert!(
        maps.is_identity,
        "uniform ordered selectors should detect identity"
    );
}

/// `selector_per_edge_topk` picks the top-k dims by score per edge.
#[test]
fn selector_per_edge_topk_picks_highest_scoring_dims() {
    let cx = CellComplex::from_edges(2, &[(0, 1)]);
    let d_v = 4;
    // Scores: dim 3 has highest, dim 1 second. Top-2 should pick [3, 1].
    let scores: &[&[f32]] = &[&[0.1, 0.5, 0.2, 0.9]];
    let maps = SheafMaps::selector_per_edge_topk(&cx, d_v, scores, 2);
    assert_eq!(maps.d_e, 2);
    let indices = maps.selector_edge_indices(0, 0);
    assert_eq!(
        indices,
        &[3, 1],
        "top-2 should be [3, 1] by descending score"
    );
}

/// `selector_per_edge_topk` breaks ties deterministically (lower dim wins).
#[test]
fn selector_per_edge_topk_tie_breaks_by_lower_dim() {
    let cx = CellComplex::from_edges(2, &[(0, 1)]);
    // All scores equal → ties broken by lower dim → picks [0, 1].
    let scores: &[&[f32]] = &[&[0.5, 0.5, 0.5, 0.5]];
    let maps = SheafMaps::selector_per_edge_topk(&cx, 4, scores, 2);
    assert_eq!(maps.selector_edge_indices(0, 0), &[0, 1]);
}

/// Selector fast path: matvec result matches the dense selector matvec
/// bit-for-bit (both compute `L_F z` for the same selector maps). This is
/// the correctness gate for the T3.2 gather-scatter fast path.
#[test]
fn selector_fast_path_matches_dense_selector_matvec() {
    // 4 vertices, 3 edges: 0-1, 1-2, 2-3 (path graph).
    let cx = CellComplex::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
    let d_v = 4;
    let d_e = 2;

    // Dense selector (uniform dims [1, 3] for all edges).
    let dense_maps = SheafMaps::selector(&cx, d_v, &[1, 3]);
    // Compact selector (same dims [1, 3] per edge).
    let compact_maps = SheafMaps::selector_per_edge(&cx, d_v, &[&[1, 3], &[1, 3], &[1, 3]]);

    // Random z.
    let mut z = CochainField::zeros(0, cx.n_vertices(), d_v);
    for k in 0..z.data.len() {
        z.data[k] = (0.1 * (k as f32) + 0.3).sin();
    }

    // Compute L_F z with both paths.
    let mut dense_scratch = AdmmScratch::new(&cx, d_v, d_e);
    sheaf_laplacian_via_maps(&cx, &dense_maps, &z.data, &mut dense_scratch);

    let mut compact_scratch = AdmmScratch::new(&cx, d_v, d_e);
    sheaf_laplacian_via_maps(&cx, &compact_maps, &z.data, &mut compact_scratch);

    // Both must produce the same result (selector maps are mathematically
    // identical; only the storage/compute path differs).
    for k in 0..z.data.len() {
        assert_eq!(
            dense_scratch.sheaf_laplacian_z[k], compact_scratch.sheaf_laplacian_z[k],
            "dense vs compact selector matvec mismatch at k={k}: dense={}, compact={}",
            dense_scratch.sheaf_laplacian_z[k], compact_scratch.sheaf_laplacian_z[k]
        );
    }
}

/// Selector maps reach consensus (full ADMM run with selector_per_edge).
/// Mirrors `identity_maps_reach_consensus` but with per-edge selector maps.
#[test]
fn selector_per_edge_reaches_consensus() {
    let cx = CellComplex::grid_2d(4, 4);
    let d_v = 2;
    let d_e = 2;
    // Uniform selector [0, 1] = identity for all edges (via selector_per_edge).
    let n_edges = cx.n_edges();
    let dims: Vec<&[usize]> = vec![&[0, 1]; n_edges];
    let maps = SheafMaps::selector_per_edge(&cx, d_v, &dims);
    // The maps are identity-flagged (uniform [0,1]), so they'll take the
    // identity fast path — still exercises the selector constructor + ADMM.
    assert!(maps.is_identity);

    let total = cx.n_vertices() * d_v;
    let diag_q_val = 1.0;
    let rho = 1.0;
    let mut target = vec![0.0f32; total];
    for i in 0..cx.n_vertices() {
        for d in 0..d_v {
            target[i * d_v + d] = (0.3 * (i as f32) + 0.7 * (d as f32)) * 0.5;
        }
    }
    let q: Vec<f32> = target.iter().map(|t| -t * diag_q_val).collect();
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![diag_q_val; total],
        q,
    };
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
    primal_x.data.copy_from_slice(&target);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

    let d_initial = max_edge_disagreement(&cx, &primal_x);
    for _ in 0..30 {
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            rho,
            0.2,
            50,
            &mut scratch,
        );
    }
    let d_final = max_edge_disagreement(&cx, &primal_x);
    assert!(
        d_final < 0.1 * d_initial,
        "selector_per_edge consensus failed: d_final={d_final} >= 0.1*d_initial={}",
        0.1 * d_initial
    );
}

/// Heterogeneous selector maps (different dims per edge) still drive
/// consensus on the selected dims. The non-selected dims should retain
/// their disagreement (no coordination).
#[test]
fn selector_per_edge_heterogeneous_drives_partial_consensus() {
    // Path graph 0-1-2. d_v=4, d_e=2.
    // Edge 0 selects dims [0, 1], edge 1 selects dims [2, 3].
    // After ADMM: dims 0,1 agree on edge 0's vertices; dims 2,3 agree on
    // edge 1's vertices. But since edge 0 doesn't coordinate dims 2,3 and
    // edge 1 doesn't coordinate dims 0,1, cross-edge consensus is limited.
    let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
    let d_v = 4;
    let d_e = 2;
    let maps = SheafMaps::selector_per_edge(&cx, d_v, &[&[0, 1], &[2, 3]]);

    let total = cx.n_vertices() * d_v;
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![0.0; total],
    };
    // Initial primal: vertex 0 = [1,1,1,1], v1 = [0,0,0,0], v2 = [1,1,1,1].
    let mut primal_x = CochainField::from_vec(
        0,
        d_v,
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    );
    let mut consensus_z = CochainField::zeros(0, 3, d_v);
    let mut dual_u = CochainField::zeros(0, 3, d_v);
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

    // Run 50 ADMM steps (enough to converge on a 3-vertex path).
    for _ in 0..50 {
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            1.0,
            0.25,
            50,
            &mut scratch,
        );
    }

    // Edge 0 (v0-v1) coordinates dims 0,1 → |x0[0] - x1[0]| should be small.
    let edge0_dim0_diff = (primal_x.data[0] - primal_x.data[4]).abs();
    assert!(
        edge0_dim0_diff < 0.1,
        "edge 0 dim 0 should agree: diff={edge0_dim0_diff}"
    );

    // Edge 1 (v1-v2) coordinates dims 2,3 → |x1[2] - x2[2]| should be small.
    let edge1_dim2_diff = (primal_x.data[4 + 2] - primal_x.data[8 + 2]).abs();
    assert!(
        edge1_dim2_diff < 0.1,
        "edge 1 dim 2 should agree: diff={edge1_dim2_diff}"
    );
}

// ========================================================================
// Plan 407 Phase 3 — T3.1 (conjugate-gradient z-update)
// ========================================================================

/// CG z-update reaches consensus at least as well as GD on identity maps.
#[test]
fn cg_z_update_reaches_consensus() {
    let cx = CellComplex::grid_2d(4, 4);
    let d_v = 2;
    let d_e = 2;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;
    let diag_q_val = 1.0;
    let rho = 1.0;
    let mut target = vec![0.0f32; total];
    for i in 0..cx.n_vertices() {
        for d in 0..d_v {
            target[i * d_v + d] = (0.3 * (i as f32) + 0.7 * (d as f32)) * 0.5;
        }
    }
    let q: Vec<f32> = target.iter().map(|t| -t * diag_q_val).collect();
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![diag_q_val; total],
        q,
    };
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
    primal_x.data.copy_from_slice(&target);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

    let d_initial = max_edge_disagreement(&cx, &primal_x);
    // CG with 20 iters + tight tol.
    for _ in 0..30 {
        sheaf_admm_step_cg_into(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            rho,
            20,
            1e-8,
            &mut scratch,
        );
    }
    let d_final = max_edge_disagreement(&cx, &primal_x);
    eprintln!("cg_z_update_reaches_consensus: d_initial={d_initial:.5}, d_final={d_final:.5}");
    assert!(
        d_final < 0.1 * d_initial,
        "CG consensus failed: d_final={d_final} >= 0.1*d_initial={}",
        0.1 * d_initial
    );
}

/// CG z-update produces a lower-residual projection than GD at the same
/// matvec count, on a graph where CG's convergence advantage is
/// meaningful (a larger grid where GD's `O(κ)` vs CG's `O(√κ)` matters).
#[test]
fn cg_beats_gd_on_residual_at_equal_matvec_count() {
    // 8×8 grid (64 vertices). Condition number κ ≈ λ_max/λ_min ≈
    // 8/(2−2cos(π/8)) ≈ 8/0.152 ≈ 53. CG's √κ ≈ 7.3 vs GD's κ ≈ 53.
    let cx = CellComplex::grid_2d(8, 8);
    let d_v = 1;
    let d_e = 1;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;

    // Objective with a per-vertex preferred target so the primal is
    // non-trivial (non-constant → has components outside ker(L_F)).
    // q = -target, diag_q = 1.0 → x-update = (rho*(z-u) + target) / (1+rho).
    let mut target = vec![0.0f32; total];
    for (k, t) in target.iter_mut().enumerate() {
        *t = (0.1 * (k as f32)).sin();
    }
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: target.iter().map(|t| -t).collect(),
    };

    // Identical non-zero initial state for both paths.
    let make_state = || {
        let mut x = CochainField::zeros(0, cx.n_vertices(), d_v);
        x.data[..total].copy_from_slice(&target[..total]);
        let z = CochainField::zeros(0, cx.n_vertices(), d_v);
        let u = CochainField::zeros(0, cx.n_vertices(), d_v);
        (x, z, u)
    };
    let (mut primal_gd, mut z_gd, mut u_gd) = make_state();
    let (mut primal_cg, mut z_cg, mut u_cg) = make_state();
    let mut scratch_gd = AdmmScratch::new(&cx, d_v, d_e);
    let mut scratch_cg = AdmmScratch::new(&cx, d_v, d_e);

    // One step with GD (T=20 diffusion) vs CG (20 iters, same matvec count).
    sheaf_admm_step_into(
        &cx,
        &maps,
        &mut primal_gd,
        &mut z_gd,
        &mut u_gd,
        &objective,
        1.0,
        0.2,
        20,
        &mut scratch_gd,
    );
    sheaf_admm_step_cg_into(
        &cx,
        &maps,
        &mut primal_cg,
        &mut z_cg,
        &mut u_cg,
        &objective,
        1.0,
        20,
        1e-12,
        &mut scratch_cg,
    );

    // Measure residual ‖L_F z‖ (should be near zero if z is in ker(L_F)).
    let mut scratch_r = AdmmScratch::new(&cx, d_v, d_e);
    sheaf_laplacian_via_maps(&cx, &maps, &z_gd.data, &mut scratch_r);
    let gd_residual: f32 = scratch_r.sheaf_laplacian_z.iter().map(|x| x.abs()).sum();
    sheaf_laplacian_via_maps(&cx, &maps, &z_cg.data, &mut scratch_r);
    let cg_residual: f32 = scratch_r.sheaf_laplacian_z.iter().map(|x| x.abs()).sum();

    eprintln!("cg_beats_gd: gd_residual={gd_residual:.6}, cg_residual={cg_residual:.6}");
    // CG should have a lower residual (better projection).
    assert!(
        cg_residual < gd_residual,
        "CG residual {cg_residual} should be < GD residual {gd_residual}"
    );
}

// ========================================================================
// Plan 407 Phase 3 — T3.3 (soft-constraint variant)
// ========================================================================

/// Soft-constraint with gamma=0 matches the hard-constraint path exactly.
#[test]
fn soft_constraint_gamma_zero_matches_hard() {
    let cx = CellComplex::grid_2d(3, 3);
    let d_v = 2;
    let d_e = 2;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;

    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![-0.5; total],
    };

    // Identical initial state for both paths.
    let init = |x: &mut CochainField| {
        for k in 0..total {
            x.data[k] = (0.1 * (k as f32)).sin();
        }
    };
    let mut x_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
    init(&mut x_hard);
    let mut z_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
    init(&mut z_hard);
    let mut u_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
    init(&mut u_hard);
    let mut x_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
    init(&mut x_soft);
    let mut z_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
    init(&mut z_soft);
    let mut u_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
    init(&mut u_soft);

    let mut scratch_hard = AdmmScratch::new(&cx, d_v, d_e);
    let mut scratch_soft = AdmmScratch::new(&cx, d_v, d_e);

    sheaf_admm_step_into(
        &cx,
        &maps,
        &mut x_hard,
        &mut z_hard,
        &mut u_hard,
        &objective,
        1.0,
        0.2,
        10,
        &mut scratch_hard,
    );
    sheaf_admm_step_soft_into(
        &cx,
        &maps,
        &mut x_soft,
        &mut z_soft,
        &mut u_soft,
        &objective,
        1.0,
        0.2,
        0.0,
        10,
        &mut scratch_soft,
    );

    // Bit-identical: gamma=0 takes the hard path.
    for k in 0..total {
        assert_eq!(x_hard.data[k], x_soft.data[k], "x mismatch at {k}");
        assert_eq!(z_hard.data[k], z_soft.data[k], "z mismatch at {k}");
        assert_eq!(u_hard.data[k], u_soft.data[k], "u mismatch at {k}");
    }
}

/// Soft-constraint with gamma>0 preserves individual variation: the primal
/// retains MORE disagreement than the hard-constraint path after the same
/// number of ADMM steps. The `γ(z − b)` term resists full consensus.
#[test]
fn soft_constraint_gamma_positive_preserves_variation() {
    let cx = CellComplex::grid_2d(4, 4);
    let d_v = 2;
    let d_e = 2;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;

    // Each vertex has a distinct target → the hard path drives all toward
    // consensus, the soft path retains individual variation.
    let mut target = vec![0.0f32; total];
    for i in 0..cx.n_vertices() {
        for d in 0..d_v {
            target[i * d_v + d] = (0.3 * (i as f32) + 0.7 * (d as f32)) * 0.5;
        }
    }
    let q: Vec<f32> = target.iter().map(|t| -t).collect();
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q,
    };

    let init_primal = |x: &mut CochainField| {
        x.data.copy_from_slice(&target);
    };

    let mut x_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
    init_primal(&mut x_hard);
    let mut z_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut u_hard = CochainField::zeros(0, cx.n_vertices(), d_v);

    let mut x_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
    init_primal(&mut x_soft);
    let mut z_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut u_soft = CochainField::zeros(0, cx.n_vertices(), d_v);

    let mut scratch_hard = AdmmScratch::new(&cx, d_v, d_e);
    let mut scratch_soft = AdmmScratch::new(&cx, d_v, d_e);

    for _ in 0..30 {
        sheaf_admm_step_into(
            &cx,
            &maps,
            &mut x_hard,
            &mut z_hard,
            &mut u_hard,
            &objective,
            1.0,
            0.2,
            50,
            &mut scratch_hard,
        );
        sheaf_admm_step_soft_into(
            &cx,
            &maps,
            &mut x_soft,
            &mut z_soft,
            &mut u_soft,
            &objective,
            1.0,
            0.2,
            0.5,
            50,
            &mut scratch_soft,
        );
    }

    let hard_disagree = max_edge_disagreement(&cx, &x_hard);
    let soft_disagree = max_edge_disagreement(&cx, &x_soft);
    eprintln!("soft_preserves_variation: hard={hard_disagree:.6}, soft={soft_disagree:.6}");
    // Soft should retain MORE disagreement (less consensus).
    assert!(
        soft_disagree > hard_disagree,
        "soft constraint should preserve more variation: soft={soft_disagree} > hard={hard_disagree}"
    );
}
