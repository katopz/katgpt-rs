//! T1.2/T1.3 normalization validation — the F2 core identity on small
//! fixtures: local == spectral at k = n. THROWAWAY validation run for the
//! T1.2 landing commit (the standing battery is Phase 2's F2 fixture file).

use katgpt_core::diversity::diffusion::{
    DiffusionEmbedding, LocalScratch, SimGraph, diffusion_similarity_local,
};

fn check(name: &str, n: usize, edges: &[(usize, usize, f32)], pairs: &[(usize, usize)], taus: &[usize]) {
    let g = SimGraph::build(n, edges);
    let emb = DiffusionEmbedding::build(&g);
    let mut scratch = LocalScratch::new(n);
    let mut worst = 0.0_f64;
    for tau in taus {
        for &(x, y) in pairs {
            let local = diffusion_similarity_local(&g, x, y, *tau, &mut scratch) as f64;
            let spectral = emb.similarity_spectral_exact(x, y, *tau);
            let err = (local - spectral).abs();
            worst = worst.max(err);
            assert!(
                err < 1e-6,
                "{name} FAILED at tau={tau} pair=({x},{y}): local={local} spectral={spectral} err={err}"
            );
        }
    }
    println!("{name}: worst |local-spectral| = {worst:.3e}  OK");
}

#[test]
fn f2_core_identity_small_fixtures() {
    // 6-cycle: every node degree 2, connected, bipartite (λ = -1 present).
    let cycle: Vec<(usize, usize, f32)> = (0..6)
        .map(|i| (i, (i + 1) % 6, 1.0))
        .collect();
    check(
        "6-cycle",
        6,
        &cycle,
        &[(0, 0), (0, 1), (0, 3), (1, 4)],
        &[0, 1, 2, 3],
    );

    // Two triangles joined by one bridge edge (weights differ).
    let mut two_tri = vec![
        (0, 1, 2.0),
        (1, 2, 2.0),
        (0, 2, 2.0),
        (3, 4, 0.5),
        (4, 5, 0.5),
        (3, 5, 0.5),
        (2, 3, 1.0),
    ];
    check(
        "two-triangles-bridge",
        6,
        &two_tri,
        &[(0, 3), (0, 1), (4, 5), (2, 4)],
        &[1, 2, 4],
    );
    let _ = &mut two_tri;

    // Path graph 0-1-2-3-4-5 (a tree — degrees vary 1..2).
    let path: Vec<(usize, usize, f32)> = (0..5).map(|i| (i, i + 1, 1.5)).collect();
    check("path-6", 6, &path, &[(0, 5), (2, 3), (1, 4)], &[1, 2, 3]);

    // K4 clique (complete graph, uniform degree).
    let k4: Vec<(usize, usize, f32)> = (0..4)
        .flat_map(|i| (i + 1..4).map(move |j| (i, j, 1.0)))
        .collect();
    check("k4", 4, &k4, &[(0, 1), (0, 3)], &[1, 2]);
}
