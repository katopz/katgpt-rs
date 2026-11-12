//! Plan 634 T2.2 — the F2 cross-check battery: `similarity_local ==
//! similarity_spectral` at **k = n** (the complete eigenbasis — the only
//! regime the equality is exact in) on small fixed graphs (cycles,
//! expanders, clique-graphs, disconnected pairs; n ≤ 64), to pinned ε
//! 1e-6 f64. A divergence names the graph AND node pair (loud, never
//! pooled). The top-|λ| truncation error is a separate REPORTED row
//! (k ∈ {n/2, n/4}) with the tail mass printed — never folded into the
//! equality verdict.
//!
//! Extends (not duplicates) the T1.2 seed file `plan634_t12_*`: new graph
//! shapes, all-pair sweeps at small n, and the truncation report the seed
//! never carried.

#[path = "plan634/mod.rs"]
mod plan634_common;

use katgpt_core::diversity::diffusion::{
    DiffusionEmbedding, LocalScratch, SimGraph, diffusion_similarity_local,
};
use plan634_common::{
    circulant_expander, eigenvalue_tail_mass, planted_clique_graph, truncation_abs_cell_error,
};

const EPS: f64 = 1e-6;

struct NamedGraph {
    name: &'static str,
    g: SimGraph,
}

fn named_graphs() -> Vec<NamedGraph> {
    let cycle8: Vec<(usize, usize, f32)> = (0..8).map(|i| (i, (i + 1) % 8, 1.0)).collect();
    let path16: Vec<(usize, usize, f32)> = (0..15).map(|i| (i, i + 1, 0.75)).collect();
    let star8: Vec<(usize, usize, f32)> = (1..8).map(|j| (0, j, 1.25)).collect();
    // Two disconnected K5s — the zero-crossing edge case (a pair across
    // components reads 0.0 in BOTH postures).
    let mut two_k5: Vec<(usize, usize, f32)> = (0..5)
        .flat_map(|i| (i + 1..5).map(move |j| (i, j, 1.0)))
        .collect();
    two_k5.extend((5..10).flat_map(|i| (i + 1..10).map(move |j| (i, j, 1.0))));
    // Weighted clique-cluster (the F1 two-cluster seed, weights differing).
    let two_clique = vec![
        (0, 1, 2.0),
        (1, 2, 2.0),
        (0, 2, 2.0),
        (3, 4, 0.5),
        (4, 5, 0.5),
        (3, 5, 0.5),
        (2, 3, 1.0),
    ];
    vec![
        NamedGraph {
            name: "cycle-8",
            g: SimGraph::build(8, &cycle8),
        },
        NamedGraph {
            name: "path-16",
            g: SimGraph::build(16, &path16),
        },
        NamedGraph {
            name: "star-8",
            g: SimGraph::build(8, &star8),
        },
        NamedGraph {
            name: "two-k5-disconnected",
            g: SimGraph::build(10, &two_k5),
        },
        NamedGraph {
            name: "two-clique-bridge",
            g: SimGraph::build(6, &two_clique),
        },
        NamedGraph {
            name: "expander-c48",
            g: circulant_expander(48, &[1, 2, 4]),
        },
        {
            let (g, _, _, _, _) = planted_clique_graph(6, 5);
            NamedGraph {
                name: "planted-K6-M5",
                g,
            }
        },
    ]
}

/// All-pairs at n ≤ 16, every (x+y) % stride == 0 pair above — the sweep
/// stays O(n²) small while covering intra-, cross-, and same-node pairs.
fn pair_plan(n: usize) -> Vec<(usize, usize)> {
    let stride = if n <= 16 { 1 } else { 7 };
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for x in 0..n {
        for y in 0..n {
            if (x + y) % stride == 0 || x == y {
                pairs.push((x, y));
            }
        }
    }
    pairs.shrink_to_fit();
    pairs
}

#[test]
fn t22_f2_local_spectral_equality_named_battery() {
    let taus = [0_usize, 1, 2, 3];
    for NamedGraph { name, g } in named_graphs() {
        let n = g.n;
        let emb = DiffusionEmbedding::build(&g);
        let mut scratch = LocalScratch::new(n);
        let mut worst = (0.0_f64, 0_usize, 0_usize, 0_usize);
        for tau in taus {
            for &(x, y) in &pair_plan(n) {
                let local = diffusion_similarity_local(&g, x, y, tau, &mut scratch) as f64;
                let spectral = emb.similarity_spectral_exact(x, y, tau);
                let err = (local - spectral).abs();
                if err > worst.0 {
                    worst = (err, tau, x, y);
                }
                assert!(
                    err < EPS,
                    "{name} FAILED at tau={tau} pair=({x},{y}): local={local:.10} \
                     spectral={spectral:.10} err={err:.3e}"
                );
            }
        }
        println!(
            "{name}: n={n}, worst |local−spectral| = {:.3e} (tau={} pair=({},{})) OK",
            worst.0, worst.1, worst.2, worst.3
        );
    }
}

#[test]
fn t22_f2_truncation_reported_rows() {
    // REPORTED, never a bar: the top-|λ| truncation error at k ∈ {n/2, n/4}
    // with the dropped tail mass `Σ_{i>k} λ_i^{2τ}ψψ/(√π√π)` relative to the
    // full cell, printed per graph — the consumer-facing D choice reads
    // this table, the equality verdict above never sees it.
    let taus = [1_usize, 2];
    for NamedGraph { name, g } in named_graphs() {
        let n = g.n;
        let emb = DiffusionEmbedding::build(&g);
        // Sample a representative pair set: same-node + the first node of
        // each distinct component class + cross pairs at stride.
        let pairs: Vec<(usize, usize)> = (0..n)
            .step_by((n / 8).max(1))
            .flat_map(|x| [(x, x), (x, (x + 1) % n), (x, n / 2)])
            .collect();
        for tau in taus {
            for &k in &[n / 2, n / 4] {
                // The eigenvalue-level tail (the plan's literal statistic)
                // + the worst ABSOLUTE cell error (never a ratio: the
                // kernel's near-zero cells make relative errors explode).
                let (tail, total) = eigenvalue_tail_mass(&emb, tau, k);
                let share = if total > 0.0 { tail / total } else { 0.0 };
                let mut worst_cell = 0.0_f64;
                for &(x, y) in &pairs {
                    worst_cell = worst_cell.max(truncation_abs_cell_error(&emb, x, y, tau, k));
                }
                println!(
                    "  truncation {name} τ={tau} k={k}/{n}: tail mass {tail:.3e}/{total:.3e} \
                     (share {share:.3e}), worst |φ_full−φ_trunc| {worst_cell:.3e}"
                );
            }
        }
    }
}
