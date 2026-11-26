//! Issue 655 G2 — latency ladder for the selection-propagation POC.
//!
//! N=1024 memories, budget k=32 (the issue's perf corner): per-query cost of
//! single-hop vs BFS-decay (shipped defaults + a k_hop sweep) vs propagation
//! (early stop vs forced max_iters). Also reports the average fixpoint
//! iteration count + BFS visited-set size — the "early-stop may be cheaper
//! than O(degree^k) BFS at equal recall" axis.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/k655 cargo bench -p katgpt-core \
//!   --features selection_propagation --bench bench_655_propagation_latency
//! ```

#![cfg(feature = "selection_propagation")]

use katgpt_core::selection_propagation::{
    PropagationConfig, SelectionPropagationScratch, propagate_selection_to_fixpoint_into,
};
use std::hint::black_box;
use std::time::Instant;

const D: usize = 8;

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0f32 / ((1u64 << 24) as f32))
    }
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0f32 * u1.ln()).sqrt();
        r * (2.0f32 * core::f32::consts::PI * u2).cos()
    }
}

/// The G1 fixture shape at the perf corner: 32 chains × (4 + 24 distractors)
/// + 128 background = 1024 nodes.
struct World {
    n: usize,
    embeds: Vec<f32>,
    offsets: Vec<u32>,
    targets: Vec<u32>,
    weights: Vec<f32>,
    queries: Vec<Vec<f32>>,
}

fn build_world(seed: u64) -> World {
    let n_chains = 32usize;
    let chain_len = 4usize;
    let dpc = 24usize;
    let background = 128usize;
    let n = n_chains * (chain_len + dpc) + background;
    let mut rng = Lcg::new(seed);

    let mut embeds = vec![0.0f32; n * D];
    for i in 0..n {
        for j in 0..D {
            embeds[i * D + j] = rng.next_gaussian();
        }
        let norm = embeds[i * D..i * D + D]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if norm > 1e-10 {
            for j in 0..D {
                embeds[i * D + j] /= norm;
            }
        }
    }

    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for c in 0..n_chains {
        let base = c * (chain_len + dpc);
        for k in 0..chain_len - 1 {
            let w = 0.75 + 0.2 * rng.next_f32();
            edges.push((base + k, base + k + 1, w));
        }
        for d in 0..dpc {
            let w = 0.25 + 0.25 * rng.next_f32();
            edges.push((base, base + chain_len + d, w));
        }
        for d in 0..dpc.saturating_sub(1) {
            if rng.next_f32() < 0.35 {
                let w = 0.25 + 0.25 * rng.next_f32();
                edges.push((base + chain_len + d, base + chain_len + d + 1, w));
            }
        }
    }
    for _ in 0..2 * n {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        if a != b {
            let w = 0.05 + 0.25 * rng.next_f32();
            edges.push((a, b, w));
        }
    }

    let mut sym: Vec<(usize, usize, f32)> = edges
        .iter()
        .flat_map(|&(a, b, w)| [(a, b, w), (b, a, w)])
        .collect();
    sym.sort_unstable_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));

    let mut offsets = vec![0u32; n + 1];
    let mut targets = Vec::with_capacity(sym.len());
    let mut weights = Vec::with_capacity(sym.len());
    let mut cur = 0usize;
    for &(s, d, w) in &sym {
        while cur < s {
            cur += 1;
            offsets[cur] = targets.len() as u32;
        }
        targets.push(d as u32);
        weights.push(w);
    }
    offsets[n] = targets.len() as u32;

    let queries = (0..n_chains)
        .map(|c| {
            let head = c * (chain_len + dpc);
            let mut q = vec![0.0f32; D];
            for j in 0..D {
                q[j] = embeds[head * D + j] + 0.15 * rng.next_gaussian();
            }
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut q {
                *x /= norm;
            }
            q
        })
        .collect();

    World {
        n,
        embeds,
        offsets,
        targets,
        weights,
        queries,
    }
}

fn main() {
    let world = build_world(42);
    let n = world.n;
    println!("N = {n} nodes, {} edges (symmetrized)", world.targets.len());

    // ── Seeds (sigmoid(4·dot)) precomputed for every query ──
    let seeds: Vec<Vec<f32>> = world
        .queries
        .iter()
        .map(|q| {
            (0..n)
                .map(|i| {
                    let dot: f32 = q
                        .iter()
                        .zip(&world.embeds[i * D..i * D + D])
                        .map(|(a, b)| a * b)
                        .sum();
                    katgpt_core::sigmoid(4.0 * dot)
                })
                .collect()
        })
        .collect();

    let mut out = vec![0.0f32; n];
    let mut scratch = SelectionPropagationScratch::with_capacity(n, 32);
    let cfg = PropagationConfig::default();
    // The no-early-stop upper bound: same operator, max_iters raised to 64 —
    // shows the membership fixpoint is what keeps the cost down (stable runs
    // halt in a handful of iterations).
    let cfg_bound = PropagationConfig {
        max_iters: 64,
        ..Default::default()
    };

    // ── Propagation: early-stop (membership fixpoint) ──
    let mut total_iters = 0usize;
    let mut stable_count = 0usize;
    let t0 = Instant::now();
    for seed in black_box(&seeds) {
        let o = propagate_selection_to_fixpoint_into(
            &world.offsets,
            &world.targets,
            &world.weights,
            seed,
            n,
            32,
            &cfg,
            &mut out,
            &mut scratch,
        );
        total_iters += o.iters;
        stable_count += o.stable as usize;
    }
    let prop_early_us = t0.elapsed().as_secs_f64() / seeds.len() as f64 * 1e6;
    let avg_iters = total_iters as f64 / seeds.len() as f64;

    // ── Propagation: max_iters=64 upper bound ──
    let t0 = Instant::now();
    for seed in black_box(&seeds) {
        let _ = propagate_selection_to_fixpoint_into(
            &world.offsets,
            &world.targets,
            &world.weights,
            seed,
            n,
            32,
            &cfg_bound,
            &mut out,
            &mut scratch,
        );
    }
    let prop_bound_us = t0.elapsed().as_secs_f64() / seeds.len() as f64 * 1e6;

    // ── Single-hop: dot + top-32 ──
    let mut idx: Vec<usize> = (0..n).collect();
    let t0 = Instant::now();
    for seed in black_box(&seeds) {
        idx.clear();
        idx.extend(0..n);
        idx.sort_by(|a, b| seed[*b].total_cmp(&seed[*a]).then(a.cmp(b)));
        idx.truncate(32);
        black_box(&idx);
    }
    let single_us = t0.elapsed().as_secs_f64() / seeds.len() as f64 * 1e6;

    // ── BFS-decay: shipped defaults (k_hop=2) + k_hop=4 sweep ──
    for k_hop in [2usize, 4] {
        let mut dist = vec![u32::MAX; n];
        let mut fused = vec![0.0f32; n];
        let mut visited_total = 0usize;
        let t0 = Instant::now();
        for seed in black_box(&seeds) {
            // Entity link: top-1.
            let mut seed_node = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (i, &s) in seed.iter().enumerate() {
                if s > best {
                    best = s;
                    seed_node = i;
                }
            }
            // BFS.
            dist.fill(u32::MAX);
            let mut frontier = vec![seed_node];
            let mut next_frontier: Vec<usize> = Vec::new();
            dist[seed_node] = 0;
            for d in 1..=k_hop {
                next_frontier.clear();
                for &node in &frontier {
                    for &t in world.targets
                        [world.offsets[node] as usize..world.offsets[node + 1] as usize]
                        .iter()
                    {
                        let t = t as usize;
                        if dist[t] == u32::MAX {
                            dist[t] = d as u32;
                            next_frontier.push(t);
                        }
                    }
                }
                core::mem::swap(&mut frontier, &mut next_frontier);
                if frontier.is_empty() {
                    break;
                }
            }
            visited_total += dist.iter().filter(|&&d| d != u32::MAX).count();
            // Fuse + top-32.
            fused.copy_from_slice(seed);
            for i in 0..n {
                let d = dist[i];
                if d != u32::MAX && d > 0 {
                    fused[i] += 1.0 / (1.0 + (1.5 * d as f32).exp());
                }
            }
            idx.clear();
            idx.extend(0..n);
            idx.sort_by(|a, b| fused[*b].total_cmp(&fused[*a]).then(a.cmp(b)));
            idx.truncate(32);
            black_box(&idx);
        }
        let bfs_us = t0.elapsed().as_secs_f64() / seeds.len() as f64 * 1e6;
        println!(
            "BFS-decay  k_hop={k_hop}: {bfs_us:8.1} µs/query (avg visited {visited:.0})",
            visited = visited_total as f64 / seeds.len() as f64
        );
    }

    println!("single-hop          : {single_us:8.1} µs/query");
    println!(
        "propagation early   : {prop_early_us:8.1} µs/query (avg {avg_iters:.1} iters, {stable}/{total} stable-fixpoint)",
        stable = stable_count,
        total = seeds.len()
    );
    println!("propagation mi=64   : {prop_bound_us:8.1} µs/query (no-early-stop upper bound)");
}
