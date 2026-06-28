//! Tropical `(max, +)` Semiring — G1 Non-Redundancy GOAT Gate (Plan 337 Phase 2).
//!
//! Answers the central question from Research 321 §3: **does the tropical
//! `(max, +)` signal carry information that the linear `(ℝ, +, ·)` signal
//! misses on a representative substrate?** If yes by a clear margin AND a
//! product selling point emerges → promote `tropical_algebra` toward
//! default-on. If no → keep opt-in as a curiosity primitive.
//!
//! Unlike Clifford's wedge (mathematically orthogonal to the dot product by
//! construction), the tropical max is NOT mathematically orthogonal to the
//! sum — they are different aggregations of the same data. Non-redundancy is
//! an empirical question. This bench settles it.
//!
//! # Three substrates
//!
//! 1. **DEC game-map cochain** (2D grid 16×16, planted threat hotspot) —
//!    compare `exterior_derivative` (sum-flux) vs `tropical_exterior_derivative`
//!    (max-flux). Metric: do their top-3 edge rankings diverge?
//!    **PASS: ≥1 of 3 differ. STRETCH: ≥2 differ.**
//! 2. **HLA pairs coherence** (8-dim, 64 random NPC pairs) — compare
//!    cosine-similarity coherence (linear) vs tropical-dot coherence (max).
//!    Metric: Spearman rank correlation. **PASS: < 0.85. STRETCH: < 0.7.**
//! 3. **Path bottleneck vs path total** (DEC rank-1 cochain, 10 random paths
//!    on 16×16 grid) — `tropical_line_integral` (bottleneck) vs
//!    `line_integral` (sum). Metric: Spearman rank correlation.
//!    **PASS: < 0.85. STRETCH: < 0.7.**
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features tropical_algebra --bench bench_337_tropical_goat --release -- --nocapture
//! ```

#![cfg(feature = "tropical_algebra")]

use katgpt_core::algebra::tropical::{
    tropical_dot_into, tropical_exterior_derivative, tropical_line_integral,
};
use katgpt_core::dec::{exterior_derivative, line_integral, CellComplex, CochainField};

// ─── Deterministic PRNG (xorshift32) — reproducible across runs ─────────────

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        // Avoid the all-zero state.
        Self(if seed == 0 { 0x9E37_79B9 } else { seed })
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform float in [0, 1).
    #[inline]
    fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / ((1u32 << 24) as f32)
    }
    /// Standard-normal-ish via sum of 12 uniforms (Irwin–Hall, mean 0 var 1).
    #[inline]
    fn gaussian(&mut self) -> f32 {
        let mut s = -6.0f32;
        for _ in 0..12 {
            s += self.uniform();
        }
        s
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Pearson correlation coefficient (also used as Spearman = Pearson on ranks).
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mx: f32 = xs.iter().sum::<f32>() / n;
    let my: f32 = ys.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    cov / (vx.sqrt() * vy.sqrt() + 1e-30)
}

/// Spearman rank correlation: rank both arrays, then Pearson on the ranks.
/// Ties get average rank (fractional).
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    let rx = rank_desc(xs);
    let ry = rank_desc(ys);
    pearson(&rx, &ry)
}

/// Assign ranks in descending order (largest value → rank 1). Ties get the
/// average of the ranks they span.
fn rank_desc(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    // Build index array sorted by value descending.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        // Find the run of ties starting at i.
        let mut j = i + 1;
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // Ties occupy sorted positions [i, j). Average 1-based rank = (i+1 + j) / 2.
        let avg_rank = ((i + 1) + j) as f32 / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Symmetric difference size between two sets (as sorted slices).
fn sym_diff_size(a: &[usize], b: &[usize]) -> usize {
    let mut count = 0;
    let set_a: std::collections::HashSet<usize> = a.iter().copied().collect();
    let set_b: std::collections::HashSet<usize> = b.iter().copied().collect();
    for &x in a {
        if !set_b.contains(&x) {
            count += 1;
        }
    }
    for &x in b {
        if !set_a.contains(&x) {
            count += 1;
        }
    }
    count
}

// ─── Substrate 1: DEC game-map cochain top-3 ranking divergence ────────────

struct Substrate1Result {
    top3_sum: Vec<usize>,
    top3_max: Vec<usize>,
    sym_diff: usize,
    pass: bool,
    stretch: bool,
}

fn run_substrate_1(rng_seed: u32) -> Substrate1Result {
    let cx = CellComplex::grid_2d(16, 16);
    let n_vertices = cx.n_cells(0);
    let n_edges = cx.n_cells(1);

    // Rank-0 vertex "threat field", dim=1. Most vertices random in [0,1).
    // Plant a hotspot at grid position (8,8) = vertex 8*16+8 = 136 with value
    // 100.0, and its 4 grid-neighbors with value 50.0.
    let mut rng = Rng::new(rng_seed);
    let mut threat = CochainField::zeros(0, n_vertices, 1);
    for v in 0..n_vertices {
        threat.data[v] = rng.uniform();
    }
    let hotspot = 8 * 16 + 8;
    threat.data[hotspot] = 100.0;
    // Grid neighbors of (8,8): (7,8), (9,8), (8,7), (8,9).
    threat.data[7 * 16 + 8] = 50.0;
    threat.data[9 * 16 + 8] = 50.0;
    threat.data[8 * 16 + 7] = 50.0;
    threat.data[8 * 16 + 9] = 50.0;

    // Linear: exterior_derivative (sum-flux).
    let sum_flux = exterior_derivative(&cx, &threat);
    // Tropical: tropical_exterior_derivative (max-flux).
    let max_flux = tropical_exterior_derivative(&cx, &threat);

    assert_eq!(sum_flux.data.len(), n_edges);
    assert_eq!(max_flux.data.len(), n_edges);

    // Rank edges by abs(sum-flux) desc → top-3 edge indices.
    let mut sum_ranked: Vec<(usize, f32)> = (0..n_edges)
        .map(|e| (e, sum_flux.data[e].abs()))
        .collect();
    sum_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top3_sum: Vec<usize> = sum_ranked.iter().take(3).map(|(e, _)| *e).collect();

    // Rank edges by abs(max-flux) desc → top-3 edge indices.
    let mut max_ranked: Vec<(usize, f32)> = (0..n_edges)
        .map(|e| (e, max_flux.data[e].abs()))
        .collect();
    max_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top3_max: Vec<usize> = max_ranked.iter().take(3).map(|(e, _)| *e).collect();

    let sd = sym_diff_size(&top3_sum, &top3_max);

    Substrate1Result {
        top3_sum,
        top3_max,
        sym_diff: sd,
        pass: sd >= 1,
        stretch: sd >= 2,
    }
}

// ─── Substrate 2: HLA pairs coherence (mean-cosine vs max-cosine) ───────────

struct Substrate2Result {
    spearman: f32,
    pass: bool,
    stretch: bool,
}

fn run_substrate_2(rng_seed: u32) -> Substrate2Result {
    let n_pairs = 64;
    let dim = 8;
    let mut rng = Rng::new(rng_seed);

    // Generate n_pairs random (source, target) pairs, each 8-dim gaussian.
    let mut sources: Vec<Vec<f32>> = Vec::with_capacity(n_pairs);
    let mut targets: Vec<Vec<f32>> = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let src: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
        let tgt: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
        sources.push(src);
        targets.push(tgt);
    }

    // Mean-cosine coherence (linear baseline): dot/(||s||*||t||).
    let mean_cosines: Vec<f32> = (0..n_pairs)
        .map(|i| {
            let s = &sources[i];
            let t = &targets[i];
            let dot: f32 = s.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
            let ns: f32 = s.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nt: f32 = t.iter().map(|a| a * a).sum::<f32>().sqrt();
            dot / (ns * nt + 1e-30)
        })
        .collect();

    // Max-cosine coherence (tropical): max_k (source[k] + target[k]).
    let max_cosines: Vec<f32> = (0..n_pairs)
        .map(|i| {
            let mut v = f32::NEG_INFINITY;
            tropical_dot_into(&sources[i], &targets[i], &mut v, dim);
            v
        })
        .collect();

    // Spearman rank correlation between the two orderings.
    let sp = spearman(&mean_cosines, &max_cosines);

    Substrate2Result {
        spearman: sp,
        pass: sp < 0.85,
        stretch: sp < 0.7,
    }
}

// ─── Substrate 3: Path bottleneck vs path total ─────────────────────────────

struct Substrate3Result {
    spearman: f32,
    pass: bool,
    stretch: bool,
}

/// Generate `n_paths` random walks on a w×h grid. Each walk starts at a random
/// vertex and takes 4-7 random orthogonal steps (staying in bounds).
fn gen_random_paths(rng: &mut Rng, w: usize, h: usize, n_paths: usize) -> Vec<Vec<u32>> {
    let mut paths = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        // Start at a random interior-ish vertex to avoid immediate edge.
        let mut x = (rng.next_u32() as usize) % w;
        let mut y = (rng.next_u32() as usize) % h;
        let n_steps = 4 + (rng.next_u32() as usize) % 4; // 4..=7 steps → 5..=8 vertices
        let mut path: Vec<u32> = vec![(y * w + x) as u32];
        for _ in 0..n_steps {
            // Pick a random direction (0=up,1=down,2=left,3=right).
            let dir = (rng.next_u32() as usize) % 4;
            let (nx, ny) = match dir {
                0 if y + 1 < h => (x, y + 1),
                1 if y > 0 => (x, y - 1),
                2 if x + 1 < w => (x + 1, y),
                3 if x > 0 => (x - 1, y),
                _ => (x, y), // stay (no valid move in this direction)
            };
            x = nx;
            y = ny;
            path.push((y * w + x) as u32);
        }
        paths.push(path);
    }
    paths
}

fn run_substrate_3(rng_seed: u32) -> Substrate3Result {
    let w = 16;
    let h = 16;
    let cx = CellComplex::grid_2d(w, h);
    let n_edges = cx.n_cells(1);

    // Rank-1 edge cochain, dim=1, random values in [0, 10).
    let mut rng = Rng::new(rng_seed);
    let mut edge_field = CochainField::zeros(1, n_edges, 1);
    for e in 0..n_edges {
        edge_field.data[e] = rng.uniform() * 10.0;
    }

    // 10 random paths, each 5-8 vertices.
    let paths = gen_random_paths(&mut rng, w, h, 10);

    // Linear: line_integral (sum) for each path.
    let linear_scores: Vec<f32> = paths
        .iter()
        .map(|p| line_integral(&cx, &edge_field, p))
        .collect();

    // Tropical: tropical_line_integral (bottleneck/max) for each path.
    let tropical_scores: Vec<f32> = paths
        .iter()
        .map(|p| tropical_line_integral(&cx, &edge_field, p))
        .collect();

    let sp = spearman(&linear_scores, &tropical_scores);

    Substrate3Result {
        spearman: sp,
        pass: sp < 0.85,
        stretch: sp < 0.7,
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 337 — Tropical (max, +) Semiring G1 Non-Redundancy     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let seed: u32 = 0x0337_0001;

    // ── Substrate 1: DEC game-map cochain top-3 divergence ──
    println!("── Substrate 1: DEC game-map cochain (sum-flux vs max-flux top-3) ──");
    let s1 = run_substrate_1(seed);
    println!("    top-3 sum-flux edges:  {:?}", s1.top3_sum);
    println!("    top-3 max-flux edges:  {:?}", s1.top3_max);
    println!("    symmetric difference:  {}", s1.sym_diff);
    println!(
        "    verdict:               {}{}",
        if s1.pass { "PASS" } else { "FAIL" },
        if s1.stretch { " (STRETCH)" } else if s1.pass { "" } else { "" }
    );
    println!(
        "    thresholds:            PASS |Δ△|≥1, STRETCH |Δ△|≥2"
    );
    println!();

    // ── Substrate 2: HLA pairs coherence Spearman ──
    println!("── Substrate 2: HLA pairs coherence (mean-cosine vs max-cosine) ──");
    let s2 = run_substrate_2(seed.wrapping_add(1));
    println!("    Spearman ρ:            {:+.4}", s2.spearman);
    println!(
        "    verdict:               {}{}",
        if s2.pass { "PASS" } else { "FAIL" },
        if s2.stretch { " (STRETCH)" } else if s2.pass { "" } else { "" }
    );
    println!("    thresholds:            PASS ρ<0.85, STRETCH ρ<0.70");
    println!();

    // ── Substrate 3: Path bottleneck vs path total Spearman ──
    println!("── Substrate 3: Path bottleneck vs path total (tropical vs linear) ──");
    let s3 = run_substrate_3(seed.wrapping_add(2));
    println!("    Spearman ρ:            {:+.4}", s3.spearman);
    println!(
        "    verdict:               {}{}",
        if s3.pass { "PASS" } else { "FAIL" },
        if s3.stretch { " (STRETCH)" } else if s3.pass { "" } else { "" }
    );
    println!("    thresholds:            PASS ρ<0.85, STRETCH ρ<0.70");
    println!();

    // ── Summary ──
    let pass_count = [s1.pass, s2.pass, s3.pass].iter().filter(|&&p| p).count();
    let verdict = match pass_count {
        2 | 3 => "PASS",
        1 => "MARGINAL",
        _ => "FAIL",
    };
    println!("════════════════════════════════════════════════════════════════");
    println!(
        "  G1 NON-REDUNDANCY VERDICT: {}/3 substrates PASS  →  {}",
        pass_count, verdict
    );
    println!("    S1 (DEC cochain):    {} (|Δ△|={})",
        if s1.pass { "PASS" } else { "FAIL" }, s1.sym_diff);
    println!("    S2 (HLA pairs):      {} (ρ={:+.4})",
        if s2.pass { "PASS" } else { "FAIL" }, s2.spearman);
    println!("    S3 (path bottleneck):{} (ρ={:+.4})",
        if s3.pass { "PASS" } else { "FAIL" }, s3.spearman);
    if pass_count >= 2 {
        println!("  → Non-redundant. Proceed to Phase 3 (promote toward default).");
    } else if pass_count == 1 {
        println!("  → Marginal. Keep opt-in, document partial result, defer promotion.");
    } else {
        println!("  → Redundant. Keep opt-in curiosity, document negative result.");
    }
    println!("════════════════════════════════════════════════════════════════");

    // Exit code: 0 on PASS (≥2/3), 1 on marginal or fail.
    if pass_count < 2 {
        std::process::exit(1);
    }
}
