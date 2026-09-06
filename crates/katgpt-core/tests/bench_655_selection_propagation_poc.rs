//! Issue 655 / Research 483 — selection-set fixpoint propagation POC
//! (defend-wrong §3.6 head-to-head).
//!
//! **Falsifiable claim:** a query-seeded importance propagation iterated
//! until the *selected set* stabilizes (KEEP M3 / HippoRAG's PPR class)
//! **beats** the shipped BFS k-hop + inverse-sigmoid hop-decay traversal
//! (`KgTripleIndex::k_hop_neighbors` + `riir-rag fuse_graph_candidates`) on
//! multi-hop chain recall at equal selection budget, and beats single-hop
//! trivially.
//!
//! # Competitors (equal selection budget `k`)
//!
//! 1. **Single-hop** — top-k by query similarity. The engram latent-lookup
//!    shape.
//! 2. **BFS-decay** — the shipped `fuse_graph_candidates` shape: top-1
//!    entity linking (the query mentions one entity — the KEEP door), BFS
//!    over the symmetrized adjacency (k_hop_neighbors unions spo+osp),
//!    `graph_score = 1/(1+exp(λ·d))` with the shipped defaults
//!    `k_hop=2, λ=1.5`, fused as `query_sim + graph_score` (the packer's
//!    unweighted sum), distance-0 skipped.
//! 3. **Propagation (Mass)** — `propagate_selection_to_fixpoint_into` with
//!    `PropagationBlend::Mass` (PPR-style, the primary arm).
//! 4. **Propagation (Mean)** — the literal KEEP `edge_avg` shape, kept so
//!    its single-supporter weight-cancellation degeneracy is MEASURED, not
//!    assumed.
//!
//! # Toy domain (T1)
//!
//! Synthetic memory chains in the KEEP Fig-6 shape: the ground-truth action
//! requires a 2-3-hop chain (*locked door → key → table*). Per chain: a head
//! node the query is similar to, `hop` tail nodes with their OWN random
//! embeddings (invisible to the query — single-hop cannot find them), chain
//! edges at high reliability `U[0.75, 0.95]`, distractors calibrated to HIGH
//! query-similarity but ZERO chain-relevance attached to the head at LOW
//! reliability `U[0.25, 0.5]` (the case BFS-decay under-ranks — it cannot see
//! edge weights), plus background noise edges/nodes.
//!
//! # Metric
//!
//! Chain-recall@k = |top-k ∩ chain| / |chain| (full chain, head included) +
//! tail-recall@k (excluding the head — the part ONLY a graph method can find).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/k655 cargo test -p katgpt-core \
//!   --features selection_propagation --test bench_655_selection_propagation_poc -- --nocapture
//! ```

#![cfg(feature = "selection_propagation")]

use katgpt_core::selection_propagation::{
    PropagationBlend, PropagationConfig, SelectionPropagationScratch,
    propagate_selection_to_fixpoint_into,
};

// ── Deterministic LCG PRNG (mirrors tests/set_attention_clr_weighted_g8.rs) ─

struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform [0, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        ((u >> 40) as f32) * (1.0f32 / ((1u64 << 24) as f32))
    }

    /// Uniform [lo, hi).
    #[inline]
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Standard normal via Box-Muller.
    #[inline]
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * core::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ── T1: the chain-corpus generator ─────────────────────────────────────────

/// Embedding dimension (the house 8-D latent shape).
const D: usize = 8;

struct ChainWorldConfig {
    /// Planted chains.
    n_chains: usize,
    /// Nodes per chain INCLUDING the head (= hop distance + 1).
    chain_len: usize,
    /// Calibrated distractors per chain (high query-sim, zero chain relevance).
    distractors_per_chain: usize,
    /// Background filler nodes (random embeddings, noise edges only).
    background: usize,
    /// Low-reliability noise edges per node (background graph realism).
    noise_edges_per_node: usize,
    seed: u64,
}

impl ChainWorldConfig {
    fn n(&self) -> usize {
        self.n_chains * (self.chain_len + self.distractors_per_chain) + self.background
    }
}

struct ChainQuery {
    /// Query embedding (≈ head + noise).
    query: Vec<f32>,
    /// Ground-truth chain node indices, head first.
    chain: Vec<usize>,
}

struct ChainWorld {
    n: usize,
    /// Row-major n×D embeddings (unit-normalized).
    embeds: Vec<f32>,
    /// Symmetrized CSR adjacency (chain + distractor + noise edges, both
    /// directions — k_hop_neighbors unions spo+osp, so all three competitors
    /// consume the same symmetrized graph).
    offsets: Vec<u32>,
    targets: Vec<u32>,
    weights: Vec<f32>,
    queries: Vec<ChainQuery>,
}

const CHAIN_EDGE_W: (f32, f32) = (0.75, 0.95);
const DISTRACTOR_EDGE_W: (f32, f32) = (0.25, 0.5);
const NOISE_EDGE_W: (f32, f32) = (0.05, 0.3);

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn build_world(cfg: &ChainWorldConfig) -> ChainWorld {
    let n = cfg.n();
    let mut rng = Lcg::new(cfg.seed);

    // Layout: chain c occupies nodes [c*(chain_len+dpc), c*(chain_len+dpc)
    // + chain_len) and its distractors the next dpc slots; background last.
    let mut embeds = vec![0.0f32; n * D];
    let mut chains: Vec<Vec<usize>> = Vec::with_capacity(cfg.n_chains);
    let mut chain_dirs: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_chains);

    for c in 0..cfg.n_chains {
        let base = c * (cfg.chain_len + cfg.distractors_per_chain);
        // The chain's own direction — the query lives near it.
        let mut dir = vec![0.0f32; D];
        for x in dir.iter_mut() {
            *x = rng.next_gaussian();
        }
        normalize(&mut dir);
        chain_dirs.push(dir.clone());

        // Head: strongly aligned with the chain direction.
        let head = base;
        for j in 0..D {
            embeds[head * D + j] = dir[j] + 0.05 * rng.next_gaussian();
        }
        normalize(&mut embeds[head * D..head * D + D]);

        // Tail nodes: their OWN random directions — invisible to the query
        // (this is what makes the task multi-hop: single-hop cannot find them
        // by similarity).
        for k in 1..cfg.chain_len {
            let idx = base + k;
            for j in 0..D {
                embeds[idx * D + j] = rng.next_gaussian();
            }
            normalize(&mut embeds[idx * D..idx * D + D]);
        }
        chains.push((0..cfg.chain_len).map(|k| base + k).collect());

        // Distractors: calibrated high query-similarity (near the chain
        // direction, weaker than the head), zero chain relevance — they hang
        // off the HEAD at low reliability.
        for d in 0..cfg.distractors_per_chain {
            let idx = base + cfg.chain_len + d;
            for j in 0..D {
                embeds[idx * D + j] = dir[j] + 0.35 * rng.next_gaussian();
            }
            normalize(&mut embeds[idx * D..idx * D + D]);
        }
    }

    // Background: fully random.
    for b in 0..cfg.background {
        let idx = cfg.n_chains * (cfg.chain_len + cfg.distractors_per_chain) + b;
        for j in 0..D {
            embeds[idx * D + j] = rng.next_gaussian();
        }
        normalize(&mut embeds[idx * D..idx * D + D]);
    }

    // ── Edges ──
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for c in 0..cfg.n_chains {
        let base = c * (cfg.chain_len + cfg.distractors_per_chain);
        // Chain: m_k -> m_{k+1}, high reliability.
        for k in 0..cfg.chain_len - 1 {
            edges.push((base + k, base + k + 1, rng.next_range(CHAIN_EDGE_W.0, CHAIN_EDGE_W.1)));
        }
        // Head -> distractors, low reliability.
        for d in 0..cfg.distractors_per_chain {
            edges.push((
                base,
                base + cfg.chain_len + d,
                rng.next_range(DISTRACTOR_EDGE_W.0, DISTRACTOR_EDGE_W.1),
            ));
        }
        // A few distractor↔distractor edges (low reliability) so the
        // distractor cluster is a connected blob a uniform BFS happily floods.
        for d in 0..cfg.distractors_per_chain.saturating_sub(1) {
            if rng.next_f32() < 0.35 {
                edges.push((
                    base + cfg.chain_len + d,
                    base + cfg.chain_len + d + 1,
                    rng.next_range(DISTRACTOR_EDGE_W.0, DISTRACTOR_EDGE_W.1),
                ));
            }
        }
    }
    // Background noise edges: random pairs, lowest reliability.
    let noise = cfg.noise_edges_per_node * n;
    for _ in 0..noise {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        if a != b {
            edges.push((a, b, rng.next_range(NOISE_EDGE_W.0, NOISE_EDGE_W.1)));
        }
    }

    // Symmetrize (k_hop_neighbors unions outgoing + incoming).
    let mut sym: Vec<(usize, usize, f32)> = edges
        .iter()
        .flat_map(|&(a, b, w)| [(a, b, w), (b, a, w)])
        .collect();
    // CSR build: sort by source (then destination) for a deterministic
    // summation order.
    sym.sort_unstable_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));

    let mut offsets = vec![0u32; n + 1];
    let mut targets = Vec::with_capacity(sym.len());
    let mut weights = Vec::with_capacity(sym.len());
    let mut cur_src = 0usize;
    for &(s, d, w) in &sym {
        while cur_src < s {
            cur_src += 1;
            offsets[cur_src] = targets.len() as u32;
        }
        targets.push(d as u32);
        weights.push(w);
    }
    offsets[n] = targets.len() as u32;

    // ── Queries: one per chain, ≈ head + noise ──
    let queries = chains
        .iter()
        .map(|chain| {
            let head = chain[0];
            let mut q = vec![0.0f32; D];
            for j in 0..D {
                q[j] = embeds[head * D + j] + 0.15 * rng.next_gaussian();
            }
            normalize(&mut q);
            ChainQuery { query: q, chain: chain.clone() }
        })
        .collect();

    ChainWorld { n, embeds, offsets, targets, weights, queries }
}

// ── Competitor 1: single-hop ───────────────────────────────────────────────

/// query_sim[i] = dot(query, embed_i). Top-k (ties by ascending index).
fn query_sims(world: &ChainWorld, query: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.reserve(world.n);
    for i in 0..world.n {
        out.push(dot(query, &world.embeds[i * D..i * D + D]));
    }
}

/// Top-k indices by score (score desc, index asc). Caller-owned `scratch`.
fn top_k_indices(scores: &[f32], k: usize, scratch: &mut Vec<usize>) {
    scratch.clear();
    scratch.extend(0..scores.len());
    // Partial selection: nth_element-style via sort (n ≤ ~1k in this POC).
    scratch.sort_by(|&a, &b| {
        scores[b].total_cmp(&scores[a])
            .then(a.cmp(&b))
    });
    scratch.truncate(k);
}

// ── Competitor 2: BFS-decay (the shipped fuse_graph_candidates shape) ──────

/// Multi-source BFS over the symmetrized CSR adjacency. Returns the min hop
/// distance per node into `dist` (u32::MAX = unreached). Caller-owned.
fn bfs_distances(world: &ChainWorld, seeds: &[usize], k_hop: usize, dist: &mut Vec<u32>) {
    dist.clear();
    dist.resize(world.n, u32::MAX);
    let mut frontier: Vec<usize> = Vec::with_capacity(world.n);
    let mut next_frontier: Vec<usize> = Vec::with_capacity(world.n);
    for &s in seeds {
        dist[s] = 0;
        frontier.push(s);
    }
    for d in 1..=k_hop {
        next_frontier.clear();
        for &node in &frontier {
            for e in world.offsets[node] as usize..world.offsets[node + 1] as usize {
                let t = world.targets[e] as usize;
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
}

/// The shipped fusion: `query_sim + 1/(1+exp(λ·d))`, distance-0 skipped
/// (fuse_graph_candidates `continue`s on d==0 — the query entity itself is
/// already covered by the latent stage). Top-k of the fused score; the
/// selection is left in `idx_scratch` (truncated to k).
#[allow(clippy::too_many_arguments)] // harness plumbing — scratch buffers in, selection out
fn bfs_decay_select(
    world: &ChainWorld,
    query: &[f32],
    k: usize,
    k_hop: usize,
    lambda: f32,
    n_seeds: usize,
    sims_scratch: &mut Vec<f32>,
    idx_scratch: &mut Vec<usize>,
    dist_scratch: &mut Vec<u32>,
    fused_scratch: &mut Vec<f32>,
) {
    query_sims(world, query, sims_scratch);
    top_k_indices(sims_scratch, n_seeds, idx_scratch);
    let seeds: Vec<usize> = idx_scratch.clone();
    bfs_distances(world, &seeds, k_hop, dist_scratch);
    fused_scratch.clear();
    fused_scratch.extend_from_slice(sims_scratch);
    for i in 0..world.n {
        let d = dist_scratch[i];
        if d != u32::MAX && d > 0 {
            fused_scratch[i] += 1.0 / (1.0 + (lambda * d as f32).exp());
        }
    }
    top_k_indices(fused_scratch, k, idx_scratch);
}

// ── Competitor 3/4: propagation (Mass + Mean) ──────────────────────────────

#[allow(clippy::too_many_arguments)] // harness plumbing — scratch buffers in, selection out
fn propagation_select(
    world: &ChainWorld,
    query: &[f32],
    k: usize,
    blend: PropagationBlend,
    seed_beta: f32,
    prop_scratch: &mut SelectionPropagationScratch,
    scores_scratch: &mut Vec<f32>,
    idx_scratch: &mut Vec<usize>,
    out: &mut Vec<usize>,
) -> usize {
    scores_scratch.clear();
    scores_scratch.reserve(world.n);
    for i in 0..world.n {
        let s = dot(query, &world.embeds[i * D..i * D + D]);
        scores_scratch.push(katgpt_core::sigmoid(seed_beta * s));
    }
    let mut final_scores = vec![0.0f32; world.n];
    let cfg = PropagationConfig { blend, ..Default::default() };
    let outcome = propagate_selection_to_fixpoint_into(
        &world.offsets,
        &world.targets,
        &world.weights,
        scores_scratch,
        world.n,
        k,
        &cfg,
        &mut final_scores,
        prop_scratch,
    );
    top_k_indices(&final_scores, k, idx_scratch);
    out.clear();
    out.extend_from_slice(idx_scratch);
    outcome.iters
}

// ── Metric ──────────────────────────────────────────────────────────────────

/// recall of `selected` over `truth`. `k` = len(selected).
fn recall_at(selected: &[usize], truth: &[usize]) -> f64 {
    if truth.is_empty() {
        return 0.0;
    }
    let hits = selected.iter().filter(|s| truth.contains(s)).count();
    hits as f64 / truth.len() as f64
}

/// Tail truth = chain minus the head.
fn tail_truth(chain: &[usize]) -> &[usize] {
    &chain[1..]
}

// ── One instance = one (world, query, budget) cell ─────────────────────────

struct SelectorStats {
    chain_recall: f64,
    tail_recall: f64,
}

fn eval_single_hop(
    world: &ChainWorld,
    qi: usize,
    k: usize,
    sims: &mut Vec<f32>,
    idx: &mut Vec<usize>,
) -> SelectorStats {
    query_sims(world, &world.queries[qi].query, sims);
    top_k_indices(sims, k, idx);
    SelectorStats {
        chain_recall: recall_at(idx, &world.queries[qi].chain),
        tail_recall: recall_at(idx, tail_truth(&world.queries[qi].chain)),
    }
}

fn eval_bfs_decay(
    world: &ChainWorld,
    qi: usize,
    k: usize,
    sims: &mut Vec<f32>,
    idx: &mut Vec<usize>,
    dist: &mut Vec<u32>,
    fused: &mut Vec<f32>,
) -> SelectorStats {
    // Shipped defaults: k_hop=2, λ=1.5 (GraphRagConfig::default), 1 query
    // entity (the query mentions one entity — the KEEP door).
    bfs_decay_select(world, &world.queries[qi].query, k, 2, 1.5, 1, sims, idx, dist, fused);
    SelectorStats {
        chain_recall: recall_at(idx, &world.queries[qi].chain),
        tail_recall: recall_at(idx, tail_truth(&world.queries[qi].chain)),
    }
}

fn eval_propagation(
    world: &ChainWorld,
    qi: usize,
    k: usize,
    blend: PropagationBlend,
    prop_scratch: &mut SelectionPropagationScratch,
    scores: &mut Vec<f32>,
    idx: &mut Vec<usize>,
) -> SelectorStats {
    let mut out: Vec<usize> = Vec::new();
    let _iters = propagation_select(
        world, &world.queries[qi].query, k, blend, 4.0, prop_scratch, scores, idx, &mut out,
    );
    SelectorStats {
        chain_recall: recall_at(&out, &world.queries[qi].chain),
        tail_recall: recall_at(&out, tail_truth(&world.queries[qi].chain)),
    }
}

// ── The G1 sweep ────────────────────────────────────────────────────────────

struct Cell {
    hop: usize,
    distractors: usize,
    k: usize,
}

fn run_cell(
    cell: &Cell,
    seeds: &[u64],
    cfg_base: &ChainWorldConfig,
) -> [f64; 8] {
    // Returns [single_chain, single_tail, bfs_chain, bfs_tail, mass_chain,
    // mass_tail, mean_chain, mean_tail] averaged over all instances.
    let mut acc = [0.0f64; 8];
    let mut instances = 0usize;
    for &seed in seeds {
        let cfg = ChainWorldConfig {
            n_chains: cfg_base.n_chains,
            chain_len: cell.hop + 1,
            distractors_per_chain: cell.distractors,
            background: cfg_base.background,
            noise_edges_per_node: cfg_base.noise_edges_per_node,
            seed,
        };
        let world = build_world(&cfg);
        let mut sims = Vec::new();
        let mut idx = Vec::new();
        let mut dist = Vec::new();
        let mut fused = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut prop_scratch = SelectionPropagationScratch::new();

        for qi in 0..world.queries.len() {
            let s = eval_single_hop(&world, qi, cell.k, &mut sims, &mut idx);
            acc[0] += s.chain_recall;
            acc[1] += s.tail_recall;
            let b = eval_bfs_decay(&world, qi, cell.k, &mut sims, &mut idx, &mut dist, &mut fused);
            acc[2] += b.chain_recall;
            acc[3] += b.tail_recall;
            let m = eval_propagation(&world, qi, cell.k, PropagationBlend::Mass, &mut prop_scratch, &mut scores, &mut idx);
            acc[4] += m.chain_recall;
            acc[5] += m.tail_recall;
            let n = eval_propagation(&world, qi, cell.k, PropagationBlend::Mean, &mut prop_scratch, &mut scores, &mut idx);
            acc[6] += n.chain_recall;
            acc[7] += n.tail_recall;
            instances += 1;
        }
    }
    for a in &mut acc {
        *a /= instances as f64;
    }
    acc
}



/// Full sweep: hop × distractors × budget. Prints the verdict table.
fn run_sweep() -> Vec<(Cell, [f64; 8])> {
    let hops = [1usize, 2, 3, 4];
    let distractor_levels = [4usize, 12, 24];
    let budgets = [4usize, 8, 16, 32];
    let seeds = [42u64, 123, 456, 789, 1024];
    let cfg_base = ChainWorldConfig {
        n_chains: 12,
        chain_len: 0, // per-cell
        distractors_per_chain: 0,
        background: 32,
        noise_edges_per_node: 2,
        seed: 0,
    };

    let mut rows = Vec::with_capacity(hops.len());
    eprintln!("hop | distr |  k | single-hop chain/tail | BFS-decay chain/tail | prop(Mass) chain/tail | prop(Mean) chain/tail");
    eprintln!("----|-------|----|----------------------|----------------------|----------------------|--------------------");
    for &hop in &hops {
        for &dr in &distractor_levels {
            for &k in &budgets {
                let cell = Cell { hop, distractors: dr, k };
                let acc = run_cell(&cell, &seeds, &cfg_base);
                eprintln!(
                    " {hop}  | {dr:5} | {k:2} |   {0:.3} / {1:.3}      |   {2:.3} / {3:.3}      |   {4:.3} / {5:.3}      |   {6:.3} / {7:.3}",
                    acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]
                );
                rows.push((cell, acc));
            }
        }
    }
    rows.shrink_to_fit();
    rows
}

// ── Gates ───────────────────────────────────────────────────────────────────

/// G1 (load-bearing): propagation(Mass) ≥ BFS-decay on ≥2-hop chains, both
/// chain- and tail-recall, averaged over every distractor × budget cell; and
/// propagation strictly beats single-hop on ≥2-hop tail recall (the part only
/// a graph method can find).
#[test]
fn g1_propagation_beats_bfs_decay_on_multihop() {
    let rows = run_sweep();

    let mut bfs_wins = 0usize;
    let mut ties = 0usize;
    let mut losses = 0usize;
    let mut means = [0.0f64; 8];
    let mut cells_h2 = 0usize;
    for (cell, acc) in &rows {
        if cell.hop >= 2 {
            for (m, a) in means.iter_mut().zip(acc.iter()) {
                *m += a;
            }
            cells_h2 += 1;
            match acc[4].total_cmp(&acc[2]) {
                core::cmp::Ordering::Greater => bfs_wins += 1,
                core::cmp::Ordering::Equal => ties += 1,
                core::cmp::Ordering::Less => losses += 1,
            }
        }
    }
    for m in &mut means {
        *m /= cells_h2 as f64;
    }
    eprintln!(
        "G1 h≥2 means: single {0:.3}/{1:.3} | bfs {2:.3}/{3:.3} | mass {4:.3}/{5:.3} | mean {6:.3}/{7:.3} (chain/tail)",
        means[0], means[1], means[2], means[3], means[4], means[5], means[6], means[7]
    );
    eprintln!(
        "G1 h≥2 per-cell vs BFS-decay (chain recall): prop wins {bfs_wins}, ties {ties}, losses {losses}"
    );

    assert!(
        means[4] >= means[2] - 1e-9,
        "G1 FAIL: propagation(Mass) chain recall {0:.3} < BFS-decay {1:.3} on h≥2",
        means[4], means[2]
    );
    assert!(
        means[5] >= means[3] - 1e-9,
        "G1 FAIL: propagation(Mass) tail recall {0:.3} < BFS-decay {1:.3} on h≥2",
        means[5], means[3]
    );
    assert!(
        means[5] > means[1],
        "G1 FAIL: propagation(Mass) tail recall {0:.3} does not beat single-hop {1:.3} on h≥2",
        means[5], means[1]
    );
    // Honest reporting: every per-cell loss is printed above (losses count).
}

/// Control: 1-hop + zero-distractor pressure — the regime where BFS-decay has
/// everything it needs (no distractors to under-rank against). Propagation
/// must TIE within noise, not lose.
#[test]
fn g1_control_one_hop_and_no_distractors_ties() {
    let seeds = [42u64, 123, 456];
    let cfg_base = ChainWorldConfig {
        n_chains: 12,
        chain_len: 0,
        distractors_per_chain: 0,
        background: 32,
        noise_edges_per_node: 2,
        seed: 0,
    };

    // 1-hop, low distractors (4), mid budget. The prediction is a tie; a
    // propagation WIN is equally acceptable (it never under-ranks the chain
    // it owns) — the gate is tie-or-better.
    let cell = Cell { hop: 1, distractors: 4, k: 8 };
    let acc = run_cell(&cell, &seeds, &cfg_base);
    eprintln!(
        "control h=1 d=4 k=8: single {0:.3}/{1:.3} bfs {2:.3}/{3:.3} mass {4:.3}/{5:.3}",
        acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]
    );
    assert!(
        acc[4] >= acc[2] - 0.05,
        "1-hop control: propagation chain recall {0:.3} fell >5pp below BFS {1:.3}",
        acc[4], acc[2]
    );
    assert!(
        acc[5] >= acc[3] - 0.05,
        "1-hop control: propagation tail recall {0:.3} fell >5pp below BFS {1:.3}",
        acc[5], acc[3]
    );

    // Multi-hop with NO distractors: BFS's best case — nothing under-ranks.
    // Propagation must still match it (weights see a clean chain).
    let cell = Cell { hop: 3, distractors: 0, k: 8 };
    let acc = run_cell(&cell, &seeds, &cfg_base);
    eprintln!(
        "control h=3 d=0 k=8: single {0:.3}/{1:.3} bfs {2:.3}/{3:.3} mass {4:.3}/{5:.3}",
        acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]
    );
    assert!(
        acc[4] >= acc[2] - 0.05,
        "no-distractor control: propagation chain recall {0:.3} < BFS {1:.3} (weights must not hurt a clean chain)",
        acc[4], acc[2]
    );
    assert!(
        acc[5] >= acc[3] - 0.05,
        "no-distractor control: propagation tail recall {0:.3} < BFS {1:.3}",
        acc[5], acc[3]
    );
}

/// Mean-blend degeneracy measured (not assumed): under calibrated
/// distractors, the literal edge_avg loses Mass's discrimination — its
/// tail recall must sit ≤ Mass's (documents WHY Mass is the primary arm).
#[test]
fn g1_mean_blend_degeneracy_measured() {
    let rows = run_sweep();
    let mut mass_tail = 0.0f64;
    let mut mean_tail = 0.0f64;
    let mut count = 0usize;
    for (cell, acc) in &rows {
        if cell.hop >= 2 {
            mass_tail += acc[5];
            mean_tail += acc[7];
            count += 1;
        }
    }
    mass_tail /= count as f64;
    mean_tail /= count as f64;
    eprintln!("mean-blend degeneracy h≥2: Mass tail {mass_tail:.3} vs Mean tail {mean_tail:.3}");
    assert!(
        mass_tail >= mean_tail,
        "expected Mass ≥ Mean on tail recall; got Mass {mass_tail:.3} < Mean {mean_tail:.3}"
    );
}
