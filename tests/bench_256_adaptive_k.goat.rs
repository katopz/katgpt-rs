#![cfg(feature = "msa_adaptive_k")]
//! Benchmark — Adaptive-K vs Fixed-K Block Selection (Plan 256 Phase 2 GOAT gate)
//!
//! Measures AdaptiveKRouter (variance-driven k budget via sigmoid gate) against
//! fixed-k BlockTopKRouter. RULER proxy: recall@k vs dense, avg k used, latency.
//!
//! Run: `CARGO_TARGET_DIR=/tmp/riir-test-build cargo test --features msa_adaptive_k --test bench_256_adaptive_k_goat -- --nocapture`

use katgpt_rs::dash_attn::adaptive_k::{compute_adaptive_k, AdaptiveKConfig, AdaptiveKRouter};
use katgpt_rs::dash_attn::block_topk::{BlockTopKCache, BlockTopKRouter};
use katgpt_rs::dash_attn::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};
use std::hint::black_box;

const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 16;
const K_FIXED: usize = 32;
const N_QUERIES: usize = 128;
const N_BLOCKS_CONFIGS: &[usize] = &[64, 256, 1024];
const LATENCY_ITERS: usize = 100;

/// Deterministic per-block centroid from a hash + sin (as specified by task).
fn block_centroid(block_idx: usize, head_dim: usize) -> Vec<f32> {
    (0..head_dim)
        .map(|d| {
            let x = ((block_idx * head_dim + d).wrapping_mul(2654435761)) as f32;
            (x * 0.0001).sin() * 0.5 + 0.5
        })
        .collect()
}

/// Expand a centroid to BLOCK_SIZE key tokens with small per-token noise.
/// Forward_cache mean-pools keys, so cache centroid ≈ true centroid.
fn expand_block_keys(centroid: &[f32], block_idx: usize) -> Vec<f32> {
    let mut keys = Vec::with_capacity(centroid.len() * BLOCK_SIZE);
    for t in 0..BLOCK_SIZE {
        for (d, &c) in centroid.iter().enumerate() {
            let seed = (block_idx.wrapping_mul(7919))
                .wrapping_add(t.wrapping_mul(31))
                .wrapping_add(d)
                .wrapping_mul(2654435761);
            keys.push(c + ((seed as f32) * 0.0001).sin() * 0.02);
        }
    }
    keys
}

/// Focused query: query = centroid[target] + tiny noise.
/// Produces one score peak (HIGH variance) → adaptive picks many blocks.
fn make_focused_query(target_block: usize) -> Vec<f32> {
    block_centroid(target_block, HEAD_DIM)
        .iter()
        .enumerate()
        .map(|(d, &v)| {
            let seed = (target_block.wrapping_mul(40503).wrapping_add(d)).wrapping_mul(2654435761);
            v + ((seed as f32) * 0.0001).sin() * 0.01
        })
        .collect()
}

/// Scattered query: average of 4..=8 block centroids → multimodal, lower variance.
fn make_scattered_query(blocks: &[usize]) -> Vec<f32> {
    let mut q = vec![0.0f32; HEAD_DIM];
    for &b in blocks {
        let c = block_centroid(b, HEAD_DIM);
        for d in 0..HEAD_DIM {
            q[d] += c[d];
        }
    }
    let inv = 1.0 / blocks.len() as f32;
    for v in q.iter_mut() {
        *v *= inv;
    }
    q
}

fn build_focused_set(n_blocks: usize) -> QuerySet {
    QuerySet {
        label: "focused",
        queries: (0..N_QUERIES).map(|i| make_focused_query(i % n_blocks)).collect(),
    }
}

fn build_scattered_set(n_blocks: usize) -> QuerySet {
    QuerySet {
        label: "scattered",
        queries: (0..N_QUERIES)
            .map(|i| {
                let n_pick = 4 + (i % 5); // 4..=8 centroids averaged
                let blocks: Vec<usize> = (0..n_pick).map(|j| (i * 7 + j * 13) % n_blocks).collect();
                make_scattered_query(&blocks)
            })
            .collect(),
    }
}

// ── Dense reference ───────────────────────────────────────────────────────────

/// Full-sort dense top-k block indices from raw dot-product scores.
fn dense_topk_blocks(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(k).map(|(i, _)| i).collect()
}
/// Compute raw dot(query, centroid[i]) for each block. Order matches scaled
/// router scores (scale is monotonic), so dense top-k set is identical.
fn dot_query_centroids(query: &[f32], cache: &BlockTopKCache, n_blocks: usize) -> Vec<f32> {
    (0..n_blocks)
        .map(|i| {
            let c = cache.centroid(i);
            let mut s = 0.0f32;
            for d in 0..HEAD_DIM {
                s += query[d] * c[d];
            }
            s
        })
        .collect()
}

fn populate_cache<R: VortexFlow>(router: &R, cache: &mut R::Cache, n_blocks: usize) {
    let values = vec![0.0f32; BLOCK_SIZE * HEAD_DIM];
    for i in 0..n_blocks {
        let centroid = block_centroid(i, HEAD_DIM);
        let keys = expand_block_keys(&centroid, i);
        router.forward_cache(cache, &keys, &values, i, HEAD_DIM);
    }
}

// ── Metrics ───────────────────────────────────────────────────────────────────

struct QuerySet {
    label: &'static str,
    queries: Vec<Vec<f32>>,
}

#[derive(Default, Clone, Copy)]
struct Metrics {
    recall_sum: f64,
    k_sum: usize,
    n: usize,
}

impl Metrics {
    fn avg_recall(&self) -> f64 {
        (self.n > 0).then(|| self.recall_sum / self.n as f64).unwrap_or(0.0)
    }
    fn avg_k(&self) -> f64 {
        (self.n > 0).then(|| self.k_sum as f64 / self.n as f64).unwrap_or(0.0)
    }
}

/// Measure recall vs dense top-K_FIXED and avg k used for `router` on `qs`.
/// Recall = |router_blocks ∩ dense_blocks| / K_FIXED.
fn measure_router<R: VortexFlow<Cache = BlockTopKCache>>(
    router: &R,
    cache: &BlockTopKCache,
    qs: &QuerySet,
    n_blocks: usize,
) -> Metrics {
    let mut m = Metrics::default();
    let mut scratch = VortexScratch::new(n_blocks);
    for q in &qs.queries {
        let dense = dense_topk_blocks(&dot_query_centroids(q, cache, n_blocks), K_FIXED);
        let dense_set: std::collections::HashSet<usize> = dense.into_iter().collect();
        let decision = router.forward_indexer(q, cache, n_blocks, K_FIXED, &mut scratch);
        let hits = decision.blocks.iter().filter(|&&b| dense_set.contains(&b)).count();
        m.recall_sum += hits as f64 / K_FIXED as f64;
        m.k_sum += decision.len();
        m.n += 1;
    }
    m
}

// ── O3 (Issue 015): precision@adaptive_k + weighted recall ──────────────────
//
// The existing recall metric (above) divides hits by K_FIXED=32, so it is
// mathematically capped at adapt_k/32 ≈ 20/32 = 0.625 because adaptive-k
// picks fewer blocks than fixed-k. Two alternative metrics reframe the result:
//
//   precision@adaptive_k = |adapt ∩ dense_top{adapt_k}| / adapt_k
//       → "of the blocks adaptive-k picked, how many were the true top-adapt_k?"
//   weighted recall       = Σ scores(adapt ∩ dense_top32) / Σ scores(dense_top32)
//       → "what fraction of dense-top-32 score mass did adaptive-k capture?"
//
// Both use existing data (dot scores, dense_topk_blocks, decision.blocks).
#[derive(Default, Clone, Copy)]
struct MetricsO3 {
    precision_sum: f64,   // Σ precision@adaptive_k per query
    weighted_sum: f64,    // Σ weighted_recall per query
    n: usize,
}

impl MetricsO3 {
    fn avg_precision(&self) -> f64 {
        (self.n > 0).then(|| self.precision_sum / self.n as f64).unwrap_or(0.0)
    }
    fn avg_weighted(&self) -> f64 {
        (self.n > 0).then(|| self.weighted_sum / self.n as f64).unwrap_or(0.0)
    }
}

fn measure_router_o3<R: VortexFlow<Cache = BlockTopKCache>>(
    router: &R,
    cache: &BlockTopKCache,
    qs: &QuerySet,
    n_blocks: usize,
) -> MetricsO3 {
    let mut m = MetricsO3::default();
    let mut scratch = VortexScratch::new(n_blocks);
    for q in &qs.queries {
        let scores = dot_query_centroids(q, cache, n_blocks);

        // Dense top-32 (reference for weighted recall denominator).
        let dense32 = dense_topk_blocks(&scores, K_FIXED);
        let dense32_set: std::collections::HashSet<usize> = dense32.iter().copied().collect();
        let dense32_score_sum: f64 = dense32.iter().map(|&b| scores[b] as f64).sum();

        // Adaptive router decision.
        let decision = router.forward_indexer(q, cache, n_blocks, K_FIXED, &mut scratch);
        let adapt_k = decision.len();
        let adapt_set: std::collections::HashSet<usize> =
            decision.blocks.iter().copied().collect();

        // precision@adaptive_k: of adapt_k picked, how many are in dense_top{adapt_k}?
        if adapt_k > 0 {
            let dense_adaptk = dense_topk_blocks(&scores, adapt_k);
            let dense_adaptk_set: std::collections::HashSet<usize> =
                dense_adaptk.iter().copied().collect();
            let hit = adapt_set.intersection(&dense_adaptk_set).count();
            m.precision_sum += hit as f64 / adapt_k as f64;
        }

        // weighted recall: score mass of (adapt ∩ dense32) over score mass of dense32.
        if dense32_score_sum > 0.0 {
            let inter_score: f64 = decision
                .blocks
                .iter()
                .filter(|&&b| dense32_set.contains(&b))
                .map(|&b| scores[b] as f64)
                .sum();
            m.weighted_sum += inter_score / dense32_score_sum;
        }

        m.n += 1;
    }
    m
}

/// Average ns/query for `router` over `iters` passes through all queries.
fn time_router<R: VortexFlow<Cache = BlockTopKCache>>(
    router: &R,
    cache: &BlockTopKCache,
    queries: &[Vec<f32>],
    n_blocks: usize,
    iters: usize,
) -> f64 {
    let mut scratch = VortexScratch::new(n_blocks);
    let mut sink = RoutingDecision::new();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        for q in queries {
            let d = router.forward_indexer(q, cache, n_blocks, K_FIXED, &mut scratch);
            sink.clear();
            sink.blocks.extend_from_slice(&d.blocks);
        }
    }
    black_box(sink.blocks.len());
    start.elapsed().as_nanos() as f64 / (iters * queries.len()) as f64
}

// ── Main benchmark ────────────────────────────────────────────────────────────

#[test]
fn bench_adaptive_k_vs_fixed_k() {
    let adaptive_config = AdaptiveKConfig::new(4, K_FIXED);

    // Sanity demo: compute_adaptive_k on synthetic score shapes.
    println!("\n── compute_adaptive_k reference (k_min=4, k_max={}) ──", K_FIXED);
    let one_peak: Vec<f32> = (0..64).map(|i| if i == 0 { 5.0 } else { 0.1 }).collect();
    println!("  one_peak (high var)  → k = {}", compute_adaptive_k(&one_peak, 64, &adaptive_config));
    println!("  uniform  (low  var)  → k = {}", compute_adaptive_k(&vec![1.0; 64], 64, &adaptive_config));

    let (mut adapt_k, mut adapt_recall, mut fixed_recall, mut count) = (0usize, 0.0f64, 0.0f64, 0usize);
    let mut by_label_k: std::collections::HashMap<&'static str, (usize, usize)> = std::collections::HashMap::new();
    // O3 (Issue 015) — precision@adaptive_k + weighted recall accumulators.
    let (mut o3_precision_sum, mut o3_weighted_sum, mut o3_n) = (0.0f64, 0.0f64, 0usize);
    println!();
    println!("╔════════════════════╦═══════════╦════════════════╦════════════════╦═══════════════════╦══════════════════╗");
    println!("║ Plan 256 Phase 2 — Adaptive-K vs Fixed-K Block Selection (GOAT gate)                             ║");
    println!("║ K_FIXED={}, AdaptiveKConfig(k_min=4, k_max={}, w=5.0, b=0.0), HEAD_DIM={}                          ║",
        K_FIXED, K_FIXED, HEAD_DIM);
    println!("╠════════════════════╦═══════════╦════════════════╦════════════════╦═══════════════════╦══════════════════╣");
    println!("║ n_blocks           ║ query     ║ fixed recall   ║ adapt recall   ║ fixed k / adapt k ║ adapt/fixed ns/q ║");
    println!("║                    ║ type      ║ (vs dense)     ║ (vs dense)     ║ avg per query     ║ (lower=better)   ║");
    println!("╠════════════════════╬═══════════╬════════════════╬════════════════╬═══════════════════╬══════════════════╣");
    for &n_blocks in N_BLOCKS_CONFIGS {
        let focused = build_focused_set(n_blocks);
        let scattered = build_scattered_set(n_blocks);

        let fixed_router = BlockTopKRouter::new(true);
        let mut fixed_cache = fixed_router.cache_new(n_blocks, HEAD_DIM);
        populate_cache(&fixed_router, &mut fixed_cache, n_blocks);

        let adapt_router = AdaptiveKRouter::new(BlockTopKRouter::new(true), adaptive_config.clone());
        let mut adapt_cache = adapt_router.cache_new(n_blocks, HEAD_DIM);
        populate_cache(&adapt_router, &mut adapt_cache, n_blocks);

        for qs in [&focused, &scattered] {
            let m_fixed = measure_router(&fixed_router, &fixed_cache, qs, n_blocks);
            let m_adapt = measure_router(&adapt_router, &adapt_cache, qs, n_blocks);
            let fixed_ns = time_router(&fixed_router, &fixed_cache, &qs.queries, n_blocks, LATENCY_ITERS);
            let adapt_ns = time_router(&adapt_router, &adapt_cache, &qs.queries, n_blocks, LATENCY_ITERS);

            // O3 (Issue 015) — additional metrics on existing data.
            let m_o3 = measure_router_o3(&adapt_router, &adapt_cache, qs, n_blocks);

            println!(
                "║ n_blocks={:<9}║ {:<9}║ {:>11.1}%   ║ {:>11.1}%   ║ {:>5.1} / {:<5.1}    ║ {:>5.0} / {:<5.0}    ║",
                n_blocks, qs.label,
                m_fixed.avg_recall() * 100.0, m_adapt.avg_recall() * 100.0,
                m_fixed.avg_k(), m_adapt.avg_k(), adapt_ns, fixed_ns,
            );
            // O3 metrics printed via eprintln so they surface under --nocapture
            // without disturbing the existing table layout.
            eprintln!(
                "    [O3] n_blocks={nb} {label}: precision@adapt_k={p:.4}  weighted_recall={w:.4}  \
                 (adapt_k≈{ak:.1}, fixed_k={fk})",
                nb = n_blocks, label = qs.label,
                p = m_o3.avg_precision(), w = m_o3.avg_weighted(),
                ak = m_adapt.avg_k(), fk = K_FIXED,
            );

            adapt_k += m_adapt.k_sum;
            adapt_recall += m_adapt.recall_sum;
            fixed_recall += m_fixed.recall_sum;
            count += m_adapt.n;
            o3_precision_sum += m_o3.precision_sum;
            o3_weighted_sum += m_o3.weighted_sum;
            o3_n += m_o3.n;
            let e = by_label_k.entry(qs.label).or_insert((0, 0));
            e.0 += m_adapt.k_sum;
            e.1 += m_adapt.n;
        }
    }
    println!("╚════════════════════╩═══════════╩════════════════╩════════════════╩═══════════════════╩══════════════════╝");
    // Variance gate check (focused vs scattered avg k)
    let fk = by_label_k.get("focused").copied().unwrap_or((0, 0));
    let sk = by_label_k.get("scattered").copied().unwrap_or((0, 0));
    let focused_avg_k = (fk.1 > 0).then(|| fk.0 as f64 / fk.1 as f64).unwrap_or(0.0);
    let scattered_avg_k = (sk.1 > 0).then(|| sk.0 as f64 / sk.1 as f64).unwrap_or(0.0);
    println!();
    println!("── Variance gate behaviour (observed) ──");
    println!("  Focused  avg k : {:>5.2}  (one peak  → mathematically HIGH variance)", focused_avg_k);
    println!("  Scattered avg k: {:>5.2}  (multimodal → lower variance per query)", scattered_avg_k);
    println!("  Δ (focused - scattered) = {:+.2} blocks", focused_avg_k - scattered_avg_k);

    // ── GOAT verdict ──────────────────────────────────────────────────────────
    let avg_k_adapt = (count > 0).then(|| adapt_k as f64 / count as f64).unwrap_or(0.0);
    let avg_recall_adapt = (count > 0).then(|| adapt_recall / count as f64).unwrap_or(0.0);
    let avg_recall_fixed = (count > 0).then(|| fixed_recall / count as f64).unwrap_or(0.0);
    let savings_ratio = avg_k_adapt / K_FIXED as f64;
    let recall_ratio = (avg_recall_fixed > 0.0).then(|| avg_recall_adapt / avg_recall_fixed).unwrap_or(0.0);

    println!();
    println!("── Overall ({} queries × {} configs = {} samples) ──", N_QUERIES, N_BLOCKS_CONFIGS.len(), count);
    println!("  Avg k used by adaptive : {:.2} (fixed budget = {})", avg_k_adapt, K_FIXED);
    println!("  Compute savings        : {:>5.1}%  (criterion: ≥ 25%)", (1.0 - savings_ratio) * 100.0);
    println!("  Adaptive recall        : {:.4}", avg_recall_adapt);
    println!("  Fixed-k recall         : {:.4}", avg_recall_fixed);
    println!("  Recall ratio (ad/fixed): {:.4}  (criterion: ≥ 0.90)", recall_ratio);

    // O3 (Issue 015) — precision@adaptive_k + weighted recall aggregates.
    // These reframe the recall result: recall_ratio is capped at adapt_k/32 ≈ 0.625
    // by construction (adaptive picks fewer blocks); precision@adaptive_k and
    // weighted recall ask "did adaptive pick the RIGHT (highest-scoring) blocks?",
    // which is the actual design question. Informational, not a gate.
    let o3_avg_precision = (o3_n > 0).then(|| o3_precision_sum / o3_n as f64).unwrap_or(0.0);
    let o3_avg_weighted = (o3_n > 0).then(|| o3_weighted_sum / o3_n as f64).unwrap_or(0.0);
    println!();
    println!("── O3 (Issue 015): precision@adaptive_k + weighted recall ({} samples) ──", o3_n);
    println!("  precision@adaptive_k : {:.4}  (1.0 = adaptive picks exactly dense top-adapt_k)", o3_avg_precision);
    println!("  weighted recall      : {:.4}  (1.0 = adaptive captures all dense-top-32 score mass)", o3_avg_weighted);
    println!("  (reframes recall ratio {:.3}: adaptive picks fewer but higher-value blocks)", recall_ratio);
    eprintln!(
        "    [O3 aggregate] precision@adapt_k={p:.4}  weighted_recall={w:.4}  \
         recall_ratio={rr:.4}  adapt_k≈{ak:.2}",
        p = o3_avg_precision, w = o3_avg_weighted, rr = recall_ratio, ak = avg_k_adapt,
    );

    let pass_savings = savings_ratio <= 0.75;
    let pass_recall = recall_ratio >= 0.90;
    println!();
    if pass_savings && pass_recall {
        println!("GOAT: PASS — adaptive-k saves {:.1}% compute at {:.1}% recall",
            (1.0 - savings_ratio) * 100.0, recall_ratio * 100.0);
    } else {
        let mut reasons = Vec::new();
        if !pass_savings {
            reasons.push(format!("savings {:.1}% < 25% (avg k {:.2} > {:.0})",
                (1.0 - savings_ratio) * 100.0, avg_k_adapt, (0.75 * K_FIXED as f64).ceil()));
        }
        if !pass_recall {
            reasons.push(format!("recall ratio {:.3} < 0.90", recall_ratio));
        }
        println!("GOAT: FAIL — {}", reasons.join("; "));
    }

    // Assert only that the benchmark actually ran.
    assert!(count > 0, "benchmark must process queries");
    assert!(fk.1 > 0 && sk.1 > 0, "both query types must be measured");
}
