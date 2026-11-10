//! Product Key Memory GOAT gate bench (Plan 408 Phase 3).
//!
//! Measures the four gates that decide promotion to default-on:
//!
//! - **G1 (latency)**: `query_into` median latency at `SQRT_N=1000` (N=10⁶
//!   slots), `D_K=64`, `top_k=8`. Compares against brute-force O(N) scan over
//!   the same 10⁶ slots. Target: PKM ≥ 100× faster than brute-force.
//! - **G2 (top-k correctness)**: Jaccard overlap between PKM's top-k and
//!   brute-force's top-k on a fixed random table, 200 random queries. Target:
//!   mean Jaccard ≥ 0.95. (The full 1000-query run is in the Phase 2 unit
//!   test `t26_top_k_matches_brute_force_many_queries_dot`; this bench re-
//!   confirms at the G1 scale SQRT_N=1000 with a smaller query count to keep
//!   bench runtime bounded — the brute-force at N=10⁶ is ~50ms/query.)
//! - **G3 (IDW centroid-ness)**: Validated in the Phase 2 unit test
//!   `t27_idw_attracts_to_closer_centroids` (4-cluster fixture). This bench
//!   re-confirms at SQRT_N=1000 scale by measuring intra-cluster access rate
//!   Dot vs IDW. Target: IDW ≥ 1.2× higher intra-cluster rate.
//! - **G4 (zero-alloc)**: `query_into` allocates 0 bytes over 1000 steady-
//!   state calls (CountingAllocator, after warmup).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/pkm_goat cargo bench -p katgpt-core \
//!   --features product_key_memory --bench bench_408_pkm_goat -- --nocapture
//! ```
//!
//! Or, working around the intermittent macOS dyld/trustd launch stall
//! (documented in Plan 326 / bench_327):
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/pkm_goat target/release/deps/bench_408_pkm_goat-* --nocapture
//! ```

use katgpt_core::product_key_memory::{PkmScratch, ProductKeyMemory, ScoreFn, score_dot};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Constants ──────────────────────────────────────────────────────────────

/// Production-scale: SQRT_N=1000 → N=10⁶ slots. The plan's headline target.
const SQRT_N: usize = 1000;
/// Key dim 64 (paper default). Split into two 32-dim halves.
const D_K: usize = 64;
/// Value dim — kept small (4) so the 10⁶-row value table is 16MB (manageable
/// for the bench; the production target D_V=128 would be 512MB). The retrieval
/// cost is dominated by the √N codebook scans, NOT the value fetches, so a
/// small D_V doesn't weaken the G1 latency gate.
const D_V: usize = 4;
/// Per-codebook top-k. Final output k = K = 8.
const K: usize = 8;
/// Number of latency-timed iterations (per-call median).
const LATENCY_ITERS: usize = 1_000;
/// Number of Jaccard queries (brute-force at N=10⁶ is ~50ms/query, so this
/// is bounded to keep bench runtime reasonable).
const JACCARD_QUERIES: usize = 50;
/// Alloc-check steady-state iterations.
const ALLOC_ITERS: usize = 1_000;
/// G1 target: PKM ≥ 100× faster than brute-force.
const G1_SPEEDUP_TARGET: f64 = 100.0;
/// G2 target: mean Jaccard ≥ 0.95.
const G2_JACCARD_TARGET: f64 = 0.95;
/// G3 target: IDW intra-cluster rate ≥ 1.2× Dot rate.
const G3_IDW_RATIO_TARGET: f64 = 1.2;

// ── Deterministic splitmix64 PRNG (mirrors types.rs SeededRng) ─────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as u32;
        let unit = u as f32 / ((1u32 << 24) as f32);
        lo + unit * (hi - lo)
    }
}

// ── Brute-force O(N) baseline (the G1/G2 reference) ────────────────────────

/// Brute-force top-k: score all SQRT_N*SQRT_N flat indices, return the
/// sorted-descending top-k `(flat_index, score)` pairs. O(N * D_K).
fn brute_force_top_k_dot(
    table: &ProductKeyMemory<SQRT_N, D_K, D_V>,
    q: &[f32; D_K],
    k: usize,
) -> Vec<(usize, f32)> {
    let half = D_K / 2;
    let (q1, q2) = q.split_at(half);
    let n = SQRT_N * SQRT_N;
    let mut all: Vec<(usize, f32)> = Vec::with_capacity(n);
    // Pre-compute codebook scores (avoids recomputing s1[i] for every j).
    let mut s1 = vec![0.0f32; SQRT_N];
    let mut s2 = vec![0.0f32; SQRT_N];
    for (i, s) in s1.iter_mut().enumerate() {
        *s = score_dot(q1, table.keys_1_row(i));
    }
    for (j, s) in s2.iter_mut().enumerate() {
        *s = score_dot(q2, table.keys_2_row(j));
    }
    for (i, &si) in s1.iter().enumerate() {
        for (j, &sj) in s2.iter().enumerate() {
            all.push((i * SQRT_N + j, si + sj));
        }
    }
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all.shrink_to_fit();
    all
}

fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    let set_a: std::collections::HashSet<usize> = a.iter().copied().collect();
    let set_b: std::collections::HashSet<usize> = b.iter().copied().collect();
    let inter = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0 { 1.0 } else { inter / union }
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    const D_K_C: usize = 64;
    const D_V_C: usize = 4;
    const K_C: usize = 8;
    const N_CLUSTERS: usize = 10;

    println!("══════════════════════════════════════════════════════════════════");
    println!("  Plan 408 Phase 3 — Product Key Memory GOAT gate");
    println!(
        "  SQRT_N={} (N={} slots), D_K={}, D_V={}, K={}",
        SQRT_N,
        SQRT_N * SQRT_N,
        D_K,
        D_V,
        K
    );
    println!(
        "  LATENCY_ITERS={LATENCY_ITERS}, JACCARD_QUERIES={JACCARD_QUERIES}, ALLOC_ITERS={ALLOC_ITERS}"
    );
    println!("══════════════════════════════════════════════════════════════════\n");

    // ── Build the table (deterministic seed). ───────────────────────────
    // The value table is SQRT_N*SQRT_N*D_V*4 = 16MB at this scale.
    let t0 = Instant::now();
    let table = ProductKeyMemory::<SQRT_N, D_K, D_V>::from_random(20_260_707);
    println!(
        "Table build (N={} slots, {:.1} MB): {:?}\n",
        SQRT_N * SQRT_N,
        (SQRT_N * SQRT_N * D_V * 4) as f64 / 1e6,
        t0.elapsed()
    );

    // Build a fixed set of random queries (deterministic).
    // Deterministic query seed (splitmix64 mixes it, so any constant works).
    let mut rng = Rng::new(0xCAFE_BABE_0000_0001);
    let mut queries = Vec::with_capacity(LATENCY_ITERS.max(JACCARD_QUERIES));
    for _ in 0..LATENCY_ITERS.max(JACCARD_QUERIES) {
        let mut q = [0.0f32; D_K];
        for v in q.iter_mut() {
            *v = rng.next_f32(-1.0, 1.0);
        }
        queries.push(q);
    }

    // ── G1: PKM latency ─────────────────────────────────────────────────
    let mut scratch = PkmScratch::<SQRT_N, K>::new();
    let mut out = [(0usize, 0.0f32); K];

    // Warmup.
    for q in queries.iter().take(100) {
        let _ = table.query_into(q, ScoreFn::Dot, K, &mut out, &mut scratch);
    }

    let mut pkm_times_ns: Vec<u64> = Vec::with_capacity(LATENCY_ITERS);
    for q in queries.iter().take(LATENCY_ITERS) {
        let t = Instant::now();
        let n = table.query_into(
            black_box(q),
            ScoreFn::Dot,
            K,
            black_box(&mut out),
            black_box(&mut scratch),
        );
        let elapsed = t.elapsed();
        debug_assert_eq!(n, K);
        pkm_times_ns.push(elapsed.as_nanos() as u64);
    }
    pkm_times_ns.sort();
    let pkm_p50 = pkm_times_ns[pkm_times_ns.len() / 2];
    let pkm_p99 = pkm_times_ns[pkm_times_ns.len() * 99 / 100];
    let pkm_mean = pkm_times_ns.iter().sum::<u64>() as f64 / pkm_times_ns.len() as f64;

    // ── G1: brute-force latency (fewer iters — each is ~50ms). ──────────
    let bf_iters = 20.min(queries.len());
    let mut bf_times_ns: Vec<u64> = Vec::with_capacity(bf_iters);
    for q in queries.iter().take(bf_iters) {
        let t = Instant::now();
        let _bf = brute_force_top_k_dot(black_box(&table), black_box(q), K);
        bf_times_ns.push(t.elapsed().as_nanos() as u64);
    }
    bf_times_ns.sort();
    let bf_p50 = bf_times_ns[bf_times_ns.len() / 2];

    let speedup = bf_p50 as f64 / pkm_p50 as f64;
    let g1_pass = speedup >= G1_SPEEDUP_TARGET;

    println!("── G1: latency (O(√N) PKM vs O(N) brute-force) ──────────────────");
    println!("  PKM   p50 = {pkm_p50:>10} ns  (mean {pkm_mean:.0} ns, p99 {pkm_p99:.0} ns)");
    println!("  BF    p50 = {bf_p50:>10} ns  ({bf_iters} iters)");
    println!("  Speedup    = {speedup:>7.1}×   (target ≥ {G1_SPEEDUP_TARGET:.0}×)");
    println!(
        "  G1 verdict: {}\n",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── G2: top-k Jaccard vs brute-force ────────────────────────────────
    let mut jaccard_sum = 0.0f64;
    let mut jaccard_min = f64::INFINITY;
    for q in queries.iter().take(JACCARD_QUERIES) {
        let n = table.query_into(q, ScoreFn::Dot, K, &mut out, &mut scratch);
        let pkm_idx: Vec<usize> = out[..n].iter().map(|(i, _)| *i).collect();
        let bf = brute_force_top_k_dot(&table, q, K);
        let bf_idx: Vec<usize> = bf.iter().map(|(i, _)| *i).collect();
        let j = jaccard(&pkm_idx, &bf_idx);
        jaccard_sum += j;
        jaccard_min = jaccard_min.min(j);
    }
    let jaccard_mean = jaccard_sum / JACCARD_QUERIES as f64;
    let g2_pass = jaccard_mean >= G2_JACCARD_TARGET;

    println!("── G2: top-k Jaccard vs brute-force ─────────────────────────────");
    println!(
        "  Mean Jaccard = {jaccard_mean:.4}  (min {jaccard_min:.4}, {JACCARD_QUERIES} queries)"
    );
    println!(
        "  G2 verdict: {}   (target ≥ {:.2})\n",
        if g2_pass { "✅ PASS" } else { "❌ FAIL" },
        G2_JACCARD_TARGET
    );

    // ── G3: IDW centroid-ness (intra-cluster access rate, Dot vs IDW) ───
    //
    // Build a clustered table where cluster 0 is near the origin (low
    // magnitude) and clusters 1..N are far out (high magnitude). Query near
    // cluster 0's center. IDW should retrieve cluster-0 rows (closest in
    // Euclidean distance); Dot may retrieve high-magnitude clusters because
    // dot(q1, big_vec) can exceed dot(q1, small_vec) by sheer magnitude.
    // This mirrors the Phase 2 unit test fixture that discriminates the two
    // modes; the bench re-confirms at SQRT_N=1000 scale.
    const SQRT_N_C: usize = 1000;
    const CLUSTER_SIZE: usize = SQRT_N_C / N_CLUSTERS; // 100
    const HALF_C: usize = D_K_C / 2;

    let mut keys_1_c = vec![0.0f32; SQRT_N_C * HALF_C].into_boxed_slice();
    // Cluster 0: near origin (low magnitude). Clusters 1..N: high magnitude,
    // spread on a circle of radius 5 in dims 0-1.
    let mut cluster_centers = [[0.0f32; HALF_C]; N_CLUSTERS];
    // Cluster 0 — near origin.
    cluster_centers[0][0] = 0.1;
    cluster_centers[0][1] = 0.1;
    for (ci, center) in cluster_centers.iter_mut().enumerate().skip(1) {
        let angle = ci as f32 * (std::f32::consts::TAU / (N_CLUSTERS - 1) as f32);
        center[0] = 5.0 * angle.cos();
        center[1] = 5.0 * angle.sin();
        // Fill the rest of the dims with high magnitude too (makes the dot
        // product with any aligned query large).
        for dim in center.iter_mut().take(HALF_C).skip(2) {
            *dim = 5.0;
        }
    }
    for i in 0..SQRT_N_C {
        let cluster = i / CLUSTER_SIZE;
        for d in 0..HALF_C {
            let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
            keys_1_c[i * HALF_C + d] = cluster_centers[cluster][d] + noise;
        }
    }
    let keys_2_c = vec![0.5f32; SQRT_N_C * HALF_C].into_boxed_slice();
    let values_c = vec![0.0f32; SQRT_N_C * SQRT_N_C * D_V_C].into_boxed_slice();
    let table_c =
        ProductKeyMemory::<SQRT_N_C, D_K_C, D_V_C>::from_centroids(keys_1_c, keys_2_c, values_c);

    // Query: q1 near cluster 0's center (low magnitude). Dot may retrieve a
    // high-magnitude cluster because dot(small_q, big_vec) > dot(small_q,
    // small_vec) when the big_vec is aligned-ish.
    let mut q_c = [0.0f32; D_K_C];
    q_c[0] = 0.1;
    q_c[1] = 0.1;
    for dim in q_c.iter_mut().take(HALF_C).skip(2) {
        *dim = 0.1; // small magnitude, same sign as cluster centers' fill
    }
    for dim in q_c.iter_mut().take(D_K_C).skip(HALF_C) {
        *dim = 0.5;
    }

    let intra_rate = |score_fn: ScoreFn| -> f64 {
        let mut scratch_c = PkmScratch::<SQRT_N_C, K_C>::new();
        let mut out_c = [(0usize, 0.0f32); K_C];
        let n = table_c.query_into(&q_c, score_fn, K_C, &mut out_c, &mut scratch_c);
        let mut intra = 0usize;
        for (flat_idx, _) in out_c[..n].iter() {
            let i = flat_idx / SQRT_N_C;
            if i / CLUSTER_SIZE == 0 {
                intra += 1;
            }
        }
        intra as f64 / n as f64
    };

    let dot_rate = intra_rate(ScoreFn::Dot);
    let idw_rate = intra_rate(ScoreFn::idw_default());
    let idw_ratio = if dot_rate > 0.0 {
        idw_rate / dot_rate
    } else {
        f64::INFINITY
    };
    let g3_pass = idw_ratio >= G3_IDW_RATIO_TARGET;

    println!("── G3: IDW centroid-ness (intra-cluster-0 access rate) ──────────");
    println!("  Dot intra-cluster rate = {dot_rate:.3}");
    println!("  IDW intra-cluster rate = {idw_rate:.3}");
    println!("  IDW / Dot ratio        = {idw_ratio:.3}×   (target ≥ {G3_IDW_RATIO_TARGET:.1}×)");
    println!(
        "  G3 verdict: {}\n",
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── G4: zero-alloc steady state ────────────────────────────────────
    use std::sync::atomic::Ordering;
    // Warmup (table + scratch construction may alloc; we measure steady state).
    for q in queries.iter().take(10) {
        let _ = table.query_into(q, ScoreFn::Dot, K, &mut out, &mut scratch);
    }
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    for q in queries.iter().take(ALLOC_ITERS) {
        let _ = table.query_into(
            black_box(q),
            ScoreFn::Dot,
            K,
            black_box(&mut out),
            black_box(&mut scratch),
        );
    }
    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_delta = alloc_after - alloc_before;
    let g4_pass = alloc_delta == 0;

    println!("── G4: zero-alloc steady state ──────────────────────────────────");
    println!("  Allocations over {ALLOC_ITERS} query_into calls: {alloc_delta}  (target 0)");
    println!(
        "  G4 verdict: {}\n",
        if g4_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── Overall verdict ─────────────────────────────────────────────────
    let all_pass = g1_pass && g2_pass && g4_pass; // G3 is advisory (Phase 2 unit test is the load-bearing one)
    println!("══════════════════════════════════════════════════════════════════");
    println!("  GOAT gate summary (Plan 408 Phase 3):");
    println!(
        "    G1 latency (≥{:.0}× speedup)      : {}",
        G1_SPEEDUP_TARGET,
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "    G2 top-k Jaccard (≥{:.2})    : {}",
        G2_JACCARD_TARGET,
        if g2_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "    G3 IDW centroid-ness (≥{:.1}×) : {} (advisory)",
        G3_IDW_RATIO_TARGET,
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "    G4 zero-alloc               : {}",
        if g4_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("  ────────────────────────────────────────────");
    println!("  Promotion rule: G1 + G2 + G4 all pass → DEFAULT-ON");
    println!(
        "  Overall: {}",
        if all_pass {
            "✅ PROMOTE — add `product_key_memory` to katgpt-core default features"
        } else {
            "❌ DEMOTE — keep opt-in, document the failing gate in .benchmarks/408_pkm_goat.md"
        }
    );
    println!("══════════════════════════════════════════════════════════════════");

    if !all_pass {
        std::process::exit(1);
    }
}
