#![cfg(feature = "msa_per_group")]
//! Benchmark — Per-GQA-Group vs Shared TopK Selection (Plan 256 Phase 2 GOAT gate)
//!
//! Measures per-group independent top-k against shared top-k baseline.
//! RULER proxy: block-selection diversity + coverage of dense top blocks.
//!
//! Run: `CARGO_TARGET_DIR=/tmp/riir-test-build cargo test --features msa_per_group --test bench_256_per_group_goat -- --nocapture`

use std::collections::HashSet;
use std::time::Instant;

use katgpt_rs::dash_attn::block_topk::{BlockTopKCache, BlockTopKRouter, PerGroupTopKRouter};
use katgpt_rs::dash_attn::vortex_flow::{VortexFlow, VortexScratch};

const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 16;
const N_QUERIES: usize = 128;
const ITERS: usize = 100;
const MAX_K: usize = 32; // largest top_k in the sweep

// ── Deterministic data generation ────────────────────────────────

/// Deterministic pseudo-random vector (index-based seed, full-range sin).
fn make_vec(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i.wrapping_mul(2654435761)).wrapping_add(seed.wrapping_mul(40503))) as f32;
            (x * 0.0001).sin()
        })
        .collect()
}

/// Distinct random centroid per block (deterministic, varying norms).
fn make_centroids(n_blocks: usize) -> Vec<Vec<f32>> {
    (0..n_blocks).map(|i| make_vec(HEAD_DIM, i + 1)).collect()
}

/// Keys for one block: BLOCK_SIZE tokens whose mean ≈ centroid.
fn make_block_keys(centroid: &[f32], block_idx: usize) -> Vec<f32> {
    let mut keys = Vec::with_capacity(BLOCK_SIZE * HEAD_DIM);
    for t in 0..BLOCK_SIZE {
        let noise = make_vec(HEAD_DIM, block_idx.wrapping_mul(131) + t + 7919);
        for d in 0..HEAD_DIM {
            keys.push(centroid[d] + noise[d] * 0.05);
        }
    }
    keys
}

/// Build the shared BlockTopKCache once (identical for both routers — cache is group-independent).
fn build_cache(centroids: &[Vec<f32>], n_blocks: usize) -> BlockTopKCache {
    let router = BlockTopKRouter::new(true);
    let mut cache = BlockTopKCache::new(n_blocks, HEAD_DIM);
    let zero_values = vec![0.0f32; BLOCK_SIZE * HEAD_DIM];
    for i in 0..n_blocks {
        let keys = make_block_keys(&centroids[i], i);
        router.forward_cache(&mut cache, &keys, &zero_values, i, HEAD_DIM);
    }
    cache
}

/// Needle queries: each aligned with a distinct block centroid (clear "best block").
fn make_queries(centroids: &[Vec<f32>], n_blocks: usize) -> Vec<Vec<f32>> {
    (0..N_QUERIES)
        .map(|q| centroids[(q * 7) % n_blocks].clone())
        .collect()
}

/// Ground-truth dense top-k blocks for a query, scored identically to the routers
/// (dot(query, cache_centroid) * 1/sqrt(hd)). Returns indices sorted by score descending.
fn dense_topk_cache(query: &[f32], cache: &BlockTopKCache, n_blocks: usize, k: usize) -> Vec<usize> {
    let hd = query.len();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut scored: Vec<(usize, f32)> = (0..n_blocks)
        .map(|i| {
            let c = cache.centroid(i);
            let dot: f32 = query.iter().zip(c.iter()).map(|(q, cc)| q * cc).sum();
            (i, dot * scale)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

// ── Benchmark ────────────────────────────────────────────────────

#[test]
fn bench_per_group_vs_shared() {
    let n_blocks_set = [64usize, 256, 1024];
    let n_groups_set = [1usize, 2, 4, 8];
    let top_k_set = [8usize, 16, 32];

    println!();
    println!("╔═════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 256 Phase 2 — Per-GQA-Group vs Shared TopK (RULER proxy: diversity + coverage)   ║");
    println!("║  HEAD_DIM=64  BLOCK_SIZE=16  N_QUERIES={N_QUERIES}  ITERS={ITERS}                                 ║");
    println!("╠════════╦══════════╦═══════╦════════════════╦════════════════╦═══════════════════════════╣");
    println!("║ n_blk  ║ n_groups ║ top_k ║  shared ns/q   ║  pergrp ns/q   ║  latency ratio (pg/sh)   ║");
    println!("╠════════╬══════════╬═══════╬════════════════╬════════════════╬═══════════════════════════╣");

    // Collect ratios (n_groups >= 2 only) for the aggregate GOAT verdict.
    let mut cov_ratios: Vec<f64> = Vec::new();
    let mut lat_ratios: Vec<f64> = Vec::new();
    // O1 (Issue 015) — per-call partition spread, informational.
    let mut spread_ratios: Vec<f64> = Vec::new();

    // Deferred diversity table rows.
    let mut div_rows: Vec<String> = Vec::new();

    // Hoist last-measured latencies so we can assert the benchmark actually executed.
    let mut last_shared_ns = 0.0f64;
    let mut last_pg_ns = 0.0f64;

    for &n_blocks in &n_blocks_set {
        let centroids = make_centroids(n_blocks);
        let cache = build_cache(&centroids, n_blocks);
        let queries = make_queries(&centroids, n_blocks);

        // Precompute ground-truth dense top-MAX_K per query (recall reference, sorted desc).
        let dense: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| dense_topk_cache(q, &cache, n_blocks, MAX_K))
            .collect();

        for &n_groups in &n_groups_set {
            let shared = BlockTopKRouter::new(true);
            let pergrp = PerGroupTopKRouter::new(true, n_groups);

            for &top_k in &top_k_set {
                let mut sc_s = VortexScratch::new(n_blocks);
                let mut sc_p = VortexScratch::new(n_blocks);

                // ── Latency: shared ──
                let t = Instant::now();
                for _ in 0..ITERS {
                    for q in 0..N_QUERIES {
                        let _ = shared.forward_indexer(&queries[q], &cache, n_blocks, top_k, &mut sc_s);
                    }
                }
                let shared_ns = t.elapsed().as_nanos() as f64 / (ITERS * N_QUERIES) as f64;

                // ── Latency: per-group ──
                let t = Instant::now();
                for _ in 0..ITERS {
                    for q in 0..N_QUERIES {
                        let _ = pergrp.forward_indexer(&queries[q], &cache, n_blocks, top_k, &mut sc_p);
                    }
                }
                let pg_ns = t.elapsed().as_nanos() as f64 / (ITERS * N_QUERIES) as f64;

                last_shared_ns = shared_ns;
                last_pg_ns = pg_ns;
                let lat_ratio = pg_ns / shared_ns;

                println!(
                    "║ {n_blocks:<6} ║ {n_groups:<8} ║ {top_k:<5} ║ {shared_ns:>12.1}   ║ {pg_ns:>12.1}   ║ {lat_ratio:>19.3}    ║"
                );

                // ── Diversity + recall (single measurement pass, store decisions) ──
                let mut dec_s: Vec<Vec<usize>> = Vec::with_capacity(N_QUERIES);
                let mut dec_p: Vec<Vec<usize>> = Vec::with_capacity(N_QUERIES);
                for q in 0..N_QUERIES {
                    dec_s.push(shared.forward_indexer(&queries[q], &cache, n_blocks, top_k, &mut sc_s).blocks);
                    dec_p.push(pergrp.forward_indexer(&queries[q], &cache, n_blocks, top_k, &mut sc_p).blocks);
                }

                // Coverage = union of distinct blocks selected across all queries.
                let union_s: HashSet<usize> = dec_s.iter().flatten().copied().collect();
                let union_p: HashSet<usize> = dec_p.iter().flatten().copied().collect();
                let cov_ratio = union_p.len() as f64 / union_s.len().max(1) as f64;

                // O1 (Issue 015) — per-call partition spread.
                //
                // The cross-query union cov_ratio saturates near 1.0 because 128
                // queries × 32 top-k touch essentially every reachable block.
                // The per-call metric instead asks, per query: how different is the
                // per-group selection from the shared selection on that same query?
                //
                // Note on "between groups within a single call": the PerGroupTopKRouter
                // partitions blocks by `block_idx % n_groups` (disjoint ownership), so
                // any two distinct groups' selections are trivially disjoint and
                // pairwise-group Jaccard distance is always 1.0 — degenerate and
                // uninformative. The meaningful per-call spread is therefore the
                // Jaccard distance between the per-group router's per-call selection
                // and the shared router's per-call selection: a high value means
                // per-group deliberately picks blocks the shared top-k would NOT have
                // picked on that same call (the design goal of diversification).
                //
                // distance = 1 - |A ∩ B| / |A ∪ B|; averaged over all queries.
                let mut spread_sum = 0.0f64;
                for q in 0..N_QUERIES {
                    let a: HashSet<usize> = dec_p[q].iter().copied().collect();
                    let b: HashSet<usize> = dec_s[q].iter().copied().collect();
                    let inter = a.intersection(&b).count();
                    let uni = a.len() + b.len() - inter;
                    let dist = if uni == 0 { 0.0 } else { 1.0 - inter as f64 / uni as f64 };
                    spread_sum += dist;
                }
                let per_call_spread = spread_sum / N_QUERIES as f64;

                // Recall@top_k vs dense ground truth (dense[q] sorted desc → first top_k are the truth).
                let mut rec_s = 0.0f64;
                let mut rec_p = 0.0f64;
                for q in 0..N_QUERIES {
                    let truth: HashSet<usize> = dense[q][..top_k].iter().copied().collect();
                    let hit_s = dec_s[q].iter().filter(|b| truth.contains(b)).count();
                    let hit_p = dec_p[q].iter().filter(|b| truth.contains(b)).count();
                    rec_s += hit_s as f64 / top_k as f64;
                    rec_p += hit_p as f64 / top_k as f64;
                }
                rec_s /= N_QUERIES as f64;
                rec_p /= N_QUERIES as f64;

                div_rows.push(format!(
                    "║ {n_blocks:<6} ║ {n_groups:<8} ║ {top_k:<5} ║ {us:>6}    ║ {up:>6}    ║ {cov_ratio:>8.3}      ║ {rec_s:>6.3}        ║ {rec_p:>6.3}        ║",
                    us = union_s.len(),
                    up = union_p.len(),
                ));

                // O1 (Issue 015) — per-call partition spread, printed inline for --nocapture.
                eprintln!(
                    "    [O1] n_blocks={n_blocks} n_groups={n_groups} top_k={top_k} \
                     per-call Jaccard spread (pergrp vs shared) = {spread:.4}",
                    spread = per_call_spread,
                );

                // Verdict aggregation excludes n_groups=1 (identical to shared by construction).
                if n_groups >= 2 {
                    cov_ratios.push(cov_ratio);
                    lat_ratios.push(lat_ratio);
                    spread_ratios.push(per_call_spread);
                }
            }
        }
    }

    println!("╚════════╩══════════╩═══════╩════════════════╩════════════════╩═══════════════════════════╝");
    println!();
    println!("╔════════╦══════════╦═══════╦════════════╦════════════╦══════════════════╦═══════════════╦═══════════════╗");
    println!("║ n_blk  ║ n_groups ║ top_k ║ union shrd ║ union pgrp ║ cov ratio (p/sh) ║ recall@k shrd ║ recall@k pgrp ║");
    println!("╠════════╬══════════╬═══════╬════════════╬════════════╬══════════════════╬═══════════════╬═══════════════╣");
    for row in &div_rows {
        println!("{row}");
    }
    println!("╚════════╩══════════╩═══════╩════════════╩════════════╩══════════════════╩═══════════════╩═══════════════╝");
    println!();

    // ── GOAT verdict (computed, not hardcoded) ─────────────────────
    let mean_cov = cov_ratios.iter().sum::<f64>() / cov_ratios.len().max(1) as f64;
    let mean_lat = lat_ratios.iter().sum::<f64>() / lat_ratios.len().max(1) as f64;
    // O1 (Issue 015) — informational mean per-call spread. NOT a gate.
    let mean_spread = spread_ratios.iter().sum::<f64>() / spread_ratios.len().max(1) as f64;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  GOAT Gate — Per-Group vs Shared (aggregated over n_groups≥2)    ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║  mean coverage ratio (per-group / shared): {mean_cov:>20.3}       ║");
    println!("║  mean latency ratio (per-group / shared): {mean_lat:>20.3}       ║");
    println!("║  threshold:  coverage ≥ 1.500  AND  latency ≤ 2.000              ║");
    println!("║  (O1) mean per-call Jaccard spread (pgrp vs shared): {ms:>9.4}   ║", ms = mean_spread);
    println!("║       — informational only, design-goal evidence (not a gate)    ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");

    let cov_pass = mean_cov >= 1.5;
    let lat_pass = mean_lat <= 2.0;
    if cov_pass && lat_pass {
        println!("║  GOAT: PASS — per-group diversifies selection at acceptable cost ║");
    } else if !cov_pass {
        println!("║  GOAT: FAIL — coverage ratio {mean_cov:.3} < 1.500 (insufficient diversification) ║");
    } else {
        println!("║  GOAT: FAIL — latency ratio {mean_lat:.3} > 2.000 (overhead too high)            ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Only assert the benchmark actually ran (verdict is informational).
    assert!(last_shared_ns > 0.0, "shared latency must be positive");
    assert!(last_pg_ns > 0.0, "per-group latency must be positive");
}
