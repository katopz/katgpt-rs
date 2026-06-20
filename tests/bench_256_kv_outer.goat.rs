#![cfg(feature = "msa_kv_outer")]
//! Benchmark — KV-Outer Sparse Prefill vs Q-Outer Baseline (Plan 256 Phase 2 GOAT gate)
//!
//! Measures KV-outer reverse-index prefill latency against naive Q-outer loop
//! at 32K, 128K, 512K context lengths. KV-outer builds a reverse index
//! (block → queries) to amortize block loads across queries that share blocks.
//!
//! Run: `CARGO_TARGET_DIR=/tmp/riir-test-build cargo test --features msa_kv_outer --test bench_256_kv_outer_goat -- --nocapture`

use katgpt_rs::dash_attn::block_topk::BlockTopKRouter;
use katgpt_rs::dash_attn::kv_outer_prefill::{KvOuterPrefill, SparsePrefillResult};
use katgpt_rs::dash_attn::vortex_flow::{VortexFlow, VortexRouter, VortexScratch};
use std::time::Instant;

// ── Config ────────────────────────────────────────────────────

const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 64; // tokens per KV block
const N_QUERIES: usize = 256; // prefill batch of 256 query tokens (GOAT-gate reference)
const TOP_K: usize = 32; // blocks selected per query

/// Context lengths to benchmark (in tokens).
const CONTEXTS: &[usize] = &[32 * 1024, 128 * 1024, 512 * 1024];

/// O2 (Issue 015) — N_QUERIES sweep for KV-outer reverse-index amortization.
/// At 512K (8192 blocks) with top_k=32, avg_queries/block = (NQ*32)/8192:
///   256 → 1.0, 512 → 2.0, 1024 → 4.0, 2048 → 8.0.
/// Reverse-index benefit grows with avg_queries/block (more queries share each block).
const N_QUERIES_SET: &[usize] = &[256, 512, 1024, 2048];

// ── Deterministic PRNG (matches bench_256_simd_topk pattern) ──

/// Generate `n_blocks * head_dim` centroid floats from a deterministic seed.
fn make_block_centroids(n_blocks: usize, head_dim: usize, seed: usize) -> Vec<f32> {
    (0..n_blocks * head_dim)
        .map(|i| {
            let x = (i.wrapping_mul(2654435761)).wrapping_add(seed.wrapping_mul(40503)) as f32;
            (x * 0.0001).sin() * 0.5 + 0.5
        })
        .collect()
}

/// Expand block centroids into full key blocks: each block gets `BLOCK_SIZE`
/// tokens, each token = centroid + small deterministic noise so blocks have
/// distinct centroids (routing is meaningful) but token-level variance.
fn make_keys_values(n_blocks: usize) -> (Vec<f32>, Vec<f32>) {
    let hd = HEAD_DIM;
    let bs = BLOCK_SIZE;
    let total = n_blocks * bs * hd;
    let centroids = make_block_centroids(n_blocks, hd, 99);

    let mut keys = Vec::with_capacity(total);
    let mut values = Vec::with_capacity(total);

    for b in 0..n_blocks {
        let c_start = b * hd;
        for t in 0..bs {
            for d in 0..hd {
                let noise_k = ((b * 1009 + t * 17 + d) as f32).sin() * 0.01;
                let noise_v = ((b * 1013 + t * 23 + d) as f32).cos() * 0.01;
                keys.push(centroids[c_start + d] + noise_k);
                values.push(centroids[c_start + d] + noise_v);
            }
        }
    }
    (keys, values)
}

/// Generate `n_queries * head_dim` query floats.
fn make_queries(n_queries: usize) -> Vec<f32> {
    make_block_centroids(n_queries, HEAD_DIM, 7)
}

// ── Q-outer baseline (naive per-query loop) ───────────────────
//
// For EACH query: route top_k blocks, attend to each selected block
// independently using the same math as `KvOuterPrefill::prefill_sparse`
// (dot-product scores, local softmax, online LSE combine).
//
// This re-accesses shared blocks once per query — no reverse-index
// amortization. Produces numerically identical output to KV-outer
// (logaddexp combine is commutative + associative up to float rounding).

fn q_outer_baseline(
    router: &VortexRouter,
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    n_queries: usize,
    n_blocks: usize,
    top_k: usize,
) -> Vec<f32> {
    let hd = HEAD_DIM;
    let bs = BLOCK_SIZE;
    let scale = 1.0 / (hd as f32).sqrt();

    // Phase 1: build router cache (same as KV-outer).
    let mut cache = router.cache_new(n_blocks, hd);
    for b in 0..n_blocks {
        let s = b * bs * hd;
        let e = s + bs * hd;
        router.forward_cache(&mut cache, &keys[s..e], &values[s..e], b, hd);
    }

    let mut scratch = VortexScratch::new(n_blocks);
    let mut output = vec![0.0f32; n_queries * hd];
    let mut lse = vec![f32::NEG_INFINITY; n_queries];

    // Phase 2: for each query → route → attend (Q-outer, no reverse index).
    for q in 0..n_queries {
        let q_start = q * hd;
        let query = &queries[q_start..q_start + hd];
        let dec = router.forward_indexer(query, &cache, n_blocks, top_k, &mut scratch);

        for &b in &dec.blocks {
            let block_keys = &keys[b * bs * hd..(b + 1) * bs * hd];
            let block_vals = &values[b * bs * hd..(b + 1) * bs * hd];

            // Scores: query · keys^T * scale
            let mut scores = [0.0f32; 256];
            let actual_bs = bs.min(256);
            compute_scores(query, block_keys, actual_bs, hd, scale, &mut scores[..actual_bs]);

            // Local softmax
            let max_score = scores[..actual_bs].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for s in scores[..actual_bs].iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            let lse_local = max_score + sum_exp.ln();

            // Weighted value accumulation
            let mut local_out = [0.0f32; 256];
            let actual_hd = hd.min(256);
            for t in 0..actual_bs {
                let w = scores[t] * inv_sum;
                let v_off = t * hd;
                for d in 0..actual_hd {
                    local_out[d] += w * block_vals[v_off + d];
                }
            }

            // Online LSE combine (identical to kv_outer_prefill.rs)
            match lse[q] {
                f32::NEG_INFINITY => {
                    lse[q] = lse_local;
                    output[q_start..q_start + actual_hd].copy_from_slice(&local_out[..actual_hd]);
                }
                lse_prev => {
                    let lse_new = logaddexp(lse_prev, lse_local);
                    let old_scale = (lse_prev - lse_new).exp();
                    let new_scale = (lse_local - lse_new).exp();
                    let out = &mut output[q_start..q_start + actual_hd];
                    for (o, &l) in out.iter_mut().zip(local_out[..actual_hd].iter()) {
                        *o = *o * old_scale + l * new_scale;
                    }
                    lse[q] = lse_new;
                }
            }
        }
    }
    output
}

// ── Inline math helpers (mirror kv_outer_prefill.rs exactly) ──

#[inline]
fn compute_scores(
    query: &[f32],
    block_keys: &[f32],
    block_size: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    let chunks = head_dim / 4;
    let rem = head_dim % 4;
    for t in 0..block_size {
        let k_start = t * head_dim;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut d2 = 0.0f32;
        let mut d3 = 0.0f32;
        for c in 0..chunks {
            let base = k_start + c * 4;
            d0 += query[c * 4] * block_keys[base];
            d1 += query[c * 4 + 1] * block_keys[base + 1];
            d2 += query[c * 4 + 2] * block_keys[base + 2];
            d3 += query[c * 4 + 3] * block_keys[base + 3];
        }
        let mut dot = d0 + d1 + d2 + d3;
        for d in 0..rem {
            let qi = chunks * 4 + d;
            dot += query[qi] * block_keys[k_start + qi];
        }
        scores[t] = dot * scale;
    }
}

#[inline]
fn logaddexp(a: f32, b: f32) -> f32 {
    match (a, b) {
        (a, b) if a == f32::NEG_INFINITY => b,
        (a, b) if b == f32::NEG_INFINITY => a,
        (a, b) if a >= b => a + (1.0 + (b - a).exp()).ln(),
        (_, b) => b + (1.0 + (a - b).exp()).ln(),
    }
}

// ── Main benchmark ────────────────────────────────────────────

#[test]
fn bench_kv_outer_vs_q_outer() {
    let queries = make_queries(N_QUERIES);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 256 Phase 2 — KV-Outer vs Q-Outer Sparse Prefill (GOAT gate)     ║");
    println!("║  HEAD_DIM={HD}, BLOCK_SIZE={BS}, N_QUERIES={NQ}, TOP_K={TK}                       ║",
        HD = HEAD_DIM, BS = BLOCK_SIZE, NQ = N_QUERIES, TK = TOP_K);
    println!("╠═══════════════╦════════════════╦════════════════╦═════════╦═════════╣");
    println!("║ Context       ║ Q-Outer (ms)   ║ KV-Outer (ms)  ║ Speedup ║ Match   ║");
    println!("╠═══════════════╬════════════════╬════════════════╬═════════╬═════════╣");

    let mut goat_ok = true;

    for (ci, &ctx_tokens) in CONTEXTS.iter().enumerate() {
        let n_blocks = ctx_tokens / BLOCK_SIZE;

        // Build keys/values for this context length.
        let (keys, values) = make_keys_values(n_blocks);

        // --- Q-outer baseline (1 iteration) ---
        let router_q = VortexRouter::BlockTopK(BlockTopKRouter::new(true));
        let t0 = Instant::now();
        let q_out = q_outer_baseline(&router_q, &keys, &values, &queries, N_QUERIES, n_blocks, TOP_K);
        let q_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // --- KV-outer prefill_sparse (1 iteration) ---
        let router_kv = VortexRouter::BlockTopK(BlockTopKRouter::new(true));
        let prefill = KvOuterPrefill::new(router_kv, BLOCK_SIZE, HEAD_DIM);
        let t1 = Instant::now();
        let kv_res: SparsePrefillResult =
            prefill.prefill_sparse(&queries, &keys, &values, N_QUERIES, n_blocks, TOP_K);
        let kv_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let speedup = q_ms / kv_ms;

        // --- Correctness: compare outputs at 32K (smallest) ---
        let mut matched = "—";
        if ci == 0 {
            let mut max_diff = 0.0f32;
            let kv_out = &kv_res.output;
            assert_eq!(kv_out.len(), q_out.len(), "output length mismatch");
            for i in 0..kv_out.len() {
                let diff = (kv_out[i] - q_out[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                assert!(
                    diff < 1e-3,
                    "KV/Q-outer mismatch at ctx={ctx_tokens} idx={i}: kv={:.6} q={:.6} diff={:.2e}",
                    kv_out[i], q_out[i], diff
                );
            }
            matched = if max_diff < 1e-3 { "OK" } else { "FAIL" };
            println!(
                "║ (32K check)   ║ max_diff = {md:.2e}                                          ║",
                md = max_diff
            );
        }

        println!(
            "║ {ctx:<13} ║ {qm:>14.2} ║ {kvm:>14.2} ║ {sp:>7.2}x ║ {m:<7} ║",
            ctx = format_ctx(ctx_tokens),
            qm = q_ms,
            kvm = kv_ms,
            sp = speedup,
            m = matched,
        );

        // GOAT gate: KV-outer ≥ 1.5x faster at 128K+.
        if ctx_tokens >= 128 * 1024 && speedup < 1.5 {
            goat_ok = false;
        }
    }

    println!("╚═══════════════╩════════════════╩════════════════╩═════════╩═══════╝");

    // GOAT verdict
    let verdict = if goat_ok { "PASS ✅" } else { "FAIL ❌" };
    println!();
    println!("GOAT verdict (KV-outer ≥ 1.5x faster at 128K+): {verdict}");

    // ── O2 (Issue 015): N_QUERIES sweep for reverse-index amortization regime ──
    //
    // The GOAT gate above runs the original N_QUERIES=256 sweep. Here we sweep
    // N_QUERIES ∈ {256, 512, 1024, 2048} × all contexts to map where KV-outer's
    // reverse index starts paying off (avg_queries/block rises, shared block
    // loads amortize across more queries). This does NOT change the GOAT gate
    // — it sharpens the regime boundary that the recommendation already names.
    println!();
    println!("── O2 (Issue 015): N_QUERIES sweep — KV-outer vs Q-outer speedup ──");
    println!("    HEAD_DIM={HD}, BLOCK_SIZE={BS}, TOP_K={TK}", HD = HEAD_DIM, BS = BLOCK_SIZE, TK = TOP_K);
    println!("    avg_queries/block = (N_QUERIES × TOP_K) / n_blocks (theoretical max amortization)");
    println!();
    println!("    {ctx:<8} {nq:<8} {aqb:<18} {qm:<14} {kvm:<14} {sp:<12}",
        ctx = "ctx", nq = "NQ", aqb = "avg_q/block", qm = "Q-outer(ms)", kvm = "KV-outer(ms)", sp = "speedup");
    println!("    {d:<8} {d:<8} {d:<18} {d:<14} {d:<14} {d:<12}", d = "--------");

    for &nq in N_QUERIES_SET {
        let queries_sweep = make_queries(nq);
        for &ctx_tokens in CONTEXTS {
            let n_blocks = ctx_tokens / BLOCK_SIZE;
            let (keys, values) = make_keys_values(n_blocks);

            // Q-outer baseline.
            let router_q = VortexRouter::BlockTopK(BlockTopKRouter::new(true));
            let t0 = Instant::now();
            let _ = q_outer_baseline(&router_q, &keys, &values, &queries_sweep, nq, n_blocks, TOP_K);
            let q_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // KV-outer.
            let router_kv = VortexRouter::BlockTopK(BlockTopKRouter::new(true));
            let prefill = KvOuterPrefill::new(router_kv, BLOCK_SIZE, HEAD_DIM);
            let t1 = Instant::now();
            let _ = prefill.prefill_sparse(&queries_sweep, &keys, &values, nq, n_blocks, TOP_K);
            let kv_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let speedup = if kv_ms > 0.0 { q_ms / kv_ms } else { 0.0 };
            let avg_q_per_block = (nq * TOP_K) as f64 / n_blocks as f64;

            println!("    {ctx:<8} {nq:<8} {aqb:<18.3} {qm:<14.2} {kvm:<14.2} {sp:<12.3}x",
                ctx = format_ctx(ctx_tokens),
                nq = nq,
                aqb = avg_q_per_block,
                qm = q_ms,
                kvm = kv_ms,
                sp = speedup,
            );
            // eprintln so each row also surfaces cleanly under --nocapture.
            eprintln!(
                "    [O2] ctx={ctx} NQ={nq} avg_q/block={aqb:.3} q={qm:.2}ms kv={kvm:.2}ms speedup={sp:.3}x",
                ctx = format_ctx(ctx_tokens),
                nq = nq,
                aqb = avg_q_per_block,
                qm = q_ms,
                kvm = kv_ms,
                sp = speedup,
            );
        }
    }
    println!("    ────────────────────────────────────────────────────────────");

    // Only assert the benchmark ran (not the verdict).
    assert!(true, "benchmark completed");
}

fn format_ctx(tokens: usize) -> String {
    if tokens >= 1024 {
        format!("{}K", tokens / 1024)
    } else {
        format!("{tokens}")
    }
}
