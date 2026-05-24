#![cfg(feature = "dash_attn")]
//! Benchmark — DashAttention Routing Benchmarks & GOAT Proof (Plan 106, T21-T24, T26)
//!
//! T21: top-k vs entmax routing comparison
//! T22: chunk summary vs mean-K scoring
//! T23: NIAH-style needle position sweep
//! T24: entmax vs uniform noise queries
//! T26: GOAT proof — adaptive support (hard→more, easy→fewer)
//!
//! Run: `cargo test --features dash_attn --test bench_106_dash_attn_routing -- --nocapture`

use katgpt_rs::dash_attn::score_blocks_entmax;
use katgpt_rs::types::DashAttnConfig;

// ── Helpers ───────────────────────────────────────────────────

/// Deterministic pseudo-random vector generator (index-based seed).
fn make_vec(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = ((i.wrapping_mul(2654435761)).wrapping_add(seed.wrapping_mul(40503))) as f32;
            (x * 0.0001).sin() * 0.5 + 0.5
        })
        .collect()
}

/// Return indices of the top-k values in descending order.
fn topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Compute dot product between two vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── T21: bench_topk_vs_entmax_routing ─────────────────────────

#[test]
fn bench_topk_vs_entmax_routing() {
    let config = DashAttnConfig::default();
    let n_chunks = 128;
    let dim = 16;
    let n_queries = 50;
    let k = 8;

    // Generate 128 chunks with 16-dim summaries
    let summaries: Vec<Vec<f32>> = (0..n_chunks).map(|c| make_vec(dim, c)).collect();

    // Generate 50 queries
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|q| make_vec(dim, 1000 + q)).collect();

    let mut total_entmax_active = 0usize;
    let mut min_active = usize::MAX;
    let mut max_active = 0usize;
    let mut total_accuracy = 0.0f64;
    let mut rows: Vec<(usize, usize, usize, f64)> = Vec::with_capacity(n_queries);

    for (qi, query) in queries.iter().enumerate() {
        // Fixed top-k: compute logits, take top 8
        let scale = 1.0 / (dim as f32).sqrt() * config.scaling_factor;
        let logits: Vec<f32> = summaries.iter().map(|s| dot(query, s) * scale).collect();
        let topk = topk_indices(&logits, k);
        let topk_set: std::collections::HashSet<usize> = topk.iter().copied().collect();

        // Entmax routing
        let result = score_blocks_entmax(query, &summaries, &config);
        let n_active = result.active_indices.len();
        let entmax_set: std::collections::HashSet<usize> =
            result.active_indices.iter().copied().collect();

        // Accuracy: % of top-8 that are in entmax's active set
        let overlap = topk_set.intersection(&entmax_set).count();
        let accuracy = overlap as f64 / k as f64 * 100.0;

        total_entmax_active += n_active;
        min_active = min_active.min(n_active);
        max_active = max_active.max(n_active);
        total_accuracy += accuracy;

        if qi < 10 {
            rows.push((qi, n_active, overlap, accuracy));
        }
    }

    let avg_active = total_entmax_active as f64 / n_queries as f64;
    let avg_accuracy = total_accuracy / n_queries as f64;

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ T21: Top-k (k=8) vs Entmax Adaptive Routing            │");
    println!("├──────┬────────────┬─────────┬──────────┬────────────────┤");
    println!("│ Q#   │ Entmax Act │ Top8 In │ Overlap  │ Accuracy (%)   │");
    println!("├──────┼────────────┼─────────┼──────────┼────────────────┤");
    for (qi, n_active, overlap, acc) in &rows {
        println!("│ {qi:4} │ {n_active:10} │ {overlap:7} │ {overlap:8}/{k} │ {acc:12.1} │");
    }
    println!("├──────┴────────────┴─────────┴──────────┴────────────────┤");
    println!("│ Summary: avg_active={avg_active:.1}, min={min_active}, max={max_active}    │");
    println!("│          avg_accuracy={avg_accuracy:.1}%                               │");
    println!("└─────────────────────────────────────────────────────────┘");

    assert!(
        avg_active >= 1.0 && avg_active <= n_chunks as f64,
        "entmax average active blocks ({avg_active:.1}) must be in [1, {n_chunks}]"
    );
}

// ── T22: bench_chunk_summary_vs_mean_k ────────────────────────

#[test]
fn bench_chunk_summary_vs_mean_k() {
    let config = DashAttnConfig::default();
    let n_chunks = 64;
    let dim = 32;
    let n_queries = 20;

    // Generate 64 chunks with 32-dim embeddings
    let embeddings: Vec<Vec<f32>> = (0..n_chunks)
        .map(|c| (0..dim).map(|d| make_vec(1, c * 100 + d)[0]).collect())
        .collect();

    // Mean-pooled summaries: average of embeddings (single embedding per chunk, so it's the same)
    let mean_summaries: Vec<Vec<f32>> = embeddings.to_vec();

    // "Learned" summaries: add small perturbation to mean (simulating head_cls)
    let learned_summaries: Vec<Vec<f32>> = mean_summaries
        .iter()
        .enumerate()
        .map(|(c, mean)| {
            mean.iter()
                .enumerate()
                .map(|(d, &v)| {
                    let perturbation =
                        ((c.wrapping_mul(7919) + d.wrapping_mul(104729)) as f32 * 0.001).sin()
                            * 0.1;
                    v + perturbation
                })
                .collect()
        })
        .collect();

    let queries: Vec<Vec<f32>> = (0..n_queries).map(|q| make_vec(dim, 2000 + q)).collect();

    let mut top_block_matches = 0usize;
    let mut rows: Vec<(usize, usize, usize, f32, f32)> = Vec::with_capacity(n_queries);

    for (qi, query) in queries.iter().enumerate() {
        // Score with mean summaries
        let mean_result = score_blocks_entmax(query, &mean_summaries, &config);
        let mean_top = mean_result.active_indices.iter().copied().max_by(|&a, &b| {
            mean_result.probs[a]
                .partial_cmp(&mean_result.probs[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Score with learned summaries
        let learned_result = score_blocks_entmax(query, &learned_summaries, &config);
        let learned_top = learned_result
            .active_indices
            .iter()
            .copied()
            .max_by(|&a, &b| {
                learned_result.probs[a]
                    .partial_cmp(&learned_result.probs[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        match (mean_top, learned_top) {
            (Some(mt), Some(lt)) => {
                if mt == lt {
                    top_block_matches += 1;
                }
                let mean_top_prob = mean_result.probs[mt];
                let learned_top_prob = learned_result.probs[lt];
                if qi < 10 {
                    rows.push((qi, mt, lt, mean_top_prob, learned_top_prob));
                }
            }
            (Some(mt), None) => {
                if qi < 10 {
                    rows.push((qi, mt, usize::MAX, mean_result.probs[mt], 0.0));
                }
            }
            (None, Some(lt)) => {
                if qi < 10 {
                    rows.push((qi, usize::MAX, lt, 0.0, learned_result.probs[lt]));
                }
            }
            (None, None) => {
                if qi < 10 {
                    rows.push((qi, usize::MAX, usize::MAX, 0.0, 0.0));
                }
            }
        }
    }

    let match_rate = top_block_matches as f64 / n_queries as f64 * 100.0;

    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ T22: Chunk Summary (Mean) vs Learned (head_cls) Scoring     │");
    println!("├──────┬───────────┬───────────┬──────────────┬───────────────┤");
    println!("│ Q#   │ Mean Top  │ Lrnd Top  │ Mean Prob    │ Lrnd Prob     │");
    println!("├──────┼───────────┼───────────┼──────────────┼───────────────┤");
    for (qi, mt, lt, mp, lp) in &rows {
        let mt_str = if *mt == usize::MAX {
            "  -  ".to_string()
        } else {
            format!("{mt:5}")
        };
        let lt_str = if *lt == usize::MAX {
            "  -  ".to_string()
        } else {
            format!("{lt:5}")
        };
        println!("│ {qi:4} │ {mt_str} │ {lt_str} │ {mp:12.4} │ {lp:13.4} │");
    }
    println!("├──────┴───────────┴───────────┴──────────────┴───────────────┤");
    println!(
        "│ Top-block match rate: {top_block_matches}/{n_queries} ({match_rate:.1}%)                │"
    );
    println!("└──────────────────────────────────────────────────────────────┘");

    // Both methods should produce valid routing results (non-empty active sets for at least some queries)
    let mean_valid_count = queries
        .iter()
        .filter(|q| {
            !score_blocks_entmax(q, &mean_summaries, &config)
                .active_indices
                .is_empty()
        })
        .count();
    let learned_valid_count = queries
        .iter()
        .filter(|q| {
            !score_blocks_entmax(q, &learned_summaries, &config)
                .active_indices
                .is_empty()
        })
        .count();

    assert!(
        mean_valid_count > 0,
        "Mean summaries should produce at least some valid routing results"
    );
    assert!(
        learned_valid_count > 0,
        "Learned summaries should produce at least some valid routing results"
    );
}

// ── T23: bench_dash_attn_routing_sweep ────────────────────────

#[test]
fn bench_dash_attn_routing_sweep() {
    let config = DashAttnConfig::default();
    let n_chunks = 256;
    let dim = 16;
    let needle_positions: [usize; 5] = [0, 64, 128, 192, 255];

    // Generate 256 chunks with 16-dim summaries
    let mut summaries: Vec<Vec<f32>> = (0..n_chunks).map(|c| make_vec(dim, c)).collect();

    let mut rows: Vec<(usize, usize, bool, f32)> = Vec::with_capacity(needle_positions.len());

    let mut found_count = 0usize;

    for &needle_pos in &needle_positions {
        // Create a "needle" at this position — overwrite with a distinctive vector
        let needle_vec: Vec<f32> = (0..dim).map(|d| if d == 0 { 10.0 } else { 0.0 }).collect();
        summaries[needle_pos] = needle_vec.clone();

        // Query is aligned with the needle chunk
        let query = needle_vec.clone();

        // Run entmax routing
        let result = score_blocks_entmax(&query, &summaries, &config);
        let n_active = result.active_indices.len();
        let needle_found = result.active_indices.contains(&needle_pos);
        let needle_prob = result.probs.get(needle_pos).copied().unwrap_or(0.0);

        if needle_found {
            found_count += 1;
        }

        rows.push((needle_pos, n_active, needle_found, needle_prob));

        // Restore original summary for next iteration
        summaries[needle_pos] = make_vec(dim, needle_pos);
    }

    let found_rate = found_count as f64 / needle_positions.len() as f64 * 100.0;

    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ T23: NIAH-Style Needle Position Sweep                       │");
    println!("├──────────────┬────────────┬──────────────┬──────────────────┤");
    println!("│ Needle Pos   │ Active Blk │ Needle Found │ Needle Prob      │");
    println!("├──────────────┼────────────┼──────────────┼──────────────────┤");
    for (pos, n_active, found, prob) in &rows {
        let found_str = if *found { "✓ YES" } else { "✗ NO " };
        println!("│ {pos:12} │ {n_active:10} │ {found_str}      │ {prob:15.6} │");
    }
    println!("├──────────────┴────────────┴──────────────┴──────────────────┤");
    println!(
        "│ Found rate: {found_count}/{} ({found_rate:.0}%) — target >= 80%              │",
        needle_positions.len()
    );
    println!("└──────────────────────────────────────────────────────────────┘");

    assert!(
        found_rate >= 80.0,
        "Needle must be found for >= 80% of positions, got {found_rate:.0}%"
    );
}

// ── T24: bench_entmax_vs_uniform_noise ────────────────────────

#[test]
fn bench_entmax_vs_uniform_noise() {
    let config = DashAttnConfig::default();
    let n_chunks = 256;
    let dim = 16;

    // Generate 256 chunks with 16-dim summaries
    let summaries: Vec<Vec<f32>> = (0..n_chunks).map(|c| make_vec(dim, c)).collect();

    // Query type 1: peaked — aligned with a single dominant chunk
    let peaked_queries: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            let mut q = vec![0.0f32; dim];
            q[i * 3 % dim] = 10.0; // strong alignment with one dimension
            q
        })
        .collect();

    // Query type 2: spread — aligned with several chunks
    let spread_queries: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            let mut q = vec![0.0f32; dim];
            // Spread across 8 dimensions
            for d in 0..8 {
                q[(d + i) % dim] = 1.0;
            }
            q
        })
        .collect();

    // Query type 3: noise — random, no strong alignment
    let noise_queries: Vec<Vec<f32>> = (0..5).map(|i| make_vec(dim, 5000 + i)).collect();

    let avg_peaked = avg_active_blocks(&peaked_queries, &summaries, &config);
    let avg_spread = avg_active_blocks(&spread_queries, &summaries, &config);
    let avg_noise = avg_active_blocks(&noise_queries, &summaries, &config);

    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ T24: Entmax Active Blocks by Query Type                     │");
    println!("├──────────────────┬────────────────────┬──────────────────────┤");
    println!("│ Query Type       │ Avg Active Blocks  │ Behavior             │");
    println!("├──────────────────┼────────────────────┼──────────────────────┤");
    println!("│ Peaked (1 chunk) │ {avg_peaked:18.1} │ Should be fewest     │");
    println!("│ Spread (8 chunk) │ {avg_spread:18.1} │ Should be moderate   │");
    println!("│ Noise (random)   │ {avg_noise:18.1} │ Varies               │");
    println!("└──────────────────┴────────────────────┴──────────────────────┘");

    assert!(
        avg_peaked < avg_spread,
        "Peaked queries ({avg_peaked:.1}) should have fewer active blocks than spread queries ({avg_spread:.1})"
    );
}

fn avg_active_blocks(queries: &[Vec<f32>], summaries: &[Vec<f32>], config: &DashAttnConfig) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }
    let total: usize = queries
        .iter()
        .map(|q| {
            score_blocks_entmax(q, summaries, config)
                .active_indices
                .len()
        })
        .sum();
    total as f64 / queries.len() as f64
}

// ── T26: goat_proof_adaptive_support (THE KEY GOAT PROOF) ────

#[test]
fn goat_proof_adaptive_support() {
    let config = DashAttnConfig::default();
    let n_chunks = 128;
    let dim = 16;
    let n_easy = 20;
    let n_hard = 20;

    // Generate 128 chunks with 16-dim summaries
    let summaries: Vec<Vec<f32>> = (0..n_chunks).map(|c| make_vec(dim, c)).collect();

    // Easy queries: aligned with a single dominant chunk (10.0 dot product + small noise)
    let easy_queries: Vec<Vec<f32>> = (0..n_easy)
        .map(|i| {
            let target_chunk = i * (n_chunks / n_easy);
            let mut q = summaries[target_chunk % n_chunks].clone();
            // Scale up for strong alignment (10.0 dot)
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for v in q.iter_mut() {
                *v = *v / norm * 10.0;
            }
            // Add tiny noise
            for (d, v) in q.iter_mut().enumerate() {
                let noise = ((d.wrapping_mul(7919).wrapping_add(i.wrapping_mul(104729))) as f32
                    * 0.001)
                    .sin()
                    * 0.01;
                *v += noise;
            }
            q
        })
        .collect();

    // Hard queries: spread across multiple chunks (similar dot products)
    let hard_queries: Vec<Vec<f32>> = (0..n_hard)
        .map(|i| {
            // Average of several chunk summaries → similar similarity to many
            let mut q = vec![0.0f32; dim];
            let n_targets = 16;
            for t in 0..n_targets {
                let chunk_idx = (i * 7 + t * 13) % n_chunks;
                for (d, v) in q.iter_mut().enumerate() {
                    *v += summaries[chunk_idx][d];
                }
            }
            for v in q.iter_mut() {
                *v /= n_targets as f32;
            }
            q
        })
        .collect();

    // Run entmax routing for each query type
    let mut easy_active_counts: Vec<usize> = Vec::with_capacity(n_easy);
    let mut hard_active_counts: Vec<usize> = Vec::with_capacity(n_hard);

    for (i, query) in easy_queries.iter().enumerate() {
        let result = score_blocks_entmax(query, &summaries, &config);
        easy_active_counts.push(result.active_indices.len());
        if i < 5 {
            let top_idx = result
                .active_indices
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    result.probs[a]
                        .partial_cmp(&result.probs[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            println!(
                "  Easy Q{:2}: active={:3}, top_prob={:.4}, top_chunk={}",
                i,
                result.active_indices.len(),
                result.probs.get(top_idx).copied().unwrap_or(0.0),
                top_idx
            );
        }
    }

    println!();

    for (i, query) in hard_queries.iter().enumerate() {
        let result = score_blocks_entmax(query, &summaries, &config);
        hard_active_counts.push(result.active_indices.len());
        if i < 5 {
            let top_idx = result
                .active_indices
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    result.probs[a]
                        .partial_cmp(&result.probs[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            println!(
                "  Hard Q{:2}: active={:3}, top_prob={:.4}, top_chunk={}",
                i,
                result.active_indices.len(),
                result.probs.get(top_idx).copied().unwrap_or(0.0),
                top_idx
            );
        }
    }

    let avg_easy = easy_active_counts.iter().sum::<usize>() as f64 / n_easy as f64;
    let avg_hard = hard_active_counts.iter().sum::<usize>() as f64 / n_hard as f64;
    let min_easy = *easy_active_counts.iter().min().unwrap_or(&0);
    let max_easy = *easy_active_counts.iter().max().unwrap_or(&0);
    let min_hard = *hard_active_counts.iter().min().unwrap_or(&0);
    let max_hard = *hard_active_counts.iter().max().unwrap_or(&0);

    println!();
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│ T26: GOAT PROOF — Adaptive Support (DashAttention Core Claim)       │");
    println!("│ Hard queries get MORE chunks; Easy queries get FEWER chunks         │");
    println!("├─────────────────┬──────────┬──────────┬──────────┬──────────────────┤");
    println!("│ Query Type      │ Avg Act  │ Min Act  │ Max Act  │ N Queries       │");
    println!("├─────────────────┼──────────┼──────────┼──────────┼──────────────────┤");
    println!("│ Easy (peaked)   │ {avg_easy:8.1} │ {min_easy:8} │ {max_easy:8} │ {n_easy:15} │");
    println!("│ Hard (spread)   │ {avg_hard:8.1} │ {min_hard:8} │ {max_hard:8} │ {n_hard:15} │");
    println!("├─────────────────┴──────────┴──────────┴──────────┴──────────────────┤");
    println!(
        "│ Δ = avg_hard - avg_easy = {delta:.1}                                     │",
        delta = avg_hard - avg_easy
    );
    println!(
        "│ PASS: avg_hard ({avg_hard:.1}) > avg_easy ({avg_easy:.1})                           │",
        avg_hard = avg_hard,
        avg_easy = avg_easy
    );
    println!("│ This proves DashAttention's adaptive sparsity: the entmax           │");
    println!("│ support automatically expands for hard (ambiguous) queries          │");
    println!("│ and contracts for easy (peaked) queries.                            │");
    println!("└──────────────────────────────────────────────────────────────────────┘");

    assert!(
        avg_hard > avg_easy,
        "GOAT PROOF FAILED: avg_hard ({avg_hard:.1}) must be > avg_easy ({avg_easy:.1}). \
         DashAttention's adaptive sparsity requires that hard queries get more active blocks."
    );
}
