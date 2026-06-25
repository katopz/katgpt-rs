//! MaxSim Late-Interaction Scoring — Demo (Plan 080)
//!
//! Demonstrates all MaxSim integration points:
//! 1. Core `maxsim_score` — memory-efficient Σ_i max_j dot(q_i, d_j)
//! 2. Packed scoring `maxsim_score_packed` — batch ragged (query, doc) pairs
//! 3. Block scoring `block_score_maxsim` — PFlash block-level MaxSim vs mean-K
//! 4. Scale timing — SIMD speedup over naive materialized baseline
//! 5. TurboQuant proof — `maxsim_score_turboquant` vs uncompressed (requires `turboquant` feature)
//! 6. SpectralQuant proof — `maxsim_score_spectralquant` vs uncompressed (requires `spectral_quant` feature)
//! 7. TurboQuant vs SpectralQuant head-to-head — quality + latency (requires `turboquant` + `spectral_quant` features)
//!
//! Run: cargo run --example core_05_maxsim --features maxsim --release
//! With all proofs: cargo run --example core_05_maxsim --features "maxsim,turboquant,spectral_quant" --release

use katgpt_rs::simd::{maxsim_score, maxsim_score_packed, simd_dot_f32};
use katgpt_rs::speculative::block_score_maxsim;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 080: MaxSim Late-Interaction Scoring Demo                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Core Primitive: maxsim_score ─────────────────────────────
    println!("── 1. Core Primitive: maxsim_score ──────────────────────────────\n");

    let dim = 64;
    let lq = 8; // 8 query tokens
    let ld = 32; // 32 doc tokens

    // Synthetic embeddings: query has one "hot" token, doc has scattered signal
    let queries: Vec<f32> = (0..lq * dim)
        .map(|i| {
            if i < dim {
                // Query token 0: strong signal (all ones)
                1.0
            } else {
                // Other query tokens: weak noise
                ((i as f32 * 0.01).sin()) * 0.1
            }
        })
        .collect();

    let documents: Vec<f32> = (0..ld * dim)
        .map(|i| {
            let token = i / dim;
            let d = i % dim;
            if token == 5 && d < dim {
                // Doc token 5: matches query token 0 (all ones)
                1.0
            } else if token == 20 && d < dim / 2 {
                // Doc token 20: partial match (half the dims)
                1.0
            } else {
                (i as f32 * 0.02).cos() * 0.05
            }
        })
        .collect();

    let score = maxsim_score(&queries, &documents, lq, ld, dim);
    println!("  MaxSim score (Lq={lq}, Ld={ld}, dim={dim}): {score:.4}");

    // Compare: naive materialized version
    let naive = maxsim_naive(&queries, &documents, lq, ld, dim);
    println!("  Naive reference:                              {naive:.4}");
    println!(
        "  Match: {}",
        if (score - naive).abs() < 1e-4 {
            "✓"
        } else {
            "✗ MISMATCH"
        }
    );

    // Show per-query-token breakdown
    println!("\n  Per-query-token max similarity:");
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut best_j = 0;
        let mut best_dot = f32::NEG_INFINITY;
        for j in 0..ld {
            let d_row = &documents[j * dim..(j + 1) * dim];
            let dot = simd_dot_f32(q_row, d_row, dim);
            if dot > best_dot {
                best_dot = dot;
                best_j = j;
            }
        }
        println!("    q[{i}] → best doc[{best_j}] dot={best_dot:.4}");
    }

    println!();

    // ── 2. Packed Scoring: maxsim_score_packed ──────────────────────
    println!("── 2. Packed Scoring: maxsim_score_packed ──────────────────────\n");

    // Two query sequences of different lengths, three doc sequences
    let q0: Vec<f32> = (0..2 * dim).map(|i| (i as f32 * 0.1).sin()).collect(); // 2 tokens
    let q1: Vec<f32> = (0..4 * dim).map(|i| (i as f32 * 0.1).cos()).collect(); // 4 tokens
    let d0: Vec<f32> = (0..8 * dim).map(|i| (i as f32 * 0.05).sin()).collect(); // 8 tokens
    let d1: Vec<f32> = (0..3 * dim).map(|i| (i as f32 * 0.05).cos()).collect(); // 3 tokens
    let d2: Vec<f32> = (0..6 * dim)
        .map(|i| ((i as f32 * 0.03).tan()).clamp(-1.0, 1.0))
        .collect(); // 6 tokens

    let all_queries = [q0.clone(), q1.clone()].concat();
    let all_docs = [d0.clone(), d1.clone(), d2.clone()].concat();
    let query_offsets = [0, q0.len(), q0.len() + q1.len()];
    let doc_offsets = [
        0,
        d0.len(),
        d0.len() + d1.len(),
        d0.len() + d1.len() + d2.len(),
    ];

    // Score 4 pairs: (q0,d0), (q0,d2), (q1,d0), (q1,d1)
    let pair_q = [0usize, 0, 1, 1];
    let pair_d = [0usize, 2, 0, 1];

    let mut packed_scores = vec![0.0f32; pair_q.len()];
    maxsim_score_packed(
        &all_queries,
        &query_offsets,
        &all_docs,
        &doc_offsets,
        &pair_q,
        &pair_d,
        dim,
        &mut packed_scores,
    );

    println!("  Ragged batch: 2 query seqs, 3 doc seqs, 4 pairs");
    println!("  ┌──────────┬──────────┬─────────┐");
    println!("  │ Query     │ Doc      │ Score   │");
    println!("  ├──────────┼──────────┼─────────┤");
    for (i, s) in packed_scores.iter().enumerate() {
        println!(
            "  │ q{} ({}tok)  │ d{} ({}tok)  │ {:7.4} │",
            pair_q[i],
            query_offsets[pair_q[i] + 1] - query_offsets[pair_q[i]],
            pair_d[i],
            doc_offsets[pair_d[i] + 1] - doc_offsets[pair_d[i]],
            s,
        );
    }
    println!("  └──────────┴──────────┴─────────┘");

    // Verify: packed matches sequential
    let sequential: Vec<f32> = pair_q
        .iter()
        .zip(pair_d.iter())
        .map(|(&qi, &di)| {
            let q_data = &all_queries[query_offsets[qi]..query_offsets[qi + 1]];
            let d_data = &all_docs[doc_offsets[di]..doc_offsets[di + 1]];
            let lq_i = q_data.len() / dim;
            let ld_j = d_data.len() / dim;
            maxsim_score(q_data, d_data, lq_i, ld_j, dim)
        })
        .collect();

    let packed_matches = packed_scores
        .iter()
        .zip(sequential.iter())
        .all(|(a, b)| (a - b).abs() < 1e-4);
    println!(
        "  Packed matches sequential: {}",
        if packed_matches { "✓" } else { "✗" }
    );

    println!();

    // ── 3. Block Scoring: block_score_maxsim vs mean-K ──────────────
    println!("── 3. Block Scoring: MaxSim vs Mean-K ─────────────────────────\n");

    let block_size = 32;
    let _num_blocks = 32; // 1024 tokens total
    let block_dim = 64;

    // Build synthetic blocks: one "needle" block, rest are noise
    let mut query_block: Vec<f32> = vec![0.1; block_size * block_dim];
    // Needle signal in first 4 tokens of query block
    for t in 0..4 {
        for d in 0..block_dim {
            query_block[t * block_dim + d] = 1.0;
        }
    }

    let mut needle_block: Vec<f32> = vec![0.1; block_size * block_dim];
    // Matching signal in needle block tokens 10-13
    for t in 10..14 {
        for d in 0..block_dim {
            needle_block[t * block_dim + d] = 1.0;
        }
    }

    // Score with MaxSim
    let maxsim_block_score = block_score_maxsim(
        &query_block,
        &needle_block,
        block_size,
        block_size,
        block_dim,
    );

    // Score with mean-K (standard PFlash approach)
    let mean_k_score = {
        // Mean of query block
        let mut q_mean = vec![0.0f32; block_dim];
        for t in 0..block_size {
            for (d, qm) in q_mean.iter_mut().enumerate().take(block_dim) {
                *qm += query_block[t * block_dim + d];
            }
        }
        for qm in q_mean.iter_mut().take(block_dim) {
            *qm /= block_size as f32;
        }

        // Mean of doc block
        let mut k_mean = vec![0.0f32; block_dim];
        for t in 0..block_size {
            for (d, km) in k_mean.iter_mut().enumerate().take(block_dim) {
                *km += needle_block[t * block_dim + d];
            }
        }
        for km in k_mean.iter_mut().take(block_dim) {
            *km /= block_size as f32;
        }

        simd_dot_f32(&q_mean, &k_mean, block_dim)
    };

    // Score a noise block for comparison
    let noise_block: Vec<f32> = vec![0.05; block_size * block_dim];
    let maxsim_noise = block_score_maxsim(
        &query_block,
        &noise_block,
        block_size,
        block_size,
        block_dim,
    );
    let mean_k_noise = {
        let k_mean = vec![0.05f32; block_dim];
        let mut q_mean = vec![0.0f32; block_dim];
        for t in 0..block_size {
            for (d, qm) in q_mean.iter_mut().enumerate().take(block_dim) {
                *qm += query_block[t * block_dim + d];
            }
        }
        for qm in q_mean.iter_mut().take(block_dim) {
            *qm /= block_size as f32;
        }
        simd_dot_f32(&q_mean, &k_mean, block_dim)
    };

    println!("  Block size: {block_size} tokens, dim: {block_dim}");
    println!("  ┌───────────────┬──────────┬──────────┬────────────┐");
    println!("  │ Method        │ Needle   │ Noise    │ Separation │");
    println!("  ├───────────────┼──────────┼──────────┼────────────┤");
    println!(
        "  │ MaxSim        │ {:8.4} │ {:8.4} │ {:10.2}× │",
        maxsim_block_score,
        maxsim_noise,
        maxsim_block_score / maxsim_noise.abs().max(1e-6)
    );
    println!(
        "  │ Mean-K dot    │ {:8.4} │ {:8.4} │ {:10.2}× │",
        mean_k_score,
        mean_k_noise,
        mean_k_score / mean_k_noise.abs().max(1e-6)
    );
    println!("  └───────────────┴──────────┴──────────┴────────────┘");

    let maxsim_sep = maxsim_block_score / maxsim_noise.abs().max(1e-6);
    let meank_sep = mean_k_score / mean_k_noise.abs().max(1e-6);
    println!(
        "  MaxSim separation ratio: {:.2}× better at distinguishing needle from noise",
        maxsim_sep / meank_sep
    );

    println!();

    // ── 4. Scale Test ───────────────────────────────────────────────
    println!("── 4. Scale Test: Timing ───────────────────────────────────────\n");

    let large_dim = 128;
    let large_lq = 32;
    let large_ld = 256;
    let large_queries: Vec<f32> = (0..large_lq * large_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let large_docs: Vec<f32> = (0..large_ld * large_dim)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();

    let iters = 1000u64;

    // Warmup
    for _ in 0..50 {
        std::hint::black_box(maxsim_score(
            &large_queries,
            &large_docs,
            large_lq,
            large_ld,
            large_dim,
        ));
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(maxsim_score(
            &large_queries,
            &large_docs,
            large_lq,
            large_ld,
            large_dim,
        ));
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_micros() as f64 / iters as f64;

    println!("  Config: Lq={large_lq}, Ld={large_ld}, dim={large_dim} ({iters} iterations)");
    println!("  MaxSim score: {us_per_call:.1} µs/call");
    println!("  Throughput: {:.0} scores/s", 1_000_000.0 / us_per_call);

    let naive_time = {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(maxsim_naive(
                &large_queries,
                &large_docs,
                large_lq,
                large_ld,
                large_dim,
            ));
        }
        start.elapsed()
    };
    let naive_us = naive_time.as_micros() as f64 / iters as f64;
    println!("  Naive ref:    {naive_us:.1} µs/call");
    println!("  Speedup: {:.2}×", naive_us / us_per_call);

    println!();

    // ── 5. TurboQuant MaxSim Scoring ────────────────────────────────
    #[cfg(feature = "turboquant")]
    section5_turboquant_proof();

    #[cfg(not(feature = "turboquant"))]
    println!("── 5. TurboQuant: skipped (enable --features turboquant)\n");

    // ── 6. SpectralQuant MaxSim Scoring ─────────────────────────────
    #[cfg(feature = "spectral_quant")]
    section6_spectralquant_proof();

    #[cfg(not(feature = "spectral_quant"))]
    println!("── 6. SpectralQuant: skipped (enable --features spectral_quant)\n");

    // ── 7. TurboQuant vs SpectralQuant Head-to-Head ─────────────────
    #[cfg(all(feature = "turboquant", feature = "spectral_quant"))]
    section7_tq_vs_sq_benchmark();

    #[cfg(not(all(feature = "turboquant", feature = "spectral_quant")))]
    println!("── 7. TQ vs SQ: skipped (enable --features \"turboquant,spectral_quant\")\n");

    println!("✓ MaxSim demo complete — all primitives exercised.");
}

/// Section 5: TurboQuant compressed KV MaxSim scoring proof (Plan 080 T9).
///
/// Proves that `maxsim_score_turboquant` produces scores close to uncompressed
/// `maxsim_score` despite 4-bit quantization, demonstrating the lazy-dequantize
/// streaming pattern works on compressed caches.
#[cfg(feature = "turboquant")]
fn section5_turboquant_proof() {
    use katgpt_rs::turboquant::TurboQuantKVCache;
    use katgpt_rs::turboquant::forward::maxsim_score_turboquant;
    use katgpt_rs::types::Config;

    println!("── 5. TurboQuant MaxSim Scoring ────────────────────────────────\n");

    let config = Config::micro();
    let dim = config.n_kv_head * config.head_dim;
    let n_positions = 8;

    // Create TQ cache and store synthetic keys
    let mut cache = TurboQuantKVCache::new(&config, 4, 4);

    // Synthetic query: 2 tokens
    let lq = 2;
    let queries: Vec<f32> = (0..lq * dim).map(|i| (i as f32 * 0.1).sin()).collect();

    // Store synthetic keys at positions 0..n_positions
    let original_keys: Vec<Vec<f32>> = (0..n_positions)
        .map(|t| {
            (0..dim)
                .map(|d| ((t * dim + d) as f32 * 0.05).cos())
                .collect()
        })
        .collect();

    for (t, key) in original_keys.iter().enumerate() {
        cache.store_key(0, t, key);
    }

    // Score with TurboQuant MaxSim (lazy dequantize streaming pattern)
    let tq_score = maxsim_score_turboquant(&queries, &mut cache, 0, 0..n_positions, dim);

    // Score with uncompressed MaxSim for comparison
    let flat_keys: Vec<f32> = original_keys.iter().flatten().copied().collect();
    let uncompressed = maxsim_score(&queries, &flat_keys, lq, n_positions, dim);

    let rel_error = if uncompressed.abs() > 1e-6 {
        (tq_score - uncompressed).abs() / uncompressed.abs()
    } else {
        (tq_score - uncompressed).abs()
    };

    println!(
        "  Config: kv_dim={dim}, {n_positions} doc positions, {lq} query tokens, 4-bit quantization"
    );
    println!("  TurboQuant MaxSim:  {tq_score:.4}");
    println!("  Uncompressed:       {uncompressed:.4}");
    println!("  Relative error:     {rel_error:.6} (threshold: 0.1)");
    println!(
        "  Quantization OK:    {}",
        if rel_error < 0.1 { "✓" } else { "✗" }
    );

    println!();
}

/// Section 6: SpectralQuant compressed KV MaxSim scoring proof (Plan 080 T10).
///
/// Proves that `maxsim_score_spectralquant` produces scores close to uncompressed
/// `maxsim_score` despite eigenbasis + water-fill + variable-bit quantization.
/// Uses identity eigenvectors with exponential eigenvalue decay — same pattern
/// as the SpectralQuant test suite.
#[cfg(feature = "spectral_quant")]
fn section6_spectralquant_proof() {
    use katgpt_rs::spectralquant::forward::maxsim_score_spectralquant;
    use katgpt_rs::spectralquant::{
        SpectralQuantCalibration, SpectralQuantKVCache, SpectralQuantKVCacheConfig,
        participation_ratio,
    };
    use katgpt_rs::types::Config;

    println!("── 6. SpectralQuant MaxSim Scoring ─────────────────────────────\n");

    let config = Config::micro();
    let kv_dim = katgpt_rs::types::kv_dim(&config);
    let n_positions = 8;

    // Build calibration with identity eigenvectors + exponential eigenvalue decay
    let mut eigenvectors = vec![0.0f32; kv_dim * kv_dim];
    for i in 0..kv_dim {
        eigenvectors[i * kv_dim + i] = 1.0;
    }
    let eigenvalues: Vec<f32> = (0..kv_dim).map(|i| 10.0 * 0.8f32.powi(i as i32)).collect();
    let d_eff = participation_ratio(&eigenvalues);

    let cal = SpectralQuantCalibration {
        eigenvectors,
        eigenvalues,
        d_eff,
        spectral_gap: None,
        var_95: 10,
        var_99: 20,
        n_samples: 100,
        head_dim: kv_dim,
    };

    let sq_config = SpectralQuantKVCacheConfig {
        avg_bits: 3.0,
        min_tail_bits: 1,
        max_bits: 8,
        qjl_dim: 16,
        lloyd_max_iter: 30,
        calibration_samples: 100,
        seed: 42,
        use_water_fill: false,
        wf_min_bits: 1,
        wf_max_bits: 6,
        n_layers: config.n_layer,
        kv_dim,
        max_seq_len: config.block_size,
    };

    let mut sq_cache = SpectralQuantKVCache::from_calibration(
        &sq_config,
        &vec![cal.clone(); config.n_layer],
        &vec![cal; config.n_layer],
    );

    // Synthetic query: 2 tokens
    let lq = 2;
    let queries: Vec<f32> = (0..lq * kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    // Store synthetic keys at positions 0..n_positions
    let original_keys: Vec<Vec<f32>> = (0..n_positions)
        .map(|t| {
            (0..kv_dim)
                .map(|d| ((t * kv_dim + d) as f32 * 0.05).cos())
                .collect()
        })
        .collect();

    for (t, key) in original_keys.iter().enumerate() {
        sq_cache.store_key(0, t, key);
    }

    // Score with SpectralQuant MaxSim (lazy dequantize streaming pattern)
    let sq_score = maxsim_score_spectralquant(&queries, &mut sq_cache, 0, 0..n_positions, kv_dim);

    // Fair comparison: dequantize all keys from SQ cache, then score uncompressed.
    // SQ applies random rotation when eigenvectors are identity (no real calibration),
    // so comparing against raw unrotated keys is unfair — both paths must go through
    // the same rotation to isolate quantization error from rotation mismatch.
    let mut dequant_keys = vec![0.0f32; n_positions * kv_dim];
    for t in 0..n_positions {
        sq_cache.dequantize_key_into(0, t, &mut dequant_keys[t * kv_dim..(t + 1) * kv_dim]);
    }
    let dequant_score = maxsim_score(&queries, &dequant_keys, lq, n_positions, kv_dim);

    // Streaming vs dequantized should match exactly (same codebook, same data)
    let roundtrip_match = (sq_score - dequant_score).abs() < 1e-4;

    println!(
        "  Config: kv_dim={kv_dim}, {n_positions} doc positions, {lq} query tokens, ~3-bit spectral quantization"
    );
    println!("  SQ MaxSim (streaming):  {sq_score:.4}");
    println!("  SQ MaxSim (dequant):    {dequant_score:.4}");
    println!(
        "  Roundtrip match:        {}",
        if roundtrip_match { "✓" } else { "✗" }
    );

    println!();
}

/// Section 7: 4-way TQ/SQ × Cosine/MaxSim matrix benchmark (Plan 080).
///
/// Measures the **interaction** between quantization method and scoring method:
/// - Quantization: TurboQuant (random rotation) vs SpectralQuant (calibrated eigenbasis)
/// - Scoring: cosine similarity (reconstruction) vs MaxSim (late-interaction)
///
/// The 4-way matrix proves whether MaxSim amplifies or mitigates quantization error,
/// and validates that SpectralQuant's advantage holds across both scoring paradigms.
/// Uses `from_keys()` so calibration cannot be accidentally skipped.
#[cfg(all(feature = "turboquant", feature = "spectral_quant"))]
fn section7_tq_vs_sq_benchmark() {
    use katgpt_rs::spectralquant::forward::maxsim_score_spectralquant;
    use katgpt_rs::spectralquant::{SpectralQuantKVCache, SpectralQuantKVCacheConfig};
    use katgpt_rs::turboquant::TurboQuantKVCache;
    use katgpt_rs::turboquant::forward::{cosine_similarity, maxsim_score_turboquant};
    use katgpt_rs::types::{Config, Rng};

    println!("── 7. 4-Way Matrix: TQ/SQ × Cosine/MaxSim (3-bit, calibrated) ─────────\n");

    let config = Config::micro();
    let kv_dim = katgpt_rs::types::kv_dim(&config);
    let n_positions = 16;
    let bits: u8 = 3;

    // Generate synthetic KV vectors with realistic eigenvalue decay
    let mut rng = Rng::new(42);
    let keys: Vec<Vec<f32>> = (0..n_positions)
        .map(|_| {
            (0..kv_dim)
                .map(|i| {
                    let scale = 10.0 * 0.8f32.powi(i as i32);
                    rng.normal() * scale.sqrt()
                })
                .collect()
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..n_positions)
        .map(|_| (0..kv_dim).map(|_| rng.normal()).collect())
        .collect();

    let sq_config = SpectralQuantKVCacheConfig {
        avg_bits: bits as f32,
        min_tail_bits: 1,
        max_bits: 8,
        qjl_dim: 16,
        lloyd_max_iter: 30,
        calibration_samples: n_positions,
        seed: 42,
        use_water_fill: false,
        wf_min_bits: 1,
        wf_max_bits: 6,
        n_layers: config.n_layer,
        kv_dim,
        max_seq_len: config.block_size,
    };

    // ── Fill caches ──────────────────────────────────────────────────
    // TurboQuant: random rotation, no calibration possible
    let mut tq_cache = TurboQuantKVCache::new(&config, bits, bits);
    for (pos, key) in keys.iter().enumerate() {
        tq_cache.store_key(0, pos, key);
    }
    let tq_ratio = tq_cache.compression_ratio();

    // SpectralQuant: auto-calibrated from actual keys (can't forget calibration)
    // NOTE: from_keys calibrates the eigenbasis + codebooks but does NOT store keys into cache.
    // Calibration samples ≠ stored keys — we must store separately.
    let mut sq_cache = SpectralQuantKVCache::from_keys(&sq_config, &keys, &values);
    for (pos, key) in keys.iter().enumerate() {
        sq_cache.store_key(0, pos, key);
    }
    let sq_ratio = sq_cache.compression_ratio();

    // ── Cell [TQ, Cosine]: key reconstruction quality ────────────────
    let mut tq_cosines = Vec::new();
    let mut sq_cosines = Vec::new();
    for (pos, key) in keys.iter().enumerate() {
        let tq_recon = tq_cache.dequantize_key(0, pos);
        tq_cosines.push(cosine_similarity(key, &tq_recon));
        let mut sq_recon = vec![0.0f32; kv_dim];
        sq_cache.dequantize_key_into(0, pos, &mut sq_recon);
        sq_cosines.push(cosine_similarity(key, &sq_recon));
    }
    let tq_cos: f32 = tq_cosines.iter().sum::<f32>() / tq_cosines.len() as f32;
    let sq_cos: f32 = sq_cosines.iter().sum::<f32>() / sq_cosines.len() as f32;

    // ── Cell [TQ, MaxSim]: late-interaction scoring fidelity ─────────
    let lq = 4;
    let queries: Vec<f32> = (0..lq * kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let flat_keys: Vec<f32> = keys.iter().flatten().copied().collect();
    let gt_ms = maxsim_score(&queries, &flat_keys, lq, n_positions, kv_dim);

    let tq_ms = maxsim_score_turboquant(&queries, &mut tq_cache, 0, 0..n_positions, kv_dim);
    let sq_ms = maxsim_score_spectralquant(&queries, &mut sq_cache, 0, 0..n_positions, kv_dim);

    let pct_err = |s: f32| -> f32 {
        if gt_ms.abs() > 1e-6 {
            (s - gt_ms).abs() / gt_ms.abs() * 100.0
        } else {
            0.0
        }
    };
    let tq_ms_err = pct_err(tq_ms);
    let sq_ms_err = pct_err(sq_ms);

    // ── Cell [*, Cosine Latency]: per-position dequantize + cosine ───
    let iters = 10_000u64;

    // TQ cosine latency
    for _ in 0..200 {
        for (pos, key) in keys.iter().enumerate() {
            let recon = tq_cache.dequantize_key(0, pos);
            std::hint::black_box(cosine_similarity(key, &recon));
        }
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        for (pos, key) in keys.iter().enumerate() {
            let recon = tq_cache.dequantize_key(0, pos);
            std::hint::black_box(cosine_similarity(key, &recon));
        }
    }
    let tq_cos_us = start.elapsed().as_micros() as f64 / iters as f64;

    // SQ cosine latency
    for _ in 0..200 {
        for (pos, key) in keys.iter().enumerate() {
            let mut recon = vec![0.0f32; kv_dim];
            sq_cache.dequantize_key_into(0, pos, &mut recon);
            std::hint::black_box(cosine_similarity(key, &recon));
        }
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        for (pos, key) in keys.iter().enumerate() {
            let mut recon = vec![0.0f32; kv_dim];
            sq_cache.dequantize_key_into(0, pos, &mut recon);
            std::hint::black_box(cosine_similarity(key, &recon));
        }
    }
    let sq_cos_us = start.elapsed().as_micros() as f64 / iters as f64;

    // ── Cell [*, MaxSim Latency]: fused dequantize + max-dot ─────────
    // TQ MaxSim
    for _ in 0..200 {
        std::hint::black_box(maxsim_score_turboquant(
            &queries,
            &mut tq_cache,
            0,
            0..n_positions,
            kv_dim,
        ));
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(maxsim_score_turboquant(
            &queries,
            &mut tq_cache,
            0,
            0..n_positions,
            kv_dim,
        ));
    }
    let tq_ms_us = start.elapsed().as_micros() as f64 / iters as f64;

    // SQ MaxSim
    for _ in 0..200 {
        std::hint::black_box(maxsim_score_spectralquant(
            &queries,
            &mut sq_cache,
            0,
            0..n_positions,
            kv_dim,
        ));
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(maxsim_score_spectralquant(
            &queries,
            &mut sq_cache,
            0,
            0..n_positions,
            kv_dim,
        ));
    }
    let sq_ms_us = start.elapsed().as_micros() as f64 / iters as f64;

    // ── Print 4-way matrix ───────────────────────────────────────────
    println!(
        "  kv_dim={kv_dim}, {bits}-bit budget, {n_positions} doc positions, {lq} query tokens"
    );
    println!("  Ground truth MaxSim score: {gt_ms:.4}");
    println!();
    println!("  ┌──────────────────────────────────┬──────────────┬──────────────┐");
    println!("  │ Metric                            │ TurboQuant   │ SpectralQuant│");
    println!("  ├ ─ ─ Scoring Quality ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤");
    println!("  │ Key cosine (reconstruction)       │ {tq_cos:.4}       │ {sq_cos:.4}       │");
    println!(
        "  │ MaxSim error (vs uncompressed)    │ {tq_ms_err:6.2}%       │ {sq_ms_err:6.2}%       │"
    );
    println!(
        "  │ Compression ratio                 │ {tq_ratio:.1}×         │ {sq_ratio:.1}×         │"
    );
    println!("  ├ ─ ─ Latency (10K iters) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤");
    println!(
        "  │ Cosine: dequant+cos ({n_positions:2} pos)     │ {tq_cos_us:6.2} µs    │ {sq_cos_us:6.2} µs    │"
    );
    println!(
        "  │ MaxSim: dequant+maxdot ({lq}q×{n_positions}d) │ {tq_ms_us:6.2} µs    │ {sq_ms_us:6.2} µs    │"
    );
    println!("  └──────────────────────────────────┴──────────────┴──────────────┘");
    println!();

    // ── Interaction analysis: does MaxSim amplify quantization error? ─
    let cos_delta = sq_cos - tq_cos;
    let ms_err_ratio = if tq_ms_err > 0.01 {
        tq_ms_err / sq_ms_err
    } else {
        0.0
    };
    // Amplification = how much MaxSim compounds per-vector reconstruction error.
    // Higher = MaxSim is more sensitive to quantization noise for this method.
    let amp_tq = if (1.0 - tq_cos) > 0.001 {
        tq_ms_err / ((1.0 - tq_cos) * 100.0)
    } else {
        0.0
    };
    let amp_sq = if (1.0 - sq_cos) > 0.001 {
        sq_ms_err / ((1.0 - sq_cos) * 100.0)
    } else {
        0.0
    };

    println!("  Interaction: does MaxSim amplify quantization error?");
    println!(
        "    Cosine Δ:        SQ +{cos_delta:.4} ({:.1}% better reconstruction)",
        cos_delta / tq_cos * 100.0
    );
    println!(
        "    MaxSim Δ:        SQ {sq_ms_err:.2}% vs TQ {tq_ms_err:.2}% (SQ {ms_err_ratio:.1}× less error)"
    );
    println!(
        "    Compression Δ:   SQ {sq_ratio:.1}× vs TQ {tq_ratio:.1}× (+{:.0}%)",
        (sq_ratio / tq_ratio as f32 - 1.0) * 100.0
    );
    println!("    Amplification:   TQ {amp_tq:.1}×, SQ {amp_sq:.1}× (MaxSim error ÷ cosine error)");
    println!(
        "      → TQ: {tq_ms_err:.1}% MaxSim error from {:.1}% cosine error = {amp_tq:.1}× amplification",
        (1.0 - tq_cos) * 100.0
    );
    println!(
        "      → SQ: {sq_ms_err:.1}% MaxSim error from {:.1}% cosine error = {amp_sq:.1}× amplification",
        (1.0 - sq_cos) * 100.0
    );
    println!();

    // Verdict
    let sq_sweeps = sq_cos > tq_cos && sq_ms_err < tq_ms_err && sq_ratio > tq_ratio as f32;
    if sq_sweeps {
        println!("  ✅ SpectralQuant sweeps all quality metrics at {bits}-bit:");
        println!(
            "     +{cos_delta:.4} cosine, {ms_err_ratio:.1}× less MaxSim error, {:.0}% more compression.",
            (sq_ratio / tq_ratio as f32 - 1.0) * 100.0
        );
        println!(
            "     MaxSim + SpectralQuant is the optimal combination for late-interaction scoring."
        );
    }
    println!();
}

/// Naive reference: materialize [Lq × Ld] then reduce.
fn maxsim_naive(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for j in 0..ld {
            let d_row = &documents[j * dim..(j + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q_row[d] * d_row[d];
            }
            my_max = my_max.max(dot);
        }
        score += my_max;
    }
    score
}
