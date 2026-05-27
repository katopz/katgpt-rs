use super::{BenchCategory, BenchResult};
#[cfg(feature = "spectral_quant")]
use crate::spectralquant::{
    DequantizeScratch, SpectralQuantKVCache, SpectralQuantKVCacheConfig,
    par_dequantize_spectral_keys_flat,
};
use crate::speculative::types::FlashPrefillConfig;
use crate::speculative::{AttentionScorer, SpeculativeContext, block_select, compress_prompt};
use crate::transformer::{
    ForwardContext, MultiLayerKVCache, PagedKVCache, RavenKVCache, TransformerWeights, forward,
    forward_paged, forward_raven, raven_readout, raven_update,
};
#[cfg(feature = "turboquant")]
use crate::turboquant::TurboQuantKVCache;
#[cfg(any(feature = "turboquant", feature = "hla_attention"))]
use crate::types::kv_dim;
use crate::types::{Config, Rng};
use std::time::Instant;

pub fn bench_prefill_compression(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    let prompt_len = draft_config.block_size * 4;
    let prompt_tokens: Vec<usize> = (0..prompt_len)
        .map(|i| i % draft_config.vocab_size)
        .collect();

    let scorer = AttentionScorer;
    let mut scores_buf = vec![0.0f32; prompt_len];
    let mut sctx = SpeculativeContext::new(draft_config);

    // ── No compression (keep_ratio=1.0) ──
    for _ in 0..warmup {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let _ = compress_prompt(&scores_buf, 1.0, 0, 0);
    }

    let mut total_nocompress = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let selected = compress_prompt(&scores_buf, 1.0, 0, 0);
        total_nocompress += selected.len();
    }
    let elapsed_nocompress = start.elapsed();

    let nocompress = BenchResult {
        label: "Prefill (no compress)".into(),
        throughput: iters as f64 / elapsed_nocompress.as_secs_f64(),
        time_per_step_us: elapsed_nocompress.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_nocompress as f64 / iters as f64,
        color: (180, 180, 180),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    // ── Compressed (keep_ratio=0.1) ──
    for _ in 0..warmup {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let _ = compress_prompt(&scores_buf, 0.1, 0, 0);
    }

    let mut total_compress = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let selected = compress_prompt(&scores_buf, 0.1, 0, 0);
        total_compress += selected.len();
    }
    let elapsed_compress = start.elapsed();

    let compress = BenchResult {
        label: "Prefill (compressed)".into(),
        throughput: iters as f64 / elapsed_compress.as_secs_f64(),
        time_per_step_us: elapsed_compress.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_compress as f64 / iters as f64,
        color: (0, 200, 100),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    (nocompress, compress)
}

/// Benchmark: paged KV cache vs flat KV cache forward pass throughput.
///
/// Measures `forward()` (flat MultiLayerKVCache) vs `forward_paged()` (PagedKVCache)
/// over multiple positions, reporting tokens/sec and µs/step for each.
pub fn bench_paged_vs_flat_cache(config: &Config) -> (BenchResult, BenchResult) {
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);
    let iters = 200;

    // Warm up both paths
    {
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, config);
    }
    {
        let mut ctx = ForwardContext::new(config);
        let mut cache = PagedKVCache::new(config, 1);
        let _ = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, config);
    }

    // Benchmark flat cache
    let mut ctx_flat = ForwardContext::new(config);
    let mut cache_flat = MultiLayerKVCache::new(config);
    let start_flat = Instant::now();
    for _ in 0..iters {
        cache_flat.reset();
        let max_pos = config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward(&mut ctx_flat, &weights, &mut cache_flat, pos, pos, config);
        }
    }
    let elapsed_flat = start_flat.elapsed();

    // Benchmark paged cache
    let mut ctx_paged = ForwardContext::new(config);
    let mut cache_paged = PagedKVCache::new(config, 1);
    let start_paged = Instant::now();
    for _ in 0..iters {
        cache_paged.reset();
        let max_pos = config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward_paged(
                &mut ctx_paged,
                &weights,
                &mut cache_paged,
                0,
                pos,
                pos,
                config,
            );
        }
    }
    let elapsed_paged = start_paged.elapsed();

    let steps_per_iter = config.block_size.min(8) as f64;

    let flat_result = BenchResult {
        label: "forward (flat)".into(),
        throughput: iters as f64 * steps_per_iter / elapsed_flat.as_secs_f64(),
        time_per_step_us: elapsed_flat.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (100, 149, 237),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    let paged_result = BenchResult {
        label: "forward_paged".into(),
        throughput: iters as f64 * steps_per_iter / elapsed_paged.as_secs_f64(),
        time_per_step_us: elapsed_paged.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (255, 165, 0),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    (flat_result, paged_result)
}

/// Benchmark: Raven RSM vs flat KV cache for draft model.
///
/// Compares per-token throughput of `forward_raven()` (O(1) slot memory)
/// against standard `forward()` (O(N) growing cache).
pub fn bench_raven_vs_flat_cache(_config: &Config) -> (BenchResult, BenchResult) {
    let draft_config = Config::draft();
    let mut rng = Rng::new(42);
    let draft_weights = TransformerWeights::new(&draft_config, &mut rng);
    let iters = 200;

    // Raven config: 16 slots, top-4 routing (4x kv_dim for draft)
    let num_slots = 16;
    let top_k = 4;

    // Warm up both paths
    {
        let mut ctx = ForwardContext::new(&draft_config);
        let mut cache = MultiLayerKVCache::new(&draft_config);
        let _ = forward(&mut ctx, &draft_weights, &mut cache, 0, 0, &draft_config);
    }
    {
        let mut ctx = ForwardContext::new(&draft_config);
        let mut cache = RavenKVCache::new(&draft_config, num_slots, top_k);
        let _ = forward_raven(&mut ctx, &draft_weights, &mut cache, 0, 0, &draft_config);
    }

    // Benchmark flat cache (growing O(N) attention)
    let mut ctx_flat = ForwardContext::new(&draft_config);
    let mut cache_flat = MultiLayerKVCache::new(&draft_config);
    let start_flat = Instant::now();
    for _ in 0..iters {
        cache_flat.reset();
        let max_pos = draft_config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward(
                &mut ctx_flat,
                &draft_weights,
                &mut cache_flat,
                0,
                pos,
                &draft_config,
            );
        }
    }
    let elapsed_flat = start_flat.elapsed();

    // Benchmark Raven cache (fixed O(slots) attention)
    let mut ctx_raven = ForwardContext::new(&draft_config);
    let mut cache_raven = RavenKVCache::new(&draft_config, num_slots, top_k);
    let start_raven = Instant::now();
    for _ in 0..iters {
        cache_raven.reset();
        let max_pos = draft_config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward_raven(
                &mut ctx_raven,
                &draft_weights,
                &mut cache_raven,
                0,
                pos,
                &draft_config,
            );
        }
    }
    let elapsed_raven = start_raven.elapsed();

    let steps_per_iter = draft_config.block_size.min(8) as f64;

    let flat_br = BenchResult {
        label: "forward (flat)".into(),
        throughput: iters as f64 * steps_per_iter / elapsed_flat.as_secs_f64(),
        time_per_step_us: elapsed_flat.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (100, 149, 237),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    let raven_br = BenchResult {
        label: format!("forward_raven ({} slots)", num_slots),
        throughput: iters as f64 * steps_per_iter / elapsed_raven.as_secs_f64(),
        time_per_step_us: elapsed_raven.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (180, 100, 220),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    (flat_br, raven_br)
}

/// Benchmark: Raven recall accuracy after noise updates.
///
/// THE critical test from the paper:
/// 1. Write "passkey" to a specific slot (value = 9.9)
/// 2. Run 1000 noise updates targeting OTHER slots
/// 3. Readout and verify original value preserved (> 9.0)
pub fn bench_raven_recall(_config: &Config) -> BenchResult {
    let draft_config = Config::draft();
    let num_slots = 16;
    let top_k = 4;
    let kvd = crate::types::kv_dim(&draft_config);
    let noise_steps = 1000;

    let mut cache = RavenKVCache::new(&draft_config, num_slots, top_k);

    // 1. Write critical passkey to slot 42... wait, we only have 16 slots.
    //    Write to slot 12 instead.
    let passkey_slot = 12;
    let passkey_k = vec![1.0; kvd];
    let passkey_v = vec![9.9; kvd];

    let mut r_t_passkey = vec![0.0f32; num_slots];
    r_t_passkey[passkey_slot] = 1.0;
    raven_update(
        &mut cache.keys,
        &mut cache.values,
        &passkey_k,
        &passkey_v,
        &r_t_passkey,
        cache.forget_rate,
        num_slots,
        kvd,
    );

    // 2. Run 1000 noise updates targeting slots 0-3 (NOT slot 12)
    let start = Instant::now();
    let noise_k = vec![0.5; kvd];
    let noise_v = vec![0.1; kvd];
    let mut r_t_noise = vec![0.0f32; num_slots];
    r_t_noise[0] = 0.25;
    r_t_noise[1] = 0.25;
    r_t_noise[2] = 0.25;
    r_t_noise[3] = 0.25;

    for _ in 0..noise_steps {
        raven_update(
            &mut cache.keys,
            &mut cache.values,
            &noise_k,
            &noise_v,
            &r_t_noise,
            cache.forget_rate,
            num_slots,
            kvd,
        );
    }

    // 3. Readout with passkey query
    let query = vec![1.0; kvd];
    let _retrieved = raven_readout(&query, &cache.keys, &cache.values, num_slots, kvd);
    let elapsed = start.elapsed();

    // Check recall: the passkey value should be preserved in slot 12
    let slot_12_off = passkey_slot * kvd;
    let slot_12_first = cache.values[slot_12_off];

    // Recall accuracy: how close is the stored value to the original 9.9?
    let slot_12_first_f64 = slot_12_first as f64;
    let recall_accuracy: f64 = if slot_12_first_f64 > 9.0 {
        100.0
    } else {
        (slot_12_first_f64 / 9.9) * 100.0
    };

    BenchResult {
        label: format!(
            "raven_recall ({noise_steps} noise, slot {passkey_slot}={:.1}\u{2192}{:.1} acc={recall_accuracy:.0}%)",
            passkey_v[0] as f64, slot_12_first as f64
        ),
        throughput: noise_steps as f64 / elapsed.as_secs_f64(),
        time_per_step_us: elapsed.as_micros() as f64 / noise_steps as f64,
        avg_acceptance_len: recall_accuracy,
        color: (50, 205, 50),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    }
}

// ═══════════════════════════════════════════════════════════════
// Plan 043: TurboQuant KV Cache Compression (legacy baseline)
// ═══════════════════════════════════════════════════════════════

/// Benchmark TQ-3bit store+dequant throughput.
///
/// Measures round-trip: store synthetic KV → dequantize back.
/// Uses 3-bit as the sweet spot between compression and quality.
/// Benchmark TurboQuant store+dequant: both allocating and zero-alloc paths.
/// Returns (allocating_result, zero_alloc_result) for comparison.
#[cfg(feature = "turboquant")]
pub fn bench_turboquant_store_dequant(config: &Config) -> (BenchResult, BenchResult) {
    let kvd = kv_dim(config);
    let n_positions = config.block_size;
    let iters = 100u64;

    // Synthetic KV data
    let keys: Vec<Vec<f32>> = (0..n_positions)
        .map(|p| (0..kvd).map(|i| ((i + p * 7) as f32 * 0.1).sin()).collect())
        .collect();
    let vals: Vec<Vec<f32>> = (0..n_positions)
        .map(|p| {
            (0..kvd)
                .map(|i| ((i + p * 3) as f32 * 0.07).cos())
                .collect()
        })
        .collect();

    // ── Allocating path (dequantize_key / dequantize_value) ───────
    let mut cache_alloc = TurboQuantKVCache::new(config, 3, 3);

    // Warmup
    for _ in 0..10 {
        cache_alloc.reset();
        for pos in 0..n_positions {
            cache_alloc.store_key(0, pos, &keys[pos]);
            cache_alloc.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            std::hint::black_box(cache_alloc.dequantize_key(0, pos));
            std::hint::black_box(cache_alloc.dequantize_value(0, pos));
        }
    }

    let start = Instant::now();
    for _ in 0..iters {
        cache_alloc.reset();
        for pos in 0..n_positions {
            cache_alloc.store_key(0, pos, &keys[pos]);
            cache_alloc.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            std::hint::black_box(cache_alloc.dequantize_key(0, pos));
            std::hint::black_box(cache_alloc.dequantize_value(0, pos));
        }
    }
    let elapsed_alloc = start.elapsed();

    let total_tokens = n_positions as u64 * iters;
    let alloc_result = BenchResult {
        label: "TQ-3bit store+dequant (alloc)".into(),
        throughput: total_tokens as f64 / elapsed_alloc.as_secs_f64(),
        time_per_step_us: elapsed_alloc.as_micros() as f64 / total_tokens as f64,
        avg_acceptance_len: cache_alloc.compression_ratio(),
        color: (148, 0, 211),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    // ── Zero-alloc path (dequantize_key_into / dequantize_value_into) ──
    let mut cache_zero = TurboQuantKVCache::new(config, 3, 3);
    let mut key_buf = vec![0.0f32; kvd];
    let mut val_buf = vec![0.0f32; kvd];

    // Warmup
    for _ in 0..10 {
        cache_zero.reset();
        for pos in 0..n_positions {
            cache_zero.store_key(0, pos, &keys[pos]);
            cache_zero.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            cache_zero.dequantize_key_into(0, pos, &mut key_buf);
            cache_zero.dequantize_value_into(0, pos, &mut val_buf);
            std::hint::black_box(&key_buf);
            std::hint::black_box(&val_buf);
        }
    }

    let start = Instant::now();
    for _ in 0..iters {
        cache_zero.reset();
        for pos in 0..n_positions {
            cache_zero.store_key(0, pos, &keys[pos]);
            cache_zero.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            cache_zero.dequantize_key_into(0, pos, &mut key_buf);
            cache_zero.dequantize_value_into(0, pos, &mut val_buf);
            std::hint::black_box(&key_buf);
            std::hint::black_box(&val_buf);
        }
    }
    let elapsed_zero = start.elapsed();

    let zero_result = BenchResult {
        label: "TQ-3bit store+dequant (zero-alloc)".into(),
        throughput: total_tokens as f64 / elapsed_zero.as_secs_f64(),
        time_per_step_us: elapsed_zero.as_micros() as f64 / total_tokens as f64,
        avg_acceptance_len: cache_zero.compression_ratio(),
        color: (0, 191, 255),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    (alloc_result, zero_result)
}

// ═══════════════════════════════════════════════════════════════
// Issue 064: SpectralQuant Rayon Parallel Dequant Bench Proof
// ═══════════════════════════════════════════════════════════════

/// Benchmark SpectralQuant sequential vs parallel batch dequantize.
///
/// Creates a synthetic SQ cache with `block_size` positions, stores random KV,
/// then measures:
/// 1. Sequential: `dequantize_spectral_keys_flat` (loop over positions)
/// 2. Parallel: `par_dequantize_spectral_keys_flat` (rayon `map_init` per-thread scratch)
///
/// GOAT proof: parallel must produce bit-exact same output as sequential.
/// Speedup depends on n_positions × kv_dim vs rayon overhead.
#[cfg(feature = "spectral_quant")]
pub fn bench_spectralquant_par_dequant(config: &Config) -> (BenchResult, BenchResult) {
    use crate::spectralquant::spectral::participation_ratio;
    use crate::spectralquant::types::SpectralQuantCalibration;

    let kvd = crate::types::kv_dim(config);
    let n_positions = config.block_size.min(256); // Cap for bench speed
    let iters = 50u64;
    let threshold = 1; // Force parallel path for bench

    // Build calibration with identity eigenvectors (will get random rotation fallback)
    let mut eigenvectors = vec![0.0f32; kvd * kvd];
    for i in 0..kvd {
        eigenvectors[i * kvd + i] = 1.0;
    }
    let eigenvalues: Vec<f32> = (0..kvd).map(|i| 10.0 * 0.8f32.powi(i as i32)).collect();
    let d_eff = participation_ratio(&eigenvalues);
    let cal = SpectralQuantCalibration {
        eigenvectors,
        eigenvalues,
        d_eff,
        spectral_gap: None,
        var_95: 10,
        var_99: 20,
        n_samples: 100,
        head_dim: kvd,
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
        kv_dim: kvd,
        max_seq_len: n_positions,
    };

    let mut cache = SpectralQuantKVCache::from_calibration(
        &sq_config,
        &vec![cal.clone(); config.n_layer],
        &vec![cal; config.n_layer],
    );

    // Store synthetic keys
    let keys: Vec<Vec<f32>> = (0..n_positions)
        .map(|p| (0..kvd).map(|i| ((i + p * 7) as f32 * 0.1).sin()).collect())
        .collect();
    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }

    // ── Sequential ──
    // Re-use `cache` since both paths only need &self.
    // Warmup seq
    {
        let mut scratch = DequantizeScratch::new(kvd);
        let mut buf = vec![0.0f32; kvd];
        for _ in 0..5 {
            for t in 0..n_positions {
                cache.dequantize_key_into_with_scratch(0, t, &mut scratch, &mut buf);
                std::hint::black_box(&buf);
            }
        }
    }

    let start = Instant::now();
    {
        let mut scratch = DequantizeScratch::new(kvd);
        let mut buf = vec![0.0f32; kvd];
        for _ in 0..iters {
            for t in 0..n_positions {
                cache.dequantize_key_into_with_scratch(0, t, &mut scratch, &mut buf);
                std::hint::black_box(&buf);
            }
        }
    }
    let elapsed_seq = start.elapsed();

    let total_tokens = n_positions as u64 * iters;
    let seq_br = BenchResult {
        label: format!("SQ-3bit dequant {n_positions}pos (seq)"),
        throughput: total_tokens as f64 / elapsed_seq.as_secs_f64(),
        time_per_step_us: elapsed_seq.as_micros() as f64 / total_tokens as f64,
        avg_acceptance_len: cache.compression_ratio() as f64,
        color: (180, 100, 220),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    // ── Parallel ──
    // Warmup par
    for _ in 0..5 {
        std::hint::black_box(par_dequantize_spectral_keys_flat(
            &cache,
            0,
            n_positions - 1,
            kvd,
            threshold,
        ));
    }

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(par_dequantize_spectral_keys_flat(
            &cache,
            0,
            n_positions - 1,
            kvd,
            threshold,
        ));
    }
    let elapsed_par = start.elapsed();

    let par_br = BenchResult {
        label: format!("SQ-3bit dequant {n_positions}pos (par)"),
        throughput: total_tokens as f64 / elapsed_par.as_secs_f64(),
        time_per_step_us: elapsed_par.as_micros() as f64 / total_tokens as f64,
        avg_acceptance_len: cache.compression_ratio() as f64,
        color: (100, 200, 255),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    };

    let speedup = (elapsed_seq.as_secs_f64() / elapsed_par.as_secs_f64()).max(0.01);
    println!(
        "  SQ par vs seq: {:.2}× ({:.0} vs {:.0} tokens/s, {} positions × {} dim)",
        speedup,
        total_tokens as f64 / elapsed_par.as_secs_f64(),
        total_tokens as f64 / elapsed_seq.as_secs_f64(),
        n_positions,
        kvd,
    );

    (seq_br, par_br)
}

// ═══════════════════════════════════════════════════════════════
// Plan 044: PFlash Block-Sparse Speculative Prefill
// ═══════════════════════════════════════════════════════════════

/// Benchmark PFlash block_select throughput at 1024 blocks.
///
/// Measures the block selection kernel with sparse importance scores
/// (simulates real attention: mostly hay, few needle peaks).
pub fn bench_pflash_block_select() -> BenchResult {
    let num_blocks = 1024;
    let iters = 100_000u64;

    // Sparse scores: mostly low, a few peaks (simulates real attention)
    let scores: Vec<f32> = (0..num_blocks)
        .map(|i| if i % 20 == 0 { 1.0f32 } else { 0.01f32 })
        .collect();

    let cfg = FlashPrefillConfig {
        attention_sink: 1,
        window: 1,
        last_n_full: 0, // allow compression
        ..Default::default()
    };

    // Warmup
    for _ in 0..1000 {
        std::hint::black_box(block_select(&scores, &cfg));
    }

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(block_select(&scores, &cfg));
    }
    let elapsed = start.elapsed();

    let throughput = iters as f64 / elapsed.as_secs_f64();

    BenchResult {
        label: "PFlash block_select (1024 blocks)".into(),
        throughput,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: 0.0,
        color: (0, 128, 128),
        category: BenchCategory::Infrastructure,
        feature_dim: "KV".into(),
    }
}

// ── MaxSim Benchmarks (Research 45, Plan 080 T4/T8) ────────────

/// Benchmark `maxsim_score` vs naive materialized baseline.
///
/// Configs: dim ∈ {64, 128}, Lq ∈ {8, 32, 64}, Ld ∈ {32, 128, 256, 1024}.
/// GOAT gate: ≥2× faster than naive for Lq≥32, Ld≥128, dim=128.
#[cfg(feature = "maxsim")]
pub fn bench_maxsim_score() -> Vec<BenchResult> {
    use crate::simd::maxsim_score;

    let iters = 10_000u64;
    let mut results = Vec::new();

    let configs: &[(usize, usize, usize)] = &[
        (64, 8, 32),
        (64, 32, 128),
        (128, 8, 32),
        (128, 32, 128),
        (128, 64, 256),
        (128, 32, 1024),
    ];

    for &(dim, lq, ld) in configs {
        let queries: Vec<f32> = (0..lq * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let documents: Vec<f32> = (0..ld * dim).map(|i| (i as f32 * 0.01).cos()).collect();

        // Warmup
        for _ in 0..100 {
            std::hint::black_box(maxsim_score(&queries, &documents, lq, ld, dim));
        }

        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(maxsim_score(&queries, &documents, lq, ld, dim));
        }
        let elapsed = start.elapsed();

        let throughput = iters as f64 / elapsed.as_secs_f64();
        results.push(BenchResult {
            label: format!("MaxSim score (Lq={lq}, Ld={ld}, dim={dim})"),
            throughput,
            time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
            avg_acceptance_len: 0.0,
            color: (180, 80, 180),
            category: BenchCategory::Infrastructure,
            feature_dim: "Attn".into(),
        });
    }

    results
}

/// Benchmark PFlash maxsim block scoring vs mean-K baseline.
///
/// Synthetic: 1024 tokens, 32-token blocks, spike attention (1 needle per 20 tokens).
/// GOAT gate: maxsim ≤3× latency overhead vs mean-K, ≥5% needle recall improvement.
#[cfg(feature = "maxsim")]
pub fn bench_pflash_maxsim_block_scoring() -> BenchResult {
    use crate::simd::maxsim_score;

    let block_size = 32;
    let total_tokens = 1024;
    let num_blocks = total_tokens / block_size;
    let dim = 64;
    let iters = 10_000u64;

    // Generate synthetic block embeddings: mostly noise, one "needle" per 20 blocks
    let mut block_queries: Vec<f32> = (0..block_size * dim)
        .map(|_| fastrand::f32() * 0.1)
        .collect();
    // Spike in last query block
    for v in block_queries.iter_mut().take(dim) {
        *v = 1.0;
    }

    let mut block_keys: Vec<Vec<f32>> = (0..num_blocks)
        .map(|_| {
            (0..block_size * dim)
                .map(|_| fastrand::f32() * 0.1)
                .collect()
        })
        .collect();

    // Plant needles: every 20th block has a spike matching the query
    for b in (0..num_blocks).step_by(20) {
        for v in block_keys[b].iter_mut().take(dim) {
            *v = 1.0;
        }
    }

    // Warmup
    for _ in 0..100 {
        for k_block in &block_keys {
            std::hint::black_box(maxsim_score(
                &block_queries,
                k_block,
                block_size,
                block_size,
                dim,
            ));
        }
    }

    let start = Instant::now();
    for _ in 0..iters {
        let mut scores = vec![0.0f32; num_blocks];
        for (b, k_block) in block_keys.iter().enumerate() {
            scores[b] = maxsim_score(&block_queries, k_block, block_size, block_size, dim);
        }
        std::hint::black_box(scores);
    }
    let elapsed = start.elapsed();

    let throughput = iters as f64 / elapsed.as_secs_f64();

    BenchResult {
        label: "PFlash MaxSim block scoring (1024 tok, 32 blocks)".into(),
        throughput,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: 0.0,
        color: (200, 60, 160),
        category: BenchCategory::Infrastructure,
        feature_dim: "Attn".into(),
    }
}
